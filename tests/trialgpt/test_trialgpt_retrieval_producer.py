from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import taim.pipelines.trialgpt.retrieval_producer as producer
from taim import release_cli
from taim.data import PREPARED_MANIFEST_FILENAME, load_prepared_benchmark
from taim.data.complete_eligibility import prepare_profile_complete_eligibility_snapshot
from taim.data.trec_ct_2021 import prepare_trec_ct_2021
from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.generation import GenerationCache, ProviderResponse
from taim.patient_to_trial import PatientToTrialRunRequest
from taim.pipelines.trialgpt.paper import TrialGPTTrialView
from taim.pipelines.trialgpt.publication import (
    TrialGPTPublicationError,
    load_frozen_trialgpt_retrieval,
    validate_frozen_retrieval_for_snapshot,
    validate_frozen_retrieval_for_task_input,
)
from taim.pipelines.trialgpt.retrieval_backends import (
    TaimControlledBM25,
    controlled_bm25_provenance,
)
from taim.pipelines.trialgpt.retrieval_producer import (
    TRIALGPT_RETRIEVAL_SOURCE_ARMS,
    TRIALGPT_RETRIEVAL_TARGETS,
    FrozenRetrievalBinding,
    GeneratedArmQuery,
    RawArmRetrieval,
    RetrievalQueryTopic,
    SourceArmCandidate,
    SourceArmRanking,
    WrittenFrozenRetrieval,
    build_source_arm_ranking,
    compress_three_luna_rankings,
    generate_three_luna_queries,
    load_generated_arm_queries,
    resolve_trialgpt_retrieval_target,
    retrieve_three_luna_queries,
    validate_fresh_retrieval_producer_contract,
    write_frozen_three_luna_retrieval,
    write_generated_arm_queries,
)
from taim.pipelines.trialgpt.system import TrialGPTTAIMLunaV1System
from taim.release_profiles import OFFICIAL_FULL_PROFILE, resolve_benchmark_profile
from taim.release_support import FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256
from tests.model_test_support import ScriptedModel
from tests.snapshot_helpers import TREC_FIXED_TIME, complete_trial_xml, write_trec_source_bundle

_L4 = {
    "device": "cuda",
    "torch": "2.13.0",
    "card": "NVIDIA L4",
    "memory_bytes": 23_034_396_672,
    "compute_capability": "8.9",
    "driver_version": "580.82.07",
    "cuda_runtime": "13.0",
    "cudnn": 92000,
    "tf32_matmul": False,
    "torch_arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
}


def _retrieval_implementation() -> dict[str, object]:
    return {
        "bm25": controlled_bm25_provenance(),
        "fusion": {
            "condition_weight": "1 / (zero_based_condition_index + 1)",
            "rrf_k": 20,
            "tie_rule": "score descending, then trial_id ascending",
        },
        "medcpt": {
            "article_model": "ncbi/MedCPT-Article-Encoder",
            "article_revision": "d05a736da4bb84ee4057b7f7999485be6ed85465",
            "article_resolved_revision": "d05a736da4bb84ee4057b7f7999485be6ed85465",
            "index": "exact_numpy_inner_product",
            "normalization": "none",
            "pooling": "last_hidden_state[:, 0, :]",
            "precision": "float32",
            "query_model": "ncbi/MedCPT-Query-Encoder",
            "query_revision": "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc",
            "query_resolved_revision": "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc",
            "device": "cuda",
            "hardware": dict(_L4),
            "trial_embeddings": {
                "input_signature": "sha256:" + "8" * 64,
                "sha256": "sha256:" + "9" * 64,
            },
        },
    }


_PINNED_V11 = producer.FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256
_RELEASE = {
    "package_artifact_id": "sha256:" + "b" * 64,
    "public_tree_id": "sha256:" + "a" * 64,
    "release_manifest_id": "sha256:" + "9" * 64,
}


def _receipt(plans_sha256: str, **overrides: object) -> dict[str, object]:
    receipt: dict[str, object] = {
        "artifact_type": "taim-trialgpt-query-plan-receipt",
        "schema_version": "1.0",
        "protocol_id": producer.TRIALGPT_RETRIEVAL_PROTOCOL_ID,
        "protocol_sha256": _PINNED_V11,
        "system_id": "TrialGPT-TAIM-Luna-v1",
        "run_id": "trialgpt-trec21-three-luna-v2",
        "track": "trec-ct-2021",
        "profile": "trec-ct-2021-external-fidelity-26149",
        "prepared_snapshot_id": "sha256:" + "1" * 64,
        "release": dict(_RELEASE),
        "query_plans_sha256": plans_sha256,
        "logical_calls": 225,
        "generation": {
            "generation_cache_sha256": "sha256:" + "c" * 64,
            "interruptions": [],
        },
        "generation_ordinal": 1,
    }
    receipt.update(overrides)
    return receipt


def _binding(**overrides: object) -> FrozenRetrievalBinding:
    values: dict[str, object] = {
        "run_id": "trialgpt-trec21-three-luna-v2",
        "benchmark_lineage": "trec-ct-2021",
        "prepared_snapshot_id": "sha256:" + "1" * 64,
        "evaluation_package_id": "sha256:" + "2" * 64,
        "task_input_id": "sha256:" + "3" * 64,
        "pool_receipt_id": "sha256:" + "4" * 64,
        "pool_count": 26_149,
        "pool_ids_sha256": "sha256:" + "5" * 64,
        "protocol_approval_id": _PINNED_V11,
        "producer_git_commit": "7" * 40,
        "producer_dependency_lock_sha256": "sha256:" + "8" * 64,
        "release_manifest_id": "sha256:" + "9" * 64,
        "public_tree_id": "sha256:" + "a" * 64,
        "package_artifact_id": "sha256:" + "b" * 64,
        "query_plan_receipt": {},
        "query_plan_receipt_sha256": "sha256:" + "f" * 64,
    }
    return FrozenRetrievalBinding(**{**values, **overrides})  # type: ignore[arg-type]


def _write(
    tmp_path: Path, raw_retrievals: tuple[RawArmRetrieval, ...], **receipt_overrides: object
) -> WrittenFrozenRetrieval:
    """Freeze ``raw_retrievals`` with a receipt that records the plan file the freeze writes."""

    probe = write_frozen_three_luna_retrieval(
        raw_retrievals,
        binding=_binding(),
        retrieval_implementation=_retrieval_implementation(),
        output_directory=tmp_path / "probe",
    )
    plans_sha256 = "sha256:" + hashlib.sha256(probe.query_plans_path.read_bytes()).hexdigest()
    logical_calls = len(raw_retrievals)
    return write_frozen_three_luna_retrieval(
        raw_retrievals,
        binding=_binding(
            query_plan_receipt=_receipt(
                plans_sha256, logical_calls=logical_calls, **receipt_overrides
            )
        ),
        retrieval_implementation=_retrieval_implementation(),
        output_directory=tmp_path / "run",
    )


def test_retrieval_targets_are_the_closed_trialgpt_paper_matrix() -> None:
    assert [
        (target.track, target.profile, target.topic_count) for target in TRIALGPT_RETRIEVAL_TARGETS
    ] == [
        ("trec-ct-2021", "trec-ct-2021-external-fidelity-26149", 75),
        ("trec-ct-2022", "trec-ct-2022-judgment-union", 50),
        (
            "sigir-ct-2016",
            "sigir-ct-2016-description-judgment-union",
            60,
        ),
        ("sigir-ct-2016", "sigir-ct-2016-summary-judgment-union", 60),
        ("trec-ct-2023", "trec-ct-2023-judgment-union", 40),
    ]
    for track in ("trec-ct-2022", "trec-ct-2023"):
        assert (
            resolve_trialgpt_retrieval_target(
                track=track,
                profile=f"{track}-judgment-union",
            ).supplied_pool_membership
            is False
        )
    with pytest.raises(ValueError, match="not admitted"):
        resolve_trialgpt_retrieval_target(
            track="trec-ct-2023",
            profile="confirmation-full",
        )


def test_the_retrieval_protocol_pin_is_v11_and_the_release_binds_it() -> None:
    from taim.pipelines.trialgpt.trialgpt_extension import EXTENSION

    document = Path(__file__).parents[2] / producer.TRIALGPT_RETRIEVAL_PROTOCOL_PATH
    assert "sha256:" + hashlib.sha256(document.read_bytes()).hexdigest() == _PINNED_V11
    assert f"**Protocol ID:** `{producer.TRIALGPT_RETRIEVAL_PROTOCOL_ID}`" in document.read_text(
        encoding="utf-8"
    )
    assert EXTENSION.protocol_bindings == (
        (
            "src/taim/pipelines/trialgpt/retrieval_producer.py",
            "FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256",
            producer.TRIALGPT_RETRIEVAL_PROTOCOL_PATH,
        ),
    )


def test_controlled_bm25_v2_counts_documents_not_repeated_terms() -> None:
    trials = (
        TrialGPTTrialView(
            trial_id="NCT00000001",
            brief_title="Lung cancer study",
            diseases=("Lung cancer",),
            interventions=(),
            brief_summary="",
            inclusion_criteria="",
            exclusion_criteria="",
            retrieval_text="Lung cancer",
        ),
        TrialGPTTrialView(
            trial_id="NCT00000002",
            brief_title="Diabetes study",
            diseases=("Diabetes",),
            interventions=(),
            brief_summary="",
            inclusion_criteria="",
            exclusion_criteria="",
            retrieval_text="Diabetes",
        ),
    )

    backend = TaimControlledBM25(trials)

    assert backend.document_frequency["lung"] == 1
    assert backend.rank("lung cancer", 2) == ["NCT00000001", "NCT00000002"]
    assert controlled_bm25_provenance()["implementation"] == "taim-controlled-bm25-v2"


class _QueryProvider(ScriptedModel):
    provider_id = "query-provider-test-v1"
    model_id = "gpt-5.6-luna"

    def __init__(self, reasoning_effort: str) -> None:
        self.reasoning_effort = reasoning_effort

    def provenance(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model": "gpt-5.6-luna",
            "reasoning_effort": self.reasoning_effort,
        }

    def respond(self, request, *, prompt, attempt, logical_call_id) -> ProviderResponse:
        payload = request.input_payload
        output_schema = request.output_schema
        del output_schema
        output = {
            "summary": f"summary {payload['topic_id']} {payload['arm_id']}",
            "conditions": [f"condition {payload['topic_id']} {payload['arm_id']}"],
        }
        return ProviderResponse(__import__("json").dumps(output))


def test_three_luna_query_generation_pins_prompts_models_and_effort(tmp_path) -> None:
    cache = GenerationCache(tmp_path / "generation.sqlite3")
    try:
        queries = generate_three_luna_queries(
            (
                RetrievalQueryTopic("1", "Patient one."),
                RetrievalQueryTopic("2", "Patient two."),
            ),
            models={
                "medium": _QueryProvider("medium"),
                "xhigh": _QueryProvider("xhigh"),
                "recall-explicit": _QueryProvider("medium"),
            },
            max_workers=2,
        )
    finally:
        cache.close()

    assert [(query.topic_id, query.arm_id) for query in queries] == [
        ("1", "medium"),
        ("1", "xhigh"),
        ("1", "recall-explicit"),
        ("2", "medium"),
        ("2", "xhigh"),
        ("2", "recall-explicit"),
    ]
    assert queries[0].prompt_contract_id == "trialgpt-paper-keyword-v1"
    assert queries[2].prompt_contract_id == "trialgpt-recall-explicit-v2"
    assert queries[1].provider_configuration["reasoning_effort"] == "xhigh"
    assert queries[0].generation_trace["status"] == "resolved"

    query_path = tmp_path / "query-plans.jsonl"
    write_generated_arm_queries(queries, query_path)
    assert load_generated_arm_queries(query_path) == queries


def test_generated_queries_feed_both_raw_retrieval_channels() -> None:
    queries = tuple(_raw_arm("1", arm_id).query for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS)
    trial_ids = tuple(f"NCT-{rank:04d}" for rank in range(1, 2_001))

    class Lexical:
        def rank(self, condition, depth):
            assert condition.startswith("condition 1")
            assert depth == 2_000
            return list(trial_ids)

    class Dense:
        def __init__(self) -> None:
            self.model_configuration = {"backend": "test"}

        def rank(self, *, conditions, trials, depth):
            assert len(conditions) == 1
            assert trials == ()
            assert depth == 2_000
            return [list(reversed(trial_ids))]

    retrieved = retrieve_three_luna_queries(
        queries,
        trials=(),
        lexical_backend=Lexical(),
        dense_backend=Dense(),
    )

    assert len(retrieved) == 3
    assert retrieved[0].bm25_rankings[0][0] == "NCT-0001"
    assert retrieved[0].medcpt_rankings[0][0] == "NCT-2000"


def _source_arm(
    arm_id: str,
    leaders: tuple[tuple[str, bool, bool], ...],
) -> SourceArmRanking:
    candidates = [
        SourceArmCandidate(
            trial_id=trial_id,
            rank=rank,
            lexical_exposed=lexical_exposed,
            dense_exposed=dense_exposed,
        )
        for rank, (trial_id, lexical_exposed, dense_exposed) in enumerate(leaders, start=1)
    ]
    candidates.extend(
        SourceArmCandidate(
            trial_id=f"NCT-{arm_id}-{rank:04d}",
            rank=rank,
            lexical_exposed=True,
            dense_exposed=False,
        )
        for rank in range(len(candidates) + 1, 2_001)
    )
    return SourceArmRanking(topic_id="1", arm_id=arm_id, candidates=tuple(candidates))


def _raw_arm(topic_id: str, arm_id: str) -> RawArmRetrieval:
    trial_ids = tuple(f"NCT-{topic_id}-{arm_id}-{rank:04d}" for rank in range(1, 2_001))
    prompt_contract_id = (
        "trialgpt-recall-explicit-v2"
        if arm_id == "recall-explicit"
        else "trialgpt-paper-keyword-v1"
    )
    return RawArmRetrieval(
        query=GeneratedArmQuery(
            topic_id=topic_id,
            arm_id=arm_id,
            summary=f"summary {topic_id} {arm_id}",
            conditions=(f"condition {topic_id} {arm_id}",),
            patient_text_sha256="sha256:" + "d" * 64,
            prompt_contract_id=prompt_contract_id,
            provider_configuration={
                "model": "gpt-5.6-luna",
                "reasoning_effort": "xhigh" if arm_id == "xhigh" else "medium",
            },
            generation_trace={
                "attempts": [{"attempt": 1}],
                "identity_sha256": "sha256:" + "a" * 64,
                "logical_call_id": "sha256:" + "b" * 64,
                "output_sha256": "sha256:" + "c" * 64,
                "provider_id": "query-provider-test-v1",
                "selected_attempt": 1,
                "status": "resolved",
            },
        ),
        bm25_rankings=(trial_ids,),
        medcpt_rankings=(trial_ids,),
    )


def test_source_arm_is_rebuilt_from_raw_condition_rankings() -> None:
    fillers = tuple(f"NCT-fill-{rank:04d}" for rank in range(1, 2_001))
    raw = RawArmRetrieval(
        query=GeneratedArmQuery(
            topic_id="1",
            arm_id="medium",
            summary="summary",
            conditions=("condition",),
            patient_text_sha256="sha256:" + "d" * 64,
            prompt_contract_id="trialgpt-paper-keyword-v1",
            provider_configuration={"model": "gpt-5.6-luna", "reasoning_effort": "medium"},
            generation_trace={
                "attempts": [{"attempt": 1}],
                "identity_sha256": "sha256:" + "a" * 64,
                "logical_call_id": "sha256:" + "b" * 64,
                "output_sha256": "sha256:" + "c" * 64,
                "provider_id": "query-provider-test-v1",
                "selected_attempt": 1,
                "status": "resolved",
            },
        ),
        bm25_rankings=(("NCT-A", "NCT-B", *fillers[:1_998]),),
        medcpt_rankings=(("NCT-A", "NCT-C", *fillers[:1_998]),),
    )

    ranking = build_source_arm_ranking(raw)

    assert ranking.candidates[0] == SourceArmCandidate(
        trial_id="NCT-A",
        rank=1,
        lexical_exposed=True,
        dense_exposed=True,
    )
    lexical_only = next(
        candidate for candidate in ranking.candidates if candidate.trial_id == "NCT-B"
    )
    assert lexical_only.lexical_exposed is True
    assert lexical_only.dense_exposed is False


def test_three_luna_compressor_applies_the_frozen_consensus_features() -> None:
    rankings = (
        _source_arm(
            "medium",
            (
                ("NCT-A", True, True),
                ("NCT-B", True, True),
            ),
        ),
        _source_arm(
            "xhigh",
            (
                ("NCT-A", True, True),
                ("NCT-C", True, False),
            ),
        ),
        _source_arm(
            "recall-explicit",
            (
                ("NCT-D", True, True),
                ("NCT-A", True, True),
            ),
        ),
    )

    result = compress_three_luna_rankings(rankings)

    assert tuple(result) == ("1",)
    assert TRIALGPT_RETRIEVAL_SOURCE_ARMS == ("medium", "xhigh", "recall-explicit")
    assert [candidate.trial_id for candidate in result["1"][:4]] == [
        "NCT-A",
        "NCT-D",
        "NCT-B",
        "NCT-C",
    ]
    assert result["1"][0].score == pytest.approx(0.9607843137254902)
    assert result["1"][0].features.arm_count == 3
    assert result["1"][0].features.best_rank_quality == 1.0
    assert result["1"][0].features.per_arm_reciprocal_rank == (1.0, 1.0, 0.5)
    assert result["1"][0].features.lexical_dense_agreement_count == 3


def test_producer_writes_a_preflight_compatible_retrieval_and_lock(tmp_path) -> None:
    raw_retrievals = tuple(
        _raw_arm(topic_id, arm_id)
        for topic_id in ("1", "2")
        for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
    )
    binding = _binding()

    written = _write(tmp_path, raw_retrievals)

    retrieval = load_frozen_trialgpt_retrieval(
        written.retrieval_path,
        lock_path=written.lock_path,
    )
    validate_frozen_retrieval_for_snapshot(
        retrieval,
        benchmark_lineage="trec-ct-2021",
        retrieval_source_snapshot_id=binding.prepared_snapshot_id,
        topic_ids=("1", "2"),
        trial_ids=tuple(
            sorted(
                {
                    candidate
                    for raw in raw_retrievals
                    for ranking in raw.bm25_rankings
                    for candidate in ranking
                }
            )
        ),
    )
    validate_frozen_retrieval_for_task_input(
        retrieval,
        task_input_id=binding.task_input_id,
    )
    with pytest.raises(TrialGPTPublicationError, match="Task Input does not match"):
        validate_frozen_retrieval_for_task_input(
            retrieval,
            task_input_id="sha256:" + "4" * 64,
        )
    validate_fresh_retrieval_producer_contract(
        retrieval,
        artifact_directory=written.lock_path.parent,
    )

    assert written.source_rankings_path.is_file()
    assert written.raw_channels_path.is_file()
    assert written.query_plans_path.is_file()
    assert written.consensus_features_path.is_file()
    assert retrieval.lock["method"]["producer_contract_id"] == ("trialgpt-three-luna-consensus-v2")
    assert retrieval.lock["source_input_lock"]["task_input_id"] == binding.task_input_id
    assert retrieval.lock["source_input_lock"]["release_identity"] == {
        "package_artifact_id": binding.package_artifact_id,
        "public_tree_id": binding.public_tree_id,
        "release_manifest_id": binding.release_manifest_id,
    }
    assert tuple(retrieval.rankings) == ("1", "2")


def test_fresh_contract_rejects_a_lock_without_the_v2_producer_identity(tmp_path) -> None:
    raw_retrievals = tuple(_raw_arm("1", arm_id) for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS)
    written = write_frozen_three_luna_retrieval(
        raw_retrievals,
        binding=_binding(),
        retrieval_implementation=_retrieval_implementation(),
        output_directory=tmp_path,
    )
    lock = __import__("json").loads(written.lock_path.read_text(encoding="utf-8"))
    del lock["method"]["producer_contract_id"]
    written.lock_path.write_text(__import__("json").dumps(lock), encoding="utf-8")
    retrieval = load_frozen_trialgpt_retrieval(
        written.retrieval_path,
        lock_path=written.lock_path,
    )

    with pytest.raises(ValueError, match="producer method contract"):
        validate_fresh_retrieval_producer_contract(
            retrieval,
            artifact_directory=written.lock_path.parent,
        )


def test_fresh_contract_rejects_a_lock_without_release_identity(tmp_path) -> None:
    raw_retrievals = tuple(_raw_arm("1", arm_id) for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS)
    written = write_frozen_three_luna_retrieval(
        raw_retrievals,
        binding=_binding(),
        retrieval_implementation=_retrieval_implementation(),
        output_directory=tmp_path,
    )
    lock = __import__("json").loads(written.lock_path.read_text(encoding="utf-8"))
    del lock["source_input_lock"]["release_identity"]
    written.lock_path.write_text(__import__("json").dumps(lock), encoding="utf-8")
    retrieval = load_frozen_trialgpt_retrieval(
        written.retrieval_path,
        lock_path=written.lock_path,
    )

    with pytest.raises(ValueError, match="source input lock is incomplete"):
        validate_fresh_retrieval_producer_contract(
            retrieval,
            artifact_directory=written.lock_path.parent,
        )


def test_fresh_contract_rejects_the_defective_bm25_v1_identity(tmp_path) -> None:
    raw_retrievals = tuple(_raw_arm("1", arm_id) for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS)
    written = _write(tmp_path, raw_retrievals)
    lock = __import__("json").loads(written.lock_path.read_text(encoding="utf-8"))
    lock["retrieval_implementation"]["bm25"]["implementation"] = "taim-controlled-bm25-v1"
    written.lock_path.write_text(__import__("json").dumps(lock), encoding="utf-8")
    retrieval = load_frozen_trialgpt_retrieval(
        written.retrieval_path,
        lock_path=written.lock_path,
    )

    with pytest.raises(ValueError, match="frozen BM25 v2"):
        validate_fresh_retrieval_producer_contract(
            retrieval,
            artifact_directory=written.lock_path.parent,
        )


# Query plans are generated per run and recorded by receipt (protocol v11). The four files protocol
# v6 Section 2 admitted are unavailable; a refusal names them and points to the receipt route.

_FREEZE_SCRIPT = Path(__file__).parents[2] / "scripts" / "freeze_trialgpt_retrieval.py"


def _hand_built_plans(path: Path, topic_count: int) -> Path:
    """A well-formed plan file that no model produced: conditions and traces are made up."""

    efforts = {"medium": "medium", "xhigh": "xhigh", "recall-explicit": "medium"}
    rows = []
    for number in range(1, topic_count + 1):
        text = f"A synthetic patient {number} with type 2 diabetes and hypertension."
        for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS:
            label = f"hand-built:{number}:{arm_id}".encode()
            rows.append(
                {
                    "arm_id": arm_id,
                    "conditions": ["diabetes", "hypertension"],
                    "generation_trace": {
                        "attempts": [{"attempt": 1, "kind": "hand-built"}],
                        "identity_sha256": "sha256:" + hashlib.sha256(label).hexdigest(),
                        "logical_call_id": "sha256:" + hashlib.sha256(label).hexdigest(),
                        "output_sha256": "sha256:" + hashlib.sha256(b"o" + label).hexdigest(),
                        "provider_id": "not-a-provider",
                        "selected_attempt": 1,
                        "status": "resolved",
                    },
                    "patient_text_sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
                    "prompt_contract_id": (
                        "trialgpt-recall-explicit-v2"
                        if arm_id == "recall-explicit"
                        else "trialgpt-paper-keyword-v1"
                    ),
                    "provider_configuration": {
                        "model": "gpt-5.6-luna",
                        "reasoning_effort": efforts[arm_id],
                    },
                    "summary": text,
                    "topic_id": str(number),
                }
            )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(
    tmp_path: Path, *extra: str, protocol: str = _PINNED_V11
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the release's freeze script, unmodified, in its own process on a hand-built plan."""

    plans = _hand_built_plans(tmp_path / "own-query-plans.jsonl", 75)
    arguments = [
        str(_FREEZE_SCRIPT),
        "--track", "trec-ct-2021",
        "--profile", "trec-ct-2021-external-fidelity-26149",
        "--data-dir", str(tmp_path / "prepared"),
        "--pool-file", str(tmp_path / "pool.txt"),
        "--pool-count", "26149",
        "--pool-ids-sha256", "sha256:" + "5" * 64,
        "--pool-receipt-id", "sha256:" + "4" * 64,
        "--protocol-approval-id", protocol,
        "--release-manifest", str(tmp_path / "manifest.json"),
        "--package-artifact", str(tmp_path / "package.whl"),
        "--clinical-as-of", "2021-04-27T00:00:00Z",
        "--query-plans", str(plans),
        "--run-id", "trialgpt-trec21-three-luna-v2",
        "--workspace", str(tmp_path / "workspace"),
        "--output-dir", str(tmp_path / "frozen"),
        *extra,
    ]  # fmt: skip
    completed = subprocess.run(
        [sys.executable, *arguments], capture_output=True, text=True, check=False
    )
    return completed, plans


def _assert_names_the_route(message: str) -> None:
    for profile, digest in producer.PROTOCOL_V6_ADMITTED_QUERY_PLANS.items():
        assert f"{profile}: {digest}" in message
    assert "These four files are not available." in message
    assert "So no one can produce a TrialGPT-TAIM-Luna-v1 retrieval from them." in message
    assert "--query-plan-receipt" in message
    assert producer.TRIALGPT_RETRIEVAL_PROTOCOL_ID in message
    assert producer.TRIALGPT_RETRIEVAL_PROTOCOL_PATH in message


def test_the_four_digests_are_exactly_protocol_v6_section_2() -> None:
    root = Path(__file__).parents[2]
    protocol = root / "docs" / "paper-analysis-protocol-2026-09-02-v6.md"
    section = protocol.read_text(encoding="utf-8").split("## 2. Query-plan continuity")[1]
    table = section.split("The admitted query-plan files are exactly:")[1].split("\n\n")[1]
    assert re.findall(r"`(sha256:[0-9a-f]{64})`", table) == list(
        producer.PROTOCOL_V6_ADMITTED_QUERY_PLANS.values()
    )
    # v6 admitted plans for the four targets it knew; the TREC 2023 target came later.
    assert list(producer.PROTOCOL_V6_ADMITTED_QUERY_PLANS) == [
        target.profile for target in TRIALGPT_RETRIEVAL_TARGETS if target.track != "trec-ct-2023"
    ]


def test_the_freeze_refuses_plans_without_a_receipt_before_any_retrieval(tmp_path) -> None:
    completed, plans = _freeze(tmp_path)

    assert completed.returncode != 0
    message = completed.stderr
    assert f"query plans {plans} (SHA-256 {_sha256(plans)}) have no query-plan receipt." in message
    _assert_names_the_route(message)
    # Refused before the clean-tree check, the package load and any retrieval: nothing written.
    assert not (tmp_path / "frozen").exists()
    assert not (tmp_path / "workspace").exists()


def test_the_freeze_refuses_plans_their_receipt_does_not_describe(tmp_path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(_receipt("sha256:" + "0" * 64)), encoding="utf-8")

    completed, _ = _freeze(tmp_path, "--query-plan-receipt", str(receipt))

    assert completed.returncode != 0
    assert "the query-plan receipt's query_plans_sha256 is" in completed.stderr
    _assert_names_the_route(completed.stderr)
    assert not (tmp_path / "frozen").exists()


def test_the_freeze_refuses_a_protocol_other_than_v11(tmp_path) -> None:
    v7 = "sha256:a617a46c2d61e76ad25963e709cad2c3ff99d4be466f173b8add02f1754927ef"

    completed, _ = _freeze(tmp_path, protocol=v7)

    assert completed.returncode != 0
    assert "--protocol-approval-id must be protocol v11's digest" in completed.stderr


def test_a_malformed_plan_is_still_refused_by_the_loader(tmp_path) -> None:
    plans = _hand_built_plans(tmp_path / "plans.jsonl", 2)
    rows = [json.loads(line) for line in plans.read_text(encoding="utf-8").splitlines()]
    rows[0]["provider_configuration"]["model"] = "gpt-4o"
    plans.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="provider contract does not match v2"):
        load_generated_arm_queries(plans)


def test_a_receipt_that_describes_the_run_is_accepted(tmp_path) -> None:
    plans = _hand_built_plans(tmp_path / "plans.jsonl", 75)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_receipt(_sha256(plans))), encoding="utf-8")

    receipt, digest = producer.require_query_plan_receipt(
        path,
        target=TRIALGPT_RETRIEVAL_TARGETS[0],
        run_id="trialgpt-trec21-three-luna-v2",
        query_plans=plans,
    )

    assert digest == _sha256(path)
    producer.require_query_plan_receipt_identity(
        receipt, release_identity=_RELEASE, prepared_snapshot_id="sha256:" + "1" * 64
    )
    with pytest.raises(ValueError, match="another release"):
        producer.require_query_plan_receipt_identity(
            receipt,
            release_identity={**_RELEASE, "package_artifact_id": "sha256:" + "0" * 64},
            prepared_snapshot_id="sha256:" + "1" * 64,
        )
    with pytest.raises(ValueError, match="another Prepared Snapshot"):
        producer.require_query_plan_receipt_identity(
            receipt, release_identity=_RELEASE, prepared_snapshot_id="sha256:" + "0" * 64
        )


@pytest.mark.parametrize(
    ("overrides", "defect"),
    [
        ({"run_id": "another-run"}, "run_id"),
        ({"query_plans_sha256": "sha256:" + "0" * 64}, "query_plans_sha256"),
        ({"mode": "per-run"}, "exactly the protocol v11 Section 6 fields"),
        ({"protocol_sha256": "sha256:" + "0" * 64}, "protocol_sha256"),
        ({"profile": "trec-ct-2022-judgment-union"}, "profile"),
        ({"logical_calls": 224}, "logical_calls"),
        ({"generation_ordinal": 0}, "generation_ordinal"),
        ({"generation": {"interruptions": []}}, "generation_cache_sha256"),
        ({"account": "subscription"}, "exactly the protocol v11 Section 6 fields"),
    ],
)
def test_a_receipt_that_does_not_describe_the_run_is_refused(tmp_path, overrides, defect) -> None:
    plans = _hand_built_plans(tmp_path / "plans.jsonl", 75)
    path = tmp_path / "receipt.json"
    receipt = _receipt(_sha256(plans), **overrides)
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match=defect) as refusal:
        producer.require_query_plan_receipt(
            path,
            target=TRIALGPT_RETRIEVAL_TARGETS[0],
            run_id="trialgpt-trec21-three-luna-v2",
            query_plans=plans,
        )
    _assert_names_the_route(str(refusal.value))


def test_a_lock_binds_v11_and_carries_the_receipt_that_describes_it(tmp_path) -> None:
    raw_retrievals = tuple(_raw_arm("1", arm_id) for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS)
    written = _write(tmp_path, raw_retrievals)
    retrieval = load_frozen_trialgpt_retrieval(written.retrieval_path, lock_path=written.lock_path)

    validate_fresh_retrieval_producer_contract(
        retrieval, artifact_directory=written.lock_path.parent
    )
    source = retrieval.lock["source_input_lock"]
    assert source["protocol_approval_id"] == _PINNED_V11
    assert source["query_plan_receipt"]["query_plans_sha256"] == source["query_plans_sha256"]

    tampered = json.loads(written.lock_path.read_text(encoding="utf-8"))
    tampered["source_input_lock"]["query_plan_receipt"]["run_id"] = "another-run"
    with pytest.raises(ValueError, match="run_id"):
        producer.validate_query_plan_receipt_binding(tampered)
    tampered["source_input_lock"]["query_plan_receipt"]["run_id"] = source["run_id"]
    tampered["source_input_lock"]["protocol_approval_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="does not bind protocol v11"):
        producer.validate_query_plan_receipt_binding(tampered)


def test_a_lock_whose_receipt_does_not_record_its_plans_is_refused(tmp_path) -> None:
    raw_retrievals = tuple(_raw_arm("1", arm_id) for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS)
    written = _write(tmp_path, raw_retrievals, query_plans_sha256="sha256:" + "0" * 64)
    retrieval = load_frozen_trialgpt_retrieval(written.retrieval_path, lock_path=written.lock_path)

    with pytest.raises(ValueError, match="query_plans_sha256"):
        validate_fresh_retrieval_producer_contract(
            retrieval, artifact_directory=written.lock_path.parent
        )
    with pytest.raises(ValueError, match="query_plans_sha256"):
        producer.validate_query_plan_receipt_binding(retrieval.lock)


def test_a_lock_records_the_hardware_its_medcpt_embeddings_were_made_on(tmp_path) -> None:
    implementation = _retrieval_implementation()
    medcpt = implementation["medcpt"]
    assert isinstance(medcpt, dict)
    for hardware in (
        None,
        {**_L4, "device": "cpu"},
        {key: value for key, value in _L4.items() if key != "driver_version"},
        {**_L4, "torch_arch_list": []},
    ):
        medcpt["hardware"] = hardware
        with pytest.raises(producer.TrialGPTRetrievalProducerError, match=r"hardware|card"):
            producer._validate_retrieval_implementation(implementation)
    medcpt.update(
        device="cpu",
        hardware={
            "device": "cpu",
            "torch": "2.13.0",
            "machine": "x86_64",
            "processor": "x86_64",
            "operating_system": "Linux",
        },
    )
    producer._validate_retrieval_implementation(implementation)


def test_the_medcpt_hardware_record_reads_the_card_and_its_driver(monkeypatch) -> None:
    import types

    from taim.pipelines.trialgpt import retrieval_backends

    torch = types.SimpleNamespace(
        __version__="2.13.0",
        version=types.SimpleNamespace(cuda="13.0"),
        backends=types.SimpleNamespace(
            cudnn=types.SimpleNamespace(version=lambda: 92000),
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        ),
        cuda=types.SimpleNamespace(
            current_device=lambda: 0,
            get_device_capability=lambda index: (8, 9),
            get_device_name=lambda index: "NVIDIA L4",
            get_device_properties=lambda index: types.SimpleNamespace(total_memory=23_034_396_672),
            get_arch_list=lambda: list(_L4["torch_arch_list"]),
        ),
    )
    monkeypatch.setattr(retrieval_backends, "resolve_executable", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        retrieval_backends.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "580.82.07\n", ""),
    )
    record = retrieval_backends.medcpt_hardware(
        torch, types.SimpleNamespace(type="cuda", index=None)
    )
    assert record == _L4
    cpu = retrieval_backends.medcpt_hardware(torch, types.SimpleNamespace(type="cpu", index=None))
    assert cpu["device"] == "cpu" and cpu["torch"] == "2.13.0" and cpu["machine"]


# Both Systems read the trials' complete eligibility text, and a target's one generation, receipt
# and freeze bind its Prepared Snapshot. A stratum package carries an excerpt of long eligibility
# text, so every step refuses it before anything is spent.

_GENERATE_SCRIPT = Path(__file__).parents[2] / "scripts" / "generate_trialgpt_retrieval_queries.py"
_MISSING = "complete_eligibility_text"


def _packages(tmp_path: Path) -> tuple[Path, Path]:
    """A stratum package at ``prepared``, where the freeze looks, and its complete twin."""

    source, lock = write_trec_source_bundle(
        tmp_path / "bundle", trials={"NCT00000001": complete_trial_xml()}
    )
    stratum = prepare_trec_ct_2021(
        source, tmp_path / "prepared", source_lock_path=lock, created_at=TREC_FIXED_TIME
    ).output_directory
    complete = prepare_profile_complete_eligibility_snapshot(
        stratum, source, tmp_path / "complete", profile_id=OFFICIAL_FULL_PROFILE
    ).output_directory
    return stratum, complete


def test_the_system_binding_accepts_only_the_complete_eligibility_package(tmp_path) -> None:
    stratum, complete = _packages(tmp_path)

    def request(directory: Path) -> PatientToTrialRunRequest:
        return PatientToTrialRunRequest.for_benchmark(
            resolve_benchmark_profile(load_prepared_benchmark(directory)),
            run_id="capabilities",
            top_k=1,
            metric_cutoff=1,
            options={},
            clinical_as_of=TREC_FIXED_TIME,
            patient_evidence_profile=source_grounded_patient_evidence_profile(),
            required_capabilities=TrialGPTTAIMLunaV1System.required_capabilities,
            optional_capabilities=TrialGPTTAIMLunaV1System.optional_capabilities,
        )

    with pytest.raises(ValueError, match=f"missing required System capabilities: {_MISSING}"):
        request(stratum)
    accepted = request(complete)
    assert accepted.required_capabilities <= accepted.snapshot.available_capabilities
    producer.require_trialgpt_capabilities(complete / PREPARED_MANIFEST_FILENAME)
    with pytest.raises(ValueError, match=r"prepare_trialgpt_profile_snapshot\.py") as refusal:
        producer.require_trialgpt_capabilities(stratum / PREPARED_MANIFEST_FILENAME)
    assert f"TrialGPT-TAIM-Luna-v1 requires: {_MISSING}" in str(refusal.value)


def test_generation_refuses_a_stratum_package_before_any_provider_call(tmp_path) -> None:
    stratum, _ = _packages(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(_GENERATE_SCRIPT),
            "--track",
            "trec-ct-2021",
            "--profile",
            "trec-ct-2021-external-fidelity-26149",
            "--prepared-manifest",
            str(stratum / PREPARED_MANIFEST_FILENAME),
            "--run-id",
            "trialgpt-trec21-three-luna-v2",
            "--cache",
            str(tmp_path / "cache" / "generation.sqlite3"),
            "--output",
            str(tmp_path / "plans.jsonl"),
            "--codex-executable",
            str(tmp_path / "no-codex"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert f"TrialGPT-TAIM-Luna-v1 requires: {_MISSING}" in completed.stderr
    assert not (tmp_path / "plans.jsonl").exists()
    assert not (tmp_path / "cache").exists()


def test_the_freeze_refuses_a_stratum_package_before_it_loads_it(tmp_path) -> None:
    _packages(tmp_path)
    plans = _hand_built_plans(tmp_path / "own-query-plans.jsonl", 75)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(_receipt(_sha256(plans))), encoding="utf-8")

    completed, _ = _freeze(tmp_path, "--query-plan-receipt", str(receipt))

    assert completed.returncode != 0
    assert f"TrialGPT-TAIM-Luna-v1 requires: {_MISSING}" in completed.stderr
    assert not (tmp_path / "frozen").exists()
    assert not (tmp_path / "workspace").exists()


def test_preflight_refuses_a_stratum_package_before_it_verifies_the_release(
    tmp_path, capsys
) -> None:
    stratum, _ = _packages(tmp_path)

    with pytest.raises(SystemExit):
        release_cli.main(
            [
                "patient-to-trial",
                "trialgpt",
                "preflight",
                "--data-dir",
                str(stratum),
                "--clinical-as-of",
                "2021-04-27T00:00:00Z",
                "--pool-file",
                str(tmp_path / "pool.json"),
                "--pool-count",
                "26149",
                "--pool-ids-sha256",
                "sha256:" + "1" * 64,
                "--pool-receipt-id",
                "sha256:" + "2" * 64,
                "--protocol-approval-id",
                FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256,
                "--frozen-retrieval",
                str(tmp_path / "retrieval.jsonl"),
                "--frozen-retrieval-lock",
                str(tmp_path / "retrieval-lock.json"),
                "--release-manifest",
                str(tmp_path / "missing-manifest.json"),
                "--package-artifact",
                str(tmp_path / "missing.whl"),
                "--dependency-lock",
                str(tmp_path / "missing-uv.lock"),
                "--output",
                str(tmp_path / "contract.json"),
            ]
        )

    assert f"TrialGPT-TAIM-Luna-v1 requires: {_MISSING}" in capsys.readouterr().err
    assert not (tmp_path / "contract.json").exists()
