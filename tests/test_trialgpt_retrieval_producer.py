from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from taim.adapters.trialgpt import TrialGPTGeneration
from taim.trialgpt_generation import GenerationCache
from taim.trialgpt_paper import TrialGPTTrialView
from taim.trialgpt_publication import (
    TrialGPTPublicationError,
    load_frozen_trialgpt_retrieval,
    validate_frozen_retrieval_for_snapshot,
    validate_frozen_retrieval_for_task_input,
)
from taim.trialgpt_retrieval_backends import TaimControlledBM25, controlled_bm25_provenance
from taim.trialgpt_retrieval_producer import (
    FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
    TRIALGPT_RETRIEVAL_SOURCE_ARMS,
    TRIALGPT_RETRIEVAL_TARGETS,
    FrozenRetrievalBinding,
    GeneratedArmQuery,
    RawArmRetrieval,
    RetrievalQueryTopic,
    SourceArmCandidate,
    SourceArmRanking,
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
            "trial_embeddings": {
                "input_signature": "sha256:" + "8" * 64,
                "sha256": "sha256:" + "9" * 64,
            },
        },
    }


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
    ]
    assert (
        resolve_trialgpt_retrieval_target(
            track="trec-ct-2022",
            profile="trec-ct-2022-judgment-union",
        ).supplied_pool_membership
        is False
    )
    with pytest.raises(ValueError, match="not admitted"):
        resolve_trialgpt_retrieval_target(
            track="trec-ct-2023",
            profile="trec-ct-2023-judgment-union",
        )


def test_frozen_trialgpt_retrieval_protocol_bytes_match_the_code_pin() -> None:
    root = Path(__file__).parents[1]
    candidates = (
        root / "docs" / "protocols" / "paper-analysis-protocol-2026-09-02-v7.md",
        root / "docs" / "paper-analysis-protocol-2026-09-02-v7.md",
    )
    protocol = next((path for path in candidates if path.is_file()), None)
    assert protocol is not None
    assert "sha256:" + hashlib.sha256(protocol.read_bytes()).hexdigest() == (
        FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256
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


class _QueryProvider:
    provider_id = "query-provider-test-v1"

    def __init__(self, reasoning_effort: str) -> None:
        self.reasoning_effort = reasoning_effort

    def provenance(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model": "gpt-5.6-luna",
            "reasoning_effort": self.reasoning_effort,
        }

    def generate(self, *, stage, prompt, output_schema, payload) -> TrialGPTGeneration:
        del output_schema
        output = {
            "summary": f"summary {payload['topic_id']} {payload['arm_id']}",
            "conditions": [f"condition {payload['topic_id']} {payload['arm_id']}"],
        }
        return TrialGPTGeneration(
            stage=stage,
            prompt=prompt,
            output=output,
            raw_response=__import__("json").dumps(output),
            provider_id=self.provider_id,
        )


def test_three_luna_query_generation_pins_prompts_models_and_effort(tmp_path) -> None:
    cache = GenerationCache(tmp_path / "generation.sqlite3")
    try:
        queries = generate_three_luna_queries(
            (
                RetrievalQueryTopic("1", "Patient one."),
                RetrievalQueryTopic("2", "Patient two."),
            ),
            providers={
                "medium": _QueryProvider("medium"),
                "xhigh": _QueryProvider("xhigh"),
                "recall-explicit": _QueryProvider("medium"),
            },
            cache=cache,
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
    binding = FrozenRetrievalBinding(
        run_id="trialgpt-trec21-three-luna-v2",
        benchmark_lineage="trec-ct-2021",
        prepared_snapshot_id="sha256:" + "1" * 64,
        evaluation_package_id="sha256:" + "2" * 64,
        task_input_id="sha256:" + "3" * 64,
        pool_receipt_id="sha256:" + "4" * 64,
        pool_count=26_149,
        pool_ids_sha256="sha256:" + "5" * 64,
        protocol_approval_id=FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
        producer_git_commit="7" * 40,
        producer_dependency_lock_sha256="sha256:" + "8" * 64,
        release_manifest_id="sha256:" + "9" * 64,
        public_tree_id="sha256:" + "a" * 64,
        package_artifact_id="sha256:" + "b" * 64,
    )

    written = write_frozen_three_luna_retrieval(
        raw_retrievals,
        binding=binding,
        retrieval_implementation=_retrieval_implementation(),
        output_directory=tmp_path,
    )

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
        binding=FrozenRetrievalBinding(
            run_id="trialgpt-trec21-three-luna-v2",
            benchmark_lineage="trec-ct-2021",
            prepared_snapshot_id="sha256:" + "1" * 64,
            evaluation_package_id="sha256:" + "2" * 64,
            task_input_id="sha256:" + "3" * 64,
            pool_receipt_id="sha256:" + "4" * 64,
            pool_count=26_149,
            pool_ids_sha256="sha256:" + "5" * 64,
            protocol_approval_id=FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
            producer_git_commit="7" * 40,
            producer_dependency_lock_sha256="sha256:" + "8" * 64,
            release_manifest_id="sha256:" + "9" * 64,
            public_tree_id="sha256:" + "a" * 64,
            package_artifact_id="sha256:" + "b" * 64,
        ),
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
        binding=FrozenRetrievalBinding(
            run_id="trialgpt-trec21-three-luna-v2",
            benchmark_lineage="trec-ct-2021",
            prepared_snapshot_id="sha256:" + "1" * 64,
            evaluation_package_id="sha256:" + "2" * 64,
            task_input_id="sha256:" + "3" * 64,
            pool_receipt_id="sha256:" + "4" * 64,
            pool_count=26_149,
            pool_ids_sha256="sha256:" + "5" * 64,
            protocol_approval_id=FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
            producer_git_commit="7" * 40,
            producer_dependency_lock_sha256="sha256:" + "8" * 64,
            release_manifest_id="sha256:" + "9" * 64,
            public_tree_id="sha256:" + "a" * 64,
            package_artifact_id="sha256:" + "b" * 64,
        ),
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
    written = write_frozen_three_luna_retrieval(
        raw_retrievals,
        binding=FrozenRetrievalBinding(
            run_id="trialgpt-trec21-three-luna-v2",
            benchmark_lineage="trec-ct-2021",
            prepared_snapshot_id="sha256:" + "1" * 64,
            evaluation_package_id="sha256:" + "2" * 64,
            task_input_id="sha256:" + "3" * 64,
            pool_receipt_id="sha256:" + "4" * 64,
            pool_count=26_149,
            pool_ids_sha256="sha256:" + "5" * 64,
            protocol_approval_id=FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
            producer_git_commit="7" * 40,
            producer_dependency_lock_sha256="sha256:" + "8" * 64,
            release_manifest_id="sha256:" + "9" * 64,
            public_tree_id="sha256:" + "a" * 64,
            package_artifact_id="sha256:" + "b" * 64,
        ),
        retrieval_implementation=_retrieval_implementation(),
        output_directory=tmp_path,
    )
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
