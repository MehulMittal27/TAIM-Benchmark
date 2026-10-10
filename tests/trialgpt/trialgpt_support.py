"""Synthetic inputs for the TrialGPT tests: a frozen retrieval and its lock, and TrialGPT 1.0 runs.

The Codex stand-in answers in process, so a run makes no provider call. The frozen retrieval and
its lock are shared with the TrialGPT 2.0 tests, which re-rank the same retrieval.
"""

# ruff: noqa: S101
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

import taim.pipelines.trialgpt.retrieval_producer as retrieval_producer
from taim.artifacts import sha256_file
from taim.eligibility import (
    ELIGIBILITY_CRITERION_VIEW_CAPABILITY,
    build_eligibility_criterion_view,
    derive_complete_eligibility_section,
)
from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.model_codex import codex_preflight
from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialTaskInput
from taim.pipelines.trialgpt.paper import TRIALGPT_TOKENIZER_ID
from taim.pipelines.trialgpt.publication import (
    load_frozen_trialgpt_retrieval,
    make_publication_contract,
)
from taim.pipelines.trialgpt.system import (
    TRIALGPT_CODEX_MODEL,
    trialgpt_codex_model,
    trialgpt_publication_method_contract,
)
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
    CAPABILITY_FIELD_PROVENANCE,
    CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
    BenchmarkSnapshot,
    BenchmarkTopic,
    FieldProvenance,
    SemanticTextSection,
    SourceRecordIdentity,
    Trial,
)

TRACK = "trec-ct-2021"
TARGET_PROFILE = "trec-ct-2021-external-fidelity-26149"
FROZEN_DEPTH = 2_000
RELEASE_IDENTITY = {
    "release_manifest_id": "sha256:" + "1" * 64,
    "public_tree_id": "sha256:" + "2" * 64,
    "package_artifact_id": "sha256:" + "3" * 64,
}
PREPARED_SNAPSHOT_ID = "sha256:" + "4" * 64
EVALUATION_PACKAGE_ID = "sha256:" + "5" * 64
TRIALGPT_CODEX_VERSION = "codex-cli 0.0.0-taim-trialgpt-test"
CONTROLLER_COMMIT = "a" * 40
SNAPSHOT_CAPABILITIES = frozenset(
    {
        CAPABILITY_CANONICAL_PATIENT_TEXT,
        CAPABILITY_CANONICAL_TRIAL_TEXT,
        CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
        CAPABILITY_FIELD_PROVENANCE,
        CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
    }
)


def synthetic_section(role: str, text: str, ordinal: int) -> SemanticTextSection:
    provenance = FieldProvenance(
        artifact_id="fixture",
        artifact_sha256="sha256:" + "a" * 64,
        source_record_id="fixture",
        locator_kind="json_pointer",
        location=f"/{role}",
        raw_value=text,
        transformation_rule="synthetic fixture",
    )
    return SemanticTextSection(role, text, ordinal, (provenance,))


def write_frozen_retrieval(
    root: Path,
    task_input: PatientToTrialTaskInput,
    *,
    release_identity: Mapping[str, str] = RELEASE_IDENTITY,
    profile: str = TARGET_PROFILE,
) -> tuple[Path, Path]:
    """A frozen consensus ranking and its lock, binding v11 and carrying its receipt."""

    rows = []
    top_digests = {}
    for topic in task_input.topics:
        # Each topic ranks the corpus in its own deterministic order.
        ordered = sorted(
            task_input.trials,
            key=lambda trial: hashlib.sha256(
                f"{topic.topic_id}/{trial.trial_id}".encode()
            ).digest(),
        )
        topic_rows = [
            {
                "configuration": "judgment-blind-consensus-all-luna",
                "rank": rank,
                "score": float(FROZEN_DEPTH + 1 - rank),
                "topic_id": topic.topic_id,
                "trial_id": trial.trial_id,
            }
            for rank, trial in enumerate(ordered, start=1)
        ]
        rows.extend(topic_rows)
        top = [
            {"rank": row["rank"], "score": row["score"], "trial_id": row["trial_id"]}
            for row in topic_rows[:500]
        ]
        top_digests[topic.topic_id] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(top, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
        )
    artifact = root / "frozen-retrieval.jsonl"
    artifact.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    receipt = {
        "artifact_type": "taim-trialgpt-query-plan-receipt",
        "schema_version": "1.0",
        "protocol_id": retrieval_producer.TRIALGPT_RETRIEVAL_PROTOCOL_ID,
        "protocol_sha256": retrieval_producer.FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
        "system_id": "TrialGPT-TAIM-Luna-v1",
        "run_id": "synthetic-retrieval",
        "track": TRACK,
        "profile": profile,
        "prepared_snapshot_id": PREPARED_SNAPSHOT_ID,
        "release": dict(release_identity),
        "query_plans_sha256": "sha256:" + "9" * 64,
        "logical_calls": 3 * len(task_input.topics),
        "generation": {"generation_cache_sha256": "sha256:" + "7" * 64, "interruptions": []},
        "generation_ordinal": 1,
    }
    lock = {
        "artifact_type": "taim-trialgpt-frozen-retrieval-lock",
        "schema_version": "1.0",
        "system_id": "TrialGPT-TAIM-Luna-v1",
        "benchmark_lineage": TRACK,
        "prepared_snapshot_id": PREPARED_SNAPSHOT_ID,
        "evaluation_package_id": EVALUATION_PACKAGE_ID,
        "source_input_lock": {
            "task_input_id": task_input.task_input_id,
            "run_id": "synthetic-retrieval",
            "protocol_approval_id": retrieval_producer.FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
            "query_plans_sha256": "sha256:" + "9" * 64,
            "query_plan_receipt": receipt,
            "query_plan_receipt_sha256": "sha256:" + "8" * 64,
            "release_identity": dict(release_identity),
        },
        "ranking": {
            "byte_size": artifact.stat().st_size,
            "configuration": "judgment-blind-consensus-all-luna",
            "filename": artifact.name,
            "output_depth": FROZEN_DEPTH,
            "publication_candidate_depth": 500,
            "row_count": FROZEN_DEPTH * len(task_input.topics),
            "sha256": sha256_file(artifact),
            "topic_count": len(task_input.topics),
        },
        "method": {"producer_contract_id": "synthetic"},
        "retrieval_implementation": {"implementation": "synthetic"},
        "source_rankings": {"arms": ["medium", "xhigh", "recall-explicit"]},
        "tokenizer": {"id": TRIALGPT_TOKENIZER_ID},
        "top_500_sha256_by_topic": top_digests,
        "redistribution": {"terms": "synthetic fixture"},
    }
    lock_path = root / "frozen-retrieval-lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return artifact, lock_path


def trialgpt_trial(index: int) -> Trial:
    """A trial with one inclusion and one exclusion criterion, as TrialGPT 1.0 matches them."""

    trial_id = f"NCT{index:08d}"
    title = f"Synthetic trial {index}"
    eligibility = derive_complete_eligibility_section(
        synthetic_section("eligibility", "Adult patient without active infection.", 2),
        # ClinicalTrials.gov's layout: a blank line after each heading and between items.
        source_text=(
            f"Inclusion Criteria:\n\n  - Adult patient {index}.\n\n"
            "Exclusion Criteria:\n\n  - Active infection."
        ),
    )
    return Trial(
        trial_id,
        SourceRecordIdentity("taim.fixture", trial_id),
        f"{title}\n[ELIGIBILITY] {eligibility.text}",
        (
            synthetic_section("brief_title", title, 0),
            synthetic_section("summary", "A synthetic trial.", 1),
            eligibility,
        ),
    )


def trialgpt_task_input(topic_count: int) -> PatientToTrialTaskInput:
    """A TREC 2021 Task Input with the eligibility criterion view TrialGPT 1.0 requires."""

    topics = tuple(
        BenchmarkTopic(
            topic_id=str(number),
            source_identity=SourceRecordIdentity("taim.fixture", str(number)),
            canonical_text=f"Patient {number} is an adult with a synthetic condition.",
        )
        for number in range(1, topic_count + 1)
    )
    trials = tuple(trialgpt_trial(index) for index in range(1, FROZEN_DEPTH + 1))
    snapshot = BenchmarkSnapshot(
        benchmark_lineage=TRACK,
        snapshot_name=f"{TRACK}-trialgpt-qualification",
        topics=topics,
        trials=trials,
        available_capabilities=SNAPSHOT_CAPABILITIES,
    )
    return PatientToTrialTaskInput(
        benchmark_lineage=TRACK,
        topics=topics,
        trials=trials,
        available_capabilities=SNAPSHOT_CAPABILITIES | {ELIGIBILITY_CRITERION_VIEW_CAPABILITY},
        derived_views=(build_eligibility_criterion_view(snapshot),),
        prepared_snapshot_id=PREPARED_SNAPSHOT_ID,
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )


@dataclass
class TrialGPTCodex:
    """What the in-process Codex stand-in answered, and which trials it answers malformed."""

    calls: list[str] = field(default_factory=list)
    malformed_trials: set[str] = field(default_factory=set)


def _events(payload: str) -> str:
    item = {"type": "item.completed", "item": {"type": "agent_message", "text": payload}}
    return json.dumps(item) + "\n"


def install_trialgpt_codex(monkeypatch: pytest.MonkeyPatch) -> TrialGPTCodex:
    """Answer every ``codex`` command in process: version, bundled catalog and ``exec``.

    Matching answers name criterion 0 of each list; the aggregation answer's scores depend on
    the prompt, so candidates rank deterministically. A trial in ``malformed_trials`` (by its
    title number) gets an inclusion answer that never parses, so it is abstained.
    """

    codex = TrialGPTCodex()
    catalog = json.dumps({"models": [{"slug": TRIALGPT_CODEX_MODEL, "context_window": 1}]})
    run = subprocess.run

    def serve(command, *args, input=None, **kwargs):
        if not (isinstance(command, list | tuple) and command and command[0] == "codex"):
            return run(command, *args, input=input, **kwargs)
        if list(command[1:]) == ["--version"]:
            return subprocess.CompletedProcess(command, 0, TRIALGPT_CODEX_VERSION + "\n", "")
        if list(command[1:]) == ["debug", "models", "--bundled"]:
            return subprocess.CompletedProcess(command, 0, catalog, "")
        assert command[1] == "exec"
        codex.calls.append(input)
        if "the inclusion criteria of a clinical trial" in input:
            if any(f"Synthetic trial {trial}\n" in input for trial in codex.malformed_trials):
                return subprocess.CompletedProcess(command, 0, _events("not json"), "")
            answer = {"criteria": {"0": ["Synthetic assessment.", [0], "included"]}}
        elif "the exclusion criteria of a clinical trial" in input:
            answer = {"criteria": {"0": ["Synthetic assessment.", [0], "not excluded"]}}
        else:
            digest = hashlib.sha256(input.encode("utf-8")).digest()
            relevance = digest[0] % 101
            answer = {
                "relevance_explanation": "Synthetic relevance.",
                "relevance_score_R": relevance,
                "eligibility_explanation": "Synthetic eligibility.",
                # Within the schema's bounds, -R to R.
                "eligibility_score_E": digest[1] % (2 * relevance + 1) - relevance,
            }
        return subprocess.CompletedProcess(command, 0, _events(json.dumps(answer)), "")

    monkeypatch.setattr(subprocess, "run", serve)
    return codex


def publication_request(
    root: Path,
    task_input: PatientToTrialTaskInput,
    *,
    run_id: str,
    workers: int = 8,
    profile: str = TARGET_PROFILE,
) -> PatientToTrialRunRequest:
    """A TrialGPT 1.0 request with its frozen retrieval and preflight contract, as the CLI binds."""

    root.mkdir(parents=True, exist_ok=True)
    artifact, lock = write_frozen_retrieval(root, task_input, profile=profile)
    contract = make_publication_contract(
        taim_git_commit=CONTROLLER_COMMIT,
        prepared_snapshot_id=PREPARED_SNAPSHOT_ID,
        evaluation_package_id=EVALUATION_PACKAGE_ID,
        frozen_retrieval=load_frozen_trialgpt_retrieval(artifact, lock_path=lock),
        provider_configuration=codex_preflight(
            trialgpt_codex_model(codex_executable="codex").configuration
        ),
        method_configuration=trialgpt_publication_method_contract(),
        generation_workers=workers,
        topic_ids=[topic.topic_id for topic in task_input.topics],
    )
    contract_path = root / "publication-contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return PatientToTrialRunRequest(
        snapshot=task_input,
        run_id=run_id,
        top_k=20,
        metric_cutoff=10,
        options={
            "publication_contract_path": contract_path,
            "frozen_retrieval_path": artifact,
            "frozen_retrieval_lock_path": lock,
            "workspace": root / "workspace",
            "controller_git_commit": CONTROLLER_COMMIT,
            "controller_working_tree_dirty": False,
            "retrieval_source_snapshot_id": PREPARED_SNAPSHOT_ID,
            "evaluation_package_id": EVALUATION_PACKAGE_ID,
            "generation_workers": workers,
            "codex_executable": "codex",
            "executing_release_identity": dict(RELEASE_IDENTITY),
        },
    )


__all__ = [
    "EVALUATION_PACKAGE_ID",
    "FROZEN_DEPTH",
    "PREPARED_SNAPSHOT_ID",
    "RELEASE_IDENTITY",
    "TARGET_PROFILE",
    "TRACK",
    "TrialGPTCodex",
    "install_trialgpt_codex",
    "publication_request",
    "synthetic_section",
    "trialgpt_task_input",
    "trialgpt_trial",
    "write_frozen_retrieval",
]
