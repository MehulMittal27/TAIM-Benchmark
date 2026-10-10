from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taim.contracts import SNAPSHOT_CONTRACT_VERSION, content_sha256
from taim.data import PreparedBenchmark, SnapshotPreparationManifest
from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.evaluation_package import EvaluationPackage
from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialTaskInput
from taim.schemas import PRIMARY_RANKING_CANDIDATES, RelevanceJudgment, RunManifest
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    BenchmarkSnapshot,
    BenchmarkTopic,
    SourceRecordIdentity,
    Trial,
)
from taim.source import SourceArtifact, SourceBundle


def topic(topic_id: str = "1", text: str = "patient") -> BenchmarkTopic:
    return BenchmarkTopic(
        topic_id=topic_id,
        source_identity=SourceRecordIdentity("taim.test", topic_id),
        canonical_text=text,
    )


def trial(trial_id: str = "NCT00000001", text: str = "trial") -> Trial:
    return Trial(
        trial_id=trial_id,
        source_identity=SourceRecordIdentity("taim.test", trial_id),
        canonical_text=text,
    )


def prepared_benchmark(
    *,
    topics: tuple[BenchmarkTopic, ...] | None = None,
    trials: tuple[Trial, ...] | None = None,
    judgments: tuple[RelevanceJudgment, ...] | None = None,
    lineage: str = "test-benchmark",
    available_capabilities: frozenset[str] | None = None,
) -> PreparedBenchmark:
    topic_rows = topics or (topic(),)
    trial_rows = trials or (trial(),)
    judgment_rows = judgments or (
        RelevanceJudgment(topic_rows[0].topic_id, trial_rows[0].trial_id, 2),
    )
    snapshot = BenchmarkSnapshot(
        benchmark_lineage=lineage,
        snapshot_name=lineage,
        topics=topic_rows,
        trials=trial_rows,
        available_capabilities=available_capabilities
        or frozenset({CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}),
    )
    evaluation_package = EvaluationPackage(
        benchmark_lineage=lineage,
        task_id="test-ranking",
        snapshot_id=snapshot.snapshot_id,
        judgments=judgment_rows,
        provenance={"synthetic": True},
    )
    source_artifact = SourceArtifact(
        artifact_id="synthetic",
        role="synthetic",
        filename="synthetic.jsonl",
        byte_size=1,
        sha256="sha256:" + "1" * 64,
        acquisition_date="2026-01-01",
        access_terms="synthetic fixture",
        redistribution_terms="synthetic fixture",
        url="fixture://synthetic",
    )
    source_bundle = SourceBundle(
        artifacts=(source_artifact,),
        lock_filename="source-lock.json",
        lock_sha256="sha256:" + "2" * 64,
    )
    recipe = {
        "recipe_id": "test-snapshot-recipe-v1",
        "snapshot_contract_version": SNAPSHOT_CONTRACT_VERSION,
        "benchmark_lineage": lineage,
        "connector": {"name": "tests.benchmark_helpers", "version": "1.0"},
        "parsers": {"synthetic": "canonical-test-records-v1"},
        "field_disposition_table": [
            {
                "source": "synthetic canonical records",
                "disposition": "intentionally_ignored",
                "rationale": "records already satisfy the Snapshot contract",
            }
        ],
        "canonical_patient_text_recipe": {
            "id": "test-patient-text-v1",
            "whitespace": "preserve canonical text",
            "unicode_normalization": "none",
            "separator": "not applicable",
            "omission": "canonical text is required",
        },
        "canonical_trial_text_recipe": {
            "id": "test-trial-text-v1",
            "whitespace": "preserve canonical text",
            "unicode_normalization": "none",
            "separator": "not applicable",
            "omission": "canonical text is required",
            "repetition": "preserve canonical order",
            "section_order": ["canonical_text"],
            "typed_field_order": ["none"],
            "section_labels": {"canonical_text": "[CANONICAL_TEXT]"},
            "typed_labels": {"none": "[NONE]"},
        },
        "diagnostic_policy": {
            "version": "1.0",
            "field_invalidity": "reject fixture",
            "record_invalidity": "reject fixture",
            "conflict": "reject fixture",
            "order": "canonical record order",
        },
        "external_enrichment": "forbidden",
        "allowed_source_roles": ["synthetic"],
        "allowed_artifact_ids": ["synthetic"],
        "system_input_artifact_ids": ["synthetic"],
    }
    preparation = SnapshotPreparationManifest(
        source_recipe_id=recipe["recipe_id"],
        source_recipe_hash=content_sha256(recipe),
        source_bundle_id=source_bundle.source_bundle_id,
    )
    manifest = {
        "schema_version": SNAPSHOT_CONTRACT_VERSION,
        "manifest_type": "taim-prepared-benchmark",
        "dataset_id": "test",
        "source_bundle": source_bundle.to_dict(),
        "snapshot_source_recipe": {
            "recipe_id": preparation.source_recipe_id,
            "sha256": preparation.source_recipe_hash,
            "definition": recipe,
        },
        "snapshot_preparation": preparation.to_dict(),
        "snapshot": snapshot.manifest_dict(),
        "evaluation_package": evaluation_package.manifest_dict(),
        "package_checksums": {},
        "build_provenance": {"connector": "tests.benchmark_helpers", "version": "1.0"},
    }
    return PreparedBenchmark(
        dataset_id="test",
        directory=Path("/synthetic-prepared"),
        snapshot=snapshot,
        evaluation_package=evaluation_package,
        source_bundle=source_bundle,
        preparation=preparation,
        diagnostics=(),
        package_checksums={},
        prepared_manifest=manifest,
        prepared_manifest_hash=content_sha256(manifest),
    )


def task_input_for(
    prepared: PreparedBenchmark,
    *,
    clinical_as_of: datetime | None = None,
    prepared_snapshot_id: str | None = None,
) -> PatientToTrialTaskInput:
    """The Task Input a run over ``prepared`` binds, and the source its compact record needs."""

    return PatientToTrialTaskInput(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        topics=prepared.topics,
        trials=prepared.trials,
        available_capabilities=prepared.snapshot.available_capabilities,
        derived_views=prepared.snapshot.derived_views,
        prepared_snapshot_id=prepared_snapshot_id or prepared.snapshot.snapshot_id,
        clinical_as_of=clinical_as_of or datetime(2021, 4, 27, tzinfo=UTC),
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )


def manifest_for(
    prepared: PreparedBenchmark,
    *,
    benchmark_profile: dict[str, Any],
    run_id: str = "run-1",
    system_id: str = "synthetic-system",
    pipeline_depth: str = "retrieval",
    budget_k: int = 2,
    metric_cutoff: int | None = None,
    options: Mapping[str, object] | None = None,
    clinical_as_of: datetime | None = None,
    prepared_snapshot_id: str | None = None,
    stage_rankings: tuple[Any, ...] = (),
) -> RunManifest:
    selected_prepared_snapshot_id = prepared_snapshot_id or prepared.snapshot.snapshot_id
    system_snapshot = task_input_for(
        prepared,
        clinical_as_of=clinical_as_of,
        prepared_snapshot_id=selected_prepared_snapshot_id,
    )
    request = PatientToTrialRunRequest(
        snapshot=system_snapshot,
        run_id=run_id,
        top_k=budget_k,
        metric_cutoff=metric_cutoff or budget_k,
        options=options or {},
    )
    return RunManifest(
        run_id=run_id,
        system_id=system_id,
        created_at=datetime(2026, 7, 21, tzinfo=UTC),
        git_commit="abc123",
        configuration={
            "top_k": budget_k,
            "system_input": request.system_input_record_dict(),
        },
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        snapshot_contract_version=prepared.snapshot.contract_version,
        prepared_snapshot_id=selected_prepared_snapshot_id,
        task_input_id=system_snapshot.task_input_id,
        system_input_id=request.system_input_id,
        evaluation_package_id=prepared.evaluation_package.evaluation_package_id,
        benchmark_profile=benchmark_profile,
        primary_ranking=PRIMARY_RANKING_CANDIDATES,
        pipeline_depth=pipeline_depth,
        budget_k=budget_k,
        clinical_as_of=system_snapshot.clinical_as_of,
        patient_evidence_profile=system_snapshot.patient_evidence_profile.to_dict(),
        query_patient_versions=tuple(
            (version.patient_id, version.patient_version_id)
            for version in system_snapshot.patient_versions
        ),
        trial_corpus_versions=tuple(
            (version.trial_id, version.trial_version_id)
            for version in system_snapshot.trial_versions
        ),
        stage_rankings=stage_rankings,
    )


def _fixture_task_source(directory):
    from taim.entity_versions import PatientEvidenceProfile
    from taim.patient_to_trial import PatientToTrialTaskInput
    from taim.profiles import resolve_benchmark_profile
    from taim.release_fixture import load_release_fixture
    from taim.schemas import read_manifest

    manifest = read_manifest(directory / "manifest.json")
    return PatientToTrialTaskInput.from_benchmark(
        resolve_benchmark_profile(
            load_release_fixture(manifest.benchmark_lineage),
            profile_id=manifest.benchmark_profile["profile_id"],
        ),
        clinical_as_of=manifest.clinical_as_of,
        patient_evidence_profile=PatientEvidenceProfile.from_dict(
            manifest.patient_evidence_profile
        ),
    )
