"""Packaged synthetic benchmark loaded through the canonical contracts."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from taim.contracts import SNAPSHOT_CONTRACT_VERSION, content_sha256
from taim.data import PreparedBenchmark, SnapshotPreparationManifest
from taim.data.prepared import validate_prepared_benchmark
from taim.evaluation_package import EvaluationPackage
from taim.schemas import JsonValue, RelevanceJudgment, SchemaValidationError
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    BenchmarkSnapshot,
    BenchmarkTopic,
    TrialDocument,
)
from taim.source import SourceArtifact, SourceBundle

FIXTURE_BENCHMARK_NAME = "taim-trec-mini"
_FIXTURE_DIRECTORY = files("taim").joinpath("fixtures", "trec-mini")


def _records(resource: Traversable) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(resource.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise SchemaValidationError(f"blank record in {resource.name} at line {line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                f"invalid JSON in {resource.name} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise SchemaValidationError(
                f"record in {resource.name} at line {line_number} must be an object"
            )
        records.append(payload)
    return records


def _sha256(resource: Traversable) -> str:
    return f"sha256:{hashlib.sha256(resource.read_bytes()).hexdigest()}"


def _source_artifact(resource: Traversable, role: str) -> SourceArtifact:
    return SourceArtifact(
        artifact_id=role,
        role=role,
        filename=resource.name,
        byte_size=len(resource.read_bytes()),
        sha256=_sha256(resource),
        acquisition_date="2026-01-01",
        access_terms="packaged synthetic fixture",
        redistribution_terms="repository fixture",
        url=f"fixture://trec-mini/{resource.name}",
    )


def load_fixture() -> PreparedBenchmark:
    """Load the packaged synthetic benchmark as one Prepared Benchmark."""

    topic_resource = _FIXTURE_DIRECTORY.joinpath("topics.jsonl")
    trial_resource = _FIXTURE_DIRECTORY.joinpath("trials.jsonl")
    judgment_resource = _FIXTURE_DIRECTORY.joinpath("qrels.jsonl")
    topics = tuple(
        sorted(
            (BenchmarkTopic.from_dict(row) for row in _records(topic_resource)),
            key=lambda item: item.topic_id,
        )
    )
    trials = tuple(
        sorted(
            (TrialDocument.from_dict(row) for row in _records(trial_resource)),
            key=lambda item: item.trial_id,
        )
    )
    judgments = tuple(
        sorted(
            (RelevanceJudgment.from_dict(row) for row in _records(judgment_resource)),
            key=lambda item: (item.topic_id, item.trial_id),
        )
    )

    topic_ids = {topic.topic_id for topic in topics}
    trial_ids = {trial.trial_id for trial in trials}
    if len(topic_ids) != len(topics) or len(trial_ids) != len(trials):
        raise SchemaValidationError("fixture Snapshot identities must be unique")
    judgment_keys = [(item.topic_id, item.trial_id) for item in judgments]
    if len(judgment_keys) != len(set(judgment_keys)):
        raise SchemaValidationError("fixture Judgments must be unique")
    if any(item.topic_id not in topic_ids or item.trial_id not in trial_ids for item in judgments):
        raise SchemaValidationError(
            "fixture Evaluation Package references unknown Snapshot records"
        )
    if topic_ids - {item.topic_id for item in judgments}:
        raise SchemaValidationError("fixture topics without Judgments")

    artifacts = (
        _source_artifact(topic_resource, "topics"),
        _source_artifact(trial_resource, "trials"),
        _source_artifact(judgment_resource, "judgments"),
    )
    source_bundle = SourceBundle(
        artifacts=artifacts,
        lock_filename="trec-mini-source-lock.json",
        lock_sha256=content_sha256(
            {"schema_version": "1.0", "artifacts": [item.identity_dict() for item in artifacts]}
        ),
    )
    snapshot = BenchmarkSnapshot(
        benchmark_lineage=FIXTURE_BENCHMARK_NAME,
        snapshot_name=FIXTURE_BENCHMARK_NAME,
        topics=topics,
        trials=trials,
        available_capabilities=frozenset(
            {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
        ),
    )
    evaluation_package = EvaluationPackage(
        benchmark_lineage=snapshot.benchmark_lineage,
        task_id="synthetic-patient-to-trial-ranking",
        snapshot_id=snapshot.snapshot_id,
        judgments=judgments,
        provenance={
            "source_artifact_id": "judgments",
            "source_artifact_sha256": artifacts[2].sha256,
            "synthetic": True,
        },
    )
    recipe_id = "taim-trec-mini-snapshot-recipe-v1"
    recipe: dict[str, JsonValue] = {
        "recipe_id": recipe_id,
        "snapshot_contract_version": SNAPSHOT_CONTRACT_VERSION,
        "benchmark_lineage": snapshot.benchmark_lineage,
        "connector": {"name": "taim.fixtures", "version": "1.0"},
        "parsers": {"jsonl": "taim-fixture-jsonl-v1"},
        "field_disposition_table": [
            {
                "source": "packaged canonical fixture records",
                "disposition": "intentionally_ignored",
                "rationale": "fixture records already conform to the shared contract",
            }
        ],
        "canonical_patient_text_recipe": {
            "id": "taim-fixture-patient-text-v1",
            "whitespace": "preserve packaged canonical text",
            "unicode_normalization": "none",
            "separator": "not applicable",
            "omission": "omit absent optional layers",
        },
        "canonical_trial_text_recipe": {
            "id": "taim-fixture-trial-text-v1",
            "whitespace": "preserve packaged canonical text",
            "unicode_normalization": "none",
            "separator": "not applicable",
            "omission": "omit absent optional layers",
            "repetition": "preserve packaged order",
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
            "order": "source order",
        },
        "external_enrichment": "forbidden",
        "allowed_source_roles": ["topics", "trials", "judgments"],
        "allowed_artifact_ids": ["topics", "trials", "judgments"],
        "system_input_artifact_ids": ["topics", "trials"],
    }
    preparation = SnapshotPreparationManifest(
        source_recipe_id=recipe_id,
        source_recipe_hash=content_sha256(recipe),
        source_bundle_id=source_bundle.identity_for_artifacts(("topics", "trials")),
    )
    package_checksums = {
        resource.name: _sha256(resource)
        for resource in (topic_resource, trial_resource, judgment_resource)
    }
    prepared_manifest = {
        "schema_version": SNAPSHOT_CONTRACT_VERSION,
        "manifest_type": "taim-prepared-benchmark",
        "dataset_id": "trec-mini",
        "source_bundle": source_bundle.to_dict(),
        "snapshot_source_recipe": {
            "recipe_id": preparation.source_recipe_id,
            "sha256": preparation.source_recipe_hash,
            "definition": recipe,
        },
        "snapshot_preparation": preparation.to_dict(),
        "snapshot": snapshot.manifest_dict(),
        "evaluation_package": evaluation_package.manifest_dict(),
        "package_checksums": package_checksums,
        "build_provenance": {"connector": "taim.fixtures", "version": "1.0"},
    }
    return validate_prepared_benchmark(
        PreparedBenchmark(
            dataset_id="trec-mini",
            directory=Path(str(_FIXTURE_DIRECTORY)).resolve(),
            snapshot=snapshot,
            evaluation_package=evaluation_package,
            source_bundle=source_bundle,
            preparation=preparation,
            diagnostics=(),
            package_checksums=package_checksums,
            prepared_manifest=prepared_manifest,
            prepared_manifest_hash=content_sha256(prepared_manifest),
        )
    )


__all__ = ["FIXTURE_BENCHMARK_NAME", "load_fixture"]
