# ruff: noqa: S101
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import replace

import pytest

from taim.artifacts import sha256_file
from taim.data import load_prepared_benchmark
from taim.data.prepared import (
    PREPARED_MANIFEST_FILENAME,
    validate_prepared_benchmark,
    write_prepared_benchmark,
)
from taim.data.trec_ct_2021 import prepare_trec_ct_2021
from taim.evaluation_package import ConnectorDiagnostic, EvaluationPackage
from taim.fixtures import load_fixture
from taim.schemas import SchemaValidationError
from taim.snapshot import (
    CAPABILITY_TYPED_TRIAL_CORE,
    DerivedView,
    FieldProvenance,
    content_sha256,
)
from tests.benchmark_helpers import prepared_benchmark
from tests.snapshot_helpers import (
    TREC_FIXED_TIME,
    complete_trial_xml,
    write_trec_source_bundle,
)


def _prepared(tmp_path):
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": complete_trial_xml()},
    )
    return load_prepared_benchmark(
        prepare_trec_ct_2021(
            source,
            tmp_path / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        ).output_directory
    )


def _repacked(prepared, tmp_path):
    directory = tmp_path / "repacked"
    shutil.copytree(prepared.directory, directory)
    manifest_path = directory / PREPARED_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return directory, manifest_path, manifest


def _write_manifest(path, manifest) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluation_identity(directory, evaluation) -> str:
    judgments = [
        json.loads(line)
        for line in (directory / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return content_sha256(
        {
            "schema_version": evaluation["schema_version"],
            "benchmark_lineage": evaluation["benchmark_lineage"],
            "task_id": evaluation["task_id"],
            "snapshot_id": evaluation["snapshot_id"],
            "judgments": judgments,
            "provenance": evaluation["provenance"],
        }
    )


def _resign_recipe(manifest) -> None:
    recipe = manifest["snapshot_source_recipe"]
    recipe["sha256"] = content_sha256(recipe["definition"])
    manifest["snapshot_preparation"]["source_recipe_hash"] = recipe["sha256"]


def _resign_output(manifest, directory, role: str) -> None:
    output = manifest["outputs"][role]
    path = directory / output["filename"]
    output["byte_size"] = path.stat().st_size
    output["sha256"] = sha256_file(path)


def _qrels_provenance(prepared) -> FieldProvenance:
    qrels = prepared.source_bundle.artifact_by_id("qrels")
    return FieldProvenance(
        artifact_id="qrels",
        artifact_sha256=qrels.sha256,
        source_record_id="1",
        locator_kind="archive_member",
        location="zip-member:qrels2021.txt",
        raw_value="1 0 NCT00000001 2",
        transformation_rule="synthetic-leak-regression-v1",
    )


def test_shared_prepared_benchmark_fixture_satisfies_the_canonical_contract() -> None:
    prepared = prepared_benchmark()

    assert validate_prepared_benchmark(prepared) is prepared


def test_writer_hashes_jsonl_outputs_while_streaming(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepared_benchmark()

    def reject_second_file_traversal(path) -> str:
        raise AssertionError(f"output was reopened for hashing: {path}")

    monkeypatch.setattr("taim.data.prepared.sha256_file", reject_second_file_traversal)
    result = write_prepared_benchmark(
        tmp_path / "streamed-output-hashes",
        dataset_id=prepared.dataset_id,
        snapshot=prepared.snapshot,
        evaluation_package=prepared.evaluation_package,
        source_bundle=prepared.source_bundle,
        source_recipe=prepared.prepared_manifest["snapshot_source_recipe"]["definition"],
        diagnostics=prepared.diagnostics,
        created_at=TREC_FIXED_TIME,
        build_provenance=prepared.prepared_manifest["build_provenance"],
    )

    for output in result.manifest["outputs"].values():
        path = result.output_directory / output["filename"]
        assert output["sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepared_contract_content_is_recursively_immutable() -> None:
    prepared = prepared_benchmark()
    original_manifest_hash = prepared.prepared_manifest_hash
    build_provenance = prepared.prepared_manifest["build_provenance"]
    assert isinstance(build_provenance, Mapping)

    with pytest.raises(TypeError):
        prepared.package_checksums["new.jsonl"] = "sha256:" + "f" * 64  # type: ignore[index]
    with pytest.raises(TypeError):
        prepared.prepared_manifest["dataset_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        build_provenance["connector"] = "changed"  # type: ignore[index]

    assert prepared.prepared_manifest_hash == original_manifest_hash


def test_preparation_result_manifest_is_recursively_immutable(tmp_path) -> None:
    prepared = prepared_benchmark()
    result = write_prepared_benchmark(
        tmp_path / "immutable-prepared",
        dataset_id=prepared.dataset_id,
        snapshot=prepared.snapshot,
        evaluation_package=prepared.evaluation_package,
        source_bundle=prepared.source_bundle,
        source_recipe=prepared.prepared_manifest["snapshot_source_recipe"]["definition"],
        diagnostics=prepared.diagnostics,
        created_at=TREC_FIXED_TIME,
        build_provenance=prepared.prepared_manifest["build_provenance"],
    )
    build_provenance = result.manifest["build_provenance"]
    assert isinstance(build_provenance, Mapping)

    with pytest.raises(TypeError):
        result.manifest["dataset_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        build_provenance["connector"] = "changed"  # type: ignore[index]

    serialized = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert json.loads(json.dumps(result.manifest)) == serialized
    assert serialized["dataset_id"] == prepared.dataset_id


def test_in_memory_fixture_rejects_incomplete_judgment_artifact_provenance() -> None:
    prepared = load_fixture()
    malformed_evaluation = replace(
        prepared.evaluation_package,
        provenance={"source_artifact_id": "judgments", "synthetic": True},
    )

    with pytest.raises(
        SchemaValidationError,
        match="Evaluation Package provenance artifact identity is malformed",
    ):
        validate_prepared_benchmark(replace(prepared, evaluation_package=malformed_evaluation))


def test_snapshot_provenance_cannot_reference_an_evaluator_only_artifact_when_writing(
    tmp_path,
) -> None:
    prepared = _prepared(tmp_path)
    trial = prepared.snapshot.trials[0]
    leaked_trial = replace(
        trial,
        sections=(
            replace(trial.sections[0], provenance=(_qrels_provenance(prepared),)),
            *trial.sections[1:],
        ),
    )
    leaked_snapshot = replace(prepared.snapshot, trials=(leaked_trial,))
    evaluation_package = EvaluationPackage(
        benchmark_lineage=prepared.evaluation_package.benchmark_lineage,
        task_id=prepared.evaluation_package.task_id,
        snapshot_id=leaked_snapshot.snapshot_id,
        judgments=prepared.evaluation_package.judgments,
        provenance=prepared.evaluation_package.provenance,
    )

    with pytest.raises(SchemaValidationError, match="System-input artifacts"):
        write_prepared_benchmark(
            tmp_path / "leaked-prepared",
            dataset_id=prepared.dataset_id,
            snapshot=leaked_snapshot,
            evaluation_package=evaluation_package,
            source_bundle=prepared.source_bundle,
            source_recipe=prepared.prepared_manifest["snapshot_source_recipe"]["definition"],
            diagnostics=prepared.diagnostics,
            created_at=TREC_FIXED_TIME,
            build_provenance=prepared.prepared_manifest["build_provenance"],
        )


def test_snapshot_provenance_cannot_reference_an_evaluator_only_artifact_when_loading(
    tmp_path,
) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    trial = prepared.snapshot.trials[0]
    leaked_trial = replace(
        trial,
        sections=(
            replace(trial.sections[0], provenance=(_qrels_provenance(prepared),)),
            *trial.sections[1:],
        ),
    )
    leaked_snapshot = replace(prepared.snapshot, trials=(leaked_trial,))
    (directory / "trials.jsonl").write_text(f"{leaked_trial.to_json()}\n", encoding="utf-8")
    _resign_output(manifest, directory, "trials")
    manifest["snapshot"] = leaked_snapshot.manifest_dict()
    manifest["evaluation_package"]["snapshot_id"] = leaked_snapshot.snapshot_id
    manifest["evaluation_package"]["evaluation_package_id"] = _evaluation_identity(
        directory, manifest["evaluation_package"]
    )
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="System-input artifacts"):
        load_prepared_benchmark(directory)


def test_connector_diagnostics_may_reference_a_recipe_allowed_evaluator_artifact(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    diagnostic = ConnectorDiagnostic(
        code="synthetic-evaluator-diagnostic",
        severity="field_nonfatal",
        message="evaluator-side source artifact remains audit-visible",
        record_id="1",
        field=None,
        provenance=(_qrels_provenance(prepared),),
    )
    result = write_prepared_benchmark(
        tmp_path / "diagnostic-prepared",
        dataset_id=prepared.dataset_id,
        snapshot=prepared.snapshot,
        evaluation_package=prepared.evaluation_package,
        source_bundle=prepared.source_bundle,
        source_recipe=prepared.prepared_manifest["snapshot_source_recipe"]["definition"],
        diagnostics=(diagnostic,),
        created_at=TREC_FIXED_TIME,
        build_provenance=prepared.prepared_manifest["build_provenance"],
    )

    loaded = load_prepared_benchmark(result.output_directory)
    assert loaded.diagnostics == (diagnostic,)
    assert not hasattr(loaded.snapshot, "diagnostics")


def test_evaluation_package_identity_includes_snapshot_binding(tmp_path) -> None:
    evaluation_package = _prepared(tmp_path).evaluation_package

    rebound = replace(evaluation_package, snapshot_id="sha256:" + "f" * 64)

    assert rebound.evaluation_package_id != evaluation_package.evaluation_package_id


def test_evaluation_package_owns_its_reloadable_serialization(tmp_path) -> None:
    evaluation_package = _prepared(tmp_path).evaluation_package

    serialized = evaluation_package.to_dict()

    assert EvaluationPackage.from_dict(serialized) == evaluation_package
    assert serialized["evaluation_package_id"] == evaluation_package.evaluation_package_id


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contract_version", "1.0", "contract_version"),
        ("contract_version", "3.0", "contract_version"),
        ("canonicalization_version", "unknown-canonicalizer", "canonicalization_version"),
    ],
)
def test_snapshot_manifest_rejects_unsupported_contract_semantics(
    tmp_path,
    field: str,
    value: str,
    message: str,
) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    manifest["snapshot"][field] = value
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match=message):
        load_prepared_benchmark(directory)


@pytest.mark.parametrize("field", ["source_recipe_id", "source_recipe_hash", "source_bundle_id"])
def test_snapshot_manifest_rejects_injected_preparation_metadata(tmp_path, field: str) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    manifest["snapshot"][field] = "preparation-only"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="Snapshot manifest"):
        load_prepared_benchmark(directory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("created_at", "not-a-time", "created_at"),
        ("build_provenance", "not-an-object", "build_provenance"),
    ],
)
def test_prepared_manifest_rejects_malformed_build_metadata(
    tmp_path,
    field: str,
    value,
    message: str,
) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    manifest[field] = value
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match=message):
        load_prepared_benchmark(directory)


def test_snapshot_recipe_identity_must_match_the_frozen_recipe_manifest(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    manifest["snapshot_source_recipe"]["recipe_id"] = "different-recipe"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="Source Recipe id"):
        load_prepared_benchmark(directory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("snapshot_contract_version", "3.0", "contract version"),
        ("benchmark_lineage", "another-benchmark", "benchmark lineage"),
        ("allowed_artifact_ids", ["topics"], "allowed artifacts"),
        ("external_enrichment", "allowed", "external_enrichment"),
    ],
)
def test_source_recipe_must_authorize_the_exact_snapshot_inputs(
    tmp_path,
    field: str,
    value,
    message: str,
) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    manifest["snapshot_source_recipe"]["definition"][field] = value
    _resign_recipe(manifest)
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match=message):
        load_prepared_benchmark(directory)


@pytest.mark.parametrize(
    "field",
    [
        "connector",
        "parsers",
        "field_disposition_table",
        "canonical_patient_text_recipe",
        "canonical_trial_text_recipe",
        "diagnostic_policy",
        "external_enrichment",
    ],
)
def test_source_recipe_rejects_missing_required_frozen_sections(tmp_path, field: str) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    del manifest["snapshot_source_recipe"]["definition"][field]
    _resign_recipe(manifest)
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="Source Recipe"):
        load_prepared_benchmark(directory)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("connector", "version"),
        ("canonical_patient_text_recipe", "id"),
        ("canonical_trial_text_recipe", "section_order"),
        ("diagnostic_policy", "version"),
    ],
)
def test_source_recipe_rejects_incomplete_required_subsections(
    tmp_path, section: str, field: str
) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    del manifest["snapshot_source_recipe"]["definition"][section][field]
    _resign_recipe(manifest)
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="Source Recipe"):
        load_prepared_benchmark(directory)


def test_source_recipe_rejects_a_mapping_without_normalization_rule(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    mapping = manifest["snapshot_source_recipe"]["definition"]["field_disposition_table"][0]
    del mapping["normalization_rule"]
    _resign_recipe(manifest)
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="normalization_rule"):
        load_prepared_benchmark(directory)


def test_evaluation_package_rejects_an_unsupported_schema_version(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    manifest["evaluation_package"]["schema_version"] = "2.0"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="Evaluation Package schema_version"):
        load_prepared_benchmark(directory)


def test_source_bundle_rejects_an_unsupported_schema_version(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    manifest["source_bundle"]["schema_version"] = "2.0"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="Source Bundle schema_version"):
        load_prepared_benchmark(directory)


def test_evaluation_package_must_share_the_snapshot_benchmark_lineage(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    evaluation = manifest["evaluation_package"]
    evaluation["benchmark_lineage"] = "another-benchmark"
    evaluation["evaluation_package_id"] = _evaluation_identity(directory, evaluation)
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="benchmark lineage"):
        load_prepared_benchmark(directory)


def test_evaluation_provenance_must_resolve_to_the_source_bundle(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    evaluation = manifest["evaluation_package"]
    evaluation["provenance"]["source_artifact_sha256"] = "sha256:" + "b" * 64
    evaluation["evaluation_package_id"] = _evaluation_identity(directory, evaluation)
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="Evaluation Package provenance digest"):
        load_prepared_benchmark(directory)


def test_malformed_derived_view_entries_are_rejected_not_dropped(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    manifest["snapshot"]["derived_views"] = [None]
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="Derived Views must contain objects"):
        load_prepared_benchmark(directory)


def test_derived_view_rejects_an_unsupported_schema_version(tmp_path) -> None:
    snapshot = _prepared(tmp_path).snapshot
    view = DerivedView(
        name="criterion_split",
        version="1",
        input_snapshot_id=snapshot.base_snapshot_id,
        configuration={"method": "synthetic"},
        content={"NCT00000001": ["criterion"]},
    )
    serialized = view.to_dict()
    serialized["schema_version"] = "3.0"

    with pytest.raises(SchemaValidationError, match="Derived View schema_version"):
        DerivedView.from_dict(serialized)


def test_package_output_paths_cannot_escape_the_prepared_directory(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    directory, manifest_path, manifest = _repacked(prepared, tmp_path)
    manifest["outputs"]["topics"]["filename"] = "../repacked/topics.jsonl"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SchemaValidationError, match="safe relative POSIX path"):
        load_prepared_benchmark(directory)


def test_snapshot_capability_declarations_are_closed_and_content_consistent(tmp_path) -> None:
    snapshot = _prepared(tmp_path).snapshot

    with pytest.raises(SchemaValidationError, match="unsupported Snapshot capability"):
        replace(snapshot, available_capabilities=snapshot.available_capabilities | {"future_core"})
    with pytest.raises(SchemaValidationError, match="typed trial core content"):
        replace(
            snapshot,
            available_capabilities=snapshot.available_capabilities - {CAPABILITY_TYPED_TRIAL_CORE},
        )


def test_snapshot_rejects_duplicate_benchmark_facing_identities(tmp_path) -> None:
    snapshot = _prepared(tmp_path).snapshot

    with pytest.raises(SchemaValidationError, match="topic identities must be unique"):
        replace(snapshot, topics=(snapshot.topics[0], snapshot.topics[0]))
    with pytest.raises(SchemaValidationError, match="trial identities must be unique"):
        replace(snapshot, trials=(snapshot.trials[0], snapshot.trials[0]))
