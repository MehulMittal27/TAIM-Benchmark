# ruff: noqa: S101
from __future__ import annotations

import gzip
import json
import shutil
from dataclasses import replace

import pytest

from taim.artifacts import sha256_file
from taim.data import load_prepared_benchmark
from taim.data.prepared import PREPARED_MANIFEST_FILENAME
from taim.data.trec_ct_2021 import prepare_trec_ct_2021
from taim.schemas import SchemaValidationError
from taim.snapshot import (
    CAPABILITY_TYPED_PATIENT_CORE,
    AgeBound,
    AgeQuantity,
    DerivedView,
    FieldProvenance,
    PatientCoreAssertion,
    SourceRecordIdentity,
    TypedPatientCore,
)
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


def test_synthetic_trec_2021_sealing_hashes_remain_byte_compatible(tmp_path) -> None:
    prepared = _prepared(tmp_path)

    assert prepared.snapshot.contract_version == "2.0"
    assert prepared.snapshot.canonicalization_version == "taim-snapshot-canonical-json-v1"
    assert prepared.snapshot.snapshot_id == (
        "sha256:88bb717d44c7d5deb8f6b0d0ab1f41eaec4b708ffe207020681dc3e47cee9131"
    )
    assert prepared.snapshot.base_snapshot_id == prepared.snapshot.snapshot_id
    assert prepared.snapshot.logical_content_hashes() == {
        "topics": "sha256:fee2d693345c10e8a5c4d79af077ebbb751b3fc78eadcaceb2efa3cb99b748d9",
        "trials": "sha256:fc486729f4ecde180dbaa46ff8f43eba500d297220176abf9571a4884eb5581d",
        "derived_views": (
            "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        ),
    }
    assert prepared.evaluation_package.evaluation_package_id == (
        "sha256:92a27c4c6191a221754f4015cd2aa3e2ce33b21e06d615f13d1e91c1b3d10998"
    )
    assert {
        role: output["sha256"] for role, output in prepared.prepared_manifest["outputs"].items()
    } == {
        "topics": "sha256:8f9a6379fa513f3885033e1952bc19209726bf4220464994f2993ac893e2f2b6",
        "trials": "sha256:3f2e0be0370a9210f335349dc8c12d010e84b424a6a82e4dbcd14417c6983042",
        "diagnostics": ("sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        "judgments": ("sha256:cb78af031b3bf187c48de04da5d67926998fb17d68b4e315526e760d94b26264"),
    }
    assert prepared.prepared_manifest_hash == (
        "sha256:a2c6ac0aaa15127a59e280f3d95efa29a5cb4882610ff77652f519f562e83ad5"
    )


def test_logical_identity_ignores_json_whitespace_timestamp_and_build_version(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    repacked_directory = tmp_path / "repacked"
    shutil.copytree(prepared.directory, repacked_directory)

    topics_path = repacked_directory / "topics.jsonl"
    topic_payload = json.loads(topics_path.read_text(encoding="utf-8"))
    topics_path.write_text(
        json.dumps(topic_payload, ensure_ascii=False, separators=(", ", ": ")) + "\n",
        encoding="utf-8",
    )
    manifest_path = repacked_directory / PREPARED_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_topics_checksum = manifest["outputs"]["topics"]["sha256"]
    manifest["outputs"]["topics"].update(
        {
            "byte_size": topics_path.stat().st_size,
            "sha256": sha256_file(topics_path),
        }
    )
    manifest["created_at"] = "2030-01-01T00:00:00Z"
    manifest["build_provenance"]["connector_version"] = "refactored-only"
    manifest_path.write_text(
        json.dumps(manifest, indent=4, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    repacked = load_prepared_benchmark(repacked_directory)

    assert repacked.snapshot.snapshot_id == prepared.snapshot.snapshot_id
    assert repacked.package_checksums["topics.jsonl"] != original_topics_checksum
    assert repacked.prepared_manifest_hash != prepared.prepared_manifest_hash


def test_every_system_visible_layer_changes_snapshot_identity(tmp_path) -> None:
    snapshot = _prepared(tmp_path).snapshot
    topic = snapshot.topics[0]
    trial = snapshot.trials[0]
    section = trial.sections[0]
    core = trial.typed_clinical_core
    assert core is not None and core.minimum_age is not None

    changed_text = replace(
        snapshot,
        trials=(replace(trial, canonical_text=trial.canonical_text + " changed"),),
    )
    changed_patient_text = replace(
        snapshot,
        topics=(replace(topic, canonical_text=topic.canonical_text + " changed"),),
    )
    changed_patient_evidence = replace(
        snapshot,
        topics=(
            replace(
                topic,
                evidence_items=(
                    replace(
                        topic.evidence_items[0],
                        text=topic.evidence_items[0].text + " changed",
                    ),
                ),
            ),
        ),
    )
    changed_section = replace(
        snapshot,
        trials=(
            replace(
                trial,
                sections=(replace(section, text=section.text + " changed"), *trial.sections[1:]),
            ),
        ),
    )
    changed_core = replace(
        snapshot,
        trials=(
            replace(
                trial,
                typed_clinical_core=replace(
                    core,
                    minimum_age=AgeBound(
                        AgeQuantity("19", "years"),
                        core.minimum_age.provenance,
                    ),
                ),
            ),
        ),
    )
    changed_source_identity = replace(
        snapshot,
        trials=(
            replace(
                trial,
                source_identity=SourceRecordIdentity(
                    namespace=trial.source_identity.namespace,
                    record_id="NCT99999999",
                ),
            ),
        ),
    )
    changed_registry_extension = replace(
        snapshot,
        trials=(replace(trial, registry_extensions={"ctgov.phase": "phase-2"}),),
    )
    changed_additive_field = replace(
        snapshot,
        topics=(replace(topic, additional_fields={"future_layer": {"value": "new"}}),),
    )
    changed_typed_patient_core = replace(
        snapshot,
        topics=(
            replace(
                topic,
                typed_patient_core=TypedPatientCore(
                    profile_id="synthetic-patient-core-v1",
                    fields={
                        "age": PatientCoreAssertion(
                            value={"amount": "50", "unit": "years"},
                            provenance=topic.evidence_items[0].provenance,
                        )
                    },
                ),
            ),
        ),
        available_capabilities=snapshot.available_capabilities | {CAPABILITY_TYPED_PATIENT_CORE},
    )
    original_provenance = section.provenance[0]
    changed_provenance = replace(
        snapshot,
        trials=(
            replace(
                trial,
                sections=(
                    replace(
                        section,
                        provenance=(
                            FieldProvenance(
                                artifact_id=original_provenance.artifact_id,
                                artifact_sha256=original_provenance.artifact_sha256,
                                source_record_id=original_provenance.source_record_id,
                                locator_kind=original_provenance.locator_kind,
                                location=original_provenance.location + "/changed",
                                raw_value=original_provenance.raw_value,
                                transformation_rule=original_provenance.transformation_rule,
                            ),
                        ),
                    ),
                    *trial.sections[1:],
                ),
            ),
        ),
    )
    changed_capabilities = replace(
        snapshot,
        available_capabilities=snapshot.available_capabilities | {CAPABILITY_TYPED_PATIENT_CORE},
    )
    derived_view = DerivedView(
        name="criterion_split",
        version="1",
        input_snapshot_id=snapshot.base_snapshot_id,
        configuration={"method": "synthetic"},
        content={"NCT00000001": ["criterion one"]},
    )
    with_derived_view = replace(
        snapshot,
        available_capabilities=snapshot.available_capabilities | {derived_view.capability},
        derived_views=(derived_view,),
    )

    assert topic.evidence_items[0].provenance[0] in tuple(
        changed_typed_patient_core.iter_provenance()
    )

    assert {
        changed_text.snapshot_id,
        changed_patient_text.snapshot_id,
        changed_patient_evidence.snapshot_id,
        changed_section.snapshot_id,
        changed_core.snapshot_id,
        changed_source_identity.snapshot_id,
        changed_registry_extension.snapshot_id,
        changed_additive_field.snapshot_id,
        changed_typed_patient_core.snapshot_id,
        changed_provenance.snapshot_id,
        changed_capabilities.snapshot_id,
        with_derived_view.snapshot_id,
    }.isdisjoint({snapshot.snapshot_id})


def test_unknown_nested_optional_fields_survive_loading_and_change_identity(tmp_path) -> None:
    snapshot = _prepared(tmp_path).snapshot
    trial = snapshot.trials[0]
    section_payload = trial.sections[0].to_dict()
    section_payload["language"] = "en"
    section = type(trial.sections[0]).from_dict(section_payload)
    changed = replace(
        snapshot,
        trials=(replace(trial, sections=(section, *trial.sections[1:])),),
    )

    assert section.to_dict()["language"] == "en"
    assert changed.snapshot_id != snapshot.snapshot_id


def test_derived_views_use_name_order_as_their_canonical_set_order(tmp_path) -> None:
    snapshot = _prepared(tmp_path).snapshot
    views = tuple(
        DerivedView(
            name=name,
            version="1",
            input_snapshot_id=snapshot.base_snapshot_id,
            configuration={"method": name},
            content={"value": name},
        )
        for name in ("alpha_view", "zeta_view")
    )
    capabilities = snapshot.available_capabilities | {view.capability for view in views}

    ordered = replace(snapshot, available_capabilities=capabilities, derived_views=views)
    reversed_input = replace(
        snapshot,
        available_capabilities=capabilities,
        derived_views=tuple(reversed(views)),
    )

    assert reversed_input.snapshot_id == ordered.snapshot_id
    assert [view.name for view in reversed_input.derived_views] == [
        "alpha_view",
        "zeta_view",
    ]


def test_physical_corruption_fails_before_logical_loading(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    with (prepared.directory / "trials.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(" ")

    with pytest.raises(SchemaValidationError, match="byte size"):
        load_prepared_benchmark(prepared.directory)


def test_filename_compression_and_sharding_do_not_change_logical_identity(tmp_path) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={
            "NCT00000001": complete_trial_xml(trial_id="NCT00000001"),
            "NCT00000002": complete_trial_xml(trial_id="NCT00000002"),
        },
    )
    prepared = load_prepared_benchmark(
        prepare_trec_ct_2021(
            source,
            tmp_path / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        ).output_directory
    )
    repacked_directory = tmp_path / "repacked-layout"
    shutil.copytree(prepared.directory, repacked_directory)
    records_directory = repacked_directory / "records"
    records_directory.mkdir()

    original_topics = repacked_directory / "topics.jsonl"
    compressed_topics = records_directory / "patient-topics.jsonl.gz"
    compressed_topics.write_bytes(gzip.compress(original_topics.read_bytes(), mtime=0))
    original_topics.unlink()

    original_trials = repacked_directory / "trials.jsonl"
    trial_lines = original_trials.read_bytes().splitlines(keepends=True)
    assert len(trial_lines) == 2
    trial_shards = [
        records_directory / "trial-shard-000.jsonl",
        records_directory / "trial-shard-001.jsonl",
    ]
    for path, line in zip(trial_shards, trial_lines, strict=True):
        path.write_bytes(line)
    original_trials.unlink()

    manifest_path = repacked_directory / PREPARED_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["topics"] = [
        {
            "filename": "records/patient-topics.jsonl.gz",
            "record_count": 1,
            "byte_size": compressed_topics.stat().st_size,
            "sha256": sha256_file(compressed_topics),
            "compression": "gzip",
            "shard_index": 0,
        }
    ]
    manifest["outputs"]["trials"] = [
        {
            "filename": f"records/{path.name}",
            "record_count": 1,
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
            "compression": "none",
            "shard_index": index,
        }
        for index, path in enumerate(trial_shards)
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    repacked = load_prepared_benchmark(repacked_directory)

    assert repacked.snapshot.snapshot_id == prepared.snapshot.snapshot_id
    assert {trial.trial_id for trial in repacked.trials} == {
        "NCT00000001",
        "NCT00000002",
    }
    assert "records/patient-topics.jsonl.gz" in repacked.package_checksums
    assert "records/trial-shard-001.jsonl" in repacked.package_checksums
