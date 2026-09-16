"""The external-fidelity Profile's public checks, which do not depend on any external System."""

from __future__ import annotations

import hashlib

# ruff: noqa: S101
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import taim.public_patient_to_trial as public_patient_to_trial
import taim.release_cli as release_cli
import taim.release_profiles as release_profiles
from taim.pipeline_extensions import pipeline_extensions
from taim.public_patient_to_trial import (
    _validate_judgment_union_membership,
    load_local_evaluation_package,
    load_public_patient_to_trial_run,
)
from taim.release_cli import _verified_external_pool_ids, main
from taim.release_fixture import load_release_fixture
from taim.release_profiles import (
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
    inspect_external_fidelity_profile,
    ordered_trial_ids_sha256,
    resolve_benchmark_profile,
)
from taim.release_support import (
    FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256,
    load_release_support,
    require_frozen_effectiveness_protocol,
)
from taim.schemas import SchemaValidationError

# The external-fidelity Profile's Systems that no pipeline extension declares. Two of them are the
# staged row's components, which are not reportable rows.
TAIM_EXTERNAL_FIDELITY_SYSTEMS = {
    "bm25",
    "bm25-folded",
    "dense-bge-m3",
    "dense-qwen3-embedding-0.6b",
    "dense-qwen3-embedding-0.6b-no-template",
    "rrf",
}


def test_external_fidelity_rows_outside_the_shipped_extensions_are_the_taim_systems() -> None:
    support_matrix = load_release_support()["support_matrix"]
    rows = [row for row in support_matrix if row["profile"] == EXTERNAL_FIDELITY_TREC_2021_PROFILE]
    assert {(row["track"], row["task"], row["evidence_scope"]) for row in rows} == {
        ("trec-ct-2021", "patient_to_trial", "external_fidelity_effectiveness")
    }
    # A pipeline extension's own public test checks the Systems it declares.
    extension_systems = {
        system for extension in pipeline_extensions() for system in extension.system_tasks
    }
    assert {row["system"] for row in rows} - extension_systems == TAIM_EXTERNAL_FIDELITY_SYSTEMS
    # Equality, not a denylist: an undeclared System fails here without this test naming it.
    assert {row["system"] for row in support_matrix} - extension_systems == {
        *TAIM_EXTERNAL_FIDELITY_SYSTEMS,
        "bm25-trial-to-patient",
    }


def test_external_fidelity_pool_requires_exact_caller_supplied_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = load_release_fixture("trec-ct-2021")
    selected_ids = tuple(trial.trial_id for trial in prepared.trials[:-1])
    production_profile = release_profiles.load_profile(EXTERNAL_FIDELITY_TREC_2021_PROFILE)
    assert production_profile.expected_pool_count == 26_149
    fixture_profile = replace(production_profile, expected_pool_count=len(selected_ids))
    original_load_profile = release_profiles.load_profile
    monkeypatch.setattr(
        release_profiles,
        "load_profile",
        lambda profile_id: (
            fixture_profile
            if profile_id == EXTERNAL_FIDELITY_TREC_2021_PROFILE
            else original_load_profile(profile_id)
        ),
    )
    receipt = inspect_external_fidelity_profile(
        prepared,
        profile_id=EXTERNAL_FIDELITY_TREC_2021_PROFILE,
        supplied_pool_ids=selected_ids,
        expected_pool_count=len(selected_ids),
    )
    assert receipt["selection_policy"] == "caller_supplied_external_fidelity_membership"
    assert receipt["pool_ids_sha256"] == ordered_trial_ids_sha256(selected_ids)

    resolved = resolve_benchmark_profile(
        prepared,
        profile_id=EXTERNAL_FIDELITY_TREC_2021_PROFILE,
        expected_pool_count=len(selected_ids),
        expected_pool_ids_sha256=receipt["pool_ids_sha256"],
        expected_pool_receipt_id=receipt["pool_receipt_id"],
        supplied_pool_ids=selected_ids,
    )
    assert tuple(trial.trial_id for trial in resolved.trials) == selected_ids
    assert resolved.manifest_configuration()["corpus_policy"] == "external_fidelity_pool"
    assert {judgment.trial_id for judgment in resolved.evaluation_package.judgments} == set(
        selected_ids
    )
    assert (
        resolved.evaluation_package.evaluation_package_id
        != prepared.evaluation_package.evaluation_package_id
    )
    assert resolved.manifest_configuration()["pool_source_sha256"] == (
        resolved.evaluation_package.evaluation_package_id
    )
    assert resolved.evaluation_package.provenance["release_profile_projection"] == {
        "profile_id": EXTERNAL_FIDELITY_TREC_2021_PROFILE,
        "source_evaluation_package_id": prepared.evaluation_package.evaluation_package_id,
        "selection_policy": "caller_supplied_external_fidelity_membership",
    }
    _validate_judgment_union_membership(
        SimpleNamespace(
            benchmark_profile=resolved.manifest_configuration(),
            trial_corpus_versions=tuple(
                (trial.trial_id, "sha256:" + "1" * 64) for trial in resolved.trials
            ),
            prepared_snapshot_id=resolved.prepared_snapshot_id,
            evaluation_package_id=resolved.evaluation_package_id,
        ),
        resolved.evaluation_package,
    )

    with pytest.raises(ValueError, match="caller-supplied pool membership"):
        resolve_benchmark_profile(
            prepared,
            profile_id=EXTERNAL_FIDELITY_TREC_2021_PROFILE,
            expected_pool_count=len(selected_ids),
            expected_pool_ids_sha256=receipt["pool_ids_sha256"],
            expected_pool_receipt_id=receipt["pool_receipt_id"],
        )
    with pytest.raises(ValueError, match="unknown trial ID"):
        inspect_external_fidelity_profile(
            prepared,
            profile_id=EXTERNAL_FIDELITY_TREC_2021_PROFILE,
            supplied_pool_ids=(*selected_ids, "NCT-NOT-IN-PREPARED-CORPUS"),
            expected_pool_count=len(selected_ids) + 1,
        )


def test_external_pool_source_verification_extracts_only_sorted_ids(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"
    source.write_text(
        '{"_id":"NCT-B","brief_title":"not copied"}\n{"_id":"NCT-A","brief_title":"not copied"}\n',
        encoding="utf-8",
    )
    source_sha256 = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    assert _verified_external_pool_ids(
        source,
        expected_byte_size=source.stat().st_size,
        expected_sha256=source_sha256,
        expected_count=2,
        expected_sorted_ids_sha256=ordered_trial_ids_sha256(("NCT-A", "NCT-B")),
    ) == ("NCT-A", "NCT-B")

    with pytest.raises(ValueError, match="checksum lock"):
        _verified_external_pool_ids(
            source,
            expected_byte_size=source.stat().st_size,
            expected_sha256="sha256:" + "0" * 64,
            expected_count=2,
            expected_sorted_ids_sha256=ordered_trial_ids_sha256(("NCT-A", "NCT-B")),
        )


def test_external_fidelity_public_cli_round_trip_uses_filtered_evaluation_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = load_release_fixture("trec-ct-2021")
    selected_ids = tuple(trial.trial_id for trial in prepared.trials[:-1])
    pool_file = tmp_path / "pool.json"
    pool_file.write_text(json.dumps(selected_ids), encoding="utf-8")
    production_profile = release_profiles.load_profile(EXTERNAL_FIDELITY_TREC_2021_PROFILE)
    fixture_profile = replace(production_profile, expected_pool_count=len(selected_ids))
    original_load_profile = release_profiles.load_profile

    def fixture_load_profile(profile_id: str):
        return (
            fixture_profile
            if profile_id == EXTERNAL_FIDELITY_TREC_2021_PROFILE
            else original_load_profile(profile_id)
        )

    monkeypatch.setattr(release_profiles, "load_profile", fixture_load_profile)
    monkeypatch.setattr(release_cli, "load_profile", fixture_load_profile)
    monkeypatch.setattr(release_cli, "load_prepared_benchmark", lambda _path: prepared)
    monkeypatch.setattr(
        release_cli,
        "release_source_identity",
        lambda _prepared, *, track, profile, evidence_scope: {
            "track": track,
            "profile": profile,
            "evidence_scope": evidence_scope,
        },
    )
    monkeypatch.setattr(
        public_patient_to_trial,
        "validate_release_source_identity",
        lambda value, **_kwargs: value,
    )
    protocol_id = "sha256:" + "9" * 64
    monkeypatch.setattr(
        release_cli,
        "require_frozen_effectiveness_protocol",
        lambda _profile, supplied, _role, **_kwargs: supplied if supplied == protocol_id else None,
    )
    monkeypatch.setattr(
        public_patient_to_trial,
        "require_frozen_effectiveness_protocol",
        lambda _profile, supplied, _role, **_kwargs: supplied if supplied == protocol_id else None,
    )
    release_identity = {
        "release_manifest_id": "sha256:" + "1" * 64,
        "public_tree_id": "sha256:" + "2" * 64,
        "package_artifact_id": "sha256:" + "3" * 64,
    }
    dependency_environment = {
        "schema_version": "1.0",
        "dependency_lock_sha256": "sha256:" + "4" * 64,
        "python_version": "3.13.7",
        "python_implementation": "CPython",
        "platform": "test",
        "machine": "test",
        "distributions": [{"name": "taim", "version": "0.1.0"}],
        "environment_id": "sha256:" + "5" * 64,
    }
    monkeypatch.setattr(release_cli, "_verified_release_identity", lambda _args: release_identity)
    monkeypatch.setattr(
        release_cli,
        "_verified_dependency_environment",
        lambda _args, *, system_id: dependency_environment,
    )
    monkeypatch.setattr(release_cli, "_git_state", lambda: ("a" * 40, False))

    assert (
        main(
            [
                "data",
                "pool",
                "inspect",
                "--track",
                "trec-ct-2021",
                "--profile",
                EXTERNAL_FIDELITY_TREC_2021_PROFILE,
                "--data-dir",
                str(tmp_path / "prepared"),
                "--pool-file",
                str(pool_file),
                "--protocol-approval-id",
                protocol_id,
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["pool_count"] == len(selected_ids)

    run_id = "external-bm25-round-trip"
    assert (
        main(
            [
                "patient-to-trial",
                "benchmark",
                "run",
                "--track",
                "trec-ct-2021",
                "--profile",
                EXTERNAL_FIDELITY_TREC_2021_PROFILE,
                "--system",
                "bm25",
                "--data-dir",
                str(tmp_path / "prepared"),
                "--clinical-as-of",
                "2021-04-27T00:00:00Z",
                "--pool-file",
                str(pool_file),
                "--pool-count",
                str(receipt["pool_count"]),
                "--pool-ids-sha256",
                receipt["pool_ids_sha256"],
                "--pool-receipt-id",
                receipt["pool_receipt_id"],
                "--protocol-approval-id",
                protocol_id,
                "--release-manifest",
                str(tmp_path / "release-manifest.json"),
                "--package-artifact",
                str(tmp_path / "package.whl"),
                "--dependency-lock",
                str(tmp_path / "uv.lock"),
                "--top-k",
                str(len(selected_ids)),
                "--run-id",
                run_id,
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    run_dir = tmp_path / run_id
    stored = load_public_patient_to_trial_run(run_dir)
    evaluation_package = load_local_evaluation_package(run_dir)
    assert report["evaluation_package_id"] == evaluation_package.evaluation_package_id
    assert stored.manifest.evaluation_package_id == evaluation_package.evaluation_package_id
    assert evaluation_package.evaluation_package_id != (
        prepared.evaluation_package.evaluation_package_id
    )
    assert evaluation_package.provenance["release_profile_projection"] == {
        "profile_id": EXTERNAL_FIDELITY_TREC_2021_PROFILE,
        "source_evaluation_package_id": prepared.evaluation_package.evaluation_package_id,
        "selection_policy": "caller_supplied_external_fidelity_membership",
    }
    assert main(["patient-to-trial", "run", "validate", "--run-dir", str(run_dir)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["evaluation_package_id"] == evaluation_package.evaluation_package_id
    assert main(["patient-to-trial", "run", "evaluate", "--run-dir", str(run_dir)]) == 0
    evaluation = json.loads(capsys.readouterr().out)
    assert evaluation["evaluation_package_id"] == evaluation_package.evaluation_package_id


def test_external_fidelity_profile_protocol_gate_matches_release_state() -> None:
    protocol_id = FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256
    if protocol_id is None:
        with pytest.raises(SchemaValidationError, match="not frozen by explicit human approval"):
            require_frozen_effectiveness_protocol(
                EXTERNAL_FIDELITY_TREC_2021_PROFILE,
                "sha256:" + "0" * 64,
                "external protocol",
            )
        return
    assert (
        require_frozen_effectiveness_protocol(
            EXTERNAL_FIDELITY_TREC_2021_PROFILE,
            protocol_id,
            "external protocol",
        )
        == protocol_id
    )
