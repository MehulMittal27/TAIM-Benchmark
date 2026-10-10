"""The external-fidelity Profile's public checks, which do not depend on any external System."""

from __future__ import annotations

# ruff: noqa: S101
import json
from dataclasses import replace
from pathlib import Path

import pytest

import taim.public_patient_to_trial as public_patient_to_trial
import taim.release_cli as release_cli
import taim.release_profiles as release_profiles
from taim import release_producer
from taim.public_patient_to_trial import (
    load_local_evaluation_package,
    load_public_patient_to_trial_run,
)
from taim.release_cli import main
from taim.release_fixture import load_release_fixture
from taim.release_profiles import (
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
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
    monkeypatch.setattr(
        release_producer, "verified_release_identity", lambda _args: release_identity
    )
    monkeypatch.setattr(
        release_producer,
        "verified_dependency_environment",
        lambda _args, *, system_id: dependency_environment,
    )
    monkeypatch.setattr(release_producer, "git_state", lambda: ("a" * 40, False))

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
