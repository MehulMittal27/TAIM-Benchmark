from __future__ import annotations

# ruff: noqa: S101
import json
from dataclasses import replace
from pathlib import Path

import pytest

from taim.entity_versions import PatientEvidenceProfile
from taim.patient_to_trial import PatientToTrialTaskInput
from taim.public_patient_to_trial import load_public_patient_to_trial_run
from taim.release_cli import main
from taim.release_fixture import load_release_fixture
from taim.release_profiles import resolve_benchmark_profile


def test_patient_to_trial_fixture_quick_start(tmp_path: Path, capsys) -> None:
    run_directory = tmp_path / "runs" / "quick-start-patient-to-trial"
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "quick-start-patient-to-trial",
                "--output-dir",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["patient-to-trial", "run", "validate", "--run-dir", str(run_directory)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["system"] == "bm25"
    assert report["profile"] == "synthetic-patient-to-trial"
    assert report["task_input_id"] != report["system_input_id"]

    assert main(["patient-to-trial", "run", "evaluate", "--run-dir", str(run_directory)]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["profile"] == "synthetic-patient-to-trial"
    assert "relevant_or_eligible_precision_at_5" in (metrics["metrics_by_cutoff"]["5"]["aggregate"])

    stored = load_public_patient_to_trial_run(run_directory)
    manifest = stored.manifest
    record = manifest.configuration["system_input"]
    assert record["trial_corpus_encoding"] == "lineage-v1"
    assert record["trial_corpus_count"] == len(manifest.trial_corpus_versions)
    assert all(
        set(row) == {"trial_id", "trial_version_id"} for row in record["trial_corpus_versions"]
    )
    prepared = load_release_fixture(manifest.benchmark_lineage)
    source = PatientToTrialTaskInput.from_benchmark(
        resolve_benchmark_profile(prepared, profile_id=manifest.benchmark_profile["profile_id"]),
        clinical_as_of=manifest.clinical_as_of,
        patient_evidence_profile=PatientEvidenceProfile.from_dict(
            manifest.patient_evidence_profile
        ),
    )
    assert load_public_patient_to_trial_run(run_directory, source_snapshot=source) == stored
    with pytest.raises(ValueError, match="matching prepared source"):
        manifest.resolve_task_input()
    with pytest.raises(ValueError, match="does not reproduce"):
        load_public_patient_to_trial_run(
            run_directory,
            source_snapshot=replace(source, prepared_snapshot_id="sha256:" + "f" * 64),
        )
