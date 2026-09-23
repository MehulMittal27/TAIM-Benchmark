from __future__ import annotations

# ruff: noqa: S101
import json
from pathlib import Path

import pytest

from taim.release_cli import main


def test_trial_to_patient_fixture_quick_start(tmp_path: Path, capsys) -> None:
    run_directory = tmp_path / "runs" / "quick-start-trial-to-patient"
    assert (
        main(
            [
                "trial-to-patient",
                "fixture",
                "run",
                "--run-id",
                "quick-start-trial-to-patient",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["trial-to-patient", "run", "validate", "--run-dir", str(run_directory)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["task"] == "trial_to_patient"
    assert report["profile"] == "synthetic-trial-to-patient"
    assert main(["trial-to-patient", "run", "evaluate", "--run-dir", str(run_directory)]) == 0
    metrics = json.loads(capsys.readouterr().out)
    for field in (
        "release_version",
        "track",
        "task",
        "profile",
        "system",
        "prepared_snapshot_id",
        "task_input_id",
        "system_input_id",
        "evaluation_package_id",
    ):
        assert metrics[field] == report[field]


def test_reverse_evaluate_rejects_a_run_without_release_identity(tmp_path: Path, capsys) -> None:
    run_directory = tmp_path / "runs" / "missing-release-identity"
    assert (
        main(
            [
                "trial-to-patient",
                "fixture",
                "run",
                "--run-id",
                "missing-release-identity",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["configuration"]["release_support"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit):
        main(["trial-to-patient", "run", "evaluate", "--run-dir", str(run_directory)])


def test_direction_specific_run_validation_rejects_extra_and_nested_files(
    tmp_path: Path, capsys
) -> None:
    forward = tmp_path / "runs" / "closed-forward"
    reverse = tmp_path / "runs" / "closed-reverse"
    for task, run_id, top_k in (
        ("patient-to-trial", "closed-forward", "8"),
        ("trial-to-patient", "closed-reverse", "3"),
    ):
        assert (
            main(
                [
                    task,
                    "fixture",
                    "run",
                    "--run-id",
                    run_id,
                    "--output-dir",
                    str(tmp_path / "runs"),
                    "--top-k",
                    top_k,
                ]
            )
            == 0
        )
        capsys.readouterr()
    (reverse / "candidates.jsonl").write_bytes((forward / "candidates.jsonl").read_bytes())
    with pytest.raises(SystemExit):
        main(["trial-to-patient", "run", "validate", "--run-dir", str(reverse)])
    (reverse / "candidates.jsonl").unlink()
    (forward / "nested").mkdir()
    (forward / "nested" / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["patient-to-trial", "run", "validate", "--run-dir", str(forward)])
