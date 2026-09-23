from __future__ import annotations

# ruff: noqa: S101
import json
from pathlib import Path

from taim.release_cli import main


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
                "--top-k",
                "8",
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
