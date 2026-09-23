from __future__ import annotations

# ruff: noqa: S101
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from taim.contracts import content_sha256
from taim.file_hash import sha256_file
from taim.release_cli import main
from taim.submissions import load_task, validate_submission

EXAMPLE = Path(__file__).parents[1] / "examples/external-pipeline/pipeline.py"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _export(direction: str, root: Path, *extra: str) -> tuple[Path, Path]:
    task, evaluator = root / "task", root / "evaluator"
    command = [direction, "submission", "export", "--fixture", *extra]
    assert main([*command, "--output", str(task), "--evaluator-output", str(evaluator)]) == 0
    return task, evaluator


def _run_example(task: Path, predictions: Path, *, cwd: Path) -> None:
    # Only the input file crosses into the external pipeline's process.
    subprocess.run(
        [sys.executable, "-I", str(EXAMPLE), str(task / "input.json"), str(predictions)],
        check=True,
        cwd=cwd,
    )


def _import(direction: str, task: Path, evaluator: Path, predictions: Path, output: Path) -> int:
    return main(
        [
            direction,
            "submission",
            "import",
            "--task-dir",
            str(task),
            "--evaluator-dir",
            str(evaluator),
            "--ranking",
            str(predictions / "ranking.jsonl"),
            "--pipeline",
            str(predictions / "pipeline.json"),
            "--output",
            str(output),
        ]
    )


@pytest.mark.parametrize("direction", ["patient-to-trial", "trial-to-patient"])
def test_external_pipeline_runs_separately_and_recomputes_metrics(
    tmp_path: Path, direction: str
) -> None:
    task, evaluator = _export(direction, tmp_path)
    predictions, submission = tmp_path / "predictions", tmp_path / "submission"
    assert {path.name for path in task.iterdir()} == {"input.json"}
    payload = json.loads((task / "input.json").read_text())
    assert "judgments" not in json.dumps(payload["input"])
    _run_example(task, predictions, cwd=tmp_path)
    assert _import(direction, task, evaluator, predictions, submission) == 0
    task_name = direction.replace("-", "_")
    result = validate_submission(submission, task_dir=task, evaluator_dir=evaluator, task=task_name)
    assert result["paper_matrix_admitted"] is False
    assert result["system_id"] == "external:token-overlap"
    changed = dict(result, system_id="external:another")
    changed["submission_id"] = content_sha256(
        {key: value for key, value in changed.items() if key != "submission_id"}
    )
    _write(submission / "submission.json", changed)
    with pytest.raises(ValueError, match="identities differ"):
        validate_submission(submission, task_dir=task, evaluator_dir=evaluator, task=task_name)
    _write(submission / "submission.json", result)
    _write(submission / "metrics.json", {"fabricated": 1})
    result["artifacts"]["metrics.json"] = sha256_file(submission / "metrics.json")
    result["submission_id"] = content_sha256(
        {key: value for key, value in result.items() if key != "submission_id"}
    )
    _write(submission / "submission.json", result)
    with pytest.raises(ValueError, match="metrics do not reproduce"):
        validate_submission(submission, task_dir=task, evaluator_dir=evaluator, task=task_name)


@pytest.mark.parametrize("change", ["incomplete", "duplicate", "wrong-task", "nan"])
def test_external_submission_rejects_invalid_rankings(tmp_path: Path, change: str) -> None:
    task, evaluator = _export("patient-to-trial", tmp_path, "--top-k", "2")
    predictions = tmp_path / "predictions"
    _run_example(task, predictions, cwd=tmp_path)
    path = predictions / "ranking.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "incomplete":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "wrong-task":
        rows[0]["task"] = "trial_to_patient"
    else:
        rows[0]["score"] = float("nan")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        _import("patient-to-trial", task, evaluator, predictions, tmp_path / "submission")
    assert error.value.code == 2
    assert not (tmp_path / "submission").exists()


@pytest.mark.parametrize(
    ("track", "profile"),
    [("trec-ct-2022", "confirmation-full"), ("trec-ct-2021", "trec-ct-2021-judgment-union")],
)
def test_protocol_bound_submission_requires_protocol_before_opening_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, track: str, profile: str
) -> None:
    import taim.submission_cli as cli

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("protected input was opened before protocol validation")

    monkeypatch.setattr(cli, "load_prepared_benchmark", forbidden)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "patient-to-trial",
                "submission",
                "export",
                "--track",
                track,
                "--data-dir",
                str(tmp_path),
                "--profile",
                profile,
                "--clinical-as-of",
                "2022-01-01T00:00:00Z",
                "--output",
                str(tmp_path / "task"),
                "--evaluator-output",
                str(tmp_path / "evaluator"),
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "task").exists()


def test_submission_rejects_rehashed_track_profile_and_release_claims(tmp_path: Path) -> None:
    task, evaluator = _export("patient-to-trial", tmp_path)
    original = json.loads((task / "input.json").read_text())
    for profile in ("description", "trec-ct-2021-judgment-union", "official-full"):
        payload = dict(original, profile=profile, release_identity={"invented": "identity"})
        payload["export_id"] = content_sha256(
            {key: value for key, value in payload.items() if key != "export_id"}
        )
        _write(task / "input.json", payload)
        with pytest.raises(ValueError):
            load_task(task, evaluator_dir=evaluator, task="patient_to_trial")


def test_submission_rejects_changed_source_recipe_even_after_rehash(tmp_path: Path) -> None:
    task, evaluator = _export("patient-to-trial", tmp_path)
    payload = json.loads((task / "input.json").read_text())
    payload["release_source"]["source_recipe_id"] = "older-custom-recipe"
    payload["export_id"] = content_sha256(
        {key: value for key, value in payload.items() if key != "export_id"}
    )
    _write(task / "input.json", payload)
    with pytest.raises(ValueError):
        load_task(task, evaluator_dir=evaluator, task="patient_to_trial")


def test_the_evaluator_package_never_sits_beside_the_input(tmp_path: Path) -> None:
    task, evaluator = _export("patient-to-trial", tmp_path)
    shutil.copy(evaluator / "evaluation-package.json", task / "evaluation-package.json")
    with pytest.raises(ValueError, match=r"must hold only input\.json"):
        load_task(task, evaluator_dir=evaluator, task="patient_to_trial")
    (task / "evaluation-package.json").unlink()
    with pytest.raises(ValueError, match=r"must hold only evaluation-package\.json"):
        load_task(task, evaluator_dir=task, task="patient_to_trial")
    with pytest.raises(SystemExit):
        main(
            [
                "patient-to-trial",
                "submission",
                "export",
                "--fixture",
                "--output",
                str(tmp_path / "again"),
                "--evaluator-output",
                str(evaluator),
            ]
        )
    assert not (tmp_path / "again").exists()


def _reference_rows(direction: str, run_dir: Path, task: Path) -> list[dict[str, object]]:
    payload = json.loads((task / "input.json").read_text())
    name = "candidates.jsonl" if direction == "patient-to-trial" else "patient-candidates.jsonl"
    query, document = (
        ("topic_id", "trial_id") if direction == "patient-to-trial" else ("trial_id", "patient_id")
    )
    rows = []
    for line in (run_dir / name).read_text().splitlines():
        row = json.loads(line)
        rows.append(
            {
                "task": payload["task"],
                "task_input_id": payload["input"]["task_input_id"],
                "system_id": "external:bm25-replay",
                query: row[query],
                document: row[document],
                "rank": row["rank"],
                "score": row["score"],
            }
        )
    return rows


@pytest.mark.parametrize(
    ("direction", "top_k"), [("patient-to-trial", "1000"), ("trial-to-patient", "20")]
)
def test_a_submission_scores_exactly_as_the_same_catalogue_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], direction: str, top_k: str
) -> None:
    """The comparison the guide promises: same input identities, same numbers."""

    task, evaluator = _export(direction, tmp_path, "--top-k", top_k)
    run = [direction, "fixture", "run", "--run-id", "reference", "--top-k", top_k]
    assert main([*run, "--output-dir", str(tmp_path / "runs")]) == 0
    capsys.readouterr()
    assert main([direction, "run", "evaluate", "--run-dir", str(tmp_path / "runs/reference")]) == 0
    reference = json.loads(capsys.readouterr().out)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    rows = _reference_rows(direction, tmp_path / "runs/reference", task)
    (predictions / "ranking.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _write(
        predictions / "pipeline.json",
        {
            "system_id": "external:bm25-replay",
            "source_revision": "replay",
            "source_sha256": "sha256:" + "0" * 64,
            "dependency_lock_sha256": "sha256:" + "0" * 64,
            "configuration": {},
        },
    )
    assert _import(direction, task, evaluator, predictions, tmp_path / "submission") == 0
    manifest = json.loads((tmp_path / "submission/submission.json").read_text())
    metrics = json.loads((tmp_path / "submission/metrics.json").read_text())
    for key in ("task_input_id", "evaluation_package_id"):
        assert manifest[key] == reference[key]
    if direction == "patient-to-trial":
        assert metrics == reference["metrics_by_cutoff"]
        return
    mean = reference["aggregate"]["mean"]
    assert [int(cutoff) for cutoff in metrics] == reference["cutoffs"]
    for cutoff, card in metrics.items():
        recall = card["aggregate"]["recall"]
        assert card["aggregate"][f"ndcg_at_{cutoff}"] == mean[f"ndcg_at_{cutoff}"]
        assert recall[f"eligible_at_{cutoff}"] == mean[f"eligible_recall_at_{cutoff}"]
        assert (
            recall[f"relevant_or_eligible_at_{cutoff}"]
            == mean[f"relevant_or_eligible_recall_at_{cutoff}"]
        )
