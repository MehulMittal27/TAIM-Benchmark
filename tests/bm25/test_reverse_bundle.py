from __future__ import annotations

# ruff: noqa: S101
from pathlib import Path

from taim.release_cli import main
from taim.release_result_bundle import (
    build_release_result_bundle,
    validate_release_result_bundle,
)

from ..bundle_support import (
    _clean_test_producer as _clean_test_producer,
)
from ..bundle_support import (
    _declaration,
    _release_identity,
)


def test_reverse_run_projects_to_a_separate_safe_bundle(tmp_path: Path, capsys) -> None:
    run_directory = tmp_path / "runs" / "reverse-bundle-source"
    assert (
        main(
            [
                "trial-to-patient",
                "fixture",
                "run",
                "--run-id",
                "reverse-bundle-source",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    bundle = build_release_result_bundle(
        run_directory,
        tmp_path / "reverse-bundle",
        release_identity=_release_identity(),
        declaration=_declaration(),
    )
    report = validate_release_result_bundle(bundle.directory)
    assert report["bundle_id"] == bundle.bundle_id
    assert report["task"] == "trial_to_patient"
    assert report["system"] == "bm25-trial-to-patient"


def test_bundle_rejects_mixed_direction_ranking(tmp_path: Path, capsys) -> None:
    forward = tmp_path / "runs" / "forward"
    reverse = tmp_path / "runs" / "reverse"
    for task, run_id in (("patient-to-trial", "forward"), ("trial-to-patient", "reverse")):
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
                    "3",
                ]
            )
            == 0
        )
        capsys.readouterr()
    (forward / "candidates.jsonl").write_bytes((reverse / "patient-candidates.jsonl").read_bytes())
    try:
        build_release_result_bundle(
            forward,
            tmp_path / "mixed-bundle",
            release_identity=_release_identity(),
            declaration=_declaration(),
        )
    except ValueError as exc:
        assert any(marker in str(exc) for marker in ("Candidate", "hash", "direction"))
    else:
        raise AssertionError("mixed-direction ranking was accepted")
