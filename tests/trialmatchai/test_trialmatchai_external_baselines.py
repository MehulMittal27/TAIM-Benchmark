# ruff: noqa: S101
"""Synthetic conformance shared by the selected TrialMatchAI release units."""

import json
import shutil
from pathlib import Path

import pytest

from taim.pipeline_extensions import pipeline_extensions
from taim.pipelines.trialmatchai.corpus import filter_trialmatchai_corpus, normalize_ranked_trials
from taim.release_cli import main
from taim.release_result_bundle import _expected_pipeline_depth
from taim.release_support import load_release_support


def _systems():
    return {
        system
        for extension in pipeline_extensions()
        for system in extension.system_tasks
        if system.startswith("trialmatchai-current-cuda-l4-")
    }


def test_selected_units_match_the_support_catalogue() -> None:
    systems = _systems()
    assert systems
    rows = load_release_support()["support_matrix"]
    assert {row["system"] for row in rows if row["system"] in systems} == systems
    assert all(_expected_pipeline_depth(system) == "post_eligibility" for system in systems)


def test_normalization_is_direction_specific_and_deterministic() -> None:
    for system in sorted(_systems()):
        result = normalize_ranked_trials(
            {
                "RankedTrials": [
                    {"TrialID": "trial-b", "Score": 0.5},
                    {"TrialID": "trial-a", "Score": 0.5},
                ]
            },
            run_id="synthetic",
            topic_id="patient",
            system_id=system,
        )
        assert [(row.trial_id, row.rank) for row in result.candidates] == [
            ("trial-a", 1),
            ("trial-b", 2),
        ]
        assert all(
            row.system_id == system and row.topic_id == "patient" for row in result.candidates
        )


def _prepared(tmp_path: Path) -> tuple[Path, Path, Path]:
    snapshot = tmp_path / "trials.jsonl"
    snapshot.write_text(
        '{"trial_id":"NCT00000002"}\n\n{"trial_id":"NCT00000001"}\n', encoding="utf-8"
    )
    trials = tmp_path / "processed_trials"
    trials.mkdir()
    for trial_id in ("NCT00000001", "NCT00000002", "NCT00000003"):
        (trials / f"{trial_id}.json").write_text(f'{{"nct_id":"{trial_id}"}}', encoding="utf-8")
    criteria = tmp_path / "processed_criteria"
    for trial_id in ("NCT00000001", "NCT00000003"):
        (criteria / trial_id).mkdir(parents=True)
        (criteria / trial_id / "c1.json").write_text('{"criterion":"invented"}', encoding="utf-8")
    return snapshot, trials, criteria


def _filter_command(snapshot: Path, trials: Path, criteria: Path, output: Path) -> list[str]:
    return [
        "patient-to-trial",
        "trialmatchai",
        "filter-corpus",
        "--snapshot-trials",
        str(snapshot),
        "--source-trials",
        str(trials),
        "--source-criteria",
        str(criteria),
        "--output",
        str(output),
    ]


def test_filter_corpus_keeps_exactly_the_snapshot_trials_and_records_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot, trials, criteria = _prepared(tmp_path)
    output = tmp_path / "corpus"

    assert main(_filter_command(snapshot, trials, criteria, output)) == 0

    assert sorted(path.name for path in (output / "processed_trials").iterdir()) == [
        "NCT00000001.json",
        "NCT00000002.json",
    ]
    assert [path.name for path in (output / "processed_criteria").iterdir()] == ["NCT00000001"]
    assert (output / "processed_trials" / "NCT00000001.json").samefile(trials / "NCT00000001.json")
    manifest = {
        "source_snapshot_trials": str(snapshot.resolve()),
        "source_processed_trials": str(trials.resolve()),
        "source_processed_criteria": str(criteria.resolve()),
        "output_processed_trials": str(output.resolve() / "processed_trials"),
        "output_processed_criteria": str(output.resolve() / "processed_criteria"),
        "trial_count": 2,
        "criteria_count": 1,
        "criteria_trial_count": 1,
        "criteria_missing_trial_count": 1,
        "criteria_missing_trial_ids": ["NCT00000002"],
        "selection": "exact trial-ID membership from the frozen TAIM Snapshot",
        "embeddings": "reused from TrialMatchAI prepared JSON records",
        "link_modes": {"hardlink": 3, "copy": 0, "existing": 0},
    }
    # The bytes the private script wrote, which the published row's corpus digest was taken over.
    assert (output / "filter-manifest.json").read_text(encoding="utf-8") == (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    assert json.loads(capsys.readouterr().out) == manifest


def test_filter_corpus_completes_an_existing_output_and_refuses_different_bytes(
    tmp_path: Path,
) -> None:
    snapshot, trials, criteria = _prepared(tmp_path)
    output = tmp_path / "corpus"
    (output / "processed_trials").mkdir(parents=True)
    shutil.copy2(trials / "NCT00000001.json", output / "processed_trials")

    with pytest.raises(ValueError, match="output already exists"):
        filter_trialmatchai_corpus(
            snapshot_trials=snapshot, source_trials=trials, source_criteria=criteria, output=output
        )
    manifest = filter_trialmatchai_corpus(
        snapshot_trials=snapshot,
        source_trials=trials,
        source_criteria=criteria,
        output=output,
        allow_existing=True,
    )
    assert manifest["link_modes"] == {"hardlink": 2, "copy": 0, "existing": 1}

    (output / "processed_trials" / "NCT00000002.json").unlink()
    (output / "processed_trials" / "NCT00000002.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="different contents"):
        filter_trialmatchai_corpus(
            snapshot_trials=snapshot,
            source_trials=trials,
            source_criteria=criteria,
            output=output,
            allow_existing=True,
        )


def test_filter_corpus_refuses_a_snapshot_trial_the_prepared_corpus_lacks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot, trials, criteria = _prepared(tmp_path)
    (trials / "NCT00000002.json").unlink()

    with pytest.raises(SystemExit) as refused:
        main(_filter_command(snapshot, trials, criteria, tmp_path / "corpus"))
    assert refused.value.code == 2
    assert "TrialMatchAI trial is missing: NCT00000002" in capsys.readouterr().err
