# ruff: noqa: S101
"""Synthetic conformance shared by the selected TrialMatchAI release units."""

from taim.adapters.trialmatchai import normalize_ranked_trials
from taim.pipeline_extensions import pipeline_extensions
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
