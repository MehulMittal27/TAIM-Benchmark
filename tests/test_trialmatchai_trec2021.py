# ruff: noqa: S101
"""Synthetic conformance for one selectable TrialMatchAI binding."""

from taim.adapters.trialmatchai import normalize_ranked_trials
from taim.pipeline_extensions.trialmatchai_trec2021 import (
    EXTENSION,
    TRIALMATCHAI_L4_SYSTEM_ID,
    TrialMatchAIL4System,
)


def test_trec2021_binding_and_normalization() -> None:
    system = TRIALMATCHAI_L4_SYSTEM_ID
    assert set(EXTENSION.system_tasks) == {system}
    assert TrialMatchAIL4System().system_id == system
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
    assert [(row.trial_id, row.rank, row.system_id) for row in result.candidates] == [
        ("trial-a", 1, system),
        ("trial-b", 2, system),
    ]
