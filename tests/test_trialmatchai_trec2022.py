# ruff: noqa: S101
"""Synthetic conformance for one selectable TrialMatchAI binding."""

from taim.adapters.trialmatchai import normalize_ranked_trials
from taim.pipeline_extensions.trialmatchai_trec2022 import (
    EXTENSION,
    TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID,
    TrialMatchAIL4TREC2022System,
)


def test_trec2022_binding_and_normalization() -> None:
    system = TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID
    assert set(EXTENSION.system_tasks) == {system}
    assert TrialMatchAIL4TREC2022System().system_id == system
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
