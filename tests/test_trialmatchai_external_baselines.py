from __future__ import annotations

# ruff: noqa: S101
import json
from pathlib import Path

import pytest

from taim.adapters.trialmatchai import (
    TRIALMATCHAI_L4_COMMIT,
    TRIALMATCHAI_L4_REPOSITORY,
    TRIALMATCHAI_L4_SYSTEM_ID,
    TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID,
    TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID,
    TrialMatchAIL4TREC2022System,
    TrialMatchAIL4TREC2023System,
    normalize_ranked_trials,
)
from taim.pipeline_extensions import pipeline_extensions
from taim.release_profiles import (
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
)
from taim.release_result_bundle import _expected_pipeline_depth
from taim.release_support import load_release_support

TRIALMATCHAI_SYSTEMS = {
    TRIALMATCHAI_L4_SYSTEM_ID,
    TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID,
    TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID,
}


def test_the_trialmatchai_extension_declares_its_systems_and_its_external_fidelity_row() -> None:
    # The core external-fidelity test checks every System outside the pipeline extensions.
    rows = [
        row
        for row in load_release_support()["support_matrix"]
        if row["profile"] == EXTERNAL_FIDELITY_TREC_2021_PROFILE
    ]
    (extension,) = [
        extension
        for extension in pipeline_extensions()
        if extension.external_baseline and TRIALMATCHAI_L4_SYSTEM_ID in extension.system_tasks
    ]
    assert set(extension.system_tasks) == TRIALMATCHAI_SYSTEMS
    assert {row["system"] for row in rows} & TRIALMATCHAI_SYSTEMS == {TRIALMATCHAI_L4_SYSTEM_ID}


@pytest.mark.parametrize("system", sorted(TRIALMATCHAI_SYSTEMS))
def test_trialmatchai_result_bundle_pipeline_depth_is_post_eligibility(system: str) -> None:
    assert _expected_pipeline_depth(system) == "post_eligibility"


def test_trialmatchai_adaptations_use_existing_other_track_judgment_unions() -> None:
    rows = load_release_support()["support_matrix"]
    observed = {
        (row["track"], row["profile"], row["system"])
        for row in rows
        if row["track"] != "trec-ct-2021"
        and row["system"].startswith("trialmatchai-current-cuda-l4-")
    }
    assert observed == {
        (
            "trec-ct-2022",
            "trec-ct-2022-judgment-union",
            TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID,
        ),
        (
            "trec-ct-2023",
            "trec-ct-2023-judgment-union",
            TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID,
        ),
    }


def test_trialmatchai_normalization_is_direction_specific_and_deterministic() -> None:
    result = normalize_ranked_trials(
        {
            "RankedTrials": [
                {"TrialID": "NCT-B", "Score": 0.5},
                {"TrialID": "NCT-A", "Score": 0.5},
            ]
        },
        run_id="external-fixture",
        topic_id="patient-1",
        system_id=TRIALMATCHAI_L4_SYSTEM_ID,
    )
    assert [(row.trial_id, row.rank) for row in result.candidates] == [
        ("NCT-A", 1),
        ("NCT-B", 2),
    ]
    assert all(row.topic_id == "patient-1" for row in result.candidates)


def test_trialmatchai_track_adaptations_have_distinct_system_identities() -> None:
    assert TrialMatchAIL4TREC2022System.system_id == TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID
    assert TrialMatchAIL4TREC2023System.system_id == TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID
    assert {
        TRIALMATCHAI_L4_SYSTEM_ID,
        TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID,
        TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID,
    } == {
        "trialmatchai-current-cuda-l4-trec21-development-v3",
        "trialmatchai-current-cuda-l4-trec22-development-v3",
        "trialmatchai-current-cuda-l4-trec23-development-v3",
    }


def test_projected_trialmatchai_lock_is_metadata_only_and_license_bound() -> None:
    public_root = Path(__file__).parents[1]
    projected_tree = (public_root / "external").is_dir()
    root = public_root
    if not projected_tree:
        root = Path(__file__).parents[3]
    trialmatchai = json.loads((root / "external" / "trialmatchai-lock.json").read_text())
    assert trialmatchai["current"]["license"]["expression"] == "MIT"
    assert trialmatchai["paper"]["license"]["expression"] == "MIT"
    assert trialmatchai["current"]["commit"] == TRIALMATCHAI_L4_COMMIT
    assert trialmatchai["current"]["repository"] == TRIALMATCHAI_L4_REPOSITORY
    if projected_tree:
        assert not any((root / "external").glob("*/"))


def test_public_trialmatchai_adapter_surface_is_selected() -> None:
    candidate_root = Path(__file__).parents[1]
    if (candidate_root / "src" / "taim").is_dir():
        root = candidate_root
    else:
        root = Path(__file__).parents[3]
    if (root / "release" / "public").is_dir():
        trialmatchai = (root / "release" / "public" / "trialmatchai-adapter.py").read_text()
    else:
        trialmatchai = (root / "src" / "taim" / "adapters" / "trialmatchai.py").read_text()
    for forbidden in (
        "class PaperTrialGPTSystem",
        "class TrialGPTCodexFailureAwareSystem",
        "class TrialGPTCodexSystem",
        "class TrialGPTFakeSystem",
        "class TrialMatchAISystem",
    ):
        assert forbidden not in trialmatchai
