# ruff: noqa: S101
"""The TREC 2023 binding of TrialMatchAI v0.9.1: its own identity on the tag's TREC path."""

from taim.pipeline_extensions import extension_for
from taim.pipelines.trialmatchai_v091.system import TrialMatchAIUpstreamV091Base
from taim.pipelines.trialmatchai_v091.trec23 import (
    TRIALMATCHAI_UPSTREAM_TREC_2023_SYSTEM_ID,
    TrialMatchAIUpstreamTREC2023System,
)
from taim.pipelines.trialmatchai_v091.trec23_extension import EXTENSION


def test_the_trec23_binding_differs_from_the_base_only_in_identity_and_track() -> None:
    system = TRIALMATCHAI_UPSTREAM_TREC_2023_SYSTEM_ID
    assert system == "trialmatchai-upstream-v091-cuda-trec23-development-v1"
    assert EXTENSION.system_tasks == {system: "patient_to_trial"}
    assert EXTENSION.external_baseline
    assert extension_for(system) is EXTENSION
    assert TrialMatchAIUpstreamTREC2023System.__bases__ == (TrialMatchAIUpstreamV091Base,)
    own = {name for name in vars(TrialMatchAIUpstreamTREC2023System) if not name.startswith("__")}
    assert own == {"system_id", "trec_track_key", "trec_id_prefix"}
    assert (
        TrialMatchAIUpstreamTREC2023System.trec_track_key,
        TrialMatchAIUpstreamTREC2023System.trec_id_prefix,
    ) == ("23", "trec-2023")
    assert TrialMatchAIUpstreamTREC2023System().system_id == system
