# ruff: noqa: S101
"""The TREC 2022 binding of TrialMatchAI v0.9.1: its own identity on the tag's TREC path."""

from taim.pipeline_extensions import extension_for
from taim.pipelines.trialmatchai_v091.system import TrialMatchAIUpstreamV091Base
from taim.pipelines.trialmatchai_v091.trec22 import (
    TRIALMATCHAI_UPSTREAM_TREC_2022_SYSTEM_ID,
    TrialMatchAIUpstreamTREC2022System,
)
from taim.pipelines.trialmatchai_v091.trec22_extension import EXTENSION


def test_the_trec22_binding_differs_from_the_base_only_in_identity_and_track() -> None:
    system = TRIALMATCHAI_UPSTREAM_TREC_2022_SYSTEM_ID
    assert system == "trialmatchai-upstream-v091-cuda-trec22-development-v1"
    assert EXTENSION.system_tasks == {system: "patient_to_trial"}
    assert EXTENSION.external_baseline
    assert extension_for(system) is EXTENSION
    assert TrialMatchAIUpstreamTREC2022System.__bases__ == (TrialMatchAIUpstreamV091Base,)
    own = {name for name in vars(TrialMatchAIUpstreamTREC2022System) if not name.startswith("__")}
    assert own == {"system_id", "trec_track_key", "trec_id_prefix"}
    assert (
        TrialMatchAIUpstreamTREC2022System.trec_track_key,
        TrialMatchAIUpstreamTREC2022System.trec_id_prefix,
    ) == ("22", "trec-2022")
    assert TrialMatchAIUpstreamTREC2022System().system_id == system
