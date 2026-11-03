# ruff: noqa: S101
"""The SIGIR 2016 binding of TrialMatchAI v0.9.1: its own identity on the tag's TREC path."""

from taim.pipeline_extensions import extension_for
from taim.pipelines.trialmatchai_v091.sigir16 import (
    TRIALMATCHAI_UPSTREAM_SIGIR_2016_SYSTEM_ID,
    TrialMatchAIUpstreamSIGIR2016System,
)
from taim.pipelines.trialmatchai_v091.sigir16_extension import EXTENSION
from taim.pipelines.trialmatchai_v091.system import TrialMatchAIUpstreamV091Base


def test_the_sigir16_binding_differs_from_the_base_only_in_identity_and_track() -> None:
    system = TRIALMATCHAI_UPSTREAM_SIGIR_2016_SYSTEM_ID
    assert system == "trialmatchai-upstream-v091-cuda-sigir16-development-v1"
    assert EXTENSION.system_tasks == {system: "patient_to_trial"}
    assert EXTENSION.external_baseline
    assert extension_for(system) is EXTENSION
    assert TrialMatchAIUpstreamSIGIR2016System.__bases__ == (TrialMatchAIUpstreamV091Base,)
    own = {name for name in vars(TrialMatchAIUpstreamSIGIR2016System) if not name.startswith("__")}
    assert own == {"system_id", "trec_track_key", "trec_id_prefix"}
    assert (
        TrialMatchAIUpstreamSIGIR2016System.trec_track_key,
        TrialMatchAIUpstreamSIGIR2016System.trec_id_prefix,
    ) == ("sigir", "sigir-2014")
    assert TrialMatchAIUpstreamSIGIR2016System().system_id == system
