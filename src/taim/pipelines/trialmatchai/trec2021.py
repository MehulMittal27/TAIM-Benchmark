"""Frozen track-specific TrialMatchAI System identity."""

from taim.pipelines.trialmatchai.system import TrialMatchAIL4Base

TRIALMATCHAI_L4_SYSTEM_ID = "trialmatchai-current-cuda-l4-trec21-development-v4"


class TrialMatchAIL4System(TrialMatchAIL4Base):
    """Frozen method under its track-specific identity."""

    system_id = TRIALMATCHAI_L4_SYSTEM_ID
