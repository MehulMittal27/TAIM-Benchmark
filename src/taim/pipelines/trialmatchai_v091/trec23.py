"""Frozen TREC 2023 identity of the TrialMatchAI v0.9.1 System."""

from taim.pipelines.trialmatchai_v091.system import TrialMatchAIUpstreamV091Base

TRIALMATCHAI_UPSTREAM_TREC_2023_SYSTEM_ID = "trialmatchai-upstream-v091-cuda-trec23-development-v1"


class TrialMatchAIUpstreamTREC2023System(TrialMatchAIUpstreamV091Base):
    """Upstream TrialMatchAI v0.9.1 as it ships for TREC, under its TREC 2023 identity."""

    system_id = TRIALMATCHAI_UPSTREAM_TREC_2023_SYSTEM_ID
    trec_track_key = "23"
    trec_id_prefix = "trec-2023"
