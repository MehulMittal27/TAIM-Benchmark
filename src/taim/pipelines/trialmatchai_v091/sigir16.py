"""Frozen SIGIR 2016 identity of the TrialMatchAI v0.9.1 System."""

from taim.pipelines.trialmatchai_v091.system import TrialMatchAIUpstreamV091Base

TRIALMATCHAI_UPSTREAM_SIGIR_2016_SYSTEM_ID = (
    "trialmatchai-upstream-v091-cuda-sigir16-development-v1"
)


class TrialMatchAIUpstreamSIGIR2016System(TrialMatchAIUpstreamV091Base):
    """Upstream TrialMatchAI v0.9.1 as it ships for TREC, under its SIGIR 2016 identity."""

    system_id = TRIALMATCHAI_UPSTREAM_SIGIR_2016_SYSTEM_ID
    trec_track_key = "sigir"
    #: The tag names SIGIR's topics by their TREC CDS 2014 and 2015 origin
    #: (``_ID_PREFIX`` in ``trec/corpus.py``).
    trec_id_prefix = "sigir-2014"
