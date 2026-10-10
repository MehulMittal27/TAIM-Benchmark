"""The deterministic lexical retrieval release unit."""

from __future__ import annotations

import time
from typing import Any

from taim.patient_to_trial import (
    PatientToTrialRunRequest,
    PatientToTrialRunResult,
    StrictSystemOptions,
    required_option,
)
from taim.schemas import PIPELINE_DEPTH_RETRIEVAL, PRIMARY_RANKING_CANDIDATES
from taim.snapshot import CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT


class ReleaseBM25System(StrictSystemOptions):
    system_id = "bm25"
    option_names = frozenset({"b", "k1"})
    required_capabilities = frozenset(
        {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
    )
    optional_capabilities: frozenset[str] = frozenset()

    def run(self, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
        from taim.baselines.bm25 import BM25Retriever, bm25_ranking_configuration

        started_at = time.perf_counter()
        retriever = BM25Retriever(
            request.snapshot.trials,
            k1=required_option(request, "k1"),  # type: ignore[arg-type]
            b=required_option(request, "b"),  # type: ignore[arg-type]
        )
        candidates = retriever.retrieve_many(
            request.snapshot.topics,
            run_id=request.run_id,
            system_id=self.system_id,
            top_k=request.top_k,
        )
        runtime_seconds = time.perf_counter() - started_at
        ranking = bm25_ranking_configuration(
            k1=retriever.k1,
            b=retriever.b,
            requested_top_k=request.top_k,
            document_count=len(retriever.documents),
        )
        configuration: dict[str, Any] = {
            **ranking,
            "model_identity": {"kind": "none", "reason": "deterministic_bm25"},
            "index_identity": {
                "implementation": "taim-okapi-bm25-v1",
                "persistent": False,
                "corpus_statistics": retriever.corpus_statistics,
            },
            "metric_cutoff": request.metric_cutoff,
            "corpus_statistics": retriever.corpus_statistics,
            "runtime_seconds": runtime_seconds,
        }
        return PatientToTrialRunResult(
            candidates=tuple(candidates),
            configuration=configuration,
            runtime_seconds=runtime_seconds,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth=PIPELINE_DEPTH_RETRIEVAL,
        )
