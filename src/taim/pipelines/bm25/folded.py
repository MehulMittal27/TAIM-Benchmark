"""The checksum-bound lexical query transformation release unit."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace

from taim.patient_to_trial import (
    PatientToTrialRunRequest,
    PatientToTrialRunResult,
    StrictSystemOptions,
    required_option,
)
from taim.schemas import PIPELINE_DEPTH_RETRIEVAL, PRIMARY_RANKING_CANDIDATES
from taim.snapshot import CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT


class ReleaseFoldedBM25System(StrictSystemOptions):
    """BM25 over a checksum-bound, locally derived folded-query bundle."""

    system_id = "bm25-folded"
    option_names = frozenset({"b", "folded_query_bundle", "k1"})
    required_capabilities = frozenset(
        {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
    )
    optional_capabilities: frozenset[str] = frozenset()

    def run(self, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
        from taim.baselines.bm25 import BM25Retriever, bm25_ranking_configuration
        from taim.folded_query_bundle import validate_folded_query_bundle

        raw_bundle = required_option(request, "folded_query_bundle")
        if not isinstance(raw_bundle, Mapping):
            raise ValueError("bm25-folded requires a folded query bundle object")
        queries, bundle_id = validate_folded_query_bundle(
            raw_bundle,
            expected_task_input_id=request.snapshot.task_input_id,
            expected_topic_ids=tuple(topic.topic_id for topic in request.snapshot.topics),
        )
        topics = tuple(
            replace(topic, canonical_text=query.canonical_text)
            for topic, query in zip(request.snapshot.topics, queries, strict=True)
        )
        started_at = time.perf_counter()
        retriever = BM25Retriever(
            request.snapshot.trials,
            k1=required_option(request, "k1"),  # type: ignore[arg-type]
            b=required_option(request, "b"),  # type: ignore[arg-type]
        )
        candidates = retriever.retrieve_many(
            topics,
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
        return PatientToTrialRunResult(
            candidates=tuple(candidates),
            configuration={
                **ranking,
                "model_identity": {"kind": "none", "reason": "deterministic_bm25"},
                "index_identity": {
                    "implementation": "taim-okapi-bm25-v1",
                    "persistent": False,
                    "corpus_statistics": retriever.corpus_statistics,
                },
                "query_transformation": {
                    "bundle_id": bundle_id,
                    "policy": raw_bundle["policy"],
                    "provenance": raw_bundle["provenance"],
                    "topic_count": len(queries),
                },
                "metric_cutoff": request.metric_cutoff,
                "corpus_statistics": retriever.corpus_statistics,
                "runtime_seconds": runtime_seconds,
            },
            runtime_seconds=runtime_seconds,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth=PIPELINE_DEPTH_RETRIEVAL,
        )
