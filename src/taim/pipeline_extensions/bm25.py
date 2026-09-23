"""The deterministic lexical retrieval release unit."""

from __future__ import annotations

import argparse
import time
from typing import Any

from taim.pipeline_extensions import PipelineExtension, add_argument_once
from taim.release_reference_systems import _option
from taim.schemas import PIPELINE_DEPTH_RETRIEVAL, PRIMARY_RANKING_CANDIDATES
from taim.snapshot import CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT
from taim.system_contracts import StrictSystemOptions, SystemRunRequest, SystemRunResult


class ReleaseBM25System(StrictSystemOptions):
    system_id = "bm25"
    option_names = frozenset({"b", "k1"})
    required_capabilities = frozenset(
        {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
    )
    optional_capabilities: frozenset[str] = frozenset()

    def run(self, request: SystemRunRequest) -> SystemRunResult:
        from taim.baselines.bm25 import BM25Retriever, bm25_ranking_configuration

        started_at = time.perf_counter()
        retriever = BM25Retriever(
            request.snapshot.trials,
            k1=_option(request, "k1"),  # type: ignore[arg-type]
            b=_option(request, "b"),  # type: ignore[arg-type]
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
        return SystemRunResult(
            candidates=tuple(candidates),
            configuration=configuration,
            runtime_seconds=runtime_seconds,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth=PIPELINE_DEPTH_RETRIEVAL,
        )


def _add_options(parser: argparse.ArgumentParser) -> None:
    add_argument_once(parser, "--k1", type=float, default=1.2)
    add_argument_once(parser, "--b", type=float, default=0.75)


def _options(args, benchmark, task_input_id):
    return {"b": args.b, "k1": args.k1}


def _run(system_id, request):
    from taim.release_reference_systems import run_normalized_system

    if system_id != ReleaseBM25System.system_id:
        raise ValueError(f"System {system_id!r} is not this release unit")
    return run_normalized_system(ReleaseBM25System(), request)


EXTENSION = PipelineExtension(
    system_tasks={"bm25": "patient_to_trial"},
    pipeline_depth="retrieval",
    external_baseline=False,
    subcommands={},
    run=_run,
    add_run_options=_add_options,
    run_options=_options,
    default_system="bm25",
)
