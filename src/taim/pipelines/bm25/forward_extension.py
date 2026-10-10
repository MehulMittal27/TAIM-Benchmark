"""The deterministic lexical retrieval release unit."""

from __future__ import annotations

import argparse

from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialRunResult
from taim.pipeline_extensions import PipelineExtension, RankingExecution
from taim.pipelines.bm25.forward import ReleaseBM25System


def _add_options(parser: argparse.ArgumentParser) -> None:
    from taim.cli_support import add_bm25_options

    add_bm25_options(parser)


def _options(args, benchmark, task_input_id):
    return {"b": args.b, "k1": args.k1}


def _run(system_id: str, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
    from taim.release_reference_systems import run_normalized_system

    if system_id != ReleaseBM25System.system_id:
        raise ValueError(f"System {system_id!r} is not this release unit")
    return run_normalized_system(ReleaseBM25System(), request)


EXTENSION = PipelineExtension(
    system_tasks={"bm25": "patient_to_trial"},
    pipeline_depth="retrieval",
    external_baseline=False,
    subcommands={},
    patient_to_trial=RankingExecution(run=_run),
    add_run_options=_add_options,
    run_options=_options,
    default_system="bm25",
)
