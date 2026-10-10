"""The lexical reverse-direction release unit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from taim.pipeline_extensions import PipelineExtension, RankingExecution
from taim.pipelines.bm25.reverse import (
    TRIAL_TO_PATIENT_BM25_SYSTEM_ID,
    TrialToPatientBM25System,
)

if TYPE_CHECKING:
    from taim.trial_to_patient import TrialToPatientRunRequest, TrialToPatientRunResult


def _system(system_id: str) -> TrialToPatientBM25System:
    if system_id != TRIAL_TO_PATIENT_BM25_SYSTEM_ID:
        raise ValueError(f"System {system_id!r} is not this release unit")
    return TrialToPatientBM25System()


def _prepare(system_id: str, request: TrialToPatientRunRequest) -> TrialToPatientRunRequest:
    from taim.trial_to_patient import normalize_trial_to_patient_request

    return normalize_trial_to_patient_request(_system(system_id), request)


def _run(system_id: str, request: TrialToPatientRunRequest) -> TrialToPatientRunResult:
    return _system(system_id).run(request)


def _add_options(parser) -> None:
    pass


EXTENSION = PipelineExtension(
    system_tasks={"bm25-trial-to-patient": "trial_to_patient"},
    pipeline_depth="retrieval",
    external_baseline=False,
    subcommands={
        ("trial-to-patient", "benchmark"): "bm25-trial-to-patient",
        ("trial-to-patient", "fixture"): "bm25-trial-to-patient",
    },
    trial_to_patient=RankingExecution(run=_run, prepare_request=_prepare),
    add_run_options=_add_options,
    default_system="bm25-trial-to-patient",
)
