"""The lexical reverse-direction release unit."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from taim.snapshot import CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT

if TYPE_CHECKING:
    from taim.trial_to_patient import TrialToPatientRunRequest, TrialToPatientRunResult

TRIAL_TO_PATIENT_BM25_SYSTEM_ID = "bm25-trial-to-patient"


def _reverse_option(request: TrialToPatientRunRequest, name: str) -> object:
    try:
        return request.options[name]
    except KeyError as exc:
        raise ValueError(f"{request.run_id} is missing reverse System option {name!r}") from exc


class TrialToPatientBM25System:
    system_id = TRIAL_TO_PATIENT_BM25_SYSTEM_ID
    option_names = frozenset({"b", "k1"})
    required_capabilities = frozenset(
        {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
    )
    optional_capabilities: frozenset[str] = frozenset()

    def run(self, request: TrialToPatientRunRequest) -> TrialToPatientRunResult:
        from taim.baselines.bm25 import BM25Retriever, bm25_ranking_configuration
        from taim.trial_to_patient import TrialToPatientCandidate, TrialToPatientRunResult

        started_at = time.perf_counter()
        index = BM25Retriever(
            request.task_input.patient_versions,
            document_id=lambda patient: patient.patient_id,
            document_text=lambda patient: patient.topic.canonical_text,
            k1=cast(float, _reverse_option(request, "k1")),
            b=cast(float, _reverse_option(request, "b")),
        )
        candidates: list[TrialToPatientCandidate] = []
        for trial_version in request.task_input.trial_versions:
            ranked = index.rank(
                trial_version.canonical_text,
                top_k=request.top_k,
            )
            candidates.extend(
                TrialToPatientCandidate(
                    run_id=request.run_id,
                    system_id=self.system_id,
                    trial_id=trial_version.trial_id,
                    trial_version_id=trial_version.trial_version_id,
                    patient_id=patient.patient_id,
                    patient_version_id=patient.patient_version_id,
                    rank=rank,
                    score=score,
                )
                for rank, (patient, score) in enumerate(ranked, start=1)
            )
        runtime_seconds = time.perf_counter() - started_at
        ranking = bm25_ranking_configuration(
            k1=index.k1,
            b=index.b,
            requested_top_k=request.top_k,
            document_count=len(index.documents),
        )
        return TrialToPatientRunResult(
            candidates=tuple(candidates),
            system_id=self.system_id,
            system_input_id=request.system_input_id,
            configuration={
                "implementation": ("taim.pipelines.bm25.reverse.TrialToPatientBM25System"),
                "system_identity": {"system_id": self.system_id},
                "model_identity": {
                    "kind": "none",
                    "reason": "deterministic_bm25",
                },
                "index_identity": {
                    "implementation": ranking["retrieval_formula"]["id"],
                    "persistent": False,
                },
                "retrieval_formula": ranking["retrieval_formula"],
                "parameters": ranking["parameters"],
                "requested_top_k": ranking["requested_top_k"],
                "effective_top_k": ranking["effective_top_k"],
                "metric_cutoff": request.metric_cutoff,
                "score_tie_breaking": "score descending, patient_id ascending",
                "indexed_fields": ["PatientEntityVersion.topic.canonical_text"],
                "query_fields": ["TrialVersion.trial.canonical_text"],
                "corpus_statistics": index.corpus_statistics,
                "runtime_seconds": runtime_seconds,
            },
            runtime_seconds=runtime_seconds,
        )
