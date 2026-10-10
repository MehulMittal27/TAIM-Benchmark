"""Patient-to-trial dispatch and exact-request provenance."""

from __future__ import annotations

from taim.patient_to_trial import (
    PatientToTrialRunRequest,
    PatientToTrialRunResult,
    PatientToTrialSystem,
    normalize_system_request,
)
from taim.pipeline_extensions import extension_for


def run_release_reference_system(
    system_id: str,
    request: PatientToTrialRunRequest,
) -> PatientToTrialRunResult:
    """Run one exported non-fusion System through its exact input contract."""

    if not isinstance(request, PatientToTrialRunRequest):
        raise TypeError("request must be a PatientToTrialRunRequest")
    extension = extension_for(system_id)
    if (
        extension is not None
        and extension.patient_to_trial is not None
        and extension.system_tasks[system_id] == "patient_to_trial"
    ):
        execution = extension.patient_to_trial
        return execution.run(system_id, execution.prepare(system_id, request))
    raise ValueError(f"System {system_id!r} is not an exported direct reference System")


def run_normalized_system(
    system: PatientToTrialSystem, request: PatientToTrialRunRequest
) -> PatientToTrialRunResult:
    """Run one System on its normalized request, recording the exact System Input it saw."""

    effective = normalize_system_request(system, request)
    result = system.run(effective)
    return PatientToTrialRunResult(
        candidates=result.candidates,
        configuration={
            **result.configuration,
            "system_input": effective.system_input_record_dict(),
        },
        runtime_seconds=result.runtime_seconds,
        primary_ranking=result.primary_ranking,
        pipeline_depth=result.pipeline_depth,
        stage_rankings=result.stage_rankings,
    )
