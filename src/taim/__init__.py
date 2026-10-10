"""Deliberate public interface for TAIM benchmark contracts."""

from taim.patient_to_trial import (
    PatientToTrialRunRequest,
    PatientToTrialRunResult,
    PatientToTrialSystem,
    PatientToTrialTaskInput,
    StrictSystemOptions,
)
from taim.trial_to_patient import (
    ResolvedTrialToPatientBenchmark,
    StoredTrialToPatientRun,
    TrialToPatientBenchmarkProfile,
    TrialToPatientEvaluationPackage,
    TrialToPatientRunManifest,
    TrialToPatientRunRequest,
    TrialToPatientTaskInput,
    evaluate_trial_to_patient_run,
    load_trial_to_patient_run,
)

__version__ = "0.2.0"

__all__ = [
    "PatientToTrialRunRequest",
    "PatientToTrialRunResult",
    "PatientToTrialSystem",
    "PatientToTrialTaskInput",
    "ResolvedTrialToPatientBenchmark",
    "StoredTrialToPatientRun",
    "StrictSystemOptions",
    "TrialToPatientBenchmarkProfile",
    "TrialToPatientEvaluationPackage",
    "TrialToPatientRunManifest",
    "TrialToPatientRunRequest",
    "TrialToPatientTaskInput",
    "__version__",
    "evaluate_trial_to_patient_run",
    "load_trial_to_patient_run",
]
