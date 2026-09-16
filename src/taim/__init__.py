"""Deliberate public interface for TAIM benchmark contracts."""

from taim.system_contracts import (
    StrictSystemOptions,
    System,
    SystemBenchmarkSnapshot,
    SystemRunRequest,
    SystemRunResult,
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

__version__ = "0.1.0"

__all__ = [
    "ResolvedTrialToPatientBenchmark",
    "StoredTrialToPatientRun",
    "StrictSystemOptions",
    "System",
    "SystemBenchmarkSnapshot",
    "SystemRunRequest",
    "SystemRunResult",
    "TrialToPatientBenchmarkProfile",
    "TrialToPatientEvaluationPackage",
    "TrialToPatientRunManifest",
    "TrialToPatientRunRequest",
    "TrialToPatientTaskInput",
    "__version__",
    "evaluate_trial_to_patient_run",
    "load_trial_to_patient_run",
]
