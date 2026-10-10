"""The independently selectable TrialMatchAI TREC 2023 System binding."""

from __future__ import annotations

from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialRunResult
from taim.pipeline_extensions import PipelineExtension, RankingExecution
from taim.pipelines.trialmatchai.execution import run_system, trialmatchai_add_run_options
from taim.pipelines.trialmatchai.release import (
    trialmatchai_add_patient_to_trial_commands,
    trialmatchai_run_options,
)
from taim.pipelines.trialmatchai.trec2023 import (
    TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID,
    TrialMatchAIL4TREC2023System,
)


def _run(system_id: str, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
    if system_id != TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID:
        raise ValueError(f"System {system_id!r} is not this release unit")
    return run_system(TrialMatchAIL4TREC2023System, request)


EXTENSION = PipelineExtension(
    system_tasks={"trialmatchai-current-cuda-l4-trec23-development-v4": "patient_to_trial"},
    pipeline_depth="post_eligibility",
    external_baseline=True,
    subcommands={},
    patient_to_trial=RankingExecution(run=_run),
    add_run_options=trialmatchai_add_run_options,
    run_options=trialmatchai_run_options,
    add_patient_to_trial_commands=trialmatchai_add_patient_to_trial_commands,
    module_exports={
        "src/taim/pipelines/trialmatchai/corpus.py": (
            "TrialMatchAIAdapterError",
            "TrialMatchAIHarnessInputs",
            "TrialMatchAIResult",
            "load_and_normalize_ranked_trials",
            "normalize_ranked_trials",
        ),
        "src/taim/pipelines/trialmatchai/system.py": (
            "PATIENT_RENDERER_VERSION",
            "TRIALMATCHAI_L4_ADAPTER_SHA256",
            "TRIALMATCHAI_L4_ADAPTER_WEIGHT_FILENAME",
            "TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH",
            "TRIALMATCHAI_L4_CONFIG_SHA256",
            "TRIALMATCHAI_L4_FORBIDDEN_ENVIRONMENT",
            "TRIALMATCHAI_L4_PROCESS_EXECUTION_MODE",
            "TRIALMATCHAI_L4_REQUIRED_ENVIRONMENT",
            "TRIALMATCHAI_OPTIONAL_CAPABILITIES",
            "TRIALMATCHAI_REQUIRED_CAPABILITIES",
            "TRIALMATCHAI_SERVING_BACKEND_OVERRIDES",
            "TrialMatchAIBase",
            "TrialMatchAIL4Base",
            "render_patient_input",
            "run_trialmatchai_command",
            "trialmatchai_l4_runtime_contract",
            "verify_trialmatchai_l4_adapter_weights",
            "write_patient_input",
        ),
    },
    forbidden_classes=("TrialMatchAISystem",),
    protected_reference_paths=(
        "src/taim/pipelines/trialmatchai/trec2023.py",
        "src/taim/pipelines/trialmatchai/trec2023_extension.py",
        "tests/trialmatchai/test_trialmatchai_trec2023.py",
        "docs/external-baselines.md",
        "docs/trialmatchai-external-baseline.md",
    ),
    dependency_declarations=("src/taim/data/external-system-dependencies-v1.json",),
)
