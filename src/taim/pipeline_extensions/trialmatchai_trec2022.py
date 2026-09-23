"""The independently selectable TrialMatchAI TREC 2022 System binding."""

from __future__ import annotations

from taim.adapters.trialmatchai import TrialMatchAIL4Base
from taim.pipeline_extensions import PipelineExtension
from taim.trialmatchai_release import (
    _add_patient_to_trial_commands,
    _add_run_options,
    _run_options,
    run_system,
)

TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID = "trialmatchai-current-cuda-l4-trec22-development-v4"


class TrialMatchAIL4TREC2022System(TrialMatchAIL4Base):
    """Frozen method under its track-specific identity."""

    system_id = TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID


def _run(system_id, request):
    if system_id != TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID:
        raise ValueError(f"System {system_id!r} is not this release unit")
    return run_system(TrialMatchAIL4TREC2022System, request)


EXTENSION = PipelineExtension(
    system_tasks={"trialmatchai-current-cuda-l4-trec22-development-v4": "patient_to_trial"},
    pipeline_depth="post_eligibility",
    external_baseline=True,
    subcommands={},
    run=_run,
    add_run_options=_add_run_options,
    run_options=_run_options,
    add_patient_to_trial_commands=_add_patient_to_trial_commands,
    module_exports={
        "src/taim/adapters/trialmatchai.py": (
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
            "TrialMatchAIAdapterError",
            "TrialMatchAIBase",
            "TrialMatchAIHarnessInputs",
            "TrialMatchAIL4Base",
            "TrialMatchAIResult",
            "load_and_normalize_ranked_trials",
            "normalize_ranked_trials",
            "render_patient_input",
            "run_trialmatchai_command",
            "trialmatchai_l4_runtime_contract",
            "verify_trialmatchai_l4_adapter_weights",
            "write_patient_input",
        ),
    },
    forbidden_classes=("TrialMatchAISystem",),
    protected_reference_paths=(
        "src/taim/pipeline_extensions/trialmatchai_trec2022.py",
        "tests/test_trialmatchai_trec2022.py",
        "docs/external-baselines.md",
        "docs/trialmatchai-external-baseline.md",
    ),
    dependency_declarations=("src/taim/data/external-system-dependencies-v1.json",),
)
