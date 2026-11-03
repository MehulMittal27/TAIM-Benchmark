"""The independently selectable TrialMatchAI v0.9.1 TREC 2023 System binding."""

from __future__ import annotations

from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialRunResult
from taim.pipeline_extensions import PipelineExtension, RankingExecution
from taim.pipelines.trialmatchai.execution import run_system, trialmatchai_add_run_options
from taim.pipelines.trialmatchai_v091.release import (
    trialmatchai_upstream_add_patient_to_trial_commands,
)
from taim.pipelines.trialmatchai_v091.system import trialmatchai_upstream_run_options
from taim.pipelines.trialmatchai_v091.trec23 import (
    TRIALMATCHAI_UPSTREAM_TREC_2023_SYSTEM_ID,
    TrialMatchAIUpstreamTREC2023System,
)


def _run(system_id: str, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
    if system_id != TRIALMATCHAI_UPSTREAM_TREC_2023_SYSTEM_ID:
        raise ValueError(f"System {system_id!r} is not this release unit")
    return run_system(TrialMatchAIUpstreamTREC2023System, request)


EXTENSION = PipelineExtension(
    system_tasks={"trialmatchai-upstream-v091-cuda-trec23-development-v1": "patient_to_trial"},
    pipeline_depth="post_eligibility",
    external_baseline=True,
    subcommands={},
    patient_to_trial=RankingExecution(run=_run),
    add_run_options=trialmatchai_add_run_options,
    run_options=trialmatchai_upstream_run_options,
    add_patient_to_trial_commands=trialmatchai_upstream_add_patient_to_trial_commands,
    module_exports={
        "src/taim/pipelines/trialmatchai_v091/system.py": (
            "TRIALMATCHAI_UPSTREAM_COMMIT",
            "TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_MANIFEST_SHA256",
            "TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_RULING",
            "TRIALMATCHAI_UPSTREAM_CONFIG_RESOURCE",
            "TRIALMATCHAI_UPSTREAM_CONFIG_SHA256",
            "TRIALMATCHAI_UPSTREAM_DRIVER_SHA256",
            "TRIALMATCHAI_UPSTREAM_MODEL_PINS",
            "TRIALMATCHAI_UPSTREAM_REPOSITORY",
            "TRIALMATCHAI_UPSTREAM_REQUIRED_ENVIRONMENT",
            "TRIALMATCHAI_UPSTREAM_SNAPSHOT_FILES",
            "TRIALMATCHAI_UPSTREAM_TOPIC_RENDERER_VERSION",
            "TRIALMATCHAI_UPSTREAM_TREC_PRESET",
            "TrialMatchAIUpstreamV091Base",
            "trialmatchai_upstream_config_bytes",
            "trialmatchai_upstream_linking_evidence",
            "trialmatchai_upstream_run_options",
            "trialmatchai_upstream_runtime_contract",
            "verify_trialmatchai_upstream_model_cache",
        ),
    },
    protected_reference_paths=(
        "src/taim/pipelines/trialmatchai_v091/trec23.py",
        "src/taim/pipelines/trialmatchai_v091/trec23_extension.py",
        "tests/trialmatchai_v091/test_trialmatchai_upstream_trec23.py",
        "docs/trialmatchai-v091-external-baseline.md",
    ),
    dependency_declarations=("src/taim/data/external-system-dependencies-v1.json",),
)
