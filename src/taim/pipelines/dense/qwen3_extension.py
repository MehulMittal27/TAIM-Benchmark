"""One selectable dense encoder binding."""

from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialRunResult
from taim.pipeline_extensions import PipelineExtension, RankingExecution


def _add_options(parser):
    from taim.pipelines.dense.release import add_dense_options

    add_dense_options(parser)


def _options(args, benchmark, task_input_id):
    from taim.pipelines.dense.release import dense_options

    return dense_options(args, benchmark, task_input_id)


def _run(system_id: str, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
    from taim.pipelines.dense.release import run_dense_system

    return run_dense_system(system_id, request)


EXTENSION = PipelineExtension(
    system_tasks={"dense-qwen3-embedding-0.6b": "patient_to_trial"},
    pipeline_depth="retrieval",
    external_baseline=False,
    subcommands={},
    patient_to_trial=RankingExecution(run=_run),
    add_run_options=_add_options,
    run_options=_options,
    required_distributions=(
        "huggingface-hub",
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers",
    ),
)
