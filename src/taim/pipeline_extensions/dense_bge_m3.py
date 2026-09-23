"""One selectable dense encoder binding."""

from taim.pipeline_extensions import PipelineExtension


def _add_options(parser):
    from taim.release_reference_systems import add_dense_options

    add_dense_options(parser)


def _options(args, benchmark, task_input_id):
    from taim.release_reference_systems import dense_options

    return dense_options(args, benchmark, task_input_id)


def _run(system_id, request):
    from taim.release_reference_systems import run_dense_system

    return run_dense_system(system_id, request)


EXTENSION = PipelineExtension(
    system_tasks={"dense-bge-m3": "patient_to_trial"},
    pipeline_depth="retrieval",
    external_baseline=False,
    subcommands={},
    run=_run,
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
