"""The ``submission`` commands: export a Task Input, then import and validate an outside ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from taim.data import load_prepared_benchmark
from taim.entity_versions import (
    parse_clinical_as_of,
    source_grounded_patient_evidence_profile,
)
from taim.patient_to_trial import PatientToTrialRunRequest
from taim.release_fixture import FIXTURE_CLINICAL_AS_OF, load_release_fixture
from taim.release_profiles import ResolvedBenchmark, resolve_benchmark_profile


def _print(value: object) -> int:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


def _reference_system(task: str) -> str:
    # Export asks the catalogue which Track and Profile pairs exist through the reference System's
    # rows, so a submission is scored on exactly the inputs a catalogue run of that System gets.
    from taim.release_support import reference_system_for_task

    return reference_system_for_task(task)


def _require_fixture_only(args: argparse.Namespace) -> None:
    if (
        args.data_dir is not None
        or args.profile is not None
        or args.clinical_as_of is not None
        or args.protocol_approval_id is not None
    ):
        raise ValueError("--fixture cannot be combined with real-data input flags")


def _require_protocol(args: argparse.Namespace) -> None:
    """Apply the benchmark run's protocol rule before any real input is opened."""

    from taim.release_support import (
        require_frozen_effectiveness_protocol,
        require_supported_combination,
        requires_effectiveness_protocol,
    )

    if args.profile is None:
        raise ValueError("real input requires --profile")
    require_supported_combination(
        track=args.track,
        task=args.task,
        profile=args.profile,
        system=_reference_system(args.task),
    )
    if requires_effectiveness_protocol(args.profile):
        require_frozen_effectiveness_protocol(
            args.profile,
            args.protocol_approval_id,
            "external submission protocol_approval_id",
            system=_reference_system(args.task),
        )
    elif args.protocol_approval_id is not None:
        raise ValueError(
            "protocol approval identities apply only to judgment-union or confirmation profiles"
        )


def _forward_input(args: argparse.Namespace) -> ResolvedBenchmark:
    from taim.cli_support import read_pool_ids
    from taim.release_support import source_profile_id

    if args.fixture:
        return resolve_benchmark_profile(
            load_release_fixture(args.track), profile_id="synthetic-patient-to-trial"
        )
    if args.data_dir is None or args.clinical_as_of is None:
        raise ValueError("real input requires --data-dir, --profile, and --clinical-as-of")
    prepared = load_prepared_benchmark(args.data_dir)
    if prepared.snapshot.benchmark_lineage != args.track:
        raise ValueError("prepared input belongs to a different Track")
    if (
        args.track == "sigir-ct-2016"
        and prepared.dataset_id != f"sigir-ct-2016-{source_profile_id(args.profile)}"
    ):
        raise ValueError("SIGIR prepared query profile does not match --profile")
    return resolve_benchmark_profile(
        prepared,
        profile_id=args.profile,
        expected_pool_count=args.pool_count,
        expected_pool_ids_sha256=args.pool_ids_sha256,
        expected_pool_receipt_id=args.pool_receipt_id,
        supplied_pool_ids=read_pool_ids(args.pool_file) if args.pool_file is not None else None,
    )


def _export(args: argparse.Namespace) -> int:
    from taim.release_producer import verified_release_identity
    from taim.reverse_fixture import load_reverse_fixture
    from taim.reverse_viability import qualify_reverse_viability_profile
    from taim.submissions import export_task

    if args.fixture:
        _require_fixture_only(args)
    else:
        _require_protocol(args)
    identity = None if args.fixture else verified_release_identity(args)
    if args.task == "patient_to_trial":
        benchmark = _forward_input(args)
        request = PatientToTrialRunRequest.for_benchmark(
            benchmark,
            run_id="external-task-export",
            top_k=args.top_k,
            metric_cutoff=5,
            options={},
            clinical_as_of=FIXTURE_CLINICAL_AS_OF if args.fixture else args.clinical_as_of,
            patient_evidence_profile=source_grounded_patient_evidence_profile(),
        )
        native = request.snapshot.task_input_dict()
        package = benchmark.evaluation_package.to_dict()
        profile = benchmark.profile.profile_id
        prepared_source = benchmark.prepared
        projection = benchmark.manifest_configuration()
    else:
        if args.fixture:
            reverse = load_reverse_fixture(args.track)
        else:
            if (
                args.track != "trec-ct-2021"
                or args.profile != "trec-ct-2021-reverse-complete10"
                or args.data_dir is None
                or args.clinical_as_of is None
            ):
                raise ValueError(
                    "real reverse export requires the trec-ct-2021-reverse-complete10 Profile, "
                    "--data-dir, and --clinical-as-of"
                )
            reverse = qualify_reverse_viability_profile(
                load_prepared_benchmark(args.data_dir),
                clinical_as_of=args.clinical_as_of,
                patient_evidence_profile=source_grounded_patient_evidence_profile(),
            )
        native = reverse.task_input.to_dict()
        package = reverse.evaluation_package.to_dict()
        profile = reverse.profile.profile_id
        prepared_source = reverse.prepared
        projection = reverse.profile.to_dict()
    from taim.release_support import release_source_identity, require_supported_combination

    capability = require_supported_combination(
        track=args.track,
        task=args.task,
        profile=profile,
        system=_reference_system(args.task),
    )
    source = release_source_identity(
        prepared_source,
        track=args.track,
        profile=profile,
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    result = export_task(
        task=args.task,
        native_input=native,
        evaluation_package=package,
        profile=profile,
        benchmark_profile=projection,
        release_source=source,
        top_k=args.top_k,
        release_identity=identity,
        output=args.output,
        evaluator_output=args.evaluator_output,
    )
    return _print(
        {
            "export_id": result["export_id"],
            "task_input_id": result["input"]["task_input_id"],
            "evaluation_package_id": result["evaluation_package_id"],
            "task": args.task,
            "profile": profile,
            "output": str(args.output),
            "evaluator_output": str(args.evaluator_output),
        }
    )


def _import(args: argparse.Namespace) -> int:
    from taim.submissions import import_submission

    return _print(
        import_submission(
            task_dir=args.task_dir,
            evaluator_dir=args.evaluator_dir,
            ranking=args.ranking,
            pipeline=args.pipeline,
            task=args.task,
            output=args.output,
        )
    )


def _validate(args: argparse.Namespace) -> int:
    from taim.submissions import validate_submission

    return _print(
        validate_submission(
            args.submission_dir,
            task_dir=args.task_dir,
            evaluator_dir=args.evaluator_dir,
            task=args.task,
        )
    )


def _input_options(parser: argparse.ArgumentParser) -> None:
    from taim.cli_support import add_pool_options, add_release_identity_options
    from taim.release_support import SUPPORTED_TRACKS

    parser.add_argument(
        "--track",
        choices=SUPPORTED_TRACKS,
        default="trec-ct-2021",
        help="Track of the task to export",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="export packaged synthetic data without downloads",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="directory written by data prepare; required without --fixture",
    )
    parser.add_argument(
        "--profile", help="Benchmark Profile of the task; required without --fixture"
    )
    parser.add_argument(
        "--clinical-as-of",
        type=parse_clinical_as_of,
        help="clinical as-of time, bound into the Task Input ID; required without --fixture",
    )
    add_pool_options(parser)
    add_release_identity_options(parser, rule="every export without --fixture requires them")


def add_submission_commands(commands: Any, *, reverse: bool) -> None:
    from taim.cli_support import positive_integer

    parent = commands.add_parser(
        "submission", help="export inputs and evaluate an external pipeline's rankings"
    )
    sub = parent.add_subparsers(dest="submission_operation", required=True)
    for name, handler, help_text in (
        (
            "export",
            _export,
            "write the Task Input a pipeline ranks and, apart from it, the judgments that score it",
        ),
        (
            "import",
            _import,
            "check a pipeline's ranking and declaration against the exported task and score it",
        ),
        (
            "validate",
            _validate,
            "recompute an imported submission's checksums and metrics",
        ),
    ):
        parser = sub.add_parser(name, help=help_text)
        parser.set_defaults(
            handler=handler, task="trial_to_patient" if reverse else "patient_to_trial"
        )
        if name == "export":
            _input_options(parser)
            parser.add_argument(
                "--top-k",
                type=positive_integer,
                default=1000,
                help="depth the pipeline must rank each query to",
            )
            parser.add_argument(
                "--output",
                type=Path,
                required=True,
                help="new task directory: input.json only, the one file the pipeline reads",
            )
            parser.add_argument(
                "--evaluator-output",
                type=Path,
                required=True,
                help="new evaluator directory: evaluation-package.json only; keep it from the "
                "pipeline",
            )
            continue
        parser.add_argument(
            "--task-dir",
            type=Path,
            required=True,
            help="task directory export wrote; it may hold input.json only",
        )
        parser.add_argument(
            "--evaluator-dir",
            type=Path,
            required=True,
            help="evaluator directory export wrote",
        )
        if name == "import":
            parser.add_argument(
                "--ranking",
                type=Path,
                required=True,
                help="the pipeline's ranking.jsonl, one JSON row per ranked pair",
            )
            parser.add_argument(
                "--pipeline",
                type=Path,
                required=True,
                help="the pipeline's pipeline.json declaration",
            )
            parser.add_argument(
                "--output",
                type=Path,
                required=True,
                help="directory that receives the checked submission and its metrics",
            )
        else:
            parser.add_argument(
                "--submission-dir",
                type=Path,
                required=True,
                help="directory import wrote",
            )
