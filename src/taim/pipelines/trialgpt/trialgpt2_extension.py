"""TrialGPT 2.0's CLI options, qualification commands, and release declarations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from taim.pipeline_extensions import PipelineExtension, RankingExecution, add_argument_once
from taim.pipelines.trialgpt.trialgpt2_execution import run_trialgpt2

if TYPE_CHECKING:
    from taim.release_profiles import ResolvedBenchmark

_SYSTEM_ID = "TrialGPT2-TAIM-Luna-v1"


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    from taim.cli_support import positive_integer

    # Every TrialGPT unit shares these three, first and in this order (see the TrialGPT unit's).
    add_argument_once(
        parser,
        "--trialgpt-frozen-retrieval",
        type=Path,
        help="fresh frozen retrieval artifact for this Task Input, from the retrieval producer; "
        "required by TrialGPT",
    )
    add_argument_once(
        parser,
        "--trialgpt-frozen-retrieval-lock",
        type=Path,
        help="the lock written with that artifact; required by TrialGPT",
    )
    _add_codex_executable_option(parser)
    parser.add_argument(
        "--trialgpt2-checkout",
        type=Path,
        help="a clean checkout of TrialGPT 2.0 at the pinned commit; required by TrialGPT 2.0",
    )
    parser.add_argument(
        "--trialgpt2-workspace",
        type=Path,
        help="an empty directory, or the workspace of an interrupted TrialGPT 2.0 run to resume "
        "in place; omitted, .trialgpt2/<run ID> under --output-dir",
    )
    parser.add_argument(
        "--trialgpt2-generation-workers",
        type=positive_integer,
        default=48,
        help="TrialGPT 2.0 candidate pairs scored at once, 1 to 48",
    )


def _add_codex_executable_option(parser: argparse.ArgumentParser) -> None:
    add_argument_once(
        parser,
        "--codex-executable",
        default="codex",
        help="Codex CLI that makes the provider calls; name it bare, not by absolute location, "
        "or the run cannot be bundled",
    )


def _options(
    args: argparse.Namespace, benchmark: ResolvedBenchmark, *, workspace: Path
) -> dict[str, object]:
    from taim import release_producer

    if (
        args.trialgpt_frozen_retrieval is None
        or args.trialgpt_frozen_retrieval_lock is None
        or args.trialgpt2_checkout is None
    ):
        raise ValueError(
            "TrialGPT2-TAIM-Luna-v1 requires --trialgpt-frozen-retrieval, "
            "--trialgpt-frozen-retrieval-lock and --trialgpt2-checkout"
        )
    return {
        "frozen_retrieval_path": args.trialgpt_frozen_retrieval.resolve(),
        "frozen_retrieval_lock_path": args.trialgpt_frozen_retrieval_lock.resolve(),
        "workspace": workspace.resolve(),
        "upstream_checkout": args.trialgpt2_checkout.resolve(),
        "retrieval_source_snapshot_id": benchmark.prepared.snapshot.snapshot_id,
        # The run consumes the retrieval only on the release it was frozen on.
        "executing_release_identity": release_producer.verified_release_identity(args),
        "generation_workers": args.trialgpt2_generation_workers,
        "codex_executable": args.codex_executable,
    }


def _run_options(
    args: argparse.Namespace,
    benchmark: ResolvedBenchmark,
    task_input_id: str,
) -> dict[str, object]:
    del task_input_id
    workspace = args.trialgpt2_workspace or (
        Path(args.output_dir) / ".trialgpt2" / (args.run_id or "run")
    )
    return _options(args, benchmark, workspace=workspace)


def _add_patient_to_trial_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    from taim.cli_support import (
        add_pool_options,
        add_real_input_options,
        add_release_identity_options,
    )
    from taim.pipelines.trialgpt.qualification import QUALIFICATION_GATES

    trialgpt2 = commands.add_parser(
        "trialgpt2", help="qualify TrialGPT 2.0 on TREC 2021, and validate its qualification runs"
    )
    trialgpt2_commands = trialgpt2.add_subparsers(dest="trialgpt2_operation", required=True)
    qualify = trialgpt2_commands.add_parser(
        "qualify",
        help="run one qualification gate over TREC 2021's external-fidelity frozen retrieval and "
        "write its record; makes provider calls, computes no effectiveness value",
    )
    qualify.add_argument(
        "--gate",
        choices=tuple(QUALIFICATION_GATES),
        required=True,
        help="validation, the one-topic run, or repeat, one of the three ten-topic runs",
    )
    qualify.add_argument("--run-id", required=True, help="a new run ID for this gate run")
    add_real_input_options(qualify)
    add_pool_options(qualify, required=True)
    _add_run_options(qualify)
    add_release_identity_options(qualify, rule="qualification requires them")
    qualify.set_defaults(handler=_trialgpt2_qualify)

    validate = trialgpt2_commands.add_parser(
        "validate-qualification",
        help="re-account for gate runs from the files they left: one validation record, or the "
        "three repeat records; reads no judgment",
    )
    validate.add_argument(
        "--gate", choices=tuple(QUALIFICATION_GATES), required=True, help="the gate validated"
    )
    validate.add_argument(
        "--record",
        type=Path,
        action="append",
        required=True,
        help="a gate run's trialgpt2-qualification.json; once for validation, three times for "
        "the repeats",
    )
    validate.set_defaults(handler=_trialgpt2_validate)


def _trialgpt2_qualify(args: argparse.Namespace) -> int:
    from taim import release_producer
    from taim.cli_support import DEFAULT_METRIC_CUTOFF, print_payload, read_pool_ids
    from taim.data import load_prepared_benchmark
    from taim.entity_versions import source_grounded_patient_evidence_profile
    from taim.patient_to_trial import PatientToTrialRunRequest
    from taim.pipelines.trialgpt.qualification import QUALIFICATION_PROFILE, QUALIFICATION_TRACK
    from taim.pipelines.trialgpt.trialgpt2_qualification import qualify_trialgpt2
    from taim.release_profiles import resolve_benchmark_profile
    from taim.release_support import (
        release_source_identity,
        require_frozen_effectiveness_protocol,
        require_supported_combination,
    )

    require_frozen_effectiveness_protocol(
        QUALIFICATION_PROFILE,
        args.protocol_approval_id,
        "external-baseline protocol_approval_id",
        system=_SYSTEM_ID,
    )
    if args.trialgpt2_workspace is None:
        raise ValueError("TrialGPT 2.0 qualification requires --trialgpt2-workspace")
    if release_producer.verified_release_identity(args) is None:
        raise ValueError(
            "TrialGPT 2.0 qualification requires --release-manifest, --package-artifact, and "
            "--dependency-lock"
        )
    release_producer.verified_dependency_environment(args, system_id=_SYSTEM_ID)
    prepared = load_prepared_benchmark(args.data_dir)
    capability = require_supported_combination(
        track=QUALIFICATION_TRACK,
        task="patient_to_trial",
        profile=QUALIFICATION_PROFILE,
        system=_SYSTEM_ID,
    )
    release_source_identity(
        prepared,
        track=QUALIFICATION_TRACK,
        profile=QUALIFICATION_PROFILE,
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    benchmark = resolve_benchmark_profile(
        prepared,
        profile_id=QUALIFICATION_PROFILE,
        expected_pool_count=args.pool_count,
        expected_pool_ids_sha256=args.pool_ids_sha256,
        expected_pool_receipt_id=args.pool_receipt_id,
        supplied_pool_ids=(read_pool_ids(args.pool_file) if args.pool_file is not None else None),
    )
    request = PatientToTrialRunRequest.for_benchmark(
        benchmark,
        run_id=args.run_id,
        top_k=20,
        metric_cutoff=DEFAULT_METRIC_CUTOFF,
        options=_options(args, benchmark, workspace=args.trialgpt2_workspace),
        clinical_as_of=args.clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )
    record = qualify_trialgpt2(request, gate=args.gate)
    print_payload(
        {
            "system": _SYSTEM_ID,
            "gate": record["gate"],
            "run_id": record["run_id"],
            "topic_ids": record["topic_ids"],
            "record": str(Path(str(record["run_workspace"])) / "trialgpt2-qualification.json"),
            "bridge_accounting": record["bridge_accounting"],
        }
    )
    return 0


def _trialgpt2_validate(args: argparse.Namespace) -> int:
    from taim.pipelines.trialgpt.trialgpt2_qualification import (
        validate_qualification_repeats,
        validate_qualification_run,
    )

    if args.gate == "validation":
        if len(args.record) != 1:
            raise ValueError("the validation gate is one record")
        report = validate_qualification_run(args.record[0])
    else:
        report = validate_qualification_repeats(args.record)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


EXTENSION = PipelineExtension(
    system_tasks={"TrialGPT2-TAIM-Luna-v1": "patient_to_trial"},
    pipeline_depth="post_eligibility",
    external_baseline=True,
    subcommands={("patient-to-trial", "trialgpt2"): "TrialGPT2-TAIM-Luna-v1"},
    patient_to_trial=RankingExecution(run=run_trialgpt2),
    add_run_options=_add_run_options,
    run_options=_run_options,
    add_patient_to_trial_commands=_add_patient_to_trial_commands,
    protected_reference_paths=(
        "docs/paper-analysis-protocol-2026-10-07-v14.md",
        "docs/paper-analysis-protocol-2026-10-07-v15.md",
        "docs/paper-analysis-protocol-2026-11-02-v17.md",
        "src/taim/pipelines/trialgpt/paper_row_declarations.py",
    ),
    dependency_declarations=("src/taim/data/external-system-dependencies-v1.json",),
    required_distributions=("openai", "tqdm"),
)
