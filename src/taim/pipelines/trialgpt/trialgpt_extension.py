"""TrialGPT CLI options, provider-free preflight, and release declarations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from taim.model_codex import codex_preflight
from taim.pipeline_extensions import PipelineExtension, RankingExecution, add_argument_once
from taim.pipelines.trialgpt.execution import run_publication

if TYPE_CHECKING:
    from taim.artifacts import RunEvaluationExtension
    from taim.release_profiles import ResolvedBenchmark

_SYSTEM_ID = "TrialGPT-TAIM-Luna-v1"


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    from taim.cli_support import positive_integer

    # Every TrialGPT unit shares these three, first and in this order, so a release that holds a
    # unit back lists the others' options in the order the full reference does.
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
        "--trialgpt-publication-contract",
        type=Path,
        help="contract written by trialgpt preflight; the run must match it exactly. Required by "
        "TrialGPT",
    )
    parser.add_argument(
        "--trialgpt-workspace",
        type=Path,
        help="an empty directory, or the workspace of an interrupted run to resume in place; "
        "omitted, .trialgpt/<run ID> under --output-dir",
    )
    parser.add_argument(
        "--trialgpt-generation-workers",
        type=positive_integer,
        default=48,
        help="concurrent provider calls, 1 to 48; must equal the preflight contract's",
    )


def _add_codex_executable_option(parser: argparse.ArgumentParser) -> None:
    add_argument_once(
        parser,
        "--codex-executable",
        default="codex",
        help="Codex CLI that makes the provider calls; name it bare, not by absolute location, "
        "or the run cannot be bundled",
    )


def _run_options(
    args: argparse.Namespace,
    benchmark: ResolvedBenchmark,
    task_input_id: str,
) -> dict[str, object]:
    del task_input_id
    workspace = args.trialgpt_workspace or (
        Path(args.output_dir) / ".trialgpt" / (args.run_id or "publication-run")
    )
    return _options(args, benchmark, workspace=workspace)


def _options(
    args: argparse.Namespace, benchmark: ResolvedBenchmark, *, workspace: Path
) -> dict[str, object]:
    from taim import release_producer

    if (
        args.trialgpt_publication_contract is None
        or args.trialgpt_frozen_retrieval is None
        or args.trialgpt_frozen_retrieval_lock is None
    ):
        raise ValueError(
            "TrialGPT-TAIM-Luna-v1 requires --trialgpt-publication-contract and "
            "fresh --trialgpt-frozen-retrieval and --trialgpt-frozen-retrieval-lock inputs"
        )
    commit = release_producer.release_source_commit(args)
    dirty = False
    return {
        "publication_contract_path": args.trialgpt_publication_contract.resolve(),
        "frozen_retrieval_path": args.trialgpt_frozen_retrieval.resolve(),
        "frozen_retrieval_lock_path": args.trialgpt_frozen_retrieval_lock.resolve(),
        "workspace": workspace.resolve(),
        "controller_git_commit": commit,
        "controller_working_tree_dirty": dirty,
        "retrieval_source_snapshot_id": benchmark.prepared.snapshot.snapshot_id,
        "evaluation_package_id": benchmark.evaluation_package.evaluation_package_id,
        "generation_workers": args.trialgpt_generation_workers,
        "codex_executable": args.codex_executable,
        # The run consumes the retrieval only on the release it was frozen on.
        "executing_release_identity": release_producer.verified_release_identity(args),
    }


def _add_patient_to_trial_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    from taim.cli_support import (
        add_pool_options,
        add_real_input_options,
        add_release_identity_options,
        positive_integer,
    )
    from taim.pipelines.trialgpt.qualification import QUALIFICATION_GATES
    from taim.release_profiles import EXTERNAL_FIDELITY_TREC_2021_PROFILE

    trialgpt = commands.add_parser(
        "trialgpt",
        help="prepare the provider-free TrialGPT execution contract, qualify TrialGPT on TREC "
        "2021, and validate its qualification runs",
    )
    trialgpt_commands = trialgpt.add_subparsers(dest="trialgpt_operation", required=True)
    preflight = trialgpt_commands.add_parser(
        "preflight",
        help="check a frozen retrieval artifact against one Task Input and write the publication "
        "contract a TrialGPT run must match; makes no provider call",
    )
    preflight.add_argument(
        "--track",
        choices=("sigir-ct-2016", "trec-ct-2021", "trec-ct-2022", "trec-ct-2023"),
        default="trec-ct-2021",
        help="Track of the run the contract is for",
    )
    preflight.add_argument(
        "--profile",
        default=EXTERNAL_FIDELITY_TREC_2021_PROFILE,
        help="Profile of the run the contract is for",
    )
    add_real_input_options(preflight)
    add_pool_options(preflight, required=True)
    preflight.add_argument(
        "--frozen-retrieval",
        type=Path,
        required=True,
        help="fresh frozen retrieval artifact for this Task Input, from the retrieval producer",
    )
    preflight.add_argument(
        "--frozen-retrieval-lock",
        type=Path,
        required=True,
        help="the lock written with that artifact",
    )
    preflight.add_argument(
        "--generation-workers",
        type=positive_integer,
        default=48,
        help="concurrent provider calls the run will make, 1 to 48",
    )
    _add_codex_executable_option(preflight)
    preflight.add_argument(
        "--output", type=Path, required=True, help="file that receives the publication contract"
    )
    add_release_identity_options(preflight, rule="preflight requires them")
    preflight.set_defaults(handler=_trialgpt_preflight)

    qualify = trialgpt_commands.add_parser(
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
    qualify.set_defaults(handler=_trialgpt_qualify)

    validate = trialgpt_commands.add_parser(
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
        help="a gate run's trialgpt-qualification.json; once for validation, three times for "
        "the repeats",
    )
    validate.set_defaults(handler=_trialgpt_validate)


def _trialgpt_preflight(args: argparse.Namespace) -> int:
    from taim import release_producer
    from taim.cli_support import (
        DEFAULT_METRIC_CUTOFF,
        DEFAULT_RETRIEVAL_DEPTH,
        print_payload,
        read_pool_ids,
    )
    from taim.data import PREPARED_MANIFEST_FILENAME, load_prepared_benchmark
    from taim.entity_versions import source_grounded_patient_evidence_profile
    from taim.patient_to_trial import PatientToTrialRunRequest
    from taim.pipelines.trialgpt.publication import (
        load_frozen_trialgpt_retrieval,
        make_publication_contract,
        validate_frozen_retrieval_for_release,
        validate_frozen_retrieval_for_snapshot,
        validate_frozen_retrieval_for_task_input,
    )
    from taim.pipelines.trialgpt.retrieval_producer import (
        require_trialgpt_capabilities,
        validate_query_plan_receipt_binding,
    )
    from taim.pipelines.trialgpt.system import (
        TrialGPTTAIMLunaV1System,
        trialgpt_codex_model,
        trialgpt_publication_method_contract,
    )
    from taim.release_profiles import resolve_benchmark_profile
    from taim.release_support import (
        RELEASE_VERSION,
        release_source_identity,
        require_frozen_effectiveness_protocol,
        require_supported_combination,
    )

    require_frozen_effectiveness_protocol(
        args.profile,
        args.protocol_approval_id,
        "external-baseline protocol_approval_id",
        system=_SYSTEM_ID,
    )
    require_trialgpt_capabilities(args.data_dir / PREPARED_MANIFEST_FILENAME)
    release_identity = release_producer.verified_release_identity(args)
    if release_identity is None:
        raise ValueError(
            "TrialGPT preflight requires --release-manifest, --package-artifact, and "
            "--dependency-lock"
        )
    release_producer.verified_dependency_environment(args, system_id=_SYSTEM_ID)
    commit = release_producer.release_source_commit(args)
    prepared = load_prepared_benchmark(args.data_dir)
    capability = require_supported_combination(
        track=args.track,
        task="patient_to_trial",
        profile=args.profile,
        system=_SYSTEM_ID,
    )
    release_source_identity(
        prepared,
        track=args.track,
        profile=args.profile,
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    benchmark = resolve_benchmark_profile(
        prepared,
        profile_id=args.profile,
        expected_pool_count=args.pool_count,
        expected_pool_ids_sha256=args.pool_ids_sha256,
        expected_pool_receipt_id=args.pool_receipt_id,
        supplied_pool_ids=(read_pool_ids(args.pool_file) if args.pool_file is not None else None),
    )
    request = PatientToTrialRunRequest.for_benchmark(
        benchmark,
        run_id="trialgpt-preflight-identity",
        top_k=DEFAULT_RETRIEVAL_DEPTH,
        metric_cutoff=DEFAULT_METRIC_CUTOFF,
        options={},
        clinical_as_of=args.clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
        required_capabilities=TrialGPTTAIMLunaV1System.required_capabilities,
        optional_capabilities=TrialGPTTAIMLunaV1System.optional_capabilities,
    )
    retrieval = load_frozen_trialgpt_retrieval(
        args.frozen_retrieval,
        lock_path=args.frozen_retrieval_lock,
    )
    validate_frozen_retrieval_for_snapshot(
        retrieval,
        benchmark_lineage=args.track,
        retrieval_source_snapshot_id=benchmark.prepared.snapshot.snapshot_id,
        topic_ids=[topic.topic_id for topic in benchmark.topics],
        trial_ids=[trial.trial_id for trial in benchmark.trials],
    )
    validate_frozen_retrieval_for_task_input(
        retrieval,
        task_input_id=request.snapshot.task_input_id,
    )
    # Refuse, before provider work, a lock that does not bind protocol v11 and carry its receipt,
    # or that was frozen on another release than this one.
    validate_query_plan_receipt_binding(retrieval.lock)
    validate_frozen_retrieval_for_release(retrieval, release_identity=release_identity)
    provider = trialgpt_codex_model(codex_executable=args.codex_executable)
    contract = make_publication_contract(
        taim_git_commit=commit,
        prepared_snapshot_id=benchmark.prepared.snapshot.snapshot_id,
        evaluation_package_id=benchmark.evaluation_package.evaluation_package_id,
        frozen_retrieval=retrieval,
        provider_configuration=codex_preflight(provider.configuration),
        method_configuration=trialgpt_publication_method_contract(),
        generation_workers=args.generation_workers,
        topic_ids=[topic.topic_id for topic in benchmark.topics],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print_payload(
        {
            "release": RELEASE_VERSION,
            "track": args.track,
            "task": "patient_to_trial",
            "profile": args.profile,
            "system": _SYSTEM_ID,
            "prepared_snapshot_id": benchmark.prepared.snapshot.snapshot_id,
            "task_input_id": request.snapshot.task_input_id,
            "evaluation_package_id": benchmark.evaluation_package.evaluation_package_id,
            "contract_id": contract["contract_id"],
            "output": str(args.output.resolve()),
        }
    )
    return 0


def _trialgpt_qualify(args: argparse.Namespace) -> int:
    from taim import release_producer
    from taim.cli_support import DEFAULT_METRIC_CUTOFF, print_payload, read_pool_ids
    from taim.data import load_prepared_benchmark
    from taim.entity_versions import source_grounded_patient_evidence_profile
    from taim.patient_to_trial import PatientToTrialRunRequest
    from taim.pipelines.trialgpt.qualification import (
        QUALIFICATION_PROFILE,
        QUALIFICATION_TRACK,
        qualify_trialgpt,
    )
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
    if args.trialgpt_workspace is None:
        raise ValueError("TrialGPT qualification requires --trialgpt-workspace")
    if release_producer.verified_release_identity(args) is None:
        raise ValueError(
            "TrialGPT qualification requires --release-manifest, --package-artifact, and "
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
        options=_options(args, benchmark, workspace=args.trialgpt_workspace),
        clinical_as_of=args.clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )
    record = qualify_trialgpt(request, gate=args.gate)
    print_payload(
        {
            "system": _SYSTEM_ID,
            "gate": record["gate"],
            "run_id": record["run_id"],
            "topic_ids": record["topic_ids"],
            "record": str(Path(str(record["run_workspace"])) / "trialgpt-qualification.json"),
            "generation_coverage": record["generation_coverage"],
        }
    )
    return 0


def _trialgpt_validate(args: argparse.Namespace) -> int:
    from taim.pipelines.trialgpt.qualification import (
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


def _evaluation_extension() -> RunEvaluationExtension:
    from taim.pipelines.trialgpt.evaluation import TRIALGPT_EVALUATION_EXTENSION

    return TRIALGPT_EVALUATION_EXTENSION


EXTENSION = PipelineExtension(
    system_tasks={"TrialGPT-TAIM-Luna-v1": "patient_to_trial"},
    pipeline_depth="post_eligibility",
    external_baseline=True,
    subcommands={("patient-to-trial", "trialgpt"): "TrialGPT-TAIM-Luna-v1"},
    patient_to_trial=RankingExecution(run=run_publication),
    add_run_options=_add_run_options,
    run_options=_run_options,
    add_patient_to_trial_commands=_add_patient_to_trial_commands,
    protocol_bindings=(
        (
            "src/taim/pipelines/trialgpt/retrieval_producer.py",
            "FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256",
            "docs/paper-analysis-protocol-2026-09-28-v11.md",
        ),
    ),
    module_exports={
        "src/taim/pipelines/trialgpt/system.py": (
            "TRIALGPT_CODEX_MODEL",
            "FrozenTrialGPTRetrievalStrategy",
            "TrialGPTArtifacts",
            "TrialGPTCandidateOutcome",
            "TrialGPTCandidateTask",
            "TrialGPTGeneration",
            "TrialGPTRetrievalResult",
            "TrialGPTRetrievalSession",
            "TrialGPTRetrievalStrategy",
            "TrialGPTTAIMLunaV1System",
            "trialgpt_publication_method_contract",
            "trialgpt_run_workspace",
        ),
    },
    forbidden_classes=(
        "MlxMedCPT",
        "PaperTrialGPTSystem",
        "TorchMedCPT",
        "TrialGPTCodexFailureAwareSystem",
        "TrialGPTCodexSystem",
        "TrialGPTFakeSystem",
    ),
    forbidden_text={"src/taim/pipelines/trialgpt/paper.py": ("MedCPT",)},
    protected_reference_paths=(
        "docs/external-baselines.md",
        "docs/paper-analysis-protocol-2026-09-02-v5.md",
        "docs/paper-analysis-protocol-2026-09-02-v6.md",
        "docs/paper-analysis-protocol-2026-10-07-v14.md",
        "docs/paper-analysis-protocol-2026-10-07-v15.md",
        "docs/paper-analysis-protocol-2026-10-09-v16.md",
        "docs/paper-analysis-protocol-2026-11-02-v17.md",
        "docs/trialgpt-external-baseline.md",
        "docs/trialgpt-retrieval-producer.md",
        "src/taim/pipelines/trialgpt/trialgpt_extension.py",
        "src/taim/pipelines/trialgpt/publication.py",
        "src/taim/pipelines/trialgpt/retrieval_producer.py",
        "tests/trialgpt/test_trialgpt_external_baseline.py",
        "tests/trialgpt/test_trialgpt_retrieval_producer.py",
    ),
    dependency_declarations=("src/taim/data/external-system-dependencies-v1.json",),
    evaluation_profiles=("trialgpt-paper-style",),
    evaluation_extension=_evaluation_extension,
)
