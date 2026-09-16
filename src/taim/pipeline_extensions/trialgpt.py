"""TrialGPT-TAIM-Luna-v1 in the public release: its run options, its preflight, and its run.

Only the TrialGPT-TAIM-Luna-v1 pipeline owns this module; see :mod:`taim.pipeline_extensions`. The
release command helpers are reached through :mod:`taim.release_cli` when called, not imported by
name, so what replaces them on that module replaces them here too.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from taim.pipeline_extensions import PipelineExtension

if TYPE_CHECKING:
    from taim.artifacts import RunEvaluationExtension
    from taim.release_profiles import ResolvedBenchmark
    from taim.system_contracts import SystemRunRequest, SystemRunResult

_SYSTEM_ID = "TrialGPT-TAIM-Luna-v1"


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    from taim.release_cli import _positive_integer

    parser.add_argument("--trialgpt-publication-contract", type=Path)
    parser.add_argument("--trialgpt-frozen-retrieval", type=Path)
    parser.add_argument("--trialgpt-frozen-retrieval-lock", type=Path)
    parser.add_argument("--trialgpt-workspace", type=Path)
    parser.add_argument("--trialgpt-generation-workers", type=_positive_integer, default=48)
    parser.add_argument("--codex-executable", default="codex")


def _run_options(
    args: argparse.Namespace,
    benchmark: ResolvedBenchmark,
    task_input_id: str,
) -> dict[str, object]:
    import taim.release_cli as release_cli

    if (
        args.trialgpt_publication_contract is None
        or args.trialgpt_frozen_retrieval is None
        or args.trialgpt_frozen_retrieval_lock is None
    ):
        raise ValueError(
            "TrialGPT-TAIM-Luna-v1 requires --trialgpt-publication-contract and "
            "fresh --trialgpt-frozen-retrieval and --trialgpt-frozen-retrieval-lock inputs"
        )
    commit = release_cli._release_source_commit(args)
    dirty = False
    workspace = args.trialgpt_workspace or (
        Path(args.output_dir) / ".trialgpt" / (args.run_id or "publication-run")
    )
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
    }


def _add_patient_to_trial_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    from taim.entity_versions import parse_clinical_as_of
    from taim.release_cli import _add_release_identity_options, _positive_integer, _sha256_identity
    from taim.release_profiles import EXTERNAL_FIDELITY_TREC_2021_PROFILE

    trialgpt = commands.add_parser(
        "trialgpt", help="prepare the provider-free TrialGPT execution contract"
    )
    trialgpt_commands = trialgpt.add_subparsers(dest="trialgpt_operation", required=True)
    preflight = trialgpt_commands.add_parser("preflight")
    preflight.add_argument(
        "--track",
        choices=("sigir-ct-2016", "trec-ct-2021", "trec-ct-2022"),
        default="trec-ct-2021",
    )
    preflight.add_argument("--profile", default=EXTERNAL_FIDELITY_TREC_2021_PROFILE)
    preflight.add_argument("--data-dir", type=Path, required=True)
    preflight.add_argument("--clinical-as-of", type=parse_clinical_as_of, required=True)
    preflight.add_argument("--pool-file", type=Path)
    preflight.add_argument("--pool-count", type=_positive_integer, required=True)
    preflight.add_argument("--pool-ids-sha256", type=_sha256_identity, required=True)
    preflight.add_argument("--pool-receipt-id", type=_sha256_identity, required=True)
    preflight.add_argument("--protocol-approval-id", type=_sha256_identity, required=True)
    preflight.add_argument("--frozen-retrieval", type=Path, required=True)
    preflight.add_argument("--frozen-retrieval-lock", type=Path, required=True)
    preflight.add_argument("--generation-workers", type=_positive_integer, default=48)
    preflight.add_argument("--codex-executable", default="codex")
    preflight.add_argument("--output", type=Path, required=True)
    _add_release_identity_options(preflight)
    preflight.set_defaults(handler=_trialgpt_preflight)


def _trialgpt_preflight(args: argparse.Namespace) -> int:
    import taim.release_cli as release_cli
    from taim.adapters.trialgpt import (
        TRIALGPT_CODEX_MODEL,
        CodexTrialGPTProvider,
        trialgpt_publication_method_contract,
    )
    from taim.entity_versions import source_grounded_patient_evidence_profile
    from taim.system_contracts import SystemRunRequest
    from taim.trialgpt_publication import (
        load_frozen_trialgpt_retrieval,
        make_publication_contract,
        validate_frozen_retrieval_for_snapshot,
        validate_frozen_retrieval_for_task_input,
    )

    release_cli.require_frozen_effectiveness_protocol(
        args.profile,
        args.protocol_approval_id,
        "external-baseline protocol_approval_id",
        system=_SYSTEM_ID,
    )
    release_identity = release_cli._verified_release_identity(args)
    if release_identity is None:
        raise ValueError(
            "TrialGPT preflight requires --release-manifest, --package-artifact, and "
            "--dependency-lock"
        )
    release_cli._verified_dependency_environment(args, system_id=_SYSTEM_ID)
    commit = release_cli._release_source_commit(args)
    prepared = release_cli.load_prepared_benchmark(args.data_dir)
    capability = release_cli.require_supported_combination(
        track=args.track,
        task="patient_to_trial",
        profile=args.profile,
        system=_SYSTEM_ID,
    )
    release_cli.release_source_identity(
        prepared,
        track=args.track,
        profile=args.profile,
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    benchmark = release_cli.resolve_benchmark_profile(
        prepared,
        profile_id=args.profile,
        expected_pool_count=args.pool_count,
        expected_pool_ids_sha256=args.pool_ids_sha256,
        expected_pool_receipt_id=args.pool_receipt_id,
        supplied_pool_ids=(
            release_cli._read_pool_ids(args.pool_file) if args.pool_file is not None else None
        ),
    )
    request = SystemRunRequest.for_benchmark(
        benchmark,
        run_id="trialgpt-preflight-identity",
        top_k=release_cli.DEFAULT_RETRIEVAL_DEPTH,
        metric_cutoff=release_cli.DEFAULT_METRIC_CUTOFF,
        options={},
        clinical_as_of=args.clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
        identity_version="2.0",
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
    provider = CodexTrialGPTProvider(
        model=TRIALGPT_CODEX_MODEL,
        reasoning_effort=None,
        codex_executable=args.codex_executable,
    )
    contract = make_publication_contract(
        taim_git_commit=commit,
        prepared_snapshot_id=benchmark.prepared.snapshot.snapshot_id,
        evaluation_package_id=benchmark.evaluation_package.evaluation_package_id,
        frozen_retrieval=retrieval,
        provider_configuration=provider.provenance(),
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
    release_cli._print(
        {
            "release": release_cli.RELEASE_VERSION,
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


def _run(system_id: str, request: SystemRunRequest) -> SystemRunResult:
    """Resolve physical TrialGPT inputs into the frozen publication System Input."""

    if system_id != _SYSTEM_ID:
        raise ValueError(f"System {system_id!r} is not an exported direct reference System")
    from taim.adapters.trialgpt import (
        TRIALGPT_CODEX_MODEL,
        CodexTrialGPTProvider,
        TrialGPTTAIMLunaV1System,
        trialgpt_publication_method_contract,
        trialgpt_run_workspace,
    )
    from taim.system_contracts import bind_system_capabilities
    from taim.trialgpt_publication import (
        TRIALGPT_PUBLICATION_CANDIDATE_DEPTH,
        load_frozen_trialgpt_retrieval,
        load_publication_contract,
        validate_frozen_retrieval_for_snapshot,
        validate_frozen_retrieval_for_task_input,
        validate_publication_preflight,
    )

    def required(name: str, expected: type) -> object:
        value = request.options.get(name)
        if not isinstance(value, expected):
            raise ValueError(f"TrialGPT publication requires {name}")
        return value

    publication_contract_path = cast(Path, required("publication_contract_path", Path))
    frozen_retrieval_path = cast(Path, required("frozen_retrieval_path", Path))
    frozen_retrieval_lock_path = cast(Path, required("frozen_retrieval_lock_path", Path))
    execution_workspace = cast(Path, required("workspace", Path))
    controller_git_commit = cast(str, required("controller_git_commit", str))
    retrieval_source_snapshot_id = cast(str, required("retrieval_source_snapshot_id", str))
    evaluation_package_id = cast(str, required("evaluation_package_id", str))
    if request.options.get("controller_working_tree_dirty") is not False:
        raise ValueError("TrialGPT publication requires a clean public release checkout")
    generation_workers = request.options.get("generation_workers", 8)
    if isinstance(generation_workers, bool) or not isinstance(generation_workers, int):
        raise ValueError("TrialGPT publication generation_workers must be an integer")
    codex_executable = request.options.get("codex_executable", "codex")
    if not isinstance(codex_executable, str) or not codex_executable:
        raise ValueError("TrialGPT publication codex_executable must be a non-empty string")

    retrieval = load_frozen_trialgpt_retrieval(
        frozen_retrieval_path, lock_path=frozen_retrieval_lock_path
    )
    validate_frozen_retrieval_for_snapshot(
        retrieval,
        benchmark_lineage=request.snapshot.benchmark_lineage,
        retrieval_source_snapshot_id=retrieval_source_snapshot_id,
        topic_ids=[topic.topic_id for topic in request.snapshot.topics],
        trial_ids=[trial.trial_id for trial in request.snapshot.trials],
    )
    validate_frozen_retrieval_for_task_input(
        retrieval,
        task_input_id=request.snapshot.task_input_id,
    )
    provider = CodexTrialGPTProvider(
        model=TRIALGPT_CODEX_MODEL,
        reasoning_effort=None,
        codex_executable=codex_executable,
    )
    provider_configuration = provider.provenance()
    contract = load_publication_contract(publication_contract_path)
    validate_publication_preflight(
        contract,
        taim_git_commit=controller_git_commit,
        working_tree_dirty=False,
        prepared_snapshot_id=request.snapshot.prepared_snapshot_id,
        evaluation_package_id=evaluation_package_id,
        frozen_retrieval=retrieval,
        provider_configuration=provider_configuration,
        method_configuration=trialgpt_publication_method_contract(),
        generation_workers=generation_workers,
        topic_ids=[topic.topic_id for topic in request.snapshot.topics],
    )
    system = TrialGPTTAIMLunaV1System(
        provider=provider,
        frozen_retrieval=retrieval,
        execution_workspace=execution_workspace,
    )
    effective_request = bind_system_capabilities(
        system,
        replace(
            request,
            options=MappingProxyType(
                {
                    "publication_contract_id": contract["contract_id"],
                    "frozen_retrieval_lock_sha256": retrieval.lock_sha256,
                    "frozen_retrieval_artifact_sha256": retrieval.artifact_sha256,
                    "provider_configuration": provider_configuration,
                    "generation_workers": generation_workers,
                    "retrieval_depth": 2_000,
                    "llm_candidate_depth": TRIALGPT_PUBLICATION_CANDIDATE_DEPTH,
                }
            ),
        ),
    )
    run_workspace = trialgpt_run_workspace(execution_workspace.resolve(), request.run_id)
    binding_path = run_workspace / "trialgpt-publication-cache-binding.json"
    binding = {
        "contract_id": contract["contract_id"],
        "run_id": request.run_id,
        "system_input_id": effective_request.system_input_id,
    }
    cache_path = run_workspace / "trialgpt-generation-cache.sqlite3"
    if cache_path.exists():
        if (
            not binding_path.is_file()
            or json.loads(binding_path.read_text(encoding="utf-8")) != binding
        ):
            raise ValueError("existing TrialGPT cache is not an in-place resume of this run")
    else:
        run_workspace.mkdir(parents=True, exist_ok=True)
        binding_path.write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    result = system.run(effective_request)
    return replace(
        result,
        configuration={
            **result.configuration,
            "system_input": effective_request.system_input_dict(),
        },
    )


def _evaluation_extension() -> RunEvaluationExtension:
    from taim.trialgpt_evaluation import TRIALGPT_EVALUATION_EXTENSION

    return TRIALGPT_EVALUATION_EXTENSION


EXTENSION = PipelineExtension(
    system_tasks={"TrialGPT-TAIM-Luna-v1": "patient_to_trial"},
    pipeline_depth="post_eligibility",
    external_baseline=True,
    subcommands={("patient-to-trial", "trialgpt"): "TrialGPT-TAIM-Luna-v1"},
    run=_run,
    add_run_options=_add_run_options,
    run_options=_run_options,
    add_patient_to_trial_commands=_add_patient_to_trial_commands,
    protocol_bindings=(
        (
            "src/taim/trialgpt_retrieval_producer.py",
            "FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256",
            "docs/paper-analysis-protocol-2026-09-02-v7.md",
        ),
    ),
    module_exports={
        "src/taim/adapters/__init__.py": ("TrialGPTTAIMLunaV1System",),
        "src/taim/adapters/trialgpt.py": (
            "TRIALGPT_CODEX_MODEL",
            "CodexTrialGPTProvider",
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
    forbidden_text={"src/taim/trialgpt_paper.py": ("MedCPT",)},
    protected_reference_paths=(
        "docs/external-baselines.md",
        "docs/paper-analysis-protocol-2026-09-02-v5.md",
        "docs/paper-analysis-protocol-2026-09-02-v6.md",
        "docs/trialgpt-external-baseline.md",
        "docs/trialgpt-retrieval-producer.md",
        "src/taim/adapters/__init__.py",
        "src/taim/pipeline_extensions/trialgpt.py",
        "src/taim/trialgpt_publication.py",
        "src/taim/trialgpt_retrieval_producer.py",
        "tests/test_trialgpt_external_baseline.py",
        "tests/test_trialgpt_retrieval_producer.py",
    ),
    dependency_declarations=("src/taim/data/external-system-dependencies-v1.json",),
    evaluation_profiles=("trialgpt-paper-style",),
    evaluation_extension=_evaluation_extension,
)
