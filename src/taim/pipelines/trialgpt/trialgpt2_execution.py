"""Resolve and execute ``TrialGPT2-TAIM-Luna-v1`` over its frozen retrieval, for rows and gates.

The System consumes ``TrialGPT-TAIM-Luna-v1``'s frozen retrieval lock exactly as that System's
publication run does: it loads and verifies every row, refuses a lock that does not bind protocol
v11 and carry its query-plan receipt, binds the lock to the Snapshot and the Task Input, and
refuses a lock frozen on another release than the one executing the run. The lock keeps its own
``system_id``; this System only reads it. Every input is verified before the first provider call.

A run then writes TAIM's inputs in their file formats, serves the bridge on loopback, and runs
their scorer unchanged with ``--resume``. An interrupted run resumes in place: running the same
command again with the same workspace continues where it stopped, and the bridge's response cache
answers every pair it already answered. A run is refused unless every candidate is scored and
every bridge request is accounted for.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from taim.artifacts import sha256_file
from taim.generation import GenerationCache, GenerationCacheRunLease
from taim.model_codex import codex_preflight
from taim.models import Model
from taim.patient_to_trial import StrictSystemOptions, bind_system_capabilities
from taim.pipelines.trialgpt.publication import (
    TRIALGPT_PUBLICATION_RETRIEVAL_STAGE,
    FrozenTrialGPTRetrieval,
    load_frozen_trialgpt_retrieval,
    validate_frozen_retrieval_for_release,
    validate_frozen_retrieval_for_snapshot,
    validate_frozen_retrieval_for_task_input,
)
from taim.pipelines.trialgpt.retrieval_producer import validate_query_plan_receipt_binding
from taim.pipelines.trialgpt.system import trialgpt_run_workspace
from taim.pipelines.trialgpt.trialgpt2_bridge import (
    BRIDGE_VERSION,
    SYSTEM_PREAMBLE,
    TrialGPT2Bridge,
    accounting_defects,
    ledger_accounting,
    read_ledger,
)
from taim.pipelines.trialgpt.trialgpt2_convert import (
    TRIALGPT2_INPUT_CONVERTER_VERSION,
    PromptBuilder,
    TrialGPT2ConversionError,
    expected_prompts,
    read_upstream_ranking,
    upstream_case,
    upstream_output_file,
    upstream_trial,
    write_upstream_inputs,
)
from taim.pipelines.trialgpt.trialgpt2_declaration import (
    BRIDGE_DISTORTIONS,
    CANDIDATE_DEPTH,
    GENERATION_WORKERS,
    MODEL_LABEL_SENT,
    MODEL_SERVED,
    TRIALGPT2_TAIM_LUNA_V1,
    TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID,
    UPSTREAM_OUTPUT_SCHEMA_SHA256,
    UPSTREAM_RUNNER_MODULE,
    UPSTREAM_SYSTEM_PROMPT_SHA256,
)
from taim.pipelines.trialgpt.trialgpt2_upstream import (
    UpstreamCheckout,
    load_prompt_builder,
    run_upstream,
    upstream_arguments,
    upstream_runtime,
    upstream_settings,
    verify_upstream_checkout,
)
from taim.schemas import Candidate, StageRanking, json_value_to_builtins
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
    CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
)

if TYPE_CHECKING:
    from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialRunResult

TRIALGPT2_SCORER_STAGE = "trialgpt2-scorer-ranking"
_BINDING_FILENAME = "trialgpt2-run-binding.json"
_INVOCATIONS_FILENAME = "trialgpt2-invocations.jsonl"


class TrialGPT2ExecutionError(ValueError):
    """Raised when a TrialGPT 2.0 run cannot start, resume or complete exactly."""


def trialgpt2_method_contract() -> dict[str, object]:
    """Everything about how this System runs their scorer that is the same for every run."""

    return {
        "method_version": "trialgpt2-taim-luna-v1",
        "upstream_runner": UPSTREAM_RUNNER_MODULE,
        "upstream_arguments": upstream_arguments("<generation_workers>"),
        "upstream_environment": upstream_settings(),
        "system_prompt_sha256": UPSTREAM_SYSTEM_PROMPT_SHA256,
        "output_schema_sha256": UPSTREAM_OUTPUT_SCHEMA_SHA256,
        "model_label_sent": MODEL_LABEL_SENT,
        "bridge": {
            "version": BRIDGE_VERSION,
            "joined_messages": f"{SYSTEM_PREAMBLE}\\n<system>\\n\\n<user>",
            "output_schema": "their strict json_schema, passed to codex exec as --output-schema",
            "codex_executions_per_request": 1,
            "response_cache": (
                "a schema-conforming answer, served again only to an identical request of the "
                "same run"
            ),
            "distortions": [distortion.name for distortion in BRIDGE_DISTORTIONS],
        },
        "input_converter": TRIALGPT2_INPUT_CONVERTER_VERSION,
        "candidate_depth": CANDIDATE_DEPTH,
        "ranking": (
            "their final_score (score + confidence) descending, ties by trial ID; the Primary "
            "Ranking is the first top_k"
        ),
    }


@dataclass(frozen=True, slots=True)
class ResolvedTrialGPT2Inputs:
    """Every physical input of a run, verified before any provider call."""

    retrieval: FrozenTrialGPTRetrieval
    upstream: UpstreamCheckout
    build_messages: PromptBuilder
    model_configuration: Mapping[str, object]
    provider_configuration: Mapping[str, object]
    upstream_runtime: Mapping[str, str]
    release_identity: Mapping[str, str]
    workspace: Path
    generation_workers: int
    python: str


class TrialGPT2TAIMLunaV1System(StrictSystemOptions):
    """TrialGPT 2.0's scorer re-ranking the top 500 of TrialGPT-TAIM-Luna-v1's retrieval."""

    system_id = TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID
    declaration = TRIALGPT2_TAIM_LUNA_V1
    option_names = frozenset(
        {
            "frozen_retrieval_lock_sha256",
            "frozen_retrieval_artifact_sha256",
            "provider_configuration",
            "upstream",
            "upstream_runtime",
            "method",
            "generation_workers",
            "llm_candidate_depth",
            "qualification_topic_ids",
        }
    )
    required_capabilities = frozenset(
        {
            CAPABILITY_CANONICAL_PATIENT_TEXT,
            CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
            CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
        }
    )
    optional_capabilities: frozenset[str] = frozenset()

    def __init__(self, resolved: ResolvedTrialGPT2Inputs | None = None) -> None:
        self.resolved = resolved

    def run(self, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
        if self.resolved is None:
            raise TrialGPT2ExecutionError(
                "TrialGPT2-TAIM-Luna-v1 runs only from inputs verified by run_trialgpt2"
            )
        return _score(self.resolved, request)


def _required(request: PatientToTrialRunRequest, name: str, expected: type) -> object:
    value = request.options.get(name)
    if not isinstance(value, expected):
        raise TrialGPT2ExecutionError(f"TrialGPT 2.0 requires {name}")
    return value


def resolve_trialgpt2(
    request: PatientToTrialRunRequest,
    *,
    qualification_topic_ids: Sequence[str] | None = None,
) -> tuple[ResolvedTrialGPT2Inputs, PatientToTrialRunRequest]:
    """Verify every physical input and bind them into one content-bound effective request."""

    frozen_retrieval_path = cast(Path, _required(request, "frozen_retrieval_path", Path))
    lock_path = cast(Path, _required(request, "frozen_retrieval_lock_path", Path))
    workspace = cast(Path, _required(request, "workspace", Path))
    checkout = cast(Path, _required(request, "upstream_checkout", Path))
    source_snapshot_id = cast(str, _required(request, "retrieval_source_snapshot_id", str))
    executing_release = request.options.get("executing_release_identity")
    generation_workers = request.options.get("generation_workers", GENERATION_WORKERS)
    if (
        isinstance(generation_workers, bool)
        or not isinstance(generation_workers, int)
        or not 1 <= generation_workers <= GENERATION_WORKERS
    ):
        raise TrialGPT2ExecutionError("TrialGPT 2.0 generation_workers must be from 1 to 48")
    codex_executable = request.options.get("codex_executable", "codex")
    if not isinstance(codex_executable, str) or not codex_executable:
        raise TrialGPT2ExecutionError("TrialGPT 2.0 codex_executable must be a non-empty string")

    retrieval = load_frozen_trialgpt_retrieval(frozen_retrieval_path, lock_path=lock_path)
    validate_query_plan_receipt_binding(retrieval.lock)
    validate_frozen_retrieval_for_snapshot(
        retrieval,
        benchmark_lineage=request.snapshot.benchmark_lineage,
        retrieval_source_snapshot_id=source_snapshot_id,
        topic_ids=[topic.topic_id for topic in request.snapshot.topics],
        trial_ids=[trial.trial_id for trial in request.snapshot.trials],
    )
    validate_frozen_retrieval_for_task_input(
        retrieval, task_input_id=request.snapshot.task_input_id
    )
    release_identity = (
        cast(Mapping[str, str], executing_release)
        if isinstance(executing_release, Mapping)
        else None
    )
    validate_frozen_retrieval_for_release(retrieval, release_identity=release_identity)
    topic_ids = [topic.topic_id for topic in request.snapshot.topics]
    if qualification_topic_ids is not None and (
        not qualification_topic_ids
        or len(set(qualification_topic_ids)) != len(qualification_topic_ids)
        or not set(qualification_topic_ids) <= set(topic_ids)
    ):
        raise TrialGPT2ExecutionError("qualification topics must be distinct Task Input topics")

    upstream = verify_upstream_checkout(checkout)
    build_messages = load_prompt_builder(upstream)
    runtime = upstream_runtime()
    model = Model(
        model_id=MODEL_SERVED,
        backend="codex",
        parameters={"reasoning_effort": None},
        codex_executable=codex_executable,
    )
    model_configuration = cast(Mapping[str, object], json_value_to_builtins(model.configuration))
    provider_configuration = codex_preflight(model.configuration)
    options: dict[str, object] = {
        "frozen_retrieval_lock_sha256": retrieval.lock_sha256,
        "frozen_retrieval_artifact_sha256": retrieval.artifact_sha256,
        "provider_configuration": provider_configuration,
        "upstream": upstream.identity(),
        "upstream_runtime": runtime,
        "method": trialgpt2_method_contract(),
        "generation_workers": generation_workers,
        "llm_candidate_depth": CANDIDATE_DEPTH,
    }
    if qualification_topic_ids is not None:
        options["qualification_topic_ids"] = list(qualification_topic_ids)
    system = TrialGPT2TAIMLunaV1System()
    effective = bind_system_capabilities(
        system,
        replace(request, options=MappingProxyType(system.normalize_options(options))),
    )
    resolved = ResolvedTrialGPT2Inputs(
        retrieval=retrieval,
        upstream=upstream,
        build_messages=build_messages,
        model_configuration=model_configuration,
        provider_configuration=provider_configuration,
        upstream_runtime=runtime,
        release_identity=cast(Mapping[str, str], release_identity),
        workspace=workspace.resolve(),
        generation_workers=generation_workers,
        python=sys.executable,
    )
    return resolved, effective


def run_trialgpt2(system_id: str, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
    """Run one row: every topic of the Task Input, over its frozen top 500."""

    if system_id != TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID:
        raise TrialGPT2ExecutionError(f"System {system_id!r} is not TrialGPT2-TAIM-Luna-v1")
    resolved, effective = resolve_trialgpt2(request)
    result = TrialGPT2TAIMLunaV1System(resolved=resolved).run(effective)
    return replace(
        result,
        configuration={
            **result.configuration,
            "system_input": effective.system_input_record_dict(),
        },
    )


def _bind_workspace(run_workspace: Path, request: PatientToTrialRunRequest) -> int:
    """Refuse a workspace that is not this run's; the number of this invocation in it."""

    binding_path = run_workspace / _BINDING_FILENAME
    binding = {"run_id": request.run_id, "system_input_id": request.system_input_id}
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
            raise TrialGPT2ExecutionError(
                "existing TrialGPT 2.0 workspace is not an in-place resume of this run"
            )
    elif run_workspace.exists() and any(run_workspace.iterdir()):
        raise TrialGPT2ExecutionError(f"TrialGPT 2.0 run workspace is not empty: {run_workspace}")
    else:
        run_workspace.mkdir(parents=True, exist_ok=True)
        binding_path.write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    invocations = run_workspace / _INVOCATIONS_FILENAME
    invocation = (
        len(invocations.read_text(encoding="utf-8").splitlines()) + 1 if invocations.exists() else 1
    )
    with invocations.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "invocation": invocation,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            + "\n"
        )
    return invocation


def _set_aside_incomplete_outputs(
    directory: Path,
    candidates: Mapping[str, Sequence[str]],
    invocation: int,
) -> list[dict[str, str]]:
    """Move aside an output an interruption left incomplete, so their resume redoes that topic.

    Their runner writes a patient's file in place when the patient finishes, and its resume skips
    any patient whose file exists, so a file cut short by an interruption would never be redone.
    """

    set_aside: list[dict[str, str]] = []
    for topic_id, trial_ids in candidates.items():
        path = upstream_output_file(directory, topic_id)
        if not path.exists():
            continue
        try:
            read_upstream_ranking(directory, topic_id, trial_ids)
        except TrialGPT2ConversionError as exc:
            target = path.with_name(f"{path.name}.incomplete-{invocation}")
            path.rename(target)
            set_aside.append({"topic_id": topic_id, "path": str(target), "reason": str(exc)})
    return set_aside


def _score(
    resolved: ResolvedTrialGPT2Inputs, request: PatientToTrialRunRequest
) -> PatientToTrialRunResult:
    from taim.patient_to_trial import PatientToTrialRunResult

    started = time.perf_counter()
    qualification = request.options.get("qualification_topic_ids")
    selected = set(cast(Sequence[str], qualification)) if qualification is not None else None
    topics = [
        topic for topic in request.snapshot.topics if selected is None or topic.topic_id in selected
    ]
    run_workspace = trialgpt_run_workspace(resolved.workspace, request.run_id)
    invocation = _bind_workspace(run_workspace, request)

    retrieval_rows = {
        topic.topic_id: resolved.retrieval.selected_rows(topic.topic_id) for topic in topics
    }
    candidate_ids = {
        topic_id: [row.trial_id for row in rows] for topic_id, rows in retrieval_rows.items()
    }
    trials_by_id = {trial.trial_id: trial for trial in request.snapshot.trials}
    trials = {
        trial_id: upstream_trial(trials_by_id[trial_id])
        for trial_ids in candidate_ids.values()
        for trial_id in trial_ids
    }
    cases = [upstream_case(topic, candidate_ids[topic.topic_id]) for topic in topics]
    input_directory = run_workspace / "upstream-input"
    inputs = write_upstream_inputs(input_directory, cases, trials)
    prompts = expected_prompts(resolved.build_messages, cases, trials)
    output_directory = run_workspace / "upstream-output"
    set_aside = _set_aside_incomplete_outputs(output_directory, candidate_ids, invocation)

    cache_path = run_workspace / "bridge-cache.sqlite3"
    ledger_path = run_workspace / "bridge-ledger.jsonl"
    log_path = run_workspace / "upstream.log"
    cache = GenerationCache(cache_path)
    try:
        with (
            GenerationCacheRunLease(cache_path, run_id=request.run_id),
            TrialGPT2Bridge(
                model_configuration=resolved.model_configuration,
                cache=cache,
                ledger=ledger_path,
                expected_prompts=prompts,
                label=MODEL_LABEL_SENT,
                invocation=invocation,
            ) as bridge,
        ):
            status = run_upstream(
                resolved.python,
                resolved.upstream,
                cases=input_directory / "benchmark_cases.jsonl",
                trials=input_directory / "trials",
                output=output_directory,
                generation_workers=resolved.generation_workers,
                base_url=bridge.base_url,
                log=log_path,
            )
    finally:
        cache.close()
    if status != 0:
        raise TrialGPT2ExecutionError(
            f"TrialGPT 2.0's scorer exited with status {status}; its log is {log_path}. Run the "
            "same command again with the same workspace to resume in place"
        )

    rankings = {
        topic_id: read_upstream_ranking(output_directory, topic_id, trial_ids)
        for topic_id, trial_ids in candidate_ids.items()
    }
    pairs = [(topic_id, trial_id) for topic_id, ids in candidate_ids.items() for trial_id in ids]
    accounting = ledger_accounting(read_ledger(ledger_path), pairs)
    defects = accounting_defects(accounting)
    if defects:
        raise TrialGPT2ExecutionError(
            "TrialGPT 2.0 bridge requests are not all accounted for: " + "; ".join(defects)
        )

    retrieval_candidates: list[Candidate] = []
    scorer_candidates: list[Candidate] = []
    primary: list[Candidate] = []
    system_id = TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID
    for topic in topics:
        topic_id = topic.topic_id
        retrieval_candidates.extend(
            Candidate(request.run_id, system_id, topic_id, row.trial_id, row.rank, row.score)
            for row in retrieval_rows[topic_id]
        )
        ranked = [
            Candidate(request.run_id, system_id, topic_id, row.trial_id, rank, row.final_score)
            for rank, row in enumerate(rankings[topic_id], start=1)
        ]
        scorer_candidates.extend(ranked)
        primary.extend(ranked[: request.top_k])

    provider_transport = resolved.provider_configuration.get("transport")
    transport = provider_transport if isinstance(provider_transport, Mapping) else {}
    declaration = TRIALGPT2_TAIM_LUNA_V1
    runtime_seconds = time.perf_counter() - started
    configuration: dict[str, object] = {
        "declaration": {
            "first_stage": declaration.first_stage,
            "license_expression": declaration.license_expression,
            "bridge_distortions": [
                {"name": item.name, "what": item.what} for item in declaration.bridge_distortions
            ],
            "comparability": {
                "comparable_to": sorted(declaration.comparability.comparable_to),
                "statement": declaration.comparability.statement,
                "caption_note": declaration.comparability.caption_note,
            },
        },
        "model_identity": {
            "kind": "trialgpt2_codex_bridge",
            "provider_id": resolved.provider_configuration.get("provider_id"),
            "model": resolved.provider_configuration.get("model"),
            "reasoning_effort": resolved.provider_configuration.get("reasoning_effort"),
            "provider_version": transport.get("codex_version"),
            "model_provenance": transport.get("model_provenance"),
            "model_label_sent": MODEL_LABEL_SENT,
            "bridge_version": BRIDGE_VERSION,
        },
        "index_identity": {
            "kind": "trialgpt_frozen_retrieval",
            "persistent": True,
            "task_input_id": request.snapshot.task_input_id,
            "source_snapshot_id": resolved.retrieval.source_snapshot_id,
            **resolved.retrieval.provenance([topic.topic_id for topic in topics]),
        },
        "frozen_retrieval": resolved.retrieval.provenance([topic.topic_id for topic in topics]),
        "executing_release_identity": dict(resolved.release_identity),
        "provider_configuration": dict(resolved.provider_configuration),
        "upstream": resolved.upstream.identity(),
        "upstream_runtime": dict(resolved.upstream_runtime),
        "method": trialgpt2_method_contract(),
        "generation_workers": resolved.generation_workers,
        "llm_candidate_depth": CANDIDATE_DEPTH,
        "topic_ids": [topic.topic_id for topic in topics],
        "run_workspace": str(run_workspace),
        "invocation": invocation,
        "interrupted_outputs_set_aside": set_aside,
        "upstream_input": inputs,
        "upstream_output": {
            "path": str(output_directory),
            "files": {
                topic_id: sha256_file(upstream_output_file(output_directory, topic_id))
                for topic_id in candidate_ids
            },
        },
        "bridge_ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
        "bridge_accounting": accounting,
        "generation_cache": {"path": str(cache_path)},
        "upstream_log": {"path": str(log_path)},
        "runtime_seconds": runtime_seconds,
    }
    return PatientToTrialRunResult(
        candidates=tuple(primary),
        configuration=configuration,
        runtime_seconds=runtime_seconds,
        primary_ranking="candidates",
        pipeline_depth="post_eligibility",
        stage_rankings=(
            StageRanking(
                TRIALGPT_PUBLICATION_RETRIEVAL_STAGE, "retrieval", tuple(retrieval_candidates)
            ),
            StageRanking(TRIALGPT2_SCORER_STAGE, "post_eligibility", tuple(scorer_candidates)),
        ),
    )


__all__ = [
    "TRIALGPT2_SCORER_STAGE",
    "ResolvedTrialGPT2Inputs",
    "TrialGPT2ExecutionError",
    "TrialGPT2TAIMLunaV1System",
    "resolve_trialgpt2",
    "run_trialgpt2",
    "trialgpt2_method_contract",
]
