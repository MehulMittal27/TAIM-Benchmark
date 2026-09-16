"""The TrialGPT-TAIM-Luna-v1 publication System.

The System selects each topic's candidate pool from a frozen three-Luna consensus ranking, then
preserves the pinned TrialGPT matching, aggregation, and score arithmetic while injecting a
provider-neutral generation protocol.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast

from taim.artifacts import sha256_file
from taim.eligibility import (
    ELIGIBILITY_CRITERION_VIEW_CAPABILITY,
    find_eligibility_criterion_view,
)
from taim.schemas import Candidate, StageRanking
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
    CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
    CAPABILITY_TYPED_PATIENT_CORE,
)
from taim.system_contracts import StrictSystemOptions
from taim.trialgpt_codex import CodexProvider
from taim.trialgpt_codex import schema_hash as trialgpt_schema_hash
from taim.trialgpt_generation import (
    GenerationCache,
    GenerationRequest,
    GenerationResult,
    ProviderResponse,
    Validation,
    canonical_json,
    provider_failure_classification,
    run_generation,
)
from taim.trialgpt_paper import (
    TRIALGPT_LLM_CANDIDATE_DEPTH,
    TRIALGPT_TOKENIZER_ID,
    TrialGPTTrialView,
    aggregation_prompt,
    final_score,
    matching_prompt,
    matching_score,
    normalize_matching_output,
    numbered_patient,
    patient_sentences,
    trial_view,
    validate_aggregation_output,
    validate_matching_output,
)
from taim.trialgpt_publication import (
    TRIALGPT_PUBLICATION_LOGICAL_CALLS_PER_TOPIC,
    TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
    FrozenTrialGPTRetrieval,
    TrialGPTPublicationError,
)

if TYPE_CHECKING:
    from taim.system_contracts import SystemRunRequest, SystemRunResult


TRIALGPT_REQUIRED_CAPABILITIES = frozenset(
    {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_SEMANTIC_TRIAL_SECTIONS}
)
TRIALGPT_OPTIONAL_CAPABILITIES = frozenset({CAPABILITY_TYPED_PATIENT_CORE})
TRIALGPT_PAPER_RUNTIME_OPTION_PROFILE = "trialgpt-paper"
TRIALGPT_OPTION_NAMES = frozenset({"workspace", "device", "precision", "retrieval_depth"})
TRIALGPT_PAPER_OPTION_NAMES = TRIALGPT_OPTION_NAMES | frozenset(
    {
        "batch_size",
        "model_cache_dir",
        "embedding_cache_dir",
        "expected_embedding_sha256",
        "generation_workers",
    }
)
TRIALGPT_PAPER_PROMPT_VERSION = "trialgpt-paper-faithful-prompts-v3"
TRIALGPT_CODEX_MODEL = "gpt-5.6-luna"
TRIALGPT_FULL_GENERATION_WORKERS = 8
TRIALGPT_MAX_GENERATION_WORKERS = 48
TRIALGPT_GENERATION_CACHE_FLUSH_EVERY = 1
TRIALGPT_PROVENANCE_VERSION = "trialgpt-provenance-v3"
TRIALGPT_ENVIRONMENT_POLICY_VERSION = "trialgpt-safe-environment-v1"
TRIALGPT_FAILURE_AWARE_POLICY_VERSION = "trialgpt-failure-aware-abstention-v2"
TRIALGPT_MATCHING_NORMALIZER_VERSION = "trialgpt-matching-normalizer-v2"
TRIALGPT_MATCHING_OUTPUT_CONTRACT_VERSION = "trialgpt-matching-output-contract-v2"
TRIALGPT_ABSTAIN_AFTER_RETRY_EXHAUSTION = "abstain_after_retry_exhaustion"
TRIALGPT_SAFE_ENVIRONMENT_VARIABLES = frozenset(
    {
        "PATH",
        "HOME",
        "CODEX_HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "TRIALGPT_RETRIEVAL_DEPTH",
        "TRIALGPT_GENERATION_WORKERS",
    }
)
TRIALGPT_ENVIRONMENT_VARIABLE_PREFIXES = ("TAIM_", "TRIALGPT_")
_TaskT = TypeVar("_TaskT")
_ResultT = TypeVar("_ResultT")
PAPER_MATCHING_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "object",
            "properties": {},
            "patternProperties": {
                "^[0-9]+$": {
                    "type": "array",
                    "prefixItems": [
                        {"type": "string"},
                        {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                        {"type": "string"},
                    ],
                    "minItems": 3,
                    "maxItems": 3,
                    # Draft 7 spelling; maxItems closes the tuple for 2020-12 validators.
                    "additionalItems": False,
                },
            },
            "additionalProperties": False,
        },
    },
    "required": ["criteria"],
    "additionalProperties": False,
}
PAPER_AGGREGATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "relevance_explanation": {"type": "string"},
        "relevance_score_R": {"type": "number"},
        "eligibility_explanation": {"type": "string"},
        "eligibility_score_E": {"type": "number"},
    },
    "required": [
        "relevance_explanation",
        "relevance_score_R",
        "eligibility_explanation",
        "eligibility_score_E",
    ],
    "additionalProperties": False,
}


class TrialGPTAdapterError(ValueError):
    """Raised when a TrialGPT Snapshot or generated artifact is invalid."""


class TrialGPTProvider(Protocol):
    """Provider-neutral generation seam the TrialGPT stages call."""

    provider_id: str

    def generate(
        self,
        *,
        stage: str,
        prompt: str,
        output_schema: Mapping[str, object],
        payload: Mapping[str, object],
    ) -> TrialGPTGeneration: ...


class TrialGPTGeneration:
    """One normalized provider response and its raw trace."""

    def __init__(
        self,
        *,
        stage: str,
        prompt: str,
        output: Mapping[str, object],
        raw_response: str,
        provider_id: str,
        metadata: Mapping[str, object] | None = None,
        usage: Mapping[str, object] | None = None,
    ) -> None:
        self.stage = stage
        self.prompt = prompt
        self.output = dict(output)
        self.raw_response = raw_response
        self.provider_id = provider_id
        self.metadata = dict(metadata or {})
        self.usage = dict(usage or {})
        self.prompt_sha256 = _sha256_text(prompt)
        self.output_sha256 = _sha256_text(_canonical_json(output))

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "provider_id": self.provider_id,
            "prompt_sha256": self.prompt_sha256,
            "output_sha256": self.output_sha256,
            "raw_response": self.raw_response,
            "output": self.output,
            "metadata": self.metadata,
            "usage": self.usage,
        }


class _EmptyEligibilityProvider:
    """Record an empty matching result without asking a model to invent criteria."""

    provider_id = "trialgpt-empty-eligibility-v1"

    def generate(
        self,
        *,
        stage: str,
        prompt: str,
        output_schema: Mapping[str, object],
        payload: Mapping[str, object],
    ) -> TrialGPTGeneration:
        del output_schema, payload
        output: dict[str, object] = {"criteria": {}}
        return TrialGPTGeneration(
            stage=stage,
            prompt=prompt,
            output=output,
            raw_response=_canonical_json(output),
            provider_id=self.provider_id,
            metadata={
                "provider_call": {
                    "status": "not_called",
                    "reason": "no_usable_eligibility_criteria",
                }
            },
        )


class CodexTrialGPTProvider:
    """Adapt the isolated Codex transport to TrialGPT's generation seam."""

    provider_id = "codex-cli-v1"

    def __init__(
        self,
        *,
        model: str | None = TRIALGPT_CODEX_MODEL,
        reasoning_effort: str | None = None,
        codex_executable: str = "codex",
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.transport = CodexProvider(
            codex_executable=codex_executable,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    def provenance(self) -> dict[str, object]:
        """Return reproducibility settings without copying the provider environment."""

        return {
            "provider_id": self.provider_id,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "transport": self.transport.provenance(),
        }

    def generate(
        self,
        *,
        stage: str,
        prompt: str,
        output_schema: Mapping[str, object],
        payload: Mapping[str, object],
    ) -> TrialGPTGeneration:
        response = self.transport.generate(
            prompt=prompt,
            output_schema=output_schema,
            stage=stage,
            logical_call_id=":".join(
                str(payload.get(key, stage)) for key in ("topic_id", "trial_id", "kind")
            ),
            input_hash=_sha256_text(_canonical_json(payload)),
        )
        if not isinstance(response, ProviderResponse):
            raise TrialGPTAdapterError("Codex transport returned an invalid provider response")
        metadata = dict(response.metadata)
        metadata["provider_call"] = {"status": "called"}
        return TrialGPTGeneration(
            stage=stage,
            prompt=prompt,
            output={},
            raw_response=response.raw_output,
            provider_id=self.provider_id,
            metadata=metadata,
            usage=response.usage,
        )


class _LegacyProviderBridge:
    """Adapt a TrialGPT provider to the provider-neutral generation protocol."""

    def __init__(
        self,
        provider: TrialGPTProvider,
        *,
        stage: str,
        payload: Mapping[str, object],
    ) -> None:
        self.provider = provider
        self.stage = stage
        self.payload = payload
        self.provider_id = provider.provider_id

    def generate(
        self,
        *,
        prompt: str,
        output_schema: Mapping[str, object],
        attempt: int,
        logical_call_id: str,
    ) -> ProviderResponse:
        del attempt, logical_call_id
        response = self.provider.generate(
            stage=self.stage,
            prompt=prompt,
            output_schema=output_schema,
            payload=self.payload,
        )
        return ProviderResponse(
            response.raw_response,
            usage=response.usage,
            metadata=response.metadata,
        )


def _run_generation_stage(
    *,
    provider: TrialGPTProvider,
    cache: GenerationCache,
    stage: str,
    prompt: str,
    output_schema: Mapping[str, object],
    payload: Mapping[str, object],
    validator: Validation,
    normalizer: Callable[[object], object] | None = None,
    normalizer_version: str | None = None,
    allow_unresolved: bool = False,
) -> GenerationResult:
    provider_config = _provider_provenance(provider)
    result = run_generation(
        GenerationRequest(
            stage=stage,
            prompt=prompt,
            output_schema=output_schema,
            input_payload=payload,
            provider_config=provider_config,
            validator=validator,
            normalizer=normalizer,
            normalizer_version=normalizer_version,
        ),
        _LegacyProviderBridge(provider, stage=stage, payload=payload),
        cache=cache,
        reuse_unresolved=allow_unresolved,
    )
    if not result.resolved and not allow_unresolved:
        raise TrialGPTAdapterError(
            f"TrialGPT generation unresolved at {stage}: "
            + "; ".join(result.attempts[-1].validation_errors)
        )
    return result


def _validate_paper_matching_output(
    value: object,
    *,
    kind: str,
    criterion_count: int | None = None,
    patient_sentence_count: int | None = None,
) -> tuple[str, ...]:
    selected = value.get("criteria", value) if isinstance(value, Mapping) else value
    errors = list(
        validate_matching_output(
            selected,
            kind=kind,
            patient_sentence_count=patient_sentence_count,
        )
    )
    if criterion_count is not None and isinstance(selected, Mapping):
        expected_ids = {str(index) for index in range(criterion_count)}
        missing_ids = sorted(expected_ids - set(selected), key=int)
        if missing_ids:
            errors.append(f"missing criterion IDs: {', '.join(missing_ids)}")
        for criterion_id in selected:
            if (
                isinstance(criterion_id, str)
                and criterion_id.isdigit()
                and int(criterion_id) >= criterion_count
            ):
                errors.append(f"criterion {criterion_id} is outside the trial criteria")
    return tuple(errors)


def _ordered_generation_map(
    tasks: list[_TaskT],
    operation: Callable[[_TaskT], _ResultT],
    *,
    max_workers: int,
) -> list[_ResultT]:
    """Run generation tasks concurrently while returning results in task order."""

    if max_workers == 1:
        return [operation(task) for task in tasks]
    if not tasks:
        return []
    results: list[_ResultT | None] = [None] * len(tasks)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    pending: dict[Future[_ResultT], int] = {}
    next_index = 0
    try:
        while next_index < min(max_workers, len(tasks)):
            pending[executor.submit(operation, tasks[next_index])] = next_index
            next_index += 1
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            failure = None
            for future in completed:
                failure = future.exception()
                if failure is not None:
                    break
            if failure is not None:
                for future in pending:
                    future.cancel()
                raise failure
            for future in completed:
                index = pending.pop(future)
                results[index] = future.result()
            while next_index < len(tasks) and len(pending) < max_workers:
                pending[executor.submit(operation, tasks[next_index])] = next_index
                next_index += 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return [cast(_ResultT, result) for result in results]


_canonical_json = canonical_json


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_environment_provenance() -> dict[str, object]:
    """Record effective non-secret environment inputs under a versioned allowlist policy."""

    variables: dict[str, str] = {}
    redacted_keys: list[str] = []
    for key, value in sorted(os.environ.items()):
        if key in TRIALGPT_SAFE_ENVIRONMENT_VARIABLES:
            variables[key] = value
        elif key.startswith(TRIALGPT_ENVIRONMENT_VARIABLE_PREFIXES):
            redacted_keys.append(key)
    return {
        "policy_version": TRIALGPT_ENVIRONMENT_POLICY_VERSION,
        "variables": variables,
        "redacted_keys": redacted_keys,
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
    }


def _provider_provenance(provider: TrialGPTProvider) -> dict[str, object]:
    provenance = getattr(provider, "provenance", None)
    if callable(provenance):
        value = provenance()
        if isinstance(value, Mapping):
            return dict(value)
    return {
        "provider_id": provider.provider_id,
        "model": getattr(provider, "model", None),
        "reasoning_effort": getattr(provider, "reasoning_effort", None),
    }


#: Provider configuration fields deliberately left out of a TrialGPT ``model_identity``, each
#: with its reason. Recorded inside the identity so a reader of a published record meets a
#: decision, not a gap.
TRIALGPT_MODEL_IDENTITY_OMITTED_FIELDS: Mapping[str, str] = {
    "codex_executable": (
        "excluded from model_identity because an executable location is not a property of "
        "the model; where a Codex transport is used, the value remains in the stored run "
        "manifest under provider_configuration.transport.codex_executable and, for "
        "TrialGPT-TAIM-Luna-v1, in the declared System Input options"
    ),
}


def _trialgpt_model_identity(provider_configuration: Mapping[str, object]) -> dict[str, object]:
    """Identify the models a TrialGPT run generated with.

    The release writer requires ``model_identity`` so a result traces back to what scored
    it. Every generation call goes through the one provider whose provenance the run
    records, so the identity is that provenance: the requested model, the provider's
    reported version, and its model provenance (for Codex, the pinned catalog entry and
    its identity digest). This System ranks from frozen retrieval and runs no dense model;
    the retrieval producers are identified by the frozen retrieval lock in
    ``index_identity``. The executable location is caller-local and stays in
    ``provider_configuration`` only, because the public projection keeps every key that is
    not path-shaped.
    """

    raw_transport = provider_configuration.get("transport")
    transport = raw_transport if isinstance(raw_transport, Mapping) else {}
    return {
        "kind": "trialgpt_generation_provider",
        "provider_id": provider_configuration.get("provider_id"),
        "model": provider_configuration.get("model"),
        "reasoning_effort": provider_configuration.get("reasoning_effort"),
        "provider_version": transport.get("codex_version"),
        "model_provenance": transport.get("model_provenance"),
        "tokenizer_id": TRIALGPT_TOKENIZER_ID,
        # Frozen retrieval loads no dense model in the run, so there is none to identify.
        "dense_retrieval_model": None,
        # transport.codex_executable is excluded on purpose: an executable location is not a
        # property of the model. The statement travels with the identity so the absence reads
        # as a decision. It does not keep the value out of the published System Input options.
        "omitted_fields": dict(TRIALGPT_MODEL_IDENTITY_OMITTED_FIELDS),
    }


def _trialgpt_index_identity(
    request: SystemRunRequest, *, frozen_retrieval: FrozenTrialGPTRetrieval
) -> dict[str, object]:
    """Identify what first-level retrieval searched.

    The publication System selects its candidate pool from a frozen three-Luna consensus
    ranking produced offline and reused, so its index is that ranking: the lock and
    artifact digests, the per-topic top-500 digests the run selected, and the Snapshot
    and Task Input it was produced against.
    """

    return {
        "kind": "trialgpt_frozen_retrieval",
        "persistent": True,
        "task_input_id": request.snapshot.task_input_id,
        "source_snapshot_id": frozen_retrieval.source_snapshot_id,
        **frozen_retrieval.provenance([topic.topic_id for topic in request.snapshot.topics]),
    }


def trialgpt_publication_method_contract() -> dict[str, object]:
    """Return exact prompt, schema, normalization, scoring, and tokenizer identities."""

    implementation_sources = {
        function.__name__: inspect.getsource(function)
        for function in (
            aggregation_prompt,
            final_score,
            matching_prompt,
            matching_score,
            normalize_matching_output,
            numbered_patient,
            patient_sentences,
        )
    }
    return {
        "prompt_version": TRIALGPT_PAPER_PROMPT_VERSION,
        "prompt_and_scoring_implementation_sha256": "sha256:"
        + hashlib.sha256(_canonical_json(implementation_sources).encode("utf-8")).hexdigest(),
        "matching_schema_sha256": trialgpt_schema_hash(PAPER_MATCHING_SCHEMA),
        "aggregation_schema_sha256": trialgpt_schema_hash(PAPER_AGGREGATION_SCHEMA),
        "failure_policy_version": TRIALGPT_FAILURE_AWARE_POLICY_VERSION,
        "matching_normalizer_version": TRIALGPT_MATCHING_NORMALIZER_VERSION,
        "matching_output_contract_version": TRIALGPT_MATCHING_OUTPUT_CONTRACT_VERSION,
        "tokenizer_id": TRIALGPT_TOKENIZER_ID,
        "retrieval_stage": "three-luna-consensus",
    }


def trialgpt_workspace_key(value: str) -> str:
    """Return the stable filesystem key used by TrialGPT run and topic workspaces."""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "value"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:64]}-{digest}"


def _topic_workspace_key(topic_id: str) -> str:
    return trialgpt_workspace_key(topic_id)


def trialgpt_run_workspace(workspace: Path, run_id: str) -> Path:
    """Return an isolated artifact root for one run in a reusable workspace."""
    return workspace / "runs" / trialgpt_workspace_key(run_id)


def _write_json_artifact(path: Path, payload: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"path": str(path), "sha256": sha256_file(path)}


def _write_topic_provenance(
    *,
    workspace: Path,
    run_id: str,
    system_id: str,
    prepared_snapshot_id: str,
    task_input_id: str,
    system_input_id: str,
    topic_id: str,
    traces: list[dict[str, object]],
    prompt_audit: list[dict[str, object]],
    retrieval_candidates: list[Candidate],
    matching_artifacts: list[dict[str, object]],
    aggregation_artifacts: list[dict[str, object]],
    final_candidates: list[Candidate],
    abstention_artifacts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    workspace_key = _topic_workspace_key(topic_id)
    topic_workspace = workspace / "topics" / workspace_key
    artifacts = {
        "generation_trace": _write_json_artifact(
            topic_workspace / "trialgpt-generation-traces.json", traces
        ),
        "prompt_audit": _write_json_artifact(
            topic_workspace / "trialgpt-prompt-audit.json", prompt_audit
        ),
        "retrieval_stage": _write_json_artifact(
            topic_workspace / "trialgpt-hybrid-retrieval.json",
            [candidate.to_dict() for candidate in retrieval_candidates],
        ),
        "criterion_matching": _write_json_artifact(
            topic_workspace / "trialgpt-criterion-matching.json", matching_artifacts
        ),
        "aggregation": _write_json_artifact(
            topic_workspace / "trialgpt-aggregation.json", aggregation_artifacts
        ),
        "primary_ranking_artifact": _write_json_artifact(
            topic_workspace / "trialgpt-final-ranking.json",
            [candidate.to_dict() for candidate in final_candidates],
        ),
    }
    counts = {
        "generation_trace": len(traces),
        "prompt_audit": len(prompt_audit),
        "retrieval_candidates": len(retrieval_candidates),
        "criterion_matching": len(matching_artifacts),
        "aggregation": len(aggregation_artifacts),
        "primary_ranking": len(final_candidates),
    }
    if abstention_artifacts is not None:
        artifacts["abstentions"] = _write_json_artifact(
            topic_workspace / "trialgpt-abstentions.json", abstention_artifacts
        )
        counts["abstentions"] = len(abstention_artifacts)
    return {
        "run_id": run_id,
        "system_id": system_id,
        "prepared_snapshot_id": prepared_snapshot_id,
        "task_input_id": task_input_id,
        "system_input_id": system_input_id,
        "topic_id": topic_id,
        "workspace": str(topic_workspace),
        "workspace_key": workspace_key,
        "artifacts": artifacts,
        "counts": counts,
    }


def _trace_record(
    result: GenerationResult, *, topic_id: str, trial_id: str | None = None
) -> dict[str, object]:
    record = result.to_dict()
    record["topic_id"] = topic_id
    if trial_id is not None:
        record["trial_id"] = trial_id
    return record


def _option(request: SystemRunRequest, name: str, default: object = None) -> object:
    return request.options.get(name, default)


class TrialGPTTAIMLunaV1System(StrictSystemOptions):
    """Publication System with the issue-22 three-Luna consensus frozen at top 500.

    It runs TrialGPT's paper-faithful matching, aggregation, and scoring over the frozen
    candidate pool and ranks every structured-output exhaustion as an abstention.
    """

    system_id = TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID
    runtime_option_profile = TRIALGPT_PAPER_RUNTIME_OPTION_PROFILE
    retry_exhaustion_policy = TRIALGPT_ABSTAIN_AFTER_RETRY_EXHAUSTION
    option_names = TRIALGPT_PAPER_OPTION_NAMES | frozenset(
        {
            "model",
            "reasoning_effort",
            "codex_executable",
        }
    )
    required_capabilities = TRIALGPT_REQUIRED_CAPABILITIES | frozenset(
        {
            CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
            ELIGIBILITY_CRITERION_VIEW_CAPABILITY,
        }
    )
    optional_capabilities = TRIALGPT_OPTIONAL_CAPABILITIES

    def __init__(
        self,
        provider: TrialGPTProvider | None = None,
        frozen_retrieval: FrozenTrialGPTRetrieval | None = None,
        execution_workspace: Path | None = None,
    ) -> None:
        self._injected_provider = provider is not None
        self.provider = provider or CodexTrialGPTProvider()
        self.frozen_retrieval = frozen_retrieval
        self.execution_workspace = execution_workspace

    def run(self, request: SystemRunRequest) -> SystemRunResult:
        if self.frozen_retrieval is None:
            raise TrialGPTAdapterError(
                "TrialGPT-TAIM-Luna-v1 requires a preflight-validated frozen retrieval artifact"
            )
        result = self._run_paper_stages(request)
        from dataclasses import replace

        return replace(
            result,
            configuration={
                **result.configuration,
                "failure_policy": {
                    **cast(dict[str, object], result.configuration["failure_policy"]),
                    "keyword_generation": "not_executed_frozen_three_luna_consensus",
                },
                "publication_method": trialgpt_publication_method_contract(),
            },
        )

    def _run_paper_stages(self, request: SystemRunRequest) -> SystemRunResult:
        """Run the paper-faithful TrialGPT stages over TAIM Snapshot inputs."""

        from taim.system_contracts import SystemRunResult

        started = time.perf_counter()
        workspace, device, precision, depth, generation_workers = self._runtime_options(request)
        run_workspace = trialgpt_run_workspace(workspace, request.run_id)
        run_workspace.mkdir(parents=True, exist_ok=True)
        if not self._injected_provider:
            self.provider = CodexTrialGPTProvider(
                model=cast(str, _option(request, "model", TRIALGPT_CODEX_MODEL)),
                reasoning_effort=cast(str | None, _option(request, "reasoning_effort")),
                codex_executable=cast(str, _option(request, "codex_executable", "codex")),
            )
        eligibility_criterion_view = find_eligibility_criterion_view(request.snapshot.derived_views)
        views = tuple(
            trial_view(
                trial,
                eligibility_criterion_view=eligibility_criterion_view,
            )
            for trial in request.snapshot.trials
        )
        views_by_trial_id = {view.trial_id: view for view in views}
        if self.frozen_retrieval is None:
            raise TrialGPTAdapterError(
                "the public TrialGPT System requires a preflight-validated frozen "
                "retrieval artifact"
            )
        if depth != 2_000 or request.top_k != 10:
            raise TrialGPTAdapterError(
                "TrialGPT publication requires retrieval_depth 2000 and final top_k 10"
            )
        cache = GenerationCache(
            run_workspace / "trialgpt-generation-cache.sqlite3",
            flush_every=TRIALGPT_GENERATION_CACHE_FLUSH_EVERY,
        )
        traces: list[dict[str, object]] = []
        prompt_audit: list[dict[str, object]] = []
        retrieval_candidates: list[Candidate] = []
        llm_ranking_candidates: list[Candidate] = []
        final_candidates: list[Candidate] = []
        matching_artifacts: list[dict[str, object]] = []
        aggregation_artifacts: list[dict[str, object]] = []
        abstention_artifacts: list[dict[str, object]] = []
        topic_provenance: list[dict[str, object]] = []
        per_topic_generation: list[dict[str, object]] = []
        allow_trial_abstention = (
            self.retry_exhaustion_policy == TRIALGPT_ABSTAIN_AFTER_RETRY_EXHAUSTION
        )

        for topic in request.snapshot.topics:
            topic_traces: list[dict[str, object]] = []
            topic_prompt_audit: list[dict[str, object]] = []
            topic_retrieval_candidates: list[Candidate] = []
            topic_matching_artifacts: list[dict[str, object]] = []
            topic_aggregation_artifacts: list[dict[str, object]] = []
            topic_abstention_artifacts: list[dict[str, object]] = []
            try:
                retrieval_fused = [
                    (row.trial_id, row.score)
                    for row in self.frozen_retrieval.rankings[topic.topic_id]
                ]
                fused = [
                    (row.trial_id, row.score)
                    for row in self.frozen_retrieval.selected_rows(topic.topic_id)
                ]
            except TrialGPTPublicationError as exc:
                raise TrialGPTAdapterError(str(exc)) from exc
            topic_retrieval_candidates.extend(
                Candidate(
                    request.run_id,
                    self.system_id,
                    topic.topic_id,
                    trial_id,
                    rank,
                    score,
                )
                for rank, (trial_id, score) in enumerate(retrieval_fused, start=1)
            )
            llm_fused = fused[: min(depth, TRIALGPT_LLM_CANDIDATE_DEPTH)]
            candidate_tasks = [
                (
                    trial_id,
                    retrieval_score,
                    views_by_trial_id[trial_id],
                )
                for trial_id, retrieval_score in llm_fused
            ]
            topic_final: list[Candidate] = []
            patient = numbered_patient(topic.canonical_text)
            patient_sentence_count = len(patient_sentences(topic.canonical_text))
            topic_id = topic.topic_id
            matching_tasks = [
                (trial_id, view, kind)
                for trial_id, _retrieval_score, view in candidate_tasks
                for kind in ("inclusion", "exclusion")
            ]

            def run_matching(
                task: tuple[str, TrialGPTTrialView, str],
                *,
                selected_patient: str = patient,
                selected_topic_id: str = topic_id,
                selected_patient_sentence_count: int = patient_sentence_count,
            ) -> GenerationResult:
                trial_id, raw_view, raw_kind = task
                kind = cast(Literal["inclusion", "exclusion"], raw_kind)
                view = raw_view
                criteria = view.inclusion if kind == "inclusion" else view.exclusion
                criterion_ids = (
                    view.inclusion_criterion_ids
                    if kind == "inclusion"
                    else view.exclusion_criterion_ids
                )
                payload = {
                    "topic_id": selected_topic_id,
                    "trial_id": trial_id,
                    "kind": kind,
                    "patient": selected_patient,
                    "criteria": list(criteria),
                    "criterion_ids": list(criterion_ids),
                    "eligibility_view_sha256": view.eligibility_view_sha256,
                    "eligibility_source_sha256": view.eligibility_source_sha256,
                    "eligibility_criterion_inventory_sha256": (
                        view.eligibility_criterion_inventory_sha256
                    ),
                }
                criterion_prompt = matching_prompt(view, kind, selected_patient)
                matching_provider: TrialGPTProvider = (
                    self.provider if criteria else _EmptyEligibilityProvider()
                )
                result = _run_generation_stage(
                    provider=matching_provider,
                    cache=cache,
                    stage=f"criterion_matching_{kind}",
                    prompt=criterion_prompt,
                    output_schema=PAPER_MATCHING_SCHEMA,
                    payload=payload,
                    validator=lambda value: _validate_paper_matching_output(
                        value,
                        kind=kind,
                        criterion_count=len(criteria),
                        patient_sentence_count=selected_patient_sentence_count,
                    ),
                    normalizer=lambda value: normalize_matching_output(value, kind=kind),
                    normalizer_version=TRIALGPT_MATCHING_NORMALIZER_VERSION,
                    allow_unresolved=allow_trial_abstention,
                )
                if not result.resolved:
                    failed_transport_attempts = [
                        attempt
                        for attempt in result.attempts
                        if provider_failure_classification(attempt) is not None
                    ]
                    if any(
                        provider_failure_classification(attempt) == "authentication"
                        for attempt in failed_transport_attempts
                    ):
                        raise TrialGPTAdapterError(
                            f"TrialGPT authentication failed at {result.stage}; "
                            "failure-aware execution cannot continue"
                        )
                    if failed_transport_attempts:
                        raise TrialGPTAdapterError(
                            f"TrialGPT transport unresolved at {result.stage}; "
                            "failure-aware abstention accepts only structured-output exhaustion"
                        )
                    if len(result.attempts) != 3:
                        raise TrialGPTAdapterError(
                            f"TrialGPT unresolved cache entry at {result.stage} "
                            "does not contain exactly three attempts"
                        )
                return result

            def make_matching_candidate(
                candidate_task: tuple[str, float, TrialGPTTrialView],
                matching_results_for_candidate: dict[str, GenerationResult],
            ) -> dict[str, object]:
                trial_id, retrieval_score, view = candidate_task
                matching: dict[str, object] = {}
                unresolved_stages: list[str] = []
                for kind in ("inclusion", "exclusion"):
                    result = matching_results_for_candidate[kind]
                    if not result.resolved:
                        matching[kind] = None
                        unresolved_stages.append(result.stage)
                        continue
                    matching_output = result.output
                    if isinstance(matching_output, Mapping) and isinstance(
                        matching_output.get("criteria"), Mapping
                    ):
                        matching_output = matching_output["criteria"]
                    matching[kind] = matching_output
                return {
                    "trial_id": trial_id,
                    "retrieval_score": retrieval_score,
                    "view": view,
                    "matching": matching,
                    "matching_results": matching_results_for_candidate,
                    "unresolved_stages": tuple(unresolved_stages),
                }

            def run_aggregation(
                task: tuple[str, float, TrialGPTTrialView, dict[str, object]],
                *,
                selected_patient: str = patient,
                selected_topic_id: str = topic_id,
            ) -> GenerationResult:
                trial_id, retrieval_score, raw_view, matching = task
                view = raw_view
                aggregation_prompt_text = aggregation_prompt(view, selected_patient, matching)
                return _run_generation_stage(
                    provider=self.provider,
                    cache=cache,
                    stage="final_aggregation",
                    prompt=aggregation_prompt_text,
                    output_schema=PAPER_AGGREGATION_SCHEMA,
                    payload={
                        "topic_id": selected_topic_id,
                        "trial_id": trial_id,
                        "matching": matching,
                        "retrieval_score": retrieval_score,
                    },
                    validator=validate_aggregation_output,
                )

            aggregation_results_by_trial: dict[str, GenerationResult] = {}
            if generation_workers == 1:
                matching_by_candidate = []
                for candidate_task in candidate_tasks:
                    trial_id, retrieval_score, view = candidate_task
                    matching_results_for_candidate = {
                        kind: run_matching((trial_id, view, kind))
                        for kind in ("inclusion", "exclusion")
                    }
                    candidate = make_matching_candidate(
                        candidate_task, matching_results_for_candidate
                    )
                    matching_by_candidate.append(candidate)
                    if not candidate["unresolved_stages"]:
                        aggregation_results_by_trial[trial_id] = run_aggregation(
                            (
                                trial_id,
                                retrieval_score,
                                view,
                                cast(dict[str, object], candidate["matching"]),
                            )
                        )
            else:
                matching_results = _ordered_generation_map(
                    matching_tasks,
                    run_matching,
                    max_workers=generation_workers,
                )
                matching_by_trial: dict[tuple[str, str], GenerationResult] = {
                    (trial_id, kind): result
                    for (trial_id, _view, kind), result in zip(
                        matching_tasks, matching_results, strict=True
                    )
                }
                matching_by_candidate = [
                    make_matching_candidate(
                        candidate_task,
                        {
                            kind: matching_by_trial[(candidate_task[0], kind)]
                            for kind in ("inclusion", "exclusion")
                        },
                    )
                    for candidate_task in candidate_tasks
                ]
                aggregation_tasks: list[tuple[str, float, TrialGPTTrialView, dict[str, object]]] = [
                    (
                        cast(str, candidate["trial_id"]),
                        cast(float, candidate["retrieval_score"]),
                        cast(TrialGPTTrialView, candidate["view"]),
                        cast(dict[str, object], candidate["matching"]),
                    )
                    for candidate in matching_by_candidate
                    if not candidate["unresolved_stages"]
                ]
                aggregation_results = _ordered_generation_map(
                    aggregation_tasks,
                    run_aggregation,
                    max_workers=generation_workers,
                )
                aggregation_results_by_trial = {
                    task[0]: result
                    for task, result in zip(aggregation_tasks, aggregation_results, strict=True)
                }
            for candidate in matching_by_candidate:
                trial_id = cast(str, candidate["trial_id"])
                retrieval_score = cast(float, candidate["retrieval_score"])
                view = cast(TrialGPTTrialView, candidate["view"])
                matching = cast(dict[str, object], candidate["matching"])
                matching_results_for_candidate = cast(
                    dict[str, GenerationResult], candidate["matching_results"]
                )
                for kind in ("inclusion", "exclusion"):
                    result = matching_results_for_candidate[kind]
                    criterion_prompt = matching_prompt(view, kind, patient)
                    payload = {
                        "topic_id": topic.topic_id,
                        "trial_id": trial_id,
                        "kind": kind,
                        "patient": patient,
                        "criteria": list(view.inclusion if kind == "inclusion" else view.exclusion),
                        "criterion_ids": list(
                            view.inclusion_criterion_ids
                            if kind == "inclusion"
                            else view.exclusion_criterion_ids
                        ),
                        "eligibility_view_sha256": view.eligibility_view_sha256,
                        "eligibility_source_sha256": view.eligibility_source_sha256,
                        "eligibility_criterion_inventory_sha256": (
                            view.eligibility_criterion_inventory_sha256
                        ),
                    }
                    trace = _trace_record(result, topic_id=topic.topic_id, trial_id=trial_id)
                    traces.append(trace)
                    topic_traces.append(trace)
                    audit: dict[str, object] = {
                        "stage": f"criterion_matching_{kind}",
                        "topic_id": topic.topic_id,
                        "trial_id": trial_id,
                        "prompt": criterion_prompt,
                        "payload": payload,
                        "output_schema": PAPER_MATCHING_SCHEMA,
                        "result": trace,
                    }
                    prompt_audit.append(audit)
                    topic_prompt_audit.append(audit)
                matching_artifact: dict[str, object] = {
                    "topic_id": topic.topic_id,
                    "trial_id": trial_id,
                    "matching": matching,
                    "inclusion_criterion_ids": list(view.inclusion_criterion_ids),
                    "exclusion_criterion_ids": list(view.exclusion_criterion_ids),
                    "eligibility_view_sha256": view.eligibility_view_sha256,
                    "eligibility_source_sha256": view.eligibility_source_sha256,
                    "eligibility_criterion_inventory_sha256": (
                        view.eligibility_criterion_inventory_sha256
                    ),
                }
                if allow_trial_abstention:
                    matching_artifact["status"] = (
                        "unresolved" if candidate["unresolved_stages"] else "resolved"
                    )
                matching_artifacts.append(matching_artifact)
                topic_matching_artifacts.append(matching_artifact)
                unresolved_matching_results = [
                    result
                    for result in matching_results_for_candidate.values()
                    if not result.resolved
                ]
                if unresolved_matching_results:
                    abstention = {
                        "topic_id": topic.topic_id,
                        "trial_id": trial_id,
                        "retrieval_rank": next(
                            item.rank
                            for item in topic_retrieval_candidates
                            if item.trial_id == trial_id
                        ),
                        "reason": "unresolved_criterion_matching",
                        "unresolved_stages": sorted(
                            result.stage for result in unresolved_matching_results
                        ),
                        "logical_call_ids": sorted(
                            result.logical_call_id for result in unresolved_matching_results
                        ),
                    }
                    abstention_artifacts.append(abstention)
                    topic_abstention_artifacts.append(abstention)
                    continue
                aggregation_result = aggregation_results_by_trial[trial_id]
                aggregation_prompt_text = aggregation_prompt(view, patient, matching)
                aggregation_trace = _trace_record(
                    aggregation_result, topic_id=topic.topic_id, trial_id=trial_id
                )
                traces.append(aggregation_trace)
                topic_traces.append(aggregation_trace)
                aggregation_audit: dict[str, object] = {
                    "stage": "final_aggregation",
                    "topic_id": topic.topic_id,
                    "trial_id": trial_id,
                    "prompt": aggregation_prompt_text,
                    "payload": {
                        "topic_id": topic.topic_id,
                        "trial_id": trial_id,
                        "matching": matching,
                        "retrieval_score": retrieval_score,
                    },
                    "output_schema": PAPER_AGGREGATION_SCHEMA,
                    "result": aggregation_trace,
                }
                prompt_audit.append(aggregation_audit)
                topic_prompt_audit.append(aggregation_audit)
                aggregation = cast(Mapping[str, object], aggregation_result.output)
                score = final_score(matching, aggregation)
                aggregation_artifact = {
                    "topic_id": topic.topic_id,
                    "trial_id": trial_id,
                    "retrieval_score": retrieval_score,
                    "matching_score": matching_score(matching),
                    "aggregation": aggregation,
                    "final_score": score,
                }
                if allow_trial_abstention:
                    aggregation_artifact["status"] = "resolved"
                aggregation_artifacts.append(aggregation_artifact)
                topic_aggregation_artifacts.append(aggregation_artifact)
                topic_final.append(
                    Candidate(request.run_id, self.system_id, topic.topic_id, trial_id, 1, score)
                )
            if topic_abstention_artifacts:
                for abstention in sorted(
                    topic_abstention_artifacts,
                    key=lambda item: (
                        cast(int, item["retrieval_rank"]),
                        cast(str, item["trial_id"]),
                    ),
                ):
                    topic_final.append(
                        Candidate(
                            request.run_id,
                            self.system_id,
                            topic.topic_id,
                            cast(str, abstention["trial_id"]),
                            1,
                            -3.0,
                        )
                    )
            abstention_order = {
                cast(str, item["trial_id"]): cast(int, item["retrieval_rank"])
                for item in topic_abstention_artifacts
            }
            topic_final.sort(
                key=lambda candidate: (
                    (1, abstention_order[candidate.trial_id], candidate.trial_id)
                    if candidate.trial_id in abstention_order
                    else (0, -candidate.score, candidate.trial_id)
                )
            )
            topic_llm_ranked = [
                Candidate(
                    candidate.run_id,
                    candidate.system_id,
                    candidate.topic_id,
                    candidate.trial_id,
                    rank,
                    candidate.score,
                )
                for rank, candidate in enumerate(topic_final, start=1)
            ]
            topic_final_ranked = topic_llm_ranked[: request.top_k]
            primary_ranks = {candidate.trial_id: candidate.rank for candidate in topic_final_ranked}
            for abstention in topic_abstention_artifacts:
                abstention["primary_ranking_rank"] = primary_ranks.get(
                    cast(str, abstention["trial_id"])
                )
            retrieval_candidates.extend(topic_retrieval_candidates)
            llm_ranking_candidates.extend(topic_llm_ranked)
            final_candidates.extend(topic_final_ranked)
            expected_topic_calls = (
                3 * len(candidate_tasks)
                if self.frozen_retrieval is not None
                else 1 + 3 * len(candidate_tasks)
            )
            unresolved_topic_calls = sum(trace["status"] == "unresolved" for trace in topic_traces)
            transport_failure_attempts = sum(
                provider_failure_classification(attempt) == "transport"
                for trace in topic_traces
                for attempt in cast(list[dict[str, object]], trace["attempts"])
            )
            authentication_failure_attempts = sum(
                provider_failure_classification(attempt) == "authentication"
                for trace in topic_traces
                for attempt in cast(list[dict[str, object]], trace["attempts"])
            )
            recovered_transport_logical_calls = sum(
                trace["status"] == "resolved"
                and any(
                    provider_failure_classification(attempt) == "transport"
                    for attempt in cast(list[dict[str, object]], trace["attempts"])
                )
                for trace in topic_traces
            )
            per_topic_generation.append(
                {
                    "topic_id": topic.topic_id,
                    "retrieval_candidate_count": len(topic_retrieval_candidates),
                    "llm_candidate_count": len(candidate_tasks),
                    "expected_logical_call_count": expected_topic_calls,
                    "recorded_logical_call_count": len(topic_traces),
                    "resolved_logical_call_count": sum(
                        trace["status"] == "resolved" for trace in topic_traces
                    ),
                    "semantic_unresolved_logical_call_count": unresolved_topic_calls,
                    "semantic_abstention_count": len(topic_abstention_artifacts),
                    "transport_failure_attempt_count": transport_failure_attempts,
                    "recovered_transport_logical_call_count": (recovered_transport_logical_calls),
                    "authentication_failure_attempt_count": authentication_failure_attempts,
                    "generation_coverage": (
                        (len(topic_traces) - unresolved_topic_calls) / expected_topic_calls
                    ),
                }
            )
            topic_provenance.append(
                _write_topic_provenance(
                    workspace=run_workspace,
                    run_id=request.run_id,
                    system_id=self.system_id,
                    prepared_snapshot_id=request.snapshot.prepared_snapshot_id,
                    task_input_id=request.snapshot.task_input_id,
                    system_input_id=request.system_input_id,
                    topic_id=topic.topic_id,
                    traces=topic_traces,
                    prompt_audit=topic_prompt_audit,
                    retrieval_candidates=topic_retrieval_candidates,
                    matching_artifacts=topic_matching_artifacts,
                    aggregation_artifacts=topic_aggregation_artifacts,
                    final_candidates=topic_final_ranked,
                    abstention_artifacts=(
                        topic_abstention_artifacts if allow_trial_abstention else None
                    ),
                )
            )

        cache.close()
        trace_path = run_workspace / "trialgpt-generation-traces.json"
        prompt_audit_path = run_workspace / "trialgpt-prompt-audit.json"
        retrieval_path = run_workspace / "trialgpt-hybrid-retrieval.json"
        final_path = run_workspace / "trialgpt-final-ranking.json"
        matching_path = run_workspace / "trialgpt-criterion-matching.json"
        aggregation_path = run_workspace / "trialgpt-aggregation.json"
        trace_path.write_text(
            json.dumps(traces, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        prompt_audit_path.write_text(
            json.dumps(prompt_audit, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        retrieval_path.write_text(
            json.dumps([candidate.to_dict() for candidate in retrieval_candidates], sort_keys=True),
            encoding="utf-8",
        )
        final_path.write_text(
            json.dumps([candidate.to_dict() for candidate in final_candidates], sort_keys=True),
            encoding="utf-8",
        )
        matching_path.write_text(
            json.dumps(matching_artifacts, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        aggregation_path.write_text(
            json.dumps(aggregation_artifacts, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        abstention_artifact: dict[str, object] | None = None
        generation_coverage: dict[str, object] | None = None
        if allow_trial_abstention:
            abstention_path = run_workspace / "trialgpt-abstentions.json"
            abstention_artifact = _write_json_artifact(abstention_path, abstention_artifacts)
            retrieved_trial_count = len(retrieval_candidates)
            llm_candidate_count = len(llm_ranking_candidates)
            scored_trial_count = len(aggregation_artifacts)
            abstained_trial_count = len(abstention_artifacts)
            expected_logical_call_count = (
                3 * llm_candidate_count
                if self.frozen_retrieval is not None
                else len(request.snapshot.topics) + 3 * llm_candidate_count
            )
            generation_coverage = {
                "expected_logical_call_count": expected_logical_call_count,
                "generation_call_count": len(traces),
                "resolved_logical_call_count": sum(
                    trace["status"] == "resolved" for trace in traces
                ),
                "unresolved_logical_call_count": sum(
                    trace["status"] == "unresolved" for trace in traces
                ),
                "retrieved_trial_count": retrieved_trial_count,
                "llm_candidate_count": llm_candidate_count,
                "scored_trial_count": scored_trial_count,
                "abstained_trial_count": abstained_trial_count,
                "skipped_aggregation_count": abstained_trial_count,
                "transport_failure_attempt_count": sum(
                    cast(int, topic["transport_failure_attempt_count"])
                    for topic in per_topic_generation
                ),
                "recovered_transport_logical_call_count": sum(
                    cast(int, topic["recovered_transport_logical_call_count"])
                    for topic in per_topic_generation
                ),
                "authentication_failure_attempt_count": sum(
                    cast(int, topic["authentication_failure_attempt_count"])
                    for topic in per_topic_generation
                ),
                "trial_coverage": (
                    scored_trial_count / llm_candidate_count if llm_candidate_count else 1.0
                ),
                "semantic_abstention_identities": [
                    {
                        "topic_id": abstention["topic_id"],
                        "trial_id": abstention["trial_id"],
                    }
                    for abstention in abstention_artifacts
                ],
            }
            if scored_trial_count + abstained_trial_count != llm_candidate_count:
                raise TrialGPTAdapterError("failure-aware trial coverage does not reconcile")
            if len(traces) + abstained_trial_count != expected_logical_call_count:
                raise TrialGPTAdapterError("failure-aware generation coverage does not reconcile")
        provider_configuration = _provider_provenance(self.provider)
        runtime = time.perf_counter() - started
        return SystemRunResult(
            candidates=tuple(final_candidates),
            configuration={
                "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
                "provider": self.provider.provider_id,
                "provider_configuration": provider_configuration,
                "model_identity": _trialgpt_model_identity(provider_configuration),
                "index_identity": _trialgpt_index_identity(
                    request, frozen_retrieval=self.frozen_retrieval
                ),
                "codex_model": getattr(self.provider, "model", None),
                "prompt_version": TRIALGPT_PAPER_PROMPT_VERSION,
                "matching_output_contract_version": TRIALGPT_MATCHING_OUTPUT_CONTRACT_VERSION,
                "matching_normalizer_version": TRIALGPT_MATCHING_NORMALIZER_VERSION,
                "tokenizer_id": TRIALGPT_TOKENIZER_ID,
                "sentence_id_bounds": "0 <= sentence_id < numbered_patient_sentence_count",
                "provenance_version": TRIALGPT_PROVENANCE_VERSION,
                "workspace": str(workspace),
                "run_workspace": str(run_workspace),
                "environment": _safe_environment_provenance(),
                "topic_provenance": topic_provenance,
                "generation_cache": {
                    "path": str(run_workspace / "trialgpt-generation-cache.sqlite3"),
                    "flush_every": TRIALGPT_GENERATION_CACHE_FLUSH_EVERY,
                    "cache_hit_count": sum(bool(trace["cache_hit"]) for trace in traces),
                    "cache_miss_count": sum(not bool(trace["cache_hit"]) for trace in traces),
                },
                "paper_faithful": not allow_trial_abstention,
                "reference_provider": "azure_openai",
                "provider_is_paper_backend": self.provider.provider_id == "azure-openai",
                "provider_effect_is_scientific": True,
                "device": device,
                "precision": precision,
                "retrieval_depth": depth,
                "llm_candidate_depth": min(depth, TRIALGPT_LLM_CANDIDATE_DEPTH),
                "generation_workers": generation_workers,
                "requested_generation_workers": request.options.get("generation_workers"),
                "effective_generation_workers": generation_workers,
                "per_topic_generation": per_topic_generation,
                "generation_cache_flush_every": TRIALGPT_GENERATION_CACHE_FLUSH_EVERY,
                "frozen_retrieval": self.frozen_retrieval.provenance(
                    [topic.topic_id for topic in request.snapshot.topics]
                ),
                "keyword_generation": "not_executed_frozen_three_luna_consensus",
                "expected_logical_calls_per_topic": TRIALGPT_PUBLICATION_LOGICAL_CALLS_PER_TOPIC,
                "qrels_used": False,
                "prepared_snapshot_id": request.snapshot.prepared_snapshot_id,
                "task_input_id": request.snapshot.task_input_id,
                "system_input_id": request.system_input_id,
                "generation_trace": {"path": str(trace_path), "sha256": sha256_file(trace_path)},
                "prompt_audit": {
                    "path": str(prompt_audit_path),
                    "sha256": sha256_file(prompt_audit_path),
                    "record_count": len(prompt_audit),
                },
                "retrieval_stage": {
                    "path": str(retrieval_path),
                    "sha256": sha256_file(retrieval_path),
                    "candidate_count": len(retrieval_candidates),
                },
                "criterion_matching": {
                    "path": str(matching_path),
                    "sha256": sha256_file(matching_path),
                },
                "aggregation": {
                    "path": str(aggregation_path),
                    "sha256": sha256_file(aggregation_path),
                },
                "primary_ranking_artifact": {
                    "path": str(final_path),
                    "sha256": sha256_file(final_path),
                    "candidate_count": len(final_candidates),
                },
                **(
                    {
                        # The System these factors are declared against is not part of the
                        # release, so the record keeps the factors and names no parent.
                        "controlled_variant": {
                            "declared_factors": [
                                "retry_exhaustion_policy",
                                "unambiguous_matching_tuple_normalization",
                                "patient_sentence_id_bounds_validation",
                            ],
                        },
                        "failure_policy": {
                            "policy_version": TRIALGPT_FAILURE_AWARE_POLICY_VERSION,
                            "keyword_generation": "fail_before_retrieval",
                            "criterion_matching": (
                                "abstain_after_three_structured_output_attempts"
                            ),
                            "aggregation": "fail_before_ranking",
                            "transport": "retry_then_fail_unresolved_before_ranking",
                            "authentication": "fail_immediately_before_ranking",
                            "ranking": "resolved_then_abstentions_by_retrieval_rank_and_trial_id",
                            "abstention_score": -3.0,
                        },
                        "generation_coverage": generation_coverage,
                        "abstentions": {
                            **cast(dict[str, object], abstention_artifact),
                            "record_count": len(abstention_artifacts),
                        },
                    }
                    if allow_trial_abstention
                    else {}
                ),
                "runtime_seconds": runtime,
            },
            runtime_seconds=runtime,
            primary_ranking="candidates",
            pipeline_depth="post_eligibility",
            stage_rankings=(
                StageRanking(
                    "hybrid-retrieval",
                    "retrieval",
                    tuple(retrieval_candidates),
                    sha256_file(retrieval_path),
                ),
                StageRanking(
                    "trialgpt-llm-ranking",
                    "post_eligibility",
                    tuple(llm_ranking_candidates),
                ),
            ),
        )

    def _runtime_options(self, request: SystemRunRequest) -> tuple[Path, str, str, int, int]:
        if self.execution_workspace is None:
            raise TrialGPTAdapterError(
                "TrialGPT-TAIM-Luna-v1 requires a controller-owned execution workspace"
            )
        generation_workers = _option(
            request, "generation_workers", TRIALGPT_FULL_GENERATION_WORKERS
        )
        if (
            isinstance(generation_workers, bool)
            or not isinstance(generation_workers, int)
            or not 1 <= generation_workers <= TRIALGPT_MAX_GENERATION_WORKERS
        ):
            raise TrialGPTAdapterError(
                f"generation_workers must be an integer from 1 to {TRIALGPT_MAX_GENERATION_WORKERS}"
            )
        return self.execution_workspace.resolve(), "cpu", "float32", 2_000, generation_workers


__all__ = [
    "TRIALGPT_CODEX_MODEL",
    "CodexTrialGPTProvider",
    "TrialGPTTAIMLunaV1System",
    "trialgpt_publication_method_contract",
    "trialgpt_run_workspace",
]
