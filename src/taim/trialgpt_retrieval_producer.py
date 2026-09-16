"""Produce judgment-blind three-Luna retrieval inputs for TrialGPT publication runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast

from taim.adapters.trialgpt import TrialGPTProvider
from taim.artifacts import sha256_file
from taim.contracts import content_sha256, require_sha256
from taim.schemas import JsonValue, SchemaValidationError
from taim.trialgpt_generation import (
    GenerationCache,
    GenerationRequest,
    GenerationResult,
    ProviderResponse,
    run_generation,
    validate_keyword_output,
)
from taim.trialgpt_paper import (
    TRIALGPT_TOKENIZER_ID,
    TrialGPTTrialView,
)
from taim.trialgpt_publication import (
    TRIALGPT_PUBLICATION_CANDIDATE_DEPTH,
    TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
    FrozenTrialGPTRetrieval,
)

TRIALGPT_RETRIEVAL_PRODUCER_CONTRACT_ID = "trialgpt-three-luna-consensus-v2"
TRIALGPT_RETRIEVAL_SOURCE_ARMS = ("medium", "xhigh", "recall-explicit")
TRIALGPT_RETRIEVAL_SOURCE_DEPTH = 2_000
TRIALGPT_RETRIEVAL_OUTPUT_DEPTH = 2_000
TRIALGPT_RETRIEVAL_BM25_ID = "taim-controlled-bm25-v2"
FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256 = (
    "sha256:a617a46c2d61e76ad25963e709cad2c3ff99d4be466f173b8add02f1754927ef"
)


@dataclass(frozen=True, slots=True)
class TrialGPTRetrievalTarget:
    """One paper-admitted Track/Profile pair for the external TrialGPT System."""

    track: str
    profile: str
    prepared_dataset_id: str
    snapshot_name: str
    topic_count: int
    supplied_pool_membership: bool


TRIALGPT_RETRIEVAL_TARGETS = (
    TrialGPTRetrievalTarget(
        track="trec-ct-2021",
        profile="trec-ct-2021-external-fidelity-26149",
        prepared_dataset_id="trec-ct-2021",
        snapshot_name="trec-ct-2021",
        topic_count=75,
        supplied_pool_membership=True,
    ),
    TrialGPTRetrievalTarget(
        track="trec-ct-2022",
        profile="trec-ct-2022-judgment-union",
        prepared_dataset_id="trec-ct-2022",
        snapshot_name="trec-ct-2022",
        topic_count=50,
        supplied_pool_membership=False,
    ),
    TrialGPTRetrievalTarget(
        track="sigir-ct-2016",
        profile="sigir-ct-2016-description-judgment-union",
        prepared_dataset_id="sigir-ct-2016-description",
        snapshot_name="sigir-ct-2016-description",
        topic_count=60,
        supplied_pool_membership=False,
    ),
    TrialGPTRetrievalTarget(
        track="sigir-ct-2016",
        profile="sigir-ct-2016-summary-judgment-union",
        prepared_dataset_id="sigir-ct-2016-summary",
        snapshot_name="sigir-ct-2016-summary",
        topic_count=60,
        supplied_pool_membership=False,
    ),
)

_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_RANKING_CONFIGURATION = "judgment-blind-consensus-all-luna"
_RANKING_FILENAME = f"{_RANKING_CONFIGURATION}.jsonl"
_LOCK_FILENAME = f"{_RANKING_CONFIGURATION}-lock.json"
_SOURCE_RANKINGS_FILENAME = "three-luna-source-rankings.jsonl"
_CONSENSUS_FEATURES_FILENAME = "three-luna-consensus-features.jsonl"
_QUERY_PLANS_FILENAME = "three-luna-query-plans.jsonl"
_RAW_CHANNELS_FILENAME = "three-luna-raw-channels.jsonl"
_QUERY_STAGE = "keyword_generation"
_KEYWORD_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "conditions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "conditions"],
    "additionalProperties": False,
}
_DEFAULT_QUERY_PROMPT_CONTRACT_ID = "trialgpt-paper-keyword-v1"
_RECALL_QUERY_PROMPT_CONTRACT_ID = "trialgpt-recall-explicit-v2"
_DEFAULT_QUERY_SYSTEM = (
    "You are a helpful assistant and your task is to help search relevant clinical trials "
    "for a given patient description. Please first summarize the main medical problems of "
    "the patient. Then generate up to 32 key conditions for searching relevant clinical "
    "trials for this patient. The key condition list should be ranked by priority. Please "
    'output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": '
    "List[Str(condition)]}}."
)
_RECALL_QUERY_SYSTEM = (
    "You are a helpful assistant searching for clinical trials for a patient. First summarize "
    "the patient's main medical problems. Then return up to 32 ranked, medically supported "
    "search concepts spanning diagnoses, biomarkers, symptoms, prior treatments, interventions, "
    "and clinically relevant synonyms when those facts are stated or directly supported by the "
    "patient description. Do not invent diagnoses, biomarkers, symptoms, treatments, "
    "interventions, or synonyms. Rank concepts by likely retrieval value. Output only a JSON "
    'dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'
)


class TrialGPTRetrievalProducerError(ValueError):
    """Raised when source retrieval evidence cannot produce the frozen consensus."""


def resolve_trialgpt_retrieval_target(
    *,
    track: str,
    profile: str,
) -> TrialGPTRetrievalTarget:
    """Resolve the closed paper matrix and reject unsupported Track/Profile combinations."""

    matches = tuple(
        target
        for target in TRIALGPT_RETRIEVAL_TARGETS
        if target.track == track and target.profile == profile
    )
    if len(matches) != 1:
        raise TrialGPTRetrievalProducerError(
            f"TrialGPT retrieval is not admitted for Track={track!r}, Profile={profile!r}"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class SourceArmCandidate:
    """One candidate in a source arm plus its raw-channel exposure facts."""

    trial_id: str
    rank: int
    lexical_exposed: bool
    dense_exposed: bool


@dataclass(frozen=True, slots=True)
class SourceArmRanking:
    """One complete depth-2,000 source ranking for one topic and Luna arm."""

    topic_id: str
    arm_id: str
    candidates: tuple[SourceArmCandidate, ...]


@dataclass(frozen=True, slots=True)
class GeneratedArmQuery:
    """One provider-generated, identity-bound query plan for one topic and arm."""

    topic_id: str
    arm_id: str
    summary: str
    conditions: tuple[str, ...]
    patient_text_sha256: str
    prompt_contract_id: str
    provider_configuration: Mapping[str, JsonValue]
    generation_trace: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class RetrievalQueryTopic:
    """The exact public-safe topic surface used by retrieval query generation."""

    topic_id: str
    patient_text: str


class LexicalRetrievalBackend(Protocol):
    """Minimal exact ranking seam required from the lexical producer arm."""

    def rank(self, condition: str, depth: int) -> list[str]: ...


class DenseRetrievalBackend(Protocol):
    """Minimal exact ranking seam required from the dense producer arm."""

    model_configuration: Mapping[str, object]

    def rank(
        self,
        *,
        conditions: Sequence[str],
        trials: Sequence[TrialGPTTrialView],
        depth: int,
    ) -> list[list[str]]: ...


@dataclass(frozen=True, slots=True)
class RawArmRetrieval:
    """The complete per-condition lexical and dense evidence for one source arm."""

    query: GeneratedArmQuery
    bm25_rankings: tuple[tuple[str, ...], ...]
    medcpt_rankings: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ConsensusFeatures:
    """The four frozen, judgment-blind features used by the consensus score."""

    arm_count: int
    best_rank_quality: float
    per_arm_reciprocal_rank: tuple[float, float, float]
    lexical_dense_agreement_count: int


@dataclass(frozen=True, slots=True)
class ConsensusCandidate:
    """One final consensus row with independently auditable feature values."""

    trial_id: str
    rank: int
    score: float
    features: ConsensusFeatures


@dataclass(frozen=True, slots=True)
class FrozenRetrievalBinding:
    """External identities that bind a fresh retrieval production event."""

    run_id: str
    benchmark_lineage: str
    prepared_snapshot_id: str
    evaluation_package_id: str
    task_input_id: str
    pool_receipt_id: str
    pool_count: int
    pool_ids_sha256: str
    protocol_approval_id: str
    producer_git_commit: str
    producer_dependency_lock_sha256: str
    release_manifest_id: str
    public_tree_id: str
    package_artifact_id: str


@dataclass(frozen=True, slots=True)
class WrittenFrozenRetrieval:
    """Paths written by one complete frozen-retrieval production."""

    retrieval_path: Path
    lock_path: Path
    source_rankings_path: Path
    consensus_features_path: Path
    query_plans_path: Path
    raw_channels_path: Path


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    return (0, int(topic_id)) if topic_id.isdigit() else (1, topic_id)


def _query_prompt(arm_id: str, patient_text: str) -> tuple[str, str]:
    if arm_id in {"medium", "xhigh"}:
        contract_id = _DEFAULT_QUERY_PROMPT_CONTRACT_ID
        system = _DEFAULT_QUERY_SYSTEM
    elif arm_id == "recall-explicit":
        contract_id = _RECALL_QUERY_PROMPT_CONTRACT_ID
        system = _RECALL_QUERY_SYSTEM
    else:
        raise TrialGPTRetrievalProducerError(f"unsupported query arm {arm_id!r}")
    prompt = f"{system}\n\nHere is the patient description: \n{patient_text}\n\nJSON output:"
    return contract_id, prompt


class _QueryProviderBridge:
    def __init__(
        self,
        provider: TrialGPTProvider,
        *,
        payload: Mapping[str, object],
    ) -> None:
        self.provider = provider
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
            stage=_QUERY_STAGE,
            prompt=prompt,
            output_schema=output_schema,
            payload=self.payload,
        )
        return ProviderResponse(
            response.raw_response,
            usage=response.usage,
            metadata=response.metadata,
        )


def _provider_configuration(provider: TrialGPTProvider, arm_id: str) -> dict[str, JsonValue]:
    provenance = getattr(provider, "provenance", None)
    raw = (
        provenance()
        if callable(provenance)
        else {
            "provider_id": provider.provider_id,
            "model": getattr(provider, "model", None),
            "reasoning_effort": getattr(provider, "reasoning_effort", None),
        }
    )
    try:
        normalized = json.loads(
            json.dumps(raw, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        )
    except (TypeError, ValueError) as exc:
        raise TrialGPTRetrievalProducerError(
            f"query provider provenance for arm {arm_id!r} is not JSON-compatible"
        ) from exc
    if not isinstance(normalized, dict):
        raise TrialGPTRetrievalProducerError(
            f"query provider provenance for arm {arm_id!r} must be an object"
        )
    expected_effort = "xhigh" if arm_id == "xhigh" else "medium"
    if (
        normalized.get("model") != "gpt-5.6-luna"
        or normalized.get("reasoning_effort") != expected_effort
    ):
        raise TrialGPTRetrievalProducerError(
            f"query arm {arm_id!r} requires gpt-5.6-luna at {expected_effort} effort"
        )
    return cast(dict[str, JsonValue], normalized)


def _stable_generation_trace(result: GenerationResult) -> dict[str, JsonValue]:
    return {
        "attempts": [
            {
                "attempt": attempt.attempt,
                "kind": attempt.kind,
                "parsed_output_sha256": attempt.parsed_output_sha256,
                "prompt_sha256": attempt.prompt_sha256,
                "raw_output_sha256": attempt.raw_output_sha256,
                "retry_reason": attempt.retry_reason,
                "transport_error": attempt.transport_error,
                "validation_errors": list(attempt.validation_errors),
            }
            for attempt in result.attempts
        ],
        "identity_sha256": result.identity_sha256,
        "logical_call_id": result.logical_call_id,
        "output_sha256": content_sha256(result.output),
        "provider_id": result.provider_id,
        "selected_attempt": result.selected_attempt,
        "status": result.status,
    }


def generate_three_luna_queries(
    topics: Sequence[RetrievalQueryTopic],
    *,
    providers: Mapping[str, TrialGPTProvider],
    cache: GenerationCache,
    max_workers: int = 8,
) -> tuple[GeneratedArmQuery, ...]:
    """Generate all three frozen query arms with validated, restartable provider calls."""

    if isinstance(max_workers, bool) or not 1 <= max_workers <= 48:
        raise TrialGPTRetrievalProducerError("query generation max_workers must be from 1 to 48")
    if not topics or len({topic.topic_id for topic in topics}) != len(topics):
        raise TrialGPTRetrievalProducerError("query generation topics must be non-empty and unique")
    if any(not topic.topic_id or not topic.patient_text.strip() for topic in topics):
        raise TrialGPTRetrievalProducerError("query generation topics require IDs and patient text")
    if set(providers) != set(TRIALGPT_RETRIEVAL_SOURCE_ARMS):
        raise TrialGPTRetrievalProducerError(
            "query generation requires exactly medium, xhigh, and recall-explicit providers"
        )
    provider_configs = {
        arm_id: _provider_configuration(providers[arm_id], arm_id)
        for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
    }
    tasks = tuple(
        (topic, arm_id)
        for topic in sorted(topics, key=lambda item: _topic_sort_key(item.topic_id))
        for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
    )

    def generate(task: tuple[RetrievalQueryTopic, str]) -> GeneratedArmQuery:
        topic, arm_id = task
        prompt_contract_id, prompt = _query_prompt(arm_id, topic.patient_text)
        payload = {
            "arm_id": arm_id,
            "patient_text": topic.patient_text,
            "topic_id": topic.topic_id,
        }
        result = run_generation(
            GenerationRequest(
                stage=_QUERY_STAGE,
                prompt=prompt,
                output_schema=_KEYWORD_OUTPUT_SCHEMA,
                input_payload=payload,
                provider_config=provider_configs[arm_id],
                validator=validate_keyword_output,
            ),
            _QueryProviderBridge(providers[arm_id], payload=payload),
            cache=cache,
        )
        if not result.resolved or not isinstance(result.output, Mapping):
            raise TrialGPTRetrievalProducerError(
                f"query generation unresolved for topic {topic.topic_id!r} arm {arm_id!r}"
            )
        summary = result.output.get("summary")
        conditions = result.output.get("conditions")
        if not isinstance(summary, str) or not isinstance(conditions, list):
            raise AssertionError("validated keyword output changed shape")
        return GeneratedArmQuery(
            topic_id=topic.topic_id,
            arm_id=arm_id,
            summary=summary,
            conditions=tuple(cast(list[str], conditions)),
            patient_text_sha256="sha256:"
            + hashlib.sha256(topic.patient_text.encode("utf-8")).hexdigest(),
            prompt_contract_id=prompt_contract_id,
            provider_configuration=provider_configs[arm_id],
            generation_trace=_stable_generation_trace(result),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return tuple(executor.map(generate, tasks))


def retrieve_three_luna_queries(
    queries: Sequence[GeneratedArmQuery],
    *,
    trials: Sequence[TrialGPTTrialView],
    lexical_backend: LexicalRetrievalBackend,
    dense_backend: DenseRetrievalBackend,
) -> tuple[RawArmRetrieval, ...]:
    """Run both frozen retrieval channels for every generated arm query."""

    membership = [(query.topic_id, query.arm_id) for query in queries]
    if len(membership) != len(set(membership)):
        raise TrialGPTRetrievalProducerError("generated retrieval queries contain duplicates")
    by_topic: defaultdict[str, set[str]] = defaultdict(set)
    for query in queries:
        by_topic[query.topic_id].add(query.arm_id)
    expected_arms = set(TRIALGPT_RETRIEVAL_SOURCE_ARMS)
    if not by_topic or any(arms != expected_arms for arms in by_topic.values()):
        raise TrialGPTRetrievalProducerError(
            "generated retrieval queries require all three source arms for every topic"
        )
    ordered_queries = sorted(
        queries,
        key=lambda query: (
            _topic_sort_key(query.topic_id),
            TRIALGPT_RETRIEVAL_SOURCE_ARMS.index(query.arm_id),
        ),
    )
    output: list[RawArmRetrieval] = []
    for query in ordered_queries:
        bm25_rankings = tuple(
            tuple(lexical_backend.rank(condition, TRIALGPT_RETRIEVAL_SOURCE_DEPTH))
            for condition in query.conditions
        )
        medcpt_rankings = tuple(
            tuple(ranking)
            for ranking in dense_backend.rank(
                conditions=query.conditions,
                trials=trials,
                depth=TRIALGPT_RETRIEVAL_SOURCE_DEPTH,
            )
        )
        raw = RawArmRetrieval(
            query=query,
            bm25_rankings=bm25_rankings,
            medcpt_rankings=medcpt_rankings,
        )
        _validate_raw_arm(raw)
        output.append(raw)
    return tuple(output)


def _validate_query_contract(query: GeneratedArmQuery) -> None:
    expected_prompt = (
        _RECALL_QUERY_PROMPT_CONTRACT_ID
        if query.arm_id == "recall-explicit"
        else _DEFAULT_QUERY_PROMPT_CONTRACT_ID
    )
    expected_effort = "xhigh" if query.arm_id == "xhigh" else "medium"
    if query.prompt_contract_id != expected_prompt:
        raise TrialGPTRetrievalProducerError(
            f"query arm {query.arm_id!r} prompt contract does not match v2"
        )
    try:
        require_sha256(query.patient_text_sha256, f"query arm {query.arm_id} patient text")
    except SchemaValidationError as exc:
        raise TrialGPTRetrievalProducerError(str(exc)) from exc
    if query.provider_configuration.get("model") != "gpt-5.6-luna" or (
        query.provider_configuration.get("reasoning_effort") != expected_effort
    ):
        raise TrialGPTRetrievalProducerError(
            f"query arm {query.arm_id!r} provider contract does not match v2"
        )
    trace = query.generation_trace
    trace_keys = {
        "attempts",
        "identity_sha256",
        "logical_call_id",
        "output_sha256",
        "provider_id",
        "selected_attempt",
        "status",
    }
    if set(trace) != trace_keys or trace.get("status") != "resolved":
        raise TrialGPTRetrievalProducerError(
            f"query arm {query.arm_id!r} generation trace is incomplete"
        )
    for name in ("identity_sha256", "logical_call_id", "output_sha256"):
        try:
            require_sha256(trace[name], f"query arm {query.arm_id} {name}")
        except SchemaValidationError as exc:
            raise TrialGPTRetrievalProducerError(str(exc)) from exc
    if (
        not isinstance(trace.get("provider_id"), str)
        or not isinstance(trace.get("selected_attempt"), int)
        or isinstance(trace.get("selected_attempt"), bool)
        or not isinstance(trace.get("attempts"), list)
        or not trace["attempts"]
    ):
        raise TrialGPTRetrievalProducerError(
            f"query arm {query.arm_id!r} generation trace values are invalid"
        )


def _validate_raw_arm(raw: RawArmRetrieval) -> None:
    query = raw.query
    if not query.topic_id or query.arm_id not in TRIALGPT_RETRIEVAL_SOURCE_ARMS:
        raise TrialGPTRetrievalProducerError("raw retrieval has an invalid topic or source arm")
    if not query.summary.strip() or not query.prompt_contract_id.strip():
        raise TrialGPTRetrievalProducerError(
            "raw retrieval query summary and prompt contract must be non-empty"
        )
    _validate_query_contract(query)
    if not 1 <= len(query.conditions) <= 32 or any(
        not condition.strip() for condition in query.conditions
    ):
        raise TrialGPTRetrievalProducerError(
            "raw retrieval query must contain 1 to 32 non-empty conditions"
        )
    if len(raw.bm25_rankings) != len(query.conditions) or len(raw.medcpt_rankings) != len(
        query.conditions
    ):
        raise TrialGPTRetrievalProducerError(
            "raw retrieval requires one BM25 and MedCPT ranking per condition"
        )
    for channel, rankings in (
        ("BM25", raw.bm25_rankings),
        ("MedCPT", raw.medcpt_rankings),
    ):
        for condition_index, ranking in enumerate(rankings):
            if len(ranking) != TRIALGPT_RETRIEVAL_SOURCE_DEPTH:
                raise TrialGPTRetrievalProducerError(
                    f"raw {channel} ranking {condition_index} must contain exactly "
                    f"{TRIALGPT_RETRIEVAL_SOURCE_DEPTH} trials"
                )
            if any(not trial_id for trial_id in ranking) or len(set(ranking)) != len(ranking):
                raise TrialGPTRetrievalProducerError(
                    f"raw {channel} ranking {condition_index} has empty or duplicate trial IDs"
                )


def build_source_arm_ranking(raw: RawArmRetrieval) -> SourceArmRanking:
    """Fuse raw condition rankings and preserve each final trial's channel exposure."""

    _validate_raw_arm(raw)
    scores: dict[str, float] = {}
    lexical_exposure: set[str] = set()
    dense_exposure: set[str] = set()
    for condition_index, (bm25, medcpt) in enumerate(
        zip(raw.bm25_rankings, raw.medcpt_rankings, strict=True)
    ):
        condition_factor = 1 / (condition_index + 1)
        lexical_exposure.update(bm25)
        dense_exposure.update(medcpt)
        for ranking in (bm25, medcpt):
            for zero_based_rank, trial_id in enumerate(ranking):
                scores[trial_id] = (
                    scores.get(trial_id, 0.0) + (1 / (zero_based_rank + 20)) * condition_factor
                )
    ranked = sorted(scores, key=lambda trial_id: (-scores[trial_id], trial_id))
    if len(ranked) < TRIALGPT_RETRIEVAL_SOURCE_DEPTH:
        raise TrialGPTRetrievalProducerError(
            f"raw retrieval for topic {raw.query.topic_id!r} arm {raw.query.arm_id!r} "
            f"produced fewer than {TRIALGPT_RETRIEVAL_SOURCE_DEPTH} candidates"
        )
    return SourceArmRanking(
        topic_id=raw.query.topic_id,
        arm_id=raw.query.arm_id,
        candidates=tuple(
            SourceArmCandidate(
                trial_id=trial_id,
                rank=rank,
                lexical_exposed=trial_id in lexical_exposure,
                dense_exposed=trial_id in dense_exposure,
            )
            for rank, trial_id in enumerate(ranked[:TRIALGPT_RETRIEVAL_SOURCE_DEPTH], start=1)
        ),
    )


def _validate_source_ranking(ranking: SourceArmRanking) -> None:
    if not ranking.topic_id:
        raise TrialGPTRetrievalProducerError("source arm topic_id must be non-empty")
    if ranking.arm_id not in TRIALGPT_RETRIEVAL_SOURCE_ARMS:
        raise TrialGPTRetrievalProducerError(f"unsupported source arm {ranking.arm_id!r}")
    if len(ranking.candidates) != TRIALGPT_RETRIEVAL_SOURCE_DEPTH:
        raise TrialGPTRetrievalProducerError(
            f"source arm {ranking.arm_id!r} for topic {ranking.topic_id!r} must contain "
            f"exactly {TRIALGPT_RETRIEVAL_SOURCE_DEPTH} candidates"
        )
    trial_ids: set[str] = set()
    for expected_rank, candidate in enumerate(ranking.candidates, start=1):
        if candidate.rank != expected_rank:
            raise TrialGPTRetrievalProducerError(
                f"source arm {ranking.arm_id!r} for topic {ranking.topic_id!r} has "
                "non-contiguous ranks"
            )
        if not candidate.trial_id:
            raise TrialGPTRetrievalProducerError("source arm trial_id must be non-empty")
        if candidate.trial_id in trial_ids:
            raise TrialGPTRetrievalProducerError(
                f"source arm {ranking.arm_id!r} for topic {ranking.topic_id!r} contains "
                f"duplicate trial {candidate.trial_id!r}"
            )
        if not isinstance(candidate.lexical_exposed, bool) or not isinstance(
            candidate.dense_exposed, bool
        ):
            raise TrialGPTRetrievalProducerError(
                "source arm channel exposure values must be booleans"
            )
        trial_ids.add(candidate.trial_id)


def _features(
    trial_id: str,
    arm_candidates: Mapping[str, Mapping[str, SourceArmCandidate]],
) -> ConsensusFeatures:
    candidates = tuple(
        arm_candidates[arm_id].get(trial_id) for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
    )
    ranks = (
        candidates[0].rank if candidates[0] is not None else None,
        candidates[1].rank if candidates[1] is not None else None,
        candidates[2].rank if candidates[2] is not None else None,
    )
    observed_ranks = tuple(rank for rank in ranks if rank is not None)
    if not observed_ranks:
        raise AssertionError("consensus candidate must occur in at least one source arm")
    best_rank = min(observed_ranks)
    best_rank_quality = 1.0 - (best_rank - 1) / (TRIALGPT_RETRIEVAL_SOURCE_DEPTH - 1)
    reciprocal_ranks = (
        0.0 if ranks[0] is None else 1.0 / ranks[0],
        0.0 if ranks[1] is None else 1.0 / ranks[1],
        0.0 if ranks[2] is None else 1.0 / ranks[2],
    )
    agreement_count = sum(
        1
        for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
        if (
            (candidate := arm_candidates[arm_id].get(trial_id)) is not None
            and candidate.lexical_exposed
            and candidate.dense_exposed
        )
    )
    return ConsensusFeatures(
        arm_count=len(observed_ranks),
        best_rank_quality=best_rank_quality,
        per_arm_reciprocal_rank=reciprocal_ranks,
        lexical_dense_agreement_count=agreement_count,
    )


def _score(features: ConsensusFeatures) -> float:
    return (
        (6 / 17) * (features.arm_count / 3)
        + (4 / 17) * features.best_rank_quality
        + (4 / 17) * (sum(features.per_arm_reciprocal_rank) / 3)
        + (3 / 17) * (features.lexical_dense_agreement_count / 3)
    )


def compress_three_luna_rankings(
    rankings: Sequence[SourceArmRanking],
) -> Mapping[str, tuple[ConsensusCandidate, ...]]:
    """Compress exact three-arm rankings into one deterministic depth-2,000 ranking per topic."""

    by_topic: defaultdict[str, dict[str, SourceArmRanking]] = defaultdict(dict)
    for ranking in rankings:
        _validate_source_ranking(ranking)
        if ranking.arm_id in by_topic[ranking.topic_id]:
            raise TrialGPTRetrievalProducerError(
                f"duplicate source arm {ranking.arm_id!r} for topic {ranking.topic_id!r}"
            )
        by_topic[ranking.topic_id][ranking.arm_id] = ranking
    if not by_topic:
        raise TrialGPTRetrievalProducerError("at least one topic is required")

    output: dict[str, tuple[ConsensusCandidate, ...]] = {}
    expected_arms = set(TRIALGPT_RETRIEVAL_SOURCE_ARMS)
    for topic_id in sorted(by_topic, key=_topic_sort_key):
        topic_rankings = by_topic[topic_id]
        if set(topic_rankings) != expected_arms:
            missing = sorted(expected_arms - set(topic_rankings))
            extra = sorted(set(topic_rankings) - expected_arms)
            raise TrialGPTRetrievalProducerError(
                f"topic {topic_id!r} source arms do not match the producer contract: "
                f"missing={missing}, extra={extra}"
            )
        arm_candidates = {
            arm_id: {
                candidate.trial_id: candidate for candidate in topic_rankings[arm_id].candidates
            }
            for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
        }
        trial_ids = set().union(*(set(candidates) for candidates in arm_candidates.values()))
        scored = [
            (trial_id, (features := _features(trial_id, arm_candidates)), _score(features))
            for trial_id in trial_ids
        ]
        scored.sort(key=lambda item: (-item[2], item[0]))
        output[topic_id] = tuple(
            ConsensusCandidate(
                trial_id=trial_id,
                rank=rank,
                score=score,
                features=features,
            )
            for rank, (trial_id, features, score) in enumerate(
                scored[:TRIALGPT_RETRIEVAL_OUTPUT_DEPTH], start=1
            )
        )
    return MappingProxyType(output)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_sha256(payload: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, JsonValue]]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(_canonical_json_bytes(row))
            stream.write(b"\n")


def _validate_binding(binding: FrozenRetrievalBinding) -> None:
    for name in ("run_id", "benchmark_lineage"):
        if not getattr(binding, name).strip():
            raise TrialGPTRetrievalProducerError(f"retrieval binding {name} must be non-empty")
    for name in (
        "prepared_snapshot_id",
        "evaluation_package_id",
        "task_input_id",
        "pool_receipt_id",
        "pool_ids_sha256",
        "protocol_approval_id",
        "producer_dependency_lock_sha256",
        "release_manifest_id",
        "public_tree_id",
        "package_artifact_id",
    ):
        try:
            require_sha256(getattr(binding, name), f"retrieval binding {name}")
        except SchemaValidationError as exc:
            raise TrialGPTRetrievalProducerError(str(exc)) from exc
    if isinstance(binding.pool_count, bool) or binding.pool_count < TRIALGPT_RETRIEVAL_OUTPUT_DEPTH:
        raise TrialGPTRetrievalProducerError(
            f"retrieval binding pool_count must be at least {TRIALGPT_RETRIEVAL_OUTPUT_DEPTH}"
        )
    if binding.protocol_approval_id != FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256:
        raise TrialGPTRetrievalProducerError(
            "retrieval binding does not name the frozen TrialGPT producer protocol"
        )
    if _GIT_COMMIT.fullmatch(binding.producer_git_commit) is None:
        raise TrialGPTRetrievalProducerError(
            "retrieval binding producer_git_commit must be a lowercase 40-character Git commit"
        )


def _validate_retrieval_implementation(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TrialGPTRetrievalProducerError("retrieval implementation must be an object")
    bm25 = value.get("bm25")
    medcpt = value.get("medcpt")
    fusion = value.get("fusion")
    if not isinstance(bm25, Mapping) or bm25.get("implementation") != TRIALGPT_RETRIEVAL_BM25_ID:
        raise TrialGPTRetrievalProducerError("retrieval implementation does not use frozen BM25 v2")
    required_medcpt = {
        "article_model": "ncbi/MedCPT-Article-Encoder",
        "article_revision": "d05a736da4bb84ee4057b7f7999485be6ed85465",
        "article_resolved_revision": "d05a736da4bb84ee4057b7f7999485be6ed85465",
        "index": "exact_numpy_inner_product",
        "normalization": "none",
        "pooling": "last_hidden_state[:, 0, :]",
        "precision": "float32",
        "query_model": "ncbi/MedCPT-Query-Encoder",
        "query_revision": "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc",
        "query_resolved_revision": "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc",
    }
    if not isinstance(medcpt, Mapping) or any(
        medcpt.get(name) != expected for name, expected in required_medcpt.items()
    ):
        raise TrialGPTRetrievalProducerError(
            "retrieval implementation does not use the frozen exact MedCPT configuration"
        )
    trial_embeddings = medcpt.get("trial_embeddings")
    if not isinstance(trial_embeddings, Mapping):
        raise TrialGPTRetrievalProducerError(
            "retrieval implementation lacks MedCPT embedding evidence"
        )
    for name in ("input_signature", "sha256"):
        try:
            require_sha256(trial_embeddings.get(name), f"MedCPT trial embeddings {name}")
        except SchemaValidationError as exc:
            raise TrialGPTRetrievalProducerError(str(exc)) from exc
    if (
        not isinstance(fusion, Mapping)
        or fusion.get("condition_weight") != "1 / (zero_based_condition_index + 1)"
        or fusion.get("rrf_k") != 20
        or fusion.get("tie_rule") != "score descending, then trial_id ascending"
    ):
        raise TrialGPTRetrievalProducerError("retrieval implementation does not use frozen fusion")


def _source_rows(rankings: Sequence[SourceArmRanking]) -> list[dict[str, JsonValue]]:
    return [
        {
            "arm_id": ranking.arm_id,
            "dense_exposed": candidate.dense_exposed,
            "lexical_exposed": candidate.lexical_exposed,
            "rank": candidate.rank,
            "topic_id": ranking.topic_id,
            "trial_id": candidate.trial_id,
        }
        for ranking in sorted(
            rankings,
            key=lambda item: (
                _topic_sort_key(item.topic_id),
                TRIALGPT_RETRIEVAL_SOURCE_ARMS.index(item.arm_id),
            ),
        )
        for candidate in ranking.candidates
    ]


def _raw_sort_key(raw: RawArmRetrieval) -> tuple[tuple[int, int | str], int]:
    return (
        _topic_sort_key(raw.query.topic_id),
        TRIALGPT_RETRIEVAL_SOURCE_ARMS.index(raw.query.arm_id),
    )


def _query_rows(raw_retrievals: Sequence[RawArmRetrieval]) -> list[dict[str, JsonValue]]:
    return [
        {
            "arm_id": raw.query.arm_id,
            "conditions": list(raw.query.conditions),
            "generation_trace": dict(raw.query.generation_trace),
            "patient_text_sha256": raw.query.patient_text_sha256,
            "prompt_contract_id": raw.query.prompt_contract_id,
            "provider_configuration": dict(raw.query.provider_configuration),
            "summary": raw.query.summary,
            "topic_id": raw.query.topic_id,
        }
        for raw in sorted(raw_retrievals, key=_raw_sort_key)
    ]


def _raw_channel_rows(raw_retrievals: Sequence[RawArmRetrieval]) -> list[dict[str, JsonValue]]:
    return [
        {
            "arm_id": raw.query.arm_id,
            "bm25_trial_ids": list(bm25),
            "condition": condition,
            "condition_index": condition_index,
            "medcpt_trial_ids": list(medcpt),
            "topic_id": raw.query.topic_id,
        }
        for raw in sorted(raw_retrievals, key=_raw_sort_key)
        for condition_index, (condition, bm25, medcpt) in enumerate(
            zip(raw.query.conditions, raw.bm25_rankings, raw.medcpt_rankings, strict=True)
        )
    ]


def _feature_rows(
    consensus: Mapping[str, tuple[ConsensusCandidate, ...]],
) -> list[dict[str, JsonValue]]:
    return [
        {
            "arm_count": candidate.features.arm_count,
            "best_rank_quality": candidate.features.best_rank_quality,
            "lexical_dense_agreement_count": candidate.features.lexical_dense_agreement_count,
            "per_arm_reciprocal_rank": list(candidate.features.per_arm_reciprocal_rank),
            "rank": candidate.rank,
            "score": candidate.score,
            "topic_id": topic_id,
            "trial_id": candidate.trial_id,
        }
        for topic_id in sorted(consensus, key=_topic_sort_key)
        for candidate in consensus[topic_id]
    ]


def _retrieval_rows(
    consensus: Mapping[str, tuple[ConsensusCandidate, ...]],
) -> list[dict[str, JsonValue]]:
    return [
        {
            "configuration": _RANKING_CONFIGURATION,
            "rank": candidate.rank,
            "score": candidate.score,
            "topic_id": topic_id,
            "trial_id": candidate.trial_id,
        }
        for topic_id in sorted(consensus, key=_topic_sort_key)
        for candidate in consensus[topic_id]
    ]


def _producer_method_contract() -> dict[str, JsonValue]:
    return {
        "classification": "judgment-blind deterministic fixed-pool compressor",
        "feature_definitions": {
            "arm_count": "number of source top-2000 rankings containing the trial",
            "best_rank_quality": "1 - (best_rank - 1) / 1999",
            "lexical_dense_agreement_count": (
                "number of source arms where both raw BM25 and raw MedCPT exposed the trial"
            ),
            "per_arm_reciprocal_rank": (
                "1 / rank when present, else 0, ordered medium, xhigh, recall-explicit"
            ),
        },
        "formula": (
            "(6/17)*(arm_count/3) + (4/17)*best_rank_quality + "
            "(4/17)*(sum(per_arm_reciprocal_rank)/3) + "
            "(3/17)*(lexical_dense_agreement_count/3)"
        ),
        "producer_contract_id": TRIALGPT_RETRIEVAL_PRODUCER_CONTRACT_ID,
        "provider_calls_during_compression": 0,
        "qrels_access_during_compression": False,
        "source_depth": TRIALGPT_RETRIEVAL_SOURCE_DEPTH,
        "source_order": list(TRIALGPT_RETRIEVAL_SOURCE_ARMS),
        "tie_rule": "score descending, then trial_id ascending",
    }


def _sidecar_path(artifact_directory: Path, value: object, *, role: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise TrialGPTRetrievalProducerError(f"fresh retrieval {role} filename is invalid")
    path = artifact_directory.resolve() / value
    if not path.is_file():
        raise TrialGPTRetrievalProducerError(f"fresh retrieval {role} does not exist: {path}")
    return path


def _read_source_rankings(path: Path) -> tuple[list[dict[str, JsonValue]], list[SourceArmRanking]]:
    row_keys = {
        "arm_id",
        "dense_exposed",
        "lexical_exposed",
        "rank",
        "topic_id",
        "trial_id",
    }
    source_rows: list[dict[str, JsonValue]] = []
    grouped: defaultdict[tuple[str, str], list[SourceArmCandidate]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval source row {line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, dict) or set(row) != row_keys:
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval source row {line_number} has unexpected fields"
                )
            topic_id = row["topic_id"]
            arm_id = row["arm_id"]
            trial_id = row["trial_id"]
            rank = row["rank"]
            lexical_exposed = row["lexical_exposed"]
            dense_exposed = row["dense_exposed"]
            if not isinstance(topic_id, str) or not isinstance(arm_id, str):
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval source row {line_number} has invalid topic or arm"
                )
            if not isinstance(trial_id, str) or not trial_id:
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval source row {line_number} has invalid trial ID"
                )
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval source row {line_number} has invalid rank"
                )
            if not isinstance(lexical_exposed, bool) or not isinstance(dense_exposed, bool):
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval source row {line_number} has invalid channel exposure"
                )
            source_rows.append(row)
            grouped[(topic_id, arm_id)].append(
                SourceArmCandidate(
                    trial_id=trial_id,
                    rank=rank,
                    lexical_exposed=lexical_exposed,
                    dense_exposed=dense_exposed,
                )
            )
    rankings = [
        SourceArmRanking(topic_id, arm_id, tuple(candidates))
        for (topic_id, arm_id), candidates in sorted(
            grouped.items(),
            key=lambda item: (
                _topic_sort_key(item[0][0]),
                TRIALGPT_RETRIEVAL_SOURCE_ARMS.index(item[0][1])
                if item[0][1] in TRIALGPT_RETRIEVAL_SOURCE_ARMS
                else len(TRIALGPT_RETRIEVAL_SOURCE_ARMS),
            ),
        )
    ]
    if source_rows != _source_rows(rankings):
        raise TrialGPTRetrievalProducerError(
            "fresh retrieval source rankings are not in canonical topic, arm, rank order"
        )
    return source_rows, rankings


def _string_tuple(value: object, *, role: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TrialGPTRetrievalProducerError(f"fresh retrieval {role} must be a string list")
    return tuple(value)


def _query_from_row(row: object, *, line_number: int) -> GeneratedArmQuery:
    query_keys = {
        "arm_id",
        "conditions",
        "generation_trace",
        "patient_text_sha256",
        "prompt_contract_id",
        "provider_configuration",
        "summary",
        "topic_id",
    }
    if not isinstance(row, dict) or set(row) != query_keys:
        raise TrialGPTRetrievalProducerError(
            f"fresh retrieval query row {line_number} has unexpected fields"
        )
    topic_id = row["topic_id"]
    arm_id = row["arm_id"]
    summary = row["summary"]
    prompt_contract_id = row["prompt_contract_id"]
    patient_text_sha256 = row["patient_text_sha256"]
    provider_configuration = row["provider_configuration"]
    generation_trace = row["generation_trace"]
    if (
        not all(
            isinstance(value, str)
            for value in (topic_id, arm_id, summary, prompt_contract_id, patient_text_sha256)
        )
        or not isinstance(provider_configuration, dict)
        or not isinstance(generation_trace, dict)
    ):
        raise TrialGPTRetrievalProducerError(
            f"fresh retrieval query row {line_number} has invalid values"
        )
    query = GeneratedArmQuery(
        topic_id=topic_id,
        arm_id=arm_id,
        summary=summary,
        conditions=_string_tuple(row["conditions"], role="query conditions"),
        patient_text_sha256=patient_text_sha256,
        prompt_contract_id=prompt_contract_id,
        provider_configuration=provider_configuration,
        generation_trace=generation_trace,
    )
    _validate_query_contract(query)
    return query


def write_generated_arm_queries(queries: Sequence[GeneratedArmQuery], path: Path) -> None:
    """Write restartable provider output for transfer to the retrieval machine."""

    raw_shells = tuple(
        RawArmRetrieval(query=query, bm25_rankings=(), medcpt_rankings=()) for query in queries
    )
    membership = {(query.topic_id, query.arm_id) for query in queries}
    topics = {query.topic_id for query in queries}
    expected = {
        (topic_id, arm_id) for topic_id in topics for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
    }
    if not queries or len(membership) != len(queries) or membership != expected:
        raise TrialGPTRetrievalProducerError(
            "generated query artifact requires exactly three arms per unique topic"
        )
    for query in queries:
        _validate_query_contract(query)
    if path.exists():
        raise TrialGPTRetrievalProducerError(
            f"query artifact writer refuses to overwrite existing path: {path.resolve()}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, _query_rows(raw_shells))


def load_generated_arm_queries(path: Path) -> tuple[GeneratedArmQuery, ...]:
    """Load and fail closed on a transferred three-arm query-plan artifact."""

    queries: list[GeneratedArmQuery] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrialGPTRetrievalProducerError(
                        f"fresh retrieval query row {line_number} is not valid JSON"
                    ) from exc
                queries.append(_query_from_row(row, line_number=line_number))
    except OSError as exc:
        raise TrialGPTRetrievalProducerError(f"query artifact is not readable: {path}") from exc
    ordered = tuple(
        sorted(
            queries,
            key=lambda query: (
                _topic_sort_key(query.topic_id),
                TRIALGPT_RETRIEVAL_SOURCE_ARMS.index(query.arm_id),
            ),
        )
    )
    if tuple(queries) != ordered:
        raise TrialGPTRetrievalProducerError("query artifact is not in canonical order")
    membership = {(query.topic_id, query.arm_id) for query in ordered}
    topics = {query.topic_id for query in ordered}
    expected = {
        (topic_id, arm_id) for topic_id in topics for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
    }
    if not ordered or len(membership) != len(ordered) or membership != expected:
        raise TrialGPTRetrievalProducerError(
            "query artifact requires exactly three arms per unique topic"
        )
    return ordered


def _read_raw_retrievals(query_path: Path, raw_path: Path) -> list[RawArmRetrieval]:
    queries: dict[tuple[str, str], GeneratedArmQuery] = {}
    query_rows: list[dict[str, JsonValue]] = []
    with query_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval query row {line_number} is not valid JSON"
                ) from exc
            query = _query_from_row(row, line_number=line_number)
            topic_id = query.topic_id
            arm_id = query.arm_id
            key = (topic_id, arm_id)
            if key in queries:
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval query row {line_number} duplicates {key!r}"
                )
            query_rows.append(row)
            queries[key] = query

    raw_keys = {
        "arm_id",
        "bm25_trial_ids",
        "condition",
        "condition_index",
        "medcpt_trial_ids",
        "topic_id",
    }
    raw_rows: list[dict[str, JsonValue]] = []
    channels: defaultdict[
        tuple[str, str], list[tuple[int, str, tuple[str, ...], tuple[str, ...]]]
    ] = defaultdict(list)
    with raw_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval raw-channel row {line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, dict) or set(row) != raw_keys:
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval raw-channel row {line_number} has unexpected fields"
                )
            topic_id = row["topic_id"]
            arm_id = row["arm_id"]
            condition = row["condition"]
            condition_index = row["condition_index"]
            if (
                not isinstance(topic_id, str)
                or not isinstance(arm_id, str)
                or not isinstance(condition, str)
                or isinstance(condition_index, bool)
                or not isinstance(condition_index, int)
            ):
                raise TrialGPTRetrievalProducerError(
                    f"fresh retrieval raw-channel row {line_number} has invalid values"
                )
            raw_rows.append(row)
            channels[(topic_id, arm_id)].append(
                (
                    condition_index,
                    condition,
                    _string_tuple(row["bm25_trial_ids"], role="BM25 ranking"),
                    _string_tuple(row["medcpt_trial_ids"], role="MedCPT ranking"),
                )
            )

    if set(queries) != set(channels):
        raise TrialGPTRetrievalProducerError(
            "fresh retrieval query-plan and raw-channel membership do not match"
        )
    raw_retrievals: list[RawArmRetrieval] = []
    for key, query in queries.items():
        channel_rows = sorted(channels[key])
        if [item[0] for item in channel_rows] != list(range(len(channel_rows))):
            raise TrialGPTRetrievalProducerError(
                f"fresh retrieval raw-channel condition indexes are not contiguous for {key!r}"
            )
        if tuple(item[1] for item in channel_rows) != query.conditions:
            raise TrialGPTRetrievalProducerError(
                f"fresh retrieval raw-channel conditions do not match query plan for {key!r}"
            )
        raw_retrievals.append(
            RawArmRetrieval(
                query=query,
                bm25_rankings=tuple(item[2] for item in channel_rows),
                medcpt_rankings=tuple(item[3] for item in channel_rows),
            )
        )
    if query_rows != _query_rows(raw_retrievals) or raw_rows != _raw_channel_rows(raw_retrievals):
        raise TrialGPTRetrievalProducerError(
            "fresh retrieval query or raw-channel rows are not in canonical order"
        )
    return raw_retrievals


def validate_fresh_retrieval_producer_contract(
    retrieval: FrozenTrialGPTRetrieval,
    *,
    artifact_directory: Path,
) -> None:
    """Prove a retrieval was freshly produced by the complete source-side v2 contract."""

    if retrieval.lock.get("method") != _producer_method_contract():
        raise TrialGPTRetrievalProducerError(
            "fresh retrieval producer method contract does not match v2"
        )
    source_lock = retrieval.lock.get("source_input_lock")
    expected_source_lock_keys = {
        "artifact_sha256",
        "benchmark_lineage",
        "consensus_features_filename",
        "consensus_features_sha256",
        "input_lock_id",
        "pool_count",
        "pool_ids_sha256",
        "pool_receipt_id",
        "producer_contract_id",
        "producer_dependency_lock_sha256",
        "producer_git_commit",
        "protocol_approval_id",
        "query_plans_filename",
        "query_plans_sha256",
        "raw_channels_filename",
        "raw_channels_sha256",
        "release_identity",
        "run_id",
        "source_rankings_filename",
        "source_rankings_sha256",
        "task_input_id",
        "working_tree_dirty",
    }
    if not isinstance(source_lock, Mapping) or set(source_lock) != expected_source_lock_keys:
        raise TrialGPTRetrievalProducerError("fresh retrieval source input lock is incomplete")
    if (
        source_lock["producer_contract_id"] != TRIALGPT_RETRIEVAL_PRODUCER_CONTRACT_ID
        or source_lock["benchmark_lineage"] != retrieval.lock["benchmark_lineage"]
        or source_lock["working_tree_dirty"] is not False
    ):
        raise TrialGPTRetrievalProducerError("fresh retrieval source input identity is invalid")
    for name in (
        "artifact_sha256",
        "consensus_features_sha256",
        "input_lock_id",
        "pool_ids_sha256",
        "pool_receipt_id",
        "producer_dependency_lock_sha256",
        "protocol_approval_id",
        "query_plans_sha256",
        "raw_channels_sha256",
        "source_rankings_sha256",
        "task_input_id",
    ):
        try:
            require_sha256(source_lock[name], f"fresh retrieval {name}")
        except SchemaValidationError as exc:
            raise TrialGPTRetrievalProducerError(str(exc)) from exc
    if source_lock["protocol_approval_id"] != FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256:
        raise TrialGPTRetrievalProducerError(
            "fresh retrieval does not name the frozen TrialGPT producer protocol"
        )
    producer_commit = source_lock["producer_git_commit"]
    if not isinstance(producer_commit, str) or _GIT_COMMIT.fullmatch(producer_commit) is None:
        raise TrialGPTRetrievalProducerError("fresh retrieval producer Git commit is invalid")
    release_identity = source_lock["release_identity"]
    if not isinstance(release_identity, Mapping) or set(release_identity) != {
        "package_artifact_id",
        "public_tree_id",
        "release_manifest_id",
    }:
        raise TrialGPTRetrievalProducerError("fresh retrieval release identity is incomplete")
    for name in ("package_artifact_id", "public_tree_id", "release_manifest_id"):
        try:
            require_sha256(release_identity[name], f"fresh retrieval release identity {name}")
        except SchemaValidationError as exc:
            raise TrialGPTRetrievalProducerError(str(exc)) from exc

    source_path = _sidecar_path(
        artifact_directory,
        source_lock["source_rankings_filename"],
        role="source rankings",
    )
    features_path = _sidecar_path(
        artifact_directory,
        source_lock["consensus_features_filename"],
        role="consensus features",
    )
    query_path = _sidecar_path(
        artifact_directory,
        source_lock["query_plans_filename"],
        role="query plans",
    )
    raw_path = _sidecar_path(
        artifact_directory,
        source_lock["raw_channels_filename"],
        role="raw channels",
    )
    if (
        sha256_file(source_path) != source_lock["source_rankings_sha256"]
        or source_lock["artifact_sha256"] != source_lock["source_rankings_sha256"]
        or sha256_file(features_path) != source_lock["consensus_features_sha256"]
        or sha256_file(query_path) != source_lock["query_plans_sha256"]
        or sha256_file(raw_path) != source_lock["raw_channels_sha256"]
    ):
        raise TrialGPTRetrievalProducerError("fresh retrieval sidecar SHA-256 does not match")

    raw_retrievals = _read_raw_retrievals(query_path, raw_path)
    reproduced_rankings = [build_source_arm_ranking(raw) for raw in raw_retrievals]
    source_rows, rankings = _read_source_rankings(source_path)
    if _source_rows(reproduced_rankings) != source_rows:
        raise TrialGPTRetrievalProducerError(
            "fresh retrieval source rankings do not reproduce from raw channels"
        )
    consensus = compress_three_luna_rankings(rankings)
    if set(consensus) != set(retrieval.rankings):
        raise TrialGPTRetrievalProducerError("fresh retrieval topics do not match source rankings")
    for topic_id, candidates in consensus.items():
        actual = retrieval.rankings[topic_id]
        expected = tuple(
            (candidate.rank, candidate.trial_id, candidate.score) for candidate in candidates
        )
        if tuple((row.rank, row.trial_id, row.score) for row in actual) != expected:
            raise TrialGPTRetrievalProducerError(
                f"fresh retrieval topic {topic_id} does not reproduce from source rankings"
            )
    try:
        feature_rows = [json.loads(line) for line in features_path.read_text().splitlines()]
    except json.JSONDecodeError as exc:
        raise TrialGPTRetrievalProducerError(
            "fresh retrieval consensus features are not valid JSON"
        ) from exc
    if feature_rows != _feature_rows(consensus):
        raise TrialGPTRetrievalProducerError(
            "fresh retrieval consensus features do not reproduce from source rankings"
        )
    source_rankings = retrieval.lock.get("source_rankings")
    expected_arm_hashes = {
        arm_id: content_sha256([row for row in source_rows if row["arm_id"] == arm_id])
        for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
    }
    if source_rankings != expected_arm_hashes:
        raise TrialGPTRetrievalProducerError(
            "fresh retrieval source-arm hashes do not match source rankings"
        )
    _validate_retrieval_implementation(retrieval.lock.get("retrieval_implementation"))


def write_frozen_three_luna_retrieval(
    raw_retrievals: Sequence[RawArmRetrieval],
    *,
    binding: FrozenRetrievalBinding,
    retrieval_implementation: Mapping[str, JsonValue],
    output_directory: Path,
) -> WrittenFrozenRetrieval:
    """Write a fresh v2 retrieval package accepted by the publication preflight."""

    _validate_binding(binding)
    _validate_retrieval_implementation(retrieval_implementation)
    rankings = tuple(build_source_arm_ranking(raw) for raw in raw_retrievals)
    consensus = compress_three_luna_rankings(rankings)
    query_rows = _query_rows(raw_retrievals)
    raw_channel_rows = _raw_channel_rows(raw_retrievals)
    source_rows = _source_rows(rankings)
    feature_rows = _feature_rows(consensus)
    retrieval_rows = _retrieval_rows(consensus)

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    written = WrittenFrozenRetrieval(
        retrieval_path=output_directory / _RANKING_FILENAME,
        lock_path=output_directory / _LOCK_FILENAME,
        source_rankings_path=output_directory / _SOURCE_RANKINGS_FILENAME,
        consensus_features_path=output_directory / _CONSENSUS_FEATURES_FILENAME,
        query_plans_path=output_directory / _QUERY_PLANS_FILENAME,
        raw_channels_path=output_directory / _RAW_CHANNELS_FILENAME,
    )
    targets = (
        written.retrieval_path,
        written.lock_path,
        written.source_rankings_path,
        written.consensus_features_path,
        written.query_plans_path,
        written.raw_channels_path,
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise TrialGPTRetrievalProducerError(
            f"retrieval producer refuses to overwrite existing artifacts: {existing}"
        )

    _write_jsonl(written.query_plans_path, query_rows)
    _write_jsonl(written.raw_channels_path, raw_channel_rows)
    _write_jsonl(written.source_rankings_path, source_rows)
    _write_jsonl(written.consensus_features_path, feature_rows)
    _write_jsonl(written.retrieval_path, retrieval_rows)

    source_rankings: dict[str, JsonValue] = {
        arm_id: content_sha256([row for row in source_rows if row["arm_id"] == arm_id])
        for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
    }
    source_lock_core: dict[str, JsonValue] = {
        "benchmark_lineage": binding.benchmark_lineage,
        "consensus_features_filename": written.consensus_features_path.name,
        "consensus_features_sha256": sha256_file(written.consensus_features_path),
        "pool_count": binding.pool_count,
        "pool_ids_sha256": binding.pool_ids_sha256,
        "pool_receipt_id": binding.pool_receipt_id,
        "producer_contract_id": TRIALGPT_RETRIEVAL_PRODUCER_CONTRACT_ID,
        "producer_dependency_lock_sha256": binding.producer_dependency_lock_sha256,
        "producer_git_commit": binding.producer_git_commit,
        "protocol_approval_id": binding.protocol_approval_id,
        "query_plans_filename": written.query_plans_path.name,
        "query_plans_sha256": sha256_file(written.query_plans_path),
        "raw_channels_filename": written.raw_channels_path.name,
        "raw_channels_sha256": sha256_file(written.raw_channels_path),
        "release_identity": {
            "package_artifact_id": binding.package_artifact_id,
            "public_tree_id": binding.public_tree_id,
            "release_manifest_id": binding.release_manifest_id,
        },
        "run_id": binding.run_id,
        "source_rankings_filename": written.source_rankings_path.name,
        "source_rankings_sha256": sha256_file(written.source_rankings_path),
        "task_input_id": binding.task_input_id,
        "working_tree_dirty": False,
    }
    source_input_lock: dict[str, JsonValue] = {
        **source_lock_core,
        "artifact_sha256": source_lock_core["source_rankings_sha256"],
        "input_lock_id": content_sha256(source_lock_core),
    }
    top_hashes: dict[str, JsonValue] = {
        topic_id: _json_sha256(
            [
                {
                    "rank": candidate.rank,
                    "score": candidate.score,
                    "trial_id": candidate.trial_id,
                }
                for candidate in candidates[:TRIALGPT_PUBLICATION_CANDIDATE_DEPTH]
            ]
        )
        for topic_id, candidates in consensus.items()
    }
    lock: dict[str, JsonValue] = {
        "artifact_type": "taim-trialgpt-frozen-retrieval-lock",
        "benchmark_lineage": binding.benchmark_lineage,
        "evaluation_package_id": binding.evaluation_package_id,
        "method": _producer_method_contract(),
        "prepared_snapshot_id": binding.prepared_snapshot_id,
        "ranking": {
            "byte_size": written.retrieval_path.stat().st_size,
            "configuration": _RANKING_CONFIGURATION,
            "filename": written.retrieval_path.name,
            "output_depth": TRIALGPT_RETRIEVAL_OUTPUT_DEPTH,
            "publication_candidate_depth": TRIALGPT_PUBLICATION_CANDIDATE_DEPTH,
            "row_count": len(retrieval_rows),
            "sha256": sha256_file(written.retrieval_path),
            "topic_count": len(consensus),
        },
        "redistribution": (
            "ranking artifact remains local and must not be copied into a Published Result Bundle"
        ),
        "retrieval_implementation": dict(retrieval_implementation),
        "schema_version": "1.0",
        "source_input_lock": source_input_lock,
        "source_rankings": source_rankings,
        "system_id": TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
        "tokenizer": {
            "id": TRIALGPT_TOKENIZER_ID,
            "sentence_rule": "split on whitespace after ASCII .!? punctuation; strip empty parts",
            "word_rule": "Unicode casefold then regex [a-z0-9]+",
        },
        "top_500_sha256_by_topic": top_hashes,
    }
    written.lock_path.write_text(
        json.dumps(lock, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return written


__all__ = [
    "FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256",
    "TRIALGPT_RETRIEVAL_BM25_ID",
    "TRIALGPT_RETRIEVAL_OUTPUT_DEPTH",
    "TRIALGPT_RETRIEVAL_PRODUCER_CONTRACT_ID",
    "TRIALGPT_RETRIEVAL_SOURCE_ARMS",
    "TRIALGPT_RETRIEVAL_SOURCE_DEPTH",
    "TRIALGPT_RETRIEVAL_TARGETS",
    "ConsensusCandidate",
    "ConsensusFeatures",
    "DenseRetrievalBackend",
    "FrozenRetrievalBinding",
    "GeneratedArmQuery",
    "LexicalRetrievalBackend",
    "RawArmRetrieval",
    "RetrievalQueryTopic",
    "SourceArmCandidate",
    "SourceArmRanking",
    "TrialGPTRetrievalProducerError",
    "TrialGPTRetrievalTarget",
    "WrittenFrozenRetrieval",
    "build_source_arm_ranking",
    "compress_three_luna_rankings",
    "generate_three_luna_queries",
    "load_generated_arm_queries",
    "resolve_trialgpt_retrieval_target",
    "retrieve_three_luna_queries",
    "validate_fresh_retrieval_producer_contract",
    "write_frozen_three_luna_retrieval",
    "write_generated_arm_queries",
]
