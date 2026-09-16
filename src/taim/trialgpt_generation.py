"""Provider-neutral, validated generation for the TrialGPT pipeline.

This module deliberately knows nothing about a transport vendor.  Providers return raw text;
the runner owns JSON parsing, semantic validation, bounded repair, provenance, and caching.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Protocol, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised by Windows package import smoke
    fcntl = None  # type: ignore[assignment]

PROTOCOL_VERSION = "trialgpt-generation-v1"
CACHE_SCHEMA_VERSION = "trialgpt-generation-cache-v1"
MAX_ATTEMPTS = 3
Validation = Callable[[object], tuple[str, ...]]


class GenerationTransportError(RuntimeError):
    """A provider transport failure that is safe to retry with the original prompt."""


class GenerationCacheInUseError(RuntimeError):
    """The cache belongs to another active benchmark process."""


class GenerationCacheRunLease:
    """Hold one crash-safe, process-exclusive lease for a generation cache."""

    def __init__(self, cache_path: Path, *, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        resolved_cache = cache_path.resolve()
        self.cache_path = resolved_cache
        self.run_id = run_id
        self.lock_path = resolved_cache.with_name(resolved_cache.name + ".run.lock")
        self._stream: TextIO | None = None

    def __enter__(self) -> GenerationCacheRunLease:
        if fcntl is None:
            raise GenerationCacheInUseError(
                "TrialGPT provider execution requires POSIX advisory file locking"
            )
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise GenerationCacheInUseError(
                f"TrialGPT run {self.run_id!r} is already active for generation cache "
                f"{self.cache_path}"
            ) from exc
        self._stream = stream
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        stream = self._stream
        self._stream = None
        if stream is not None:
            if fcntl is None:  # pragma: no cover - __enter__ cannot succeed on Windows
                stream.close()
                return
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


class GenerationProvider(Protocol):
    """The only interface a Codex, Ollama, or test provider needs to implement."""

    provider_id: str

    def generate(
        self,
        *,
        prompt: str,
        output_schema: Mapping[str, object],
        attempt: int,
        logical_call_id: str,
    ) -> ProviderResponse: ...


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Raw provider output and optional usage metadata."""

    raw_output: str
    usage: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One logical stage call, independent of any provider transport."""

    stage: str
    prompt: str
    output_schema: Mapping[str, object]
    input_payload: Mapping[str, object]
    provider_config: Mapping[str, object] = field(default_factory=dict)
    validator: Validation | None = None
    normalizer: Callable[[object], object] | None = None
    normalizer_version: str | None = None
    reject_duplicate_keys: bool = False

    def __post_init__(self) -> None:
        if (self.normalizer is None) != (self.normalizer_version is None):
            raise ValueError("normalizer and normalizer_version must be provided together")


@dataclass(frozen=True, slots=True)
class GenerationAttempt:
    """An immutable record of one original or repair attempt."""

    attempt: int
    kind: str
    prompt_sha256: str
    raw_output: str | None
    parsed_output: object | None
    raw_output_sha256: str | None
    parsed_output_sha256: str | None
    validation_errors: tuple[str, ...]
    elapsed_seconds: float
    usage: Mapping[str, object]
    retry_reason: str | None
    transport_error: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "kind": self.kind,
            "prompt_sha256": self.prompt_sha256,
            "raw_output": self.raw_output,
            "parsed_output": self.parsed_output,
            "raw_output_sha256": self.raw_output_sha256,
            "parsed_output_sha256": self.parsed_output_sha256,
            "validation_errors": list(self.validation_errors),
            "elapsed_seconds": self.elapsed_seconds,
            "usage": dict(self.usage),
            "retry_reason": self.retry_reason,
            "transport_error": self.transport_error,
            "metadata": dict(self.metadata),
        }


def provider_failure_classification(
    attempt: GenerationAttempt | Mapping[str, object],
) -> str | None:
    """Return the exact failed provider-call class, or ``None`` for a nonfailure."""

    transport_error: object
    metadata: Mapping[str, object]
    if isinstance(attempt, GenerationAttempt):
        transport_error = attempt.transport_error
        metadata = attempt.metadata
    else:
        transport_error = attempt.get("transport_error")
        raw_metadata = attempt.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    provider_call = metadata.get("provider_call")
    if (
        not isinstance(transport_error, str)
        or not isinstance(provider_call, Mapping)
        or provider_call.get("status") != "failed"
    ):
        return None
    failure_kind = metadata.get("failure_classification", "transport")
    return failure_kind if failure_kind in ("transport", "authentication") else None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The selected valid output or an explicit unresolved result."""

    logical_call_id: str
    identity_sha256: str
    stage: str
    provider_id: str
    status: str
    output: object | None
    attempts: tuple[GenerationAttempt, ...]
    selected_attempt: int | None
    cache_hit: bool = False
    topic_id: str | None = None
    trial_id: str | None = None

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and self.output is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "logical_call_id": self.logical_call_id,
            "identity_sha256": self.identity_sha256,
            "stage": self.stage,
            "provider_id": self.provider_id,
            "status": self.status,
            "output": self.output,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_attempt": self.selected_attempt,
            "cache_hit": self.cache_hit,
            "topic_id": self.topic_id,
            "trial_id": self.trial_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GenerationResult:
        attempts: list[GenerationAttempt] = []
        raw_attempts = value.get("attempts")
        if not isinstance(raw_attempts, list):
            raise ValueError("generation cache entry attempts must be a list")
        for raw in raw_attempts:
            if not isinstance(raw, Mapping):
                raise ValueError("generation cache attempt must be an object")
            errors = raw.get("validation_errors", [])
            if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
                raise ValueError("generation cache validation_errors must be strings")
            attempts.append(
                GenerationAttempt(
                    attempt=_positive_int(raw.get("attempt"), "attempt"),
                    kind=_string(raw.get("kind"), "kind"),
                    prompt_sha256=_string(raw.get("prompt_sha256"), "prompt_sha256"),
                    raw_output=(
                        raw.get("raw_output") if isinstance(raw.get("raw_output"), str) else None
                    ),
                    parsed_output=raw.get("parsed_output"),
                    raw_output_sha256=raw.get("raw_output_sha256")
                    if isinstance(raw.get("raw_output_sha256"), str)
                    else None,
                    parsed_output_sha256=raw.get("parsed_output_sha256")
                    if isinstance(raw.get("parsed_output_sha256"), str)
                    else None,
                    validation_errors=tuple(errors),
                    elapsed_seconds=_number(raw.get("elapsed_seconds"), "elapsed_seconds"),
                    usage=_mapping(raw.get("usage"), "usage"),
                    retry_reason=raw.get("retry_reason")
                    if isinstance(raw.get("retry_reason"), str)
                    else None,
                    transport_error=raw.get("transport_error")
                    if isinstance(raw.get("transport_error"), str)
                    else None,
                    metadata=_mapping(raw.get("metadata"), "metadata"),
                )
            )
        selected = value.get("selected_attempt")
        if selected is not None and (isinstance(selected, bool) or not isinstance(selected, int)):
            raise ValueError("generation cache selected_attempt must be an integer or null")
        topic_id = value.get("topic_id")
        trial_id = value.get("trial_id")
        return cls(
            logical_call_id=_string(value.get("logical_call_id"), "logical_call_id"),
            identity_sha256=_string(value.get("identity_sha256"), "identity_sha256"),
            stage=_string(value.get("stage"), "stage"),
            provider_id=_string(value.get("provider_id"), "provider_id"),
            status=_string(value.get("status"), "status"),
            output=value.get("output"),
            attempts=tuple(attempts),
            selected_attempt=selected,
            cache_hit=True,
            topic_id=topic_id if isinstance(topic_id, str) else None,
            trial_id=trial_id if isinstance(trial_id, str) else None,
        )


class GenerationCache:
    """Transactional cache keyed by the complete logical-call identity.

    SQLite avoids rewriting the complete cache at every checkpoint. Completed logical calls are
    committed in bounded transactions, while callers can force a final durable flush.
    """

    def __init__(self, path: Path, *, flush_every: int = 1) -> None:
        if isinstance(flush_every, bool) or not isinstance(flush_every, int) or flush_every < 1:
            raise ValueError("flush_every must be a positive integer")
        self.path = path
        self.flush_every = flush_every
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None
        self._pending: dict[str, str] = {}

    def _connection_locked(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS generation_results ("
                "identity_sha256 TEXT PRIMARY KEY, result_json TEXT NOT NULL)"
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                    (CACHE_SCHEMA_VERSION,),
                )
                connection.commit()
            elif row[0] != CACHE_SCHEMA_VERSION:
                raise ValueError(
                    f"generation cache schema is {row[0]!r}, expected {CACHE_SCHEMA_VERSION!r}"
                )
        except (OSError, sqlite3.DatabaseError) as exc:
            raise ValueError(f"invalid generation cache {self.path}: {exc}") from exc
        self._connection = connection
        return connection

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        connection = self._connection_locked()
        try:
            connection.executemany(
                "INSERT OR REPLACE INTO generation_results(identity_sha256, result_json) "
                "VALUES (?, ?)",
                tuple(self._pending.items()),
            )
            connection.commit()
        except sqlite3.DatabaseError:
            connection.rollback()
            raise
        self._pending.clear()

    def load(self, identity: str, *, include_unresolved: bool = False) -> GenerationResult | None:
        with self._lock:
            raw = self._pending.get(identity)
            if raw is None:
                row = (
                    self._connection_locked()
                    .execute(
                        "SELECT result_json FROM generation_results WHERE identity_sha256 = ?",
                        (identity,),
                    )
                    .fetchone()
                )
                raw = None if row is None else row[0]
            if raw is None:
                return None
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid cached generation result {identity}: {exc}") from exc
            if not isinstance(entry, Mapping):
                return None
            status = entry.get("status")
            if status != "resolved" and not (include_unresolved and status == "unresolved"):
                return None
            return GenerationResult.from_dict(entry)

    def store(self, result: GenerationResult) -> None:
        with self._lock:
            self._pending[result.identity_sha256] = canonical_json(result.to_dict())
            if len(self._pending) >= self.flush_every:
                self._flush_locked()

    def flush(self) -> None:
        """Persist all results accepted since the last checkpoint."""

        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush pending results and close the database connection."""

        with self._lock:
            self._flush_locked()
            if self._connection is not None:
                self._connection.close()
                self._connection = None


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_identity(request: GenerationRequest, provider_id: str) -> str:
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "provider_id": provider_id,
        "stage": request.stage,
        "prompt": request.prompt,
        "output_schema": request.output_schema,
        "input_payload": request.input_payload,
        "provider_config": request.provider_config,
    }
    if request.normalizer_version is not None:
        identity["normalizer_version"] = request.normalizer_version
    if request.reject_duplicate_keys:
        identity["json_decoding"] = "reject-duplicate-keys-v1"
    return sha256_text(canonical_json(identity))


def repair_prompt(request: GenerationRequest, raw_output: str, errors: Sequence[str]) -> str:
    return (
        request.prompt + "\n\nYour previous response was invalid. Return only one JSON value "
        "matching this schema.\n"
        + "Validation errors:\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\nPrevious response:\n"
        + raw_output
        + "\nRequired schema:\n"
        + canonical_json(request.output_schema)
    )


class _DuplicateJSONKeyError(ValueError):
    """An ambiguous model response must be repaired, not treated as a transport failure."""


def _unique_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def run_generation(
    request: GenerationRequest,
    provider: GenerationProvider,
    *,
    cache: GenerationCache | None = None,
    reuse_unresolved: bool = False,
) -> GenerationResult:
    """Run at most three attempts and return the first semantically valid JSON result."""

    identity = request_identity(request, provider.provider_id)
    prior_transport_attempts: tuple[GenerationAttempt, ...] = ()
    cached = cache.load(identity, include_unresolved=True) if cache is not None else None
    if cached is not None and not cached.resolved:
        failure_kinds = {provider_failure_classification(attempt) for attempt in cached.attempts}
        if "authentication" in failure_kinds:
            return cached
        final_invocation_size = len(cached.attempts) % MAX_ATTEMPTS or MAX_ATTEMPTS
        final_invocation_failure_kinds = {
            provider_failure_classification(attempt)
            for attempt in cached.attempts[-final_invocation_size:]
        }
        if "transport" in final_invocation_failure_kinds:
            prior_transport_attempts = cached.attempts
            cached = None
        elif not reuse_unresolved:
            cached = None
    if cached is not None:
        return cached
    topic_id = request.input_payload.get("topic_id")
    trial_id = request.input_payload.get("trial_id")
    topic_id = topic_id if isinstance(topic_id, str) else None
    trial_id = trial_id if isinstance(trial_id, str) else None

    attempts = list(prior_transport_attempts)
    attempt_offset = len(attempts)
    prompt = request.prompt
    retry_reason: str | None = "transport_failure" if prior_transport_attempts else None
    for invocation_attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_number = attempt_offset + invocation_attempt
        kind = "original" if prompt == request.prompt else "repair"
        started = time.perf_counter()
        try:
            response = provider.generate(
                prompt=prompt,
                output_schema=request.output_schema,
                attempt=attempt_number,
                logical_call_id=identity,
            )
        except Exception as exc:  # provider transports are intentionally opaque to the protocol
            failure_kind = getattr(exc, "failure_kind", "transport")
            if failure_kind not in ("transport", "authentication"):
                failure_kind = "transport"
            attempts.append(
                GenerationAttempt(
                    attempt_number,
                    kind,
                    sha256_text(prompt),
                    None,
                    None,
                    None,
                    None,
                    ("transport failure",),
                    time.perf_counter() - started,
                    {},
                    "transport_failure",
                    "provider transport failure; detail_sha256="
                    + sha256_text(f"{type(exc).__name__}: {exc}"),
                    {
                        "failure_classification": failure_kind,
                        "provider_call": {"status": "failed"},
                        "provider_configuration": dict(request.provider_config),
                    },
                )
            )
            if failure_kind == "authentication":
                result = GenerationResult(
                    identity,
                    identity,
                    request.stage,
                    provider.provider_id,
                    "unresolved",
                    None,
                    tuple(attempts),
                    None,
                    topic_id=topic_id,
                    trial_id=trial_id,
                )
                if cache is not None:
                    cache.store(result)
                    cache.flush()
                return result
            retry_reason = "transport_failure"
            prompt = request.prompt
            continue

        raw = response.raw_output
        parsed: object | None = None
        errors: tuple[str, ...]
        try:
            parsed = json.loads(
                raw, object_pairs_hook=_unique_json_keys if request.reject_duplicate_keys else None
            )
            errors = (
                ()
                if isinstance(parsed, (dict, list))
                else ("response must be a JSON object or array",)
            )
        except json.JSONDecodeError as exc:
            errors = (f"invalid JSON: {exc.msg}",)
        except _DuplicateJSONKeyError as exc:
            errors = (f"invalid JSON: {exc}",)
        if not errors and request.normalizer is not None:
            try:
                parsed = request.normalizer(parsed)
            except (TypeError, ValueError) as exc:
                errors = (f"normalization failed: {exc}",)
        if not errors and request.validator is not None:
            errors = tuple(request.validator(parsed))
        attempts.append(
            GenerationAttempt(
                attempt_number,
                kind,
                sha256_text(prompt),
                raw,
                parsed,
                sha256_text(raw),
                sha256_text(canonical_json(parsed)) if parsed is not None else None,
                errors,
                time.perf_counter() - started,
                response.usage,
                retry_reason,
                metadata=response.metadata,
            )
        )
        if not errors:
            result = GenerationResult(
                identity,
                identity,
                request.stage,
                provider.provider_id,
                "resolved",
                parsed,
                tuple(attempts),
                attempt_number,
                topic_id=topic_id,
                trial_id=trial_id,
            )
            if cache is not None:
                cache.store(result)
            return result
        retry_reason = (
            "invalid_json" if errors[0].startswith("invalid JSON") else "semantic_validation"
        )
        prompt = repair_prompt(request, raw, errors)

    result = GenerationResult(
        identity,
        identity,
        request.stage,
        provider.provider_id,
        "unresolved",
        None,
        tuple(attempts),
        None,
        topic_id=topic_id,
        trial_id=trial_id,
    )
    if cache is not None:
        cache.store(result)
        cache.flush()
    return result


def validate_keyword_output(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("keyword output must be an object",)
    errors: list[str] = []
    if not isinstance(value.get("summary"), str) or not value["summary"].strip():
        errors.append("summary must be a non-empty string")
    conditions = value.get("conditions")
    if not isinstance(conditions, list) or not conditions or len(conditions) > 32:
        errors.append("conditions must contain 1 to 32 items")
    elif any(not isinstance(item, str) or not item.strip() for item in conditions):
        errors.append("conditions must contain non-empty strings")
    return tuple(errors)


def validate_criterion_matching_output(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("criterion matching output must be an object",)
    assessments = value.get("assessments")
    if not isinstance(assessments, list):
        return ("assessments must be a list",)
    errors: list[str] = []
    allowed = {"supported", "unknown", "not_supported", "included", "excluded", "no_information"}
    for index, assessment in enumerate(assessments):
        if not isinstance(assessment, Mapping):
            errors.append(f"assessment {index} must be an object")
            continue
        if not isinstance(assessment.get("criterion"), str) or not assessment["criterion"].strip():
            errors.append(f"assessment {index} criterion must be a non-empty string")
        if assessment.get("status") not in allowed:
            errors.append(f"assessment {index} status is invalid")
    return tuple(errors)


def validate_final_ranking_output(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("final ranking output must be an object",)
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        return ("score must be a finite number",)
    return ()


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"generation cache {name} must be a non-empty string")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"generation cache {name} must be an object")
    return dict(value)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"generation cache {name} must be numeric")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"generation cache {name} must be positive")
    return value


__all__ = [
    "MAX_ATTEMPTS",
    "PROTOCOL_VERSION",
    "GenerationAttempt",
    "GenerationCache",
    "GenerationCacheInUseError",
    "GenerationCacheRunLease",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResult",
    "GenerationTransportError",
    "ProviderResponse",
    "canonical_json",
    "provider_failure_classification",
    "repair_prompt",
    "request_identity",
    "run_generation",
    "sha256_text",
    "validate_criterion_matching_output",
    "validate_final_ranking_output",
    "validate_keyword_output",
]
