"""Clearing cached failed provider calls before a TrialGPT generation resumes.

Protocol v11 Section 7.1. The release stores a sign-in failure as a final unresolved result, and a
resume returns it unchanged instead of asking again. So before every resume, while no process holds
the cache, the entries that hold nothing but failed provider calls are removed, and the check is
recorded. Nothing holding a model response is ever removed, and resolved entries are never touched.

The check runs on the two caches that store failed calls: the query-plan generation's and the
row's of ``TrialGPT-TAIM-Luna-v1``. It refuses, clearing nothing, when an unresolved entry holds
provider output that the release could not retry or would overwrite (rule 4). Every check, refused
or not, writes one record, and appends the interruption with that record to the interruptions file
the receipt reads. A System that caches only resolved answers has nothing to clear.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from taim.artifacts import sha256_file
from taim.generation import (
    GenerationAttempt,
    GenerationCacheRunLease,
    GenerationResult,
    canonical_json,
    provider_failure_classification,
    sha256_text,
)
from taim.pipelines.trialgpt.query_plan_records import generation_cache_path
from taim.pipelines.trialgpt.retrieval_producer import (
    FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
    TRIALGPT_RETRIEVAL_PROTOCOL_ID,
)
from taim.pipelines.trialgpt.system import trialgpt_run_workspace
from taim.schemas import JsonValue

CACHE_CLEARING_ARTIFACT_TYPE = "taim-trialgpt-generation-cache-clearing"
TRIALGPT_SYSTEM_ID = "TrialGPT-TAIM-Luna-v1"
QUERY_STAGE = "query"
SYSTEM_STAGE = "system"
#: The query-plan generation's one stage (``taim.pipelines.trialgpt.retrieval_producer``).
_KEYWORD_GENERATION = "keyword_generation"
#: Generic causes only: v11 Section 7 never describes an interruption by account or plan.
INTERRUPTION_CAUSES = {
    "usage-allowance": "usage allowance exhausted",
    "signed-out": "expired or signed-out session",
    "provider-failure": "provider or transport failure",
    "host-restart": "host restart",
    "interrupted-process": "interrupted process",
    "codex-version-change": "unavoidable Codex CLI version change",
}


class CacheClearingRefused(RuntimeError):
    """The resume is refused under v11 Section 7.1 rule 4; nothing was cleared."""

    def __init__(self, message: str, record: Path) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True, slots=True)
class ResumeCache:
    """One cache a resume reads, and how the release treats its unresolved entries."""

    path: Path
    stage: str
    system_id: str | None


def resolve_generation_cache(
    *, stage: str, run_id: str, cache: Path | None, system_id: str | None, workspace: Path | None
) -> ResumeCache:
    """The cache file a resume of this stage reads, as the release lays it out."""

    if stage == QUERY_STAGE:
        if cache is None or system_id is not None or workspace is not None:
            raise ValueError("the query stage takes --cache, the --cache the generation was given")
        return ResumeCache(generation_cache_path(cache), QUERY_STAGE, None)
    if stage != SYSTEM_STAGE:
        raise ValueError(f"unknown stage {stage!r}")
    if cache is not None or workspace is None:
        raise ValueError("the system stage takes --system and --workspace, not --cache")
    if system_id != TRIALGPT_SYSTEM_ID:
        raise ValueError(
            f"only {TRIALGPT_SYSTEM_ID}'s cache stores failed calls, not {system_id!r}'s"
        )
    path = trialgpt_run_workspace(workspace, run_id) / "model-generation-v1-cache.sqlite3"
    return ResumeCache(path, SYSTEM_STAGE, system_id)


def _holds_output(attempt: GenerationAttempt) -> bool:
    return any(
        value is not None
        for value in (
            attempt.raw_output,
            attempt.parsed_output,
            attempt.raw_output_sha256,
            attempt.parsed_output_sha256,
        )
    )


def _clearable(entry: GenerationResult) -> bool:
    """Rule 1: unresolved, every attempt a failed provider call, and no provider output."""

    return (
        not entry.resolved
        and bool(entry.attempts)
        and all(
            provider_failure_classification(attempt) in ("authentication", "transport")
            and not _holds_output(attempt)
            for attempt in entry.attempts
        )
    )


def _refusal(entry: GenerationResult, cache: ResumeCache) -> str | None:
    """Rule 4: why the resume is refused for this entry, or ``None``."""

    if entry.resolved or not any(_holds_output(attempt) for attempt in entry.attempts):
        return None
    if any(provider_failure_classification(attempt) is not None for attempt in entry.attempts):
        return "4(a): a failed provider call beside provider output, which cannot be cleared"
    # The System stage resumes with reuse_unresolved, keeping such entries as abstentions.
    if cache.stage == SYSTEM_STAGE:
        return None
    if provider_failure_classification(entry.attempts[-1]) == "transport":
        return None
    return "4(b): provider output the query stage would ask again for, overwriting it"


def _stage_matches(entry: GenerationResult, cache: ResumeCache) -> bool:
    return (entry.stage == _KEYWORD_GENERATION) == (cache.stage == QUERY_STAGE)


def _entry_record(raw: str, entry: GenerationResult) -> dict[str, JsonValue]:
    verbatim = json.loads(raw)
    return {
        "logical_call_id": entry.logical_call_id,
        "identity_sha256": entry.identity_sha256,
        "topic_id": entry.topic_id,
        "trial_id": entry.trial_id,
        "stage": entry.stage,
        "attempt_count": len(entry.attempts),
        "attempt_failure_classes": cast(
            JsonValue, [provider_failure_classification(attempt) for attempt in entry.attempts]
        ),
        "entry_sha256": sha256_text(canonical_json(verbatim)),
        "entry": cast(JsonValue, verbatim),
    }


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _append_interruption(path: Path, interruption: Mapping[str, JsonValue]) -> None:
    interruptions: list[object] = []
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"interruptions file {path} is not a JSON list")
        interruptions = loaded
    interruptions.append(dict(interruption))
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".new")
    staged.write_text(json.dumps(interruptions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staged.replace(path)


def clear_failed_provider_calls(
    cache: ResumeCache,
    *,
    run_id: str,
    cause: str,
    record_dir: Path,
    interruptions: Path,
    now: datetime | None = None,
) -> dict[str, JsonValue]:
    """Run the pre-resume check on one cache; return its record, or raise if it is refused."""

    if cause not in INTERRUPTION_CAUSES:
        raise ValueError(f"cause must be one of {sorted(INTERRUPTION_CAUSES)}")
    if not cache.path.is_file():
        raise ValueError(f"generation cache does not exist: {cache.path}")
    checked_at = _utc(now or datetime.now(UTC))
    record_path = record_dir / f"{run_id}.clearing-{checked_at.replace(':', '')}.json"
    if record_path.exists():
        raise ValueError(f"a clearing record is written once: {record_path} exists")
    # The lease proves no process holds the cache; generation and System runs take the same one.
    with GenerationCacheRunLease(cache.path, run_id=run_id):
        connection = sqlite3.connect(cache.path)
        try:
            # A process that stopped mid-write leaves a log the next open would merge anyway;
            # merge it now so the digests below are those of the whole cache.
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            before = sha256_file(cache.path)
            rows = connection.execute(
                "SELECT identity_sha256, result_json FROM generation_results "
                "ORDER BY identity_sha256"
            ).fetchall()
            entries = [(key, raw, GenerationResult.from_dict(json.loads(raw))) for key, raw in rows]
            foreign = sorted(
                {entry.stage for _, _, entry in entries if not _stage_matches(entry, cache)}
            )
            if foreign:
                raise ValueError(
                    f"{cache.path} holds stages {foreign}, not the {cache.stage} stage's cache"
                )
            refusals = [
                {"logical_call_id": entry.logical_call_id, "rule": reason}
                for _, _, entry in entries
                if (reason := _refusal(entry, cache)) is not None
            ]
            cleared = [] if refusals else [item for item in entries if _clearable(item[2])]
            if cleared:
                with connection:
                    connection.executemany(
                        "DELETE FROM generation_results WHERE identity_sha256 = ?",
                        [(key,) for key, _, _ in cleared],
                    )
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()
        after = sha256_file(cache.path)
    record: dict[str, JsonValue] = {
        "artifact_type": CACHE_CLEARING_ARTIFACT_TYPE,
        "schema_version": "1.0",
        "protocol_id": TRIALGPT_RETRIEVAL_PROTOCOL_ID,
        "protocol_sha256": FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
        "checked_at": checked_at,
        "run_id": run_id,
        "stage": cache.stage,
        "system_id": cache.system_id,
        "cache": cache.path.name,
        "cache_sha256_before": before,
        "cache_sha256_after": after,
        "entries_checked": len(entries),
        "entries_removed": len(cleared),
        "removed": cast(JsonValue, [_entry_record(raw, entry) for _, raw, entry in cleared]),
        "refusal_check": {
            "rule": "protocol v11 Section 7.1 rule 4",
            "matching_entries": len(refusals),
            "entries": cast(JsonValue, refusals),
        },
        "status": "refused" if refusals else "cleared",
    }
    record_dir.mkdir(parents=True, exist_ok=True)
    with record_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    _append_interruption(
        interruptions,
        {
            "at": checked_at,
            "cause": INTERRUPTION_CAUSES[cause],
            "clearing_record": record_path.name,
            "clearing_record_sha256": sha256_file(record_path),
        },
    )
    if refusals:
        raise CacheClearingRefused(
            f"resume refused: {len(refusals)} unresolved entries hold provider output "
            f"(protocol v11 Section 7.1 rule 4); nothing was cleared. Record: {record_path}",
            record_path,
        )
    return {"record": str(record_path), **record}


__all__ = [
    "CACHE_CLEARING_ARTIFACT_TYPE",
    "INTERRUPTION_CAUSES",
    "CacheClearingRefused",
    "ResumeCache",
    "clear_failed_provider_calls",
    "resolve_generation_cache",
]
