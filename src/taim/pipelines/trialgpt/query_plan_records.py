"""The two records a query-plan generation leaves: its receipt and its recomputation.

Protocol v11 Section 6 requires a receipt, written and committed before the freeze, recording the
generation that made a target's query plans. Section 8 requires, after the freeze and before any
effectiveness value is read, that the plans be recomputed from the retained generation cache by
the query method itself, with a provider that refuses every call, rebuilding a byte-identical
file; the result is committed beside the receipt as ``<run-id>.recomputation.json``.

Both read the generation cache without changing it: the receipt reads it read-only, and the
recomputation runs against a copy. The recomputation's Models carry the generation's exact
configuration, Codex executable included, so every logical-call identity is the generation's; only
their transport is replaced, by one that refuses. No Codex process is started for a call.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from taim.artifacts import sha256_file
from taim.executables import ExecutableNotFoundError, resolve_executable
from taim.generation import (
    GenerationCache,
    GenerationRequest,
    GenerationResult,
    ProviderResponse,
    run_generation,
)
from taim.model_codex import CodexTransportError
from taim.models import Model
from taim.pipelines.trialgpt.retrieval_producer import (
    FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
    QUERY_PLAN_RECEIPT_ARTIFACT_TYPE,
    TRIALGPT_RETRIEVAL_PROTOCOL_ID,
    TRIALGPT_RETRIEVAL_SOURCE_ARMS,
    RetrievalQueryTopic,
    TrialGPTRetrievalProducerError,
    generate_three_luna_queries,
    load_generated_arm_queries,
    require_query_plan_receipt,
    require_query_plan_receipt_identity,
    require_trialgpt_capabilities,
    resolve_trialgpt_retrieval_target,
    write_generated_arm_queries,
)
from taim.pipelines.trialgpt.system import TRIALGPT_CODEX_MODEL
from taim.schemas import JsonValue, json_value_to_builtins
from taim.snapshot import BenchmarkTopic

QUERY_PLAN_RECOMPUTATION_ARTIFACT_TYPE = "taim-trialgpt-query-plan-recomputation"
_ARM_EFFORTS = {"medium": "medium", "xhigh": "xhigh", "recall-explicit": "medium"}


def generation_cache_path(cache: Path) -> Path:
    """The cache file a generation writes for the ``--cache`` it was given."""

    return cache.with_name(cache.name + ".model-v1")


def query_plan_models(*, cache: GenerationCache | None, codex_executable: str) -> dict[str, Model]:
    """The three query arms' Models, as the generation configures them."""

    return {
        arm_id: Model(
            backend="codex",
            cache=cache,
            model_id=TRIALGPT_CODEX_MODEL,
            parameters={"reasoning_effort": _ARM_EFFORTS[arm_id]},
            codex_executable=codex_executable,
        )
        for arm_id in TRIALGPT_RETRIEVAL_SOURCE_ARMS
    }


def load_query_topics(
    manifest_path: Path, *, track: str, profile: str
) -> tuple[tuple[RetrievalQueryTopic, ...], str]:
    """A target's complete topic set and Prepared Snapshot ID, verified against its manifest.

    The package must be the target's complete-eligibility package (protocol v15): generation,
    receipt and recomputation all bind its Prepared Snapshot.
    """

    target = resolve_trialgpt_retrieval_target(track=track, profile=profile)
    require_trialgpt_capabilities(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("prepared manifest must be a JSON object")
    snapshot = manifest.get("snapshot")
    if (
        manifest.get("dataset_id") != target.prepared_dataset_id
        or not isinstance(snapshot, dict)
        or snapshot.get("benchmark_lineage") != target.track
        or snapshot.get("snapshot_name") != target.snapshot_name
    ):
        raise ValueError("prepared manifest does not match the admitted Track/Profile target")
    snapshot_id = snapshot.get("snapshot_id")
    outputs = manifest.get("outputs")
    topics_output = outputs.get("topics") if isinstance(outputs, dict) else None
    if (
        not isinstance(snapshot_id, str)
        or not isinstance(topics_output, dict)
        or not isinstance(topics_output.get("filename"), str)
        or not isinstance(topics_output.get("sha256"), str)
    ):
        raise ValueError("prepared manifest has no single-file topics output contract")
    path = manifest_path.parent / topics_output["filename"]
    if sha256_file(path) != topics_output["sha256"]:
        raise ValueError("prepared topics file does not match its manifest")
    topics: list[RetrievalQueryTopic] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                topic = BenchmarkTopic.from_json(line)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid prepared topic at line {line_number}") from exc
            topics.append(RetrievalQueryTopic(topic.topic_id, topic.canonical_text))
    if len(topics) != topics_output.get("record_count"):
        raise ValueError("prepared topics record count does not match its manifest")
    if len(topics) != target.topic_count:
        raise ValueError("prepared topics do not contain the frozen complete topic set")
    return tuple(topics), snapshot_id


def _closed_cache(cache: Path) -> Path:
    """The cache file, refused while a process may still hold it or its log is unmerged."""

    if not cache.is_file():
        raise ValueError(f"generation cache does not exist: {cache}")
    pending = cache.with_name(cache.name + "-wal")
    if pending.exists() and pending.stat().st_size:
        raise ValueError(
            f"generation cache {cache} has an unmerged write-ahead log; let the generation "
            "finish or exit before it is recorded"
        )
    return cache


def _cache_entries(cache: Path) -> list[GenerationResult]:
    # The file is closed and unchanging: read it without a lock, a log or a write.
    connection = sqlite3.connect(f"{cache.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        rows = connection.execute("SELECT result_json FROM generation_results").fetchall()
    finally:
        connection.close()
    return [GenerationResult.from_dict(json.loads(row[0])) for row in rows]


def _generation_block(
    entries: Sequence[GenerationResult],
    plans: Path,
    *,
    cache_sha256: str,
    generation_run_id: str,
    interruptions: Sequence[Mapping[str, JsonValue]],
) -> dict[str, JsonValue]:
    queries = load_generated_arm_queries(plans)
    resolved = {entry.logical_call_id for entry in entries if entry.resolved}
    missing = [
        f"{query.topic_id}/{query.arm_id}"
        for query in queries
        if query.generation_trace.get("logical_call_id") not in resolved
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} query plans have no resolved call in this generation cache, "
            f"for example {missing[0]}"
        )
    versions: set[str] = set()
    executables: set[str] = set()
    for entry in entries:
        for attempt in entry.attempts:
            configuration = attempt.metadata.get("provider_configuration")
            transport = (
                configuration.get("transport") if isinstance(configuration, Mapping) else None
            )
            if isinstance(transport, Mapping) and isinstance(
                transport.get("codex_executable"), str
            ):
                executables.add(cast(str, transport["codex_executable"]))
            codex = attempt.metadata.get("codex")
            if isinstance(codex, Mapping) and isinstance(codex.get("codex_version"), str):
                versions.add(cast(str, codex["codex_version"]))
    if len(executables) != 1:
        raise ValueError(
            "a generation runs one Codex executable throughout (protocol v11 Section 7); "
            f"this cache records {sorted(executables)}"
        )
    (executable,) = executables
    try:
        executable_path = executable if "/" in executable else resolve_executable(executable)
    except ExecutableNotFoundError:
        executable_path = None
    arms = {
        query.arm_id: {
            "prompt_contract_id": query.prompt_contract_id,
            "reasoning_effort": query.provider_configuration.get("reasoning_effort"),
        }
        for query in queries
    }
    return {
        "generation_run_id": generation_run_id,
        "model": TRIALGPT_CODEX_MODEL,
        "arms": cast(JsonValue, arms),
        "codex_versions": cast(JsonValue, sorted(versions)),
        "codex_executable": executable,
        "codex_executable_path": executable_path,
        "generation_cache_sha256": cache_sha256,
        "cache_entries": len(entries),
        "resolved_logical_calls": len(resolved),
        "provider_attempts": sum(len(entry.attempts) for entry in entries),
        "interruptions": cast(JsonValue, [dict(item) for item in interruptions]),
    }


def _interruptions(value: object) -> list[Mapping[str, JsonValue]]:
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("at"), str)
        or not isinstance(item.get("cause"), str)
        for item in value
    ):
        raise ValueError("interruptions must be a list of objects, each with 'at' and 'cause'")
    return cast(list[Mapping[str, JsonValue]], value)


def write_query_plan_receipt(
    *,
    track: str,
    profile: str,
    prepared_manifest: Path,
    run_id: str,
    generation_run_id: str,
    cache: Path,
    query_plans: Path,
    release_identity: Mapping[str, str],
    interruptions: object,
    output: Path,
) -> dict[str, JsonValue]:
    """Write the receipt for the freeze ``run_id`` from the generation's cache and plans."""

    if output.exists():
        raise ValueError(f"a receipt is written once: {output} exists")
    target = resolve_trialgpt_retrieval_target(track=track, profile=profile)
    topics, snapshot_id = load_query_topics(prepared_manifest, track=track, profile=profile)
    cache_file = _closed_cache(generation_cache_path(cache))
    cache_sha256 = sha256_file(cache_file)
    entries = _cache_entries(cache_file)
    if sha256_file(cache_file) != cache_sha256:
        raise ValueError("the generation cache changed while it was read")
    logical_calls = len(TRIALGPT_RETRIEVAL_SOURCE_ARMS) * target.topic_count
    receipt: dict[str, JsonValue] = {
        "artifact_type": QUERY_PLAN_RECEIPT_ARTIFACT_TYPE,
        "schema_version": "1.0",
        "protocol_id": TRIALGPT_RETRIEVAL_PROTOCOL_ID,
        "protocol_sha256": FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
        "system_id": "TrialGPT-TAIM-Luna-v1",
        "run_id": run_id,
        "track": target.track,
        "profile": target.profile,
        "prepared_snapshot_id": snapshot_id,
        "release": cast(JsonValue, dict(release_identity)),
        "query_plans_sha256": sha256_file(query_plans),
        "logical_calls": logical_calls,
        "generation": _generation_block(
            entries,
            query_plans,
            cache_sha256=cache_sha256,
            generation_run_id=generation_run_id,
            interruptions=_interruptions(interruptions),
        ),
        "generation_ordinal": 1,
    }
    if len(load_generated_arm_queries(query_plans)) != logical_calls or len(topics) * 3 != (
        logical_calls
    ):
        raise ValueError("the query plans do not cover the target's complete topic set")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # The freeze's own checks, so a receipt it would refuse is never left to be committed.
    try:
        require_query_plan_receipt(output, target=target, run_id=run_id, query_plans=query_plans)
        require_query_plan_receipt_identity(
            receipt, release_identity=release_identity, prepared_snapshot_id=snapshot_id
        )
    except TrialGPTRetrievalProducerError:
        output.unlink()
        raise
    return receipt


class _RefusingTransport:
    """The provider of a recomputation: it refuses every call."""

    provider_id = "codex-cli-v1"

    def __init__(self) -> None:
        self.refused = 0

    def generate(
        self, request: GenerationRequest, *, prompt: str, attempt: int, logical_call_id: str
    ) -> ProviderResponse:
        del request, prompt, attempt, logical_call_id
        self.refused += 1
        raise CodexTransportError("provider call refused during recomputation")


class _RecomputingModel(Model):
    """A query arm's exact Model, answering from the generation's cache or not at all."""

    def __init__(
        self, model: Model, *, cache: GenerationCache, transport: _RefusingTransport
    ) -> None:
        configuration = model.configuration
        parameters = {"reasoning_effort": configuration.get("reasoning_effort")}
        transport_configuration = cast(Mapping[str, object], configuration["transport"])
        super().__init__(
            model_id=str(configuration["model"]),
            backend="codex",
            parameters=cast(Mapping[str, JsonValue], parameters),
            codex_executable=str(transport_configuration["codex_executable"]),
        )
        if self.configuration != configuration:
            raise AssertionError("a recomputing Model must carry the generation's configuration")
        self._replay_cache = cache
        self._refusing = transport

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return run_generation(
            request,
            self._refusing,
            provider_config=cast(Mapping[str, object], json_value_to_builtins(self.configuration)),
            cache=self._replay_cache,
        )


def _codex_version(executable: str) -> str | None:
    """``codex --version``, which reads the CLI's own version and makes no provider call."""

    try:
        path = executable if "/" in executable else resolve_executable(executable)
        completed = subprocess.run(  # noqa: S603 - resolved executable and a fixed flag
            [path, "--version"], capture_output=True, text=True, check=False, timeout=60
        )
    except (OSError, ExecutableNotFoundError, subprocess.TimeoutExpired):
        return None
    return (
        (completed.stdout.strip() or completed.stderr.strip() or None)
        if (completed.returncode == 0)
        else None
    )


def recompute_query_plans(
    *,
    track: str,
    profile: str,
    prepared_manifest: Path,
    cache: Path,
    query_plans: Path,
    receipt_path: Path,
    workers: int,
    output: Path | None = None,
) -> tuple[dict[str, JsonValue], Path]:
    """Recompute the plan file from the retained cache; the record and where it was written."""

    raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    run_id = raw_receipt.get("run_id") if isinstance(raw_receipt, Mapping) else None
    if not isinstance(run_id, str):
        raise ValueError("query-plan receipt has no run ID")
    target = resolve_trialgpt_retrieval_target(track=track, profile=profile)
    receipt, receipt_sha256 = require_query_plan_receipt(
        receipt_path, target=target, run_id=run_id, query_plans=query_plans
    )
    destination = output or receipt_path.with_name(f"{run_id}.recomputation.json")
    if destination.exists():
        raise ValueError(f"a recomputation is written once: {destination} exists")
    generation = cast(Mapping[str, object], receipt["generation"])
    codex_executable = generation.get("codex_executable")
    if not isinstance(codex_executable, str):
        raise ValueError("the receipt does not record the generation's Codex executable")
    topics, snapshot_id = load_query_topics(prepared_manifest, track=track, profile=profile)
    if snapshot_id != receipt["prepared_snapshot_id"]:
        raise ValueError("the prepared manifest is not the receipt's Prepared Snapshot")
    cache_file = _closed_cache(generation_cache_path(cache))
    cache_sha256 = sha256_file(cache_file)
    record: dict[str, JsonValue] = {
        "artifact_type": QUERY_PLAN_RECOMPUTATION_ARTIFACT_TYPE,
        "schema_version": "1.0",
        "protocol_id": TRIALGPT_RETRIEVAL_PROTOCOL_ID,
        "protocol_sha256": FROZEN_TRIALGPT_RETRIEVAL_PROTOCOL_SHA256,
        "run_id": run_id,
        "track": target.track,
        "profile": target.profile,
        "query_plan_receipt_sha256": receipt_sha256,
        "query_plans_sha256": receipt["query_plans_sha256"],
        "generation_cache_sha256": cache_sha256,
        "codex_executable": codex_executable,
        "codex_version": _codex_version(codex_executable),
        "provider_calls_made": 0,
    }
    failure: str | None = None
    if cache_sha256 != generation.get("generation_cache_sha256"):
        failure = "the generation cache is not the one the receipt records"
    versions = generation.get("codex_versions")
    if failure is None and (
        not isinstance(versions, list) or record["codex_version"] not in versions
    ):
        failure = "the installed Codex CLI is not the version the generation ran"
    refusing = _RefusingTransport()
    recomputed: str | None = None
    if failure is None:
        with tempfile.TemporaryDirectory(prefix="taim-recomputation-") as temporary:
            copy = Path(temporary) / cache_file.name
            shutil.copyfile(cache_file, copy)
            replay = GenerationCache(copy)
            try:
                models = {
                    arm_id: _RecomputingModel(model, cache=replay, transport=refusing)
                    for arm_id, model in query_plan_models(
                        cache=None, codex_executable=codex_executable
                    ).items()
                }
                queries = generate_three_luna_queries(topics, models=models, max_workers=workers)
                rebuilt = Path(temporary) / query_plans.name
                write_generated_arm_queries(queries, rebuilt)
                recomputed = sha256_file(rebuilt)
            except TrialGPTRetrievalProducerError as exc:
                failure = f"the query method did not resolve from the cache: {exc}"
            finally:
                replay.close()
        if sha256_file(cache_file) != cache_sha256:
            failure = "the generation cache changed during the recomputation"
    if failure is None and recomputed != receipt["query_plans_sha256"]:
        failure = "the recomputed plan file is not byte-identical to the receipt's"
    record.update(
        {
            "logical_calls": len(TRIALGPT_RETRIEVAL_SOURCE_ARMS) * len(topics),
            "provider_calls_refused": refusing.refused,
            "recomputed_query_plans_sha256": recomputed,
            "status": "byte-identical" if failure is None else "failed",
            "failure": failure,
            "recomputed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record, destination


__all__ = [
    "QUERY_PLAN_RECOMPUTATION_ARTIFACT_TYPE",
    "generation_cache_path",
    "load_query_topics",
    "query_plan_models",
    "recompute_query_plans",
    "write_query_plan_receipt",
]
