"""Clearing cached failed provider calls before a resume (protocol v11 Section 7.1).

A Codex stand-in signs out part-way through a real query-plan generation; the resume that follows
the clearing completes it, and the receipt and the recomputation accept the result. The rules that
refuse a resume are exercised on caches written by ``run_generation`` itself.
"""

# ruff: noqa: S101
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import taim.pipelines.trialgpt.query_plan_records as records
from taim.artifacts import sha256_file
from taim.generation import (
    GenerationCache,
    GenerationCacheInUseError,
    GenerationCacheRunLease,
    GenerationRequest,
    ProviderResponse,
    run_generation,
)
from taim.model_codex import CodexTransportError
from taim.pipelines.trialgpt.cache_clearing import (
    CacheClearingRefused,
    clear_failed_provider_calls,
    resolve_generation_cache,
)
from taim.pipelines.trialgpt.query_plan_records import (
    generation_cache_path,
    load_query_topics,
    query_plan_models,
    recompute_query_plans,
    write_query_plan_receipt,
)
from taim.pipelines.trialgpt.retrieval_producer import (
    TrialGPTRetrievalProducerError,
    generate_three_luna_queries,
    write_generated_arm_queries,
)
from taim.pipelines.trialgpt.system import trialgpt_run_workspace
from tests.trialgpt.query_plan_support import prepared_topics_manifest

_TRACK = "trec-ct-2021"
_PROFILE = "trec-ct-2021-external-fidelity-26149"
_SCRIPT = Path(__file__).parents[2] / "scripts" / "clear_trialgpt_failed_calls.py"
#: Codex signs out once this many generation calls have been made.
_SIGN_OUT_AFTER = 60
_CODEX = """#!{python}
import hashlib, json, os, sys
from pathlib import Path
state = Path({state!r})
if sys.argv[1:] == ["--version"]:
    print("codex-cli 9.9.9")
elif sys.argv[1:] == ["debug", "models", "--bundled"]:
    print(json.dumps({{"models": [{{"slug": "gpt-5.6-luna"}}]}}))
elif sys.argv[1] == "exec":
    prompt = sys.stdin.read()
    with open(state / "calls", "a") as calls:
        calls.write(".")
    if (state / "calls").stat().st_size > {sign_out_after} and not (state / "signed-in").exists():
        print("Error: Not logged in. Run codex login.", file=sys.stderr)
        sys.exit(1)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    answer = {{"summary": "Synthetic " + digest[:6], "conditions": [digest[:8], digest[8:16]]}}
    item = {{"type": "agent_message", "text": json.dumps(answer)}}
    print(json.dumps({{"type": "item.completed", "item": item}}))
else:
    sys.exit(64)
"""


@pytest.fixture
def signed_out_generation(tmp_path, monkeypatch):
    """A TREC 2021 generation that Codex signed out of part-way through."""

    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    codex = bin_directory / "codex"
    codex.write_text(
        _CODEX.format(python=sys.executable, state=str(tmp_path), sign_out_after=_SIGN_OUT_AFTER),
        encoding="utf-8",
    )
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_directory}:/usr/bin:/bin")
    records.resolve_executable.cache_clear()
    manifest = prepared_topics_manifest(tmp_path)
    topics, _ = load_query_topics(manifest, track=_TRACK, profile=_PROFILE)
    base = tmp_path / "query-generation.sqlite3"

    def generate() -> None:
        cache = GenerationCache(generation_cache_path(base))
        try:
            with GenerationCacheRunLease(cache.path, run_id="paper-trialgpt-trec21-plans-1"):
                queries = generate_three_luna_queries(
                    topics,
                    models=query_plan_models(cache=cache, codex_executable="codex"),
                    max_workers=8,
                )
        finally:
            cache.close()
        write_generated_arm_queries(queries, tmp_path / "three-luna-query-plans.jsonl")

    with pytest.raises(TrialGPTRetrievalProducerError, match="unresolved"):
        generate()
    yield {"root": tmp_path, "cache": base, "manifest": manifest, "generate": generate}
    records.resolve_executable.cache_clear()


def _entries(path: Path) -> dict[str, dict]:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT identity_sha256, result_json FROM generation_results")
        return {key: json.loads(raw) for key, raw in rows}
    finally:
        connection.close()


def _failure_classes(entry: dict) -> list[str | None]:
    return [attempt["metadata"].get("failure_classification") for attempt in entry["attempts"]]


def _clear(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    # The command as the runbook gives it, from a directory of its own.
    workdir = root / "cwd"
    workdir.mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *arguments],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_sign_out_is_cleared_and_the_resumed_generation_completes(
    signed_out_generation,
) -> None:
    root = signed_out_generation["root"]
    cache_file = generation_cache_path(signed_out_generation["cache"])
    stored = _entries(cache_file)
    signed_out = {
        key for key, entry in stored.items() if "authentication" in _failure_classes(entry)
    }
    assert signed_out and all(stored[key]["status"] == "unresolved" for key in signed_out)
    (root / "signed-in").touch()

    # Signed in again, a plain resume still returns each cached sign-in failure unchanged: it
    # asks only for the calls the stopped generation never made, and fails again.
    with pytest.raises(TrialGPTRetrievalProducerError, match="unresolved"):
        signed_out_generation["generate"]()
    entries = _entries(cache_file)
    assert {key: entries[key] for key in signed_out} == {key: stored[key] for key in signed_out}
    assert signed_out == {
        key for key, entry in entries.items() if "authentication" in _failure_classes(entry)
    }
    resolved = {key for key, entry in entries.items() if entry["status"] == "resolved"}
    resolved_bytes = {key: json.dumps(entries[key], sort_keys=True) for key in resolved}

    completed = _clear(
        root,
        "--stage", "query",
        "--run-id", "paper-trialgpt-trec21-plans-1",
        "--cache", str(signed_out_generation["cache"]),
        "--cause", "signed-out",
        "--record-dir", str(root / "receipts"),
        "--interruptions", str(root / "interruptions.json"),
    )  # fmt: skip
    assert completed.returncode == 0, completed.stderr
    printed = json.loads(completed.stdout)
    assert printed["status"] == "cleared"
    assert printed["entries_removed"] == len(signed_out)
    assert printed["refusal_check"]["matching_entries"] == 0
    assert printed["cache_sha256_after"] == sha256_file(cache_file)
    record_path = Path(printed["record"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert {item["identity_sha256"] for item in record["removed"]} == signed_out
    for item in record["removed"]:
        assert item["entry"] == entries[item["identity_sha256"]]
        assert set(item["attempt_failure_classes"]) <= {"authentication", "transport"}
        assert item["attempt_count"] == len(item["entry"]["attempts"])
    after = _entries(cache_file)
    assert set(after) == set(entries) - signed_out
    assert {key: json.dumps(after[key], sort_keys=True) for key in resolved} == resolved_bytes

    # Signed in again, the resume asks again only for what was cleared.
    signed_out_generation["generate"]()
    interruptions = json.loads((root / "interruptions.json").read_text(encoding="utf-8"))
    assert interruptions == [
        {
            "at": record["checked_at"],
            "cause": "expired or signed-out session",
            "clearing_record": record_path.name,
            "clearing_record_sha256": sha256_file(record_path),
        }
    ]
    receipt = root / "receipts" / "paper-trialgpt-trec21-retrieval-1.json"
    written = write_query_plan_receipt(
        track=_TRACK,
        profile=_PROFILE,
        prepared_manifest=signed_out_generation["manifest"],
        run_id="paper-trialgpt-trec21-retrieval-1",
        generation_run_id="paper-trialgpt-trec21-plans-1",
        cache=signed_out_generation["cache"],
        query_plans=root / "three-luna-query-plans.jsonl",
        release_identity={
            "release_manifest_id": "sha256:" + "1" * 64,
            "public_tree_id": "sha256:" + "2" * 64,
            "package_artifact_id": "sha256:" + "3" * 64,
        },
        interruptions=interruptions,
        output=receipt,
    )
    assert written["generation"]["interruptions"] == interruptions
    assert written["generation"]["resolved_logical_calls"] == 225
    recomputation, _ = recompute_query_plans(
        track=_TRACK,
        profile=_PROFILE,
        prepared_manifest=signed_out_generation["manifest"],
        cache=signed_out_generation["cache"],
        query_plans=root / "three-luna-query-plans.jsonl",
        receipt_path=receipt,
        workers=8,
    )
    assert recomputation["status"] == "byte-identical"


class _Provider:
    """A provider whose next answers are scripted per prompt: text, or a failure class."""

    provider_id = "scripted"

    def __init__(self, script: dict[str, tuple[str, ...]]) -> None:
        self.script = {name: list(answers) for name, answers in script.items()}

    def generate(self, request, *, prompt, attempt, logical_call_id) -> ProviderResponse:
        del attempt, logical_call_id
        answer = self.script[request.input_payload["name"]].pop(0)
        if answer in ("authentication", "transport"):
            raise CodexTransportError(f"scripted {answer} failure", failure_kind=answer)
        return ProviderResponse(answer)


def _cache(path: Path, stage: str, script: dict[str, tuple[str, ...]], *, reuse: bool) -> None:
    cache = GenerationCache(path)
    provider = _Provider(script)
    try:
        for name in script:
            run_generation(
                GenerationRequest(
                    stage=stage,
                    prompt=f"prompt {name}",
                    output_schema={"type": "object"},
                    input_payload={"name": name},
                ),
                provider,
                cache=cache,
                reuse_unresolved=reuse,
            )
    finally:
        cache.close()


def _check(cache, tmp_path: Path, run_id: str):
    return clear_failed_provider_calls(
        cache,
        run_id=run_id,
        cause="signed-out",
        record_dir=tmp_path / "records",
        interruptions=tmp_path / "interruptions.json",
        now=datetime(2026, 10, 8, 12, 0, tzinfo=UTC),
    )


_SIGNED_OUT = ("authentication",)
_TRANSPORT_EXHAUSTED = ("transport", "transport", "transport")
_INVALID_THRICE = ("not json", "not json", "not json")
_OUTPUT_THEN_SIGNED_OUT = ("not json", "authentication")


def test_the_query_stage_refuses_output_it_would_ask_again_for(tmp_path) -> None:
    base = tmp_path / "query-generation.sqlite3"
    _cache(
        generation_cache_path(base),
        "keyword_generation",
        {"ok": ('{"a": 1}',), "signed-out": _SIGNED_OUT, "invalid": _INVALID_THRICE},
        reuse=False,
    )
    cache = resolve_generation_cache(
        stage="query", run_id="plans-1", cache=base, system_id=None, workspace=None
    )
    before = sha256_file(cache.path)
    with pytest.raises(CacheClearingRefused, match="rule 4") as refused:
        _check(cache, tmp_path, "plans-1")
    record = json.loads(refused.value.record.read_text(encoding="utf-8"))
    assert record["status"] == "refused" and record["entries_removed"] == 0
    assert [item["rule"][:4] for item in record["refusal_check"]["entries"]] == ["4(b)"]
    assert sha256_file(cache.path) == before == record["cache_sha256_after"]
    assert len(_entries(cache.path)) == 3
    interruptions = json.loads((tmp_path / "interruptions.json").read_text(encoding="utf-8"))
    assert interruptions[0]["clearing_record"] == refused.value.record.name


def test_trialgpt_keeps_its_abstentions_and_clears_only_failed_calls(tmp_path) -> None:
    workspace = tmp_path / "trialgpt-workspace"
    path = trialgpt_run_workspace(workspace, "row-1") / "model-generation-v1-cache.sqlite3"
    path.parent.mkdir(parents=True)
    _cache(
        path,
        "criterion_matching_inclusion",
        {
            "ok": ('{"a": 1}',),
            "signed-out": _SIGNED_OUT,
            "transport": _TRANSPORT_EXHAUSTED,
            "abstained": _INVALID_THRICE,
        },
        reuse=True,
    )
    cache = resolve_generation_cache(
        stage="system",
        run_id="row-1",
        cache=None,
        system_id="TrialGPT-TAIM-Luna-v1",
        workspace=workspace,
    )
    record = _check(cache, tmp_path, "row-1")
    assert record["status"] == "cleared" and record["entries_removed"] == 2
    removed = sorted(tuple(item["attempt_failure_classes"]) for item in record["removed"])
    assert removed == [("authentication",), ("transport", "transport", "transport")]
    statuses = sorted(entry["status"] for entry in _entries(path).values())
    assert statuses == ["resolved", "unresolved"]


def test_a_failed_call_beside_provider_output_refuses_the_resume(tmp_path) -> None:
    workspace = tmp_path / "trialgpt-workspace"
    path = trialgpt_run_workspace(workspace, "row-1") / "model-generation-v1-cache.sqlite3"
    path.parent.mkdir(parents=True)
    _cache(
        path,
        "final_aggregation",
        {"signed-out": _SIGNED_OUT, "mixed": _OUTPUT_THEN_SIGNED_OUT},
        reuse=True,
    )
    cache = resolve_generation_cache(
        stage="system",
        run_id="row-1",
        cache=None,
        system_id="TrialGPT-TAIM-Luna-v1",
        workspace=workspace,
    )
    with pytest.raises(CacheClearingRefused) as refused:
        _check(cache, tmp_path, "row-1")
    record = json.loads(refused.value.record.read_text(encoding="utf-8"))
    assert [item["rule"][:4] for item in record["refusal_check"]["entries"]] == ["4(a)"]
    assert len(_entries(path)) == 2


def test_a_check_refuses_a_held_cache_a_foreign_stage_and_an_unlisted_cause(tmp_path) -> None:
    base = tmp_path / "query-generation.sqlite3"
    _cache(generation_cache_path(base), "keyword_generation", {"ok": ('{"a": 1}',)}, reuse=False)
    query = resolve_generation_cache(
        stage="query", run_id="plans-1", cache=base, system_id=None, workspace=None
    )
    with (
        GenerationCacheRunLease(query.path, run_id="plans-1"),
        pytest.raises(GenerationCacheInUseError),
    ):
        _check(query, tmp_path, "plans-1")
    with pytest.raises(ValueError, match="cause must be one of"):
        clear_failed_provider_calls(
            query,
            run_id="plans-1",
            cause="the Pro account ran out",
            record_dir=tmp_path / "records",
            interruptions=tmp_path / "interruptions.json",
        )
    workspace = tmp_path / "trialgpt-workspace"
    path = trialgpt_run_workspace(workspace, "row-1") / "model-generation-v1-cache.sqlite3"
    path.parent.mkdir(parents=True)
    generation_cache_path(base).replace(path)
    row = resolve_generation_cache(
        stage="system",
        run_id="row-1",
        cache=None,
        system_id="TrialGPT-TAIM-Luna-v1",
        workspace=workspace,
    )
    with pytest.raises(ValueError, match="not the system stage's cache"):
        _check(row, tmp_path, "row-1")
    assert not (tmp_path / "records").exists()
