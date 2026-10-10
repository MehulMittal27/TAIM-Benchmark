"""The query-plan receipt and recomputation, over a generation made through a Codex stand-in."""

# ruff: noqa: S101
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

import taim.pipelines.trialgpt.query_plan_records as records
from taim.artifacts import sha256_file
from taim.generation import GenerationCache, GenerationRequest
from taim.pipelines.trialgpt.query_plan_records import (
    generation_cache_path,
    load_query_topics,
    query_plan_models,
    recompute_query_plans,
    write_query_plan_receipt,
)
from taim.pipelines.trialgpt.retrieval_producer import (
    generate_three_luna_queries,
    require_query_plan_receipt,
    resolve_trialgpt_retrieval_target,
    write_generated_arm_queries,
)
from tests.trialgpt.query_plan_support import prepared_topics_manifest

_TRACK = "trec-ct-2021"
_PROFILE = "trec-ct-2021-external-fidelity-26149"
_RELEASE = {
    "release_manifest_id": "sha256:" + "1" * 64,
    "public_tree_id": "sha256:" + "2" * 64,
    "package_artifact_id": "sha256:" + "3" * 64,
}
_CODEX = """#!{python}
import hashlib, json, sys
version = {version!r}
if sys.argv[1:] == ["--version"]:
    print(version)
elif sys.argv[1:] == ["debug", "models", "--bundled"]:
    print(json.dumps({{"models": [{{"slug": "gpt-5.6-luna"}}]}}))
elif sys.argv[1] == "exec":
    prompt = sys.stdin.read()
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    answer = {{"summary": "Synthetic " + digest[:6], "conditions": [digest[:8], digest[8:16]]}}
    item = {{"type": "agent_message", "text": json.dumps(answer)}}
    print(json.dumps({{"type": "item.completed", "item": item}}))
else:
    sys.exit(64)
"""


def _codex(directory: Path, version: str) -> None:
    script = directory / "codex"
    script.write_text(_CODEX.format(python=sys.executable, version=version), encoding="utf-8")
    script.chmod(0o755)


@pytest.fixture(scope="module")
def _generated(tmp_path_factory) -> Path:
    """A complete TREC 2021 generation, made once through the stand-in like the release command."""

    root = tmp_path_factory.mktemp("query-generation")
    bin_directory = root / "bin"
    bin_directory.mkdir()
    _codex(bin_directory, "codex-cli 9.9.9")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("PATH", f"{bin_directory}:/usr/bin:/bin")
        records.resolve_executable.cache_clear()
        manifest = prepared_topics_manifest(root)
        topics, _ = load_query_topics(manifest, track=_TRACK, profile=_PROFILE)
        cache = GenerationCache(generation_cache_path(root / "query-generation.sqlite3"))
        try:
            queries = generate_three_luna_queries(
                topics,
                models=query_plan_models(cache=cache, codex_executable="codex"),
                max_workers=8,
            )
        finally:
            cache.close()
        write_generated_arm_queries(queries, root / "three-luna-query-plans.jsonl")
    records.resolve_executable.cache_clear()
    return root


@pytest.fixture
def query_generation(_generated, tmp_path, monkeypatch):
    """This test's own copy of the generation, its stand-in first on ``PATH``."""

    root = tmp_path / "generation"
    shutil.copytree(_generated, root)
    monkeypatch.setenv("PATH", f"{root / 'bin'}:/usr/bin:/bin")
    records.resolve_executable.cache_clear()
    yield {
        "manifest": root / "prepared-manifest.json",
        "cache": root / "query-generation.sqlite3",
        "plans": root / "three-luna-query-plans.jsonl",
        "bin": root / "bin",
    }
    records.resolve_executable.cache_clear()


def _receipt(tmp_path: Path, query_generation, **overrides) -> Path:
    output = tmp_path / "receipts" / "paper-trialgpt-trec21-retrieval-1.json"
    write_query_plan_receipt(
        **{
            "track": _TRACK,
            "profile": _PROFILE,
            "prepared_manifest": query_generation["manifest"],
            "run_id": "paper-trialgpt-trec21-retrieval-1",
            "generation_run_id": "paper-trialgpt-trec21-plans-1",
            "cache": query_generation["cache"],
            "query_plans": query_generation["plans"],
            "release_identity": _RELEASE,
            "interruptions": [],
            "output": output,
            **overrides,
        }
    )
    return output


def test_the_receipt_records_the_generation_and_the_freeze_accepts_it(
    tmp_path, query_generation
) -> None:
    path = _receipt(tmp_path, query_generation)
    target = resolve_trialgpt_retrieval_target(track=_TRACK, profile=_PROFILE)
    receipt, _ = require_query_plan_receipt(
        path,
        target=target,
        run_id="paper-trialgpt-trec21-retrieval-1",
        query_plans=query_generation["plans"],
    )
    assert len(receipt) == 14
    assert receipt["release"] == _RELEASE and receipt["logical_calls"] == 225
    block = receipt["generation"]
    cache_file = generation_cache_path(query_generation["cache"])
    assert block["generation_cache_sha256"] == sha256_file(cache_file)
    assert block["codex_versions"] == ["codex-cli 9.9.9"]
    assert block["codex_executable"] == "codex"
    assert block["codex_executable_path"] == str(query_generation["bin"] / "codex")
    assert block["resolved_logical_calls"] == block["provider_attempts"] == 225
    assert block["arms"]["xhigh"] == {
        "prompt_contract_id": "trialgpt-paper-keyword-v1",
        "reasoning_effort": "xhigh",
    }
    assert block["interruptions"] == []
    with pytest.raises(ValueError, match="written once"):
        _receipt(tmp_path, query_generation)


def test_a_receipt_names_its_interruptions_in_the_protocol_shape(
    tmp_path, query_generation
) -> None:
    with pytest.raises(ValueError, match="'at' and 'cause'"):
        _receipt(tmp_path, query_generation, interruptions=[{"cause": "usage allowance exhausted"}])
    interruption = {"at": "2026-10-08T03:00:00Z", "cause": "usage allowance exhausted"}
    path = _receipt(tmp_path, query_generation, interruptions=[interruption])
    assert json.loads(path.read_text())["generation"]["interruptions"] == [interruption]


def test_plans_from_another_cache_get_no_receipt(tmp_path, query_generation) -> None:
    other = tmp_path / "other.jsonl"
    rows = query_generation["plans"].read_text().splitlines()
    row = json.loads(rows[0])
    row["generation_trace"]["logical_call_id"] = "sha256:" + "0" * 64
    other.write_text("\n".join([json.dumps(row), *rows[1:]]) + "\n")
    with pytest.raises(ValueError, match="no resolved call in this generation cache"):
        _receipt(tmp_path, query_generation, query_plans=other)


def test_the_recomputation_rebuilds_the_plans_byte_for_byte_without_a_call(
    tmp_path, query_generation
) -> None:
    receipt = _receipt(tmp_path, query_generation)
    cache_file = generation_cache_path(query_generation["cache"])
    before = sha256_file(cache_file)
    record, path = recompute_query_plans(
        track=_TRACK,
        profile=_PROFILE,
        prepared_manifest=query_generation["manifest"],
        cache=query_generation["cache"],
        query_plans=query_generation["plans"],
        receipt_path=receipt,
        workers=8,
    )
    assert path == receipt.with_name("paper-trialgpt-trec21-retrieval-1.recomputation.json")
    assert record["status"] == "byte-identical", record["failure"]
    assert record["recomputed_query_plans_sha256"] == sha256_file(query_generation["plans"])
    assert (record["provider_calls_made"], record["provider_calls_refused"]) == (0, 0)
    assert record["codex_version"] == "codex-cli 9.9.9"
    assert sha256_file(cache_file) == before
    with pytest.raises(ValueError, match="written once"):
        recompute_query_plans(
            track=_TRACK,
            profile=_PROFILE,
            prepared_manifest=query_generation["manifest"],
            cache=query_generation["cache"],
            query_plans=query_generation["plans"],
            receipt_path=receipt,
            workers=8,
        )


def test_the_recomputation_fails_on_another_codex_version_or_cache(
    tmp_path, query_generation
) -> None:
    receipt = _receipt(tmp_path, query_generation)
    _codex(query_generation["bin"], "codex-cli 9.9.10")
    record, _ = recompute_query_plans(
        track=_TRACK,
        profile=_PROFILE,
        prepared_manifest=query_generation["manifest"],
        cache=query_generation["cache"],
        query_plans=query_generation["plans"],
        receipt_path=receipt,
        workers=8,
        output=tmp_path / "version.recomputation.json",
    )
    assert record["status"] == "failed"
    assert record["failure"] == "the installed Codex CLI is not the version the generation ran"

    _codex(query_generation["bin"], "codex-cli 9.9.9")
    cache_file = generation_cache_path(query_generation["cache"])
    connection = sqlite3.connect(cache_file)
    connection.execute("DELETE FROM generation_results WHERE rowid = 1")
    connection.commit()
    connection.close()
    record, _ = recompute_query_plans(
        track=_TRACK,
        profile=_PROFILE,
        prepared_manifest=query_generation["manifest"],
        cache=query_generation["cache"],
        query_plans=query_generation["plans"],
        receipt_path=receipt,
        workers=8,
        output=tmp_path / "cache.recomputation.json",
    )
    assert record["status"] == "failed"
    assert record["failure"] == "the generation cache is not the one the receipt records"


def test_a_recomputing_model_refuses_every_call_its_cache_cannot_answer(tmp_path) -> None:
    cache = GenerationCache(tmp_path / "empty.sqlite3")
    refusing = records._RefusingTransport()
    model = query_plan_models(cache=None, codex_executable="codex")["medium"]
    replay = records._RecomputingModel(model, cache=cache, transport=refusing)
    assert replay.configuration == model.configuration
    result = replay.generate(
        GenerationRequest(
            stage="keyword_generation",
            prompt="Patient",
            output_schema={"type": "object"},
            input_payload={"digest": hashlib.sha256(b"x").hexdigest()},
        )
    )
    cache.close()
    assert not result.resolved
    assert refusing.refused == 3


def test_the_receipt_and_recomputation_refuse_a_package_without_complete_eligibility(
    tmp_path, query_generation
) -> None:
    # Protocol v15: a target's receipt and recomputation bind its complete-eligibility package.
    receipt = _receipt(tmp_path, query_generation)
    manifest = json.loads(query_generation["manifest"].read_text())
    manifest["snapshot"]["available_capabilities"] = []
    query_generation["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete-eligibility package"):
        _receipt(tmp_path / "again", query_generation)
    with pytest.raises(ValueError, match="complete-eligibility package"):
        recompute_query_plans(
            track=_TRACK,
            profile=_PROFILE,
            prepared_manifest=query_generation["manifest"],
            cache=query_generation["cache"],
            query_plans=query_generation["plans"],
            receipt_path=receipt,
            workers=8,
        )
