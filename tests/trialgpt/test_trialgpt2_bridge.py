"""The TrialGPT 2.0 Codex bridge: what it answers, refuses, caches and records."""

# ruff: noqa: S101, S310
from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from taim.generation import GenerationCache
from taim.model_codex import CodexProvider, schema_hash, sha256_text
from taim.models import Model
from taim.pipelines.trialgpt.trialgpt2_bridge import (
    TrialGPT2Bridge,
    conforms_to_schema,
    join_messages,
    ledger_accounting,
    read_ledger,
)
from taim.pipelines.trialgpt.trialgpt2_declaration import UPSTREAM_OUTPUT_SCHEMA_SHA256
from tests.trialgpt.trialgpt2_support import UPSTREAM_SCHEMA, answer_for, codex_events

LABEL = "gpt-5.6-luna"
PAIR = ("1", "NCT00000001")


def _body(user: str = "Patient and trial", **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": LABEL,
        "messages": [
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "trial_match_result",
                "strict": True,
                "schema": UPSTREAM_SCHEMA,
            },
        },
        "temperature": 0.0,
    }
    body.update(overrides)
    return body


class _Codex:
    """A Codex stand-in that answers in process and counts executions."""

    def __init__(self) -> None:
        self.executions = 0
        self.answer: str | None = None
        self.fail = False

    def __call__(self, command, *args, input=None, **kwargs):
        del args, kwargs
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, "codex-cli test\n", "")
        if command[1:] == ["debug", "models", "--bundled"]:
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"models": [{"slug": LABEL}]}), ""
            )
        self.executions += 1
        if self.fail:
            return subprocess.CompletedProcess(command, 1, "", "not logged in")
        text = self.answer if self.answer is not None else answer_for(input)
        return subprocess.CompletedProcess(command, 0, codex_events(text), "")


@pytest.fixture
def codex() -> _Codex:
    return _Codex()


@pytest.fixture
def codex_bridge(tmp_path: Path, codex: _Codex):
    model = Model(model_id=LABEL, backend="codex", parameters={"reasoning_effort": None})
    configuration = json.loads(json.dumps(dict(model.configuration), default=dict))
    cache = GenerationCache(tmp_path / "cache.sqlite3")
    prompt = join_messages("Policy", "Patient and trial")
    codex_bridge = TrialGPT2Bridge(
        model_configuration=configuration,
        cache=cache,
        ledger=tmp_path / "ledger.jsonl",
        expected_prompts={sha256_text(prompt): [PAIR]},
        label=LABEL,
        invocation=1,
        provider=CodexProvider(model=LABEL, runner=codex),
    )
    yield codex_bridge
    cache.close()


def test_the_pinned_schema_digest_is_their_schema() -> None:
    assert schema_hash(UPSTREAM_SCHEMA) == UPSTREAM_OUTPUT_SCHEMA_SHA256


def test_an_answer_is_one_execution_and_a_repeat_is_served_from_the_cache(
    codex_bridge, codex
) -> None:
    status, body = codex_bridge.answer(_body())
    assert status == 200
    content = body["choices"][0]["message"]["content"]
    assert conforms_to_schema(content, UPSTREAM_SCHEMA)
    assert body["usage"] == {
        "prompt_tokens": 1200,
        "completion_tokens": 90,
        "total_tokens": 1290,
        "prompt_tokens_details": {"cached_tokens": 400},
    }
    again_status, again = codex_bridge.answer(_body())
    assert (again_status, again["choices"]) == (200, body["choices"])
    assert codex.executions == 1
    first, second = read_ledger(codex_bridge.ledger)
    assert (first["outcome"], second["outcome"]) == ("codex", "cache_hit")
    assert first["pairs"] == [list(PAIR)] and first["sampling_not_applied"] == {"temperature": 0.0}
    assert first["prompt_sha256"] == sha256_text(join_messages("Policy", "Patient and trial"))
    assert first["codex_version"] == "codex-cli test"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (_body(model="gpt-4.1"), "model label"),
        (_body(user="Another patient"), "not one of this run's candidate pairs"),
        (_body(stream=True), "fields"),
        (_body(messages=[{"role": "user", "content": "Patient and trial"}]), "one system"),
        (
            _body(response_format={"type": "json_object"}),
            "strict json_schema",
        ),
        (
            _body(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "x", "strict": True, "schema": {"type": "object"}},
                }
            ),
            "pinned schema",
        ),
        ("not an object", "not a JSON object"),
    ],
)
def test_anything_but_their_call_for_a_run_pair_is_refused(
    codex_bridge, codex, body, reason
) -> None:
    status, response = codex_bridge.answer(body)
    assert status == 400 and reason in response["error"]["message"]
    assert codex.executions == 0
    (record,) = read_ledger(codex_bridge.ledger)
    assert record["outcome"] == "refused"


def test_a_failed_execution_is_a_502_and_their_client_decides_the_retry(
    codex_bridge, codex
) -> None:
    codex.fail = True
    status, _ = codex_bridge.answer(_body())
    assert status == 502
    (record,) = read_ledger(codex_bridge.ledger)
    assert (record["outcome"], record["failure_classification"]) == (
        "codex_failure",
        "authentication",
    )
    codex.fail = False
    assert codex_bridge.answer(_body())[0] == 200
    assert codex.executions == 2


def test_an_answer_outside_the_schema_is_returned_but_never_cached(codex_bridge, codex) -> None:
    codex.answer = json.dumps({"score": 101})
    status, body = codex_bridge.answer(_body())
    assert status == 200 and body["choices"][0]["message"]["content"] == codex.answer
    codex.answer = None
    codex_bridge.answer(_body())
    assert codex.executions == 2
    records = read_ledger(codex_bridge.ledger)
    assert [record["conforms_to_schema"] for record in records] == [False, True]
    accounting = ledger_accounting(records, [PAIR])
    assert accounting["nonconforming_answers"] == 1 and accounting["unanswered_pairs"] == 0


def test_the_endpoint_serves_loopback_chat_completions(codex_bridge, codex) -> None:
    with codex_bridge:
        request = urllib.request.Request(
            codex_bridge.base_url + "/chat/completions",
            data=json.dumps(_body()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            assert json.loads(response.read())["object"] == "chat.completion"
        missing = urllib.request.Request(codex_bridge.base_url + "/embeddings", data=b"{}")
        with pytest.raises(urllib.error.HTTPError, match="404"):
            urllib.request.urlopen(missing)
    assert codex.executions == 1


@pytest.mark.parametrize(
    ("text", "conforms"),
    [
        (answer_for("x"), True),
        ('{"score": 5}', False),
        ("not json", False),
    ],
)
def test_schema_conformance_covers_their_schema(text, conforms) -> None:
    assert conforms_to_schema(text, UPSTREAM_SCHEMA) is conforms


def test_conformance_refuses_out_of_range_and_extra_fields() -> None:
    answer = json.loads(answer_for("x"))
    assert not conforms_to_schema(json.dumps({**answer, "score": 101}), UPSTREAM_SCHEMA)
    assert not conforms_to_schema(json.dumps({**answer, "confidence": 1.5}), UPSTREAM_SCHEMA)
    assert not conforms_to_schema(json.dumps({**answer, "score": True}), UPSTREAM_SCHEMA)
    assert not conforms_to_schema(json.dumps({**answer, "extra": "x"}), UPSTREAM_SCHEMA)
    assert conforms_to_schema(json.dumps({**answer, "score": 100}), UPSTREAM_SCHEMA)
