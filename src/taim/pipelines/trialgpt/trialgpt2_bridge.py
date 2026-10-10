"""The model boundary of ``TrialGPT2-TAIM-Luna-v1``: a chat-completions endpoint answered by Codex.

TrialGPT 2.0's client speaks the OpenAI chat-completions wire. With ``LLM_PROVIDER=openai`` it
builds ``OpenAI(api_key=...)`` with no base URL, so the SDK reads ``OPENAI_BASE_URL`` and every
call reaches this loopback endpoint without a change to their code. Codex has no such endpoint;
TAIM reaches it as the ``codex exec`` subprocess that :mod:`taim.model_codex` runs. This bridge
joins the two, and its declared distortions are on the System
(:data:`taim.pipelines.trialgpt.trialgpt2_declaration.BRIDGE_DISTORTIONS`).

For each request it:

- refuses anything that is not their client's call for one of this run's candidate pairs: the
  label the run passed, a system and a user message, and their strict schema by digest;
- joins the two messages into the one prompt ``codex exec`` reads, the system message becoming a
  ``SYSTEM INSTRUCTIONS:`` preamble, and records the sampling controls it cannot apply;
- answers from the response cache when an earlier execution of the same run already received a
  schema-conforming answer to the identical request, and otherwise makes exactly one Codex
  execution, with their schema as ``--output-schema``. A failed execution is a 502, and their
  client's own retry policy decides what happens next;
- returns the answer with its token usage, and appends one ledger record per request.

It binds to loopback only and refuses any other client.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import threading
import time
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import cast

from taim.generation import (
    GenerationAttempt,
    GenerationCache,
    GenerationRequest,
    GenerationResult,
    canonical_json,
    request_identity,
)
from taim.model_codex import CodexProvider, schema_hash, sha256_text
from taim.pipelines.trialgpt.trialgpt2_declaration import (
    TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID,
    UPSTREAM_OUTPUT_SCHEMA_SHA256,
)

BRIDGE_VERSION = "trialgpt2-codex-bridge-v1"
SCORER_STAGE = "trialgpt2-scorer"
SYSTEM_PREAMBLE = "SYSTEM INSTRUCTIONS:"
SAMPLING_CONTROLS = ("temperature", "top_p", "seed", "max_tokens", "max_completion_tokens")
_REQUIRED_KEYS = frozenset({"model", "messages", "response_format"})
_PROVIDER_ID = "codex-cli-v1"

Pair = tuple[str, str]


def join_messages(system: str, user: str) -> str:
    """The one prompt ``codex exec`` reads for a system and a user message."""

    return f"{SYSTEM_PREAMBLE}\n{system}\n\n{user}"


def codex_provider(configuration: Mapping[str, object]) -> CodexProvider:
    """The Codex transport of a configured Codex ``Model``, for exactly one execution per call."""

    transport = configuration.get("transport")
    if configuration.get("provider_id") != _PROVIDER_ID or not isinstance(transport, Mapping):
        raise ValueError("the TrialGPT 2.0 bridge requires a Codex Model configuration")
    return CodexProvider(
        model=str(configuration["model"]),
        reasoning_effort=cast(str | None, configuration.get("reasoning_effort")),
        codex_executable=str(transport["codex_executable"]),
        request_timeout_seconds=cast(int, transport["request_timeout_seconds"]),
        max_request_bytes=cast(int, transport["max_request_bytes"]),
    )


def conforms_to_schema(text: str, schema: Mapping[str, object]) -> bool:
    """Whether ``text`` is a JSON value the schema admits, for the keywords their schema uses.

    A keyword outside that set never conforms, so an unexpected schema is never cached.
    """

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return _conforms(value, schema)


def _conforms(value: object, schema: Mapping[str, object]) -> bool:
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(value, dict) or not isinstance(properties, Mapping):
            return False
        if not isinstance(required, list) or any(name not in value for name in required):
            return False
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return False
        return all(
            isinstance(properties[name], Mapping)
            and _conforms(item, cast(Mapping[str, object], properties[name]))
            for name, item in value.items()
            if name in properties
        )
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        admitted = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        admitted = (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    else:
        return False
    if not admitted:
        return False
    number = float(cast(float, value))
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    return (not isinstance(minimum, int | float) or number >= minimum) and (
        not isinstance(maximum, int | float) or number <= maximum
    )


def _completion_usage(usage: Mapping[str, object]) -> dict[str, object]:
    """Codex's token usage on the chat-completion wire, which their pair logs read."""

    def count(name: str) -> int:
        value = usage.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    prompt_tokens = count("input_tokens")
    completion_tokens = count("output_tokens")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": count("cached_input_tokens")},
    }


def completion_body(text: str, *, label: str, usage: Mapping[str, object]) -> dict[str, object]:
    """An OpenAI ``chat.completion`` carrying ``text``, echoing the caller's model label."""

    return {
        "id": "chatcmpl-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": label,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": _completion_usage(usage),
    }


class TrialGPT2Bridge:
    """One run's chat-completions endpoint; a context manager that serves on a free port."""

    def __init__(
        self,
        *,
        model_configuration: Mapping[str, object],
        cache: GenerationCache,
        ledger: Path,
        expected_prompts: Mapping[str, Sequence[Pair]],
        label: str,
        invocation: int,
        output_schema_sha256: str = UPSTREAM_OUTPUT_SCHEMA_SHA256,
        provider: CodexProvider | None = None,
    ) -> None:
        self.model_configuration = dict(model_configuration)
        self.provider = provider or codex_provider(model_configuration)
        self.cache = cache
        self.ledger = ledger
        self.expected_prompts = {key: tuple(pairs) for key, pairs in expected_prompts.items()}
        self.label = label
        self.invocation = invocation
        self.output_schema_sha256 = output_schema_sha256
        self._lock = threading.Lock()
        self._counter = 0
        self._server: _BridgeServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("the TrialGPT 2.0 bridge is not serving")
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self) -> TrialGPT2Bridge:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self._server = _BridgeServer(self)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join()

    def _request_problem(self, body: object) -> str | None:
        """Why ``body`` is not their client's scorer call, or None."""

        if not isinstance(body, Mapping):
            return "request body is not a JSON object"
        unexpected = set(body) - _REQUIRED_KEYS - set(SAMPLING_CONTROLS)
        if unexpected or not set(body) >= _REQUIRED_KEYS:
            return "request fields are not their client's scorer call"
        if body["model"] != self.label:
            return f"model label {body['model']!r} is not the run's {self.label!r}"
        messages = body["messages"]
        if (
            not isinstance(messages, list)
            or [
                message.get("role") if isinstance(message, Mapping) else None
                for message in messages
            ]
            != ["system", "user"]
            or any(
                set(message) != {"role", "content"} or not isinstance(message["content"], str)
                for message in messages
            )
        ):
            return "messages are not one system and one user text message"
        response_format = body["response_format"]
        json_schema = (
            response_format.get("json_schema") if isinstance(response_format, Mapping) else None
        )
        if (
            not isinstance(response_format, Mapping)
            or response_format.get("type") != "json_schema"
            or not isinstance(json_schema, Mapping)
            or json_schema.get("strict") is not True
            or not isinstance(json_schema.get("schema"), Mapping)
        ):
            return "response_format is not a strict json_schema"
        if schema_hash(json_schema["schema"]) != self.output_schema_sha256:
            return "response_format schema is not TrialGPT 2.0's pinned schema"
        return None

    def answer(self, body: object) -> tuple[int, dict[str, object]]:
        """Answer one request; the HTTP status and the response body."""

        with self._lock:
            self._counter += 1
            call = self._counter
        record: dict[str, object] = {
            "call": call,
            "invocation": self.invocation,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bridge_version": BRIDGE_VERSION,
        }
        started = time.perf_counter()
        try:
            status, response = self._answer(body, record)
        finally:
            record["latency_seconds"] = round(time.perf_counter() - started, 3)
            self._append(record)
        return status, response

    def _answer(self, body: object, record: dict[str, object]) -> tuple[int, dict[str, object]]:
        problem = self._request_problem(body)
        if problem is not None:
            record.update({"outcome": "refused", "status": 400, "reason": problem})
            return 400, {"error": {"message": problem}}
        body = cast(Mapping[str, object], body)
        messages = cast(list[Mapping[str, str]], body["messages"])
        response_format = cast(Mapping[str, Mapping[str, object]], body["response_format"])
        schema = cast(Mapping[str, object], response_format["json_schema"]["schema"])
        prompt = join_messages(messages[0]["content"], messages[1]["content"])
        prompt_sha256 = sha256_text(prompt)
        record.update(
            {
                "requested_label": body["model"],
                "prompt_sha256": prompt_sha256,
                "output_schema_sha256": self.output_schema_sha256,
                "sampling_not_applied": {k: body[k] for k in SAMPLING_CONTROLS if k in body},
            }
        )
        pairs = self.expected_prompts.get(prompt_sha256)
        if pairs is None:
            reason = "request is not one of this run's candidate pairs"
            record.update({"outcome": "refused", "status": 400, "reason": reason})
            return 400, {"error": {"message": reason}}
        record["pairs"] = [list(pair) for pair in pairs]
        request = GenerationRequest(
            stage=SCORER_STAGE,
            prompt=prompt,
            output_schema=schema,
            input_payload={
                "system_id": TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID,
                "bridge_version": BRIDGE_VERSION,
            },
        )
        identity = request_identity(request, _PROVIDER_ID, self.model_configuration)
        record["request_identity"] = identity
        cached = self.cache.load(identity)
        if cached is not None and isinstance(cached.output, str):
            text = cached.output
            usage = cached.attempts[-1].usage
            record.update({"outcome": "cache_hit", "status": 200, "conforms_to_schema": True})
        else:
            attempt_started = time.perf_counter()
            try:
                response = self.provider.generate(
                    request, prompt=prompt, attempt=1, logical_call_id=identity
                )
            except Exception as exc:  # the transport is opaque; its class is recorded
                failure = getattr(exc, "failure_kind", "transport")
                record.update(
                    {
                        "outcome": "codex_failure",
                        "status": 502,
                        "failure_classification": (
                            failure if failure in ("transport", "authentication") else "transport"
                        ),
                        "detail_sha256": sha256_text(f"{type(exc).__name__}: {exc}"),
                    }
                )
                return 502, {"error": {"message": "codex execution failed"}}
            text = response.raw_output
            usage = response.usage
            codex = response.metadata.get("codex")
            codex = codex if isinstance(codex, Mapping) else {}
            provenance = codex.get("model_provenance")
            conforms = conforms_to_schema(text, schema)
            record.update(
                {
                    "outcome": "codex",
                    "status": 200,
                    "conforms_to_schema": conforms,
                    "codex_version": codex.get("codex_version"),
                    "reported_model": codex.get("reported_model"),
                    "model_provenance": (
                        provenance.get("status") if isinstance(provenance, Mapping) else None
                    ),
                }
            )
            if conforms:
                self.cache.store(
                    GenerationResult(
                        identity,
                        identity,
                        SCORER_STAGE,
                        _PROVIDER_ID,
                        "resolved",
                        text,
                        (
                            GenerationAttempt(
                                1,
                                "original",
                                prompt_sha256,
                                text,
                                text,
                                sha256_text(text),
                                sha256_text(canonical_json(text)),
                                (),
                                time.perf_counter() - attempt_started,
                                usage,
                                None,
                                metadata={
                                    **response.metadata,
                                    "provider_configuration": self.model_configuration,
                                },
                            ),
                        ),
                        1,
                    )
                )
                self.cache.flush()
        record.update({"response_sha256": sha256_text(text), "usage": dict(usage)})
        return 200, completion_body(text, label=self.label, usage=usage)

    def _append(self, record: Mapping[str, object]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, self.ledger.open("a", encoding="utf-8") as handle:
            handle.write(line)


LEDGER_OUTCOMES = frozenset({"codex", "codex_failure", "cache_hit", "refused"})


def read_ledger(path: Path) -> list[dict[str, object]]:
    """The bridge's ledger records, in the order they were written; absent means none."""

    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"bridge ledger line {number} is not an object")
        records.append(record)
    return records


def ledger_accounting(
    records: Sequence[Mapping[str, object]], expected_pairs: Collection[Pair]
) -> dict[str, object]:
    """Account for every request against the pairs a run's candidates produce."""

    expected = set(expected_pairs)
    outcomes = Counter(str(record.get("outcome")) for record in records)
    failures = Counter(
        str(record.get("failure_classification"))
        for record in records
        if record.get("outcome") == "codex_failure"
    )
    answered: Counter[Pair] = Counter()
    foreign = 0
    usage: Counter[str] = Counter()
    for record in records:
        raw_pairs = record.get("pairs", [])
        pairs = [
            (str(pair[0]), str(pair[1]))
            for pair in (raw_pairs if isinstance(raw_pairs, list) else [])
            if isinstance(pair, list) and len(pair) == 2
        ]
        if any(pair not in expected for pair in pairs):
            foreign += 1
        if record.get("status") == 200:
            answered.update(pairs)
        record_usage = record.get("usage")
        if record.get("outcome") == "codex" and isinstance(record_usage, Mapping):
            for name in ("input_tokens", "cached_input_tokens", "output_tokens"):
                value = record_usage.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    usage[name] += value
    return {
        "requests": len(records),
        "codex_executions": outcomes["codex"] + outcomes["codex_failure"],
        "codex_answers": outcomes["codex"],
        "codex_failures": dict(sorted(failures.items())),
        "cache_hits": outcomes["cache_hit"],
        "nonconforming_answers": sum(
            record.get("outcome") == "codex" and record.get("conforms_to_schema") is False
            for record in records
        ),
        "refused_requests": outcomes["refused"],
        "unknown_outcomes": sum(
            count for outcome, count in outcomes.items() if outcome not in LEDGER_OUTCOMES
        ),
        "foreign_requests": foreign,
        "pairs": len(expected),
        "unanswered_pairs": len(expected - set(answered)),
        "invocations": sorted(
            {
                cast(int, record["invocation"])
                for record in records
                if isinstance(record.get("invocation"), int)
            }
        ),
        "codex_versions": sorted(
            {str(record["codex_version"]) for record in records if record.get("codex_version")}
        ),
        "usage": dict(sorted(usage.items())),
    }


def accounting_defects(accounting: Mapping[str, object]) -> list[str]:
    """Why a run's requests are not all accounted for; empty when every call is."""

    defects = [
        f"{accounting[name]} {label}"
        for name, label in (
            ("refused_requests", "requests the bridge refused"),
            ("unknown_outcomes", "requests without a known outcome"),
            ("foreign_requests", "requests for pairs outside the run"),
            ("unanswered_pairs", "candidate pairs never answered"),
        )
        if accounting.get(name) != 0
    ]
    return defects


class _BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, bridge: TrialGPT2Bridge) -> None:
        super().__init__(("127.0.0.1", 0), _BridgeHandler)
        self.bridge = bridge


class _BridgeHandler(BaseHTTPRequestHandler):
    server: _BridgeServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _reply(self, status: int, body: Mapping[str, object]) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.client_address[0] not in {"127.0.0.1", "::1"}:
            self._reply(403, {"error": {"message": "loopback clients only"}})
            return
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._reply(404, {"error": {"message": f"unsupported path {path}"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body: object = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = None
        status, response = self.server.bridge.answer(body)
        self._reply(status, response)


__all__ = [
    "BRIDGE_VERSION",
    "LEDGER_OUTCOMES",
    "SAMPLING_CONTROLS",
    "SCORER_STAGE",
    "SYSTEM_PREAMBLE",
    "TrialGPT2Bridge",
    "accounting_defects",
    "codex_provider",
    "completion_body",
    "conforms_to_schema",
    "join_messages",
    "ledger_accounting",
    "read_ledger",
]
