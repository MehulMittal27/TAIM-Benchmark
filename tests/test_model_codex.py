# ruff: noqa: S101
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from taim.model_codex import (
    CodexProvider,
    CodexProviderError,
    CodexRequest,
    CodexTransportError,
    build_codex_command,
    schema_hash,
)

SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def _request() -> CodexRequest:
    return CodexRequest(
        stage="keyword_generation",
        prompt="Return the requested JSON object.",
        output_schema=SCHEMA,
        model="gpt-5.6",
        reasoning_effort="medium",
        input_hash="sha256:" + "a" * 64,
        logical_call_id="topic-1-keywords",
    )


def test_command_isolated_and_does_not_enable_tools(tmp_path: Path) -> None:
    command = build_codex_command(
        _request(),
        schema_path=tmp_path / "schema.json",
        workdir=tmp_path,
    )

    assert command[:3] == ["codex", "exec", "--ephemeral"]
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "-a" not in command
    assert "--output-schema" in command
    assert "--search" not in command
    assert 'web_search="disabled"' in command
    assert "--add-dir" not in command
    assert "workspace-write" not in command
    assert "danger-full-access" not in command
    assert command[command.index("--model") + 1] == "gpt-5.6"


def test_schema_hash_is_order_independent() -> None:
    assert schema_hash({"b": 2, "a": 1}) == schema_hash({"a": 1, "b": 2})


def test_default_model_omits_model_override(tmp_path: Path) -> None:
    request = CodexRequest(
        stage="smoke_test",
        prompt="Return JSON.",
        output_schema=SCHEMA,
        model=None,
    )
    command = build_codex_command(
        request,
        schema_path=tmp_path / "schema.json",
        workdir=tmp_path,
    )

    assert "--model" not in command


def test_generation_preserves_events_usage_identity_and_isolation(monkeypatch) -> None:
    monkeypatch.setenv("TAIM_SECRET", "must-not-leak")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.1.0\n", stderr="")
        schema_path = Path(command[command.index("--output-schema") + 1])
        assert schema_path.exists()
        assert json.loads(schema_path.read_text(encoding="utf-8")) == SCHEMA
        assert kwargs["cwd"] == schema_path.parent
        assert "TAIM_SECRET" not in kwargs["env"]
        stream = "\n".join(
            [
                json.dumps({"type": "turn.started", "model": "gpt-5.6-2026-07-01"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": '{"summary":"ok"}'},
                    }
                ),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4}}),
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=stream, stderr="")

    result = CodexProvider(runner=runner).generate_request(_request())

    assert result.text == '{"summary":"ok"}'
    assert result.requested_model == "gpt-5.6"
    assert result.reported_model == "gpt-5.6-2026-07-01"
    assert result.model_identity_matches is True
    assert result.codex_version == "codex-cli 0.1.0"
    assert result.usage == {"input_tokens": 4}
    assert result.attempts == 1
    assert result.prompt_sha256 == _request().prompt_sha256
    assert result.output_schema_sha256 == _request().output_schema_sha256
    assert result.request_identity == _request().identity
    assert result.logical_call_id == "topic-1-keywords"
    assert result.input_hash == "sha256:" + "a" * 64
    assert len(calls) == 2


def test_generation_bounds_codex_subprocess_runtime() -> None:
    observed_timeouts: list[object] = []

    def runner(command, **kwargs):
        observed_timeouts.append(kwargs.get("timeout"))
        stream = "\n".join(
            [
                json.dumps({"type": "turn.started", "model": "gpt-5.6-2026-07-01"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": '{"summary":"ok"}'},
                    }
                ),
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=stream, stderr="")

    CodexProvider(version="codex-cli 0.146.0", runner=runner).generate_request(_request())

    assert observed_timeouts == [600]


def test_generation_timeout_is_a_retryable_transport_failure() -> None:
    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(CodexTransportError, match="timed out after 600 seconds") as raised:
        CodexProvider(version="codex-cli 0.146.0", runner=runner).generate_request(_request())

    assert raised.value.failure_kind == "transport"


def test_nonzero_exit_preserves_transport_failure() -> None:
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="network down")

    with pytest.raises(CodexTransportError, match="exit 7") as raised:
        CodexProvider(runner=runner).generate_request(_request())

    assert raised.value.failure_kind == "transport"


def test_nonzero_authentication_exit_is_safely_classified() -> None:
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="authentication required")

    with pytest.raises(CodexTransportError) as raised:
        CodexProvider(runner=runner).generate_request(_request())

    assert raised.value.failure_kind == "authentication"


def test_missing_final_event_is_not_a_negative_result() -> None:
    def runner(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.1.0", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"type": "turn.completed", "usage": {"output_tokens": 0}}),
            stderr="",
        )

    with pytest.raises(CodexProviderError, match="no final output event"):
        CodexProvider(runner=runner).generate_request(_request())


def test_model_identity_mismatch_fails_closed() -> None:
    def runner(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.1.0", stderr="")
        stream = "\n".join(
            [
                json.dumps({"type": "turn.started", "model": "gpt-5.5"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": '{"summary":"ok"}'},
                    }
                ),
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=stream, stderr="")

    with pytest.raises(CodexProviderError, match="does not match requested model"):
        CodexProvider(runner=runner).generate_request(_request())


def test_requested_model_without_reported_identity_fails_closed() -> None:
    def runner(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.146.0", stderr="")
        stream = "\n".join(
            [
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": '{"summary":"ok"}'},
                    }
                ),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4}}),
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=stream, stderr="")

    with pytest.raises(CodexProviderError, match="reported model identity"):
        CodexProvider(runner=runner).generate_request(_request())


def test_bundled_catalog_pins_model_when_exec_events_omit_identity() -> None:
    catalog_entry = {
        "slug": "gpt-5.6",
        "display_name": "GPT-5.6",
        "default_reasoning_level": "medium",
        "supported_in_api": True,
    }

    def runner(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.146.0", stderr="")
        if command[1:] == ["debug", "models", "--bundled"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"models": [catalog_entry]}),
                stderr="",
            )
        stream = "\n".join(
            [
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": '{"summary":"ok"}'},
                    }
                ),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4}}),
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=stream, stderr="")

    result = CodexProvider(runner=runner).generate_request(_request())

    assert result.reported_model is None
    assert result.model_identity_matches is None
    assert result.model_provenance == {
        "status": "catalog_pinned_provider_revision_unavailable",
        "requested_model": "gpt-5.6",
        "provider_reported_model": None,
        "effective_model": "gpt-5.6",
        "resolution_source": "codex_cli_bundled_catalog",
        "catalog_entry_sha256": schema_hash(catalog_entry),
        "codex_version": "codex-cli 0.146.0",
        "provider_revision_available": False,
        "identity_sha256": schema_hash(
            {
                "catalog_entry_sha256": schema_hash(catalog_entry),
                "codex_version": "codex-cli 0.146.0",
                "effective_model": "gpt-5.6",
                "provider_reported_model": None,
                "requested_model": "gpt-5.6",
                "resolution_source": "codex_cli_bundled_catalog",
            }
        ),
    }


def test_free_text_request_passes_no_schema_to_the_cli(tmp_path: Path) -> None:
    from dataclasses import replace

    request = replace(_request(), output_schema=None)
    command = build_codex_command(request, schema_path=None, workdir=tmp_path)
    structured = build_codex_command(
        _request(), schema_path=tmp_path / "schema.json", workdir=tmp_path
    )

    assert "--output-schema" not in command
    schema_flag = structured.index("--output-schema")
    assert command == structured[:schema_flag] + structured[schema_flag + 2 :]
    assert request.output_schema_sha256 is None
    assert request.identity != _request().identity


def test_free_text_generation_writes_no_schema_file_and_keeps_the_text() -> None:
    from dataclasses import replace

    workdirs: list[Path] = []

    def runner(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.1.0\n", stderr="")
        workdirs.append(Path(kwargs["cwd"]))
        assert "--output-schema" not in command
        assert list(Path(kwargs["cwd"]).iterdir()) == []
        stream = "\n".join(
            [
                json.dumps({"type": "turn.started", "model": "gpt-5.6"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "plain words, not JSON"},
                    }
                ),
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=stream, stderr="")

    result = CodexProvider(runner=runner).generate_request(replace(_request(), output_schema=None))

    assert result.text == "plain words, not JSON"
    assert result.output_schema_sha256 is None
    assert len(workdirs) == 1


def test_unsent_schema_stays_on_the_request_and_off_the_cli() -> None:
    import hashlib
    from dataclasses import replace

    from taim.model_codex import codex_preflight

    def runner(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.1.0\n", stderr="")
        assert "--output-schema" not in command
        assert list(Path(kwargs["cwd"]).iterdir()) == []
        event = {
            "type": "item.completed",
            "model": "gpt-5.6",
            "item": {"type": "agent_message", "text": '{"summary": "ok"}'},
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(event), stderr="")

    strict = _request()
    unsent = replace(strict, output_schema_transport="none")
    result = CodexProvider(runner=runner, output_schema_transport="none").generate_request(unsent)

    assert result.text == '{"summary": "ok"}'
    assert result.output_schema_sha256 == schema_hash(SCHEMA)
    assert result.as_dict()["output_schema_transport"] == "none"
    # A strict request's identity is the one it had before the option existed.
    payload = {
        "stage": strict.stage,
        "prompt_sha256": strict.prompt_sha256,
        "output_schema_sha256": strict.output_schema_sha256,
        "model": strict.model,
        "reasoning_effort": strict.reasoning_effort,
        "input_hash": strict.input_hash,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert strict.identity == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    assert unsent.identity != strict.identity
    with pytest.raises(ValueError, match="output_schema_transport"):
        replace(strict, output_schema_transport="lenient")
    with pytest.raises(ValueError, match="output_schema_transport"):
        CodexProvider(output_schema_transport="lenient")
    configuration = {
        "provider_id": "codex-cli-v1",
        "model": "gpt-5.6",
        "transport": {
            "codex_executable": "codex",
            "request_timeout_seconds": 600,
            "max_request_bytes": 1_048_576,
            "output_schema_transport": "none",
        },
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CodexProvider, "codex_version", lambda self: "codex-cli 0.1.0")
        patch.setattr(CodexProvider, "_catalog_model_provenance", lambda self, model: {})
        preflight = codex_preflight(configuration)
    assert preflight["transport"]["output_schema_transport"] == "none"
