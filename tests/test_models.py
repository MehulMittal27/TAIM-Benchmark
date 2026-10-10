# ruff: noqa: S101
"""Configured generation at the Model boundary, without provider calls."""

import subprocess

import pytest

from taim.models import Model


def test_model_configuration_is_inert_and_cannot_change(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("construction must not launch a process")

    import sqlite3
    import tempfile

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", forbidden)
    parameters = {"reasoning_effort": "medium"}
    model = Model(model_id="gpt-5.6-terra", backend="codex", parameters=parameters)
    parameters["reasoning_effort"] = "high"
    assert model.configuration["model"] == "gpt-5.6-terra"
    assert model.configuration["reasoning_effort"] == "medium"
    assert model.configuration["provider_id"] == "codex-cli-v1"
    with pytest.raises(TypeError):
        model.configuration["model"] = "other"
    with pytest.raises(TypeError):
        model.configuration["transport"]["codex_executable"] = "other"


@pytest.mark.parametrize(
    "parameters",
    [
        {"temperature": 0},
        {"api_key": "secret"},
        {"reasoning_effort": "bogus"},
        {"output_schema_transport": "lenient"},
    ],
)
def test_unsupported_parameters_fail_before_any_process(monkeypatch, parameters):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(ValueError):
        Model(model_id="test", backend="codex", parameters=parameters)


def test_repair_trace_and_optional_cache_bind_actual_configuration(monkeypatch, tmp_path):
    import json

    from taim.generation import GenerationCache, GenerationRequest

    calls = []
    outputs = iter(['{"score":"bad"}', '{"score":1}', '{"score":2}'])

    def runner(command, **kwargs):
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, "codex-test", "")
        calls.append((command, kwargs))
        event = {
            "type": "item.completed",
            "model": "gpt-5.6-terra",
            "item": {"type": "agent_message", "text": next(outputs)},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(event), "")

    monkeypatch.setattr(subprocess, "run", runner)
    request = GenerationRequest(
        "synthetic",
        "score",
        {"type": "object"},
        {"contract_version": "test-v1"},
        validator=lambda value: (
            () if isinstance(value.get("score"), int) else ("score must be integer",)
        ),
    )
    cache = GenerationCache(tmp_path / "responses.sqlite3")
    try:
        model = Model(model_id="gpt-5.6-terra", backend="codex", cache=cache)
        result = model.generate(request)
        assert result.output == {"score": 1}
        assert [a.metadata["attempt"] for a in result.attempts] == [1, 2]
        assert all(
            a.metadata["codex"]["logical_call_id"] == result.logical_call_id
            for a in result.attempts
        )
        assert all(a.metadata["codex"]["stage"] == "synthetic" for a in result.attempts)
        assert "score must be integer" in calls[1][1]["input"]
        assert model.generate(request).cache_hit
        assert len(calls) == 2
        changed = Model(
            model_id="gpt-5.6-terra",
            backend="codex",
            parameters={"reasoning_effort": "high"},
            cache=cache,
        )
        assert changed.generate(request).output == {"score": 2}
        assert len(calls) == 3
    finally:
        cache.close()


@pytest.mark.parametrize(
    "backend,model_id,parameters",
    [
        ("http", "test", {}),
        ("codex", "", {}),
        ("codex", "test", {"seed": 1}),
        ("codex", "test", {"max_tokens": 10}),
    ],
)
def test_invalid_configuration_refuses_inference(backend, model_id, parameters):
    with pytest.raises(ValueError):
        Model(model_id=model_id, backend=backend, parameters=parameters)


def test_uncached_model_works_and_missing_reported_model_stays_missing(monkeypatch):
    import json

    from taim.generation import GenerationRequest

    def runner(command, **kwargs):
        if command[1] == "--version":
            value = "codex-test"
        elif command[1] == "debug":
            value = json.dumps({"models": [{"slug": "gpt-5.6-terra"}]})
        else:
            value = json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}}
            )
        return subprocess.CompletedProcess(command, 0, value, "")

    monkeypatch.setattr(subprocess, "run", runner)
    model = Model(model_id="gpt-5.6-terra", backend="codex")
    result = model.generate(GenerationRequest("synthetic", "{}", {}, {"contract_version": "1"}))
    assert result.resolved and not result.cache_hit
    trace = result.attempts[0].metadata["codex"]
    assert trace["reported_model"] is None
    assert trace["model_provenance"]["status"] == "catalog_pinned_provider_revision_unavailable"
    assert "codex_version" not in model.configuration["transport"]
    assert result.attempts[0].metadata["provider_configuration"]["model"] == "gpt-5.6-terra"


@pytest.mark.parametrize("backend", ["http", "opencode", "codex"])
def test_system_prompt_is_common_and_binds_configuration(backend):
    options = {"endpoint": "https://synthetic.invalid/v1"} if backend == "http" else {}
    original = Model(model_id="provider/model", backend=backend, **options)
    configured = Model(
        model_id="provider/model", backend=backend, system_prompt="Nur Deutsch.", **options
    )
    assert original.configuration != configured.configuration
    assert configured._transport.system_prompt == "Nur Deutsch."
    with pytest.raises(ValueError, match="system_prompt"):
        Model(model_id="provider/model", backend=backend, system_prompt=12, **options)


@pytest.mark.parametrize("instructions", ['Deutsch. 🧪\n"Quoted"', "DEL: \x7f; control: \x00\t\r"])
def test_codex_delivers_system_instructions_without_rewriting_request(monkeypatch, instructions):
    import json

    from taim.generation import GenerationRequest

    calls = []

    def runner(command, **kwargs):
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, "codex-test", "")
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "type": "item.completed",
                    "model": "synthetic",
                    "item": {"type": "agent_message", "text": "{}"},
                }
            ),
            "",
        )

    monkeypatch.setattr(subprocess, "run", runner)
    request = GenerationRequest("synthetic", "Evidence only", {}, {"contract_version": "1"})
    model = Model(model_id="synthetic", backend="codex", system_prompt=instructions)
    result = model.generate(request)
    assert result.resolved
    command, kwargs = next(call for call in calls if call[0][1] == "exec")
    assert kwargs["input"] == request.prompt
    import tomllib

    option = next(value for value in command if value.startswith("developer_instructions="))
    assert tomllib.loads(option)["developer_instructions"] == model._transport.system_prompt
    assert request.prompt == "Evidence only"


def test_codex_system_instructions_count_towards_request_limit(monkeypatch):
    from taim.generation import GenerationRequest

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unexpected process"))
    model = Model(
        model_id="synthetic", backend="codex", system_prompt="ü" * 100, max_request_bytes=200
    )
    result = model.generate(GenerationRequest("synthetic", "evidence", {}, {}))
    assert not result.resolved
    assert result.attempts[0].metadata["failure_code"] == "model_request_too_large"


def test_free_text_uses_the_same_codex_path_and_provenance(monkeypatch):
    import json

    from taim.generation import GenerationRequest

    commands = []

    def runner(command, **kwargs):
        if command[1] == "--version":
            value = "codex-test"
        elif command[1] == "debug":
            value = json.dumps({"models": [{"slug": "gpt-5.6-luna"}]})
        else:
            commands.append((command, kwargs["input"]))
            value = json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "(assert (>= age 18))"},
                }
            )
        return subprocess.CompletedProcess(command, 0, value, "")

    monkeypatch.setattr(subprocess, "run", runner)
    model = Model(model_id="gpt-5.6-luna", backend="codex")
    request = GenerationRequest("chat", "age >= 18", None, {"contract_version": "free-text-v1"})
    result = model.generate(request)

    assert result.resolved and result.output == "(assert (>= age 18))"
    ((command, stdin),) = commands
    assert stdin == "age >= 18"
    assert "--output-schema" not in command
    assert command[command.index("--model") + 1] == "gpt-5.6-luna"
    trace = result.attempts[0].metadata["codex"]
    assert trace["output_schema_sha256"] is None
    assert trace["model_provenance"]["status"] == "catalog_pinned_provider_revision_unavailable"


@pytest.mark.parametrize("instructions", ["a" * 131072, "🧪" * 32768, "\x7f" * 22000])
def test_codex_rejects_oversized_argv_during_construction(monkeypatch, instructions):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(ValueError, match="CLI argument limit"):
        Model(model_id="synthetic", backend="codex", system_prompt=instructions)


def test_codex_schema_counts_towards_request_limit(monkeypatch):
    import json

    from taim.generation import GenerationRequest

    schema = {"type": "string", "description": "ü" * 100}
    prompt = "evidence"
    serialized = json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    size = len(prompt.encode()) + len(serialized.encode())
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unexpected process"))
    model = Model(model_id="synthetic", backend="codex", max_request_bytes=size - 1)
    result = model.generate(GenerationRequest("synthetic", prompt, schema, {}))
    assert not result.resolved
    assert result.attempts[0].metadata["failure_code"] == "model_request_too_large"


def test_codex_preflight_retains_request_cap(monkeypatch):
    import json

    from taim.model_codex import codex_preflight

    def runner(command, **kwargs):
        output = (
            "codex-test"
            if command[1] == "--version"
            else json.dumps({"models": [{"slug": "synthetic"}]})
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(subprocess, "run", runner)
    configurations = [
        codex_preflight(
            Model(model_id="synthetic", backend="codex", max_request_bytes=cap).configuration
        )
        for cap in (1234, 5678)
    ]
    assert [value["transport"]["max_request_bytes"] for value in configurations] == [1234, 5678]
    assert configurations[0] != configurations[1]


def test_codex_output_schema_transport_is_recorded_only_when_the_schema_stays_local():
    sent = Model(model_id="gpt-5.6-luna", backend="codex", parameters={"reasoning_effort": None})
    unsent = Model(
        model_id="gpt-5.6-luna",
        backend="codex",
        parameters={"reasoning_effort": None, "output_schema_transport": "none"},
    )
    default_transport = {
        "codex_executable": "codex",
        "max_request_bytes": 1_048_576,
        "request_timeout_seconds": 600,
    }
    # The default configuration, and every request identity it binds, is unchanged.
    assert sent.configuration == {
        "configuration_version": "model-v1",
        "provider_id": "codex-cli-v1",
        "model": "gpt-5.6-luna",
        "reasoning_effort": None,
        "transport": default_transport,
    }
    assert unsent.configuration["transport"] == {
        **default_transport,
        "output_schema_transport": "none",
    }
