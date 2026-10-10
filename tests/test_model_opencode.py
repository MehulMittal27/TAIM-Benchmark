# ruff: noqa: S101
"""Actual executable fixtures for OpenCode's documented stdin and JSON event protocol."""

import json
import os
import sys
import time
from contextlib import closing
from dataclasses import replace

import pytest

from taim.generation import GenerationCache, GenerationRequest
from taim.models import Model

pytestmark = pytest.mark.skipif(os.name != "posix", reason="OpenCode process groups require POSIX")


def _script(tmp_path, body):
    executable = tmp_path / "opencode-fixture"
    executable.write_text(
        f"#!{sys.executable}\nimport sys, os, json, time\n"
        "if sys.argv[1:] == ['--version']:\n print('1.18.32'); sys.exit(0)\n" + body
    )
    executable.chmod(0o700)
    return str(executable)


def _event(kind, **part):
    return {
        "type": kind,
        "timestamp": 1,
        "sessionID": "synthetic-session",
        "part": {"messageID": "synthetic-message", **part},
    }


def _completed(text='{"score":1}', *, reason="stop"):
    return [
        _event("step_start"),
        _event("text", id="text-1", text=text),
        _event("step_finish", reason=reason),
    ]


def _output_script(events):
    return "\n".join(f"print({json.dumps(json.dumps(e))})" for e in events)


def _request():
    return GenerationRequest(
        "synthetic",
        "Use only supplied synthetic evidence.",
        {"type": "object"},
        {"contract_version": "fixture-v1"},
        validator=lambda value: () if value == {"score": 1} else ("expected score 1",),
    )


def _model(executable, **kwargs):
    return Model(
        model_id="provider/test-model", backend="opencode", opencode_executable=executable, **kwargs
    )


@pytest.mark.parametrize("credential", ["SYNTHETIC_API_KEY", "OPENCODE_API_KEY"])
def test_cli_flags_stdin_environment_repair_cache_and_provenance(tmp_path, monkeypatch, credential):
    monkeypatch.setenv("INHERITED_PRIVATE_API_KEY", "must-not-reach-child")
    capture = tmp_path / "capture.jsonl"
    body = f"""
record = {{'args':sys.argv[1:], 'prompt':sys.stdin.read(), 'home':os.environ['HOME']}}
assert os.environ[{credential!r}] == 'private-token'
assert 'INHERITED_PRIVATE_API_KEY' not in os.environ
config = json.loads(os.environ['OPENCODE_CONFIG_CONTENT'])
assert config['permission'] == {{'*':'deny'}}
assert config['share'] == 'disabled'
assert config['agent']['taim-generation']['permission'] == {{'*':'deny'}}
with open({str(capture)!r}, 'a') as f: f.write(json.dumps(record)+'\\n')
text = '{{"score":1}}' if 'expected score 1' in record['prompt'] else '{{"score":0}}'
for e in {_completed()!r}:
 if e['type'] == 'text': e['part']['text'] = text
 print(json.dumps(e))
"""
    executable = _script(tmp_path, body)
    with closing(GenerationCache(tmp_path / "responses.sqlite3")) as cache:
        configured = _model(
            executable,
            parameters={"variant": "high"},
            cache=cache,
            opencode_environment=lambda: {credential: "private-token"},
        )
        result = configured.generate(_request())
        assert result.resolved and len(result.attempts) == 2
        assert configured.generate(_request()).cache_hit
    records = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(records) == 2
    args = records[0]["args"]
    assert args[:5] == ["run", "--format", "json", "--model", "provider/test-model"]
    assert args[-2:] == ["--variant", "high"]
    assert "--pure" in args and "--auto" not in args and "--output-schema" not in args
    assert _request().prompt not in args
    assert '"type": "object"' in records[0]["prompt"]
    assert all(not os.path.exists(record["home"]) for record in records)
    trace = result.attempts[-1].metadata["opencode"]
    assert trace["requested_model"] == "provider/test-model"
    assert trace["reported_model"] is None
    assert trace["opencode_version"] == "1.18.32"
    assert trace["logical_call_id"] == result.logical_call_id
    assert "private-token" not in json.dumps(result.to_dict())
    assert b"private-token" not in (tmp_path / "responses.sqlite3").read_bytes()


@pytest.mark.parametrize(
    "events",
    [
        _completed(reason="length"),
        _completed('{"score":1,"score":1}'),
        _completed("[]"),
        _completed()[:-1],
        [_event("tool_use", tool="bash")],
        [_event("error", error={"message": "private-token"})],
        _completed() + _completed(),
        [*_completed()[:2], {**_completed()[-1], "sessionID": "different"}],
    ],
)
def test_rejects_incomplete_malformed_actions_and_mixed_sessions(tmp_path, events):
    result = _model(_script(tmp_path, _output_script(events))).generate(_request())
    assert not result.resolved and result.output is None
    assert len(result.attempts) == 3
    assert all(a.metadata["failure_code"] == "invalid_model_response" for a in result.attempts)
    assert "private-token" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("body", ["print('x'*10000)", "sys.stderr.write('x'*10000)"])
def test_bounds_stdout_and_stderr(tmp_path, body):
    result = _model(_script(tmp_path, body), max_response_bytes=100).generate(_request())
    assert not result.resolved
    assert result.attempts[-1].metadata["failure_code"] == "model_response_too_large"


def test_input_ceiling_prevents_executable_start(tmp_path):
    marker = tmp_path / "started"
    result = _model(
        _script(tmp_path, f"open({str(marker)!r},'w').close()"), max_request_bytes=1
    ).generate(_request())
    assert not marker.exists()
    assert result.attempts[-1].metadata["failure_code"] == "model_request_too_large"


def test_expired_zero_deadline_never_starts_executable(tmp_path):
    # A missing executable must not be consulted once the caller deadline has expired.
    result = _model(str(tmp_path / "absent"), deadline=lambda: 0).generate(_request())
    assert not result.resolved
    assert all(a.metadata["failure_code"] == "model_timeout" for a in result.attempts)


def test_total_deadline_kills_process_group(tmp_path):
    pid_file = tmp_path / "child-pid"
    executable = _script(
        tmp_path,
        f"""
import subprocess
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
with open({str(pid_file)!r}, 'w') as f: f.write(str(child.pid))
time.sleep(30)
""",
    )
    started = time.monotonic()
    result = _model(executable, deadline=lambda: started + 2).generate(_request())
    assert time.monotonic() - started < 5
    assert not result.resolved
    assert result.attempts[-1].metadata["failure_code"] == "model_timeout"
    pid = int(pid_file.read_text())
    # A killed orphan can briefly remain a zombie until init reaps it.
    import subprocess

    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    assert not state or state.startswith("Z")


def test_exit_and_missing_executable_errors_are_sanitized(tmp_path):
    for executable in [
        _script(tmp_path, "sys.stderr.write('private-token'); sys.exit(1)"),
        str(tmp_path / "absent"),
    ]:
        result = _model(executable).generate(_request())
        assert not result.resolved and len(result.attempts) == 3
        assert "private-token" not in json.dumps(result.to_dict())


@pytest.mark.parametrize(
    "environment",
    [
        {"HOME": "/unsafe"},
        {"OPENCODE_CONFIG_CONTENT": "{}"},
        {"OPENCODE_CONFIG": "/unsafe/config.json"},
        {"OPENCODE_CONFIG_DIR": "/unsafe"},
        {"OPENCODE_CONFIG_API_KEY": "private-token"},
        {"OPENCODE_AUTH_TOKEN": "private-token"},
        {"XDG_CONFIG_HOME": "/unsafe"},
        {"XDG_API_KEY": "private-token"},
        {"OPENCODE_API_KEY": 123},
        {"OPENCODE_API_KEY": "private-token", "OPENCODE_CONFIG_CONTENT": "{}"},
    ],
)
def test_runtime_environment_cannot_override_isolation(tmp_path, environment):
    result = _model(
        _script(tmp_path, _output_script(_completed())),
        opencode_environment=lambda: environment,
    ).generate(_request())
    assert not result.resolved
    assert result.attempts[0].metadata["failure_code"] == "invalid_runtime_environment"


def test_credential_loader_errors_do_not_leave_credential_dependent_hashes(tmp_path):
    errors = []
    for detail in ("first-private-token", "second-private-token"):

        def fail(detail=detail):
            raise RuntimeError(detail)

        result = _model(str(tmp_path / "absent"), opencode_environment=fail).generate(_request())
        assert not result.resolved and len(result.attempts) == 1
        assert result.attempts[0].metadata["failure_code"] == "model_unavailable"
        errors.append(result.attempts[0].transport_error)
    assert errors[0] == errors[1]


@pytest.mark.parametrize(
    "options",
    [
        {"model_id": "unqualified"},
        {"parameters": {"temperature": 0}},
        {"parameters": {"variant": ""}},
        {"opencode_executable": ""},
    ],
)
def test_invalid_opencode_configuration(options):
    with pytest.raises(ValueError):
        Model(backend="opencode", **{"model_id": "provider/model", **options})


@pytest.mark.parametrize("structured", [True, False])
def test_system_instructions_reach_isolated_agent_and_bind_cache(tmp_path, structured):
    capture = tmp_path / "instructions.jsonl"
    body = f"""
config = json.loads(os.environ['OPENCODE_CONFIG_CONTENT'])
with open({str(capture)!r}, 'a') as f:
 f.write(json.dumps(config['agent']['taim-generation']['prompt'])+'\\n')
""" + _output_script(_completed())
    executable = _script(tmp_path, body)
    request = _request() if structured else replace(_request(), output_schema=None, validator=None)
    with closing(GenerationCache(tmp_path / "system-cache.sqlite3")) as cache:
        model = _model(executable, system_prompt="Nur Deutsch.", cache=cache)
        result = model.generate(request)
        assert result.resolved
        assert request.prompt == "Use only supplied synthetic evidence."
        assert model.generate(request).cache_hit
        assert _model(executable, system_prompt="English.", cache=cache).generate(request).resolved
    assert [json.loads(line) for line in capture.read_text().splitlines()] == [
        "Nur Deutsch.",
        "English.",
    ]


def test_system_instructions_count_towards_request_limit(tmp_path):
    executable = _script(tmp_path, "raise AssertionError('must not generate')")
    result = _model(executable, system_prompt="ü" * 100, max_request_bytes=200).generate(_request())
    assert not result.resolved
    assert result.attempts[0].metadata["failure_code"] == "model_request_too_large"


def test_free_text_prompt_carries_no_schema_and_output_is_the_text(tmp_path):
    capture = tmp_path / "capture.json"
    body = f"""
config = json.loads(os.environ['OPENCODE_CONFIG_CONTENT'])
record = {{'prompt': sys.stdin.read(), 'agent': config['agent']['taim-generation']['prompt']}}
with open({str(capture)!r}, 'w') as f: f.write(json.dumps(record))
for e in {_completed("Plain answer, not JSON.")!r}: print(json.dumps(e))
"""
    request = GenerationRequest(
        "synthetic", "Use only supplied synthetic evidence.", None, {"contract_version": "t-v1"}
    )
    result = _model(_script(tmp_path, body)).generate(request)

    assert result.resolved and result.output == "Plain answer, not JSON."
    record = json.loads(capture.read_text())
    assert record["prompt"] == "Use only supplied synthetic evidence."
    assert "JSON" not in record["agent"]
    trace = result.attempts[0].metadata["opencode"]
    assert trace["schema_enforcement"] == "free_text"
