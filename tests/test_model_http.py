# ruff: noqa: S101
"""HTTP Model qualification with in-memory transports and synthetic payloads only."""

import asyncio
import json
import time
from contextlib import closing

import httpx
import pytest

from taim.generation import GenerationCache, GenerationRequest
from taim.models import Model


def _request():
    return GenerationRequest(
        "synthetic",
        "Explain supplied evidence in German.",
        {"type": "object"},
        {"contract_version": "synthetic-v1"},
        validator=lambda value: () if value == {"score": 1} else ("expected score 1",),
    )


def _envelope(content='{"score":1}', **updates):
    return {
        "model": "observed-model",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                },
            }
        ],
        **updates,
    }


def _response(value=None, *, raw=None, status=200, headers=None):
    return httpx.Response(
        status,
        headers=headers,
        stream=httpx.ByteStream(raw if raw is not None else json.dumps(value).encode()),
    )


def _model(handler, **kwargs):
    return Model(
        model_id="requested-model",
        backend="http",
        endpoint="https://synthetic.invalid/v1/chat/completions",
        http_transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_http_repairs_cache_and_runtime_credentials(tmp_path):
    calls = []
    auth_calls = []

    def credentials():
        auth_calls.append(True)
        return {"Authorization": "Bearer synthetic-private-token"}

    def handler(req):
        calls.append(json.loads(req.content))
        assert req.headers["Authorization"] == "Bearer synthetic-private-token"
        assert req.headers["Accept-Encoding"] == "identity"
        assert req.url.path == "/v1/chat/completions"
        return _response(_envelope('{"score":0}' if len(calls) == 1 else '{"score":1}'))

    with closing(GenerationCache(tmp_path / "cache.sqlite3")) as cache:
        configured = _model(
            handler, auth_headers=credentials, system_prompt="Nur Deutsch.", cache=cache
        )
        assert not auth_calls
        result = configured.generate(_request())
        assert result.resolved and len(result.attempts) == 2
        assert "expected score 1" in calls[1]["messages"][1]["content"]
        assert calls[0]["messages"][0]["content"].startswith("Nur Deutsch.")
        assert calls[0]["response_format"]["json_schema"]["strict"] is True
        assert configured.generate(_request()).cache_hit
        assert len(calls) == len(auth_calls) == 2
        provenance = result.attempts[0].metadata["http"]
        assert provenance["requested_model"] == "requested-model"
        assert provenance["reported_model"] == "observed-model"
        assert provenance["model_identity_matches"] is False
        assert provenance["runtime_version"] == httpx.__version__
        assert "synthetic-private-token" not in json.dumps(result.to_dict())
        assert "synthetic-private-token" not in str(configured.configuration)
        changed = _model(handler, auth_headers=credentials, system_prompt="Different", cache=cache)
        assert not changed.generate(_request()).cache_hit
    assert b"synthetic-private-token" not in (tmp_path / "cache.sqlite3").read_bytes()


@pytest.mark.parametrize("status,attempts", [(301, 3), (401, 1), (403, 1), (429, 3), (500, 3)])
def test_status_failures_are_bounded_and_do_not_follow_redirects(status, attempts):
    calls = []

    def handler(req):
        calls.append(req)
        return _response(
            raw=b"synthetic-secret-body",
            status=status,
            headers={"Location": "https://other.invalid"},
        )

    result = _model(handler).generate(_request())
    assert not result.resolved and result.output is None
    assert len(calls) == len(result.attempts) == attempts
    assert all(a.metadata["failure_code"] == "model_request_failed" for a in result.attempts)
    assert "synthetic-secret-body" not in json.dumps(result.to_dict())


@pytest.mark.parametrize(
    "value",
    [
        _envelope('{"score":1,"score":1}'),
        _envelope('{"score":NaN}'),
        _envelope("[]"),
        _envelope("not-json"),
        _envelope(choices=[]),
        _envelope(choices=[{"finish_reason": "length", "message": {"content": '{"score":1}'}}]),
        _envelope(
            choices=[
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"score":1}',
                        "tool_calls": [{}],
                    },
                }
            ]
        ),
        _envelope(
            choices=[
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"score":1}',
                        "refusal": "refused",
                    },
                }
            ]
        ),
    ],
)
def test_strict_envelope_rejection(value):
    result = _model(lambda req: _response(value)).generate(_request())
    assert not result.resolved
    assert len(result.attempts) == 3
    assert result.attempts[-1].metadata["failure_code"] == "invalid_model_response"


def test_duplicate_envelope_and_compression_rejected():
    for reply in [
        lambda: _response(raw=b'{"choices":[],"choices":[]}'),
        lambda: _response(_envelope(), headers={"Content-Encoding": "gzip"}),
    ]:
        result = _model(lambda req, reply=reply: reply()).generate(_request())
        assert not result.resolved
        assert result.attempts[-1].metadata["failure_code"] == "invalid_model_response"


def test_request_and_response_byte_limits():
    calls = []

    def handler(req):
        calls.append(req)
        return _response(_envelope())

    result = _model(handler, max_request_bytes=10).generate(_request())
    assert not calls
    assert result.attempts[0].metadata["failure_code"] == "model_request_too_large"
    result = _model(handler, max_response_bytes=10).generate(_request())
    assert result.attempts[0].metadata["failure_code"] == "model_response_too_large"


def test_expired_zero_deadline_never_calls_transport():
    def forbidden(req):
        pytest.fail("expired deadline must not send data")

    result = _model(forbidden, deadline=lambda: 0).generate(_request())
    assert not result.resolved
    assert all(a.metadata["failure_code"] == "model_timeout" for a in result.attempts)


def test_credential_loader_errors_do_not_even_leave_credential_dependent_hashes():
    errors = []
    for detail in ("first-private-token", "second-private-token"):

        def fail(detail=detail):
            raise RuntimeError(detail)

        result = _model(lambda req: pytest.fail("must not send"), auth_headers=fail).generate(
            _request()
        )
        assert not result.resolved and len(result.attempts) == 1
        assert result.attempts[0].metadata["failure_code"] == "model_unavailable"
        errors.append(result.attempts[0].transport_error)
    assert errors[0] == errors[1]


def test_whole_invocation_deadline_covers_slow_chunks_and_closes_stream():
    closed = []
    calls = []

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b" "
                await asyncio.sleep(0.01)

        async def aclose(self):
            closed.append(True)

    def handler(req):
        calls.append(req)
        return httpx.Response(200, stream=SlowStream())

    started = time.monotonic()
    result = _model(handler, deadline=lambda: started + 0.5).generate(_request())
    assert time.monotonic() - started < 3
    assert not result.resolved
    assert len(calls) == len(closed) == 1
    assert all(a.metadata["failure_code"] == "model_timeout" for a in result.attempts)


def test_sync_generate_inside_running_loop_and_json_object_mode():
    def handler(req):
        assert json.loads(req.content)["response_format"] == {"type": "json_object"}
        return _response(_envelope())

    async def run():
        return _model(handler, parameters={"response_format": "json_object"}).generate(_request())

    assert asyncio.run(run()).resolved


def test_fractional_temperature_is_immutable_and_sent_unchanged():
    def handler(req):
        assert json.loads(req.content)["temperature"] == 0.25
        return _response(_envelope())

    configured = _model(handler, parameters={"temperature": 0.25})
    assert configured.configuration["parameters"]["temperature"] == 0.25
    with pytest.raises(TypeError):
        configured.configuration["parameters"]["temperature"] = 1
    assert configured.generate(_request()).resolved


@pytest.mark.parametrize(
    "options",
    [
        {"endpoint": "https://user:password@synthetic.invalid"},
        {"endpoint": "https://synthetic.invalid?api_key=secret"},
        {"parameters": {"api_key": "secret"}},
        {"parameters": {"max_tokens": True}},
        {"parameters": {"temperature": float("nan")}},
        {"parameters": {"response_format": "text"}},
        {"max_response_bytes": 0},
    ],
)
def test_bad_configuration_fails_inertly(options):
    with pytest.raises(ValueError):
        Model(
            model_id="test",
            backend="http",
            **{
                "endpoint": "https://synthetic.invalid/v1/chat/completions",
                **options,
            },
        )


def test_all_backend_constructors_are_inert(monkeypatch):
    import subprocess
    import tempfile

    def forbidden(*args, **kwargs):
        pytest.fail("construction performed an effect")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", forbidden)
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    Model(
        model_id="x",
        backend="http",
        endpoint="https://synthetic.invalid",
        auth_headers=forbidden,
        deadline=forbidden,
    )
    Model(model_id="provider/x", backend="opencode", opencode_environment=forbidden)


def _text_request():
    return GenerationRequest(
        "synthetic", "Explain supplied evidence in German.", None, {"contract_version": "t-v1"}
    )


@pytest.mark.parametrize("system_prompt", ["", "Nur Deutsch."])
def test_free_text_sends_no_schema_and_keeps_the_text(system_prompt):
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        return _response(_envelope("Die Evidenz reicht nicht aus."))

    result = _model(handler, system_prompt=system_prompt).generate(_text_request())

    assert result.resolved and result.output == "Die Evidenz reicht nicht aus."
    (payload,) = calls
    assert "response_format" not in payload
    expected = [{"role": "system", "content": system_prompt}] if system_prompt else []
    assert payload["messages"] == [
        *expected,
        {"role": "user", "content": "Explain supplied evidence in German."},
    ]
    assert "schema" not in json.dumps(payload)


def test_free_text_still_rejects_a_non_text_or_actioned_envelope():
    for envelope in (
        _envelope(None),
        _envelope("text", choices=[{"finish_reason": "length", "message": {}}]),
    ):
        result = _model(lambda req, envelope=envelope: _response(envelope)).generate(
            _text_request()
        )
        assert not result.resolved
        assert {a.metadata["failure_code"] for a in result.attempts} == {"invalid_model_response"}
