# ruff: noqa: S101
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from taim.generation import (
    GenerationCache,
    GenerationCacheInUseError,
    GenerationCacheRunLease,
    GenerationRequest,
    ProviderResponse,
    request_identity,
    run_generation,
)
from taim.pipelines.trialgpt.system import validate_final_ranking_output, validate_keyword_output


class ScriptedProvider:
    provider_id = "fake-provider-v1"

    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[int, str]] = []

    def generate(
        self, request, *, prompt: str, attempt: int, logical_call_id: str
    ) -> ProviderResponse:
        del request, logical_call_id
        self.calls.append((attempt, prompt))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return ProviderResponse(str(response), {"output_tokens": attempt})


def _request(*, validator=validate_final_ranking_output, runtime="test") -> GenerationRequest:
    return GenerationRequest(
        stage="final_ranking",
        prompt="Return the score as JSON.",
        output_schema={"type": "object", "required": ["score"]},
        input_payload={"runtime": runtime, "topic_id": "topic-1", "trial_id": "NCT1"},
        validator=validator,
    )


def test_generation_cache_run_lease_rejects_concurrent_owners_and_releases(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "generation-cache.sqlite3"

    with (
        GenerationCacheRunLease(cache_path, run_id="run-v1"),
        pytest.raises(GenerationCacheInUseError, match="already active"),
        GenerationCacheRunLease(cache_path, run_id="run-v1"),
    ):
        raise AssertionError("a second cache owner acquired the lease")

    with GenerationCacheRunLease(cache_path, run_id="run-v1"):
        pass


def test_valid_first_attempt_records_provenance() -> None:
    provider = ScriptedProvider('{"score": 0.75}')
    result = run_generation(_request(), provider)

    assert result.resolved
    assert result.output == {"score": 0.75}
    assert result.selected_attempt == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].validation_errors == ()
    assert result.attempts[0].raw_output == '{"score": 0.75}'


def test_transport_failure_retries_original_prompt() -> None:
    provider = ScriptedProvider(RuntimeError("temporary"), '{"score": 0.5}')
    result = run_generation(_request(), provider)

    assert result.resolved
    assert result.selected_attempt == 2
    assert [attempt.kind for attempt in result.attempts] == ["original", "original"]
    assert result.attempts[0].retry_reason == "transport_failure"
    assert result.attempts[0].metadata == {
        "failure_classification": "transport",
        "provider_call": {"status": "failed"},
        "provider_configuration": {},
    }
    assert result.attempts[0].usage == {}
    assert result.attempts[0].transport_error is not None
    assert result.attempts[0].transport_error.startswith(
        "provider transport failure; detail_sha256=sha256:"
    )
    assert result.attempts[1].retry_reason == "transport_failure"
    assert result.attempts[1].transport_error is None
    assert result.attempts[1].validation_errors == ()
    assert "temporary" not in result.attempts[0].transport_error
    assert provider.calls[0][1] == provider.calls[1][1]


def test_authentication_failure_is_persisted_and_stops_without_retry(tmp_path: Path) -> None:
    class AuthenticationFailure(RuntimeError):
        failure_kind = "authentication"

    cache_path = tmp_path / "generation-cache.sqlite3"
    cache = GenerationCache(cache_path)
    provider = ScriptedProvider(AuthenticationFailure("secret detail"), '{"score": 0.5}')
    result = run_generation(_request(), provider, cache=cache)
    cache.close()

    assert not result.resolved
    assert result.selected_attempt is None
    assert len(result.attempts) == 1
    assert len(provider.calls) == 1
    assert result.attempts[0].metadata["failure_classification"] == "authentication"
    assert "secret detail" not in (result.attempts[0].transport_error or "")
    with sqlite3.connect(cache_path) as connection:
        persisted = connection.execute("SELECT result_json FROM generation_results").fetchone()
    assert persisted is not None
    assert json.loads(persisted[0])["status"] == "unresolved"

    resumed_cache = GenerationCache(cache_path)
    resumed_provider = ScriptedProvider('{"score": 0.5}')
    resumed = run_generation(_request(), resumed_provider, cache=resumed_cache)
    resumed_cache.close()
    assert not resumed.resolved
    assert resumed.cache_hit
    assert resumed.attempts[0].metadata["failure_classification"] == "authentication"
    assert resumed_provider.calls == []


def test_malformed_json_uses_repair_prompt() -> None:
    provider = ScriptedProvider("not json", '{"score": 0.25}')
    result = run_generation(_request(), provider)

    assert result.resolved
    assert result.selected_attempt == 2
    assert result.attempts[1].retry_reason == "invalid_json"
    assert result.attempts[1].kind == "repair"
    assert "Previous response:" in provider.calls[1][1]
    assert "invalid JSON" in provider.calls[1][1]


def test_semantic_error_is_repaired() -> None:
    provider = ScriptedProvider('{"score": "high"}', '{"score": 0.9}')
    result = run_generation(_request(), provider)

    assert result.resolved
    assert result.attempts[0].validation_errors == ("score must be a finite number",)
    assert result.attempts[1].retry_reason == "semantic_validation"


def test_normalizer_runs_before_semantic_validation() -> None:
    provider = ScriptedProvider('{"sentence_ids":["0", "12"]}')

    def normalize(value: object) -> object:
        assert isinstance(value, dict)
        return {"sentence_ids": [int(item) for item in value["sentence_ids"]]}

    def validate(value: object) -> tuple[str, ...]:
        return () if value == {"sentence_ids": [0, 12]} else ("not normalized",)

    request = GenerationRequest(
        stage="criterion_matching_inclusion",
        prompt="Return matching JSON.",
        output_schema={"type": "object"},
        input_payload={"trial_id": "NCT1"},
        validator=validate,
        normalizer=normalize,
        normalizer_version="test-normalizer-v1",
    )
    result = run_generation(request, provider)

    assert result.resolved
    assert result.output == {"sentence_ids": [0, 12]}
    assert result.attempts[0].raw_output == '{"sentence_ids":["0", "12"]}'


def test_normalizer_requires_version() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        GenerationRequest(
            stage="criterion_matching_inclusion",
            prompt="Return matching JSON.",
            output_schema={"type": "object"},
            input_payload={"trial_id": "NCT1"},
            normalizer=lambda value: value,
        )


def test_normalizer_version_requires_normalizer() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        GenerationRequest(
            stage="criterion_matching_inclusion",
            prompt="Return matching JSON.",
            output_schema={"type": "object"},
            input_payload={"trial_id": "NCT1"},
            normalizer_version="test-normalizer-v1",
        )


def test_normalizer_version_changes_cache_identity_without_changing_prompt_or_schema() -> None:
    common = {
        "stage": "criterion_matching_inclusion",
        "prompt": "Frozen matching prompt.",
        "output_schema": {"type": "object"},
        "input_payload": {"topic_id": "1", "trial_id": "NCT1"},
        "normalizer": lambda value: value,
    }
    v1 = GenerationRequest(**common, normalizer_version="trialgpt-matching-normalizer-v1")
    v2 = GenerationRequest(**common, normalizer_version="trialgpt-matching-normalizer-v2")

    assert v1.prompt == v2.prompt
    assert v1.output_schema == v2.output_schema
    assert request_identity(v1, "provider") != request_identity(v2, "provider")


def test_exhaustion_is_unresolved_and_never_a_negative_score() -> None:
    provider = ScriptedProvider("bad", "still bad", "nope")
    result = run_generation(_request(), provider)

    assert not result.resolved
    assert result.status == "unresolved"
    assert result.output is None
    assert result.selected_attempt is None
    assert len(result.attempts) == 3


def test_exhaustion_is_durable_but_not_reused_as_a_cache_hit(tmp_path: Path) -> None:
    class ObservedProvider(ScriptedProvider):
        def generate(self, request, **kwargs: object) -> ProviderResponse:
            response = super().generate(request, **kwargs)
            return ProviderResponse(
                response.raw_output,
                response.usage,
                {
                    "provider_call": {"status": "called"},
                    "model_provenance": {"identity_sha256": "sha256:" + "a" * 64},
                },
            )

    path = tmp_path / "generation-cache.sqlite3"
    cache = GenerationCache(path)
    request = _request()
    provider = ObservedProvider(
        '{"score":"bad"}',
        '{"score":"still bad"}',
        '{"score":"nope"}',
    )

    result = run_generation(request, provider, cache=cache)
    cache.close()

    identity = request_identity(request, provider.provider_id)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT result_json FROM generation_results WHERE identity_sha256 = ?",
            (identity,),
        ).fetchone()
    assert row is not None
    persisted = json.loads(row[0])
    assert persisted["status"] == "unresolved"
    assert persisted["logical_call_id"] == identity
    assert persisted["topic_id"] == "topic-1"
    assert persisted["trial_id"] == "NCT1"
    assert persisted["attempts"] == [attempt.to_dict() for attempt in result.attempts]
    assert all(
        attempt["raw_output_sha256"].startswith("sha256:") for attempt in persisted["attempts"]
    )
    assert all(
        attempt["parsed_output_sha256"].startswith("sha256:") for attempt in persisted["attempts"]
    )
    assert [attempt["usage"] for attempt in persisted["attempts"]] == [
        {"output_tokens": 1},
        {"output_tokens": 2},
        {"output_tokens": 3},
    ]
    assert all(attempt["validation_errors"] for attempt in persisted["attempts"])
    assert all(
        attempt["metadata"]["provider_call"] == {"status": "called"}
        for attempt in persisted["attempts"]
    )

    resumed_cache = GenerationCache(path)
    assert resumed_cache.load(identity) is None
    preserved = resumed_cache.load(identity, include_unresolved=True)
    assert preserved is not None
    assert preserved.status == "unresolved"
    assert preserved.cache_hit
    assert preserved.attempts == result.attempts
    resumed_cache.close()


def test_cache_requires_exact_identity_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "generation-cache.sqlite3"
    cache = GenerationCache(path)
    provider = ScriptedProvider('{"score": 0.1}')
    first = run_generation(_request(), provider, cache=cache)
    cache.close()
    resumed_cache = GenerationCache(path)
    second_provider = ScriptedProvider('{"score": 0.9}')
    second = run_generation(_request(), second_provider, cache=resumed_cache)

    assert first.output == second.output == {"score": 0.1}
    assert second.cache_hit
    assert second_provider.calls == []
    assert path.is_file()

    changed = run_generation(_request(runtime="changed"), second_provider, cache=resumed_cache)
    assert changed.output == {"score": 0.9}
    assert not changed.cache_hit
    assert request_identity(_request(), provider.provider_id) != request_identity(
        _request(runtime="changed"), provider.provider_id
    )
    resumed_cache.close()


def test_failure_aware_cache_retries_transport_exhaustion_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "generation-cache.sqlite3"
    request = _request()
    failed_provider = ScriptedProvider(
        RuntimeError("temporary-1"),
        RuntimeError("temporary-2"),
        RuntimeError("temporary-3"),
    )
    failed_cache = GenerationCache(path)
    failed = run_generation(
        request,
        failed_provider,
        cache=failed_cache,
        reuse_unresolved=True,
    )
    failed_cache.close()
    assert not failed.resolved

    resumed_provider = ScriptedProvider('{"score": 0.4}')
    resumed_cache = GenerationCache(path)
    resumed = run_generation(
        request,
        resumed_provider,
        cache=resumed_cache,
        reuse_unresolved=True,
    )

    assert resumed.resolved
    assert resumed.output == {"score": 0.4}
    assert not resumed.cache_hit
    assert [attempt.attempt for attempt in resumed.attempts] == [1, 2, 3, 4]
    assert resumed.selected_attempt == 4
    assert resumed.attempts[:3] == failed.attempts
    assert resumed.attempts[3].retry_reason == "transport_failure"
    assert [attempt.transport_error is not None for attempt in resumed.attempts] == [
        True,
        True,
        True,
        False,
    ]
    assert resumed_provider.calls == [(4, request.prompt)]
    resumed_cache.close()

    observer = GenerationCache(path)
    preserved = observer.load(request_identity(request, resumed_provider.provider_id))
    observer.close()
    assert preserved is not None
    assert preserved.resolved
    assert preserved.attempts == resumed.attempts


def test_failure_aware_cache_reuses_semantic_exhaustion_after_transport_recovery(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generation-cache.sqlite3"
    request = _request()
    failed_cache = GenerationCache(path)
    transport = run_generation(
        request,
        ScriptedProvider(
            RuntimeError("temporary-1"),
            RuntimeError("temporary-2"),
            RuntimeError("temporary-3"),
        ),
        cache=failed_cache,
        reuse_unresolved=True,
    )
    failed_cache.close()
    assert not transport.resolved

    semantic_cache = GenerationCache(path)
    semantic = run_generation(
        request,
        ScriptedProvider("not-json-1", "not-json-2", "not-json-3"),
        cache=semantic_cache,
        reuse_unresolved=True,
    )
    semantic_cache.close()
    assert not semantic.resolved
    assert [attempt.attempt for attempt in semantic.attempts] == [1, 2, 3, 4, 5, 6]
    assert all(attempt.transport_error is None for attempt in semantic.attempts[-3:])

    observer = GenerationCache(path)
    unused_provider = ScriptedProvider('{"score": 0.9}')
    preserved = run_generation(
        request,
        unused_provider,
        cache=observer,
        reuse_unresolved=True,
    )
    observer.close()
    assert not preserved.resolved
    assert preserved.cache_hit
    assert preserved.attempts == semantic.attempts
    assert unused_provider.calls == []


def test_cache_supports_batched_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "generation-cache.sqlite3"
    cache = GenerationCache(path, flush_every=2)
    provider = ScriptedProvider('{"score": 0.1}', '{"score": 0.2}', '{"score": 0.3}')
    one = _request(runtime="one")
    two = _request(runtime="two")
    three = _request(runtime="three")

    run_generation(one, provider, cache=cache)
    observer = GenerationCache(path)
    assert observer.load(request_identity(one, provider.provider_id)) is None
    observer.close()
    run_generation(two, provider, cache=cache)
    observer = GenerationCache(path)
    assert observer.load(request_identity(one, provider.provider_id)) is not None
    assert observer.load(request_identity(two, provider.provider_id)) is not None
    observer.close()

    run_generation(three, provider, cache=cache)
    observer = GenerationCache(path)
    assert observer.load(request_identity(three, provider.provider_id)) is None
    observer.close()
    cache.flush()
    observer = GenerationCache(path)
    assert observer.load(request_identity(three, provider.provider_id)) is not None
    observer.close()
    cache.close()


def test_keyword_validator_rejects_missing_conditions() -> None:
    assert validate_keyword_output({"summary": "patient"}) == (
        "conditions must contain 1 to 32 items",
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.5", True])
def test_final_validator_rejects_non_finite_or_non_numeric(value: object) -> None:
    assert validate_final_ranking_output({"score": value}) == ("score must be a finite number",)


def test_strict_json_policy_changes_identity_without_changing_legacy_decoding():
    from dataclasses import replace

    request = _request()
    strict = replace(request, reject_duplicate_keys=True)
    assert request_identity(strict, "provider") != request_identity(request, "provider")
    assert request_identity(request, "provider") == request_identity(
        replace(request, reject_duplicate_keys=False), "provider"
    )
    duplicate = '{"score": 0.25, "score": 0.75}'
    legacy = run_generation(request, ScriptedProvider(duplicate))
    assert legacy.resolved and legacy.output == {"score": 0.75}
    strict_result = run_generation(strict, ScriptedProvider(duplicate, '{"score": 0.5}'))
    assert strict_result.resolved and strict_result.output == {"score": 0.5}
    assert strict_result.attempts[0].transport_error is None
    assert strict_result.attempts[0].validation_errors == (
        "invalid JSON: duplicate JSON key: score",
    )
    assert strict_result.attempts[1].kind == "repair"


def _text_request(**updates: object) -> GenerationRequest:
    from dataclasses import replace

    request = GenerationRequest(
        stage="chat",
        prompt="SYSTEM INSTRUCTIONS:\nTranslate to SMT.\n\nage >= 18",
        output_schema=None,
        input_payload={"contract_version": "free-text-v1"},
    )
    return replace(request, **updates)


def test_structured_identity_is_unchanged_by_the_free_text_option() -> None:
    # Recorded on main at 3616e34, before a request could omit its schema.
    assert request_identity(_request(), "fake-provider-v1") == (
        "sha256:657620b01510746fb2c3963257a9453dca99c625e3ab6050827fcb56cc624852"
    )
    assert request_identity(_text_request(), "fake-provider-v1") != request_identity(
        _text_request(output_schema={}), "fake-provider-v1"
    )


def test_free_text_output_is_the_text_itself_never_parsed() -> None:
    provider = ScriptedProvider('{"score": 1} is my answer', "unused")
    result = run_generation(_text_request(), provider)

    assert result.resolved and result.output == '{"score": 1} is my answer'
    assert provider.calls == [(1, _text_request().prompt)]
    (attempt,) = result.attempts
    assert attempt.parsed_output == attempt.raw_output == result.output
    assert attempt.validation_errors == ()
    # JSON that parses is still text: the request asked for none.
    assert run_generation(_text_request(), ScriptedProvider('{"score": 1}')).output == (
        '{"score": 1}'
    )


def test_empty_free_text_is_repaired_without_a_schema_in_the_prompt() -> None:
    provider = ScriptedProvider("  \n", "(assert (>= age 18))")
    result = run_generation(_text_request(), provider)

    assert result.resolved and result.output == "(assert (>= age 18))"
    first, second = result.attempts
    assert first.validation_errors == ("response must be non-empty text",)
    assert (second.kind, second.retry_reason) == ("repair", "empty_text")
    repair = provider.calls[1][1]
    assert repair.startswith(_text_request().prompt + "\n\nYour previous response was invalid.")
    assert "schema" not in repair.removeprefix(_text_request().prompt)


def test_free_text_validator_sees_the_string_and_cache_round_trips(tmp_path: Path) -> None:
    request = _text_request(
        validator=lambda value: () if "assert" in str(value) else ("expected an assertion",)
    )
    provider = ScriptedProvider("no idea", "(assert true)")
    cache = GenerationCache(tmp_path / "cache.sqlite3")
    try:
        result = run_generation(request, provider, cache=cache)
        assert result.output == "(assert true)"
        assert result.attempts[0].validation_errors == ("expected an assertion",)
        cached = run_generation(request, ScriptedProvider(), cache=cache)
        assert cached.cache_hit and cached.output == "(assert true)"
    finally:
        cache.close()
