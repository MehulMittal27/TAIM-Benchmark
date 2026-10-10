"""Bounded OpenAI-compatible chat transport. Authorization belongs to the caller."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from taim.generation import GenerationRequest, GenerationTransportError, ProviderResponse

if TYPE_CHECKING:
    import httpx


class ModelTransportError(GenerationTransportError):
    """Fixed, credential-free failure code; response bodies never become exceptions."""

    def __init__(self, code: str, *, authentication: bool = False) -> None:
        super().__init__(code)
        self.failure_kind = "authentication" if authentication else "transport"


def positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError("non-finite JSON number")


def strict_json(value: str | bytes) -> object:
    return json.loads(value, object_pairs_hook=unique_object, parse_constant=_reject_constant)


def token_usage(value: object) -> dict[str, object]:
    """Retain known numeric counters, never arbitrary provider metadata."""
    if not isinstance(value, dict):
        return {}
    names = {"prompt_tokens", "completion_tokens", "total_tokens", "input", "output", "reasoning"}
    return {
        key: number
        for key, number in value.items()
        if key in names and isinstance(number, int) and not isinstance(number, bool) and number >= 0
    }


@dataclass
class HTTPProvider:
    model: str
    endpoint: str
    parameters: Mapping[str, object]
    timeout_seconds: int
    max_request_bytes: int
    max_response_bytes: int
    system_prompt: str = ""
    auth_headers: Callable[[], Mapping[str, str]] | None = None
    transport: httpx.AsyncBaseTransport | None = None
    deadline: float | None = None
    provider_id: str = "openai-compatible-http-v1"

    def __post_init__(self) -> None:
        url = urlsplit(self.endpoint)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or any(char.isspace() for char in self.endpoint)
        ):
            raise ValueError("endpoint must be an HTTP(S) URL without credentials or query")
        if set(self.parameters) - {"temperature", "max_tokens", "response_format"}:
            raise ValueError("unsupported HTTP parameters")
        temperature = self.parameters.get("temperature", 0)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be finite and between 0 and 2")
        positive_integer(self.parameters.get("max_tokens", 4096), "max_tokens")
        if self.parameters.get("response_format", "json_schema") not in {
            "json_schema",
            "json_object",
        }:
            raise ValueError("response_format must be json_schema or json_object")

    def with_deadline(self, deadline: float) -> HTTPProvider:
        return replace(self, deadline=deadline)

    def generate(
        self, request: GenerationRequest, *, prompt: str, attempt: int, logical_call_id: str
    ) -> ProviderResponse:
        # Keep import and runtime credential resolution out of Model construction.
        import httpx

        structured = request.output_schema is not None
        messages: list[dict[str, str]] = []
        if structured:
            messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt
                    + "\nReturn only JSON matching this schema:\n"
                    + json.dumps(request.output_schema, ensure_ascii=False, allow_nan=False),
                }
            )
        elif self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.parameters.get("temperature", 0),
            "max_tokens": self.parameters.get("max_tokens", 4096),
        }
        if structured:
            # response_format constrains structured requests only; free text asks for none.
            response_format: dict[str, object] = {
                "type": self.parameters.get("response_format", "json_schema")
            }
            if response_format["type"] == "json_schema":
                response_format["json_schema"] = {
                    "name": "generation",
                    "strict": True,
                    "schema": request.output_schema,
                }
            payload["response_format"] = response_format
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        if len(body) > self.max_request_bytes:
            raise ModelTransportError("model_request_too_large")
        deadline = (
            self.deadline if self.deadline is not None else time.monotonic() + self.timeout_seconds
        )
        if deadline <= time.monotonic():
            raise ModelTransportError("model_timeout")
        headers = {"Accept-Encoding": "identity", "Content-Type": "application/json"}
        try:
            try:
                supplied = dict(self.auth_headers()) if self.auth_headers else {}
            except Exception:
                # Even an exception digest must not depend on a credential loader's message.
                raise ModelTransportError("model_unavailable", authentication=True) from None
            if any(key.lower() != "authorization" for key in supplied):
                raise ModelTransportError("unsupported_auth_header")
            headers.update(supplied)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                raw = asyncio.run(self._request(body, headers, deadline))
            else:
                # Generates is synchronous, including when called from an async application.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    raw = executor.submit(
                        lambda: asyncio.run(self._request(body, headers, deadline))
                    ).result()
        except ModelTransportError:
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise ModelTransportError("model_timeout") from None
        except Exception:
            raise ModelTransportError("model_unavailable") from None
        try:
            decoded = strict_json(raw)
            if not isinstance(decoded, dict):
                raise ValueError("expected object")
            choices = decoded["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("expected one choice")
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
                raise ValueError("incomplete response")
            message = choice["message"]
            if (
                not isinstance(message, dict)
                or message.get("role") != "assistant"
                or message.get("tool_calls")
                or message.get("function_call")
                or message.get("refusal")
            ):
                raise ValueError("unexpected model action")
            content = message["content"]
            if not isinstance(content, str) or (
                structured and not isinstance(strict_json(content), dict)
            ):
                raise ValueError("expected JSON object" if structured else "expected text")
            reported = decoded.get("model")
            if reported is not None and (not isinstance(reported, str) or not reported):
                raise ValueError("invalid model identity")
        except (KeyError, TypeError, ValueError, RecursionError):
            raise ModelTransportError("invalid_model_response") from None
        return ProviderResponse(
            raw_output=content,
            usage=token_usage(decoded.get("usage")),
            metadata={
                "trace_version": "model-generation-v1",
                "attempt": attempt,
                "provider_call": {"status": "called"},
                "http": {
                    "requested_model": self.model,
                    "reported_model": reported,
                    "model_identity_matches": reported == self.model if reported else None,
                    "runtime": "httpx",
                    "runtime_version": httpx.__version__,
                    "provider_revision_available": False,
                    "logical_call_id": logical_call_id,
                },
            },
        )

    async def _request(self, body: bytes, headers: dict[str, str], deadline: float) -> bytes:
        import httpx

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelTransportError("model_timeout")
        result = bytearray()
        async with (
            asyncio.timeout(remaining),
            httpx.AsyncClient(
                timeout=remaining,
                follow_redirects=False,
                trust_env=False,
                transport=self.transport,
            ) as client,
            client.stream("POST", self.endpoint, content=body, headers=headers) as response,
        ):
            if response.status_code != 200:
                raise ModelTransportError(
                    "model_request_failed", authentication=response.status_code in {401, 403}
                )
            if response.headers.get("Content-Encoding", "identity").casefold() not in {
                "",
                "identity",
            }:
                raise ModelTransportError("invalid_model_response")
            async for chunk in response.aiter_raw():
                if len(result) + len(chunk) > self.max_response_bytes:
                    raise ModelTransportError("model_response_too_large")
                result.extend(chunk)
        return bytes(result)
