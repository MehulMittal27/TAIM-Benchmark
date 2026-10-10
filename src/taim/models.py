"""Configured structured or free-text generation with the existing bounded generation runner.

Construction and configuration access are inert. ``generate`` can transmit supplied content,
spend provider quota, and write sensitive responses to an injected cache. The runner owns
input authorization, storage, and cache lifecycle. Codex uses a CLI runtime, not necessarily
local inference or a raw model API. Its flags do not establish hospital-local suitability.

Schemas are passed to the transport, except by a Codex Model configured with
``output_schema_transport="none"``; local acceptance depends on the request's validator.
New consumers must validate output shape and semantics and bind contract versions in their
input payload. Validator code is not automatically part of cache identity. A request with no
output schema asks for free text through the same path: the output is the model's text.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, cast

from taim.generation import GenerationCache, GenerationRequest, GenerationResult, run_generation
from taim.model_codex import CODEX_OUTPUT_SCHEMA_TRANSPORTS, CodexProvider
from taim.model_http import HTTPProvider, positive_integer
from taim.model_opencode import OpenCodeProvider
from taim.schemas import JsonValue, freeze_json_value, json_value_to_builtins

if TYPE_CHECKING:
    import httpx


class Generates(Protocol):
    """One-method structural seam for consumers that only need generation."""

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class Model:
    """One explicit model configuration; no authentication or lifecycle manager.

    Codex system instructions must fit in a CLI argument of fewer than 128 KiB
    after TOML escaping, including the option name. Oversized instructions are rejected
    during construction.

    ``system_prompt`` supplies application instructions through the native transport:
    an HTTP system message, OpenCode agent prompt, or Codex developer instructions.
    It binds configuration/cache identity without rewriting the GenerationRequest.
    CLI/model built-in instructions remain in force.
    """

    def __init__(
        self,
        *,
        model_id: str,
        backend: str,
        parameters: Mapping[str, JsonValue] | None = None,
        cache: GenerationCache | None = None,
        codex_executable: str = "codex",
        request_timeout_seconds: int = 600,
        reuse_unresolved: bool = False,
        endpoint: str | None = None,
        system_prompt: str = "",
        max_request_bytes: int = 1_048_576,
        max_response_bytes: int = 1_048_576,
        auth_headers: Callable[[], Mapping[str, str]] | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
        deadline: Callable[[], float] | None = None,
        opencode_executable: str = "opencode",
        opencode_environment: Callable[[], Mapping[str, str]] | None = None,
    ) -> None:
        if backend not in {"codex", "http", "opencode"}:
            raise ValueError("unsupported model backend")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be explicit")
        positive_integer(request_timeout_seconds, "request_timeout_seconds")
        positive_integer(max_request_bytes, "max_request_bytes")
        positive_integer(max_response_bytes, "max_response_bytes")
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string")
        if backend != "http" and (
            endpoint is not None or auth_headers is not None or http_transport is not None
        ):
            raise ValueError("HTTP options require the HTTP backend")
        if backend != "opencode" and opencode_environment is not None:
            raise ValueError("OpenCode environment requires the OpenCode backend")
        if backend == "codex" and deadline is not None:
            raise ValueError("external deadline is supported by HTTP and OpenCode only")
        params = dict(parameters or {})
        self._transport: CodexProvider | HTTPProvider | OpenCodeProvider
        configuration: dict[str, JsonValue]
        if backend == "codex":
            if not isinstance(codex_executable, str) or not codex_executable.strip():
                raise ValueError("codex_executable must be non-empty")
            if set(params) - {"reasoning_effort", "output_schema_transport"}:
                raise ValueError("unsupported Codex parameters")
            effort = params.get("reasoning_effort")
            if effort is not None and (
                not isinstance(effort, str)
                or effort
                not in {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
            ):
                raise ValueError("invalid Codex reasoning_effort")
            schema_transport = params.get("output_schema_transport", "strict")
            if (
                not isinstance(schema_transport, str)
                or schema_transport not in CODEX_OUTPUT_SCHEMA_TRANSPORTS
            ):
                raise ValueError("invalid Codex output_schema_transport")
            self._transport = CodexProvider(
                model=model_id,
                reasoning_effort=effort,
                codex_executable=codex_executable,
                request_timeout_seconds=request_timeout_seconds,
                system_prompt=system_prompt,
                max_request_bytes=max_request_bytes,
                output_schema_transport=schema_transport,
            )
            codex_transport: dict[str, JsonValue] = {
                "codex_executable": codex_executable,
                "max_request_bytes": max_request_bytes,
                "request_timeout_seconds": request_timeout_seconds,
            }
            # Recorded only when it departs from the default, so every strict configuration and
            # the request identities it binds stay as they were.
            if schema_transport != "strict":
                codex_transport["output_schema_transport"] = schema_transport
            configuration = {
                "configuration_version": "model-v1",
                "provider_id": "codex-cli-v1",
                "model": model_id,
                "reasoning_effort": effort,
                "transport": codex_transport,
            }
        else:
            if backend == "http":
                if not isinstance(endpoint, str) or not endpoint:
                    raise ValueError("HTTP endpoint must be explicit")
                params = {
                    "temperature": 0,
                    "max_tokens": 4096,
                    "response_format": "json_schema",
                    **params,
                }
                self._transport = HTTPProvider(
                    model=model_id,
                    endpoint=endpoint,
                    parameters=params,
                    timeout_seconds=request_timeout_seconds,
                    max_request_bytes=max_request_bytes,
                    max_response_bytes=max_response_bytes,
                    system_prompt=system_prompt,
                    auth_headers=auth_headers,
                    transport=http_transport,
                )
                transport_config: dict[str, JsonValue] = {
                    "endpoint": endpoint,
                    "system_prompt": system_prompt,
                }
            else:
                if "/" not in model_id or not all(model_id.split("/", 1)):
                    raise ValueError("OpenCode model must be provider/model")
                if not isinstance(opencode_executable, str) or not opencode_executable.strip():
                    raise ValueError("opencode_executable must be non-empty")
                if set(params) - {"variant"}:
                    raise ValueError("unsupported OpenCode parameters")
                variant = params.get("variant")
                if variant is not None and (not isinstance(variant, str) or not variant.strip()):
                    raise ValueError("OpenCode variant must be a non-empty string")
                self._transport = OpenCodeProvider(
                    model=model_id,
                    executable=opencode_executable,
                    variant=variant,
                    timeout_seconds=request_timeout_seconds,
                    max_request_bytes=max_request_bytes,
                    max_response_bytes=max_response_bytes,
                    system_prompt=system_prompt,
                    environment=opencode_environment,
                )
                transport_config = {"opencode_executable": opencode_executable}
            configuration = {
                "configuration_version": "model-v1",
                "provider_id": self._transport.provider_id,
                "model": model_id,
                "parameters": params,
                "transport": {
                    **transport_config,
                    "request_timeout_seconds": request_timeout_seconds,
                    "max_request_bytes": max_request_bytes,
                    "max_response_bytes": max_response_bytes,
                },
            }
        if system_prompt and backend != "http":
            configuration["system_prompt"] = system_prompt
        self._configuration = cast(Mapping[str, JsonValue], freeze_json_value(configuration))
        self._deadline = deadline
        self._timeout_seconds = request_timeout_seconds
        self._cache = cache
        self._reuse_unresolved = reuse_unresolved

    @property
    def configuration(self) -> Mapping[str, JsonValue]:
        """Immutable requested settings. Observations live in generation provenance."""
        return self._configuration

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate with up to three attempts per invocation, using optional response reuse.

        ``request.output_schema`` selects the output: a schema for JSON, ``None`` for free text.
        """
        transport = self._transport
        if isinstance(transport, (HTTPProvider, OpenCodeProvider)):
            deadline = time.monotonic() + self._timeout_seconds
            if self._deadline is not None:
                external = self._deadline()
                if not math.isfinite(external):
                    raise ValueError("deadline must be finite")
                deadline = min(deadline, external)
            transport = transport.with_deadline(deadline)
        return run_generation(
            request,
            transport,
            provider_config=cast(Mapping[str, object], json_value_to_builtins(self.configuration)),
            cache=self._cache,
            reuse_unresolved=self._reuse_unresolved,
        )
