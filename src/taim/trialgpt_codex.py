"""Constrained Codex CLI transport for the TrialGPT provider boundary.

This module deliberately owns transport and provenance only. Prompt construction, semantic
validation, and bounded repair belong to the provider-neutral TrialGPT protocol (issue #13).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from taim.trialgpt_generation import ProviderResponse

CODEX_REQUEST_TIMEOUT_SECONDS = 600


class CodexProviderError(RuntimeError):
    """Base error for a failed or malformed Codex transport call."""


class CodexTransportError(CodexProviderError):
    """Raised when Codex cannot complete a generation request."""

    def __init__(self, message: str, *, failure_kind: str = "transport") -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


_AUTHENTICATION_FAILURE = re.compile(
    r"(?:\b401\b|\b403\b|unauthori[sz]ed|forbidden|authentication|not logged in|api[ _-]?key)",
    re.IGNORECASE,
)


def _transport_failure_kind(raw: str) -> str:
    return "authentication" if _AUTHENTICATION_FAILURE.search(raw) else "transport"


def sha256_text(value: str) -> str:
    """Return the stable hash format used by TAIM provenance records."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_bytes(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def schema_hash(schema: Mapping[str, object]) -> str:
    """Hash a JSON schema canonically so key order cannot change identity."""

    return sha256_text(_json_bytes(schema))


@dataclass(frozen=True)
class CodexRequest:
    """One provider-neutral semantic generation request.

    ``prompt`` is the already-frozen TrialGPT semantic prompt. The Codex transport must not
    rewrite it or add provider-specific clinical instructions.
    """

    stage: str
    prompt: str
    output_schema: Mapping[str, object]
    model: str | None
    reasoning_effort: str | None = None
    input_hash: str | None = None
    logical_call_id: str | None = None

    @property
    def prompt_sha256(self) -> str:
        return sha256_text(self.prompt)

    @property
    def output_schema_sha256(self) -> str:
        return schema_hash(self.output_schema)

    @property
    def identity(self) -> str:
        payload = {
            "stage": self.stage,
            "prompt_sha256": self.prompt_sha256,
            "output_schema_sha256": self.output_schema_sha256,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "input_hash": self.input_hash,
        }
        return sha256_text(_json_bytes(payload))


@dataclass
class CodexResult:
    """Raw and normalized evidence returned from one successful Codex call."""

    text: str
    requested_model: str | None
    reported_model: str | None
    codex_version: str
    stage: str
    prompt_sha256: str
    output_schema_sha256: str
    request_identity: str
    logical_call_id: str | None
    input_hash: str | None
    reasoning_effort: str | None
    raw_event_stream: str
    events: list[dict[str, object]] = field(default_factory=list)
    usage: dict[str, object] | None = None
    model_identity_matches: bool | None = None
    model_provenance: dict[str, object] = field(default_factory=dict)
    attempts: int = 1
    latency_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        """Serialize the result into a JSON-compatible provenance record."""

        return {
            "text": self.text,
            "requested_model": self.requested_model,
            "reported_model": self.reported_model,
            "codex_version": self.codex_version,
            "stage": self.stage,
            "prompt_sha256": self.prompt_sha256,
            "output_schema_sha256": self.output_schema_sha256,
            "request_identity": self.request_identity,
            "logical_call_id": self.logical_call_id,
            "input_hash": self.input_hash,
            "reasoning_effort": self.reasoning_effort,
            "raw_event_stream": self.raw_event_stream,
            "events": self.events,
            "usage": self.usage,
            "model_identity_matches": self.model_identity_matches,
            "model_provenance": self.model_provenance,
            "attempts": self.attempts,
            "latency_seconds": self.latency_seconds,
        }


def build_codex_command(
    request: CodexRequest,
    *,
    schema_path: Path,
    workdir: Path,
    codex_executable: str = "codex",
) -> list[str]:
    """Build the constrained CLI invocation used for every request.

    The command intentionally omits ``--search``, MCP configuration, repository directories, and
    writable mounts. ``--ignore-user-config`` prevents user config from adding tools or changing
    generation behavior. The empty ephemeral workdir is not a Git repository.
    """

    command = [
        codex_executable,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--json",
        "-c",
        'web_search="disabled"',
        "--output-schema",
        str(schema_path),
        "--cd",
        str(workdir),
    ]
    if request.model:
        command.extend(["--model", request.model])
    if request.reasoning_effort:
        command.extend(["-c", f"model_reasoning_effort={json.dumps(request.reasoning_effort)}"])
    return command


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_text(item) for item in value]
        joined = "".join(part for part in parts if part is not None)
        return joined or None
    if isinstance(value, Mapping):
        for key in ("text", "content", "output_text", "message"):
            candidate = _text(value.get(key))
            if candidate:
                return candidate
    return None


def _event_text(event: Mapping[str, object]) -> str | None:
    item = _mapping(event.get("item"))
    if item is not None:
        item_type = item.get("type")
        if item_type in {"agent_message", "assistant_message", "final_answer", "output_text"}:
            return _text(item)
    event_type = event.get("type")
    if event_type in {"response.output_text.done", "message.completed", "final_answer"}:
        return _text(event)
    return None


def _reported_model(events: Sequence[Mapping[str, object]]) -> str | None:
    for event in events:
        for key in ("model", "model_name", "reported_model"):
            value = event.get(key)
            if isinstance(value, str) and value:
                return value
        response = _mapping(event.get("response"))
        if response is not None:
            value = response.get("model")
            if isinstance(value, str) and value:
                return value
    return None


def _usage(events: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    for event in reversed(events):
        value = event.get("usage")
        if isinstance(value, Mapping):
            return dict(value)
        response = _mapping(event.get("response"))
        response_usage = response.get("usage") if response is not None else None
        if isinstance(response_usage, Mapping):
            return dict(response_usage)
    return None


def _model_identity_matches(requested: str | None, reported: str | None) -> bool | None:
    if not requested or reported is None:
        return None
    return reported == requested or reported.startswith(requested + "-")


def _parse_events(raw: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            events.append(dict(value))
    return events


def _minimal_environment() -> dict[str, str]:
    """Keep execution environment small while retaining normal CLI authentication."""

    allowed = {"PATH", "HOME", "CODEX_HOME", "LANG", "LC_ALL", "LC_CTYPE", "TERM"}
    return {key: value for key, value in os.environ.items() if key in allowed}


class CodexProvider:
    """Run Codex as a non-agentic, isolated TrialGPT generation backend."""

    def __init__(
        self,
        *,
        codex_executable: str = "codex",
        version: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        request_timeout_seconds: int = CODEX_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, int)
            or request_timeout_seconds < 1
        ):
            raise ValueError("request_timeout_seconds must be a positive integer")
        self.codex_executable = codex_executable
        self._version = version
        self._runner = runner or subprocess.run
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.request_timeout_seconds = request_timeout_seconds
        self.provider_id = "codex-cli-v1"
        self._bundled_model_entries: dict[str, dict[str, object]] = {}

    def codex_version(self) -> str:
        if self._version is None:
            try:
                completed = self._runner(
                    [self.codex_executable, "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                    env=_minimal_environment(),
                    timeout=self.request_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexTransportError(
                    f"Codex version lookup timed out after {self.request_timeout_seconds} seconds"
                ) from exc
            if completed.returncode != 0:
                raise CodexTransportError(
                    f"unable to determine Codex version (exit {completed.returncode})"
                )
            self._version = completed.stdout.strip() or completed.stderr.strip() or "unknown"
        return self._version

    def provenance(self) -> dict[str, object]:
        """Return reproducibility settings without exposing the subprocess environment."""

        return {
            "provider_id": self.provider_id,
            "codex_executable": self.codex_executable,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "request_timeout_seconds": self.request_timeout_seconds,
            "codex_version": self.codex_version(),
            "model_provenance": (
                self._catalog_model_provenance(self.model) if self.model is not None else None
            ),
        }

    def _bundled_model_entry(self, model: str) -> dict[str, object]:
        cached = self._bundled_model_entries.get(model)
        if cached is not None:
            return cached
        try:
            completed = self._runner(
                [self.codex_executable, "debug", "models", "--bundled"],
                capture_output=True,
                text=True,
                check=False,
                env=_minimal_environment(),
                timeout=self.request_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodexTransportError(
                f"Codex model catalog lookup timed out after {self.request_timeout_seconds} seconds"
            ) from exc
        if completed.returncode != 0:
            raise CodexProviderError(
                "Codex returned no reported model identity and its bundled model catalog "
                f"could not be read (exit {completed.returncode})"
            )
        try:
            catalog = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CodexProviderError(
                "Codex returned no reported model identity and its bundled model catalog "
                "was not valid JSON"
            ) from exc
        raw_models = catalog.get("models") if isinstance(catalog, Mapping) else None
        matches = (
            [
                entry
                for entry in raw_models
                if isinstance(entry, Mapping) and entry.get("slug") == model
            ]
            if isinstance(raw_models, list)
            else []
        )
        if len(matches) != 1:
            raise CodexProviderError(
                "Codex returned no reported model identity and the requested model "
                f"{model!r} did not resolve exactly once in the bundled model catalog"
            )
        entry = dict(matches[0])
        self._bundled_model_entries[model] = entry
        return entry

    def _catalog_model_provenance(self, model: str) -> dict[str, object]:
        entry = self._bundled_model_entry(model)
        catalog_entry_sha256 = schema_hash(entry)
        core = {
            "catalog_entry_sha256": catalog_entry_sha256,
            "codex_version": self.codex_version(),
            "effective_model": model,
            "provider_reported_model": None,
            "requested_model": model,
            "resolution_source": "codex_cli_bundled_catalog",
        }
        return {
            "status": "catalog_pinned_provider_revision_unavailable",
            "requested_model": model,
            "provider_reported_model": None,
            "effective_model": model,
            "resolution_source": "codex_cli_bundled_catalog",
            "catalog_entry_sha256": catalog_entry_sha256,
            "codex_version": self.codex_version(),
            "provider_revision_available": False,
            "identity_sha256": schema_hash(core),
        }

    def _model_provenance(
        self, requested_model: str | None, reported_model: str | None
    ) -> dict[str, object]:
        if reported_model is None:
            if requested_model is None:
                raise CodexProviderError(
                    "Codex returned no reported model identity and no model was "
                    "explicitly requested"
                )
            return self._catalog_model_provenance(requested_model)
        if requested_model is not None and not _model_identity_matches(
            requested_model, reported_model
        ):
            raise CodexProviderError(
                f"provider-reported model {reported_model!r} does not match requested model "
                f"{requested_model!r}"
            )
        core = {
            "catalog_entry_sha256": None,
            "codex_version": self.codex_version(),
            "effective_model": reported_model,
            "provider_reported_model": reported_model,
            "requested_model": requested_model,
            "resolution_source": "provider_event",
        }
        return {
            "status": "provider_reported",
            "requested_model": requested_model,
            "provider_reported_model": reported_model,
            "effective_model": reported_model,
            "resolution_source": "provider_event",
            "catalog_entry_sha256": None,
            "codex_version": self.codex_version(),
            "provider_revision_available": True,
            "identity_sha256": schema_hash(core),
        }

    def generate_request(self, request: CodexRequest) -> CodexResult:
        """Execute one request and preserve its complete event/usage provenance."""

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="taim-codex-") as temporary:
            workdir = Path(temporary)
            schema_path = workdir / "output-schema.json"
            schema_path.write_text(
                json.dumps(request.output_schema, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
            command = build_codex_command(
                request,
                schema_path=schema_path,
                workdir=workdir,
                codex_executable=self.codex_executable,
            )
            try:
                completed = self._runner(
                    command,
                    input=request.prompt,
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=workdir,
                    env=_minimal_environment(),
                    timeout=self.request_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexTransportError(
                    f"Codex generation timed out after {self.request_timeout_seconds} seconds"
                ) from exc
        raw = completed.stdout + (completed.stderr or "")
        events = _parse_events(raw)
        if completed.returncode != 0:
            raise CodexTransportError(
                "Codex generation failed with exit "
                f"{completed.returncode}; detail_sha256={sha256_text(raw)}",
                failure_kind=_transport_failure_kind(raw),
            )
        output_texts = [text for event in events if (text := _event_text(event))]
        if not output_texts:
            raise CodexProviderError("Codex returned no final output event")
        reported_model = _reported_model(events)
        model_provenance = self._model_provenance(request.model, reported_model)
        return CodexResult(
            text=output_texts[-1],
            requested_model=request.model,
            reported_model=reported_model,
            codex_version=self.codex_version(),
            stage=request.stage,
            prompt_sha256=request.prompt_sha256,
            output_schema_sha256=request.output_schema_sha256,
            request_identity=request.identity,
            logical_call_id=request.logical_call_id,
            input_hash=request.input_hash,
            reasoning_effort=request.reasoning_effort,
            raw_event_stream=raw,
            events=events,
            usage=_usage(events),
            model_identity_matches=_model_identity_matches(request.model, reported_model),
            model_provenance=model_provenance,
            latency_seconds=time.perf_counter() - started,
        )

    def generate(
        self,
        request: CodexRequest | None = None,
        *,
        prompt: str | None = None,
        output_schema: Mapping[str, object] | None = None,
        attempt: int = 1,
        logical_call_id: str = "",
        stage: str = "trialgpt-generation",
        input_hash: str | None = None,
    ) -> CodexResult | ProviderResponse:
        """Support both the low-level transport and Issue #13's provider protocol."""

        if request is not None:
            return self.generate_request(request)
        if prompt is None or output_schema is None:
            raise TypeError("prompt and output_schema are required")
        codex_result = self.generate_request(
            CodexRequest(
                stage=stage,
                prompt=prompt,
                output_schema=output_schema,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                input_hash=input_hash,
                logical_call_id=logical_call_id,
            )
        )
        return ProviderResponse(
            raw_output=codex_result.text,
            usage=codex_result.usage or {},
            metadata={
                "codex": codex_result.as_dict(),
                "attempt": attempt,
            },
        )


__all__ = [
    "CODEX_REQUEST_TIMEOUT_SECONDS",
    "CodexProvider",
    "CodexProviderError",
    "CodexRequest",
    "CodexResult",
    "CodexTransportError",
    "build_codex_command",
    "schema_hash",
    "sha256_text",
]
