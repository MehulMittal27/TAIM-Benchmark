"""OpenCode run JSON-event transport, qualified against CLI 1.18.32 semantics.

Uses an isolated temporary home and explicit runtime-injected provider environment. The CLI
has no schema flag and does not report model identity in its JSON stream. CLI permissions are
not a security boundary; use this for separately authorized synthetic experiments.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import selectors
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from taim.generation import GenerationRequest, ProviderResponse
from taim.model_http import ModelTransportError, strict_json, token_usage


def _run(
    command: list[str],
    *,
    content: bytes,
    workdir: Path,
    environment: dict[str, str],
    deadline: float,
    max_output_bytes: int,
) -> bytes:
    """Drain both pipes within one byte/deadline budget and reap the process group."""
    if os.name != "posix":
        raise ModelTransportError("unsupported_runtime")
    if deadline <= time.monotonic():
        raise ModelTransportError("model_timeout")
    with tempfile.TemporaryFile() as source, selectors.DefaultSelector() as selector:
        source.write(content)
        source.seek(0)
        try:
            process = subprocess.Popen(  # noqa: S603 - explicit executable, no shell
                command,
                stdin=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
                env=environment,
                start_new_session=True,
            )
        except (OSError, ValueError):
            raise ModelTransportError("model_unavailable") from None
        output = bytearray()
        size = 0
        try:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    selector.register(pipe, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModelTransportError("model_timeout")
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fd, min(65536, max_output_bytes - size + 1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    size += len(chunk)
                    if size > max_output_bytes:
                        raise ModelTransportError("model_response_too_large")
                    if key.fileobj is process.stdout:
                        output.extend(chunk)
            try:
                code = process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise ModelTransportError("model_timeout") from None
            if code != 0:
                raise ModelTransportError("model_request_failed")
            return bytes(output)
        finally:
            # Also remove descendants left after a successfully exiting parent.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()


@dataclass
class OpenCodeProvider:
    model: str
    executable: str
    variant: str | None
    timeout_seconds: int
    max_request_bytes: int
    max_response_bytes: int
    system_prompt: str = ""
    environment: Callable[[], Mapping[str, str]] | None = None
    deadline: float | None = None
    provider_id: str = "opencode-cli-v1"

    def with_deadline(self, deadline: float) -> OpenCodeProvider:
        return replace(self, deadline=deadline)

    def generate(
        self, request: GenerationRequest, *, prompt: str, attempt: int, logical_call_id: str
    ) -> ProviderResponse:
        structured = request.output_schema is not None
        content = (
            prompt
            + "\n\nReturn only JSON matching this output schema:\n"
            + json.dumps(request.output_schema, ensure_ascii=False, allow_nan=False)
            if structured
            else prompt
        ).encode()
        if len(content) + len(self.system_prompt.encode("utf-8")) > self.max_request_bytes:
            raise ModelTransportError("model_request_too_large")
        deadline = (
            self.deadline if self.deadline is not None else time.monotonic() + self.timeout_seconds
        )
        if deadline <= time.monotonic():
            raise ModelTransportError("model_timeout")
        with tempfile.TemporaryDirectory(prefix="taim-opencode-") as temporary:
            workdir = Path(temporary)
            try:
                supplied = dict(self.environment()) if self.environment else {}
            except Exception:
                raise ModelTransportError("model_unavailable", authentication=True) from None
            # Provider credentials can be injected, never configuration/search-path overrides.
            if any(
                not key.endswith(("_API_KEY", "_AUTH_TOKEN"))
                or (key.startswith(("OPENCODE_", "XDG_")) and key != "OPENCODE_API_KEY")
                or not isinstance(value, str)
                for key, value in supplied.items()
            ):
                raise ModelTransportError("invalid_runtime_environment")
            config = {
                "share": "disabled",
                "autoupdate": False,
                "permission": {"*": "deny"},
                "agent": {
                    "taim-generation": {
                        "mode": "primary",
                        "permission": {"*": "deny"},
                        "prompt": self.system_prompt
                        or (
                            "Generate structured JSON using only the supplied input."
                            if structured
                            else "Answer using only the supplied input."
                        ),
                    }
                },
            }
            environment = {
                "PATH": os.environ.get("PATH", os.defpath),
                "HOME": temporary,
                "XDG_CONFIG_HOME": str(workdir / "config"),
                "XDG_DATA_HOME": str(workdir / "data"),
                "XDG_CACHE_HOME": str(workdir / "cache"),
                "XDG_STATE_HOME": str(workdir / "state"),
                "OPENCODE_CONFIG_CONTENT": json.dumps(config),
                **supplied,
            }
            raw_version = _run(
                [self.executable, "--version"],
                content=b"",
                workdir=workdir,
                environment=environment,
                deadline=deadline,
                max_output_bytes=4096,
            )
            try:
                version = raw_version.decode().strip()
                if len(version) > 128 or not re.fullmatch(
                    r"\d+\.\d+\.\d+(?:[-+][\w.-]+)?", version
                ):
                    raise ValueError("invalid version")
            except (ValueError, UnicodeError):
                raise ModelTransportError("invalid_model_response") from None
            command = [
                self.executable,
                "run",
                "--format",
                "json",
                "--model",
                self.model,
                "--agent",
                "taim-generation",
                "--pure",
                "--title",
                "TAIM generation",
                "--dir",
                temporary,
            ]
            if self.variant is not None:
                command.extend(["--variant", self.variant])
            raw = _run(
                command,
                content=content,
                workdir=workdir,
                environment=environment,
                deadline=deadline,
                max_output_bytes=self.max_response_bytes,
            )
        try:
            events = [strict_json(line) for line in raw.splitlines() if line.strip()]
            texts: dict[str, str] = {}
            finished = False
            session: str | None = None
            message: str | None = None
            usage: dict[str, object] = {}
            for event in events:
                if not isinstance(event, dict):
                    raise ValueError("invalid event")
                kind = event.get("type")
                if kind not in {"step_start", "text", "step_finish"}:
                    raise ValueError("unexpected event or model action")
                part = event["part"]
                if not isinstance(part, dict):
                    raise ValueError("invalid part")
                event_session, event_message = event.get("sessionID"), part.get("messageID")
                if not isinstance(event_session, str) or not isinstance(event_message, str):
                    raise ValueError("missing event identity")
                session = session or event_session
                message = message or event_message
                if event_session != session or event_message != message or finished:
                    raise ValueError("mixed or repeated completion")
                if kind == "text":
                    part_id, text = part["id"], part["text"]
                    if not isinstance(part_id, str) or not isinstance(text, str):
                        raise ValueError("invalid text")
                    texts[part_id] = text
                if kind == "step_finish":
                    if part.get("reason") != "stop":
                        raise ValueError("incomplete response")
                    finished = True
                    usage = token_usage(part.get("tokens"))
            if not finished or not texts:
                raise ValueError("missing completion")
            content_text = "".join(texts.values())
            if structured and not isinstance(strict_json(content_text), dict):
                raise ValueError("expected JSON object")
        except (KeyError, TypeError, ValueError, RecursionError):
            raise ModelTransportError("invalid_model_response") from None
        return ProviderResponse(
            raw_output=content_text,
            usage=usage,
            metadata={
                "trace_version": "model-generation-v1",
                "attempt": attempt,
                "provider_call": {"status": "called"},
                "opencode": {
                    "requested_model": self.model,
                    "reported_model": None,
                    "model_identity_matches": None,
                    "opencode_version": version,
                    "model_identity_status": "requested_only_cli_does_not_report_model",
                    "provider_revision_available": False,
                    "logical_call_id": logical_call_id,
                    "schema_enforcement": "prompt_and_local_validator"
                    if structured
                    else "free_text",
                },
            },
        )
