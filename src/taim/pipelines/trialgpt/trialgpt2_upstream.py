"""TrialGPT 2.0's pinned code: verify the checkout, then run it unchanged as a subprocess.

TAIM never edits their code and does not redistribute it. A run names a checkout of
``TrialGPT2/TrialGPT2.0`` at the pinned commit; it must be clean, and every file the scorer
executes must hash as declared. Their runner then runs in TAIM's own interpreter, which carries
their two requirements, ``openai`` and ``tqdm``, through the release's lock.

The one thing TAIM runs in its own process is their prompt module, which uses only the standard
library: its ``build_trial_matcher_messages`` tells the bridge, before any call, the exact prompt
each candidate pair produces.

The runner's environment is built from nothing, so no variable of the operator's shell can change
their client: their own ``LLM_PROVIDER``, ``OPENAI_BASE_URL``, ``OPENAI_API_KEY`` and
``LLM_TIMEOUT_S`` route every call to the bridge, and every other setting keeps their default.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import cast

from taim.artifacts import sha256_file
from taim.executables import ExecutableNotFoundError, resolve_executable
from taim.model_codex import CODEX_REQUEST_TIMEOUT_SECONDS, sha256_text
from taim.pipelines.trialgpt.trialgpt2_convert import PromptBuilder
from taim.pipelines.trialgpt.trialgpt2_declaration import (
    MODEL_LABEL_SENT,
    UPSTREAM_COMMIT,
    UPSTREAM_EXECUTED_FILES,
    UPSTREAM_PROMPT_MODULE,
    UPSTREAM_REPOSITORY,
    UPSTREAM_RUNNER_MODULE,
    UPSTREAM_SYSTEM_PROMPT_SHA256,
    UPSTREAM_TREE,
)

#: Their client's timeout, above the bridge's Codex timeout, so a slow call ends at the bridge.
UPSTREAM_TIMEOUT_SECONDS = CODEX_REQUEST_TIMEOUT_SECONDS + 60
#: Their client refuses to start without a key; the bridge reads none.
LOOPBACK_API_KEY = "taim-loopback-bridge"
UPSTREAM_NOTE_FIELD = "full_note"
UPSTREAM_REQUIREMENTS = ("openai", "tqdm")
_INHERITED_VARIABLES = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SYSTEMROOT")


class TrialGPT2UpstreamError(ValueError):
    """Raised when the pinned TrialGPT 2.0 code cannot be verified or run."""


@dataclass(frozen=True, slots=True)
class UpstreamCheckout:
    """A verified checkout of TrialGPT 2.0 at the pin."""

    path: Path
    commit: str
    tree: str
    executed_files: Mapping[str, str]

    def identity(self) -> dict[str, object]:
        """What the System Input binds: the pin and the bytes the scorer executes."""

        return {
            "repository": UPSTREAM_REPOSITORY,
            "commit": self.commit,
            "tree": self.tree,
            "executed_files": dict(sorted(self.executed_files.items())),
        }


def _git(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - resolved executable and fixed git argv
            [resolve_executable("git"), *arguments],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
            # The checkout is a blob:none partial clone: a read must never fetch a blob.
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
    except (OSError, ExecutableNotFoundError, subprocess.CalledProcessError) as exc:
        raise TrialGPT2UpstreamError(
            f"cannot inspect the TrialGPT 2.0 checkout at {checkout}"
        ) from exc
    return completed.stdout.strip()


def verify_upstream_checkout(
    checkout: Path,
    *,
    commit: str = UPSTREAM_COMMIT,
    tree: str = UPSTREAM_TREE,
    executed_files: Mapping[str, str] = UPSTREAM_EXECUTED_FILES,
) -> UpstreamCheckout:
    """Refuse a checkout that is not the pin, clean, with every executed file as declared."""

    checkout = checkout.resolve()
    actual_commit = _git(checkout, "rev-parse", "--verify", "HEAD")
    if actual_commit != commit:
        raise TrialGPT2UpstreamError(
            f"TrialGPT 2.0 checkout is at {actual_commit}, not the pinned {commit}"
        )
    actual_tree = _git(checkout, "rev-parse", "HEAD^{tree}")
    if actual_tree != tree:
        raise TrialGPT2UpstreamError(f"TrialGPT 2.0 checkout tree is {actual_tree}, not {tree}")
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise TrialGPT2UpstreamError(f"TrialGPT 2.0 checkout has local changes: {checkout}")
    actual_files: dict[str, str] = {}
    for relative, digest in executed_files.items():
        path = checkout / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise TrialGPT2UpstreamError(f"TrialGPT 2.0 file {relative} is missing or changed")
        actual_files[relative] = digest
    return UpstreamCheckout(
        path=checkout,
        commit=actual_commit,
        tree=actual_tree,
        executed_files=MappingProxyType(actual_files),
    )


def load_prompt_builder(
    checkout: UpstreamCheckout,
    *,
    system_prompt_sha256: str = UPSTREAM_SYSTEM_PROMPT_SHA256,
) -> PromptBuilder:
    """Their ``build_trial_matcher_messages``, from the verified checkout's prompt module."""

    path = checkout.path / UPSTREAM_PROMPT_MODULE
    # Compiled here rather than imported, so no bytecode cache is written into their checkout.
    namespace: dict[str, object] = {"__name__": "taim_trialgpt2_upstream_prompt"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)  # noqa: S102 - their verified prompt module
    system_prompt = namespace.get("SYSTEM_PROMPT")
    builder = namespace.get("build_trial_matcher_messages")
    if not isinstance(system_prompt, str) or sha256_text(system_prompt) != system_prompt_sha256:
        raise TrialGPT2UpstreamError("TrialGPT 2.0 SYSTEM_PROMPT is not the declared prompt")
    if not callable(builder):
        raise TrialGPT2UpstreamError("TrialGPT 2.0 prompt module has no message builder")
    return cast(PromptBuilder, builder)


def upstream_runtime() -> dict[str, str]:
    """The versions of their two requirements in this interpreter, which runs their code."""

    versions: dict[str, str] = {}
    for name in UPSTREAM_REQUIREMENTS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise TrialGPT2UpstreamError(
                f"TrialGPT 2.0 needs {name} in this environment; install the release's "
                "trialgpt2 extra"
            ) from exc
    return versions


def upstream_arguments(generation_workers: int | str) -> list[str]:
    """Their runner's options, without the run's paths.

    ``--resume`` lets an interrupted run continue in place, and ``--continue_on_error`` is
    deliberately absent: with it, their runner drops a failed pair from the ranking.
    """

    return [
        "--note_field",
        UPSTREAM_NOTE_FIELD,
        "--model",
        MODEL_LABEL_SENT,
        "--nprocs",
        str(generation_workers),
        "--resume",
    ]


def scorer_environment(base_url: str) -> dict[str, str]:
    """The runner's whole environment: the routing to the bridge, and nothing of the caller's."""

    environment = {name: os.environ[name] for name in _INHERITED_VARIABLES if name in os.environ}
    environment.update(upstream_settings())
    environment["OPENAI_BASE_URL"] = base_url
    return environment


def upstream_settings() -> dict[str, str]:
    """The variables TAIM sets for their client, other than the bridge's per-run address."""

    return {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": LOOPBACK_API_KEY,
        "LLM_TIMEOUT_S": str(UPSTREAM_TIMEOUT_SECONDS),
        # Their checkout must stay clean, so their modules write no bytecode into it.
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }


def run_upstream(
    python: str,
    checkout: UpstreamCheckout,
    *,
    cases: Path,
    trials: Path,
    output: Path,
    generation_workers: int,
    base_url: str,
    log: Path,
) -> int:
    """Run their scorer over the cases; its exit status. Output and errors go to ``log``."""

    command = [
        python,
        "-m",
        UPSTREAM_RUNNER_MODULE,
        "--cases_path",
        str(cases),
        "--trial_dir",
        str(trials),
        "--out_dir",
        str(output),
        *upstream_arguments(generation_workers),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        completed = subprocess.run(  # noqa: S603 - this interpreter and their fixed module
            command,
            cwd=checkout.path,
            env=scorer_environment(base_url),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


__all__ = [
    "LOOPBACK_API_KEY",
    "UPSTREAM_NOTE_FIELD",
    "UPSTREAM_REQUIREMENTS",
    "UPSTREAM_TIMEOUT_SECONDS",
    "TrialGPT2UpstreamError",
    "UpstreamCheckout",
    "load_prompt_builder",
    "run_upstream",
    "scorer_environment",
    "upstream_arguments",
    "upstream_runtime",
    "upstream_settings",
    "verify_upstream_checkout",
]
