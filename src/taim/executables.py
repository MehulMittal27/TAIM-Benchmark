"""Resolve external executables to absolute paths before running them."""

from __future__ import annotations

import shutil
from functools import cache

__all__ = ["ExecutableNotFoundError", "resolve_executable"]


class ExecutableNotFoundError(RuntimeError):
    """Raised when a required external executable is not on ``PATH``."""


@cache
def resolve_executable(name: str) -> str:
    """Return the absolute path of ``name``, or raise if it is unavailable.

    Passing a bare program name to :mod:`subprocess` defers lookup to ``PATH``
    at execution time, so whichever match happens to come first wins.  Callers
    resolve once here instead, which pins the executable and turns a missing
    tool into an explicit error rather than an opaque ``FileNotFoundError``.
    """

    if not name or "/" in name:
        raise ValueError("resolve_executable expects a bare program name")
    resolved = shutil.which(name)
    if resolved is None:
        raise ExecutableNotFoundError(f"{name} was not found on PATH")
    return resolved
