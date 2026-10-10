from __future__ import annotations

import subprocess
from pathlib import Path


def git(cwd: Path, *arguments: str) -> str:
    """Run Git for a test repository and return stripped stdout."""

    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
