# ruff: noqa: S101
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from taim.executables import ExecutableNotFoundError, resolve_executable


def _make_executable(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_resolve_executable_returns_an_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _make_executable(tmp_path, "taim-fake-tool")
    monkeypatch.setenv("PATH", str(tmp_path))
    resolve_executable.cache_clear()

    resolved = resolve_executable("taim-fake-tool")

    assert Path(resolved).is_absolute()
    assert Path(resolved) == expected


def test_resolve_executable_pins_the_first_path_entry_not_a_later_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred_directory = tmp_path / "preferred"
    shadow_directory = tmp_path / "shadow"
    preferred_directory.mkdir()
    shadow_directory.mkdir()
    preferred = _make_executable(preferred_directory, "taim-fake-tool")
    _make_executable(shadow_directory, "taim-fake-tool")
    monkeypatch.setenv("PATH", os.pathsep.join([str(preferred_directory), str(shadow_directory)]))
    resolve_executable.cache_clear()

    assert Path(resolve_executable("taim-fake-tool")) == preferred


def test_resolve_executable_reports_a_missing_tool_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    resolve_executable.cache_clear()

    with pytest.raises(ExecutableNotFoundError, match="taim-absent-tool was not found on PATH"):
        resolve_executable("taim-absent-tool")


@pytest.mark.parametrize("name", ["", "/usr/bin/git", "dir/git"])
def test_resolve_executable_rejects_anything_but_a_bare_program_name(name: str) -> None:
    with pytest.raises(ValueError, match="bare program name"):
        resolve_executable(name)


def test_resolve_executable_finds_git_on_this_machine() -> None:
    resolve_executable.cache_clear()

    assert Path(resolve_executable("git")).is_absolute()
