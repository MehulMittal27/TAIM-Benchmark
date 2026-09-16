"""Deterministic, fail-closed construction of a public Benchmark Release."""

from __future__ import annotations

import ast
import base64
import configparser
import csv
import fnmatch
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from email.message import Message
from email.parser import Parser
from email.utils import formataddr
from pathlib import Path, PurePosixPath
from typing import cast

from taim.contracts import content_sha256
from taim.pipeline_extensions import LITERAL_FIELDS
from taim.release_support import capability_key, validate_support_matrix

RELEASE_DEFINITION_VERSION = "3.0"
RELEASE_MANIFEST_VERSION = "3.0"
RELEASE_PROVENANCE_FILENAME = "RELEASE-PROVENANCE.json"
RELEASE_PROVENANCE_VERSION = "2.0"

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
_EXPECTED_SCHEMA_VERSIONS = {
    "benchmark_snapshot": "2.0",
    "run_manifest": "7.0",
    "published_result_bundle": "1.0",
}
_DEFINITION_FIELDS = {
    "artifact_type",
    "schema_version",
    "release_version",
    "release_approval",
    "support_matrix",
    "project",
    "license_identities",
    "license_bindings",
    "published_pipelines",
    "pipelines",
    "schema_versions",
    "files",
    "generated_files",
    "published_result_bundles",
    "quick_start",
}
_MANIFEST_FIELDS = _DEFINITION_FIELDS | {
    "source_commit",
    "expected_public_tree_id",
    "manifest_id",
}
_PROJECT_FIELDS = {
    "repository_name",
    "repository_url",
    "distribution_name",
    "package_version",
    "package_summary",
    "authors",
    "package_requires_dist",
    "package_provides_extra",
    "requires_python",
    "tested_python_versions",
    "license_expression",
    "license_file",
    "readme_file",
    "citation_file",
    "contributing_file",
    "security_file",
    "third_party_notices_file",
    "public_cli",
    "release_commit",
}
_RELEASE_COMMIT_FIELDS = {
    "branch",
    "message",
    "author_name",
    "author_email",
    "timestamp",
    "previous_public_commit",
}
_APPROVAL_FIELDS = {"status", "reviewer", "approved_at"}
_FILE_DEFINITION_FIELDS = {
    "source",
    "destination",
    "license_id",
    "redistribution",
    "data_classification",
    "owners",
}
_PIPELINE_FIELDS = {"components"}
_FILE_MANIFEST_FIELDS = _FILE_DEFINITION_FIELDS | {"sha256", "byte_size"}
_LICENSE_FIELDS = {
    "license_id",
    "subject",
    "kind",
    "expression",
    "notice_file",
    "status",
}
_LICENSE_BINDING_FIELDS = {"kind", "locator", "license_id"}
_LICENSE_BINDING_KINDS = {
    "model",
    "python-distribution",
    "release-asset-file",
    "release-project-file",
    "runtime-input",
    "workflow-action",
}
_BUNDLE_FIELDS = {"bundle_id", "destination", "schema_version", "package_artifact_id"}
_PROHIBITED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".faiss",
    ".index",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
}
_PROHIBITED_FILENAMES = {
    ".env",
    "credentials.json",
    "provider-trace.jsonl",
    "qrels.jsonl",
    "trace.jsonl",
}
_INSPECTABLE_TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".pyi",
    ".toml",
    ".tsv",
    ".txt",
    ".typed",
    ".yaml",
    ".yml",
}
_INSPECTABLE_TEXT_FILENAMES = {".gitignore", "LICENSE", "SHA256SUMS"}
_TREC_QRELS_ROW = re.compile(r"^\S+\s+\S+\s+\S+\s+[0-2]\s*$")


class BenchmarkReleaseError(ValueError):
    """A Benchmark Release is not safe, closed, or reproducible."""


@dataclass(frozen=True, slots=True)
class BenchmarkReleaseCandidate:
    """A validated release tree with no Git history; ``taim release chain`` makes its commit."""

    directory: Path
    manifest_id: str
    public_tree_id: str


@dataclass(frozen=True, slots=True)
class BenchmarkReleasePackageArtifact:
    """A distribution archive checked against one frozen release manifest."""

    release_manifest_id: str
    public_tree_id: str
    package_artifact_id: str
    package_files: tuple[tuple[str, str, int], ...]

    def release_identity(self) -> dict[str, str]:
        return {
            "release_manifest_id": self.release_manifest_id,
            "public_tree_id": self.public_tree_id,
            "package_artifact_id": self.package_artifact_id,
        }


def _json_object(path: Path, *, role: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkReleaseError(f"{role} is not readable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkReleaseError(f"{role} must be a JSON object")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _exact_fields(value: Mapping[str, object], expected: set[str], *, role: str) -> None:
    missing = expected - value.keys()
    unexpected = value.keys() - expected
    if missing:
        raise BenchmarkReleaseError(f"{role} is missing fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise BenchmarkReleaseError(
            f"{role} has unexpected fields: {', '.join(sorted(unexpected))}"
        )


def _string(value: object, *, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkReleaseError(f"{role} must be a non-empty string")
    return value


def _release_path(value: object, *, role: str) -> str:
    text = _string(value, role=role)
    if "\\" in text:
        raise BenchmarkReleaseError(f"{role} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", "..", ".git"} for part in path.parts):
        raise BenchmarkReleaseError(f"{role} must be a safe relative path")
    return path.as_posix()


def _sha256_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(
    root: Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise BenchmarkReleaseError("git is required to construct a Benchmark Release")
    process = subprocess.run(  # noqa: S603
        [executable, *arguments],
        cwd=root,
        env=None if environment is None else {**os.environ, **environment},
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "git command failed"
        raise BenchmarkReleaseError(detail)
    return process.stdout.strip()


def _clean_source_commit(source_root: Path) -> str:
    root = source_root.resolve()
    if not root.is_dir():
        raise BenchmarkReleaseError(f"source root is not a directory: {root}")
    top_level = Path(_git(root, ("rev-parse", "--show-toplevel"))).resolve()
    if top_level != root:
        raise BenchmarkReleaseError("source root must be the Git repository root")
    commit = _git(root, ("rev-parse", "--verify", "HEAD"))
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise BenchmarkReleaseError("source commit must be an exact 40-character Git SHA")
    if _git(root, ("status", "--porcelain", "--untracked-files=all")):
        raise BenchmarkReleaseError("Benchmark Release requires a clean source worktree")
    return commit


def _generate_support_catalogue(source: bytes, definition: Mapping[str, object]) -> bytes:
    """The packaged support catalogue with only the rows of the published pipelines."""

    try:
        payload = json.loads(source)
        rows = cast(list[Mapping[str, object]], payload["support_matrix"])
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BenchmarkReleaseError("support catalogue source is not a readable catalogue") from exc
    published = set(cast(list[str], definition["published_pipelines"]))
    kept = [row for row in rows if row["system"] in published]
    if len(kept) == len(rows):
        return source
    return (json.dumps({**payload, "support_matrix": kept}, indent=2) + "\n").encode("utf-8")


def _generate_external_system_dependencies(
    source: bytes, definition: Mapping[str, object]
) -> bytes:
    """The external System dependency declaration with only the published Systems' entries.

    An entry without a string System ID is kept, so the declaration's own validation refuses it.
    """

    try:
        payload = json.loads(source)
        systems = cast(list[object], payload["systems"])
        kept = [
            system
            for system in systems
            if not isinstance(system, Mapping)
            or not isinstance(system.get("system_id"), str)
            or system["system_id"] in cast(list[str], definition["published_pipelines"])
        ]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BenchmarkReleaseError(
            "external System dependency declaration is not a readable declaration"
        ) from exc
    if len(kept) == len(systems):
        return source
    return (json.dumps({**payload, "systems": kept}, indent=2) + "\n").encode("utf-8")


_TOML_TABLE = re.compile(r"^\[([^\]]+)\]\s*$")
_TOML_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
_PACKAGE_DATA_ROOT = "src/taim/"


def _is_omitted(path: str, shipped: Collection[str]) -> bool:
    """Whether a literal path names a file the release does not ship; a glob never does."""

    return not set("*?[") & set(path) and path not in shipped


def _pyproject_without_omitted_entries(text: str, shipped: Collection[str]) -> str:
    """Remove the package-data elements and ruff per-file-ignore keys that name omitted files.

    The edit is line by line, because the approved file is hand-formatted and tomllib cannot write.
    It reaches only a single-line package-data array; the verification refuses any other shape.
    """

    table: str | None = None
    kept_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        header = _TOML_TABLE.match(line)
        if header is not None:
            table = header.group(1)
        elif (
            table == "tool.setuptools.package-data"
            and line.startswith("taim = [")
            and line.rstrip().endswith("]")
        ):
            elements = _TOML_STRING.findall(line)
            kept = [
                item for item in elements if not _is_omitted(_PACKAGE_DATA_ROOT + item, shipped)
            ]
            if kept != elements:
                ending = line[len(line.rstrip("\r\n")) :]
                kept_lines.append(
                    "taim = [" + ", ".join(f'"{item}"' for item in kept) + "]" + ending
                )
                continue
        elif table == "tool.ruff.lint.per-file-ignores":
            key = _TOML_STRING.match(line)
            if key is not None and _is_omitted(key.group(1), shipped):
                continue
        kept_lines.append(line)
    return "".join(kept_lines)


def _refuse_project_entries_for_omitted_files(text: str, shipped: Collection[str]) -> None:
    """Refuse a project file whose package data or ruff keys still name an omitted file."""

    try:
        tool = tomllib.loads(text).get("tool", {})
    except tomllib.TOMLDecodeError as exc:
        raise BenchmarkReleaseError("generated project file is not valid TOML") from exc
    package_data = tool.get("setuptools", {}).get("package-data", {}).get("taim", [])
    ignores = tool.get("ruff", {}).get("lint", {}).get("per-file-ignores", {})
    named = [_PACKAGE_DATA_ROOT + item for item in package_data] + list(ignores)
    remaining = sorted({path for path in named if _is_omitted(path, shipped)})
    if remaining:
        raise BenchmarkReleaseError(
            "project file still names files the release does not ship: " + ", ".join(remaining)
        )


def _generate_pyproject(source: bytes, definition: Mapping[str, object]) -> bytes:
    """The project file without the package data or ruff keys of files the release omits."""

    shipped = {
        cast(str, entry["destination"])
        for entry in cast(list[Mapping[str, object]], definition["files"])
    }
    text = source.decode("utf-8")
    generated = _pyproject_without_omitted_entries(text, shipped)
    _refuse_project_entries_for_omitted_files(generated, shipped)
    return source if generated == text else generated.encode("utf-8")


_SECTION_OPEN = re.compile(r"<!-- pipeline-section: ([^,<>]+(?:, [^,<>]+)*) -->")
_SECTION_CLOSE = "<!-- /pipeline-section -->"
_SECTION_WORD = "pipeline-section"
_ROW_MARKER = re.compile(r"(\|.*\|) <!-- pipeline-row: ([^,<>]+(?:, [^,<>]+)*) -->")
_ROW_WORD = "pipeline-row"


def _markdown_sections(text: str) -> list[tuple[int, int, tuple[str, ...]]]:
    """The pipeline sections of a Markdown text: opening line, closing line (1-based), owners.

    A section opens with ``<!-- pipeline-section: <owner>[, <owner>...] -->`` alone on a line
    and closes with ``<!-- /pipeline-section -->``. Any other line naming ``pipeline-section``
    is refused.
    """

    sections: list[tuple[int, int, tuple[str, ...]]] = []
    opened: tuple[int, tuple[str, ...]] | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        if _SECTION_WORD not in line:
            continue
        opening = _SECTION_OPEN.fullmatch(line)
        if opening is not None:
            if opened is not None:
                raise BenchmarkReleaseError(
                    f"pipeline section opened at line {number} inside the one opened at line "
                    f"{opened[0]}"
                )
            owners = tuple(opening.group(1).split(", "))
            if list(owners) != sorted(set(owners)):
                raise BenchmarkReleaseError(
                    f"pipeline section owners must be sorted and unique at line {number}"
                )
            opened = (number, owners)
        elif line == _SECTION_CLOSE:
            if opened is None:
                raise BenchmarkReleaseError(
                    f"pipeline section closed at line {number} was never opened"
                )
            sections.append((opened[0], number, opened[1]))
            opened = None
        else:
            raise BenchmarkReleaseError(f"malformed pipeline section marker at line {number}")
    if opened is not None:
        raise BenchmarkReleaseError(f"pipeline section opened at line {opened[0]} is never closed")
    return sections


def _markdown_row_markers(text: str) -> list[tuple[int, tuple[str, ...]]]:
    """The marked table rows of a Markdown text: line (1-based) and owners.

    A marked row ends with the extra cell ``<!-- pipeline-row: <owner>[, <owner>...] -->``. A
    table ignores cells beyond its header's count, so the marker renders nowhere, while a marker
    line inside a table would end the table. Any other line naming ``pipeline-row`` is refused, as
    is a marked row without two table lines above it: a header or delimiter row, whose loss breaks
    the table, is never marked.
    """

    rows: list[tuple[int, tuple[str, ...]]] = []
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        if _ROW_WORD not in line:
            continue
        marker = _ROW_MARKER.fullmatch(line)
        if marker is None:
            raise BenchmarkReleaseError(f"malformed pipeline row marker at line {number}")
        if number < 3 or not all(lines[number - above].startswith("|") for above in (2, 3)):
            raise BenchmarkReleaseError(
                f"pipeline row marker at line {number} is not on a table body row"
            )
        owners = tuple(marker.group(2).split(", "))
        if list(owners) != sorted(set(owners)):
            raise BenchmarkReleaseError(
                f"pipeline row owners must be sorted and unique at line {number}"
            )
        rows.append((number, owners))
    return rows


def _markdown_without_unpublished_sections(text: str, published: Collection[str]) -> str:
    """Drop every section none of whose owners is published, and every marker from what stays.

    Markers belong in sources. A kept section loses its two marker lines and a kept table row loses
    its marker cell, so the emitted document names no pipeline through a marker.
    """

    lines = text.splitlines(keepends=True)
    dropped: set[int] = set()
    for first, last, owners in _markdown_sections(text):
        if not set(owners) & set(published):
            dropped.update(range(first - 1, last))
        else:
            dropped.update((first - 1, last - 1))
    for number, row_owners in _markdown_row_markers(text):
        if not set(row_owners) & set(published):
            dropped.add(number - 1)
        else:
            line = lines[number - 1]
            body = line.rstrip("\r\n")
            row = cast(re.Match[str], _ROW_MARKER.fullmatch(body)).group(1)
            lines[number - 1] = row + line[len(body) :]
    # A run of dropped lines between blank lines takes the blank line after it, so adjacent
    # sections dropped together leave no doubled blank line.
    for index in sorted(dropped):
        after = index + 1
        if after in dropped or after >= len(lines) or lines[after].strip():
            continue
        start = index
        while start - 1 in dropped:
            start -= 1
        if start > 0 and not lines[start - 1].strip():
            dropped.add(after)
    return "".join(line for index, line in enumerate(lines) if index not in dropped)


def _refuse_emitted_markdown_markers(text: str) -> None:
    """Refuse generated Markdown that still carries a pipeline section or row marker."""

    sections = _markdown_sections(text)
    rows = _markdown_row_markers(text)
    if sections or rows:
        lines = sorted({first for first, _last, _owners in sections} | {row for row, _ in rows})
        raise BenchmarkReleaseError(
            "generated Markdown still carries pipeline markers at lines "
            + ", ".join(str(line) for line in lines)
        )


def _refuse_sections_of_unpublished_pipelines(text: str, published: Collection[str]) -> None:
    """Refuse generated Markdown that still holds a section with no published owner."""

    remaining = sorted(
        {
            owner
            for _first, _last, owners in _markdown_sections(text)
            if not set(owners) & set(published)
            for owner in owners
        }
    )
    if remaining:
        raise BenchmarkReleaseError(
            "generated Markdown still holds sections of unpublished pipelines: "
            + ", ".join(remaining)
        )
    remaining_rows = sorted(
        {
            owner
            for _number, owners in _markdown_row_markers(text)
            if not set(owners) & set(published)
            for owner in owners
        }
    )
    if remaining_rows:
        raise BenchmarkReleaseError(
            "generated Markdown still holds table rows of unpublished pipelines: "
            + ", ".join(remaining_rows)
        )


def _generate_markdown_sections(source: bytes, definition: Mapping[str, object]) -> bytes:
    """The Markdown file without the sections of pipelines the release does not publish."""

    published = set(cast(list[str], definition["published_pipelines"]))
    text = source.decode("utf-8")
    generated = _markdown_without_unpublished_sections(text, published)
    _refuse_sections_of_unpublished_pipelines(generated, published)
    _refuse_emitted_markdown_markers(generated)
    return source if generated == text else generated.encode("utf-8")


def _validate_markdown_section_owners(
    files: Mapping[str, Path], definition: Mapping[str, object]
) -> None:
    """Refuse a section owner the unprojected definition does not declare; a typo drops nothing."""

    declared = set(cast(Mapping[str, object], definition["pipelines"]))
    for destination, generator in cast(Mapping[str, str], definition["generated_files"]).items():
        if generator != "markdown-sections":
            continue
        text = files[destination].read_text(encoding="utf-8")
        for first, _last, owners in _markdown_sections(text):
            undeclared = sorted(set(owners) - declared)
            if undeclared:
                raise BenchmarkReleaseError(
                    f"{destination} line {first}: pipeline section names undeclared pipelines: "
                    + ", ".join(undeclared)
                )
        for number, row_owners in _markdown_row_markers(text):
            undeclared_row = sorted(set(row_owners) - declared)
            if undeclared_row:
                raise BenchmarkReleaseError(
                    f"{destination} line {number}: pipeline row names undeclared pipelines: "
                    + ", ".join(undeclared_row)
                )


_SHIPS_WITH = re.compile(r"(?P<code>.*\S)\s+# ships-with: (?P<path>[^\s#]+)")
_SHIPS_WITH_WORD = "ships-with"


def _python_ships_with_markers(text: str) -> list[tuple[int, str]]:
    """The lines of a Python text that ship only with one file: line (1-based) and that file.

    A marker is the trailing comment ``# ships-with: <destination>`` on a line of code. Any other
    line naming ``ships-with`` is refused.
    """

    markers: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _SHIPS_WITH_WORD not in line:
            continue
        marker = _SHIPS_WITH.fullmatch(line)
        if marker is None:
            raise BenchmarkReleaseError(f"malformed ships-with marker at line {number}")
        markers.append((number, marker.group("path")))
    return markers


def _python_without_unshipped_lines(text: str, shipped: Collection[str]) -> str:
    """Drop each marked statement or list item whose file the release does not ship.

    A marker on the first line of a module-level statement stands for the whole statement; one on
    a single-line list item, a line of code ending in a comma, stands for that line alone.
    """

    markers = _python_ships_with_markers(text)
    if not markers:
        return text
    try:
        module = ast.parse(text)
    except SyntaxError as exc:
        raise BenchmarkReleaseError(
            "a Python source with ships-with markers does not parse"
        ) from exc
    statements = {node.lineno: cast(int, node.end_lineno) for node in module.body}
    lines = text.splitlines(keepends=True)
    dropped: set[int] = set()
    for number, path in markers:
        last = statements.get(number)
        code = cast(re.Match[str], _SHIPS_WITH.fullmatch(lines[number - 1].rstrip("\r\n")))
        if last is None and not code.group("code").endswith(","):
            raise BenchmarkReleaseError(
                f"ships-with marker at line {number} is on neither the first line of a "
                "module-level statement nor a single-line list item"
            )
        if path not in shipped:
            dropped.update(range(number - 1, last or number))
        else:
            ending = lines[number - 1][len(lines[number - 1].rstrip("\r\n")) :]
            lines[number - 1] = code.group("code") + ending
    return "".join(line for index, line in enumerate(lines) if index not in dropped)


def _generate_python_sections(source: bytes, definition: Mapping[str, object]) -> bytes:
    """The Python module without the lines that ship only with files the release does not ship."""

    shipped = {
        cast(str, entry["destination"])
        for entry in cast(list[Mapping[str, object]], definition["files"])
    }
    text = source.decode("utf-8")
    generated = _python_without_unshipped_lines(text, shipped)
    remaining = sorted(
        {path for _number, path in _python_ships_with_markers(generated) if path not in shipped}
    )
    if remaining:
        raise BenchmarkReleaseError(
            "generated Python still holds lines that ship only with files the release does not "
            "ship: " + ", ".join(remaining)
        )
    left = _python_ships_with_markers(generated)
    if left:
        raise BenchmarkReleaseError(
            "generated Python still carries ships-with markers at lines "
            + ", ".join(str(number) for number, _path in left)
        )
    return source if generated == text else generated.encode("utf-8")


def _validate_python_ships_with_files(
    files: Mapping[str, Path], definition: Mapping[str, object]
) -> None:
    """Refuse a ships-with marker naming a file the unprojected definition does not allowlist.

    Such a line would drop from every release without a word, so it is refused; an unmarked import
    of a file no release ships is refused by the public import closure instead.
    """

    allowlisted = {
        cast(str, entry["destination"])
        for entry in cast(list[Mapping[str, object]], definition["files"])
    }
    for destination, generator in cast(Mapping[str, str], definition["generated_files"]).items():
        if generator != "python-sections":
            continue
        text = files[destination].read_text(encoding="utf-8")
        for number, path in _python_ships_with_markers(text):
            if path not in allowlisted:
                raise BenchmarkReleaseError(
                    f"{destination} line {number}: ships-with names a file the definition does "
                    f"not allowlist: {path}"
                )


_CI_PYTHON = "uv run python -c "
_WORKFLOW_JOB_OPEN = re.compile(r'    "(?P<name>[^"]+)": \{')
_WORKFLOW = ".github/workflows/ci.yml"


def _ci_command_imports(command: str, *, role: str) -> set[str]:
    """The taim modules one ``uv run python -c`` command imports.

    Each import must name its module (``import taim.m`` or ``from taim.m import n``): a module a CI
    command imports is a file the release must ship, and ``from taim import n`` names no file.
    """

    if not command.startswith(_CI_PYTHON):
        return set()
    try:
        code = ast.parse(shlex.split(command)[4])
    except (ValueError, IndexError, SyntaxError) as exc:
        raise BenchmarkReleaseError(
            f"public CI {role} job has an unreadable python command"
        ) from exc
    modules: set[str] = set()
    for node in ast.walk(code):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("taim."))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "taim":
            if node.level or node.module == "taim":
                raise BenchmarkReleaseError(
                    f"public CI {role} job imports from the taim package without naming a module"
                )
            modules.add(cast(str, node.module))
    return modules


def _ci_job_imports(job: object, *, role: str) -> set[str]:
    return {
        module
        for command in _run_commands(_job_steps(job, role=role))
        for module in _ci_command_imports(command, role=role)
    }


def _ci_job_syncs(job: object, *, role: str) -> list[set[str]]:
    """The extras each ``uv sync --locked`` command in one job installs."""

    syncs: list[set[str]] = []
    for command in _run_commands(_job_steps(job, role=role)):
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise BenchmarkReleaseError(f"public CI {role} job has an unreadable command") from exc
        if tokens[:3] == ["uv", "sync", "--locked"]:
            syncs.append(
                {tokens[index + 1] for index, word in enumerate(tokens[:-1]) if word == "--extra"}
            )
    return syncs


def _module_ships(module: str, destinations: Collection[str]) -> bool:
    relative = "src/" + module.replace(".", "/")
    return f"{relative}.py" in destinations or f"{relative}/__init__.py" in destinations


def _generate_workflow_jobs(source: bytes, definition: Mapping[str, object]) -> bytes:
    """The CI workflow without each job that imports a taim module the release does not ship.

    A job's own imports name the files it needs, as a ships-with marker does, so a job ships exactly
    while every module it imports ships. The removal is textual, keeping the reviewed layout, and
    the result must parse equal to the source without those jobs.
    """

    destinations = {
        cast(str, entry["destination"])
        for entry in cast(list[Mapping[str, object]], definition["files"])
    }
    text = source.decode("utf-8")
    payload = cast(dict[str, object], json.loads(text))
    jobs = cast(Mapping[str, object], payload["jobs"])
    dropped = [
        name
        for name, job in jobs.items()
        if not all(
            _module_ships(module, destinations) for module in _ci_job_imports(job, role=name)
        )
    ]
    if not dropped:
        return source
    lines = text.splitlines(keepends=True)
    for name in dropped:
        start = next(
            (
                index
                for index, line in enumerate(lines)
                if (opened := _WORKFLOW_JOB_OPEN.fullmatch(line.rstrip("\n"))) is not None
                and opened["name"] == name
            ),
            None,
        )
        end = (
            None
            if start is None
            else next(
                (
                    index
                    for index in range(start, len(lines))
                    if lines[index].rstrip("\n") in {"    },", "    }"}
                ),
                None,
            )
        )
        if start is None or end is None:
            raise BenchmarkReleaseError(f"generated workflow cannot locate job {name}")
        last = lines[end].rstrip("\n") == "    }"
        del lines[start : end + 1]
        if last and start > 0 and lines[start - 1].rstrip("\n") == "    },":
            lines[start - 1] = "    }\n"
    generated = "".join(lines)
    kept = {name: job for name, job in jobs.items() if name not in dropped}
    try:
        emitted = json.loads(generated)
    except json.JSONDecodeError as exc:
        raise BenchmarkReleaseError("generated workflow is not valid JSON") from exc
    if emitted != {**payload, "jobs": kept} or list(emitted["jobs"]) != list(kept):
        raise BenchmarkReleaseError(
            "generated workflow is not the source without jobs " + ", ".join(sorted(dropped))
        )
    return generated.encode("utf-8")


# Generators make a shipped file from its committed source and the projected definition. Markers
# belong in sources: each generator strips its markers from what it emits, in every selection, and
# otherwise returns the source bytes unchanged when the release holds nothing back.
_GENERATORS: Mapping[str, Callable[[bytes, Mapping[str, object]], bytes]] = {
    "external-system-dependencies": _generate_external_system_dependencies,
    "markdown-sections": _generate_markdown_sections,
    "pyproject": _generate_pyproject,
    "python-sections": _generate_python_sections,
    "support-catalogue": _generate_support_catalogue,
    "workflow-jobs": _generate_workflow_jobs,
}


def _shipped_bytes(definition: Mapping[str, object], destination: str, source: bytes) -> bytes:
    """The bytes a release ships for one allowlisted file: its source, or its generator's output."""

    generator = cast(Mapping[str, str], definition["generated_files"]).get(destination)
    return source if generator is None else _GENERATORS[generator](source, definition)


def _validate_pipelines(
    definition: Mapping[str, object],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Validate the declared pipelines and the published selection; return both."""

    raw_pipelines = definition["pipelines"]
    if not isinstance(raw_pipelines, Mapping) or not raw_pipelines:
        raise BenchmarkReleaseError("pipelines must be a non-empty object")
    if list(raw_pipelines) != sorted(raw_pipelines):
        raise BenchmarkReleaseError("pipelines must be sorted by System ID")
    components: dict[str, tuple[str, ...]] = {}
    for system_id, raw_pipeline in raw_pipelines.items():
        role = f"pipeline {system_id}"
        if not isinstance(raw_pipeline, Mapping):
            raise BenchmarkReleaseError(f"{role} must be an object")
        _exact_fields(raw_pipeline, _PIPELINE_FIELDS, role=role)
        declared = raw_pipeline["components"]
        if (
            not isinstance(declared, list)
            or not all(isinstance(item, str) and item for item in declared)
            or declared != sorted(set(declared))
        ):
            raise BenchmarkReleaseError(f"{role} components must be sorted unique System IDs")
        if system_id in declared or not set(declared) <= set(raw_pipelines):
            raise BenchmarkReleaseError(f"{role} components must be other declared pipelines")
        components[cast(str, system_id)] = tuple(declared)
    published = definition["published_pipelines"]
    if (
        not isinstance(published, list)
        or not published
        or not all(isinstance(item, str) and item for item in published)
        or published != sorted(set(published))
    ):
        raise BenchmarkReleaseError("published_pipelines must be sorted unique System IDs")
    undeclared = sorted(set(published) - set(components))
    if undeclared:
        raise BenchmarkReleaseError(
            "published pipelines are not declared: " + ", ".join(undeclared)
        )
    unpublished_components = sorted(
        {component for system_id in published for component in components[system_id]}
        - set(published)
    )
    if unpublished_components:
        raise BenchmarkReleaseError(
            "published pipelines need components that are not published: "
            + ", ".join(unpublished_components)
        )
    return components, tuple(published)


def _validate_definition(definition: Mapping[str, object], *, projected: bool = True) -> None:
    """Validate a release definition, in its private or its projected form.

    The private definition declares every public-eligible pipeline and marks the files each owns.
    The projected form, which a manifest carries, names only the published pipelines and the files
    they or no pipeline own. Every other rule applies to both.
    """

    _exact_fields(definition, _DEFINITION_FIELDS, role="release definition")
    if (
        definition["artifact_type"] != "taim-benchmark-release-definition"
        or definition["schema_version"] != RELEASE_DEFINITION_VERSION
    ):
        raise BenchmarkReleaseError("unsupported Benchmark Release definition")
    _string(definition["release_version"], role="release_version")
    approval = definition["release_approval"]
    if not isinstance(approval, Mapping):
        raise BenchmarkReleaseError("release_approval must be an object")
    _exact_fields(approval, _APPROVAL_FIELDS, role="release_approval")
    if approval["status"] != "approved":
        raise BenchmarkReleaseError("repository identity and project license are not approved")
    _string(approval["reviewer"], role="release_approval.reviewer")
    _aware_datetime(approval["approved_at"], role="release_approval.approved_at")

    try:
        support_matrix = validate_support_matrix(definition["support_matrix"])
    except ValueError as exc:
        raise BenchmarkReleaseError(f"invalid release support matrix: {exc}") from exc

    project = definition["project"]
    if not isinstance(project, Mapping):
        raise BenchmarkReleaseError("project must be an object")
    _validate_project(project)

    licenses = definition["license_identities"]
    if not isinstance(licenses, list) or not licenses:
        raise BenchmarkReleaseError("license_identities must be a non-empty array")
    license_ids: set[str] = set()
    license_kinds: dict[str, object] = {}
    for index, raw_license in enumerate(licenses):
        if not isinstance(raw_license, Mapping):
            raise BenchmarkReleaseError(f"license identity {index} must be an object")
        _exact_fields(raw_license, _LICENSE_FIELDS, role=f"license identity {index}")
        license_id = _string(raw_license["license_id"], role="license_id")
        _string(raw_license["subject"], role=f"license {license_id} subject")
        if raw_license["kind"] not in {"project", "dependency", "model", "asset"}:
            raise BenchmarkReleaseError(f"license {license_id} has an invalid subject kind")
        expression = _string(raw_license["expression"], role=f"license {license_id} expression")
        if expression.casefold() in {
            "unknown",
            "unresolved",
            "tbd",
            "noassertion",
            "licenseref-requires-human-review",
        }:
            raise BenchmarkReleaseError(f"license {license_id} is unresolved")
        if raw_license["status"] != "approved":
            raise BenchmarkReleaseError(f"license {license_id} is not approved")
        _release_path(raw_license["notice_file"], role=f"license {license_id} notice_file")
        if license_id in license_ids:
            raise BenchmarkReleaseError("license identities must be unique")
        license_ids.add(license_id)
        license_kinds[license_id] = raw_license["kind"]
    if project["license_expression"] not in {
        raw["expression"] for raw in cast(list[Mapping[str, object]], licenses)
    }:
        raise BenchmarkReleaseError("project license lacks an approved license identity")

    bindings = definition["license_bindings"]
    if not isinstance(bindings, list):
        raise BenchmarkReleaseError("license_bindings must be an array")
    binding_keys: list[tuple[str, str]] = []
    for index, raw_binding in enumerate(bindings):
        if not isinstance(raw_binding, Mapping):
            raise BenchmarkReleaseError(f"license binding {index} must be an object")
        _exact_fields(raw_binding, _LICENSE_BINDING_FIELDS, role=f"license binding {index}")
        kind = _string(raw_binding["kind"], role=f"license binding {index} kind")
        locator = _string(raw_binding["locator"], role=f"license binding {index} locator")
        if kind not in _LICENSE_BINDING_KINDS:
            raise BenchmarkReleaseError(f"license binding {index} has an invalid kind")
        if raw_binding["license_id"] not in license_ids:
            raise BenchmarkReleaseError(f"license binding {index} has no approved license")
        expected_license_kind = {
            "model": "model",
            "python-distribution": "dependency",
            "release-asset-file": "asset",
            "release-project-file": "project",
            "runtime-input": "asset",
            "workflow-action": "dependency",
        }[kind]
        if license_kinds[cast(str, raw_binding["license_id"])] != expected_license_kind:
            raise BenchmarkReleaseError(
                f"license binding {index} does not use a {expected_license_kind} identity"
            )
        binding_keys.append((kind, locator))
    if binding_keys != sorted(binding_keys) or len(binding_keys) != len(set(binding_keys)):
        raise BenchmarkReleaseError("license bindings must be sorted and unique")

    pipelines, published = _validate_pipelines(definition)
    matrix_systems = {capability_key(item)[3] for item in support_matrix}
    if projected:
        if set(pipelines) != set(published):
            raise BenchmarkReleaseError("a projected release declares only its published pipelines")
        if matrix_systems != set(published):
            raise BenchmarkReleaseError("support matrix and published pipelines do not match")
    elif not set(published) <= matrix_systems <= set(pipelines):
        raise BenchmarkReleaseError(
            "support matrix must cover every published pipeline and name only declared ones"
        )
    schemas = definition["schema_versions"]
    if not isinstance(schemas, Mapping) or dict(schemas) != _EXPECTED_SCHEMA_VERSIONS:
        raise BenchmarkReleaseError("schema_versions do not match the public artifact contracts")

    raw_files = definition["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise BenchmarkReleaseError("files must be a non-empty allowlist")
    destinations: list[str] = []
    sources: list[str] = []
    classifications = {
        "release_source",
        "synthetic_fixture",
        "published_result_bundle",
        "license_notice",
    }
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, Mapping):
            raise BenchmarkReleaseError(f"release file {index} must be an object")
        expected = _FILE_MANIFEST_FIELDS if "sha256" in raw_file else _FILE_DEFINITION_FIELDS
        _exact_fields(raw_file, expected, role=f"release file {index}")
        sources.append(_release_path(raw_file["source"], role=f"release file {index} source"))
        destinations.append(
            _release_path(raw_file["destination"], role=f"release file {index} destination")
        )
        if raw_file["license_id"] not in license_ids:
            raise BenchmarkReleaseError(f"release file {index} has no approved license")
        if raw_file["redistribution"] != "approved":
            raise BenchmarkReleaseError(f"release file {index} is not approved for redistribution")
        if raw_file["data_classification"] not in classifications:
            raise BenchmarkReleaseError(f"release file {index} has an invalid classification")
        file_owners = raw_file["owners"]
        if (
            not isinstance(file_owners, list)
            or not all(isinstance(owner, str) and owner for owner in file_owners)
            or file_owners != sorted(set(file_owners))
        ):
            raise BenchmarkReleaseError(
                f"release file {index} owners must be sorted unique System IDs"
            )
        if not set(file_owners) <= set(pipelines):
            raise BenchmarkReleaseError(f"release file {index} is owned by an undeclared pipeline")
        if expected == _FILE_MANIFEST_FIELDS:
            if (
                not isinstance(raw_file["sha256"], str)
                or _SHA256.fullmatch(cast(str, raw_file["sha256"])) is None
            ):
                raise BenchmarkReleaseError(f"release file {index} has an invalid SHA-256")
            if (
                isinstance(raw_file["byte_size"], bool)
                or not isinstance(raw_file["byte_size"], int)
                or cast(int, raw_file["byte_size"]) < 0
            ):
                raise BenchmarkReleaseError(f"release file {index} has an invalid byte size")
    if destinations != sorted(destinations):
        raise BenchmarkReleaseError("release files must be sorted by destination")
    if len(destinations) != len(set(destinations)) or len(sources) != len(set(sources)):
        raise BenchmarkReleaseError("release source and destination paths must be unique")
    if RELEASE_PROVENANCE_FILENAME in destinations:
        raise BenchmarkReleaseError("release provenance is generated, not copied")

    destination_set = set(destinations)
    generated = definition["generated_files"]
    if not isinstance(generated, Mapping) or list(generated) != sorted(generated):
        raise BenchmarkReleaseError("generated_files must be an object sorted by destination")
    for generated_destination, generator in generated.items():
        if generated_destination not in destination_set:
            raise BenchmarkReleaseError(
                f"generated file is not an allowlisted destination: {generated_destination}"
            )
        if generator not in _GENERATORS:
            raise BenchmarkReleaseError(
                f"generated file {generated_destination} names an unknown generator"
            )
    if (
        any(
            capability["profile"] == "trec-ct-2021-external-fidelity-26149"
            for capability in support_matrix
        )
        and "docs/paper-analysis-protocol-2026-08-31-v4.md" not in destination_set
    ):
        raise BenchmarkReleaseError(
            "external-fidelity support requires the human-approved v4 protocol"
        )
    missing_capability_files = {
        path
        for capability in support_matrix
        for path in cast(list[str], capability["required_files"])
        if path not in destination_set
    }
    if missing_capability_files:
        raise BenchmarkReleaseError(
            "support matrix requires absent projected files: "
            + ", ".join(sorted(missing_capability_files))
        )
    file_owners_by_destination = {
        destination: set(cast(list[str], raw_file["owners"]))
        for raw_file, destination in zip(
            cast(list[Mapping[str, object]], raw_files), destinations, strict=True
        )
    }
    published_set = set(published)
    held_back_requirements = {
        f"{capability['system']} requires {path}"
        for capability in support_matrix
        if capability["system"] in published_set
        for path in cast(list[str], capability["required_files"])
        if (path_owners := file_owners_by_destination[path]) and not path_owners & published_set
    }
    if held_back_requirements:
        raise BenchmarkReleaseError(
            "published support rows require files only unpublished pipelines own: "
            + ", ".join(sorted(held_back_requirements))
        )
    required_project_files = {
        cast(str, project[field])
        for field in (
            "license_file",
            "readme_file",
            "citation_file",
            "contributing_file",
            "security_file",
            "third_party_notices_file",
        )
    } | {".github/workflows/ci.yml", "pyproject.toml", "uv.lock"}
    missing_project_files = required_project_files - destination_set
    if missing_project_files:
        raise BenchmarkReleaseError(
            "release omits required project files: " + ", ".join(sorted(missing_project_files))
        )
    missing_notices = {
        cast(str, license_identity["notice_file"])
        for license_identity in cast(list[Mapping[str, object]], licenses)
    } - destination_set
    if missing_notices:
        raise BenchmarkReleaseError(
            "release omits approved license notices: " + ", ".join(sorted(missing_notices))
        )

    bundles = definition["published_result_bundles"]
    if not isinstance(bundles, list):
        raise BenchmarkReleaseError("published_result_bundles must be an array")
    bundle_ids: set[str] = set()
    bundle_destinations: list[str] = []
    for index, bundle in enumerate(bundles):
        if not isinstance(bundle, Mapping):
            raise BenchmarkReleaseError(f"Published Result Bundle {index} must be an object")
        _exact_fields(bundle, _BUNDLE_FIELDS, role=f"Published Result Bundle {index}")
        bundle_id = _string(bundle["bundle_id"], role="Published Result Bundle ID")
        destination = _release_path(bundle["destination"], role="bundle destination")
        _string(bundle["schema_version"], role="bundle schema_version")
        if (
            not isinstance(bundle["package_artifact_id"], str)
            or _SHA256.fullmatch(cast(str, bundle["package_artifact_id"])) is None
        ):
            raise BenchmarkReleaseError("Published Result Bundle package_artifact_id is invalid")
        if bundle_id in bundle_ids:
            raise BenchmarkReleaseError("Published Result Bundle IDs must be unique")
        if not any(
            path == destination or path.startswith(destination + "/") for path in destinations
        ):
            raise BenchmarkReleaseError(f"Published Result Bundle {bundle_id} has no release files")
        bundle_ids.add(bundle_id)
        bundle_destinations.append(destination)
    for raw_file, destination in zip(
        cast(list[Mapping[str, object]], raw_files), destinations, strict=True
    ):
        if raw_file["data_classification"] != "published_result_bundle":
            continue
        owners = [
            bundle_destination
            for bundle_destination in bundle_destinations
            if destination == bundle_destination or destination.startswith(bundle_destination + "/")
        ]
        if len(owners) != 1:
            raise BenchmarkReleaseError(
                "every Published Result Bundle file must belong to exactly one declared bundle"
            )

    quick_start = definition["quick_start"]
    if (
        not isinstance(quick_start, list)
        or not quick_start
        or any(
            not isinstance(command, list)
            or not command
            or not all(isinstance(token, str) and token for token in command)
            for command in quick_start
        )
    ):
        raise BenchmarkReleaseError("quick_start must be a non-empty array of argv arrays")
    if any(command[0] != project["public_cli"] for command in cast(list[list[str]], quick_start)):
        raise BenchmarkReleaseError("every quick_start command must use the declared public CLI")


def _validate_project(project: Mapping[str, object]) -> None:
    _exact_fields(project, _PROJECT_FIELDS, role="project")
    for name in _PROJECT_FIELDS - {
        "authors",
        "release_commit",
        "package_provides_extra",
        "package_requires_dist",
        "tested_python_versions",
    }:
        if name.endswith("_file"):
            _release_path(project[name], role=f"project.{name}")
        else:
            _string(project[name], role=f"project.{name}")
    versions = project["tested_python_versions"]
    if (
        not isinstance(versions, list)
        or not versions
        or versions != sorted(set(versions))
        or not all(isinstance(item, str) and re.fullmatch(r"3\.1[1-4]", item) for item in versions)
    ):
        raise BenchmarkReleaseError("tested_python_versions must be a sorted supported matrix")
    for field in ("package_requires_dist", "package_provides_extra"):
        values = project[field]
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise BenchmarkReleaseError(f"project.{field} must be sorted unique strings")
    authors = project["authors"]
    if not isinstance(authors, list) or not authors:
        raise BenchmarkReleaseError("project.authors must be a non-empty array")
    canonical_authors: set[tuple[str | None, str | None]] = set()
    for index, author in enumerate(authors):
        if not isinstance(author, Mapping):
            raise BenchmarkReleaseError(f"project author {index} must be an object")
        fields = set(author)
        if not fields or not fields <= {"name", "email"}:
            raise BenchmarkReleaseError(f"project author {index} has invalid fields")
        author_name = author.get("name")
        author_email = author.get("email")
        if author_name is not None:
            author_name = _string(author_name, role=f"project author {index} name")
        if author_email is not None:
            author_email = _string(author_email, role=f"project author {index} email")
        identity = (author_name, author_email)
        if identity in canonical_authors:
            raise BenchmarkReleaseError("project authors must be unique")
        canonical_authors.add(identity)
    release_commit = project["release_commit"]
    if not isinstance(release_commit, Mapping):
        raise BenchmarkReleaseError("project.release_commit must be an object")
    _exact_fields(release_commit, _RELEASE_COMMIT_FIELDS, role="project.release_commit")
    for name in _RELEASE_COMMIT_FIELDS - {"previous_public_commit"}:
        _string(release_commit[name], role=f"project.release_commit.{name}")
    if "\n" in cast(str, release_commit["message"]):
        raise BenchmarkReleaseError("project.release_commit.message must be one line")
    _aware_datetime(release_commit["timestamp"], role="release commit timestamp")
    previous = release_commit["previous_public_commit"]
    if previous is not None and (
        not isinstance(previous, str) or _GIT_COMMIT.fullmatch(previous) is None
    ):
        raise BenchmarkReleaseError(
            "project.release_commit.previous_public_commit must be null or a 40-character Git SHA"
        )


def _aware_datetime(value: object, *, role: str) -> datetime:
    text = _string(value, role=role)
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise BenchmarkReleaseError(f"{role} must be ISO 8601") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise BenchmarkReleaseError(f"{role} must include a UTC offset")
    return timestamp


def _definition_from_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    files = []
    for raw_file in cast(list[Mapping[str, object]], manifest["files"]):
        files.append({key: raw_file[key] for key in _FILE_DEFINITION_FIELDS})
    return {
        key: (
            "taim-benchmark-release-definition"
            if key == "artifact_type"
            else RELEASE_DEFINITION_VERSION
            if key == "schema_version"
            else files
            if key == "files"
            else manifest[key]
        )
        for key in _DEFINITION_FIELDS
    }


def load_benchmark_release_manifest(path: str | Path) -> dict[str, object]:
    manifest = _json_object(Path(path), role="Benchmark Release manifest")
    _exact_fields(manifest, _MANIFEST_FIELDS, role="Benchmark Release manifest")
    if (
        manifest["artifact_type"] != "taim-benchmark-release-manifest"
        or manifest["schema_version"] != RELEASE_MANIFEST_VERSION
    ):
        raise BenchmarkReleaseError("unsupported Benchmark Release manifest")
    definition = _definition_from_manifest(manifest)
    _validate_definition(definition)
    # Validate the manifest-only file fields through the same allowlist path.
    manifest_definition = {**definition, "files": manifest["files"]}
    _validate_definition(manifest_definition)
    if (
        not isinstance(manifest["source_commit"], str)
        or _GIT_COMMIT.fullmatch(cast(str, manifest["source_commit"])) is None
    ):
        raise BenchmarkReleaseError("Benchmark Release source_commit is invalid")
    if (
        not isinstance(manifest["expected_public_tree_id"], str)
        or _SHA256.fullmatch(cast(str, manifest["expected_public_tree_id"])) is None
    ):
        raise BenchmarkReleaseError("expected_public_tree_id is invalid")
    manifest_id = manifest["manifest_id"]
    core = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest_id != content_sha256(core):
        raise BenchmarkReleaseError("Benchmark Release manifest ID does not match its content")
    return manifest


def _source_file(source_root: Path, relative: str, *, source_commit: str) -> Path:
    root = source_root.resolve()
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise BenchmarkReleaseError(
            f"release source is missing or is not a regular file: {relative}"
        )
    try:
        object_type = _git(root, ("cat-file", "-t", f"{source_commit}:{relative}"))
    except BenchmarkReleaseError as exc:
        raise BenchmarkReleaseError(
            f"release source is not tracked by the pinned commit: {relative}"
        ) from exc
    if object_type != "blob":
        raise BenchmarkReleaseError(
            f"release source is not a regular blob at the pinned commit: {relative}"
        )
    return path


def _normalized_distribution(requirement: object, *, role: str) -> str:
    text = _string(requirement, role=role)
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", text)
    if match is None:
        raise BenchmarkReleaseError(f"{role} does not begin with a distribution name")
    return re.sub(r"[-_.]+", "-", match.group(0)).casefold()


def _metadata_requirement(requirement: object, *, extra: str | None) -> str:
    text = _string(requirement, role="public Python requirement")
    requirement_text, separator, raw_marker = text.partition(";")
    marker = ""
    if separator:
        marker_match = re.fullmatch(
            r"\s*python_version\s*(?P<operator><=|>=|==|!=|<|>)\s*"
            r"(?P<quote>['\"])(?P<version>[0-9]+(?:\.[0-9]+)*)\2\s*",
            raw_marker,
        )
        if marker_match is None:
            raise BenchmarkReleaseError(
                "public package requirement marker must be one static python_version comparison"
            )
        marker = (
            f'python_version {marker_match.group("operator")} "{marker_match.group("version")}"'
        )
    match = re.fullmatch(
        r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<spec>(?:(?:~=|==|!=|<=|>=|<|>)[^,;\s]+)(?:,(?:~=|==|!=|<=|>=|<|>)[^,;\s]+)*)?",
        requirement_text.strip(),
    )
    if match is None:
        raise BenchmarkReleaseError(
            "public package requirements must use name and version-specifier syntax"
        )
    name = re.sub(r"[-_.]+", "-", match.group("name")).casefold()
    raw_specifiers = match.group("spec")
    specifiers = ""
    if raw_specifiers:
        specifiers = ",".join(sorted(raw_specifiers.split(",")))
    rendered = name + specifiers
    markers = [value for value in (marker, f'extra == "{extra}"' if extra else "") if value]
    return rendered if not markers else f"{rendered}; {' and '.join(markers)}"


def _validate_package_project_metadata(
    files: Mapping[str, Path], project: Mapping[str, object]
) -> None:
    pyproject_path = files.get("pyproject.toml")
    if pyproject_path is None:
        raise BenchmarkReleaseError("package metadata closure requires pyproject.toml")
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise BenchmarkReleaseError("package metadata closure cannot read pyproject.toml") from exc
    metadata = pyproject.get("project")
    if not isinstance(metadata, Mapping):
        raise BenchmarkReleaseError("public pyproject.toml lacks project metadata")

    observed_name = metadata.get("name")
    expected_name = cast(str, project["distribution_name"])
    if (
        not isinstance(observed_name, str)
        or re.sub(r"[-_.]+", "-", observed_name).casefold()
        != re.sub(r"[-_.]+", "-", expected_name).casefold()
        or metadata.get("version") != project["package_version"]
        or metadata.get("description") != project["package_summary"]
        or metadata.get("authors") != project["authors"]
        or metadata.get("readme") != project["readme_file"]
        or metadata.get("requires-python") != project["requires_python"]
        or metadata.get("license") != project["license_expression"]
        or metadata.get("scripts") != {cast(str, project["public_cli"]): "taim.release_cli:main"}
        or metadata.get("urls") != {"Repository": project["repository_url"]}
    ):
        raise BenchmarkReleaseError(
            "public pyproject.toml does not match the reviewed project metadata"
        )
    if any(field in metadata for field in ("classifiers", "dynamic", "keywords", "maintainers")):
        raise BenchmarkReleaseError("public pyproject.toml contains undeclared project metadata")

    raw_dependencies = metadata.get("dependencies", [])
    raw_optional = metadata.get("optional-dependencies", {})
    if not isinstance(raw_dependencies, list) or not isinstance(raw_optional, Mapping):
        raise BenchmarkReleaseError("public pyproject.toml dependency metadata is invalid")
    dependencies = [
        _metadata_requirement(requirement, extra=None) for requirement in raw_dependencies
    ]
    extras: list[str] = []
    for raw_extra, requirements in raw_optional.items():
        if not isinstance(raw_extra, str) or not raw_extra or not isinstance(requirements, list):
            raise BenchmarkReleaseError("public pyproject.toml optional dependency is invalid")
        extras.append(raw_extra)
        dependencies.extend(
            _metadata_requirement(requirement, extra=raw_extra) for requirement in requirements
        )
    if (
        sorted(dependencies) != project["package_requires_dist"]
        or sorted(extras) != project["package_provides_extra"]
    ):
        raise BenchmarkReleaseError(
            "project package dependency metadata does not match public pyproject.toml"
        )


def _python_distributions(files: Mapping[str, Path], project: Mapping[str, object]) -> set[str]:
    pyproject_path = files.get("pyproject.toml")
    if pyproject_path is None:
        raise BenchmarkReleaseError("license closure requires pyproject.toml")
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise BenchmarkReleaseError("license closure cannot read pyproject.toml") from exc
    requirements: list[object] = []
    build_system = pyproject.get("build-system")
    if isinstance(build_system, Mapping):
        raw_build = build_system.get("requires", [])
        if isinstance(raw_build, list):
            requirements.extend(raw_build)
    metadata = pyproject.get("project")
    if isinstance(metadata, Mapping):
        raw_dependencies = metadata.get("dependencies", [])
        if isinstance(raw_dependencies, list):
            requirements.extend(raw_dependencies)
        optional = metadata.get("optional-dependencies", {})
        if isinstance(optional, Mapping):
            for group in optional.values():
                if isinstance(group, list):
                    requirements.extend(group)
    distributions = {
        _normalized_distribution(requirement, role="public Python requirement")
        for requirement in requirements
    }
    lock_path = files.get("uv.lock")
    if lock_path is None:
        raise BenchmarkReleaseError("license closure requires uv.lock")
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise BenchmarkReleaseError("license closure cannot read uv.lock") from exc
    packages = lock.get("package", [])
    if not isinstance(packages, list):
        raise BenchmarkReleaseError("public uv.lock package table is invalid")
    for index, package in enumerate(packages):
        if not isinstance(package, Mapping):
            raise BenchmarkReleaseError(f"public uv.lock package {index} is invalid")
        distributions.add(
            _normalized_distribution(package.get("name"), role=f"uv.lock package {index}")
        )
    project_distribution = re.sub(
        r"[-_.]+", "-", cast(str, project["distribution_name"])
    ).casefold()
    distributions.discard(project_distribution)
    return distributions


def _ci_jobs(
    files: Mapping[str, Path], definition: Mapping[str, object] | None = None
) -> Mapping[str, object]:
    workflow = files.get(_WORKFLOW)
    if workflow is None:
        raise BenchmarkReleaseError("release requires .github/workflows/ci.yml")
    try:
        source = workflow.read_bytes()
        # With a definition, the jobs are read as the release ships them.
        shipped = source if definition is None else _shipped_bytes(definition, _WORKFLOW, source)
        payload = json.loads(shipped.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkReleaseError(
            "public CI workflow must be JSON-formatted YAML for structural validation"
        ) from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("jobs"), Mapping):
        raise BenchmarkReleaseError("public CI workflow lacks a jobs object")
    return cast(Mapping[str, object], payload["jobs"])


def _job_steps(job: object, *, role: str) -> list[Mapping[str, object]]:
    if not isinstance(job, Mapping) or not isinstance(job.get("steps"), list):
        raise BenchmarkReleaseError(f"public CI {role} job lacks executable steps")
    steps = cast(list[object], job["steps"])
    if not all(isinstance(step, Mapping) for step in steps):
        raise BenchmarkReleaseError(f"public CI {role} job has an invalid step")
    condition = cast(Mapping[str, object], job).get("if", True)
    if condition is not True and not (
        isinstance(condition, str) and condition.strip().casefold() in {"true", "${{ true }}"}
    ):
        raise BenchmarkReleaseError(f"public CI {role} job is conditionally disabled")
    mapped_steps = cast(list[Mapping[str, object]], steps)
    for step in mapped_steps:
        condition = step.get("if", True)
        if condition is not True and not (
            isinstance(condition, str) and condition.strip().casefold() in {"true", "${{ true }}"}
        ):
            raise BenchmarkReleaseError(f"public CI {role} step is conditionally disabled")
    return mapped_steps


def _run_commands(steps: Sequence[Mapping[str, object]]) -> set[str]:
    return {
        line.strip()
        for step in steps
        if isinstance(step.get("run"), str)
        for line in cast(str, step["run"]).splitlines()
        if line.strip()
    }


def _local_action_steps(reference: str, files: Mapping[str, Path]) -> list[Mapping[str, object]]:
    directory = _release_path(reference.removeprefix("./"), role="local workflow action")
    candidates = [f"{directory}/action.yml", f"{directory}/action.yaml"]
    matches = [candidate for candidate in candidates if candidate in files]
    if len(matches) != 1:
        raise BenchmarkReleaseError(f"local workflow action is not closed: {reference}")
    try:
        action = json.loads(files[matches[0]].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkReleaseError(
            f"local workflow action must be JSON-formatted YAML: {reference}"
        ) from exc
    runs = action.get("runs") if isinstance(action, Mapping) else None
    steps = runs.get("steps") if isinstance(runs, Mapping) else None
    if runs is None or runs.get("using") != "composite" or not isinstance(steps, list):
        raise BenchmarkReleaseError(
            f"local workflow action must be an inspectable composite action: {reference}"
        )
    if not all(isinstance(step, Mapping) for step in steps):
        raise BenchmarkReleaseError(f"local workflow action has invalid steps: {reference}")
    return cast(list[Mapping[str, object]], steps)


def _workflow_actions(files: Mapping[str, Path]) -> set[str]:
    actions: set[str] = set()
    visited_local_actions: set[str] = set()

    def visit_steps(steps: Sequence[Mapping[str, object]]) -> None:
        for step in steps:
            reference = step.get("uses")
            if reference is None:
                continue
            if not isinstance(reference, str):
                raise BenchmarkReleaseError("workflow action reference must be a string")
            if reference.startswith("./"):
                if reference not in visited_local_actions:
                    visited_local_actions.add(reference)
                    visit_steps(_local_action_steps(reference, files))
                continue
            if reference.startswith("docker://"):
                if re.fullmatch(r"docker://[^@\s]+@sha256:[0-9a-f]{64}", reference) is None:
                    raise BenchmarkReleaseError(
                        f"workflow Docker action must use an exact image digest: {reference}"
                    )
                actions.add(reference)
                continue
            if "@" not in reference:
                raise BenchmarkReleaseError(f"workflow action is not pinned: {reference}")
            _repository, revision = reference.rsplit("@", 1)
            if _GIT_COMMIT.fullmatch(revision) is None:
                raise BenchmarkReleaseError(
                    f"workflow action must use an exact 40-character commit: {reference}"
                )
            actions.add(reference)

    for job_name, job in _ci_jobs(files).items():
        visit_steps(_job_steps(job, role=str(job_name)))
    return actions


def _validate_ci_contract(
    files: Mapping[str, Path], definition: Mapping[str, object] | None = None
) -> None:
    jobs = _ci_jobs(files, definition)
    verify = jobs.get("verify")
    if not isinstance(verify, Mapping):
        raise BenchmarkReleaseError("public CI lacks the verify job")
    strategy = verify.get("strategy")
    matrix = strategy.get("matrix") if isinstance(strategy, Mapping) else None
    versions = matrix.get("python-version") if isinstance(matrix, Mapping) else None
    if versions != ["3.11", "3.12", "3.13", "3.14"]:
        raise BenchmarkReleaseError("public CI verify job does not execute the Python matrix")
    verify_steps = _job_steps(verify, role="verify")
    verify_commands = _run_commands(verify_steps)
    matrix_setup = any(
        isinstance(step.get("with"), Mapping)
        and cast(Mapping[str, object], step["with"]).get("python-version")
        == "${{ matrix.python-version }}"
        for step in verify_steps
    ) or any(
        command.startswith("uv ") and "${{ matrix.python-version }}" in command
        for command in verify_commands
    )
    if not matrix_setup:
        raise BenchmarkReleaseError("public CI verify steps do not consume the Python matrix")
    required_verify = {
        "uv lock --check",
        "uv run mypy",
        "uv run pytest -q",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv sync --locked --extra dev",
    }
    missing_verify = sorted(required_verify - verify_commands)
    if missing_verify:
        raise BenchmarkReleaseError(
            "public CI verify job omits executable steps: " + ", ".join(missing_verify)
        )
    package_steps = _job_steps(jobs.get("package-smoke"), role="package-smoke")
    package_commands = _run_commands(package_steps)
    package_tokens = [command.split() for command in package_commands]
    has_sdist_install = any(
        tokens[:3] == ["uv", "pip", "install"] and "dist/*.tar.gz" in tokens
        for tokens in package_tokens
    )
    has_wheel_install = any(
        tokens[:3] == ["uv", "pip", "install"] and "dist/*.whl" in tokens
        for tokens in package_tokens
    )
    public_commands = [
        tokens
        for tokens in package_tokens
        if tokens and tokens[0].rsplit("/", 1)[-1] == "trial-benchmark"
    ]
    required_public_operations = {
        ("support", "list"),
        ("patient-to-trial", "fixture", "run"),
        ("patient-to-trial", "run", "evaluate"),
        ("patient-to-trial", "run", "validate"),
        ("trial-to-patient", "fixture", "run"),
        ("trial-to-patient", "run", "evaluate"),
        ("trial-to-patient", "run", "validate"),
    }
    public_operations = {
        operation
        for operation in required_public_operations
        if any(tuple(tokens[1 : 1 + len(operation)]) == operation for tokens in public_commands)
    }
    if (
        "uv build" not in package_commands
        or not has_sdist_install
        or not has_wheel_install
        or not required_public_operations <= public_operations
    ):
        raise BenchmarkReleaseError("public CI package-smoke job is incomplete")
    dense_commands = _run_commands(
        _job_steps(jobs.get("dense-import-smoke"), role="dense-import-smoke")
    )
    if "uv sync --locked --extra dense" not in dense_commands or not any(
        command.startswith("uv run python -c ") and "DENSE_ENCODER_REGISTRY" in command
        for command in dense_commands
    ):
        raise BenchmarkReleaseError("public CI dense-import-smoke job is incomplete")
    # No job may import a module the release leaves out: it would fail on the first push.
    coverage = {
        str(name): (_ci_job_syncs(job, role=str(name)), _ci_job_imports(job, role=str(name)))
        for name, job in jobs.items()
    }
    for name, (_syncs, imports) in coverage.items():
        unshipped = sorted(module for module in imports if not _module_ships(module, files))
        if unshipped:
            raise BenchmarkReleaseError(
                f"public CI {name} job imports taim modules the release does not ship: "
                + ", ".join(unshipped)
            )
    # Core needs a folding job; each shipped pipeline extension declares the jobs it needs.
    requirements: list[tuple[set[str], set[str], str]] = [
        ({"folding"}, {"taim.query_folding"}, "core")
    ]
    for relative, declaration in _extension_declarations(files).items():
        for requirement in cast(
            tuple[Mapping[str, tuple[str, ...]], ...], declaration.get("ci_jobs", ())
        ):
            requirements.append((set(requirement["extras"]), set(requirement["imports"]), relative))
    for extras, required_imports, owner in requirements:
        if not any(
            required_imports <= imports and any(extras <= sync for sync in syncs)
            for syncs, imports in coverage.values()
        ):
            raise BenchmarkReleaseError(
                f"public CI lacks a job syncing extras {', '.join(sorted(extras))} and importing "
                f"{', '.join(sorted(required_imports))} ({owner})"
            )


def _ast_static_value(node: ast.expr, constants: Mapping[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.Dict):
        result: dict[object, object] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                expanded = _ast_static_value(value, constants)
                if not isinstance(expanded, Mapping):
                    raise BenchmarkReleaseError("encoder policy expansion must be a static mapping")
                result.update(expanded)
            else:
                result[_ast_static_value(key, constants)] = _ast_static_value(value, constants)
        return result
    raise BenchmarkReleaseError("encoder policy references must resolve to static literals")


def _model_references(files: Mapping[str, Path], definition: Mapping[str, object]) -> set[str]:
    registry = files.get("src/taim/baselines/encoder_registry.py")
    if registry is None:
        return set()
    try:
        tree = ast.parse(registry.read_text(encoding="utf-8"), filename=str(registry))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise BenchmarkReleaseError("encoder registry is not inspectable Python source") from exc
    constants: dict[str, object] = {}
    for statement in tree.body:
        name: str | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            name = statement.target.id
            value = statement.value
        if name is None or value is None:
            continue
        try:
            constants[name] = _ast_static_value(value, constants)
        except BenchmarkReleaseError:
            continue
    models: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not (
            isinstance(node.func, ast.Name) and node.func.id == "EncoderPolicy"
        ):
            continue
        values: dict[str, object] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                resolved = _ast_static_value(keyword.value, constants)
                if not isinstance(resolved, Mapping) or not all(
                    isinstance(key, str) for key in resolved
                ):
                    raise BenchmarkReleaseError("encoder policy expansion must be a static mapping")
                values.update(cast(Mapping[str, object], resolved))
            elif keyword.arg in {
                "model_id",
                "model_revision",
                "tokenizer_id",
                "tokenizer_revision",
                "document_model_id",
                "document_model_revision",
                "document_tokenizer_id",
                "document_tokenizer_revision",
            }:
                values[keyword.arg] = _ast_static_value(keyword.value, constants)
        for identity_field, revision_field in (
            ("model_id", "model_revision"),
            ("tokenizer_id", "tokenizer_revision"),
        ):
            identity = values.get(identity_field)
            revision = values.get(revision_field)
            if (
                not isinstance(identity, str)
                or not identity
                or not isinstance(revision, str)
                or _GIT_COMMIT.fullmatch(revision) is None
            ):
                raise BenchmarkReleaseError(
                    "encoder model and tokenizer references must have exact static revisions"
                )
            models.add(f"{identity}@{revision}")
        # An asymmetric policy routes documents to a second checkpoint. That
        # tower ships in the release contract like the query-side pair, so when
        # declared it is audited identically; declaring half an identity fails.
        for identity_field, revision_field in (
            ("document_model_id", "document_model_revision"),
            ("document_tokenizer_id", "document_tokenizer_revision"),
        ):
            identity = values.get(identity_field)
            revision = values.get(revision_field)
            if identity is None and revision is None:
                continue
            if (
                not isinstance(identity, str)
                or not identity
                or not isinstance(revision, str)
                or _GIT_COMMIT.fullmatch(revision) is None
            ):
                raise BenchmarkReleaseError(
                    "encoder document-tower references must have exact static revisions"
                )
            models.add(f"{identity}@{revision}")
    pipeline_prefix = "src/taim/data/pipelines/"
    for destination, path in files.items():
        if not destination.startswith(pipeline_prefix) or not destination.endswith(".json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BenchmarkReleaseError("pipeline model declaration is not readable JSON") from exc
        reranker = payload.get("reranker") if isinstance(payload, Mapping) else None
        if not isinstance(reranker, Mapping):
            continue
        model_id = reranker.get("model_id")
        model_revision = reranker.get("model_revision")
        tokenizer_revision = reranker.get("tokenizer_revision")
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(model_revision, str)
            or _GIT_COMMIT.fullmatch(model_revision) is None
            or tokenizer_revision != model_revision
        ):
            raise BenchmarkReleaseError(
                "pipeline reranker references must have exact matching model/tokenizer revisions"
            )
        models.add(f"{model_id}@{model_revision}")
    external_models, _external_inputs = _external_system_dependencies(files, definition)
    models.update(external_models)
    return models


def _external_system_dependencies(
    files: Mapping[str, Path],
    definition: Mapping[str, object],
) -> tuple[set[str], set[str]]:
    """External models and runtime inputs, from the dependency files shipped extensions declare.

    Each declaration is read as the definition ships it: a generated declaration keeps only the
    published Systems' entries, and every entry left must name a pipeline the definition declares.
    """

    declared_systems = frozenset(cast(Mapping[str, object], definition["pipelines"]))
    declared_files = sorted(
        {
            path
            for declaration in _extension_declarations(files).values()
            for path in cast(tuple[str, ...], declaration.get("dependency_declarations", ()))
        }
    )
    models: set[str] = set()
    inputs: set[str] = set()
    for relative in declared_files:
        path = files.get(relative)
        if path is None:
            raise BenchmarkReleaseError(
                f"external System dependency declaration is absent: {relative}"
            )
        try:
            source = path.read_bytes()
        except OSError as exc:
            raise BenchmarkReleaseError(
                "external System dependency declaration is unreadable"
            ) from exc
        shipped = _shipped_bytes(definition, relative, source)
        file_models, file_inputs = _read_external_system_dependencies(shipped, declared_systems)
        models.update(file_models)
        inputs.update(file_inputs)
    return models, inputs


def _read_external_system_dependencies(
    content: bytes,
    declared_systems: Collection[str],
) -> tuple[set[str], set[str]]:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkReleaseError("external System dependency declaration is unreadable") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "systems"}:
        raise BenchmarkReleaseError("external System dependency declaration has invalid fields")
    systems = payload["systems"]
    if payload["schema_version"] != "1.0" or not isinstance(systems, list) or not systems:
        raise BenchmarkReleaseError("external System dependency declaration is invalid")
    models: set[str] = set()
    inputs: set[str] = set()
    system_ids: list[str] = []
    for index, raw_system in enumerate(systems):
        if not isinstance(raw_system, Mapping) or set(raw_system) != {
            "system_id",
            "models",
            "runtime_inputs",
        }:
            raise BenchmarkReleaseError(f"external System dependency {index} has invalid fields")
        system_id = _string(raw_system["system_id"], role="external System dependency ID")
        if system_id not in declared_systems:
            raise BenchmarkReleaseError(
                f"external dependency names System {system_id!r}, "
                "which the release does not declare"
            )
        system_ids.append(system_id)
        raw_models = raw_system["models"]
        raw_inputs = raw_system["runtime_inputs"]
        if not isinstance(raw_models, list) or not isinstance(raw_inputs, list):
            raise BenchmarkReleaseError("external System dependency arrays are invalid")
        for model in raw_models:
            if not isinstance(model, Mapping) or set(model) != {"model_id", "revision"}:
                raise BenchmarkReleaseError("external model dependency has invalid fields")
            model_id = _string(model["model_id"], role="external model ID")
            revision = _string(model["revision"], role="external model revision")
            if _GIT_COMMIT.fullmatch(revision) is None:
                raise BenchmarkReleaseError("external model revision is not an exact commit")
            models.add(f"{model_id}@{revision}")
        for runtime_input in raw_inputs:
            if not isinstance(runtime_input, Mapping) or set(runtime_input) != {
                "acquisition",
                "input_id",
                "redistribution",
            }:
                raise BenchmarkReleaseError("external runtime input has invalid fields")
            if runtime_input["redistribution"] != "not-included" or runtime_input[
                "acquisition"
            ] not in {
                "separately-authorized-provider",
                "user-supplied",
                "user-supplied-licensed-inputs",
            }:
                raise BenchmarkReleaseError("external runtime input is not fail-closed")
            inputs.add(_string(runtime_input["input_id"], role="external runtime input ID"))
    if system_ids != sorted(set(system_ids)):
        raise BenchmarkReleaseError("external System dependencies must be sorted and unique")
    return models, inputs


def _runtime_input_references(
    files: Mapping[str, Path], definition: Mapping[str, object]
) -> set[str]:
    references: set[str] = set()
    pipeline_prefix = "src/taim/data/pipelines/"
    for destination, path in files.items():
        if not destination.startswith(pipeline_prefix) or not destination.endswith(".json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BenchmarkReleaseError(
                "pipeline licensed-input declaration is unreadable"
            ) from exc
        raw_inputs = payload.get("licensed_inputs") if isinstance(payload, Mapping) else None
        if not isinstance(raw_inputs, list):
            continue
        for raw in raw_inputs:
            if (
                not isinstance(raw, Mapping)
                or set(raw)
                != {"acquisition", "input_id", "output_policy", "redistribution", "role"}
                or raw.get("acquisition") != "user-supplied"
                or raw.get("redistribution") != "not-included"
            ):
                raise BenchmarkReleaseError("pipeline licensed runtime input is not fail-closed")
            references.add(_string(raw["input_id"], role="pipeline licensed runtime input"))
    _external_models, external_inputs = _external_system_dependencies(files, definition)
    references.update(external_inputs)
    return references


def _derived_license_bindings(
    files: Mapping[str, Path], definition: Mapping[str, object]
) -> set[tuple[str, str]]:
    """Every licence binding a file set needs: distributions, files, actions, models and inputs."""

    project = cast(Mapping[str, object], definition["project"])
    license_kinds = {
        cast(str, item["license_id"]): cast(str, item["kind"])
        for item in cast(list[Mapping[str, object]], definition["license_identities"])
    }
    release_files = cast(list[Mapping[str, object]], definition["files"])
    return {
        *(("python-distribution", item) for item in _python_distributions(files, project)),
        *(
            (
                f"release-{license_kinds[cast(str, item['license_id'])]}-file",
                cast(str, item["destination"]),
            )
            for item in release_files
        ),
        *(("workflow-action", item) for item in _workflow_actions(files)),
        *(("model", item) for item in _model_references(files, definition)),
        *(("runtime-input", item) for item in _runtime_input_references(files, definition)),
    }


def _validate_license_closure(files: Mapping[str, Path], definition: Mapping[str, object]) -> None:
    _validate_ci_contract(files, definition)
    project = cast(Mapping[str, object], definition["project"])
    _validate_package_project_metadata(files, project)
    derived = _derived_license_bindings(files, definition)
    declared = {
        (cast(str, item["kind"]), cast(str, item["locator"]))
        for item in cast(list[Mapping[str, object]], definition["license_bindings"])
    }
    if declared != derived:
        missing = sorted(derived - declared)
        extra = sorted(declared - derived)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(f"{kind}:{locator}" for kind, locator in missing))
        if extra:
            details.append("extra " + ", ".join(f"{kind}:{locator}" for kind, locator in extra))
        raise BenchmarkReleaseError(
            "license bindings do not exactly cover exported components (" + "; ".join(details) + ")"
        )


def _with_every_pipeline_published(definition: Mapping[str, object]) -> dict[str, object]:
    """The private definition as if nothing were held back, for checks over every pipeline."""

    pipelines = cast(Mapping[str, object], definition["pipelines"])
    return {**definition, "published_pipelines": sorted(pipelines)}


def _project_definition(
    definition: Mapping[str, object], files: Mapping[str, Path]
) -> tuple[dict[str, object], dict[str, Path]]:
    """Project a private definition onto the pipelines it publishes.

    A file stays when no pipeline owns it or a published pipeline does, and its owners narrow to the
    published ones. Held-back pipelines' entries and support rows go. A licence binding stays when
    the projected files still derive it, and a licence identity goes only with the last binding or
    file that used it.
    """

    published = frozenset(cast(list[str], definition["published_pipelines"]))
    kept_files: list[dict[str, object]] = []
    for raw_file in cast(list[Mapping[str, object]], definition["files"]):
        owners = cast(list[str], raw_file["owners"])
        if owners and not published.intersection(owners):
            continue
        kept_files.append({**raw_file, "owners": [owner for owner in owners if owner in published]})
    projected_files = {
        cast(str, entry["destination"]): files[cast(str, entry["destination"])]
        for entry in kept_files
    }
    pipelines = cast(Mapping[str, object], definition["pipelines"])
    projected: dict[str, object] = {
        **definition,
        "files": kept_files,
        "generated_files": {
            destination: generator
            for destination, generator in cast(
                Mapping[str, str], definition["generated_files"]
            ).items()
            if destination in projected_files
        },
        "pipelines": {
            system_id: pipeline
            for system_id, pipeline in pipelines.items()
            if system_id in published
        },
        "support_matrix": [
            row
            for row in cast(list[Mapping[str, object]], definition["support_matrix"])
            if row["system"] in published
        ],
    }
    derived = _derived_license_bindings(projected_files, projected)
    all_bindings = cast(list[Mapping[str, object]], definition["license_bindings"])
    bindings = [item for item in all_bindings if (item["kind"], item["locator"]) in derived]
    used_before = {item["license_id"] for item in all_bindings} | {
        item["license_id"] for item in cast(list[Mapping[str, object]], definition["files"])
    }
    used_after = {item["license_id"] for item in bindings} | {
        item["license_id"] for item in kept_files
    }
    projected["license_bindings"] = bindings
    projected["license_identities"] = [
        identity
        for identity in cast(list[Mapping[str, object]], definition["license_identities"])
        if identity["license_id"] in used_after or identity["license_id"] not in used_before
    ]
    return projected, projected_files


def _provenance(manifest: Mapping[str, object]) -> dict[str, object]:
    # The source commit is recorded bare: a URL would name the private repository.
    generated = cast(Mapping[str, str], manifest["generated_files"])
    return {
        "artifact_type": "taim-benchmark-release-provenance",
        "schema_version": RELEASE_PROVENANCE_VERSION,
        "release_version": manifest["release_version"],
        "taim_source_commit": manifest["source_commit"],
        "source_files": [
            {
                "source": entry["source"],
                "destination": entry["destination"],
                "sha256": entry["sha256"],
                "byte_size": entry["byte_size"],
                **(
                    {"generator": generated[cast(str, entry["destination"])]}
                    if entry["destination"] in generated
                    else {}
                ),
            }
            for entry in cast(list[Mapping[str, object]], manifest["files"])
        ],
    }


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _tree_records(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    records = [
        {
            "path": entry["destination"],
            "sha256": entry["sha256"],
            "byte_size": entry["byte_size"],
        }
        for entry in cast(list[Mapping[str, object]], manifest["files"])
    ]
    provenance = _json_bytes(_provenance(manifest))
    records.append(
        {
            "path": RELEASE_PROVENANCE_FILENAME,
            "sha256": _sha256_bytes(provenance),
            "byte_size": len(provenance),
        }
    )
    return sorted(records, key=lambda item: cast(str, item["path"]))


def _public_tree_id(records: Sequence[Mapping[str, object]]) -> str:
    return content_sha256({"files": [dict(record) for record in records]})


def _release_package_files(manifest: Mapping[str, object]) -> dict[str, tuple[str, int]]:
    records: dict[str, tuple[str, int]] = {}
    for entry in cast(list[Mapping[str, object]], manifest["files"]):
        destination = cast(str, entry["destination"])
        if not destination.startswith("src/taim/"):
            continue
        relative = destination.removeprefix("src/taim/")
        if relative in records:
            raise BenchmarkReleaseError(
                f"release manifest contains duplicate package file {relative!r}"
            )
        records[relative] = (cast(str, entry["sha256"]), cast(int, entry["byte_size"]))
    if not records:
        raise BenchmarkReleaseError("release manifest projects no TAIM package files")
    return dict(sorted(records.items()))


def _artifact_package_files(
    path: Path, manifest: Mapping[str, object]
) -> dict[str, tuple[str, int]]:
    records: dict[str, tuple[str, int]] = {}
    manifest_files = {
        cast(str, entry["destination"]): (
            cast(str, entry["sha256"]),
            cast(int, entry["byte_size"]),
        )
        for entry in cast(list[Mapping[str, object]], manifest["files"])
    }
    project = cast(Mapping[str, object], manifest["project"])
    distribution_name = cast(str, project["distribution_name"])
    package_version = cast(str, project["package_version"])
    normalized_distribution = re.sub(r"[-_.]+", "_", distribution_name).casefold()

    def add(relative: str, payload: bytes) -> None:
        if relative in records:
            raise BenchmarkReleaseError(
                f"package artifact contains duplicate TAIM file {relative!r}"
            )
        records[relative] = (_sha256_bytes(payload), len(payload))

    def safe_member(name: str) -> PurePosixPath:
        member = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or member.is_absolute()
            or any(part in {"", ".", ".."} for part in member.parts)
        ):
            raise BenchmarkReleaseError(f"package artifact member path is unsafe: {name!r}")
        return member

    def metadata_identity(payload: bytes, *, role: str) -> Message:
        try:
            metadata = Parser().parsestr(payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise BenchmarkReleaseError(f"{role} is not UTF-8") from exc

        allowed_fields = {
            "Author",
            "Author-email",
            "Description-Content-Type",
            "Dynamic",
            "License-Expression",
            "License-File",
            "Metadata-Version",
            "Name",
            "Project-URL",
            "Provides-Extra",
            "Requires-Dist",
            "Requires-Python",
            "Summary",
            "Version",
        }
        if set(metadata.keys()) - allowed_fields:
            raise BenchmarkReleaseError(f"{role} added undeclared project metadata")

        def singleton(field: str) -> str | None:
            values = metadata.get_all(field, [])
            if len(values) != 1:
                raise BenchmarkReleaseError(f"{role} {field} must occur exactly once")
            return values[0]

        observed_name = singleton("Name")
        observed_version = singleton("Version")
        if (
            singleton("Metadata-Version") != "2.4"
            or not isinstance(observed_name, str)
            or re.sub(r"[-_.]+", "_", observed_name).casefold() != normalized_distribution
            or observed_version != package_version
        ):
            raise BenchmarkReleaseError(f"{role} does not match the release project identity")
        observed_python = singleton("Requires-Python")
        expected_python = cast(str, project["requires_python"])
        if not isinstance(observed_python, str) or {
            item.strip() for item in observed_python.split(",")
        } != {item.strip() for item in expected_python.split(",")}:
            raise BenchmarkReleaseError(f"{role} Requires-Python changed")
        if singleton("License-Expression") != project["license_expression"]:
            raise BenchmarkReleaseError(f"{role} license expression changed")
        author_names: list[str] = []
        author_emails: list[str] = []
        for author in cast(list[Mapping[str, str]], project["authors"]):
            name = author.get("name", "")
            email = author.get("email")
            if email is None:
                author_names.append(name)
            else:
                author_emails.append(formataddr((name, email)))
        expected_author_names = [", ".join(author_names)] if author_names else []
        expected_author_emails = [", ".join(author_emails)] if author_emails else []
        if (
            singleton("Summary") != project["package_summary"]
            or metadata.get_all("Author", []) != expected_author_names
            or metadata.get_all("Author-email", []) != expected_author_emails
            or singleton("Description-Content-Type") != "text/markdown"
            or metadata.get_all("Project-URL", []) != [f"Repository, {project['repository_url']}"]
            or metadata.get_all("License-File", [])
            != [PurePosixPath(cast(str, project["license_file"])).name]
            or metadata.get_all("Dynamic", []) != ["license-file"]
        ):
            raise BenchmarkReleaseError(f"{role} changed reviewed project metadata")
        description = metadata.get_payload()
        readme_file = cast(str, project["readme_file"])
        if not isinstance(description, str):
            raise BenchmarkReleaseError(f"{role} description payload is invalid")
        description_bytes = description.encode("utf-8")
        if (_sha256_bytes(description_bytes), len(description_bytes)) != manifest_files[
            readme_file
        ]:
            raise BenchmarkReleaseError(f"{role} description does not match the reviewed README")
        observed_dependencies = metadata.get_all("Requires-Dist", [])
        observed_extras = metadata.get_all("Provides-Extra", [])
        if (
            len(observed_dependencies) != len(set(observed_dependencies))
            or sorted(observed_dependencies) != project["package_requires_dist"]
            or len(observed_extras) != len(set(observed_extras))
            or sorted(observed_extras) != project["package_provides_extra"]
        ):
            raise BenchmarkReleaseError(f"{role} dependency closure changed")
        return metadata

    def validate_entry_points(payload: bytes, *, role: str) -> None:
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(payload.decode("utf-8"))
        except (UnicodeDecodeError, configparser.Error) as exc:
            raise BenchmarkReleaseError(f"{role} is invalid") from exc
        expected = {cast(str, project["public_cli"]): "taim.release_cli:main"}
        if (
            set(parser.sections()) != {"console_scripts"}
            or dict(parser["console_scripts"]) != expected
        ):
            raise BenchmarkReleaseError(f"{role} does not match the release CLI")

    def requires_dist_from_egg_info(payload: bytes) -> list[str]:
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise BenchmarkReleaseError(
                "release source distribution requires metadata is not UTF-8"
            ) from exc
        extra: str | None = None
        section_marker: str | None = None
        requirements: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                raw_extra, separator, raw_marker = stripped[1:-1].partition(":")
                extra = raw_extra.strip()
                section_marker = raw_marker.strip() if separator else None
                if not extra or (separator and not section_marker):
                    raise BenchmarkReleaseError(
                        "release source distribution requires metadata is unsupported"
                    )
                continue
            requirement = stripped if section_marker is None else f"{stripped}; {section_marker}"
            requirements.append(_metadata_requirement(requirement, extra=extra))
        return sorted(requirements)

    if path.suffix == ".whl":
        filename_parts = path.name.removesuffix(".whl").split("-")
        if len(filename_parts) not in {5, 6} or not zipfile.is_zipfile(path):
            raise BenchmarkReleaseError("release wheel filename or ZIP structure is invalid")
        distribution, version = filename_parts[:2]
        if distribution.casefold() != normalized_distribution or version != package_version:
            raise BenchmarkReleaseError("release wheel filename does not match the release project")
        dist_info = f"{distribution}-{version}.dist-info"
        with zipfile.ZipFile(path) as zip_archive:
            wheel_payloads: dict[str, bytes] = {}
            for zip_item in zip_archive.infolist():
                safe_member(zip_item.filename.rstrip("/"))
                mode = zip_item.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise BenchmarkReleaseError("release wheel contains a symlink")
                if zip_item.is_dir():
                    continue
                if zip_item.filename in wheel_payloads:
                    raise BenchmarkReleaseError(
                        f"release wheel contains duplicate file {zip_item.filename!r}"
                    )
                wheel_payloads[zip_item.filename] = zip_archive.read(zip_item)
            required = {
                f"{dist_info}/METADATA",
                f"{dist_info}/RECORD",
                f"{dist_info}/WHEEL",
                f"{dist_info}/entry_points.txt",
                f"{dist_info}/top_level.txt",
            }
            license_file = cast(str, project["license_file"])
            license_member = f"{dist_info}/licenses/{PurePosixPath(license_file).name}"
            required.add(license_member)
            names = set(wheel_payloads)
            if not required <= names:
                raise BenchmarkReleaseError("release wheel lacks required dist-info metadata")
            allowed = {name for name in names if name.startswith("taim/")}
            allowed.update(required)
            if names - allowed:
                raise BenchmarkReleaseError(
                    "release wheel contains payload outside the closed package allowlist"
                )
            wheel_metadata = Parser().parsestr(wheel_payloads[f"{dist_info}/WHEEL"].decode("utf-8"))
            expected_tag = "-".join(filename_parts[-3:])
            if (
                set(wheel_metadata.keys())
                != {"Wheel-Version", "Generator", "Root-Is-Purelib", "Tag"}
                or wheel_metadata.get_all("Wheel-Version", []) != ["1.0"]
                or len(wheel_metadata.get_all("Generator", [])) != 1
                or wheel_metadata.get_all("Root-Is-Purelib", []) != ["true"]
                or wheel_metadata.get_all("Tag", []) != [expected_tag]
            ):
                raise BenchmarkReleaseError("release wheel WHEEL metadata is invalid")
            metadata_identity(
                wheel_payloads[f"{dist_info}/METADATA"], role="release wheel METADATA"
            )
            validate_entry_points(
                wheel_payloads[f"{dist_info}/entry_points.txt"],
                role="release wheel entry points",
            )
            if wheel_payloads[f"{dist_info}/top_level.txt"].decode("utf-8").split() != ["taim"]:
                raise BenchmarkReleaseError("release wheel top-level metadata is invalid")
            if (
                _sha256_bytes(wheel_payloads[license_member]),
                len(wheel_payloads[license_member]),
            ) != manifest_files[license_file]:
                raise BenchmarkReleaseError(
                    "release wheel license file does not match the release manifest"
                )
            try:
                record_rows = list(
                    csv.reader(io.StringIO(wheel_payloads[f"{dist_info}/RECORD"].decode("utf-8")))
                )
            except UnicodeDecodeError as exc:
                raise BenchmarkReleaseError("release wheel RECORD is not UTF-8") from exc
            if any(len(row) != 3 for row in record_rows):
                raise BenchmarkReleaseError("release wheel RECORD row is invalid")
            record_by_name = {row[0]: (row[1], row[2]) for row in record_rows}
            if len(record_by_name) != len(record_rows) or set(record_by_name) != names:
                raise BenchmarkReleaseError("release wheel RECORD does not cover the exact archive")
            record_name = f"{dist_info}/RECORD"
            for name, payload in wheel_payloads.items():
                declared_hash, declared_size = record_by_name[name]
                if name == record_name:
                    if declared_hash or declared_size:
                        raise BenchmarkReleaseError("release wheel RECORD self-row is invalid")
                    continue
                expected_hash = "sha256=" + base64.urlsafe_b64encode(
                    hashlib.sha256(payload).digest()
                ).decode("ascii").rstrip("=")
                if declared_hash != expected_hash or declared_size != str(len(payload)):
                    raise BenchmarkReleaseError("release wheel RECORD hash or size is invalid")
            for name, payload in wheel_payloads.items():
                if name.startswith("taim/"):
                    add(name.removeprefix("taim/"), payload)
    elif path.name.endswith(".tar.gz"):
        if not tarfile.is_tarfile(path):
            raise BenchmarkReleaseError("release source distribution is not a readable tarball")
        expected_root = path.name.removesuffix(".tar.gz")
        if expected_root != f"{normalized_distribution}-{package_version}":
            raise BenchmarkReleaseError(
                "release source distribution filename does not match the release project"
            )
        with tarfile.open(path, mode="r:*") as tar_archive:
            members = tar_archive.getmembers()
            for tar_item in members:
                safe_member(tar_item.name)
                if not (tar_item.isfile() or tar_item.isdir()):
                    raise BenchmarkReleaseError(
                        "release source distribution contains a special filesystem node"
                    )
                if tar_item.name.split("/", 1)[0] != expected_root:
                    raise BenchmarkReleaseError(
                        "release source distribution does not have one filename-bound root"
                    )
            member_by_name = {item.name: item for item in members if item.isfile()}
            if len(member_by_name) != sum(item.isfile() for item in members):
                raise BenchmarkReleaseError("release source distribution contains duplicate files")
            required = {
                f"{expected_root}/PKG-INFO",
                f"{expected_root}/pyproject.toml",
            }
            if not required <= member_by_name.keys():
                raise BenchmarkReleaseError(
                    "release source distribution lacks required build metadata"
                )
            source_payloads: dict[str, bytes] = {}
            for name, tar_item in member_by_name.items():
                stream = tar_archive.extractfile(tar_item)
                if stream is None:
                    raise BenchmarkReleaseError(
                        f"cannot read package artifact member {tar_item.name!r}"
                    )
                source_payloads[name.removeprefix(f"{expected_root}/")] = stream.read()
            metadata_identity(
                source_payloads["PKG-INFO"], role="release source distribution PKG-INFO"
            )
            try:
                pyproject = tomllib.loads(source_payloads["pyproject.toml"].decode("utf-8"))
            except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
                raise BenchmarkReleaseError(
                    "release source distribution pyproject.toml is invalid"
                ) from exc
            if not isinstance(pyproject.get("build-system"), Mapping) or not isinstance(
                pyproject.get("project"), Mapping
            ):
                raise BenchmarkReleaseError(
                    "release source distribution pyproject.toml lacks build metadata"
                )
            project_metadata = cast(Mapping[str, object], pyproject["project"])
            source_project_name = project_metadata.get("name")
            if (
                not isinstance(source_project_name, str)
                or re.sub(r"[-_.]+", "_", source_project_name).casefold() != normalized_distribution
                or project_metadata.get("version") != package_version
            ):
                raise BenchmarkReleaseError(
                    "release source distribution pyproject.toml project identity changed"
                )
            egg_info = f"src/{normalized_distribution}.egg-info"
            generated = {
                "PKG-INFO",
                "setup.cfg",
                f"{egg_info}/PKG-INFO",
                f"{egg_info}/SOURCES.txt",
                f"{egg_info}/dependency_links.txt",
                f"{egg_info}/entry_points.txt",
                f"{egg_info}/requires.txt",
                f"{egg_info}/top_level.txt",
            }
            required_source_members = {
                "PKG-INFO",
                "pyproject.toml",
                "setup.cfg",
                cast(str, project["license_file"]),
                cast(str, project["readme_file"]),
                f"{egg_info}/PKG-INFO",
                f"{egg_info}/SOURCES.txt",
                f"{egg_info}/dependency_links.txt",
                f"{egg_info}/entry_points.txt",
                f"{egg_info}/requires.txt",
                f"{egg_info}/top_level.txt",
            }
            if not required_source_members <= source_payloads.keys():
                raise BenchmarkReleaseError(
                    "release source distribution omits required source or generated metadata"
                )
            unknown = set(source_payloads) - manifest_files.keys() - generated
            if unknown:
                raise BenchmarkReleaseError(
                    "release source distribution contains payload outside the closed source "
                    "allowlist"
                )
            for relative, payload in source_payloads.items():
                if (
                    relative in manifest_files
                    and (
                        _sha256_bytes(payload),
                        len(payload),
                    )
                    != manifest_files[relative]
                ):
                    raise BenchmarkReleaseError(
                        f"release source distribution file changed: {relative}"
                    )
            egg_pkg_info = f"{egg_info}/PKG-INFO"
            if (
                egg_pkg_info in source_payloads
                and source_payloads[egg_pkg_info] != source_payloads["PKG-INFO"]
            ):
                raise BenchmarkReleaseError(
                    "release source distribution PKG-INFO projections disagree"
                )
            egg_entry_points = f"{egg_info}/entry_points.txt"
            validate_entry_points(
                source_payloads[egg_entry_points],
                role="release source distribution entry points",
            )
            egg_top_level = f"{egg_info}/top_level.txt"
            if source_payloads[egg_top_level].decode("utf-8").split() != ["taim"]:
                raise BenchmarkReleaseError(
                    "release source distribution top-level metadata is invalid"
                )
            setup = configparser.ConfigParser(interpolation=None)
            try:
                setup.read_string(source_payloads["setup.cfg"].decode("utf-8"))
            except (UnicodeDecodeError, configparser.Error) as exc:
                raise BenchmarkReleaseError(
                    "release source distribution setup.cfg is invalid"
                ) from exc
            if set(setup.sections()) != {"egg_info"} or dict(setup["egg_info"]) != {
                "tag_build": "",
                "tag_date": "0",
            }:
                raise BenchmarkReleaseError(
                    "release source distribution setup.cfg changed build semantics"
                )
            dependency_links = f"{egg_info}/dependency_links.txt"
            if source_payloads[dependency_links].decode("utf-8").strip():
                raise BenchmarkReleaseError(
                    "release source distribution dependency links must be empty"
                )
            egg_requires = f"{egg_info}/requires.txt"
            if (
                requires_dist_from_egg_info(source_payloads[egg_requires])
                != project["package_requires_dist"]
            ):
                raise BenchmarkReleaseError(
                    "release source distribution dependency closure changed"
                )
            sources_name = f"{egg_info}/SOURCES.txt"
            try:
                source_rows = source_payloads[sources_name].decode("utf-8").splitlines()
            except UnicodeDecodeError as exc:
                raise BenchmarkReleaseError(
                    "release source distribution SOURCES.txt is not UTF-8"
                ) from exc
            expected_sources = set(source_payloads) - {"PKG-INFO", "setup.cfg"}
            if len(source_rows) != len(set(source_rows)) or set(source_rows) != expected_sources:
                raise BenchmarkReleaseError(
                    "release source distribution SOURCES.txt does not cover the exact sources"
                )
            for relative, payload in source_payloads.items():
                if relative.startswith("src/taim/"):
                    package_relative = relative.removeprefix("src/taim/")
                    if not package_relative:
                        continue
                    add(package_relative, payload)
    else:
        raise BenchmarkReleaseError(
            "release package artifact must be a .whl or .tar.gz distribution"
        )
    if not records:
        raise BenchmarkReleaseError("release package artifact contains no TAIM package files")
    return dict(sorted(records.items()))


def validate_release_package_artifact(
    manifest_path: str | Path,
    package_artifact_path: str | Path,
) -> BenchmarkReleasePackageArtifact:
    """Bind one exact distribution archive to a verified frozen release manifest."""

    manifest = load_benchmark_release_manifest(manifest_path)
    observed_tree_id = _public_tree_id(_tree_records(manifest))
    if observed_tree_id != manifest["expected_public_tree_id"]:
        raise BenchmarkReleaseError("public tree identity does not match the release manifest")
    artifact = Path(package_artifact_path)
    if not artifact.is_file():
        raise BenchmarkReleaseError(f"release package artifact does not exist: {artifact}")
    archived = _artifact_package_files(artifact, manifest)
    if archived != _release_package_files(manifest):
        raise BenchmarkReleaseError(
            "release package artifact does not match the frozen release manifest"
        )
    return BenchmarkReleasePackageArtifact(
        release_manifest_id=cast(str, manifest["manifest_id"]),
        public_tree_id=observed_tree_id,
        package_artifact_id=_sha256_file(artifact),
        package_files=tuple(
            (relative, digest, byte_size) for relative, (digest, byte_size) in archived.items()
        ),
    )


def freeze_benchmark_release_manifest(
    source_root: str | Path,
    definition_path: str | Path,
    manifest_path: str | Path,
    *,
    gates: Sequence[Callable[[Mapping[str, object]], None]] = (),
) -> dict[str, object]:
    """Bind approved release inputs and an exact file allowlist to one clean commit.

    The manifest carries the private definition projected onto its published pipelines, so it names
    no held-back pipeline, no file only those pipelines own, and no licence binding only they need.
    Each gate receives the validated manifest before it is written, and refuses it by raising. The
    private exporter passes its release lock check here.
    """

    source = Path(source_root).resolve()
    output = Path(manifest_path)
    if output.resolve().is_relative_to(source):
        raise BenchmarkReleaseError("release manifest must be outside the source tree")
    if output.exists():
        raise BenchmarkReleaseError(f"release manifest already exists: {output}")
    definition = _json_object(Path(definition_path), role="Benchmark Release definition")
    _validate_definition(definition, projected=False)
    source_commit = _clean_source_commit(source)
    sources = {
        cast(str, raw_file["destination"]): _source_file(
            source,
            cast(str, raw_file["source"]),
            source_commit=source_commit,
        )
        for raw_file in cast(list[Mapping[str, object]], definition["files"])
    }
    # The private definition must close over every pipeline it declares before any is held back,
    # so its generated files are read as if every declared pipeline were published.
    _validate_license_closure(sources, _with_every_pipeline_published(definition))
    # Section owners are checked while the definition still declares the held-back pipelines.
    _validate_markdown_section_owners(sources, definition)
    _validate_python_ships_with_files(sources, definition)
    projected, projected_sources = _project_definition(definition, sources)
    _validate_definition(projected)
    _validate_license_closure(projected_sources, projected)
    files: list[dict[str, object]] = []
    for raw_file in cast(list[Mapping[str, object]], projected["files"]):
        destination = cast(str, raw_file["destination"])
        shipped = _shipped_bytes(
            projected, destination, projected_sources[destination].read_bytes()
        )
        files.append({**raw_file, "sha256": _sha256_bytes(shipped), "byte_size": len(shipped)})
    manifest: dict[str, object] = {
        **projected,
        "artifact_type": "taim-benchmark-release-manifest",
        "schema_version": RELEASE_MANIFEST_VERSION,
        "source_commit": source_commit,
        "files": files,
        "expected_public_tree_id": "sha256:" + "0" * 64,
    }
    manifest["expected_public_tree_id"] = _public_tree_id(_tree_records(manifest))
    manifest["manifest_id"] = content_sha256(manifest)
    loadable = {**manifest}
    _validate_definition(_definition_from_manifest(loadable))
    for gate in gates:
        gate(loadable)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, loadable)
    return load_benchmark_release_manifest(output)


def _copy_release_files(source: Path, destination: Path, manifest: Mapping[str, object]) -> None:
    for entry in cast(list[Mapping[str, object]], manifest["files"]):
        relative_source = cast(str, entry["source"])
        source_file = _source_file(
            source,
            relative_source,
            source_commit=cast(str, manifest["source_commit"]),
        )
        shipped = _shipped_bytes(
            manifest, cast(str, entry["destination"]), source_file.read_bytes()
        )
        if _sha256_bytes(shipped) != entry["sha256"] or len(shipped) != entry["byte_size"]:
            raise BenchmarkReleaseError(f"release source bytes changed: {relative_source}")
        target = destination.joinpath(*PurePosixPath(cast(str, entry["destination"])).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(shipped)
    (destination / RELEASE_PROVENANCE_FILENAME).write_bytes(_json_bytes(_provenance(manifest)))


def build_benchmark_release(
    source_root: str | Path,
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    gates: Sequence[Callable[[Path, Path], None]] = (),
) -> BenchmarkReleaseCandidate:
    """Construct a complete candidate atomically, with no Git history.

    Each gate receives the staged tree and the manifest path once the tree validates, before it
    gains history or moves into place, and refuses it by raising. The private exporter passes its
    excluded-reference scan here. The candidate is admitted as a tree before it moves into place;
    ``taim release chain`` gives it its only commit.

    Guarantee: deny-by-default makes leaking a FILE impossible.

    It does not cover:
    1. Content inside a named file. That takes a separate content rule.
    2. Re-verification on a published release. The excluded-reference scan is a private build-time gate - it searches for held-back pipeline names, so it cannot ship publicly without publishing them.
    3. Any push shape other than explicit release refspecs. The public release check does not see the history a push carries, so a mirror push publishes whatever the repository holds.
    4. Release assets and release notes. They live outside the release tree, which is all the deny-by-default walk sees. Assets are verified against what was built and uploaded; release notes remain editable after publication and are not covered by any check.
    """  # noqa: E501

    source = Path(source_root).resolve()
    output = Path(output_directory).resolve()
    if output.is_relative_to(source):
        raise BenchmarkReleaseError("release output must be outside the source tree")
    if output.exists():
        raise BenchmarkReleaseError(f"release output already exists: {output}")
    manifest = load_benchmark_release_manifest(manifest_path)
    source_commit = _clean_source_commit(source)
    if source_commit != manifest["source_commit"]:
        raise BenchmarkReleaseError("source HEAD does not match the frozen release commit")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _copy_release_files(source, staging, manifest)
        validate_benchmark_release(staging, manifest_path, require_git=False)
        for gate in gates:
            gate(staging, Path(manifest_path))
        report = admit_release_tree(staging, manifest_path)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return BenchmarkReleaseCandidate(
        directory=output,
        manifest_id=cast(str, manifest["manifest_id"]),
        public_tree_id=cast(str, report["public_tree_id"]),
    )


def _tree_ignore_rules(candidate: Path) -> tuple[tuple[str, bool, bool], ...]:
    """Parse the tree's own root ``.gitignore`` into (pattern, directory_only, anchored) rules.

    The release ships that file and its manifest binds the file's digest, so a changed rule fails
    the content check. Git treats what it names as outside the tree, and so does the public walk:
    the files the shipped instructions create in a clone (``.venv/``, ``runs/``, caches) are not
    release content. Negation rules are refused.
    """

    ignore = candidate / ".gitignore"
    if ignore.is_symlink() or not ignore.is_file():
        return ()
    rules: list[tuple[str, bool, bool]] = []
    for raw in ignore.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            raise BenchmarkReleaseError("release .gitignore negation rules are not supported")
        directory_only = line.endswith("/")
        pattern = line.rstrip("/")
        anchored = pattern.startswith("/") or "/" in pattern
        rules.append((pattern.lstrip("/"), directory_only, anchored))
    return tuple(rules)


def _ignored_by_tree(
    relative: PurePosixPath, is_directory: bool, rules: Sequence[tuple[str, bool, bool]]
) -> bool:
    for pattern, directory_only, anchored in rules:
        if directory_only and not is_directory:
            continue
        if anchored:
            if fnmatch.fnmatchcase(relative.as_posix(), pattern):
                return True
        elif fnmatch.fnmatchcase(relative.name, pattern):
            return True
    return False


def _candidate_files(candidate: Path, *, honour_ignore: bool = True) -> dict[str, Path]:
    """Return every file of the release tree, skipping ``.git`` and, by default, what it ignores.

    A symlink anywhere in the walked tree is refused. With ``honour_ignore`` a path the tree's own
    ``.gitignore`` covers is neither collected nor inspected, because it is not part of the tree; a
    clone that followed the shipped instructions validates in place. Admission walks strictly.
    """

    rules = _tree_ignore_rules(candidate) if honour_ignore else ()
    files: dict[str, Path] = {}
    for root, directories, names in os.walk(candidate, followlinks=False):
        root_path = Path(root)
        kept: list[str] = []
        for name in sorted(directories):
            relative = PurePosixPath(root_path.joinpath(name).relative_to(candidate).as_posix())
            if relative.parts[0] == ".git" or _ignored_by_tree(relative, True, rules):
                continue
            if root_path.joinpath(name).is_symlink():
                raise BenchmarkReleaseError(f"release tree contains a symlink: {relative}")
            kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            path = root_path / name
            relative = PurePosixPath(path.relative_to(candidate).as_posix())
            if relative.as_posix() == ".git" or _ignored_by_tree(relative, False, rules):
                continue
            if path.is_symlink():
                raise BenchmarkReleaseError(f"release tree contains a symlink: {relative}")
            if path.is_file():
                files[str(relative)] = path
    return files


def _validate_package_metadata(candidate: Path, project: Mapping[str, object]) -> None:
    try:
        package = tomllib.loads((candidate / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BenchmarkReleaseError("public pyproject.toml is invalid") from exc
    metadata = package.get("project")
    if not isinstance(metadata, Mapping):
        raise BenchmarkReleaseError("public pyproject.toml lacks project metadata")
    expected = {
        "name": project["distribution_name"],
        "version": project["package_version"],
        "requires-python": project["requires_python"],
        "readme": project["readme_file"],
        "license": project["license_expression"],
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise BenchmarkReleaseError("public package metadata does not match the release manifest")
    scripts = metadata.get("scripts")
    if not isinstance(scripts, Mapping) or project["public_cli"] not in scripts:
        raise BenchmarkReleaseError("public package does not expose its declared CLI")
    urls = metadata.get("urls")
    if not isinstance(urls, Mapping) or urls.get("Repository") != project["repository_url"]:
        raise BenchmarkReleaseError("public package repository URL does not match the manifest")


def _validate_published_result_bundles(
    candidate: Path,
    manifest: Mapping[str, object],
) -> None:
    from taim.release_result_bundle import validate_release_result_bundle

    for entry in cast(list[Mapping[str, object]], manifest["published_result_bundles"]):
        destination = candidate.joinpath(*PurePosixPath(cast(str, entry["destination"])).parts)
        report = validate_release_result_bundle(destination)
        if (
            report.get("bundle_id") != entry["bundle_id"]
            or report.get("schema_version") != entry["schema_version"]
        ):
            raise BenchmarkReleaseError(
                f"Published Result Bundle identity does not match: {entry['bundle_id']}"
            )
        bundle_key = (
            report.get("track"),
            report.get("task"),
            report.get("profile"),
            report.get("system"),
        )
        declared_keys = {
            capability_key(capability)
            for capability in validate_support_matrix(manifest["support_matrix"])
        }
        if bundle_key not in declared_keys:
            raise BenchmarkReleaseError(
                "Published Result Bundle is direction-mismatched or outside release support"
            )
        if report.get("release_manifest_id") != manifest["manifest_id"]:
            raise BenchmarkReleaseError("Published Result Bundle release identity does not match")
        if report.get("public_tree_id") != manifest["expected_public_tree_id"]:
            raise BenchmarkReleaseError(
                "Published Result Bundle public tree identity does not match"
            )
        if report.get("package_artifact_id") != entry["package_artifact_id"]:
            raise BenchmarkReleaseError("Published Result Bundle package identity does not match")


def _validate_document_links(candidate: Path, files: Mapping[str, Path]) -> None:
    for relative, path in files.items():
        if path.suffix.casefold() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(text):
            target = match.group(1).strip().split(maxsplit=1)[0].strip("<>")
            if (
                not target
                or target.startswith("#")
                or re.match(r"[a-z][a-z0-9+.-]*:", target, re.I)
            ):
                continue
            target_path = target.split("#", 1)[0]
            resolved = (path.parent / target_path).resolve()
            if not resolved.is_relative_to(candidate.resolve()) or not resolved.exists():
                raise BenchmarkReleaseError(
                    f"documentation link does not resolve: {relative} -> {target}"
                )


def _validate_support_catalog(
    files: Mapping[str, Path],
    manifest: Mapping[str, object],
) -> None:
    path = files.get("src/taim/data/release-support-v0.1.json")
    if path is None:
        raise BenchmarkReleaseError("release omits its executable support catalog")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_matrix = payload["support_matrix"]
        catalog = validate_support_matrix(raw_matrix)
        declared = validate_support_matrix(manifest["support_matrix"])
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkReleaseError(f"release support catalog is invalid: {exc}") from exc
    if catalog != declared:
        raise BenchmarkReleaseError(
            "executable support catalog does not match the release manifest"
        )
    declarations = _extension_declarations(files)
    external_systems = {
        system
        for declaration in declarations.values()
        if declaration.get("external_baseline") is True
        for system in cast(dict[str, str], declaration["system_tasks"])
    }
    external_capabilities = [
        capability
        for capability in catalog
        if capability["profile"] == "trec-ct-2021-external-fidelity-26149"
        or capability["system"] in external_systems
    ]
    if external_capabilities:
        _validate_external_protocol_closure(files)
    _validate_extension_protocol_bindings(files, declarations)
    actual_files = set(files)
    for capability in catalog:
        key = capability_key(capability)
        missing = set(cast(list[str], capability["required_files"])) - actual_files
        if missing:
            raise BenchmarkReleaseError(
                "documented combination cannot run: "
                + "/".join(key)
                + " missing "
                + ", ".join(sorted(missing))
            )


def _validate_external_protocol_closure(files: Mapping[str, Path]) -> None:
    protocol_chain = {
        "docs/paper-analysis-protocol-2026-08-29.md",
        "docs/paper-analysis-protocol-2026-08-31-v2.md",
        "docs/paper-analysis-protocol-2026-08-31-v3.md",
        "docs/paper-analysis-protocol-2026-08-31-v4.md",
    }
    missing_protocols = protocol_chain - set(files)
    if missing_protocols:
        raise BenchmarkReleaseError(
            "external-fidelity support requires the complete normative protocol chain: "
            + ", ".join(sorted(missing_protocols))
        )
    protocol_path = files.get("docs/paper-analysis-protocol-2026-08-31-v4.md")
    support_path = files.get("src/taim/release_support.py")
    if protocol_path is None or support_path is None:
        raise BenchmarkReleaseError(
            "external-fidelity support requires the approved v4 protocol and runtime gate"
        )
    if "docs/paper-analysis-protocol-v4.draft.md" in files:
        raise BenchmarkReleaseError("external-fidelity support cannot project a draft protocol")
    try:
        tree = ast.parse(support_path.read_text(encoding="utf-8"), filename=str(support_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise BenchmarkReleaseError("external-fidelity runtime gate is not inspectable") from exc
    digest: str | None = None
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256":
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            digest = node.value.value
        break
    if digest is None or _SHA256.fullmatch(digest) is None:
        raise BenchmarkReleaseError(
            "external-fidelity runtime protocol digest is not human approved"
        )
    if _sha256_file(protocol_path) != digest:
        raise BenchmarkReleaseError(
            "external-fidelity runtime digest does not match the approved v4 protocol"
        )


def _validate_extension_protocol_bindings(
    files: Mapping[str, Path], declarations: Mapping[str, Mapping[str, object]]
) -> None:
    """A module a shipped extension binds to a protocol must carry the protocol's exact digest."""

    for relative, declaration in declarations.items():
        bindings = cast(tuple[tuple[str, str, str], ...], declaration.get("protocol_bindings", ()))
        for module, constant, document in bindings:
            missing = [path for path in (module, document) if path not in files]
            if missing:
                raise BenchmarkReleaseError(
                    f"{relative} binds {constant} in {module} to {document}; the release lacks "
                    + ", ".join(missing)
                )
            try:
                tree = ast.parse(files[module].read_text(encoding="utf-8"), filename=module)
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                raise BenchmarkReleaseError(
                    f"protocol-bound module is not inspectable: {module}"
                ) from exc
            digest: str | None = None
            for node in tree.body:
                targets: tuple[ast.expr, ...]
                value: ast.expr | None
                if isinstance(node, ast.Assign):
                    targets, value = tuple(node.targets), node.value
                elif isinstance(node, ast.AnnAssign):
                    targets, value = (node.target,), node.value
                else:
                    continue
                if not any(
                    isinstance(target, ast.Name) and target.id == constant for target in targets
                ):
                    continue
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    digest = value.value
                break
            if digest is None or _SHA256.fullmatch(digest) is None:
                raise BenchmarkReleaseError(
                    f"{module} {constant} is not an approved protocol digest"
                )
            if _sha256_file(files[document]) != digest:
                raise BenchmarkReleaseError(
                    f"{module} {constant} does not match the approved protocol {document}"
                )


def _module_package(relative: str) -> str:
    dotted = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
    if relative.endswith("/__init__.py"):
        return dotted.removesuffix(".__init__")
    return dotted.rpartition(".")[0]


def _import_target(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    base = parts[: len(parts) - node.level + 1]
    return ".".join([*base, node.module] if node.module else base)


def _module_bindings(module: ast.Module, package: str) -> frozenset[str] | None:
    """The names a package's module binds at its top level, or None when it may bind any name.

    A name the package binds by importing its own submodule is not counted: that import is what
    the closure checks.
    """

    names: set[str] = set()
    pending: list[ast.stmt] = list(module.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == "__getattr__":
                return None
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                item.id
                for target in node.targets
                for item in ast.walk(target)
                if isinstance(item, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            names.update(item.id for item in ast.walk(node.target) if isinstance(item, ast.Name))
        elif isinstance(node, ast.Import):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                return None
            if _import_target(node, package) != package:
                names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.If | ast.With):
            pending.extend(node.body)
            if isinstance(node, ast.If):
                pending.extend(node.orelse)
        elif isinstance(node, ast.Try):
            pending.extend([*node.body, *node.orelse, *node.finalbody])
            for handler in node.handlers:
                pending.extend(handler.body)
    return frozenset(names)


def _taim_imports(relative: str, module: ast.Module) -> set[tuple[str, str | None]]:
    """Each taim module a module imports, with the name it takes from that module, if any.

    Relative imports resolve against the module's package, and a literal ``import_module`` or
    ``__import__`` argument is an import too.
    """

    package = _module_package(relative)
    found: set[tuple[str, str | None]] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            found.update((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = _import_target(node, package)
            found.update((target, alias.name) for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id in {"import_module", "__import__"}
                )
            )
        ):
            found.add((node.args[0].value, None))
    return {(target, name) for target, name in found if target.split(".")[0] == "taim"}


def _validate_public_import_closure(files: Mapping[str, Path]) -> None:
    """Every taim module a public module or shipped test imports ships.

    That covers absolute and relative imports, a submodule imported by name from its package, and
    a literal ``import_module`` argument, so no shipped module imports one the release left out.
    A shipped test importing a module the release leaves out could not even be collected.
    """

    def module_file(module: str) -> str | None:
        relative = "src/" + module.replace(".", "/")
        return next(
            (path for path in (f"{relative}.py", f"{relative}/__init__.py") if path in files),
            None,
        )

    modules: dict[str, ast.Module] = {}
    tests: dict[str, ast.Module] = {}
    for relative, path in sorted(files.items()):
        if path.suffix != ".py" or not relative.startswith(("src/taim/", "tests/")):
            continue
        try:
            parsed = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise BenchmarkReleaseError(
                f"public Python module is not inspectable: {relative}"
            ) from exc
        (modules if relative.startswith("src/taim/") else tests)[relative] = parsed
    for relative, module in {**modules, **tests}.items():
        missing: set[str] = set()
        for target, name in _taim_imports(relative, module):
            owner = module_file(target)
            if owner is None:
                missing.add(target)
                continue
            if name is None or not owner.endswith("/__init__.py") or owner not in modules:
                continue
            bound = _module_bindings(modules[owner], target)
            if bound is not None and name not in bound and module_file(f"{target}.{name}") is None:
                missing.add(f"{target}.{name}")
        if missing:
            raise BenchmarkReleaseError(
                f"public Python import closure is incomplete for {relative}: "
                + ", ".join(sorted(missing))
            )


class _UnresolvedResource(Exception):
    """A package-data path segment the static walk cannot read."""


def _module_string_constants(module: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value.value
    return constants


def _resource_segment(node: ast.expr, constants: Mapping[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            elif (
                isinstance(part, ast.FormattedValue)
                and part.conversion == -1
                and part.format_spec is None
            ):
                parts.append(_resource_segment(part.value, constants))
            else:
                raise _UnresolvedResource
        return "".join(parts)
    raise _UnresolvedResource


def _is_package_files_root(node: ast.expr) -> bool:
    """``files("taim")``, ``resources.files("taim")`` or ``importlib.resources.files("taim")``."""

    if not isinstance(node, ast.Call) or not node.args:
        return False
    named = (isinstance(node.func, ast.Name) and node.func.id == "files") or (
        isinstance(node.func, ast.Attribute) and node.func.attr == "files"
    )
    first = node.args[0]
    return named and isinstance(first, ast.Constant) and first.value == "taim"


def _module_relative_parts(
    node: ast.expr, module_parts: Sequence[str], constants: Mapping[str, str]
) -> list[str] | None:
    """Tree path parts of a ``Path(__file__)`` expression and its parent or sibling steps."""

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "resolve" and not node.args:
            return _module_relative_parts(node.func.value, module_parts, constants)
        if node.func.attr == "with_name" and len(node.args) == 1:
            base = _module_relative_parts(node.func.value, module_parts, constants)
            if base is None:
                return None
            return [*base[:-1], _resource_segment(node.args[0], constants)]
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
        argument = node.args[0] if len(node.args) == 1 else None
        if isinstance(argument, ast.Name) and argument.id == "__file__":
            return list(module_parts)
        return None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _module_relative_parts(node.value, module_parts, constants)
        return base[:-1] if base else None
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "parents"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ):
        base = _module_relative_parts(node.value.value, module_parts, constants)
        return base[: -(node.slice.value + 1)] if base else None
    return None


def _package_data_reads(
    files: Mapping[str, Path],
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Package-data paths shipped modules name, resolved, and the reads the walk cannot follow."""

    resolved: list[tuple[str, int, str]] = []
    unresolved: list[tuple[str, int, str]] = []
    for relative, path in sorted(files.items()):
        if not relative.startswith("src/taim/") or path.suffix != ".py":
            continue
        try:
            module = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise BenchmarkReleaseError(
                f"public Python module is not inspectable: {relative}"
            ) from exc
        constants = _module_string_constants(module)
        module_parts = relative.split("/")
        nested: set[int] = set()
        for node in ast.walk(module):
            if id(node) in nested:
                continue
            target: list[str] | None
            try:
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "joinpath"
                ):
                    calls: list[ast.Call] = []
                    current: ast.expr = node
                    while (
                        isinstance(current, ast.Call)
                        and isinstance(current.func, ast.Attribute)
                        and current.func.attr == "joinpath"
                    ):
                        calls.insert(0, current)
                        nested.add(id(current))
                        current = current.func.value
                    if not _is_package_files_root(current):
                        continue
                    target = [
                        "src",
                        "taim",
                        *(_resource_segment(arg, constants) for call in calls for arg in call.args),
                    ]
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    operands: list[ast.expr] = []
                    current = node
                    while isinstance(current, ast.BinOp) and isinstance(current.op, ast.Div):
                        operands.insert(0, current.right)
                        nested.add(id(current))
                        current = current.left
                    base = _module_relative_parts(current, module_parts, constants)
                    if base is None:
                        continue
                    target = [*base, *(_resource_segment(item, constants) for item in operands)]
                else:
                    continue
            except _UnresolvedResource:
                expression = cast(ast.expr, node)
                unresolved.append((relative, expression.lineno, ast.unparse(expression)))
                continue
            location = "/".join(target)
            if location.startswith("src/taim/"):
                resolved.append((relative, node.lineno, location))
    return resolved, unresolved


def _validate_packaged_resource_closure(files: Mapping[str, Path]) -> list[str]:
    """Refuse a shipped module that reads package data the release does not ship.

    Import closure follows ``import`` statements only, so a module that loads a packaged file by
    name is invisible to it. Every read whose path the walk resolves must name a shipped file or
    directory. A read the walk cannot follow, such as a variable filename, is returned for the
    validation report: visible, never silently skipped.
    """

    resolved, unresolved = _package_data_reads(files)
    absent = sorted(
        f"{module}:{line} -> {location}"
        for module, line, location in resolved
        if not any(path == location or path.startswith(location + "/") for path in files)
    )
    if absent:
        raise BenchmarkReleaseError(
            "a shipped module reads packaged data the release does not ship: " + "; ".join(absent)
        )
    return sorted(f"{module}:{line}: {expression}" for module, line, expression in unresolved)


_TASK_COMMANDS = {"patient-to-trial": "patient_to_trial", "trial-to-patient": "trial_to_patient"}
_DISPATCH_MODULES = (
    "src/taim/release_cli.py",
    "src/taim/release_reference_systems.py",
    "src/taim/release_result_bundle.py",
)
_SYSTEM_NAMES = frozenset({"system", "system_id"})
_EXTENSION_PREFIX = "src/taim/pipeline_extensions/"
_CLI_SURFACE = r"""
import argparse
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
import taim

if not Path(taim.__file__).resolve().is_relative_to(source):
    raise SystemExit(f"taim resolved outside the release tree: {taim.__file__}")
from taim.release_cli import build_parser
from taim.release_support import SUBCOMMAND_SYSTEMS, SUPPORTED_SYSTEMS, SYSTEM_TASKS


def walk(parser, path):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                yield "command", [*path, name], []
                yield from walk(subparser, [*path, name])
        elif "--system" in action.option_strings:
            yield "system", path, [str(choice) for choice in action.choices or ()]


surface = list(walk(build_parser(), []))
print(json.dumps({
    "supported": list(SUPPORTED_SYSTEMS),
    "system_tasks": sorted(SYSTEM_TASKS),
    "subcommands": [[list(path), system] for path, system in SUBCOMMAND_SYSTEMS.items()],
    "commands": [path for kind, path, _ in surface if kind == "command"],
    "system_options": [[path, choices] for kind, path, choices in surface if kind == "system"],
}))
"""


def _names_system(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id in _SYSTEM_NAMES) or (
        isinstance(node, ast.Attribute) and node.attr in _SYSTEM_NAMES
    )


def _dispatch_value(
    node: ast.expr, local: Mapping[str, str], imported: Mapping[str, str | None]
) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return local.get(node.id) or imported.get(node.id)
    return None


def _extension_modules(relatives: Collection[str]) -> list[str]:
    """The pipeline extension modules among release paths."""

    return sorted(
        relative
        for relative in relatives
        if relative.startswith(_EXTENSION_PREFIX)
        and relative.endswith(".py")
        and relative != _EXTENSION_PREFIX + "__init__.py"
    )


def _strings(value: object) -> bool:
    return isinstance(value, tuple) and all(isinstance(item, str) and item for item in value)


def _literal_field_is_valid(name: str, value: object) -> bool:
    if name in {"forbidden_classes", "protected_reference_paths", "dependency_declarations"}:
        return _strings(value)
    if name in {"module_exports", "forbidden_text"}:
        return isinstance(value, dict) and all(
            isinstance(key, str) and _strings(item) for key, item in value.items()
        )
    if name == "system_tasks":
        return (
            isinstance(value, dict)
            and bool(value)
            and all(isinstance(key, str) and isinstance(item, str) for key, item in value.items())
        )
    if name == "subcommands":
        return isinstance(value, dict) and all(
            isinstance(key, tuple) and len(key) == 2 and _strings(key) and isinstance(item, str)
            for key, item in value.items()
        )
    if name == "protocol_bindings":
        return isinstance(value, tuple) and all(
            isinstance(item, tuple) and len(item) == 3 and _strings(item) for item in value
        )
    if name == "pipeline_depth":
        return isinstance(value, str) and bool(value)
    if name == "ci_jobs":
        return isinstance(value, tuple) and all(
            isinstance(item, dict)
            and set(item) == {"extras", "imports"}
            and _strings(item["extras"])
            and _strings(item["imports"])
            and bool(item["imports"])
            for item in value
        )
    return name == "external_baseline" and isinstance(value, bool)


def _extension_declaration(tree: ast.Module, relative: str) -> dict[str, object]:
    """The literal fields of the one ``EXTENSION = PipelineExtension(...)`` a module assigns."""

    assigned: list[ast.expr | None] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "EXTENSION" for target in targets):
            assigned.append(None if isinstance(node, ast.AugAssign) else node.value)
    call = assigned[0] if len(assigned) == 1 else None
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != "PipelineExtension"
        or call.args
    ):
        raise BenchmarkReleaseError(
            "pipeline extension must assign EXTENSION once, to a PipelineExtension call: "
            + relative
        )
    declaration: dict[str, object] = {}
    for keyword in call.keywords:
        name = keyword.arg
        if name is None or name not in LITERAL_FIELDS:
            continue
        try:
            value: object = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            value = None
        if not _literal_field_is_valid(name, value):
            raise BenchmarkReleaseError(
                f"pipeline extension {name} is not a valid literal: {relative}"
            )
        declaration[name] = value
    if "system_tasks" not in declaration:
        raise BenchmarkReleaseError(f"pipeline extension declares no system_tasks: {relative}")
    return declaration


def _extension_declarations(files: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    """Each shipped pipeline extension module's literal declaration, by release path."""

    declarations: dict[str, dict[str, object]] = {}
    for relative in _extension_modules(files):
        try:
            tree = ast.parse(files[relative].read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise BenchmarkReleaseError(
                f"pipeline extension is not inspectable: {relative}"
            ) from exc
        declarations[relative] = _extension_declaration(tree, relative)
    return declarations


def _dispatch_system_names(
    files: Mapping[str, Path],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """System ids, and id prefixes, the dispatch code compares a System name with, and where.

    Dispatch code is the three core dispatch modules and every shipped pipeline extension module.
    An extension's declared ``system_tasks`` count as dispatched, because the core modules route
    each of those Systems to the extension that declares it.
    """

    trees: dict[str, ast.Module] = {}
    module_constants: dict[str, dict[str, str]] = {}
    for relative, path in files.items():
        if not relative.startswith("src/taim/") or path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        trees[relative] = tree
        module = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        module_constants[module.removesuffix(".__init__")] = _module_string_constants(tree)
    literals: dict[str, list[str]] = {}
    prefixes: dict[str, list[str]] = {}
    extension_modules = _extension_modules(trees)
    for relative in extension_modules:
        declaration = _extension_declaration(trees[relative], relative)
        for system in cast(dict[str, str], declaration["system_tasks"]):
            literals.setdefault(system, []).append(f"{relative}:EXTENSION")
    for relative in (*_DISPATCH_MODULES, *extension_modules):
        dispatch_tree = trees.get(relative)
        if dispatch_tree is None:
            continue
        local = _module_string_constants(dispatch_tree)
        imported = {
            alias.asname or alias.name: module_constants.get(node.module, {}).get(alias.name)
            for node in ast.walk(dispatch_tree)
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
            for alias in node.names
        }
        for node in ast.walk(dispatch_tree):
            if isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                if not any(_names_system(operand) for operand in operands):
                    continue
                for operand in operands:
                    if _names_system(operand):
                        continue
                    elements = (
                        operand.elts
                        if isinstance(operand, ast.Tuple | ast.Set | ast.List)
                        else [operand]
                    )
                    for element in elements:
                        found = _dispatch_value(element, local, imported)
                        if found is not None:
                            literals.setdefault(found, []).append(f"{relative}:{node.lineno}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "startswith"
                and _names_system(node.func.value)
                and node.args
            ):
                found = _dispatch_value(node.args[0], local, imported)
                if found is not None:
                    prefixes.setdefault(found, []).append(f"{relative}:{node.lineno}")
    return literals, prefixes


def _catalogue_cli_dispatch_findings(candidate: Path, files: Mapping[str, Path]) -> list[str]:
    """Every way the catalogue, the CLI's System surface and the dispatch branches disagree."""

    catalogue_path = files.get("src/taim/data/release-support-v0.1.json")
    if catalogue_path is None:
        return ["release omits its executable support catalog"]
    by_task: dict[str, set[str]] = {}
    for row in json.loads(catalogue_path.read_text(encoding="utf-8")).get("support_matrix", []):
        by_task.setdefault(row["task"], set()).add(row["system"])
    catalogue = {system for systems in by_task.values() for system in systems}
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-B", "-E", "-s", "-c", _CLI_SURFACE, str(candidate / "src")],
        cwd=candidate,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = (process.stderr.strip().splitlines() or ["no output"])[-1]
        return [f"the release CLI and support declarations cannot be read: {detail}"]
    surface = json.loads(process.stdout)
    findings: list[str] = []
    for label, declared in (
        ("SUPPORTED_SYSTEMS", set(surface["supported"])),
        ("SYSTEM_TASKS", set(surface["system_tasks"])),
    ):
        if declared != catalogue:
            findings.append(
                f"{label} does not name the catalogue's Systems (only in {label}: "
                f"{sorted(declared - catalogue)}; only in the catalogue: "
                f"{sorted(catalogue - declared)})"
            )
    commands = {tuple(path) for path in surface["commands"]}
    reachable: dict[str, set[str]] = {}
    bound: set[str] = set()
    for path, system in surface["subcommands"]:
        key = tuple(path)
        task = _TASK_COMMANDS.get(key[0]) if key else None
        if key not in commands or task is None:
            findings.append(
                f"SUBCOMMAND_SYSTEMS binds {system} to '{' '.join(key)}', "
                "which is no task subcommand of the CLI"
            )
            continue
        reachable.setdefault(task, set()).add(system)
        bound.add(system)
    for path, choices in surface["system_options"]:
        where = " ".join(path) or "the top-level command"
        if not choices:
            findings.append(f"--system at '{where}' declares no choices")
        task = _TASK_COMMANDS.get(path[0]) if path else None
        if task is None:
            findings.append(f"--system at '{where}' is outside the task subcommands")
            continue
        reachable.setdefault(task, set()).update(choices)
    for task in sorted(set(by_task) | set(reachable)):
        offered, declared = reachable.get(task, set()), by_task.get(task, set())
        if offered != declared:
            findings.append(
                f"the CLI and the catalogue disagree for Task {task} (only in the CLI: "
                f"{sorted(offered - declared)}; only in the catalogue: "
                f"{sorted(declared - offered)})"
            )
    literals, prefixes = _dispatch_system_names(files)
    for system in sorted(set(literals) - catalogue):
        findings.append(
            f"dispatch compares with a System outside the catalogue: {system} at "
            + ", ".join(literals[system])
        )
    for prefix in sorted(prefixes):
        if not any(system.startswith(prefix) for system in catalogue):
            findings.append(
                f"dispatch prefix matches no catalogue System: {prefix} at "
                + ", ".join(prefixes[prefix])
            )
    for system in sorted(catalogue):
        if (
            system not in literals
            and system not in bound
            and not any(system.startswith(prefix) for prefix in prefixes)
        ):
            findings.append(f"no dispatch branch or subcommand runs the catalogue System {system}")
    return findings


def _validate_catalogue_cli_dispatch(candidate: Path, files: Mapping[str, Path]) -> None:
    """Refuse a release whose catalogue, CLI System choices and dispatch branches disagree.

    The support catalogue promises Systems; the CLI's ``--system`` choices and its declared
    subcommand Systems are what a user can reach; the dispatch branches are what actually runs. A
    release that filtered one of the three and not the others ships a promise it cannot keep, or
    a branch for a System it no longer declares.
    """

    findings = _catalogue_cli_dispatch_findings(candidate, files)
    if findings:
        raise BenchmarkReleaseError("catalogue, CLI and dispatch disagree: " + "; ".join(findings))


def _validate_extension_module_surfaces(
    files: Mapping[str, Path], declarations: Mapping[str, Mapping[str, object]]
) -> None:
    """Modules a shipped extension exports from expose exactly the declared names, and nothing it
    excludes."""

    expected: dict[str, set[str]] = {}
    forbidden_classes: set[str] = set()
    forbidden_text: dict[str, set[str]] = {}
    for declaration in declarations.values():
        exports = cast(dict[str, tuple[str, ...]], declaration.get("module_exports", {}))
        for module, names in exports.items():
            expected.setdefault(module, set()).update(names)
        forbidden_classes.update(cast(tuple[str, ...], declaration.get("forbidden_classes", ())))
        excluded = cast(dict[str, tuple[str, ...]], declaration.get("forbidden_text", {}))
        for module, texts in excluded.items():
            forbidden_text.setdefault(module, set()).update(texts)
    for relative in sorted(expected):
        path = files.get(relative)
        if path is None:
            raise BenchmarkReleaseError(
                f"a module a pipeline extension exports from is absent: {relative}"
            )
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise BenchmarkReleaseError(
                f"extension-exported module is not inspectable: {relative}"
            ) from exc
        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        leaked = sorted(classes & forbidden_classes)
        if leaked:
            raise BenchmarkReleaseError(
                f"extension-exported module defines excluded classes in {relative}: "
                + ", ".join(leaked)
            )
        exports_value: object = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                try:
                    exports_value = ast.literal_eval(node.value)
                except ValueError as exc:
                    raise BenchmarkReleaseError(
                        f"extension-exported module exports are not static: {relative}"
                    ) from exc
        if (
            not isinstance(exports_value, list)
            or set(exports_value) != expected[relative]
            or len(exports_value) != len(expected[relative])
        ):
            raise BenchmarkReleaseError(
                f"extension-exported module exports do not match the declared surface: {relative}"
            )
    for relative, excluded_texts in sorted(forbidden_text.items()):
        path = files.get(relative)
        if path is None:
            raise BenchmarkReleaseError(
                f"a module a pipeline extension constrains is absent: {relative}"
            )
        found = sorted(item for item in excluded_texts if item in path.read_text(encoding="utf-8"))
        if found:
            raise BenchmarkReleaseError(
                f"{relative} contains text a pipeline extension excludes: " + ", ".join(found)
            )


_PROTECTED_TRACK = re.compile(
    r"(?:trec[-_/ ]?(?:ct[-_/ ]?)?|clinical[-_/ ]trials[-_/ ]?)202[23]",
    re.I,
)
# Core files that may name the protected TREC 2022 and 2023 Tracks. A file a pipeline owns is
# declared by that pipeline's extension instead, so each list ships with the files it names.
_CORE_PROTECTED_REFERENCE_PATHS = frozenset(
    {
        "README.md",
        RELEASE_PROVENANCE_FILENAME,
        "THIRD_PARTY_NOTICES.md",
        "docs/compatibility.md",
        "docs/method-contracts.md",
        "docs/paper-analysis-protocol-2026-08-29.md",
        "docs/paper-analysis-protocol-2026-08-31-v2.md",
        "docs/paper-analysis-protocol-2026-08-31-v3.md",
        "docs/paper-analysis-protocol-2026-08-31-v4.md",
        "docs/reporting-policy.md",
        "docs/trec-2022-acquisition.md",
        "docs/trec-2023-acquisition.md",
        "src/taim/benchmark_release.py",
        "src/taim/data/locks/trec-ct-2022.json",
        "src/taim/data/locks/trec-ct-2023.json",
        "src/taim/data/profiles/trec-ct-2022-judgment-union-v1.json",
        "src/taim/data/profiles/trec-ct-2023-judgment-union-v1.json",
        "src/taim/data/release-support-v0.1.json",
        "src/taim/data/trec_ct.py",
        "src/taim/release_cli.py",
        "src/taim/release_profiles.py",
        "src/taim/release_support.py",
        "tests/test_bundle_roundtrip.py",
        "tests/test_connectors.py",
        "tests/test_quick_start.py",
    }
)


def _protected_reference_paths(files: Mapping[str, Path]) -> frozenset[str]:
    """The files that may name the protected Tracks: the core list and each shipped extension's."""

    return _CORE_PROTECTED_REFERENCE_PATHS | {
        path
        for declaration in _extension_declarations(files).values()
        for path in cast(tuple[str, ...], declaration.get("protected_reference_paths", ()))
    }


def _validate_no_emitted_markers(files: Mapping[str, Path]) -> None:
    """Refuse a release file that still carries a source marker: markers belong in sources.

    A Markdown marker is a line that is a pipeline section's opening or closing marker, or a table
    row ending in a pipeline-row cell; a Python marker is a line of code ending in a ships-with
    comment.
    """

    found: list[str] = []
    for relative, path in sorted(files.items()):
        if path.suffix not in {".md", ".py"}:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if (
                _SECTION_OPEN.fullmatch(line)
                or line == _SECTION_CLOSE
                or _ROW_MARKER.fullmatch(line)
                or (path.suffix == ".py" and _SHIPS_WITH.fullmatch(line))
            ):
                found.append(f"{relative}:{number}")
    if found:
        raise BenchmarkReleaseError("release files carry source markers: " + ", ".join(found))


def _validate_protected_reference_paths(files: Mapping[str, Path]) -> None:
    """Refuse an extension's protected-reference path that this release does not ship.

    An extension's list ships with the extension, so every file it names must ship too. The core
    list names only files no pipeline owns, which the release template's tests hold it to.
    """

    unshipped = sorted(
        _protected_reference_paths(files) - _CORE_PROTECTED_REFERENCE_PATHS - set(files)
    )
    if unshipped:
        raise BenchmarkReleaseError(
            "a pipeline extension names protected-track reference paths the release does not ship: "
            + ", ".join(unshipped)
        )


def _validate_forbidden_content(
    files: Mapping[str, Path], manifest_files: Mapping[str, Mapping[str, object]]
) -> None:
    secret_patterns = (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"ghp_[A-Za-z0-9]{30,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile("-----BEGIN " + "PRIVATE KEY-----"),
        re.compile(r"Authorization:\s*Bearer\s+[A-Za-z0-9._-]+", re.I),
    )
    personal_path = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")
    safe_protected_reference_paths = _protected_reference_paths(files)
    for relative, path in files.items():
        lowered = relative.casefold()
        declaration = manifest_files.get(relative)
        synthetic_qrels = (
            path.name.casefold() == "qrels.jsonl"
            and declaration is not None
            and declaration["data_classification"] == "synthetic_fixture"
        )
        if path.suffix.casefold() in _PROHIBITED_SUFFIXES or (
            path.name.casefold() in _PROHIBITED_FILENAMES and not synthetic_qrels
        ):
            raise BenchmarkReleaseError(f"release contains a prohibited artifact: {relative}")
        if (
            path.suffix.casefold() not in _INSPECTABLE_TEXT_SUFFIXES
            and path.name not in _INSPECTABLE_TEXT_FILENAMES
        ):
            raise BenchmarkReleaseError(
                f"release artifact is not an inspectable UTF-8 text format: {relative}"
            )
        if _PROTECTED_TRACK.search(relative) and relative not in safe_protected_reference_paths:
            raise BenchmarkReleaseError(f"release contains a protected-track path: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise BenchmarkReleaseError(
                f"release artifact is not inspectable UTF-8 text: {relative}"
            ) from exc
        if any(ord(character) < 32 and character not in "\n\r\t" for character in text):
            raise BenchmarkReleaseError(
                f"release artifact is not inspectable UTF-8 text: {relative}"
            )
        if text and (
            personal_path.search(text) or any(pattern.search(text) for pattern in secret_patterns)
        ):
            raise BenchmarkReleaseError(
                f"release contains secret or private-path content: {relative}"
            )
        if _PROTECTED_TRACK.search(text) and relative not in safe_protected_reference_paths:
            raise BenchmarkReleaseError(f"release contains protected-track content: {relative}")
        if declaration is None:
            continue
        synthetic = declaration["data_classification"] == "synthetic_fixture"
        if not synthetic and any(_TREC_QRELS_ROW.fullmatch(line) for line in text.splitlines()):
            raise BenchmarkReleaseError(f"release contains raw Judgment rows: {relative}")
        if declaration["data_classification"] != "published_result_bundle":
            continue
        if path.suffix.casefold() in {".csv", ".tsv", ".txt"} or (
            path.suffix.casefold() == ".md" and path.name != "scorecard.md"
        ):
            raise BenchmarkReleaseError(
                f"Published Result Bundle contains unstructured text: {relative}"
            )
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping):
                continue
            keys = set(row)
            if {"topic_id", "trial_id", "label"} <= keys:
                raise BenchmarkReleaseError(
                    f"Published Result Bundle contains raw Judgment rows: {relative}"
                )
            if (
                "patient_text" in keys
                or "trial_text" in keys
                or ("canonical_text" in keys and ({"topic_id", "trial_id"} & keys))
            ):
                raise BenchmarkReleaseError(
                    f"Published Result Bundle contains prohibited clinical text: {relative}"
                )
        if any(token in lowered for token in ("model", "index", "provider", "credential")):
            raise BenchmarkReleaseError(
                f"Published Result Bundle path contradicts redistribution policy: {relative}"
            )


def _commit_parents(candidate: Path, revision: str) -> list[str]:
    """The parent lines of a commit object, read from the object itself.

    ``rev-list --parents`` shows no parent at a shallow clone's boundary, while the object keeps
    its parent lines, so a shallow clone of a chained release checks the same link.
    """

    body = _git(candidate, ("cat-file", "commit", revision))
    parents: list[str] = []
    for line in body.splitlines():
        if not line:
            break
        if line.startswith("parent "):
            parents.append(line.removeprefix("parent "))
    return parents


def _validate_git_history(
    candidate: Path,
    manifest: Mapping[str, object],
    expected_files: set[str],
) -> str:
    """The public commit-identity rule: what any checkout of the release commit satisfies.

    It checks the commit's tracked tree, metadata and parent link, so the builder's candidate after
    chaining, a plain clone with ``origin``, a detached tag checkout and a chained release all
    pass. Whether a repository is wired to a destination, which refs it holds and which branch is
    checked out are properties of a working repository, not of the release commit; the private
    admission checks them at each act that could publish (``admit_release_candidate``).
    """

    if not (candidate / ".git").is_dir():
        raise BenchmarkReleaseError("release candidate lacks public Git history")
    project = cast(Mapping[str, object], manifest["project"])
    release_commit = cast(Mapping[str, object], project["release_commit"])
    previous = release_commit["previous_public_commit"]
    if _commit_parents(candidate, "HEAD") != ([] if previous is None else [previous]):
        raise BenchmarkReleaseError("public release commit parent does not match the manifest")
    if _git(candidate, ("status", "--porcelain", "--untracked-files=all")):
        raise BenchmarkReleaseError("public release worktree is dirty")
    tracked = set(_git(candidate, ("ls-files",)).splitlines())
    if tracked != expected_files:
        raise BenchmarkReleaseError("public Git commit does not track the exact release tree")
    details = _git(
        candidate,
        ("show", "-s", "--format=%s%n%an%n%ae%n%aI%n%cn%n%ce%n%cI", "HEAD"),
    ).splitlines()
    expected_timestamp = _aware_datetime(
        release_commit["timestamp"], role="release commit timestamp"
    )
    if len(details) != 7:
        raise BenchmarkReleaseError("public release commit metadata does not match the manifest")
    try:
        author_timestamp = datetime.fromisoformat(details[3])
        committer_timestamp = datetime.fromisoformat(details[6])
    except ValueError as exc:
        raise BenchmarkReleaseError(
            "public release commit metadata does not match the manifest"
        ) from exc
    expected_identity = [
        release_commit["message"],
        release_commit["author_name"],
        release_commit["author_email"],
        release_commit["author_name"],
        release_commit["author_email"],
    ]
    actual_identity = [details[0], details[1], details[2], details[4], details[5]]
    timestamps_match = all(
        timestamp.replace(tzinfo=None) == expected_timestamp.replace(tzinfo=None)
        and timestamp.utcoffset() == expected_timestamp.utcoffset()
        for timestamp in (author_timestamp, committer_timestamp)
    )
    if actual_identity != expected_identity or not timestamps_match:
        raise BenchmarkReleaseError("public release commit metadata does not match the manifest")
    return _git(candidate, ("rev-parse", "HEAD"))


def _admit_strict_tree(candidate: Path, manifest: Mapping[str, object]) -> None:
    expected = {
        cast(str, entry["destination"])
        for entry in cast(list[Mapping[str, object]], manifest["files"])
    } | {RELEASE_PROVENANCE_FILENAME}
    ignored = sorted(set(_candidate_files(candidate, honour_ignore=False)) - expected)
    if ignored:
        raise BenchmarkReleaseError(
            "public release candidate holds paths outside the release tree: " + ", ".join(ignored)
        )


def admit_release_tree(
    candidate_directory: str | Path, manifest_path: str | Path
) -> dict[str, object]:
    """Private admission of a built tree before ``taim release chain`` gives it history.

    Admission concerns publication wiring and refs. It inspects no content beyond the public
    validation it runs first, and a published clone cannot re-run it.
    """

    candidate = Path(candidate_directory).resolve()
    report = validate_benchmark_release(candidate, manifest_path, require_git=False)
    if (candidate / ".git").exists():
        raise BenchmarkReleaseError("release candidate must not carry Git history before chain")
    _admit_strict_tree(candidate, load_benchmark_release_manifest(manifest_path))
    return report


def admit_release_candidate(
    candidate_directory: str | Path,
    manifest_path: str | Path,
    *,
    allowed_refs: Collection[str] = (),
    recorded_commits: Collection[str] = (),
) -> dict[str, object]:
    """Private admission of a chained candidate, run by chain and publish-plan as each one acts.

    On top of the public commit-identity rule, the candidate must hold only the release tree
    (strict walk), must not be wired to any destination, must hold exactly its release branch and
    ``allowed_refs``, must have that branch checked out, and every ancestor of the release commit
    must be one of ``recorded_commits``: the public commits the private ledger records.

    Admission concerns publication wiring and refs. It inspects no content beyond the public
    validation it runs first, and a published clone cannot re-run it.
    """

    candidate = Path(candidate_directory).resolve()
    manifest = load_benchmark_release_manifest(manifest_path)
    report = validate_benchmark_release(candidate, manifest_path)
    _admit_strict_tree(candidate, manifest)
    local_config = _git(candidate, ("config", "--local", "--list")).splitlines()
    if any(line.startswith("remote.") for line in local_config):
        raise BenchmarkReleaseError("public release candidate must not create a remote")
    release_commit = cast(
        Mapping[str, object], cast(Mapping[str, object], manifest["project"])["release_commit"]
    )
    branch = cast(str, release_commit["branch"])
    refs = set(_git(candidate, ("for-each-ref", "--format=%(refname)")).splitlines())
    foreign = sorted(refs - {f"refs/heads/{branch}", *allowed_refs})
    if foreign:
        raise BenchmarkReleaseError(
            "public release candidate holds refs other than its release refs: " + ", ".join(foreign)
        )
    if _git(candidate, ("branch", "--show-current")) != branch:
        raise BenchmarkReleaseError("public default branch does not match the manifest")
    ancestors = (
        _git(candidate, ("rev-list", "HEAD^@")).splitlines()
        if release_commit["previous_public_commit"]
        else []
    )
    unrecorded = sorted(set(ancestors) - set(recorded_commits))
    if unrecorded:
        raise BenchmarkReleaseError(
            "public release history holds commits the ledger does not record: "
            + ", ".join(unrecorded)
        )
    return report


def validate_benchmark_release(
    candidate_directory: str | Path,
    manifest_path: str | Path,
    *,
    require_git: bool = True,
    execute_quick_start: bool = False,
    quick_start_executable: str | Path | None = None,
) -> dict[str, object]:
    """Validate a generated tree from disk without trusting builder state."""

    candidate = Path(candidate_directory).resolve()
    if not candidate.is_dir():
        raise BenchmarkReleaseError(f"release candidate is not a directory: {candidate}")
    manifest = load_benchmark_release_manifest(manifest_path)
    manifest_entries = {
        cast(str, entry["destination"]): entry
        for entry in cast(list[Mapping[str, object]], manifest["files"])
    }
    expected_files = set(manifest_entries) | {RELEASE_PROVENANCE_FILENAME}
    actual_files = _candidate_files(candidate)
    if set(actual_files) != expected_files:
        extra = sorted(set(actual_files) - expected_files)
        missing = sorted(expected_files - set(actual_files))
        details = []
        if extra:
            details.append("extra: " + ", ".join(extra))
        if missing:
            details.append("missing: " + ", ".join(missing))
        raise BenchmarkReleaseError("release tree is not closed (" + "; ".join(details) + ")")
    records: list[dict[str, object]] = []
    for relative, path in sorted(actual_files.items()):
        digest = _sha256_file(path)
        size = path.stat().st_size
        if relative in manifest_entries:
            entry = manifest_entries[relative]
            if digest != entry["sha256"] or size != entry["byte_size"]:
                raise BenchmarkReleaseError(f"release file content does not match: {relative}")
        records.append({"path": relative, "sha256": digest, "byte_size": size})
    expected_provenance = _json_bytes(_provenance(manifest))
    if actual_files[RELEASE_PROVENANCE_FILENAME].read_bytes() != expected_provenance:
        raise BenchmarkReleaseError("release provenance does not match the frozen source mapping")
    tree_id = _public_tree_id(records)
    if tree_id != manifest["expected_public_tree_id"]:
        raise BenchmarkReleaseError("public tree identity does not match the release manifest")
    project = cast(Mapping[str, object], manifest["project"])
    _validate_package_metadata(candidate, project)
    _validate_license_closure(actual_files, manifest)
    _validate_support_catalog(actual_files, manifest)
    _validate_catalogue_cli_dispatch(candidate, actual_files)
    _validate_public_import_closure(actual_files)
    unresolved_resource_reads = _validate_packaged_resource_closure(actual_files)
    _validate_extension_module_surfaces(actual_files, _extension_declarations(actual_files))
    _validate_document_links(candidate, actual_files)
    _validate_no_emitted_markers(actual_files)
    _validate_protected_reference_paths(actual_files)
    _validate_forbidden_content(actual_files, manifest_entries)
    _validate_published_result_bundles(candidate, manifest)
    public_commit = (
        _validate_git_history(candidate, manifest, expected_files) if require_git else "UNCOMMITTED"
    )
    if execute_quick_start:
        commands = cast(list[list[str]], manifest["quick_start"])
        executable = (
            str(Path(quick_start_executable).resolve())
            if quick_start_executable is not None
            else cast(str, cast(Mapping[str, object], manifest["project"])["public_cli"])
        )
        with tempfile.TemporaryDirectory(prefix="taim-public-quick-start-") as workspace:
            for command in commands:
                process = subprocess.run(  # noqa: S603
                    [executable, *command[1:]],
                    cwd=workspace,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if process.returncode != 0:
                    detail = process.stderr.strip() or process.stdout.strip()
                    raise BenchmarkReleaseError(
                        f"public quick start failed at {' '.join(command)}: {detail}"
                    )
    return {
        "artifact_type": "taim-benchmark-release-validation",
        "schema_version": "1.0",
        "manifest_id": manifest["manifest_id"],
        "source_commit": manifest["source_commit"],
        "public_tree_id": tree_id,
        "public_commit": public_commit,
        "file_count": len(records),
        "published_pipelines": manifest["published_pipelines"],
        "published_result_bundle_ids": [
            entry["bundle_id"]
            for entry in cast(list[Mapping[str, object]], manifest["published_result_bundles"])
        ],
        "quick_start_executed": execute_quick_start,
        "unresolved_packaged_resource_reads": cast(list[object], unresolved_resource_reads),
    }


__all__ = [
    "RELEASE_PROVENANCE_FILENAME",
    "BenchmarkReleaseCandidate",
    "BenchmarkReleaseError",
    "BenchmarkReleasePackageArtifact",
    "admit_release_candidate",
    "admit_release_tree",
    "build_benchmark_release",
    "freeze_benchmark_release_manifest",
    "load_benchmark_release_manifest",
    "validate_benchmark_release",
    "validate_release_package_artifact",
]
