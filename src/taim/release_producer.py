"""Verify the executing release producer and identities of component runs."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Protocol, cast

from taim import __version__
from taim.benchmark_release import (
    load_benchmark_release_manifest,
    validate_release_package_artifact,
)
from taim.contracts import content_sha256
from taim.file_hash import sha256_file
from taim.pipeline_extensions import extension_for
from taim.release_result_bundle import validate_dependency_environment
from taim.schemas import JsonValue


class ReleaseProducerInput(Protocol):
    release_manifest: Path | None
    package_artifact: Path | None
    dependency_lock: Path | None


def _installed_package_files() -> dict[str, tuple[str, int]]:
    package_root = Path(__file__).resolve().parent
    records: dict[str, tuple[str, int]] = {}
    for path in sorted(package_root.rglob("*")):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"executing TAIM package contains a symlink: {relative.as_posix()}")
        if path.is_file():
            records[relative.as_posix()] = (sha256_file(path), path.stat().st_size)
    return records


def _distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _installed_distributions() -> list[dict[str, str]]:
    installed: dict[str, str] = {}
    for distribution in metadata.distributions():
        raw_name = distribution.metadata["Name"]
        if not raw_name:
            raise ValueError("installed Python distribution lacks its canonical name")
        name = _distribution_name(raw_name)
        version = distribution.version
        existing = installed.get(name)
        if existing is not None and existing != version:
            raise ValueError(f"installed Python distribution {name!r} has conflicting versions")
        installed[name] = version
    return [{"name": name, "version": version} for name, version in sorted(installed.items())]


def _locked_distribution_versions(path: Path) -> dict[str, frozenset[str]]:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("release dependency lock is not readable TOML") from exc
    packages = payload.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("release dependency lock has no package records")
    versions: dict[str, set[str]] = {}
    for raw in packages:
        if not isinstance(raw, Mapping):
            raise ValueError("release dependency lock package record is invalid")
        name = raw.get("name")
        version = raw.get("version")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise ValueError("release dependency lock package identity is invalid")
        versions.setdefault(_distribution_name(name), set()).add(version)
    return {name: frozenset(values) for name, values in versions.items()}


def _manifest_dependency_lock(args: ReleaseProducerInput) -> Path:
    lock_path = cast(Path, args.dependency_lock)
    manifest = load_benchmark_release_manifest(cast(Path, args.release_manifest))
    records = [
        raw
        for raw in cast(list[Mapping[str, object]], manifest["files"])
        if raw.get("destination") == "uv.lock"
    ]
    if len(records) != 1:
        raise ValueError("release manifest must contain exactly one uv.lock")
    record = records[0]
    if (
        record.get("sha256") != sha256_file(lock_path)
        or record.get("byte_size") != lock_path.stat().st_size
    ):
        raise ValueError("release dependency lock does not match the supplied release manifest")
    return lock_path


def verified_dependency_environment(
    args: ReleaseProducerInput,
    *,
    system_id: str,
) -> dict[str, JsonValue]:
    lock_path = _manifest_dependency_lock(args)
    locked = _locked_distribution_versions(lock_path)
    distributions = _installed_distributions()
    installed = {row["name"]: row["version"] for row in distributions}
    for name, version in installed.items():
        allowed = locked.get(name)
        if allowed is not None and version not in allowed:
            raise ValueError(
                f"installed Python distribution {name}=={version} is not resolved by uv.lock"
            )
    required: set[str] = set()
    extension = extension_for(system_id)
    if extension is not None:
        required.update(extension.required_distributions)
    missing = sorted(name for name in required if name not in installed or name not in locked)
    if missing:
        raise ValueError(
            f"release System {system_id!r} lacks locked installed distributions: "
            + ", ".join(missing)
        )
    core: dict[str, JsonValue] = {
        "schema_version": "1.0",
        "dependency_lock_sha256": sha256_file(lock_path),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "distributions": cast(JsonValue, distributions),
    }
    return validate_dependency_environment({**core, "environment_id": content_sha256(core)})


def verified_release_identity(args: ReleaseProducerInput) -> dict[str, str] | None:
    manifest_path = args.release_manifest
    package_artifact = args.package_artifact
    dependency_lock = args.dependency_lock
    if manifest_path is None and package_artifact is None and dependency_lock is None:
        return None
    if manifest_path is None or package_artifact is None or dependency_lock is None:
        raise ValueError(
            "release producer identity requires --release-manifest, --package-artifact, and "
            "--dependency-lock together"
        )
    verified = validate_release_package_artifact(manifest_path, package_artifact)
    archived = {
        relative: (digest, byte_size) for relative, digest, byte_size in verified.package_files
    }
    if _installed_package_files() != archived:
        raise ValueError("executing TAIM package does not match the supplied release artifact")
    return verified.release_identity()


def release_source_commit(args: ReleaseProducerInput) -> str:
    if args.release_manifest is None:
        return git_state()[0]
    manifest = load_benchmark_release_manifest(cast(Path, args.release_manifest))
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("release manifest source commit is invalid")
    return source_commit


def _package_repository() -> Path | None:
    module = Path(__file__).resolve()
    for candidate in module.parents:
        expected = candidate / "src" / "taim" / "release_producer.py"
        if (candidate / ".git").exists() and expected.exists() and expected.resolve() == module:
            return candidate
    return None


def git_state() -> tuple[str, bool]:
    git = shutil.which("git")
    repository = _package_repository()
    if git is None or repository is None:
        return (f"packaged-taim-{__version__}", False)
    commit = subprocess.run(  # noqa: S603
        [git, "rev-parse", "--verify", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(  # noqa: S603
        [git, "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode or status.returncode:
        return (f"packaged-taim-{__version__}", False)
    return (commit.stdout.strip(), bool(status.stdout.strip()))


def revalidate_release_producer(
    args: ReleaseProducerInput,
    *,
    system_id: str,
    release_identity: Mapping[str, str] | None,
    dependency_environment: Mapping[str, JsonValue] | None,
    expected_git_state: tuple[str, bool],
) -> tuple[str, bool]:
    if release_identity is None:
        return git_state()
    current_identity = verified_release_identity(args)
    current_environment = verified_dependency_environment(args, system_id=system_id)
    current_git_state = git_state()
    if (
        current_identity != release_identity
        or current_environment != dependency_environment
        or current_git_state != expected_git_state
    ):
        raise ValueError("release producer identity changed during benchmark execution")
    return current_git_state


def rrf_component_producer_identity(
    component,
    *,
    expected_release_identity: Mapping[str, str],
    expected_lock_sha256: str,
) -> dict[str, JsonValue]:
    manifest = component.manifest
    execution = manifest.configuration.get("execution_provenance")
    if not isinstance(execution, Mapping) or execution.get("git_commit") != manifest.git_commit:
        raise ValueError("RRF component producer commit does not match its Run Manifest")
    if execution.get("working_tree_dirty") is not False:
        raise ValueError("RRF component producer must be a clean checkout")
    release_identity = manifest.configuration.get("producer_release_identity")
    if release_identity != expected_release_identity:
        raise ValueError("RRF component producer release identity does not match")
    dependency_environment = validate_dependency_environment(
        manifest.configuration.get("producer_dependency_environment")
    )
    if dependency_environment["dependency_lock_sha256"] != expected_lock_sha256:
        raise ValueError("RRF component dependency lock does not match the fusion producer")
    return {
        "git_commit": manifest.git_commit,
        "working_tree_dirty": False,
        "release_identity": cast(JsonValue, dict(expected_release_identity)),
        "dependency_environment": dependency_environment,
    }
