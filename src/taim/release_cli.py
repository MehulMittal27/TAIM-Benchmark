"""Public command line for Benchmark Release 0.1."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import cast

from taim import __version__
from taim.benchmark_release import (
    load_benchmark_release_manifest,
    validate_release_package_artifact,
)
from taim.contracts import content_sha256, require_sha256
from taim.data import PreparedBenchmark, load_prepared_benchmark
from taim.entity_versions import parse_clinical_as_of, source_grounded_patient_evidence_profile
from taim.file_hash import sha256_file
from taim.pipeline_extensions import ComponentRun, extension_for, pipeline_extensions
from taim.public_patient_to_trial import (
    evaluate_public_patient_to_trial_run,
    load_local_evaluation_package,
    load_public_patient_to_trial_run,
    write_public_patient_to_trial_run,
)
from taim.release_fixture import FIXTURE_CLINICAL_AS_OF, load_release_fixture
from taim.release_profiles import (
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
    inspect_external_fidelity_profile,
    inspect_judgment_union_profile,
    load_profile,
    ordered_trial_ids_sha256,
    resolve_benchmark_profile,
)
from taim.release_reference_systems import run_release_reference_system
from taim.release_result_bundle import validate_dependency_environment
from taim.release_support import (
    RELEASE_VERSION,
    SUBCOMMAND_SYSTEMS,
    SUPPORTED_TRACKS,
    SYSTEM_TASKS,
    load_release_support,
    release_source_identity,
    require_frozen_effectiveness_protocol,
    require_frozen_paper_protocol,
    require_supported_combination,
    requires_effectiveness_protocol,
    source_profile_id,
    validate_public_reverse_run_tree,
    validate_reverse_release_manifest,
)
from taim.reverse_fixture import load_reverse_fixture
from taim.reverse_viability import qualify_reverse_viability_profile
from taim.schemas import (
    JsonValue,
    RunManifest,
    StageRanking,
)
from taim.submission_cli import add_submission_commands
from taim.system_contracts import SystemRunRequest
from taim.trial_to_patient import (
    StoredTrialToPatientRun,
    TrialToPatientEvaluationPackage,
    TrialToPatientExecutionProvenance,
    TrialToPatientRunRequest,
    evaluate_trial_to_patient_run,
    load_trial_to_patient_run,
    run_trial_to_patient_system,
    write_trial_to_patient_run,
)

DEFAULT_RETRIEVAL_DEPTH = 1_000
DEFAULT_METRIC_CUTOFF = 5


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _sha256_identity(value: str) -> str:
    try:
        require_sha256(value, "SHA-256 identity")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


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


def _manifest_dependency_lock(args: argparse.Namespace) -> Path:
    lock_path = cast(Path, args.dependency_lock)
    manifest = load_benchmark_release_manifest(args.release_manifest)
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


def _verified_dependency_environment(
    args: argparse.Namespace,
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


def _verified_release_identity(args: argparse.Namespace) -> dict[str, str] | None:
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


def _release_source_commit(args: argparse.Namespace) -> str:
    if args.release_manifest is None:
        return _git_state()[0]
    manifest = load_benchmark_release_manifest(args.release_manifest)
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("release manifest source commit is invalid")
    return source_commit


def _package_repository() -> Path | None:
    module = Path(__file__).resolve()
    for candidate in module.parents:
        expected = candidate / "src" / "taim" / "release_cli.py"
        if (candidate / ".git").exists() and expected.exists() and expected.resolve() == module:
            return candidate
    return None


def _git_state() -> tuple[str, bool]:
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


def _run_id(system_id: str) -> str:
    return f"public-{system_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"


def _print(payload: object) -> None:
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


def _support_list(_args: argparse.Namespace) -> int:
    _print(load_release_support())
    return 0


def _read_json_object(path: Path, role: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {role}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must contain a JSON object")
    return payload


def _read_pool_ids(path: Path | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    try:
        if path.suffix.casefold() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("JSON pool file must contain an array")
            values = tuple(payload)
        else:
            values = tuple(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read external-fidelity pool file: {exc}") from exc
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("external-fidelity pool file must contain non-empty trial IDs")
    if len(values) != len(set(values)):
        raise ValueError("external-fidelity pool file contains duplicate trial IDs")
    return cast(tuple[str, ...], values)


def _verified_external_pool_ids(
    source: Path,
    *,
    expected_byte_size: int,
    expected_sha256: str,
    expected_count: int,
    expected_sorted_ids_sha256: str,
) -> tuple[str, ...]:
    if source.stat().st_size != expected_byte_size or sha256_file(source) != expected_sha256:
        raise ValueError("external-fidelity source does not match its packaged checksum lock")
    trial_ids: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"external-fidelity source record {line_number} is invalid JSON"
                ) from exc
            trial_id = record.get("_id") if isinstance(record, Mapping) else None
            if not isinstance(trial_id, str) or not trial_id:
                raise ValueError(f"external-fidelity source record {line_number} lacks a trial ID")
            trial_ids.append(trial_id)
    ordered = tuple(sorted(trial_ids))
    if len(ordered) != expected_count or len(set(ordered)) != expected_count:
        raise ValueError("external-fidelity source record count or ID uniqueness changed")
    if ordered_trial_ids_sha256(ordered) != expected_sorted_ids_sha256:
        raise ValueError("external-fidelity source trial-ID membership changed")
    return ordered


def _prepare_external_pool(args: argparse.Namespace) -> int:
    if args.profile != EXTERNAL_FIDELITY_TREC_2021_PROFILE:
        raise ValueError("Release 0.1 has one external-fidelity source Profile")
    lock_path = (
        Path(__file__).parent / "data" / "locks" / "trialgpt-trec-2021-external-fidelity.json"
    )
    lock = _read_json_object(lock_path, "external-fidelity source lock")
    source = cast(Mapping[str, object], lock["source"])
    membership = cast(Mapping[str, object], lock["membership"])
    trial_ids = _verified_external_pool_ids(
        args.source,
        expected_byte_size=cast(int, source["byte_size"]),
        expected_sha256=cast(str, source["sha256"]),
        expected_count=cast(int, membership["record_count"]),
        expected_sorted_ids_sha256=cast(str, membership["sorted_unique_trial_ids_sha256"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(trial_ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _print(
        {
            "release": RELEASE_VERSION,
            "track": "trec-ct-2021",
            "task": "patient_to_trial",
            "profile": args.profile,
            "source_sha256": source["sha256"],
            "pool_count": len(trial_ids),
            "sorted_pool_ids_sha256": ordered_trial_ids_sha256(trial_ids),
            "output": str(args.output.resolve()),
        }
    )
    return 0


def _forward_options(
    args: argparse.Namespace,
    benchmark,
    *,
    task_input_id: str,
) -> dict[str, object]:
    extension = extension_for(args.system)
    if extension is not None:
        return extension.run_options(args, benchmark, task_input_id)
    raise ValueError("fusion and staged options are supplied as direction-specific artifacts")


def _revalidate_release_producer(
    args: argparse.Namespace,
    *,
    system_id: str,
    release_identity: Mapping[str, str] | None,
    dependency_environment: Mapping[str, JsonValue] | None,
    git_state: tuple[str, bool],
) -> tuple[str, bool]:
    if release_identity is None:
        return _git_state()
    current_identity = _verified_release_identity(args)
    current_environment = _verified_dependency_environment(args, system_id=system_id)
    current_git_state = _git_state()
    if (
        current_identity != release_identity
        or current_environment != dependency_environment
        or current_git_state != git_state
    ):
        raise ValueError("release producer identity changed during benchmark execution")
    return current_git_state


def _rrf_component_producer_identity(
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


def _store_forward(
    prepared: PreparedBenchmark,
    args: argparse.Namespace,
    *,
    track: str,
    profile_id: str,
    clinical_as_of: datetime,
    protocol_approval_id: str | None = None,
) -> Path:
    producer_release_identity = _verified_release_identity(args)
    if (
        profile_id.endswith("-judgment-union") or profile_id == EXTERNAL_FIDELITY_TREC_2021_PROFILE
    ) and producer_release_identity is None:
        raise ValueError(
            "compute-bounded execution requires --release-manifest, --package-artifact, and "
            "--dependency-lock"
        )
    producer_dependency_environment = (
        _verified_dependency_environment(args, system_id=args.system)
        if producer_release_identity is not None
        else None
    )
    producer_git_state = _git_state()
    capability = require_supported_combination(
        track=track,
        task="patient_to_trial",
        profile=profile_id,
        system=args.system,
    )
    evidence_scope = cast(str, capability["evidence_scope"])
    source_identity = release_source_identity(
        prepared,
        track=track,
        profile=profile_id,
        evidence_scope=evidence_scope,
    )
    requires_protocol = requires_effectiveness_protocol(profile_id)
    if requires_protocol:
        require_frozen_effectiveness_protocol(
            profile_id,
            protocol_approval_id,
            "effectiveness protocol_approval_id",
            system=args.system,
        )
    elif protocol_approval_id is not None:
        raise ValueError(
            "protocol approval identities apply only to judgment-union or confirmation profiles"
        )
    benchmark = resolve_benchmark_profile(
        prepared,
        profile_id=profile_id,
        expected_pool_count=getattr(args, "pool_count", None),
        expected_pool_ids_sha256=getattr(args, "pool_ids_sha256", None),
        expected_pool_receipt_id=getattr(args, "pool_receipt_id", None),
        supplied_pool_ids=_read_pool_ids(getattr(args, "pool_file", None)),
    )
    evaluation_package = benchmark.evaluation_package
    run_id = args.run_id or _run_id(args.system)
    request = SystemRunRequest.for_benchmark(
        benchmark,
        run_id=run_id,
        top_k=args.top_k,
        metric_cutoff=args.metric_cutoff,
        options={},
        clinical_as_of=clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
        identity_version="2.0",
    )
    configuration: dict[str, JsonValue]
    stage_rankings: tuple[StageRanking, ...] = ()
    extension = extension_for(args.system)
    if extension is not None and extension.run_from_components is not None:
        result = extension.run_from_components(
            ComponentRun(
                args=args,
                benchmark=benchmark,
                request=request,
                profile_id=profile_id,
                run_id=run_id,
                producer_release_identity=producer_release_identity,
                producer_dependency_environment=producer_dependency_environment,
            )
        )
        candidates = tuple(result.candidates)
        configuration = dict(result.configuration)
        runtime_seconds = result.runtime_seconds
        primary_ranking = result.primary_ranking
        pipeline_depth = result.pipeline_depth
        stage_rankings = result.stage_rankings
    else:
        request = replace(
            request,
            options=_forward_options(
                args,
                benchmark,
                task_input_id=request.snapshot.task_input_id,
            ),
        )
        result = run_release_reference_system(args.system, request)
        candidates = tuple(result.candidates)
        configuration = dict(result.configuration)
        runtime_seconds = result.runtime_seconds
        primary_ranking = result.primary_ranking
        pipeline_depth = result.pipeline_depth
        stage_rankings = result.stage_rankings
    system_input = configuration.get("system_input")
    if not isinstance(system_input, Mapping) or not isinstance(
        system_input.get("system_input_id"), str
    ):
        raise ValueError("reference System omitted its exact System Input identity")
    system_input_id = str(system_input["system_input_id"])
    commit, dirty = _revalidate_release_producer(
        args,
        system_id=args.system,
        release_identity=producer_release_identity,
        dependency_environment=producer_dependency_environment,
        git_state=producer_git_state,
    )
    configuration.update(
        {
            "release_support": {
                "release_version": RELEASE_VERSION,
                "track": track,
                "task": "patient_to_trial",
                "profile": profile_id,
                "system": args.system,
                "protocol_approval_id": protocol_approval_id,
            },
            "release_source": source_identity,
            "execution_provenance": {
                "git_commit": commit,
                "working_tree_dirty": dirty,
                "environment": {"python": platform.python_version(), "taim": __version__},
                "determinism": "reference-system-defined",
            },
        }
    )
    if producer_release_identity is not None:
        configuration["producer_release_identity"] = cast(JsonValue, producer_release_identity)
        configuration["producer_dependency_environment"] = cast(
            JsonValue,
            producer_dependency_environment,
        )
    output = Path(args.output_dir) / run_id
    stored = write_public_patient_to_trial_run(
        output,
        manifest=RunManifest(
            run_id=run_id,
            system_id=args.system,
            created_at=datetime.now(UTC),
            git_commit=commit,
            configuration=configuration,
            benchmark_lineage=track,
            snapshot_contract_version=request.snapshot.contract_version,
            prepared_snapshot_id=benchmark.prepared_snapshot_id,
            task_input_id=request.snapshot.task_input_id,
            system_input_id=system_input_id,
            evaluation_package_id=evaluation_package.evaluation_package_id,
            benchmark_profile=benchmark.manifest_configuration(),
            primary_ranking=primary_ranking,
            pipeline_depth=pipeline_depth,
            budget_k=args.top_k,
            clinical_as_of=clinical_as_of,
            patient_evidence_profile=request.snapshot.patient_evidence_profile.to_dict(),
            query_patient_versions=tuple(
                (item.patient_id, item.patient_version_id)
                for item in request.snapshot.patient_versions
            ),
            trial_corpus_versions=tuple(
                (item.trial_id, item.trial_version_id) for item in request.snapshot.trial_versions
            ),
            runtime_seconds=runtime_seconds,
            stage_rankings=stage_rankings,
        ),
        candidates=candidates,
        evaluation_package=evaluation_package,
    )
    _print(
        {
            "release": RELEASE_VERSION,
            "track": track,
            "task": "patient_to_trial",
            "profile": profile_id,
            "system": args.system,
            "prepared_snapshot_id": stored.manifest.prepared_snapshot_id,
            "task_input_id": stored.manifest.task_input_id,
            "system_input_id": stored.manifest.system_input_id,
            "evaluation_package_id": stored.manifest.evaluation_package_id,
            "run_directory": str(stored.directory),
        }
    )
    return output


def _forward_fixture_run(args: argparse.Namespace) -> int:
    prepared = load_release_fixture(args.track)
    _store_forward(
        prepared,
        args,
        track=args.track,
        profile_id="synthetic-patient-to-trial",
        clinical_as_of=FIXTURE_CLINICAL_AS_OF,
    )
    return 0


def _forward_benchmark_run(args: argparse.Namespace) -> int:
    profile = args.profile or (
        "official-full"
        if args.track == "trec-ct-2021"
        else "confirmation-full"
        if args.track.startswith("trec-ct-")
        else None
    )
    if profile is None:
        raise ValueError("SIGIR 2016 requires --profile description or summary")
    capability = require_supported_combination(
        track=args.track,
        task="patient_to_trial",
        profile=profile,
        system=args.system,
    )
    if capability["evidence_scope"] not in {
        "real_effectiveness",
        "real_effectiveness_requires_protocol",
        "within_pool_effectiveness",
        "within_pool_effectiveness_requires_protocol",
        "external_fidelity_effectiveness",
    }:
        raise ValueError(
            "benchmark run requires a real effectiveness Profile; use fixture run for "
            "synthetic conformance"
        )
    prepared = load_prepared_benchmark(args.data_dir)
    if prepared.snapshot.benchmark_lineage != args.track:
        raise ValueError("prepared Track does not match --track")
    source_profile = source_profile_id(profile)
    if args.track == "sigir-ct-2016" and prepared.dataset_id != f"sigir-ct-2016-{source_profile}":
        raise ValueError("SIGIR prepared query profile does not match --profile")
    _store_forward(
        prepared,
        args,
        track=args.track,
        profile_id=profile,
        clinical_as_of=args.clinical_as_of,
        protocol_approval_id=args.protocol_approval_id,
    )
    return 0


def _forward_validate(args: argparse.Namespace) -> int:
    stored = load_public_patient_to_trial_run(args.run_dir)
    profile = load_profile(cast(str, stored.manifest.benchmark_profile["profile_id"]))
    evaluate_public_patient_to_trial_run(
        stored,
        load_local_evaluation_package(args.run_dir),
        profile,
    )
    support = cast(Mapping[str, object], stored.manifest.configuration["release_support"])
    _print(
        {
            "release_version": support["release_version"],
            "track": support["track"],
            "task": stored.manifest.task,
            "profile": support["profile"],
            "profile_definition_sha256": stored.manifest.benchmark_profile["definition_sha256"],
            "system": stored.manifest.system_id,
            "run_id": stored.manifest.run_id,
            "prepared_snapshot_id": stored.manifest.prepared_snapshot_id,
            "task_input_id": stored.manifest.task_input_id,
            "system_input_id": stored.manifest.system_input_id,
            "evaluation_package_id": stored.manifest.evaluation_package_id,
        }
    )
    return 0


def _forward_evaluate(args: argparse.Namespace) -> int:
    stored = load_public_patient_to_trial_run(args.run_dir)
    package = load_local_evaluation_package(args.run_dir)
    profile = load_profile(cast(str, stored.manifest.benchmark_profile["profile_id"]))
    if args.stage is None:
        _print(evaluate_public_patient_to_trial_run(stored, package, profile))
        return 0
    stages = {stage.name: stage for stage in stored.stage_rankings}
    try:
        stage = stages[args.stage]
    except KeyError as exc:
        available = ", ".join(sorted(stages)) or "none"
        raise ValueError(f"unknown stage {args.stage!r}; available stages: {available}") from exc
    scorecard = evaluate_public_patient_to_trial_run(
        replace(stored, candidates=stage.candidates),
        package,
        profile,
    )
    _print(
        {
            "evaluation_scope": "stage_diagnostic",
            "stage": stage.name,
            "pipeline_depth": stage.pipeline_depth,
            "artifact_hash": stage.artifact_hash,
            "scorecard": scorecard,
        }
    )
    return 0


def _reverse_provenance(
    result,
    *,
    git_state: tuple[str, bool],
) -> TrialToPatientExecutionProvenance:
    commit, dirty = git_state
    model_identity = result.configuration.get("model_identity")
    index_identity = result.configuration.get("index_identity")
    if not isinstance(model_identity, Mapping) or not isinstance(index_identity, Mapping):
        raise ValueError("reverse System omitted model or index identity")
    return TrialToPatientExecutionProvenance(
        git_commit=commit,
        working_tree_dirty=dirty,
        environment={"python": platform.python_version(), "taim": __version__},
        seeds={"deterministic_bm25": 0},
        model_identity=model_identity,
        index_identity=index_identity,
    )


def _store_reverse(resolved, args: argparse.Namespace, *, track: str, profile: str) -> Path:
    system_id = SUBCOMMAND_SYSTEMS[("trial-to-patient", "fixture")]
    producer_release_identity = _verified_release_identity(args)
    producer_dependency_environment = (
        _verified_dependency_environment(args, system_id=system_id)
        if producer_release_identity is not None
        else None
    )
    producer_git_state = _git_state()
    capability = require_supported_combination(
        track=track,
        task="trial_to_patient",
        profile=profile,
        system=system_id,
    )
    source_identity = release_source_identity(
        resolved.prepared,
        track=track,
        profile=profile,
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    run_id = args.run_id or _run_id(system_id)
    request = TrialToPatientRunRequest.for_benchmark(
        resolved,
        run_id=run_id,
        top_k=args.top_k,
        metric_cutoff=args.metric_cutoff,
        options={"b": args.b, "k1": args.k1},
        identity_version="2.0",
    )
    result = run_trial_to_patient_system(system_id, request)
    final_git_state = _revalidate_release_producer(
        args,
        system_id=system_id,
        release_identity=producer_release_identity,
        dependency_environment=producer_dependency_environment,
        git_state=producer_git_state,
    )
    configuration = {
        **result.configuration,
        "release_support": {
            "release_version": RELEASE_VERSION,
            "track": track,
            "task": "trial_to_patient",
            "profile": profile,
            "system": system_id,
        },
        "release_source": source_identity,
    }
    if producer_release_identity is not None:
        configuration["producer_release_identity"] = producer_release_identity
        configuration["producer_dependency_environment"] = producer_dependency_environment
    result = replace(result, configuration=configuration)
    output = Path(args.output_dir) / run_id
    stored = write_trial_to_patient_run(
        output,
        resolved=resolved,
        request=request,
        result=result,
        execution_provenance=_reverse_provenance(result, git_state=final_git_state),
        created_at=datetime.now(UTC),
    )
    (output / "evaluation-package.json").write_text(
        json.dumps(
            resolved.evaluation_package.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _print(
        {
            "release": RELEASE_VERSION,
            "track": track,
            "task": "trial_to_patient",
            "profile": profile,
            "system": stored.manifest.system_id,
            "prepared_snapshot_id": stored.manifest.prepared_snapshot_id,
            "task_input_id": stored.manifest.task_input_id,
            "system_input_id": stored.manifest.system_input_id,
            "evaluation_package_id": stored.manifest.evaluation_package_id,
            "run_directory": str(stored.directory),
        }
    )
    return output


def _reverse_fixture_run(args: argparse.Namespace) -> int:
    _store_reverse(
        load_reverse_fixture(args.track),
        args,
        track=args.track,
        profile="synthetic-trial-to-patient",
    )
    return 0


def _reverse_benchmark_run(args: argparse.Namespace) -> int:
    resolved = qualify_reverse_viability_profile(
        load_prepared_benchmark(args.data_dir),
        clinical_as_of=args.clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )
    _store_reverse(
        resolved,
        args,
        track="trec-ct-2021",
        profile="trec-ct-2021-reverse-complete10",
    )
    return 0


def _validated_reverse_run(
    run_dir: Path,
) -> tuple[
    StoredTrialToPatientRun,
    TrialToPatientEvaluationPackage,
    Mapping[str, object],
    dict[str, object],
]:
    """Load, release-validate, and evaluate one reverse Local Run."""

    stored = load_trial_to_patient_run(run_dir)
    validate_public_reverse_run_tree(run_dir, stored.manifest)
    support = validate_reverse_release_manifest(stored.manifest)
    try:
        payload = json.loads((run_dir / "evaluation-package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read reverse Evaluation Package: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("reverse Evaluation Package must be a JSON object")
    package = TrialToPatientEvaluationPackage.from_dict(payload)
    metrics = evaluate_trial_to_patient_run(stored, package)
    return stored, package, support, metrics


def _reverse_identity_payload(
    stored: StoredTrialToPatientRun,
    support: Mapping[str, object],
) -> dict[str, object]:
    manifest = stored.manifest
    return {
        "release_version": RELEASE_VERSION,
        "track": support["track"],
        "task": manifest.task,
        "profile": manifest.benchmark_profile["profile_id"],
        "system": manifest.system_id,
        "run_id": manifest.run_id,
        "prepared_snapshot_id": manifest.prepared_snapshot_id,
        "task_input_id": manifest.task_input_id,
        "system_input_id": manifest.system_input_id,
        "evaluation_package_id": manifest.evaluation_package_id,
    }


def _reverse_validate(args: argparse.Namespace) -> int:
    stored, _package, support, _metrics = _validated_reverse_run(args.run_dir)
    _print(_reverse_identity_payload(stored, support))
    return 0


def _reverse_evaluate(args: argparse.Namespace) -> int:
    stored, _package, support, metrics = _validated_reverse_run(args.run_dir)
    _print({**metrics, **_reverse_identity_payload(stored, support)})
    return 0


def _prepare(args: argparse.Namespace) -> int:
    confirmation = args.track in {"trec-ct-2022", "trec-ct-2023"}
    if confirmation:
        require_frozen_paper_protocol(
            args.protocol_approval_id, "confirmation protocol_approval_id"
        )
    elif args.protocol_approval_id is not None:
        raise ValueError("protocol approval identities apply only to confirmation Tracks")
    if args.track == "trec-ct-2021":
        from taim.data.trec_ct_2021 import prepare_trec_ct_2021

        result = prepare_trec_ct_2021(args.source, args.output_dir)
    elif args.track in {"trec-ct-2022", "trec-ct-2023"}:
        from taim.data.trec_ct import prepare_trec_ct

        result = prepare_trec_ct(
            args.track,
            args.source,
            args.output_dir,
        )
    else:
        from taim.data.sigir_ct_2016 import prepare_sigir_ct_2016

        if args.query_profile not in {"description", "summary"}:
            raise ValueError("SIGIR 2016 requires --query-profile description or summary")
        result = prepare_sigir_ct_2016(
            args.source,
            args.output_dir,
            query_variant=args.query_profile,
        )
    _print(
        {
            "release": RELEASE_VERSION,
            "track": args.track,
            "query_profile": args.query_profile,
            "protocol_approval_id": args.protocol_approval_id,
            "output_directory": str(result.output_directory.resolve()),
        }
    )
    return 0


def _inspect_pool(args: argparse.Namespace) -> int:
    is_external = args.profile == EXTERNAL_FIDELITY_TREC_2021_PROFILE
    if not args.profile.endswith("-judgment-union") and not is_external:
        raise ValueError("data pool inspect requires a compute-bounded Profile")
    if args.protocol_approval_id is None:
        raise ValueError("data pool inspect requires --protocol-approval-id")
    require_frozen_effectiveness_protocol(
        args.profile,
        args.protocol_approval_id,
        "pool inspection protocol_approval_id",
    )
    # The pool belongs to the Profile, not to a System: any published System declared for this
    # Track and Profile carries its evidence scope. The first System in the table need not be one.
    declared = [
        row
        for row in cast(Sequence[Mapping[str, JsonValue]], load_release_support()["support_matrix"])
        if (row["track"], row["task"], row["profile"])
        == (args.track, "patient_to_trial", args.profile)
    ]
    if not declared:
        raise ValueError(
            "unsupported release combination: "
            f"Track={args.track}, Task=patient_to_trial, Profile={args.profile}"
        )
    capability = declared[0]
    prepared = load_prepared_benchmark(args.data_dir)
    if prepared.snapshot.benchmark_lineage != args.track:
        raise ValueError("prepared Track does not match --track")
    source_profile = source_profile_id(args.profile)
    if args.track == "sigir-ct-2016" and prepared.dataset_id != f"sigir-ct-2016-{source_profile}":
        raise ValueError("SIGIR prepared query profile does not match --profile")
    release_source_identity(
        prepared,
        track=args.track,
        profile=args.profile,
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    inspection = (
        inspect_external_fidelity_profile(
            prepared,
            profile_id=args.profile,
            supplied_pool_ids=_read_pool_ids(args.pool_file) or (),
            expected_pool_count=load_profile(args.profile).expected_pool_count or 0,
        )
        if is_external
        else inspect_judgment_union_profile(prepared, profile_id=args.profile)
    )
    _print(
        {
            "release": RELEASE_VERSION,
            "track": args.track,
            "task": "patient_to_trial",
            "protocol_approval_id": args.protocol_approval_id,
            **inspection,
        }
    )
    return 0


def _release_validate(args: argparse.Namespace) -> int:
    from taim.benchmark_release import validate_benchmark_release

    report = validate_benchmark_release(
        args.candidate,
        args.manifest,
        execute_quick_start=args.runtime,
        quick_start_executable=sys.argv[0] if args.runtime else None,
    )
    _print(report)
    return 0


def _bundle_validate(args: argparse.Namespace) -> int:
    from taim.release_result_bundle import validate_release_result_bundle

    _print(validate_release_result_bundle(args.bundle_dir))
    return 0


def _add_forward_options(parser: argparse.ArgumentParser, *, real: bool) -> None:
    parser.add_argument(
        "--system",
        choices=tuple(
            system for system, tasks in SYSTEM_TASKS.items() if "patient_to_trial" in tasks
        ),
        default=next(
            (
                extension.default_system
                for extension in pipeline_extensions()
                if extension.default_system is not None
                and extension.system_tasks[extension.default_system] == "patient_to_trial"
            ),
            next(
                (system for system, tasks in SYSTEM_TASKS.items() if "patient_to_trial" in tasks),
                None,
            ),
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--top-k", type=_positive_integer, default=DEFAULT_RETRIEVAL_DEPTH)
    parser.add_argument("--metric-cutoff", type=_positive_integer, default=DEFAULT_METRIC_CUTOFF)
    option_adders = dict.fromkeys(extension.add_run_options for extension in pipeline_extensions())
    for add_options in option_adders:
        add_options(parser)
    _add_release_identity_options(parser)
    if real:
        parser.add_argument("--data-dir", type=Path, required=True)
        parser.add_argument("--clinical-as-of", type=parse_clinical_as_of, required=True)
        parser.add_argument(
            "--pool-file",
            type=Path,
            help="caller-supplied trial-ID membership for an external-fidelity Profile",
        )
        parser.add_argument(
            "--pool-count",
            type=_positive_integer,
            help="frozen count emitted by data pool inspect for a judgment-union Profile",
        )
        parser.add_argument(
            "--pool-ids-sha256",
            type=_sha256_identity,
            help="frozen ID-list digest emitted by data pool inspect",
        )
        parser.add_argument(
            "--pool-receipt-id",
            type=_sha256_identity,
            help="content identity emitted by data pool inspect",
        )
        parser.add_argument(
            "--protocol-approval-id",
            type=_sha256_identity,
            help="required for judgment-union and TREC 2022/2023 confirmation runs",
        )


def _add_reverse_options(parser: argparse.ArgumentParser, *, real: bool) -> None:
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--top-k", type=_positive_integer, default=20)
    parser.add_argument("--metric-cutoff", type=_positive_integer, default=5)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    _add_release_identity_options(parser)
    if real:
        parser.add_argument("--data-dir", type=Path, required=True)
        parser.add_argument("--clinical-as-of", type=parse_clinical_as_of, required=True)


def _add_release_identity_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--package-artifact", type=Path)
    parser.add_argument(
        "--dependency-lock",
        type=Path,
        help="public uv.lock whose bytes are bound by the release manifest",
    )


def _add_direction_commands(parent: argparse.ArgumentParser, *, reverse: bool) -> None:
    commands = parent.add_subparsers(dest="operation", required=True)
    fixture = commands.add_parser("fixture", help="run deterministic conformance data")
    fixture_commands = fixture.add_subparsers(dest="fixture_operation", required=True)
    fixture_run = fixture_commands.add_parser("run")
    fixture_run.add_argument("--track", choices=SUPPORTED_TRACKS, default="trec-ct-2021")
    if reverse:
        _add_reverse_options(fixture_run, real=False)
        fixture_run.set_defaults(handler=_reverse_fixture_run)
    else:
        _add_forward_options(fixture_run, real=False)
        fixture_run.set_defaults(handler=_forward_fixture_run)
    benchmark = commands.add_parser("benchmark", help="run a qualified real-data profile")
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_operation", required=True)
    benchmark_run = benchmark_commands.add_parser("run")
    if reverse:
        _add_reverse_options(benchmark_run, real=True)
        benchmark_run.set_defaults(handler=_reverse_benchmark_run)
    else:
        benchmark_run.add_argument("--track", choices=SUPPORTED_TRACKS, required=True)
        benchmark_run.add_argument("--profile")
        _add_forward_options(benchmark_run, real=True)
        benchmark_run.set_defaults(handler=_forward_benchmark_run)
    run = commands.add_parser("run", help="validate or evaluate a closed run")
    run_commands = run.add_subparsers(dest="run_operation", required=True)
    validate = run_commands.add_parser("validate")
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.set_defaults(handler=_reverse_validate if reverse else _forward_validate)
    evaluate = run_commands.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    if not reverse:
        evaluate.add_argument(
            "--stage",
            help="evaluate one named stage as a diagnostic, leaving the Primary Ranking unchanged",
        )
    evaluate.set_defaults(handler=_reverse_evaluate if reverse else _forward_evaluate)
    add_submission_commands(commands, reverse=reverse)
    if not reverse:
        command_adders = dict.fromkeys(
            extension.add_patient_to_trial_commands
            for extension in pipeline_extensions()
            if extension.add_patient_to_trial_commands is not None
        )
        for add_commands in command_adders:
            add_commands(commands)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trial-benchmark",
        description="TAIM Benchmark Release 0.1",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    support = commands.add_parser("support", help="inspect executable release capabilities")
    support_commands = support.add_subparsers(dest="support_operation", required=True)
    support_list = support_commands.add_parser("list")
    support_list.set_defaults(handler=_support_list)
    if any("patient_to_trial" in tasks for tasks in SYSTEM_TASKS.values()):
        forward = commands.add_parser("patient-to-trial", help="patient query against trial corpus")
        _add_direction_commands(forward, reverse=False)
    if any("trial_to_patient" in tasks for tasks in SYSTEM_TASKS.values()):
        reverse = commands.add_parser("trial-to-patient", help="trial query against patient corpus")
        _add_direction_commands(reverse, reverse=True)
    data = commands.add_parser("data", help="prepare checksum-locked local source files")
    data_commands = data.add_subparsers(dest="data_operation", required=True)
    prepare = data_commands.add_parser("prepare")
    prepare.add_argument("--track", choices=SUPPORTED_TRACKS, required=True)
    prepare.add_argument("--query-profile")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--protocol-approval-id",
        type=_sha256_identity,
        help="required before a TREC 2022/2023 source file is opened",
    )
    prepare.set_defaults(handler=_prepare)
    pool = data_commands.add_parser(
        "pool", help="inspect a compute-bounded corpus before effectiveness execution"
    )
    pool_commands = pool.add_subparsers(dest="pool_operation", required=True)
    pool_inspect = pool_commands.add_parser("inspect")
    pool_inspect.add_argument("--track", choices=SUPPORTED_TRACKS, required=True)
    pool_inspect.add_argument("--profile", required=True)
    pool_inspect.add_argument("--data-dir", type=Path, required=True)
    pool_inspect.add_argument("--pool-file", type=Path)
    pool_inspect.add_argument("--protocol-approval-id", type=_sha256_identity)
    pool_inspect.set_defaults(handler=_inspect_pool)
    external_pool = data_commands.add_parser(
        "external-pool",
        help="verify an external source lock and extract ID-only pool membership",
    )
    external_pool_commands = external_pool.add_subparsers(
        dest="external_pool_operation", required=True
    )
    external_pool_prepare = external_pool_commands.add_parser("prepare")
    external_pool_prepare.add_argument(
        "--profile",
        choices=(EXTERNAL_FIDELITY_TREC_2021_PROFILE,),
        required=True,
    )
    external_pool_prepare.add_argument("--source", type=Path, required=True)
    external_pool_prepare.add_argument("--output", type=Path, required=True)
    external_pool_prepare.set_defaults(handler=_prepare_external_pool)
    release = commands.add_parser("release", help="validate a generated public candidate")
    release_commands = release.add_subparsers(dest="release_operation", required=True)
    release_validate = release_commands.add_parser("validate")
    release_validate.add_argument("--candidate", type=Path, required=True)
    release_validate.add_argument("--manifest", type=Path, required=True)
    release_validate.add_argument("--runtime", action="store_true")
    release_validate.set_defaults(handler=_release_validate)
    bundle = commands.add_parser("bundle", help="validate a Published Result Bundle")
    bundle_commands = bundle.add_subparsers(dest="bundle_operation", required=True)
    bundle_validate = bundle_commands.add_parser("validate")
    bundle_validate.add_argument("--bundle-dir", type=Path, required=True)
    bundle_validate.set_defaults(handler=_bundle_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
