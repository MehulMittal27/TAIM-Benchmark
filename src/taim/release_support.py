"""Canonical capabilities advertised by Benchmark Release 0.1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, cast

from taim.contracts import content_sha256, require_exact_keys, require_non_empty, require_sha256
from taim.pipeline_extensions import pipeline_extensions
from taim.schemas import JsonValue, SchemaValidationError, json_value_to_builtins
from taim.source import SourceArtifact, SourceBundle

if TYPE_CHECKING:
    from taim.data import PreparedBenchmark
    from taim.trial_to_patient import TrialToPatientRunManifest

RELEASE_SUPPORT_SCHEMA_VERSION = "1.0"
RELEASE_VERSION = "0.1.0"
RELEASE_SUPPORT_RESOURCE = "release-support-v0.1.json"
FROZEN_PAPER_PROTOCOL_SHA256 = (
    "sha256:18a9c9bd3c76455038d01458a5cb2d933059d1e72924285e26500a8aaab1baf6"
)
EXTERNAL_FIDELITY_PROFILE = "trec-ct-2021-external-fidelity-26149"
# Exact repository-owner-approved v4 protocol bytes.
FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256: str | None = (
    "sha256:55bb91f69d6494d58eff62b96f65f7b59759de45a366dc551e4ad5586fdfc896"
)

_TRACK_LOCKS = {
    "sigir-ct-2016": "sigir-ct-2016.json",
    "trec-ct-2021": "trec-ct-2021.json",
    "trec-ct-2022": "trec-ct-2022.json",
    "trec-ct-2023": "trec-ct-2023.json",
}
_RELEASE_SOURCE_FIELDS = {
    "scope",
    "prepared_dataset_id",
    "source_recipe_id",
    "source_recipe_sha256",
    "source_bundle_id",
    "lock_filename",
    "source_lock_sha256",
    "record_counts",
    "artifacts",
}
_RELEASE_SOURCE_ARTIFACT_FIELDS = {"role", "filename", "byte_size", "sha256"}
_RECORD_COUNT_FIELDS = {"topics", "trials"}

_REAL_PREPARATION_RECORD_COUNTS: Mapping[tuple[str, str], Mapping[str, int]] = {
    ("sigir-ct-2016", "description"): {"topics": 60, "trials": 204_855},
    ("sigir-ct-2016", "summary"): {"topics": 60, "trials": 204_855},
    ("trec-ct-2021", "official-full"): {"topics": 75, "trials": 375_580},
    ("trec-ct-2021", "trec-ct-2021-reverse-complete10"): {
        "topics": 75,
        "trials": 375_580,
    },
    ("trec-ct-2022", "confirmation-full"): {"topics": 50, "trials": 375_580},
    ("trec-ct-2023", "confirmation-full"): {"topics": 40, "trials": 451_538},
}

SUPPORTED_TRACKS = (
    "sigir-ct-2016",
    "trec-ct-2021",
    "trec-ct-2022",
    "trec-ct-2023",
)
SUPPORTED_TASKS = ("patient_to_trial", "trial_to_patient")
SUPPORTED_PROFILES = (
    "confirmation-full",
    "description",
    "official-full",
    "sigir-ct-2016-description-judgment-union",
    "sigir-ct-2016-summary-judgment-union",
    "summary",
    "synthetic-patient-to-trial",
    "synthetic-trial-to-patient",
    EXTERNAL_FIDELITY_PROFILE,
    "trec-ct-2021-judgment-union",
    "trec-ct-2021-reverse-complete10",
    "trec-ct-2022-judgment-union",
    "trec-ct-2023-judgment-union",
)


def _packaged_system_tasks() -> dict[str, frozenset[str]]:
    """Each System the packaged support catalogue declares, with the Tasks it runs there."""

    resource = files("taim").joinpath("data", RELEASE_SUPPORT_RESOURCE)
    try:
        rows = json.loads(resource.read_text(encoding="utf-8"))["support_matrix"]
        pairs = [(row["system"], row["task"]) for row in rows]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(
            f"packaged release support catalog is unreadable: {exc}"
        ) from exc
    tasks: dict[str, set[str]] = {}
    for system, task in pairs:
        if not isinstance(system, str) or not isinstance(task, str):
            raise SchemaValidationError(
                "packaged release support catalog names a non-string System"
            )
        tasks.setdefault(system, set()).add(task)
    return {system: frozenset(tasks[system]) for system in sorted(tasks)}


def _subcommand_systems() -> dict[tuple[str, str], str]:
    bindings = dict(_CORE_SUBCOMMAND_SYSTEMS)
    for extension in pipeline_extensions():
        for command, system in extension.subcommands.items():
            if command in bindings:
                raise ValueError(f"two declarations bind the subcommand {' '.join(command)}")
            bindings[command] = system
    return dict(sorted(bindings.items()))


# The packaged catalogue is the declaration of which Systems this release runs: a release that holds
# a System back ships a catalogue without it, and these sets follow.
SYSTEM_TASKS: Mapping[str, frozenset[str]] = _packaged_system_tasks()
SUPPORTED_SYSTEMS = tuple(SYSTEM_TASKS)
# The Systems whose own pipeline extension marks them as external baselines.
EXTERNAL_BASELINE_SYSTEMS = frozenset(
    system
    for extension in pipeline_extensions()
    if extension.external_baseline
    for system in extension.system_tasks
)
# The System a CLI subcommand runs when it takes no --system option. The release validator's
# catalogue-CLI-dispatch check and the documented-command checker both read this declaration. A
# pipeline extension declares its own subcommands, which are added here.
_CORE_SUBCOMMAND_SYSTEMS: Mapping[tuple[str, str], str] = {
    ("patient-to-trial", "queries"): "bm25-folded",
    ("trial-to-patient", "benchmark"): "bm25-trial-to-patient",
    ("trial-to-patient", "fixture"): "bm25-trial-to-patient",
}
SUBCOMMAND_SYSTEMS: Mapping[tuple[str, str], str] = _subcommand_systems()
PROFILE_TASKS: Mapping[str, str] = {
    "confirmation-full": "patient_to_trial",
    "description": "patient_to_trial",
    "official-full": "patient_to_trial",
    "sigir-ct-2016-description-judgment-union": "patient_to_trial",
    "sigir-ct-2016-summary-judgment-union": "patient_to_trial",
    "summary": "patient_to_trial",
    "synthetic-patient-to-trial": "patient_to_trial",
    "synthetic-trial-to-patient": "trial_to_patient",
    EXTERNAL_FIDELITY_PROFILE: "patient_to_trial",
    "trec-ct-2021-judgment-union": "patient_to_trial",
    "trec-ct-2021-reverse-complete10": "trial_to_patient",
    "trec-ct-2022-judgment-union": "patient_to_trial",
    "trec-ct-2023-judgment-union": "patient_to_trial",
}
PROFILE_TRACKS: Mapping[str, frozenset[str]] = {
    "confirmation-full": frozenset({"trec-ct-2022", "trec-ct-2023"}),
    "description": frozenset({"sigir-ct-2016"}),
    "official-full": frozenset({"trec-ct-2021"}),
    "sigir-ct-2016-description-judgment-union": frozenset({"sigir-ct-2016"}),
    "sigir-ct-2016-summary-judgment-union": frozenset({"sigir-ct-2016"}),
    "summary": frozenset({"sigir-ct-2016"}),
    "synthetic-patient-to-trial": frozenset(SUPPORTED_TRACKS),
    "synthetic-trial-to-patient": frozenset(SUPPORTED_TRACKS),
    EXTERNAL_FIDELITY_PROFILE: frozenset({"trec-ct-2021"}),
    "trec-ct-2021-judgment-union": frozenset({"trec-ct-2021"}),
    "trec-ct-2021-reverse-complete10": frozenset({"trec-ct-2021"}),
    "trec-ct-2022-judgment-union": frozenset({"trec-ct-2022"}),
    "trec-ct-2023-judgment-union": frozenset({"trec-ct-2023"}),
}
EVIDENCE_SCOPES = frozenset(
    {
        "real_effectiveness",
        "external_fidelity_effectiveness",
        "real_effectiveness_requires_protocol",
        "reverse_viability_effectiveness",
        "synthetic_conformance",
        "within_pool_effectiveness",
        "within_pool_effectiveness_requires_protocol",
    }
)
PROFILE_EVIDENCE_SCOPE: Mapping[str, str] = {
    "confirmation-full": "real_effectiveness_requires_protocol",
    "description": "real_effectiveness",
    "official-full": "real_effectiveness",
    "sigir-ct-2016-description-judgment-union": "within_pool_effectiveness",
    "sigir-ct-2016-summary-judgment-union": "within_pool_effectiveness",
    "summary": "real_effectiveness",
    "synthetic-patient-to-trial": "synthetic_conformance",
    "synthetic-trial-to-patient": "synthetic_conformance",
    EXTERNAL_FIDELITY_PROFILE: "external_fidelity_effectiveness",
    "trec-ct-2021-judgment-union": "within_pool_effectiveness",
    "trec-ct-2021-reverse-complete10": "reverse_viability_effectiveness",
    "trec-ct-2022-judgment-union": "within_pool_effectiveness_requires_protocol",
    "trec-ct-2023-judgment-union": "within_pool_effectiveness_requires_protocol",
}

_SOURCE_PROFILE_IDS = {
    "sigir-ct-2016-description-judgment-union": "description",
    "sigir-ct-2016-summary-judgment-union": "summary",
    "trec-ct-2021-judgment-union": "official-full",
    EXTERNAL_FIDELITY_PROFILE: "official-full",
    "trec-ct-2022-judgment-union": "confirmation-full",
    "trec-ct-2023-judgment-union": "confirmation-full",
}

_CAPABILITY_FIELDS = {
    "track",
    "task",
    "profile",
    "system",
    "evidence_scope",
    "required_files",
}


def _require_string(value: object, *, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(f"{role} must be a non-empty string")
    return value


def capability_key(capability: Mapping[str, object]) -> tuple[str, str, str, str]:
    """Return the canonical Track x Task x Profile x System key."""

    return cast(
        tuple[str, str, str, str],
        tuple(
            _require_string(capability[field], role=f"support capability {field}")
            for field in ("track", "task", "profile", "system")
        ),
    )


def source_profile_id(profile: str) -> str:
    """Return the complete-source Profile used to prepare one effective corpus Profile."""

    return _SOURCE_PROFILE_IDS.get(profile, profile)


def requires_effectiveness_protocol(profile: str) -> bool:
    """Return whether a real effectiveness Profile requires a frozen protocol identity."""

    try:
        evidence_scope = PROFILE_EVIDENCE_SCOPE[profile]
    except KeyError as exc:
        raise ValueError(f"unknown release Profile {profile!r}") from exc
    return (
        profile.endswith("-judgment-union")
        or profile == EXTERNAL_FIDELITY_PROFILE
        or evidence_scope.endswith("requires_protocol")
    )


def require_frozen_paper_protocol(value: object, role: str) -> str:
    """Require the exact protocol projected into the public Release 0.1 candidate."""

    protocol_id = require_sha256(value, role)
    if protocol_id != FROZEN_PAPER_PROTOCOL_SHA256:
        raise SchemaValidationError(f"{role} does not match the frozen public paper protocol")
    return protocol_id


def require_frozen_effectiveness_protocol(
    profile: str,
    value: object,
    role: str,
    *,
    system: str | None = None,
) -> str:
    """Require the protocol frozen for one effectiveness Profile and System."""

    if profile != EXTERNAL_FIDELITY_PROFILE and system not in EXTERNAL_BASELINE_SYSTEMS:
        return require_frozen_paper_protocol(value, role)
    if FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256 is None:
        raise SchemaValidationError(
            "the external-fidelity protocol is not frozen by explicit human approval"
        )
    protocol_id = require_sha256(value, role)
    if protocol_id != FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256:
        raise SchemaValidationError(f"{role} does not match the frozen external-fidelity protocol")
    return protocol_id


def validate_support_matrix(value: object) -> tuple[dict[str, JsonValue], ...]:
    """Validate, normalize, and canonicalize a release support matrix."""

    if not isinstance(value, list) or not value:
        raise SchemaValidationError("release support_matrix must be a non-empty array")
    normalized: list[dict[str, JsonValue]] = []
    keys: list[tuple[str, str, str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _CAPABILITY_FIELDS:
            raise SchemaValidationError(
                f"release support capability {index} has unexpected or missing fields"
            )
        track, task, profile, system = capability_key(raw)
        if track not in SUPPORTED_TRACKS:
            raise SchemaValidationError(f"unknown release Track {track!r}")
        if task not in SUPPORTED_TASKS:
            raise SchemaValidationError(f"unknown release Task {task!r}")
        if profile not in SUPPORTED_PROFILES:
            raise SchemaValidationError(f"unknown release Profile {profile!r}")
        if system not in SUPPORTED_SYSTEMS:
            raise SchemaValidationError(f"unknown release System {system!r}")
        if task not in SYSTEM_TASKS[system]:
            raise SchemaValidationError(f"release System {system!r} cannot execute Task {task!r}")
        if PROFILE_TASKS[profile] != task or track not in PROFILE_TRACKS[profile]:
            raise SchemaValidationError(
                f"release Profile {profile!r} is direction- or Track-mismatched"
            )
        evidence_scope = _require_string(
            raw["evidence_scope"], role=f"support capability {index} evidence_scope"
        )
        if evidence_scope not in EVIDENCE_SCOPES:
            raise SchemaValidationError(f"support capability {index} has an unknown evidence_scope")
        if evidence_scope != PROFILE_EVIDENCE_SCOPE[profile]:
            raise SchemaValidationError(
                f"support capability {index} evidence_scope does not match its Profile"
            )
        required_files = raw["required_files"]
        if (
            not isinstance(required_files, list)
            or not required_files
            or any(not isinstance(path, str) or not path for path in required_files)
        ):
            raise SchemaValidationError(
                f"support capability {index} required_files must be non-empty paths"
            )
        normalized.append(
            cast(
                dict[str, JsonValue],
                json_value_to_builtins(
                    {
                        "track": track,
                        "task": task,
                        "profile": profile,
                        "system": system,
                        "evidence_scope": evidence_scope,
                        "required_files": sorted(set(required_files)),
                    }
                ),
            )
        )
        keys.append((track, task, profile, system))
    if len(keys) != len(set(keys)):
        raise SchemaValidationError("release support matrix contains a duplicate combination")
    return tuple(capability for _, capability in sorted(zip(keys, normalized, strict=True)))


def validate_support_payload(payload: object) -> dict[str, JsonValue]:
    """Validate the packaged support catalog consumed by the public CLI."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "release_version",
        "support_matrix",
    }:
        raise SchemaValidationError("release support catalog has unexpected or missing fields")
    if payload["schema_version"] != RELEASE_SUPPORT_SCHEMA_VERSION:
        raise SchemaValidationError("release support catalog schema_version is unsupported")
    if payload["release_version"] != RELEASE_VERSION:
        raise SchemaValidationError("release support catalog release_version is unsupported")
    matrix = validate_support_matrix(payload["support_matrix"])
    return {
        "schema_version": RELEASE_SUPPORT_SCHEMA_VERSION,
        "release_version": RELEASE_VERSION,
        "support_matrix": cast(JsonValue, list(matrix)),
    }


def load_release_support() -> dict[str, JsonValue]:
    """Load the installed release support catalog."""

    resource = files("taim").joinpath("data", RELEASE_SUPPORT_RESOURCE)
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid release support catalog JSON: {exc}") from exc
    return validate_support_payload(payload)


def require_supported_combination(
    *,
    track: str,
    task: str,
    profile: str,
    system: str,
) -> Mapping[str, JsonValue]:
    """Return one exact supported combination or fail closed."""

    key = (track, task, profile, system)
    for capability in cast(
        Sequence[Mapping[str, JsonValue]], load_release_support()["support_matrix"]
    ):
        if capability_key(cast(Mapping[str, object], capability)) == key:
            return capability
    raise ValueError(
        "unsupported release combination: "
        f"Track={track}, Task={task}, Profile={profile}, System={system}"
    )


def _official_source_bundle(track: str) -> SourceBundle:
    try:
        lock_filename = _TRACK_LOCKS[track]
    except KeyError as exc:
        raise SchemaValidationError(f"unknown release Track {track!r}") from exc
    resource = files("taim").joinpath("data", "locks", lock_filename)
    try:
        serialized = resource.read_bytes()
        payload = json.loads(serialized)
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"cannot read packaged source lock for {track}") from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "dataset_id", "lock_status", "sources"}
        or payload["dataset_id"] != track
        or payload["lock_status"] != "complete"
    ):
        raise SchemaValidationError(f"packaged source lock for {track} is not complete")
    sources = payload["sources"]
    if not isinstance(sources, list) or any(not isinstance(item, Mapping) for item in sources):
        raise SchemaValidationError(f"packaged source lock for {track} has invalid sources")
    artifacts: list[SourceArtifact] = []
    for raw in cast(list[Mapping[str, object]], sources):
        if set(raw) != {
            "role",
            "filename",
            "url",
            "byte_size",
            "sha256",
            "acquisition_date",
            "access_terms",
            "redistribution_terms",
        }:
            raise SchemaValidationError(f"packaged source lock for {track} has invalid fields")
        role = cast(str, raw["role"])
        artifacts.append(
            SourceArtifact.from_dict(
                {
                    **raw,
                    "artifact_id": role,
                }
            )
        )
    return SourceBundle(
        artifacts=tuple(artifacts),
        lock_filename=lock_filename,
        lock_sha256=f"sha256:{hashlib.sha256(serialized).hexdigest()}",
    )


def _source_artifacts(bundle: SourceBundle) -> list[dict[str, JsonValue]]:
    return [
        {
            "role": artifact.role,
            "filename": artifact.filename,
            "byte_size": artifact.byte_size,
            "sha256": artifact.sha256,
        }
        for artifact in bundle.artifacts
    ]


def _expected_preparation_identity(track: str, profile: str) -> tuple[str, str, str, SourceBundle]:
    profile = source_profile_id(profile)
    if profile in {"synthetic-patient-to-trial", "synthetic-trial-to-patient"}:
        from taim.release_fixture import fixture_preparation_identity

        return fixture_preparation_identity(track)
    if track == "trec-ct-2021" and profile in {
        "official-full",
        "trec-ct-2021-reverse-complete10",
    }:
        from taim.data.trec_ct_2021 import DATASET_ID, SOURCE_RECIPE, SOURCE_RECIPE_ID

        return (
            DATASET_ID,
            SOURCE_RECIPE_ID,
            content_sha256(SOURCE_RECIPE),
            _official_source_bundle(track),
        )
    if track in {"trec-ct-2022", "trec-ct-2023"} and profile == "confirmation-full":
        from taim.data.trec_ct import full_preparation_identity as trec_preparation_identity

        return (*trec_preparation_identity(track), _official_source_bundle(track))
    if track == "sigir-ct-2016" and profile in {"description", "summary"}:
        from taim.data.sigir_ct_2016 import (
            QueryVariant,
        )
        from taim.data.sigir_ct_2016 import (
            full_preparation_identity as sigir_preparation_identity,
        )

        return (
            *sigir_preparation_identity(cast(QueryVariant, profile)),
            _official_source_bundle(track),
        )
    raise SchemaValidationError(
        f"release Profile {profile!r} has no prepared input contract for {track!r}"
    )


def _expected_record_counts(track: str, profile: str) -> dict[str, int]:
    profile = source_profile_id(profile)
    if profile in {"synthetic-patient-to-trial", "synthetic-trial-to-patient"}:
        from taim.release_fixture import load_release_fixture

        snapshot = load_release_fixture(track).snapshot
        return {"topics": len(snapshot.topics), "trials": len(snapshot.trials)}
    try:
        return dict(_REAL_PREPARATION_RECORD_COUNTS[(track, profile)])
    except KeyError as exc:
        raise SchemaValidationError(
            f"release Profile {profile!r} has no frozen preparation effects for {track!r}"
        ) from exc


def release_source_identity(
    prepared: PreparedBenchmark,
    *,
    track: str,
    profile: str,
    evidence_scope: str,
) -> dict[str, JsonValue]:
    """Bind a generated run to its exact prepared recipe and Source Bundle."""

    bundle = prepared.source_bundle
    _, _, _, expected_bundle = _expected_preparation_identity(track, profile)
    if bundle != expected_bundle:
        raise SchemaValidationError(
            f"release Profile for {track} requires its exact packaged Source Bundle"
        )
    identity: dict[str, JsonValue] = {
        "scope": evidence_scope,
        "prepared_dataset_id": prepared.dataset_id,
        "source_recipe_id": prepared.preparation.source_recipe_id,
        "source_recipe_sha256": prepared.preparation.source_recipe_hash,
        "source_bundle_id": bundle.source_bundle_id,
        "lock_filename": bundle.lock_filename,
        "source_lock_sha256": bundle.lock_sha256,
        "record_counts": {
            "topics": len(prepared.snapshot.topics),
            "trials": len(prepared.snapshot.trials),
        },
        "artifacts": cast(JsonValue, _source_artifacts(bundle)),
    }
    validate_release_source_identity(
        identity,
        track=track,
        profile=profile,
        evidence_scope=evidence_scope,
    )
    return identity


def validate_release_source_identity(
    value: object,
    *,
    track: str,
    profile: str,
    evidence_scope: str,
) -> Mapping[str, object]:
    """Validate stored release-source lineage, including the prepared recipe."""

    if not isinstance(value, Mapping):
        raise SchemaValidationError("release run lacks its Source Bundle identity")
    require_exact_keys(value, _RELEASE_SOURCE_FIELDS, role="release Source Bundle identity")
    if value["scope"] != evidence_scope:
        raise SchemaValidationError("release Source Bundle evidence scope does not match")
    require_non_empty(value["prepared_dataset_id"], "release prepared dataset ID")
    require_non_empty(value["source_recipe_id"], "release source recipe ID")
    require_sha256(value["source_recipe_sha256"], "release source recipe SHA-256")
    require_sha256(value["source_bundle_id"], "release source_bundle_id")
    require_non_empty(value["lock_filename"], "release source lock filename")
    require_sha256(value["source_lock_sha256"], "release source lock SHA-256")
    raw_counts = value["record_counts"]
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != _RECORD_COUNT_FIELDS:
        raise SchemaValidationError("release prepared record_counts are invalid")
    for role in sorted(_RECORD_COUNT_FIELDS):
        count = raw_counts[role]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise SchemaValidationError(f"release prepared {role} count is invalid")
    raw_artifacts = value["artifacts"]
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise SchemaValidationError("release Source Bundle artifacts must be non-empty")
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise SchemaValidationError("release Source Bundle artifact must be an object")
        require_exact_keys(
            raw,
            _RELEASE_SOURCE_ARTIFACT_FIELDS,
            role="release Source Bundle artifact",
        )
        require_non_empty(raw["role"], "release source artifact role")
        require_non_empty(raw["filename"], "release source artifact filename")
        if (
            isinstance(raw["byte_size"], bool)
            or not isinstance(raw["byte_size"], int)
            or raw["byte_size"] < 0
        ):
            raise SchemaValidationError("release source artifact byte_size is invalid")
        require_sha256(raw["sha256"], "release source artifact SHA-256")
    expected_dataset_id, expected_recipe_id, expected_recipe_sha256, expected_bundle = (
        _expected_preparation_identity(track, profile)
    )
    if (
        value["prepared_dataset_id"] != expected_dataset_id
        or value["source_recipe_id"] != expected_recipe_id
        or value["source_recipe_sha256"] != expected_recipe_sha256
    ):
        raise SchemaValidationError(
            f"release Profile {profile!r} is not bound to its complete prepared recipe"
        )
    expected_counts = _expected_record_counts(track, profile)
    if dict(raw_counts) != expected_counts:
        raise SchemaValidationError(
            f"release Profile {profile!r} does not have its frozen full-preparation record counts"
        )
    expected = {
        "scope": evidence_scope,
        "prepared_dataset_id": expected_dataset_id,
        "source_recipe_id": expected_recipe_id,
        "source_recipe_sha256": expected_recipe_sha256,
        "source_bundle_id": expected_bundle.source_bundle_id,
        "lock_filename": expected_bundle.lock_filename,
        "source_lock_sha256": expected_bundle.lock_sha256,
        "record_counts": expected_counts,
        "artifacts": _source_artifacts(expected_bundle),
    }
    if dict(value) != expected:
        raise SchemaValidationError(
            f"release Profile for {track} is not bound to its exact packaged Source Bundle"
        )
    return value


def validate_reverse_release_manifest(
    manifest: TrialToPatientRunManifest,
) -> Mapping[str, object]:
    """Validate the release and source identities of one closed reverse run."""

    support = manifest.configuration.get("release_support")
    if not isinstance(support, Mapping):
        raise SchemaValidationError("reverse run lacks direction-specific release support identity")
    require_exact_keys(
        support,
        {"release_version", "track", "task", "profile", "system"},
        role="reverse release support identity",
    )
    if (
        support["release_version"] != RELEASE_VERSION
        or support["task"] != "trial_to_patient"
        or support["track"] != manifest.benchmark_lineage
        or support["profile"] != manifest.benchmark_profile["profile_id"]
        or support["system"] != manifest.system_id
    ):
        raise SchemaValidationError("reverse release support identity is direction-mismatched")
    capability = require_supported_combination(
        track=cast(str, support["track"]),
        task="trial_to_patient",
        profile=cast(str, support["profile"]),
        system=manifest.system_id,
    )
    validate_release_source_identity(
        manifest.configuration.get("release_source"),
        track=cast(str, support["track"]),
        profile=cast(str, support["profile"]),
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    return support


def validate_public_reverse_run_tree(
    directory: str | Path,
    manifest: TrialToPatientRunManifest,
) -> None:
    """Require the exact public reverse Local Run inventory, including its evaluator sidecar."""

    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise SchemaValidationError("public reverse Local Run must be a real directory")
    expected = {
        "manifest.json",
        "patient-candidates.jsonl",
        "evaluation-package.json",
        *(f"stage-{stage.name}.jsonl" for stage in manifest.stage_rankings),
    }
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.parent != root:
            raise SchemaValidationError(
                "public reverse Local Run contains a symlink, nested path, or directory"
            )
        actual.add(path.name)
    if actual != expected:
        raise SchemaValidationError("public reverse Local Run tree is incomplete or has extras")


__all__ = [
    "EXTERNAL_BASELINE_SYSTEMS",
    "EXTERNAL_FIDELITY_PROFILE",
    "FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256",
    "FROZEN_PAPER_PROTOCOL_SHA256",
    "PROFILE_EVIDENCE_SCOPE",
    "PROFILE_TASKS",
    "PROFILE_TRACKS",
    "RELEASE_SUPPORT_RESOURCE",
    "RELEASE_SUPPORT_SCHEMA_VERSION",
    "RELEASE_VERSION",
    "SUBCOMMAND_SYSTEMS",
    "SUPPORTED_PROFILES",
    "SUPPORTED_SYSTEMS",
    "SUPPORTED_TASKS",
    "SUPPORTED_TRACKS",
    "SYSTEM_TASKS",
    "capability_key",
    "load_release_support",
    "release_source_identity",
    "require_frozen_effectiveness_protocol",
    "require_frozen_paper_protocol",
    "require_supported_combination",
    "requires_effectiveness_protocol",
    "source_profile_id",
    "validate_public_reverse_run_tree",
    "validate_release_source_identity",
    "validate_reverse_release_manifest",
    "validate_support_matrix",
    "validate_support_payload",
]
