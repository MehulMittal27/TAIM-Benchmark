"""Public-safe, direction-specific Result Bundle contract for Release 0.1."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from taim.contracts import (
    content_sha256,
    portable_system_input_identity_value,
    require_exact_keys,
    require_non_empty,
    require_sha256,
    task_input_membership_id,
)
from taim.file_hash import sha256_file
from taim.judgments import (
    SIGIR_CT_2016_JUDGMENT_SCHEME,
    TREC_CT_JUDGMENT_SCHEME,
    JudgmentScheme,
)
from taim.pipeline_extensions import extension_for
from taim.public_patient_to_trial import (
    evaluate_public_patient_to_trial_run,
    load_local_evaluation_package,
    load_public_patient_to_trial_run,
)
from taim.release_profiles import (
    judgment_union_pool_receipt,
    load_profile,
    ordered_trial_ids_sha256,
    pool_receipt_policy,
)
from taim.release_support import (
    RELEASE_VERSION,
    require_frozen_effectiveness_protocol,
    require_supported_combination,
    requires_effectiveness_protocol,
    validate_public_reverse_run_tree,
    validate_release_source_identity,
    validate_reverse_release_manifest,
)
from taim.schemas import (
    PIPELINE_DEPTH_RETRIEVAL,
    PRIMARY_RANKING_CANDIDATES,
    Candidate,
    JsonValue,
    RunManifest,
    SchemaValidationError,
    StageRanking,
)
from taim.trial_to_patient import (
    TRIAL_TO_PATIENT_PRIMARY_RANKING,
    TRIAL_TO_PATIENT_SCHEMA_VERSION,
    TrialToPatientBenchmarkProfile,
    TrialToPatientCandidate,
    TrialToPatientEvaluationPackage,
    evaluate_trial_to_patient_run,
    load_trial_to_patient_run,
)

RELEASE_RESULT_BUNDLE_VERSION = "1.0"
_SHA256SUMS = "SHA256SUMS"
_BASE_BUNDLE_FILES = frozenset(
    {"bundle.json", "candidates.jsonl", "metrics.json", "run.json", _SHA256SUMS}
)
_RELEASE_IDENTITY_FIELDS = {
    "release_manifest_id",
    "public_tree_id",
    "package_artifact_id",
}
_DECLARATION_FIELDS = {
    "analysis_plan_id",
    "subset_rule",
    "metrics",
    "inclusion_rule",
    "score_guided_selection",
    "protocol_approval_id",
}
_RUN_FIELDS = {
    "artifact_type",
    "schema_version",
    "release_version",
    "track",
    "task",
    "profile",
    "system",
    "run_id",
    "prepared_snapshot_id",
    "task_input_id",
    "system_input_id",
    "evaluation_package_id",
    "candidates_sha256",
    "budget_k",
    "metric_cutoff",
    "benchmark_profile_definition_sha256",
    "benchmark_profile",
    "primary_ranking",
    "pipeline_depth",
    "protocol_approval_id",
    "source_bundle_id",
    "source_lock_sha256",
    "stage_rankings",
    "evidence_scope",
    "release_source",
    "producer_identity",
    "system_input_projection",
    "patient_entity_versions",
    "trial_entity_versions",
}
_PRODUCER_IDENTITY_FIELDS = {
    "git_commit",
    "working_tree_dirty",
    "release_identity",
    "dependency_environment",
}
_DEPENDENCY_ENVIRONMENT_FIELDS = {
    "schema_version",
    "dependency_lock_sha256",
    "python_version",
    "python_implementation",
    "platform",
    "machine",
    "distributions",
    "environment_id",
}
_DEPENDENCY_DISTRIBUTION_FIELDS = {"name", "version"}
_SYSTEM_INPUT_PROJECTION_FIELDS = {
    "system_input_id",
    "identity_version",
    "top_k",
    "metric_cutoff",
    "effective_options",
    "required_capabilities",
    "optional_capabilities",
    "model_identity",
    "index_identity",
    "omitted_local_fields",
    "projection_id",
}
_BUNDLE_FIELDS = {
    "artifact_type",
    "schema_version",
    "bundle_id",
    "release_identity",
    "run",
    "admission",
    "artifacts",
}


@dataclass(frozen=True, slots=True)
class ReleaseResultBundle:
    directory: Path
    bundle_id: str


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path, *, role: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, ValueError) as exc:
        raise SchemaValidationError(f"{role} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"{role} must be a JSON object")
    return payload


def _json_identity(value: object) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def _positive_int(value: object, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaValidationError(f"{role} must be a positive integer")
    return value


def _score(value: object, *, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise SchemaValidationError(f"{role} must be a finite score in [0, 1]")
    return float(value)


# A string that is itself an absolute filesystem location: rooted at "/" (including "//"
# network roots), home-anchored ("~/" or "~user/"), rooted at a Windows drive or UNC share, or
# a file URI. Bare names, relative paths, and other URLs are not caller-local locations.
_ABSOLUTE_FILESYSTEM_PATH = re.compile(
    r"\A(?:/|~(?:[A-Za-z_][A-Za-z0-9._-]*)?/|[A-Za-z]:[\\/]|\\\\|file:)", re.IGNORECASE
)


def _is_absolute_filesystem_path(value: str) -> bool:
    return _ABSOLUTE_FILESYSTEM_PATH.match(value) is not None


def _absolute_path_locations(value: object, *, location: str) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _absolute_path_locations(
                nested, location=f"{location}.{key}" if location else str(key)
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _absolute_path_locations(nested, location=f"{location}[{index}]")
    elif isinstance(value, str) and _is_absolute_filesystem_path(value):
        yield location


def _refuse_absolute_path_values(documents: Mapping[str, object]) -> None:
    """Refuse published JSON documents carrying a string that is itself an absolute path.

    The refusal names every offending bundle file and JSON location, but not the value, so
    the error does not repeat the caller-local location it refuses.
    """

    locations = sorted(
        f"{name} {location}"
        for name, document in documents.items()
        for location in _absolute_path_locations(document, location="")
    )
    if locations:
        raise SchemaValidationError(
            "Result Bundle publishes an absolute filesystem path at: " + "; ".join(locations)
        )


def _safe_mapping(
    value: Mapping[str, object],
    *,
    prefix: str,
) -> tuple[dict[str, JsonValue], list[str]]:
    """Drop caller-local path fields from a redistributable configuration."""

    result: dict[str, JsonValue] = {}
    omitted: list[str] = []
    for key, raw in sorted(value.items()):
        if not isinstance(key, str) or not key:
            raise SchemaValidationError("System configuration keys must be non-empty strings")
        field = f"{prefix}.{key}"
        if key in {"directory", "path"} or key.endswith("_dir") or key.endswith("_path"):
            omitted.append(field)
        else:
            safe, nested_omitted = _safe_value(raw, prefix=field)
            result[key] = safe
            omitted.extend(nested_omitted)
    return result, omitted


def _safe_value(value: object, *, prefix: str) -> tuple[JsonValue, list[str]]:
    if isinstance(value, Mapping):
        return _safe_mapping(value, prefix=prefix)
    if isinstance(value, list):
        result: list[JsonValue] = []
        omitted: list[str] = []
        for index, item in enumerate(value):
            safe, nested_omitted = _safe_value(item, prefix=f"{prefix}[{index}]")
            result.append(safe)
            omitted.extend(nested_omitted)
        return result, omitted
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError("System configuration contains a non-finite value")
        return format(value, ".17g"), []
    if value is None or isinstance(value, str | int | bool):
        return cast(JsonValue, value), []
    raise SchemaValidationError("System configuration is not JSON-compatible")


def _validate_safe_projection_value(value: object, *, prefix: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise SchemaValidationError("System Input projection key is invalid")
            if key in {"directory", "path"} or key.endswith("_dir") or key.endswith("_path"):
                raise SchemaValidationError(
                    f"System Input projection contains caller-local field {prefix}.{key}"
                )
            _validate_safe_projection_value(nested, prefix=f"{prefix}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_safe_projection_value(nested, prefix=f"{prefix}[{index}]")
        return
    if isinstance(value, float):
        raise SchemaValidationError("System Input projection must encode floats as exact strings")
    if value is not None and not isinstance(value, str | int | bool):
        raise SchemaValidationError("System Input projection contains a non-JSON value")


def _system_input_projection(
    configuration: Mapping[str, object],
    *,
    system_input_id: str,
) -> dict[str, JsonValue]:
    system_input = configuration.get("system_input")
    model_identity = configuration.get("model_identity")
    index_identity = configuration.get("index_identity")
    if not all(
        isinstance(item, Mapping) for item in (system_input, model_identity, index_identity)
    ):
        raise SchemaValidationError("run lacks redistributable System configuration")
    typed_input = cast(Mapping[str, object], system_input)
    if typed_input.get("system_input_id") != system_input_id:
        raise SchemaValidationError("embedded System Input identity does not match the run")
    options = typed_input.get("options")
    required = typed_input.get("required_capabilities")
    optional = typed_input.get("optional_capabilities")
    if not isinstance(options, Mapping):
        raise SchemaValidationError("System Input options must be an object")
    for name, capabilities in (("required", required), ("optional", optional)):
        if (
            not isinstance(capabilities, list)
            or capabilities != sorted(set(capabilities))
            or any(not isinstance(item, str) or not item for item in capabilities)
        ):
            raise SchemaValidationError(f"System Input {name} capabilities are invalid")
    identity_version = typed_input.get("identity_version")
    if identity_version != "2.0":
        raise SchemaValidationError("release run requires System Input identity_version 2.0")
    safe_options_value, option_omissions = portable_system_input_identity_value(
        options,
        path="options",
    )
    if not isinstance(safe_options_value, dict):
        raise SchemaValidationError("System Input options projection must be an object")
    safe_options = safe_options_value
    safe_model, model_omissions = _safe_mapping(
        cast(Mapping[str, object], model_identity), prefix="model_identity"
    )
    safe_index, index_omissions = _safe_mapping(
        cast(Mapping[str, object], index_identity), prefix="index_identity"
    )
    core: dict[str, JsonValue] = {
        "system_input_id": system_input_id,
        "identity_version": identity_version,
        "top_k": _positive_int(typed_input.get("top_k"), role="System Input top_k"),
        "metric_cutoff": _positive_int(
            typed_input.get("metric_cutoff"), role="System Input metric_cutoff"
        ),
        "effective_options": safe_options,
        "required_capabilities": cast(JsonValue, required),
        "optional_capabilities": cast(JsonValue, optional),
        "model_identity": safe_model,
        "index_identity": safe_index,
        "omitted_local_fields": cast(
            JsonValue,
            sorted([*option_omissions, *model_omissions, *index_omissions]),
        ),
    }
    return {**core, "projection_id": _json_identity(core)}


def _validate_system_input_projection(value: object, *, system_input_id: str) -> None:
    if not isinstance(value, Mapping):
        raise SchemaValidationError("Result Bundle System Input projection must be an object")
    require_exact_keys(value, _SYSTEM_INPUT_PROJECTION_FIELDS, role="System Input projection")
    if value["system_input_id"] != system_input_id:
        raise SchemaValidationError("System Input projection identity does not match the run")
    if value["identity_version"] != "2.0":
        raise SchemaValidationError("Result Bundle System Input identity_version is unsupported")
    require_sha256(value["system_input_id"], "System Input projection system_input_id")
    _positive_int(value["top_k"], role="System Input projection top_k")
    _positive_int(value["metric_cutoff"], role="System Input projection metric_cutoff")
    for field in ("effective_options", "model_identity", "index_identity"):
        if not isinstance(value[field], Mapping):
            raise SchemaValidationError(f"System Input projection {field} must be an object")
        _validate_safe_projection_value(value[field], prefix=field)
    for field in ("required_capabilities", "optional_capabilities", "omitted_local_fields"):
        raw = value[field]
        if (
            not isinstance(raw, list)
            or raw != sorted(set(raw))
            or any(not isinstance(item, str) or not item for item in raw)
        ):
            raise SchemaValidationError(f"System Input projection {field} is invalid")
    core = {key: value[key] for key in _SYSTEM_INPUT_PROJECTION_FIELDS - {"projection_id"}}
    require_sha256(value["projection_id"], "System Input projection ID")
    if value["projection_id"] != _json_identity(core):
        raise SchemaValidationError("System Input projection identity changed")


def _validate_projected_system_input_identity(
    value: Mapping[str, object],
    *,
    task_input_id: str,
    membership_id: str,
    system_input_id: str,
) -> None:
    expected = content_sha256(
        {
            "task_input_id": task_input_id,
            "identity_version": value["identity_version"],
            "task_input_membership_id": membership_id,
            "top_k": value["top_k"],
            "metric_cutoff": value["metric_cutoff"],
            "options": value["effective_options"],
            "required_capabilities": value["required_capabilities"],
            "optional_capabilities": value["optional_capabilities"],
        }
    )
    if expected != system_input_id:
        raise SchemaValidationError(
            "Result Bundle System Input projection does not match system_input_id"
        )


def _expected_profile(track: str, task: str, profile: str) -> tuple[str, tuple[int, ...]]:
    if task == "patient_to_trial":
        loaded = load_profile(profile)
        return loaded.definition_sha256, loaded.cutoffs
    if profile == "synthetic-trial-to-patient":
        reverse = TrialToPatientBenchmarkProfile(
            profile_id=profile,
            profile_version="1.0",
            cutoffs=(1, 3, 5),
            corpus_policy="all_packaged_synthetic_patient_versions",
            query_policy="all_packaged_synthetic_trial_versions",
            gain_mapping={"0": 0, "1": 1, "2": 2},
        )
    elif track == "trec-ct-2021" and profile == "trec-ct-2021-reverse-complete10":
        reverse = TrialToPatientBenchmarkProfile(
            profile_id=profile,
            profile_version="1.0",
            cutoffs=(5, 10, 20),
            corpus_policy="frozen_complete-ten-trial-matrix",
            query_policy="ten_frozen_complete_matrix_trial_queries",
            gain_mapping={"0": 0, "1": 1, "2": 2},
        )
    else:
        raise SchemaValidationError("Result Bundle reverse Profile is unsupported")
    return reverse.definition_sha256, reverse.cutoffs


def _validate_benchmark_profile_projection(
    value: object,
    *,
    task: str,
    profile_id: str,
    expected_definition_sha256: str,
    prepared_snapshot_id: str,
    evaluation_package_id: str,
    trial_entity_versions: frozenset[tuple[str, str]],
) -> None:
    if not isinstance(value, Mapping):
        raise SchemaValidationError("Result Bundle Benchmark Profile must be an object")
    if task == "trial_to_patient":
        reverse = TrialToPatientBenchmarkProfile.from_dict(value)
        if (
            reverse.profile_id != profile_id
            or reverse.definition_sha256 != expected_definition_sha256
        ):
            raise SchemaValidationError("Result Bundle reverse Benchmark Profile identity changed")
        return

    require_exact_keys(
        value,
        {
            "profile_id",
            "profile_version",
            "definition_sha256",
            "corpus_policy",
            "source_corpus_sha256",
            "effective_corpus_sha256",
            "effective_corpus_count",
            "pool_ids_sha256",
            "pool_source_sha256",
            "pool_receipt_id",
            "paper_pool_identity_status",
            "unverified_membership_acknowledged",
            "evaluation",
        },
        role="Result Bundle Benchmark Profile projection",
    )
    profile = load_profile(profile_id)
    expected_static = {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "definition_sha256": profile.definition_sha256,
        "corpus_policy": profile.corpus_policy,
        "paper_pool_identity_status": profile.paper_pool_identity_status,
        "unverified_membership_acknowledged": False,
        "evaluation": profile.evaluation_configuration(),
    }
    if any(value.get(field) != expected for field, expected in expected_static.items()):
        raise SchemaValidationError("Result Bundle Benchmark Profile policy changed")
    if value["definition_sha256"] != expected_definition_sha256:
        raise SchemaValidationError("Result Bundle Benchmark Profile identity changed")
    for field in ("source_corpus_sha256", "effective_corpus_sha256", "pool_ids_sha256"):
        require_sha256(value[field], f"Result Bundle Benchmark Profile {field}")
    trial_ids = tuple(entity_id for entity_id, _version_id in sorted(trial_entity_versions))
    pool_ids_sha256 = ordered_trial_ids_sha256(trial_ids)
    if (
        value["effective_corpus_count"] != len(trial_ids)
        or value["pool_ids_sha256"] != pool_ids_sha256
    ):
        raise SchemaValidationError(
            "Result Bundle Benchmark Profile does not match Task Input membership"
        )
    if profile.expected_pool_count is not None and len(trial_ids) != profile.expected_pool_count:
        raise SchemaValidationError("Result Bundle Benchmark Profile pool count changed")

    if profile.corpus_policy == "full_prepared_corpus":
        if (
            value["source_corpus_sha256"] != value["effective_corpus_sha256"]
            or value["pool_source_sha256"] is not None
            or value["pool_receipt_id"] is not None
        ):
            raise SchemaValidationError("full-corpus Result Bundle cannot claim a pool receipt")
        return

    require_sha256(value["pool_source_sha256"], "Result Bundle pool source SHA-256")
    require_sha256(value["pool_receipt_id"], "Result Bundle pool receipt ID")
    if value["pool_source_sha256"] != evaluation_package_id:
        raise SchemaValidationError("Result Bundle pool source is not its Evaluation Package")
    selection_policy, claim_scope = pool_receipt_policy(profile.profile_id)
    receipt = judgment_union_pool_receipt(
        selection_policy=selection_policy,
        claim_scope=claim_scope,
        profile=profile.profile_id,
        profile_definition_sha256=profile.definition_sha256,
        prepared_snapshot_id=prepared_snapshot_id,
        evaluation_package_id=evaluation_package_id,
        pool_count=len(trial_ids),
        pool_ids_sha256=pool_ids_sha256,
    )
    if value["pool_receipt_id"] != receipt["pool_receipt_id"]:
        raise SchemaValidationError("Result Bundle compute-bounded pool receipt changed")


def _entity_version_projection(
    rows: tuple[tuple[str, str], ...],
    *,
    entity_field: str,
    version_field: str,
) -> list[dict[str, JsonValue]]:
    return [{entity_field: entity_id, version_field: version_id} for entity_id, version_id in rows]


def _validate_entity_version_projection(
    value: object,
    *,
    entity_field: str,
    version_field: str,
) -> frozenset[tuple[str, str]]:
    if not isinstance(value, list) or not value:
        raise SchemaValidationError(f"Result Bundle {entity_field} projection must be non-empty")
    rows: list[tuple[str, str]] = []
    expected_fields = {entity_field, version_field}
    for raw in value:
        if not isinstance(raw, Mapping):
            raise SchemaValidationError("Result Bundle entity-version row must be an object")
        require_exact_keys(raw, expected_fields, role="Result Bundle entity-version row")
        entity_id = require_non_empty(raw[entity_field], f"Result Bundle {entity_field}")
        require_sha256(raw[version_field], f"Result Bundle {version_field}")
        rows.append((entity_id, cast(str, raw[version_field])))
    if rows != sorted(set(rows)):
        raise SchemaValidationError("Result Bundle entity-version rows must be sorted and unique")
    return frozenset(rows)


def _release_identity(value: Mapping[str, object]) -> dict[str, str]:
    require_exact_keys(value, _RELEASE_IDENTITY_FIELDS, role="release identity")
    result: dict[str, str] = {}
    for field in sorted(_RELEASE_IDENTITY_FIELDS):
        require_sha256(value[field], f"release identity {field}")
        result[field] = cast(str, value[field])
    return result


def validate_dependency_environment(value: object) -> dict[str, JsonValue]:
    """Validate the exact installed Python environment bound to a release run."""

    if not isinstance(value, Mapping):
        raise SchemaValidationError("release producer lacks its dependency environment")
    require_exact_keys(
        value,
        _DEPENDENCY_ENVIRONMENT_FIELDS,
        role="release dependency environment",
    )
    if value["schema_version"] != "1.0":
        raise SchemaValidationError("release dependency environment schema_version is unsupported")
    require_sha256(
        value["dependency_lock_sha256"],
        "release dependency lock SHA-256",
    )
    for field in (
        "python_version",
        "python_implementation",
        "platform",
        "machine",
    ):
        require_non_empty(value[field], f"release dependency environment {field}")
    raw_distributions = value["distributions"]
    if not isinstance(raw_distributions, list) or not raw_distributions:
        raise SchemaValidationError("release dependency distributions must be non-empty")
    distributions: list[dict[str, str]] = []
    for raw in raw_distributions:
        if not isinstance(raw, Mapping):
            raise SchemaValidationError("release dependency distribution must be an object")
        require_exact_keys(
            raw,
            _DEPENDENCY_DISTRIBUTION_FIELDS,
            role="release dependency distribution",
        )
        distributions.append(
            {
                "name": require_non_empty(raw["name"], "release dependency distribution name"),
                "version": require_non_empty(
                    raw["version"], "release dependency distribution version"
                ),
            }
        )
    if distributions != sorted(distributions, key=lambda row: (row["name"], row["version"])):
        raise SchemaValidationError("release dependency distributions must be sorted")
    names = [row["name"] for row in distributions]
    if len(names) != len(set(names)):
        raise SchemaValidationError("release dependency distribution names must be unique")
    core: dict[str, JsonValue] = {
        "schema_version": "1.0",
        "dependency_lock_sha256": cast(str, value["dependency_lock_sha256"]),
        "python_version": cast(str, value["python_version"]),
        "python_implementation": cast(str, value["python_implementation"]),
        "platform": cast(str, value["platform"]),
        "machine": cast(str, value["machine"]),
        "distributions": cast(JsonValue, distributions),
    }
    require_sha256(value["environment_id"], "release dependency environment ID")
    if value["environment_id"] != content_sha256(core):
        raise SchemaValidationError("release dependency environment ID does not match its content")
    return {**core, "environment_id": cast(str, value["environment_id"])}


def _validated_producer_identity(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError("release run lacks its producer identity")
    require_exact_keys(value, _PRODUCER_IDENTITY_FIELDS, role="release producer identity")
    git_commit = require_non_empty(value["git_commit"], "release producer git commit")
    if value["working_tree_dirty"] is not False:
        raise SchemaValidationError("Published Result Bundle producer must be a clean checkout")
    release_identity = value["release_identity"]
    if not isinstance(release_identity, Mapping):
        raise SchemaValidationError("release producer identity lacks its release identity")
    return {
        "git_commit": git_commit,
        "working_tree_dirty": False,
        "release_identity": cast(JsonValue, _release_identity(release_identity)),
        "dependency_environment": validate_dependency_environment(value["dependency_environment"]),
    }


def _validate_rrf_component_producers(
    system_input_projection: Mapping[str, object],
    *,
    producer_identity: Mapping[str, JsonValue],
) -> None:
    index_identity = system_input_projection["index_identity"]
    if not isinstance(index_identity, Mapping):
        raise SchemaValidationError("RRF Result Bundle lacks its index identity")
    require_exact_keys(
        index_identity,
        {
            "bm25_candidates_sha256",
            "dense_candidates_sha256",
            "component_producer_identities",
            "persistent",
        },
        role="RRF Result Bundle index identity",
    )
    require_sha256(index_identity["bm25_candidates_sha256"], "RRF BM25 candidate SHA-256")
    require_sha256(index_identity["dense_candidates_sha256"], "RRF dense candidate SHA-256")
    if index_identity["persistent"] is not False:
        raise SchemaValidationError("RRF Result Bundle index identity must be non-persistent")
    raw_components = index_identity["component_producer_identities"]
    if not isinstance(raw_components, Mapping) or set(raw_components) != {
        "bm25",
        "dense-bge-m3",
    }:
        raise SchemaValidationError("RRF Result Bundle component producers are incomplete")
    expected_release = producer_identity["release_identity"]
    producer_environment = cast(
        Mapping[str, object],
        producer_identity["dependency_environment"],
    )
    expected_lock = producer_environment["dependency_lock_sha256"]
    for system_id in ("bm25", "dense-bge-m3"):
        component = _validated_producer_identity(raw_components[system_id])
        if component["release_identity"] != expected_release:
            raise SchemaValidationError("RRF component producer release identity does not match")
        component_environment = cast(
            Mapping[str, object],
            component["dependency_environment"],
        )
        if component_environment["dependency_lock_sha256"] != expected_lock:
            raise SchemaValidationError("RRF component dependency lock does not match")


def _forward_producer_identity(manifest: RunManifest) -> dict[str, JsonValue]:
    execution = manifest.configuration.get("execution_provenance")
    if not isinstance(execution, Mapping) or execution.get("git_commit") != manifest.git_commit:
        raise SchemaValidationError("forward producer commit does not match its Run Manifest")
    return _validated_producer_identity(
        {
            "git_commit": manifest.git_commit,
            "working_tree_dirty": execution.get("working_tree_dirty"),
            "release_identity": manifest.configuration.get("producer_release_identity"),
            "dependency_environment": manifest.configuration.get("producer_dependency_environment"),
        }
    )


def _declaration(
    value: Mapping[str, object], *, track: str, profile: str, system: str
) -> dict[str, JsonValue]:
    require_exact_keys(value, _DECLARATION_FIELDS, role="Result Bundle declaration")
    require_sha256(value["analysis_plan_id"], "Result Bundle analysis_plan_id")
    for field in ("subset_rule", "inclusion_rule"):
        require_non_empty(value[field], f"Result Bundle {field}")
    metrics = value["metrics"]
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(not isinstance(metric, str) or not metric for metric in metrics)
        or metrics != sorted(set(metrics))
    ):
        raise SchemaValidationError("Result Bundle metrics must be sorted unique names")
    if value["score_guided_selection"] is not False:
        raise SchemaValidationError("Result Bundle subset selection must be score-independent")
    protocol = value["protocol_approval_id"]
    requires_protocol = requires_effectiveness_protocol(profile)
    if requires_protocol:
        require_frozen_effectiveness_protocol(
            profile,
            protocol,
            "Result Bundle protocol_approval_id",
            system=system,
        )
    elif protocol is not None:
        raise SchemaValidationError("Result Bundle has an inapplicable protocol approval ID")
    return cast(dict[str, JsonValue], json.loads(json.dumps(value, allow_nan=False)))


def _forward_projection(run_directory: Path) -> tuple[dict[str, JsonValue], dict[str, object]]:
    stored = load_public_patient_to_trial_run(run_directory)
    manifest = stored.manifest
    package = load_local_evaluation_package(run_directory)
    support = manifest.configuration.get("release_support")
    system_input = manifest.configuration.get("system_input")
    if not isinstance(support, Mapping) or not isinstance(system_input, Mapping):
        raise SchemaValidationError("forward run lacks release or System Input identity")
    source = manifest.configuration.get("release_source")
    if not isinstance(source, Mapping):
        raise SchemaValidationError("forward run lacks release source identity")
    profile = load_profile(cast(str, support["profile"]))
    capability = require_supported_combination(
        track=manifest.benchmark_lineage,
        task=manifest.task,
        profile=cast(str, support["profile"]),
        system=manifest.system_id,
    )
    metrics = evaluate_public_patient_to_trial_run(stored, package, profile)
    projection: dict[str, JsonValue] = {
        "artifact_type": "taim-release-run-projection",
        "schema_version": RELEASE_RESULT_BUNDLE_VERSION,
        "release_version": cast(str, support["release_version"]),
        "track": manifest.benchmark_lineage,
        "task": manifest.task,
        "profile": cast(str, support["profile"]),
        "system": manifest.system_id,
        "run_id": manifest.run_id,
        "prepared_snapshot_id": manifest.prepared_snapshot_id,
        "task_input_id": manifest.task_input_id,
        "system_input_id": manifest.system_input_id,
        "evaluation_package_id": manifest.evaluation_package_id,
        "candidates_sha256": manifest.candidates_sha256,
        "budget_k": manifest.budget_k,
        "metric_cutoff": cast(int, system_input["metric_cutoff"]),
        "benchmark_profile_definition_sha256": cast(
            str, manifest.benchmark_profile["definition_sha256"]
        ),
        "benchmark_profile": cast(JsonValue, dict(manifest.benchmark_profile)),
        "primary_ranking": manifest.primary_ranking,
        "pipeline_depth": manifest.pipeline_depth,
        "stage_rankings": cast(
            JsonValue, [stage.manifest_dict() for stage in manifest.stage_rankings]
        ),
        "protocol_approval_id": cast(JsonValue, support["protocol_approval_id"]),
        "source_bundle_id": cast(str, source["source_bundle_id"]),
        "source_lock_sha256": cast(str, source["source_lock_sha256"]),
        "evidence_scope": cast(str, capability["evidence_scope"]),
        "release_source": cast(JsonValue, dict(source)),
        "producer_identity": _forward_producer_identity(manifest),
        "system_input_projection": _system_input_projection(
            manifest.configuration,
            system_input_id=manifest.system_input_id,
        ),
        "patient_entity_versions": cast(
            JsonValue,
            _entity_version_projection(
                manifest.query_patient_versions,
                entity_field="patient_id",
                version_field="patient_version_id",
            ),
        ),
        "trial_entity_versions": cast(
            JsonValue,
            _entity_version_projection(
                manifest.trial_corpus_versions,
                entity_field="trial_id",
                version_field="trial_version_id",
            ),
        ),
    }
    return projection, metrics


def _reverse_projection(run_directory: Path) -> tuple[dict[str, JsonValue], dict[str, object]]:
    stored = load_trial_to_patient_run(run_directory)
    manifest = stored.manifest
    validate_public_reverse_run_tree(run_directory, manifest)
    support = validate_reverse_release_manifest(manifest)
    source = manifest.configuration.get("release_source")
    if not isinstance(source, Mapping):
        raise SchemaValidationError("reverse run lacks release source identity")
    try:
        package_payload = _read_json(
            run_directory / "evaluation-package.json",
            role="reverse Evaluation Package",
        )
        package = TrialToPatientEvaluationPackage.from_dict(package_payload)
    except (OSError, ValueError) as exc:
        raise SchemaValidationError("reverse Evaluation Package is invalid") from exc
    metrics = evaluate_trial_to_patient_run(stored, package)
    capability = require_supported_combination(
        track=manifest.benchmark_lineage,
        task=manifest.task,
        profile=cast(str, support["profile"]),
        system=manifest.system_id,
    )
    system_input = manifest.configuration.get("system_input")
    if not isinstance(system_input, Mapping) or not isinstance(
        system_input.get("metric_cutoff"), int
    ):
        raise SchemaValidationError("reverse run lacks its metric cutoff")
    projection: dict[str, JsonValue] = {
        "artifact_type": "taim-release-run-projection",
        "schema_version": RELEASE_RESULT_BUNDLE_VERSION,
        "release_version": cast(str, support["release_version"]),
        "track": cast(str, support["track"]),
        "task": manifest.task,
        "profile": cast(str, support["profile"]),
        "system": manifest.system_id,
        "run_id": manifest.run_id,
        "prepared_snapshot_id": manifest.prepared_snapshot_id,
        "task_input_id": manifest.task_input_id,
        "system_input_id": manifest.system_input_id,
        "evaluation_package_id": manifest.evaluation_package_id,
        "candidates_sha256": manifest.candidates_sha256,
        "budget_k": manifest.budget_k,
        "metric_cutoff": cast(int, system_input["metric_cutoff"]),
        "benchmark_profile_definition_sha256": cast(
            str, manifest.benchmark_profile["definition_sha256"]
        ),
        "benchmark_profile": cast(JsonValue, dict(manifest.benchmark_profile)),
        "primary_ranking": manifest.primary_ranking,
        "pipeline_depth": manifest.pipeline_depth,
        "stage_rankings": cast(
            JsonValue, [stage.manifest_dict() for stage in manifest.stage_rankings]
        ),
        "protocol_approval_id": None,
        "source_bundle_id": cast(str, source["source_bundle_id"]),
        "source_lock_sha256": cast(str, source["source_lock_sha256"]),
        "evidence_scope": cast(str, capability["evidence_scope"]),
        "release_source": cast(JsonValue, dict(source)),
        "producer_identity": _validated_producer_identity(
            {
                "git_commit": manifest.execution_provenance.git_commit,
                "working_tree_dirty": manifest.execution_provenance.working_tree_dirty,
                "release_identity": manifest.configuration.get("producer_release_identity"),
                "dependency_environment": manifest.configuration.get(
                    "producer_dependency_environment"
                ),
            }
        ),
        "system_input_projection": _system_input_projection(
            manifest.configuration,
            system_input_id=manifest.system_input_id,
        ),
        "patient_entity_versions": cast(
            JsonValue,
            _entity_version_projection(
                manifest.patient_corpus_versions,
                entity_field="patient_id",
                version_field="patient_version_id",
            ),
        ),
        "trial_entity_versions": cast(
            JsonValue,
            _entity_version_projection(
                manifest.query_trial_versions,
                entity_field="trial_id",
                version_field="trial_version_id",
            ),
        ),
    }
    if (
        support["release_version"] != RELEASE_VERSION
        or support["task"] != manifest.task
        or support["system"] != manifest.system_id
        or support["track"] != manifest.benchmark_lineage
        or support["profile"] != manifest.benchmark_profile["profile_id"]
    ):
        raise SchemaValidationError("reverse release support identity is direction-mismatched")
    return projection, metrics


def _projection(run_directory: Path) -> tuple[dict[str, JsonValue], dict[str, object]]:
    manifest = _read_json(run_directory / "manifest.json", role="release run manifest")
    task = manifest.get("task")
    if task == "patient_to_trial":
        return _forward_projection(run_directory)
    if task == "trial_to_patient":
        return _reverse_projection(run_directory)
    raise SchemaValidationError("release run has an unknown or missing Task")


def _artifact(path: Path, *, role: str) -> dict[str, JsonValue]:
    return {
        "path": path.name,
        "role": role,
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def _validate_metrics_payload(
    payload: Mapping[str, object],
    *,
    projection: Mapping[str, object],
    declared_metrics: list[str],
) -> None:
    identity = {
        "task": projection["task"],
        "run_id": projection["run_id"],
        "task_input_id": projection["task_input_id"],
        "evaluation_package_id": projection["evaluation_package_id"],
    }
    system_field = "system" if projection["task"] == "patient_to_trial" else "system_id"
    identity[system_field] = projection["system"]
    if any(payload.get(field) != value for field, value in identity.items()):
        raise SchemaValidationError("Result Bundle metrics do not match the run identity")
    profile_id = cast(str, projection["profile"])
    _definition, expected_cutoffs = _expected_profile(
        cast(str, projection["track"]),
        cast(str, projection["task"]),
        profile_id,
    )
    if projection["task"] == "patient_to_trial":
        require_exact_keys(
            payload,
            {
                "schema_version",
                "task",
                "release_version",
                "track",
                "profile",
                "system",
                "run_id",
                "prepared_snapshot_id",
                "task_input_id",
                "system_input_id",
                "evaluation_package_id",
                "judgment_scheme",
                "metrics_by_cutoff",
            },
            role="forward Result Bundle scorecard",
        )
        if (
            payload.get("schema_version") != RunManifest.schema_version
            or payload.get("task") != "patient_to_trial"
            or payload.get("release_version") != RELEASE_VERSION
            or payload.get("track") != projection["track"]
            or payload.get("profile") != projection["profile"]
            or payload.get("prepared_snapshot_id") != projection["prepared_snapshot_id"]
            or payload.get("system_input_id") != projection["system_input_id"]
        ):
            raise SchemaValidationError("forward Result Bundle metrics identity is incomplete")
        by_cutoff = payload.get("metrics_by_cutoff")
        expected_cutoff_keys = {str(cutoff) for cutoff in expected_cutoffs}
        if not isinstance(by_cutoff, Mapping) or set(by_cutoff) != expected_cutoff_keys:
            raise SchemaValidationError("forward Result Bundle metrics are empty")
        scheme_payload = payload["judgment_scheme"]
        if not isinstance(scheme_payload, Mapping):
            raise SchemaValidationError("forward scorecard Judgment Scheme is invalid")
        scheme = JudgmentScheme.from_dict(scheme_payload)
        profile = load_profile(profile_id)
        expected_scheme = (
            SIGIR_CT_2016_JUDGMENT_SCHEME
            if profile_id in {"description", "summary"}
            else TREC_CT_JUDGMENT_SCHEME
        )
        if scheme.to_dict() != expected_scheme.to_dict():
            raise SchemaValidationError(
                "forward scorecard Judgment Scheme does not match its Track and Profile"
            )
        precision_key = profile.precision_metric
        supplemental_precision_key = (
            "eligible_precision"
            if profile_id in {"official-full", "trec-ct-2021-external-fidelity-26149"}
            else None
        )
        for cutoff, raw_metrics in by_cutoff.items():
            if not isinstance(cutoff, str) or not isinstance(raw_metrics, Mapping):
                raise SchemaValidationError("forward Result Bundle cutoff metrics are invalid")
            cutoff_value = int(cutoff)
            cutoff_fields = {
                "judgment_scheme",
                "topic_count",
                "cutoff",
                "aggregate",
                "per_topic",
                "precision_policy",
            }
            if supplemental_precision_key is not None:
                cutoff_fields.add("supplemental_precision_policy")
            require_exact_keys(raw_metrics, cutoff_fields, role="forward cutoff scorecard")
            if raw_metrics["judgment_scheme"] != scheme.to_dict():
                raise SchemaValidationError("forward cutoff Judgment Scheme changed")
            if raw_metrics["cutoff"] != cutoff_value:
                raise SchemaValidationError("forward scorecard cutoff identity changed")
            aggregate = raw_metrics["aggregate"]
            per_topic = raw_metrics["per_topic"]
            if not isinstance(aggregate, Mapping) or not isinstance(per_topic, Mapping):
                raise SchemaValidationError("forward scorecard values must be objects")
            topic_count = _positive_int(
                raw_metrics["topic_count"], role="forward scorecard topic_count"
            )
            if topic_count != len(per_topic):
                raise SchemaValidationError("forward scorecard topic_count changed")
            recall_keys = {f"{name}_at_{cutoff}" for name in scheme.relevance_sets}
            aggregate_keys = {
                "recall",
                f"ndcg_at_{cutoff}",
                "mrr",
                f"{precision_key}_at_{cutoff}",
            }
            if supplemental_precision_key is not None:
                aggregate_keys.add(f"{supplemental_precision_key}_at_{cutoff}")
            require_exact_keys(aggregate, aggregate_keys, role="forward aggregate scorecard")
            recall = aggregate["recall"]
            if not isinstance(recall, Mapping) or set(recall) != recall_keys:
                raise SchemaValidationError("forward aggregate recall schema changed")
            scalar_keys = aggregate_keys - {"recall"}
            for key, value in recall.items():
                _score(value, role=f"forward aggregate {key}")
            for key in scalar_keys:
                _score(aggregate[key], role=f"forward aggregate {key}")
            topic_keys = recall_keys | {
                "recall",
                f"ndcg_at_{cutoff}",
                "mrr",
                "first_relevant_rank",
                f"{precision_key}_at_{cutoff}",
            }
            if supplemental_precision_key is not None:
                topic_keys.add(f"{supplemental_precision_key}_at_{cutoff}")
            topic_keys -= recall_keys
            values_by_metric: dict[str, list[float]] = {
                key: [] for key in recall_keys | scalar_keys
            }
            for topic_id, topic_metrics in per_topic.items():
                if (
                    not isinstance(topic_id, str)
                    or not topic_id
                    or not isinstance(topic_metrics, Mapping)
                ):
                    raise SchemaValidationError("forward per-topic scorecard is invalid")
                require_exact_keys(topic_metrics, topic_keys, role="forward topic scorecard")
                topic_recall = topic_metrics["recall"]
                if not isinstance(topic_recall, Mapping) or set(topic_recall) != recall_keys:
                    raise SchemaValidationError("forward topic recall schema changed")
                for key, value in topic_recall.items():
                    values_by_metric[key].append(_score(value, role=f"forward topic {key}"))
                for key in topic_keys - {"recall", "first_relevant_rank"}:
                    values_by_metric[key].append(
                        _score(topic_metrics[key], role=f"forward topic {key}")
                    )
                first = topic_metrics["first_relevant_rank"]
                if first is not None:
                    _positive_int(first, role="forward first relevant rank")
            for key, values in values_by_metric.items():
                observed = recall[key] if key in recall_keys else aggregate[key]
                if not math.isclose(
                    _score(observed, role=f"forward aggregate {key}"),
                    sum(values) / len(values),
                ):
                    raise SchemaValidationError(
                        "forward aggregate values do not match per-topic values"
                    )
            expected_precision_policy = {
                "metric": precision_key,
                "relevance_set": precision_key.removesuffix("_precision"),
                "relevance_minimum": profile.precision_relevance_minimum,
                "denominator": profile.precision_denominator,
            }
            if raw_metrics["precision_policy"] != expected_precision_policy:
                raise SchemaValidationError("forward precision policy changed")
            if supplemental_precision_key is not None:
                expected_supplemental_policy = {
                    "metric": supplemental_precision_key,
                    "relevance_set": "eligible",
                    "relevance_minimum": 2,
                    "denominator": profile.precision_denominator,
                }
                if raw_metrics["supplemental_precision_policy"] != expected_supplemental_policy:
                    raise SchemaValidationError("forward supplemental precision policy changed")
            for metric in declared_metrics:
                if metric == "ndcg":
                    present = f"ndcg_at_{cutoff}" in aggregate
                elif metric.endswith("_recall"):
                    recall = aggregate.get("recall")
                    present = isinstance(recall, Mapping) and (
                        f"{metric.removesuffix('_recall')}_at_{cutoff}" in recall
                    )
                elif metric.endswith("_precision"):
                    present = f"{metric}_at_{cutoff}" in aggregate
                else:
                    present = metric in aggregate
                if not present:
                    raise SchemaValidationError(
                        f"declared metric {metric!r} is absent from forward metrics"
                    )
    else:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "task",
                "run_id",
                "system_id",
                "task_input_id",
                "evaluation_package_id",
                "query_count",
                "primary_metric_cutoff",
                "cutoffs",
                "aggregate",
                "per_trial",
            },
            role="reverse Result Bundle scorecard",
        )
        if (
            payload.get("schema_version") != TRIAL_TO_PATIENT_SCHEMA_VERSION
            or payload.get("task") != "trial_to_patient"
        ):
            raise SchemaValidationError("reverse scorecard Task changed")
        aggregate = payload.get("aggregate")
        means = aggregate.get("mean") if isinstance(aggregate, Mapping) else None
        medians = aggregate.get("median") if isinstance(aggregate, Mapping) else None
        cutoffs = payload.get("cutoffs")
        per_trial = payload.get("per_trial")
        if (
            not isinstance(aggregate, Mapping)
            or set(aggregate) != {"mean", "median"}
            or not isinstance(means, Mapping)
            or not isinstance(medians, Mapping)
            or cutoffs != list(expected_cutoffs)
            or not isinstance(per_trial, Mapping)
            or not per_trial
        ):
            raise SchemaValidationError("reverse Result Bundle metrics are empty")
        if payload["primary_metric_cutoff"] != projection["metric_cutoff"] or payload[
            "query_count"
        ] != len(per_trial):
            raise SchemaValidationError("reverse scorecard count or cutoff changed")
        metric_keys = {
            f"{name}_at_{cutoff}"
            for cutoff in expected_cutoffs
            for name in ("ndcg", "eligible_recall", "relevant_or_eligible_recall")
        }
        if set(means) != metric_keys or set(medians) != metric_keys:
            raise SchemaValidationError("reverse aggregate metric schema changed")
        reverse_values_by_metric: dict[str, list[float]] = {key: [] for key in metric_keys}
        for trial_id, raw in per_trial.items():
            if not isinstance(trial_id, str) or not trial_id or not isinstance(raw, Mapping):
                raise SchemaValidationError("reverse per-trial scorecard is invalid")
            require_exact_keys(
                raw,
                metric_keys | {"trial_version_id"},
                role="reverse trial scorecard",
            )
            require_sha256(raw["trial_version_id"], "reverse scorecard trial_version_id")
            for key in metric_keys:
                reverse_values_by_metric[key].append(_score(raw[key], role=f"reverse trial {key}"))
        for key, values in reverse_values_by_metric.items():
            observed_mean = _score(means[key], role=f"reverse mean {key}")
            observed_median = _score(medians[key], role=f"reverse median {key}")
            if not math.isclose(observed_mean, sum(values) / len(values)) or not math.isclose(
                observed_median, float(statistics.median(values))
            ):
                raise SchemaValidationError(
                    "reverse aggregate values do not match per-trial values"
                )
        for metric in declared_metrics:
            if any(f"{metric}_at_{cutoff}" not in means for cutoff in cutoffs):
                raise SchemaValidationError(
                    f"declared metric {metric!r} is absent from reverse metrics"
                )


def build_release_result_bundle(
    run_directory: str | Path,
    output_directory: str | Path,
    *,
    release_identity: Mapping[str, object],
    declaration: Mapping[str, object],
) -> ReleaseResultBundle:
    """Build a public bundle without copying Evaluation Packages or source text."""

    source = Path(run_directory)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Result Bundle already exists: {output}")
    projection, metrics = _projection(source)
    track, task, profile, system = _validate_run_projection(projection)
    approved_release = _release_identity(release_identity)
    producer = _validated_producer_identity(projection["producer_identity"])
    if producer["release_identity"] != approved_release:
        raise SchemaValidationError("Result Bundle release identity does not match its producer")
    admission = _declaration(declaration, track=track, profile=profile, system=system)
    if admission["protocol_approval_id"] != projection["protocol_approval_id"]:
        raise SchemaValidationError("Result Bundle protocol identity does not match the run")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_json(staging / "run.json", projection)
        _write_json(staging / "metrics.json", metrics)
        source_candidates = (
            source / "candidates.jsonl"
            if task == "patient_to_trial"
            else source / "patient-candidates.jsonl"
        )
        shutil.copyfile(source_candidates, staging / "candidates.jsonl")
        artifacts = [
            _artifact(staging / "candidates.jsonl", role="direction_specific_ranking"),
            _artifact(staging / "metrics.json", role="direction_specific_metrics"),
            _artifact(staging / "run.json", role="run_identity_projection"),
        ]
        stage_paths: list[str] = []
        raw_stages = projection["stage_rankings"]
        if not isinstance(raw_stages, list):
            raise SchemaValidationError("Result Bundle stage ranking declaration is invalid")
        for raw_stage in raw_stages:
            stage = StageRanking.from_manifest_dict(raw_stage)
            name = f"stage-{stage.name}.jsonl"
            shutil.copyfile(source / name, staging / name)
            artifacts.append(_artifact(staging / name, role=f"stage_ranking:{stage.name}"))
            stage_paths.append(name)
        core: dict[str, object] = {
            "artifact_type": "taim-release-result-bundle",
            "schema_version": RELEASE_RESULT_BUNDLE_VERSION,
            "release_identity": approved_release,
            "run": projection,
            "admission": admission,
            "artifacts": artifacts,
        }
        manifest = {**core, "bundle_id": content_sha256(core)}
        _write_json(staging / "bundle.json", manifest)
        checksum_paths = (
            "bundle.json",
            "candidates.jsonl",
            "metrics.json",
            "run.json",
            *stage_paths,
        )
        (staging / _SHA256SUMS).write_text(
            "".join(
                f"{sha256_file(staging / name).removeprefix('sha256:')}  {name}\n"
                for name in checksum_paths
            ),
            encoding="utf-8",
            newline="\n",
        )
        report = validate_release_result_bundle(staging)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ReleaseResultBundle(output.resolve(), cast(str, report["bundle_id"]))


def _bundle_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SchemaValidationError("Result Bundle contains a symlink")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            PurePosixPath(relative)
            result[relative] = path
    if not set(result) >= _BASE_BUNDLE_FILES or any(
        relative not in _BASE_BUNDLE_FILES
        and not (relative.startswith("stage-") and relative.endswith(".jsonl"))
        for relative in result
    ):
        raise SchemaValidationError("Result Bundle tree is incomplete or contains extra files")
    return result


def _expected_pipeline_depth(system: str) -> str:
    extension = extension_for(system)
    if extension is not None:
        return extension.pipeline_depth
    return PIPELINE_DEPTH_RETRIEVAL


def _validate_run_projection(projection: Mapping[str, object]) -> tuple[str, str, str, str]:
    require_exact_keys(projection, _RUN_FIELDS, role="Result Bundle run projection")
    if (
        projection["artifact_type"] != "taim-release-run-projection"
        or projection["schema_version"] != RELEASE_RESULT_BUNDLE_VERSION
        or projection["release_version"] != RELEASE_VERSION
    ):
        raise SchemaValidationError("Result Bundle run projection version is unsupported")
    for field in ("track", "task", "profile", "system", "run_id"):
        require_non_empty(projection[field], f"Result Bundle run {field}")
    for field in (
        "prepared_snapshot_id",
        "task_input_id",
        "system_input_id",
        "evaluation_package_id",
        "candidates_sha256",
        "benchmark_profile_definition_sha256",
        "source_bundle_id",
        "source_lock_sha256",
    ):
        require_sha256(projection[field], f"Result Bundle run {field}")
    budget_k = _positive_int(projection["budget_k"], role="Result Bundle budget_k")
    metric_cutoff = _positive_int(projection["metric_cutoff"], role="Result Bundle metric_cutoff")
    track = cast(str, projection["track"])
    task = cast(str, projection["task"])
    profile = cast(str, projection["profile"])
    system = cast(str, projection["system"])
    capability = require_supported_combination(
        track=track, task=task, profile=profile, system=system
    )
    expected_evidence_scope = cast(str, capability["evidence_scope"])
    if projection["evidence_scope"] != expected_evidence_scope:
        raise SchemaValidationError("Result Bundle evidence scope does not match its capability")
    source_identity = validate_release_source_identity(
        projection["release_source"],
        track=track,
        profile=profile,
        evidence_scope=expected_evidence_scope,
    )
    if (
        projection["source_bundle_id"] != source_identity["source_bundle_id"]
        or projection["source_lock_sha256"] != source_identity["source_lock_sha256"]
    ):
        raise SchemaValidationError("Result Bundle source projection identities disagree")
    expected_definition, cutoffs = _expected_profile(track, task, profile)
    if (
        projection["benchmark_profile_definition_sha256"] != expected_definition
        or metric_cutoff not in cutoffs
    ):
        raise SchemaValidationError("Result Bundle Benchmark Profile identity changed")
    expected_primary = (
        PRIMARY_RANKING_CANDIDATES
        if task == "patient_to_trial"
        else TRIAL_TO_PATIENT_PRIMARY_RANKING
    )
    expected_pipeline_depth = _expected_pipeline_depth(system)
    if (
        projection["primary_ranking"] != expected_primary
        or projection["pipeline_depth"] != expected_pipeline_depth
    ):
        raise SchemaValidationError("Result Bundle ranking semantics changed")
    raw_stages = projection["stage_rankings"]
    if not isinstance(raw_stages, list):
        raise SchemaValidationError("Result Bundle stage rankings must be an array")
    stages = tuple(StageRanking.from_manifest_dict(raw) for raw in raw_stages)
    if len({stage.name for stage in stages}) != len(stages):
        raise SchemaValidationError("Result Bundle stage ranking names must be unique")
    extension = extension_for(system)
    stage_depths: Mapping[str, tuple[str, int]] = (
        extension.stage_depths if extension is not None else {}
    )
    if stage_depths and (
        tuple(stage.name for stage in stages) != tuple(stage_depths)
        or any(stage.pipeline_depth != stage_depths[stage.name][0] for stage in stages)
    ):
        raise SchemaValidationError("staged Result Bundle stage declaration is incomplete")
    protocol = projection["protocol_approval_id"]
    if requires_effectiveness_protocol(profile):
        require_frozen_effectiveness_protocol(
            profile,
            protocol,
            "Result Bundle protocol approval ID",
            system=system,
        )
    elif protocol is not None:
        raise SchemaValidationError("Result Bundle has an inapplicable protocol approval ID")
    _validate_system_input_projection(
        projection["system_input_projection"],
        system_input_id=cast(str, projection["system_input_id"]),
    )
    projected_input = cast(Mapping[str, object], projection["system_input_projection"])
    if projected_input["top_k"] != budget_k or projected_input["metric_cutoff"] != metric_cutoff:
        raise SchemaValidationError("Result Bundle System Input budget does not match the run")
    patient_entity_versions = _validate_entity_version_projection(
        projection["patient_entity_versions"],
        entity_field="patient_id",
        version_field="patient_version_id",
    )
    trial_entity_versions = _validate_entity_version_projection(
        projection["trial_entity_versions"],
        entity_field="trial_id",
        version_field="trial_version_id",
    )
    _validate_benchmark_profile_projection(
        projection["benchmark_profile"],
        task=task,
        profile_id=profile,
        expected_definition_sha256=expected_definition,
        prepared_snapshot_id=cast(str, projection["prepared_snapshot_id"]),
        evaluation_package_id=cast(str, projection["evaluation_package_id"]),
        trial_entity_versions=trial_entity_versions,
    )
    membership_id = task_input_membership_id(
        task=task,
        patient_entity_versions=patient_entity_versions,
        trial_entity_versions=trial_entity_versions,
    )
    _validate_projected_system_input_identity(
        projected_input,
        task_input_id=cast(str, projection["task_input_id"]),
        membership_id=membership_id,
        system_input_id=cast(str, projection["system_input_id"]),
    )
    producer_identity = _validated_producer_identity(projection["producer_identity"])
    if system == "rrf":
        _validate_rrf_component_producers(
            projected_input,
            producer_identity=producer_identity,
        )
    elif extension is not None and extension.validate_component_producers is not None:
        extension.validate_component_producers(projected_input, producer_identity)
    return track, task, profile, system


def _validate_ranking(
    lines: list[str],
    *,
    task: str,
    run_id: str,
    system: str,
    budget_k: int,
    patient_entity_versions: frozenset[tuple[str, str]],
    trial_entity_versions: frozenset[tuple[str, str]],
) -> None:
    if not lines or any(not line for line in lines):
        raise SchemaValidationError("Result Bundle ranking is empty or non-canonical")
    if task == "patient_to_trial":
        forward_rows = tuple(Candidate.from_json(line) for line in lines)
        if lines != [row.to_json() for row in forward_rows]:
            raise SchemaValidationError("forward Result Bundle ranking is not canonical")
        if [(row.topic_id, row.rank) for row in forward_rows] != sorted(
            (row.topic_id, row.rank) for row in forward_rows
        ):
            raise SchemaValidationError("forward Result Bundle ranking order is not canonical")
        common_rows: tuple[Candidate | TrialToPatientCandidate, ...] = forward_rows
        pairs = [(row.topic_id, row.trial_id) for row in forward_rows]
        ranks = [(row.topic_id, row.rank) for row in forward_rows]
        patient_ids = {entity_id for entity_id, _version_id in patient_entity_versions}
        trial_ids = {entity_id for entity_id, _version_id in trial_entity_versions}
        if any(
            row.topic_id not in patient_ids or row.trial_id not in trial_ids for row in forward_rows
        ):
            raise SchemaValidationError(
                "forward Result Bundle ranking is outside its Task Input membership"
            )
        expected_ranks = set(range(1, min(budget_k, len(trial_ids)) + 1))
        if any(
            {row.rank for row in forward_rows if row.topic_id == patient_id} != expected_ranks
            for patient_id in patient_ids
        ):
            raise SchemaValidationError(
                "forward Result Bundle ranking is incomplete at the declared budget"
            )
    elif task == "trial_to_patient":
        reverse_rows = tuple(TrialToPatientCandidate.from_json(line) for line in lines)
        if lines != [row.to_json() for row in reverse_rows]:
            raise SchemaValidationError("reverse Result Bundle ranking is not canonical")
        if [(row.trial_id, row.rank) for row in reverse_rows] != sorted(
            (row.trial_id, row.rank) for row in reverse_rows
        ):
            raise SchemaValidationError("reverse Result Bundle ranking order is not canonical")
        common_rows = reverse_rows
        pairs = [(row.trial_version_id, row.patient_version_id) for row in reverse_rows]
        ranks = [(row.trial_version_id, row.rank) for row in reverse_rows]
        if any(
            (row.trial_id, row.trial_version_id) not in trial_entity_versions
            or (row.patient_id, row.patient_version_id) not in patient_entity_versions
            for row in reverse_rows
        ):
            raise SchemaValidationError(
                "reverse Result Bundle ranking is outside its Task Input membership"
            )
        expected_ranks = set(range(1, min(budget_k, len(patient_entity_versions)) + 1))
        if any(
            {row.rank for row in reverse_rows if row.trial_version_id == trial_version_id}
            != expected_ranks
            for _trial_id, trial_version_id in trial_entity_versions
        ):
            raise SchemaValidationError(
                "reverse Result Bundle ranking is incomplete at the declared budget"
            )
    else:
        raise SchemaValidationError("Result Bundle Task is unsupported")
    if any(
        row.run_id != run_id or row.system_id != system or row.rank > budget_k
        for row in common_rows
    ):
        raise SchemaValidationError("Result Bundle ranking is run-mismatched or over budget")
    if len(pairs) != len(set(pairs)) or len(ranks) != len(set(ranks)):
        raise SchemaValidationError("Result Bundle ranking contains duplicate pairs or ranks")


def validate_release_result_bundle(directory: str | Path) -> dict[str, object]:
    """Validate hashes, capability identity, direction, and admission policy."""

    root = Path(directory)
    files = _bundle_files(root)
    manifest = _read_json(files["bundle.json"], role="Result Bundle manifest")
    require_exact_keys(manifest, _BUNDLE_FIELDS, role="Result Bundle manifest")
    if (
        manifest["artifact_type"] != "taim-release-result-bundle"
        or manifest["schema_version"] != RELEASE_RESULT_BUNDLE_VERSION
    ):
        raise SchemaValidationError("unsupported Result Bundle")
    projection = manifest["run"]
    if not isinstance(projection, Mapping):
        raise SchemaValidationError("Result Bundle run projection must be an object")
    track, task, profile, system = _validate_run_projection(projection)
    if _read_json(files["run.json"], role="Result Bundle run") != projection:
        raise SchemaValidationError("Result Bundle run copies disagree")
    release_identity = manifest["release_identity"]
    admission = manifest["admission"]
    if not isinstance(release_identity, Mapping) or not isinstance(admission, Mapping):
        raise SchemaValidationError("Result Bundle identity or admission must be an object")
    _release_identity(release_identity)
    producer = _validated_producer_identity(projection["producer_identity"])
    if producer["release_identity"] != _release_identity(release_identity):
        raise SchemaValidationError("Result Bundle release identity does not match its producer")
    validated_admission = _declaration(admission, track=track, profile=profile, system=system)
    if validated_admission["protocol_approval_id"] != projection["protocol_approval_id"]:
        raise SchemaValidationError("Result Bundle protocol identity does not match the run")
    raw_stages = projection["stage_rankings"]
    if not isinstance(raw_stages, list):
        raise SchemaValidationError("Result Bundle stage ranking declaration is invalid")
    stages = tuple(StageRanking.from_manifest_dict(raw) for raw in raw_stages)
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 3 + len(stages):
        raise SchemaValidationError("Result Bundle artifact inventory is incomplete")
    expected_artifacts = {
        "candidates.jsonl": "direction_specific_ranking",
        "metrics.json": "direction_specific_metrics",
        "run.json": "run_identity_projection",
        **{f"stage-{stage.name}.jsonl": f"stage_ranking:{stage.name}" for stage in stages},
    }
    observed_artifacts: set[str] = set()
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            raise SchemaValidationError("Result Bundle artifact entry must be an object")
        require_exact_keys(raw, {"path", "role", "sha256", "byte_size"}, role="bundle artifact")
        path = cast(str, raw["path"])
        if path not in expected_artifacts or raw["role"] != expected_artifacts[path]:
            raise SchemaValidationError("Result Bundle artifact role is invalid")
        if path in observed_artifacts:
            raise SchemaValidationError("Result Bundle artifact inventory contains duplicates")
        observed_artifacts.add(path)
        if (
            raw["sha256"] != sha256_file(files[path])
            or raw["byte_size"] != files[path].stat().st_size
        ):
            raise SchemaValidationError("Result Bundle artifact identity changed")
    if observed_artifacts != set(expected_artifacts):
        raise SchemaValidationError("Result Bundle artifact inventory is incomplete")
    metrics_payload = _read_json(files["metrics.json"], role="Result Bundle metrics")
    # A Result Bundle must not publish a caller-local filesystem location. The System Input
    # projection drops only path-shaped keys, and it cannot drop more without changing
    # system_input_id, so a path under any other key reaches this point: for example a provider's
    # executable path when the caller passes an absolute one. bundle.json (which embeds the run
    # projection, the
    # admission declaration, and the artifact inventory), run.json and metrics.json are scanned
    # here. The ranking files and SHA256SUMS need no scan: the checks below admit only ranking
    # rows whose strings are run.json's run_id, System, and Task Input entity and entity-version
    # IDs, and only checksum rows naming the fixed bundle files with their digests.
    #
    # A System whose run options resolve to absolute paths therefore cannot publish a bundle.
    # Where that is deliberate, the pipeline extension module that owns the System records why.
    # Do not loosen this check to admit such a System.
    _refuse_absolute_path_values(
        {"bundle.json": manifest, "run.json": projection, "metrics.json": metrics_payload}
    )
    declared_metrics = validated_admission["metrics"]
    if not isinstance(declared_metrics, list):
        raise SchemaValidationError("Result Bundle declared metrics are invalid")
    _validate_metrics_payload(
        metrics_payload,
        projection=projection,
        declared_metrics=cast(list[str], declared_metrics),
    )
    if projection["candidates_sha256"] != sha256_file(files["candidates.jsonl"]):
        raise SchemaValidationError("Result Bundle candidates do not match the run")
    lines = files["candidates.jsonl"].read_text(encoding="utf-8").splitlines()
    patient_entity_versions = _validate_entity_version_projection(
        projection["patient_entity_versions"],
        entity_field="patient_id",
        version_field="patient_version_id",
    )
    trial_entity_versions = _validate_entity_version_projection(
        projection["trial_entity_versions"],
        entity_field="trial_id",
        version_field="trial_version_id",
    )
    _validate_ranking(
        lines,
        task=task,
        run_id=cast(str, projection["run_id"]),
        system=system,
        budget_k=cast(int, projection["budget_k"]),
        patient_entity_versions=patient_entity_versions,
        trial_entity_versions=trial_entity_versions,
    )
    extension = extension_for(system)
    stage_depths: Mapping[str, tuple[str, int]] = (
        extension.stage_depths if extension is not None else {}
    )
    for stage in stages:
        stage_name = f"stage-{stage.name}.jsonl"
        if stage.artifact_hash != sha256_file(files[stage_name]):
            raise SchemaValidationError("Result Bundle stage ranking hash changed")
        stage_lines = files[stage_name].read_text(encoding="utf-8").splitlines()
        if stage_depths:
            stage_depth = stage_depths[stage.name][1]
        elif task == "patient_to_trial":
            stage_depth = max(Candidate.from_json(line).rank for line in stage_lines)
        else:
            stage_depth = max(TrialToPatientCandidate.from_json(line).rank for line in stage_lines)
        _validate_ranking(
            stage_lines,
            task=task,
            run_id=cast(str, projection["run_id"]),
            system=system,
            budget_k=stage_depth,
            patient_entity_versions=patient_entity_versions,
            trial_entity_versions=trial_entity_versions,
        )
    expected_checksums = [
        "bundle.json",
        "candidates.jsonl",
        "metrics.json",
        "run.json",
        *(f"stage-{stage.name}.jsonl" for stage in stages),
    ]
    observed: list[str] = []
    for line in files[_SHA256SUMS].read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise SchemaValidationError("Result Bundle checksum row is invalid") from exc
        observed.append(relative)
        if relative not in files or digest != sha256_file(files[relative]).removeprefix("sha256:"):
            raise SchemaValidationError("Result Bundle checksum mismatch")
    if observed != expected_checksums:
        raise SchemaValidationError("Result Bundle checksums do not cover the exact tree")
    core = {key: manifest[key] for key in _BUNDLE_FIELDS - {"bundle_id"}}
    if manifest["bundle_id"] != content_sha256(core):
        raise SchemaValidationError("Result Bundle identity changed")
    return {
        "schema_version": RELEASE_RESULT_BUNDLE_VERSION,
        "bundle_id": manifest["bundle_id"],
        "release_manifest_id": release_identity["release_manifest_id"],
        "public_tree_id": release_identity["public_tree_id"],
        "package_artifact_id": release_identity["package_artifact_id"],
        "track": track,
        "task": task,
        "profile": profile,
        "system": system,
        "run_id": projection["run_id"],
    }


__all__ = [
    "RELEASE_RESULT_BUNDLE_VERSION",
    "ReleaseResultBundle",
    "build_release_result_bundle",
    "validate_release_result_bundle",
]
