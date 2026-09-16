"""Dependency-neutral primitives shared by versioned TAIM contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import fields, is_dataclass
from pathlib import PurePath
from typing import cast

from taim.schemas import JsonValue, SchemaValidationError, freeze_json_value, json_value_to_builtins

SNAPSHOT_CONTRACT_VERSION = "2.0"
SNAPSHOT_CANONICALIZATION_VERSION = "taim-snapshot-canonical-json-v1"
ELIGIBILITY_SPLIT_VERSION = "taim-eligibility-criteria-split-v3"
ELIGIBILITY_CRITERION_VIEW_NAME = "eligibility_criteria"
ELIGIBILITY_CRITERION_VIEW_CONTENT_SCHEMA = "taim-eligibility-criteria-view-v1"
EVALUATION_PACKAGE_VERSION = "1.0"
SOURCE_BUNDLE_VERSION = "1.0"

_SHA256 = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_OPEN_INFERENCE_SURFACE_FIELDS = frozenset(
    {"additional_fields", "extensions", "registry_extensions", "derived_evidence"}
)


class EvaluatorOnlyMaterial:
    """Marker base for objects that must never enter System-visible inputs."""


def require_closed_inference_surfaces(value: object, *, path: str) -> None:
    """Reject every populated open entity surface before data reaches a System."""

    if is_dataclass(value) and not isinstance(value, type):
        for item_field in fields(value):
            item = getattr(value, item_field.name)
            item_path = f"{path}.{item_field.name}"
            if item_field.name in _OPEN_INFERENCE_SURFACE_FIELDS and item:
                raise SchemaValidationError(f"input uses open inference surface {item_path}")
            require_closed_inference_surfaces(item, path=item_path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in _OPEN_INFERENCE_SURFACE_FIELDS and item:
                raise SchemaValidationError(f"input uses open inference surface {item_path}")
            require_closed_inference_surfaces(item, path=item_path)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            require_closed_inference_surfaces(item, path=f"{path}[{index}]")


def require_non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must be a non-empty string")
    return value


def require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SchemaValidationError(f"{name} must be a prefixed lowercase SHA-256 hash")
    return value


def normalize_sha256(value: str) -> str:
    """Normalize a lowercase SHA-256 hex digest to TAIM's prefixed representation."""

    candidate = value if value.startswith("sha256:") else f"sha256:{value}"
    return require_sha256(candidate, "SHA-256")


def require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    role: str,
) -> None:
    """Reject missing and unknown fields in a fail-closed serialized contract."""

    missing = expected - payload.keys()
    unexpected = payload.keys() - expected
    if missing or unexpected:
        raise SchemaValidationError(
            f"{role} fields mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def system_input_identity_value(value: object, *, path: str) -> JsonValue:
    """Canonicalize options and other request-local values for identity hashing."""

    if value is None or isinstance(value, (bool, int, str)):
        return cast(JsonValue, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"{path} must contain finite numbers")
        return {"value_type": "binary64", "value": repr(value)}
    if isinstance(value, PurePath):
        return {"value_type": "path", "value": value.as_posix()}
    if isinstance(value, Mapping):
        return {
            cast(str, key): system_input_identity_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items())
        }
    if isinstance(value, list | tuple):
        return [
            system_input_identity_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, set | frozenset):
        normalized = [system_input_identity_value(item, path=f"{path}[]") for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    raise SchemaValidationError(
        f"{path} contains an unsupported identity value {type(value).__name__}"
    )


def portable_system_input_identity_value(
    value: object,
    *,
    path: str,
) -> tuple[JsonValue, tuple[str, ...]]:
    """Canonicalize a public System Input while omitting caller-local path fields."""

    normalized = system_input_identity_value(value, path=path)
    omitted: list[str] = []

    def strip(item: JsonValue, *, item_path: str) -> JsonValue:
        if isinstance(item, dict):
            result: dict[str, JsonValue] = {}
            for key, nested in sorted(item.items()):
                nested_path = f"{item_path}.{key}"
                if key in {"directory", "path"} or key.endswith("_dir") or key.endswith("_path"):
                    omitted.append(nested_path)
                else:
                    result[key] = strip(nested, item_path=nested_path)
            return result
        if isinstance(item, list):
            return [
                strip(nested, item_path=f"{item_path}[{index}]")
                for index, nested in enumerate(item)
            ]
        return item

    return strip(normalized, item_path=path), tuple(sorted(omitted))


def validate_json(value: object, path: str = "value") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        raise SchemaValidationError(
            f"{path} must not contain binary floating-point values; use exact strings"
        )
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"{path} keys must be strings")
            validate_json(item, f"{path}.{key}")
        return
    raise SchemaValidationError(f"{path} contains a non-JSON value: {type(value).__name__}")


def freeze_json_mapping(value: Mapping[str, JsonValue], *, path: str) -> Mapping[str, JsonValue]:
    """Validate, detach, and recursively freeze one exact JSON mapping."""

    validate_json(value, path)
    return cast(Mapping[str, JsonValue], freeze_json_value(json_value_to_builtins(value)))


def iter_canonical_json(payload: object, path: str = "canonical payload") -> Iterator[str]:
    """Yield established canonical JSON chunks without materializing the whole value."""

    if payload is None or isinstance(payload, (bool, int, str)):
        yield json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return
    if isinstance(payload, float):
        raise SchemaValidationError(
            f"{path} must not contain binary floating-point values; use exact strings"
        )
    if isinstance(payload, list | tuple):
        yield "["
        for index, item in enumerate(payload):
            if index:
                yield ","
            yield from iter_canonical_json(item, f"{path}[{index}]")
        yield "]"
        return
    if isinstance(payload, Mapping):
        keys = list(payload)
        for key in keys:
            if not isinstance(key, str):
                raise SchemaValidationError(f"{path} keys must be strings")
        yield "{"
        for index, key in enumerate(sorted(cast(list[str], keys))):
            if index:
                yield ","
            yield json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            yield ":"
            yield from iter_canonical_json(payload[key], f"{path}.{key}")
        yield "}"
        return
    raise SchemaValidationError(f"{path} contains a non-JSON value: {type(payload).__name__}")


def canonical_json(payload: object) -> str:
    validate_json(payload, "canonical payload")
    return json.dumps(
        json_value_to_builtins(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def content_sha256(payload: object) -> str:
    digest = hashlib.sha256()
    buffer = bytearray()
    for chunk in iter_canonical_json(payload):
        buffer.extend(chunk.encode("utf-8"))
        if len(buffer) >= 1024 * 1024:
            digest.update(buffer)
            buffer.clear()
    digest.update(buffer)
    return "sha256:" + digest.hexdigest()


def task_input_membership_id(
    *,
    task: str,
    patient_entity_versions: Iterable[tuple[str, str]],
    trial_entity_versions: Iterable[tuple[str, str]],
) -> str:
    """Commit the public-safe entity/version membership of one Task Input."""

    require_non_empty(task, "Task Input membership task")

    def normalize(
        rows: Iterable[tuple[str, str]],
        *,
        entity_field: str,
        version_field: str,
    ) -> list[dict[str, str]]:
        normalized: list[tuple[str, str]] = []
        for entity_id, version_id in rows:
            require_non_empty(entity_id, f"Task Input membership {entity_field}")
            require_sha256(version_id, f"Task Input membership {version_field}")
            normalized.append((entity_id, version_id))
        if not normalized or len(normalized) != len(set(normalized)):
            raise SchemaValidationError(
                f"Task Input membership {entity_field} rows must be non-empty and unique"
            )
        normalized.sort()
        return [
            {entity_field: entity_id, version_field: version_id}
            for entity_id, version_id in normalized
        ]

    return content_sha256(
        {
            "task": task,
            "patient_entity_versions": normalize(
                patient_entity_versions,
                entity_field="patient_id",
                version_field="patient_version_id",
            ),
            "trial_entity_versions": normalize(
                trial_entity_versions,
                entity_field="trial_id",
                version_field="trial_version_id",
            ),
        }
    )
