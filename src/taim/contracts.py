"""Dependency-neutral primitives shared by versioned TAIM contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import PurePath
from typing import Any, cast

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


#: Why a declared number is that number. A recorded reason is one of the first six kinds; the last
#: two record its absence, and they differ on purpose:
#:
#: - ``decided``: someone chose this value and a record argues for it; ``ref`` cites the record.
#: - ``derived``: the value follows from other values by a stated rule; ``ref`` states the rule.
#: - ``measured``: a measurement chose or confirmed this value; ``ref`` cites the evidence.
#: - ``inherited``: the value came from elsewhere without a decision as that number; ``ref`` names
#:   the source, such as an upstream ``source@commit`` or an earlier script.
#: - ``control``: the value is held fixed so a comparison is not confounded, and its magnitude was
#:   not chosen; ``ref`` names the comparison.
#: - ``fact``: the number is a property of an artefact - a byte size, a count, a model's output
#:   dimension, a reported result - not a choice; ``ref`` names what it is re-derivable from.
#: - ``unknown``: the record was searched and cannot say; ``ref`` states what was searched.
#: - ``unwritten``: nobody has written the reason down, and only its owner can; ``ref`` states what
#:   is established and the question that would settle it. A basis check counts it as missing.
PARAMETER_BASIS_KINDS = (
    "decided",
    "derived",
    "measured",
    "inherited",
    "control",
    "fact",
    "unknown",
    "unwritten",
)
_PARAMETER_BASIS_FIELDS = frozenset({"kind", "ref", "reviewed_by", "reviewed_at"})
_PARAMETER_RECORD_FIELDS = frozenset({"value", "basis"})
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


@dataclass(frozen=True, slots=True)
class ParameterBasis:
    """Why one declared number is that number: recorded beside the value, never hashed with it."""

    kind: str
    ref: str
    reviewed_by: str
    reviewed_at: str

    def __post_init__(self) -> None:
        if self.kind not in PARAMETER_BASIS_KINDS:
            raise SchemaValidationError(
                f"parameter basis kind must be one of {', '.join(PARAMETER_BASIS_KINDS)}"
            )
        require_non_empty(self.ref, "parameter basis ref")
        require_non_empty(self.reviewed_by, "parameter basis reviewed_by")
        if not isinstance(self.reviewed_at, str) or _DATE.fullmatch(self.reviewed_at) is None:
            raise SchemaValidationError("parameter basis reviewed_at must be a YYYY-MM-DD date")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "ref": self.ref,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
        }


def parameter_basis(raw: object, *, path: str) -> ParameterBasis:
    """Read one serialized basis, refusing any shape but its exact four fields."""

    if isinstance(raw, ParameterBasis):
        return raw
    if not isinstance(raw, Mapping) or set(raw) != _PARAMETER_BASIS_FIELDS:
        raise SchemaValidationError(
            f"{path} basis must be an object with exactly {sorted(_PARAMETER_BASIS_FIELDS)}"
        )
    try:
        return ParameterBasis(
            kind=raw["kind"],
            ref=raw["ref"],
            reviewed_by=raw["reviewed_by"],
            reviewed_at=raw["reviewed_at"],
        )
    except SchemaValidationError as exc:
        raise SchemaValidationError(f"{path}: {exc}") from exc


def _pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def is_parameter_record(value: object) -> bool:
    """A parameter record is an object whose ``basis`` is itself an object.

    An ordinary object may carry a prose field that happens to be named ``basis`` - a string - and
    is not a record; a number inside it is still refused as a bare number.
    """

    return isinstance(value, Mapping) and isinstance(value.get("basis"), Mapping)


def split_parameter_records(
    payload: object, *, role: str
) -> tuple[JsonValue, dict[str, tuple[JsonValue, ParameterBasis]]]:
    """Split a method declaration into its hashed projection and the basis of every parameter.

    A method declaration writes every number as a parameter record ``{"value": ..., "basis":
    {...}}``. The projection replaces each record with its value, so it is byte for byte the
    declaration as it was before any basis was recorded: identities hash the projection, and
    writing or amending a basis moves none of them. The bases come back keyed by the record's JSON
    pointer. A bare number anywhere outside a record is refused - the schema has a slot for its
    reason, and an unrecorded reason is written ``unwritten``, never left out.
    """

    bases: dict[str, tuple[JsonValue, ParameterBasis]] = {}

    def refuse_nested_numbers(value: object, pointer: str) -> None:
        if is_parameter_record(value):
            raise SchemaValidationError(f"{role} {pointer}: parameter records may not nest")
        if isinstance(value, Mapping):
            for name, item in value.items():
                refuse_nested_numbers(item, f"{pointer}/{_pointer_token(str(name))}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                refuse_nested_numbers(item, f"{pointer}/{index}")

    def project(value: object, pointer: str) -> JsonValue:
        if is_parameter_record(value):
            record = cast(Mapping[str, object], value)
            if set(record) != _PARAMETER_RECORD_FIELDS:
                raise SchemaValidationError(
                    f"{role} {pointer}: a parameter record has exactly the fields value and basis"
                )
            refuse_nested_numbers(record["value"], f"{pointer}/value")
            bases[pointer] = (
                cast(JsonValue, record["value"]),
                parameter_basis(record["basis"], path=f"{role} {pointer}"),
            )
            return cast(JsonValue, record["value"])
        if isinstance(value, Mapping):
            return {
                str(name): project(item, f"{pointer}/{_pointer_token(str(name))}")
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [project(item, f"{pointer}/{index}") for index, item in enumerate(value)]
        if _is_number(value):
            raise SchemaValidationError(
                f"{role} {pointer or '/'}: a number in a method declaration must be a "
                '{"value", "basis"} parameter record'
            )
        return cast(JsonValue, value)

    return project(payload, ""), bases


def parameter_values(payload: object, *, role: str) -> Any:
    """A method declaration's hashed projection: each parameter record replaced by its value.

    It is typed as ``json.loads`` is, because it is what ``json.loads`` returned for the
    declaration before any basis was recorded, and loaders validate it exactly as they did then.
    """

    return split_parameter_records(payload, role=role)[0]


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
