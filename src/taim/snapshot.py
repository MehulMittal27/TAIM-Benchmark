"""Benchmark Snapshot logical contracts and content identities.

The module contains only System-visible logical inputs. Evaluation Judgments,
source bundles, preparation metadata, and physical package layout live outside
:class:`BenchmarkSnapshot`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import PurePath, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, cast

from taim.contracts import (
    SNAPSHOT_CANONICALIZATION_VERSION,
    SNAPSHOT_CONTRACT_VERSION,
    EvaluatorOnlyMaterial,
    canonical_json,
    content_sha256,
    iter_canonical_json,
    require_non_empty,
    require_sha256,
    validate_json,
)
from taim.schemas import (
    JsonValue,
    RelevanceJudgment,
    SchemaValidationError,
    freeze_json_value,
    json_value_to_builtins,
)

CAPABILITY_CANONICAL_PATIENT_TEXT = "canonical_patient_text"
CAPABILITY_CANONICAL_TRIAL_TEXT = "canonical_trial_text"
CAPABILITY_PATIENT_EVIDENCE = "patient_evidence_items"
CAPABILITY_SEMANTIC_TRIAL_SECTIONS = "semantic_trial_sections"
CAPABILITY_TYPED_PATIENT_CORE = "typed_patient_core"
CAPABILITY_TYPED_TRIAL_CORE = "typed_trial_core"
CAPABILITY_FIELD_PROVENANCE = "field_provenance"
CAPABILITY_COMPLETE_ELIGIBILITY_TEXT = "complete_eligibility_text"
DERIVED_VIEW_CAPABILITY_PREFIX = "derived_view:"
ProvenanceLocatorKind = Literal[
    "archive_member",
    "fhir_element_path",
    "json_pointer",
    "source_fragment",
    "xml_path",
]
PROVENANCE_LOCATOR_KINDS: frozenset[ProvenanceLocatorKind] = frozenset(
    {"archive_member", "fhir_element_path", "json_pointer", "source_fragment", "xml_path"}
)

STANDARD_CAPABILITIES = frozenset(
    {
        CAPABILITY_CANONICAL_PATIENT_TEXT,
        CAPABILITY_CANONICAL_TRIAL_TEXT,
        CAPABILITY_PATIENT_EVIDENCE,
        CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
        CAPABILITY_TYPED_PATIENT_CORE,
        CAPABILITY_TYPED_TRIAL_CORE,
        CAPABILITY_FIELD_PROVENANCE,
        CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
    }
)

AGE_UNITS = frozenset({"minutes", "hours", "days", "weeks", "months", "years"})
SEX_VALUES = frozenset({"female", "male", "all"})
SECTION_ROLES = frozenset(
    {"brief_title", "official_title", "summary", "condition", "intervention", "eligibility"}
)

_CAPABILITY = re.compile(r"\A[a-z][a-z0-9_]*(?::[a-z0-9]+(?:[_-][a-z0-9]+)*)?\Z")
_SLUG = re.compile(r"\A[a-z0-9]+(?:[_-][a-z0-9]+)*\Z")
_NAMESPACE = re.compile(r"\A[a-z0-9]+(?:[._-][a-z0-9]+)*\Z")
_EVALUATOR_ONLY_KEYS = frozenset(
    {
        "evaluation_label",
        "evaluation_labels",
        "evaluation_package",
        "evaluation_package_path",
        "evaluation_packages",
        "evaluation_provenance",
        "evaluator_path",
        "evaluator_provenance",
        "judgement",
        "judgements",
        "judgment",
        "judgment_path",
        "judgments",
        "judgments_hash",
        "judgments_path",
        "qrel",
        "qrels",
        "qrels_hash",
        "qrels_path",
        "relevance_judgment",
        "relevance_judgments",
    }
)
_SERIALIZED_JUDGMENT_FIELDS = frozenset({"topic_id", "trial_id", "label"})
_SERIALIZED_EVALUATION_PACKAGE_FIELDS = frozenset(
    {
        "schema_version",
        "benchmark_lineage",
        "task_id",
        "snapshot_id",
        "evaluation_package_id",
        "judgment_count",
        "provenance",
    }
)
_LOCAL_PATH_PREFIXES = (
    "/applications/",
    "/library/",
    "/network/",
    "/system/",
    "/users/",
    "/volumes/",
    "/bin/",
    "/dev/",
    "/etc/",
    "/home/",
    "/media/",
    "/mnt/",
    "/opt/",
    "/private/",
    "/proc/",
    "/root/",
    "/run/",
    "/sbin/",
    "/snap/",
    "/srv/",
    "/sys/",
    # Denylist entry, not a temporary path this process ever opens or writes:
    # _validate_provenance_location rejects any location with this prefix.
    "/tmp/",  # noqa: S108
    "/usr/",
    "/var/",
)


def _require_ordinal(value: object, name: str = "ordinal") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaValidationError(f"{name} must be a non-negative integer")
    return value


def _validate_system_input_json(value: object, path: str) -> None:
    """Reject evaluator-reserved keys anywhere in opaque System-visible JSON."""

    validate_json(value, path)

    def reject_reserved(item: object, item_path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalized_key = key.casefold().replace("-", "_").rsplit(".", 1)[-1]
                if normalized_key in _EVALUATOR_ONLY_KEYS:
                    raise SchemaValidationError(
                        f"{item_path} contains evaluator-only field {key!r}"
                    )
                reject_reserved(nested, f"{item_path}.{key}")
        elif isinstance(item, list | tuple):
            for index, nested in enumerate(item):
                reject_reserved(nested, f"{item_path}[{index}]")

    reject_reserved(value, path)


def _copy_system_input_json(
    value: Mapping[str, JsonValue], *, path: str
) -> Mapping[str, JsonValue]:
    _validate_system_input_json(value, path)
    detached = json_value_to_builtins(value)
    return cast(
        Mapping[str, JsonValue],
        freeze_json_value(detached),
    )


def _copy_system_input_value(value: JsonValue, *, path: str) -> JsonValue:
    _validate_system_input_json(value, path)
    detached = json_value_to_builtins(value)
    return freeze_json_value(detached)


def freeze_system_input_options(
    value: Mapping[str, object], *, path: str = "SystemRunRequest options"
) -> Mapping[str, object]:
    """Freeze only explicit, value-safe System option types.

    Options are adapter configuration, not a general object transport.  The
    narrow value grammar below preserves command-line paths and nested scalar
    configuration while making evaluator carriers impossible to smuggle
    through an otherwise harmless-looking mapping.
    """

    def is_serialized_evaluator_material(item: Mapping[object, object]) -> bool:
        keys = frozenset(item)
        return keys >= _SERIALIZED_JUDGMENT_FIELDS or keys >= _SERIALIZED_EVALUATION_PACKAGE_FIELDS

    def freeze(item: object, item_path: str) -> object:
        if isinstance(item, RelevanceJudgment | EvaluatorOnlyMaterial):
            raise SchemaValidationError(f"{item_path} contains evaluator-only material")
        if item is None or isinstance(item, (bool, int, str, PurePath)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise SchemaValidationError(f"{item_path} must contain finite scalar values")
            return item
        if isinstance(item, Mapping):
            if is_serialized_evaluator_material(item):
                raise SchemaValidationError(
                    f"{item_path} contains evaluator-only serialized material"
                )
            detached: dict[str, object] = {}
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise SchemaValidationError(f"{item_path} keys must be strings")
                normalized_key = key.casefold().replace("-", "_").rsplit(".", 1)[-1]
                if normalized_key in _EVALUATOR_ONLY_KEYS:
                    raise SchemaValidationError(
                        f"{item_path} contains evaluator-only field {key!r}"
                    )
                detached[key] = freeze(nested, f"{item_path}.{key}")
            return MappingProxyType(detached)
        if isinstance(item, list | tuple):
            return tuple(
                freeze(nested, f"{item_path}[{index}]") for index, nested in enumerate(item)
            )
        if isinstance(item, set | frozenset):
            return frozenset(freeze(nested, item_path) for nested in item)
        raise SchemaValidationError(
            f"{item_path} contains unsupported safe option value {type(item).__name__}"
        )

    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{path} must be an object")
    return cast(Mapping[str, object], freeze(value, path))


def _validate_extensions(value: Mapping[str, JsonValue], *, path: str) -> Mapping[str, JsonValue]:
    for key in value:
        if _NAMESPACE.fullmatch(key) is None or "." not in key:
            raise SchemaValidationError(
                f"{path} keys must use a lowercase '<namespace>.<name>' key: {key!r}"
            )
    return _copy_system_input_json(value, path=path)


def _additional_fields(
    payload: Mapping[str, object], known_fields: Iterable[str], *, path: str
) -> Mapping[str, JsonValue]:
    unknown = {
        key: cast(JsonValue, value)
        for key, value in payload.items()
        if key not in frozenset(known_fields)
    }
    return _copy_system_input_json(unknown, path=path)


def _copy_additional_fields(
    value: Mapping[str, JsonValue], known_fields: Iterable[str], *, path: str
) -> Mapping[str, JsonValue]:
    conflicts = set(value) & frozenset(known_fields)
    if conflicts:
        raise SchemaValidationError(
            f"{path} cannot replace known fields: {', '.join(sorted(conflicts))}"
        )
    return _copy_system_input_json(value, path=path)


def _include_additional_fields(
    payload: dict[str, JsonValue], additional_fields: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    payload.update(cast(Mapping[str, JsonValue], json_value_to_builtins(additional_fields)))
    return payload


def _canonical_decimal(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int | Decimal):
        raise SchemaValidationError("age amount must be an exact decimal string or integer")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise SchemaValidationError(f"invalid exact age amount {value!r}") from exc
    if not amount.is_finite() or amount < 0:
        raise SchemaValidationError("age amount must be finite and non-negative")
    if amount == 0:
        return "0"
    normalized = format(amount, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _contains_local_path(location: str, *, rooted_posix_is_local: bool = False) -> bool:
    folded = location.casefold()
    return bool(
        "://" in location
        or "\\" in location
        or location.startswith("//")
        or (rooted_posix_is_local and location.startswith("/"))
        or re.match(r"\A[A-Za-z]:[\\/]", location)
        or folded.startswith(_LOCAL_PATH_PREFIXES)
    )


def _validate_provenance_location(locator_kind: object, location: str) -> None:
    if not isinstance(locator_kind, str) or locator_kind not in PROVENANCE_LOCATOR_KINDS:
        raise SchemaValidationError(f"unsupported provenance locator_kind {locator_kind!r}")
    if _contains_local_path(location):
        raise SchemaValidationError("provenance location must not contain a local path")
    if locator_kind == "archive_member":
        prefix = "zip-member:"
        member_name = location.removeprefix(prefix)
        member = PurePosixPath(member_name)
        if (
            not location.startswith(prefix)
            or not member_name
            or member.is_absolute()
            or any(part in {"", ".", ".."} for part in member.parts)
        ):
            raise SchemaValidationError("archive provenance location is invalid")
    elif locator_kind == "json_pointer":
        if not location.startswith("/") or re.search(r"~(?![01])", location):
            raise SchemaValidationError("JSON provenance location must be a JSON Pointer")
    elif locator_kind == "xml_path":
        if (
            not location.startswith("/")
            or location.endswith("/")
            or "//" in location
            or any(part in {"", ".", ".."} for part in location[1:].split("/"))
        ):
            raise SchemaValidationError("XML provenance location must be a rooted element path")
    elif locator_kind == "fhir_element_path":
        if re.fullmatch(r"fhir-resource:[^#\s]+#[A-Za-z][A-Za-z0-9_.\[\]-]*", location) is None:
            raise SchemaValidationError(
                "FHIR provenance location must include resource identity and element path"
            )
    else:
        fragment = re.fullmatch(r"fragment:[a-z][a-z0-9_-]*:(?P<locator>[^\s].*)", location)
        if fragment is None:
            raise SchemaValidationError(
                "source-fragment provenance must be 'fragment:<scheme>:<locator>'"
            )
        if _contains_local_path(fragment.group("locator"), rooted_posix_is_local=True):
            raise SchemaValidationError("provenance location must not contain a local path")


@dataclass(frozen=True, slots=True)
class SourceRecordIdentity:
    """Native source identity kept separate from a benchmark-facing record ID."""

    namespace: str
    record_id: str
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        namespace = require_non_empty(self.namespace, "source identity namespace")
        if _NAMESPACE.fullmatch(namespace) is None:
            raise SchemaValidationError("source identity namespace must be a lowercase name")
        require_non_empty(self.record_id, "source identity record_id")
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {"namespace", "record_id"},
                path="source identity additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return _include_additional_fields(
            {"namespace": self.namespace, "record_id": self.record_id},
            self.additional_fields,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SourceRecordIdentity:
        return cls(
            namespace=cast(str, payload.get("namespace")),
            record_id=cast(str, payload.get("record_id")),
            additional_fields=_additional_fields(
                payload,
                {"namespace", "record_id"},
                path="source identity additional fields",
            ),
        )


@dataclass(frozen=True, slots=True)
class FieldProvenance:
    """Exact source link and deterministic transformation for one assertion."""

    artifact_id: str
    artifact_sha256: str
    source_record_id: str
    locator_kind: ProvenanceLocatorKind
    location: str
    raw_value: str
    transformation_rule: str
    raw_value_sha256: str | None = None
    raw_value_byte_length: int | None = None
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.artifact_id, "provenance artifact_id")
        require_sha256(self.artifact_sha256, "provenance artifact_sha256")
        require_non_empty(self.source_record_id, "provenance source_record_id")
        location = require_non_empty(self.location, "provenance location")
        _validate_provenance_location(self.locator_kind, location)
        if not isinstance(self.raw_value, str):
            raise SchemaValidationError("provenance raw_value must be a string")
        excerpt_size = len(self.raw_value.encode("utf-8"))
        if excerpt_size > 4_096:
            raise SchemaValidationError("provenance raw_value exceeds the bounded excerpt limit")
        require_non_empty(self.transformation_rule, "provenance transformation_rule")
        has_digest = self.raw_value_sha256 is not None
        has_length = self.raw_value_byte_length is not None
        if has_digest != has_length:
            raise SchemaValidationError(
                "provenance bounded excerpt metadata requires both digest and byte length"
            )
        if has_digest:
            require_sha256(self.raw_value_sha256, "provenance raw_value_sha256")
            if (
                isinstance(self.raw_value_byte_length, bool)
                or not isinstance(self.raw_value_byte_length, int)
                or self.raw_value_byte_length <= excerpt_size
            ):
                raise SchemaValidationError(
                    "provenance bounded excerpt byte length must exceed the excerpt size"
                )
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {
                    "artifact_id",
                    "artifact_sha256",
                    "source_record_id",
                    "locator_kind",
                    "location",
                    "raw_value",
                    "transformation_rule",
                    "raw_value_sha256",
                    "raw_value_byte_length",
                },
                path="provenance additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "source_record_id": self.source_record_id,
            "locator_kind": self.locator_kind,
            "location": self.location,
            "raw_value": self.raw_value,
            "transformation_rule": self.transformation_rule,
        }
        if self.raw_value_sha256 is not None:
            payload["raw_value_sha256"] = self.raw_value_sha256
            payload["raw_value_byte_length"] = cast(int, self.raw_value_byte_length)
        return _include_additional_fields(payload, self.additional_fields)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FieldProvenance:
        return cls(
            artifact_id=cast(str, payload.get("artifact_id")),
            artifact_sha256=cast(str, payload.get("artifact_sha256")),
            source_record_id=cast(str, payload.get("source_record_id")),
            locator_kind=cast(ProvenanceLocatorKind, payload.get("locator_kind")),
            location=cast(str, payload.get("location")),
            raw_value=cast(str, payload.get("raw_value")),
            transformation_rule=cast(str, payload.get("transformation_rule")),
            raw_value_sha256=cast(str | None, payload.get("raw_value_sha256")),
            raw_value_byte_length=cast(int | None, payload.get("raw_value_byte_length")),
            additional_fields=_additional_fields(
                payload,
                {
                    "artifact_id",
                    "artifact_sha256",
                    "source_record_id",
                    "locator_kind",
                    "location",
                    "raw_value",
                    "transformation_rule",
                    "raw_value_sha256",
                    "raw_value_byte_length",
                },
                path="provenance additional fields",
            ),
        )


def field_provenance_from_payload(value: object, *, field_name: str) -> tuple[FieldProvenance, ...]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{field_name} provenance must be an array")
    if any(not isinstance(item, Mapping) for item in value):
        raise SchemaValidationError(f"{field_name} provenance entries must be objects")
    return tuple(FieldProvenance.from_dict(cast(Mapping[str, object], item)) for item in value)


@dataclass(frozen=True, slots=True)
class PatientEvidenceItem:
    kind: str
    ordinal: int
    text: str
    provenance: tuple[FieldProvenance, ...]
    timestamp: str | None = None
    interval_start: str | None = None
    interval_end: str | None = None
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.kind, "patient evidence kind")
        _require_ordinal(self.ordinal)
        require_non_empty(self.text, "patient evidence text")
        provenance = tuple(self.provenance)
        if not provenance or any(not isinstance(item, FieldProvenance) for item in provenance):
            raise SchemaValidationError("patient evidence must contain FieldProvenance objects")
        object.__setattr__(self, "provenance", provenance)
        for name in ("timestamp", "interval_start", "interval_end"):
            value = getattr(self, name)
            if value is not None:
                require_non_empty(value, f"patient evidence {name}")
        if (self.interval_start is None) != (self.interval_end is None):
            raise SchemaValidationError("patient evidence intervals require both endpoints")
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {
                    "kind",
                    "ordinal",
                    "text",
                    "provenance",
                    "timestamp",
                    "interval_start",
                    "interval_end",
                },
                path="patient evidence additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "kind": self.kind,
            "ordinal": self.ordinal,
            "text": self.text,
            "provenance": [item.to_dict() for item in self.provenance],
        }
        for name in ("timestamp", "interval_start", "interval_end"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return _include_additional_fields(payload, self.additional_fields)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PatientEvidenceItem:
        return cls(
            kind=cast(str, payload.get("kind")),
            ordinal=cast(int, payload.get("ordinal")),
            text=cast(str, payload.get("text")),
            provenance=field_provenance_from_payload(
                payload.get("provenance"), field_name="evidence"
            ),
            timestamp=cast(str | None, payload.get("timestamp")),
            interval_start=cast(str | None, payload.get("interval_start")),
            interval_end=cast(str | None, payload.get("interval_end")),
            additional_fields=_additional_fields(
                payload,
                {
                    "kind",
                    "ordinal",
                    "text",
                    "provenance",
                    "timestamp",
                    "interval_start",
                    "interval_end",
                },
                path="patient evidence additional fields",
            ),
        )


@dataclass(frozen=True, slots=True)
class CriterionItem:
    """A source or deterministically derived trial criterion, never a patient assessment."""

    ordinal: int
    text: str
    polarity: Literal["inclusion", "exclusion", "unspecified"]
    provenance: tuple[FieldProvenance, ...]
    identifier: str | None = None
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)
    origin: Literal["source_authored", "derived"] = "source_authored"

    def __post_init__(self) -> None:
        _require_ordinal(self.ordinal, "criterion ordinal")
        require_non_empty(self.text, "criterion text")
        if self.polarity not in {"inclusion", "exclusion", "unspecified"}:
            raise SchemaValidationError("criterion polarity is unsupported")
        provenance = tuple(self.provenance)
        if not provenance or any(not isinstance(item, FieldProvenance) for item in provenance):
            raise SchemaValidationError("criterion item must contain FieldProvenance objects")
        object.__setattr__(self, "provenance", provenance)
        if self.identifier is not None:
            require_non_empty(self.identifier, "criterion identifier")
        if self.origin not in {"source_authored", "derived"}:
            raise SchemaValidationError("criterion origin is unsupported")
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {"ordinal", "text", "polarity", "provenance", "identifier", "origin"},
                path="criterion additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "ordinal": self.ordinal,
            "text": self.text,
            "polarity": self.polarity,
            "provenance": [item.to_dict() for item in self.provenance],
        }
        if self.identifier is not None:
            payload["identifier"] = self.identifier
        if self.origin != "source_authored":
            payload["origin"] = self.origin
        return _include_additional_fields(payload, self.additional_fields)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CriterionItem:
        return cls(
            ordinal=cast(int, payload.get("ordinal")),
            text=cast(str, payload.get("text")),
            polarity=cast(Any, payload.get("polarity")),
            provenance=field_provenance_from_payload(
                payload.get("provenance"), field_name="criterion"
            ),
            identifier=cast(str | None, payload.get("identifier")),
            additional_fields=_additional_fields(
                payload,
                {"ordinal", "text", "polarity", "provenance", "identifier", "origin"},
                path="criterion additional fields",
            ),
            origin=cast(Any, payload.get("origin", "source_authored")),
        )


@dataclass(frozen=True, slots=True)
class SemanticTextSection:
    role: str
    text: str
    ordinal: int
    provenance: tuple[FieldProvenance, ...]
    criteria: tuple[CriterionItem, ...] = ()
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)
    source_text: str | None = None
    source_text_sha256: str | None = None
    source_text_byte_length: int | None = None

    def __post_init__(self) -> None:
        if self.role not in SECTION_ROLES:
            raise SchemaValidationError(f"unsupported shared semantic section role {self.role!r}")
        require_non_empty(self.text, "semantic section text")
        _require_ordinal(self.ordinal, "semantic section ordinal")
        provenance = tuple(self.provenance)
        if not provenance or any(not isinstance(item, FieldProvenance) for item in provenance):
            raise SchemaValidationError("semantic section must contain FieldProvenance objects")
        criteria = tuple(self.criteria)
        if any(not isinstance(item, CriterionItem) for item in criteria):
            raise SchemaValidationError("semantic section criteria must be CriterionItem objects")
        if criteria and self.role != "eligibility":
            raise SchemaValidationError("criterion structure is valid only on eligibility sections")
        if [item.ordinal for item in criteria] != list(range(len(criteria))):
            raise SchemaValidationError("criterion items must use contiguous source order")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "criteria", criteria)
        has_source_text = self.source_text is not None
        if has_source_text != (self.source_text_sha256 is not None) or has_source_text != (
            self.source_text_byte_length is not None
        ):
            raise SchemaValidationError(
                "complete source text requires text, SHA-256, and byte length together"
            )
        if has_source_text:
            if self.role != "eligibility":
                raise SchemaValidationError(
                    "complete source text is valid only on eligibility sections"
                )
            source_text = cast(str, self.source_text)
            require_non_empty(source_text, "complete eligibility source text")
            require_sha256(self.source_text_sha256, "complete eligibility source text SHA-256")
            source_bytes = source_text.encode("utf-8")
            if self.source_text_byte_length != len(source_bytes):
                raise SchemaValidationError(
                    "complete eligibility source text byte length does not match"
                )
            expected = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
            if self.source_text_sha256 != expected:
                raise SchemaValidationError(
                    "complete eligibility source text SHA-256 does not match"
                )
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {
                    "role",
                    "text",
                    "ordinal",
                    "provenance",
                    "criteria",
                    "source_text",
                    "source_text_sha256",
                    "source_text_byte_length",
                },
                path="semantic section additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "role": self.role,
            "text": self.text,
            "ordinal": self.ordinal,
            "provenance": [item.to_dict() for item in self.provenance],
        }
        if self.criteria:
            payload["criteria"] = [item.to_dict() for item in self.criteria]
        if self.source_text is not None:
            payload["source_text"] = self.source_text
            payload["source_text_sha256"] = cast(str, self.source_text_sha256)
            payload["source_text_byte_length"] = cast(int, self.source_text_byte_length)
        return _include_additional_fields(payload, self.additional_fields)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SemanticTextSection:
        raw_criteria = payload.get("criteria", [])
        if not isinstance(raw_criteria, list):
            raise SchemaValidationError("semantic section criteria must be an array")
        if any(not isinstance(item, Mapping) for item in raw_criteria):
            raise SchemaValidationError("semantic section criteria must contain objects")
        return cls(
            role=cast(str, payload.get("role")),
            text=cast(str, payload.get("text")),
            ordinal=cast(int, payload.get("ordinal")),
            provenance=field_provenance_from_payload(
                payload.get("provenance"), field_name="section"
            ),
            criteria=tuple(
                CriterionItem.from_dict(cast(Mapping[str, object], item)) for item in raw_criteria
            ),
            additional_fields=_additional_fields(
                payload,
                {
                    "role",
                    "text",
                    "ordinal",
                    "provenance",
                    "criteria",
                    "source_text",
                    "source_text_sha256",
                    "source_text_byte_length",
                },
                path="semantic section additional fields",
            ),
            source_text=cast(str | None, payload.get("source_text")),
            source_text_sha256=cast(str | None, payload.get("source_text_sha256")),
            source_text_byte_length=cast(int | None, payload.get("source_text_byte_length")),
        )


@dataclass(frozen=True, slots=True)
class AgeQuantity:
    amount: str
    unit: str
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", _canonical_decimal(self.amount))
        if self.unit not in AGE_UNITS:
            raise SchemaValidationError(f"unsupported age unit {self.unit!r}")
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {"amount", "unit"},
                path="age quantity additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return _include_additional_fields(
            {"amount": self.amount, "unit": self.unit}, self.additional_fields
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AgeQuantity:
        amount = payload.get("amount")
        unit = payload.get("unit")
        if not isinstance(amount, str) or not isinstance(unit, str):
            raise SchemaValidationError("serialized age quantity must use string amount and unit")
        return cls(
            amount,
            unit,
            _additional_fields(payload, {"amount", "unit"}, path="age quantity additional fields"),
        )


@dataclass(frozen=True, slots=True)
class AgeBound:
    value: AgeQuantity | Literal["unbounded"]
    provenance: tuple[FieldProvenance, ...]
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value != "unbounded" and not isinstance(self.value, AgeQuantity):
            raise SchemaValidationError("age bound must be an AgeQuantity or 'unbounded'")
        provenance = tuple(self.provenance)
        if not provenance or any(not isinstance(item, FieldProvenance) for item in provenance):
            raise SchemaValidationError("age bound must contain FieldProvenance objects")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {"value", "provenance"},
                path="age bound additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        value: JsonValue = "unbounded" if self.value == "unbounded" else self.value.to_dict()
        return _include_additional_fields(
            {"value": value, "provenance": [item.to_dict() for item in self.provenance]},
            self.additional_fields,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AgeBound:
        raw_value = payload.get("value")
        value: AgeQuantity | Literal["unbounded"]
        if raw_value == "unbounded":
            value = "unbounded"
        elif isinstance(raw_value, Mapping):
            value = AgeQuantity.from_dict(raw_value)
        else:
            raise SchemaValidationError("serialized age bound value is invalid")
        return cls(
            value=value,
            provenance=field_provenance_from_payload(payload.get("provenance"), field_name="age"),
            additional_fields=_additional_fields(
                payload, {"value", "provenance"}, path="age bound additional fields"
            ),
        )


@dataclass(frozen=True, slots=True)
class SexEligibility:
    value: str
    provenance: tuple[FieldProvenance, ...]
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value not in SEX_VALUES:
            raise SchemaValidationError(f"unsupported sex eligibility value {self.value!r}")
        provenance = tuple(self.provenance)
        if not provenance or any(not isinstance(item, FieldProvenance) for item in provenance):
            raise SchemaValidationError("sex eligibility must contain FieldProvenance objects")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {"value", "provenance"},
                path="sex additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return _include_additional_fields(
            {"value": self.value, "provenance": [item.to_dict() for item in self.provenance]},
            self.additional_fields,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SexEligibility:
        return cls(
            cast(str, payload.get("value")),
            field_provenance_from_payload(payload.get("provenance"), field_name="sex"),
            _additional_fields(payload, {"value", "provenance"}, path="sex additional fields"),
        )


@dataclass(frozen=True, slots=True)
class HealthyVolunteerAcceptance:
    value: bool
    provenance: tuple[FieldProvenance, ...]
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.value, bool):
            raise SchemaValidationError("healthy-volunteer value must be boolean")
        provenance = tuple(self.provenance)
        if not provenance or any(not isinstance(item, FieldProvenance) for item in provenance):
            raise SchemaValidationError(
                "healthy-volunteer value must contain FieldProvenance objects"
            )
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {"value", "provenance"},
                path="healthy-volunteer additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return _include_additional_fields(
            {"value": self.value, "provenance": [item.to_dict() for item in self.provenance]},
            self.additional_fields,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HealthyVolunteerAcceptance:
        return cls(
            cast(bool, payload.get("value")),
            field_provenance_from_payload(
                payload.get("provenance"), field_name="healthy_volunteers"
            ),
            _additional_fields(
                payload,
                {"value", "provenance"},
                path="healthy-volunteer additional fields",
            ),
        )


@dataclass(frozen=True, slots=True)
class CoreFieldConflict:
    """Conflicting normalized assertions for one Typed Clinical Core field."""

    field_name: str
    values: tuple[JsonValue, ...]
    provenance: tuple[FieldProvenance, ...]

    def __post_init__(self) -> None:
        if self.field_name not in {
            "minimum_age",
            "maximum_age",
            "sex",
            "healthy_volunteers",
        }:
            raise SchemaValidationError("core field conflict field_name is invalid")
        values = tuple(
            _copy_system_input_value(value, path="core field conflict value")
            for value in self.values
        )
        provenance = tuple(self.provenance)
        if len(values) < 2 or len(values) != len(provenance):
            raise SchemaValidationError(
                "core field conflict must pair at least two values with provenance"
            )
        if any(not isinstance(item, FieldProvenance) for item in provenance):
            raise SchemaValidationError("core field conflict must contain FieldProvenance objects")
        if len({canonical_json(value) for value in values}) < 2:
            raise SchemaValidationError("core field conflict values must disagree")
        for value in values:
            if self.field_name in {"minimum_age", "maximum_age"}:
                if value == "unbounded" and self.field_name == "maximum_age":
                    continue
                if not isinstance(value, Mapping):
                    raise SchemaValidationError("core field conflict age value is invalid")
                AgeQuantity.from_dict(value)
            elif self.field_name == "sex" and value not in SEX_VALUES:
                raise SchemaValidationError("core field conflict sex value is invalid")
            elif self.field_name == "healthy_volunteers" and not isinstance(value, bool):
                raise SchemaValidationError(
                    "core field conflict healthy-volunteer value is invalid"
                )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "provenance", provenance)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "field_name": self.field_name,
            "values": [json_value_to_builtins(value) for value in self.values],
            "provenance": [item.to_dict() for item in self.provenance],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CoreFieldConflict:
        if set(payload) != {"field_name", "values", "provenance"}:
            raise SchemaValidationError("serialized core field conflict shape is invalid")
        values = payload.get("values")
        if not isinstance(values, list):
            raise SchemaValidationError("serialized core field conflict values are invalid")
        return cls(
            field_name=cast(str, payload.get("field_name")),
            values=tuple(cast(JsonValue, value) for value in values),
            provenance=field_provenance_from_payload(
                payload.get("provenance"), field_name="core field conflict"
            ),
        )


@dataclass(frozen=True, slots=True)
class TypedClinicalCore:
    minimum_age: AgeBound | None = None
    maximum_age: AgeBound | None = None
    sex: SexEligibility | None = None
    healthy_volunteers: HealthyVolunteerAcceptance | None = None
    conflicts: tuple[CoreFieldConflict, ...] = ()
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_types = {
            "minimum_age": AgeBound,
            "maximum_age": AgeBound,
            "sex": SexEligibility,
            "healthy_volunteers": HealthyVolunteerAcceptance,
        }
        for name, expected_type in expected_types.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected_type):
                raise SchemaValidationError(
                    f"typed clinical core {name} must be {expected_type.__name__}"
                )
        conflicts = tuple(self.conflicts)
        if any(not isinstance(item, CoreFieldConflict) for item in conflicts):
            raise SchemaValidationError(
                "typed clinical core conflicts must contain CoreFieldConflict objects"
            )
        conflict_fields = [item.field_name for item in conflicts]
        if len(conflict_fields) != len(set(conflict_fields)):
            raise SchemaValidationError("typed clinical core conflict fields must be unique")
        if any(getattr(self, field_name) is not None for field_name in conflict_fields):
            raise SchemaValidationError(
                "typed clinical core field cannot be both known and conflicting"
            )
        object.__setattr__(
            self, "conflicts", tuple(sorted(conflicts, key=lambda item: item.field_name))
        )
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {"minimum_age", "maximum_age", "sex", "healthy_volunteers", "conflicts"},
                path="typed clinical core additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {}
        for name in ("minimum_age", "maximum_age", "sex", "healthy_volunteers"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value.to_dict()
        if self.conflicts:
            payload["conflicts"] = [item.to_dict() for item in self.conflicts]
        return _include_additional_fields(payload, self.additional_fields)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TypedClinicalCore:
        def nested(name: str, schema: type[Any]) -> Any | None:
            value = payload.get(name)
            if value is None:
                return None
            if not isinstance(value, Mapping):
                raise SchemaValidationError(f"typed clinical core {name} must be an object")
            return schema.from_dict(value)

        def conflicts() -> tuple[CoreFieldConflict, ...]:
            raw_conflicts = payload.get("conflicts", [])
            if not isinstance(raw_conflicts, list):
                raise SchemaValidationError("typed clinical core conflicts must be an array")
            parsed: list[CoreFieldConflict] = []
            for item in raw_conflicts:
                if not isinstance(item, Mapping):
                    raise SchemaValidationError("typed clinical core conflict must be an object")
                parsed.append(CoreFieldConflict.from_dict(item))
            return tuple(parsed)

        return cls(
            minimum_age=nested("minimum_age", AgeBound),
            maximum_age=nested("maximum_age", AgeBound),
            sex=nested("sex", SexEligibility),
            healthy_volunteers=nested("healthy_volunteers", HealthyVolunteerAcceptance),
            conflicts=conflicts(),
            additional_fields=_additional_fields(
                payload,
                {"minimum_age", "maximum_age", "sex", "healthy_volunteers", "conflicts"},
                path="typed clinical core additional fields",
            ),
        )


@dataclass(frozen=True, slots=True)
class PatientCoreAssertion:
    """One profile-defined patient fact with mandatory source provenance."""

    value: JsonValue
    provenance: tuple[FieldProvenance, ...]
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value is None:
            raise SchemaValidationError("patient core assertion value must not be null")
        object.__setattr__(
            self,
            "value",
            _copy_system_input_value(self.value, path="patient core assertion value"),
        )
        provenance = tuple(self.provenance)
        if not provenance or any(not isinstance(item, FieldProvenance) for item in provenance):
            raise SchemaValidationError("PatientCoreAssertion must contain FieldProvenance objects")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {"value", "provenance"},
                path="patient core assertion additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return _include_additional_fields(
            {
                "value": json_value_to_builtins(self.value),
                "provenance": [item.to_dict() for item in self.provenance],
            },
            self.additional_fields,
        )

    def iter_provenance(self) -> Iterable[FieldProvenance]:
        return iter(self.provenance)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PatientCoreAssertion:
        return cls(
            value=cast(JsonValue, payload.get("value")),
            provenance=field_provenance_from_payload(
                payload.get("provenance"), field_name="patient core assertion"
            ),
            additional_fields=_additional_fields(
                payload,
                {"value", "provenance"},
                path="patient core assertion additional fields",
            ),
        )


@dataclass(frozen=True, slots=True)
class TypedPatientCore:
    """Profiled patient fields without claiming one universal EHR fact schema."""

    profile_id: str
    fields: Mapping[str, PatientCoreAssertion]
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.profile_id, "typed patient core profile_id")
        if not isinstance(self.fields, Mapping) or not self.fields:
            raise SchemaValidationError("typed patient core fields must be a non-empty object")
        for name, assertion in self.fields.items():
            require_non_empty(name, "typed patient core field name")
            if _NAMESPACE.fullmatch(name) is None:
                raise SchemaValidationError(
                    "typed patient core field names must be lowercase profile concepts"
                )
            if not isinstance(assertion, PatientCoreAssertion):
                raise SchemaValidationError(
                    "typed patient core fields must contain PatientCoreAssertion objects"
                )
        ordered_fields = dict(sorted(self.fields.items()))
        _validate_system_input_json(
            {name: assertion.to_dict() for name, assertion in ordered_fields.items()},
            "typed patient core fields",
        )
        object.__setattr__(self, "fields", MappingProxyType(ordered_fields))
        object.__setattr__(
            self,
            "additional_fields",
            _copy_additional_fields(
                self.additional_fields,
                {"profile_id", "fields"},
                path="typed patient core additional fields",
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return _include_additional_fields(
            {
                "profile_id": self.profile_id,
                "fields": {name: assertion.to_dict() for name, assertion in self.fields.items()},
            },
            self.additional_fields,
        )

    def iter_provenance(self) -> Iterable[FieldProvenance]:
        for assertion in self.fields.values():
            yield from assertion.iter_provenance()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TypedPatientCore:
        fields = payload.get("fields")
        if not isinstance(fields, Mapping):
            raise SchemaValidationError("typed patient core fields must be an object")
        if any(
            not isinstance(name, str) or not isinstance(value, Mapping)
            for name, value in fields.items()
        ):
            raise SchemaValidationError(
                "typed patient core fields must contain PatientCoreAssertion objects"
            )
        return cls(
            cast(str, payload.get("profile_id")),
            {
                cast(str, name): PatientCoreAssertion.from_dict(cast(Mapping[str, object], value))
                for name, value in fields.items()
            },
            _additional_fields(
                payload,
                {"profile_id", "fields"},
                path="typed patient core additional fields",
            ),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkTopic:
    topic_id: str
    source_identity: SourceRecordIdentity
    canonical_text: str
    evidence_items: tuple[PatientEvidenceItem, ...] = ()
    typed_patient_core: TypedPatientCore | None = None
    extensions: Mapping[str, JsonValue] = field(default_factory=dict)
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    schema_version = SNAPSHOT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.topic_id, "topic_id")
        if not isinstance(self.source_identity, SourceRecordIdentity):
            raise SchemaValidationError("topic source_identity is required")
        require_non_empty(self.canonical_text, "canonical patient text")
        evidence = tuple(self.evidence_items)
        if any(not isinstance(item, PatientEvidenceItem) for item in evidence):
            raise SchemaValidationError(
                "patient evidence items must contain PatientEvidenceItem objects"
            )
        if [item.ordinal for item in evidence] != list(range(len(evidence))):
            raise SchemaValidationError("patient evidence items must use contiguous source order")
        object.__setattr__(self, "evidence_items", evidence)
        if not isinstance(self.extensions, Mapping):
            raise SchemaValidationError("topic extensions must be an object")
        if self.typed_patient_core is not None and not isinstance(
            self.typed_patient_core, TypedPatientCore
        ):
            raise SchemaValidationError("typed_patient_core must be TypedPatientCore")
        extensions = _validate_extensions(self.extensions, path="topic extensions")
        object.__setattr__(self, "extensions", extensions)
        core_concepts = {"typed_patient_core"}
        if self.typed_patient_core is not None:
            for name in self.typed_patient_core.fields:
                core_concepts.add(name)
                core_concepts.add(name.rsplit(".", 1)[-1])
        if any(key.rsplit(".", 1)[-1] in core_concepts for key in extensions):
            raise SchemaValidationError(
                "topic extensions cannot duplicate a populated patient core concept"
            )
        reserved = {
            "schema_version",
            "topic_id",
            "source_identity",
            "canonical_text",
            "evidence_items",
            "typed_patient_core",
            "extensions",
        }
        if not isinstance(self.additional_fields, Mapping):
            raise SchemaValidationError("topic additional_fields must be an object")
        if reserved & self.additional_fields.keys():
            raise SchemaValidationError("topic additional_fields cannot replace known fields")
        object.__setattr__(
            self,
            "additional_fields",
            _copy_system_input_json(self.additional_fields, path="topic additional_fields"),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "topic_id": self.topic_id,
            "source_identity": self.source_identity.to_dict(),
            "canonical_text": self.canonical_text,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
        }
        if self.typed_patient_core is not None:
            payload["typed_patient_core"] = self.typed_patient_core.to_dict()
        if self.extensions:
            payload["extensions"] = json_value_to_builtins(self.extensions)
        payload.update(
            cast(
                Mapping[str, JsonValue],
                json_value_to_builtins(self.additional_fields),
            )
        )
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BenchmarkTopic:
        if payload.get("schema_version") != SNAPSHOT_CONTRACT_VERSION:
            raise SchemaValidationError("unsupported BenchmarkTopic schema_version")
        raw_evidence = payload.get("evidence_items", [])
        if not isinstance(raw_evidence, list):
            raise SchemaValidationError("topic evidence_items must be an array")
        if any(not isinstance(item, Mapping) for item in raw_evidence):
            raise SchemaValidationError("topic evidence_items must contain objects")
        source_identity_payload = payload.get("source_identity")
        if not isinstance(source_identity_payload, Mapping):
            raise SchemaValidationError("topic source_identity must be an object")
        core_payload = payload.get("typed_patient_core")
        if core_payload is not None and not isinstance(core_payload, Mapping):
            raise SchemaValidationError("typed_patient_core must be an object")
        core = (
            TypedPatientCore.from_dict(core_payload) if isinstance(core_payload, Mapping) else None
        )
        raw_extensions = payload.get("extensions", {})
        if not isinstance(raw_extensions, Mapping):
            raise SchemaValidationError("topic extensions must be an object")
        known_fields = {
            "schema_version",
            "topic_id",
            "source_identity",
            "canonical_text",
            "evidence_items",
            "typed_patient_core",
            "extensions",
        }
        return cls(
            topic_id=cast(str, payload.get("topic_id")),
            source_identity=SourceRecordIdentity.from_dict(source_identity_payload),
            canonical_text=cast(str, payload.get("canonical_text")),
            evidence_items=tuple(
                PatientEvidenceItem.from_dict(cast(Mapping[str, object], item))
                for item in raw_evidence
            ),
            typed_patient_core=core,
            extensions=cast(Mapping[str, JsonValue], raw_extensions),
            additional_fields=cast(
                Mapping[str, JsonValue],
                {key: value for key, value in payload.items() if key not in known_fields},
            ),
        )

    @classmethod
    def from_json(cls, serialized: str) -> BenchmarkTopic:
        payload = json.loads(serialized)
        if not isinstance(payload, dict):
            raise SchemaValidationError("BenchmarkTopic must be a JSON object")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class TrialDocument:
    trial_id: str
    source_identity: SourceRecordIdentity
    canonical_text: str
    sections: tuple[SemanticTextSection, ...] = ()
    typed_clinical_core: TypedClinicalCore | None = None
    registry_extensions: Mapping[str, JsonValue] = field(default_factory=dict)
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)

    schema_version = SNAPSHOT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.trial_id, "trial_id")
        if not isinstance(self.source_identity, SourceRecordIdentity):
            raise SchemaValidationError("trial source_identity is required")
        require_non_empty(self.canonical_text, "canonical trial text")
        sections = tuple(self.sections)
        if any(not isinstance(item, SemanticTextSection) for item in sections):
            raise SchemaValidationError(
                "semantic sections must contain SemanticTextSection objects"
            )
        if [item.ordinal for item in sections] != list(range(len(sections))):
            raise SchemaValidationError("semantic sections must use contiguous source order")
        object.__setattr__(self, "sections", sections)
        if not isinstance(self.registry_extensions, Mapping):
            raise SchemaValidationError("registry_extensions must be an object")
        if self.typed_clinical_core is not None and not isinstance(
            self.typed_clinical_core, TypedClinicalCore
        ):
            raise SchemaValidationError("typed_clinical_core must be TypedClinicalCore")
        extensions = _validate_extensions(
            self.registry_extensions,
            path="registry extensions",
        )
        core_concepts = {"minimum_age", "maximum_age", "sex", "healthy_volunteers"}
        if any(key.rsplit(".", 1)[-1] in core_concepts for key in extensions):
            raise SchemaValidationError(
                "registry extensions cannot duplicate a shared core concept"
            )
        object.__setattr__(self, "registry_extensions", extensions)
        reserved = {
            "schema_version",
            "trial_id",
            "source_identity",
            "canonical_text",
            "sections",
            "typed_clinical_core",
            "registry_extensions",
        }
        if not isinstance(self.additional_fields, Mapping):
            raise SchemaValidationError("trial additional_fields must be an object")
        if reserved & self.additional_fields.keys():
            raise SchemaValidationError("trial additional_fields cannot replace known fields")
        object.__setattr__(
            self,
            "additional_fields",
            _copy_system_input_json(self.additional_fields, path="trial additional fields"),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "source_identity": self.source_identity.to_dict(),
            "canonical_text": self.canonical_text,
            "sections": [item.to_dict() for item in self.sections],
        }
        if self.typed_clinical_core is not None:
            payload["typed_clinical_core"] = self.typed_clinical_core.to_dict()
        if self.registry_extensions:
            payload["registry_extensions"] = json_value_to_builtins(self.registry_extensions)
        payload.update(
            cast(
                Mapping[str, JsonValue],
                json_value_to_builtins(self.additional_fields),
            )
        )
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrialDocument:
        if payload.get("schema_version") != SNAPSHOT_CONTRACT_VERSION:
            raise SchemaValidationError("unsupported TrialDocument schema_version")
        raw_sections = payload.get("sections", [])
        if not isinstance(raw_sections, list):
            raise SchemaValidationError("trial sections must be an array")
        if any(not isinstance(item, Mapping) for item in raw_sections):
            raise SchemaValidationError("trial sections must contain objects")
        source_identity_payload = payload.get("source_identity")
        if not isinstance(source_identity_payload, Mapping):
            raise SchemaValidationError("trial source_identity must be an object")
        core_payload = payload.get("typed_clinical_core")
        if core_payload is not None and not isinstance(core_payload, Mapping):
            raise SchemaValidationError("typed_clinical_core must be an object")
        core = (
            TypedClinicalCore.from_dict(core_payload) if isinstance(core_payload, Mapping) else None
        )
        raw_extensions = payload.get("registry_extensions", {})
        if not isinstance(raw_extensions, Mapping):
            raise SchemaValidationError("registry_extensions must be an object")
        known_fields = {
            "schema_version",
            "trial_id",
            "source_identity",
            "canonical_text",
            "sections",
            "typed_clinical_core",
            "registry_extensions",
        }
        return cls(
            trial_id=cast(str, payload.get("trial_id")),
            source_identity=SourceRecordIdentity.from_dict(source_identity_payload),
            canonical_text=cast(str, payload.get("canonical_text")),
            sections=tuple(
                SemanticTextSection.from_dict(cast(Mapping[str, object], item))
                for item in raw_sections
            ),
            typed_clinical_core=core,
            registry_extensions=cast(Mapping[str, JsonValue], raw_extensions),
            additional_fields=cast(
                Mapping[str, JsonValue],
                {key: value for key, value in payload.items() if key not in known_fields},
            ),
        )

    @classmethod
    def from_json(cls, serialized: str) -> TrialDocument:
        payload = json.loads(serialized)
        if not isinstance(payload, dict):
            raise SchemaValidationError("TrialDocument must be a JSON object")
        return cls.from_dict(payload)


class FrozenTrialDocuments(Sequence[TrialDocument]):
    """Marker interface for immutable, repeatable trial collections."""


@dataclass(frozen=True, slots=True)
class DerivedView:
    name: str
    version: str
    input_snapshot_id: str
    configuration: Mapping[str, JsonValue]
    content: JsonValue
    additional_fields: Mapping[str, JsonValue] = field(default_factory=dict)
    view_id: str = field(init=False)

    schema_version = SNAPSHOT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        name = require_non_empty(self.name, "Derived View name")
        if _SLUG.fullmatch(name) is None:
            raise SchemaValidationError("Derived View name must be a lowercase slug")
        require_non_empty(self.version, "Derived View version")
        require_sha256(self.input_snapshot_id, "Derived View input_snapshot_id")
        configuration = _copy_system_input_json(
            self.configuration, path="Derived View configuration"
        )
        detached_content = _copy_system_input_value(self.content, path="Derived View content")
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "content", detached_content)
        additional_fields = _copy_additional_fields(
            self.additional_fields,
            {
                "schema_version",
                "name",
                "version",
                "input_snapshot_id",
                "configuration",
                "content",
                "view_id",
            },
            path="Derived View additional fields",
        )
        object.__setattr__(self, "additional_fields", additional_fields)
        object.__setattr__(
            self,
            "view_id",
            content_sha256(
                _include_additional_fields(
                    {
                        "schema_version": self.schema_version,
                        "name": self.name,
                        "version": self.version,
                        "input_snapshot_id": self.input_snapshot_id,
                        "configuration": json_value_to_builtins(configuration),
                        "content": json_value_to_builtins(detached_content),
                    },
                    additional_fields,
                )
            ),
        )

    @property
    def capability(self) -> str:
        return f"{DERIVED_VIEW_CAPABILITY_PREFIX}{self.name}"

    def to_dict(self) -> dict[str, JsonValue]:
        return _include_additional_fields(
            {
                "schema_version": self.schema_version,
                "name": self.name,
                "version": self.version,
                "input_snapshot_id": self.input_snapshot_id,
                "configuration": json_value_to_builtins(self.configuration),
                "content": json_value_to_builtins(self.content),
                "view_id": self.view_id,
            },
            self.additional_fields,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DerivedView:
        if payload.get("schema_version") != SNAPSHOT_CONTRACT_VERSION:
            raise SchemaValidationError("unsupported Derived View schema_version")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise SchemaValidationError("Derived View configuration must be an object")
        view = cls(
            name=cast(str, payload.get("name")),
            version=cast(str, payload.get("version")),
            input_snapshot_id=cast(str, payload.get("input_snapshot_id")),
            configuration=cast(Mapping[str, JsonValue], configuration),
            content=cast(JsonValue, payload.get("content")),
            additional_fields=_additional_fields(
                payload,
                {
                    "schema_version",
                    "name",
                    "version",
                    "input_snapshot_id",
                    "configuration",
                    "content",
                    "view_id",
                },
                path="Derived View additional fields",
            ),
        )
        if payload.get("view_id") != view.view_id:
            raise SchemaValidationError("Derived View identity does not match its content")
        return view


def validate_capability_declarations(
    capabilities: Iterable[str],
    derived_views: Iterable[DerivedView],
    *,
    role: str,
) -> tuple[frozenset[str], tuple[DerivedView, ...]]:
    """Validate capability names and their exact Derived View declarations."""

    normalized_capabilities = frozenset(capabilities)
    for capability in normalized_capabilities:
        if not isinstance(capability, str) or _CAPABILITY.fullmatch(capability) is None:
            raise SchemaValidationError(f"invalid {role} capability {capability!r}")
        if capability not in STANDARD_CAPABILITIES and not capability.startswith(
            DERIVED_VIEW_CAPABILITY_PREFIX
        ):
            raise SchemaValidationError(f"unsupported {role} capability {capability!r}")
    normalized_views = tuple(derived_views)
    if any(not isinstance(view, DerivedView) for view in normalized_views):
        raise SchemaValidationError(f"{role} Derived Views must contain DerivedView objects")
    declared_view_capabilities = {
        capability
        for capability in normalized_capabilities
        if capability.startswith(DERIVED_VIEW_CAPABILITY_PREFIX)
    }
    if declared_view_capabilities != {view.capability for view in normalized_views}:
        raise SchemaValidationError(f"{role} Derived View capabilities do not match its views")
    return normalized_capabilities, normalized_views


def _update_digests(digests: tuple[Any, ...], value: bytes) -> None:
    for digest in digests:
        digest.update(value)


def _update_canonical_value(digests: tuple[Any, ...], value: object) -> None:
    for chunk in iter_canonical_json(value):
        _update_digests(digests, chunk.encode("utf-8"))


def _update_record_array(
    digests: tuple[Any, ...],
    records: Iterable[BenchmarkTopic | TrialDocument | DerivedView],
) -> None:
    _update_digests(digests, b"[")
    for index, record in enumerate(records):
        if index:
            _update_digests(digests, b",")
        _update_canonical_value(digests, record.to_dict())
    _update_digests(digests, b"]")


def _start_snapshot_digest(
    digest: Any,
    *,
    benchmark_lineage: str,
    available_capabilities: Iterable[str],
) -> None:
    digest.update(b'{"available_capabilities":')
    _update_canonical_value((digest,), sorted(available_capabilities))
    digest.update(b',"benchmark_lineage":')
    _update_canonical_value((digest,), benchmark_lineage)
    digest.update(b',"canonicalization_version":')
    _update_canonical_value((digest,), SNAPSHOT_CANONICALIZATION_VERSION)
    digest.update(b',"contract_version":')
    _update_canonical_value((digest,), SNAPSHOT_CONTRACT_VERSION)
    digest.update(b',"derived_views":')


def _seal_snapshot_content(
    *,
    benchmark_lineage: str,
    topics: tuple[BenchmarkTopic, ...],
    trials: Sequence[TrialDocument],
    base_capabilities: Iterable[str],
    available_capabilities: Iterable[str],
    derived_views: tuple[DerivedView, ...],
) -> tuple[str, str, Mapping[str, str], tuple[bool, bool, bool, bool]]:
    """Hash every logical Snapshot component in one canonical record traversal."""

    base_digest = hashlib.sha256()
    topics_digest = hashlib.sha256()
    trials_digest = hashlib.sha256()
    derived_views_digest = hashlib.sha256()
    _start_snapshot_digest(
        base_digest,
        benchmark_lineage=benchmark_lineage,
        available_capabilities=base_capabilities,
    )

    final_digest: Any = base_digest
    if derived_views:
        _update_record_array((base_digest,), ())
        base_digest.update(b',"topics":')
        final_digest = hashlib.sha256()
        _start_snapshot_digest(
            final_digest,
            benchmark_lineage=benchmark_lineage,
            available_capabilities=available_capabilities,
        )
        _update_record_array((final_digest, derived_views_digest), derived_views)
        final_digest.update(b',"topics":')
    else:
        _update_record_array((base_digest, derived_views_digest), ())
        base_digest.update(b',"topics":')

    snapshot_digests = (
        (base_digest, final_digest) if final_digest is not base_digest else (base_digest,)
    )
    _update_record_array((*snapshot_digests, topics_digest), topics)
    _update_digests(snapshot_digests, b',"trials":')
    trial_digests = (*snapshot_digests, trials_digest)
    _update_digests(trial_digests, b"[")
    previous_trial_id: str | None = None
    has_trial_sections = False
    has_trial_core = False
    has_complete_eligibility = False
    has_incomplete_eligibility = False
    for index, trial in enumerate(trials):
        if not isinstance(trial, TrialDocument):
            raise SchemaValidationError("Snapshot trials must contain TrialDocument objects")
        if previous_trial_id is not None and trial.trial_id <= previous_trial_id:
            if trial.trial_id == previous_trial_id:
                raise SchemaValidationError("Snapshot trial identities must be unique")
            raise SchemaValidationError("Snapshot trials must use deterministic identity order")
        if index:
            _update_digests(trial_digests, b",")
        _update_canonical_value(trial_digests, trial.to_dict())
        previous_trial_id = trial.trial_id
        has_trial_sections = has_trial_sections or bool(trial.sections)
        has_trial_core = has_trial_core or trial.typed_clinical_core is not None
        for section in trial.sections:
            if section.role != "eligibility":
                continue
            if section.source_text is None:
                has_incomplete_eligibility = True
            else:
                has_complete_eligibility = True
    if previous_trial_id is None:
        raise SchemaValidationError("Snapshot must contain topics and trials")
    _update_digests(trial_digests, b"]")
    _update_digests(snapshot_digests, b"}")

    base_snapshot_id = "sha256:" + base_digest.hexdigest()
    snapshot_id = "sha256:" + final_digest.hexdigest()
    logical_content_hashes = MappingProxyType(
        {
            "topics": "sha256:" + topics_digest.hexdigest(),
            "trials": "sha256:" + trials_digest.hexdigest(),
            "derived_views": "sha256:" + derived_views_digest.hexdigest(),
        }
    )
    return (
        base_snapshot_id,
        snapshot_id,
        logical_content_hashes,
        (
            has_trial_sections,
            has_trial_core,
            has_complete_eligibility,
            has_incomplete_eligibility,
        ),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkSnapshot:
    """Immutable, judgment-free logical inputs available to one System."""

    benchmark_lineage: str
    snapshot_name: str = field(compare=False)
    topics: tuple[BenchmarkTopic, ...]
    trials: Sequence[TrialDocument]
    available_capabilities: frozenset[str]
    derived_views: tuple[DerivedView, ...] = ()
    base_snapshot_id: str = field(init=False)
    snapshot_id: str = field(init=False)
    _logical_content_hashes: Mapping[str, str] = field(init=False, compare=False, repr=False)

    contract_version = SNAPSHOT_CONTRACT_VERSION
    canonicalization_version = SNAPSHOT_CANONICALIZATION_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.benchmark_lineage, "benchmark_lineage")
        require_non_empty(self.snapshot_name, "snapshot_name")
        topics = tuple(self.topics)
        trials = (
            self.trials
            if isinstance(self.trials, (tuple, FrozenTrialDocuments))
            else tuple(self.trials)
        )
        if not topics or len(trials) == 0:
            raise SchemaValidationError("Snapshot must contain topics and trials")
        if any(not isinstance(item, BenchmarkTopic) for item in topics):
            raise SchemaValidationError("Snapshot topics must contain BenchmarkTopic objects")
        topic_ids = [item.topic_id for item in topics]
        if len(topic_ids) != len(set(topic_ids)):
            raise SchemaValidationError("Snapshot topic identities must be unique")
        if topic_ids != sorted(topic_ids):
            raise SchemaValidationError("Snapshot topics must use deterministic identity order")
        capabilities, raw_derived_views = validate_capability_declarations(
            self.available_capabilities,
            self.derived_views,
            role="Snapshot",
        )
        derived_views = tuple(sorted(raw_derived_views, key=lambda item: item.name))
        required_text_capabilities = {
            CAPABILITY_CANONICAL_PATIENT_TEXT,
            CAPABILITY_CANONICAL_TRIAL_TEXT,
        }
        if not required_text_capabilities <= capabilities:
            raise SchemaValidationError(
                "Snapshot must declare canonical patient and trial text capabilities"
            )
        content_capabilities: tuple[tuple[bool, str, str], ...] = (
            (
                any(topic.evidence_items for topic in topics),
                CAPABILITY_PATIENT_EVIDENCE,
                "patient evidence content",
            ),
            (
                any(topic.typed_patient_core is not None for topic in topics),
                CAPABILITY_TYPED_PATIENT_CORE,
                "typed patient core content",
            ),
        )
        names = [item.name for item in derived_views]
        if len(names) != len(set(names)):
            raise SchemaValidationError("Snapshot Derived View names must be unique")
        base_capabilities = sorted(
            item for item in capabilities if not item.startswith(DERIVED_VIEW_CAPABILITY_PREFIX)
        )
        base_snapshot_id, snapshot_id, logical_content_hashes, trial_summary = (
            _seal_snapshot_content(
                benchmark_lineage=self.benchmark_lineage,
                topics=topics,
                trials=trials,
                base_capabilities=base_capabilities,
                available_capabilities=sorted(capabilities),
                derived_views=derived_views,
            )
        )
        (
            has_trial_sections,
            has_trial_core,
            has_complete_eligibility,
            has_incomplete_eligibility,
        ) = trial_summary
        content_capabilities += (
            (
                has_trial_sections,
                CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
                "semantic trial section content",
            ),
            (
                has_trial_core,
                CAPABILITY_TYPED_TRIAL_CORE,
                "typed trial core content",
            ),
        )
        for populated, capability, label in content_capabilities:
            if populated and capability not in capabilities:
                raise SchemaValidationError(f"Snapshot {label} requires capability {capability!r}")
        if has_complete_eligibility and CAPABILITY_COMPLETE_ELIGIBILITY_TEXT not in capabilities:
            raise SchemaValidationError(
                "Snapshot complete eligibility source text requires its declared capability"
            )
        if CAPABILITY_COMPLETE_ELIGIBILITY_TEXT in capabilities and has_incomplete_eligibility:
            raise SchemaValidationError(
                "Snapshot complete eligibility capability requires every eligibility section"
            )
        has_field_provenance = (
            any(topic.evidence_items for topic in topics)
            or any(topic.typed_patient_core is not None for topic in topics)
            or has_trial_sections
            or has_trial_core
        )
        if has_field_provenance and CAPABILITY_FIELD_PROVENANCE not in capabilities:
            raise SchemaValidationError(
                "Snapshot provenance-bearing content requires field_provenance capability"
            )
        for view in derived_views:
            if view.input_snapshot_id != base_snapshot_id:
                raise SchemaValidationError(
                    f"Derived View {view.name!r} does not reference the base Snapshot identity"
                )
            if view.capability not in capabilities:
                raise SchemaValidationError(
                    f"Derived View {view.name!r} is missing its Snapshot capability"
                )
        object.__setattr__(self, "topics", topics)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "available_capabilities", capabilities)
        object.__setattr__(self, "derived_views", derived_views)
        object.__setattr__(self, "base_snapshot_id", base_snapshot_id)
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "_logical_content_hashes", logical_content_hashes)

    def logical_content_hashes(self) -> dict[str, str]:
        return dict(self._logical_content_hashes)

    def iter_provenance(self) -> Iterable[FieldProvenance]:
        """Yield every source link carried by the logical Snapshot contract."""

        for topic in self.topics:
            for evidence in topic.evidence_items:
                yield from evidence.provenance
            if topic.typed_patient_core is not None:
                yield from topic.typed_patient_core.iter_provenance()
        for trial in self.trials:
            for section in trial.sections:
                yield from section.provenance
                for criterion in section.criteria:
                    yield from criterion.provenance
            core = trial.typed_clinical_core
            if core is None:
                continue
            for value in (core.minimum_age, core.maximum_age, core.sex, core.healthy_volunteers):
                if value is not None:
                    yield from value.provenance
            for conflict in core.conflicts:
                yield from conflict.provenance

    def manifest_dict(self) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            {
                "contract_version": self.contract_version,
                "canonicalization_version": self.canonicalization_version,
                "benchmark_lineage": self.benchmark_lineage,
                "snapshot_name": self.snapshot_name,
                "snapshot_id": self.snapshot_id,
                "base_snapshot_id": self.base_snapshot_id,
                "available_capabilities": sorted(self.available_capabilities),
                "record_counts": {"topics": len(self.topics), "trials": len(self.trials)},
                "logical_content_hashes": self.logical_content_hashes(),
                "derived_views": [item.to_dict() for item in self.derived_views],
            },
        )


__all__ = [
    "AGE_UNITS",
    "CAPABILITY_CANONICAL_PATIENT_TEXT",
    "CAPABILITY_CANONICAL_TRIAL_TEXT",
    "CAPABILITY_FIELD_PROVENANCE",
    "CAPABILITY_PATIENT_EVIDENCE",
    "CAPABILITY_SEMANTIC_TRIAL_SECTIONS",
    "CAPABILITY_TYPED_PATIENT_CORE",
    "CAPABILITY_TYPED_TRIAL_CORE",
    "DERIVED_VIEW_CAPABILITY_PREFIX",
    "PROVENANCE_LOCATOR_KINDS",
    "SECTION_ROLES",
    "SEX_VALUES",
    "SNAPSHOT_CANONICALIZATION_VERSION",
    "SNAPSHOT_CONTRACT_VERSION",
    "STANDARD_CAPABILITIES",
    "AgeBound",
    "AgeQuantity",
    "BenchmarkSnapshot",
    "BenchmarkTopic",
    "CoreFieldConflict",
    "CriterionItem",
    "DerivedView",
    "FieldProvenance",
    "FrozenTrialDocuments",
    "HealthyVolunteerAcceptance",
    "PatientCoreAssertion",
    "PatientEvidenceItem",
    "ProvenanceLocatorKind",
    "SemanticTextSection",
    "SexEligibility",
    "SourceRecordIdentity",
    "TrialDocument",
    "TypedClinicalCore",
    "TypedPatientCore",
    "canonical_json",
    "content_sha256",
    "field_provenance_from_payload",
    "freeze_system_input_options",
]
