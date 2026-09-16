"""Versioned, dependency-free schemas used by the TAIM benchmark harness.

The JSON representation of every schema includes ``schema_version``.  The
version is intentionally kept outside the dataclass fields so callers cannot
accidentally create records with mixed versions.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar, cast

SCHEMA_VERSION = "1.0"
# Primary Ranking, Pipeline Depth, Stage Rankings, and budget K are required run semantics.
RUN_MANIFEST_SCHEMA_VERSION = "7.0"

# Primary Ranking artifact id for the ranked Candidate rows in candidates.jsonl.
PRIMARY_RANKING_CANDIDATES = "candidates"
PIPELINE_DEPTH_RETRIEVAL = "retrieval"

_RECOMMENDED_PIPELINE_DEPTHS = frozenset({"retrieval", "rerank", "post_eligibility"})
_OTHER_PIPELINE_DEPTH = re.compile(r"\Aother:[a-z0-9]+(?:[_-][a-z0-9]+)*\Z")
_STAGE_RANKING_NAME = re.compile(r"\A[a-z0-9]+(?:[_-][a-z0-9]+)*\Z")
_SHA256 = re.compile(r"\Asha256:[0-9a-f]{64}\Z")

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class SchemaValidationError(ValueError):
    """Raised when a serialized record does not conform to a TAIM schema."""


def _unsupported_schema_version(
    schema_name: str,
    actual: object,
    expected: str,
) -> SchemaValidationError:
    historical = ""
    if isinstance(actual, str):
        try:
            if int(actual.split(".", 1)[0]) < int(expected.split(".", 1)[0]):
                historical = "historical "
        except ValueError:
            pass
    return SchemaValidationError(
        f"unsupported {historical}{schema_name} schema_version {actual!r}; expected {expected!r}"
    )


def _require_non_empty_string(value: object, field_name: str) -> str:
    """Validate and narrow ``value`` to ``str``; callers may discard the result."""

    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a non-empty string")
    return value


def validate_pipeline_depth(value: object) -> str:
    """Accept recommended Pipeline Depth labels or ``other:<slug>``."""

    if not isinstance(value, str) or not value:
        raise SchemaValidationError(
            "pipeline_depth must be 'retrieval', 'rerank', 'post_eligibility', or 'other:<slug>'"
        )
    if value in _RECOMMENDED_PIPELINE_DEPTHS or _OTHER_PIPELINE_DEPTH.fullmatch(value):
        return value
    raise SchemaValidationError(
        "pipeline_depth must be 'retrieval', 'rerank', 'post_eligibility', "
        f"or 'other:<slug>'; got {value!r}"
    )


def validate_primary_ranking(value: object) -> str:
    """Accept the designated Primary Ranking artifact id for a run."""

    primary_ranking = _require_non_empty_string(value, "primary_ranking")
    if primary_ranking != PRIMARY_RANKING_CANDIDATES:
        raise SchemaValidationError(
            f"primary_ranking must be {PRIMARY_RANKING_CANDIDATES!r}; got {primary_ranking!r}"
        )
    return primary_ranking


def _require_exact_int(value: object, field_name: str) -> None:
    # bool is an int subclass, but accepting True as rank 1 or label 1 hides
    # malformed input.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{field_name} must be an integer")


def normalize_pair_assessment_lineage(
    references: tuple[str, ...],
    value: Mapping[str, object],
) -> dict[str, JsonValue]:
    """Validate collection-derived Pair Assessment lineage on ranking artifacts."""

    if set(value) != {"pair_assessment_collections"}:
        raise SchemaValidationError(
            "ranking Pair Assessment lineage must contain only pair_assessment_collections"
        )
    raw_collections = value["pair_assessment_collections"]
    if not isinstance(raw_collections, list | tuple) or any(
        not isinstance(item, Mapping) for item in raw_collections
    ):
        raise SchemaValidationError(
            "ranking Pair Assessment lineage must contain a collection array"
        )
    collections: list[dict[str, JsonValue]] = []
    expected_keys = {
        "collection_id",
        "source_collection_id",
        "reused_assessment_ids",
        "reused_pair_assessment_input_ids",
    }
    for raw_item in raw_collections:
        item = cast(Mapping[str, object], raw_item)
        if set(item) != expected_keys:
            raise SchemaValidationError(
                "ranking Pair Assessment collection lineage fields are invalid"
            )
        collection_id = item["collection_id"]
        if not isinstance(collection_id, str) or _SHA256.fullmatch(collection_id) is None:
            raise SchemaValidationError("Pair Assessment collection_id must be a SHA-256 digest")
        source_collection_id = item["source_collection_id"]
        if source_collection_id is not None and (
            not isinstance(source_collection_id, str)
            or _SHA256.fullmatch(source_collection_id) is None
        ):
            raise SchemaValidationError(
                "Pair Assessment source_collection_id must be a SHA-256 digest or null"
            )
        raw_assessment_ids = item["reused_assessment_ids"]
        raw_input_ids = item["reused_pair_assessment_input_ids"]
        if not isinstance(raw_assessment_ids, list | tuple) or not isinstance(
            raw_input_ids, list | tuple
        ):
            raise SchemaValidationError("Pair Assessment reused identities must be arrays")
        if any(
            not isinstance(identity, str) or _SHA256.fullmatch(identity) is None
            for identity in (*raw_assessment_ids, *raw_input_ids)
        ):
            raise SchemaValidationError("Pair Assessment reused identities must be SHA-256 digests")
        assessment_ids = tuple(cast(Iterable[str], raw_assessment_ids))
        input_ids = tuple(cast(Iterable[str], raw_input_ids))
        if len(assessment_ids) != len(input_ids):
            raise SchemaValidationError("Pair Assessment reused identity counts do not match")
        if assessment_ids != tuple(sorted(set(assessment_ids))) or input_ids != tuple(
            sorted(set(input_ids))
        ):
            raise SchemaValidationError(
                "Pair Assessment reused identities must be unique and sorted"
            )
        if source_collection_id is None and (assessment_ids or input_ids):
            raise SchemaValidationError(
                "Pair Assessment reused identities require a source collection"
            )
        if source_collection_id is not None and not assessment_ids:
            raise SchemaValidationError(
                "Pair Assessment source collection requires reused identities"
            )
        collections.append(
            {
                "collection_id": collection_id,
                "source_collection_id": cast(str | None, source_collection_id),
                "reused_assessment_ids": list(assessment_ids),
                "reused_pair_assessment_input_ids": list(input_ids),
            }
        )
    collection_ids = tuple(cast(str, item["collection_id"]) for item in collections)
    if collection_ids != tuple(sorted(set(collection_ids))):
        raise SchemaValidationError("Pair Assessment collection lineage must be unique and sorted")
    if collection_ids != references:
        raise SchemaValidationError(
            "Pair Assessment references do not match their collection lineage"
        )
    return {"pair_assessment_collections": cast(list[JsonValue], collections)}


def _validate_json_value(value: object, path: str = "configuration") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"{path} keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise SchemaValidationError(f"{path} contains a non-JSON value: {type(value).__name__}")


def json_value_to_builtins(value: object) -> JsonValue:
    """Copy validated JSON-like data into plain dict/list containers for serialization."""

    if isinstance(value, Mapping):
        return {cast(str, key): json_value_to_builtins(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value_to_builtins(item) for item in value]
    return cast(JsonValue, value)


def freeze_json_value(value: object) -> JsonValue:
    """Recursively freeze JSON data using containers with no mutable base-class API."""

    if isinstance(value, Mapping):
        return cast(
            JsonValue,
            MappingProxyType({key: freeze_json_value(item) for key, item in value.items()}),
        )
    if isinstance(value, list | tuple):
        return cast(JsonValue, tuple(freeze_json_value(item) for item in value))
    return cast(JsonValue, value)


class _ImmutableJsonDict(dict[str, Any]):
    """A JSON-serializable mapping that rejects in-place mutation."""

    @staticmethod
    def _reject(*_args: object, **_kwargs: object) -> None:
        raise TypeError("JSON value is immutable")

    __setitem__ = _reject
    __delitem__ = _reject
    clear = _reject
    pop = _reject
    popitem = _reject  # type: ignore[assignment]
    setdefault = _reject
    update = _reject
    __ior__ = _reject  # type: ignore[assignment]


class _ImmutableJsonList(list[Any]):
    """A JSON-serializable list that rejects in-place mutation."""

    @staticmethod
    def _reject(*_args: object, **_kwargs: object) -> None:
        raise TypeError("JSON value is immutable")

    __setitem__ = _reject  # type: ignore[assignment]
    __delitem__ = _reject  # type: ignore[assignment]
    append = _reject  # type: ignore[assignment]
    clear = _reject
    extend = _reject  # type: ignore[assignment]
    insert = _reject  # type: ignore[assignment]
    pop = _reject  # type: ignore[assignment]
    remove = _reject  # type: ignore[assignment]
    reverse = _reject
    sort = _reject  # type: ignore[assignment]
    __iadd__ = _reject  # type: ignore[assignment]
    __imul__ = _reject  # type: ignore[assignment]


def freeze_json_value_serializable(value: object) -> JsonValue:
    """Validate, detach, and freeze JSON data without breaking ``json.dumps``."""

    detached = json_value_to_builtins(value)
    _validate_json_value(detached)

    def freeze(item: JsonValue) -> JsonValue:
        if isinstance(item, dict):
            return cast(
                JsonValue,
                _ImmutableJsonDict({key: freeze(nested) for key, nested in item.items()}),
            )
        if isinstance(item, list):
            return cast(JsonValue, _ImmutableJsonList(freeze(nested) for nested in item))
        return item

    return freeze(detached)


@dataclass(frozen=True, slots=True)
class _VersionedSchema:
    """Serialization behavior shared by all concrete schemas."""

    schema_version: ClassVar[str] = SCHEMA_VERSION
    # Field names that may be absent in serialized payloads (defaulted on load).
    _optional_fields: ClassVar[frozenset[str]] = frozenset()

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"schema_version": self.schema_version}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if isinstance(value, datetime):
                value = value.astimezone(UTC).isoformat().replace("+00:00", "Z")
            if field_info.name in self._optional_fields and value == {}:
                # Omit empty optional maps so core-only records stay byte-stable.
                continue
            payload[field_info.name] = cast(JsonValue, value)
        return payload

    def to_json(self) -> str:
        """Return stable compact JSON suitable for hashing and JSON Lines."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls: type[_SchemaT], payload: Mapping[str, object]) -> _SchemaT:
        if not isinstance(payload, Mapping):
            raise SchemaValidationError(f"{cls.__name__} must be a JSON object")

        if "schema_version" in payload and payload["schema_version"] != cls.schema_version:
            raise _unsupported_schema_version(
                cls.__name__, payload["schema_version"], cls.schema_version
            )

        expected_fields = {field_info.name for field_info in fields(cls)}
        expected_keys = expected_fields | {"schema_version"}
        missing = (expected_keys - cls._optional_fields) - payload.keys()
        unexpected = payload.keys() - expected_keys
        if missing:
            raise SchemaValidationError(
                f"{cls.__name__} is missing fields: {', '.join(sorted(missing))}"
            )
        if unexpected:
            raise SchemaValidationError(
                f"{cls.__name__} has unexpected fields: {', '.join(sorted(unexpected))}"
            )
        values: dict[str, object] = {}
        for name in expected_fields:
            if name in payload:
                values[name] = payload[name]
        return cls._from_serialized_values(values)

    @classmethod
    def from_json(cls: type[_SchemaT], serialized: str) -> _SchemaT:
        try:
            payload = json.loads(serialized)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SchemaValidationError(f"invalid {cls.__name__} JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SchemaValidationError(f"{cls.__name__} must be a JSON object")
        return cls.from_dict(payload)

    @classmethod
    def _from_serialized_values(cls: type[_SchemaT], values: dict[str, object]) -> _SchemaT:
        try:
            return cls(**values)
        except TypeError as exc:
            raise SchemaValidationError(f"invalid {cls.__name__}: {exc}") from exc


_SchemaT = TypeVar("_SchemaT", bound=_VersionedSchema)


@dataclass(frozen=True, slots=True)
class Candidate(_VersionedSchema):
    run_id: str
    system_id: str
    topic_id: str
    trial_id: str
    rank: int
    score: float

    def __post_init__(self) -> None:
        for field_name in ("run_id", "system_id", "topic_id", "trial_id"):
            _require_non_empty_string(getattr(self, field_name), field_name)
        _require_exact_int(self.rank, "rank")
        if self.rank < 1:
            raise SchemaValidationError("rank must be at least 1")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise SchemaValidationError("score must be a number")
        if not math.isfinite(self.score):
            raise SchemaValidationError("score must be finite")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class RelevanceJudgment(_VersionedSchema):
    topic_id: str
    trial_id: str
    label: int

    def __post_init__(self) -> None:
        _require_non_empty_string(self.topic_id, "topic_id")
        _require_non_empty_string(self.trial_id, "trial_id")
        _require_exact_int(self.label, "label")
        if self.label < 0:
            raise SchemaValidationError("label must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class StageRanking:
    """An optional named ranking emitted at a declared Pipeline Depth."""

    name: str
    pipeline_depth: str
    candidates: tuple[Candidate, ...] = ()
    artifact_hash: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or self.name == PRIMARY_RANKING_CANDIDATES
            or _STAGE_RANKING_NAME.fullmatch(self.name) is None
        ):
            raise SchemaValidationError(
                "stage ranking name must be a lowercase slug other than 'candidates'"
            )
        object.__setattr__(self, "pipeline_depth", validate_pipeline_depth(self.pipeline_depth))
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, Candidate) for candidate in candidates):
            raise SchemaValidationError("stage ranking candidates must be Candidate instances")
        object.__setattr__(self, "candidates", candidates)
        if self.artifact_hash is not None and _SHA256.fullmatch(self.artifact_hash) is None:
            raise SchemaValidationError("stage ranking artifact_hash must be a SHA-256 digest")

    def manifest_dict(self) -> dict[str, JsonValue]:
        """Return the Stage Ranking metadata persisted in a run manifest."""

        if self.artifact_hash is None:
            raise SchemaValidationError("stage ranking artifact_hash is required in a manifest")
        return {
            "name": self.name,
            "pipeline_depth": self.pipeline_depth,
            "artifact_hash": self.artifact_hash,
        }

    @classmethod
    def from_manifest_dict(cls, payload: object) -> StageRanking:
        if not isinstance(payload, Mapping):
            raise SchemaValidationError("stage ranking must be a JSON object")
        expected = {"name", "pipeline_depth", "artifact_hash"}
        missing = expected - payload.keys()
        unexpected = payload.keys() - expected
        if missing:
            raise SchemaValidationError(
                f"stage ranking is missing fields: {', '.join(sorted(missing))}"
            )
        if unexpected:
            raise SchemaValidationError(
                f"stage ranking has unexpected fields: {', '.join(sorted(unexpected))}"
            )
        return cls(
            cast(str, payload["name"]),
            cast(str, payload["pipeline_depth"]),
            artifact_hash=cast(str, payload["artifact_hash"]),
        )


@dataclass(frozen=True, slots=True)
class RunManifest(_VersionedSchema):
    schema_version: ClassVar[str] = RUN_MANIFEST_SCHEMA_VERSION

    run_id: str
    system_id: str
    created_at: datetime
    git_commit: str
    configuration: dict[str, JsonValue]
    benchmark_lineage: str
    snapshot_contract_version: str
    prepared_snapshot_id: str
    task_input_id: str
    system_input_id: str
    evaluation_package_id: str
    benchmark_profile: dict[str, JsonValue]
    primary_ranking: str
    pipeline_depth: str
    budget_k: int
    clinical_as_of: datetime
    patient_evidence_profile: dict[str, JsonValue]
    query_patient_versions: tuple[tuple[str, str], ...]
    trial_corpus_versions: tuple[tuple[str, str], ...]
    runtime_seconds: float | None = None
    candidates_sha256: str | None = None
    stage_rankings: tuple[StageRanking, ...] = ()
    pair_assessment_references: tuple[str, ...] = ()
    reuse_lineage: dict[str, JsonValue] = field(
        default_factory=lambda: {"pair_assessment_collections": []}
    )
    evaluation_artifacts: dict[str, JsonValue] = field(default_factory=dict)
    task: str = field(default="patient_to_trial", init=False)

    _optional_fields: ClassVar[frozenset[str]] = frozenset({"stage_rankings"})

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "system_id",
            "git_commit",
            "benchmark_lineage",
            "snapshot_contract_version",
            "prepared_snapshot_id",
            "task_input_id",
            "system_input_id",
            "evaluation_package_id",
        ):
            _require_non_empty_string(getattr(self, field_name), field_name)
        if _SHA256.fullmatch(self.prepared_snapshot_id) is None:
            raise SchemaValidationError("prepared_snapshot_id must be a SHA-256 digest")
        if _SHA256.fullmatch(self.task_input_id) is None:
            raise SchemaValidationError("task_input_id must be a SHA-256 digest")
        if _SHA256.fullmatch(self.system_input_id) is None:
            raise SchemaValidationError("system_input_id must be a SHA-256 digest")
        if _SHA256.fullmatch(self.evaluation_package_id) is None:
            raise SchemaValidationError("evaluation_package_id must be a SHA-256 digest")
        if not isinstance(self.created_at, datetime):
            raise SchemaValidationError("created_at must be a datetime")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise SchemaValidationError("created_at must include a UTC offset")
        if not isinstance(self.clinical_as_of, datetime):
            raise SchemaValidationError("clinical_as_of must be a datetime")
        if self.clinical_as_of.tzinfo is None or self.clinical_as_of.utcoffset() is None:
            raise SchemaValidationError("clinical_as_of must include a UTC offset")
        if not isinstance(self.configuration, Mapping):
            raise SchemaValidationError("configuration must be a JSON object")
        _validate_json_value(self.configuration)
        if not isinstance(self.benchmark_profile, Mapping):
            raise SchemaValidationError("benchmark_profile must be a JSON object")
        _validate_json_value(self.benchmark_profile, "benchmark_profile")
        _require_non_empty_string(self.benchmark_profile.get("profile_id"), "profile_id")
        definition_sha256 = self.benchmark_profile.get("definition_sha256")
        if not isinstance(definition_sha256, str) or _SHA256.fullmatch(definition_sha256) is None:
            raise SchemaValidationError(
                "benchmark_profile.definition_sha256 must be a SHA-256 digest"
            )
        if not isinstance(self.patient_evidence_profile, Mapping):
            raise SchemaValidationError("patient_evidence_profile must be a JSON object")
        _validate_json_value(self.patient_evidence_profile, "patient_evidence_profile")
        _require_non_empty_string(
            self.patient_evidence_profile.get("profile_id"),
            "patient_evidence_profile.profile_id",
        )
        evidence_profile_hash = self.patient_evidence_profile.get("definition_sha256")
        if (
            not isinstance(evidence_profile_hash, str)
            or _SHA256.fullmatch(evidence_profile_hash) is None
        ):
            raise SchemaValidationError(
                "patient_evidence_profile.definition_sha256 must be a SHA-256 digest"
            )
        object.__setattr__(self, "primary_ranking", validate_primary_ranking(self.primary_ranking))
        object.__setattr__(self, "pipeline_depth", validate_pipeline_depth(self.pipeline_depth))
        _require_exact_int(self.budget_k, "budget_k")
        if self.budget_k < 1:
            raise SchemaValidationError("budget_k must be at least 1")
        if self.runtime_seconds is not None:
            if (
                isinstance(self.runtime_seconds, bool)
                or not isinstance(self.runtime_seconds, int | float)
                or not math.isfinite(self.runtime_seconds)
                or self.runtime_seconds < 0
            ):
                raise SchemaValidationError("runtime_seconds must be a finite non-negative number")
            object.__setattr__(self, "runtime_seconds", float(self.runtime_seconds))
        if self.candidates_sha256 is not None and (
            not isinstance(self.candidates_sha256, str)
            or _SHA256.fullmatch(self.candidates_sha256) is None
        ):
            raise SchemaValidationError("candidates_sha256 must be a SHA-256 digest")
        stage_rankings = tuple(self.stage_rankings)
        if any(not isinstance(stage, StageRanking) for stage in stage_rankings):
            raise SchemaValidationError("stage_rankings must contain StageRanking instances")
        names = [stage.name for stage in stage_rankings]
        if len(names) != len(set(names)):
            raise SchemaValidationError("stage_rankings must have unique names")
        object.__setattr__(self, "stage_rankings", stage_rankings)

        query_versions = self._normalize_entity_lineage(
            self.query_patient_versions,
            entity_name="patient",
        )
        trial_versions = self._normalize_entity_lineage(
            self.trial_corpus_versions,
            entity_name="trial",
        )
        references = tuple(self.pair_assessment_references)
        if any(
            not isinstance(reference, str) or _SHA256.fullmatch(reference) is None
            for reference in references
        ):
            raise SchemaValidationError("pair_assessment_references must contain SHA-256 digests")
        if references != tuple(sorted(set(references))):
            raise SchemaValidationError("pair_assessment_references must be unique and sorted")
        if not isinstance(self.reuse_lineage, Mapping):
            raise SchemaValidationError("reuse_lineage must be a JSON object")
        reuse_lineage = normalize_pair_assessment_lineage(references, self.reuse_lineage)
        if not isinstance(self.evaluation_artifacts, Mapping):
            raise SchemaValidationError("evaluation_artifacts must be a JSON object")
        if self.evaluation_artifacts and set(self.evaluation_artifacts) != {
            "scorecard_json_sha256",
            "scorecard_markdown_sha256",
        }:
            raise SchemaValidationError(
                "evaluation_artifacts must contain the exact scorecard hash fields"
            )
        if any(
            not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
            for digest in self.evaluation_artifacts.values()
        ):
            raise SchemaValidationError("evaluation_artifacts values must be SHA-256 digests")
        system_input = self.configuration.get("system_input")
        if not isinstance(system_input, Mapping):
            raise SchemaValidationError(
                "configuration.system_input must contain the exact patient-to-trial task input"
            )
        self._validate_embedded_system_input(
            system_input,
            query_versions=query_versions,
            trial_versions=trial_versions,
        )
        object.__setattr__(self, "query_patient_versions", query_versions)
        object.__setattr__(self, "trial_corpus_versions", trial_versions)
        object.__setattr__(self, "pair_assessment_references", references)
        object.__setattr__(self, "reuse_lineage", reuse_lineage)
        normalized_artifacts = json.loads(
            json.dumps(self.evaluation_artifacts, allow_nan=False, ensure_ascii=False)
        )
        object.__setattr__(self, "evaluation_artifacts", normalized_artifacts)

        # Detach the manifest from a caller-owned mutable mapping and normalize
        # nested Mapping implementations to plain JSON containers.
        normalized = json.loads(json.dumps(self.configuration, allow_nan=False, ensure_ascii=False))
        object.__setattr__(self, "configuration", normalized)
        normalized_profile = json.loads(
            json.dumps(self.benchmark_profile, allow_nan=False, ensure_ascii=False)
        )
        object.__setattr__(self, "benchmark_profile", normalized_profile)
        normalized_evidence_profile = json.loads(
            json.dumps(self.patient_evidence_profile, allow_nan=False, ensure_ascii=False)
        )
        object.__setattr__(self, "patient_evidence_profile", normalized_evidence_profile)
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        object.__setattr__(self, "clinical_as_of", self.clinical_as_of.astimezone(UTC))

    def _validate_embedded_system_input(
        self,
        system_input: Mapping[str, object],
        *,
        query_versions: tuple[tuple[str, str], ...],
        trial_versions: tuple[tuple[str, str], ...],
    ) -> None:
        """Recompute both forward identities from their complete embedded content."""

        # Local imports avoid a schemas -> entity_versions -> schemas import cycle.
        from taim.contracts import (
            content_sha256,
            portable_system_input_identity_value,
            system_input_identity_value,
            task_input_membership_id,
        )
        from taim.entity_versions import (
            PatientEntityVersion,
            PatientEvidenceProfile,
            TrialVersion,
            clinical_as_of_text,
            parse_clinical_as_of,
        )
        from taim.snapshot import DerivedView, freeze_system_input_options

        task_fields = {
            "snapshot_contract_version",
            "task",
            "benchmark_lineage",
            "prepared_snapshot_id",
            "clinical_as_of",
            "patient_evidence_profile",
            "query_patient_versions",
            "trial_corpus_versions",
            "available_capabilities",
            "derived_views",
            "task_input_id",
        }
        identity_version = system_input.get("identity_version", "1.0")
        if identity_version not in {"1.0", "2.0"}:
            raise SchemaValidationError("unsupported forward System Input identity_version")
        expected_fields = task_fields | {
            "system_input_id",
            "top_k",
            "metric_cutoff",
            "options",
            "required_capabilities",
            "optional_capabilities",
        }
        if identity_version == "2.0":
            expected_fields.add("identity_version")
        if set(system_input) != expected_fields:
            raise SchemaValidationError(
                "configuration.system_input fields do not match the forward System input contract"
            )
        expected_input_values = {
            "snapshot_contract_version": self.snapshot_contract_version,
            "task": self.task,
            "benchmark_lineage": self.benchmark_lineage,
            "prepared_snapshot_id": self.prepared_snapshot_id,
            "clinical_as_of": clinical_as_of_text(self.clinical_as_of),
        }
        if any(system_input[name] != value for name, value in expected_input_values.items()):
            raise SchemaValidationError(
                "configuration.system_input does not match the forward manifest identity"
            )

        profile_payload = system_input["patient_evidence_profile"]
        raw_patient_versions = system_input["query_patient_versions"]
        raw_trial_versions = system_input["trial_corpus_versions"]
        raw_derived_views = system_input["derived_views"]
        if not isinstance(profile_payload, Mapping):
            raise SchemaValidationError(
                "configuration.system_input Patient Evidence Profile must be an object"
            )
        if (
            not isinstance(raw_patient_versions, list | tuple)
            or not isinstance(raw_trial_versions, list | tuple)
            or not isinstance(raw_derived_views, list | tuple)
        ):
            raise SchemaValidationError(
                "configuration.system_input entity versions and Derived Views must be arrays"
            )
        if any(
            not isinstance(value, Mapping)
            for values in (raw_patient_versions, raw_trial_versions, raw_derived_views)
            for value in values
        ):
            raise SchemaValidationError(
                "configuration.system_input entity versions and Derived Views must be objects"
            )

        profile = PatientEvidenceProfile.from_dict(cast(Mapping[str, object], profile_payload))
        patient_entities = tuple(
            PatientEntityVersion.from_dict(cast(Mapping[str, object], value))
            for value in raw_patient_versions
        )
        trial_entities = tuple(
            TrialVersion.from_dict(cast(Mapping[str, object], value))
            for value in raw_trial_versions
        )
        derived_views = tuple(
            DerivedView.from_dict(cast(Mapping[str, object], value)) for value in raw_derived_views
        )
        input_clinical_as_of = parse_clinical_as_of(system_input["clinical_as_of"])
        if input_clinical_as_of != self.clinical_as_of.astimezone(UTC):
            raise SchemaValidationError(
                "configuration.system_input clinical_as_of does not match the forward manifest"
            )
        if profile.to_dict() != self.patient_evidence_profile:
            raise SchemaValidationError(
                "configuration.system_input Patient Evidence Profile does not match"
            )
        if any(
            entity.patient_evidence_profile != profile
            or entity.clinical_as_of != input_clinical_as_of
            for entity in patient_entities
        ):
            raise SchemaValidationError(
                "forward patient entity versions do not match the Patient Evidence Profile "
                "and clinical cutoff"
            )
        input_query_versions = tuple(
            (entity.patient_id, entity.patient_version_id) for entity in patient_entities
        )
        input_trial_versions = tuple(
            (entity.trial_id, entity.trial_version_id) for entity in trial_entities
        )
        if input_query_versions != query_versions:
            raise SchemaValidationError("forward patient entity-version lineage does not match")
        if input_trial_versions != trial_versions:
            raise SchemaValidationError("forward trial entity-version lineage does not match")

        available = self._sorted_unique_strings(
            system_input["available_capabilities"],
            field_name="available_capabilities",
        )
        required = self._sorted_unique_strings(
            system_input["required_capabilities"],
            field_name="required_capabilities",
        )
        optional = self._sorted_unique_strings(
            system_input["optional_capabilities"],
            field_name="optional_capabilities",
        )
        if set(required) & set(optional):
            raise SchemaValidationError(
                "forward required and optional System capabilities must not overlap"
            )
        if not set(required) <= set(available):
            raise SchemaValidationError(
                "forward task input is missing required System capabilities"
            )

        top_k = system_input["top_k"]
        metric_cutoff = system_input["metric_cutoff"]
        _require_exact_int(top_k, "configuration.system_input.top_k")
        _require_exact_int(metric_cutoff, "configuration.system_input.metric_cutoff")
        if cast(int, top_k) < 1 or cast(int, metric_cutoff) < 1:
            raise SchemaValidationError(
                "configuration.system_input ranking cutoffs must be positive"
            )
        if top_k != self.budget_k:
            raise SchemaValidationError(
                "configuration.system_input.top_k does not match manifest budget_k"
            )
        options = system_input["options"]
        if not isinstance(options, Mapping):
            raise SchemaValidationError("configuration.system_input.options must be an object")
        validated_options = freeze_system_input_options(
            options,
            path="configuration.system_input.options",
        )

        task_payload: dict[str, JsonValue] = {
            "snapshot_contract_version": self.snapshot_contract_version,
            "task": self.task,
            "benchmark_lineage": self.benchmark_lineage,
            "prepared_snapshot_id": self.prepared_snapshot_id,
            "clinical_as_of": clinical_as_of_text(input_clinical_as_of),
            "patient_evidence_profile": profile.to_dict(),
            "query_patient_versions": [entity.to_dict() for entity in patient_entities],
            "trial_corpus_versions": [entity.to_dict() for entity in trial_entities],
            "available_capabilities": list(available),
            "derived_views": [view.to_dict() for view in derived_views],
        }
        task_input_id = content_sha256(task_payload)
        if system_input["task_input_id"] != task_input_id or self.task_input_id != task_input_id:
            raise SchemaValidationError(
                "forward task_input_id does not match embedded task input content"
            )
        identity_options = system_input_identity_value(validated_options, path="options")
        identity_payload: dict[str, JsonValue] = {
            "task_input_id": task_input_id,
            "top_k": cast(int, top_k),
            "metric_cutoff": cast(int, metric_cutoff),
            "options": identity_options,
            "required_capabilities": list(required),
            "optional_capabilities": list(optional),
        }
        if identity_version == "2.0":
            identity_options, _omitted = portable_system_input_identity_value(
                validated_options,
                path="options",
            )
            identity_payload.update(
                {
                    "identity_version": "2.0",
                    "task_input_membership_id": task_input_membership_id(
                        task="patient_to_trial",
                        patient_entity_versions=input_query_versions,
                        trial_entity_versions=input_trial_versions,
                    ),
                    "options": identity_options,
                }
            )
        system_input_id = content_sha256(identity_payload)
        if (
            system_input["system_input_id"] != system_input_id
            or self.system_input_id != system_input_id
        ):
            raise SchemaValidationError(
                "forward system_input_id does not match embedded System input content"
            )

    @staticmethod
    def _sorted_unique_strings(value: object, *, field_name: str) -> tuple[str, ...]:
        if not isinstance(value, list | tuple) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise SchemaValidationError(
                f"configuration.system_input.{field_name} must be an array of strings"
            )
        normalized = cast(tuple[str, ...], tuple(value))
        if normalized != tuple(sorted(set(normalized))):
            raise SchemaValidationError(
                f"configuration.system_input.{field_name} must be unique and sorted"
            )
        return normalized

    @staticmethod
    def _normalize_entity_lineage(
        value: object,
        *,
        entity_name: str,
    ) -> tuple[tuple[str, str], ...]:
        if not isinstance(value, list | tuple):
            raise SchemaValidationError(f"{entity_name} entity-version lineage must be an array")
        rows = tuple(value)
        if not rows or any(
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not row[0]
            or not isinstance(row[1], str)
            or _SHA256.fullmatch(row[1]) is None
            for row in rows
        ):
            raise SchemaValidationError(f"{entity_name} entity-version lineage is invalid")
        normalized = cast(tuple[tuple[str, str], ...], rows)
        if normalized != tuple(sorted(set(normalized))):
            raise SchemaValidationError(
                f"{entity_name} entity-version lineage must be unique and sorted"
            )
        return normalized

    def to_dict(self) -> dict[str, JsonValue]:
        if self.runtime_seconds is None or self.candidates_sha256 is None:
            raise SchemaValidationError(
                "runtime_seconds and candidates_sha256 are required in a saved run manifest"
            )
        payload = _VersionedSchema.to_dict(self)
        payload["stage_rankings"] = [stage.manifest_dict() for stage in self.stage_rankings]
        payload["query_patient_versions"] = [
            {"patient_id": entity_id, "patient_version_id": version_id}
            for entity_id, version_id in self.query_patient_versions
        ]
        payload["trial_corpus_versions"] = [
            {"trial_id": entity_id, "trial_version_id": version_id}
            for entity_id, version_id in self.trial_corpus_versions
        ]
        payload["pair_assessment_references"] = list(self.pair_assessment_references)
        return payload

    @classmethod
    def _from_serialized_values(cls, values: dict[str, object]) -> RunManifest:
        created_at = values.get("created_at")
        if not isinstance(created_at, str):
            raise SchemaValidationError("created_at must be an ISO 8601 string")
        try:
            parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError("created_at must be a valid ISO 8601 datetime") from exc
        clinical_as_of = values.get("clinical_as_of")
        if not isinstance(clinical_as_of, str):
            raise SchemaValidationError("clinical_as_of must be an ISO 8601 string")
        try:
            parsed_clinical_as_of = datetime.fromisoformat(clinical_as_of.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError("clinical_as_of must be a valid ISO 8601 datetime") from exc
        if values.get("task") != "patient_to_trial":
            raise SchemaValidationError("RunManifest task must be 'patient_to_trial'")
        stage_rankings: tuple[StageRanking, ...] = ()
        if "stage_rankings" in values:
            serialized_stages = values["stage_rankings"]
            if not isinstance(serialized_stages, list):
                raise SchemaValidationError("stage_rankings must be an array")
            stage_rankings = tuple(
                StageRanking.from_manifest_dict(stage) for stage in serialized_stages
            )
        query_patient_versions = cls._parse_serialized_entity_lineage(
            values.get("query_patient_versions"),
            entity_name="patient",
        )
        trial_corpus_versions = cls._parse_serialized_entity_lineage(
            values.get("trial_corpus_versions"),
            entity_name="trial",
        )
        pair_assessment_references = values.get("pair_assessment_references")
        if not isinstance(pair_assessment_references, list):
            raise SchemaValidationError("pair_assessment_references must be an array")
        reuse_lineage = values.get("reuse_lineage")
        if not isinstance(reuse_lineage, Mapping):
            raise SchemaValidationError("reuse_lineage must be a JSON object")
        evaluation_artifacts = values.get("evaluation_artifacts")
        if not isinstance(evaluation_artifacts, Mapping):
            raise SchemaValidationError("evaluation_artifacts must be a JSON object")
        try:
            return cls(
                run_id=cast(str, values["run_id"]),
                system_id=cast(str, values["system_id"]),
                created_at=parsed_created_at,
                git_commit=cast(str, values["git_commit"]),
                configuration=cast(dict[str, JsonValue], values["configuration"]),
                benchmark_lineage=cast(str, values["benchmark_lineage"]),
                snapshot_contract_version=cast(str, values["snapshot_contract_version"]),
                prepared_snapshot_id=cast(str, values["prepared_snapshot_id"]),
                task_input_id=cast(str, values["task_input_id"]),
                system_input_id=cast(str, values["system_input_id"]),
                evaluation_package_id=cast(str, values["evaluation_package_id"]),
                benchmark_profile=cast(dict[str, JsonValue], values["benchmark_profile"]),
                primary_ranking=cast(str, values["primary_ranking"]),
                pipeline_depth=cast(str, values["pipeline_depth"]),
                budget_k=cast(int, values["budget_k"]),
                clinical_as_of=parsed_clinical_as_of,
                patient_evidence_profile=cast(
                    dict[str, JsonValue], values["patient_evidence_profile"]
                ),
                query_patient_versions=query_patient_versions,
                trial_corpus_versions=trial_corpus_versions,
                runtime_seconds=cast(float, values["runtime_seconds"]),
                candidates_sha256=cast(str, values["candidates_sha256"]),
                stage_rankings=stage_rankings,
                pair_assessment_references=tuple(cast(list[str], pair_assessment_references)),
                reuse_lineage=cast(dict[str, JsonValue], reuse_lineage),
                evaluation_artifacts=cast(dict[str, JsonValue], evaluation_artifacts),
            )
        except TypeError as exc:
            raise SchemaValidationError(f"invalid RunManifest: {exc}") from exc

    @staticmethod
    def _parse_serialized_entity_lineage(
        value: object,
        *,
        entity_name: str,
    ) -> tuple[tuple[str, str], ...]:
        if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
            raise SchemaValidationError(f"{entity_name} entity-version lineage is invalid")
        entity_id_key = f"{entity_name}_id"
        version_id_key = f"{entity_name}_version_id"
        rows: list[tuple[str, str]] = []
        for raw_row in value:
            row = cast(Mapping[str, object], raw_row)
            if set(row) != {entity_id_key, version_id_key}:
                raise SchemaValidationError(
                    f"{entity_name} entity-version lineage fields are invalid"
                )
            rows.append(
                (
                    cast(str, row[entity_id_key]),
                    cast(str, row[version_id_key]),
                )
            )
        return tuple(rows)


def write_candidates_jsonl(path: str | Path, candidates: Iterable[Candidate]) -> None:
    """Write candidates as deterministic UTF-8 JSON Lines."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for candidate in candidates:
            if not isinstance(candidate, Candidate):
                raise SchemaValidationError("candidate records must be Candidate instances")
            stream.write(candidate.to_json())
            stream.write("\n")


def _write_schema_jsonl(
    path: str | Path,
    records: Iterable[_SchemaT],
    schema_type: type[_SchemaT],
    record_name: str,
) -> None:
    """Write one concrete schema type as canonical UTF-8 JSON Lines."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            if not isinstance(record, schema_type):
                raise SchemaValidationError(
                    f"{record_name} records must be {schema_type.__name__} instances"
                )
            stream.write(record.to_json())
            stream.write("\n")


def write_qrels_jsonl(path: str | Path, qrels: Iterable[RelevanceJudgment]) -> None:
    """Write relevance judgments as canonical UTF-8 JSON Lines."""

    _write_schema_jsonl(path, qrels, RelevanceJudgment, "qrel")


def read_candidates_jsonl(path: str | Path) -> list[Candidate]:
    """Load and validate candidate JSON Lines, rejecting blank records."""

    candidates: list[Candidate] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise SchemaValidationError(f"blank candidate record at line {line_number}")
            try:
                candidates.append(Candidate.from_json(line))
            except SchemaValidationError as exc:
                raise SchemaValidationError(
                    f"invalid candidate record at line {line_number}: {exc}"
                ) from exc
    return candidates


def _read_schema_jsonl(
    path: str | Path,
    schema_type: type[_SchemaT],
    record_name: str,
) -> list[_SchemaT]:
    """Load one concrete schema type, rejecting blank or malformed records."""

    records: list[_SchemaT] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise SchemaValidationError(f"blank {record_name} record at line {line_number}")
            try:
                records.append(schema_type.from_json(line))
            except SchemaValidationError as exc:
                raise SchemaValidationError(
                    f"invalid {record_name} record at line {line_number}: {exc}"
                ) from exc
    return records


def read_qrels_jsonl(path: str | Path) -> list[RelevanceJudgment]:
    """Load and validate relevance judgment JSON Lines."""

    return _read_schema_jsonl(path, RelevanceJudgment, "qrel")


def write_manifest(path: str | Path, manifest: RunManifest) -> None:
    """Write one validated run manifest as stable, human-readable JSON."""

    if not isinstance(manifest, RunManifest):
        raise SchemaValidationError("manifest must be a RunManifest instance")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        manifest.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    destination.write_text(f"{serialized}\n", encoding="utf-8")


def read_manifest(path: str | Path) -> RunManifest:
    """Load and validate a run manifest."""

    try:
        serialized = Path(path).read_text(encoding="utf-8")
    except OSError:
        raise
    return RunManifest.from_json(serialized)


__all__ = [
    "PIPELINE_DEPTH_RETRIEVAL",
    "PRIMARY_RANKING_CANDIDATES",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "Candidate",
    "JsonValue",
    "RelevanceJudgment",
    "RunManifest",
    "SchemaValidationError",
    "StageRanking",
    "freeze_json_value",
    "json_value_to_builtins",
    "read_candidates_jsonl",
    "read_manifest",
    "read_qrels_jsonl",
    "validate_pipeline_depth",
    "validate_primary_ranking",
    "write_candidates_jsonl",
    "write_manifest",
    "write_qrels_jsonl",
]
