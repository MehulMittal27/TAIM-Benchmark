"""Shared immutable entity-version contracts for bidirectional tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import Literal, cast

from taim.constraint_facts import (
    CONSTRAINT_FACT_VIEW_NAME,
    CONSTRAINT_FACT_VIEW_VERSION,
    project_constraint_fact_records,
    project_constraint_fact_view,
)
from taim.contracts import (
    ELIGIBILITY_CRITERION_VIEW_CONTENT_SCHEMA,
    ELIGIBILITY_CRITERION_VIEW_NAME,
    ELIGIBILITY_SPLIT_VERSION,
    canonical_json,
    content_sha256,
    freeze_json_mapping,
    require_exact_keys,
    require_non_empty,
    require_sha256,
)
from taim.schemas import JsonValue, SchemaValidationError, json_value_to_builtins
from taim.snapshot import (
    BenchmarkTopic,
    DerivedView,
    PatientEvidenceItem,
    TrialDocument,
    TypedPatientCore,
    freeze_system_input_options,
)

ENTITY_VERSION_SCHEMA_VERSION = "1.0"
_PATIENT_PROFILE_RULES = frozenset(
    {
        "visible_patient_fields",
        "source_mappings",
        "terminology_mapping",
        "time_policy",
        "missingness",
        "conflict",
        "normalization",
    }
)
_VISIBLE_PATIENT_FIELDS = frozenset(
    {
        "canonical_text",
        "evidence_items",
        "typed_patient_core",
        "extensions",
        "additional_fields",
        "derived_evidence",
    }
)
_PATIENT_EFFECTIVE_TIME_FIELDS = frozenset(
    {
        "timestamp",
        "interval_start",
        "interval_end",
        "effective_time",
        "effective_datetime",
        "effectivetime",
        "effectivedatetime",
    }
)
_OMIT_TEMPORAL_VALUE = object()


def validate_clinical_as_of(value: object) -> datetime:
    """Validate and normalize one clinical cutoff to UTC."""

    if not isinstance(value, datetime):
        raise SchemaValidationError("clinical_as_of must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise SchemaValidationError("clinical_as_of must include a UTC offset")
    return value.astimezone(UTC)


def clinical_as_of_text(value: datetime) -> str:
    return validate_clinical_as_of(value).isoformat().replace("+00:00", "Z")


def parse_clinical_as_of(value: object) -> datetime:
    if not isinstance(value, str):
        raise SchemaValidationError("clinical_as_of must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError("clinical_as_of must be a valid ISO 8601 datetime") from exc
    return validate_clinical_as_of(parsed)


@dataclass(frozen=True, slots=True)
class PatientEvidenceProfile:
    """Named rules defining the exact patient evidence visible to one task."""

    profile_id: str
    profile_version: str
    definition: Mapping[str, JsonValue]
    definition_sha256: str = field(init=False)

    schema_version = ENTITY_VERSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.profile_id, "Patient Evidence Profile profile_id")
        require_non_empty(self.profile_version, "Patient Evidence Profile profile_version")
        if not isinstance(self.definition, Mapping) or not self.definition:
            raise SchemaValidationError(
                "Patient Evidence Profile definition must be a non-empty object"
            )
        missing_rules = _PATIENT_PROFILE_RULES - self.definition.keys()
        if missing_rules:
            raise SchemaValidationError(
                "Patient Evidence Profile definition is missing rules: "
                + ", ".join(sorted(missing_rules))
            )
        visible_fields = self.definition.get("visible_patient_fields")
        if not isinstance(visible_fields, list | tuple) or not visible_fields:
            raise SchemaValidationError(
                "Patient Evidence Profile visible_patient_fields must be a non-empty array"
            )
        if any(
            not isinstance(item, str) or item not in _VISIBLE_PATIENT_FIELDS
            for item in visible_fields
        ):
            raise SchemaValidationError(
                "Patient Evidence Profile visible_patient_fields contains an unsupported field"
            )
        if len(visible_fields) != len(set(visible_fields)):
            raise SchemaValidationError(
                "Patient Evidence Profile visible_patient_fields must be unique"
            )
        if "canonical_text" not in visible_fields:
            raise SchemaValidationError(
                "Patient Evidence Profile must expose canonical_text for the ranking contracts"
            )
        if self.definition.get("time_policy") != "exclude_evidence_after_clinical_as_of":
            raise SchemaValidationError(
                "Patient Evidence Profile time_policy must be "
                "'exclude_evidence_after_clinical_as_of'"
            )
        safe_definition = freeze_system_input_options(
            self.definition,
            path="Patient Evidence Profile definition",
        )
        definition = freeze_json_mapping(
            cast(Mapping[str, JsonValue], safe_definition),
            path="Patient Evidence Profile definition",
        )
        object.__setattr__(self, "definition", definition)
        object.__setattr__(
            self,
            "definition_sha256",
            content_sha256(
                {
                    "profile_id": self.profile_id,
                    "profile_version": self.profile_version,
                    "definition": json_value_to_builtins(definition),
                }
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "definition": json_value_to_builtins(self.definition),
            "definition_sha256": self.definition_sha256,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PatientEvidenceProfile:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "profile_id",
                "profile_version",
                "definition",
                "definition_sha256",
            },
            role="PatientEvidenceProfile",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported PatientEvidenceProfile schema_version")
        definition = payload["definition"]
        if not isinstance(definition, Mapping):
            raise SchemaValidationError("Patient Evidence Profile definition must be an object")
        profile = cls(
            profile_id=cast(str, payload["profile_id"]),
            profile_version=cast(str, payload["profile_version"]),
            definition=cast(Mapping[str, JsonValue], definition),
        )
        require_sha256(payload["definition_sha256"], "Patient Evidence Profile definition_sha256")
        if payload["definition_sha256"] != profile.definition_sha256:
            raise SchemaValidationError(
                "Patient Evidence Profile definition identity does not match its content"
            )
        return profile


def source_grounded_patient_evidence_profile() -> PatientEvidenceProfile:
    """Return TAIM's named all-source-fields research projection."""

    return PatientEvidenceProfile(
        profile_id="taim-source-grounded-patient-evidence",
        profile_version="1.0",
        definition={
            "visible_patient_fields": cast(list[JsonValue], sorted(_VISIBLE_PATIENT_FIELDS)),
            "source_mappings": "preserve_source_identity",
            "terminology_mapping": "preserve_supplied_code_system_and_version",
            "time_policy": "exclude_evidence_after_clinical_as_of",
            "missingness": "absence_is_unknown",
            "conflict": "preserve_all_sources",
            "normalization": "source_grounded_only",
        },
    )


@dataclass(frozen=True, slots=True)
class PatientEntityVersion:
    """One content-addressed patient projection under an evidence profile."""

    patient_id: str
    topic: BenchmarkTopic
    patient_evidence_profile: PatientEvidenceProfile
    clinical_as_of: datetime
    derived_evidence: Mapping[str, JsonValue] = field(default_factory=dict)
    patient_version_id: str = field(init=False)

    schema_version = ENTITY_VERSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.patient_id, "patient_id")
        if not isinstance(self.topic, BenchmarkTopic):
            raise SchemaValidationError("PatientEntityVersion topic must be a BenchmarkTopic")
        if self.patient_id != self.topic.topic_id:
            raise SchemaValidationError("patient_id must match BenchmarkTopic.topic_id")
        if not isinstance(self.patient_evidence_profile, PatientEvidenceProfile):
            raise SchemaValidationError("patient_evidence_profile must be a PatientEvidenceProfile")
        clinical_as_of = validate_clinical_as_of(self.clinical_as_of)
        object.__setattr__(self, "clinical_as_of", clinical_as_of)
        object.__setattr__(
            self,
            "topic",
            _project_patient_topic(
                self.topic,
                profile=self.patient_evidence_profile,
                clinical_as_of=clinical_as_of,
            ),
        )
        if not isinstance(self.derived_evidence, Mapping):
            raise SchemaValidationError("patient derived_evidence must be an object")
        visible_fields = set(
            cast(
                tuple[str, ...], self.patient_evidence_profile.definition["visible_patient_fields"]
            )
        )
        source_derived_evidence = (
            self.derived_evidence if "derived_evidence" in visible_fields else {}
        )
        projected_derived_evidence = _project_temporal_json(
            source_derived_evidence,
            clinical_as_of=clinical_as_of,
        )
        if projected_derived_evidence is _OMIT_TEMPORAL_VALUE:
            projected_derived_evidence = {}
        if not isinstance(projected_derived_evidence, Mapping):
            raise SchemaValidationError("projected patient derived_evidence must be an object")
        safe_derived_evidence = freeze_system_input_options(
            projected_derived_evidence,
            path="patient derived_evidence",
        )
        derived_evidence = freeze_json_mapping(
            cast(Mapping[str, JsonValue], safe_derived_evidence),
            path="patient derived_evidence",
        )
        object.__setattr__(self, "derived_evidence", derived_evidence)
        object.__setattr__(
            self,
            "patient_version_id",
            content_sha256(self._identity_payload()),
        )

    def _identity_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "entity_type": "patient",
            "patient_id": self.patient_id,
            "patient_evidence_profile": self.patient_evidence_profile.to_dict(),
            "clinical_as_of": clinical_as_of_text(self.clinical_as_of),
            "topic": self.topic.to_dict(),
            "derived_evidence": json_value_to_builtins(self.derived_evidence),
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {**self._identity_payload(), "patient_version_id": self.patient_version_id}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PatientEntityVersion:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "entity_type",
                "patient_id",
                "patient_evidence_profile",
                "clinical_as_of",
                "topic",
                "derived_evidence",
                "patient_version_id",
            },
            role="PatientEntityVersion",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported PatientEntityVersion schema_version")
        if payload["entity_type"] != "patient":
            raise SchemaValidationError("PatientEntityVersion entity_type must be 'patient'")
        profile_payload = payload["patient_evidence_profile"]
        topic_payload = payload["topic"]
        derived_evidence = payload["derived_evidence"]
        if not all(
            isinstance(item, Mapping) for item in (profile_payload, topic_payload, derived_evidence)
        ):
            raise SchemaValidationError("PatientEntityVersion nested fields must be objects")
        version = cls(
            patient_id=cast(str, payload["patient_id"]),
            topic=BenchmarkTopic.from_dict(cast(Mapping[str, object], topic_payload)),
            patient_evidence_profile=PatientEvidenceProfile.from_dict(
                cast(Mapping[str, object], profile_payload)
            ),
            clinical_as_of=parse_clinical_as_of(payload["clinical_as_of"]),
            derived_evidence=cast(Mapping[str, JsonValue], derived_evidence),
        )
        require_sha256(payload["patient_version_id"], "patient_version_id")
        if payload["patient_version_id"] != version.patient_version_id:
            raise SchemaValidationError("patient_version_id does not match entity content")
        return version


@dataclass(frozen=True, slots=True)
class TrialVersion:
    """One content-addressed trial projection used by either task direction."""

    trial_id: str
    trial: TrialDocument
    derived_evidence: Mapping[str, JsonValue] = field(default_factory=dict)
    trial_version_id: str = field(init=False)

    schema_version = ENTITY_VERSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.trial_id, "trial_id")
        if not isinstance(self.trial, TrialDocument):
            raise SchemaValidationError("TrialVersion trial must be a TrialDocument")
        if self.trial_id != self.trial.trial_id:
            raise SchemaValidationError("trial_id must match TrialDocument.trial_id")
        if not isinstance(self.derived_evidence, Mapping):
            raise SchemaValidationError("trial derived_evidence must be an object")
        safe_derived_evidence = freeze_system_input_options(
            self.derived_evidence,
            path="trial derived_evidence",
        )
        derived_evidence = freeze_json_mapping(
            cast(Mapping[str, JsonValue], safe_derived_evidence),
            path="trial derived_evidence",
        )
        object.__setattr__(self, "derived_evidence", derived_evidence)
        object.__setattr__(self, "trial_version_id", content_sha256(self._identity_payload()))

    def _identity_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "entity_type": "trial",
            "trial_id": self.trial_id,
            "trial": self.trial.to_dict(),
            "derived_evidence": json_value_to_builtins(self.derived_evidence),
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {**self._identity_payload(), "trial_version_id": self.trial_version_id}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrialVersion:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "entity_type",
                "trial_id",
                "trial",
                "derived_evidence",
                "trial_version_id",
            },
            role="TrialVersion",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported TrialVersion schema_version")
        if payload["entity_type"] != "trial":
            raise SchemaValidationError("TrialVersion entity_type must be 'trial'")
        trial_payload = payload["trial"]
        derived_evidence = payload["derived_evidence"]
        if not isinstance(trial_payload, Mapping) or not isinstance(derived_evidence, Mapping):
            raise SchemaValidationError("TrialVersion nested fields must be objects")
        version = cls(
            trial_id=cast(str, payload["trial_id"]),
            trial=TrialDocument.from_dict(cast(Mapping[str, object], trial_payload)),
            derived_evidence=cast(Mapping[str, JsonValue], derived_evidence),
        )
        require_sha256(payload["trial_version_id"], "trial_version_id")
        if payload["trial_version_id"] != version.trial_version_id:
            raise SchemaValidationError("trial_version_id does not match entity content")
        return version


def version_patients(
    topics: tuple[BenchmarkTopic, ...],
    profile: PatientEvidenceProfile,
    *,
    clinical_as_of: datetime,
    derived_views: tuple[DerivedView, ...] = (),
) -> tuple[PatientEntityVersion, ...]:
    projected_evidence = _project_derived_evidence(
        derived_views,
        tuple(topic.topic_id for topic in topics),
        entity_kind="patient",
    )
    return tuple(
        PatientEntityVersion(
            topic.topic_id,
            topic,
            profile,
            clinical_as_of,
            projected_evidence[topic.topic_id],
        )
        for topic in topics
    )


def _parse_patient_evidence_time(value: str, *, field_name: str) -> datetime:
    if len(value) == 10:
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError as exc:
            raise SchemaValidationError(
                f"patient evidence {field_name} must be a valid ISO 8601 date or datetime"
            ) from exc
        # A date-only source identifies an uncertain instant within that UTC day. Using
        # the latest possible instant keeps the visibility decision fail-closed.
        return datetime.combine(parsed_date, time.max, tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError(
            f"patient evidence {field_name} must be a valid ISO 8601 date or datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SchemaValidationError(f"patient evidence {field_name} must include a UTC offset")
    return parsed.astimezone(UTC)


def _project_temporal_json(
    value: object,
    *,
    clinical_as_of: datetime,
) -> JsonValue | object:
    """Drop structured records whose own effective time is after the cutoff."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = key.casefold().replace("-", "_")
            if normalized_key not in _PATIENT_EFFECTIVE_TIME_FIELDS:
                continue
            if not isinstance(item, str):
                raise SchemaValidationError(f"patient evidence {key} must be an ISO 8601 string")
            if _parse_patient_evidence_time(item, field_name=key) > clinical_as_of:
                return _OMIT_TEMPORAL_VALUE
        projected_mapping: dict[str, JsonValue] = {}
        for key, item in value.items():
            projected = _project_temporal_json(item, clinical_as_of=clinical_as_of)
            if projected is not _OMIT_TEMPORAL_VALUE:
                projected_mapping[key] = cast(JsonValue, projected)
        return projected_mapping
    if isinstance(value, list | tuple):
        projected_items: list[JsonValue] = []
        for item in value:
            projected = _project_temporal_json(item, clinical_as_of=clinical_as_of)
            if projected is not _OMIT_TEMPORAL_VALUE:
                projected_items.append(cast(JsonValue, projected))
        return projected_items
    return cast(JsonValue, value)


def _contains_evidence_after_cutoff(
    value: object,
    *,
    clinical_as_of: datetime,
) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = key.casefold().replace("-", "_")
            if normalized_key not in _PATIENT_EFFECTIVE_TIME_FIELDS:
                continue
            if not isinstance(item, str):
                raise SchemaValidationError(f"patient evidence {key} must be an ISO 8601 string")
            if _parse_patient_evidence_time(item, field_name=key) > clinical_as_of:
                return True
        return any(
            _contains_evidence_after_cutoff(item, clinical_as_of=clinical_as_of)
            for item in value.values()
        )
    if isinstance(value, list | tuple):
        return any(
            _contains_evidence_after_cutoff(item, clinical_as_of=clinical_as_of) for item in value
        )
    return False


def _project_typed_patient_core(
    core: TypedPatientCore | None,
    *,
    clinical_as_of: datetime,
) -> TypedPatientCore | None:
    if core is None:
        return None
    projected = _project_temporal_json(core.to_dict(), clinical_as_of=clinical_as_of)
    if projected is _OMIT_TEMPORAL_VALUE or not isinstance(projected, Mapping):
        return None
    fields = projected.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        return None
    return TypedPatientCore.from_dict(cast(Mapping[str, object], projected))


def _project_patient_topic(
    topic: BenchmarkTopic,
    *,
    profile: PatientEvidenceProfile,
    clinical_as_of: datetime,
) -> BenchmarkTopic:
    """Apply one profile before a patient record becomes System-visible."""

    visible_fields = set(cast(tuple[str, ...], profile.definition["visible_patient_fields"]))
    retained_source_evidence: list[PatientEvidenceItem] = []
    omitted_source_evidence: list[PatientEvidenceItem] = []
    for item in topic.evidence_items:
        effective_times = tuple(
            _parse_patient_evidence_time(value, field_name=name)
            for name, value in (
                ("timestamp", item.timestamp),
                ("interval_start", item.interval_start),
                ("interval_end", item.interval_end),
            )
            if value is not None
        )
        destination = (
            omitted_source_evidence
            if effective_times and max(effective_times) > clinical_as_of
            else retained_source_evidence
        )
        destination.append(item)

    visible_evidence: list[PatientEvidenceItem] = []
    if "evidence_items" in visible_fields:
        for item in retained_source_evidence:
            projected_item = _project_temporal_json(
                item.to_dict(),
                clinical_as_of=clinical_as_of,
            )
            if projected_item is _OMIT_TEMPORAL_VALUE:
                continue
            if not isinstance(projected_item, dict):
                raise SchemaValidationError("projected patient evidence item must be an object")
            projected_item["ordinal"] = len(visible_evidence)
            visible_evidence.append(
                PatientEvidenceItem.from_dict(cast(Mapping[str, object], projected_item))
            )
    projected_extensions = _project_temporal_json(
        topic.extensions if "extensions" in visible_fields else {},
        clinical_as_of=clinical_as_of,
    )
    projected_additional_fields = _project_temporal_json(
        topic.additional_fields if "additional_fields" in visible_fields else {},
        clinical_as_of=clinical_as_of,
    )
    structured_temporal_projection_required = any(
        _contains_evidence_after_cutoff(value, clinical_as_of=clinical_as_of)
        for value in (
            (
                topic.typed_patient_core.to_dict()
                if topic.typed_patient_core is not None and "typed_patient_core" in visible_fields
                else {}
            ),
            topic.extensions if "extensions" in visible_fields else {},
            topic.additional_fields if "additional_fields" in visible_fields else {},
        )
    )
    if projected_extensions is _OMIT_TEMPORAL_VALUE:
        projected_extensions = {}
    if projected_additional_fields is _OMIT_TEMPORAL_VALUE:
        projected_additional_fields = {}
    return BenchmarkTopic(
        topic_id=topic.topic_id,
        source_identity=topic.source_identity,
        canonical_text=_project_patient_canonical_text(
            topic.canonical_text,
            source_evidence=topic.evidence_items,
            retained_evidence=tuple(retained_source_evidence),
            temporal_projection_required=(
                bool(omitted_source_evidence) or structured_temporal_projection_required
            ),
        ),
        evidence_items=tuple(visible_evidence),
        typed_patient_core=(
            _project_typed_patient_core(
                topic.typed_patient_core,
                clinical_as_of=clinical_as_of,
            )
            if "typed_patient_core" in visible_fields
            else None
        ),
        extensions=cast(Mapping[str, JsonValue], projected_extensions),
        additional_fields=cast(Mapping[str, JsonValue], projected_additional_fields),
    )


def _project_patient_canonical_text(
    canonical_text: str,
    *,
    source_evidence: tuple[PatientEvidenceItem, ...],
    retained_evidence: tuple[PatientEvidenceItem, ...],
    temporal_projection_required: bool,
) -> str:
    """Remove cutoff evidence only when the source text recipe is provable."""

    if not temporal_projection_required:
        return canonical_text
    source_texts = [item.text for item in source_evidence]
    retained_ids = {id(item) for item in retained_evidence}
    retained_texts = [item.text for item in source_evidence if id(item) in retained_ids]
    separators = ("\n\n", "\n")
    if len(source_texts) == 1 and canonical_text == source_texts[0]:
        if not retained_texts:
            raise SchemaValidationError(
                "patient canonical_text cannot be safely projected to an empty value"
            )
        return retained_texts[0]
    for separator in separators:
        if canonical_text != separator.join(source_texts):
            continue
        projected = separator.join(retained_texts)
        if not projected:
            raise SchemaValidationError(
                "patient canonical_text cannot be safely projected to an empty value"
            )
        return projected
    raise SchemaValidationError(
        "patient canonical_text cannot be safely projected across clinical_as_of"
    )


def version_trials(
    trials: Sequence[TrialDocument],
    *,
    derived_views: tuple[DerivedView, ...] = (),
) -> tuple[TrialVersion, ...]:
    projected_evidence = _project_derived_evidence(
        derived_views,
        tuple(trial.trial_id for trial in trials),
        entity_kind="trial",
    )
    return tuple(
        TrialVersion(
            trial.trial_id,
            trial,
            projected_evidence[trial.trial_id],
        )
        for trial in trials
    )


def resolve_task_derived_views(
    views: tuple[DerivedView, ...],
    *,
    topics: tuple[BenchmarkTopic, ...],
    trials: tuple[TrialDocument, ...],
    profile: PatientEvidenceProfile,
    clinical_as_of: datetime,
) -> tuple[DerivedView, ...]:
    """Resolve raw Snapshot views to the exact entities and patient profile in a task."""

    visible_fields = set(cast(tuple[str, ...], profile.definition["visible_patient_fields"]))
    patient_derived_evidence_visible = "derived_evidence" in visible_fields
    projected_topics = tuple(
        _project_patient_topic(topic, profile=profile, clinical_as_of=clinical_as_of)
        for topic in topics
    )
    source_topic_by_id = {topic.topic_id: topic for topic in topics}
    projected_topic_by_id = {topic.topic_id: topic for topic in projected_topics}
    patient_ids = frozenset(source_topic_by_id)
    unchanged_patient_ids = frozenset(
        topic_id
        for topic_id in patient_ids
        if source_topic_by_id[topic_id].to_dict() == projected_topic_by_id[topic_id].to_dict()
    )
    trial_ids = frozenset(trial.trial_id for trial in trials)
    resolved: list[DerivedView] = []
    for view in views:
        if not isinstance(view, DerivedView):
            raise SchemaValidationError("task Derived Views must contain DerivedView objects")
        if view.name == CONSTRAINT_FACT_VIEW_NAME and view.version == CONSTRAINT_FACT_VIEW_VERSION:
            visible_constraint_topic_ids = frozenset(
                topic_id
                for topic_id in patient_ids
                if patient_derived_evidence_visible
                and _constraint_fact_patient_inputs_unchanged(
                    source_topic_by_id[topic_id],
                    projected_topic_by_id[topic_id],
                )
            )
            projected_view = project_constraint_fact_view(
                view,
                topic_ids=visible_constraint_topic_ids,
                trial_ids=trial_ids,
            )
            content = cast(Mapping[str, JsonValue], projected_view.content)
            if content.get("topics") or content.get("trials"):
                resolved.append(projected_view)
            continue
        if (
            view.name == ELIGIBILITY_CRITERION_VIEW_NAME
            and view.version == ELIGIBILITY_SPLIT_VERSION
        ):
            eligibility_projected_view = _project_eligibility_task_view(
                view,
                trial_ids=trial_ids,
            )
            if eligibility_projected_view is None:
                continue
            resolved.append(eligibility_projected_view)
            continue
        if not isinstance(view.content, Mapping):
            resolved.append(view)
            continue
        projected_content: dict[str, JsonValue] = {}
        for entity_id in sorted(trial_ids):
            if entity_id in view.content:
                projected_content[entity_id] = cast(JsonValue, view.content[entity_id])
        if patient_derived_evidence_visible:
            for entity_id in sorted(unchanged_patient_ids):
                if entity_id not in view.content:
                    continue
                projected = _project_temporal_json(
                    view.content[entity_id],
                    clinical_as_of=clinical_as_of,
                )
                if projected is not _OMIT_TEMPORAL_VALUE:
                    projected_content[entity_id] = cast(JsonValue, projected)
        if not projected_content:
            continue
        if canonical_json(projected_content) == canonical_json(view.content):
            resolved.append(view)
            continue
        resolved.append(
            DerivedView(
                name=view.name,
                version=view.version,
                input_snapshot_id=view.input_snapshot_id,
                configuration=view.configuration,
                content=projected_content,
                additional_fields=view.additional_fields,
            )
        )
    return tuple(resolved)


def _project_eligibility_task_view(
    view: DerivedView,
    *,
    trial_ids: frozenset[str],
) -> DerivedView | None:
    """Project the authoritative eligibility view to the task's trial corpus."""

    if not isinstance(view.content, Mapping):
        raise SchemaValidationError("eligibility Derived View content must be an object")
    if view.content.get("schema_version") != ELIGIBILITY_CRITERION_VIEW_CONTENT_SCHEMA:
        raise SchemaValidationError("unsupported eligibility Derived View content schema")
    trial_records = view.content.get("trials")
    if not isinstance(trial_records, Mapping):
        raise SchemaValidationError("eligibility Derived View trials content must be an object")
    visible_trials = {
        trial_id: cast(JsonValue, trial_records[trial_id])
        for trial_id in sorted(trial_ids)
        if trial_id in trial_records
    }
    if not visible_trials:
        return None
    projected_content = {
        key: cast(JsonValue, value) for key, value in view.content.items() if key != "trials"
    }
    projected_content["trials"] = visible_trials
    if canonical_json(projected_content) == canonical_json(view.content):
        return view
    return DerivedView(
        name=view.name,
        version=view.version,
        input_snapshot_id=view.input_snapshot_id,
        configuration=view.configuration,
        content=projected_content,
        additional_fields=view.additional_fields,
    )


def _constraint_fact_patient_inputs_unchanged(
    source: BenchmarkTopic,
    projected: BenchmarkTopic,
) -> bool:
    source_core = source.typed_patient_core
    projected_core = projected.typed_patient_core
    return (
        source.canonical_text == projected.canonical_text
        and source.evidence_items == projected.evidence_items
        and (source_core.to_dict() if source_core is not None else None)
        == (projected_core.to_dict() if projected_core is not None else None)
    )


def _project_derived_evidence(
    views: tuple[DerivedView, ...],
    entity_ids: tuple[str, ...],
    *,
    entity_kind: Literal["patient", "trial"],
) -> Mapping[str, Mapping[str, JsonValue]]:
    """Project entity-keyed views for a whole corpus in one pass per view."""

    projected: dict[str, dict[str, JsonValue]] = {entity_id: {} for entity_id in entity_ids}
    selected_ids = frozenset(entity_ids)
    for view in views:
        content = view.content
        if view.name == CONSTRAINT_FACT_VIEW_NAME and view.version == CONSTRAINT_FACT_VIEW_VERSION:
            constraint_records = project_constraint_fact_records(
                view,
                record_kind="topic" if entity_kind == "patient" else "trial",
                record_ids=selected_ids,
            )
            visible_by_entity = {
                entity_id: cast(JsonValue, record)
                for entity_id, record in constraint_records.items()
            }
            scope = "entity"
        elif isinstance(content, Mapping):
            visible_by_entity = {
                entity_id: cast(JsonValue, content[entity_id])
                for entity_id in entity_ids
                if entity_id in content
            }
            scope = "entity"
        else:
            visible_by_entity = {entity_id: content for entity_id in entity_ids}
            scope = "global"
        for entity_id, visible_content in visible_by_entity.items():
            projected[entity_id][view.name] = {
                "version": view.version,
                "configuration": json_value_to_builtins(view.configuration),
                "scope": scope,
                "content": json_value_to_builtins(visible_content),
                "additional_fields": json_value_to_builtins(view.additional_fields),
            }
    return projected


__all__ = [
    "ENTITY_VERSION_SCHEMA_VERSION",
    "PatientEntityVersion",
    "PatientEvidenceProfile",
    "TrialVersion",
    "clinical_as_of_text",
    "parse_clinical_as_of",
    "resolve_task_derived_views",
    "validate_clinical_as_of",
    "version_patients",
    "version_trials",
]
