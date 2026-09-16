"""Direction-independent, judgment-free Pair Assessment contracts."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

from taim.contracts import (
    SNAPSHOT_CONTRACT_VERSION,
    EvaluatorOnlyMaterial,
    content_sha256,
    freeze_json_mapping,
    require_exact_keys,
    require_non_empty,
    require_sha256,
)
from taim.entity_versions import (
    PatientEntityVersion,
    TrialVersion,
    clinical_as_of_text,
    parse_clinical_as_of,
    validate_clinical_as_of,
)
from taim.file_hash import sha256_file
from taim.pair_evidence import pair_evidence_reference_paths
from taim.schemas import (
    JsonValue,
    SchemaValidationError,
    freeze_json_value,
    json_value_to_builtins,
)
from taim.snapshot import freeze_system_input_options

PAIR_ASSESSMENT_SCHEMA_VERSION = "1.0"
PAIR_ASSESSMENT_TASK = "pair_assessment"
PAIR_ASSESSMENT_CATEGORIES = frozenset(
    {"potential_candidate", "needs_information", "not_candidate_for_run"}
)
NEEDS_INFORMATION_TARGETS = frozenset({"patient_evidence", "trial_clarification", "both"})


def _require_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise SchemaValidationError(f"{name} must be an array")
    result = tuple(value)
    for item in result:
        require_non_empty(item, name)
    return cast(tuple[str, ...], result)


def _require_label(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2):
        raise SchemaValidationError("Pair Assessment Judgment label must be 0, 1, or 2")
    return value


def _canonical_raw_output_bytes(raw_output: Mapping[str, object]) -> bytes:
    try:
        serialized = json.dumps(
            json_value_to_builtins(raw_output),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("raw assessor output must be a finite JSON object") from exc
    return f"{serialized}\n".encode()


@dataclass(frozen=True, slots=True)
class PairAssessmentProfile:
    """Named aggregation and evidence policy for one Pair Assessment."""

    profile_id: str
    profile_version: str
    definition: Mapping[str, JsonValue]
    definition_sha256: str = field(init=False)

    schema_version = PAIR_ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.profile_id, "Pair Assessment Profile profile_id")
        require_non_empty(self.profile_version, "Pair Assessment Profile profile_version")
        if not isinstance(self.definition, Mapping) or not self.definition:
            raise SchemaValidationError(
                "Pair Assessment Profile definition must be a non-empty object"
            )
        safe_definition = freeze_system_input_options(
            self.definition,
            path="Pair Assessment Profile definition",
        )
        definition = freeze_json_mapping(
            cast(Mapping[str, JsonValue], safe_definition),
            path="Pair Assessment Profile definition",
        )
        object.__setattr__(self, "definition", definition)
        object.__setattr__(
            self,
            "definition_sha256",
            content_sha256(self._definition_payload()),
        )

    def _definition_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "definition": json_value_to_builtins(self.definition),
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {**self._definition_payload(), "definition_sha256": self.definition_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PairAssessmentProfile:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "profile_id",
                "profile_version",
                "definition",
                "definition_sha256",
            },
            role="PairAssessmentProfile",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported PairAssessmentProfile schema_version")
        definition = payload["definition"]
        if not isinstance(definition, Mapping):
            raise SchemaValidationError("Pair Assessment Profile definition must be an object")
        profile = cls(
            profile_id=cast(str, payload["profile_id"]),
            profile_version=cast(str, payload["profile_version"]),
            definition=cast(Mapping[str, JsonValue], definition),
        )
        require_sha256(payload["definition_sha256"], "Pair Assessment Profile definition_sha256")
        if payload["definition_sha256"] != profile.definition_sha256:
            raise SchemaValidationError(
                "Pair Assessment Profile identity does not match its definition"
            )
        return profile


@dataclass(frozen=True, slots=True)
class AssessorIdentity:
    """Complete method identity required before an assessment can be reused."""

    assessor_id: str
    implementation_version: str
    dependency_id: str
    model_id: str
    prompt_id: str
    configuration: Mapping[str, JsonValue]
    aggregation_policy_id: str
    assessor_identity_id: str = field(init=False)

    schema_version = PAIR_ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "assessor_id",
            "implementation_version",
            "dependency_id",
            "model_id",
            "prompt_id",
            "aggregation_policy_id",
        ):
            require_non_empty(getattr(self, name), f"Assessor Identity {name}")
        if not isinstance(self.configuration, Mapping):
            raise SchemaValidationError("Assessor Identity configuration must be an object")
        safe_configuration = freeze_system_input_options(
            self.configuration,
            path="Assessor Identity configuration",
        )
        configuration = freeze_json_mapping(
            cast(Mapping[str, JsonValue], safe_configuration),
            path="Assessor Identity configuration",
        )
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(
            self,
            "assessor_identity_id",
            content_sha256(self._identity_payload()),
        )

    def _identity_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "assessor_id": self.assessor_id,
            "implementation_version": self.implementation_version,
            "dependency_id": self.dependency_id,
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "configuration": json_value_to_builtins(self.configuration),
            "aggregation_policy_id": self.aggregation_policy_id,
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {**self._identity_payload(), "assessor_identity_id": self.assessor_identity_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AssessorIdentity:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "assessor_id",
                "implementation_version",
                "dependency_id",
                "model_id",
                "prompt_id",
                "configuration",
                "aggregation_policy_id",
                "assessor_identity_id",
            },
            role="AssessorIdentity",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported AssessorIdentity schema_version")
        configuration = payload["configuration"]
        if not isinstance(configuration, Mapping):
            raise SchemaValidationError("Assessor Identity configuration must be an object")
        identity = cls(
            assessor_id=cast(str, payload["assessor_id"]),
            implementation_version=cast(str, payload["implementation_version"]),
            dependency_id=cast(str, payload["dependency_id"]),
            model_id=cast(str, payload["model_id"]),
            prompt_id=cast(str, payload["prompt_id"]),
            configuration=cast(Mapping[str, JsonValue], configuration),
            aggregation_policy_id=cast(str, payload["aggregation_policy_id"]),
        )
        require_sha256(payload["assessor_identity_id"], "assessor_identity_id")
        if payload["assessor_identity_id"] != identity.assessor_identity_id:
            raise SchemaValidationError("assessor_identity_id does not match assessor content")
        return identity


@dataclass(frozen=True, slots=True)
class PairAssessmentInput:
    """Exact pair-local evidence visible to an assessor, independent of ranking direction."""

    patient_version: PatientEntityVersion
    trial_version: TrialVersion
    clinical_as_of: datetime
    assessment_profile: PairAssessmentProfile
    visible_pair_evidence: Mapping[str, JsonValue]
    pair_assessment_input_id: str = field(init=False)

    schema_version = PAIR_ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.patient_version, PatientEntityVersion):
            raise SchemaValidationError("patient_version must be a PatientEntityVersion")
        if not isinstance(self.trial_version, TrialVersion):
            raise SchemaValidationError("trial_version must be a TrialVersion")
        if not isinstance(self.assessment_profile, PairAssessmentProfile):
            raise SchemaValidationError("assessment_profile must be a PairAssessmentProfile")
        if not isinstance(self.visible_pair_evidence, Mapping):
            raise SchemaValidationError("visible_pair_evidence must be an object")
        clinical_as_of = validate_clinical_as_of(self.clinical_as_of)
        if clinical_as_of != self.patient_version.clinical_as_of:
            raise SchemaValidationError(
                "Pair Assessment clinical_as_of must match its patient projection"
            )
        object.__setattr__(self, "clinical_as_of", clinical_as_of)
        safe_evidence = freeze_system_input_options(
            self.visible_pair_evidence,
            path="Pair Assessment visible evidence",
        )
        visible_evidence = freeze_json_mapping(
            cast(Mapping[str, JsonValue], safe_evidence),
            path="Pair Assessment visible evidence",
        )
        object.__setattr__(self, "visible_pair_evidence", visible_evidence)
        object.__setattr__(
            self,
            "pair_assessment_input_id",
            content_sha256(self._identity_payload()),
        )

    def _identity_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "patient_version": self.patient_version.to_dict(),
            "trial_version": self.trial_version.to_dict(),
            "clinical_as_of": clinical_as_of_text(self.clinical_as_of),
            "assessment_profile": self.assessment_profile.to_dict(),
            "visible_pair_evidence": json_value_to_builtins(self.visible_pair_evidence),
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            **self._identity_payload(),
            "pair_assessment_input_id": self.pair_assessment_input_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PairAssessmentInput:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "patient_version",
                "trial_version",
                "clinical_as_of",
                "assessment_profile",
                "visible_pair_evidence",
                "pair_assessment_input_id",
            },
            role="PairAssessmentInput",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported PairAssessmentInput schema_version")
        patient = payload["patient_version"]
        trial = payload["trial_version"]
        profile = payload["assessment_profile"]
        evidence = payload["visible_pair_evidence"]
        if not all(isinstance(item, Mapping) for item in (patient, trial, profile, evidence)):
            raise SchemaValidationError("PairAssessmentInput nested fields must be objects")
        pair_input = cls(
            patient_version=PatientEntityVersion.from_dict(cast(Mapping[str, object], patient)),
            trial_version=TrialVersion.from_dict(cast(Mapping[str, object], trial)),
            clinical_as_of=parse_clinical_as_of(payload["clinical_as_of"]),
            assessment_profile=PairAssessmentProfile.from_dict(cast(Mapping[str, object], profile)),
            visible_pair_evidence=cast(Mapping[str, JsonValue], evidence),
        )
        require_sha256(payload["pair_assessment_input_id"], "pair_assessment_input_id")
        if payload["pair_assessment_input_id"] != pair_input.pair_assessment_input_id:
            raise SchemaValidationError(
                "pair_assessment_input_id does not match exact pair input content"
            )
        return pair_input


@dataclass(frozen=True, slots=True)
class AssessmentEvidence:
    criterion_scope: str
    criterion: str
    justification: str
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.criterion_scope, "Assessment Evidence criterion_scope")
        require_non_empty(self.criterion, "Assessment Evidence criterion")
        require_non_empty(self.justification, "Assessment Evidence justification")
        references = tuple(self.references)
        if references != tuple(sorted(set(references))):
            raise SchemaValidationError("Assessment Evidence references must be unique and sorted")
        for reference in references:
            require_non_empty(reference, "Assessment Evidence reference")
        object.__setattr__(self, "references", references)

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "criterion_scope": self.criterion_scope,
            "criterion": self.criterion,
            "justification": self.justification,
        }
        if self.references:
            payload["references"] = list(self.references)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AssessmentEvidence:
        allowed = {"criterion_scope", "criterion", "justification", "references"}
        if set(payload) not in (allowed - {"references"}, allowed):
            raise SchemaValidationError("AssessmentEvidence has unexpected or missing fields")
        references = payload.get("references", [])
        if not isinstance(references, list) or any(
            not isinstance(item, str) for item in references
        ):
            raise SchemaValidationError("AssessmentEvidence references must be strings")
        return cls(
            criterion_scope=cast(str, payload["criterion_scope"]),
            criterion=cast(str, payload["criterion"]),
            justification=cast(str, payload["justification"]),
            references=tuple(cast(list[str], references)),
        )


@dataclass(frozen=True, slots=True)
class MissingInformation:
    target: str
    criterion: str
    justification: str

    def __post_init__(self) -> None:
        if self.target not in NEEDS_INFORMATION_TARGETS:
            raise SchemaValidationError(
                "Missing Information target must be patient_evidence, trial_clarification, or both"
            )
        require_non_empty(self.criterion, "Missing Information criterion")
        require_non_empty(self.justification, "Missing Information justification")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "target": self.target,
            "criterion": self.criterion,
            "justification": self.justification,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MissingInformation:
        require_exact_keys(
            payload,
            {"target", "criterion", "justification"},
            role="MissingInformation",
        )
        return cls(
            target=cast(str, payload["target"]),
            criterion=cast(str, payload["criterion"]),
            justification=cast(str, payload["justification"]),
        )


@dataclass(frozen=True, slots=True)
class RawOutputProvenance:
    artifact_name: str
    raw_output_sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        require_non_empty(self.artifact_name, "raw output artifact_name")
        require_sha256(self.raw_output_sha256, "raw_output_sha256")
        if isinstance(self.byte_length, bool) or not isinstance(self.byte_length, int):
            raise SchemaValidationError("raw output byte_length must be an integer")
        if self.byte_length < 1:
            raise SchemaValidationError("raw output byte_length must be positive")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "artifact_name": self.artifact_name,
            "raw_output_sha256": self.raw_output_sha256,
            "byte_length": self.byte_length,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RawOutputProvenance:
        require_exact_keys(
            payload,
            {"artifact_name", "raw_output_sha256", "byte_length"},
            role="RawOutputProvenance",
        )
        return cls(
            artifact_name=cast(str, payload["artifact_name"]),
            raw_output_sha256=cast(str, payload["raw_output_sha256"]),
            byte_length=cast(int, payload["byte_length"]),
        )


@dataclass(frozen=True, slots=True)
class PairAssessment:
    """Immutable assessment output; never a ranking or downstream decision."""

    pair_assessment_input_id: str
    patient_id: str
    patient_version_id: str
    trial_id: str
    trial_version_id: str
    category: str
    needs_information_target: str | None
    reasons: tuple[str, ...]
    supporting_evidence: tuple[AssessmentEvidence, ...]
    missing_information: tuple[MissingInformation, ...]
    assessor: AssessorIdentity
    raw_output_provenance: RawOutputProvenance
    created_at: datetime
    assessment_id: str = field(init=False)

    schema_version = PAIR_ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_sha256(self.pair_assessment_input_id, "pair_assessment_input_id")
        require_non_empty(self.patient_id, "Pair Assessment patient_id")
        require_sha256(self.patient_version_id, "Pair Assessment patient_version_id")
        require_non_empty(self.trial_id, "Pair Assessment trial_id")
        require_sha256(self.trial_version_id, "Pair Assessment trial_version_id")
        if self.category not in PAIR_ASSESSMENT_CATEGORIES:
            raise SchemaValidationError("invalid Pair Assessment category")
        reasons = tuple(self.reasons)
        for reason in reasons:
            require_non_empty(reason, "Pair Assessment reason")
        support = tuple(self.supporting_evidence)
        missing = tuple(self.missing_information)
        if any(not isinstance(item, AssessmentEvidence) for item in support):
            raise SchemaValidationError(
                "supporting_evidence must contain AssessmentEvidence records"
            )
        if any(not isinstance(item, MissingInformation) for item in missing):
            raise SchemaValidationError(
                "missing_information must contain MissingInformation records"
            )
        if not isinstance(self.assessor, AssessorIdentity):
            raise SchemaValidationError("assessor must be an AssessorIdentity")
        if not isinstance(self.raw_output_provenance, RawOutputProvenance):
            raise SchemaValidationError("raw_output_provenance must be RawOutputProvenance")
        if self.category == "not_candidate_for_run":
            if not reasons or not support:
                raise SchemaValidationError(
                    "not_candidate_for_run requires reasons and supporting evidence"
                )
            if self.needs_information_target is not None:
                raise SchemaValidationError(
                    "not_candidate_for_run cannot have a needs-information target"
                )
            if "condition_mismatch" in reasons and not any(
                item.criterion_scope == "condition" for item in support
            ):
                raise SchemaValidationError(
                    "condition_mismatch requires explicit condition evidence"
                )
        elif self.category == "needs_information":
            if self.needs_information_target not in NEEDS_INFORMATION_TARGETS or not missing:
                raise SchemaValidationError(
                    "needs_information requires a valid target and missing evidence"
                )
            missing_targets = {item.target for item in missing}
            expected_target = (
                next(iter(missing_targets))
                if len(missing_targets) == 1 and "both" not in missing_targets
                else "both"
            )
            if self.needs_information_target != expected_target:
                raise SchemaValidationError(
                    "needs_information target does not summarize missing evidence"
                )
            if reasons:
                raise SchemaValidationError("needs_information cannot have mismatch reasons")
        else:
            if self.needs_information_target is not None or reasons or missing:
                raise SchemaValidationError(
                    "potential_candidate cannot have reasons or missing-information fields"
                )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "supporting_evidence", support)
        object.__setattr__(self, "missing_information", missing)
        object.__setattr__(self, "created_at", validate_clinical_as_of(self.created_at))
        object.__setattr__(self, "assessment_id", content_sha256(self._identity_payload()))

    @property
    def assessor_identity_id(self) -> str:
        return self.assessor.assessor_identity_id

    def _identity_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "pair_assessment_input_id": self.pair_assessment_input_id,
            "patient_id": self.patient_id,
            "patient_version_id": self.patient_version_id,
            "trial_id": self.trial_id,
            "trial_version_id": self.trial_version_id,
            "category": self.category,
            "needs_information_target": self.needs_information_target,
            "reasons": list(self.reasons),
            "supporting_evidence": [item.to_dict() for item in self.supporting_evidence],
            "missing_information": [item.to_dict() for item in self.missing_information],
            "assessor": self.assessor.to_dict(),
            "raw_output_provenance": self.raw_output_provenance.to_dict(),
            "created_at": clinical_as_of_text(self.created_at),
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {**self._identity_payload(), "assessment_id": self.assessment_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PairAssessment:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "pair_assessment_input_id",
                "patient_id",
                "patient_version_id",
                "trial_id",
                "trial_version_id",
                "category",
                "needs_information_target",
                "reasons",
                "supporting_evidence",
                "missing_information",
                "assessor",
                "raw_output_provenance",
                "created_at",
                "assessment_id",
            },
            role="PairAssessment",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported PairAssessment schema_version")
        support = payload["supporting_evidence"]
        missing = payload["missing_information"]
        assessor = payload["assessor"]
        provenance = payload["raw_output_provenance"]
        if not isinstance(support, list) or not isinstance(missing, list):
            raise SchemaValidationError("Pair Assessment evidence fields must be arrays")
        if not isinstance(assessor, Mapping) or not isinstance(provenance, Mapping):
            raise SchemaValidationError("Pair Assessment provenance fields must be objects")
        if any(not isinstance(item, Mapping) for item in (*support, *missing)):
            raise SchemaValidationError("Pair Assessment evidence entries must be objects")
        needs_target = payload["needs_information_target"]
        if needs_target is not None and not isinstance(needs_target, str):
            raise SchemaValidationError("needs_information_target must be a string or null")
        assessment = cls(
            pair_assessment_input_id=cast(str, payload["pair_assessment_input_id"]),
            patient_id=cast(str, payload["patient_id"]),
            patient_version_id=cast(str, payload["patient_version_id"]),
            trial_id=cast(str, payload["trial_id"]),
            trial_version_id=cast(str, payload["trial_version_id"]),
            category=cast(str, payload["category"]),
            needs_information_target=needs_target,
            reasons=_require_string_tuple(payload["reasons"], "Pair Assessment reasons"),
            supporting_evidence=tuple(
                AssessmentEvidence.from_dict(cast(Mapping[str, object], item)) for item in support
            ),
            missing_information=tuple(
                MissingInformation.from_dict(cast(Mapping[str, object], item)) for item in missing
            ),
            assessor=AssessorIdentity.from_dict(cast(Mapping[str, object], assessor)),
            raw_output_provenance=RawOutputProvenance.from_dict(
                cast(Mapping[str, object], provenance)
            ),
            created_at=parse_clinical_as_of(payload["created_at"]),
        )
        require_sha256(payload["assessment_id"], "assessment_id")
        if payload["assessment_id"] != assessment.assessment_id:
            raise SchemaValidationError("assessment_id does not match assessment content")
        return assessment


@dataclass(frozen=True, slots=True)
class PairAssessmentResult:
    pair_input: PairAssessmentInput
    assessment: PairAssessment
    raw_output: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.pair_input, PairAssessmentInput):
            raise SchemaValidationError("pair_input must be a PairAssessmentInput")
        if not isinstance(self.assessment, PairAssessment):
            raise SchemaValidationError("assessment must be a PairAssessment")
        _validate_assessment_entity_lineage(self.pair_input, self.assessment)
        if not isinstance(self.raw_output, Mapping):
            raise SchemaValidationError("raw_output must be an object")
        raw_bytes = _canonical_raw_output_bytes(self.raw_output)
        expected = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
        if self.assessment.raw_output_provenance.raw_output_sha256 != expected:
            raise SchemaValidationError("raw output hash does not match its provenance")
        if self.assessment.raw_output_provenance.byte_length != len(raw_bytes):
            raise SchemaValidationError("raw output length does not match its provenance")
        detached = json.loads(raw_bytes)
        object.__setattr__(
            self,
            "raw_output",
            cast(Mapping[str, object], freeze_json_value(detached)),
        )


@dataclass(frozen=True, slots=True)
class PairAssessmentJudgment:
    pair_assessment_input_id: str
    patient_version_id: str
    trial_version_id: str
    label: int

    schema_version = PAIR_ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_sha256(self.pair_assessment_input_id, "pair_assessment_input_id")
        require_sha256(self.patient_version_id, "patient_version_id")
        require_sha256(self.trial_version_id, "trial_version_id")
        object.__setattr__(self, "label", _require_label(self.label))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "pair_assessment_input_id": self.pair_assessment_input_id,
            "patient_version_id": self.patient_version_id,
            "trial_version_id": self.trial_version_id,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PairAssessmentJudgment:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "pair_assessment_input_id",
                "patient_version_id",
                "trial_version_id",
                "label",
            },
            role="PairAssessmentJudgment",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported PairAssessmentJudgment schema_version")
        return cls(
            pair_assessment_input_id=cast(str, payload["pair_assessment_input_id"]),
            patient_version_id=cast(str, payload["patient_version_id"]),
            trial_version_id=cast(str, payload["trial_version_id"]),
            label=_require_label(payload["label"]),
        )


@dataclass(frozen=True, slots=True)
class PairAssessmentEvaluationPackage(EvaluatorOnlyMaterial):
    judgments: tuple[PairAssessmentJudgment, ...]
    provenance: Mapping[str, JsonValue]
    evaluation_package_id: str = field(init=False)

    schema_version = PAIR_ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        judgments = tuple(sorted(self.judgments, key=lambda item: item.pair_assessment_input_id))
        if not judgments:
            raise SchemaValidationError("Pair Assessment Evaluation Package requires judgments")
        if any(not isinstance(item, PairAssessmentJudgment) for item in judgments):
            raise SchemaValidationError(
                "Pair Assessment Evaluation Package contains an invalid judgment"
            )
        ids = [item.pair_assessment_input_id for item in judgments]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError(
                "Pair Assessment Evaluation Package has duplicate pair inputs"
            )
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise SchemaValidationError(
                "Pair Assessment Evaluation Package provenance must be a non-empty object"
            )
        provenance = freeze_json_mapping(
            self.provenance,
            path="Pair Assessment Evaluation Package provenance",
        )
        object.__setattr__(self, "judgments", judgments)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "evaluation_package_id",
            content_sha256(self._identity_payload()),
        )

    def _identity_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "task": PAIR_ASSESSMENT_TASK,
            "judgments": [item.to_dict() for item in self.judgments],
            "provenance": json_value_to_builtins(self.provenance),
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            **self._identity_payload(),
            "evaluation_package_id": self.evaluation_package_id,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> PairAssessmentEvaluationPackage:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "task",
                "judgments",
                "provenance",
                "evaluation_package_id",
            },
            role="PairAssessmentEvaluationPackage",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError(
                "unsupported PairAssessmentEvaluationPackage schema_version"
            )
        if payload["task"] != PAIR_ASSESSMENT_TASK:
            raise SchemaValidationError("Pair Assessment Evaluation Package has the wrong task")
        judgments = payload["judgments"]
        provenance = payload["provenance"]
        if not isinstance(judgments, list) or not isinstance(provenance, Mapping):
            raise SchemaValidationError(
                "Pair Assessment Evaluation Package nested fields have invalid types"
            )
        if any(not isinstance(item, Mapping) for item in judgments):
            raise SchemaValidationError(
                "Pair Assessment Evaluation Package judgments must be objects"
            )
        package = cls(
            judgments=tuple(
                PairAssessmentJudgment.from_dict(cast(Mapping[str, object], item))
                for item in judgments
            ),
            provenance=cast(Mapping[str, JsonValue], provenance),
        )
        require_sha256(payload["evaluation_package_id"], "evaluation_package_id")
        if payload["evaluation_package_id"] != package.evaluation_package_id:
            raise SchemaValidationError(
                "Pair Assessment Evaluation Package identity does not match its content"
            )
        return package


@dataclass(frozen=True, slots=True)
class PairAssessmentExecutionProvenance:
    """Mandatory run-level provenance for one persisted assessment collection."""

    git_commit: str
    working_tree_dirty: bool
    environment: Mapping[str, object]
    seeds: Mapping[str, object]
    runtime_seconds: float

    def __post_init__(self) -> None:
        require_non_empty(self.git_commit, "Pair Assessment provenance git_commit")
        if not isinstance(self.working_tree_dirty, bool):
            raise SchemaValidationError(
                "Pair Assessment provenance working_tree_dirty must be a boolean"
            )
        for name in ("environment", "seeds"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise SchemaValidationError(
                    f"Pair Assessment provenance {name} must be a non-empty object"
                )
            safe_value = freeze_system_input_options(
                value,
                path=f"Pair Assessment provenance {name}",
            )
            object.__setattr__(
                self,
                name,
                freeze_json_mapping(
                    cast(Mapping[str, JsonValue], safe_value),
                    path=f"Pair Assessment provenance {name}",
                ),
            )
        if (
            isinstance(self.runtime_seconds, bool)
            or not isinstance(self.runtime_seconds, int | float)
            or not math.isfinite(self.runtime_seconds)
            or self.runtime_seconds < 0
        ):
            raise SchemaValidationError(
                "Pair Assessment provenance runtime_seconds must be finite and non-negative"
            )
        object.__setattr__(self, "runtime_seconds", float(self.runtime_seconds))

    def to_dict(self) -> dict[str, object]:
        return {
            "git_commit": self.git_commit,
            "working_tree_dirty": self.working_tree_dirty,
            "environment": json_value_to_builtins(self.environment),
            "seeds": json_value_to_builtins(self.seeds),
            "runtime_seconds": self.runtime_seconds,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> PairAssessmentExecutionProvenance:
        require_exact_keys(
            payload,
            {
                "git_commit",
                "working_tree_dirty",
                "environment",
                "seeds",
                "runtime_seconds",
            },
            role="PairAssessmentExecutionProvenance",
        )
        environment = payload["environment"]
        seeds = payload["seeds"]
        if not isinstance(environment, Mapping) or not isinstance(seeds, Mapping):
            raise SchemaValidationError(
                "Pair Assessment execution provenance nested fields must be objects"
            )
        return cls(
            git_commit=cast(str, payload["git_commit"]),
            working_tree_dirty=cast(bool, payload["working_tree_dirty"]),
            environment=cast(Mapping[str, object], environment),
            seeds=cast(Mapping[str, object], seeds),
            runtime_seconds=cast(float, payload["runtime_seconds"]),
        )


@dataclass(frozen=True, slots=True)
class StoredPairAssessmentRun:
    directory: Path
    assessments: tuple[PairAssessment, ...]
    inputs: Mapping[str, PairAssessmentInput]
    raw_outputs: Mapping[str, Mapping[str, object]]
    snapshot_contract_version: str
    assessor_identity_id: str
    execution_provenance: PairAssessmentExecutionProvenance
    reuse_lineage: Mapping[str, object]
    collection_id: str
    assessments_hash: str
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class PairAssessmentReusePlan:
    source_collection_id: str | None
    reused: tuple[PairAssessment, ...]
    compute: tuple[PairAssessmentInput, ...]
    reused_results: tuple[PairAssessmentResult, ...]

    def __post_init__(self) -> None:
        reused = tuple(self.reused)
        compute = tuple(self.compute)
        reused_results = tuple(self.reused_results)
        if any(not isinstance(item, PairAssessment) for item in reused):
            raise SchemaValidationError("reuse plan reused values must be Pair Assessments")
        if any(not isinstance(item, PairAssessmentInput) for item in compute):
            raise SchemaValidationError("reuse plan compute values must be Pair Assessment Inputs")
        if any(not isinstance(item, PairAssessmentResult) for item in reused_results):
            raise SchemaValidationError("reuse plan results must be Pair Assessment Results")
        if reused != tuple(item.assessment for item in reused_results):
            raise SchemaValidationError(
                "reuse plan assessments do not match their persisted results"
            )
        if reused or self.source_collection_id is not None:
            require_sha256(self.source_collection_id, "reuse plan source_collection_id")
        reused_input_ids = tuple(item.pair_assessment_input_id for item in reused)
        compute_input_ids = tuple(item.pair_assessment_input_id for item in compute)
        if len(reused_input_ids) != len(set(reused_input_ids)) or len(compute_input_ids) != len(
            set(compute_input_ids)
        ):
            raise SchemaValidationError("reuse plan contains duplicate Pair Assessment inputs")
        if set(reused_input_ids) & set(compute_input_ids):
            raise SchemaValidationError("reuse plan cannot both reuse and compute one input")
        object.__setattr__(self, "reused", reused)
        object.__setattr__(self, "compute", compute)
        object.__setattr__(self, "reused_results", reused_results)


def pair_assessment_collection_lineage(
    runs: Iterable[StoredPairAssessmentRun],
) -> tuple[tuple[str, ...], dict[str, JsonValue]]:
    """Derive ranking-artifact references from validated stored collections."""

    stored_runs = tuple(runs)
    if any(not isinstance(run, StoredPairAssessmentRun) for run in stored_runs):
        raise TypeError("Pair Assessment references must be StoredPairAssessmentRun instances")
    if len({run.collection_id for run in stored_runs}) != len(stored_runs):
        raise SchemaValidationError("Pair Assessment references must be unique")
    rows: list[dict[str, JsonValue]] = []
    for run in sorted(stored_runs, key=lambda item: item.collection_id):
        if load_pair_assessment_run(run.directory) != run:
            raise SchemaValidationError(
                "Pair Assessment reference does not match its stored collection"
            )
        lineage = _normalized_reuse_lineage(run.reuse_lineage)
        rows.append(
            {
                "collection_id": run.collection_id,
                "source_collection_id": cast(str | None, lineage["source_collection_id"]),
                "reused_assessment_ids": cast(list[JsonValue], lineage["reused_assessment_ids"]),
                "reused_pair_assessment_input_ids": cast(
                    list[JsonValue], lineage["reused_pair_assessment_input_ids"]
                ),
            }
        )
    references = tuple(cast(str, row["collection_id"]) for row in rows)
    return references, {"pair_assessment_collections": cast(list[JsonValue], rows)}


def _validate_assessment_entity_lineage(
    pair_input: PairAssessmentInput,
    assessment: PairAssessment,
) -> None:
    if assessment.pair_assessment_input_id != pair_input.pair_assessment_input_id:
        raise SchemaValidationError("Pair Assessment row input does not match output")
    if (
        assessment.patient_id != pair_input.patient_version.patient_id
        or assessment.patient_version_id != pair_input.patient_version.patient_version_id
        or assessment.trial_id != pair_input.trial_version.trial_id
        or assessment.trial_version_id != pair_input.trial_version.trial_version_id
    ):
        raise SchemaValidationError(
            "Pair Assessment entity lineage does not match its embedded exact input"
        )
    allowed_references = pair_evidence_reference_paths(pair_input.to_dict())
    unresolved = {
        reference
        for evidence in assessment.supporting_evidence
        for reference in evidence.references
        if reference not in allowed_references
    }
    if unresolved:
        raise SchemaValidationError(
            "Pair Assessment evidence references must resolve inside the supplied input: "
            + ", ".join(sorted(unresolved))
        )


def _normalized_reuse_lineage(value: Mapping[str, object]) -> dict[str, object]:
    require_exact_keys(
        value,
        {
            "source_collection_id",
            "reused_assessment_ids",
            "reused_pair_assessment_input_ids",
        },
        role="Pair Assessment reuse lineage",
    )
    source = value["source_collection_id"]
    assessment_ids = _require_string_tuple(
        value["reused_assessment_ids"],
        "reused_assessment_ids",
    )
    input_ids = _require_string_tuple(
        value["reused_pair_assessment_input_ids"],
        "reused_pair_assessment_input_ids",
    )
    if source is None:
        if assessment_ids or input_ids:
            raise SchemaValidationError(
                "Pair Assessment reuse lineage requires a source collection"
            )
    else:
        require_sha256(source, "Pair Assessment reuse source_collection_id")
        if not assessment_ids:
            raise SchemaValidationError(
                "Pair Assessment reuse source requires reused assessment identities"
            )
    if len(assessment_ids) != len(input_ids):
        raise SchemaValidationError("Pair Assessment reuse lineage counts do not match")
    if assessment_ids != tuple(sorted(set(assessment_ids))):
        raise SchemaValidationError("reused_assessment_ids must be unique and sorted")
    if input_ids != tuple(sorted(set(input_ids))):
        raise SchemaValidationError("reused_pair_assessment_input_ids must be unique and sorted")
    for value_id in assessment_ids + input_ids:
        require_sha256(value_id, "Pair Assessment reused identity")
    return {
        "source_collection_id": source,
        "reused_assessment_ids": list(assessment_ids),
        "reused_pair_assessment_input_ids": list(input_ids),
    }


def make_raw_output_provenance(
    artifact_name: str,
    raw_output: Mapping[str, object],
) -> RawOutputProvenance:
    """Describe the exact canonical raw-output bytes persisted by the writer."""

    raw_bytes = _canonical_raw_output_bytes(raw_output)
    return RawOutputProvenance(
        artifact_name=artifact_name,
        raw_output_sha256="sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
        byte_length=len(raw_bytes),
    )


def write_pair_assessment_run(
    directory: str | Path,
    results: Iterable[PairAssessmentResult],
    *,
    execution_provenance: PairAssessmentExecutionProvenance,
    reuse_plan: PairAssessmentReusePlan | None = None,
) -> StoredPairAssessmentRun:
    """Atomically persist a deterministic Pair Assessment collection and raw outputs."""

    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"Pair Assessment directory already exists: {destination}")
    if not isinstance(execution_provenance, PairAssessmentExecutionProvenance):
        raise TypeError("execution_provenance must be a PairAssessmentExecutionProvenance")
    fresh_rows = tuple(results)
    if any(
        not evidence.references
        for result in fresh_rows
        for evidence in result.assessment.supporting_evidence
    ):
        raise SchemaValidationError(
            "new Pair Assessment supporting evidence must cite the supplied exact input"
        )
    if reuse_plan is None:
        reused_rows: tuple[PairAssessmentResult, ...] = ()
        reuse_lineage = _normalized_reuse_lineage(
            {
                "source_collection_id": None,
                "reused_assessment_ids": [],
                "reused_pair_assessment_input_ids": [],
            }
        )
    else:
        if not isinstance(reuse_plan, PairAssessmentReusePlan):
            raise TypeError("reuse_plan must be a PairAssessmentReusePlan")
        expected_compute_ids = {item.pair_assessment_input_id for item in reuse_plan.compute}
        actual_compute_ids = {item.pair_input.pair_assessment_input_id for item in fresh_rows}
        if actual_compute_ids != expected_compute_ids:
            raise SchemaValidationError("fresh Pair Assessment results do not close the reuse plan")
        reused_rows = reuse_plan.reused_results
        reuse_lineage = _normalized_reuse_lineage(
            {
                "source_collection_id": (reuse_plan.source_collection_id if reused_rows else None),
                "reused_assessment_ids": sorted(
                    item.assessment.assessment_id for item in reused_rows
                ),
                "reused_pair_assessment_input_ids": sorted(
                    item.pair_input.pair_assessment_input_id for item in reused_rows
                ),
            }
        )
    rows = tuple(
        sorted(
            (*reused_rows, *fresh_rows),
            key=lambda item: (
                item.pair_input.patient_version.patient_id,
                item.pair_input.trial_version.trial_id,
                item.pair_input.pair_assessment_input_id,
            ),
        )
    )
    if not rows:
        raise SchemaValidationError("Pair Assessment collection must be non-empty")
    input_ids = [item.pair_input.pair_assessment_input_id for item in rows]
    assessment_ids = [item.assessment.assessment_id for item in rows]
    if len(input_ids) != len(set(input_ids)) or len(assessment_ids) != len(set(assessment_ids)):
        raise SchemaValidationError("Pair Assessment collection contains duplicate identities")
    assessor_identity_ids = {item.assessment.assessor_identity_id for item in rows}
    if len(assessor_identity_ids) != 1:
        raise SchemaValidationError(
            "Pair Assessment collection must use one exact assessor identity"
        )
    assessor_identity_id = assessor_identity_ids.pop()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name or 'pair-assessments'}.",
            dir=destination.parent,
        )
    )
    try:
        raw_directory = staging / "raw"
        raw_directory.mkdir()
        raw_artifacts: list[dict[str, JsonValue]] = []
        assessment_path = staging / "pair-assessments.jsonl"
        with assessment_path.open("w", encoding="utf-8", newline="\n") as stream:
            for result in rows:
                raw_name = f"{result.assessment.assessment_id}.json"
                raw_path = raw_directory / raw_name
                raw_bytes = _canonical_raw_output_bytes(result.raw_output)
                raw_path.write_bytes(raw_bytes)
                if (
                    sha256_file(raw_path)
                    != result.assessment.raw_output_provenance.raw_output_sha256
                ):
                    raise SchemaValidationError("raw output changed before persistence")
                raw_artifacts.append(
                    {
                        "assessment_id": result.assessment.assessment_id,
                        "path": f"raw/{raw_name}",
                        "sha256": sha256_file(raw_path),
                        "byte_length": len(raw_bytes),
                    }
                )
                serialized = json.dumps(
                    {
                        "schema_version": PAIR_ASSESSMENT_SCHEMA_VERSION,
                        "task": PAIR_ASSESSMENT_TASK,
                        "input": result.pair_input.to_dict(),
                        "assessment": result.assessment.to_dict(),
                    },
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                stream.write(f"{serialized}\n")
        assessments_hash = sha256_file(assessment_path)
        collection_id = content_sha256(
            {
                "schema_version": PAIR_ASSESSMENT_SCHEMA_VERSION,
                "task": PAIR_ASSESSMENT_TASK,
                "assessment_ids": assessment_ids,
                "input_ids": input_ids,
                "assessments_sha256": assessments_hash,
                "raw_artifacts": raw_artifacts,
                "reuse_lineage": reuse_lineage,
            }
        )
        manifest = {
            "schema_version": PAIR_ASSESSMENT_SCHEMA_VERSION,
            "snapshot_contract_version": SNAPSHOT_CONTRACT_VERSION,
            "artifact_type": "taim-pair-assessment-run",
            "task": PAIR_ASSESSMENT_TASK,
            "assessor_identity_id": assessor_identity_id,
            "execution_provenance": execution_provenance.to_dict(),
            "collection_id": collection_id,
            "assessment_count": len(rows),
            "assessments_sha256": assessments_hash,
            "raw_artifacts": raw_artifacts,
            "reuse_lineage": reuse_lineage,
        }
        (staging / "manifest.json").write_text(
            json.dumps(
                manifest,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_pair_assessment_run(destination)


def _read_json_object(path: Path, *, role: str) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                SchemaValidationError(f"{role} contains non-finite value {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"cannot read {role}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"{role} must be a JSON object")
    return cast(dict[str, object], payload)


def load_pair_assessment_run(directory: str | Path) -> StoredPairAssessmentRun:
    """Load a closed collection and verify every declared byte hash."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    assessment_path = root / "pair-assessments.jsonl"
    manifest = _read_json_object(manifest_path, role="Pair Assessment manifest")
    require_exact_keys(
        manifest,
        {
            "schema_version",
            "snapshot_contract_version",
            "artifact_type",
            "task",
            "assessor_identity_id",
            "execution_provenance",
            "collection_id",
            "assessment_count",
            "assessments_sha256",
            "raw_artifacts",
            "reuse_lineage",
        },
        role="Pair Assessment manifest",
    )
    if manifest["schema_version"] != PAIR_ASSESSMENT_SCHEMA_VERSION:
        raise SchemaValidationError("unsupported Pair Assessment manifest schema_version")
    if manifest["snapshot_contract_version"] != SNAPSHOT_CONTRACT_VERSION:
        raise SchemaValidationError("unsupported Pair Assessment Snapshot contract version")
    if (
        manifest["artifact_type"] != "taim-pair-assessment-run"
        or manifest["task"] != PAIR_ASSESSMENT_TASK
    ):
        raise SchemaValidationError("invalid Pair Assessment manifest type or task")
    require_sha256(manifest["collection_id"], "Pair Assessment collection_id")
    assessor_identity_id = require_sha256(
        manifest["assessor_identity_id"],
        "Pair Assessment assessor_identity_id",
    )
    execution_provenance_value = manifest["execution_provenance"]
    if not isinstance(execution_provenance_value, Mapping):
        raise SchemaValidationError("Pair Assessment execution_provenance must be an object")
    execution_provenance = PairAssessmentExecutionProvenance.from_dict(execution_provenance_value)
    require_sha256(manifest["assessments_sha256"], "Pair Assessment assessments_sha256")
    reuse_lineage_value = manifest["reuse_lineage"]
    if not isinstance(reuse_lineage_value, Mapping):
        raise SchemaValidationError("Pair Assessment reuse_lineage must be an object")
    reuse_lineage = _normalized_reuse_lineage(reuse_lineage_value)
    assessment_count = manifest["assessment_count"]
    if (
        isinstance(assessment_count, bool)
        or not isinstance(assessment_count, int)
        or assessment_count < 1
    ):
        raise SchemaValidationError("Pair Assessment assessment_count must be positive")
    if sha256_file(assessment_path) != manifest["assessments_sha256"]:
        raise SchemaValidationError("Pair Assessment rows hash does not match manifest")
    try:
        lines = assessment_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SchemaValidationError(f"cannot read Pair Assessment rows: {exc}") from exc
    if not lines or any(not line for line in lines):
        raise SchemaValidationError("Pair Assessment rows must be non-empty JSON Lines")
    assessments: list[PairAssessment] = []
    inputs: dict[str, PairAssessmentInput] = {}
    for line in lines:
        try:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    SchemaValidationError(f"Pair Assessment row contains non-finite value {value}")
                ),
            )
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"invalid Pair Assessment row: {exc}") from exc
        if not isinstance(row, Mapping):
            raise SchemaValidationError("Pair Assessment row must be an object")
        require_exact_keys(
            row,
            {"schema_version", "task", "input", "assessment"},
            role="Pair Assessment row",
        )
        if (
            row["schema_version"] != PAIR_ASSESSMENT_SCHEMA_VERSION
            or row["task"] != PAIR_ASSESSMENT_TASK
        ):
            raise SchemaValidationError("Pair Assessment row has invalid schema or task")
        input_payload = row["input"]
        assessment_payload = row["assessment"]
        if not isinstance(input_payload, Mapping) or not isinstance(assessment_payload, Mapping):
            raise SchemaValidationError("Pair Assessment row payloads must be objects")
        pair_input = PairAssessmentInput.from_dict(cast(Mapping[str, object], input_payload))
        assessment = PairAssessment.from_dict(cast(Mapping[str, object], assessment_payload))
        _validate_assessment_entity_lineage(pair_input, assessment)
        inputs[pair_input.pair_assessment_input_id] = pair_input
        assessments.append(assessment)
    row_order = [
        (
            inputs[item.pair_assessment_input_id].patient_version.patient_id,
            inputs[item.pair_assessment_input_id].trial_version.trial_id,
            item.pair_assessment_input_id,
        )
        for item in assessments
    ]
    if row_order != sorted(row_order):
        raise SchemaValidationError("Pair Assessment rows are not deterministically ordered")
    if len(inputs) != len(assessments):
        raise SchemaValidationError("Pair Assessment rows contain duplicate input identities")
    raw_entries = manifest["raw_artifacts"]
    if not isinstance(raw_entries, list) or any(
        not isinstance(item, Mapping) for item in raw_entries
    ):
        raise SchemaValidationError("Pair Assessment raw_artifacts must be an array of objects")
    raw_outputs: dict[str, Mapping[str, object]] = {}
    normalized_raw_entries: list[dict[str, JsonValue]] = []
    expected_raw_paths: set[Path] = set()
    for item in raw_entries:
        entry = cast(Mapping[str, object], item)
        require_exact_keys(
            entry,
            {"assessment_id", "path", "sha256", "byte_length"},
            role="Pair Assessment raw artifact",
        )
        assessment_id = require_sha256(entry["assessment_id"], "raw assessment_id")
        digest = require_sha256(entry["sha256"], "raw artifact sha256")
        path_text = require_non_empty(entry["path"], "raw artifact path")
        byte_length = entry["byte_length"]
        if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length < 1:
            raise SchemaValidationError("raw artifact byte_length must be positive")
        relative = Path(path_text)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("raw",):
            raise SchemaValidationError("raw artifact path must stay under raw/")
        raw_path = root / relative
        expected_raw_paths.add(raw_path.resolve())
        if sha256_file(raw_path) != digest or raw_path.stat().st_size != byte_length:
            raise SchemaValidationError("raw assessor output does not match its manifest")
        raw_payload = _read_json_object(raw_path, role="raw assessor output")
        raw_outputs[assessment_id] = raw_payload
        normalized_raw_entries.append(
            {
                "assessment_id": assessment_id,
                "path": path_text,
                "sha256": digest,
                "byte_length": byte_length,
            }
        )
    actual_raw_paths = {path.resolve() for path in (root / "raw").glob("*.json")}
    if actual_raw_paths != expected_raw_paths:
        raise SchemaValidationError("raw assessor output files do not match the manifest")
    if len(assessments) != assessment_count:
        raise SchemaValidationError("Pair Assessment manifest count does not match rows")
    by_id = {item.assessment_id: item for item in assessments}
    if {item.assessor_identity_id for item in assessments} != {assessor_identity_id}:
        raise SchemaValidationError(
            "Pair Assessment rows do not match the manifest assessor identity"
        )
    if set(by_id) != set(raw_outputs) or len(by_id) != len(assessments):
        raise SchemaValidationError("Pair Assessment and raw output identities do not match")
    reused_assessment_ids = set(cast(list[str], reuse_lineage["reused_assessment_ids"]))
    reused_input_ids = set(cast(list[str], reuse_lineage["reused_pair_assessment_input_ids"]))
    if not reused_assessment_ids <= set(by_id) or not reused_input_ids <= set(inputs):
        raise SchemaValidationError(
            "Pair Assessment reuse lineage references identities outside the collection"
        )
    raw_entry_ids = [cast(str, item["assessment_id"]) for item in normalized_raw_entries]
    if raw_entry_ids != [item.assessment_id for item in assessments]:
        raise SchemaValidationError(
            "Pair Assessment raw artifacts are not in deterministic row order"
        )
    for assessment in assessments:
        loaded_raw_payload = raw_outputs[assessment.assessment_id]
        raw_bytes = _canonical_raw_output_bytes(loaded_raw_payload)
        if assessment.raw_output_provenance.raw_output_sha256 != "sha256:" + hashlib.sha256(
            raw_bytes
        ).hexdigest() or assessment.raw_output_provenance.byte_length != len(raw_bytes):
            raise SchemaValidationError("Pair Assessment raw provenance does not match bytes")
    input_ids = [item.pair_assessment_input_id for item in assessments]
    assessment_ids = [item.assessment_id for item in assessments]
    expected_collection_id = content_sha256(
        {
            "schema_version": PAIR_ASSESSMENT_SCHEMA_VERSION,
            "task": PAIR_ASSESSMENT_TASK,
            "assessment_ids": assessment_ids,
            "input_ids": input_ids,
            "assessments_sha256": manifest["assessments_sha256"],
            "raw_artifacts": normalized_raw_entries,
            "reuse_lineage": reuse_lineage,
        }
    )
    if manifest["collection_id"] != expected_collection_id:
        raise SchemaValidationError("Pair Assessment collection_id does not match its content")
    return StoredPairAssessmentRun(
        directory=root.resolve(),
        assessments=tuple(assessments),
        inputs=inputs,
        raw_outputs=raw_outputs,
        snapshot_contract_version=SNAPSHOT_CONTRACT_VERSION,
        assessor_identity_id=assessor_identity_id,
        execution_provenance=execution_provenance,
        reuse_lineage=reuse_lineage,
        collection_id=expected_collection_id,
        assessments_hash=cast(str, manifest["assessments_sha256"]),
        manifest_hash=sha256_file(manifest_path),
    )


def plan_pair_assessment_reuse(
    pair_inputs: Iterable[PairAssessmentInput],
    *,
    assessor: AssessorIdentity,
    prior: StoredPairAssessmentRun | None,
) -> PairAssessmentReusePlan:
    """Reuse only exact pair inputs produced under the exact assessor identity."""

    inputs = tuple(sorted(pair_inputs, key=lambda item: item.pair_assessment_input_id))
    if len(inputs) != len({item.pair_assessment_input_id for item in inputs}):
        raise SchemaValidationError("Pair Assessment reuse inputs contain duplicates")
    prior_by_input = (
        {
            assessment.pair_assessment_input_id: assessment
            for assessment in prior.assessments
            if assessment.assessor_identity_id == assessor.assessor_identity_id
        }
        if prior is not None
        else {}
    )
    reused = tuple(
        prior_by_input[item.pair_assessment_input_id]
        for item in inputs
        if item.pair_assessment_input_id in prior_by_input
    )
    compute = tuple(item for item in inputs if item.pair_assessment_input_id not in prior_by_input)
    reused_results = (
        tuple(
            PairAssessmentResult(
                pair_input=prior.inputs[assessment.pair_assessment_input_id],
                assessment=assessment,
                raw_output=prior.raw_outputs[assessment.assessment_id],
            )
            for assessment in reused
        )
        if prior is not None
        else ()
    )
    return PairAssessmentReusePlan(
        source_collection_id=prior.collection_id if prior is not None else None,
        reused=reused,
        compute=compute,
        reused_results=reused_results,
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_pair_assessments(
    stored: StoredPairAssessmentRun,
    evaluation_package: PairAssessmentEvaluationPackage,
) -> dict[str, object]:
    """Evaluate closed Pair Assessments without exposing judgments to the assessor."""

    if not isinstance(stored, StoredPairAssessmentRun):
        raise TypeError("stored must be a StoredPairAssessmentRun")
    if not isinstance(evaluation_package, PairAssessmentEvaluationPackage):
        raise TypeError("evaluation_package must be a PairAssessmentEvaluationPackage")
    judgments = {item.pair_assessment_input_id: item for item in evaluation_package.judgments}
    judged: list[tuple[PairAssessment, PairAssessmentJudgment]] = []
    for assessment in stored.assessments:
        judgment = judgments.get(assessment.pair_assessment_input_id)
        if judgment is None:
            continue
        if (
            assessment.patient_version_id != judgment.patient_version_id
            or assessment.trial_version_id != judgment.trial_version_id
        ):
            raise SchemaValidationError(
                "Pair Assessment judgment versions do not match the closed assessment"
            )
        judged.append((assessment, judgment))
    abstained = [item for item in judged if item[0].category == "needs_information"]
    selective = [item for item in judged if item[0].category != "needs_information"]
    correct = sum(
        (judgment.label == 2 and assessment.category == "potential_candidate")
        or (judgment.label in (0, 1) and assessment.category == "not_candidate_for_run")
        for assessment, judgment in selective
    )
    label_counts = {
        label: sum(judgment.label == label for _, judgment in judged) for label in (0, 1, 2)
    }
    label2_false_negatives = sum(
        judgment.label == 2 and assessment.category == "not_candidate_for_run"
        for assessment, judgment in judged
    )
    inappropriate_candidates = sum(
        judgment.label in (0, 1) and assessment.category == "potential_candidate"
        for assessment, judgment in judged
    )
    non_candidate_labels = label_counts[0] + label_counts[1]
    return {
        "schema_version": PAIR_ASSESSMENT_SCHEMA_VERSION,
        "task": PAIR_ASSESSMENT_TASK,
        "collection_id": stored.collection_id,
        "evaluation_package_id": evaluation_package.evaluation_package_id,
        "assessment_count": len(stored.assessments),
        "judged_assessment_count": len(judged),
        "unjudged_assessment_count": len(stored.assessments) - len(judged),
        "missing_assessment_count": len(judgments) - len(judged),
        "coverage": _rate(len(selective), len(judged)),
        "selective_accuracy": _rate(correct, len(selective)),
        "label2_false_negative_rate": _rate(
            label2_false_negatives,
            label_counts[2],
        ),
        "label2_false_negative_count": label2_false_negatives,
        "inappropriate_candidate_rate_labels_0_1": _rate(
            inappropriate_candidates,
            non_candidate_labels,
        ),
        "inappropriate_candidate_rate_by_label": {
            str(label): _rate(
                sum(
                    judgment.label == label and assessment.category == "potential_candidate"
                    for assessment, judgment in judged
                ),
                label_counts[label],
            )
            for label in (0, 1)
        },
        "abstention_rate_by_label": {
            str(label): _rate(
                sum(
                    judgment.label == label and assessment.category == "needs_information"
                    for assessment, judgment in abstained
                ),
                label_counts[label],
            )
            for label in (0, 1, 2)
        },
    }


__all__ = [
    "NEEDS_INFORMATION_TARGETS",
    "PAIR_ASSESSMENT_CATEGORIES",
    "PAIR_ASSESSMENT_TASK",
    "AssessmentEvidence",
    "AssessorIdentity",
    "MissingInformation",
    "PairAssessment",
    "PairAssessmentEvaluationPackage",
    "PairAssessmentExecutionProvenance",
    "PairAssessmentInput",
    "PairAssessmentJudgment",
    "PairAssessmentProfile",
    "PairAssessmentResult",
    "PairAssessmentReusePlan",
    "RawOutputProvenance",
    "StoredPairAssessmentRun",
    "evaluate_pair_assessments",
    "load_pair_assessment_run",
    "make_raw_output_provenance",
    "pair_assessment_collection_lineage",
    "plan_pair_assessment_reuse",
    "write_pair_assessment_run",
]
