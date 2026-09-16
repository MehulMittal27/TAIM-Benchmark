"""Direction-specific trial-to-patient benchmark contracts."""

from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Protocol, cast

from taim.baselines.bm25 import BM25Retriever, bm25_ranking_configuration
from taim.contracts import (
    SNAPSHOT_CONTRACT_VERSION,
    EvaluatorOnlyMaterial,
    canonical_json,
    content_sha256,
    freeze_json_mapping,
    portable_system_input_identity_value,
    require_exact_keys,
    require_non_empty,
    require_sha256,
    system_input_identity_value,
    task_input_membership_id,
)
from taim.data import PreparedBenchmark
from taim.entity_versions import (
    PatientEntityVersion,
    PatientEvidenceProfile,
    TrialVersion,
    clinical_as_of_text,
    parse_clinical_as_of,
    validate_clinical_as_of,
    version_patients,
    version_trials,
)
from taim.evaluation import evaluate_topic
from taim.file_hash import sha256_file
from taim.pair_assessment import (
    StoredPairAssessmentRun,
    pair_assessment_collection_lineage,
)
from taim.schemas import (
    JsonValue,
    SchemaValidationError,
    freeze_json_value,
    json_value_to_builtins,
    normalize_pair_assessment_lineage,
    validate_pipeline_depth,
)
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    DerivedView,
    freeze_system_input_options,
    validate_capability_declarations,
)

TRIAL_TO_PATIENT_SCHEMA_VERSION = "1.0"
TRIAL_TO_PATIENT_RUN_MANIFEST_VERSION = "1.0"
TRIAL_TO_PATIENT_TASK = "trial_to_patient"
TRIAL_TO_PATIENT_PRIMARY_RANKING = "patient_candidates"
TRIAL_TO_PATIENT_BM25_SYSTEM_ID = "bm25-trial-to-patient"
DEFAULT_TRIAL_TO_PATIENT_PROFILE_ID = "trec-ct-2021-trial-to-patient"
REVERSE_TRANSPOSE_COMPLETE_PAIR_MATRIX_V1 = "transpose-patient-trial-pairs-v1"
REVERSE_PRESERVE_JUDGED_PAIRS_V1 = "transpose-judged-pairs-preserve-unjudged-v1"
_REVERSE_STAGE_NAME = re.compile(r"\A[a-z0-9]+(?:[_-][a-z0-9]+)*\Z")


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaValidationError(f"{name} must be a positive integer")
    return value


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be a number")
    if not math.isfinite(value):
        raise SchemaValidationError(f"{name} must be finite")
    return float(value)


def _plain_json_mapping(
    value: Mapping[str, object],
    *,
    path: str,
) -> dict[str, JsonValue]:
    try:
        normalized = json_value_to_builtins(value)
        serialized = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{path} must contain finite JSON values") from exc
    payload = json.loads(serialized)
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"{path} must be a JSON object")
    return cast(dict[str, JsonValue], payload)


@dataclass(frozen=True, slots=True)
class TrialToPatientBenchmarkProfile:
    """Static evaluation and corpus policy for reverse ranking."""

    profile_id: str
    profile_version: str
    cutoffs: tuple[int, ...]
    corpus_policy: str
    query_policy: str
    gain_mapping: Mapping[str, int]
    definition_sha256: str = field(init=False)

    schema_version = TRIAL_TO_PATIENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.profile_id, "trial-to-patient profile_id")
        require_non_empty(self.profile_version, "trial-to-patient profile_version")
        cutoffs = tuple(self.cutoffs)
        if not cutoffs or cutoffs != tuple(sorted(set(cutoffs))):
            raise SchemaValidationError(
                "trial-to-patient profile cutoffs must be sorted and unique"
            )
        for cutoff in cutoffs:
            _require_positive_int(cutoff, "trial-to-patient profile cutoff")
        require_non_empty(self.corpus_policy, "trial-to-patient corpus_policy")
        require_non_empty(self.query_policy, "trial-to-patient query_policy")
        if dict(self.gain_mapping) != {"0": 0, "1": 1, "2": 2}:
            raise SchemaValidationError(
                "trial-to-patient gain_mapping must be {'0': 0, '1': 1, '2': 2}"
            )
        object.__setattr__(self, "cutoffs", cutoffs)
        object.__setattr__(
            self,
            "gain_mapping",
            MappingProxyType(dict(sorted(self.gain_mapping.items()))),
        )
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
            "cutoffs": list(self.cutoffs),
            "corpus_policy": self.corpus_policy,
            "query_policy": self.query_policy,
            "gain_mapping": dict(self.gain_mapping),
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {**self._definition_payload(), "definition_sha256": self.definition_sha256}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> TrialToPatientBenchmarkProfile:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "profile_id",
                "profile_version",
                "cutoffs",
                "corpus_policy",
                "query_policy",
                "gain_mapping",
                "definition_sha256",
            },
            role="TrialToPatientBenchmarkProfile",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported TrialToPatientBenchmarkProfile schema_version")
        cutoffs = payload["cutoffs"]
        gain_mapping = payload["gain_mapping"]
        if not isinstance(cutoffs, list) or not isinstance(gain_mapping, Mapping):
            raise SchemaValidationError(
                "reverse Benchmark Profile cutoffs or gain_mapping is invalid"
            )
        profile = cls(
            profile_id=cast(str, payload["profile_id"]),
            profile_version=cast(str, payload["profile_version"]),
            cutoffs=tuple(cast(list[int], cutoffs)),
            corpus_policy=cast(str, payload["corpus_policy"]),
            query_policy=cast(str, payload["query_policy"]),
            gain_mapping=cast(Mapping[str, int], gain_mapping),
        )
        require_sha256(payload["definition_sha256"], "profile definition_sha256")
        if payload["definition_sha256"] != profile.definition_sha256:
            raise SchemaValidationError(
                "reverse Benchmark Profile identity does not match its content"
            )
        return profile


def default_trial_to_patient_profile() -> TrialToPatientBenchmarkProfile:
    return TrialToPatientBenchmarkProfile(
        profile_id=DEFAULT_TRIAL_TO_PATIENT_PROFILE_ID,
        profile_version="1.0",
        cutoffs=(5, 10, 20),
        corpus_policy="all_profile_visible_patient_versions",
        query_policy="all_profile_visible_trial_versions",
        gain_mapping={"0": 0, "1": 1, "2": 2},
    )


def _pair_coverage_policy(profile: TrialToPatientBenchmarkProfile) -> str:
    if profile.profile_id == "synthetic-trial-to-patient":
        return "preserve_unjudged"
    return "complete_pair_matrix"


@dataclass(frozen=True, slots=True)
class TrialToPatientTaskInput:
    """Judgment-free reverse query and corpus values."""

    benchmark_lineage: str
    prepared_snapshot_id: str
    clinical_as_of: datetime
    patient_evidence_profile: PatientEvidenceProfile
    benchmark_profile: TrialToPatientBenchmarkProfile
    trial_versions: tuple[TrialVersion, ...]
    patient_versions: tuple[PatientEntityVersion, ...]
    available_capabilities: frozenset[str]
    derived_views: tuple[DerivedView, ...] = ()
    query_set_id: str = field(init=False)
    patient_corpus_id: str = field(init=False)
    task_input_id: str = field(init=False)

    schema_version = TRIAL_TO_PATIENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.benchmark_lineage, "benchmark_lineage")
        require_sha256(self.prepared_snapshot_id, "prepared_snapshot_id")
        object.__setattr__(self, "clinical_as_of", validate_clinical_as_of(self.clinical_as_of))
        if not isinstance(self.patient_evidence_profile, PatientEvidenceProfile):
            raise SchemaValidationError("patient_evidence_profile must be a PatientEvidenceProfile")
        if not isinstance(self.benchmark_profile, TrialToPatientBenchmarkProfile):
            raise SchemaValidationError(
                "benchmark_profile must be a TrialToPatientBenchmarkProfile"
            )
        trial_versions = tuple(self.trial_versions)
        patient_versions = tuple(self.patient_versions)
        if not trial_versions or any(
            not isinstance(version, TrialVersion) for version in trial_versions
        ):
            raise SchemaValidationError("trial_versions must contain TrialVersion objects")
        if not patient_versions or any(
            not isinstance(version, PatientEntityVersion) for version in patient_versions
        ):
            raise SchemaValidationError(
                "patient_versions must contain PatientEntityVersion objects"
            )
        trial_ids = [version.trial_id for version in trial_versions]
        patient_ids = [version.patient_id for version in patient_versions]
        if trial_ids != sorted(set(trial_ids)):
            raise SchemaValidationError("trial_versions must use unique deterministic trial order")
        if patient_ids != sorted(set(patient_ids)):
            raise SchemaValidationError(
                "patient_versions must use unique deterministic patient order"
            )
        if any(
            version.patient_evidence_profile != self.patient_evidence_profile
            for version in patient_versions
        ):
            raise SchemaValidationError(
                "patient_versions must share the task Patient Evidence Profile"
            )
        if any(version.clinical_as_of != self.clinical_as_of for version in patient_versions):
            raise SchemaValidationError(
                "patient_versions must share the task clinical_as_of cutoff"
            )
        capabilities = frozenset(self.available_capabilities)
        if any(not isinstance(capability, str) or not capability for capability in capabilities):
            raise SchemaValidationError("available_capabilities must contain non-empty strings")
        required = {
            CAPABILITY_CANONICAL_PATIENT_TEXT,
            CAPABILITY_CANONICAL_TRIAL_TEXT,
        }
        if not required <= capabilities:
            raise SchemaValidationError(
                "trial-to-patient input requires canonical patient and trial text"
            )
        derived_views = tuple(self.derived_views)
        capabilities, derived_views = validate_capability_declarations(
            capabilities,
            derived_views,
            role="trial-to-patient input",
        )
        object.__setattr__(self, "trial_versions", trial_versions)
        object.__setattr__(self, "patient_versions", patient_versions)
        object.__setattr__(self, "available_capabilities", capabilities)
        object.__setattr__(self, "derived_views", derived_views)
        query_set_id = content_sha256(
            {
                "task": TRIAL_TO_PATIENT_TASK,
                "role": "queries",
                "trial_versions": [version.to_dict() for version in trial_versions],
            }
        )
        patient_corpus_id = content_sha256(
            {
                "task": TRIAL_TO_PATIENT_TASK,
                "role": "patient_corpus",
                "patient_evidence_profile": self.patient_evidence_profile.to_dict(),
                "patient_versions": [version.to_dict() for version in patient_versions],
            }
        )
        object.__setattr__(self, "query_set_id", query_set_id)
        object.__setattr__(self, "patient_corpus_id", patient_corpus_id)
        object.__setattr__(
            self,
            "task_input_id",
            content_sha256(self.identity_payload()),
        )

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "task": TRIAL_TO_PATIENT_TASK,
            "benchmark_lineage": self.benchmark_lineage,
            "prepared_snapshot_id": self.prepared_snapshot_id,
            "clinical_as_of": clinical_as_of_text(self.clinical_as_of),
            "patient_evidence_profile": self.patient_evidence_profile.to_dict(),
            "benchmark_profile": self.benchmark_profile.to_dict(),
            "query_set_id": self.query_set_id,
            "patient_corpus_id": self.patient_corpus_id,
            "trial_versions": [version.to_dict() for version in self.trial_versions],
            "patient_versions": [version.to_dict() for version in self.patient_versions],
            "available_capabilities": cast(list[JsonValue], sorted(self.available_capabilities)),
            "derived_views": [view.to_dict() for view in self.derived_views],
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {**self.identity_payload(), "task_input_id": self.task_input_id}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrialToPatientTaskInput:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "task",
                "benchmark_lineage",
                "prepared_snapshot_id",
                "clinical_as_of",
                "patient_evidence_profile",
                "benchmark_profile",
                "query_set_id",
                "patient_corpus_id",
                "trial_versions",
                "patient_versions",
                "available_capabilities",
                "derived_views",
                "task_input_id",
            },
            role="TrialToPatientTaskInput",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported TrialToPatientTaskInput schema_version")
        if payload["task"] != TRIAL_TO_PATIENT_TASK:
            raise SchemaValidationError("TrialToPatientTaskInput task must be 'trial_to_patient'")
        evidence_profile = payload["patient_evidence_profile"]
        benchmark_profile = payload["benchmark_profile"]
        trial_versions = payload["trial_versions"]
        patient_versions = payload["patient_versions"]
        capabilities = payload["available_capabilities"]
        derived_views = payload["derived_views"]
        if not isinstance(evidence_profile, Mapping) or not isinstance(benchmark_profile, Mapping):
            raise SchemaValidationError("reverse task profiles must be objects")
        if not isinstance(trial_versions, list) or any(
            not isinstance(version, Mapping) for version in trial_versions
        ):
            raise SchemaValidationError("reverse task trial_versions must be an array")
        if not isinstance(patient_versions, list) or any(
            not isinstance(version, Mapping) for version in patient_versions
        ):
            raise SchemaValidationError("reverse task patient_versions must be an array")
        if not isinstance(capabilities, list) or any(
            not isinstance(capability, str) for capability in capabilities
        ):
            raise SchemaValidationError(
                "reverse task available_capabilities must be an array of strings"
            )
        if not isinstance(derived_views, list) or any(
            not isinstance(view, Mapping) for view in derived_views
        ):
            raise SchemaValidationError("reverse task derived_views must be an array")
        task_input = cls(
            benchmark_lineage=cast(str, payload["benchmark_lineage"]),
            prepared_snapshot_id=cast(str, payload["prepared_snapshot_id"]),
            clinical_as_of=parse_clinical_as_of(payload["clinical_as_of"]),
            patient_evidence_profile=PatientEvidenceProfile.from_dict(
                cast(Mapping[str, object], evidence_profile)
            ),
            benchmark_profile=TrialToPatientBenchmarkProfile.from_dict(
                cast(Mapping[str, object], benchmark_profile)
            ),
            trial_versions=tuple(
                TrialVersion.from_dict(cast(Mapping[str, object], version))
                for version in trial_versions
            ),
            patient_versions=tuple(
                PatientEntityVersion.from_dict(cast(Mapping[str, object], version))
                for version in patient_versions
            ),
            available_capabilities=frozenset(cast(list[str], capabilities)),
            derived_views=tuple(
                DerivedView.from_dict(cast(Mapping[str, object], view)) for view in derived_views
            ),
        )
        for name in ("query_set_id", "patient_corpus_id", "task_input_id"):
            require_sha256(payload[name], name)
            if payload[name] != getattr(task_input, name):
                raise SchemaValidationError(f"{name} does not match reverse task input content")
        return task_input

    @classmethod
    def from_json(cls, serialized: str) -> TrialToPatientTaskInput:
        try:
            payload = json.loads(serialized)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SchemaValidationError(f"invalid TrialToPatientTaskInput JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SchemaValidationError("TrialToPatientTaskInput must be a JSON object")
        return cls.from_dict(cast(Mapping[str, object], payload))


@dataclass(frozen=True, slots=True)
class TrialToPatientJudgment:
    trial_id: str
    trial_version_id: str
    patient_id: str
    patient_version_id: str
    label: int

    schema_version = TRIAL_TO_PATIENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.trial_id, "judgment trial_id")
        require_sha256(self.trial_version_id, "judgment trial_version_id")
        require_non_empty(self.patient_id, "judgment patient_id")
        require_sha256(self.patient_version_id, "judgment patient_version_id")
        if isinstance(self.label, bool) or self.label not in (0, 1, 2):
            raise SchemaValidationError("judgment label must be 0, 1, or 2")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "trial_version_id": self.trial_version_id,
            "patient_id": self.patient_id,
            "patient_version_id": self.patient_version_id,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrialToPatientJudgment:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "trial_id",
                "trial_version_id",
                "patient_id",
                "patient_version_id",
                "label",
            },
            role="TrialToPatientJudgment",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported TrialToPatientJudgment schema_version")
        return cls(
            trial_id=cast(str, payload["trial_id"]),
            trial_version_id=cast(str, payload["trial_version_id"]),
            patient_id=cast(str, payload["patient_id"]),
            patient_version_id=cast(str, payload["patient_version_id"]),
            label=cast(int, payload["label"]),
        )


@dataclass(frozen=True, slots=True)
class TrialToPatientEvaluationPackage(EvaluatorOnlyMaterial):
    benchmark_lineage: str
    prepared_snapshot_id: str
    query_set_id: str
    patient_corpus_id: str
    benchmark_profile_definition_sha256: str
    judgments: tuple[TrialToPatientJudgment, ...]
    provenance: Mapping[str, JsonValue]
    evaluation_package_id: str = field(init=False)

    schema_version = TRIAL_TO_PATIENT_SCHEMA_VERSION
    task = TRIAL_TO_PATIENT_TASK

    def __post_init__(self) -> None:
        require_non_empty(self.benchmark_lineage, "evaluation benchmark_lineage")
        for name in (
            "prepared_snapshot_id",
            "query_set_id",
            "patient_corpus_id",
            "benchmark_profile_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        judgments = tuple(self.judgments)
        if not judgments or any(
            not isinstance(judgment, TrialToPatientJudgment) for judgment in judgments
        ):
            raise SchemaValidationError(
                "reverse Evaluation Package must contain TrialToPatientJudgment objects"
            )
        keys = [(judgment.trial_version_id, judgment.patient_version_id) for judgment in judgments]
        if len(keys) != len(set(keys)):
            raise SchemaValidationError("reverse Evaluation Package contains duplicate pairs")
        ordered = tuple(
            sorted(
                judgments,
                key=lambda row: (row.trial_id, row.patient_id),
            )
        )
        provenance = freeze_json_mapping(
            self.provenance,
            path="reverse Evaluation Package provenance",
        )
        object.__setattr__(self, "judgments", ordered)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "evaluation_package_id",
            content_sha256(self.identity_payload()),
        )

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "benchmark_lineage": self.benchmark_lineage,
            "prepared_snapshot_id": self.prepared_snapshot_id,
            "query_set_id": self.query_set_id,
            "patient_corpus_id": self.patient_corpus_id,
            "benchmark_profile_definition_sha256": (self.benchmark_profile_definition_sha256),
            "judgments": [judgment.to_dict() for judgment in self.judgments],
            "provenance": json_value_to_builtins(self.provenance),
        }

    def manifest_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "benchmark_lineage": self.benchmark_lineage,
            "prepared_snapshot_id": self.prepared_snapshot_id,
            "query_set_id": self.query_set_id,
            "patient_corpus_id": self.patient_corpus_id,
            "benchmark_profile_definition_sha256": (self.benchmark_profile_definition_sha256),
            "judgment_count": len(self.judgments),
            "evaluation_package_id": self.evaluation_package_id,
            "provenance": json_value_to_builtins(self.provenance),
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            **self.identity_payload(),
            "evaluation_package_id": self.evaluation_package_id,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> TrialToPatientEvaluationPackage:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "task",
                "benchmark_lineage",
                "prepared_snapshot_id",
                "query_set_id",
                "patient_corpus_id",
                "benchmark_profile_definition_sha256",
                "judgments",
                "provenance",
                "evaluation_package_id",
            },
            role="TrialToPatientEvaluationPackage",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError(
                "unsupported TrialToPatientEvaluationPackage schema_version"
            )
        if payload["task"] != TRIAL_TO_PATIENT_TASK:
            raise SchemaValidationError(
                "TrialToPatientEvaluationPackage task must be 'trial_to_patient'"
            )
        judgments = payload["judgments"]
        provenance = payload["provenance"]
        if not isinstance(judgments, list) or any(
            not isinstance(judgment, Mapping) for judgment in judgments
        ):
            raise SchemaValidationError("reverse Evaluation Package judgments must be an array")
        if not isinstance(provenance, Mapping):
            raise SchemaValidationError("reverse Evaluation Package provenance must be an object")
        package = cls(
            benchmark_lineage=cast(str, payload["benchmark_lineage"]),
            prepared_snapshot_id=cast(str, payload["prepared_snapshot_id"]),
            query_set_id=cast(str, payload["query_set_id"]),
            patient_corpus_id=cast(str, payload["patient_corpus_id"]),
            benchmark_profile_definition_sha256=cast(
                str,
                payload["benchmark_profile_definition_sha256"],
            ),
            judgments=tuple(
                TrialToPatientJudgment.from_dict(cast(Mapping[str, object], judgment))
                for judgment in judgments
            ),
            provenance=cast(Mapping[str, JsonValue], provenance),
        )
        require_sha256(payload["evaluation_package_id"], "evaluation_package_id")
        if payload["evaluation_package_id"] != package.evaluation_package_id:
            raise SchemaValidationError(
                "reverse Evaluation Package identity does not match its content"
            )
        return package

    @classmethod
    def from_json(cls, serialized: str) -> TrialToPatientEvaluationPackage:
        try:
            payload = json.loads(serialized)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SchemaValidationError(
                f"invalid TrialToPatientEvaluationPackage JSON: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise SchemaValidationError("TrialToPatientEvaluationPackage must be a JSON object")
        return cls.from_dict(cast(Mapping[str, object], payload))


def _validate_reverse_judgment_membership(
    judgments: tuple[TrialToPatientJudgment, ...],
    *,
    query_trial_versions: tuple[tuple[str, str], ...],
    patient_corpus_versions: tuple[tuple[str, str], ...],
    boundary: str,
) -> None:
    query_versions = set(query_trial_versions)
    patient_versions = set(patient_corpus_versions)
    if any(
        (judgment.trial_id, judgment.trial_version_id) not in query_versions
        or (judgment.patient_id, judgment.patient_version_id) not in patient_versions
        for judgment in judgments
    ):
        raise SchemaValidationError(
            f"reverse Evaluation Package contains a Judgment outside the {boundary}"
        )


def _validate_reverse_judgment_closure(
    judgments: tuple[TrialToPatientJudgment, ...],
    *,
    query_trial_versions: tuple[tuple[str, str], ...],
    patient_corpus_versions: tuple[tuple[str, str], ...],
    boundary: str,
) -> None:
    """Require an explicit Judgment for every reverse query-corpus pair."""

    expected_pairs = {
        (trial_version_id, patient_version_id)
        for _trial_id, trial_version_id in query_trial_versions
        for _patient_id, patient_version_id in patient_corpus_versions
    }
    actual_pairs = {
        (judgment.trial_version_id, judgment.patient_version_id) for judgment in judgments
    }
    if actual_pairs != expected_pairs:
        missing_count = len(expected_pairs - actual_pairs)
        unexpected_count = len(actual_pairs - expected_pairs)
        raise SchemaValidationError(
            f"reverse Evaluation Package is not a closed pair matrix at the {boundary}; "
            f"missing_pairs={missing_count}, unexpected_pairs={unexpected_count}"
        )


@dataclass(frozen=True, slots=True)
class ResolvedTrialToPatientBenchmark:
    prepared: PreparedBenchmark
    task_input: TrialToPatientTaskInput
    profile: TrialToPatientBenchmarkProfile
    evaluation_package: TrialToPatientEvaluationPackage

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedBenchmark):
            raise TypeError("prepared must be a PreparedBenchmark")
        if self.task_input.benchmark_profile != self.profile:
            raise SchemaValidationError("resolved reverse profile does not match its task input")
        package = self.evaluation_package
        if not isinstance(package, TrialToPatientEvaluationPackage):
            raise TypeError("evaluation_package must be a TrialToPatientEvaluationPackage")
        if package.benchmark_lineage != self.task_input.benchmark_lineage:
            raise SchemaValidationError(
                "reverse Evaluation Package benchmark lineage does not match its task input"
            )
        if (
            package.prepared_snapshot_id != self.task_input.prepared_snapshot_id
            or package.query_set_id != self.task_input.query_set_id
            or package.patient_corpus_id != self.task_input.patient_corpus_id
            or package.benchmark_profile_definition_sha256 != self.profile.definition_sha256
        ):
            raise SchemaValidationError(
                "reverse Evaluation Package does not match the task query, corpus, "
                "and Benchmark Profile"
            )
        _validate_reverse_judgment_membership(
            package.judgments,
            query_trial_versions=tuple(
                (version.trial_id, version.trial_version_id)
                for version in self.task_input.trial_versions
            ),
            patient_corpus_versions=tuple(
                (version.patient_id, version.patient_version_id)
                for version in self.task_input.patient_versions
            ),
            boundary="reverse task input",
        )
        if _pair_coverage_policy(self.profile) == "complete_pair_matrix":
            _validate_reverse_judgment_closure(
                package.judgments,
                query_trial_versions=tuple(
                    (version.trial_id, version.trial_version_id)
                    for version in self.task_input.trial_versions
                ),
                patient_corpus_versions=tuple(
                    (version.patient_id, version.patient_version_id)
                    for version in self.task_input.patient_versions
                ),
                boundary="reverse task input",
            )


def resolve_trial_to_patient_benchmark(
    prepared: PreparedBenchmark,
    *,
    clinical_as_of: datetime,
    patient_evidence_profile: PatientEvidenceProfile,
    transposition_policy: str,
    profile: TrialToPatientBenchmarkProfile | None = None,
) -> ResolvedTrialToPatientBenchmark:
    """Transpose evaluator pairs around trial queries without exposing them to Systems."""

    if not isinstance(prepared, PreparedBenchmark):
        raise TypeError("prepared must be a PreparedBenchmark")
    if transposition_policy not in {
        REVERSE_TRANSPOSE_COMPLETE_PAIR_MATRIX_V1,
        REVERSE_PRESERVE_JUDGED_PAIRS_V1,
    }:
        raise SchemaValidationError(
            "reverse construction requires an explicit complete-pair-matrix or "
            "preserve-judged-pairs transposition policy"
        )
    selected_profile = profile or default_trial_to_patient_profile()
    expected_policy = (
        "complete_pair_matrix"
        if transposition_policy == REVERSE_TRANSPOSE_COMPLETE_PAIR_MATRIX_V1
        else "preserve_unjudged"
    )
    if _pair_coverage_policy(selected_profile) != expected_policy:
        raise SchemaValidationError(
            "reverse transposition policy does not match the Benchmark Profile judgment policy"
        )
    patient_versions = version_patients(
        prepared.topics,
        patient_evidence_profile,
        clinical_as_of=clinical_as_of,
        derived_views=prepared.snapshot.derived_views,
    )
    trial_versions = version_trials(
        prepared.trials,
        derived_views=prepared.snapshot.derived_views,
    )
    task_input = TrialToPatientTaskInput(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        clinical_as_of=clinical_as_of,
        patient_evidence_profile=patient_evidence_profile,
        benchmark_profile=selected_profile,
        trial_versions=trial_versions,
        patient_versions=patient_versions,
        available_capabilities=prepared.snapshot.available_capabilities,
        derived_views=prepared.snapshot.derived_views,
    )
    patient_by_id = {version.patient_id: version for version in patient_versions}
    trial_by_id = {version.trial_id: version for version in trial_versions}
    reverse_judgments: list[TrialToPatientJudgment] = []
    for judgment in prepared.evaluation_package.judgments:
        try:
            patient = patient_by_id[judgment.topic_id]
            trial = trial_by_id[judgment.trial_id]
        except KeyError as exc:
            raise SchemaValidationError(
                "source Evaluation Package references an entity outside the reverse task input"
            ) from exc
        reverse_judgments.append(
            TrialToPatientJudgment(
                trial_id=trial.trial_id,
                trial_version_id=trial.trial_version_id,
                patient_id=patient.patient_id,
                patient_version_id=patient.patient_version_id,
                label=judgment.label,
            )
        )
    evaluation_package = TrialToPatientEvaluationPackage(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        query_set_id=task_input.query_set_id,
        patient_corpus_id=task_input.patient_corpus_id,
        benchmark_profile_definition_sha256=selected_profile.definition_sha256,
        judgments=tuple(reverse_judgments),
        provenance={
            "mapping": transposition_policy,
            "source_evaluation_package_id": (prepared.evaluation_package.evaluation_package_id),
        },
    )
    return ResolvedTrialToPatientBenchmark(
        prepared=prepared,
        task_input=task_input,
        profile=selected_profile,
        evaluation_package=evaluation_package,
    )


@dataclass(frozen=True, slots=True)
class TrialToPatientRunRequest:
    """One reverse System invocation with no evaluator material."""

    task_input: TrialToPatientTaskInput
    run_id: str
    top_k: int
    metric_cutoff: int
    options: Mapping[str, object]
    required_capabilities: frozenset[str] = frozenset()
    optional_capabilities: frozenset[str] = frozenset()
    identity_version: str = "1.0"
    system_input_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.task_input, TrialToPatientTaskInput):
            raise TypeError("task_input must be a TrialToPatientTaskInput")
        require_non_empty(self.run_id, "run_id")
        if self.identity_version not in {"1.0", "2.0"}:
            raise SchemaValidationError(
                "reverse System Input identity_version must be '1.0' or '2.0'"
            )
        _require_positive_int(self.top_k, "top_k")
        _require_positive_int(self.metric_cutoff, "metric_cutoff")
        if self.metric_cutoff not in self.task_input.benchmark_profile.cutoffs:
            raise SchemaValidationError(
                "metric_cutoff must be one of the reverse Benchmark Profile cutoffs"
            )
        required = frozenset(self.required_capabilities)
        optional = frozenset(self.optional_capabilities)
        if required & optional:
            raise SchemaValidationError(
                "required and optional reverse System capabilities must not overlap"
            )
        if any(not isinstance(item, str) or not item for item in required | optional):
            raise SchemaValidationError("reverse System capabilities must be non-empty strings")
        missing = required - self.task_input.available_capabilities
        if missing:
            raise SchemaValidationError(
                "reverse task input is missing required capabilities: " + ", ".join(sorted(missing))
            )
        options = freeze_system_input_options(
            self.options,
            path="TrialToPatientRunRequest options",
        )
        object.__setattr__(self, "required_capabilities", required)
        object.__setattr__(self, "optional_capabilities", optional)
        object.__setattr__(self, "options", options)
        identity_options = system_input_identity_value(options, path="options")
        identity_payload: dict[str, JsonValue] = {
            "task_input_id": self.task_input.task_input_id,
            "top_k": self.top_k,
            "metric_cutoff": self.metric_cutoff,
            "options": identity_options,
            "required_capabilities": cast(list[JsonValue], sorted(required)),
            "optional_capabilities": cast(list[JsonValue], sorted(optional)),
        }
        if self.identity_version == "2.0":
            identity_options, _omitted = portable_system_input_identity_value(
                options,
                path="options",
            )
            identity_payload.update(
                {
                    "identity_version": self.identity_version,
                    "task_input_membership_id": task_input_membership_id(
                        task=TRIAL_TO_PATIENT_TASK,
                        patient_entity_versions=(
                            (version.patient_id, version.patient_version_id)
                            for version in self.task_input.patient_versions
                        ),
                        trial_entity_versions=(
                            (version.trial_id, version.trial_version_id)
                            for version in self.task_input.trial_versions
                        ),
                    ),
                    "options": identity_options,
                }
            )
        object.__setattr__(self, "system_input_id", content_sha256(identity_payload))

    def system_input_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            **self.task_input.to_dict(),
            "system_input_id": self.system_input_id,
            "top_k": self.top_k,
            "metric_cutoff": self.metric_cutoff,
            "options": system_input_identity_value(self.options, path="options"),
            "required_capabilities": cast(list[JsonValue], sorted(self.required_capabilities)),
            "optional_capabilities": cast(list[JsonValue], sorted(self.optional_capabilities)),
        }
        if self.identity_version == "2.0":
            payload["identity_version"] = self.identity_version
        return payload

    @classmethod
    def for_benchmark(
        cls,
        benchmark: ResolvedTrialToPatientBenchmark,
        *,
        run_id: str,
        top_k: int,
        metric_cutoff: int,
        options: Mapping[str, object],
        required_capabilities: Iterable[str] = (),
        optional_capabilities: Iterable[str] = (),
        identity_version: str = "1.0",
    ) -> TrialToPatientRunRequest:
        if not isinstance(benchmark, ResolvedTrialToPatientBenchmark):
            raise TypeError("benchmark must be a resolved trial-to-patient benchmark")
        return cls(
            task_input=benchmark.task_input,
            run_id=run_id,
            top_k=top_k,
            metric_cutoff=metric_cutoff,
            options=options,
            required_capabilities=frozenset(required_capabilities),
            optional_capabilities=frozenset(optional_capabilities),
            identity_version=identity_version,
        )


@dataclass(frozen=True, slots=True)
class TrialToPatientCandidate:
    run_id: str
    system_id: str
    trial_id: str
    trial_version_id: str
    patient_id: str
    patient_version_id: str
    rank: int
    score: float

    schema_version = TRIAL_TO_PATIENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "system_id", "trial_id", "patient_id"):
            require_non_empty(getattr(self, name), name)
        require_sha256(self.trial_version_id, "trial_version_id")
        require_sha256(self.patient_version_id, "patient_version_id")
        _require_positive_int(self.rank, "rank")
        object.__setattr__(self, "score", _require_finite_number(self.score, "score"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "system_id": self.system_id,
            "trial_id": self.trial_id,
            "trial_version_id": self.trial_version_id,
            "patient_id": self.patient_id,
            "patient_version_id": self.patient_version_id,
            "rank": self.rank,
            "score": self.score,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrialToPatientCandidate:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "run_id",
                "system_id",
                "trial_id",
                "trial_version_id",
                "patient_id",
                "patient_version_id",
                "rank",
                "score",
            },
            role="TrialToPatientCandidate",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported TrialToPatientCandidate schema_version")
        return cls(
            run_id=cast(str, payload["run_id"]),
            system_id=cast(str, payload["system_id"]),
            trial_id=cast(str, payload["trial_id"]),
            trial_version_id=cast(str, payload["trial_version_id"]),
            patient_id=cast(str, payload["patient_id"]),
            patient_version_id=cast(str, payload["patient_version_id"]),
            rank=cast(int, payload["rank"]),
            score=cast(float, payload["score"]),
        )

    @classmethod
    def from_json(cls, serialized: str) -> TrialToPatientCandidate:
        try:
            payload = json.loads(serialized)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SchemaValidationError(f"invalid TrialToPatientCandidate JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SchemaValidationError("TrialToPatientCandidate must be a JSON object")
        return cls.from_dict(cast(Mapping[str, object], payload))


@dataclass(frozen=True, slots=True)
class TrialToPatientStageRanking:
    """Optional direction-specific diagnostic ranking."""

    name: str
    pipeline_depth: str
    candidates: tuple[TrialToPatientCandidate, ...] = ()
    artifact_hash: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or self.name == TRIAL_TO_PATIENT_PRIMARY_RANKING
            or _REVERSE_STAGE_NAME.fullmatch(self.name) is None
        ):
            raise SchemaValidationError(
                "reverse stage name must be a lowercase slug other than patient_candidates"
            )
        object.__setattr__(
            self,
            "pipeline_depth",
            validate_pipeline_depth(self.pipeline_depth),
        )
        candidates = tuple(self.candidates)
        if any(not isinstance(item, TrialToPatientCandidate) for item in candidates):
            raise SchemaValidationError(
                "reverse stage candidates must be TrialToPatientCandidate objects"
            )
        object.__setattr__(self, "candidates", candidates)
        if self.artifact_hash is not None:
            require_sha256(self.artifact_hash, "reverse stage artifact_hash")

    def manifest_dict(self) -> dict[str, JsonValue]:
        if self.artifact_hash is None:
            raise SchemaValidationError("reverse stage artifact_hash is required in a manifest")
        return {
            "name": self.name,
            "pipeline_depth": self.pipeline_depth,
            "artifact_hash": self.artifact_hash,
        }

    @classmethod
    def from_manifest_dict(
        cls,
        payload: Mapping[str, object],
    ) -> TrialToPatientStageRanking:
        require_exact_keys(
            payload,
            {"name", "pipeline_depth", "artifact_hash"},
            role="TrialToPatientStageRanking",
        )
        return cls(
            name=cast(str, payload["name"]),
            pipeline_depth=cast(str, payload["pipeline_depth"]),
            artifact_hash=cast(str, payload["artifact_hash"]),
        )


@dataclass(frozen=True, slots=True)
class TrialToPatientRunResult:
    candidates: tuple[TrialToPatientCandidate, ...]
    system_id: str
    system_input_id: str
    configuration: Mapping[str, object]
    runtime_seconds: float
    primary_ranking: str = TRIAL_TO_PATIENT_PRIMARY_RANKING
    pipeline_depth: str = "retrieval"
    stage_rankings: tuple[TrialToPatientStageRanking, ...] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.system_id, "reverse result system_id")
        require_sha256(self.system_input_id, "reverse result system_input_id")
        candidates = tuple(self.candidates)
        if not candidates:
            raise SchemaValidationError("reverse Primary Ranking must not be empty")
        if any(not isinstance(row, TrialToPatientCandidate) for row in candidates):
            raise SchemaValidationError(
                "reverse run candidates must be TrialToPatientCandidate objects"
            )
        if {row.system_id for row in candidates} != {self.system_id}:
            raise SchemaValidationError(
                "reverse run candidates do not match the result System identity"
            )
        if self.primary_ranking != TRIAL_TO_PATIENT_PRIMARY_RANKING:
            raise SchemaValidationError(
                f"reverse primary_ranking must be {TRIAL_TO_PATIENT_PRIMARY_RANKING!r}"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "configuration",
            freeze_json_value(
                _plain_json_mapping(self.configuration, path="reverse configuration")
            ),
        )
        object.__setattr__(
            self,
            "runtime_seconds",
            _require_finite_number(self.runtime_seconds, "runtime_seconds"),
        )
        if self.runtime_seconds < 0:
            raise SchemaValidationError("runtime_seconds must not be negative")
        object.__setattr__(self, "pipeline_depth", validate_pipeline_depth(self.pipeline_depth))
        stage_rankings = tuple(self.stage_rankings)
        if any(not isinstance(item, TrialToPatientStageRanking) for item in stage_rankings):
            raise SchemaValidationError(
                "reverse stage_rankings must contain TrialToPatientStageRanking objects"
            )
        names = [item.name for item in stage_rankings]
        if len(names) != len(set(names)):
            raise SchemaValidationError("reverse stage ranking names must be unique")
        object.__setattr__(self, "stage_rankings", stage_rankings)


class TrialToPatientSystem(Protocol):
    system_id: str
    option_names: frozenset[str]
    required_capabilities: frozenset[str]
    optional_capabilities: frozenset[str]

    def run(self, request: TrialToPatientRunRequest) -> TrialToPatientRunResult: ...


def _reverse_option(request: TrialToPatientRunRequest, name: str) -> object:
    try:
        return request.options[name]
    except KeyError as exc:
        raise ValueError(f"{request.run_id} is missing reverse System option {name!r}") from exc


class TrialToPatientBM25System:
    system_id = TRIAL_TO_PATIENT_BM25_SYSTEM_ID
    option_names = frozenset({"b", "k1"})
    required_capabilities = frozenset(
        {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
    )
    optional_capabilities: frozenset[str] = frozenset()

    def run(self, request: TrialToPatientRunRequest) -> TrialToPatientRunResult:
        started_at = time.perf_counter()
        index = BM25Retriever(
            request.task_input.patient_versions,
            document_id=lambda patient: patient.patient_id,
            document_text=lambda patient: patient.topic.canonical_text,
            k1=cast(float, _reverse_option(request, "k1")),
            b=cast(float, _reverse_option(request, "b")),
        )
        candidates: list[TrialToPatientCandidate] = []
        for trial_version in request.task_input.trial_versions:
            ranked = index.rank(
                trial_version.trial.canonical_text,
                top_k=request.top_k,
            )
            candidates.extend(
                TrialToPatientCandidate(
                    run_id=request.run_id,
                    system_id=self.system_id,
                    trial_id=trial_version.trial_id,
                    trial_version_id=trial_version.trial_version_id,
                    patient_id=patient.patient_id,
                    patient_version_id=patient.patient_version_id,
                    rank=rank,
                    score=score,
                )
                for rank, (patient, score) in enumerate(ranked, start=1)
            )
        runtime_seconds = time.perf_counter() - started_at
        ranking = bm25_ranking_configuration(
            k1=index.k1,
            b=index.b,
            requested_top_k=request.top_k,
            document_count=len(index.documents),
        )
        return TrialToPatientRunResult(
            candidates=tuple(candidates),
            system_id=self.system_id,
            system_input_id=request.system_input_id,
            configuration={
                "implementation": ("taim.trial_to_patient.TrialToPatientBM25System"),
                "system_identity": {"system_id": self.system_id},
                "model_identity": {
                    "kind": "none",
                    "reason": "deterministic_bm25",
                },
                "index_identity": {
                    "implementation": ranking["retrieval_formula"]["id"],
                    "persistent": False,
                },
                "retrieval_formula": ranking["retrieval_formula"],
                "parameters": ranking["parameters"],
                "requested_top_k": ranking["requested_top_k"],
                "effective_top_k": ranking["effective_top_k"],
                "metric_cutoff": request.metric_cutoff,
                "score_tie_breaking": "score descending, patient_id ascending",
                "indexed_fields": ["PatientEntityVersion.topic.canonical_text"],
                "query_fields": ["TrialVersion.trial.canonical_text"],
                "corpus_statistics": index.corpus_statistics,
                "runtime_seconds": runtime_seconds,
            },
            runtime_seconds=runtime_seconds,
        )


TRIAL_TO_PATIENT_SYSTEMS: dict[str, TrialToPatientSystem] = {
    TRIAL_TO_PATIENT_BM25_SYSTEM_ID: TrialToPatientBM25System()
}


def _effective_trial_to_patient_request(
    system_id: str,
    request: TrialToPatientRunRequest,
) -> tuple[TrialToPatientSystem, TrialToPatientRunRequest]:
    try:
        system = TRIAL_TO_PATIENT_SYSTEMS[system_id]
    except KeyError as exc:
        known = ", ".join(sorted(TRIAL_TO_PATIENT_SYSTEMS))
        raise ValueError(
            f"unknown trial-to-patient System {system_id!r}; available Systems: {known}"
        ) from exc
    required = request.required_capabilities | system.required_capabilities
    optional = (request.optional_capabilities | system.optional_capabilities) - required
    filtered_options = {
        name: request.options[name] for name in system.option_names if name in request.options
    }
    system_request = replace(
        request,
        options=MappingProxyType(filtered_options),
        required_capabilities=required,
        optional_capabilities=optional,
    )
    return system, system_request


def run_trial_to_patient_system(
    system_id: str,
    request: TrialToPatientRunRequest,
) -> TrialToPatientRunResult:
    system, system_request = _effective_trial_to_patient_request(system_id, request)
    result = system.run(system_request)
    if not isinstance(result, TrialToPatientRunResult):
        raise TypeError(
            f"trial-to-patient System {system_id!r} must return TrialToPatientRunResult"
        )
    if any(row.run_id != request.run_id or row.system_id != system_id for row in result.candidates):
        raise SchemaValidationError(
            "trial-to-patient System candidates do not match the invocation"
        )
    if result.system_input_id != system_request.system_input_id:
        raise SchemaValidationError(
            "trial-to-patient System result does not bind the effective System Input"
        )
    _validate_reverse_candidates(result.candidates, request=system_request, manifest=None)
    for stage in result.stage_rankings:
        if not stage.candidates:
            continue
        if any(
            row.run_id != request.run_id or row.system_id != system_id for row in stage.candidates
        ):
            raise SchemaValidationError(
                "trial-to-patient stage candidates do not match the invocation"
            )
        _validate_reverse_candidates(
            stage.candidates,
            request=system_request,
            manifest=None,
            require_complete=False,
        )
    return result


@dataclass(frozen=True, slots=True)
class TrialToPatientExecutionProvenance:
    """Mandatory execution state needed to reproduce one reverse run."""

    git_commit: str
    working_tree_dirty: bool
    environment: Mapping[str, object]
    seeds: Mapping[str, object]
    model_identity: Mapping[str, object]
    index_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        require_non_empty(self.git_commit, "execution provenance git_commit")
        if not isinstance(self.working_tree_dirty, bool):
            raise SchemaValidationError("execution provenance working_tree_dirty must be a boolean")
        for name in ("environment", "seeds", "model_identity", "index_identity"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise SchemaValidationError(
                    f"execution provenance {name} must be a non-empty object"
                )
            object.__setattr__(
                self,
                name,
                MappingProxyType(_plain_json_mapping(value, path=f"execution provenance {name}")),
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "git_commit": self.git_commit,
            "working_tree_dirty": self.working_tree_dirty,
            "environment": json_value_to_builtins(self.environment),
            "seeds": json_value_to_builtins(self.seeds),
            "model_identity": json_value_to_builtins(self.model_identity),
            "index_identity": json_value_to_builtins(self.index_identity),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> TrialToPatientExecutionProvenance:
        require_exact_keys(
            payload,
            {
                "git_commit",
                "working_tree_dirty",
                "environment",
                "seeds",
                "model_identity",
                "index_identity",
            },
            role="TrialToPatientExecutionProvenance",
        )
        nested_names = ("environment", "seeds", "model_identity", "index_identity")
        if any(not isinstance(payload[name], Mapping) for name in nested_names):
            raise SchemaValidationError(
                "reverse execution provenance nested fields must be objects"
            )
        return cls(
            git_commit=cast(str, payload["git_commit"]),
            working_tree_dirty=cast(bool, payload["working_tree_dirty"]),
            environment=cast(Mapping[str, object], payload["environment"]),
            seeds=cast(Mapping[str, object], payload["seeds"]),
            model_identity=cast(Mapping[str, object], payload["model_identity"]),
            index_identity=cast(Mapping[str, object], payload["index_identity"]),
        )


@dataclass(frozen=True, slots=True)
class TrialToPatientRunManifest:
    run_id: str
    system_id: str
    created_at: datetime
    execution_provenance: TrialToPatientExecutionProvenance
    configuration: Mapping[str, object]
    runtime_seconds: float
    benchmark_lineage: str
    prepared_snapshot_id: str
    task_input_id: str
    system_input_id: str
    query_set_id: str
    patient_corpus_id: str
    clinical_as_of: datetime
    patient_evidence_profile: Mapping[str, object]
    benchmark_profile: Mapping[str, object]
    evaluation_package_id: str
    query_trial_versions: tuple[tuple[str, str], ...]
    patient_corpus_versions: tuple[tuple[str, str], ...]
    primary_ranking: str
    pipeline_depth: str
    budget_k: int
    metric_cutoff: int
    candidates_sha256: str
    stage_rankings: tuple[TrialToPatientStageRanking, ...] = ()
    pair_assessment_references: tuple[str, ...] = ()
    reuse_lineage: Mapping[str, object] = field(
        default_factory=lambda: {"pair_assessment_collections": []}
    )
    task: str = field(default=TRIAL_TO_PATIENT_TASK, init=False)
    snapshot_contract_version: str = field(
        default=SNAPSHOT_CONTRACT_VERSION,
        init=False,
    )

    schema_version: ClassVar[str] = TRIAL_TO_PATIENT_RUN_MANIFEST_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "system_id", "benchmark_lineage"):
            require_non_empty(getattr(self, name), name)
        if not isinstance(self.execution_provenance, TrialToPatientExecutionProvenance):
            raise SchemaValidationError(
                "execution_provenance must be TrialToPatientExecutionProvenance"
            )
        for name in (
            "prepared_snapshot_id",
            "task_input_id",
            "system_input_id",
            "query_set_id",
            "patient_corpus_id",
            "evaluation_package_id",
            "candidates_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if not isinstance(self.created_at, datetime):
            raise SchemaValidationError("created_at must be a datetime")
        object.__setattr__(self, "created_at", validate_clinical_as_of(self.created_at))
        object.__setattr__(self, "clinical_as_of", validate_clinical_as_of(self.clinical_as_of))
        object.__setattr__(
            self,
            "configuration",
            MappingProxyType(
                _plain_json_mapping(self.configuration, path="manifest configuration")
            ),
        )
        object.__setattr__(
            self,
            "runtime_seconds",
            _require_finite_number(self.runtime_seconds, "manifest runtime_seconds"),
        )
        if self.runtime_seconds < 0:
            raise SchemaValidationError("manifest runtime_seconds must not be negative")
        object.__setattr__(
            self,
            "patient_evidence_profile",
            MappingProxyType(
                _plain_json_mapping(
                    self.patient_evidence_profile,
                    path="manifest Patient Evidence Profile",
                )
            ),
        )
        benchmark_profile = TrialToPatientBenchmarkProfile.from_dict(
            _plain_json_mapping(
                self.benchmark_profile,
                path="manifest reverse Benchmark Profile",
            )
        )
        object.__setattr__(
            self,
            "benchmark_profile",
            MappingProxyType(benchmark_profile.to_dict()),
        )
        if self.primary_ranking != TRIAL_TO_PATIENT_PRIMARY_RANKING:
            raise SchemaValidationError("invalid reverse Primary Ranking")
        object.__setattr__(self, "pipeline_depth", validate_pipeline_depth(self.pipeline_depth))
        _require_positive_int(self.budget_k, "budget_k")
        _require_positive_int(self.metric_cutoff, "metric_cutoff")
        if self.metric_cutoff not in benchmark_profile.cutoffs:
            raise SchemaValidationError(
                "metric_cutoff must be one of the reverse Benchmark Profile cutoffs"
            )
        query_versions = tuple(self.query_trial_versions)
        patient_versions = tuple(self.patient_corpus_versions)
        if not query_versions or not patient_versions:
            raise SchemaValidationError(
                "reverse manifest requires query and patient entity-version lineage"
            )
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
            for item in query_versions + patient_versions
        ):
            raise SchemaValidationError("reverse manifest entity-version lineage is invalid")
        for _entity_id, version_id in query_versions + patient_versions:
            require_sha256(version_id, "manifest entity version ID")
        if list(query_versions) != sorted(set(query_versions)):
            raise SchemaValidationError("query_trial_versions must be unique and sorted")
        if list(patient_versions) != sorted(set(patient_versions)):
            raise SchemaValidationError("patient_corpus_versions must be unique and sorted")
        references = tuple(self.pair_assessment_references)
        for reference in references:
            require_sha256(reference, "pair assessment reference")
        if len(references) != len(set(references)) or references != tuple(sorted(references)):
            raise SchemaValidationError("pair assessment references must be unique and sorted")
        object.__setattr__(self, "query_trial_versions", query_versions)
        object.__setattr__(self, "patient_corpus_versions", patient_versions)
        self._validate_embedded_system_input(
            query_versions=query_versions,
            patient_versions=patient_versions,
        )
        stage_rankings = tuple(self.stage_rankings)
        if any(not isinstance(item, TrialToPatientStageRanking) for item in stage_rankings):
            raise SchemaValidationError(
                "reverse manifest stage_rankings must contain direction-specific stages"
            )
        stage_names = [item.name for item in stage_rankings]
        if len(stage_names) != len(set(stage_names)) or stage_names != sorted(stage_names):
            raise SchemaValidationError("reverse manifest stage rankings must be unique and sorted")
        if any(item.artifact_hash is None or item.candidates for item in stage_rankings):
            raise SchemaValidationError(
                "reverse manifest stages require hashes and no embedded candidates"
            )
        object.__setattr__(self, "stage_rankings", stage_rankings)
        object.__setattr__(self, "pair_assessment_references", references)
        normalized_reuse_lineage = normalize_pair_assessment_lineage(
            references,
            self.reuse_lineage,
        )
        object.__setattr__(
            self,
            "reuse_lineage",
            MappingProxyType(normalized_reuse_lineage),
        )

    def _validate_embedded_system_input(
        self,
        *,
        query_versions: tuple[tuple[str, str], ...],
        patient_versions: tuple[tuple[str, str], ...],
    ) -> None:
        system_input = self.configuration.get("system_input")
        if not isinstance(system_input, Mapping):
            raise SchemaValidationError(
                "reverse configuration.system_input must contain the exact System input"
            )
        task_fields = {
            "schema_version",
            "task",
            "benchmark_lineage",
            "prepared_snapshot_id",
            "clinical_as_of",
            "patient_evidence_profile",
            "benchmark_profile",
            "query_set_id",
            "patient_corpus_id",
            "trial_versions",
            "patient_versions",
            "available_capabilities",
            "derived_views",
            "task_input_id",
        }
        identity_version = system_input.get("identity_version", "1.0")
        if identity_version not in {"1.0", "2.0"}:
            raise SchemaValidationError("unsupported reverse System Input identity_version")
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
                "reverse configuration.system_input fields do not match the System input contract"
            )
        task_input = TrialToPatientTaskInput.from_dict(
            {name: system_input[name] for name in task_fields}
        )
        if task_input.task_input_id != self.task_input_id:
            raise SchemaValidationError(
                "reverse task_input_id does not match embedded task input content"
            )
        if (
            task_input.benchmark_lineage != self.benchmark_lineage
            or task_input.prepared_snapshot_id != self.prepared_snapshot_id
            or task_input.query_set_id != self.query_set_id
            or task_input.patient_corpus_id != self.patient_corpus_id
            or task_input.clinical_as_of != self.clinical_as_of
            or task_input.patient_evidence_profile.to_dict() != dict(self.patient_evidence_profile)
            or task_input.benchmark_profile.to_dict() != dict(self.benchmark_profile)
        ):
            raise SchemaValidationError(
                "embedded reverse task input does not match the manifest identity"
            )
        embedded_query_versions = tuple(
            (version.trial_id, version.trial_version_id) for version in task_input.trial_versions
        )
        embedded_patient_versions = tuple(
            (version.patient_id, version.patient_version_id)
            for version in task_input.patient_versions
        )
        if embedded_query_versions != query_versions:
            raise SchemaValidationError("embedded reverse trial-version lineage does not match")
        if embedded_patient_versions != patient_versions:
            raise SchemaValidationError("embedded reverse patient-version lineage does not match")
        options = system_input["options"]
        required = system_input["required_capabilities"]
        optional = system_input["optional_capabilities"]
        if not isinstance(options, Mapping):
            raise SchemaValidationError(
                "reverse configuration.system_input.options must be an object"
            )
        for name, value in (
            ("required_capabilities", required),
            ("optional_capabilities", optional),
        ):
            if (
                not isinstance(value, list)
                or any(not isinstance(item, str) or not item for item in value)
                or value != sorted(set(value))
            ):
                raise SchemaValidationError(
                    f"reverse configuration.system_input.{name} must be unique and sorted"
                )
        request = TrialToPatientRunRequest(
            task_input=task_input,
            run_id=self.run_id,
            top_k=cast(int, system_input["top_k"]),
            metric_cutoff=cast(int, system_input["metric_cutoff"]),
            options=cast(Mapping[str, object], options),
            required_capabilities=frozenset(cast(list[str], required)),
            optional_capabilities=frozenset(cast(list[str], optional)),
            identity_version=cast(str, identity_version),
        )
        if request.top_k != self.budget_k or request.metric_cutoff != self.metric_cutoff:
            raise SchemaValidationError(
                "embedded reverse System input cutoffs do not match the manifest"
            )
        if request.system_input_id != self.system_input_id or request.system_input_dict() != dict(
            system_input
        ):
            raise SchemaValidationError(
                "reverse system_input_id does not match embedded System input content"
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "snapshot_contract_version": self.snapshot_contract_version,
            "task": self.task,
            "run_id": self.run_id,
            "system_id": self.system_id,
            "created_at": clinical_as_of_text(self.created_at),
            "execution_provenance": self.execution_provenance.to_dict(),
            "configuration": json_value_to_builtins(self.configuration),
            "runtime_seconds": self.runtime_seconds,
            "benchmark_lineage": self.benchmark_lineage,
            "prepared_snapshot_id": self.prepared_snapshot_id,
            "task_input_id": self.task_input_id,
            "system_input_id": self.system_input_id,
            "query_set_id": self.query_set_id,
            "patient_corpus_id": self.patient_corpus_id,
            "clinical_as_of": clinical_as_of_text(self.clinical_as_of),
            "patient_evidence_profile": json_value_to_builtins(self.patient_evidence_profile),
            "benchmark_profile": json_value_to_builtins(self.benchmark_profile),
            "evaluation_package_id": self.evaluation_package_id,
            "query_trial_versions": [
                {"trial_id": entity_id, "trial_version_id": version_id}
                for entity_id, version_id in self.query_trial_versions
            ],
            "patient_corpus_versions": [
                {"patient_id": entity_id, "patient_version_id": version_id}
                for entity_id, version_id in self.patient_corpus_versions
            ],
            "primary_ranking": self.primary_ranking,
            "pipeline_depth": self.pipeline_depth,
            "budget_k": self.budget_k,
            "metric_cutoff": self.metric_cutoff,
            "candidates_sha256": self.candidates_sha256,
            "stage_rankings": [item.manifest_dict() for item in self.stage_rankings],
            "pair_assessment_references": list(self.pair_assessment_references),
            "reuse_lineage": json_value_to_builtins(self.reuse_lineage),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrialToPatientRunManifest:
        expected = {
            "schema_version",
            "snapshot_contract_version",
            "task",
            "run_id",
            "system_id",
            "created_at",
            "execution_provenance",
            "configuration",
            "runtime_seconds",
            "benchmark_lineage",
            "prepared_snapshot_id",
            "task_input_id",
            "system_input_id",
            "query_set_id",
            "patient_corpus_id",
            "clinical_as_of",
            "patient_evidence_profile",
            "benchmark_profile",
            "evaluation_package_id",
            "query_trial_versions",
            "patient_corpus_versions",
            "primary_ranking",
            "pipeline_depth",
            "budget_k",
            "metric_cutoff",
            "candidates_sha256",
            "stage_rankings",
            "pair_assessment_references",
            "reuse_lineage",
        }
        require_exact_keys(payload, expected, role="TrialToPatientRunManifest")
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported TrialToPatientRunManifest schema_version")
        if payload["task"] != TRIAL_TO_PATIENT_TASK:
            raise SchemaValidationError("reverse manifest task must be 'trial_to_patient'")
        if payload["snapshot_contract_version"] != SNAPSHOT_CONTRACT_VERSION:
            raise SchemaValidationError("unsupported reverse Snapshot contract version")
        configuration = payload["configuration"]
        execution_provenance = payload["execution_provenance"]
        evidence_profile = payload["patient_evidence_profile"]
        benchmark_profile = payload["benchmark_profile"]
        reuse_lineage = payload["reuse_lineage"]
        if not all(
            isinstance(value, Mapping)
            for value in (
                configuration,
                execution_provenance,
                evidence_profile,
                benchmark_profile,
                reuse_lineage,
            )
        ):
            raise SchemaValidationError("reverse manifest object fields are invalid")
        query_versions = cls._parse_lineage(
            payload["query_trial_versions"],
            entity_field="trial_id",
            version_field="trial_version_id",
        )
        patient_versions = cls._parse_lineage(
            payload["patient_corpus_versions"],
            entity_field="patient_id",
            version_field="patient_version_id",
        )
        references = payload["pair_assessment_references"]
        if not isinstance(references, list):
            raise SchemaValidationError("pair_assessment_references must be an array")
        stage_rankings = payload["stage_rankings"]
        if not isinstance(stage_rankings, list) or any(
            not isinstance(item, Mapping) for item in stage_rankings
        ):
            raise SchemaValidationError("reverse stage_rankings must be an array of objects")
        return cls(
            run_id=cast(str, payload["run_id"]),
            system_id=cast(str, payload["system_id"]),
            created_at=parse_clinical_as_of(payload["created_at"]),
            execution_provenance=TrialToPatientExecutionProvenance.from_dict(
                cast(Mapping[str, object], execution_provenance)
            ),
            configuration=cast(Mapping[str, object], configuration),
            runtime_seconds=cast(float, payload["runtime_seconds"]),
            benchmark_lineage=cast(str, payload["benchmark_lineage"]),
            prepared_snapshot_id=cast(str, payload["prepared_snapshot_id"]),
            task_input_id=cast(str, payload["task_input_id"]),
            system_input_id=cast(str, payload["system_input_id"]),
            query_set_id=cast(str, payload["query_set_id"]),
            patient_corpus_id=cast(str, payload["patient_corpus_id"]),
            clinical_as_of=parse_clinical_as_of(payload["clinical_as_of"]),
            patient_evidence_profile=cast(Mapping[str, object], evidence_profile),
            benchmark_profile=cast(Mapping[str, object], benchmark_profile),
            evaluation_package_id=cast(str, payload["evaluation_package_id"]),
            query_trial_versions=query_versions,
            patient_corpus_versions=patient_versions,
            primary_ranking=cast(str, payload["primary_ranking"]),
            pipeline_depth=cast(str, payload["pipeline_depth"]),
            budget_k=cast(int, payload["budget_k"]),
            metric_cutoff=cast(int, payload["metric_cutoff"]),
            candidates_sha256=cast(str, payload["candidates_sha256"]),
            stage_rankings=tuple(
                TrialToPatientStageRanking.from_manifest_dict(cast(Mapping[str, object], item))
                for item in stage_rankings
            ),
            pair_assessment_references=tuple(cast(list[str], references)),
            reuse_lineage=cast(Mapping[str, object], reuse_lineage),
        )

    @staticmethod
    def _parse_lineage(
        value: object,
        *,
        entity_field: str,
        version_field: str,
    ) -> tuple[tuple[str, str], ...]:
        if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
            raise SchemaValidationError("reverse manifest lineage must be an array of objects")
        result: list[tuple[str, str]] = []
        for item in value:
            row = cast(Mapping[str, object], item)
            require_exact_keys(
                row,
                {entity_field, version_field},
                role="reverse manifest lineage row",
            )
            result.append(
                (
                    cast(str, row[entity_field]),
                    cast(str, row[version_field]),
                )
            )
        return tuple(result)


@dataclass(frozen=True, slots=True)
class StoredTrialToPatientRun:
    directory: Path
    manifest: TrialToPatientRunManifest
    candidates: tuple[TrialToPatientCandidate, ...]
    stage_rankings: tuple[TrialToPatientStageRanking, ...]
    manifest_hash: str
    candidates_hash: str


def _validate_reverse_candidates(
    candidates: tuple[TrialToPatientCandidate, ...],
    *,
    request: TrialToPatientRunRequest | None,
    manifest: TrialToPatientRunManifest | None,
    require_complete: bool = True,
) -> None:
    if not candidates:
        raise SchemaValidationError("reverse run must contain patient candidates")
    run_ids = {row.run_id for row in candidates}
    system_ids = {row.system_id for row in candidates}
    if len(run_ids) != 1 or len(system_ids) != 1:
        raise SchemaValidationError("reverse candidates must share one run_id and system_id")
    order = [(row.trial_id, row.rank) for row in candidates]
    if order != sorted(order):
        raise SchemaValidationError(
            "reverse candidates must use deterministic trial and rank order"
        )
    pair_keys: set[tuple[str, str]] = set()
    rank_keys: set[tuple[str, int]] = set()
    for row in candidates:
        pair_key = (row.trial_version_id, row.patient_version_id)
        rank_key = (row.trial_version_id, row.rank)
        if pair_key in pair_keys:
            raise SchemaValidationError("reverse candidates contain a duplicate patient pair")
        if rank_key in rank_keys:
            raise SchemaValidationError("reverse candidates contain a duplicate query rank")
        pair_keys.add(pair_key)
        rank_keys.add(rank_key)
    if request is not None:
        expected_trials = {
            (version.trial_id, version.trial_version_id)
            for version in request.task_input.trial_versions
        }
        expected_patients = {
            (version.patient_id, version.patient_version_id)
            for version in request.task_input.patient_versions
        }
        if any(
            (row.trial_id, row.trial_version_id) not in expected_trials
            or (row.patient_id, row.patient_version_id) not in expected_patients
            for row in candidates
        ):
            raise SchemaValidationError("reverse candidate references an unknown entity version")
        if require_complete:
            expected_depth = min(request.top_k, len(expected_patients))
            for trial_id, trial_version_id in expected_trials:
                ranks = {
                    row.rank
                    for row in candidates
                    if (row.trial_id, row.trial_version_id) == (trial_id, trial_version_id)
                }
                if ranks != set(range(1, expected_depth + 1)):
                    raise SchemaValidationError(
                        "reverse candidate ranking is incomplete for a trial query"
                    )
    if manifest is not None:
        if run_ids != {manifest.run_id} or system_ids != {manifest.system_id}:
            raise SchemaValidationError("reverse candidates do not match their manifest")
        query_versions = set(manifest.query_trial_versions)
        patient_versions = set(manifest.patient_corpus_versions)
        if any(
            (row.trial_id, row.trial_version_id) not in query_versions
            or (row.patient_id, row.patient_version_id) not in patient_versions
            for row in candidates
        ):
            raise SchemaValidationError(
                "reverse candidate entity versions do not match their manifest"
            )
        if require_complete:
            expected_depth = min(manifest.budget_k, len(patient_versions))
            for trial_id, trial_version_id in query_versions:
                ranks = {
                    row.rank
                    for row in candidates
                    if (row.trial_id, row.trial_version_id) == (trial_id, trial_version_id)
                }
                if ranks != set(range(1, expected_depth + 1)):
                    raise SchemaValidationError(
                        "reverse artifact ranking is incomplete for a trial query"
                    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_reverse_candidates(
    path: Path,
    candidates: tuple[TrialToPatientCandidate, ...],
) -> None:
    path.write_text(
        "".join(f"{candidate.to_json()}\n" for candidate in candidates),
        encoding="utf-8",
        newline="\n",
    )


def write_trial_to_patient_run(
    directory: str | Path,
    *,
    resolved: ResolvedTrialToPatientBenchmark,
    request: TrialToPatientRunRequest,
    result: TrialToPatientRunResult,
    execution_provenance: TrialToPatientExecutionProvenance,
    created_at: datetime,
    pair_assessment_runs: Iterable[StoredPairAssessmentRun] = (),
) -> StoredTrialToPatientRun:
    """Atomically persist one closed reverse Primary Ranking."""

    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"reverse run directory already exists: {destination}")
    if request.task_input != resolved.task_input:
        raise SchemaValidationError("reverse request does not match the resolved benchmark")
    if result.primary_ranking != TRIAL_TO_PATIENT_PRIMARY_RANKING:
        raise SchemaValidationError("reverse result has an invalid Primary Ranking")
    if any(row.run_id != request.run_id for row in result.candidates):
        raise SchemaValidationError("reverse result candidates do not match the request")
    if not isinstance(execution_provenance, TrialToPatientExecutionProvenance):
        raise TypeError("execution_provenance must be a TrialToPatientExecutionProvenance")
    system_id = result.system_id
    if system_id not in TRIAL_TO_PATIENT_SYSTEMS:
        raise SchemaValidationError("reverse result was not produced by a registered System")
    _system, effective_request = _effective_trial_to_patient_request(system_id, request)
    if result.system_input_id != effective_request.system_input_id:
        raise SchemaValidationError("reverse result does not bind the effective System Input")
    system_identity = result.configuration.get("system_identity")
    if not isinstance(system_identity, Mapping) or system_identity.get("system_id") != system_id:
        raise SchemaValidationError("reverse configuration does not bind the System identity")
    for name in ("model_identity", "index_identity"):
        configured_identity = result.configuration.get(name)
        recorded_identity = getattr(execution_provenance, name)
        if not isinstance(configured_identity, Mapping) or dict(configured_identity) != dict(
            recorded_identity
        ):
            raise SchemaValidationError(
                f"reverse execution provenance {name} does not match the System result"
            )
    _validate_reverse_candidates(
        result.candidates,
        request=effective_request,
        manifest=None,
    )
    stored_pair_assessments = tuple(pair_assessment_runs)
    expected_patient_versions = {
        version.patient_version_id for version in effective_request.task_input.patient_versions
    }
    expected_trial_versions = {
        version.trial_version_id for version in effective_request.task_input.trial_versions
    }
    for stored_pair_run in stored_pair_assessments:
        if not isinstance(stored_pair_run, StoredPairAssessmentRun):
            raise TypeError("pair_assessment_runs must contain StoredPairAssessmentRun instances")
        if any(
            pair_input.patient_version.patient_version_id not in expected_patient_versions
            or pair_input.trial_version.trial_version_id not in expected_trial_versions
            for pair_input in stored_pair_run.inputs.values()
        ):
            raise SchemaValidationError(
                "Pair Assessment reference contains entity versions outside the reverse task"
            )
    pair_assessment_references, reuse_lineage = pair_assessment_collection_lineage(
        stored_pair_assessments
    )
    sorted_stages = tuple(sorted(result.stage_rankings, key=lambda item: item.name))
    for stage in sorted_stages:
        if stage.candidates:
            if any(
                row.run_id != request.run_id or row.system_id != system_id
                for row in stage.candidates
            ):
                raise SchemaValidationError(
                    "reverse stage candidates do not match the Primary Ranking"
                )
            _validate_reverse_candidates(
                stage.candidates,
                request=effective_request,
                manifest=None,
                require_complete=False,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name or 'reverse-run'}.", dir=destination.parent)
    )
    try:
        candidates_path = staging / "patient-candidates.jsonl"
        _write_reverse_candidates(candidates_path, result.candidates)
        candidates_hash = sha256_file(candidates_path)
        manifest_stages: list[TrialToPatientStageRanking] = []
        for stage in sorted_stages:
            stage_path = staging / f"stage-{stage.name}.jsonl"
            _write_reverse_candidates(stage_path, stage.candidates)
            manifest_stages.append(
                TrialToPatientStageRanking(
                    name=stage.name,
                    pipeline_depth=stage.pipeline_depth,
                    artifact_hash=sha256_file(stage_path),
                )
            )
        manifest = TrialToPatientRunManifest(
            run_id=request.run_id,
            system_id=system_id,
            created_at=created_at,
            execution_provenance=execution_provenance,
            configuration={
                **result.configuration,
                "system_input": effective_request.system_input_dict(),
            },
            runtime_seconds=result.runtime_seconds,
            benchmark_lineage=request.task_input.benchmark_lineage,
            prepared_snapshot_id=request.task_input.prepared_snapshot_id,
            task_input_id=request.task_input.task_input_id,
            system_input_id=effective_request.system_input_id,
            query_set_id=request.task_input.query_set_id,
            patient_corpus_id=request.task_input.patient_corpus_id,
            clinical_as_of=request.task_input.clinical_as_of,
            patient_evidence_profile=(request.task_input.patient_evidence_profile.to_dict()),
            benchmark_profile=request.task_input.benchmark_profile.to_dict(),
            evaluation_package_id=resolved.evaluation_package.evaluation_package_id,
            query_trial_versions=tuple(
                (version.trial_id, version.trial_version_id)
                for version in request.task_input.trial_versions
            ),
            patient_corpus_versions=tuple(
                (version.patient_id, version.patient_version_id)
                for version in request.task_input.patient_versions
            ),
            primary_ranking=result.primary_ranking,
            pipeline_depth=result.pipeline_depth,
            budget_k=request.top_k,
            metric_cutoff=request.metric_cutoff,
            candidates_sha256=candidates_hash,
            stage_rankings=tuple(manifest_stages),
            pair_assessment_references=pair_assessment_references,
            reuse_lineage=reuse_lineage,
        )
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest.to_dict())
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_trial_to_patient_run(destination)


def load_trial_to_patient_run(directory: str | Path) -> StoredTrialToPatientRun:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    candidates_path = root / "patient-candidates.jsonl"
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"cannot read reverse run manifest: {exc}") from exc
    if not isinstance(manifest_payload, Mapping):
        raise SchemaValidationError("reverse run manifest must be a JSON object")
    manifest = TrialToPatientRunManifest.from_dict(cast(Mapping[str, object], manifest_payload))
    try:
        lines = candidates_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SchemaValidationError(f"cannot read reverse candidates: {exc}") from exc
    if not lines or any(not line for line in lines):
        raise SchemaValidationError("reverse candidates must be non-empty JSON Lines")
    candidates = tuple(TrialToPatientCandidate.from_json(line) for line in lines)
    if lines != [candidate.to_json() for candidate in candidates]:
        raise SchemaValidationError("reverse candidates are not canonical JSON Lines")
    candidates_hash = sha256_file(candidates_path)
    if candidates_hash != manifest.candidates_sha256:
        raise SchemaValidationError("reverse candidates hash does not match the manifest")
    _validate_reverse_candidates(candidates, request=None, manifest=manifest)
    loaded_stages: list[TrialToPatientStageRanking] = []
    for stage in manifest.stage_rankings:
        stage_path = root / f"stage-{stage.name}.jsonl"
        try:
            stage_lines = stage_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise SchemaValidationError(f"cannot read reverse stage {stage.name}: {exc}") from exc
        if any(not line for line in stage_lines):
            raise SchemaValidationError("reverse stage must use canonical JSON Lines")
        stage_candidates = tuple(TrialToPatientCandidate.from_json(line) for line in stage_lines)
        if stage_lines != [candidate.to_json() for candidate in stage_candidates]:
            raise SchemaValidationError(f"reverse stage {stage.name} is not canonical JSON Lines")
        if sha256_file(stage_path) != stage.artifact_hash:
            raise SchemaValidationError(
                f"reverse stage {stage.name} hash does not match the manifest"
            )
        if stage_candidates:
            _validate_reverse_candidates(
                stage_candidates,
                request=None,
                manifest=manifest,
                require_complete=False,
            )
        loaded_stages.append(
            TrialToPatientStageRanking(
                name=stage.name,
                pipeline_depth=stage.pipeline_depth,
                candidates=stage_candidates,
                artifact_hash=stage.artifact_hash,
            )
        )
    expected_stage_paths = {
        (root / f"stage-{stage.name}.jsonl").resolve() for stage in manifest.stage_rankings
    }
    actual_stage_paths = {path.resolve() for path in root.glob("stage-*.jsonl")}
    if actual_stage_paths != expected_stage_paths:
        raise SchemaValidationError("reverse stage artifacts do not match the manifest")
    return StoredTrialToPatientRun(
        directory=root.resolve(),
        manifest=manifest,
        candidates=candidates,
        stage_rankings=tuple(loaded_stages),
        manifest_hash=sha256_file(manifest_path),
        candidates_hash=candidates_hash,
    )


def evaluate_trial_to_patient_run(
    stored: StoredTrialToPatientRun,
    evaluation_package: TrialToPatientEvaluationPackage,
) -> dict[str, object]:
    """Evaluate a closed reverse artifact with separately supplied judgments."""

    if not isinstance(stored, StoredTrialToPatientRun):
        raise TypeError("stored must be a StoredTrialToPatientRun")
    if not isinstance(evaluation_package, TrialToPatientEvaluationPackage):
        raise TypeError("evaluation_package must be a TrialToPatientEvaluationPackage")
    manifest = stored.manifest
    benchmark_profile = TrialToPatientBenchmarkProfile.from_dict(manifest.benchmark_profile)
    if (
        manifest.evaluation_package_id != evaluation_package.evaluation_package_id
        or manifest.benchmark_lineage != evaluation_package.benchmark_lineage
        or manifest.query_set_id != evaluation_package.query_set_id
        or manifest.patient_corpus_id != evaluation_package.patient_corpus_id
        or manifest.prepared_snapshot_id != evaluation_package.prepared_snapshot_id
        or evaluation_package.benchmark_profile_definition_sha256
        != benchmark_profile.definition_sha256
    ):
        raise SchemaValidationError("reverse Evaluation Package does not match the closed run")
    _validate_reverse_judgment_membership(
        evaluation_package.judgments,
        query_trial_versions=manifest.query_trial_versions,
        patient_corpus_versions=manifest.patient_corpus_versions,
        boundary="closed reverse run",
    )
    if _pair_coverage_policy(benchmark_profile) == "complete_pair_matrix":
        _validate_reverse_judgment_closure(
            evaluation_package.judgments,
            query_trial_versions=manifest.query_trial_versions,
            patient_corpus_versions=manifest.patient_corpus_versions,
            boundary="closed reverse run",
        )
    rankings: dict[str, list[TrialToPatientCandidate]] = defaultdict(list)
    for candidate in stored.candidates:
        rankings[candidate.trial_version_id].append(candidate)
    relevance: dict[str, dict[str, int]] = defaultdict(dict)
    for judgment in evaluation_package.judgments:
        relevance[judgment.trial_version_id][judgment.patient_version_id] = judgment.label
    ranked_patient_versions = {
        trial_version_id: [
            row.patient_version_id
            for row in sorted(rankings.get(trial_version_id, []), key=lambda row: row.rank)
        ]
        for _trial_id, trial_version_id in manifest.query_trial_versions
    }
    per_trial: dict[str, dict[str, object]] = {
        trial_id: {"trial_version_id": trial_version_id}
        for trial_id, trial_version_id in manifest.query_trial_versions
    }
    aggregate_values: dict[str, list[float]] = defaultdict(list)
    metric_names = (
        "ndcg",
        "eligible_recall",
        "relevant_or_eligible_recall",
    )
    for cutoff in benchmark_profile.cutoffs:
        for trial_id, trial_version_id in manifest.query_trial_versions:
            metrics = evaluate_topic(
                ranked_patient_versions[trial_version_id],
                relevance.get(trial_version_id, {}),
                k=cutoff,
            )
            for metric_name in metric_names:
                key = f"{metric_name}_at_{cutoff}"
                value = cast(float, getattr(metrics, metric_name))
                per_trial[trial_id][key] = value
                aggregate_values[key].append(value)
    divisor = len(per_trial)
    return {
        "schema_version": TRIAL_TO_PATIENT_SCHEMA_VERSION,
        "task": TRIAL_TO_PATIENT_TASK,
        "run_id": manifest.run_id,
        "system_id": manifest.system_id,
        "task_input_id": manifest.task_input_id,
        "evaluation_package_id": evaluation_package.evaluation_package_id,
        "query_count": divisor,
        "primary_metric_cutoff": manifest.metric_cutoff,
        "cutoffs": list(benchmark_profile.cutoffs),
        "aggregate": {
            "mean": {key: sum(values) / divisor for key, values in aggregate_values.items()},
            "median": {
                key: float(statistics.median(values)) for key, values in aggregate_values.items()
            },
        },
        "per_trial": per_trial,
    }


__all__ = [
    "DEFAULT_TRIAL_TO_PATIENT_PROFILE_ID",
    "REVERSE_PRESERVE_JUDGED_PAIRS_V1",
    "REVERSE_TRANSPOSE_COMPLETE_PAIR_MATRIX_V1",
    "TRIAL_TO_PATIENT_BM25_SYSTEM_ID",
    "TRIAL_TO_PATIENT_PRIMARY_RANKING",
    "TRIAL_TO_PATIENT_SYSTEMS",
    "TRIAL_TO_PATIENT_TASK",
    "PatientEvidenceProfile",
    "ResolvedTrialToPatientBenchmark",
    "StoredTrialToPatientRun",
    "TrialToPatientBenchmarkProfile",
    "TrialToPatientCandidate",
    "TrialToPatientEvaluationPackage",
    "TrialToPatientExecutionProvenance",
    "TrialToPatientJudgment",
    "TrialToPatientRunManifest",
    "TrialToPatientRunRequest",
    "TrialToPatientRunResult",
    "TrialToPatientStageRanking",
    "TrialToPatientTaskInput",
    "default_trial_to_patient_profile",
    "evaluate_trial_to_patient_run",
    "load_trial_to_patient_run",
    "resolve_trial_to_patient_benchmark",
    "run_trial_to_patient_system",
    "write_trial_to_patient_run",
]
