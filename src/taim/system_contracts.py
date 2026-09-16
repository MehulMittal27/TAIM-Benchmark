"""Registry-independent contracts for public patient-to-trial Systems."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, cast, runtime_checkable

from taim.contracts import (
    canonical_json,
    content_sha256,
    portable_system_input_identity_value,
    require_exact_keys,
    require_non_empty,
    require_sha256,
    system_input_identity_value,
    task_input_membership_id,
)
from taim.entity_versions import (
    PatientEntityVersion,
    PatientEvidenceProfile,
    TrialVersion,
    clinical_as_of_text,
    parse_clinical_as_of,
    resolve_task_derived_views,
    validate_clinical_as_of,
    version_patients,
    version_trials,
)
from taim.schemas import (
    Candidate,
    JsonValue,
    SchemaValidationError,
    StageRanking,
    freeze_json_value_serializable,
    validate_pipeline_depth,
    validate_primary_ranking,
)
from taim.snapshot import (
    CAPABILITY_PATIENT_EVIDENCE,
    CAPABILITY_TYPED_PATIENT_CORE,
    DERIVED_VIEW_CAPABILITY_PREFIX,
    BenchmarkSnapshot,
    BenchmarkTopic,
    DerivedView,
    TrialDocument,
    freeze_system_input_options,
    validate_capability_declarations,
)

if TYPE_CHECKING:
    from taim.data import PreparedBenchmark

PATIENT_TO_TRIAL_EXECUTION_CONTRACT_VERSION: Final[str] = "1.0"


@runtime_checkable
class ResolvedBenchmarkContract(Protocol):
    """Structural input shared by private and public-safe Profile registries."""

    @property
    def prepared(self) -> PreparedBenchmark: ...

    @property
    def trials(self) -> Sequence[TrialDocument]: ...


@dataclass(frozen=True, slots=True)
class SystemBenchmarkSnapshot:
    """Exact patient-to-trial task input exposed to a System."""

    benchmark_lineage: str
    topics: tuple[BenchmarkTopic, ...]
    trials: Sequence[TrialDocument]
    available_capabilities: frozenset[str]
    derived_views: tuple[DerivedView, ...]
    prepared_snapshot_id: str
    clinical_as_of: datetime
    patient_evidence_profile: PatientEvidenceProfile
    patient_versions: tuple[PatientEntityVersion, ...] = field(init=False)
    trial_versions: tuple[TrialVersion, ...] = field(init=False)
    task_input_id: str = field(init=False)

    contract_version = BenchmarkSnapshot.contract_version
    canonicalization_version = BenchmarkSnapshot.canonicalization_version

    def __post_init__(self) -> None:
        require_non_empty(self.benchmark_lineage, "benchmark_lineage")
        topics = tuple(self.topics)
        trials = tuple(self.trials)
        derived_views = tuple(self.derived_views)
        capabilities = frozenset(self.available_capabilities)
        if not topics or not trials:
            raise ValueError("patient-to-trial task input requires patients and trials")
        if any(not isinstance(item, BenchmarkTopic) for item in topics):
            raise TypeError("patient-to-trial topics must be BenchmarkTopic instances")
        if any(not isinstance(item, TrialDocument) for item in trials):
            raise TypeError("patient-to-trial trials must be TrialDocument instances")
        if len({topic.topic_id for topic in topics}) != len(topics):
            raise SchemaValidationError("patient-to-trial patient identities must be unique")
        if len({trial.trial_id for trial in trials}) != len(trials):
            raise SchemaValidationError("patient-to-trial trial identities must be unique")
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise TypeError("available capabilities must be non-empty strings")
        capabilities, derived_views = validate_capability_declarations(
            capabilities,
            derived_views,
            role="Snapshot",
        )
        if not isinstance(self.patient_evidence_profile, PatientEvidenceProfile):
            raise TypeError("patient_evidence_profile must be a PatientEvidenceProfile")
        clinical_as_of = validate_clinical_as_of(self.clinical_as_of)
        require_sha256(self.prepared_snapshot_id, "prepared_snapshot_id")
        derived_views = resolve_task_derived_views(
            derived_views,
            topics=topics,
            trials=trials,
            profile=self.patient_evidence_profile,
            clinical_as_of=clinical_as_of,
        )
        visible_patient_fields = set(
            cast(
                tuple[str, ...],
                self.patient_evidence_profile.definition["visible_patient_fields"],
            )
        )
        capabilities = frozenset(
            capability
            for capability in capabilities
            if not capability.startswith(DERIVED_VIEW_CAPABILITY_PREFIX)
            and not (
                capability == CAPABILITY_PATIENT_EVIDENCE
                and "evidence_items" not in visible_patient_fields
            )
            and not (
                capability == CAPABILITY_TYPED_PATIENT_CORE
                and "typed_patient_core" not in visible_patient_fields
            )
        ) | frozenset(view.capability for view in derived_views)
        patient_versions = version_patients(
            topics,
            self.patient_evidence_profile,
            clinical_as_of=clinical_as_of,
            derived_views=derived_views,
        )
        trial_versions = version_trials(trials, derived_views=derived_views)
        object.__setattr__(self, "clinical_as_of", clinical_as_of)
        object.__setattr__(self, "topics", tuple(item.topic for item in patient_versions))
        object.__setattr__(self, "trials", tuple(item.trial for item in trial_versions))
        object.__setattr__(self, "available_capabilities", capabilities)
        object.__setattr__(self, "derived_views", derived_views)
        object.__setattr__(self, "patient_versions", patient_versions)
        object.__setattr__(self, "trial_versions", trial_versions)
        object.__setattr__(self, "task_input_id", content_sha256(self._identity_payload()))

    def _identity_payload(self) -> dict[str, JsonValue]:
        return {
            "snapshot_contract_version": self.contract_version,
            "task": "patient_to_trial",
            "benchmark_lineage": self.benchmark_lineage,
            "prepared_snapshot_id": self.prepared_snapshot_id,
            "clinical_as_of": clinical_as_of_text(self.clinical_as_of),
            "patient_evidence_profile": self.patient_evidence_profile.to_dict(),
            "query_patient_versions": [item.to_dict() for item in self.patient_versions],
            "trial_corpus_versions": [item.to_dict() for item in self.trial_versions],
            "available_capabilities": cast(list[JsonValue], sorted(self.available_capabilities)),
            "derived_views": [item.to_dict() for item in self.derived_views],
        }

    def task_input_dict(self) -> dict[str, JsonValue]:
        """Return the complete, content-addressed forward task contract."""

        return {**self._identity_payload(), "task_input_id": self.task_input_id}

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the existing authoritative Task Input payload."""

        return self.task_input_dict()

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SystemBenchmarkSnapshot:
        require_exact_keys(
            payload,
            {
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
            },
            role="SystemBenchmarkSnapshot",
        )
        if payload["snapshot_contract_version"] != cls.contract_version:
            raise SchemaValidationError("unsupported SystemBenchmarkSnapshot contract version")
        if payload["task"] != "patient_to_trial":
            raise SchemaValidationError("SystemBenchmarkSnapshot task must be 'patient_to_trial'")
        evidence_profile = payload["patient_evidence_profile"]
        patient_versions = payload["query_patient_versions"]
        trial_versions = payload["trial_corpus_versions"]
        capabilities = payload["available_capabilities"]
        derived_views = payload["derived_views"]
        if not isinstance(evidence_profile, Mapping):
            raise SchemaValidationError("patient_evidence_profile must be an object")
        if not isinstance(patient_versions, list) or any(
            not isinstance(version, Mapping) for version in patient_versions
        ):
            raise SchemaValidationError("query_patient_versions must be an array")
        if not isinstance(trial_versions, list) or any(
            not isinstance(version, Mapping) for version in trial_versions
        ):
            raise SchemaValidationError("trial_corpus_versions must be an array")
        if not isinstance(capabilities, list) or any(
            not isinstance(capability, str) for capability in capabilities
        ):
            raise SchemaValidationError("available_capabilities must be an array of strings")
        if not isinstance(derived_views, list) or any(
            not isinstance(view, Mapping) for view in derived_views
        ):
            raise SchemaValidationError("derived_views must be an array")

        profile = PatientEvidenceProfile.from_dict(cast(Mapping[str, object], evidence_profile))
        parsed_patient_versions = tuple(
            PatientEntityVersion.from_dict(cast(Mapping[str, object], version))
            for version in patient_versions
        )
        parsed_trial_versions = tuple(
            TrialVersion.from_dict(cast(Mapping[str, object], version))
            for version in trial_versions
        )
        clinical_as_of = parse_clinical_as_of(payload["clinical_as_of"])
        if any(
            version.patient_evidence_profile != profile or version.clinical_as_of != clinical_as_of
            for version in parsed_patient_versions
        ):
            raise SchemaValidationError(
                "query_patient_versions must share the Task Input profile and clinical_as_of"
            )
        parsed_views = tuple(
            DerivedView.from_dict(cast(Mapping[str, object], view)) for view in derived_views
        )
        parsed_capabilities = frozenset(cast(list[str], capabilities))
        task_input = cls(
            benchmark_lineage=cast(str, payload["benchmark_lineage"]),
            topics=tuple(version.topic for version in parsed_patient_versions),
            trials=tuple(version.trial for version in parsed_trial_versions),
            available_capabilities=parsed_capabilities,
            derived_views=parsed_views,
            prepared_snapshot_id=cast(str, payload["prepared_snapshot_id"]),
            clinical_as_of=clinical_as_of,
            patient_evidence_profile=profile,
        )
        require_sha256(payload["task_input_id"], "task_input_id")
        if payload["task_input_id"] != task_input.task_input_id:
            raise SchemaValidationError("task_input_id does not match forward task input content")
        if task_input.to_dict() != dict(payload):
            raise SchemaValidationError(
                "serialized SystemBenchmarkSnapshot does not match reconstructed content"
            )
        return task_input

    @classmethod
    def from_json(cls, serialized: str) -> SystemBenchmarkSnapshot:
        try:
            payload = json.loads(serialized)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SchemaValidationError(f"invalid SystemBenchmarkSnapshot JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SchemaValidationError("SystemBenchmarkSnapshot must be a JSON object")
        return cls.from_dict(cast(Mapping[str, object], payload))

    @classmethod
    def from_benchmark(
        cls,
        benchmark: ResolvedBenchmarkContract,
        *,
        clinical_as_of: datetime,
        patient_evidence_profile: PatientEvidenceProfile,
    ) -> SystemBenchmarkSnapshot:
        snapshot = benchmark.prepared.snapshot
        return cls(
            benchmark_lineage=snapshot.benchmark_lineage,
            topics=snapshot.topics,
            trials=benchmark.trials,
            available_capabilities=snapshot.available_capabilities,
            derived_views=snapshot.derived_views,
            prepared_snapshot_id=snapshot.snapshot_id,
            clinical_as_of=clinical_as_of,
            patient_evidence_profile=patient_evidence_profile,
        )


@dataclass(frozen=True, slots=True)
class SystemRunRequest:
    """One content-addressed request to a public System."""

    snapshot: SystemBenchmarkSnapshot
    run_id: str
    top_k: int
    metric_cutoff: int
    options: Mapping[str, object]
    required_capabilities: frozenset[str] = frozenset()
    optional_capabilities: frozenset[str] = frozenset()
    identity_version: str = "1.0"
    system_input_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, SystemBenchmarkSnapshot):
            raise TypeError("snapshot must be a judgment-free BenchmarkSnapshot")
        require_non_empty(self.run_id, "run_id")
        if self.identity_version not in {"1.0", "2.0"}:
            raise ValueError("System Input identity_version must be '1.0' or '2.0'")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if (
            isinstance(self.metric_cutoff, bool)
            or not isinstance(self.metric_cutoff, int)
            or self.metric_cutoff < 1
        ):
            raise ValueError("metric_cutoff must be a positive integer")
        required = frozenset(self.required_capabilities)
        optional = frozenset(self.optional_capabilities)
        if required & optional:
            raise ValueError("required and optional System capabilities must not overlap")
        if any(not isinstance(item, str) or not item for item in required | optional):
            raise TypeError("System capabilities must be non-empty strings")
        missing = required - self.snapshot.available_capabilities
        if missing:
            raise ValueError(
                "Snapshot is missing required System capabilities: " + ", ".join(sorted(missing))
            )
        object.__setattr__(self, "required_capabilities", required)
        object.__setattr__(self, "optional_capabilities", optional)
        options = freeze_system_input_options(self.options)
        object.__setattr__(self, "options", options)
        identity_options = system_input_identity_value(options, path="options")
        identity_payload: dict[str, JsonValue] = {
            "task_input_id": self.snapshot.task_input_id,
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
                        task="patient_to_trial",
                        patient_entity_versions=(
                            (version.patient_id, version.patient_version_id)
                            for version in self.snapshot.patient_versions
                        ),
                        trial_entity_versions=(
                            (version.trial_id, version.trial_version_id)
                            for version in self.snapshot.trial_versions
                        ),
                    ),
                    "options": identity_options,
                }
            )
        object.__setattr__(self, "system_input_id", content_sha256(identity_payload))

    def system_input_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            **self.snapshot.task_input_dict(),
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
        benchmark: ResolvedBenchmarkContract,
        *,
        run_id: str,
        top_k: int,
        metric_cutoff: int,
        options: Mapping[str, object],
        clinical_as_of: datetime,
        patient_evidence_profile: PatientEvidenceProfile,
        required_capabilities: Iterable[str] = (),
        optional_capabilities: Iterable[str] = (),
        identity_version: str = "1.0",
    ) -> SystemRunRequest:
        """Build a System request without exposing eval-side judgments."""

        if not isinstance(benchmark, ResolvedBenchmarkContract):
            raise TypeError("benchmark must be a profile-resolved PreparedBenchmark")
        return cls(
            snapshot=SystemBenchmarkSnapshot.from_benchmark(
                benchmark,
                clinical_as_of=clinical_as_of,
                patient_evidence_profile=patient_evidence_profile,
            ),
            run_id=run_id,
            top_k=top_k,
            metric_cutoff=metric_cutoff,
            options=options,
            required_capabilities=frozenset(required_capabilities),
            optional_capabilities=frozenset(optional_capabilities),
            identity_version=identity_version,
        )


@dataclass(frozen=True, slots=True)
class SystemRunResult:
    """The Primary Ranking and provenance emitted by a System."""

    candidates: Sequence[Candidate]
    configuration: Mapping[str, Any]
    runtime_seconds: float
    primary_ranking: str
    pipeline_depth: str
    stage_rankings: tuple[StageRanking, ...] = ()

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, Candidate) for candidate in candidates):
            raise TypeError("candidates must contain Candidate instances")
        if not isinstance(self.configuration, Mapping):
            raise TypeError("configuration must be a mapping")
        try:
            frozen_configuration = freeze_json_value_serializable(self.configuration)
        except SchemaValidationError as exc:
            raise ValueError("configuration must contain finite JSON values") from exc
        if (
            isinstance(self.runtime_seconds, bool)
            or not isinstance(self.runtime_seconds, int | float)
            or not math.isfinite(self.runtime_seconds)
            or self.runtime_seconds < 0
        ):
            raise ValueError("runtime_seconds must be a finite non-negative number")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "configuration",
            frozen_configuration,
        )
        object.__setattr__(self, "runtime_seconds", float(self.runtime_seconds))
        object.__setattr__(self, "primary_ranking", validate_primary_ranking(self.primary_ranking))
        object.__setattr__(self, "pipeline_depth", validate_pipeline_depth(self.pipeline_depth))
        stage_rankings = tuple(self.stage_rankings)
        if any(not isinstance(stage, StageRanking) for stage in stage_rankings):
            raise TypeError("stage_rankings must contain StageRanking instances")
        names = [stage.name for stage in stage_rankings]
        if len(names) != len(set(names)):
            raise ValueError("stage_rankings must have unique names")
        object.__setattr__(self, "stage_rankings", stage_rankings)


class System(Protocol):
    """Small public interface implemented by one ranking method."""

    system_id: str
    option_names: frozenset[str]
    required_capabilities: frozenset[str]
    optional_capabilities: frozenset[str]

    def normalize_options(self, options: Mapping[str, object]) -> Mapping[str, object]: ...

    def run(self, request: SystemRunRequest) -> SystemRunResult: ...


class StrictSystemOptions:
    """Default System-owned option contract with fail-closed unknown-key handling."""

    option_names: frozenset[str]

    def normalize_options(self, options: Mapping[str, object]) -> Mapping[str, object]:
        unexpected = set(options) - self.option_names
        if unexpected:
            raise ValueError("unknown method options: " + ", ".join(sorted(unexpected)))
        return MappingProxyType({name: options[name] for name in sorted(options)})


def bind_system_capabilities(system: System, request: SystemRunRequest) -> SystemRunRequest:
    """Bind a System's declared capability contract into one effective request."""

    required = request.required_capabilities | frozenset(system.required_capabilities)
    optional = (request.optional_capabilities | frozenset(system.optional_capabilities)) - required
    return replace(
        request,
        required_capabilities=required,
        optional_capabilities=optional,
    )


def normalize_system_options(
    system: System,
    request: SystemRunRequest,
    *,
    discard_adapter_options: bool = False,
) -> SystemRunRequest:
    """Ask the System to validate and normalize only its method configuration."""

    options = request.options
    if discard_adapter_options:
        options = MappingProxyType(
            {name: request.options[name] for name in system.option_names if name in request.options}
        )
    normalized = system.normalize_options(options)
    return replace(request, options=normalized)


def normalize_system_request(system: System, request: SystemRunRequest) -> SystemRunRequest:
    """Construct the strict public request presented to a System."""

    return normalize_system_options(system, bind_system_capabilities(system, request))


__all__ = [
    "PATIENT_TO_TRIAL_EXECUTION_CONTRACT_VERSION",
    "StrictSystemOptions",
    "System",
    "SystemBenchmarkSnapshot",
    "SystemRunRequest",
    "SystemRunResult",
    "bind_system_capabilities",
    "normalize_system_options",
    "normalize_system_request",
]
