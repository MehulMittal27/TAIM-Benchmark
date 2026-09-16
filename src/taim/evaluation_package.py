"""Evaluator-only packages and connector diagnostics."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from taim.contracts import (
    EVALUATION_PACKAGE_VERSION,
    SNAPSHOT_CONTRACT_VERSION,
    EvaluatorOnlyMaterial,
    canonical_json,
    content_sha256,
    freeze_json_mapping,
    require_non_empty,
    require_sha256,
)
from taim.judgments import TREC_CT_JUDGMENT_SCHEME, JudgmentScheme
from taim.schemas import (
    JsonValue,
    RelevanceJudgment,
    SchemaValidationError,
    json_value_to_builtins,
)
from taim.snapshot import FieldProvenance, field_provenance_from_payload

_SLUG = re.compile(r"\A[a-z0-9]+(?:[_-][a-z0-9]+)*\Z")


@dataclass(frozen=True, slots=True)
class EvaluationPackage(EvaluatorOnlyMaterial):
    """Evaluator-only Judgments and provenance, identified independently."""

    benchmark_lineage: str
    task_id: str
    snapshot_id: str
    judgments: tuple[RelevanceJudgment, ...]
    provenance: Mapping[str, JsonValue]
    judgment_scheme: JudgmentScheme = TREC_CT_JUDGMENT_SCHEME
    evaluation_package_id: str = field(init=False)

    schema_version = EVALUATION_PACKAGE_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.benchmark_lineage, "evaluation benchmark_lineage")
        require_non_empty(self.task_id, "evaluation task_id")
        require_sha256(self.snapshot_id, "evaluation snapshot_id")
        judgments = tuple(self.judgments)
        if not judgments or any(not isinstance(item, RelevanceJudgment) for item in judgments):
            raise SchemaValidationError("Evaluation Package must contain Judgments")
        if not isinstance(self.judgment_scheme, JudgmentScheme):
            raise SchemaValidationError("Evaluation Package requires a Judgment Scheme")
        for judgment in judgments:
            self.judgment_scheme.validate_label(judgment.label)
        keys = [(item.topic_id, item.trial_id) for item in judgments]
        if len(keys) != len(set(keys)):
            raise SchemaValidationError("Evaluation Package Judgments must be unique")
        if keys != sorted(keys):
            raise SchemaValidationError(
                "Evaluation Package Judgments must be deterministically ordered"
            )
        if not isinstance(self.provenance, Mapping):
            raise SchemaValidationError("evaluation provenance must be an object")
        provenance = freeze_json_mapping(self.provenance, path="evaluation provenance")
        object.__setattr__(self, "judgments", judgments)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "evaluation_package_id",
            content_sha256(self._identity_payload()),
        )

    def _identity_payload(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "benchmark_lineage": self.benchmark_lineage,
            "task_id": self.task_id,
            "snapshot_id": self.snapshot_id,
            "judgments": [item.to_dict() for item in self.judgments],
            "provenance": json_value_to_builtins(self.provenance),
        }
        if self.judgment_scheme != TREC_CT_JUDGMENT_SCHEME:
            payload["judgment_scheme"] = self.judgment_scheme.to_dict()
        return payload

    def manifest_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "benchmark_lineage": self.benchmark_lineage,
            "task_id": self.task_id,
            "snapshot_id": self.snapshot_id,
            "evaluation_package_id": self.evaluation_package_id,
            "judgment_count": len(self.judgments),
            "provenance": json_value_to_builtins(self.provenance),
        }
        if self.judgment_scheme != TREC_CT_JUDGMENT_SCHEME:
            payload["judgment_scheme"] = self.judgment_scheme.to_dict()
        return payload

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the complete evaluator-owned package artifact."""

        return {
            **self._identity_payload(),
            "evaluation_package_id": self.evaluation_package_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EvaluationPackage:
        """Load the complete artifact and reject noncanonical or stale schemas."""

        if payload.get("schema_version") != EVALUATION_PACKAGE_VERSION:
            raise SchemaValidationError("unsupported Evaluation Package schema_version")
        raw_judgments = payload.get("judgments")
        raw_provenance = payload.get("provenance")
        if not isinstance(raw_judgments, list) or any(
            not isinstance(judgment, Mapping) for judgment in raw_judgments
        ):
            raise SchemaValidationError("Evaluation Package Judgments must be an array of objects")
        if not isinstance(raw_provenance, Mapping):
            raise SchemaValidationError("Evaluation Package provenance must be an object")
        raw_scheme = payload.get("judgment_scheme")
        if raw_scheme is not None and not isinstance(raw_scheme, Mapping):
            raise SchemaValidationError("Evaluation Package Judgment Scheme must be an object")
        evaluation_package = cls(
            benchmark_lineage=cast(str, payload.get("benchmark_lineage")),
            task_id=cast(str, payload.get("task_id")),
            snapshot_id=cast(str, payload.get("snapshot_id")),
            judgments=tuple(
                RelevanceJudgment.from_dict(dict(cast(Mapping[str, object], judgment)))
                for judgment in raw_judgments
            ),
            provenance=cast(Mapping[str, JsonValue], raw_provenance),
            judgment_scheme=(
                JudgmentScheme.from_dict(cast(Mapping[str, object], raw_scheme))
                if raw_scheme is not None
                else TREC_CT_JUDGMENT_SCHEME
            ),
        )
        if dict(payload) != evaluation_package.to_dict():
            raise SchemaValidationError(
                "Evaluation Package serialized content does not match its identity"
            )
        return evaluation_package


@dataclass(frozen=True, slots=True)
class ConnectorDiagnostic:
    """Preparation/audit diagnostic, deliberately outside the Snapshot."""

    code: str
    severity: Literal["field_nonfatal", "record_fatal"]
    message: str
    record_id: str | None
    field: str | None
    provenance: tuple[FieldProvenance, ...] = ()

    schema_version = SNAPSHOT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        code = require_non_empty(self.code, "diagnostic code")
        if _SLUG.fullmatch(code) is None:
            raise SchemaValidationError("diagnostic code must be a lowercase slug")
        if self.severity not in {"field_nonfatal", "record_fatal"}:
            raise SchemaValidationError("diagnostic severity is unsupported")
        require_non_empty(self.message, "diagnostic message")
        if self.record_id is not None:
            require_non_empty(self.record_id, "diagnostic record_id")
        if self.field is not None:
            require_non_empty(self.field, "diagnostic field")
        provenance = tuple(self.provenance)
        if any(not isinstance(item, FieldProvenance) for item in provenance):
            raise SchemaValidationError(
                "diagnostic provenance must contain FieldProvenance objects"
            )
        object.__setattr__(self, "provenance", provenance)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "record_id": self.record_id,
            "field": self.field,
            "provenance": [item.to_dict() for item in self.provenance],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ConnectorDiagnostic:
        if payload.get("schema_version") != SNAPSHOT_CONTRACT_VERSION:
            raise SchemaValidationError("unsupported ConnectorDiagnostic schema_version")
        return cls(
            code=cast(str, payload.get("code")),
            severity=cast(Any, payload.get("severity")),
            message=cast(str, payload.get("message")),
            record_id=cast(str | None, payload.get("record_id")),
            field=cast(str | None, payload.get("field")),
            provenance=field_provenance_from_payload(
                payload.get("provenance"), field_name="diagnostic"
            ),
        )

    @classmethod
    def from_json(cls, serialized: str) -> ConnectorDiagnostic:
        payload = json.loads(serialized)
        if not isinstance(payload, dict):
            raise SchemaValidationError("ConnectorDiagnostic must be a JSON object")
        return cls.from_dict(payload)


__all__ = ["ConnectorDiagnostic", "EvaluationPackage"]
