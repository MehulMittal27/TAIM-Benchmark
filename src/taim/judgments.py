"""Versioned label semantics for evaluator-owned Judgments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from taim.contracts import content_sha256, require_exact_keys, require_non_empty, require_sha256
from taim.schemas import JsonValue, SchemaValidationError

JUDGMENT_SCHEME_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class JudgmentScheme:
    """Closed label meanings, gains, and relevance projections for one task."""

    scheme_id: str
    scheme_version: str
    labels: Mapping[int, str]
    gains: Mapping[int, int]
    relevance_sets: Mapping[str, frozenset[int]]
    reciprocal_rank_relevance_set: str
    definition_sha256: str = field(init=False)

    schema_version = JUDGMENT_SCHEME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.scheme_id, "Judgment Scheme scheme_id")
        require_non_empty(self.scheme_version, "Judgment Scheme scheme_version")
        labels = dict(sorted(self.labels.items()))
        gains = dict(sorted(self.gains.items()))
        if not labels or set(labels) != set(gains):
            raise SchemaValidationError(
                "Judgment Scheme labels and gains must cover the same labels"
            )
        for label, name in labels.items():
            if isinstance(label, bool) or not isinstance(label, int) or label < 0:
                raise SchemaValidationError("Judgment Scheme labels must be non-negative integers")
            require_non_empty(name, "Judgment Scheme label name")
        for gain in gains.values():
            if isinstance(gain, bool) or not isinstance(gain, int) or gain < 0:
                raise SchemaValidationError("Judgment Scheme gains must be non-negative integers")
        if 0 not in gains.values():
            raise SchemaValidationError("Judgment Scheme requires at least one zero-gain label")
        relevance_sets: dict[str, frozenset[int]] = {}
        for name, raw_labels in sorted(self.relevance_sets.items()):
            require_non_empty(name, "Judgment Scheme relevance-set name")
            relevance_labels = tuple(raw_labels)
            if any(
                isinstance(label, bool) or not isinstance(label, int) for label in relevance_labels
            ):
                raise SchemaValidationError(
                    "Judgment Scheme relevance sets must contain non-boolean integers"
                )
            selected = frozenset(relevance_labels)
            if not selected or not selected <= set(labels):
                raise SchemaValidationError(
                    "Judgment Scheme relevance sets must be non-empty label subsets"
                )
            relevance_sets[name] = selected
        if self.reciprocal_rank_relevance_set not in relevance_sets:
            raise SchemaValidationError(
                "Judgment Scheme reciprocal-rank set must name a declared relevance set"
            )
        object.__setattr__(self, "labels", MappingProxyType(labels))
        object.__setattr__(self, "gains", MappingProxyType(gains))
        object.__setattr__(self, "relevance_sets", MappingProxyType(relevance_sets))
        object.__setattr__(self, "definition_sha256", content_sha256(self._definition_payload()))

    def _definition_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "scheme_id": self.scheme_id,
            "scheme_version": self.scheme_version,
            "labels": {str(label): name for label, name in self.labels.items()},
            "gains": {str(label): gain for label, gain in self.gains.items()},
            "relevance_sets": {
                name: cast(JsonValue, sorted(labels))
                for name, labels in self.relevance_sets.items()
            },
            "reciprocal_rank_relevance_set": self.reciprocal_rank_relevance_set,
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {**self._definition_payload(), "definition_sha256": self.definition_sha256}

    def validate_label(self, label: object) -> int:
        if isinstance(label, bool) or not isinstance(label, int) or label not in self.labels:
            raise SchemaValidationError(
                f"Judgment label must be one of {sorted(self.labels)} under {self.scheme_id!r}"
            )
        return label

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> JudgmentScheme:
        require_exact_keys(
            payload,
            {
                "schema_version",
                "scheme_id",
                "scheme_version",
                "labels",
                "gains",
                "relevance_sets",
                "reciprocal_rank_relevance_set",
                "definition_sha256",
            },
            role="JudgmentScheme",
        )
        if payload["schema_version"] != cls.schema_version:
            raise SchemaValidationError("unsupported Judgment Scheme schema_version")
        raw_labels = payload["labels"]
        raw_gains = payload["gains"]
        raw_sets = payload["relevance_sets"]
        if (
            not isinstance(raw_labels, Mapping)
            or not isinstance(raw_gains, Mapping)
            or not isinstance(raw_sets, Mapping)
        ):
            raise SchemaValidationError("Judgment Scheme mappings must be objects")
        try:
            labels = {int(key): cast(str, value) for key, value in raw_labels.items()}
            gains = {int(key): cast(int, value) for key, value in raw_gains.items()}
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError("Judgment Scheme label keys must be integers") from exc
        relevance_sets: dict[str, frozenset[int]] = {}
        for name, values in raw_sets.items():
            if not isinstance(name, str) or not isinstance(values, list):
                raise SchemaValidationError("Judgment Scheme relevance sets are invalid")
            relevance_sets[name] = frozenset(cast(list[int], values))
        scheme = cls(
            scheme_id=cast(str, payload["scheme_id"]),
            scheme_version=cast(str, payload["scheme_version"]),
            labels=labels,
            gains=gains,
            relevance_sets=relevance_sets,
            reciprocal_rank_relevance_set=cast(str, payload["reciprocal_rank_relevance_set"]),
        )
        require_sha256(payload["definition_sha256"], "Judgment Scheme definition_sha256")
        if payload["definition_sha256"] != scheme.definition_sha256:
            raise SchemaValidationError("Judgment Scheme identity does not match its content")
        return scheme


TREC_CT_JUDGMENT_SCHEME = JudgmentScheme(
    scheme_id="trec-clinical-trials-2021-labels",
    scheme_version="1.0",
    labels={0: "not_relevant", 1: "excluded", 2: "eligible"},
    gains={0: 0, 1: 1, 2: 2},
    relevance_sets={
        "eligible": frozenset({2}),
        "relevant_or_eligible": frozenset({1, 2}),
    },
    reciprocal_rank_relevance_set="eligible",
)

SIGIR_CT_2016_JUDGMENT_SCHEME = JudgmentScheme(
    scheme_id="sigir-clinical-trials-2016-referral-labels",
    scheme_version="1.0",
    labels={
        0: "would_not_refer",
        1: "consider_referral_after_further_investigation",
        2: "highly_likely_to_refer",
    },
    gains={0: 0, 1: 1, 2: 2},
    relevance_sets={
        "highly_likely_referral": frozenset({2}),
        "referral_candidate": frozenset({1, 2}),
    },
    reciprocal_rank_relevance_set="highly_likely_referral",
)


__all__ = [
    "JUDGMENT_SCHEME_SCHEMA_VERSION",
    "SIGIR_CT_2016_JUDGMENT_SCHEME",
    "TREC_CT_JUDGMENT_SCHEME",
    "JudgmentScheme",
]
