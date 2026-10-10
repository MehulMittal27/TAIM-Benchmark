"""Immutable profile values and identity serialization shared by admission policies.

Research and release registries own their admission and evaluator projection rules. These values
carry admitted policies without deciding which corpus, receipt or method a registry may accept.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from taim.evaluation_package import EvaluationPackage
from taim.schemas import JsonValue, RelevanceJudgment
from taim.snapshot import BenchmarkTopic, Trial

if TYPE_CHECKING:
    from taim.data import PreparedBenchmark

UnjudgedPolicy = Literal["retain_as_zero_gain", "condense_before_cutoff"]
PrecisionDenominator = Literal["fixed_cutoff", "available_condensed_results", "maximum_graded_gain"]


@dataclass(frozen=True, slots=True)
class ProfileDefinition:
    """Validated static definition of one benchmark comparison profile."""

    profile_id: str
    profile_version: str
    corpus_policy: str
    expected_pool_count: int | None
    paper_pool_identity_status: str | None
    cutoffs: tuple[int, ...]
    discount: str
    unjudged_policy: UnjudgedPolicy
    gain_mapping: Mapping[str, int]
    precision_metric: str
    precision_relevance_minimum: int | None
    precision_denominator: PrecisionDenominator
    ranking_source: str
    ideal_policy: str
    topic_policy: str
    definition_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoffs", tuple(self.cutoffs))
        object.__setattr__(self, "gain_mapping", MappingProxyType(dict(self.gain_mapping)))

    def evaluation_configuration(self) -> dict[str, JsonValue]:
        """Return the complete JSON-safe evaluation policy."""

        return {
            "cutoffs": list(self.cutoffs),
            "discount": self.discount,
            "unjudged_policy": self.unjudged_policy,
            "gain_mapping": dict(self.gain_mapping),
            "precision_metric": self.precision_metric,
            "precision_relevance_minimum": self.precision_relevance_minimum,
            "precision_denominator": self.precision_denominator,
            "ranking_source": self.ranking_source,
            "ideal_policy": self.ideal_policy,
            "topic_policy": self.topic_policy,
        }


ProfileT = TypeVar("ProfileT", bound=ProfileDefinition)


@dataclass(frozen=True, slots=True)
class ResolvedBenchmarkValue(Generic[ProfileT]):
    """A prepared benchmark resolved to one immutable comparison corpus."""

    prepared: PreparedBenchmark
    profile: ProfileT
    trials: Sequence[Trial]
    corpus_hash: str
    pool_ids_hash: str
    pool_source_hash: str | None
    evaluation_package: EvaluationPackage = field(init=False)

    @property
    def name(self) -> str:
        return self.prepared.name

    @property
    def benchmark_lineage(self) -> str:
        return self.prepared.snapshot.benchmark_lineage

    @property
    def snapshot_contract_version(self) -> str:
        return self.prepared.snapshot.contract_version

    @property
    def prepared_snapshot_id(self) -> str:
        return self.prepared.snapshot.snapshot_id

    @property
    def evaluation_package_id(self) -> str:
        return self.evaluation_package.evaluation_package_id

    @property
    def dataset_id(self) -> str:
        return self.prepared.dataset_id

    @property
    def directory(self) -> Path:
        return self.prepared.directory

    @property
    def topics(self) -> tuple[BenchmarkTopic, ...]:
        return self.prepared.topics

    @property
    def judgments(self) -> tuple[RelevanceJudgment, ...]:
        return self.evaluation_package.judgments

    def profile_manifest_configuration(
        self, *, unverified_membership_acknowledged: bool
    ) -> dict[str, JsonValue]:
        """Return the complete profile provenance stored with every run."""

        return {
            "profile_id": self.profile.profile_id,
            "profile_version": self.profile.profile_version,
            "definition_sha256": self.profile.definition_sha256,
            "corpus_policy": self.profile.corpus_policy,
            "source_corpus_sha256": self.prepared.snapshot.logical_content_hashes()["trials"],
            "effective_corpus_sha256": self.corpus_hash,
            "effective_corpus_count": len(self.trials),
            "pool_ids_sha256": self.pool_ids_hash,
            "pool_source_sha256": self.pool_source_hash,
            "paper_pool_identity_status": self.profile.paper_pool_identity_status,
            "unverified_membership_acknowledged": unverified_membership_acknowledged,
            "evaluation": self.profile.evaluation_configuration(),
        }


def profile_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def profile_bytes_sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def profile_lines_sha256(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"
