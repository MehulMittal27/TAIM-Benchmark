"""Auditable corpus and evaluation profiles for prepared benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from taim.contracts import require_non_empty, require_sha256
from taim.evaluation_package import EvaluationPackage
from taim.schemas import JsonValue, RelevanceJudgment
from taim.snapshot import BenchmarkTopic, TrialDocument

if TYPE_CHECKING:
    from taim.data import PreparedBenchmark

OFFICIAL_FULL_PROFILE = "official-full"
CONFIRMATION_FULL_PROFILE = "confirmation-full"
SIGIR_DESCRIPTION_PROFILE = "description"
SIGIR_SUMMARY_PROFILE = "summary"
SYNTHETIC_PATIENT_TO_TRIAL_PROFILE = "synthetic-patient-to-trial"
# The public release ships this module, so it declares only the profiles whose definition files the
# release ships. A caller loads any other profile by passing its definition to
# load_profile_definition.
PROFILE_IDS = (
    CONFIRMATION_FULL_PROFILE,
    SIGIR_DESCRIPTION_PROFILE,
    OFFICIAL_FULL_PROFILE,
    SIGIR_SUMMARY_PROFILE,
    SYNTHETIC_PATIENT_TO_TRIAL_PROFILE,
)

_PROFILE_FILES = {
    CONFIRMATION_FULL_PROFILE: "confirmation-full-v1.json",
    SIGIR_DESCRIPTION_PROFILE: "description-v1.json",
    OFFICIAL_FULL_PROFILE: "official-full-v1.json",
    SIGIR_SUMMARY_PROFILE: "summary-v1.json",
    SYNTHETIC_PATIENT_TO_TRIAL_PROFILE: "synthetic-patient-to-trial-v1.json",
}
_TRIAL_ID = re.compile(r"NCT[0-9]{8}\Z")

CorpusPolicy = Literal["full_prepared_corpus", "user_locked_pool"]
UnjudgedPolicy = Literal["retain_as_zero_gain", "condense_before_cutoff"]
PrecisionDenominator = Literal["fixed_cutoff", "available_condensed_results", "maximum_graded_gain"]


@dataclass(frozen=True, slots=True)
class ProfileDefinitionPolicy:
    """The policy one profile definition file must state; its loader refuses a file that differs.

    ``precision`` is the precision metric, its relevance minimum and its denominator. ``corpus`` is
    the corpus policy, the expected pool count, the paper-pool identity status and the unjudged
    policy.
    """

    precision: tuple[str, int | None, PrecisionDenominator]
    corpus: tuple[CorpusPolicy, int | None, str | None, UnjudgedPolicy]
    cutoffs: tuple[int, ...] = (5, 10, 20)
    ranking_source: str = "primary"


_PROFILE_POLICIES = {
    CONFIRMATION_FULL_PROFILE: ProfileDefinitionPolicy(
        precision=(
            "eligible_precision",
            2,
            "fixed_cutoff",
        ),
        corpus=(
            "full_prepared_corpus",
            None,
            "confirmation-track-separate-protocol-required",
            "retain_as_zero_gain",
        ),
    ),
    SIGIR_DESCRIPTION_PROFILE: ProfileDefinitionPolicy(
        precision=(
            "referral_candidate_precision",
            1,
            "fixed_cutoff",
        ),
        corpus=(
            "full_prepared_corpus",
            None,
            "sigir-description-query-view",
            "retain_as_zero_gain",
        ),
    ),
    OFFICIAL_FULL_PROFILE: ProfileDefinitionPolicy(
        precision=("relevant_or_eligible_precision", 1, "fixed_cutoff"),
        corpus=("full_prepared_corpus", None, None, "retain_as_zero_gain"),
    ),
    SIGIR_SUMMARY_PROFILE: ProfileDefinitionPolicy(
        precision=(
            "referral_candidate_precision",
            1,
            "fixed_cutoff",
        ),
        corpus=(
            "full_prepared_corpus",
            None,
            "sigir-summary-query-view",
            "retain_as_zero_gain",
        ),
    ),
    SYNTHETIC_PATIENT_TO_TRIAL_PROFILE: ProfileDefinitionPolicy(
        precision=(
            "relevant_or_eligible_precision",
            1,
            "fixed_cutoff",
        ),
        corpus=(
            "full_prepared_corpus",
            None,
            "packaged-synthetic-conformance-only",
            "retain_as_zero_gain",
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class BenchmarkProfileEvaluationPolicy:
    """Validated evaluation policy shared by profile and scorecard consumers."""

    cutoffs: tuple[int, ...]
    discount: str
    unjudged_policy: UnjudgedPolicy
    gain_mapping: Mapping[str, int]
    precision_relevance_minimum: int | None
    precision_denominator: PrecisionDenominator
    ranking_source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoffs", tuple(self.cutoffs))
        object.__setattr__(self, "gain_mapping", MappingProxyType(dict(self.gain_mapping)))


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    """Validated static definition of one benchmark comparison profile."""

    profile_id: str
    profile_version: str
    corpus_policy: CorpusPolicy
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


@dataclass(frozen=True, slots=True)
class ResolvedBenchmark:
    """A prepared benchmark resolved to one immutable comparison corpus."""

    prepared: PreparedBenchmark
    profile: BenchmarkProfile
    trials: Sequence[TrialDocument]
    corpus_hash: str
    pool_ids_hash: str
    pool_source_hash: str | None
    unverified_membership_acknowledged: bool
    evaluation_package: EvaluationPackage = field(init=False)

    def __post_init__(self) -> None:
        if self.trials is self.prepared.trials:
            evaluation_package = self.prepared.evaluation_package
        else:
            effective_trial_ids = {trial.trial_id for trial in self.trials}
            evaluation_package = EvaluationPackage(
                benchmark_lineage=self.prepared.evaluation_package.benchmark_lineage,
                task_id=self.prepared.evaluation_package.task_id,
                snapshot_id=self.prepared.evaluation_package.snapshot_id,
                judgments=tuple(
                    judgment
                    for judgment in self.prepared.evaluation_package.judgments
                    if judgment.trial_id in effective_trial_ids
                ),
                provenance=self.prepared.evaluation_package.provenance,
                judgment_scheme=self.prepared.evaluation_package.judgment_scheme,
            )
        object.__setattr__(self, "evaluation_package", evaluation_package)

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

    def manifest_configuration(self) -> dict[str, JsonValue]:
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
            "unverified_membership_acknowledged": self.unverified_membership_acknowledged,
            "evaluation": self.profile.evaluation_configuration(),
        }


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def resolve_benchmark_profile_evaluation(
    benchmark_profile: Mapping[str, object],
) -> BenchmarkProfileEvaluationPolicy:
    """Validate and return one Benchmark Profile's shared evaluation policy."""

    require_non_empty(benchmark_profile.get("profile_id"), "benchmark_profile.profile_id")
    require_sha256(
        benchmark_profile.get("definition_sha256"),
        "benchmark_profile.definition_sha256",
    )
    evaluation = benchmark_profile.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("benchmark_profile evaluation policy is missing")
    cutoffs = evaluation.get("cutoffs")
    if (
        not isinstance(cutoffs, list)
        or not cutoffs
        or any(
            isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1
            for cutoff in cutoffs
        )
        or cutoffs != sorted(set(cutoffs))
    ):
        raise ValueError("benchmark profile cutoffs must be sorted unique positive integers")
    gain_mapping = evaluation.get("gain_mapping")
    if gain_mapping is not None and (
        not isinstance(gain_mapping, Mapping)
        or dict(gain_mapping) != {"0": 0, "1": 1, "2": 2}
        or any(isinstance(value, bool) for value in gain_mapping.values())
    ):
        raise ValueError("benchmark profile gain_mapping must be linear labels 0/1/2")
    discount = evaluation.get("discount")
    if discount is not None and discount != "log2(rank + 1)":
        raise ValueError("benchmark profile discount must be log2(rank + 1)")
    unjudged = evaluation.get("unjudged_policy")
    if unjudged not in ("retain_as_zero_gain", "condense_before_cutoff"):
        raise ValueError("benchmark profile unjudged_policy is unsupported")
    denominator = evaluation.get("precision_denominator")
    if denominator not in (
        "fixed_cutoff",
        "available_condensed_results",
        "maximum_graded_gain",
    ):
        raise ValueError("benchmark profile precision_denominator is unsupported")
    relevance_minimum = evaluation.get("precision_relevance_minimum")
    if relevance_minimum is not None and (
        isinstance(relevance_minimum, bool)
        or not isinstance(relevance_minimum, int)
        or relevance_minimum not in (1, 2)
    ):
        raise ValueError("benchmark profile precision_relevance_minimum must be 1, 2, or null")
    if (relevance_minimum is None) != (denominator == "maximum_graded_gain"):
        raise ValueError(
            "maximum_graded_gain requires null precision_relevance_minimum and vice versa"
        )
    ranking_source = evaluation.get("ranking_source", "primary")
    if not isinstance(ranking_source, str) or (
        ranking_source != "primary"
        and (not ranking_source.startswith("stage:") or not ranking_source.removeprefix("stage:"))
    ):
        raise ValueError("benchmark profile ranking_source is unsupported")
    return BenchmarkProfileEvaluationPolicy(
        cutoffs=tuple(cast(list[int], cutoffs)),
        discount=cast(str, discount or "log2(rank + 1)"),
        unjudged_policy=cast(UnjudgedPolicy, unjudged),
        gain_mapping=cast(Mapping[str, int], gain_mapping or {"0": 0, "1": 1, "2": 2}),
        precision_relevance_minimum=cast(int | None, relevance_minimum),
        precision_denominator=cast(PrecisionDenominator, denominator),
        ranking_source=ranking_source,
    )


def _require_exact_keys(payload: dict[str, object], expected: set[str], *, role: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unexpected = sorted(set(payload) - expected)
        raise ValueError(f"{role} fields mismatch; missing={missing}, unexpected={unexpected}")


def load_profile(profile_id: str) -> BenchmarkProfile:
    """Load and strictly validate a packaged profile definition."""

    try:
        filename = _PROFILE_FILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark profile {profile_id!r}") from exc
    path = Path(__file__).parent / "data" / "profiles" / filename
    return load_profile_definition(
        path, profile_id=profile_id, policy=_PROFILE_POLICIES[profile_id]
    )


def load_profile_definition(
    path: Path, *, profile_id: str, policy: ProfileDefinitionPolicy
) -> BenchmarkProfile:
    """Load and strictly validate one profile definition file against the policy it must state."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid benchmark profile JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("benchmark profile must contain a JSON object")
    _require_exact_keys(
        payload,
        {
            "profile_id",
            "profile_version",
            "corpus_policy",
            "expected_pool_count",
            "paper_pool_identity_status",
            "evaluation",
        },
        role="benchmark profile",
    )
    evaluation = payload["evaluation"]
    if not isinstance(evaluation, dict):
        raise ValueError("benchmark profile evaluation must be a JSON object")
    _require_exact_keys(
        evaluation,
        {
            "cutoffs",
            "discount",
            "unjudged_policy",
            "gain_mapping",
            "precision_metric",
            "precision_relevance_minimum",
            "precision_denominator",
            "ideal_policy",
            "topic_policy",
            "ranking_source",
        },
        role="benchmark profile evaluation",
    )
    definition_sha256 = _sha256_bytes(_canonical_json_bytes(payload))
    evaluation_policy = resolve_benchmark_profile_evaluation(
        {
            "profile_id": payload["profile_id"],
            "definition_sha256": definition_sha256,
            "evaluation": evaluation,
        }
    )
    if evaluation_policy.cutoffs != policy.cutoffs:
        raise ValueError("packaged benchmark profile cutoffs must be " + str(list(policy.cutoffs)))
    if payload["profile_id"] != profile_id:
        raise ValueError("benchmark profile ID does not match its packaged filename")
    if payload["profile_version"] != "1.0":
        raise ValueError("unsupported benchmark profile version")
    corpus_policy = payload["corpus_policy"]
    if corpus_policy not in ("full_prepared_corpus", "user_locked_pool"):
        raise ValueError("unsupported benchmark profile corpus_policy")
    expected_count = payload["expected_pool_count"]
    if expected_count is not None and (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise ValueError("expected_pool_count must be null or a positive integer")
    unjudged_policy = evaluation_policy.unjudged_policy
    denominator = evaluation_policy.precision_denominator
    relevance_minimum = evaluation_policy.precision_relevance_minimum
    if evaluation["ideal_policy"] != "all_topic_qrels":
        raise ValueError("unsupported benchmark profile ideal_policy")
    if evaluation["topic_policy"] != "all_declared_topics":
        raise ValueError("unsupported benchmark profile topic_policy")
    if evaluation["ranking_source"] != policy.ranking_source:
        raise ValueError("benchmark profile ranking_source is inconsistent")
    actual_precision = (
        evaluation["precision_metric"],
        relevance_minimum,
        denominator,
    )
    if actual_precision != policy.precision:
        raise ValueError("benchmark profile precision policy is inconsistent")
    if (corpus_policy, expected_count, payload["paper_pool_identity_status"], unjudged_policy) != (
        policy.corpus
    ):
        raise ValueError("benchmark profile corpus or unjudged policy is inconsistent")
    identity_status = payload["paper_pool_identity_status"]
    if identity_status is not None and (
        not isinstance(identity_status, str) or not identity_status
    ):
        raise ValueError("paper_pool_identity_status must be null or non-empty")
    return BenchmarkProfile(
        profile_id=profile_id,
        profile_version="1.0",
        corpus_policy=cast(CorpusPolicy, corpus_policy),
        expected_pool_count=cast(int | None, expected_count),
        paper_pool_identity_status=cast(str | None, identity_status),
        cutoffs=evaluation_policy.cutoffs,
        discount=evaluation_policy.discount,
        unjudged_policy=unjudged_policy,
        gain_mapping=evaluation_policy.gain_mapping,
        precision_metric=cast(str, evaluation["precision_metric"]),
        precision_relevance_minimum=cast(int | None, relevance_minimum),
        precision_denominator=denominator,
        ranking_source=evaluation_policy.ranking_source,
        ideal_policy=cast(str, evaluation["ideal_policy"]),
        topic_policy=cast(str, evaluation["topic_policy"]),
        definition_sha256=definition_sha256,
    )


def _load_trial_pool(path: Path, *, expected_count: int) -> tuple[tuple[str, ...], str]:
    try:
        source_bytes = path.read_bytes()
        text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("trial pool must be UTF-8 text") from exc
    lines = text.splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"trial pool must contain exactly {expected_count} IDs; found {len(lines)}"
        )
    for line_number, trial_id in enumerate(lines, start=1):
        if not trial_id or trial_id != trial_id.strip():
            raise ValueError(f"trial pool line {line_number} is blank or contains whitespace")
        if _TRIAL_ID.fullmatch(trial_id) is None:
            raise ValueError(f"trial pool line {line_number} is not canonical: {trial_id!r}")
    if len(lines) != len(set(lines)):
        raise ValueError("trial pool must not contain duplicate IDs")
    if lines != sorted(lines):
        raise ValueError("trial pool IDs must be in canonical ascending order")
    return tuple(lines), _sha256_bytes(source_bytes)


def resolve_benchmark_profile(
    prepared: PreparedBenchmark,
    *,
    profile_id: str = OFFICIAL_FULL_PROFILE,
    trial_pool_file: str | Path | None = None,
    allow_unverified_paper_pool: bool = False,
) -> ResolvedBenchmark:
    """Resolve and validate a prepared benchmark against one static profile."""

    return resolve_loaded_profile(
        prepared,
        load_profile(profile_id),
        trial_pool_file=trial_pool_file,
        allow_unverified_paper_pool=allow_unverified_paper_pool,
    )


def resolve_loaded_profile(
    prepared: PreparedBenchmark,
    profile: BenchmarkProfile,
    *,
    trial_pool_file: str | Path | None = None,
    allow_unverified_paper_pool: bool = False,
) -> ResolvedBenchmark:
    """Resolve and validate a prepared benchmark against one loaded profile."""

    prepared_trial_ids = tuple(trial.trial_id for trial in prepared.trials)
    if profile.corpus_policy == "full_prepared_corpus":
        if trial_pool_file is not None:
            raise ValueError("--trial-pool-file is not valid with profile official-full")
        if allow_unverified_paper_pool:
            raise ValueError(
                "--allow-unverified-paper-pool is not valid with profile official-full"
            )
        return ResolvedBenchmark(
            prepared=prepared,
            profile=profile,
            trials=prepared.trials,
            corpus_hash=prepared.snapshot.logical_content_hashes()["trials"],
            pool_ids_hash=_sha256_lines(prepared_trial_ids),
            pool_source_hash=None,
            unverified_membership_acknowledged=False,
        )

    if trial_pool_file is None:
        raise ValueError(
            f"profile {profile.profile_id} requires --trial-pool-file with exactly "
            f"{profile.expected_pool_count} canonical IDs"
        )
    if not allow_unverified_paper_pool:
        raise ValueError(
            f"{profile.profile_id} membership is not publicly verifiable; pass "
            "--allow-unverified-paper-pool to acknowledge that results are paper-shaped, "
            "not paper-exact"
        )
    if profile.expected_pool_count is None:
        raise ValueError("user-locked pool profile is missing expected_pool_count")
    pool_ids, pool_source_hash = _load_trial_pool(
        Path(trial_pool_file), expected_count=profile.expected_pool_count
    )
    prepared_id_set = set(prepared_trial_ids)
    judged_trial_ids = {judgment.trial_id for judgment in prepared.evaluation_package.judgments}
    missing_from_corpus = sorted(set(pool_ids) - prepared_id_set)
    if missing_from_corpus:
        raise ValueError(f"trial pool ID is absent from prepared corpus: {missing_from_corpus[0]}")
    missing_from_judgments = sorted(set(pool_ids) - judged_trial_ids)
    if missing_from_judgments:
        raise ValueError(
            f"trial pool ID is absent from the Evaluation Package: {missing_from_judgments[0]}"
        )
    pool_id_set = set(pool_ids)
    selected_trials = tuple(trial for trial in prepared.trials if trial.trial_id in pool_id_set)
    if len(selected_trials) != profile.expected_pool_count:
        raise ValueError("resolved trial pool count does not match profile")
    corpus_hash = _sha256_lines(trial.to_json() for trial in selected_trials)
    return ResolvedBenchmark(
        prepared=prepared,
        profile=profile,
        trials=selected_trials,
        corpus_hash=corpus_hash,
        pool_ids_hash=_sha256_lines(pool_ids),
        pool_source_hash=pool_source_hash,
        unverified_membership_acknowledged=True,
    )


__all__ = [
    "CONFIRMATION_FULL_PROFILE",
    "OFFICIAL_FULL_PROFILE",
    "PROFILE_IDS",
    "SIGIR_DESCRIPTION_PROFILE",
    "SIGIR_SUMMARY_PROFILE",
    "BenchmarkProfile",
    "BenchmarkProfileEvaluationPolicy",
    "ProfileDefinitionPolicy",
    "ResolvedBenchmark",
    "load_profile",
    "load_profile_definition",
    "resolve_benchmark_profile",
    "resolve_benchmark_profile_evaluation",
    "resolve_loaded_profile",
]
