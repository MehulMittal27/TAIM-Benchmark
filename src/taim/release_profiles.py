"""Closed Benchmark Profile registry for public Benchmark Release 0.1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from taim.contracts import content_sha256, require_sha256
from taim.evaluation_package import EvaluationPackage
from taim.schemas import JsonValue, RelevanceJudgment
from taim.snapshot import BenchmarkTopic, TrialDocument

if TYPE_CHECKING:
    from taim.data import PreparedBenchmark

CONFIRMATION_FULL_PROFILE = "confirmation-full"
OFFICIAL_FULL_PROFILE = "official-full"
SIGIR_DESCRIPTION_PROFILE = "description"
SIGIR_SUMMARY_PROFILE = "summary"
SYNTHETIC_PATIENT_TO_TRIAL_PROFILE = "synthetic-patient-to-trial"
EXTERNAL_FIDELITY_TREC_2021_PROFILE = "trec-ct-2021-external-fidelity-26149"
SIGIR_DESCRIPTION_JUDGMENT_UNION_PROFILE = "sigir-ct-2016-description-judgment-union"
SIGIR_SUMMARY_JUDGMENT_UNION_PROFILE = "sigir-ct-2016-summary-judgment-union"
TREC_2021_JUDGMENT_UNION_PROFILE = "trec-ct-2021-judgment-union"
TREC_2022_JUDGMENT_UNION_PROFILE = "trec-ct-2022-judgment-union"
TREC_2023_JUDGMENT_UNION_PROFILE = "trec-ct-2023-judgment-union"
PROFILE_IDS = (
    CONFIRMATION_FULL_PROFILE,
    SIGIR_DESCRIPTION_PROFILE,
    SIGIR_DESCRIPTION_JUDGMENT_UNION_PROFILE,
    OFFICIAL_FULL_PROFILE,
    SIGIR_SUMMARY_PROFILE,
    SIGIR_SUMMARY_JUDGMENT_UNION_PROFILE,
    SYNTHETIC_PATIENT_TO_TRIAL_PROFILE,
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
    TREC_2021_JUDGMENT_UNION_PROFILE,
    TREC_2022_JUDGMENT_UNION_PROFILE,
    TREC_2023_JUDGMENT_UNION_PROFILE,
)

_PROFILE_FILES = {
    CONFIRMATION_FULL_PROFILE: "confirmation-full-v1.json",
    SIGIR_DESCRIPTION_PROFILE: "description-v1.json",
    SIGIR_DESCRIPTION_JUDGMENT_UNION_PROFILE: ("sigir-ct-2016-description-judgment-union-v1.json"),
    OFFICIAL_FULL_PROFILE: "official-full-v1.json",
    SIGIR_SUMMARY_PROFILE: "summary-v1.json",
    SIGIR_SUMMARY_JUDGMENT_UNION_PROFILE: "sigir-ct-2016-summary-judgment-union-v1.json",
    SYNTHETIC_PATIENT_TO_TRIAL_PROFILE: "synthetic-patient-to-trial-v1.json",
    EXTERNAL_FIDELITY_TREC_2021_PROFILE: "trec-ct-2021-external-fidelity-26149-v1.json",
    TREC_2021_JUDGMENT_UNION_PROFILE: "trec-ct-2021-judgment-union-v1.json",
    TREC_2022_JUDGMENT_UNION_PROFILE: "trec-ct-2022-judgment-union-v1.json",
    TREC_2023_JUDGMENT_UNION_PROFILE: "trec-ct-2023-judgment-union-v1.json",
}
_PRECISION_METRICS = {
    CONFIRMATION_FULL_PROFILE: "eligible_precision",
    SIGIR_DESCRIPTION_PROFILE: "referral_candidate_precision",
    SIGIR_DESCRIPTION_JUDGMENT_UNION_PROFILE: "referral_candidate_precision",
    OFFICIAL_FULL_PROFILE: "relevant_or_eligible_precision",
    SIGIR_SUMMARY_PROFILE: "referral_candidate_precision",
    SIGIR_SUMMARY_JUDGMENT_UNION_PROFILE: "referral_candidate_precision",
    SYNTHETIC_PATIENT_TO_TRIAL_PROFILE: "relevant_or_eligible_precision",
    EXTERNAL_FIDELITY_TREC_2021_PROFILE: "relevant_or_eligible_precision",
    TREC_2021_JUDGMENT_UNION_PROFILE: "relevant_or_eligible_precision",
    TREC_2022_JUDGMENT_UNION_PROFILE: "eligible_precision",
    TREC_2023_JUDGMENT_UNION_PROFILE: "eligible_precision",
}
_PROFILE_STATUS = {
    CONFIRMATION_FULL_PROFILE: "confirmation-track-separate-protocol-required",
    SIGIR_DESCRIPTION_PROFILE: "sigir-description-query-view",
    SIGIR_DESCRIPTION_JUDGMENT_UNION_PROFILE: "judgment-union-within-pool-only",
    OFFICIAL_FULL_PROFILE: None,
    SIGIR_SUMMARY_PROFILE: "sigir-summary-query-view",
    SIGIR_SUMMARY_JUDGMENT_UNION_PROFILE: "judgment-union-within-pool-only",
    SYNTHETIC_PATIENT_TO_TRIAL_PROFILE: "packaged-synthetic-conformance-only",
    EXTERNAL_FIDELITY_TREC_2021_PROFILE: "caller-supplied-membership-within-pool-only",
    TREC_2021_JUDGMENT_UNION_PROFILE: "judgment-union-within-pool-only",
    TREC_2022_JUDGMENT_UNION_PROFILE: "judgment-union-within-pool-only",
    TREC_2023_JUDGMENT_UNION_PROFILE: "judgment-union-within-pool-only",
}
_PRECISION_RELEVANCE_MINIMUM = {
    CONFIRMATION_FULL_PROFILE: 2,
    SIGIR_DESCRIPTION_PROFILE: 1,
    SIGIR_DESCRIPTION_JUDGMENT_UNION_PROFILE: 1,
    OFFICIAL_FULL_PROFILE: 1,
    SIGIR_SUMMARY_PROFILE: 1,
    SIGIR_SUMMARY_JUDGMENT_UNION_PROFILE: 1,
    SYNTHETIC_PATIENT_TO_TRIAL_PROFILE: 1,
    EXTERNAL_FIDELITY_TREC_2021_PROFILE: 1,
    TREC_2021_JUDGMENT_UNION_PROFILE: 1,
    TREC_2022_JUDGMENT_UNION_PROFILE: 2,
    TREC_2023_JUDGMENT_UNION_PROFILE: 2,
}
_CORPUS_POLICIES = {
    profile_id: (
        "external_fidelity_pool"
        if profile_id == EXTERNAL_FIDELITY_TREC_2021_PROFILE
        else "judgment_union"
        if profile_id.endswith("-judgment-union")
        else "full_prepared_corpus"
    )
    for profile_id in PROFILE_IDS
}
_EXPECTED_POOL_COUNTS = {
    profile_id: (
        26_162
        if profile_id == TREC_2021_JUDGMENT_UNION_PROFILE
        else 26_149
        if profile_id == EXTERNAL_FIDELITY_TREC_2021_PROFILE
        else None
    )
    for profile_id in PROFILE_IDS
}

CorpusPolicy = Literal["external_fidelity_pool", "full_prepared_corpus", "judgment_union"]
UnjudgedPolicy = Literal["retain_as_zero_gain"]
PrecisionDenominator = Literal["fixed_cutoff"]


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    """Validated static definition of one public Release 0.1 Profile."""

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
    precision_relevance_minimum: int
    precision_denominator: PrecisionDenominator
    ranking_source: str
    ideal_policy: str
    topic_policy: str
    definition_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoffs", tuple(self.cutoffs))
        object.__setattr__(self, "gain_mapping", MappingProxyType(dict(self.gain_mapping)))

    def evaluation_configuration(self) -> dict[str, JsonValue]:
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
    """One full prepared corpus resolved under a public Release 0.1 Profile."""

    prepared: PreparedBenchmark
    profile: BenchmarkProfile
    trials: Sequence[TrialDocument]
    corpus_hash: str
    pool_ids_hash: str
    pool_source_hash: str | None
    pool_receipt_id: str | None
    unverified_membership_acknowledged: Literal[False]
    evaluation_package: EvaluationPackage = field(init=False)

    def __post_init__(self) -> None:
        if self.profile.corpus_policy == "full_prepared_corpus":
            if self.trials != self.prepared.trials:
                raise ValueError("full-corpus Release 0.1 Profile omitted prepared trials")
        elif self.profile.corpus_policy == "judgment_union":
            expected = {
                judgment.trial_id for judgment in self.prepared.evaluation_package.judgments
            }
            observed = {trial.trial_id for trial in self.trials}
            if observed != expected:
                raise ValueError(
                    "judgment-union Release 0.1 Profile must contain every and only judged trial"
                )
        else:
            prepared_ids = {trial.trial_id for trial in self.prepared.trials}
            judged_ids = {
                judgment.trial_id for judgment in self.prepared.evaluation_package.judgments
            }
            observed = {trial.trial_id for trial in self.trials}
            if not observed <= prepared_ids or not observed <= judged_ids:
                raise ValueError(
                    "external-fidelity Release 0.1 Profile must contain only prepared, judged "
                    "trials"
                )
            if self.profile.expected_pool_count != len(self.trials):
                raise ValueError(
                    f"{self.profile.profile_id} requires exactly "
                    f"{self.profile.expected_pool_count} trials"
                )
        evaluation_package = (
            _project_evaluation_package(self.prepared, self.profile, self.trials)
            if self.profile.corpus_policy == "external_fidelity_pool"
            else self.prepared.evaluation_package
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
            "pool_receipt_id": self.pool_receipt_id,
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


def ordered_trial_ids_sha256(trial_ids: Iterable[str]) -> str:
    """Hash canonically ordered trial IDs for a judgment-union receipt."""

    return _sha256_lines(trial_ids)


def judgment_union_pool_receipt(
    *,
    selection_policy: str,
    claim_scope: str,
    profile: str,
    profile_definition_sha256: str,
    prepared_snapshot_id: str,
    evaluation_package_id: str,
    pool_count: int,
    pool_ids_sha256: str,
) -> dict[str, JsonValue]:
    """Return one content-addressed structural judgment-union receipt."""

    core: dict[str, JsonValue] = {
        "selection_policy": selection_policy,
        "claim_scope": claim_scope,
        "profile": profile,
        "profile_definition_sha256": profile_definition_sha256,
        "prepared_snapshot_id": prepared_snapshot_id,
        "evaluation_package_id": evaluation_package_id,
        "pool_count": pool_count,
        "pool_ids_sha256": pool_ids_sha256,
    }
    return {**core, "pool_receipt_id": content_sha256(core)}


def pool_receipt_policy(profile_id: str) -> tuple[str, str]:
    """Return the exact selection and claim policies bound into a pool receipt."""

    if profile_id == EXTERNAL_FIDELITY_TREC_2021_PROFILE:
        return (
            "caller_supplied_external_fidelity_membership",
            "external_system_fidelity_within_pool_not_full_corpus_retrieval",
        )
    if profile_id.endswith("-judgment-union"):
        return (
            "union_of_trial_ids_present_in_the_evaluation_package",
            "within_pool_ranking_not_full_corpus_retrieval",
        )
    raise ValueError(f"Profile {profile_id!r} does not define a pool receipt")


def _project_evaluation_package(
    prepared: PreparedBenchmark,
    profile: BenchmarkProfile,
    trials: Sequence[TrialDocument],
) -> EvaluationPackage:
    """Project evaluator material to one strict-subset Task Input."""

    trial_ids = {trial.trial_id for trial in trials}
    source = prepared.evaluation_package
    judgments = tuple(judgment for judgment in source.judgments if judgment.trial_id in trial_ids)
    if {judgment.trial_id for judgment in judgments} != trial_ids:
        raise ValueError("external-fidelity Task Input lacks complete evaluator membership")
    return EvaluationPackage(
        benchmark_lineage=source.benchmark_lineage,
        task_id=source.task_id,
        snapshot_id=source.snapshot_id,
        judgments=judgments,
        provenance={
            **source.provenance,
            "release_profile_projection": {
                "profile_id": profile.profile_id,
                "source_evaluation_package_id": source.evaluation_package_id,
                "selection_policy": pool_receipt_policy(profile.profile_id)[0],
            },
        },
        judgment_scheme=source.judgment_scheme,
    )


def _judgment_union(
    prepared: PreparedBenchmark,
    profile: BenchmarkProfile,
) -> tuple[tuple[TrialDocument, ...], str]:
    if profile.corpus_policy != "judgment_union":
        raise ValueError(f"Profile {profile.profile_id!r} is not a judgment-union Profile")
    judged_trial_ids = {judgment.trial_id for judgment in prepared.evaluation_package.judgments}
    selected_trials = tuple(
        trial for trial in prepared.trials if trial.trial_id in judged_trial_ids
    )
    if len(selected_trials) != len(judged_trial_ids):
        raise ValueError("Evaluation Package references a trial outside the prepared corpus")
    if (
        profile.expected_pool_count is not None
        and len(selected_trials) != profile.expected_pool_count
    ):
        raise ValueError(
            f"{profile.profile_id} requires exactly {profile.expected_pool_count} "
            "judged-union trials"
        )
    return selected_trials, _sha256_lines(trial.trial_id for trial in selected_trials)


def inspect_judgment_union_profile(
    prepared: PreparedBenchmark,
    *,
    profile_id: str,
) -> dict[str, JsonValue]:
    """Return the receipt that must be frozen before a judgment-union effectiveness run."""

    profile = load_profile(profile_id)
    selected_trials, pool_ids_hash = _judgment_union(prepared, profile)
    return judgment_union_pool_receipt(
        selection_policy="union_of_trial_ids_present_in_the_evaluation_package",
        claim_scope="within_pool_ranking_not_full_corpus_retrieval",
        profile=profile.profile_id,
        profile_definition_sha256=profile.definition_sha256,
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        evaluation_package_id=prepared.evaluation_package.evaluation_package_id,
        pool_count=len(selected_trials),
        pool_ids_sha256=pool_ids_hash,
    )


def _external_fidelity_pool(
    prepared: PreparedBenchmark,
    profile: BenchmarkProfile,
    supplied_pool_ids: Iterable[str],
) -> tuple[tuple[TrialDocument, ...], str]:
    if profile.corpus_policy != "external_fidelity_pool":
        raise ValueError(f"Profile {profile.profile_id!r} is not an external-fidelity Profile")
    requested = tuple(supplied_pool_ids)
    if not requested or any(
        not isinstance(trial_id, str) or not trial_id for trial_id in requested
    ):
        raise ValueError("external-fidelity pool IDs must be non-empty strings")
    if len(requested) != len(set(requested)):
        raise ValueError("external-fidelity pool contains a duplicate trial ID")
    prepared_by_id = {trial.trial_id: trial for trial in prepared.trials}
    unknown = sorted(set(requested) - set(prepared_by_id))
    if unknown:
        raise ValueError(f"external-fidelity pool contains unknown trial ID {unknown[0]!r}")
    judged_ids = {judgment.trial_id for judgment in prepared.evaluation_package.judgments}
    unjudged = sorted(set(requested) - judged_ids)
    if unjudged:
        raise ValueError(
            f"external-fidelity pool contains trial ID {unjudged[0]!r} with no supplied judgment"
        )
    requested_set = set(requested)
    selected = tuple(trial for trial in prepared.trials if trial.trial_id in requested_set)
    if profile.expected_pool_count is not None and len(selected) != profile.expected_pool_count:
        raise ValueError(
            f"{profile.profile_id} requires exactly {profile.expected_pool_count} trials"
        )
    return selected, _sha256_lines(trial.trial_id for trial in selected)


def inspect_external_fidelity_profile(
    prepared: PreparedBenchmark,
    *,
    profile_id: str,
    supplied_pool_ids: Iterable[str],
    expected_pool_count: int,
) -> dict[str, JsonValue]:
    """Return a receipt for caller-supplied, externally authored pool membership."""

    profile = load_profile(profile_id)
    selected_trials, pool_ids_hash = _external_fidelity_pool(prepared, profile, supplied_pool_ids)
    if expected_pool_count != len(selected_trials):
        raise ValueError("caller-supplied pool membership does not match --pool-count")
    evaluation_package = _project_evaluation_package(prepared, profile, selected_trials)
    selection_policy, claim_scope = pool_receipt_policy(profile.profile_id)
    return judgment_union_pool_receipt(
        selection_policy=selection_policy,
        claim_scope=claim_scope,
        profile=profile.profile_id,
        profile_definition_sha256=profile.definition_sha256,
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        evaluation_package_id=evaluation_package.evaluation_package_id,
        pool_count=len(selected_trials),
        pool_ids_sha256=pool_ids_hash,
    )


def load_profile(profile_id: str) -> BenchmarkProfile:
    """Load one public Profile and reject any non-release registry entry."""

    try:
        filename = _PROFILE_FILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown public Release 0.1 Profile {profile_id!r}") from exc
    path = Path(__file__).parent / "data" / "profiles" / filename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid public Benchmark Profile: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "profile_id",
        "profile_version",
        "corpus_policy",
        "expected_pool_count",
        "paper_pool_identity_status",
        "evaluation",
    }:
        raise ValueError("public Benchmark Profile fields do not match the release contract")
    evaluation = payload["evaluation"]
    expected_evaluation = {
        "cutoffs": [5, 10, 20],
        "discount": "log2(rank + 1)",
        "unjudged_policy": "retain_as_zero_gain",
        "gain_mapping": {"0": 0, "1": 1, "2": 2},
        "precision_metric": _PRECISION_METRICS[profile_id],
        "precision_relevance_minimum": _PRECISION_RELEVANCE_MINIMUM[profile_id],
        "precision_denominator": "fixed_cutoff",
        "ideal_policy": "all_topic_qrels",
        "topic_policy": "all_declared_topics",
        "ranking_source": "primary",
    }
    if (
        payload["profile_id"] != profile_id
        or payload["profile_version"] != "1.0"
        or payload["corpus_policy"] != _CORPUS_POLICIES[profile_id]
        or payload["expected_pool_count"] != _EXPECTED_POOL_COUNTS[profile_id]
        or payload["paper_pool_identity_status"] != _PROFILE_STATUS[profile_id]
        or evaluation != expected_evaluation
    ):
        raise ValueError("public Benchmark Profile policy is inconsistent")
    return BenchmarkProfile(
        profile_id=profile_id,
        profile_version="1.0",
        corpus_policy=cast(CorpusPolicy, payload["corpus_policy"]),
        expected_pool_count=cast(int | None, payload["expected_pool_count"]),
        paper_pool_identity_status=cast(str | None, payload["paper_pool_identity_status"]),
        cutoffs=(5, 10, 20),
        discount="log2(rank + 1)",
        unjudged_policy="retain_as_zero_gain",
        gain_mapping={"0": 0, "1": 1, "2": 2},
        precision_metric=_PRECISION_METRICS[profile_id],
        precision_relevance_minimum=_PRECISION_RELEVANCE_MINIMUM[profile_id],
        precision_denominator="fixed_cutoff",
        ranking_source="primary",
        ideal_policy="all_topic_qrels",
        topic_policy="all_declared_topics",
        definition_sha256=_sha256_bytes(_canonical_json_bytes(payload)),
    )


def resolve_benchmark_profile(
    prepared: PreparedBenchmark,
    *,
    profile_id: str = OFFICIAL_FULL_PROFILE,
    expected_pool_count: int | None = None,
    expected_pool_ids_sha256: str | None = None,
    expected_pool_receipt_id: str | None = None,
    supplied_pool_ids: Iterable[str] | None = None,
) -> ResolvedBenchmark:
    """Resolve a prepared benchmark to one full or frozen compute-bounded corpus."""

    profile = load_profile(profile_id)
    trial_ids = tuple(trial.trial_id for trial in prepared.trials)
    if profile.corpus_policy == "full_prepared_corpus":
        if any(
            value is not None
            for value in (
                expected_pool_count,
                expected_pool_ids_sha256,
                expected_pool_receipt_id,
                supplied_pool_ids,
            )
        ):
            raise ValueError("full-corpus Profiles do not accept judgment-union pool pins")
        return ResolvedBenchmark(
            prepared=prepared,
            profile=profile,
            trials=prepared.trials,
            corpus_hash=prepared.snapshot.logical_content_hashes()["trials"],
            pool_ids_hash=_sha256_lines(trial_ids),
            pool_source_hash=None,
            pool_receipt_id=None,
            unverified_membership_acknowledged=False,
        )

    if (
        expected_pool_count is None
        or expected_pool_ids_sha256 is None
        or expected_pool_receipt_id is None
    ):
        raise ValueError(
            "judgment-union Profile requires --pool-count, --pool-ids-sha256, and "
            "--pool-receipt-id from a pre-effectiveness pool inspection"
        )
    if isinstance(expected_pool_count, bool) or expected_pool_count < 1:
        raise ValueError("--pool-count must be a positive integer")
    require_sha256(expected_pool_ids_sha256, "judgment-union pool IDs SHA-256")
    if profile.corpus_policy == "external_fidelity_pool":
        if supplied_pool_ids is None:
            raise ValueError("external-fidelity Profile requires caller-supplied pool membership")
        selected_trials, pool_ids_hash = _external_fidelity_pool(
            prepared, profile, supplied_pool_ids
        )
        inspection = inspect_external_fidelity_profile(
            prepared,
            profile_id=profile_id,
            supplied_pool_ids=supplied_pool_ids,
            expected_pool_count=expected_pool_count,
        )
    else:
        if supplied_pool_ids is not None:
            raise ValueError(
                "judgment-union Profiles do not accept caller-supplied pool membership"
            )
        selected_trials, pool_ids_hash = _judgment_union(prepared, profile)
        inspection = inspect_judgment_union_profile(prepared, profile_id=profile_id)
    selected_ids = tuple(trial.trial_id for trial in selected_trials)
    if pool_ids_hash != _sha256_lines(selected_ids):
        raise ValueError("judgment-union trial ordering changed during resolution")
    if (
        len(selected_trials) != expected_pool_count
        or pool_ids_hash != expected_pool_ids_sha256
        or inspection["pool_receipt_id"] != expected_pool_receipt_id
    ):
        raise ValueError("compute-bounded pool does not match the frozen count and IDs SHA-256")
    return ResolvedBenchmark(
        prepared=prepared,
        profile=profile,
        trials=selected_trials,
        corpus_hash=_sha256_lines(trial.to_json() for trial in selected_trials),
        pool_ids_hash=pool_ids_hash,
        pool_source_hash=(
            _project_evaluation_package(prepared, profile, selected_trials).evaluation_package_id
            if profile.corpus_policy == "external_fidelity_pool"
            else prepared.evaluation_package.evaluation_package_id
        ),
        pool_receipt_id=expected_pool_receipt_id,
        unverified_membership_acknowledged=False,
    )


__all__ = [
    "CONFIRMATION_FULL_PROFILE",
    "EXTERNAL_FIDELITY_TREC_2021_PROFILE",
    "OFFICIAL_FULL_PROFILE",
    "PROFILE_IDS",
    "SIGIR_DESCRIPTION_JUDGMENT_UNION_PROFILE",
    "SIGIR_DESCRIPTION_PROFILE",
    "SIGIR_SUMMARY_JUDGMENT_UNION_PROFILE",
    "SIGIR_SUMMARY_PROFILE",
    "SYNTHETIC_PATIENT_TO_TRIAL_PROFILE",
    "TREC_2021_JUDGMENT_UNION_PROFILE",
    "TREC_2022_JUDGMENT_UNION_PROFILE",
    "TREC_2023_JUDGMENT_UNION_PROFILE",
    "BenchmarkProfile",
    "ResolvedBenchmark",
    "inspect_external_fidelity_profile",
    "inspect_judgment_union_profile",
    "judgment_union_pool_receipt",
    "load_profile",
    "ordered_trial_ids_sha256",
    "pool_receipt_policy",
    "resolve_benchmark_profile",
]
