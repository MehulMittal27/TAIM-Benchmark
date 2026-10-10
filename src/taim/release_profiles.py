"""Closed Benchmark Profile registry for public Benchmark Release 0.1."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence, Set
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from taim.contracts import content_sha256, parameter_values, require_sha256
from taim.evaluation_package import EvaluationPackage
from taim.profile_values import (
    ProfileDefinition,
    ResolvedBenchmarkValue,
    profile_bytes_sha256,
    profile_json_bytes,
    profile_lines_sha256,
)
from taim.schemas import JsonValue
from taim.snapshot import Trial

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
class BenchmarkProfile(ProfileDefinition):
    """Profile admitted by this module's corpus and evaluation policy."""

    corpus_policy: CorpusPolicy
    unjudged_policy: UnjudgedPolicy
    precision_relevance_minimum: int
    precision_denominator: PrecisionDenominator


@dataclass(frozen=True, slots=True)
class ResolvedBenchmark(ResolvedBenchmarkValue[BenchmarkProfile]):
    """An immutable corpus admitted under this module's profile policy."""

    pool_receipt_id: str | None
    unverified_membership_acknowledged: Literal[False]
    #: The prepared corpus's trial IDs when resolution has already read them, so the corpus is
    #: parsed once; omitted, the external-fidelity check reads the corpus for them.
    prepared_trial_ids: InitVar[Set[str] | None] = None

    def __post_init__(self, prepared_trial_ids: Set[str] | None) -> None:
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
            prepared_ids = (
                prepared_trial_ids
                if prepared_trial_ids is not None
                else {trial.trial_id for trial in self.prepared.trials}
            )
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

    def manifest_configuration(self) -> dict[str, JsonValue]:
        return {
            **self.profile_manifest_configuration(
                unverified_membership_acknowledged=self.unverified_membership_acknowledged
            ),
            "pool_receipt_id": self.pool_receipt_id,
        }


def ordered_trial_ids_sha256(trial_ids: Iterable[str]) -> str:
    """Hash canonically ordered trial IDs for a judgment-union receipt."""

    return profile_lines_sha256(trial_ids)


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
    trials: Sequence[Trial],
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
) -> tuple[tuple[Trial, ...], str]:
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
    return selected_trials, profile_lines_sha256(trial.trial_id for trial in selected_trials)


def inspect_judgment_union_profile(
    prepared: PreparedBenchmark,
    *,
    profile_id: str,
) -> dict[str, JsonValue]:
    """Return the receipt that must be frozen before a judgment-union effectiveness run."""

    profile = load_profile(profile_id)
    selected_trials, pool_ids_hash = _judgment_union(prepared, profile)
    return _judgment_union_receipt(prepared, profile, selected_trials, pool_ids_hash)


def _judgment_union_receipt(
    prepared: PreparedBenchmark,
    profile: BenchmarkProfile,
    selected_trials: Sequence[Trial],
    pool_ids_hash: str,
) -> dict[str, JsonValue]:
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
) -> tuple[tuple[Trial, ...], str, frozenset[str]]:
    """The pool's trials in prepared order, their IDs' hash, and every prepared trial ID."""

    if profile.corpus_policy != "external_fidelity_pool":
        raise ValueError(f"Profile {profile.profile_id!r} is not an external-fidelity Profile")
    requested = tuple(supplied_pool_ids)
    if not requested or any(
        not isinstance(trial_id, str) or not trial_id for trial_id in requested
    ):
        raise ValueError("external-fidelity pool IDs must be non-empty strings")
    if len(requested) != len(set(requested)):
        raise ValueError("external-fidelity pool contains a duplicate trial ID")
    requested_set = set(requested)
    # One pass over the prepared corpus, which a real package parses on every pass.
    prepared_ids: set[str] = set()
    pool_trials: list[Trial] = []
    for trial in prepared.trials:
        prepared_ids.add(trial.trial_id)
        if trial.trial_id in requested_set:
            pool_trials.append(trial)
    unknown = sorted(requested_set - prepared_ids)
    if unknown:
        raise ValueError(f"external-fidelity pool contains unknown trial ID {unknown[0]!r}")
    judged_ids = {judgment.trial_id for judgment in prepared.evaluation_package.judgments}
    unjudged = sorted(requested_set - judged_ids)
    if unjudged:
        raise ValueError(
            f"external-fidelity pool contains trial ID {unjudged[0]!r} with no supplied judgment"
        )
    selected = tuple(pool_trials)
    if profile.expected_pool_count is not None and len(selected) != profile.expected_pool_count:
        raise ValueError(
            f"{profile.profile_id} requires exactly {profile.expected_pool_count} trials"
        )
    return (
        selected,
        profile_lines_sha256(trial.trial_id for trial in selected),
        frozenset(prepared_ids),
    )


def inspect_external_fidelity_profile(
    prepared: PreparedBenchmark,
    *,
    profile_id: str,
    supplied_pool_ids: Iterable[str],
    expected_pool_count: int,
) -> dict[str, JsonValue]:
    """Return a receipt for caller-supplied, externally authored pool membership."""

    profile = load_profile(profile_id)
    selected_trials, pool_ids_hash, _ = _external_fidelity_pool(
        prepared, profile, supplied_pool_ids
    )
    return _external_fidelity_receipt(
        prepared, profile, selected_trials, pool_ids_hash, expected_pool_count
    )


def _external_fidelity_receipt(
    prepared: PreparedBenchmark,
    profile: BenchmarkProfile,
    selected_trials: Sequence[Trial],
    pool_ids_hash: str,
    expected_pool_count: int,
) -> dict[str, JsonValue]:
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
        payload = parameter_values(
            json.loads(path.read_text(encoding="utf-8")), role="public Benchmark Profile"
        )
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
        definition_sha256=profile_bytes_sha256(profile_json_bytes(payload)),
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
    # Each branch reads the prepared corpus once: a real package parses it on every pass.
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
        trial_ids = tuple(trial.trial_id for trial in prepared.trials)
        return ResolvedBenchmark(
            prepared=prepared,
            profile=profile,
            trials=prepared.trials,
            corpus_hash=prepared.snapshot.logical_content_hashes()["trials"],
            pool_ids_hash=profile_lines_sha256(trial_ids),
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
    prepared_trial_ids: frozenset[str] | None = None
    if profile.corpus_policy == "external_fidelity_pool":
        if supplied_pool_ids is None:
            raise ValueError("external-fidelity Profile requires caller-supplied pool membership")
        selected_trials, pool_ids_hash, prepared_trial_ids = _external_fidelity_pool(
            prepared, profile, supplied_pool_ids
        )
        inspection = _external_fidelity_receipt(
            prepared, profile, selected_trials, pool_ids_hash, expected_pool_count
        )
    else:
        if supplied_pool_ids is not None:
            raise ValueError(
                "judgment-union Profiles do not accept caller-supplied pool membership"
            )
        selected_trials, pool_ids_hash = _judgment_union(prepared, profile)
        inspection = _judgment_union_receipt(prepared, profile, selected_trials, pool_ids_hash)
    selected_ids = tuple(trial.trial_id for trial in selected_trials)
    if pool_ids_hash != profile_lines_sha256(selected_ids):
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
        corpus_hash=profile_lines_sha256(trial.to_json() for trial in selected_trials),
        pool_ids_hash=pool_ids_hash,
        pool_source_hash=(
            _project_evaluation_package(prepared, profile, selected_trials).evaluation_package_id
            if profile.corpus_policy == "external_fidelity_pool"
            else prepared.evaluation_package.evaluation_package_id
        ),
        pool_receipt_id=expected_pool_receipt_id,
        unverified_membership_acknowledged=False,
        prepared_trial_ids=prepared_trial_ids,
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
