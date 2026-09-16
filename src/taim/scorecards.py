"""Versioned, stage-aware evaluation scorecards.

The shared scorecard is evaluator-owned.  It consumes closed ranking artifacts
and separately supplied Judgments; none of its inputs are exposed to Systems.
Method-specific Benchmark Profiles remain authoritative for their own corpus,
unjudged, denominator, and ranking-source policies.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from taim.contracts import (
    content_sha256,
    require_exact_keys,
    require_non_empty,
    require_sha256,
)
from taim.evaluation import MetricValue, evaluate_run
from taim.profiles import resolve_benchmark_profile_evaluation
from taim.schemas import (
    Candidate,
    RelevanceJudgment,
    SchemaValidationError,
    validate_pipeline_depth,
)

SCORECARD_SCHEMA_VERSION = "1.0"
SCORECARD_PROFILE_VERSION = "2.0"
SCORECARD_EVALUATOR_VERSION = "taim-scorecard-evaluator-v1"
FINAL_RANKING_CUTOFFS = (5, 10, 20)

PositiveLabelPolicy: TypeAlias = Literal["label_2", "labels_1_or_2"]
ScoreDirection: TypeAlias = Literal["higher_is_positive", "lower_is_positive"]
ScoreIntent: TypeAlias = Literal["ranking", "probability"]
LabelCompleteness: TypeAlias = Literal["complete", "judged_only"]
FilteringPresentation: TypeAlias = Literal["compacted", "no_refill"]


@dataclass(frozen=True, slots=True)
class ScoreSemantics:
    """Evaluator-side declaration for one comparable score identity."""

    score_id: str
    direction: ScoreDirection
    intent: ScoreIntent
    comparable_across_pairs: bool

    def __post_init__(self) -> None:
        if not isinstance(self.score_id, str) or not self.score_id:
            raise ValueError("score_id must be a non-empty string")
        if self.direction not in ("higher_is_positive", "lower_is_positive"):
            raise ValueError("score direction is unsupported")
        if self.intent not in ("ranking", "probability"):
            raise ValueError("score intent is unsupported")
        if not isinstance(self.comparable_across_pairs, bool):
            raise ValueError("comparable_across_pairs must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "score_id": self.score_id,
            "direction": self.direction,
            "intent": self.intent,
            "comparable_across_pairs": self.comparable_across_pairs,
        }


@dataclass(frozen=True, slots=True)
class ScoredPair:
    """One evaluator-side population row after scores and labels are joined."""

    pair_id: str
    label: int | None
    score: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("pair_id must be a non-empty string")
        if self.label is not None and (
            isinstance(self.label, bool)
            or not isinstance(self.label, int)
            or self.label not in (0, 1, 2)
        ):
            raise ValueError("label must be null or an integer in {0, 1, 2}")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, int | float)
            or not math.isfinite(self.score)
        ):
            raise ValueError("score must be null or a finite number")
        if self.score is not None:
            object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class ScorecardRunIdentity:
    """Exact benchmark and System identities bound into a persisted scorecard."""

    run_id: str
    system_id: str
    benchmark_lineage: str
    prepared_snapshot_id: str
    task_input_id: str
    system_input_id: str
    evaluation_package_id: str

    def __post_init__(self) -> None:
        for field_name in ("run_id", "system_id", "benchmark_lineage"):
            require_non_empty(getattr(self, field_name), field_name)
        for field_name in (
            "prepared_snapshot_id",
            "task_input_id",
            "system_input_id",
            "evaluation_package_id",
        ):
            require_sha256(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "system_id": self.system_id,
            "benchmark_lineage": self.benchmark_lineage,
            "prepared_snapshot_id": self.prepared_snapshot_id,
            "task_input_id": self.task_input_id,
            "system_input_id": self.system_input_id,
            "evaluation_package_id": self.evaluation_package_id,
        }


def _bound_candidate_rows(
    candidates: Iterable[Candidate],
    *,
    identity: ScorecardRunIdentity,
    collection_name: str,
    binding_name: str,
) -> tuple[Candidate, ...]:
    rows = tuple(candidates)
    if any(not isinstance(candidate, Candidate) for candidate in rows):
        raise TypeError(f"{collection_name} must contain Candidate instances")
    if any(candidate.run_id != identity.run_id for candidate in rows):
        raise ValueError(f"candidate run_id does not match {binding_name}")
    if any(candidate.system_id != identity.system_id for candidate in rows):
        raise ValueError(f"candidate system_id does not match {binding_name}")
    return rows


@dataclass(frozen=True, slots=True)
class ScorecardEvaluationBinding:
    """Exact evaluator-owned provenance for a deployment diagnostic."""

    identity: ScorecardRunIdentity
    benchmark_profile: Mapping[str, object]
    source_artifacts: Mapping[str, str]
    score_semantics: ScoreSemantics | None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ScorecardRunIdentity):
            raise TypeError("identity must be a ScorecardRunIdentity")
        if not isinstance(self.benchmark_profile, Mapping):
            raise TypeError("benchmark_profile must be a mapping")
        resolve_benchmark_profile_evaluation(self.benchmark_profile)
        if not self.source_artifacts:
            raise ValueError("source_artifacts must contain at least one artifact identity")
        for role, digest in self.source_artifacts.items():
            require_non_empty(role, "source_artifacts role")
            require_sha256(digest, f"source_artifacts.{role}")
        if self.score_semantics is not None and not isinstance(
            self.score_semantics, ScoreSemantics
        ):
            raise TypeError("score_semantics must be null or a ScoreSemantics instance")
        object.__setattr__(
            self,
            "benchmark_profile",
            json.loads(json.dumps(self.benchmark_profile, allow_nan=False, ensure_ascii=False)),
        )
        object.__setattr__(self, "source_artifacts", dict(self.source_artifacts))

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity.to_dict(),
            "benchmark_profile": self.benchmark_profile,
            "source_artifacts": dict(self.source_artifacts),
            "score_semantics": (
                self.score_semantics.to_dict()
                if self.score_semantics is not None
                else {"status": "not_applicable"}
            ),
            "evaluator_version": SCORECARD_EVALUATOR_VERSION,
        }


def _metric_contract(
    metric_id: str,
    *,
    positive_label_policy: str,
    cutoffs: Sequence[int] = (),
    denominator_policy: str,
    unjudged_policy: str,
    ranking_source: str,
    gain_policy: str = "binary",
) -> dict[str, object]:
    return {
        "metric_id": metric_id,
        "positive_label_policy": positive_label_policy,
        "cutoffs": list(cutoffs),
        "denominator_policy": denominator_policy,
        "unjudged_policy": unjudged_policy,
        "aggregation": [
            "per_topic",
            "macro_mean",
            "median",
            "confidence_interval_when_supported",
        ],
        "ranking_source": ranking_source,
        "gain_policy": gain_policy,
    }


def shared_scorecard_profile() -> dict[str, object]:
    """Return the immutable shared scorecard profile with distinct stage contracts."""

    final_metrics = [
        _metric_contract(
            "graded_ndcg",
            positive_label_policy="graded_labels_0_1_2",
            cutoffs=FINAL_RANKING_CUTOFFS,
            denominator_policy="ideal_discounted_gain_from_all_topic_judgments",
            unjudged_policy="bound_benchmark_profile",
            ranking_source="primary_ranking",
            gain_policy="linear_labels_0_1_2",
        ),
        _metric_contract(
            "eligible_precision",
            positive_label_policy="label_2",
            cutoffs=FINAL_RANKING_CUTOFFS,
            denominator_policy="bound_benchmark_profile",
            unjudged_policy="bound_benchmark_profile",
            ranking_source="primary_ranking",
        ),
        _metric_contract(
            "eligible_recall",
            positive_label_policy="label_2",
            cutoffs=FINAL_RANKING_CUTOFFS,
            denominator_policy="all_topic_label_2_judgments",
            unjudged_policy="bound_benchmark_profile",
            ranking_source="primary_ranking",
        ),
        _metric_contract(
            "relevant_or_eligible_precision",
            positive_label_policy="labels_1_or_2",
            cutoffs=FINAL_RANKING_CUTOFFS,
            denominator_policy="bound_benchmark_profile",
            unjudged_policy="bound_benchmark_profile",
            ranking_source="primary_ranking",
        ),
        _metric_contract(
            "relevant_or_eligible_recall",
            positive_label_policy="labels_1_or_2",
            cutoffs=FINAL_RANKING_CUTOFFS,
            denominator_policy="all_topic_label_1_or_2_judgments",
            unjudged_policy="bound_benchmark_profile",
            ranking_source="primary_ranking",
        ),
        _metric_contract(
            "eligible_mrr",
            positive_label_policy="label_2",
            denominator_policy="first_label_2_rank",
            unjudged_policy="bound_benchmark_profile",
            ranking_source="primary_ranking",
        ),
        _metric_contract(
            "first_eligible_rank",
            positive_label_policy="label_2",
            denominator_policy="declared_candidate_positions",
            unjudged_policy="bound_benchmark_profile",
            ranking_source="primary_ranking",
        ),
        _metric_contract(
            "eligible_recall_at_declared_max_candidate_depth",
            positive_label_policy="label_2",
            denominator_policy="all_topic_label_2_judgments",
            unjudged_policy="bound_benchmark_profile",
            ranking_source="primary_ranking",
        ),
        _metric_contract(
            "relevant_or_eligible_recall_at_declared_max_candidate_depth",
            positive_label_policy="labels_1_or_2",
            denominator_policy="all_topic_label_1_or_2_judgments",
            unjudged_policy="bound_benchmark_profile",
            ranking_source="primary_ranking",
        ),
    ]
    reranking_metrics = [
        {**metric, "ranking_source": "selected_stage_ranking"} for metric in final_metrics
    ]
    reranking_metrics.extend(
        [
            _metric_contract(
                "candidate_exposure",
                positive_label_policy="not_applicable",
                denominator_policy="candidates_from_immediate_input",
                unjudged_policy="include_all_candidate_ids",
                ranking_source="paired_input_and_selected_stage_rankings",
            ),
            _metric_contract(
                "unchanged_tail_identity",
                positive_label_policy="not_applicable",
                denominator_policy="declared_unchanged_tail_positions",
                unjudged_policy="include_all_candidate_ids",
                ranking_source="paired_input_and_selected_stage_rankings",
            ),
        ]
    )
    payload: dict[str, object] = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "profile_id": "taim-shared-scorecard",
        "profile_version": SCORECARD_PROFILE_VERSION,
        "stage_contracts": {
            "retrieval": {
                "ranking_source": "retrieval_stage_ranking",
                "metrics": [
                    _metric_contract(
                        "eligible_recall",
                        positive_label_policy="label_2",
                        cutoffs=(100, 500, 1_000, 2_000),
                        denominator_policy="all_topic_label_2_judgments",
                        unjudged_policy="bound_benchmark_profile",
                        ranking_source="retrieval_stage_ranking",
                    ),
                    _metric_contract(
                        "relevant_or_eligible_recall",
                        positive_label_policy="labels_1_or_2",
                        cutoffs=(100, 500, 1_000, 2_000),
                        denominator_policy="all_topic_label_1_or_2_judgments",
                        unjudged_policy="bound_benchmark_profile",
                        ranking_source="retrieval_stage_ranking",
                    ),
                    _metric_contract(
                        "judged_candidate_and_score_coverage",
                        positive_label_policy="not_applicable",
                        denominator_policy="emitted_stage_candidates",
                        unjudged_policy="report_separately",
                        ranking_source="retrieval_stage_ranking",
                    ),
                    _metric_contract(
                        "judged_eligible_retrieval_misses",
                        positive_label_policy="label_2",
                        denominator_policy="all_topic_label_2_judgments",
                        unjudged_policy="exclude_from_miss_count",
                        ranking_source="retrieval_stage_ranking",
                    ),
                    _metric_contract(
                        "candidate_overlap",
                        positive_label_policy="not_applicable",
                        denominator_policy="union_of_compared_candidate_sets",
                        unjudged_policy="include_all_candidate_ids",
                        ranking_source="paired_retrieval_stage_rankings",
                    ),
                    _metric_contract(
                        "indexing_query_latency_and_memory",
                        positive_label_policy="not_applicable",
                        denominator_policy="declared_operations",
                        unjudged_policy="not_applicable",
                        ranking_source="retrieval_stage_runtime",
                    ),
                ],
            },
            "filtering": {
                "ranking_source": "paired_before_and_after_stage_rankings",
                "metrics": [
                    _metric_contract(
                        "eligible_retention_rate",
                        positive_label_policy="label_2",
                        denominator_policy="label_2_candidates_before_filtering",
                        unjudged_policy="report_separately",
                        ranking_source="paired_before_and_after_stage_rankings",
                    ),
                    _metric_contract(
                        "candidate_reduction_ratio",
                        positive_label_policy="not_applicable",
                        denominator_policy="candidates_before_filtering",
                        unjudged_policy="include_all_candidates",
                        ranking_source="paired_before_and_after_stage_rankings",
                    ),
                    _metric_contract(
                        "label_2_false_negative_count_and_rate",
                        positive_label_policy="label_2",
                        denominator_policy="label_2_candidates_before_filtering",
                        unjudged_policy="exclude_from_false_negative_count",
                        ranking_source="paired_before_and_after_stage_rankings",
                    ),
                    _metric_contract(
                        "removals_by_label_and_unjudged",
                        positive_label_policy="labels_0_1_2_and_unjudged_separate",
                        denominator_policy="all_removed_candidates",
                        unjudged_policy="separate_class",
                        ranking_source="paired_before_and_after_stage_rankings",
                    ),
                    _metric_contract(
                        "paired_per_topic_metric_delta",
                        positive_label_policy="declared_by_nested_metric",
                        denominator_policy="paired_declared_topics",
                        unjudged_policy="bound_benchmark_profile",
                        ranking_source="paired_before_and_after_stage_rankings",
                    ),
                ],
                "presentation_identity_policy": "compacted_and_no_refill_are_distinct",
            },
            "pair_classification": {
                "ranking_source": "pair_classification_artifact",
                "top_k_allowed": False,
                "metrics": [
                    _metric_contract(
                        "three_class_confusion_and_per_class_metrics",
                        positive_label_policy="labels_0_1_2_separate",
                        denominator_policy="all_expected_judged_pairs",
                        unjudged_policy="report_separately",
                        ranking_source="pair_classification_artifact",
                    ),
                    _metric_contract(
                        "macro_precision_recall_f1_balanced_accuracy_and_exact_accuracy",
                        positive_label_policy="labels_0_1_2_separate",
                        denominator_policy="all_valid_judged_pair_predictions",
                        unjudged_policy="report_separately",
                        ranking_source="pair_classification_artifact",
                    ),
                    _metric_contract(
                        "eligible_and_relevant_or_eligible_binary_projections",
                        positive_label_policy="label_2_and_labels_1_or_2_reported_separately",
                        denominator_policy="all_valid_judged_pair_predictions",
                        unjudged_policy="report_separately",
                        ranking_source="pair_classification_artifact",
                    ),
                    _metric_contract(
                        "multiclass_brier_and_log_loss",
                        positive_label_policy="labels_0_1_2_separate",
                        denominator_policy="all_valid_judged_pair_probabilities",
                        unjudged_policy="report_separately",
                        ranking_source="complete_pair_probability_artifact",
                    ),
                    _metric_contract(
                        "classwise_and_top_label_calibration_status",
                        positive_label_policy="labels_0_1_2_separate",
                        denominator_policy="frozen_minimum_support_policy",
                        unjudged_policy="report_separately",
                        ranking_source="complete_pair_probability_artifact",
                    ),
                    _metric_contract(
                        "replicate_and_per_trial_raw_counts",
                        positive_label_policy="labels_0_1_2_separate",
                        denominator_policy="same_complete_expected_pair_population",
                        unjudged_policy="report_separately",
                        ranking_source="pair_classification_artifact",
                    ),
                ],
                "missing_prediction_policy": "operational_failure_and_coverage_loss_not_label_0",
            },
            "pair_assessment": {
                "ranking_source": "pair_assessment_artifact",
                "metrics": [
                    _metric_contract(
                        "coverage_abstention_selective_alignment",
                        positive_label_policy="declared_target",
                        denominator_policy="all_requested_pair_assessments",
                        unjudged_policy="report_separately",
                        ranking_source="pair_assessment_artifact",
                    ),
                    _metric_contract(
                        "eligible_retention_and_false_exclusion",
                        positive_label_policy="label_2",
                        denominator_policy="all_supplied_label_2_pairs",
                        unjudged_policy="exclude_and_report_coverage",
                        ranking_source="pair_assessment_artifact",
                    ),
                    _metric_contract(
                        "review_workload",
                        positive_label_policy="needs_information_abstention",
                        denominator_policy="all_valid_pair_assessments",
                        unjudged_policy="include_and_report_separately",
                        ranking_source="pair_assessment_artifact",
                    ),
                    _metric_contract(
                        "abstention_by_original_label",
                        positive_label_policy="labels_0_1_2_separate",
                        denominator_policy="all_supplied_judged_pair_assessments",
                        unjudged_policy="report_separately",
                        ranking_source="pair_assessment_artifact",
                    ),
                    _metric_contract(
                        "separate_label_0_and_label_1_diagnostics",
                        positive_label_policy="labels_0_and_1_never_merged",
                        denominator_policy="per_original_label_support",
                        unjudged_policy="report_separately",
                        ranking_source="pair_assessment_artifact",
                    ),
                ],
                "trec_probability_semantics_allowed": False,
            },
            "derived_pair_ranking": {
                "ranking_source": "separately_identified_derived_pair_ranking",
                "metrics": reranking_metrics,
                "direction_required": True,
                "source_prediction_lineage_required": True,
                "source_classification_mutation_allowed": False,
                "cutoff_policy": "ranking_artifact_and_evaluator_only",
            },
            "reranking": {
                "ranking_source": "selected_stage_ranking",
                "metrics": reranking_metrics,
                "comparison": "paired_per_topic_change_from_immediate_input",
            },
            "final_selection": {
                "ranking_source": "primary_ranking",
                "metrics": final_metrics,
                "primary_endpoint": "graded_ndcg_at_10",
            },
        },
        "report_sections": [
            "effectiveness",
            "deployment_diagnostics",
            "operational_reliability",
        ],
    }
    return {**payload, "definition_sha256": content_sha256(payload)}


def _numeric_topic_metrics(result: Mapping[str, object], topic_id: str) -> dict[str, object]:
    per_topic = result.get("per_topic")
    if not isinstance(per_topic, Mapping):
        raise TypeError("internal per-topic metrics must be a mapping")
    topic = per_topic.get(topic_id)
    if not isinstance(topic, Mapping):
        raise TypeError("internal topic metrics must be a mapping")
    return dict(topic)


def _bootstrap_intervals(
    per_topic: Mapping[str, Mapping[str, MetricValue]],
    metric_names: Sequence[str],
    *,
    samples: int,
) -> dict[str, object]:
    if samples == 0:
        return {
            "status": "not_computed",
            "reason": "bootstrap_samples is zero",
            "samples": 0,
            "seed": 0,
        }
    if isinstance(samples, bool) or samples < 100:
        raise ValueError("bootstrap_samples must be zero or at least 100")
    topic_ids = tuple(per_topic)
    if len(topic_ids) < 2:
        return {
            "status": "not_supported",
            "reason": "at least two topics are required for uncertainty",
            "samples": samples,
            "seed": 0,
        }
    intervals: dict[str, object] = {}
    for metric_name in metric_names:
        values = [per_topic[topic_id][metric_name] for topic_id in topic_ids]
        numeric = [float(value) for value in values if isinstance(value, int | float)]
        if len(numeric) != len(topic_ids):
            continue
        # Per-metric seeds keep validation independent of mapping iteration order.
        generator = random.Random(f"0:{metric_name}")  # noqa: S311
        means = sorted(
            statistics.fmean(generator.choice(numeric) for _ in numeric) for _ in range(samples)
        )
        lower_index = max(0, int(0.025 * samples) - 1)
        upper_index = min(samples - 1, int(0.975 * samples))
        intervals[metric_name] = {
            "lower": means[lower_index],
            "upper": means[upper_index],
        }
    return {
        "status": "computed",
        "method": "deterministic_topic_bootstrap_percentile",
        "confidence_level": 0.95,
        "samples": samples,
        "seed": 0,
        "mean_intervals": intervals,
    }


def evaluate_final_ranking_scorecard(
    candidates: Iterable[Candidate],
    judgments: Iterable[RelevanceJudgment],
    *,
    topic_ids: Iterable[str],
    benchmark_profile: Mapping[str, object],
    declared_max_candidate_depth: int,
    bootstrap_samples: int = 2_000,
) -> dict[str, object]:
    """Evaluate the required final Primary Ranking metrics under a bound profile."""

    profile_policy = resolve_benchmark_profile_evaluation(benchmark_profile)
    unjudged_policy = profile_policy.unjudged_policy
    precision_denominator = profile_policy.precision_denominator
    if precision_denominator == "maximum_graded_gain":
        # The paper profile uses maximum graded gain only for its graded P@10.
        # Shared scorecard precision remains binary and therefore uses the
        # profile's condensed result set as its denominator.
        precision_denominator = "available_condensed_results"
    if (
        isinstance(declared_max_candidate_depth, bool)
        or not isinstance(declared_max_candidate_depth, int)
        or declared_max_candidate_depth < 1
    ):
        raise ValueError("declared_max_candidate_depth must be a positive integer")
    candidate_rows = tuple(candidates)
    judgment_rows = tuple(judgments)
    selected_topics = tuple(topic_ids)
    if not selected_topics or len(selected_topics) != len(set(selected_topics)):
        raise ValueError("topic_ids must contain unique topic identifiers")
    if any(row.rank > declared_max_candidate_depth for row in candidate_rows):
        raise ValueError("candidate rank exceeds declared_max_candidate_depth")

    per_topic: dict[str, dict[str, MetricValue]] = {topic_id: {} for topic_id in selected_topics}
    first_eligible: dict[str, int | None] = {}
    metric_names: list[str] = []
    for cutoff in FINAL_RANKING_CUTOFFS:
        eligible = evaluate_run(
            candidate_rows,
            judgment_rows,
            topic_ids=selected_topics,
            k=cutoff,
            unjudged_policy=unjudged_policy,
            precision_relevance_minimum=2,
            precision_denominator=precision_denominator,
        )
        relevant = evaluate_run(
            candidate_rows,
            judgment_rows,
            topic_ids=selected_topics,
            k=cutoff,
            unjudged_policy=unjudged_policy,
            precision_relevance_minimum=1,
            precision_denominator=precision_denominator,
        )
        names = (
            f"graded_ndcg_at_{cutoff}",
            f"eligible_precision_at_{cutoff}",
            f"eligible_recall_at_{cutoff}",
            f"relevant_or_eligible_precision_at_{cutoff}",
            f"relevant_or_eligible_recall_at_{cutoff}",
        )
        metric_names.extend(names)
        for topic_id in selected_topics:
            eligible_topic = _numeric_topic_metrics(eligible, topic_id)
            relevant_topic = _numeric_topic_metrics(relevant, topic_id)
            per_topic[topic_id].update(
                {
                    names[0]: cast(float, eligible_topic[f"ndcg_at_{cutoff}"]),
                    names[1]: cast(float, eligible_topic[f"eligible_precision_at_{cutoff}"]),
                    names[2]: cast(float, eligible_topic[f"eligible_recall_at_{cutoff}"]),
                    names[3]: cast(
                        float,
                        relevant_topic[f"relevant_or_eligible_precision_at_{cutoff}"],
                    ),
                    names[4]: cast(
                        float,
                        relevant_topic[f"relevant_or_eligible_recall_at_{cutoff}"],
                    ),
                }
            )
            if cutoff == FINAL_RANKING_CUTOFFS[0]:
                rank = eligible_topic["first_eligible_rank"]
                first_eligible[topic_id] = cast(int | None, rank)
                per_topic[topic_id]["eligible_mrr"] = cast(float, eligible_topic["mrr"])
                per_topic[topic_id]["first_eligible_rank"] = cast(int | None, rank)

    maximum = evaluate_run(
        candidate_rows,
        judgment_rows,
        topic_ids=selected_topics,
        k=declared_max_candidate_depth,
        unjudged_policy=unjudged_policy,
    )
    max_names = (
        "eligible_recall_at_declared_max_candidate_depth",
        "relevant_or_eligible_recall_at_declared_max_candidate_depth",
    )
    metric_names.extend(("eligible_mrr", *max_names))
    for topic_id in selected_topics:
        maximum_topic = _numeric_topic_metrics(maximum, topic_id)
        per_topic[topic_id][max_names[0]] = cast(
            float,
            maximum_topic[f"eligible_recall_at_{declared_max_candidate_depth}"],
        )
        per_topic[topic_id][max_names[1]] = cast(
            float,
            maximum_topic[f"relevant_or_eligible_recall_at_{declared_max_candidate_depth}"],
        )

    mean = {
        name: statistics.fmean(
            cast(float, per_topic[topic_id][name]) for topic_id in selected_topics
        )
        for name in metric_names
    }
    median = {
        name: float(
            statistics.median(
                cast(float, per_topic[topic_id][name]) for topic_id in selected_topics
            )
        )
        for name in metric_names
    }
    observed_ranks = [rank for rank in first_eligible.values() if rank is not None]
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "evaluator_version": SCORECARD_EVALUATOR_VERSION,
        "ranking_source": "primary_ranking",
        "declared_max_candidate_depth": declared_max_candidate_depth,
        "cutoffs": list(FINAL_RANKING_CUTOFFS),
        "topic_count": len(selected_topics),
        "policies": {
            "ranking": {
                "unjudged_policy": unjudged_policy,
                "precision_denominator": precision_denominator,
                "gain_mapping": {"0": 0, "1": 1, "2": 2},
                "discount": "log2(rank + 1)",
                "aggregation": ["per_topic", "macro_mean", "median"],
                "shared_scorecard_cutoffs": list(FINAL_RANKING_CUTOFFS),
                "method_profile_cutoffs": list(
                    cast(
                        Sequence[int],
                        cast(Mapping[str, object], benchmark_profile["evaluation"])["cutoffs"],
                    )
                ),
                "benchmark_profile_id": benchmark_profile["profile_id"],
                "benchmark_profile_definition_sha256": benchmark_profile["definition_sha256"],
            },
            "eligible": {"positive_label_policy": "label_2"},
            "relevant_or_eligible": {"positive_label_policy": "labels_1_or_2"},
        },
        "per_topic": per_topic,
        "aggregate": {"mean": mean, "median": median},
        "first_eligible": {
            "per_topic_rank": first_eligible,
            "median_policy": "observed_ranks_only",
            "median_rank": (float(statistics.median(observed_ranks)) if observed_ranks else None),
            "topics_without_any_eligible_result_rate": 1.0
            - (len(observed_ranks) / len(selected_topics)),
            "no_eligible_in_top_k_rate": {
                str(cutoff): sum(rank is None or rank > cutoff for rank in first_eligible.values())
                / len(selected_topics)
                for cutoff in FINAL_RANKING_CUTOFFS
            },
        },
        "uncertainty": _bootstrap_intervals(
            per_topic,
            metric_names,
            samples=bootstrap_samples,
        ),
    }


def _positive(label: int, policy: PositiveLabelPolicy) -> bool:
    if policy == "label_2":
        return label == 2
    if policy == "labels_1_or_2":
        return label >= 1
    raise ValueError("positive_label_policy must be 'label_2' or 'labels_1_or_2'")


def _target_id(policy: PositiveLabelPolicy) -> str:
    return (
        "strict_eligibility_label_2_vs_0_or_1"
        if policy == "label_2"
        else "retrieval_relevance_labels_1_or_2_vs_0"
    )


def _validate_scored_population(rows: Iterable[ScoredPair]) -> tuple[ScoredPair, ...]:
    selected = tuple(rows)
    if not selected:
        raise ValueError("threshold population must contain at least one pair")
    if any(not isinstance(row, ScoredPair) for row in selected):
        raise TypeError("threshold population must contain ScoredPair instances")
    pair_ids = [row.pair_id for row in selected]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("threshold population pair_id values must be unique")
    return selected


def _auroc(
    positives: Sequence[ScoredPair],
    negatives: Sequence[ScoredPair],
    *,
    direction: ScoreDirection,
) -> float:
    concordance = 0.0
    for positive in positives:
        for negative in negatives:
            positive_score = cast(float, positive.score)
            negative_score = cast(float, negative.score)
            if positive_score == negative_score:
                concordance += 0.5
            elif (direction == "higher_is_positive" and positive_score > negative_score) or (
                direction == "lower_is_positive" and positive_score < negative_score
            ):
                concordance += 1.0
    return concordance / (len(positives) * len(negatives))


def evaluate_threshold_scorecard(
    rows: Iterable[ScoredPair],
    *,
    score_semantics: ScoreSemantics,
    binding: ScorecardEvaluationBinding,
    positive_label_policy: PositiveLabelPolicy,
    label_completeness: LabelCompleteness,
    evaluated_population_id: str,
    recall_targets: Sequence[float] = (0.95, 0.98, 0.99),
    high_recall_minimum: float = 0.95,
) -> dict[str, object]:
    """Evaluate a complete cohort or an explicitly judged-only score diagnostic.

    Every distinct threshold consumes the complete score tie group.  This
    prevents row ordering within ties from manufacturing unsupported operating
    points.
    """

    population = _validate_scored_population(rows)
    if not isinstance(score_semantics, ScoreSemantics):
        raise TypeError("score_semantics must be a ScoreSemantics instance")
    if not isinstance(binding, ScorecardEvaluationBinding):
        raise TypeError("binding must be a ScorecardEvaluationBinding")
    if binding.score_semantics != score_semantics:
        raise ValueError("binding score_semantics do not match threshold score semantics")
    if set(binding.source_artifacts) != {"score_artifact"}:
        raise ValueError("threshold binding requires exactly one score_artifact identity")
    if not score_semantics.comparable_across_pairs:
        raise ValueError("score_semantics.comparable_across_pairs must be true")
    if positive_label_policy not in ("label_2", "labels_1_or_2"):
        raise ValueError("positive_label_policy is unsupported")
    if label_completeness not in ("complete", "judged_only"):
        raise ValueError("label_completeness is unsupported")
    if not isinstance(evaluated_population_id, str) or not evaluated_population_id:
        raise ValueError("evaluated_population_id must be a non-empty string")
    if label_completeness == "complete" and any(row.label is None for row in population):
        raise ValueError("deployment threshold evaluation requires complete labels")
    labeled = tuple(row for row in population if row.label is not None)
    if any(row.score is None for row in population):
        raise ValueError("threshold evaluation requires complete score coverage")
    if any(
        isinstance(target, bool) or not isinstance(target, int | float) or not 0 < target <= 1
        for target in recall_targets
    ):
        raise ValueError("recall_targets must be numbers in (0, 1]")
    normalized_targets = tuple(float(target) for target in recall_targets)
    if tuple(sorted(set(normalized_targets))) != normalized_targets:
        raise ValueError("recall_targets must be sorted and unique")
    if (
        isinstance(high_recall_minimum, bool)
        or not isinstance(high_recall_minimum, int | float)
        or not 0 <= high_recall_minimum < 1
    ):
        raise ValueError("high_recall_minimum must be in [0, 1)")

    positives = tuple(
        row for row in labeled if _positive(cast(int, row.label), positive_label_policy)
    )
    negatives = tuple(row for row in labeled if row not in positives)
    if not positives or not negatives:
        raise ValueError("threshold evaluation requires both positive and negative classes")
    reverse = score_semantics.direction == "higher_is_positive"
    ordered = sorted(labeled, key=lambda row: cast(float, row.score), reverse=reverse)
    groups: list[tuple[float, list[ScoredPair]]] = []
    for row in ordered:
        score = cast(float, row.score)
        if not groups or groups[-1][0] != score:
            groups.append((score, []))
        groups[-1][1].append(row)

    curve: list[dict[str, object]] = [
        {
            "threshold": None,
            "precision": 1.0,
            "recall": 0.0,
            "review_fraction": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(positives),
        }
    ]
    true_positives = 0
    false_positives = 0
    reviewed = 0
    previous_recall = 0.0
    auprc = 0.0
    partial_auprc = 0.0
    for threshold, group in groups:
        group_positives = sum(
            _positive(cast(int, row.label), positive_label_policy) for row in group
        )
        true_positives += group_positives
        false_positives += len(group) - group_positives
        reviewed += len(group)
        recall = true_positives / len(positives)
        precision = true_positives / reviewed
        recall_delta = recall - previous_recall
        auprc += precision * recall_delta
        partial_auprc += precision * max(
            0.0,
            recall - max(previous_recall, float(high_recall_minimum)),
        )
        curve.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "review_fraction": reviewed / len(labeled),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": len(positives) - true_positives,
            }
        )
        previous_recall = recall

    operating_points: dict[str, object] = {}
    for target in normalized_targets:
        point = next(point for point in curve[1:] if cast(float, point["recall"]) >= target)
        precision = cast(float, point["precision"])
        operating_points[str(target)] = {
            "minimum_recall_target": target,
            "threshold": point["threshold"],
            "precision": precision,
            "recall": point["recall"],
            "review_fraction": point["review_fraction"],
            "number_needed_to_review": 1.0 / precision if precision else None,
            "false_negatives_per_1000": cast(int, point["false_negatives"]) / len(labeled) * 1_000,
            "false_negative_denominator": "evaluated_pairs",
        }

    claim_scope = "deployment" if label_completeness == "complete" else "judged_only_diagnostic"
    missing_label_policy = (
        "reject_missing_labels"
        if label_completeness == "complete"
        else "exclude_unlabeled_pairs_and_report_label_coverage"
    )
    curve_context = {
        "evaluated_population_id": evaluated_population_id,
        "missing_label_policy": missing_label_policy,
        "positive_label_policy": positive_label_policy,
        "prevalence": len(positives) / len(labeled),
        "label_coverage": len(labeled) / len(population),
        "score_coverage": sum(row.score is not None for row in population) / len(population),
    }
    for point in curve:
        point["population_context"] = curve_context
    for operating_point in operating_points.values():
        cast(dict[str, object], operating_point)["population_context"] = curve_context
    result: dict[str, object] = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "artifact_type": (
            "taim-deployment-threshold-evaluation"
            if claim_scope == "deployment"
            else "taim-judged-only-threshold-diagnostic"
        ),
        "evaluator_version": SCORECARD_EVALUATOR_VERSION,
        "artifact_binding": binding.to_dict(),
        "claim_scope": claim_scope,
        "evaluated_population_id": evaluated_population_id,
        "label_completeness": label_completeness,
        "missing_label_policy": missing_label_policy,
        "target_id": _target_id(positive_label_policy),
        "positive_label_policy": positive_label_policy,
        "score_semantics": score_semantics.to_dict(),
        "coverage": {
            "population_pairs": len(population),
            "labeled_pairs": len(labeled),
            "label_coverage": len(labeled) / len(population),
            "scored_pairs": sum(row.score is not None for row in population),
            "scored_labeled_pairs": len(labeled),
            "score_coverage": sum(row.score is not None for row in population) / len(population),
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "prevalence": len(positives) / len(labeled),
        },
        "curve_integration": "stepwise_precision_over_recall_increments",
        "precision_recall_curve": curve,
        "auprc": auprc,
        "auroc": _auroc(
            positives,
            negatives,
            direction=score_semantics.direction,
        ),
        "partial_auprc": {
            "minimum_recall": float(high_recall_minimum),
            "raw_area": partial_auprc,
            "normalized_area": partial_auprc / (1.0 - float(high_recall_minimum)),
        },
        "operating_points_by_minimum_recall": operating_points,
    }
    if claim_scope == "judged_only_diagnostic":
        result["caveat"] = (
            "Pooled judged-only results are not deployment precision, recall, prevalence, "
            "calibration, or threshold performance."
        )
    return result


def _calibration_regression(
    probabilities: Sequence[float], labels: Sequence[int]
) -> dict[str, object]:
    if len(probabilities) < 4:
        return {
            "status": "not_supported",
            "reason": "at least four labeled pairs are required",
        }
    if any(probability in (0.0, 1.0) for probability in probabilities):
        return {
            "status": "not_supported",
            "reason": "calibration regression requires probabilities strictly between 0 and 1",
        }
    logits = [math.log(value / (1.0 - value)) for value in probabilities]
    intercept = 0.0
    slope = 1.0
    for _iteration in range(100):
        fitted = [
            1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, intercept + slope * value))))
            for value in logits
        ]
        weights = [value * (1.0 - value) for value in fitted]
        gradient_0 = sum(label - value for label, value in zip(labels, fitted, strict=True))
        gradient_1 = sum(
            (label - value) * logit
            for label, value, logit in zip(labels, fitted, logits, strict=True)
        )
        information_00 = sum(weights)
        information_01 = sum(weight * logit for weight, logit in zip(weights, logits, strict=True))
        information_11 = sum(
            weight * logit * logit for weight, logit in zip(weights, logits, strict=True)
        )
        determinant = information_00 * information_11 - information_01 * information_01
        if determinant < 1e-12:
            return {
                "status": "not_supported",
                "reason": "calibration regression is singular or completely separated",
            }
        delta_intercept = (information_11 * gradient_0 - information_01 * gradient_1) / determinant
        delta_slope = (-information_01 * gradient_0 + information_00 * gradient_1) / determinant
        intercept += delta_intercept
        slope += delta_slope
        if max(abs(intercept), abs(slope)) > 50:
            return {
                "status": "not_supported",
                "reason": "calibration regression is completely separated",
            }
        if max(abs(delta_intercept), abs(delta_slope)) < 1e-10:
            return {"status": "computed", "intercept": intercept, "slope": slope}
    return {"status": "not_supported", "reason": "calibration regression did not converge"}


def evaluate_calibration_scorecard(
    rows: Iterable[ScoredPair],
    *,
    score_semantics: ScoreSemantics,
    binding: ScorecardEvaluationBinding,
    positive_label_policy: PositiveLabelPolicy,
    label_completeness: LabelCompleteness,
    evaluated_population_id: str,
    bins: int = 10,
) -> dict[str, object]:
    """Evaluate probability calibration after enforcing the probability gate."""

    population = _validate_scored_population(rows)
    if not isinstance(score_semantics, ScoreSemantics):
        raise TypeError("score_semantics must be a ScoreSemantics instance")
    if not isinstance(binding, ScorecardEvaluationBinding):
        raise TypeError("binding must be a ScorecardEvaluationBinding")
    if binding.score_semantics != score_semantics:
        raise ValueError("binding score_semantics do not match calibration score semantics")
    if set(binding.source_artifacts) != {"score_artifact"}:
        raise ValueError("calibration binding requires exactly one score_artifact identity")
    if score_semantics.intent != "probability":
        raise ValueError("calibration metrics require probability score intent")
    if score_semantics.direction != "higher_is_positive":
        raise ValueError("probability scores must use higher_is_positive direction")
    if not score_semantics.comparable_across_pairs:
        raise ValueError("score_semantics.comparable_across_pairs must be true")
    if positive_label_policy not in ("label_2", "labels_1_or_2"):
        raise ValueError("positive_label_policy is unsupported")
    if label_completeness not in ("complete", "judged_only"):
        raise ValueError("label_completeness is unsupported")
    if not isinstance(evaluated_population_id, str) or not evaluated_population_id:
        raise ValueError("evaluated_population_id must be a non-empty string")
    if label_completeness == "complete" and any(row.label is None for row in population):
        raise ValueError("deployment calibration requires complete labels")
    if any(row.score is None for row in population):
        raise ValueError("calibration requires complete score coverage")
    if any(not 0.0 <= cast(float, row.score) <= 1.0 for row in population):
        raise ValueError("probability scores must be within [0, 1]")
    labeled = tuple(row for row in population if row.label is not None)
    if not labeled:
        raise ValueError("calibration requires labeled pairs")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least two")

    probabilities = [cast(float, row.score) for row in labeled]
    labels = [int(_positive(cast(int, row.label), positive_label_policy)) for row in labeled]
    if not any(labels) or all(labels):
        raise ValueError("calibration requires both positive and negative classes")
    brier = statistics.fmean(
        (probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)
    )
    by_bin: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for probability, label in zip(probabilities, labels, strict=True):
        index = min(int(probability * bins), bins - 1)
        by_bin[index].append((probability, label))
    curve: list[dict[str, object]] = []
    ece = 0.0
    for index in range(bins):
        values = by_bin.get(index, [])
        if not values:
            continue
        mean_score = statistics.fmean(value[0] for value in values)
        observed = statistics.fmean(value[1] for value in values)
        ece += (len(values) / len(labeled)) * abs(mean_score - observed)
        curve.append(
            {
                "bin_index": index,
                "lower_bound": index / bins,
                "lower_inclusive": True,
                "upper_bound": 1.0 if index == bins - 1 else (index + 1) / bins,
                "upper_inclusive": index == bins - 1,
                "count": len(values),
                "mean_score": mean_score,
                "observed_positive_fraction": observed,
            }
        )
    claim_scope = "deployment" if label_completeness == "complete" else "judged_only_diagnostic"
    missing_label_policy = (
        "reject_missing_labels"
        if label_completeness == "complete"
        else "exclude_unlabeled_pairs_and_report_label_coverage"
    )
    population_context = {
        "evaluated_population_id": evaluated_population_id,
        "missing_label_policy": missing_label_policy,
        "positive_label_policy": positive_label_policy,
        "prevalence": sum(labels) / len(labels),
        "label_coverage": len(labeled) / len(population),
        "score_coverage": sum(row.score is not None for row in population) / len(population),
    }
    for point in curve:
        point["population_context"] = population_context
    result: dict[str, object] = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "artifact_type": (
            "taim-deployment-calibration-evaluation"
            if claim_scope == "deployment"
            else "taim-judged-only-calibration-diagnostic"
        ),
        "evaluator_version": SCORECARD_EVALUATOR_VERSION,
        "artifact_binding": binding.to_dict(),
        "claim_scope": claim_scope,
        "evaluated_population_id": evaluated_population_id,
        "label_completeness": label_completeness,
        "missing_label_policy": missing_label_policy,
        "target_id": _target_id(positive_label_policy),
        "positive_label_policy": positive_label_policy,
        "score_semantics": score_semantics.to_dict(),
        "pair_count": len(labeled),
        "coverage": {
            "population_pairs": len(population),
            "labeled_pairs": len(labeled),
            "scored_pairs": sum(row.score is not None for row in population),
            "label_coverage": len(labeled) / len(population),
            "score_coverage": sum(row.score is not None for row in population) / len(population),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "prevalence": sum(labels) / len(labels),
        },
        "binning": {"strategy": "equal_width", "bin_count": bins},
        "reliability_curve": curve,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "calibration_slope_and_intercept": _calibration_regression(probabilities, labels),
    }
    if claim_scope == "judged_only_diagnostic":
        result["caveat"] = (
            "Pooled judged-only calibration is not deployment calibration or threshold evidence."
        )
    return result


def evaluate_filtering_scorecard(
    before: Iterable[Candidate],
    after: Iterable[Candidate],
    judgments: Iterable[RelevanceJudgment],
    *,
    binding: ScorecardEvaluationBinding,
    topic_ids: Iterable[str],
    presentation_identity: FilteringPresentation,
) -> dict[str, object]:
    """Evaluate paired candidate loss across one filtering stage."""

    if not isinstance(binding, ScorecardEvaluationBinding):
        raise TypeError("binding must be a ScorecardEvaluationBinding")
    if binding.score_semantics is not None:
        raise ValueError("filtering binding score_semantics must be null")
    if set(binding.source_artifacts) != {
        "before_ranking_artifact",
        "after_ranking_artifact",
    }:
        raise ValueError("filtering binding requires before and after ranking artifact identities")
    if presentation_identity not in ("compacted", "no_refill"):
        raise ValueError("presentation_identity must be 'compacted' or 'no_refill'")
    before_rows = _bound_candidate_rows(
        before,
        identity=binding.identity,
        collection_name="filtering rankings",
        binding_name="filtering binding",
    )
    after_rows = _bound_candidate_rows(
        after,
        identity=binding.identity,
        collection_name="filtering rankings",
        binding_name="filtering binding",
    )
    judgment_rows = tuple(judgments)
    if any(not isinstance(row, RelevanceJudgment) for row in judgment_rows):
        raise TypeError("judgments must contain RelevanceJudgment instances")
    selected_topics = tuple(topic_ids)
    if not selected_topics or len(selected_topics) != len(set(selected_topics)):
        raise ValueError("topic_ids must contain unique topic identifiers")
    unknown_topics = {row.topic_id for row in (*before_rows, *after_rows)} - set(selected_topics)
    if unknown_topics:
        raise ValueError(
            "filtering rankings contain candidates outside topic_ids: "
            + ", ".join(sorted(unknown_topics))
        )

    def keyed(rows: Sequence[Candidate], role: str) -> dict[tuple[str, str], Candidate]:
        result = {(row.topic_id, row.trial_id): row for row in rows}
        if len(result) != len(rows):
            raise ValueError(f"{role} filtering ranking contains duplicate pairs")
        if len({(row.topic_id, row.rank) for row in rows}) != len(rows):
            raise ValueError(f"{role} filtering ranking contains duplicate ranks")
        return result

    before_by_pair = keyed(before_rows, "before")
    after_by_pair = keyed(after_rows, "after")
    if not set(after_by_pair) <= set(before_by_pair):
        raise ValueError("after filtering ranking must be a subset of the before ranking")
    if presentation_identity == "compacted":
        for topic_id in selected_topics:
            topic_after = sorted(
                (row for row in after_rows if row.topic_id == topic_id),
                key=lambda row: row.rank,
            )
            ranks = [row.rank for row in topic_after]
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError("compacted filtering rankings must use contiguous ranks from one")
            before_ranks = [
                before_by_pair[(row.topic_id, row.trial_id)].rank for row in topic_after
            ]
            if before_ranks != sorted(before_ranks):
                raise ValueError("compacted filtering rankings must preserve before-stage order")
    elif any(row.rank != before_by_pair[(row.topic_id, row.trial_id)].rank for row in after_rows):
        raise ValueError("no_refill filtering rankings must preserve before-stage ranks")
    relevance: dict[tuple[str, str], int] = {}
    for judgment in judgment_rows:
        key = (judgment.topic_id, judgment.trial_id)
        if key in relevance:
            raise ValueError("filtering scorecard contains duplicate Judgments")
        relevance[key] = judgment.label
    removed = set(before_by_pair) - set(after_by_pair)
    removals = {"label_2": 0, "label_1": 0, "label_0": 0, "unjudged": 0}
    for key in removed:
        label = relevance.get(key)
        removals["unjudged" if label is None else f"label_{label}"] += 1
    before_eligible = sum(relevance.get(key) == 2 for key in before_by_pair)
    removed_eligible = removals["label_2"]
    per_topic: dict[str, dict[str, object]] = {}
    for topic_id in selected_topics:
        eligible_judgments = {
            trial_id
            for (judgment_topic, trial_id), label in relevance.items()
            if judgment_topic == topic_id and label == 2
        }
        before_trials = {
            trial_id for candidate_topic, trial_id in before_by_pair if candidate_topic == topic_id
        }
        after_trials = {
            trial_id for candidate_topic, trial_id in after_by_pair if candidate_topic == topic_id
        }
        denominator = len(eligible_judgments)
        recall_before = (
            len(before_trials & eligible_judgments) / denominator if denominator else 0.0
        )
        recall_after = len(after_trials & eligible_judgments) / denominator if denominator else 0.0
        per_topic[topic_id] = {
            "eligible_recall_before": recall_before,
            "eligible_recall_after": recall_after,
            "eligible_recall_delta": recall_after - recall_before,
            "candidate_count_before": len(before_trials),
            "candidate_count_after": len(after_trials),
        }
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "artifact_type": "taim-filtering-stage-scorecard",
        "evaluator_version": SCORECARD_EVALUATOR_VERSION,
        "artifact_binding": binding.to_dict(),
        "stage_type": "filtering",
        "presentation_identity": presentation_identity,
        "candidate_count_before": len(before_rows),
        "candidate_count_after": len(after_rows),
        "candidate_reduction_ratio": (
            (len(before_rows) - len(after_rows)) / len(before_rows) if before_rows else 0.0
        ),
        "eligible_retention_rate": (
            (before_eligible - removed_eligible) / before_eligible if before_eligible else None
        ),
        "label_2_false_negative_count": removed_eligible,
        "label_2_false_negative_rate": (
            removed_eligible / before_eligible if before_eligible else None
        ),
        "removals": removals,
        "per_topic": per_topic,
    }


_SCORECARD_FIELDS = {
    "schema_version",
    "artifact_type",
    "artifact_version",
    "scorecard_profile",
    "evaluator_version",
    "run_id",
    "system_id",
    "benchmark_lineage",
    "prepared_snapshot_id",
    "task_input_id",
    "system_input_id",
    "evaluation_package_id",
    "benchmark_profile",
    "ranking_artifact",
    "score_semantics",
    "effectiveness",
    "deployment_diagnostics",
    "operational_reliability",
    "trec_eval_parity",
}
_EVALUATOR_ARTIFACT_FIELDS = {
    "metrics_sha256",
    "profile_metrics_sha256",
}

_DEPLOYMENT_REASON_CODES = (
    "pooled_judgments_are_not_a_complete_deployment_population",
    "score_semantics_are_undeclared",
)
_DEPLOYMENT_INPUT_GATES = (
    "score_identity",
    "monotonic_direction",
    "cross_pair_comparability",
    "finite_complete_score_coverage",
    "both_class_coverage",
    "sufficiently_complete_labels_or_frozen_correction_protocol",
)
_UNAVAILABLE_RELIABILITY_FIELDS = (
    "generation_or_decision_coverage",
    "semantic_abstention_count_and_rate",
    "invalid_or_missing_output_rate",
    "transport_authentication_and_quota_failures",
    "logical_calls_provider_attempts_retries_and_cache_reuse",
    "latency_p50_p95_p99_max_throughput_and_peak_memory",
    "token_api_usage_and_estimated_cost",
)
_TREC_EVAL_OVERLAPPING_METRICS = (
    "graded_ndcg_at_5",
    "graded_ndcg_at_10",
    "graded_ndcg_at_20",
    "eligible_precision_at_5",
    "eligible_precision_at_10",
    "eligible_precision_at_20",
    "eligible_recall_at_5",
    "eligible_recall_at_10",
    "eligible_mrr",
    "relevant_or_eligible_precision_at_5",
    "relevant_or_eligible_precision_at_10",
    "relevant_or_eligible_precision_at_20",
    "relevant_or_eligible_recall_at_5",
    "relevant_or_eligible_recall_at_10",
)


def _undeclared_score_semantics() -> dict[str, object]:
    return {"status": "undeclared", "threshold_evaluation_eligible": False}


def _closed_deployment_diagnostics() -> dict[str, object]:
    return {
        "status": "not_emitted",
        "claim_scope": "none",
        "reason_codes": list(_DEPLOYMENT_REASON_CODES),
        "required_complete_input_gates": list(_DEPLOYMENT_INPUT_GATES),
    }


def _partial_operational_reliability(runtime_seconds: float) -> dict[str, object]:
    return {
        "status": "partial",
        "runtime_seconds": runtime_seconds,
        "available_fields": ["runtime_seconds"],
        "unavailable_fields": list(_UNAVAILABLE_RELIABILITY_FIELDS),
    }


def _closed_trec_eval_parity() -> dict[str, object]:
    return {
        "status": "not_attested",
        "publication_gate_passed": False,
        "overlapping_standard_metrics": list(_TREC_EVAL_OVERLAPPING_METRICS),
    }


def _reconcile_effectiveness(effectiveness: Mapping[str, object]) -> None:
    expected_fields = {
        "schema_version",
        "evaluator_version",
        "ranking_source",
        "declared_max_candidate_depth",
        "cutoffs",
        "topic_count",
        "policies",
        "per_topic",
        "aggregate",
        "first_eligible",
        "uncertainty",
    }
    require_exact_keys(effectiveness, expected_fields, role="scorecard effectiveness")
    if effectiveness.get("schema_version") != SCORECARD_SCHEMA_VERSION:
        raise SchemaValidationError("unsupported scorecard effectiveness schema_version")
    if effectiveness.get("evaluator_version") != SCORECARD_EVALUATOR_VERSION:
        raise SchemaValidationError("unsupported scorecard evaluator_version")
    if effectiveness.get("ranking_source") != "primary_ranking":
        raise SchemaValidationError("scorecard effectiveness must use the Primary Ranking")
    if effectiveness.get("cutoffs") != list(FINAL_RANKING_CUTOFFS):
        raise SchemaValidationError("scorecard effectiveness cutoffs are unsupported")
    depth = effectiveness.get("declared_max_candidate_depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise SchemaValidationError("scorecard declared maximum depth must be positive")
    policies = effectiveness.get("policies")
    if not isinstance(policies, Mapping) or set(policies) != {
        "ranking",
        "eligible",
        "relevant_or_eligible",
    }:
        raise SchemaValidationError("scorecard effectiveness policies are invalid")
    ranking_policy = policies["ranking"]
    if not isinstance(ranking_policy, Mapping) or set(ranking_policy) != {
        "unjudged_policy",
        "precision_denominator",
        "gain_mapping",
        "discount",
        "aggregation",
        "shared_scorecard_cutoffs",
        "method_profile_cutoffs",
        "benchmark_profile_id",
        "benchmark_profile_definition_sha256",
    }:
        raise SchemaValidationError("scorecard ranking policy is invalid")
    if ranking_policy.get("unjudged_policy") not in (
        "retain_as_zero_gain",
        "condense_before_cutoff",
    ) or ranking_policy.get("precision_denominator") not in (
        "fixed_cutoff",
        "available_condensed_results",
    ):
        raise SchemaValidationError("scorecard ranking policy is unsupported")
    if (
        ranking_policy.get("gain_mapping") != {"0": 0, "1": 1, "2": 2}
        or ranking_policy.get("discount") != "log2(rank + 1)"
        or ranking_policy.get("aggregation") != ["per_topic", "macro_mean", "median"]
        or ranking_policy.get("shared_scorecard_cutoffs") != list(FINAL_RANKING_CUTOFFS)
    ):
        raise SchemaValidationError("scorecard ranking metric semantics are unsupported")
    method_cutoffs = ranking_policy.get("method_profile_cutoffs")
    if (
        not isinstance(method_cutoffs, list)
        or not method_cutoffs
        or any(
            isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1
            for cutoff in method_cutoffs
        )
        or method_cutoffs != sorted(set(method_cutoffs))
    ):
        raise SchemaValidationError("scorecard method profile cutoffs are invalid")
    if policies["eligible"] != {"positive_label_policy": "label_2"} or policies[
        "relevant_or_eligible"
    ] != {"positive_label_policy": "labels_1_or_2"}:
        raise SchemaValidationError("scorecard positive-label policies are unsupported")
    per_topic = effectiveness.get("per_topic")
    aggregate = effectiveness.get("aggregate")
    topic_count = effectiveness.get("topic_count")
    if not isinstance(per_topic, Mapping) or not isinstance(aggregate, Mapping):
        raise SchemaValidationError("scorecard effectiveness metrics must be objects")
    if isinstance(topic_count, bool) or not isinstance(topic_count, int):
        raise SchemaValidationError("scorecard topic_count must be an integer")
    if len(per_topic) != topic_count or not per_topic:
        raise SchemaValidationError("scorecard topic_count does not match per_topic")
    if set(aggregate) != {"mean", "median"}:
        raise SchemaValidationError("scorecard aggregate fields are invalid")
    mean = aggregate["mean"]
    median = aggregate["median"]
    if not isinstance(mean, Mapping) or not isinstance(median, Mapping) or set(mean) != set(median):
        raise SchemaValidationError("scorecard aggregate metric sets do not match")
    expected_metric_names = {
        *(
            f"{prefix}_at_{cutoff}"
            for cutoff in FINAL_RANKING_CUTOFFS
            for prefix in (
                "graded_ndcg",
                "eligible_precision",
                "eligible_recall",
                "relevant_or_eligible_precision",
                "relevant_or_eligible_recall",
            )
        ),
        "eligible_mrr",
        "eligible_recall_at_declared_max_candidate_depth",
        "relevant_or_eligible_recall_at_declared_max_candidate_depth",
    }
    if set(mean) != expected_metric_names:
        raise SchemaValidationError("scorecard aggregate metric set is unsupported")
    topic_rows: list[Mapping[str, object]] = []
    for topic_id, topic_metrics in per_topic.items():
        if not isinstance(topic_id, str) or not isinstance(topic_metrics, Mapping):
            raise SchemaValidationError("scorecard per_topic rows are invalid")
        if set(topic_metrics) != expected_metric_names | {"first_eligible_rank"}:
            raise SchemaValidationError("scorecard per_topic metric fields are unsupported")
        topic_rows.append(topic_metrics)
    for metric_name in mean:
        values: list[float] = []
        for topic_metrics in topic_rows:
            value = topic_metrics.get(metric_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise SchemaValidationError(
                    f"scorecard per_topic metric {metric_name!r} must be numeric"
                )
            if not math.isfinite(value):
                raise SchemaValidationError("scorecard metrics must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise SchemaValidationError("scorecard metrics must be within [0, 1]")
            values.append(float(value))
        expected_mean = statistics.fmean(values)
        expected_median = float(statistics.median(values))
        if mean[metric_name] != expected_mean or median[metric_name] != expected_median:
            raise ValueError(f"scorecard aggregate {metric_name!r} does not reconcile")
    first_eligible = effectiveness.get("first_eligible")
    if not isinstance(first_eligible, Mapping) or set(first_eligible) != {
        "per_topic_rank",
        "median_policy",
        "median_rank",
        "topics_without_any_eligible_result_rate",
        "no_eligible_in_top_k_rate",
    }:
        raise SchemaValidationError("scorecard first-eligible summary is invalid")
    expected_ranks = {
        topic_id: row.get("first_eligible_rank")
        for topic_id, row in zip(
            per_topic,
            topic_rows,
            strict=True,
        )
    }
    if any(
        rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1)
        for rank in expected_ranks.values()
    ):
        raise SchemaValidationError("scorecard first-eligible ranks are invalid")
    observed_ranks = [cast(int, rank) for rank in expected_ranks.values() if rank is not None]
    expected_first = {
        "per_topic_rank": expected_ranks,
        "median_policy": "observed_ranks_only",
        "median_rank": float(statistics.median(observed_ranks)) if observed_ranks else None,
        "topics_without_any_eligible_result_rate": 1.0 - len(observed_ranks) / topic_count,
        "no_eligible_in_top_k_rate": {
            str(cutoff): sum(
                rank is None or cast(int, rank) > cutoff for rank in expected_ranks.values()
            )
            / topic_count
            for cutoff in FINAL_RANKING_CUTOFFS
        },
    }
    if first_eligible != expected_first:
        raise ValueError("scorecard first-eligible summary does not reconcile")
    uncertainty = effectiveness.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise SchemaValidationError("scorecard uncertainty must be an object")
    samples = uncertainty.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise SchemaValidationError("scorecard uncertainty samples must be an integer")
    expected_uncertainty = _bootstrap_intervals(
        cast(Mapping[str, Mapping[str, MetricValue]], per_topic),
        sorted(expected_metric_names),
        samples=samples,
    )
    if uncertainty != expected_uncertainty:
        raise ValueError("scorecard uncertainty does not reconcile")


def validate_scorecard_artifact(payload: Mapping[str, object]) -> None:
    """Fail closed on version, identity, policy, and aggregate drift."""

    fields = set(payload)
    if fields not in (_SCORECARD_FIELDS, _SCORECARD_FIELDS | {"evaluator_artifacts"}):
        require_exact_keys(payload, _SCORECARD_FIELDS, role="final scorecard")
    if payload.get("schema_version") != SCORECARD_SCHEMA_VERSION:
        raise SchemaValidationError("unsupported final scorecard schema_version")
    if payload.get("artifact_type") != "taim-final-scorecard":
        raise SchemaValidationError("unsupported final scorecard artifact_type")
    if payload.get("artifact_version") != SCORECARD_PROFILE_VERSION:
        raise SchemaValidationError("unsupported final scorecard artifact_version")
    if payload.get("evaluator_version") != SCORECARD_EVALUATOR_VERSION:
        raise SchemaValidationError("unsupported final scorecard evaluator_version")
    if payload.get("scorecard_profile") != shared_scorecard_profile():
        raise SchemaValidationError("final scorecard profile identity or definition is unsupported")
    if "evaluator_artifacts" in payload:
        evaluator_artifacts = payload["evaluator_artifacts"]
        if not isinstance(evaluator_artifacts, Mapping):
            raise SchemaValidationError("scorecard evaluator_artifacts must be an object")
        require_exact_keys(
            evaluator_artifacts,
            _EVALUATOR_ARTIFACT_FIELDS,
            role="scorecard evaluator artifacts",
        )
        for field_name in sorted(_EVALUATOR_ARTIFACT_FIELDS):
            require_sha256(
                evaluator_artifacts.get(field_name),
                f"evaluator_artifacts.{field_name}",
            )
    for field_name in ("run_id", "system_id", "benchmark_lineage"):
        require_non_empty(payload.get(field_name), field_name)
    for field_name in (
        "prepared_snapshot_id",
        "task_input_id",
        "system_input_id",
        "evaluation_package_id",
    ):
        require_sha256(payload.get(field_name), field_name)
    benchmark_profile = payload.get("benchmark_profile")
    if not isinstance(benchmark_profile, Mapping):
        raise SchemaValidationError("final scorecard benchmark_profile must be an object")
    profile_policy = resolve_benchmark_profile_evaluation(benchmark_profile)
    ranking_artifact = payload.get("ranking_artifact")
    if not isinstance(ranking_artifact, Mapping):
        raise SchemaValidationError("final scorecard ranking_artifact must be an object")
    require_exact_keys(
        ranking_artifact,
        {
            "artifact_role",
            "artifact_sha256",
            "pipeline_depth",
            "declared_max_candidate_depth",
        },
        role="scorecard ranking artifact",
    )
    if ranking_artifact.get("artifact_role") != "primary_ranking":
        raise SchemaValidationError("scorecard ranking artifact must be the Primary Ranking")
    require_sha256(ranking_artifact.get("artifact_sha256"), "ranking_artifact.artifact_sha256")
    validate_pipeline_depth(ranking_artifact.get("pipeline_depth"))
    depth = ranking_artifact.get("declared_max_candidate_depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise SchemaValidationError("declared_max_candidate_depth must be a positive integer")
    effectiveness = payload.get("effectiveness")
    if not isinstance(effectiveness, Mapping):
        raise SchemaValidationError("scorecard effectiveness must be an object")
    if effectiveness.get("declared_max_candidate_depth") != depth:
        raise SchemaValidationError("scorecard ranking depth does not match effectiveness")
    policies = effectiveness.get("policies")
    if not isinstance(policies, Mapping) or not isinstance(policies.get("ranking"), Mapping):
        raise SchemaValidationError("scorecard effectiveness policies are invalid")
    ranking_policy = cast(Mapping[str, object], policies["ranking"])
    if ranking_policy.get("benchmark_profile_id") != benchmark_profile.get(
        "profile_id"
    ) or ranking_policy.get("benchmark_profile_definition_sha256") != benchmark_profile.get(
        "definition_sha256"
    ):
        raise SchemaValidationError("scorecard effectiveness is not bound to Benchmark Profile")
    profile_evaluation = benchmark_profile.get("evaluation")
    if not isinstance(profile_evaluation, Mapping):
        raise SchemaValidationError("scorecard Benchmark Profile evaluation is invalid")
    scorecard_precision_denominator = profile_policy.precision_denominator
    if scorecard_precision_denominator == "maximum_graded_gain":
        scorecard_precision_denominator = "available_condensed_results"
    if (
        ranking_policy.get("unjudged_policy") != profile_policy.unjudged_policy
        or ranking_policy.get("precision_denominator") != scorecard_precision_denominator
        or ranking_policy.get("method_profile_cutoffs") != profile_evaluation.get("cutoffs")
    ):
        raise SchemaValidationError("scorecard ranking policies drift from Benchmark Profile")
    _reconcile_effectiveness(effectiveness)
    if payload.get("score_semantics") != _undeclared_score_semantics():
        raise SchemaValidationError("final scorecard score semantics are unsupported")
    if payload.get("deployment_diagnostics") != _closed_deployment_diagnostics():
        raise SchemaValidationError(
            "final scorecard deployment diagnostics must remain fail-closed"
        )
    reliability = payload.get("operational_reliability")
    if not isinstance(reliability, Mapping):
        raise SchemaValidationError("final scorecard operational_reliability must be an object")
    runtime_seconds = reliability.get("runtime_seconds")
    if (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, int | float)
        or not math.isfinite(runtime_seconds)
        or runtime_seconds < 0
        or reliability != _partial_operational_reliability(float(runtime_seconds))
    ):
        raise SchemaValidationError("final scorecard operational reliability is unsupported")
    if payload.get("trec_eval_parity") != _closed_trec_eval_parity():
        raise SchemaValidationError("final scorecard trec_eval parity must remain fail-closed")


def build_final_scorecard_artifact(
    candidates: Iterable[Candidate],
    judgments: Iterable[RelevanceJudgment],
    *,
    topic_ids: Iterable[str],
    benchmark_profile: Mapping[str, object],
    identity: ScorecardRunIdentity,
    ranking_artifact_sha256: str,
    pipeline_depth: str,
    budget_k: int,
    runtime_seconds: float,
    evaluator_artifacts: Mapping[str, str] | None = None,
    bootstrap_samples: int = 2_000,
) -> dict[str, object]:
    """Build the bound machine-readable final scorecard for one closed run."""

    if not isinstance(identity, ScorecardRunIdentity):
        raise TypeError("identity must be a ScorecardRunIdentity")
    require_sha256(ranking_artifact_sha256, "ranking_artifact_sha256")
    pipeline_depth = validate_pipeline_depth(pipeline_depth)
    if isinstance(budget_k, bool) or not isinstance(budget_k, int) or budget_k < 1:
        raise ValueError("budget_k must be a positive integer")
    if (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, int | float)
        or not math.isfinite(runtime_seconds)
        or runtime_seconds < 0
    ):
        raise ValueError("runtime_seconds must be a finite non-negative number")
    bound_evaluator_artifacts: dict[str, str] | None = None
    if evaluator_artifacts is not None:
        if set(evaluator_artifacts) != _EVALUATOR_ARTIFACT_FIELDS:
            raise ValueError("evaluator_artifacts must contain the exact metric hash fields")
        bound_evaluator_artifacts = dict(evaluator_artifacts)
        for field_name, digest in bound_evaluator_artifacts.items():
            require_sha256(digest, f"evaluator_artifacts.{field_name}")
    candidate_rows = _bound_candidate_rows(
        candidates,
        identity=identity,
        collection_name="scorecard candidates",
        binding_name="scorecard identity",
    )
    effectiveness = evaluate_final_ranking_scorecard(
        candidate_rows,
        judgments,
        topic_ids=topic_ids,
        benchmark_profile=benchmark_profile,
        declared_max_candidate_depth=budget_k,
        bootstrap_samples=bootstrap_samples,
    )
    artifact: dict[str, object] = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "artifact_type": "taim-final-scorecard",
        "artifact_version": SCORECARD_PROFILE_VERSION,
        "scorecard_profile": shared_scorecard_profile(),
        "evaluator_version": SCORECARD_EVALUATOR_VERSION,
        **identity.to_dict(),
        "benchmark_profile": json.loads(
            json.dumps(benchmark_profile, allow_nan=False, ensure_ascii=False)
        ),
        "ranking_artifact": {
            "artifact_role": "primary_ranking",
            "artifact_sha256": ranking_artifact_sha256,
            "pipeline_depth": pipeline_depth,
            "declared_max_candidate_depth": budget_k,
        },
        "score_semantics": _undeclared_score_semantics(),
        "effectiveness": effectiveness,
        "deployment_diagnostics": _closed_deployment_diagnostics(),
        "operational_reliability": _partial_operational_reliability(float(runtime_seconds)),
        "trec_eval_parity": _closed_trec_eval_parity(),
    }
    if bound_evaluator_artifacts is not None:
        artifact["evaluator_artifacts"] = bound_evaluator_artifacts
    validate_scorecard_artifact(artifact)
    return artifact


def read_scorecard(path: str | Path) -> dict[str, object]:
    """Read and validate a current final scorecard artifact."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid scorecard JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError("final scorecard must be a JSON object")
    validate_scorecard_artifact(payload)
    return payload


def _display_metric(aggregate: Mapping[str, object], metric_name: str) -> str:
    value = aggregate.get(metric_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaValidationError(f"scorecard aggregate metric {metric_name!r} is missing")
    return f"{float(value):.4f}"


def _numeric_value(payload: Mapping[str, object], field_name: str) -> float:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise SchemaValidationError(f"scorecard field {field_name!r} must be finite numeric")
    return float(value)


def render_scorecard_markdown(payload: Mapping[str, object]) -> str:
    """Render the concise human-readable companion to a final scorecard."""

    validate_scorecard_artifact(payload)
    effectiveness = cast(Mapping[str, object], payload["effectiveness"])
    aggregate = cast(Mapping[str, object], effectiveness["aggregate"])
    mean = cast(Mapping[str, object], aggregate["mean"])
    first = cast(Mapping[str, object], effectiveness["first_eligible"])
    no_eligible = cast(Mapping[str, object], first["no_eligible_in_top_k_rate"])
    reliability = cast(Mapping[str, object], payload["operational_reliability"])
    profile = cast(Mapping[str, object], payload["benchmark_profile"])
    lines = [
        f"# TAIM final scorecard: {payload['run_id']}",
        "",
        (
            f"System `{payload['system_id']}` under Benchmark Profile "
            f"`{profile['profile_id']}`. Primary endpoint: graded nDCG@10."
        ),
        "",
        "## Effectiveness",
        "",
        "| Metric | @5 | @10 | @20 |",
        "| --- | ---: | ---: | ---: |",
    ]
    metric_rows = (
        ("Graded nDCG", "graded_ndcg"),
        ("Eligible precision", "eligible_precision"),
        ("Eligible recall", "eligible_recall"),
        ("Relevant-or-eligible precision", "relevant_or_eligible_precision"),
        ("Relevant-or-eligible recall", "relevant_or_eligible_recall"),
    )
    for label, prefix in metric_rows:
        values = [
            _display_metric(mean, f"{prefix}_at_{cutoff}") for cutoff in FINAL_RANKING_CUTOFFS
        ]
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            f"Eligible MRR: {_display_metric(mean, 'eligible_mrr')}",
            f"Median first eligible rank: {first['median_rank']}",
            (
                "Topics without an eligible result in the top 5/10/20: "
                f"{_numeric_value(no_eligible, '5'):.1%} / "
                f"{_numeric_value(no_eligible, '10'):.1%} / "
                f"{_numeric_value(no_eligible, '20'):.1%}."
            ),
            "",
            "## Deployment diagnostics",
            "",
            (
                "No deployment threshold claim was emitted. Pooled Judgments are incomplete, "
                "and this ranking artifact does not declare comparable score semantics."
            ),
            "",
            "## Operational reliability",
            "",
            (
                f"Runtime: {_numeric_value(reliability, 'runtime_seconds'):.3f} seconds. Other "
                "operational "
                "fields were not available from the closed ranking artifact."
            ),
            "",
            "## TREC evaluator parity",
            "",
            (
                "Not attested in this artifact; the publication gate for overlapping standard "
                "metrics remains closed until the locked trec_eval check passes."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "FINAL_RANKING_CUTOFFS",
    "SCORECARD_EVALUATOR_VERSION",
    "SCORECARD_PROFILE_VERSION",
    "SCORECARD_SCHEMA_VERSION",
    "ScoreSemantics",
    "ScorecardEvaluationBinding",
    "ScorecardRunIdentity",
    "ScoredPair",
    "build_final_scorecard_artifact",
    "evaluate_calibration_scorecard",
    "evaluate_filtering_scorecard",
    "evaluate_final_ranking_scorecard",
    "evaluate_threshold_scorecard",
    "read_scorecard",
    "render_scorecard_markdown",
    "shared_scorecard_profile",
    "validate_scorecard_artifact",
]
