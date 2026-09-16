"""TrialGPT-owned evaluation extension and diagnostic artifact."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass

from taim.artifacts import (
    RunEvaluationArtifact,
    RunEvaluationContext,
    RunEvaluationExtensionResult,
)
from taim.schemas import Candidate, RelevanceJudgment

MetricsDictionary = dict[str, object]


@dataclass(frozen=True, slots=True)
class _QrelLabelCounts:
    label_0: int = 0
    label_1: int = 0
    label_2: int = 0

    @classmethod
    def from_labels(cls, labels: Iterable[int | None]) -> _QrelLabelCounts:
        values = tuple(labels)
        return cls(
            label_0=sum(value == 0 for value in values),
            label_1=sum(value == 1 for value in values),
            label_2=sum(value == 2 for value in values),
        )

    def __add__(self, other: _QrelLabelCounts) -> _QrelLabelCounts:
        return _QrelLabelCounts(
            label_0=self.label_0 + other.label_0,
            label_1=self.label_1 + other.label_1,
            label_2=self.label_2 + other.label_2,
        )

    def to_dict(self) -> dict[str, int]:
        return {"0": self.label_0, "1": self.label_1, "2": self.label_2}


@dataclass(frozen=True, slots=True)
class _TrialGPTPipelineCounts:
    judged_eligible_total: int = 0
    judged_eligible_in_retrieval: int = 0
    judged_eligible_in_llm_candidates: int = 0
    judged_eligible_lost_at_llm_cap: int = 0
    judged_eligible_missed_at_retrieval_depth: int = 0
    judged_llm_candidate_count: int = 0
    semantic_abstention_count_in_judged_candidates: int = 0
    label_counts_total: _QrelLabelCounts = _QrelLabelCounts()
    label_counts_in_retrieval: _QrelLabelCounts = _QrelLabelCounts()
    label_counts_in_llm_candidates: _QrelLabelCounts = _QrelLabelCounts()
    label_counts_missed_at_retrieval_depth: _QrelLabelCounts = _QrelLabelCounts()
    label_counts_lost_at_llm_cap: _QrelLabelCounts = _QrelLabelCounts()

    def __add__(self, other: _TrialGPTPipelineCounts) -> _TrialGPTPipelineCounts:
        return _TrialGPTPipelineCounts(
            judged_eligible_total=self.judged_eligible_total + other.judged_eligible_total,
            judged_eligible_in_retrieval=(
                self.judged_eligible_in_retrieval + other.judged_eligible_in_retrieval
            ),
            judged_eligible_in_llm_candidates=(
                self.judged_eligible_in_llm_candidates + other.judged_eligible_in_llm_candidates
            ),
            judged_eligible_lost_at_llm_cap=(
                self.judged_eligible_lost_at_llm_cap + other.judged_eligible_lost_at_llm_cap
            ),
            judged_eligible_missed_at_retrieval_depth=(
                self.judged_eligible_missed_at_retrieval_depth
                + other.judged_eligible_missed_at_retrieval_depth
            ),
            judged_llm_candidate_count=(
                self.judged_llm_candidate_count + other.judged_llm_candidate_count
            ),
            semantic_abstention_count_in_judged_candidates=(
                self.semantic_abstention_count_in_judged_candidates
                + other.semantic_abstention_count_in_judged_candidates
            ),
            label_counts_total=self.label_counts_total + other.label_counts_total,
            label_counts_in_retrieval=(
                self.label_counts_in_retrieval + other.label_counts_in_retrieval
            ),
            label_counts_in_llm_candidates=(
                self.label_counts_in_llm_candidates + other.label_counts_in_llm_candidates
            ),
            label_counts_missed_at_retrieval_depth=(
                self.label_counts_missed_at_retrieval_depth
                + other.label_counts_missed_at_retrieval_depth
            ),
            label_counts_lost_at_llm_cap=(
                self.label_counts_lost_at_llm_cap + other.label_counts_lost_at_llm_cap
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "judged_eligible_total": self.judged_eligible_total,
            "judged_eligible_in_retrieval": self.judged_eligible_in_retrieval,
            "judged_eligible_in_llm_candidates": self.judged_eligible_in_llm_candidates,
            "judged_eligible_lost_at_llm_cap": self.judged_eligible_lost_at_llm_cap,
            "judged_eligible_missed_at_retrieval_depth": (
                self.judged_eligible_missed_at_retrieval_depth
            ),
            "judged_llm_candidate_count": self.judged_llm_candidate_count,
            "semantic_abstention_count_in_judged_candidates": (
                self.semantic_abstention_count_in_judged_candidates
            ),
            "label_counts_total": self.label_counts_total.to_dict(),
            "label_counts_in_retrieval": self.label_counts_in_retrieval.to_dict(),
            "label_counts_in_llm_candidates": self.label_counts_in_llm_candidates.to_dict(),
            "label_counts_missed_at_retrieval_depth": (
                self.label_counts_missed_at_retrieval_depth.to_dict()
            ),
            "label_counts_lost_at_llm_cap": self.label_counts_lost_at_llm_cap.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _ThresholdDiagnosticCounts:
    true_positive: int = 0
    false_positive_qrel_0: int = 0
    false_positive_qrel_1: int = 0
    resolved_eligible: int = 0

    def __add__(self, other: _ThresholdDiagnosticCounts) -> _ThresholdDiagnosticCounts:
        return _ThresholdDiagnosticCounts(
            true_positive=self.true_positive + other.true_positive,
            false_positive_qrel_0=(self.false_positive_qrel_0 + other.false_positive_qrel_0),
            false_positive_qrel_1=(self.false_positive_qrel_1 + other.false_positive_qrel_1),
            resolved_eligible=self.resolved_eligible + other.resolved_eligible,
        )

    @property
    def predicted_positive(self) -> int:
        return self.true_positive + self.false_positive_qrel_0 + self.false_positive_qrel_1

    def to_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive_qrel_0": self.false_positive_qrel_0,
            "false_positive_qrel_1": self.false_positive_qrel_1,
            "resolved_eligible": self.resolved_eligible,
        }


def evaluate_trialgpt_pipeline_diagnostics(
    retrieval_candidates: Iterable[Candidate],
    llm_candidates: Iterable[Candidate],
    judgments: Iterable[RelevanceJudgment],
    *,
    topic_ids: Iterable[str],
    score_thresholds: Sequence[float] = (0.0, 1.0, 2.0),
    abstention_score: float | None = None,
    abstention_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> MetricsDictionary:
    """Report retrieval losses and explicitly diagnostic eligibility thresholds."""

    thresholds = tuple(float(value) for value in score_thresholds)
    if not thresholds or any(not math.isfinite(value) for value in thresholds):
        raise ValueError("score_thresholds must contain finite values")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("score_thresholds must be sorted and unique")
    selected_topics = tuple(topic_ids)
    retrieval_by_topic: dict[str, list[Candidate]] = defaultdict(list)
    llm_by_topic: dict[str, list[Candidate]] = defaultdict(list)
    relevance_by_topic: dict[str, dict[str, int]] = defaultdict(dict)
    for candidate in retrieval_candidates:
        retrieval_by_topic[candidate.topic_id].append(candidate)
    for candidate in llm_candidates:
        llm_by_topic[candidate.topic_id].append(candidate)
    for judgment in judgments:
        relevance_by_topic[judgment.topic_id][judgment.trial_id] = judgment.label

    all_llm_rows = [row for rows in llm_by_topic.values() for row in rows]
    all_llm_pairs = {(row.topic_id, row.trial_id) for row in all_llm_rows}
    all_retrieval_pairs = {
        (row.topic_id, row.trial_id) for rows in retrieval_by_topic.values() for row in rows
    }
    if not all_llm_pairs.issubset(all_retrieval_pairs):
        raise ValueError("LLM candidates must be a subset of retrieval candidates")
    missing_abstentions = abstention_pairs - all_llm_pairs
    if missing_abstentions:
        raise ValueError("semantic abstention identities are missing from the LLM ranking")
    if abstention_score is not None:
        mismatched_abstentions = [
            row
            for row in all_llm_rows
            if ((row.topic_id, row.trial_id) in abstention_pairs)
            != (float(row.score) == abstention_score)
        ]
        if mismatched_abstentions:
            raise ValueError("semantic abstention identities do not match the declared score")

    per_topic: dict[str, dict[str, object]] = {}
    aggregate_counts = _TrialGPTPipelineCounts()
    threshold_totals = {threshold: _ThresholdDiagnosticCounts() for threshold in thresholds}
    for topic_id in selected_topics:
        relevance = relevance_by_topic.get(topic_id, {})
        eligible = {trial_id for trial_id, label in relevance.items() if label == 2}
        retrieval_rows = sorted(retrieval_by_topic.get(topic_id, ()), key=lambda row: row.rank)
        llm_rows = sorted(llm_by_topic.get(topic_id, ()), key=lambda row: row.rank)
        retrieval_ids = {row.trial_id for row in retrieval_rows}
        llm_ids = {row.trial_id for row in llm_rows}
        judged_llm = [row for row in llm_rows if row.trial_id in relevance]
        abstained = [row for row in judged_llm if (row.topic_id, row.trial_id) in abstention_pairs]
        resolved_judged = [row for row in judged_llm if row not in abstained]
        resolved_eligible = sum(relevance[row.trial_id] == 2 for row in resolved_judged)
        threshold_metrics: dict[str, object] = {}
        for threshold in thresholds:
            predicted_positive = [row for row in resolved_judged if float(row.score) >= threshold]
            true_positive = sum(relevance[row.trial_id] == 2 for row in predicted_positive)
            false_positive_qrel_0 = sum(relevance[row.trial_id] == 0 for row in predicted_positive)
            false_positive_qrel_1 = sum(relevance[row.trial_id] == 1 for row in predicted_positive)
            threshold_totals[threshold] += _ThresholdDiagnosticCounts(
                true_positive=true_positive,
                false_positive_qrel_0=false_positive_qrel_0,
                false_positive_qrel_1=false_positive_qrel_1,
                resolved_eligible=resolved_eligible,
            )
            threshold_metrics[str(threshold)] = {
                "comparator": "score >= threshold",
                "positive_gold_label": 2,
                "negative_gold_labels": [0, 1],
                "unjudged_policy": "exclude",
                "semantic_abstention_policy": "exclude_from_resolved; count_negative_end_to_end",
                "predicted_positive_count": len(predicted_positive),
                "true_positive": true_positive,
                "false_positive_qrel_0": false_positive_qrel_0,
                "false_positive_qrel_1": false_positive_qrel_1,
                "precision_among_judged_resolved": (
                    true_positive / len(predicted_positive) if predicted_positive else 0.0
                ),
                "recall_among_judged_resolved_eligible": (
                    true_positive / resolved_eligible if resolved_eligible else 0.0
                ),
                "end_to_end_recall_of_all_judged_eligible": (
                    true_positive / len(eligible) if eligible else 0.0
                ),
            }
        topic_counts = _TrialGPTPipelineCounts(
            judged_eligible_total=len(eligible),
            judged_eligible_in_retrieval=len(eligible & retrieval_ids),
            judged_eligible_in_llm_candidates=len(eligible & llm_ids),
            judged_eligible_lost_at_llm_cap=len((eligible & retrieval_ids) - llm_ids),
            judged_eligible_missed_at_retrieval_depth=len(eligible - retrieval_ids),
            judged_llm_candidate_count=len(judged_llm),
            semantic_abstention_count_in_judged_candidates=len(abstained),
            label_counts_total=_QrelLabelCounts.from_labels(relevance.values()),
            label_counts_in_retrieval=_QrelLabelCounts.from_labels(
                relevance.get(row.trial_id) for row in retrieval_rows
            ),
            label_counts_in_llm_candidates=_QrelLabelCounts.from_labels(
                relevance.get(row.trial_id) for row in llm_rows
            ),
            label_counts_missed_at_retrieval_depth=_QrelLabelCounts.from_labels(
                value for trial_id, value in relevance.items() if trial_id not in retrieval_ids
            ),
            label_counts_lost_at_llm_cap=_QrelLabelCounts.from_labels(
                relevance.get(trial_id) for trial_id in retrieval_ids if trial_id not in llm_ids
            ),
        )
        aggregate_counts += topic_counts
        per_topic[topic_id] = {
            **topic_counts.to_dict(),
            "score_thresholds": threshold_metrics,
        }

    aggregate_thresholds: dict[str, object] = {}
    eligible_total = aggregate_counts.judged_eligible_total
    for threshold in thresholds:
        counts = threshold_totals[threshold]
        aggregate_predicted_positive = counts.predicted_positive
        aggregate_thresholds[str(threshold)] = {
            **counts.to_dict(),
            "precision_among_judged_resolved": (
                counts.true_positive / aggregate_predicted_positive
                if aggregate_predicted_positive
                else 0.0
            ),
            "recall_among_judged_resolved_eligible": (
                counts.true_positive / counts.resolved_eligible if counts.resolved_eligible else 0.0
            ),
            "end_to_end_recall_of_all_judged_eligible": (
                counts.true_positive / eligible_total if eligible_total else 0.0
            ),
        }
    return {
        "policy": {
            "classification_status": "diagnostic_threshold_not_official_trialgpt_decision",
            "positive_gold_label": 2,
            "negative_gold_labels": [0, 1],
            "unjudged_policy": "exclude",
            "semantic_abstention_score": abstention_score,
        },
        "aggregate": {**aggregate_counts.to_dict(), "score_thresholds": aggregate_thresholds},
        "per_topic": per_topic,
    }


class TrialGPTEvaluationExtension:
    """Add paper-style generation coverage and pipeline diagnostics."""

    # Run re-evaluation selects this extension for runs under this profile.
    profile_id = "trialgpt-paper-style"

    def evaluate(self, context: RunEvaluationContext) -> RunEvaluationExtensionResult:
        if context.benchmark_profile.get("profile_id") != self.profile_id:
            raise ValueError(f"TrialGPT evaluation requires the {self.profile_id} profile")
        stored = context.stored_run
        profile_evaluation = deepcopy(dict(context.profile_evaluation))
        raw_generation = stored.manifest.configuration.get("per_topic_generation", [])
        generation_by_topic = {
            item["topic_id"]: item
            for item in (raw_generation if isinstance(raw_generation, list) else [])
            if isinstance(item, dict) and isinstance(item.get("topic_id"), str)
        }
        per_topic_evaluation = profile_evaluation.get("per_topic")
        if isinstance(per_topic_evaluation, dict):
            for topic_id, topic_metrics in per_topic_evaluation.items():
                if isinstance(topic_metrics, dict) and topic_id in generation_by_topic:
                    generation = dict(generation_by_topic[topic_id])
                    semantic_abstentions = generation.get("semantic_abstention_count")
                    llm_candidate_count = generation.get("llm_candidate_count")
                    if (
                        isinstance(semantic_abstentions, int)
                        and isinstance(llm_candidate_count, int)
                        and llm_candidate_count > 0
                    ):
                        generation["generation_score_coverage_complete"] = semantic_abstentions == 0
                        generation["resolved_candidate_fraction"] = (
                            llm_candidate_count - semantic_abstentions
                        ) / llm_candidate_count
                    topic_metrics["generation"] = generation
        try:
            retrieval_ranked = next(
                stage.candidates
                for stage in stored.stage_rankings
                if stage.name == "hybrid-retrieval"
            )
            llm_ranked = next(
                stage.candidates
                for stage in stored.stage_rankings
                if stage.name == "trialgpt-llm-ranking"
            )
        except StopIteration as exc:
            raise ValueError(
                "TrialGPT paper-style evaluation requires retrieval and LLM stage rankings"
            ) from exc
        diagnostics = evaluate_trialgpt_pipeline_diagnostics(
            retrieval_ranked,
            llm_ranked,
            context.judgments,
            topic_ids=context.topic_ids,
            abstention_score=context.abstention_score,
            abstention_pairs=context.abstention_pairs,
        )
        profile_evaluation["trialgpt_pipeline_diagnostics"] = diagnostics
        artifact = RunEvaluationArtifact(
            path="trialgpt-diagnostic-evaluation.json",
            payload={
                "schema_version": "1.0",
                "artifact_type": "trialgpt-diagnostic-evaluation",
                "run_id": stored.manifest.run_id,
                "system_id": stored.manifest.system_id,
                "prepared_snapshot_id": stored.manifest.prepared_snapshot_id,
                "system_input_id": stored.manifest.system_input_id,
                "evaluation_package_id": stored.manifest.evaluation_package_id,
                "diagnostics": diagnostics,
            },
        )
        return RunEvaluationExtensionResult(
            profile_evaluation=profile_evaluation,
            artifacts=(artifact,),
        )


TRIALGPT_EVALUATION_EXTENSION = TrialGPTEvaluationExtension()

__all__ = [
    "TRIALGPT_EVALUATION_EXTENSION",
    "TrialGPTEvaluationExtension",
    "evaluate_trialgpt_pipeline_diagnostics",
]
