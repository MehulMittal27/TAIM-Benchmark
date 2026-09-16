"""Deterministic retrieval metrics for TAIM candidate runs.

The historical profile evaluator follows the TREC Clinical Trials labels: 0 is
not relevant, 1 is condition-relevant but excluded, and 2 is eligible. Generic
callers use an explicit Judgment Scheme. Recall with no positive judgments is
defined as zero, keeping macro averages explicit and JSON-friendly.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from taim.judgments import JudgmentScheme
from taim.schemas import Candidate, RelevanceJudgment

MetricValue: TypeAlias = float | int | None
MetricsDictionary: TypeAlias = dict[str, object]
UnjudgedPolicy: TypeAlias = Literal["retain_as_zero_gain", "condense_before_cutoff"]
PrecisionDenominator: TypeAlias = Literal[
    "fixed_cutoff", "available_condensed_results", "maximum_graded_gain"
]


@dataclass(frozen=True, slots=True)
class TopicMetrics:
    """Metrics for one topic at one cutoff."""

    eligible_recall: float
    relevant_or_eligible_recall: float
    ndcg: float
    reciprocal_rank: float
    first_eligible_rank: int | None

    def to_dict(self, *, k: int) -> dict[str, MetricValue]:
        return {
            f"eligible_recall_at_{k}": self.eligible_recall,
            f"relevant_or_eligible_recall_at_{k}": self.relevant_or_eligible_recall,
            f"ndcg_at_{k}": self.ndcg,
            "mrr": self.reciprocal_rank,
            "first_eligible_rank": self.first_eligible_rank,
        }


@dataclass(frozen=True, slots=True)
class SchemeTopicMetrics:
    """Dataset-scheme metrics for one topic at one cutoff."""

    recall: Mapping[str, float]
    ndcg: float
    reciprocal_rank: float
    first_relevant_rank: int | None

    def to_dict(self, *, k: int) -> dict[str, object]:
        return {
            "recall": {f"{name}_at_{k}": value for name, value in self.recall.items()},
            f"ndcg_at_{k}": self.ndcg,
            "mrr": self.reciprocal_rank,
            "first_relevant_rank": self.first_relevant_rank,
        }


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def evaluate_scheme_topic(
    ranked_item_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    scheme: JudgmentScheme,
    k: int,
) -> SchemeTopicMetrics:
    """Evaluate one ranking using only the supplied Judgment Scheme semantics."""

    _validate_k(k)
    if not isinstance(scheme, JudgmentScheme):
        raise TypeError("scheme must be a JudgmentScheme")
    if len(ranked_item_ids) != len(set(ranked_item_ids)):
        raise ValueError("ranked item IDs must not contain duplicates")
    for item_id, label in relevance.items():
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("relevance item IDs must be non-empty strings")
        scheme.validate_label(label)
    retrieved = set(ranked_item_ids[:k])
    recall: dict[str, float] = {}
    for name, positive_labels in scheme.relevance_sets.items():
        positives = {item_id for item_id, label in relevance.items() if label in positive_labels}
        recall[name] = len(positives & retrieved) / len(positives) if positives else 0.0
    ranked_gains = [
        scheme.gains.get(relevance.get(item_id, -1), 0) for item_id in ranked_item_ids[:k]
    ]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ranked_gains, start=1))
    ideal_gains = sorted((scheme.gains[label] for label in relevance.values()), reverse=True)[:k]
    ideal_dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    reciprocal_labels = scheme.relevance_sets[scheme.reciprocal_rank_relevance_set]
    first_rank = next(
        (
            rank
            for rank, item_id in enumerate(ranked_item_ids, start=1)
            if relevance.get(item_id) in reciprocal_labels
        ),
        None,
    )
    return SchemeTopicMetrics(
        recall=recall,
        ndcg=dcg / ideal_dcg if ideal_dcg else 0.0,
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        first_relevant_rank=first_rank,
    )


def evaluate_scheme_run(
    candidates: Iterable[Candidate],
    judgments: Iterable[RelevanceJudgment],
    *,
    scheme: JudgmentScheme,
    topic_ids: Iterable[str] | None = None,
    k: int = 10,
    unjudged_policy: UnjudgedPolicy = "retain_as_zero_gain",
    precision_metric: str | None = None,
    precision_relevance_minimum: int | None = None,
    precision_denominator: PrecisionDenominator = "fixed_cutoff",
) -> MetricsDictionary:
    """Evaluate a closed candidate run under a dataset-supplied Judgment Scheme."""

    _validate_k(k)
    if unjudged_policy not in ("retain_as_zero_gain", "condense_before_cutoff"):
        raise ValueError(f"unsupported unjudged policy {unjudged_policy!r}")
    precision_relevance_set: str | None = None
    if precision_metric is None:
        if precision_relevance_minimum is not None:
            raise ValueError("precision_relevance_minimum requires a precision_metric")
    else:
        if not precision_metric.endswith("_precision"):
            raise ValueError("precision_metric must end with '_precision'")
        precision_relevance_set = precision_metric.removesuffix("_precision")
        if precision_relevance_set not in scheme.relevance_sets:
            raise ValueError("precision_metric does not name a Judgment Scheme relevance set")
        if precision_relevance_minimum not in (1, 2):
            raise ValueError("precision_relevance_minimum must be 1 or 2")
        expected_labels = frozenset(
            label for label in scheme.labels if label >= precision_relevance_minimum
        )
        if scheme.relevance_sets[precision_relevance_set] != expected_labels:
            raise ValueError(
                "precision_relevance_minimum does not match the Judgment Scheme relevance set"
            )
        if precision_denominator not in ("fixed_cutoff", "available_condensed_results"):
            raise ValueError("scheme precision denominator is unsupported")
    candidate_rows = tuple(candidates)
    judgment_rows = tuple(judgments)
    rankings: dict[str, list[Candidate]] = defaultdict(list)
    candidate_pairs: set[tuple[str, str]] = set()
    candidate_ranks: set[tuple[str, int]] = set()
    for candidate in candidate_rows:
        if not isinstance(candidate, Candidate):
            raise TypeError("candidates must contain Candidate instances")
        pair = (candidate.topic_id, candidate.trial_id)
        rank = (candidate.topic_id, candidate.rank)
        if pair in candidate_pairs or rank in candidate_ranks:
            raise ValueError("candidate run contains duplicate pairs or ranks")
        candidate_pairs.add(pair)
        candidate_ranks.add(rank)
        rankings[candidate.topic_id].append(candidate)
    if len({row.run_id for row in candidate_rows}) > 1:
        raise ValueError("candidates must share one run_id")
    if len({row.system_id for row in candidate_rows}) > 1:
        raise ValueError("candidates must share one system_id")
    relevance_by_topic: dict[str, dict[str, int]] = defaultdict(dict)
    for judgment in judgment_rows:
        if not isinstance(judgment, RelevanceJudgment):
            raise TypeError("judgments must contain RelevanceJudgment instances")
        scheme.validate_label(judgment.label)
        topic_relevance = relevance_by_topic[judgment.topic_id]
        if judgment.trial_id in topic_relevance:
            raise ValueError("Judgments contain a duplicate topic-item pair")
        topic_relevance[judgment.trial_id] = judgment.label
    selected_topics = (
        tuple(sorted(set(rankings) | set(relevance_by_topic)))
        if topic_ids is None
        else tuple(topic_ids)
    )
    if not selected_topics or len(selected_topics) != len(set(selected_topics)):
        raise ValueError("topic_ids must be non-empty and unique")
    per_topic: dict[str, dict[str, object]] = {}
    topic_metrics: list[SchemeTopicMetrics] = []
    precision_values: list[float] = []
    for topic_id in selected_topics:
        rows = sorted(rankings.get(topic_id, ()), key=lambda row: row.rank)
        relevance = relevance_by_topic.get(topic_id, {})
        if unjudged_policy == "condense_before_cutoff":
            ranked_ids = [row.trial_id for row in rows if row.trial_id in relevance]
        else:
            ranked_ids = []
            next_rank = 1
            for row in rows:
                while next_rank < row.rank:
                    ranked_ids.append(f"__taim_unretrieved_rank_{next_rank}__")
                    next_rank += 1
                ranked_ids.append(row.trial_id)
                next_rank += 1
        metrics = evaluate_scheme_topic(ranked_ids, relevance, scheme=scheme, k=k)
        topic_metrics.append(metrics)
        topic_payload = metrics.to_dict(k=k)
        if precision_metric is not None and precision_relevance_set is not None:
            selected = ranked_ids[:k]
            divisor = k if precision_denominator == "fixed_cutoff" else len(selected)
            positive_labels = scheme.relevance_sets[precision_relevance_set]
            precision = (
                sum(relevance.get(item_id) in positive_labels for item_id in selected) / divisor
                if divisor
                else 0.0
            )
            topic_payload[f"{precision_metric}_at_{k}"] = precision
            precision_values.append(precision)
        per_topic[topic_id] = topic_payload
    aggregate: dict[str, object] = {
        "recall": {
            f"{name}_at_{k}": _mean(metric.recall[name] for metric in topic_metrics)
            for name in scheme.relevance_sets
        },
        f"ndcg_at_{k}": _mean(metric.ndcg for metric in topic_metrics),
        "mrr": _mean(metric.reciprocal_rank for metric in topic_metrics),
    }
    if precision_metric is not None:
        aggregate[f"{precision_metric}_at_{k}"] = _mean(precision_values)
    result: MetricsDictionary = {
        "judgment_scheme": scheme.to_dict(),
        "topic_count": len(selected_topics),
        "cutoff": k,
        "aggregate": aggregate,
        "per_topic": per_topic,
    }
    if precision_metric is not None:
        result["precision_policy"] = {
            "metric": precision_metric,
            "relevance_set": precision_relevance_set,
            "relevance_minimum": precision_relevance_minimum,
            "denominator": precision_denominator,
        }
    return result


def _recall_at_k(
    ranked_trial_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    minimum_label: int,
    k: int,
) -> float:
    positive = {trial_id for trial_id, label in relevance.items() if label >= minimum_label}
    if not positive:
        return 0.0
    retrieved = set(ranked_trial_ids[:k])
    return len(positive & retrieved) / len(positive)


def eligible_recall_at_k(
    ranked_trial_ids: Sequence[str], relevance: Mapping[str, int], *, k: int = 5
) -> float:
    """Recall@k where only label 2 is relevant."""

    _validate_k(k)
    return _recall_at_k(ranked_trial_ids, relevance, minimum_label=2, k=k)


def relevant_or_eligible_recall_at_k(
    ranked_trial_ids: Sequence[str], relevance: Mapping[str, int], *, k: int = 5
) -> float:
    """Recall@k where labels 1 and 2 are relevant."""

    _validate_k(k)
    return _recall_at_k(ranked_trial_ids, relevance, minimum_label=1, k=k)


def precision_at_k(
    ranked_trial_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    minimum_label: int,
    k: int = 5,
    denominator: PrecisionDenominator = "fixed_cutoff",
) -> float:
    """Return binary precision with an explicit relevance and denominator policy."""

    _validate_k(k)
    if minimum_label not in (1, 2):
        raise ValueError("minimum_label must be 1 or 2")
    if denominator not in ("fixed_cutoff", "available_condensed_results"):
        raise ValueError(f"unsupported precision denominator {denominator!r}")
    selected = ranked_trial_ids[:k]
    divisor = k if denominator == "fixed_cutoff" else len(selected)
    if divisor == 0:
        return 0.0
    relevant = sum(relevance.get(trial_id, 0) >= minimum_label for trial_id in selected)
    return relevant / divisor


def graded_precision_at_k(
    ranked_trial_ids: Sequence[str], relevance: Mapping[str, int], *, k: int = 10
) -> float:
    """Return paper-style graded precision as observed gain over maximum gain ``2 * k``."""

    _validate_k(k)
    _validate_relevance_mapping(relevance)
    return sum(relevance.get(trial_id, 0) for trial_id in ranked_trial_ids[:k]) / (2 * k)


def _binary_auroc(scored_labels: Sequence[tuple[float, int]]) -> float | None:
    positives = [score for score, label in scored_labels if label >= 1]
    negatives = [score for score, label in scored_labels if label == 0]
    if not positives or not negatives:
        return None
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return favorable / (len(positives) * len(negatives))


def evaluate_graded_condensed_run(
    candidates: Iterable[Candidate],
    judgments: Iterable[RelevanceJudgment],
    *,
    topic_ids: Iterable[str],
    k: int = 10,
    abstention_score: float | None = None,
    abstention_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> MetricsDictionary:
    """Evaluate judged-condensed and literal raw-cutoff graded ranking views."""

    _validate_k(k)
    candidate_rows = tuple(candidates)
    judgment_rows = tuple(judgments)
    selected_topics = tuple(topic_ids)
    relevance_by_topic: dict[str, dict[str, int]] = defaultdict(dict)
    for judgment in judgment_rows:
        relevance_by_topic[judgment.topic_id][judgment.trial_id] = judgment.label
    rankings: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidate_rows:
        rankings[candidate.topic_id].append(candidate)
    per_topic: dict[str, dict[str, object]] = {}
    ndcg_values: list[float] = []
    graded_precision_values: list[float] = []
    raw_ndcg_values: list[float] = []
    raw_graded_precision_values: list[float] = []
    auroc_values: list[float] = []
    resolved_only_auroc_values: list[float] = []
    coverage_totals = {
        "candidate_count": 0,
        "semantic_abstention_count": 0,
        "judged_candidate_count": 0,
        "judged_semantic_abstention_count": 0,
        "topics_with_complete_judged_score_coverage": 0,
    }
    abstention_label_totals = {"0": 0, "1": 0, "2": 0, "unjudged": 0}
    for topic_id in selected_topics:
        rows = sorted(rankings.get(topic_id, ()), key=lambda row: row.rank)
        relevance = relevance_by_topic.get(topic_id, {})
        condensed = [row for row in rows if row.trial_id in relevance]
        condensed_ids = [row.trial_id for row in condensed]
        raw_ids = [row.trial_id for row in rows]
        ndcg = ndcg_at_k(condensed_ids, relevance, k=k)
        graded_precision = graded_precision_at_k(condensed_ids, relevance, k=k)
        raw_ndcg = ndcg_at_k(raw_ids, relevance, k=k)
        raw_graded_precision = graded_precision_at_k(raw_ids, relevance, k=k)
        abstained = [row for row in condensed if (row.topic_id, row.trial_id) in abstention_pairs]
        all_abstained = [row for row in rows if (row.topic_id, row.trial_id) in abstention_pairs]
        finite_score_coverage_complete = all(math.isfinite(float(row.score)) for row in condensed)
        candidate_score_coverage_complete = finite_score_coverage_complete and not abstained
        scored_labels = [(float(row.score), relevance[row.trial_id]) for row in condensed]
        resolved_scored_labels = [
            (float(row.score), relevance[row.trial_id]) for row in condensed if row not in abstained
        ]
        auroc = _binary_auroc(scored_labels) if candidate_score_coverage_complete else None
        resolved_only_auroc = _binary_auroc(resolved_scored_labels)
        labels = {str(label): 0 for label in (0, 1, 2)}
        for row in condensed:
            labels[str(relevance[row.trial_id])] += 1
        raw_top_k_labels = {"0": 0, "1": 0, "2": 0, "unjudged": 0}
        for row in rows[:k]:
            label = relevance.get(row.trial_id)
            raw_top_k_labels["unjudged" if label is None else str(label)] += 1
        condensed_top_k_labels = {str(label): 0 for label in (0, 1, 2)}
        for row in condensed[:k]:
            condensed_top_k_labels[str(relevance[row.trial_id])] += 1
        abstention_label_counts = {"0": 0, "1": 0, "2": 0, "unjudged": 0}
        for row in all_abstained:
            label = relevance.get(row.trial_id)
            abstention_label_counts["unjudged" if label is None else str(label)] += 1
        coverage_totals["candidate_count"] += len(rows)
        coverage_totals["semantic_abstention_count"] += len(all_abstained)
        coverage_totals["judged_candidate_count"] += len(condensed)
        coverage_totals["judged_semantic_abstention_count"] += len(abstained)
        coverage_totals["topics_with_complete_judged_score_coverage"] += int(
            candidate_score_coverage_complete
        )
        for abstention_label_key, count in abstention_label_counts.items():
            abstention_label_totals[abstention_label_key] += count
        per_topic[topic_id] = {
            f"ndcg_at_{k}": ndcg,
            f"graded_precision_at_{k}": graded_precision,
            "auroc": auroc,
            "resolved_only_auroc_sensitivity": resolved_only_auroc,
            "judged_trials_in_candidate_ranking": len(condensed),
            "retained_after_judged_only_condensation": len(condensed),
            "label_counts": labels,
            "finite_score_coverage_complete": finite_score_coverage_complete,
            "judged_candidate_score_coverage_complete": candidate_score_coverage_complete,
            "candidate_score_coverage_complete": candidate_score_coverage_complete,
            "semantic_abstention_count_in_judged_candidates": len(abstained),
            "semantic_abstention_label_counts": abstention_label_counts,
            "candidate_score_coverage": {
                "candidate_count": len(rows),
                "resolved_candidate_count": len(rows) - len(all_abstained),
                "resolved_candidate_fraction": (
                    (len(rows) - len(all_abstained)) / len(rows) if rows else 1.0
                ),
                "judged_candidate_count": len(condensed),
                "judged_resolved_candidate_count": len(condensed) - len(abstained),
                "judged_resolved_candidate_fraction": (
                    (len(condensed) - len(abstained)) / len(condensed) if condensed else 1.0
                ),
            },
            "raw_top_k": {
                f"ndcg_at_{k}": raw_ndcg,
                f"graded_precision_at_{k}": raw_graded_precision,
                "label_counts": raw_top_k_labels,
                "ranking_view": "literal_first_k_before_judgment_condensation",
            },
            "judged_condensed_top_k": {
                f"ndcg_at_{k}": ndcg,
                f"graded_precision_at_{k}": graded_precision,
                "label_counts": condensed_top_k_labels,
                "ranking_view": "remove_unjudged_then_take_first_k",
            },
        }
        ndcg_values.append(ndcg)
        graded_precision_values.append(graded_precision)
        raw_ndcg_values.append(raw_ndcg)
        raw_graded_precision_values.append(raw_graded_precision)
        if auroc is not None:
            auroc_values.append(auroc)
        if resolved_only_auroc is not None:
            resolved_only_auroc_values.append(resolved_only_auroc)
    return {
        "cutoffs": [k],
        "topic_count": len(selected_topics),
        "policy": {
            "unjudged": "condense_before_cutoff",
            "gain_mapping": {"0": 0, "1": 1, "2": 2},
            "ndcg_discount": "log2(rank + 1)",
            "graded_precision": "sum(gain) / (2 * cutoff)",
            "auroc_positive_labels": [1, 2],
            "auroc_requires_both_classes": True,
            "auroc_requires_complete_judged_candidate_scores": True,
            "semantic_abstention_score": abstention_score,
            "raw_top_k_view": "literal_first_k_before_judgment_condensation",
            "judged_condensed_top_k_view": "remove_unjudged_then_take_first_k",
        },
        "aggregate": {
            "score_coverage": {
                **coverage_totals,
                "resolved_candidate_count": (
                    coverage_totals["candidate_count"]
                    - coverage_totals["semantic_abstention_count"]
                ),
                "resolved_candidate_fraction": (
                    (
                        coverage_totals["candidate_count"]
                        - coverage_totals["semantic_abstention_count"]
                    )
                    / coverage_totals["candidate_count"]
                    if coverage_totals["candidate_count"]
                    else 1.0
                ),
                "judged_resolved_candidate_count": (
                    coverage_totals["judged_candidate_count"]
                    - coverage_totals["judged_semantic_abstention_count"]
                ),
                "judged_resolved_candidate_fraction": (
                    (
                        coverage_totals["judged_candidate_count"]
                        - coverage_totals["judged_semantic_abstention_count"]
                    )
                    / coverage_totals["judged_candidate_count"]
                    if coverage_totals["judged_candidate_count"]
                    else 1.0
                ),
                "semantic_abstention_label_counts": abstention_label_totals,
            },
            "mean": {
                f"ndcg_at_{k}": _mean(ndcg_values),
                f"graded_precision_at_{k}": _mean(graded_precision_values),
                f"raw_ndcg_at_{k}": _mean(raw_ndcg_values),
                f"raw_graded_precision_at_{k}": _mean(raw_graded_precision_values),
                "auroc": _mean(auroc_values) if len(auroc_values) == len(selected_topics) else None,
                "resolved_only_auroc_sensitivity": (
                    _mean(resolved_only_auroc_values)
                    if len(resolved_only_auroc_values) == len(selected_topics)
                    else None
                ),
            },
            "median": {
                f"ndcg_at_{k}": float(statistics.median(ndcg_values)),
                f"graded_precision_at_{k}": float(statistics.median(graded_precision_values)),
                f"raw_ndcg_at_{k}": float(statistics.median(raw_ndcg_values)),
                f"raw_graded_precision_at_{k}": float(
                    statistics.median(raw_graded_precision_values)
                ),
                "auroc": (
                    float(statistics.median(auroc_values))
                    if len(auroc_values) == len(selected_topics)
                    else None
                ),
                "resolved_only_auroc_sensitivity": (
                    float(statistics.median(resolved_only_auroc_values))
                    if len(resolved_only_auroc_values) == len(selected_topics)
                    else None
                ),
            },
        },
        "per_topic": per_topic,
    }


def _gain(label: int) -> float:
    # TREC Clinical Trials uses the judgment label directly as graded gain:
    # eligible=2, condition-relevant-but-excluded=1, and not relevant=0.
    return float(label)


def ndcg_at_k(
    ranked_trial_ids: Sequence[str], relevance: Mapping[str, int], *, k: int = 5
) -> float:
    """Graded nDCG@k using the TREC label as gain and log2 discount."""

    _validate_k(k)
    dcg = sum(
        _gain(relevance.get(trial_id, 0)) / math.log2(rank + 1)
        for rank, trial_id in enumerate(ranked_trial_ids[:k], start=1)
    )
    ideal_labels = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = sum(
        _gain(label) / math.log2(rank + 1) for rank, label in enumerate(ideal_labels, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def first_eligible_rank(
    ranked_trial_ids: Sequence[str], relevance: Mapping[str, int]
) -> int | None:
    """Return the one-based rank of the first label-2 trial, or ``None``."""

    for rank, trial_id in enumerate(ranked_trial_ids, start=1):
        if relevance.get(trial_id, 0) == 2:
            return rank
    return None


def eligible_reciprocal_rank(
    ranked_trial_ids: Sequence[str], relevance: Mapping[str, int]
) -> float:
    """Reciprocal rank with only eligible (label 2) trials relevant."""

    rank = first_eligible_rank(ranked_trial_ids, relevance)
    return 0.0 if rank is None else 1.0 / rank


def evaluate_topic(
    ranked_trial_ids: Sequence[str], relevance: Mapping[str, int], *, k: int = 5
) -> TopicMetrics:
    """Evaluate one ordered topic result list."""

    _validate_k(k)
    _validate_relevance_mapping(relevance)
    if len(ranked_trial_ids) != len(set(ranked_trial_ids)):
        raise ValueError("ranked_trial_ids must not contain duplicates")
    rank = first_eligible_rank(ranked_trial_ids, relevance)
    return TopicMetrics(
        eligible_recall=eligible_recall_at_k(ranked_trial_ids, relevance, k=k),
        relevant_or_eligible_recall=relevant_or_eligible_recall_at_k(
            ranked_trial_ids, relevance, k=k
        ),
        ndcg=ndcg_at_k(ranked_trial_ids, relevance, k=k),
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
        first_eligible_rank=rank,
    )


def _validate_relevance_mapping(relevance: Mapping[str, int]) -> None:
    for trial_id, label in relevance.items():
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError("relevance trial IDs must be non-empty strings")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1, 2):
            raise ValueError("relevance labels must be integers in {0, 1, 2}")


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def evaluate_run(
    candidates: Iterable[Candidate],
    judgments: Iterable[RelevanceJudgment],
    *,
    topic_ids: Iterable[str] | None = None,
    k: int = 5,
    unjudged_policy: UnjudgedPolicy = "retain_as_zero_gain",
    precision_relevance_minimum: int | None = None,
    precision_denominator: PrecisionDenominator = "fixed_cutoff",
) -> MetricsDictionary:
    """Evaluate a candidate run and return a JSON-serializable metrics object.

    Topics default to the union of candidate and judgment topic IDs.  Passing
    ``topic_ids`` is recommended for a benchmark because it also scores topics
    with no candidates and no judgments.  Candidate ranks must be unique per
    topic; gaps are allowed and are preserved when calculating first rank/MRR.
    """

    _validate_k(k)
    if unjudged_policy not in ("retain_as_zero_gain", "condense_before_cutoff"):
        raise ValueError(f"unsupported unjudged policy {unjudged_policy!r}")
    if precision_relevance_minimum is not None and precision_relevance_minimum not in (1, 2):
        raise ValueError("precision_relevance_minimum must be 1, 2, or None")
    if precision_denominator not in ("fixed_cutoff", "available_condensed_results"):
        raise ValueError(f"unsupported precision denominator {precision_denominator!r}")
    candidate_rows = list(candidates)
    judgment_rows = list(judgments)

    rankings: dict[str, list[Candidate]] = defaultdict(list)
    candidate_keys: set[tuple[str, str]] = set()
    candidate_ranks: set[tuple[str, int]] = set()
    run_ids: set[str] = set()
    system_ids: set[str] = set()
    for candidate in candidate_rows:
        if not isinstance(candidate, Candidate):
            raise TypeError("candidates must contain Candidate instances")
        trial_key = (candidate.topic_id, candidate.trial_id)
        rank_key = (candidate.topic_id, candidate.rank)
        if trial_key in candidate_keys:
            raise ValueError(
                f"duplicate candidate for topic {candidate.topic_id!r} and "
                f"trial {candidate.trial_id!r}"
            )
        if rank_key in candidate_ranks:
            raise ValueError(
                f"duplicate candidate rank {candidate.rank} for topic {candidate.topic_id!r}"
            )
        candidate_keys.add(trial_key)
        candidate_ranks.add(rank_key)
        run_ids.add(candidate.run_id)
        system_ids.add(candidate.system_id)
        rankings[candidate.topic_id].append(candidate)

    if len(run_ids) > 1:
        raise ValueError("candidates must share one run_id")
    if len(system_ids) > 1:
        raise ValueError("candidates must share one system_id")

    relevance_by_topic: dict[str, dict[str, int]] = defaultdict(dict)
    for judgment in judgment_rows:
        if not isinstance(judgment, RelevanceJudgment):
            raise TypeError("judgments must contain RelevanceJudgment instances")
        topic_relevance = relevance_by_topic[judgment.topic_id]
        if judgment.trial_id in topic_relevance:
            raise ValueError(
                f"duplicate judgment for topic {judgment.topic_id!r} and "
                f"trial {judgment.trial_id!r}"
            )
        topic_relevance[judgment.trial_id] = judgment.label

    if topic_ids is None:
        selected_topics = sorted(set(rankings) | set(relevance_by_topic))
    else:
        selected_topics = list(topic_ids)
        if len(selected_topics) != len(set(selected_topics)):
            raise ValueError("topic_ids must not contain duplicates")
        if any(not isinstance(topic_id, str) or not topic_id for topic_id in selected_topics):
            raise ValueError("topic_ids must contain non-empty strings")
    if not selected_topics:
        raise ValueError("at least one topic is required for evaluation")

    per_topic_metrics: dict[str, TopicMetrics] = {}
    precision_by_topic: dict[str, float] = {}
    for topic_id in selected_topics:
        rows = sorted(rankings.get(topic_id, []), key=lambda row: row.rank)
        relevance = relevance_by_topic.get(topic_id, {})

        if unjudged_policy == "condense_before_cutoff":
            # TrialMatchAI v0.01 removes unjudged rows before assigning metric
            # positions. Explicit label 0 rows remain judged and therefore stay.
            ranked_trial_ids = [row.trial_id for row in rows if row.trial_id in relevance]
        else:
            # Most runs use contiguous ranks. Construct placeholders for gaps so
            # the declared Candidate.rank remains authoritative for cutoff metrics,
            # first eligible rank, and MRR.
            ranked_trial_ids = []
            next_rank = 1
            for row in rows:
                while next_rank < row.rank:
                    ranked_trial_ids.append(f"__taim_unretrieved_rank_{next_rank}__")
                    next_rank += 1
                ranked_trial_ids.append(row.trial_id)
                next_rank += 1
        per_topic_metrics[topic_id] = evaluate_topic(ranked_trial_ids, relevance, k=k)
        if precision_relevance_minimum is not None:
            precision_by_topic[topic_id] = precision_at_k(
                ranked_trial_ids,
                relevance,
                minimum_label=precision_relevance_minimum,
                k=k,
                denominator=precision_denominator,
            )

    metric_key_eligible = f"eligible_recall_at_{k}"
    metric_key_relevant = f"relevant_or_eligible_recall_at_{k}"
    metric_key_ndcg = f"ndcg_at_{k}"
    per_topic = {topic_id: metrics.to_dict(k=k) for topic_id, metrics in per_topic_metrics.items()}
    result: MetricsDictionary = {
        "topic_count": len(selected_topics),
        metric_key_eligible: _mean(
            metrics.eligible_recall for metrics in per_topic_metrics.values()
        ),
        metric_key_relevant: _mean(
            metrics.relevant_or_eligible_recall for metrics in per_topic_metrics.values()
        ),
        metric_key_ndcg: _mean(metrics.ndcg for metrics in per_topic_metrics.values()),
        "mrr": _mean(metrics.reciprocal_rank for metrics in per_topic_metrics.values()),
        "first_eligible_rank": {
            topic_id: metrics.first_eligible_rank for topic_id, metrics in per_topic_metrics.items()
        },
        "per_topic": per_topic,
    }
    if precision_relevance_minimum is not None:
        precision_name = (
            "eligible_precision"
            if precision_relevance_minimum == 2
            else "relevant_or_eligible_precision"
        )
        precision_key = f"{precision_name}_at_{k}"
        result[precision_key] = _mean(precision_by_topic.values())
        for topic_id, value in precision_by_topic.items():
            per_topic[topic_id][precision_key] = value
    return result


def evaluate_profile_run(
    candidates: Iterable[Candidate],
    judgments: Iterable[RelevanceJudgment],
    *,
    topic_ids: Iterable[str],
    cutoffs: Sequence[int],
    unjudged_policy: UnjudgedPolicy,
    precision_relevance_minimum: int | None,
    precision_denominator: PrecisionDenominator,
    abstention_score: float | None = None,
    abstention_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> MetricsDictionary:
    """Evaluate all cutoffs in a profile with mean, median, and per-topic output."""

    selected_cutoffs = tuple(cutoffs)
    if not selected_cutoffs or any(
        isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1
        for cutoff in selected_cutoffs
    ):
        raise ValueError("cutoffs must contain positive integers")
    if tuple(sorted(set(selected_cutoffs))) != selected_cutoffs:
        raise ValueError("cutoffs must be sorted and unique")
    candidate_rows = tuple(candidates)
    judgment_rows = tuple(judgments)
    selected_topics = tuple(topic_ids)
    if precision_relevance_minimum is None:
        if selected_cutoffs != (10,) or precision_denominator != "maximum_graded_gain":
            raise ValueError("graded paper-style evaluation requires cutoff 10 and maximum gain")
        return evaluate_graded_condensed_run(
            candidate_rows,
            judgment_rows,
            topic_ids=selected_topics,
            k=10,
            abstention_score=abstention_score,
            abstention_pairs=abstention_pairs,
        )
    per_topic: dict[str, dict[str, MetricValue]] = {topic_id: {} for topic_id in selected_topics}
    mean: dict[str, float] = {}
    metric_keys: list[str] = []
    for cutoff in selected_cutoffs:
        result = evaluate_run(
            candidate_rows,
            judgment_rows,
            topic_ids=selected_topics,
            k=cutoff,
            unjudged_policy=unjudged_policy,
            precision_relevance_minimum=precision_relevance_minimum,
            precision_denominator=precision_denominator,
        )
        cutoff_keys = (
            f"ndcg_at_{cutoff}",
            (
                f"eligible_precision_at_{cutoff}"
                if precision_relevance_minimum == 2
                else f"relevant_or_eligible_precision_at_{cutoff}"
            ),
        )
        result_per_topic = result["per_topic"]
        if not isinstance(result_per_topic, dict):
            raise TypeError("internal per-topic metrics must be a dictionary")
        for metric_key in cutoff_keys:
            metric_keys.append(metric_key)
            value = result[metric_key]
            if not isinstance(value, (int, float)):
                raise TypeError("internal aggregate metric must be numeric")
            mean[metric_key] = float(value)
            for topic_id in selected_topics:
                topic_metrics = result_per_topic[topic_id]
                if not isinstance(topic_metrics, dict):
                    raise TypeError("internal topic metrics must be a dictionary")
                topic_value = topic_metrics[metric_key]
                if not isinstance(topic_value, (int, float)):
                    raise TypeError("internal topic metric must be numeric")
                per_topic[topic_id][metric_key] = float(topic_value)
    median = {
        metric_key: float(
            statistics.median(
                cast(float, per_topic[topic_id][metric_key]) for topic_id in selected_topics
            )
        )
        for metric_key in metric_keys
    }
    return {
        "cutoffs": list(selected_cutoffs),
        "topic_count": len(selected_topics),
        "aggregate": {"mean": mean, "median": median},
        "per_topic": per_topic,
    }


# A concise alias for callers that already have schema objects.
evaluate = evaluate_run


__all__ = [
    "SchemeTopicMetrics",
    "TopicMetrics",
    "eligible_recall_at_k",
    "eligible_reciprocal_rank",
    "evaluate",
    "evaluate_graded_condensed_run",
    "evaluate_profile_run",
    "evaluate_run",
    "evaluate_scheme_run",
    "evaluate_scheme_topic",
    "evaluate_topic",
    "first_eligible_rank",
    "graded_precision_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "relevant_or_eligible_recall_at_k",
]
