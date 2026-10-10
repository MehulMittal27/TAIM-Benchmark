# ruff: noqa: S101
from __future__ import annotations

import math

import pytest

from taim.evaluation import (
    eligible_recall_at_k,
    eligible_reciprocal_rank,
    evaluate_graded_condensed_run,
    evaluate_profile_run,
    evaluate_run,
    evaluate_topic,
    first_eligible_rank,
    ndcg_at_k,
    precision_at_k,
    relevant_or_eligible_recall_at_k,
)
from taim.schemas import Candidate, RelevanceJudgment


def test_excluded_trial_above_eligible_trial_has_distinct_metric_semantics() -> None:
    # Trial E is condition-relevant but excluded; trial A is eligible.
    ranking = ["E", "A", "N"]
    relevance = {"E": 1, "A": 2, "N": 0}

    metrics = evaluate_topic(ranking, relevance, k=5)

    assert metrics.eligible_recall == 1.0
    assert metrics.relevant_or_eligible_recall == 1.0
    assert metrics.reciprocal_rank == 0.5
    assert metrics.first_eligible_rank == 2
    expected_dcg = 1.0 + (2.0 / math.log2(3))
    expected_ideal_dcg = 2.0 + (1.0 / math.log2(3))
    assert metrics.ndcg == pytest.approx(expected_dcg / expected_ideal_dcg)
    assert metrics.ndcg < 1.0


def test_missing_eligible_retrieval_is_zero_and_first_rank_is_null() -> None:
    ranking = ["excluded", "unjudged"]
    relevance = {"eligible": 2, "excluded": 1}

    metrics = evaluate_topic(ranking, relevance, k=5)

    assert metrics.eligible_recall == 0.0
    assert metrics.relevant_or_eligible_recall == 0.5
    assert metrics.reciprocal_rank == 0.0
    assert metrics.first_eligible_rank is None
    assert metrics.ndcg > 0.0


def test_topic_with_no_eligible_judgments_uses_explicit_zero() -> None:
    ranking = ["condition-match"]
    relevance = {"condition-match": 1}

    metrics = evaluate_topic(ranking, relevance, k=5)

    assert metrics.eligible_recall == 0.0
    assert metrics.relevant_or_eligible_recall == 1.0
    assert metrics.reciprocal_rank == 0.0
    assert metrics.first_eligible_rank is None
    assert metrics.ndcg == 1.0


def test_recall_uses_all_positive_qrels_as_the_denominator() -> None:
    relevance = {
        "eligible-retrieved": 2,
        "eligible-missed": 2,
        "excluded-retrieved": 1,
        "excluded-missed": 1,
    }
    ranking = ["excluded-retrieved", "eligible-retrieved"]

    assert eligible_recall_at_k(ranking, relevance, k=5) == 0.5
    assert relevant_or_eligible_recall_at_k(ranking, relevance, k=5) == 0.5


def test_ndcg_uses_trec_graded_gain_and_treats_unjudged_as_zero() -> None:
    ranking = ["label-1", "unjudged", "label-2"]
    relevance = {"label-2": 2, "label-1": 1}
    dcg = 1.0 + (2.0 / math.log2(4))
    ideal_dcg = 2.0 + (1.0 / math.log2(3))

    assert ndcg_at_k(ranking, relevance, k=3) == pytest.approx(dcg / ideal_dcg)


def test_mrr_and_first_eligible_rank_are_not_cut_off_at_five() -> None:
    ranking = [f"N{rank}" for rank in range(1, 6)] + ["eligible"]
    relevance = {"eligible": 2}

    assert eligible_recall_at_k(ranking, relevance, k=5) == 0.0
    assert first_eligible_rank(ranking, relevance) == 6
    assert eligible_reciprocal_rank(ranking, relevance) == pytest.approx(1 / 6)


def test_evaluate_run_macro_averages_all_selected_topics() -> None:
    candidates = [
        Candidate("run", "synthetic-system", "T1", "excluded", 1, 3.0),
        Candidate("run", "synthetic-system", "T1", "eligible", 2, 2.0),
        Candidate("run", "synthetic-system", "T2", "irrelevant", 1, 1.0),
        Candidate("run", "synthetic-system", "T3", "condition", 1, 1.0),
    ]
    judgments = [
        RelevanceJudgment("T1", "excluded", 1),
        RelevanceJudgment("T1", "eligible", 2),
        RelevanceJudgment("T2", "not-retrieved", 2),
        RelevanceJudgment("T3", "condition", 1),
    ]
    t1_ndcg = (1.0 + (2.0 / math.log2(3))) / (2.0 + (1.0 / math.log2(3)))

    result = evaluate_run(candidates, judgments, topic_ids=["T1", "T2", "T3"])

    assert result["topic_count"] == 3
    assert result["eligible_recall_at_5"] == pytest.approx(1 / 3)
    assert result["relevant_or_eligible_recall_at_5"] == pytest.approx(2 / 3)
    assert result["ndcg_at_5"] == pytest.approx((t1_ndcg + 1.0) / 3)
    assert result["mrr"] == pytest.approx(1 / 6)
    assert result["first_eligible_rank"] == {"T1": 2, "T2": None, "T3": None}


def test_evaluate_run_preserves_declared_rank_gaps() -> None:
    candidates = [Candidate("run", "synthetic-system", "T1", "eligible", 6, 1.0)]
    judgments = [RelevanceJudgment("T1", "eligible", 2)]

    result = evaluate_run(candidates, judgments, topic_ids=["T1"], k=5)

    assert result["eligible_recall_at_5"] == 0.0
    assert result["ndcg_at_5"] == 0.0
    assert result["mrr"] == pytest.approx(1 / 6)
    assert result["first_eligible_rank"] == {"T1": 6}


def test_evaluate_run_includes_explicit_topic_with_no_rows() -> None:
    result = evaluate_run([], [], topic_ids=["empty-topic"])

    assert result["eligible_recall_at_5"] == 0.0
    assert result["relevant_or_eligible_recall_at_5"] == 0.0
    assert result["ndcg_at_5"] == 0.0
    assert result["mrr"] == 0.0
    assert result["first_eligible_rank"] == {"empty-topic": None}


def test_evaluate_run_rejects_duplicate_trials_ranks_and_qrels() -> None:
    candidate = Candidate("run", "synthetic-system", "T1", "A", 1, 1.0)
    other_at_same_rank = Candidate("run", "synthetic-system", "T1", "B", 1, 0.5)
    judgment = RelevanceJudgment("T1", "A", 2)

    with pytest.raises(ValueError, match="duplicate candidate for topic"):
        evaluate_run([candidate, candidate], [judgment])
    with pytest.raises(ValueError, match="duplicate candidate rank"):
        evaluate_run([candidate, other_at_same_rank], [judgment])
    with pytest.raises(ValueError, match="duplicate judgment"):
        evaluate_run([candidate], [judgment, judgment])


def test_evaluate_run_rejects_mixed_runs_and_systems() -> None:
    first = Candidate("run-1", "synthetic-system", "T1", "A", 1, 1.0)
    different_run = Candidate("run-2", "synthetic-system", "T2", "B", 1, 1.0)
    different_system = Candidate("run-1", "synthetic-vector", "T2", "B", 1, 1.0)

    with pytest.raises(ValueError, match="one run_id"):
        evaluate_run([first, different_run], [])
    with pytest.raises(ValueError, match="one system_id"):
        evaluate_run([first, different_system], [])


def test_unjudged_condensation_changes_cutoff_but_explicit_zero_does_not() -> None:
    candidates = [
        Candidate("run", "system", "1", "unjudged", 1, 3.0),
        Candidate("run", "system", "1", "judged-zero", 2, 2.0),
        Candidate("run", "system", "1", "eligible", 3, 1.0),
    ]
    judgments = [
        RelevanceJudgment("1", "judged-zero", 0),
        RelevanceJudgment("1", "eligible", 2),
    ]

    official = evaluate_run(candidates, judgments, topic_ids=["1"], k=2)
    condensed = evaluate_run(
        candidates,
        judgments,
        topic_ids=["1"],
        k=2,
        unjudged_policy="condense_before_cutoff",
    )

    assert official["eligible_recall_at_2"] == 0.0
    assert condensed["eligible_recall_at_2"] == 1.0
    assert condensed["first_eligible_rank"] == {"1": 2}


def test_precision_has_explicit_relevance_and_denominator_policies() -> None:
    ranking = ["eligible", "excluded"]
    relevance = {"eligible": 2, "excluded": 1}

    assert precision_at_k(ranking, relevance, minimum_label=1, k=5) == 0.4
    assert precision_at_k(ranking, relevance, minimum_label=2, k=5) == 0.2
    assert (
        precision_at_k(
            ranking,
            relevance,
            minimum_label=2,
            k=5,
            denominator="available_condensed_results",
        )
        == 0.5
    )


def test_profile_evaluation_emits_all_cutoffs_means_medians_and_topics() -> None:
    candidates = [
        Candidate("run", "system", "1", "unjudged", 1, 2.0),
        Candidate("run", "system", "1", "eligible", 2, 1.0),
        Candidate("run", "system", "2", "zero", 1, 1.0),
    ]
    judgments = [
        RelevanceJudgment("1", "eligible", 2),
        RelevanceJudgment("2", "zero", 0),
        RelevanceJudgment("2", "missing", 2),
    ]

    result = evaluate_profile_run(
        candidates,
        judgments,
        topic_ids=["1", "2"],
        cutoffs=[1, 2],
        unjudged_policy="condense_before_cutoff",
        precision_relevance_minimum=2,
        precision_denominator="available_condensed_results",
    )

    assert result["cutoffs"] == [1, 2]
    assert result["topic_count"] == 2
    aggregate = result["aggregate"]
    assert aggregate["mean"]["eligible_precision_at_1"] == 0.5
    assert aggregate["median"]["ndcg_at_2"] == 0.5
    assert set(result["per_topic"]) == {"1", "2"}


def test_trialgpt_paper_style_condenses_unjudged_and_uses_graded_precision() -> None:
    candidates = [
        Candidate("run", "system", "1", "unjudged", 1, 99.0),
        Candidate("run", "system", "1", "eligible", 2, 0.9),
        Candidate("run", "system", "1", "excluded", 3, 0.8),
        Candidate("run", "system", "1", "negative", 4, 0.1),
    ]
    judgments = [
        RelevanceJudgment("1", "eligible", 2),
        RelevanceJudgment("1", "excluded", 1),
        RelevanceJudgment("1", "negative", 0),
    ]

    result = evaluate_graded_condensed_run(candidates, judgments, topic_ids=["1"], k=10)

    topic = result["per_topic"]["1"]
    assert topic["judged_trials_in_candidate_ranking"] == 3
    assert topic["retained_after_judged_only_condensation"] == 3
    assert topic["label_counts"] == {"0": 1, "1": 1, "2": 1}
    assert topic["graded_precision_at_10"] == pytest.approx(3 / 20)
    assert topic["ndcg_at_10"] == 1.0
    assert topic["auroc"] == 1.0
    assert topic["raw_top_k"]["label_counts"] == {
        "0": 1,
        "1": 1,
        "2": 1,
        "unjudged": 1,
    }
    assert topic["judged_condensed_top_k"]["label_counts"] == {"0": 1, "1": 1, "2": 1}
    assert result["aggregate"]["mean"]["raw_graded_precision_at_10"] == pytest.approx(3 / 20)


def test_trialgpt_paper_style_withholds_auroc_for_judged_abstention() -> None:
    candidates = [
        Candidate("run", "system", "1", "eligible", 1, 1.0),
        Candidate("run", "system", "1", "negative", 2, -3.0),
    ]
    judgments = [
        RelevanceJudgment("1", "eligible", 2),
        RelevanceJudgment("1", "negative", 0),
    ]
    result = evaluate_graded_condensed_run(
        candidates,
        judgments,
        topic_ids=["1"],
        abstention_score=-3.0,
        abstention_pairs=frozenset({("1", "negative")}),
    )

    topic = result["per_topic"]["1"]
    assert topic["candidate_score_coverage_complete"] is False
    assert topic["semantic_abstention_count_in_judged_candidates"] == 1
    assert topic["auroc"] is None
    assert topic["resolved_only_auroc_sensitivity"] is None
    assert topic["semantic_abstention_label_counts"] == {
        "0": 1,
        "1": 0,
        "2": 0,
        "unjudged": 0,
    }
    assert topic["candidate_score_coverage"]["judged_resolved_candidate_fraction"] == 0.5
    assert result["aggregate"]["score_coverage"]["judged_resolved_candidate_fraction"] == 0.5
