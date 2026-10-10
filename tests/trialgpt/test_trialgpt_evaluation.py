# ruff: noqa: S101
from __future__ import annotations

import pytest

from taim.pipelines.trialgpt.evaluation import evaluate_trialgpt_pipeline_diagnostics
from taim.schemas import Candidate, RelevanceJudgment


def test_trialgpt_pipeline_diagnostics_separates_depth_losses_and_qrel_labels() -> None:
    retrieval = [
        Candidate("run", "system", "1", "eligible-kept", 1, 2.0),
        Candidate("run", "system", "1", "qrel-one", 2, 1.5),
        Candidate("run", "system", "1", "qrel-zero", 3, 1.2),
        Candidate("run", "system", "1", "eligible-cap-loss", 501, 0.5),
    ]
    llm = [
        Candidate("run", "system", "1", "eligible-kept", 1, 2.0),
        Candidate("run", "system", "1", "qrel-one", 2, 1.5),
        Candidate("run", "system", "1", "qrel-zero", 3, 1.2),
    ]
    judgments = [
        RelevanceJudgment("1", "eligible-kept", 2),
        RelevanceJudgment("1", "eligible-cap-loss", 2),
        RelevanceJudgment("1", "eligible-depth-loss", 2),
        RelevanceJudgment("1", "qrel-one", 1),
        RelevanceJudgment("1", "qrel-zero", 0),
    ]
    result = evaluate_trialgpt_pipeline_diagnostics(
        retrieval,
        llm,
        judgments,
        topic_ids=["1"],
        score_thresholds=[1.0],
    )

    aggregate = result["aggregate"]
    assert aggregate["judged_eligible_total"] == 3
    assert aggregate["judged_eligible_in_retrieval"] == 2
    assert aggregate["judged_eligible_in_llm_candidates"] == 1
    assert aggregate["judged_eligible_lost_at_llm_cap"] == 1
    assert aggregate["judged_eligible_missed_at_retrieval_depth"] == 1
    assert aggregate["label_counts_total"] == {"0": 1, "1": 1, "2": 3}
    assert aggregate["label_counts_missed_at_retrieval_depth"] == {"0": 0, "1": 0, "2": 1}
    assert aggregate["label_counts_lost_at_llm_cap"] == {"0": 0, "1": 0, "2": 1}
    threshold = aggregate["score_thresholds"]["1.0"]
    assert threshold["true_positive"] == 1
    assert threshold["false_positive_qrel_0"] == 1
    assert threshold["false_positive_qrel_1"] == 1
    assert threshold["precision_among_judged_resolved"] == pytest.approx(1 / 3)


def test_trialgpt_pipeline_diagnostics_aggregate_typed_counts_across_topics() -> None:
    retrieval = [
        Candidate("run", "system", "1", "eligible", 1, 2.0),
        Candidate("run", "system", "1", "excluded", 2, 1.5),
        Candidate("run", "system", "2", "irrelevant", 1, 1.0),
    ]
    llm = list(retrieval)
    judgments = [
        RelevanceJudgment("1", "eligible", 2),
        RelevanceJudgment("1", "excluded", 1),
        RelevanceJudgment("2", "irrelevant", 0),
        RelevanceJudgment("2", "eligible-depth-loss", 2),
    ]

    result = evaluate_trialgpt_pipeline_diagnostics(
        retrieval,
        llm,
        judgments,
        topic_ids=["1", "2"],
        score_thresholds=[1.0],
    )

    aggregate = result["aggregate"]
    assert aggregate["judged_eligible_total"] == 2
    assert aggregate["judged_eligible_in_retrieval"] == 1
    assert aggregate["judged_eligible_missed_at_retrieval_depth"] == 1
    assert aggregate["judged_llm_candidate_count"] == 3
    assert aggregate["label_counts_total"] == {"0": 1, "1": 1, "2": 2}
    assert aggregate["label_counts_in_retrieval"] == {"0": 1, "1": 1, "2": 1}
    assert aggregate["label_counts_missed_at_retrieval_depth"] == {"0": 0, "1": 0, "2": 1}
    assert aggregate["score_thresholds"]["1.0"] == {
        "true_positive": 1,
        "false_positive_qrel_0": 1,
        "false_positive_qrel_1": 1,
        "resolved_eligible": 1,
        "precision_among_judged_resolved": pytest.approx(1 / 3),
        "recall_among_judged_resolved_eligible": 1.0,
        "end_to_end_recall_of_all_judged_eligible": 0.5,
    }


def test_trialgpt_pipeline_diagnostics_requires_llm_subset_of_retrieval() -> None:
    llm = [Candidate("run", "system", "1", "unexpected", 1, 1.0)]

    with pytest.raises(ValueError, match="LLM candidates must be a subset"):
        evaluate_trialgpt_pipeline_diagnostics([], llm, [], topic_ids=["1"])
