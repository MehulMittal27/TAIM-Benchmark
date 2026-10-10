# ruff: noqa: S101
from __future__ import annotations

import math

import pytest

from taim.contracts import EVALUATION_PACKAGE_VERSION, content_sha256
from taim.evaluation import evaluate_run, evaluate_scheme_run
from taim.evaluation_package import EvaluationPackage
from taim.judgments import (
    SIGIR_CT_2016_JUDGMENT_SCHEME,
    TREC_CT_JUDGMENT_SCHEME,
    JudgmentScheme,
)
from taim.release_profiles import load_profile
from taim.schemas import Candidate, RelevanceJudgment, SchemaValidationError

_SNAPSHOT_ID = "sha256:" + "a" * 64


def _custom_scheme() -> JudgmentScheme:
    return JudgmentScheme(
        scheme_id="binary-clinical-match",
        scheme_version="1.0",
        labels={0: "not_a_match", 3: "match"},
        gains={0: 0, 3: 1},
        relevance_sets={"match": frozenset({3})},
        reciprocal_rank_relevance_set="match",
    )


def test_default_trec_scheme_preserves_legacy_package_identity_and_bytes() -> None:
    judgments = (RelevanceJudgment("topic", "trial", 2),)
    package = EvaluationPackage(
        benchmark_lineage="trec-ct-2021",
        task_id="patient-to-trial",
        snapshot_id=_SNAPSHOT_ID,
        judgments=judgments,
        provenance={"source": "fixture"},
    )
    legacy_identity_payload = {
        "schema_version": EVALUATION_PACKAGE_VERSION,
        "benchmark_lineage": "trec-ct-2021",
        "task_id": "patient-to-trial",
        "snapshot_id": _SNAPSHOT_ID,
        "judgments": [judgment.to_dict() for judgment in judgments],
        "provenance": {"source": "fixture"},
    }

    assert package.judgment_scheme is TREC_CT_JUDGMENT_SCHEME
    assert "judgment_scheme" not in package.to_dict()
    assert "judgment_scheme" not in package.manifest_dict()
    assert package.evaluation_package_id == content_sha256(legacy_identity_payload)
    assert EvaluationPackage.from_dict(package.to_dict()) == package


def test_custom_scheme_is_identity_bound_and_round_trips() -> None:
    scheme = _custom_scheme()
    package = EvaluationPackage(
        benchmark_lineage="custom-benchmark",
        task_id="patient-to-trial",
        snapshot_id=_SNAPSHOT_ID,
        judgments=(RelevanceJudgment("topic", "trial", 3),),
        provenance={"source": "fixture"},
        judgment_scheme=scheme,
    )

    assert package.to_dict()["judgment_scheme"] == scheme.to_dict()
    assert EvaluationPackage.from_dict(package.to_dict()) == package

    different_scheme = JudgmentScheme(
        scheme_id=scheme.scheme_id,
        scheme_version="1.1",
        labels=scheme.labels,
        gains={0: 0, 3: 2},
        relevance_sets=scheme.relevance_sets,
        reciprocal_rank_relevance_set=scheme.reciprocal_rank_relevance_set,
    )
    changed = EvaluationPackage(
        benchmark_lineage=package.benchmark_lineage,
        task_id=package.task_id,
        snapshot_id=package.snapshot_id,
        judgments=package.judgments,
        provenance=package.provenance,
        judgment_scheme=different_scheme,
    )
    assert changed.evaluation_package_id != package.evaluation_package_id


def test_scheme_rejects_undeclared_labels_and_is_recursively_immutable() -> None:
    scheme = _custom_scheme()

    with pytest.raises(SchemaValidationError, match="Judgment label"):
        scheme.validate_label(2)
    with pytest.raises(TypeError):
        scheme.gains[3] = 5  # type: ignore[index]


@pytest.mark.parametrize("label", [True, False])
def test_scheme_rejects_boolean_relevance_labels_loaded_from_json(label: bool) -> None:
    payload = TREC_CT_JUDGMENT_SCHEME.to_dict()
    payload["relevance_sets"] = {
        "eligible": [label],
        "relevant_or_eligible": [1, 2],
    }

    with pytest.raises(SchemaValidationError, match="non-boolean integers"):
        JudgmentScheme.from_dict(payload)


def test_generic_evaluator_uses_declared_semantics_without_changing_trec_evaluator() -> None:
    scheme = _custom_scheme()
    candidates = [
        Candidate("run", "system", "topic", "negative", 1, 2.0),
        Candidate("run", "system", "topic", "positive", 2, 1.0),
    ]
    judgments = [
        RelevanceJudgment("topic", "negative", 0),
        RelevanceJudgment("topic", "positive", 3),
    ]

    result = evaluate_scheme_run(candidates, judgments, scheme=scheme, k=2)

    assert result["aggregate"] == {
        "recall": {"match_at_2": 1.0},
        "ndcg_at_2": pytest.approx(1 / math.log2(3)),
        "mrr": 0.5,
    }
    per_topic = result["per_topic"]
    assert isinstance(per_topic, dict)
    assert per_topic["topic"]["first_relevant_rank"] == 2
    with pytest.raises(ValueError, match="relevance labels"):
        evaluate_run(candidates, judgments, topic_ids=["topic"], k=2)


@pytest.mark.parametrize(
    ("profile_id", "scheme", "positive_label", "relevance_minimum"),
    [
        ("official-full", TREC_CT_JUDGMENT_SCHEME, 1, 1),
        ("confirmation-full", TREC_CT_JUDGMENT_SCHEME, 2, 2),
        ("description", SIGIR_CT_2016_JUDGMENT_SCHEME, 1, 1),
        ("summary", SIGIR_CT_2016_JUDGMENT_SCHEME, 1, 1),
    ],
)
def test_every_real_release_profile_executes_its_declared_precision(
    profile_id: str,
    scheme: JudgmentScheme,
    positive_label: int,
    relevance_minimum: int,
) -> None:
    profile = load_profile(profile_id)
    candidates = [
        Candidate("run", "system", "topic", "positive", 1, 2.0),
        Candidate("run", "system", "topic", "negative", 2, 1.0),
    ]
    judgments = [
        RelevanceJudgment("topic", "negative", 0),
        RelevanceJudgment("topic", "positive", positive_label),
    ]

    result = evaluate_scheme_run(
        candidates,
        judgments,
        scheme=scheme,
        k=5,
        precision_metric=profile.precision_metric,
        precision_relevance_minimum=profile.precision_relevance_minimum,
        precision_denominator=profile.precision_denominator,
    )

    assert result["aggregate"][f"{profile.precision_metric}_at_5"] == pytest.approx(0.2)
    assert result["precision_policy"] == {
        "metric": profile.precision_metric,
        "relevance_set": profile.precision_metric.removesuffix("_precision"),
        "relevance_minimum": relevance_minimum,
        "denominator": "fixed_cutoff",
    }
