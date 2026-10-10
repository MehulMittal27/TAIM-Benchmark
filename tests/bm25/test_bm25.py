# ruff: noqa: S101
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from taim.baselines.bm25 import (
    BM25Retriever,
    StreamingBM25Index,
    query_vocabulary,
    tokenize,
)
from taim.snapshot import BenchmarkTopic, Trial
from tests.benchmark_helpers import trial

FIXTURE_DIRECTORY = Path(__file__).parents[2] / "src/taim/fixtures/trec-mini"


def _read_jsonl(name: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (FIXTURE_DIRECTORY / name).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _fixture_topics() -> list[BenchmarkTopic]:
    return [BenchmarkTopic.from_dict(row) for row in _read_jsonl("topics.jsonl")]


def _fixture_trials() -> list[Trial]:
    return [Trial.from_dict(row) for row in _read_jsonl("trials.jsonl")]


def _reference_scores(
    documents: Sequence[Trial],
    query: str,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """Small Counter-per-document reference kept independent of the production index."""

    term_frequencies = [Counter(tokenize(document.canonical_text)) for document in documents]
    document_lengths = [sum(frequencies.values()) for frequencies in term_frequencies]
    average_document_length = (
        sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
    )
    document_frequency: Counter[str] = Counter()
    for frequencies in term_frequencies:
        document_frequency.update(frequencies.keys())
    query_term_frequency = Counter(tokenize(query))

    scores: list[float] = []
    for frequencies, document_length in zip(
        term_frequencies,
        document_lengths,
        strict=True,
    ):
        length_ratio = document_length / average_document_length if average_document_length else 0.0
        score = 0.0
        for term, query_frequency in query_term_frequency.items():
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (len(documents) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (1.0 - b + b * length_ratio)
            score += (
                query_frequency * inverse_document_frequency * frequency * (k1 + 1.0) / denominator
            )
        scores.append(score)
    return scores


def _reference_rank(
    documents: Sequence[Trial],
    query: str,
    *,
    top_k: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    scores = _reference_scores(documents, query, k1=k1, b=b)
    ranked = sorted(
        zip(documents, scores, strict=True),
        key=lambda item: (-item[1], item[0].trial_id),
    )
    return [(document.trial_id, score) for document, score in ranked[:top_k]]


def test_tokenize_is_case_insensitive_and_deterministic() -> None:
    assert tokenize("HER2-positive, HER2+ cancer") == (
        "her2",
        "positive",
        "her2",
        "cancer",
    )


def test_bm25_preserves_scores_and_one_based_ranks() -> None:
    retriever = BM25Retriever(_fixture_trials())

    candidates = retriever.retrieve(_fixture_topics()[0], run_id="fixture-test", top_k=5)

    assert [candidate.rank for candidate in candidates] == [1, 2, 3, 4, 5]
    assert all(candidate.run_id == "fixture-test" for candidate in candidates)
    assert all(candidate.system_id == "bm25" for candidate in candidates)
    assert candidates[0].score > candidates[1].score


@pytest.mark.parametrize(
    "query",
    [
        "HER2 breast cancer HER2",
        "type 2 diabetes chronic kidney disease",
        "relapsed lymphoma CAR T-cell",
        "terms absent from every trial",
        "",
    ],
)
def test_inverted_index_scores_and_ranks_match_local_reference(query: str) -> None:
    documents = _fixture_trials()
    retriever = BM25Retriever(documents)
    expected_scores = _reference_scores(documents, query)

    assert [retriever.score(query, index) for index in range(len(documents))] == expected_scores
    assert [
        (document.trial_id, score) for document, score in retriever.rank(query, top_k=8)
    ] == _reference_rank(documents, query, top_k=8)


@pytest.mark.parametrize(("k1", "b"), [(0.5, 0.0), (2.0, 1.0)])
def test_custom_bm25_parameters_match_local_reference(k1: float, b: float) -> None:
    documents = _fixture_trials()
    query = "HER2 HER2 cancer treatment"
    retriever = BM25Retriever(documents, k1=k1, b=b)

    assert [retriever.score(query, index) for index in range(len(documents))] == _reference_scores(
        documents,
        query,
        k1=k1,
        b=b,
    )
    assert [
        (document.trial_id, score) for document, score in retriever.rank(query, top_k=5)
    ] == _reference_rank(documents, query, top_k=5, k1=k1, b=b)


def test_condition_match_excluded_trial_can_rank_above_eligible_trial() -> None:
    retriever = BM25Retriever(_fixture_trials())

    candidates = retriever.retrieve(_fixture_topics()[0], run_id="fixture-test", top_k=8)
    rank_by_trial = {candidate.trial_id: candidate.rank for candidate in candidates}

    assert rank_by_trial["NCT-BREAST-EXCLUDED"] < rank_by_trial["NCT-BREAST-ELIGIBLE"]


def test_exact_score_ties_are_broken_by_trial_id() -> None:
    documents = [
        trial("NCT-Z", "unmatched text"),
        trial("NCT-A", "unmatched text"),
    ]
    retriever = BM25Retriever(documents)

    ranked = retriever.rank("different query", top_k=2)

    assert [(document.trial_id, score) for document, score in ranked] == [
        ("NCT-A", 0.0),
        ("NCT-Z", 0.0),
    ]


def test_zero_score_documents_fill_remaining_ranks_by_trial_id() -> None:
    documents = [
        trial("NCT-Z", "matched term"),
        trial("NCT-B", "unrelated text"),
        trial("NCT-A", "other words"),
    ]

    ranked = BM25Retriever(documents).rank("matched", top_k=3)

    assert [(document.trial_id, score) for document, score in ranked] == [
        ("NCT-Z", _reference_scores(documents, "matched")[0]),
        ("NCT-A", 0.0),
        ("NCT-B", 0.0),
    ]


def test_input_order_does_not_change_ranking() -> None:
    documents = _fixture_trials()
    patient_text = _fixture_topics()[1].canonical_text

    forward = BM25Retriever(documents).rank(patient_text, top_k=8)
    reverse = BM25Retriever(list(reversed(documents))).rank(patient_text, top_k=8)

    assert [(document.trial_id, score) for document, score in forward] == [
        (document.trial_id, score) for document, score in reverse
    ]


def test_rank_tokenizes_the_query_once(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = BM25Retriever(_fixture_trials())
    calls = 0

    def counting_tokenize(text: str) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        return tokenize(text)

    monkeypatch.setattr("taim.baselines.bm25.tokenize", counting_tokenize)

    retriever.rank("repeated query terms query", top_k=8)

    assert calls == 1


def test_corpus_statistics_are_complete_and_json_safe() -> None:
    documents = _fixture_trials()
    all_tokens = [token for document in documents for token in tokenize(document.canonical_text)]

    statistics = BM25Retriever(documents).corpus_statistics

    assert statistics == {
        "document_count": len(documents),
        "total_terms": len(all_tokens),
        "average_document_length": len(all_tokens) / len(documents),
        "vocabulary_size": len(set(all_tokens)),
        "index_representation": "inverted_postings_uint32",
    }
    assert json.loads(json.dumps(statistics, allow_nan=False)) == statistics


def test_empty_corpus_statistics_and_ranking() -> None:
    retriever = BM25Retriever([])

    assert retriever.corpus_statistics == {
        "document_count": 0,
        "total_terms": 0,
        "average_document_length": 0.0,
        "vocabulary_size": 0,
        "index_representation": "inverted_postings_uint32",
    }
    assert retriever.rank("query", top_k=5) == []
    with pytest.raises(IndexError, match="outside the corpus"):
        retriever.score("query", 0)


def test_retriever_rejects_invalid_configuration_and_duplicate_ids() -> None:
    document = trial("NCT-1", "trial")

    for k1 in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="k1"):
            BM25Retriever([document], k1=k1)
    for b in (1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="b"):
            BM25Retriever([document], b=b)
    with pytest.raises(ValueError, match="unique"):
        BM25Retriever([document, document])


def test_bm25_ignores_unknown_topic_and_trial_extensions() -> None:
    plain_topics = _fixture_topics()
    plain_trials = _fixture_trials()
    extended_topics = [
        replace(
            topic,
            extensions={"demo.note": f"topic-{topic.topic_id}"},
        )
        for topic in plain_topics
    ]
    extended_trials = [
        replace(
            trial,
            registry_extensions={"demo.phase": "ignored"},
        )
        for trial in plain_trials
    ]

    plain = BM25Retriever(plain_trials).retrieve_many(plain_topics, run_id="plain", top_k=5)
    extended = BM25Retriever(extended_trials).retrieve_many(
        extended_topics, run_id="extended", top_k=5
    )

    assert [(row.topic_id, row.trial_id, row.rank, row.score) for row in extended] == [
        (row.topic_id, row.trial_id, row.rank, row.score) for row in plain
    ]


def test_retrieve_many_emits_top_k_per_topic() -> None:
    topics = _fixture_topics()
    candidates = BM25Retriever(_fixture_trials()).retrieve_many(
        topics, run_id="fixture-test", top_k=3
    )

    assert len(candidates) == len(topics) * 3
    assert [candidate.topic_id for candidate in candidates[::3]] == [
        topic.topic_id for topic in topics
    ]


# --- StreamingBM25Index: the same ranking without holding the corpus ----------


def _streaming_documents() -> list[tuple[str, str]]:
    return [(row.trial_id, row.canonical_text) for row in _fixture_trials()]


def test_streamed_index_ranks_identically_to_the_in_memory_retriever() -> None:
    """The reason the folded arm can be run on a laptop, pinned rather than claimed."""

    documents = _streaming_documents()
    queries = {topic.topic_id: topic.canonical_text for topic in _fixture_topics()}
    full = BM25Retriever(_fixture_trials())
    streamed = StreamingBM25Index(
        iter(documents), query_vocabulary=query_vocabulary(queries.values())
    )

    for depth in (1, 3, len(documents)):
        for query in queries.values():
            expected = [(row.trial_id, score) for row, score in full.rank(query, top_k=depth)]
            assert streamed.rank(query, top_k=depth) == expected


def test_streamed_index_normalises_by_the_whole_corpus_not_the_query_terms() -> None:
    """Length normalisation has to see every token, or the scores are not BM25's."""

    documents = _streaming_documents()
    streamed = StreamingBM25Index(iter(documents), query_vocabulary={"aspirin"})

    assert (
        streamed.corpus_statistics["average_document_length"]
        == BM25Retriever(_fixture_trials()).corpus_statistics["average_document_length"]
    )
    assert streamed.corpus_statistics["document_count"] == len(documents)


def test_streamed_index_refuses_a_query_it_was_not_built_for() -> None:
    """Silently scoring zero for an unindexed term would look like a real miss."""

    streamed = StreamingBM25Index(iter(_streaming_documents()), query_vocabulary={"cancer"})

    with pytest.raises(ValueError, match="outside the declared vocabulary"):
        streamed.rank("cancer aspirin", top_k=5)


def test_streamed_retrieve_many_emits_candidates_in_numeric_topic_order() -> None:
    queries = {"10": "cancer", "2": "cancer"}
    streamed = StreamingBM25Index(
        iter(_streaming_documents()), query_vocabulary=query_vocabulary(queries.values())
    )

    candidates = streamed.retrieve_many(queries, run_id="streamed", top_k=2)

    assert [row.topic_id for row in candidates] == ["2", "2", "10", "10"]
    assert {row.system_id for row in candidates} == {"bm25"}
    assert [row.rank for row in candidates] == [1, 2, 1, 2]
