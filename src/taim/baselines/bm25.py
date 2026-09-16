"""A small, deterministic BM25 implementation with no service dependencies."""

from __future__ import annotations

import math
import re
from array import array
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from heapq import nsmallest
from typing import Any, Generic, Protocol, TypeVar, cast

from taim.schemas import Candidate
from taim.snapshot import BenchmarkTopic

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_POSTING_ARRAY_TYPE = "I"
_INDEX_REPRESENTATION = "inverted_postings_uint32"


class _TrialBM25Document(Protocol):
    @property
    def trial_id(self) -> str: ...

    @property
    def canonical_text(self) -> str: ...


_BM25DocumentT = TypeVar("_BM25DocumentT")


def _trial_document_id(document: _TrialBM25Document) -> str:
    return document.trial_id


def _canonical_document_text(document: _TrialBM25Document) -> str:
    return document.canonical_text


def bm25_ranking_configuration(
    *,
    k1: object,
    b: object,
    requested_top_k: int,
    document_count: int,
) -> dict[str, Any]:
    """Describe the shared ranking policy implemented by :class:`BM25Retriever`."""

    return {
        "parameters": {"b": b, "k1": k1},
        "requested_top_k": requested_top_k,
        "effective_top_k": min(requested_top_k, document_count),
        "tokenizer": {
            "normalization": "str.casefold",
            "pattern": "[a-z0-9]+",
            "implementation": "taim.baselines.bm25.tokenize",
        },
        "score_precision": "IEEE-754 binary64 serialized with Python JSON number semantics",
        "score_tie_breaking": "score descending, trial_id ascending",
        "retrieval_formula": {
            "id": "taim-okapi-bm25-v1",
            "implementation": "taim.baselines.bm25.BM25Retriever",
            "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))",
            "length_normalization": "k1 * (1 - b + b * dl / avgdl)",
            "term_contribution": "qtf * idf * tf * (k1 + 1) / (tf + length_normalization)",
            "score": "sum(term_contribution)",
        },
    }


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize text deterministically for the fixture BM25 baseline."""

    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


class BM25Retriever(Generic[_BM25DocumentT]):
    """In-memory Okapi BM25 over identified canonical-text documents.

    The implementation deliberately remains small and inspectable. All corpus
    documents participate in the ranking, including documents with a zero
    score. Exact score ties are ordered by the configured document ID so results do not depend
    on corpus input order.
    """

    def __init__(
        self,
        documents: Sequence[_BM25DocumentT],
        *,
        document_id: Callable[[_BM25DocumentT], str] | None = None,
        document_text: Callable[[_BM25DocumentT], str] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and greater than zero")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between zero and one")

        identity = document_id or cast(
            Callable[[_BM25DocumentT], str],
            _trial_document_id,
        )
        text = document_text or cast(
            Callable[[_BM25DocumentT], str],
            _canonical_document_text,
        )
        document_ids = [identity(document) for document in documents]
        if any(not isinstance(item, str) or not item for item in document_ids):
            raise ValueError("document_id values must be non-empty strings")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be unique")

        self.documents = tuple(documents)
        self._document_ids = tuple(document_ids)
        self._document_text = text
        self.k1 = k1
        self.b = b
        self._document_lengths = array(_POSTING_ARRAY_TYPE)
        postings: dict[str, tuple[array, array]] = {}
        for document_index, document in enumerate(self.documents):
            term_frequencies = Counter(tokenize(self._document_text(document)))
            self._document_lengths.append(sum(term_frequencies.values()))
            for term, frequency in term_frequencies.items():
                posting = postings.get(term)
                if posting is None:
                    posting = (
                        array(_POSTING_ARRAY_TYPE),
                        array(_POSTING_ARRAY_TYPE),
                    )
                    postings[term] = posting
                document_indexes, frequencies = posting
                document_indexes.append(document_index)
                frequencies.append(frequency)
        self._postings = postings

        total_terms = sum(self._document_lengths)
        self._average_document_length = (
            total_terms / len(self._document_lengths) if self._document_lengths else 0.0
        )
        self._length_normalizations = array(
            "d",
            (
                self.k1
                * (1.0 - self.b + self.b * (document_length / self._average_document_length))
                if self._average_document_length
                else self.k1 * (1.0 - self.b)
                for document_length in self._document_lengths
            ),
        )
        self._document_indexes_by_id = array(
            _POSTING_ARRAY_TYPE,
            sorted(
                range(len(self.documents)),
                key=lambda index: self._document_ids[index],
            ),
        )
        self._corpus_statistics: dict[str, int | float | str] = {
            "document_count": len(self.documents),
            "total_terms": total_terms,
            "average_document_length": self._average_document_length,
            "vocabulary_size": len(self._postings),
            "index_representation": _INDEX_REPRESENTATION,
        }

    @property
    def corpus_statistics(self) -> dict[str, int | float | str]:
        """Return JSON-safe statistics describing the immutable corpus index."""

        return dict(self._corpus_statistics)

    def _inverse_document_frequency(self, document_frequency: int) -> float:
        corpus_size = len(self.documents)
        return math.log(1.0 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5))

    def _term_contribution(
        self,
        *,
        document_index: int,
        frequency: int,
        query_frequency: int,
        inverse_document_frequency: float,
    ) -> float:
        denominator = frequency + self._length_normalizations[document_index]
        return (
            query_frequency * inverse_document_frequency * frequency * (self.k1 + 1.0) / denominator
        )

    def score(self, query: str, document_index: int) -> float:
        """Return the BM25 score for one indexed document."""

        if not 0 <= document_index < len(self.documents):
            raise IndexError("document_index is outside the corpus")
        query_term_frequency = Counter(tokenize(query))
        score = 0.0
        for term, query_frequency in query_term_frequency.items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            document_indexes, frequencies = posting
            posting_index = bisect_left(document_indexes, document_index)
            if (
                posting_index == len(document_indexes)
                or document_indexes[posting_index] != document_index
            ):
                continue
            inverse_document_frequency = self._inverse_document_frequency(len(document_indexes))
            score += self._term_contribution(
                document_index=document_index,
                frequency=frequencies[posting_index],
                query_frequency=query_frequency,
                inverse_document_frequency=inverse_document_frequency,
            )
        return score

    def rank_positions(
        self,
        query_text: str,
        *,
        top_k: int = 5,
    ) -> tuple[list[int], dict[int, float]]:
        """Rank document positions and return every score the query touched.

        The first element is the ranking :meth:`rank` reports, as corpus
        positions rather than documents.  The second is the exact score of
        every document at least one query term reaches; a document absent from
        it scored exactly zero, because BM25 sums per-term contributions and a
        document outside every query term's postings has no contribution to
        sum.  A hybrid blend needs that second value for candidates its own
        leg surfaced, which a top-k ranking alone cannot supply.
        """

        if top_k < 0:
            raise ValueError("top_k must not be negative")
        if not self.documents:
            return [], {}

        query_term_frequency = Counter(tokenize(query_text))
        if top_k == 0:
            return [], {}

        scores: dict[int, float] = {}
        for term, query_frequency in query_term_frequency.items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            document_indexes, frequencies = posting
            inverse_document_frequency = self._inverse_document_frequency(len(document_indexes))
            for document_index, frequency in zip(document_indexes, frequencies, strict=True):
                contribution = self._term_contribution(
                    document_index=document_index,
                    frequency=frequency,
                    query_frequency=query_frequency,
                    inverse_document_frequency=inverse_document_frequency,
                )
                scores[document_index] = scores.get(document_index, 0.0) + contribution

        limit = min(top_k, len(self.documents))

        def ranking_key(index: int) -> tuple[float, str]:
            return (-scores[index], self._document_ids[index])

        if len(scores) > limit:
            ranked_indexes = nsmallest(limit, scores, key=ranking_key)
        else:
            ranked_indexes = sorted(scores, key=ranking_key)

        if len(ranked_indexes) < limit:
            for document_index in self._document_indexes_by_id:
                if document_index in scores:
                    continue
                ranked_indexes.append(document_index)
                if len(ranked_indexes) == limit:
                    break

        return ranked_indexes, scores

    def rank(
        self,
        query_text: str,
        *,
        top_k: int = 5,
    ) -> list[tuple[_BM25DocumentT, float]]:
        """Rank documents for query text, ordered by score then document ID."""

        ranked_indexes, scores = self.rank_positions(query_text, top_k=top_k)
        return [(self.documents[index], scores.get(index, 0.0)) for index in ranked_indexes]

    def retrieve(
        self,
        topic: BenchmarkTopic,
        *,
        run_id: str,
        system_id: str = "bm25",
        top_k: int = 5,
    ) -> list[Candidate]:
        """Retrieve candidates for one topic using the common TAIM schema."""

        return [
            Candidate(
                run_id=run_id,
                system_id=system_id,
                topic_id=topic.topic_id,
                trial_id=cast(_TrialBM25Document, document).trial_id,
                rank=rank,
                score=score,
            )
            for rank, (document, score) in enumerate(
                self.rank(topic.canonical_text, top_k=top_k), start=1
            )
        ]

    def retrieve_many(
        self,
        topics: Iterable[BenchmarkTopic],
        *,
        run_id: str,
        system_id: str = "bm25",
        top_k: int = 5,
    ) -> list[Candidate]:
        """Retrieve candidates for each topic in iterable order."""

        return [
            candidate
            for topic in topics
            for candidate in self.retrieve(
                topic,
                run_id=run_id,
                system_id=system_id,
                top_k=top_k,
            )
        ]


class StreamingBM25Index:
    """Okapi BM25 over a corpus too large to hold, restricted to one query vocabulary.

    :class:`BM25Retriever` keeps every document and every posting, which on the
    375,580-trial ``official-full`` corpus is a run the harness measured at about
    106 GiB of peak memory.  BM25's ranking, though, is a closed-form function of
    the corpus and of the query terms only: a term absent from every query
    contributes nothing to any score.  This index therefore streams the corpus
    once, keeps each document's *total* length so the length normalisation stays
    the whole corpus's, and retains postings **only for the terms the declared
    queries actually contain**.

    The ranking is :class:`BM25Retriever`'s, not an approximation of it: same IDF,
    same length normalisation, same term contribution, same
    ``(-score, document_id)`` tie-break, same zero-score fill.
    ``tests/test_bm25.py`` pins the two against each other on a shared corpus.
    """

    def __init__(
        self,
        documents: Iterable[tuple[str, str]],
        *,
        query_vocabulary: Iterable[str],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and greater than zero")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between zero and one")
        vocabulary = frozenset(query_vocabulary)
        if any(not isinstance(term, str) or not term for term in vocabulary):
            raise ValueError("query_vocabulary terms must be non-empty strings")

        self.k1 = k1
        self.b = b
        self._vocabulary = vocabulary
        document_ids: list[str] = []
        self._document_lengths = array(_POSTING_ARRAY_TYPE)
        postings: dict[str, tuple[array, array]] = {
            term: (array(_POSTING_ARRAY_TYPE), array(_POSTING_ARRAY_TYPE)) for term in vocabulary
        }
        total_terms = 0
        for document_index, (document_id, document_text) in enumerate(documents):
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("document_id values must be non-empty strings")
            term_frequencies = Counter(tokenize(document_text))
            length = sum(term_frequencies.values())
            document_ids.append(document_id)
            self._document_lengths.append(length)
            total_terms += length
            for term in vocabulary.intersection(term_frequencies):
                document_indexes, frequencies = postings[term]
                document_indexes.append(document_index)
                frequencies.append(term_frequencies[term])

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be unique")
        self._document_ids = tuple(document_ids)
        self._postings = {term: posting for term, posting in postings.items() if posting[0]}
        self._average_document_length = (
            total_terms / len(self._document_lengths) if self._document_lengths else 0.0
        )
        self._length_normalizations = array(
            "d",
            (
                self.k1
                * (1.0 - self.b + self.b * (document_length / self._average_document_length))
                if self._average_document_length
                else self.k1 * (1.0 - self.b)
                for document_length in self._document_lengths
            ),
        )
        self._document_indexes_by_id = array(
            _POSTING_ARRAY_TYPE,
            sorted(range(len(self._document_ids)), key=lambda index: self._document_ids[index]),
        )
        self._corpus_statistics: dict[str, int | float | str] = {
            "document_count": len(self._document_ids),
            "total_terms": total_terms,
            "average_document_length": self._average_document_length,
            "indexed_query_terms": len(self._postings),
            "declared_query_vocabulary_size": len(vocabulary),
            "index_representation": _INDEX_REPRESENTATION + "_query_terms_only",
        }

    @property
    def document_ids(self) -> tuple[str, ...]:
        return self._document_ids

    @property
    def corpus_statistics(self) -> dict[str, int | float | str]:
        """Return JSON-safe statistics describing the streamed corpus index."""

        return dict(self._corpus_statistics)

    def _inverse_document_frequency(self, document_frequency: int) -> float:
        corpus_size = len(self._document_ids)
        return math.log(1.0 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5))

    def rank(self, query_text: str, *, top_k: int) -> list[tuple[str, float]]:
        """Rank document IDs for query text, ordered by score then document ID."""

        if top_k < 0:
            raise ValueError("top_k must not be negative")
        query_term_frequency = Counter(tokenize(query_text))
        unknown = sorted(set(query_term_frequency) - self._vocabulary)
        if unknown:
            raise ValueError(
                "query contains terms outside the declared vocabulary this index was built "
                f"for: {unknown[0]!r}; rebuild the index with the full query vocabulary"
            )
        if top_k == 0 or not self._document_ids:
            return []

        scores: dict[int, float] = {}
        for term, query_frequency in query_term_frequency.items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            document_indexes, frequencies = posting
            inverse_document_frequency = self._inverse_document_frequency(len(document_indexes))
            for document_index, frequency in zip(document_indexes, frequencies, strict=True):
                contribution = (
                    query_frequency
                    * inverse_document_frequency
                    * frequency
                    * (self.k1 + 1.0)
                    / (frequency + self._length_normalizations[document_index])
                )
                scores[document_index] = scores.get(document_index, 0.0) + contribution

        limit = min(top_k, len(self._document_ids))

        def ranking_key(index: int) -> tuple[float, str]:
            return (-scores[index], self._document_ids[index])

        if len(scores) > limit:
            ranked_indexes = nsmallest(limit, scores, key=ranking_key)
        else:
            ranked_indexes = sorted(scores, key=ranking_key)

        if len(ranked_indexes) < limit:
            for document_index in self._document_indexes_by_id:
                if document_index in scores:
                    continue
                ranked_indexes.append(document_index)
                if len(ranked_indexes) == limit:
                    break

        return [(self._document_ids[index], scores.get(index, 0.0)) for index in ranked_indexes]

    def retrieve_many(
        self,
        queries: Mapping[str, str],
        *,
        run_id: str,
        system_id: str = "bm25",
        top_k: int,
    ) -> list[Candidate]:
        """Retrieve candidates for a mapping of topic ID to query text."""

        return [
            Candidate(
                run_id=run_id,
                system_id=system_id,
                topic_id=topic_id,
                trial_id=document_id,
                rank=rank,
                score=score,
            )
            for topic_id in sorted(queries, key=_topic_sort_key)
            for rank, (document_id, score) in enumerate(
                self.rank(queries[topic_id], top_k=top_k), start=1
            )
        ]


def query_vocabulary(queries: Iterable[str]) -> frozenset[str]:
    """The union of the BM25 terms of every declared query."""

    return frozenset(term for query in queries for term in tokenize(query))


def _topic_sort_key(topic_id: str) -> tuple[int, str]:
    """Numeric topic order where the IDs are numeric, lexical otherwise."""

    return (int(topic_id), "") if topic_id.isdigit() else (0, topic_id)
