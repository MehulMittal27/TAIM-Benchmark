"""A negation in folded-query assertion never reaches past the end of its line."""

from __future__ import annotations

# ruff: noqa: S101
from collections.abc import Iterable
from typing import Any

import pytest

from taim.query_folding import (
    LINE_BREAK_SENTENCES,
    ConceptLexicon,
    build_context_pipeline,
    build_extraction,
    normalize_concept_text,
)


def _lexicon(forms: Iterable[str]) -> ConceptLexicon:
    surface_forms = {
        " ".join(normalize_concept_text(form)): frozenset({f"concept-{index}"})
        for index, form in enumerate(forms)
    }
    return ConceptLexicon(
        surface_forms=surface_forms,
        semantic_tag={concept: "finding" for ids in surface_forms.values() for concept in ids},
        release_name="synthetic",
        description_file_sha256="sha256:" + "0" * 64,
    )


@pytest.fixture(scope="module")
def context_pipeline() -> tuple[Any, dict[str, Any]]:
    pytest.importorskip("medspacy")
    return build_context_pipeline()


def _query(text: str, forms: tuple[str, ...], pipeline: tuple[Any, dict[str, Any]]) -> str:
    nlp, provenance = pipeline
    per_topic: Any = build_extraction({"1": text}, _lexicon(forms), nlp, provenance)["per_topic"]
    return str(per_topic["1"]["folded_query"])


@pytest.mark.parametrize(
    ("text", "query"),
    [
        # PyRuSH alone keeps these lines in one sentence, so "no" negated all three.
        ("No fever\n[Cough] present\n[Asthma] present", "Cough, Asthma"),
        ("[History] no fever\n[Symptoms] cough", "cough"),
        ("No fever  \n  [Cough] present", "Cough"),
    ],
    ids=["bracket-lines", "negation-inside-a-line", "spaced-line-break"],
)
def test_a_negation_stops_at_the_end_of_its_line(
    text: str, query: str, context_pipeline: tuple[Any, dict[str, Any]]
) -> None:
    assert _query(text, ("fever", "cough", "asthma"), context_pipeline) == query


def test_a_negation_still_reaches_across_its_own_line(
    context_pipeline: tuple[Any, dict[str, Any]],
) -> None:
    text = "[Symptoms] no fever, cough or asthma\n[Other] wheeze"
    forms = ("fever", "cough", "asthma", "wheeze")

    assert _query(text, forms, context_pipeline) == "wheeze"


def test_text_without_line_breaks_splits_as_before(
    context_pipeline: tuple[Any, dict[str, Any]],
) -> None:
    nlp, _ = context_pipeline
    text = "No fever, cough. Asthma [since 2010] and no wheeze; she denies pain"
    pyrush_only = build_context_pipeline()[0]
    pyrush_only.remove_pipe(LINE_BREAK_SENTENCES)

    assert [s.text for s in nlp(text).sents] == [s.text for s in pyrush_only(text).sents]


def test_the_pipeline_records_the_line_break_rule(
    context_pipeline: tuple[Any, dict[str, Any]],
) -> None:
    _, provenance = context_pipeline

    assert provenance["pipeline"] == [
        "medspacy_pyrush",
        LINE_BREAK_SENTENCES,
        "medspacy_context",
    ]
