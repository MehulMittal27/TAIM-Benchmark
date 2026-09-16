"""Folded query construction: the topic's asserted clinical content, unchanged.

A *folded* query is what is left of a TREC topic when the narrative is removed
and nothing else is done to it.  Three decisions define it, and all three are
deliberate:

* **asserted terms only** - a topic that says *"she does not smoke"* names
  smoking, and lifting that term into a query would retrieve smoking studies
  more strongly than the raw narrative did.  Assertion detection is therefore
  not optional.  It runs medspaCy's ConText with its packaged English rule set
  **plus the stated** :data:`POLARITY_CONTEXT_RULES`, and a word negated by the
  ``non`` prefix never starts a term.  Both close ways the packaged rules let a
  negated term through: a blinded audit of the 75 TREC 2021 queries (2026-09-11)
  found ten searching for a condition the note says the patient does not have;
* **no normalisation** - each surviving term is written back in the topic's own
  surface form, not in a lexicon form, so the query says what the patient's note
  said;
* **duplicates kept** - a term the topic states three times enters the query
  three times.  Repetition multiplies a term's BM25 weight, and folding spends
  that weight on what the note actually emphasises.

The extraction that feeds it is rule-based end to end: spans of the topic that
name a clinically tagged SNOMED CT concept, plus the patient's age and sex by two
stated patterns.  No model is involved, so there is no prompt to record; what
would be a prompt is the lexicon digest, the semantic-tag set, and the two
patterns, all of which this module writes into its provenance.

SNOMED CT is licensed content.  Every term this module emits is a span of the
topic text - that is how it matched - and no concept identifier or SNOMED
description text reaches an output artifact.  The release is identified by name
and by the SHA-256 of its description file.

medspaCy is an optional dependency: composing a folded query from an existing
extraction, and every invariant pinned over one, need neither medspaCy nor a
SNOMED release.  Only :func:`build_context_pipeline` and :func:`build_extraction`
import it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

from taim.schemas import JsonValue

# --- the lexicon -----------------------------------------------------------
# The lexicon, its clinical-tag filter and its normalisation are the ones the
# worst-topic diagnosis used.  A second answer to "which spans are clinical" is
# how two analyses of the same topics stop being comparable, so this is one
# answer, stated once.

FULLY_SPECIFIED_NAME = "900000000000003001"
CLINICAL_SEMANTIC_TAGS = frozenset(
    {
        "disorder",
        "finding",
        "procedure",
        "morphologic abnormality",
        "substance",
        "product",
        "medicinal product",
        "medicinal product form",
        "clinical drug",
        "body structure",
        "cell structure",
        "cell",
        "organism",
        "situation",
        "event",
        "regime/therapy",
        "observable entity",
        "specimen",
        "physical object",
    }
)
# Which SNOMED semantic tags count as the patient's *condition*.
CONDITION_TAGS = frozenset({"disorder", "morphologic abnormality"})
MINIMUM_TERM_CHARACTERS = 4
MAXIMUM_TERM_TOKENS = 15

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_SEMANTIC_TAG = re.compile(r"\(([^)]*)\)$")
_WORD = re.compile(r"[a-z0-9]+")

# --- assertion vocabulary --------------------------------------------------
# HISTORICAL is deliberately absent: a resolved or past condition is still the
# patient's, and trial eligibility routinely turns on it, so it is recorded as a
# flag and does not remove a term from the query.
ASSERTION_BY_CONTEXT_CATEGORY = {
    "NEGATED_EXISTENCE": "negated",
    "FAMILY": "family_history",
    "HYPOTHETICAL": "hypothetical",
    "POSSIBLE_EXISTENCE": "uncertain",
}
# Applied in this order, so a term that is both negated and family-scoped is
# reported as negated.  Ordered most-exclusionary first.
ASSERTION_PRECEDENCE = ("negated", "family_history", "hypothetical", "uncertain")
ASSERTED = "asserted"
QUERY_ASSERTIONS = frozenset({ASSERTED})

# --- polarity rules --------------------------------------------------------
# Stated additions to medspaCy's packaged English ConText rules.  Each closes a
# way the packaged rules let a term the note negates reach the query as asserted.
#
# medspaCy prunes overlapping cue matches to the longest and, on a tie, keeps the
# earlier one.  That tie is the first defect, and it is why every addition is a
# whole phrase: a cue shorter than a packaged cue it overlaps never fires.
POLARITY_CONTEXT_RULES: tuple[dict[str, JsonValue], ...] = (
    # "The biopsy was negative for carcinoma": the packaged backward "was negative"
    # beat the forward "negative for" on the tie, so the biopsy was negated and the
    # carcinoma asserted.  The optional "has" also outgrows "has been negative",
    # and ":" keeps "Beta hcg: negative for pregnancy" ahead of ": negative" below.
    {
        "literal": "was negative for",
        "category": "NEGATED_EXISTENCE",
        "direction": "FORWARD",
        "pattern": [
            {"LOWER": {"IN": ["has", "have"]}, "OP": "?"},
            {"LOWER": {"IN": ["is", "are", "was", "were", "be", "been", ":"]}},
            {"LOWER": {"IN": ["negative", "neg"]}},
            {"LOWER": "for"},
        ],
    },
    # A positive result ends a negation's scope: "negative for carcinoma and was
    # only remarkable for ... chronic viral hepatitis".  PSEUDO in the negation's
    # own category modifies nothing and limits only NEGATED_EXISTENCE scopes; a
    # TERMINATE cue would also cut family scopes, and "his family history is
    # significant for asthma in his mother" must stay the mother's.
    {
        "literal": "remarkable for",
        "category": "NEGATED_EXISTENCE",
        "direction": "PSEUDO",
        "pattern": [
            {"LOWER": {"IN": ["positive", "remarkable", "significant"]}},
            {"LOWER": "for"},
        ],
    },
    # "Leukocyte esterase: negative" negates the label that ends at the colon and
    # nothing before it, so a lab block's earlier labels keep their own results.
    {
        "literal": ": negative",
        "category": "NEGATED_EXISTENCE",
        "direction": "BACKWARD",
        "max_scope": 1,
        "pattern": [{"LOWER": ":"}, {"LOWER": {"IN": ["negative", "neg"]}}],
    },
    {
        "literal": "negative history of",
        "category": "NEGATED_EXISTENCE",
        "direction": "FORWARD",
        "pattern": [
            {"LOWER": {"IN": ["negative", "neg"]}},
            {"LOWER": {"IN": ["history", "hx"]}},
            {"LOWER": {"IN": ["of", "for"]}},
        ],
    },
    # "unable to perform thyroidectomy --high bleed risk, proximity to trachea ...":
    # the object is one short noun phrase, not the run-on list after it.
    {
        "literal": "unable to perform",
        "category": "NEGATED_EXISTENCE",
        "direction": "FORWARD",
        "max_scope": 3,
        "pattern": [
            {"LOWER": "unable"},
            {"LOWER": "to"},
            {"LOWER": {"IN": ["perform", "undergo"]}},
        ],
    },
)
POLARITY_CONTEXT_RULES_SHA256 = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(POLARITY_CONTEXT_RULES, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
)
# "non-bloody diarrhea" is diarrhea that is not bloody.  ConText negates whole
# terms, so a cue would drop the diarrhea as well; instead the word directly after
# this prefix never starts a term and matching resumes at the word after it.  A
# lexicon form that itself begins with the prefix ("non-small cell lung cancer")
# still matches from the prefix.
NEGATING_PREFIX = "non"

# --- demographics ----------------------------------------------------------
# Not SNOMED spans.  These two patterns are the whole of the demographic rule
# set; they are the only hand-written extraction rules in this module.
AGE_PATTERN = re.compile(
    r"\b(?P<value>\d{1,3})[\s-]*(?:year|years|yr|yrs|y)[\s-]*(?:old|o)\b"
    r"|\b(?P<yo>\d{1,3})\s*(?:yo|y/o)\b",
    re.IGNORECASE,
)
SEX_TERMS = {
    "man": "male",
    "men": "male",
    "male": "male",
    "males": "male",
    "boy": "male",
    "boys": "male",
    "gentleman": "male",
    "woman": "female",
    "women": "female",
    "female": "female",
    "females": "female",
    "girl": "female",
    "girls": "female",
    "lady": "female",
}

# --- the folding policy ----------------------------------------------------

FOLDED_QUERY_POLICY: dict[str, JsonValue] = {
    "form": "folded",
    "selection": "asserted terms only",
    "normalisation": "none; each term is written back in the topic's own surface form",
    "duplicates": "kept",
    "order": "first appearance in the topic",
    "joiner": ", ",
    "implementation": "taim.query_folding.fold_terms",
}
FOLDED_QUERY_JOINER = ", "


class Word(NamedTuple):
    """One normalised word of the topic, with its span in the original text."""

    text: str
    start: int
    end: int


class Extraction(NamedTuple):
    """One extracted term, before assertion detection has run."""

    term: str
    kind: str
    semantic_tags: tuple[str, ...]
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ConceptLexicon:
    """SNOMED surface forms, and the semantic tag of every concept they name."""

    surface_forms: dict[str, frozenset[str]]
    semantic_tag: dict[str, str]
    release_name: str
    description_file_sha256: str

    @property
    def maximum_tokens(self) -> int:
        return max((len(form.split()) for form in self.surface_forms), default=1)

    def provenance(self) -> dict[str, JsonValue]:
        return {
            "release": self.release_name,
            "description_file_sha256": self.description_file_sha256,
            "indexed_surface_forms": len(self.surface_forms),
            "clinical_semantic_tags": cast(list[JsonValue], sorted(CLINICAL_SEMANTIC_TAGS)),
            "minimum_term_characters": MINIMUM_TERM_CHARACTERS,
            "maximum_term_tokens": MAXIMUM_TERM_TOKENS,
        }


def normalize(text: str) -> list[str]:
    """Casefold and split on anything that is not a letter or digit."""

    return _NON_ALPHANUMERIC.sub(" ", text.lower()).split()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _release_file(release: Path, prefix: str) -> Path:
    directory = release / "Snapshot" / "Terminology"
    matches = sorted(directory.glob(f"{prefix}*.txt"))
    if not matches:
        raise ValueError(f"no {prefix}* file under {directory}")
    return matches[0]


def load_concept_lexicon(release: Path) -> ConceptLexicon:
    """Index the active English surface forms of clinically meaningful concepts."""

    active: set[str] = set()
    with _release_file(release, "sct2_Concept_Snapshot").open(encoding="utf-8") as stream:
        next(stream)
        for line in stream:
            fields = line.split("\t")
            if fields[2] == "1":
                active.add(fields[0])

    description_file = _release_file(release, "sct2_Description_Snapshot-en")
    semantic_tag: dict[str, str] = {}
    terms: dict[str, set[str]] = {}
    with description_file.open(encoding="utf-8") as stream:
        next(stream)
        for line in stream:
            fields = line.split("\t")
            if fields[2] != "1":
                continue
            concept_id = fields[4]
            if concept_id not in active:
                continue
            term = fields[7]
            if fields[6] == FULLY_SPECIFIED_NAME:
                # Only the fully specified name carries the semantic tag, and the
                # tag is not part of the term anyone writes in a clinical note.
                match = _SEMANTIC_TAG.search(term)
                if match:
                    semantic_tag[concept_id] = match.group(1)
                    term = term[: match.start()].strip()
            terms.setdefault(concept_id, set()).add(term)

    surface_forms: dict[str, set[str]] = {}
    for concept_id, concept_terms in terms.items():
        if semantic_tag.get(concept_id) not in CLINICAL_SEMANTIC_TAGS:
            continue
        for term in concept_terms:
            form = " ".join(normalize(term))
            if len(form) < MINIMUM_TERM_CHARACTERS:
                continue
            if len(form.split()) > MAXIMUM_TERM_TOKENS:
                continue
            surface_forms.setdefault(form, set()).add(concept_id)

    return ConceptLexicon(
        surface_forms={form: frozenset(ids) for form, ids in surface_forms.items()},
        semantic_tag=semantic_tag,
        release_name=release.name,
        description_file_sha256=sha256_file(description_file),
    )


def words(text: str) -> list[Word]:
    """Split exactly as :func:`normalize` does, but keeping offsets.

    ``normalize`` collapses every non-alphanumeric run and lowercases, which is
    the same partition as ``[a-z0-9]+`` over the lowered text - and the same
    partition :func:`taim.baselines.bm25.tokenize` uses, so a term written back
    into a folded query survives BM25 tokenisation unchanged.
    """

    lowered = text.lower()
    found = [Word(match.group(), match.start(), match.end()) for match in _WORD.finditer(lowered)]
    if [word.text for word in found] != normalize(text):
        raise ValueError("offset-preserving tokenisation disagrees with normalize()")
    return found


def lexicon_spans(
    text: str,
    surface_forms: Mapping[str, frozenset[str]],
    semantic_tag: Mapping[str, str],
    maximum_tokens: int,
) -> list[Extraction]:
    """Longest-match, left-to-right, non-overlapping lexicon spans.

    Non-overlapping matters: it is what keeps ``colon cancer`` from also
    contributing ``colon`` and ``cancer`` as separate query terms and silently
    tripling that concept's weight.  Longest-first matters because specificity is
    the whole point - ``major depressive disorder`` beats ``disorder``.  No span
    starts at a word that :data:`NEGATING_PREFIX` negates.
    """

    stream = words(text)
    spans: list[Extraction] = []
    index = 0
    while index < len(stream):
        if index > 0 and stream[index - 1].text == NEGATING_PREFIX:
            index += 1
            continue
        limit = min(maximum_tokens, len(stream) - index)
        for length in range(limit, 0, -1):
            form = " ".join(word.text for word in stream[index : index + length])
            concepts = surface_forms.get(form)
            if not concepts:
                continue
            tags = sorted({semantic_tag[c] for c in concepts if c in semantic_tag})
            spans.append(
                Extraction(
                    term=form,
                    kind="snomed_concept",
                    semantic_tags=tuple(tags),
                    start=stream[index].start,
                    end=stream[index + length - 1].end,
                )
            )
            index += length
            break
        else:
            index += 1
    return spans


def demographic_spans(text: str) -> list[Extraction]:
    """The patient's age and sex, by the two stated patterns and nothing else."""

    spans: list[Extraction] = []
    for match in AGE_PATTERN.finditer(text):
        value = match.group("value") or match.group("yo")
        spans.append(
            Extraction(
                term=f"{int(value)} year old",
                kind="demographic_age",
                semantic_tags=(),
                start=match.start(),
                end=match.end(),
            )
        )
    for word in words(text):
        sex = SEX_TERMS.get(word.text)
        if sex:
            spans.append(
                Extraction(
                    term=sex,
                    kind="demographic_sex",
                    semantic_tags=(),
                    start=word.start,
                    end=word.end,
                )
            )
    return spans


def merge_spans(*groups: Sequence[Extraction]) -> list[Extraction]:
    """Order by position and drop any span contained in an earlier-kept one.

    Demographic spans are offered first so ``45 year old`` survives against a
    lexicon span covering the same characters.
    """

    kept: list[Extraction] = []
    for group in groups:
        for span in group:
            if any(span.start < other.end and other.start < span.end for other in kept):
                continue
            kept.append(span)
    return sorted(kept, key=lambda span: (span.start, span.end))


def build_context_pipeline() -> tuple[Any, dict[str, JsonValue]]:
    """medspaCy's default English pipeline, minus its target matcher.

    Targets come from the SNOMED lexicon, not from medspaCy's rule matcher, so
    the matcher is disabled; the sentence splitter is the default, and ConText
    runs its packaged rules plus :data:`POLARITY_CONTEXT_RULES`.  The returned
    provenance carries the library version, the SHA-256 of the rule file actually
    loaded and the SHA-256 of the additions, which is what makes an extraction
    reproducible: the same code against a different packaged rule set is a
    different pipeline.
    """

    import medspacy
    from medspacy.context import ConTextRule

    nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_context"])
    context = nlp.get_pipe("medspacy_context")
    default_rules = Path(context.DEFAULT_RULES_FILEPATH)
    context.add([ConTextRule.from_dict(copy.deepcopy(rule)) for rule in POLARITY_CONTEXT_RULES])
    provenance: dict[str, JsonValue] = {
        "library": "medspacy",
        "library_version": medspacy.__version__,
        "pipeline": [name for name, _ in nlp.pipeline],
        "target_extraction": (
            "SNOMED lexicon longest-match spans plus the two demographic patterns; "
            "medspacy_target_matcher is disabled and contributes nothing"
        ),
        "context_rules": "medspaCy packaged default English rule set plus stated polarity rules",
        "context_rules_path": str(default_rules),
        "context_rules_sha256": sha256_file(default_rules),
        "context_rule_count": len(context.rules),
        "context_rule_categories": dict(
            sorted(Counter(rule.category for rule in context.rules).items())
        ),
        "extra_context_rules": cast(list[JsonValue], list(POLARITY_CONTEXT_RULES)),
        "extra_context_rules_sha256": POLARITY_CONTEXT_RULES_SHA256,
    }
    try:
        import spacy

        provenance["spacy_version"] = spacy.__version__
    except ImportError:  # pragma: no cover - medspaCy cannot load without spaCy
        pass
    return nlp, provenance


def assert_terms(nlp: Any, text: str, spans: Sequence[Extraction]) -> list[dict[str, JsonValue]]:
    """Run ConText over the extracted spans and tag each one."""

    # ConText runs twice: once inside nlp() on a doc with no entities, where it
    # has nothing to attach to and does nothing, and once below on the doc whose
    # entities are the lexicon spans.
    doc = nlp(text)
    entities: list[tuple[Extraction, Any]] = []
    claimed: set[int] = set()
    for span in spans:
        entity = doc.char_span(span.start, span.end, label=span.kind, alignment_mode="expand")
        if entity is None:  # pragma: no cover - expand never returns None on a real span
            continue
        # Character spans can be disjoint and still land inside one spaCy token -
        # "45-year-old" is a single token here. spaCy allows one entity per token,
        # so the first span to claim a token keeps it and later ones are dropped.
        tokens = set(range(entity.start, entity.end))
        if tokens & claimed:
            continue
        claimed |= tokens
        entities.append((span, entity))
    doc.ents = tuple(entity for _, entity in entities)
    doc = nlp.get_pipe("medspacy_context")(doc)

    tagged: list[dict[str, JsonValue]] = []
    for span, entity in zip((s for s, _ in entities), doc.ents, strict=True):
        categories = sorted({modifier.category.upper() for modifier in entity._.modifiers})
        assertions = {
            ASSERTION_BY_CONTEXT_CATEGORY[category]
            for category in categories
            if category in ASSERTION_BY_CONTEXT_CATEGORY
        }
        assertion = next((name for name in ASSERTION_PRECEDENCE if name in assertions), ASSERTED)
        triggers = [
            {
                "category": modifier.category.upper(),
                "trigger_text": doc[slice(*modifier.modifier_span)].text,
                "rule_literal": modifier.rule.literal,
                "direction": modifier.direction,
            }
            for modifier in entity._.modifiers
        ]
        tagged.append(
            {
                "term": span.term,
                "kind": span.kind,
                "semantic_tags": list(span.semantic_tags),
                "is_condition": bool(set(span.semantic_tags) & CONDITION_TAGS),
                "text_span": text[span.start : span.end],
                "character_start": span.start,
                "character_end": span.end,
                "sentence": entity.sent.text.strip(),
                "assertion": assertion,
                "is_historical": "HISTORICAL" in categories,
                "context_modifiers": cast(list[JsonValue], triggers),
                "in_query": assertion in QUERY_ASSERTIONS,
            }
        )
    return tagged


def fold_terms(terms: Iterable[Mapping[str, object]]) -> str:
    """Compose the folded query from one topic's tagged extraction terms.

    Asserted terms only, in the topic's own surface form, duplicates kept, in
    order of first appearance, joined by ``", "``.  This is the whole of the
    folding rule and it takes no options: a knob here is a knob on the headline
    number, and the configuration this implements was decided before it ran.
    """

    return FOLDED_QUERY_JOINER.join(
        str(term["text_span"]) for term in terms if term["assertion"] == ASSERTED
    )


def folded_queries(extraction: Mapping[str, object]) -> dict[str, str]:
    """Fold every topic of a committed extraction artifact."""

    per_topic = extraction["per_topic"]
    if not isinstance(per_topic, Mapping):
        raise ValueError("extraction must carry a per_topic mapping")
    folded: dict[str, str] = {}
    for topic_id, topic in per_topic.items():
        if not isinstance(topic, Mapping):
            raise ValueError(f"extraction entry for topic {topic_id!r} must be a JSON object")
        terms = topic["terms"]
        if not isinstance(terms, list):
            raise ValueError(f"extraction entry for topic {topic_id!r} must carry its terms")
        folded[str(topic_id)] = fold_terms(terms)
    return folded


def build_extraction(
    topics: Mapping[str, str],
    lexicon: ConceptLexicon,
    nlp: Any,
    assertion_provenance: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Extract, assert and fold every topic, and record how it was done."""

    maximum = min(lexicon.maximum_tokens, MAXIMUM_TERM_TOKENS)
    per_topic: dict[str, JsonValue] = {}
    counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    for topic_id, text in sorted(topics.items(), key=lambda item: int(item[0])):
        spans = merge_spans(
            demographic_spans(text),
            lexicon_spans(text, lexicon.surface_forms, lexicon.semantic_tag, maximum),
        )
        terms = assert_terms(nlp, text, spans)
        query = fold_terms(terms)
        for term in terms:
            counts[str(term["assertion"])] += 1
            if term["is_condition"]:
                condition_counts[str(term["assertion"])] += 1
        per_topic[topic_id] = {
            "raw_word_count": len(text.split()),
            "folded_query": query,
            "folded_query_word_count": len(query.split()),
            "extracted_term_count": len(terms),
            "query_term_count": sum(1 for term in terms if term["in_query"]),
            "dropped_term_count": sum(1 for term in terms if not term["in_query"]),
            "assertion_counts": dict(sorted(Counter(str(t["assertion"]) for t in terms).items())),
            "terms": cast(list[JsonValue], terms),
        }
    return {
        "artifact_type": "taim-folded-query-extraction",
        "artifact_version": "1.1",
        "topics": len(topics),
        "snomed_release": lexicon.provenance(),
        "extraction": {
            "target_rule": (
                "longest-match, left-to-right, non-overlapping spans of the topic text that name "
                "a SNOMED concept whose semantic tag is in clinical_semantic_tags, plus the two "
                "demographic patterns; no span starts at the word directly after negating_prefix"
            ),
            "negating_prefix": NEGATING_PREFIX,
            "age_pattern": AGE_PATTERN.pattern,
            "sex_terms": dict(sorted(SEX_TERMS.items())),
            "assertion_precedence": cast(list[JsonValue], list(ASSERTION_PRECEDENCE)),
            "context_category_map": cast(dict[str, JsonValue], dict(ASSERTION_BY_CONTEXT_CATEGORY)),
            "historical_policy": (
                "HISTORICAL is recorded as a flag and does not remove a term from the query: "
                "a past condition is still the patient's and trial eligibility turns on it"
            ),
        },
        "query_composition": dict(FOLDED_QUERY_POLICY),
        "assertion_detection": dict(assertion_provenance),
        "totals": {
            "extracted_terms": sum(counts.values()),
            "by_assertion": dict(sorted(counts.items())),
            "condition_terms_by_assertion": dict(sorted(condition_counts.items())),
        },
        "per_topic": per_topic,
    }


def assertion_provenance_of(extraction: Mapping[str, object]) -> dict[str, JsonValue]:
    """Read back the assertion-detection identity a run manifest has to record.

    The medspaCy version and the SHA-256 of the rule file are not decoration:
    the same folding code against a different packaged rule set drops a
    different set of terms, so an extraction without them is not reproducible.
    """

    provenance = extraction.get("assertion_detection")
    if not isinstance(provenance, Mapping):
        raise ValueError("extraction does not record its assertion-detection provenance")
    required = ("library", "library_version", "context_rules", "context_rules_sha256")
    missing = [field_name for field_name in required if not provenance.get(field_name)]
    if missing:
        raise ValueError(
            "extraction assertion-detection provenance is missing "
            + ", ".join(missing)
            + "; the medspaCy version and the rule-set hash are required in a run manifest"
        )
    return dict(provenance)


__all__ = [
    "AGE_PATTERN",
    "ASSERTED",
    "ASSERTION_BY_CONTEXT_CATEGORY",
    "ASSERTION_PRECEDENCE",
    "CLINICAL_SEMANTIC_TAGS",
    "CONDITION_TAGS",
    "FOLDED_QUERY_JOINER",
    "FOLDED_QUERY_POLICY",
    "MAXIMUM_TERM_TOKENS",
    "NEGATING_PREFIX",
    "POLARITY_CONTEXT_RULES",
    "POLARITY_CONTEXT_RULES_SHA256",
    "QUERY_ASSERTIONS",
    "SEX_TERMS",
    "ConceptLexicon",
    "Extraction",
    "Word",
    "assert_terms",
    "assertion_provenance_of",
    "build_context_pipeline",
    "build_extraction",
    "demographic_spans",
    "fold_terms",
    "folded_queries",
    "lexicon_spans",
    "load_concept_lexicon",
    "merge_spans",
    "normalize",
    "sha256_file",
    "words",
]
