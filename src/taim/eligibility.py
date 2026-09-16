"""Source-grounded complete eligibility text and deterministic criterion boundaries."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from taim.contracts import (
    ELIGIBILITY_CRITERION_VIEW_CONTENT_SCHEMA,
    ELIGIBILITY_CRITERION_VIEW_NAME,
    ELIGIBILITY_SPLIT_VERSION,
    canonical_json,
    content_sha256,
    require_sha256,
)
from taim.schemas import JsonValue, SchemaValidationError
from taim.snapshot import (
    DERIVED_VIEW_CAPABILITY_PREFIX,
    BenchmarkSnapshot,
    CriterionItem,
    DerivedView,
    SemanticTextSection,
)

ELIGIBILITY_CRITERION_VIEW_CAPABILITY = (
    f"{DERIVED_VIEW_CAPABILITY_PREFIX}{ELIGIBILITY_CRITERION_VIEW_NAME}"
)
ELIGIBILITY_VIEW_CONFIGURATION: dict[str, JsonValue] = {
    "criterion_origin": "derived",
    "missing_heading_policy": "unlabelled_block_is_unspecified",
    "non_criterion_item_policy": "preserve_typed_source_spans",
    "source_span_offset_unit": "unicode_code_point",
    "source_span_end": "exclusive",
}

CriterionPolarity = Literal["inclusion", "exclusion", "unspecified"]
ListMarkerType = Literal["bullet", "number", "letter"]
CriterionChildRelationship = Literal["all_of", "any_of", "at_least_n", "unknown"]
EligibilityParentType = Literal["criterion", "group_header"]
EligibilityItemType = Literal["group_header", "context", "declared_empty", "unavailable"]
EligibilityPolarityStatus = Literal[
    "criteria",
    "declared_empty",
    "unavailable",
    "not_identified",
]
EligibilityParseTier = Literal[
    "explicit_list",
    "explicit_prose",
    "implicit_polarity",
    "incomplete_polarity",
    "empty",
]
EligibilityReviewReason = Literal[
    "missing_inclusion_criteria",
    "missing_exclusion_criteria",
    "implicit_inclusion_polarity",
    "implicit_exclusion_polarity",
    "unmarked_criterion_boundaries",
    "unspecified_polarity",
    "non_criterion_items",
    "declared_empty_inclusion",
    "declared_empty_exclusion",
    "unavailable_inclusion",
    "unavailable_exclusion",
]
ELIGIBILITY_PARSE_TIERS = frozenset(
    {
        "explicit_list",
        "explicit_prose",
        "implicit_polarity",
        "incomplete_polarity",
        "empty",
    }
)
ELIGIBILITY_REVIEW_REASONS = frozenset(
    {
        "missing_inclusion_criteria",
        "missing_exclusion_criteria",
        "implicit_inclusion_polarity",
        "implicit_exclusion_polarity",
        "unmarked_criterion_boundaries",
        "unspecified_polarity",
        "non_criterion_items",
        "declared_empty_inclusion",
        "declared_empty_exclusion",
        "unavailable_inclusion",
        "unavailable_exclusion",
    }
)

_LIST_MARKER = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-*•·▪‣]|\[\d{1,3}\]|\[[A-Za-z]\]|"
    r"\(\d{1,3}\)[.)]?|\([A-Za-z]\)[.)]?|\d{1,3}[.)]|[A-Za-z][.)])"
    r"(?P<spacing>[ \t]*)(?=\S)"
)
_INLINE_LIST_MARKER = re.compile(r"(?<=[.;:])(?P<gap>[ \t]+)(?P<marker>[-*•·▪‣])[ \t]*(?=\S)")
_HEADING_LEAD = (
    r"\s*(?:[-*•·▪‣]\s*)?(?:[<\[(]\s*)?"
    r"(?:(?:\d+(?:\.\d+)+|\d+)[.)]?\s+)?(?:the\s+(?:following\s+)?)?"
)
_HEADING_CORE = re.compile(
    rf"^{_HEADING_LEAD}(?P<qualifiers>(?:[\w'\u2019./-]+\s+){{0,5}}?)"
    r"(?P<kind>non[-\s]+inclusion|inclusion|exclusion)\s+"
    r"(?P<noun>[\w]+(?:-[\w]+)*)(?P<rest>.*)$",
    re.IGNORECASE,
)
_BARE_HEADING = re.compile(
    rf"^{_HEADING_LEAD}(?P<kind>inclusion|exclusion)s?\b(?P<rest>.*)$",
    re.IGNORECASE,
)
_COMBINED_HEADING = re.compile(
    rf"^{_HEADING_LEAD}(?:key\s+)?inclusion\s*(?:/|&|and)\s*exclusion"
    r"(?:\s+(?P<noun>[\w]+(?:-[\w]+)*))?(?P<rest>.*)$",
    re.IGNORECASE,
)
_GENERIC_HEADING = re.compile(
    rf"^{_HEADING_LEAD}(?:general\s+)?eligibility"
    r"(?:\s+(?P<noun>[\w]+(?:-[\w]+)*))?"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
_REVERSED_HEADING = re.compile(
    rf"^{_HEADING_LEAD}(?P<noun>[\w]+(?:-[\w]+)*)\s+for\s+"
    r"(?P<qualifiers>(?:[\w'\u2019./-]+\s+){0,5}?)(?P<kind>non[-\s]+inclusion|inclusion|exclusion)"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
_SUBJECT_DECLARATION = re.compile(
    rf"^{_HEADING_LEAD}(?:patients?|subjects?|participants?|individuals?)\s+"
    r"(?:(?:will\s+be|are|were)\s+(?P<kind>included|excluded)|"
    r"will\s+be\s+(?P<enrolled>enrolled))\s*"
    r"(?:(?:if|when)\b)?(?P<rest>.*)$",
    re.IGNORECASE,
)
_POLARITY_CANDIDATE = re.compile(
    r"\b(?:non[-\s]+inclusion|inclusion|exclusion|patients?|subjects?|participants?|"
    r"individuals?|eligibility)\b",
    re.IGNORECASE,
)
_HEADING_VERB = re.compile(
    r"^\s*(?:are|is|were|was|will\s+(?:be|include)|must\s+be|should\s+be|included|includes?|"
    r"(?:requires?|required)(?:\s+that)?|consist(?:s)?\s+of)\b\s*:?[ \t]*"
    r"(?P<tail>.*)$",
    re.IGNORECASE,
)
_SCOPED_HEADING_VERB = re.compile(
    r"^\s*(?:for|of)\s+.{1,80}?\s+(?:are|is|were|was|included|includes?|"
    r"(?:requires?|required)(?:\s+that)?|consist(?:s)?\s+of)\b\s*:?[ \t]*"
    r"(?P<tail>.*)$",
    re.IGNORECASE,
)
_HEADING_SCOPE = re.compile(
    r"^\s*(?:(?:for|of)\s+.{1,80}|(?:specific|shared|applicable)\s+"
    r"(?:for|to|by)\s+.{1,80}|(?:the\s+)?"
    r"(?:patients?|subjects?|participants?|individuals?|caregivers?|controls?|volunteers?|"
    r"donors?|recipients?)\b.{0,80}?)\s*:?\s*$",
    re.IGNORECASE,
)
_HEADING_POINTER = re.compile(
    r"^\s*(?:can|may)\s+be\s+found\b.*$",
    re.IGNORECASE,
)
_EMPTY_HEADING_TAIL = re.compile(
    r"(?:as\s+follows|the\s+following|those\s+who|"
    r"(?:patients?|subjects?|participants?|individuals?)\s+(?:who|must(?:\s+not)?))\s*:?",
    re.IGNORECASE,
)
_QUALIFIER_STOP_WORDS = frozenset(
    {
        "absence",
        "absent",
        "any",
        "fail",
        "failed",
        "failing",
        "failure",
        "lack",
        "lacking",
        "meet",
        "meeting",
        "met",
        "must",
        "no",
        "not",
        "satisfy",
        "satisfying",
        "without",
    }
)
_GROUP_HEADER = re.compile(r".{1,100}:\s*$")
_MEASUREMENT_COMPARISON = re.compile(r"(?:<=|>=|[<>=≤≥])\s*[+-]?\d")
_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_COUNT_PATTERN = r"\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten"
_ALL_OF_CHILDREN = re.compile(
    r"\b(?:all|each|both)\s+(?:of\s+)?(?:the\s+)?(?:following|criteria|conditions|requirements)\b",
    re.IGNORECASE,
)
_ANY_OF_CHILDREN = re.compile(
    r"\b(?:any|either)\s+of\s+(?:the\s+)?(?:following|criteria|conditions|requirements)\b",
    re.IGNORECASE,
)
_AT_LEAST_N_CHILDREN = re.compile(
    rf"\b(?:at\s+least\s+)?(?P<count>{_COUNT_PATTERN})\s+of\s+(?:the\s+)?"
    rf"(?:(?:{_COUNT_PATTERN})\s+)?(?:following|criteria|conditions|requirements|items)"
    r"(?:\s+below)?\b",
    re.IGNORECASE,
)
_LEGACY_CONTEXT = re.compile(
    r"(?:disease|patient|subject|protocol|prior concurrent therapy|age|sex|gender|"
    r"performance status)\s+characteristics?\s*:?"
)
_BOOLEAN_CONTEXT = re.compile(r"(?:and|or|and/or)\s*[:;,.]?", re.IGNORECASE)
_DECLARED_EMPTY = re.compile(
    r"(?:[-\u2013\u2014]+|none|nil|(?:there\s+(?:are|is)\s+)?non?\s+(?:specific\s+)?"
    r"(?:inclusion|exclusion|eligibility)?\s*(?:criteria|requirements?)"
    r"(?:\s+for\s+.{1,80})?)\s*[.!]?",
    re.IGNORECASE,
)
_UNAVAILABLE = re.compile(
    r"(?:n/?a|not\s+(?:available|applicable)|unknown|to\s+be\s+determined|"
    r"(?:(?:other|additional)\s+criteria\s+(?:may\s+)?apply\s*[,;.]\s*)?"
    r"(?:please\s+)?contact\s+(?:the\s+)?(?:site(?:\s+director)?|investigator|study staff|"
    r"trial personnel|research team)(?:\s+directly)?"
    r"(?:\s+(?:for|regarding)\s+.{1,100})?)\s*[.!]?",
    re.IGNORECASE,
)
_OTHER_CRITERIA_UNAVAILABLE = re.compile(
    r"(?:note\s*:\s*)?other(?:\s+protocol[- ]defined)?\s+"
    r"(?:inclusion\s*/\s*exclusion\s+)?criteria\s+(?:may\s+)?apply\s*[.!]?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EligibilityCriterionBoundary:
    text: str
    polarity: CriterionPolarity
    source_start: int
    source_end: int
    list_marked: bool
    list_depth: int = 0
    list_marker: str | None = None
    list_marker_type: ListMarkerType | None = None
    list_marker_ordinal: int | None = None
    list_indent: str = ""
    parent_source_start: int | None = None
    parent_type: EligibilityParentType | None = None
    child_relationship: CriterionChildRelationship = "unknown"
    required_child_count: int | None = None
    group_path: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EligibilityNonCriterionBoundary:
    text: str
    item_type: EligibilityItemType
    polarity: CriterionPolarity
    source_start: int
    source_end: int
    list_marked: bool
    list_depth: int = 0
    list_marker: str | None = None
    list_marker_type: ListMarkerType | None = None
    list_marker_ordinal: int | None = None
    list_indent: str = ""
    parent_source_start: int | None = None
    parent_type: EligibilityParentType | None = None
    child_relationship: CriterionChildRelationship = "unknown"
    required_child_count: int | None = None


@dataclass(frozen=True, slots=True)
class EligibilityCriterionParse:
    """Deterministic criterion boundaries plus structural review signals."""

    criteria: tuple[EligibilityCriterionBoundary, ...]
    other_items: tuple[EligibilityNonCriterionBoundary, ...]
    explicit_polarities: frozenset[CriterionPolarity]
    inclusion_status: EligibilityPolarityStatus
    exclusion_status: EligibilityPolarityStatus
    parse_tier: EligibilityParseTier
    review_reasons: tuple[EligibilityReviewReason, ...]

    @property
    def inclusion(self) -> tuple[EligibilityCriterionBoundary, ...]:
        return tuple(item for item in self.criteria if item.polarity == "inclusion")

    @property
    def exclusion(self) -> tuple[EligibilityCriterionBoundary, ...]:
        return tuple(item for item in self.criteria if item.polarity == "exclusion")


@dataclass(frozen=True, slots=True)
class EligibilityCriterionViewItem:
    identifier: str
    ordinal: int
    text: str
    polarity: CriterionPolarity
    source_start: int
    source_end: int
    source_span_text: str
    source_span_sha256: str
    list_marked: bool
    list_depth: int
    list_marker: str | None
    list_marker_type: ListMarkerType | None
    list_marker_ordinal: int | None
    list_indent: str
    parent_identifier: str | None
    parent_source_start: int | None
    parent_type: EligibilityParentType | None
    child_relationship: CriterionChildRelationship
    required_child_count: int | None
    group_path: tuple[str, ...]
    derived_text_sha256: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "identifier": self.identifier,
            "ordinal": self.ordinal,
            "text": self.text,
            "polarity": self.polarity,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "source_span_text": self.source_span_text,
            "source_span_sha256": self.source_span_sha256,
            "list_marked": self.list_marked,
            "list_depth": self.list_depth,
            "list_marker": self.list_marker,
            "list_marker_type": self.list_marker_type,
            "list_marker_ordinal": self.list_marker_ordinal,
            "list_indent": self.list_indent,
            "parent_identifier": self.parent_identifier,
            "parent_source_start": self.parent_source_start,
            "parent_type": self.parent_type,
            "child_relationship": self.child_relationship,
            "required_child_count": self.required_child_count,
            "group_path": list(self.group_path),
            "derived_text_sha256": self.derived_text_sha256,
        }


@dataclass(frozen=True, slots=True)
class EligibilityNonCriterionViewItem:
    identifier: str
    ordinal: int
    text: str
    item_type: EligibilityItemType
    polarity: CriterionPolarity
    source_start: int
    source_end: int
    source_span_text: str
    source_span_sha256: str
    list_marked: bool
    list_depth: int
    list_marker: str | None
    list_marker_type: ListMarkerType | None
    list_marker_ordinal: int | None
    list_indent: str
    parent_identifier: str | None
    parent_source_start: int | None
    parent_type: EligibilityParentType | None
    child_relationship: CriterionChildRelationship
    required_child_count: int | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "identifier": self.identifier,
            "ordinal": self.ordinal,
            "text": self.text,
            "item_type": self.item_type,
            "polarity": self.polarity,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "source_span_text": self.source_span_text,
            "source_span_sha256": self.source_span_sha256,
            "list_marked": self.list_marked,
            "list_depth": self.list_depth,
            "list_marker": self.list_marker,
            "list_marker_type": self.list_marker_type,
            "list_marker_ordinal": self.list_marker_ordinal,
            "list_indent": self.list_indent,
            "parent_identifier": self.parent_identifier,
            "parent_source_start": self.parent_source_start,
            "parent_type": self.parent_type,
            "child_relationship": self.child_relationship,
            "required_child_count": self.required_child_count,
        }


@dataclass(frozen=True, slots=True)
class EligibilityCriterionTrialRecord:
    trial_id: str
    source_text_sha256: str
    source_provenance_sha256: str
    view_sha256: str
    criterion_inventory_sha256: str
    parse_tier: EligibilityParseTier
    review_reasons: tuple[EligibilityReviewReason, ...]
    inclusion_status: EligibilityPolarityStatus
    exclusion_status: EligibilityPolarityStatus
    inclusion: tuple[EligibilityCriterionViewItem, ...]
    exclusion: tuple[EligibilityCriterionViewItem, ...]
    unspecified: tuple[EligibilityCriterionViewItem, ...]
    other_items: tuple[EligibilityNonCriterionViewItem, ...]

    @property
    def criteria(self) -> tuple[EligibilityCriterionViewItem, ...]:
        return tuple(
            sorted(
                (*self.inclusion, *self.exclusion, *self.unspecified),
                key=lambda item: item.ordinal,
            )
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "trial_id": self.trial_id,
            "source_text_sha256": self.source_text_sha256,
            "source_provenance_sha256": self.source_provenance_sha256,
            "view_sha256": self.view_sha256,
            "criterion_inventory_sha256": self.criterion_inventory_sha256,
            "criterion_count": len(self.criteria),
            "inclusion_criterion_count": len(self.inclusion),
            "exclusion_criterion_count": len(self.exclusion),
            "unspecified_criterion_count": len(self.unspecified),
            "other_item_count": len(self.other_items),
            "inclusion_criteria": [item.to_dict() for item in self.inclusion],
            "exclusion_criteria": [item.to_dict() for item in self.exclusion],
            "unspecified_criteria": [item.to_dict() for item in self.unspecified],
            "other_items": [item.to_dict() for item in self.other_items],
            "inclusion_status": self.inclusion_status,
            "exclusion_status": self.exclusion_status,
            "parse_tier": self.parse_tier,
            "review_reasons": list(self.review_reasons),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class _LayoutEvent:
    kind: Literal["blank", "text", "heading"]
    text: str
    source_start: int
    source_end: int
    list_marked: bool = False
    list_depth: int = 0
    list_marker: str | None = None
    list_marker_type: ListMarkerType | None = None
    list_marker_ordinal: int | None = None
    list_indent: str = ""
    relative_indent: int = 0
    polarity: CriterionPolarity = "unspecified"


@dataclass(frozen=True, slots=True)
class _HeadingMatch:
    polarity: CriterionPolarity
    tail_start: int


def _source_lines(text: str) -> tuple[tuple[str, int, int], ...]:
    lines: list[tuple[str, int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        lines.append((content, offset, offset + len(content)))
        offset += len(line)
    if not lines or offset < len(text):
        lines.append((text[offset:], offset, len(text)))
    return tuple(lines)


def _leading_whitespace(value: str) -> int:
    return len(value) - len(value.lstrip(" \t"))


def _list_marker_metadata(marker: str) -> tuple[ListMarkerType, int | None]:
    if marker in {"-", "*", "•", "·", "▪", "‣"}:
        return "bullet", None
    token = marker.strip("[]().").casefold()
    if token.isdigit():
        return "number", int(token)
    if len(token) == 1 and token.isalpha():
        return "letter", ord(token) - ord("a") + 1
    raise SchemaValidationError("eligibility list marker is unsupported")


def _child_relationship(text: str) -> tuple[CriterionChildRelationship, int | None]:
    if _ALL_OF_CHILDREN.search(text) is not None:
        return "all_of", None
    if _ANY_OF_CHILDREN.search(text) is not None:
        return "any_of", None
    match = _AT_LEAST_N_CHILDREN.search(text)
    if match is None:
        return "unknown", None
    count_token = match.group("count").casefold()
    count = int(count_token) if count_token.isdigit() else _COUNT_WORDS[count_token]
    return "at_least_n", count


def _trimmed_text_event(
    value: str,
    source_start: int,
    *,
    list_marked: bool,
    list_depth: int,
    list_marker: str | None,
    list_marker_type: ListMarkerType | None,
    list_marker_ordinal: int | None,
    list_indent: str,
    relative_indent: int,
) -> _LayoutEvent | None:
    leading = len(value) - len(value.lstrip())
    trailing = len(value) - len(value.rstrip())
    text = value.strip()
    if not text:
        return None
    return _LayoutEvent(
        kind="text",
        text=text,
        source_start=source_start + leading,
        source_end=source_start + len(value) - trailing,
        list_marked=list_marked,
        list_depth=list_depth,
        list_marker=list_marker,
        list_marker_type=list_marker_type,
        list_marker_ordinal=list_marker_ordinal,
        list_indent=list_indent,
        relative_indent=relative_indent,
    )


def _layout_events(text: str) -> tuple[_LayoutEvent, ...]:
    lines = _source_lines(text)
    nonempty_indents = [_leading_whitespace(line) for line, _start, _end in lines if line.strip()]
    common_margin = min(nonempty_indents, default=0)
    events: list[_LayoutEvent] = []
    list_indents: list[int] = []
    for raw_line, start, _end in lines:
        if not raw_line.strip():
            events.append(_LayoutEvent("blank", "", start, start))
            continue
        leading = _leading_whitespace(raw_line)
        relative_indent = max(0, leading - common_margin)
        marker = _LIST_MARKER.match(raw_line)
        content_start = marker.end() if marker is not None else leading
        content = raw_line[content_start:]
        list_marked = marker is not None
        list_marker = marker.group("marker") if marker is not None else None
        if not list_marked:
            heading = _find_heading(content.strip(), allow_compact=False)
            if heading is not None and heading[0] == 0:
                list_indents = []
        marker_type, marker_ordinal = (
            _list_marker_metadata(list_marker) if list_marker is not None else (None, None)
        )
        if list_marked:
            while list_indents and leading < list_indents[-1]:
                list_indents.pop()
            if not list_indents or leading > list_indents[-1]:
                list_indents.append(leading)
            list_depth = len(list_indents)
        else:
            list_depth = sum(indent < leading for indent in list_indents)
        list_indent = raw_line[:leading]
        cursor = 0
        inline_markers = tuple(_INLINE_LIST_MARKER.finditer(content))
        for inline in inline_markers:
            event = _trimmed_text_event(
                content[cursor : inline.start("gap")],
                start + content_start + cursor,
                list_marked=list_marked if cursor == 0 else True,
                list_depth=list_depth if cursor == 0 else max(1, list_depth + 1),
                list_marker=list_marker if cursor == 0 else inline.group("marker"),
                list_marker_type=marker_type if cursor == 0 else "bullet",
                list_marker_ordinal=marker_ordinal if cursor == 0 else None,
                list_indent=list_indent if cursor == 0 else inline.group("gap"),
                relative_indent=relative_indent,
            )
            if event is not None:
                events.append(event)
            cursor = inline.end()
        event = _trimmed_text_event(
            content[cursor:],
            start + content_start + cursor,
            list_marked=list_marked if cursor == 0 else True,
            list_depth=list_depth if cursor == 0 else max(1, list_depth + 1),
            list_marker=list_marker
            if cursor == 0
            else inline_markers[-1].group("marker")
            if inline_markers
            else None,
            list_marker_type=marker_type if cursor == 0 else "bullet",
            list_marker_ordinal=marker_ordinal if cursor == 0 else None,
            list_indent=list_indent
            if cursor == 0
            else inline_markers[-1].group("gap")
            if inline_markers
            else list_indent,
            relative_indent=relative_indent,
        )
        if event is not None:
            events.append(event)
    return tuple(events)


def _edit_distance_at_most(left: str, right: str, limit: int) -> bool:
    if abs(len(left) - len(right)) > limit:
        return False
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


def _is_heading_noun(value: str) -> bool:
    normalized = value.casefold().replace("-", "")
    expected = ("criteria", "criterion", "requirement", "requirements", "condition", "conditions")
    leading = value.casefold().split("-", 1)[0]
    return (
        normalized in expected
        or leading in expected
        or (
            len(normalized) >= 7
            and any(_edit_distance_at_most(normalized, candidate, 2) for candidate in expected)
        )
    )


def _canonical_heading_kind(value: str) -> CriterionPolarity:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    ).replace("_", "-")
    if "exclusion" in normalized or normalized.startswith("non"):
        return "exclusion"
    if normalized in {"included", "inclusion", "enrolled"}:
        return "inclusion"
    if normalized == "excluded":
        return "exclusion"
    raise SchemaValidationError("eligibility heading polarity is unsupported")


def _valid_qualifiers(value: str) -> bool:
    if any(character in value for character in ".;:!?"):
        return False
    words = {word.casefold().strip("./-") for word in value.split()}
    return not words.intersection(_QUALIFIER_STOP_WORDS)


def _heading_remainder(value: str, offset: int) -> int | None:
    leading = len(value) - len(value.lstrip())
    remainder = value[leading:]
    cursor = offset + leading
    if remainder[:1] in {">", "]", ")"}:
        remainder = remainder[1:]
        cursor += 1
        whitespace = len(remainder) - len(remainder.lstrip())
        remainder = remainder[whitespace:]
        cursor += whitespace
    if not remainder:
        return cursor
    if re.fullmatch(r"\.\s*", remainder) is not None:
        return offset + len(value)
    scope = re.match(r"\([^\r\n)]{1,100}\)\s*", remainder)
    if scope is not None:
        remainder = remainder[scope.end() :]
        cursor += scope.end()
        if not remainder:
            return cursor
    separator = re.match(r"[:;\uff1a\ufe55\-\u2013\u2014]+[ \t]*", remainder)
    if separator is not None:
        tail = remainder[separator.end() :]
        if _EMPTY_HEADING_TAIL.fullmatch(tail.strip()) is not None:
            return offset + len(value)
        return cursor + separator.end() + len(tail) - len(tail.lstrip())
    if _HEADING_POINTER.fullmatch(remainder) is not None:
        return offset + len(value)
    for pattern in (_HEADING_VERB, _SCOPED_HEADING_VERB):
        match = pattern.fullmatch(remainder)
        if match is not None:
            tail = match.group("tail")
            if _EMPTY_HEADING_TAIL.fullmatch(tail.strip()) is not None:
                return offset + len(value)
            return cursor + match.start("tail") + len(tail) - len(tail.lstrip())
    if _HEADING_SCOPE.fullmatch(remainder) is not None:
        return offset + len(value)
    return None


def _declarative_heading(value: str) -> _HeadingMatch | None:
    match = _SUBJECT_DECLARATION.fullmatch(value)
    if match is None:
        return None
    rest = match.group("rest")
    leading = len(rest) - len(rest.lstrip())
    tail = rest[leading:]
    tail_start = match.start("rest") + leading
    conditional = re.match(r"(?:if|when)\b[ \t]*", tail, re.IGNORECASE)
    if conditional is not None:
        tail = tail[conditional.end() :]
        tail_start += conditional.end()
    if tail.endswith(":") and re.fullmatch(
        r"(?:they\s+)?(?:have|meet|satisfy)(?:\s+any\s+of\s+the\s+following)?\s*:",
        tail,
        re.IGNORECASE,
    ):
        tail_start = len(value)
    elif tail.startswith((":", ";", "\uff1a", "\ufe55")):
        tail_start += 1 + len(tail[1:]) - len(tail[1:].lstrip())
    kind = match.group("kind") or match.group("enrolled")
    return _HeadingMatch(_canonical_heading_kind(kind), tail_start)


def _consume_heading(value: str) -> _HeadingMatch | None:
    declarative = _declarative_heading(value)
    if declarative is not None:
        return declarative
    combined = _COMBINED_HEADING.fullmatch(value)
    if combined is not None:
        noun = combined.group("noun")
        if noun is None or _is_heading_noun(noun):
            tail_start = _heading_remainder(combined.group("rest"), combined.start("rest"))
            if tail_start is not None:
                return _HeadingMatch("unspecified", tail_start)
    generic = _GENERIC_HEADING.fullmatch(value)
    if generic is not None:
        noun = generic.group("noun")
        if noun is None or _is_heading_noun(noun):
            tail_start = _heading_remainder(generic.group("rest"), generic.start("rest"))
            if tail_start is not None:
                return _HeadingMatch("unspecified", tail_start)
    for pattern in (_HEADING_CORE, _REVERSED_HEADING):
        match = pattern.fullmatch(value)
        if match is None:
            continue
        if not _is_heading_noun(match.group("noun")) or not _valid_qualifiers(
            match.group("qualifiers")
        ):
            continue
        tail_start = _heading_remainder(match.group("rest"), match.start("rest"))
        if tail_start is not None:
            return _HeadingMatch(_canonical_heading_kind(match.group("kind")), tail_start)
    bare = _BARE_HEADING.fullmatch(value)
    if bare is not None:
        tail_start = _heading_remainder(bare.group("rest"), bare.start("rest"))
        if tail_start is not None:
            return _HeadingMatch(_canonical_heading_kind(bare.group("kind")), tail_start)
    return None


def _find_heading(value: str, *, allow_compact: bool) -> tuple[int, _HeadingMatch] | None:
    candidates = {0}
    punctuation_starts = {match.end() for match in re.finditer(r"[.;][ \t]+", value)}
    candidates.update(punctuation_starts)
    if allow_compact:
        candidates.update(match.start() for match in _POLARITY_CANDIDATE.finditer(value))
    for start in sorted(candidates, key=lambda candidate: (candidate == 0, candidate)):
        prefix = value[:start].strip()
        if start and value[start - 1 : start] not in {" ", "\t"}:
            continue
        if (
            prefix
            and start not in punctuation_starts
            and (not allow_compact or len(prefix) > 100 or not _valid_qualifiers(prefix))
        ):
            continue
        heading = _consume_heading(value[start:])
        if heading is not None:
            compact_value = value[start:].strip().casefold()
            if allow_compact and compact_value in {
                "inclusion",
                "inclusions",
                "exclusion",
                "exclusions",
            }:
                continue
            return start, heading
    return None


def _expand_headings(events: Sequence[_LayoutEvent]) -> tuple[_LayoutEvent, ...]:
    expanded: list[_LayoutEvent] = []
    active_list_indent: int | None = None
    for event in events:
        if event.kind != "text":
            expanded.append(event)
            active_list_indent = None
            continue
        if event.list_marked:
            active_list_indent = event.relative_indent
        remaining = event.text
        source_start = event.source_start
        allow_compact = False
        if (
            active_list_indent is not None
            and not event.list_marked
            and event.relative_indent > active_list_indent
            and re.search(r"[:;\uff1a\ufe55\u2013\u2014]\s*$", remaining) is None
        ):
            expanded.append(event)
            continue
        if not event.list_marked:
            active_list_indent = None
        if event.list_marked and re.fullmatch(
            r"non[-\s]+(?:inclusion|exclusion)\s+criteria",
            remaining,
            re.IGNORECASE,
        ):
            expanded.append(event)
            continue
        if (
            event.list_marked
            and re.search(
                r"\b(?:inclusion|exclusion)\s+(?:criteria|criterion|requirements?|conditions?)"
                r"\s+\S",
                remaining,
                re.IGNORECASE,
            )
            and re.search(r"[:;\uff1a\ufe55\u2013\u2014]\s*$", remaining) is None
        ):
            expanded.append(event)
            continue
        while remaining:
            found = _find_heading(remaining, allow_compact=allow_compact)
            if found is None:
                text_event = _trimmed_text_event(
                    remaining,
                    source_start,
                    list_marked=event.list_marked and not allow_compact,
                    list_depth=event.list_depth,
                    list_marker=event.list_marker if not allow_compact else None,
                    list_marker_type=event.list_marker_type if not allow_compact else None,
                    list_marker_ordinal=event.list_marker_ordinal if not allow_compact else None,
                    list_indent=event.list_indent if not allow_compact else "",
                    relative_indent=event.relative_indent,
                )
                if text_event is not None:
                    expanded.append(text_event)
                break
            heading_start, heading = found
            prefix = _trimmed_text_event(
                remaining[:heading_start],
                source_start,
                list_marked=event.list_marked and not allow_compact,
                list_depth=event.list_depth,
                list_marker=event.list_marker if not allow_compact else None,
                list_marker_type=event.list_marker_type if not allow_compact else None,
                list_marker_ordinal=event.list_marker_ordinal if not allow_compact else None,
                list_indent=event.list_indent if not allow_compact else "",
                relative_indent=event.relative_indent,
            )
            if prefix is not None:
                expanded.append(prefix)
            expanded.append(
                _LayoutEvent(
                    "heading",
                    "",
                    source_start + heading_start,
                    source_start + heading_start + heading.tail_start,
                    polarity=heading.polarity,
                )
            )
            consumed = heading_start + heading.tail_start
            if consumed >= len(remaining):
                break
            source_start += consumed
            remaining = remaining[consumed:]
            allow_compact = True
    return tuple(expanded)


def _next_text_event(events: Sequence[_LayoutEvent], index: int) -> _LayoutEvent | None:
    for candidate in events[index + 1 :]:
        if candidate.kind == "heading":
            return None
        if candidate.kind == "text":
            return candidate
    return None


def _context_item_type(
    event: _LayoutEvent,
    next_event: _LayoutEvent | None,
) -> EligibilityItemType | None:
    if _BOOLEAN_CONTEXT.fullmatch(event.text) is not None:
        return "context"
    if _LEGACY_CONTEXT.fullmatch(event.text.casefold()) is not None:
        return "context"
    colon_header = _GROUP_HEADER.fullmatch(event.text) is not None
    uppercase_header = (
        len(event.text) <= 100
        and any(character.isalpha() for character in event.text)
        and event.text.upper() == event.text
        and _MEASUREMENT_COMPARISON.search(event.text) is None
    )
    if colon_header:
        if event.list_marked:
            return None
        return "group_header" if next_event is not None else "context"
    if next_event is not None and uppercase_header:
        return "group_header" if next_event.list_marked else "context"
    return None


def _special_item(
    text: str,
    current_polarity: CriterionPolarity,
) -> tuple[EligibilityItemType, CriterionPolarity, frozenset[CriterionPolarity]] | None:
    normalized = text.strip()
    declared = _DECLARED_EMPTY.fullmatch(normalized)
    unavailable = _UNAVAILABLE.fullmatch(normalized) or _OTHER_CRITERIA_UNAVAILABLE.fullmatch(
        normalized
    )
    if declared is None and unavailable is None:
        return None
    mentioned: CriterionPolarity = current_polarity
    casefolded = normalized.casefold()
    if "exclusion" in casefolded:
        mentioned = "exclusion"
    elif "inclusion" in casefolded:
        mentioned = "inclusion"
    both = ("eligibility" in casefolded and "criteria" in casefolded) or (
        "inclusion" in casefolded and "exclusion" in casefolded
    )
    polarities: frozenset[CriterionPolarity] = (
        frozenset({"inclusion", "exclusion"})
        if both
        else frozenset({mentioned})
        if mentioned != "unspecified"
        else frozenset()
    )
    return (
        "declared_empty" if declared is not None else "unavailable",
        mentioned,
        polarities,
    )


def analyze_complete_eligibility(text: str) -> EligibilityCriterionParse:
    """Parse source-grounded eligibility layout into typed, offset-preserving items."""

    events = _expand_headings(_layout_events(text))
    first_heading_polarity = next(
        (event.polarity for event in events if event.kind == "heading"),
        None,
    )
    seen_heading = False
    current_polarity: CriterionPolarity = "unspecified"
    current_parts: list[str] = []
    current_start = 0
    current_end = 0
    current_is_list_item = False
    current_list_depth = 0
    current_list_marker: str | None = None
    current_list_marker_type: ListMarkerType | None = None
    current_list_marker_ordinal: int | None = None
    current_list_indent = ""
    current_indent = 0
    current_group_path: tuple[str, ...] = ()
    active_groups: list[tuple[int, str]] = []
    active_parents: list[tuple[int, int, EligibilityParentType]] = []
    boundaries: list[EligibilityCriterionBoundary] = []
    other_items: list[EligibilityNonCriterionBoundary] = []
    explicit_polarities: set[CriterionPolarity] = set()
    declared_empty: set[CriterionPolarity] = set()
    unavailable: set[CriterionPolarity] = set()

    def finish() -> None:
        nonlocal current_is_list_item, current_parts
        criterion = "\n".join(part.strip() for part in current_parts if part.strip()).strip()
        if criterion:
            while active_parents and current_list_depth <= active_parents[-1][0]:
                active_parents.pop()
            parent = active_parents[-1] if active_parents else None
            parent_source_start = parent[1] if parent is not None else None
            parent_type = parent[2] if parent is not None else None
            child_relationship, required_child_count = _child_relationship(criterion)
            special = _special_item(criterion, current_polarity)
            if special is None:
                boundaries.append(
                    EligibilityCriterionBoundary(
                        text=criterion,
                        polarity=current_polarity,
                        source_start=current_start,
                        source_end=current_end,
                        list_marked=current_is_list_item,
                        list_depth=current_list_depth,
                        list_marker=current_list_marker,
                        list_marker_type=current_list_marker_type,
                        list_marker_ordinal=current_list_marker_ordinal,
                        list_indent=current_list_indent,
                        parent_source_start=parent_source_start,
                        parent_type=parent_type,
                        child_relationship=child_relationship,
                        required_child_count=required_child_count,
                        group_path=current_group_path,
                    )
                )
                if current_is_list_item:
                    active_parents.append((current_list_depth, current_start, "criterion"))
            else:
                item_type, polarity, affected = special
                other_items.append(
                    EligibilityNonCriterionBoundary(
                        criterion,
                        item_type,
                        polarity,
                        current_start,
                        current_end,
                        current_is_list_item,
                        current_list_depth,
                        current_list_marker,
                        current_list_marker_type,
                        current_list_marker_ordinal,
                        current_list_indent,
                        parent_source_start,
                        parent_type,
                        child_relationship,
                        required_child_count,
                    )
                )
                target = declared_empty if item_type == "declared_empty" else unavailable
                target.update(affected)
        current_parts = []
        current_is_list_item = False

    for index, event in enumerate(events):
        if event.kind == "blank":
            finish()
            continue
        if event.kind == "heading":
            finish()
            seen_heading = True
            current_polarity = event.polarity
            if current_polarity != "unspecified":
                explicit_polarities.add(current_polarity)
            active_groups = []
            active_parents = []
            continue
        if event.list_marked:
            finish()
        next_event = _next_text_event(events, index)
        candidate_context_type = _context_item_type(event, next_event)
        starts_like_continuation = event.text[:1].islower() or event.text.startswith(
            (",", ";", ":", ")", "]")
        )
        standalone_parenthetical_continuation = (
            re.fullmatch(r"\([A-Z][A-Z0-9/-]{1,20}\)[.,;:]?", event.text) is not None
        )
        previous_is_open = (
            bool(current_parts)
            and re.search(
                r"(?:[,;:]|\b(?:and|or|with|including|following))\s*$",
                current_parts[-1],
                re.IGNORECASE,
            )
            is not None
        )
        continues = bool(current_parts) and (
            starts_like_continuation
            or standalone_parenthetical_continuation
            or (current_is_list_item and previous_is_open)
            or (event.relative_indent > current_indent and candidate_context_type is None)
        )
        context_type = None if continues else candidate_context_type
        if (
            context_type is None
            and not continues
            and not seen_heading
            and first_heading_polarity == "inclusion"
        ):
            context_type = "context"
        if context_type is not None:
            finish()
            while active_parents and event.list_depth <= active_parents[-1][0]:
                active_parents.pop()
            parent = active_parents[-1] if active_parents else None
            child_relationship, required_child_count = _child_relationship(event.text)
            other_items.append(
                EligibilityNonCriterionBoundary(
                    event.text,
                    context_type,
                    current_polarity,
                    event.source_start,
                    event.source_end,
                    event.list_marked,
                    event.list_depth,
                    event.list_marker,
                    event.list_marker_type,
                    event.list_marker_ordinal,
                    event.list_indent,
                    parent[1] if parent is not None else None,
                    parent[2] if parent is not None else None,
                    child_relationship,
                    required_child_count,
                )
            )
            active_groups = [group for group in active_groups if group[0] < event.list_depth]
            if context_type == "group_header":
                active_groups.append((event.list_depth, event.text))
                active_parents.append((event.list_depth, event.source_start, "group_header"))
            continue
        if current_parts and not continues:
            finish()
        if not current_parts:
            active_groups = [group for group in active_groups if group[0] < event.list_depth]
            current_parts = [event.text]
            current_start = event.source_start
            current_end = event.source_end
            current_is_list_item = event.list_marked
            current_list_depth = event.list_depth
            current_list_marker = event.list_marker
            current_list_marker_type = event.list_marker_type
            current_list_marker_ordinal = event.list_marker_ordinal
            current_list_indent = event.list_indent
            current_indent = event.relative_indent
            current_group_path = tuple(group[1] for group in active_groups)
        else:
            current_parts.append(event.text)
            current_end = event.source_end
    finish()
    criteria = tuple(boundaries)
    explicit = frozenset(explicit_polarities)

    def polarity_status(polarity: Literal["inclusion", "exclusion"]) -> EligibilityPolarityStatus:
        if any(item.polarity == polarity for item in criteria):
            return "criteria"
        if polarity in declared_empty:
            return "declared_empty"
        if polarity in unavailable:
            return "unavailable"
        return "not_identified"

    inclusion_status = polarity_status("inclusion")
    exclusion_status = polarity_status("exclusion")
    has_unspecified = any(item.polarity == "unspecified" for item in criteria)
    has_both_headings = explicit == {"inclusion", "exclusion"}
    resolved_both = inclusion_status != "not_identified" and exclusion_status != "not_identified"
    if not criteria and not resolved_both:
        parse_tier: EligibilityParseTier = "empty"
    elif has_unspecified or not resolved_both:
        parse_tier = "incomplete_polarity"
    elif not has_both_headings:
        parse_tier = "implicit_polarity"
    elif criteria and all(item.list_marked for item in criteria):
        parse_tier = "explicit_list"
    else:
        parse_tier = "explicit_prose"

    review_reasons: list[EligibilityReviewReason] = []
    if inclusion_status == "not_identified":
        review_reasons.append("missing_inclusion_criteria")
    if exclusion_status == "not_identified":
        review_reasons.append("missing_exclusion_criteria")
    if any(item.polarity == "inclusion" for item in criteria) and "inclusion" not in explicit:
        review_reasons.append("implicit_inclusion_polarity")
    if any(item.polarity == "exclusion" for item in criteria) and "exclusion" not in explicit:
        review_reasons.append("implicit_exclusion_polarity")
    if criteria and any(not item.list_marked for item in criteria):
        review_reasons.append("unmarked_criterion_boundaries")
    if has_unspecified:
        review_reasons.append("unspecified_polarity")
    if other_items:
        review_reasons.append("non_criterion_items")
    for polarity, status in (
        ("inclusion", inclusion_status),
        ("exclusion", exclusion_status),
    ):
        if status in {"declared_empty", "unavailable"}:
            review_reasons.append(cast(EligibilityReviewReason, f"{status}_{polarity}"))
    return EligibilityCriterionParse(
        criteria=criteria,
        other_items=tuple(other_items),
        explicit_polarities=explicit,
        inclusion_status=inclusion_status,
        exclusion_status=exclusion_status,
        parse_tier=parse_tier,
        review_reasons=tuple(review_reasons),
    )


def split_complete_eligibility(text: str) -> tuple[EligibilityCriterionBoundary, ...]:
    """Return deterministic boundaries for callers that do not need review metadata."""

    return analyze_complete_eligibility(text).criteria


def derive_complete_eligibility_section(
    section: SemanticTextSection,
    *,
    source_text: str,
) -> SemanticTextSection:
    """Attach complete source text without changing source-grounded criterion structure."""

    if section.role != "eligibility":
        raise ValueError("complete eligibility derivation requires an eligibility section")
    source_bytes = source_text.encode("utf-8")
    source_sha256 = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
    return replace(
        section,
        source_text=source_text,
        source_text_sha256=source_sha256,
        source_text_byte_length=len(source_bytes),
    )


def _criterion_view_item(
    criterion: CriterionItem,
    boundary: EligibilityCriterionBoundary,
    *,
    source_text: str,
    parent_identifier: str | None,
) -> EligibilityCriterionViewItem:
    identifier = criterion.identifier
    if identifier is None:
        raise SchemaValidationError("derived eligibility criterion identifier is missing")
    source_span_text = source_text[boundary.source_start : boundary.source_end]
    return EligibilityCriterionViewItem(
        identifier=identifier,
        ordinal=criterion.ordinal,
        text=criterion.text,
        polarity=boundary.polarity,
        source_start=boundary.source_start,
        source_end=boundary.source_end,
        source_span_text=source_span_text,
        source_span_sha256=(
            "sha256:" + hashlib.sha256(source_span_text.encode("utf-8")).hexdigest()
        ),
        list_marked=boundary.list_marked,
        list_depth=boundary.list_depth,
        list_marker=boundary.list_marker,
        list_marker_type=boundary.list_marker_type,
        list_marker_ordinal=boundary.list_marker_ordinal,
        list_indent=boundary.list_indent,
        parent_identifier=parent_identifier,
        parent_source_start=boundary.parent_source_start,
        parent_type=boundary.parent_type,
        child_relationship=boundary.child_relationship,
        required_child_count=boundary.required_child_count,
        group_path=boundary.group_path,
        derived_text_sha256=cast(str, criterion.additional_fields["derived_text_sha256"]),
    )


def _noncriterion_identifier(
    boundary: EligibilityNonCriterionBoundary,
    *,
    source_text: str,
    source_text_sha256: str,
) -> str:
    source_span_text = source_text[boundary.source_start : boundary.source_end]
    source_span_sha256 = "sha256:" + hashlib.sha256(source_span_text.encode("utf-8")).hexdigest()
    return content_sha256(
        {
            "version": ELIGIBILITY_SPLIT_VERSION,
            "source_text_sha256": source_text_sha256,
            "item_type": boundary.item_type,
            "polarity": boundary.polarity,
            "source_start": boundary.source_start,
            "source_end": boundary.source_end,
            "source_span_sha256": source_span_sha256,
        }
    )


def _noncriterion_view_item(
    ordinal: int,
    boundary: EligibilityNonCriterionBoundary,
    *,
    source_text: str,
    source_text_sha256: str,
    identifier: str,
    parent_identifier: str | None,
) -> EligibilityNonCriterionViewItem:
    source_span_text = source_text[boundary.source_start : boundary.source_end]
    source_span_sha256 = "sha256:" + hashlib.sha256(source_span_text.encode("utf-8")).hexdigest()
    return EligibilityNonCriterionViewItem(
        identifier=identifier,
        ordinal=ordinal,
        text=boundary.text,
        item_type=boundary.item_type,
        polarity=boundary.polarity,
        source_start=boundary.source_start,
        source_end=boundary.source_end,
        source_span_text=source_span_text,
        source_span_sha256=source_span_sha256,
        list_marked=boundary.list_marked,
        list_depth=boundary.list_depth,
        list_marker=boundary.list_marker,
        list_marker_type=boundary.list_marker_type,
        list_marker_ordinal=boundary.list_marker_ordinal,
        list_indent=boundary.list_indent,
        parent_identifier=parent_identifier,
        parent_source_start=boundary.parent_source_start,
        parent_type=boundary.parent_type,
        child_relationship=boundary.child_relationship,
        required_child_count=boundary.required_child_count,
    )


def _trial_view_payload(
    items: Sequence[EligibilityCriterionViewItem],
    other_items: Sequence[EligibilityNonCriterionViewItem],
    *,
    inclusion_status: EligibilityPolarityStatus,
    exclusion_status: EligibilityPolarityStatus,
) -> dict[str, JsonValue]:
    return {
        "criteria": [item.to_dict() for item in items],
        "other_items": [item.to_dict() for item in other_items],
        "inclusion_status": inclusion_status,
        "exclusion_status": exclusion_status,
    }


def _criterion_inventory(
    record_items: Sequence[EligibilityCriterionViewItem],
    *,
    source_text_sha256: str,
    source_provenance_sha256: str,
) -> list[JsonValue]:
    return [
        {
            "identifier": item.identifier,
            "ordinal": item.ordinal,
            "polarity": item.polarity,
            "origin": "derived",
            "source_start": item.source_start,
            "source_end": item.source_end,
            "source_span_sha256": item.source_span_sha256,
            "derived_text_sha256": item.derived_text_sha256,
            "source_text_sha256": source_text_sha256,
            "provenance_sha256": source_provenance_sha256,
        }
        for item in record_items
    ]


def _build_trial_record(
    trial_id: str,
    section: SemanticTextSection,
) -> EligibilityCriterionTrialRecord:
    source_text = section.source_text
    source_text_sha256 = section.source_text_sha256
    if source_text is None or source_text_sha256 is None:
        raise SchemaValidationError(
            f"trial {trial_id} eligibility criterion view requires complete source text"
        )
    parsed = analyze_complete_eligibility(source_text)
    criteria = _derive_eligibility_criteria(
        section,
        source_text=source_text,
        boundaries=parsed.criteria,
    )
    noncriterion_identifiers = tuple(
        _noncriterion_identifier(
            boundary,
            source_text=source_text,
            source_text_sha256=source_text_sha256,
        )
        for boundary in parsed.other_items
    )
    node_identifiers: dict[int, str] = {}
    for boundary, criterion in zip(parsed.criteria, criteria, strict=True):
        if criterion.identifier is None:
            raise SchemaValidationError("derived eligibility criterion identifier is missing")
        node_identifiers[boundary.source_start] = criterion.identifier
    for noncriterion_boundary, identifier in zip(
        parsed.other_items,
        noncriterion_identifiers,
        strict=True,
    ):
        if noncriterion_boundary.source_start in node_identifiers:
            raise SchemaValidationError("eligibility structural source starts must be unique")
        node_identifiers[noncriterion_boundary.source_start] = identifier
    items = tuple(
        _criterion_view_item(
            criterion,
            boundary,
            source_text=source_text,
            parent_identifier=(
                node_identifiers[boundary.parent_source_start]
                if boundary.parent_source_start is not None
                else None
            ),
        )
        for criterion, boundary in zip(criteria, parsed.criteria, strict=True)
    )
    other_items = tuple(
        _noncriterion_view_item(
            ordinal,
            boundary,
            source_text=source_text,
            source_text_sha256=source_text_sha256,
            identifier=identifier,
            parent_identifier=(
                node_identifiers[boundary.parent_source_start]
                if boundary.parent_source_start is not None
                else None
            ),
        )
        for ordinal, (boundary, identifier) in enumerate(
            zip(parsed.other_items, noncriterion_identifiers, strict=True)
        )
    )
    source_provenance_sha256 = content_sha256([item.to_dict() for item in section.provenance])
    inventory = _criterion_inventory(
        items,
        source_text_sha256=source_text_sha256,
        source_provenance_sha256=source_provenance_sha256,
    )
    return EligibilityCriterionTrialRecord(
        trial_id=trial_id,
        source_text_sha256=source_text_sha256,
        source_provenance_sha256=source_provenance_sha256,
        view_sha256=content_sha256(
            _trial_view_payload(
                items,
                other_items,
                inclusion_status=parsed.inclusion_status,
                exclusion_status=parsed.exclusion_status,
            )
        ),
        criterion_inventory_sha256=content_sha256(inventory),
        parse_tier=parsed.parse_tier,
        review_reasons=parsed.review_reasons,
        inclusion_status=parsed.inclusion_status,
        exclusion_status=parsed.exclusion_status,
        inclusion=tuple(item for item in items if item.polarity == "inclusion"),
        exclusion=tuple(item for item in items if item.polarity == "exclusion"),
        unspecified=tuple(item for item in items if item.polarity == "unspecified"),
        other_items=other_items,
    )


def build_eligibility_criterion_trial_record(
    trial_id: str,
    sections: Sequence[SemanticTextSection],
) -> EligibilityCriterionTrialRecord | None:
    """Build one record without requiring the full Snapshot in memory."""

    eligibility_sections = tuple(
        section
        for section in sections
        if section.role == "eligibility" and section.source_text is not None
    )
    if len(eligibility_sections) > 1:
        raise SchemaValidationError(f"trial {trial_id} has multiple complete eligibility sections")
    if not eligibility_sections:
        return None
    return _build_trial_record(trial_id, eligibility_sections[0])


def build_eligibility_criterion_view(snapshot: BenchmarkSnapshot) -> DerivedView:
    """Build the supplied criterion segmentation without mutating source-grounded trials."""

    if not isinstance(snapshot, BenchmarkSnapshot):
        raise TypeError("snapshot must be a BenchmarkSnapshot")
    trial_records: dict[str, JsonValue] = {}
    for trial in snapshot.trials:
        record = build_eligibility_criterion_trial_record(trial.trial_id, trial.sections)
        if record is not None:
            trial_records[trial.trial_id] = record.to_dict()
    return DerivedView(
        name=ELIGIBILITY_CRITERION_VIEW_NAME,
        version=ELIGIBILITY_SPLIT_VERSION,
        input_snapshot_id=snapshot.base_snapshot_id,
        configuration=ELIGIBILITY_VIEW_CONFIGURATION,
        content={
            "schema_version": ELIGIBILITY_CRITERION_VIEW_CONTENT_SCHEMA,
            "trials": trial_records,
        },
    )


def _view_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _validated_list_structure(
    item: Mapping[str, object],
    *,
    field_name: str,
) -> tuple[bool, int, str | None, ListMarkerType | None, int | None, str]:
    list_marked = item.get("list_marked")
    list_depth = item.get("list_depth")
    list_marker = item.get("list_marker")
    list_marker_type = item.get("list_marker_type")
    list_marker_ordinal = item.get("list_marker_ordinal")
    list_indent = item.get("list_indent")
    if not isinstance(list_marked, bool):
        raise SchemaValidationError(f"{field_name} list marker flag is invalid")
    if isinstance(list_depth, bool) or not isinstance(list_depth, int) or list_depth < 0:
        raise SchemaValidationError(f"{field_name} list depth is invalid")
    if not isinstance(list_indent, str) or list_indent.strip(" \t"):
        raise SchemaValidationError(f"{field_name} list indentation is invalid")
    if not list_marked:
        if (
            list_marker is not None
            or list_marker_type is not None
            or list_marker_ordinal is not None
        ):
            raise SchemaValidationError(f"{field_name} unmarked list metadata is invalid")
        return False, list_depth, None, None, None, list_indent
    if not isinstance(list_marker, str) or not list_marker:
        raise SchemaValidationError(f"{field_name} raw list marker is invalid")
    expected_type, expected_ordinal = _list_marker_metadata(list_marker)
    if list_marker_type != expected_type or list_marker_ordinal != expected_ordinal:
        raise SchemaValidationError(f"{field_name} list marker metadata does not match")
    return (
        True,
        list_depth,
        list_marker,
        expected_type,
        expected_ordinal,
        list_indent,
    )


def _validated_relationship(
    item: Mapping[str, object],
    *,
    field_name: str,
) -> tuple[
    str | None,
    int | None,
    EligibilityParentType | None,
    CriterionChildRelationship,
    int | None,
]:
    parent_identifier = item.get("parent_identifier")
    parent_source_start = item.get("parent_source_start")
    parent_type = item.get("parent_type")
    child_relationship = item.get("child_relationship")
    required_child_count = item.get("required_child_count")
    if parent_identifier is None:
        if parent_source_start is not None or parent_type is not None:
            raise SchemaValidationError(f"{field_name} parent metadata is incomplete")
    else:
        require_sha256(parent_identifier, f"{field_name} parent identifier")
        if (
            isinstance(parent_source_start, bool)
            or not isinstance(parent_source_start, int)
            or parent_source_start < 0
        ):
            raise SchemaValidationError(f"{field_name} parent source start is invalid")
        if parent_type not in {"criterion", "group_header"}:
            raise SchemaValidationError(f"{field_name} parent type is invalid")
    if child_relationship not in {"all_of", "any_of", "at_least_n", "unknown"}:
        raise SchemaValidationError(f"{field_name} child relationship is invalid")
    if child_relationship == "at_least_n":
        if (
            isinstance(required_child_count, bool)
            or not isinstance(required_child_count, int)
            or required_child_count < 1
        ):
            raise SchemaValidationError(f"{field_name} required child count is invalid")
    elif required_child_count is not None:
        raise SchemaValidationError(f"{field_name} unexpected required child count")
    return (
        cast(str | None, parent_identifier),
        cast(int | None, parent_source_start),
        cast(EligibilityParentType | None, parent_type),
        cast(CriterionChildRelationship, child_relationship),
        cast(int | None, required_child_count),
    )


def _view_item(
    value: object,
    *,
    expected_polarity: CriterionPolarity,
) -> EligibilityCriterionViewItem:
    item = _view_mapping(value, field_name="eligibility criterion")
    identifier = item.get("identifier")
    ordinal = item.get("ordinal")
    text = item.get("text")
    polarity = item.get("polarity")
    source_start = item.get("source_start")
    source_end = item.get("source_end")
    source_span_text = item.get("source_span_text")
    source_span_sha256 = item.get("source_span_sha256")
    group_path = item.get("group_path")
    derived_text_sha256 = item.get("derived_text_sha256")
    require_sha256(identifier, "eligibility criterion identifier")
    require_sha256(derived_text_sha256, "eligibility criterion text SHA-256")
    require_sha256(source_span_sha256, "eligibility criterion source span SHA-256")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise SchemaValidationError("eligibility criterion ordinal is invalid")
    if not isinstance(text, str) or not text:
        raise SchemaValidationError("eligibility criterion text is invalid")
    if polarity != expected_polarity:
        raise SchemaValidationError("eligibility criterion polarity does not match its list")
    if (
        isinstance(source_start, bool)
        or not isinstance(source_start, int)
        or isinstance(source_end, bool)
        or not isinstance(source_end, int)
        or source_start < 0
        or source_end <= source_start
    ):
        raise SchemaValidationError("eligibility criterion source span is invalid")
    (
        list_marked,
        list_depth,
        list_marker,
        list_marker_type,
        list_marker_ordinal,
        list_indent,
    ) = _validated_list_structure(item, field_name="eligibility criterion")
    (
        parent_identifier,
        parent_source_start,
        parent_type,
        child_relationship,
        required_child_count,
    ) = _validated_relationship(item, field_name="eligibility criterion")
    if (
        not isinstance(group_path, Sequence)
        or isinstance(group_path, (str, bytes))
        or any(not isinstance(group, str) or not group for group in group_path)
    ):
        raise SchemaValidationError("eligibility criterion group path is invalid")
    if not isinstance(source_span_text, str) or not source_span_text:
        raise SchemaValidationError("eligibility criterion source span text is invalid")
    expected_source_span_sha256 = (
        "sha256:" + hashlib.sha256(source_span_text.encode("utf-8")).hexdigest()
    )
    if source_span_sha256 != expected_source_span_sha256:
        raise SchemaValidationError("eligibility criterion source span hash does not match")
    return EligibilityCriterionViewItem(
        identifier=cast(str, identifier),
        ordinal=ordinal,
        text=text,
        polarity=expected_polarity,
        source_start=source_start,
        source_end=source_end,
        source_span_text=source_span_text,
        source_span_sha256=cast(str, source_span_sha256),
        list_marked=list_marked,
        list_depth=list_depth,
        list_marker=list_marker,
        list_marker_type=list_marker_type,
        list_marker_ordinal=list_marker_ordinal,
        list_indent=list_indent,
        parent_identifier=parent_identifier,
        parent_source_start=parent_source_start,
        parent_type=parent_type,
        child_relationship=child_relationship,
        required_child_count=required_child_count,
        group_path=tuple(cast(Sequence[str], group_path)),
        derived_text_sha256=cast(str, derived_text_sha256),
    )


def _noncriterion_record_item(value: object) -> EligibilityNonCriterionViewItem:
    item = _view_mapping(value, field_name="eligibility non-criterion item")
    identifier = item.get("identifier")
    ordinal = item.get("ordinal")
    text = item.get("text")
    item_type = item.get("item_type")
    polarity = item.get("polarity")
    source_start = item.get("source_start")
    source_end = item.get("source_end")
    source_span_text = item.get("source_span_text")
    source_span_sha256 = item.get("source_span_sha256")
    require_sha256(identifier, "eligibility non-criterion identifier")
    require_sha256(source_span_sha256, "eligibility non-criterion source span SHA-256")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise SchemaValidationError("eligibility non-criterion ordinal is invalid")
    if not isinstance(text, str) or not text:
        raise SchemaValidationError("eligibility non-criterion text is invalid")
    if item_type not in {"group_header", "context", "declared_empty", "unavailable"}:
        raise SchemaValidationError("eligibility non-criterion item type is invalid")
    if polarity not in {"inclusion", "exclusion", "unspecified"}:
        raise SchemaValidationError("eligibility non-criterion polarity is invalid")
    if (
        isinstance(source_start, bool)
        or not isinstance(source_start, int)
        or isinstance(source_end, bool)
        or not isinstance(source_end, int)
        or source_start < 0
        or source_end <= source_start
    ):
        raise SchemaValidationError("eligibility non-criterion source span is invalid")
    if not isinstance(source_span_text, str) or not source_span_text:
        raise SchemaValidationError("eligibility non-criterion source span text is invalid")
    (
        list_marked,
        list_depth,
        list_marker,
        list_marker_type,
        list_marker_ordinal,
        list_indent,
    ) = _validated_list_structure(item, field_name="eligibility non-criterion")
    (
        parent_identifier,
        parent_source_start,
        parent_type,
        child_relationship,
        required_child_count,
    ) = _validated_relationship(item, field_name="eligibility non-criterion")
    expected_source_span_sha256 = (
        "sha256:" + hashlib.sha256(source_span_text.encode("utf-8")).hexdigest()
    )
    if source_span_sha256 != expected_source_span_sha256:
        raise SchemaValidationError("eligibility non-criterion source span hash does not match")
    return EligibilityNonCriterionViewItem(
        identifier=cast(str, identifier),
        ordinal=ordinal,
        text=text,
        item_type=cast(EligibilityItemType, item_type),
        polarity=cast(CriterionPolarity, polarity),
        source_start=source_start,
        source_end=source_end,
        source_span_text=source_span_text,
        source_span_sha256=cast(str, source_span_sha256),
        list_marked=list_marked,
        list_depth=list_depth,
        list_marker=list_marker,
        list_marker_type=list_marker_type,
        list_marker_ordinal=list_marker_ordinal,
        list_indent=list_indent,
        parent_identifier=parent_identifier,
        parent_source_start=parent_source_start,
        parent_type=parent_type,
        child_relationship=child_relationship,
        required_child_count=required_child_count,
    )


def eligibility_criterion_trial(
    view: DerivedView,
    trial_id: str,
) -> EligibilityCriterionTrialRecord | None:
    """Read and validate one trial record from the supplied eligibility criterion view."""

    if view.name != ELIGIBILITY_CRITERION_VIEW_NAME or view.version != ELIGIBILITY_SPLIT_VERSION:
        raise SchemaValidationError("eligibility criterion Derived View identity is unsupported")
    content = _view_mapping(view.content, field_name="eligibility criterion view content")
    if content.get("schema_version") != ELIGIBILITY_CRITERION_VIEW_CONTENT_SCHEMA:
        raise SchemaValidationError("eligibility criterion view content schema is unsupported")
    trials = _view_mapping(content.get("trials"), field_name="eligibility criterion view trials")
    raw_record = trials.get(trial_id)
    if raw_record is None:
        return None
    record = _view_mapping(raw_record, field_name=f"eligibility criterion trial {trial_id}")
    if record.get("trial_id") != trial_id:
        raise SchemaValidationError("eligibility criterion trial identity does not match")
    raw_inclusion = record.get("inclusion_criteria")
    raw_exclusion = record.get("exclusion_criteria")
    raw_unspecified = record.get("unspecified_criteria")
    raw_other_items = record.get("other_items")
    if not isinstance(raw_inclusion, Sequence) or isinstance(raw_inclusion, (str, bytes)):
        raise SchemaValidationError("eligibility inclusion criteria must be an array")
    if not isinstance(raw_exclusion, Sequence) or isinstance(raw_exclusion, (str, bytes)):
        raise SchemaValidationError("eligibility exclusion criteria must be an array")
    if not isinstance(raw_unspecified, Sequence) or isinstance(raw_unspecified, (str, bytes)):
        raise SchemaValidationError("eligibility unspecified criteria must be an array")
    if not isinstance(raw_other_items, Sequence) or isinstance(raw_other_items, (str, bytes)):
        raise SchemaValidationError("eligibility non-criterion items must be an array")
    inclusion = tuple(_view_item(item, expected_polarity="inclusion") for item in raw_inclusion)
    exclusion = tuple(_view_item(item, expected_polarity="exclusion") for item in raw_exclusion)
    unspecified = tuple(
        _view_item(item, expected_polarity="unspecified") for item in raw_unspecified
    )
    other_items = tuple(_noncriterion_record_item(item) for item in raw_other_items)
    items = tuple(sorted((*inclusion, *exclusion, *unspecified), key=lambda item: item.ordinal))
    if [item.ordinal for item in items] != list(range(len(items))):
        raise SchemaValidationError("eligibility criterion ordinals must be contiguous")
    structural_items: tuple[
        EligibilityCriterionViewItem | EligibilityNonCriterionViewItem,
        ...,
    ] = (*items, *other_items)
    by_identifier = {item.identifier: item for item in structural_items}
    if len(by_identifier) != len(structural_items):
        raise SchemaValidationError("eligibility structural identifiers must be unique")
    criterion_identifiers = {item.identifier for item in items}
    group_identifiers = {
        item.identifier for item in other_items if item.item_type == "group_header"
    }
    for item in structural_items:
        if item.parent_identifier is None:
            continue
        parent = by_identifier.get(item.parent_identifier)
        if parent is None:
            raise SchemaValidationError("eligibility structural parent is missing")
        if item.parent_source_start != parent.source_start:
            raise SchemaValidationError("eligibility structural parent source start does not match")
        if item.parent_type == "criterion" and parent.identifier not in criterion_identifiers:
            raise SchemaValidationError("eligibility criterion parent type does not match")
        if item.parent_type == "group_header" and parent.identifier not in group_identifiers:
            raise SchemaValidationError("eligibility group parent type does not match")
        if parent.source_start >= item.source_start or parent.list_depth >= item.list_depth:
            raise SchemaValidationError("eligibility structural parent order is invalid")
    source_text_sha256 = record.get("source_text_sha256")
    source_provenance_sha256 = record.get("source_provenance_sha256")
    view_sha256 = record.get("view_sha256")
    criterion_inventory_sha256 = record.get("criterion_inventory_sha256")
    require_sha256(source_text_sha256, "eligibility source text SHA-256")
    require_sha256(source_provenance_sha256, "eligibility source provenance SHA-256")
    require_sha256(view_sha256, "eligibility trial view SHA-256")
    require_sha256(criterion_inventory_sha256, "eligibility criterion inventory SHA-256")
    parse_tier = record.get("parse_tier")
    if parse_tier not in ELIGIBILITY_PARSE_TIERS:
        raise SchemaValidationError("eligibility parse tier is invalid")
    inclusion_status = record.get("inclusion_status")
    exclusion_status = record.get("exclusion_status")
    valid_statuses = {"criteria", "declared_empty", "unavailable", "not_identified"}
    if inclusion_status not in valid_statuses or exclusion_status not in valid_statuses:
        raise SchemaValidationError("eligibility polarity status is invalid")
    review_reasons = record.get("review_reasons")
    if (
        not isinstance(review_reasons, Sequence)
        or isinstance(review_reasons, (str, bytes))
        or any(reason not in ELIGIBILITY_REVIEW_REASONS for reason in review_reasons)
    ):
        raise SchemaValidationError("eligibility review reasons must be an array of strings")
    built = EligibilityCriterionTrialRecord(
        trial_id=trial_id,
        source_text_sha256=cast(str, source_text_sha256),
        source_provenance_sha256=cast(str, source_provenance_sha256),
        view_sha256=cast(str, view_sha256),
        criterion_inventory_sha256=cast(str, criterion_inventory_sha256),
        parse_tier=cast(EligibilityParseTier, parse_tier),
        review_reasons=tuple(cast(Sequence[EligibilityReviewReason], review_reasons)),
        inclusion_status=cast(EligibilityPolarityStatus, inclusion_status),
        exclusion_status=cast(EligibilityPolarityStatus, exclusion_status),
        inclusion=inclusion,
        exclusion=exclusion,
        unspecified=unspecified,
        other_items=other_items,
    )
    if record.get("criterion_count") != len(items):
        raise SchemaValidationError("eligibility criterion count does not match")
    if record.get("inclusion_criterion_count") != len(inclusion):
        raise SchemaValidationError("eligibility inclusion criterion count does not match")
    if record.get("exclusion_criterion_count") != len(exclusion):
        raise SchemaValidationError("eligibility exclusion criterion count does not match")
    if record.get("unspecified_criterion_count") != len(unspecified):
        raise SchemaValidationError("eligibility unspecified criterion count does not match")
    if record.get("other_item_count") != len(other_items):
        raise SchemaValidationError("eligibility non-criterion item count does not match")
    if [item.ordinal for item in other_items] != list(range(len(other_items))):
        raise SchemaValidationError("eligibility non-criterion ordinals must be contiguous")
    if built.view_sha256 != content_sha256(
        _trial_view_payload(
            items,
            other_items,
            inclusion_status=built.inclusion_status,
            exclusion_status=built.exclusion_status,
        )
    ):
        raise SchemaValidationError("eligibility trial view hash does not match")
    inventory = _criterion_inventory(
        items,
        source_text_sha256=built.source_text_sha256,
        source_provenance_sha256=built.source_provenance_sha256,
    )
    if built.criterion_inventory_sha256 != content_sha256(inventory):
        raise SchemaValidationError("eligibility criterion inventory hash does not match")
    return built


def find_eligibility_criterion_view(views: Iterable[DerivedView]) -> DerivedView | None:
    """Return the supported eligibility criterion view, if the Snapshot supplies one."""

    view = next(
        (item for item in views if item.name == ELIGIBILITY_CRITERION_VIEW_NAME),
        None,
    )
    if view is not None and view.version != ELIGIBILITY_SPLIT_VERSION:
        raise SchemaValidationError("eligibility criterion Derived View version is unsupported")
    return view


def derive_eligibility_criteria(
    section: SemanticTextSection,
    *,
    source_text: str,
) -> tuple[CriterionItem, ...]:
    """Return stable derived criterion records for a complete eligibility source view."""

    if section.role != "eligibility":
        raise ValueError("eligibility criterion derivation requires an eligibility section")
    return _derive_eligibility_criteria(
        section,
        source_text=source_text,
        boundaries=analyze_complete_eligibility(source_text).criteria,
    )


def _derive_eligibility_criteria(
    section: SemanticTextSection,
    *,
    source_text: str,
    boundaries: tuple[EligibilityCriterionBoundary, ...],
) -> tuple[CriterionItem, ...]:
    source_sha256 = "sha256:" + hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    criteria: list[CriterionItem] = []
    for ordinal, boundary in enumerate(boundaries):
        text_sha256 = "sha256:" + hashlib.sha256(boundary.text.encode("utf-8")).hexdigest()
        identity_payload = {
            "version": ELIGIBILITY_SPLIT_VERSION,
            "source_sha256": source_sha256,
            "polarity": boundary.polarity,
            "source_start": boundary.source_start,
            "source_end": boundary.source_end,
            "list_marker": boundary.list_marker,
            "list_marker_type": boundary.list_marker_type,
            "list_marker_ordinal": boundary.list_marker_ordinal,
            "list_indent": boundary.list_indent,
            "list_depth": boundary.list_depth,
            "parent_source_start": boundary.parent_source_start,
            "parent_type": boundary.parent_type,
            "child_relationship": boundary.child_relationship,
            "required_child_count": boundary.required_child_count,
            "group_path": list(boundary.group_path),
            "text_sha256": text_sha256,
        }
        criteria.append(
            CriterionItem(
                ordinal=ordinal,
                text=boundary.text,
                polarity=boundary.polarity,
                provenance=section.provenance,
                identifier=content_sha256(identity_payload),
                additional_fields={
                    "derivation_version": ELIGIBILITY_SPLIT_VERSION,
                    "source_text_sha256": source_sha256,
                    "source_start": boundary.source_start,
                    "source_end": boundary.source_end,
                    "list_marked": boundary.list_marked,
                    "list_marker": boundary.list_marker,
                    "list_marker_type": boundary.list_marker_type,
                    "list_marker_ordinal": boundary.list_marker_ordinal,
                    "list_indent": boundary.list_indent,
                    "list_depth": boundary.list_depth,
                    "parent_source_start": boundary.parent_source_start,
                    "parent_type": boundary.parent_type,
                    "child_relationship": boundary.child_relationship,
                    "required_child_count": boundary.required_child_count,
                    "group_path": list(boundary.group_path),
                    "derived_text_sha256": text_sha256,
                },
                origin="derived",
            )
        )
    return tuple(criteria)


__all__ = [
    "ELIGIBILITY_CRITERION_VIEW_CAPABILITY",
    "ELIGIBILITY_CRITERION_VIEW_CONTENT_SCHEMA",
    "ELIGIBILITY_CRITERION_VIEW_NAME",
    "ELIGIBILITY_SPLIT_VERSION",
    "ELIGIBILITY_VIEW_CONFIGURATION",
    "EligibilityCriterionBoundary",
    "EligibilityCriterionParse",
    "EligibilityCriterionTrialRecord",
    "EligibilityCriterionViewItem",
    "EligibilityItemType",
    "EligibilityNonCriterionBoundary",
    "EligibilityNonCriterionViewItem",
    "EligibilityParseTier",
    "EligibilityPolarityStatus",
    "EligibilityReviewReason",
    "analyze_complete_eligibility",
    "build_eligibility_criterion_trial_record",
    "build_eligibility_criterion_view",
    "derive_complete_eligibility_section",
    "derive_eligibility_criteria",
    "eligibility_criterion_trial",
    "find_eligibility_criterion_view",
    "split_complete_eligibility",
]
