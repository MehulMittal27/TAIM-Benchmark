"""Shared, provenance-preserving Constraint Fact Derived View."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Literal, Protocol, cast

from taim.contracts import canonical_json
from taim.schemas import JsonValue, SchemaValidationError, json_value_to_builtins
from taim.snapshot import (
    AgeBound,
    AgeQuantity,
    BenchmarkSnapshot,
    BenchmarkTopic,
    DerivedView,
    FieldProvenance,
    HealthyVolunteerAcceptance,
    SexEligibility,
    TrialDocument,
)

CONSTRAINT_FACT_VIEW_NAME = "constraint_facts"
CONSTRAINT_FACT_VIEW_VERSION = "1.0"
CONSTRAINT_FACT_CONTENT_SCHEMA = "taim-constraint-facts-v1"
CONSTRAINT_FACT_EXTRACTOR_ID = "taim-rule-constraint-facts"
CONSTRAINT_FACT_EXTRACTOR_VERSION = "1.0"

TOPIC_FACT_NAMES = ("age", "sex", "healthy_volunteer", "smoking", "alcohol")
TRIAL_FACT_NAMES = (
    "minimum_age",
    "maximum_age",
    "sex",
    "healthy_volunteers",
    "smoking",
    "alcohol",
)

FactName = Literal[
    "age",
    "minimum_age",
    "maximum_age",
    "sex",
    "healthy_volunteer",
    "healthy_volunteers",
    "smoking",
    "alcohol",
]
RecordKind = Literal["topic", "trial"]

_AGE_PATTERN = re.compile(
    r"\b(?P<amount>[0-9]{1,6}(?:\.[0-9]+)?)\s*[- ]\s*"
    r"(?P<unit>minutes?|hours?|days?|weeks?|months?|years?)[ -]old\b",
    re.IGNORECASE,
)
_TRIAL_AGE_RANGE_PATTERN = re.compile(
    r"\baged\s+(?P<minimum>[0-9]{1,6}(?:\.[0-9]+)?)\s+to\s+"
    r"(?P<maximum>[0-9]{1,6}(?:\.[0-9]+)?)\s*"
    r"(?P<unit>minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_TRIAL_AGE_BOUND_PATTERN = re.compile(
    r"\b(?P<bound>minimum|maximum)\s+age(?:\s+is|:)?\s+"
    r"(?P<amount>[0-9]{1,6}(?:\.[0-9]+)?)\s*"
    r"(?P<unit>minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_TOPIC_SEX_PATTERNS = (
    re.compile(r"\b(?P<value>female|male)\s+(?:patient|participant|subject)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:patient|participant|subject)\s+(?:is\s+)?(?P<value>female|male)\b",
        re.IGNORECASE,
    ),
)
_TRIAL_SEX_PATTERNS = (
    re.compile(r"\b(?P<value>female|male)[ -]only\b", re.IGNORECASE),
    re.compile(
        r"\b(?:accepts|enrolls?)\s+(?P<value>female|male)\s+"
        r"(?:participants|patients|subjects)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsex\s*:\s*(?P<value>female|male|all)\b", re.IGNORECASE),
)
_TOPIC_HEALTHY_PATTERNS: tuple[tuple[JsonValue, str, re.Pattern[str]], ...] = (
    (
        False,
        "constraint-fact-healthy-volunteer-negation-v1",
        re.compile(r"\bnot\s+(?:a\s+)?healthy volunteer\b", re.IGNORECASE),
    ),
    (
        True,
        "constraint-fact-healthy-volunteer-affirmation-v1",
        re.compile(r"\b(?:is|am)\s+(?:a\s+)?healthy volunteer\b", re.IGNORECASE),
    ),
)
_TRIAL_HEALTHY_PATTERNS: tuple[tuple[JsonValue, str, re.Pattern[str]], ...] = (
    (
        False,
        "constraint-fact-trial-healthy-volunteer-no-v1",
        re.compile(r"\bhealthy volunteers?\s*:\s*no\b", re.IGNORECASE),
    ),
    (
        True,
        "constraint-fact-trial-healthy-volunteer-yes-v1",
        re.compile(r"\bhealthy volunteers?\s*:\s*yes\b", re.IGNORECASE),
    ),
    (
        False,
        "constraint-fact-trial-healthy-volunteer-rejection-v1",
        re.compile(
            r"\b(?:does not|doesn't)\s+(?:accept|allow)\s+healthy volunteers?\b",
            re.IGNORECASE,
        ),
    ),
    (
        True,
        "constraint-fact-trial-healthy-volunteer-acceptance-v1",
        re.compile(
            r"\b(?:accepts|allows)\s+healthy volunteers?\b|"
            r"\bhealthy volunteers?\s+(?:are\s+)?(?:accepted|allowed|eligible)\b",
            re.IGNORECASE,
        ),
    ),
)
_LIFESTYLE_UNSUPPORTED_PATTERNS: tuple[tuple[FactName, JsonValue, str, re.Pattern[str]], ...] = (
    (
        "smoking",
        None,
        "constraint-fact-smoking-unsupported-v1",
        re.compile(r"\b(?:former smoker|ex-smoker)\b", re.IGNORECASE),
    ),
    (
        "alcohol",
        None,
        "constraint-fact-alcohol-unsupported-v1",
        re.compile(r"\b(?:former drinker|social alcohol intake)\b", re.IGNORECASE),
    ),
)
_LIFESTYLE_NEGATION_PATTERNS: tuple[tuple[FactName, JsonValue, str, re.Pattern[str]], ...] = (
    (
        "smoking",
        False,
        "constraint-fact-smoking-negation-v1",
        re.compile(
            r"\b(?:does not smoke|doesn't smoke|never smoked|non-smoker|nonsmoker|"
            r"denies smoking|no (?:smoking|tobacco use))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "alcohol",
        False,
        "constraint-fact-alcohol-negation-v1",
        re.compile(
            r"\b(?:does not drink alcohol|doesn't drink alcohol|never drinks alcohol|"
            r"denies alcohol use|no alcohol use)\b",
            re.IGNORECASE,
        ),
    ),
)
_TRIAL_LIFESTYLE_NEGATION_PATTERNS: tuple[tuple[FactName, JsonValue, str, re.Pattern[str]], ...] = (
    (
        "smoking",
        False,
        "constraint-fact-trial-non-smoking-eligibility-v1",
        re.compile(
            r"\bnon[- ]smokers?\s+(?:are\s+)?(?:eligible|accepted|allowed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "alcohol",
        False,
        "constraint-fact-trial-non-drinking-eligibility-v1",
        re.compile(
            r"\bnon[- ]drinkers?\s+(?:are\s+)?(?:eligible|accepted|allowed)\b",
            re.IGNORECASE,
        ),
    ),
)
_TOPIC_LIFESTYLE_AFFIRMATION_PATTERNS: tuple[
    tuple[FactName, JsonValue, str, re.Pattern[str]], ...
] = (
    (
        "smoking",
        True,
        "constraint-fact-smoking-affirmation-v1",
        re.compile(
            r"\b(?:patient|participant|subject)\s+(?:currently\s+)?"
            r"(?:smokes|uses tobacco)\b|"
            r"\b(?:patient|participant|subject)\s+(?:is|reports being)\s+"
            r"(?:a\s+)?current smoker\b",
            re.IGNORECASE,
        ),
    ),
    (
        "alcohol",
        True,
        "constraint-fact-alcohol-affirmation-v1",
        re.compile(
            r"\b(?:patient|participant|subject)\s+(?:currently\s+)?drinks alcohol\b|"
            r"\b(?:patient|participant|subject)\s+(?:is|reports being)\s+"
            r"(?:a\s+)?current drinker\b",
            re.IGNORECASE,
        ),
    ),
)
_TRIAL_LIFESTYLE_AFFIRMATION_PATTERNS: tuple[
    tuple[FactName, JsonValue, str, re.Pattern[str]], ...
] = (
    (
        "smoking",
        True,
        "constraint-fact-trial-smoking-eligibility-v1",
        re.compile(
            r"\b(?:requires?|accepts?|enrolls?)\s+(?:current\s+)?"
            r"(?:smokers?|tobacco users?)\b|"
            r"\b(?:smokers?|tobacco users?)\s+(?:are\s+)?"
            r"(?:eligible|accepted|allowed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "alcohol",
        True,
        "constraint-fact-trial-alcohol-eligibility-v1",
        re.compile(
            r"\b(?:requires?|accepts?|enrolls?)\s+(?:current\s+)?(?:drinkers?|"
            r"participants who drink alcohol)\b|"
            r"\b(?:drinkers?|participants who drink alcohol)\s+(?:are\s+)?"
            r"(?:eligible|accepted|allowed)\b",
            re.IGNORECASE,
        ),
    ),
)

_NEGATING_PREFIX = re.compile(
    r"(?:\bnot(?:\s+a)?|\bno|\bwithout|\bdoes not|\bdoesn't|\bdo not|\bdon't|"
    r"\bexclude(?:s|d|ing)?|\bprohibit(?:s|ed|ing)?)\s*$",
    re.IGNORECASE,
)
_REJECTING_SUFFIX = re.compile(
    r"^\s+(?:(?:is|are|was|were)\s+)?(?:excluded|ineligible|prohibited|"
    r"not\s+(?:eligible|allowed|accepted))\b",
    re.IGNORECASE,
)
_CRITERIA_HEADING = re.compile(
    r"\b(?P<polarity>inclusion|exclusion)\s+criteria\s*:",
    re.IGNORECASE,
)
_AGE_MAXIMA = {
    "minutes": 130 * 366 * 24 * 60,
    "hours": 130 * 366 * 24,
    "days": 130 * 366,
    "weeks": 130 * 53,
    "months": 130 * 12,
    "years": 130,
}


class ConstraintExtractionError(RuntimeError):
    """A declared text-extractor failure that must resolve fail-open."""


@dataclass(frozen=True, slots=True)
class ExtractedAssertion:
    fact_name: FactName
    value: JsonValue
    start: int
    end: int
    rule_id: str


@dataclass(frozen=True, slots=True)
class PatientCoreFactMapping:
    """Exact Typed Patient Core profile-and-field mapping for shared facts."""

    mapping_id: str
    version: str
    profiles: Mapping[str, Mapping[str, FactName]]

    def __post_init__(self) -> None:
        if not isinstance(self.mapping_id, str) or not self.mapping_id.strip():
            raise ValueError("patient core fact mapping_id must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("patient core fact mapping version must be a non-empty string")
        if not isinstance(self.profiles, Mapping):
            raise TypeError("patient core fact mapping profiles must be an object")
        normalized_profiles: dict[str, Mapping[str, FactName]] = {}
        for profile_id, fields in sorted(self.profiles.items()):
            if not isinstance(profile_id, str) or not profile_id.strip():
                raise ValueError("patient core fact mapping profile_id must be non-empty")
            if not isinstance(fields, Mapping):
                raise TypeError("patient core fact mapping fields must be an object")
            normalized_fields: dict[str, FactName] = {}
            for field_name, fact_name in sorted(fields.items()):
                if not isinstance(field_name, str) or not field_name.strip():
                    raise ValueError("patient core fact mapping field name must be non-empty")
                if fact_name not in TOPIC_FACT_NAMES:
                    raise ValueError("patient core fact mapping target is invalid")
                normalized_fields[field_name] = cast(FactName, fact_name)
            normalized_profiles[profile_id] = MappingProxyType(normalized_fields)
        object.__setattr__(self, "profiles", MappingProxyType(normalized_profiles))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.mapping_id,
            "version": self.version,
            "profiles": {profile_id: dict(fields) for profile_id, fields in self.profiles.items()},
        }


DEFAULT_PATIENT_CORE_FACT_MAPPING = PatientCoreFactMapping(
    mapping_id="taim-no-patient-core-fact-mapping",
    version="1.0",
    profiles={},
)


def _is_negated_or_rejected(text: str, start: int, end: int) -> bool:
    return bool(
        _NEGATING_PREFIX.search(text[max(0, start - 48) : start])
        or _REJECTING_SUFFIX.search(text[end : end + 64])
    )


def _is_trial_assertion_rejected(text: str, start: int, end: int) -> bool:
    active_heading: re.Match[str] | None = None
    for heading in _CRITERIA_HEADING.finditer(text, 0, start):
        active_heading = heading
    return _is_negated_or_rejected(text, start, end) or bool(
        active_heading is not None and active_heading.group("polarity").casefold() == "exclusion"
    )


class ConstraintFactExtractor(Protocol):
    extractor_id: str
    version: str
    implementation_id: str
    configuration: Mapping[str, JsonValue]

    def extract(
        self,
        text: str,
        *,
        record_kind: RecordKind,
        requested_facts: frozenset[FactName],
    ) -> tuple[ExtractedAssertion, ...]: ...


class RuleBasedConstraintFactExtractor:
    """Pinned deterministic rules qualified only on source-text references."""

    extractor_id = CONSTRAINT_FACT_EXTRACTOR_ID
    version = CONSTRAINT_FACT_EXTRACTOR_VERSION
    implementation_id = "taim.constraint_facts.RuleBasedConstraintFactExtractor"
    configuration: Mapping[str, JsonValue] = MappingProxyType(
        {
            "ruleset": "taim-constraint-fact-rules-v1",
            "unit_bearing_age_only": True,
            "record_qualified_sex": True,
            "qualified_healthy_volunteer": True,
        }
    )

    def extract(
        self,
        text: str,
        *,
        record_kind: RecordKind,
        requested_facts: frozenset[FactName],
    ) -> tuple[ExtractedAssertion, ...]:
        assertions: list[ExtractedAssertion] = []
        if record_kind == "topic" and "age" in requested_facts:
            for match in _AGE_PATTERN.finditer(text):
                value = _normalize_age(
                    {"amount": match.group("amount"), "unit": match.group("unit")}
                )
                if value is not None:
                    assertions.append(
                        ExtractedAssertion(
                            "age",
                            value,
                            match.start(),
                            match.end(),
                            "constraint-fact-text-age-v1",
                        )
                    )
        elif (
            record_kind == "trial"
            and {
                "minimum_age",
                "maximum_age",
            }
            & requested_facts
        ):
            for match in _TRIAL_AGE_RANGE_PATTERN.finditer(text):
                if _is_trial_assertion_rejected(text, match.start(), match.end()):
                    continue
                for fact_name, group_name in (
                    ("minimum_age", "minimum"),
                    ("maximum_age", "maximum"),
                ):
                    if fact_name not in requested_facts:
                        continue
                    value = _normalize_age(
                        {"amount": match.group(group_name), "unit": match.group("unit")}
                    )
                    if value is not None:
                        assertions.append(
                            ExtractedAssertion(
                                cast(FactName, fact_name),
                                value,
                                match.start(),
                                match.end(),
                                "constraint-fact-text-trial-age-range-v1",
                            )
                        )
            for match in _TRIAL_AGE_BOUND_PATTERN.finditer(text):
                if _is_trial_assertion_rejected(text, match.start(), match.end()):
                    continue
                fact_name = cast(FactName, f"{match.group('bound').casefold()}_age")
                if fact_name not in requested_facts:
                    continue
                value = _normalize_age(
                    {"amount": match.group("amount"), "unit": match.group("unit")}
                )
                if value is not None:
                    assertions.append(
                        ExtractedAssertion(
                            fact_name,
                            value,
                            match.start(),
                            match.end(),
                            "constraint-fact-text-trial-age-bound-v1",
                        )
                    )
        if "sex" in requested_facts:
            for pattern in _TOPIC_SEX_PATTERNS if record_kind == "topic" else _TRIAL_SEX_PATTERNS:
                for match in pattern.finditer(text):
                    rejected = (
                        _is_trial_assertion_rejected(text, match.start(), match.end())
                        if record_kind == "trial"
                        else _is_negated_or_rejected(text, match.start(), match.end())
                    )
                    if rejected:
                        continue
                    assertions.append(
                        ExtractedAssertion(
                            "sex",
                            match.group("value").casefold(),
                            match.start(),
                            match.end(),
                            f"constraint-fact-text-{record_kind}-qualified-sex-v1",
                        )
                    )
        occupied: dict[FactName, list[tuple[int, int]]] = {}
        healthy_patterns = (
            _TOPIC_HEALTHY_PATTERNS if record_kind == "topic" else _TRIAL_HEALTHY_PATTERNS
        )
        if "healthy_volunteer" in requested_facts:
            for pattern_value, rule_id, pattern in healthy_patterns:
                for match in pattern.finditer(text):
                    rejected = (
                        _is_trial_assertion_rejected(text, match.start(), match.end())
                        if record_kind == "trial"
                        else pattern_value is True
                        and _is_negated_or_rejected(text, match.start(), match.end())
                    )
                    if rejected:
                        continue
                    if any(
                        match.start() < end and start < match.end()
                        for start, end in occupied.get("healthy_volunteer", [])
                    ):
                        continue
                    occupied.setdefault("healthy_volunteer", []).append(
                        (match.start(), match.end())
                    )
                    assertions.append(
                        ExtractedAssertion(
                            "healthy_volunteer",
                            pattern_value,
                            match.start(),
                            match.end(),
                            rule_id,
                        )
                    )
        lifestyle_patterns = (
            *_LIFESTYLE_UNSUPPORTED_PATTERNS,
            *_LIFESTYLE_NEGATION_PATTERNS,
            *(_TRIAL_LIFESTYLE_NEGATION_PATTERNS if record_kind == "trial" else ()),
            *(
                _TOPIC_LIFESTYLE_AFFIRMATION_PATTERNS
                if record_kind == "topic"
                else _TRIAL_LIFESTYLE_AFFIRMATION_PATTERNS
            ),
        )
        for fact_name, pattern_value, rule_id, pattern in lifestyle_patterns:
            if fact_name not in requested_facts:
                continue
            for match in pattern.finditer(text):
                rejected = (
                    _is_trial_assertion_rejected(text, match.start(), match.end())
                    if record_kind == "trial"
                    else pattern_value is True
                    and _is_negated_or_rejected(text, match.start(), match.end())
                )
                if rejected:
                    continue
                if any(
                    match.start() < end and start < match.end()
                    for start, end in occupied.get(fact_name, [])
                ):
                    continue
                occupied.setdefault(fact_name, []).append((match.start(), match.end()))
                assertions.append(
                    ExtractedAssertion(
                        fact_name,
                        pattern_value,
                        match.start(),
                        match.end(),
                        rule_id,
                    )
                )
        return tuple(sorted(assertions, key=lambda item: (item.start, item.end)))


DEFAULT_CONSTRAINT_FACT_EXTRACTOR = RuleBasedConstraintFactExtractor()


def _extractor_identity(extractor: ConstraintFactExtractor) -> dict[str, JsonValue]:
    values = {
        "id": extractor.extractor_id,
        "version": extractor.version,
        "implementation": extractor.implementation_id,
    }
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise TypeError("Constraint Fact extractor identity fields must be non-empty strings")
    if not isinstance(extractor.configuration, Mapping):
        raise TypeError("Constraint Fact extractor configuration must be an object")
    configuration = json_value_to_builtins(extractor.configuration)
    if not isinstance(configuration, dict):
        raise TypeError("Constraint Fact extractor configuration must serialize as an object")
    return {**values, "configuration": configuration}


def _normalize_age(value: object) -> dict[str, JsonValue] | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, Mapping):
        return None
    amount = value.get("amount")
    raw_unit = value.get("unit")
    if (
        isinstance(amount, bool)
        or not isinstance(amount, int | str)
        or not isinstance(raw_unit, str)
    ):
        return None
    unit = raw_unit.casefold().strip()
    if not unit.endswith("s"):
        unit += "s"
    try:
        quantity = AgeQuantity(str(amount), unit)
    except ValueError:
        return None
    if Decimal(quantity.amount) > _AGE_MAXIMA[quantity.unit]:
        return None
    return quantity.to_dict()


def _normalize_structured_value(fact_name: FactName, value: object) -> JsonValue | None:
    if fact_name in {"age", "minimum_age", "maximum_age"}:
        return _normalize_age(value)
    if fact_name == "sex":
        if not isinstance(value, str):
            return None
        normalized = value.casefold().strip()
        aliases = {"f": "female", "female": "female", "m": "male", "male": "male"}
        return aliases.get(normalized)
    if fact_name in {"healthy_volunteer", "healthy_volunteers", "smoking", "alcohol"}:
        return value if isinstance(value, bool) else None
    return None


def _evidence(
    *,
    source_kind: Literal["structured", "text"],
    source_field: str,
    provenance: Sequence[FieldProvenance],
    transformation_rule: str,
    raw_value: JsonValue,
    normalized_value: JsonValue | None,
    text: str | None = None,
    start: int | None = None,
    end: int | None = None,
    extractor_identity: Mapping[str, JsonValue] | None = None,
    outcome: str | None = None,
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "source_kind": source_kind,
        "source_field": source_field,
        "raw_value": raw_value,
        "transformation_rule": transformation_rule,
        "extractor": (None if source_kind == "structured" else dict(extractor_identity or {})),
        "outcome": outcome or ("valid" if normalized_value is not None else "invalid"),
        "provenance": [item.to_dict() for item in provenance],
    }
    if normalized_value is not None:
        payload["normalized_value"] = normalized_value
    if text is not None and start is not None and end is not None:
        payload["text_span"] = {"start": start, "end": end, "text": text[start:end]}
    return payload


def _known_fact(
    value: JsonValue,
    evidence: Sequence[dict[str, JsonValue]],
    *,
    resolution: Literal["structured_precedence", "text_fallback"],
) -> dict[str, JsonValue]:
    return {
        "state": "known",
        "value": value,
        "resolution": resolution,
        "evidence": list(evidence),
    }


def _unknown_fact(
    reason: str,
    evidence: Sequence[dict[str, JsonValue]] = (),
) -> dict[str, JsonValue]:
    return {"state": "unknown", "reason": reason, "evidence": list(evidence)}


def _resolve_assertions(
    assertions: Sequence[tuple[JsonValue, dict[str, JsonValue]]],
    *,
    resolution: Literal["structured_precedence", "text_fallback"],
) -> dict[str, JsonValue]:
    if not assertions:
        return _unknown_fact("no_usable_evidence")
    by_value: dict[str, JsonValue] = {}
    for value, _ in assertions:
        by_value[canonical_json(value)] = value
    evidence = [item for _, item in assertions]
    if len(by_value) != 1:
        return _unknown_fact(f"{resolution}_conflict", evidence)
    return _known_fact(next(iter(by_value.values())), evidence, resolution=resolution)


def _structured_topic_assertions(
    topic: BenchmarkTopic,
    patient_core_mapping: PatientCoreFactMapping,
) -> tuple[
    dict[FactName, list[tuple[JsonValue, dict[str, JsonValue]]]],
    dict[FactName, list[dict[str, JsonValue]]],
]:
    valid: dict[FactName, list[tuple[JsonValue, dict[str, JsonValue]]]] = {}
    invalid: dict[FactName, list[dict[str, JsonValue]]] = {}
    core = topic.typed_patient_core
    if core is None:
        return valid, invalid
    profile_mapping = patient_core_mapping.profiles.get(core.profile_id, {})
    for field_name, assertion in core.fields.items():
        fact_name = profile_mapping.get(field_name)
        if fact_name is None:
            continue
        raw_value = cast(JsonValue, json_value_to_builtins(assertion.value))
        normalized = _normalize_structured_value(fact_name, raw_value)
        item = _evidence(
            source_kind="structured",
            source_field=f"typed_patient_core.{field_name}",
            provenance=assertion.provenance,
            transformation_rule=f"constraint-fact-structured-{fact_name.replace('_', '-')}-v1",
            raw_value=raw_value,
            normalized_value=normalized,
        )
        if normalized is None:
            invalid.setdefault(fact_name, []).append(item)
        else:
            valid.setdefault(fact_name, []).append((normalized, item))
    return valid, invalid


def _extract_text_assertions(
    sources: Sequence[tuple[str, str, Sequence[FieldProvenance]]],
    extractor: ConstraintFactExtractor,
    *,
    record_kind: RecordKind,
    fact_mapping: Mapping[FactName, FactName],
    requested_facts: frozenset[FactName],
) -> tuple[
    dict[FactName, list[tuple[JsonValue, dict[str, JsonValue]]]],
    dict[FactName, list[dict[str, JsonValue]]],
    list[dict[str, JsonValue]],
]:
    assertions: dict[FactName, list[tuple[JsonValue, dict[str, JsonValue]]]] = {}
    invalid: dict[FactName, list[dict[str, JsonValue]]] = {}
    failures: list[dict[str, JsonValue]] = []
    requested_extractions = frozenset(
        source_fact
        for source_fact, target_fact in fact_mapping.items()
        if target_fact in requested_facts
    )
    if not requested_extractions:
        return assertions, invalid, failures
    for source_field, text, provenance in sources:
        try:
            extracted_assertions = extractor.extract(
                text,
                record_kind=record_kind,
                requested_facts=requested_extractions,
            )
        except ConstraintExtractionError:
            failures.append(
                _evidence(
                    source_kind="text",
                    source_field=source_field,
                    provenance=provenance,
                    transformation_rule="constraint-fact-extractor-failure-v1",
                    raw_value=text,
                    normalized_value=None,
                    extractor_identity=_extractor_identity(extractor),
                    outcome="extractor_failure",
                )
            )
            continue
        for extracted in extracted_assertions:
            if extracted.fact_name not in requested_extractions:
                continue
            fact_name = fact_mapping.get(extracted.fact_name)
            if fact_name is None:
                continue
            item = _evidence(
                source_kind="text",
                source_field=source_field,
                provenance=provenance,
                transformation_rule=extracted.rule_id,
                raw_value=text[extracted.start : extracted.end],
                normalized_value=extracted.value,
                text=text,
                start=extracted.start,
                end=extracted.end,
                extractor_identity=_extractor_identity(extractor),
            )
            if extracted.value is None:
                invalid.setdefault(fact_name, []).append(item)
            else:
                assertions.setdefault(fact_name, []).append((extracted.value, item))
    return assertions, invalid, failures


def _fact_record(
    record_id: str,
    fact_names: Sequence[str],
    structured: Mapping[FactName, Sequence[tuple[JsonValue, dict[str, JsonValue]]]],
    invalid_structured: Mapping[FactName, Sequence[dict[str, JsonValue]]],
    text: Mapping[FactName, Sequence[tuple[JsonValue, dict[str, JsonValue]]]],
    invalid_text: Mapping[FactName, Sequence[dict[str, JsonValue]]],
    failures: Sequence[dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    facts: dict[str, JsonValue] = {}
    for fact_name in fact_names:
        typed_name = cast(FactName, fact_name)
        structured_assertions = structured.get(typed_name, [])
        if structured_assertions:
            facts[fact_name] = _resolve_assertions(
                structured_assertions,
                resolution="structured_precedence",
            )
            continue
        text_assertions = text.get(typed_name, [])
        if text_assertions:
            resolved = _resolve_assertions(text_assertions, resolution="text_fallback")
            preceding_evidence = [
                *invalid_structured.get(typed_name, []),
                *invalid_text.get(typed_name, []),
                *failures,
            ]
            if preceding_evidence:
                existing = cast(list[JsonValue], resolved["evidence"])
                resolved["evidence"] = [*preceding_evidence, *existing]
            facts[fact_name] = resolved
        elif failures:
            facts[fact_name] = _unknown_fact(
                "extractor_failure",
                [
                    *invalid_structured.get(typed_name, []),
                    *invalid_text.get(typed_name, []),
                    *failures,
                ],
            )
        elif invalid_text.get(typed_name):
            facts[fact_name] = _unknown_fact("unsupported_text_evidence", invalid_text[typed_name])
        elif invalid_structured.get(typed_name):
            facts[fact_name] = _unknown_fact(
                "invalid_structured_evidence", invalid_structured[typed_name]
            )
        else:
            facts[fact_name] = _unknown_fact("no_usable_evidence")
    return {"record_id": record_id, "facts": facts}


def _topic_record(
    topic: BenchmarkTopic,
    extractor: ConstraintFactExtractor,
    patient_core_mapping: PatientCoreFactMapping,
) -> dict[str, JsonValue]:
    structured, invalid_structured = _structured_topic_assertions(topic, patient_core_mapping)
    text, invalid_text, failures = _extract_text_assertions(
        tuple(
            (
                f"evidence_items[{item.ordinal}].text",
                item.text,
                item.provenance,
            )
            for item in topic.evidence_items
        ),
        extractor,
        record_kind="topic",
        fact_mapping={name: name for name in cast(tuple[FactName, ...], TOPIC_FACT_NAMES)},
        requested_facts=frozenset(
            cast(FactName, name) for name in TOPIC_FACT_NAMES if name not in structured
        ),
    )
    return _fact_record(
        topic.topic_id,
        TOPIC_FACT_NAMES,
        structured,
        invalid_structured,
        text,
        invalid_text,
        failures,
    )


def _structured_trial_assertions(
    trial: TrialDocument,
) -> dict[FactName, list[tuple[JsonValue, dict[str, JsonValue]]]]:
    core = trial.typed_clinical_core
    if core is None:
        return {}
    values: tuple[
        tuple[FactName, AgeBound | SexEligibility | HealthyVolunteerAcceptance | None], ...
    ] = (
        ("minimum_age", core.minimum_age),
        ("maximum_age", core.maximum_age),
        ("sex", core.sex),
        ("healthy_volunteers", core.healthy_volunteers),
    )
    assertions: dict[FactName, list[tuple[JsonValue, dict[str, JsonValue]]]] = {}
    for fact_name, wrapped in values:
        if wrapped is None:
            continue
        provenance = wrapped.provenance
        if isinstance(wrapped, AgeBound):
            normalized: JsonValue = (
                "unbounded" if wrapped.value == "unbounded" else wrapped.value.to_dict()
            )
        else:
            normalized = cast(JsonValue, wrapped.value)
        item = _evidence(
            source_kind="structured",
            source_field=f"typed_clinical_core.{fact_name}",
            provenance=provenance,
            transformation_rule=f"constraint-fact-structured-{fact_name.replace('_', '-')}-v1",
            raw_value=normalized,
            normalized_value=normalized,
        )
        assertions[fact_name] = [(normalized, item)]
    for conflict in core.conflicts:
        fact_name = cast(FactName, conflict.field_name)
        assertions[fact_name] = [
            (
                value,
                _evidence(
                    source_kind="structured",
                    source_field=f"typed_clinical_core.conflicts.{conflict.field_name}",
                    provenance=(provenance,),
                    transformation_rule="constraint-fact-structured-conflict-v1",
                    raw_value=value,
                    normalized_value=value,
                ),
            )
            for value, provenance in zip(conflict.values, conflict.provenance, strict=True)
        ]
    return assertions


def _text_trial_assertions(
    trial: TrialDocument,
    extractor: ConstraintFactExtractor,
    requested_facts: frozenset[FactName],
) -> tuple[
    dict[FactName, list[tuple[JsonValue, dict[str, JsonValue]]]],
    dict[FactName, list[dict[str, JsonValue]]],
    list[dict[str, JsonValue]],
]:
    fact_mapping: Mapping[FactName, FactName] = {
        "minimum_age": "minimum_age",
        "maximum_age": "maximum_age",
        "sex": "sex",
        "healthy_volunteer": "healthy_volunteers",
        "smoking": "smoking",
        "alcohol": "alcohol",
    }
    return _extract_text_assertions(
        tuple(
            (f"sections[{section.ordinal}].text", section.text, section.provenance)
            for section in trial.sections
        ),
        extractor,
        record_kind="trial",
        fact_mapping=fact_mapping,
        requested_facts=requested_facts,
    )


def _trial_record(
    trial: TrialDocument,
    extractor: ConstraintFactExtractor,
) -> dict[str, JsonValue]:
    structured = _structured_trial_assertions(trial)
    text, invalid_text, failures = _text_trial_assertions(
        trial,
        extractor,
        frozenset(cast(FactName, name) for name in TRIAL_FACT_NAMES if name not in structured),
    )
    return _fact_record(
        trial.trial_id,
        TRIAL_FACT_NAMES,
        structured,
        {},
        text,
        invalid_text,
        failures,
    )


def build_constraint_fact_view(
    snapshot: BenchmarkSnapshot,
    *,
    extractor: ConstraintFactExtractor = DEFAULT_CONSTRAINT_FACT_EXTRACTOR,
    patient_core_mapping: PatientCoreFactMapping = DEFAULT_PATIENT_CORE_FACT_MAPPING,
) -> DerivedView:
    """Build the judgment-free shared view without mutating source-grounded Snapshot fields."""

    if not isinstance(snapshot, BenchmarkSnapshot):
        raise TypeError("snapshot must be a BenchmarkSnapshot")
    if not isinstance(patient_core_mapping, PatientCoreFactMapping):
        raise TypeError("patient_core_mapping must be a PatientCoreFactMapping")
    configuration: dict[str, JsonValue] = {
        "extractor": _extractor_identity(extractor),
        "patient_core_fact_mapping": patient_core_mapping.to_dict(),
        "structured_precedence": "valid structured values precede text fallback",
        "conflict_policy": "different normalized values resolve to unknown",
        "failure_policy": "missing, invalid, unsupported, or failed extraction resolves to unknown",
        "fact_names": {
            "topics": list(TOPIC_FACT_NAMES),
            "trials": list(TRIAL_FACT_NAMES),
        },
        "pairwise_decisions": "forbidden",
        "method_owned_retrieval_enrichment": "forbidden",
    }
    content: dict[str, JsonValue] = {
        "schema_version": CONSTRAINT_FACT_CONTENT_SCHEMA,
        "topics": [
            _topic_record(item, extractor, patient_core_mapping) for item in snapshot.topics
        ],
        "trials": [_trial_record(item, extractor) for item in snapshot.trials],
    }
    return DerivedView(
        name=CONSTRAINT_FACT_VIEW_NAME,
        version=CONSTRAINT_FACT_VIEW_VERSION,
        input_snapshot_id=snapshot.base_snapshot_id,
        configuration=configuration,
        content=content,
    )


_KNOWN_RESOLUTIONS = {"structured_precedence", "text_fallback"}
_UNKNOWN_REASONS = {
    "no_usable_evidence",
    "invalid_structured_evidence",
    "unsupported_text_evidence",
    "extractor_failure",
    "structured_precedence_conflict",
    "text_fallback_conflict",
}
_EVIDENCE_OUTCOMES = {"valid", "invalid", "extractor_failure"}


def _validate_configuration(configuration: object) -> dict[str, JsonValue]:
    if not isinstance(configuration, Mapping) or set(configuration) != {
        "extractor",
        "patient_core_fact_mapping",
        "structured_precedence",
        "conflict_policy",
        "failure_policy",
        "fact_names",
        "pairwise_decisions",
        "method_owned_retrieval_enrichment",
    }:
        raise SchemaValidationError("Constraint Fact configuration shape is invalid")
    constants = {
        "structured_precedence": "valid structured values precede text fallback",
        "conflict_policy": "different normalized values resolve to unknown",
        "failure_policy": "missing, invalid, unsupported, or failed extraction resolves to unknown",
        "pairwise_decisions": "forbidden",
        "method_owned_retrieval_enrichment": "forbidden",
    }
    if any(configuration.get(key) != value for key, value in constants.items()):
        raise SchemaValidationError("Constraint Fact configuration policy is invalid")
    fact_names = configuration.get("fact_names")
    if not isinstance(fact_names, Mapping) or set(fact_names) != {"topics", "trials"}:
        raise SchemaValidationError("Constraint Fact configured fact names are invalid")
    if fact_names.get("topics") != list(TOPIC_FACT_NAMES) or fact_names.get("trials") != list(
        TRIAL_FACT_NAMES
    ):
        raise SchemaValidationError("Constraint Fact configured fact names are invalid")
    extractor = configuration.get("extractor")
    if not isinstance(extractor, Mapping) or set(extractor) != {
        "id",
        "version",
        "implementation",
        "configuration",
    }:
        raise SchemaValidationError("Constraint Fact extractor identity is invalid")
    for key in ("id", "version", "implementation"):
        if not isinstance(extractor.get(key), str) or not extractor[key]:
            raise SchemaValidationError("Constraint Fact extractor identity is invalid")
    if not isinstance(extractor.get("configuration"), Mapping):
        raise SchemaValidationError("Constraint Fact extractor configuration is invalid")
    patient_core_mapping = configuration.get("patient_core_fact_mapping")
    if not isinstance(patient_core_mapping, Mapping) or set(patient_core_mapping) != {
        "id",
        "version",
        "profiles",
    }:
        raise SchemaValidationError("Constraint Fact patient core mapping is invalid")
    try:
        PatientCoreFactMapping(
            mapping_id=cast(str, patient_core_mapping.get("id")),
            version=cast(str, patient_core_mapping.get("version")),
            profiles=cast(
                Mapping[str, Mapping[str, FactName]],
                patient_core_mapping.get("profiles"),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("Constraint Fact patient core mapping is invalid") from exc
    return cast(dict[str, JsonValue], json_value_to_builtins(extractor))


def _validate_fact_value(
    value: object,
    *,
    record_kind: RecordKind,
    fact_name: str,
) -> None:
    if fact_name in {"age", "minimum_age", "maximum_age"}:
        if value == "unbounded" and record_kind == "trial" and fact_name != "age":
            return
        if not isinstance(value, Mapping) or set(value) != {"amount", "unit"}:
            raise SchemaValidationError(f"Constraint Fact {fact_name} value is invalid")
        try:
            quantity = AgeQuantity.from_dict(value)
        except (SchemaValidationError, TypeError, ValueError) as exc:
            raise SchemaValidationError(f"Constraint Fact {fact_name} value is invalid") from exc
        if Decimal(quantity.amount) > _AGE_MAXIMA[quantity.unit]:
            raise SchemaValidationError(f"Constraint Fact {fact_name} value is invalid")
        return
    if fact_name == "sex":
        allowed = {"female", "male"} if record_kind == "topic" else {"female", "male", "all"}
        if value not in allowed:
            raise SchemaValidationError("Constraint Fact sex value is invalid")
        return
    if fact_name in {
        "healthy_volunteer",
        "healthy_volunteers",
        "smoking",
        "alcohol",
    } and isinstance(value, bool):
        return
    raise SchemaValidationError(f"Constraint Fact {fact_name} value is invalid")


def _validate_evidence(
    evidence: object,
    *,
    record_kind: RecordKind,
    fact_name: str,
    extractor_identity: Mapping[str, JsonValue],
) -> tuple[str, JsonValue | None]:
    required = {
        "source_kind",
        "source_field",
        "raw_value",
        "transformation_rule",
        "extractor",
        "outcome",
        "provenance",
    }
    if not isinstance(evidence, Mapping) or not required <= set(evidence):
        raise SchemaValidationError("Constraint Fact evidence shape is invalid")
    if set(evidence) - (required | {"normalized_value", "text_span"}):
        raise SchemaValidationError("Constraint Fact evidence shape is invalid")
    source_kind = evidence.get("source_kind")
    source_field = evidence.get("source_field")
    rule = evidence.get("transformation_rule")
    outcome = evidence.get("outcome")
    if source_kind not in {"structured", "text"}:
        raise SchemaValidationError("Constraint Fact evidence source_kind is invalid")
    if not isinstance(source_field, str) or not source_field:
        raise SchemaValidationError("Constraint Fact evidence source_field is invalid")
    if not isinstance(rule, str) or not rule or outcome not in _EVIDENCE_OUTCOMES:
        raise SchemaValidationError("Constraint Fact evidence outcome is invalid")
    provenance = evidence.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise SchemaValidationError("Constraint Fact evidence provenance is invalid")
    for item in provenance:
        if not isinstance(item, Mapping):
            raise SchemaValidationError("Constraint Fact evidence provenance is invalid")
        FieldProvenance.from_dict(item)
    if source_kind == "structured":
        if evidence.get("extractor") is not None or "text_span" in evidence:
            raise SchemaValidationError("structured Constraint Fact evidence is invalid")
    elif evidence.get("extractor") != extractor_identity:
        raise SchemaValidationError("text evidence extractor identity does not match the view")
    if outcome == "extractor_failure" and source_kind != "text":
        raise SchemaValidationError("extractor failure evidence must be textual")
    if outcome == "valid":
        if "normalized_value" not in evidence:
            raise SchemaValidationError("valid Constraint Fact evidence lacks normalized_value")
        normalized = evidence["normalized_value"]
        _validate_fact_value(normalized, record_kind=record_kind, fact_name=fact_name)
    else:
        if "normalized_value" in evidence:
            raise SchemaValidationError("non-valid Constraint Fact evidence has normalized_value")
        normalized = None
    span = evidence.get("text_span")
    if span is not None:
        if (
            source_kind != "text"
            or not isinstance(span, Mapping)
            or set(span) != {"start", "end", "text"}
            or isinstance(span.get("start"), bool)
            or not isinstance(span.get("start"), int)
            or isinstance(span.get("end"), bool)
            or not isinstance(span.get("end"), int)
            or span["start"] < 0
            or span["end"] <= span["start"]
            or not isinstance(span.get("text"), str)
            or span["end"] - span["start"] != len(span["text"])
            or evidence.get("raw_value") != span["text"]
        ):
            raise SchemaValidationError("Constraint Fact evidence text_span is invalid")
    elif source_kind == "text" and outcome != "extractor_failure":
        raise SchemaValidationError("text Constraint Fact evidence must contain a span")
    return cast(str, outcome), cast(JsonValue | None, normalized)


def _validate_fact(
    fact: object,
    *,
    record_kind: RecordKind,
    fact_name: str,
    extractor_identity: Mapping[str, JsonValue],
) -> None:
    if not isinstance(fact, Mapping) or fact.get("state") not in {"known", "unknown"}:
        raise SchemaValidationError("Constraint Fact state is invalid")
    expected_keys = (
        {"state", "value", "resolution", "evidence"}
        if fact["state"] == "known"
        else {"state", "reason", "evidence"}
    )
    if set(fact) != expected_keys or not isinstance(fact.get("evidence"), list):
        raise SchemaValidationError("Constraint Fact shape is invalid")
    evidence_results = [
        _validate_evidence(
            item,
            record_kind=record_kind,
            fact_name=fact_name,
            extractor_identity=extractor_identity,
        )
        for item in fact["evidence"]
    ]
    if fact["state"] == "known":
        if fact.get("resolution") not in _KNOWN_RESOLUTIONS:
            raise SchemaValidationError("known Constraint Fact resolution is invalid")
        _validate_fact_value(fact["value"], record_kind=record_kind, fact_name=fact_name)
        valid_values = [value for outcome, value in evidence_results if outcome == "valid"]
        if not valid_values or any(
            canonical_json(value) != canonical_json(fact["value"]) for value in valid_values
        ):
            raise SchemaValidationError("known Constraint Fact evidence does not support its value")
        expected_source = "structured" if fact["resolution"] == "structured_precedence" else "text"
        if any(
            outcome == "valid" and item.get("source_kind") != expected_source
            for item, (outcome, _) in zip(fact["evidence"], evidence_results, strict=True)
        ):
            raise SchemaValidationError(
                "known Constraint Fact resolution evidence source is inconsistent"
            )
        if not any(
            outcome == "valid" and item.get("source_kind") == expected_source
            for item, (outcome, _) in zip(fact["evidence"], evidence_results, strict=True)
        ):
            raise SchemaValidationError("known Constraint Fact resolution lacks matching evidence")
        return
    reason = fact.get("reason")
    if reason not in _UNKNOWN_REASONS:
        raise SchemaValidationError("unknown Constraint Fact reason is invalid")
    if reason == "no_usable_evidence" and evidence_results:
        raise SchemaValidationError("no_usable_evidence must not contain evidence")
    if reason != "no_usable_evidence" and not evidence_results:
        raise SchemaValidationError("unknown Constraint Fact reason lacks evidence")
    if reason == "extractor_failure" and not any(
        outcome == "extractor_failure" for outcome, _ in evidence_results
    ):
        raise SchemaValidationError("extractor_failure reason lacks extractor failure evidence")
    if reason == "invalid_structured_evidence" and not all(
        outcome == "invalid" and item.get("source_kind") == "structured"
        for item, (outcome, _) in zip(fact["evidence"], evidence_results, strict=True)
    ):
        raise SchemaValidationError("invalid_structured_evidence lacks matching evidence")
    if reason == "unsupported_text_evidence" and not all(
        outcome == "invalid" and item.get("source_kind") == "text"
        for item, (outcome, _) in zip(fact["evidence"], evidence_results, strict=True)
    ):
        raise SchemaValidationError("unsupported_text_evidence lacks matching evidence")
    if not (isinstance(reason, str) and reason.endswith("_conflict")) and any(
        outcome == "valid" for outcome, _ in evidence_results
    ):
        raise SchemaValidationError("unknown Constraint Fact reason has valid evidence")
    if isinstance(reason, str) and reason.endswith("_conflict"):
        distinct = {
            canonical_json(value)
            for outcome, value in evidence_results
            if outcome == "valid" and value is not None
        }
        if len(distinct) < 2:
            raise SchemaValidationError("Constraint Fact conflict lacks conflicting evidence")
        expected_source = "structured" if reason.startswith("structured_") else "text"
        if any(
            item.get("source_kind") != expected_source
            for item, (outcome, _) in zip(fact["evidence"], evidence_results, strict=True)
            if outcome == "valid"
        ):
            raise SchemaValidationError("Constraint Fact conflict evidence source is invalid")


def _validate_fact_records(
    records: object,
    *,
    record_kind: RecordKind,
    fact_names: tuple[str, ...],
    extractor_identity: Mapping[str, JsonValue],
) -> None:
    if not isinstance(records, list):
        raise SchemaValidationError(f"Constraint Fact {record_kind} records must be an array")
    record_ids: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"record_id", "facts"}:
            raise SchemaValidationError(f"Constraint Fact {record_kind} record shape is invalid")
        record_id = record.get("record_id")
        facts = record.get("facts")
        if not isinstance(record_id, str) or not record_id:
            raise SchemaValidationError(f"Constraint Fact {record_kind} record_id is invalid")
        if not isinstance(facts, Mapping) or set(facts) != set(fact_names):
            raise SchemaValidationError(f"Constraint Fact {record_kind} fact set is invalid")
        record_ids.append(record_id)
        for fact_name, fact in facts.items():
            _validate_fact(
                fact,
                record_kind=record_kind,
                fact_name=fact_name,
                extractor_identity=extractor_identity,
            )
    if record_ids != sorted(record_ids) or len(record_ids) != len(set(record_ids)):
        raise SchemaValidationError(f"Constraint Fact {record_kind} records are not ordered")


def load_constraint_fact_view(payload: Mapping[str, object]) -> DerivedView:
    """Load, identity-check, and validate one serialized Constraint Fact Derived View."""

    view = DerivedView.from_dict(payload)
    if view.name != CONSTRAINT_FACT_VIEW_NAME or view.version != CONSTRAINT_FACT_VIEW_VERSION:
        raise SchemaValidationError("unsupported Constraint Fact Derived View identity")
    configuration = json_value_to_builtins(view.configuration)
    extractor_identity = _validate_configuration(configuration)
    content = json_value_to_builtins(view.content)
    if not isinstance(content, dict) or set(content) != {"schema_version", "topics", "trials"}:
        raise SchemaValidationError("Constraint Fact Derived View content shape is invalid")
    if content.get("schema_version") != CONSTRAINT_FACT_CONTENT_SCHEMA:
        raise SchemaValidationError("unsupported Constraint Fact content schema")
    _validate_fact_records(
        content["topics"],
        record_kind="topic",
        fact_names=TOPIC_FACT_NAMES,
        extractor_identity=extractor_identity,
    )
    _validate_fact_records(
        content["trials"],
        record_kind="trial",
        fact_names=TRIAL_FACT_NAMES,
        extractor_identity=extractor_identity,
    )
    return view


def project_constraint_fact_records(
    view: DerivedView,
    *,
    record_kind: RecordKind,
    record_ids: frozenset[str],
) -> Mapping[str, Mapping[str, JsonValue]]:
    """Return validated entity-local records with one view scan."""

    loaded = load_constraint_fact_view(view.to_dict())
    content = json_value_to_builtins(loaded.content)
    if not isinstance(content, dict):
        raise AssertionError("validated Constraint Fact content must be an object")
    collection = "topics" if record_kind == "topic" else "trials"
    records = content[collection]
    if not isinstance(records, list):
        raise AssertionError("validated Constraint Fact records must be an array")
    projected: dict[str, Mapping[str, JsonValue]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise AssertionError("validated Constraint Fact record must be an object")
        record_id = record.get("record_id")
        if isinstance(record_id, str) and record_id in record_ids:
            projected[record_id] = cast(Mapping[str, JsonValue], record)
    return MappingProxyType(projected)


def project_constraint_fact_view(
    view: DerivedView,
    *,
    topic_ids: frozenset[str],
    trial_ids: frozenset[str],
) -> DerivedView:
    """Restrict a validated view to the entities visible in one task input."""

    loaded = load_constraint_fact_view(view.to_dict())
    content = json_value_to_builtins(loaded.content)
    if not isinstance(content, dict):
        raise AssertionError("validated Constraint Fact content must be an object")
    projected_content: dict[str, JsonValue] = {
        "schema_version": CONSTRAINT_FACT_CONTENT_SCHEMA,
        "topics": [
            record
            for record in cast(list[JsonValue], content["topics"])
            if isinstance(record, dict) and record.get("record_id") in topic_ids
        ],
        "trials": [
            record
            for record in cast(list[JsonValue], content["trials"])
            if isinstance(record, dict) and record.get("record_id") in trial_ids
        ],
    }
    if canonical_json(projected_content) == canonical_json(content):
        return view
    projected = DerivedView(
        name=view.name,
        version=view.version,
        input_snapshot_id=view.input_snapshot_id,
        configuration=view.configuration,
        content=projected_content,
        additional_fields=view.additional_fields,
    )
    return load_constraint_fact_view(projected.to_dict())


__all__ = [
    "CONSTRAINT_FACT_CONTENT_SCHEMA",
    "CONSTRAINT_FACT_EXTRACTOR_ID",
    "CONSTRAINT_FACT_EXTRACTOR_VERSION",
    "CONSTRAINT_FACT_VIEW_NAME",
    "CONSTRAINT_FACT_VIEW_VERSION",
    "ConstraintExtractionError",
    "ConstraintFactExtractor",
    "ExtractedAssertion",
    "PatientCoreFactMapping",
    "RuleBasedConstraintFactExtractor",
    "build_constraint_fact_view",
    "load_constraint_fact_view",
    "project_constraint_fact_records",
    "project_constraint_fact_view",
]
