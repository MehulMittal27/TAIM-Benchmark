"""TREC Clinical Trials 2021 Dataset Connector for Benchmark Snapshot.

Only caller-supplied artifacts matching the frozen TREC 2021 Source Bundle
lock are read.  This module performs no network access or live registry
enrichment.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
import xml.parsers.expat as expat
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import TypeVar, cast

from taim import safe_xml
from taim.data.prepared import PreparationResult, write_prepared_benchmark
from taim.evaluation_package import ConnectorDiagnostic, EvaluationPackage
from taim.file_hash import sha256_file
from taim.schemas import JsonValue, RelevanceJudgment, SchemaValidationError
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    CAPABILITY_FIELD_PROVENANCE,
    CAPABILITY_PATIENT_EVIDENCE,
    CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
    CAPABILITY_TYPED_TRIAL_CORE,
    AgeBound,
    AgeQuantity,
    BenchmarkSnapshot,
    BenchmarkTopic,
    CoreFieldConflict,
    CriterionItem,
    FieldProvenance,
    HealthyVolunteerAcceptance,
    PatientEvidenceItem,
    ProvenanceLocatorKind,
    SemanticTextSection,
    SexEligibility,
    SourceRecordIdentity,
    TrialDocument,
    TypedClinicalCore,
)
from taim.source import SourceArtifact, SourceBundle

DATASET_ID = "trec-ct-2021"
BENCHMARK_LINEAGE = "trec-ct-2021"
SNAPSHOT_NAME = "trec-ct-2021"
SOURCE_RECIPE_ID = "trec-ct-2021-snapshot-recipe-v3"
CONNECTOR_NAME = "taim.data.trec_ct_2021"
CONNECTOR_VERSION = "2.2"
SOURCE_BUNDLE_LOCK_VERSION = "2.0"

TOPICS_FILENAME = "topics2021.xml"
QRELS_FILENAME = "qrels2021.txt"
TRIAL_ARCHIVE_FILENAMES = tuple(f"ClinicalTrials.2021-04-27.part{part}.zip" for part in range(1, 6))
DEFAULT_LOCK_PATH = Path(__file__).with_name("locks") / "trec-ct-2021.json"


class IngestionError(ValueError):
    """Raised when a source, lock, or parsed dataset violates the contract."""


class SourceVerificationError(IngestionError):
    """Raised when a source artifact does not match its immutable lock."""


class MalformedTrialError(IngestionError):
    """A malformed individual trial that can be recorded and skipped."""

    def __init__(self, message: str, *, trial_id: str | None = None) -> None:
        super().__init__(message)
        self.trial_id = trial_id


EXPECTED_SOURCE_ROLES = (
    "topics",
    "qrels",
    "trials_part_1",
    "trials_part_2",
    "trials_part_3",
    "trials_part_4",
    "trials_part_5",
)

_EXECUTED_SECTION_MAPPINGS = (
    ("clinical_study/brief_title", "brief_title", "semantic_text", "ctgov-xml-text-whitespace-v1"),
    (
        "clinical_study/official_title",
        "official_title",
        "semantic_text",
        "ctgov-xml-text-whitespace-v1",
    ),
    (
        "clinical_study/brief_summary/textblock",
        "summary",
        "semantic_text",
        "ctgov-xml-text-whitespace-v1",
    ),
    (
        "clinical_study/detailed_description/textblock",
        "summary",
        "semantic_text",
        "ctgov-xml-text-whitespace-v1",
    ),
    ("clinical_study/condition", "condition", "semantic_text", "ctgov-xml-text-whitespace-v1"),
    (
        "clinical_study/intervention",
        "intervention",
        "semantic_text",
        "ctgov-xml-intervention-text-v1",
    ),
    (
        "clinical_study/eligibility/criteria/textblock",
        "eligibility",
        "semantic_text_blob",
        "ctgov-xml-text-whitespace-v1",
    ),
)
_EXECUTED_TYPED_MAPPINGS = (
    ("clinical_study/eligibility/minimum_age", "minimum_age", "shared_core", "ctgov-xml-age-v1"),
    ("clinical_study/eligibility/maximum_age", "maximum_age", "shared_core", "ctgov-xml-age-v1"),
    ("clinical_study/eligibility/gender", "sex", "shared_core", "ctgov-xml-sex-v1"),
    (
        "clinical_study/eligibility/healthy_volunteers",
        "healthy_volunteers",
        "shared_core",
        "ctgov-xml-healthy-volunteers-v2",
    ),
)


def _field_disposition(
    source: str,
    target: str,
    disposition: str,
    rule: str,
) -> dict[str, JsonValue]:
    return {
        "source": source,
        "disposition": disposition,
        "target": target,
        "normalization_rule": rule,
    }


FIELD_DISPOSITION_TABLE: tuple[dict[str, JsonValue], ...] = (
    {
        "source": "topics/topic",
        "disposition": "patient_evidence",
        "target": "narrative",
        "normalization_rule": "trec-topic-narrative-whitespace-v1",
    },
    *(
        _field_disposition(source, target, disposition, rule)
        for source, target, disposition, rule in (
            *_EXECUTED_SECTION_MAPPINGS,
            *_EXECUTED_TYPED_MAPPINGS,
        )
    ),
    {
        "source": "clinical_study/* (all other fields)",
        "disposition": "intentionally_ignored",
        "rationale": "outside the conservative Snapshot shared core",
    },
)

CANONICAL_TEXT_RECIPE: dict[str, JsonValue] = {
    "id": "trec-ct-2021-canonical-trial-text-v1",
    "whitespace": "collapse Unicode whitespace to one ASCII space and strip",
    "unicode_normalization": "none; preserve source code points",
    "source_markup": "exclude XML markup; retain decoded text content",
    "section_order": [
        "brief_title",
        "official_title",
        "summary",
        "condition",
        "intervention",
        "eligibility",
    ],
    "section_labels": {
        "brief_title": "[BRIEF_TITLE]",
        "official_title": "[OFFICIAL_TITLE]",
        "summary": "[SUMMARY]",
        "condition": "[CONDITION]",
        "intervention": "[INTERVENTION]",
        "eligibility": "[ELIGIBILITY]",
    },
    "typed_field_order": ["minimum_age", "maximum_age", "sex", "healthy_volunteers"],
    "typed_labels": {
        "minimum_age": "[MINIMUM_AGE]",
        "maximum_age": "[MAXIMUM_AGE]",
        "sex": "[SEX]",
        "healthy_volunteers": "[HEALTHY_VOLUNTEERS]",
    },
    "separator": "LF",
    "repetition": "preserve every non-empty source item in source order within role",
    "omission": "omit absent sections and typed fields without defaults",
    "unbounded_token": "unbounded",
    "boolean_tokens": {"true": "true", "false": "false"},
}

CANONICAL_PATIENT_TEXT_RECIPE: dict[str, JsonValue] = {
    "id": "trec-ct-2021-canonical-patient-text-v1",
    "source_evidence_kind": "narrative",
    "whitespace": "collapse Unicode whitespace to one ASCII space and strip",
    "unicode_normalization": "none; preserve source code points",
    "separator": "LF",
    "omission": "narrative is required; do not extract typed patient facts",
}

SOURCE_RECIPE: dict[str, JsonValue] = {
    "recipe_id": SOURCE_RECIPE_ID,
    "snapshot_contract_version": "2.0",
    "benchmark_lineage": BENCHMARK_LINEAGE,
    "allowed_source_roles": [
        "topics",
        "qrels",
        "trials_part_1",
        "trials_part_2",
        "trials_part_3",
        "trials_part_4",
        "trials_part_5",
    ],
    "allowed_artifact_ids": [
        "topics",
        "qrels",
        "trials_part_1",
        "trials_part_2",
        "trials_part_3",
        "trials_part_4",
        "trials_part_5",
    ],
    "system_input_artifact_ids": [
        "topics",
        "trials_part_1",
        "trials_part_2",
        "trials_part_3",
        "trials_part_4",
        "trials_part_5",
    ],
    "connector": {"name": CONNECTOR_NAME, "version": CONNECTOR_VERSION},
    "parsers": {
        "topics": "trec-ct-2021-topics-xml-v1",
        "trials": "clinicaltrials-gov-historical-xml-v1",
        "archives": "zip-member-order-by-name-v1",
    },
    "field_disposition_table": list(FIELD_DISPOSITION_TABLE),
    "canonical_patient_text_recipe": CANONICAL_PATIENT_TEXT_RECIPE,
    "canonical_trial_text_recipe": CANONICAL_TEXT_RECIPE,
    "diagnostic_policy": {
        "version": "trec-ct-2021-snapshot-diagnostics-v1",
        "field_invalidity": "omit typed value and emit field_nonfatal diagnostic",
        "record_invalidity": "skip record and emit record_fatal diagnostic",
        "conflict": (
            "omit the known typed value, preserve normalized assertions as a System-visible "
            "Core Field Conflict, and emit a field_nonfatal diagnostic"
        ),
        "order": "source artifact, record, then frozen core field mapping order",
    },
    "external_enrichment": "forbidden",
}

_NCT_ID = re.compile(r"NCT[0-9]{8}\Z")
_TOPIC_ID = re.compile(r"[1-9][0-9]*\Z")
_AGE = re.compile(
    r"\A([0-9]+(?:\.[0-9]+)?)\s+(minute|hour|day|week|month|year)s?\Z",
    re.IGNORECASE,
)
_UNIT_PLURALS = {
    "minute": "minutes",
    "hour": "hours",
    "day": "days",
    "week": "weeks",
    "month": "months",
    "year": "years",
}


def parse_qrels(path: str | Path) -> tuple[RelevanceJudgment, ...]:
    """Parse four-column TREC Judgments without collapsing explicit zero labels."""

    judgments: list[RelevanceJudgment] = []
    seen_keys: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise IngestionError(f"blank Judgment record at line {line_number}")
            fields = line.split()
            if len(fields) != 4:
                raise IngestionError(
                    f"Judgment line {line_number} must contain four whitespace-separated fields"
                )
            topic_id, iteration, trial_id, serialized_label = fields
            if _TOPIC_ID.fullmatch(topic_id) is None:
                raise IngestionError(
                    f"invalid topic ID at Judgment line {line_number}: {topic_id!r}"
                )
            if iteration != "0":
                raise IngestionError(f"Judgment iteration must be 0 at line {line_number}")
            if _NCT_ID.fullmatch(trial_id) is None:
                raise IngestionError(f"invalid NCT ID at Judgment line {line_number}: {trial_id!r}")
            if serialized_label not in {"0", "1", "2"}:
                raise IngestionError(f"Judgment label must be 0, 1, or 2 at line {line_number}")
            key = (topic_id, trial_id)
            if key in seen_keys:
                raise IngestionError(f"duplicate Judgment for {topic_id}/{trial_id}")
            try:
                judgments.append(
                    RelevanceJudgment(
                        topic_id=topic_id,
                        trial_id=trial_id,
                        label=int(serialized_label),
                    )
                )
            except SchemaValidationError as exc:
                raise IngestionError(f"invalid Judgment line {line_number}: {exc}") from exc
            seen_keys.add(key)
    if not judgments:
        raise IngestionError("Judgments must contain at least one record")
    judgments.sort(key=lambda judgment: (judgment.topic_id, judgment.trial_id))
    return tuple(judgments)


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


class _ClinicalTrialsMarkupText(HTMLParser):
    _BLOCK_TAGS = frozenset(
        {"br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ol", "p", "pre", "ul"}
    )
    _IGNORED_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in self._IGNORED_TAGS:
            self.ignored_depth += 1
        elif not self.ignored_depth and normalized in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in self._IGNORED_TAGS:
            self.ignored_depth = max(0, self.ignored_depth - 1)
        elif not self.ignored_depth and normalized in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def _clinicaltrials_markup_text(value: str) -> str:
    parser = _ClinicalTrialsMarkupText()
    parser.feed(value)
    parser.close()
    return _collapse_whitespace("".join(parser.parts))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    return [child for child in element if _local_name(child.tag) == name]


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    children = _children(element, name)
    return children[0] if children else None


def _element_text(element: ET.Element | None) -> str:
    return _collapse_whitespace(_element_raw_text(element))


def _element_raw_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext())


def extract_complete_eligibility_text(
    serialized: bytes,
) -> tuple[str, str | None, str | None]:
    """Return the trial ID plus decoded and lexical eligibility textblock views.

    The decoded view preserves the source text-node layout while resolving XML entities for
    downstream consumers.  The lexical view preserves the exact inner XML and exists only for
    verification against the frozen source provenance.
    """

    try:
        root = safe_xml.fromstring(serialized)
    except safe_xml.UnsafeXmlError as exc:
        raise IngestionError(f"unsafe trial XML: {exc}") from exc
    except ET.ParseError as exc:
        raise IngestionError(f"malformed trial XML: {exc}") from exc
    if _local_name(root.tag) != "clinical_study":
        raise IngestionError("trial XML root must be <clinical_study>")
    lexical = _XmlLexicalContent(serialized, root)
    trial_id = _element_text(_child(_child(root, "id_info"), "nct_id"))
    if _NCT_ID.fullmatch(trial_id) is None:
        raise IngestionError(f"invalid or missing NCT ID {trial_id!r}")
    eligibility = _child(root, "eligibility")
    criteria = _child(eligibility, "criteria")
    textblock = _child(criteria, "textblock")
    decoded = _element_raw_text(textblock)
    raw = lexical.raw(textblock)
    return trial_id, decoded if decoded else None, raw if raw else None


class _XmlLexicalContent:
    """Exact inner-XML slices paired with ElementTree nodes in source order."""

    def __init__(self, serialized: bytes, root: ET.Element) -> None:
        parser = expat.ParserCreate()
        records: list[tuple[str, int, int] | None] = []
        stack: list[tuple[int, str, int]] = []

        def start(name: str, _attributes: object) -> None:
            start_index = parser.CurrentByteIndex
            inner_start = self._open_tag_end(serialized, start_index)
            ordinal = len(records)
            records.append(None)
            stack.append((ordinal, name, inner_start))

        def end(_name: str) -> None:
            ordinal, name, inner_start = stack.pop()
            records[ordinal] = (name, inner_start, parser.CurrentByteIndex)

        parser.StartElementHandler = start
        parser.EndElementHandler = end
        parser.Parse(serialized, True)

        elements = list(root.iter())
        completed = [record for record in records if record is not None]
        if len(elements) != len(completed):
            raise IngestionError("XML lexical provenance does not match parsed elements")
        encoding_match = re.match(rb"\s*<\?xml[^>]*encoding=['\"]([^'\"]+)", serialized)
        encoding = (
            encoding_match.group(1).decode("ascii") if encoding_match is not None else "utf-8"
        )
        self._raw_by_element: dict[int, str] = {}
        for element, (name, inner_start, inner_end) in zip(elements, completed, strict=True):
            if _local_name(element.tag) != _local_name(name):
                raise IngestionError("XML lexical provenance element order is inconsistent")
            self._raw_by_element[id(element)] = serialized[inner_start:inner_end].decode(encoding)

    @staticmethod
    def _open_tag_end(serialized: bytes, start_index: int) -> int:
        quote: int | None = None
        for index in range(start_index, len(serialized)):
            byte = serialized[index]
            if quote is None and byte in {ord("'"), ord('"')}:
                quote = byte
            elif quote == byte:
                quote = None
            elif quote is None and byte == ord(">"):
                return index + 1
        raise IngestionError("XML start tag is not terminated")

    def raw(self, element: ET.Element | None) -> str:
        if element is None:
            return ""
        return self._raw_by_element[id(element)]


def _source_trial_id(member_name: str) -> str | None:
    matches = re.findall(r"NCT[0-9]{8}", member_name)
    return matches[-1] if matches else None


def _source_artifact(raw: Mapping[str, object]) -> SourceArtifact:
    required = {
        "role",
        "filename",
        "url",
        "byte_size",
        "sha256",
        "acquisition_date",
        "access_terms",
        "redistribution_terms",
    }
    if set(raw) != required:
        missing = sorted(required - raw.keys())
        unexpected = sorted(raw.keys() - required)
        raise IngestionError(
            f"Source Bundle entry fields mismatch; missing={missing}, unexpected={unexpected}"
        )
    role = raw.get("role")
    return SourceArtifact(
        artifact_id=cast(str, role),
        role=cast(str, role),
        filename=cast(str, raw.get("filename")),
        url=cast(str, raw.get("url")),
        byte_size=cast(int, raw.get("byte_size")),
        sha256=cast(str, raw.get("sha256")),
        acquisition_date=cast(str, raw.get("acquisition_date")),
        access_terms=cast(str, raw.get("access_terms")),
        redistribution_terms=cast(str, raw.get("redistribution_terms")),
    )


def load_source_bundle_lock(path: str | Path | None = None) -> SourceBundle:
    """Load the raw TREC 2021 Source Bundle independently of connector parsing."""

    lock_path = Path(path) if path is not None else DEFAULT_LOCK_PATH
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestionError(f"invalid Source Bundle lock JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise IngestionError("Source Bundle lock must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != SOURCE_BUNDLE_LOCK_VERSION:
        qualifier = "historical " if schema_version == "1.0" else ""
        raise IngestionError(
            f"unsupported {qualifier}Source Bundle lock schema_version "
            f"{schema_version!r}; expected {SOURCE_BUNDLE_LOCK_VERSION!r}"
        )
    expected_fields = {"schema_version", "dataset_id", "lock_status", "sources"}
    if set(payload) != expected_fields:
        missing = sorted(expected_fields - payload.keys())
        unexpected = sorted(payload.keys() - expected_fields)
        raise IngestionError(
            f"Source Bundle lock fields mismatch; missing={missing}, unexpected={unexpected}"
        )
    if payload.get("dataset_id") != DATASET_ID:
        raise IngestionError(f"Source Bundle dataset_id must be {DATASET_ID!r}")
    if payload.get("lock_status") != "complete":
        raise IngestionError("Source Bundle lock must be complete")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list):
        raise IngestionError("Source Bundle sources must be an array")
    artifacts = tuple(
        _source_artifact(cast(Mapping[str, object], item))
        for item in raw_sources
        if isinstance(item, Mapping)
    )
    if len(artifacts) != len(raw_sources):
        raise IngestionError("every Source Bundle source must be an object")
    roles = [artifact.role for artifact in artifacts]
    if len(roles) != len(set(roles)):
        raise IngestionError("Source Bundle source roles must be unique")
    by_role = {artifact.role: artifact for artifact in artifacts}
    if set(by_role) != set(EXPECTED_SOURCE_ROLES):
        raise IngestionError("Source Bundle source roles do not match TREC 2021")
    expected_filenames = {
        "topics": TOPICS_FILENAME,
        "qrels": QRELS_FILENAME,
        **{
            f"trials_part_{part}": filename
            for part, filename in enumerate(TRIAL_ARCHIVE_FILENAMES, start=1)
        },
    }
    for role, filename in expected_filenames.items():
        if by_role[role].filename != filename:
            raise IngestionError(f"Source Bundle role {role!r} must use {filename!r}")
    return SourceBundle(
        artifacts=tuple(by_role[role] for role in EXPECTED_SOURCE_ROLES),
        lock_filename=lock_path.name,
        lock_sha256=sha256_file(lock_path),
    )


def _verify_source_bundle(source_directory: Path, source_bundle: SourceBundle) -> None:
    missing = [
        artifact.filename
        for artifact in source_bundle.artifacts
        if not (source_directory / artifact.filename).is_file()
    ]
    if missing:
        raise SourceVerificationError(
            "missing required Source Bundle files: " + ", ".join(sorted(missing))
        )
    errors: list[str] = []
    for artifact in source_bundle.artifacts:
        path = source_directory / artifact.filename
        if path.stat().st_size != artifact.byte_size:
            errors.append(f"{artifact.filename}: byte size does not match Source Bundle lock")
            continue
        if sha256_file(path) != artifact.sha256:
            errors.append(f"{artifact.filename}: SHA-256 does not match Source Bundle lock")
    if errors:
        raise SourceVerificationError("Source Bundle verification failed: " + "; ".join(errors))


def _provenance(
    artifact: SourceArtifact,
    *,
    record_id: str,
    locator_kind: ProvenanceLocatorKind,
    location: str,
    raw_value: str,
    rule: str,
) -> FieldProvenance:
    raw_bytes = raw_value.encode("utf-8")
    raw_value_sha256: str | None = None
    raw_value_byte_length: int | None = None
    excerpt = raw_value
    if len(raw_bytes) > 4_096:
        excerpt = raw_bytes[:4_096].decode("utf-8", errors="ignore")
        raw_value_sha256 = f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"
        raw_value_byte_length = len(raw_bytes)
    return FieldProvenance(
        artifact_id=artifact.artifact_id,
        artifact_sha256=artifact.sha256,
        source_record_id=record_id,
        locator_kind=locator_kind,
        location=location,
        raw_value=excerpt,
        transformation_rule=rule,
        raw_value_sha256=raw_value_sha256,
        raw_value_byte_length=raw_value_byte_length,
    )


def parse_topics_xml(
    path: str | Path,
    *,
    artifact: SourceArtifact,
) -> tuple[BenchmarkTopic, ...]:
    serialized = Path(path).read_bytes()
    try:
        root = safe_xml.fromstring(serialized)
    except safe_xml.UnsafeXmlError as exc:
        raise IngestionError(f"unsafe topics XML: {exc}") from exc
    except ET.ParseError as exc:
        raise IngestionError(f"malformed topics XML: {exc}") from exc
    lexical = _XmlLexicalContent(serialized, root)
    if _local_name(root.tag) != "topics":
        raise IngestionError("topics XML root must be <topics>")
    if root.attrib.get("task") != "2021 TREC Clinical Trials":
        raise IngestionError("topics XML task must be '2021 TREC Clinical Trials'")
    topics: list[BenchmarkTopic] = []
    seen: set[str] = set()
    for element in root:
        if _local_name(element.tag) != "topic":
            raise IngestionError(f"unexpected topics XML element <{_local_name(element.tag)}>")
        topic_id = element.attrib.get("number")
        if not isinstance(topic_id, str) or _TOPIC_ID.fullmatch(topic_id) is None:
            raise IngestionError("topic number must be a positive decimal integer")
        if topic_id in seen:
            raise IngestionError(f"duplicate topic number {topic_id}")
        decoded_text = _element_raw_text(element)
        text = _collapse_whitespace(decoded_text)
        if not text:
            raise IngestionError(f"topic {topic_id} has empty patient text")
        provenance = _provenance(
            artifact,
            record_id=topic_id,
            locator_kind="xml_path",
            location=f"/topics/topic[@number='{topic_id}']",
            raw_value=lexical.raw(element),
            rule="trec-topic-narrative-whitespace-v1",
        )
        topics.append(
            BenchmarkTopic(
                topic_id=topic_id,
                source_identity=SourceRecordIdentity(
                    namespace="trec-ct-2021.topics",
                    record_id=topic_id,
                ),
                canonical_text=text,
                evidence_items=(
                    PatientEvidenceItem(
                        kind="narrative",
                        ordinal=0,
                        text=text,
                        provenance=(provenance,),
                    ),
                ),
            )
        )
        seen.add(topic_id)
    if not topics:
        raise IngestionError("topics XML must contain at least one topic")
    return tuple(sorted(topics, key=lambda item: item.topic_id))


def normalize_ctgov_age(
    raw_value: str,
    *,
    maximum: bool,
) -> AgeQuantity | str | None:
    """Normalize one official XML/JSON age lexical without unit conversion."""

    value = _collapse_whitespace(raw_value)
    if not value:
        return None
    if value.casefold() in {"n/a", "not applicable"}:
        return "unbounded" if maximum else None
    match = _AGE.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported age lexical {value!r}")
    amount, unit = match.groups()
    return AgeQuantity(amount, _UNIT_PLURALS[unit.casefold()])


def normalize_ctgov_sex(raw_value: str) -> str | None:
    value = _collapse_whitespace(raw_value).casefold()
    if not value:
        return None
    normalized = {"female": "female", "male": "male", "all": "all"}.get(value)
    if normalized is None:
        raise ValueError(f"unsupported sex lexical {raw_value!r}")
    return normalized


def normalize_ctgov_healthy_volunteers(raw_value: str) -> bool | None:
    value = _collapse_whitespace(raw_value).casefold()
    if not value:
        return None
    if value in {"yes", "true", "accepts healthy volunteers"}:
        return True
    if value in {"no", "false"}:
        return False
    raise ValueError(f"unsupported healthy-volunteer lexical {raw_value!r}")


def map_ctgov_json_eligibility(
    study: Mapping[str, object],
    *,
    artifact: SourceArtifact,
) -> tuple[TypedClinicalCore | None, tuple[ConnectorDiagnostic, ...]]:
    """Map one synthetic/native JSON eligibility module through official JSON Pointers."""

    protocol_section = study.get("protocolSection")
    if not isinstance(protocol_section, Mapping):
        raise SchemaValidationError("ClinicalTrials.gov JSON protocolSection must be an object")
    identification = protocol_section.get("identificationModule")
    if not isinstance(identification, Mapping):
        raise SchemaValidationError("ClinicalTrials.gov JSON identificationModule is required")
    trial_id = identification.get("nctId")
    if not isinstance(trial_id, str) or _NCT_ID.fullmatch(trial_id) is None:
        raise SchemaValidationError("ClinicalTrials.gov JSON nctId is invalid")
    eligibility = protocol_section.get("eligibilityModule")
    if eligibility is None:
        return None, ()
    if not isinstance(eligibility, Mapping):
        raise SchemaValidationError("ClinicalTrials.gov JSON eligibilityModule must be an object")

    diagnostics: list[ConnectorDiagnostic] = []

    def provenance(field: str, raw: object, rule: str) -> FieldProvenance:
        raw_value = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
        if isinstance(raw, str):
            raw_value = raw
        return _provenance(
            artifact,
            record_id=trial_id,
            locator_kind="json_pointer",
            location=f"/protocolSection/eligibilityModule/{field}",
            raw_value=raw_value,
            rule=rule,
        )

    def invalid(
        *,
        code: str,
        field: str,
        raw: object,
        rule: str,
        message: str,
    ) -> None:
        diagnostics.append(
            ConnectorDiagnostic(
                code=code,
                severity="field_nonfatal",
                message=message,
                record_id=trial_id,
                field=field,
                provenance=(provenance(field, raw, rule),),
            )
        )

    def age(field: str, *, maximum: bool) -> AgeBound | None:
        raw = eligibility.get(field)
        if raw is None:
            return None
        rule = "ctgov-json-age-v1"
        if not isinstance(raw, str):
            invalid(
                code=f"invalid-{field.replace('Age', '-age').lower()}",
                field=field,
                raw=raw,
                rule=rule,
                message=f"{field} must be a NormalizedTime string",
            )
            return None
        source = provenance(field, raw, rule)
        try:
            normalized = normalize_ctgov_age(raw, maximum=maximum)
        except (SchemaValidationError, ValueError) as exc:
            invalid(
                code=f"invalid-{field.replace('Age', '-age').lower()}",
                field=field,
                raw=raw,
                rule=rule,
                message=str(exc),
            )
            return None
        return AgeBound(cast(AgeQuantity | str, normalized), (source,)) if normalized else None  # type: ignore[arg-type]

    raw_sex = eligibility.get("sex")
    sex: SexEligibility | None = None
    if raw_sex is not None:
        if isinstance(raw_sex, str):
            source = provenance("sex", raw_sex, "ctgov-json-sex-v1")
            try:
                normalized_sex = normalize_ctgov_sex(raw_sex)
            except ValueError as exc:
                invalid(
                    code="unrecognized-sex",
                    field="sex",
                    raw=raw_sex,
                    rule="ctgov-json-sex-v1",
                    message=str(exc),
                )
            else:
                if normalized_sex is not None:
                    sex = SexEligibility(normalized_sex, (source,))
        else:
            invalid(
                code="unrecognized-sex",
                field="sex",
                raw=raw_sex,
                rule="ctgov-json-sex-v1",
                message="sex must be a string enum",
            )

    raw_healthy = eligibility.get("healthyVolunteers")
    healthy: HealthyVolunteerAcceptance | None = None
    if raw_healthy is not None:
        if isinstance(raw_healthy, bool):
            healthy = HealthyVolunteerAcceptance(
                raw_healthy,
                (
                    provenance(
                        "healthyVolunteers",
                        raw_healthy,
                        "ctgov-json-healthy-volunteers-v1",
                    ),
                ),
            )
        else:
            invalid(
                code="invalid-healthy-volunteers",
                field="healthyVolunteers",
                raw=raw_healthy,
                rule="ctgov-json-healthy-volunteers-v1",
                message="healthyVolunteers must be boolean",
            )

    core = TypedClinicalCore(
        minimum_age=age("minimumAge", maximum=False),
        maximum_age=age("maximumAge", maximum=True),
        sex=sex,
        healthy_volunteers=healthy,
    )
    return (core if core.to_dict() else None), tuple(diagnostics)


def map_ctgov_json_trial_layers(
    study: Mapping[str, object],
    *,
    artifact: SourceArtifact,
) -> tuple[
    tuple[SemanticTextSection, ...],
    TypedClinicalCore | None,
    tuple[ConnectorDiagnostic, ...],
]:
    """Map the shared eligibility section and typed core from one modern JSON study."""

    core, core_diagnostics = map_ctgov_json_eligibility(study, artifact=artifact)
    protocol_section = cast(Mapping[str, object], study["protocolSection"])
    identification = cast(Mapping[str, object], protocol_section["identificationModule"])
    trial_id = cast(str, identification["nctId"])
    eligibility = protocol_section.get("eligibilityModule")
    if eligibility is None:
        return (), core, core_diagnostics
    eligibility_module = cast(Mapping[str, object], eligibility)
    raw_criteria = eligibility_module.get("eligibilityCriteria")
    if raw_criteria is None:
        return (), core, core_diagnostics
    location = "/protocolSection/eligibilityModule/eligibilityCriteria"
    raw_value = (
        raw_criteria
        if isinstance(raw_criteria, str)
        else json.dumps(raw_criteria, ensure_ascii=False, separators=(",", ":"))
    )
    provenance = _provenance(
        artifact,
        record_id=trial_id,
        locator_kind="json_pointer",
        location=location,
        raw_value=raw_value,
        rule="ctgov-json-markup-text-v1",
    )
    if not isinstance(raw_criteria, str):
        diagnostic = ConnectorDiagnostic(
            code="invalid-eligibility-criteria",
            severity="field_nonfatal",
            message="eligibilityCriteria must be a markup string",
            record_id=trial_id,
            field="eligibilityCriteria",
            provenance=(provenance,),
        )
        return (), core, (*core_diagnostics, diagnostic)
    text = _clinicaltrials_markup_text(raw_criteria)
    if not text:
        return (), core, core_diagnostics
    return (
        (
            SemanticTextSection(
                role="eligibility",
                text=text,
                ordinal=0,
                provenance=(provenance,),
            ),
        ),
        core,
        core_diagnostics,
    )


def map_source_authored_eligibility_section(
    source: Mapping[str, object],
    *,
    artifact: SourceArtifact,
    trial_id: str,
) -> SemanticTextSection:
    """Map an explicit criterion source without deriving any new boundaries.

    This is intentionally a narrow source-to-contract seam for registries
    that already encode individual criteria.  A missing ``criteria`` array is
    an unstructured source blob, not permission to split its text.
    """

    text = source.get("text")
    if not isinstance(text, str) or not text:
        raise SchemaValidationError("source-authored eligibility text must be non-empty")
    raw_criteria = source.get("criteria")
    if raw_criteria is not None and not isinstance(raw_criteria, list):
        raise SchemaValidationError("source-authored criteria must be an array when supplied")
    text_provenance = _provenance(
        artifact,
        record_id=trial_id,
        locator_kind="json_pointer",
        location="/text",
        raw_value=text,
        rule="source-authored-criteria-text-v1",
    )
    criteria: list[CriterionItem] = []
    for ordinal, raw_criterion in enumerate(raw_criteria or []):
        if not isinstance(raw_criterion, Mapping):
            raise SchemaValidationError("source-authored criterion entries must be objects")
        identifier = raw_criterion.get("identifier")
        polarity = raw_criterion.get("polarity")
        criterion_text = raw_criterion.get("text")
        if (
            not isinstance(identifier, str)
            or not isinstance(polarity, str)
            or not isinstance(criterion_text, str)
        ):
            raise SchemaValidationError(
                "source-authored criterion requires identifier, polarity, and text"
            )
        if polarity not in {"inclusion", "exclusion"}:
            raise SchemaValidationError("source-authored criterion polarity is invalid")
        criteria.append(
            CriterionItem(
                ordinal=ordinal,
                identifier=identifier,
                polarity=polarity,  # type: ignore[arg-type]
                text=criterion_text,
                provenance=(
                    _provenance(
                        artifact,
                        record_id=trial_id,
                        locator_kind="json_pointer",
                        location=f"/criteria/{ordinal}",
                        raw_value=json.dumps(
                            raw_criterion, ensure_ascii=False, separators=(",", ":")
                        ),
                        rule="source-authored-criterion-v1",
                    ),
                ),
            )
        )
    return SemanticTextSection(
        role="eligibility",
        text=text,
        ordinal=0,
        provenance=(text_provenance,),
        criteria=tuple(criteria),
    )


def _recipe_trial_text_mapping(
    recipe: Mapping[str, JsonValue], source: str
) -> Mapping[str, JsonValue]:
    dispositions = recipe["field_disposition_table"]
    if not isinstance(dispositions, list):
        raise SchemaValidationError("frozen Source Recipe field dispositions are invalid")
    for disposition in dispositions:
        if isinstance(disposition, Mapping) and disposition.get("source") == source:
            return cast(Mapping[str, JsonValue], disposition)
    raise SchemaValidationError(f"frozen Source Recipe has no mapping for {source}")


def _recipe_trial_text_recipe(recipe: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    rendering = recipe.get("canonical_trial_text_recipe")
    if not isinstance(rendering, Mapping):
        raise SchemaValidationError("frozen Source Recipe trial rendering is invalid")
    return cast(Mapping[str, JsonValue], rendering)


def validate_trec_ct_2021_recipe_parity(
    recipe: Mapping[str, JsonValue] | None = None,
) -> None:
    """Reject a recipe that no longer attests the connector's executed mapping."""

    recipe_definition = SOURCE_RECIPE if recipe is None else recipe
    expected_patient_rendering = {
        "id": "trec-ct-2021-canonical-patient-text-v1",
        "source_evidence_kind": "narrative",
        "whitespace": "collapse Unicode whitespace to one ASCII space and strip",
        "unicode_normalization": "none; preserve source code points",
        "separator": "LF",
        "omission": "narrative is required; do not extract typed patient facts",
    }
    expected_trial_rendering = {
        "id": "trec-ct-2021-canonical-trial-text-v1",
        "whitespace": "collapse Unicode whitespace to one ASCII space and strip",
        "unicode_normalization": "none; preserve source code points",
        "source_markup": "exclude XML markup; retain decoded text content",
        "separator": "LF",
        "repetition": "preserve every non-empty source item in source order within role",
        "omission": "omit absent sections and typed fields without defaults",
    }
    patient_recipe = recipe_definition.get("canonical_patient_text_recipe")
    if not isinstance(patient_recipe, Mapping) or any(
        patient_recipe.get(key) != value for key, value in expected_patient_rendering.items()
    ):
        raise SchemaValidationError("frozen Source Recipe patient rendering is out of parity")

    section_sources = tuple(item[0] for item in _EXECUTED_SECTION_MAPPINGS)
    typed_sources = tuple(item[0] for item in _EXECUTED_TYPED_MAPPINGS)
    trial_rendering = _recipe_trial_text_recipe(recipe_definition)
    if any(trial_rendering.get(key) != value for key, value in expected_trial_rendering.items()):
        raise SchemaValidationError("frozen Source Recipe trial rendering is out of parity")
    section_order = trial_rendering.get("section_order")
    typed_order = trial_rendering.get("typed_field_order")
    section_labels = trial_rendering.get("section_labels")
    typed_labels = trial_rendering.get("typed_labels")
    if not all(isinstance(value, list) for value in (section_order, typed_order)) or not all(
        isinstance(value, Mapping) for value in (section_labels, typed_labels)
    ):
        raise SchemaValidationError("frozen Source Recipe rendering shape is invalid")
    expected_section_mappings = [item[1:] for item in _EXECUTED_SECTION_MAPPINGS]
    expected_typed_mappings = [item[1:] for item in _EXECUTED_TYPED_MAPPINGS]
    mapping_fields = ("target", "disposition", "normalization_rule")
    section_mappings = [
        tuple(
            _recipe_trial_text_mapping(recipe_definition, source).get(field)
            for field in mapping_fields
        )
        for source in section_sources
    ]
    typed_mappings = [
        tuple(
            _recipe_trial_text_mapping(recipe_definition, source).get(field)
            for field in mapping_fields
        )
        for source in typed_sources
    ]
    expected_section_order = [
        "brief_title",
        "official_title",
        "summary",
        "condition",
        "intervention",
        "eligibility",
    ]
    expected_typed_order = ["minimum_age", "maximum_age", "sex", "healthy_volunteers"]
    if section_mappings != expected_section_mappings or section_order != expected_section_order:
        raise SchemaValidationError("frozen Source Recipe section mapping is out of parity")
    if set(cast(Mapping[str, object], section_labels)) != set(expected_section_order):
        raise SchemaValidationError("frozen Source Recipe section mapping is out of parity")
    if typed_mappings != expected_typed_mappings or typed_order != expected_typed_order:
        raise SchemaValidationError("frozen Source Recipe typed mapping is out of parity")
    if set(cast(Mapping[str, object], typed_labels)) != set(expected_typed_order):
        raise SchemaValidationError("frozen Source Recipe typed mapping is out of parity")


_NormalizedT = TypeVar("_NormalizedT")


def _normalize_repeated(
    eligibility: ET.Element | None,
    *,
    name: str,
    trial_id: str,
    artifact: SourceArtifact,
    rule: str,
    normalize: Callable[[str], _NormalizedT | None],
    diagnostic_code: str,
    lexical: _XmlLexicalContent,
) -> tuple[
    tuple[tuple[_NormalizedT, FieldProvenance], ...],
    tuple[ConnectorDiagnostic, ...],
]:
    normalized: list[tuple[_NormalizedT, FieldProvenance]] = []
    diagnostics: list[ConnectorDiagnostic] = []
    for index, element in enumerate(_children(eligibility, name), start=1):
        raw = _element_raw_text(element)
        provenance = _provenance(
            artifact,
            record_id=trial_id,
            locator_kind="xml_path",
            location=f"/clinical_study/eligibility/{name}[{index}]",
            raw_value=lexical.raw(element),
            rule=rule,
        )
        try:
            value = normalize(raw)
        except (SchemaValidationError, ValueError) as exc:
            diagnostics.append(
                ConnectorDiagnostic(
                    code=diagnostic_code,
                    severity="field_nonfatal",
                    message=str(exc),
                    record_id=trial_id,
                    field=name,
                    provenance=(provenance,),
                )
            )
            continue
        if value is not None:
            normalized.append((value, provenance))
    return tuple(normalized), tuple(diagnostics)


def _coalesce(
    assertions: tuple[tuple[_NormalizedT, FieldProvenance], ...],
    *,
    field_name: str,
) -> tuple[_NormalizedT | None, tuple[FieldProvenance, ...], CoreFieldConflict | None]:
    if not assertions:
        return None, (), None
    first = assertions[0][0]
    provenance = tuple(item[1] for item in assertions)
    if all(item[0] == first for item in assertions[1:]):
        return first, provenance, None
    return (
        None,
        (),
        CoreFieldConflict(
            field_name=field_name,
            values=tuple(
                cast(JsonValue, value.to_dict() if isinstance(value, AgeQuantity) else value)
                for value, _ in assertions
            ),
            provenance=provenance,
        ),
    )


def _conflict_diagnostic(
    conflict: CoreFieldConflict,
    *,
    trial_id: str,
) -> ConnectorDiagnostic:
    return ConnectorDiagnostic(
        code=f"conflicting-{conflict.field_name.replace('_', '-')}",
        severity="field_nonfatal",
        message=f"conflicting source assertions for {conflict.field_name}",
        record_id=trial_id,
        field=conflict.field_name,
        provenance=conflict.provenance,
    )


def _typed_core(
    eligibility: ET.Element | None,
    *,
    trial_id: str,
    artifact: SourceArtifact,
    lexical: _XmlLexicalContent,
) -> tuple[TypedClinicalCore | None, tuple[ConnectorDiagnostic, ...]]:
    diagnostics: list[ConnectorDiagnostic] = []
    conflicts: list[CoreFieldConflict] = []

    def rule_for(source: str) -> str:
        rule = _recipe_trial_text_mapping(SOURCE_RECIPE, source).get("normalization_rule")
        if not isinstance(rule, str):
            raise SchemaValidationError(f"frozen Source Recipe mapping for {source} is invalid")
        return rule

    minimum_rows, minimum_diagnostics = _normalize_repeated(
        eligibility,
        name="minimum_age",
        trial_id=trial_id,
        artifact=artifact,
        rule=rule_for("clinical_study/eligibility/minimum_age"),
        normalize=lambda raw: normalize_ctgov_age(raw, maximum=False),
        diagnostic_code="invalid-minimum-age",
        lexical=lexical,
    )
    diagnostics.extend(minimum_diagnostics)
    minimum, minimum_provenance, conflict = _coalesce(
        minimum_rows,
        field_name="minimum_age",
    )
    if conflict is not None:
        conflicts.append(conflict)
        diagnostics.append(_conflict_diagnostic(conflict, trial_id=trial_id))

    maximum_rows, maximum_diagnostics = _normalize_repeated(
        eligibility,
        name="maximum_age",
        trial_id=trial_id,
        artifact=artifact,
        rule=rule_for("clinical_study/eligibility/maximum_age"),
        normalize=lambda raw: normalize_ctgov_age(raw, maximum=True),
        diagnostic_code="invalid-maximum-age",
        lexical=lexical,
    )
    diagnostics.extend(maximum_diagnostics)
    maximum, maximum_provenance, conflict = _coalesce(
        maximum_rows,
        field_name="maximum_age",
    )
    if conflict is not None:
        conflicts.append(conflict)
        diagnostics.append(_conflict_diagnostic(conflict, trial_id=trial_id))

    sex_rows, sex_diagnostics = _normalize_repeated(
        eligibility,
        name="gender",
        trial_id=trial_id,
        artifact=artifact,
        rule=rule_for("clinical_study/eligibility/gender"),
        normalize=normalize_ctgov_sex,
        diagnostic_code="unrecognized-sex",
        lexical=lexical,
    )
    diagnostics.extend(sex_diagnostics)
    sex, sex_provenance, conflict = _coalesce(
        sex_rows,
        field_name="sex",
    )
    if conflict is not None:
        conflicts.append(conflict)
        diagnostics.append(_conflict_diagnostic(conflict, trial_id=trial_id))

    healthy_rows, healthy_diagnostics = _normalize_repeated(
        eligibility,
        name="healthy_volunteers",
        trial_id=trial_id,
        artifact=artifact,
        rule=rule_for("clinical_study/eligibility/healthy_volunteers"),
        normalize=normalize_ctgov_healthy_volunteers,
        diagnostic_code="invalid-healthy-volunteers",
        lexical=lexical,
    )
    diagnostics.extend(healthy_diagnostics)
    healthy, healthy_provenance, conflict = _coalesce(
        healthy_rows,
        field_name="healthy_volunteers",
    )
    if conflict is not None:
        conflicts.append(conflict)
        diagnostics.append(_conflict_diagnostic(conflict, trial_id=trial_id))

    core = TypedClinicalCore(
        minimum_age=(
            AgeBound(cast(AgeQuantity, minimum), minimum_provenance)
            if minimum is not None
            else None
        ),
        maximum_age=(
            AgeBound(cast(AgeQuantity | str, maximum), maximum_provenance)  # type: ignore[arg-type]
            if maximum is not None
            else None
        ),
        sex=SexEligibility(cast(str, sex), sex_provenance) if sex is not None else None,
        healthy_volunteers=(
            HealthyVolunteerAcceptance(cast(bool, healthy), healthy_provenance)
            if healthy is not None
            else None
        ),
        conflicts=tuple(conflicts),
    )
    if not core.to_dict():
        return None, tuple(diagnostics)
    return core, tuple(diagnostics)


def _section_provenance(
    artifact: SourceArtifact,
    *,
    trial_id: str,
    location: str,
    raw: str,
    rule: str = "ctgov-xml-text-whitespace-v1",
) -> tuple[FieldProvenance, ...]:
    return (
        _provenance(
            artifact,
            record_id=trial_id,
            locator_kind="xml_path",
            location=location,
            raw_value=raw,
            rule=rule,
        ),
    )


def _intervention_text(intervention: ET.Element) -> str:
    values = {
        "TYPE": _element_text(_child(intervention, "intervention_type")),
        "NAME": _element_text(_child(intervention, "intervention_name")),
        "DESCRIPTION": _element_text(_child(intervention, "description")),
    }
    return "; ".join(f"{name}={value}" for name, value in values.items() if value)


def _sections(
    root: ET.Element,
    *,
    trial_id: str,
    artifact: SourceArtifact,
    lexical: _XmlLexicalContent,
    include_complete_eligibility_text: bool,
) -> tuple[SemanticTextSection, ...]:
    sections: list[SemanticTextSection] = []

    def append(
        source: str,
        text: str,
        location: str,
        *,
        raw_value: str | None = None,
        source_text: str | None = None,
    ) -> None:
        if not text:
            return
        mapping = _recipe_trial_text_mapping(SOURCE_RECIPE, source)
        role = mapping.get("target")
        rule = mapping.get("normalization_rule")
        if not isinstance(role, str) or not isinstance(rule, str):
            raise SchemaValidationError(f"frozen Source Recipe mapping for {source} is invalid")
        source_bytes = source_text.encode("utf-8") if source_text is not None else None
        sections.append(
            SemanticTextSection(
                role=role,
                text=text,
                ordinal=len(sections),
                provenance=_section_provenance(
                    artifact,
                    trial_id=trial_id,
                    location=location,
                    raw=text if raw_value is None else raw_value,
                    rule=rule,
                ),
                source_text=source_text,
                source_text_sha256=(
                    "sha256:" + hashlib.sha256(source_bytes).hexdigest()
                    if source_bytes is not None
                    else None
                ),
                source_text_byte_length=(len(source_bytes) if source_bytes is not None else None),
            )
        )

    brief_title = _child(root, "brief_title")
    append(
        "clinical_study/brief_title",
        _element_text(brief_title),
        "/clinical_study/brief_title[1]",
        raw_value=lexical.raw(brief_title),
    )
    official_title = _child(root, "official_title")
    append(
        "clinical_study/official_title",
        _element_text(official_title),
        "/clinical_study/official_title[1]",
        raw_value=lexical.raw(official_title),
    )
    summary = _child(root, "brief_summary")
    summary_text = _child(summary, "textblock")
    append(
        "clinical_study/brief_summary/textblock",
        _element_text(summary_text),
        "/clinical_study/brief_summary[1]/textblock[1]",
        raw_value=lexical.raw(summary_text),
    )
    description = _child(root, "detailed_description")
    description_text = _child(description, "textblock")
    append(
        "clinical_study/detailed_description/textblock",
        _element_text(description_text),
        "/clinical_study/detailed_description[1]/textblock[1]",
        raw_value=lexical.raw(description_text),
    )
    for index, condition in enumerate(_children(root, "condition"), start=1):
        append(
            "clinical_study/condition",
            _element_text(condition),
            f"/clinical_study/condition[{index}]",
            raw_value=lexical.raw(condition),
        )
    for index, intervention in enumerate(_children(root, "intervention"), start=1):
        append(
            "clinical_study/intervention",
            _intervention_text(intervention),
            f"/clinical_study/intervention[{index}]",
            raw_value=lexical.raw(intervention),
        )
    eligibility = _child(root, "eligibility")
    criteria = _child(eligibility, "criteria")
    eligibility_text = _child(criteria, "textblock")
    append(
        "clinical_study/eligibility/criteria/textblock",
        _element_text(eligibility_text),
        "/clinical_study/eligibility[1]/criteria[1]/textblock[1]",
        raw_value=lexical.raw(eligibility_text),
        source_text=(
            _element_raw_text(eligibility_text) if include_complete_eligibility_text else None
        ),
    )
    return tuple(sections)


def render_canonical_trial_text(
    sections: tuple[SemanticTextSection, ...],
    core: TypedClinicalCore | None,
) -> str:
    recipe = _recipe_trial_text_recipe(SOURCE_RECIPE)
    section_order = cast(list[str], recipe["section_order"])
    section_labels = cast(Mapping[str, str], recipe["section_labels"])
    typed_order = cast(list[str], recipe["typed_field_order"])
    typed_labels = cast(Mapping[str, str], recipe["typed_labels"])
    unbounded_token = cast(str, recipe["unbounded_token"])
    boolean_tokens = cast(Mapping[str, str], recipe["boolean_tokens"])
    separator_name = recipe.get("separator")
    separators = {"LF": "\n"}
    if not isinstance(separator_name, str) or separator_name not in separators:
        raise SchemaValidationError("frozen Source Recipe has an unsupported text separator")
    separator = separators[separator_name]
    lines = [
        f"{section_labels[section.role]} {section.text}"
        for role in section_order
        for section in sections
        if section.role == role
    ]
    if core is not None:
        values: Mapping[str, object] = {
            "minimum_age": core.minimum_age,
            "maximum_age": core.maximum_age,
            "sex": core.sex,
            "healthy_volunteers": core.healthy_volunteers,
        }
        for field_name in typed_order:
            value = values.get(field_name)
            if value is None:
                continue
            if field_name in {"minimum_age", "maximum_age"}:
                bound = cast(AgeBound, value)
                rendered = (
                    unbounded_token
                    if bound.value == "unbounded"
                    else f"{bound.value.amount} {bound.value.unit}"
                )
            elif field_name == "sex":
                rendered = cast(SexEligibility, value).value
            elif field_name == "healthy_volunteers":
                rendered = boolean_tokens[
                    "true" if cast(HealthyVolunteerAcceptance, value).value else "false"
                ]
            else:
                raise SchemaValidationError(
                    f"frozen Source Recipe has unsupported typed field {field_name}"
                )
            lines.append(f"{typed_labels[field_name]} {rendered}")
    return separator.join(lines)


def _nct_aliases(root: ET.Element, *, trial_id: str) -> tuple[str, ...]:
    id_info = _child(root, "id_info")
    aliases = {
        alias
        for element in _children(id_info, "nct_alias")
        if (alias := _element_text(element)) != trial_id and _NCT_ID.fullmatch(alias) is not None
    }
    return tuple(sorted(aliases))


def extract_trial_nct_aliases(serialized: bytes) -> tuple[str, ...]:
    """Return valid source-authored NCT aliases without mapping the full trial record."""

    try:
        root = safe_xml.fromstring(serialized)
    except (safe_xml.UnsafeXmlError, ET.ParseError):
        return ()
    if _local_name(root.tag) != "clinical_study":
        return ()
    trial_id = _element_text(_child(_child(root, "id_info"), "nct_id"))
    if _NCT_ID.fullmatch(trial_id) is None:
        return ()
    return _nct_aliases(root, trial_id=trial_id)


def _parse_trial_xml_with_aliases(
    serialized: bytes,
    *,
    artifact: SourceArtifact,
    member_name: str,
    include_complete_eligibility_text: bool = False,
) -> tuple[TrialDocument, tuple[ConnectorDiagnostic, ...], tuple[str, ...]]:
    fallback_trial_id = _source_trial_id(member_name)
    try:
        root = safe_xml.fromstring(serialized)
    except safe_xml.UnsafeXmlError as exc:
        raise MalformedTrialError(f"unsafe trial XML: {exc}", trial_id=fallback_trial_id) from exc
    except ET.ParseError as exc:
        raise MalformedTrialError(
            f"malformed trial XML: {exc}", trial_id=fallback_trial_id
        ) from exc
    if _local_name(root.tag) != "clinical_study":
        raise MalformedTrialError(
            "trial XML root must be <clinical_study>", trial_id=fallback_trial_id
        )
    lexical = _XmlLexicalContent(serialized, root)
    trial_id = _element_text(_child(_child(root, "id_info"), "nct_id"))
    if _NCT_ID.fullmatch(trial_id) is None:
        raise MalformedTrialError(
            f"invalid or missing NCT ID {trial_id!r}", trial_id=trial_id or fallback_trial_id
        )
    if fallback_trial_id is not None and fallback_trial_id != trial_id:
        raise MalformedTrialError(
            f"XML NCT ID {trial_id} does not match member name {fallback_trial_id}",
            trial_id=trial_id,
        )
    aliases = _nct_aliases(root, trial_id=trial_id)
    sections = _sections(
        root,
        trial_id=trial_id,
        artifact=artifact,
        lexical=lexical,
        include_complete_eligibility_text=include_complete_eligibility_text,
    )
    core, diagnostics = _typed_core(
        _child(root, "eligibility"),
        trial_id=trial_id,
        artifact=artifact,
        lexical=lexical,
    )
    canonical_text = render_canonical_trial_text(sections, core)
    if not canonical_text:
        raise MalformedTrialError("trial has no canonical input text", trial_id=trial_id)
    trial = TrialDocument(
        trial_id=trial_id,
        source_identity=SourceRecordIdentity(
            namespace="clinicaltrials.gov",
            record_id=trial_id,
        ),
        canonical_text=canonical_text,
        sections=sections,
        typed_clinical_core=core,
    )
    return trial, diagnostics, aliases


def parse_trial_xml(
    serialized: bytes,
    *,
    artifact: SourceArtifact,
    member_name: str,
    include_complete_eligibility_text: bool = False,
) -> tuple[TrialDocument, tuple[ConnectorDiagnostic, ...]]:
    trial, diagnostics, _aliases = _parse_trial_xml_with_aliases(
        serialized,
        artifact=artifact,
        member_name=member_name,
        include_complete_eligibility_text=include_complete_eligibility_text,
    )
    return trial, diagnostics


def _record_fatal_diagnostic(
    exc: MalformedTrialError,
    *,
    artifact: SourceArtifact,
    member_name: str,
    serialized: bytes,
) -> ConnectorDiagnostic:
    record_id = exc.trial_id or _source_trial_id(member_name)
    provenance = _provenance(
        artifact,
        record_id=record_id or member_name,
        locator_kind="archive_member",
        location=f"zip-member:{member_name}",
        raw_value=serialized.decode("utf-8", errors="replace"),
        rule="ctgov-xml-record-parse-v1",
    )
    return ConnectorDiagnostic(
        code="malformed-trial-record",
        severity="record_fatal",
        message=str(exc),
        record_id=record_id,
        field=None,
        provenance=(provenance,),
    )


def _parse_trials(
    source_directory: Path,
    *,
    source_bundle: SourceBundle,
) -> tuple[tuple[TrialDocument, ...], tuple[ConnectorDiagnostic, ...]]:
    trials: dict[str, TrialDocument] = {}
    diagnostics: list[ConnectorDiagnostic] = []
    for artifact in source_bundle.artifacts:
        if not artifact.role.startswith("trials_part_"):
            continue
        try:
            archive = zipfile.ZipFile(source_directory / artifact.filename)
        except zipfile.BadZipFile as exc:
            raise IngestionError(f"invalid ZIP archive {artifact.filename}: {exc}") from exc
        with archive:
            for member in sorted(archive.infolist(), key=lambda item: item.filename):
                if member.is_dir() or not member.filename.casefold().endswith(".xml"):
                    continue
                with archive.open(member) as stream:
                    serialized = stream.read()
                try:
                    trial, field_diagnostics = parse_trial_xml(
                        serialized,
                        artifact=artifact,
                        member_name=member.filename,
                    )
                except MalformedTrialError as exc:
                    diagnostics.append(
                        _record_fatal_diagnostic(
                            exc,
                            artifact=artifact,
                            member_name=member.filename,
                            serialized=serialized,
                        )
                    )
                    continue
                if trial.trial_id in trials:
                    raise IngestionError(f"duplicate trial ID {trial.trial_id}")
                trials[trial.trial_id] = trial
                diagnostics.extend(field_diagnostics)
    if not trials:
        raise IngestionError("trial archives contain no valid XML records")
    return tuple(trials[key] for key in sorted(trials)), tuple(diagnostics)


def _validate_relations(
    topics: tuple[BenchmarkTopic, ...],
    trials: tuple[TrialDocument, ...],
    judgments: tuple[RelevanceJudgment, ...],
) -> None:
    topic_ids = {item.topic_id for item in topics}
    trial_ids = {item.trial_id for item in trials}
    for judgment in judgments:
        if judgment.topic_id not in topic_ids:
            raise IngestionError(f"Judgment references unknown topic {judgment.topic_id}")
        if judgment.trial_id not in trial_ids:
            raise IngestionError(f"Judgment references unknown trial {judgment.trial_id}")


def prepare_trec_ct_2021(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    source_lock_path: str | Path | None = None,
    created_at: datetime | None = None,
) -> PreparationResult:
    """Verify frozen TREC 2021 sources and atomically prepare a Snapshot."""

    source = Path(source_directory)
    if not source.is_dir():
        raise SourceVerificationError(f"source directory does not exist: {source}")
    timestamp = created_at if created_at is not None else datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise IngestionError("created_at must include a UTC offset")
    validate_trec_ct_2021_recipe_parity()
    source_bundle = load_source_bundle_lock(source_lock_path)

    # No source parser runs until every Source Bundle artifact is verified.
    _verify_source_bundle(source, source_bundle)
    topic_artifact = source_bundle.artifact_by_id("topics")
    qrels_artifact = source_bundle.artifact_by_id("qrels")
    topics = parse_topics_xml(source / TOPICS_FILENAME, artifact=topic_artifact)
    trials, diagnostics = _parse_trials(source, source_bundle=source_bundle)
    judgments = tuple(
        sorted(
            parse_qrels(source / QRELS_FILENAME),
            key=lambda item: (item.topic_id, item.trial_id),
        )
    )
    _validate_relations(topics, trials, judgments)

    available_capabilities = frozenset(
        {
            CAPABILITY_CANONICAL_PATIENT_TEXT,
            CAPABILITY_CANONICAL_TRIAL_TEXT,
            CAPABILITY_PATIENT_EVIDENCE,
            CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
            CAPABILITY_TYPED_TRIAL_CORE,
            CAPABILITY_FIELD_PROVENANCE,
        }
    )
    snapshot = BenchmarkSnapshot(
        benchmark_lineage=BENCHMARK_LINEAGE,
        snapshot_name=SNAPSHOT_NAME,
        topics=topics,
        trials=trials,
        available_capabilities=available_capabilities,
    )
    evaluation_package = EvaluationPackage(
        benchmark_lineage=BENCHMARK_LINEAGE,
        task_id="trec-clinical-trials-patient-to-trial-ranking",
        snapshot_id=snapshot.snapshot_id,
        judgments=judgments,
        provenance={
            "source_artifact_id": qrels_artifact.artifact_id,
            "source_artifact_sha256": qrels_artifact.sha256,
            "format": "TREC four-column Judgments",
            "parser": "trec-four-column-qrels-v1",
            "transformation_rule": "trec-ct-2021-qrels-v1",
        },
    )
    return write_prepared_benchmark(
        output_directory,
        dataset_id=DATASET_ID,
        snapshot=snapshot,
        evaluation_package=evaluation_package,
        source_bundle=source_bundle,
        source_recipe=SOURCE_RECIPE,
        diagnostics=diagnostics,
        created_at=timestamp,
        build_provenance={
            "connector_name": CONNECTOR_NAME,
            "connector_version": CONNECTOR_VERSION,
            "source_lock_sha256": source_bundle.lock_sha256,
        },
    )


__all__ = [
    "BENCHMARK_LINEAGE",
    "CANONICAL_PATIENT_TEXT_RECIPE",
    "CANONICAL_TEXT_RECIPE",
    "CONNECTOR_NAME",
    "CONNECTOR_VERSION",
    "DATASET_ID",
    "FIELD_DISPOSITION_TABLE",
    "SNAPSHOT_NAME",
    "SOURCE_RECIPE",
    "SOURCE_RECIPE_ID",
    "load_source_bundle_lock",
    "map_ctgov_json_eligibility",
    "map_ctgov_json_trial_layers",
    "map_source_authored_eligibility_section",
    "normalize_ctgov_age",
    "normalize_ctgov_healthy_volunteers",
    "normalize_ctgov_sex",
    "parse_topics_xml",
    "parse_trial_xml",
    "prepare_trec_ct_2021",
    "render_canonical_trial_text",
    "validate_trec_ct_2021_recipe_parity",
]
