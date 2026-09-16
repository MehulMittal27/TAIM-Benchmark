"""Checksum-locked TREC Clinical Trials 2022 and 2023 connectors."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from taim import safe_xml
from taim.contracts import content_sha256
from taim.data.prepared import PreparationResult, write_prepared_benchmark
from taim.data.trec_ct_2021 import (
    CANONICAL_TEXT_RECIPE,
    FIELD_DISPOSITION_TABLE,
    IngestionError,
    MalformedTrialError,
    SourceVerificationError,
    _collapse_whitespace,
    _element_raw_text,
    _local_name,
    _parse_trial_xml_with_aliases,
    _provenance,
    _record_fatal_diagnostic,
    _source_artifact,
    _source_trial_id,
    _XmlLexicalContent,
    extract_trial_nct_aliases,
    parse_qrels,
)
from taim.evaluation_package import ConnectorDiagnostic, EvaluationPackage
from taim.file_hash import sha256_file
from taim.schemas import JsonValue, RelevanceJudgment
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    CAPABILITY_FIELD_PROVENANCE,
    CAPABILITY_PATIENT_EVIDENCE,
    CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
    CAPABILITY_TYPED_TRIAL_CORE,
    BenchmarkSnapshot,
    BenchmarkTopic,
    PatientEvidenceItem,
    SourceRecordIdentity,
    TrialDocument,
)
from taim.source import SourceArtifact, SourceBundle

SOURCE_BUNDLE_LOCK_VERSION = "2.0"
CONNECTOR_NAME = "taim.data.trec_ct"
CONNECTOR_VERSION = "1.1"

TopicFormat = Literal["narrative", "questionnaire"]


@dataclass(frozen=True, slots=True)
class TrecClinicalTrialsTrack:
    dataset_id: str
    year: int
    topics_filename: str
    qrels_filename: str
    trial_archive_filenames: tuple[str, ...]
    expected_topic_task: str
    topic_format: TopicFormat

    @property
    def benchmark_lineage(self) -> str:
        return self.dataset_id

    @property
    def snapshot_name(self) -> str:
        return self.dataset_id

    @property
    def source_recipe_id(self) -> str:
        return f"{self.dataset_id}-snapshot-recipe-v2"

    @property
    def archive_roles(self) -> tuple[str, ...]:
        return tuple(
            f"trials_part_{index}" for index in range(1, len(self.trial_archive_filenames) + 1)
        )

    @property
    def expected_source_roles(self) -> tuple[str, ...]:
        return ("topics", "qrels", *self.archive_roles)

    @property
    def expected_filenames(self) -> Mapping[str, str]:
        return {
            "topics": self.topics_filename,
            "qrels": self.qrels_filename,
            **dict(zip(self.archive_roles, self.trial_archive_filenames, strict=True)),
        }

    @property
    def default_lock_path(self) -> Path:
        return Path(__file__).with_name("locks") / f"{self.dataset_id}.json"


TRACKS: Mapping[str, TrecClinicalTrialsTrack] = {
    "trec-ct-2022": TrecClinicalTrialsTrack(
        dataset_id="trec-ct-2022",
        year=2022,
        topics_filename="topics2022.xml",
        qrels_filename="qrels2022.txt",
        trial_archive_filenames=tuple(
            f"ClinicalTrials.2021-04-27.part{part}.zip" for part in range(1, 6)
        ),
        # The official file retains this 2021 task attribute even though it contains 2022 topics.
        expected_topic_task="2021 TREC Clinical Trials",
        topic_format="narrative",
    ),
    "trec-ct-2023": TrecClinicalTrialsTrack(
        dataset_id="trec-ct-2023",
        year=2023,
        topics_filename="topics2023.xml",
        qrels_filename="qrels2023.txt",
        trial_archive_filenames=tuple(
            f"ClinicalTrials.2023-05-08.trials{part}.zip" for part in range(6)
        ),
        expected_topic_task="2023 TREC Clinical Trials",
        topic_format="questionnaire",
    ),
}


def _load_source_bundle_lock(
    track: TrecClinicalTrialsTrack,
    path: str | Path | None,
) -> SourceBundle:
    lock_path = Path(path) if path is not None else track.default_lock_path
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestionError(f"invalid Source Bundle lock JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise IngestionError("Source Bundle lock must be a JSON object")
    expected_fields = {"schema_version", "dataset_id", "lock_status", "sources"}
    if set(payload) != expected_fields:
        raise IngestionError("Source Bundle lock has unexpected or missing fields")
    if payload.get("schema_version") != SOURCE_BUNDLE_LOCK_VERSION:
        raise IngestionError("unsupported Source Bundle lock schema version")
    if payload.get("dataset_id") != track.dataset_id:
        raise IngestionError(f"Source Bundle dataset_id must be {track.dataset_id!r}")
    if payload.get("lock_status") != "complete":
        raise IngestionError("Source Bundle lock must be complete")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or any(
        not isinstance(item, Mapping) for item in raw_sources
    ):
        raise IngestionError("Source Bundle sources must be an array of objects")
    artifacts = tuple(_source_artifact(cast(Mapping[str, object], item)) for item in raw_sources)
    by_role = {artifact.role: artifact for artifact in artifacts}
    if len(by_role) != len(artifacts) or set(by_role) != set(track.expected_source_roles):
        raise IngestionError(f"Source Bundle source roles do not match {track.dataset_id}")
    for role, filename in track.expected_filenames.items():
        if by_role[role].filename != filename:
            raise IngestionError(f"Source Bundle role {role!r} must use {filename!r}")
    return SourceBundle(
        artifacts=tuple(by_role[role] for role in track.expected_source_roles),
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
        elif sha256_file(path) != artifact.sha256:
            errors.append(f"{artifact.filename}: SHA-256 does not match Source Bundle lock")
    if errors:
        raise SourceVerificationError("Source Bundle verification failed: " + "; ".join(errors))


def _questionnaire_text(topic: ET.Element) -> str:
    template = _collapse_whitespace(topic.attrib.get("template", ""))
    if not template:
        raise IngestionError("questionnaire topic template must be non-empty")
    lines = [f"[TEMPLATE] {template}"]
    for field in topic:
        if _local_name(field.tag) != "field":
            raise IngestionError("questionnaire topic contains an unexpected element")
        name = _collapse_whitespace(field.attrib.get("name", ""))
        if not name:
            raise IngestionError("questionnaire field name must be non-empty")
        value = _collapse_whitespace(_element_raw_text(field))
        if value:
            lines.append(f"[FIELD] {name}: {value}")
    return "\n".join(lines)


def parse_topics_xml(
    path: str | Path,
    *,
    artifact: SourceArtifact,
    track: TrecClinicalTrialsTrack,
) -> tuple[BenchmarkTopic, ...]:
    serialized = Path(path).read_bytes()
    try:
        root = safe_xml.fromstring(serialized)
    except safe_xml.UnsafeXmlError as exc:
        raise IngestionError(f"unsafe topics XML: {exc}") from exc
    except ET.ParseError as exc:
        raise IngestionError(f"malformed topics XML: {exc}") from exc
    if _local_name(root.tag) != "topics":
        raise IngestionError("topics XML root must be <topics>")
    if root.attrib.get("task") != track.expected_topic_task:
        raise IngestionError("topics XML task does not match the frozen track lock")
    lexical = _XmlLexicalContent(serialized, root)
    topics: list[BenchmarkTopic] = []
    seen: set[str] = set()
    for element in root:
        if _local_name(element.tag) != "topic":
            raise IngestionError("topics XML contains an unexpected element")
        topic_id = element.attrib.get("number")
        if not isinstance(topic_id, str) or not topic_id.isdecimal() or int(topic_id) < 1:
            raise IngestionError("topic number must be a positive decimal integer")
        if topic_id in seen:
            raise IngestionError("topics XML contains a duplicate topic number")
        text = (
            _collapse_whitespace(_element_raw_text(element))
            if track.topic_format == "narrative"
            else _questionnaire_text(element)
        )
        if not text:
            raise IngestionError("topics XML contains empty patient text")
        rule = (
            "trec-topic-narrative-whitespace-v1"
            if track.topic_format == "narrative"
            else "trec-2023-questionnaire-canonical-text-v1"
        )
        provenance = _provenance(
            artifact,
            record_id=topic_id,
            locator_kind="xml_path",
            location=f"/topics/topic[@number='{topic_id}']",
            raw_value=lexical.raw(element),
            rule=rule,
        )
        topics.append(
            BenchmarkTopic(
                topic_id=topic_id,
                source_identity=SourceRecordIdentity(
                    namespace=f"{track.dataset_id}.topics",
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


def _select_topics(
    topics: tuple[BenchmarkTopic, ...],
    judgments: tuple[RelevanceJudgment, ...],
    smoke_topics: int | None,
) -> tuple[
    tuple[BenchmarkTopic, ...],
    tuple[RelevanceJudgment, ...],
    frozenset[str] | None,
]:
    if smoke_topics is None:
        return topics, judgments, None
    if smoke_topics < 1 or smoke_topics > len(topics):
        raise IngestionError("smoke_topics must be between one and the track topic count")
    selected = tuple(sorted(topics, key=lambda topic: int(topic.topic_id))[:smoke_topics])
    selected = tuple(sorted(selected, key=lambda topic: topic.topic_id))
    selected_ids = {topic.topic_id for topic in selected}
    selected_judgments = tuple(
        judgment for judgment in judgments if judgment.topic_id in selected_ids
    )
    if not selected_judgments:
        raise IngestionError("smoke topic selection contains no Judgments")
    trial_ids = frozenset(judgment.trial_id for judgment in selected_judgments)
    return selected, selected_judgments, trial_ids


def _parse_trials(
    source_directory: Path,
    *,
    source_bundle: SourceBundle,
    archive_roles: frozenset[str],
    include_trial_ids: frozenset[str] | None,
) -> tuple[
    tuple[TrialDocument, ...],
    tuple[ConnectorDiagnostic, ...],
    Mapping[str, str],
]:
    trials: dict[str, TrialDocument] = {}
    diagnostics: list[ConnectorDiagnostic] = []
    alias_to_trial_id: dict[str, str] = {}
    unresolved_selected_aliases: frozenset[str] = frozenset()
    if include_trial_ids is not None:
        direct_trial_ids: set[str] = set()
        for artifact in source_bundle.artifacts:
            if artifact.role not in archive_roles:
                continue
            try:
                with zipfile.ZipFile(source_directory / artifact.filename) as archive:
                    direct_trial_ids.update(
                        trial_id
                        for member in archive.infolist()
                        if not member.is_dir()
                        and member.filename.casefold().endswith(".xml")
                        and (trial_id := _source_trial_id(member.filename)) is not None
                    )
            except zipfile.BadZipFile as exc:
                raise IngestionError(f"invalid ZIP archive {artifact.filename}: {exc}") from exc
        unresolved_selected_aliases = frozenset(include_trial_ids - direct_trial_ids)
    for artifact in source_bundle.artifacts:
        if artifact.role not in archive_roles:
            continue
        try:
            archive = zipfile.ZipFile(source_directory / artifact.filename)
        except zipfile.BadZipFile as exc:
            raise IngestionError(f"invalid ZIP archive {artifact.filename}: {exc}") from exc
        with archive:
            for member in sorted(archive.infolist(), key=lambda item: item.filename):
                if member.is_dir() or not member.filename.casefold().endswith(".xml"):
                    continue
                member_trial_id = _source_trial_id(member.filename)
                directly_selected = (
                    include_trial_ids is None or member_trial_id in include_trial_ids
                )
                if not directly_selected and not unresolved_selected_aliases:
                    continue
                with archive.open(member) as stream:
                    serialized = stream.read()
                if not directly_selected and not (
                    unresolved_selected_aliases.intersection(extract_trial_nct_aliases(serialized))
                ):
                    continue
                try:
                    trial, field_diagnostics, aliases = _parse_trial_xml_with_aliases(
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
                    raise IngestionError("trial archives contain a duplicate trial ID")
                trials[trial.trial_id] = trial
                for alias in aliases:
                    existing = alias_to_trial_id.get(alias)
                    if existing is not None and existing != trial.trial_id:
                        raise IngestionError("trial archives contain a conflicting NCT alias")
                    alias_to_trial_id[alias] = trial.trial_id
                diagnostics.extend(field_diagnostics)
    if not trials:
        raise IngestionError("trial archives contain no selected valid XML records")
    if any(
        alias in trials and alias != canonical for alias, canonical in alias_to_trial_id.items()
    ):
        raise IngestionError("trial archives contain an NCT alias that overlaps a canonical ID")
    return (
        tuple(trials[key] for key in sorted(trials)),
        tuple(diagnostics),
        dict(sorted(alias_to_trial_id.items())),
    )


def _normalize_judgment_trial_ids(
    judgments: tuple[RelevanceJudgment, ...],
    trials: tuple[TrialDocument, ...],
    alias_to_trial_id: Mapping[str, str],
) -> tuple[
    tuple[RelevanceJudgment, ...],
    int,
    int,
    tuple[tuple[str, str], ...],
]:
    trial_ids = {trial.trial_id for trial in trials}
    normalized: list[RelevanceJudgment] = []
    seen: dict[tuple[str, str], RelevanceJudgment] = {}
    resolved_judgment_count = 0
    collapsed_equal_duplicate_count = 0
    resolved_aliases: dict[str, str] = {}
    for judgment in judgments:
        trial_id = judgment.trial_id
        if trial_id not in trial_ids and trial_id in alias_to_trial_id:
            canonical_trial_id = alias_to_trial_id[trial_id]
            resolved_aliases[trial_id] = canonical_trial_id
            trial_id = canonical_trial_id
            resolved_judgment_count += 1
        key = (judgment.topic_id, trial_id)
        normalized_judgment = RelevanceJudgment(
            topic_id=judgment.topic_id,
            trial_id=trial_id,
            label=judgment.label,
        )
        existing = seen.get(key)
        if existing is not None:
            if existing.label != normalized_judgment.label:
                raise IngestionError(
                    "NCT alias normalization creates conflicting duplicate Judgments"
                )
            collapsed_equal_duplicate_count += 1
            continue
        normalized.append(normalized_judgment)
        seen[key] = normalized_judgment
    normalized.sort(key=lambda judgment: (judgment.topic_id, judgment.trial_id))
    return (
        tuple(normalized),
        resolved_judgment_count,
        collapsed_equal_duplicate_count,
        tuple(sorted(resolved_aliases.items())),
    )


def _validate_relations(
    topics: tuple[BenchmarkTopic, ...],
    trials: tuple[TrialDocument, ...],
    judgments: tuple[RelevanceJudgment, ...],
) -> None:
    topic_ids = {topic.topic_id for topic in topics}
    trial_ids = {trial.trial_id for trial in trials}
    for judgment in judgments:
        if judgment.topic_id not in topic_ids:
            raise IngestionError("Judgment references an unknown topic")
        if judgment.trial_id not in trial_ids:
            raise IngestionError("Judgment references a trial absent from the historical corpus")


def _source_recipe(
    track: TrecClinicalTrialsTrack,
    *,
    smoke_topics: int | None,
) -> dict[str, JsonValue]:
    canonical_trial_text = dict(CANONICAL_TEXT_RECIPE)
    canonical_trial_text["id"] = f"{track.dataset_id}-canonical-trial-text-v1"
    if track.topic_format == "narrative":
        topic_disposition: dict[str, JsonValue] = {
            "source": "topics/topic",
            "disposition": "patient_evidence",
            "target": "narrative",
            "normalization_rule": "trec-topic-narrative-whitespace-v1",
        }
        patient_recipe: dict[str, JsonValue] = {
            "id": f"{track.dataset_id}-canonical-patient-text-v1",
            "source_evidence_kind": "narrative",
            "whitespace": "collapse Unicode whitespace to one ASCII space and strip",
            "unicode_normalization": "none; preserve source code points",
            "separator": "LF",
            "omission": "narrative is required; do not extract typed patient facts",
        }
    else:
        topic_disposition = {
            "source": "topics/topic/@template and topics/topic/field",
            "disposition": "patient_evidence",
            "target": "narrative",
            "normalization_rule": "trec-2023-questionnaire-canonical-text-v1",
        }
        patient_recipe = {
            "id": f"{track.dataset_id}-canonical-patient-text-v1",
            "source_evidence_kind": "narrative",
            "whitespace": "collapse Unicode whitespace to one ASCII space and strip",
            "unicode_normalization": "none; preserve source code points",
            "separator": "LF",
            "omission": "require template; omit fields with empty values",
            "line_order": "template, then non-empty fields in source order",
            "template_rendering": "[TEMPLATE] {template}",
            "field_rendering": "[FIELD] {name}: {value}",
        }
    selection: dict[str, JsonValue] = {
        "topic_policy": "all official topics",
        "trial_policy": "all historical corpus trials",
    }
    if smoke_topics is not None:
        selection = {
            "topic_policy": f"lowest {smoke_topics} numeric topic IDs",
            "trial_policy": "union of trials with Judgments for selected topics",
            "purpose": "parser smoke verification only; do not use for effectiveness claims",
        }
    allowed_roles: list[JsonValue] = list(track.expected_source_roles)
    system_input_artifact_ids: list[JsonValue] = ["topics", *track.archive_roles]
    if smoke_topics is not None:
        system_input_artifact_ids.append("qrels")
    recipe_id = (
        track.source_recipe_id
        if smoke_topics is None
        else f"{track.dataset_id}-parser-smoke-{smoke_topics}-recipe-v2"
    )
    return {
        "recipe_id": recipe_id,
        "snapshot_contract_version": "2.0",
        "benchmark_lineage": track.benchmark_lineage,
        "allowed_source_roles": allowed_roles,
        "allowed_artifact_ids": allowed_roles,
        "system_input_artifact_ids": system_input_artifact_ids,
        "connector": {"name": CONNECTOR_NAME, "version": CONNECTOR_VERSION},
        "parsers": {
            "topics": f"{track.dataset_id}-{track.topic_format}-topics-xml-v1",
            "qrels": "trec-four-column-qrels-v1",
            "trials": "clinicaltrials-gov-historical-xml-v1",
            "archives": "zip-member-order-by-name-v1",
            "judgment_trial_identity": "ctgov-nct-alias-to-canonical-v1",
        },
        "selection": selection,
        "field_disposition_table": [
            topic_disposition,
            *FIELD_DISPOSITION_TABLE[1:],
            {
                "source": "clinical_study/id_info/nct_alias",
                "disposition": "evaluator_identity_normalization",
                "target": "Judgment.trial_id",
                "normalization_rule": "ctgov-nct-alias-to-canonical-v1",
                "rationale": (
                    "resolve a Judgment ID only when the parsed historical corpus record "
                    "declares it as an alias; collapse only equal duplicate labels and reject "
                    "conflicts; aliases are not System input"
                ),
            },
        ],
        "canonical_patient_text_recipe": patient_recipe,
        "canonical_trial_text_recipe": canonical_trial_text,
        "diagnostic_policy": {
            "version": f"{track.dataset_id}-snapshot-diagnostics-v1",
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


def prepare_trec_ct(
    dataset_id: str,
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    source_lock_path: str | Path | None = None,
    smoke_topics: int | None = None,
    created_at: datetime | None = None,
) -> PreparationResult:
    """Verify and prepare one frozen TREC Clinical Trials 2022 or 2023 Snapshot."""

    try:
        track = TRACKS[dataset_id]
    except KeyError as exc:
        raise IngestionError(f"unsupported TREC Clinical Trials dataset {dataset_id!r}") from exc
    source = Path(source_directory)
    if not source.is_dir():
        raise SourceVerificationError(f"source directory does not exist: {source}")
    timestamp = created_at if created_at is not None else datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise IngestionError("created_at must include a UTC offset")
    source_bundle = _load_source_bundle_lock(track, source_lock_path)
    _verify_source_bundle(source, source_bundle)
    topics = parse_topics_xml(
        source / track.topics_filename,
        artifact=source_bundle.artifact_by_id("topics"),
        track=track,
    )
    try:
        judgments = parse_qrels(source / track.qrels_filename)
    except IngestionError:
        raise IngestionError("protected Judgments failed structural validation") from None
    topics, judgments, selected_trial_ids = _select_topics(topics, judgments, smoke_topics)
    trials, diagnostics, alias_to_trial_id = _parse_trials(
        source,
        source_bundle=source_bundle,
        archive_roles=frozenset(track.archive_roles),
        include_trial_ids=selected_trial_ids,
    )
    source_judgment_count = len(judgments)
    (
        judgments,
        resolved_judgment_count,
        collapsed_equal_duplicate_count,
        resolved_aliases,
    ) = _normalize_judgment_trial_ids(judgments, trials, alias_to_trial_id)
    _validate_relations(topics, trials, judgments)
    snapshot_name = (
        track.snapshot_name
        if smoke_topics is None
        else f"{track.snapshot_name}-parser-smoke-{smoke_topics}"
    )
    prepared_dataset_id = snapshot_name
    capabilities = frozenset(
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
        benchmark_lineage=track.benchmark_lineage,
        snapshot_name=snapshot_name,
        topics=topics,
        trials=trials,
        available_capabilities=capabilities,
    )
    qrels_artifact = source_bundle.artifact_by_id("qrels")
    evaluation_package = EvaluationPackage(
        benchmark_lineage=track.benchmark_lineage,
        task_id="trec-clinical-trials-patient-to-trial-ranking",
        snapshot_id=snapshot.snapshot_id,
        judgments=judgments,
        provenance={
            "source_artifact_id": qrels_artifact.artifact_id,
            "source_artifact_sha256": qrels_artifact.sha256,
            "format": "TREC four-column Judgments",
            "parser": "trec-four-column-qrels-v1",
            "transformation_rule": (f"{track.dataset_id}-qrels-with-nct-alias-normalization-v1"),
            "trial_identity_normalization": {
                "rule": "ctgov-nct-alias-to-canonical-v1",
                "source_judgment_count": source_judgment_count,
                "normalized_judgment_count": len(judgments),
                "resolved_judgment_count": resolved_judgment_count,
                "collapsed_equal_duplicate_count": collapsed_equal_duplicate_count,
                "resolved_aliases": [
                    {
                        "judgment_trial_id": alias,
                        "corpus_trial_id": canonical,
                    }
                    for alias, canonical in resolved_aliases
                ],
                "conflicting_duplicate_behavior": "reject",
                "unresolved_behavior": "reject",
            },
        },
    )
    return write_prepared_benchmark(
        output_directory,
        dataset_id=prepared_dataset_id,
        snapshot=snapshot,
        evaluation_package=evaluation_package,
        source_bundle=source_bundle,
        source_recipe=_source_recipe(track, smoke_topics=smoke_topics),
        diagnostics=diagnostics,
        created_at=timestamp,
        build_provenance={
            "connector_name": CONNECTOR_NAME,
            "connector_version": CONNECTOR_VERSION,
            "source_lock_sha256": source_bundle.lock_sha256,
        },
    )


def full_preparation_identity(track: str) -> tuple[str, str, str]:
    """Return the connector-owned identity of one complete Track preparation."""

    try:
        configuration = TRACKS[track]
    except KeyError as exc:
        raise ValueError(f"unsupported TREC Clinical Trials Track {track!r}") from exc
    recipe = _source_recipe(configuration, smoke_topics=None)
    return (
        configuration.dataset_id,
        configuration.source_recipe_id,
        content_sha256(recipe),
    )


__all__ = [
    "CONNECTOR_NAME",
    "CONNECTOR_VERSION",
    "TRACKS",
    "TrecClinicalTrialsTrack",
    "full_preparation_identity",
    "parse_topics_xml",
    "prepare_trec_ct",
]
