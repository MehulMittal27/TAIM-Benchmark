"""Checksum-locked SIGIR 2016 patient-to-clinical-trial connector."""

from __future__ import annotations

import json
import re
import tarfile
from collections.abc import Mapping
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Literal, cast

from taim.contracts import content_sha256
from taim.data.prepared import PreparationResult, write_prepared_benchmark
from taim.data.trec_ct_2021 import (
    CANONICAL_TEXT_RECIPE,
    FIELD_DISPOSITION_TABLE,
    IngestionError,
    MalformedTrialError,
    SourceVerificationError,
    _collapse_whitespace,
    _provenance,
    _record_fatal_diagnostic,
    _source_artifact,
    _source_trial_id,
    parse_qrels,
    parse_trial_xml,
)
from taim.evaluation_package import ConnectorDiagnostic, EvaluationPackage
from taim.file_hash import sha256_file
from taim.judgments import SIGIR_CT_2016_JUDGMENT_SCHEME
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

DATASET_ID = "sigir-ct-2016"
BENCHMARK_LINEAGE = DATASET_ID
CONNECTOR_NAME = "taim.data.sigir_ct_2016"
CONNECTOR_VERSION = "1.0"
SOURCE_BUNDLE_LOCK_VERSION = "2.0"
DEFAULT_LOCK_PATH = Path(__file__).with_name("locks") / f"{DATASET_ID}.json"

QueryVariant = Literal["description", "summary"]

SOURCE_FILENAMES: Mapping[str, str] = {
    "adhoc_queries": "adhoc-queries.json",
    "trials": "clinicaltrials.gov-16_dec_2015.tgz",
    "terms": "CLINICALTRIALS.GOV_TERMS_AND_CONDITIONS.txt",
    "qrels": "qrels-clinical_trials.txt",
    "readme": "README.html",
    "topics_description": "topics-2014_2015-description.topics",
    "topics_summary": "topics-2014_2015-summary.topics",
    "expected_relevant_counts": "Ts.tsv",
}
EXPECTED_SOURCE_ROLES = tuple(SOURCE_FILENAMES)


def _load_source_bundle_lock(path: str | Path | None) -> SourceBundle:
    lock_path = Path(path) if path is not None else DEFAULT_LOCK_PATH
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestionError(f"invalid Source Bundle lock JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise IngestionError("Source Bundle lock must be a JSON object")
    if set(payload) != {"schema_version", "dataset_id", "lock_status", "sources"}:
        raise IngestionError("Source Bundle lock has unexpected or missing fields")
    if payload.get("schema_version") != SOURCE_BUNDLE_LOCK_VERSION:
        raise IngestionError("unsupported Source Bundle lock schema version")
    if payload.get("dataset_id") != DATASET_ID:
        raise IngestionError(f"Source Bundle dataset_id must be {DATASET_ID!r}")
    if payload.get("lock_status") != "complete":
        raise IngestionError("Source Bundle lock must be complete")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or any(
        not isinstance(item, Mapping) for item in raw_sources
    ):
        raise IngestionError("Source Bundle sources must be an array of objects")
    artifacts = tuple(_source_artifact(cast(Mapping[str, object], item)) for item in raw_sources)
    by_role = {artifact.role: artifact for artifact in artifacts}
    if len(by_role) != len(artifacts) or set(by_role) != set(EXPECTED_SOURCE_ROLES):
        raise IngestionError("Source Bundle source roles do not match SIGIR 2016")
    for role, filename in SOURCE_FILENAMES.items():
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
        elif sha256_file(path) != artifact.sha256:
            errors.append(f"{artifact.filename}: SHA-256 does not match Source Bundle lock")
    if errors:
        raise SourceVerificationError("Source Bundle verification failed: " + "; ".join(errors))


_TOPIC_RECORD = re.compile(
    rb"<TOP>\s*<NUM>(?P<number>.*?)</NUM>\s*<TITLE>(?P<title>.*?)</TOP>",
    re.IGNORECASE | re.DOTALL,
)
_OPTIONAL_TITLE_END = re.compile(r"</TITLE>\s*\Z", re.IGNORECASE)


def parse_topics(
    path: str | Path,
    *,
    artifact: SourceArtifact,
    query_variant: QueryVariant,
) -> tuple[BenchmarkTopic, ...]:
    serialized = Path(path).read_bytes()
    try:
        serialized.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError("SIGIR topics must be UTF-8") from exc
    topics: list[BenchmarkTopic] = []
    seen: set[str] = set()
    position = 0
    for match in _TOPIC_RECORD.finditer(serialized):
        if serialized[position : match.start()].strip():
            raise IngestionError("SIGIR topics contain content outside TOP records")
        topic_id = _collapse_whitespace(match.group("number").decode("utf-8"))
        raw_title = match.group("title").decode("utf-8")
        raw_title = _OPTIONAL_TITLE_END.sub("", raw_title)
        text = _collapse_whitespace(unescape(raw_title))
        if not topic_id.isdecimal() or int(topic_id) < 1:
            raise IngestionError("SIGIR topic number must be a positive decimal integer")
        if topic_id in seen:
            raise IngestionError(f"duplicate topic number {topic_id}")
        if not text:
            raise IngestionError(f"SIGIR topic {topic_id} has empty patient text")
        provenance = _provenance(
            artifact,
            record_id=topic_id,
            locator_kind="xml_path",
            location=f"/TOP[NUM='{topic_id}']/TITLE",
            raw_value=raw_title,
            rule=f"sigir-ct-2016-{query_variant}-title-text-v1",
        )
        topics.append(
            BenchmarkTopic(
                topic_id=topic_id,
                source_identity=SourceRecordIdentity(
                    namespace=f"{DATASET_ID}.{query_variant}.topics",
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
        position = match.end()
    if serialized[position:].strip():
        raise IngestionError("SIGIR topics contain trailing content outside TOP records")
    if not topics:
        raise IngestionError("SIGIR topics must contain at least one topic")
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
        raise IngestionError("smoke_topics must be between one and the collection topic count")
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
    artifact: SourceArtifact,
    include_trial_ids: frozenset[str] | None,
) -> tuple[tuple[TrialDocument, ...], tuple[ConnectorDiagnostic, ...]]:
    trials: dict[str, TrialDocument] = {}
    diagnostics: list[ConnectorDiagnostic] = []
    try:
        with tarfile.open(source_directory / artifact.filename, mode="r:gz") as archive:
            members = sorted(archive.getmembers(), key=lambda item: item.name)
            for member in members:
                if not member.isfile() or not member.name.casefold().endswith(".xml"):
                    continue
                member_trial_id = _source_trial_id(member.name)
                if include_trial_ids is not None and member_trial_id not in include_trial_ids:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise IngestionError("SIGIR trial archive member could not be read")
                with stream:
                    serialized = stream.read()
                try:
                    trial, field_diagnostics = parse_trial_xml(
                        serialized,
                        artifact=artifact,
                        member_name=member.name,
                    )
                except MalformedTrialError as exc:
                    diagnostics.append(
                        _record_fatal_diagnostic(
                            exc,
                            artifact=artifact,
                            member_name=member.name,
                            serialized=serialized,
                        )
                    )
                    continue
                if trial.trial_id in trials:
                    raise IngestionError(f"duplicate trial ID {trial.trial_id}")
                trials[trial.trial_id] = trial
                diagnostics.extend(field_diagnostics)
    except tarfile.TarError as exc:
        raise IngestionError(f"invalid tar archive {artifact.filename}: {exc}") from exc
    if not trials:
        raise IngestionError("trial archive contains no selected valid XML records")
    return tuple(trials[key] for key in sorted(trials)), tuple(diagnostics)


def _validate_relations(
    topics: tuple[BenchmarkTopic, ...],
    trials: tuple[TrialDocument, ...],
    judgments: tuple[RelevanceJudgment, ...],
) -> None:
    topic_ids = {topic.topic_id for topic in topics}
    trial_ids = {trial.trial_id for trial in trials}
    for judgment in judgments:
        if judgment.topic_id not in topic_ids:
            raise IngestionError("Judgment references an unknown SIGIR topic")
        if judgment.trial_id not in trial_ids:
            raise IngestionError("Judgment references a trial absent from the SIGIR corpus")


def _source_recipe(
    *,
    query_variant: QueryVariant,
    smoke_topics: int | None,
) -> dict[str, JsonValue]:
    snapshot_name = f"{DATASET_ID}-{query_variant}"
    canonical_trial_text = dict(CANONICAL_TEXT_RECIPE)
    canonical_trial_text["id"] = f"{DATASET_ID}-canonical-trial-text-v1"
    selection: dict[str, JsonValue] = {
        "topic_policy": f"all official {query_variant} topics",
        "trial_policy": "all historical corpus trials",
    }
    if smoke_topics is not None:
        selection = {
            "topic_policy": f"lowest {smoke_topics} numeric topic IDs",
            "trial_policy": "union of trials with Judgments for selected topics",
            "purpose": "parser smoke verification only; do not use for effectiveness claims",
        }
    recipe_id = (
        f"{snapshot_name}-snapshot-recipe-v1"
        if smoke_topics is None
        else f"{snapshot_name}-parser-smoke-{smoke_topics}-recipe-v1"
    )
    active_topics_role = f"topics_{query_variant}"
    dispositions: list[JsonValue] = [
        {
            "source": f"{SOURCE_FILENAMES[active_topics_role]}/TOP/TITLE",
            "disposition": "patient_evidence",
            "target": "narrative",
            "normalization_rule": f"sigir-ct-2016-{query_variant}-title-text-v1",
        },
        *FIELD_DISPOSITION_TABLE[1:],
        {
            "source": "alternate topic representation and assessor ad-hoc queries",
            "disposition": "intentionally_ignored",
            "rationale": f"the {query_variant} Snapshot exposes exactly one frozen query view",
        },
        {
            "source": "README, terms, and expected relevant-count metadata",
            "disposition": "intentionally_ignored",
            "rationale": "provenance and collection documentation are not System inputs",
        },
    ]
    allowed_roles: list[JsonValue] = list(EXPECTED_SOURCE_ROLES)
    system_input_artifact_ids: list[JsonValue] = [active_topics_role, "trials"]
    if smoke_topics is not None:
        system_input_artifact_ids.append("qrels")
    return {
        "recipe_id": recipe_id,
        "snapshot_contract_version": "2.0",
        "benchmark_lineage": BENCHMARK_LINEAGE,
        "allowed_source_roles": allowed_roles,
        "allowed_artifact_ids": allowed_roles,
        "system_input_artifact_ids": system_input_artifact_ids,
        "connector": {"name": CONNECTOR_NAME, "version": CONNECTOR_VERSION},
        "parsers": {
            "topics": "sigir-ct-2016-trec-topics-v1",
            "qrels": "trec-four-column-qrels-v1",
            "trials": "clinicaltrials-gov-historical-xml-v1",
            "archive": "tar-gzip-member-order-by-name-v1",
        },
        "selection": selection,
        "field_disposition_table": dispositions,
        "canonical_patient_text_recipe": {
            "id": f"{snapshot_name}-canonical-patient-text-v1",
            "source_evidence_kind": "narrative",
            "whitespace": "collapse Unicode whitespace to one ASCII space and strip",
            "unicode_normalization": "none; preserve source code points",
            "separator": "LF",
            "omission": "TITLE is required; do not extract typed patient facts",
        },
        "canonical_trial_text_recipe": canonical_trial_text,
        "diagnostic_policy": {
            "version": "sigir-ct-2016-snapshot-diagnostics-v1",
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


def prepare_sigir_ct_2016(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    source_lock_path: str | Path | None = None,
    query_variant: QueryVariant = "description",
    smoke_topics: int | None = None,
    created_at: datetime | None = None,
) -> PreparationResult:
    """Verify and prepare one SIGIR 2016 query representation as a Snapshot."""

    if query_variant not in {"description", "summary"}:
        raise IngestionError("query_variant must be 'description' or 'summary'")
    source = Path(source_directory)
    if not source.is_dir():
        raise SourceVerificationError(f"source directory does not exist: {source}")
    timestamp = created_at if created_at is not None else datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise IngestionError("created_at must include a UTC offset")
    source_bundle = _load_source_bundle_lock(source_lock_path)
    _verify_source_bundle(source, source_bundle)
    topics_role = f"topics_{query_variant}"
    topics = parse_topics(
        source / SOURCE_FILENAMES[topics_role],
        artifact=source_bundle.artifact_by_id(topics_role),
        query_variant=query_variant,
    )
    judgments = parse_qrels(source / SOURCE_FILENAMES["qrels"])
    topics, judgments, selected_trial_ids = _select_topics(topics, judgments, smoke_topics)
    trials, diagnostics = _parse_trials(
        source,
        artifact=source_bundle.artifact_by_id("trials"),
        include_trial_ids=selected_trial_ids,
    )
    _validate_relations(topics, trials, judgments)
    snapshot_name = f"{DATASET_ID}-{query_variant}"
    if smoke_topics is not None:
        snapshot_name += f"-parser-smoke-{smoke_topics}"
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
        benchmark_lineage=BENCHMARK_LINEAGE,
        snapshot_name=snapshot_name,
        topics=topics,
        trials=trials,
        available_capabilities=capabilities,
    )
    qrels_artifact = source_bundle.artifact_by_id("qrels")
    evaluation_package = EvaluationPackage(
        benchmark_lineage=BENCHMARK_LINEAGE,
        task_id="sigir-clinical-trials-patient-to-trial-ranking",
        snapshot_id=snapshot.snapshot_id,
        judgments=judgments,
        provenance={
            "source_artifact_id": qrels_artifact.artifact_id,
            "source_artifact_sha256": qrels_artifact.sha256,
            "format": "TREC four-column judgments",
            "parser": "trec-four-column-qrels-v1",
            "label_scale": (
                "0 would not refer; 1 consider referral after further investigation; "
                "2 highly likely to refer"
            ),
            "transformation_rule": "sigir-ct-2016-qrels-identity-v1",
        },
        judgment_scheme=SIGIR_CT_2016_JUDGMENT_SCHEME,
    )
    return write_prepared_benchmark(
        output_directory,
        dataset_id=snapshot_name,
        snapshot=snapshot,
        evaluation_package=evaluation_package,
        source_bundle=source_bundle,
        source_recipe=_source_recipe(
            query_variant=query_variant,
            smoke_topics=smoke_topics,
        ),
        diagnostics=diagnostics,
        created_at=timestamp,
        build_provenance={
            "connector_name": CONNECTOR_NAME,
            "connector_version": CONNECTOR_VERSION,
            "source_lock_sha256": source_bundle.lock_sha256,
            "collection_version": "CSIRO v4",
            "collection_doi": "10.4225/08/58e2e83d92c2b",
        },
    )


def full_preparation_identity(query_variant: QueryVariant) -> tuple[str, str, str]:
    """Return the connector-owned identity of one complete query-profile preparation."""

    if query_variant not in {"description", "summary"}:
        raise ValueError(f"unsupported SIGIR 2016 query variant {query_variant!r}")
    recipe = _source_recipe(query_variant=query_variant, smoke_topics=None)
    return (
        f"{DATASET_ID}-{query_variant}",
        cast(str, recipe["recipe_id"]),
        content_sha256(recipe),
    )


__all__ = [
    "BENCHMARK_LINEAGE",
    "CONNECTOR_NAME",
    "CONNECTOR_VERSION",
    "DATASET_ID",
    "SOURCE_FILENAMES",
    "full_preparation_identity",
    "parse_topics",
    "prepare_sigir_ct_2016",
]
