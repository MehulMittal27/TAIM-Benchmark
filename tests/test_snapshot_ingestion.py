# ruff: noqa: S101
from __future__ import annotations

import gc
import json
import tracemalloc
import zipfile
from pathlib import Path

import pytest

from taim.artifacts import sha256_file
from taim.data import load_prepared_benchmark
from taim.data.prepared import PREPARED_MANIFEST_FILENAME
from taim.data.trec_ct_2021 import (
    IngestionError,
    MalformedTrialError,
    SourceVerificationError,
    load_source_bundle_lock,
    parse_qrels,
    parse_topics_xml,
    parse_trial_xml,
    prepare_trec_ct_2021,
)
from taim.schemas import SchemaValidationError
from taim.source import SourceArtifact
from tests.snapshot_helpers import TREC_FIXED_TIME, complete_trial_xml, write_trec_source_bundle


def _artifact(*, artifact_id: str, role: str, filename: str) -> SourceArtifact:
    return SourceArtifact(
        artifact_id=artifact_id,
        role=role,
        filename=filename,
        byte_size=1,
        sha256="sha256:" + "a" * 64,
        acquisition_date="2021-04-27",
        access_terms="synthetic test fixture",
        redistribution_terms="synthetic test fixture",
        url=f"https://example.test/{filename}",
    )


@pytest.mark.parametrize(
    ("serialized", "message"),
    [
        ("<topic />", "root"),
        ("<topics task='wrong'><topic number='1'>patient</topic></topics>", "task"),
        (
            "<topics task='2021 TREC Clinical Trials'><topic>patient</topic></topics>",
            "positive decimal integer",
        ),
        (
            "<topics task='2021 TREC Clinical Trials'><topic number='0'>patient</topic></topics>",
            "positive decimal integer",
        ),
        (
            "<topics task='2021 TREC Clinical Trials'><topic number='1'>a</topic>"
            "<topic number='1'>b</topic></topics>",
            "duplicate",
        ),
        (
            "<topics task='2021 TREC Clinical Trials'><topic number='1'>  </topic></topics>",
            "empty",
        ),
        ("<topics", "malformed"),
    ],
)
def test_topics_parser_rejects_invalid_source_records(
    tmp_path: Path, serialized: str, message: str
) -> None:
    path = tmp_path / "topics.xml"
    path.write_text(serialized, encoding="utf-8")

    with pytest.raises(IngestionError, match=message):
        parse_topics_xml(
            path,
            artifact=_artifact(artifact_id="topics", role="topics", filename=path.name),
        )


def test_topics_parser_normalizes_whitespace_preserves_unicode_and_orders_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "topics.xml"
    path.write_text(
        "<topics task='2021 TREC Clinical Trials'>"
        "<topic number='2'> second\tpatient </topic>"
        "<topic number='1'> A\u030a\npatient </topic>"
        "</topics>",
        encoding="utf-8",
    )

    topics = parse_topics_xml(
        path,
        artifact=_artifact(artifact_id="topics", role="topics", filename=path.name),
    )

    assert [(topic.topic_id, topic.canonical_text) for topic in topics] == [
        ("1", "A\u030a patient"),
        ("2", "second patient"),
    ]


@pytest.mark.parametrize(
    ("serialized", "message"),
    [
        ("\n", "blank"),
        ("1 0 NCT00000001\n", "four"),
        ("topic 0 NCT00000001 2\n", "topic ID"),
        ("1 1 NCT00000001 2\n", "iteration"),
        ("1 0 bad-id 2\n", "NCT ID"),
        ("1 0 NCT00000001 eligible\n", "label"),
        ("1 0 NCT00000001 3\n", "label"),
        ("1 0 NCT00000001 2\n1 0 NCT00000001 1\n", "duplicate"),
    ],
)
def test_qrels_parser_rejects_invalid_records(
    tmp_path: Path, serialized: str, message: str
) -> None:
    path = tmp_path / "qrels.txt"
    path.write_text(serialized, encoding="utf-8")

    with pytest.raises(IngestionError, match=message):
        parse_qrels(path)


def test_qrels_parser_preserves_label_zero_and_orders_pairs(tmp_path: Path) -> None:
    path = tmp_path / "qrels.txt"
    path.write_text(
        "2 0 NCT00000002 2\n10 0 NCT00000003 2\n1 0 NCT00000002 1\n1 0 NCT00000001 0\n",
        encoding="utf-8",
    )

    judgments = parse_qrels(path)

    assert [(row.topic_id, row.trial_id, row.label) for row in judgments] == [
        ("1", "NCT00000001", 0),
        ("1", "NCT00000002", 1),
        ("10", "NCT00000003", 2),
        ("2", "NCT00000002", 2),
    ]


@pytest.mark.parametrize(
    ("serialized", "member_name", "message", "trial_id"),
    [
        (b"<record />", "NCT00000001.xml", "root", "NCT00000001"),
        (
            b"<clinical_study><id_info /></clinical_study>",
            "missing.xml",
            "invalid or missing NCT ID",
            None,
        ),
        (
            b"<clinical_study><id_info><nct_id>NCT123</nct_id></id_info>"
            b"<brief_title>x</brief_title></clinical_study>",
            "record.xml",
            "invalid or missing NCT ID",
            "NCT123",
        ),
        (
            b"<clinical_study><id_info><nct_id>NCT00000002</nct_id></id_info>"
            b"<brief_title>x</brief_title></clinical_study>",
            "NCT00000001.xml",
            "does not match",
            "NCT00000002",
        ),
        (
            b"<clinical_study><id_info><nct_id>NCT00000001</nct_id></id_info></clinical_study>",
            "NCT00000001.xml",
            "no canonical input text",
            "NCT00000001",
        ),
    ],
)
def test_trial_parser_reports_record_identity_for_fatal_errors(
    serialized: bytes,
    member_name: str,
    message: str,
    trial_id: str | None,
) -> None:
    with pytest.raises(MalformedTrialError, match=message) as caught:
        parse_trial_xml(
            serialized,
            artifact=_artifact(
                artifact_id="trials_part_1",
                role="trials_part_1",
                filename="trials.zip",
            ),
            member_name=member_name,
        )

    assert caught.value.trial_id == trial_id


def test_trial_parser_accepts_namespaced_clinicaltrials_elements() -> None:
    serialized = b"""
    <clinical_study xmlns="urn:clinicaltrials">
      <id_info><nct_id>NCT12345678</nct_id></id_info>
      <brief_title> Trial\n title </brief_title>
      <condition>Condition A</condition>
    </clinical_study>
    """

    trial, diagnostics = parse_trial_xml(
        serialized,
        artifact=_artifact(
            artifact_id="trials_part_1",
            role="trials_part_1",
            filename="trials.zip",
        ),
        member_name="NCT12345678.xml",
    )

    assert trial.canonical_text == "[BRIEF_TITLE] Trial title\n[CONDITION] Condition A"
    assert diagnostics == ()


def test_trial_parser_can_attach_complete_layout_preserving_eligibility() -> None:
    serialized = (
        complete_trial_xml()
        .replace(
            "Adults with alpha syndrome.\n        Excludes cardiac disease.",
            "Inclusion Criteria:\n- Adults with alpha syndrome.\n"
            "Exclusion Criteria:\n- Cardiac disease.",
        )
        .encode()
    )

    trial, diagnostics = parse_trial_xml(
        serialized,
        artifact=_artifact(
            artifact_id="trials_part_1",
            role="trials_part_1",
            filename="trials.zip",
        ),
        member_name="NCT00000001.xml",
        include_complete_eligibility_text=True,
    )

    eligibility = next(section for section in trial.sections if section.role == "eligibility")
    assert eligibility.text == (
        "Inclusion Criteria: - Adults with alpha syndrome. Exclusion Criteria: - Cardiac disease."
    )
    assert eligibility.source_text == (
        "Inclusion Criteria:\n- Adults with alpha syndrome.\n"
        "Exclusion Criteria:\n- Cardiac disease."
    )
    assert eligibility.source_text_byte_length == len(eligibility.source_text.encode())
    assert diagnostics == ()


def test_archive_preparation_emits_stable_record_fatal_diagnostics(tmp_path: Path) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={
            "NCT00000001": complete_trial_xml(),
            "NCT00000002": "<record />",
        },
    )

    prepared = load_prepared_benchmark(
        prepare_trec_ct_2021(
            source,
            tmp_path / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        ).output_directory
    )

    assert [trial.trial_id for trial in prepared.trials] == ["NCT00000001"]
    assert len(prepared.diagnostics) == 1
    diagnostic = prepared.diagnostics[0]
    assert diagnostic.code == "malformed-trial-record"
    assert diagnostic.severity == "record_fatal"
    assert diagnostic.message == "trial XML root must be <clinical_study>"
    assert diagnostic.record_id == "NCT00000002"
    assert diagnostic.provenance[0].location == "zip-member:records/NCT00000002.xml"


def _loading_peak_allocated_bytes(root: Path, *, trial_count: int) -> int:
    trials = {
        f"NCT{index:08d}": complete_trial_xml(trial_id=f"NCT{index:08d}")
        for index in range(1, trial_count + 1)
    }
    source, source_lock = write_trec_source_bundle(root, trials=trials)
    prepared_directory = prepare_trec_ct_2021(
        source,
        root / "prepared",
        source_lock_path=source_lock,
        created_at=TREC_FIXED_TIME,
    ).output_directory
    del trials
    gc.collect()

    tracemalloc.start()
    try:
        prepared = load_prepared_benchmark(prepared_directory)
        assert len(prepared.trials) == trial_count
        assert tuple(trial.trial_id for trial in prepared.trials) == tuple(
            f"NCT{index:08d}" for index in range(1, trial_count + 1)
        )
        assert prepared.trials[0].trial_id == "NCT00000001"
        assert prepared.trials[-1].trial_id == f"NCT{trial_count:08d}"
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_default_prepared_loader_does_not_retain_the_trial_collection_in_memory(
    tmp_path: Path,
) -> None:
    small_peak = _loading_peak_allocated_bytes(tmp_path / "small", trial_count=20)
    large_peak = _loading_peak_allocated_bytes(tmp_path / "large", trial_count=400)

    assert large_peak - small_peak < 2 * 1024 * 1024, (
        f"loading allocation grew by {large_peak - small_peak:,} bytes "
        f"(small={small_peak:,}, large={large_peak:,})"
    )


def test_archive_preparation_rejects_duplicate_trial_ids_across_archives(
    tmp_path: Path,
) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": complete_trial_xml()},
    )
    second_archive = source / "ClinicalTrials.2021-04-27.part2.zip"
    with zipfile.ZipFile(second_archive, "a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("duplicate/NCT00000001.xml", complete_trial_xml())
    lock = json.loads(source_lock.read_text(encoding="utf-8"))
    second = next(item for item in lock["sources"] if item["role"] == "trials_part_2")
    second["byte_size"] = second_archive.stat().st_size
    second["sha256"] = sha256_file(second_archive)
    source_lock.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(IngestionError, match="duplicate trial ID NCT00000001"):
        prepare_trec_ct_2021(
            source,
            tmp_path / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("qrels", "message"),
    [
        ("2 0 NCT00000001 2\n", "unknown topic 2"),
        ("1 0 NCT99999999 2\n", "unknown trial NCT99999999"),
    ],
)
def test_preparation_rejects_judgments_for_unknown_snapshot_records(
    tmp_path: Path, qrels: str, message: str
) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": complete_trial_xml()},
        qrels=qrels,
    )

    with pytest.raises(IngestionError, match=message):
        prepare_trec_ct_2021(
            source,
            tmp_path / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate-role", "roles must be unique"),
        ("missing-role", "roles do not match"),
        ("invalid-digest", "SHA-256"),
        ("missing-digest", "missing=.*sha256"),
    ],
)
def test_source_bundle_lock_rejects_malformed_roles_and_digests(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": complete_trial_xml()},
    )
    payload = json.loads(source_lock.read_text(encoding="utf-8"))
    if mutation == "duplicate-role":
        payload["sources"][1]["role"] = payload["sources"][0]["role"]
    elif mutation == "missing-role":
        payload["sources"].pop()
    elif mutation == "invalid-digest":
        payload["sources"][0]["sha256"] = "pending"
    else:
        del payload["sources"][0]["sha256"]
    source_lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((IngestionError, SchemaValidationError), match=message):
        load_source_bundle_lock(source_lock)


def test_preparation_rejects_a_missing_required_source_artifact(tmp_path: Path) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": complete_trial_xml()},
    )
    (source / "topics2021.xml").unlink()

    with pytest.raises(SourceVerificationError, match="missing required Source Bundle files"):
        prepare_trec_ct_2021(
            source,
            tmp_path / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("role", "message"),
    [
        ("topics", "topics must use deterministic identity order"),
        ("trials", "trials must use deterministic identity order"),
    ],
)
def test_prepared_loader_rejects_noncanonical_snapshot_record_order(
    tmp_path: Path, role: str, message: str
) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        topics_xml=(
            "<topics task='2021 TREC Clinical Trials'>"
            "<topic number='1'>first</topic><topic number='2'>second</topic></topics>"
        ),
        trials={
            "NCT00000001": complete_trial_xml(trial_id="NCT00000001"),
            "NCT00000002": complete_trial_xml(trial_id="NCT00000002"),
        },
        qrels="1 0 NCT00000001 2\n2 0 NCT00000002 2\n",
    )
    prepared_directory = prepare_trec_ct_2021(
        source,
        tmp_path / "prepared",
        source_lock_path=source_lock,
        created_at=TREC_FIXED_TIME,
    ).output_directory
    output_path = prepared_directory / f"{role}.jsonl"
    lines = output_path.read_text(encoding="utf-8").splitlines()
    output_path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    manifest_path = prepared_directory / PREPARED_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][role]["byte_size"] = output_path.stat().st_size
    manifest["outputs"][role]["sha256"] = sha256_file(output_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SchemaValidationError, match=message):
        load_prepared_benchmark(prepared_directory)
