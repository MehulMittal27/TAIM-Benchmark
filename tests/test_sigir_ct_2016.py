# ruff: noqa: S101
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from taim.data import load_prepared_benchmark
from taim.data.sigir_ct_2016 import parse_topics, prepare_sigir_ct_2016
from taim.source import SourceArtifact
from tests.snapshot_helpers import write_sigir_source_bundle

FIXED_TIME = datetime(2026, 8, 26, tzinfo=UTC)


def _artifact(filename: str) -> SourceArtifact:
    return SourceArtifact(
        artifact_id="topics_description",
        role="topics_description",
        filename=filename,
        byte_size=1,
        sha256="sha256:" + "a" * 64,
        acquisition_date="2017-04-04",
        access_terms="synthetic test fixture",
        redistribution_terms="synthetic test fixture",
        url=f"https://example.test/{filename}",
    )


def test_topics_parser_accepts_rootless_trec_records_and_uses_snapshot_identity_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "topics.topics"
    path.write_text(
        "<TOP><NUM>20142</NUM><TITLE> Second patient. </TITLE></TOP>"
        "<TOP><NUM>20141</NUM><TITLE>First patient.</TITLE></TOP>"
        "<TOP><NUM>201410</NUM><TITLE>Tenth patient.</TITLE></TOP>",
        encoding="utf-8",
    )

    topics = parse_topics(
        path,
        artifact=_artifact(path.name),
        query_variant="description",
    )

    assert [(topic.topic_id, topic.canonical_text) for topic in topics] == [
        ("20141", "First patient."),
        ("201410", "Tenth patient."),
        ("20142", "Second patient."),
    ]
    assert topics[0].source_identity.namespace == "sigir-ct-2016.description.topics"


def test_summary_smoke_preparation_keeps_one_query_view_and_judged_trial_union(
    tmp_path: Path,
) -> None:
    source, lock_path = write_sigir_source_bundle(tmp_path)

    result = prepare_sigir_ct_2016(
        source,
        tmp_path / "prepared",
        source_lock_path=lock_path,
        query_variant="summary",
        smoke_topics=1,
        created_at=FIXED_TIME,
    )
    prepared = load_prepared_benchmark(result.output_directory)

    assert prepared.dataset_id == "sigir-ct-2016-summary-parser-smoke-1"
    assert [(topic.topic_id, topic.canonical_text) for topic in prepared.topics] == [
        ("20141", "First summary.")
    ]
    assert [trial.trial_id for trial in prepared.trials] == ["NCT00000001"]
    assert [
        (row.topic_id, row.trial_id, row.label) for row in prepared.evaluation_package.judgments
    ] == [("20141", "NCT00000001", 2)]
    recipe = prepared.prepared_manifest["snapshot_source_recipe"]["definition"]
    assert recipe["system_input_artifact_ids"] == ["topics_summary", "trials", "qrels"]
    assert prepared.evaluation_package.provenance["label_scale"].startswith("0 would not refer")
