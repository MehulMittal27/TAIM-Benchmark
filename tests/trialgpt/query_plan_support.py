"""A prepared TREC 2021 topic set for the query-plan records' tests."""

from __future__ import annotations

import json
from pathlib import Path

from taim.artifacts import sha256_file
from taim.pipelines.trialgpt.system import TrialGPTTAIMLunaV1System
from taim.snapshot import BenchmarkTopic, SourceRecordIdentity

_TRACK = "trec-ct-2021"


def prepared_topics_manifest(directory: Path) -> Path:
    """A prepared TREC 2021 manifest and topics file: what query generation reads."""

    topics = [
        BenchmarkTopic(
            topic_id=str(number),
            source_identity=SourceRecordIdentity("taim.fixture", str(number)),
            canonical_text=f"Patient {number} with a synthetic condition.",
        )
        for number in range(1, 76)
    ]
    topics_path = directory / "topics.jsonl"
    topics_path.write_text("".join(topic.to_json() + "\n" for topic in topics), encoding="utf-8")
    manifest = {
        "dataset_id": "trec-ct-2021",
        "snapshot": {
            "benchmark_lineage": _TRACK,
            "snapshot_name": "trec-ct-2021",
            "snapshot_id": "sha256:" + "4" * 64,
            # A complete-eligibility package, as query generation requires (protocol v15).
            "available_capabilities": sorted(TrialGPTTAIMLunaV1System.required_capabilities),
        },
        "outputs": {
            "topics": {
                "filename": topics_path.name,
                "sha256": sha256_file(topics_path),
                "record_count": len(topics),
            }
        },
    }
    path = directory / "prepared-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path
