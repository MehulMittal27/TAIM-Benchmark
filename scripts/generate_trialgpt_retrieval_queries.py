#!/usr/bin/env python3
"""Generate the three frozen TrialGPT retrieval query arms with explicit provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taim.adapters.trialgpt import TRIALGPT_CODEX_MODEL, CodexTrialGPTProvider
from taim.artifacts import sha256_file
from taim.snapshot import BenchmarkTopic
from taim.trialgpt_generation import GenerationCache, GenerationCacheRunLease
from taim.trialgpt_retrieval_producer import (
    RetrievalQueryTopic,
    generate_three_luna_queries,
    resolve_trialgpt_retrieval_target,
    write_generated_arm_queries,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Make the explicit Luna provider calls for one admitted three-arm TrialGPT "
            "retrieval target. Reusing the same cache resumes exact completed calls."
        )
    )
    parser.add_argument("--track", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--codex-executable", default="codex")
    return parser


def _load_topics(
    manifest_path: Path,
    *,
    track: str,
    profile: str,
) -> tuple[tuple[RetrievalQueryTopic, ...], str]:
    target = resolve_trialgpt_retrieval_target(track=track, profile=profile)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("prepared manifest must be a JSON object")
    snapshot = manifest.get("snapshot")
    if (
        manifest.get("dataset_id") != target.prepared_dataset_id
        or not isinstance(snapshot, dict)
        or snapshot.get("benchmark_lineage") != target.track
        or snapshot.get("snapshot_name") != target.snapshot_name
    ):
        raise ValueError("prepared manifest does not match the admitted Track/Profile target")
    snapshot_id = snapshot.get("snapshot_id")
    outputs = manifest.get("outputs")
    topics_output = outputs.get("topics") if isinstance(outputs, dict) else None
    if (
        not isinstance(snapshot_id, str)
        or not isinstance(topics_output, dict)
        or not isinstance(topics_output.get("filename"), str)
        or not isinstance(topics_output.get("sha256"), str)
    ):
        raise ValueError("prepared manifest has no single-file topics output contract")
    path = manifest_path.parent / topics_output["filename"]
    if sha256_file(path) != topics_output["sha256"]:
        raise ValueError("prepared topics file does not match its manifest")
    topics: list[RetrievalQueryTopic] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                topic = BenchmarkTopic.from_json(line)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid prepared topic at line {line_number}") from exc
            topics.append(RetrievalQueryTopic(topic.topic_id, topic.canonical_text))
    if len(topics) != topics_output.get("record_count"):
        raise ValueError("prepared topics record count does not match its manifest")
    if len(topics) != target.topic_count:
        raise ValueError("prepared topics do not contain the frozen complete topic set")
    return tuple(topics), snapshot_id


def main() -> int:
    args = _parser().parse_args()
    topics, snapshot_id = _load_topics(
        args.prepared_manifest,
        track=args.track,
        profile=args.profile,
    )
    providers = {
        "medium": CodexTrialGPTProvider(
            model=TRIALGPT_CODEX_MODEL,
            reasoning_effort="medium",
            codex_executable=args.codex_executable,
        ),
        "xhigh": CodexTrialGPTProvider(
            model=TRIALGPT_CODEX_MODEL,
            reasoning_effort="xhigh",
            codex_executable=args.codex_executable,
        ),
        "recall-explicit": CodexTrialGPTProvider(
            model=TRIALGPT_CODEX_MODEL,
            reasoning_effort="medium",
            codex_executable=args.codex_executable,
        ),
    }
    cache = GenerationCache(args.cache)
    try:
        with GenerationCacheRunLease(args.cache, run_id=args.run_id):
            queries = generate_three_luna_queries(
                topics,
                providers=providers,
                cache=cache,
                max_workers=args.workers,
            )
            write_generated_arm_queries(queries, args.output)
    finally:
        cache.close()
    print(
        json.dumps(
            {
                "artifact_sha256": sha256_file(args.output),
                "arms": 3,
                "logical_calls": len(queries),
                "output": str(args.output.resolve()),
                "provider_calls_authorized_by_invocation": True,
                "prepared_snapshot_id": snapshot_id,
                "profile": args.profile,
                "track": args.track,
                "topics": len(topics),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
