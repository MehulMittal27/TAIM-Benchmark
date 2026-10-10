#!/usr/bin/env python3
"""Generate the three frozen TrialGPT retrieval query arms with explicit provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taim.artifacts import sha256_file
from taim.generation import GenerationCache, GenerationCacheRunLease
from taim.pipelines.trialgpt.query_plan_records import (
    generation_cache_path,
    load_query_topics,
    query_plan_models,
)
from taim.pipelines.trialgpt.retrieval_producer import (
    generate_three_luna_queries,
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


def main() -> int:
    args = _parser().parse_args()
    topics, snapshot_id = load_query_topics(
        args.prepared_manifest,
        track=args.track,
        profile=args.profile,
    )
    cache = GenerationCache(generation_cache_path(args.cache))
    models = query_plan_models(cache=cache, codex_executable=args.codex_executable)
    try:
        with GenerationCacheRunLease(cache.path, run_id=args.run_id):
            queries = generate_three_luna_queries(
                topics,
                models=models,
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
