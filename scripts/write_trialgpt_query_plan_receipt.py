#!/usr/bin/env python3
"""Write the protocol v11 Section 6 receipt of one query-plan generation, before the freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taim.benchmark_release import validate_release_package_artifact
from taim.pipelines.trialgpt.query_plan_records import write_query_plan_receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a completed query-plan generation's cache and plans and write its receipt, "
            "named by the freeze's run ID. Commit the receipt before the freeze starts. Makes no "
            "provider call and never changes the cache."
        )
    )
    parser.add_argument("--track", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument(
        "--run-id", required=True, help="the run ID the freeze of these plans will be given"
    )
    parser.add_argument(
        "--generation-run-id", required=True, help="the run ID the generation was given"
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="the --cache the generation was given"
    )
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--package-artifact", type=Path, required=True)
    parser.add_argument(
        "--interruptions",
        type=Path,
        help="JSON list of the generation's interruptions, each with 'at' and 'cause' and any "
        "clearing record; omitted, the generation was not interrupted",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    release_identity = validate_release_package_artifact(
        args.release_manifest, args.package_artifact
    ).release_identity()
    interruptions = (
        json.loads(args.interruptions.read_text(encoding="utf-8"))
        if args.interruptions is not None
        else []
    )
    receipt = write_query_plan_receipt(
        track=args.track,
        profile=args.profile,
        prepared_manifest=args.prepared_manifest,
        run_id=args.run_id,
        generation_run_id=args.generation_run_id,
        cache=args.cache,
        query_plans=args.query_plans,
        release_identity=release_identity,
        interruptions=interruptions,
        output=args.output,
    )
    print(json.dumps({"receipt": str(args.output.resolve()), **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
