#!/usr/bin/env python3
"""Recompute a target's query plans from its retained cache, as protocol v11 Section 8 requires."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taim.pipelines.trialgpt.query_plan_records import recompute_query_plans


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the query method against a copy of the generation's cache with a provider "
            "that refuses every call, and write <run-id>.recomputation.json beside the receipt. "
            "It passes only if the plan file rebuilds byte for byte. Makes no provider call."
        )
    )
    parser.add_argument("--track", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, required=True, help="the --cache the generation was given"
    )
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--query-plan-receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, help="omitted, <run-id>.recomputation.json beside the receipt"
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    record, path = recompute_query_plans(
        track=args.track,
        profile=args.profile,
        prepared_manifest=args.prepared_manifest,
        cache=args.cache,
        query_plans=args.query_plans,
        receipt_path=args.query_plan_receipt,
        workers=args.workers,
        output=args.output,
    )
    print(json.dumps({"recomputation": str(path.resolve()), **record}, sort_keys=True))
    return 0 if record["status"] == "byte-identical" else 1


if __name__ == "__main__":
    raise SystemExit(main())
