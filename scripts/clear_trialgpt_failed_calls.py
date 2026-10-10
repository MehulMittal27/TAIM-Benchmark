#!/usr/bin/env python3
"""Check a TrialGPT generation cache before a resume, clearing cached failed provider calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from taim.pipelines.trialgpt.cache_clearing import (
    INTERRUPTION_CAUSES,
    QUERY_STAGE,
    SYSTEM_STAGE,
    CacheClearingRefused,
    clear_failed_provider_calls,
    resolve_generation_cache,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run before every resume of an interrupted TrialGPT generation or row, while no "
            "process holds its cache (protocol v11 Section 7.1). Removes only unresolved entries "
            "whose every attempt is a failed sign-in or transport call with no provider output, "
            "writes the clearing record, and appends the interruption to the interruptions file. "
            "Refuses, clearing nothing, when an unresolved entry holds provider output the "
            "release could not retry or would overwrite. Makes no provider call."
        )
    )
    parser.add_argument("--stage", choices=(QUERY_STAGE, SYSTEM_STAGE), required=True)
    parser.add_argument(
        "--run-id",
        required=True,
        help="the generation's run ID for the query stage; the row's or gate's for a System",
    )
    parser.add_argument(
        "--cache", type=Path, help="query stage: the --cache the generation was given"
    )
    parser.add_argument(
        "--system",
        choices=("TrialGPT-TAIM-Luna-v1",),
        help="system stage: the System whose run resumes; the only one that caches failed calls",
    )
    parser.add_argument(
        "--workspace", type=Path, help="system stage: that run's --trialgpt-workspace"
    )
    parser.add_argument("--cause", choices=sorted(INTERRUPTION_CAUSES), required=True)
    parser.add_argument(
        "--record-dir", type=Path, required=True, help="where the clearing record is written"
    )
    parser.add_argument(
        "--interruptions",
        type=Path,
        required=True,
        help="the interruptions JSON list this check appends to; the receipt reads it",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    cache = resolve_generation_cache(
        stage=args.stage,
        run_id=args.run_id,
        cache=args.cache,
        system_id=args.system,
        workspace=args.workspace,
    )
    try:
        record = clear_failed_provider_calls(
            cache,
            run_id=args.run_id,
            cause=args.cause,
            record_dir=args.record_dir,
            interruptions=args.interruptions,
        )
    except CacheClearingRefused as refused:
        print(str(refused), file=sys.stderr)
        return 2
    # The record without the removed entries, which it holds verbatim.
    print(json.dumps({k: v for k, v in record.items() if k != "removed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
