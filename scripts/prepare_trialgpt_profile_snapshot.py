#!/usr/bin/env python3
"""Build one TrialGPT target's prepared package with complete eligibility text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from taim.cli_support import read_pool_ids
from taim.contracts import require_sha256
from taim.data import load_prepared_benchmark
from taim.data.complete_eligibility import prepare_profile_complete_eligibility_snapshot
from taim.entity_versions import parse_clinical_as_of, source_grounded_patient_evidence_profile
from taim.patient_to_trial import PatientToTrialRunRequest
from taim.pipelines.trialgpt.retrieval_producer import resolve_trialgpt_retrieval_target
from taim.pipelines.trialgpt.system import TrialGPTTAIMLunaV1System
from taim.release_profiles import (
    inspect_external_fidelity_profile,
    inspect_judgment_union_profile,
    resolve_benchmark_profile,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-read one admitted TrialGPT target's pool trials from the verified Source Bundle "
            "with their complete eligibility text, require each to reproduce the stratum's own "
            "record, and write them with the criterion Derived View as a new prepared package. "
            "Makes no provider call."
        )
    )
    parser.add_argument("--track", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="the stratum's prepared package"
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="the Track's acquired Source Bundle files"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-file", type=Path)
    parser.add_argument("--pool-count", type=int, required=True)
    parser.add_argument("--pool-ids-sha256", required=True)
    parser.add_argument("--pool-receipt-id", required=True)
    parser.add_argument(
        "--clinical-as-of",
        type=parse_clinical_as_of,
        required=True,
        help="the target's clinical as-of time, for the Task Input ID the command prints",
    )
    return parser


def _progress(message: str) -> None:
    print(f"prepare-trialgpt-profile-snapshot: {message}", file=sys.stderr, flush=True)


def main() -> int:
    args = _parser().parse_args()
    target = resolve_trialgpt_retrieval_target(track=args.track, profile=args.profile)
    if target.supplied_pool_membership != (args.pool_file is not None):
        requirement = "requires" if target.supplied_pool_membership else "does not accept"
        raise ValueError(f"Profile {target.profile!r} {requirement} --pool-file")
    for name in ("pool_ids_sha256", "pool_receipt_id"):
        require_sha256(getattr(args, name), name.replace("_", " "))
    pool_ids = read_pool_ids(args.pool_file)
    result = prepare_profile_complete_eligibility_snapshot(
        args.data_dir,
        args.source,
        args.output_dir,
        profile_id=target.profile,
        expected_pool_count=args.pool_count,
        expected_pool_ids_sha256=args.pool_ids_sha256,
        expected_pool_receipt_id=args.pool_receipt_id,
        supplied_pool_ids=pool_ids,
        progress=_progress,
    )
    prepared = load_prepared_benchmark(result.output_directory)
    receipt = (
        inspect_judgment_union_profile(prepared, profile_id=target.profile)
        if pool_ids is None
        else inspect_external_fidelity_profile(
            prepared,
            profile_id=target.profile,
            supplied_pool_ids=pool_ids,
            expected_pool_count=args.pool_count,
        )
    )
    # The Task Input a run derives from the package, bound to the System's capabilities as the
    # freeze and preflight bind it.
    request = PatientToTrialRunRequest.for_benchmark(
        resolve_benchmark_profile(
            prepared,
            profile_id=target.profile,
            expected_pool_count=args.pool_count,
            expected_pool_ids_sha256=cast(str, receipt["pool_ids_sha256"]),
            expected_pool_receipt_id=cast(str, receipt["pool_receipt_id"]),
            supplied_pool_ids=pool_ids,
        ),
        run_id="complete-eligibility-identity",
        top_k=1,
        metric_cutoff=1,
        options={},
        clinical_as_of=args.clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
        required_capabilities=TrialGPTTAIMLunaV1System.required_capabilities,
        optional_capabilities=TrialGPTTAIMLunaV1System.optional_capabilities,
    )
    print(
        json.dumps(
            {
                "output": str(result.output_directory),
                "available_capabilities": sorted(prepared.snapshot.available_capabilities),
                "source_prepared_snapshot_id": result.manifest["snapshot_source_recipe"][
                    "definition"
                ]["source_stratum"]["prepared_snapshot_id"],
                "source_pool_receipt_id": args.pool_receipt_id,
                "task_input_id": request.snapshot.task_input_id,
                **receipt,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
