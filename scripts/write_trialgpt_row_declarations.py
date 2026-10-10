#!/usr/bin/env python3
"""Write the declarations of one target's two TrialGPT paper rows, before its generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from taim.cli_support import read_pool_ids
from taim.data import PREPARED_MANIFEST_FILENAME, load_prepared_benchmark
from taim.entity_versions import parse_clinical_as_of, source_grounded_patient_evidence_profile
from taim.patient_to_trial import PatientToTrialRunRequest
from taim.pipelines.trialgpt.paper_row_declarations import (
    PaperRowDeclarationError,
    PaperRowStratum,
    paper_row_stratum,
    release_declaration_identity,
    write_paper_row_declarations,
)
from taim.pipelines.trialgpt.retrieval_producer import (
    require_trialgpt_capabilities,
    resolve_trialgpt_retrieval_target,
)
from taim.pipelines.trialgpt.system import TrialGPTTAIMLunaV1System
from taim.release_profiles import (
    inspect_external_fidelity_profile,
    inspect_judgment_union_profile,
    resolve_benchmark_profile,
)


def package_stratum(
    data_dir: Path, *, track: str, profile: str, pool_ids: Sequence[str] | None
) -> PaperRowStratum:
    """The identities a run on this package records, derived from the package."""

    target = resolve_trialgpt_retrieval_target(track=track, profile=profile)
    if target.supplied_pool_membership != (pool_ids is not None):
        requirement = "requires" if target.supplied_pool_membership else "does not accept"
        raise PaperRowDeclarationError(f"Profile {profile!r} {requirement} a pool file")
    expected = paper_row_stratum(track, profile)
    require_trialgpt_capabilities(data_dir / PREPARED_MANIFEST_FILENAME)
    prepared = load_prepared_benchmark(data_dir)
    receipt = (
        inspect_judgment_union_profile(prepared, profile_id=profile)
        if pool_ids is None
        else inspect_external_fidelity_profile(
            prepared,
            profile_id=profile,
            supplied_pool_ids=pool_ids,
            expected_pool_count=expected.pool_count,
        )
    )
    # The Task Input a run derives, bound to the System's capabilities as the freeze binds it.
    request = PatientToTrialRunRequest.for_benchmark(
        resolve_benchmark_profile(
            prepared,
            profile_id=profile,
            expected_pool_count=cast(int, receipt["pool_count"]),
            expected_pool_ids_sha256=cast(str, receipt["pool_ids_sha256"]),
            expected_pool_receipt_id=cast(str, receipt["pool_receipt_id"]),
            supplied_pool_ids=pool_ids,
        ),
        run_id="paper-row-declaration",
        top_k=1,
        metric_cutoff=1,
        options={},
        clinical_as_of=parse_clinical_as_of(expected.clinical_as_of),
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
        required_capabilities=TrialGPTTAIMLunaV1System.required_capabilities,
        optional_capabilities=TrialGPTTAIMLunaV1System.optional_capabilities,
    )
    return PaperRowStratum(
        track=track,
        profile=profile,
        profile_definition_sha256=cast(str, receipt["profile_definition_sha256"]),
        prepared_snapshot_id=cast(str, receipt["prepared_snapshot_id"]),
        task_input_id=request.snapshot.task_input_id,
        evaluation_package_id=cast(str, receipt["evaluation_package_id"]),
        pool_count=cast(int, receipt["pool_count"]),
        pool_ids_sha256=cast(str, receipt["pool_ids_sha256"]),
        pool_receipt_id=cast(str, receipt["pool_receipt_id"]),
        clinical_as_of=expected.clinical_as_of,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive and write the protocol v12 Section 7 declarations of a target's "
            "TrialGPT-TAIM-Luna-v1 and TrialGPT2-TAIM-Luna-v1 rows, each with protocol v14's "
            "amendment, from the release and the target's complete-eligibility package. Refuses "
            "a package that does not reproduce protocol v15 Section 4. Commit both before the "
            "target's generation starts. Makes no provider call."
        )
    )
    parser.add_argument("--track", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="the target's complete-eligibility package"
    )
    parser.add_argument("--pool-file", type=Path, help="TREC 2021's supplied pool, only there")
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--package-artifact", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--trialgpt-run-id", required=True, help="the TrialGPT 1.0 row's run ID")
    parser.add_argument("--trialgpt2-run-id", required=True, help="the TrialGPT 2.0 row's run ID")
    comparisons = parser.add_mutually_exclusive_group(required=True)
    comparisons.add_argument(
        "--trialgpt-comparison",
        action="append",
        metavar="RUN_ID",
        help="a paper row of this stratum the TrialGPT 1.0 row is compared with (protocol v12 "
        "Section 6); once per comparison",
    )
    comparisons.add_argument(
        "--no-trialgpt-comparisons",
        action="store_true",
        help="the TrialGPT 1.0 row registers no comparison",
    )
    parser.add_argument("--declared-by", required=True, help="who commits the declarations")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    release = release_declaration_identity(
        args.release_manifest, args.package_artifact, args.dependency_lock
    )
    stratum = package_stratum(
        args.data_dir,
        track=args.track,
        profile=args.profile,
        pool_ids=read_pool_ids(args.pool_file),
    )
    written = write_paper_row_declarations(
        stratum=stratum,
        release=release,
        trialgpt_run_id=args.trialgpt_run_id,
        trialgpt2_run_id=args.trialgpt2_run_id,
        trialgpt_comparisons=args.trialgpt_comparison or [],
        declared_by=args.declared_by,
        output_dir=args.output_dir,
    )
    print(json.dumps({"declarations": [str(path.resolve()) for path in written]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
