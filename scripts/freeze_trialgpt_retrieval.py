#!/usr/bin/env python3
"""Build and freeze the provider-free half of the TrialGPT retrieval producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

from taim.artifacts import sha256_file
from taim.benchmark_release import validate_release_package_artifact
from taim.contracts import require_sha256
from taim.data import load_prepared_benchmark
from taim.eligibility import find_eligibility_criterion_view
from taim.entity_versions import parse_clinical_as_of, source_grounded_patient_evidence_profile
from taim.release_profiles import (
    resolve_benchmark_profile,
)
from taim.schemas import JsonValue
from taim.system_contracts import SystemRunRequest
from taim.trialgpt_paper import trial_view
from taim.trialgpt_publication import (
    load_frozen_trialgpt_retrieval,
    validate_frozen_retrieval_for_snapshot,
    validate_frozen_retrieval_for_task_input,
)
from taim.trialgpt_retrieval_backends import (
    TaimControlledBM25,
    TorchMedCPT,
    controlled_bm25_provenance,
)
from taim.trialgpt_retrieval_producer import (
    FrozenRetrievalBinding,
    load_generated_arm_queries,
    resolve_trialgpt_retrieval_target,
    retrieve_three_luna_queries,
    validate_fresh_retrieval_producer_contract,
    write_frozen_three_luna_retrieval,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify transferred Luna query plans, run pinned BM25 and MedCPT without provider "
            "calls, and freeze one admitted TrialGPT retrieval target."
        )
    )
    parser.add_argument("--track", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pool-file", type=Path)
    parser.add_argument("--pool-count", type=int, required=True)
    parser.add_argument("--pool-ids-sha256", required=True)
    parser.add_argument("--pool-receipt-id", required=True)
    parser.add_argument("--protocol-approval-id", required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--package-artifact", type=Path, required=True)
    parser.add_argument("--clinical-as-of", type=parse_clinical_as_of, required=True)
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--offline", action="store_true")
    return parser


def _pool_ids(path: Path) -> tuple[str, ...]:
    values = tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not values or len(values) != len(set(values)):
        raise ValueError("pool file must contain unique non-empty trial IDs")
    return values


def _clean_git_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    git = shutil.which("git")
    if git is None:
        raise ValueError("git executable is required to bind the retrieval producer commit")
    commit = subprocess.run(  # noqa: S603 - executable is resolved by shutil.which
        [git, "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(  # noqa: S603 - executable is resolved by shutil.which
        [git, "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("retrieval production requires a clean, committed TAIM working tree")
    return commit


def _producer_dependency_lock_sha256() -> str:
    lock = Path(__file__).resolve().parents[1] / "uv.lock"
    if not lock.is_file():
        raise ValueError("retrieval producer requires its source checkout uv.lock")
    return sha256_file(lock)


def _patient_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    args = _parser().parse_args()
    target = resolve_trialgpt_retrieval_target(track=args.track, profile=args.profile)
    if target.supplied_pool_membership != (args.pool_file is not None):
        requirement = "requires" if target.supplied_pool_membership else "does not accept"
        raise ValueError(f"Profile {target.profile!r} {requirement} --pool-file")
    for name in ("pool_ids_sha256", "pool_receipt_id", "protocol_approval_id"):
        require_sha256(getattr(args, name), name.replace("_", " "))
    producer_commit = _clean_git_commit()
    release_identity = validate_release_package_artifact(
        args.release_manifest,
        args.package_artifact,
    ).release_identity()
    prepared = load_prepared_benchmark(args.data_dir)
    if (
        prepared.dataset_id != target.prepared_dataset_id
        or prepared.snapshot.benchmark_lineage != target.track
        or prepared.snapshot.snapshot_name != target.snapshot_name
        or len(prepared.topics) != target.topic_count
    ):
        raise ValueError("prepared benchmark does not match the admitted Track/Profile target")
    benchmark = resolve_benchmark_profile(
        prepared,
        profile_id=target.profile,
        expected_pool_count=args.pool_count,
        expected_pool_ids_sha256=args.pool_ids_sha256,
        expected_pool_receipt_id=args.pool_receipt_id,
        supplied_pool_ids=_pool_ids(args.pool_file) if args.pool_file is not None else None,
    )
    request = SystemRunRequest.for_benchmark(
        benchmark,
        run_id=args.run_id,
        top_k=1_000,
        metric_cutoff=5,
        options={},
        clinical_as_of=args.clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
        identity_version="2.0",
    )
    queries = load_generated_arm_queries(args.query_plans)
    topic_hashes = {
        topic.topic_id: _patient_hash(topic.canonical_text) for topic in benchmark.topics
    }
    if {(query.topic_id, query.patient_text_sha256) for query in queries} != {
        (topic_id, digest) for topic_id, digest in topic_hashes.items()
    }:
        raise ValueError("query plans do not match the prepared Snapshot topic text")

    eligibility_view = find_eligibility_criterion_view(prepared.snapshot.derived_views)
    views = tuple(
        trial_view(trial, eligibility_criterion_view=eligibility_view) for trial in benchmark.trials
    )
    lexical = TaimControlledBM25(views)
    dense = TorchMedCPT(
        workspace=args.workspace,
        device=args.device,
        precision="float32",
        model_cache_dir=args.model_cache,
        offline=args.offline,
        batch_size=args.batch_size,
    )
    dense.build_trial_embeddings(views)
    raw = retrieve_three_luna_queries(
        queries,
        trials=views,
        lexical_backend=lexical,
        dense_backend=dense,
    )
    retrieval_implementation: dict[str, JsonValue] = {
        "bm25": cast(dict[str, JsonValue], controlled_bm25_provenance()),
        "fusion": {
            "condition_weight": "1 / (zero_based_condition_index + 1)",
            "rrf_k": 20,
            "tie_rule": "score descending, then trial_id ascending",
        },
        "medcpt": cast(dict[str, JsonValue], dict(dense.model_configuration)),
    }
    written = write_frozen_three_luna_retrieval(
        raw,
        binding=FrozenRetrievalBinding(
            run_id=args.run_id,
            benchmark_lineage=target.track,
            prepared_snapshot_id=prepared.snapshot.snapshot_id,
            evaluation_package_id=benchmark.evaluation_package.evaluation_package_id,
            task_input_id=request.snapshot.task_input_id,
            pool_receipt_id=args.pool_receipt_id,
            pool_count=args.pool_count,
            pool_ids_sha256=args.pool_ids_sha256,
            protocol_approval_id=args.protocol_approval_id,
            producer_git_commit=producer_commit,
            producer_dependency_lock_sha256=_producer_dependency_lock_sha256(),
            release_manifest_id=release_identity["release_manifest_id"],
            public_tree_id=release_identity["public_tree_id"],
            package_artifact_id=release_identity["package_artifact_id"],
        ),
        retrieval_implementation=retrieval_implementation,
        output_directory=args.output_dir,
    )
    retrieval = load_frozen_trialgpt_retrieval(
        written.retrieval_path,
        lock_path=written.lock_path,
    )
    validate_frozen_retrieval_for_snapshot(
        retrieval,
        benchmark_lineage=target.track,
        retrieval_source_snapshot_id=prepared.snapshot.snapshot_id,
        topic_ids=tuple(topic_hashes),
        trial_ids=tuple(trial.trial_id for trial in benchmark.trials),
    )
    validate_frozen_retrieval_for_task_input(
        retrieval,
        task_input_id=request.snapshot.task_input_id,
    )
    validate_fresh_retrieval_producer_contract(
        retrieval,
        artifact_directory=args.output_dir,
    )
    print(
        json.dumps(
            {
                "lock": str(written.lock_path),
                "lock_sha256": sha256_file(written.lock_path),
                "provider_calls_during_freeze": 0,
                "profile": target.profile,
                "release_identity": release_identity,
                "retrieval": str(written.retrieval_path),
                "retrieval_sha256": sha256_file(written.retrieval_path),
                "task_input_id": request.snapshot.task_input_id,
                "track": target.track,
                "topics": len(topic_hashes),
                "trials": len(benchmark.trials),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
