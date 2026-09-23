"""The frozen two-arm fusion release unit and its producer checks."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from taim.baselines.rank_fusion import (
    DEFAULT_RRF_OUTPUT_DEPTH,
    RRF_COMPONENT_DEPTH,
    RRF_CONSTANT,
    _non_negative_integer,
    _validated_component,
)
from taim.contracts import require_exact_keys, require_sha256
from taim.pipeline_extensions import PipelineExtension
from taim.schemas import (
    PIPELINE_DEPTH_RETRIEVAL,
    PRIMARY_RANKING_CANDIDATES,
    Candidate,
    JsonValue,
    SchemaValidationError,
)
from taim.system_contracts import SystemRunRequest, SystemRunResult


def _validated_components(
    bm25_candidates: Iterable[Candidate],
    dense_candidates: Iterable[Candidate],
) -> tuple[tuple[Candidate, ...], str, tuple[Candidate, ...], str]:
    bm25_rows, bm25_run_id = _validated_component(
        bm25_candidates,
        expected_system_id="bm25",
    )
    dense_rows, dense_run_id = _validated_component(
        dense_candidates,
        expected_system_id="dense-bge-m3",
    )
    if bm25_run_id == dense_run_id:
        raise ValueError("BM25 and dense component runs must have distinct run_id values")
    return bm25_rows, bm25_run_id, dense_rows, dense_run_id


def fuse_rrf(
    bm25_candidates: Iterable[Candidate],
    dense_candidates: Iterable[Candidate],
    *,
    run_id: str,
    system_id: str = "rrf",
    top_k: int = DEFAULT_RRF_OUTPUT_DEPTH,
) -> list[Candidate]:
    """Fuse the frozen BM25 and dense component top-1000 ranks using ``1 / (60 + rank)``."""

    from taim.baselines.rank_fusion import fuse_rrf as fuse_pair

    return fuse_pair(
        bm25_candidates,
        dense_candidates,
        expected_system_ids=("bm25", "dense-bge-m3"),
        run_id=run_id,
        system_id=system_id,
        top_k=top_k,
    )


def rrf_configuration(
    bm25_candidates: Sequence[Candidate],
    dense_candidates: Sequence[Candidate],
    *,
    top_k: int = DEFAULT_RRF_OUTPUT_DEPTH,
) -> dict[str, JsonValue]:
    """Return the complete JSON-compatible RRF retrieval configuration."""

    output_depth = _non_negative_integer(top_k, "top_k")
    bm25_rows, bm25_run_id, dense_rows, dense_run_id = _validated_components(
        bm25_candidates,
        dense_candidates,
    )
    return {
        "implementation": "taim.baselines.rrf.fuse_rrf",
        "formula": "sum(1 / (60 + component_rank))",
        "constant": RRF_CONSTANT,
        "component_retrieval_depth": RRF_COMPONENT_DEPTH,
        "requested_top_k": output_depth,
        "raw_component_scores_used": False,
        "tie_breaking": "RRF score descending, trial_id ascending",
        "topic_order": "topic_id ascending",
        "components": [
            {
                "system_id": "bm25",
                "run_id": bm25_run_id,
                "candidate_count": len(bm25_rows),
                "maximum_input_rank": max(candidate.rank for candidate in bm25_rows),
            },
            {
                "system_id": "dense-bge-m3",
                "run_id": dense_run_id,
                "candidate_count": len(dense_rows),
                "maximum_input_rank": max(candidate.rank for candidate in dense_rows),
            },
        ],
    }


def _validate_rrf_components(
    args: argparse.Namespace,
    *,
    request: SystemRunRequest,
    benchmark_profile: Mapping[str, object],
    evaluation_package_id: str,
    producer_release_identity: Mapping[str, str] | None,
    producer_dependency_environment: Mapping[str, JsonValue] | None,
):
    from taim.public_patient_to_trial import load_public_patient_to_trial_run
    from taim.release_cli import _rrf_component_producer_identity

    if args.bm25_run is None or args.dense_run is None:
        raise ValueError("RRF requires --bm25-run and --dense-run")
    bm25 = load_public_patient_to_trial_run(args.bm25_run)
    dense = load_public_patient_to_trial_run(args.dense_run)
    if bm25.manifest.system_id != "bm25" or dense.manifest.system_id != "dense-bge-m3":
        raise ValueError("RRF component Systems must be bm25 and dense-bge-m3")
    expected_patients = {item.patient_id for item in request.snapshot.patient_versions}
    expected_trials = {item.trial_id for item in request.snapshot.trial_versions}
    expected_ranks = set(range(1, min(RRF_COMPONENT_DEPTH, len(expected_trials)) + 1))
    component_producers: dict[str, JsonValue] = {}
    for component in (bm25, dense):
        manifest = component.manifest
        if (
            manifest.benchmark_lineage != request.snapshot.benchmark_lineage
            or manifest.prepared_snapshot_id != request.snapshot.prepared_snapshot_id
            or manifest.task_input_id != request.snapshot.task_input_id
            or manifest.evaluation_package_id != evaluation_package_id
            or manifest.benchmark_profile != benchmark_profile
            or manifest.query_patient_versions
            != tuple(
                (item.patient_id, item.patient_version_id)
                for item in request.snapshot.patient_versions
            )
            or manifest.trial_corpus_versions
            != tuple(
                (item.trial_id, item.trial_version_id) for item in request.snapshot.trial_versions
            )
        ):
            raise ValueError("RRF component direction or benchmark identity does not match")
        if manifest.budget_k != RRF_COMPONENT_DEPTH:
            raise ValueError(f"RRF component {manifest.system_id} must request top-1000")
        ranks_by_patient: dict[str, set[int]] = {
            patient_id: set() for patient_id in expected_patients
        }
        for row in component.candidates:
            if row.topic_id not in expected_patients or row.trial_id not in expected_trials:
                raise ValueError("RRF component contains an entity outside the Task Input")
            ranks_by_patient[row.topic_id].add(row.rank)
        incomplete = sorted(
            patient_id for patient_id, ranks in ranks_by_patient.items() if ranks != expected_ranks
        )
        if incomplete:
            raise ValueError(
                f"RRF component {manifest.system_id} lacks complete top-{len(expected_ranks)} "
                "rankings for: " + ", ".join(incomplete)
            )
        if producer_release_identity is not None:
            if producer_dependency_environment is None:
                raise ValueError("RRF fusion producer lacks its dependency environment")
            component_producers[manifest.system_id] = _rrf_component_producer_identity(
                component,
                expected_release_identity=producer_release_identity,
                expected_lock_sha256=cast(
                    str,
                    producer_dependency_environment["dependency_lock_sha256"],
                ),
            )
    return bm25, dense, component_producers


def _validate_rrf_component_producers(
    system_input_projection: Mapping[str, object],
    producer_identity: Mapping[str, JsonValue],
) -> None:
    from taim.release_result_bundle import _validated_producer_identity

    index_identity = system_input_projection["index_identity"]
    if not isinstance(index_identity, Mapping):
        raise SchemaValidationError("RRF Result Bundle lacks its index identity")
    require_exact_keys(
        index_identity,
        {
            "bm25_candidates_sha256",
            "dense_candidates_sha256",
            "component_producer_identities",
            "persistent",
        },
        role="RRF Result Bundle index identity",
    )
    require_sha256(index_identity["bm25_candidates_sha256"], "RRF BM25 candidate SHA-256")
    require_sha256(index_identity["dense_candidates_sha256"], "RRF dense candidate SHA-256")
    if index_identity["persistent"] is not False:
        raise SchemaValidationError("RRF Result Bundle index identity must be non-persistent")
    raw_components = index_identity["component_producer_identities"]
    if not isinstance(raw_components, Mapping) or set(raw_components) != {
        "bm25",
        "dense-bge-m3",
    }:
        raise SchemaValidationError("RRF Result Bundle component producers are incomplete")
    expected_release = producer_identity["release_identity"]
    producer_environment = cast(
        Mapping[str, object],
        producer_identity["dependency_environment"],
    )
    expected_lock = producer_environment["dependency_lock_sha256"]
    for system_id in ("bm25", "dense-bge-m3"):
        component = _validated_producer_identity(raw_components[system_id])
        if component["release_identity"] != expected_release:
            raise SchemaValidationError("RRF component producer release identity does not match")
        component_environment = cast(
            Mapping[str, object],
            component["dependency_environment"],
        )
        if component_environment["dependency_lock_sha256"] != expected_lock:
            raise SchemaValidationError("RRF component dependency lock does not match")


def _run_from_components(run):
    args, request, run_id = run.args, run.request, run.run_id
    bm25, dense, component_producers = _validate_rrf_components(
        args,
        request=request,
        benchmark_profile=run.benchmark.manifest_configuration(),
        evaluation_package_id=run.benchmark.evaluation_package.evaluation_package_id,
        producer_release_identity=run.producer_release_identity,
        producer_dependency_environment=run.producer_dependency_environment,
    )
    request = SystemRunRequest(
        snapshot=request.snapshot,
        run_id=run_id,
        top_k=args.top_k,
        metric_cutoff=args.metric_cutoff,
        options={
            "bm25_candidates_sha256": bm25.candidates_hash,
            "dense_candidates_sha256": dense.candidates_hash,
        },
        identity_version="2.0",
    )
    started_at = time.perf_counter()
    candidates = tuple(fuse_rrf(bm25.candidates, dense.candidates, run_id=run_id, top_k=args.top_k))
    runtime_seconds = time.perf_counter() - started_at
    configuration = {
        **rrf_configuration(bm25.candidates, dense.candidates, top_k=args.top_k),
        "runtime_seconds": runtime_seconds,
        "runtime_scope": "RRF fusion over validated top-1000 component rankings",
        "system_input": request.system_input_dict(),
        "model_identity": {"kind": "none", "reason": "deterministic_rrf"},
        "index_identity": {
            "bm25_candidates_sha256": bm25.candidates_hash,
            "dense_candidates_sha256": dense.candidates_hash,
            **(
                {"component_producer_identities": component_producers}
                if component_producers
                else {}
            ),
            "persistent": False,
        },
    }
    primary_ranking = PRIMARY_RANKING_CANDIDATES
    pipeline_depth = PIPELINE_DEPTH_RETRIEVAL
    return SystemRunResult(
        candidates=candidates,
        configuration=configuration,
        runtime_seconds=runtime_seconds,
        primary_ranking=primary_ranking,
        pipeline_depth=pipeline_depth,
    )


def _add_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bm25-run", type=Path)
    parser.add_argument("--dense-run", type=Path)


def _run(system_id, request):
    raise ValueError(f"System {system_id!r} requires validated component runs")


def _options(args, benchmark, task_input_id):
    raise ValueError("fusion options are supplied as direction-specific artifacts")


EXTENSION = PipelineExtension(
    system_tasks={"rrf": "patient_to_trial"},
    pipeline_depth="retrieval",
    external_baseline=False,
    subcommands={},
    run=_run,
    add_run_options=_add_options,
    run_options=_options,
    run_from_components=_run_from_components,
    validate_component_producers=_validate_rrf_component_producers,
)
