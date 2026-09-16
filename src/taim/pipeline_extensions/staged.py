"""The staged paper System's release commands and its Result Bundle rules.

The System fuses two validated component runs, scores the fused pool with the pinned reranker and
assembles its stages. Its run, options, dependency needs and bundle rules live here, so a release
that does not publish the System ships none of them.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from taim.pipeline_extensions import ComponentRun, PipelineExtension
from taim.schemas import PIPELINE_DEPTH_RETRIEVAL

if TYPE_CHECKING:
    from taim.public_patient_to_trial import StoredPublicPatientToTrialRun
    from taim.release_profiles import ResolvedBenchmark
    from taim.schemas import JsonValue
    from taim.system_contracts import SystemRunRequest, SystemRunResult

_STAGE_DEPTHS = {
    "bm25-folded-depth5000": (PIPELINE_DEPTH_RETRIEVAL, 5_000),
    "qwen3-no-template-depth5000": (PIPELINE_DEPTH_RETRIEVAL, 5_000),
    "rrf-scoring-pool-depth2000": (PIPELINE_DEPTH_RETRIEVAL, 2_000),
    "rerank-scored-depth2000": ("rerank", 2_000),
}


def _run(system_id: str, request: SystemRunRequest) -> SystemRunResult:
    raise ValueError(f"System {system_id!r} is not an exported direct reference System")


def _run_options(
    args: argparse.Namespace,
    benchmark: ResolvedBenchmark,
    task_input_id: str,
) -> dict[str, object]:
    raise ValueError("fusion and staged options are supplied as direction-specific artifacts")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--folded-bm25-run", type=Path)
    parser.add_argument("--no-template-qwen-run", type=Path)
    parser.add_argument(
        "--reranker-score-artifact",
        type=Path,
        help="create or reuse raw, unevaluated scores for the staged reranker pool",
    )
    parser.add_argument(
        "--reranker-checkpoint",
        type=Path,
        help="resume the pinned reranker from an identity-bound batch checkpoint",
    )
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument(
        "--reranker-attention-backend",
        choices=("flash", "mem_efficient", "math"),
        default="flash",
    )
    parser.add_argument(
        "--reranker-offline",
        action="store_true",
        help="require the pinned reranker revision to exist in the local model cache",
    )


def _validate_components(
    run: ComponentRun, *, folded_query_bundle_id: str
) -> tuple[StoredPublicPatientToTrialRun, StoredPublicPatientToTrialRun, dict[str, JsonValue]]:
    from taim.public_patient_to_trial import load_public_patient_to_trial_run
    from taim.release_cli import _rrf_component_producer_identity
    from taim.release_staged_pipeline import load_staged_pipeline

    args = run.args
    request = run.request
    benchmark_profile = run.benchmark.manifest_configuration()
    evaluation_package_id = run.benchmark.evaluation_package.evaluation_package_id
    pipeline = load_staged_pipeline()
    if args.folded_bm25_run is None or args.no_template_qwen_run is None:
        raise ValueError("staged System requires --folded-bm25-run and --no-template-qwen-run")
    folded = load_public_patient_to_trial_run(args.folded_bm25_run)
    qwen = load_public_patient_to_trial_run(args.no_template_qwen_run)
    expected = {
        "bm25-folded": (folded, "bm25-folded"),
        "qwen3-raw-no-template": (
            qwen,
            "dense-qwen3-embedding-0.6b-no-template",
        ),
    }
    expected_patients = {item.patient_id for item in request.snapshot.patient_versions}
    expected_trials = {item.trial_id for item in request.snapshot.trial_versions}
    expected_ranks = set(range(1, min(pipeline.component_depth, len(expected_trials)) + 1))
    component_producers: dict[str, JsonValue] = {}
    for arm, (stored, system_id) in expected.items():
        manifest = stored.manifest
        if manifest.system_id != system_id:
            raise ValueError(f"staged arm {arm!r} requires System {system_id!r}")
        if (
            manifest.benchmark_lineage != request.snapshot.benchmark_lineage
            or manifest.prepared_snapshot_id != request.snapshot.prepared_snapshot_id
            or manifest.task_input_id != request.snapshot.task_input_id
            or manifest.evaluation_package_id != evaluation_package_id
            or manifest.benchmark_profile != benchmark_profile
        ):
            raise ValueError(f"staged arm {arm!r} direction or benchmark identity does not match")
        if manifest.budget_k != pipeline.component_depth:
            raise ValueError(f"staged arm {arm!r} must request top-{pipeline.component_depth}")
        ranks_by_patient: dict[str, set[int]] = {
            patient_id: set() for patient_id in expected_patients
        }
        for row in stored.candidates:
            if row.topic_id not in expected_patients or row.trial_id not in expected_trials:
                raise ValueError(f"staged arm {arm!r} contains an entity outside the Task Input")
            ranks_by_patient[row.topic_id].add(row.rank)
        incomplete = sorted(
            patient_id for patient_id, ranks in ranks_by_patient.items() if ranks != expected_ranks
        )
        if incomplete:
            raise ValueError(
                f"staged arm {arm!r} lacks complete top-{len(expected_ranks)} rankings for: "
                + ", ".join(incomplete)
            )
        if run.producer_release_identity is not None:
            if run.producer_dependency_environment is None:
                raise ValueError("staged pipeline producer lacks its dependency environment")
            component_producers[arm] = _rrf_component_producer_identity(
                stored,
                expected_release_identity=run.producer_release_identity,
                expected_lock_sha256=cast(
                    str,
                    run.producer_dependency_environment["dependency_lock_sha256"],
                ),
            )
    query_transformation = folded.manifest.configuration.get("query_transformation")
    if (
        not isinstance(query_transformation, Mapping)
        or query_transformation.get("bundle_id") != folded_query_bundle_id
    ):
        raise ValueError("folded BM25 component does not use the supplied folded query bundle")
    model_identity = qwen.manifest.configuration.get("model_identity")
    if (
        not isinstance(model_identity, Mapping)
        or model_identity.get("encoder_id") != "qwen3-embedding-0.6b-no-template"
        or not isinstance(model_identity.get("query_format"), Mapping)
        or cast(Mapping[str, object], model_identity["query_format"]).get("template") != "{text}"
    ):
        raise ValueError("Qwen3 component does not use the frozen no-template encoder policy")
    return folded, qwen, component_producers


def _run_from_components(run: ComponentRun) -> SystemRunResult:
    from taim.baselines.rrf import fuse_rrf_n
    from taim.file_hash import sha256_file
    from taim.folded_query_bundle import validate_folded_query_bundle
    from taim.release_cli import _read_json_object
    from taim.release_staged_pipeline import (
        STAGED_PIPELINE_SYSTEM_ID,
        assemble_staged_pipeline,
        load_reranker_score_artifact,
        load_staged_pipeline,
        render_f_summary,
        score_staged_pairs,
        staged_scoring_input_sha256,
        write_reranker_score_artifact,
    )
    from taim.schemas import PRIMARY_RANKING_CANDIDATES, JsonValue
    from taim.system_contracts import SystemRunResult

    args = run.args
    run_id = run.run_id
    request = run.request
    pipeline = load_staged_pipeline()
    if args.top_k != pipeline.output_depth:
        raise ValueError(f"staged System is frozen at --top-k {pipeline.output_depth}")
    if args.folded_query_bundle is None:
        raise ValueError("staged System requires --folded-query-bundle")
    bundle = _read_json_object(args.folded_query_bundle, "folded query bundle")
    folded_queries, folded_query_bundle_id = validate_folded_query_bundle(
        bundle,
        expected_task_input_id=request.snapshot.task_input_id,
        expected_topic_ids=tuple(topic.topic_id for topic in request.snapshot.topics),
    )
    folded, qwen, component_producers = _validate_components(
        run, folded_query_bundle_id=folded_query_bundle_id
    )
    request = replace(
        request,
        options={
            "attention_backend": args.reranker_attention_backend,
            "bm25_folded_candidates_sha256": folded.candidates_hash,
            "folded_query_bundle_id": folded_query_bundle_id,
            "pipeline_definition_sha256": pipeline.definition_sha256,
            "qwen_no_template_candidates_sha256": qwen.candidates_hash,
            "reranker_model_id": pipeline.reranker["model_id"],
            "reranker_model_revision": pipeline.reranker["model_revision"],
        },
    )
    components = {
        "bm25-folded": folded.candidates,
        "qwen3-raw-no-template": qwen.candidates,
    }
    fusion_started_at = time.perf_counter()
    fused = tuple(
        fuse_rrf_n(
            components,
            run_id=run_id,
            system_id=STAGED_PIPELINE_SYSTEM_ID,
            top_k=pipeline.fusion_depth,
            component_depth=pipeline.component_depth,
        )
    )
    fused_ranks: dict[str, set[int]] = {topic.topic_id: set() for topic in request.snapshot.topics}
    for row in fused:
        fused_ranks[row.topic_id].add(row.rank)
    expected_fused_ranks = set(range(1, pipeline.fusion_depth + 1))
    incomplete_fusion = sorted(
        topic_id for topic_id, ranks in fused_ranks.items() if ranks != expected_fused_ranks
    )
    if incomplete_fusion:
        raise ValueError(
            f"staged RRF lacks complete top-{pipeline.fusion_depth} scoring pools for: "
            + ", ".join(incomplete_fusion)
        )
    if args.reranker_score_artifact is None:
        raise ValueError("staged System requires --reranker-score-artifact")
    query_text = {query.topic_id: query.canonical_text for query in folded_queries}
    trial_ids = {row.trial_id for row in fused}
    trial_text: dict[str, str] = {}
    fallback_count = 0
    for trial in request.snapshot.trials:
        if trial.trial_id not in trial_ids:
            continue
        rendered, used_fallback = render_f_summary(trial)
        trial_text[trial.trial_id] = rendered
        fallback_count += int(used_fallback)
    scoring_input_id = staged_scoring_input_sha256(
        fused,
        folded_queries=query_text,
        trial_documents=trial_text,
    )
    score_path = cast(Path, args.reranker_score_artifact)
    if score_path.exists():
        scores, reranker_runtime = load_reranker_score_artifact(
            score_path,
            fused_candidates=fused,
            pipeline_definition_sha256=pipeline.definition_sha256,
            scoring_input_sha256=scoring_input_id,
        )
        score_action = "reused"
    else:
        scores, reranker_runtime = score_staged_pairs(
            fused,
            folded_queries=query_text,
            trial_documents=trial_text,
            model_cache=args.model_cache,
            attention_backend=args.reranker_attention_backend,
            local_files_only=args.reranker_offline,
            checkpoint_path=args.reranker_checkpoint,
        )
        reranker_runtime["f_summary_fallback_count"] = fallback_count
        write_reranker_score_artifact(
            score_path,
            scores=scores,
            runtime=reranker_runtime,
            pipeline_definition_sha256=pipeline.definition_sha256,
            scoring_input_sha256=scoring_input_id,
        )
        score_action = "created"
    candidates, stage_rankings, configuration = assemble_staged_pipeline(
        components,
        scores,
        benchmark_profile_id=run.profile_id,
        run_id=run_id,
    )
    index_identity = configuration.get("index_identity")
    if not isinstance(index_identity, dict):
        raise ValueError("staged System omitted its component index identity")
    index_identity.update(
        {
            "component_producer_identities": component_producers,
            "folded_query_bundle_id": folded_query_bundle_id,
            "persistent": False,
            "reranker_score_artifact_sha256": sha256_file(score_path),
            "scoring_input_sha256": scoring_input_id,
        }
    )
    runtime_seconds = time.perf_counter() - fusion_started_at
    configuration.update(
        {
            "runtime_seconds": runtime_seconds,
            "runtime_scope": "validated component fusion, reranker scoring, and assembly",
            "system_input": request.system_input_dict(),
            "folded_query_bundle": {
                "bundle_id": folded_query_bundle_id,
                "policy": cast(JsonValue, bundle["policy"]),
                "provenance": cast(JsonValue, bundle["provenance"]),
            },
            "reranker_score_artifact": {
                "action": score_action,
                "path": str(score_path.resolve()),
                "sha256": sha256_file(score_path),
                "scoring_input_sha256": scoring_input_id,
                "runtime": reranker_runtime,
                "effectiveness_evaluation": False,
            },
            "component_manifests": {
                "bm25-folded": folded.manifest_hash,
                "qwen3-raw-no-template": qwen.manifest_hash,
            },
            **(
                {"component_producer_identities": component_producers}
                if component_producers
                else {}
            ),
        }
    )
    return SystemRunResult(
        candidates=candidates,
        configuration=configuration,
        runtime_seconds=runtime_seconds,
        primary_ranking=PRIMARY_RANKING_CANDIDATES,
        pipeline_depth="rerank",
        stage_rankings=stage_rankings,
    )


def _validate_component_producers(
    system_input_projection: Mapping[str, object],
    producer_identity: Mapping[str, JsonValue],
) -> None:
    from taim.contracts import require_exact_keys, require_non_empty, require_sha256
    from taim.release_result_bundle import _positive_int, _validated_producer_identity
    from taim.schemas import SchemaValidationError

    index_identity = system_input_projection["index_identity"]
    if not isinstance(index_identity, Mapping):
        raise SchemaValidationError("staged Result Bundle lacks its index identity")
    require_exact_keys(
        index_identity,
        {
            "components",
            "component_producer_identities",
            "folded_query_bundle_id",
            "kind",
            "persistent",
            "reranker_score_artifact_sha256",
            "scoring_input_sha256",
        },
        role="staged Result Bundle index identity",
    )
    for field in (
        "folded_query_bundle_id",
        "reranker_score_artifact_sha256",
        "scoring_input_sha256",
    ):
        require_sha256(index_identity[field], f"staged Result Bundle {field}")
    if (
        index_identity["kind"] != "validated_component_runs"
        or index_identity["persistent"] is not False
    ):
        raise SchemaValidationError("staged Result Bundle index semantics changed")
    components = index_identity["components"]
    if not isinstance(components, Mapping) or set(components) != {
        "bm25-folded",
        "qwen3-raw-no-template",
    }:
        raise SchemaValidationError("staged Result Bundle component index is incomplete")
    for raw in components.values():
        if not isinstance(raw, Mapping):
            raise SchemaValidationError("staged Result Bundle component index is invalid")
        require_exact_keys(
            raw,
            {"candidate_count", "candidates_sha256", "run_id"},
            role="staged Result Bundle component index",
        )
        _positive_int(raw["candidate_count"], role="staged component candidate_count")
        require_sha256(raw["candidates_sha256"], "staged component candidates_sha256")
        require_non_empty(raw["run_id"], "staged component run_id")
    raw_producers = index_identity["component_producer_identities"]
    if not isinstance(raw_producers, Mapping) or set(raw_producers) != {
        "bm25-folded",
        "qwen3-raw-no-template",
    }:
        raise SchemaValidationError("staged Result Bundle component producers are incomplete")
    expected_release = producer_identity["release_identity"]
    producer_environment = cast(Mapping[str, object], producer_identity["dependency_environment"])
    expected_lock = producer_environment["dependency_lock_sha256"]
    for raw in raw_producers.values():
        component = _validated_producer_identity(raw)
        if component["release_identity"] != expected_release:
            raise SchemaValidationError("staged component producer release identity does not match")
        component_environment = cast(Mapping[str, object], component["dependency_environment"])
        if component_environment["dependency_lock_sha256"] != expected_lock:
            raise SchemaValidationError("staged component dependency lock does not match")


EXTENSION = PipelineExtension(
    system_tasks={"staged-bm25-qwen3-rrf-rerank": "patient_to_trial"},
    pipeline_depth="rerank",
    external_baseline=False,
    subcommands={},
    run=_run,
    add_run_options=_add_run_options,
    run_options=_run_options,
    required_distributions=(
        "huggingface-hub",
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers",
    ),
    ci_jobs=({"extras": ("staged",), "imports": ("taim.release_staged_pipeline",)},),
    run_from_components=_run_from_components,
    stage_depths=_STAGE_DEPTHS,
    validate_component_producers=_validate_component_producers,
)
