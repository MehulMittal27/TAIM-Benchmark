"""Public Release 0.1 BM25 and dense reference Systems."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from taim.baselines.bm25 import BM25Retriever, bm25_ranking_configuration
from taim.baselines.encoder_registry import DENSE_ENCODER_REGISTRY
from taim.pipeline_extensions import extension_for
from taim.schemas import PIPELINE_DEPTH_RETRIEVAL, PRIMARY_RANKING_CANDIDATES, JsonValue
from taim.snapshot import CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT
from taim.system_contracts import (
    StrictSystemOptions,
    System,
    SystemRunRequest,
    SystemRunResult,
    normalize_system_request,
)


def _option(request: SystemRunRequest, name: str) -> object:
    try:
        return request.options[name]
    except KeyError as exc:
        raise ValueError(f"{request.run_id} is missing System option {name!r}") from exc


class ReleaseBM25System(StrictSystemOptions):
    system_id = "bm25"
    option_names = frozenset({"b", "k1"})
    required_capabilities = frozenset(
        {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
    )
    optional_capabilities: frozenset[str] = frozenset()

    def run(self, request: SystemRunRequest) -> SystemRunResult:
        started_at = time.perf_counter()
        retriever = BM25Retriever(
            request.snapshot.trials,
            k1=_option(request, "k1"),  # type: ignore[arg-type]
            b=_option(request, "b"),  # type: ignore[arg-type]
        )
        candidates = retriever.retrieve_many(
            request.snapshot.topics,
            run_id=request.run_id,
            system_id=self.system_id,
            top_k=request.top_k,
        )
        runtime_seconds = time.perf_counter() - started_at
        ranking = bm25_ranking_configuration(
            k1=retriever.k1,
            b=retriever.b,
            requested_top_k=request.top_k,
            document_count=len(retriever.documents),
        )
        configuration: dict[str, Any] = {
            **ranking,
            "model_identity": {"kind": "none", "reason": "deterministic_bm25"},
            "index_identity": {
                "implementation": "taim-okapi-bm25-v1",
                "persistent": False,
                "corpus_statistics": retriever.corpus_statistics,
            },
            "metric_cutoff": request.metric_cutoff,
            "corpus_statistics": retriever.corpus_statistics,
            "runtime_seconds": runtime_seconds,
        }
        return SystemRunResult(
            candidates=tuple(candidates),
            configuration=configuration,
            runtime_seconds=runtime_seconds,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth=PIPELINE_DEPTH_RETRIEVAL,
        )


class ReleaseFoldedBM25System(StrictSystemOptions):
    """BM25 over a checksum-bound, locally derived folded-query bundle."""

    system_id = "bm25-folded"
    option_names = frozenset({"b", "folded_query_bundle", "k1"})
    required_capabilities = ReleaseBM25System.required_capabilities
    optional_capabilities: frozenset[str] = frozenset()

    def run(self, request: SystemRunRequest) -> SystemRunResult:
        from taim.folded_query_bundle import validate_folded_query_bundle

        raw_bundle = _option(request, "folded_query_bundle")
        if not isinstance(raw_bundle, Mapping):
            raise ValueError("bm25-folded requires a folded query bundle object")
        queries, bundle_id = validate_folded_query_bundle(
            raw_bundle,
            expected_task_input_id=request.snapshot.task_input_id,
            expected_topic_ids=tuple(topic.topic_id for topic in request.snapshot.topics),
        )
        topics = tuple(
            replace(topic, canonical_text=query.canonical_text)
            for topic, query in zip(request.snapshot.topics, queries, strict=True)
        )
        started_at = time.perf_counter()
        retriever = BM25Retriever(
            request.snapshot.trials,
            k1=_option(request, "k1"),  # type: ignore[arg-type]
            b=_option(request, "b"),  # type: ignore[arg-type]
        )
        candidates = retriever.retrieve_many(
            topics,
            run_id=request.run_id,
            system_id=self.system_id,
            top_k=request.top_k,
        )
        runtime_seconds = time.perf_counter() - started_at
        ranking = bm25_ranking_configuration(
            k1=retriever.k1,
            b=retriever.b,
            requested_top_k=request.top_k,
            document_count=len(retriever.documents),
        )
        return SystemRunResult(
            candidates=tuple(candidates),
            configuration={
                **ranking,
                "model_identity": {"kind": "none", "reason": "deterministic_bm25"},
                "index_identity": {
                    "implementation": "taim-okapi-bm25-v1",
                    "persistent": False,
                    "corpus_statistics": retriever.corpus_statistics,
                },
                "query_transformation": {
                    "bundle_id": bundle_id,
                    "policy": raw_bundle["policy"],
                    "provenance": raw_bundle["provenance"],
                    "topic_count": len(queries),
                },
                "metric_cutoff": request.metric_cutoff,
                "corpus_statistics": retriever.corpus_statistics,
                "runtime_seconds": runtime_seconds,
            },
            runtime_seconds=runtime_seconds,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth=PIPELINE_DEPTH_RETRIEVAL,
        )


class ReleaseDenseSystem(StrictSystemOptions):
    option_names = frozenset(
        {
            "corpus_hash",
            "device",
            "document_batch_size",
            "index_dir",
            "query_batch_size",
            "rebuild_index",
        }
    )
    required_capabilities = frozenset(
        {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
    )
    optional_capabilities: frozenset[str] = frozenset()

    def __init__(self, system_id: str) -> None:
        DENSE_ENCODER_REGISTRY.binding_for_system(system_id)
        self.system_id = system_id

    def run(self, request: SystemRunRequest) -> SystemRunResult:
        from taim.baselines.dense import (
            DENSE_BUILD_PROGRESS_FILENAME,
            DENSE_ID_MAP_FILENAME,
            DENSE_INDEX_MANIFEST_FILENAME,
            DENSE_MATRIX_FILENAME,
            DenseRetriever,
            TextEncoder,
            prepare_dense_index,
            reset_dense_index,
        )

        started_at = time.perf_counter()
        index_directory = Path(cast(str | Path, _option(request, "index_dir")))
        if _option(request, "rebuild_index"):
            reset_dense_index(index_directory)
        final_paths = (
            index_directory / DENSE_MATRIX_FILENAME,
            index_directory / DENSE_ID_MAP_FILENAME,
            index_directory / DENSE_INDEX_MANIFEST_FILENAME,
        )
        progress_path = index_directory / DENSE_BUILD_PROGRESS_FILENAME
        completed_rows_at_start = 0
        if all(path.is_file() for path in final_paths):
            index_action = "reused"
        elif progress_path.is_file():
            index_action = "resumed"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            completed = progress.get("completed_rows") if isinstance(progress, dict) else None
            if isinstance(completed, int) and not isinstance(completed, bool):
                completed_rows_at_start = completed
        else:
            index_action = "built"
        encoder = cast(
            TextEncoder,
            DENSE_ENCODER_REGISTRY.create_encoder(
                self.system_id,
                device=_option(request, "device"),  # type: ignore[arg-type]
            ),
        )
        index_started_at = time.perf_counter()
        index = prepare_dense_index(
            request.snapshot.trials,
            directory=index_directory,
            corpus_hash=_option(request, "corpus_hash"),  # type: ignore[arg-type]
            encoder=encoder,
            document_batch_size=_option(request, "document_batch_size"),  # type: ignore[arg-type]
        )
        index_seconds = time.perf_counter() - index_started_at
        retriever = DenseRetriever(
            index,
            encoder,
            query_batch_size=_option(request, "query_batch_size"),  # type: ignore[arg-type]
        )
        retrieval_started_at = time.perf_counter()
        candidates = retriever.retrieve_many(
            request.snapshot.topics,
            run_id=request.run_id,
            system_id=self.system_id,
            top_k=request.top_k,
        )
        retrieval_seconds = time.perf_counter() - retrieval_started_at
        runtime_seconds = time.perf_counter() - started_at
        configuration = retriever.configuration(top_k=request.top_k)
        policy = DENSE_ENCODER_REGISTRY.policy_for_system(self.system_id)
        configuration.update(
            {
                "system_identity": {
                    "system_id": self.system_id,
                    "encoder_id": policy.encoder_id,
                    "encoder_policy_sha256": policy.policy_hash(),
                },
                "metric_cutoff": request.metric_cutoff,
                "model_identity": policy.to_dict(),
                "index_identity": {
                    "corpus_hash": cast(JsonValue, _option(request, "corpus_hash")),
                    "directory": str(index_directory.resolve()),
                    "index_action": index_action,
                    "persistent": True,
                },
                "execution": {
                    "index_action": index_action,
                    "completed_rows_at_start": completed_rows_at_start,
                    "index_seconds": index_seconds,
                    "retrieval_seconds": retrieval_seconds,
                },
                "runtime_seconds": runtime_seconds,
            }
        )
        return SystemRunResult(
            candidates=tuple(candidates),
            configuration=configuration,
            runtime_seconds=runtime_seconds,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth=PIPELINE_DEPTH_RETRIEVAL,
        )


def run_release_reference_system(
    system_id: str,
    request: SystemRunRequest,
) -> SystemRunResult:
    """Run one exported non-fusion System through its exact input contract."""

    extension = extension_for(system_id)
    if extension is not None:
        return extension.run(system_id, request)
    if system_id == "bm25":
        system: System = ReleaseBM25System()
    elif system_id == "bm25-folded":
        system = ReleaseFoldedBM25System()
    elif system_id in {
        "dense-bge-m3",
        "dense-qwen3-embedding-0.6b",
        "dense-qwen3-embedding-0.6b-no-template",
    }:
        system = ReleaseDenseSystem(system_id)
    else:
        raise ValueError(f"System {system_id!r} is not an exported direct reference System")
    return run_normalized_system(system, request)


def run_normalized_system(system: System, request: SystemRunRequest) -> SystemRunResult:
    """Run one System on its normalized request, recording the exact System Input it saw."""

    effective = normalize_system_request(system, request)
    result = system.run(effective)
    return SystemRunResult(
        candidates=result.candidates,
        configuration={**result.configuration, "system_input": effective.system_input_dict()},
        runtime_seconds=result.runtime_seconds,
        primary_ranking=result.primary_ranking,
        pipeline_depth=result.pipeline_depth,
        stage_rankings=result.stage_rankings,
    )


__all__ = [
    "ReleaseBM25System",
    "ReleaseDenseSystem",
    "ReleaseFoldedBM25System",
    "run_release_reference_system",
]
