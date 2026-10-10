"""Shared dense execution and request normalization for selected release units."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

from taim.baselines.encoder_registry import DENSE_ENCODER_REGISTRY, EncoderRegistry
from taim.patient_to_trial import (
    PatientToTrialRunRequest,
    PatientToTrialRunResult,
    StrictSystemOptions,
    required_option,
)
from taim.schemas import PIPELINE_DEPTH_RETRIEVAL, PRIMARY_RANKING_CANDIDATES, JsonValue
from taim.snapshot import CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT


class ReleaseDenseSystem(StrictSystemOptions):
    option_names = frozenset(
        {
            "corpus_hash",
            "device",
            "document_batch_size",
            "checkpoint_interval_rows",
            "index_dir",
            "query_batch_size",
            "rebuild_index",
        }
    )
    required_capabilities = frozenset(
        {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
    )
    optional_capabilities: frozenset[str] = frozenset()

    def __init__(
        self, system_id: str, *, encoder_registry: EncoderRegistry = DENSE_ENCODER_REGISTRY
    ) -> None:
        encoder_registry.binding_for_system(system_id)
        self.system_id = system_id
        self.encoder_registry = encoder_registry

    def run(self, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
        from taim.baselines.dense import (
            DEFAULT_CHECKPOINT_INTERVAL_ROWS,
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
        index_directory = Path(cast(str | Path, required_option(request, "index_dir")))
        if required_option(request, "rebuild_index"):
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
            self.encoder_registry.create_encoder(
                self.system_id,
                device=required_option(request, "device"),  # type: ignore[arg-type]
            ),
        )
        index_started_at = time.perf_counter()
        index = prepare_dense_index(
            request.snapshot.trials,
            directory=index_directory,
            corpus_hash=required_option(request, "corpus_hash"),  # type: ignore[arg-type]
            encoder=encoder,
            document_batch_size=required_option(request, "document_batch_size"),  # type: ignore[arg-type]
            checkpoint_interval_rows=cast(
                int,
                request.options.get("checkpoint_interval_rows", DEFAULT_CHECKPOINT_INTERVAL_ROWS),
            ),
        )
        index_seconds = time.perf_counter() - index_started_at
        retriever = DenseRetriever(
            index,
            encoder,
            query_batch_size=required_option(request, "query_batch_size"),  # type: ignore[arg-type]
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
        policy = self.encoder_registry.policy_for_system(self.system_id)
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
                    "corpus_hash": cast(JsonValue, required_option(request, "corpus_hash")),
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
        return PatientToTrialRunResult(
            candidates=tuple(candidates),
            configuration=configuration,
            runtime_seconds=runtime_seconds,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth=PIPELINE_DEPTH_RETRIEVAL,
        )
