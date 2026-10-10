"""The frozen two-arm fusion release unit and its producer checks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from taim.baselines.rank_fusion import (
    DEFAULT_RRF_OUTPUT_DEPTH,
    RRF_COMPONENT_DEPTH,
    RRF_CONSTANT,
    non_negative_integer,
    validated_component,
)
from taim.schemas import (
    Candidate,
    JsonValue,
)


def validated_components(
    bm25_candidates: Iterable[Candidate],
    dense_candidates: Iterable[Candidate],
) -> tuple[tuple[Candidate, ...], str, tuple[Candidate, ...], str]:
    bm25_rows, bm25_run_id = validated_component(
        bm25_candidates,
        expected_system_id="bm25",
    )
    dense_rows, dense_run_id = validated_component(
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

    output_depth = non_negative_integer(top_k, "top_k")
    bm25_rows, bm25_run_id, dense_rows, dense_run_id = validated_components(
        bm25_candidates,
        dense_candidates,
    )
    return {
        "implementation": "taim.pipelines.rrf.fusion.fuse_rrf",
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
