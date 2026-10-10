# ruff: noqa: S101
from __future__ import annotations

from fractions import Fraction

import pytest

from taim.baselines.rank_fusion import DEFAULT_RRF_OUTPUT_DEPTH, RRF_COMPONENT_DEPTH, RRF_CONSTANT
from taim.pipelines.rrf.fusion import fuse_rrf, rrf_configuration
from taim.schemas import Candidate


def _candidate(
    system_id: str,
    topic_id: str,
    trial_id: str,
    rank: int,
    *,
    score: float = 1.0,
    run_id: str | None = None,
) -> Candidate:
    return Candidate(
        run_id=run_id or f"{system_id}-run",
        system_id=system_id,
        topic_id=topic_id,
        trial_id=trial_id,
        rank=rank,
        score=score,
    )


def test_rrf_uses_the_fixed_formula_over_the_component_union() -> None:
    bm25 = [
        _candidate("bm25", "T1", "A", 1, score=100.0),
        _candidate("bm25", "T1", "B", 2, score=99.0),
    ]
    dense = [
        _candidate("dense-bge-m3", "T1", "B", 1, score=-4.0),
        _candidate("dense-bge-m3", "T1", "C", 2, score=-5.0),
    ]

    fused = fuse_rrf(bm25, dense, run_id="rrf-run")

    assert [(row.trial_id, row.rank) for row in fused] == [("B", 1), ("A", 2), ("C", 3)]
    assert fused[0].score == float(Fraction(1, 62) + Fraction(1, 61))
    assert fused[1].score == float(Fraction(1, 61))
    assert fused[2].score == float(Fraction(1, 62))
    assert all(row.run_id == "rrf-run" and row.system_id == "rrf" for row in fused)


def test_rrf_ties_are_broken_by_trial_id_not_component_or_input_order() -> None:
    bm25 = [
        _candidate("bm25", "T1", "Z", 1),
        _candidate("bm25", "T2", "D", 1),
    ]
    dense = [
        _candidate("dense-bge-m3", "T1", "A", 1),
        _candidate("dense-bge-m3", "T2", "C", 1),
    ]

    forward = fuse_rrf(bm25, dense, run_id="run")
    reverse = fuse_rrf(reversed(bm25), reversed(dense), run_id="run")

    expected = [
        ("T1", "A", 1),
        ("T1", "Z", 2),
        ("T2", "C", 1),
        ("T2", "D", 2),
    ]
    assert [(row.topic_id, row.trial_id, row.rank) for row in forward] == expected
    assert forward == reverse


def test_rrf_ignores_raw_component_scores() -> None:
    first = fuse_rrf(
        [_candidate("bm25", "T", "A", 1, score=1e300)],
        [_candidate("dense-bge-m3", "T", "B", 1, score=-1e300)],
        run_id="run",
    )
    second = fuse_rrf(
        [_candidate("bm25", "T", "A", 1, score=-1e300)],
        [_candidate("dense-bge-m3", "T", "B", 1, score=1e300)],
        run_id="run",
    )

    assert first == second
    assert [row.trial_id for row in first] == ["A", "B"]


def test_rrf_uses_only_component_ranks_through_1000() -> None:
    bm25 = [
        _candidate("bm25", "T", "included", 1_000),
        _candidate("bm25", "T", "excluded", 1_001),
    ]
    dense = [_candidate("dense-bge-m3", "T", "dense-only", 1_000)]

    fused = fuse_rrf(bm25, dense, run_id="run")

    assert [row.trial_id for row in fused] == ["dense-only", "included"]
    assert all(row.score == float(Fraction(1, 1_060)) for row in fused)
    assert RRF_COMPONENT_DEPTH == 1_000


def test_rrf_applies_output_depth_per_topic() -> None:
    bm25 = [
        _candidate("bm25", "T2", "B", 1),
        _candidate("bm25", "T1", "A", 1),
        _candidate("bm25", "T1", "B", 2),
    ]
    dense = [
        _candidate("dense-bge-m3", "T2", "C", 1),
        _candidate("dense-bge-m3", "T1", "C", 1),
    ]

    fused = fuse_rrf(bm25, dense, run_id="run", top_k=1)

    assert [(row.topic_id, row.trial_id, row.rank) for row in fused] == [
        ("T1", "A", 1),
        ("T2", "B", 1),
    ]
    assert fuse_rrf(bm25, dense, run_id="run", top_k=0) == []


def test_rrf_configuration_records_component_runs_and_fixed_inputs() -> None:
    bm25 = [_candidate("bm25", "T", "A", 4)]
    dense = [_candidate("dense-bge-m3", "T", "B", 7)]

    configuration = rrf_configuration(bm25, dense, top_k=25)

    assert configuration == {
        "implementation": "taim.pipelines.rrf.fusion.fuse_rrf",
        "formula": "sum(1 / (60 + component_rank))",
        "constant": 60,
        "component_retrieval_depth": 1_000,
        "requested_top_k": 25,
        "raw_component_scores_used": False,
        "tie_breaking": "RRF score descending, trial_id ascending",
        "topic_order": "topic_id ascending",
        "components": [
            {
                "system_id": "bm25",
                "run_id": "bm25-run",
                "candidate_count": 1,
                "maximum_input_rank": 4,
            },
            {
                "system_id": "dense-bge-m3",
                "run_id": "dense-bge-m3-run",
                "candidate_count": 1,
                "maximum_input_rank": 7,
            },
        ],
    }
    assert RRF_CONSTANT == 60
    assert DEFAULT_RRF_OUTPUT_DEPTH == 1_000


def test_rrf_rejects_duplicate_trials_and_duplicate_ranks() -> None:
    dense = [_candidate("dense-bge-m3", "T", "D", 1)]
    duplicate_trial = _candidate("bm25", "T", "A", 1)
    with pytest.raises(ValueError, match="duplicate bm25 candidate"):
        fuse_rrf([duplicate_trial, duplicate_trial], dense, run_id="run")

    with pytest.raises(ValueError, match="duplicate bm25 rank"):
        fuse_rrf(
            [
                _candidate("bm25", "T", "A", 1),
                _candidate("bm25", "T", "B", 1),
            ],
            dense,
            run_id="run",
        )


def test_rrf_rejects_empty_mixed_or_mislabeled_component_runs() -> None:
    bm25 = [_candidate("bm25", "T", "A", 1)]
    dense = [_candidate("dense-bge-m3", "T", "B", 1)]

    with pytest.raises(ValueError, match="must not be empty"):
        fuse_rrf([], dense, run_id="run")
    with pytest.raises(ValueError, match="share one run_id"):
        fuse_rrf(
            [
                *bm25,
                _candidate("bm25", "T", "C", 2, run_id="other-bm25-run"),
            ],
            dense,
            run_id="run",
        )
    with pytest.raises(ValueError, match="contains system_id"):
        fuse_rrf(
            [_candidate("dense-bge-m3", "T", "A", 1, run_id="wrong-run")],
            dense,
            run_id="run",
        )
    with pytest.raises(ValueError, match="distinct run_id"):
        fuse_rrf(
            [_candidate("bm25", "T", "A", 1, run_id="same-run")],
            [_candidate("dense-bge-m3", "T", "B", 1, run_id="same-run")],
            run_id="run",
        )


@pytest.mark.parametrize("top_k", [-1, True, 1.5])
def test_rrf_rejects_invalid_output_depth(top_k: object) -> None:
    bm25 = [_candidate("bm25", "T", "A", 1)]
    dense = [_candidate("dense-bge-m3", "T", "B", 1)]

    with pytest.raises(ValueError, match="top_k"):
        fuse_rrf(bm25, dense, run_id="run", top_k=top_k)  # type: ignore[arg-type]
