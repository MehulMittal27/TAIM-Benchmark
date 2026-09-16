"""Deterministic reciprocal-rank fusion over two or more candidate runs.

:func:`fuse_rrf` is the frozen two-component baseline and is unchanged.
:func:`fuse_rrf_n` is the same formula, the same constant, the same component
depth and the same tie-break generalised to any number of components; both call
one scoring kernel, so the two-component case is the N-component case by
construction rather than by assertion.  ``tests/test_rrf.py`` pins that.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import cast

from taim.schemas import Candidate, JsonValue

RRF_CONSTANT = 60
RRF_COMPONENT_DEPTH = 1_000
DEFAULT_RRF_OUTPUT_DEPTH = 1_000


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _validated_component(
    candidates: Iterable[Candidate],
    *,
    expected_system_id: str,
) -> tuple[tuple[Candidate, ...], str]:
    rows = tuple(candidates)
    if not rows:
        raise ValueError(f"{expected_system_id} component run must not be empty")

    run_ids: set[str] = set()
    trial_keys: set[tuple[str, str]] = set()
    rank_keys: set[tuple[str, int]] = set()
    for candidate in rows:
        if not isinstance(candidate, Candidate):
            raise TypeError("RRF components must contain Candidate instances")
        if candidate.system_id != expected_system_id:
            raise ValueError(
                f"{expected_system_id} component contains system_id {candidate.system_id!r}"
            )
        run_ids.add(candidate.run_id)
        trial_key = (candidate.topic_id, candidate.trial_id)
        rank_key = (candidate.topic_id, candidate.rank)
        if trial_key in trial_keys:
            raise ValueError(
                f"duplicate {expected_system_id} candidate for topic "
                f"{candidate.topic_id!r} and trial {candidate.trial_id!r}"
            )
        if rank_key in rank_keys:
            raise ValueError(
                f"duplicate {expected_system_id} rank {candidate.rank} "
                f"for topic {candidate.topic_id!r}"
            )
        trial_keys.add(trial_key)
        rank_keys.add(rank_key)
    if len(run_ids) != 1:
        raise ValueError(f"{expected_system_id} candidates must share one run_id")
    return rows, next(iter(run_ids))


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


def _fused_ranking(
    components: Sequence[tuple[Candidate, ...]],
    *,
    run_id: str,
    system_id: str,
    component_depth: int,
    output_depth: int,
) -> list[Candidate]:
    """The one scoring kernel every public fusion entry point goes through.

    Component scores and iterable order are deliberately ignored.  Fractions
    remain exact through ranking, and are converted to floats only when the
    common :class:`~taim.schemas.Candidate` output is constructed, so the fused
    order never depends on binary floating-point rounding.
    """

    scores_by_topic: dict[str, dict[str, Fraction]] = defaultdict(lambda: defaultdict(Fraction))
    for rows in components:
        for candidate in rows:
            if candidate.rank <= component_depth:
                scores_by_topic[candidate.topic_id][candidate.trial_id] += Fraction(
                    1,
                    RRF_CONSTANT + candidate.rank,
                )

    fused: list[Candidate] = []
    for topic_id in sorted(scores_by_topic):
        ranked = sorted(
            scores_by_topic[topic_id].items(),
            key=lambda item: (-item[1], item[0]),
        )[:output_depth]
        fused.extend(
            Candidate(
                run_id=run_id,
                system_id=system_id,
                topic_id=topic_id,
                trial_id=trial_id,
                rank=rank,
                score=float(score),
            )
            for rank, (trial_id, score) in enumerate(ranked, start=1)
        )
    return fused


def fuse_rrf(
    bm25_candidates: Iterable[Candidate],
    dense_candidates: Iterable[Candidate],
    *,
    run_id: str,
    system_id: str = "rrf",
    top_k: int = DEFAULT_RRF_OUTPUT_DEPTH,
) -> list[Candidate]:
    """Fuse the frozen BM25 and dense component top-1000 ranks using ``1 / (60 + rank)``."""

    output_depth = _non_negative_integer(top_k, "top_k")
    bm25_rows, _, dense_rows, _ = _validated_components(
        bm25_candidates,
        dense_candidates,
    )
    return _fused_ranking(
        (bm25_rows, dense_rows),
        run_id=run_id,
        system_id=system_id,
        component_depth=RRF_COMPONENT_DEPTH,
        output_depth=output_depth,
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


def _validated_named_components(
    components: Mapping[str, Iterable[Candidate]],
) -> tuple[tuple[str, tuple[Candidate, ...], str], ...]:
    """Validate each named component the way the two-arm path validates its own.

    The arm name is the caller's label for the component and is not required to
    equal the ``system_id`` its rows carry: a panel may hold two arms produced by
    the same retriever under different query forms, which is precisely the
    configuration this generalisation exists for.  What is required is that each
    component is internally consistent - one ``run_id``, one row per trial per
    topic, one row per rank per topic - and that no two components are the same
    run, since fusing a run with itself would double every one of its ranks.
    """

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping of arm name to candidate rows")
    if len(components) < 2:
        raise ValueError("RRF requires at least two components")

    validated: list[tuple[str, tuple[Candidate, ...], str]] = []
    run_ids: dict[str, str] = {}
    for name in components:
        if not isinstance(name, str) or not name:
            raise ValueError("component arm names must be non-empty strings")
        rows = tuple(components[name])
        if not rows:
            raise ValueError(f"{name} component run must not be empty")
        component_run_ids: set[str] = set()
        trial_keys: set[tuple[str, str]] = set()
        rank_keys: set[tuple[str, int]] = set()
        for candidate in rows:
            if not isinstance(candidate, Candidate):
                raise TypeError("RRF components must contain Candidate instances")
            component_run_ids.add(candidate.run_id)
            trial_key = (candidate.topic_id, candidate.trial_id)
            rank_key = (candidate.topic_id, candidate.rank)
            if trial_key in trial_keys:
                raise ValueError(
                    f"duplicate {name} candidate for topic "
                    f"{candidate.topic_id!r} and trial {candidate.trial_id!r}"
                )
            if rank_key in rank_keys:
                raise ValueError(
                    f"duplicate {name} rank {candidate.rank} for topic {candidate.topic_id!r}"
                )
            trial_keys.add(trial_key)
            rank_keys.add(rank_key)
        if len(component_run_ids) != 1:
            raise ValueError(f"{name} candidates must share one run_id")
        run_id = next(iter(component_run_ids))
        if run_id in run_ids:
            raise ValueError(
                f"components {run_ids[run_id]!r} and {name!r} are the same run {run_id!r}; "
                "fusing a run with itself would double-count every one of its ranks"
            )
        run_ids[run_id] = name
        validated.append((name, rows, run_id))
    return tuple(validated)


def fuse_rrf_n(
    components: Mapping[str, Iterable[Candidate]],
    *,
    run_id: str,
    system_id: str = "rrf",
    top_k: int = DEFAULT_RRF_OUTPUT_DEPTH,
    component_depth: int = RRF_COMPONENT_DEPTH,
) -> list[Candidate]:
    """Fuse any number of component rankings at the frozen constant.

    Same formula, same :data:`RRF_CONSTANT`, same :data:`RRF_COMPONENT_DEPTH` and
    same ``(-score, trial_id)`` tie-break as :func:`fuse_rrf`, which it shares a
    scoring kernel with, so a two-component call reproduces the frozen two-arm
    baseline exactly rather than approximately.  Components are scored in name
    order, but the result does not depend on that order: the kernel sums exact
    fractions and breaks ties on ``trial_id``.
    """

    output_depth = _non_negative_integer(top_k, "top_k")
    effective_component_depth = _non_negative_integer(component_depth, "component_depth")
    validated = _validated_named_components(components)
    return _fused_ranking(
        tuple(rows for _name, rows, _component_run_id in sorted(validated)),
        run_id=run_id,
        system_id=system_id,
        component_depth=effective_component_depth,
        output_depth=output_depth,
    )


def rrf_n_configuration(
    components: Mapping[str, Sequence[Candidate]],
    *,
    top_k: int = DEFAULT_RRF_OUTPUT_DEPTH,
    component_depth: int = RRF_COMPONENT_DEPTH,
) -> dict[str, JsonValue]:
    """Return the complete JSON-compatible N-component RRF retrieval configuration."""

    output_depth = _non_negative_integer(top_k, "top_k")
    effective_component_depth = _non_negative_integer(component_depth, "component_depth")
    validated = _validated_named_components(components)
    return {
        "implementation": "taim.baselines.rrf.fuse_rrf_n",
        "formula": "sum(1 / (60 + component_rank))",
        "constant": RRF_CONSTANT,
        "component_retrieval_depth": effective_component_depth,
        "requested_top_k": output_depth,
        "raw_component_scores_used": False,
        "tie_breaking": "RRF score descending, trial_id ascending",
        "topic_order": "topic_id ascending",
        "component_order": "arm name ascending; the fused order does not depend on it",
        "two_component_case_equals": "taim.baselines.rrf.fuse_rrf",
        "components": [
            {
                "arm": name,
                "system_id": cast(
                    list[JsonValue], sorted({candidate.system_id for candidate in rows})
                ),
                "run_id": component_run_id,
                "candidate_count": len(rows),
                "maximum_input_rank": max(candidate.rank for candidate in rows),
                "topic_count": len({candidate.topic_id for candidate in rows}),
            }
            for name, rows, component_run_id in sorted(validated)
        ],
    }


reciprocal_rank_fusion = fuse_rrf


__all__ = [
    "DEFAULT_RRF_OUTPUT_DEPTH",
    "RRF_COMPONENT_DEPTH",
    "RRF_CONSTANT",
    "fuse_rrf",
    "fuse_rrf_n",
    "reciprocal_rank_fusion",
    "rrf_configuration",
    "rrf_n_configuration",
]
