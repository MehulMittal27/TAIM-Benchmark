# ruff: noqa: S101
from __future__ import annotations

import json
from pathlib import Path

import pytest

from taim.artifacts import load_run, write_run
from taim.schemas import (
    Candidate,
    RelevanceJudgment,
    StageRanking,
)
from tests.benchmark_helpers import manifest_for, prepared_benchmark, trial


def test_write_run_reloads_stage_rankings_without_changing_default_scores(tmp_path: Path) -> None:
    directory = tmp_path / "run-with-stages"
    primary = [Candidate("run-1", "bm25", "1", "primary", 1, 2.0)]
    stages = (
        StageRanking(
            "retrieval",
            "retrieval",
            (Candidate("run-1", "bm25", "1", "eligible", 1, 3.0),),
        ),
        StageRanking(
            "rerank",
            "rerank",
            (Candidate("run-1", "bm25", "1", "eligible", 1, 4.0),),
        ),
    )
    prepared = prepared_benchmark(
        trials=(trial("eligible"), trial("primary")),
        judgments=(RelevanceJudgment("1", "eligible", 2),),
    )
    profile = {
        "profile_id": "test-profile",
        "definition_sha256": "sha256:" + "3" * 64,
        "evaluation": {
            "cutoffs": [1],
            "unjudged_policy": "retain_as_zero_gain",
            "precision_relevance_minimum": 1,
            "precision_denominator": "fixed_cutoff",
        },
    }
    manifest = manifest_for(
        prepared,
        system_id="bm25",
        benchmark_profile=profile,
        pipeline_depth="rerank",
        budget_k=1,
        stage_rankings=stages,
    )

    stored = write_run(
        directory,
        manifest=manifest,
        candidates=primary,
        topics=prepared.topics,
        evaluation_package=prepared.evaluation_package,
        metric_cutoff=1,
        runtime_seconds=0.0,
        benchmark_profile=profile,
    )

    assert [(stage.name, stage.pipeline_depth) for stage in stored.stage_rankings] == [
        ("retrieval", "retrieval"),
        ("rerank", "rerank"),
    ]
    assert stored.stage_rankings[0].candidates == stages[0].candidates
    assert (directory / "stages" / "retrieval.jsonl").is_file()
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["eligible_recall_at_1"] == 0.0

    (directory / "stages" / "retrieval.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        load_run(directory)
