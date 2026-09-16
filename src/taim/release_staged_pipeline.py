"""Release 0.1 declaration and deterministic seams for the paper pipeline."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from taim.baselines.rrf import fuse_rrf_n, rrf_n_configuration
from taim.contracts import require_sha256
from taim.folded_query_bundle import (
    FOLDED_QUERY_BUNDLE_SCHEMA_VERSION,
    FoldedQuery,
    build_folded_query_bundle,
    validate_folded_query_bundle,
)
from taim.schemas import (
    PIPELINE_DEPTH_RETRIEVAL,
    Candidate,
    JsonValue,
    StageRanking,
    json_value_to_builtins,
)
from taim.snapshot import TrialDocument

STAGED_PIPELINE_ID = "staged-bm25-qwen3-rrf-rerank-v1"
STAGED_PIPELINE_SYSTEM_ID = "staged-bm25-qwen3-rrf-rerank"

QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_INSTRUCTION = (
    "Given a clinical description of a patient, judge whether this patient is eligible to "
    "enrol in the given clinical trial."
)

ELIGIBILITY_PROVENANCE_PREFIX = "/clinical_study/eligibility"
SUMMARY_HEAD_ORDER = ("CONDITION", "BRIEF_TITLE", "INTERVENTION")
TYPED_CORE_ORDER = ("minimum_age", "maximum_age", "sex", "healthy_volunteers")

_PIPELINE_FIELDS = {
    "arms",
    "benchmark_profile_ids",
    "component_depth",
    "fusion",
    "licensed_inputs",
    "output_depth",
    "pipeline_id",
    "pipeline_version",
    "reranker",
    "system_id",
    "track",
}
_RERANKER_RUNTIME_FIELDS = {
    "attention_backend_enabled",
    "attention_backend_requested",
    "attention_backend_selected",
    "attention_implementation",
    "batch_composition_sha256",
    "batch_size",
    "cudnn",
    "dtype",
    "f_summary_fallback_count",
    "gpu",
    "gpu_compute_capability",
    "gpu_memory_bytes",
    "last_position_only_logits",
    "max_length",
    "model_id",
    "model_revision",
    "nvidia_driver",
    "pairs_per_second",
    "pairs_resumed_from_checkpoint",
    "pairs_scored",
    "pairs_truncated",
    "peak_memory_allocated_bytes",
    "python",
    "scoring_input_sha256",
    "tokenizer_revision",
    "torch",
    "torch_cuda",
    "transformers",
    "use_cache",
    "wall_seconds",
}


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        json_value_to_builtins(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _positive_int(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{role} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class StagedPipeline:
    pipeline_id: str
    system_id: str
    track: str
    profiles: tuple[str, ...]
    component_depth: int
    fusion_depth: int
    scoring_depth: int
    output_depth: int
    arms: tuple[str, ...]
    arm_system_ids: tuple[str, ...]
    fusion: Mapping[str, JsonValue]
    reranker: Mapping[str, JsonValue]
    licensed_inputs: tuple[Mapping[str, JsonValue], ...]
    definition_sha256: str


def load_staged_pipeline() -> StagedPipeline:
    """Load and fail closed over the one staged System admitted to Release 0.1."""

    resource = files("taim").joinpath("data", "pipelines", "staged-bm25-qwen3-rrf-rerank-v1.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _PIPELINE_FIELDS:
        raise ValueError("staged pipeline declaration has unexpected or missing fields")
    if (
        payload["pipeline_id"] != STAGED_PIPELINE_ID
        or payload["system_id"] != STAGED_PIPELINE_SYSTEM_ID
        or payload["pipeline_version"] != "1.0"
        or payload["track"] != "trec-ct-2021"
        or payload["benchmark_profile_ids"]
        != [
            "official-full",
            "trec-ct-2021-external-fidelity-26149",
            "trec-ct-2021-judgment-union",
        ]
    ):
        raise ValueError("staged pipeline declaration has an unsupported identity")
    arms = payload["arms"]
    if (
        not isinstance(arms, list)
        or len(arms) != 2
        or any(not isinstance(row, dict) for row in arms)
    ):
        raise ValueError("staged pipeline must declare exactly two arms")
    expected_arms = (
        ("bm25-folded", "bm25-folded", "folded"),
        (
            "qwen3-raw-no-template",
            "dense-qwen3-embedding-0.6b-no-template",
            "raw-no-template",
        ),
    )
    observed_arms = tuple(
        (row.get("name"), row.get("system_id"), row.get("query_form")) for row in arms
    )
    if observed_arms != expected_arms:
        raise ValueError("staged pipeline arms do not match the frozen paper System")
    fusion = payload["fusion"]
    reranker = payload["reranker"]
    if not isinstance(fusion, dict) or not isinstance(reranker, dict):
        raise ValueError("staged pipeline fusion and reranker must be objects")
    component_depth = _positive_int(payload["component_depth"], "component_depth")
    fusion_depth = _positive_int(fusion.get("output_depth"), "fusion output_depth")
    scoring_depth = _positive_int(reranker.get("scoring_depth"), "reranker scoring_depth")
    output_depth = _positive_int(payload["output_depth"], "output_depth")
    if (component_depth, fusion_depth, scoring_depth, output_depth) != (5_000, 2_000, 2_000, 1_000):
        raise ValueError("staged pipeline depths do not match the frozen shipping System")
    if set(fusion) != {
        "constant",
        "implementation",
        "output_depth",
        "raw_component_scores_used",
        "tie_breaking",
    } or (
        fusion.get("constant") != 60
        or fusion.get("implementation") != "taim.baselines.rrf.fuse_rrf_n"
        or fusion.get("output_depth") != scoring_depth
        or fusion.get("raw_component_scores_used") is not False
        or fusion.get("tie_breaking") != "RRF score descending, trial_id ascending"
    ):
        raise ValueError("staged pipeline fusion does not feed the exact reranker pool")
    if set(reranker) != {
        "attention_backend",
        "batch_size",
        "dtype",
        "format",
        "implementation",
        "max_length",
        "model_id",
        "model_revision",
        "ordering",
        "scoring_depth",
        "tokenizer_revision",
    } or (
        reranker.get("model_id") != "Qwen/Qwen3-Reranker-4B"
        or reranker.get("model_revision") != "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
        or reranker.get("tokenizer_revision") != reranker.get("model_revision")
        or reranker.get("batch_size") != 16
        or reranker.get("max_length") != 2_048
        or reranker.get("dtype") != "float16"
        or reranker.get("attention_backend") != "runtime-recorded"
        or reranker.get("format") != "F-summary folded query"
        or reranker.get("implementation") != "taim.release_staged_pipeline.score_staged_pairs"
        or reranker.get("ordering") != "score descending, fused rank ascending"
    ):
        raise ValueError("staged pipeline reranker does not match the frozen shipping recipe")
    licensed_inputs = payload["licensed_inputs"]
    expected_licensed_input = {
        "acquisition": "user-supplied",
        "input_id": "SNOMED-CT:user-supplied-release",
        "output_policy": "source-topic-spans-only-no-concept-identifiers-or-descriptions",
        "redistribution": "not-included",
        "role": "folded-query-lexicon",
    }
    if licensed_inputs != [expected_licensed_input]:
        raise ValueError("staged pipeline licensed runtime inputs changed")
    return StagedPipeline(
        pipeline_id=STAGED_PIPELINE_ID,
        system_id=STAGED_PIPELINE_SYSTEM_ID,
        track=cast(str, payload["track"]),
        profiles=cast(tuple[str, ...], tuple(payload["benchmark_profile_ids"])),
        component_depth=component_depth,
        fusion_depth=fusion_depth,
        scoring_depth=scoring_depth,
        output_depth=output_depth,
        arms=tuple(row[0] for row in expected_arms),
        arm_system_ids=tuple(row[1] for row in expected_arms),
        fusion=cast(Mapping[str, JsonValue], fusion),
        reranker=cast(Mapping[str, JsonValue], reranker),
        licensed_inputs=(cast(Mapping[str, JsonValue], expected_licensed_input),),
        definition_sha256="sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
    )


def _typed_core_value(field: Mapping[str, object]) -> str:
    value = field.get("value")
    if isinstance(value, Mapping):
        return f"{value['amount']} {value['unit']}"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str) and value in {"True", "False"}:
        return value.lower()
    return str(value)


def _canonical_trial_lines(record: Mapping[str, object]) -> list[tuple[str, str, str]]:
    lines: list[tuple[str, str, str]] = []
    sections = record.get("sections")
    if not isinstance(sections, list):
        raise ValueError("trial record lacks Semantic Text Sections")
    for raw in sections:
        if not isinstance(raw, Mapping):
            raise ValueError("trial Semantic Text Section is invalid")
        provenance = raw.get("provenance") or [{}]
        if (
            not isinstance(provenance, list)
            or not provenance
            or not isinstance(provenance[0], Mapping)
        ):
            raise ValueError("trial Semantic Text Section provenance is invalid")
        lines.append(
            (
                cast(str, raw["role"]).upper(),
                cast(str, raw["text"]),
                cast(str, provenance[0].get("location", "")),
            )
        )
    core = record.get("typed_clinical_core") or {}
    if not isinstance(core, Mapping):
        raise ValueError("trial Typed Clinical Core is invalid")
    for key in TYPED_CORE_ORDER:
        field = core.get(key)
        if not isinstance(field, Mapping):
            continue
        provenance = field.get("provenance") or [{}]
        if (
            not isinstance(provenance, list)
            or not provenance
            or not isinstance(provenance[0], Mapping)
        ):
            raise ValueError("trial Typed Clinical Core provenance is invalid")
        lines.append(
            (key.upper(), _typed_core_value(field), cast(str, provenance[0].get("location", "")))
        )
    return lines


def _render_trial_lines(lines: Sequence[tuple[str, str, str]]) -> str:
    return "\n".join(f"[{header}] {text}" for header, text, _location in lines)


def render_f_summary(trial: TrialDocument) -> tuple[str, bool]:
    """Render the frozen condition/title/intervention-plus-criteria reranker document."""

    record = trial.to_dict()
    lines = _canonical_trial_lines(record)
    if _render_trial_lines(lines) != trial.canonical_text:
        raise ValueError(f"trial {trial.trial_id!r} does not reconstruct its canonical text")
    selected: list[tuple[str, str, str]] = []
    for header in SUMMARY_HEAD_ORDER:
        selected.extend(line for line in lines if line[0] == header)
    selected.extend(line for line in lines if line[2].startswith(ELIGIBILITY_PROVENANCE_PREFIX))
    fallback = not selected
    if fallback:
        selected = [line for line in lines if line[0] == "BRIEF_TITLE"] or lines[:1]
    text = _render_trial_lines(selected)
    if not text:
        raise ValueError(f"trial {trial.trial_id!r} has no reranker-visible text")
    return text, fallback


def rerank_scored_candidates(
    fused_candidates: Sequence[Candidate],
    scores: Mapping[tuple[str, str], float],
    *,
    run_id: str,
    system_id: str,
) -> tuple[Candidate, ...]:
    """Sort every scored pair by score and use fused rank as the only tie-break."""

    rows = tuple(fused_candidates)
    expected = {(row.topic_id, row.trial_id) for row in rows}
    if len(expected) != len(rows) or set(scores) != expected:
        raise ValueError("reranker requires exactly one score for every fused pair")
    by_topic: dict[str, list[Candidate]] = defaultdict(list)
    for row in rows:
        by_topic[row.topic_id].append(row)
    reranked: list[Candidate] = []
    for topic_id in sorted(by_topic):
        ranked = sorted(
            by_topic[topic_id],
            key=lambda row: (-float(scores[(row.topic_id, row.trial_id)]), row.rank),
        )
        reranked.extend(
            Candidate(
                run_id=run_id,
                system_id=system_id,
                topic_id=topic_id,
                trial_id=row.trial_id,
                rank=rank,
                score=float(scores[(row.topic_id, row.trial_id)]),
            )
            for rank, row in enumerate(ranked, start=1)
        )
    return tuple(reranked)


def _as_pipeline_stage(
    candidates: Sequence[Candidate],
    *,
    run_id: str,
) -> tuple[Candidate, ...]:
    return tuple(
        replace(row, run_id=run_id, system_id=STAGED_PIPELINE_SYSTEM_ID) for row in candidates
    )


def assemble_staged_pipeline(
    components: Mapping[str, Sequence[Candidate]],
    scores: Mapping[tuple[str, str], float],
    *,
    benchmark_profile_id: str,
    run_id: str,
) -> tuple[tuple[Candidate, ...], tuple[StageRanking, ...], dict[str, JsonValue]]:
    """Assemble the fixed two-arm RRF and scored reranker stages into one System output."""

    pipeline = load_staged_pipeline()
    if benchmark_profile_id not in pipeline.profiles:
        raise ValueError(
            f"staged pipeline does not support Benchmark Profile {benchmark_profile_id!r}"
        )
    if tuple(sorted(components)) != tuple(sorted(pipeline.arms)):
        raise ValueError("staged pipeline components do not match the declared arms")
    fused = tuple(
        fuse_rrf_n(
            components,
            run_id=run_id,
            system_id=STAGED_PIPELINE_SYSTEM_ID,
            top_k=pipeline.fusion_depth,
            component_depth=pipeline.component_depth,
        )
    )
    reranked = rerank_scored_candidates(
        fused,
        scores,
        run_id=run_id,
        system_id=STAGED_PIPELINE_SYSTEM_ID,
    )
    primary = tuple(row for row in reranked if row.rank <= pipeline.output_depth)
    stages = (
        StageRanking(
            "bm25-folded-depth5000",
            PIPELINE_DEPTH_RETRIEVAL,
            _as_pipeline_stage(components["bm25-folded"], run_id=run_id),
        ),
        StageRanking(
            "qwen3-no-template-depth5000",
            PIPELINE_DEPTH_RETRIEVAL,
            _as_pipeline_stage(components["qwen3-raw-no-template"], run_id=run_id),
        ),
        StageRanking("rrf-scoring-pool-depth2000", PIPELINE_DEPTH_RETRIEVAL, fused),
        StageRanking("rerank-scored-depth2000", "rerank", reranked),
    )
    configuration: dict[str, JsonValue] = {
        "implementation": "taim.release_staged_pipeline.assemble_staged_pipeline",
        "pipeline": {
            "pipeline_id": pipeline.pipeline_id,
            "definition_sha256": pipeline.definition_sha256,
            "track": pipeline.track,
            "profile": benchmark_profile_id,
            "component_depth": pipeline.component_depth,
            "fusion_depth": pipeline.fusion_depth,
            "scoring_depth": pipeline.scoring_depth,
            "output_depth": pipeline.output_depth,
            "arms": list(pipeline.arms),
            "fusion": dict(pipeline.fusion),
            "reranker": dict(pipeline.reranker),
            "licensed_inputs": [dict(item) for item in pipeline.licensed_inputs],
        },
        "fusion": rrf_n_configuration(
            components,
            top_k=pipeline.fusion_depth,
            component_depth=pipeline.component_depth,
        ),
        "model_identity": {
            "model_id": pipeline.reranker["model_id"],
            "model_revision": pipeline.reranker["model_revision"],
            "tokenizer_revision": pipeline.reranker["tokenizer_revision"],
        },
        "index_identity": {
            "kind": "validated_component_runs",
            "components": {
                name: {
                    "run_id": rows[0].run_id,
                    "candidate_count": len(rows),
                    "candidates_sha256": "sha256:"
                    + hashlib.sha256(
                        "".join(f"{row.to_json()}\n" for row in rows).encode()
                    ).hexdigest(),
                }
                for name, rows in sorted(components.items())
            },
        },
    }
    return primary, stages, configuration


def staged_scoring_input_sha256(
    candidates: Sequence[Candidate],
    *,
    folded_queries: Mapping[str, str],
    trial_documents: Mapping[str, str],
) -> str:
    """Bind reranker scores to the exact ordered pairs and rendered input text."""

    rows = tuple(candidates)
    if any(
        row.topic_id not in folded_queries or row.trial_id not in trial_documents for row in rows
    ):
        raise ValueError("staged reranker input text does not cover the scoring pool")
    payload = [
        {
            "topic_id": row.topic_id,
            "trial_id": row.trial_id,
            "fused_rank": row.rank,
            "query": folded_queries[row.topic_id],
            "document": trial_documents[row.trial_id],
        }
        for row in rows
    ]
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validate_reranker_runtime(
    runtime: object,
    *,
    expected_pairs: int,
    expected_scoring_input_sha256: str,
) -> dict[str, JsonValue]:
    if not isinstance(runtime, Mapping) or set(runtime) != _RERANKER_RUNTIME_FIELDS:
        raise ValueError("reranker score artifact runtime has unexpected or missing fields")
    builtins = cast(dict[str, JsonValue], json_value_to_builtins(runtime))
    pipeline = load_staged_pipeline()
    exact = {
        "batch_size": pipeline.reranker["batch_size"],
        "dtype": pipeline.reranker["dtype"],
        "max_length": pipeline.reranker["max_length"],
        "model_id": pipeline.reranker["model_id"],
        "model_revision": pipeline.reranker["model_revision"],
        "scoring_input_sha256": expected_scoring_input_sha256,
        "tokenizer_revision": pipeline.reranker["tokenizer_revision"],
        "use_cache": False,
    }
    if any(builtins[key] != value for key, value in exact.items()):
        raise ValueError("reranker score artifact runtime does not match the frozen recipe")
    backend = builtins["attention_backend_selected"]
    if (
        backend not in {"flash", "mem_efficient", "math"}
        or builtins["attention_backend_requested"] != backend
        or builtins["attention_implementation"] != "sdpa"
        or builtins["attention_backend_enabled"]
        != {
            "flash": backend == "flash",
            "math": backend == "math",
            "mem_efficient": backend == "mem_efficient",
        }
    ):
        raise ValueError("reranker score artifact does not identify one executed attention backend")
    nonnegative_fields = {
        "f_summary_fallback_count",
        "pairs_resumed_from_checkpoint",
        "pairs_truncated",
        "peak_memory_allocated_bytes",
    }
    positive_fields = {"gpu_memory_bytes", "pairs_scored"}
    for key in nonnegative_fields | positive_fields:
        value = builtins[key]
        minimum = 1 if key in positive_fields else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"reranker score artifact runtime {key} is invalid")
    if (
        builtins["pairs_scored"] != expected_pairs
        or cast(int, builtins["pairs_resumed_from_checkpoint"]) > expected_pairs
        or cast(int, builtins["pairs_truncated"]) > expected_pairs
        or cast(int, builtins["f_summary_fallback_count"]) > expected_pairs
    ):
        raise ValueError("reranker score artifact runtime pair counts are inconsistent")
    for key in ("batch_composition_sha256", "scoring_input_sha256"):
        try:
            require_sha256(cast(str, builtins[key]), f"reranker runtime {key}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reranker score artifact runtime {key} is invalid") from exc
    for key in (
        "gpu",
        "gpu_compute_capability",
        "model_id",
        "model_revision",
        "nvidia_driver",
        "python",
        "tokenizer_revision",
        "torch",
        "torch_cuda",
        "transformers",
    ):
        if not isinstance(builtins[key], str) or not builtins[key]:
            raise ValueError(f"reranker score artifact runtime {key} is invalid")
    for key in ("pairs_per_second", "wall_seconds"):
        value = builtins[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"reranker score artifact runtime {key} is invalid")
    cudnn = builtins["cudnn"]
    if cudnn is not None and (isinstance(cudnn, bool) or not isinstance(cudnn, int) or cudnn < 1):
        raise ValueError("reranker score artifact runtime cudnn is invalid")
    logits = builtins["last_position_only_logits"]
    if logits is not None and logits not in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}):
        raise ValueError("reranker score artifact runtime logits optimization is invalid")
    return builtins


def load_reranker_score_artifact(
    path: Path,
    *,
    fused_candidates: Sequence[Candidate],
    pipeline_definition_sha256: str,
    scoring_input_sha256: str,
) -> tuple[dict[tuple[str, str], float], dict[str, JsonValue]]:
    """Load a resumable/offline score artifact without treating it as effectiveness output."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read reranker score artifact: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "pipeline_definition_sha256",
        "scoring_input_sha256",
        "runtime",
        "scores",
    }:
        raise ValueError("reranker score artifact has unexpected or missing fields")
    if (
        payload["schema_version"] != "1.0"
        or payload["pipeline_definition_sha256"] != pipeline_definition_sha256
        or payload["scoring_input_sha256"] != scoring_input_sha256
    ):
        raise ValueError("reranker score artifact does not match the staged scoring input")
    raw_scores = payload["scores"]
    if not isinstance(raw_scores, list):
        raise ValueError("reranker score artifact scores must be an array")
    scores: dict[tuple[str, str], float] = {}
    for raw in raw_scores:
        if not isinstance(raw, Mapping) or set(raw) != {"topic_id", "trial_id", "score"}:
            raise ValueError("reranker score row has unexpected or missing fields")
        key = (cast(str, raw["topic_id"]), cast(str, raw["trial_id"]))
        score = raw["score"]
        if (
            key in scores
            or isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            raise ValueError("reranker score rows must be unique and finite")
        scores[key] = float(score)
    expected = {(row.topic_id, row.trial_id) for row in fused_candidates}
    if set(scores) != expected:
        raise ValueError("reranker score artifact does not exactly cover the fused scoring pool")
    runtime = _validate_reranker_runtime(
        payload["runtime"],
        expected_pairs=len(expected),
        expected_scoring_input_sha256=scoring_input_sha256,
    )
    return scores, runtime


def write_reranker_score_artifact(
    path: Path,
    *,
    scores: Mapping[tuple[str, str], float],
    runtime: Mapping[str, JsonValue],
    pipeline_definition_sha256: str,
    scoring_input_sha256: str,
) -> None:
    validated_runtime = _validate_reranker_runtime(
        runtime,
        expected_pairs=len(scores),
        expected_scoring_input_sha256=scoring_input_sha256,
    )
    payload = {
        "schema_version": "1.0",
        "pipeline_definition_sha256": pipeline_definition_sha256,
        "scoring_input_sha256": scoring_input_sha256,
        "runtime": validated_runtime,
        "scores": [
            {"topic_id": topic_id, "trial_id": trial_id, "score": score}
            for (topic_id, trial_id), score in sorted(scores.items())
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _checkpoint_payload(
    *,
    completed_count: int,
    ordered_rows: Sequence[Candidate],
    scores: Mapping[tuple[str, str], float],
    pipeline_definition_sha256: str,
    scoring_input_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "pipeline_definition_sha256": pipeline_definition_sha256,
        "scoring_input_sha256": scoring_input_sha256,
        "completed_count": completed_count,
        "scores": [
            {
                "topic_id": row.topic_id,
                "trial_id": row.trial_id,
                "score": scores[(row.topic_id, row.trial_id)],
            }
            for row in ordered_rows[:completed_count]
        ],
    }


def _write_reranker_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _load_reranker_checkpoint(
    path: Path,
    *,
    ordered_rows: Sequence[Candidate],
    batch_size: int,
    pipeline_definition_sha256: str,
    scoring_input_sha256: str,
) -> tuple[int, dict[tuple[str, str], float]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read reranker checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "pipeline_definition_sha256",
        "scoring_input_sha256",
        "completed_count",
        "scores",
    }:
        raise ValueError("reranker checkpoint has unexpected or missing fields")
    if (
        payload["schema_version"] != "1.0"
        or payload["pipeline_definition_sha256"] != pipeline_definition_sha256
        or payload["scoring_input_sha256"] != scoring_input_sha256
    ):
        raise ValueError("reranker checkpoint does not match the staged scoring input")
    completed_count = payload["completed_count"]
    if (
        isinstance(completed_count, bool)
        or not isinstance(completed_count, int)
        or completed_count < 0
        or completed_count > len(ordered_rows)
        or (completed_count != len(ordered_rows) and completed_count % batch_size != 0)
    ):
        raise ValueError("reranker checkpoint is not aligned to the frozen batch schedule")
    raw_scores = payload["scores"]
    if not isinstance(raw_scores, list) or len(raw_scores) != completed_count:
        raise ValueError("reranker checkpoint score count does not match its completed prefix")
    scores: dict[tuple[str, str], float] = {}
    for row, raw in zip(ordered_rows[:completed_count], raw_scores, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {"topic_id", "trial_id", "score"}:
            raise ValueError("reranker checkpoint score row is invalid")
        score = raw["score"]
        if (
            raw["topic_id"] != row.topic_id
            or raw["trial_id"] != row.trial_id
            or isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            raise ValueError("reranker checkpoint does not match its ordered scoring prefix")
        scores[(row.topic_id, row.trial_id)] = float(score)
    return completed_count, scores


def _nvidia_driver_version() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "unavailable"
    try:
        result = subprocess.run(  # noqa: S603 - resolved executable, fixed arguments
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else "unavailable"


def score_staged_pairs(
    candidates: Sequence[Candidate],
    *,
    folded_queries: Mapping[str, str],
    trial_documents: Mapping[str, str],
    model_cache: Path | None,
    attention_backend: str,
    local_files_only: bool,
    checkpoint_path: Path | None = None,
) -> tuple[dict[tuple[str, str], float], dict[str, JsonValue]]:
    """Score the fixed 2,000-deep pool with the pinned Qwen3 reranker recipe."""

    pipeline = load_staged_pipeline()
    if attention_backend not in {"flash", "mem_efficient", "math"}:
        raise ValueError("unsupported reranker attention backend")
    rows = tuple(candidates)
    if not rows:
        raise ValueError("staged reranker scoring pool must not be empty")
    scoring_input_id = staged_scoring_input_sha256(
        rows,
        folded_queries=folded_queries,
        trial_documents=trial_documents,
    )

    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised by clean-install verification
        raise ValueError("staged reranking requires the release 'staged' extra") from exc
    if not torch.cuda.is_available():
        raise ValueError("staged reranking requires CUDA; CPU fallback would change the recipe")
    torch.backends.cuda.enable_flash_sdp(attention_backend == "flash")
    torch.backends.cuda.enable_mem_efficient_sdp(attention_backend == "mem_efficient")
    torch.backends.cuda.enable_math_sdp(attention_backend == "math")

    model_id = cast(str, pipeline.reranker["model_id"])
    revision = cast(str, pipeline.reranker["model_revision"])
    snapshot = snapshot_download(
        model_id,
        revision=revision,
        cache_dir=model_cache,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, padding_side="left", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.float16,
        local_files_only=True,
    )
    device = torch.device("cuda")
    cast(Any, model).to(device).eval()
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    if attention_implementation != "sdpa":
        raise ValueError("pinned reranker must execute through the forced SDPA attention backend")
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    if (
        not isinstance(yes_id, int)
        or not isinstance(no_id, int)
        or yes_id == no_id
        or yes_id == tokenizer.unk_token_id
        or no_id == tokenizer.unk_token_id
    ):
        raise ValueError("pinned reranker tokenizer cannot resolve distinct yes/no token IDs")
    last_position_only: dict[str, int] = {}
    for keyword in ("logits_to_keep", "num_logits_to_keep"):
        if keyword in inspect.signature(model.forward).parameters:
            last_position_only = {keyword: 1}
            break

    prefix_ids = tokenizer(QWEN_PREFIX, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(QWEN_SUFFIX, add_special_tokens=False)["input_ids"]
    maximum_length = cast(int, pipeline.reranker["max_length"])
    budget = maximum_length - len(prefix_ids) - len(suffix_ids)
    encoded: list[tuple[int, list[int]]] = []
    truncated = 0
    for start in range(0, len(rows), 256):
        chunk = rows[start : start + 256]
        bodies = [
            f"<Instruct>: {QWEN_INSTRUCTION}\n"
            f"<Query>: {folded_queries[row.topic_id]}\n"
            f"<Document>: {trial_documents[row.trial_id]}"
            for row in chunk
        ]
        for offset, body_ids in enumerate(tokenizer(bodies, add_special_tokens=False)["input_ids"]):
            truncated += len(body_ids) > budget
            encoded.append((start + offset, prefix_ids + body_ids[:budget] + suffix_ids))
    order = sorted(range(len(encoded)), key=lambda index: (-len(encoded[index][1]), index))
    batch_size = cast(int, pipeline.reranker["batch_size"])
    ordered_rows = tuple(rows[encoded[index][0]] for index in order)
    completed_count = 0
    completed_scores: dict[tuple[str, str], float] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        completed_count, completed_scores = _load_reranker_checkpoint(
            checkpoint_path,
            ordered_rows=ordered_rows,
            batch_size=batch_size,
            pipeline_definition_sha256=pipeline.definition_sha256,
            scoring_input_sha256=scoring_input_id,
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if not isinstance(pad_id, int):
        raise ValueError("pinned reranker tokenizer has no usable padding token")

    raw_scores: list[float | None] = [None] * len(rows)
    for index in order[:completed_count]:
        row = rows[encoded[index][0]]
        raw_scores[encoded[index][0]] = completed_scores[(row.topic_id, row.trial_id)]
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    with torch.inference_mode():
        for start in range(completed_count, len(order), batch_size):
            block = order[start : start + batch_size]
            batch = [encoded[index][1] for index in block]
            width = max(len(ids) for ids in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row_index, ids in enumerate(batch):
                input_ids[row_index, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
                mask[row_index, width - len(ids) :] = 1
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=mask.to(device),
                use_cache=False,
                **last_position_only,
            ).logits[:, -1, :]
            pair_logits = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
            normalised = torch.log_softmax(pair_logits, dim=1)
            block_scores = (normalised[:, 1] - normalised[:, 0]).tolist()
            for encoded_index, score in zip(block, block_scores, strict=True):
                raw_scores[encoded[encoded_index][0]] = float(score)
            if checkpoint_path is not None:
                checkpoint_scores = {
                    (row.topic_id, row.trial_id): cast(float, raw_scores[encoded[index][0]])
                    for index in order[: start + len(block)]
                    for row in (rows[encoded[index][0]],)
                }
                _write_reranker_checkpoint(
                    checkpoint_path,
                    _checkpoint_payload(
                        completed_count=start + len(block),
                        ordered_rows=ordered_rows,
                        scores=checkpoint_scores,
                        pipeline_definition_sha256=pipeline.definition_sha256,
                        scoring_input_sha256=scoring_input_id,
                    ),
                )
    elapsed = time.perf_counter() - started_at
    if any(score is None for score in raw_scores):
        raise ValueError("reranker scoring did not produce every frozen pair")
    scores = {
        (row.topic_id, row.trial_id): cast(float, score)
        for row, score in zip(rows, raw_scores, strict=True)
    }
    runtime: dict[str, JsonValue] = {
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "dtype": "float16",
        "max_length": maximum_length,
        "batch_size": batch_size,
        "attention_backend_requested": attention_backend,
        "attention_backend_selected": attention_backend,
        "attention_implementation": attention_implementation,
        "attention_backend_enabled": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "last_position_only_logits": cast(JsonValue, last_position_only or None),
        "use_cache": False,
        "pairs_scored": len(rows),
        "pairs_resumed_from_checkpoint": completed_count,
        "scoring_input_sha256": scoring_input_id,
        "pairs_truncated": truncated,
        "batch_composition_sha256": "sha256:"
        + hashlib.sha256(
            _canonical_json_bytes(
                [
                    [rows[index].topic_id, rows[index].trial_id, len(encoded[index][1])]
                    for index in order
                ]
            )
        ).hexdigest(),
        "wall_seconds": elapsed,
        "pairs_per_second": len(rows) / elapsed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability(0)
        ),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "nvidia_driver": _nvidia_driver_version(),
    }
    return scores, runtime


__all__ = [
    "FOLDED_QUERY_BUNDLE_SCHEMA_VERSION",
    "STAGED_PIPELINE_ID",
    "STAGED_PIPELINE_SYSTEM_ID",
    "FoldedQuery",
    "StagedPipeline",
    "assemble_staged_pipeline",
    "build_folded_query_bundle",
    "load_reranker_score_artifact",
    "load_staged_pipeline",
    "render_f_summary",
    "rerank_scored_candidates",
    "score_staged_pairs",
    "staged_scoring_input_sha256",
    "validate_folded_query_bundle",
    "write_reranker_score_artifact",
]
