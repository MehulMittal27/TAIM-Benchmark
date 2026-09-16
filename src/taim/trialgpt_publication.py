"""Fail-closed publication contracts for the TrialGPT TAIM Luna benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from taim.artifacts import sha256_file
from taim.contracts import content_sha256, require_exact_keys, require_sha256
from taim.schemas import (
    Candidate,
    JsonValue,
    RelevanceJudgment,
    SchemaValidationError,
    json_value_to_builtins,
)
from taim.trialgpt_paper import TRIALGPT_TOKENIZER_ID

TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID = "TrialGPT-TAIM-Luna-v1"
TRIALGPT_PUBLICATION_CONTRACT_VERSION = "1.0"
TRIALGPT_PUBLICATION_CANDIDATE_DEPTH = 500
TRIALGPT_PUBLICATION_LOGICAL_CALLS_PER_TOPIC = 1_500
TRIALGPT_PUBLICATION_RETRIEVAL_STAGE = "three-luna-consensus"

_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_ROW_KEYS = {"configuration", "rank", "score", "topic_id", "trial_id"}
_LOCK_KEYS = {
    "artifact_type",
    "schema_version",
    "system_id",
    "benchmark_lineage",
    "prepared_snapshot_id",
    "evaluation_package_id",
    "source_input_lock",
    "ranking",
    "method",
    "retrieval_implementation",
    "source_rankings",
    "tokenizer",
    "top_500_sha256_by_topic",
    "redistribution",
}
_CONTRACT_KEYS = {
    "artifact_type",
    "schema_version",
    "contract_id",
    "system_id",
    "taim_git_commit",
    "working_tree_dirty",
    "prepared_snapshot_id",
    "evaluation_package_id",
    "frozen_retrieval_lock_sha256",
    "frozen_retrieval_artifact_sha256",
    "provider_configuration",
    "generation",
    "method",
}


class TrialGPTPublicationError(ValueError):
    """Raised when publication inputs do not match their frozen contract."""


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _ranking_rows_sha256(rows: Sequence[Mapping[str, JsonValue]]) -> str:
    return _json_sha256(list(rows))


@dataclass(frozen=True, slots=True)
class FrozenRetrievalRow:
    rank: int
    trial_id: str
    score: float

    def identity_dict(self) -> dict[str, JsonValue]:
        return {"rank": self.rank, "score": self.score, "trial_id": self.trial_id}


@dataclass(frozen=True, slots=True)
class FrozenTrialGPTRetrieval:
    """Fully verified local ranking plus its redistributable package lock."""

    lock: Mapping[str, JsonValue]
    lock_sha256: str
    artifact_path: Path
    rankings: Mapping[str, tuple[FrozenRetrievalRow, ...]]

    @property
    def artifact_sha256(self) -> str:
        return cast(str, cast(Mapping[str, object], self.lock["ranking"])["sha256"])

    @property
    def prepared_snapshot_id(self) -> str:
        return cast(str, self.lock["prepared_snapshot_id"])

    @property
    def source_snapshot_id(self) -> str:
        """Snapshot identity on which the frozen retrieval was produced."""

        return self.prepared_snapshot_id

    @property
    def evaluation_package_id(self) -> str:
        return cast(str, self.lock["evaluation_package_id"])

    @property
    def top_500_sha256_by_topic(self) -> Mapping[str, str]:
        return cast(Mapping[str, str], self.lock["top_500_sha256_by_topic"])

    def selected_rows(self, topic_id: str) -> tuple[FrozenRetrievalRow, ...]:
        try:
            rows = self.rankings[topic_id]
        except KeyError as exc:
            raise TrialGPTPublicationError(
                f"frozen three-Luna retrieval has no topic {topic_id!r}"
            ) from exc
        return rows[:TRIALGPT_PUBLICATION_CANDIDATE_DEPTH]

    def candidates(self, *, topic_id: str, run_id: str) -> tuple[Candidate, ...]:
        return tuple(
            Candidate(
                run_id,
                TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
                topic_id,
                row.trial_id,
                row.rank,
                row.score,
            )
            for row in self.selected_rows(topic_id)
        )

    def provenance(self, topic_ids: Sequence[str]) -> dict[str, JsonValue]:
        selected_hashes = {
            topic_id: self.top_500_sha256_by_topic[topic_id]
            for topic_id in sorted(topic_ids, key=int)
        }
        return {
            "lock_sha256": self.lock_sha256,
            "artifact_sha256": self.artifact_sha256,
            "configuration": cast(Mapping[str, JsonValue], self.lock["ranking"])["configuration"],
            "candidate_depth": TRIALGPT_PUBLICATION_CANDIDATE_DEPTH,
            "stage_name": TRIALGPT_PUBLICATION_RETRIEVAL_STAGE,
            "top_500_sha256_by_topic": cast(dict[str, JsonValue], selected_hashes),
            "method": self.lock["method"],
            "retrieval_implementation": self.lock["retrieval_implementation"],
            "source_rankings": self.lock["source_rankings"],
        }


def load_frozen_retrieval_lock(path: Path) -> tuple[dict[str, JsonValue], str]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrialGPTPublicationError("frozen retrieval lock is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise TrialGPTPublicationError("frozen retrieval lock must be a JSON object")
    try:
        require_exact_keys(payload, _LOCK_KEYS, role="frozen retrieval lock")
    except SchemaValidationError as exc:
        raise TrialGPTPublicationError(str(exc)) from exc
    if (
        payload["artifact_type"] != "taim-trialgpt-frozen-retrieval-lock"
        or payload["schema_version"] != "1.0"
        or payload["system_id"] != TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID
        or payload["benchmark_lineage"] not in {"sigir-ct-2016", "trec-ct-2021", "trec-ct-2022"}
    ):
        raise TrialGPTPublicationError("unsupported frozen retrieval lock identity")
    for name in ("prepared_snapshot_id", "evaluation_package_id"):
        try:
            require_sha256(payload[name], f"frozen retrieval lock {name}")
        except SchemaValidationError as exc:
            raise TrialGPTPublicationError(str(exc)) from exc
    tokenizer = payload["tokenizer"]
    if not isinstance(tokenizer, Mapping) or tokenizer.get("id") != TRIALGPT_TOKENIZER_ID:
        raise TrialGPTPublicationError("frozen retrieval tokenizer does not match TAIM code")
    return (
        cast(dict[str, JsonValue], json_value_to_builtins(payload)),
        "sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def _ranking_contract(lock: Mapping[str, JsonValue]) -> Mapping[str, object]:
    ranking = lock.get("ranking")
    if not isinstance(ranking, Mapping):
        raise TrialGPTPublicationError("frozen retrieval lock ranking must be an object")
    expected = {
        "byte_size",
        "configuration",
        "filename",
        "output_depth",
        "publication_candidate_depth",
        "row_count",
        "sha256",
        "topic_count",
    }
    try:
        require_exact_keys(ranking, expected, role="frozen retrieval ranking")
        require_sha256(ranking["sha256"], "frozen retrieval ranking sha256")
    except SchemaValidationError as exc:
        raise TrialGPTPublicationError(str(exc)) from exc
    topic_count = ranking["topic_count"]
    if (
        ranking["configuration"] != "judgment-blind-consensus-all-luna"
        or ranking["output_depth"] != 2_000
        or ranking["publication_candidate_depth"] != TRIALGPT_PUBLICATION_CANDIDATE_DEPTH
        or isinstance(topic_count, bool)
        or not isinstance(topic_count, int)
        or topic_count <= 0
        or ranking["row_count"] != topic_count * 2_000
    ):
        raise TrialGPTPublicationError("unsupported frozen retrieval ranking contract")
    return ranking


def load_frozen_trialgpt_retrieval(
    artifact_path: Path,
    *,
    lock_path: Path,
) -> FrozenTrialGPTRetrieval:
    """Validate every frozen row before exposing any publication candidate."""

    lock, lock_sha256 = load_frozen_retrieval_lock(lock_path)
    ranking = _ranking_contract(lock)
    top_hashes = lock["top_500_sha256_by_topic"]
    if not isinstance(top_hashes, Mapping) or len(top_hashes) != ranking["topic_count"]:
        raise TrialGPTPublicationError("frozen retrieval lock has incomplete top-500 hashes")
    expected_topics: set[str] = set()
    for topic_id, digest in top_hashes.items():
        if not isinstance(topic_id, str) or not topic_id.isdigit() or int(topic_id) < 1:
            raise TrialGPTPublicationError("frozen retrieval lock has an invalid topic ID")
        try:
            require_sha256(digest, f"frozen retrieval topic {topic_id} top-500 hash")
        except SchemaValidationError as exc:
            raise TrialGPTPublicationError(str(exc)) from exc
        expected_topics.add(topic_id)
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_file():
        raise TrialGPTPublicationError(f"frozen retrieval artifact does not exist: {artifact_path}")
    if artifact_path.stat().st_size != ranking["byte_size"]:
        raise TrialGPTPublicationError("frozen retrieval artifact byte size does not match lock")
    if sha256_file(artifact_path) != ranking["sha256"]:
        raise TrialGPTPublicationError("frozen retrieval artifact SHA-256 does not match lock")

    rows_by_topic: defaultdict[str, list[FrozenRetrievalRow]] = defaultdict(list)
    previous_topic_number = 0
    previous_rank = 0
    previous_score: float | None = None
    previous_trial_id = ""
    row_count = 0
    with artifact_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                raw_row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrialGPTPublicationError(
                    f"frozen retrieval row {line_number} is not valid JSON"
                ) from exc
            if not isinstance(raw_row, Mapping) or set(raw_row) != _ROW_KEYS:
                raise TrialGPTPublicationError(
                    f"frozen retrieval row {line_number} has unexpected fields"
                )
            configuration = raw_row["configuration"]
            topic_id = raw_row["topic_id"]
            trial_id = raw_row["trial_id"]
            rank = raw_row["rank"]
            score = raw_row["score"]
            if configuration != ranking["configuration"]:
                raise TrialGPTPublicationError(
                    f"frozen retrieval row {line_number} changes configuration"
                )
            if not isinstance(topic_id, str) or topic_id not in expected_topics:
                raise TrialGPTPublicationError(f"frozen retrieval row {line_number} has bad topic")
            topic_number = int(topic_id)
            if not isinstance(trial_id, str) or not trial_id:
                raise TrialGPTPublicationError(
                    f"frozen retrieval row {line_number} has bad trial ID"
                )
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TrialGPTPublicationError(f"frozen retrieval row {line_number} has bad rank")
            if isinstance(score, bool) or not isinstance(score, int | float):
                raise TrialGPTPublicationError(f"frozen retrieval row {line_number} has bad score")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise TrialGPTPublicationError(
                    f"frozen retrieval row {line_number} has non-finite score"
                )
            if topic_number < previous_topic_number:
                raise TrialGPTPublicationError("frozen retrieval topics are not numerically sorted")
            if topic_number != previous_topic_number:
                if previous_topic_number and previous_rank != 2_000:
                    raise TrialGPTPublicationError(
                        f"frozen retrieval topic {previous_topic_number} is incomplete"
                    )
                previous_topic_number = topic_number
                previous_rank = 0
                previous_score = None
                previous_trial_id = ""
            if rank != previous_rank + 1:
                raise TrialGPTPublicationError(
                    f"frozen retrieval topic {topic_id} ranks are not contiguous"
                )
            if previous_score is not None and (
                numeric_score > previous_score
                or (numeric_score == previous_score and trial_id < previous_trial_id)
            ):
                raise TrialGPTPublicationError(
                    f"frozen retrieval topic {topic_id} violates its tie rule"
                )
            rows_by_topic[topic_id].append(FrozenRetrievalRow(rank, trial_id, numeric_score))
            previous_rank = rank
            previous_score = numeric_score
            previous_trial_id = trial_id
            row_count += 1

    if previous_rank != 2_000 or row_count != ranking["row_count"]:
        raise TrialGPTPublicationError("frozen retrieval artifact is incomplete")
    if set(rows_by_topic) != expected_topics:
        raise TrialGPTPublicationError(
            "frozen retrieval artifact topic membership does not match its lock"
        )
    for topic_id, rows in rows_by_topic.items():
        trial_ids = [row.trial_id for row in rows]
        if len(trial_ids) != len(set(trial_ids)):
            raise TrialGPTPublicationError(
                f"frozen retrieval topic {topic_id} contains duplicate trials"
            )
        actual = _ranking_rows_sha256(
            [row.identity_dict() for row in rows[:TRIALGPT_PUBLICATION_CANDIDATE_DEPTH]]
        )
        if top_hashes[topic_id] != actual:
            raise TrialGPTPublicationError(
                f"frozen retrieval topic {topic_id} top-500 hash does not match lock"
            )
    return FrozenTrialGPTRetrieval(
        lock=MappingProxyType(lock),
        lock_sha256=lock_sha256,
        artifact_path=artifact_path,
        rankings=MappingProxyType(
            {topic_id: tuple(rows) for topic_id, rows in rows_by_topic.items()}
        ),
    )


def validate_frozen_retrieval_for_snapshot(
    retrieval: FrozenTrialGPTRetrieval,
    *,
    benchmark_lineage: str,
    retrieval_source_snapshot_id: str,
    topic_ids: Sequence[str],
    trial_ids: Sequence[str],
) -> None:
    """Bind the complete frozen ranking to the exact current Snapshot before generation."""

    if benchmark_lineage != retrieval.lock["benchmark_lineage"]:
        raise TrialGPTPublicationError("frozen retrieval benchmark lineage does not match")
    if retrieval_source_snapshot_id != retrieval.source_snapshot_id:
        raise TrialGPTPublicationError("frozen retrieval source Snapshot does not match")
    requested_topics = tuple(topic_ids)
    if not requested_topics or len(requested_topics) != len(set(requested_topics)):
        raise TrialGPTPublicationError("publication topics must be non-empty and unique")
    if set(requested_topics) != set(retrieval.rankings):
        raise TrialGPTPublicationError("publication topics must exactly match the frozen retrieval")
    trial_pool = set(trial_ids)
    if len(trial_pool) != len(trial_ids):
        raise TrialGPTPublicationError("publication trial pool contains duplicate IDs")
    missing = sorted(
        {
            row.trial_id
            for rows in retrieval.rankings.values()
            for row in rows
            if row.trial_id not in trial_pool
        }
    )
    if missing:
        raise TrialGPTPublicationError(
            f"frozen retrieval contains {len(missing)} trials outside the Snapshot"
        )


def validate_frozen_retrieval_for_task_input(
    retrieval: FrozenTrialGPTRetrieval,
    *,
    task_input_id: str,
) -> None:
    """Bind retrieval production to the exact identity-bearing System task input."""

    try:
        require_sha256(task_input_id, "publication Task Input ID")
    except SchemaValidationError as exc:
        raise TrialGPTPublicationError(str(exc)) from exc
    source_input_lock = retrieval.lock.get("source_input_lock")
    if not isinstance(source_input_lock, Mapping):
        raise TrialGPTPublicationError("frozen retrieval source_input_lock must be an object")
    if source_input_lock.get("task_input_id") != task_input_id:
        raise TrialGPTPublicationError("frozen retrieval Task Input does not match")


def make_publication_contract(
    *,
    taim_git_commit: str,
    prepared_snapshot_id: str,
    evaluation_package_id: str,
    frozen_retrieval: FrozenTrialGPTRetrieval,
    provider_configuration: Mapping[str, object],
    method_configuration: Mapping[str, object],
    generation_workers: int,
    topic_ids: Sequence[str],
) -> dict[str, JsonValue]:
    """Create the one contract reused unchanged by all issue-93 executions."""

    if _GIT_COMMIT.fullmatch(taim_git_commit) is None:
        raise TrialGPTPublicationError("publication contract requires an exact Git commit")
    if isinstance(generation_workers, bool) or not 1 <= generation_workers <= 48:
        raise TrialGPTPublicationError("publication generation_workers must be from 1 to 48")
    require_sha256(prepared_snapshot_id, "publication prepared_snapshot_id")
    require_sha256(evaluation_package_id, "publication evaluation_package_id")
    selected_topic_ids = list(topic_ids)
    if not selected_topic_ids or len(selected_topic_ids) != len(set(selected_topic_ids)):
        raise TrialGPTPublicationError("publication contract topics must be non-empty and unique")
    core: dict[str, JsonValue] = {
        "artifact_type": "taim-trialgpt-publication-contract",
        "schema_version": TRIALGPT_PUBLICATION_CONTRACT_VERSION,
        "system_id": TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
        "taim_git_commit": taim_git_commit,
        "working_tree_dirty": False,
        "prepared_snapshot_id": prepared_snapshot_id,
        "evaluation_package_id": evaluation_package_id,
        "frozen_retrieval_lock_sha256": frozen_retrieval.lock_sha256,
        "frozen_retrieval_artifact_sha256": frozen_retrieval.artifact_sha256,
        "provider_configuration": cast(
            dict[str, JsonValue], json_value_to_builtins(provider_configuration)
        ),
        "generation": {
            "cache_policy": "fresh empty cache per logical replicate; resume only in place",
            "candidate_depth": TRIALGPT_PUBLICATION_CANDIDATE_DEPTH,
            "expected_logical_calls_per_topic": TRIALGPT_PUBLICATION_LOGICAL_CALLS_PER_TOPIC,
            "generation_workers": generation_workers,
            "model": "gpt-5.6-luna",
            "reasoning_effort": None,
            "topic_count": len(selected_topic_ids),
            "topic_ids": cast(list[JsonValue], selected_topic_ids),
            "expected_total_logical_calls": (
                len(selected_topic_ids) * TRIALGPT_PUBLICATION_LOGICAL_CALLS_PER_TOPIC
            ),
        },
        "method": cast(dict[str, JsonValue], json_value_to_builtins(method_configuration)),
    }
    return {**core, "contract_id": content_sha256(core)}


def load_publication_contract(path: Path) -> dict[str, JsonValue]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrialGPTPublicationError("publication contract is not readable JSON") from exc
    if not isinstance(payload, Mapping):
        raise TrialGPTPublicationError("publication contract must be a JSON object")
    try:
        require_exact_keys(payload, _CONTRACT_KEYS, role="TrialGPT publication contract")
    except SchemaValidationError as exc:
        raise TrialGPTPublicationError(str(exc)) from exc
    if (
        payload["artifact_type"] != "taim-trialgpt-publication-contract"
        or payload["schema_version"] != TRIALGPT_PUBLICATION_CONTRACT_VERSION
        or payload["system_id"] != TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID
        or payload["working_tree_dirty"] is not False
    ):
        raise TrialGPTPublicationError("unsupported TrialGPT publication contract identity")
    contract_id = payload["contract_id"]
    core = {key: value for key, value in payload.items() if key != "contract_id"}
    if contract_id != content_sha256(core):
        raise TrialGPTPublicationError("publication contract ID does not match content")
    if (
        not isinstance(payload["taim_git_commit"], str)
        or _GIT_COMMIT.fullmatch(cast(str, payload["taim_git_commit"])) is None
    ):
        raise TrialGPTPublicationError("publication contract Git commit is invalid")
    return cast(dict[str, JsonValue], json_value_to_builtins(payload))


def validate_publication_preflight(
    contract: Mapping[str, JsonValue],
    *,
    taim_git_commit: str,
    working_tree_dirty: bool | None,
    prepared_snapshot_id: str,
    evaluation_package_id: str,
    frozen_retrieval: FrozenTrialGPTRetrieval,
    provider_configuration: Mapping[str, object],
    method_configuration: Mapping[str, object],
    generation_workers: int,
    topic_ids: Sequence[str],
) -> None:
    """Reject cross-run drift before the first downstream provider call."""

    expected = make_publication_contract(
        taim_git_commit=taim_git_commit,
        prepared_snapshot_id=prepared_snapshot_id,
        evaluation_package_id=evaluation_package_id,
        frozen_retrieval=frozen_retrieval,
        provider_configuration=provider_configuration,
        method_configuration=method_configuration,
        generation_workers=generation_workers,
        topic_ids=topic_ids,
    )
    if working_tree_dirty is not False:
        raise TrialGPTPublicationError("publication execution requires a clean TAIM worktree")
    if json_value_to_builtins(contract) != expected:
        raise TrialGPTPublicationError(
            "publication execution does not exactly match its frozen cross-run contract"
        )


def validate_compatible_publication_contracts(
    left: Mapping[str, JsonValue], right: Mapping[str, JsonValue]
) -> None:
    """Require identical methods and runtimes while allowing different Task Input scopes."""

    shared_fields = (
        "system_id",
        "taim_git_commit",
        "working_tree_dirty",
        "frozen_retrieval_lock_sha256",
        "frozen_retrieval_artifact_sha256",
        "provider_configuration",
        "method",
    )
    if any(left[field] != right[field] for field in shared_fields):
        raise TrialGPTPublicationError("publication contracts differ in frozen method or runtime")
    left_generation = left.get("generation")
    right_generation = right.get("generation")
    if not isinstance(left_generation, Mapping) or not isinstance(right_generation, Mapping):
        raise TrialGPTPublicationError("publication contract generation method is missing")
    shared_generation_fields = (
        "cache_policy",
        "candidate_depth",
        "expected_logical_calls_per_topic",
        "generation_workers",
        "model",
        "reasoning_effort",
    )
    if any(left_generation[field] != right_generation[field] for field in shared_generation_fields):
        raise TrialGPTPublicationError(
            "publication contracts differ in generation method or runtime"
        )


def retrieval_scorecard(
    retrieval: FrozenTrialGPTRetrieval,
    judgments: Sequence[RelevanceJudgment],
    *,
    prepared_snapshot_id: str,
    evaluation_package_id: str,
) -> dict[str, JsonValue]:
    """Compute the provider-free issue-93 retrieval and candidate-cap scorecard."""

    judgments_by_topic: defaultdict[str, list[RelevanceJudgment]] = defaultdict(list)
    for judgment in judgments:
        judgments_by_topic[judgment.topic_id].append(judgment)
    if set(judgments_by_topic) != set(retrieval.rankings):
        raise TrialGPTPublicationError(
            "retrieval scorecard requires judgments for the exact frozen 75 topics"
        )
    cutoffs: dict[str, JsonValue] = {}
    strict_retained: dict[int, int] = {}
    for cutoff in (500, 1_000, 2_000):
        strict_numerator = 0
        strict_denominator = 0
        graded_numerator = 0
        graded_denominator = 0
        strict_topic_values: list[float] = []
        graded_topic_values: list[float] = []
        for topic_id, topic_judgments in judgments_by_topic.items():
            selected = {row.trial_id for row in retrieval.rankings[topic_id][:cutoff]}
            topic_strict_denominator = sum(item.label == 2 for item in topic_judgments)
            topic_strict_numerator = sum(
                item.label == 2 and item.trial_id in selected for item in topic_judgments
            )
            topic_graded_denominator = sum(item.label for item in topic_judgments)
            topic_graded_numerator = sum(
                item.label for item in topic_judgments if item.trial_id in selected
            )
            if not topic_strict_denominator or not topic_graded_denominator:
                raise TrialGPTPublicationError(
                    f"retrieval scorecard topic {topic_id} has an empty recall denominator"
                )
            strict_numerator += topic_strict_numerator
            strict_denominator += topic_strict_denominator
            graded_numerator += topic_graded_numerator
            graded_denominator += topic_graded_denominator
            strict_topic_values.append(topic_strict_numerator / topic_strict_denominator)
            graded_topic_values.append(topic_graded_numerator / topic_graded_denominator)
        strict_retained[cutoff] = strict_numerator
        cutoffs[str(cutoff)] = {
            "strict_eligible": {
                "micro": strict_numerator / strict_denominator,
                "macro": sum(strict_topic_values) / len(strict_topic_values),
                "retained_gain": strict_numerator,
                "total_gain": strict_denominator,
            },
            "graded_gain": {
                "micro": graded_numerator / graded_denominator,
                "macro": sum(graded_topic_values) / len(graded_topic_values),
                "retained_gain": graded_numerator,
                "total_gain": graded_denominator,
            },
        }
    expected_strict_retained = {500: 4_596, 1_000: 5_114, 2_000: 5_320}
    if strict_retained != expected_strict_retained:
        raise TrialGPTPublicationError(
            "frozen retrieval no longer reproduces its issue-22 strict eligible fidelity counts"
        )
    total_strict = cast(
        int,
        cast(Mapping[str, object], cast(Mapping[str, object], cutoffs["2000"])["strict_eligible"])[
            "total_gain"
        ],
    )
    core: dict[str, JsonValue] = {
        "artifact_type": "taim-trialgpt-retrieval-scorecard",
        "schema_version": "1.0",
        "system_id": TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
        "prepared_snapshot_id": prepared_snapshot_id,
        "evaluation_package_id": evaluation_package_id,
        "frozen_retrieval_lock_sha256": retrieval.lock_sha256,
        "frozen_retrieval_artifact_sha256": retrieval.artifact_sha256,
        "topic_count": 75,
        "policies": {
            "strict_eligible": "label 2 only",
            "graded_gain": "TREC label used directly as gain: 0, 1, or 2",
            "unjudged": "zero numerator contribution and rank position retained",
            "aggregation": "micro and unweighted topic macro",
        },
        "cutoffs": cutoffs,
        "strict_eligible_loss_accounting": {
            "total_judged_eligible": total_strict,
            "lost_before_or_below_retrieval_depth_2000": total_strict - strict_retained[2_000],
            "lost_at_2000_to_500_candidate_cap": (strict_retained[2_000] - strict_retained[500]),
            "retained_for_trialgpt_reasoning": strict_retained[500],
        },
        "fidelity": {
            "reference": "issue-22-gpt4-free-luna-ranking-v1",
            "expected_strict_retained": {
                str(key): value for key, value in expected_strict_retained.items()
            },
            "status": "exact",
        },
    }
    return {**core, "scorecard_id": _json_sha256(core)}


__all__ = [
    "TRIALGPT_PUBLICATION_CANDIDATE_DEPTH",
    "TRIALGPT_PUBLICATION_CONTRACT_VERSION",
    "TRIALGPT_PUBLICATION_LOGICAL_CALLS_PER_TOPIC",
    "TRIALGPT_PUBLICATION_RETRIEVAL_STAGE",
    "TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID",
    "FrozenRetrievalRow",
    "FrozenTrialGPTRetrieval",
    "TrialGPTPublicationError",
    "load_frozen_retrieval_lock",
    "load_frozen_trialgpt_retrieval",
    "load_publication_contract",
    "make_publication_contract",
    "retrieval_scorecard",
    "validate_compatible_publication_contracts",
    "validate_frozen_retrieval_for_snapshot",
    "validate_frozen_retrieval_for_task_input",
    "validate_publication_preflight",
]
