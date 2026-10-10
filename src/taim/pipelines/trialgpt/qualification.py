"""The qualification gates of ``TrialGPT-TAIM-Luna-v1``, as protocol v14 Section 6 sets them.

The gates run once, on TREC 2021's external-fidelity target, over that target's one frozen
retrieval: a one-topic validation, then three ten-topic repeats, on the qrel-independent
``trialgpt-stability-v1`` topic selection. A System that re-ranks the same retrieval qualifies on
the same topics under the same rule, from this module. A gate run is a qualification run, not a
row: it writes a qualification record, never a Run Manifest, and no effectiveness value is
computed from it.

The owner's pass rule, verbatim: "Qualification pass: option (a) - every call accounted for and
the three repeats share every identity. Rank stability reported, no numeric bar." For this System
a candidate's three logical calls (inclusion matching, exclusion matching, aggregation) are
accounted for when each is traced, or when its candidate is recorded as an abstention under the
System's failure policy, which skips the aggregation call. The repeats pass when, in addition,
they record the same identities. Rank stability is measured and reported, and no value of it
passes or fails anything. Nothing here reads a judgment, an Evaluation Package or a score against
judgments.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, cast

from taim.artifacts import sha256_file
from taim.contracts import content_sha256
from taim.schemas import json_value_to_builtins

if TYPE_CHECKING:
    from taim.patient_to_trial import PatientToTrialRunRequest

STABILITY_SELECTION_VERSION = "trialgpt-stability-v1"
QUALIFICATION_TRACK = "trec-ct-2021"
QUALIFICATION_PROFILE = "trec-ct-2021-external-fidelity-26149"
#: The two gates and their topic counts: the one-topic validation, and each ten-topic repeat.
QUALIFICATION_GATES: Mapping[str, int] = {"validation": 1, "repeat": 10}
QUALIFICATION_RECORD_FILENAME = "trialgpt-qualification.json"
#: Logical calls per candidate: inclusion matching, exclusion matching and aggregation.
_CALLS_PER_CANDIDATE = 3
_LLM_RANKING_STAGE = "trialgpt-llm-ranking"
_RECORD_TYPE = "taim-trialgpt-qualification-run"
_SHARED_IDENTITIES = (
    "system_id",
    "system_input_id",
    "task_input_id",
    "prepared_snapshot_id",
    "frozen_retrieval_lock_sha256",
    "frozen_retrieval_artifact_sha256",
    "release_identity",
    "publication_contract_id",
    "provider_configuration",
    "method_contract_sha256",
)


class TrialGPTQualificationError(ValueError):
    """Raised when a qualification gate cannot run or its records cannot be read."""


def topic_key(topic_id: str) -> tuple[int, int | str]:
    return (0, int(topic_id)) if topic_id.isdigit() else (1, topic_id)


def gate_size(gate: str) -> int:
    """How many topics a gate runs."""

    count = QUALIFICATION_GATES.get(gate)
    if count is None:
        raise TrialGPTQualificationError(f"unknown qualification gate {gate!r}")
    return count


def gate_defects(gate: str, topic_ids: Sequence[str]) -> list[str]:
    """A defect when a gate run's topic count is not its gate's."""

    if QUALIFICATION_GATES.get(gate) != len(topic_ids):
        return [f"gate {gate!r} has {len(topic_ids)} topics"]
    return []


def qualification_topics(topic_ids: Sequence[str], gate: str) -> tuple[str, ...]:
    """The gate's topics: the first of the qrel-independent ``trialgpt-stability-v1`` order."""

    count = gate_size(gate)
    ranked = sorted(
        topic_ids,
        key=lambda topic_id: (
            hashlib.sha256(f"{STABILITY_SELECTION_VERSION}\0{topic_id}".encode()).hexdigest(),
            topic_id,
        ),
    )
    return tuple(sorted(ranked[:count], key=topic_key))


def require_qualification_retrieval(lock: Mapping[str, object]) -> None:
    """Refuse a frozen retrieval that is not the external-fidelity target's."""

    source_lock = lock.get("source_input_lock")
    receipt = source_lock.get("query_plan_receipt") if isinstance(source_lock, Mapping) else None
    if not isinstance(receipt, Mapping) or receipt.get("profile") != QUALIFICATION_PROFILE:
        raise TrialGPTQualificationError(
            f"TrialGPT qualifies on the {QUALIFICATION_PROFILE} retrieval only"
        )


def qualify_trialgpt(request: PatientToTrialRunRequest, *, gate: str) -> dict[str, object]:
    """Run one gate over the external-fidelity target's frozen retrieval; its record."""

    from taim.pipelines.trialgpt.execution import run_publication
    from taim.pipelines.trialgpt.publication import TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID

    if request.snapshot.benchmark_lineage != QUALIFICATION_TRACK:
        raise TrialGPTQualificationError("TrialGPT qualifies on TREC 2021 only")
    topics = qualification_topics([topic.topic_id for topic in request.snapshot.topics], gate)
    lock_path = request.options.get("frozen_retrieval_lock_path")
    if not isinstance(lock_path, Path):
        raise TrialGPTQualificationError(
            "TrialGPT qualification requires frozen_retrieval_lock_path"
        )
    require_qualification_retrieval(json.loads(lock_path.read_text(encoding="utf-8")))
    result = run_publication(
        TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID, request, qualification_topic_ids=topics
    )
    configuration = result.configuration
    system_input = cast(Mapping[str, object], configuration["system_input"])
    options = cast(Mapping[str, object], system_input["options"])
    index_identity = cast(Mapping[str, object], configuration["index_identity"])
    llm_ranking = next(stage for stage in result.stage_rankings if stage.name == _LLM_RANKING_STAGE)
    order: dict[str, list[str]] = {topic_id: [] for topic_id in topics}
    for candidate in llm_ranking.candidates:
        order[candidate.topic_id].append(candidate.trial_id)
    record = {
        "artifact_type": _RECORD_TYPE,
        "schema_version": "1.0",
        "gate": gate,
        "selection": STABILITY_SELECTION_VERSION,
        "run_id": request.run_id,
        "topic_ids": list(topics),
        "identities": {
            "system_id": TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
            "system_input_id": configuration["system_input_id"],
            "task_input_id": configuration["task_input_id"],
            "prepared_snapshot_id": configuration["prepared_snapshot_id"],
            "frozen_retrieval_lock_sha256": index_identity["lock_sha256"],
            "frozen_retrieval_artifact_sha256": index_identity["artifact_sha256"],
            "release_identity": request.options["executing_release_identity"],
            "publication_contract_id": options["publication_contract_id"],
            "provider_configuration": configuration["provider_configuration"],
            "method_contract_sha256": content_sha256(configuration["publication_method"]),
        },
        "ranking_order": order,
        "run_workspace": configuration["run_workspace"],
        "generation_trace": configuration["generation_trace"],
        "criterion_matching": configuration["criterion_matching"],
        "abstentions": configuration["abstentions"],
        "generation_coverage": configuration["generation_coverage"],
        "effectiveness": "not computed; no judgment was read",
    }
    record = cast(dict[str, object], json_value_to_builtins(record))
    path = Path(str(configuration["run_workspace"])) / QUALIFICATION_RECORD_FILENAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _load_record(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or record.get("artifact_type") != _RECORD_TYPE
        or record.get("schema_version") != "1.0"
    ):
        raise TrialGPTQualificationError(f"{path} is not a TrialGPT qualification record")
    return record


def _recorded_rows(record: Mapping[str, object], name: str, defects: list[str]) -> list[dict]:
    """One run file the record names, if it is unchanged since the run; else a defect."""

    entry = cast(Mapping[str, object], record[name])
    path = Path(str(entry["path"]))
    if not path.is_file() or sha256_file(path) != entry["sha256"]:
        defects.append(f"the run's {name.replace('_', ' ')} is missing or changed since the run")
        return []
    return cast(list[dict], json.loads(path.read_text(encoding="utf-8")))


def validate_qualification_run(path: Path) -> dict[str, object]:
    """Re-account for one gate run from the files it left; passed only if every call is."""

    record = _load_record(path)
    topics = [str(topic) for topic in cast(list[object], record["topic_ids"])]
    gate = str(record["gate"])
    defects = gate_defects(gate, topics)
    traces = _recorded_rows(record, "generation_trace", defects)
    matching = _recorded_rows(record, "criterion_matching", defects)
    abstentions = _recorded_rows(record, "abstentions", defects)
    candidates: dict[str, list[str]] = {}
    for row in matching:
        candidates.setdefault(str(row["topic_id"]), []).append(str(row["trial_id"]))
    if matching and sorted(candidates, key=topic_key) != topics:
        defects.append("the run's candidates are not the gate's topics")
    abstained = {(str(row["topic_id"]), str(row["trial_id"])) for row in abstentions}
    calls = Counter((str(row["topic_id"]), str(row.get("trial_id"))) for row in traces)
    unresolved = {
        (str(row["topic_id"]), str(row.get("trial_id")))
        for row in traces
        if row["status"] != "resolved"
    }
    order = cast(Mapping[str, list[str]], record["ranking_order"])
    for topic in topics:
        trials = candidates.get(topic, [])
        if len(set(trials)) != len(trials) or sorted(trials) != sorted(order.get(topic, [])):
            defects.append(f"topic {topic}'s ranking is not its candidates, each once")
        for trial in trials:
            pair = (topic, trial)
            expected = _CALLS_PER_CANDIDATE - (pair in abstained)
            if calls[pair] != expected:
                defects.append(
                    f"topic {topic} trial {trial}: {calls[pair]} calls traced, {expected} expected"
                )
    unaccounted = sorted(unresolved - abstained)
    if unaccounted:
        defects.append(f"{len(unaccounted)} unresolved calls without a recorded abstention")
    stray = sorted(
        set(calls) - {(topic, trial) for topic in topics for trial in candidates.get(topic, [])}
    )
    if stray:
        defects.append(f"{len(stray)} traced calls outside the gate's candidates")
    coverage = {
        "expected_logical_call_count": _CALLS_PER_CANDIDATE * len(matching),
        "generation_call_count": len(traces),
        "resolved_logical_call_count": sum(row["status"] == "resolved" for row in traces),
        "abstained_trial_count": len(abstained),
    }
    recorded = cast(Mapping[str, object], record["generation_coverage"])
    if any(recorded.get(name) != value for name, value in coverage.items()):
        defects.append("the generation coverage does not reproduce from the run's files")
    return {
        "artifact_type": "taim-trialgpt-qualification-run-validation",
        "schema_version": "1.0",
        "status": "passed" if not defects else "failed",
        "defects": defects,
        "gate": gate,
        "run_id": record["run_id"],
        "topic_ids": topics,
        "identities": record["identities"],
        "generation_coverage": coverage,
        "rule": "every call accounted for: each candidate's three logical calls traced, or its "
        "aggregation skipped and the candidate recorded as an abstention under the System's "
        "failure policy; no call outside the gate's candidates",
    }


def spearman(left: Sequence[str], right: Sequence[str]) -> float:
    """Spearman's rank correlation of two orders of the same candidates."""

    if len(left) != len(right) or set(left) != set(right) or len(left) < 2:
        raise TrialGPTQualificationError(
            "rank stability needs the same candidates in both rankings"
        )
    position = {trial_id: rank for rank, trial_id in enumerate(right, start=1)}
    squared = sum((rank - position[trial_id]) ** 2 for rank, trial_id in enumerate(left, start=1))
    count = len(left)
    return 1.0 - (6.0 * squared) / (count * (count * count - 1))


def validate_repeats(
    paths: Sequence[Path],
    *,
    system: str,
    artifact_type: str,
    load_record: Callable[[Path], dict[str, object]],
    validate_run: Callable[[Path], dict[str, object]],
    shared_identities: Sequence[str],
) -> dict[str, object]:
    """Three ten-topic repeats: each accounted for, all sharing every identity.

    Rank stability, the topic-level Spearman correlation over each topic's fixed candidates for
    each pair of repeats, is reported and decides nothing.
    """

    if len(paths) != 3:
        raise TrialGPTQualificationError(f"the {system} repeats are exactly three runs")
    reports = [validate_run(path) for path in paths]
    records = [load_record(path) for path in paths]
    defects = [
        f"repeat {report['run_id']}: {defect}"
        for report in reports
        for defect in cast(list[str], report["defects"])
    ]
    if any(record["gate"] != "repeat" for record in records):
        defects.append("every repeat must be a ten-topic repeat run")
    run_ids = [str(record["run_id"]) for record in records]
    if len(set(run_ids)) != 3:
        defects.append("the repeats must be three distinct runs")
    if len({json.dumps(record["topic_ids"]) for record in records}) != 1:
        defects.append("the repeats differ in their topics")
    identities = [cast(Mapping[str, object], record["identities"]) for record in records]
    for name in shared_identities:
        if len({json.dumps(identity.get(name), sort_keys=True) for identity in identities}) != 1:
            defects.append(f"the repeats differ in {name}")
    pairwise: list[dict[str, object]] = []
    if not defects:
        for left, right in combinations(range(3), 2):
            left_order = cast(Mapping[str, list[str]], records[left]["ranking_order"])
            right_order = cast(Mapping[str, list[str]], records[right]["ranking_order"])
            correlations = {
                topic: spearman(left_order[topic], right_order[topic])
                for topic in sorted(left_order, key=topic_key)
            }
            pairwise.append(
                {
                    "runs": [run_ids[left], run_ids[right]],
                    "spearman_by_topic": correlations,
                    "mean_spearman": sum(correlations.values()) / len(correlations),
                    "minimum_spearman": min(correlations.values()),
                }
            )
    return {
        "artifact_type": artifact_type,
        "schema_version": "1.0",
        "status": "passed" if not defects else "failed",
        "defects": defects,
        "run_ids": run_ids,
        "shared_identities": dict(identities[0]) if not defects else None,
        "rank_stability": {
            "method": "topic-level Spearman correlation over the fixed 500-candidate set",
            "bar": "none; reported only",
            "pairwise": pairwise,
        },
        "runs": reports,
    }


def validate_qualification_repeats(paths: Sequence[Path]) -> dict[str, object]:
    """TrialGPT-TAIM-Luna-v1's three repeats (see ``validate_repeats``)."""

    return validate_repeats(
        paths,
        system="TrialGPT",
        artifact_type="taim-trialgpt-qualification-repeats-validation",
        load_record=_load_record,
        validate_run=validate_qualification_run,
        shared_identities=_SHARED_IDENTITIES,
    )


__all__ = [
    "QUALIFICATION_GATES",
    "QUALIFICATION_PROFILE",
    "QUALIFICATION_RECORD_FILENAME",
    "QUALIFICATION_TRACK",
    "STABILITY_SELECTION_VERSION",
    "TrialGPTQualificationError",
    "gate_defects",
    "gate_size",
    "qualification_topics",
    "qualify_trialgpt",
    "require_qualification_retrieval",
    "spearman",
    "topic_key",
    "validate_qualification_repeats",
    "validate_qualification_run",
    "validate_repeats",
]
