"""The qualification gates of ``TrialGPT2-TAIM-Luna-v1``, as protocol v14 Section 6 sets them.

The gates run once, on TREC 2021's external-fidelity target, over that target's one frozen
retrieval: a one-topic validation, then three ten-topic repeats, on the topics and under the rule
of ``TrialGPT-TAIM-Luna-v1``'s gates (``taim.pipelines.trialgpt.qualification``). A gate run is a
qualification run, not a row: it writes a qualification record, never a Run Manifest, and no
effectiveness value is computed from it.

The owner's pass rule, verbatim: "Qualification pass: option (a) - every call accounted for and
the three repeats share every identity. Rank stability reported, no numeric bar." So a gate run
passes when every candidate is scored and every bridge request is accounted for, and the repeats
pass when, in addition, they record the same identities. Rank stability is measured and reported,
and no value of it passes or fails anything. Nothing here reads a judgment, an Evaluation Package
or a score against judgments.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from taim.artifacts import sha256_file
from taim.pipelines.trialgpt.qualification import (
    QUALIFICATION_TRACK,
    STABILITY_SELECTION_VERSION,
    gate_defects,
    qualification_topics,
    require_qualification_retrieval,
    topic_key,
    validate_repeats,
)
from taim.pipelines.trialgpt.trialgpt2_bridge import (
    accounting_defects,
    ledger_accounting,
    read_ledger,
)
from taim.pipelines.trialgpt.trialgpt2_convert import (
    CASES_FILENAME,
    TrialGPT2ConversionError,
    read_upstream_ranking,
    upstream_output_file,
)
from taim.pipelines.trialgpt.trialgpt2_declaration import TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID
from taim.pipelines.trialgpt.trialgpt2_execution import (
    TRIALGPT2_SCORER_STAGE,
    TrialGPT2ExecutionError,
    TrialGPT2TAIMLunaV1System,
    resolve_trialgpt2,
)

if TYPE_CHECKING:
    from taim.patient_to_trial import PatientToTrialRunRequest

QUALIFICATION_RECORD_FILENAME = "trialgpt2-qualification.json"
_RECORD_TYPE = "taim-trialgpt2-qualification-run"
_SHARED_IDENTITIES = (
    "system_id",
    "system_input_id",
    "task_input_id",
    "prepared_snapshot_id",
    "frozen_retrieval_lock_sha256",
    "frozen_retrieval_artifact_sha256",
    "release_identity",
    "upstream",
    "upstream_runtime",
    "provider_configuration",
)


def qualify_trialgpt2(request: PatientToTrialRunRequest, *, gate: str) -> dict[str, object]:
    """Run one gate over the external-fidelity target's frozen retrieval; its record."""

    if request.snapshot.benchmark_lineage != QUALIFICATION_TRACK:
        raise TrialGPT2ExecutionError("TrialGPT 2.0 qualifies on TREC 2021 only")
    topics = qualification_topics([topic.topic_id for topic in request.snapshot.topics], gate)
    resolved, effective = resolve_trialgpt2(request, qualification_topic_ids=topics)
    require_qualification_retrieval(resolved.retrieval.lock)
    result = TrialGPT2TAIMLunaV1System(resolved=resolved).run(effective)
    configuration = result.configuration
    scorer = next(stage for stage in result.stage_rankings if stage.name == TRIALGPT2_SCORER_STAGE)
    order: dict[str, list[str]] = {topic_id: [] for topic_id in topics}
    for candidate in scorer.candidates:
        order[candidate.topic_id].append(candidate.trial_id)
    record: dict[str, object] = {
        "artifact_type": _RECORD_TYPE,
        "schema_version": "1.0",
        "gate": gate,
        "selection": STABILITY_SELECTION_VERSION,
        "run_id": effective.run_id,
        "topic_ids": list(topics),
        "identities": {
            "system_id": TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID,
            "system_input_id": effective.system_input_id,
            "task_input_id": effective.snapshot.task_input_id,
            "prepared_snapshot_id": effective.snapshot.prepared_snapshot_id,
            "frozen_retrieval_lock_sha256": resolved.retrieval.lock_sha256,
            "frozen_retrieval_artifact_sha256": resolved.retrieval.artifact_sha256,
            "release_identity": dict(resolved.release_identity),
            "upstream": resolved.upstream.identity(),
            "upstream_runtime": dict(resolved.upstream_runtime),
            "provider_configuration": dict(resolved.provider_configuration),
        },
        "ranking_order": order,
        "run_workspace": configuration["run_workspace"],
        "bridge_ledger": configuration["bridge_ledger"],
        "upstream_output": configuration["upstream_output"],
        "bridge_accounting": configuration["bridge_accounting"],
        "effectiveness": "not computed; no judgment was read",
    }
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
        raise TrialGPT2ExecutionError(f"{path} is not a TrialGPT 2.0 qualification record")
    return record


def validate_qualification_run(path: Path) -> dict[str, object]:
    """Re-account for one gate run from the files it left; passed only if every call is."""

    record = _load_record(path)
    workspace = Path(str(record["run_workspace"]))
    topics = [str(topic) for topic in cast(list[object], record["topic_ids"])]
    gate = str(record["gate"])
    defects = gate_defects(gate, topics)
    cases = {
        str(case["patient_id"]): [str(trial) for trial in case["candidate_trial_ids"]]
        for case in (
            json.loads(line)
            for line in (workspace / "upstream-input" / CASES_FILENAME)
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    if sorted(cases, key=topic_key) != topics:
        defects.append("the run's cases are not the gate's topics")
    ledger = cast(Mapping[str, object], record["bridge_ledger"])
    ledger_path = Path(str(ledger["path"]))
    if not ledger_path.is_file() or sha256_file(ledger_path) != ledger["sha256"]:
        defects.append("the bridge ledger is missing or changed since the run")
        accounting: dict[str, object] = {}
    else:
        pairs = [(topic, trial) for topic, trials in cases.items() for trial in trials]
        accounting = ledger_accounting(read_ledger(ledger_path), pairs)
        defects.extend(accounting_defects(accounting))
        if accounting != record["bridge_accounting"]:
            defects.append("the bridge accounting does not reproduce from the ledger")
    output = cast(Mapping[str, object], record["upstream_output"])
    output_directory = Path(str(output["path"]))
    order = cast(Mapping[str, list[str]], record["ranking_order"])
    for topic in topics:
        try:
            ranking = read_upstream_ranking(output_directory, topic, cases.get(topic, []))
        except TrialGPT2ConversionError as exc:
            defects.append(str(exc))
            continue
        if [row.trial_id for row in ranking] != order.get(topic):
            defects.append(f"topic {topic}'s ranking does not match the record")
        files = cast(Mapping[str, str], output["files"])
        if sha256_file(upstream_output_file(output_directory, topic)) != files.get(topic):
            defects.append(f"topic {topic}'s output changed since the run")
    return {
        "artifact_type": "taim-trialgpt2-qualification-run-validation",
        "schema_version": "1.0",
        "status": "passed" if not defects else "failed",
        "defects": defects,
        "gate": gate,
        "run_id": record["run_id"],
        "topic_ids": topics,
        "identities": record["identities"],
        "bridge_accounting": accounting,
        "rule": "every call accounted for: each candidate scored, every bridge request attributed "
        "to a candidate pair with a recorded outcome, nothing refused, nothing dropped",
    }


def validate_qualification_repeats(paths: Sequence[Path]) -> dict[str, object]:
    """TrialGPT 2.0's three repeats, under TrialGPT 1.0's repeat rule (``validate_repeats``)."""

    return validate_repeats(
        paths,
        system="TrialGPT 2.0",
        artifact_type="taim-trialgpt2-qualification-repeats-validation",
        load_record=_load_record,
        validate_run=validate_qualification_run,
        shared_identities=_SHARED_IDENTITIES,
    )


__all__ = [
    "QUALIFICATION_RECORD_FILENAME",
    "qualify_trialgpt2",
    "validate_qualification_repeats",
    "validate_qualification_run",
]
