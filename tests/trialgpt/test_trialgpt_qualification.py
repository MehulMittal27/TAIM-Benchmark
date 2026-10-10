"""TrialGPT 1.0's gates: every call accounted for and shared identities pass them.

Rank stability is only reported. The gate topics and the rule are the ones TrialGPT 2.0's gates
use too. One validation and three repeats run once for the module, with one candidate abstained
throughout; each repeat is one topic instead of ten, which keeps the synthetic gates small under
the same rule.
"""

# ruff: noqa: S101
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import taim.pipelines.trialgpt.qualification as qualification
from taim.pipelines.trialgpt.execution import run_publication
from taim.pipelines.trialgpt.qualification import (
    QUALIFICATION_RECORD_FILENAME,
    TrialGPTQualificationError,
    qualification_topics,
    qualify_trialgpt,
    validate_qualification_repeats,
    validate_qualification_run,
)
from taim.pipelines.trialgpt.system import TrialGPTTAIMLunaV1System
from tests.trialgpt.trialgpt_support import (
    install_trialgpt_codex,
    publication_request,
    trialgpt_task_input,
)

_SMALL_GATES = {"validation": 1, "repeat": 1}


@pytest.fixture(scope="module")
def gate_input():
    return trialgpt_task_input(topic_count=3)


def _gate(root: Path, gate_input, gate: str, run_id: str, **options) -> Path:
    request = publication_request(root / run_id, gate_input, run_id=run_id, **options)
    record = qualify_trialgpt(request, gate=gate)
    return Path(str(record["run_workspace"])) / QUALIFICATION_RECORD_FILENAME


def _first_candidate(topic_id: str, gate_input) -> str:
    """The frozen retrieval's rank-1 trial for a topic (``write_frozen_retrieval``'s order)."""

    return min(
        (trial.trial_id for trial in gate_input.trials),
        key=lambda trial_id: hashlib.sha256(f"{topic_id}/{trial_id}".encode()).digest(),
    )


@pytest.fixture(scope="module")
def gate_runs(gate_input, tmp_path_factory):
    """The validation and three repeats, run once; the stand-in stays installed for the module."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        codex = install_trialgpt_codex(monkeypatch)
        monkeypatch.setattr(qualification, "QUALIFICATION_GATES", _SMALL_GATES)
        abstained = _first_candidate("3", gate_input)
        codex.malformed_trials.add(str(int(abstained.removeprefix("NCT"))))
        root = tmp_path_factory.mktemp("gates")
        validation = _gate(root, gate_input, "validation", "gate-validation")
        repeats = [
            _gate(root, gate_input, "repeat", f"gate-repeat-{number}") for number in (1, 2, 3)
        ]
        yield SimpleNamespace(
            validation=validation,
            repeats=repeats,
            abstained=abstained,
        )


@pytest.fixture
def trialgpt_codex(monkeypatch):
    trialgpt_codex = install_trialgpt_codex(monkeypatch)
    monkeypatch.setattr(qualification, "QUALIFICATION_GATES", _SMALL_GATES)
    return trialgpt_codex


def test_the_gate_topics_are_the_stability_selection() -> None:
    topics = [str(number) for number in range(1, 76)]
    assert qualification_topics(topics, "validation") == ("9",)
    assert qualification_topics(topics, "repeat") == (
        "9",
        "28",
        "35",
        "36",
        "42",
        "56",
        "61",
        "69",
        "70",
        "74",
    )


def test_trialgpt_gates_pass_when_every_call_is_traced_and_the_repeats_agree(gate_runs) -> None:
    report = validate_qualification_run(gate_runs.validation)
    assert report["status"] == "passed", report["defects"]
    assert report["topic_ids"] == ["3"]
    record = json.loads(gate_runs.validation.read_text())
    assert record["effectiveness"] == "not computed; no judgment was read"
    assert {topic: len(order) for topic, order in record["ranking_order"].items()} == {"3": 500}

    report = validate_qualification_repeats(gate_runs.repeats)
    assert report["status"] == "passed", report["defects"]
    assert report["rank_stability"]["bar"] == "none; reported only"
    pairwise = report["rank_stability"]["pairwise"]
    assert len(pairwise) == 3
    # The Codex stand-in answers each prompt the same way, so the repeats agree exactly.
    assert {item["minimum_spearman"] for item in pairwise} == {1.0}
    assert report["shared_identities"]["system_id"] == "TrialGPT-TAIM-Luna-v1"


def test_an_abstained_candidate_is_accounted_for(gate_runs) -> None:
    report = validate_qualification_run(gate_runs.validation)
    assert report["status"] == "passed", report["defects"]
    # Its two matching calls are traced, one unresolved; its aggregation is never made.
    assert report["generation_coverage"] == {
        "expected_logical_call_count": 1_500,
        "generation_call_count": 1_499,
        "resolved_logical_call_count": 1_498,
        "abstained_trial_count": 1,
    }
    record = json.loads(gate_runs.validation.read_text())
    assert gate_runs.abstained in record["ranking_order"]["3"]


def test_a_trialgpt_gate_run_whose_traces_changed_fails(gate_runs, tmp_path) -> None:
    # The record names its run files by digest; a record whose trace digest no longer matches
    # stands for a trace file changed after the run.
    record = json.loads(gate_runs.validation.read_text())
    record["generation_trace"]["sha256"] = "sha256:" + "0" * 64
    changed = tmp_path / QUALIFICATION_RECORD_FILENAME
    changed.write_text(json.dumps(record))
    report = validate_qualification_run(changed)
    assert report["status"] == "failed"
    assert "the run's generation trace is missing or changed since the run" in report["defects"]


def test_trialgpt_repeats_that_differ_in_an_identity_fail(gate_runs, gate_input, tmp_path) -> None:
    other = _gate(tmp_path, gate_input, "repeat", "gate-repeat-4", workers=4)
    report = validate_qualification_repeats([*gate_runs.repeats[:2], other])
    assert report["status"] == "failed"
    assert "the repeats differ in system_input_id" in report["defects"]
    assert "the repeats differ in publication_contract_id" in report["defects"]
    assert report["rank_stability"]["pairwise"] == []


class _Bound(Exception):
    """Stops a run at the System, once its System Input is bound."""


def test_a_row_carries_no_gate_topics_and_a_gate_run_names_its_own(
    tmp_path, gate_input, trialgpt_codex, monkeypatch
) -> None:
    bound = []

    def stop(self, request):
        bound.append(request)
        raise _Bound

    monkeypatch.setattr(TrialGPTTAIMLunaV1System, "run", stop)
    for run_id, topics in (("row", None), ("gate", ["3"])):
        request = publication_request(tmp_path / run_id, gate_input, run_id=run_id)
        with pytest.raises(_Bound):
            run_publication("TrialGPT-TAIM-Luna-v1", request, qualification_topic_ids=topics)
    row, gate = bound
    assert "qualification_topic_ids" not in row.options
    assert list(gate.options["qualification_topic_ids"]) == ["3"]
    assert {
        name: value for name, value in gate.options.items() if name != "qualification_topic_ids"
    } == dict(row.options)
    assert gate.system_input_id != row.system_input_id
    with pytest.raises(ValueError, match="distinct Task Input topics"):
        run_publication(
            "TrialGPT-TAIM-Luna-v1",
            publication_request(tmp_path / "unknown", gate_input, run_id="unknown"),
            qualification_topic_ids=["4"],
        )
    assert trialgpt_codex.calls == []


def test_trialgpt_gates_run_only_on_the_external_fidelity_retrieval(
    tmp_path, gate_input, trialgpt_codex
) -> None:
    request = publication_request(
        tmp_path, gate_input, run_id="gate", profile="trec-ct-2021-judgment-union"
    )
    with pytest.raises(TrialGPTQualificationError, match="external-fidelity"):
        qualify_trialgpt(request, gate="validation")
    with pytest.raises(TrialGPTQualificationError, match="unknown qualification gate"):
        qualify_trialgpt(request, gate="rows")
    assert trialgpt_codex.calls == []
