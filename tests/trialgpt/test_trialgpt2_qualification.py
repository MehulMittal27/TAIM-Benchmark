"""TrialGPT 2.0's gates: coverage and shared identities pass them; stability is only reported.

The gate topics and the repeat rule are TrialGPT 1.0's (``test_trialgpt_qualification``). One
validation and three repeats run once for the module; each repeat is one topic instead of ten,
which keeps the synthetic gates small under the same rule.
"""

# ruff: noqa: S101
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import taim.pipelines.trialgpt.qualification as qualification
from taim.pipelines.trialgpt.qualification import TrialGPTQualificationError
from taim.pipelines.trialgpt.trialgpt2_qualification import (
    QUALIFICATION_RECORD_FILENAME,
    qualify_trialgpt2,
    validate_qualification_repeats,
    validate_qualification_run,
)
from tests.trialgpt.trialgpt2_support import (
    install_codex_stand_in,
    run_request,
    synthetic_task_input,
    write_stand_in_checkout,
)

_SMALL_GATES = {"validation": 1, "repeat": 1}


@pytest.fixture(scope="module")
def synthetic_input():
    return synthetic_task_input(topic_count=3)


def _gate(tmp_path, synthetic_input, stand_in, gate: str, run_id: str, **options) -> Path:
    root = tmp_path / run_id
    root.mkdir()
    request = run_request(root, synthetic_input, stand_in, run_id=run_id, **options)
    record = qualify_trialgpt2(request, gate=gate)
    return Path(str(record["run_workspace"])) / QUALIFICATION_RECORD_FILENAME


@pytest.fixture(scope="module")
def trialgpt2_gate_runs(synthetic_input, tmp_path_factory):
    """The validation and three repeats, run once; the stand-ins stay installed for the module."""

    root = tmp_path_factory.mktemp("trialgpt2-gates")
    with pytest.MonkeyPatch.context() as monkeypatch:
        stand_in = write_stand_in_checkout(root / "upstream")
        install_codex_stand_in(monkeypatch, root, stand_in)
        monkeypatch.setattr(qualification, "QUALIFICATION_GATES", _SMALL_GATES)
        validation = _gate(root, synthetic_input, stand_in, "validation", "gate-validation")
        repeats = [
            _gate(root, synthetic_input, stand_in, "repeat", f"gate-repeat-{number}")
            for number in (1, 2, 3)
        ]
        yield SimpleNamespace(stand_in=stand_in, validation=validation, repeats=repeats)


def test_the_gates_pass_when_every_call_is_accounted_for_and_the_repeats_agree(
    trialgpt2_gate_runs,
) -> None:
    report = validate_qualification_run(trialgpt2_gate_runs.validation)
    assert report["status"] == "passed", report["defects"]
    assert report["topic_ids"] == ["3"]
    assert report["bridge_accounting"]["codex_answers"] == 500

    report = validate_qualification_repeats(trialgpt2_gate_runs.repeats)
    assert report["status"] == "passed", report["defects"]
    assert report["rank_stability"]["bar"] == "none; reported only"
    pairwise = report["rank_stability"]["pairwise"]
    assert len(pairwise) == 3
    # The Codex stand-in answers each prompt the same way, so the repeats agree exactly.
    assert {item["minimum_spearman"] for item in pairwise} == {1.0}
    record = json.loads(trialgpt2_gate_runs.validation.read_text())
    assert record["effectiveness"] == "not computed; no judgment was read"


def test_a_gate_run_whose_files_changed_fails(trialgpt2_gate_runs, tmp_path) -> None:
    # The record names its ledger by digest; a record whose ledger digest no longer matches stands
    # for a ledger changed after the run.
    record = json.loads(trialgpt2_gate_runs.validation.read_text())
    record["bridge_ledger"]["sha256"] = "sha256:" + "0" * 64
    changed = tmp_path / QUALIFICATION_RECORD_FILENAME
    changed.write_text(json.dumps(record))
    report = validate_qualification_run(changed)
    assert report["status"] == "failed"
    assert "the bridge ledger is missing or changed since the run" in report["defects"]


def test_repeats_that_differ_in_an_identity_fail(
    trialgpt2_gate_runs, synthetic_input, tmp_path
) -> None:
    stand_in = trialgpt2_gate_runs.stand_in
    other = _gate(tmp_path, synthetic_input, stand_in, "repeat", "gate-repeat-4", workers=4)
    report = validate_qualification_repeats([*trialgpt2_gate_runs.repeats[:2], other])
    assert report["status"] == "failed"
    assert "the repeats differ in system_input_id" in report["defects"]
    assert report["rank_stability"]["pairwise"] == []


def test_the_gates_run_only_on_the_trec_2021_external_fidelity_retrieval(
    trialgpt2_gate_runs, synthetic_input, tmp_path
) -> None:
    stand_in = trialgpt2_gate_runs.stand_in
    calls = len(stand_in.calls)
    request = run_request(
        tmp_path, synthetic_input, stand_in, profile="trec-ct-2021-judgment-union"
    )
    with pytest.raises(TrialGPTQualificationError, match="external-fidelity"):
        qualify_trialgpt2(request, gate="validation")
    with pytest.raises(TrialGPTQualificationError, match="unknown qualification gate"):
        qualify_trialgpt2(request, gate="rows")
    assert len(stand_in.calls) == calls
