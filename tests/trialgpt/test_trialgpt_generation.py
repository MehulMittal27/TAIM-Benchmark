"""Exercise the supported generation seam without retrieval, providers or filesystem state."""

# ruff: noqa: S101

from __future__ import annotations

import json
from collections import Counter
from threading import Lock

import pytest

from taim.generation import ProviderResponse
from taim.pipelines.trialgpt.paper import TrialGPTTrialView
from taim.pipelines.trialgpt.system import TrialGPTCandidateTask, TrialGPTGeneration
from taim.snapshot import BenchmarkTopic, SourceRecordIdentity
from tests.model_test_support import ScriptedModel


class _RecordingModel(ScriptedModel):
    provider_id = "synthetic-generation-seam"

    def __init__(self, *, exhaust: str | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.exhaust = exhaust
        self.lock = Lock()

    def respond(self, request, **kwargs):
        trial_id = request.input_payload["trial_id"]
        with self.lock:
            self.calls.append((trial_id, request.stage))
        if trial_id == self.exhaust and request.stage == "criterion_matching_inclusion":
            return ProviderResponse("invalid structured output")
        if request.stage == "final_aggregation":
            output = {
                "relevance_explanation": "synthetic",
                "relevance_score_R": 80,
                "eligibility_explanation": "synthetic",
                "eligibility_score_E": 40,
            }
        else:
            label = "included" if request.stage.endswith("inclusion") else "excluded"
            output = {"criteria": {"0": ["synthetic assessment", [0], label]}}
        return ProviderResponse(json.dumps(output))


def _tasks():
    return [
        TrialGPTCandidateTask(
            trial_id,
            score,
            TrialGPTTrialView(
                trial_id,
                "Synthetic trial",
                ("alpha syndrome",),
                ("supportive care",),
                "Synthetic summary.",
                "Must have alpha syndrome.",
                "Must not have beta syndrome.",
                "Synthetic retrieval text.",
            ),
        )
        for trial_id, score in (("NCT00000002", 0.2), ("NCT00000001", 0.1))
    ]


def _topic():
    return BenchmarkTopic("topic", SourceRecordIdentity("synthetic", "topic"), "Alpha syndrome.")


@pytest.mark.parametrize("workers", [1, 4])
def test_candidate_generation_preserves_input_order_and_scheduling_contract(workers):
    model = _RecordingModel()
    outcomes = TrialGPTGeneration(model).candidates(
        _tasks(), topic=_topic(), max_workers=workers, allow_abstention=True
    )
    assert [outcome.matching.task.trial_id for outcome in outcomes] == [
        "NCT00000002",
        "NCT00000001",
    ]
    assert all(outcome.aggregation is not None for outcome in outcomes)
    assert all(not outcome.matching.unresolved_stages for outcome in outcomes)
    assert all(outcome.matching.matching["inclusion"]["0"][2] == "included" for outcome in outcomes)
    if workers == 1:
        assert model.calls == [
            (trial_id, stage)
            for trial_id in ("NCT00000002", "NCT00000001")
            for stage in (
                "criterion_matching_inclusion",
                "criterion_matching_exclusion",
                "final_aggregation",
            )
        ]
    else:
        assert all(stage != "final_aggregation" for _, stage in model.calls[:4])
        assert all(stage == "final_aggregation" for _, stage in model.calls[4:])


@pytest.mark.parametrize("workers", [1, 4])
def test_structured_exhaustion_skips_only_that_candidates_aggregation(workers):
    model = _RecordingModel(exhaust="NCT00000002")
    outcomes = TrialGPTGeneration(model).candidates(
        _tasks(), topic=_topic(), max_workers=workers, allow_abstention=True
    )
    failed, resolved = outcomes
    assert failed.matching.unresolved_stages == ("criterion_matching_inclusion",)
    assert failed.matching.matching["inclusion"] is None
    assert failed.aggregation is None
    assert resolved.aggregation is not None
    assert Counter(model.calls)[("NCT00000002", "criterion_matching_inclusion")] == 3
    assert ("NCT00000002", "final_aggregation") not in model.calls


def _codex_run(monkeypatch, model_factory):
    """Run one candidate through a real Codex Model whose subprocess answers from a script."""

    import subprocess
    from pathlib import Path

    from taim.pipelines.trialgpt.system import PAPER_AGGREGATION_SCHEMA

    executions: list[dict[str, object]] = []
    answers = {
        # The first inclusion answer fails validation, so the repair prompt is exercised too.
        "Must have alpha syndrome.": iter(
            [
                '{"criteria": {"0": ["synthetic", [0], "maybe"]}}',
                '{"criteria": {"0": ["synthetic", [0], "included"]}}',
            ]
        ),
        "Must not have beta syndrome.": iter(['{"0": ["synthetic", [0], "not excluded"]}']),
    }
    aggregation = json.dumps(
        {
            name: 80 if name.endswith("_R") else 40 if name.endswith("_E") else "synthetic"
            for name in PAPER_AGGREGATION_SCHEMA["properties"]
        }  # type: ignore[union-attr]
    )

    def runner(command, **kwargs):
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, "codex-cli 0.146.0\n", "")
        if command[1] == "debug":
            catalog = {"models": [{"slug": "gpt-5.6-luna"}]}
            return subprocess.CompletedProcess(command, 0, json.dumps(catalog), "")
        prompt = kwargs["input"]
        executions.append(
            {"command": command, "prompt": prompt, "files": sorted(Path(kwargs["cwd"]).iterdir())}
        )
        text = aggregation
        for criterion, script in answers.items():
            if criterion in prompt and "eligibility_score_E" not in prompt:
                text = next(script)
        event = {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
        return subprocess.CompletedProcess(command, 0, json.dumps(event), "")

    monkeypatch.setattr(subprocess, "run", runner)
    outcomes = TrialGPTGeneration(model_factory()).candidates(
        _tasks()[:1], topic=_topic(), max_workers=1, allow_abstention=False
    )
    return outcomes, executions


def test_trialgpt_keeps_its_schema_local_and_sends_the_same_prompts(monkeypatch):
    """The dated 2026-10-07 transport change: no schema reaches Codex, nothing else moves."""

    from taim.generation import canonical_json
    from taim.models import Model
    from taim.pipelines.trialgpt.system import (
        PAPER_MATCHING_SCHEMA,
        TRIALGPT_CODEX_MODEL,
        trialgpt_codex_model,
        trialgpt_publication_method_contract,
    )

    unsent, unsent_calls = _codex_run(monkeypatch, trialgpt_codex_model)
    sent, sent_calls = _codex_run(
        monkeypatch,
        lambda: Model(
            model_id=TRIALGPT_CODEX_MODEL, backend="codex", parameters={"reasoning_effort": None}
        ),
    )

    assert len(unsent_calls) == len(sent_calls) == 4
    assert all("--output-schema" not in call["command"] for call in unsent_calls)
    assert all(call["files"] == [] for call in unsent_calls)
    assert all("--output-schema" in call["command"] for call in sent_calls)
    # Original and repair prompts are byte-identical, and the repair still embeds the schema.
    assert [call["prompt"] for call in unsent_calls] == [call["prompt"] for call in sent_calls]
    repair = unsent_calls[1]["prompt"]
    assert "label is invalid for inclusion" in repair
    assert repair.endswith("Required schema:\n" + canonical_json(PAPER_MATCHING_SCHEMA))
    # The same local parse, normalizer and validator accept the same answers.
    assert [outcome.matching.matching for outcome in unsent] == [
        outcome.matching.matching for outcome in sent
    ]
    assert [outcome.aggregation.output for outcome in unsent] == [  # type: ignore[union-attr]
        outcome.aggregation.output
        for outcome in sent  # type: ignore[union-attr]
    ]
    unsent_result = unsent[0].matching.results["inclusion"]
    sent_result = sent[0].matching.results["inclusion"]
    codex = unsent_result.attempts[0].metadata["codex"]
    assert codex["output_schema_transport"] == "none"
    assert (
        codex["output_schema_sha256"]
        == sent_result.attempts[0].metadata["codex"]["output_schema_sha256"]
    )
    # The transport enters request identity, so no answer is reused across transports.
    assert unsent_result.identity_sha256 != sent_result.identity_sha256
    # The method contract, including the matching schema digest, does not move.
    method = trialgpt_publication_method_contract()
    assert method["matching_schema_sha256"] == (
        "sha256:84e266a114f75fc1cca5ac66db34c182081a4b9b51530e2f3a8b5ae63482a9a4"
    )
    assert method["aggregation_schema_sha256"] == (
        "sha256:b2c50aecf2aaf72e1354c0597153cf94e1dcae76a5c4b889db82b35bb083dac7"
    )
    assert "output_schema_transport" not in method


def test_a_two_part_criterion_answer_resolves_without_repair():
    """A missing sentence list counts as no sentences cited: one attempt, no abstention."""

    class _TwoPart(_RecordingModel):
        def respond(self, request, **kwargs):
            if request.stage == "criterion_matching_exclusion":
                with self.lock:
                    self.calls.append((request.input_payload["trial_id"], request.stage))
                return ProviderResponse(json.dumps({"0": ["synthetic", "not enough information"]}))
            return super().respond(request, **kwargs)

    (outcome,) = TrialGPTGeneration(_TwoPart()).candidates(
        _tasks()[:1], topic=_topic(), max_workers=1, allow_abstention=True
    )
    exclusion = outcome.matching.results["exclusion"]
    assert exclusion.resolved and len(exclusion.attempts) == 1
    assert outcome.matching.matching["exclusion"] == {
        "0": ["synthetic", [], "not enough information"]
    }
    assert outcome.aggregation is not None
