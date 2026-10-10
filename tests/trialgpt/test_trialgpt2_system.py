"""TrialGPT2-TAIM-Luna-v1 end to end: their runner's stand-in, the bridge, a Codex stand-in."""

# ruff: noqa: S101
from __future__ import annotations

import json
from pathlib import Path

import pytest

from taim.pipelines.trialgpt.publication import (
    TrialGPTPublicationError,
    load_frozen_trialgpt_retrieval,
)
from taim.pipelines.trialgpt.trialgpt2_bridge import SYSTEM_PREAMBLE, read_ledger
from taim.pipelines.trialgpt.trialgpt2_declaration import (
    TRIALGPT2_TAIM_LUNA_V1,
    TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID,
)
from taim.pipelines.trialgpt.trialgpt2_execution import (
    TRIALGPT2_SCORER_STAGE,
    TrialGPT2ExecutionError,
    TrialGPT2TAIMLunaV1System,
    run_trialgpt2,
)
from taim.pipelines.trialgpt.trialgpt2_upstream import (
    TrialGPT2UpstreamError,
    verify_upstream_checkout,
)
from tests.git_helpers import git
from tests.trialgpt.trialgpt2_support import (
    RELEASE_IDENTITY,
    install_codex_stand_in,
    run_request,
    synthetic_task_input,
    trial_titles,
    write_stand_in_checkout,
)


@pytest.fixture(scope="module")
def synthetic_input():
    return synthetic_task_input(topic_count=1)


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    stand_in = write_stand_in_checkout(tmp_path / "upstream")
    install_codex_stand_in(monkeypatch, tmp_path, stand_in)
    # A variable of the caller's shell must never reach their client.
    monkeypatch.setenv("TAIM_TEST_LEAK", "1")
    return stand_in


def test_the_declaration_makes_the_system_comparable_only_to_itself() -> None:
    comparability = TRIALGPT2_TAIM_LUNA_V1.comparability
    assert comparability.comparable_to == frozenset({TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID})
    assert comparability.statement.startswith(
        "Through Codex, TrialGPT2-TAIM-Luna-v1 is comparable only to itself"
    )
    assert [item.name for item in TRIALGPT2_TAIM_LUNA_V1.bridge_distortions] == [
        "joined-messages",
        "sampling-not-applied",
        "harness-and-backbone",
    ]


def test_a_run_scores_every_frozen_candidate_through_the_bridge(
    tmp_path, synthetic_input, stand_in
) -> None:
    request = run_request(tmp_path, synthetic_input, stand_in)
    result = run_trialgpt2(TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID, request)

    retrieval = load_frozen_trialgpt_retrieval(
        Path(request.options["frozen_retrieval_path"]),
        lock_path=Path(request.options["frozen_retrieval_lock_path"]),
    )
    handed_over = [row.trial_id for row in retrieval.selected_rows("1")]
    stages = {stage.name: stage for stage in result.stage_rankings}
    assert [row.trial_id for row in stages["three-luna-consensus"].candidates] == handed_over
    scored = stages[TRIALGPT2_SCORER_STAGE].candidates
    assert sorted(row.trial_id for row in scored) == sorted(handed_over)
    assert [(row.score, row.trial_id) for row in scored] == sorted(
        ((row.score, row.trial_id) for row in scored), key=lambda item: (-item[0], item[1])
    )
    assert list(result.candidates) == list(scored[:20])
    assert {row.system_id for row in result.candidates} == {TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID}

    # One Codex execution per candidate, each the joined system and user message.
    assert len(stand_in.calls) == 500
    assert all(call.startswith(SYSTEM_PREAMBLE + "\n") for call in stand_in.calls)
    configuration = result.configuration
    accounting = configuration["bridge_accounting"]
    assert accounting["requests"] == accounting["codex_answers"] == 500
    assert accounting["cache_hits"] == accounting["refused_requests"] == 0
    assert accounting["unanswered_pairs"] == 0
    assert accounting["codex_versions"] == ["codex-cli 0.0.0-taim-trialgpt2-test"]
    ledger = read_ledger(Path(configuration["bridge_ledger"]["path"]))
    assert {record["sampling_not_applied"]["temperature"] for record in ledger} == {0.0}
    assert {record["outcome"] for record in ledger} == {"codex"}

    options = configuration["system_input"]["options"]
    assert options["frozen_retrieval_lock_sha256"] == retrieval.lock_sha256
    assert options["upstream"]["commit"] == stand_in.commit
    assert options["provider_configuration"]["model"] == "gpt-5.6-luna"
    assert configuration["executing_release_identity"] == RELEASE_IDENTITY
    assert configuration["model_identity"]["provider_version"] == (
        "codex-cli 0.0.0-taim-trialgpt2-test"
    )
    assert configuration["index_identity"]["kind"] == "trialgpt_frozen_retrieval"
    assert result.pipeline_depth == "post_eligibility"
    # The consumed lock keeps its producer's System id.
    assert retrieval.lock["system_id"] == "TrialGPT-TAIM-Luna-v1"


def test_an_interrupted_run_resumes_in_place_from_the_bridge_cache(
    tmp_path, synthetic_input, stand_in
) -> None:
    request = run_request(tmp_path, synthetic_input, stand_in)
    retrieval = load_frozen_trialgpt_retrieval(
        Path(request.options["frozen_retrieval_path"]),
        lock_path=Path(request.options["frozen_retrieval_lock_path"]),
    )
    failing = retrieval.selected_rows("1")[7].trial_id
    stand_in.failing_trials = trial_titles({failing})
    with pytest.raises(TrialGPT2ExecutionError, match="resume in place"):
        run_trialgpt2(TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID, request)
    first_calls = len(stand_in.calls)
    # Their client retried the failing pair; every other pair was answered once.
    assert first_calls == 499 + 4

    stand_in.failing_trials = set()
    result = run_trialgpt2(TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID, request)
    assert len(stand_in.calls) == first_calls + 1
    accounting = result.configuration["bridge_accounting"]
    assert accounting["cache_hits"] == 499
    assert accounting["codex_failures"] == {"transport": 4}
    assert accounting["invocations"] == [1, 2]
    assert accounting["unanswered_pairs"] == 0


def test_an_output_cut_short_by_an_interruption_is_set_aside_and_redone(
    tmp_path, synthetic_input, stand_in
) -> None:
    request = run_request(tmp_path, synthetic_input, stand_in)
    run_trialgpt2(TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID, request)
    workspace = next((tmp_path / "workspace" / "runs").iterdir())
    output = workspace / "upstream-output" / "full_email_1.json"
    output.write_text(output.read_text()[:100])

    result = run_trialgpt2(TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID, request)
    (set_aside,) = result.configuration["interrupted_outputs_set_aside"]
    assert set_aside["topic_id"] == "1"
    assert Path(set_aside["path"]).name == "full_email_1.json.incomplete-2"
    assert result.configuration["bridge_accounting"]["cache_hits"] == 500


def test_an_answer_outside_their_schema_is_never_cached(
    tmp_path, synthetic_input, stand_in
) -> None:
    request = run_request(tmp_path, synthetic_input, stand_in)
    retrieval = load_frozen_trialgpt_retrieval(
        Path(request.options["frozen_retrieval_path"]),
        lock_path=Path(request.options["frozen_retrieval_lock_path"]),
    )
    malformed = retrieval.selected_rows("1")[3].trial_id
    stand_in.malformed_trials = trial_titles({malformed})
    with pytest.raises(TrialGPT2ExecutionError):
        run_trialgpt2(TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID, request)
    stand_in.malformed_trials = set()
    result = run_trialgpt2(TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID, request)
    accounting = result.configuration["bridge_accounting"]
    # A JSON object their client parses is not retried, so the pair failed at once; it was not
    # cached, so the resume asked again.
    assert accounting["nonconforming_answers"] == 1
    assert accounting["cache_hits"] == 499
    assert accounting["codex_answers"] == 501


@pytest.mark.parametrize(
    "release_identity",
    [None, {**RELEASE_IDENTITY, "package_artifact_id": "sha256:" + "6" * 64}],
)
def test_a_lock_frozen_on_another_release_is_refused_before_any_call(
    tmp_path, synthetic_input, stand_in, release_identity
) -> None:
    request = run_request(tmp_path, synthetic_input, stand_in, release_identity=release_identity)
    with pytest.raises(TrialGPTPublicationError, match="release"):
        run_trialgpt2(TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID, request)
    assert stand_in.calls == []


def test_a_workspace_of_another_run_is_refused(tmp_path, synthetic_input, stand_in) -> None:
    run_trialgpt2(
        TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID, run_request(tmp_path, synthetic_input, stand_in)
    )
    calls = len(stand_in.calls)
    changed = run_request(tmp_path, synthetic_input, stand_in, workers=4)
    with pytest.raises(TrialGPT2ExecutionError, match="in-place resume"):
        run_trialgpt2(TRIALGPT2_TAIM_LUNA_V1_SYSTEM_ID, changed)
    assert len(stand_in.calls) == calls


def test_the_registered_system_refuses_to_run_without_verified_inputs(synthetic_input) -> None:
    from taim.patient_to_trial import PatientToTrialRunRequest

    request = PatientToTrialRunRequest(
        snapshot=synthetic_input, run_id="unverified", top_k=20, metric_cutoff=10, options={}
    )
    with pytest.raises(TrialGPT2ExecutionError, match="verified"):
        TrialGPT2TAIMLunaV1System().run(request)


def test_the_checkout_must_be_the_clean_pin_with_its_declared_files(tmp_path) -> None:
    stand_in = write_stand_in_checkout(tmp_path / "upstream")
    pins = {
        "commit": stand_in.commit,
        "tree": stand_in.tree,
        "executed_files": stand_in.executed_files,
    }
    verified = verify_upstream_checkout(stand_in.checkout, **pins)
    assert verified.identity()["executed_files"] == dict(sorted(stand_in.executed_files.items()))

    with pytest.raises(TrialGPT2UpstreamError, match="not the pinned"):
        verify_upstream_checkout(stand_in.checkout, **{**pins, "commit": "0" * 40})
    with pytest.raises(TrialGPT2UpstreamError, match="missing or changed"):
        verify_upstream_checkout(
            stand_in.checkout,
            **{**pins, "executed_files": {"src/__init__.py": "sha256:" + "0" * 64}},
        )
    (stand_in.checkout / "notes.txt").write_text("x")
    with pytest.raises(TrialGPT2UpstreamError, match="local changes"):
        verify_upstream_checkout(stand_in.checkout, **pins)
    (stand_in.checkout / "notes.txt").unlink()
    git(stand_in.checkout, "status")


def test_the_runner_environment_carries_nothing_of_the_callers(monkeypatch) -> None:
    from taim.pipelines.trialgpt.trialgpt2_upstream import scorer_environment

    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("OPENAI_ORG_ID", "someone")
    environment = scorer_environment("http://127.0.0.1:9/v1")
    assert "LLM_MAX_RETRIES" not in environment and "OPENAI_ORG_ID" not in environment
    assert environment["OPENAI_BASE_URL"] == "http://127.0.0.1:9/v1"
    assert environment["LLM_PROVIDER"] == "openai"
    assert int(environment["LLM_TIMEOUT_S"]) > 600
    assert json.loads(json.dumps(environment)) == environment
