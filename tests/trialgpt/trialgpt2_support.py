"""Synthetic inputs for the TrialGPT 2.0 build's tests: Task Inputs and stand-ins.

Nothing here is TrialGPT 2.0's code. The stand-in checkout holds a synthetic prompt module with
their builder's signature and a standard-library runner with their runner's options, files and
failure behaviour: it scores each case's candidates in parallel through the chat-completions
wire, raises after all of a case's calls if one failed, writes ``full_email_<id>.json`` only for a
finished case, and skips a finished case under ``--resume``. The Codex stand-in answers in
process, so a run makes no provider call.

The frozen retrieval and its lock are TrialGPT 1.0's (``trialgpt_support``), re-exported here.
"""

# ruff: noqa: S101
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

import taim.pipelines.trialgpt.trialgpt2_bridge as bridge_module
import taim.pipelines.trialgpt.trialgpt2_execution as execution
from taim.artifacts import sha256_file
from taim.eligibility import derive_complete_eligibility_section
from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.model_codex import CodexProvider, sha256_text
from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialTaskInput
from taim.pipelines.trialgpt.trialgpt2_upstream import verify_upstream_checkout
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
    CAPABILITY_FIELD_PROVENANCE,
    CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
    BenchmarkTopic,
    SourceRecordIdentity,
    Trial,
)
from tests.git_helpers import git
from tests.trialgpt.trialgpt_support import (
    FROZEN_DEPTH,
    PREPARED_SNAPSHOT_ID,
    RELEASE_IDENTITY,
    TARGET_PROFILE,
    TRACK,
    synthetic_section,
    write_frozen_retrieval,
)

CODEX_VERSION = "codex-cli 0.0.0-taim-trialgpt2-test"
#: TrialGPT 2.0's strict response schema, as their client sends it; its digest is pinned.
UPSTREAM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "eligible_reasons": {"type": "string"},
        "missing_information": {"type": "string"},
        "ineligible_reasons": {"type": "string"},
        "rationale": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "eligible_reasons",
        "missing_information",
        "ineligible_reasons",
        "rationale",
        "score",
        "confidence",
    ],
}
STAND_IN_SYSTEM_PROMPT = "Synthetic matching policy for TAIM tests."

_PROMPT_MODULE = """
SYSTEM_PROMPT = "Synthetic matching policy for TAIM tests."


def build_trial_matcher_messages(*, title, brief_summary, raw_criteria, patient_note):
    user = (
        f"Title: {title.strip()}\\n"
        f"Brief summary: {brief_summary.strip()}\\n\\n"
        f"Eligibility Criteria (raw):\\n{(raw_criteria or '').strip()}\\n\\n"
        f"Patient note/summary:\\n{(patient_note or '').strip()}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
"""

_RUNNER_MODULE = """
import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.trial_matching.prompts.trial_matcher import build_trial_matcher_messages

SCHEMA = json.loads(__SCHEMA__)


def chat(model, messages):
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "trial_match_result", "strict": True, "schema": SCHEMA},
            },
            "temperature": 0.0,
        }
    ).encode("utf-8")
    error = None
    for _attempt in range(4):
        request = urllib.request.Request(
            os.environ["OPENAI_BASE_URL"] + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
            },
        )
        try:
            timeout = float(os.environ["LLM_TIMEOUT_S"])
            with urllib.request.urlopen(request, timeout=timeout) as answer:
                payload = json.loads(answer.read())
            return json.loads(payload["choices"][0]["message"]["content"])
        except Exception as exc:
            error = exc
    raise RuntimeError(f"LLM call failed after retries: {error}")


def main():
    parser = argparse.ArgumentParser()
    for name in ("--cases_path", "--trial_dir", "--out_dir", "--note_field", "--model"):
        parser.add_argument(name, required=True)
    parser.add_argument("--nprocs", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.environ.get("LLM_PROVIDER") != "openai" or "TAIM_TEST_LEAK" in os.environ:
        raise SystemExit("unexpected environment")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for line in Path(args.cases_path).read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        patient_id = case["patient_id"]
        out_path = out_dir / f"full_email_{patient_id}.json"
        if args.resume and out_path.exists():
            continue
        note = case["patient_summary"].strip()

        def score(trial_id):
            trial = json.loads((Path(args.trial_dir) / f"{trial_id}.json").read_text("utf-8"))
            parsed = chat(
                args.model,
                build_trial_matcher_messages(
                    title=trial["brief_title"].strip(),
                    brief_summary=trial["brief_summary"].strip(),
                    raw_criteria=trial["raw_criteria"].strip(),
                    patient_note=note,
                ),
            )
            value = int(parsed["score"])
            confidence = round(float(parsed["confidence"]), 3)
            return {
                "trial_id": trial_id,
                "score": value,
                "confidence": confidence,
                "final_score": value + confidence,
            }

        results = []
        error = None
        log = out_dir / "pair_logs" / f"{patient_id}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.nprocs) as pool:
            futures = {
                pool.submit(score, trial_id): trial_id for trial_id in case["candidate_trial_ids"]
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                    status = "ok"
                except Exception as exc:
                    error = exc
                    status = "error"
                line = json.dumps({"trial_id": futures[future], "status": status})
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\\n")
        if error is not None:
            raise error
        results.sort(key=lambda row: (-row["final_score"], row["trial_id"]))
        out_path.write_text(
            json.dumps(
                {
                    "patient_ID": patient_id,
                    "cover_page_list": results,
                    "metadata": {"num_trials": len(results), "num_failed_trials": 0},
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
""".replace("__SCHEMA__", repr(json.dumps(UPSTREAM_SCHEMA)))


@dataclass
class StandIn:
    """The stand-in checkout and Codex, and what the Codex stand-in saw."""

    checkout: Path
    executed_files: dict[str, str]
    commit: str
    tree: str
    calls: list[str] = field(default_factory=list)
    failing_trials: set[str] = field(default_factory=set)
    malformed_trials: set[str] = field(default_factory=set)


def write_stand_in_checkout(root: Path) -> StandIn:
    """A Git checkout with their runner's interface and a synthetic prompt module."""

    root.mkdir(parents=True)
    files = {
        "src/__init__.py": "",
        "src/trial_matching/__init__.py": "",
        "src/trial_matching/prompts/__init__.py": "",
        "src/trial_matching/prompts/trial_matcher.py": _PROMPT_MODULE,
        "src/trial_matching/run_matcher_with_usage_logging.py": _RUNNER_MODULE,
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=TAIM", "-c", "user.email=taim@example.invalid", "commit", "-qm", "x")
    return StandIn(
        checkout=root,
        executed_files={relative: sha256_file(root / relative) for relative in files},
        commit=git(root, "rev-parse", "HEAD"),
        tree=git(root, "rev-parse", "HEAD^{tree}"),
    )


def codex_events(text: str) -> str:
    item = {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
    usage = {
        "type": "turn.completed",
        "usage": {"input_tokens": 1200, "cached_input_tokens": 400, "output_tokens": 90},
    }
    return json.dumps(item) + "\n" + json.dumps(usage) + "\n"


def answer_for(prompt: str) -> str:
    """A deterministic, schema-conforming answer that depends on the prompt."""

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return json.dumps(
        {
            "eligible_reasons": "1. Synthetic.",
            "missing_information": "1. None.",
            "ineligible_reasons": "1. None.",
            "rationale": "1. Synthetic.",
            "score": int(digest[:2], 16) % 101,
            "confidence": (int(digest[2:4], 16) % 101) / 100,
        }
    )


def install_codex_stand_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stand_in: StandIn):
    """Answer every Codex command in process: version, bundled catalog and ``exec``."""

    catalog = json.dumps({"models": [{"slug": "gpt-5.6-luna", "context_window": 1}]})

    def run(command, *args, input=None, **kwargs):
        del args, kwargs
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, CODEX_VERSION + "\n", "")
        if command[1:] == ["debug", "models", "--bundled"]:
            return subprocess.CompletedProcess(command, 0, catalog, "")
        assert command[1] == "exec"
        schema_path = Path(command[command.index("--output-schema") + 1])
        assert json.loads(schema_path.read_text(encoding="utf-8")) == UPSTREAM_SCHEMA
        stand_in.calls.append(input)
        if any(f"Synthetic trial {trial}\n" in input for trial in stand_in.failing_trials):
            return subprocess.CompletedProcess(command, 1, "", "synthetic failure")
        if any(f"Synthetic trial {trial}\n" in input for trial in stand_in.malformed_trials):
            return subprocess.CompletedProcess(command, 0, codex_events('{"score": 7}'), "")
        return subprocess.CompletedProcess(command, 0, codex_events(answer_for(input)), "")

    def provider(configuration):
        transport = configuration["transport"]
        return CodexProvider(
            model=configuration["model"],
            codex_executable=transport["codex_executable"],
            runner=run,
            request_timeout_seconds=transport["request_timeout_seconds"],
        )

    def preflight(configuration):
        return {
            **configuration,
            "transport": provider(configuration).provenance(),
        }

    monkeypatch.setattr(bridge_module, "codex_provider", provider)
    monkeypatch.setattr(execution, "codex_preflight", preflight)
    monkeypatch.setattr(
        execution,
        "verify_upstream_checkout",
        lambda path: verify_upstream_checkout(
            path,
            commit=stand_in.commit,
            tree=stand_in.tree,
            executed_files=stand_in.executed_files,
        ),
    )
    original_loader = execution.load_prompt_builder
    monkeypatch.setattr(
        execution,
        "load_prompt_builder",
        lambda checkout: original_loader(
            checkout, system_prompt_sha256=sha256_text(STAND_IN_SYSTEM_PROMPT)
        ),
    )
    monkeypatch.setattr(execution, "upstream_runtime", lambda: {"openai": "test", "tqdm": "test"})


def synthetic_trial(index: int) -> Trial:
    trial_id = f"NCT{index:08d}"
    title = f"Synthetic trial {index}"
    eligibility = derive_complete_eligibility_section(
        synthetic_section("eligibility", "Adult patient.", 2),
        source_text=f"Inclusion Criteria:\n- Adult patient {index}.\nExclusion Criteria:\n- None.",
    )
    return Trial(
        trial_id,
        SourceRecordIdentity("taim.fixture", trial_id),
        f"{title}\n[ELIGIBILITY] {eligibility.text}",
        (
            synthetic_section("brief_title", title, 0),
            synthetic_section("summary", "A synthetic trial.", 1),
            eligibility,
        ),
    )


def synthetic_task_input(topic_count: int) -> PatientToTrialTaskInput:
    topics = tuple(
        BenchmarkTopic(
            topic_id=str(number),
            source_identity=SourceRecordIdentity("taim.fixture", str(number)),
            canonical_text=f"Patient {number} with a synthetic condition.",
        )
        for number in range(1, topic_count + 1)
    )
    return PatientToTrialTaskInput(
        benchmark_lineage=TRACK,
        topics=topics,
        trials=tuple(synthetic_trial(index) for index in range(1, FROZEN_DEPTH + 1)),
        available_capabilities=frozenset(
            {
                CAPABILITY_CANONICAL_PATIENT_TEXT,
                CAPABILITY_CANONICAL_TRIAL_TEXT,
                CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
                CAPABILITY_FIELD_PROVENANCE,
                CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
            }
        ),
        derived_views=(),
        prepared_snapshot_id=PREPARED_SNAPSHOT_ID,
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )


def run_request(
    tmp_path: Path,
    task_input: PatientToTrialTaskInput,
    stand_in: StandIn,
    *,
    run_id: str = "trialgpt2-test-run",
    release_identity: Mapping[str, str] | None = RELEASE_IDENTITY,
    profile: str = TARGET_PROFILE,
    workers: int = 8,
) -> PatientToTrialRunRequest:
    artifact, lock = write_frozen_retrieval(tmp_path, task_input, profile=profile)
    return PatientToTrialRunRequest(
        snapshot=task_input,
        run_id=run_id,
        top_k=20,
        metric_cutoff=10,
        options={
            "frozen_retrieval_path": artifact,
            "frozen_retrieval_lock_path": lock,
            "workspace": tmp_path / "workspace",
            "upstream_checkout": stand_in.checkout,
            "retrieval_source_snapshot_id": PREPARED_SNAPSHOT_ID,
            "executing_release_identity": (
                dict(release_identity) if release_identity is not None else None
            ),
            "generation_workers": workers,
        },
    )


def trial_titles(trial_ids: Collection[str]) -> set[str]:
    """The stand-in's failure keys: the trial numbers inside synthetic titles."""

    return {str(int(trial_id.removeprefix("NCT"))) for trial_id in trial_ids}


__all__ = [
    "CODEX_VERSION",
    "RELEASE_IDENTITY",
    "STAND_IN_SYSTEM_PROMPT",
    "TARGET_PROFILE",
    "TRACK",
    "UPSTREAM_SCHEMA",
    "StandIn",
    "answer_for",
    "codex_events",
    "install_codex_stand_in",
    "run_request",
    "synthetic_task_input",
    "synthetic_trial",
    "trial_titles",
    "write_frozen_retrieval",
    "write_stand_in_checkout",
]
