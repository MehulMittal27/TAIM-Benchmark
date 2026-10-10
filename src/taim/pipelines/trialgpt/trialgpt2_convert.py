"""TAIM inputs in TrialGPT 2.0's file formats, and its ranking back as TAIM rows.

Their loader reads one case per patient from ``benchmark_cases.jsonl`` and one JSON file per
candidate trial from a ``trials/`` directory (``matcher_with_usage_logging.py``,
``read_jsonl`` and ``load_trial_by_id``). The converter writes only the fields their prompt uses:

- a case: ``patient_id`` (the topic ID), ``patient_summary`` (the topic's canonical text) and
  ``candidate_trial_ids`` (the topic's frozen top 500, in retrieval order);
- a trial: ``nctid``, ``brief_title`` (the first title section), ``brief_summary`` (the first
  summary section, or empty) and ``raw_criteria`` (the eligibility source text, which
  ``TrialGPT-TAIM-Luna-v1`` reads too).

Their ranking for a patient is ``cover_page_list`` in ``full_email_<patient_id>.json``, sorted by
``final_score`` descending and then trial ID. It is read back unchanged, and refused unless it
scores exactly the candidates the case named.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from taim.model_codex import sha256_text
from taim.pipelines.trialgpt.trialgpt2_bridge import join_messages
from taim.snapshot import BenchmarkTopic, Trial

TRIALGPT2_INPUT_CONVERTER_VERSION = "trialgpt2-taim-input-v1"
CASES_FILENAME = "benchmark_cases.jsonl"
TRIALS_DIRECTORY = "trials"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

PromptBuilder = Callable[..., Sequence[Mapping[str, str]]]


class TrialGPT2ConversionError(ValueError):
    """Raised when TAIM inputs or TrialGPT 2.0 outputs cannot be converted exactly."""


def _section_texts(trial: Trial, role: str) -> list[str]:
    return [section.text.strip() for section in trial.sections if section.role == role]


def upstream_trial(trial: Trial) -> dict[str, str]:
    """One trial in their trial-file format."""

    titles = _section_texts(trial, "brief_title")
    if not titles:
        raise TrialGPT2ConversionError(f"trial {trial.trial_id} has no brief title")
    summaries = _section_texts(trial, "summary")
    criteria = "\n\n".join(
        (section.source_text or section.provenance[0].raw_value).strip()
        for section in trial.sections
        if section.role == "eligibility"
    )
    return {
        "nctid": trial.trial_id,
        "brief_title": titles[0],
        "brief_summary": summaries[0] if summaries else "",
        "raw_criteria": criteria.strip(),
    }


def upstream_case(topic: BenchmarkTopic, candidate_trial_ids: Sequence[str]) -> dict[str, object]:
    """One patient in their case format."""

    return {
        "patient_id": topic.topic_id,
        "patient_summary": topic.canonical_text.strip(),
        "candidate_trial_ids": list(candidate_trial_ids),
    }


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_upstream_inputs(
    directory: Path,
    cases: Sequence[Mapping[str, object]],
    trials: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    """Write their input files and return their digests; rewriting them is idempotent."""

    for name in (*(str(case["patient_id"]) for case in cases), *trials):
        if _SAFE_ID.fullmatch(name) is None:
            raise TrialGPT2ConversionError(f"identifier {name!r} is not a safe file name")
    trial_directory = directory / TRIALS_DIRECTORY
    trial_directory.mkdir(parents=True, exist_ok=True)
    cases_text = "".join(_json_text(case) + "\n" for case in cases)
    (directory / CASES_FILENAME).write_text(cases_text, encoding="utf-8", newline="\n")
    trial_digests: dict[str, str] = {}
    for trial_id in sorted(trials):
        text = _json_text(trials[trial_id]) + "\n"
        (trial_directory / f"{trial_id}.json").write_text(text, encoding="utf-8", newline="\n")
        trial_digests[trial_id] = sha256_text(text)
    return {
        "converter_version": TRIALGPT2_INPUT_CONVERTER_VERSION,
        "cases_sha256": sha256_text(cases_text),
        "trial_count": len(trials),
        "trials_sha256": sha256_text(_json_text(trial_digests)),
    }


def expected_prompts(
    build_messages: PromptBuilder,
    cases: Sequence[Mapping[str, object]],
    trials: Mapping[str, Mapping[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Every pair's prompt digest, from their own prompt builder, as the bridge will see it.

    Their scorer passes the trial's title, brief summary and raw criteria and the case's patient
    summary, each stripped (``score_trial`` and ``patient_note_for_case``). Two pairs whose
    texts are identical share one prompt.
    """

    prompts: dict[str, list[tuple[str, str]]] = {}
    for case in cases:
        topic_id = str(case["patient_id"])
        note = str(case["patient_summary"]).strip()
        for trial_id in cast(Sequence[str], case["candidate_trial_ids"]):
            trial = trials[trial_id]
            messages = list(
                build_messages(
                    title=trial["brief_title"].strip(),
                    brief_summary=trial["brief_summary"].strip(),
                    raw_criteria=trial["raw_criteria"].strip(),
                    patient_note=note,
                )
            )
            if [message.get("role") for message in messages] != ["system", "user"]:
                raise TrialGPT2ConversionError(
                    "their prompt builder no longer returns two messages"
                )
            prompt = join_messages(messages[0]["content"], messages[1]["content"])
            prompts.setdefault(sha256_text(prompt), []).append((topic_id, trial_id))
    return prompts


def upstream_output_file(directory: Path, topic_id: str) -> Path:
    """Their per-patient output file (``patient_output_path``)."""

    return directory / f"full_email_{topic_id}.json"


@dataclass(frozen=True, slots=True)
class ScoredTrial:
    """One trial of their ranking: its ID and their ``final_score``."""

    trial_id: str
    final_score: float


def read_upstream_ranking(
    directory: Path, topic_id: str, candidate_trial_ids: Sequence[str]
) -> tuple[ScoredTrial, ...]:
    """Their complete ranking of one topic, refused unless it scores exactly its candidates."""

    path = upstream_output_file(directory, topic_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrialGPT2ConversionError(f"TrialGPT 2.0 output {path} is not readable JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("patient_ID") != topic_id:
        raise TrialGPT2ConversionError(f"TrialGPT 2.0 output {path} is not topic {topic_id}")
    metadata = payload.get("metadata")
    if (
        "failures" in payload
        or not isinstance(metadata, Mapping)
        or (metadata.get("num_failed_trials") != 0)
    ):
        raise TrialGPT2ConversionError(f"TrialGPT 2.0 output {path} records failed trials")
    rows = payload.get("cover_page_list")
    if not isinstance(rows, list):
        raise TrialGPT2ConversionError(f"TrialGPT 2.0 output {path} has no ranking")
    ranking: list[ScoredTrial] = []
    for row in rows:
        trial_id = row.get("trial_id") if isinstance(row, Mapping) else None
        score = row.get("final_score") if isinstance(row, Mapping) else None
        if (
            not isinstance(trial_id, str)
            or isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            raise TrialGPT2ConversionError(f"TrialGPT 2.0 output {path} has a malformed row")
        ranking.append(ScoredTrial(trial_id, float(score)))
    trial_ids = [row.trial_id for row in ranking]
    if len(trial_ids) != len(set(trial_ids)) or set(trial_ids) != set(candidate_trial_ids):
        raise TrialGPT2ConversionError(
            f"TrialGPT 2.0 output {path} does not score exactly the topic's candidates"
        )
    if ranking != sorted(ranking, key=lambda row: (-row.final_score, row.trial_id)):
        raise TrialGPT2ConversionError(f"TrialGPT 2.0 output {path} is not in their order")
    return tuple(ranking)


__all__ = [
    "CASES_FILENAME",
    "TRIALGPT2_INPUT_CONVERTER_VERSION",
    "TRIALS_DIRECTORY",
    "ScoredTrial",
    "TrialGPT2ConversionError",
    "expected_prompts",
    "read_upstream_ranking",
    "upstream_case",
    "upstream_output_file",
    "upstream_trial",
    "write_upstream_inputs",
]
