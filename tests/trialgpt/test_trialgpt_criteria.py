"""TrialGPT-TAIM-Luna-v1's criteria: upstream's parse_criteria over sections TAIM cuts (v16).

Upstream TrialGPT splits NCBI's pre-separated ``inclusion_criteria`` and ``exclusion_criteria``
fields with ``parse_criteria``. The System cuts the complete eligibility text into those two
sections where split-v3's polarity changes and runs upstream's function, byte for byte, on each.
"""

# ruff: noqa: S101

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from taim.eligibility import ELIGIBILITY_CRITERION_VIEW_CAPABILITY
from taim.pipelines.trialgpt.paper import (
    TrialGPTTrialView,
    criterion_sections,
    matching_prompt,
    parse_criteria,
    split_criteria,
)
from taim.pipelines.trialgpt.system import (
    TRIALGPT_CRITERION_SOURCE,
    TrialGPTTAIMLunaV1System,
    trialgpt_publication_method_contract,
)
from taim.snapshot import CAPABILITY_COMPLETE_ELIGIBILITY_TEXT

# The lock sits in external/ two levels above this file, in the release and in its source. The
# checkout sits under external/ there or one level higher.
_PUBLIC_ROOT = Path(__file__).resolve().parents[2]
_LOCK = _PUBLIC_ROOT / "external" / "trialgpt-lock.json"
_UPSTREAM_FILE = next(
    (
        candidate
        for root in (_PUBLIC_ROOT, _PUBLIC_ROOT.parent)
        if (candidate := root / "external/trialgpt/trialgpt_matching/TrialGPT.py").is_file()
    ),
    None,
)
#: SHA-256 of ``trialgpt_matching/TrialGPT.py`` lines 20-37 at the locked commit ``1b3242cd``
#: (git blob ``fe2ecbc0``): upstream's ``parse_criteria``, tabs and trailing blanks included.
_UPSTREAM_PARSE_CRITERIA_SHA256 = "c4b3b96773f63f4a135718be2b0f2e106f6fe901cb8b73a1453eeec52324523f"


def test_parse_criteria_is_upstreams_function_byte_for_byte() -> None:
    source = inspect.getsource(parse_criteria)
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == _UPSTREAM_PARSE_CRITERIA_SHA256
    assert "\t" in source  # upstream indents with tabs; a reformat would change the bytes


def test_parse_criteria_pin_matches_the_locked_checkout() -> None:
    if _UPSTREAM_FILE is None:
        pytest.skip("external/trialgpt is not checked out")
    lock = json.loads(_LOCK.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "-C", str(_UPSTREAM_FILE.parents[1]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    ).stdout.strip()
    assert head == lock["canonical"]["commit"]
    upstream = "".join(_UPSTREAM_FILE.read_text(encoding="utf-8").splitlines(keepends=True)[19:37])
    assert upstream == inspect.getsource(parse_criteria)


@pytest.mark.parametrize(
    "criteria",
    [
        "",
        "inclusion criteria:\n\n Be aged 6 months \n\n Have diarrhea \n\n ",
        ": \n\n Previously enrolled \n\n Age \n\n EXCLUSION CRITERIA apply",
        "- First.\n  wrapped line\n\n\n\n- Second criterion.\n\n1234\n\n12345",
        # Upstream lowercases with str.lower; casefold would also match the long s.
        "Exclu\u017fion criteria for donors\n\n\u00a0Stra\u00dfe criterion\u00a0",
    ],
)
def test_split_criteria_returns_the_criteria_parse_criteria_numbers(criteria: str) -> None:
    numbered = "".join(f"{index}. {item}\n" for index, item in enumerate(split_criteria(criteria)))
    assert numbered == parse_criteria(criteria)


def test_the_matching_prompt_renders_each_section_with_parse_criteria() -> None:
    inclusion = "Inclusion Criteria:\n\n  - Adult.\n\n  - Signed consent."
    view = TrialGPTTrialView(
        trial_id="NCT00000001",
        brief_title="Title",
        diseases=("Disease",),
        interventions=("Drug",),
        brief_summary="Summary",
        inclusion_criteria=inclusion,
        exclusion_criteria="Exclusion Criteria:\n\n  - Pregnancy.",
        retrieval_text="",
    )
    prompt = matching_prompt(view, "inclusion", "0. Patient.")

    # Upstream's print_trial: "Inclusion criteria:\n %s\n" % parse_criteria(...).
    assert f"Inclusion criteria:\n {parse_criteria(inclusion)}\n" in prompt
    assert "There are exactly 2 inclusion criteria." in prompt


def test_sections_cut_where_split_v3_polarity_changes_and_convert_line_endings() -> None:
    text = (
        "\n        Inclusion Criteria:\r\n\r\n          1. Be aged 18 years or older\r\n\r\n"
        "          2. Have diabetes, with an HbA1c above 7% measured within\r\n"
        "             three months\r\n\r\n        Exclusion Criteria:\r\n\r\n"
        "          1. Pregnancy\r\n      "
    )

    inclusion, exclusion = criterion_sections(text)

    assert "\r" not in inclusion + exclusion
    assert split_criteria(inclusion) == (
        "1. Be aged 18 years or older",
        "2. Have diabetes, with an HbA1c above 7% measured within\n             three months",
    )
    assert split_criteria(exclusion) == ("1. Pregnancy",)
    # The section the change opens starts at its first item, list marker included.
    assert exclusion.startswith("          1. Pregnancy")
    assert "Exclusion Criteria" not in inclusion + exclusion


def test_a_heading_sharing_a_paragraph_with_its_criteria_is_left_out() -> None:
    # NCT00139113's layout. NCBI's exclusion field drops the phrase "exclusion criteria", so
    # upstream keeps these criteria; a section starting at the heading would lose them all.
    text = (
        "Inclusion Criteria: term infant, healthy at age 6 months -\r\n\r\n"
        "        Exclusion Criteria: received immune globulin; received immunosuppressive therapy"
    )

    inclusion, exclusion = criterion_sections(text)

    assert split_criteria(inclusion) == ()  # upstream's inclusion field loses it the same way
    assert split_criteria(exclusion) == (
        "received immune globulin; received immunosuppressive therapy",
    )


def test_unspecified_polarity_counts_as_inclusion() -> None:
    # No polarity heading at all: upstream's fields put the whole text in inclusion.
    text = "Adults over 18 years.\n\nAble to give consent."
    assert criterion_sections(text) == (text, "")
    # A generic heading before an exclusion heading: everything before the cut is inclusion.
    text = "Eligibility Criteria:\n\n- Adults over 18 years.\n\nExclusion Criteria:\n\n- Pregnancy."
    inclusion, exclusion = criterion_sections(text)
    assert split_criteria(inclusion) == ("Eligibility Criteria:", "- Adults over 18 years.")
    assert split_criteria(exclusion) == ("- Pregnancy.",)


def test_a_text_with_only_exclusion_criteria_has_no_inclusion_section() -> None:
    inclusion, exclusion = criterion_sections("Exclusion Criteria:\n\n- Pregnancy.")
    assert inclusion == ""
    assert split_criteria(exclusion) == ("- Pregnancy.",)


def test_the_method_contract_records_the_criterion_source() -> None:
    contract = trialgpt_publication_method_contract()
    assert contract["criterion_source"] == TRIALGPT_CRITERION_SOURCE
    assert TRIALGPT_CRITERION_SOURCE == (
        "upstream-parse-criteria-over-split-v3-polarity-sections-v1"
    )


def test_trialgpt_1_needs_the_complete_text_and_not_the_criterion_view() -> None:
    declared = (
        TrialGPTTAIMLunaV1System.required_capabilities
        | TrialGPTTAIMLunaV1System.optional_capabilities
    )
    assert CAPABILITY_COMPLETE_ELIGIBILITY_TEXT in TrialGPTTAIMLunaV1System.required_capabilities
    assert ELIGIBILITY_CRITERION_VIEW_CAPABILITY not in declared


# Agreement with upstream on real trials. Opt in with TAIM_TRIALGPT1_UPSTREAM_CHECK naming a JSON
# file {"<target>": {"trials": "<package>/trials.jsonl", "upstream": "<upstream corpus.jsonl>"}};
# the corpus files are upstream's public per-Track trial corpora, with NCBI's pre-separated
# criterion fields. It reads trial texts only. Where upstream's two fields carry the
# same words as TAIM's two sections, the criteria agree, or differ only where NCBI's preparation
# of those fields changed their layout: it cut the phrase "exclusion criteria" out of a heading
# paragraph and kept its remainder, split at "~", or joined a wrapped line that broke the phrase.
# Trials where a list marker, which NCBI's fields drop, lifts a criterion of under five characters
# past parse_criteria's length filter are a separate, counted cause and are left out here. At most
# 1% of a sample differs otherwise: upstream's files also lay some lists out without blank lines.

_CHECK = os.environ.get("TAIM_TRIALGPT1_UPSTREAM_CHECK")
_MARKER = re.compile(r"^(?:[-*•·▪‣o]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|\(?[ivxIVX]{1,4}[.)])\s+")
_PHRASE = re.compile(r"(?:inclusion|exclusion)\s+criteria", re.IGNORECASE)
_HEADING_WORDS = {"inclusion", "exclusion", "criteria"}


def _key(item: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _MARKER.sub("", item, count=1).lower())


def _words(text: str) -> Counter[str]:
    words: Counter[str] = Counter()
    for line in text.replace("\r", "\n").split("\n"):
        line = _MARKER.sub("", line.strip(), count=1).lower()
        words.update(word for word in re.findall(r"[a-z0-9]+", line) if word not in _HEADING_WORDS)
    return words


def _keys(inclusion: str, exclusion: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(_key(item) for item in split_criteria(inclusion)),
        tuple(_key(item) for item in split_criteria(exclusion)),
    )


def _explained(source: str, inclusion: str, exclusion: str, upstream: tuple[str, str]) -> bool:
    ours = _keys(inclusion.replace("~", "\n\n"), exclusion.replace("~", "\n\n"))
    theirs = _keys(*upstream)
    if ours == theirs:
        return True
    headings = [
        _key(paragraph)
        for paragraph in source.replace("\r\n", "\n").split("\n\n")
        if _PHRASE.search(paragraph)
    ]
    ours_all, theirs_all = Counter(ours[0] + ours[1]), Counter(theirs[0] + theirs[1])
    return all(
        item and any(item in heading for heading in headings) for item in theirs_all - ours_all
    ) and all(re.search(r"(?:inclusion|exclusion)criteria", item) for item in ours_all - theirs_all)


@pytest.mark.skipif(_CHECK is None, reason="set TAIM_TRIALGPT1_UPSTREAM_CHECK to run")
def test_criteria_match_upstream_where_its_fields_carry_the_same_text() -> None:
    targets = json.loads(Path(str(_CHECK)).read_text(encoding="utf-8"))
    for target, paths in targets.items():
        upstream: dict[str, tuple[str, str]] = {}
        with Path(paths["upstream"]).open(encoding="utf-8") as corpus:
            for line in corpus:
                entry = json.loads(line)
                metadata = entry["metadata"]
                upstream[entry["_id"]] = (
                    metadata["inclusion_criteria"],
                    metadata["exclusion_criteria"],
                )
        same_text: list[tuple[str, str, str, str]] = []
        with Path(paths["trials"]).open(encoding="utf-8") as trials:
            for line in trials:
                trial = json.loads(line)
                source = next(
                    (
                        section.get("source_text")
                        for section in trial["sections"]
                        if section["role"] == "eligibility"
                    ),
                    None,
                )
                if source is None or trial["trial_id"] not in upstream:
                    continue
                inclusion, exclusion = criterion_sections(source)
                fields = upstream[trial["trial_id"]]
                short = any(
                    len(_MARKER.sub("", item, count=1).strip()) < 5
                    for section in (inclusion, exclusion)
                    for item in split_criteria(section)
                )
                if (
                    not short
                    and _words(inclusion) == _words(fields[0])
                    and _words(exclusion) == _words(fields[1])
                ):
                    same_text.append((trial["trial_id"], source, inclusion, exclusion))
        assert len(same_text) > 1_000, target
        sample = random.Random(2026).sample(same_text, 500)  # noqa: S311 - a fixed sample
        unexplained = [
            trial_id
            for trial_id, source, inclusion, exclusion in sample
            if _keys(inclusion, exclusion) != _keys(*upstream[trial_id])
            and not _explained(source, inclusion, exclusion, upstream[trial_id])
        ]
        agreed = sum(
            _keys(inclusion, exclusion) == _keys(*upstream[trial_id])
            for trial_id, _source, inclusion, exclusion in sample
        )
        assert agreed >= 475, (target, agreed)
        assert len(unexplained) <= 5, (target, unexplained)
