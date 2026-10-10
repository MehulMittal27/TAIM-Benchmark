# ruff: noqa: S101
"""The TrialMatchAI release layer builds a location-free System Input and verifies what it binds.

Each check here has one refusal or one equality to own, and each test builds its own corpus: the
release layer resolves the physical inputs, refuses an absolute command token and a corpus without
a preparation receipt before any run starts, binds the receipt's content digest, checks the
corpus's names and counts by a directory walk, and hashes the search database it is given.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.patient_to_trial import PatientToTrialRunRequest, normalize_system_request
from taim.pipelines.trialmatchai.corpus import (
    trialmatchai_search_database_identity,
    write_trialmatchai_corpus_receipt,
)
from taim.pipelines.trialmatchai.execution import _located_request
from taim.pipelines.trialmatchai.system import _trialmatchai_index_identity
from taim.pipelines.trialmatchai.trec2021 import TrialMatchAIL4System
from taim.release_cli import main
from taim.release_fixture import FIXTURE_CLINICAL_AS_OF, load_release_fixture
from taim.release_profiles import resolve_benchmark_profile
from taim.release_reference_systems import run_release_reference_system
from taim.release_result_bundle import is_absolute_filesystem_path

TRACK = "trec-ct-2021"
PROFILE = "synthetic-patient-to-trial"
COMMIT = "99aa499e7296ebde836a0ebf00a5ed4948892a83"
RECEIPT = "trialmatchai-corpus-receipt.json"


def _trial_ids() -> list[str]:
    return [trial.trial_id for trial in load_release_fixture(TRACK).snapshot.trials]


def _corpus(root: Path, *, receipt: bool = True) -> Path:
    corpus = root / "corpus"
    for trial_id in _trial_ids():
        path = corpus / "processed_trials" / f"{trial_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": trial_id, "embedding": [0.25]}), encoding="utf-8")
        criterion = corpus / "processed_criteria" / trial_id / "c1.json"
        criterion.parent.mkdir(parents=True)
        criterion.write_text(json.dumps({"trial": trial_id}), encoding="utf-8")
    (corpus / "search" / "criteria.lance").mkdir(parents=True)
    (corpus / "search" / "_embedder.json").write_text('{"type": "hf"}', encoding="utf-8")
    (corpus / "search" / "criteria.lance" / "0.lance").write_bytes(b"synthetic lance bytes")
    if receipt:
        write_trialmatchai_corpus_receipt(corpus, workers=2)
    return corpus


def _request(
    root: Path,
    corpus: Path,
    *,
    run_id: str = "r1",
    timeout: float = 3600.0,
    command: tuple[str, ...] = ("uv", "run", "trialmatchai", "e2e", "--input", "{input}"),
) -> PatientToTrialRunRequest:
    from taim.pipelines.trialmatchai.system import trialmatchai_l4_runtime_contract

    checkout = root / "TrialMatchAI"
    checkout.mkdir(exist_ok=True)
    benchmark = resolve_benchmark_profile(load_release_fixture(TRACK), profile_id=PROFILE)
    return PatientToTrialRunRequest.for_benchmark(
        benchmark,
        run_id=run_id,
        top_k=1,
        metric_cutoff=10,
        options={
            "command": command,
            "cwd": checkout,
            "workspace": root / "runs" / ".trialmatchai" / run_id,
            "timeout_seconds": timeout,
            "corpus_dir": corpus,
            "search_db_path": corpus / "search",
            "trialmatchai_expected_commit": COMMIT,
            "trialmatchai_expected_repository": "https://github.com/MehulMittal27/TrialMatchAI.git",
            "trialmatchai_runtime_contract": trialmatchai_l4_runtime_contract(),
        },
        clinical_as_of=FIXTURE_CLINICAL_AS_OF,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )


def _identity(request: PatientToTrialRunRequest) -> PatientToTrialRunRequest:
    system, located, _database, _run_start = _located_request(TrialMatchAIL4System, request)
    return normalize_system_request(system, located)  # type: ignore[arg-type]


def _strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _strings(nested)]
    return [value] if isinstance(value, str) else []


def test_the_system_input_is_the_same_at_two_sites_under_two_run_ids(tmp_path: Path) -> None:
    site_a = tmp_path / "site-a"
    site_b = tmp_path / "deeper" / "site-b"
    first = _identity(_request(site_a, _corpus(site_a), run_id="r1"))
    second = _identity(_request(site_b, _corpus(site_b), run_id="r2"))

    assert first.system_input_id == second.system_input_id
    recorded = first.system_input_record_dict()
    assert not [text for text in _strings(recorded) if is_absolute_filesystem_path(text)]
    assert str(tmp_path) not in json.dumps(recorded)


def test_the_timeout_does_not_change_the_system_input_id(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    short = _located_request(TrialMatchAIL4System, _request(tmp_path, corpus, timeout=3600.0))
    long = _located_request(TrialMatchAIL4System, _request(tmp_path, corpus, timeout=10800.0))

    assert short[0].harness_inputs.timeout_seconds == 3600.0  # type: ignore[attr-defined]
    assert long[0].harness_inputs.timeout_seconds == 10800.0  # type: ignore[attr-defined]
    assert short[1].system_input_id == long[1].system_input_id


def test_an_absolute_command_token_is_refused_when_the_identity_is_built(tmp_path: Path) -> None:
    request = _request(
        tmp_path, _corpus(tmp_path), command=(sys.executable, "-m", "trialmatchai", "e2e")
    )

    with pytest.raises(ValueError, match=r"command tokens command\[0\] are absolute filesystem"):
        _located_request(TrialMatchAIL4System, request)


def test_a_harness_input_system_refuses_a_location_as_an_option(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    system, located, _database, _run_start = _located_request(
        TrialMatchAIL4System, _request(tmp_path, corpus)
    )
    slipped = PatientToTrialRunRequest(
        snapshot=located.snapshot,
        run_id=located.run_id,
        top_k=located.top_k,
        metric_cutoff=located.metric_cutoff,
        options={**located.options, "cwd": tmp_path},
    )

    with pytest.raises(ValueError, match="unknown method options: cwd"):
        normalize_system_request(system, slipped)  # type: ignore[arg-type]


def test_a_run_whose_corpus_has_no_receipt_is_refused_before_it_starts(tmp_path: Path) -> None:
    request = _request(tmp_path, _corpus(tmp_path, receipt=False))

    # Refused where the System Input is built, which runs no later check that could refuse instead.
    with pytest.raises(ValueError, match="prepared corpus has no preparation receipt"):
        _located_request(TrialMatchAIL4System, request)
    # And the release dispatch refuses it the same way, before the System or its workspace exists.
    with pytest.raises(ValueError, match="prepared corpus has no preparation receipt"):
        run_release_reference_system(TrialMatchAIL4System.system_id, request)
    assert not (tmp_path / "runs").exists()


def test_adoption_writes_a_location_free_receipt_bound_to_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(tmp_path, receipt=False)
    copy = tmp_path / "elsewhere" / "corpus"
    shutil.copytree(corpus, copy)

    adopt = ["patient-to-trial", "trialmatchai", "adopt-corpus", "--corpus-dir", str(corpus)]
    assert main(adopt) == 0
    printed = json.loads(capsys.readouterr().out)
    receipt = json.loads((corpus / RECEIPT).read_text(encoding="utf-8"))
    assert printed["content_sha256"] == receipt["content_sha256"]
    assert receipt["roles"]["processed_trials"]["file_count"] == len(_trial_ids())
    assert not [text for text in _strings(receipt) if is_absolute_filesystem_path(text)]
    assert write_trialmatchai_corpus_receipt(copy, workers=1) == receipt

    # One prepared file changes, keeping its name and size: the content digest must follow it.
    victim = copy / "processed_trials" / f"{_trial_ids()[0]}.json"
    victim.write_bytes(victim.read_bytes().replace(b"0.25", b"0.75"))
    (copy / RECEIPT).unlink()
    changed = write_trialmatchai_corpus_receipt(copy, workers=1)
    assert changed["names_sha256"] == receipt["names_sha256"]
    assert changed["content_sha256"] != receipt["content_sha256"]


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ("missing", "is missing 1 processed_criteria file"),
        ("extra", "has 1 processed_trials file"),
        ("renamed", "file names differ from its preparation receipt"),
    ],
)
def test_the_run_start_walk_catches_a_changed_file_set(
    tmp_path: Path, change: str, refusal: str
) -> None:
    corpus = _corpus(tmp_path)
    request = _request(tmp_path, corpus)
    _located_request(TrialMatchAIL4System, request)
    criterion = corpus / "processed_criteria" / _trial_ids()[0] / "c1.json"
    if change == "missing":
        criterion.unlink()
    elif change == "extra":
        (corpus / "processed_trials" / "NCT-EXTRA.json").write_text("{}", encoding="utf-8")
    else:
        os.rename(criterion, criterion.with_name("c2.json"))

    with pytest.raises(ValueError, match=refusal):
        _located_request(TrialMatchAIL4System, request)


def test_the_search_database_digest_changes_when_a_staged_file_changes(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    staged = tmp_path / "node-local" / "search"
    shutil.copytree(corpus / "search", staged)
    before = trialmatchai_search_database_identity(staged, workers=2)
    assert (
        before["content_sha256"]
        == trialmatchai_search_database_identity(corpus / "search", workers=1)["content_sha256"]
    )

    table = staged / "criteria.lance" / "0.lance"
    table.write_bytes(table.read_bytes().replace(b"synthetic", b"SYNTHETIC"))
    after = trialmatchai_search_database_identity(staged, workers=2)

    assert after["file_count"] == before["file_count"]
    assert after["byte_count"] == before["byte_count"]
    assert after["content_sha256"] != before["content_sha256"]


def test_the_index_identity_carries_no_locator(tmp_path: Path) -> None:
    request = _request(tmp_path, _corpus(tmp_path))
    identity = _trialmatchai_index_identity(
        request,
        config_payload={"search_backend": {"backend": "lancedb", "db_path": "data/search"}},
        processed_trials_folder=tmp_path / "corpus" / "processed_trials",
        processed_criteria_folder=tmp_path / "corpus" / "processed_criteria",
        search_db_path=tmp_path / "corpus" / "search",
        concept_store={
            "db_path": str(tmp_path / "concepts"),
            "manifest_path": str(tmp_path / "concepts" / "concept-store-manifest.json"),
            "manifest_sha256": "sha256:" + "1" * 64,
            "build": {"checkpoint_path": str(tmp_path / "checkpoint"), "rows": 3},
            "open_dictionary_sources": [{"path": str(tmp_path / "dictionary"), "rows": 2}],
        },
    )

    def keys(value: object) -> list[str]:
        if isinstance(value, dict):
            return [*value, *(key for nested in value.values() for key in keys(nested))]
        if isinstance(value, list):
            return [key for nested in value for key in keys(nested)]
        return []

    locators = {"directory", "path"}
    assert not [key for key in keys(identity) if key in locators or key.endswith(("_dir", "_path"))]
    assert not [text for text in _strings(identity) if is_absolute_filesystem_path(text)]
    assert identity["concept_store"]["build"] == {"rows": 3}
    assert identity["search_backend"] == {"backend": "lancedb"}
