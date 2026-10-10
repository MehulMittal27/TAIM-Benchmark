"""`queries prepare` refuses a missing or different SNOMED CT release before the corpus loads."""

from __future__ import annotations

# ruff: noqa: S101
import hashlib
import json
from pathlib import Path

import pytest

import taim.folded_query_bundle as folded_query_bundle
from taim.release_cli import main

LOCK = json.loads(folded_query_bundle.SNOMED_EDITION_LOCK.read_text(encoding="utf-8"))


class CorpusLoaded(Exception):
    """Raised in place of the prepared-corpus load, so a test sees whether it was reached."""


@pytest.fixture(autouse=True)
def corpus_load_is_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(_path: Path) -> None:
        raise CorpusLoaded

    monkeypatch.setattr("taim.data.load_prepared_benchmark", load)


def _prepare(tmp_path: Path, release: Path) -> None:
    main(
        [
            "patient-to-trial",
            "queries",
            "prepare",
            "--data-dir",
            str(tmp_path / "prepared"),
            "--clinical-as-of",
            "2021-04-27T00:00:00Z",
            "--snomed-release",
            str(release),
            "--output",
            str(tmp_path / "folded.json"),
        ]
    )


def _release(root: Path, description: bytes) -> Path:
    terminology = root / "Snapshot" / "Terminology"
    terminology.mkdir(parents=True)
    (terminology / "sct2_Concept_Snapshot_INT_20250101.txt").write_bytes(b"id\n")
    (terminology / "sct2_Description_Snapshot-en_INT_20250101.txt").write_bytes(description)
    return root


def test_a_directory_that_is_no_rf2_release_is_refused_naming_the_licence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _prepare(tmp_path, tmp_path / "not-a-release")

    error = capsys.readouterr().err
    assert "is not a SNOMED CT RF2 release: no sct2_Concept_Snapshot*.txt" in error
    assert "SNOMED CT is licensed and TAIM ships none" in error
    assert "your own licence" in error


def test_another_edition_is_refused_naming_both_editions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    release = _release(tmp_path / "an-earlier-release", b"another edition\n")
    supplied = "sha256:" + hashlib.sha256(b"another edition\n").hexdigest()

    with pytest.raises(SystemExit):
        _prepare(tmp_path, release)

    error = capsys.readouterr().err
    assert "is not the edition bm25-folded is qualified on" in error
    assert f"sct2_Description_Snapshot-en_INT_20250101.txt has {supplied}" in error
    expected = LOCK["description_file"]
    assert f"expected {LOCK['edition']}, {expected['name']} with {expected['sha256']}" in error


def test_the_locked_edition_passes_on_to_the_corpus_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _release(tmp_path / "any-name-the-user-chose", b"the locked edition\n")
    lock = {
        **LOCK,
        "description_file": {
            **LOCK["description_file"],
            "sha256": "sha256:" + hashlib.sha256(b"the locked edition\n").hexdigest(),
        },
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(folded_query_bundle, "SNOMED_EDITION_LOCK", lock_path)

    with pytest.raises(CorpusLoaded):
        _prepare(tmp_path, release)


def test_the_lock_pins_one_edition_by_its_description_file() -> None:
    assert LOCK["input_id"] == "SNOMED-CT:user-supplied-release"
    assert LOCK["description_file"]["name"].startswith("sct2_Description_Snapshot-en_")
    assert len(LOCK["description_file"]["sha256"]) == len("sha256:") + 64
    assert LOCK["release_policy"]["source_bytes_shipped"] is False


def test_the_release_option_says_it_is_licensed_data_and_names_the_edition(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["patient-to-trial", "queries", "prepare", "--help"])

    help_text = " ".join(capsys.readouterr().out.split())
    assert f"must be the {LOCK['edition']}" in help_text
    assert "licensed data you supply under your own SNOMED CT licence; TAIM ships none" in help_text
