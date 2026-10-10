# ruff: noqa: S101
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from taim.data import PREPARED_MANIFEST_FILENAME, load_prepared_benchmark
from taim.data import complete_eligibility as builder
from taim.data.complete_eligibility import prepare_profile_complete_eligibility_snapshot
from taim.data.sigir_ct_2016 import prepare_sigir_ct_2016
from taim.data.trec_ct_2021 import IngestionError, prepare_trec_ct_2021
from taim.eligibility import (
    ELIGIBILITY_CRITERION_VIEW_CAPABILITY,
    eligibility_criterion_trial,
    find_eligibility_criterion_view,
)
from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.patient_to_trial import PatientToTrialRunRequest
from taim.release_profiles import OFFICIAL_FULL_PROFILE, resolve_benchmark_profile
from taim.schemas import SchemaValidationError
from taim.snapshot import CAPABILITY_COMPLETE_ELIGIBILITY_TEXT
from tests.snapshot_helpers import (
    TREC_FIXED_TIME,
    complete_trial_xml,
    write_sigir_source_bundle,
    write_trec_source_bundle,
)

_COMPLETE = frozenset({CAPABILITY_COMPLETE_ELIGIBILITY_TEXT, ELIGIBILITY_CRITERION_VIEW_CAPABILITY})
_LONG_CRITERION = "Adults with alpha syndrome confirmed by biopsy. " * 120


def _trec_stratum(tmp_path: Path) -> tuple[Path, Path]:
    """A stratum-shaped TREC package whose eligibility text exceeds the 4,096-byte excerpt."""

    serialized = complete_trial_xml().replace(
        "<criteria><textblock>Adults with alpha syndrome.\n"
        "        Excludes cardiac disease.</textblock></criteria>",
        "<criteria><textblock>Inclusion Criteria:\n"
        f"- {_LONG_CRITERION}\n"
        "Exclusion Criteria:\n"
        "- Cardiac disease.</textblock></criteria>",
    )
    source, lock = write_trec_source_bundle(tmp_path, trials={"NCT00000001": serialized})
    prepared = prepare_trec_ct_2021(
        source, tmp_path / "stratum", source_lock_path=lock, created_at=TREC_FIXED_TIME
    )
    return prepared.output_directory, source


def _build(stratum: Path, source: Path, output: Path, profile: str = OFFICIAL_FULL_PROFILE) -> Path:
    return prepare_profile_complete_eligibility_snapshot(
        stratum,
        source,
        output,
        profile_id=profile,
        created_at=datetime(2026, 10, 7, tzinfo=UTC),
    ).output_directory


def _task_input_id(directory: Path) -> str:
    prepared = load_prepared_benchmark(directory)
    request = PatientToTrialRunRequest.for_benchmark(
        resolve_benchmark_profile(prepared, profile_id=OFFICIAL_FULL_PROFILE),
        run_id="identity",
        top_k=1,
        metric_cutoff=1,
        options={},
        clinical_as_of=TREC_FIXED_TIME,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )
    return request.snapshot.task_input_id


def test_a_complete_package_changes_only_the_trial_text_representation(tmp_path: Path) -> None:
    stratum_directory, source = _trec_stratum(tmp_path)
    stratum = load_prepared_benchmark(stratum_directory)

    complete = load_prepared_benchmark(_build(stratum_directory, source, tmp_path / "complete"))

    # Same patients, trials in the same order, judgments, dataset and Snapshot name.
    assert complete.dataset_id == stratum.dataset_id
    assert complete.snapshot.snapshot_name == stratum.snapshot.snapshot_name
    assert complete.topics == stratum.topics
    assert [t.trial_id for t in complete.trials] == [t.trial_id for t in stratum.trials]
    assert complete.evaluation_package.judgments == stratum.evaluation_package.judgments
    assert complete.source_bundle == stratum.source_bundle
    # The stratum keeps an excerpt; the complete package carries the full eligibility text.
    excerpt = next(s for s in stratum.trials[0].sections if s.role == "eligibility")
    full = next(s for s in complete.trials[0].sections if s.role == "eligibility")
    assert excerpt.source_text is None
    assert excerpt.provenance[0].raw_value_byte_length is not None
    assert full.source_text is not None
    assert _LONG_CRITERION.strip() in full.source_text
    assert len(full.source_text.encode()) > 4_096
    assert complete.snapshot.available_capabilities == (
        stratum.snapshot.available_capabilities | _COMPLETE
    )
    view = find_eligibility_criterion_view(complete.snapshot.derived_views)
    assert view is not None
    criteria = eligibility_criterion_trial(view, "NCT00000001")
    assert criteria is not None
    assert [item.text for item in criteria.exclusion] == ["Cardiac disease."]
    # Exactly the identities that hash the trial text move.
    assert complete.snapshot.snapshot_id != stratum.snapshot.snapshot_id
    assert complete.evaluation_package.evaluation_package_id != (
        stratum.evaluation_package.evaluation_package_id
    )
    assert _task_input_id(complete.directory) != _task_input_id(stratum.directory)
    assert (
        complete.snapshot.logical_content_hashes()["topics"]
        == (stratum.snapshot.logical_content_hashes()["topics"])
    )
    # Removing the complete text gives the stratum's records back.
    assert [builder._without_complete_eligibility(t).to_json() for t in complete.trials] == [
        t.to_json() for t in stratum.trials
    ]
    recipe = complete.prepared_manifest["snapshot_source_recipe"]["definition"]
    assert recipe["recipe_id"] == "trec-ct-2021-profile-complete-eligibility-v1"
    assert recipe["source_stratum"]["prepared_snapshot_id"] == stratum.snapshot.snapshot_id


def test_the_criterion_view_is_stored_beside_the_trials(tmp_path: Path) -> None:
    stratum_directory, source = _trec_stratum(tmp_path)

    directory = _build(stratum_directory, source, tmp_path / "complete")

    manifest = json.loads((directory / PREPARED_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    (descriptor,) = manifest["snapshot"]["derived_views"]
    assert "content" not in descriptor
    assert descriptor["content_reference"]["output_name"] == "eligibility_criteria"
    output = manifest["derived_view_outputs"]["eligibility_criteria"]
    lines = (directory / output["filename"]).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["trial_id"] for line in lines] == ["NCT00000001"]


def test_the_package_identity_is_reproducible(tmp_path: Path) -> None:
    stratum_directory, source = _trec_stratum(tmp_path)

    first = load_prepared_benchmark(_build(stratum_directory, source, tmp_path / "first"))
    second = load_prepared_benchmark(_build(stratum_directory, source, tmp_path / "second"))

    assert first.snapshot.snapshot_id == second.snapshot.snapshot_id
    assert first.preparation == second.preparation
    assert first.evaluation_package.evaluation_package_id == (
        second.evaluation_package.evaluation_package_id
    )


def test_a_trial_the_source_does_not_reproduce_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stratum_directory, source = _trec_stratum(tmp_path)
    parse = builder.parse_trial_xml

    def drifted(serialized: bytes, **kwargs: object) -> object:
        return parse(serialized.replace(b"Alpha treatment", b"Beta treatment"), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builder, "parse_trial_xml", drifted)

    with pytest.raises(IngestionError, match="does not reproduce the source package's record"):
        _build(stratum_directory, source, tmp_path / "complete")
    assert not (tmp_path / "complete").exists()


def test_a_trial_missing_from_the_source_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stratum_directory, source = _trec_stratum(tmp_path)
    monkeypatch.setattr(builder, "_complete_source_trials", lambda *_args: {})

    with pytest.raises(IngestionError, match="does not reproduce Profile trial NCT00000001"):
        _build(stratum_directory, source, tmp_path / "complete")


def test_a_source_bundle_that_does_not_verify_is_refused(tmp_path: Path) -> None:
    stratum_directory, source = _trec_stratum(tmp_path)
    (source / "qrels2021.txt").write_text("1 0 NCT00000001 1\n", encoding="utf-8")

    with pytest.raises(IngestionError, match=r"qrels2021\.txt: SHA-256 does not match"):
        _build(stratum_directory, source, tmp_path / "complete")


def test_a_complete_package_is_not_a_source(tmp_path: Path) -> None:
    stratum_directory, source = _trec_stratum(tmp_path)
    complete = _build(stratum_directory, source, tmp_path / "complete")

    with pytest.raises(SchemaValidationError, match="must be a stratum package"):
        _build(complete, source, tmp_path / "again")


def test_sigir_trials_are_re_read_from_the_tarball(tmp_path: Path) -> None:
    source, lock = write_sigir_source_bundle(tmp_path)
    stratum_directory = prepare_sigir_ct_2016(
        source,
        tmp_path / "stratum",
        source_lock_path=lock,
        query_variant="summary",
        created_at=TREC_FIXED_TIME,
    ).output_directory
    stratum = load_prepared_benchmark(stratum_directory)

    complete = load_prepared_benchmark(
        _build(stratum_directory, source, tmp_path / "complete", profile="summary")
    )

    assert [t.trial_id for t in complete.trials] == ["NCT00000001", "NCT00000002"]
    assert complete.topics == stratum.topics
    assert complete.evaluation_package.judgments == stratum.evaluation_package.judgments
    assert complete.snapshot.available_capabilities >= _COMPLETE
    assert all(
        section.source_text is not None
        for trial in complete.trials
        for section in trial.sections
        if section.role == "eligibility"
    )


def test_the_package_identities_are_pinned(tmp_path: Path) -> None:
    """Paper rows bind these packages' identities, so code that would move them fails here."""

    stratum_directory, source = _trec_stratum(tmp_path)
    stratum = load_prepared_benchmark(stratum_directory)

    complete = load_prepared_benchmark(_build(stratum_directory, source, tmp_path / "complete"))

    assert stratum.snapshot.snapshot_id == (
        "sha256:688bc309e995193d9cbe1c54a10bab006339727397f32bdd65ff0ff59c909a03"
    )
    assert complete.snapshot.snapshot_id == (
        "sha256:9cb44165eb44a84481e21e45162574cedc8a56bab25e32773a81726316ccf390"
    )
    assert complete.evaluation_package.evaluation_package_id == (
        "sha256:2a40549cdb7189e1ef3f69b91a83a25720b88121a99c40ed681095cf7fdcbb54"
    )
    assert complete.preparation.source_recipe_hash == (
        "sha256:b1d9404fba05a954c5f07e25313a577b0bbaf8d0b8f6da3ec66b0b407eef7d76"
    )
    assert _task_input_id(complete.directory) == (
        "sha256:2ba25321ab63dabf0014c70a4f0b3702abfd7c93f4464c53228b58b39158438c"
    )
