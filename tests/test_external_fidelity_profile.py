"""The external-fidelity Profile's public checks, which do not depend on any external System."""

from __future__ import annotations

import hashlib

# ruff: noqa: S101
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import taim.release_profiles as release_profiles
from taim.pipeline_extensions import pipeline_extensions
from taim.public_patient_to_trial import (
    _validate_judgment_union_membership,
)
from taim.release_cli import _verified_external_pool_ids
from taim.release_fixture import load_release_fixture
from taim.release_profiles import (
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
    inspect_external_fidelity_profile,
    ordered_trial_ids_sha256,
    resolve_benchmark_profile,
)
from taim.release_support import (
    FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256,
    load_release_support,
    require_frozen_effectiveness_protocol,
)
from taim.schemas import SchemaValidationError


def test_external_fidelity_rows_belong_to_the_selected_extensions() -> None:
    support_matrix = load_release_support()["support_matrix"]
    rows = [row for row in support_matrix if row["profile"] == EXTERNAL_FIDELITY_TREC_2021_PROFILE]
    assert all(
        (row["track"], row["task"], row["evidence_scope"])
        == ("trec-ct-2021", "patient_to_trial", "external_fidelity_effectiveness")
        for row in rows
    )
    # A pipeline extension's own public test checks the Systems it declares.
    extension_systems = {
        system for extension in pipeline_extensions() for system in extension.system_tasks
    }
    assert {row["system"] for row in rows} <= extension_systems
    # Equality, not a denylist: an undeclared System fails here without this test naming it.
    assert {row["system"] for row in support_matrix} == extension_systems


def test_external_fidelity_pool_requires_exact_caller_supplied_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = load_release_fixture("trec-ct-2021")
    selected_ids = tuple(trial.trial_id for trial in prepared.trials[:-1])
    production_profile = release_profiles.load_profile(EXTERNAL_FIDELITY_TREC_2021_PROFILE)
    assert production_profile.expected_pool_count == 26_149
    fixture_profile = replace(production_profile, expected_pool_count=len(selected_ids))
    original_load_profile = release_profiles.load_profile
    monkeypatch.setattr(
        release_profiles,
        "load_profile",
        lambda profile_id: (
            fixture_profile
            if profile_id == EXTERNAL_FIDELITY_TREC_2021_PROFILE
            else original_load_profile(profile_id)
        ),
    )
    receipt = inspect_external_fidelity_profile(
        prepared,
        profile_id=EXTERNAL_FIDELITY_TREC_2021_PROFILE,
        supplied_pool_ids=selected_ids,
        expected_pool_count=len(selected_ids),
    )
    assert receipt["selection_policy"] == "caller_supplied_external_fidelity_membership"
    assert receipt["pool_ids_sha256"] == ordered_trial_ids_sha256(selected_ids)

    resolved = resolve_benchmark_profile(
        prepared,
        profile_id=EXTERNAL_FIDELITY_TREC_2021_PROFILE,
        expected_pool_count=len(selected_ids),
        expected_pool_ids_sha256=receipt["pool_ids_sha256"],
        expected_pool_receipt_id=receipt["pool_receipt_id"],
        supplied_pool_ids=selected_ids,
    )
    assert tuple(trial.trial_id for trial in resolved.trials) == selected_ids
    assert resolved.manifest_configuration()["corpus_policy"] == "external_fidelity_pool"
    assert {judgment.trial_id for judgment in resolved.evaluation_package.judgments} == set(
        selected_ids
    )
    assert (
        resolved.evaluation_package.evaluation_package_id
        != prepared.evaluation_package.evaluation_package_id
    )
    assert resolved.manifest_configuration()["pool_source_sha256"] == (
        resolved.evaluation_package.evaluation_package_id
    )
    assert resolved.evaluation_package.provenance["release_profile_projection"] == {
        "profile_id": EXTERNAL_FIDELITY_TREC_2021_PROFILE,
        "source_evaluation_package_id": prepared.evaluation_package.evaluation_package_id,
        "selection_policy": "caller_supplied_external_fidelity_membership",
    }
    _validate_judgment_union_membership(
        SimpleNamespace(
            benchmark_profile=resolved.manifest_configuration(),
            trial_corpus_versions=tuple(
                (trial.trial_id, "sha256:" + "1" * 64) for trial in resolved.trials
            ),
            prepared_snapshot_id=resolved.prepared_snapshot_id,
            evaluation_package_id=resolved.evaluation_package_id,
        ),
        resolved.evaluation_package,
    )

    with pytest.raises(ValueError, match="caller-supplied pool membership"):
        resolve_benchmark_profile(
            prepared,
            profile_id=EXTERNAL_FIDELITY_TREC_2021_PROFILE,
            expected_pool_count=len(selected_ids),
            expected_pool_ids_sha256=receipt["pool_ids_sha256"],
            expected_pool_receipt_id=receipt["pool_receipt_id"],
        )
    with pytest.raises(ValueError, match="unknown trial ID"):
        inspect_external_fidelity_profile(
            prepared,
            profile_id=EXTERNAL_FIDELITY_TREC_2021_PROFILE,
            supplied_pool_ids=(*selected_ids, "NCT-NOT-IN-PREPARED-CORPUS"),
            expected_pool_count=len(selected_ids) + 1,
        )


def test_external_pool_source_verification_extracts_only_sorted_ids(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"
    source.write_text(
        '{"_id":"NCT-B","brief_title":"not copied"}\n{"_id":"NCT-A","brief_title":"not copied"}\n',
        encoding="utf-8",
    )
    source_sha256 = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    assert _verified_external_pool_ids(
        source,
        expected_byte_size=source.stat().st_size,
        expected_sha256=source_sha256,
        expected_count=2,
        expected_sorted_ids_sha256=ordered_trial_ids_sha256(("NCT-A", "NCT-B")),
    ) == ("NCT-A", "NCT-B")

    with pytest.raises(ValueError, match="checksum lock"):
        _verified_external_pool_ids(
            source,
            expected_byte_size=source.stat().st_size,
            expected_sha256="sha256:" + "0" * 64,
            expected_count=2,
            expected_sorted_ids_sha256=ordered_trial_ids_sha256(("NCT-A", "NCT-B")),
        )


def test_external_fidelity_profile_protocol_gate_matches_release_state() -> None:
    protocol_id = FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256
    if protocol_id is None:
        with pytest.raises(SchemaValidationError, match="not frozen by explicit human approval"):
            require_frozen_effectiveness_protocol(
                EXTERNAL_FIDELITY_TREC_2021_PROFILE,
                "sha256:" + "0" * 64,
                "external protocol",
            )
        return
    assert (
        require_frozen_effectiveness_protocol(
            EXTERNAL_FIDELITY_TREC_2021_PROFILE,
            protocol_id,
            "external protocol",
        )
        == protocol_id
    )
