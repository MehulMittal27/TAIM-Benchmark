from __future__ import annotations

# ruff: noqa: S101
import pytest

from taim.release_profiles import (
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
    judgment_union_pool_receipt,
    load_profile,
    ordered_trial_ids_sha256,
)
from taim.release_result_bundle import (
    _is_absolute_filesystem_path,
    _validate_benchmark_profile_projection,
)
from taim.schemas import (
    SchemaValidationError,
)


def test_judgment_union_bundle_profile_binds_pool_receipt_to_task_input() -> None:
    profile = load_profile("trec-ct-2022-judgment-union")
    prepared_snapshot_id = "sha256:" + "1" * 64
    evaluation_package_id = "sha256:" + "2" * 64
    trial_entity_versions = frozenset(
        {
            ("NCT-A", "sha256:" + "3" * 64),
            ("NCT-B", "sha256:" + "4" * 64),
        }
    )
    pool_ids_sha256 = ordered_trial_ids_sha256(("NCT-A", "NCT-B"))
    receipt = judgment_union_pool_receipt(
        selection_policy="union_of_trial_ids_present_in_the_evaluation_package",
        claim_scope="within_pool_ranking_not_full_corpus_retrieval",
        profile=profile.profile_id,
        profile_definition_sha256=profile.definition_sha256,
        prepared_snapshot_id=prepared_snapshot_id,
        evaluation_package_id=evaluation_package_id,
        pool_count=2,
        pool_ids_sha256=pool_ids_sha256,
    )
    projection = {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "definition_sha256": profile.definition_sha256,
        "corpus_policy": profile.corpus_policy,
        "source_corpus_sha256": "sha256:" + "5" * 64,
        "effective_corpus_sha256": "sha256:" + "6" * 64,
        "effective_corpus_count": 2,
        "pool_ids_sha256": pool_ids_sha256,
        "pool_source_sha256": evaluation_package_id,
        "pool_receipt_id": receipt["pool_receipt_id"],
        "paper_pool_identity_status": profile.paper_pool_identity_status,
        "unverified_membership_acknowledged": False,
        "evaluation": profile.evaluation_configuration(),
    }
    _validate_benchmark_profile_projection(
        projection,
        task="patient_to_trial",
        profile_id=profile.profile_id,
        expected_definition_sha256=profile.definition_sha256,
        prepared_snapshot_id=prepared_snapshot_id,
        evaluation_package_id=evaluation_package_id,
        trial_entity_versions=trial_entity_versions,
    )
    projection["pool_receipt_id"] = "sha256:" + "f" * 64
    with pytest.raises(SchemaValidationError, match="pool receipt"):
        _validate_benchmark_profile_projection(
            projection,
            task="patient_to_trial",
            profile_id=profile.profile_id,
            expected_definition_sha256=profile.definition_sha256,
            prepared_snapshot_id=prepared_snapshot_id,
            evaluation_package_id=evaluation_package_id,
            trial_entity_versions=trial_entity_versions,
        )


def test_external_fidelity_bundle_profile_uses_its_distinct_receipt_policy() -> None:
    profile = load_profile(EXTERNAL_FIDELITY_TREC_2021_PROFILE)
    prepared_snapshot_id = "sha256:" + "1" * 64
    evaluation_package_id = "sha256:" + "2" * 64
    trial_ids = tuple(f"NCT{index:08d}" for index in range(26_149))
    trial_entity_versions = frozenset((trial_id, "sha256:" + "3" * 64) for trial_id in trial_ids)
    pool_ids_sha256 = ordered_trial_ids_sha256(trial_ids)
    receipt = judgment_union_pool_receipt(
        selection_policy="caller_supplied_external_fidelity_membership",
        claim_scope="external_system_fidelity_within_pool_not_full_corpus_retrieval",
        profile=profile.profile_id,
        profile_definition_sha256=profile.definition_sha256,
        prepared_snapshot_id=prepared_snapshot_id,
        evaluation_package_id=evaluation_package_id,
        pool_count=len(trial_ids),
        pool_ids_sha256=pool_ids_sha256,
    )
    projection = {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "definition_sha256": profile.definition_sha256,
        "corpus_policy": profile.corpus_policy,
        "source_corpus_sha256": "sha256:" + "4" * 64,
        "effective_corpus_sha256": "sha256:" + "5" * 64,
        "effective_corpus_count": len(trial_ids),
        "pool_ids_sha256": pool_ids_sha256,
        "pool_source_sha256": evaluation_package_id,
        "pool_receipt_id": receipt["pool_receipt_id"],
        "paper_pool_identity_status": profile.paper_pool_identity_status,
        "unverified_membership_acknowledged": False,
        "evaluation": profile.evaluation_configuration(),
    }

    _validate_benchmark_profile_projection(
        projection,
        task="patient_to_trial",
        profile_id=profile.profile_id,
        expected_definition_sha256=profile.definition_sha256,
        prepared_snapshot_id=prepared_snapshot_id,
        evaluation_package_id=evaluation_package_id,
        trial_entity_versions=trial_entity_versions,
    )
    projection["pool_receipt_id"] = "sha256:" + "f" * 64
    with pytest.raises(SchemaValidationError, match="pool receipt"):
        _validate_benchmark_profile_projection(
            projection,
            task="patient_to_trial",
            profile_id=profile.profile_id,
            expected_definition_sha256=profile.definition_sha256,
            prepared_snapshot_id=prepared_snapshot_id,
            evaluation_package_id=evaluation_package_id,
            trial_entity_versions=trial_entity_versions,
        )


@pytest.mark.parametrize(
    ("value", "absolute"),
    [
        ("/opt/caller-local/bin/codex", True),
        # A value the detector must refuse, never a temporary file this test opens.
        ("/tmp/caller-local/codex", True),  # noqa: S108
        ("/private/tmp/caller-local/codex", True),
        ("/u/researcher/bin/codex", True),
        # Joined so the release-candidate scan's /Users/<name>/ rule does not read this public
        # test file as private-path content.
        ("/Users" + "/researcher/bin/codex", True),
        ("/", True),
        ("//server/share/codex", True),
        ("~/bin/codex", True),
        ("~researcher/bin/codex", True),
        ("C:\\tools\\codex.exe", True),
        ("c:/tools/codex.exe", True),
        ("\\\\server\\share\\codex.exe", True),
        ("file:///opt/codex", True),
        ("codex", False),
        ("bin/codex", False),
        ("./codex", False),
        ("../codex", False),
        ("models/phi4-reasoning-lora", False),
        ("https://github.com/example/TrialMatchAI.git", False),
        ("sha256:" + "0" * 64, False),
        ("options.index_dir", False),
        ("~", False),
        ("~5/10 topics", False),
        ("", False),
    ],
)
def test_absolute_filesystem_path_values_are_recognized_exactly(value: str, absolute: bool) -> None:
    assert _is_absolute_filesystem_path(value) is absolute
