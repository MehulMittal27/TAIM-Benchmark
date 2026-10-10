from __future__ import annotations

# ruff: noqa: S101
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from taim.evaluation_package import EvaluationPackage
from taim.judgments import TREC_CT_JUDGMENT_SCHEME
from taim.public_patient_to_trial import (
    _evaluate_profile_cutoff,
    _validate_judgment_union_membership,
)
from taim.release_fixture import load_release_fixture
from taim.release_profiles import (
    inspect_judgment_union_profile,
    judgment_union_pool_receipt,
    load_profile,
    ordered_trial_ids_sha256,
    resolve_benchmark_profile,
)
from taim.release_result_bundle import _validate_metrics_payload
from taim.release_support import RELEASE_VERSION
from taim.schemas import Candidate, RelevanceJudgment, RunManifest
from taim.snapshot import BenchmarkSnapshot, SourceRecordIdentity, Trial


@pytest.mark.parametrize(
    "profile_id",
    [
        "official-full",
        "trec-ct-2021-external-fidelity-26149",
        "trec-ct-2021-judgment-union",
    ],
)
def test_trec_2021_profiles_report_eligible_precision_without_changing_frozen_precision(
    profile_id: str,
) -> None:
    package = EvaluationPackage(
        benchmark_lineage="trec-ct-2021",
        task_id="patient_to_trial",
        snapshot_id="sha256:" + "1" * 64,
        judgments=(
            RelevanceJudgment("patient-1", "trial-eligible", 2),
            RelevanceJudgment("patient-1", "trial-excluded", 1),
        ),
        provenance={"synthetic_test": True},
        judgment_scheme=TREC_CT_JUDGMENT_SCHEME,
    )
    candidates = (
        Candidate("run", "synthetic-system", "patient-1", "trial-excluded", 1, 2.0),
        Candidate("run", "synthetic-system", "patient-1", "trial-eligible", 2, 1.0),
    )

    metrics = _evaluate_profile_cutoff(
        candidates,
        package,
        load_profile(profile_id),
        topic_ids=("patient-1",),
        cutoff=10,
    )

    aggregate = metrics["aggregate"]
    assert aggregate["relevant_or_eligible_precision_at_10"] == 0.2
    assert aggregate["eligible_precision_at_10"] == 0.1
    assert metrics["supplemental_precision_policy"] == {
        "metric": "eligible_precision",
        "relevance_set": "eligible",
        "relevance_minimum": 2,
        "denominator": "fixed_cutoff",
    }

    profile = load_profile("official-full")
    scorecard = {
        "schema_version": RunManifest.schema_version,
        "task": "patient_to_trial",
        "release_version": RELEASE_VERSION,
        "track": "trec-ct-2021",
        "profile": "official-full",
        "system": "synthetic-system",
        "run_id": "run",
        "prepared_snapshot_id": package.snapshot_id,
        "task_input_id": "sha256:" + "2" * 64,
        "system_input_id": "sha256:" + "3" * 64,
        "evaluation_package_id": package.evaluation_package_id,
        "judgment_scheme": TREC_CT_JUDGMENT_SCHEME.to_dict(),
        "metrics_by_cutoff": {
            str(cutoff): _evaluate_profile_cutoff(
                candidates,
                package,
                profile,
                topic_ids=("patient-1",),
                cutoff=cutoff,
            )
            for cutoff in profile.cutoffs
        },
    }
    projection = {
        "task": "patient_to_trial",
        "run_id": "run",
        "task_input_id": "sha256:" + "2" * 64,
        "evaluation_package_id": package.evaluation_package_id,
        "system": "synthetic-system",
        "profile": "official-full",
        "track": "trec-ct-2021",
        "prepared_snapshot_id": package.snapshot_id,
        "system_input_id": "sha256:" + "3" * 64,
    }
    _validate_metrics_payload(
        scorecard,
        projection=projection,
        declared_metrics=["ndcg", "eligible_precision"],
    )
    tampered = deepcopy(scorecard)
    del tampered["metrics_by_cutoff"]["10"]["supplemental_precision_policy"]
    with pytest.raises(ValueError, match="cutoff scorecard"):
        _validate_metrics_payload(
            tampered,
            projection=projection,
            declared_metrics=["ndcg", "eligible_precision"],
        )


def test_judgment_union_profile_excludes_unjudged_only_trials_and_requires_frozen_pool() -> None:
    base = load_release_fixture("trec-ct-2022")
    unjudged = Trial(
        trial_id="NCT-UNJUDGED",
        source_identity=SourceRecordIdentity("synthetic-test", "NCT-UNJUDGED"),
        canonical_text="Unjudged-only synthetic trial",
    )
    snapshot = BenchmarkSnapshot(
        benchmark_lineage=base.snapshot.benchmark_lineage,
        snapshot_name="synthetic-with-unjudged-only-trial",
        topics=base.snapshot.topics,
        trials=tuple(sorted((*base.snapshot.trials, unjudged), key=lambda item: item.trial_id)),
        available_capabilities=base.snapshot.available_capabilities,
    )
    package = EvaluationPackage(
        benchmark_lineage=base.evaluation_package.benchmark_lineage,
        task_id=base.evaluation_package.task_id,
        snapshot_id=snapshot.snapshot_id,
        judgments=base.evaluation_package.judgments,
        provenance=base.evaluation_package.provenance,
        judgment_scheme=base.evaluation_package.judgment_scheme,
    )
    prepared = replace(base, snapshot=snapshot, evaluation_package=package)
    expected_pool_hash = "sha256:f4ed99e8bee0511b5fe476ae8452b25349cd4d71d80fd2681bf4c396155e4548"

    inspection = inspect_judgment_union_profile(
        prepared,
        profile_id="trec-ct-2022-judgment-union",
    )
    expected_receipt = {
        "selection_policy": "union_of_trial_ids_present_in_the_evaluation_package",
        "claim_scope": "within_pool_ranking_not_full_corpus_retrieval",
        "profile": "trec-ct-2022-judgment-union",
        "profile_definition_sha256": load_profile("trec-ct-2022-judgment-union").definition_sha256,
        "prepared_snapshot_id": snapshot.snapshot_id,
        "evaluation_package_id": package.evaluation_package_id,
        "pool_count": 8,
        "pool_ids_sha256": expected_pool_hash,
    }
    assert inspection == judgment_union_pool_receipt(**expected_receipt)

    resolved = resolve_benchmark_profile(
        prepared,
        profile_id="trec-ct-2022-judgment-union",
        expected_pool_count=8,
        expected_pool_ids_sha256=expected_pool_hash,
        expected_pool_receipt_id=inspection["pool_receipt_id"],
    )

    assert len(resolved.trials) == 8
    assert "NCT-UNJUDGED" not in {trial.trial_id for trial in resolved.trials}
    assert resolved.pool_ids_hash == expected_pool_hash
    assert resolved.evaluation_package.judgments == package.judgments
    assert resolved.manifest_configuration()["corpus_policy"] == "judgment_union"
    assert resolved.manifest_configuration()["pool_receipt_id"] == inspection["pool_receipt_id"]
    assert resolved.manifest_configuration()["source_corpus_sha256"] != resolved.corpus_hash

    with pytest.raises(ValueError, match="requires --pool-count, --pool-ids-sha256"):
        resolve_benchmark_profile(
            prepared,
            profile_id="trec-ct-2022-judgment-union",
        )

    superset_ids = tuple(trial.trial_id for trial in prepared.trials)
    superset_receipt = judgment_union_pool_receipt(
        selection_policy="union_of_trial_ids_present_in_the_evaluation_package",
        claim_scope="within_pool_ranking_not_full_corpus_retrieval",
        profile="trec-ct-2022-judgment-union",
        profile_definition_sha256=load_profile("trec-ct-2022-judgment-union").definition_sha256,
        prepared_snapshot_id=snapshot.snapshot_id,
        evaluation_package_id=package.evaluation_package_id,
        pool_count=len(superset_ids),
        pool_ids_sha256=ordered_trial_ids_sha256(superset_ids),
    )
    superset_manifest = SimpleNamespace(
        benchmark_profile={
            "profile_id": "trec-ct-2022-judgment-union",
            "corpus_policy": "judgment_union",
            "effective_corpus_count": len(superset_ids),
            "pool_ids_sha256": superset_receipt["pool_ids_sha256"],
            "pool_source_sha256": package.evaluation_package_id,
            "pool_receipt_id": superset_receipt["pool_receipt_id"],
        },
        prepared_snapshot_id=snapshot.snapshot_id,
        evaluation_package_id=package.evaluation_package_id,
        trial_corpus_versions=tuple((trial_id, "sha256:" + "7" * 64) for trial_id in superset_ids),
    )
    with pytest.raises(ValueError, match="every and only"):
        _validate_judgment_union_membership(superset_manifest, package)


def test_shared_profile_values_preserve_registry_identities_and_detach_policy_mappings() -> None:
    from taim import profiles, release_profiles

    research = profiles.load_profile("official-full")
    release = release_profiles.load_profile("official-full")
    expected_hash = "sha256:3562ef6c42bb42c298c9b2d04fb1124d6954cfb8b44ca46936efa5b30e3db1b4"
    assert research.definition_sha256 == release.definition_sha256 == expected_hash
    assert research.evaluation_configuration() == release.evaluation_configuration()
    for profile in (research, release):
        supplied_gains = {"0": 0, "1": 1, "2": 2}
        detached = replace(profile, gain_mapping=supplied_gains)
        supplied_gains["2"] = 99
        assert detached.evaluation_configuration()["gain_mapping"] == {"0": 0, "1": 1, "2": 2}
        serialized = detached.evaluation_configuration()
        serialized["gain_mapping"]["2"] = 88
        assert detached.evaluation_configuration()["gain_mapping"] == {"0": 0, "1": 1, "2": 2}
        with pytest.raises(TypeError):
            detached.gain_mapping["2"] = 77


def test_shared_resolved_values_keep_release_receipts_out_of_research_manifests() -> None:
    from taim import profiles, release_profiles

    prepared = load_release_fixture("trec-ct-2021")
    research = profiles.resolve_benchmark_profile(prepared)
    release = release_profiles.resolve_benchmark_profile(prepared)
    research_manifest = research.manifest_configuration()
    release_manifest = release.manifest_configuration()
    assert "pool_receipt_id" not in research_manifest
    assert release_manifest.pop("pool_receipt_id") is None
    assert release_manifest == research_manifest
    assert research.evaluation_package is prepared.evaluation_package
    assert release.evaluation_package is prepared.evaluation_package
    with pytest.raises(ValueError, match="unknown benchmark profile"):
        profiles.load_profile("trec-ct-2021-judgment-union")
    with pytest.raises(ValueError, match="requires --pool-count"):
        release_profiles.resolve_benchmark_profile(
            prepared, profile_id="trec-ct-2021-judgment-union"
        )
