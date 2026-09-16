from __future__ import annotations

# ruff: noqa: S101
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from taim.evaluation_package import EvaluationPackage
from taim.judgments import TREC_CT_JUDGMENT_SCHEME
from taim.public_patient_to_trial import (
    _evaluate_profile_cutoff,
    _validate_judgment_union_membership,
)
from taim.release_cli import main
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
from taim.snapshot import BenchmarkSnapshot, SourceRecordIdentity, TrialDocument


def test_patient_to_trial_fixture_quick_start(tmp_path: Path, capsys) -> None:
    run_directory = tmp_path / "runs" / "quick-start-patient-to-trial"
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "quick-start-patient-to-trial",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["patient-to-trial", "run", "validate", "--run-dir", str(run_directory)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["system"] == "bm25"
    assert report["profile"] == "synthetic-patient-to-trial"
    assert report["task_input_id"] != report["system_input_id"]

    assert main(["patient-to-trial", "run", "evaluate", "--run-dir", str(run_directory)]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["profile"] == "synthetic-patient-to-trial"
    assert "relevant_or_eligible_precision_at_5" in (metrics["metrics_by_cutoff"]["5"]["aggregate"])


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
        Candidate("run", "bm25", "patient-1", "trial-excluded", 1, 2.0),
        Candidate("run", "bm25", "patient-1", "trial-eligible", 2, 1.0),
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
        "system": "bm25",
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
        "system": "bm25",
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
    unjudged = TrialDocument(
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


def test_trial_to_patient_fixture_quick_start(tmp_path: Path, capsys) -> None:
    run_directory = tmp_path / "runs" / "quick-start-trial-to-patient"
    assert (
        main(
            [
                "trial-to-patient",
                "fixture",
                "run",
                "--run-id",
                "quick-start-trial-to-patient",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["trial-to-patient", "run", "validate", "--run-dir", str(run_directory)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["task"] == "trial_to_patient"
    assert report["profile"] == "synthetic-trial-to-patient"
    assert main(["trial-to-patient", "run", "evaluate", "--run-dir", str(run_directory)]) == 0
    metrics = json.loads(capsys.readouterr().out)
    for field in (
        "release_version",
        "track",
        "task",
        "profile",
        "system",
        "prepared_snapshot_id",
        "task_input_id",
        "system_input_id",
        "evaluation_package_id",
    ):
        assert metrics[field] == report[field]


def test_reverse_evaluate_rejects_a_run_without_release_identity(tmp_path: Path, capsys) -> None:
    run_directory = tmp_path / "runs" / "missing-release-identity"
    assert (
        main(
            [
                "trial-to-patient",
                "fixture",
                "run",
                "--run-id",
                "missing-release-identity",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["configuration"]["release_support"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit):
        main(["trial-to-patient", "run", "evaluate", "--run-dir", str(run_directory)])


def test_direction_specific_run_validation_rejects_extra_and_nested_files(
    tmp_path: Path, capsys
) -> None:
    forward = tmp_path / "runs" / "closed-forward"
    reverse = tmp_path / "runs" / "closed-reverse"
    for task, run_id, top_k in (
        ("patient-to-trial", "closed-forward", "8"),
        ("trial-to-patient", "closed-reverse", "3"),
    ):
        assert (
            main(
                [
                    task,
                    "fixture",
                    "run",
                    "--run-id",
                    run_id,
                    "--output-dir",
                    str(tmp_path / "runs"),
                    "--top-k",
                    top_k,
                ]
            )
            == 0
        )
        capsys.readouterr()
    (reverse / "candidates.jsonl").write_bytes((forward / "candidates.jsonl").read_bytes())
    with pytest.raises(SystemExit):
        main(["trial-to-patient", "run", "validate", "--run-dir", str(reverse)])
    (reverse / "candidates.jsonl").unlink()
    (forward / "nested").mkdir()
    (forward / "nested" / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["patient-to-trial", "run", "validate", "--run-dir", str(forward)])
