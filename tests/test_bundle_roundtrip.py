from __future__ import annotations

# ruff: noqa: S101
import json
from dataclasses import replace
from pathlib import Path

import pytest

from taim.contracts import content_sha256
from taim.evaluation_package import EvaluationPackage
from taim.file_hash import sha256_file
from taim.judgments import TREC_CT_JUDGMENT_SCHEME, JudgmentScheme
from taim.public_patient_to_trial import (
    load_local_evaluation_package,
    load_public_patient_to_trial_run,
    write_public_patient_to_trial_run,
)
from taim.release_cli import main
from taim.release_profiles import (
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
    judgment_union_pool_receipt,
    load_profile,
    ordered_trial_ids_sha256,
)
from taim.release_result_bundle import (
    _expected_pipeline_depth,
    _is_absolute_filesystem_path,
    _validate_benchmark_profile_projection,
    build_release_result_bundle,
    validate_release_result_bundle,
)
from taim.schemas import (
    PIPELINE_DEPTH_RETRIEVAL,
    Candidate,
    RelevanceJudgment,
    SchemaValidationError,
    StageRanking,
)
from taim.system_contracts import SystemBenchmarkSnapshot, SystemRunRequest


def _release_identity() -> dict[str, str]:
    return {
        "release_manifest_id": "sha256:" + "1" * 64,
        "public_tree_id": "sha256:" + "2" * 64,
        "package_artifact_id": "sha256:" + "3" * 64,
    }


def _dependency_environment() -> dict[str, object]:
    core = {
        "schema_version": "1.0",
        "dependency_lock_sha256": "sha256:" + "4" * 64,
        "python_version": "3.13.7",
        "python_implementation": "CPython",
        "platform": "darwin",
        "machine": "arm64",
        "distributions": [{"name": "taim", "version": "0.1.0"}],
    }
    return {**core, "environment_id": content_sha256(core)}


@pytest.fixture(autouse=True)
def _clean_test_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("taim.release_cli._git_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(
        "taim.release_cli._verified_release_identity", lambda _args: _release_identity()
    )
    monkeypatch.setattr(
        "taim.release_cli._verified_dependency_environment",
        lambda _args, *, system_id: _dependency_environment(),
    )


def _declaration() -> dict[str, object]:
    return {
        "analysis_plan_id": "sha256:" + "4" * 64,
        "subset_rule": "all packaged synthetic fixture records",
        "metrics": ["ndcg", "relevant_or_eligible_recall"],
        "inclusion_rule": "all valid synthetic-fixture runs from this CI job",
        "score_guided_selection": False,
        "protocol_approval_id": None,
    }


@pytest.mark.parametrize(
    ("system", "pipeline_depth"),
    [
        ("bm25", "retrieval"),
    ],
)
def test_result_bundle_pipeline_depth_is_system_specific(
    system: str,
    pipeline_depth: str,
) -> None:
    assert _expected_pipeline_depth(system) == pipeline_depth


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


def _rehash_bundle(bundle: Path) -> None:
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        path = bundle / entry["path"]
        entry["sha256"] = sha256_file(path)
        entry["byte_size"] = path.stat().st_size
    core = {key: value for key, value in manifest.items() if key != "bundle_id"}
    manifest["bundle_id"] = content_sha256(core)
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = ("bundle.json", "candidates.jsonl", "metrics.json", "run.json")
    (bundle / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(bundle / name).removeprefix('sha256:')}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )


def _forward_bundle(tmp_path: Path, capsys, *, name: str) -> Path:
    run_directory = tmp_path / "runs" / name
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                name,
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    return build_release_result_bundle(
        run_directory,
        tmp_path / f"{name}-bundle",
        release_identity=_release_identity(),
        declaration=_declaration(),
    ).directory


def test_forward_bundle_preserves_complete_stage_rankings(tmp_path: Path, capsys) -> None:
    source = tmp_path / "runs" / "stage-source"
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "stage-source",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    stored = load_public_patient_to_trial_run(source)
    staged_source = tmp_path / "runs" / "stage-preserved"
    write_public_patient_to_trial_run(
        staged_source,
        manifest=replace(
            stored.manifest,
            candidates_sha256=None,
            stage_rankings=(
                StageRanking(
                    "retrieval-preview",
                    PIPELINE_DEPTH_RETRIEVAL,
                    stored.candidates,
                ),
            ),
        ),
        candidates=stored.candidates,
        evaluation_package=load_local_evaluation_package(source),
    )

    bundle = build_release_result_bundle(
        staged_source,
        tmp_path / "stage-bundle",
        release_identity=_release_identity(),
        declaration=_declaration(),
    )
    stage_path = bundle.directory / "stage-retrieval-preview.jsonl"
    projection = json.loads((bundle.directory / "run.json").read_text(encoding="utf-8"))
    assert stage_path.read_bytes() == (staged_source / "stage-retrieval-preview.jsonl").read_bytes()
    assert projection["stage_rankings"] == [
        {
            "artifact_hash": sha256_file(stage_path),
            "name": "retrieval-preview",
            "pipeline_depth": PIPELINE_DEPTH_RETRIEVAL,
        }
    ]
    assert validate_release_result_bundle(bundle.directory)["task"] == "patient_to_trial"


def _synthetic_dense_component(source: Path, destination: Path) -> None:
    stored = load_public_patient_to_trial_run(source)
    configuration = dict(stored.manifest.configuration)
    support = dict(configuration["release_support"])
    support["system"] = "dense-bge-m3"
    configuration.update(
        {
            "release_support": support,
            "model_identity": {"synthetic": "dense-conformance-only"},
            "index_identity": {"synthetic": "dense-conformance-only"},
        }
    )
    rows = tuple(
        replace(row, run_id="dense-component", system_id="dense-bge-m3")
        for row in stored.candidates
    )
    write_public_patient_to_trial_run(
        destination,
        manifest=replace(
            stored.manifest,
            run_id="dense-component",
            system_id="dense-bge-m3",
            configuration=configuration,
            candidates_sha256=None,
        ),
        candidates=rows,
        evaluation_package=load_local_evaluation_package(source),
    )


def _rrf_run(tmp_path: Path, capsys) -> Path:
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "bm25-component",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "1000",
            ]
        )
        == 0
    )
    capsys.readouterr()
    bm25 = tmp_path / "runs" / "bm25-component"
    dense = tmp_path / "runs" / "dense-component"
    _synthetic_dense_component(bm25, dense)
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--system",
                "rrf",
                "--run-id",
                "rrf-source",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "8",
                "--bm25-run",
                str(bm25),
                "--dense-run",
                str(dense),
            ]
        )
        == 0
    )
    capsys.readouterr()
    return tmp_path / "runs" / "rrf-source"


def test_rrf_bundle_revalidates_component_producers(tmp_path: Path, capsys) -> None:
    source = _rrf_run(tmp_path, capsys)
    bundle = build_release_result_bundle(
        source,
        tmp_path / "rrf-bundle",
        release_identity=_release_identity(),
        declaration=_declaration(),
    )
    assert validate_release_result_bundle(bundle.directory)["system"] == "rrf"

    stored = load_public_patient_to_trial_run(source)
    configuration = dict(stored.manifest.configuration)
    index_identity = dict(configuration["index_identity"])
    component_producers = dict(index_identity["component_producer_identities"])
    bm25_producer = dict(component_producers["bm25"])
    bm25_producer["working_tree_dirty"] = True
    component_producers["bm25"] = bm25_producer
    index_identity["component_producer_identities"] = component_producers
    configuration["index_identity"] = index_identity
    tampered = tmp_path / "runs" / "rrf-dirty-component"
    write_public_patient_to_trial_run(
        tampered,
        manifest=replace(
            stored.manifest,
            run_id="rrf-dirty-component",
            configuration=configuration,
            candidates_sha256=None,
        ),
        candidates=tuple(replace(row, run_id="rrf-dirty-component") for row in stored.candidates),
        evaluation_package=load_local_evaluation_package(source),
    )
    with pytest.raises(SchemaValidationError, match="clean checkout"):
        build_release_result_bundle(
            tampered,
            tmp_path / "rrf-dirty-bundle",
            release_identity=_release_identity(),
            declaration=_declaration(),
        )


def test_direction_specific_run_projects_to_a_safe_bundle(tmp_path: Path, capsys) -> None:
    run_directory = tmp_path / "runs" / "bundle-source"
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "bundle-source",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    bundle = build_release_result_bundle(
        run_directory,
        tmp_path / "bundle",
        release_identity=_release_identity(),
        declaration=_declaration(),
    )
    report = validate_release_result_bundle(bundle.directory)
    assert report["bundle_id"] == bundle.bundle_id
    assert report["task"] == "patient_to_trial"
    assert report["system"] == "bm25"
    projection = json.loads((bundle.directory / "run.json").read_text(encoding="utf-8"))
    system_input = projection["system_input_projection"]
    assert system_input["system_input_id"] == projection["system_input_id"]
    assert system_input["identity_version"] == "2.0"
    assert system_input["effective_options"] == {
        "b": {"value": "0.75", "value_type": "binary64"},
        "k1": {"value": "1.2", "value_type": "binary64"},
    }
    assert not any("directory" in field for field in system_input["index_identity"])
    assert projection["patient_entity_versions"]
    assert projection["trial_entity_versions"]
    assert projection["producer_identity"]["release_identity"] == _release_identity()
    assert projection["producer_identity"]["dependency_environment"] == _dependency_environment()
    assert main(["bundle", "validate", "--bundle-dir", str(bundle.directory)]) == 0


def test_forward_run_rejects_incomplete_declared_budget(tmp_path: Path, capsys) -> None:
    source = tmp_path / "runs" / "complete-source"
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "complete-source",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    stored = load_public_patient_to_trial_run(source)
    with pytest.raises(SchemaValidationError, match="incomplete at the declared budget"):
        write_public_patient_to_trial_run(
            tmp_path / "runs" / "incomplete",
            manifest=replace(stored.manifest, candidates_sha256=None),
            candidates=stored.candidates[:-1],
            evaluation_package=load_local_evaluation_package(source),
        )


def test_forward_run_rejects_judgment_outside_task_input(tmp_path: Path, capsys) -> None:
    source = tmp_path / "runs" / "judgment-source"
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "judgment-source",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    stored = load_public_patient_to_trial_run(source)
    original = load_local_evaluation_package(source)
    foreign = RelevanceJudgment(
        topic_id="OUTSIDE-TASK-INPUT",
        trial_id=stored.manifest.trial_corpus_versions[0][0],
        label=2,
    )
    changed = EvaluationPackage(
        benchmark_lineage=original.benchmark_lineage,
        task_id=original.task_id,
        snapshot_id=original.snapshot_id,
        judgments=tuple(
            sorted(
                (*original.judgments, foreign),
                key=lambda judgment: (judgment.topic_id, judgment.trial_id),
            )
        ),
        provenance=original.provenance,
        judgment_scheme=original.judgment_scheme,
    )
    with pytest.raises(SchemaValidationError, match="Judgment outside the Task Input"):
        write_public_patient_to_trial_run(
            tmp_path / "runs" / "foreign-judgment",
            manifest=replace(
                stored.manifest,
                evaluation_package_id=changed.evaluation_package_id,
                candidates_sha256=None,
            ),
            candidates=stored.candidates,
            evaluation_package=changed,
        )


def test_dense_bundle_uses_portable_input_identity_without_running_inference(
    tmp_path: Path, capsys
) -> None:
    source_directory = tmp_path / "runs" / "dense-projection-source"
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "dense-projection-source",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    stored = load_public_patient_to_trial_run(source_directory)
    embedded = stored.manifest.configuration["system_input"]
    task_fields = {
        "snapshot_contract_version",
        "task",
        "benchmark_lineage",
        "prepared_snapshot_id",
        "clinical_as_of",
        "patient_evidence_profile",
        "query_patient_versions",
        "trial_corpus_versions",
        "available_capabilities",
        "derived_views",
        "task_input_id",
    }
    task_input = SystemBenchmarkSnapshot.from_dict(
        {field: embedded[field] for field in task_fields}
    )
    request = SystemRunRequest(
        snapshot=task_input,
        run_id=stored.manifest.run_id,
        top_k=stored.manifest.budget_k,
        metric_cutoff=embedded["metric_cutoff"],
        options={
            "device": "cpu",
            "index_dir": tmp_path / "private-dense-index",
            "query_batch_size": 32,
        },
        identity_version="2.0",
    )
    configuration = dict(stored.manifest.configuration)
    release_support = dict(configuration["release_support"])
    release_support["system"] = "dense-bge-m3"
    configuration.update(
        {
            "system_input": request.system_input_dict(),
            "release_support": release_support,
            "model_identity": {
                "model_id": "BAAI/bge-m3",
                "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            },
            "index_identity": {
                "corpus_hash": stored.manifest.prepared_snapshot_id,
                "directory": str(tmp_path / "private-dense-index"),
            },
        }
    )
    manifest = replace(
        stored.manifest,
        system_id="dense-bge-m3",
        system_input_id=request.system_input_id,
        configuration=configuration,
    )
    candidates = tuple(replace(row, system_id="dense-bge-m3") for row in stored.candidates)
    dense_directory = tmp_path / "runs" / "dense-projection"
    write_public_patient_to_trial_run(
        dense_directory,
        manifest=manifest,
        candidates=candidates,
        evaluation_package=load_local_evaluation_package(source_directory),
    )

    bundle = build_release_result_bundle(
        dense_directory,
        tmp_path / "dense-bundle",
        release_identity=_release_identity(),
        declaration=_declaration(),
    )
    validate_release_result_bundle(bundle.directory)
    projection = json.loads((bundle.directory / "run.json").read_text(encoding="utf-8"))
    system_input = projection["system_input_projection"]
    assert system_input["identity_version"] == "2.0"
    assert "index_dir" not in system_input["effective_options"]
    assert "options.index_dir" in system_input["omitted_local_fields"]
    assert "index_identity.directory" in system_input["omitted_local_fields"]


def test_reverse_run_projects_to_a_separate_safe_bundle(tmp_path: Path, capsys) -> None:
    run_directory = tmp_path / "runs" / "reverse-bundle-source"
    assert (
        main(
            [
                "trial-to-patient",
                "fixture",
                "run",
                "--run-id",
                "reverse-bundle-source",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    bundle = build_release_result_bundle(
        run_directory,
        tmp_path / "reverse-bundle",
        release_identity=_release_identity(),
        declaration=_declaration(),
    )
    report = validate_release_result_bundle(bundle.directory)
    assert report["bundle_id"] == bundle.bundle_id
    assert report["task"] == "trial_to_patient"
    assert report["system"] == "bm25-trial-to-patient"


def test_bundle_rejects_mismatched_or_dirty_producer(tmp_path: Path, capsys) -> None:
    source = tmp_path / "runs" / "producer-source"
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "producer-source",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    wrong_identity = {**_release_identity(), "package_artifact_id": "sha256:" + "9" * 64}
    with pytest.raises(SchemaValidationError, match="does not match its producer"):
        build_release_result_bundle(
            source,
            tmp_path / "wrong-producer-bundle",
            release_identity=wrong_identity,
            declaration=_declaration(),
        )

    stored = load_public_patient_to_trial_run(source)
    configuration = dict(stored.manifest.configuration)
    execution = dict(configuration["execution_provenance"])
    execution["working_tree_dirty"] = True
    configuration["execution_provenance"] = execution
    dirty = tmp_path / "runs" / "dirty-producer"
    write_public_patient_to_trial_run(
        dirty,
        manifest=replace(
            stored.manifest,
            run_id="dirty-producer",
            configuration=configuration,
            candidates_sha256=None,
        ),
        candidates=tuple(replace(row, run_id="dirty-producer") for row in stored.candidates),
        evaluation_package=load_local_evaluation_package(source),
    )
    with pytest.raises(SchemaValidationError, match="clean checkout"):
        build_release_result_bundle(
            dirty,
            tmp_path / "dirty-producer-bundle",
            release_identity=_release_identity(),
            declaration=_declaration(),
        )


def test_bundle_rejects_mixed_direction_ranking(tmp_path: Path, capsys) -> None:
    forward = tmp_path / "runs" / "forward"
    reverse = tmp_path / "runs" / "reverse"
    for task, run_id in (("patient-to-trial", "forward"), ("trial-to-patient", "reverse")):
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
                    "3",
                ]
            )
            == 0
        )
        capsys.readouterr()
    (forward / "candidates.jsonl").write_bytes((reverse / "patient-candidates.jsonl").read_bytes())
    try:
        build_release_result_bundle(
            forward,
            tmp_path / "mixed-bundle",
            release_identity=_release_identity(),
            declaration=_declaration(),
        )
    except ValueError as exc:
        assert any(marker in str(exc) for marker in ("Candidate", "hash", "direction"))
    else:
        raise AssertionError("mixed-direction ranking was accepted")


def test_bundle_rejects_semantically_empty_metrics_even_when_rehashed(
    tmp_path: Path, capsys
) -> None:
    run_directory = tmp_path / "runs" / "metrics-source"
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "metrics-source",
                "--output-dir",
                str(tmp_path / "runs"),
                "--top-k",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    bundle = build_release_result_bundle(
        run_directory,
        tmp_path / "metrics-bundle",
        release_identity=_release_identity(),
        declaration=_declaration(),
    ).directory
    (bundle / "metrics.json").write_text("{}\n", encoding="utf-8")
    _rehash_bundle(bundle)
    with pytest.raises(ValueError, match="metrics"):
        validate_release_result_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_type", "not-a-release-run"),
        ("release_version", "99.0"),
        ("benchmark_profile_definition_sha256", "sha256:" + "f" * 64),
        ("budget_k", "8"),
    ],
)
def test_bundle_rejects_invalid_run_projection_even_when_rehashed(
    tmp_path: Path, capsys, field: str, value: object
) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name=f"projection-{field}")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    manifest["run"][field] = value
    (bundle / "run.json").write_text(
        json.dumps(manifest["run"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)
    with pytest.raises(ValueError):
        validate_release_result_bundle(bundle)


def test_bundle_rejects_non_numeric_metric_even_when_rehashed(tmp_path: Path, capsys) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="invalid-metric")
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    metrics["metrics_by_cutoff"]["5"]["aggregate"]["ndcg_at_5"] = "not-a-number"
    (bundle / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)
    with pytest.raises(ValueError, match="score"):
        validate_release_result_bundle(bundle)


def test_bundle_rejects_finite_aggregate_tampering_even_when_rehashed(
    tmp_path: Path, capsys
) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="aggregate-tamper")
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    original = metrics["metrics_by_cutoff"]["5"]["aggregate"]["ndcg_at_5"]
    metrics["metrics_by_cutoff"]["5"]["aggregate"]["ndcg_at_5"] = 0.0 if original != 0.0 else 0.5
    (bundle / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)
    with pytest.raises(ValueError, match="per-topic"):
        validate_release_result_bundle(bundle)


def test_bundle_rejects_synthetic_run_relabelled_as_real_even_when_rehashed(
    tmp_path: Path, capsys
) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="synthetic-relabel")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    run = manifest["run"]
    run["profile"] = "official-full"
    run["evidence_scope"] = "real_effectiveness"
    run["benchmark_profile_definition_sha256"] = load_profile("official-full").definition_sha256
    (bundle / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)
    with pytest.raises(ValueError, match=r"Source Bundle|source lock|source projection"):
        validate_release_result_bundle(bundle)


def test_bundle_rejects_synthetic_recipe_tampering_even_when_rehashed(
    tmp_path: Path, capsys
) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="tampered-synthetic-recipe")
    run_path = bundle / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["release_source"]["source_recipe_sha256"] = "sha256:" + "f" * 64
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"] = run
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)

    with pytest.raises(SchemaValidationError, match="complete prepared recipe"):
        validate_release_result_bundle(bundle)


def test_bundle_rejects_duplicate_ranking_even_when_rehashed(tmp_path: Path, capsys) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="duplicate-ranking")
    candidate_path = bundle / "candidates.jsonl"
    lines = candidate_path.read_text(encoding="utf-8").splitlines()
    candidate_path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    manifest["run"]["candidates_sha256"] = sha256_file(candidate_path)
    (bundle / "run.json").write_text(
        json.dumps(manifest["run"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)
    with pytest.raises(ValueError, match=r"duplicate|order"):
        validate_release_result_bundle(bundle)


def test_bundle_rejects_incomplete_ranking_even_when_rehashed(tmp_path: Path, capsys) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="incomplete-ranking")
    candidate_path = bundle / "candidates.jsonl"
    lines = candidate_path.read_text(encoding="utf-8").splitlines()[:-1]
    candidate_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    manifest["run"]["candidates_sha256"] = sha256_file(candidate_path)
    (bundle / "run.json").write_text(
        json.dumps(manifest["run"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)
    with pytest.raises(ValueError, match="incomplete at the declared budget"):
        validate_release_result_bundle(bundle)


def test_bundle_rejects_nested_local_path_even_when_rehashed(tmp_path: Path, capsys) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="nested-local-path")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    projection = manifest["run"]["system_input_projection"]
    projection["index_identity"]["nested"] = [{"directory": "/private/secret/index"}]
    projection["projection_id"] = content_sha256(
        {key: value for key, value in projection.items() if key != "projection_id"}
    )
    (bundle / "run.json").write_text(
        json.dumps(manifest["run"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)

    with pytest.raises(ValueError, match="caller-local"):
        validate_release_result_bundle(bundle)


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


def test_bundle_rejects_absolute_path_value_even_when_rehashed(tmp_path: Path, capsys) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="absolute-path-value")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    manifest["admission"]["subset_rule"] = "/opt/caller-local/subset.txt"
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)

    with pytest.raises(
        SchemaValidationError,
        match=(
            r"\AResult Bundle publishes an absolute filesystem path at: "
            r"bundle\.json admission\.subset_rule\Z"
        ),
    ):
        validate_release_result_bundle(bundle)


def test_bundle_rejects_substituted_system_input_even_when_rehashed(tmp_path: Path, capsys) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="substituted-system-input")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    projection = manifest["run"]["system_input_projection"]
    projection["effective_options"]["k1"]["value"] = "9.9"
    projection["projection_id"] = content_sha256(
        {key: value for key, value in projection.items() if key != "projection_id"}
    )
    (bundle / "run.json").write_text(
        json.dumps(manifest["run"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)

    with pytest.raises(ValueError, match="does not match system_input_id"):
        validate_release_result_bundle(bundle)


def test_bundle_rejects_substituted_judgment_scheme_even_when_rehashed(
    tmp_path: Path, capsys
) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="substituted-judgment-scheme")
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    substitute = JudgmentScheme(
        scheme_id="substituted-trec-ct-eligibility",
        scheme_version="1.0",
        labels=TREC_CT_JUDGMENT_SCHEME.labels,
        gains=TREC_CT_JUDGMENT_SCHEME.gains,
        relevance_sets=TREC_CT_JUDGMENT_SCHEME.relevance_sets,
        reciprocal_rank_relevance_set=TREC_CT_JUDGMENT_SCHEME.reciprocal_rank_relevance_set,
    ).to_dict()
    metrics["judgment_scheme"] = substitute
    for cutoff in metrics["metrics_by_cutoff"].values():
        cutoff["judgment_scheme"] = substitute
    (bundle / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)

    with pytest.raises(ValueError, match="Judgment Scheme"):
        validate_release_result_bundle(bundle)


def test_bundle_rejects_ranking_outside_task_input_even_when_rehashed(
    tmp_path: Path, capsys
) -> None:
    bundle = _forward_bundle(tmp_path, capsys, name="unknown-ranking-member")
    candidate_path = bundle / "candidates.jsonl"
    lines = candidate_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["trial_id"] = "UNKNOWN-TRIAL"
    lines[0] = Candidate.from_dict(row).to_json()
    candidate_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    manifest["run"]["candidates_sha256"] = sha256_file(candidate_path)
    manifest["run"]["trial_entity_versions"].append(
        {"trial_id": "UNKNOWN-TRIAL", "trial_version_id": "sha256:" + "9" * 64}
    )
    manifest["run"]["trial_entity_versions"].sort(key=lambda value: value["trial_id"])
    (bundle / "run.json").write_text(
        json.dumps(manifest["run"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_bundle(bundle)

    with pytest.raises(ValueError, match=r"Task Input membership|does not match system_input_id"):
        validate_release_result_bundle(bundle)
