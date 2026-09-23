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
    load_profile,
)
from taim.release_result_bundle import (
    _expected_pipeline_depth,
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

from .bundle_support import (
    _clean_test_producer as _clean_test_producer,
)
from .bundle_support import (
    _declaration,
    _dependency_environment,
    _rehash_bundle,
    _release_identity,
)


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
