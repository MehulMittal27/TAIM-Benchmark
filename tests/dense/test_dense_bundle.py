from __future__ import annotations

# ruff: noqa: S101
import json
from dataclasses import replace
from pathlib import Path

import pytest

from taim.patient_to_trial import (
    PatientToTrialRunRequest,
    PatientToTrialRunResult,
    PatientToTrialTaskInput,
)
from taim.public_patient_to_trial import (
    load_local_evaluation_package,
    load_public_patient_to_trial_run,
    write_public_patient_to_trial_run,
)
from taim.release_cli import main
from taim.release_result_bundle import (
    build_release_result_bundle,
    validate_release_result_bundle,
)
from taim.schemas import PIPELINE_DEPTH_RETRIEVAL, PRIMARY_RANKING_CANDIDATES, Candidate

from ..bundle_support import (
    _clean_test_producer as _clean_test_producer,
)
from ..bundle_support import (
    _declaration,
    _fixture_task_source,
    _release_identity,
)


def test_dense_bundle_uses_portable_input_identity_without_running_inference(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    def synthetic_dense_run(
        system_id: str, request: PatientToTrialRunRequest
    ) -> PatientToTrialRunResult:
        return PatientToTrialRunResult(
            candidates=tuple(
                Candidate(request.run_id, system_id, topic.topic_id, trial.trial_id, rank, 1 / rank)
                for topic in request.snapshot.topics
                for rank, trial in enumerate(request.snapshot.trials[: request.top_k], 1)
            ),
            configuration={
                "model_identity": {"synthetic": "portable-dense-bundle"},
                "index_identity": {"synthetic": "portable-dense-bundle"},
                "metric_cutoff": request.metric_cutoff,
                "system_input": request.system_input_record_dict(),
            },
            runtime_seconds=0,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth=PIPELINE_DEPTH_RETRIEVAL,
        )

    monkeypatch.setattr("taim.pipelines.dense.release.run_dense_system", synthetic_dense_run)
    source_directory = tmp_path / "runs" / "dense-projection-source"
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--system",
                "dense-bge-m3",
                "--index-dir",
                str(tmp_path / "synthetic-index"),
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
    from taim.entity_versions import PatientEvidenceProfile
    from taim.release_fixture import load_release_fixture
    from taim.release_profiles import resolve_benchmark_profile

    task_input = stored.manifest.resolve_task_input(
        PatientToTrialTaskInput.from_benchmark(
            resolve_benchmark_profile(
                load_release_fixture(stored.manifest.benchmark_lineage),
                profile_id=stored.manifest.benchmark_profile["profile_id"],
            ),
            clinical_as_of=stored.manifest.clinical_as_of,
            patient_evidence_profile=PatientEvidenceProfile.from_dict(
                stored.manifest.patient_evidence_profile
            ),
        )
    )
    request = PatientToTrialRunRequest(
        snapshot=task_input,
        run_id=stored.manifest.run_id,
        top_k=stored.manifest.budget_k,
        metric_cutoff=embedded["metric_cutoff"],
        options={
            "device": "cpu",
            "index_dir": tmp_path / "private-dense-index",
            "query_batch_size": 32,
        },
    )
    configuration = dict(stored.manifest.configuration)
    release_support = dict(configuration["release_support"])
    release_support["system"] = "dense-bge-m3"
    configuration.update(
        {
            "system_input": request.system_input_record_dict(),
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
        source_snapshot=_fixture_task_source(dense_directory),
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
