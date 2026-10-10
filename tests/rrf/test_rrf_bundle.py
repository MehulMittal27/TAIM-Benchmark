from __future__ import annotations

# ruff: noqa: S101
from dataclasses import replace
from pathlib import Path

import pytest

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
from taim.schemas import (
    SchemaValidationError,
)

from ..bundle_support import (
    _clean_test_producer as _clean_test_producer,
)
from ..bundle_support import (
    _declaration,
    _fixture_task_source,
    _release_identity,
)


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
        source_snapshot=_fixture_task_source(source),
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
            source_snapshot=_fixture_task_source(tampered),
            release_identity=_release_identity(),
            declaration=_declaration(),
        )
