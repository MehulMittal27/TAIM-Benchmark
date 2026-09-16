"""Track-bound projections of the packaged synthetic conformance fixture."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from taim.contracts import SNAPSHOT_CONTRACT_VERSION, content_sha256
from taim.data import PreparedBenchmark, SnapshotPreparationManifest
from taim.data.prepared import validate_prepared_benchmark
from taim.evaluation_package import EvaluationPackage
from taim.fixtures import load_fixture
from taim.schemas import JsonValue
from taim.snapshot import BenchmarkSnapshot
from taim.source import SourceBundle

FIXTURE_CLINICAL_AS_OF = datetime(2021, 4, 27, tzinfo=UTC)


def _fixture_recipe(track: str, raw_definition: dict[str, object]) -> dict[str, JsonValue]:
    dataset_id = f"{track}-synthetic-conformance"
    return cast(
        dict[str, JsonValue],
        {
            **raw_definition,
            "recipe_id": f"{dataset_id}-snapshot-recipe-v1",
            "benchmark_lineage": track,
            "connector": {"name": "taim.release_fixture", "version": "1.0"},
        },
    )


def fixture_preparation_identity(track: str) -> tuple[str, str, str, SourceBundle]:
    """Return the exact packaged recipe and Source Bundle used by a Track fixture."""

    base = load_fixture()
    raw_recipe = base.prepared_manifest["snapshot_source_recipe"]
    if not isinstance(raw_recipe, dict):
        raise ValueError("packaged fixture Snapshot Source Recipe is invalid")
    raw_definition = raw_recipe.get("definition")
    if not isinstance(raw_definition, dict):
        raise ValueError("packaged fixture Snapshot Source Recipe definition is invalid")
    dataset_id = f"{track}-synthetic-conformance"
    recipe = _fixture_recipe(track, raw_definition)
    return dataset_id, cast(str, recipe["recipe_id"]), content_sha256(recipe), base.source_bundle


def load_release_fixture(track: str) -> PreparedBenchmark:
    """Return one internally consistent synthetic Prepared Benchmark for ``track``."""

    from taim.release_support import SUPPORTED_TRACKS

    if track not in SUPPORTED_TRACKS:
        raise ValueError(f"unsupported synthetic fixture Track {track!r}")
    base = load_fixture()
    dataset_id = f"{track}-synthetic-conformance"
    snapshot = BenchmarkSnapshot(
        benchmark_lineage=track,
        snapshot_name=dataset_id,
        topics=base.snapshot.topics,
        trials=base.snapshot.trials,
        available_capabilities=base.snapshot.available_capabilities,
        derived_views=base.snapshot.derived_views,
    )
    evaluation_package = EvaluationPackage(
        benchmark_lineage=track,
        task_id=f"{track}-synthetic-patient-to-trial-conformance",
        snapshot_id=snapshot.snapshot_id,
        judgments=base.evaluation_package.judgments,
        provenance={
            **base.evaluation_package.provenance,
            "conformance_only": True,
            "declared_track": track,
        },
        judgment_scheme=base.evaluation_package.judgment_scheme,
    )
    raw_recipe = base.prepared_manifest["snapshot_source_recipe"]
    if not isinstance(raw_recipe, dict):
        raise ValueError("packaged fixture Snapshot Source Recipe is invalid")
    raw_definition = raw_recipe.get("definition")
    if not isinstance(raw_definition, dict):
        raise ValueError("packaged fixture Snapshot Source Recipe definition is invalid")
    recipe = _fixture_recipe(track, raw_definition)
    recipe_id = cast(str, recipe["recipe_id"])
    preparation = SnapshotPreparationManifest(
        source_recipe_id=recipe_id,
        source_recipe_hash=content_sha256(recipe),
        source_bundle_id=base.source_bundle.identity_for_artifacts(("topics", "trials")),
    )
    manifest: dict[str, object] = {
        "schema_version": SNAPSHOT_CONTRACT_VERSION,
        "manifest_type": "taim-prepared-benchmark",
        "dataset_id": dataset_id,
        "source_bundle": base.source_bundle.to_dict(),
        "snapshot_source_recipe": {
            "recipe_id": recipe_id,
            "sha256": preparation.source_recipe_hash,
            "definition": recipe,
        },
        "snapshot_preparation": preparation.to_dict(),
        "snapshot": snapshot.manifest_dict(),
        "evaluation_package": evaluation_package.manifest_dict(),
        "package_checksums": dict(base.package_checksums),
        "build_provenance": {"connector": "taim.release_fixture", "version": "1.0"},
    }
    return validate_prepared_benchmark(
        PreparedBenchmark(
            dataset_id=dataset_id,
            directory=Path(base.directory),
            snapshot=snapshot,
            evaluation_package=evaluation_package,
            source_bundle=base.source_bundle,
            preparation=preparation,
            diagnostics=base.diagnostics,
            package_checksums=base.package_checksums,
            prepared_manifest=manifest,
            prepared_manifest_hash=content_sha256(manifest),
        )
    )


__all__ = ["FIXTURE_CLINICAL_AS_OF", "fixture_preparation_identity", "load_release_fixture"]
