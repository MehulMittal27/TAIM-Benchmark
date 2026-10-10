# ruff: noqa: S101
"""The concept-store builder, and the subcommand that runs it in the pinned checkout's interpreter.

The builder tests drive the real build over a fixture of invented vocabulary rows with stand-in
embedders and writers, so no model or LanceDB is needed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import types
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

import taim.pipelines.trialmatchai.release as trialmatchai_release
from taim.pipelines.trialmatchai import concept_store
from taim.pipelines.trialmatchai.concept_store import (
    CONCEPT_STORE_MANIFEST_FILENAME,
    ChunkStore,
    ConceptStoreBuildError,
    ConceptStoreBuildSpec,
    DictionarySource,
    JsonChunkStore,
    JsonStoreWriter,
    LanceDBStoreWriter,
    build_concept_store,
)
from taim.pipelines.trialmatchai.release import build_trialmatchai_concept_store
from taim.pipelines.trialmatchai.system import (
    TRIALMATCHAI_CONCEPT_STORE_MANIFEST_FILENAME,
    TRIALMATCHAI_L4_COMMIT,
    TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH,
)

LOCKED_TRIALMATCHAI_COMMIT = "051ef94fba260675fca59b0a6369b654604df8a0"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "trialmatchai-concepts"


class FakeEmbedder:
    def __init__(self, marker: str = "fixture") -> None:
        self.marker = marker
        self.calls: list[int] = []

    def embed_texts(self, texts):
        self.calls.append(len(texts))
        return [
            [float((sum(map(ord, text)) + offset) % 101) for offset in range(4)] for text in texts
        ]

    def fingerprint(self) -> str:
        return f"fake-{self.marker}"


class ChangingDimensionEmbedder(FakeEmbedder):
    def __init__(self, later_dimension: int) -> None:
        super().__init__()
        self.later_dimension = later_dimension

    def embed_texts(self, texts):
        self.calls.append(len(texts))
        dimension = 4 if len(self.calls) == 1 else self.later_dimension
        return [[float(index) for index in range(dimension)] for _ in texts]


class ReadCountingChunkStore(JsonChunkStore):
    def __init__(self) -> None:
        self.read_paths: list[Path] = []

    def read(self, path: Path) -> tuple[list[dict[str, Any]], list[list[float]]]:
        self.read_paths.append(path)
        return super().read(path)


class MetadataOnlyStoreWriter:
    def write(
        self,
        db_path: Path,
        table_name: str,
        chunks: Sequence[Path],
        chunk_store: ChunkStore,
        vector_index: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "fields": [],
            "fts_index_created": True,
            "vector_index_created": True,
            "indexes": {
                "fts": {"column": "fts_text", "type": "FTS"},
                "vector": vector_index,
                "filters": [
                    {"column": "vocabulary_id", "type": "BTREE"},
                    {"column": "domain_id", "type": "BTREE"},
                ],
            },
        }


def test_lancedb_writer_builds_pinned_vector_and_filter_indexes(
    tmp_path: Path, monkeypatch
) -> None:
    calls: dict[str, Any] = {"scalar": []}

    class FakeArrowTable:
        column_names = (
            "concept_id",
            "vocabulary_id",
            "concept_code",
            "concept_name",
            "domain_id",
            "concept_class_id",
            "standard_concept",
            "synonyms",
            "fts_text",
            "embedding",
        )

    class FakeTable:
        def create_fts_index(self, column, *, replace):
            calls["fts"] = (column, replace)

        def create_index(self, **kwargs):
            calls["vector"] = kwargs

        def create_scalar_index(self, column, *, replace, index_type):
            calls["scalar"].append((column, replace, index_type))

        def list_indices(self):
            return [
                types.SimpleNamespace(columns=["fts_text"], index_type="FTS"),
                types.SimpleNamespace(columns=["embedding"], index_type="IvfPq"),
                types.SimpleNamespace(columns=["vocabulary_id"], index_type="BTree"),
                types.SimpleNamespace(columns=["domain_id"], index_type="BTree"),
            ]

    class FakeDatabase:
        def create_table(self, table_name, *, data, mode):
            calls["table"] = (table_name, data, mode)
            return FakeTable()

    monkeypatch.setitem(
        sys.modules,
        "lancedb",
        types.SimpleNamespace(connect=lambda _path: FakeDatabase()),
    )
    monkeypatch.setitem(sys.modules, "pyarrow", types.SimpleNamespace())
    monkeypatch.setattr(
        "taim.pipelines.trialmatchai.concept_store._rows_to_arrow",
        lambda _payload: FakeArrowTable(),
    )

    chunk = tmp_path / "chunk.arrow"
    chunk_store = JsonChunkStore()
    chunk_store.write(chunk, [{"concept_id": "1"}], [[1.0, 0.0, 0.0, 0.0]])
    metadata = LanceDBStoreWriter().write(
        tmp_path / "concepts",
        "concepts",
        [chunk],
        chunk_store,
        {
            "column": "embedding",
            "type": "IVF_PQ",
            "metric": "cosine",
            "num_partitions": 1024,
            "num_sub_vectors": 128,
            "num_bits": 8,
            "max_iterations": 50,
            "sample_rate": 256,
            "train": True,
        },
    )

    assert calls["fts"] == ("fts_text", True)
    assert calls["vector"] == {
        "metric": "cosine",
        "num_partitions": 1024,
        "num_sub_vectors": 128,
        "vector_column_name": "embedding",
        "replace": True,
        "index_type": "IVF_PQ",
        "num_bits": 8,
        "max_iterations": 50,
        "sample_rate": 256,
        "train": True,
    }
    assert calls["scalar"] == [
        ("vocabulary_id", True, "BTREE"),
        ("domain_id", True, "BTREE"),
    ]
    assert metadata["indexes"]["vector"]["type"] == "IVF_PQ"


def _spec(
    tmp_path: Path, *, config: dict | None = None, force: bool = True
) -> ConceptStoreBuildSpec:
    source_dir = tmp_path / "source"
    source_dir.mkdir(exist_ok=True)
    checkout_dir = tmp_path / "trialmatchai-checkout"
    checkout_dir.mkdir(exist_ok=True)
    for name in ("CONCEPT.csv", "CONCEPT_SYNONYM.csv"):
        shutil.copyfile(FIXTURE_ROOT / name, source_dir / name)
    return ConceptStoreBuildSpec(
        concept_csv=source_dir / "CONCEPT.csv",
        synonym_csv=source_dir / "CONCEPT_SYNONYM.csv",
        db_path=tmp_path / "concept-store",
        trialmatchai_root=checkout_dir,
        config=config
        or {
            "embedder": {
                "model_name": "BAAI/bge-m3",
                "revision": "fixture-revision",
                "pooling": "mean",
                "max_length": 512,
                "batch_size": 32,
                "use_fp16": False,
                "normalize": True,
            }
        },
        vocabularies=("SNOMED", "LOINC", "RxNorm", "ICD10CM"),
        chunk_rows=32,
        vector_index_sub_vectors=4,
        checkpoint_dir=tmp_path / "checkpoint",
        trialmatchai_commit=LOCKED_TRIALMATCHAI_COMMIT,
        force=force,
    )


def test_streaming_fixture_matches_upstream_filtering_and_order(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    embedder = FakeEmbedder()

    manifest = build_concept_store(
        spec,
        embedder=embedder,
        chunk_store=JsonChunkStore(),
        store_writer=JsonStoreWriter(),
    )

    assert manifest["store"]["row_count"] == 70
    assert manifest["store"]["indexes"]["vector"] == {
        "column": "embedding",
        "type": "IVF_PQ",
        "metric": "cosine",
        "num_partitions": 1024,
        "num_sub_vectors": 4,
        "num_bits": 8,
        "max_iterations": 50,
        "sample_rate": 256,
        "train": True,
    }
    assert manifest["vocabularies"] == {
        "ICD10CM": 10,
        "LOINC": 20,
        "RxNorm": 20,
        "SNOMED": 20,
    }
    payload = json.loads((spec.db_path / "concepts.json").read_text(encoding="utf-8"))
    assert [row["concept_id"] for row in payload["rows"][:3]] == ["1001", "1002", "1003"]
    assert payload["rows"][0]["synonyms"] == ["Fixture condition one"]
    assert payload["rows"][0]["fts_text"] == "Fixture condition 01 Fixture condition one"
    assert payload["rows"][-1]["concept_id"] == "4010"
    assert embedder.calls == [32, 32, 6]


def test_uninterrupted_build_updates_digest_without_rescanning_chunks(tmp_path: Path) -> None:
    chunk_store = ReadCountingChunkStore()

    manifest = build_concept_store(
        _spec(tmp_path),
        embedder=FakeEmbedder(),
        chunk_store=chunk_store,
        store_writer=MetadataOnlyStoreWriter(),
    )

    assert manifest["concept_state"]["vector_stream_digest"] == (
        "sha256:004dd1f904902976abf43cbada497ce342fcc46cf2888601ad501541b7401d9d"
    )
    assert chunk_store.read_paths == []


def test_resume_reconstructs_digest_from_completed_chunks_once(tmp_path: Path) -> None:
    chunk_store = ReadCountingChunkStore()
    spec = _spec(tmp_path)
    with pytest.raises(ConceptStoreBuildError, match="test stop requested"):
        build_concept_store(
            spec,
            embedder=FakeEmbedder(),
            chunk_store=chunk_store,
            store_writer=MetadataOnlyStoreWriter(),
            stop_after_chunks=2,
        )
    chunk_store.read_paths.clear()

    manifest = build_concept_store(
        replace(spec, force=False),
        embedder=FakeEmbedder(),
        chunk_store=chunk_store,
        store_writer=MetadataOnlyStoreWriter(),
    )

    chunks_dir = spec.checkpoint_dir / "chunks"
    assert chunk_store.read_paths == [
        chunks_dir / "chunk-00000000.arrow",
        chunks_dir / "chunk-00000001.arrow",
    ]
    assert manifest["concept_state"]["vector_stream_digest"] == (
        "sha256:004dd1f904902976abf43cbada497ce342fcc46cf2888601ad501541b7401d9d"
    )


@pytest.mark.parametrize("rejected_dimension", [0, 5])
def test_invalid_stream_dimension_does_not_block_resume(
    tmp_path: Path, rejected_dimension: int
) -> None:
    spec = _spec(tmp_path)
    with pytest.raises(ConceptStoreBuildError, match="dimension"):
        build_concept_store(
            spec,
            embedder=ChangingDimensionEmbedder(rejected_dimension),
            chunk_store=JsonChunkStore(),
            store_writer=MetadataOnlyStoreWriter(),
        )
    chunks_after_failure = sorted(path.name for path in spec.checkpoint_dir.glob("chunks/*.arrow"))

    resumed_embedder = ChangingDimensionEmbedder(4)
    manifest = build_concept_store(
        replace(spec, force=False),
        embedder=resumed_embedder,
        chunk_store=JsonChunkStore(),
        store_writer=MetadataOnlyStoreWriter(),
    )

    assert chunks_after_failure == ["chunk-00000000.arrow"]
    assert resumed_embedder.calls == [32, 6]
    assert manifest["store"]["row_count"] == 70


def test_resume_reuses_completed_chunks_and_rejects_changed_config(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    first_embedder = FakeEmbedder("first")
    with pytest.raises(ConceptStoreBuildError, match="test stop requested"):
        build_concept_store(
            spec,
            embedder=first_embedder,
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
            stop_after_chunks=2,
        )
    assert first_embedder.calls == [32, 32]

    resumed_embedder = FakeEmbedder("first")
    resumed_spec = _spec(tmp_path, force=False)
    manifest = build_concept_store(
        resumed_spec,
        embedder=resumed_embedder,
        chunk_store=JsonChunkStore(),
        store_writer=JsonStoreWriter(),
    )
    assert resumed_embedder.calls == [6]
    assert manifest["store"]["row_count"] == 70

    changed_spec = _spec(
        tmp_path,
        config={
            "embedder": {
                "model_name": "BAAI/bge-m3",
                "revision": "changed-revision",
                "batch_size": 32,
            }
        },
        force=False,
    )
    with pytest.raises(ConceptStoreBuildError, match="checkpoint identity differs"):
        build_concept_store(
            changed_spec,
            embedder=FakeEmbedder("changed"),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
        )


def test_resume_can_preserve_vector_stream_commit_across_index_builder_upgrade(
    tmp_path: Path,
) -> None:
    old_commit = "a" * 40
    current_commit = "b" * 40
    old_spec = replace(_spec(tmp_path), trialmatchai_commit=old_commit)
    build_concept_store(
        old_spec,
        embedder=FakeEmbedder(),
        chunk_store=JsonChunkStore(),
        store_writer=JsonStoreWriter(),
    )
    new_db_path = tmp_path / "indexed-store-v2"

    with pytest.raises(ConceptStoreBuildError, match="checkpoint identity differs"):
        build_concept_store(
            replace(
                old_spec,
                db_path=new_db_path,
                force=False,
                trialmatchai_commit=current_commit,
            ),
            embedder=FakeEmbedder(),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
        )

    manifest = build_concept_store(
        replace(
            old_spec,
            db_path=new_db_path,
            force=False,
            trialmatchai_commit=current_commit,
            checkpoint_trialmatchai_commit=old_commit,
        ),
        embedder=FakeEmbedder(),
        chunk_store=JsonChunkStore(),
        store_writer=JsonStoreWriter(),
    )

    assert manifest["build"]["trialmatchai_commit"] == current_commit
    assert manifest["build"]["checkpoint_trialmatchai_commit"] == old_commit
    assert manifest["build"]["identity"]["trialmatchai_commit"] == old_commit


def test_cross_commit_resume_rejects_incomplete_vector_stream(tmp_path: Path) -> None:
    old_commit = "a" * 40
    current_commit = "b" * 40
    old_spec = replace(_spec(tmp_path), trialmatchai_commit=old_commit)
    with pytest.raises(ConceptStoreBuildError, match="test stop requested"):
        build_concept_store(
            old_spec,
            embedder=FakeEmbedder(),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
            stop_after_chunks=2,
        )

    with pytest.raises(ConceptStoreBuildError, match="complete verified vector stream"):
        build_concept_store(
            replace(
                old_spec,
                force=False,
                trialmatchai_commit=current_commit,
                checkpoint_trialmatchai_commit=old_commit,
            ),
            embedder=FakeEmbedder(),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
        )


def test_cross_commit_resume_rejects_missing_vector_stream(tmp_path: Path) -> None:
    spec = replace(
        _spec(tmp_path),
        force=False,
        trialmatchai_commit="b" * 40,
        checkpoint_trialmatchai_commit="a" * 40,
    )

    with pytest.raises(ConceptStoreBuildError, match="existing complete vector stream"):
        build_concept_store(
            spec,
            embedder=FakeEmbedder(),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
        )


def test_force_rejects_cross_commit_reuse_before_deleting_checkpoint(tmp_path: Path) -> None:
    spec = replace(
        _spec(tmp_path),
        trialmatchai_commit="b" * 40,
        checkpoint_trialmatchai_commit="a" * 40,
    )
    marker = spec.checkpoint_dir / "preserved-vector-stream"
    marker.parent.mkdir(parents=True)
    marker.write_text("must survive", encoding="utf-8")

    with pytest.raises(ConceptStoreBuildError, match="would delete the preserved vector stream"):
        build_concept_store(
            spec,
            embedder=FakeEmbedder(),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
        )

    assert marker.read_text(encoding="utf-8") == "must survive"


def test_resume_rejects_changed_source_hash(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    with pytest.raises(ConceptStoreBuildError, match="test stop requested"):
        build_concept_store(
            spec,
            embedder=FakeEmbedder(),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
            stop_after_chunks=1,
        )
    with spec.synonym_csv.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ConceptStoreBuildError, match="checkpoint identity differs"):
        build_concept_store(
            replace(spec, force=False),
            embedder=FakeEmbedder(),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
        )


def test_resume_rejects_tampered_chunk_digest(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    with pytest.raises(ConceptStoreBuildError, match="test stop requested"):
        build_concept_store(
            spec,
            embedder=FakeEmbedder(),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
            stop_after_chunks=1,
        )
    chunk_path = spec.checkpoint_dir / "chunks" / "chunk-00000000.arrow"
    payload = json.loads(chunk_path.read_text(encoding="utf-8"))
    payload["vectors"][0][0] += 1.0
    chunk_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConceptStoreBuildError, match="vector_stream_digest"):
        build_concept_store(
            replace(spec, force=False),
            embedder=FakeEmbedder(),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
        )


def test_chunk_size_must_preserve_global_embedding_batches(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    invalid = replace(spec, chunk_rows=33)
    with pytest.raises(ConceptStoreBuildError, match="multiple of embedding_batch_size"):
        invalid.validate()


def test_dictionary_rows_are_appended_after_omop_rows(tmp_path: Path) -> None:
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("D1||Dictionary name|Alias\n", encoding="utf-8")
    spec = _spec(tmp_path)
    spec = replace(
        spec,
        dictionary_sources=(DictionarySource("Fixture", "Test", dictionary),),
    )
    manifest = build_concept_store(
        spec,
        embedder=FakeEmbedder(),
        chunk_store=JsonChunkStore(),
        store_writer=JsonStoreWriter(),
    )
    payload = json.loads((spec.db_path / "concepts.json").read_text(encoding="utf-8"))
    assert manifest["store"]["row_count"] == 71
    assert payload["rows"][-1]["concept_id"] == "Fixture:D1"


def test_checkout_at_another_commit_than_expected_is_refused(tmp_path: Path) -> None:
    spec = replace(_spec(tmp_path), expected_trialmatchai_commit="b" * 40)

    with pytest.raises(ConceptStoreBuildError, match="does not match the locked commit"):
        build_concept_store(
            spec,
            embedder=FakeEmbedder(),
            chunk_store=JsonChunkStore(),
            store_writer=JsonStoreWriter(),
        )


def test_builder_writes_the_manifest_the_run_reads() -> None:
    assert CONCEPT_STORE_MANIFEST_FILENAME == TRIALMATCHAI_CONCEPT_STORE_MANIFEST_FILENAME


def test_builder_loads_with_the_standard_library_alone() -> None:
    """The checkout's interpreter has no TAIM installed, so the builder must import none of it.

    ``-S`` drops site-packages: any import beyond the standard library at load time fails here.
    """

    completed = subprocess.run(
        [sys.executable, "-I", "-S", concept_store.__file__, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--expected-trialmatchai-commit" in completed.stdout


# Fixtures are named apart from their functions: the release exclusion scan treats every top-level
# name of a file a release unit drops as a needle, and "checkout" is a word the shipped prose uses.
@pytest.fixture(name="checkout")
def _verified_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A checkout with an environment; its Git identity is taken as verified."""

    root = tmp_path / "TrialMatchAI"
    interpreter = root / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)

    def verified(cwd: Path, command: Sequence[str], **expected: object) -> dict[str, object]:
        assert cwd == root and list(command) == ["--config", TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH]
        assert expected["expected_commit"] == TRIALMATCHAI_L4_COMMIT
        return {}

    monkeypatch.setattr(trialmatchai_release, "trialmatchai_checkout_configuration", verified)
    return root


def _build(checkout: Path, tmp_path: Path, **overrides: Any) -> tuple[int, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def call(argv: list[str], *, cwd: Path, env: dict[str, str]) -> int:
        calls.append({"argv": argv, "cwd": cwd, "env": env})
        return 7

    options: dict[str, Any] = {
        "concept_csv": tmp_path / "vocabulary" / "CONCEPT.csv",
        "synonym_csv": tmp_path / "vocabulary" / "CONCEPT_SYNONYM.csv",
        "source_release": "Invented release 1",
        "db_path": tmp_path / "store",
        "work_dir": tmp_path / "work",
        "call": call,
        **overrides,
    }
    return build_trialmatchai_concept_store(checkout, **options), calls


def test_build_runs_the_qualified_recipe_in_the_checkout_interpreter(
    checkout: Path, tmp_path: Path
) -> None:
    status, [run] = _build(checkout, tmp_path)

    argv = run["argv"]
    assert status == 7
    assert argv[:3] == [
        str(checkout / ".venv" / "bin" / "python"),
        "-I",
        str(Path(concept_store.__file__).resolve()),
    ]
    options = dict(zip(argv[3::2], argv[4::2], strict=True))
    assert options == {
        "--config": str(checkout / TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH),
        "--trialmatchai-root": str(checkout),
        "--expected-trialmatchai-commit": TRIALMATCHAI_L4_COMMIT,
        "--concept-csv": str(tmp_path / "vocabulary" / "CONCEPT.csv"),
        "--synonym-csv": str(tmp_path / "vocabulary" / "CONCEPT_SYNONYM.csv"),
        "--source-release": "Invented release 1",
        "--sources": "open",
        "--concept-cache": str(tmp_path / "work" / "concept_dicts"),
        "--checkpoint-dir": str(tmp_path / "work" / "checkpoint"),
        "--db-path": str(tmp_path / "store"),
    }
    assert run["cwd"] == checkout
    assert run["env"]["TRIALMATCHAI_EMBEDDER_USE_GPU"] == "true"


def test_build_passes_force_through(checkout: Path, tmp_path: Path) -> None:
    _status, [run] = _build(checkout, tmp_path, force=True)

    assert run["argv"][-1] == "--force"


@pytest.mark.parametrize("output", ["db_path", "work_dir"])
def test_build_refuses_an_output_inside_the_checkout(
    checkout: Path, tmp_path: Path, output: str
) -> None:
    with pytest.raises(ValueError, match="must be outside the checkout"):
        _build(checkout, tmp_path, **{output: checkout / "data" / "concepts"})


def test_build_without_the_checkout_environment_names_build_env(
    checkout: Path, tmp_path: Path
) -> None:
    shutil.rmtree(checkout / ".venv")

    with pytest.raises(ValueError, match=r"no environment.*build-env"):
        _build(checkout, tmp_path)


# The outside-build identity: rows by their text, dictionaries by their pinned versions.

INVENTED_ROWS = [
    {
        "concept_id": f"{index}",
        "vocabulary_id": "Invented",
        "concept_code": f"I{index}",
        "concept_name": f"zorvek {index}",
        "domain_id": "Condition",
        "concept_class_id": "Finding",
        "standard_concept": "S",
        "synonyms": [f"lumtra {index}"],
        "fts_text": f"zorvek {index} lumtra {index}",
    }
    for index in range(5)
]


def test_row_digest_ignores_row_order_and_vectors() -> None:
    reordered = [dict(row, embedding=[0.5]) for row in reversed(INVENTED_ROWS)]

    assert concept_store.row_digest(reordered) == concept_store.row_digest(INVENTED_ROWS)
    assert concept_store.row_digest(INVENTED_ROWS)["count"] == 5


@pytest.mark.parametrize("field", concept_store.ROW_DIGEST_FIELDS)
def test_row_digest_binds_every_text_field(field: str) -> None:
    changed = [dict(INVENTED_ROWS[0], **{field: ["changed"]}), *INVENTED_ROWS[1:]]

    assert concept_store.row_digest(changed) != concept_store.row_digest(INVENTED_ROWS)


def test_row_digest_command_loads_with_the_standard_library_alone() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", concept_store.__file__, "row-digest", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--db-path" in completed.stdout


def test_pinned_dictionaries_follow_upstream_source_order() -> None:
    """The order is the stream's row order, so it is the qualified store's dictionary order."""

    pins = concept_store.PINNED_OPEN_DICTIONARIES
    assert [pin.key for pin in pins] == [
        "genes",
        "diseases",
        "chemicals",
        "cell_lines",
        "cell_types",
        "phenotypes",
    ]
    for pin in pins:
        assert (pin.url is None) == (pin.published_name is not None)
        assert pin.url is None or pin.url.startswith("https://")
    archived = {pin.key for pin in pins if pin.url is None}
    assert archived == {"genes", "cell_lines"}


def test_zenodo_file_url_reads_the_record_from_the_doi() -> None:
    url = "https://zenodo.org/api/records/123/files/cellosaurus.obo.gz/content"
    assert concept_store.zenodo_file_url("10.5281/zenodo.123", "cellosaurus.obo.gz") == url
    assert concept_store.zenodo_file_url("https://doi.org/10.5281/zenodo.123", "x") == (
        "https://zenodo.org/api/records/123/files/x/content"
    )
    with pytest.raises(ValueError, match="not a Zenodo record DOI"):
        concept_store.zenodo_file_url("10.1234/other.123", "x")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class InventedSources:
    """Stand-ins for TrialMatchAI's OPEN_SOURCES and write_dictionary over invented files."""

    RAW: ClassVar[dict[str, bytes]] = {
        "remote.obo": b"invented remote ontology\n",
        "archived.txt": b"invented archive\n",
    }

    def __init__(self) -> None:
        self.sources = {
            "remote": types.SimpleNamespace(
                filename="remote.obo",
                dict_filename="dict_Remote.txt",
                vocabulary_id="RemoteVocab",
                domain_id="Remote",
            ),
            "archived": types.SimpleNamespace(
                filename="archived.txt",
                dict_filename="dict_Archived.txt",
                vocabulary_id="ArchivedVocab",
                domain_id="Archived",
            ),
        }

    @staticmethod
    def write_dictionary(source: Any, raw_path: Path, dict_path: Path) -> int:
        dict_path.write_bytes(b"ID:1||" + raw_path.read_bytes())
        return 1

    def pins(self) -> tuple[concept_store.PinnedDictionarySource, ...]:
        archived = gzip.compress(self.RAW["archived.txt"], mtime=0)
        return (
            concept_store.PinnedDictionarySource(
                key="remote",
                version="remote 1",
                filename="remote.obo",
                sha256=_sha256(self.RAW["remote.obo"]),
                dictionary="dictionary:RemoteVocab:dict_Remote.txt",
                dictionary_sha256=_sha256(b"ID:1||" + self.RAW["remote.obo"]),
                url="https://invented.example/remote/1/remote.obo",
            ),
            concept_store.PinnedDictionarySource(
                key="archived",
                version="archived 1",
                filename="archived.txt",
                sha256=_sha256(self.RAW["archived.txt"]),
                dictionary="dictionary:ArchivedVocab:dict_Archived.txt",
                dictionary_sha256=_sha256(b"ID:1||" + self.RAW["archived.txt"]),
                published_name="archived.txt.gz",
                published_sha256=_sha256(archived),
            ),
        )

    def download(self, served: dict[str, bytes]):
        fetched: list[str] = []

        def download(url: str, destination: Path) -> None:
            fetched.append(url)
            destination.write_bytes(served[url])

        return download, fetched


@pytest.fixture(name="invented")
def _invented_sources(monkeypatch: pytest.MonkeyPatch) -> InventedSources:
    sources = InventedSources()
    monkeypatch.setattr(concept_store, "PINNED_OPEN_DICTIONARIES", sources.pins())
    return sources


ARCHIVE_URL = "https://zenodo.org/api/records/7/files/archived.txt.gz/content"


def _served(sources: InventedSources) -> dict[str, bytes]:
    return {
        "https://invented.example/remote/1/remote.obo": sources.RAW["remote.obo"],
        ARCHIVE_URL: gzip.compress(sources.RAW["archived.txt"], mtime=0),
    }


def test_pinned_dictionaries_download_verify_and_parse_in_order(
    invented: InventedSources, tmp_path: Path
) -> None:
    download, fetched = invented.download(_served(invented))

    dictionaries = concept_store.pinned_open_dictionaries(
        tmp_path,
        invented.sources,
        invented.write_dictionary,
        doi="10.5281/zenodo.7",
        download=download,
    )

    assert fetched == ["https://invented.example/remote/1/remote.obo", ARCHIVE_URL]
    assert [(source.vocabulary_id, source.path.name) for source in dictionaries] == [
        ("RemoteVocab", "dict_Remote.txt"),
        ("ArchivedVocab", "dict_Archived.txt"),
    ]
    # The archive is stored compressed; the parser reads the unmodified file.
    assert (tmp_path / "raw" / "archived.txt").read_bytes() == invented.RAW["archived.txt"]
    assert not (tmp_path / "raw" / "archived.txt.gz").exists()


def test_a_verified_raw_file_is_not_downloaded_again(
    invented: InventedSources, tmp_path: Path
) -> None:
    (tmp_path / "raw").mkdir()
    for name, data in invented.RAW.items():
        (tmp_path / "raw" / name).write_bytes(data)
    download, fetched = invented.download({})

    concept_store.pinned_open_dictionaries(
        tmp_path, invented.sources, invented.write_dictionary, doi=None, download=download
    )

    assert fetched == []


def test_an_archived_source_without_a_doi_names_the_file_to_place(
    invented: InventedSources, tmp_path: Path
) -> None:
    download, _fetched = invented.download(_served(invented))

    with pytest.raises(ConceptStoreBuildError, match=r"archived 1.*has no DOI yet; place"):
        concept_store.pinned_open_dictionaries(
            tmp_path, invented.sources, invented.write_dictionary, doi=None, download=download
        )


def test_a_raw_file_at_another_version_is_refused(
    invented: InventedSources, tmp_path: Path
) -> None:
    served = _served(invented)
    served["https://invented.example/remote/1/remote.obo"] = b"invented remote ontology, later\n"
    download, _fetched = invented.download(served)

    with pytest.raises(ConceptStoreBuildError, match=r"remote\.obo is not remote 1"):
        concept_store.pinned_open_dictionaries(
            tmp_path,
            invented.sources,
            invented.write_dictionary,
            doi="10.5281/zenodo.7",
            download=download,
        )


def test_an_archive_that_is_not_the_published_file_is_refused(
    invented: InventedSources, tmp_path: Path
) -> None:
    served = _served(invented)
    served[ARCHIVE_URL] = b"not the archived file"
    download, _fetched = invented.download(served)

    with pytest.raises(ConceptStoreBuildError, match=r"is not the published archived\.txt\.gz"):
        concept_store.pinned_open_dictionaries(
            tmp_path,
            invented.sources,
            invented.write_dictionary,
            doi="10.5281/zenodo.7",
            download=download,
        )


def test_a_parser_that_writes_another_dictionary_is_refused(
    invented: InventedSources, tmp_path: Path
) -> None:
    download, _fetched = invented.download(_served(invented))

    def other_parser(source: Any, raw_path: Path, dict_path: Path) -> int:
        dict_path.write_bytes(b"ID:2||other\n")
        return 1

    with pytest.raises(ConceptStoreBuildError, match=r"did not turn remote\.obo into the pinned"):
        concept_store.pinned_open_dictionaries(
            tmp_path,
            invented.sources,
            other_parser,
            doi="10.5281/zenodo.7",
            download=download,
        )


def test_other_upstream_sources_are_refused(invented: InventedSources, tmp_path: Path) -> None:
    sources = {"remote": invented.sources["remote"]}

    with pytest.raises(ConceptStoreBuildError, match="not the pinned ones"):
        concept_store.pinned_open_dictionaries(
            tmp_path, sources, invented.write_dictionary, doi=None
        )


def test_a_dictionary_source_is_fetched_over_https_only(tmp_path: Path) -> None:
    with pytest.raises(ConceptStoreBuildError, match="over https only"):
        concept_store._download("http://invented.example/remote.obo", tmp_path / "remote.obo")
