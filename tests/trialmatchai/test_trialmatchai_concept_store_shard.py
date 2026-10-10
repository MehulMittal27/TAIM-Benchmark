# ruff: noqa: S101
"""The sharded embedding pass must reproduce the single-GPU chunk stream exactly.

These tests run the real builder and the real shard runner side by side over the committed
fixture with a deterministic, deliberately *batch-sensitive* embedder: every vector depends
on which other texts share its batch of 32 and on its position inside that batch, the way a
real dynamically-padded transformer batch does.  A sharding scheme that quietly changed
chunk boundaries or batch composition would therefore produce different numbers here, not
merely a differently-ordered stream.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from taim.pipelines.trialmatchai.concept_store import (
    ConceptStoreBuildError,
    ConceptStoreBuildSpec,
    JsonChunkStore,
    JsonStoreWriter,
    _make_synonym_database,
    build_concept_store,
    chunk_texts,
    chunk_vector_digest,
    run_shard,
    seal_checkpoint,
)

LOCKED_TRIALMATCHAI_COMMIT = "051ef94fba260675fca59b0a6369b654604df8a0"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "trialmatchai-concepts"
BATCH_SIZE = 32


class BatchSensitiveEmbedder:
    """Deterministic stand-in whose output depends on batch composition.

    A real embedder pads each batch of 32 to the longest member, so a text's vector is a
    function of its batch, not only of itself.  This reproduces that coupling cheaply: the
    batch's longest text and the row's position inside the batch both feed the value.  It
    makes the tests sensitive to exactly the mistake sharding could introduce.
    """

    def __init__(self, batch_size: int = BATCH_SIZE) -> None:
        self.batch_size = batch_size
        self.batch_sizes: list[int] = []

    def embed_texts(self, texts):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            self.batch_sizes.append(len(batch))
            pad = max(len(text) for text in batch)
            for position, text in enumerate(batch):
                seed = sum(map(ord, text))
                vectors.append(
                    [float((seed + pad * 7 + position * 13 + offset) % 1009) for offset in range(4)]
                )
        return vectors

    def fingerprint(self) -> str:
        return "batch-sensitive-fixture"


def _spec(
    tmp_path: Path, *, chunk_rows: int = BATCH_SIZE, name: str = "ref"
) -> ConceptStoreBuildSpec:
    source_dir = tmp_path / f"{name}-source"
    source_dir.mkdir(parents=True, exist_ok=True)
    checkout_dir = tmp_path / "trialmatchai-checkout"
    checkout_dir.mkdir(exist_ok=True)
    for filename in ("CONCEPT.csv", "CONCEPT_SYNONYM.csv"):
        shutil.copyfile(FIXTURE_ROOT / filename, source_dir / filename)
    return ConceptStoreBuildSpec(
        concept_csv=source_dir / "CONCEPT.csv",
        synonym_csv=source_dir / "CONCEPT_SYNONYM.csv",
        db_path=tmp_path / f"{name}-store",
        trialmatchai_root=checkout_dir,
        config={
            "embedder": {
                "model_name": "BAAI/bge-m3",
                "revision": "fixture-revision",
                "pooling": "mean",
                "max_length": 512,
                "batch_size": BATCH_SIZE,
                "use_fp16": False,
                "normalize": True,
            }
        },
        vocabularies=("SNOMED", "LOINC", "RxNorm", "ICD10CM"),
        chunk_rows=chunk_rows,
        vector_index_sub_vectors=4,
        checkpoint_dir=tmp_path / f"{name}-checkpoint",
        trialmatchai_commit=LOCKED_TRIALMATCHAI_COMMIT,
        force=True,
    )


def _reference_chunks(tmp_path: Path, chunk_rows: int) -> dict[int, tuple[list, list]]:
    """Run the untouched single-GPU builder and return its chunk stream."""
    spec = _spec(tmp_path, chunk_rows=chunk_rows, name="ref")
    chunk_store = JsonChunkStore()
    build_concept_store(
        spec,
        embedder=BatchSensitiveEmbedder(),
        chunk_store=chunk_store,
        store_writer=JsonStoreWriter(),
    )
    chunks = sorted((spec.checkpoint_dir / "chunks").glob("chunk-*.arrow"))
    return {int(path.stem.split("-")[1]): chunk_store.read(path) for path in chunks}


def _shard_chunks(
    tmp_path: Path, chunk_rows: int, world_size: int
) -> tuple[dict[int, tuple[list, list]], list[dict]]:
    """Run every rank of a `world_size` shard and return the merged chunk stream."""
    chunk_dir = tmp_path / f"shard-{world_size}" / "chunks"
    chunk_store = JsonChunkStore()
    summaries = []
    for rank in range(world_size):
        spec = _spec(tmp_path, chunk_rows=chunk_rows, name=f"shard{world_size}r{rank}")
        synonyms = tmp_path / f"shard{world_size}r{rank}-synonyms.sqlite"
        connection = _make_synonym_database(synonyms, spec.synonym_csv)
        summaries.append(
            run_shard(
                spec,
                connection,
                embedder=BatchSensitiveEmbedder(),
                chunk_store=chunk_store,
                chunk_dir=chunk_dir,
                rank=rank,
                world_size=world_size,
            )
        )
        connection.close()
    chunks = sorted(chunk_dir.glob("chunk-*.arrow"))
    merged = {int(path.stem.split("-")[1]): chunk_store.read(path) for path in chunks}
    return merged, summaries


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 5])
def test_sharded_stream_is_bit_identical_to_the_single_gpu_stream(
    tmp_path: Path, world_size: int
) -> None:
    reference = _reference_chunks(tmp_path, BATCH_SIZE)
    candidate, summaries = _shard_chunks(tmp_path, BATCH_SIZE, world_size)

    assert set(candidate) == set(reference)
    assert sorted(reference) == list(range(len(reference)))
    for index in sorted(reference):
        reference_rows, reference_vectors = reference[index]
        candidate_rows, candidate_vectors = candidate[index]
        assert [row["concept_id"] for row in candidate_rows] == [
            row["concept_id"] for row in reference_rows
        ], f"concept_id drift in chunk {index}"
        assert candidate_vectors == reference_vectors, f"vector drift in chunk {index}"
        assert chunk_vector_digest(candidate_rows, candidate_vectors) == chunk_vector_digest(
            reference_rows, reference_vectors
        )

    total_rows = sum(len(rows) for rows, _ in reference.values())
    assert sum(summary["owned_rows"] for summary in summaries) == total_rows
    assert sum(summary["owned_chunks"] for summary in summaries) == len(reference)
    # Every rank streams the whole row iterator, so all of them must agree on what that
    # stream was.  Disagreement here is the off-by-one shard boundary, before any vector.
    assert {summary["row_order_identity"] for summary in summaries} == {
        summaries[0]["row_order_identity"]
    }
    assert {summary["streamed_rows"] for summary in summaries} == {total_rows}


def test_chunk_ownership_partitions_the_stream_exactly_once(tmp_path: Path) -> None:
    reference = _reference_chunks(tmp_path, BATCH_SIZE)
    for world_size in (2, 3, 4):
        owners = [index % world_size for index in sorted(reference)]
        for rank in range(world_size):
            expected = [index for index in sorted(reference) if index % world_size == rank]
            assert expected == [
                index
                for index, owner in zip(sorted(reference), owners, strict=True)
                if owner == rank
            ]
        assert sorted(
            index
            for rank in range(world_size)
            for index in sorted(reference)
            if index % world_size == rank
        ) == sorted(reference)


def test_stand_in_embedder_really_depends_on_batch_composition(tmp_path: Path) -> None:
    """Negative control for the stand-in: batch composition changes its output.

    Without this, an identical-stream result above could just mean the stand-in ignores
    batching, and the comparison would prove nothing.  Note what it also shows: two texts
    keep their vectors when they stay in the same batch of 32 and lose them when they do
    not - which is exactly why the sharding contract is stated in whole chunks.
    """
    embedder = BatchSensitiveEmbedder()
    texts = [f"concept {index} " + "x" * (index % 17) for index in range(64)]

    aligned = embedder.embed_texts(texts)
    first_batch_only = embedder.embed_texts(texts[:32])
    misaligned = embedder.embed_texts(texts[16:48])

    # Same batch of 32 -> same vectors, whoever runs it.
    assert first_batch_only == aligned[:32]
    # Shifted by half a batch -> every shared text gets a different vector.
    assert all(misaligned[position] != aligned[16 + position] for position in range(32))


def test_a_one_row_drift_changes_every_downstream_chunk_digest(tmp_path: Path) -> None:
    """The failure this whole exercise exists to catch, shown failing.

    A stream that slips one row still produces perfectly plausible vectors; only the
    (concept_id, vector) pairing is wrong.  The per-chunk digest must move when that
    happens, otherwise a silent misalignment would survive the comparison.
    """
    reference = _reference_chunks(tmp_path, BATCH_SIZE)
    flat_rows = [row for index in sorted(reference) for row in reference[index][0]]
    flat_vectors = [vector for index in sorted(reference) for vector in reference[index][1]]

    drifted_rows = flat_rows[1:]
    drifted_vectors = flat_vectors[:-1]
    assert len(drifted_rows) == len(drifted_vectors)

    honest = chunk_vector_digest(flat_rows[1:], flat_vectors[1:])
    drifted = chunk_vector_digest(drifted_rows, drifted_vectors)
    assert honest != drifted


def test_shard_resumes_from_its_existing_chunks_without_re_embedding(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "resume" / "chunks"
    chunk_store = JsonChunkStore()
    spec = _spec(tmp_path, name="resume")
    synonyms = tmp_path / "resume-synonyms.sqlite"

    connection = _make_synonym_database(synonyms, spec.synonym_csv)
    first = run_shard(
        spec,
        connection,
        embedder=BatchSensitiveEmbedder(),
        chunk_store=chunk_store,
        chunk_dir=chunk_dir,
        rank=0,
        world_size=2,
    )
    connection.close()

    class ExplodingEmbedder:
        def embed_texts(self, texts):  # pragma: no cover - must never be called
            raise AssertionError("a resumed shard must not re-embed a completed chunk")

    connection = sqlite3.connect(f"file:{synonyms}?mode=ro", uri=True)
    second = run_shard(
        spec,
        connection,
        embedder=ExplodingEmbedder(),
        chunk_store=chunk_store,
        chunk_dir=chunk_dir,
        rank=0,
        world_size=2,
    )
    connection.close()

    assert second["owned_chunks"] == first["owned_chunks"]
    assert second["reused_chunks"] == first["owned_chunks"]
    assert second["owned_rows"] == first["owned_rows"]


def test_shard_rejects_an_invalid_rank(tmp_path: Path) -> None:
    spec = _spec(tmp_path, name="badrank")
    synonyms = tmp_path / "badrank-synonyms.sqlite"
    connection = _make_synonym_database(synonyms, spec.synonym_csv)
    try:
        for rank, world_size in ((2, 2), (-1, 2), (0, 0)):
            with pytest.raises(ConceptStoreBuildError):
                run_shard(
                    spec,
                    connection,
                    embedder=BatchSensitiveEmbedder(),
                    chunk_store=JsonChunkStore(),
                    chunk_dir=tmp_path / "badrank-chunks",
                    rank=rank,
                    world_size=world_size,
                )
    finally:
        connection.close()


def test_chunk_texts_matches_the_builder_embedding_text(tmp_path: Path) -> None:
    """The shard must build the same string the builder embeds, character for character."""
    rows = [
        {"concept_name": "Asthma", "synonyms": ["Bronchial asthma", "Reactive airway"]},
        {"concept_name": "Sepsis", "synonyms": []},
    ]
    builder_texts = [
        " | ".join([str(row["concept_name"]), *row["synonyms"]]).strip() for row in rows
    ]
    assert chunk_texts(rows) == builder_texts


class ExplodingEmbedder:
    """Answers the identity contract but refuses to embed. Any embedding call is the bug."""

    def embed_texts(self, texts):  # pragma: no cover - must never be called
        raise AssertionError("a sealed chunk stream must not be re-embedded")

    def fingerprint(self) -> str:
        return "batch-sensitive-fixture"


def _seal_a_sharded_stream(tmp_path: Path, world_size: int = 3):
    """Shard the fixture across `world_size` ranks, then seal it for the builder."""
    checkpoint_dir = tmp_path / "sealed-checkpoint"
    chunk_dir = checkpoint_dir / "chunks"
    chunk_store = JsonChunkStore()
    for rank in range(world_size):
        spec = _spec(tmp_path, name=f"seal-r{rank}")
        connection = _make_synonym_database(tmp_path / f"seal-r{rank}.sqlite", spec.synonym_csv)
        run_shard(
            spec,
            connection,
            embedder=BatchSensitiveEmbedder(),
            chunk_store=chunk_store,
            chunk_dir=chunk_dir,
            rank=rank,
            world_size=world_size,
        )
        connection.close()

    spec = replace(_spec(tmp_path, name="seal-build"), checkpoint_dir=checkpoint_dir, force=False)
    connection = _make_synonym_database(tmp_path / "seal-build.sqlite", spec.synonym_csv)
    summary = seal_checkpoint(
        spec,
        connection,
        embedder=BatchSensitiveEmbedder(),
        chunk_store=chunk_store,
        checkpoint_dir=checkpoint_dir,
    )
    connection.close()
    return spec, checkpoint_dir, chunk_store, summary


def test_sealed_sharded_stream_is_assembled_without_re_embedding(tmp_path: Path) -> None:
    """The whole sharded path end to end: embed on N ranks, seal, let the builder assemble.

    The builder gets an embedder that raises on any embed call, so the store it produces can
    only have come from the sharded chunks.
    """
    spec, _, chunk_store, summary = _seal_a_sharded_stream(tmp_path)
    assert summary["validated"] is True
    assert summary["rows_in_chunks"] == summary["row_count"]

    manifest = build_concept_store(
        spec,
        embedder=ExplodingEmbedder(),
        chunk_store=chunk_store,
        store_writer=JsonStoreWriter(),
    )
    assert manifest["store"]["row_count"] == summary["row_count"]
    assert manifest["concept_state"]["vector_stream_digest"] == summary["vector_stream_digest"]

    # And the assembled rows carry the sharded vectors, pair for pair.
    reference = _reference_chunks(tmp_path / "ref-for-seal", BATCH_SIZE)
    expected = [
        (row["concept_id"], vector)
        for index in sorted(reference)
        for row, vector in zip(*reference[index], strict=True)
    ]
    payload = json.loads((spec.db_path / f"{spec.table_name}.json").read_text(encoding="utf-8"))
    actual = [
        (row["concept_id"], vector)
        for row, vector in zip(payload["rows"], payload["vectors"], strict=True)
    ]
    assert actual == expected


def test_seal_refuses_an_incomplete_chunk_stream(tmp_path: Path) -> None:
    """Sealing a partial stream would hand the builder a store missing rows, silently."""
    checkpoint_dir = tmp_path / "partial-checkpoint"
    chunk_dir = checkpoint_dir / "chunks"
    chunk_store = JsonChunkStore()
    spec = _spec(tmp_path, name="partial")
    connection = _make_synonym_database(tmp_path / "partial.sqlite", spec.synonym_csv)
    # rank 1 of 3 only, so chunk 1 exists and chunk 0 does not: the stream has a hole at
    # its very start, which is what a missing shard actually looks like on disk.
    run_shard(
        spec,
        connection,
        embedder=BatchSensitiveEmbedder(),
        chunk_store=chunk_store,
        chunk_dir=chunk_dir,
        rank=1,
        world_size=3,
    )
    sealed = replace(spec, checkpoint_dir=checkpoint_dir, force=False)
    with pytest.raises(ConceptStoreBuildError, match="contiguous prefix"):
        seal_checkpoint(
            sealed,
            connection,
            embedder=BatchSensitiveEmbedder(),
            chunk_store=chunk_store,
            checkpoint_dir=checkpoint_dir,
        )
    connection.close()


def test_seal_refuses_a_truncated_but_contiguous_stream(tmp_path: Path) -> None:
    """A contiguous prefix that stops early is the dangerous case: it looks healthy."""
    checkpoint_dir = tmp_path / "short-checkpoint"
    chunk_dir = checkpoint_dir / "chunks"
    chunk_store = JsonChunkStore()
    spec = _spec(tmp_path, name="short")
    connection = _make_synonym_database(tmp_path / "short.sqlite", spec.synonym_csv)
    run_shard(
        spec,
        connection,
        embedder=BatchSensitiveEmbedder(),
        chunk_store=chunk_store,
        chunk_dir=chunk_dir,
        rank=0,
        world_size=1,
        max_chunks=1,
    )
    sealed = replace(spec, checkpoint_dir=checkpoint_dir, force=False)
    with pytest.raises(ConceptStoreBuildError, match="incomplete"):
        seal_checkpoint(
            sealed,
            connection,
            embedder=BatchSensitiveEmbedder(),
            chunk_store=chunk_store,
            checkpoint_dir=checkpoint_dir,
        )
    connection.close()


def test_resume_checks_row_counts_without_reading_vectors(tmp_path: Path) -> None:
    """Resume must confirm a chunk's row count without materializing its vectors.

    Re-checking ~950 chunks by reading them back was the slowest part of resuming a real
    build - far slower than the embedding it was there to skip - because each read turns
    roughly four million doubles into Python floats just to look at a length.
    """

    class ReadIsAnError(JsonChunkStore):
        def read(self, path):  # pragma: no cover - must never be called on the resume path
            raise AssertionError("resume must not read a chunk's vectors to count its rows")

    chunk_dir = tmp_path / "fastresume" / "chunks"
    spec = _spec(tmp_path, name="fastresume")
    synonyms = tmp_path / "fastresume.sqlite"
    connection = _make_synonym_database(synonyms, spec.synonym_csv)
    run_shard(
        spec,
        connection,
        embedder=BatchSensitiveEmbedder(),
        chunk_store=JsonChunkStore(),
        chunk_dir=chunk_dir,
        rank=0,
        world_size=1,
    )
    resumed = run_shard(
        spec,
        connection,
        embedder=ExplodingEmbedder(),
        chunk_store=ReadIsAnError(),
        chunk_dir=chunk_dir,
        rank=0,
        world_size=1,
    )
    connection.close()
    assert resumed["reused_chunks"] == resumed["owned_chunks"] > 0


def test_row_count_agrees_with_a_full_read(tmp_path: Path) -> None:
    """The cheap count and the expensive one must not disagree."""
    store = JsonChunkStore()
    path = tmp_path / "chunk-00000000.arrow"
    rows = [{"concept_id": str(i)} for i in range(7)]
    store.write(path, rows, [[float(i)] * 4 for i in range(7)])
    assert store.row_count(path) == len(store.read(path)[0]) == 7
