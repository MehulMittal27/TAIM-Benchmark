"""Bounded-memory concept-store construction for the pinned TrialMatchAI checkout.

The TrialMatchAI checkout remains an external, read-only system.  This module owns the
orchestration around it and imports its embedder only at runtime in the environment that
contains the pinned checkout's dependencies.

It builds a store from vocabulary files the user supplies and is licensed to use; it carries no
vocabulary content of its own.  ``trial-benchmark patient-to-trial trialmatchai
build-concept-store`` runs :func:`main` in the checkout's own interpreter, so this module imports
only the standard library at load time and nothing from ``taim``: it is executed there by path.

The sharded embedding pass at the end embeds the same chunk stream on several GPUs, one rank per
process, and seals a checkpoint the single-GPU build then assembles the store from.
"""

# The only subprocesses below are fixed, read-only probes for the pinned checkout and NVIDIA
# telemetry.  They do not execute user-provided commands.
# ruff: noqa: S603, S607

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import resource
import shutil
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Concept-store manifest fields classified as build event by
# taim.artifacts.IDENTITY_DIGEST_RULE_VERSION. Everything not listed here is
# identity - including build.trialmatchai_commit, build.builder_version and
# build.chunk_rows, which live in the same block but are determined by the
# pinned inputs and the recipe.
CONCEPT_STORE_BUILD_EVENT_PATHS = (
    "build.checkpoint_path",
    "build.concepts_per_second",
    "build.disk_bytes_after",
    "build.disk_bytes_before",
    "build.disk_bytes_growth",
    "build.elapsed_seconds",
    "build.peak_rss_bytes",
    "build.gpu_utilization.average_percent",
    "build.gpu_utilization.maximum_percent",
    "build.gpu_utilization.samples",
)


BUILDER_VERSION = "taim-trialmatchai-concept-store-v2"
# The embedding stream itself is unchanged from v1. Keep this value in checkpoint identity so
# the completed 3.9M-row v1 Arrow stream can be verified and reused while a new indexed store is
# assembled from it.
CHECKPOINT_BUILDER_VERSION = "taim-trialmatchai-concept-store-v1"
MANIFEST_VERSION = "2"
CONCEPT_STORE_MANIFEST_FILENAME = "concept-store-manifest.json"
DEFAULT_TABLE_NAME = "concepts"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_CHUNK_ROWS = 4096
DEFAULT_VECTOR_INDEX_TYPE = "IVF_PQ"
DEFAULT_VECTOR_INDEX_PARTITIONS = 1024
DEFAULT_VECTOR_INDEX_SUB_VECTORS = 128
DEFAULT_VECTOR_INDEX_NUM_BITS = 8
DEFAULT_VECTOR_INDEX_MAX_ITERATIONS = 50
DEFAULT_VECTOR_INDEX_SAMPLE_RATE = 256
EXPECTED_EMBEDDER_MODEL = "BAAI/bge-m3"
EXPECTED_SCHEMA_FIELDS = (
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
DEFAULT_VOCABULARIES = (
    "SNOMED",
    "LOINC",
    "RxNorm",
    "RxNorm Extension",
    "ATC",
    "ICD10",
    "ICD10CM",
    "MeSH",
    "OMOP Extension",
)

# What an outside build shares with the qualified store, ``candidate-21352305``. Rebuilding its
# bytes is impossible: LanceDB names every file at random, stamps every table version, and trains
# the IVF_PQ and full-text indexes afresh with no seed to fix; and bge-m3 vectors differ in their
# last bits between card classes. So an outside build is accepted on its rows - the same text,
# whatever their order - and on the pinned dictionary versions, and its vectors and indexes are
# recorded as its own.
ROW_DIGEST_FIELDS = EXPECTED_SCHEMA_FIELDS[:-1]
ROW_DIGEST_RULE = (
    "SHA-256 over the sorted SHA-256 digests of one canonical JSON array per row: concept_id, "
    "vocabulary_id, concept_code, concept_name, domain_id, concept_class_id, standard_concept, "
    "synonyms, fts_text. Row order and the embedding column are not part of it"
)

#: The Zenodo version DOI of the project's unmodified archive of the open dictionary files whose
#: own sources no longer serve the versions the qualified store was built from (record 23088342,
#: published 2026-10-01). A version DOI, never the all-versions one, which would follow later
#: versions of the record.
PROJECT_DICTIONARY_DOI: str | None = "10.5281/zenodo.23088342"


def zenodo_file_url(doi: str, name: str) -> str:
    """The download URL of file ``name`` in the Zenodo record that ``doi`` names."""

    record = doi.removeprefix("https://doi.org/").removeprefix("10.5281/zenodo.")
    if record == doi.removeprefix("https://doi.org/") or not record.isdigit():
        raise ValueError(f"not a Zenodo record DOI (10.5281/zenodo.<record>): {doi}")
    return f"https://zenodo.org/api/records/{record}/files/{name}/content"


@dataclass(frozen=True, slots=True)
class PinnedDictionarySource:
    """One of TrialMatchAI's open dictionary sources, at the qualified store's version.

    ``filename`` is the raw file TrialMatchAI's own parser reads, and ``sha256`` its bytes; the
    parser then writes the dictionary recorded under ``dictionary`` with ``dictionary_sha256``.
    A source without ``url`` is archived by the project under :data:`PROJECT_DICTIONARY_DOI`
    as ``published_name``, gzip-compressed when ``published_sha256`` is set.
    """

    key: str
    version: str
    filename: str
    sha256: str
    dictionary: str
    dictionary_sha256: str
    url: str | None = None
    published_name: str | None = None
    published_sha256: str | None = None


# In TrialMatchAI's source order, which is the stream's row order.
PINNED_OPEN_DICTIONARIES = (
    PinnedDictionarySource(
        key="genes",
        version="NCBI Gene Homo_sapiens.gene_info.gz as served 2026-08-05 (daily file, no archive)",
        filename="Homo_sapiens.gene_info.gz",
        sha256="sha256:6747e9aab54f5568d67134a1f4c6803fd80ee3b27a8a655594ae7bf74d8c8c49",
        dictionary="dictionary:NCBIGene:dict_Gene.txt",
        dictionary_sha256="sha256:e83c12e5ab960b76898052ef2cc7ba3bf5f03ac9686dce7fef52c0710e51847a",
        published_name="Homo_sapiens.gene_info.gz",
    ),
    PinnedDictionarySource(
        key="diseases",
        version="Disease Ontology release 2026-07-31",
        filename="doid.obo",
        sha256="sha256:5b9803aa17eeabf4c70f144c64216294d01e66335da3e560576d4eb2dc9ff490",
        dictionary="dictionary:DOID:dict_Disease.txt",
        dictionary_sha256="sha256:6fc9d35ebe9a97c2a0c93cc3e17116353c07af2cd8e04d2c92e9675058036d04",
        url="https://purl.obolibrary.org/obo/doid/releases/2026-07-31/doid.obo",
    ),
    PinnedDictionarySource(
        key="chemicals",
        version="ChEBI release 253 (lite)",
        filename="chebi_lite.obo",
        sha256="sha256:6281295295e1a34d49fa809632a3a70f93fdc308b63d8896230e3cb05c558071",
        dictionary="dictionary:ChEBI:dict_Chemical.txt",
        dictionary_sha256="sha256:a282524bf4a7e108f20cae3def515fe9ecd23e03d0a60af0398b422d3fb50192",
        url="https://ftp.ebi.ac.uk/pub/databases/chebi/archive/rel253/ontology/chebi_lite.obo",
    ),
    PinnedDictionarySource(
        key="cell_lines",
        version="Cellosaurus release 56.0 (current-only download, no archive)",
        filename="cellosaurus.obo",
        sha256="sha256:bebb8bd8e63af2a051aed67adfac8f90246e9bfade92d97d871e99e9b4d4b365",
        dictionary="dictionary:Cellosaurus:dict_CellLine.txt",
        dictionary_sha256="sha256:ad86dc539d874fcf0824788016cfd2ffad4a236c34af57407bdc40b3105d6946",
        published_name="cellosaurus.obo.gz",
        published_sha256="sha256:a99592b0cc205921016d2d39cc7149e99ebab27ce54decaf4fee55b1fe529901",
    ),
    PinnedDictionarySource(
        key="cell_types",
        version="Cell Ontology release 2026-06-08",
        filename="cl.obo",
        sha256="sha256:7e19b4aef8e7fe7720b0c4474c6a5b8713c904ea9ab7b7d5d96f2cb28b8fc51f",
        dictionary="dictionary:CL:dict_CellType.txt",
        dictionary_sha256="sha256:92e7ae510c9143ec4a320e377db9ffa264bcd8edf9ba108d85f0d0e7d35e4f7e",
        url="https://purl.obolibrary.org/obo/cl/releases/2026-06-08/cl.obo",
    ),
    PinnedDictionarySource(
        key="phenotypes",
        version="Human Phenotype Ontology release 2026-06-23",
        filename="hp.obo",
        sha256="sha256:a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b",
        dictionary="dictionary:HPO:dict_Phenotype.txt",
        dictionary_sha256="sha256:7ee936a3d0f63996c8d8b09f3fd15395c7f1f9a149ae77a89d685b348a1f95de",
        url="https://purl.obolibrary.org/obo/hp/releases/2026-06-23/hp.obo",
    ),
)


class ConceptStoreBuildError(RuntimeError):
    """Raised when a build cannot safely start, resume, or finalize."""


class Embedder(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class ChunkStore(Protocol):
    """Persistence for bounded chunks.

    The production implementation is Arrow IPC.  Tests can inject a small deterministic
    implementation without installing LanceDB or the TrialMatchAI ML dependencies.
    """

    def write(
        self, path: Path, rows: Sequence[Mapping[str, Any]], vectors: Sequence[Sequence[float]]
    ) -> None: ...

    def read(self, path: Path) -> tuple[list[dict[str, Any]], list[list[float]]]: ...

    def row_count(self, path: Path) -> int:
        """Rows in a chunk, without materializing its vectors.

        Resume only needs to confirm a completed chunk still holds the rows the stream says
        it should. Going through ``read`` for that turns ~4M doubles per chunk into Python
        floats, which dominates the cost of resuming a long build.
        """
        ...


class StoreWriter(Protocol):
    def write(
        self,
        db_path: Path,
        table_name: str,
        chunks: Sequence[Path],
        chunk_store: ChunkStore,
        vector_index: Mapping[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class DictionarySource:
    vocabulary_id: str
    domain_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class ConceptStoreBuildSpec:
    concept_csv: Path
    synonym_csv: Path
    db_path: Path
    trialmatchai_root: Path
    config: Mapping[str, Any]
    vocabularies: tuple[str, ...] = DEFAULT_VOCABULARIES
    dictionary_sources: tuple[DictionarySource, ...] = ()
    source_release: str = "Athena v20260227"
    table_name: str = DEFAULT_TABLE_NAME
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    vector_index_partitions: int = DEFAULT_VECTOR_INDEX_PARTITIONS
    vector_index_sub_vectors: int = DEFAULT_VECTOR_INDEX_SUB_VECTORS
    checkpoint_dir: Path | None = None
    normalization: str = "tab-delimited-quote-none-v1"
    trialmatchai_commit: str | None = None
    checkpoint_trialmatchai_commit: str | None = None
    expected_trialmatchai_commit: str | None = None
    force: bool = False

    def validate(self) -> None:
        if self.chunk_rows <= 0 or self.chunk_rows % self.embedding_batch_size:
            raise ConceptStoreBuildError(
                "chunk_rows must be a positive multiple of embedding_batch_size so global "
                "embedding batches remain unchanged"
            )
        if self.embedding_batch_size != DEFAULT_EMBEDDING_BATCH_SIZE:
            raise ConceptStoreBuildError(
                "embedding_batch_size must remain 32 to match TrialMatchAI"
            )
        if self.vector_index_partitions <= 0:
            raise ConceptStoreBuildError("vector_index_partitions must be positive")
        if self.vector_index_sub_vectors <= 0:
            raise ConceptStoreBuildError("vector_index_sub_vectors must be positive")
        if not self.source_release.strip():
            raise ConceptStoreBuildError("source_release must be non-empty")
        if not self.concept_csv.is_file():
            raise ConceptStoreBuildError(f"concept source does not exist: {self.concept_csv}")
        if not self.synonym_csv.is_file():
            raise ConceptStoreBuildError(f"synonym source does not exist: {self.synonym_csv}")
        if not self.trialmatchai_root.is_dir():
            raise ConceptStoreBuildError(
                f"TrialMatchAI checkout does not exist: {self.trialmatchai_root}"
            )
        embedder = self.config.get("embedder")
        if not isinstance(embedder, Mapping):
            raise ConceptStoreBuildError("config.embedder must be an object")
        model_name = str(embedder.get("model_name", ""))
        if model_name != EXPECTED_EMBEDDER_MODEL:
            raise ConceptStoreBuildError(
                f"the bounded builder requires {EXPECTED_EMBEDDER_MODEL}, got {model_name!r}"
            )
        revision = embedder.get("revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ConceptStoreBuildError(
                "config.embedder.revision must pin the BGE-M3 model revision"
            )
        configured_batch = int(embedder.get("batch_size", self.embedding_batch_size))
        if configured_batch != self.embedding_batch_size:
            raise ConceptStoreBuildError(
                "config.embedder.batch_size must be 32 to preserve TrialMatchAI batching"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _digest_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class GpuSampler:
    def __init__(self, interval_seconds: float = 5.0) -> None:
        self.interval_seconds = interval_seconds
        self.values: list[float] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuSampler:
        self._thread = threading.Thread(target=self._run, name="taim-gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                values = [
                    float(line.strip()) for line in completed.stdout.splitlines() if line.strip()
                ]
                self.values.extend(values)
            except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError) as exc:
                self.error = str(exc)
                return
            self._stop.wait(self.interval_seconds)

    def as_dict(self) -> dict[str, Any]:
        if not self.values:
            return {"available": False, "samples": 0, "error": self.error or "no samples"}
        return {
            "available": True,
            "samples": len(self.values),
            "average_percent": sum(self.values) / len(self.values),
            "maximum_percent": max(self.values),
        }


def _dedupe_strings(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return tuple(result)


def _concept_row(
    *,
    concept_id: str,
    vocabulary_id: str,
    concept_code: str,
    concept_name: str,
    domain_id: str,
    concept_class_id: str,
    standard_concept: str,
    synonyms: Iterable[str],
) -> dict[str, Any]:
    synonym_values = _dedupe_strings(synonyms)
    return {
        "concept_id": concept_id,
        "vocabulary_id": vocabulary_id,
        "concept_code": concept_code,
        "concept_name": concept_name,
        "domain_id": domain_id,
        "concept_class_id": concept_class_id,
        "standard_concept": standard_concept,
        "synonyms": list(synonym_values),
        "fts_text": " ".join([concept_name, *synonym_values]),
    }


def _make_synonym_database(path: Path, source: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute(
        "CREATE TABLE synonyms ("
        "concept_id TEXT NOT NULL, ordinal INTEGER NOT NULL, name TEXT NOT NULL)"
    )
    ordinal = 0
    pending: list[tuple[str, int, str]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            concept_id = (row.get("concept_id") or "").strip()
            synonym = (row.get("concept_synonym_name") or "").strip()
            if concept_id and synonym:
                pending.append((concept_id, ordinal, synonym))
                ordinal += 1
                if len(pending) >= 10_000:
                    connection.executemany("INSERT INTO synonyms VALUES (?, ?, ?)", pending)
                    connection.commit()
                    pending.clear()
    if pending:
        connection.executemany("INSERT INTO synonyms VALUES (?, ?, ?)", pending)
    connection.commit()
    connection.execute("CREATE INDEX synonyms_concept_id ON synonyms (concept_id, ordinal)")
    connection.commit()
    return connection


def _synonyms(connection: sqlite3.Connection, concept_id: str) -> tuple[str, ...]:
    values = connection.execute(
        "SELECT name FROM synonyms WHERE concept_id = ? ORDER BY ordinal", (concept_id,)
    )
    return _dedupe_strings(row[0] for row in values)


def _iter_omop_rows(
    concept_csv: Path, connection: sqlite3.Connection, vocabularies: frozenset[str]
) -> Iterator[dict[str, Any]]:
    with concept_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for source_row in reader:
            vocabulary_id = (source_row.get("vocabulary_id") or "").strip()
            if vocabularies and vocabulary_id.casefold() not in vocabularies:
                continue
            if (source_row.get("invalid_reason") or "").strip():
                continue
            concept_id = (source_row.get("concept_id") or "").strip()
            concept_name = (source_row.get("concept_name") or "").strip()
            concept_code = (source_row.get("concept_code") or "").strip()
            if not concept_id or not concept_name or not concept_code:
                continue
            yield _concept_row(
                concept_id=concept_id,
                vocabulary_id=vocabulary_id,
                concept_code=concept_code,
                concept_name=concept_name,
                domain_id=(source_row.get("domain_id") or "").strip(),
                concept_class_id=(source_row.get("concept_class_id") or "").strip(),
                standard_concept=(source_row.get("standard_concept") or "").strip(),
                synonyms=_synonyms(connection, concept_id),
            )


def _strip_dictionary_prefix(identifier: str, vocabulary_id: str) -> str:
    if not identifier:
        return ""
    if ":" not in identifier:
        return identifier
    prefix, code = identifier.split(":", 1)
    return code if prefix.casefold() == vocabulary_id.casefold() else identifier


def _iter_dictionary_rows(source: DictionarySource) -> Iterator[dict[str, Any]]:
    with source.path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or "||" not in stripped:
                continue
            identifiers, names = stripped.split("||", 1)
            synonyms = _dedupe_strings(names.split("|"))
            if not synonyms:
                continue
            for raw_identifier in identifiers.split(","):
                concept_code = _strip_dictionary_prefix(
                    raw_identifier.strip(), source.vocabulary_id
                )
                if not concept_code:
                    continue
                yield _concept_row(
                    concept_id=f"{source.vocabulary_id}:{concept_code}",
                    vocabulary_id=source.vocabulary_id,
                    concept_code=concept_code,
                    concept_name=synonyms[0],
                    domain_id=source.domain_id,
                    concept_class_id="Dictionary",
                    standard_concept="",
                    synonyms=synonyms,
                )


def _iter_rows(
    spec: ConceptStoreBuildSpec, connection: sqlite3.Connection
) -> Iterator[dict[str, Any]]:
    vocabularies = frozenset(value.casefold() for value in spec.vocabularies)
    yield from _iter_omop_rows(spec.concept_csv, connection, vocabularies)
    for source in spec.dictionary_sources:
        yield from _iter_dictionary_rows(source)


def _row_order_identity(
    spec: ConceptStoreBuildSpec, connection: sqlite3.Connection
) -> tuple[str, int, dict[str, int]]:
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    rows = 0
    for row in _iter_rows(spec, connection):
        row_identity_update(digest, row)
        counts[str(row["vocabulary_id"])] += 1
        rows += 1
    return digest.hexdigest(), rows, dict(sorted(counts.items()))


def _embedding_contract(config: Mapping[str, Any], embedder: Embedder) -> dict[str, Any]:
    values = config.get("embedder")
    if not isinstance(values, Mapping):
        raise ConceptStoreBuildError("config.embedder must be an object")
    contract = {
        "model_name": values.get("model_name"),
        "revision": values.get("revision"),
        "pooling": values.get("pooling", "mean"),
        "max_length": int(values.get("max_length", 512)),
        "batch_size": int(values.get("batch_size", DEFAULT_EMBEDDING_BATCH_SIZE)),
        "use_fp16": bool(values.get("use_fp16", False)),
        "normalize": bool(values.get("normalize", True)),
        "native_metric": values.get("native_metric"),
    }
    fingerprint = getattr(embedder, "fingerprint", None)
    if callable(fingerprint):
        contract["runtime_fingerprint"] = str(fingerprint())
    return contract


def _source_identity(spec: ConceptStoreBuildSpec) -> dict[str, Any]:
    return {
        "files": {
            "CONCEPT.csv": _sha256_file(spec.concept_csv),
            "CONCEPT_SYNONYM.csv": _sha256_file(spec.synonym_csv),
            **{
                f"dictionary:{source.vocabulary_id}:{source.path.name}": _sha256_file(source.path)
                for source in spec.dictionary_sources
            },
        },
        "release": spec.source_release,
        "normalization": spec.normalization,
        "omop_vocabularies": list(spec.vocabularies),
        "open_dictionary_sources": [
            {
                "vocabulary_id": source.vocabulary_id,
                "domain_id": source.domain_id,
                "path": str(source.path),
            }
            for source in spec.dictionary_sources
        ],
    }


def _checkout_commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConceptStoreBuildError(
            f"could not resolve TrialMatchAI checkout commit: {root}"
        ) from exc
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise ConceptStoreBuildError(f"invalid TrialMatchAI checkout commit: {commit!r}")
    return commit


def _identity(
    spec: ConceptStoreBuildSpec,
    *,
    source: Mapping[str, Any],
    row_order: str,
    row_count: int,
    vocabulary_counts: Mapping[str, int],
    embedding: Mapping[str, Any],
    commit: str,
) -> dict[str, Any]:
    return {
        "builder_version": CHECKPOINT_BUILDER_VERSION,
        "trialmatchai_commit": commit,
        "source": dict(source),
        "row_order_identity": row_order,
        "row_count": row_count,
        "vocabulary_counts": dict(vocabulary_counts),
        "table_name": spec.table_name,
        "schema_fields": list(EXPECTED_SCHEMA_FIELDS),
        "chunk_rows": spec.chunk_rows,
        "embedding_batch_size": spec.embedding_batch_size,
        "embedding": dict(embedding),
    }


def _chunk_paths(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("chunks/chunk-*.arrow"))


class _VectorStreamDigest:
    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.dimension: int | None = None

    def update(self, rows: Sequence[Mapping[str, Any]], vectors: Sequence[Sequence[float]]) -> None:
        for row, vector in zip(rows, vectors, strict=True):
            dimension = len(vector)
            if dimension == 0:
                raise ConceptStoreBuildError("embedding chunk stream has no vector dimension")
            if self.dimension is None:
                self.dimension = dimension
            elif dimension != self.dimension:
                raise ConceptStoreBuildError(
                    f"embedding dimension is not uniform: {self.dimension} and {dimension}"
                )
            self._digest.update(str(row["concept_id"]).encode("utf-8"))
            self._digest.update(struct.pack(f"<{dimension}d", *[float(value) for value in vector]))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ValidatedCheckpoint:
    completed_chunks: int
    chunk_row_counts: tuple[int, ...]
    vector_digest: _VectorStreamDigest


def _checkpoint_payload(
    identity: Mapping[str, Any], completed_chunks: int, row_count: int, vector_digest: str
) -> dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "identity": dict(identity),
        "completed_chunks": completed_chunks,
        "row_count": row_count,
        "vector_stream_digest": f"sha256:{vector_digest}",
    }


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConceptStoreBuildError(f"checkpoint is unreadable: {path}") from exc
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != 1:
        raise ConceptStoreBuildError(f"unsupported checkpoint: {path}")
    return payload


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    identity: Mapping[str, Any],
    checkpoint_dir: Path,
    chunk_store: ChunkStore,
) -> _ValidatedCheckpoint:
    if checkpoint.get("identity") != dict(identity):
        raise ConceptStoreBuildError(
            "checkpoint identity differs from the current inputs, model, schema, or row order; "
            "use a new checkpoint directory or --force"
        )
    completed = checkpoint.get("completed_chunks")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise ConceptStoreBuildError("checkpoint completed_chunks is invalid")
    paths = _chunk_paths(checkpoint_dir)
    expected = [
        checkpoint_dir / "chunks" / f"chunk-{index:08d}.arrow" for index in range(completed)
    ]
    if paths != expected:
        raise ConceptStoreBuildError("checkpoint chunk files do not match completed_chunks")
    recorded_digest = checkpoint.get("vector_stream_digest")
    if not isinstance(recorded_digest, str) or not recorded_digest.startswith("sha256:"):
        raise ConceptStoreBuildError("checkpoint vector_stream_digest is invalid")
    actual_digest = _VectorStreamDigest()
    chunk_row_counts: list[int] = []
    for path in expected:
        rows, vectors = chunk_store.read(path)
        if len(rows) != len(vectors):
            raise ConceptStoreBuildError(f"checkpoint chunk has invalid row count: {path}")
        chunk_row_counts.append(len(rows))
        actual_digest.update(rows, vectors)
    if recorded_digest != f"sha256:{actual_digest.hexdigest()}":
        raise ConceptStoreBuildError("checkpoint vector_stream_digest does not match chunk files")
    return _ValidatedCheckpoint(completed, tuple(chunk_row_counts), actual_digest)


class JsonChunkStore:
    """Small JSON chunk codec used only by local unit tests."""

    def write(
        self, path: Path, rows: Sequence[Mapping[str, Any]], vectors: Sequence[Sequence[float]]
    ) -> None:
        _atomic_json_write(
            path, {"rows": list(rows), "vectors": [list(vector) for vector in vectors]}
        )

    def read(self, path: Path) -> tuple[list[dict[str, Any]], list[list[float]]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload["rows"]), [list(vector) for vector in payload["vectors"]]

    def row_count(self, path: Path) -> int:
        return len(json.loads(path.read_text(encoding="utf-8"))["rows"])


class JsonStoreWriter:
    """Test writer that preserves final logical rows without LanceDB."""

    def write(
        self,
        db_path: Path,
        table_name: str,
        chunks: Sequence[Path],
        chunk_store: ChunkStore,
        vector_index: Mapping[str, Any],
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        for path in chunks:
            chunk_rows, chunk_vectors = chunk_store.read(path)
            rows.extend(chunk_rows)
            vectors.extend(chunk_vectors)
        db_path.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(db_path / f"{table_name}.json", {"rows": rows, "vectors": vectors})
        return {
            "fields": list(EXPECTED_SCHEMA_FIELDS),
            "fts_index_created": True,
            "vector_index_created": True,
            "indexes": _index_metadata(vector_index),
        }


class ArrowChunkStore:
    """Arrow IPC chunk codec used by the CUDA/TrialMatchAI environment."""

    def write(
        self, path: Path, rows: Sequence[Mapping[str, Any]], vectors: Sequence[Sequence[float]]
    ) -> None:
        import pyarrow as pa

        payload = [
            dict(row, embedding=list(vector)) for row, vector in zip(rows, vectors, strict=True)
        ]
        table = pa.Table.from_pylist(payload)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with (
                pa.OSFile(str(temporary), "wb") as sink,
                pa.ipc.new_file(sink, table.schema) as writer,
            ):
                writer.write_table(table)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def read(self, path: Path) -> tuple[list[dict[str, Any]], list[list[float]]]:
        import pyarrow as pa

        with pa.memory_map(str(path), "r") as source:
            table = pa.ipc.open_file(source).read_all()
        payload = table.to_pylist()
        return (
            [{key: value for key, value in row.items() if key != "embedding"} for row in payload],
            [list(row["embedding"]) for row in payload],
        )

    def row_count(self, path: Path) -> int:
        """Read the row count out of the Arrow footer, not out of the vectors."""
        import pyarrow as pa

        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_file(source)
            return sum(
                reader.get_batch(index).num_rows for index in range(reader.num_record_batches)
            )


class LanceDBStoreWriter:
    """Write Arrow chunks and build every index required by canonical concept linking."""

    def write(
        self,
        db_path: Path,
        table_name: str,
        chunks: Sequence[Path],
        chunk_store: ChunkStore,
        vector_index: Mapping[str, Any],
    ) -> dict[str, Any]:
        import lancedb
        import pyarrow as pa

        del pa
        if not chunks:
            raise ConceptStoreBuildError("cannot create an empty concept store")
        db = lancedb.connect(str(db_path))
        table = None
        field_names: list[str] = []
        for index, path in enumerate(chunks):
            rows, vectors = chunk_store.read(path)
            payload = [
                dict(row, embedding=list(vector)) for row, vector in zip(rows, vectors, strict=True)
            ]
            arrow_table = _rows_to_arrow(payload)
            if index == 0:
                if list(arrow_table.column_names) != list(EXPECTED_SCHEMA_FIELDS):
                    raise ConceptStoreBuildError(
                        f"unexpected LanceDB schema: {list(arrow_table.column_names)}"
                    )
                table = db.create_table(table_name, data=arrow_table, mode="overwrite")
                field_names = list(arrow_table.column_names)
            else:
                if table is None:
                    raise ConceptStoreBuildError("LanceDB table was not created")
                table.add(arrow_table)
        if table is None:
            raise ConceptStoreBuildError("LanceDB table was not created")
        vector_index = dict(vector_index)
        try:
            table.create_fts_index("fts_text", replace=True)
        except Exception as exc:
            raise ConceptStoreBuildError("LanceDB FTS index creation failed") from exc
        try:
            table.create_index(
                metric=vector_index["metric"],
                num_partitions=vector_index["num_partitions"],
                num_sub_vectors=vector_index["num_sub_vectors"],
                vector_column_name=vector_index["column"],
                replace=True,
                index_type=vector_index["type"],
                num_bits=vector_index["num_bits"],
                max_iterations=vector_index["max_iterations"],
                sample_rate=vector_index["sample_rate"],
                train=vector_index["train"],
            )
        except Exception as exc:
            raise ConceptStoreBuildError("LanceDB vector index creation failed") from exc
        try:
            for column in ("vocabulary_id", "domain_id"):
                table.create_scalar_index(column, replace=True, index_type="BTREE")
        except Exception as exc:
            raise ConceptStoreBuildError("LanceDB concept-filter index creation failed") from exc
        try:
            actual_indexes = {
                (
                    tuple(index.columns),
                    "".join(
                        character
                        for character in str(index.index_type).casefold()
                        if character.isalnum()
                    ),
                )
                for index in table.list_indices()
            }
        except Exception as exc:
            raise ConceptStoreBuildError("LanceDB index inspection failed") from exc
        required_indexes = {
            (("fts_text",), "fts"),
            (("embedding",), "ivfpq"),
            (("vocabulary_id",), "btree"),
            (("domain_id",), "btree"),
        }
        if not required_indexes.issubset(actual_indexes):
            raise ConceptStoreBuildError(
                "LanceDB did not report every required FTS, IVF-PQ, and BTree index"
            )
        return {
            "fields": field_names,
            "fts_index_created": True,
            "vector_index_created": True,
            "indexes": _index_metadata(vector_index),
        }


def _index_metadata(vector_index: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fts": {"column": "fts_text", "type": "FTS"},
        "vector": dict(vector_index),
        "filters": [
            {"column": "vocabulary_id", "type": "BTREE"},
            {"column": "domain_id", "type": "BTREE"},
        ],
    }


def _vector_index_contract(
    spec: ConceptStoreBuildSpec, *, metric: str, dimension: int
) -> dict[str, Any]:
    if dimension % spec.vector_index_sub_vectors:
        raise ConceptStoreBuildError(
            "embedding dimension must be divisible by vector_index_sub_vectors"
        )
    return {
        "column": "embedding",
        "type": DEFAULT_VECTOR_INDEX_TYPE,
        "metric": metric,
        "num_partitions": spec.vector_index_partitions,
        "num_sub_vectors": spec.vector_index_sub_vectors,
        "num_bits": DEFAULT_VECTOR_INDEX_NUM_BITS,
        "max_iterations": DEFAULT_VECTOR_INDEX_MAX_ITERATIONS,
        "sample_rate": DEFAULT_VECTOR_INDEX_SAMPLE_RATE,
        "train": True,
    }


def _rows_to_arrow(payload: Sequence[Mapping[str, Any]]) -> Any:
    import pyarrow as pa

    return pa.Table.from_pylist(list(payload))


def build_concept_store(
    spec: ConceptStoreBuildSpec,
    *,
    embedder: Embedder | None = None,
    chunk_store: ChunkStore | None = None,
    store_writer: StoreWriter | None = None,
    stop_after_chunks: int | None = None,
    gpu_utilization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build or resume a bounded concept store.

    Chunks are materialized as atomic Arrow files before final LanceDB assembly.  This keeps
    restart semantics safe: an interrupted run resumes embedding from the last complete chunk,
    reconstructs digest state while verifying that completed prefix once, and then reconstructs
    the LanceDB table from the verified chunk stream.
    """

    spec.validate()
    chunk_store = chunk_store or ArrowChunkStore()
    store_writer = store_writer or LanceDBStoreWriter()
    checkpoint_dir = spec.checkpoint_dir or spec.db_path.parent / f".{spec.db_path.name}.checkpoint"
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    chunks_dir = checkpoint_dir / "chunks"
    source = _source_identity(spec)
    commit = spec.trialmatchai_commit or _checkout_commit(spec.trialmatchai_root)
    checkpoint_commit = spec.checkpoint_trialmatchai_commit or commit
    expected_commit = spec.expected_trialmatchai_commit
    if expected_commit and commit != expected_commit:
        raise ConceptStoreBuildError(
            f"TrialMatchAI checkout {commit} does not match the locked commit {expected_commit}"
        )
    if spec.force and checkpoint_commit != commit:
        raise ConceptStoreBuildError(
            "--force cannot be combined with cross-commit checkpoint reuse because it would "
            "delete the preserved vector stream"
        )

    if spec.force:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(spec.db_path, ignore_errors=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    disk_before = _directory_size(spec.db_path) + _directory_size(checkpoint_dir)
    synonyms_path = checkpoint_dir / "synonyms.sqlite"
    connection = _make_synonym_database(synonyms_path, spec.synonym_csv)
    try:
        row_order, row_count, vocabulary_counts = _row_order_identity(spec, connection)
        if row_count <= 0:
            raise ConceptStoreBuildError("concept sources produced no included concepts")
        active_embedder = embedder or build_external_embedder(spec.config)
        embedding = _embedding_contract(spec.config, active_embedder)
        identity = _identity(
            spec,
            source=source,
            row_order=row_order,
            row_count=row_count,
            vocabulary_counts=vocabulary_counts,
            embedding=embedding,
            commit=checkpoint_commit,
        )
        checkpoint = _load_checkpoint(checkpoint_path)
        completed = 0
        completed_chunk_row_counts: tuple[int, ...] = ()
        vector_digest = _VectorStreamDigest()
        if checkpoint is None and checkpoint_commit != commit:
            raise ConceptStoreBuildError(
                "cross-commit checkpoint reuse requires an existing complete vector stream"
            )
        if checkpoint is not None:
            validated_checkpoint = _validate_checkpoint(
                checkpoint, identity, checkpoint_dir, chunk_store
            )
            completed = validated_checkpoint.completed_chunks
            completed_chunk_row_counts = validated_checkpoint.chunk_row_counts
            vector_digest = validated_checkpoint.vector_digest
            if checkpoint_commit != commit and (
                checkpoint.get("status") != "complete"
                or checkpoint.get("row_count") != row_count
                or sum(completed_chunk_row_counts) != row_count
            ):
                raise ConceptStoreBuildError(
                    "cross-commit checkpoint reuse requires a complete verified vector stream"
                )

        processed_rows = 0
        for chunk_index, rows in enumerate(_chunked(_iter_rows(spec, connection), spec.chunk_rows)):
            chunk_path = chunks_dir / f"chunk-{chunk_index:08d}.arrow"
            if chunk_index < completed:
                existing_row_count = completed_chunk_row_counts[chunk_index]
                if existing_row_count != len(rows):
                    raise ConceptStoreBuildError(
                        f"checkpoint chunk has invalid row count: {chunk_path}"
                    )
                processed_rows += existing_row_count
                continue
            vectors = [list(vector) for vector in active_embedder.embed_texts(chunk_texts(rows))]
            if len(vectors) != len(rows):
                raise ConceptStoreBuildError(
                    f"embedder returned {len(vectors)} vectors for {len(rows)} rows"
                )
            dimensions = {len(vector) for vector in vectors}
            if len(dimensions) != 1 or not dimensions:
                raise ConceptStoreBuildError(f"embedding dimension is not uniform: {dimensions}")
            vector_digest.update(rows, vectors)
            chunk_store.write(chunk_path, rows, vectors)
            completed = chunk_index + 1
            processed_rows += len(rows)
            _atomic_json_write(
                checkpoint_path,
                _checkpoint_payload(identity, completed, processed_rows, vector_digest.hexdigest()),
            )
            if stop_after_chunks is not None and completed >= stop_after_chunks:
                raise ConceptStoreBuildError("test stop requested after checkpointed chunks")

        if processed_rows != row_count:
            raise ConceptStoreBuildError(
                f"streamed row count {processed_rows} differs from preflight count {row_count}"
            )
        chunks = _chunk_paths(checkpoint_dir)
        if len(chunks) != completed:
            raise ConceptStoreBuildError("chunk stream is incomplete")
        final_vector_digest = vector_digest.hexdigest()
        embedding_dimension = vector_digest.dimension
        if embedding_dimension is None:
            raise ConceptStoreBuildError("embedding chunk stream has no vector dimension")
        embedding_metric = embedding.get("native_metric") or (
            "cosine" if embedding.get("normalize", True) else "dot"
        )
        vector_index = _vector_index_contract(
            spec,
            metric=str(embedding_metric),
            dimension=embedding_dimension,
        )
        writer_metadata = store_writer.write(
            spec.db_path,
            spec.table_name,
            chunks,
            chunk_store,
            vector_index,
        )
        indexes = writer_metadata.get("indexes")
        if not isinstance(indexes, Mapping) or indexes.get("vector") != vector_index:
            raise ConceptStoreBuildError("store writer did not confirm the required vector index")
        store_identity = {
            "concepts": identity,
            "indexes": indexes,
            "builder_version": BUILDER_VERSION,
        }
        elapsed = time.perf_counter() - started
        disk_after = _directory_size(spec.db_path) + _directory_size(checkpoint_dir)
        manifest: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "sources": {
                **source,
                "format": "OMOP Athena tab-delimited export",
            },
            "store": {
                "table": spec.table_name,
                "row_count": row_count,
                "fields": writer_metadata.get("fields", []),
                "indexes": indexes,
            },
            "vocabularies": dict(sorted(vocabulary_counts.items())),
            "embeddings": {
                "mode": "vector",
                "skip_embeddings": False,
                "present": True,
                "dimension": embedding_dimension,
                "metric": embedding_metric,
                "model_name": embedding["model_name"],
                "revision": embedding["revision"],
                "pooling": embedding["pooling"],
                "max_length": embedding["max_length"],
                "batch_size": embedding["batch_size"],
                "use_fp16": embedding["use_fp16"],
                "normalize": embedding["normalize"],
            },
            "concept_state": {
                "state_version": "3",
                "fingerprint": f"sha256:{_digest_json(store_identity)}",
                "vector_stream_digest": f"sha256:{final_vector_digest}",
            },
            "build": {
                "builder_version": BUILDER_VERSION,
                "trialmatchai_commit": commit,
                "checkpoint_trialmatchai_commit": checkpoint_commit,
                "identity": identity,
                "checkpoint_path": str(checkpoint_path),
                "chunk_rows": spec.chunk_rows,
                "elapsed_seconds": elapsed,
                "concepts_per_second": row_count / elapsed if elapsed else 0.0,
                "peak_rss_bytes": _peak_rss_bytes(),
                "disk_bytes_before": disk_before,
                "disk_bytes_after": disk_after,
                "disk_bytes_growth": disk_after - disk_before,
                "gpu_utilization": dict(
                    gpu_utilization
                    or {
                        "available": False,
                        "error": "use CLI monitor for GPU samples",
                    }
                ),
                "fts_index_created": bool(writer_metadata.get("fts_index_created")),
                "vector_index_created": bool(writer_metadata.get("vector_index_created")),
            },
        }
        state_path = spec.db_path / ".concepts_state.json"
        _atomic_json_write(
            state_path,
            {"fingerprint": _digest_json(store_identity), "state_version": "3"},
        )
        manifest["concept_state"]["file_sha256"] = _sha256_file(state_path)
        _atomic_json_write(spec.db_path / CONCEPT_STORE_MANIFEST_FILENAME, manifest)
        _atomic_json_write(
            checkpoint_path,
            {
                **_checkpoint_payload(identity, completed, row_count, final_vector_digest),
                "status": "complete",
            },
        )
        return manifest
    finally:
        connection.close()


def _chunked(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def build_external_embedder(config: Mapping[str, Any]) -> Embedder:
    try:
        from trialmatchai.models.embedding import build_embedder
    except ImportError as exc:
        raise ConceptStoreBuildError(
            "TrialMatchAI embedding dependencies are unavailable; run this command in the "
            "pinned external checkout environment"
        ) from exc
    return build_embedder(dict(config))


# Sharding the embedding pass across several GPUs, chunk by chunk.
#
# The single-GPU builder above streams rows in a fixed order, cuts them into fixed-size chunks,
# and hands each chunk's texts to the pinned embedder in one call.  The embedder then slices that
# call into fixed batches of 32 in input order.  So the vectors a chunk produces depend on exactly
# two things: which rows are in the chunk, and in what order.
#
# The shard pass keeps both of those identical and changes only *which process* embeds a chunk:
# shard ``r`` of ``N`` owns chunk ``k`` iff ``k % N == r``.  Every rank streams the whole row
# iterator and skips chunks it does not own, so chunk boundaries are derived from the same global
# stream in every rank rather than recomputed by arithmetic - the off-by-one shard boundary is the
# failure this design is built to make impossible.
#
# Nothing here changes the batch size, the model, the pooling, the dimension or the row order.
# Only the assignment of chunks to processes.

SHARD_TOOL_VERSION = "taim-concept-store-shard-v1"


def chunk_vector_digest(
    rows: Sequence[Mapping[str, Any]], vectors: Sequence[Sequence[float]]
) -> str:
    """Digest the ``(concept_id, raw vector bytes)`` pairs of one chunk.

    Byte-for-byte the slice of the builder's own ``_VectorStreamDigest`` covering this
    chunk, so chunk digests chained in index order reproduce the global vector-stream
    digest, and one chunk digest is a sound equality test for that chunk's pairs.
    """
    digest = hashlib.sha256()
    for row, vector in zip(rows, vectors, strict=True):
        digest.update(str(row["concept_id"]).encode("utf-8"))
        digest.update(struct.pack(f"<{len(vector)}d", *[float(value) for value in vector]))
    return digest.hexdigest()


def row_identity_update(digest: Any, row: Mapping[str, Any]) -> None:
    """Fold one row into the builder's row-order identity digest.

    The builder computes this in a separate preflight pass; a shard folds it into its single
    streaming pass so every rank can assert it saw the same row stream without paying for a
    second pass over 3.9M indexed SQLite lookups.
    """
    digest.update(
        json.dumps(
            [row["concept_id"], row["vocabulary_id"], row["concept_code"], row["concept_name"]],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def chunk_texts(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The builder's embedding text, verbatim. Divergence here would be silent."""
    return [" | ".join([str(row["concept_name"]), *row["synonyms"]]).strip() for row in rows]


def open_synonym_index(
    synonym_csv: Path, cache: Path, *, rebuild: bool = False
) -> sqlite3.Connection:
    """Open the shared synonym index, building it only when it is not already there.

    The index is a deterministic function of ``CONCEPT_SYNONYM.csv``, so ranks share one
    read-only copy instead of each paying an identical build.
    """
    if rebuild or not cache.exists():
        connection = _make_synonym_database(cache, synonym_csv)
        connection.close()
    return sqlite3.connect(f"file:{cache}?mode=ro", uri=True)


def run_shard(
    spec: ConceptStoreBuildSpec,
    connection: sqlite3.Connection,
    *,
    embedder: Embedder,
    chunk_store: ChunkStore,
    chunk_dir: Path,
    rank: int,
    world_size: int,
    max_chunks: int | None = None,
    on_chunk: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Embed this rank's chunks of the stream and return its evidence summary."""
    if world_size <= 0:
        raise ConceptStoreBuildError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ConceptStoreBuildError("rank must be in [0, world_size)")
    # Reuses the builder's own guard rails: chunk_rows stays a multiple of 32 and the batch
    # size stays 32, so a shard cannot quietly change the batching it exists to preserve.
    spec.validate()
    chunk_dir.mkdir(parents=True, exist_ok=True)

    row_identity = hashlib.sha256()
    streamed_rows = 0
    owned_chunks = 0
    owned_rows = 0
    reused_chunks = 0
    embed_seconds = 0.0
    dimension: int | None = None
    started = time.time()

    for chunk_index, rows in enumerate(_chunked(_iter_rows(spec, connection), spec.chunk_rows)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        for row in rows:
            row_identity_update(row_identity, row)
        streamed_rows += len(rows)
        if chunk_index % world_size != rank:
            continue

        chunk_path = chunk_dir / f"chunk-{chunk_index:08d}.arrow"
        if chunk_path.exists():
            # Chunks are published with os.replace, so a present file is a complete file.
            # Its row count is still re-checked against this stream rather than trusted - but
            # via the Arrow footer, not by reading the vectors back. Resuming a 953-chunk
            # build re-checks nearly a thousand chunks, and materializing ~4M doubles per
            # chunk to look at a length made that the slowest part of a resume by far.
            existing = chunk_store.row_count(chunk_path)
            if existing != len(rows):
                raise ConceptStoreBuildError(
                    f"existing chunk has {existing} rows, stream has {len(rows)}: {chunk_path}"
                )
            owned_chunks += 1
            reused_chunks += 1
            owned_rows += len(rows)
            continue

        started_chunk = time.time()
        vectors = [list(vector) for vector in embedder.embed_texts(chunk_texts(rows))]
        elapsed = time.time() - started_chunk
        embed_seconds += elapsed
        if len(vectors) != len(rows):
            raise ConceptStoreBuildError(
                f"embedder returned {len(vectors)} vectors for {len(rows)} rows in "
                f"chunk {chunk_index}"
            )
        widths = {len(vector) for vector in vectors}
        if len(widths) != 1:
            raise ConceptStoreBuildError(f"embedding dimension is not uniform: {widths}")
        width = next(iter(widths))
        if dimension is None:
            dimension = width
        elif width != dimension:
            raise ConceptStoreBuildError(
                f"embedding dimension is not uniform: {dimension} and {width}"
            )

        chunk_store.write(chunk_path, rows, vectors)
        owned_chunks += 1
        owned_rows += len(rows)
        if on_chunk is not None:
            on_chunk(
                {
                    "tool": SHARD_TOOL_VERSION,
                    "rank": rank,
                    "world_size": world_size,
                    "chunk_index": chunk_index,
                    "rows": len(rows),
                    "dimension": width,
                    "first_concept_id": str(rows[0]["concept_id"]),
                    "last_concept_id": str(rows[-1]["concept_id"]),
                    "chunk_vector_digest": chunk_vector_digest(rows, vectors),
                    "embed_seconds": round(elapsed, 4),
                    "rows_per_second": round(len(rows) / elapsed, 2) if elapsed else None,
                    "wall_offset_seconds": round(time.time() - started, 2),
                }
            )

    return {
        "tool": SHARD_TOOL_VERSION,
        "rank": rank,
        "world_size": world_size,
        "chunk_rows": spec.chunk_rows,
        "embedding_batch_size": spec.embedding_batch_size,
        "dimension": dimension,
        "row_order_identity": row_identity.hexdigest(),
        "streamed_rows": streamed_rows,
        "owned_chunks": owned_chunks,
        "reused_chunks": reused_chunks,
        "owned_rows": owned_rows,
        "embed_seconds": round(embed_seconds, 2),
        "wall_seconds": round(time.time() - started, 2),
        "owned_rows_per_second": (round(owned_rows / embed_seconds, 2) if embed_seconds else None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def seal_checkpoint(
    spec: ConceptStoreBuildSpec,
    connection: sqlite3.Connection,
    *,
    embedder: Embedder,
    chunk_store: ChunkStore,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Write the ``checkpoint.json`` that lets the single-GPU builder resume a sharded stream.

    The sharded pass writes chunks but no checkpoint, so the builder would re-embed every row
    rather than assemble the store from chunks that already exist.  This is the missing half:
    embed on N GPUs, seal, then let the untouched builder do the LanceDB assembly and every
    index exactly as it always does.

    The builder's resume is deliberately unforgiving - it recomputes the identity itself and
    rejects a checkpoint differing in any field, then re-derives the vector-stream digest from
    the chunk files.  So nothing here is hand-built: identity, payload and digest all come from
    the builder's own functions, and the result is handed back to the builder's own
    ``_validate_checkpoint`` before it is written.  A seal that would not survive a resume
    fails here rather than an hour into the assembly.
    """
    spec.validate()
    chunks = _chunk_paths(checkpoint_dir)
    if not chunks:
        raise ConceptStoreBuildError(f"no chunks found under {checkpoint_dir / 'chunks'}")
    expected = [
        checkpoint_dir / "chunks" / f"chunk-{index:08d}.arrow" for index in range(len(chunks))
    ]
    if chunks != expected:
        missing = next(
            (index for index, path in enumerate(expected) if not path.exists()), len(chunks)
        )
        raise ConceptStoreBuildError(
            f"the chunk stream is not a contiguous prefix; first missing index {missing}"
        )

    row_order, row_count, vocabulary_counts = _row_order_identity(spec, connection)
    identity = _identity(
        spec,
        source=_source_identity(spec),
        row_order=row_order,
        row_count=row_count,
        vocabulary_counts=vocabulary_counts,
        embedding=_embedding_contract(spec.config, embedder),
        commit=spec.trialmatchai_commit or _checkout_commit(spec.trialmatchai_root),
    )

    digest = _VectorStreamDigest()
    streamed = 0
    for path in chunks:
        rows, vectors = chunk_store.read(path)
        digest.update(rows, vectors)
        streamed += len(rows)
    if streamed != row_count:
        raise ConceptStoreBuildError(
            f"the chunk stream holds {streamed} rows but the sources produce {row_count}; "
            "the sharded pass is incomplete, so there is nothing to seal"
        )

    payload = _checkpoint_payload(identity, len(chunks), row_count, digest.hexdigest())
    payload["status"] = "complete"
    _validate_checkpoint(payload, identity, checkpoint_dir, chunk_store)
    _atomic_json_write(checkpoint_dir / "checkpoint.json", payload)
    return {
        "checkpoint": str(checkpoint_dir / "checkpoint.json"),
        "completed_chunks": len(chunks),
        "row_count": row_count,
        "rows_in_chunks": streamed,
        "vector_stream_digest": payload["vector_stream_digest"],
        "row_order_identity": row_order,
        "vocabulary_counts": vocabulary_counts,
        "validated": True,
    }


def row_digest(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Identify a store's rows by their text, whatever their order, without their vectors."""

    digests = sorted(
        hashlib.sha256(
            json.dumps(
                [row[field] for field in ROW_DIGEST_FIELDS],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest()
        for row in rows
    )
    total = hashlib.sha256()
    for digest in digests:
        total.update(digest)
    return {"count": len(digests), "digest": f"sha256:{total.hexdigest()}"}


def store_row_digest(db_path: Path, table_name: str = DEFAULT_TABLE_NAME) -> dict[str, Any]:
    """Digest a LanceDB store's rows by :data:`ROW_DIGEST_RULE`, never reading its vectors."""

    import lancedb

    table = lancedb.connect(str(db_path)).open_table(table_name)
    batches = table.search().select(list(ROW_DIGEST_FIELDS)).limit(None).to_batches(65536)
    return row_digest(row for batch in batches for row in batch.to_pylist())


def _download(url: str, destination: Path) -> None:
    """Stream ``url`` into ``destination`` through a temporary file."""

    if not url.startswith("https://"):
        raise ConceptStoreBuildError(f"dictionary sources are fetched over https only: {url}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    try:
        # The scheme is checked above; every URL comes from the pinned table or its base.
        with urllib.request.urlopen(url, timeout=1800) as response, temporary.open("wb") as sink:  # noqa: S310
            shutil.copyfileobj(response, sink, 1024 * 1024)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _fetch_pinned_source(
    pin: PinnedDictionarySource,
    raw_path: Path,
    *,
    doi: str | None,
    download: Callable[[str, Path], None],
) -> None:
    """Put the pinned raw file at ``raw_path``, from its source or the project's copy."""

    if pin.url is not None:
        url = pin.url
    elif doi and pin.published_name:
        url = zenodo_file_url(doi, pin.published_name)
    else:
        raise ConceptStoreBuildError(
            f"{pin.filename} at {pin.version}, {pin.sha256}, is no longer served by its source, "
            "and the project's archive of it has no DOI yet; place that file at "
            f"{raw_path} and run the build again"
        )
    if pin.published_sha256 is None:
        download(url, raw_path)
        return
    compressed = raw_path.with_name(f"{raw_path.name}.gz")
    try:
        download(url, compressed)
        if _sha256_file(compressed) != pin.published_sha256:
            raise ConceptStoreBuildError(
                f"{url} is not the published {pin.published_name} {pin.published_sha256}"
            )
        temporary = raw_path.with_name(f".{raw_path.name}.{os.getpid()}.part")
        try:
            with gzip.open(compressed, "rb") as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
            os.replace(temporary, raw_path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        compressed.unlink(missing_ok=True)


def pinned_open_dictionaries(
    work_dir: Path,
    open_sources: Mapping[str, Any],
    write_dictionary: Callable[[Any, Path, Path], int],
    *,
    force_download: bool = False,
    doi: str | None = PROJECT_DICTIONARY_DOI,
    download: Callable[[str, Path], None] = _download,
) -> tuple[DictionarySource, ...]:
    """Build TrialMatchAI's open dictionaries from the qualified store's pinned versions.

    Each raw file is used only when its SHA-256 is the pinned one, and is fetched only when it
    is absent or ``force_download`` asks. TrialMatchAI's own parser (``open_sources`` and
    ``write_dictionary``) turns it into the dictionary, which must then be the pinned one too.
    """

    if [pin.key for pin in PINNED_OPEN_DICTIONARIES] != list(open_sources):
        raise ConceptStoreBuildError(
            f"TrialMatchAI's open dictionary sources are not the pinned ones: {list(open_sources)}"
        )
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dictionaries: list[DictionarySource] = []
    for pin in PINNED_OPEN_DICTIONARIES:
        source = open_sources[pin.key]
        if source.filename != pin.filename:
            raise ConceptStoreBuildError(
                f"TrialMatchAI reads {pin.key} from {source.filename}, not {pin.filename}"
            )
        raw_path = raw_dir / pin.filename
        if force_download or not raw_path.is_file():
            _fetch_pinned_source(pin, raw_path, doi=doi, download=download)
        if _sha256_file(raw_path) != pin.sha256:
            raise ConceptStoreBuildError(
                f"{raw_path} is not {pin.version}: expected {pin.sha256}; delete it or run with "
                "--force-download"
            )
        dictionary_path = work_dir / source.dict_filename
        write_dictionary(source, raw_path, dictionary_path)
        dictionary_key = f"dictionary:{source.vocabulary_id}:{dictionary_path.name}"
        if dictionary_key != pin.dictionary or _sha256_file(dictionary_path) != (
            pin.dictionary_sha256
        ):
            raise ConceptStoreBuildError(
                f"TrialMatchAI's parser did not turn {pin.filename} into the pinned "
                f"{pin.dictionary} {pin.dictionary_sha256}"
            )
        dictionaries.append(
            DictionarySource(source.vocabulary_id, source.domain_id, dictionary_path.resolve())
        )
    return tuple(dictionaries)


def _dictionary_spec(value: str) -> DictionarySource:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError("--dictionary must use VOCABULARY:DOMAIN:PATH")
    return DictionarySource(parts[0], parts[1], Path(parts[2]).expanduser().resolve())


def main(argv: Sequence[str] | None = None) -> int:
    """Build the store in the pinned checkout's interpreter and print its summary.

    ``trial-benchmark patient-to-trial trialmatchai build-concept-store`` runs this with the
    qualified recipe; the options beyond that recipe serve a resumed or re-indexed build.
    """

    parser = argparse.ArgumentParser(
        description="Build the pinned TrialMatchAI concept store with bounded memory."
    )
    parser.add_argument("--config", default=None, help="TrialMatchAI config JSON")
    parser.add_argument("--concept-csv", required=True, type=Path)
    parser.add_argument("--synonym-csv", required=True, type=Path)
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--trialmatchai-root", required=True, type=Path)
    parser.add_argument("--trialmatchai-commit", default=None)
    parser.add_argument(
        "--checkpoint-trialmatchai-commit",
        default=None,
        help=(
            "TrialMatchAI commit recorded by an existing vector-stream checkpoint; "
            "omit for new embedding work"
        ),
    )
    parser.add_argument(
        "--expected-trialmatchai-commit",
        default=None,
        help="refuse a checkout at any other commit",
    )
    parser.add_argument("--source-release", default="Athena v20260227")
    parser.add_argument("--vocabulary", action="append", default=None)
    parser.add_argument("--dictionary", action="append", type=_dictionary_spec, default=[])
    parser.add_argument(
        "--sources",
        choices=["open"],
        default=None,
        help="Build TrialMatchAI's open dictionaries from their pinned versions before streaming "
        "them",
    )
    parser.add_argument("--concept-cache", type=Path, default=Path("data/concept_dicts"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        "--vector-index-partitions",
        type=int,
        default=DEFAULT_VECTOR_INDEX_PARTITIONS,
    )
    parser.add_argument(
        "--vector-index-sub-vectors",
        type=int,
        default=DEFAULT_VECTOR_INDEX_SUB_VECTORS,
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        from trialmatchai.config.config_loader import load_config
    except ImportError as exc:
        parser.error(
            "run this builder with the pinned TrialMatchAI checkout's own environment: "
            "trial-benchmark patient-to-trial trialmatchai build-concept-store"
        )
        raise AssertionError from exc

    config = load_config(args.config)
    if args.model_revision:
        config.setdefault("embedder", {})["revision"] = args.model_revision

    dictionary_sources = list(args.dictionary)
    if args.sources == "open":
        from trialmatchai.entities.concept_sources import OPEN_SOURCES, write_dictionary

        try:
            dictionary_sources.extend(
                pinned_open_dictionaries(
                    args.concept_cache.resolve(),
                    OPEN_SOURCES,
                    write_dictionary,
                    force_download=args.force_download,
                )
            )
        except ConceptStoreBuildError as exc:
            parser.error(str(exc))

    vocabularies = tuple(args.vocabulary or DEFAULT_VOCABULARIES)
    spec = ConceptStoreBuildSpec(
        concept_csv=args.concept_csv.resolve(),
        synonym_csv=args.synonym_csv.resolve(),
        db_path=args.db_path.resolve(),
        trialmatchai_root=args.trialmatchai_root.resolve(),
        config=config,
        vocabularies=vocabularies,
        dictionary_sources=tuple(dictionary_sources),
        source_release=args.source_release,
        chunk_rows=args.chunk_rows,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        vector_index_partitions=args.vector_index_partitions,
        vector_index_sub_vectors=args.vector_index_sub_vectors,
        checkpoint_dir=args.checkpoint_dir.resolve() if args.checkpoint_dir else None,
        force=args.force,
        trialmatchai_commit=args.trialmatchai_commit,
        checkpoint_trialmatchai_commit=args.checkpoint_trialmatchai_commit,
        expected_trialmatchai_commit=args.expected_trialmatchai_commit,
    )

    try:
        with GpuSampler() as gpu_sampler:
            manifest = build_concept_store(spec, gpu_utilization=gpu_sampler.as_dict())
            gpu_metrics = gpu_sampler.as_dict()
        manifest["build"]["gpu_utilization"] = gpu_metrics
        manifest_path = spec.db_path / CONCEPT_STORE_MANIFEST_FILENAME
        _atomic_json_write(manifest_path, manifest)
    except ConceptStoreBuildError as exc:
        parser.error(str(exc))
        return 2

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "row_count": manifest["store"]["row_count"],
                "vocabularies": manifest["vocabularies"],
                "vector_stream_digest": manifest["concept_state"]["vector_stream_digest"],
                "peak_rss_bytes": manifest["build"]["peak_rss_bytes"],
                "concepts_per_second": manifest["build"]["concepts_per_second"],
                "gpu_utilization": manifest["build"]["gpu_utilization"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def row_digest_main(argv: Sequence[str] | None = None) -> int:
    """Print a built store's row digest; a run reads it to accept an outside build."""

    parser = argparse.ArgumentParser(
        prog="concept_store.py row-digest",
        description="Digest a concept store's rows by " + ROW_DIGEST_RULE,
    )
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--table", default=DEFAULT_TABLE_NAME)
    args = parser.parse_args(argv)
    print(json.dumps(store_row_digest(args.db_path, args.table), sort_keys=True))
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["row-digest"]:
        raise SystemExit(row_digest_main(sys.argv[2:]))
    raise SystemExit(main())
