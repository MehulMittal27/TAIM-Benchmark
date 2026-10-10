"""Write the manifest of a TrialMatchAI v0.9.1 open-only concept store, verified against the
receipt of the source snapshot it was built from.

``trial-benchmark patient-to-trial trialmatchai verify-concept-store`` runs this module by path in
the pinned checkout's own environment, because it reads the store with ``lancedb``. It imports
nothing from TAIM. Run it after ``trialmatchai build-concepts --sources open`` has written the
store from the snapshotted sources.

It checks the six raw source files and their converted dictionaries against the receipt, then the
store's layout against the tag's recipe (one ``concepts`` table, a full-text index on ``fts_text``
and nothing else, 1,024-wide embeddings), and its per-vocabulary row counts against the receipt.
It records the store's content identity three ways: a digest over the rows' text fields in table
order, a digest over the embeddings as little-endian float32 in the same order, and a per-file
SHA-256 manifest of the store, leaving out a manifest already bound at its root. Any disagreement
is written into the output and the exit status is non-zero.

The output is the manifest a run binds: written as ``concept-store-manifest.json`` at the store's
root, its SHA-256 is the System's pinned ``expected_concept_store_manifest_sha256``. Rows are read
in slices, so memory stays near the size of the table's Arrow buffers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TEXT_FIELDS = (
    "concept_id",
    "vocabulary_id",
    "concept_code",
    "concept_name",
    "domain_id",
    "concept_class_id",
    "standard_concept",
    "synonyms",
    "fts_text",
)
EMBEDDING_DIMENSION = 1024
#: The name a run reads the bound manifest under, at the store's root.
MANIFEST_FILENAME = "concept-store-manifest.json"
_SLICE_ROWS = 50_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def file_manifest(root: Path) -> list[tuple[str, int, str]]:
    bound = root / MANIFEST_FILENAME
    return [
        (path.relative_to(root).as_posix(), path.stat().st_size, sha256_file(path))
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != bound
    ]


def verified_manifest(receipt_path: Path, concept_cache: Path, store: Path) -> dict[str, object]:
    """The store's manifest, with every disagreement with the receipt under ``failures``."""

    import lancedb
    import numpy as np

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    sources = []
    for source in receipt["sources"]:
        raw = concept_cache / "raw" / source["filename"]
        dictionary = concept_cache / f"dict_{source['domain_id']}.txt"
        raw_sha256 = sha256_file(raw) if raw.is_file() else None
        dict_sha256 = sha256_file(dictionary) if dictionary.is_file() else None
        dict_rows = (
            sum(1 for _ in dictionary.open(encoding="utf-8")) if dictionary.is_file() else None
        )
        if raw_sha256 != source["sha256"]:
            failures.append(f"raw {source['filename']}: {raw_sha256} != {source['sha256']}")
        expected = source["converted_dictionary"]
        if dict_sha256 != expected["sha256"] or dict_rows != expected["rows"]:
            failures.append(
                f"dictionary {dictionary.name}: {dict_rows} rows {dict_sha256}, "
                f"receipt {expected['rows']} rows {expected['sha256']}"
            )
        sources.append(
            {
                "source": source["source"],
                "raw_sha256": raw_sha256,
                "dictionary": dictionary.name,
                "dictionary_rows": dict_rows,
                "dictionary_sha256": dict_sha256,
            }
        )

    db = lancedb.connect(str(store))
    tables = sorted(db.table_names())
    if tables != ["concepts"]:
        failures.append(f"tables: {tables} != ['concepts']")
    table = db.open_table("concepts")
    schema = {field.name: str(field.type) for field in table.schema}
    for name in (*TEXT_FIELDS, "embedding"):
        if name not in schema:
            failures.append(f"schema lacks column {name}")
    indices = [
        {"name": index.name, "type": str(index.index_type), "columns": list(index.columns)}
        for index in table.list_indices()
    ]
    if [(item["columns"], item["type"]) for item in indices] != [(["fts_text"], "FTS")]:
        failures.append(f"indices {indices} are not exactly one FTS index on fts_text")

    data = table.to_arrow()
    rows = data.num_rows
    if rows != receipt["total_rows"]:
        failures.append(f"row count {rows} != receipt {receipt['total_rows']}")
    vocabulary_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    text_digest = hashlib.sha256()
    vector_digest = hashlib.sha256()
    values = 0
    finite = True
    norm_min: float | None = None
    norm_max: float | None = None
    for offset in range(0, rows, _SLICE_ROWS):
        part = data.slice(offset, _SLICE_ROWS)
        columns = {name: part.column(name).to_pylist() for name in TEXT_FIELDS}
        for vocabulary, domain in zip(columns["vocabulary_id"], columns["domain_id"], strict=True):
            vocabulary_counts[vocabulary] = vocabulary_counts.get(vocabulary, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        for index in range(part.num_rows):
            line = json.dumps(
                [columns[name][index] for name in TEXT_FIELDS],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            text_digest.update(line.encode("utf-8") + b"\n")
        del columns
        embedding = part.column("embedding").combine_chunks()
        flat = np.asarray(embedding.flatten().to_numpy(zero_copy_only=False), dtype="<f4")
        values += flat.size
        finite = finite and bool(np.isfinite(flat).all())
        vector_digest.update(flat.tobytes())
        if flat.size == part.num_rows * EMBEDDING_DIMENSION and part.num_rows:
            norms = np.linalg.norm(flat.reshape(-1, EMBEDDING_DIMENSION), axis=1)
            low, high = float(norms.min()), float(norms.max())
            norm_min = low if norm_min is None else min(norm_min, low)
            norm_max = high if norm_max is None else max(norm_max, high)
    expected_vocabularies = {
        source["vocabulary_id"]: source["converted_dictionary"]["rows"]
        for source in receipt["sources"]
    }
    if vocabulary_counts != expected_vocabularies:
        failures.append(f"vocabulary counts {vocabulary_counts} != {expected_vocabularies}")
    if values != rows * EMBEDDING_DIMENSION:
        failures.append(f"embedding values {values} != {rows} x {EMBEDDING_DIMENSION}")
    if not finite:
        failures.append("embedding holds non-finite values")

    state_path = store / ".concepts_state.json"
    listed = file_manifest(store)
    manifest = [list(row) for row in listed]
    manifest_digest = hashlib.sha256(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in manifest).encode()
    ).hexdigest()
    return {
        "kind": "trialmatchai-upstream-open-concept-store-verification",
        "schema_version": "1",
        "receipt_sha256": sha256_file(receipt_path),
        "sources": sources,
        "tables": tables,
        "schema": schema,
        "indices": indices,
        "rows": rows,
        "vocabulary_counts": dict(sorted(vocabulary_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "embedding": {
            "dimension": EMBEDDING_DIMENSION,
            "finite": finite,
            "norm_min": norm_min,
            "norm_max": norm_max,
        },
        "content_identity": {
            "text_rows_sha256": f"sha256:{text_digest.hexdigest()}",
            "text_rows_rule": (
                "SHA-256 over one compact JSON array per row, in table order, of "
                + ", ".join(TEXT_FIELDS)
            ),
            "embedding_float32_sha256": f"sha256:{vector_digest.hexdigest()}",
            "embedding_rule": (
                "SHA-256 over every embedding as little-endian float32, in table order"
            ),
            "concepts_state": (
                json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
            ),
            "file_manifest_sha256": f"sha256:{manifest_digest}",
            "file_manifest_rule": (
                "SHA-256 over one compact JSON [path, bytes, sha256] line per file"
            ),
            "files": len(manifest),
            "bytes": sum(size for _path, size, _sha256 in listed),
        },
        "file_manifest": manifest,
        "failures": failures,
        "verified": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--concept-cache", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = verified_manifest(args.receipt, args.concept_cache, args.store)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: report[key] for key in ("rows", "vocabulary_counts", "indices", "failures")}
        )
    )
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
