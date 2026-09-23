"""Minimal stdlib-only external pipeline. Reads only the exported judgment-free input."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def main() -> None:
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    payload = json.loads(source.read_text(encoding="utf-8"))
    native = payload["input"]
    forward = payload["task"] == "patient_to_trial"
    patients = native["query_patient_versions" if forward else "patient_versions"]
    trials = native["trial_corpus_versions" if forward else "trial_versions"]
    patient_text = {row["patient_id"]: tokens(row["topic"]["canonical_text"]) for row in patients}
    trial_text = {row["trial_id"]: tokens(row["trial"]["canonical_text"]) for row in trials}
    queries, documents = (patient_text, trial_text) if forward else (trial_text, patient_text)
    query_key, document_key = ("topic_id", "trial_id") if forward else ("trial_id", "patient_id")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "ranking.jsonl").open("w", encoding="utf-8") as stream:
        for query, words in sorted(queries.items()):
            scored = sorted(
                ((document, len(words & text)) for document, text in documents.items()),
                key=lambda row: (-row[1], row[0]),
            )
            for rank, (document, score) in enumerate(scored[: payload["top_k"]], start=1):
                row = {
                    "task": payload["task"],
                    "task_input_id": native["task_input_id"],
                    "system_id": "external:token-overlap",
                    query_key: query,
                    document_key: document,
                    "rank": rank,
                    "score": score,
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    directory = Path(__file__).parent

    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = {
        "system_id": "external:token-overlap",
        "source_revision": "example-v1",
        "source_sha256": digest(Path(__file__)),
        "dependency_lock_sha256": digest(directory / "requirements.lock"),
        "configuration": {
            "python": sys.version,
            "score": "unique lowercase token intersection",
            "ties": "candidate ID ascending",
        },
    }
    (output / "pipeline.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
