"""The TrialMatchAI System pinned directly to upstream ``cbib/TrialMatchAI`` at tag ``v0.9.1``.

Their code is not changed. The run executes the tag's own TREC track steps - the funnel preset,
topic importer, query expansion, index and matching of ``trialmatchai.trec.runner.run_tracks`` - by
calling their functions from a small TAIM driver in their environment. The driver replaces only
what ``trialmatchai trec`` would fetch for itself: TAIM supplies the topics, the pool (the prepared
corpus and its search database) and, afterwards, the evaluation. Everything else is theirs,
including what their TREC path does not do: it runs no entity recognition on the patient note and
no demographic extraction beyond its regular expressions (``trec/topics.py:196-247``).

What TAIM adds is integration only:

- the adapter, which verifies the pinned checkout and every input before launch;
- a TAIM-owned config, the tag's ``config.json`` with only the four model revision fields set, so
  that the pins bind
  (``taim/pipelines/trialmatchai_v091/data/trialmatchai-upstream-v091-config.json``);
- the two LoRA adapters, which the tag's loader reads as local paths
  (``config_loader.py:213-216``), staged as local snapshot directories and pointed at through the
  tag's own ``TRIALMATCHAI_MODEL_*_ADAPTER_PATH`` overrides;
- the six model revisions, pinned by TAIM and verified in the supplied Hugging Face cache;
- evidence that concept linking ran, since the tag's linker falls back silently (see
  :func:`trialmatchai_upstream_linking_evidence`).

This module imports nothing from the checkout; upstream runs as a subprocess.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from taim.artifacts import sha256_file
from taim.patient_to_trial import StrictSystemOptions
from taim.pipelines.trialmatchai.corpus import (
    TRIALMATCHAI_HARNESS_OPTIONS,
    TRIALMATCHAI_PREPARED_CORPUS_OPTION,
    TrialMatchAIAdapterError,
    TrialMatchAIHarnessInputs,
    corpus_filter_manifest_sha256,
    load_and_normalize_ranked_trials,
    trialmatchai_checkout_configuration,
    validate_snapshot_corpus,
    without_locators,
)
from taim.schemas import PRIMARY_RANKING_CANDIDATES, Candidate, StageRanking
from taim.snapshot import CAPABILITY_CANONICAL_PATIENT_TEXT, BenchmarkTopic

if TYPE_CHECKING:
    import argparse

    from taim.patient_to_trial import (
        PatientToTrialRunRequest,
        PatientToTrialRunResult,
        ResolvedBenchmarkContract,
    )

TRIALMATCHAI_UPSTREAM_REPOSITORY = "https://github.com/cbib/TrialMatchAI.git"
#: Tag ``v0.9.1``. The tag is lightweight, so the commit is the pin, never the tag name.
TRIALMATCHAI_UPSTREAM_COMMIT = "3bc0e5f44d742ef069f8ef73bbc4187573d9db6a"
TRIALMATCHAI_UPSTREAM_CONFIG_RESOURCE = "trialmatchai-upstream-v091-config.json"
TRIALMATCHAI_UPSTREAM_CONFIG_SHA256 = (
    "sha256:3d6a1540f7a1ab0bcaa1e77bcbc23030fa4bcab7f12dcbe356741275cd7b571b"
)
TRIALMATCHAI_UPSTREAM_TOPIC_RENDERER_VERSION = "trialmatchai-upstream-trec-topic-text-v1"
TRIALMATCHAI_UPSTREAM_PROCESS_EXECUTION_MODE = "trec_track_batch"
TRIALMATCHAI_UPSTREAM_REQUIRED_ENVIRONMENT = (
    ("HF_HUB_OFFLINE", "1"),
    # Their embedder initialises CUDA before the vLLM reranker worker starts.
    ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
)
#: The one TrialMatchAI variable an operator supplies: where the bound concept store is.
TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_VARIABLE = "TRIALMATCHAI_CONCEPT_DB_PATH"
#: The Hugging Face hub cache that holds the six pinned snapshots, supplied by the operator.
TRIALMATCHAI_UPSTREAM_MODEL_CACHE_VARIABLE = "HF_HUB_CACHE"
TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_MANIFEST = "concept-store-manifest.json"
#: The bound concept store: the tag's open-only recipe, built from the 2026-09-26 source snapshot on
#: Delta (job 22448052) and verified against its receipt; the manifest is that verification.
TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_MANIFEST_SHA256 = (
    "sha256:1cd46b8634f3d01b02991940effe4e5d50ccc825b9f75245be56915d0ea556b3"
)

#: The six models on the tag's path, with the revisions TAIM pins. The tag's config names every
#: one with a ``null`` revision, so upstream resolves each repository's ``main`` at load time. The
#: pins are TAIM's decision, not upstream's: each is the revision whose model files are identical,
#: file for file, to what ``main`` served when v0.9.1 was released on 2026-09-19.
TRIALMATCHAI_UPSTREAM_MODEL_PINS = (
    ("entity_extraction", "fastino/gliner2-base-v1", "f5b2ecedebe4381b088c1cf276f5bf72a52cac54"),
    ("embedder", "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
    ("eligibility_base", "microsoft/phi-4", "2db69c1c3e91a05d2c64a3185acfbaf36f744e25"),
    (
        "eligibility_adapter",
        "majdabd33/trialmatchai-phi4-reasoning-lora",
        "9eaddaf048c8d4266a291884ff89db1cf05b07fc",
    ),
    ("reranker_base", "google/gemma-2-2b-it", "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"),
    (
        "reranker_adapter",
        "majdabd33/trialmatchai-gemma2-reranker-lora",
        "3118ba76d545f71f3aaa7952d2f031ad5fdef8c9",
    ),
)
#: Every file of the three snapshots whose revision no loader on the tag's path binds: the two LoRA
#: adapters, which upstream reads as local directories, and gliner2, whose ``from_pretrained``
#: ignores the revision it is passed (gliner2 1.3.1 ``model.py:675-707``). These are verified by
#: content at every run start; the other three load at their pinned revision.
TRIALMATCHAI_UPSTREAM_SNAPSHOT_FILES: Mapping[str, Mapping[str, str]] = {
    "fastino/gliner2-base-v1": {
        ".gitattributes": "sha256:8b418b78653e5c4d418c0840e9c3664601428c098b77b11e463efcad19acce46",
        "README.md": "sha256:b954086c69ecabf1b4bec0949842a3576cd71a7cc9b788095f7dfc39546bdf48",
        "added_tokens.json": (
            "sha256:3c274ca559be8b7b18febf616724ec05f8a8abccd428bd2744608f45f7a0b067"
        ),
        "config.json": "sha256:33d8aa009deed4b5435d162e08b5a720b252a0084e59d30d366b37da8c0193c5",
        "encoder_config/config.json": (
            "sha256:9840a4db70bc007e6b65d336ebe2bddc53bc2ce210dc5757e50d5bb17122f7cd"
        ),
        "image/GitHub.png": (
            "sha256:c945cbd7b6178aa0e80ed2f10382d6edf59799e3e0dd9b24e3e8c5fe58c02deb"
        ),
        "model.safetensors": (
            "sha256:845fc4bd93c525b86124c58ab4f56c9eacf8587953086b14c501fab25957c007"
        ),
        "special_tokens_map.json": (
            "sha256:84ea70143f533d7e99b393d87f20010887a9ac2cba955828ef313886e4e83f4f"
        ),
        "spm.model": "sha256:c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
        "tokenizer.json": "sha256:1b7fbabfb4c690bed84c6793bfecae9b8dfe205751b04f9ffd1e76a1e7df9c16",
        "tokenizer_config.json": (
            "sha256:4e6c6a8d26478b670692916266438485403b0c54af31fd1a12bab8a465d062e9"
        ),
    },
    "majdabd33/trialmatchai-phi4-reasoning-lora": {
        ".gitattributes": "sha256:11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
        "README.md": "sha256:cf59920d9e5d9f803ca671850cdafff3782bcc1843aee3a5eba73ecac9ff8a13",
        "adapter_config.json": (
            "sha256:516034e07744cca4d89c94366ed3ff1c8ea5b99607bd17e0486f5547bcb6ce59"
        ),
        "adapter_model.safetensors": (
            "sha256:f9665b6540cbab908cd531e343a8d1d330b1a9b0f8bf4a83c1a9b39250e50b15"
        ),
        "merges.txt": "sha256:b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502",
        "special_tokens_map.json": (
            "sha256:49fb71eef3a959cf088a17dfdfb78a40affbb576e44b585a44a051bd1f3101ff"
        ),
        "tokenizer.json": "sha256:00fc41ba94dde84336d4a1cb485a6f92b293fc4e11a7d7679cb1bd849ca5698c",
        "tokenizer_config.json": (
            "sha256:28c83e72dea519802ebef093d3340a109f892d4e15578f441f48099b253c8c43"
        ),
        "vocab.json": "sha256:741943243bd0035a8082ef0ccc09ef62e7ce7ec2f2f053aeedd808dba1c2ffc5",
    },
    "majdabd33/trialmatchai-gemma2-reranker-lora": {
        ".gitattributes": "sha256:34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
        "README.md": "sha256:a4ee6f7662af78cf2be377fbcbdf8e538fcb5158d8d0ea37f3b6a58e0e758741",
        "adapter_config.json": (
            "sha256:b8ee19f6e35e2eafc769b932babfb4d5b1b48638ade8f96270186773019527b1"
        ),
        "adapter_model.safetensors": (
            "sha256:851aeb9f82dbe40a797b1bc085381ccb8aec6ab9d4c3d2c4ba1371356daa770a"
        ),
        "special_tokens_map.json": (
            "sha256:f3719b6d3c201e2dc1ddfa18c167d04d66e9f67379ed1080d538cea3ffa24b53"
        ),
        "tokenizer.json": "sha256:e6ce83119bb404f7f0a6e621b76759d476357dcd01241a90f9ca136ae2b3c11c",
        "tokenizer.model": (
            "sha256:61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"
        ),
        "tokenizer_config.json": (
            "sha256:bfe1e06768278106b39fb53c8f8285b673fd3d4d0bd9334216a8221e80953ee5"
        ),
    },
}
#: The environment overrides that point the tag's config at the staged adapter directories
#: (``settings.py:352,364``). Their config keeps the Hub identifiers it ships with.
_ADAPTER_PATH_VARIABLES = (
    ("TRIALMATCHAI_MODEL_COT_ADAPTER_PATH", "majdabd33/trialmatchai-phi4-reasoning-lora"),
    ("TRIALMATCHAI_MODEL_RERANKER_ADAPTER_PATH", "majdabd33/trialmatchai-gemma2-reranker-lora"),
)
#: The funnel the tag's TREC path applies, ``_track_config`` in ``trec/runner.py:34-55``. The driver
#: calls that function; this record says what it sets, and a test holds it to the tag's bytes.
TRIALMATCHAI_UPSTREAM_TREC_PRESET = {
    "search.max_trials_first_level": 1000,
    "search.max_trials_second_level": 500,
    "search.second_level_keep_divisor": 1,
    "rag.max_trials_rag": 250,
    "query_expansion.enabled": True,
    "reporting.emit_html": False,
}
#: Why this System binds the open-only store, recorded with every run.
TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_RULING = (
    "Captain's ruling 2026-09-26 (S2): the tag's default recipe, `trialmatchai build --concepts`, "
    "which builds from six open sources and reads no SNOMED. The store behind upstream's "
    "published TREC figures could not be settled with certainty: their run configurations were "
    "not stored (docs/trec-evaluation.md:34-38). The evidence leans open-only - the author "
    "described 'a 600k-concept store' (f2167af, 2026-07-01) and 'the open-vocabulary store "
    "actually built' (a2d2423), against 616,925 rows measured for the open-only recipe - and by "
    "the captain's rule the store that matches their figures is the store as they ship it. The "
    "procedure and test coverage the open sources lack is theirs and stays."
)

#: Upstream's own log lines that mark concept linking degrading instead of running. Any one fails
#: the run: ``annotator.py:103-121`` (no store), ``linker.py:137-139,155-157`` (a query failed),
#: ``matching/retrieval/synonyms.py:15-31`` (no annotator, or extraction failed).
_LINKING_FALLBACK_MESSAGES = (
    (
        "trialmatchai.entities.annotator",
        "Concept linker enabled but concept_linker.db_path is empty",
    ),
    ("trialmatchai.entities.annotator", "Concept DB path does not exist; linking will degrade"),
    ("trialmatchai.entities.annotator", "Concept DB unavailable; linking will degrade"),
    ("trialmatchai.entities.linker", "LanceDB FTS concept search failed"),
    ("trialmatchai.entities.linker", "LanceDB vector concept search failed"),
    ("trialmatchai.matching.retrieval.synonyms", "Entity annotator disabled"),
    ("trialmatchai.matching.retrieval.synonyms", "Entity synonym extraction failed"),
)

#: The TAIM driver that runs in the pinned checkout's environment. It performs the steps of
#: ``trialmatchai.trec.runner.run_tracks`` for one track, calling their functions in their order,
#: with TAIM's topics in place of their topic download and with no qrels or evaluation step.
_TREC_TRACK_DRIVER = '''\
"""TAIM driver: the tag's TREC track steps over TAIM's topics, pool and evaluator."""

import argparse
import json
import sys
from pathlib import Path

from trialmatchai.config.config_loader import load_config
from trialmatchai.interop.exporters import profile_to_matching_summary
from trialmatchai.orchestration import build_index, expand_queries, free_models, run_matching
from trialmatchai.trec.corpus import TrackSpec
from trialmatchai.trec.runner import _track_config
from trialmatchai.trec.topics import _clean, build_profile_from_topic


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "config", "topics", "track-root", "search-db", "processed-trials",
        "processed-criteria", "track-key", "id-prefix",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    root = Path(args.track_root)
    spec = TrackSpec(
        key=args.track_key,
        id_prefix=args.id_prefix,
        trec_dir=root / "trec",
        profile_dir=root / "profiles",
        summary_dir=root / "summaries",
        db_path=Path(args.search_db),
        output_dir=root / "outputs",
    )
    cfg = _track_config(load_config(args.config), spec)

    # 1) import_topics' write loop (trec/topics.py:314-321), over TAIM's topic text.
    spec.profile_dir.mkdir(parents=True, exist_ok=True)
    spec.summary_dir.mkdir(parents=True, exist_ok=True)
    topics = json.loads(Path(args.topics).read_text(encoding="utf-8"))
    for patient_id, raw_text in topics.items():
        profile = build_profile_from_topic(patient_id, _clean(raw_text))
        (spec.profile_dir / f"{patient_id}.json").write_text(
            profile.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
        )
        (spec.summary_dir / f"{patient_id}.json").write_text(
            json.dumps(profile_to_matching_summary(profile), indent=2), encoding="utf-8"
        )
    # 3) runtime query expansion, then free its engine before matching loads its own.
    expand_queries(cfg, force=False)
    free_models()
    # 4) the index; their build_index skips when the tables are present.
    build_index(
        cfg,
        processed_trials_folder=args.processed_trials,
        processed_criteria_folder=args.processed_criteria,
        nct_filter=None,
        force=False,
    )
    # 5) matching with per-patient resume.
    code = run_matching(cfg, resume=True, force=False)
    free_models()
    return code


if __name__ == "__main__":
    sys.exit(main())
'''
TRIALMATCHAI_UPSTREAM_DRIVER_SHA256 = (
    f"sha256:{hashlib.sha256(_TREC_TRACK_DRIVER.encode()).hexdigest()}"
)


def trialmatchai_upstream_runtime_contract() -> dict[str, object]:
    """Return the exact execution settings bound to the System Input."""

    return {
        "environment": dict(TRIALMATCHAI_UPSTREAM_REQUIRED_ENVIRONMENT),
        "process_execution_mode": TRIALMATCHAI_UPSTREAM_PROCESS_EXECUTION_MODE,
        "driver_sha256": TRIALMATCHAI_UPSTREAM_DRIVER_SHA256,
    }


def trialmatchai_upstream_config_bytes() -> bytes:
    """The TAIM-owned config, refused unless it is the pinned bytes."""

    payload = (
        files("taim")
        .joinpath("pipelines", "trialmatchai_v091", "data", TRIALMATCHAI_UPSTREAM_CONFIG_RESOURCE)
        .read_bytes()
    )
    actual = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if actual != TRIALMATCHAI_UPSTREAM_CONFIG_SHA256:
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI v0.9.1 config expected {TRIALMATCHAI_UPSTREAM_CONFIG_SHA256}, "
            f"found {actual}"
        )
    return payload


def _snapshot_directory(cache: Path, repository: str, revision: str) -> Path:
    return cache / ("models--" + repository.replace("/", "--")) / "snapshots" / revision


def _normalized_contract(contract: Mapping[str, object]) -> dict[str, object]:
    """A runtime contract compared indifferently to list-versus-tuple, as it arrives frozen."""

    return {
        key: dict(value)
        if isinstance(value, Mapping)
        else tuple(value)
        if isinstance(value, list | tuple)
        else value
        for key, value in contract.items()
    }


def _topic_text(topic: BenchmarkTopic) -> str:
    """The topic's canonical patient text, the only text their TREC importer is given."""

    text = topic.canonical_text if isinstance(topic, BenchmarkTopic) else None
    if not isinstance(text, str) or not text.strip():
        raise TrialMatchAIAdapterError(f"topic {getattr(topic, 'topic_id', topic)!r} has no text")
    return text


def verify_trialmatchai_upstream_model_cache(cache: Path) -> dict[str, dict[str, object]]:
    """Verify the six pinned models in a Hugging Face hub cache, before any model loads.

    Each repository's ``refs/main`` must name the pinned revision, because two loaders on the tag's
    path resolve ``main`` whatever revision they are given, and the pinned snapshot must be present.
    The three snapshots in :data:`TRIALMATCHAI_UPSTREAM_SNAPSHOT_FILES` must hold exactly the pinned
    files. Returns the record of what was verified, without locations.
    """

    record: dict[str, dict[str, object]] = {}
    for role, repository, revision in TRIALMATCHAI_UPSTREAM_MODEL_PINS:
        repository_dir = cache / ("models--" + repository.replace("/", "--"))
        refs_main = repository_dir / "refs" / "main"
        found = refs_main.read_text(encoding="utf-8").strip() if refs_main.is_file() else None
        if found != revision:
            raise TrialMatchAIAdapterError(
                f"model cache refs/main for {repository} names {found!r}, not the pinned "
                f"revision {revision}"
            )
        snapshot = _snapshot_directory(cache, repository, revision)
        if not snapshot.is_dir():
            raise TrialMatchAIAdapterError(
                f"model cache lacks {repository} at its pinned revision {revision}"
            )
        entry: dict[str, object] = {"repository": repository, "revision": revision}
        expected = TRIALMATCHAI_UPSTREAM_SNAPSHOT_FILES.get(repository)
        if expected is not None:
            actual = {
                path.relative_to(snapshot).as_posix(): sha256_file(path)
                for path in sorted(snapshot.rglob("*"))
                if path.is_file()
            }
            if actual != dict(expected):
                differing = sorted(
                    name
                    for name in set(actual) | set(expected)
                    if actual.get(name) != expected.get(name)
                )
                raise TrialMatchAIAdapterError(
                    f"model cache snapshot of {repository}@{revision} differs from its pinned "
                    f"files: {', '.join(differing)}"
                )
            entry["snapshot_files_sha256"] = dict(actual)
        record[role] = entry
    return record


def _concept_store_identity(db_path: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    """Bind the concept store by its TAIM manifest, and the store by every file it lists."""

    manifest_path = db_path / TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_MANIFEST
    if not manifest_path.is_file():
        raise TrialMatchAIAdapterError(f"concept store has no manifest: {manifest_path}")
    actual = sha256_file(manifest_path)
    if actual != expected_manifest_sha256:
        raise TrialMatchAIAdapterError(
            f"concept store manifest expected {expected_manifest_sha256}, found {actual}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("verified") is not True:
        raise TrialMatchAIAdapterError("concept store manifest does not record a verified store")
    listed = {str(path): (size, digest) for path, size, digest in manifest["file_manifest"]}
    present = {
        path.relative_to(db_path).as_posix(): path
        for path in sorted(db_path.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    if set(present) != set(listed):
        raise TrialMatchAIAdapterError(
            "concept store files differ from its manifest: "
            + ", ".join(sorted(set(present) ^ set(listed)))
        )
    for relative, path in present.items():
        size, digest = listed[relative]
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise TrialMatchAIAdapterError(f"concept store file changed: {relative}")
    return {
        "manifest_sha256": actual,
        "rows": manifest["rows"],
        "vocabulary_counts": manifest["vocabulary_counts"],
        "indices": manifest["indices"],
        "content_identity": manifest["content_identity"],
        "ruling": TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_RULING,
    }


def trialmatchai_upstream_linking_evidence(
    log_path: Path, plan_paths: Mapping[str, Path]
) -> dict[str, object]:
    """Evidence, per run, that the tag's concept linking ran rather than fell back.

    The tag degrades silently: a missing or unreadable store, a failed LanceDB query or a failed
    synonym extraction is logged and the run completes with a ranking. On the TREC path the linker
    runs at match time, turning each topic's conditions into ``concept_synonym`` query terms. So
    the run must show both: none of upstream's degradation lines in its JSON log, and at least one
    topic whose first-level query plan carries linker-sourced ``concept_synonym`` terms. Either
    failure raises; the evidence is returned for the run record otherwise.
    """

    lines = 0
    fallbacks: list[dict[str, str]] = []
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(entry, dict) or not str(entry.get("logger", "")).startswith(
            "trialmatchai"
        ):
            continue
        lines += 1
        message = str(entry.get("message", ""))
        for logger, prefix in _LINKING_FALLBACK_MESSAGES:
            if entry.get("logger") == logger and message.startswith(prefix):
                fallbacks.append({"logger": logger, "message": message[:500]})
    per_topic: dict[str, dict[str, object]] = {}
    for topic_id, plan_path in sorted(plan_paths.items()):
        plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
        channels = plan.get("channels", []) if isinstance(plan, Mapping) else []
        terms = [
            term
            for channel in channels
            if channel.get("kind") == "concept_synonym"
            and channel.get("source") == "concept_linker"
            for term in channel.get("terms", [])
        ]
        filters = plan.get("filters", {}) if isinstance(plan, Mapping) else {}
        per_topic[topic_id] = {
            "query_plan_present": plan is not None,
            "concept_synonym_terms": len(terms),
            "filters": {key: filters.get(key) for key in ("age", "sex", "overall_status")},
        }
    linked_topics = sum(1 for item in per_topic.values() if item["concept_synonym_terms"])
    evidence = {
        "rule": (
            "no upstream linking-degradation log line, and at least one topic with linker-sourced "
            "concept_synonym query terms"
        ),
        "upstream_log_lines": lines,
        "fallback_lines": fallbacks,
        "topics_with_concept_synonym_terms": linked_topics,
        "per_topic": per_topic,
    }
    if lines == 0:
        raise TrialMatchAIAdapterError(
            "TrialMatchAI wrote no JSON log line, so concept linking cannot be evidenced"
        )
    if fallbacks:
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI concept linking fell back in this run ({len(fallbacks)} line(s), first: "
            f"{fallbacks[0]['message']!r}); the ranking is not this System's"
        )
    if not linked_topics:
        raise TrialMatchAIAdapterError(
            "no topic's first-level query plan carries concept_synonym terms from the linker; "
            "concept linking cannot be shown to have run"
        )
    return evidence


def _database_listing(path: Path) -> list[tuple[str, int, int]]:
    return [
        (item.relative_to(path).as_posix(), item.stat().st_size, item.stat().st_mtime_ns)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    stdout: Path,
    stderr: Path,
) -> None:
    try:
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            # Trusted argv: the launcher tokens are the validated System Input command and the
            # rest are paths this adapter wrote; the list form never reaches a shell.
            completed = subprocess.run(  # noqa: S603
                list(command),
                cwd=cwd,
                env=dict(environment),
                stdout=out,
                stderr=err,
                check=False,
                timeout=timeout_seconds,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrialMatchAIAdapterError(f"TrialMatchAI v0.9.1 failed to run: {exc}") from exc
    if completed.returncode != 0:
        tail = stderr.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI v0.9.1 returned {completed.returncode}: {tail}"
        )


class TrialMatchAIUpstreamV091Base(StrictSystemOptions):
    """Upstream TrialMatchAI v0.9.1 on its TREC path; a release unit supplies the identity.

    Built with :class:`~taim.pipelines.trialmatchai.corpus.TrialMatchAIHarnessInputs`, as the
    release layer builds it, the System takes its locations and timeout from them and binds the
    prepared corpus by its receipt digest.
    """

    system_id: str
    #: The ``TrackSpec`` key and topic prefix the tag's TREC path uses for this identity's Track.
    trec_track_key: str
    trec_id_prefix: str
    #: The SHA-256 of the bound concept store's manifest; the System refuses to run without one.
    expected_concept_store_manifest_sha256: str | None = (
        TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_MANIFEST_SHA256
    )
    option_names = frozenset(
        {
            "command",
            "cwd",
            "workspace",
            "timeout_seconds",
            "processed_trials_folder",
            "processed_criteria_folder",
            "search_db_path",
            "trialmatchai_expected_commit",
            "trialmatchai_expected_repository",
            "trialmatchai_runtime_contract",
        }
    )
    required_capabilities = frozenset({CAPABILITY_CANONICAL_PATIENT_TEXT})
    optional_capabilities: frozenset[str] = frozenset()

    def __init__(self, harness_inputs: TrialMatchAIHarnessInputs | None = None) -> None:
        self.harness_inputs = harness_inputs
        if harness_inputs is not None:
            self.option_names = (type(self).option_names - TRIALMATCHAI_HARNESS_OPTIONS) | {
                TRIALMATCHAI_PREPARED_CORPUS_OPTION
            }

    def _locations(
        self, request: PatientToTrialRunRequest
    ) -> tuple[Path, Path, float, Path, Path, Path]:
        harness = self.harness_inputs
        if harness is None:
            options = request.options
            values = [
                options.get(name)
                for name in (
                    "cwd",
                    "workspace",
                    "processed_trials_folder",
                    "processed_criteria_folder",
                    "search_db_path",
                )
            ]
            if any(not isinstance(value, (str, Path)) for value in values):
                raise TrialMatchAIAdapterError(
                    "TrialMatchAI v0.9.1 requires cwd, workspace, processed_trials_folder, "
                    "processed_criteria_folder and search_db_path options"
                )
            cwd, workspace, trials, criteria, search_db = (Path(str(v)).resolve() for v in values)
            timeout = options.get("timeout_seconds")
        else:
            prepared = request.options.get(TRIALMATCHAI_PREPARED_CORPUS_OPTION)
            if not isinstance(prepared, str) or not prepared.startswith("sha256:"):
                raise TrialMatchAIAdapterError(
                    f"{TRIALMATCHAI_PREPARED_CORPUS_OPTION} must be a SHA-256 digest"
                )
            cwd = harness.cwd.resolve()
            workspace = harness.workspace.resolve()
            trials = harness.processed_trials_folder.resolve()
            criteria = harness.processed_criteria_folder.resolve()
            search_db = harness.search_db_path.resolve()
            timeout = harness.timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TrialMatchAIAdapterError("TrialMatchAI v0.9.1 timeout must be a number")
        if not math.isfinite(timeout) or timeout <= 0:
            raise TrialMatchAIAdapterError("TrialMatchAI v0.9.1 timeout must be positive")
        return cwd, workspace, float(timeout), trials, criteria, search_db

    def _checked_environment(self, request: PatientToTrialRunRequest) -> dict[str, str]:
        contract = request.options.get("trialmatchai_runtime_contract")
        expected = trialmatchai_upstream_runtime_contract()
        if not isinstance(contract, Mapping) or _normalized_contract(
            contract
        ) != _normalized_contract(expected):
            raise TrialMatchAIAdapterError(
                f"{self.system_id} requires exact trialmatchai_runtime_contract {expected}"
            )
        environment = dict(os.environ)
        # TAIM runs from its own uv environment; the checkout has a separate one.
        environment.pop("VIRTUAL_ENV", None)
        for name, value in TRIALMATCHAI_UPSTREAM_REQUIRED_ENVIRONMENT:
            if environment.get(name) != value:
                raise TrialMatchAIAdapterError(
                    f"{self.system_id} requires {name}={value}, found {environment.get(name)!r}"
                )
        # Their config is theirs as verified: no inherited variable may rewrite it after the
        # digest check. The concept store location is the one TrialMatchAI variable an operator
        # supplies; the adapter sets the rest itself.
        inherited = sorted(
            name
            for name in environment
            if name.startswith("TRIALMATCHAI_")
            and name != TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_VARIABLE
        )
        if inherited:
            raise TrialMatchAIAdapterError(
                f"{self.system_id} refuses inherited TrialMatchAI overrides: {', '.join(inherited)}"
            )
        return environment

    def run(self, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
        from taim.patient_to_trial import PatientToTrialRunResult

        started = time.perf_counter()
        cwd, workspace, timeout, trials, criteria, search_db = self._locations(request)
        environment = self._checked_environment(request)
        command_template = request.options.get("command")
        if (
            not isinstance(command_template, (tuple, list))
            or not command_template
            or any(not isinstance(token, str) or not token for token in command_template)
        ):
            raise TrialMatchAIAdapterError("System option 'command' must be non-empty tokens")
        if "{driver}" not in command_template:
            raise TrialMatchAIAdapterError("TrialMatchAI v0.9.1 command must name {driver}")
        expected_commit = request.options.get("trialmatchai_expected_commit")
        expected_repository = request.options.get("trialmatchai_expected_repository")
        if (expected_commit, expected_repository) != (
            TRIALMATCHAI_UPSTREAM_COMMIT,
            TRIALMATCHAI_UPSTREAM_REPOSITORY,
        ):
            raise TrialMatchAIAdapterError(
                f"{self.system_id} is pinned to {TRIALMATCHAI_UPSTREAM_REPOSITORY} at "
                f"{TRIALMATCHAI_UPSTREAM_COMMIT}"
            )
        upstream_checkout = trialmatchai_checkout_configuration(
            cwd,
            (),
            expected_commit=TRIALMATCHAI_UPSTREAM_COMMIT,
            expected_repository=TRIALMATCHAI_UPSTREAM_REPOSITORY,
        )

        if self.expected_concept_store_manifest_sha256 is None:
            raise TrialMatchAIAdapterError(
                f"{self.system_id} has no bound concept store yet; it runs only once the store is "
                "built by the tag's recipe and its manifest digest is pinned"
            )
        store_path = environment.get(TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_VARIABLE)
        if not store_path or not Path(store_path).is_absolute() or not Path(store_path).is_dir():
            raise TrialMatchAIAdapterError(
                f"{TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_VARIABLE} must name the bound concept "
                "store directory by an absolute path"
            )
        concept_store = _concept_store_identity(
            Path(store_path), self.expected_concept_store_manifest_sha256
        )
        cache = environment.get(TRIALMATCHAI_UPSTREAM_MODEL_CACHE_VARIABLE)
        if not cache or not Path(cache).is_absolute() or not Path(cache).is_dir():
            raise TrialMatchAIAdapterError(
                f"{TRIALMATCHAI_UPSTREAM_MODEL_CACHE_VARIABLE} must name the model cache "
                "directory by an absolute path"
            )
        models = verify_trialmatchai_upstream_model_cache(Path(cache))

        validate_snapshot_corpus(request, trials)
        if not criteria.is_dir():
            raise TrialMatchAIAdapterError(f"processed criteria directory missing: {criteria}")
        if not search_db.is_dir():
            raise TrialMatchAIAdapterError(f"search database missing: {search_db}")

        workspace.mkdir(parents=True, exist_ok=True)
        config_path = workspace / "trialmatchai-config.json"
        config_path.write_bytes(trialmatchai_upstream_config_bytes())
        config = json.loads(config_path.read_text(encoding="utf-8"))
        driver_path = workspace / "trialmatchai_trec_track.py"
        driver_path.write_text(_TREC_TRACK_DRIVER, encoding="utf-8")
        topics = request.snapshot.topics
        topics_path = workspace / "topics.json"
        topics_path.write_text(
            json.dumps(
                {topic.topic_id: _topic_text(topic) for topic in topics},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        track_root = workspace / "track"
        for name, repository in _ADAPTER_PATH_VARIABLES:
            revision = next(
                rev for _, repo, rev in TRIALMATCHAI_UPSTREAM_MODEL_PINS if repo == repository
            )
            environment[name] = str(_snapshot_directory(Path(cache), repository, revision))
        environment["TRIALMATCHAI_TRIALS_JSON_FOLDER"] = str(trials)
        environment["TRIALMATCHAI_LOG_JSON"] = "1"
        environment["TRIALMATCHAI_LOG_LEVEL"] = "INFO"

        command = [
            token.format(checkout=str(cwd), driver=str(driver_path)) for token in command_template
        ] + [
            "--config", str(config_path),
            "--topics", str(topics_path),
            "--track-root", str(track_root),
            "--search-db", str(search_db),
            "--processed-trials", str(trials),
            "--processed-criteria", str(criteria),
            "--track-key", self.trec_track_key,
            "--id-prefix", self.trec_id_prefix,
        ]  # fmt: skip
        stdout = workspace / "trialmatchai.stdout.jsonl"
        stderr = workspace / "trialmatchai.stderr.log"
        before = _database_listing(search_db)
        # The working directory is the workspace, not the checkout, so no relative default in
        # their config (for example ``data/processed_trials``) can pick up stray files there.
        _run_logged(
            command,
            cwd=workspace,
            environment=environment,
            timeout_seconds=timeout * max(1, len(topics)),
            stdout=stdout,
            stderr=stderr,
        )
        if _database_listing(search_db) != before:
            raise TrialMatchAIAdapterError(
                "the search database changed during the run; its run-start digest no longer "
                "describes what was searched"
            )

        outputs = track_root / "outputs"
        candidates: list[Candidate] = []
        retrieval: list[Candidate] = []
        assessments: dict[str, object] = {}
        raw_paths: list[Path] = []
        for topic in topics:
            ranked_path = outputs / topic.topic_id / "ranked_trials.json"
            if not ranked_path.is_file():
                raise TrialMatchAIAdapterError(
                    f"TrialMatchAI v0.9.1 produced no ranked_trials.json for {topic.topic_id}"
                )
            first_level = ranked_path.parent / "first_level_scores.json"
            raw_paths.extend(path for path in (ranked_path, first_level) if path.is_file())
            normalized = load_and_normalize_ranked_trials(
                ranked_path,
                run_id=request.run_id,
                topic_id=topic.topic_id,
                system_id=self.system_id,
                first_level_scores_path=first_level if first_level.is_file() else None,
            )
            candidates.extend(normalized.candidates[: request.top_k])
            retrieval.extend(normalized.diagnostic_rankings.get("retrieval", ()))
            run_block = normalized.raw_payload.get("Run")
            assessments[topic.topic_id] = (
                {
                    key: run_block.get(key)
                    for key in ("mode", "assessment_status", "candidate_count")
                }
                | {"assessed": len(run_block.get("assessed_trial_ids") or ())}
                if isinstance(run_block, Mapping)
                else None
            )
        linking = trialmatchai_upstream_linking_evidence(
            stdout,
            {
                topic.topic_id: outputs / topic.topic_id / "first_level_query_plan.json"
                for topic in topics
            },
        )

        stage_rankings: list[StageRanking] = []
        if retrieval:
            diagnostic = workspace / "retrieval_candidates.json"
            diagnostic.write_text(
                json.dumps([candidate.to_dict() for candidate in retrieval], sort_keys=True),
                encoding="utf-8",
            )
            stage_rankings.append(
                StageRanking(
                    name="retrieval",
                    pipeline_depth="retrieval",
                    candidates=tuple(retrieval),
                    artifact_hash=sha256_file(diagnostic),
                )
            )
        model_identity = {
            "kind": "trialmatchai_upstream_pinned_models",
            "upstream_repository": TRIALMATCHAI_UPSTREAM_REPOSITORY,
            "upstream_commit": TRIALMATCHAI_UPSTREAM_COMMIT,
            "config_sha256": TRIALMATCHAI_UPSTREAM_CONFIG_SHA256,
            "revisions_pinned_by": "TAIM, not upstream: the tag's config names no revision",
            "models": models,
            "query_expansion": {
                "model": "microsoft/phi-4",
                "adapter": None,
                "note": "their config keeps query_expansion.adapter null and _resolve_settings "
                "does not replace a null (matching/query_expansion.py:90)",
            },
            "serving_backends": {"rag": "vllm", "LLM_reranker": "vllm"},
        }
        index_identity = {
            "kind": "trialmatchai_search_index",
            "persistent": True,
            "task_input_id": request.snapshot.task_input_id,
            "corpus": {
                "snapshot_trial_ids_validated": True,
                "trial_count": len(request.snapshot.trials),
                "filter_manifest_sha256": corpus_filter_manifest_sha256(trials),
            },
            "search_backend": without_locators(config.get("search_backend", {})),
            "concept_store": concept_store,
        }
        return PatientToTrialRunResult(
            candidates=tuple(candidates),
            configuration={
                "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
                "upstream_system": self.system_id,
                "upstream_checkout": upstream_checkout,
                "task_input_id": request.snapshot.task_input_id,
                "renderer_version": TRIALMATCHAI_UPSTREAM_TOPIC_RENDERER_VERSION,
                "runtime_contract": trialmatchai_upstream_runtime_contract(),
                "config": {
                    "resource": TRIALMATCHAI_UPSTREAM_CONFIG_RESOURCE,
                    "sha256": TRIALMATCHAI_UPSTREAM_CONFIG_SHA256,
                    "trec_preset": dict(TRIALMATCHAI_UPSTREAM_TREC_PRESET),
                },
                "model_identity": model_identity,
                "index_identity": index_identity,
                "concept_linking_evidence": linking,
                "assessment_runs": assessments,
                "topics": len(topics),
                "process_execution": {
                    "mode": TRIALMATCHAI_UPSTREAM_PROCESS_EXECUTION_MODE,
                    "subprocesses": 1,
                    "driver_sha256": TRIALMATCHAI_UPSTREAM_DRIVER_SHA256,
                    "timeout_seconds_per_topic": timeout,
                },
                "command_template": list(command_template),
                "environment": {
                    key: value
                    for key, value in sorted(environment.items())
                    if (key.startswith(("TRIALMATCHAI_", "HF_", "VLLM_")) and "TOKEN" not in key)
                },
                "logs": {
                    "stdout_sha256": sha256_file(stdout),
                    "stderr_sha256": sha256_file(stderr),
                },
                "raw_ranked_trial_artifacts": [
                    {"path": str(path), "sha256": sha256_file(path)} for path in raw_paths
                ],
            },
            runtime_seconds=time.perf_counter() - started,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth="post_eligibility",
            stage_rankings=tuple(stage_rankings),
        )


def trialmatchai_upstream_run_options(
    args: argparse.Namespace,
    benchmark: ResolvedBenchmarkContract,
    task_input_id: str,
) -> dict[str, object]:
    """The physical run options the release layer binds into harness inputs and a System Input."""

    del benchmark, task_input_id
    if args.trialmatchai_cwd is None or args.trialmatchai_corpus_dir is None:
        raise ValueError("TrialMatchAI requires --trialmatchai-cwd and --trialmatchai-corpus-dir")
    corpus = args.trialmatchai_corpus_dir.resolve()
    command = tuple(args.trialmatchai_command or ()) or (
        "uv",
        "run",
        "--frozen",
        "--project",
        "{checkout}",
        "python",
        "{driver}",
    )
    workspace = args.trialmatchai_workspace or (
        Path(args.output_dir) / ".trialmatchai" / (args.run_id or "external-run")
    )
    return {
        "command": command,
        "cwd": args.trialmatchai_cwd.resolve(),
        "workspace": workspace.resolve(),
        "timeout_seconds": args.trialmatchai_timeout_seconds,
        "corpus_dir": corpus,
        "search_db_path": (args.trialmatchai_search_db_path or (corpus / "search")).resolve(),
        "trialmatchai_expected_commit": TRIALMATCHAI_UPSTREAM_COMMIT,
        "trialmatchai_expected_repository": TRIALMATCHAI_UPSTREAM_REPOSITORY,
        "trialmatchai_runtime_contract": trialmatchai_upstream_runtime_contract(),
    }


__all__ = [
    "TRIALMATCHAI_UPSTREAM_COMMIT",
    "TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_MANIFEST_SHA256",
    "TRIALMATCHAI_UPSTREAM_CONCEPT_STORE_RULING",
    "TRIALMATCHAI_UPSTREAM_CONFIG_RESOURCE",
    "TRIALMATCHAI_UPSTREAM_CONFIG_SHA256",
    "TRIALMATCHAI_UPSTREAM_DRIVER_SHA256",
    "TRIALMATCHAI_UPSTREAM_MODEL_PINS",
    "TRIALMATCHAI_UPSTREAM_REPOSITORY",
    "TRIALMATCHAI_UPSTREAM_REQUIRED_ENVIRONMENT",
    "TRIALMATCHAI_UPSTREAM_SNAPSHOT_FILES",
    "TRIALMATCHAI_UPSTREAM_TOPIC_RENDERER_VERSION",
    "TRIALMATCHAI_UPSTREAM_TREC_PRESET",
    "TrialMatchAIUpstreamV091Base",
    "trialmatchai_upstream_config_bytes",
    "trialmatchai_upstream_linking_evidence",
    "trialmatchai_upstream_run_options",
    "trialmatchai_upstream_runtime_contract",
    "verify_trialmatchai_upstream_model_cache",
]
