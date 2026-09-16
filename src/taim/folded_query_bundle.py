"""The folded-query bundle: locally derived folded queries bound to one Task Input.

Folded BM25 ranks with these queries and the staged paper System reuses them, so the bundle's
schema, construction and validation live in neither System's module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from taim.contracts import require_sha256
from taim.query_folding import FOLDED_QUERY_POLICY
from taim.schemas import JsonValue, json_value_to_builtins
from taim.snapshot import BenchmarkTopic

# 2.0 requires provenance.extra_context_rules_sha256: folding now runs TAIM's stated polarity
# rules, so a 1.0 bundle was folded without them and is refused by name, not as "unsupported".
FOLDED_QUERY_BUNDLE_SCHEMA_VERSION = "2.0"
_PRE_POLARITY_BUNDLE_SCHEMA_VERSION = "1.0"
_BUNDLE_FIELDS = {
    "schema_version",
    "task",
    "task_input_id",
    "policy",
    "provenance",
    "queries",
}


@dataclass(frozen=True, slots=True)
class FoldedQuery:
    topic_id: str
    canonical_text: str


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        json_value_to_builtins(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def validate_folded_query_bundle(
    payload: object,
    *,
    expected_task_input_id: str,
    expected_topic_ids: Sequence[str],
) -> tuple[tuple[FoldedQuery, ...], str]:
    """Validate locally derived folded queries and bind them to one Task Input."""

    if not isinstance(payload, Mapping) or set(payload) != _BUNDLE_FIELDS:
        raise ValueError("folded query bundle has unexpected or missing fields")
    schema_version = payload["schema_version"]
    if schema_version == _PRE_POLARITY_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"folded query bundle schema version {schema_version} has no "
            "provenance.extra_context_rules_sha256, which schema version "
            f"{FOLDED_QUERY_BUNDLE_SCHEMA_VERSION} requires: it was folded without the polarity "
            "rules, so rebuild it with `trial-benchmark queries prepare`"
        )
    if schema_version != FOLDED_QUERY_BUNDLE_SCHEMA_VERSION:
        raise ValueError("folded query bundle has an unsupported schema version")
    if payload["task"] != "patient_to_trial":
        raise ValueError("folded query bundle is direction-mismatched")
    require_sha256(payload["task_input_id"], "folded query Task Input ID")
    if payload["task_input_id"] != expected_task_input_id:
        raise ValueError("folded query bundle Task Input does not match")
    policy = payload["policy"]
    if not isinstance(policy, Mapping) or dict(policy) != FOLDED_QUERY_POLICY:
        raise ValueError("folded query bundle does not declare the folded query policy")
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("folded query bundle provenance must be an object")
    for field in (
        "context_rules_sha256",
        "extra_context_rules_sha256",
        "snomed_description_sha256",
    ):
        require_sha256(provenance.get(field), f"folded query {field}")
    raw_queries = payload["queries"]
    if (
        not isinstance(raw_queries, Sequence | tuple)
        or isinstance(raw_queries, str | bytes)
        or any(not isinstance(row, Mapping) for row in raw_queries)
    ):
        raise ValueError("folded query bundle queries must be an array of objects")
    queries: list[FoldedQuery] = []
    for row in raw_queries:
        if set(row) != {"topic_id", "canonical_text"}:
            raise ValueError("folded query row has unexpected or missing fields")
        topic_id = row["topic_id"]
        text = row["canonical_text"]
        if not isinstance(topic_id, str) or not topic_id or not isinstance(text, str) or not text:
            raise ValueError("folded query rows require non-empty topic_id and canonical_text")
        queries.append(FoldedQuery(topic_id, text))
    observed = tuple(query.topic_id for query in queries)
    expected = tuple(expected_topic_ids)
    if observed != expected or len(observed) != len(set(observed)):
        raise ValueError("folded query bundle does not exactly cover Task Input topics in order")
    return tuple(queries), "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def build_folded_query_bundle(
    topics: Sequence[BenchmarkTopic],
    *,
    task_input_id: str,
    snomed_release: Path,
) -> dict[str, JsonValue]:
    """Derive folded queries locally while preserving every licensed-input identity."""

    from taim.query_folding import build_context_pipeline, build_extraction, load_concept_lexicon

    require_sha256(task_input_id, "folded query Task Input ID")
    lexicon = load_concept_lexicon(snomed_release)
    nlp, assertion_provenance = build_context_pipeline()
    source = {topic.topic_id: topic.canonical_text for topic in topics}
    extraction = build_extraction(source, lexicon, nlp, assertion_provenance)
    per_topic = cast(Mapping[str, Mapping[str, object]], extraction["per_topic"])
    payload: dict[str, JsonValue] = {
        "schema_version": FOLDED_QUERY_BUNDLE_SCHEMA_VERSION,
        "task": "patient_to_trial",
        "task_input_id": task_input_id,
        "policy": dict(FOLDED_QUERY_POLICY),
        "provenance": {
            "medspacy_version": assertion_provenance["library_version"],
            "spacy_version": assertion_provenance.get("spacy_version"),
            "context_rules_sha256": assertion_provenance["context_rules_sha256"],
            "extra_context_rules_sha256": assertion_provenance["extra_context_rules_sha256"],
            "context_rule_count": assertion_provenance["context_rule_count"],
            "snomed_release": lexicon.release_name,
            "snomed_description_sha256": lexicon.description_file_sha256,
            "licensed_input_policy": (
                "SNOMED CT is user-supplied and is not redistributed; emitted query terms are "
                "source-topic spans and contain no concept identifiers"
            ),
        },
        "queries": [
            {
                "topic_id": topic.topic_id,
                "canonical_text": cast(str, per_topic[topic.topic_id]["folded_query"]),
            }
            for topic in topics
        ],
    }
    validate_folded_query_bundle(
        payload,
        expected_task_input_id=task_input_id,
        expected_topic_ids=tuple(topic.topic_id for topic in topics),
    )
    return payload


__all__ = [
    "FOLDED_QUERY_BUNDLE_SCHEMA_VERSION",
    "FoldedQuery",
    "build_folded_query_bundle",
    "validate_folded_query_bundle",
]
