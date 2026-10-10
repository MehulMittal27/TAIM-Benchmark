"""The folded-query unit executes from a bound synthetic bundle without extraction libraries."""

from __future__ import annotations

# ruff: noqa: S101
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.folded_query_bundle import FOLDED_QUERY_BUNDLE_SCHEMA_VERSION
from taim.patient_to_trial import PatientToTrialRunRequest
from taim.query_folding import FOLDED_QUERY_POLICY
from taim.release_fixture import load_release_fixture
from taim.release_profiles import resolve_benchmark_profile
from taim.release_reference_systems import run_release_reference_system


def test_folded_unit_executes_and_refuses_mismatched_task_input() -> None:
    prepared = load_release_fixture("trec-ct-2021")
    benchmark = resolve_benchmark_profile(prepared, profile_id="synthetic-patient-to-trial")
    base_request = PatientToTrialRunRequest.for_benchmark(
        benchmark,
        run_id="folded",
        top_k=3,
        metric_cutoff=3,
        options={},
        clinical_as_of=datetime(2026, 8, 31, tzinfo=UTC),
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )
    bundle = {
        "schema_version": FOLDED_QUERY_BUNDLE_SCHEMA_VERSION,
        "task": "patient_to_trial",
        "task_input_id": base_request.snapshot.task_input_id,
        "policy": dict(FOLDED_QUERY_POLICY),
        "provenance": {
            "context_rules_sha256": "sha256:" + "2" * 64,
            "extra_context_rules_sha256": "sha256:" + "4" * 64,
            "snomed_description_sha256": "sha256:" + "3" * 64,
        },
        "queries": [
            {"topic_id": topic.topic_id, "canonical_text": topic.canonical_text.split()[0]}
            for topic in benchmark.topics
        ],
    }
    request = PatientToTrialRunRequest.for_benchmark(
        benchmark,
        run_id="folded",
        top_k=3,
        metric_cutoff=3,
        options={"b": 0.75, "folded_query_bundle": bundle, "k1": 1.2},
        clinical_as_of=datetime(2026, 8, 31, tzinfo=UTC),
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )

    result = run_release_reference_system("bm25-folded", request)

    assert result.candidates
    assert {row.system_id for row in result.candidates} == {"bm25-folded"}
    assert result.configuration["query_transformation"]["bundle_id"].startswith("sha256:")

    mismatched = {**bundle, "task_input_id": "sha256:" + "0" * 64}
    with pytest.raises(ValueError, match="Task Input does not match"):
        run_release_reference_system(
            "bm25-folded",
            replace(request, options={**request.options, "folded_query_bundle": mismatched}),
        )
