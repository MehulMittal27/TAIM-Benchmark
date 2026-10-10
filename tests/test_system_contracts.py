# ruff: noqa: S101
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints

import pytest

from taim.entity_versions import PatientEvidenceProfile, source_grounded_patient_evidence_profile
from taim.patient_to_trial import (
    PatientToTrialRunRequest,
    PatientToTrialRunResult,
    PatientToTrialSystem,
    PatientToTrialTaskInput,
)
from taim.schemas import Candidate, json_value_to_builtins
from taim.snapshot import (
    CAPABILITY_PATIENT_EVIDENCE,
    CAPABILITY_TYPED_PATIENT_CORE,
    DerivedView,
)
from tests.benchmark_helpers import prepared_benchmark


def test_public_system_contracts_are_registry_independent() -> None:
    assert PatientToTrialTaskInput.__module__ == "taim.patient_to_trial"
    assert PatientToTrialRunRequest.__module__ == "taim.patient_to_trial"
    assert PatientToTrialRunResult.__module__ == "taim.patient_to_trial"
    assert get_type_hints(PatientToTrialSystem.run)["request"] is PatientToTrialRunRequest


def _system_snapshot() -> PatientToTrialTaskInput:
    prepared = prepared_benchmark()
    return PatientToTrialTaskInput(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        topics=prepared.topics,
        trials=prepared.trials,
        available_capabilities=prepared.snapshot.available_capabilities,
        derived_views=prepared.snapshot.derived_views,
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )


def test_portable_system_input_identity_omits_only_machine_local_paths() -> None:
    snapshot = _system_snapshot()
    shared = {
        "snapshot": snapshot,
        "run_id": "portable-v2",
        "top_k": 10,
        "metric_cutoff": 10,
    }
    first = PatientToTrialRunRequest(
        **shared,
        options={"index_dir": Path("/machine-a/index"), "query_batch_size": 32},
    )
    second = PatientToTrialRunRequest(
        **shared,
        options={"index_dir": Path("/machine-b/index"), "query_batch_size": 32},
    )
    changed_method = PatientToTrialRunRequest(
        **shared,
        options={"index_dir": Path("/machine-b/index"), "query_batch_size": 64},
    )

    assert first.system_input_id == second.system_input_id
    assert first.system_input_id != changed_method.system_input_id
    assert first.system_input_record_dict()["identity_version"] == "2.0"
    assert "index_dir" in first.system_input_record_dict()["options"]
    # The portable recipe is the only one: there is no identity_version to choose.
    with pytest.raises(TypeError):
        PatientToTrialRunRequest(**shared, options={}, identity_version="1.0")  # type: ignore[call-arg]


def test_system_snapshot_freezes_capabilities_before_hashing() -> None:
    prepared = prepared_benchmark()
    capabilities = set(prepared.snapshot.available_capabilities)
    snapshot = PatientToTrialTaskInput(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        topics=prepared.topics,
        trials=prepared.trials,
        available_capabilities=capabilities,  # type: ignore[arg-type]
        derived_views=prepared.snapshot.derived_views,
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )
    original_id = snapshot.task_input_id

    capabilities.add("mutable-after-construction")

    assert isinstance(snapshot.available_capabilities, frozenset)
    assert "mutable-after-construction" not in snapshot.available_capabilities
    assert snapshot.task_input_id == original_id


def test_system_result_detaches_and_freezes_outputs() -> None:
    candidates = [Candidate("run", "system", "topic", "trial", 1, 1.0)]
    configuration = {"nested": {"value": 1}, "items": [1]}

    result = PatientToTrialRunResult(
        candidates=candidates,
        configuration=configuration,
        runtime_seconds=0.1,
        primary_ranking="candidates",
        pipeline_depth="retrieval",
    )
    candidates.clear()
    configuration["nested"]["value"] = 2

    assert len(result.candidates) == 1
    assert isinstance(result.candidates, tuple)
    assert result.configuration["nested"]["value"] == 1
    assert json.loads(json.dumps(result.configuration)) == {
        "nested": {"value": 1},
        "items": [1],
    }
    with pytest.raises(TypeError):
        result.configuration["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        result.configuration["nested"]["value"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        result.configuration["items"].append(2)


@pytest.mark.parametrize("runtime", [-1.0, float("nan"), float("inf"), True])
def test_system_result_rejects_invalid_runtime(runtime: object) -> None:
    with pytest.raises(ValueError, match="runtime_seconds"):
        PatientToTrialRunResult(
            candidates=(),
            configuration={},
            runtime_seconds=runtime,  # type: ignore[arg-type]
            primary_ranking="candidates",
            pipeline_depth="retrieval",
        )


def test_system_snapshot_exposes_only_profile_resolved_views_and_capabilities() -> None:
    prepared = prepared_benchmark()
    patient_id = prepared.topics[0].topic_id
    trial_id = prepared.trials[0].trial_id
    view = DerivedView(
        name="structured_projection",
        version="1.0",
        input_snapshot_id=prepared.snapshot.base_snapshot_id,
        configuration={"normalizer": "test-v1"},
        content={
            patient_id: {"secret_patient_fact": "hidden"},
            trial_id: {"public_trial_fact": "visible"},
        },
    )
    source_profile = source_grounded_patient_evidence_profile()
    canonical_only = replace(
        source_profile,
        definition={
            **dict(source_profile.definition),
            "visible_patient_fields": ["canonical_text"],
        },
    )

    snapshot = PatientToTrialTaskInput(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        topics=prepared.topics,
        trials=prepared.trials,
        available_capabilities=prepared.snapshot.available_capabilities
        | {CAPABILITY_PATIENT_EVIDENCE, CAPABILITY_TYPED_PATIENT_CORE, view.capability},
        derived_views=(view,),
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
        patient_evidence_profile=canonical_only,
    )

    assert snapshot.derived_views[0].content == {trial_id: {"public_trial_fact": "visible"}}
    assert snapshot.patient_versions[0].derived_evidence == {}
    assert snapshot.trial_versions[0].derived_evidence[view.name]["content"] == {
        "public_trial_fact": "visible"
    }
    assert CAPABILITY_PATIENT_EVIDENCE not in snapshot.available_capabilities
    assert CAPABILITY_TYPED_PATIENT_CORE not in snapshot.available_capabilities
    assert view.capability in snapshot.available_capabilities
    assert "secret_patient_fact" not in str(snapshot.task_input_dict())


def test_system_snapshot_drops_empty_resolved_view_and_capability() -> None:
    prepared = prepared_benchmark()
    view = DerivedView(
        name="patient_only_projection",
        version="1.0",
        input_snapshot_id=prepared.snapshot.base_snapshot_id,
        configuration={},
        content={prepared.topics[0].topic_id: {"secret": True}},
    )
    source_profile = source_grounded_patient_evidence_profile()
    canonical_only = PatientEvidenceProfile(
        "canonical-only",
        "1.0",
        {
            **dict(source_profile.definition),
            "visible_patient_fields": ["canonical_text"],
        },
    )

    snapshot = PatientToTrialTaskInput(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        topics=prepared.topics,
        trials=prepared.trials,
        available_capabilities=prepared.snapshot.available_capabilities | {view.capability},
        derived_views=(view,),
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
        patient_evidence_profile=canonical_only,
    )

    assert snapshot.derived_views == ()
    assert view.capability not in snapshot.available_capabilities


def test_system_snapshot_preserves_global_view_for_trial_versions() -> None:
    prepared = prepared_benchmark()
    view = DerivedView(
        name="global_projection",
        version="1.0",
        input_snapshot_id=prepared.snapshot.base_snapshot_id,
        configuration={},
        content=["shared"],
    )
    source_profile = source_grounded_patient_evidence_profile()
    canonical_only = replace(
        source_profile,
        definition={
            **dict(source_profile.definition),
            "visible_patient_fields": ["canonical_text"],
        },
    )

    snapshot = PatientToTrialTaskInput(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        topics=prepared.topics,
        trials=prepared.trials,
        available_capabilities=prepared.snapshot.available_capabilities | {view.capability},
        derived_views=(view,),
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
        patient_evidence_profile=canonical_only,
    )

    assert snapshot.derived_views == (view,)
    assert snapshot.patient_versions[0].derived_evidence == {}
    assert json_value_to_builtins(snapshot.trial_versions[0].derived_evidence[view.name]) == {
        "version": view.version,
        "configuration": {},
        "scope": "global",
        "content": ["shared"],
        "additional_fields": {},
    }
    assert view.capability in snapshot.available_capabilities


def test_system_snapshot_drops_patient_view_when_its_undeclared_inputs_are_projected() -> None:
    prepared = prepared_benchmark()
    source_topic = replace(prepared.topics[0], additional_fields={"hidden_source": True})
    view = DerivedView(
        name="patient_projection_with_unknown_dependencies",
        version="1.0",
        input_snapshot_id=prepared.snapshot.base_snapshot_id,
        configuration={},
        content={source_topic.topic_id: {"derived": True}},
    )
    source_profile = source_grounded_patient_evidence_profile()
    profile = PatientEvidenceProfile(
        "canonical-and-derived-only",
        "1.0",
        {
            **dict(source_profile.definition),
            "visible_patient_fields": ["canonical_text", "derived_evidence"],
        },
    )

    snapshot = PatientToTrialTaskInput(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        topics=(source_topic,),
        trials=prepared.trials,
        available_capabilities=prepared.snapshot.available_capabilities | {view.capability},
        derived_views=(view,),
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
        patient_evidence_profile=profile,
    )

    assert snapshot.derived_views == ()
    assert view.capability not in snapshot.available_capabilities
