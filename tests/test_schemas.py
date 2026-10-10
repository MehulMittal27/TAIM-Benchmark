# ruff: noqa: S101
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from taim.contracts import content_sha256
from taim.schemas import (
    PRIMARY_RANKING_CANDIDATES,
    RUN_MANIFEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Candidate,
    RelevanceJudgment,
    RunManifest,
    SchemaValidationError,
    StageRanking,
    read_candidates_jsonl,
    read_manifest,
    read_qrels_jsonl,
    validate_pipeline_depth,
    write_candidates_jsonl,
    write_manifest,
    write_qrels_jsonl,
)
from tests.benchmark_helpers import manifest_for, prepared_benchmark, task_input_for

_PREPARED = prepared_benchmark()
_PROFILE = {
    "profile_id": "test-profile",
    "definition_sha256": "sha256:" + "3" * 64,
    "evaluation": {},
}


def _manifest(**overrides: object) -> RunManifest:
    budget_k = overrides.pop("budget_k", 100)
    embedded_budget_k = (
        budget_k
        if isinstance(budget_k, int) and not isinstance(budget_k, bool) and budget_k > 0
        else 100
    )
    return replace(
        manifest_for(
            _PREPARED,
            benchmark_profile=_PROFILE,
            budget_k=embedded_budget_k,
        ),
        runtime_seconds=0.0,
        candidates_sha256="sha256:" + "0" * 64,
        budget_k=budget_k,
        **overrides,
    )


def test_simple_schemas_round_trip_with_explicit_version() -> None:
    records = [
        Candidate("run-1", "synthetic-system", "T1", "NCT1", 1, 3),
        RelevanceJudgment("T1", "NCT1", 2),
    ]

    for record in records:
        assert record.to_dict()["schema_version"] == SCHEMA_VERSION
        assert type(record).from_json(record.to_json()) == record

    assert records[0].score == 3.0


def test_non_extensible_schemas_reject_extensions_field() -> None:
    candidate = Candidate("run-1", "synthetic-system", "T1", "NCT1", 1, 1.0).to_dict()
    with pytest.raises(SchemaValidationError, match="unexpected fields: extensions"):
        Candidate.from_dict({**candidate, "extensions": {"demo.x": 1}})


@pytest.mark.parametrize("rank", [0, -1, True, 1.5])
def test_candidate_rejects_invalid_ranks(rank: object) -> None:
    with pytest.raises(SchemaValidationError, match="rank"):
        Candidate("run-1", "synthetic-system", "T1", "NCT1", rank, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "1.0"])
def test_candidate_rejects_non_finite_or_non_numeric_scores(score: object) -> None:
    with pytest.raises(SchemaValidationError, match="score"):
        Candidate("run-1", "synthetic-system", "T1", "NCT1", 1, score)  # type: ignore[arg-type]


@pytest.mark.parametrize("label", [-1, True, 1.0])
def test_relevance_judgment_requires_non_negative_integer_label(label: object) -> None:
    with pytest.raises(SchemaValidationError, match="label"):
        RelevanceJudgment("T1", "NCT1", label)  # type: ignore[arg-type]


def test_relevance_judgment_defers_dataset_label_semantics_to_evaluation_package() -> None:
    assert RelevanceJudgment("T1", "NCT1", 3).label == 3


@pytest.mark.parametrize(
    "depth",
    ["retrieval", "rerank", "post_eligibility", "other:phi_rerank", "other:a"],
)
def test_pipeline_depth_accepts_recommended_values_and_other_slug(depth: str) -> None:
    assert validate_pipeline_depth(depth) == depth


@pytest.mark.parametrize(
    "depth",
    ["", "final", "version", "other:", "other:Final", "other:has space", True, 1],
)
def test_pipeline_depth_rejects_illegal_values(depth: object) -> None:
    with pytest.raises(SchemaValidationError, match="pipeline_depth"):
        validate_pipeline_depth(depth)  # type: ignore[arg-type]


def test_manifest_normalizes_time_and_round_trips(tmp_path) -> None:
    base = _manifest(budget_k=5)
    manifest = replace(
        base,
        created_at=datetime(2026, 7, 21, 12, 30, tzinfo=timezone(timedelta(hours=2))),
        configuration={
            "top_k": 5,
            "tokenizer": {"lowercase": True},
            "system_input": base.configuration["system_input"],
        },
    )
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)
    loaded = read_manifest(path)

    assert loaded == manifest
    assert loaded.primary_ranking == PRIMARY_RANKING_CANDIDATES
    assert loaded.pipeline_depth == "retrieval"
    assert loaded.budget_k == 5
    assert loaded.created_at == datetime(2026, 7, 21, 10, 30, tzinfo=UTC)
    serialized = json.loads(path.read_text(encoding="utf-8"))
    assert serialized["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert RUN_MANIFEST_SCHEMA_VERSION == "7.0"
    assert serialized["created_at"] == "2026-07-21T10:30:00Z"
    assert serialized["snapshot_contract_version"] == "2.0"
    assert serialized["prepared_snapshot_id"] == _PREPARED.snapshot.snapshot_id
    assert serialized["system_input_id"] == manifest.system_input_id
    assert serialized["task"] == "patient_to_trial"
    assert serialized["clinical_as_of"] == "2021-04-27T00:00:00Z"
    assert serialized["runtime_seconds"] == 0.0
    assert serialized["candidates_sha256"] == "sha256:" + "0" * 64
    assert serialized["query_patient_versions"]
    assert serialized["trial_corpus_versions"]
    assert "snapshot_id" not in serialized
    assert serialized["evaluation_package_id"] == (
        _PREPARED.evaluation_package.evaluation_package_id
    )
    assert serialized["benchmark_profile"] == _PROFILE
    assert serialized["evaluation_artifacts"] == {}


def test_manifest_v7_refuses_a_system_input_without_the_portable_identity_recipe() -> None:
    payload = _manifest().to_dict()
    system_input = payload["configuration"]["system_input"]
    assert system_input["identity_version"] == "2.0"
    assert RunManifest.from_dict(payload).to_dict() == payload

    del system_input["identity_version"]
    with pytest.raises(SchemaValidationError, match="identity_version"):
        RunManifest.from_dict(payload)


def test_manifest_round_trips_optional_stage_ranking_metadata(tmp_path) -> None:
    manifest = _manifest(
        pipeline_depth="rerank",
        stage_rankings=(
            StageRanking("retrieval", "retrieval", artifact_hash="sha256:" + "0" * 64),
        ),
    )
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)

    assert read_manifest(path).stage_rankings == manifest.stage_rankings


def test_forward_manifest_requires_typed_task_lineage_and_exact_pair_references() -> None:
    payload = _manifest().to_dict()
    without_cutoff = {key: value for key, value in payload.items() if key != "clinical_as_of"}

    with pytest.raises(SchemaValidationError, match="clinical_as_of"):
        RunManifest.from_dict(without_cutoff)

    payload["reuse_lineage"] = {
        "pair_assessment_collections": [
            {
                "collection_id": "sha256:" + "a" * 64,
                "source_collection_id": None,
                "reused_assessment_ids": [],
                "reused_pair_assessment_input_ids": [],
            }
        ]
    }
    with pytest.raises(SchemaValidationError, match="references do not match"):
        RunManifest.from_dict(payload)


@pytest.mark.parametrize(
    "tamper",
    ("patient_content", "options", "budget"),
)
def test_forward_manifest_recomputes_embedded_task_and_system_input_identities(
    tamper: str,
) -> None:
    payload = _manifest().to_dict()
    system_input = payload["configuration"]["system_input"]
    if tamper == "patient_content":
        system_input["query_patient_versions"][0]["topic"]["canonical_text"] = "tampered"
        expected = "patient_version_id"
    elif tamper == "options":
        system_input["options"] = {"tampered": True}
        expected = "system_input_id"
    else:
        system_input["top_k"] = payload["budget_k"] + 1
        expected = "budget_k"

    with pytest.raises(SchemaValidationError, match=expected):
        RunManifest.from_dict(payload)


def test_forward_manifest_rejects_self_consistent_evaluator_only_embedded_options() -> None:
    payload = _manifest().to_dict()
    system_input = payload["configuration"]["system_input"]
    system_input["options"] = {"nested": {"qrels_path": "/evaluator/qrels.jsonl"}}
    recomputed_system_input_id = content_sha256(
        {
            "task_input_id": system_input["task_input_id"],
            "top_k": system_input["top_k"],
            "metric_cutoff": system_input["metric_cutoff"],
            "options": system_input["options"],
            "required_capabilities": system_input["required_capabilities"],
            "optional_capabilities": system_input["optional_capabilities"],
        }
    )
    system_input["system_input_id"] = recomputed_system_input_id
    payload["system_input_id"] = recomputed_system_input_id

    with pytest.raises(SchemaValidationError, match="evaluator-only"):
        RunManifest.from_dict(payload)


@pytest.mark.parametrize("name", ["", "candidates", "../retrieval", "Retrieval"])
def test_stage_ranking_rejects_invalid_names(name: str) -> None:
    with pytest.raises(SchemaValidationError, match="stage ranking"):
        StageRanking(name, "retrieval")


def test_manifest_rejects_non_array_stage_rankings() -> None:
    payload = _manifest().to_dict()
    payload["stage_rankings"] = {}

    with pytest.raises(SchemaValidationError, match="stage_rankings must be an array"):
        RunManifest.from_dict(payload)


def test_manifest_defaults_absent_stage_rankings_to_empty() -> None:
    payload = _manifest().to_dict()
    del payload["stage_rankings"]

    assert RunManifest.from_dict(payload).stage_rankings == ()


def test_manifest_rejects_previous_schema_version() -> None:
    payload = {
        "schema_version": "4.0",
        "run_id": "historical-run",
        "system_id": "synthetic-system",
        "created_at": "2026-07-31T00:00:00Z",
        "git_commit": "0977708",
        "configuration": {},
        "benchmark_name": "trec-ct-2021",
        "corpus_hash": "sha256:" + "1" * 64,
        "topic_hash": "sha256:" + "2" * 64,
        "qrels_hash": "sha256:" + "3" * 64,
        "primary_ranking": PRIMARY_RANKING_CANDIDATES,
        "pipeline_depth": "retrieval",
        "budget_k": 100,
        "stage_rankings": [],
    }

    with pytest.raises(
        SchemaValidationError,
        match=r"unsupported historical RunManifest schema_version '4\.0'",
    ):
        RunManifest.from_dict(payload)


def test_manifest_requires_scientific_identities() -> None:
    payload = _manifest().to_dict()
    for field in (
        "benchmark_lineage",
        "snapshot_contract_version",
        "prepared_snapshot_id",
        "system_input_id",
        "evaluation_package_id",
        "benchmark_profile",
        "primary_ranking",
        "pipeline_depth",
        "budget_k",
    ):
        incomplete = {key: value for key, value in payload.items() if key != field}
        with pytest.raises(SchemaValidationError, match=field):
            RunManifest.from_dict(incomplete)


def test_manifest_validates_primary_ranking_depth_and_hash_identities() -> None:
    with pytest.raises(SchemaValidationError, match="primary_ranking"):
        _manifest(primary_ranking="stage-rerank")
    with pytest.raises(SchemaValidationError, match="pipeline_depth"):
        _manifest(pipeline_depth="final")
    with pytest.raises(SchemaValidationError, match="prepared_snapshot_id"):
        _manifest(prepared_snapshot_id="snapshot")
    with pytest.raises(SchemaValidationError, match="system_input_id"):
        _manifest(system_input_id="snapshot")
    with pytest.raises(SchemaValidationError, match="evaluation_package_id"):
        _manifest(evaluation_package_id="package")
    with pytest.raises(SchemaValidationError, match="definition_sha256"):
        _manifest(benchmark_profile={"profile_id": "test", "definition_sha256": "bad"})


def test_manifest_requires_offset_aware_time_and_json_configuration() -> None:
    with pytest.raises(SchemaValidationError, match="UTC offset"):
        _manifest(created_at=datetime(2026, 7, 21))
    with pytest.raises(SchemaValidationError, match="non-finite"):
        _manifest(configuration={"bad": float("nan")})


@pytest.mark.parametrize("budget_k", [0, -1, True, 1.0])
def test_manifest_requires_positive_integer_budget_k(budget_k: object) -> None:
    with pytest.raises(SchemaValidationError, match="budget_k"):
        _manifest(budget_k=budget_k)


def test_candidate_jsonl_writer_and_reader_validate_records(tmp_path) -> None:
    candidates = [
        Candidate("run-1", "synthetic-system", "T1", "NCT1", 1, 2.5),
        Candidate("run-1", "synthetic-system", "T1", "NCT2", 2, 1.5),
    ]
    path = tmp_path / "nested" / "candidates.jsonl"

    write_candidates_jsonl(path, candidates)

    assert read_candidates_jsonl(path) == candidates
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_candidate_jsonl_reader_reports_line_number(tmp_path) -> None:
    path = tmp_path / "candidates.jsonl"
    valid = Candidate("run-1", "synthetic-system", "T1", "NCT1", 1, 2.5).to_json()
    path.write_text(f"{valid}\n{{not-json}}\n", encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="line 2"):
        read_candidates_jsonl(path)


def test_qrel_jsonl_helpers_round_trip_without_changing_its_schema(tmp_path) -> None:
    judgments = [RelevanceJudgment("1", "NCT1", 0), RelevanceJudgment("2", "NCT2", 2)]
    path = tmp_path / "qrels.jsonl"

    write_qrels_jsonl(path, judgments)

    assert read_qrels_jsonl(path) == judgments
    assert path.read_bytes().endswith(b"\n")


def _compact_manifest():
    from taim.patient_to_trial import PatientToTrialRunRequest

    legacy = _manifest()
    source = task_input_for(_PREPARED)
    request = PatientToTrialRunRequest(
        snapshot=source,
        run_id=legacy.run_id,
        top_k=legacy.budget_k,
        metric_cutoff=legacy.budget_k,
        options={},
    )
    compact = replace(
        legacy,
        system_input_id=request.system_input_id,
        configuration={"system_input": request.system_input_record_dict()},
    )
    return compact, source


def test_compact_manifest_retains_identities_and_resolves_only_matching_source():
    from taim.patient_to_trial import PatientToTrialRunRequest

    compact, source = _compact_manifest()
    restored = RunManifest.from_json(compact.to_json())
    assert restored.resolve_task_input(source) is source
    assert restored.task_input_id == _manifest().task_input_id
    request = PatientToTrialRunRequest(
        source,
        compact.run_id,
        compact.budget_k,
        compact.budget_k,
        {},
    )
    assert restored.system_input_id == request.system_input_id
    # Full trial content stays available from the Task Input; the run record carries lineage only.
    assert request.snapshot.task_input_dict()["trial_corpus_versions"][0]["trial"]
    assert all(
        set(row) == {"trial_id", "trial_version_id"}
        for row in restored.configuration["system_input"]["trial_corpus_versions"]
    )
    with pytest.raises(SchemaValidationError, match="matching prepared source"):
        restored.resolve_task_input()
    for wrong_source in (
        replace(source, prepared_snapshot_id="sha256:" + "f" * 64),
        replace(source, trials=(replace(source.trials[0], canonical_text="different content"),)),
        replace(source, clinical_as_of=source.clinical_as_of + timedelta(days=1)),
    ):
        with pytest.raises(SchemaValidationError, match="does not reproduce"):
            restored.resolve_task_input(wrong_source)


@pytest.mark.parametrize(
    "mutation",
    [
        "count",
        "boolean_count",
        "digest",
        "version",
        "extra",
        "missing",
        "mixed",
        "encoding",
        "empty",
        "duplicate",
        "options",
    ],
)
def test_compact_manifest_refuses_malformed_or_tampered_records(mutation):
    compact, source = _compact_manifest()
    payload = compact.to_dict()
    record = payload["configuration"]["system_input"]
    if mutation == "count":
        record["trial_corpus_count"] += 1
    elif mutation == "boolean_count":
        record["trial_corpus_count"] = True
    elif mutation == "digest":
        record["trial_corpus_lineage_sha256"] = "sha256:" + "f" * 64
    elif mutation == "version":
        record["trial_corpus_versions"][0]["trial_version_id"] = "sha256:" + "f" * 64
    elif mutation == "extra":
        record["qrels"] = []
    elif mutation == "missing":
        del record["derived_views"]
    elif mutation == "mixed":
        record["trial_corpus_versions"][0] = source.trial_versions[0].to_dict()
    elif mutation == "encoding":
        record["trial_corpus_encoding"] = "unknown"
    elif mutation == "empty":
        record["trial_corpus_versions"] = []
    elif mutation == "duplicate":
        record["trial_corpus_versions"].append(record["trial_corpus_versions"][0])
    else:
        record["options"] = {"qrels_path": "/evaluator-only"}
    with pytest.raises(SchemaValidationError):
        RunManifest.from_dict(payload)


def test_compact_record_never_serializes_full_trial_content(monkeypatch):
    from taim.entity_versions import Trial
    from taim.patient_to_trial import PatientToTrialRunRequest

    legacy = _manifest()
    source = task_input_for(_PREPARED)
    request = PatientToTrialRunRequest(source, legacy.run_id, legacy.budget_k, legacy.budget_k, {})

    def forbidden(*args, **kwargs):
        raise AssertionError("compact records must not allocate full trial payloads")

    monkeypatch.setattr(Trial, "to_dict", forbidden)
    record = request.system_input_record_dict()
    compact = replace(legacy, configuration={"system_input": record})
    restored = RunManifest.from_json(compact.to_json())
    assert restored.resolve_task_input(source) is source


def test_a_forward_manifest_without_compact_lineage_is_refused():
    payload = _manifest().to_dict()
    system_input = payload["configuration"]["system_input"]
    for field_name in (
        "trial_corpus_encoding",
        "trial_corpus_count",
        "trial_corpus_lineage_sha256",
    ):
        del system_input[field_name]

    with pytest.raises(SchemaValidationError, match="trial_corpus_encoding lineage-v1"):
        RunManifest.from_dict(payload)


def test_compact_manifest_size_does_not_scale_with_trial_text():
    from taim.patient_to_trial import PatientToTrialRunRequest
    from tests.benchmark_helpers import trial

    sizes = []
    for text_size in (100, 100_000):
        prepared = prepared_benchmark(trials=(trial("NCT00000001", "x" * text_size),))
        legacy = replace(
            manifest_for(prepared, benchmark_profile=_PROFILE),
            runtime_seconds=0.0,
            candidates_sha256="sha256:" + "0" * 64,
        )
        source = task_input_for(prepared)
        request = PatientToTrialRunRequest(
            source, legacy.run_id, legacy.budget_k, legacy.budget_k, {}
        )
        compact = replace(
            legacy, configuration={"system_input": request.system_input_record_dict()}
        )
        sizes.append(len(compact.to_json().encode()))
        assert compact.task_input_id == legacy.task_input_id
        assert compact.system_input_id == legacy.system_input_id
    assert sizes[0] == sizes[1]
    # The record stays far smaller than the full Task Input content it stands for.
    assert len(compact.to_json()) < len(json.dumps(source.task_input_dict())) / 10


def test_compact_derived_view_lineage_avoids_content_and_verifies_source(monkeypatch):
    from taim.patient_to_trial import PatientToTrialRunRequest
    from taim.snapshot import DerivedView

    sizes = []
    sources = []
    for text_size in (100, 100_000):
        prepared = prepared_benchmark()
        view = DerivedView(
            name="corpus_view",
            version="1",
            input_snapshot_id=prepared.snapshot.snapshot_id,
            configuration={"recipe": "x" * text_size},
            content={prepared.trials[0].trial_id: "y" * text_size},
        )
        prepared = replace(
            prepared,
            snapshot=replace(
                prepared.snapshot,
                available_capabilities=prepared.snapshot.available_capabilities | {view.capability},
                derived_views=(view,),
            ),
        )
        legacy = replace(
            manifest_for(prepared, benchmark_profile=_PROFILE),
            runtime_seconds=0.0,
            candidates_sha256="sha256:" + "0" * 64,
        )
        source = task_input_for(prepared)
        sources.append(source)
        request = PatientToTrialRunRequest(
            source, legacy.run_id, legacy.budget_k, legacy.budget_k, {}
        )
        with monkeypatch.context() as patch:

            def forbidden(*args, **kwargs):
                raise AssertionError("compact records must not serialize Derived View content")

            patch.setattr(DerivedView, "to_dict", forbidden)
            compact = replace(
                legacy, configuration={"system_input": request.system_input_record_dict()}
            )
            restored = RunManifest.from_json(compact.to_json())
            assert restored.resolve_task_input(source) is source
        assert compact.task_input_id == legacy.task_input_id
        assert compact.system_input_id == legacy.system_input_id
        sizes.append(len(compact.to_json()))
    assert sizes[0] == sizes[1]
    # The record stays far smaller than the full Task Input content it stands for.
    assert len(compact.to_json()) < len(json.dumps(source.task_input_dict())) / 10
    with pytest.raises(SchemaValidationError, match="does not reproduce"):
        compact.resolve_task_input(sources[0])
    payload = compact.to_dict()
    payload["configuration"]["system_input"]["derived_views"][0]["view_id"] = "sha256:" + "f" * 64
    forged = RunManifest.from_dict(payload)
    with pytest.raises(SchemaValidationError, match="does not reproduce"):
        forged.resolve_task_input(source)
    payload["configuration"]["system_input"]["derived_views"][0]["content"] = "text"
    with pytest.raises(SchemaValidationError, match="Derived View fields"):
        RunManifest.from_dict(payload)
