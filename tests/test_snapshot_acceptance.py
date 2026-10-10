# ruff: noqa: S101
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from taim.data import load_prepared_benchmark
from taim.data.prepared import PreparedBenchmark
from taim.data.trec_ct_2021 import prepare_trec_ct_2021
from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.evaluation_package import EvaluationPackage
from taim.patient_to_trial import PatientToTrialRunRequest
from taim.profiles import OFFICIAL_FULL_PROFILE, resolve_benchmark_profile
from taim.schemas import RelevanceJudgment, SchemaValidationError
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    CAPABILITY_FIELD_PROVENANCE,
    CAPABILITY_PATIENT_EVIDENCE,
    CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
    CAPABILITY_TYPED_TRIAL_CORE,
    AgeQuantity,
    freeze_system_input_options,
)
from tests.snapshot_helpers import (
    TREC_FIXED_TIME,
    complete_trial_xml,
    write_trec_source_bundle,
)

CLINICAL_AS_OF = datetime(2021, 4, 27, tzinfo=UTC)


def _resolved(prepared: PreparedBenchmark):
    return resolve_benchmark_profile(prepared, profile_id=OFFICIAL_FULL_PROFILE)


@pytest.fixture(scope="module")
def prepared_for_option_carriers(
    tmp_path_factory: pytest.TempPathFactory,
) -> PreparedBenchmark:
    root = tmp_path_factory.mktemp("option-carriers")
    source, source_lock = write_trec_source_bundle(
        root,
        trials={"NCT00000001": complete_trial_xml()},
    )
    return load_prepared_benchmark(
        prepare_trec_ct_2021(
            source,
            root / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        ).output_directory
    )


def test_locked_trec_2021_reaches_a_judgment_free_system_request(tmp_path) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": complete_trial_xml()},
    )

    result = prepare_trec_ct_2021(
        source,
        tmp_path / "prepared",
        source_lock_path=source_lock,
        created_at=TREC_FIXED_TIME,
    )
    prepared = load_prepared_benchmark(result.output_directory)
    request = PatientToTrialRunRequest.for_benchmark(
        _resolved(prepared),
        run_id="snapshot-acceptance",
        top_k=10,
        metric_cutoff=10,
        options={},
        clinical_as_of=CLINICAL_AS_OF,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
        required_capabilities={
            CAPABILITY_CANONICAL_PATIENT_TEXT,
            CAPABILITY_CANONICAL_TRIAL_TEXT,
            CAPABILITY_FIELD_PROVENANCE,
            CAPABILITY_PATIENT_EVIDENCE,
            CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
            CAPABILITY_TYPED_TRIAL_CORE,
        },
    )

    snapshot = prepared.snapshot
    assert snapshot.contract_version == "2.0"
    assert snapshot.benchmark_lineage == "trec-ct-2021"
    assert snapshot.snapshot_name == "trec-ct-2021"
    assert snapshot.snapshot_id.startswith("sha256:")
    assert snapshot.snapshot_id != prepared.evaluation_package.evaluation_package_id
    assert prepared.preparation.source_bundle_id != prepared.source_bundle.source_bundle_id
    assert snapshot.available_capabilities == frozenset(
        {
            CAPABILITY_CANONICAL_PATIENT_TEXT,
            CAPABILITY_CANONICAL_TRIAL_TEXT,
            CAPABILITY_FIELD_PROVENANCE,
            CAPABILITY_PATIENT_EVIDENCE,
            CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
            CAPABILITY_TYPED_TRIAL_CORE,
        }
    )

    topic = snapshot.topics[0]
    assert topic.topic_id == "1"
    assert topic.source_identity.namespace == "trec-ct-2021.topics"
    assert topic.source_identity.record_id == "1"
    assert topic.canonical_text == "Adult with alpha syndrome."
    assert [(item.kind, item.ordinal, item.text) for item in topic.evidence_items] == [
        ("narrative", 0, "Adult with alpha syndrome.")
    ]
    assert topic.typed_patient_core is None

    trial = snapshot.trials[0]
    assert trial.trial_id == "NCT00000001"
    assert trial.source_identity.namespace == "clinicaltrials.gov"
    assert trial.source_identity.record_id == "NCT00000001"
    assert [(section.role, section.ordinal) for section in trial.sections] == [
        ("brief_title", 0),
        ("official_title", 1),
        ("summary", 2),
        ("summary", 3),
        ("condition", 4),
        ("condition", 5),
        ("intervention", 6),
        ("eligibility", 7),
    ]
    assert trial.canonical_text == (
        "[BRIEF_TITLE] Alpha treatment\n"
        "[OFFICIAL_TITLE] Official alpha study\n"
        "[SUMMARY] Tests alpha therapy.\n"
        "[SUMMARY] Extended alpha study description.\n"
        "[CONDITION] Alpha syndrome\n"
        "[CONDITION] Progressive disease\n"
        "[INTERVENTION] TYPE=Drug; NAME=AlphaMed; DESCRIPTION=Once daily\n"
        "[ELIGIBILITY] Adults with alpha syndrome. Excludes cardiac disease.\n"
        "[MINIMUM_AGE] 18 years\n"
        "[MAXIMUM_AGE] 65 years\n"
        "[SEX] all\n"
        "[HEALTHY_VOLUNTEERS] true"
    )
    assert trial.sections[3].provenance[0].location == (
        "/clinical_study/detailed_description[1]/textblock[1]"
    )
    assert trial.sections[-1].criteria == ()
    core = trial.typed_clinical_core
    assert core is not None
    assert core.minimum_age is not None
    assert core.minimum_age.value == AgeQuantity("18", "years")
    assert core.maximum_age is not None
    assert core.maximum_age.value == AgeQuantity("65", "years")
    assert core.sex is not None and core.sex.value == "all"
    assert core.healthy_volunteers is not None
    assert core.healthy_volunteers.value is True

    provenance = core.minimum_age.provenance[0]
    assert provenance.location == "/clinical_study/eligibility/minimum_age[1]"
    assert provenance.raw_value == "18 Years"
    assert provenance.transformation_rule == "ctgov-xml-age-v1"
    assert prepared.resolve_provenance(provenance).role == "trials_part_1"

    assert set(prepared.package_checksums) == {
        "diagnostics.jsonl",
        "judgments.jsonl",
        "topics.jsonl",
        "trials.jsonl",
    }
    assert prepared.diagnostics == ()
    assert [judgment.label for judgment in prepared.evaluation_package.judgments] == [2]
    assert prepared.evaluation_package.snapshot_id == snapshot.snapshot_id

    assert request.snapshot is not snapshot
    assert request.snapshot.prepared_snapshot_id == snapshot.snapshot_id
    assert request.snapshot.task_input_id != snapshot.snapshot_id
    assert not hasattr(request.snapshot, "system_input_id")
    assert request.system_input_id != request.snapshot.task_input_id
    assert request.snapshot.task_input_dict()["clinical_as_of"] == "2021-04-27T00:00:00Z"
    assert request.snapshot.patient_versions[0].patient_id == "1"
    assert request.snapshot.trial_versions[0].trial_id == "NCT00000001"
    assert not hasattr(request.snapshot, "snapshot_id")
    assert request.required_capabilities <= snapshot.available_capabilities
    assert not hasattr(request.snapshot, "snapshot_name")
    assert not hasattr(request.snapshot, "source_recipe_id")
    assert not hasattr(request.snapshot, "source_recipe_hash")
    assert not hasattr(request.snapshot, "source_bundle_id")
    assert not hasattr(request.snapshot, "qrels")
    assert not hasattr(request.snapshot, "judgments")
    assert "RelevanceJudgment" not in repr(request)
    assert "qrels" not in repr(request)

    rebuilt = replace(
        snapshot,
        snapshot_name="same-inputs-new-label",
    )
    rebuilt_request = PatientToTrialRunRequest.for_benchmark(
        _resolved(replace(prepared, snapshot=rebuilt)),
        run_id=request.run_id,
        top_k=request.top_k,
        metric_cutoff=request.metric_cutoff,
        options={},
        clinical_as_of=CLINICAL_AS_OF,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
        required_capabilities=request.required_capabilities,
    )
    assert rebuilt.snapshot_id == snapshot.snapshot_id
    assert rebuilt_request.snapshot == request.snapshot
    assert repr(rebuilt_request) == repr(request)


def test_system_request_options_reject_evaluator_material_recursively(
    prepared_for_option_carriers: PreparedBenchmark,
) -> None:
    with pytest.raises(SchemaValidationError, match="evaluator-only"):
        PatientToTrialRunRequest.for_benchmark(
            _resolved(prepared_for_option_carriers),
            run_id="leaky-options",
            top_k=10,
            metric_cutoff=10,
            options={"adapter": {"qrels": ["1 0 NCT00000001 2"]}},
            clinical_as_of=CLINICAL_AS_OF,
            patient_evidence_profile=source_grounded_patient_evidence_profile(),
        )


@pytest.mark.parametrize(
    "carrier",
    (
        lambda prepared: prepared,
        lambda prepared: prepared.evaluation_package,
        lambda prepared: prepared.evaluation_package.judgments[0],
        lambda prepared: {"nested": [prepared]},
    ),
    ids=("prepared-benchmark", "evaluation-package", "judgment", "nested-prepared"),
)
def test_system_request_options_reject_object_carriers_before_serialization(
    prepared_for_option_carriers: PreparedBenchmark, carrier
) -> None:
    prepared = prepared_for_option_carriers

    assert isinstance(prepared, PreparedBenchmark)
    assert isinstance(prepared.evaluation_package, EvaluationPackage)
    assert isinstance(prepared.evaluation_package.judgments[0], RelevanceJudgment)
    with pytest.raises(SchemaValidationError, match=r"evaluator-only|safe option value"):
        PatientToTrialRunRequest.for_benchmark(
            _resolved(prepared),
            run_id="object-carrier",
            top_k=10,
            metric_cutoff=10,
            options={"adapter": carrier(prepared)},
            clinical_as_of=CLINICAL_AS_OF,
            patient_evidence_profile=source_grounded_patient_evidence_profile(),
        )


def test_system_request_options_reject_serialized_evaluator_carriers(
    prepared_for_option_carriers: PreparedBenchmark,
) -> None:
    prepared = prepared_for_option_carriers
    for carrier in (
        {"topic_id": "1", "trial_id": "NCT00000001", "label": 2},
        prepared.evaluation_package.judgments[0].to_dict(),
        prepared.evaluation_package.manifest_dict(),
    ):
        with pytest.raises(SchemaValidationError, match="evaluator-only"):
            PatientToTrialRunRequest.for_benchmark(
                _resolved(prepared),
                run_id="serialized-evaluator-carrier",
                top_k=10,
                metric_cutoff=10,
                options={"adapter": {"serialized": carrier}},
                clinical_as_of=CLINICAL_AS_OF,
                patient_evidence_profile=source_grounded_patient_evidence_profile(),
            )


def test_system_request_options_preserve_safe_pipeline_configuration_values() -> None:
    options = freeze_system_input_options(
        {
            "synthetic-system": {"k1": 1.2, "b": 0.75},
            "synthetic-vector": {"query_batch_size": 32, "index": Path("dense-index")},
            "synthetic-fusion": {
                "synthetic-system_run": Path("synthetic-system-run"),
                "dense_run": Path("dense-run"),
            },
            "feature_flags": [True, None, "stable"],
        }
    )

    assert options["synthetic-system"] == {"k1": 1.2, "b": 0.75}
    assert options["synthetic-vector"]["index"] == Path("dense-index")  # type: ignore[index]
    assert options["synthetic-fusion"]["synthetic-system_run"] == Path("synthetic-system-run")  # type: ignore[index]
    assert options["feature_flags"] == (True, None, "stable")


def test_evaluation_changes_do_not_change_system_inputs(tmp_path) -> None:
    first_source, first_lock = write_trec_source_bundle(
        tmp_path / "first",
        trials={"NCT00000001": complete_trial_xml()},
        qrels="1 0 NCT00000001 2\n",
    )
    second_source, second_lock = write_trec_source_bundle(
        tmp_path / "second",
        trials={"NCT00000001": complete_trial_xml()},
        qrels="1 0 NCT00000001 1\n",
    )

    first = load_prepared_benchmark(
        prepare_trec_ct_2021(
            first_source,
            tmp_path / "first-prepared",
            source_lock_path=first_lock,
            created_at=TREC_FIXED_TIME,
        ).output_directory
    )
    second = load_prepared_benchmark(
        prepare_trec_ct_2021(
            second_source,
            tmp_path / "second-prepared",
            source_lock_path=second_lock,
            created_at=TREC_FIXED_TIME,
        ).output_directory
    )

    assert first.snapshot.snapshot_id == second.snapshot.snapshot_id
    assert first.preparation.source_bundle_id == second.preparation.source_bundle_id
    assert first.source_bundle.source_bundle_id != second.source_bundle.source_bundle_id
    assert (
        first.evaluation_package.evaluation_package_id
        != second.evaluation_package.evaluation_package_id
    )
