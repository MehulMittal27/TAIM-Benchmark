# ruff: noqa: S101
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from taim.constraint_facts import (
    CONSTRAINT_FACT_VIEW_NAME,
    ConstraintExtractionError,
    FactName,
    PatientCoreFactMapping,
    RuleBasedConstraintFactExtractor,
    build_constraint_fact_view,
    load_constraint_fact_view,
)
from taim.entity_versions import (
    resolve_task_derived_views,
    source_grounded_patient_evidence_profile,
    version_patients,
    version_trials,
)
from taim.schemas import SchemaValidationError, json_value_to_builtins
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    CAPABILITY_FIELD_PROVENANCE,
    CAPABILITY_PATIENT_EVIDENCE,
    CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
    CAPABILITY_TYPED_PATIENT_CORE,
    CAPABILITY_TYPED_TRIAL_CORE,
    AgeBound,
    AgeQuantity,
    BenchmarkSnapshot,
    CoreFieldConflict,
    DerivedView,
    FieldProvenance,
    HealthyVolunteerAcceptance,
    PatientCoreAssertion,
    PatientEvidenceItem,
    SemanticTextSection,
    SexEligibility,
    TypedClinicalCore,
    TypedPatientCore,
)
from tests.benchmark_helpers import topic, trial

QUALIFICATION_FIXTURE = Path(__file__).parent / "fixtures" / "constraint-facts" / "v1.json"
SYNTHETIC_PATIENT_CORE_MAPPING = PatientCoreFactMapping(
    mapping_id="synthetic-patient-core-constraint-facts",
    version="1.0",
    profiles={
        "synthetic-patient-core-v1": {
            "age": "age",
            "sex": "sex",
            "a.age": "age",
            "z.age": "age",
            "a.sex": "sex",
            "b.sex": "sex",
        }
    },
)


def _provenance(raw_value: str, *, location: str = "/topic[1]") -> FieldProvenance:
    return FieldProvenance(
        artifact_id="synthetic-topics",
        artifact_sha256="sha256:" + "b" * 64,
        source_record_id="1",
        locator_kind="xml_path",
        location=location,
        raw_value=raw_value,
        transformation_rule="synthetic-source-v1",
    )


def _topic(
    text: str,
    *,
    structured: dict[str, PatientCoreAssertion] | None = None,
    patient_core_profile_id: str = "synthetic-patient-core-v1",
):
    source = _provenance(text)
    return replace(
        topic("1", text),
        evidence_items=(PatientEvidenceItem("narrative", 0, text, (source,)),),
        typed_patient_core=(
            TypedPatientCore(patient_core_profile_id, structured)
            if structured is not None
            else None
        ),
    )


def _snapshot(patient_topic, *, trial_document=None) -> BenchmarkSnapshot:
    capabilities = {
        CAPABILITY_CANONICAL_PATIENT_TEXT,
        CAPABILITY_CANONICAL_TRIAL_TEXT,
        CAPABILITY_FIELD_PROVENANCE,
        CAPABILITY_PATIENT_EVIDENCE,
    }
    if patient_topic.typed_patient_core is not None:
        capabilities.add(CAPABILITY_TYPED_PATIENT_CORE)
    trial_row = trial_document or trial("NCT-A", "synthetic trial")
    if trial_row.sections:
        capabilities.add(CAPABILITY_SEMANTIC_TRIAL_SECTIONS)
    if trial_row.typed_clinical_core is not None:
        capabilities.add(CAPABILITY_TYPED_TRIAL_CORE)
    return BenchmarkSnapshot(
        benchmark_lineage="synthetic-constraint-facts",
        snapshot_name="synthetic-constraint-facts",
        topics=(patient_topic,),
        trials=(trial_row,),
        available_capabilities=frozenset(capabilities),
    )


def _topic_facts(view) -> dict[str, object]:
    content = json_value_to_builtins(view.content)
    return content["topics"][0]["facts"]


def _with_recomputed_identity(payload: dict[str, object]) -> dict[str, object]:
    return DerivedView(
        name=payload["name"],
        version=payload["version"],
        input_snapshot_id=payload["input_snapshot_id"],
        configuration=payload["configuration"],
        content=payload["content"],
    ).to_dict()


def test_constraint_fact_view_projects_records_into_entity_versions() -> None:
    snapshot = _snapshot(_topic("A 40-year-old patient."))
    view = build_constraint_fact_view(snapshot)

    patient = version_patients(
        snapshot.topics,
        source_grounded_patient_evidence_profile(),
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
        derived_views=(view,),
    )[0]
    trial_version = version_trials(snapshot.trials, derived_views=(view,))[0]

    patient_record = patient.derived_evidence[CONSTRAINT_FACT_VIEW_NAME]["content"]
    trial_record = trial_version.derived_evidence[CONSTRAINT_FACT_VIEW_NAME]["content"]
    assert patient_record["record_id"] == snapshot.topics[0].topic_id
    assert set(patient_record["facts"]) == {"age", "sex", "healthy_volunteer", "smoking", "alcohol"}
    assert trial_record["record_id"] == snapshot.trials[0].trial_id
    assert set(trial_record["facts"]) == {
        "minimum_age",
        "maximum_age",
        "sex",
        "healthy_volunteers",
        "smoking",
        "alcohol",
    }


def test_constraint_patient_facts_are_hidden_when_profile_hides_their_source_evidence() -> None:
    snapshot = _snapshot(_topic("A 40-year-old patient."))
    view = build_constraint_fact_view(snapshot)
    source_profile = source_grounded_patient_evidence_profile()
    profile = replace(
        source_profile,
        definition={
            **dict(source_profile.definition),
            "visible_patient_fields": ["canonical_text", "derived_evidence"],
        },
    )

    (resolved,) = resolve_task_derived_views(
        (view,),
        topics=snapshot.topics,
        trials=snapshot.trials,
        profile=profile,
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
    )
    content = json_value_to_builtins(resolved.content)

    assert content["topics"] == []
    assert [record["record_id"] for record in content["trials"]] == ["NCT-A"]


def test_structured_age_precedes_conflicting_text_age_with_source_provenance() -> None:
    text = "A 50-year-old patient."
    structured_source = _provenance("45 Years", location="/patient/age")
    snapshot = _snapshot(
        _topic(
            text,
            structured={
                "age": PatientCoreAssertion(
                    {"amount": "45", "unit": "years"},
                    (structured_source,),
                )
            },
        )
    )

    view = build_constraint_fact_view(
        snapshot,
        patient_core_mapping=SYNTHETIC_PATIENT_CORE_MAPPING,
    )
    age = _topic_facts(view)["age"]

    assert view.name == CONSTRAINT_FACT_VIEW_NAME
    assert view.input_snapshot_id == snapshot.base_snapshot_id
    assert view.configuration["patient_core_fact_mapping"] == {
        "id": "synthetic-patient-core-constraint-facts",
        "version": "1.0",
        "profiles": SYNTHETIC_PATIENT_CORE_MAPPING.to_dict()["profiles"],
    }
    assert age["state"] == "known"
    assert age["value"] == {"amount": "45", "unit": "years"}
    assert age["resolution"] == "structured_precedence"
    assert age["evidence"][0]["source_field"] == "typed_patient_core.age"
    assert age["evidence"][0]["provenance"] == [structured_source.to_dict()]


def test_text_fallback_extracts_unit_age_and_negated_reusable_facts_with_spans() -> None:
    text = (
        "A 6-month-old female patient who is not a healthy volunteer, does not smoke, "
        "and does not drink alcohol."
    )
    view = build_constraint_fact_view(_snapshot(_topic(text)))
    facts = _topic_facts(view)

    assert facts["age"]["value"] == {"amount": "6", "unit": "months"}
    assert facts["sex"]["value"] == "female"
    assert facts["healthy_volunteer"]["value"] is False
    assert facts["smoking"]["value"] is False
    assert facts["alcohol"]["value"] is False
    for fact_name in ("age", "sex", "healthy_volunteer", "smoking", "alcohol"):
        fact = facts[fact_name]
        assert fact["resolution"] == "text_fallback"
        evidence = fact["evidence"][0]
        span = evidence["text_span"]
        assert text[span["start"] : span["end"]] == span["text"]
        assert evidence["provenance"]


@pytest.mark.parametrize(
    ("profile_id", "field_name"),
    [
        ("unmapped-patient-core-v1", "age"),
        ("synthetic-patient-core-v1", "unmapped.age"),
    ],
)
def test_unmapped_patient_core_profile_or_field_does_not_receive_structured_precedence(
    profile_id: str,
    field_name: str,
) -> None:
    source = _provenance("45 Years", location="/patient/age")
    view = build_constraint_fact_view(
        _snapshot(
            _topic(
                "Patient age is not stated.",
                structured={
                    field_name: PatientCoreAssertion(
                        {"amount": "45", "unit": "years"},
                        (source,),
                    )
                },
                patient_core_profile_id=profile_id,
            )
        ),
        patient_core_mapping=SYNTHETIC_PATIENT_CORE_MAPPING,
    )

    assert _topic_facts(view)["age"] == {
        "state": "unknown",
        "reason": "no_usable_evidence",
        "evidence": [],
    }


def test_conflicting_structured_values_remain_unknown_and_text_cannot_choose_a_winner() -> None:
    source = _provenance("structured sex", location="/patient/sex")
    view = build_constraint_fact_view(
        _snapshot(
            _topic(
                "Female patient.",
                structured={
                    "a.sex": PatientCoreAssertion("female", (source,)),
                    "b.sex": PatientCoreAssertion("male", (source,)),
                },
            )
        ),
        patient_core_mapping=SYNTHETIC_PATIENT_CORE_MAPPING,
    )
    sex = _topic_facts(view)["sex"]

    assert sex["state"] == "unknown"
    assert sex["reason"] == "structured_precedence_conflict"
    assert [item["source_field"] for item in sex["evidence"]] == [
        "typed_patient_core.a.sex",
        "typed_patient_core.b.sex",
    ]
    assert {item["source_kind"] for item in sex["evidence"]} == {"structured"}


def test_trial_core_conflict_is_system_visible_and_blocks_text_fallback() -> None:
    lower_source = _provenance("65 Years", location="/trial/maximum_age[1]")
    upper_source = _provenance("70 Years", location="/trial/maximum_age[2]")
    section_text = "Maximum age: 80 years."
    section_source = _provenance(section_text, location="/trial/eligibility/text")
    conflict = CoreFieldConflict(
        field_name="maximum_age",
        values=(
            {"amount": "65", "unit": "years"},
            {"amount": "70", "unit": "years"},
        ),
        provenance=(lower_source, upper_source),
    )
    trial_row = replace(
        trial("NCT-A", "canonical trial"),
        sections=(SemanticTextSection("eligibility", section_text, 0, (section_source,)),),
        typed_clinical_core=TypedClinicalCore(conflicts=(conflict,)),
    )
    snapshot = _snapshot(_topic("A 40-year-old patient."), trial_document=trial_row)

    assert snapshot.trials[0].to_dict()["typed_clinical_core"]["conflicts"] == [conflict.to_dict()]
    maximum_age = json_value_to_builtins(build_constraint_fact_view(snapshot).content)["trials"][0][
        "facts"
    ]["maximum_age"]

    assert maximum_age["state"] == "unknown"
    assert maximum_age["reason"] == "structured_precedence_conflict"
    assert [item["normalized_value"] for item in maximum_age["evidence"]] == [
        {"amount": "65", "unit": "years"},
        {"amount": "70", "unit": "years"},
    ]
    assert {item["source_kind"] for item in maximum_age["evidence"]} == {"structured"}


def test_equivalent_structured_assertions_coalesce_deterministically() -> None:
    source = _provenance("45 Years", location="/patient/age")
    facts = _topic_facts(
        build_constraint_fact_view(
            _snapshot(
                _topic(
                    "Patient age is not restated.",
                    structured={
                        "z.age": PatientCoreAssertion({"amount": "45", "unit": "Years"}, (source,)),
                        "a.age": PatientCoreAssertion({"amount": 45, "unit": "years"}, (source,)),
                    },
                )
            ),
            patient_core_mapping=SYNTHETIC_PATIENT_CORE_MAPPING,
        )
    )

    assert facts["age"]["state"] == "known"
    assert facts["age"]["value"] == {"amount": "45", "unit": "years"}
    assert [item["source_field"] for item in facts["age"]["evidence"]] == [
        "typed_patient_core.a.age",
        "typed_patient_core.z.age",
    ]


def test_unitless_structured_age_is_invalid_and_remains_unknown() -> None:
    source = _provenance("45", location="/patient/age")
    age = _topic_facts(
        build_constraint_fact_view(
            _snapshot(
                _topic(
                    "Patient age is not restated.",
                    structured={"age": PatientCoreAssertion(45, (source,))},
                )
            ),
            patient_core_mapping=SYNTHETIC_PATIENT_CORE_MAPPING,
        )
    )["age"]

    assert age["state"] == "unknown"
    assert age["reason"] == "invalid_structured_evidence"


def test_invalid_structured_value_is_preserved_before_usable_text_fallback() -> None:
    source = _provenance("unknown", location="/patient/sex")
    sex = _topic_facts(
        build_constraint_fact_view(
            _snapshot(
                _topic(
                    "Female patient.",
                    structured={"sex": PatientCoreAssertion("unknown", (source,))},
                )
            ),
            patient_core_mapping=SYNTHETIC_PATIENT_CORE_MAPPING,
        )
    )["sex"]

    assert sex["state"] == "known"
    assert sex["value"] == "female"
    assert sex["resolution"] == "text_fallback"
    assert [item["outcome"] for item in sex["evidence"]] == ["invalid", "valid"]


def test_trial_facts_preserve_typed_core_and_use_text_only_for_missing_concepts() -> None:
    structured_source = _provenance("structured trial core", location="/trial/eligibility")
    section_text = "No smoking and no alcohol use."
    section_source = _provenance(section_text, location="/trial/eligibility/text")
    trial_row = replace(
        trial("NCT-A", "canonical trial"),
        typed_clinical_core=TypedClinicalCore(
            minimum_age=AgeBound(AgeQuantity("18", "years"), (structured_source,)),
            maximum_age=AgeBound("unbounded", (structured_source,)),
            sex=SexEligibility("all", (structured_source,)),
            healthy_volunteers=HealthyVolunteerAcceptance(True, (structured_source,)),
        ),
        sections=(
            SemanticTextSection(
                "eligibility",
                section_text,
                0,
                (section_source,),
            ),
        ),
    )
    view = build_constraint_fact_view(
        _snapshot(_topic("A 40-year-old patient."), trial_document=trial_row)
    )
    content = json_value_to_builtins(view.content)
    facts = content["trials"][0]["facts"]

    assert facts["minimum_age"]["value"] == {"amount": "18", "unit": "years"}
    assert facts["maximum_age"]["value"] == "unbounded"
    assert facts["sex"]["value"] == "all"
    assert facts["healthy_volunteers"]["value"] is True
    assert facts["smoking"]["value"] is False
    assert facts["alcohol"]["value"] is False
    assert facts["smoking"]["resolution"] == "text_fallback"


class _FailingExtractor(RuleBasedConstraintFactExtractor):
    extractor_id = "synthetic-failing-extractor"
    version = "failure-v1"
    implementation_id = "tests.test_constraint_facts._FailingExtractor"

    def __init__(self) -> None:
        self.configuration = {"failure_mode": "always"}

    def extract(self, text: str, *, record_kind, requested_facts):
        del text, record_kind, requested_facts
        raise ConstraintExtractionError("synthetic extractor failure")


class _RequestedFactsSpyExtractor(RuleBasedConstraintFactExtractor):
    extractor_id = "synthetic-requested-facts-spy"
    version = "1.0"
    implementation_id = "tests.test_constraint_facts._RequestedFactsSpyExtractor"

    def __init__(self) -> None:
        self.configuration = {"failure_mode": "covered_fact_requested"}
        self.requests: list[tuple[str, frozenset[str]]] = []

    def extract(self, text: str, *, record_kind, requested_facts):
        if {"age", "sex"} & requested_facts:
            raise AssertionError("covered structured fact was requested from text")
        self.requests.append((record_kind, requested_facts))
        return super().extract(
            text,
            record_kind=record_kind,
            requested_facts=requested_facts,
        )


def test_text_extractor_receives_only_facts_missing_usable_structured_assertions() -> None:
    source = _provenance("structured patient core", location="/patient/core")
    extractor = _RequestedFactsSpyExtractor()
    view = build_constraint_fact_view(
        _snapshot(
            _topic(
                "A 50-year-old female patient.",
                structured={
                    "age": PatientCoreAssertion(
                        {"amount": "45", "unit": "years"},
                        (source,),
                    ),
                    "a.sex": PatientCoreAssertion("female", (source,)),
                    "b.sex": PatientCoreAssertion("male", (source,)),
                },
            )
        ),
        extractor=extractor,
        patient_core_mapping=SYNTHETIC_PATIENT_CORE_MAPPING,
    )
    facts = _topic_facts(view)

    assert extractor.requests == [("topic", frozenset({"healthy_volunteer", "smoking", "alcohol"}))]
    assert facts["age"]["value"] == {"amount": "45", "unit": "years"}
    assert facts["sex"]["reason"] == "structured_precedence_conflict"


def test_extractor_failure_fails_open_with_versioned_failure_provenance() -> None:
    view = build_constraint_fact_view(
        _snapshot(_topic("A 45-year-old female patient.")),
        extractor=_FailingExtractor(),
    )
    facts = _topic_facts(view)

    assert view.configuration["extractor"] == {
        "id": "synthetic-failing-extractor",
        "version": "failure-v1",
        "implementation": "tests.test_constraint_facts._FailingExtractor",
        "configuration": {"failure_mode": "always"},
    }
    assert all(fact["state"] == "unknown" for fact in facts.values())
    assert all(fact["reason"] == "extractor_failure" for fact in facts.values())
    assert all(fact["evidence"][0]["outcome"] == "extractor_failure" for fact in facts.values())


def test_unsupported_age_and_lifestyle_wording_remains_unknown_with_span() -> None:
    text = "A 131-year-old former smoker with social alcohol intake."
    facts = _topic_facts(build_constraint_fact_view(_snapshot(_topic(text))))

    assert facts["age"]["state"] == "unknown"
    assert facts["smoking"]["state"] == "unknown"
    assert facts["smoking"]["reason"] == "unsupported_text_evidence"
    span = facts["smoking"]["evidence"][0]["text_span"]
    assert text[span["start"] : span["end"]] == "former smoker"
    assert facts["alcohol"]["state"] == "unknown"


def test_constraint_fact_view_round_trips_and_rejects_provenance_tampering() -> None:
    view = build_constraint_fact_view(_snapshot(_topic("A 45-year-old female patient.")))
    serialized = view.to_dict()

    assert load_constraint_fact_view(serialized) == view
    assert load_constraint_fact_view(json.loads(json.dumps(serialized, sort_keys=True))) == view

    tampered = deepcopy(serialized)
    tampered["content"]["topics"][0]["facts"]["age"]["evidence"][0]["provenance"][0][
        "raw_value"
    ] = "tampered"
    with pytest.raises(SchemaValidationError, match="identity does not match"):
        load_constraint_fact_view(tampered)

    invalid_mapping = deepcopy(serialized)
    invalid_mapping["configuration"]["patient_core_fact_mapping"]["profiles"] = {
        "synthetic-patient-core-v1": {"age": "minimum_age"}
    }
    with pytest.raises(SchemaValidationError, match="patient core mapping is invalid"):
        load_constraint_fact_view(_with_recomputed_identity(invalid_mapping))


def test_view_identity_changes_with_extractor_configuration_and_snapshot_input() -> None:
    snapshot = _snapshot(_topic("A 45-year-old patient."))
    original = build_constraint_fact_view(snapshot)

    class _RevisionExtractor(RuleBasedConstraintFactExtractor):
        version = "1.1"
        implementation_id = "tests.test_constraint_facts._RevisionExtractor"

    changed_extractor = build_constraint_fact_view(snapshot, extractor=_RevisionExtractor())
    changed_input = build_constraint_fact_view(_snapshot(_topic("A 46-year-old patient.")))

    assert len({original.view_id, changed_extractor.view_id, changed_input.view_id}) == 3


def test_view_identity_binds_extractor_implementation_and_configuration() -> None:
    snapshot = _snapshot(_topic("A 45-year-old patient."))

    class _ConfiguredExtractor(RuleBasedConstraintFactExtractor):
        extractor_id = "same-declared-extractor"
        version = "same-version"
        implementation_id = "tests.test_constraint_facts._ConfiguredExtractor"

        def __init__(self, revision: str) -> None:
            self.configuration = {"revision": revision}

    first = build_constraint_fact_view(snapshot, extractor=_ConfiguredExtractor("a"))
    second = build_constraint_fact_view(snapshot, extractor=_ConfiguredExtractor("b"))

    assert _topic_facts(first)["age"]["value"] == _topic_facts(second)["age"]["value"]
    assert first.view_id != second.view_id


def test_loader_rejects_identity_valid_but_semantically_invalid_fact_content() -> None:
    view = build_constraint_fact_view(_snapshot(_topic("A 45-year-old female patient.")))
    payload = deepcopy(view.to_dict())
    payload["content"]["topics"][0]["facts"]["age"]["value"] = {
        "amount": "45",
        "unit": "fortnights",
    }
    repaired_identity = _with_recomputed_identity(payload)

    with pytest.raises(SchemaValidationError, match="age value is invalid"):
        load_constraint_fact_view(repaired_identity)


def test_loader_rejects_mixed_valid_sources_for_structured_precedence() -> None:
    source = _provenance("45 Years", location="/patient/age")
    structured_view = build_constraint_fact_view(
        _snapshot(
            _topic(
                "Patient age is not restated.",
                structured={
                    "age": PatientCoreAssertion(
                        {"amount": "45", "unit": "years"},
                        (source,),
                    )
                },
            )
        ),
        patient_core_mapping=SYNTHETIC_PATIENT_CORE_MAPPING,
    )
    text_view = build_constraint_fact_view(_snapshot(_topic("A 45-year-old patient.")))
    payload = deepcopy(structured_view.to_dict())
    payload["content"]["topics"][0]["facts"]["age"]["evidence"].append(
        text_view.to_dict()["content"]["topics"][0]["facts"]["age"]["evidence"][0]
    )

    with pytest.raises(SchemaValidationError, match="resolution evidence source is inconsistent"):
        load_constraint_fact_view(_with_recomputed_identity(payload))


def test_polarity_sensitive_text_fallback_does_not_invert_trial_constraints() -> None:
    section_text = (
        "Healthy Volunteers: No. Male participants are excluded. "
        "Smoking cessation intervention. Alcohol use is a secondary outcome."
    )
    section_source = _provenance(section_text, location="/trial/eligibility/text")
    trial_row = replace(
        trial("NCT-A", "canonical trial"),
        sections=(SemanticTextSection("eligibility", section_text, 0, (section_source,)),),
    )
    content = json_value_to_builtins(
        build_constraint_fact_view(
            _snapshot(_topic("A 40-year-old patient."), trial_document=trial_row)
        ).content
    )
    facts = content["trials"][0]["facts"]

    assert facts["healthy_volunteers"]["value"] is False
    assert facts["sex"] == {
        "state": "unknown",
        "reason": "no_usable_evidence",
        "evidence": [],
    }
    assert facts["smoking"]["state"] == "unknown"
    assert facts["alcohol"]["state"] == "unknown"


@pytest.mark.parametrize(
    ("section_text", "rejected_fact_names"),
    [
        (
            "Participants aged 18 to 65 years are excluded.",
            ("minimum_age", "maximum_age"),
        ),
        ("Minimum age: 18 years is excluded.", ("minimum_age",)),
    ],
)
def test_rejected_trial_age_assertions_do_not_become_constraint_facts(
    section_text: str,
    rejected_fact_names: tuple[str, ...],
) -> None:
    section_source = _provenance(section_text, location="/trial/eligibility/text")
    trial_row = replace(
        trial("NCT-A", "canonical trial"),
        sections=(SemanticTextSection("eligibility", section_text, 0, (section_source,)),),
    )
    content = json_value_to_builtins(
        build_constraint_fact_view(
            _snapshot(_topic("A 40-year-old patient."), trial_document=trial_row)
        ).content
    )

    for fact_name in rejected_fact_names:
        assert content["trials"][0]["facts"][fact_name] == {
            "state": "unknown",
            "reason": "no_usable_evidence",
            "evidence": [],
        }


@pytest.mark.parametrize(
    ("assertion", "requested_facts", "expected"),
    [
        (
            "Participants aged 18 to 65 years.",
            frozenset({"minimum_age", "maximum_age"}),
            [
                ("minimum_age", {"amount": "18", "unit": "years"}),
                ("maximum_age", {"amount": "65", "unit": "years"}),
            ],
        ),
        (
            "Minimum age: 18 years.",
            frozenset({"minimum_age"}),
            [("minimum_age", {"amount": "18", "unit": "years"})],
        ),
        ("Sex: Male.", frozenset({"sex"}), [("sex", "male")]),
        (
            "Healthy Volunteers: Yes.",
            frozenset({"healthy_volunteer"}),
            [("healthy_volunteer", True)],
        ),
        (
            "Healthy Volunteers: No.",
            frozenset({"healthy_volunteer"}),
            [("healthy_volunteer", False)],
        ),
        ("Smokers are eligible.", frozenset({"smoking"}), [("smoking", True)]),
        ("Non-smokers are eligible.", frozenset({"smoking"}), [("smoking", False)]),
        ("Drinkers are eligible.", frozenset({"alcohol"}), [("alcohol", True)]),
        ("Non-drinkers are eligible.", frozenset({"alcohol"}), [("alcohol", False)]),
    ],
)
def test_trial_extraction_respects_criteria_heading_polarity(
    assertion: str,
    requested_facts: frozenset[FactName],
    expected: list[tuple[FactName, object]],
) -> None:
    extractor = RuleBasedConstraintFactExtractor()

    excluded = extractor.extract(
        f"Exclusion criteria: {assertion}",
        record_kind="trial",
        requested_facts=requested_facts,
    )
    included = extractor.extract(
        f"Inclusion criteria: {assertion}",
        record_kind="trial",
        requested_facts=requested_facts,
    )
    included_after_exclusion = extractor.extract(
        f"Exclusion criteria: Unrelated. Inclusion criteria: {assertion}",
        record_kind="trial",
        requested_facts=requested_facts,
    )

    assert excluded == ()
    assert [(item.fact_name, item.value) for item in included] == expected
    assert [(item.fact_name, item.value) for item in included_after_exclusion] == expected


def test_criteria_heading_polarity_does_not_apply_to_topic_extraction() -> None:
    assertions = RuleBasedConstraintFactExtractor().extract(
        "Exclusion criteria: Female patient.",
        record_kind="topic",
        requested_facts=frozenset({"sex"}),
    )

    assert [(item.fact_name, item.value) for item in assertions] == [("sex", "female")]


def test_negated_assertions_do_not_become_affirmative_facts() -> None:
    text = (
        "This is not a male patient. Patient is not currently smoking. "
        "Patient quit smoking. Alcohol use was discontinued."
    )
    facts = _topic_facts(build_constraint_fact_view(_snapshot(_topic(text))))

    for fact_name in ("sex", "smoking", "alcohol"):
        assert facts[fact_name] == {
            "state": "unknown",
            "reason": "no_usable_evidence",
            "evidence": [],
        }


def test_qualified_plural_lifestyle_negation_resolves_false() -> None:
    section_text = "Non-smokers are eligible. Non-drinkers are eligible."
    section_source = _provenance(section_text, location="/trial/eligibility/text")
    trial_row = replace(
        trial("NCT-A", "canonical trial"),
        sections=(SemanticTextSection("eligibility", section_text, 0, (section_source,)),),
    )
    content = json_value_to_builtins(
        build_constraint_fact_view(
            _snapshot(_topic("A 40-year-old patient."), trial_document=trial_row)
        ).content
    )
    facts = content["trials"][0]["facts"]

    assert facts["smoking"]["value"] is False
    assert facts["alcohol"]["value"] is False


def test_loader_rejects_invalid_text_span_and_reason_evidence_pairing() -> None:
    text_view = build_constraint_fact_view(_snapshot(_topic("A 45-year-old female patient.")))
    invalid_span = deepcopy(text_view.to_dict())
    invalid_span["content"]["topics"][0]["facts"]["age"]["evidence"][0]["text_span"]["end"] += 1
    with pytest.raises(SchemaValidationError, match="text_span is invalid"):
        load_constraint_fact_view(_with_recomputed_identity(invalid_span))

    source = _provenance("45", location="/patient/age")
    invalid_age_view = build_constraint_fact_view(
        _snapshot(
            _topic(
                "Patient age is not restated.",
                structured={"age": PatientCoreAssertion(45, (source,))},
            )
        ),
        patient_core_mapping=SYNTHETIC_PATIENT_CORE_MAPPING,
    )
    invalid_reason = deepcopy(invalid_age_view.to_dict())
    invalid_reason["content"]["topics"][0]["facts"]["age"]["reason"] = "extractor_failure"
    with pytest.raises(SchemaValidationError, match="lacks extractor failure evidence"):
        load_constraint_fact_view(_with_recomputed_identity(invalid_reason))


def test_pinned_extractor_matches_qrel_free_source_level_references() -> None:
    fixture = json.loads(QUALIFICATION_FIXTURE.read_text(encoding="utf-8"))
    extractor = RuleBasedConstraintFactExtractor()

    assert fixture["extractor"] == {
        "id": extractor.extractor_id,
        "version": extractor.version,
    }
    assert fixture["qualification"]["ranking_qrels_used"] is False
    for reference in fixture["references"]:
        text = reference["text"]
        observed = [
            {
                "fact_name": item.fact_name,
                "value": item.value,
                "text": text[item.start : item.end],
            }
            for item in extractor.extract(
                text,
                record_kind=reference["record_kind"],
                requested_facts=frozenset(
                    ("age", "sex", "healthy_volunteer", "smoking", "alcohol")
                    if reference["record_kind"] == "topic"
                    else (
                        "minimum_age",
                        "maximum_age",
                        "sex",
                        "healthy_volunteer",
                        "smoking",
                        "alcohol",
                    )
                ),
            )
        ]
        assert observed == reference["expected"], reference["id"]


def test_view_is_per_record_only_and_does_not_mutate_source_grounded_inputs() -> None:
    patient_topic = _topic(
        "A 45-year-old patient.",
        structured={
            "age": PatientCoreAssertion(
                {"amount": "45", "unit": "years"},
                (_provenance("45 Years", location="/patient/age"),),
            )
        },
    )
    snapshot = _snapshot(patient_topic)
    original_topic = patient_topic.to_dict()
    original_trial = snapshot.trials[0].to_dict()

    view = build_constraint_fact_view(
        snapshot,
        patient_core_mapping=SYNTHETIC_PATIENT_CORE_MAPPING,
    )
    content = json_value_to_builtins(view.content)
    serialized = json.dumps(content, sort_keys=True)

    assert patient_topic.to_dict() == original_topic
    assert snapshot.trials[0].to_dict() == original_trial
    assert set(content) == {"schema_version", "topics", "trials"}
    assert all(set(record) == {"record_id", "facts"} for record in content["topics"])
    assert all(set(record) == {"record_id", "facts"} for record in content["trials"])
    assert all(term not in serialized for term in ("pairwise", "eligibility_decision", "disease"))
