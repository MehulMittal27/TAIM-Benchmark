# ruff: noqa: S101
from __future__ import annotations

import hashlib
from dataclasses import replace

from taim.data.trec_ct_2021 import extract_complete_eligibility_text
from taim.eligibility import (
    ELIGIBILITY_CRITERION_VIEW_NAME,
    analyze_complete_eligibility,
    build_eligibility_criterion_trial_record,
    build_eligibility_criterion_view,
    derive_complete_eligibility_section,
    eligibility_criterion_trial,
    split_complete_eligibility,
)
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
    CAPABILITY_FIELD_PROVENANCE,
    CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
    BenchmarkSnapshot,
    BenchmarkTopic,
    FieldProvenance,
    SemanticTextSection,
    SourceRecordIdentity,
    Trial,
)


def _provenance(text: str) -> FieldProvenance:
    raw = text.encode("utf-8")
    excerpt = raw[:4096].decode("utf-8", errors="ignore")
    return FieldProvenance(
        artifact_id="trec-2021-fixture",
        artifact_sha256="sha256:" + "a" * 64,
        source_record_id="NCT00000001",
        locator_kind="xml_path",
        location="/clinical_study/eligibility[1]/criteria[1]/textblock[1]",
        raw_value=excerpt,
        transformation_rule="ctgov-xml-text-whitespace-v1",
        raw_value_sha256=("sha256:" + hashlib.sha256(raw).hexdigest() if len(raw) > 4096 else None),
        raw_value_byte_length=len(raw) if len(raw) > 4096 else None,
    )


def test_split_handles_headings_lists_single_newlines_and_wrapped_continuations() -> None:
    source = """Key Inclusion Criteria:
1. Adults with alpha syndrome
   confirmed by laboratory testing.
2) Stable medication.

Requirements for exclusion:
- Prior surgery
  within thirty days.
* Active infection."""

    criteria = split_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in criteria] == [
        ("inclusion", "Adults with alpha syndrome\nconfirmed by laboratory testing."),
        ("inclusion", "Stable medication."),
        ("exclusion", "Prior surgery\nwithin thirty days."),
        ("exclusion", "Active infection."),
    ]
    parsed = analyze_complete_eligibility(source)
    assert parsed.parse_tier == "explicit_list"
    assert parsed.review_reasons == ()
    assert all(item.list_marked for item in parsed.criteria)


def test_split_recognizes_qualified_compact_plural_and_declarative_headings() -> None:
    source = """Patient Inclusion Criteria:
Adult participant.
-INCLUSION CRITERIA
Stable medication.
Major Exclusion Criteria:
Prior surgery.
Exclusions:
Active infection.
The exclusion criteria are:
Pregnancy."""

    parsed = analyze_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in parsed.criteria] == [
        ("inclusion", "Adult participant."),
        ("inclusion", "Stable medication."),
        ("exclusion", "Prior surgery."),
        ("exclusion", "Active infection."),
        ("exclusion", "Pregnancy."),
    ]
    assert parsed.explicit_polarities == {"inclusion", "exclusion"}
    assert parsed.parse_tier == "explicit_prose"


def test_split_maps_non_inclusion_heading_to_exclusion() -> None:
    source = """Donor Inclusion Criteria:
- Able to consent.
Non-inclusion Criteria:
- Active infection."""

    parsed = analyze_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in parsed.criteria] == [
        ("inclusion", "Able to consent."),
        ("exclusion", "Active infection."),
    ]
    assert parsed.parse_tier == "explicit_list"
    assert parsed.review_reasons == ()


def test_derived_view_canonicalizes_unicode_heading_case_and_semicolon() -> None:
    source = "İnclusion criteria;\n- Adult.\nExclusion criteria;\n- Active infection."
    section = derive_complete_eligibility_section(
        SemanticTextSection("eligibility", "normalized", 0, (_provenance(source),)),
        source_text=source,
    )
    trial = Trial(
        "NCT03955237",
        SourceRecordIdentity("fixture", "NCT03955237"),
        "trial",
        (section,),
    )
    snapshot = BenchmarkSnapshot(
        benchmark_lineage="fixture",
        snapshot_name="fixture",
        topics=(BenchmarkTopic("1", SourceRecordIdentity("fixture", "1"), "patient"),),
        trials=(trial,),
        available_capabilities=frozenset(
            {
                CAPABILITY_CANONICAL_PATIENT_TEXT,
                CAPABILITY_CANONICAL_TRIAL_TEXT,
                CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
                CAPABILITY_FIELD_PROVENANCE,
                CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
            }
        ),
    )

    record = eligibility_criterion_trial(
        build_eligibility_criterion_view(snapshot),
        trial.trial_id,
    )

    assert record is not None
    assert [(item.polarity, item.text) for item in record.criteria] == [
        ("inclusion", "Adult."),
        ("exclusion", "Active infection."),
    ]
    assert record.parse_tier == "explicit_list"
    assert record.review_reasons == ()


def test_split_declarative_heading_keeps_inline_criterion_and_source_offset() -> None:
    source = "The inclusion criteria are age 18 years or older."

    (criterion,) = split_complete_eligibility(source)

    assert criterion.polarity == "inclusion"
    assert criterion.text == "age 18 years or older."
    assert source[criterion.source_start : criterion.source_end] == criterion.text


def test_heading_verbs_do_not_consume_criterion_word_prefixes() -> None:
    source = """Inclusion criteria includes adults.
Inclusion criteria island residents.
Exclusion criteria areas with malaria."""

    criteria = split_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in criteria] == [
        ("inclusion", "adults."),
        ("inclusion", "Inclusion criteria island residents."),
        ("inclusion", "Exclusion criteria areas with malaria."),
    ]

    absence = analyze_complete_eligibility(
        "Inclusion criteria:\n- Adult.\nExclusion criteria:\n"
        "- Absence of any inclusion criteria.\n- Active infection."
    )
    assert [item.text for item in absence.exclusion] == [
        "Absence of any inclusion criteria.",
        "Active infection.",
    ]


def test_split_discards_heading_scope_and_source_pointer_text() -> None:
    source = """The inclusion criteria for the study are:
- Adult participant.
Patient Exclusion Criteria (for chart review):
- Active infection.
Additional donor inclusion criteria can be found in the Donor Companion Manual"""

    parsed = analyze_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in parsed.criteria] == [
        ("inclusion", "Adult participant."),
        ("exclusion", "Active infection."),
    ]


def test_split_heading_scope_keeps_inline_criterion_and_source_offset() -> None:
    source = "The inclusion criteria for the study are: age 18 years or older."

    (criterion,) = split_complete_eligibility(source)

    assert criterion.text == "age 18 years or older."
    assert source[criterion.source_start : criterion.source_end] == criterion.text


def test_split_handles_double_newline_missing_heading_and_empty_section() -> None:
    missing_heading = split_complete_eligibility(
        "Adults with alpha syndrome.\ncontinued detail.\n\nStable medication."
    )
    assert [(item.polarity, item.text) for item in missing_heading] == [
        ("unspecified", "Adults with alpha syndrome.\ncontinued detail."),
        ("unspecified", "Stable medication."),
    ]

    empty_inclusion = split_complete_eligibility(
        "Inclusion criteria:\n\nExclusion criteria:\n- Active infection."
    )
    assert [(item.polarity, item.text) for item in empty_inclusion] == [
        ("exclusion", "Active infection."),
    ]


def test_split_distinguishes_single_newline_criteria_from_wrapped_continuations() -> None:
    criteria = split_complete_eligibility(
        "Inclusion criteria:\n"
        "Adults with alpha syndrome.\n"
        "Stable medication is required.\n"
        "Laboratory confirmation must be available and\n"
        "  documented within thirty days."
    )

    assert [item.text for item in criteria] == [
        "Adults with alpha syndrome.",
        "Stable medication is required.",
        "Laboratory confirmation must be available and\ndocumented within thirty days.",
    ]
    parsed = analyze_complete_eligibility(
        "Inclusion criteria:\nAdult.\nExclusion criteria:\nActive infection."
    )
    assert parsed.parse_tier == "explicit_prose"
    assert parsed.review_reasons == ("unmarked_criterion_boundaries",)


def test_parse_tiers_keep_unlabelled_text_separate_from_explicit_polarity() -> None:
    mixed = analyze_complete_eligibility(
        "Adults with alpha syndrome.\n\nExclusion criteria:\n- Active infection."
    )
    incomplete = analyze_complete_eligibility("Inclusion criteria:\n- Adults only.")
    empty = analyze_complete_eligibility("Inclusion criteria:\n\nExclusion criteria:")

    assert mixed.parse_tier == "incomplete_polarity"
    assert mixed.review_reasons == (
        "missing_inclusion_criteria",
        "unmarked_criterion_boundaries",
        "unspecified_polarity",
    )
    assert incomplete.parse_tier == "incomplete_polarity"
    assert incomplete.review_reasons == ("missing_exclusion_criteria",)
    assert empty.parse_tier == "empty"
    assert empty.review_reasons == (
        "missing_inclusion_criteria",
        "missing_exclusion_criteria",
    )


def test_split_recognizes_registry_heading_and_layout_variants() -> None:
    source = """[Inclusion Criteria]
1.Adult participant
Presurgical Exclusion Critera\uff1a
•Active infection
Postsurgical Exclusion Conditions — Unable to consent."""

    parsed = analyze_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in parsed.criteria] == [
        ("inclusion", "Adult participant"),
        ("exclusion", "Active infection"),
        ("exclusion", "Unable to consent."),
    ]
    assert parsed.inclusion_status == "criteria"
    assert parsed.exclusion_status == "criteria"
    assert all(
        source[item.source_start : item.source_end].strip() == item.text for item in parsed.criteria
    )


def test_split_handles_multiple_compact_headings_on_one_line() -> None:
    source = "Inclusion Criteria:Healthy Exclusion Criteria:prior gastrointestinal disease"

    parsed = analyze_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in parsed.criteria] == [
        ("inclusion", "Healthy"),
        ("exclusion", "prior gastrointestinal disease"),
    ]
    assert [source[item.source_start : item.source_end] for item in parsed.criteria] == [
        "Healthy",
        "prior gastrointestinal disease",
    ]


def test_split_handles_inline_and_subject_declaration_headings() -> None:
    source = """Participants must be adults. Exclusion criteria: Mentally incompetent patients.
Inclusion Criteria of the patients include those who:
- Are age 18 years or older.
Patients will be excluded if they have:
- Active infection."""

    parsed = analyze_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in parsed.criteria] == [
        ("unspecified", "Participants must be adults."),
        ("exclusion", "Mentally incompetent patients."),
        ("inclusion", "Are age 18 years or older."),
        ("exclusion", "Active infection."),
    ]

    enrolled = analyze_complete_eligibility(
        "Subjects will be enrolled if they meet the following inclusion criteria:\n"
        "- Adult.\nSubjects will be excluded if they have:\n- Active infection."
    )
    assert [(item.polarity, item.text) for item in enrolled.criteria] == [
        ("inclusion", "Adult."),
        ("exclusion", "Active infection."),
    ]


def test_split_recognizes_scoped_period_and_narrative_heading_forms() -> None:
    source = """INCLUSION CRITERIA FOR ALL SUBJECTS:
- Adult.
Specific inclusion criteria-patient:
- Stable medication.
Exclusion criteria.
- Pregnant.
Inclusion criteria required that subjects have laboratory confirmation.
Exclusion criteria included active infection."""

    parsed = analyze_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in parsed.criteria] == [
        ("inclusion", "Adult."),
        ("inclusion", "Stable medication."),
        ("exclusion", "Pregnant."),
        ("inclusion", "subjects have laboratory confirmation."),
        ("exclusion", "active infection."),
    ]


def test_scoped_headings_and_context_headers_preserve_current_polarity() -> None:
    source = """Inclusion Criteria specific for each observatory:
Disease Characteristics:
- Adult.
Exclusion criteria will include:
- Active infection."""

    parsed = analyze_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in parsed.criteria] == [
        ("inclusion", "Adult."),
        ("exclusion", "Active infection."),
    ]
    assert [(item.item_type, item.text) for item in parsed.other_items] == [
        ("context", "Disease Characteristics:"),
    ]


def test_possessive_headings_and_wrapped_heading_words_are_disambiguated() -> None:
    source = """Teachers' Inclusion Criteria:
- Adult teacher who must meet other
  eligibility requirements.
Teachers' Exclusion Criteria: Active infection.
Children's Exclusion Criteria: There are no exclusion criteria for children."""

    parsed = analyze_complete_eligibility(source)

    assert [(item.polarity, item.text) for item in parsed.criteria] == [
        ("inclusion", "Adult teacher who must meet other\neligibility requirements."),
        ("exclusion", "Active infection."),
    ]
    assert parsed.exclusion_status == "criteria"
    assert any(
        item.item_type == "declared_empty" and "children" in item.text
        for item in parsed.other_items
    )


def test_nct04700683_uppercase_wrapped_line_stays_in_criterion() -> None:
    source = """Inclusion Criteria:
- Subject reports having a medical diagnosis of primary idiopathic restless legs
  syndrome (RLS) or Willis-Ekbom Disorder (WED) OR Investigator has diagnosed the
  subject with primary idiopathic restless legs syndrome (RLS) or Willis-Ekbom Disorder
  (WED).
- Subject is 18 years of age or older."""

    parsed = analyze_complete_eligibility(source)

    assert [item.text for item in parsed.criteria] == [
        "Subject reports having a medical diagnosis of primary idiopathic restless legs\n"
        "syndrome (RLS) or Willis-Ekbom Disorder (WED) OR Investigator has diagnosed the\n"
        "subject with primary idiopathic restless legs syndrome (RLS) or Willis-Ekbom Disorder\n"
        "(WED).",
        "Subject is 18 years of age or older.",
    ]
    first = parsed.criteria[0]
    assert source[first.source_start : first.source_end] == (
        "Subject reports having a medical diagnosis of primary idiopathic restless legs\n"
        "  syndrome (RLS) or Willis-Ekbom Disorder (WED) OR Investigator has diagnosed the\n"
        "  subject with primary idiopathic restless legs syndrome (RLS) or Willis-Ekbom Disorder\n"
        "  (WED)."
    )
    assert parsed.other_items == ()


def test_indented_uppercase_group_heading_is_not_merged_into_previous_criterion() -> None:
    source = """Inclusion Criteria:
- Adult participant whose screening evaluation is complete.
    REQUIRED LABORATORY VALUES
    - BMI below the protocol threshold.
- Able to provide consent."""

    parsed = analyze_complete_eligibility(source)

    assert [item.text for item in parsed.criteria] == [
        "Adult participant whose screening evaluation is complete.",
        "BMI below the protocol threshold.",
        "Able to provide consent.",
    ]
    assert [(item.item_type, item.text) for item in parsed.other_items] == [
        ("group_header", "REQUIRED LABORATORY VALUES"),
    ]
    assert parsed.criteria[1].group_path == ("REQUIRED LABORATORY VALUES",)


def test_uppercase_measurements_remain_criteria() -> None:
    source = """Inclusion Criteria:
- BMI ≥ 27
- Subjects must be able to provide written informed consent
Exclusion Criteria:
- BMI > 42
- Pregnancy and breastfeeding
Inclusion Criteria:
1. BMI<35
2. Eligible for controlled ovarian simulation"""

    parsed = analyze_complete_eligibility(source)

    assert [item.text for item in parsed.criteria] == [
        "BMI ≥ 27",
        "Subjects must be able to provide written informed consent",
        "BMI > 42",
        "Pregnancy and breastfeeding",
        "BMI<35",
        "Eligible for controlled ovarian simulation",
    ]
    assert all(
        item.text not in {other.text for other in parsed.other_items} for item in parsed.criteria
    )


def test_list_structure_preserves_raw_markers_ordinals_and_indentation() -> None:
    source = """Inclusion Criteria:
  [1] Adult participant.
    a) Laboratory confirmation.
  * Stable medication."""

    parsed = analyze_complete_eligibility(source)

    assert [item.text for item in parsed.criteria] == [
        "Adult participant.",
        "Laboratory confirmation.",
        "Stable medication.",
    ]
    assert [item.list_marker for item in parsed.criteria] == ["[1]", "a)", "*"]
    assert [item.list_marker_type for item in parsed.criteria] == [
        "number",
        "letter",
        "bullet",
    ]
    assert [item.list_marker_ordinal for item in parsed.criteria] == [1, 1, None]
    assert [item.list_indent for item in parsed.criteria] == ["  ", "    ", "  "]
    assert [item.list_depth for item in parsed.criteria] == [1, 2, 1]


def test_list_depth_is_local_to_heading_and_ignores_numbering_restart() -> None:
    source = """Inclusion Criteria:
  1. Adult participant.
    1. Laboratory confirmation.
    1. Stable medication.
Exclusion Criteria:
      1. Active infection.
      2. Pregnancy."""

    criteria = analyze_complete_eligibility(source).criteria

    assert [item.list_depth for item in criteria] == [1, 2, 2, 1, 1]
    assert [item.parent_source_start for item in criteria] == [
        None,
        criteria[0].source_start,
        criteria[0].source_start,
        None,
        None,
    ]


def test_nct02033694_nested_lists_preserve_parents_relationships_and_dedents() -> None:
    source = """General Inclusion Criteria:
- Clinical presenting symptoms meeting one of the three criteria below:
    1. Subjects presenting with an acute coronary syndrome including at least one
       of the following:
        1. Elevated cardiac biomarkers;
        2. ST depression or ST elevation;
    2. Unstable angina pectoris;
- Greater than 18 years of age."""

    parsed = analyze_complete_eligibility(source)
    criteria = parsed.criteria

    assert [item.text for item in criteria] == [
        "Clinical presenting symptoms meeting one of the three criteria below:",
        "Subjects presenting with an acute coronary syndrome including at least one\n"
        "of the following:",
        "Elevated cardiac biomarkers;",
        "ST depression or ST elevation;",
        "Unstable angina pectoris;",
        "Greater than 18 years of age.",
    ]
    assert [item.list_depth for item in criteria] == [1, 2, 3, 3, 2, 1]
    assert [item.parent_source_start for item in criteria] == [
        None,
        criteria[0].source_start,
        criteria[1].source_start,
        criteria[1].source_start,
        criteria[0].source_start,
        None,
    ]
    assert [item.child_relationship for item in criteria] == [
        "at_least_n",
        "at_least_n",
        "unknown",
        "unknown",
        "unknown",
        "unknown",
    ]
    assert [item.required_child_count for item in criteria] == [1, 1, None, None, None, None]


def test_nct03306680_emits_parent_identifiers_and_unknown_relationship() -> None:
    source = """Inclusion Criteria:
2. Stage T1-3, N0, M0, tumor size < 6 cm, prior to
   registration, based upon the following minimum diagnostic workup:
    1. History/physical examination within 4 weeks prior to registration
    2. CT scan with contrast within 12 weeks of registration
3. ECOG performance status 0-2;"""
    section = derive_complete_eligibility_section(
        SemanticTextSection("eligibility", "normalized", 0, (_provenance(source),)),
        source_text=source,
    )

    record = build_eligibility_criterion_trial_record("NCT03306680", (section,))

    assert record is not None
    criteria = record.criteria
    assert [item.text for item in criteria] == [
        "Stage T1-3, N0, M0, tumor size < 6 cm, prior to\n"
        "registration, based upon the following minimum diagnostic workup:",
        "History/physical examination within 4 weeks prior to registration",
        "CT scan with contrast within 12 weeks of registration",
        "ECOG performance status 0-2;",
    ]
    assert criteria[0].child_relationship == "unknown"
    assert criteria[0].required_child_count is None
    assert [item.parent_identifier for item in criteria] == [
        None,
        criteria[0].identifier,
        criteria[0].identifier,
        None,
    ]
    assert [item.parent_type for item in criteria] == [
        None,
        "criterion",
        "criterion",
        None,
    ]
    assert all(
        item.source_span_text == source[item.source_start : item.source_end] for item in criteria
    )


def test_explicit_all_of_and_any_of_relationships_are_preserved() -> None:
    source = """Inclusion Criteria:
- Participants must meet all of the following:
    1. Adult participant.
    2. Able to consent.
- Participants may meet any of the following:
    a) Stable medication.
    b) No current medication."""

    criteria = analyze_complete_eligibility(source).criteria

    assert [item.child_relationship for item in criteria] == [
        "all_of",
        "unknown",
        "unknown",
        "any_of",
        "unknown",
        "unknown",
    ]
    assert all(item.required_child_count is None for item in criteria)
    assert [item.parent_source_start for item in criteria] == [
        None,
        criteria[0].source_start,
        criteria[0].source_start,
        None,
        criteria[3].source_start,
        criteria[3].source_start,
    ]


def test_group_headers_are_preserved_but_not_emitted_as_criteria() -> None:
    source = """Inclusion criteria:
Concurrent Medication:
- Allowed therapy.
Recommended:
- Vaccination.
Exclusion criteria:
- Active infection."""

    parsed = analyze_complete_eligibility(source)

    assert [item.text for item in parsed.criteria] == [
        "Allowed therapy.",
        "Vaccination.",
        "Active infection.",
    ]
    assert [(item.item_type, item.text) for item in parsed.other_items] == [
        ("group_header", "Concurrent Medication:"),
        ("group_header", "Recommended:"),
    ]
    assert parsed.criteria[0].group_path == ("Concurrent Medication:",)
    assert parsed.criteria[1].group_path == ("Recommended:",)


def test_boolean_separators_and_terminal_labels_are_noncriteria() -> None:
    parsed = analyze_complete_eligibility(
        "Inclusion criteria:\n- Adult.\n- Or\nControl Cohort:\n"
        "Exclusion criteria:\n- Active infection."
    )

    assert [item.text for item in parsed.criteria] == ["Adult.", "Active infection."]
    assert [(item.item_type, item.text) for item in parsed.other_items] == [
        ("context", "Or"),
        ("context", "Control Cohort:"),
    ]


def test_declared_empty_and_unavailable_are_statuses_not_criteria() -> None:
    parsed = analyze_complete_eligibility(
        "Inclusion criteria:\nNone\nExclusion criteria:\nContact site for details"
    )

    assert parsed.criteria == ()
    assert parsed.inclusion_status == "declared_empty"
    assert parsed.exclusion_status == "unavailable"
    assert [(item.item_type, item.polarity) for item in parsed.other_items] == [
        ("declared_empty", "inclusion"),
        ("unavailable", "exclusion"),
    ]

    misspelled_empty = analyze_complete_eligibility(
        "Inclusion criteria:\n- Adult\nExclusion criteria:\n- non exclusion criteria"
    )
    assert misspelled_empty.exclusion_status == "declared_empty"
    assert [item.text for item in misspelled_empty.criteria] == ["Adult"]

    unavailable = analyze_complete_eligibility(
        "Inclusion criteria:\n"
        "- Other criteria may apply; please contact the investigator for more information."
    )
    assert unavailable.inclusion_status == "unavailable"
    assert unavailable.criteria == ()

    direct_contact = analyze_complete_eligibility(
        "Exclusion criteria:\n- Please contact the site directly for additional information."
    )
    assert direct_contact.exclusion_status == "unavailable"
    assert direct_contact.criteria == ()

    both_unavailable = analyze_complete_eligibility(
        "Note: Other protocol-defined Inclusion/ Exclusion criteria may apply."
    )
    assert both_unavailable.inclusion_status == "unavailable"
    assert both_unavailable.exclusion_status == "unavailable"


def test_unlabelled_lists_remain_unspecified_instead_of_guessing_inclusion() -> None:
    parsed = analyze_complete_eligibility("- Adult\n- Not pregnant")

    assert [(item.polarity, item.text) for item in parsed.criteria] == [
        ("unspecified", "Adult"),
        ("unspecified", "Not pregnant"),
    ]
    assert parsed.inclusion_status == "not_identified"
    assert parsed.exclusion_status == "not_identified"


def test_common_indentation_and_line_per_criterion_layout_is_deterministic() -> None:
    source = """    Inclusion criteria:
    Adult participant.
    Stable medication.
    Laboratory confirmation and
      documentation within thirty days.
    Exclusion criteria:
    Active infection."""

    parsed = analyze_complete_eligibility(source)

    assert [item.text for item in parsed.criteria] == [
        "Adult participant.",
        "Stable medication.",
        "Laboratory confirmation and\ndocumentation within thirty days.",
        "Active infection.",
    ]


def test_heading_and_marker_formatting_variants_preserve_logical_criteria() -> None:
    variants = (
        "Inclusion criteria:\n- Adult.\nExclusion criteria:\n- Pregnant.",
        "<INCLUSION CRITERIA>\n•Adult.\n<EXCLUSION CRITERIA>\n•Pregnant.",
        "  [Inclusion Criteria]\n  1.Adult.\n  [Exclusion Criteria]\n  2)Pregnant.",
    )

    logical = [
        [(item.polarity, item.text) for item in split_complete_eligibility(source)]
        for source in variants
    ]

    assert logical == [logical[0]] * len(logical)


def test_inline_heading_tail_uses_the_matched_tail_source_offset() -> None:
    source = "Inclusion Criteria: Inclusion"

    (criterion,) = split_complete_eligibility(source)

    assert criterion.text == "Inclusion"
    assert criterion.source_start == source.rindex(criterion.text)
    assert source[criterion.source_start : criterion.source_end] == criterion.text


def test_complete_source_section_is_source_grounded_and_not_truncated() -> None:
    source = "Inclusion criteria:\n- " + ("eligible " * 700) + "\nExclusion criteria:\n- none"
    section = SemanticTextSection(
        role="eligibility",
        text="normalized eligibility",
        ordinal=0,
        provenance=(_provenance(source),),
    )

    first = derive_complete_eligibility_section(section, source_text=source)
    second = derive_complete_eligibility_section(section, source_text=source)

    assert len(source.encode("utf-8")) > 4096
    assert first.source_text == source
    assert first.source_text_byte_length == len(source.encode("utf-8"))
    assert first.source_text_sha256 == "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    assert first.to_dict() == second.to_dict()
    assert first.criteria == ()
    assert first.additional_fields == {}


def test_eligibility_criteria_are_a_snapshot_derived_view() -> None:
    source = "Inclusion criteria:\n- Adult.\nExclusion criteria:\n- Active infection."
    section = derive_complete_eligibility_section(
        SemanticTextSection("eligibility", "normalized", 0, (_provenance(source),)),
        source_text=source,
    )
    trial = Trial(
        "NCT00000001",
        SourceRecordIdentity("fixture", "NCT00000001"),
        "trial",
        (section,),
    )
    snapshot = BenchmarkSnapshot(
        benchmark_lineage="fixture",
        snapshot_name="fixture",
        topics=(
            BenchmarkTopic(
                "1",
                SourceRecordIdentity("fixture", "1"),
                "patient",
            ),
        ),
        trials=(trial,),
        available_capabilities=frozenset(
            {
                CAPABILITY_CANONICAL_PATIENT_TEXT,
                CAPABILITY_CANONICAL_TRIAL_TEXT,
                CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
                CAPABILITY_FIELD_PROVENANCE,
                CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
            }
        ),
    )

    view = build_eligibility_criterion_view(snapshot)
    supplied = replace(
        snapshot,
        available_capabilities=snapshot.available_capabilities | {view.capability},
        derived_views=(view,),
    )
    record = eligibility_criterion_trial(view, "NCT00000001")

    assert view.name == ELIGIBILITY_CRITERION_VIEW_NAME
    assert view.input_snapshot_id == snapshot.base_snapshot_id
    assert supplied.base_snapshot_id == snapshot.base_snapshot_id
    assert supplied.snapshot_id != snapshot.snapshot_id
    assert record is not None
    assert record.parse_tier == "explicit_list"
    assert record.review_reasons == ()
    assert [item.text for item in record.inclusion] == ["Adult."]
    assert [item.text for item in record.exclusion] == ["Active infection."]
    assert [item.list_marker for item in record.criteria] == ["-", "-"]
    assert [item.list_marker_type for item in record.criteria] == ["bullet", "bullet"]
    assert [item.list_marker_ordinal for item in record.criteria] == [None, None]
    assert [item.list_indent for item in record.criteria] == ["", ""]
    assert record.source_text_sha256 == section.source_text_sha256
    assert record.criterion_inventory_sha256.startswith("sha256:")
    assert section.additional_fields == {}
    assert section.criteria == ()


def test_complete_eligibility_exposes_decoded_text_and_preserves_lexical_provenance() -> None:
    serialized = b"""<?xml version="1.0" encoding="UTF-8"?>
<clinical_study>
  <id_info><nct_id>NCT00000001</nct_id></id_info>
  <eligibility><criteria><textblock>Inclusion Criteria:
  - Age &gt;= 18 &amp; able to consent

Exclusion Criteria:
  - Value &lt; 2</textblock></criteria></eligibility>
</clinical_study>"""

    trial_id, source_text, provenance_text = extract_complete_eligibility_text(serialized)

    assert trial_id == "NCT00000001"
    assert source_text == (
        "Inclusion Criteria:\n  - Age >= 18 & able to consent\n\nExclusion Criteria:\n  - Value < 2"
    )
    assert provenance_text == (
        "Inclusion Criteria:\n"
        "  - Age &gt;= 18 &amp; able to consent\n\n"
        "Exclusion Criteria:\n"
        "  - Value &lt; 2"
    )


def test_decoded_complete_eligibility_drives_derived_criteria_without_collapsing_layout() -> None:
    serialized = b"""<clinical_study>
  <id_info><nct_id>NCT00000001</nct_id></id_info>
  <eligibility><criteria><textblock>Inclusion Criteria:
- Measurement &gt;= 10
  confirmed twice.

Exclusion Criteria:
- Measurement &lt; 2</textblock></criteria></eligibility>
</clinical_study>"""
    _trial_id, source_text, provenance_text = extract_complete_eligibility_text(serialized)
    assert source_text is not None
    assert provenance_text is not None
    section = SemanticTextSection(
        role="eligibility",
        text="normalized eligibility",
        ordinal=0,
        provenance=(_provenance(provenance_text),),
    )

    derived = derive_complete_eligibility_section(section, source_text=source_text)
    trial = Trial(
        "NCT00000001",
        SourceRecordIdentity("fixture", "NCT00000001"),
        "trial",
        (derived,),
    )
    snapshot = BenchmarkSnapshot(
        benchmark_lineage="fixture",
        snapshot_name="fixture",
        topics=(BenchmarkTopic("1", SourceRecordIdentity("fixture", "1"), "patient"),),
        trials=(trial,),
        available_capabilities=frozenset(
            {
                CAPABILITY_CANONICAL_PATIENT_TEXT,
                CAPABILITY_CANONICAL_TRIAL_TEXT,
                CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
                CAPABILITY_FIELD_PROVENANCE,
                CAPABILITY_SEMANTIC_TRIAL_SECTIONS,
            }
        ),
    )
    record = eligibility_criterion_trial(
        build_eligibility_criterion_view(snapshot),
        trial.trial_id,
    )

    assert derived.source_text == source_text
    assert derived.criteria == ()
    assert record is not None
    assert [(item.polarity, item.text) for item in record.criteria] == [
        ("inclusion", "Measurement >= 10\nconfirmed twice."),
        ("exclusion", "Measurement < 2"),
    ]
    assert [item.source_span_text for item in record.criteria] == [
        "Measurement >= 10\n  confirmed twice.",
        "Measurement < 2",
    ]
    assert all(
        item.source_span_text == source_text[item.source_start : item.source_end]
        for item in record.criteria
    )
