# ruff: noqa: S101
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from taim.data import load_prepared_benchmark
from taim.data.trec_ct_2021 import (
    FIELD_DISPOSITION_TABLE,
    SOURCE_RECIPE,
    IngestionError,
    SourceVerificationError,
    load_source_bundle_lock,
    map_ctgov_json_eligibility,
    map_ctgov_json_trial_layers,
    map_source_authored_eligibility_section,
    prepare_trec_ct_2021,
    validate_trec_ct_2021_recipe_parity,
)
from taim.schemas import SchemaValidationError, json_value_to_builtins
from taim.snapshot import AgeQuantity
from taim.source import SourceArtifact
from tests.snapshot_helpers import (
    TREC_FIXED_TIME,
    complete_trial_xml,
    write_trec_source_bundle,
)

JSON_ELIGIBILITY = (
    Path(__file__).parent / "fixtures" / "snapshot" / "ctgov-eligibility-equivalent.json"
)
SOURCE_AUTHORED_CRITERIA = (
    Path(__file__).parent / "fixtures" / "snapshot" / "source-authored-criteria.json"
)
SOURCE_ELIGIBILITY_BLOB = (
    Path(__file__).parent / "fixtures" / "snapshot" / "source-eligibility-blob.json"
)


def _prepare_one(tmp_path, serialized: str):
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": serialized},
    )
    return load_prepared_benchmark(
        prepare_trec_ct_2021(
            source,
            tmp_path / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        ).output_directory
    )


def test_xml_and_json_source_mappings_normalize_equivalent_core_values(tmp_path) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={
            "NCT00000001": complete_trial_xml().replace(
                "<maximum_age>65 Years</maximum_age>",
                "<maximum_age>N/A</maximum_age>",
            )
        },
    )
    prepared = load_prepared_benchmark(
        prepare_trec_ct_2021(
            source,
            tmp_path / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        ).output_directory
    )
    xml_core = prepared.trials[0].typed_clinical_core
    assert xml_core is not None

    json_artifact = SourceArtifact(
        artifact_id="ctgov-json",
        role="trials",
        filename="synthetic.json",
        byte_size=1,
        sha256="sha256:" + "a" * 64,
        acquisition_date="2026-07-31",
        access_terms="synthetic",
        redistribution_terms="synthetic",
        url="https://example.test/synthetic.json",
    )
    json_core, diagnostics = map_ctgov_json_eligibility(
        json.loads(JSON_ELIGIBILITY.read_text(encoding="utf-8")),
        artifact=json_artifact,
    )
    json_sections, layered_core, layered_diagnostics = map_ctgov_json_trial_layers(
        json.loads(JSON_ELIGIBILITY.read_text(encoding="utf-8")),
        artifact=json_artifact,
    )

    assert diagnostics == ()
    assert layered_diagnostics == ()
    assert layered_core == json_core
    assert json_core is not None
    assert json_core.minimum_age is not None and xml_core.minimum_age is not None
    assert json_core.minimum_age.value == xml_core.minimum_age.value
    assert json_core.maximum_age is not None and xml_core.maximum_age is not None
    assert json_core.maximum_age.value == xml_core.maximum_age.value == "unbounded"
    assert json_core.sex is not None and xml_core.sex is not None
    assert json_core.sex.value == xml_core.sex.value
    assert json_core.healthy_volunteers is not None
    assert xml_core.healthy_volunteers is not None
    assert json_core.healthy_volunteers.value == xml_core.healthy_volunteers.value
    assert json_core.minimum_age.provenance[0].location == (
        "/protocolSection/eligibilityModule/minimumAge"
    )
    assert json_core.minimum_age.provenance[0].transformation_rule == "ctgov-json-age-v1"
    assert xml_core.minimum_age.provenance[0].transformation_rule == "ctgov-xml-age-v1"
    assert [(section.role, section.text) for section in json_sections] == [
        (
            "eligibility",
            "Adults with alpha syndrome. Excludes cardiac disease.",
        )
    ]
    assert json_sections[0].criteria == ()
    assert json_sections[0].provenance[0].location == (
        "/protocolSection/eligibilityModule/eligibilityCriteria"
    )
    assert json_sections[0].provenance[0].raw_value.startswith("<p><strong>Adults")
    assert json_sections[0].provenance[0].transformation_rule == ("ctgov-json-markup-text-v1")


def test_source_authored_criterion_structure_is_mapped_without_heuristic_splitting() -> None:
    artifact = SourceArtifact(
        artifact_id="source-authored-criteria",
        role="synthetic-criteria",
        filename="criteria.json",
        byte_size=1,
        sha256="sha256:" + "a" * 64,
        acquisition_date="2026-07-31",
        access_terms="synthetic",
        redistribution_terms="synthetic",
        url="https://example.test/criteria.json",
    )
    structured = map_source_authored_eligibility_section(
        json.loads(SOURCE_AUTHORED_CRITERIA.read_text(encoding="utf-8")),
        artifact=artifact,
        trial_id="NCT00000001",
    )
    blob = map_source_authored_eligibility_section(
        json.loads(SOURCE_ELIGIBILITY_BLOB.read_text(encoding="utf-8")),
        artifact=artifact,
        trial_id="NCT00000001",
    )

    assert [
        (item.identifier, item.polarity, item.ordinal, item.text) for item in structured.criteria
    ] == [
        ("INC-01", "inclusion", 0, "Confirmed alpha syndrome"),
        ("EXC-02", "exclusion", 1, "Active cardiac disease"),
    ]
    assert structured.provenance[0].location == "/text"
    assert [item.provenance[0].location for item in structured.criteria] == [
        "/criteria/0",
        "/criteria/1",
    ]
    assert blob.criteria == ()
    assert blob.text == "Inclusion: adult. Exclusion: renal failure."


# Public preparation owns the frozen recipe and covers successful parity. Detached
# definitions isolate drift failures without mutating the connector's module state.
def test_frozen_source_recipe_rejects_a_mapping_that_the_connector_does_not_execute() -> None:
    recipe = json_value_to_builtins(SOURCE_RECIPE)
    brief_title_mapping = next(
        item
        for item in recipe["field_disposition_table"]
        if item["source"] == "clinical_study/brief_title"
    )
    assert isinstance(brief_title_mapping, dict)
    brief_title_mapping["target"] = "summary"

    with pytest.raises(SchemaValidationError, match="out of parity"):
        validate_trec_ct_2021_recipe_parity(recipe)


def test_frozen_source_recipe_rejects_an_unexecuted_normalization_rule() -> None:
    recipe = json_value_to_builtins(SOURCE_RECIPE)
    minimum_age_mapping = next(
        item
        for item in recipe["field_disposition_table"]
        if item["source"] == "clinical_study/eligibility/minimum_age"
    )
    assert isinstance(minimum_age_mapping, dict)
    minimum_age_mapping["normalization_rule"] = "made-up-normalizer-v1"

    with pytest.raises(SchemaValidationError, match="out of parity"):
        validate_trec_ct_2021_recipe_parity(recipe)


@pytest.mark.parametrize(
    ("recipe_name", "field", "value"),
    [
        ("canonical_trial_text_recipe", "separator", "NUL"),
        ("canonical_patient_text_recipe", "whitespace", "preserve source whitespace"),
    ],
)
def test_frozen_source_recipe_rejects_unexecuted_rendering_semantics(
    recipe_name: str,
    field: str,
    value: str,
) -> None:
    detached_recipe = json_value_to_builtins(SOURCE_RECIPE)
    rendering = detached_recipe[recipe_name]
    assert isinstance(rendering, dict)
    rendering[field] = value

    with pytest.raises(SchemaValidationError, match="out of parity"):
        validate_trec_ct_2021_recipe_parity(detached_recipe)


def test_invalid_structured_values_stay_in_diagnostics_not_core(tmp_path) -> None:
    prepared = _prepare_one(
        tmp_path,
        """
        <clinical_study>
          <id_info><nct_id>NCT00000001</nct_id></id_info>
          <brief_title>Malformed typed fields</brief_title>
          <eligibility>
            <minimum_age>adult</minimum_age>
            <maximum_age>999</maximum_age>
            <gender>Unknown</gender>
            <healthy_volunteers>Maybe</healthy_volunteers>
          </eligibility>
        </clinical_study>
        """,
    )

    assert prepared.trials[0].typed_clinical_core is None
    assert [diagnostic.code for diagnostic in prepared.diagnostics] == [
        "invalid-minimum-age",
        "invalid-maximum-age",
        "unrecognized-sex",
        "invalid-healthy-volunteers",
    ]
    assert {diagnostic.severity for diagnostic in prepared.diagnostics} == {"field_nonfatal"}
    raw_values = {
        provenance.raw_value
        for diagnostic in prepared.diagnostics
        for provenance in diagnostic.provenance
    }
    assert raw_values == {"999", "Maybe", "Unknown", "adult"}
    assert all(
        prepared.resolve_provenance(provenance).role == "trials_part_1"
        for diagnostic in prepared.diagnostics
        for provenance in diagnostic.provenance
    )


def test_equivalent_assertions_coalesce_and_conflicts_remain_unresolved(tmp_path) -> None:
    prepared = _prepare_one(
        tmp_path,
        """
        <clinical_study>
          <id_info><nct_id>NCT00000001</nct_id></id_info>
          <brief_title>Repeated typed fields</brief_title>
          <eligibility>
            <minimum_age>18 Years</minimum_age>
            <minimum_age>18 Years</minimum_age>
            <maximum_age>65 Years</maximum_age>
            <maximum_age>70 Years</maximum_age>
            <gender>All</gender>
            <gender>ALL</gender>
            <healthy_volunteers>Yes</healthy_volunteers>
            <healthy_volunteers>No</healthy_volunteers>
          </eligibility>
        </clinical_study>
        """,
    )

    core = prepared.trials[0].typed_clinical_core
    assert core is not None
    assert core.minimum_age is not None
    assert len(core.minimum_age.provenance) == 2
    assert core.sex is not None and core.sex.value == "all"
    assert len(core.sex.provenance) == 2
    assert core.maximum_age is None
    assert core.healthy_volunteers is None
    assert [(conflict.field_name, list(conflict.values)) for conflict in core.conflicts] == [
        ("healthy_volunteers", [True, False]),
        (
            "maximum_age",
            [
                {"amount": "65", "unit": "years"},
                {"amount": "70", "unit": "years"},
            ],
        ),
    ]
    assert all(len(conflict.provenance) == 2 for conflict in core.conflicts)
    assert [diagnostic.code for diagnostic in prepared.diagnostics] == [
        "conflicting-maximum-age",
        "conflicting-healthy-volunteers",
    ]
    assert all(len(diagnostic.provenance) == 2 for diagnostic in prepared.diagnostics)
    source_recipe = json_value_to_builtins(
        prepared.prepared_manifest["snapshot_source_recipe"]["definition"]
    )
    assert prepared.preparation.source_recipe_id == "trec-ct-2021-snapshot-recipe-v3"
    assert source_recipe["connector"]["version"] == "2.2"
    assert "System-visible Core Field Conflict" in source_recipe["diagnostic_policy"]["conflict"]


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1 Minute", AgeQuantity("1", "minutes")),
        ("2 Hours", AgeQuantity("2", "hours")),
        ("3 Days", AgeQuantity("3", "days")),
        ("4 Weeks", AgeQuantity("4", "weeks")),
        ("5 Months", AgeQuantity("5", "months")),
        ("18.0 Years", AgeQuantity("18", "years")),
    ],
)
def test_age_normalization_preserves_exact_calendar_units(
    tmp_path,
    raw_value: str,
    expected: AgeQuantity,
) -> None:
    prepared = _prepare_one(
        tmp_path,
        f"""
        <clinical_study>
          <id_info><nct_id>NCT00000001</nct_id></id_info>
          <brief_title>Age unit trial</brief_title>
          <eligibility><minimum_age>{raw_value}</minimum_age></eligibility>
        </clinical_study>
        """,
    )

    core = prepared.trials[0].typed_clinical_core
    assert core is not None and core.minimum_age is not None
    assert core.minimum_age.value == expected


def test_not_applicable_maximum_age_is_explicitly_unbounded(tmp_path) -> None:
    prepared = _prepare_one(
        tmp_path,
        """
        <clinical_study>
          <id_info><nct_id>NCT00000001</nct_id></id_info>
          <brief_title>Bounded semantics</brief_title>
          <eligibility>
            <maximum_age>N/A</maximum_age>
            <gender>Female</gender>
            <healthy_volunteers>No</healthy_volunteers>
          </eligibility>
        </clinical_study>
        """,
    )

    core = prepared.trials[0].typed_clinical_core
    assert core is not None
    assert core.minimum_age is None
    assert core.maximum_age is not None
    assert core.maximum_age.value == "unbounded"
    assert core.sex is not None and core.sex.value == "female"
    assert core.healthy_volunteers is not None
    assert core.healthy_volunteers.value is False
    assert "[MAXIMUM_AGE] unbounded" in prepared.trials[0].canonical_text
    assert "N/A" not in prepared.trials[0].canonical_text
    assert prepared.diagnostics == ()


def test_historical_healthy_volunteer_lexical_maps_to_true(tmp_path) -> None:
    prepared = _prepare_one(
        tmp_path,
        """
        <clinical_study>
          <id_info><nct_id>NCT00000001</nct_id></id_info>
          <brief_title>Healthy volunteer study</brief_title>
          <eligibility>
            <healthy_volunteers>Accepts Healthy Volunteers</healthy_volunteers>
          </eligibility>
        </clinical_study>
        """,
    )

    core = prepared.trials[0].typed_clinical_core
    assert core is not None and core.healthy_volunteers is not None
    assert core.healthy_volunteers.value is True
    assert core.healthy_volunteers.provenance[0].transformation_rule == (
        "ctgov-xml-healthy-volunteers-v2"
    )
    assert prepared.diagnostics == ()


def test_normalization_preserves_the_raw_xml_lexical_value_in_provenance(tmp_path) -> None:
    prepared = _prepare_one(
        tmp_path,
        """
        <clinical_study>
          <id_info><nct_id>NCT00000001</nct_id></id_info>
          <brief_title>Whitespace provenance</brief_title>
          <eligibility><minimum_age>  18   Years  </minimum_age></eligibility>
        </clinical_study>
        """,
    )

    minimum_age = prepared.trials[0].typed_clinical_core.minimum_age

    assert minimum_age is not None
    assert minimum_age.value == AgeQuantity("18", "years")
    assert minimum_age.provenance[0].raw_value == "  18   Years  "


def test_large_raw_values_use_a_bounded_excerpt_with_digest_and_length(tmp_path) -> None:
    raw_eligibility = "ä" * 3_000
    prepared = _prepare_one(
        tmp_path,
        (
            "<clinical_study>"
            "<id_info><nct_id>NCT00000001</nct_id></id_info>"
            "<brief_title>Long source value</brief_title>"
            f"<eligibility><criteria><textblock>{raw_eligibility}</textblock></criteria></eligibility>"
            "</clinical_study>"
        ),
    )

    provenance = prepared.trials[0].sections[-1].provenance[0]

    assert len(provenance.raw_value.encode("utf-8")) <= 4_096
    assert provenance.raw_value_byte_length == len(raw_eligibility.encode("utf-8"))
    assert provenance.raw_value_sha256 == (
        "sha256:" + hashlib.sha256(raw_eligibility.encode("utf-8")).hexdigest()
    )


def test_canonical_text_excludes_source_markup_and_serialization_noise(tmp_path) -> None:
    prepared = _prepare_one(
        tmp_path,
        """
        <clinical_study>
          <id_info><nct_id>NCT00000001</nct_id></id_info>
          <brief_title><em>Alpha</em> &amp; beta</brief_title>
          <brief_summary><textblock>Line one
          line two</textblock></brief_summary>
          <eligibility><criteria>
            <textblock><strong>Adults</strong> only</textblock>
          </criteria></eligibility>
        </clinical_study>
        """,
    )

    assert prepared.trials[0].canonical_text == (
        "[BRIEF_TITLE] Alpha & beta\n[SUMMARY] Line one line two\n[ELIGIBILITY] Adults only"
    )
    assert "<em>" not in prepared.trials[0].canonical_text
    assert "clinical_study" not in prepared.trials[0].canonical_text


def test_malformed_trial_is_a_stable_record_fatal_diagnostic(tmp_path) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={
            "NCT00000001": complete_trial_xml(),
            "NCT00000002": "<clinical_study><broken>",
        },
    )
    prepared = load_prepared_benchmark(
        prepare_trec_ct_2021(
            source,
            tmp_path / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        ).output_directory
    )

    assert [trial.trial_id for trial in prepared.trials] == ["NCT00000001"]
    assert len(prepared.diagnostics) == 1
    diagnostic = prepared.diagnostics[0]
    assert diagnostic.code == "malformed-trial-record"
    assert diagnostic.severity == "record_fatal"
    assert diagnostic.record_id == "NCT00000002"
    assert diagnostic.provenance[0].location == "zip-member:records/NCT00000002.xml"


def test_source_bundle_is_fully_verified_before_any_parser_runs(
    tmp_path,
) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": complete_trial_xml()},
    )
    (source / "topics2021.xml").write_text("<malformed", encoding="utf-8")

    with pytest.raises(SourceVerificationError, match="verification failed"):
        prepare_trec_ct_2021(
            source,
            tmp_path / "prepared",
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        )
    assert not (tmp_path / "prepared").exists()


def test_source_recipe_contains_reviewable_mapping_and_disposition_policy() -> None:
    dispositions = {entry["disposition"] for entry in FIELD_DISPOSITION_TABLE}

    assert {
        "shared_core",
        "semantic_text",
        "semantic_text_blob",
        "intentionally_ignored",
    } <= dispositions
    assert all(
        "normalization_rule" in entry
        for entry in FIELD_DISPOSITION_TABLE
        if entry["disposition"] != "intentionally_ignored"
    )
    assert json_value_to_builtins(SOURCE_RECIPE["field_disposition_table"]) == list(
        FIELD_DISPOSITION_TABLE
    )
    assert SOURCE_RECIPE["canonical_patient_text_recipe"]
    assert SOURCE_RECIPE["parsers"]
    assert SOURCE_RECIPE["external_enrichment"] == "forbidden"
    assert "qrels" not in SOURCE_RECIPE["system_input_artifact_ids"]
    assert "qrels" in SOURCE_RECIPE["allowed_artifact_ids"]
    assert any(
        entry.get("source") == "clinical_study/detailed_description/textblock"
        and entry.get("target") == "summary"
        for entry in FIELD_DISPOSITION_TABLE
    )


def test_xml_provenance_retains_exact_inner_lexical_content(tmp_path) -> None:
    prepared = _prepare_one(
        tmp_path,
        """
        <clinical_study>
          <id_info><nct_id>NCT00000001</nct_id></id_info>
          <brief_title><em>Alpha</em> &amp; beta</brief_title>
        </clinical_study>
        """,
    )

    section = prepared.trials[0].sections[0]
    assert section.text == "Alpha & beta"
    assert section.provenance[0].raw_value == "<em>Alpha</em> &amp; beta"


def test_source_bundle_lock_rejects_an_unsupported_schema_version(tmp_path) -> None:
    _, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": complete_trial_xml()},
    )
    payload = json.loads(source_lock.read_text(encoding="utf-8"))
    payload["schema_version"] = "1.0"
    payload["benchmark_name"] = "trec-ct-2021"
    payload["canonical_outputs"] = {}
    payload["parser"] = {}
    source_lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IngestionError, match=r"unsupported historical.*schema_version.*1\.0"):
        load_source_bundle_lock(source_lock)


@pytest.mark.parametrize(
    ("location", "field"),
    [("top", "canonical_outputs"), ("source", "parser")],
)
def test_source_bundle_lock_rejects_ignored_legacy_metadata(
    tmp_path, location: str, field: str
) -> None:
    _, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": complete_trial_xml()},
    )
    payload = json.loads(source_lock.read_text(encoding="utf-8"))
    target = payload if location == "top" else payload["sources"][0]
    target[field] = {}
    source_lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IngestionError, match="fields mismatch"):
        load_source_bundle_lock(source_lock)


def test_packaged_source_bundle_lock_has_only_current_raw_source_metadata() -> None:
    from taim.data.trec_ct_2021 import DEFAULT_LOCK_PATH

    payload = json.loads(DEFAULT_LOCK_PATH.read_text(encoding="utf-8"))

    assert set(payload) == {"schema_version", "dataset_id", "lock_status", "sources"}
    assert payload["schema_version"] == "2.0"
    assert all(
        set(source)
        == {
            "role",
            "filename",
            "url",
            "byte_size",
            "sha256",
            "acquisition_date",
            "access_terms",
            "redistribution_terms",
        }
        for source in payload["sources"]
    )


def test_preparation_is_byte_deterministic_and_never_overwrites_existing_output(
    tmp_path,
) -> None:
    source, source_lock = write_trec_source_bundle(
        tmp_path,
        trials={"NCT00000001": complete_trial_xml()},
    )
    first = prepare_trec_ct_2021(
        source,
        tmp_path / "first",
        source_lock_path=source_lock,
        created_at=TREC_FIXED_TIME,
    )
    second = prepare_trec_ct_2021(
        source,
        tmp_path / "second",
        source_lock_path=source_lock,
        created_at=TREC_FIXED_TIME,
    )
    first_files = {
        path.name: path.read_bytes() for path in first.output_directory.iterdir() if path.is_file()
    }
    second_files = {
        path.name: path.read_bytes() for path in second.output_directory.iterdir() if path.is_file()
    }

    assert second_files == first_files
    with pytest.raises(SchemaValidationError, match="already exists"):
        prepare_trec_ct_2021(
            source,
            first.output_directory,
            source_lock_path=source_lock,
            created_at=TREC_FIXED_TIME,
        )
    assert {
        path.name: path.read_bytes() for path in first.output_directory.iterdir() if path.is_file()
    } == first_files
