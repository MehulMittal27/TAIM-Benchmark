# ruff: noqa: S101
from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import pytest

from taim.contracts import canonical_json as support_canonical_json
from taim.contracts import content_sha256, normalize_sha256
from taim.data.trec_ct_2021 import render_canonical_trial_text
from taim.schemas import JsonValue, SchemaValidationError, json_value_to_builtins
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    AgeBound,
    AgeQuantity,
    BenchmarkSnapshot,
    BenchmarkTopic,
    CriterionItem,
    DerivedView,
    FieldProvenance,
    PatientCoreAssertion,
    PatientEvidenceItem,
    SemanticTextSection,
    SexEligibility,
    SourceRecordIdentity,
    Trial,
    TypedClinicalCore,
    TypedPatientCore,
    canonical_json,
)


def _provenance(*, raw_value: str = "source value") -> FieldProvenance:
    return FieldProvenance(
        artifact_id="synthetic-source",
        artifact_sha256="sha256:" + "a" * 64,
        source_record_id="native-1",
        locator_kind="json_pointer",
        location="/records/0/value",
        raw_value=raw_value,
        transformation_rule="synthetic-copy-v1",
    )


def _source_identity() -> SourceRecordIdentity:
    return SourceRecordIdentity(namespace="example.registry", record_id="native-1")


def test_normalize_sha256_adds_the_contract_prefix_and_validates() -> None:
    digest = "a" * 64

    assert normalize_sha256(digest) == f"sha256:{digest}"
    assert normalize_sha256(f"sha256:{digest}") == f"sha256:{digest}"
    with pytest.raises(SchemaValidationError, match="prefixed lowercase SHA-256"):
        normalize_sha256("not-a-digest")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"value": 1.5}, "binary floating-point values"),
        ({"value": object()}, "non-JSON value"),
    ],
)
def test_canonical_json_has_one_strict_contract(payload: object, message: str) -> None:
    with pytest.raises(SchemaValidationError, match=message):
        canonical_json(payload)


def test_snapshot_canonical_json_is_the_shared_contract_alias() -> None:
    assert canonical_json is support_canonical_json


def test_streamed_canonical_json_preserves_every_established_byte() -> None:
    payload = MappingProxyType(
        {
            "unicode": "\u03b1\u2028\u03b2",
            "escapes": 'line 1\nline 2\t"quoted"\\slash',
            "numbers": (0, -1, 2**80),
            "nested": MappingProxyType({"z": None, "a": (True, False)}),
        }
    )
    expected = json.dumps(
        json_value_to_builtins(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    assert canonical_json(payload) == expected
    assert content_sha256(payload) == "sha256:" + hashlib.sha256(expected.encode()).hexdigest()


def test_unknown_optional_topic_fields_are_preserved_but_ignorable() -> None:
    topic = BenchmarkTopic.from_dict(
        {
            "schema_version": "2.0",
            "topic_id": "1",
            "source_identity": _source_identity().to_dict(),
            "canonical_text": "patient narrative",
            "evidence_items": [],
            "future_optional_layer": {"version": "1", "value": "new"},
        }
    )

    assert topic.canonical_text == "patient narrative"
    assert topic.additional_fields == {"future_optional_layer": {"version": "1", "value": "new"}}
    assert topic.to_dict()["future_optional_layer"] == {
        "version": "1",
        "value": "new",
    }


def test_nested_additional_fields_cannot_replace_known_fields() -> None:
    with pytest.raises(SchemaValidationError, match="cannot replace known fields"):
        SemanticTextSection(
            role="brief_title",
            text="original",
            ordinal=0,
            provenance=(_provenance(),),
            additional_fields={"role": "condition"},
        )

    with pytest.raises(SchemaValidationError, match="cannot replace known fields"):
        DerivedView(
            name="synthetic_view",
            version="1",
            input_snapshot_id="sha256:" + "a" * 64,
            configuration={},
            content={},
            additional_fields={"name": "replacement"},
        )


def test_identity_bearing_system_json_is_recursively_immutable() -> None:
    topic_extensions = {"example.payload": {"items": ["original"]}}
    topic = BenchmarkTopic(
        topic_id="1",
        source_identity=_source_identity(),
        canonical_text="patient narrative",
        extensions=topic_extensions,
    )
    trial_extensions = {"example.payload": {"items": ["original"]}}
    trial = Trial(
        trial_id="NCT00000001",
        source_identity=SourceRecordIdentity(
            namespace="clinicaltrials.gov", record_id="NCT00000001"
        ),
        canonical_text="trial text",
        registry_extensions=trial_extensions,
    )
    snapshot = BenchmarkSnapshot(
        benchmark_lineage="synthetic",
        snapshot_name="synthetic",
        topics=(topic,),
        trials=(trial,),
        available_capabilities=frozenset(
            {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
        ),
    )
    snapshot_id = snapshot.snapshot_id
    assert snapshot.snapshot_id == snapshot.base_snapshot_id
    topic_extensions["example.payload"]["items"].append("source mutation")
    trial_extensions["example.payload"]["items"].clear()
    assert topic.to_dict()["extensions"] == {"example.payload": {"items": ["original"]}}
    assert trial.to_dict()["registry_extensions"] == {"example.payload": {"items": ["original"]}}
    with pytest.raises(TypeError):
        topic.extensions["example.new"] = "mutated"

    assertion_source = {"items": ["original"]}
    core = TypedPatientCore(
        profile_id="synthetic-patient-core-v1",
        fields={
            "observation": PatientCoreAssertion(
                value=assertion_source,
                provenance=(_provenance(),),
            )
        },
    )
    assertion_source["items"].append("source mutation")
    assert core.to_dict()["fields"]["observation"]["value"] == {"items": ["original"]}
    with pytest.raises(TypeError):
        core.fields["new"] = core.fields["observation"]

    view_configuration = {"steps": ["original"]}
    view_content = {"items": ["original"]}
    view = DerivedView(
        name="synthetic_view",
        version="1",
        input_snapshot_id=snapshot.base_snapshot_id,
        configuration=view_configuration,
        content=view_content,
    )
    view_id = view.view_id
    view_configuration["steps"].append("source mutation")
    view_content["items"].append("source mutation")
    assert view.to_dict()["configuration"] == {"steps": ["original"]}
    assert view.to_dict()["content"] == {"items": ["original"]}
    assert view.view_id == view_id

    assert snapshot.snapshot_id == snapshot_id
    assert (
        BenchmarkSnapshot(
            benchmark_lineage=snapshot.benchmark_lineage,
            snapshot_name=snapshot.snapshot_name,
            topics=snapshot.topics,
            trials=snapshot.trials,
            available_capabilities=snapshot.available_capabilities,
        ).snapshot_id
        == snapshot_id
    )


def test_snapshot_sealing_preserves_established_content_hashes() -> None:
    capabilities = frozenset({CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT})
    topics = tuple(
        BenchmarkTopic(
            topic_id=topic_id,
            source_identity=SourceRecordIdentity("taim.test", topic_id),
            canonical_text=text,
        )
        for topic_id, text in (("1", "patient alpha"), ("2", "patient beta"))
    )
    trials = tuple(
        Trial(
            trial_id=trial_id,
            source_identity=SourceRecordIdentity("taim.test", trial_id),
            canonical_text=text,
        )
        for trial_id, text in (
            ("NCT00000001", "trial alpha"),
            ("NCT00000002", "trial beta"),
        )
    )
    base = BenchmarkSnapshot(
        benchmark_lineage="synthetic-hash-compatibility-v1",
        snapshot_name="synthetic",
        topics=topics,
        trials=trials,
        available_capabilities=capabilities,
    )
    view = DerivedView(
        name="synthetic_view",
        version="1",
        input_snapshot_id=base.base_snapshot_id,
        configuration={"method": "fixture"},
        content={"NCT00000001": ["criterion"]},
    )
    derived = BenchmarkSnapshot(
        benchmark_lineage="synthetic-hash-compatibility-v1",
        snapshot_name="synthetic",
        topics=topics,
        trials=trials,
        available_capabilities=capabilities | {view.capability},
        derived_views=(view,),
    )

    assert base.snapshot_id == (
        "sha256:e97addba531c74fd2d7df66312989cb9f85b2cd9273e6e1a0b321c1ea9666d75"
    )
    assert derived.base_snapshot_id == base.snapshot_id
    assert derived.snapshot_id == (
        "sha256:44d0c5eccca81121bffe6eaa2289464db10d9a4ac83508c33c865ecd6d8bf9e8"
    )
    assert derived.logical_content_hashes() == {
        "topics": "sha256:d4aaca9e9137c0aaeaf9f659ba4bd6b9d58fed7d0f9789c86f8b7db5eff6d72d",
        "trials": "sha256:ebbba88bd8a69749401561e6d826a9184e6d13ffa8ea5b4cfb8e641fda2e0b81",
        "derived_views": (
            "sha256:9032026134706d00039b227882ca95aa03420a41ef5d3e55bd8a8e370dab5719"
        ),
    }


def test_snapshot_sealing_serializes_each_trial_once(monkeypatch: pytest.MonkeyPatch) -> None:
    capabilities = frozenset({CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT})
    topics = (
        BenchmarkTopic(
            topic_id="1",
            source_identity=SourceRecordIdentity("taim.test", "1"),
            canonical_text="patient",
        ),
    )
    trials = tuple(
        Trial(
            trial_id=f"NCT{index:08d}",
            source_identity=SourceRecordIdentity("taim.test", f"NCT{index:08d}"),
            canonical_text="trial text",
        )
        for index in range(100)
    )
    base = BenchmarkSnapshot(
        benchmark_lineage="synthetic-traversal-regression-v1",
        snapshot_name="synthetic",
        topics=topics,
        trials=trials,
        available_capabilities=capabilities,
    )
    view = DerivedView(
        name="synthetic_view",
        version="1",
        input_snapshot_id=base.base_snapshot_id,
        configuration={},
        content={"value": "derived"},
    )
    calls = 0
    original_to_dict = Trial.to_dict

    def counted_to_dict(document: Trial) -> dict[str, JsonValue]:
        nonlocal calls
        calls += 1
        return original_to_dict(document)

    monkeypatch.setattr(Trial, "to_dict", counted_to_dict)
    snapshot = BenchmarkSnapshot(
        benchmark_lineage="synthetic-traversal-regression-v1",
        snapshot_name="synthetic",
        topics=topics,
        trials=trials,
        available_capabilities=capabilities | {view.capability},
        derived_views=(view,),
    )
    _ = snapshot.manifest_dict()

    assert calls == len(trials)


@pytest.mark.parametrize(
    "additional_fields",
    [
        {"qrels": []},
        {"future_optional_layer": {"judgments": []}},
        {"future_optional_layer": {"evaluation_package": {}}},
    ],
)
def test_additive_system_input_fields_reject_evaluator_only_material(
    additional_fields,
) -> None:
    with pytest.raises(SchemaValidationError, match="evaluator-only"):
        BenchmarkTopic(
            topic_id="1",
            source_identity=_source_identity(),
            canonical_text="patient narrative",
            additional_fields=additional_fields,
        )


def test_trial_additive_fields_and_extensions_reject_evaluator_only_material() -> None:
    common = {
        "trial_id": "NCT00000001",
        "source_identity": SourceRecordIdentity(
            namespace="clinicaltrials.gov", record_id="NCT00000001"
        ),
        "canonical_text": "trial text",
    }

    with pytest.raises(SchemaValidationError, match="evaluator-only"):
        Trial(**common, additional_fields={"qrels": []})
    with pytest.raises(SchemaValidationError, match="evaluator-only"):
        Trial(
            **common,
            registry_extensions={"example.payload": {"judgments": []}},
        )


def test_typed_patient_core_requires_source_grounded_assertions() -> None:
    provenance = _provenance(raw_value="50 Years")
    core = TypedPatientCore(
        profile_id="synthetic-patient-core-v1",
        fields={
            "age": PatientCoreAssertion(
                value={"amount": "50", "unit": "years"},
                provenance=(provenance,),
            )
        },
    )

    assert TypedPatientCore.from_dict(core.to_dict()) == core
    assert tuple(core.iter_provenance()) == (provenance,)
    with pytest.raises(SchemaValidationError, match="PatientCoreAssertion"):
        TypedPatientCore(
            profile_id="synthetic-patient-core-v1",
            fields={"age": {"amount": "50", "unit": "years"}},
        )


def test_topic_extensions_cannot_duplicate_populated_patient_core_concepts() -> None:
    core = TypedPatientCore(
        profile_id="synthetic-patient-core-v1",
        fields={
            "age": PatientCoreAssertion(
                value={"amount": "50", "unit": "years"},
                provenance=(_provenance(),),
            )
        },
    )

    with pytest.raises(SchemaValidationError, match="patient core concept"):
        BenchmarkTopic(
            topic_id="1",
            source_identity=_source_identity(),
            canonical_text="patient narrative",
            typed_patient_core=core,
            extensions={"registry.age": {"amount": "60", "unit": "years"}},
        )


def test_benchmark_identity_is_separate_from_native_source_identity() -> None:
    topic = BenchmarkTopic(
        topic_id="benchmark-topic-7",
        source_identity=_source_identity(),
        canonical_text="patient narrative",
    )
    trial = Trial(
        trial_id="benchmark-trial-9",
        source_identity=SourceRecordIdentity(
            namespace="clinicaltrials.gov",
            record_id="NCT00000009",
        ),
        canonical_text="trial text",
    )

    assert BenchmarkTopic.from_json(topic.to_json()) == topic
    assert Trial.from_json(trial.to_json()) == trial
    assert topic.topic_id != topic.source_identity.record_id
    assert trial.trial_id != trial.source_identity.record_id


def test_multiple_and_longitudinal_patient_evidence_remains_ordered() -> None:
    provenance = _provenance()
    topic = BenchmarkTopic(
        topic_id="1",
        source_identity=_source_identity(),
        canonical_text="first note\nsecond observation",
        evidence_items=(
            PatientEvidenceItem(
                kind="clinical_note",
                ordinal=0,
                text="first note",
                timestamp="2026-01-01T10:00:00Z",
                provenance=(provenance,),
            ),
            PatientEvidenceItem(
                kind="observation",
                ordinal=1,
                text="second observation",
                interval_start="2026-01-02",
                interval_end="2026-01-03",
                provenance=(provenance,),
            ),
        ),
    )

    round_tripped = BenchmarkTopic.from_json(topic.to_json())

    assert [item.kind for item in round_tripped.evidence_items] == [
        "clinical_note",
        "observation",
    ]
    assert round_tripped.evidence_items[0].timestamp == "2026-01-01T10:00:00Z"
    assert round_tripped.evidence_items[1].interval_end == "2026-01-03"
    assert round_tripped.typed_patient_core is None


def test_only_source_authored_criteria_are_structured() -> None:
    provenance = _provenance()
    blob = SemanticTextSection(
        role="eligibility",
        text="Inclusion: adult. Exclusion: renal failure.",
        ordinal=0,
        provenance=(provenance,),
    )
    structured = SemanticTextSection(
        role="eligibility",
        text="Adult; no renal failure",
        ordinal=0,
        provenance=(provenance,),
        criteria=(
            CriterionItem(
                ordinal=0,
                identifier="criterion-1",
                polarity="inclusion",
                text="Adult",
                provenance=(provenance,),
            ),
            CriterionItem(
                ordinal=1,
                identifier="criterion-2",
                polarity="exclusion",
                text="Renal failure",
                provenance=(provenance,),
            ),
        ),
    )

    assert blob.criteria == ()
    assert [item.polarity for item in structured.criteria] == ["inclusion", "exclusion"]
    assert SemanticTextSection.from_dict(structured.to_dict()) == structured


def test_bounded_raw_excerpt_requires_full_value_digest_and_length() -> None:
    full_raw_value = "ä" * 3_000
    encoded = full_raw_value.encode("utf-8")
    excerpt = encoded[:4_096].decode("utf-8")
    digest = "sha256:" + hashlib.sha256(encoded).hexdigest()

    provenance = FieldProvenance(
        artifact_id="synthetic-source",
        artifact_sha256="sha256:" + "a" * 64,
        source_record_id="native-1",
        locator_kind="json_pointer",
        location="/records/0/value",
        raw_value=excerpt,
        raw_value_sha256=digest,
        raw_value_byte_length=len(encoded),
        transformation_rule="synthetic-copy-v1",
    )

    assert FieldProvenance.from_dict(provenance.to_dict()) == provenance
    with pytest.raises(SchemaValidationError, match="bounded excerpt metadata"):
        FieldProvenance(
            artifact_id="synthetic-source",
            artifact_sha256="sha256:" + "a" * 64,
            source_record_id="native-1",
            locator_kind="json_pointer",
            location="/records/0/value",
            raw_value=excerpt,
            raw_value_sha256=None,
            raw_value_byte_length=len(encoded),
            transformation_rule="synthetic-copy-v1",
        )


def test_source_fragment_provenance_supports_dataset_owned_tabular_locators() -> None:
    provenance = FieldProvenance(
        artifact_id="synthetic-source",
        artifact_sha256="sha256:" + "a" * 64,
        source_record_id="native-1",
        locator_kind="source_fragment",
        location="fragment:csv-cell:row=17;column=eligibility",
        raw_value="value",
        transformation_rule="synthetic-copy-v1",
    )

    assert FieldProvenance.from_dict(provenance.to_dict()) == provenance


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": "2.0",
            "topic_id": "1",
            "source_identity": {
                "namespace": "example.registry",
                "record_id": "native-1",
            },
            "canonical_text": "patient",
            "evidence_items": [None],
        },
        {
            "schema_version": "2.0",
            "trial_id": "NCT00000001",
            "source_identity": {
                "namespace": "clinicaltrials.gov",
                "record_id": "NCT00000001",
            },
            "canonical_text": "trial",
            "sections": [None],
        },
    ],
)
def test_malformed_nested_records_are_rejected_instead_of_dropped(payload) -> None:
    schema = BenchmarkTopic if "topic_id" in payload else Trial

    with pytest.raises(SchemaValidationError, match="must contain objects"):
        schema.from_dict(payload)


def test_registry_extensions_cannot_duplicate_shared_core_authority() -> None:
    provenance = _provenance()

    with pytest.raises(SchemaValidationError, match="shared core concept"):
        Trial(
            trial_id="NCT00000001",
            source_identity=SourceRecordIdentity(
                namespace="clinicaltrials.gov",
                record_id="NCT00000001",
            ),
            canonical_text="trial",
            typed_clinical_core=TypedClinicalCore(
                minimum_age=AgeBound(AgeQuantity("18", "years"), (provenance,)),
            ),
            registry_extensions={"ctgov.minimum_age": "20 Years"},
        )


def test_registry_extensions_require_an_explicit_namespace() -> None:
    with pytest.raises(SchemaValidationError, match="namespace"):
        Trial(
            trial_id="NCT00000001",
            source_identity=SourceRecordIdentity(
                namespace="clinicaltrials.gov",
                record_id="NCT00000001",
            ),
            canonical_text="trial",
            registry_extensions={"phase": "phase-2"},
        )


@pytest.mark.parametrize("unit", ["year", "fortnights", "YEARS"])
def test_age_units_are_a_closed_versioned_vocabulary(unit: str) -> None:
    with pytest.raises(SchemaValidationError, match="unsupported age unit"):
        AgeQuantity("18", unit)


def test_sex_eligibility_is_a_closed_versioned_vocabulary() -> None:
    with pytest.raises(SchemaValidationError, match="unsupported sex eligibility"):
        SexEligibility("unknown", (_provenance(),))


def test_explicit_source_authored_lower_unboundedness_is_not_missing() -> None:
    core = TypedClinicalCore(
        minimum_age=AgeBound("unbounded", (_provenance(raw_value="No lower limit"),)),
    )

    assert core.minimum_age is not None
    assert core.minimum_age.value == "unbounded"
    assert render_canonical_trial_text((), core) == "[MINIMUM_AGE] unbounded"


def test_exact_decimal_encoding_has_one_canonical_zero() -> None:
    assert AgeQuantity("-0.000", "years").amount == "0"


@pytest.mark.parametrize(
    "location",
    [
        "file:///tmp/source.xml",
        "C:\\data\\source.xml",
        "/Volumes/data/source.xml",
        "/media/research/source.xml",
        "/mnt/research/source.xml",
        "/opt/data/source.xml",
        "/snap/research/source.xml",
    ],
)
def test_provenance_forbids_local_filesystem_locations(location: str) -> None:
    with pytest.raises(SchemaValidationError, match="local path"):
        FieldProvenance(
            artifact_id="synthetic-source",
            artifact_sha256="sha256:" + "a" * 64,
            source_record_id="native-1",
            locator_kind="xml_path",
            location=location,
            raw_value="value",
            transformation_rule="synthetic-copy-v1",
        )


@pytest.mark.parametrize(
    "location",
    [
        "fragment:file:/tmp/synthetic-private.csv",
        "fragment:csv-cell:/etc/passwd",
    ],
)
def test_source_fragment_provenance_forbids_embedded_local_paths(location: str) -> None:
    with pytest.raises(SchemaValidationError, match="local path"):
        FieldProvenance(
            artifact_id="synthetic-source",
            artifact_sha256="sha256:" + "a" * 64,
            source_record_id="native-1",
            locator_kind="source_fragment",
            location=location,
            raw_value="value",
            transformation_rule="synthetic-copy-v1",
        )


@pytest.mark.parametrize(
    ("locator_kind", "location", "message"),
    [
        ("local_path", "/data/source.xml", "locator_kind"),
        ("archive_member", "zip-member:../source.xml", "archive"),
        ("json_pointer", "/records/~2value", "JSON Pointer"),
        ("xml_path", "clinical_study/brief_title", "rooted element path"),
        ("fhir_element_path", "Observation/value", "resource identity"),
        ("source_fragment", "row=17;column=value", "source-fragment"),
    ],
)
def test_provenance_locator_schemas_reject_malformed_locations(
    locator_kind: str,
    location: str,
    message: str,
) -> None:
    with pytest.raises(SchemaValidationError, match=message):
        FieldProvenance(
            artifact_id="synthetic-source",
            artifact_sha256="sha256:" + "a" * 64,
            source_record_id="native-1",
            locator_kind=locator_kind,
            location=location,
            raw_value="value",
            transformation_rule="synthetic-copy-v1",
        )


@pytest.mark.parametrize(
    ("recipe_name", "digest"),
    [
        ("SOURCE_RECIPE", "86a437499cb1363df09040154f20c6e5f63931664719d77d764f4ea48b02233a"),
        (
            "CANONICAL_TEXT_RECIPE",
            "94257b52fd18e4d215416bc3ae8ad8fb66ef5246b4ee7e56b2ff0dbb0b9b53c9",
        ),
        (
            "CANONICAL_PATIENT_TEXT_RECIPE",
            "44a5fae9f72d5b5f7bc90ffafc8d51e16ed8ca477fc4116b619ddc1a44a55ed0",
        ),
    ],
)
def test_recipe_freezing_preserves_canonical_identity(recipe_name, digest) -> None:
    from taim.data import trec_ct_2021

    assert content_sha256(getattr(trec_ct_2021, recipe_name)) == "sha256:" + digest


def test_nested_recipe_mutation_cannot_change_identical_input_rendering() -> None:
    from taim.data.trec_ct_2021 import CANONICAL_TEXT_RECIPE, SOURCE_RECIPE

    sections = (
        SemanticTextSection(role="summary", text="alpha", ordinal=0, provenance=(_provenance(),)),
    )
    before = render_canonical_trial_text(sections, None)
    labels = CANONICAL_TEXT_RECIPE["section_labels"]
    with pytest.raises(TypeError):
        labels["summary"] = "CHANGED"
    with pytest.raises(TypeError):
        dict.__setitem__(labels, "summary", "CHANGED")
    with pytest.raises(TypeError):
        SOURCE_RECIPE["canonical_trial_text_recipe"]["section_order"][0] = "summary"
    detached = json_value_to_builtins(SOURCE_RECIPE)
    detached["canonical_trial_text_recipe"]["section_labels"]["summary"] = "CHANGED"
    assert render_canonical_trial_text(sections, None) == before == "[SUMMARY] alpha"


def test_eligibility_configuration_is_immutable_without_changing_identity() -> None:
    from taim.eligibility import ELIGIBILITY_VIEW_CONFIGURATION

    with pytest.raises(TypeError):
        ELIGIBILITY_VIEW_CONFIGURATION["missing_heading_policy"] = "invent_polarity"
    assert content_sha256(ELIGIBILITY_VIEW_CONFIGURATION) == (
        "sha256:c92f9d09972ebace72339c47f95981f6bd8d7f887ba65b118bfb4a937337219b"
    )
