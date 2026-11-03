from __future__ import annotations

# ruff: noqa: S101
import pytest

from taim.baselines.encoder_registry import DENSE_ENCODER_REGISTRY, QWEN3_EMBEDDING_06B_POLICY
from taim.pipelines.staged.pipeline import load_staged_pipeline, profiled_operator_names
from taim.release_result_bundle import _expected_pipeline_depth
from taim.release_support import load_release_support, require_supported_combination


def test_paper_pipeline_is_exactly_scoped_to_trec_2021_profiles() -> None:
    pipeline = load_staged_pipeline()

    assert pipeline.system_id == "staged-bm25-qwen3-rrf-rerank"
    assert pipeline.track == "trec-ct-2021"
    assert pipeline.profiles == (
        "official-full",
        "trec-ct-2021-external-fidelity-26149",
        "trec-ct-2021-judgment-union",
    )
    assert (pipeline.component_depth, pipeline.fusion_depth) == (5_000, 2_000)
    assert (pipeline.scoring_depth, pipeline.output_depth) == (2_000, 1_000)
    assert pipeline.reranker["model_revision"] == ("22e683669bc0f0bd69640a1354a6d0aebcfeede5")
    assert (
        require_supported_combination(
            track="trec-ct-2021",
            task="patient_to_trial",
            profile="trec-ct-2021-external-fidelity-26149",
            system=pipeline.system_id,
        )["evidence_scope"]
        == "external_fidelity_effectiveness"
    )
    assert (
        require_supported_combination(
            track="trec-ct-2021",
            task="patient_to_trial",
            profile="official-full",
            system=pipeline.system_id,
        )["evidence_scope"]
        == "real_effectiveness"
    )
    assert (
        require_supported_combination(
            track="trec-ct-2021",
            task="patient_to_trial",
            profile="trec-ct-2021-judgment-union",
            system=pipeline.system_id,
        )["evidence_scope"]
        == "within_pool_effectiveness"
    )


def test_every_staged_profile_supports_every_component_system() -> None:
    pipeline = load_staged_pipeline()

    for profile in pipeline.profiles:
        for system in pipeline.arm_system_ids:
            capability = require_supported_combination(
                track=pipeline.track,
                task="patient_to_trial",
                profile=profile,
                system=system,
            )
            assert capability["profile"] == profile
            assert capability["system"] == system


def test_no_template_qwen_policy_is_distinct_from_the_main_matrix_policy() -> None:
    instructed = QWEN3_EMBEDDING_06B_POLICY
    no_template = DENSE_ENCODER_REGISTRY.policy_for_system("dense-qwen3-embedding-0.6b-no-template")

    assert instructed.model_revision == no_template.model_revision
    assert instructed.query_format.render("patient") != "patient"
    assert no_template.query_format.render("patient") == "patient"


def test_result_bundle_pipeline_depth_of_the_staged_system_is_rerank() -> None:
    assert _expected_pipeline_depth("staged-bm25-qwen3-rrf-rerank") == "rerank"


def test_the_raw_profiler_event_list_names_the_operators_that_ran() -> None:
    # The reranker reads the attention kernels from torch's private raw event list, never the
    # public one. A torch that drops or reshapes that list fails here, by version, before a
    # reranker run refuses it.
    torch = pytest.importorskip("torch")
    query = torch.ones(1, 1, 2, 4)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        torch.zeros(1)
        torch.nn.functional.scaled_dot_product_attention(query, query, query)
    try:
        names = set(profiled_operator_names(torch, trace))
    except ValueError as exc:
        pytest.fail(str(exc))
    public = {event.name for event in trace.events()}
    assert (
        names == public
        and {"aten::zeros", "aten::scaled_dot_product_attention"} <= names
        and any(name.startswith("aten::_scaled_dot_product_") for name in names)
    ), f"torch {torch.__version__}: raw list {sorted(names)}, public list {sorted(public)}"


def test_every_staged_support_row_supports_its_components_on_the_same_profile() -> None:
    pipeline = load_staged_pipeline()
    rows = [
        row
        for row in load_release_support()["support_matrix"]
        if row["system"] == pipeline.system_id
    ]
    assert {row["profile"] for row in rows} >= set(pipeline.profiles)
    for row in rows:
        for system in pipeline.arm_system_ids:
            require_supported_combination(
                track=row["track"], task=row["task"], profile=row["profile"], system=system
            )


def test_the_declaration_is_one_definition_wherever_the_system_runs() -> None:
    # The support rows say where the System runs; its declaration, and this digest, never change
    # with them.
    assert load_staged_pipeline().definition_sha256 == (
        "sha256:eb2f0716019fedc036ef90dcad75a30a199dcb7c50c49ce0156a2945a3c5b317"
    )
