from __future__ import annotations

# ruff: noqa: S101
from taim.baselines.encoder_registry import DENSE_ENCODER_REGISTRY
from taim.release_result_bundle import _expected_pipeline_depth
from taim.release_staged_pipeline import load_staged_pipeline
from taim.release_support import require_supported_combination


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
    instructed = DENSE_ENCODER_REGISTRY.policy_for_system("dense-qwen3-embedding-0.6b")
    no_template = DENSE_ENCODER_REGISTRY.policy_for_system("dense-qwen3-embedding-0.6b-no-template")

    assert instructed.model_revision == no_template.model_revision
    assert instructed.query_format.render("patient") != "patient"
    assert no_template.query_format.render("patient") == "patient"


def test_result_bundle_pipeline_depth_of_the_staged_system_is_rerank() -> None:
    assert _expected_pipeline_depth("staged-bm25-qwen3-rrf-rerank") == "rerank"
