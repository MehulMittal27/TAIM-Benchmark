"""Selected dense units construct their pinned encoder without loading model weights."""

from __future__ import annotations

# ruff: noqa: S101
import pytest

from taim.baselines.encoder_registry import DENSE_ENCODER_REGISTRY
from taim.pipeline_extensions import extension_for
from taim.release_reference_systems import ReleaseDenseSystem


@pytest.mark.parametrize("system_id", DENSE_ENCODER_REGISTRY.available_system_ids())
def test_selected_dense_unit_binds_its_production_encoder_policy(system_id: str) -> None:
    policy = DENSE_ENCODER_REGISTRY.policy_for_system(system_id)
    extension = extension_for(system_id)
    assert extension is not None
    assert extension.system_tasks[system_id] == "patient_to_trial"
    assert ReleaseDenseSystem(system_id).system_id == system_id

    encoder = DENSE_ENCODER_REGISTRY.create_encoder(system_id, device="cpu")
    assert encoder.policy is policy
    configuration = encoder.configuration()
    for field in (
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "embedding_dimension",
        "maximum_length",
        "pooling",
        "normalization",
    ):
        assert configuration[field] == getattr(policy, field)
    assert len(policy.model_revision) == 40
    assert policy.policy_hash().startswith("sha256:")
