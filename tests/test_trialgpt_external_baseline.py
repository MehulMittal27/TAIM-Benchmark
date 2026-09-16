from __future__ import annotations

# ruff: noqa: S101
import json
from pathlib import Path

import pytest

from taim.pipeline_extensions import pipeline_extensions
from taim.release_cli import build_parser
from taim.release_profiles import (
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
)
from taim.release_result_bundle import _expected_pipeline_depth
from taim.release_support import (
    FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256,
    load_release_support,
    require_frozen_effectiveness_protocol,
)
from taim.schemas import SchemaValidationError
from taim.trialgpt_publication import TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID


def test_the_trialgpt_extension_declares_its_system_and_its_external_fidelity_row() -> None:
    # The core external-fidelity test checks every System outside the pipeline extensions.
    rows = [
        row
        for row in load_release_support()["support_matrix"]
        if row["profile"] == EXTERNAL_FIDELITY_TREC_2021_PROFILE
    ]
    (extension,) = [
        extension
        for extension in pipeline_extensions()
        if extension.external_baseline and TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID in extension.system_tasks
    ]
    assert set(extension.system_tasks) == {TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID}
    assert {row["system"] for row in rows} & set(extension.system_tasks) == {
        TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID
    }


def test_trialgpt_result_bundle_pipeline_depth_is_post_eligibility() -> None:
    assert _expected_pipeline_depth(TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID) == "post_eligibility"


def test_trialgpt_adaptations_use_existing_other_track_judgment_unions() -> None:
    rows = load_release_support()["support_matrix"]
    observed = {
        (row["track"], row["profile"], row["system"])
        for row in rows
        if row["track"] != "trec-ct-2021" and row["system"] == TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID
    }
    assert observed == {
        (
            "sigir-ct-2016",
            "sigir-ct-2016-description-judgment-union",
            TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
        ),
        (
            "sigir-ct-2016",
            "sigir-ct-2016-summary-judgment-union",
            TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
        ),
        (
            "trec-ct-2022",
            "trec-ct-2022-judgment-union",
            TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
        ),
    }


def test_projected_trialgpt_lock_is_metadata_only_and_license_bound() -> None:
    public_root = Path(__file__).parents[1]
    projected_tree = (public_root / "external").is_dir()
    root = public_root
    if not projected_tree:
        root = Path(__file__).parents[3]
    trialgpt = json.loads((root / "external" / "trialgpt-lock.json").read_text())
    assert trialgpt["canonical"]["license"]["expression"] == "LicenseRef-NCBI-Public-Domain"
    if projected_tree:
        assert not any((root / "external").glob("*/"))


def test_external_effectiveness_protocol_gate_matches_release_state() -> None:
    protocol_id = FROZEN_EXTERNAL_FIDELITY_PROTOCOL_SHA256
    if protocol_id is None:
        with pytest.raises(SchemaValidationError, match="not frozen by explicit human approval"):
            require_frozen_effectiveness_protocol(
                "trec-ct-2022-judgment-union",
                "sha256:" + "0" * 64,
                "external protocol",
                system=TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
            )
        return
    assert (
        require_frozen_effectiveness_protocol(
            "trec-ct-2022-judgment-union",
            protocol_id,
            "external protocol",
            system=TRIALGPT_TAIM_LUNA_V1_SYSTEM_ID,
        )
        == protocol_id
    )


def test_trialgpt_preflight_is_a_public_direction_specific_cli() -> None:
    args = build_parser().parse_args(
        [
            "patient-to-trial",
            "trialgpt",
            "preflight",
            "--data-dir",
            "prepared",
            "--clinical-as-of",
            "2021-04-27T00:00:00Z",
            "--pool-file",
            "pool.json",
            "--pool-count",
            "26149",
            "--pool-ids-sha256",
            "sha256:" + "1" * 64,
            "--pool-receipt-id",
            "sha256:" + "2" * 64,
            "--protocol-approval-id",
            "sha256:" + "3" * 64,
            "--frozen-retrieval",
            "retrieval.jsonl",
            "--frozen-retrieval-lock",
            "retrieval-lock.json",
            "--output",
            "contract.json",
        ]
    )
    assert args.command == "patient-to-trial"
    assert args.operation == "trialgpt"
    assert args.trialgpt_operation == "preflight"
    assert args.generation_workers == 48


def test_public_trialgpt_adapter_surface_is_selected_and_producer_is_pinned() -> None:
    candidate_root = Path(__file__).parents[1]
    if (candidate_root / "src" / "taim").is_dir():
        root = candidate_root
    else:
        root = Path(__file__).parents[3]
    if (root / "release" / "public").is_dir():
        trialgpt = (root / "release" / "public" / "trialgpt-adapter.py").read_text()
        method = (root / "release" / "public" / "trialgpt-method.py").read_text()
        producer = (root / "src" / "taim" / "trialgpt_retrieval_backends.py").read_text()
    else:
        trialgpt = (root / "src" / "taim" / "adapters" / "trialgpt.py").read_text()
        method = (root / "src" / "taim" / "trialgpt_paper.py").read_text()
        producer = (root / "src" / "taim" / "trialgpt_retrieval_backends.py").read_text()
    for forbidden in (
        "class PaperTrialGPTSystem",
        "class TrialGPTCodexFailureAwareSystem",
        "class TrialGPTCodexSystem",
        "class TrialGPTFakeSystem",
        "class TrialMatchAISystem",
    ):
        assert forbidden not in trialgpt
    assert "MedCPT" not in method
    assert "ncbi/MedCPT-Article-Encoder" in producer
    assert "d05a736da4bb84ee4057b7f7999485be6ed85465" in producer
    assert "ncbi/MedCPT-Query-Encoder" in producer
    assert "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc" in producer
