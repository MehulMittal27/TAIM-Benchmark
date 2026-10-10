from __future__ import annotations

# ruff: noqa: S101
import json
import re
from importlib.resources import files
from pathlib import Path

from taim.data.trec_ct_2021 import load_source_bundle_lock
from taim.file_hash import sha256_file
from taim.pipeline_extensions import pipeline_extensions
from taim.release_cli import main
from taim.release_support import (
    FROZEN_PAPER_PROTOCOL_SHA256,
    RELEASE_SUPPORT_RESOURCE,
    SUPPORTED_TRACKS,
    capability_key,
    load_release_support,
)


def test_support_catalog_uses_declared_tracks_and_selected_tasks() -> None:
    matrix = load_release_support()["support_matrix"]
    assert matrix
    assert {row["track"] for row in matrix} <= set(SUPPORTED_TRACKS)
    assert {row["task"] for row in matrix} == {
        task for extension in pipeline_extensions() for task in extension.system_tasks.values()
    }


def test_frozen_protocol_identity_matches_public_protocol_bytes() -> None:
    protocol = Path(__file__).parents[1] / "docs" / "paper-analysis-protocol-2026-08-31-v3.md"
    assert sha256_file(protocol) == FROZEN_PAPER_PROTOCOL_SHA256


def test_support_command_is_machine_readable(capsys) -> None:
    assert main(["support", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)["support_matrix"]
    # A release publishes a selection of pipelines, so completeness is measured against the
    # catalogue this release ships, not against one selection's row count.
    shipped = json.loads(
        files("taim").joinpath("data", RELEASE_SUPPORT_RESOURCE).read_text(encoding="utf-8")
    )["support_matrix"]
    assert shipped
    assert sorted(capability_key(row) for row in listed) == sorted(
        capability_key(row) for row in shipped
    )


def test_trec_2021_acquisition_guide_lists_the_packaged_lock() -> None:
    # The guide copies each file's URL, size and digest out of the lock so a reader can download
    # without opening JSON; this keeps the copy from drifting.
    guide = Path(__file__).parents[1] / "docs" / "trec-2021-acquisition.md"
    rows = re.findall(
        r"^\| \[`([^`]+)`\]\((\S+)\) \| ([\d,]+) \| `([0-9a-f]{64})` \|$",
        guide.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert [
        (name, url, int(size.replace(",", "")), f"sha256:{digest}")
        for name, url, size, digest in rows
    ] == [
        (artifact.filename, artifact.url, artifact.byte_size, artifact.sha256)
        for artifact in load_source_bundle_lock().artifacts
    ]
