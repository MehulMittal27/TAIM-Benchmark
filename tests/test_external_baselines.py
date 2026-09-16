from __future__ import annotations

# ruff: noqa: S101
import json
from pathlib import Path

from taim.pipeline_extensions import pipeline_extensions


def test_the_dependency_registry_covers_exactly_the_shipped_external_systems() -> None:
    # Each external family's own test file covers its Systems; this holds whichever families ship.
    root = Path(__file__).parents[1]
    if not (root / "external").is_dir():
        root = Path(__file__).parents[3]
    registry = json.loads((root / "src/taim/data/external-system-dependencies-v1.json").read_text())
    external_systems = {
        system
        for extension in pipeline_extensions()
        if extension.external_baseline
        for system in extension.system_tasks
    }
    assert external_systems
    assert {entry["system_id"] for entry in registry["systems"]} == external_systems
