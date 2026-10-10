from __future__ import annotations

import json
from pathlib import Path

import pytest

from taim.contracts import content_sha256
from taim.file_hash import sha256_file


def _release_identity() -> dict[str, str]:
    return {
        "release_manifest_id": "sha256:" + "1" * 64,
        "public_tree_id": "sha256:" + "2" * 64,
        "package_artifact_id": "sha256:" + "3" * 64,
    }


def _dependency_environment() -> dict[str, object]:
    core = {
        "schema_version": "1.0",
        "dependency_lock_sha256": "sha256:" + "4" * 64,
        "python_version": "3.13.7",
        "python_implementation": "CPython",
        "platform": "darwin",
        "machine": "arm64",
        "distributions": [{"name": "taim", "version": "0.1.0"}],
    }
    return {**core, "environment_id": content_sha256(core)}


@pytest.fixture(autouse=True)
def _clean_test_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("taim.release_producer.git_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(
        "taim.release_producer.verified_release_identity", lambda _args: _release_identity()
    )
    monkeypatch.setattr(
        "taim.release_producer.verified_dependency_environment",
        lambda _args, *, system_id: _dependency_environment(),
    )


def _declaration() -> dict[str, object]:
    return {
        "analysis_plan_id": "sha256:" + "4" * 64,
        "subset_rule": "all packaged synthetic fixture records",
        "metrics": ["ndcg", "relevant_or_eligible_recall"],
        "inclusion_rule": "all valid synthetic-fixture runs from this CI job",
        "score_guided_selection": False,
        "protocol_approval_id": None,
    }


def _rehash_bundle(bundle: Path) -> None:
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        path = bundle / entry["path"]
        entry["sha256"] = sha256_file(path)
        entry["byte_size"] = path.stat().st_size
    core = {key: value for key, value in manifest.items() if key != "bundle_id"}
    manifest["bundle_id"] = content_sha256(core)
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = ("bundle.json", "candidates.jsonl", "metrics.json", "run.json")
    (bundle / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(bundle / name).removeprefix('sha256:')}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )


def _fixture_task_source(directory):
    from taim.entity_versions import PatientEvidenceProfile
    from taim.patient_to_trial import PatientToTrialTaskInput
    from taim.profiles import resolve_benchmark_profile
    from taim.release_fixture import load_release_fixture
    from taim.schemas import read_manifest

    manifest = read_manifest(directory / "manifest.json")
    return PatientToTrialTaskInput.from_benchmark(
        resolve_benchmark_profile(
            load_release_fixture(manifest.benchmark_lineage),
            profile_id=manifest.benchmark_profile["profile_id"],
        ),
        clinical_as_of=manifest.clinical_as_of,
        patient_evidence_profile=PatientEvidenceProfile.from_dict(
            manifest.patient_evidence_profile
        ),
    )
