"""The declarations of a target's two TrialGPT paper rows, derived from the release and the package.

The pinned identities are checked against the frozen protocols' own tables; the derivation runs on
a synthetic SIGIR summary complete-eligibility package, which the pins refuse, as they must.
"""

# ruff: noqa: S101
from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import taim.pipelines.trialgpt.paper_row_declarations as declarations
from taim.artifacts import sha256_file
from taim.data import load_prepared_benchmark
from taim.data.complete_eligibility import prepare_profile_complete_eligibility_snapshot
from taim.data.sigir_ct_2016 import prepare_sigir_ct_2016
from taim.pipelines.trialgpt.paper_row_declarations import (
    PAPER_ROW_AMENDMENT,
    PAPER_ROW_BUDGET_K,
    PAPER_ROW_PROTOCOL_ID,
    PAPER_ROW_PROTOCOL_SHA256,
    PAPER_ROW_STRATA,
    PaperRowDeclarationError,
    release_declaration_identity,
    write_paper_row_declarations,
)
from taim.release_profiles import inspect_judgment_union_profile
from tests.snapshot_helpers import TREC_FIXED_TIME, write_sigir_source_bundle

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_PROFILE = "sigir-ct-2016-summary-judgment-union"
_RELEASE = {
    "release_version": "9.9.9",
    "release_manifest_id": "sha256:" + "1" * 64,
    "public_tree_id": "sha256:" + "2" * 64,
    "package_artifact_id": "sha256:" + "3" * 64,
    "dependency_lock_sha256": "sha256:" + "4" * 64,
}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_taim_{name}", _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: The command derives a package's stratum; pipeline code does not read the release's Profiles.
package_stratum = _script("write_trialgpt_row_declarations").package_stratum


def _section(document: str, heading: str) -> str:
    text = (_DOCS / document).read_text(encoding="utf-8")
    start = text.index(f"\n## {heading}")
    return text[start : text.index("\n## ", start + 1)]


def _row(section: str, profile: str) -> str:
    (line,) = [
        line for line in section.splitlines() if line.startswith("|") and f"`{profile}`" in line
    ]
    return line


def test_the_pinned_identities_are_the_frozen_protocols() -> None:
    v15 = _section("paper-analysis-protocol-2026-10-07-v15.md", "4. The identities")
    v12 = _section(
        "paper-analysis-protocol-2026-09-28-v12.md", "4. The strata and their identities"
    )
    for stratum in PAPER_ROW_STRATA:
        row = _row(v15, stratum.profile)
        for value in (
            stratum.prepared_snapshot_id,
            stratum.task_input_id,
            stratum.evaluation_package_id,
            stratum.pool_ids_sha256,
            stratum.pool_receipt_id,
        ):
            assert f"`{value}`" in row, (stratum.profile, value)
        assert f"| {stratum.pool_count:,} |" in row
        assert f"| {stratum.clinical_as_of[:10]} |" in row
        assert f"`{stratum.profile_definition_sha256}`" in _row(v12, stratum.profile)
    assert len(PAPER_ROW_STRATA) == len(re.findall(r"^\| (TREC|SIGIR)", v15, re.MULTILINE))
    assert sha256_file(_DOCS / "paper-analysis-protocol-2026-09-28-v12.md") == (
        PAPER_ROW_PROTOCOL_SHA256
    )
    assert (
        sha256_file(_DOCS / "paper-analysis-protocol-2026-10-07-v14.md")
        == (PAPER_ROW_AMENDMENT["protocol_sha256"])
    )
    assert PAPER_ROW_PROTOCOL_ID.endswith("2026-09-28-v12")
    assert PAPER_ROW_AMENDMENT["protocol_id"].endswith("2026-10-07-v14")


def test_the_budget_and_cutoff_are_those_the_rows_are_run_with(tmp_path) -> None:
    runner = _script("run_trialgpt_target")
    assert str(PAPER_ROW_BUDGET_K) == runner.TOP_K
    args = runner._parser().parse_args(
        [
            *("--track", "sigir-ct-2016", "--profile", _PROFILE),
            *("--data-dir", str(tmp_path), "--clinical-as-of", "2015-12-16T00:00:00Z"),
            *("--pool-count", "3626", "--pool-ids-sha256", "sha256:" + "1" * 64),
            *("--pool-receipt-id", "sha256:" + "2" * 64),
            *("--protocol-approval-id", "sha256:" + "3" * 64),
            *("--frozen-retrieval", str(tmp_path), "--frozen-retrieval-lock", str(tmp_path)),
            *("--release-manifest", str(tmp_path), "--package-artifact", str(tmp_path)),
            *("--dependency-lock", str(tmp_path), "--trialgpt2-checkout", str(tmp_path)),
            *("--trialgpt-run-id", "row-1", "--trialgpt2-run-id", "row-2"),
            *("--output-dir", str(tmp_path), "--work-dir", str(tmp_path)),
        ]
    )
    # Neither row is given a cutoff, so each runs at the CLI's default, which the declaration names.
    for command in (runner.trialgpt_row_command(args), runner.trialgpt2_row_command(args)):
        assert "--metric-cutoff" not in command
        assert command[command.index("--top-k") + 1] == str(PAPER_ROW_BUDGET_K)


@pytest.fixture(scope="module")
def complete_package(tmp_path_factory) -> Path:
    """A synthetic SIGIR summary complete-eligibility package."""

    root = tmp_path_factory.mktemp("sigir")
    source, lock = write_sigir_source_bundle(root)
    stratum = prepare_sigir_ct_2016(
        source,
        root / "stratum",
        source_lock_path=lock,
        query_variant="summary",
        created_at=TREC_FIXED_TIME,
    ).output_directory
    pool = inspect_judgment_union_profile(load_prepared_benchmark(stratum), profile_id=_PROFILE)
    return prepare_profile_complete_eligibility_snapshot(
        stratum,
        source,
        root / "complete",
        profile_id=_PROFILE,
        expected_pool_count=pool["pool_count"],
        expected_pool_ids_sha256=pool["pool_ids_sha256"],
        expected_pool_receipt_id=pool["pool_receipt_id"],
        created_at=datetime(2026, 10, 7, tzinfo=UTC),
    ).output_directory


def _write(stratum, output: Path, **overrides):
    return write_paper_row_declarations(
        **{
            "stratum": stratum,
            "release": _RELEASE,
            "trialgpt_run_id": "paper-trialgpt1-sigirs-row-1",
            "trialgpt2_run_id": "paper-trialgpt2-sigirs-row-1",
            "trialgpt_comparisons": [],
            "declared_by": "A. Researcher",
            "output_dir": output,
            "now": datetime(2026, 10, 8, 12, 0, tzinfo=UTC),
            **overrides,
        }
    )


def test_a_package_other_than_v15_s_is_refused_and_nothing_is_declared(
    complete_package, tmp_path
) -> None:
    derived = package_stratum(
        complete_package, track="sigir-ct-2016", profile=_PROFILE, pool_ids=None
    )
    with pytest.raises(PaperRowDeclarationError, match="does not reproduce protocol v15") as error:
        _write(derived, tmp_path / "declarations")
    message = str(error.value)
    assert f"prepared_snapshot_id: the package has {derived.prepared_snapshot_id}" in message
    assert "task_input_id" in message and "pool_receipt_id" in message
    assert "profile_definition_sha256" not in message
    assert not (tmp_path / "declarations").exists()


def test_both_rows_are_declared_from_the_package(complete_package, tmp_path, monkeypatch) -> None:
    derived = package_stratum(
        complete_package, track="sigir-ct-2016", profile=_PROFILE, pool_ids=None
    )
    # Pinned to this synthetic package, as v15 Section 4 pins the real one.
    monkeypatch.setattr(declarations, "PAPER_ROW_STRATA", (derived,))
    first, second = _write(
        derived, tmp_path / "declarations", trialgpt_comparisons=["paper-bm25-sigirs-row-1"]
    )
    trialgpt, trialgpt2 = (json.loads(path.read_text(encoding="utf-8")) for path in (first, second))
    assert first.name == "paper-trialgpt1-sigirs-row-1.json"
    assert (trialgpt["system_id"], trialgpt2["system_id"]) == (
        "TrialGPT-TAIM-Luna-v1",
        "TrialGPT2-TAIM-Luna-v1",
    )
    for declaration in (trialgpt, trialgpt2):
        assert declaration["artifact_type"] == "taim-paper-row-declaration"
        assert declaration["protocol_sha256"] == PAPER_ROW_PROTOCOL_SHA256
        assert declaration["release"] == _RELEASE
        assert declaration["stratum"] == {
            "task": "patient-to-trial",
            "track": "sigir-ct-2016",
            "profile": _PROFILE,
            "profile_definition_sha256": derived.profile_definition_sha256,
            "prepared_snapshot_id": derived.prepared_snapshot_id,
            "task_input_id": derived.task_input_id,
            "evaluation_package_id": derived.evaluation_package_id,
            "pool_count": derived.pool_count,
            "pool_ids_sha256": derived.pool_ids_sha256,
            "pool_receipt_id": derived.pool_receipt_id,
            "clinical_as_of": "2015-12-16T00:00:00Z",
        }
        assert (declaration["budget_k"], declaration["metric_cutoff"]) == (20, 5)
        assert (declaration["environment"], declaration["ordinal"]) == ("CPU", 1)
        assert declaration["amendment"] == dict(PAPER_ROW_AMENDMENT)
        assert declaration["declared_at"] == "2026-10-08T12:00:00Z"
    assert [item["run_id"] for item in trialgpt["comparisons"]] == ["paper-bm25-sigirs-row-1"]
    assert trialgpt2["comparisons"] == []
    with pytest.raises(PaperRowDeclarationError, match="written once"):
        _write(derived, tmp_path / "declarations")
    with pytest.raises(PaperRowDeclarationError, match="v14 Section 7"):
        _write(
            derived,
            tmp_path / "other",
            trialgpt_comparisons=["paper-trialgpt2-sigirs-row-1"],
        )


def test_a_stratum_package_and_a_wrong_pool_file_are_refused(complete_package, tmp_path) -> None:
    stratum = complete_package.parent / "stratum"
    with pytest.raises(ValueError, match=r"prepare_trialgpt_profile_snapshot\.py"):
        package_stratum(stratum, track="sigir-ct-2016", profile=_PROFILE, pool_ids=None)
    with pytest.raises(PaperRowDeclarationError, match="does not accept a pool file"):
        package_stratum(complete_package, track="sigir-ct-2016", profile=_PROFILE, pool_ids=("x",))
    with pytest.raises(PaperRowDeclarationError, match="requires a pool file"):
        package_stratum(
            complete_package,
            track="trec-ct-2021",
            profile="trec-ct-2021-external-fidelity-26149",
            pool_ids=None,
        )


def test_the_release_is_the_verified_manifest_and_its_own_lock(tmp_path, monkeypatch) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "release_version": "9.9.9",
                "files": [{"destination": "uv.lock", "sha256": sha256_file(lock)}],
            }
        ),
        encoding="utf-8",
    )
    verified = {key: _RELEASE[key] for key in ("release_manifest_id", "public_tree_id")}
    verified["package_artifact_id"] = _RELEASE["package_artifact_id"]
    monkeypatch.setattr(
        declarations,
        "validate_release_package_artifact",
        lambda *_: SimpleNamespace(release_identity=lambda: dict(verified)),
    )
    assert release_declaration_identity(manifest, tmp_path / "wheel.whl", lock) == {
        **_RELEASE,
        "dependency_lock_sha256": sha256_file(lock),
    }
    lock.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(PaperRowDeclarationError, match=r"is not the release's uv\.lock"):
        release_declaration_identity(manifest, tmp_path / "wheel.whl", lock)


def test_the_command_requires_a_comparison_decision_for_trialgpt(tmp_path, capsys) -> None:
    parser = _script("write_trialgpt_row_declarations")._parser()
    required = [
        *("--track", "sigir-ct-2016", "--profile", _PROFILE, "--data-dir", str(tmp_path)),
        *("--release-manifest", str(tmp_path), "--package-artifact", str(tmp_path)),
        *("--dependency-lock", str(tmp_path), "--declared-by", "A. Researcher"),
        *("--trialgpt-run-id", "row-1", "--trialgpt2-run-id", "row-2"),
        *("--output-dir", str(tmp_path)),
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(required)
    assert "--trialgpt-comparison" in capsys.readouterr().err
    assert parser.parse_args([*required, "--no-trialgpt-comparisons"]).trialgpt_comparison is None
    chosen = parser.parse_args(
        [*required, "--trialgpt-comparison", "a", "--trialgpt-comparison", "b"]
    )
    assert chosen.trialgpt_comparison == ["a", "b"]


def test_a_derived_stratum_differing_in_one_identity_names_only_it() -> None:
    pinned = PAPER_ROW_STRATA[0]
    moved = replace(pinned, task_input_id="sha256:" + "f" * 64)
    with pytest.raises(PaperRowDeclarationError) as error:
        declarations.require_paper_row_stratum(moved)
    assert str(error.value).count("the package has") == 1
    declarations.require_paper_row_stratum(pinned)
