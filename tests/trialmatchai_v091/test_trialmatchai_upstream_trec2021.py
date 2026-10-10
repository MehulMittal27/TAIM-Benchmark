# ruff: noqa: S101
"""Synthetic conformance for the selectable TrialMatchAI v0.9.1 binding."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import types
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from taim import executables
from taim.pipeline_extensions import pipeline_extensions
from taim.pipelines.trialmatchai.corpus import normalize_ranked_trials
from taim.pipelines.trialmatchai_v091 import release as v091_release
from taim.pipelines.trialmatchai_v091 import stage_models, store_manifest
from taim.pipelines.trialmatchai_v091.extension import EXTENSION
from taim.pipelines.trialmatchai_v091.release import (
    TRIALMATCHAI_UPSTREAM_CONCEPT_RECEIPT,
    TRIALMATCHAI_UPSTREAM_CONCEPT_RECEIPT_SHA256,
    stage_trialmatchai_upstream_models,
    verify_trialmatchai_upstream_concept_store,
)
from taim.pipelines.trialmatchai_v091.system import (
    TRIALMATCHAI_UPSTREAM_COMMIT,
    TRIALMATCHAI_UPSTREAM_CONFIG_SHA256,
    TRIALMATCHAI_UPSTREAM_MODEL_PINS,
    TRIALMATCHAI_UPSTREAM_REPOSITORY,
    trialmatchai_upstream_config_bytes,
    trialmatchai_upstream_runtime_contract,
)
from taim.pipelines.trialmatchai_v091.trec2021 import (
    TRIALMATCHAI_UPSTREAM_TREC_2021_SYSTEM_ID,
    TrialMatchAIUpstreamTREC2021System,
)
from taim.release_cli import build_parser


def test_upstream_trec2021_binding_config_and_normalization() -> None:
    system = TRIALMATCHAI_UPSTREAM_TREC_2021_SYSTEM_ID
    assert set(EXTENSION.system_tasks) == {system}
    assert TrialMatchAIUpstreamTREC2021System().system_id == system
    payload = trialmatchai_upstream_config_bytes()
    assert f"sha256:{hashlib.sha256(payload).hexdigest()}" == TRIALMATCHAI_UPSTREAM_CONFIG_SHA256
    assert json.loads(payload)["search_backend"]["backend"] == "lancedb"
    assert trialmatchai_upstream_runtime_contract()["environment"] == {
        "HF_HUB_OFFLINE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    result = normalize_ranked_trials(
        {
            "RankedTrials": [
                {"TrialID": "trial-b", "Score": 0.5},
                {"TrialID": "trial-a", "Score": 0.5},
            ],
            "Run": {"schema_version": 1, "mode": "eligibility_assessment"},
        },
        run_id="synthetic",
        topic_id="patient",
        system_id=system,
    )
    assert [(row.trial_id, row.rank, row.system_id) for row in result.candidates] == [
        ("trial-a", 1, system),
        ("trial-b", 2, system),
    ]


def _trialmatchai_commands(extensions) -> dict[str, argparse.ArgumentParser]:
    def subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
        return next(
            action.choices
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )

    patient_to_trial = subcommands(build_parser(extensions))["patient-to-trial"]
    return subcommands(subcommands(patient_to_trial)["trialmatchai"])


def test_v091_adds_the_shared_corpus_commands_and_its_own() -> None:
    assert list(_trialmatchai_commands((EXTENSION,))) == [
        "filter-corpus",
        "adopt-corpus",
        "stage-models",
        "verify-concept-store",
    ]
    # Families add to one group, whichever adds first.
    installed = pipeline_extensions()
    assert set(_trialmatchai_commands(installed)) == set(
        _trialmatchai_commands(tuple(reversed(installed)))
    )


def test_the_installed_preflight_never_offers_v091() -> None:
    """The preflight checks the fork's frozen method, which a v0.9.1 checkout can never pass."""

    preflight = _trialmatchai_commands(pipeline_extensions()).get("preflight")
    if preflight is None:
        return
    choices = next(action.choices for action in preflight._actions if action.dest == "system")
    assert TRIALMATCHAI_UPSTREAM_TREC_2021_SYSTEM_ID not in choices


def test_importing_v091_loads_only_the_shared_trialmatchai_modules() -> None:
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, taim.pipelines.trialmatchai_v091.extension; "
            "print(sorted(name for name in sys.modules "
            "if name.startswith('taim.pipelines.trialmatchai')))",
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    assert ast.literal_eval(loaded) == [
        "taim.pipelines.trialmatchai",
        "taim.pipelines.trialmatchai.corpus",
        "taim.pipelines.trialmatchai.execution",
        "taim.pipelines.trialmatchai_v091",
        "taim.pipelines.trialmatchai_v091.extension",
        "taim.pipelines.trialmatchai_v091.release",
        "taim.pipelines.trialmatchai_v091.stage_models",
        "taim.pipelines.trialmatchai_v091.store_manifest",
        "taim.pipelines.trialmatchai_v091.system",
        "taim.pipelines.trialmatchai_v091.trec2021",
    ]


def test_the_shipped_concept_receipt_is_the_one_the_bound_manifest_records() -> None:
    receipt = (
        files("taim")
        .joinpath("pipelines", "trialmatchai_v091", "data", TRIALMATCHAI_UPSTREAM_CONCEPT_RECEIPT)
        .read_bytes()
    )
    assert f"sha256:{hashlib.sha256(receipt).hexdigest()}" == (
        TRIALMATCHAI_UPSTREAM_CONCEPT_RECEIPT_SHA256
    )


def _verified_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "TrialMatchAI"
    root.mkdir()

    def verified(cwd: Path, command: Sequence[str], **expected: object) -> dict[str, object]:
        assert cwd == root and tuple(command) == ()
        assert expected == {
            "expected_commit": TRIALMATCHAI_UPSTREAM_COMMIT,
            "expected_repository": TRIALMATCHAI_UPSTREAM_REPOSITORY,
        }
        return {}

    monkeypatch.setattr(v091_release, "trialmatchai_checkout_configuration", verified)
    monkeypatch.setattr(executables, "resolve_executable", lambda name: f"/bin/{name}")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "taim-env"))
    return root


def test_verify_concept_store_runs_the_writer_in_the_checkout_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _verified_checkout(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []

    def call(argv: list[str], **options: Any) -> int:
        receipt = Path(argv[argv.index("--receipt") + 1])
        calls.append({"argv": argv, "receipt": receipt.read_bytes(), **options})
        return 0

    status = verify_trialmatchai_upstream_concept_store(
        root,
        concept_cache=tmp_path / "cache",
        store=tmp_path / "store",
        output=tmp_path / "manifest.json",
        call=call,
    )

    assert status == 0
    [run] = calls
    argv = run["argv"]
    writer = str(Path(store_manifest.__file__).resolve())
    assert argv[:8] == [
        "/bin/uv",
        "run",
        "--frozen",
        "--project",
        str(root),
        "python",
        "-I",
        writer,
    ]
    assert argv[argv.index("--store") + 1] == str((tmp_path / "store").resolve())
    assert argv[argv.index("--output") + 1] == str((tmp_path / "manifest.json").resolve())
    assert f"sha256:{hashlib.sha256(run['receipt']).hexdigest()}" == (
        TRIALMATCHAI_UPSTREAM_CONCEPT_RECEIPT_SHA256
    )
    assert run["cwd"] == root.resolve() and "VIRTUAL_ENV" not in run["env"]


def test_verify_concept_store_refuses_a_manifest_inside_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _verified_checkout(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside the checkout"):
        verify_trialmatchai_upstream_concept_store(
            root,
            concept_cache=tmp_path / "cache",
            store=tmp_path / "store",
            output=root / "manifest.json",
            call=lambda *_args, **_options: 0,
        )


def test_stage_models_runs_the_stager_with_every_pin_in_the_checkout_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _verified_checkout(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []

    def call(argv: list[str], **options: Any) -> int:
        calls.append({"argv": argv, **options})
        return 0

    assert stage_trialmatchai_upstream_models(root, cache=tmp_path / "hub", call=call) == 0
    [run] = calls
    argv = run["argv"]
    stager = str(Path(stage_models.__file__).resolve())
    assert argv[:8] == [
        "/bin/uv",
        "run",
        "--frozen",
        "--project",
        str(root),
        "python",
        "-I",
        stager,
    ]
    assert argv[8:10] == ["--cache", str((tmp_path / "hub").resolve())]
    assert argv[10:] == [
        token
        for _, repository, revision in TRIALMATCHAI_UPSTREAM_MODEL_PINS
        for token in ("--pin", f"{repository}@{revision}")
    ]
    assert run["cwd"] == root.resolve() and "VIRTUAL_ENV" not in run["env"]


def test_stage_models_refuses_a_cache_inside_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _verified_checkout(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside the checkout"):
        stage_trialmatchai_upstream_models(
            root, cache=root / "hub", call=lambda *_args, **_options: 0
        )


def test_the_stager_points_refs_main_at_each_pinned_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads: list[tuple[str, str]] = []

    def snapshot_download(repository: str, *, revision: str, cache_dir: Path) -> str:
        downloads.append((repository, revision))
        snapshot = cache_dir / f"models--{repository.replace('/', '--')}" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        return str(snapshot)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    cache = tmp_path / "hub"
    pins = ["org/model@" + "a" * 40, "org/adapter@" + "b" * 40]

    assert stage_models.main(["--cache", str(cache), *(f"--pin={pin}" for pin in pins)]) == 0

    assert downloads == [tuple(pin.split("@")) for pin in pins]
    assert (cache / "models--org--model" / "refs" / "main").read_text() == "a" * 40
    assert (cache / "models--org--adapter" / "refs" / "main").read_text() == "b" * 40


class _Column:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def to_pylist(self) -> list[Any]:
        return list(self.values)

    def combine_chunks(self) -> _Column:
        return self

    def flatten(self) -> _Column:
        return _Column([value for row in self.values for value in row])

    def to_numpy(self, zero_copy_only: bool) -> Any:
        import numpy as np

        return np.asarray(self.values, dtype="<f4")


class _Arrow:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.num_rows = len(rows)

    def slice(self, offset: int, length: int) -> _Arrow:
        return _Arrow(self.rows[offset : offset + length])

    def column(self, name: str) -> _Column:
        return _Column([row[name] for row in self.rows])


def _stub_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """A five-row store behind a stub lancedb, its two-source cache, and the receipt."""

    pytest.importorskip("numpy")
    cache, store = tmp_path / "cache", tmp_path / "store"
    (cache / "raw").mkdir(parents=True)
    store.mkdir()
    (store / "data.lance").write_bytes(b"rows")
    sources = []
    rows: list[dict[str, Any]] = []
    for index, (vocabulary, domain, count) in enumerate(
        (("DOID", "Disease", 3), ("HPO", "Phenotype", 2))
    ):
        raw = cache / "raw" / f"{vocabulary}.obo"
        raw.write_bytes(vocabulary.encode())
        dictionary = cache / f"dict_{domain}.txt"
        dictionary.write_text("".join(f"{vocabulary}:{n}\n" for n in range(count)), "utf-8")
        sources.append(
            {
                "source": vocabulary,
                "filename": raw.name,
                "domain_id": domain,
                "vocabulary_id": vocabulary,
                "sha256": "sha256:" + hashlib.sha256(raw.read_bytes()).hexdigest(),
                "converted_dictionary": {
                    "rows": count,
                    "sha256": "sha256:" + hashlib.sha256(dictionary.read_bytes()).hexdigest(),
                },
            }
        )
        for n in range(count):
            row = {name: f"{vocabulary}-{name}-{n}" for name in store_manifest.TEXT_FIELDS}
            row |= {"vocabulary_id": vocabulary, "domain_id": domain}
            row["embedding"] = [float(index + n + 1)] * store_manifest.EMBEDDING_DIMENSION
            rows.append(row)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"sources": sources, "total_rows": len(rows)}), "utf-8")

    class Index:
        name, index_type, columns = "fts_text_idx", "FTS", ["fts_text"]

    class Field:
        def __init__(self, name: str) -> None:
            self.name, self.type = name, "string"

    class Table:
        def __init__(self) -> None:
            self.schema = [Field(name) for name in (*store_manifest.TEXT_FIELDS, "embedding")]

        def list_indices(self) -> list[Index]:
            return [Index()]

        def to_arrow(self) -> _Arrow:
            return _Arrow(rows)

    class Database:
        def table_names(self) -> list[str]:
            return ["concepts"]

        def open_table(self, name: str) -> Table:
            assert name == "concepts"
            return Table()

    monkeypatch.setitem(
        sys.modules, "lancedb", types.SimpleNamespace(connect=lambda _path: Database())
    )
    return receipt, cache, store


def test_the_manifest_is_the_same_however_the_rows_are_sliced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, cache, store = _stub_store(tmp_path, monkeypatch)
    whole = store_manifest.verified_manifest(receipt, cache, store)
    monkeypatch.setattr(store_manifest, "_SLICE_ROWS", 2)
    sliced = store_manifest.verified_manifest(receipt, cache, store)

    assert sliced == whole
    assert whole["verified"] is True and whole["failures"] == []
    assert whole["vocabulary_counts"] == {"DOID": 3, "HPO": 2}
    expected = hashlib.sha256()
    for vocabulary, domain, count in (("DOID", "Disease", 3), ("HPO", "Phenotype", 2)):
        for n in range(count):
            row = {name: f"{vocabulary}-{name}-{n}" for name in store_manifest.TEXT_FIELDS}
            row |= {"vocabulary_id": vocabulary, "domain_id": domain}
            line = json.dumps(
                [row[name] for name in store_manifest.TEXT_FIELDS], separators=(",", ":")
            )
            expected.update(line.encode() + b"\n")
    assert whole["content_identity"]["text_rows_sha256"] == f"sha256:{expected.hexdigest()}"
    assert whole["embedding"]["norm_min"] == pytest.approx(32.0)


def test_a_bound_manifest_at_the_store_root_is_left_out_of_its_own_file_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, cache, store = _stub_store(tmp_path, monkeypatch)
    first = store_manifest.verified_manifest(receipt, cache, store)
    (store / store_manifest.MANIFEST_FILENAME).write_text(json.dumps(first), "utf-8")

    assert store_manifest.verified_manifest(receipt, cache, store) == first
    assert [row[0] for row in first["file_manifest"]] == ["data.lance"]


def test_a_store_that_differs_from_the_receipt_is_written_with_its_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, cache, store = _stub_store(tmp_path, monkeypatch)
    (cache / "raw" / "HPO.obo").write_bytes(b"refetched")

    report = store_manifest.verified_manifest(receipt, cache, store)

    assert report["verified"] is False
    assert [failure.split(":")[0] for failure in report["failures"]] == ["raw HPO.obo"]
