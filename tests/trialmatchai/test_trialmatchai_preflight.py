# ruff: noqa: S101
"""The TrialMatchAI static preflight names each cause that would stop or silently change a run.

Each test builds one environment that is complete except for the single defect it plants, then
runs the real ``preflight`` command: the checkout is a real Git repository, the environment check
calls a stand-in ``uv`` that records its argv, and the probe runs as the checkout's interpreter.
The four failures that motivated the preflight each appear here as a planted defect: a destroyed
environment, one rebuilt with two of its three extras, a card with CUDA available but no kernels
(a V100 against a cu130 build), and a card too small for the frozen memory shares.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import taim.pipelines.trialmatchai.preflight as preflight
from taim.artifacts import sha256_file
from taim.pipelines.trialmatchai import system as adapter
from taim.pipelines.trialmatchai.concept_store import PINNED_OPEN_DICTIONARIES, ROW_DIGEST_RULE
from taim.pipelines.trialmatchai.corpus import write_trialmatchai_corpus_receipt
from taim.pipelines.trialmatchai.preflight import (
    _PROBE,
    TRIALMATCHAI_L4_MIN_PROVEN_GPU_MEMORY_BYTES,
    arch_list_supports,
)
from taim.pipelines.trialmatchai.system import (
    TRIALMATCHAI_CONCEPT_STORE_DIGEST_RULE,
    TrialMatchAIL4Base,
)
from taim.pipelines.trialmatchai.trec2021 import TRIALMATCHAI_L4_SYSTEM_ID
from taim.release_cli import main
from tests.git_helpers import git

# What the fixture config's backends need, derived as build-env derives them: gliner2 entities,
# the hf embedder, and vLLM engines with bitsandbytes quantization, as in the frozen L4 config.
DERIVED_EXTRAS = ["entity", "gpu", "llm"]
DERIVED_MODULES = ["bitsandbytes", "gliner2", "torch", "transformers", "vllm"]
# torch 2.11.0+cu130, the fork's locked build: no Volta cubin.
CU130_ARCH_LIST = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
GIB = 1024**3
_REPOSITORY = "https://example.invalid/TrialMatchAI.git"
PINS = {
    "microsoft/phi-4": "a" * 40,
    "google/gemma-2-2b-it": "b" * 40,
    "BAAI/bge-m3": "c" * 40,
    "fastino/gliner2-base-v1": "d" * 40,
}


_CONCEPT_STORE_MANIFEST = {
    "manifest_version": "2",
    "store": {
        "table": "concepts",
        "row_count": 123,
        "indexes": {"fts": {"column": "fts_text", "type": "FTS"}},
    },
    "sources": {
        "release": "v20260227",
        "files": {
            "CONCEPT.csv": "sha256:" + "a" * 64,
            "CONCEPT_SYNONYM.csv": "sha256:" + "b" * 64,
        },
    },
    "vocabularies": {"SNOMED": 123},
    "embeddings": {
        "mode": "fts-only",
        "skip_embeddings": True,
        "present": False,
        "model_name": "BAAI/bge-m3",
        "revision": None,
        "dimension": None,
        "pooling": None,
        "max_length": None,
        "batch_size": None,
        "use_fp16": False,
        "normalize": True,
    },
    "concept_state": {"state_version": "2", "fingerprint": "sha256:" + "c" * 64},
}


def _write_concept_store(store: Path) -> None:
    """A synthetic store: the manifest, and one file in each group a store identity hashes."""

    for relative, content in (
        ("concept-store-manifest.json", json.dumps(_CONCEPT_STORE_MANIFEST)),
        ("concepts.lance/data/rows.lance", "rows and vectors"),
        ("concepts.lance/_indices/ivf-pq/auxiliary.idx", "trained index"),
        ("concepts.lance/_versions/1.manifest", "version 1"),
    ):
        path = store / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _concept_store_identity(store: Path) -> dict[str, object]:
    """The store's identity record, computed here from the digest rule, not by the adapter."""

    groups = {
        "content.table_data": hashlib.sha256(),
        "content.indexes": hashlib.sha256(),
        "content.table_metadata": hashlib.sha256(),
    }
    for path in sorted(item for item in store.rglob("*") if item.is_file()):
        name = path.relative_to(store).as_posix()
        if name.startswith("concepts.lance/data/"):
            group = "content.table_data"
        elif name.startswith("concepts.lance/_indices/"):
            group = "content.indexes"
        else:
            group = "content.table_metadata"
        line = json.dumps([name, sha256_file(path)], separators=(",", ":")) + "\n"
        groups[group].update(line.encode("utf-8"))
    return {
        "schema_version": "1.0",
        "store": "synthetic-store",
        "digest_rule": TRIALMATCHAI_CONCEPT_STORE_DIGEST_RULE,
        "identity": {
            "manifest_version": _CONCEPT_STORE_MANIFEST["manifest_version"],
            "store.row_count": _CONCEPT_STORE_MANIFEST["store"]["row_count"],
            "embeddings.mode": _CONCEPT_STORE_MANIFEST["embeddings"]["mode"],
            "concept_state.vector_stream_digest": None,
            **{field: f"sha256:{digest.hexdigest()}" for field, digest in groups.items()},
        },
    }


def _write_concept_store_manifest(db_path, *, mode="fts-only"):
    db_path.mkdir()
    skipped = mode == "fts-only"
    payload = {
        "manifest_version": "2",
        "store": {
            "table": "concepts",
            "row_count": 123,
            "indexes": {"fts": {"column": "fts_text", "type": "FTS"}},
        },
        "sources": {
            "release": "v20260227",
            "files": {
                "CONCEPT.csv": "sha256:" + "a" * 64,
                "CONCEPT_SYNONYM.csv": "sha256:" + "b" * 64,
            },
        },
        "vocabularies": {"SNOMED": 123},
        "embeddings": {
            "mode": mode,
            "skip_embeddings": skipped,
            "present": not skipped,
            "model_name": "BAAI/bge-m3",
            "revision": None if skipped else "model-revision",
            "dimension": None if skipped else 1024,
            "pooling": None if skipped else "mean",
            "max_length": None if skipped else 512,
            "batch_size": None if skipped else 32,
            "use_fp16": False,
            "normalize": True,
        },
        "concept_state": {
            "state_version": "2" if skipped else "3",
            "fingerprint": "sha256:" + "c" * 64,
            **({} if skipped else {"vector_stream_digest": "sha256:" + "d" * 64}),
        },
    }
    if mode == "vector":
        payload["store"]["indexes"].update(
            {
                "vector": {
                    "column": "embedding",
                    "type": "IVF_PQ",
                    "metric": "cosine",
                    "num_partitions": 1024,
                    "num_sub_vectors": 128,
                    "num_bits": 8,
                    "max_iterations": 50,
                    "sample_rate": 256,
                    "train": True,
                },
                "filters": [
                    {"column": "vocabulary_id", "type": "BTREE"},
                    {"column": "domain_id", "type": "BTREE"},
                ],
            }
        )
        payload["embeddings"]["metric"] = "cosine"
    (db_path / "concept-store-manifest.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("card", "capability", "arch_list", "supported"),
    [
        ("V100", (7, 0), CU130_ARCH_LIST, False),
        ("A100", (8, 0), CU130_ARCH_LIST, True),
        ("A40", (8, 6), CU130_ARCH_LIST, True),
        ("L40S runs the sm_86 cubin", (8, 9), CU130_ARCH_LIST, True),
        ("H200", (9, 0), CU130_ARCH_LIST, True),
        ("sm_86 cubin does not run on an 8.0 card", (8, 0), ["sm_86"], False),
        ("architecture-specific build, own card", (9, 0), ["sm_90a"], True),
        ("architecture-specific build, other card", (10, 0), ["sm_90a"], False),
        ("PTX only is not proof", (7, 0), ["compute_70"], False),
        ("no CUDA build", (8, 0), [], False),
    ],
)
def test_available_is_not_supported(
    card: str, capability: tuple[int, int], arch_list: list[str], supported: bool
) -> None:
    assert arch_list_supports(capability, arch_list) is supported, card


def _probe_result(**gpu: object) -> dict[str, object]:
    return {
        "platform": "linux",
        "python": "3.12.3",
        "modules": {module: {"version": "1.0"} for module in DERIVED_MODULES},
        "gpu": {
            "torch": "2.11.0+cu130",
            "cuda": "13.0",
            "available": True,
            "arch_list": CU130_ARCH_LIST,
            "device_count": 1,
            "name": "NVIDIA A40",
            "capability": [8, 6],
            "free_bytes": 44 * GIB,
            "total_bytes": int(44.4 * GIB),
            **gpu,
        },
        "model_cache": "/cache/hub",
        "models": {repo: {"problems": [], "main": revision} for repo, revision in PINS.items()},
    }


class Environment:
    """A complete L4 environment in ``root``; each test breaks one part of it."""

    def __init__(
        self,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
        config: dict[str, object] | None = None,
    ) -> None:
        self.root = root
        self.checkout = root / "TrialMatchAI"
        config_path = self.checkout / adapter.TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps(config or self.config()), encoding="utf-8")
        weights = []
        for option in ("cot", "reranker"):
            path = (
                self.checkout / "models" / option / adapter.TRIALMATCHAI_L4_ADAPTER_WEIGHT_FILENAME
            )
            path.parent.mkdir(parents=True)
            path.write_bytes(f"{option} adapter".encode())
            weights.append(
                (
                    f"{option}_adapter_path",
                    "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        git(self.checkout, "init")
        git(self.checkout, "config", "user.email", "taim-test@example.invalid")
        git(self.checkout, "config", "user.name", "TAIM test")
        git(self.checkout, "remote", "add", "origin", _REPOSITORY)
        git(self.checkout, "add", ".")
        git(self.checkout, "commit", "-m", "fixture")
        monkeypatch.setattr(
            preflight, "TRIALMATCHAI_L4_COMMIT", git(self.checkout, "rev-parse", "HEAD")
        )
        monkeypatch.setattr(preflight, "TRIALMATCHAI_L4_REPOSITORY", _REPOSITORY)
        config_sha256 = "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest()
        monkeypatch.setattr(TrialMatchAIL4Base, "expected_config_sha256", config_sha256)
        monkeypatch.setattr(adapter, "TRIALMATCHAI_L4_CONFIG_SHA256", config_sha256)
        monkeypatch.setattr(adapter, "TRIALMATCHAI_L4_ADAPTER_SHA256", tuple(weights))

        self.corpus = root / "corpus"
        for role, name in (
            ("processed_trials", "NCT00000001.json"),
            ("processed_criteria", "NCT00000001/c1.json"),
        ):
            path = self.corpus / role / name
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")
        for table in ("trials", "criteria"):
            path = self.corpus / "search" / f"{table}.lance" / "data.lance"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"lance")
        write_trialmatchai_corpus_receipt(self.corpus, workers=1)
        self.store = root / "concepts"
        _write_concept_store(self.store)
        monkeypatch.setattr(
            TrialMatchAIL4Base, "expected_concept_store", _concept_store_identity(self.store)
        )

        # The environment lives outside the checkout, as UV_PROJECT_ENVIRONMENT allows, so the
        # checkout stays clean. Its interpreter is a stand-in that prints the planted probe result.
        self.probe_path = root / "probe.json"
        self.write_probe(_probe_result())
        interpreter = root / "env" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text(f"#!/bin/sh\ncat '{self.probe_path}'\n", encoding="utf-8")
        interpreter.chmod(0o755)
        self.uv_argv = root / "uv-argv"
        self.uv_outdated = root / "uv-outdated"
        uv = root / "bin" / "uv"
        uv.parent.mkdir()
        uv.write_text(
            "#!/bin/sh\n"
            f"echo \"$@\" > '{self.uv_argv}'\n"
            f"if [ -f '{self.uv_outdated}' ]; then cat '{self.uv_outdated}' >&2; exit 1; fi\n"
            "echo 'Would make no changes' >&2\n",
            encoding="utf-8",
        )
        uv.chmod(0o755)
        monkeypatch.setattr(preflight, "resolve_executable", lambda name: str(uv))
        for name in adapter.TRIALMATCHAI_L4_FORBIDDEN_ENVIRONMENT:
            monkeypatch.delenv(name, raising=False)
        for name in adapter.TRIALMATCHAI_CONFIG_ENV_OVERRIDES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        monkeypatch.setenv("TRIALMATCHAI_CONCEPT_DB_PATH", str(self.store))
        monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(root / "env"))

    @staticmethod
    def config() -> dict[str, object]:
        return {
            "model": {
                "base_model": "microsoft/phi-4",
                "base_model_revision": PINS["microsoft/phi-4"],
                "cot_adapter_path": "models/cot",
                "reranker_model_path": "google/gemma-2-2b-it",
                "reranker_model_revision": PINS["google/gemma-2-2b-it"],
                "reranker_adapter_path": "models/reranker",
            },
            "embedder": {"model_name": "BAAI/bge-m3", "revision": PINS["BAAI/bge-m3"]},
            "entity_extraction": {
                "backend": "gliner2",
                "model_name": "fastino/gliner2-base-v1",
                "model_revision": PINS["fastino/gliner2-base-v1"],
            },
            "global": {"device": 0},
            "search_backend": {"trials_table": "trials", "criteria_table": "criteria"},
            "rag": {"backend": "vllm"},
            "LLM_reranker": {"backend": "vllm", "gpu_memory_utilization": 0.22},
            "vllm": {"gpu_memory_utilization": 0.6, "quantization": "bitsandbytes"},
        }

    def write_probe(self, result: dict[str, object]) -> None:
        self.probe_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    def run_preflight(self, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, str], str]:
        code = main(
            [
                "patient-to-trial",
                "trialmatchai",
                "preflight",
                "--system",
                TRIALMATCHAI_L4_SYSTEM_ID,
                "--trialmatchai-cwd",
                str(self.checkout),
                "--trialmatchai-corpus-dir",
                str(self.corpus),
            ]
        )
        lines = capsys.readouterr().out.strip().splitlines()
        checks = {}
        for line in lines[:-1]:
            verdict, _, rest = line.partition(" ")
            name, _, detail = rest.partition(": ")
            checks[name] = f"{verdict} {detail}"
        return code, checks, lines[-1]


@pytest.fixture(name="environment")
def _environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Environment:
    return Environment(tmp_path, monkeypatch)


def _failed(checks: dict[str, str]) -> dict[str, str]:
    return {name: result for name, result in checks.items() if result.startswith("FAIL")}


def test_a_complete_environment_passes_only_the_static_half(
    environment: Environment, capsys: pytest.CaptureFixture[str]
) -> None:
    code, checks, last = environment.run_preflight(capsys)

    assert not _failed(checks)
    assert "is the qualified store synthetic-store" in checks["run-inputs"]
    assert code == 0
    assert last.startswith("preflight: 0 FAIL of 8 static checks")
    assert "does not show that the engines start" in last
    # The lock is compared, never synced: frozen, offline, with exactly the derived extras.
    argv = environment.uv_argv.read_text(encoding="utf-8").split()
    assert argv[:3] == ["sync", "--frozen", "--offline"] and argv[-1] == "--check"
    assert argv[4:-1:2] == DERIVED_EXTRAS
    assert checks["modules"] == "PASS " + ", ".join(f"{m} 1.0" for m in DERIVED_MODULES)
    assert "NVIDIA A40 (sm_86) is supported" in checks["gpu-supported"]
    assert git(environment.checkout, "status", "--porcelain", "--untracked-files=all") == ""


def test_a_store_that_is_not_the_qualified_one_fails_the_runs_own_check(
    environment: Environment, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = environment.root / "unqualified-concepts"
    _write_concept_store_manifest(store, mode="vector")
    monkeypatch.setenv("TRIALMATCHAI_CONCEPT_DB_PATH", str(store))
    monkeypatch.setattr(
        TrialMatchAIL4Base,
        "expected_concept_store",
        adapter._load_concept_store_identity(adapter.TRIALMATCHAI_L4_CONCEPT_STORE_LOCK),
    )
    code, checks, last = environment.run_preflight(capsys)

    # A well-formed manifest is not the qualified store: the run's own check names the first
    # field that differs, and stops there as the run does, so nothing reads its config after it.
    assert "is not the qualified store candidate-21352305" in checks["run-inputs"]
    assert "store.row_count expected 3899744, found 123" in checks["run-inputs"]
    not_checked = "FAIL not checked: the run's inputs did not verify"
    for name in ("search-database", "environment-lock", "modules", "gpu-memory", "model-cache"):
        assert checks[name] == not_checked, name
    assert checks["prepared-corpus"].startswith("PASS")
    assert checks["gpu-supported"].startswith("PASS")
    assert not environment.uv_argv.exists()
    assert last == "preflight: 6 FAIL of 8 - not runnable"
    assert code == 1


def test_the_run_inputs_refuse_in_the_runs_order(
    environment: Environment, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two defects at once: a dirty checkout and a forbidden retrieval cap. The run checks its
    # runtime contract before the checkout, so the preflight names the cap, as the run would.
    (environment.checkout / "stray.txt").write_text("local change", encoding="utf-8")
    monkeypatch.setenv("TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS", "15")
    _code, checks, _last = environment.run_preflight(capsys)

    assert checks["run-inputs"] == (
        f"FAIL {TRIALMATCHAI_L4_SYSTEM_ID} forbids TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS; found "
        "'15'. It caps retrieval depth below the verified config and must be unset."
    )
    monkeypatch.delenv("TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS")
    _code, checks, _last = environment.run_preflight(capsys)

    assert checks["run-inputs"] == (
        f"FAIL TrialMatchAI checkout has local changes: {environment.checkout}"
    )


def test_an_outside_build_passes_and_is_named_as_one(
    environment: Environment,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild: the qualified rows and pinned dictionaries, an index trained again."""

    manifest_path = environment.store / "concept-store-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"]["files"].update(
        {pin.dictionary: pin.dictionary_sha256 for pin in PINNED_OPEN_DICTIONARIES}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows = {"count": 123, "digest": "sha256:" + "e" * 64}
    record = {
        **_concept_store_identity(environment.store),
        "outside_build": {
            "schema_version": "1.0",
            "store": "synthetic-store",
            "row_digest_rule": ROW_DIGEST_RULE,
            "rows": rows,
            "recipe": {
                "manifest_version": manifest["manifest_version"],
                "embeddings": manifest["embeddings"],
                "indexes": manifest["store"]["indexes"],
            },
        },
    }
    (environment.store / "concepts.lance" / "_indices" / "ivf-pq" / "auxiliary.idx").write_text(
        "retrained index", encoding="utf-8"
    )
    monkeypatch.setattr(TrialMatchAIL4Base, "expected_concept_store", record)
    monkeypatch.setattr(
        adapter, "_checkout_row_digest", lambda cwd, environment: lambda db, table: rows
    )

    code, checks, _last = environment.run_preflight(capsys)

    assert not _failed(checks)
    assert "is an outside build of the qualified store synthetic-store" in checks["run-inputs"]
    assert "its vectors and indexes are its own" in checks["run-inputs"]
    assert code == 0


@pytest.mark.parametrize(
    ("outdated", "shown"),
    [
        # This week's `uv sync --frozen`: every extra's packages gone.
        (
            " + gliner2==1.3.1\n + torch==2.11.0\n + vllm==0.23.0\n",
            "remove 0 and install 3 packages (install: gliner2==1.3.1, torch==2.11.0, vllm",
        ),
        # This week's rebuild with --extra llm --extra gpu: the entity extra gone.
        (
            " + gliner2==1.3.1\n + gliner==0.2.24\n",
            "remove 0 and install 2 packages (install: gliner2==1.3.1, gliner==0.2.24)",
        ),
    ],
)
def test_an_environment_that_is_not_its_lock_is_refused_with_the_rebuild_command(
    environment: Environment, capsys: pytest.CaptureFixture[str], outdated: str, shown: str
) -> None:
    environment.uv_outdated.write_text(outdated + "The environment is outdated\n", encoding="utf-8")
    code, checks, last = environment.run_preflight(capsys)

    assert "environment-lock" in _failed(checks)
    assert shown in checks["environment-lock"]
    assert checks["environment-lock"].endswith(
        "Rebuild it with: trial-benchmark patient-to-trial trialmatchai build-env --checkout "
        f"{environment.checkout} --receipt <file outside the checkout>"
    )
    assert code == 1 and last.endswith("not runnable")


def test_the_extras_and_modules_follow_the_config_not_a_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = Environment.config()
    config["entity_extraction"] = {**config["entity_extraction"], "backend": "regex"}  # type: ignore[dict-item]
    environment = Environment(tmp_path, monkeypatch, config)
    _code, checks, _last = environment.run_preflight(capsys)

    # A config that extracts entities by regex needs no entity extra, so the lock is compared
    # without it and gliner2 is not required to import.
    argv = environment.uv_argv.read_text(encoding="utf-8").split()
    assert argv[4:-1:2] == ["gpu", "llm"]
    assert checks["environment-lock"].startswith("PASS")
    assert "gliner2" not in checks["modules"]


def test_a_backend_with_no_known_extra_is_refused_before_uv_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = Environment.config()
    config["embedder"] = {**config["embedder"], "type": "onnx"}  # type: ignore[dict-item]
    environment = Environment(tmp_path, monkeypatch, config)
    code, checks, _last = environment.run_preflight(capsys)

    refusal = "embedder.type='onnx' maps to no known extra; refusing to guess the environment"
    assert refusal in checks["environment-lock"]
    assert refusal in checks["modules"]
    assert not environment.uv_argv.exists()
    assert code == 1


def test_a_v100_has_cuda_available_and_is_refused(
    environment: Environment, capsys: pytest.CaptureFixture[str]
) -> None:
    environment.write_probe(
        _probe_result(name="Tesla V100-SXM2-32GB", capability=[7, 0], total_bytes=32 * GIB)
    )
    _code, checks, _last = environment.run_preflight(capsys)

    assert checks["gpu-supported"] == (
        "FAIL Tesla V100-SXM2-32GB (sm_70) has CUDA available but is not supported: "
        "torch 2.11.0+cu130 (CUDA 13.0) carries kernels for " + ", ".join(CU130_ARCH_LIST)
    )


@pytest.mark.parametrize(
    ("total", "free", "shown"),
    [
        (int(22.5 * GIB), int(22 * GIB), "the card has 22.5 GiB, below the 39.0 GiB"),
        (
            int(44.4 * GIB),
            30 * GIB,
            "30.0 GiB of 44.4 GiB is free, but the two vLLM engines claim 0.82",
        ),
    ],
)
def test_a_card_without_room_for_the_frozen_memory_shares_is_refused(
    environment: Environment,
    capsys: pytest.CaptureFixture[str],
    total: int,
    free: int,
    shown: str,
) -> None:
    environment.write_probe(_probe_result(total_bytes=total, free_bytes=free))
    _code, checks, _last = environment.run_preflight(capsys)

    assert shown in checks["gpu-memory"]
    assert checks["gpu-memory"].startswith("FAIL")
    assert TRIALMATCHAI_L4_MIN_PROVEN_GPU_MEMORY_BYTES == 39 * GIB


def test_a_module_the_lock_skips_on_this_platform_is_refused(
    environment: Environment, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _probe_result()
    result["platform"] = "darwin"
    result["modules"]["vllm"] = {"error": "ModuleNotFoundError: No module named 'vllm'"}  # type: ignore[index]
    environment.write_probe(result)
    _code, checks, _last = environment.run_preflight(capsys)

    assert checks["modules"] == (
        "FAIL vllm (extra gpu): ModuleNotFoundError: No module named 'vllm'. This host is darwin; "
        "the lock installs vllm only on Linux"
    )


def test_an_unpinned_gliner2_resolution_and_a_missing_model_are_refused(
    environment: Environment, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _probe_result()
    models = result["models"]
    models["fastino/gliner2-base-v1"]["main"] = "e" * 40  # type: ignore[index]
    models["google/gemma-2-2b-it"] = {"error": "LocalEntryNotFoundError: not cached"}  # type: ignore[index]
    environment.write_probe(result)
    _code, checks, _last = environment.run_preflight(capsys)

    assert "gliner2 loads refs/main, which names " + "e" * 40 in checks["model-cache"]
    assert (
        "reranker base google/gemma-2-2b-it@" + "b" * 40 + " does not resolve offline"
        in (checks["model-cache"])
    )


def test_a_probe_without_a_result_fails_every_check_that_needs_it(
    environment: Environment, capsys: pytest.CaptureFixture[str]
) -> None:
    environment.probe_path.write_text("", encoding="utf-8")
    _code, checks, last = environment.run_preflight(capsys)

    for name in ("modules", "gpu-supported", "gpu-memory", "model-cache"):
        assert checks[name].startswith("FAIL not checked: the probe in "), name
    assert last == "preflight: 4 FAIL of 8 - not runnable"


def test_the_probe_reads_the_model_cache_the_way_the_run_resolves_it(tmp_path: Path) -> None:
    """The probe's own code, run against a stand-in huggingface_hub over a planted cache."""

    stub = tmp_path / "stub" / "huggingface_hub"
    stub.mkdir(parents=True)
    (stub / "constants.py").write_text(
        "import os\nHF_HUB_CACHE = os.environ['HF_HUB_CACHE']\n", encoding="utf-8"
    )
    (stub / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from . import constants\n"
        "def snapshot_download(repo, revision, local_files_only):\n"
        "    assert local_files_only\n"
        "    folder = Path(constants.HF_HUB_CACHE) / ('models--' + repo.replace('/', '--'))\n"
        "    folder = folder / 'snapshots' / revision\n"
        "    if not folder.is_dir():\n"
        "        raise FileNotFoundError(repo)\n"
        "    return str(folder)\n",
        encoding="utf-8",
    )
    cache = tmp_path / "hub"

    def snapshot(repo: str, revision: str, main: str | None) -> Path:
        root = cache / ("models--" + repo.replace("/", "--"))
        folder = root / "snapshots" / revision
        folder.mkdir(parents=True)
        if main is not None:
            (root / "refs").mkdir()
            (root / "refs" / "main").write_text(main, encoding="utf-8")
        return folder

    snapshot("org/complete", "1" * 40, "1" * 40).joinpath("model.safetensors").write_bytes(b"w")
    sharded = snapshot("org/sharded", "2" * 40, None)
    (sharded / "model-1.safetensors").write_bytes(b"w")
    (sharded / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}),
        encoding="utf-8",
    )
    dangling = snapshot("org/dangling", "3" * 40, "4" * 40)
    (dangling / "pytorch_model.bin").symlink_to(tmp_path / "missing-blob")
    snapshot("org/config-only", "5" * 40, None).joinpath("config.json").write_text("{}")

    spec = {
        "modules": ["json", "taim_no_such_module"],
        "device": 0,
        "models": [
            ["org/complete", "1" * 40],
            ["org/sharded", "2" * 40],
            ["org/dangling", "3" * 40],
            ["org/config-only", "5" * 40],
            ["org/absent", "6" * 40],
        ],
    }
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(spec)],
        capture_output=True,
        text=True,
        check=True,
        env={"HF_HUB_CACHE": str(cache), "PYTHONPATH": str(stub.parent)},
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["modules"]["json"] == {"version": json.__version__}
    assert result["modules"]["taim_no_such_module"]["error"].startswith("ModuleNotFoundError")
    assert "gpu" in result
    models = result["models"]
    assert models["org/complete"] == {"problems": [], "main": "1" * 40}
    assert models["org/sharded"] == {
        "problems": ["model-2.safetensors, named by model.safetensors.index.json, is missing"],
        "main": None,
    }
    assert models["org/dangling"]["problems"] == [
        "pytorch_model.bin is a dangling link",
        "it holds no weight file",
    ]
    assert models["org/config-only"]["problems"] == ["it holds no weight file"]
    assert models["org/absent"]["error"] == "FileNotFoundError: org/absent"


def test_a_system_this_installation_does_not_ship_is_refused(
    environment: Environment, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["patient-to-trial", "trialmatchai", "preflight", "--system", "not-a-system"]
    arguments += ["--trialmatchai-cwd", str(environment.checkout)]
    arguments += ["--trialmatchai-corpus-dir", str(environment.corpus)]
    with pytest.raises(SystemExit) as refused:
        main(arguments)

    assert refused.value.code == 2
    error = capsys.readouterr().err
    assert "argument --system: invalid choice: 'not-a-system'" in error
    assert TRIALMATCHAI_L4_SYSTEM_ID in error
