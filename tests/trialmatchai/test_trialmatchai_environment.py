# ruff: noqa: S101
"""``trialmatchai build-env`` derives the extras from the frozen config, and refuses to guess.

On Delta the TrialMatchAI environment was rebuilt with two of the three extras its configured
backends need (``llm`` and ``gpu``, as upstream's own vLLM preflight message advises), the sync
exited 0, and the row failed on gliner2. Separately a bare ``uv sync --frozen`` removed every extra.
The command therefore takes no extras from an operator: it reads them from the config's backends,
refuses a backend it cannot map, refuses before any change when the lock does not install a
derived package on this host (a Linux-only extra "installs" on a Mac with exit 0), states what the
exact sync removes, and writes its receipt only after the environment is checked against the lock
and every derived module imports.

The build tests drive the real function with a recording stand-in for ``uv`` and the checkout's
interpreter, so no environment is touched. The drift test reads the pinned checkout, when present.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from collections.abc import Sequence
from pathlib import Path

import pytest

import taim.pipelines.trialmatchai.release as trialmatchai_release
from taim.pipelines.trialmatchai.release import (
    TRIALMATCHAI_BACKEND_REQUIREMENTS,
    build_trialmatchai_environment,
    derive_trialmatchai_requirements,
)
from taim.pipelines.trialmatchai.system import (
    TRIALMATCHAI_L4_COMMIT,
    TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH,
    TRIALMATCHAI_L4_REPOSITORY,
)
from taim.release_cli import main


def _pinned_checkout() -> Path:
    """The checkout under the repository's ``external/`` whose HEAD is the pinned commit."""

    for marker in sorted((Path(__file__).resolve().parents[3] / "external").glob("*/.git")):
        head = subprocess.run(
            ["git", "-C", str(marker.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        ).stdout.strip()
        if head == TRIALMATCHAI_L4_COMMIT:
            return marker.parent
    pytest.skip("frozen TrialMatchAI checkout is not present")


# The backends of the frozen L4 config (taim_l4_cuda.json at the pinned commit), and nothing else.
L4_BACKENDS: dict[str, object] = {
    "entity_extraction": {"backend": "gliner2"},
    "embedder": {"model_name": "BAAI/bge-m3"},
    "search_backend": {"backend": "lancedb"},
    "use_cot_reasoning": True,
    "rag": {"enabled": True, "backend": "vllm"},
    "LLM_reranker": {"enabled": True, "backend": "vllm", "quantization": "bitsandbytes"},
    "vllm": {"quantization": "bitsandbytes"},
    "query_expansion": {"enabled": True, "backend": None},
}
LINUX_ENVIRONMENT = (
    *("accelerate", "bitsandbytes", "gliner", "gliner2", "lancedb", "numpy"),
    *("peft", "torch", "transformers", "vllm"),
)


def _extras(config: dict[str, object]) -> list[str]:
    return sorted(
        {extra for _s, _v, rows in derive_trialmatchai_requirements(config) for extra, *_ in rows}
    )


def _with(**blocks: object) -> dict[str, object]:
    return {**json.loads(json.dumps(L4_BACKENDS)), **blocks}


# --- derivation --------------------------------------------------------------------------------


def test_frozen_config_derives_all_three_extras() -> None:
    derived = {
        (setting, value): rows
        for setting, value, rows in derive_trialmatchai_requirements(L4_BACKENDS)
    }

    assert _extras(L4_BACKENDS) == ["entity", "gpu", "llm"]
    # The extra the hand-listed rebuild dropped is derived from the recognizer, not remembered.
    assert {extra for extra, *_ in derived[("entity_extraction.backend", "gliner2")]} == {"entity"}
    assert set(derived) == {
        ("entity_extraction.backend", "gliner2"),
        ("embedder.type", "hf"),
        ("search_backend.type", "lancedb"),
        ("rag.backend", "vllm"),
        ("vllm.quantization", "bitsandbytes"),
        ("LLM_reranker.backend", "vllm"),
        ("LLM_reranker.quantization", "bitsandbytes"),
        ("query_expansion.backend", "vllm"),
    }


def test_frozen_config_file_derives_all_three_extras() -> None:
    config = _pinned_checkout() / TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH
    if not config.is_file():
        pytest.skip("frozen TrialMatchAI checkout is not present")
    assert _extras(json.loads(config.read_text(encoding="utf-8"))) == ["entity", "gpu", "llm"]


def test_absent_sections_take_upstream_defaults() -> None:
    assert _extras({}) == ["entity", "gpu", "llm"]


def test_disabled_stages_need_nothing() -> None:
    config = _with(
        entity_extraction={"backend": "disabled"},
        embedder={"type": "hashing"},
        use_cot_reasoning=False,
        LLM_reranker={"enabled": False, "backend": "vllm"},
        query_expansion={"enabled": False},
    )
    assert _extras(config) == []


def test_transformers_serving_needs_llm_only() -> None:
    config = _with(
        rag={"backend": "transformers"},
        LLM_reranker={"backend": "transformers", "quantization": "bitsandbytes"},
    )
    # Quantization is a vLLM engine setting; the transformers path never loads bitsandbytes.
    assert _extras(config) == ["entity", "llm"]


def test_reranker_inherits_the_shared_vllm_quantization() -> None:
    config = _with(
        use_cot_reasoning=False,
        query_expansion={"enabled": False},
        LLM_reranker={"backend": "vllm"},
        vllm={"quantization": "bitsandbytes"},
    )
    derived = {
        (setting, value) for setting, value, _rows in derive_trialmatchai_requirements(config)
    }
    assert ("LLM_reranker.quantization", "bitsandbytes") in derived


@pytest.mark.parametrize(
    ("blocks", "named"),
    [
        ({"rag": {"backend": "sglang"}}, "rag.backend='sglang'"),
        ({"entity_extraction": {"backend": "spacy"}}, "entity_extraction.backend='spacy'"),
        ({"embedder": {"type": "sentence-transformers"}}, "embedder.type='sentence-transformers'"),
        ({"search_backend": {"backend": "qdrant"}}, "search_backend.type='qdrant'"),
        ({"LLM_reranker": {"backend": "mlx_vlm"}}, "LLM_reranker.backend='mlx_vlm'"),
        ({"vllm": {"quantization": "awq"}}, "vllm.quantization='awq'"),
    ],
)
def test_unknown_backend_is_refused_not_guessed(blocks: dict[str, object], named: str) -> None:
    with pytest.raises(ValueError, match=re.escape(named) + ".*maps to no known extra"):
        derive_trialmatchai_requirements(_with(**blocks))


# --- the build, with a recording stand-in for uv ----------------------------------------------


class _Host:
    """Answers the build's commands the way uv and the checkout's interpreter would."""

    def __init__(
        self,
        installable: Sequence[str] = LINUX_ENVIRONMENT,
        *,
        removes: Sequence[str] = (),
        in_sync: bool = True,
        import_failures: dict[str, str] | None = None,
        sync_code: int = 0,
    ) -> None:
        self.installable = installable
        self.removes = removes
        self.in_sync = in_sync
        self.import_failures = import_failures or {}
        self.sync_code = sync_code
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []

    def __call__(
        self, argv: Sequence[str], cwd: Path, env: dict[str, str] | None
    ) -> subprocess.CompletedProcess[str]:
        argv = list(argv)
        self.calls.append((argv, env))
        if argv[1:] == ["--version"]:
            return self._done("uv 0.11.31\n")
        if argv[1] == "-c":
            return self._done(
                json.dumps(
                    {
                        "python": "3.11.15",
                        "platform": "linux-x86_64",
                        "failed": self.import_failures,
                        "distributions": ["gliner2==1.3.1", "torch==2.11.0", "vllm==0.23.0"],
                        "torch": {"version": "2.11.0", "cuda": "13.0", "arch_list": ["sm_80"]},
                    }
                )
            )
        assert argv[1:3] == ["sync", "--frozen"]
        if "--dry-run" in argv:
            fresh = env is not None and "UV_PROJECT_ENVIRONMENT" in env
            changes = (
                [{"name": name, "action": "installed"} for name in self.installable]
                if fresh
                else [{"name": name, "action": "uninstalled"} for name in self.removes]
            )
            return self._done(self._report(changes))
        if "--check" in argv:
            if not self.in_sync:
                return self._done("", "The environment is outdated\n", 1)
            return self._done(self._report([]))
        return self._done("", "Installed 93 packages\n", self.sync_code)

    @property
    def syncs(self) -> list[list[str]]:
        return [
            argv
            for argv, _env in self.calls
            if argv[1:2] == ["sync"] and "--dry-run" not in argv and "--check" not in argv
        ]

    @staticmethod
    def _report(changes: list[dict[str, str]]) -> str:
        python = {"path": "/checkout/.venv/bin/python3", "version": "3.11.15"}
        return json.dumps({"sync": {"environment": {"python": python}, "changes": changes}})

    @staticmethod
    def _done(stdout: str, stderr: str = "", code: int = 0) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], code, stdout, stderr)


@pytest.fixture(name="checkout")
def _checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A checkout holding the frozen config's backends; its Git identity is taken as verified."""

    root = tmp_path / "TrialMatchAI"
    config = root / TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(L4_BACKENDS), encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    def verified(cwd: Path, command: Sequence[str], **_expected: object) -> dict[str, object]:
        assert cwd == root and list(command) == ["--config", TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH]
        return {
            "repository": TRIALMATCHAI_L4_REPOSITORY,
            "commit": TRIALMATCHAI_L4_COMMIT,
            "tree": "5948a176ce881cd2a25431e26b5a253a85213326",
            "config_sha256": "sha256:" + "a" * 64,
        }

    monkeypatch.setattr(trialmatchai_release, "trialmatchai_checkout_configuration", verified)
    return root


def test_build_syncs_exactly_the_derived_extras_and_receipts_them(
    checkout: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _Host(removes=["datasets"])
    receipt_path = tmp_path / "receipts" / "environment.json"

    receipt = build_trialmatchai_environment(checkout, receipt_path, run=host)

    assert [argv[2:] for argv in host.syncs] == [
        ["--frozen", "--extra=entity", "--extra=gpu", "--extra=llm"]
    ]
    assert receipt["extras"] == ["entity", "gpu", "llm"]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert receipt["torch"] == {"version": "2.11.0", "cuda": "13.0", "arch_list": ["sm_80"]}
    probe = next(argv for argv, _env in host.calls if argv[1] == "-c")
    assert json.loads(probe[3]) == ["bitsandbytes", "gliner2", "torch", "transformers", "vllm"]
    # The exact sync's removals are named before it runs.
    assert "removes 1 (datasets)" in capsys.readouterr().err


def test_host_that_cannot_install_a_derived_package_is_refused_before_any_change(
    checkout: Path, tmp_path: Path
) -> None:
    # The lock's vllm and bitsandbytes carry Linux markers: a Mac sync of all three extras
    # exits 0 without them.
    host = _Host([name for name in LINUX_ENVIRONMENT if name not in {"vllm", "bitsandbytes"}])
    receipt_path = tmp_path / "environment.json"
    receipt_path.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"needs bitsandbytes \(extra gpu.*vllm \(extra gpu, for rag.backend=vllm.*"
        r"nothing was changed",
    ):
        build_trialmatchai_environment(checkout, receipt_path, run=host)

    # Only the dry run into a scratch environment ran: not the real one's, and no sync.
    [(argv, env)] = host.calls
    assert "--dry-run" in argv and env is not None and "UV_PROJECT_ENVIRONMENT" in env
    assert receipt_path.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize(
    ("host", "reason"),
    [
        (_Host(sync_code=2), "environment sync failed, exit 2"),
        (_Host(in_sync=False), "check that the environment is exactly extras entity, gpu, llm"),
        (_Host(import_failures={"vllm": "ImportError: libcuda.so.1"}), "vllm does not import"),
    ],
)
def test_a_build_that_did_not_take_leaves_no_receipt(
    checkout: Path, tmp_path: Path, host: _Host, reason: str
) -> None:
    receipt_path = tmp_path / "environment.json"
    receipt_path.write_text('{"stale": true}', encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(reason)):
        build_trialmatchai_environment(checkout, receipt_path, run=host)

    # The environment changed, so the previous receipt no longer describes it.
    assert not receipt_path.exists()
    assert not receipt_path.with_name("environment.json.partial").exists()


def test_receipt_inside_the_checkout_is_refused(checkout: Path) -> None:
    host = _Host()
    with pytest.raises(ValueError, match="outside the checkout"):
        build_trialmatchai_environment(checkout, checkout / "receipt.json", run=host)
    assert host.calls == []


def test_cli_refuses_in_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "no-checkout"
    missing.mkdir()
    receipt = str(tmp_path / "r.json")
    args = ["patient-to-trial", "trialmatchai", "build-env", "--checkout", str(missing)]
    with pytest.raises(SystemExit) as exit_info:
        main([*args, "--receipt", receipt])

    assert exit_info.value.code == 2
    assert (
        capsys.readouterr()
        .err.splitlines()[-1]
        .startswith("trial-benchmark: error: cannot inspect the TrialMatchAI Git checkout")
    )


# --- the table against upstream -----------------------------------------------------------------


def _upstream(relative: str) -> str:
    path = _pinned_checkout() / "src/trialmatchai" / relative
    if not path.is_file():
        pytest.skip("frozen TrialMatchAI checkout is not present")
    return path.read_text(encoding="utf-8")


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9._-]+", requirement)
    assert match, requirement
    return match.group(0).lower()


def test_backend_requirements_match_upstream() -> None:
    """The table mirrors upstream by hand, so upstream's own source must agree with it.

    A backend upstream adds, drops or moves to another extra fails here instead of producing an
    environment that syncs cleanly and dies at its first stage.
    """

    main_source = _upstream("main.py")
    known = {table: set(values) for table, values in TRIALMATCHAI_BACKEND_REQUIREMENTS.items()}

    # Every value upstream dispatches on is a row, and every row is a value upstream dispatches on.
    dispatched = {
        "rag.backend": set(re.findall(r'rag_backend == "(\w+)"', main_source)),
        "LLM_reranker.backend": set(
            re.findall(r'_reranker_backend\(config\) == "(\w+)"', main_source)
        ),
        "query_expansion.backend": set(
            re.findall(r'self\.backend == "(\w+)"', _upstream("matching/query_expansion.py"))
        ),
        "entity_extraction.backend": set(
            re.findall(r'backend == "(\w+)"', _upstream("entities/recognizers.py"))
        ),
        "embedder.type": set(
            re.findall(
                r'register\("embedder", "(\w+)"\)', _upstream("models/embedding/text_embedder.py")
            )
        ),
        "search_backend.type": set(
            re.findall(
                r'register\("search_backend", "(\w+)"\)', _upstream("search/lancedb_backend.py")
            )
        ),
        "quantization": set(
            re.findall(r'quantization == "(\w+)"', _upstream("models/llm/vllm_loader.py"))
        ),
    }
    assert dispatched == known

    # Every row's distribution is declared by the extra the row names.
    pyproject = tomllib.loads((_pinned_checkout() / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        extra: {_requirement_name(item) for item in requirements}
        for extra, requirements in pyproject["project"]["optional-dependencies"].items()
    }
    for values in TRIALMATCHAI_BACKEND_REQUIREMENTS.values():
        for rows in values.values():
            for extra, distribution, _module in rows:
                assert distribution in declared[extra], (extra, distribution)

    # The extras the table derives for a backend are the ones upstream's own advice names.
    preflight = _upstream("services/preflight.py")

    def advised(message: str) -> list[str]:
        found = re.search(re.escape(message) + r"[^`]*`uv sync ([^`]+)`", preflight)
        assert found, message
        return sorted(re.findall(r"--extra ([\w-]+)", found.group(1)))

    def extras(table: str, value: str) -> list[str]:
        return sorted({extra for extra, *_ in TRIALMATCHAI_BACKEND_REQUIREMENTS[table][value]})

    assert extras("entity_extraction.backend", "gliner2") == advised(
        "entity_extraction.backend=gliner2 requires the entity extra"
    )
    assert extras("rag.backend", "vllm") == advised("vLLM is required")
    assert extras("rag.backend", "transformers") == advised(
        "Transformers CPU backend requires transformers"
    )
    assert extras("rag.backend", "mlx") == advised("MLX backend requires mlx-lm")
    assert extras("rag.backend", "mlx_vlm") == advised("MLX-VLM backend requires mlx-vlm")
    embedder = re.search(r"`uv sync ([^`]+)`", _upstream("models/embedding/text_embedder.py"))
    assert embedder and extras("embedder.type", "hf") == sorted(
        re.findall(r"--extra ([\w-]+)", embedder.group(1))
    )
