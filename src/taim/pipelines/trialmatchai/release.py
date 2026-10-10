"""The TrialMatchAI L4 Systems in the public release: their run options and pinned command, the
build of the checkout's environment and of a concept store, and the static preflight.

What both TrialMatchAI families share, the run, its location-free System Input and the corpus
commands, is in :mod:`taim.pipelines.trialmatchai.execution` and
:mod:`taim.pipelines.trialmatchai.corpus`. The L4 commands here join the same ``trialmatchai``
group.

``trial-benchmark patient-to-trial trialmatchai build-env`` builds the pinned checkout's own locked
environment with exactly the extras the frozen config's backends need, derived and never listed by
hand, and writes an environment receipt outside the checkout. The run does not bind that receipt:
its System Input, and so every published identity, is unchanged.

``trial-benchmark patient-to-trial trialmatchai build-concept-store`` runs
:mod:`taim.pipelines.trialmatchai.concept_store` in that environment over vocabulary files the user
supplies, with the qualified store's recipe and pinned dictionary versions, and writes the manifest
a run reads. A run accepts such a store only as an outside build of the qualified store, on its
rows; it is never taken for the qualified store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from taim.artifacts import sha256_file
from taim.pipelines.trialmatchai.corpus import trialmatchai_checkout_configuration
from taim.pipelines.trialmatchai.execution import trialmatchai_command_group

if TYPE_CHECKING:
    from taim.release_profiles import ResolvedBenchmark


def trialmatchai_pinned_command() -> tuple[str, ...]:
    """The pinned end-to-end command a run uses when it is given none."""

    from taim.pipelines.trialmatchai.system import TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH

    return (
        "uv",
        "run",
        "trialmatchai",
        "e2e",
        "--input",
        "{input}",
        "--format",
        "text",
        "--processed-trials-folder",
        "{processed_trials}",
        "--processed-criteria-folder",
        "{processed_criteria}",
        "--reingest",
        "--rematch",
        "--config",
        TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH,
    )


def trialmatchai_run_options(
    args: argparse.Namespace,
    benchmark: ResolvedBenchmark,
    task_input_id: str,
) -> dict[str, object]:
    """The physical run options. :func:`run_system` binds harness inputs and a System Input."""

    from taim.pipelines.trialmatchai.system import (
        TRIALMATCHAI_L4_COMMIT,
        TRIALMATCHAI_L4_REPOSITORY,
        trialmatchai_l4_runtime_contract,
    )

    if args.trialmatchai_cwd is None or args.trialmatchai_corpus_dir is None:
        raise ValueError("TrialMatchAI requires --trialmatchai-cwd and --trialmatchai-corpus-dir")
    corpus = args.trialmatchai_corpus_dir.resolve()
    command = tuple(args.trialmatchai_command or ()) or trialmatchai_pinned_command()
    workspace = args.trialmatchai_workspace or (
        Path(args.output_dir) / ".trialmatchai" / (args.run_id or "external-run")
    )
    return {
        "command": command,
        "cwd": args.trialmatchai_cwd.resolve(),
        "workspace": workspace.resolve(),
        "timeout_seconds": args.trialmatchai_timeout_seconds,
        "corpus_dir": corpus,
        "search_db_path": (args.trialmatchai_search_db_path or (corpus / "search")).resolve(),
        "trialmatchai_expected_commit": TRIALMATCHAI_L4_COMMIT,
        "trialmatchai_expected_repository": TRIALMATCHAI_L4_REPOSITORY,
        "trialmatchai_runtime_contract": trialmatchai_l4_runtime_contract(),
    }


# What each backend the frozen config can select needs installed in the checkout's environment,
# as (extra, distribution, module). Upstream's extras are exact: ``uv sync`` removes every package
# the named extras do not pull in, so an environment built from a hand-listed subset loses a stage
# it needs and still exits 0. A row was once rebuilt with ``llm`` and ``gpu`` only, as upstream's
# own vLLM preflight message advises, and failed on gliner2. The extras are therefore derived from
# the config, never listed by an operator. A backend value missing here is refused rather than
# guessed; ``test_backend_requirements_match_upstream`` parses upstream's dispatch, extras and
# preflight advice, so a new or moved backend fails the test instead of building a partial
# environment.
Requirement = tuple[str, str, str]
_TORCH = ("llm", "torch", "torch")
_TRANSFORMERS = ("llm", "transformers", "transformers")
_HF_REQUIREMENTS = (_TORCH, _TRANSFORMERS)
_VLLM_REQUIREMENTS = (("gpu", "vllm", "vllm"), _TORCH, _TRANSFORMERS)
_MLX_REQUIREMENTS = (("mlx", "mlx-lm", "mlx_lm"),)
TRIALMATCHAI_BACKEND_REQUIREMENTS: dict[str, dict[str, tuple[Requirement, ...]]] = {
    "entity_extraction.backend": {
        "gliner2": (("entity", "gliner2", "gliner2"),),
        "regex": (),
        "disabled": (),
    },
    "embedder.type": {"hf": _HF_REQUIREMENTS, "hashing": ()},
    "search_backend.type": {"lancedb": ()},
    "rag.backend": {
        "vllm": _VLLM_REQUIREMENTS,
        "transformers": _HF_REQUIREMENTS,
        "mlx": _MLX_REQUIREMENTS,
        "mlx_vlm": (("mlx-vlm", "mlx-vlm", "mlx_vlm"),),
    },
    "LLM_reranker.backend": {
        "vllm": _VLLM_REQUIREMENTS,
        "transformers": _HF_REQUIREMENTS,
        "mlx": _MLX_REQUIREMENTS,
    },
    "query_expansion.backend": {
        "vllm": _VLLM_REQUIREMENTS,
        "transformers": (*_HF_REQUIREMENTS, ("llm", "peft", "peft")),
        "mlx": _MLX_REQUIREMENTS,
    },
    "quantization": {"bitsandbytes": (("gpu", "bitsandbytes", "bitsandbytes"),)},
}
_ENVIRONMENT_RECEIPT_KIND = "trialmatchai-environment-receipt"
_ENVIRONMENT_RECEIPT_SCHEMA_VERSION = "1.0"
# Runs in the checkout's interpreter: an exit status is not an installed capability, so every
# derived module is imported, and the installed distributions are listed for the receipt.
_ENVIRONMENT_PROBE = """\
import importlib, importlib.metadata, json, platform, re, sys
failed = {}
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed[name] = f"{type(exc).__name__}: {exc}"
distributions = sorted(
    {
        re.sub(r"[-_.]+", "-", str(item.metadata["Name"])).lower() + "==" + str(item.version)
        for item in importlib.metadata.distributions()
    }
)
report = {
    "python": platform.python_version(),
    "platform": f"{sys.platform}-{platform.machine()}",
    "failed": failed,
    "distributions": distributions,
}
torch = sys.modules.get("torch")
if torch is not None:
    report["torch"] = {
        "version": str(torch.__version__),
        "cuda": torch.version.cuda,
        "arch_list": list(torch.cuda.get_arch_list()),
    }
print(json.dumps(report))
"""

CommandRunner = Callable[
    [Sequence[str], Path, "dict[str, str] | None"], "subprocess.CompletedProcess[str]"
]


def _config_block(config: dict[str, object], name: str) -> dict[str, object]:
    block = config.get(name)
    return block if isinstance(block, dict) else {}


def _selected_backends(config: dict[str, object]) -> list[tuple[str, str, str]]:
    """Every backend the config selects, as (setting, value, requirement table), read the way
    upstream reads it: the same keys, defaults and enabled switches."""

    entity = _config_block(config, "entity_extraction")
    embedder = _config_block(config, "embedder")
    search = _config_block(config, "search_backend")
    rag = _config_block(config, "rag")
    reranker = _config_block(config, "LLM_reranker")
    vllm = _config_block(config, "vllm")
    expansion = _config_block(config, "query_expansion")
    selected = [
        ("entity_extraction.backend", str(entity.get("backend", "gliner2")).lower()),
        ("embedder.type", str(embedder.get("type") or embedder.get("backend") or "hf")),
        ("search_backend.type", str(search.get("type") or search.get("backend") or "lancedb")),
    ]
    rows = [(setting, value, setting) for setting, value in selected]
    served: list[tuple[str, str, str, object]] = []
    if bool(config.get("use_cot_reasoning", True)) and bool(rag.get("enabled", True)):
        quantization = vllm.get("quantization")
        served.append(("rag.backend", str(rag.get("backend", "vllm")), "vllm", quantization))
    if bool(reranker.get("enabled", True)):
        quantization = reranker.get("quantization", vllm.get("quantization"))
        backend = str(reranker.get("backend", "vllm"))
        served.append(("LLM_reranker.backend", backend, "LLM_reranker", quantization))
    if expansion.get("enabled"):
        backend = str(expansion.get("backend") or rag.get("backend") or "vllm")
        served.append(("query_expansion.backend", backend, "vllm", vllm.get("quantization")))
    for setting, backend, quantization_block, quantization in served:
        rows.append((setting, backend, setting))
        if backend == "vllm" and quantization:
            rows.append((f"{quantization_block}.quantization", str(quantization), "quantization"))
    return list(dict.fromkeys(rows))


def derive_trialmatchai_requirements(
    config: dict[str, object],
) -> list[tuple[str, str, tuple[Requirement, ...]]]:
    """Each selected backend with what it needs installed; refuses a backend with no known row."""

    derived = []
    for setting, value, table in _selected_backends(config):
        known = TRIALMATCHAI_BACKEND_REQUIREMENTS[table]
        if value not in known:
            raise ValueError(
                f"TrialMatchAI config {setting}={value!r} maps to no known extra; refusing to "
                "guess the environment. Add its row to TRIALMATCHAI_BACKEND_REQUIREMENTS from "
                "upstream's pyproject.toml"
            )
        derived.append((setting, value, known[value]))
    return derived


def _run_command(
    argv: Sequence[str], cwd: Path, env: dict[str, str] | None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - resolved executables and a fixed argv
        list(argv), cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def _command_failure(completed: subprocess.CompletedProcess[str]) -> str:
    lines = [line for line in (completed.stderr or completed.stdout or "").splitlines() if line]
    return f"exit {completed.returncode}: {lines[-1] if lines else 'no output'}"


def _uv_sync_report(completed: subprocess.CompletedProcess[str], role: str) -> dict[str, Any]:
    """The ``sync`` object of ``uv sync --output-format json``, or a refusal naming ``role``."""

    if completed.returncode != 0:
        raise ValueError(f"TrialMatchAI {role} failed, {_command_failure(completed)}")
    try:
        sync = json.loads(completed.stdout)["sync"]
        if not all(isinstance(change["name"], str) for change in sync["changes"]):
            raise TypeError("a change has no name")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"TrialMatchAI {role}: uv reported no JSON sync plan ({exc}); this needs a uv whose "
            "`uv sync` accepts --output-format json"
        ) from exc
    return sync  # type: ignore[no-any-return]


def _changed(sync: dict[str, Any], action: str) -> list[str]:
    return sorted(change["name"] for change in sync["changes"] if change.get("action") == action)


def _distribution_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _say(message: str) -> None:
    print(f"trialmatchai build-env: {message}", file=sys.stderr, flush=True)


def build_trialmatchai_environment(
    checkout: Path,
    receipt_path: Path,
    *,
    run: CommandRunner = _run_command,
) -> dict[str, object]:
    """Derive the extras from the frozen config, sync exactly those, verify, and write a receipt.

    Every refusal that needs nothing changed comes first: the checkout, the config, a backend with
    no known extra, and a derived package the lock does not install on this host. Only then is a
    previous receipt removed and the environment synced, after naming what the sync removes. The
    receipt is written last, and only when the environment matches the lock for exactly the
    derived extras and every derived module imports in it.
    """

    from taim.contracts import content_sha256
    from taim.executables import resolve_executable
    from taim.pipelines.trialmatchai.system import (
        TRIALMATCHAI_L4_COMMIT,
        TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH,
        TRIALMATCHAI_L4_CONFIG_SHA256,
        TRIALMATCHAI_L4_REPOSITORY,
    )

    checkout = checkout.resolve()
    receipt_path = receipt_path.resolve()
    if receipt_path.is_relative_to(checkout):
        raise ValueError(
            "TrialMatchAI environment receipt must be outside the checkout, which must stay clean"
        )
    source = trialmatchai_checkout_configuration(
        checkout,
        ("--config", TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH),
        expected_commit=TRIALMATCHAI_L4_COMMIT,
        expected_repository=TRIALMATCHAI_L4_REPOSITORY,
        expected_config_sha256=TRIALMATCHAI_L4_CONFIG_SHA256,
    )
    config = json.loads((checkout / TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH).read_text("utf-8"))
    derived = derive_trialmatchai_requirements(config)
    derivation = [
        {"setting": setting, "value": value, "extras": sorted({row[0] for row in requirements})}
        for setting, value, requirements in derived
    ]
    for row in derivation:
        _say(f"{row['setting']}={row['value']} needs {', '.join(row['extras']) or 'no extra'}")
    needed = {requirement for _setting, _value, rows in derived for requirement in rows}
    extras = sorted({extra for extra, _distribution, _module in needed})
    uv = resolve_executable("uv")
    sync = [uv, "sync", "--frozen", *(f"--extra={extra}" for extra in extras)]

    # What a fresh environment of exactly these extras holds on this host. A platform marker
    # drops a package without a word and the sync still exits 0, so the lock is asked first.
    with tempfile.TemporaryDirectory(prefix="trialmatchai-build-env-") as scratch:
        fresh = _uv_sync_report(
            run(
                [*sync, "--dry-run", "--output-format", "json"],
                checkout,
                {**os.environ, "UV_PROJECT_ENVIRONMENT": str(Path(scratch) / "environment")},
            ),
            "dry run into a fresh environment",
        )
    installable = {_distribution_key(name) for name in _changed(fresh, "installed")}
    unavailable: dict[tuple[str, str], list[str]] = {}
    for setting, value, rows in derived:
        for extra, distribution, _module in rows:
            if _distribution_key(distribution) not in installable:
                unavailable.setdefault((distribution, extra), []).append(f"{setting}={value}")
    if unavailable:
        missing = "; ".join(
            f"{distribution} (extra {extra}, for {', '.join(settings)})"
            for (distribution, extra), settings in sorted(unavailable.items())
        )
        raise ValueError(
            f"TrialMatchAI config needs {missing}, which the lock does not install on "
            f"{sys.platform}-{platform.machine()}; nothing was changed"
        )

    plan = _uv_sync_report(
        run([*sync, "--dry-run", "--output-format", "json"], checkout, None),
        "dry run against the environment",
    )
    removed = _changed(plan, "uninstalled")
    _say(
        f"syncing exactly extras {', '.join(extras)}: installs {len(_changed(plan, 'installed'))}, "
        f"removes {len(removed)}" + (f" ({', '.join(removed)})" if removed else "")
    )
    if receipt_path.exists():
        receipt_path.unlink()
        _say(f"removed the previous receipt {receipt_path}")
    completed = run(sync, checkout, None)
    if completed.returncode != 0:
        raise ValueError(f"TrialMatchAI environment sync failed, {_command_failure(completed)}")

    check = _uv_sync_report(
        run([*sync, "--check", "--output-format", "json"], checkout, None),
        f"check that the environment is exactly extras {', '.join(extras)}",
    )
    interpreter = check.get("environment", {}).get("python", {}).get("path")
    if not isinstance(interpreter, str):
        raise ValueError("TrialMatchAI environment check did not name the environment's Python")
    modules = sorted({module for _extra, _distribution, module in needed})
    probed = run([interpreter, "-c", _ENVIRONMENT_PROBE, json.dumps(modules)], checkout, None)
    if probed.returncode != 0:
        raise ValueError(f"TrialMatchAI environment probe failed, {_command_failure(probed)}")
    probe = json.loads(probed.stdout)
    if probe["failed"]:
        raise ValueError(
            "TrialMatchAI environment synced, but "
            + "; ".join(
                f"{module} does not import ({error})" for module, error in probe["failed"].items()
            )
        )
    installed = "".join(f"{line}\n" for line in probe["distributions"]).encode()
    core: dict[str, object] = {
        "schema_version": _ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
        "kind": _ENVIRONMENT_RECEIPT_KIND,
        "checkout": {key: source[key] for key in ("repository", "commit", "tree")},
        "config": {"path": TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH, "sha256": source["config_sha256"]},
        "derivation": derivation,
        "extras": extras,
        "uv_lock_sha256": sha256_file(checkout / "uv.lock"),
        "uv_version": run([uv, "--version"], checkout, None).stdout.strip(),
        "python": probe["python"],
        "platform": probe["platform"],
        "torch": probe.get("torch"),
        "installed_distributions": {
            "count": len(probe["distributions"]),
            "sha256": f"sha256:{hashlib.sha256(installed).hexdigest()}",
        },
    }
    receipt = {**core, "receipt_id": content_sha256(core)}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    partial = receipt_path.with_name(f"{receipt_path.name}.partial")
    partial.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    partial.replace(receipt_path)
    return receipt


def trialmatchai_add_patient_to_trial_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add the L4 commands to the ``trialmatchai`` group, beside the corpus commands."""

    trialmatchai_commands = trialmatchai_command_group(commands)
    build_env = trialmatchai_commands.add_parser(
        "build-env",
        help="install exactly the extras the frozen config's backends need into the pinned "
        "checkout's environment, verify them, and write an environment receipt",
    )
    build_env.add_argument(
        "--checkout", type=Path, required=True, help="the clean pinned TrialMatchAI checkout"
    )
    build_env.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="file that receives the environment receipt, outside the checkout",
    )
    build_env.set_defaults(handler=_build_env)
    _add_build_concept_store_command(trialmatchai_commands)
    _add_preflight_command(trialmatchai_commands)


def _build_env(args: argparse.Namespace) -> int:
    from taim.cli_support import print_payload

    started = time.perf_counter()
    receipt = build_trialmatchai_environment(args.checkout, args.receipt)
    print_payload(
        {
            "receipt": str(args.receipt.resolve()),
            "receipt_id": receipt["receipt_id"],
            "extras": receipt["extras"],
            "seconds": round(time.perf_counter() - started, 3),
        }
    )
    return 0


def _add_build_concept_store_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    build_store = commands.add_parser(
        "build-concept-store",
        help="build the concept store from your own licensed OMOP vocabulary files in the pinned "
        "checkout's environment, with its provenance manifest; resumes an interrupted build",
    )
    build_store.add_argument(
        "--checkout",
        type=Path,
        required=True,
        help="the clean pinned TrialMatchAI checkout, its environment built by build-env",
    )
    build_store.add_argument(
        "--concept-csv",
        type=Path,
        required=True,
        help="CONCEPT.csv of your own OMOP vocabulary download, tab-delimited",
    )
    build_store.add_argument(
        "--synonym-csv",
        type=Path,
        required=True,
        help="CONCEPT_SYNONYM.csv of the same download",
    )
    build_store.add_argument(
        "--source-release",
        required=True,
        help="the vocabulary release the files came from, recorded in the manifest",
    )
    build_store.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="directory that receives the concept store, outside the checkout",
    )
    build_store.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="directory for the downloaded open dictionaries and the embedding checkpoint, "
        "outside the checkout; a build resumes from it",
    )
    build_store.add_argument(
        "--force",
        action="store_true",
        help="delete the checkpoint and the store and embed every row again",
    )
    build_store.set_defaults(handler=_build_concept_store)


def _build_concept_store(args: argparse.Namespace) -> int:
    return build_trialmatchai_concept_store(
        args.checkout,
        concept_csv=args.concept_csv,
        synonym_csv=args.synonym_csv,
        source_release=args.source_release,
        db_path=args.db_path,
        work_dir=args.work_dir,
        force=args.force,
    )


def build_trialmatchai_concept_store(
    checkout: Path,
    *,
    concept_csv: Path,
    synonym_csv: Path,
    source_release: str,
    db_path: Path,
    work_dir: Path,
    force: bool = False,
    call: Callable[..., int] = subprocess.call,
) -> int:
    """Run the concept-store builder in the verified checkout's own interpreter.

    The recipe is the qualified store's: the frozen config's pinned embedder, every OMOP
    vocabulary the L4 Systems link against, TrialMatchAI's open dictionaries at their pinned
    versions, and the pinned chunking and index parameters. The builder module is executed by
    path, isolated, so it imports nothing from this installation. Its output streams through; its
    exit status is returned.
    """

    from taim.pipelines.trialmatchai import concept_store
    from taim.pipelines.trialmatchai.preflight import PreflightFailure, project_interpreter
    from taim.pipelines.trialmatchai.system import (
        TRIALMATCHAI_L4_COMMIT,
        TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH,
        TRIALMATCHAI_L4_CONFIG_SHA256,
        TRIALMATCHAI_L4_REPOSITORY,
    )

    checkout = checkout.resolve()
    db_path = db_path.resolve()
    work_dir = work_dir.resolve()
    for role, path in (("concept store", db_path), ("work directory", work_dir)):
        if path.is_relative_to(checkout):
            raise ValueError(
                f"TrialMatchAI {role} must be outside the checkout, which must stay clean"
            )
    trialmatchai_checkout_configuration(
        checkout,
        ("--config", TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH),
        expected_commit=TRIALMATCHAI_L4_COMMIT,
        expected_repository=TRIALMATCHAI_L4_REPOSITORY,
        expected_config_sha256=TRIALMATCHAI_L4_CONFIG_SHA256,
    )
    try:
        interpreter = project_interpreter(checkout, os.environ)
    except PreflightFailure as exc:
        raise ValueError(f"{exc}; build it first with trialmatchai build-env") from exc
    argv = [
        str(interpreter),
        "-I",
        str(Path(concept_store.__file__).resolve()),
        *("--config", str(checkout / TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH)),
        *("--trialmatchai-root", str(checkout)),
        *("--expected-trialmatchai-commit", TRIALMATCHAI_L4_COMMIT),
        *("--concept-csv", str(concept_csv.resolve())),
        *("--synonym-csv", str(synonym_csv.resolve())),
        *("--source-release", source_release),
        *("--sources", "open", "--concept-cache", str(work_dir / "concept_dicts")),
        *("--checkpoint-dir", str(work_dir / "checkpoint")),
        *("--db-path", str(db_path)),
        *(("--force",) if force else ()),
    ]
    # The frozen config embeds on the CPU at run time; the qualified store was embedded on a GPU.
    environment = {**os.environ, "TRIALMATCHAI_EMBEDDER_USE_GPU": "true"}
    return call(argv, cwd=checkout, env=environment)


def _add_preflight_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    preflight = commands.add_parser(
        "preflight",
        help="check, loading no model, that this environment can run the TrialMatchAI L4 "
        "pipeline; the last line is the total",
    )
    preflight.add_argument(
        "--system",
        required=True,
        choices=_trialmatchai_systems(),
        metavar="SYSTEM",
        help="the TrialMatchAI System the run will run; support list reports the Systems",
    )
    preflight.add_argument(
        "--trialmatchai-cwd",
        type=Path,
        required=True,
        help="the pinned TrialMatchAI checkout the run will use",
    )
    preflight.add_argument(
        "--trialmatchai-corpus-dir",
        type=Path,
        required=True,
        help="the prepared corpus the run will use",
    )
    preflight.add_argument(
        "--trialmatchai-search-db-path",
        type=Path,
        help="the search database the run will use; omitted, search/ in the corpus",
    )
    preflight.set_defaults(handler=_preflight)


def _trialmatchai_systems() -> list[str]:
    """The TrialMatchAI L4 Systems this installation ships, each checked by this preflight."""

    from taim.pipeline_extensions import pipeline_extensions

    return sorted(
        system
        for extension in pipeline_extensions()
        if extension.add_patient_to_trial_commands is trialmatchai_add_patient_to_trial_commands
        for system in extension.system_tasks
    )


def _preflight(args: argparse.Namespace) -> int:
    from taim.pipelines.trialmatchai.preflight import run_static_preflight, summary_line

    corpus = args.trialmatchai_corpus_dir.resolve()
    checks = run_static_preflight(
        system_id=args.system,
        cwd=args.trialmatchai_cwd.resolve(),
        corpus_dir=corpus,
        search_db_path=(args.trialmatchai_search_db_path or corpus / "search").resolve(),
        environment=os.environ,
        emit=lambda check: print(check.line(), flush=True),
    )
    print(summary_line(checks), flush=True)
    return 1 if any(not check.passed for check in checks) else 0
