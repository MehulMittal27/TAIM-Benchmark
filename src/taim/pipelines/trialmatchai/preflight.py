"""Check, before a TrialMatchAI L4 run, that this environment can run that pipeline.

A TrialMatchAI row takes most of a day on one card, and its environment failures have surfaced
hours into a row: an environment rebuilt without one of the extras its config needs, a card for
which torch reports CUDA available but carries no kernels, a concept store that nothing compared
with the qualified one. A check that runs a different System proves nothing about this one, so
every check here reads what the L4 pipeline itself reads: the pinned checkout and its locked
environment, with the extras derived from the frozen config as ``build-env`` derives them, the
config itself, the card its device index names, the model cache upstream resolves through, the
concept store and the prepared corpus.

This is the static half. It loads no model and changes nothing: the environment is compared with
its lock by ``uv sync --check``, and the checkout's interpreter is probed with bytecode writing off
and the Hugging Face Hub offline. It cannot show that the two vLLM engines start and reach an
eligibility decision; only running the pipeline shows that, and that dynamic check is not built.

Every check prints ``PASS`` or ``FAIL`` with its reason, and the last line is the total, so a
truncated view cannot hide a failure above later ``PASS`` lines.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taim.baselines.dense import arch_list_supports
from taim.executables import ExecutableNotFoundError, resolve_executable
from taim.pipelines.trialmatchai.corpus import (
    TrialMatchAIAdapterError,
    check_trialmatchai_corpus_names,
    load_trialmatchai_corpus_receipt,
)
from taim.pipelines.trialmatchai.release import (
    derive_trialmatchai_requirements,
    trialmatchai_pinned_command,
)
from taim.pipelines.trialmatchai.system import (
    OUTSIDE_BUILD_STORE,
    TRIALMATCHAI_L4_COMMIT,
    TRIALMATCHAI_L4_REPOSITORY,
    TrialMatchAIL4Base,
    VerifiedInputs,
    verify_trialmatchai_static_inputs,
)

#: The smallest card the frozen config has been shown to start both engines on is an
#: A100-SXM4-40GB, which reports 39.5 GiB. Less is unproven: a 24 GB L4 cannot hold the fp8 KV
#: cache at the config's max_model_len and max_num_seqs within its 0.6 share.
TRIALMATCHAI_L4_MIN_PROVEN_GPU_MEMORY_BYTES = 39 * 1024**3
_LOCK_CHECK_TIMEOUT_SECONDS = 300.0
# Importing vLLM from an environment on a cold network filesystem has taken minutes.
_PROBE_TIMEOUT_SECONDS = 900.0

# Runs in the checkout's own interpreter, which TAIM must not import from. It reads a JSON spec
# from argv and prints one JSON line: what imports, what torch sees, and what the model cache
# resolves offline.
_PROBE = r"""
import importlib, json, os, sys
from pathlib import Path
from typing import Any

spec = json.loads(sys.argv[1])
result = {"platform": sys.platform, "python": sys.version.split()[0], "modules": {}, "models": {}}
for name in spec["modules"]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        result["modules"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        result["modules"][name] = {"version": str(getattr(module, "__version__", "unknown"))}
try:
    import torch

    gpu = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "available": torch.cuda.is_available(),
        "arch_list": torch.cuda.get_arch_list(),
    }
    if gpu["available"]:
        gpu["device_count"] = torch.cuda.device_count()
        index = spec["device"]
        if 0 <= index < gpu["device_count"]:
            free, total = torch.cuda.mem_get_info(index)
            gpu.update(
                name=torch.cuda.get_device_name(index),
                capability=list(torch.cuda.get_device_capability(index)),
                free_bytes=free,
                total_bytes=total,
            )
except Exception as exc:
    gpu = {"error": f"{type(exc).__name__}: {exc}"}
result["gpu"] = gpu
try:
    from huggingface_hub import constants, snapshot_download

    result["model_cache"] = str(constants.HF_HUB_CACHE)
except Exception as exc:
    result["model_cache_error"] = f"{type(exc).__name__}: {exc}"
else:
    for repo, revision in spec["models"]:
        entry = result["models"][repo] = {}
        try:
            snapshot = Path(snapshot_download(repo, revision=revision, local_files_only=True))
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            continue
        problems = sorted(
            str(path.relative_to(snapshot)) + " is a dangling link"
            for path in snapshot.rglob("*")
            if path.is_symlink() and not path.exists()
        )
        for index_file in snapshot.glob("*.index.json"):
            shards = set(json.loads(index_file.read_text()).get("weight_map", {}).values())
            problems += sorted(
                f"{shard}, named by {index_file.name}, is missing"
                for shard in shards
                if not (snapshot / shard).exists()
            )
        weights = (path for path in snapshot.rglob("*") if path.suffix in (".safetensors", ".bin"))
        if not any(path.is_file() for path in weights):
            problems.append("it holds no weight file")
        entry["problems"] = problems
        main = snapshot.parents[1] / "refs" / "main"
        entry["main"] = main.read_text().strip() if main.is_file() else None
print(json.dumps(result))
"""


class PreflightFailure(Exception):
    """One check's refusal, with its reason."""


_REFUSALS = (PreflightFailure, TrialMatchAIAdapterError, ValueError, OSError)


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """The outcome of one check: its name, whether it passed, and what it found or refused."""

    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'} {self.name}: {self.detail}"


def summary_line(checks: Sequence[PreflightCheck]) -> str:
    """The last line of every preflight: the total, which alone decides whether to run."""

    failed = sum(not check.passed for check in checks)
    if failed:
        return f"preflight: {failed} FAIL of {len(checks)} - not runnable"
    return (
        f"preflight: 0 FAIL of {len(checks)} static checks - the dynamic check that runs the "
        "pipeline itself is not built, so this does not show that the engines start"
    )


def _section(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = config.get(name)
    return section if isinstance(section, Mapping) else {}


def _gib(value: float) -> str:
    return f"{value / 1024**3:.1f} GiB"


def project_interpreter(cwd: Path, environment: Mapping[str, str]) -> Path:
    """The interpreter of the environment ``uv run`` would use in the checkout, never created.

    ``uv run`` creates a missing environment even with ``--no-sync``, so the preflight resolves
    the location the way uv does and refuses when nothing is there.
    """

    configured = environment.get("UV_PROJECT_ENVIRONMENT")
    root = Path(configured) if configured else Path(".venv")
    root = root if root.is_absolute() else cwd / root
    interpreter = root / "bin" / "python"
    if not interpreter.is_file():
        raise PreflightFailure(f"the checkout has no environment at {root}")
    return interpreter


def _verify_run_inputs(system_id: str, cwd: Path, environment: Mapping[str, str]) -> VerifiedInputs:
    """What the run verifies before its first model, through the run's own function and order."""

    # Every System this preflight checks is the frozen L4 method under the identity it names.
    system = TrialMatchAIL4Base()
    system.system_id = system_id
    inputs = verify_trialmatchai_static_inputs(
        system,
        command=trialmatchai_pinned_command(),
        cwd=cwd,
        environment=environment,
        expected_commit=TRIALMATCHAI_L4_COMMIT,
        expected_repository=TRIALMATCHAI_L4_REPOSITORY,
    )
    if inputs.concept_store is None:
        raise PreflightFailure("the pinned command reads no concept store")
    return inputs


def _describe_run_inputs(inputs: VerifiedInputs) -> str:
    checkout, store = inputs.checkout, inputs.concept_store or {}
    adapters = checkout.get("adapter_sha256")
    return (
        f"{checkout['commit']} from {checkout['repository']}, clean; "
        f"config {checkout['config_sha256']}; both LoRA adapters match their content pins: "
        + ", ".join(
            f"{option} {digest}"
            for option, digest in (adapters.items() if isinstance(adapters, Mapping) else ())
        )
        + "; the required variables are set, the forbidden ones are absent, nothing overrides "
        f"the config, and the serving backends are {inputs.backends['effective']}; "
        + _describe_concept_store(store)
    )


def _describe_concept_store(store: Mapping[str, Any]) -> str:
    verified = store["verified_identity"]
    if verified["store"] == OUTSIDE_BUILD_STORE:
        return (
            f"the store at {store['db_path']} is an outside build of the qualified store "
            f"{verified['qualified_store']}: its manifest claims the qualified recipe and pinned "
            "dictionaries, and its rows, read from the store, are the qualified store's; its "
            "vectors and indexes are its own, so a run records it as an outside build and its "
            "results are that build's"
        )
    return (
        f"the store at {store['db_path']} is the qualified store "
        f"{verified['store']}, by its manifest and every file's content"
    )


def _requirements(config: Mapping[str, object]) -> tuple[list[str], dict[str, str]]:
    """The extras and the modules, each with its extra, that the verified config's backends
    need, derived as ``build-env`` derives them.

    ``uv sync`` is exact, so the lock check names exactly these extras. The lock check alone
    passes on a host whose platform markers skip a package (the lock installs vLLM on Linux
    only), so each module must also import in the checkout's interpreter.
    """

    needed = {
        requirement
        for _setting, _value, requirements in derive_trialmatchai_requirements(dict(config))
        for requirement in requirements
    }
    extras = sorted({extra for extra, _distribution, _module in needed})
    modules = {module: extra for extra, _distribution, module in sorted(needed)}
    return extras, dict(sorted(modules.items()))


def _probed(probe: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = probe.get(key)
    if not isinstance(value, Mapping):
        raise PreflightFailure(f"the probe reported no {key}")
    return value


def _check_prepared_corpus(corpus_dir: Path) -> str:
    receipt = load_trialmatchai_corpus_receipt(corpus_dir)
    check_trialmatchai_corpus_names(corpus_dir, receipt)
    roles = receipt["roles"]
    counts = (
        ", ".join(f"{entry['file_count']} {role}" for role, entry in roles.items())
        if isinstance(roles, Mapping)
        else ""
    )
    return f"receipt {receipt['receipt_id']}; the file names match it ({counts})"


def _check_search_database(config: Mapping[str, object], search_db_path: Path) -> str:
    backend = _section(config, "search_backend")
    if not search_db_path.is_dir():
        raise PreflightFailure(f"the search database does not exist: {search_db_path}")
    tables = [str(backend.get(key)) for key in ("trials_table", "criteria_table")]
    missing = [
        table
        for table in tables
        if not (search_db_path / f"{table}.lance").is_dir()
        or not any((search_db_path / f"{table}.lance").iterdir())
    ]
    if missing:
        raise PreflightFailure(
            f"the search database at {search_db_path} has no "
            + " or ".join(f"{table}.lance" for table in missing)
            + " table, which the config's search_backend names; build it with upstream's "
            "index command over the prepared corpus"
        )
    return f"{' and '.join(f'{table}.lance' for table in tables)} present"


def _check_environment_lock(
    config: Mapping[str, object], cwd: Path, environment: Mapping[str, str]
) -> str:
    required, _modules = _requirements(config)
    extras = [argument for extra in required for argument in ("--extra", extra)]
    try:
        uv = resolve_executable("uv")
    except ExecutableNotFoundError as exc:
        raise PreflightFailure(f"uv cannot compare the lock: {exc}") from exc
    try:
        completed = subprocess.run(  # noqa: S603 - resolved executable and fixed argv
            [uv, "sync", "--frozen", "--offline", *extras, "--check"],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=_LOCK_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise PreflightFailure(f"uv sync --check did not finish: {exc}") from exc
    if completed.returncode != 0:
        output = (completed.stderr + completed.stdout).splitlines()
        removed = [line.strip()[2:] for line in output if line.strip().startswith("- ")]
        added = [line.strip()[2:] for line in output if line.strip().startswith("+ ")]
        if removed or added:
            found = (
                f"a sync would remove {len(removed)} and install {len(added)} packages "
                f"(install: {', '.join(added[:6]) or 'none'}"
                + (", ..." if len(added) > 6 else "")
                + ")"
            )
        else:
            found = output[-1] if output else f"uv exited {completed.returncode}"
        raise PreflightFailure(
            f"the checkout's environment is not its lock with the extras "
            f"{', '.join(required)}: {found}. Rebuild it with: trial-benchmark "
            f"patient-to-trial trialmatchai build-env --checkout {cwd} --receipt "
            "<file outside the checkout>"
        )
    return (
        f"the environment is exactly the lock with the extras {', '.join(required)}, "
        "derived from the config's backends"
    )


def _run_probe(
    cwd: Path, environment: Mapping[str, str], config: Mapping[str, object] | None
) -> Mapping[str, object]:
    """Run the probe once in the checkout's interpreter and return its result."""

    interpreter = project_interpreter(cwd, environment)
    device = 0
    models: list[tuple[str, str]] = []
    modules: list[str] = []
    if config is not None:
        try:
            modules = list(_requirements(config)[1])
        except ValueError:
            modules = []
        device_index = _section(config, "global").get("device", 0)
        device = device_index if isinstance(device_index, int) else 0
        models = [(repo, revision) for _role, repo, revision in _pinned_models(config)]
    spec = {
        "modules": modules,
        "device": device,
        "models": models,
    }
    probe_environment = {**environment, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        completed = subprocess.run(  # noqa: S603 - the checkout's interpreter, fixed argv
            [str(interpreter), "-c", _PROBE, json.dumps(spec)],
            cwd=cwd,
            env=probe_environment,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreflightFailure(f"the probe in {interpreter} did not finish: {exc}") from exc
    lines = completed.stdout.strip().splitlines()
    try:
        probe = json.loads(lines[-1]) if completed.returncode == 0 and lines else None
    except ValueError:
        probe = None
    if not isinstance(probe, dict):
        tail = " | ".join(completed.stderr.strip().splitlines()[-3:])
        raise PreflightFailure(
            f"the probe in {interpreter} exited {completed.returncode} with no result"
            + (f": {tail}" if tail else "")
        )
    return probe


def _check_modules(config: Mapping[str, object], probe: Mapping[str, object]) -> str:
    _extras, required = _requirements(config)
    modules = _probed(probe, "modules")
    failed = []
    for module, extra in required.items():
        entry = modules.get(module)
        if not isinstance(entry, Mapping) or "error" in entry:
            error = entry.get("error") if isinstance(entry, Mapping) else "not probed"
            failed.append(f"{module} (extra {extra}): {error}")
    if failed:
        platform = str(probe.get("platform"))
        note = (
            f". This host is {platform}; the lock installs vllm only on Linux"
            if platform != "linux"
            else ""
        )
        raise PreflightFailure("; ".join(failed) + note)
    return ", ".join(
        f"{module} {modules[module]['version']}"  # type: ignore[index]
        for module in required
    )


def _check_gpu_supported(probe: Mapping[str, object]) -> str:
    gpu = _probed(probe, "gpu")
    if "error" in gpu:
        raise PreflightFailure(f"torch could not report on CUDA: {gpu['error']}")
    build = f"torch {gpu.get('torch')} (CUDA {gpu.get('cuda')})"
    if not gpu.get("available"):
        raise PreflightFailure(f"{build} sees no usable CUDA device")
    capability = gpu.get("capability")
    if not isinstance(capability, list):
        raise PreflightFailure(
            f"{build} sees {gpu.get('device_count')} device(s), none at the config's device index"
        )
    raw_arch_list = gpu.get("arch_list")
    arch_list = [str(arch) for arch in raw_arch_list] if isinstance(raw_arch_list, list) else []
    name = f"{gpu.get('name')} (sm_{capability[0]}{capability[1]})"
    built = ", ".join(arch_list) or "no architecture"
    if not arch_list_supports(capability, arch_list):
        raise PreflightFailure(
            f"{name} has CUDA available but is not supported: {build} carries kernels for {built}"
        )
    return f"{name} is supported by {build}, built for {built}"


def _check_gpu_memory(config: Mapping[str, object], probe: Mapping[str, object]) -> str:
    gpu = _probed(probe, "gpu")
    total, free = gpu.get("total_bytes"), gpu.get("free_bytes")
    if not isinstance(total, int) or not isinstance(free, int):
        raise PreflightFailure("not checked: no CUDA device was measured")
    claimed = 0.0
    for section in ("vllm", "LLM_reranker"):
        share = _section(config, section).get("gpu_memory_utilization")
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise PreflightFailure(f"the config declares no {section}.gpu_memory_utilization")
        claimed += share
    problems = []
    if total < TRIALMATCHAI_L4_MIN_PROVEN_GPU_MEMORY_BYTES:
        problems.append(
            f"the card has {_gib(total)}, below the "
            f"{_gib(TRIALMATCHAI_L4_MIN_PROVEN_GPU_MEMORY_BYTES)} of the smallest card this "
            "config is shown to start on"
        )
    if free < claimed * total:
        problems.append(
            f"{_gib(free)} of {_gib(total)} is free, but the two vLLM engines claim "
            f"{claimed:.2f} of the card ({_gib(claimed * total)})"
        )
    if problems:
        raise PreflightFailure("; ".join(problems))
    return (
        f"{_gib(free)} of {_gib(total)} free; the engines claim {claimed:.2f} "
        f"({_gib(claimed * total)})"
    )


def _check_model_cache(config: Mapping[str, object], probe: Mapping[str, object]) -> str:
    if "model_cache_error" in probe:
        raise PreflightFailure(
            f"huggingface_hub is not usable in the checkout: {probe['model_cache_error']}"
        )
    models = _probed(probe, "models")
    cache = probe.get("model_cache")
    problems = []
    pinned = _pinned_models(config)
    for role, repo, revision in pinned:
        entry = models.get(repo)
        if not isinstance(entry, Mapping):
            problems.append(f"{role} {repo} was not probed")
        elif "error" in entry:
            problems.append(
                f"{role} {repo}@{revision} does not resolve offline in {cache} "
                f"({entry['error']}); download it at that revision before the run (a gated "
                "repository also needs its licence accepted and HF_TOKEN set)"
            )
        else:
            problems += [f"{role} {repo}@{revision}: {problem}" for problem in entry["problems"]]
            if role == "entity extraction" and entry.get("main") != revision:
                # gliner2 1.3.1 downloads without a revision, so it loads whatever refs/main
                # names in the cache, whatever revision the config pins.
                problems.append(
                    f"{role} {repo}: gliner2 loads refs/main, which names "
                    f"{entry.get('main')}, not the pinned {revision}; write the pin to "
                    "refs/main in the cache"
                )
    if problems:
        raise PreflightFailure("; ".join(problems))
    return f"{len(pinned)} pinned repositories resolve offline at their revisions in {cache}"


def _pinned_models(config: Mapping[str, object]) -> list[tuple[str, str, str]]:
    """Each Hub repository the frozen config loads, as (role, repository, revision)."""

    model = _section(config, "model")
    embedder = _section(config, "embedder")
    entity = _section(config, "entity_extraction")
    return [
        (role, str(repo), str(revision))
        for role, repo, revision in (
            ("eligibility base", model.get("base_model"), model.get("base_model_revision")),
            (
                "reranker base",
                model.get("reranker_model_path"),
                model.get("reranker_model_revision"),
            ),
            ("embedder", embedder.get("model_name"), embedder.get("revision")),
            ("entity extraction", entity.get("model_name"), entity.get("model_revision")),
        )
    ]


def run_static_preflight(
    *,
    system_id: str,
    cwd: Path,
    corpus_dir: Path,
    search_db_path: Path,
    environment: Mapping[str, str],
    emit: Callable[[PreflightCheck], None] = lambda _check: None,
) -> list[PreflightCheck]:
    """Run every static check, emitting each as it completes, and return them in order.

    The first check is the run's own: :func:`verify_trialmatchai_static_inputs` refuses what the
    run would refuse, in the run's order, and stops at the first refusal as the run does. The
    checks after it read the config it verified, and the card and model cache through one probe.
    """

    checks: list[PreflightCheck] = []

    def report(result: PreflightCheck) -> None:
        checks.append(result)
        emit(result)

    def record(name: str, check: Callable[[], str]) -> None:
        try:
            result = PreflightCheck(name, True, check())
        except _REFUSALS as exc:
            result = PreflightCheck(name, False, str(exc))
        report(result)

    # The run gives upstream the environment without TAIM's own virtual environment.
    environment = {name: value for name, value in environment.items() if name != "VIRTUAL_ENV"}
    inputs: VerifiedInputs | None = None
    try:
        inputs = _verify_run_inputs(system_id, cwd, environment)
    except _REFUSALS as exc:
        report(PreflightCheck("run-inputs", False, str(exc)))
    else:
        report(PreflightCheck("run-inputs", True, _describe_run_inputs(inputs)))

    def config() -> Mapping[str, object]:
        if inputs is None or inputs.config is None:
            raise PreflightFailure("not checked: the run's inputs did not verify")
        return inputs.config

    record("prepared-corpus", lambda: _check_prepared_corpus(corpus_dir))
    record("search-database", lambda: _check_search_database(config(), search_db_path))
    record("environment-lock", lambda: _check_environment_lock(config(), cwd, environment))
    probe: Mapping[str, object] | None = None
    probe_failure = ""
    try:
        probe = _run_probe(cwd, environment, inputs.config if inputs else None)
    except PreflightFailure as exc:
        probe_failure = str(exc)

    def probed() -> Mapping[str, object]:
        if probe is None:
            raise PreflightFailure(f"not checked: {probe_failure}")
        return probe

    record("modules", lambda: _check_modules(config(), probed()))
    record("gpu-supported", lambda: _check_gpu_supported(probed()))
    record("gpu-memory", lambda: _check_gpu_memory(config(), probed()))
    record("model-cache", lambda: _check_model_cache(config(), probed()))
    return checks


__all__ = [
    "TRIALMATCHAI_L4_MIN_PROVEN_GPU_MEMORY_BYTES",
    "PreflightCheck",
    "PreflightFailure",
    "arch_list_supports",
    "run_static_preflight",
    "summary_line",
]
