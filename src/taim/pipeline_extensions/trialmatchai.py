"""The TrialMatchAI L4 Systems in the public release: their run options, their run, and the receipt
that identifies a prepared corpus.

Only the three TrialMatchAI L4 pipelines own this module; see :mod:`taim.pipeline_extensions`.

A run's System Input is location-free, so its Result Bundle publishes. The release layer here
resolves and verifies the physical inputs (the pinned checkout, the workspace, the prepared corpus
folders and the search database) and gives them to the System as
:class:`taim.adapters.trialmatchai.TrialMatchAIHarnessInputs`, beside the System Input and never
hashed into its ``system_input_id``. The timeout travels the same way: it aborts a run, so it
cannot change a completed ranking. What the System Input carries instead:

- the command tokens, refused when one is an absolute filesystem path, before any run starts;
- the pinned upstream commit, repository and runtime contract;
- the content digest from the prepared corpus's preparation receipt. A corpus without a receipt is
  refused. ``trial-benchmark patient-to-trial trialmatchai adopt-corpus`` writes one, hashing
  every prepared file once. At run start the receipt is bound, and the corpus's file names and
  counts are checked against it by a directory walk; the corpus is not hashed again.

The search database the run searches is hashed from the directory the run is given (on a cluster,
the staged node-local copy) at every run start. Its digest is recorded in the index identity, and
the hash's wall time in the run record.

The Result Bundle validator's refusal of absolute filesystem paths is unchanged: a location that
slips back into the System Input is still refused there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from taim.pipeline_extensions import PipelineExtension

if TYPE_CHECKING:
    from taim.release_profiles import ResolvedBenchmark
    from taim.system_contracts import SystemRunRequest, SystemRunResult

_RECEIPT_FILENAME = "trialmatchai-corpus-receipt.json"
_RUN_START_RECORD_FILENAME = "run-start-verification.json"
_RECEIPT_KIND = "trialmatchai-prepared-corpus-receipt"
_RECEIPT_SCHEMA_VERSION = "1.0"
_CORPUS_ROLES = ("processed_trials", "processed_criteria")
_DIGEST_RULE = (
    "SHA-256 over one canonical JSON line per regular file, sorted by role then POSIX path "
    "relative to the role folder: [role, path, file SHA-256] for content_sha256 and [role, path] "
    "for names_sha256"
)
_SEARCH_DATABASE_DIGEST_RULE = (
    "SHA-256 over one canonical JSON line per regular file, sorted by POSIX path relative to the "
    "search database folder: [path, file SHA-256]"
)
_DEFAULT_HASH_WORKERS = 8


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trialmatchai-cwd", type=Path)
    parser.add_argument("--trialmatchai-workspace", type=Path)
    parser.add_argument("--trialmatchai-corpus-dir", type=Path)
    parser.add_argument("--trialmatchai-search-db-path", type=Path)
    parser.add_argument("--trialmatchai-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--trialmatchai-command", nargs="+")


def _run_options(
    args: argparse.Namespace,
    benchmark: ResolvedBenchmark,
    task_input_id: str,
) -> dict[str, object]:
    """The physical run options. :func:`_run` turns them into harness inputs and a System Input."""

    from taim.adapters.trialmatchai import (
        TRIALMATCHAI_L4_COMMIT,
        TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH,
        TRIALMATCHAI_L4_REPOSITORY,
        trialmatchai_l4_runtime_contract,
    )

    if args.trialmatchai_cwd is None or args.trialmatchai_corpus_dir is None:
        raise ValueError("TrialMatchAI requires --trialmatchai-cwd and --trialmatchai-corpus-dir")
    corpus = args.trialmatchai_corpus_dir.resolve()
    command = tuple(args.trialmatchai_command or ()) or (
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


def _canonical_line(row: Sequence[str]) -> bytes:
    return (json.dumps(list(row), ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _regular_files(root: Path, *, role: str) -> Iterator[str]:
    """Every regular file under ``root``, as a POSIX path relative to it, walked by readdir.

    ``os.scandir`` reports each entry's type from the directory listing, so the walk stats no
    file. Anything that is neither a directory nor a regular file is refused: a receipt names
    regular files only.
    """

    pending = [""]
    while pending:
        relative = pending.pop()
        with os.scandir(root / relative if relative else root) as entries:
            for entry in entries:
                name = f"{relative}/{entry.name}" if relative else entry.name
                if entry.is_dir(follow_symlinks=False):
                    pending.append(name)
                elif entry.is_file(follow_symlinks=False):
                    yield name
                else:
                    raise ValueError(
                        f"TrialMatchAI {role} holds an entry that is neither a folder nor a "
                        f"regular file: {name}"
                    )


def _role_folders(corpus: Path) -> dict[str, Path]:
    folders = {role: corpus / role for role in _CORPUS_ROLES}
    missing = [role for role, folder in folders.items() if not folder.is_dir()]
    if missing:
        raise ValueError(
            "--trialmatchai-corpus-dir must contain processed_trials/ and processed_criteria/"
        )
    return folders


def _corpus_names(corpus: Path) -> list[tuple[str, str]]:
    return sorted(
        (role, name)
        for role, folder in _role_folders(corpus).items()
        for name in _regular_files(folder, role=role)
    )


def _names_sha256(rows: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_line(row))
    return f"sha256:{digest.hexdigest()}"


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _hashed(paths: Sequence[Path], workers: int) -> list[tuple[str, int]]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(_hash_file, paths, chunksize=64))


def _count_by_role(rows: Sequence[tuple[str, str]]) -> dict[str, int]:
    return {role: sum(1 for row_role, _name in rows if row_role == role) for role in _CORPUS_ROLES}


def _write_prepared_corpus_receipt(
    corpus: Path, *, workers: int = _DEFAULT_HASH_WORKERS
) -> dict[str, object]:
    """Hash every prepared file once and write the corpus's location-free preparation receipt."""

    from taim.contracts import content_sha256

    corpus = corpus.resolve()
    receipt_path = corpus / _RECEIPT_FILENAME
    if receipt_path.exists():
        raise ValueError(f"TrialMatchAI prepared corpus already has a receipt: {receipt_path}")
    folders = _role_folders(corpus)
    rows = _corpus_names(corpus)
    hashed = _hashed([folders[role] / name for role, name in rows], workers)
    content = hashlib.sha256()
    byte_counts = dict.fromkeys(_CORPUS_ROLES, 0)
    for (role, name), (file_sha256, size) in zip(rows, hashed, strict=True):
        content.update(_canonical_line((role, name, file_sha256)))
        byte_counts[role] += size
    file_counts = _count_by_role(rows)
    core: dict[str, object] = {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "kind": _RECEIPT_KIND,
        "digest_rule": _DIGEST_RULE,
        "roles": {
            role: {"file_count": file_counts[role], "byte_count": byte_counts[role]}
            for role in _CORPUS_ROLES
        },
        "names_sha256": _names_sha256(rows),
        "content_sha256": f"sha256:{content.hexdigest()}",
    }
    receipt = {**core, "receipt_id": content_sha256(core)}
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return receipt


def _load_prepared_corpus_receipt(corpus: Path) -> dict[str, object]:
    from taim.contracts import content_sha256

    receipt_path = corpus / _RECEIPT_FILENAME
    if not receipt_path.is_file():
        raise ValueError(
            f"TrialMatchAI prepared corpus has no preparation receipt: {receipt_path}. Write one "
            "once with: trial-benchmark patient-to-trial trialmatchai adopt-corpus --corpus-dir "
            f"{corpus}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"TrialMatchAI preparation receipt is not readable JSON: {exc}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("kind") != _RECEIPT_KIND
        or receipt.get("schema_version") != _RECEIPT_SCHEMA_VERSION
        or receipt.get("digest_rule") != _DIGEST_RULE
    ):
        raise ValueError("TrialMatchAI preparation receipt is not a version 1.0 receipt")
    core = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if receipt.get("receipt_id") != content_sha256(core):
        raise ValueError("TrialMatchAI preparation receipt does not match its own receipt_id")
    return receipt


def _check_prepared_corpus_names(corpus: Path, receipt: dict[str, object]) -> None:
    """Bind a run to its receipt: the corpus holds exactly the receipted names, without hashing."""

    rows = _corpus_names(corpus)
    observed = _count_by_role(rows)
    roles = receipt.get("roles")
    for role in _CORPUS_ROLES:
        entry = roles.get(role) if isinstance(roles, dict) else None
        expected = entry.get("file_count") if isinstance(entry, dict) else None
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError(f"TrialMatchAI preparation receipt has no {role} file count")
        if observed[role] < expected:
            raise ValueError(
                f"TrialMatchAI prepared corpus is missing {expected - observed[role]} {role} "
                "file(s) its preparation receipt names"
            )
        if observed[role] > expected:
            raise ValueError(
                f"TrialMatchAI prepared corpus has {observed[role] - expected} {role} file(s) "
                "its preparation receipt does not name"
            )
    if _names_sha256(rows) != receipt.get("names_sha256"):
        raise ValueError(
            "TrialMatchAI prepared corpus file names differ from its preparation receipt "
            "(a file was renamed)"
        )


def _search_database_identity(search_db: Path, *, workers: int) -> dict[str, object]:
    """Hash the search database the run is given, as it is at run start."""

    if not search_db.is_dir():
        raise ValueError(f"TrialMatchAI search database does not exist: {search_db}")
    names = sorted(_regular_files(search_db, role="search database"))
    if not names:
        raise ValueError(f"TrialMatchAI search database holds no files: {search_db}")
    hashed = _hashed([search_db / name for name in names], workers)
    digest = hashlib.sha256()
    for name, (file_sha256, _size) in zip(names, hashed, strict=True):
        digest.update(_canonical_line((name, file_sha256)))
    return {
        "digest_rule": _SEARCH_DATABASE_DIGEST_RULE,
        "content_sha256": f"sha256:{digest.hexdigest()}",
        "file_count": len(names),
        "byte_count": sum(size for _digest, size in hashed),
    }


def _refuse_absolute_command_tokens(command: Sequence[str]) -> None:
    from taim.release_result_bundle import _is_absolute_filesystem_path

    absolute = [index for index, token in enumerate(command) if _is_absolute_filesystem_path(token)]
    if absolute:
        raise ValueError(
            "TrialMatchAI command tokens "
            + ", ".join(f"command[{index}]" for index in absolute)
            + " are absolute filesystem paths; name the executable, and let the checkout and "
            "corpus roles ({input}, {processed_trials}, {processed_criteria}) carry locations"
        )


def _physical_path(request: SystemRunRequest, name: str) -> Path:
    value = request.options.get(name)
    if not isinstance(value, Path):
        raise ValueError(f"TrialMatchAI release run requires the {name} run option")
    return value


def _located_request(
    system_class: Callable[..., object],
    request: SystemRunRequest,
    *,
    hash_workers: int = _DEFAULT_HASH_WORKERS,
) -> tuple[object, SystemRunRequest, dict[str, object], dict[str, object]]:
    """Verify the physical inputs, and build the System, its System Input and the run-start record.

    Returns the System holding its harness inputs, the location-free request, the search
    database's identity and the run-start timings. Cheap refusals come first: the command, the
    checkout, the corpus folders and its receipt, then the names walk, then the database hash.
    """

    from taim.adapters.trialmatchai import (
        TRIALMATCHAI_PREPARED_CORPUS_OPTION,
        TrialMatchAIHarnessInputs,
    )

    command = request.options.get("command")
    if not isinstance(command, tuple | list) or any(
        not isinstance(token, str) or not token for token in command
    ):
        raise ValueError("TrialMatchAI release run requires a command of non-empty tokens")
    _refuse_absolute_command_tokens(command)
    cwd = _physical_path(request, "cwd")
    if not cwd.is_dir():
        raise ValueError(f"TrialMatchAI checkout does not exist: {cwd}")
    corpus = _physical_path(request, "corpus_dir")
    folders = _role_folders(corpus)
    receipt = _load_prepared_corpus_receipt(corpus)
    started = time.perf_counter()
    _check_prepared_corpus_names(corpus, receipt)
    names_seconds = time.perf_counter() - started
    started = time.perf_counter()
    search_database = _search_database_identity(
        _physical_path(request, "search_db_path"), workers=hash_workers
    )
    search_database_seconds = time.perf_counter() - started
    timeout = request.options.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int | float):
        raise ValueError("TrialMatchAI release run requires a numeric timeout_seconds")
    harness_inputs = TrialMatchAIHarnessInputs(
        cwd=cwd,
        workspace=_physical_path(request, "workspace"),
        processed_trials_folder=folders["processed_trials"],
        processed_criteria_folder=folders["processed_criteria"],
        search_db_path=_physical_path(request, "search_db_path"),
        timeout_seconds=float(timeout),
    )
    identity_options = {
        name: request.options[name]
        for name in (
            "trialmatchai_expected_commit",
            "trialmatchai_expected_repository",
            "trialmatchai_runtime_contract",
        )
    }
    located = replace(
        request,
        options={
            **identity_options,
            "command": tuple(command),
            TRIALMATCHAI_PREPARED_CORPUS_OPTION: receipt["content_sha256"],
        },
    )
    run_start = {
        "prepared_corpus_receipt_id": receipt["receipt_id"],
        "prepared_corpus_names_check_seconds": names_seconds,
        "search_database_hash_seconds": search_database_seconds,
        "search_database_hash_workers": hash_workers,
    }
    return system_class(harness_inputs=harness_inputs), located, search_database, run_start


def _run(system_id: str, request: SystemRunRequest) -> SystemRunResult:
    from taim.adapters.trialmatchai import (
        TRIALMATCHAI_L4_SYSTEM_ID,
        TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID,
        TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID,
        TrialMatchAIL4System,
        TrialMatchAIL4TREC2022System,
        TrialMatchAIL4TREC2023System,
    )
    from taim.release_reference_systems import run_normalized_system
    from taim.system_contracts import normalize_system_request

    trialmatchai_systems = {
        TRIALMATCHAI_L4_SYSTEM_ID: TrialMatchAIL4System,
        TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID: TrialMatchAIL4TREC2022System,
        TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID: TrialMatchAIL4TREC2023System,
    }
    try:
        system_class = trialmatchai_systems[system_id]
    except KeyError as exc:
        raise ValueError(
            f"System {system_id!r} is not an exported direct reference System"
        ) from exc
    system, located, search_database, run_start = _located_request(system_class, request)
    effective = normalize_system_request(system, located)  # type: ignore[arg-type]
    # Written before the System starts, so a run that stops later still shows what its start
    # verified. The workspace is the run's private working record; nothing here is published.
    workspace = _physical_path(request, "workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / _RUN_START_RECORD_FILENAME).write_text(
        json.dumps(
            {
                "system_input_id": effective.system_input_id,
                "trialmatchai_prepared_corpus_sha256": located.options[
                    "trialmatchai_prepared_corpus_sha256"
                ],
                "search_database": search_database,
                **run_start,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = run_normalized_system(system, located)  # type: ignore[arg-type]
    index_identity = dict(result.configuration["index_identity"])
    index_identity["search_database"] = search_database
    return replace(
        result,
        configuration={
            **result.configuration,
            "index_identity": index_identity,
            "run_start_verification": run_start,
        },
    )


def _add_patient_to_trial_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    from taim.release_cli import _positive_integer

    trialmatchai = commands.add_parser(
        "trialmatchai", help="identify a prepared TrialMatchAI corpus"
    )
    trialmatchai_commands = trialmatchai.add_subparsers(
        dest="trialmatchai_operation", required=True
    )
    adopt = trialmatchai_commands.add_parser(
        "adopt-corpus",
        help="hash a prepared corpus once and write its preparation receipt beside its folders",
    )
    adopt.add_argument("--corpus-dir", type=Path, required=True)
    adopt.add_argument("--workers", type=_positive_integer, default=_DEFAULT_HASH_WORKERS)
    adopt.set_defaults(handler=_adopt_corpus)


def _adopt_corpus(args: argparse.Namespace) -> int:
    import taim.release_cli as release_cli

    started = time.perf_counter()
    receipt = _write_prepared_corpus_receipt(args.corpus_dir, workers=args.workers)
    release_cli._print(
        {
            "receipt": str(args.corpus_dir.resolve() / _RECEIPT_FILENAME),
            "receipt_id": receipt["receipt_id"],
            "content_sha256": receipt["content_sha256"],
            "roles": receipt["roles"],
            "seconds": round(time.perf_counter() - started, 3),
        }
    )
    return 0


EXTENSION = PipelineExtension(
    system_tasks={
        "trialmatchai-current-cuda-l4-trec21-development-v3": "patient_to_trial",
        "trialmatchai-current-cuda-l4-trec22-development-v3": "patient_to_trial",
        "trialmatchai-current-cuda-l4-trec23-development-v3": "patient_to_trial",
    },
    pipeline_depth="post_eligibility",
    external_baseline=True,
    subcommands={},
    run=_run,
    add_run_options=_add_run_options,
    run_options=_run_options,
    add_patient_to_trial_commands=_add_patient_to_trial_commands,
    module_exports={
        "src/taim/adapters/__init__.py": (
            "TrialMatchAIL4System",
            "TrialMatchAIL4TREC2022System",
            "TrialMatchAIL4TREC2023System",
        ),
        "src/taim/adapters/trialmatchai.py": (
            "PATIENT_RENDERER_VERSION",
            "TRIALMATCHAI_L4_ADAPTER_SHA256",
            "TRIALMATCHAI_L4_ADAPTER_WEIGHT_FILENAME",
            "TRIALMATCHAI_L4_COMMIT",
            "TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH",
            "TRIALMATCHAI_L4_CONFIG_SHA256",
            "TRIALMATCHAI_L4_FORBIDDEN_ENVIRONMENT",
            "TRIALMATCHAI_L4_PROCESS_EXECUTION_MODE",
            "TRIALMATCHAI_L4_REPOSITORY",
            "TRIALMATCHAI_L4_REQUIRED_ENVIRONMENT",
            "TRIALMATCHAI_L4_SYSTEM_ID",
            "TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID",
            "TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID",
            "TrialMatchAIAdapterError",
            "TrialMatchAIHarnessInputs",
            "TrialMatchAIL4System",
            "TrialMatchAIL4TREC2022System",
            "TrialMatchAIL4TREC2023System",
            "TrialMatchAIResult",
            "load_and_normalize_ranked_trials",
            "normalize_ranked_trials",
            "render_patient_input",
            "run_trialmatchai_command",
            "trialmatchai_l4_runtime_contract",
            "verify_trialmatchai_l4_adapter_weights",
            "write_patient_input",
        ),
    },
    forbidden_classes=("TrialMatchAISystem",),
    protected_reference_paths=(
        "docs/external-baselines.md",
        "docs/trialmatchai-external-baseline.md",
        "src/taim/adapters/__init__.py",
        "src/taim/adapters/trialmatchai.py",
        "src/taim/pipeline_extensions/trialmatchai.py",
        "tests/test_trialmatchai_external_baselines.py",
    ),
    dependency_declarations=("src/taim/data/external-system-dependencies-v1.json",),
)
