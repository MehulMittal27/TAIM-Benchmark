"""The TrialMatchAI run in the public release, shared by both families: its run options, the
verification of its physical inputs, the run itself, and the ``trialmatchai`` command group with
the corpus commands every TrialMatchAI System uses.

A run's System Input is location-free, so its Result Bundle publishes. The release layer here
resolves and verifies the physical inputs (the pinned checkout, the workspace, the prepared corpus
folders and the search database) and gives them to the System as
:class:`taim.pipelines.trialmatchai.corpus.TrialMatchAIHarnessInputs`, beside the System Input
and never hashed into its ``system_input_id``. The timeout aborts a run,
so it cannot change a completed ranking. What the System Input carries instead:

- the command tokens, refused when one is an absolute filesystem path, before any run starts;
- the pinned upstream commit, repository and runtime contract;
- the content digest from the prepared corpus's preparation receipt. A corpus without a receipt is
  refused (see :mod:`taim.pipelines.trialmatchai.corpus`).

The search database's digest is recorded in the index identity, and the hash's wall time in the
run record.

The Result Bundle validator's refusal of absolute filesystem paths is unchanged: a location that
slips back into the System Input is still refused there.

Both families add to one ``trialmatchai`` command group. :func:`trialmatchai_command_group`
creates it, with ``filter-corpus`` and ``adopt-corpus``, the first time it is called on a parser
and returns it every later time, so a family adds only its own commands to it.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from taim.pipelines.trialmatchai.corpus import (
    TRIALMATCHAI_CORPUS_RECEIPT_FILENAME,
    TRIALMATCHAI_HASH_WORKERS,
    TRIALMATCHAI_PREPARED_CORPUS_OPTION,
    TrialMatchAIHarnessInputs,
    check_trialmatchai_corpus_names,
    filter_trialmatchai_corpus,
    load_trialmatchai_corpus_receipt,
    trialmatchai_corpus_folders,
    trialmatchai_search_database_identity,
    write_trialmatchai_corpus_receipt,
)

if TYPE_CHECKING:
    from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialRunResult


_RUN_START_RECORD_FILENAME = "run-start-verification.json"


def trialmatchai_add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trialmatchai-cwd",
        type=Path,
        help="the pinned TrialMatchAI checkout the command runs in; required by TrialMatchAI",
    )
    parser.add_argument(
        "--trialmatchai-workspace",
        type=Path,
        help="an empty working directory; omitted, .trialmatchai/<run ID> under --output-dir",
    )
    parser.add_argument(
        "--trialmatchai-corpus-dir",
        type=Path,
        help="prepared corpus holding processed_trials/, processed_criteria/ and the receipt "
        "trialmatchai adopt-corpus wrote; required by TrialMatchAI",
    )
    parser.add_argument(
        "--trialmatchai-search-db-path",
        type=Path,
        help="search database, hashed at every run start; omitted, search/ in the corpus",
    )
    parser.add_argument(
        "--trialmatchai-timeout-seconds",
        type=float,
        default=3600.0,
        help="wall-time limit of one TrialMatchAI invocation",
    )
    parser.add_argument(
        "--trialmatchai-command",
        nargs="+",
        help="command that runs TrialMatchAI, with {input}, {processed_trials} and "
        "{processed_criteria} placeholders; name the executable bare, since an absolute "
        "location is refused. Omitted, the pinned end-to-end command",
    )


def _refuse_absolute_command_tokens(command: Sequence[str]) -> None:
    from taim.release_result_bundle import is_absolute_filesystem_path

    absolute = [index for index, token in enumerate(command) if is_absolute_filesystem_path(token)]
    if absolute:
        raise ValueError(
            "TrialMatchAI command tokens "
            + ", ".join(f"command[{index}]" for index in absolute)
            + " are absolute filesystem paths; name the executable, and let the checkout and "
            "corpus roles ({input}, {processed_trials}, {processed_criteria}) carry locations"
        )


def _physical_path(request: PatientToTrialRunRequest, name: str) -> Path:
    value = request.options.get(name)
    if not isinstance(value, Path):
        raise ValueError(f"TrialMatchAI release run requires the {name} run option")
    return value


def _located_request(
    system_class: Callable[..., object],
    request: PatientToTrialRunRequest,
    *,
    hash_workers: int = TRIALMATCHAI_HASH_WORKERS,
) -> tuple[object, PatientToTrialRunRequest, dict[str, object], dict[str, object]]:
    """Verify the physical inputs, and build the System, its System Input and the run-start record.

    Returns the System holding its harness inputs, the location-free request, the search
    database's identity and the run-start timings. Cheap refusals come first: the command, the
    checkout, the corpus folders and its receipt, then the names walk, then the database hash.
    """

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
    folders = trialmatchai_corpus_folders(corpus)
    receipt = load_trialmatchai_corpus_receipt(corpus)
    started = time.perf_counter()
    check_trialmatchai_corpus_names(corpus, receipt)
    names_seconds = time.perf_counter() - started
    started = time.perf_counter()
    search_database = trialmatchai_search_database_identity(
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


def run_system(system_class, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
    from taim.patient_to_trial import normalize_system_request
    from taim.release_reference_systems import run_normalized_system

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


def trialmatchai_command_group(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse._SubParsersAction[argparse.ArgumentParser]:
    """The ``trialmatchai`` group's subcommands, created with the corpus commands at the first call.

    Each family's extension adds the group; the release CLI calls each distinct adder once, so the
    group is looked up before it is created.
    """

    from taim.cli_support import positive_integer

    existing = commands.choices.get("trialmatchai")
    if existing is not None:
        return next(
            action for action in existing._actions if isinstance(action, argparse._SubParsersAction)
        )
    trialmatchai = commands.add_parser(
        "trialmatchai",
        help="cut or identify a TrialMatchAI prepared corpus, or build or check what the installed "
        "TrialMatchAI Systems need before a run",
    )
    trialmatchai_commands = trialmatchai.add_subparsers(
        dest="trialmatchai_operation", required=True
    )
    filter_corpus = trialmatchai_commands.add_parser(
        "filter-corpus",
        help="cut a TrialMatchAI-prepared corpus to exactly a Snapshot's trial IDs, linking its "
        "prepared files, and write filter-manifest.json",
    )
    filter_corpus.add_argument(
        "--snapshot-trials",
        type=Path,
        required=True,
        help="Snapshot trial records, JSON Lines with a trial_id on each",
    )
    filter_corpus.add_argument(
        "--source-trials",
        type=Path,
        required=True,
        help="TrialMatchAI's processed_trials/, one JSON file per trial named by its trial ID",
    )
    filter_corpus.add_argument(
        "--source-criteria",
        type=Path,
        required=True,
        help="TrialMatchAI's processed_criteria/, one folder per trial named by its trial ID",
    )
    filter_corpus.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new corpus folder that receives processed_trials/, processed_criteria/ and the "
        "manifest",
    )
    filter_corpus.add_argument(
        "--reuse-existing",
        action="store_true",
        help="complete an existing output, refusing any file whose contents differ",
    )
    filter_corpus.set_defaults(handler=_filter_corpus)
    adopt = trialmatchai_commands.add_parser(
        "adopt-corpus",
        help="hash a prepared corpus once and write its preparation receipt beside its folders",
    )
    adopt.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="prepared corpus holding processed_trials/ and processed_criteria/",
    )
    adopt.add_argument(
        "--workers",
        type=positive_integer,
        default=TRIALMATCHAI_HASH_WORKERS,
        help="files hashed concurrently",
    )
    adopt.set_defaults(handler=_adopt_corpus)
    return trialmatchai_commands


def _adopt_corpus(args: argparse.Namespace) -> int:
    from taim.cli_support import print_payload

    started = time.perf_counter()
    receipt = write_trialmatchai_corpus_receipt(args.corpus_dir, workers=args.workers)
    print_payload(
        {
            "receipt": str(args.corpus_dir.resolve() / TRIALMATCHAI_CORPUS_RECEIPT_FILENAME),
            "receipt_id": receipt["receipt_id"],
            "content_sha256": receipt["content_sha256"],
            "roles": receipt["roles"],
            "seconds": round(time.perf_counter() - started, 3),
        }
    )
    return 0


def _filter_corpus(args: argparse.Namespace) -> int:
    from taim.cli_support import print_payload

    print_payload(
        filter_trialmatchai_corpus(
            snapshot_trials=args.snapshot_trials,
            source_trials=args.source_trials,
            source_criteria=args.source_criteria,
            output=args.output,
            allow_existing=args.reuse_existing,
        )
    )
    return 0
