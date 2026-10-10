"""The TrialMatchAI harness both families share: the error type, the harness inputs, the pinned
checkout's verification, the prepared corpus and its receipt, the search database's identity, and
the normalisation of a ranked-trials artifact into TAIM Candidates.

The fork's L4 Systems and the upstream v0.9.1 System both run TrialMatchAI over a prepared corpus
from a pinned checkout, so both use what is here. Nothing here imports either family, the concept
store, or a release integration module, so either family can ship without the other.

A run refuses a corpus whose trial IDs differ from its Snapshot's.
``trial-benchmark patient-to-trial trialmatchai filter-corpus`` cuts TrialMatchAI's prepared
corpus to them with :func:`filter_trialmatchai_corpus`, linking the prepared files, and writes
``filter-manifest.json``, whose digest a run records in its index identity.

A prepared corpus is identified by its preparation receipt.
``trial-benchmark patient-to-trial trialmatchai adopt-corpus`` writes one, hashing every prepared
file once. At run start the receipt is bound, and the corpus's file names and counts are checked
against it by a directory walk; the corpus is not hashed again.

The search database the run searches is hashed from the directory the run is given (on a cluster,
the staged node-local copy) at every run start, by :func:`trialmatchai_search_database_identity`.

This module deliberately does not import TrialMatchAI. The upstream checkout is an external system
and is invoked as a pinned subprocess.
"""

from __future__ import annotations

import filecmp
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from taim.artifacts import sha256_file
from taim.executables import ExecutableNotFoundError, resolve_executable
from taim.schemas import Candidate

if TYPE_CHECKING:
    from taim.patient_to_trial import PatientToTrialRunRequest


class TrialMatchAIAdapterError(ValueError):
    """Raised when an upstream TrialMatchAI artifact is invalid."""


#: The System options that say where a run's inputs are or how long it may take. None of them
#: can change a completed ranking, so a harness that supplies them as
#: :class:`TrialMatchAIHarnessInputs` keeps them out of the System Input and its identity.
TRIALMATCHAI_HARNESS_OPTIONS = frozenset(
    {
        "cwd",
        "workspace",
        "timeout_seconds",
        "processed_trials_folder",
        "processed_criteria_folder",
        "search_db_path",
    }
)


#: The System option that binds the prepared corpus by content when its folders are harness inputs.
TRIALMATCHAI_PREPARED_CORPUS_OPTION = "trialmatchai_prepared_corpus_sha256"


@dataclass(frozen=True, slots=True)
class TrialMatchAIHarnessInputs:
    """Where one run's physical inputs are, supplied by the harness beside the System Input.

    The harness resolves and verifies these before it builds the System Input, which then carries
    the prepared corpus's content digest instead of its folders. Nothing here is hashed into the
    ``system_input_id``: two runs of the same inputs from different directories, under different
    run ids or timeouts, are the same System Input.
    """

    cwd: Path
    workspace: Path
    processed_trials_folder: Path
    processed_criteria_folder: Path
    search_db_path: Path
    timeout_seconds: float


def git_checkout_output(cwd: Path, *arguments: str) -> str:
    try:
        git = resolve_executable("git")
        completed = subprocess.run(  # noqa: S603 - resolved executable and fixed git argv
            [git, *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            # The checkout is a blob:none partial clone: an object read must never fetch.
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
    except (OSError, ExecutableNotFoundError, subprocess.CalledProcessError) as exc:
        raise TrialMatchAIAdapterError(
            f"cannot inspect the TrialMatchAI Git checkout at {cwd}"
        ) from exc
    return completed.stdout.strip()


def _command_config_path(command: Sequence[str], *, cwd: Path) -> Path | None:
    config_paths: list[Path] = []
    index = 0
    while index < len(command):
        argument = command[index]
        option_name = argument.partition("=")[0]
        if (
            len(option_name) > 2
            and option_name != "--config"
            and "--config".startswith(option_name)
        ):
            raise TrialMatchAIAdapterError(
                f"TrialMatchAI command uses abbreviated --config option {option_name!r}"
            )
        if argument == "--config":
            if index + 1 >= len(command):
                raise TrialMatchAIAdapterError("TrialMatchAI --config is missing its path")
            config_paths.append(Path(command[index + 1]))
            index += 2
            continue
        elif argument.startswith("--config="):
            path = Path(argument.partition("=")[2])
            if not str(path):
                raise TrialMatchAIAdapterError("TrialMatchAI --config is missing its path")
            config_paths.append(path)
        index += 1
    if len(config_paths) > 1:
        raise TrialMatchAIAdapterError("TrialMatchAI command contains duplicate --config options")
    if not config_paths:
        return None
    path = config_paths[0]
    return (cwd / path).resolve() if not path.is_absolute() else path.resolve()


def trialmatchai_checkout_configuration(
    cwd: Path,
    command: Sequence[str],
    *,
    expected_commit: str,
    expected_repository: str,
    expected_config_sha256: str | None = None,
) -> dict[str, object]:
    """Verify and record the exact clean upstream source and explicit config."""

    actual_commit = git_checkout_output(cwd, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI checkout expected commit {expected_commit}, found {actual_commit}"
        )
    actual_repository = git_checkout_output(cwd, "remote", "get-url", "origin")
    if actual_repository != expected_repository:
        raise TrialMatchAIAdapterError(
            "TrialMatchAI checkout expected origin "
            f"{expected_repository}, found {actual_repository}"
        )
    if git_checkout_output(cwd, "status", "--porcelain", "--untracked-files=all"):
        raise TrialMatchAIAdapterError(f"TrialMatchAI checkout has local changes: {cwd}")

    configuration: dict[str, object] = {
        "repository": actual_repository,
        "commit": actual_commit,
        "tree": git_checkout_output(cwd, "rev-parse", "HEAD^{tree}"),
        "dirty": False,
    }
    config_path = _command_config_path(command, cwd=cwd)
    if expected_config_sha256 is not None and config_path is None:
        raise TrialMatchAIAdapterError(
            "TrialMatchAI expected config "
            f"{expected_config_sha256}, but the command has no --config"
        )
    if config_path is not None:
        if not config_path.is_file():
            raise TrialMatchAIAdapterError(f"TrialMatchAI config does not exist: {config_path}")
        config_sha256 = sha256_file(config_path)
        if expected_config_sha256 is not None and config_sha256 != expected_config_sha256:
            raise TrialMatchAIAdapterError(
                "TrialMatchAI expected config "
                f"{expected_config_sha256}, found {config_sha256} at {config_path}"
            )
        configuration.update(
            {
                "config_path": str(config_path),
                "config_sha256": config_sha256,
            }
        )
    return configuration


#: Sidecar written by the Snapshot corpus filter beside the
#: ``processed_trials`` and ``processed_criteria`` folders it derives from a Snapshot.
TRIALMATCHAI_CORPUS_FILTER_MANIFEST_FILENAME = "filter-manifest.json"


def validate_snapshot_corpus(
    request: PatientToTrialRunRequest, processed_trials_folder: Path
) -> None:
    if not processed_trials_folder.is_dir():
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI processed trials directory does not exist: {processed_trials_folder}"
        )
    expected_ids = {trial.trial_id for trial in request.snapshot.trials}
    actual_ids = {path.stem for path in processed_trials_folder.glob("*.json")}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        detail = []
        if missing:
            detail.append(f"missing {missing[0]}")
        if unexpected:
            detail.append(f"unexpected {unexpected[0]}")
        raise TrialMatchAIAdapterError(
            "TrialMatchAI corpus does not exactly match the TAIM Snapshot trial IDs"
            + (f" ({'; '.join(detail)})" if detail else "")
        )


def without_locators(value: object) -> object:
    """Drop every key that names a location, at any depth, as the public projection does."""

    if isinstance(value, Mapping):
        return {
            key: without_locators(nested)
            for key, nested in value.items()
            if not (key in {"directory", "path"} or key.endswith(("_dir", "_path")))
        }
    if isinstance(value, (list, tuple)):
        return [without_locators(nested) for nested in value]
    return value


def corpus_filter_manifest_sha256(processed_trials_folder: Path | None) -> str | None:
    if processed_trials_folder is None:
        return None
    manifest = processed_trials_folder.parent / TRIALMATCHAI_CORPUS_FILTER_MANIFEST_FILENAME
    return sha256_file(manifest) if manifest.is_file() else None


@dataclass(frozen=True, slots=True)
class TrialMatchAIResult:
    """Normalized final and diagnostic rankings from one TrialMatchAI topic."""

    candidates: tuple[Candidate, ...]
    diagnostic_rankings: dict[str, tuple[Candidate, ...]]
    raw_payload: Mapping[str, Any]


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrialMatchAIAdapterError(f"{name} must be a non-empty string")
    return value


def _finite_score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrialMatchAIAdapterError(f"{name} must be a number")
    if not math.isfinite(value):
        raise TrialMatchAIAdapterError(f"{name} must be finite")
    return float(value)


def _ranked_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("RankedTrials")
    if not isinstance(rows, list):
        raise TrialMatchAIAdapterError("ranked_trials.json must contain a RankedTrials array")
    result: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TrialMatchAIAdapterError(f"RankedTrials[{index}] must be an object")
        result.append(row)
    return result


def _candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    system_id: str,
    topic_id: str,
    score_field: str,
) -> tuple[Candidate, ...]:
    parsed: list[tuple[str, float]] = []
    for index, row in enumerate(rows):
        trial_id = _non_empty_string(row.get("TrialID"), f"RankedTrials[{index}].TrialID")
        score = _finite_score(row.get(score_field), f"RankedTrials[{index}].{score_field}")
        parsed.append((trial_id, score))
    if len({trial_id for trial_id, _score in parsed}) != len(parsed):
        raise TrialMatchAIAdapterError("RankedTrials contains duplicate TrialID values")

    # The upstream score is retained, but rank assignment is owned by TAIM:
    # score descending followed by trial ID ascending for deterministic ties.
    parsed.sort(key=lambda item: (-item[1], item[0]))
    return tuple(
        Candidate(run_id, system_id, topic_id, trial_id, rank, score)
        for rank, (trial_id, score) in enumerate(parsed, start=1)
    )


def normalize_ranked_trials(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    topic_id: str,
    system_id: str,
) -> TrialMatchAIResult:
    """Normalize TrialMatchAI final and available diagnostic rankings.

    ``ranked_trials.json`` is the final post-eligibility output.  The optional
    ``first_level_scores`` mapping is accepted as a diagnostic retrieval stage;
    it is never silently treated as the Primary Ranking.
    """

    run_id = _non_empty_string(run_id, "run_id")
    topic_id = _non_empty_string(topic_id, "topic_id")
    final_rows = _ranked_rows(payload)
    candidates = _candidate_rows(
        final_rows,
        run_id=run_id,
        system_id=system_id,
        topic_id=topic_id,
        score_field="Score",
    )

    diagnostics: dict[str, tuple[Candidate, ...]] = {}
    first_level = payload.get("first_level_scores")
    if first_level is not None:
        if not isinstance(first_level, Mapping):
            raise TrialMatchAIAdapterError("first_level_scores must be an object")
        first_rows = [
            {"TrialID": trial_id, "Score": score} for trial_id, score in first_level.items()
        ]
        diagnostics["retrieval"] = _candidate_rows(
            first_rows,
            run_id=run_id,
            system_id=system_id,
            topic_id=topic_id,
            score_field="Score",
        )

    return TrialMatchAIResult(
        candidates=candidates,
        diagnostic_rankings=diagnostics,
        raw_payload=payload,
    )


def load_and_normalize_ranked_trials(
    path: str | Path,
    *,
    run_id: str,
    topic_id: str,
    system_id: str,
    first_level_scores_path: str | Path | None = None,
) -> TrialMatchAIResult:
    """Load a JSON artifact and normalize it without executing upstream code."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrialMatchAIAdapterError(f"cannot read ranked TrialMatchAI artifact: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TrialMatchAIAdapterError("ranked_trials.json must contain an object")
    if first_level_scores_path is not None:
        scores_path = Path(first_level_scores_path)
        try:
            first_level_scores = json.loads(scores_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrialMatchAIAdapterError(
                f"cannot read TrialMatchAI first-level scores artifact: {exc}"
            ) from exc
        if not isinstance(first_level_scores, Mapping):
            raise TrialMatchAIAdapterError("first_level_scores.json must contain an object")
        payload = dict(payload)
        payload["first_level_scores"] = first_level_scores
    return normalize_ranked_trials(
        payload,
        run_id=run_id,
        topic_id=topic_id,
        system_id=system_id,
    )


TRIALMATCHAI_CORPUS_RECEIPT_FILENAME = "trialmatchai-corpus-receipt.json"
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
TRIALMATCHAI_HASH_WORKERS = 8


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


def trialmatchai_corpus_folders(corpus: Path) -> dict[str, Path]:
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
        for role, folder in trialmatchai_corpus_folders(corpus).items()
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


def write_trialmatchai_corpus_receipt(
    corpus: Path, *, workers: int = TRIALMATCHAI_HASH_WORKERS
) -> dict[str, object]:
    """Hash every prepared file once and write the corpus's location-free preparation receipt."""

    from taim.contracts import content_sha256

    corpus = corpus.resolve()
    receipt_path = corpus / TRIALMATCHAI_CORPUS_RECEIPT_FILENAME
    if receipt_path.exists():
        raise ValueError(f"TrialMatchAI prepared corpus already has a receipt: {receipt_path}")
    folders = trialmatchai_corpus_folders(corpus)
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


def load_trialmatchai_corpus_receipt(corpus: Path) -> dict[str, object]:
    from taim.contracts import content_sha256

    receipt_path = corpus / TRIALMATCHAI_CORPUS_RECEIPT_FILENAME
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


def check_trialmatchai_corpus_names(corpus: Path, receipt: dict[str, object]) -> None:
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


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination) or filecmp.cmp(source, destination, shallow=False):
            return "existing"
        raise ValueError(f"output file already exists with different contents: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _snapshot_trial_ids(snapshot_trials: Path) -> set[str]:
    trial_ids: set[str] = set()
    with snapshot_trials.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                trial_id = str(json.loads(line)["trial_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid Snapshot trial record at {snapshot_trials}:{line_number}"
                ) from exc
            if not trial_id:
                raise ValueError(f"empty trial_id at {snapshot_trials}:{line_number}")
            trial_ids.add(trial_id)
    if not trial_ids:
        raise ValueError(f"Snapshot contains no trial IDs: {snapshot_trials}")
    return trial_ids


def filter_trialmatchai_corpus(
    *,
    snapshot_trials: Path,
    source_trials: Path,
    source_criteria: Path,
    output: Path,
    allow_existing: bool = False,
) -> dict[str, object]:
    """Cut a TrialMatchAI-prepared corpus to exactly a Snapshot's trial IDs, as a run requires.

    Prepared records, and so their embeddings, are hard-linked into ``output`` when the
    filesystem allows and copied otherwise. ``filter-manifest.json`` records the selection; a run
    hashes it into its index identity when present.
    """

    snapshot_trials = snapshot_trials.resolve()
    source_trials = source_trials.resolve()
    source_criteria = source_criteria.resolve()
    output = output.resolve()
    for path, label in (
        (snapshot_trials, "Snapshot trials"),
        (source_trials, "TrialMatchAI processed trials"),
        (source_criteria, "TrialMatchAI processed criteria"),
    ):
        if not path.exists():
            raise ValueError(f"{label} path does not exist: {path}")
    if output.exists() and not allow_existing:
        raise ValueError(f"output already exists: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output is not a directory: {output}")

    trial_ids = sorted(_snapshot_trial_ids(snapshot_trials))
    output_trials = output / "processed_trials"
    output_criteria = output / "processed_criteria"
    output_trials.mkdir(parents=True, exist_ok=allow_existing)
    output_criteria.mkdir(parents=True, exist_ok=allow_existing)

    link_modes = {"hardlink": 0, "copy": 0, "existing": 0}
    for trial_id in trial_ids:
        source = source_trials / f"{trial_id}.json"
        if not source.is_file():
            raise ValueError(f"TrialMatchAI trial is missing: {trial_id}")
        link_modes[_link_or_copy(source, output_trials / source.name)] += 1

    criteria_count = 0
    criteria_trial_ids: set[str] = set()
    for trial_id in trial_ids:
        source_directory = source_criteria / trial_id
        if not source_directory.is_dir():
            continue
        for source in sorted(source_directory.glob("*.json")):
            link_modes[_link_or_copy(source, output_criteria / trial_id / source.name)] += 1
            criteria_count += 1
            criteria_trial_ids.add(trial_id)

    criteria_missing = sorted(set(trial_ids) - criteria_trial_ids)
    manifest: dict[str, object] = {
        "source_snapshot_trials": str(snapshot_trials),
        "source_processed_trials": str(source_trials),
        "source_processed_criteria": str(source_criteria),
        "output_processed_trials": str(output_trials),
        "output_processed_criteria": str(output_criteria),
        "trial_count": len(trial_ids),
        "criteria_count": criteria_count,
        "criteria_trial_count": len(criteria_trial_ids),
        "criteria_missing_trial_count": len(criteria_missing),
        "criteria_missing_trial_ids": criteria_missing,
        "selection": "exact trial-ID membership from the frozen TAIM Snapshot",
        "embeddings": "reused from TrialMatchAI prepared JSON records",
        "link_modes": link_modes,
    }
    (output / TRIALMATCHAI_CORPUS_FILTER_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def trialmatchai_search_database_identity(search_db: Path, *, workers: int) -> dict[str, object]:
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


__all__ = [
    "TrialMatchAIAdapterError",
    "TrialMatchAIHarnessInputs",
    "TrialMatchAIResult",
    "load_and_normalize_ranked_trials",
    "normalize_ranked_trials",
]
