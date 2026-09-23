"""Direction-specific external ranking submissions, independent of the paper System catalogue.

An export writes two directories. The task directory holds only ``input.json``, the judgment-free
Task Input an outside pipeline reads. The evaluator directory holds only
``evaluation-package.json``, which import and validation read and the pipeline never needs.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from taim.contracts import (
    content_sha256,
    require_exact_keys,
    require_sha256,
    system_input_identity_value,
)
from taim.evaluation_package import EvaluationPackage
from taim.file_hash import sha256_file
from taim.public_patient_to_trial import _evaluate_profile_cutoff
from taim.release_profiles import load_profile
from taim.schemas import Candidate
from taim.system_contracts import SystemBenchmarkSnapshot
from taim.trial_to_patient import TrialToPatientTaskInput

_SCHEMA = "1.0"
_SYSTEM = re.compile(r"external:[a-z0-9][a-z0-9._-]*\Z")


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"submission artifact must not be a symlink: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"submission artifact must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _identity(payload: Mapping[str, Any], field: str) -> None:
    if content_sha256(
        {key: value for key, value in payload.items() if key != field}
    ) != payload.get(field):
        raise ValueError(f"submission {field} does not match its content")


def export_task(
    *,
    task: str,
    native_input: Mapping[str, Any],
    evaluation_package: Mapping[str, Any],
    profile: str,
    benchmark_profile: Mapping[str, Any],
    release_source: Mapping[str, Any],
    top_k: int,
    release_identity: Mapping[str, str] | None,
    output: Path,
    evaluator_output: Path,
) -> dict[str, Any]:
    """Write the judgment-free input and the evaluator package into two separate directories."""

    if task not in {"patient_to_trial", "trial_to_patient"}:
        raise ValueError("unknown submission direction")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("submission top_k must be a positive integer")
    for path in (output, evaluator_output):
        if path.exists():
            raise FileExistsError(f"submission export target already exists: {path}")
    if not profile.startswith("synthetic-") and release_identity is None:
        raise ValueError("real submission tasks require the exact release producer identity")
    core = {
        "schema_version": _SCHEMA,
        "artifact_type": "taim-submission-task",
        "task": task,
        "profile": profile,
        "benchmark_profile": dict(benchmark_profile),
        "release_source": dict(release_source),
        "top_k": top_k,
        "input": dict(native_input),
        "release_identity": dict(release_identity) if release_identity is not None else None,
        "evaluation_package_id": evaluation_package["evaluation_package_id"],
    }
    payload = {**core, "export_id": content_sha256(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluator_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".submission-", dir=output.parent))
    evaluator_staging = Path(
        tempfile.mkdtemp(prefix=".submission-evaluator-", dir=evaluator_output.parent)
    )
    try:
        _write(staging / "input.json", payload)
        _write(evaluator_staging / "evaluation-package.json", dict(evaluation_package))
        load_task(staging, evaluator_dir=evaluator_staging, task=task)
        evaluator_staging.replace(evaluator_output)
        try:
            staging.replace(output)
        except BaseException:
            shutil.rmtree(evaluator_output, ignore_errors=True)
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(evaluator_staging, ignore_errors=True)
        raise
    return payload


def load_task(
    directory: Path, *, evaluator_dir: Path, task: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path, name, role in (
        (directory, "input.json", "submission task"),
        (evaluator_dir, "evaluation-package.json", "submission evaluator"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{role} must be a real directory")
        if {child.name for child in path.iterdir()} != {name}:
            raise ValueError(f"{role} directory must hold only {name}")
    payload = _read(directory / "input.json")
    require_exact_keys(
        payload,
        {
            "schema_version",
            "artifact_type",
            "task",
            "profile",
            "benchmark_profile",
            "release_source",
            "top_k",
            "input",
            "release_identity",
            "evaluation_package_id",
            "export_id",
        },
        role="submission input",
    )
    if (
        payload["schema_version"] != _SCHEMA
        or payload["artifact_type"] != "taim-submission-task"
        or payload["task"] != task
    ):
        raise ValueError("submission task direction or schema differs")
    _identity(payload, "export_id")
    if task == "patient_to_trial":
        snapshot = SystemBenchmarkSnapshot.from_dict(payload["input"])
        package = EvaluationPackage.from_dict(_read(evaluator_dir / "evaluation-package.json"))
        if (
            package.snapshot_id != snapshot.prepared_snapshot_id
            or package.benchmark_lineage != snapshot.benchmark_lineage
        ):
            raise ValueError("submission Evaluation Package differs from the Task Input")
        queries = {topic.topic_id for topic in snapshot.topics}
        documents = {trial.trial_id for trial in snapshot.trials}
        if any(
            row.topic_id not in queries or row.trial_id not in documents
            for row in package.judgments
        ):
            raise ValueError("submission judgments are outside the Task Input")
        if payload["evaluation_package_id"] != package.evaluation_package_id:
            raise ValueError("submission Evaluation Package identity changed")
        profile = load_profile(payload["profile"])
        if profile.corpus_policy == "judgment_union" and documents != {
            row.trial_id for row in package.judgments
        }:
            raise ValueError(
                "submission judgment-union membership differs from the Evaluation Package"
            )
        lineage = snapshot.benchmark_lineage
        snapshot_id = snapshot.prepared_snapshot_id
        trials = frozenset((row.trial_id, row.trial_version_id) for row in snapshot.trial_versions)
        from taim.release_fixture import load_release_fixture

        fixture_snapshot_id = load_release_fixture(lineage).snapshot.snapshot_id
        if (snapshot_id == fixture_snapshot_id) != payload["profile"].startswith("synthetic-"):
            raise ValueError("submission Profile changes the synthetic versus real evidence scope")
    else:
        from taim.trial_to_patient import TrialToPatientEvaluationPackage

        reverse = TrialToPatientTaskInput.from_dict(payload["input"])
        reverse_package = TrialToPatientEvaluationPackage.from_dict(
            _read(evaluator_dir / "evaluation-package.json")
        )
        if (
            any(
                getattr(reverse_package, name) != getattr(reverse, name)
                for name in (
                    "prepared_snapshot_id",
                    "benchmark_lineage",
                    "query_set_id",
                    "patient_corpus_id",
                )
            )
            or reverse_package.benchmark_profile_definition_sha256
            != reverse.benchmark_profile.definition_sha256
            or reverse_package.evaluation_package_id != payload["evaluation_package_id"]
            or payload["profile"] != reverse.benchmark_profile.profile_id
        ):
            raise ValueError("reverse submission Evaluation Package or Profile changed")
        trials = frozenset((row.trial_id, row.trial_version_id) for row in reverse.trial_versions)
        patients = {(row.patient_id, row.patient_version_id) for row in reverse.patient_versions}
        if any(
            (row.trial_id, row.trial_version_id) not in trials
            or (row.patient_id, row.patient_version_id) not in patients
            for row in reverse_package.judgments
        ):
            raise ValueError("reverse submission judgments are outside the Task Input")
        lineage = reverse.benchmark_lineage
        snapshot_id = reverse.prepared_snapshot_id
        trials = frozenset((row.trial_id, row.trial_version_id) for row in reverse.trial_versions)
    from taim.release_result_bundle import (
        _expected_profile,
        _release_identity,
        _validate_benchmark_profile_projection,
    )
    from taim.release_support import require_supported_combination, validate_release_source_identity

    capability = require_supported_combination(
        track=lineage,
        task=task,
        profile=payload["profile"],
        system="bm25" if task == "patient_to_trial" else "bm25-trial-to-patient",
    )
    validate_release_source_identity(
        payload["release_source"],
        track=lineage,
        profile=payload["profile"],
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    definition, _ = _expected_profile(lineage, task, payload["profile"])
    _validate_benchmark_profile_projection(
        payload["benchmark_profile"],
        task=task,
        profile_id=payload["profile"],
        expected_definition_sha256=definition,
        prepared_snapshot_id=snapshot_id,
        evaluation_package_id=payload["evaluation_package_id"],
        trial_entity_versions=trials,
    )
    if payload["profile"].startswith("synthetic-"):
        if payload["release_identity"] is not None:
            raise ValueError("synthetic submission export must not claim a real producer")
    elif not isinstance(payload["release_identity"], dict):
        raise ValueError("real submission export requires a release identity")
    else:
        _release_identity(payload["release_identity"])
    k = payload["top_k"]
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("submission top_k must be a positive integer")
    return payload, _read(evaluator_dir / "evaluation-package.json")


def _members(payload: Mapping[str, Any]) -> tuple[list[str], set[str], str, str]:
    if payload["task"] == "patient_to_trial":
        forward = SystemBenchmarkSnapshot.from_dict(payload["input"])
        return (
            [topic.topic_id for topic in forward.topics],
            {trial.trial_id for trial in forward.trials},
            "topic_id",
            "trial_id",
        )
    reverse = TrialToPatientTaskInput.from_dict(payload["input"])
    return (
        [version.trial_id for version in reverse.trial_versions],
        {version.patient_id for version in reverse.patient_versions},
        "trial_id",
        "patient_id",
    )


def _pipeline(path: Path) -> dict[str, Any]:
    payload = _read(path)
    require_exact_keys(
        payload,
        {
            "system_id",
            "source_revision",
            "source_sha256",
            "dependency_lock_sha256",
            "configuration",
        },
        role="submission pipeline",
    )
    if not isinstance(payload["system_id"], str) or _SYSTEM.fullmatch(payload["system_id"]) is None:
        raise ValueError(
            "external System IDs must start with external: and use a lowercase identifier"
        )
    if not isinstance(payload["source_revision"], str) or not payload["source_revision"].strip():
        raise ValueError("external System source_revision is required")
    for field in ("source_sha256", "dependency_lock_sha256"):
        require_sha256(payload[field], field)
    if not isinstance(payload["configuration"], dict):
        raise ValueError("external System configuration must be an object")
    content_sha256(system_input_identity_value(payload, path="external pipeline"))
    return payload


def _ranking(
    path: Path, payload: Mapping[str, Any], pipeline: Mapping[str, Any]
) -> list[dict[str, Any]]:
    queries, documents, query_key, document_key = _members(payload)
    ranks: dict[str, set[int]] = defaultdict(set)
    pairs: set[tuple[str, str]] = set()
    rows = []
    if path.is_symlink():
        raise ValueError("submission ranking must not be a symlink")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("submission ranking row must be an object")
        require_exact_keys(
            row,
            {
                "task",
                "task_input_id",
                "system_id",
                query_key,
                document_key,
                "rank",
                "score",
            },
            role="submission ranking",
        )
        if (
            row["task"] != payload["task"]
            or row["task_input_id"] != payload["input"]["task_input_id"]
            or row["system_id"] != pipeline["system_id"]
        ):
            raise ValueError("submission ranking belongs to a different task or System")
        query, document, rank, score = (
            row[query_key],
            row[document_key],
            row["rank"],
            row["score"],
        )
        if (
            not isinstance(query, str)
            or not isinstance(document, str)
            or query not in queries
            or document not in documents
        ):
            raise ValueError("submission ranking has an unknown query or candidate")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 1
            or rank > payload["top_k"]
            or rank in ranks[query]
            or (query, document) in pairs
        ):
            raise ValueError("submission ranking has duplicate or invalid ranks or pairs")
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            raise ValueError("submission scores must be finite numbers")
        ranks[query].add(rank)
        pairs.add((query, document))
        rows.append(row)
    expected = set(range(1, min(payload["top_k"], len(documents)) + 1))
    if any(ranks[query] != expected for query in queries):
        raise ValueError(
            "submission ranking is incomplete at its declared depth; "
            "missing output is not a judgment"
        )
    return sorted(rows, key=lambda row: (row[query_key], row["rank"]))


def _evaluate(
    payload: Mapping[str, Any], package: Mapping[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    task_input = payload["input"]
    if payload["task"] == "patient_to_trial":
        snapshot = SystemBenchmarkSnapshot.from_dict(task_input)
        evaluation = EvaluationPackage.from_dict(package)
        profile = load_profile(payload["profile"])
        candidates = tuple(
            Candidate(
                "external-submission",
                row["system_id"],
                row["topic_id"],
                row["trial_id"],
                row["rank"],
                row["score"],
            )
            for row in rows
        )
        return {
            str(k): _evaluate_profile_cutoff(
                candidates,
                evaluation,
                profile,
                topic_ids=tuple(topic.topic_id for topic in snapshot.topics),
                cutoff=k,
            )
            for k in profile.cutoffs
        }
    from taim.evaluation import evaluate_scheme_run
    from taim.judgments import TREC_CT_JUDGMENT_SCHEME
    from taim.schemas import RelevanceJudgment
    from taim.trial_to_patient import TrialToPatientEvaluationPackage

    reverse = TrialToPatientTaskInput.from_dict(task_input)
    reverse_package = TrialToPatientEvaluationPackage.from_dict(package)
    candidates = tuple(
        Candidate(
            "external-submission",
            row["system_id"],
            row["trial_id"],
            row["patient_id"],
            row["rank"],
            row["score"],
        )
        for row in rows
    )
    judgments = tuple(
        RelevanceJudgment(row.trial_id, row.patient_id, row.label)
        for row in reverse_package.judgments
    )
    return {
        str(k): evaluate_scheme_run(
            candidates,
            judgments,
            scheme=TREC_CT_JUDGMENT_SCHEME,
            topic_ids=tuple(version.trial_id for version in reverse.trial_versions),
            k=k,
        )
        for k in reverse.benchmark_profile.cutoffs
    }


def import_submission(
    *,
    task_dir: Path,
    evaluator_dir: Path,
    ranking: Path,
    pipeline: Path,
    task: str,
    output: Path,
) -> dict[str, Any]:
    payload, package = load_task(task_dir, evaluator_dir=evaluator_dir, task=task)
    provenance = _pipeline(pipeline)
    rows = _ranking(ranking, payload, provenance)
    metrics = _evaluate(payload, package, rows)
    if output.exists():
        raise FileExistsError(f"submission output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".submission-result-", dir=output.parent))
    try:
        (staging / "ranking.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        _write(staging / "pipeline.json", provenance)
        _write(staging / "metrics.json", metrics)
        core = {
            "schema_version": _SCHEMA,
            "artifact_type": "taim-external-submission",
            "task": task,
            "export_id": payload["export_id"],
            "task_input_id": payload["input"]["task_input_id"],
            "evaluation_package_id": payload["evaluation_package_id"],
            "profile": payload["profile"],
            "release_identity": payload["release_identity"],
            "system_id": provenance["system_id"],
            "paper_matrix_admitted": False,
            "artifacts": {
                name: sha256_file(staging / name)
                for name in ("ranking.jsonl", "pipeline.json", "metrics.json")
            },
        }
        result = {**core, "submission_id": content_sha256(core)}
        _write(staging / "submission.json", result)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result


def validate_submission(
    directory: Path, *, task_dir: Path, evaluator_dir: Path, task: str
) -> dict[str, Any]:
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or {path.name for path in directory.iterdir()}
        != {"submission.json", "ranking.jsonl", "pipeline.json", "metrics.json"}
    ):
        raise ValueError("submission output file inventory changed")
    manifest = _read(directory / "submission.json")
    require_exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_type",
            "task",
            "export_id",
            "task_input_id",
            "evaluation_package_id",
            "profile",
            "release_identity",
            "system_id",
            "paper_matrix_admitted",
            "artifacts",
            "submission_id",
        },
        role="submission manifest",
    )
    _identity(manifest, "submission_id")
    if (
        manifest.get("task") != task
        or manifest.get("paper_matrix_admitted") is not False
        or manifest.get("schema_version") != _SCHEMA
    ):
        raise ValueError("submission schema, direction, or admission scope changed")
    payload, package = load_task(task_dir, evaluator_dir=evaluator_dir, task=task)
    if manifest.get("export_id") != payload["export_id"]:
        raise ValueError("submission belongs to a different exported task")
    expected = {
        name: sha256_file(directory / name)
        for name in ("ranking.jsonl", "pipeline.json", "metrics.json")
    }
    if manifest.get("artifacts") != expected:
        raise ValueError("submission artifact bytes changed")
    pipeline = _pipeline(directory / "pipeline.json")
    expected_identity = {
        "artifact_type": "taim-external-submission",
        "task_input_id": payload["input"]["task_input_id"],
        "evaluation_package_id": payload["evaluation_package_id"],
        "profile": payload["profile"],
        "release_identity": payload["release_identity"],
        "system_id": pipeline["system_id"],
    }
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("submission identities differ from the exported task or pipeline")
    rows = _ranking(directory / "ranking.jsonl", payload, pipeline)
    if _read(directory / "metrics.json") != _evaluate(payload, package, rows):
        raise ValueError("submission metrics do not reproduce")
    return manifest
