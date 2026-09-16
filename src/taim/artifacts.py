"""Validated, hashable artifacts shared by TAIM benchmark runners."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, Protocol

from taim.evaluation import evaluate_profile_run, evaluate_run
from taim.evaluation_package import EvaluationPackage
from taim.judgments import TREC_CT_JUDGMENT_SCHEME
from taim.pipeline_extensions import pipeline_extensions
from taim.profiles import resolve_benchmark_profile_evaluation
from taim.schemas import (
    PRIMARY_RANKING_CANDIDATES,
    Candidate,
    JsonValue,
    RelevanceJudgment,
    RunManifest,
    SchemaValidationError,
    StageRanking,
    read_candidates_jsonl,
    read_manifest,
    write_candidates_jsonl,
    write_manifest,
)
from taim.scorecards import (
    ScorecardRunIdentity,
    build_final_scorecard_artifact,
    read_scorecard,
    render_scorecard_markdown,
)
from taim.snapshot import BenchmarkTopic

if TYPE_CHECKING:
    from taim.pair_assessment import StoredPairAssessmentRun

_TREC_RUN_NAME = re.compile(r"[A-Za-z0-9]{1,12}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
METRICS_SCHEMA_VERSION = "2.0"
PROFILE_METRICS_SCHEMA_VERSION = "3.0"
PROFILE_METRICS_ARTIFACT_VERSION = "3.0"
TrecScorePolicy = Literal["candidate_score", "declared_rank"]


def sha256_file(path: str | Path) -> str:
    """Return a prefixed SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


# The rule that decides whether a manifest field belongs to an artifact's identity.
#
# Three pinned digests drifted on 2026-08-22 because whole-file digests were taken
# over manifests that also describe the build event. Two builds of a bit-identical
# concept store carried different manifest digests over a SLURM job id, two
# timings, a telemetry sample count and a resident-set size. See
# docs/evidence/taim-identity-digest-root-cause.md.
#
#   A field belongs to IDENTITY when its value is determined by the artifact's
#   declared inputs and its production recipe.
#
#   A field belongs to the BUILD EVENT when its value could differ between two
#   productions of the same artifact from the same inputs - that is, anything
#   determined by *when*, *where*, or *on what hardware* the production ran:
#   wall-clock instants and durations, throughput, hostnames, job and process
#   ids, filesystem paths, resource telemetry, and counts of physical storage or
#   link operations.
#
#   The test: if re-running the same recipe over the same inputs on a different
#   machine tomorrow would change the value, it is build provenance, not identity.
#
# The rule is applied per field, never per subtree. A build block routinely holds
# both kinds - the concept store's holds `trialmatchai_commit` and
# `builder_version`, which are identity, next to `peak_rss_bytes`, which is not.
# Dropping the whole block would discard the very provenance whose absence caused
# the original drift.
IDENTITY_DIGEST_RULE_VERSION = "taim-identity-content-digest-v1"


def _flatten_leaf_paths(payload: Mapping[str, object], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            paths |= _flatten_leaf_paths(value, path)
        else:
            paths.add(path)
    return paths


def _without_paths(payload: Mapping[str, object], excluded: frozenset[str], prefix: str = ""):
    kept: dict[str, object] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if path in excluded:
            continue
        kept[key] = _without_paths(value, excluded, path) if isinstance(value, Mapping) else value
    return kept


def identity_content_sha256(
    payload: Mapping[str, object],
    *,
    build_event_paths: Iterable[str],
) -> str:
    """Digest only the fields that identify an artifact, not the build that made it.

    ``build_event_paths`` are dotted leaf paths classified as build event by the
    rule above. Every leaf must be classified deliberately: a path that is not
    present raises, so a field renamed or removed cannot silently change the
    digest's meaning, and a newly added field is caught by
    ``test_identity_digest_classification_is_complete`` rather than defaulting
    into identity unnoticed.
    """

    excluded = frozenset(str(path) for path in build_event_paths)
    present = _flatten_leaf_paths(payload)
    missing = sorted(excluded - present)
    if missing:
        raise ValueError("build-event paths are not present in the payload: " + ", ".join(missing))
    canonical = json.dumps(
        _without_paths(payload, excluded),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: object) -> None:
    """Write stable, human-readable JSON with a final newline."""

    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    Path(path).write_text(f"{serialized}\n", encoding="utf-8")


def _read_json_object(path: str | Path, *, artifact_name: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid {artifact_name} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"{artifact_name} must be a JSON object")
    return payload


def _require_artifact_version(
    payload: Mapping[str, object],
    *,
    field: str,
    expected: str,
    artifact_name: str,
) -> None:
    actual = payload.get(field)
    if actual == expected:
        return
    qualifier = ""
    if isinstance(actual, str):
        try:
            if int(actual.split(".", 1)[0]) < int(expected.split(".", 1)[0]):
                qualifier = "historical "
        except ValueError:
            pass
    raise SchemaValidationError(
        f"unsupported {qualifier}{artifact_name} {field} {actual!r}; expected {expected!r}"
    )


def _validate_metric_identity(payload: Mapping[str, object], *, artifact_name: str) -> None:
    for field in ("prepared_snapshot_id", "system_input_id", "evaluation_package_id"):
        value = payload.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise SchemaValidationError(f"{artifact_name} {field} must be a SHA-256 digest")
    if "snapshot_id" in payload:
        raise SchemaValidationError(f"{artifact_name} contains obsolete snapshot_id")


def read_metrics(path: str | Path) -> dict[str, object]:
    """Read the current base metrics artifact, rejecting historical schemas first."""

    payload = _read_json_object(path, artifact_name="metrics")
    _require_artifact_version(
        payload,
        field="schema_version",
        expected=METRICS_SCHEMA_VERSION,
        artifact_name="metrics schema",
    )
    required = {
        "run_id",
        "system_id",
        "benchmark_lineage",
        "prepared_snapshot_id",
        "system_input_id",
        "evaluation_package_id",
        "cutoff",
        "primary_ranking",
        "pipeline_depth",
        "budget_k",
    }
    missing = required - payload.keys()
    if missing:
        raise SchemaValidationError("metrics is missing fields: " + ", ".join(sorted(missing)))
    _validate_metric_identity(payload, artifact_name="metrics")
    return payload


def read_profile_metrics(path: str | Path) -> dict[str, object]:
    """Read the current profile metrics artifact with explicit version gates."""

    payload = _read_json_object(path, artifact_name="profile-metrics")
    _require_artifact_version(
        payload,
        field="schema_version",
        expected=PROFILE_METRICS_SCHEMA_VERSION,
        artifact_name="profile-metrics schema",
    )
    _require_artifact_version(
        payload,
        field="artifact_version",
        expected=PROFILE_METRICS_ARTIFACT_VERSION,
        artifact_name="profile-metrics artifact",
    )
    expected_fields = {
        "schema_version",
        "artifact_type",
        "artifact_version",
        "run_id",
        "system_id",
        "benchmark_lineage",
        "prepared_snapshot_id",
        "system_input_id",
        "evaluation_package_id",
        "budget_k",
        "benchmark_profile",
        "metrics",
    }
    if set(payload) != expected_fields:
        raise SchemaValidationError("profile-metrics has unexpected or missing fields")
    if payload.get("artifact_type") != "taim-profile-metrics":
        raise SchemaValidationError("unsupported profile-metrics artifact_type")
    _validate_metric_identity(payload, artifact_name="profile-metrics")
    return payload


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(topic_id))
    except ValueError:
        return (1, topic_id)


def validate_trec_run_name(run_name: str) -> str:
    """Validate and return a TREC-compatible run identifier."""

    if _TREC_RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("TREC run name must contain 1-12 ASCII letters or digits")
    return run_name


def write_trec_run(
    path: str | Path,
    candidates: Iterable[Candidate],
    *,
    run_name: str,
    score_policy: TrecScorePolicy = "candidate_score",
) -> None:
    """Export candidates in the six-column TREC submission format."""

    validate_trec_run_name(run_name)
    if score_policy not in ("candidate_score", "declared_rank"):
        raise ValueError(f"unsupported TREC score policy {score_policy!r}")
    rows = list(candidates)
    if not rows:
        raise ValueError("TREC run must contain at least one candidate")
    if any(not isinstance(row, Candidate) for row in rows):
        raise TypeError("TREC run rows must be Candidate instances")
    if len({row.run_id for row in rows}) != 1:
        raise ValueError("TREC run candidates must share one run_id")
    if len({row.system_id for row in rows}) != 1:
        raise ValueError("TREC run candidates must share one system_id")
    ranks_by_topic: dict[str, set[int]] = {}
    trials_by_topic: dict[str, set[str]] = {}
    for row in rows:
        if any(character.isspace() for character in row.topic_id):
            raise ValueError("TREC topic IDs must not contain whitespace")
        if any(character.isspace() for character in row.trial_id):
            raise ValueError("TREC trial IDs must not contain whitespace")
        ranks = ranks_by_topic.setdefault(row.topic_id, set())
        trials = trials_by_topic.setdefault(row.topic_id, set())
        if row.rank in ranks:
            raise ValueError(f"duplicate TREC rank {row.rank} for topic {row.topic_id!r}")
        if row.trial_id in trials:
            raise ValueError(f"duplicate TREC trial {row.trial_id!r} for topic {row.topic_id!r}")
        ranks.add(row.rank)
        trials.add(row.trial_id)
    for topic_id, ranks in ranks_by_topic.items():
        if len(ranks) > 1_000 or max(ranks) > 1_000:
            raise ValueError(f"TREC topic {topic_id!r} exceeds the 1,000-result limit")
    rows.sort(
        key=lambda row: (
            _topic_sort_key(row.topic_id),
            row.topic_id,
            row.rank,
            row.trial_id,
        )
    )
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            score = row.score if score_policy == "candidate_score" else 1_001 - row.rank
            stream.write(
                f"{row.topic_id} Q0 {row.trial_id} {row.rank} {format(score, '.17g')} {run_name}\n"
            )


@dataclass(frozen=True, slots=True)
class StoredRun:
    """A run directory loaded through TAIM's versioned schemas."""

    directory: Path
    manifest: RunManifest
    candidates: tuple[Candidate, ...]
    stage_rankings: tuple[StageRanking, ...]
    manifest_hash: str
    candidates_hash: str


@dataclass(frozen=True, slots=True)
class RunEvaluationContext:
    """Closed general-evaluation inputs exposed to one method-owned extension."""

    stored_run: StoredRun
    benchmark_profile: Mapping[str, JsonValue]
    topic_ids: tuple[str, ...]
    judgments: tuple[RelevanceJudgment, ...]
    abstention_score: float | None
    abstention_pairs: frozenset[tuple[str, str]]
    profile_evaluation: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RunEvaluationArtifact:
    """One typed method-owned diagnostic artifact returned to the run writer."""

    path: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if (
            not self.path
            or path.is_absolute()
            or any(part in {"", ".", "..", ".git"} for part in path.parts)
            or path.suffix != ".json"
        ):
            raise ValueError("evaluation extension artifact must be a safe JSON path")


@dataclass(frozen=True, slots=True)
class RunEvaluationExtensionResult:
    """Replacement profile metrics plus typed diagnostic artifacts."""

    profile_evaluation: Mapping[str, object]
    artifacts: tuple[RunEvaluationArtifact, ...] = ()


class RunEvaluationExtension(Protocol):
    """Method-owned evaluation seam kept outside the general ranking evaluator."""

    def evaluate(self, context: RunEvaluationContext) -> RunEvaluationExtensionResult: ...


def _resolve_reevaluation_extensions(
    benchmark_profile: Mapping[str, JsonValue],
    supplied: Iterable[RunEvaluationExtension] | None,
) -> tuple[RunEvaluationExtension, ...]:
    extensions = None if supplied is None else tuple(supplied)
    profile_id = benchmark_profile.get("profile_id")
    # A Profile's re-evaluation extension belongs to the pipeline that owns the Profile, found
    # through the pipeline registry so this module names no pipeline.
    owners = [
        extension.evaluation_extension
        for extension in pipeline_extensions()
        if extension.evaluation_extension is not None
        and profile_id in extension.evaluation_profiles
    ]
    if not owners:
        return extensions or ()
    required = owners[0]()
    if extensions is None:
        return (required,)
    if not any(isinstance(extension, type(required)) for extension in extensions):
        raise ValueError(
            f"{profile_id} re-evaluation requires the evaluation extension of the pipeline "
            "that owns it"
        )
    return extensions


def primary_candidates(stored: StoredRun) -> tuple[Candidate, ...]:
    """Return the Candidate rows designated as the Primary Ranking."""

    if stored.manifest.primary_ranking != PRIMARY_RANKING_CANDIDATES:
        raise ValueError(f"unsupported primary_ranking {stored.manifest.primary_ranking!r}")
    return stored.candidates


def load_run(
    directory: str | Path,
    *,
    require_evaluator_artifact_identity: bool = False,
) -> StoredRun:
    """Load a run and verify its closed artifact identities.

    Current-schema runs with older evaluator artifacts remain readable. Publication callers set
    ``require_evaluator_artifact_identity`` so the hash-bound scorecard must also
    identify the two standalone metric artifacts.
    """

    return _load_run(
        directory,
        allow_incomplete_scorecard=False,
        require_evaluator_artifact_identity=require_evaluator_artifact_identity,
    )


def _load_run(
    directory: str | Path,
    *,
    allow_incomplete_scorecard: bool,
    require_evaluator_artifact_identity: bool = False,
    verify_evaluator_artifact_hashes: bool = True,
) -> StoredRun:
    """Load a run, allowing only the writer's private pre-scorecard staging state."""

    run_directory = Path(directory)
    manifest_path = run_directory / "manifest.json"
    candidates_path = run_directory / "candidates.jsonl"
    manifest = read_manifest(manifest_path)
    candidates = tuple(read_candidates_jsonl(candidates_path))
    candidates_hash = sha256_file(candidates_path)
    if candidates_hash != manifest.candidates_sha256:
        raise ValueError("candidates hash does not match manifest")
    if any(candidate.run_id != manifest.run_id for candidate in candidates):
        raise ValueError("candidate run_id does not match component manifest")
    if any(candidate.system_id != manifest.system_id for candidate in candidates):
        raise ValueError("candidate system_id does not match component manifest")
    _validate_manifest_entity_lineage(
        candidates,
        manifest=manifest,
        ranking_name="primary candidates",
    )
    _validate_ranking_candidates(
        candidates,
        ranking_name="candidate",
        maximum_rank=manifest.budget_k,
    )
    stage_rankings: list[StageRanking] = []
    for stage in manifest.stage_rankings:
        stage_path = run_directory / "stages" / f"{stage.name}.jsonl"
        if sha256_file(stage_path) != stage.artifact_hash:
            raise ValueError(f"stage ranking {stage.name!r} hash does not match manifest")
        stage_candidates = tuple(read_candidates_jsonl(stage_path))
        if any(candidate.run_id != manifest.run_id for candidate in stage_candidates):
            raise ValueError("stage candidate run_id does not match component manifest")
        if any(candidate.system_id != manifest.system_id for candidate in stage_candidates):
            raise ValueError("stage candidate system_id does not match component manifest")
        _validate_manifest_entity_lineage(
            stage_candidates,
            manifest=manifest,
            ranking_name=f"stage {stage.name!r} candidates",
        )
        _validate_ranking_candidates(
            stage_candidates,
            ranking_name=f"stage {stage.name!r} candidate",
        )
        stage_rankings.append(
            StageRanking(
                stage.name,
                stage.pipeline_depth,
                stage_candidates,
                artifact_hash=stage.artifact_hash,
            )
        )
    evaluation_artifacts = manifest.evaluation_artifacts
    scorecard_path = run_directory / "scorecard.json"
    report_path = run_directory / "scorecard.md"
    if not evaluation_artifacts and not allow_incomplete_scorecard:
        raise ValueError("Run Manifest v7 requires scorecard artifact hash identities")
    if not evaluation_artifacts and (scorecard_path.exists() or report_path.exists()):
        raise ValueError("scorecard artifacts exist without manifest hash identities")
    if evaluation_artifacts:
        if not isinstance(evaluation_artifacts, Mapping) or set(evaluation_artifacts) != {
            "scorecard_json_sha256",
            "scorecard_markdown_sha256",
        }:
            raise SchemaValidationError(
                "manifest evaluation_artifacts must contain the exact scorecard hashes"
            )
        for field_name, path in (
            ("scorecard_json_sha256", scorecard_path),
            ("scorecard_markdown_sha256", report_path),
        ):
            expected_hash = evaluation_artifacts[field_name]
            if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
                raise SchemaValidationError(f"evaluation_artifacts.{field_name} must be SHA-256")
            if sha256_file(path) != expected_hash:
                raise ValueError(f"{path.name} hash does not match manifest")
        scorecard = read_scorecard(scorecard_path)
        metrics = read_metrics(run_directory / "metrics.json")
        profile_metrics = read_profile_metrics(run_directory / "profile-metrics.json")
        for artifact_name, artifact, field_names in (
            (
                "metrics",
                metrics,
                (
                    "run_id",
                    "system_id",
                    "benchmark_lineage",
                    "prepared_snapshot_id",
                    "system_input_id",
                    "evaluation_package_id",
                    "primary_ranking",
                    "pipeline_depth",
                    "budget_k",
                ),
            ),
            (
                "profile metrics",
                profile_metrics,
                (
                    "run_id",
                    "system_id",
                    "benchmark_lineage",
                    "prepared_snapshot_id",
                    "system_input_id",
                    "evaluation_package_id",
                    "budget_k",
                ),
            ),
        ):
            for field_name in field_names:
                if artifact.get(field_name) != getattr(manifest, field_name):
                    raise ValueError(f"{artifact_name} {field_name} does not match manifest")
        if profile_metrics.get("benchmark_profile") != manifest.benchmark_profile:
            raise ValueError("profile metrics Benchmark Profile does not match manifest")
        evaluator_artifacts = scorecard.get("evaluator_artifacts")
        if evaluator_artifacts is None:
            if require_evaluator_artifact_identity:
                raise ValueError(
                    "scorecard does not bind metrics.json and profile-metrics.json; "
                    "re-evaluate the Local Run before publication"
                )
        elif not isinstance(evaluator_artifacts, Mapping):
            raise SchemaValidationError("scorecard evaluator_artifacts must be an object")
        elif verify_evaluator_artifact_hashes:
            for field_name, path in (
                ("metrics_sha256", run_directory / "metrics.json"),
                ("profile_metrics_sha256", run_directory / "profile-metrics.json"),
            ):
                if sha256_file(path) != evaluator_artifacts.get(field_name):
                    raise ValueError(f"{path.name} hash does not match scorecard")
        for field_name in (
            "run_id",
            "system_id",
            "benchmark_lineage",
            "prepared_snapshot_id",
            "task_input_id",
            "system_input_id",
            "evaluation_package_id",
        ):
            if scorecard[field_name] != getattr(manifest, field_name):
                raise ValueError(f"scorecard {field_name} does not match manifest")
        ranking_artifact = scorecard["ranking_artifact"]
        if not isinstance(ranking_artifact, Mapping):
            raise SchemaValidationError("scorecard ranking_artifact must be an object")
        if ranking_artifact.get("artifact_sha256") != candidates_hash:
            raise ValueError("scorecard Primary Ranking hash does not match candidates")
        if ranking_artifact.get("pipeline_depth") != manifest.pipeline_depth:
            raise ValueError("scorecard Pipeline Depth does not match manifest")
        if ranking_artifact.get("declared_max_candidate_depth") != manifest.budget_k:
            raise ValueError("scorecard declared maximum candidate depth does not match manifest")
        if scorecard["benchmark_profile"] != manifest.benchmark_profile:
            raise ValueError("scorecard Benchmark Profile does not match manifest")
        if report_path.read_text(encoding="utf-8") != render_scorecard_markdown(scorecard):
            raise ValueError("scorecard.md does not match the machine-readable scorecard")
    return StoredRun(
        directory=run_directory,
        manifest=manifest,
        candidates=candidates,
        stage_rankings=tuple(stage_rankings),
        manifest_hash=sha256_file(manifest_path),
        candidates_hash=candidates_hash,
    )


def _validate_manifest_entity_lineage(
    candidates: Iterable[Candidate],
    *,
    manifest: RunManifest,
    ranking_name: str,
) -> None:
    patient_ids = {patient_id for patient_id, _version_id in manifest.query_patient_versions}
    trial_ids = {trial_id for trial_id, _version_id in manifest.trial_corpus_versions}
    rows = tuple(candidates)
    unknown_patients = {row.topic_id for row in rows} - patient_ids
    unknown_trials = {row.trial_id for row in rows} - trial_ids
    if unknown_patients or unknown_trials:
        details: list[str] = []
        if unknown_patients:
            details.append("patient IDs " + ", ".join(sorted(unknown_patients)))
        if unknown_trials:
            details.append("trial IDs " + ", ".join(sorted(unknown_trials)))
        raise ValueError(
            f"{ranking_name} reference values outside manifest entity lineage: "
            + "; ".join(details)
        )


def _validate_ranking_candidates(
    candidates: Iterable[Candidate],
    *,
    ranking_name: str,
    maximum_rank: int | None = None,
) -> None:
    candidate_keys: set[tuple[str, str]] = set()
    rank_keys: set[tuple[str, int]] = set()
    for candidate in candidates:
        if maximum_rank is not None and candidate.rank > maximum_rank:
            raise ValueError(
                f"stored {ranking_name} rank {candidate.rank} exceeds budget K={maximum_rank}"
            )
        candidate_key = (candidate.topic_id, candidate.trial_id)
        rank_key = (candidate.topic_id, candidate.rank)
        if candidate_key in candidate_keys:
            raise ValueError(
                f"duplicate stored {ranking_name} for {candidate.topic_id}/{candidate.trial_id}"
            )
        if rank_key in rank_keys:
            raise ValueError(
                f"duplicate stored {ranking_name} rank {candidate.rank} for {candidate.topic_id}"
            )
        candidate_keys.add(candidate_key)
        rank_keys.add(rank_key)


def _write_evaluator_artifacts(
    directory: Path,
    *,
    stored: StoredRun,
    evaluation_package: EvaluationPackage,
    evaluation_extensions: Iterable[RunEvaluationExtension] = (),
) -> StoredRun:
    """Rebuild and seal evaluator-owned artifacts from one closed ranking."""

    manifest = stored.manifest
    if manifest.evaluation_package_id != evaluation_package.evaluation_package_id:
        raise ValueError("run manifest Evaluation Package identity does not match")
    if manifest.prepared_snapshot_id != evaluation_package.snapshot_id:
        raise ValueError(
            "run manifest prepared Snapshot identity does not match Evaluation Package"
        )
    if manifest.benchmark_lineage != evaluation_package.benchmark_lineage:
        raise ValueError("run manifest benchmark lineage does not match Evaluation Package")
    if evaluation_package.judgment_scheme != TREC_CT_JUDGMENT_SCHEME:
        raise ValueError(
            "standard Benchmark Profile evaluation requires the TREC Clinical Trials "
            "Judgment Scheme; use evaluate_scheme_run for a custom scheme"
        )
    if manifest.runtime_seconds is None:
        raise ValueError("closed run manifest must record runtime_seconds")
    system_input = manifest.configuration.get("system_input")
    if not isinstance(system_input, Mapping):
        raise ValueError("manifest configuration must contain its attested System Input")
    metric_cutoff = system_input.get("metric_cutoff")
    if isinstance(metric_cutoff, bool) or not isinstance(metric_cutoff, int) or metric_cutoff < 1:
        raise ValueError("attested System Input metric_cutoff must be a positive integer")

    benchmark_profile = manifest.benchmark_profile
    evaluation_policy = resolve_benchmark_profile_evaluation(benchmark_profile)
    ranked = primary_candidates(stored)
    judgments = evaluation_package.judgments
    topic_ids = tuple(patient_id for patient_id, _version_id in manifest.query_patient_versions)
    trial_ids = {trial_id for trial_id, _version_id in manifest.trial_corpus_versions}
    unexpected_judgment_topics = {item.topic_id for item in judgments} - set(topic_ids)
    if unexpected_judgment_topics:
        raise ValueError(
            "Evaluation Package Judgments reference undeclared topics: "
            + ", ".join(sorted(unexpected_judgment_topics))
        )
    unexpected_judgment_trials = {item.trial_id for item in judgments} - trial_ids
    if unexpected_judgment_trials:
        raise ValueError(
            "Evaluation Package Judgments reference trials outside the effective corpus: "
            + ", ".join(sorted(unexpected_judgment_trials))
        )

    metrics: dict[str, object] = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "system_id": manifest.system_id,
        "benchmark_lineage": manifest.benchmark_lineage,
        "prepared_snapshot_id": manifest.prepared_snapshot_id,
        "system_input_id": manifest.system_input_id,
        "evaluation_package_id": manifest.evaluation_package_id,
        "cutoff": metric_cutoff,
        "primary_ranking": manifest.primary_ranking,
        "pipeline_depth": manifest.pipeline_depth,
        "budget_k": manifest.budget_k,
        **evaluate_run(ranked, judgments, topic_ids=topic_ids, k=metric_cutoff),
        "runtime_seconds": manifest.runtime_seconds,
    }
    if evaluation_policy.ranking_source == "primary":
        profile_ranked = ranked
    else:
        stage_name = evaluation_policy.ranking_source.removeprefix("stage:")
        try:
            profile_ranked = next(
                stage.candidates for stage in stored.stage_rankings if stage.name == stage_name
            )
        except StopIteration as exc:
            raise ValueError(f"benchmark profile ranking stage {stage_name!r} is missing") from exc

    failure_policy = manifest.configuration.get("failure_policy")
    raw_abstention_score = (
        failure_policy.get("abstention_score") if isinstance(failure_policy, dict) else None
    )
    abstention_score = (
        float(raw_abstention_score)
        if isinstance(raw_abstention_score, int | float)
        and not isinstance(raw_abstention_score, bool)
        else None
    )
    generation_coverage = manifest.configuration.get("generation_coverage")
    raw_abstention_identities = (
        generation_coverage.get("semantic_abstention_identities")
        if isinstance(generation_coverage, dict)
        else None
    )
    parsed_abstention_pairs: set[tuple[str, str]] = set()
    for identity in (
        raw_abstention_identities if isinstance(raw_abstention_identities, list) else []
    ):
        if not isinstance(identity, dict):
            continue
        topic_id = identity.get("topic_id")
        trial_id = identity.get("trial_id")
        if isinstance(topic_id, str) and isinstance(trial_id, str):
            parsed_abstention_pairs.add((topic_id, trial_id))
    abstention_pairs = frozenset(parsed_abstention_pairs)
    profile_evaluation = evaluate_profile_run(
        profile_ranked,
        judgments,
        topic_ids=topic_ids,
        cutoffs=evaluation_policy.cutoffs,
        unjudged_policy=evaluation_policy.unjudged_policy,
        precision_relevance_minimum=evaluation_policy.precision_relevance_minimum,
        precision_denominator=evaluation_policy.precision_denominator,
        abstention_score=abstention_score,
        abstention_pairs=abstention_pairs,
    )
    extension_context = RunEvaluationContext(
        stored_run=stored,
        benchmark_profile=benchmark_profile,
        topic_ids=topic_ids,
        judgments=judgments,
        abstention_score=abstention_score,
        abstention_pairs=abstention_pairs,
        profile_evaluation=profile_evaluation,
    )
    extension_artifact_paths: set[str] = set()
    reserved_paths = {
        "candidates.jsonl",
        "manifest.json",
        "metrics.json",
        "profile-metrics.json",
        "scorecard.json",
        "scorecard.md",
    }
    for extension in evaluation_extensions:
        result = extension.evaluate(extension_context)
        if not isinstance(result, RunEvaluationExtensionResult):
            raise TypeError("evaluation extensions must return RunEvaluationExtensionResult")
        profile_evaluation = dict(result.profile_evaluation)
        extension_context = replace(extension_context, profile_evaluation=profile_evaluation)
        for artifact in result.artifacts:
            if not isinstance(artifact, RunEvaluationArtifact):
                raise TypeError("evaluation extension artifacts must be typed")
            if artifact.path in reserved_paths or artifact.path in extension_artifact_paths:
                raise ValueError(
                    f"evaluation extension artifact path is not unique: {artifact.path}"
                )
            extension_artifact_paths.add(artifact.path)
            artifact_path = directory.joinpath(*PurePosixPath(artifact.path).parts)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(artifact_path, artifact.payload)

    metrics_path = directory / "metrics.json"
    profile_metrics_path = directory / "profile-metrics.json"
    scorecard_path = directory / "scorecard.json"
    scorecard_report_path = directory / "scorecard.md"
    manifest_path = directory / "manifest.json"
    write_json(metrics_path, metrics)
    write_json(
        profile_metrics_path,
        {
            "schema_version": PROFILE_METRICS_SCHEMA_VERSION,
            "artifact_type": "taim-profile-metrics",
            "artifact_version": PROFILE_METRICS_ARTIFACT_VERSION,
            "run_id": manifest.run_id,
            "system_id": manifest.system_id,
            "benchmark_lineage": manifest.benchmark_lineage,
            "prepared_snapshot_id": manifest.prepared_snapshot_id,
            "system_input_id": manifest.system_input_id,
            "evaluation_package_id": manifest.evaluation_package_id,
            "budget_k": manifest.budget_k,
            "benchmark_profile": benchmark_profile,
            "metrics": profile_evaluation,
        },
    )
    scorecard = build_final_scorecard_artifact(
        ranked,
        judgments,
        topic_ids=topic_ids,
        benchmark_profile=benchmark_profile,
        identity=ScorecardRunIdentity(
            run_id=manifest.run_id,
            system_id=manifest.system_id,
            benchmark_lineage=manifest.benchmark_lineage,
            prepared_snapshot_id=manifest.prepared_snapshot_id,
            task_input_id=manifest.task_input_id,
            system_input_id=manifest.system_input_id,
            evaluation_package_id=manifest.evaluation_package_id,
        ),
        ranking_artifact_sha256=stored.candidates_hash,
        pipeline_depth=manifest.pipeline_depth,
        budget_k=manifest.budget_k,
        runtime_seconds=manifest.runtime_seconds,
        evaluator_artifacts={
            "metrics_sha256": sha256_file(metrics_path),
            "profile_metrics_sha256": sha256_file(profile_metrics_path),
        },
    )
    write_json(scorecard_path, scorecard)
    scorecard_report_path.write_text(
        render_scorecard_markdown(scorecard),
        encoding="utf-8",
    )
    write_manifest(
        manifest_path,
        replace(
            manifest,
            evaluation_artifacts={
                "scorecard_json_sha256": sha256_file(scorecard_path),
                "scorecard_markdown_sha256": sha256_file(scorecard_report_path),
            },
        ),
    )
    read_metrics(metrics_path)
    read_profile_metrics(profile_metrics_path)
    return load_run(directory, require_evaluator_artifact_identity=True)


def _rebind_evaluation_package_for_profile(
    manifest: RunManifest,
    *,
    target: EvaluationPackage,
    source: EvaluationPackage | None,
) -> RunManifest:
    """Rebind an old full-package run to its exact profile-filtered package."""

    if manifest.evaluation_package_id == target.evaluation_package_id:
        return manifest
    if source is None or source.evaluation_package_id != manifest.evaluation_package_id:
        raise ValueError("re-evaluation needs the source Evaluation Package to change its identity")
    if (
        source.benchmark_lineage,
        source.task_id,
        source.snapshot_id,
        source.provenance,
        source.judgment_scheme,
    ) != (
        target.benchmark_lineage,
        target.task_id,
        target.snapshot_id,
        target.provenance,
        target.judgment_scheme,
    ):
        raise ValueError("target Evaluation Package is not a profile projection of the source")
    effective_trial_ids = {trial_id for trial_id, _version_id in manifest.trial_corpus_versions}
    expected_judgments = tuple(
        judgment for judgment in source.judgments if judgment.trial_id in effective_trial_ids
    )
    if target.judgments != expected_judgments:
        raise ValueError(
            "target Evaluation Package does not exactly filter source Judgments to the run corpus"
        )
    return replace(manifest, evaluation_package_id=target.evaluation_package_id)


def reevaluate_run(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    evaluation_package: EvaluationPackage,
    source_evaluation_package: EvaluationPackage | None = None,
    evaluation_extensions: Iterable[RunEvaluationExtension] | None = None,
) -> StoredRun:
    """Copy a closed run and atomically rebuild every standard evaluator artifact.

    Candidate and stage-ranking files are copied byte-for-byte. Canonical embedded
    System Input content and identity are preserved while the surrounding manifest
    is rewritten. No System executes. The source must use the current Run Manifest
    schema. Normally the Evaluation Package matches the source manifest.
    Current-schema restricted-profile runs may also supply their original full
    ``source_evaluation_package`` and a target package that is its exact filter;
    only the evaluator identity is rebound.
    """

    if not isinstance(evaluation_package, EvaluationPackage):
        raise TypeError("evaluation_package must be an EvaluationPackage")
    if source_evaluation_package is not None and not isinstance(
        source_evaluation_package, EvaluationPackage
    ):
        raise TypeError("source_evaluation_package must be an EvaluationPackage or None")
    source = Path(source_directory).resolve()
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"re-evaluated run destination already exists: {destination}")
    if destination.is_relative_to(source):
        raise ValueError("re-evaluated run destination must not be inside the source run")
    stored = _load_run(
        source,
        allow_incomplete_scorecard=False,
        verify_evaluator_artifact_hashes=False,
    )
    resolved_evaluation_extensions = _resolve_reevaluation_extensions(
        stored.manifest.benchmark_profile,
        evaluation_extensions,
    )
    rebound_manifest = _rebind_evaluation_package_for_profile(
        stored.manifest,
        target=evaluation_package,
        source=source_evaluation_package,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name or 'taim-run'}.", dir=destination.parent)
    )
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        write_manifest(staging / "manifest.json", rebound_manifest)
        staged = replace(
            stored,
            directory=staging,
            manifest=rebound_manifest,
            manifest_hash=sha256_file(staging / "manifest.json"),
        )
        rebuilt = _write_evaluator_artifacts(
            staging,
            stored=staged,
            evaluation_package=evaluation_package,
            evaluation_extensions=resolved_evaluation_extensions,
        )
        if rebuilt.candidates_hash != stored.candidates_hash:
            raise ValueError("re-evaluation changed the Primary Ranking")
        if rebuilt.manifest.configuration.get("system_input") != stored.manifest.configuration.get(
            "system_input"
        ):
            raise ValueError("re-evaluation changed the attested System Input")
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_run(destination, require_evaluator_artifact_identity=True)


def write_run(
    output_directory: str | Path,
    *,
    manifest: RunManifest,
    candidates: Iterable[Candidate],
    topics: Iterable[BenchmarkTopic],
    evaluation_package: EvaluationPackage,
    metric_cutoff: int,
    runtime_seconds: float,
    trec_run_name: str | None = None,
    benchmark_profile: Mapping[str, JsonValue],
    trec_score_policy: TrecScorePolicy = "candidate_score",
    pair_assessment_runs: Iterable[StoredPairAssessmentRun] = (),
    evaluation_extensions: Iterable[RunEvaluationExtension] = (),
) -> StoredRun:
    """Write, re-read, evaluate, and optionally export one benchmark run."""

    if isinstance(metric_cutoff, bool) or not isinstance(metric_cutoff, int) or metric_cutoff < 1:
        raise ValueError("metric_cutoff must be a positive integer")
    system_input = manifest.configuration.get("system_input")
    if not isinstance(system_input, Mapping):
        raise ValueError("manifest configuration must contain its attested System Input")
    if system_input.get("metric_cutoff") != metric_cutoff:
        raise ValueError("metric_cutoff must match the manifest's attested System Input")
    if (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, int | float)
        or not math.isfinite(runtime_seconds)
        or runtime_seconds < 0
    ):
        raise ValueError("runtime_seconds must be a finite non-negative number")
    runtime_seconds = float(runtime_seconds)

    directory = Path(output_directory)
    if directory.exists():
        raise FileExistsError(f"run output directory already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name or 'taim-run'}.", dir=directory.parent)
    )
    try:
        candidate_path = staging / "candidates.jsonl"
        manifest_path = staging / "manifest.json"
        candidate_rows = tuple(candidates)
        stored_pair_assessments = tuple(pair_assessment_runs)
        if stored_pair_assessments:
            from taim.pair_assessment import (
                StoredPairAssessmentRun,
                pair_assessment_collection_lineage,
            )

            if any(not isinstance(run, StoredPairAssessmentRun) for run in stored_pair_assessments):
                raise TypeError(
                    "pair_assessment_runs must contain StoredPairAssessmentRun instances"
                )
            expected_patients = {
                version_id for _patient_id, version_id in manifest.query_patient_versions
            }
            expected_trials = {
                version_id for _trial_id, version_id in manifest.trial_corpus_versions
            }
            if any(
                pair_input.patient_version.patient_version_id not in expected_patients
                or pair_input.trial_version.trial_version_id not in expected_trials
                for run in stored_pair_assessments
                for pair_input in run.inputs.values()
            ):
                raise ValueError(
                    "Pair Assessment reference contains entity versions outside the forward task"
                )
            pair_references, pair_reuse_lineage = pair_assessment_collection_lineage(
                stored_pair_assessments
            )
        else:
            pair_references = ()
            pair_reuse_lineage = {"pair_assessment_collections": []}
        manifest = replace(
            manifest,
            pair_assessment_references=pair_references,
            reuse_lineage=pair_reuse_lineage,
        )
        stage_rankings = manifest.stage_rankings
        topic_rows = tuple(topics)
        if not isinstance(evaluation_package, EvaluationPackage):
            raise TypeError("evaluation_package must be an EvaluationPackage")
        judgments = evaluation_package.judgments
        topic_ids = [topic.topic_id for topic in topic_rows]
        if len(topic_ids) != len(set(topic_ids)):
            raise ValueError("run topics must have unique topic_id values")
        topic_id_set = set(topic_ids)
        manifest_topic_ids = {
            patient_id for patient_id, _version_id in manifest.query_patient_versions
        }
        if topic_id_set != manifest_topic_ids:
            raise ValueError("run topics must exactly match manifest query patient lineage")
        _validate_manifest_entity_lineage(
            candidate_rows,
            manifest=manifest,
            ranking_name="primary candidates",
        )
        for stage in stage_rankings:
            _validate_manifest_entity_lineage(
                stage.candidates,
                manifest=manifest,
                ranking_name=f"stage {stage.name!r} candidates",
            )
        unexpected_candidate_topics = {
            candidate.topic_id for candidate in candidate_rows
        } - topic_id_set
        unexpected_stage_topics = {
            candidate.topic_id for stage in stage_rankings for candidate in stage.candidates
        } - topic_id_set
        unexpected_judgment_topics = {judgment.topic_id for judgment in judgments} - topic_id_set
        if unexpected_candidate_topics:
            raise ValueError(
                "candidates reference undeclared topics: "
                + ", ".join(sorted(unexpected_candidate_topics))
            )
        if unexpected_stage_topics:
            raise ValueError(
                "stage rankings reference undeclared topics: "
                + ", ".join(sorted(unexpected_stage_topics))
            )
        if unexpected_judgment_topics:
            raise ValueError(
                "Evaluation Package Judgments reference undeclared topics: "
                + ", ".join(sorted(unexpected_judgment_topics))
            )
        write_candidates_jsonl(candidate_path, candidate_rows)
        manifest = replace(
            manifest,
            candidates_sha256=sha256_file(candidate_path),
            runtime_seconds=runtime_seconds,
        )
        stored_stage_rankings: list[StageRanking] = []
        for stage in stage_rankings:
            stage_path = staging / "stages" / f"{stage.name}.jsonl"
            write_candidates_jsonl(stage_path, stage.candidates)
            stored_stage_rankings.append(replace(stage, artifact_hash=sha256_file(stage_path)))
        write_manifest(
            manifest_path,
            replace(manifest, stage_rankings=tuple(stored_stage_rankings)),
        )

        stored = _load_run(staging, allow_incomplete_scorecard=True)
        if stored.manifest.benchmark_profile != benchmark_profile:
            raise ValueError("benchmark profile must exactly match manifest.benchmark_profile")
        stored = _write_evaluator_artifacts(
            staging,
            stored=stored,
            evaluation_package=evaluation_package,
            evaluation_extensions=evaluation_extensions,
        )
        ranked = primary_candidates(stored)
        if trec_run_name is not None:
            write_trec_run(
                staging / "run.trec",
                ranked,
                run_name=trec_run_name,
                score_policy=trec_score_policy,
            )
        staging.replace(directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_run(directory, require_evaluator_artifact_identity=True)


__all__ = [
    "METRICS_SCHEMA_VERSION",
    "PROFILE_METRICS_ARTIFACT_VERSION",
    "PROFILE_METRICS_SCHEMA_VERSION",
    "RunEvaluationArtifact",
    "RunEvaluationContext",
    "RunEvaluationExtension",
    "RunEvaluationExtensionResult",
    "StoredRun",
    "load_run",
    "primary_candidates",
    "read_metrics",
    "read_profile_metrics",
    "read_scorecard",
    "reevaluate_run",
    "sha256_file",
    "validate_trec_run_name",
    "write_json",
    "write_run",
    "write_trec_run",
]
