"""Direction-specific public patient-to-trial run artifacts."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from taim.contracts import require_sha256
from taim.evaluation import evaluate_scheme_run
from taim.evaluation_package import EvaluationPackage
from taim.file_hash import sha256_file
from taim.release_profiles import (
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
    OFFICIAL_FULL_PROFILE,
    TREC_2021_JUDGMENT_UNION_PROFILE,
    BenchmarkProfile,
    judgment_union_pool_receipt,
    ordered_trial_ids_sha256,
    pool_receipt_policy,
)
from taim.release_support import (
    RELEASE_VERSION,
    require_frozen_effectiveness_protocol,
    require_supported_combination,
    requires_effectiveness_protocol,
    validate_release_source_identity,
)
from taim.schemas import Candidate, RunManifest, SchemaValidationError, StageRanking

PATIENT_TO_TRIAL_TASK = "patient_to_trial"


# Forward Local Runs use the repository's closed Run Manifest v7 contract.  Keep the
# direction-specific public name as an alias rather than a second, weaker schema.
PublicPatientToTrialRunManifest = RunManifest


def _evaluate_profile_cutoff(
    candidates: tuple[Candidate, ...],
    evaluation_package: EvaluationPackage,
    profile: BenchmarkProfile,
    *,
    topic_ids: tuple[str, ...],
    cutoff: int,
) -> dict[str, object]:
    primary = evaluate_scheme_run(
        candidates,
        evaluation_package.judgments,
        scheme=evaluation_package.judgment_scheme,
        topic_ids=topic_ids,
        k=cutoff,
        unjudged_policy=profile.unjudged_policy,
        precision_metric=profile.precision_metric,
        precision_relevance_minimum=profile.precision_relevance_minimum,
        precision_denominator=profile.precision_denominator,
    )
    if profile.profile_id not in {
        OFFICIAL_FULL_PROFILE,
        EXTERNAL_FIDELITY_TREC_2021_PROFILE,
        TREC_2021_JUDGMENT_UNION_PROFILE,
    }:
        return primary
    eligible = evaluate_scheme_run(
        candidates,
        evaluation_package.judgments,
        scheme=evaluation_package.judgment_scheme,
        topic_ids=topic_ids,
        k=cutoff,
        unjudged_policy=profile.unjudged_policy,
        precision_metric="eligible_precision",
        precision_relevance_minimum=2,
        precision_denominator=profile.precision_denominator,
    )
    aggregate = primary["aggregate"]
    eligible_aggregate = eligible["aggregate"]
    per_topic = primary["per_topic"]
    eligible_per_topic = eligible["per_topic"]
    if not all(
        isinstance(value, dict)
        for value in (aggregate, eligible_aggregate, per_topic, eligible_per_topic)
    ):
        raise SchemaValidationError("forward evaluator returned an invalid scorecard")
    metric = f"eligible_precision_at_{cutoff}"
    cast(dict[str, object], aggregate)[metric] = cast(dict[str, object], eligible_aggregate)[metric]
    for topic_id in topic_ids:
        topic_metrics = cast(dict[str, object], per_topic).get(topic_id)
        eligible_metrics = cast(dict[str, object], eligible_per_topic).get(topic_id)
        if not isinstance(topic_metrics, dict) or not isinstance(eligible_metrics, dict):
            raise SchemaValidationError("forward evaluator omitted a declared topic")
        topic_metrics[metric] = eligible_metrics[metric]
    primary["supplemental_precision_policy"] = eligible["precision_policy"]
    return primary


@dataclass(frozen=True, slots=True)
class StoredPublicPatientToTrialRun:
    directory: Path
    manifest: RunManifest
    candidates: tuple[Candidate, ...]
    manifest_hash: str
    candidates_hash: str
    stage_rankings: tuple[StageRanking, ...] = ()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _release_support(manifest: RunManifest) -> Mapping[str, object]:
    support = manifest.configuration.get("release_support")
    if not isinstance(support, Mapping) or set(support) != {
        "release_version",
        "track",
        "task",
        "profile",
        "system",
        "protocol_approval_id",
    }:
        raise SchemaValidationError("forward run lacks its exact release support identity")
    expected = {
        "release_version": RELEASE_VERSION,
        "track": manifest.benchmark_lineage,
        "task": PATIENT_TO_TRIAL_TASK,
        "profile": manifest.benchmark_profile["profile_id"],
        "system": manifest.system_id,
    }
    if any(support.get(field) != value for field, value in expected.items()):
        raise SchemaValidationError("forward release support identity is direction-mismatched")
    capability = require_supported_combination(
        track=cast(str, support["track"]),
        task=PATIENT_TO_TRIAL_TASK,
        profile=cast(str, support["profile"]),
        system=manifest.system_id,
    )
    validate_release_source_identity(
        manifest.configuration.get("release_source"),
        track=cast(str, support["track"]),
        profile=cast(str, support["profile"]),
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    protocol = support["protocol_approval_id"]
    profile_id = cast(str, support["profile"])
    requires_protocol = requires_effectiveness_protocol(profile_id)
    if requires_protocol:
        require_frozen_effectiveness_protocol(
            profile_id,
            protocol,
            "effectiveness protocol_approval_id",
            system=manifest.system_id,
        )
    elif protocol is not None:
        raise SchemaValidationError("full non-confirmation run cannot claim a protocol approval")
    return support


def _validate_configuration(manifest: RunManifest) -> None:
    _release_support(manifest)
    for field in ("execution_provenance", "model_identity", "index_identity"):
        if not isinstance(manifest.configuration.get(field), Mapping):
            raise SchemaValidationError(f"forward run lacks {field}")


def _validate_candidates(manifest: RunManifest, candidates: tuple[Candidate, ...]) -> None:
    if not candidates:
        raise SchemaValidationError("patient-to-trial ranking is empty")
    patient_ids = {item[0] for item in manifest.query_patient_versions}
    trial_ids = {item[0] for item in manifest.trial_corpus_versions}
    pairs: set[tuple[str, str]] = set()
    ranks: dict[str, set[int]] = defaultdict(set)
    for row in candidates:
        if row.run_id != manifest.run_id or row.system_id != manifest.system_id:
            raise SchemaValidationError("patient-to-trial candidates do not match the manifest")
        if row.topic_id not in patient_ids or row.trial_id not in trial_ids:
            raise SchemaValidationError("patient-to-trial candidate is outside the Task Input")
        pair = (row.topic_id, row.trial_id)
        if pair in pairs or row.rank in ranks[row.topic_id] or row.rank > manifest.budget_k:
            raise SchemaValidationError("patient-to-trial ranking has duplicate or invalid ranks")
        pairs.add(pair)
        ranks[row.topic_id].add(row.rank)
    expected_ranks = set(range(1, min(manifest.budget_k, len(trial_ids)) + 1))
    incomplete = sorted(
        patient_id for patient_id in patient_ids if ranks[patient_id] != expected_ranks
    )
    if incomplete:
        raise SchemaValidationError(
            "patient-to-trial ranking is incomplete at the declared budget for: "
            + ", ".join(incomplete)
        )


def _validate_stage_ranking(manifest: RunManifest, stage: StageRanking) -> None:
    candidates = tuple(stage.candidates)
    if not candidates:
        raise SchemaValidationError(f"patient-to-trial stage {stage.name!r} is empty")
    patient_ids = {item[0] for item in manifest.query_patient_versions}
    trial_ids = {item[0] for item in manifest.trial_corpus_versions}
    pairs: set[tuple[str, str]] = set()
    ranks: set[tuple[str, int]] = set()
    for row in candidates:
        if row.run_id != manifest.run_id or row.system_id != manifest.system_id:
            raise SchemaValidationError(
                f"patient-to-trial stage {stage.name!r} does not match the manifest"
            )
        if row.topic_id not in patient_ids or row.trial_id not in trial_ids:
            raise SchemaValidationError(
                f"patient-to-trial stage {stage.name!r} is outside the Task Input"
            )
        pair = (row.topic_id, row.trial_id)
        rank = (row.topic_id, row.rank)
        if pair in pairs or rank in ranks:
            raise SchemaValidationError(
                f"patient-to-trial stage {stage.name!r} has duplicate pairs or ranks"
            )
        pairs.add(pair)
        ranks.add(rank)
    if [(row.topic_id, row.rank) for row in candidates] != sorted(
        (row.topic_id, row.rank) for row in candidates
    ):
        raise SchemaValidationError(
            f"patient-to-trial stage {stage.name!r} is not in canonical ranking order"
        )
    for patient_id in patient_ids:
        observed = {row.rank for row in candidates if row.topic_id == patient_id}
        if not observed or observed != set(range(1, max(observed) + 1)):
            raise SchemaValidationError(
                f"patient-to-trial stage {stage.name!r} has an incomplete topic ranking"
            )


def _validate_evaluation_package_membership(
    manifest: RunManifest,
    evaluation_package: EvaluationPackage,
) -> None:
    patient_ids = {item[0] for item in manifest.query_patient_versions}
    trial_ids = {item[0] for item in manifest.trial_corpus_versions}
    if any(
        judgment.topic_id not in patient_ids or judgment.trial_id not in trial_ids
        for judgment in evaluation_package.judgments
    ):
        raise SchemaValidationError(
            "forward Evaluation Package contains a Judgment outside the Task Input"
        )
    _validate_judgment_union_membership(manifest, evaluation_package)


def _validate_judgment_union_membership(
    manifest: RunManifest,
    evaluation_package: EvaluationPackage,
) -> None:
    profile = manifest.benchmark_profile
    profile_id = profile.get("profile_id")
    judgment_union = isinstance(profile_id, str) and profile_id.endswith("-judgment-union")
    external_fidelity = profile_id == EXTERNAL_FIDELITY_TREC_2021_PROFILE
    corpus_policy = profile.get("corpus_policy")
    if judgment_union != (corpus_policy == "judgment_union") or external_fidelity != (
        corpus_policy == "external_fidelity_pool"
    ):
        raise SchemaValidationError("forward Benchmark Profile corpus policy is inconsistent")
    if not judgment_union and not external_fidelity:
        if profile.get("pool_receipt_id") is not None:
            raise SchemaValidationError("full-corpus Profile cannot claim a pool receipt")
        return
    profile_id = cast(str, profile_id)

    trial_ids = tuple(item[0] for item in manifest.trial_corpus_versions)
    judged_trial_ids = {judgment.trial_id for judgment in evaluation_package.judgments}
    if set(trial_ids) != judged_trial_ids:
        raise SchemaValidationError(
            "compute-bounded Task Input must contain every and only Evaluation Package trial ID"
        )
    pool_ids_sha256 = ordered_trial_ids_sha256(trial_ids)
    if (
        profile.get("effective_corpus_count") != len(trial_ids)
        or profile.get("pool_ids_sha256") != pool_ids_sha256
        or profile.get("pool_source_sha256") != evaluation_package.evaluation_package_id
    ):
        raise SchemaValidationError(
            "compute-bounded Benchmark Profile pool identity does not match"
        )
    definition_sha256 = require_sha256(
        profile.get("definition_sha256"), "compute-bounded Profile definition_sha256"
    )
    selection_policy, claim_scope = pool_receipt_policy(profile_id)
    receipt = judgment_union_pool_receipt(
        selection_policy=selection_policy,
        claim_scope=claim_scope,
        profile=profile_id,
        profile_definition_sha256=definition_sha256,
        prepared_snapshot_id=manifest.prepared_snapshot_id,
        evaluation_package_id=manifest.evaluation_package_id,
        pool_count=len(trial_ids),
        pool_ids_sha256=pool_ids_sha256,
    )
    if profile.get("pool_receipt_id") != receipt["pool_receipt_id"]:
        raise SchemaValidationError("compute-bounded pool receipt identity does not match")


def write_public_patient_to_trial_run(
    directory: str | Path,
    *,
    manifest: RunManifest,
    candidates: tuple[Candidate, ...],
    evaluation_package: EvaluationPackage,
) -> StoredPublicPatientToTrialRun:
    """Write a closed forward run plus its local evaluator-only sidecar."""

    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"patient-to-trial run directory already exists: {destination}")
    _validate_configuration(manifest)
    if (
        manifest.evaluation_package_id != evaluation_package.evaluation_package_id
        or manifest.prepared_snapshot_id != evaluation_package.snapshot_id
        or manifest.benchmark_lineage != evaluation_package.benchmark_lineage
    ):
        raise SchemaValidationError("patient-to-trial Evaluation Package identity does not match")
    _validate_evaluation_package_membership(manifest, evaluation_package)
    _validate_candidates(manifest, candidates)
    for stage in manifest.stage_rankings:
        _validate_stage_ranking(manifest, stage)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        candidate_path = staging / "candidates.jsonl"
        candidate_path.write_text(
            "".join(f"{row.to_json()}\n" for row in candidates),
            encoding="utf-8",
            newline="\n",
        )
        stored_stages: list[StageRanking] = []
        for stage in manifest.stage_rankings:
            stage_path = staging / f"stage-{stage.name}.jsonl"
            stage_path.write_text(
                "".join(f"{row.to_json()}\n" for row in stage.candidates),
                encoding="utf-8",
                newline="\n",
            )
            stored_stages.append(replace(stage, artifact_hash=sha256_file(stage_path)))
        actual_manifest = replace(
            manifest,
            candidates_sha256=sha256_file(candidate_path),
            stage_rankings=tuple(stored_stages),
        )
        _write_json(staging / "manifest.json", actual_manifest.to_dict())
        _write_json(staging / "evaluation-package.json", evaluation_package.to_dict())
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_public_patient_to_trial_run(destination)


def load_public_patient_to_trial_run(
    directory: str | Path,
) -> StoredPublicPatientToTrialRun:
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise SchemaValidationError("patient-to-trial Local Run must be a real directory")
    entries = tuple(root.rglob("*"))
    names = {path.name for path in entries}
    required = {"candidates.jsonl", "evaluation-package.json", "manifest.json"}
    if (
        any(path.is_symlink() or not path.is_file() or path.parent != root for path in entries)
        or not required <= names
    ):
        raise SchemaValidationError("patient-to-trial Local Run tree is incomplete or has extras")
    try:
        manifest = RunManifest.from_json((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SchemaValidationError(f"cannot read patient-to-trial manifest: {exc}") from exc
    expected_files = {
        "candidates.jsonl",
        "evaluation-package.json",
        "manifest.json",
        *(f"stage-{stage.name}.jsonl" for stage in manifest.stage_rankings),
    }
    if names != expected_files:
        raise SchemaValidationError("patient-to-trial Local Run tree is incomplete or has extras")
    _validate_configuration(manifest)
    candidate_path = root / "candidates.jsonl"
    try:
        lines = candidate_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SchemaValidationError(f"cannot read patient-to-trial candidates: {exc}") from exc
    candidates = tuple(Candidate.from_json(line) for line in lines)
    if not candidates or lines != [row.to_json() for row in candidates]:
        raise SchemaValidationError("patient-to-trial candidates are not canonical JSON Lines")
    if sha256_file(candidate_path) != manifest.candidates_sha256:
        raise SchemaValidationError("patient-to-trial candidates hash does not match")
    _validate_candidates(manifest, candidates)
    stage_rankings: list[StageRanking] = []
    for stage in manifest.stage_rankings:
        stage_path = root / f"stage-{stage.name}.jsonl"
        try:
            stage_lines = stage_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise SchemaValidationError(
                f"cannot read patient-to-trial stage {stage.name!r}: {exc}"
            ) from exc
        stage_candidates = tuple(Candidate.from_json(line) for line in stage_lines)
        loaded = replace(stage, candidates=stage_candidates)
        if not stage_candidates or stage_lines != [row.to_json() for row in stage_candidates]:
            raise SchemaValidationError(
                f"patient-to-trial stage {stage.name!r} is not canonical JSON Lines"
            )
        if sha256_file(stage_path) != stage.artifact_hash:
            raise SchemaValidationError(
                f"patient-to-trial stage {stage.name!r} hash does not match"
            )
        _validate_stage_ranking(manifest, loaded)
        stage_rankings.append(loaded)
    package = load_local_evaluation_package(root)
    if (
        package.evaluation_package_id != manifest.evaluation_package_id
        or package.snapshot_id != manifest.prepared_snapshot_id
        or package.benchmark_lineage != manifest.benchmark_lineage
    ):
        raise SchemaValidationError("forward Evaluation Package does not match the Local Run")
    _validate_evaluation_package_membership(manifest, package)
    return StoredPublicPatientToTrialRun(
        root.resolve(),
        manifest,
        candidates,
        sha256_file(root / "manifest.json"),
        sha256_file(candidate_path),
        tuple(stage_rankings),
    )


def load_local_evaluation_package(directory: str | Path) -> EvaluationPackage:
    try:
        payload = json.loads(
            (Path(directory) / "evaluation-package.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"cannot read local Evaluation Package: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SchemaValidationError("local Evaluation Package must be a JSON object")
    return EvaluationPackage.from_dict(payload)


def evaluate_public_patient_to_trial_run(
    stored: StoredPublicPatientToTrialRun,
    evaluation_package: EvaluationPackage,
    profile: BenchmarkProfile,
) -> dict[str, object]:
    """Evaluate a forward run under its Track-specific Judgment Scheme."""

    manifest = stored.manifest
    support = _release_support(manifest)
    expected_profile = {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "definition_sha256": profile.definition_sha256,
        "evaluation": profile.evaluation_configuration(),
    }
    if (
        evaluation_package.evaluation_package_id != manifest.evaluation_package_id
        or evaluation_package.snapshot_id != manifest.prepared_snapshot_id
        or evaluation_package.benchmark_lineage != manifest.benchmark_lineage
        or any(
            manifest.benchmark_profile.get(key) != value for key, value in expected_profile.items()
        )
    ):
        raise SchemaValidationError("forward Evaluation Package or Profile does not match the run")
    _validate_evaluation_package_membership(manifest, evaluation_package)
    topic_ids = tuple(item[0] for item in manifest.query_patient_versions)
    metrics = {
        str(cutoff): _evaluate_profile_cutoff(
            stored.candidates,
            evaluation_package,
            profile,
            topic_ids=topic_ids,
            cutoff=cutoff,
        )
        for cutoff in profile.cutoffs
    }
    return {
        "schema_version": RunManifest.schema_version,
        "task": PATIENT_TO_TRIAL_TASK,
        "release_version": support["release_version"],
        "track": support["track"],
        "profile": support["profile"],
        "system": manifest.system_id,
        "run_id": manifest.run_id,
        "prepared_snapshot_id": manifest.prepared_snapshot_id,
        "task_input_id": manifest.task_input_id,
        "system_input_id": manifest.system_input_id,
        "evaluation_package_id": manifest.evaluation_package_id,
        "judgment_scheme": evaluation_package.judgment_scheme.to_dict(),
        "metrics_by_cutoff": metrics,
    }


__all__ = [
    "PATIENT_TO_TRIAL_TASK",
    "PublicPatientToTrialRunManifest",
    "StoredPublicPatientToTrialRun",
    "evaluate_public_patient_to_trial_run",
    "load_local_evaluation_package",
    "load_public_patient_to_trial_run",
    "write_public_patient_to_trial_run",
]
