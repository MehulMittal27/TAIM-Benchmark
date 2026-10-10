"""Profile-sized prepared packages that carry complete eligibility text.

A stratum's prepared package keeps a bounded excerpt of long source values, so it cannot hold the
complete eligibility text a System may require. This builder re-reads only the pool trials of one
frozen Profile from the verified Source Bundle, keeps their complete eligibility text, and builds
the criterion Derived View from it. Removing the complete-text fields from each re-read trial must
reproduce the stratum's own record exactly, or nothing is written.
"""

from __future__ import annotations

import tarfile
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from taim.data.prepared import (
    PreparationResult,
    load_prepared_benchmark,
    write_prepared_benchmark,
)
from taim.data.trec_ct_2021 import (
    IngestionError,
    MalformedTrialError,
    _source_trial_id,
    _verify_source_bundle,
    parse_trial_xml,
)
from taim.eligibility import ELIGIBILITY_SPLIT_VERSION, build_eligibility_criterion_view
from taim.evaluation_package import EvaluationPackage
from taim.release_profiles import resolve_benchmark_profile
from taim.schemas import JsonValue, SchemaValidationError, json_value_to_builtins
from taim.snapshot import CAPABILITY_COMPLETE_ELIGIBILITY_TEXT, BenchmarkSnapshot, Trial
from taim.source import SourceArtifact

BUILDER_VERSION = "taim-profile-complete-eligibility-v1"


def profile_complete_eligibility_recipe_id(dataset_id: str) -> str:
    """The Snapshot Source Recipe id of a complete-eligibility package of ``dataset_id``."""

    return f"{dataset_id}-profile-complete-eligibility-v1"


def _without_complete_eligibility(trial: Trial) -> Trial:
    return replace(
        trial,
        sections=tuple(
            replace(
                section, source_text=None, source_text_sha256=None, source_text_byte_length=None
            )
            if section.source_text is not None
            else section
            for section in trial.sections
        ),
    )


def _selected_members(
    path: Path,
    artifact: SourceArtifact,
    selected: frozenset[str],
) -> Iterator[tuple[str, bytes]]:
    """Each selected trial's XML member: TREC ships ZIP parts, SIGIR one gzip tarball."""

    if artifact.role.startswith("trials_part_"):
        try:
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.infolist(), key=lambda item: item.filename):
                    if (
                        not member.is_dir()
                        and member.filename.casefold().endswith(".xml")
                        and _source_trial_id(member.filename) in selected
                    ):
                        yield member.filename, archive.read(member)
        except zipfile.BadZipFile as exc:
            raise IngestionError(f"invalid ZIP archive {artifact.filename}: {exc}") from exc
        return
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            for tar_member in sorted(archive.getmembers(), key=lambda item: item.name):
                if (
                    not tar_member.isfile()
                    or not tar_member.name.casefold().endswith(".xml")
                    or _source_trial_id(tar_member.name) not in selected
                ):
                    continue
                stream = archive.extractfile(tar_member)
                if stream is None:
                    raise IngestionError(f"trial archive member cannot be read: {tar_member.name}")
                with stream:
                    yield tar_member.name, stream.read()
    except tarfile.TarError as exc:
        raise IngestionError(f"invalid tar archive {artifact.filename}: {exc}") from exc


def _complete_source_trials(
    source_directory: Path,
    artifacts: Iterable[SourceArtifact],
    selected: frozenset[str],
) -> dict[str, Trial]:
    trials: dict[str, Trial] = {}
    for artifact in artifacts:
        members = _selected_members(source_directory / artifact.filename, artifact, selected)
        for member_name, serialized in members:
            try:
                trial, _diagnostics = parse_trial_xml(
                    serialized,
                    artifact=artifact,
                    member_name=member_name,
                    include_complete_eligibility_text=True,
                )
            except MalformedTrialError:
                continue
            if trial.trial_id in trials:
                raise IngestionError(f"trial archives repeat trial ID {trial.trial_id}")
            trials[trial.trial_id] = trial
    return trials


def prepare_profile_complete_eligibility_snapshot(
    source_snapshot_directory: str | Path,
    source_bundle_directory: str | Path,
    output_directory: str | Path,
    *,
    profile_id: str,
    expected_pool_count: int | None = None,
    expected_pool_ids_sha256: str | None = None,
    expected_pool_receipt_id: str | None = None,
    supplied_pool_ids: Iterable[str] | None = None,
    created_at: datetime | None = None,
    progress: Callable[[str], None] | None = None,
) -> PreparationResult:
    """Write one Profile's trials, with complete eligibility text and criteria, as a package.

    The new package keeps the source package's topics, trial IDs and order, judgments, dataset and
    Snapshot name. It differs in the trials' complete eligibility text, the criterion Derived View
    and the two capabilities that declare them, so its Prepared Snapshot ID, Evaluation Package ID,
    pool receipt and Task Input are new.
    """

    source = Path(source_bundle_directory)
    destination = Path(output_directory)
    if not source.is_dir():
        raise IngestionError(f"source bundle directory does not exist: {source}")
    if destination.exists():
        raise SchemaValidationError(f"output directory already exists: {destination}")
    timestamp = created_at if created_at is not None else datetime.now(UTC)
    report = progress if progress is not None else lambda _message: None

    report("loading the source prepared package")
    prepared = load_prepared_benchmark(source_snapshot_directory)
    if (
        prepared.snapshot.derived_views
        or CAPABILITY_COMPLETE_ELIGIBILITY_TEXT in prepared.snapshot.available_capabilities
    ):
        raise SchemaValidationError(
            "the source package must be a stratum package without complete eligibility text or "
            "Derived Views"
        )
    report("verifying the Source Bundle")
    _verify_source_bundle(source, prepared.source_bundle)
    resolved = resolve_benchmark_profile(
        prepared,
        profile_id=profile_id,
        expected_pool_count=expected_pool_count,
        expected_pool_ids_sha256=expected_pool_ids_sha256,
        expected_pool_receipt_id=expected_pool_receipt_id,
        supplied_pool_ids=supplied_pool_ids,
    )
    selected = frozenset(trial.trial_id for trial in resolved.trials)
    trial_artifacts = tuple(
        artifact
        for artifact in prepared.source_bundle.artifacts
        if artifact.role == "trials" or artifact.role.startswith("trials_part_")
    )
    if not trial_artifacts:
        raise IngestionError("the Source Bundle declares no trial archive")

    report(f"re-reading {len(selected)} Profile trials with complete eligibility text")
    complete = _complete_source_trials(source, trial_artifacts, selected)
    trials: list[Trial] = []
    for source_trial in resolved.trials:
        trial = complete.get(source_trial.trial_id)
        if trial is None:
            raise IngestionError(
                f"the Source Bundle does not reproduce Profile trial {source_trial.trial_id}"
            )
        if _without_complete_eligibility(trial).to_json() != source_trial.to_json():
            raise IngestionError(
                f"trial {trial.trial_id} re-read from the Source Bundle does not reproduce the "
                "source package's record"
            )
        trials.append(trial)

    report("building the criterion Derived View")
    base = BenchmarkSnapshot(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        snapshot_name=prepared.snapshot.snapshot_name,
        topics=resolved.topics,
        trials=tuple(trials),
        available_capabilities=prepared.snapshot.available_capabilities
        | {CAPABILITY_COMPLETE_ELIGIBILITY_TEXT},
    )
    view = build_eligibility_criterion_view(base)
    snapshot = replace(
        base,
        available_capabilities=base.available_capabilities | {view.capability},
        derived_views=(view,),
    )

    stratum: dict[str, JsonValue] = {
        "prepared_snapshot_id": prepared.snapshot.snapshot_id,
        "profile_id": profile_id,
        "profile_definition_sha256": resolved.profile.definition_sha256,
        "pool_count": len(resolved.trials),
        "pool_ids_sha256": resolved.pool_ids_hash,
        "pool_receipt_id": resolved.pool_receipt_id,
    }
    source_evaluation = resolved.evaluation_package
    evaluation = EvaluationPackage(
        benchmark_lineage=source_evaluation.benchmark_lineage,
        task_id=source_evaluation.task_id,
        snapshot_id=snapshot.snapshot_id,
        judgments=source_evaluation.judgments,
        provenance={
            **source_evaluation.provenance,
            "profile_complete_eligibility_projection": {
                **stratum,
                "source_evaluation_package_id": source_evaluation.evaluation_package_id,
            },
        },
        judgment_scheme=source_evaluation.judgment_scheme,
    )
    source_recipe = cast(
        Mapping[str, JsonValue], prepared.prepared_manifest["snapshot_source_recipe"]
    )
    recipe = cast(dict[str, JsonValue], json_value_to_builtins(source_recipe["definition"]))
    recipe["recipe_id"] = profile_complete_eligibility_recipe_id(prepared.dataset_id)
    recipe["source_stratum"] = stratum
    recipe["complete_eligibility"] = {
        "capability": CAPABILITY_COMPLETE_ELIGIBILITY_TEXT,
        "source": "the Profile's trials re-read from the verified Source Bundle",
        "record_check": (
            "removing the complete eligibility text fields reproduces the source package's "
            "record exactly"
        ),
        "derived_view": view.name,
        "derived_view_version": view.version,
    }
    selected_topics = {topic.topic_id for topic in resolved.topics}
    diagnostics = tuple(
        diagnostic
        for diagnostic in prepared.diagnostics
        if diagnostic.record_id is None
        or diagnostic.record_id in selected
        or diagnostic.record_id in selected_topics
    )
    records = cast(
        Mapping[str, Mapping[str, JsonValue]], cast(Mapping[str, JsonValue], view.content)["trials"]
    )
    tiers = Counter(cast(str, record["parse_tier"]) for record in records.values())
    report("writing the package")
    result = write_prepared_benchmark(
        destination,
        dataset_id=prepared.dataset_id,
        snapshot=snapshot,
        evaluation_package=evaluation,
        source_bundle=prepared.source_bundle,
        source_recipe=recipe,
        diagnostics=diagnostics,
        created_at=timestamp,
        build_provenance={
            "builder_version": BUILDER_VERSION,
            "source_prepared_manifest_sha256": prepared.prepared_manifest_hash,
            "source_evaluation_package_id": source_evaluation.evaluation_package_id,
            "eligibility_criterion_view_id": view.view_id,
            "eligibility_split_version": ELIGIBILITY_SPLIT_VERSION,
            "trials_with_eligibility_criteria": len(records),
            "criterion_count": sum(
                cast(int, record["criterion_count"]) for record in records.values()
            ),
            "eligibility_parse_tier_counts": dict(sorted(tiers.items())),
        },
    )
    if load_prepared_benchmark(result.output_directory).snapshot.snapshot_id != (
        snapshot.snapshot_id
    ):
        raise SchemaValidationError("the written package does not reload to its Snapshot")
    return result


__all__ = [
    "BUILDER_VERSION",
    "prepare_profile_complete_eligibility_snapshot",
    "profile_complete_eligibility_recipe_id",
]
