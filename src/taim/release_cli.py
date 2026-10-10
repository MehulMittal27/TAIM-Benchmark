"""Public command line for Benchmark Release 0.1."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from taim import __version__, release_producer
from taim.cli_support import (
    DEFAULT_RETRIEVAL_DEPTH,
    add_bm25_options,
    add_pool_options,
    add_real_input_options,
    add_release_identity_options,
    add_run_output_options,
    print_payload,
    read_json_object,
    read_pool_ids,
    sha256_identity,
)
from taim.data import PreparedBenchmark, load_prepared_benchmark
from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.file_hash import sha256_file
from taim.patient_to_trial import PatientToTrialRunRequest
from taim.pipeline_extensions import (
    ComponentRun,
    PipelineExtension,
    extension_for,
    pipeline_extensions,
)
from taim.public_patient_to_trial import (
    evaluate_public_patient_to_trial_run,
    load_local_evaluation_package,
    load_public_patient_to_trial_run,
    write_public_patient_to_trial_run,
)
from taim.release_fixture import FIXTURE_CLINICAL_AS_OF, load_release_fixture
from taim.release_profiles import (
    EXTERNAL_FIDELITY_TREC_2021_PROFILE,
    inspect_external_fidelity_profile,
    inspect_judgment_union_profile,
    load_profile,
    ordered_trial_ids_sha256,
    resolve_benchmark_profile,
)
from taim.release_reference_systems import run_release_reference_system
from taim.release_support import (
    RELEASE_VERSION,
    SUBCOMMAND_SYSTEMS,
    SUPPORTED_TRACKS,
    SYSTEM_TASKS,
    load_release_support,
    release_source_identity,
    require_frozen_effectiveness_protocol,
    require_frozen_paper_protocol,
    require_supported_combination,
    requires_effectiveness_protocol,
    source_profile_id,
    validate_public_reverse_run_tree,
    validate_reverse_release_manifest,
)
from taim.reverse_fixture import load_reverse_fixture
from taim.reverse_viability import qualify_reverse_viability_profile
from taim.schemas import (
    JsonValue,
    RunManifest,
    StageRanking,
)
from taim.submission_cli import add_submission_commands
from taim.trial_to_patient import (
    StoredTrialToPatientRun,
    TrialToPatientEvaluationPackage,
    TrialToPatientExecutionProvenance,
    TrialToPatientRunRequest,
    evaluate_trial_to_patient_run,
    load_trial_to_patient_run,
    run_trial_to_patient_system,
    write_trial_to_patient_run,
)


def _run_id(system_id: str) -> str:
    return f"public-{system_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"


def _support_list(_args: argparse.Namespace) -> int:
    print_payload(load_release_support())
    return 0


def _verified_external_pool_ids(
    source: Path,
    *,
    expected_byte_size: int,
    expected_sha256: str,
    expected_count: int,
    expected_sorted_ids_sha256: str,
) -> tuple[str, ...]:
    if source.stat().st_size != expected_byte_size or sha256_file(source) != expected_sha256:
        raise ValueError("external-fidelity source does not match its packaged checksum lock")
    trial_ids: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"external-fidelity source record {line_number} is invalid JSON"
                ) from exc
            trial_id = record.get("_id") if isinstance(record, Mapping) else None
            if not isinstance(trial_id, str) or not trial_id:
                raise ValueError(f"external-fidelity source record {line_number} lacks a trial ID")
            trial_ids.append(trial_id)
    ordered = tuple(sorted(trial_ids))
    if len(ordered) != expected_count or len(set(ordered)) != expected_count:
        raise ValueError("external-fidelity source record count or ID uniqueness changed")
    if ordered_trial_ids_sha256(ordered) != expected_sorted_ids_sha256:
        raise ValueError("external-fidelity source trial-ID membership changed")
    return ordered


def _prepare_external_pool(args: argparse.Namespace) -> int:
    if args.profile != EXTERNAL_FIDELITY_TREC_2021_PROFILE:
        raise ValueError("Release 0.1 has one external-fidelity source Profile")
    lock_path = (
        Path(__file__).parent / "data" / "locks" / "trialgpt-trec-2021-external-fidelity.json"
    )
    lock = read_json_object(lock_path, "external-fidelity source lock")
    source = cast(Mapping[str, object], lock["source"])
    membership = cast(Mapping[str, object], lock["membership"])
    trial_ids = _verified_external_pool_ids(
        args.source,
        expected_byte_size=cast(int, source["byte_size"]),
        expected_sha256=cast(str, source["sha256"]),
        expected_count=cast(int, membership["record_count"]),
        expected_sorted_ids_sha256=cast(str, membership["sorted_unique_trial_ids_sha256"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(trial_ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print_payload(
        {
            "release": RELEASE_VERSION,
            "track": "trec-ct-2021",
            "task": "patient_to_trial",
            "profile": args.profile,
            "source_sha256": source["sha256"],
            "pool_count": len(trial_ids),
            "sorted_pool_ids_sha256": ordered_trial_ids_sha256(trial_ids),
            "output": str(args.output.resolve()),
        }
    )
    return 0


def _forward_options(
    args: argparse.Namespace,
    benchmark,
    *,
    task_input_id: str,
) -> dict[str, object]:
    extension = extension_for(args.system)
    if extension is not None and extension.run_options is not None:
        return extension.run_options(args, benchmark, task_input_id)
    raise ValueError("fusion and staged options are supplied as direction-specific artifacts")


def _store_forward(
    prepared: PreparedBenchmark,
    args: argparse.Namespace,
    *,
    track: str,
    profile_id: str,
    clinical_as_of: datetime,
    protocol_approval_id: str | None = None,
) -> Path:
    producer_release_identity = release_producer.verified_release_identity(args)
    if (
        profile_id.endswith("-judgment-union") or profile_id == EXTERNAL_FIDELITY_TREC_2021_PROFILE
    ) and producer_release_identity is None:
        raise ValueError(
            "compute-bounded execution requires --release-manifest, --package-artifact, and "
            "--dependency-lock"
        )
    producer_dependency_environment = (
        release_producer.verified_dependency_environment(args, system_id=args.system)
        if producer_release_identity is not None
        else None
    )
    producer_git_state = release_producer.git_state()
    capability = require_supported_combination(
        track=track,
        task="patient_to_trial",
        profile=profile_id,
        system=args.system,
    )
    evidence_scope = cast(str, capability["evidence_scope"])
    source_identity = release_source_identity(
        prepared,
        track=track,
        profile=profile_id,
        evidence_scope=evidence_scope,
    )
    requires_protocol = requires_effectiveness_protocol(profile_id)
    if requires_protocol:
        require_frozen_effectiveness_protocol(
            profile_id,
            protocol_approval_id,
            "effectiveness protocol_approval_id",
            system=args.system,
        )
    elif protocol_approval_id is not None:
        raise ValueError(
            "protocol approval identities apply only to judgment-union or confirmation profiles"
        )
    benchmark = resolve_benchmark_profile(
        prepared,
        profile_id=profile_id,
        expected_pool_count=getattr(args, "pool_count", None),
        expected_pool_ids_sha256=getattr(args, "pool_ids_sha256", None),
        expected_pool_receipt_id=getattr(args, "pool_receipt_id", None),
        supplied_pool_ids=read_pool_ids(getattr(args, "pool_file", None)),
    )
    evaluation_package = benchmark.evaluation_package
    run_id = args.run_id or _run_id(args.system)
    request = PatientToTrialRunRequest.for_benchmark(
        benchmark,
        run_id=run_id,
        top_k=args.top_k,
        metric_cutoff=args.metric_cutoff,
        options={},
        clinical_as_of=clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )
    configuration: dict[str, JsonValue]
    stage_rankings: tuple[StageRanking, ...] = ()
    extension = extension_for(args.system)
    if extension is not None and extension.run_from_components is not None:
        result = extension.run_from_components(
            ComponentRun(
                args=args,
                benchmark=benchmark,
                request=request,
                profile_id=profile_id,
                run_id=run_id,
                producer_release_identity=producer_release_identity,
                producer_dependency_environment=producer_dependency_environment,
            )
        )
        candidates = tuple(result.candidates)
        configuration = dict(result.configuration)
        runtime_seconds = result.runtime_seconds
        primary_ranking = result.primary_ranking
        pipeline_depth = result.pipeline_depth
        stage_rankings = result.stage_rankings
    else:
        request = replace(
            request,
            options=_forward_options(
                args,
                benchmark,
                task_input_id=request.snapshot.task_input_id,
            ),
        )
        result = run_release_reference_system(args.system, request)
        candidates = tuple(result.candidates)
        configuration = dict(result.configuration)
        runtime_seconds = result.runtime_seconds
        primary_ranking = result.primary_ranking
        pipeline_depth = result.pipeline_depth
        stage_rankings = result.stage_rankings
    system_input = configuration.get("system_input")
    if not isinstance(system_input, Mapping) or not isinstance(
        system_input.get("system_input_id"), str
    ):
        raise ValueError("reference System omitted its exact System Input identity")
    system_input_id = str(system_input["system_input_id"])
    commit, dirty = release_producer.revalidate_release_producer(
        args,
        system_id=args.system,
        release_identity=producer_release_identity,
        dependency_environment=producer_dependency_environment,
        expected_git_state=producer_git_state,
    )
    configuration.update(
        {
            "release_support": {
                "release_version": RELEASE_VERSION,
                "track": track,
                "task": "patient_to_trial",
                "profile": profile_id,
                "system": args.system,
                "protocol_approval_id": protocol_approval_id,
            },
            "release_source": source_identity,
            "execution_provenance": {
                "git_commit": commit,
                "working_tree_dirty": dirty,
                "environment": {"python": platform.python_version(), "taim": __version__},
                "determinism": "reference-system-defined",
            },
        }
    )
    if producer_release_identity is not None:
        configuration["producer_release_identity"] = cast(JsonValue, producer_release_identity)
        configuration["producer_dependency_environment"] = cast(
            JsonValue,
            producer_dependency_environment,
        )
    output = Path(args.output_dir) / run_id
    stored = write_public_patient_to_trial_run(
        output,
        manifest=RunManifest(
            run_id=run_id,
            system_id=args.system,
            created_at=datetime.now(UTC),
            git_commit=commit,
            configuration=configuration,
            benchmark_lineage=track,
            snapshot_contract_version=request.snapshot.contract_version,
            prepared_snapshot_id=benchmark.prepared_snapshot_id,
            task_input_id=request.snapshot.task_input_id,
            system_input_id=system_input_id,
            evaluation_package_id=evaluation_package.evaluation_package_id,
            benchmark_profile=benchmark.manifest_configuration(),
            primary_ranking=primary_ranking,
            pipeline_depth=pipeline_depth,
            budget_k=args.top_k,
            clinical_as_of=clinical_as_of,
            patient_evidence_profile=request.snapshot.patient_evidence_profile.to_dict(),
            query_patient_versions=tuple(
                (item.patient_id, item.patient_version_id)
                for item in request.snapshot.patient_versions
            ),
            trial_corpus_versions=tuple(
                (item.trial_id, item.trial_version_id) for item in request.snapshot.trial_versions
            ),
            runtime_seconds=runtime_seconds,
            stage_rankings=stage_rankings,
        ),
        candidates=candidates,
        evaluation_package=evaluation_package,
    )
    print_payload(
        {
            "release": RELEASE_VERSION,
            "track": track,
            "task": "patient_to_trial",
            "profile": profile_id,
            "system": args.system,
            "prepared_snapshot_id": stored.manifest.prepared_snapshot_id,
            "task_input_id": stored.manifest.task_input_id,
            "system_input_id": stored.manifest.system_input_id,
            "evaluation_package_id": stored.manifest.evaluation_package_id,
            "run_directory": str(stored.directory),
        }
    )
    return output


def _forward_fixture_run(args: argparse.Namespace) -> int:
    prepared = load_release_fixture(args.track)
    _store_forward(
        prepared,
        args,
        track=args.track,
        profile_id="synthetic-patient-to-trial",
        clinical_as_of=FIXTURE_CLINICAL_AS_OF,
    )
    return 0


def _forward_benchmark_run(args: argparse.Namespace) -> int:
    profile = args.profile or (
        "official-full"
        if args.track == "trec-ct-2021"
        else "confirmation-full"
        if args.track.startswith("trec-ct-")
        else None
    )
    if profile is None:
        raise ValueError("SIGIR 2016 requires --profile description or summary")
    capability = require_supported_combination(
        track=args.track,
        task="patient_to_trial",
        profile=profile,
        system=args.system,
    )
    if capability["evidence_scope"] not in {
        "real_effectiveness",
        "real_effectiveness_requires_protocol",
        "within_pool_effectiveness",
        "within_pool_effectiveness_requires_protocol",
        "external_fidelity_effectiveness",
    }:
        raise ValueError(
            "benchmark run requires a real effectiveness Profile; use fixture run for "
            "synthetic conformance"
        )
    prepared = load_prepared_benchmark(args.data_dir)
    if prepared.snapshot.benchmark_lineage != args.track:
        raise ValueError("prepared Track does not match --track")
    source_profile = source_profile_id(profile)
    if args.track == "sigir-ct-2016" and prepared.dataset_id != f"sigir-ct-2016-{source_profile}":
        raise ValueError("SIGIR prepared query profile does not match --profile")
    _store_forward(
        prepared,
        args,
        track=args.track,
        profile_id=profile,
        clinical_as_of=args.clinical_as_of,
        protocol_approval_id=args.protocol_approval_id,
    )
    return 0


def _forward_validate(args: argparse.Namespace) -> int:
    stored = load_public_patient_to_trial_run(args.run_dir)
    profile = load_profile(cast(str, stored.manifest.benchmark_profile["profile_id"]))
    evaluate_public_patient_to_trial_run(
        stored,
        load_local_evaluation_package(args.run_dir),
        profile,
    )
    support = cast(Mapping[str, object], stored.manifest.configuration["release_support"])
    print_payload(
        {
            "release_version": support["release_version"],
            "track": support["track"],
            "task": stored.manifest.task,
            "profile": support["profile"],
            "profile_definition_sha256": stored.manifest.benchmark_profile["definition_sha256"],
            "system": stored.manifest.system_id,
            "run_id": stored.manifest.run_id,
            "prepared_snapshot_id": stored.manifest.prepared_snapshot_id,
            "task_input_id": stored.manifest.task_input_id,
            "system_input_id": stored.manifest.system_input_id,
            "evaluation_package_id": stored.manifest.evaluation_package_id,
        }
    )
    return 0


def _forward_evaluate(args: argparse.Namespace) -> int:
    stored = load_public_patient_to_trial_run(args.run_dir)
    package = load_local_evaluation_package(args.run_dir)
    profile = load_profile(cast(str, stored.manifest.benchmark_profile["profile_id"]))
    if args.stage is None:
        print_payload(evaluate_public_patient_to_trial_run(stored, package, profile))
        return 0
    stages = {stage.name: stage for stage in stored.stage_rankings}
    try:
        stage = stages[args.stage]
    except KeyError as exc:
        available = ", ".join(sorted(stages)) or "none"
        raise ValueError(f"unknown stage {args.stage!r}; available stages: {available}") from exc
    scorecard = evaluate_public_patient_to_trial_run(
        replace(stored, candidates=stage.candidates),
        package,
        profile,
    )
    print_payload(
        {
            "evaluation_scope": "stage_diagnostic",
            "stage": stage.name,
            "pipeline_depth": stage.pipeline_depth,
            "artifact_hash": stage.artifact_hash,
            "scorecard": scorecard,
        }
    )
    return 0


def _reverse_provenance(
    result,
    *,
    git_state: tuple[str, bool],
) -> TrialToPatientExecutionProvenance:
    commit, dirty = git_state
    model_identity = result.configuration.get("model_identity")
    index_identity = result.configuration.get("index_identity")
    if not isinstance(model_identity, Mapping) or not isinstance(index_identity, Mapping):
        raise ValueError("reverse System omitted model or index identity")
    return TrialToPatientExecutionProvenance(
        git_commit=commit,
        working_tree_dirty=dirty,
        environment={"python": platform.python_version(), "taim": __version__},
        seeds={"deterministic_bm25": 0},
        model_identity=model_identity,
        index_identity=index_identity,
    )


def _store_reverse(resolved, args: argparse.Namespace, *, track: str, profile: str) -> Path:
    system_id = SUBCOMMAND_SYSTEMS[("trial-to-patient", "fixture")]
    producer_release_identity = release_producer.verified_release_identity(args)
    producer_dependency_environment = (
        release_producer.verified_dependency_environment(args, system_id=system_id)
        if producer_release_identity is not None
        else None
    )
    producer_git_state = release_producer.git_state()
    capability = require_supported_combination(
        track=track,
        task="trial_to_patient",
        profile=profile,
        system=system_id,
    )
    source_identity = release_source_identity(
        resolved.prepared,
        track=track,
        profile=profile,
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    run_id = args.run_id or _run_id(system_id)
    request = TrialToPatientRunRequest.for_benchmark(
        resolved,
        run_id=run_id,
        top_k=args.top_k,
        metric_cutoff=args.metric_cutoff,
        options={"b": args.b, "k1": args.k1},
    )
    result = run_trial_to_patient_system(system_id, request)
    final_git_state = release_producer.revalidate_release_producer(
        args,
        system_id=system_id,
        release_identity=producer_release_identity,
        dependency_environment=producer_dependency_environment,
        expected_git_state=producer_git_state,
    )
    configuration = {
        **result.configuration,
        "release_support": {
            "release_version": RELEASE_VERSION,
            "track": track,
            "task": "trial_to_patient",
            "profile": profile,
            "system": system_id,
        },
        "release_source": source_identity,
    }
    if producer_release_identity is not None:
        configuration["producer_release_identity"] = producer_release_identity
        configuration["producer_dependency_environment"] = producer_dependency_environment
    result = replace(result, configuration=configuration)
    output = Path(args.output_dir) / run_id
    stored = write_trial_to_patient_run(
        output,
        resolved=resolved,
        request=request,
        result=result,
        execution_provenance=_reverse_provenance(result, git_state=final_git_state),
        created_at=datetime.now(UTC),
    )
    (output / "evaluation-package.json").write_text(
        json.dumps(
            resolved.evaluation_package.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print_payload(
        {
            "release": RELEASE_VERSION,
            "track": track,
            "task": "trial_to_patient",
            "profile": profile,
            "system": stored.manifest.system_id,
            "prepared_snapshot_id": stored.manifest.prepared_snapshot_id,
            "task_input_id": stored.manifest.task_input_id,
            "system_input_id": stored.manifest.system_input_id,
            "evaluation_package_id": stored.manifest.evaluation_package_id,
            "run_directory": str(stored.directory),
        }
    )
    return output


def _reverse_fixture_run(args: argparse.Namespace) -> int:
    _store_reverse(
        load_reverse_fixture(args.track),
        args,
        track=args.track,
        profile="synthetic-trial-to-patient",
    )
    return 0


def _reverse_benchmark_run(args: argparse.Namespace) -> int:
    resolved = qualify_reverse_viability_profile(
        load_prepared_benchmark(args.data_dir),
        clinical_as_of=args.clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )
    _store_reverse(
        resolved,
        args,
        track="trec-ct-2021",
        profile="trec-ct-2021-reverse-complete10",
    )
    return 0


def _validated_reverse_run(
    run_dir: Path,
) -> tuple[
    StoredTrialToPatientRun,
    TrialToPatientEvaluationPackage,
    Mapping[str, object],
    dict[str, object],
]:
    """Load, release-validate, and evaluate one reverse Local Run."""

    stored = load_trial_to_patient_run(run_dir)
    validate_public_reverse_run_tree(run_dir, stored.manifest)
    support = validate_reverse_release_manifest(stored.manifest)
    try:
        payload = json.loads((run_dir / "evaluation-package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read reverse Evaluation Package: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("reverse Evaluation Package must be a JSON object")
    package = TrialToPatientEvaluationPackage.from_dict(payload)
    metrics = evaluate_trial_to_patient_run(stored, package)
    return stored, package, support, metrics


def _reverse_identity_payload(
    stored: StoredTrialToPatientRun,
    support: Mapping[str, object],
) -> dict[str, object]:
    manifest = stored.manifest
    return {
        "release_version": RELEASE_VERSION,
        "track": support["track"],
        "task": manifest.task,
        "profile": manifest.benchmark_profile["profile_id"],
        "system": manifest.system_id,
        "run_id": manifest.run_id,
        "prepared_snapshot_id": manifest.prepared_snapshot_id,
        "task_input_id": manifest.task_input_id,
        "system_input_id": manifest.system_input_id,
        "evaluation_package_id": manifest.evaluation_package_id,
    }


def _reverse_validate(args: argparse.Namespace) -> int:
    stored, _package, support, _metrics = _validated_reverse_run(args.run_dir)
    print_payload(_reverse_identity_payload(stored, support))
    return 0


def _reverse_evaluate(args: argparse.Namespace) -> int:
    stored, _package, support, metrics = _validated_reverse_run(args.run_dir)
    print_payload({**metrics, **_reverse_identity_payload(stored, support)})
    return 0


def _prepare(args: argparse.Namespace) -> int:
    confirmation = args.track in {"trec-ct-2022", "trec-ct-2023"}
    if confirmation:
        require_frozen_paper_protocol(
            args.protocol_approval_id, "confirmation protocol_approval_id"
        )
    elif args.protocol_approval_id is not None:
        raise ValueError("protocol approval identities apply only to confirmation Tracks")
    if args.track == "trec-ct-2021":
        from taim.data.trec_ct_2021 import prepare_trec_ct_2021

        result = prepare_trec_ct_2021(args.source, args.output_dir)
    elif args.track in {"trec-ct-2022", "trec-ct-2023"}:
        from taim.data.trec_ct import prepare_trec_ct

        result = prepare_trec_ct(
            args.track,
            args.source,
            args.output_dir,
        )
    else:
        from taim.data.sigir_ct_2016 import prepare_sigir_ct_2016

        if args.query_profile not in {"description", "summary"}:
            raise ValueError("SIGIR 2016 requires --query-profile description or summary")
        result = prepare_sigir_ct_2016(
            args.source,
            args.output_dir,
            query_variant=args.query_profile,
        )
    print_payload(
        {
            "release": RELEASE_VERSION,
            "track": args.track,
            "query_profile": args.query_profile,
            "protocol_approval_id": args.protocol_approval_id,
            "output_directory": str(result.output_directory.resolve()),
        }
    )
    return 0


def _inspect_pool(args: argparse.Namespace) -> int:
    is_external = args.profile == EXTERNAL_FIDELITY_TREC_2021_PROFILE
    if not args.profile.endswith("-judgment-union") and not is_external:
        raise ValueError("data pool inspect requires a compute-bounded Profile")
    if args.protocol_approval_id is None:
        raise ValueError("data pool inspect requires --protocol-approval-id")
    require_frozen_effectiveness_protocol(
        args.profile,
        args.protocol_approval_id,
        "pool inspection protocol_approval_id",
    )
    # The pool belongs to the Profile, not to a System: any published System declared for this
    # Track and Profile carries its evidence scope. The first System in the table need not be one.
    declared = [
        row
        for row in cast(Sequence[Mapping[str, JsonValue]], load_release_support()["support_matrix"])
        if (row["track"], row["task"], row["profile"])
        == (args.track, "patient_to_trial", args.profile)
    ]
    if not declared:
        raise ValueError(
            "unsupported release combination: "
            f"Track={args.track}, Task=patient_to_trial, Profile={args.profile}"
        )
    capability = declared[0]
    prepared = load_prepared_benchmark(args.data_dir)
    if prepared.snapshot.benchmark_lineage != args.track:
        raise ValueError("prepared Track does not match --track")
    source_profile = source_profile_id(args.profile)
    if args.track == "sigir-ct-2016" and prepared.dataset_id != f"sigir-ct-2016-{source_profile}":
        raise ValueError("SIGIR prepared query profile does not match --profile")
    release_source_identity(
        prepared,
        track=args.track,
        profile=args.profile,
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    inspection = (
        inspect_external_fidelity_profile(
            prepared,
            profile_id=args.profile,
            supplied_pool_ids=read_pool_ids(args.pool_file) or (),
            expected_pool_count=load_profile(args.profile).expected_pool_count or 0,
        )
        if is_external
        else inspect_judgment_union_profile(prepared, profile_id=args.profile)
    )
    print_payload(
        {
            "release": RELEASE_VERSION,
            "track": args.track,
            "task": "patient_to_trial",
            "protocol_approval_id": args.protocol_approval_id,
            **inspection,
        }
    )
    return 0


def _release_validate(args: argparse.Namespace) -> int:
    from taim.benchmark_release import validate_benchmark_release

    report = validate_benchmark_release(
        args.candidate,
        args.manifest,
        execute_quick_start=args.runtime,
        quick_start_executable=sys.argv[0] if args.runtime else None,
    )
    print_payload(report)
    return 0


def _bundle_validate(args: argparse.Namespace) -> int:
    from taim.release_result_bundle import validate_release_result_bundle

    print_payload(validate_release_result_bundle(args.bundle_dir))
    return 0


def _add_forward_options(
    parser: argparse.ArgumentParser,
    *,
    real: bool,
    extensions: Sequence[PipelineExtension] | None = None,
) -> None:
    extensions = pipeline_extensions() if extensions is None else extensions
    parser.add_argument(
        "--system",
        choices=tuple(
            system for system, tasks in SYSTEM_TASKS.items() if "patient_to_trial" in tasks
        ),
        metavar="SYSTEM",
        default=next(
            (
                extension.default_system
                for extension in extensions
                if extension.default_system is not None
                and extension.system_tasks[extension.default_system] == "patient_to_trial"
            ),
            next(
                (system for system, tasks in SYSTEM_TASKS.items() if "patient_to_trial" in tasks),
                None,
            ),
        ),
        help="System to run; support list reports the Systems each Track and Profile declare. "
        "Omitted, the System the release's quick start runs",
    )
    add_run_output_options(parser, top_k=DEFAULT_RETRIEVAL_DEPTH)
    option_adders = dict.fromkeys(extension.add_run_options for extension in extensions)
    for add_options in option_adders:
        add_options(parser)
    add_release_identity_options(
        parser,
        rule="only runs given all three can enter a Result Bundle, and judgment-union and "
        "external-fidelity Profiles require them",
    )
    if real:
        add_real_input_options(parser)
        add_pool_options(parser)


def _add_reverse_options(parser: argparse.ArgumentParser, *, real: bool) -> None:
    add_run_output_options(parser, top_k=20)
    add_bm25_options(parser)
    add_release_identity_options(parser, rule="only runs given all three can enter a Result Bundle")
    if real:
        add_real_input_options(parser)


def _add_direction_commands(
    parent: argparse.ArgumentParser,
    *,
    reverse: bool,
    extensions: Sequence[PipelineExtension],
) -> None:
    commands = parent.add_subparsers(dest="operation", required=True)
    fixture = commands.add_parser("fixture", help="run deterministic conformance data")
    fixture_commands = fixture.add_subparsers(dest="fixture_operation", required=True)
    fixture_run = fixture_commands.add_parser(
        "run",
        help="run one System on the packaged synthetic fixture; its scores are conformance "
        "evidence, not effectiveness",
    )
    fixture_run.add_argument(
        "--track",
        choices=SUPPORTED_TRACKS,
        default="trec-ct-2021",
        help="Track whose synthetic fixture to run",
    )
    if reverse:
        _add_reverse_options(fixture_run, real=False)
        fixture_run.set_defaults(handler=_reverse_fixture_run)
    else:
        _add_forward_options(fixture_run, real=False, extensions=extensions)
        fixture_run.set_defaults(handler=_forward_fixture_run)
    benchmark = commands.add_parser("benchmark", help="run a qualified real-data profile")
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_operation", required=True)
    if reverse:
        benchmark_run = benchmark_commands.add_parser(
            "run", help="run the TREC 2021 reverse viability Profile on prepared TREC 2021 files"
        )
        _add_reverse_options(benchmark_run, real=True)
        benchmark_run.set_defaults(handler=_reverse_benchmark_run)
    else:
        benchmark_run = benchmark_commands.add_parser(
            "run", help="run one System on a real Profile of a prepared Track"
        )
        benchmark_run.add_argument(
            "--track", choices=SUPPORTED_TRACKS, required=True, help="Track of the prepared files"
        )
        benchmark_run.add_argument(
            "--profile",
            help="Benchmark Profile to run; omitted, official-full on TREC 2021 and "
            "confirmation-full on TREC 2022/2023. SIGIR 2016 requires one",
        )
        _add_forward_options(benchmark_run, real=True, extensions=extensions)
        benchmark_run.set_defaults(handler=_forward_benchmark_run)
    run = commands.add_parser("run", help="validate or evaluate a closed run")
    run_commands = run.add_subparsers(dest="run_operation", required=True)
    validate = run_commands.add_parser(
        "validate", help="check a run directory against its manifest and print its identities"
    )
    validate.add_argument(
        "--run-dir", type=Path, required=True, help="run directory a run command wrote"
    )
    validate.set_defaults(handler=_reverse_validate if reverse else _forward_validate)
    evaluate = run_commands.add_parser(
        "evaluate", help="validate a run directory, then score its Primary Ranking"
    )
    evaluate.add_argument(
        "--run-dir", type=Path, required=True, help="run directory a run command wrote"
    )
    if not reverse:
        evaluate.add_argument(
            "--stage",
            help="evaluate one named stage as a diagnostic, leaving the Primary Ranking unchanged",
        )
    evaluate.set_defaults(handler=_reverse_evaluate if reverse else _forward_evaluate)
    add_submission_commands(commands, reverse=reverse)
    if not reverse:
        command_adders = dict.fromkeys(
            extension.add_patient_to_trial_commands
            for extension in extensions
            if extension.add_patient_to_trial_commands is not None
        )
        for add_commands in command_adders:
            add_commands(commands)


def build_parser(
    extensions: Sequence[PipelineExtension] | None = None,
) -> argparse.ArgumentParser:
    """The ``trial-benchmark`` parser, with the options and commands of ``extensions``.

    ``extensions`` defaults to every pipeline extension this installation ships.
    """

    extensions = pipeline_extensions() if extensions is None else extensions
    parser = argparse.ArgumentParser(
        prog="trial-benchmark",
        description="TAIM Benchmark Release 0.1",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    support = commands.add_parser("support", help="inspect executable release capabilities")
    support_commands = support.add_subparsers(dest="support_operation", required=True)
    support_list = support_commands.add_parser(
        "list",
        help="print the Track, Task, Profile and System combinations this release executes",
    )
    support_list.set_defaults(handler=_support_list)
    if any("patient_to_trial" in tasks for tasks in SYSTEM_TASKS.values()):
        forward = commands.add_parser("patient-to-trial", help="patient query against trial corpus")
        _add_direction_commands(forward, reverse=False, extensions=extensions)
    if any("trial_to_patient" in tasks for tasks in SYSTEM_TASKS.values()):
        reverse = commands.add_parser("trial-to-patient", help="trial query against patient corpus")
        _add_direction_commands(reverse, reverse=True, extensions=extensions)
    data = commands.add_parser("data", help="prepare checksum-locked local source files")
    data_commands = data.add_subparsers(dest="data_operation", required=True)
    prepare = data_commands.add_parser(
        "prepare",
        help="verify acquired source files against the Track's packaged lock and write the "
        "prepared files every run reads",
    )
    prepare.add_argument(
        "--track", choices=SUPPORTED_TRACKS, required=True, help="Track to prepare"
    )
    prepare.add_argument(
        "--query-profile",
        help="SIGIR 2016 query text, description or summary; required there, ignored elsewhere",
    )
    prepare.add_argument(
        "--source",
        type=Path,
        required=True,
        help="directory holding the Track's acquired source files, as its acquisition guide names "
        "them",
    )
    prepare.add_argument(
        "--output-dir", type=Path, required=True, help="directory that receives the prepared files"
    )
    prepare.add_argument(
        "--protocol-approval-id",
        type=sha256_identity,
        help="required before a TREC 2022/2023 source file is opened",
    )
    prepare.set_defaults(handler=_prepare)
    pool = data_commands.add_parser(
        "pool", help="inspect a compute-bounded corpus before effectiveness execution"
    )
    pool_commands = pool.add_subparsers(dest="pool_operation", required=True)
    pool_inspect = pool_commands.add_parser(
        "inspect",
        help="print the receipt of a judgment-union or external-fidelity pool, whose values every "
        "run on that Profile must pass",
    )
    pool_inspect.add_argument(
        "--track", choices=SUPPORTED_TRACKS, required=True, help="Track of the prepared files"
    )
    pool_inspect.add_argument(
        "--profile", required=True, help="judgment-union or external-fidelity Profile to inspect"
    )
    pool_inspect.add_argument(
        "--data-dir", type=Path, required=True, help="directory written by data prepare"
    )
    pool_inspect.add_argument(
        "--pool-file",
        type=Path,
        help="trial IDs written by data external-pool prepare; external-fidelity Profile only",
    )
    pool_inspect.add_argument(
        "--protocol-approval-id",
        type=sha256_identity,
        help="SHA-256 of the frozen protocol that governs the Profile; always required",
    )
    pool_inspect.set_defaults(handler=_inspect_pool)
    external_pool = data_commands.add_parser(
        "external-pool",
        help="verify an external source lock and extract ID-only pool membership",
    )
    external_pool_commands = external_pool.add_subparsers(
        dest="external_pool_operation", required=True
    )
    external_pool_prepare = external_pool_commands.add_parser(
        "prepare",
        help="check an external corpus against its packaged lock and write its trial IDs, sorted, "
        "as a JSON array",
    )
    external_pool_prepare.add_argument(
        "--profile",
        choices=(EXTERNAL_FIDELITY_TREC_2021_PROFILE,),
        required=True,
        help="external-fidelity Profile whose source to verify",
    )
    external_pool_prepare.add_argument(
        "--source",
        type=Path,
        required=True,
        help="the external corpus file its acquisition guide names",
    )
    external_pool_prepare.add_argument(
        "--output", type=Path, required=True, help="file that receives the trial IDs"
    )
    external_pool_prepare.set_defaults(handler=_prepare_external_pool)
    release = commands.add_parser("release", help="validate a generated public candidate")
    release_commands = release.add_subparsers(dest="release_operation", required=True)
    release_validate = release_commands.add_parser(
        "validate", help="check a candidate release tree against its frozen manifest"
    )
    release_validate.add_argument(
        "--candidate", type=Path, required=True, help="root of the candidate release tree"
    )
    release_validate.add_argument(
        "--manifest", type=Path, required=True, help="the candidate's frozen release manifest"
    )
    release_validate.add_argument(
        "--runtime",
        action="store_true",
        help="also execute the manifest's quick-start commands with this executable",
    )
    release_validate.set_defaults(handler=_release_validate)
    bundle = commands.add_parser("bundle", help="validate a Published Result Bundle")
    bundle_commands = bundle.add_subparsers(dest="bundle_operation", required=True)
    bundle_validate = bundle_commands.add_parser(
        "validate", help="recompute a Published Result Bundle's checksums, identities and metrics"
    )
    bundle_validate.add_argument(
        "--bundle-dir", type=Path, required=True, help="the Result Bundle directory"
    )
    bundle_validate.set_defaults(handler=_bundle_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
