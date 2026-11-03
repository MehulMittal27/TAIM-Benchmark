"""The checksum-bound lexical query transformation release unit."""

from __future__ import annotations

import argparse
from pathlib import Path

from taim.patient_to_trial import (
    PatientToTrialRunRequest,
    PatientToTrialRunResult,
)
from taim.pipeline_extensions import PipelineExtension, RankingExecution, add_argument_once
from taim.pipelines.bm25.folded import ReleaseFoldedBM25System
from taim.release_reference_systems import run_normalized_system


def _add_options(parser: argparse.ArgumentParser) -> None:
    from taim.cli_support import add_bm25_options

    add_bm25_options(parser)
    add_argument_once(
        parser,
        "--folded-query-bundle",
        type=Path,
        help="folded queries written by queries prepare for the same Task Input; required by "
        "the Systems that read folded queries",
    )


def _options(args, benchmark, task_input_id):
    from taim.cli_support import read_json_object
    from taim.folded_query_bundle import validate_folded_query_bundle

    if args.folded_query_bundle is None:
        raise ValueError("bm25-folded requires --folded-query-bundle")
    bundle = read_json_object(args.folded_query_bundle, "folded query bundle")
    validate_folded_query_bundle(
        bundle,
        expected_task_input_id=task_input_id,
        expected_topic_ids=tuple(topic.topic_id for topic in benchmark.topics),
    )
    return {"b": args.b, "folded_query_bundle": bundle, "k1": args.k1}


def _run(system_id: str, request: PatientToTrialRunRequest) -> PatientToTrialRunResult:
    return run_normalized_system(ReleaseFoldedBM25System(), request)


def _prepare_folded_queries(args: argparse.Namespace) -> int:
    import json
    from typing import cast

    from taim.cli_support import (
        DEFAULT_METRIC_CUTOFF,
        DEFAULT_RETRIEVAL_DEPTH,
        print_payload,
        read_pool_ids,
    )
    from taim.data import load_prepared_benchmark
    from taim.entity_versions import source_grounded_patient_evidence_profile
    from taim.folded_query_bundle import (
        build_folded_query_bundle,
        require_qualified_snomed_release,
        validate_folded_query_bundle,
    )
    from taim.release_producer import verified_dependency_environment, verified_release_identity
    from taim.release_profiles import EXTERNAL_FIDELITY_TREC_2021_PROFILE, resolve_benchmark_profile
    from taim.release_support import (
        RELEASE_VERSION,
        SUBCOMMAND_SYSTEMS,
        release_source_identity,
        require_frozen_effectiveness_protocol,
        require_supported_combination,
    )

    # The release is checked first: a missing or different edition is refused in about a
    # second, not after the multi-gigabyte prepared corpus has loaded.
    require_qualified_snomed_release(args.snomed_release)
    prepared = load_prepared_benchmark(args.data_dir)
    track = prepared.snapshot.benchmark_lineage
    # The folded queries are the input of the System this subcommand is bound to, so its support
    # rows decide the Tracks and Profiles they are prepared for.
    system = SUBCOMMAND_SYSTEMS[("patient-to-trial", "queries")]
    capability = require_supported_combination(
        track=track,
        task="patient_to_trial",
        profile=args.profile,
        system=system,
    )
    if (
        args.profile.endswith("-judgment-union")
        or args.profile == EXTERNAL_FIDELITY_TREC_2021_PROFILE
    ):
        require_frozen_effectiveness_protocol(
            args.profile,
            args.protocol_approval_id,
            "effectiveness protocol_approval_id",
        )
        if verified_release_identity(args) is None:
            raise ValueError(
                "compute-bounded execution requires --release-manifest, --package-artifact, and "
                "--dependency-lock"
            )
        verified_dependency_environment(args, system_id=system)
    elif args.protocol_approval_id is not None:
        raise ValueError("full-corpus folded queries cannot claim a protocol approval")
    release_source_identity(
        prepared,
        track=track,
        profile=args.profile,
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    benchmark = resolve_benchmark_profile(
        prepared,
        profile_id=args.profile,
        expected_pool_count=args.pool_count,
        expected_pool_ids_sha256=args.pool_ids_sha256,
        expected_pool_receipt_id=args.pool_receipt_id,
        supplied_pool_ids=read_pool_ids(args.pool_file),
    )
    request = PatientToTrialRunRequest.for_benchmark(
        benchmark,
        run_id="folded-query-preparation",
        top_k=DEFAULT_RETRIEVAL_DEPTH,
        metric_cutoff=DEFAULT_METRIC_CUTOFF,
        options={},
        clinical_as_of=args.clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )
    payload = build_folded_query_bundle(
        request.snapshot.topics,
        task_input_id=request.snapshot.task_input_id,
        snomed_release=args.snomed_release,
    )
    queries, bundle_id = validate_folded_query_bundle(
        payload,
        expected_task_input_id=request.snapshot.task_input_id,
        expected_topic_ids=tuple(topic.topic_id for topic in request.snapshot.topics),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print_payload(
        {
            "release": RELEASE_VERSION,
            "track": track,
            "task": "patient_to_trial",
            "profile": args.profile,
            "system": system,
            "task_input_id": request.snapshot.task_input_id,
            "folded_query_bundle_id": bundle_id,
            "topics": len(queries),
            "output": str(args.output.resolve()),
        }
    )
    return 0


def _add_commands(commands) -> None:
    import json

    from taim.cli_support import (
        add_pool_options,
        add_real_input_options,
        add_release_identity_options,
    )
    from taim.folded_query_bundle import SNOMED_EDITION_LOCK

    edition = json.loads(SNOMED_EDITION_LOCK.read_text(encoding="utf-8"))["edition"]
    queries = commands.add_parser(
        "queries", help="prepare the licensed local folded-query transformation"
    )
    query_commands = queries.add_subparsers(dest="query_operation", required=True)
    prepare = query_commands.add_parser(
        "prepare",
        help="fold the queries of one Task Input with a local SNOMED CT release and write the "
        "folded-query bundle",
    )
    add_real_input_options(prepare)
    prepare.add_argument(
        "--profile",
        default="official-full",
        help="Profile whose Task Input the queries are folded for; bm25-folded's support rows "
        "name the Profiles it runs on",
    )
    add_pool_options(prepare)
    prepare.add_argument(
        "--snomed-release",
        type=Path,
        required=True,
        help=f"your SNOMED CT RF2 release directory, which must be the {edition}. It is "
        "licensed data you supply under your own SNOMED CT licence; TAIM ships none. It is read, "
        "never copied into an output",
    )
    prepare.add_argument(
        "--output", type=Path, required=True, help="file that receives the folded-query bundle"
    )
    add_release_identity_options(
        prepare, rule="judgment-union and external-fidelity Profiles require them"
    )
    prepare.set_defaults(handler=_prepare_folded_queries)


EXTENSION = PipelineExtension(
    system_tasks={"bm25-folded": "patient_to_trial"},
    pipeline_depth="retrieval",
    external_baseline=False,
    subcommands={("patient-to-trial", "queries"): "bm25-folded"},
    patient_to_trial=RankingExecution(run=_run),
    add_run_options=_add_options,
    run_options=_options,
    add_patient_to_trial_commands=_add_commands,
    ci_jobs=({"extras": ("folding",), "imports": ("taim.query_folding",)},),
)
