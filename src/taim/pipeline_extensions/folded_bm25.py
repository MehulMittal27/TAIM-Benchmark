"""The checksum-bound lexical query transformation release unit."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from taim.pipeline_extensions import PipelineExtension, add_argument_once
from taim.release_reference_systems import _option, run_normalized_system
from taim.schemas import PIPELINE_DEPTH_RETRIEVAL, PRIMARY_RANKING_CANDIDATES
from taim.snapshot import CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT
from taim.system_contracts import StrictSystemOptions, SystemRunRequest, SystemRunResult


class ReleaseFoldedBM25System(StrictSystemOptions):
    """BM25 over a checksum-bound, locally derived folded-query bundle."""

    system_id = "bm25-folded"
    option_names = frozenset({"b", "folded_query_bundle", "k1"})
    required_capabilities = frozenset(
        {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
    )
    optional_capabilities: frozenset[str] = frozenset()

    def run(self, request: SystemRunRequest) -> SystemRunResult:
        from taim.baselines.bm25 import BM25Retriever, bm25_ranking_configuration
        from taim.folded_query_bundle import validate_folded_query_bundle

        raw_bundle = _option(request, "folded_query_bundle")
        if not isinstance(raw_bundle, Mapping):
            raise ValueError("bm25-folded requires a folded query bundle object")
        queries, bundle_id = validate_folded_query_bundle(
            raw_bundle,
            expected_task_input_id=request.snapshot.task_input_id,
            expected_topic_ids=tuple(topic.topic_id for topic in request.snapshot.topics),
        )
        topics = tuple(
            replace(topic, canonical_text=query.canonical_text)
            for topic, query in zip(request.snapshot.topics, queries, strict=True)
        )
        started_at = time.perf_counter()
        retriever = BM25Retriever(
            request.snapshot.trials,
            k1=_option(request, "k1"),  # type: ignore[arg-type]
            b=_option(request, "b"),  # type: ignore[arg-type]
        )
        candidates = retriever.retrieve_many(
            topics,
            run_id=request.run_id,
            system_id=self.system_id,
            top_k=request.top_k,
        )
        runtime_seconds = time.perf_counter() - started_at
        ranking = bm25_ranking_configuration(
            k1=retriever.k1,
            b=retriever.b,
            requested_top_k=request.top_k,
            document_count=len(retriever.documents),
        )
        return SystemRunResult(
            candidates=tuple(candidates),
            configuration={
                **ranking,
                "model_identity": {"kind": "none", "reason": "deterministic_bm25"},
                "index_identity": {
                    "implementation": "taim-okapi-bm25-v1",
                    "persistent": False,
                    "corpus_statistics": retriever.corpus_statistics,
                },
                "query_transformation": {
                    "bundle_id": bundle_id,
                    "policy": raw_bundle["policy"],
                    "provenance": raw_bundle["provenance"],
                    "topic_count": len(queries),
                },
                "metric_cutoff": request.metric_cutoff,
                "corpus_statistics": retriever.corpus_statistics,
                "runtime_seconds": runtime_seconds,
            },
            runtime_seconds=runtime_seconds,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth=PIPELINE_DEPTH_RETRIEVAL,
        )


def _add_options(parser: argparse.ArgumentParser) -> None:
    add_argument_once(parser, "--k1", type=float, default=1.2)
    add_argument_once(parser, "--b", type=float, default=0.75)
    add_argument_once(parser, "--folded-query-bundle", type=Path)


def _options(args, benchmark, task_input_id):
    from taim.folded_query_bundle import validate_folded_query_bundle
    from taim.release_cli import _read_json_object

    if args.folded_query_bundle is None:
        raise ValueError("bm25-folded requires --folded-query-bundle")
    bundle = _read_json_object(args.folded_query_bundle, "folded query bundle")
    validate_folded_query_bundle(
        bundle,
        expected_task_input_id=task_input_id,
        expected_topic_ids=tuple(topic.topic_id for topic in benchmark.topics),
    )
    return {"b": args.b, "folded_query_bundle": bundle, "k1": args.k1}


def _run(system_id, request):
    return run_normalized_system(ReleaseFoldedBM25System(), request)


def _prepare_folded_queries(args: argparse.Namespace) -> int:
    import json
    from typing import cast

    from taim.data import load_prepared_benchmark
    from taim.entity_versions import source_grounded_patient_evidence_profile
    from taim.folded_query_bundle import build_folded_query_bundle, validate_folded_query_bundle
    from taim.release_cli import (
        DEFAULT_METRIC_CUTOFF,
        DEFAULT_RETRIEVAL_DEPTH,
        _print,
        _read_pool_ids,
        _verified_dependency_environment,
        _verified_release_identity,
    )
    from taim.release_profiles import EXTERNAL_FIDELITY_TREC_2021_PROFILE, resolve_benchmark_profile
    from taim.release_support import (
        RELEASE_VERSION,
        SUBCOMMAND_SYSTEMS,
        release_source_identity,
        require_frozen_effectiveness_protocol,
        require_supported_combination,
    )

    prepared = load_prepared_benchmark(args.data_dir)
    if prepared.snapshot.benchmark_lineage != "trec-ct-2021":
        raise ValueError("folded queries are qualified only for TREC CT 2021")
    # The folded queries are the input of the System this subcommand is bound to.
    system = SUBCOMMAND_SYSTEMS[("patient-to-trial", "queries")]
    capability = require_supported_combination(
        track="trec-ct-2021",
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
        if _verified_release_identity(args) is None:
            raise ValueError(
                "compute-bounded execution requires --release-manifest, --package-artifact, and "
                "--dependency-lock"
            )
        _verified_dependency_environment(args, system_id=system)
    elif args.protocol_approval_id is not None:
        raise ValueError("full-corpus folded queries cannot claim a protocol approval")
    release_source_identity(
        prepared,
        track="trec-ct-2021",
        profile=args.profile,
        evidence_scope=cast(str, capability["evidence_scope"]),
    )
    benchmark = resolve_benchmark_profile(
        prepared,
        profile_id=args.profile,
        expected_pool_count=args.pool_count,
        expected_pool_ids_sha256=args.pool_ids_sha256,
        expected_pool_receipt_id=args.pool_receipt_id,
        supplied_pool_ids=_read_pool_ids(args.pool_file),
    )
    request = SystemRunRequest.for_benchmark(
        benchmark,
        run_id="folded-query-preparation",
        top_k=DEFAULT_RETRIEVAL_DEPTH,
        metric_cutoff=DEFAULT_METRIC_CUTOFF,
        options={},
        clinical_as_of=args.clinical_as_of,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
        identity_version="2.0",
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
    _print(
        {
            "release": RELEASE_VERSION,
            "track": "trec-ct-2021",
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
    from taim.entity_versions import parse_clinical_as_of
    from taim.release_cli import (
        _add_release_identity_options,
        _positive_integer,
        _sha256_identity,
    )

    queries = commands.add_parser(
        "queries", help="prepare the licensed local folded-query transformation"
    )
    query_commands = queries.add_subparsers(dest="query_operation", required=True)
    prepare = query_commands.add_parser("prepare")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--clinical-as-of", type=parse_clinical_as_of, required=True)
    prepare.add_argument("--profile", default="official-full")
    prepare.add_argument("--pool-count", type=_positive_integer)
    prepare.add_argument("--pool-ids-sha256", type=_sha256_identity)
    prepare.add_argument("--pool-receipt-id", type=_sha256_identity)
    prepare.add_argument("--pool-file", type=Path)
    prepare.add_argument("--protocol-approval-id", type=_sha256_identity)
    prepare.add_argument("--snomed-release", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    _add_release_identity_options(prepare)
    prepare.set_defaults(handler=_prepare_folded_queries)


EXTENSION = PipelineExtension(
    system_tasks={"bm25-folded": "patient_to_trial"},
    pipeline_depth="retrieval",
    external_baseline=False,
    subcommands={("patient-to-trial", "queries"): "bm25-folded"},
    run=_run,
    add_run_options=_add_options,
    run_options=_options,
    add_patient_to_trial_commands=_add_commands,
    ci_jobs=({"extras": ("folding",), "imports": ("taim.query_folding",)},),
)
