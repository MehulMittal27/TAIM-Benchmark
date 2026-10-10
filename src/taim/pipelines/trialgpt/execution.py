"""Resolve publication inputs and execute TrialGPT with its bound generation cache."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from taim.generation import GenerationCache, GenerationCacheRunLease
from taim.model_codex import codex_preflight

if TYPE_CHECKING:
    from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialRunResult

_SYSTEM_ID = "TrialGPT-TAIM-Luna-v1"


def run_publication(
    system_id: str,
    request: PatientToTrialRunRequest,
    *,
    qualification_topic_ids: Sequence[str] | None = None,
) -> PatientToTrialRunResult:
    """Resolve physical TrialGPT inputs into the frozen publication System Input.

    ``qualification_topic_ids`` is for a qualification gate only: the run covers those Task
    Input topics, and its System Input names them. A row passes none and covers every topic.
    """

    if system_id != _SYSTEM_ID:
        raise ValueError(f"System {system_id!r} is not an exported direct reference System")
    from taim.patient_to_trial import bind_system_capabilities
    from taim.pipelines.trialgpt.publication import (
        TRIALGPT_PUBLICATION_CANDIDATE_DEPTH,
        load_frozen_trialgpt_retrieval,
        load_publication_contract,
        validate_frozen_retrieval_for_release,
        validate_frozen_retrieval_for_snapshot,
        validate_frozen_retrieval_for_task_input,
        validate_publication_preflight,
    )
    from taim.pipelines.trialgpt.retrieval_producer import validate_query_plan_receipt_binding
    from taim.pipelines.trialgpt.system import (
        TrialGPTTAIMLunaV1System,
        trialgpt_codex_model,
        trialgpt_publication_method_contract,
        trialgpt_run_workspace,
    )

    def required(name: str, expected: type) -> object:
        value = request.options.get(name)
        if not isinstance(value, expected):
            raise ValueError(f"TrialGPT publication requires {name}")
        return value

    publication_contract_path = cast(Path, required("publication_contract_path", Path))
    frozen_retrieval_path = cast(Path, required("frozen_retrieval_path", Path))
    frozen_retrieval_lock_path = cast(Path, required("frozen_retrieval_lock_path", Path))
    execution_workspace = cast(Path, required("workspace", Path))
    controller_git_commit = cast(str, required("controller_git_commit", str))
    retrieval_source_snapshot_id = cast(str, required("retrieval_source_snapshot_id", str))
    evaluation_package_id = cast(str, required("evaluation_package_id", str))
    if request.options.get("controller_working_tree_dirty") is not False:
        raise ValueError("TrialGPT publication requires a clean public release checkout")
    generation_workers = request.options.get("generation_workers", 8)
    if isinstance(generation_workers, bool) or not isinstance(generation_workers, int):
        raise ValueError("TrialGPT publication generation_workers must be an integer")
    codex_executable = request.options.get("codex_executable", "codex")
    if not isinstance(codex_executable, str) or not codex_executable:
        raise ValueError("TrialGPT publication codex_executable must be a non-empty string")
    executing_release_identity = request.options.get("executing_release_identity")

    retrieval = load_frozen_trialgpt_retrieval(
        frozen_retrieval_path, lock_path=frozen_retrieval_lock_path
    )
    validate_query_plan_receipt_binding(retrieval.lock)
    validate_frozen_retrieval_for_release(
        retrieval,
        release_identity=cast(Mapping[str, str], executing_release_identity)
        if isinstance(executing_release_identity, Mapping)
        else None,
    )
    validate_frozen_retrieval_for_snapshot(
        retrieval,
        benchmark_lineage=request.snapshot.benchmark_lineage,
        retrieval_source_snapshot_id=retrieval_source_snapshot_id,
        topic_ids=[topic.topic_id for topic in request.snapshot.topics],
        trial_ids=[trial.trial_id for trial in request.snapshot.trials],
    )
    validate_frozen_retrieval_for_task_input(
        retrieval,
        task_input_id=request.snapshot.task_input_id,
    )
    topic_ids = [topic.topic_id for topic in request.snapshot.topics]
    if qualification_topic_ids is not None and (
        not qualification_topic_ids
        or len(set(qualification_topic_ids)) != len(qualification_topic_ids)
        or not set(qualification_topic_ids) <= set(topic_ids)
    ):
        raise ValueError("qualification topics must be distinct Task Input topics")
    provider = trialgpt_codex_model(codex_executable=codex_executable)
    provider_configuration = codex_preflight(provider.configuration)
    contract = load_publication_contract(publication_contract_path)
    validate_publication_preflight(
        contract,
        taim_git_commit=controller_git_commit,
        working_tree_dirty=False,
        prepared_snapshot_id=request.snapshot.prepared_snapshot_id,
        evaluation_package_id=evaluation_package_id,
        frozen_retrieval=retrieval,
        provider_configuration=provider_configuration,
        method_configuration=trialgpt_publication_method_contract(),
        generation_workers=generation_workers,
        topic_ids=topic_ids,
    )
    system = TrialGPTTAIMLunaV1System(
        model=provider,
        frozen_retrieval=retrieval,
        execution_workspace=execution_workspace,
    )
    effective_request = bind_system_capabilities(
        system,
        replace(
            request,
            options=MappingProxyType(
                {
                    "publication_contract_id": contract["contract_id"],
                    "frozen_retrieval_lock_sha256": retrieval.lock_sha256,
                    "frozen_retrieval_artifact_sha256": retrieval.artifact_sha256,
                    "provider_configuration": provider_configuration,
                    "generation_workers": generation_workers,
                    "retrieval_depth": 2_000,
                    "llm_candidate_depth": TRIALGPT_PUBLICATION_CANDIDATE_DEPTH,
                    **(
                        {"qualification_topic_ids": list(qualification_topic_ids)}
                        if qualification_topic_ids is not None
                        else {}
                    ),
                }
            ),
        ),
    )
    run_workspace = trialgpt_run_workspace(execution_workspace.resolve(), request.run_id)
    binding_path = run_workspace / "model-generation-v1-cache-binding.json"
    binding = {
        "contract_id": contract["contract_id"],
        "run_id": request.run_id,
        "system_input_id": effective_request.system_input_id,
    }
    cache_path = run_workspace / "model-generation-v1-cache.sqlite3"
    if cache_path.exists():
        if (
            not binding_path.is_file()
            or json.loads(binding_path.read_text(encoding="utf-8")) != binding
        ):
            raise ValueError("existing TrialGPT cache is not an in-place resume of this run")
    else:
        run_workspace.mkdir(parents=True, exist_ok=True)
        binding_path.write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    cache = GenerationCache(cache_path)
    try:
        with GenerationCacheRunLease(cache_path, run_id=request.run_id):
            system.model = trialgpt_codex_model(
                codex_executable=codex_executable, cache=cache, reuse_unresolved=True
            )
            result = system.run(effective_request)
    finally:
        cache.close()
    return replace(
        result,
        configuration={
            **result.configuration,
            "system_input": effective_request.system_input_record_dict(),
            "generation_cache": {
                **cast(Mapping[str, object], result.configuration["generation_cache"]),
                "path": str(cache_path),
            },
        },
    )
