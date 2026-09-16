"""Small, isolated boundary helpers for the TrialMatchAI adapter.

This module deliberately does not import TrialMatchAI.  The upstream checkout is
an external system and must be invoked as a pinned subprocess.  These helpers
stage a deterministic patient input and normalize upstream ranking artifacts
into TAIM's minimal Candidate schema.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from taim.artifacts import sha256_file
from taim.executables import ExecutableNotFoundError, resolve_executable
from taim.schemas import PRIMARY_RANKING_CANDIDATES, Candidate, StageRanking
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_TYPED_PATIENT_CORE,
    BenchmarkTopic,
)
from taim.system_contracts import StrictSystemOptions

if TYPE_CHECKING:
    from taim.system_contracts import SystemRunRequest, SystemRunResult

PATIENT_RENDERER_VERSION = "trialmatchai-patient-renderer-v1"
TRIALMATCHAI_REQUIRED_CAPABILITIES = frozenset({CAPABILITY_CANONICAL_PATIENT_TEXT})
TRIALMATCHAI_OPTIONAL_CAPABILITIES = frozenset({CAPABILITY_TYPED_PATIENT_CORE})
TRIALMATCHAI_CONCEPT_STORE_MANIFEST_FILENAME = "concept-store-manifest.json"
TRIALMATCHAI_CONCEPT_STORE_MANIFEST_VERSION = "2"
TRIALMATCHAI_L4_SYSTEM_ID = "trialmatchai-current-cuda-l4-trec21-development-v3"
TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID = "trialmatchai-current-cuda-l4-trec22-development-v3"
TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID = "trialmatchai-current-cuda-l4-trec23-development-v3"
TRIALMATCHAI_L4_REPOSITORY = "https://github.com/MehulMittal27/TrialMatchAI.git"
TRIALMATCHAI_L4_COMMIT = "7eba8f399336fcd988b00ce98f2e025e5ec04119"
TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH = "src/trialmatchai/config/taim_l4_cuda.json"
TRIALMATCHAI_L4_CONFIG_SHA256 = (
    "sha256:a6359cadb1e1db3a69ce9476b0ca5ca40169e986792d36d006b2309c3818dfd5"
)
# The four retrieval-depth variables that used to live here are gone, not relaxed. They
# pinned first-level retrieval to 15 trials of 26,149 and the shortlist to floor(15/3),
# so a mean of 3.9 trials per topic ever reached eligibility assessment. They entered the
# repository as one-topic smoke caps and were frozen into a System contract without a
# recorded reason. The frozen config already carried the correct depth - its ``search``
# block matches upstream v0.7.0 byte for byte - so removing the overrides restores the
# configured funnel and required no config change. Depth is now owned solely by the
# digest-verified config, and ``TRIALMATCHAI_L4_FORBIDDEN_ENVIRONMENT`` refuses a run if a
# stale shell export tries to reintroduce them.
TRIALMATCHAI_L4_REQUIRED_ENVIRONMENT = (("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),)

# Absence is asserted as strictly as presence: these four are the removed caps, and an
# inherited export of any of them would silently re-narrow retrieval while every digest
# still verified.
TRIALMATCHAI_L4_FORBIDDEN_ENVIRONMENT = (
    "TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS",
    "TRIALMATCHAI_FIRST_LEVEL_PER_CHANNEL_SIZE",
    "TRIALMATCHAI_SEARCH_MAX_TRIALS_FIRST_LEVEL",
    "TRIALMATCHAI_SEARCH_SECOND_LEVEL_KEEP_DIVISOR",
)
TRIALMATCHAI_L4_PROCESS_EXECUTION_MODE = "shared_model_batch"
# The LoRA adapters load from a local path, so ``model.cot_adapter_revision`` and
# ``model.reranker_adapter_revision`` in the frozen config cannot bind: upstream reads them
# only into a cache-invalidation fingerprint (``orchestration.py``), never into a loading
# call. A declared revision that is checked by nothing is the same defect class as a cap
# that is asserted but never has an effect, so the weights are pinned by content here and
# verified before launch, the way the config file's digest already is.
#
# Values confirmed against the declared revisions on 2026-08-23: each file was downloaded
# from its pinned Hugging Face revision and compared byte for byte with what is staged.
#   majdabd33/trialmatchai-phi4-reasoning-lora  @ 9eaddaf048c8d4266a291884ff89db1cf05b07fc
#   majdabd33/trialmatchai-gemma2-reranker-lora @ 3118ba76d545f71f3aaa7952d2f031ad5fdef8c9
TRIALMATCHAI_L4_ADAPTER_WEIGHT_FILENAME = "adapter_model.safetensors"
TRIALMATCHAI_L4_ADAPTER_SHA256 = (
    (
        "cot_adapter_path",
        "sha256:f9665b6540cbab908cd531e343a8d1d330b1a9b0f8bf4a83c1a9b39250e50b15",
    ),
    (
        "reranker_adapter_path",
        "sha256:851aeb9f82dbe40a797b1bc085381ccb8aec6ab9d4c3d2c4ba1371356daa770a",
    ),
)

# Upstream's ``apply_env_overrides`` maps these onto the serving backends *after*
# TAIM has verified the config file's digest, so an inherited shell export could
# move a run off the backend its verified config declares while every digest
# still matched. Each entry is the environment variable, the config section it
# overrides, and the backend upstream falls back to when the section is absent.
# Upstream's ``apply_env_overrides`` lets 82 environment variables rewrite the
# config *after* TAIM has verified its SHA-256. Closing only the two serving
# backends left 63 of them able to change what a run computes while every digest
# still verified - including 20 that can swap a pinned model revision outright or
# flip ``trust_remote_code``. The whole surface is mirrored here because the
# adapter must not import the frozen checkout;
# ``test_config_env_override_map_matches_upstream`` parses upstream's own
# settings.py and fails if this map drifts, so a new upstream override cannot
# silently reopen the hole.
TRIALMATCHAI_CONFIG_ENV_OVERRIDES = {
    "TRIALMATCHAI_CONCEPT_DB_PATH": ("concept_linker", "db_path"),
    "TRIALMATCHAI_CONCEPT_LINKER_ENABLED": ("concept_linker", "enabled"),
    "TRIALMATCHAI_CONCEPT_RETRIEVAL_LIMIT": ("concept_linker", "retrieval_limit"),
    "TRIALMATCHAI_CONCEPT_SEARCH_LIMIT": ("concept_linker", "search_limit"),
    "TRIALMATCHAI_CONCEPT_TABLE": ("concept_linker", "table"),
    "TRIALMATCHAI_CONSTRAINTS_ENABLED": ("constraints", "enabled"),
    "TRIALMATCHAI_CONSTRAINTS_LLM_EXTRACTION_ENABLED": ("constraints", "llm_extraction_enabled"),
    "TRIALMATCHAI_CONSTRAINTS_SCORE_WEIGHT": ("constraints", "score_weight"),
    "TRIALMATCHAI_CONSTRAINTS_UNKNOWN_IS_NEUTRAL": ("constraints", "unknown_is_neutral"),
    "TRIALMATCHAI_CONSTRAINTS_WRITE_REPORTS": ("constraints", "write_reports"),
    "TRIALMATCHAI_EMBEDDER_BATCH_SIZE": ("embedder", "batch_size"),
    "TRIALMATCHAI_EMBEDDER_MODEL_NAME": ("embedder", "model_name"),
    "TRIALMATCHAI_EMBEDDER_REVISION": ("embedder", "revision"),
    "TRIALMATCHAI_EMBEDDER_TRUST_REMOTE_CODE": ("embedder", "trust_remote_code"),
    "TRIALMATCHAI_EMBEDDER_USE_FP16": ("embedder", "use_fp16"),
    "TRIALMATCHAI_EMBEDDER_USE_GPU": ("embedder", "use_gpu"),
    "TRIALMATCHAI_ENTITY_BACKEND": ("entity_extraction", "backend"),
    "TRIALMATCHAI_ENTITY_BATCH_SIZE": ("entity_extraction", "batch_size"),
    "TRIALMATCHAI_ENTITY_DEVICE": ("entity_extraction", "device"),
    "TRIALMATCHAI_ENTITY_MODEL_NAME": ("entity_extraction", "model_name"),
    "TRIALMATCHAI_ENTITY_MODEL_REVISION": ("entity_extraction", "model_revision"),
    "TRIALMATCHAI_ENTITY_SCHEMA_PATH": ("entity_extraction", "schema_path"),
    "TRIALMATCHAI_ENTITY_THRESHOLD": ("entity_extraction", "threshold"),
    "TRIALMATCHAI_ENTITY_TRUST_REMOTE_CODE": ("entity_extraction", "trust_remote_code"),
    "TRIALMATCHAI_FIRST_LEVEL_ENABLED": ("search", "first_level", "enabled"),
    "TRIALMATCHAI_FIRST_LEVEL_LLM_EXPANSION_ENABLED": (
        "search",
        "first_level",
        "llm_expansion_enabled",
    ),
    "TRIALMATCHAI_FIRST_LEVEL_LLM_MAX_TERMS": ("search", "first_level", "llm_max_terms"),
    "TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS": ("search", "first_level", "max_trials"),
    "TRIALMATCHAI_FIRST_LEVEL_PER_CHANNEL_SIZE": ("search", "first_level", "per_channel_size"),
    "TRIALMATCHAI_FIRST_LEVEL_RRF_K": ("search", "first_level", "rrf_k"),
    "TRIALMATCHAI_FIRST_LEVEL_VECTOR_SCORE_THRESHOLD": (
        "search",
        "first_level",
        "vector_score_threshold",
    ),
    "TRIALMATCHAI_FIRST_LEVEL_WRITE_REPORTS": ("search", "first_level", "write_reports"),
    "TRIALMATCHAI_LINK_ACCEPT": ("concept_linker", "accept_threshold"),
    "TRIALMATCHAI_LINK_REJECT": ("concept_linker", "reject_threshold"),
    "TRIALMATCHAI_MLX_MAX_NEW_TOKENS": ("mlx", "max_new_tokens"),
    "TRIALMATCHAI_MODEL_BASE_MODEL": ("model", "base_model"),
    "TRIALMATCHAI_MODEL_BASE_MODEL_REVISION": ("model", "base_model_revision"),
    "TRIALMATCHAI_MODEL_COT_ADAPTER_PATH": ("model", "cot_adapter_path"),
    "TRIALMATCHAI_MODEL_COT_ADAPTER_REVISION": ("model", "cot_adapter_revision"),
    "TRIALMATCHAI_MODEL_RERANKER_ADAPTER_PATH": ("model", "reranker_adapter_path"),
    "TRIALMATCHAI_MODEL_RERANKER_ADAPTER_REVISION": ("model", "reranker_adapter_revision"),
    "TRIALMATCHAI_MODEL_RERANKER_MODEL_PATH": ("model", "reranker_model_path"),
    "TRIALMATCHAI_MODEL_RERANKER_MODEL_REVISION": ("model", "reranker_model_revision"),
    "TRIALMATCHAI_MODEL_TRUST_REMOTE_CODE": ("model", "trust_remote_code"),
    "TRIALMATCHAI_OUTPUT_DIR": ("paths", "output_dir"),
    "TRIALMATCHAI_PATIENT_COPY_RAW": ("patient_inputs", "copy_raw"),
    "TRIALMATCHAI_PATIENT_INPUT_FORMAT": ("patient_inputs", "default_format"),
    "TRIALMATCHAI_PATIENT_PROFILE_DIR": ("patient_inputs", "profile_dir"),
    "TRIALMATCHAI_PATIENT_RAW_DIR": ("patient_inputs", "raw_dir"),
    "TRIALMATCHAI_PATIENT_STRICT_VALIDATION": ("patient_inputs", "strict_validation"),
    "TRIALMATCHAI_PATIENT_SUMMARY_DIR": ("patient_inputs", "summary_dir"),
    "TRIALMATCHAI_QUERY_EXPANSION_ADAPTER": ("query_expansion", "adapter"),
    "TRIALMATCHAI_QUERY_EXPANSION_BACKEND": ("query_expansion", "backend"),
    "TRIALMATCHAI_QUERY_EXPANSION_ENABLED": ("query_expansion", "enabled"),
    "TRIALMATCHAI_QUERY_EXPANSION_MODEL": ("query_expansion", "model"),
    "TRIALMATCHAI_RAG_BACKEND": ("rag", "backend"),
    "TRIALMATCHAI_RAG_MAX_TRIALS": ("rag", "max_trials_rag"),
    "TRIALMATCHAI_RAG_NO_THINK": ("rag", "no_think"),
    "TRIALMATCHAI_REGISTRY_API_BASE_URL": ("registry", "api_base_url"),
    "TRIALMATCHAI_REGISTRY_FAILURE_THRESHOLD": ("registry", "failure_threshold"),
    "TRIALMATCHAI_REGISTRY_KEYWORDS_FILE": ("registry", "keywords_file"),
    "TRIALMATCHAI_REGISTRY_MANIFEST_PATH": ("registry", "manifest_path"),
    "TRIALMATCHAI_REGISTRY_MAX_STUDIES": ("registry", "max_studies"),
    "TRIALMATCHAI_REGISTRY_RATE_LIMIT_PER_SECOND": ("registry", "rate_limit_per_second"),
    "TRIALMATCHAI_REGISTRY_RAW_DIR": ("registry", "raw_dir"),
    "TRIALMATCHAI_REGISTRY_REPORTS_DIR": ("registry", "reports_dir"),
    "TRIALMATCHAI_REGISTRY_REQUEST_TIMEOUT": ("registry", "request_timeout"),
    "TRIALMATCHAI_REGISTRY_SINCE_DAYS": ("registry", "since_days"),
    "TRIALMATCHAI_REGISTRY_SOURCE": ("registry", "source"),
    "TRIALMATCHAI_RERANKER_BACKEND": ("LLM_reranker", "backend"),
    "TRIALMATCHAI_SEARCH_BACKEND": ("search_backend", "backend"),
    "TRIALMATCHAI_SEARCH_CANDIDATE_LIMIT": ("search_backend", "candidate_limit"),
    "TRIALMATCHAI_SEARCH_CRITERIA_TABLE": ("search_backend", "criteria_table"),
    "TRIALMATCHAI_SEARCH_DB_PATH": ("search_backend", "db_path"),
    "TRIALMATCHAI_SEARCH_MAX_TRIALS_FIRST_LEVEL": ("search", "max_trials_first_level"),
    "TRIALMATCHAI_SEARCH_MAX_TRIALS_SECOND_LEVEL": ("search", "max_trials_second_level"),
    "TRIALMATCHAI_SEARCH_MODE": ("search", "mode"),
    "TRIALMATCHAI_SEARCH_SECOND_LEVEL_KEEP_DIVISOR": ("search", "second_level_keep_divisor"),
    "TRIALMATCHAI_SEARCH_TRIALS_TABLE": ("search_backend", "trials_table"),
    "TRIALMATCHAI_TRIALS_JSON_FOLDER": ("paths", "trials_json_folder"),
    "TRIALMATCHAI_VLLM_BATCH_SIZE": ("vllm", "batch_size"),
    "TRIALMATCHAI_VLLM_MAX_NEW_TOKENS": ("vllm", "max_new_tokens"),
}

# Variables TAIM itself owns, exempt from the config-agreement check for two
# distinct reasons.
#
# Plumbing says *where* inputs and outputs live, never what is computed, and TAIM
# sets it per run. The adapter-forced pair is written unconditionally by
# ``run()``, so an inherited shell value can never reach upstream.
#
# A third category used to exist: four retrieval caps the contract mandated to
# DISAGREE with the config, because the config carried
# ``search.max_trials_first_level: 1000`` while the contract pinned the
# environment to 15. That disagreement was the defect, not a licence for one. The
# caps are removed, so nothing is exempt on those grounds any more and the config
# is the sole authority on retrieval depth.
TRIALMATCHAI_ADAPTER_SET_ENV_OVERRIDES = frozenset(
    {
        "TRIALMATCHAI_CONCEPT_LINKER_ENABLED",
        "TRIALMATCHAI_ENTITY_BACKEND",
    }
)

TRIALMATCHAI_PLUMBING_ENV_OVERRIDES = frozenset(
    {
        "TRIALMATCHAI_CONCEPT_DB_PATH",
        "TRIALMATCHAI_OUTPUT_DIR",
        "TRIALMATCHAI_PATIENT_PROFILE_DIR",
        "TRIALMATCHAI_PATIENT_RAW_DIR",
        "TRIALMATCHAI_PATIENT_SUMMARY_DIR",
        "TRIALMATCHAI_REGISTRY_KEYWORDS_FILE",
        "TRIALMATCHAI_REGISTRY_MANIFEST_PATH",
        "TRIALMATCHAI_REGISTRY_RAW_DIR",
        "TRIALMATCHAI_REGISTRY_REPORTS_DIR",
        "TRIALMATCHAI_SEARCH_DB_PATH",
        "TRIALMATCHAI_TRIALS_JSON_FOLDER",
    }
)


TRIALMATCHAI_TAIM_OWNED_ENV_OVERRIDES = (
    TRIALMATCHAI_PLUMBING_ENV_OVERRIDES | TRIALMATCHAI_ADAPTER_SET_ENV_OVERRIDES
)

TRIALMATCHAI_SERVING_BACKEND_OVERRIDES = (
    ("TRIALMATCHAI_RAG_BACKEND", "rag", "vllm"),
    ("TRIALMATCHAI_RERANKER_BACKEND", "LLM_reranker", "vllm"),
)


class TrialMatchAIAdapterError(ValueError):
    """Raised when an upstream TrialMatchAI artifact is invalid."""


def _request_option(request: SystemRunRequest, name: str) -> object:
    try:
        return request.options[name]
    except KeyError as exc:
        raise TrialMatchAIAdapterError(
            f"{request.run_id} is missing System option {name!r}"
        ) from exc


def _path_option(request: SystemRunRequest, name: str) -> Path:
    value = _request_option(request, name)
    if not isinstance(value, (str, Path)):
        raise TrialMatchAIAdapterError(
            f"{request.run_id} System option {name!r} must be a path string"
        )
    return Path(value)


def _optional_path_option(request: SystemRunRequest, name: str) -> Path | None:
    value = request.options.get(name)
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TrialMatchAIAdapterError(
            f"{request.run_id} System option {name!r} must be a path string"
        )
    return Path(value).resolve()


def _config_override_option(request: SystemRunRequest) -> dict[str, str] | None:
    return _string_mapping_option(request, "trialmatchai_config_override")


def _backend_override_option(request: SystemRunRequest) -> dict[str, str] | None:
    return _string_mapping_option(request, "trialmatchai_backend_override")


def _string_mapping_option(request: SystemRunRequest, name: str) -> dict[str, str] | None:
    value = request.options.get(name)
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise TrialMatchAIAdapterError(
            f"{request.run_id} System option {name!r} must be a non-empty mapping of "
            "environment variable to value"
        )
    override: dict[str, str] = {}
    for name, backend in value.items():
        if not isinstance(name, str) or not isinstance(backend, str) or not backend:
            raise TrialMatchAIAdapterError(
                f"{request.run_id} System option {name!r} must map strings to non-empty strings"
            )
        override[name] = backend
    return override


def _validate_snapshot_corpus(request: SystemRunRequest, processed_trials_folder: Path) -> None:
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


def _timeout_option(request: SystemRunRequest, name: str) -> float:
    value = _request_option(request, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrialMatchAIAdapterError(f"{request.run_id} System option {name!r} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise TrialMatchAIAdapterError(
            f"{request.run_id} System option {name!r} must be positive and finite"
        )
    return float(value)


def _git_checkout_output(cwd: Path, *arguments: str) -> str:
    try:
        git = resolve_executable("git")
        completed = subprocess.run(  # noqa: S603 - resolved executable and fixed git argv
            [git, *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
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


def _config_value_at(payload: Mapping[str, object], path: tuple[str, ...]) -> object | None:
    node: object = payload
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def _comparable(value: object) -> str:
    """Render a config value and an environment string comparably.

    Upstream coerces environment strings into the config's type, so the honest
    comparison is on the rendered value: ``"true"`` matches ``True`` and ``"15"``
    matches ``15``, while ``"transformers"`` does not match ``"vllm"``.
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value).strip().casefold()


def _resolve_config_overrides(
    *,
    config_path: Path | None,
    environment: Mapping[str, str],
    declared_override: Mapping[str, str] | None,
    system_id: str,
) -> dict[str, object]:
    """Refuse any environment variable that rewrites the verified config.

    The digest proves which config file was used; it cannot prove the config
    survived to runtime. Upstream applies 82 environment overrides after that
    verification. Location variables that TAIM itself sets are exempt because
    they say *where* inputs live, not *what* is computed; everything else must
    either agree with the config or be declared as a deviation.
    """

    authorized = dict(declared_override or {})
    unknown = sorted(set(authorized) - set(TRIALMATCHAI_CONFIG_ENV_OVERRIDES))
    if unknown:
        raise TrialMatchAIAdapterError(
            f"{system_id} config override names variables upstream does not honour: "
            + ", ".join(unknown)
        )

    payload: Mapping[str, object] | None = None
    if config_path is not None:
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TrialMatchAIAdapterError(
                f"TrialMatchAI config is not readable JSON: {config_path} ({exc})"
            ) from exc
        if not isinstance(loaded, Mapping):
            raise TrialMatchAIAdapterError(f"TrialMatchAI config is not an object: {config_path}")
        payload = loaded

    present: dict[str, str] = {}
    deviations: dict[str, dict[str, str]] = {}
    for variable, path in sorted(TRIALMATCHAI_CONFIG_ENV_OVERRIDES.items()):
        if variable in TRIALMATCHAI_TAIM_OWNED_ENV_OVERRIDES:
            continue
        value = environment.get(variable)
        if value is None or value == "":
            if variable in authorized:
                raise TrialMatchAIAdapterError(
                    f"{system_id} declares a config override for {variable}, "
                    "but that variable is not set"
                )
            continue
        present[variable] = value
        if payload is None:
            raise TrialMatchAIAdapterError(
                f"{system_id} cannot verify {variable}={value!r}: the command names no "
                "--config, so there is no declared value to compare it against"
            )
        declared = _config_value_at(payload, path)
        if _comparable(declared) == _comparable(value):
            continue
        dotted = ".".join(path)
        if authorized.get(variable) != value:
            raise TrialMatchAIAdapterError(
                f"{system_id} config declares {dotted}={declared!r} but {variable}={value!r} "
                "overrides it after the config digest was verified. Unset it, or declare the "
                "deviation in the 'trialmatchai_config_override' System option."
            )
        deviations[variable] = {
            "config_path": dotted,
            "declared": _comparable(declared),
            "effective": value,
        }

    return {
        "verified_against_config": payload is not None,
        "environment_overrides": present,
        "deviations": deviations,
        "deviation": bool(deviations),
    }


def _declared_serving_backends(config_path: Path | None) -> dict[str, str] | None:
    """Read the serving backends the verified config declares.

    ``None`` means the command named no ``--config``, so there is no verified
    declaration to compare an override against.
    """

    if config_path is None:
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI config is not readable JSON: {config_path} ({exc})"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TrialMatchAIAdapterError(f"TrialMatchAI config is not an object: {config_path}")
    declared: dict[str, str] = {}
    for _, section, default in TRIALMATCHAI_SERVING_BACKEND_OVERRIDES:
        block = payload.get(section)
        value = block.get("backend", default) if isinstance(block, Mapping) else default
        declared[section] = str(value)
    return declared


def _resolve_serving_backends(
    *,
    config_path: Path | None,
    environment: Mapping[str, str],
    declared_override: Mapping[str, str] | None,
    system_id: str,
) -> dict[str, object]:
    """Refuse a serving backend that disagrees with the verified config.

    The digest check proves *which* config file was used; it cannot prove the
    config survived to runtime. This closes that gap at launch, before any GPU
    time is spent, and records the resolved backends so the manifest states what
    actually served rather than leaving it inferable from an absent key.

    A deliberate override stays possible: the caller declares it in the
    ``trialmatchai_backend_override`` System option, and the run is then recorded
    as a deviation instead of passing as the frozen configuration.
    """

    declared = _declared_serving_backends(config_path)
    present = {
        variable: environment[variable]
        for variable, _, _ in TRIALMATCHAI_SERVING_BACKEND_OVERRIDES
        if environment.get(variable)
    }
    authorized = dict(declared_override or {})
    known = {name for name, _, _ in TRIALMATCHAI_SERVING_BACKEND_OVERRIDES}
    unknown = sorted(set(authorized) - known)
    if unknown:
        raise TrialMatchAIAdapterError(
            f"{system_id} backend override names unknown variables: {', '.join(unknown)}"
        )

    effective = dict(declared) if declared is not None else {}
    for variable, section, _ in TRIALMATCHAI_SERVING_BACKEND_OVERRIDES:
        value = present.get(variable)
        if value is None:
            if variable in authorized:
                raise TrialMatchAIAdapterError(
                    f"{system_id} declares a backend override for {variable}, "
                    "but that variable is not set"
                )
            continue
        if declared is None:
            raise TrialMatchAIAdapterError(
                f"{system_id} cannot verify {variable}={value}: the command names no --config, "
                "so there is no declared backend to compare it against"
            )
        if value == declared[section]:
            continue
        if authorized.get(variable) != value:
            raise TrialMatchAIAdapterError(
                f"{system_id} config declares {section}.backend={declared[section]!r} but "
                f"{variable}={value!r} overrides it after the config digest was verified. "
                "Unset it, or declare the deviation in the "
                "'trialmatchai_backend_override' System option."
            )
        effective[section] = value

    return {
        "declared": declared,
        "effective": effective if declared is not None else None,
        "environment_overrides": present,
        "deviation": bool(declared is not None and effective != declared),
    }


def verify_trialmatchai_l4_adapter_weights(cwd: Path, config_path: Path) -> dict[str, str]:
    """Verify the LoRA weights by content, since their declared revisions cannot bind.

    Returns the observed digests so the run manifest records what was actually loaded.
    """

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI config is not readable JSON: {config_path}"
        ) from exc
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise TrialMatchAIAdapterError(f"TrialMatchAI config has no model section: {config_path}")

    observed: dict[str, str] = {}
    for option, expected in TRIALMATCHAI_L4_ADAPTER_SHA256:
        relative = model.get(option)
        if not isinstance(relative, str) or not relative:
            raise TrialMatchAIAdapterError(
                f"TrialMatchAI config is missing model.{option}: {config_path}"
            )
        weights = Path(relative)
        if not weights.is_absolute():
            weights = cwd / weights
        weights = weights / TRIALMATCHAI_L4_ADAPTER_WEIGHT_FILENAME
        if not weights.is_file():
            raise TrialMatchAIAdapterError(
                f"TrialMatchAI adapter weights do not exist for model.{option}: {weights}"
            )
        actual = sha256_file(weights)
        if actual != expected:
            raise TrialMatchAIAdapterError(
                f"TrialMatchAI adapter weights for model.{option} expected {expected}, "
                f"found {actual} at {weights}"
            )
        observed[option] = actual
    return observed


def _normalized_contract(contract: Mapping[str, object]) -> dict[str, object]:
    """Contract comparison that is indifferent to list-versus-tuple.

    The declared contract must be JSON-serialisable because it is written into the run
    manifest; the caller-supplied copy arrives frozen, with every sequence converted to a
    tuple. Normalising both sides lets the manifest stay valid without loosening the
    equality check itself.
    """

    normalized: dict[str, object] = {}
    for key, value in contract.items():
        if isinstance(value, Mapping):
            normalized[key] = dict(value)
        elif isinstance(value, (list, tuple)):
            normalized[key] = tuple(value)
        else:
            normalized[key] = value
    return normalized


def trialmatchai_l4_runtime_contract() -> dict[str, object]:
    """Return the exact execution settings bound to the frozen L4 System input."""

    return {
        "environment": dict(TRIALMATCHAI_L4_REQUIRED_ENVIRONMENT),
        "forbidden_environment": list(TRIALMATCHAI_L4_FORBIDDEN_ENVIRONMENT),
        "process_execution_mode": TRIALMATCHAI_L4_PROCESS_EXECUTION_MODE,
    }


def _trialmatchai_checkout_configuration(
    cwd: Path,
    command: Sequence[str],
    *,
    expected_commit: str,
    expected_repository: str,
    expected_config_sha256: str | None = None,
) -> dict[str, object]:
    """Verify and record the exact clean upstream source and explicit config."""

    actual_commit = _git_checkout_output(cwd, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI checkout expected commit {expected_commit}, found {actual_commit}"
        )
    actual_repository = _git_checkout_output(cwd, "remote", "get-url", "origin")
    if actual_repository != expected_repository:
        raise TrialMatchAIAdapterError(
            "TrialMatchAI checkout expected origin "
            f"{expected_repository}, found {actual_repository}"
        )
    if _git_checkout_output(cwd, "status", "--porcelain", "--untracked-files=all"):
        raise TrialMatchAIAdapterError(f"TrialMatchAI checkout has local changes: {cwd}")

    configuration: dict[str, object] = {
        "repository": actual_repository,
        "commit": actual_commit,
        "tree": _git_checkout_output(cwd, "rev-parse", "HEAD^{tree}"),
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


@dataclass(frozen=True, slots=True)
class TrialMatchAIResult:
    """Normalized final and diagnostic rankings from one TrialMatchAI topic."""

    candidates: tuple[Candidate, ...]
    diagnostic_rankings: dict[str, tuple[Candidate, ...]]
    raw_payload: Mapping[str, Any]


TRIALMATCHAI_SUMMARY_FILTER_VERSION = "exclude-observations-and-procedures-v1"


def filter_matching_summary_non_condition_facts(
    profile_payload: Mapping[str, Any], summary_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep observations and procedures out of TrialMatchAI condition channels.

    TrialMatchAI's canonical profile still retains every fact for narrative and eligibility
    reasoning. Its generated matching summary, however, currently places observations and
    procedures in ``other_conditions``. Procedures are already consumed separately by the
    upstream therapy channel, so routing them again as condition channels is redundant. Lab
    observations should remain narrative evidence, not condition queries.
    """

    excluded_labels = {
        str(fact.get("label", "")).strip().casefold()
        for category in ("observations", "procedures")
        for fact in (profile_payload.get(category) or ())
        if isinstance(fact, Mapping) and str(fact.get("label", "")).strip()
    }
    filtered = dict(summary_payload)
    other_conditions = summary_payload.get("other_conditions", ())
    if isinstance(other_conditions, Sequence) and not isinstance(other_conditions, (str, bytes)):
        filtered["other_conditions"] = [
            term for term in other_conditions if str(term).strip().casefold() not in excluded_labels
        ]
    return filtered


def _is_staged_trialmatchai_command(command: Sequence[str]) -> bool:
    tokens = tuple(command)
    try:
        e2e_index = tokens.index("e2e")
    except ValueError:
        return False
    return e2e_index > 0 and "--input" in tokens[e2e_index + 1 :]


def _supports_shared_model_batch(command: Sequence[str], *, topic_count: int) -> bool:
    """Whether one staged import/match pair can represent every requested topic."""
    return (
        topic_count > 1
        and _is_staged_trialmatchai_command(command)
        and not any("{topic_id}" in token for token in command)
    )


def _uses_nested_topic_artifacts(command: Sequence[str]) -> bool:
    """Whether a per-topic command declares an additional topic directory."""

    return _is_staged_trialmatchai_command(command) or any(
        "{topic_id}" in token for token in command
    )


def _format_trialmatchai_command(
    template: Sequence[str],
    *,
    input_path: Path,
    output_path: Path,
    topic_id: str,
    processed_trials_folder: Path | None,
    processed_criteria_folder: Path | None,
) -> tuple[str, ...]:
    return tuple(
        token.format(
            input=str(input_path),
            output=str(output_path),
            topic_id=topic_id,
            processed_trials=(
                str(processed_trials_folder)
                if processed_trials_folder is not None
                else "data/processed_trials"
            ),
            processed_criteria=(
                str(processed_criteria_folder)
                if processed_criteria_folder is not None
                else "data/processed_criteria"
            ),
        )
        for token in template
    )


def _require_manifest_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise TrialMatchAIAdapterError(f"concept store manifest {field} must be a SHA-256 digest")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise TrialMatchAIAdapterError(f"concept store manifest {field} must be a SHA-256 digest")
    return value


def _load_concept_store_manifest(db_path: Path) -> dict[str, Any]:
    manifest_path = db_path / TRIALMATCHAI_CONCEPT_STORE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise TrialMatchAIAdapterError(
            "TrialMatchAI concept store is missing its provenance manifest: "
            f"{manifest_path}. Build or register the store before running the canonical adapter."
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI concept store manifest is unreadable: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise TrialMatchAIAdapterError("TrialMatchAI concept store manifest must be a JSON object")
    if payload.get("manifest_version") != TRIALMATCHAI_CONCEPT_STORE_MANIFEST_VERSION:
        raise TrialMatchAIAdapterError(
            "unsupported TrialMatchAI concept store manifest version: "
            f"{payload.get('manifest_version')!r}"
        )

    store = payload.get("store")
    if not isinstance(store, Mapping) or store.get("table") != "concepts":
        raise TrialMatchAIAdapterError(
            "concept store manifest must describe the TrialMatchAI concepts table"
        )
    row_count = store.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise TrialMatchAIAdapterError("concept store manifest store.row_count must be positive")
    indexes = store.get("indexes")
    fts_index = indexes.get("fts") if isinstance(indexes, Mapping) else None
    if fts_index != {"column": "fts_text", "type": "FTS"}:
        raise TrialMatchAIAdapterError("concept store manifest must record its fts_text FTS index")

    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or not isinstance(sources.get("release"), str):
        raise TrialMatchAIAdapterError(
            "concept store manifest sources.release must identify the vocabulary release"
        )
    source_files = sources.get("files")
    if not isinstance(source_files, Mapping):
        raise TrialMatchAIAdapterError(
            "concept store manifest sources.files must contain source CSV hashes"
        )
    for field in ("CONCEPT.csv", "CONCEPT_SYNONYM.csv"):
        _require_manifest_sha256(source_files.get(field), f"sources.files.{field}")

    vocabularies = payload.get("vocabularies")
    if not isinstance(vocabularies, Mapping) or not vocabularies:
        raise TrialMatchAIAdapterError(
            "concept store manifest vocabularies must contain row counts"
        )
    for vocabulary, count in vocabularies.items():
        if not isinstance(vocabulary, str) or not vocabulary:
            raise TrialMatchAIAdapterError(
                "concept store manifest vocabulary names must be strings"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TrialMatchAIAdapterError(
                f"concept store manifest vocabulary count is invalid for {vocabulary!r}"
            )

    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, Mapping):
        raise TrialMatchAIAdapterError("concept store manifest must record embedding mode")
    mode = embeddings.get("mode")
    if mode not in {"vector", "fts-only"}:
        raise TrialMatchAIAdapterError(
            "concept store manifest embeddings.mode must be 'vector' or 'fts-only'"
        )
    skipped = embeddings.get("skip_embeddings")
    present = embeddings.get("present")
    if not isinstance(skipped, bool) or not isinstance(present, bool):
        raise TrialMatchAIAdapterError(
            "concept store manifest embeddings must record boolean skip_embeddings and present"
        )
    if (mode == "fts-only") != skipped or present == skipped:
        raise TrialMatchAIAdapterError(
            "concept store manifest embedding mode does not match skip_embeddings/present"
        )
    if mode == "vector":
        dimension = embeddings.get("dimension")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise TrialMatchAIAdapterError(
                "vector concept store manifest must record a positive embedding dimension"
            )
        vector_index = indexes.get("vector") if isinstance(indexes, Mapping) else None
        if not isinstance(vector_index, Mapping):
            raise TrialMatchAIAdapterError(
                "vector concept store manifest must record its ANN vector index"
            )
        if vector_index.get("column") != "embedding":
            raise TrialMatchAIAdapterError(
                "concept store ANN index must target the embedding column"
            )
        if vector_index.get("type") != "IVF_PQ":
            raise TrialMatchAIAdapterError("concept store ANN index must use IVF_PQ")
        if vector_index.get("metric") != embeddings.get("metric"):
            raise TrialMatchAIAdapterError(
                "concept store ANN index metric must match the embedding metric"
            )
        for field in ("num_partitions", "num_sub_vectors"):
            value = vector_index.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrialMatchAIAdapterError(f"concept store ANN index {field} must be positive")
        for field in ("num_bits", "max_iterations", "sample_rate"):
            value = vector_index.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrialMatchAIAdapterError(f"concept store ANN index {field} must be positive")
        if vector_index.get("train") is not True:
            raise TrialMatchAIAdapterError("concept store ANN index must record train=true")
        filters = indexes.get("filters") if isinstance(indexes, Mapping) else None
        expected_filters = {
            ("vocabulary_id", "BTREE"),
            ("domain_id", "BTREE"),
        }
        actual_filters = {
            (item.get("column"), item.get("type"))
            for item in filters or ()
            if isinstance(item, Mapping)
        }
        if not isinstance(filters, Sequence) or actual_filters != expected_filters:
            raise TrialMatchAIAdapterError(
                "vector concept store manifest must record vocabulary_id and domain_id "
                "BTree filters"
            )
        for field in ("model_name", "revision", "pooling"):
            if not isinstance(embeddings.get(field), str) or not embeddings[field]:
                raise TrialMatchAIAdapterError(
                    f"vector concept store embeddings.{field} must be non-empty"
                )
        for field in ("max_length", "batch_size"):
            value = embeddings.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrialMatchAIAdapterError(
                    f"vector concept store embeddings.{field} must be positive"
                )
        for field in ("use_fp16", "normalize"):
            if not isinstance(embeddings.get(field), bool):
                raise TrialMatchAIAdapterError(
                    f"vector concept store embeddings.{field} must be boolean"
                )

    concept_state = payload.get("concept_state")
    if not isinstance(concept_state, Mapping):
        raise TrialMatchAIAdapterError("concept store manifest must record concept state")
    _require_manifest_sha256(concept_state.get("fingerprint"), "concept_state.fingerprint")
    if mode == "vector":
        if concept_state.get("state_version") != "3":
            raise TrialMatchAIAdapterError(
                "vector concept store manifest must use concept state version 3"
            )
        _require_manifest_sha256(
            concept_state.get("vector_stream_digest"),
            "concept_state.vector_stream_digest",
        )

    return payload


def _concept_store_configuration(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str]
) -> dict[str, Any] | None:
    if not _is_staged_trialmatchai_command(command):
        return None
    raw_path = environment.get("TRIALMATCHAI_CONCEPT_DB_PATH")
    if not raw_path:
        raise TrialMatchAIAdapterError(
            "canonical TrialMatchAI runs require an explicit TRIALMATCHAI_CONCEPT_DB_PATH; "
            "the upstream default data/concepts path is not accepted without provenance"
        )
    db_path = Path(raw_path)
    if not db_path.is_absolute():
        db_path = cwd / db_path
    db_path = db_path.resolve()
    if not db_path.is_dir():
        raise TrialMatchAIAdapterError(f"TRIALMATCHAI_CONCEPT_DB_PATH does not exist: {db_path}")
    manifest_path = db_path / TRIALMATCHAI_CONCEPT_STORE_MANIFEST_FILENAME
    manifest = _load_concept_store_manifest(db_path)
    return {
        "db_path": str(db_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        **manifest,
    }


#: Sidecar written by the Snapshot corpus filter beside the ``processed_trials`` and
#: ``processed_criteria`` folders it derives from a Snapshot.
TRIALMATCHAI_CORPUS_FILTER_MANIFEST_FILENAME = "filter-manifest.json"


def _read_config_payload(config_path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI config is not readable JSON: {config_path} ({exc})"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TrialMatchAIAdapterError(f"TrialMatchAI config is not an object: {config_path}")
    return payload


def _config_section(payload: Mapping[str, object] | None, name: str) -> Mapping[str, object]:
    section = payload.get(name) if payload is not None else None
    return section if isinstance(section, Mapping) else {}


def _trialmatchai_model_identity(
    *,
    upstream_checkout: Mapping[str, object],
    config_payload: Mapping[str, object] | None,
    backends: Mapping[str, object],
) -> dict[str, Any]:
    """Identify the models a run loaded, from the digest-verified config.

    The release writer requires ``model_identity`` so a result traces back to what
    scored it. Every model TrialMatchAI loads is declared in the config the adapter
    verified by digest, so the identity is that declaration, role by role, plus the
    content digests of the LoRA weights whose declared revisions cannot bind (see
    ``verify_trialmatchai_l4_adapter_weights``). The roles mirror the qualification
    contract's model table. The public projection drops any ``*_path`` key as
    caller-local, so a declaration that upstream keeps under a ``_path`` key is
    recorded here under a role key that the projection keeps.
    """

    if config_payload is None:
        return {
            "kind": "trialmatchai_upstream_defaults",
            "reason": "the command names no --config; models are the pinned checkout's defaults",
            "upstream_repository": upstream_checkout["repository"],
            "upstream_commit": upstream_checkout["commit"],
            "config_sha256": None,
            "models": None,
            "serving_backends": backends["effective"],
            "serving_backend_deviation": backends["deviation"],
        }
    model = _config_section(config_payload, "model")
    embedder = _config_section(config_payload, "embedder")
    entity = _config_section(config_payload, "entity_extraction")
    expansion = _config_section(config_payload, "query_expansion")
    raw_weights = upstream_checkout.get("adapter_sha256")
    weights = dict(raw_weights) if isinstance(raw_weights, Mapping) else {}
    return {
        "kind": "trialmatchai_pinned_config",
        "upstream_repository": upstream_checkout["repository"],
        "upstream_commit": upstream_checkout["commit"],
        "config_sha256": upstream_checkout["config_sha256"],
        "models": {
            "eligibility_base": {
                "model": model.get("base_model"),
                "revision": model.get("base_model_revision"),
                "trust_remote_code": model.get("trust_remote_code"),
            },
            "eligibility_adapter": {
                "reference": model.get("cot_adapter_path"),
                "revision": model.get("cot_adapter_revision"),
                "weights_sha256": weights.get("cot_adapter_path"),
            },
            "reranker_base": {
                "model": model.get("reranker_model_path"),
                "revision": model.get("reranker_model_revision"),
            },
            "reranker_adapter": {
                "reference": model.get("reranker_adapter_path"),
                "revision": model.get("reranker_adapter_revision"),
                "weights_sha256": weights.get("reranker_adapter_path"),
            },
            "entity_extraction": {
                "backend": entity.get("backend"),
                "model": entity.get("model_name"),
                "revision": entity.get("model_revision"),
            },
            "embedder": {
                "model": embedder.get("model_name"),
                "revision": embedder.get("revision"),
                "use_fp16": embedder.get("use_fp16"),
            },
            "query_expansion": {
                "enabled": expansion.get("enabled"),
                "backend": expansion.get("backend"),
                "model": expansion.get("model"),
                "adapter": expansion.get("adapter"),
            },
        },
        "serving_backends": backends["effective"],
        "serving_backend_deviation": backends["deviation"],
    }


def _without_locators(value: object) -> object:
    """Drop every key that names a location, at any depth, as the public projection does."""

    if isinstance(value, Mapping):
        return {
            key: _without_locators(nested)
            for key, nested in value.items()
            if not (key in {"directory", "path"} or key.endswith(("_dir", "_path")))
        }
    if isinstance(value, (list, tuple)):
        return [_without_locators(nested) for nested in value]
    return value


def _corpus_filter_manifest_sha256(processed_trials_folder: Path | None) -> str | None:
    if processed_trials_folder is None:
        return None
    manifest = processed_trials_folder.parent / TRIALMATCHAI_CORPUS_FILTER_MANIFEST_FILENAME
    return sha256_file(manifest) if manifest.is_file() else None


def _trialmatchai_index_identity(
    request: SystemRunRequest,
    *,
    config_payload: Mapping[str, object] | None,
    processed_trials_folder: Path | None,
    processed_criteria_folder: Path | None,
    search_db_path: Path | None,
    concept_store: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Identify what first-level retrieval searched.

    Three stores answer a TrialMatchAI query: the prepared trial corpus and its
    criteria (validated here to be exactly the Snapshot's trial IDs), the search
    database upstream builds over that corpus, and the concept store the entity
    annotator reads at query time. All three are built offline and reused, so the
    index is persistent. The identity carries no locator: paths are recorded in the
    run configuration only. What it carries is the Task Input the corpus was
    validated against, the corpus filter manifest digest when the corpus was
    derived by the Snapshot corpus filter, the declared search backend, and the
    concept store's provenance manifest.
    """

    validated = processed_trials_folder is not None
    return {
        "kind": "trialmatchai_search_index",
        "persistent": True,
        "task_input_id": request.snapshot.task_input_id,
        "corpus": {
            "snapshot_trial_ids_validated": validated,
            "trial_count": len(request.snapshot.trials) if validated else None,
            "criteria_supplied": processed_criteria_folder is not None,
            "search_db_supplied": search_db_path is not None,
            "filter_manifest_sha256": _corpus_filter_manifest_sha256(processed_trials_folder),
        },
        "search_backend": (
            _without_locators(_config_section(config_payload, "search_backend"))
            if config_payload is not None
            else None
        ),
        "concept_store": _without_locators(concept_store) if concept_store is not None else None,
    }


def _trialmatchai_staged_commands(
    command: Sequence[str],
    *,
    input_paths: Sequence[Path],
    profile_dir: Path,
    summary_dir: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Split the default upstream e2e command around summary filtering.

    The upstream checkout is deliberately kept read-only. For the canonical adapter command,
    import first, apply the TAIM-owned summary contract, then let upstream run index/match from
    the staged profile. Custom non-e2e commands remain untouched.
    """

    tokens = tuple(command)
    if not _is_staged_trialmatchai_command(tokens):
        return None
    e2e_index = tokens.index("e2e")

    executable = tokens[:e2e_index]
    tail = tokens[e2e_index + 1 :]
    input_arguments = tuple(
        argument for input_path in input_paths for argument in ("--input", str(input_path))
    )
    import_command = (
        *executable,
        "import-patient",
        *input_arguments,
        "--format",
        "text",
        "--output-dir",
        str(profile_dir),
        "--summary-dir",
        str(summary_dir),
    )
    import_options: list[str] = []
    match_tail: list[str] = []
    index = 0
    while index < len(tail):
        arg = tail[index]
        if arg in {"--input", "--format"}:
            index += 2
            continue
        if arg == "--no-entities":
            import_options.append(arg)
            index += 1
            continue
        if arg == "--config":
            if index + 1 >= len(tail):
                raise TrialMatchAIAdapterError("TrialMatchAI --config requires a value")
            import_options.extend((arg, tail[index + 1]))
            match_tail.extend((arg, tail[index + 1]))
            index += 2
            continue
        if arg.startswith("--config="):
            if not arg.partition("=")[2]:
                raise TrialMatchAIAdapterError("TrialMatchAI --config requires a value")
            import_options.append(arg)
            match_tail.append(arg)
            index += 1
            continue
        match_tail.append(arg)
        index += 1
    match_command = (*executable, "e2e", *match_tail)
    return (*import_command, *import_options), tuple(match_command)


def _run_trialmatchai_with_summary_filter(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    environment: Mapping[str, str],
    input_paths: Sequence[Path],
    profile_dir: Path,
    summary_dir: Path,
) -> bool:
    staged = _trialmatchai_staged_commands(
        command,
        input_paths=input_paths,
        profile_dir=profile_dir,
        summary_dir=summary_dir,
    )
    if staged is None:
        run_trialmatchai_command(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            environment=environment,
        )
        return False

    import_command, match_command = staged
    run_trialmatchai_command(
        import_command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    expected_profile_ids = {input_path.stem for input_path in input_paths}
    actual_profile_ids = {path.stem for path in profile_dir.glob("*.json")}
    if actual_profile_ids != expected_profile_ids:
        raise TrialMatchAIAdapterError(
            "TrialMatchAI shared patient state does not exactly match the TAIM topics"
        )
    for input_path in input_paths:
        profile_path = profile_dir / f"{input_path.stem}.json"
        summary_path = summary_dir / profile_path.name
        if not profile_path.is_file() or not summary_path.is_file():
            raise TrialMatchAIAdapterError(
                "TrialMatchAI import did not produce the expected canonical profile and summary "
                f"for {input_path.stem}"
            )
        profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        filtered_summary = filter_matching_summary_non_condition_facts(
            profile_payload,
            summary_payload,
        )
        summary_path.write_text(
            json.dumps(filtered_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    run_trialmatchai_command(
        match_command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    return True


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


class _TrialMatchAISystem(StrictSystemOptions):
    """Run a pinned TrialMatchAI checkout through TAIM's System contract.

    The command is supplied by the harness as a tokenized template. It may use
    ``{input}``, ``{output}``, ``{topic_id}``, ``{processed_trials}``, and
    ``{processed_criteria}`` placeholders. TrialMatchAI is therefore an isolated
    subprocess, not an imported TAIM dependency.

    Constructed with :class:`TrialMatchAIHarnessInputs`, the System takes its locations and
    timeout from them and refuses them as options; the prepared corpus is then bound by the
    ``trialmatchai_prepared_corpus_sha256`` option. Constructed without, it takes them as options.
    """

    system_id = TRIALMATCHAI_L4_SYSTEM_ID
    expected_config_sha256: str | None = None
    option_names = frozenset(
        {
            "command",
            "cwd",
            "workspace",
            "timeout_seconds",
            "processed_trials_folder",
            "processed_criteria_folder",
            "search_db_path",
            "trialmatchai_expected_commit",
            "trialmatchai_expected_repository",
            "trialmatchai_backend_override",
            "trialmatchai_config_override",
        }
    )
    required_capabilities = TRIALMATCHAI_REQUIRED_CAPABILITIES
    optional_capabilities = TRIALMATCHAI_OPTIONAL_CAPABILITIES

    def __init__(self, harness_inputs: TrialMatchAIHarnessInputs | None = None) -> None:
        self.harness_inputs = harness_inputs
        if harness_inputs is not None:
            self.option_names = (type(self).option_names - TRIALMATCHAI_HARNESS_OPTIONS) | {
                TRIALMATCHAI_PREPARED_CORPUS_OPTION
            }

    def _runtime_contract(self) -> dict[str, object] | None:
        return None

    def _validate_runtime_contract(
        self,
        request: SystemRunRequest,
        command_template: Sequence[str],
    ) -> dict[str, object] | None:
        expected = self._runtime_contract()
        if expected is None:
            return None
        actual = _request_option(request, "trialmatchai_runtime_contract")
        # System options are frozen by ``freeze_system_input_options``, which turns lists into
        # tuples, while the contract itself must stay JSON-safe because it is recorded in the
        # run manifest. Compare a normalised form so neither requirement forces the other.
        if not isinstance(actual, Mapping) or _normalized_contract(actual) != _normalized_contract(
            expected
        ):
            raise TrialMatchAIAdapterError(
                f"{self.system_id} requires exact trialmatchai_runtime_contract {expected}"
            )
        required_environment = expected["environment"]
        if not isinstance(required_environment, Mapping):
            raise TrialMatchAIAdapterError(
                f"{self.system_id} has an invalid internal runtime environment contract"
            )
        for name, required_value in required_environment.items():
            actual_value = os.environ.get(str(name))
            if actual_value != required_value:
                raise TrialMatchAIAdapterError(
                    f"{self.system_id} requires {name}={required_value}, found {actual_value!r}"
                )
        forbidden_environment = expected["forbidden_environment"]
        if not isinstance(forbidden_environment, (list, tuple)):
            raise TrialMatchAIAdapterError(
                f"{self.system_id} has an invalid internal forbidden-environment contract"
            )
        # Absence is a contract clause, not an omission. These variables were the retrieval
        # caps this System was qualified without; an inherited export would re-narrow the
        # funnel while the config digest still verified, which is the exact defect their
        # removal closes.
        for name in forbidden_environment:
            present = os.environ.get(str(name))
            if present is not None:
                raise TrialMatchAIAdapterError(
                    f"{self.system_id} forbids {name}; found {present!r}. It caps retrieval "
                    "depth below the verified config and must be unset."
                )
        if not _supports_shared_model_batch(
            command_template,
            topic_count=len(request.snapshot.topics),
        ):
            raise TrialMatchAIAdapterError(
                f"{self.system_id} requires shared-model batch execution for multiple topics"
            )
        return expected

    def run(self, request: SystemRunRequest) -> SystemRunResult:
        from taim.system_contracts import SystemRunResult

        started = time.perf_counter()
        command_template = _request_option(request, "command")
        harness = self.harness_inputs
        if harness is None:
            cwd = _path_option(request, "cwd").resolve()
            workspace = _path_option(request, "workspace").resolve()
            timeout = _timeout_option(request, "timeout_seconds")
            processed_trials_folder = _optional_path_option(request, "processed_trials_folder")
            processed_criteria_folder = _optional_path_option(request, "processed_criteria_folder")
            search_db_path = _optional_path_option(request, "search_db_path")
        else:
            prepared_corpus = _request_option(request, TRIALMATCHAI_PREPARED_CORPUS_OPTION)
            if not isinstance(prepared_corpus, str) or not prepared_corpus.startswith("sha256:"):
                raise TrialMatchAIAdapterError(
                    f"{request.run_id} System option {TRIALMATCHAI_PREPARED_CORPUS_OPTION!r} "
                    "must be a SHA-256 digest"
                )
            cwd = harness.cwd.resolve()
            workspace = harness.workspace.resolve()
            timeout = float(harness.timeout_seconds)
            if not math.isfinite(timeout) or timeout <= 0:
                raise TrialMatchAIAdapterError("TrialMatchAI timeout must be positive and finite")
            processed_trials_folder = harness.processed_trials_folder.resolve()
            processed_criteria_folder = harness.processed_criteria_folder.resolve()
            search_db_path = harness.search_db_path.resolve()
        if not isinstance(command_template, (tuple, list)):
            raise TrialMatchAIAdapterError("System option 'command' must be a sequence")
        if any(not isinstance(token, str) or not token for token in command_template):
            raise TrialMatchAIAdapterError("System option 'command' contains an invalid token")
        runtime_contract = self._validate_runtime_contract(request, command_template)
        if not cwd.is_dir():
            raise TrialMatchAIAdapterError(f"TrialMatchAI cwd does not exist: {cwd}")
        expected_commit = _request_option(request, "trialmatchai_expected_commit")
        expected_repository = _request_option(request, "trialmatchai_expected_repository")
        if (
            not isinstance(expected_commit, str)
            or not expected_commit
            or not isinstance(expected_repository, str)
            or not expected_repository
        ):
            raise TrialMatchAIAdapterError(
                "TrialMatchAI expected commit and repository must be non-empty strings"
            )
        upstream_checkout = _trialmatchai_checkout_configuration(
            cwd,
            command_template,
            expected_commit=expected_commit,
            expected_repository=expected_repository,
            expected_config_sha256=self.expected_config_sha256,
        )
        # Only the L4 System pins adapter weights: it is the one whose config declares
        # adapter revisions that cannot bind, so the weights are verified by content here.
        if self.expected_config_sha256 == TRIALMATCHAI_L4_CONFIG_SHA256:
            config_path = upstream_checkout.get("config_path")
            if isinstance(config_path, str):
                upstream_checkout["adapter_sha256"] = verify_trialmatchai_l4_adapter_weights(
                    cwd, Path(config_path)
                )
        workspace.mkdir(parents=True, exist_ok=True)
        inputs = workspace / "inputs"
        outputs = workspace / "outputs"
        inputs.mkdir(exist_ok=True)
        outputs.mkdir(exist_ok=True)

        if processed_trials_folder is not None:
            _validate_snapshot_corpus(request, processed_trials_folder)
        if processed_criteria_folder is not None and not processed_criteria_folder.is_dir():
            raise TrialMatchAIAdapterError(
                "TrialMatchAI processed criteria directory does not exist: "
                f"{processed_criteria_folder}"
            )

        all_candidates: list[Candidate] = []
        retrieval_candidates: list[Candidate] = []
        raw_paths: list[Path] = []
        base_environment = dict(os.environ)
        # TAIM is commonly launched through its own uv environment. The
        # upstream checkout has a separate environment, so do not let uv
        # interpret TAIM's VIRTUAL_ENV while running TrialMatchAI.
        base_environment.pop("VIRTUAL_ENV", None)
        # Entity extraction is part of TAIM's canonical TrialMatchAI comparison. Set these
        # explicitly so an inherited shell override cannot silently reproduce the old ablation.
        base_environment["TRIALMATCHAI_ENTITY_BACKEND"] = "gliner2"
        base_environment["TRIALMATCHAI_CONCEPT_LINKER_ENABLED"] = "true"
        # The serving backends are config-owned. Refuse here, before any model
        # loads, rather than leaving a Transformers run to present itself as the
        # frozen vLLM configuration.
        # A deviation declared through either option counts for both checks, so a
        # caller declares a backend change once rather than twice.
        declared_overrides = {
            **(_config_override_option(request) or {}),
            **(_backend_override_option(request) or {}),
        }
        config_overrides = _resolve_config_overrides(
            config_path=(
                Path(str(upstream_checkout["config_path"]))
                if upstream_checkout.get("config_path") is not None
                else None
            ),
            environment=base_environment,
            declared_override=declared_overrides or None,
            system_id=self.system_id,
        )
        backends = _resolve_serving_backends(
            config_path=(
                Path(str(upstream_checkout["config_path"]))
                if upstream_checkout.get("config_path") is not None
                else None
            ),
            environment=base_environment,
            declared_override=_backend_override_option(request),
            system_id=self.system_id,
        )
        concept_store_configuration = _concept_store_configuration(
            command_template,
            cwd=cwd,
            environment=base_environment,
        )
        config_payload = (
            _read_config_payload(Path(str(upstream_checkout["config_path"])))
            if upstream_checkout.get("config_path") is not None
            else None
        )
        model_identity = _trialmatchai_model_identity(
            upstream_checkout=upstream_checkout,
            config_payload=config_payload,
            backends=backends,
        )
        index_identity = _trialmatchai_index_identity(
            request,
            config_payload=config_payload,
            processed_trials_folder=processed_trials_folder,
            processed_criteria_folder=processed_criteria_folder,
            search_db_path=search_db_path,
            concept_store=concept_store_configuration,
        )
        recorded_environment_source = dict(base_environment)
        if processed_trials_folder is not None:
            recorded_environment_source["TRIALMATCHAI_TRIALS_JSON_FOLDER"] = str(
                processed_trials_folder
            )
        if search_db_path is not None:
            recorded_environment_source["TRIALMATCHAI_SEARCH_DB_PATH"] = str(search_db_path)
        recorded_environment = {
            key: value
            for key, value in sorted(recorded_environment_source.items())
            if (
                (
                    key.startswith("TRIALMATCHAI_")
                    and key != "TRIALMATCHAI_OUTPUT_DIR"
                    and not key.startswith("TRIALMATCHAI_PATIENT_")
                )
                or key == "VLLM_WORKER_MULTIPROC_METHOD"
            )
            and "TOKEN" not in key
        }
        topics = request.snapshot.topics
        input_paths = tuple(write_patient_input(topic, inputs) for topic in topics)
        summary_filter_applied = False
        topic_output_roots: dict[str, Path] = {}
        shared_batch = _supports_shared_model_batch(command_template, topic_count=len(topics))
        if shared_batch:
            patient_state = workspace / "patient_state"
            command = _format_trialmatchai_command(
                command_template,
                input_path=input_paths[0],
                output_path=outputs,
                topic_id=topics[0].topic_id,
                processed_trials_folder=processed_trials_folder,
                processed_criteria_folder=processed_criteria_folder,
            )
            environment = dict(base_environment)
            environment["TRIALMATCHAI_OUTPUT_DIR"] = str(outputs)
            environment["TRIALMATCHAI_PATIENT_PROFILE_DIR"] = str(patient_state / "profiles")
            environment["TRIALMATCHAI_PATIENT_RAW_DIR"] = str(patient_state / "raw")
            environment["TRIALMATCHAI_PATIENT_SUMMARY_DIR"] = str(patient_state / "summaries")
            if processed_trials_folder is not None:
                environment["TRIALMATCHAI_TRIALS_JSON_FOLDER"] = str(processed_trials_folder)
            if search_db_path is not None:
                environment["TRIALMATCHAI_SEARCH_DB_PATH"] = str(search_db_path)
            batch_timeout = timeout * len(topics)
            summary_filter_applied = _run_trialmatchai_with_summary_filter(
                command,
                cwd=cwd,
                timeout_seconds=batch_timeout,
                environment=environment,
                input_paths=input_paths,
                profile_dir=patient_state / "profiles",
                summary_dir=patient_state / "summaries",
            )
            topic_output_roots = {topic.topic_id: outputs for topic in topics}
            process_execution = {
                "mode": "shared_model_batch",
                "ranked_trial_artifact_layout": "outputs/<topic_id>",
                "subprocesses": 2,
                "topics_per_match_process": len(topics),
                "timeout_seconds_per_topic": timeout,
                "effective_subprocess_timeout_seconds": batch_timeout,
            }
            patient_state_layout = "<root>/{profiles,raw,summaries}"
        else:
            for topic, input_path in zip(topics, input_paths, strict=True):
                output_path = outputs / topic.topic_id
                output_path.mkdir(exist_ok=True)
                command = _format_trialmatchai_command(
                    command_template,
                    input_path=input_path,
                    output_path=output_path,
                    topic_id=topic.topic_id,
                    processed_trials_folder=processed_trials_folder,
                    processed_criteria_folder=processed_criteria_folder,
                )
                environment = dict(base_environment)
                environment["TRIALMATCHAI_OUTPUT_DIR"] = str(output_path)
                # Custom commands and one-topic qualifications retain isolated state.
                patient_state = workspace / "patient_state" / topic.topic_id
                environment["TRIALMATCHAI_PATIENT_PROFILE_DIR"] = str(patient_state / "profiles")
                environment["TRIALMATCHAI_PATIENT_RAW_DIR"] = str(patient_state / "raw")
                environment["TRIALMATCHAI_PATIENT_SUMMARY_DIR"] = str(patient_state / "summaries")
                if processed_trials_folder is not None:
                    environment["TRIALMATCHAI_TRIALS_JSON_FOLDER"] = str(processed_trials_folder)
                if search_db_path is not None:
                    environment["TRIALMATCHAI_SEARCH_DB_PATH"] = str(search_db_path)
                summary_filter_applied = (
                    _run_trialmatchai_with_summary_filter(
                        command,
                        cwd=cwd,
                        timeout_seconds=timeout,
                        environment=environment,
                        input_paths=(input_path,),
                        profile_dir=patient_state / "profiles",
                        summary_dir=patient_state / "summaries",
                    )
                    or summary_filter_applied
                )
                topic_output_roots[topic.topic_id] = output_path
            staged_subprocesses = 2 if _is_staged_trialmatchai_command(command_template) else 1
            process_execution = {
                "mode": "per_topic",
                "ranked_trial_artifact_layout": (
                    "outputs/<topic_id>/<topic_id>"
                    if _uses_nested_topic_artifacts(command_template)
                    else "outputs/<topic_id>"
                ),
                "subprocesses": staged_subprocesses * len(topics),
                "topics_per_match_process": 1,
                "timeout_seconds_per_topic": timeout,
                "effective_subprocess_timeout_seconds": timeout,
            }
            patient_state_layout = "<root>/<topic_id>/{profiles,raw,summaries}"

        for topic in topics:
            output_root = topic_output_roots[topic.topic_id]
            ranked_path = (
                output_root / topic.topic_id / "ranked_trials.json"
                if _uses_nested_topic_artifacts(command_template)
                else output_root / "ranked_trials.json"
            )
            if not ranked_path.is_file():
                artifact_description = (
                    "topic-scoped ranked_trials.json" if shared_batch else "ranked_trials.json"
                )
                raise TrialMatchAIAdapterError(
                    "TrialMatchAI did not produce "
                    f"{artifact_description} for topic {topic.topic_id}"
                )
            raw_paths.append(ranked_path)
            first_level_path = ranked_path.parent / "first_level_scores.json"
            has_first_level_scores = first_level_path.is_file()
            if has_first_level_scores:
                raw_paths.append(first_level_path)
            normalized = load_and_normalize_ranked_trials(
                ranked_path,
                run_id=request.run_id,
                topic_id=topic.topic_id,
                system_id=self.system_id,
                first_level_scores_path=first_level_path if has_first_level_scores else None,
            )
            all_candidates.extend(normalized.candidates[: request.top_k])
            # Preserve the complete upstream retrieval diagnostic. Only the final Primary
            # Ranking is constrained by the evaluator's top_k.
            retrieval_candidates.extend(normalized.diagnostic_rankings.get("retrieval", ()))

        stage_rankings: list[StageRanking] = []
        if retrieval_candidates:
            diagnostic_path = workspace / "retrieval_candidates.json"
            diagnostic_path.write_text(
                json.dumps(
                    [candidate.to_dict() for candidate in retrieval_candidates],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            stage_rankings.append(
                StageRanking(
                    name="retrieval",
                    pipeline_depth="retrieval",
                    candidates=tuple(retrieval_candidates),
                    artifact_hash=sha256_file(diagnostic_path),
                )
            )
        return SystemRunResult(
            candidates=tuple(all_candidates),
            configuration={
                "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
                "upstream_system": self.system_id,
                "upstream_checkout": upstream_checkout,
                "task_input_id": request.snapshot.task_input_id,
                "renderer_version": PATIENT_RENDERER_VERSION,
                **({"runtime_contract": runtime_contract} if runtime_contract is not None else {}),
                "concept_store": concept_store_configuration,
                "model_identity": model_identity,
                "index_identity": index_identity,
                "matching_summary_filter": (
                    TRIALMATCHAI_SUMMARY_FILTER_VERSION if summary_filter_applied else None
                ),
                "topics": len(request.snapshot.topics),
                "process_execution": process_execution,
                "command_template": list(command_template),
                "cwd": str(cwd),
                "workspace": str(workspace),
                "snapshot_corpus": {
                    "processed_trials_folder": (
                        str(processed_trials_folder)
                        if processed_trials_folder is not None
                        else None
                    ),
                    "processed_criteria_folder": (
                        str(processed_criteria_folder)
                        if processed_criteria_folder is not None
                        else None
                    ),
                    "search_db_path": str(search_db_path) if search_db_path is not None else None,
                    "trial_id_validation": processed_trials_folder is not None,
                },
                "environment": recorded_environment,
                "backends": backends,
                "config_overrides": config_overrides,
                "patient_state": {
                    "root": str(workspace / "patient_state"),
                    "layout": patient_state_layout,
                },
                "raw_ranked_trial_artifacts": [
                    {"path": str(path), "sha256": sha256_file(path)} for path in raw_paths
                ],
            },
            runtime_seconds=time.perf_counter() - started,
            primary_ranking=PRIMARY_RANKING_CANDIDATES,
            pipeline_depth="post_eligibility",
            stage_rankings=tuple(stage_rankings),
        )


class TrialMatchAIL4System(_TrialMatchAISystem):
    """Frozen CUDA-L4 TREC-2021 development variant with an exact config."""

    system_id = TRIALMATCHAI_L4_SYSTEM_ID
    expected_config_sha256 = TRIALMATCHAI_L4_CONFIG_SHA256
    option_names = _TrialMatchAISystem.option_names | {"trialmatchai_runtime_contract"}

    def _runtime_contract(self) -> dict[str, object]:
        return trialmatchai_l4_runtime_contract()


class TrialMatchAIL4TREC2022System(TrialMatchAIL4System):
    """The same frozen L4 implementation under its TREC 2022 System identity."""

    system_id = TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID


class TrialMatchAIL4TREC2023System(TrialMatchAIL4System):
    """The same frozen L4 implementation under its later TREC 2023 System identity."""

    system_id = TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID


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


def _topic_text(topic: BenchmarkTopic) -> tuple[str, str]:
    """Read the final Snapshot contract's canonical patient text."""

    if not isinstance(topic, BenchmarkTopic):
        raise TrialMatchAIAdapterError("topic must be a BenchmarkTopic")
    return topic.topic_id, _non_empty_string(topic.canonical_text, "canonical patient text")


def render_patient_input(topic: BenchmarkTopic, *, include_typed_core: bool = True) -> str:
    """Return the exact deterministic text staged for TrialMatchAI.

    Canonical patient text is always included. When a Snapshot v2 topic exposes
    a typed patient core, its values are appended in stable field order. Source
    provenance is deliberately not rendered into the model prompt.
    """

    _topic_id, text = _topic_text(topic)
    sections = [text.rstrip()]
    typed_core = topic.typed_patient_core if include_typed_core else None
    fields = typed_core.fields if typed_core is not None else None
    if fields:
        structured_lines = ["Structured patient facts:"]
        for name in sorted(fields):
            assertion = fields[name]
            if not hasattr(assertion, "value"):
                raise TrialMatchAIAdapterError(
                    f"typed patient field {name!r} does not expose a value"
                )
            value = json.dumps(
                assertion.value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            structured_lines.append(f"- {name}: {value}")
        sections.append("\n".join(structured_lines))
    return "\n\n".join(sections) + "\n"


def write_patient_input(topic: BenchmarkTopic, directory: str | Path) -> Path:
    """Write one deterministic TrialMatchAI text input and return its path."""

    topic_id, _text = _topic_text(topic)
    rendered = render_patient_input(topic)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{topic_id}.txt"
    path.write_text(rendered.rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


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
    system_id: str = TRIALMATCHAI_L4_SYSTEM_ID,
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
    system_id: str = TRIALMATCHAI_L4_SYSTEM_ID,
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


def run_trialmatchai_command(
    command: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a pinned upstream command without importing or patching its checkout."""

    if not command or any(not isinstance(item, str) or not item for item in command):
        raise TrialMatchAIAdapterError("command must contain non-empty strings")
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise TrialMatchAIAdapterError("timeout_seconds must be positive and finite")
    try:
        # Trusted argv: `command` is validated above as a non-empty sequence of
        # non-empty strings and is supplied by TAIM adapter code describing the
        # pinned upstream checkout, not by benchmark or corpus data.  The list
        # form never reaches a shell.
        completed = subprocess.run(  # noqa: S603
            list(command),
            cwd=Path(cwd),
            env=dict(environment) if environment is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrialMatchAIAdapterError(f"TrialMatchAI command failed to run: {exc}") from exc
    if completed.returncode != 0:
        raise TrialMatchAIAdapterError(
            f"TrialMatchAI command returned {completed.returncode}: {completed.stderr[-2000:]}"
        )
    return completed


__all__ = [
    "PATIENT_RENDERER_VERSION",
    "TRIALMATCHAI_L4_ADAPTER_SHA256",
    "TRIALMATCHAI_L4_ADAPTER_WEIGHT_FILENAME",
    "TRIALMATCHAI_L4_COMMIT",
    "TRIALMATCHAI_L4_CONFIG_RELATIVE_PATH",
    "TRIALMATCHAI_L4_CONFIG_SHA256",
    "TRIALMATCHAI_L4_FORBIDDEN_ENVIRONMENT",
    "TRIALMATCHAI_L4_PROCESS_EXECUTION_MODE",
    "TRIALMATCHAI_L4_REPOSITORY",
    "TRIALMATCHAI_L4_REQUIRED_ENVIRONMENT",
    "TRIALMATCHAI_L4_SYSTEM_ID",
    "TRIALMATCHAI_L4_TREC_2022_SYSTEM_ID",
    "TRIALMATCHAI_L4_TREC_2023_SYSTEM_ID",
    "TrialMatchAIAdapterError",
    "TrialMatchAIHarnessInputs",
    "TrialMatchAIL4System",
    "TrialMatchAIL4TREC2022System",
    "TrialMatchAIL4TREC2023System",
    "TrialMatchAIResult",
    "load_and_normalize_ranked_trials",
    "normalize_ranked_trials",
    "render_patient_input",
    "run_trialmatchai_command",
    "trialmatchai_l4_runtime_contract",
    "verify_trialmatchai_l4_adapter_weights",
    "write_patient_input",
]
