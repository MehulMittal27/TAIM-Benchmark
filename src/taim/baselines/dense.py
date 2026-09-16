"""Pinned BGE-M3 dense retrieval with immutable, checksum-verified indexes.

NumPy and Sentence Transformers are optional dependencies.  Importing this
module does not require either package; they are loaded only when dense index
or model operations are requested.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

from taim.baselines.encoder_registry import (
    BGE_M3_POLICY,
    DENSE_ENCODER_REGISTRY,
    QWEN3_EMBEDDING_06B_POLICY,
    EncoderPolicy,
)
from taim.schemas import Candidate, JsonValue
from taim.snapshot import BenchmarkTopic, TrialDocument

BGE_M3_MODEL_ID = BGE_M3_POLICY.model_id
BGE_M3_MODEL_REVISION = BGE_M3_POLICY.model_revision
BGE_M3_TOKENIZER_REVISION = BGE_M3_POLICY.tokenizer_revision
BGE_M3_MAX_LENGTH = BGE_M3_POLICY.maximum_length
BGE_M3_EMBEDDING_DIMENSION = BGE_M3_POLICY.embedding_dimension
BGE_M3_ATTENTION_HEAD_COUNT = 16
BGE_M3_ATTENTION_SCORE_BUDGET_BYTES = 2 * 1024**3
BGE_M3_INFERENCE_VALUE_BYTES = 4
QWEN3_EMBEDDING_06B_ATTENTION_HEAD_COUNT = 16
QWEN3_EMBEDDING_06B_ATTENTION_SCORE_BUDGET_BYTES = 2 * 1024**3
QWEN3_EMBEDDING_06B_INFERENCE_VALUE_BYTES = 4

QWEN3_EMBEDDING_06B_MODEL_FILES = MappingProxyType(
    {
        "config.json": "sha256:b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
        "model.safetensors": (
            "sha256:0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
        ),
    }
)
QWEN3_EMBEDDING_06B_TOKENIZER_FILES = MappingProxyType(
    {
        "tokenizer.json": (
            "sha256:def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
        ),
        "tokenizer_config.json": (
            "sha256:253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"
        ),
    }
)

DENSE_INDEX_SCHEMA_VERSION = "1.0"
DENSE_INDEX_ARTIFACT_TYPE = "taim.dense-index"
# EncoderPolicy.normalization values.  "L2" and "encoder-L2" both mean every
# vector leaving the encoder has unit norm, so inner product equals cosine.
# "none" means raw magnitudes are kept and the score is the native dot
# product the policy was trained under.  The value is CONTROL data: it gates
# the shared index-build normalization, the shared query-path normalization,
# and the recorded configuration.  It is never merely documentation.
NORMALIZATION_NONE = "none"
# The recorded vocabulary is deliberately narrower than the policy vocabulary:
# a manifest field named "normalization" states what the STORED ROWS carry, and
# rows are either unit-norm or raw.  Where a policy applies the normalization is
# policy provenance, recorded losslessly under build_configuration's
# encoder_policy, and does not belong in the description of the bytes.
NORMALIZATION_L2 = "L2"
DENSE_MATRIX_FILENAME = "embeddings.npy"
DENSE_ID_MAP_FILENAME = "trial_ids.jsonl"
DENSE_INDEX_MANIFEST_FILENAME = "index-manifest.json"
DENSE_BUILD_MATRIX_FILENAME = ".taim-dense-build.embeddings.f32"
DENSE_BUILD_ID_MAP_FILENAME = ".taim-dense-build.trial_ids.jsonl"
DENSE_BUILD_PROGRESS_FILENAME = ".taim-dense-build.progress.json"
DEFAULT_DOCUMENT_BATCH_SIZE = 1
DEFAULT_CHECKPOINT_INTERVAL_ROWS = 100
DEFAULT_QUERY_BATCH_SIZE = 32
DEFAULT_RETRIEVAL_DEPTH = 1_000

_DENSE_BUILD_ARTIFACT_TYPE = "taim.dense-index-build"


_DENSE_BUILD_SCHEMA_VERSION = "1.0"
_DENSE_FINAL_MATRIX_STAGING_FILENAME = ".taim-dense-final.embeddings.npy"
_DENSE_FINAL_ID_MAP_STAGING_FILENAME = ".taim-dense-final.trial_ids.jsonl"
_DENSE_FINAL_MANIFEST_STAGING_FILENAME = ".taim-dense-final.index-manifest.json"


class DenseDependencyError(ValueError):
    """Raised when an optional dense-retrieval dependency is unavailable."""


class DenseIndexError(ValueError):
    """Raised when a dense index is malformed or does not match its inputs."""


class TextEncoder(Protocol):
    """Small injection boundary used by the production and mocked encoders."""

    policy: EncoderPolicy

    def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Format and encode trial-document text."""

    def encode_queries(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Format and encode patient-query text."""

    def configuration(self) -> Mapping[str, object]:
        """Return the complete JSON-compatible encoder configuration."""


def _require_numpy() -> Any:
    try:
        import numpy
    except ModuleNotFoundError as exc:
        raise DenseDependencyError(
            "dense retrieval requires the optional dense dependencies; "
            "install TAIM with its 'dense' extra"
        ) from exc
    return numpy


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _attention_budget_configuration(
    *,
    attention_head_count: int,
    attention_score_budget_bytes: int,
    inference_value_bytes: int,
) -> dict[str, JsonValue]:
    return {
        "requested_batch_size_is_upper_bound": True,
        "batch_size_one_short_circuit": (
            "skip budget pretokenization because effective batch size is unconditionally 1"
        ),
        "length_measurement": (
            "pinned tokenizer tokens after right truncation, including special tokens"
        ),
        "effective_batch_size": (
            "min(requested, max(1, floor(attention_score_budget_bytes / "
            "(attention_head_count * max_batch_tokens^2 * inference_value_bytes))))"
        ),
        "attention_head_count": attention_head_count,
        "attention_score_budget_bytes": attention_score_budget_bytes,
        "inference_value_bytes": inference_value_bytes,
    }


def _attention_budget_batch_size(
    requested_batch_size: int,
    *,
    maximum_tokens: int,
    attention_head_count: int,
    attention_score_budget_bytes: int,
    inference_value_bytes: int,
) -> int:
    bytes_per_batch_row = (
        attention_head_count * maximum_tokens * maximum_tokens * inference_value_bytes
    )
    budget_batch_size = max(1, attention_score_budget_bytes // bytes_per_batch_row)
    return min(requested_batch_size, budget_batch_size)


def _json_mapping(value: Mapping[str, object], field_name: str) -> dict[str, JsonValue]:
    try:
        serialized = json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True)
        normalized = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only finite JSON values") from exc
    if not isinstance(normalized, dict) or any(not isinstance(key, str) for key in normalized):
        raise ValueError(f"{field_name} must be a JSON object with string keys")
    return cast(dict[str, JsonValue], normalized)


def sha256_file(path: str | Path) -> str:
    """Return a prefixed SHA-256 digest for the exact bytes at ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def qwen_last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Pool the final real token from a left-padded Qwen batch."""

    hidden_shape = getattr(last_hidden_states, "shape", None)
    mask_shape = getattr(attention_mask, "shape", None)
    if (
        hidden_shape is None
        or mask_shape is None
        or len(hidden_shape) != 3
        or len(mask_shape) != 2
        or tuple(hidden_shape[:2]) != tuple(mask_shape)
    ):
        raise DenseIndexError("Qwen hidden states and attention mask have incompatible shape")
    if hidden_shape[0] < 1 or hidden_shape[1] < 1 or hidden_shape[2] < 1:
        raise DenseIndexError(
            "Qwen hidden states must have non-empty batch, token, and feature axes"
        )
    mask_values = attention_mask
    if callable(getattr(mask_values, "detach", None)):
        mask_values = mask_values.detach()
    if callable(getattr(mask_values, "cpu", None)):
        mask_values = mask_values.cpu()
    if callable(getattr(mask_values, "numpy", None)):
        mask_values = mask_values.numpy()
    numpy = _require_numpy()
    mask = numpy.asarray(mask_values)
    if (
        not bool(numpy.isin(mask, (0, 1)).all())
        or not bool((mask[:, -1] == 1).all())
        or bool((numpy.diff(mask, axis=1) < 0).any())
    ):
        raise DenseIndexError("Qwen attention mask does not use the required left padding")
    return last_hidden_states[:, -1, :]


class BgeM3Encoder:
    """Lazy Sentence Transformers adapter for a validated BGE-M3 policy."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        policy: EncoderPolicy = BGE_M3_POLICY,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        if not isinstance(policy, EncoderPolicy):
            raise TypeError("policy must be an EncoderPolicy")
        if policy != BGE_M3_POLICY:
            raise ValueError("BgeM3Encoder requires the registered immutable BGE-M3 policy")
        self.device = device
        self.policy = policy
        self._model: Any | None = None

    def configuration(self) -> dict[str, JsonValue]:
        """Return all model, tokenizer, formatting, and runtime inputs."""

        policy = self.policy
        return {
            "implementation": policy.implementation,
            "model_id": policy.model_id,
            "model_revision": policy.model_revision,
            "tokenizer_revision": policy.tokenizer_revision,
            "query_format": policy.query_format.description,
            "document_format": policy.document_format.description,
            "maximum_length": policy.maximum_length,
            "truncation_side": policy.truncation_side,
            "padding_side": policy.padding_side,
            "pooling": policy.pooling,
            "embedding_dimension": policy.embedding_dimension,
            "model_dtype": policy.model_dtype,
            "output_dtype": policy.output_dtype,
            "normalization": policy.normalization,
            "backend": policy.backend,
            "attention_implementation": policy.attention_implementation,
            "batching": _attention_budget_configuration(
                attention_head_count=BGE_M3_ATTENTION_HEAD_COUNT,
                attention_score_budget_bytes=BGE_M3_ATTENTION_SCORE_BUDGET_BYTES,
                inference_value_bytes=BGE_M3_INFERENCE_VALUE_BYTES,
            ),
            "device": self.device,
            "trust_remote_code": policy.trust_remote_code,
            "libraries": {
                "python": platform.python_version(),
                "numpy": _package_version("numpy"),
                "sentence_transformers": _package_version("sentence-transformers"),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "tokenizers": _package_version("tokenizers"),
                "huggingface_hub": _package_version("huggingface-hub"),
            },
        }

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise DenseDependencyError(
                "BGE-M3 encoding requires the optional dense dependencies; "
                "install TAIM with its 'dense' extra"
            ) from exc

        policy = self.policy
        model = SentenceTransformer(
            policy.model_id,
            device=self.device,
            prompts={"query": "", "document": ""},
            default_prompt_name=None,
            revision=policy.model_revision,
            trust_remote_code=policy.trust_remote_code,
            model_kwargs={
                "torch_dtype": getattr(torch, policy.model_dtype),
                "attn_implementation": policy.attention_implementation,
            },
        )
        model.max_seq_length = policy.maximum_length
        model.eval()

        dimension = model.get_sentence_embedding_dimension()
        if dimension != policy.embedding_dimension:
            raise DenseIndexError(
                f"{policy.model_id} produced dimension {dimension}; "
                f"expected {policy.embedding_dimension}"
            )

        transformer = model[0]
        model_config = getattr(getattr(transformer, "auto_model", None), "config", None)
        resolved_revision = getattr(model_config, "_commit_hash", None)
        if resolved_revision not in (None, policy.model_revision):
            raise DenseIndexError(
                f"resolved model revision {resolved_revision!r} does not match "
                f"{policy.model_revision!r}"
            )
        tokenizer = getattr(transformer, "tokenizer", None)
        if not callable(tokenizer):
            raise DenseIndexError("BGE-M3 tokenizer is unavailable")
        tokenizer.truncation_side = policy.truncation_side
        tokenizer.padding_side = policy.padding_side
        tokenizer_options = getattr(tokenizer, "init_kwargs", {})
        tokenizer_revision = (
            tokenizer_options.get("_commit_hash")
            if isinstance(tokenizer_options, Mapping)
            else None
        )
        if tokenizer_revision not in (None, policy.tokenizer_revision):
            raise DenseIndexError(
                f"resolved tokenizer revision {tokenizer_revision!r} does not match "
                f"{policy.tokenizer_revision!r}"
            )

        self._model = model
        return model

    def encode(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Encode raw texts without prompts; normalization is enforced by TAIM."""

        requested_batch_size = _positive_integer(batch_size, "batch_size")
        rows = list(texts)
        if not rows:
            return []
        model = self._load_model()
        effective_batch_size = requested_batch_size
        if requested_batch_size > 1:
            tokenizer = getattr(model[0], "tokenizer", None)
            if not callable(tokenizer):
                raise DenseIndexError("BGE-M3 tokenizer is unavailable")
            tokenized = tokenizer(
                rows,
                add_special_tokens=True,
                max_length=self.policy.maximum_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
            )
            input_ids = tokenized.get("input_ids") if isinstance(tokenized, Mapping) else None
            if not isinstance(input_ids, Sequence) or len(input_ids) != len(rows):
                raise DenseIndexError("BGE-M3 tokenizer returned invalid input IDs")
            token_lengths = [len(token_ids) for token_ids in input_ids]
            if any(length < 1 for length in token_lengths):
                raise DenseIndexError("BGE-M3 tokenizer returned an empty input")
            maximum_tokens = max(token_lengths)
            effective_batch_size = _attention_budget_batch_size(
                requested_batch_size,
                maximum_tokens=maximum_tokens,
                attention_head_count=BGE_M3_ATTENTION_HEAD_COUNT,
                attention_score_budget_bytes=BGE_M3_ATTENTION_SCORE_BUDGET_BYTES,
                inference_value_bytes=BGE_M3_INFERENCE_VALUE_BYTES,
            )
        return model.encode(
            rows,
            batch_size=effective_batch_size,
            show_progress_bar=False,
            precision="float32",
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=False,
        )

    def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Apply the declared document policy before BGE encoding."""

        return self.encode(
            [self.policy.document_format.render(text) for text in texts],
            batch_size=batch_size,
        )

    def encode_queries(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Apply the declared query policy before BGE encoding."""

        return self.encode(
            [self.policy.query_format.render(text) for text in texts],
            batch_size=batch_size,
        )


class Qwen3Embedding06BEncoder:
    """Lazy Transformers adapter for the pinned Qwen dense encoder."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        policy: EncoderPolicy = QWEN3_EMBEDDING_06B_POLICY,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        if not isinstance(policy, EncoderPolicy):
            raise TypeError("policy must be an EncoderPolicy")
        if policy != QWEN3_EMBEDDING_06B_POLICY:
            raise ValueError(
                "Qwen3Embedding06BEncoder requires the registered immutable "
                "Qwen3-Embedding-0.6B policy"
            )
        self.device = device
        self.policy = policy
        self._resolved_device: str | None = None
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def configuration(self) -> dict[str, JsonValue]:
        """Return manifest-bound Qwen model, tokenizer, and inference inputs."""

        policy = self.policy
        resolved_device = self._resolve_device()
        return {
            "implementation": policy.implementation,
            "model_id": policy.model_id,
            "model_revision": policy.model_revision,
            "tokenizer_id": policy.tokenizer_id,
            "tokenizer_revision": policy.tokenizer_revision,
            "resolved_files": {
                "model": dict(QWEN3_EMBEDDING_06B_MODEL_FILES),
                "tokenizer": dict(QWEN3_EMBEDDING_06B_TOKENIZER_FILES),
            },
            "resolved_file_verification": "SHA-256 before model use",
            "query_format": policy.query_format.to_dict(),
            "document_format": policy.document_format.to_dict(),
            "maximum_length": policy.maximum_length,
            "truncation": policy.truncation,
            "truncation_side": policy.truncation_side,
            "padding": policy.padding,
            "padding_side": policy.padding_side,
            "pooling": policy.pooling,
            "embedding_dimension": policy.embedding_dimension,
            "model_dtype": policy.model_dtype,
            "output_dtype": policy.output_dtype,
            "normalization": policy.normalization,
            "backend": policy.backend,
            "attention_implementation": policy.attention_implementation,
            "batching": _attention_budget_configuration(
                attention_head_count=QWEN3_EMBEDDING_06B_ATTENTION_HEAD_COUNT,
                attention_score_budget_bytes=(QWEN3_EMBEDDING_06B_ATTENTION_SCORE_BUDGET_BYTES),
                inference_value_bytes=QWEN3_EMBEDDING_06B_INFERENCE_VALUE_BYTES,
            ),
            "device": resolved_device,
            "requested_device": self.device,
            "resolved_device": resolved_device,
            "trust_remote_code": policy.trust_remote_code,
            "libraries": {
                "python": platform.python_version(),
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "tokenizers": _package_version("tokenizers"),
                "huggingface_hub": _package_version("huggingface-hub"),
            },
        }

    def _resolve_device(self, torch: Any | None = None) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        if self.device == "cpu":
            self._resolved_device = "cpu"
            return self._resolved_device
        if torch is None:
            try:
                import torch as torch_module
            except ModuleNotFoundError as exc:
                raise DenseDependencyError(
                    "resolving a non-CPU Qwen device requires the optional dense dependencies; "
                    "install TAIM with its 'dense' extra"
                ) from exc
            torch = torch_module
        try:
            requested = torch.device(self.device)
            if requested.type == "cuda" and requested.index is None:
                requested = torch.device("cuda", torch.cuda.current_device())
            elif requested.type == "mps" and requested.index is None:
                requested = torch.device("mps", 0)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise DenseIndexError(f"Qwen device {self.device!r} cannot be resolved") from exc
        self._resolved_device = str(requested)
        return self._resolved_device

    def _verify_resolved_files(self) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:
            raise DenseDependencyError(
                "Qwen encoding requires the optional dense dependencies; "
                "install TAIM with its 'dense' extra"
            ) from exc

        for repo_id, revision, files in (
            (
                self.policy.model_id,
                self.policy.model_revision,
                QWEN3_EMBEDDING_06B_MODEL_FILES,
            ),
            (
                self.policy.tokenizer_id,
                self.policy.tokenizer_revision,
                QWEN3_EMBEDDING_06B_TOKENIZER_FILES,
            ),
        ):
            for filename, expected_sha256 in files.items():
                resolved = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                )
                observed_sha256 = sha256_file(resolved)
                if observed_sha256 != expected_sha256:
                    raise DenseIndexError(
                        f"resolved Qwen file {filename!r} has SHA-256 {observed_sha256}; "
                        f"expected {expected_sha256}"
                    )

    def _load_model(self) -> tuple[Any, Any, Any]:
        if self._torch is not None and self._tokenizer is not None and self._model is not None:
            return self._torch, self._tokenizer, self._model
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise DenseDependencyError(
                "Qwen encoding requires the optional dense dependencies; "
                "install TAIM with its 'dense' extra"
            ) from exc

        self._verify_resolved_files()
        policy = self.policy
        tokenizer = AutoTokenizer.from_pretrained(
            policy.tokenizer_id,
            revision=policy.tokenizer_revision,
            padding_side=policy.padding_side,
            truncation_side=policy.truncation_side,
            trust_remote_code=policy.trust_remote_code,
        )
        model = AutoModel.from_pretrained(
            policy.model_id,
            revision=policy.model_revision,
            torch_dtype=getattr(torch, policy.model_dtype),
            attn_implementation=policy.attention_implementation,
            trust_remote_code=policy.trust_remote_code,
        )
        model.to(self._resolve_device(torch))
        model.eval()

        self._validate_loaded_runtime(torch=torch, tokenizer=tokenizer, model=model)

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        return torch, tokenizer, model

    def _validate_loaded_runtime(self, *, torch: Any, tokenizer: Any, model: Any) -> None:
        """Fail closed unless the loaded Qwen runtime matches its manifest policy."""

        policy = self.policy
        model_config = getattr(model, "config", None)
        resolved_model_revision = getattr(model_config, "_commit_hash", None)
        if resolved_model_revision != policy.model_revision:
            raise DenseIndexError(
                f"resolved Qwen model revision {resolved_model_revision!r} does not match "
                f"{policy.model_revision!r}"
            )
        if getattr(model_config, "hidden_size", None) != policy.embedding_dimension:
            raise DenseIndexError(
                "resolved Qwen hidden size does not match the registered embedding dimension"
            )
        if (
            getattr(model_config, "num_attention_heads", None)
            != QWEN3_EMBEDDING_06B_ATTENTION_HEAD_COUNT
        ):
            raise DenseIndexError(
                "resolved Qwen attention head count does not match the registered batching policy"
            )
        expected_dtype = getattr(torch, policy.model_dtype)
        resolved_dtype = getattr(model, "dtype", None)
        if resolved_dtype != expected_dtype:
            raise DenseIndexError(
                f"resolved Qwen model dtype {resolved_dtype!r} does not match {expected_dtype!r}"
            )
        try:
            expected_device = torch.device(self._resolve_device(torch))
            resolved_device = torch.device(getattr(model, "device", None))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise DenseIndexError("resolved Qwen model device is invalid") from exc
        if resolved_device != expected_device:
            raise DenseIndexError(
                f"resolved Qwen model device {str(resolved_device)!r} does not match "
                f"{str(expected_device)!r}"
            )
        if getattr(tokenizer, "padding_side", None) != policy.padding_side:
            raise DenseIndexError("resolved Qwen tokenizer does not use left padding")
        if getattr(tokenizer, "truncation_side", None) != policy.truncation_side:
            raise DenseIndexError("resolved Qwen tokenizer has the wrong truncation side")

    def encode(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Encode already formatted Qwen inputs under the pinned native policy."""

        requested_batch_size = _positive_integer(batch_size, "batch_size")
        rows = list(texts)
        if any(not isinstance(text, str) for text in rows):
            raise TypeError("texts must contain strings")
        numpy = _require_numpy()
        if not rows:
            return numpy.empty((0, self.policy.embedding_dimension), dtype=numpy.float32)
        torch, tokenizer, model = self._load_model()
        effective_batch_size = min(requested_batch_size, len(rows))
        if effective_batch_size > 1:
            tokenized = tokenizer(
                rows,
                add_special_tokens=True,
                max_length=self.policy.maximum_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
            )
            input_ids = tokenized.get("input_ids") if isinstance(tokenized, Mapping) else None
            if not isinstance(input_ids, Sequence) or len(input_ids) != len(rows):
                raise DenseIndexError("Qwen tokenizer returned invalid input IDs for batching")
            if any(
                not isinstance(token_ids, Sequence)
                or isinstance(token_ids, (str, bytes, bytearray))
                for token_ids in input_ids
            ):
                raise DenseIndexError("Qwen tokenizer returned invalid input IDs for batching")
            token_lengths = [len(token_ids) for token_ids in input_ids]
            if any(length < 1 for length in token_lengths):
                raise DenseIndexError("Qwen tokenizer returned an empty input for batching")
            maximum_tokens = max(token_lengths)
            effective_batch_size = _attention_budget_batch_size(
                effective_batch_size,
                maximum_tokens=maximum_tokens,
                attention_head_count=QWEN3_EMBEDDING_06B_ATTENTION_HEAD_COUNT,
                attention_score_budget_bytes=(QWEN3_EMBEDDING_06B_ATTENTION_SCORE_BUDGET_BYTES),
                inference_value_bytes=QWEN3_EMBEDDING_06B_INFERENCE_VALUE_BYTES,
            )
        chunks: list[Any] = []
        for start in range(0, len(rows), effective_batch_size):
            batch_rows = rows[start : start + effective_batch_size]
            encoded = tokenizer(
                batch_rows,
                max_length=self.policy.maximum_length,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )
            if not isinstance(encoded, Mapping):
                raise DenseIndexError("Qwen tokenizer returned an invalid batch")
            input_ids = encoded.get("input_ids")
            attention_mask = encoded.get("attention_mask")
            input_shape = getattr(input_ids, "shape", None)
            if (
                input_shape is None
                or len(input_shape) != 2
                or input_shape[0] != len(batch_rows)
                or input_shape[1] < 1
            ):
                raise DenseIndexError("Qwen tokenizer returned invalid input IDs")
            if input_shape[1] > self.policy.maximum_length:
                raise DenseIndexError("Qwen tokenizer exceeded the registered maximum length")
            if attention_mask is None:
                raise DenseIndexError("Qwen tokenizer did not return an attention mask")
            resolved_device = self._resolve_device(torch)
            device_batch = {
                key: value.to(resolved_device)
                for key, value in encoded.items()
                if callable(getattr(value, "to", None))
            }
            if set(device_batch) != set(encoded):
                raise DenseIndexError("Qwen tokenizer returned a non-tensor batch value")
            with torch.inference_mode():
                outputs = model(**device_batch)
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                raise DenseIndexError("Qwen model did not return last_hidden_state")
            pooled = qwen_last_token_pool(last_hidden_state, device_batch["attention_mask"])
            pooled = pooled.detach().to(dtype=torch.float32).cpu().numpy()
            matrix = _float32_matrix(
                pooled,
                expected_rows=len(batch_rows),
                field_name="Qwen embeddings",
            )
            if matrix.shape[1] != self.policy.embedding_dimension:
                raise DenseIndexError(
                    f"Qwen embedding dimension {matrix.shape[1]} does not match "
                    f"{self.policy.embedding_dimension}"
                )
            if _policy_normalizes_vectors(self.policy):
                chunks.append(_normalize_rows(matrix, field_name="Qwen embeddings"))
            else:
                chunks.append(matrix)
        return numpy.concatenate(chunks, axis=0).astype(numpy.float32, copy=False)

    def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Encode unprompted Canonical Trial Text."""

        return self.encode(
            [self.policy.document_format.render(text) for text in texts],
            batch_size=batch_size,
        )

    def encode_queries(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Encode Canonical Patient Text with the frozen patient-to-trial instruction."""

        return self.encode(
            [self.policy.query_format.render(text) for text in texts],
            batch_size=batch_size,
        )


_BI_ENCODER_ATTENTION_SCORE_BUDGET_BYTES = 2 * 1024**3
_BI_ENCODER_INFERENCE_VALUE_BYTES = 4

# Attention head counts are declared, not discovered, because configuration()
# must describe the batching policy before any weights are loaded.  Every value
# is re-checked against the loaded config in _validate_loaded_runtime().
_BI_ENCODER_ATTENTION_HEAD_COUNTS = MappingProxyType(
    {
        "medcpt": 12,
        "medcpt-dot": 12,
        "biolord-2023": 12,
        "e5-large-v2": 16,
    }
)

_BI_ENCODER_QUERY_ROLE = "query"
_BI_ENCODER_DOCUMENT_ROLE = "document"


def cls_pool(last_hidden_states: Any) -> Any:
    """Pool the first (CLS) position of a right-padded batch."""

    hidden_shape = getattr(last_hidden_states, "shape", None)
    if hidden_shape is None or len(hidden_shape) != 3 or min(hidden_shape) < 1:
        raise DenseIndexError("CLS pooling requires non-empty batch, token, and feature axes")
    return last_hidden_states[:, 0, :]


def masked_mean_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Average every real token of a batch, ignoring padding positions."""

    hidden_shape = getattr(last_hidden_states, "shape", None)
    mask_shape = getattr(attention_mask, "shape", None)
    if (
        hidden_shape is None
        or mask_shape is None
        or len(hidden_shape) != 3
        or len(mask_shape) != 2
        or tuple(hidden_shape[:2]) != tuple(mask_shape)
    ):
        raise DenseIndexError(
            "mean pooling hidden states and attention mask have incompatible shape"
        )
    if min(hidden_shape) < 1:
        raise DenseIndexError("mean pooling requires non-empty batch, token, and feature axes")
    expanded = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
    totals = (last_hidden_states * expanded).sum(dim=1)
    counts = expanded.sum(dim=1)
    if bool((counts < 1).any()):
        raise DenseIndexError("mean pooling found a row with no unmasked tokens")
    return totals / counts


@dataclass(frozen=True, slots=True)
class _BiEncoderTower:
    """One loaded (tokenizer, model) pair plus the length it is driven at."""

    tokenizer: Any
    model: Any
    maximum_length: int
    model_id: str
    model_revision: str


class TransformerBiEncoder:
    """Policy-driven Transformers adapter for symmetric and asymmetric bi-encoders.

    One class covers every registered encoder whose runtime is "load a
    Transformers encoder, pool, L2-normalize".  Everything that differs between
    them - checkpoint, revision, per-role maximum length, pooling, and the
    per-role text template that carries any mandatory prefix - is read from the
    immutable ``EncoderPolicy`` rather than hard-coded here.

    Asymmetric policies (``document_model_id`` set) load two towers and route
    ``encode_queries`` and ``encode_documents`` to their own tower.  This is the
    property that makes MedCPT correct: encoding both roles with one tower
    yields a well-formed run that is silently wrong.
    """

    def __init__(self, *, policy: EncoderPolicy, device: str = "cpu") -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        if not isinstance(policy, EncoderPolicy):
            raise TypeError("policy must be an EncoderPolicy")
        if policy.encoder_id not in _BI_ENCODER_ATTENTION_HEAD_COUNTS:
            raise ValueError(
                f"TransformerBiEncoder has no registered batching policy for "
                f"encoder {policy.encoder_id!r}"
            )
        if policy.pooling not in ("CLS", "mean"):
            raise ValueError(
                f"TransformerBiEncoder supports CLS and mean pooling; got {policy.pooling!r}"
            )
        self.device = device
        self.policy = policy
        self._resolved_device: str | None = None
        self._torch: Any | None = None
        self._towers: dict[str, _BiEncoderTower] = {}
        self._resolved_files: dict[str, JsonValue] | None = None

    # -- identity ---------------------------------------------------------

    def _tower_specification(self, role: str) -> tuple[str, str, str, str, int]:
        """Return (model_id, model_revision, tokenizer_id, tokenizer_revision, length)."""

        policy = self.policy
        if role == _BI_ENCODER_DOCUMENT_ROLE:
            return (
                policy.effective_document_model_id,
                policy.effective_document_model_revision,
                policy.effective_document_tokenizer_id,
                policy.effective_document_tokenizer_revision,
                policy.effective_document_maximum_length,
            )
        if role == _BI_ENCODER_QUERY_ROLE:
            return (
                policy.model_id,
                policy.model_revision,
                policy.tokenizer_id,
                policy.tokenizer_revision,
                policy.effective_query_maximum_length,
            )
        raise ValueError(f"unknown encoder role {role!r}")

    def _attention_head_count(self) -> int:
        return _BI_ENCODER_ATTENTION_HEAD_COUNTS[self.policy.encoder_id]

    def _resolve_files(self) -> dict[str, JsonValue]:
        """Resolve and checksum every weight and tokenizer file actually used."""

        if self._resolved_files is not None:
            return self._resolved_files
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:
            raise DenseDependencyError(
                "dense encoding requires the optional dense dependencies; "
                "install TAIM with its 'dense' extra"
            ) from exc

        resolved: dict[str, JsonValue] = {}
        roles = [_BI_ENCODER_QUERY_ROLE]
        if self.policy.is_asymmetric:
            roles.append(_BI_ENCODER_DOCUMENT_ROLE)
        for role in roles:
            model_id, model_revision, tokenizer_id, tokenizer_revision, _ = (
                self._tower_specification(role)
            )
            entries: dict[str, JsonValue] = {}
            for repo_id, revision, filenames in (
                (model_id, model_revision, ("config.json", "model.safetensors")),
                (tokenizer_id, tokenizer_revision, ("tokenizer.json", "tokenizer_config.json")),
            ):
                for filename in filenames:
                    path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        revision=revision,
                    )
                    entries[f"{repo_id}@{revision}/{filename}"] = sha256_file(path)
            resolved[role] = dict(sorted(entries.items()))
        self._resolved_files = resolved
        return resolved

    def configuration(self) -> dict[str, JsonValue]:
        """Return the manifest-bound model, tokenizer, and inference inputs."""

        policy = self.policy
        resolved_device = self._resolve_device()
        configuration: dict[str, JsonValue] = {
            "implementation": policy.implementation,
            "encoder_id": policy.encoder_id,
            "model_id": policy.model_id,
            "model_revision": policy.model_revision,
            "tokenizer_id": policy.tokenizer_id,
            "tokenizer_revision": policy.tokenizer_revision,
            "symmetry": "asymmetric" if policy.is_asymmetric else "symmetric",
            "query_model_id": policy.model_id,
            "query_model_revision": policy.model_revision,
            "document_model_id": policy.effective_document_model_id,
            "document_model_revision": policy.effective_document_model_revision,
            "document_tokenizer_id": policy.effective_document_tokenizer_id,
            "document_tokenizer_revision": policy.effective_document_tokenizer_revision,
            "resolved_files": self._resolve_files(),
            "resolved_file_verification": "SHA-256 recorded before model use",
            "query_format": policy.query_format.to_dict(),
            "document_format": policy.document_format.to_dict(),
            "maximum_length": policy.maximum_length,
            "query_maximum_length": policy.effective_query_maximum_length,
            "document_maximum_length": policy.effective_document_maximum_length,
            "truncation": policy.truncation,
            "truncation_side": policy.truncation_side,
            "truncation_disclosure": (
                "inputs longer than the role maximum length are truncated on the "
                "truncation side; truncation changes what is retrieved and is audited "
                "separately per run"
            ),
            "padding": policy.padding,
            "padding_side": policy.padding_side,
            "pooling": policy.pooling,
            "embedding_dimension": policy.embedding_dimension,
            "model_dtype": policy.model_dtype,
            "output_dtype": policy.output_dtype,
            "normalization": policy.normalization,
            "backend": policy.backend,
            "attention_implementation": policy.attention_implementation,
            "batching": _attention_budget_configuration(
                attention_head_count=self._attention_head_count(),
                attention_score_budget_bytes=_BI_ENCODER_ATTENTION_SCORE_BUDGET_BYTES,
                inference_value_bytes=_BI_ENCODER_INFERENCE_VALUE_BYTES,
            ),
            "reduced_precision_matmul": {
                "torch.backends.cuda.matmul.allow_tf32": False,
                "torch.backends.cudnn.allow_tf32": False,
                "policy": "disabled so the run is exactly float32 on every device",
            },
            "device": resolved_device,
            "requested_device": self.device,
            "resolved_device": resolved_device,
            "trust_remote_code": policy.trust_remote_code,
            "libraries": {
                "python": platform.python_version(),
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "tokenizers": _package_version("tokenizers"),
                "huggingface_hub": _package_version("huggingface-hub"),
            },
        }
        return configuration

    # -- runtime ----------------------------------------------------------

    def _resolve_device(self, torch: Any | None = None) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        if self.device == "cpu":
            self._resolved_device = "cpu"
            return self._resolved_device
        if torch is None:
            try:
                import torch as torch_module
            except ModuleNotFoundError as exc:
                raise DenseDependencyError(
                    "resolving a non-CPU device requires the optional dense dependencies; "
                    "install TAIM with its 'dense' extra"
                ) from exc
            torch = torch_module
        try:
            requested = torch.device(self.device)
            if requested.type == "cuda" and requested.index is None:
                requested = torch.device("cuda", torch.cuda.current_device())
            elif requested.type == "mps" and requested.index is None:
                requested = torch.device("mps", 0)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise DenseIndexError(f"device {self.device!r} cannot be resolved") from exc
        self._resolved_device = str(requested)
        return self._resolved_device

    def _load_tower(self, role: str) -> tuple[Any, _BiEncoderTower]:
        policy = self.policy
        key = role if policy.is_asymmetric else _BI_ENCODER_QUERY_ROLE
        cached = self._towers.get(key)
        if cached is not None and self._torch is not None:
            return self._torch, cached
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise DenseDependencyError(
                "dense encoding requires the optional dense dependencies; "
                "install TAIM with its 'dense' extra"
            ) from exc

        # Exact float32 everywhere: TF32 would silently change the numerics of a
        # run that the manifest declares as float32.
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False

        self._resolve_files()
        model_id, model_revision, tokenizer_id, tokenizer_revision, maximum_length = (
            self._tower_specification(key)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=tokenizer_revision,
            padding_side=policy.padding_side,
            truncation_side=policy.truncation_side,
            trust_remote_code=policy.trust_remote_code,
        )
        model = AutoModel.from_pretrained(
            model_id,
            revision=model_revision,
            torch_dtype=getattr(torch, policy.model_dtype),
            attn_implementation=policy.attention_implementation,
            trust_remote_code=policy.trust_remote_code,
        )
        model.to(self._resolve_device(torch))
        model.eval()
        tower = _BiEncoderTower(
            tokenizer=tokenizer,
            model=model,
            maximum_length=maximum_length,
            model_id=model_id,
            model_revision=model_revision,
        )
        self._validate_loaded_runtime(torch=torch, tower=tower)
        self._torch = torch
        self._towers[key] = tower
        return torch, tower

    def _validate_loaded_runtime(self, *, torch: Any, tower: _BiEncoderTower) -> None:
        """Fail closed unless the loaded runtime matches its manifest policy."""

        policy = self.policy
        model_config = getattr(tower.model, "config", None)
        resolved_revision = getattr(model_config, "_commit_hash", None)
        if resolved_revision != tower.model_revision:
            raise DenseIndexError(
                f"resolved {tower.model_id} revision {resolved_revision!r} does not match "
                f"{tower.model_revision!r}"
            )
        if getattr(model_config, "hidden_size", None) != policy.embedding_dimension:
            raise DenseIndexError(
                f"resolved {tower.model_id} hidden size does not match the registered "
                f"embedding dimension {policy.embedding_dimension}"
            )
        expected_heads = self._attention_head_count()
        if getattr(model_config, "num_attention_heads", None) != expected_heads:
            raise DenseIndexError(
                f"resolved {tower.model_id} attention head count does not match the "
                f"registered batching policy ({expected_heads})"
            )
        position_limit = getattr(model_config, "max_position_embeddings", None)
        if isinstance(position_limit, int) and tower.maximum_length > position_limit:
            raise DenseIndexError(
                f"role maximum length {tower.maximum_length} exceeds the "
                f"{tower.model_id} position limit {position_limit}"
            )
        expected_dtype = getattr(torch, policy.model_dtype)
        if getattr(tower.model, "dtype", None) != expected_dtype:
            raise DenseIndexError(
                f"resolved {tower.model_id} dtype does not match {expected_dtype!r}"
            )
        try:
            expected_device = torch.device(self._resolve_device(torch))
            resolved_device = torch.device(getattr(tower.model, "device", None))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise DenseIndexError(f"resolved {tower.model_id} device is invalid") from exc
        if resolved_device != expected_device:
            raise DenseIndexError(
                f"resolved {tower.model_id} device {str(resolved_device)!r} does not match "
                f"{str(expected_device)!r}"
            )
        if getattr(tower.tokenizer, "padding_side", None) != policy.padding_side:
            raise DenseIndexError(f"resolved {tower.model_id} tokenizer has the wrong padding side")
        if getattr(tower.tokenizer, "truncation_side", None) != policy.truncation_side:
            raise DenseIndexError(
                f"resolved {tower.model_id} tokenizer has the wrong truncation side"
            )

    # -- encoding ---------------------------------------------------------

    def _encode(self, texts: Sequence[str], *, batch_size: int, role: str) -> object:
        requested_batch_size = _positive_integer(batch_size, "batch_size")
        rows = list(texts)
        if any(not isinstance(text, str) for text in rows):
            raise TypeError("texts must contain strings")
        numpy = _require_numpy()
        if not rows:
            return numpy.empty((0, self.policy.embedding_dimension), dtype=numpy.float32)
        torch, tower = self._load_tower(role)
        maximum_length = tower.maximum_length
        tokenizer = tower.tokenizer
        model = tower.model

        effective_batch_size = min(requested_batch_size, len(rows))
        if effective_batch_size > 1:
            measured = tokenizer(
                rows,
                add_special_tokens=True,
                max_length=maximum_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
            )
            input_ids = measured.get("input_ids") if isinstance(measured, Mapping) else None
            if not isinstance(input_ids, Sequence) or len(input_ids) != len(rows):
                raise DenseIndexError("tokenizer returned invalid input IDs for batching")
            token_lengths = [len(token_ids) for token_ids in input_ids]
            if any(length < 1 for length in token_lengths):
                raise DenseIndexError("tokenizer returned an empty input for batching")
            effective_batch_size = _attention_budget_batch_size(
                effective_batch_size,
                maximum_tokens=max(token_lengths),
                attention_head_count=self._attention_head_count(),
                attention_score_budget_bytes=_BI_ENCODER_ATTENTION_SCORE_BUDGET_BYTES,
                inference_value_bytes=_BI_ENCODER_INFERENCE_VALUE_BYTES,
            )

        chunks: list[Any] = []
        resolved_device = self._resolve_device(torch)
        for start in range(0, len(rows), effective_batch_size):
            batch_rows = rows[start : start + effective_batch_size]
            encoded = tokenizer(
                batch_rows,
                max_length=maximum_length,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )
            if not isinstance(encoded, Mapping):
                raise DenseIndexError("tokenizer returned an invalid batch")
            input_shape = getattr(encoded.get("input_ids"), "shape", None)
            if (
                input_shape is None
                or len(input_shape) != 2
                or input_shape[0] != len(batch_rows)
                or input_shape[1] < 1
            ):
                raise DenseIndexError("tokenizer returned invalid input IDs")
            if input_shape[1] > maximum_length:
                raise DenseIndexError("tokenizer exceeded the registered maximum length")
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                raise DenseIndexError("tokenizer did not return an attention mask")
            device_batch = {
                key: value.to(resolved_device)
                for key, value in encoded.items()
                if callable(getattr(value, "to", None))
            }
            if set(device_batch) != set(encoded):
                raise DenseIndexError("tokenizer returned a non-tensor batch value")
            with torch.inference_mode():
                outputs = model(**device_batch)
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                raise DenseIndexError("model did not return last_hidden_state")
            if self.policy.pooling == "CLS":
                pooled = cls_pool(last_hidden_state)
            else:
                pooled = masked_mean_pool(last_hidden_state, device_batch["attention_mask"])
            pooled = pooled.detach().to(dtype=torch.float32).cpu().numpy()
            matrix = _float32_matrix(
                pooled,
                expected_rows=len(batch_rows),
                field_name=f"{self.policy.encoder_id} embeddings",
            )
            if matrix.shape[1] != self.policy.embedding_dimension:
                raise DenseIndexError(
                    f"{self.policy.encoder_id} embedding dimension {matrix.shape[1]} does not "
                    f"match {self.policy.embedding_dimension}"
                )
            if _policy_normalizes_vectors(self.policy):
                chunks.append(
                    _normalize_rows(matrix, field_name=f"{self.policy.encoder_id} embeddings")
                )
            else:
                chunks.append(matrix)
        return numpy.concatenate(chunks, axis=0).astype(numpy.float32, copy=False)

    def encode_documents(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Encode Canonical Trial Text through the document tower."""

        return self._encode(
            [self.policy.document_format.render(text) for text in texts],
            batch_size=batch_size,
            role=_BI_ENCODER_DOCUMENT_ROLE,
        )

    def encode_queries(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Encode Canonical Patient Text through the query tower."""

        return self._encode(
            [self.policy.query_format.render(text) for text in texts],
            batch_size=batch_size,
            role=_BI_ENCODER_QUERY_ROLE,
        )


def _encoder_configuration(encoder: TextEncoder) -> dict[str, JsonValue]:
    try:
        configuration = encoder.configuration()
    except AttributeError as exc:
        raise ValueError("encoder must provide configuration()") from exc
    if not isinstance(configuration, Mapping):
        raise ValueError("encoder configuration must be a mapping")
    return _json_mapping(configuration, "encoder configuration")


def _encoder_policy(encoder: TextEncoder) -> EncoderPolicy:
    policy = getattr(encoder, "policy", None)
    if not isinstance(policy, EncoderPolicy):
        raise ValueError("encoder must expose a validated EncoderPolicy as policy")
    return policy


def _float32_matrix(values: object, *, expected_rows: int, field_name: str) -> Any:
    numpy = _require_numpy()
    matrix = numpy.asarray(values, dtype=numpy.float32)
    if matrix.ndim != 2:
        raise DenseIndexError(f"{field_name} must be a two-dimensional matrix")
    if matrix.shape[0] != expected_rows:
        raise DenseIndexError(f"{field_name} has {matrix.shape[0]} rows; expected {expected_rows}")
    if matrix.shape[1] < 1:
        raise DenseIndexError(f"{field_name} must have at least one column")
    if not bool(numpy.isfinite(matrix).all()):
        raise DenseIndexError(f"{field_name} contains non-finite values")
    return matrix


def _normalize_rows(matrix: Any, *, field_name: str) -> Any:
    numpy = _require_numpy()
    norms = numpy.linalg.norm(matrix, axis=1, keepdims=True)
    if not bool(numpy.isfinite(norms).all()) or bool((norms == 0).any()):
        raise DenseIndexError(f"{field_name} contains a zero or non-finite vector")
    normalized = numpy.asarray(matrix / norms, dtype=numpy.dtype("<f4"))
    if not bool(numpy.isfinite(normalized).all()):
        raise DenseIndexError(f"{field_name} normalization produced non-finite values")
    return normalized


def _policy_normalizes_vectors(policy: EncoderPolicy) -> bool:
    """Report whether the policy's vectors are L2-normalized before scoring."""

    return policy.normalization != NORMALIZATION_NONE


def _recorded_normalization(policy: EncoderPolicy) -> str:
    """Describe the normalization the recorded vectors actually carry.

    Read from the policy, so a run can never record the opposite of the metric
    it used.  It reports the state of the VECTORS rather than the policy's own
    spelling: ``"encoder-L2"`` and ``"L2"`` both leave unit-norm rows, so both
    record ``"L2"``, and only an unnormalized policy records ``"none"``.  That
    is also what keeps indexes built before ``normalization`` became control
    loadable - :func:`_select_recorded_build_configuration` compares
    ``build_configuration`` exactly, so widening this field to the policy
    spelling would strand every ``encoder-L2`` index ever built.
    """

    return NORMALIZATION_L2 if _policy_normalizes_vectors(policy) else NORMALIZATION_NONE


def _normalized_query(vector: object, *, dimension: int) -> Any:
    numpy = _require_numpy()
    query = numpy.asarray(vector, dtype=numpy.float32)
    if query.ndim != 1 or query.shape[0] != dimension:
        raise DenseIndexError(f"query vector must have shape ({dimension},)")
    if not bool(numpy.isfinite(query).all()):
        raise DenseIndexError("query vector contains non-finite values")
    norm = numpy.linalg.norm(query)
    if not bool(numpy.isfinite(norm)) or float(norm) == 0.0:
        raise DenseIndexError("query vector must have a finite, non-zero norm")
    return numpy.asarray(query / norm, dtype=numpy.float32)


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".taim-dense-", suffix=suffix, dir=directory)
    os.close(descriptor)
    return Path(name)


def _write_id_map(path: Path, trial_ids: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row, trial_id in enumerate(trial_ids):
            payload = {"row": row, "trial_id": trial_id}
            stream.write(
                json.dumps(
                    payload,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            stream.write("\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write(f"{serialized}\n")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_and_sync(source: Path, destination: Path) -> None:
    os.replace(source, destination)
    _fsync_directory(destination.parent)


def _unlink_paths(paths: Sequence[Path]) -> None:
    changed = False
    for path in paths:
        if path.exists():
            path.unlink()
            changed = True
    if changed:
        _fsync_directory(paths[0].parent)


def _build_configuration(
    encoder: TextEncoder,
    *,
    document_batch_size: int,
    checkpoint_interval_rows: int,
    expected_dimension: int | None,
) -> dict[str, JsonValue]:
    policy = _encoder_policy(encoder)
    return {
        "encoder": _encoder_configuration(encoder),
        "encoder_policy": policy.to_dict(),
        "encoder_policy_sha256": policy.policy_hash(),
        "document_batch_size": document_batch_size,
        "document_batch_size_role": "requested upper bound passed to the encoder",
        "checkpoint_interval_rows": checkpoint_interval_rows,
        "checkpoint_policy": "commit after interval rows or final encoder batch",
        "encoding_window_rows": checkpoint_interval_rows,
        "encoding_window_policy": (
            "one encoder call per durable checkpoint interval; encoder restores input row order"
        ),
        "expected_dimension": expected_dimension,
        "input_field": "TrialDocument.canonical_text",
        "row_order": "trial_id ascending",
        "storage_dtype": "float32 little-endian",
        "normalization": _recorded_normalization(policy),
    }


def _select_recorded_build_configuration(
    recorded: object,
    expected: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    if not isinstance(recorded, dict):
        raise DenseIndexError("dense index build_configuration must be an object")
    normalized = _json_mapping(recorded, "recorded dense build configuration")
    expected_configuration = dict(expected)
    if normalized == expected_configuration:
        return expected_configuration
    raise DenseIndexError("dense index build configuration does not match")


def _validate_documents(documents: Sequence[TrialDocument]) -> tuple[TrialDocument, ...]:
    rows = tuple(documents)
    if not rows:
        raise ValueError("at least one trial document is required")
    if any(
        not isinstance(document, TrialDocument)
        or not isinstance(document.trial_id, str)
        or not isinstance(document.canonical_text, str)
        for document in rows
    ):
        raise TypeError("documents must contain TrialDocument records")
    trial_ids = [document.trial_id for document in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("trial_id values must be unique")
    return tuple(sorted(rows, key=lambda document: document.trial_id))


@dataclass(slots=True)
class _DenseBuildState:
    completed_rows: int
    dimension: int | None
    embeddings_digest: Any
    trial_ids_byte_size: int
    trial_ids_sha256: str


def _build_artifact_paths(destination: Path) -> tuple[Path, Path, Path]:
    return (
        destination / DENSE_BUILD_MATRIX_FILENAME,
        destination / DENSE_BUILD_ID_MAP_FILENAME,
        destination / DENSE_BUILD_PROGRESS_FILENAME,
    )


def _staging_artifact_paths(destination: Path) -> tuple[Path, Path, Path]:
    return (
        destination / _DENSE_FINAL_MATRIX_STAGING_FILENAME,
        destination / _DENSE_FINAL_ID_MAP_STAGING_FILENAME,
        destination / _DENSE_FINAL_MANIFEST_STAGING_FILENAME,
    )


def reset_dense_index(directory: str | Path) -> None:
    """Remove TAIM final, checkpoint, and staging artifacts from one index directory."""

    destination = Path(directory)
    final_paths = (
        destination / DENSE_MATRIX_FILENAME,
        destination / DENSE_ID_MAP_FILENAME,
        destination / DENSE_INDEX_MANIFEST_FILENAME,
    )
    _unlink_paths(
        (
            *final_paths,
            *_build_artifact_paths(destination),
            *_staging_artifact_paths(destination),
        )
    )


def _digest_prefix(path: Path, byte_size: int) -> Any:
    digest = hashlib.sha256()
    remaining = byte_size
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise DenseIndexError(
                    "partial dense embedding matrix is shorter than its committed "
                    f"{byte_size} bytes"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest


def _build_progress_payload(
    *,
    corpus_hash: str,
    document_count: int,
    configuration: Mapping[str, JsonValue],
    state: _DenseBuildState,
) -> dict[str, object]:
    embeddings_byte_size = (
        0 if state.dimension is None else state.completed_rows * state.dimension * 4
    )
    return {
        "artifact_type": _DENSE_BUILD_ARTIFACT_TYPE,
        "schema_version": _DENSE_BUILD_SCHEMA_VERSION,
        "corpus_hash": corpus_hash,
        "document_count": document_count,
        "dimension": state.dimension,
        "completed_rows": state.completed_rows,
        "build_configuration": configuration,
        "partial_artifacts": {
            "embeddings": {
                "filename": DENSE_BUILD_MATRIX_FILENAME,
                "byte_size": embeddings_byte_size,
                "sha256": f"sha256:{state.embeddings_digest.hexdigest()}",
                "completed_rows": state.completed_rows,
                "dtype": "float32 little-endian",
            },
            "trial_ids": {
                "filename": DENSE_BUILD_ID_MAP_FILENAME,
                "byte_size": state.trial_ids_byte_size,
                "sha256": state.trial_ids_sha256,
                "row_count": document_count,
            },
        },
    }


def _commit_build_progress(
    destination: Path,
    *,
    corpus_hash: str,
    document_count: int,
    configuration: Mapping[str, JsonValue],
    state: _DenseBuildState,
) -> None:
    progress_path = destination / DENSE_BUILD_PROGRESS_FILENAME
    temporary = _temporary_path(destination, ".progress.json")
    try:
        _write_json(
            temporary,
            _build_progress_payload(
                corpus_hash=corpus_hash,
                document_count=document_count,
                configuration=configuration,
                state=state,
            ),
        )
        _fsync_file(temporary)
        _replace_and_sync(temporary, progress_path)
    finally:
        temporary.unlink(missing_ok=True)


def _initialize_build_state(
    destination: Path,
    *,
    trial_ids: Sequence[str],
    corpus_hash: str,
    configuration: Mapping[str, JsonValue],
) -> _DenseBuildState:
    matrix_path, ids_path, progress_path = _build_artifact_paths(destination)
    matrix_temporary = _temporary_path(destination, ".partial.f32")
    ids_temporary = _temporary_path(destination, ".partial.jsonl")
    progress_temporary = _temporary_path(destination, ".progress.json")
    published_paths = (matrix_path, ids_path, progress_path)
    try:
        _write_id_map(ids_temporary, trial_ids)
        _fsync_file(matrix_temporary)
        _fsync_file(ids_temporary)
        state = _DenseBuildState(
            completed_rows=0,
            dimension=None,
            embeddings_digest=hashlib.sha256(),
            trial_ids_byte_size=ids_temporary.stat().st_size,
            trial_ids_sha256=sha256_file(ids_temporary),
        )
        _write_json(
            progress_temporary,
            _build_progress_payload(
                corpus_hash=corpus_hash,
                document_count=len(trial_ids),
                configuration=configuration,
                state=state,
            ),
        )
        _fsync_file(progress_temporary)

        _replace_and_sync(matrix_temporary, matrix_path)
        _replace_and_sync(ids_temporary, ids_path)
        _replace_and_sync(progress_temporary, progress_path)
        return state
    except BaseException:
        _unlink_paths((*published_paths, matrix_temporary, ids_temporary, progress_temporary))
        raise
    finally:
        for temporary in (matrix_temporary, ids_temporary, progress_temporary):
            temporary.unlink(missing_ok=True)


def _partial_artifact_entry(
    progress: Mapping[str, JsonValue], name: str
) -> Mapping[str, JsonValue]:
    artifacts = progress.get("partial_artifacts")
    if not isinstance(artifacts, dict):
        raise DenseIndexError("dense build progress partial_artifacts must be an object")
    entry = artifacts.get(name)
    if not isinstance(entry, dict):
        raise DenseIndexError(f"dense build progress is missing partial artifact {name!r}")
    return entry


def _read_build_progress(path: Path) -> dict[str, JsonValue]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DenseIndexError(f"invalid dense build progress: {exc}") from exc
    if not isinstance(payload, dict):
        raise DenseIndexError("dense build progress must be a JSON object")
    return _json_mapping(payload, "dense build progress")


def _load_build_state(
    destination: Path,
    *,
    trial_ids: Sequence[str],
    corpus_hash: str,
    configuration: Mapping[str, JsonValue],
) -> _DenseBuildState:
    matrix_path, ids_path, progress_path = _build_artifact_paths(destination)
    progress = _read_build_progress(progress_path)
    if progress.get("artifact_type") != _DENSE_BUILD_ARTIFACT_TYPE:
        raise DenseIndexError("unsupported dense build progress artifact type")
    if progress.get("schema_version") != _DENSE_BUILD_SCHEMA_VERSION:
        raise DenseIndexError("unsupported dense build progress schema version")
    if progress.get("corpus_hash") != corpus_hash:
        raise DenseIndexError("dense build progress corpus hash does not match")
    if progress.get("document_count") != len(trial_ids):
        raise DenseIndexError("dense build progress document count does not match")
    if progress.get("build_configuration") != configuration:
        raise DenseIndexError("dense build progress build configuration does not match")

    ids_entry = _partial_artifact_entry(progress, "trial_ids")
    _validate_artifact(
        ids_path,
        ids_entry,
        expected_filename=DENSE_BUILD_ID_MAP_FILENAME,
    )
    if ids_entry.get("row_count") != len(trial_ids):
        raise DenseIndexError("dense build progress trial ID row count does not match")
    if _read_id_map(ids_path) != tuple(trial_ids):
        raise DenseIndexError("dense build progress trial ID map does not match")

    completed_rows = progress.get("completed_rows")
    if (
        isinstance(completed_rows, bool)
        or not isinstance(completed_rows, int)
        or not 0 <= completed_rows <= len(trial_ids)
    ):
        raise DenseIndexError("dense build progress has an invalid completed_rows")
    dimension = progress.get("dimension")
    if dimension is not None and (
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1
    ):
        raise DenseIndexError("dense build progress has an invalid dimension")
    if (completed_rows == 0) != (dimension is None):
        raise DenseIndexError("dense build progress dimension does not match completed_rows")

    matrix_entry = _partial_artifact_entry(progress, "embeddings")
    if matrix_entry.get("filename") != DENSE_BUILD_MATRIX_FILENAME:
        raise DenseIndexError("unexpected partial dense embedding filename")
    expected_byte_size = 0 if dimension is None else completed_rows * dimension * 4
    if matrix_entry.get("byte_size") != expected_byte_size:
        raise DenseIndexError("partial dense embedding byte size does not match progress")
    if matrix_entry.get("completed_rows") != completed_rows:
        raise DenseIndexError("partial dense embedding row count does not match progress")
    if matrix_entry.get("dtype") != "float32 little-endian":
        raise DenseIndexError("partial dense embedding dtype does not match")
    recorded_hash = matrix_entry.get("sha256")
    if not isinstance(recorded_hash, str):
        raise DenseIndexError("partial dense embedding SHA-256 is invalid")
    if matrix_path.stat().st_size < expected_byte_size:
        raise DenseIndexError("partial dense embedding matrix is shorter than committed progress")
    digest = _digest_prefix(matrix_path, expected_byte_size)
    if f"sha256:{digest.hexdigest()}" != recorded_hash:
        raise DenseIndexError("SHA-256 mismatch for partial dense embeddings")
    if matrix_path.stat().st_size > expected_byte_size:
        with matrix_path.open("r+b") as stream:
            stream.truncate(expected_byte_size)
            stream.flush()
            os.fsync(stream.fileno())

    return _DenseBuildState(
        completed_rows=completed_rows,
        dimension=dimension,
        embeddings_digest=digest,
        trial_ids_byte_size=ids_path.stat().st_size,
        trial_ids_sha256=sha256_file(ids_path),
    )


def _recover_or_initialize_build_state(
    destination: Path,
    *,
    trial_ids: Sequence[str],
    corpus_hash: str,
    configuration: Mapping[str, JsonValue],
) -> _DenseBuildState:
    build_paths = _build_artifact_paths(destination)
    matrix_path, ids_path, progress_path = build_paths
    present = tuple(path.is_file() for path in build_paths)
    if all(present):
        return _load_build_state(
            destination,
            trial_ids=trial_ids,
            corpus_hash=corpus_hash,
            configuration=configuration,
        )
    if progress_path.is_file():
        missing = ", ".join(
            path.name for path, exists in zip(build_paths, present, strict=True) if not exists
        )
        raise DenseIndexError(f"dense build checkpoint is incomplete; missing {missing}")
    if matrix_path.exists() or ids_path.exists():
        _unlink_paths(build_paths)
    return _initialize_build_state(
        destination,
        trial_ids=trial_ids,
        corpus_hash=corpus_hash,
        configuration=configuration,
    )


@dataclass(frozen=True, slots=True)
class DenseIndex:
    """A memory-mapped normalized embedding matrix and its ordered ID map."""

    directory: Path
    trial_ids: tuple[str, ...]
    embeddings: Any
    manifest: dict[str, JsonValue]

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def corpus_hash(self) -> str:
        value = self.manifest["corpus_hash"]
        if not isinstance(value, str):
            raise DenseIndexError("index manifest corpus_hash must be a string")
        return value

    def rank(
        self,
        query_vector: object,
        *,
        top_k: int = DEFAULT_RETRIEVAL_DEPTH,
        normalize_query: bool,
    ) -> list[tuple[str, float]]:
        """Return exact inner-product results with deterministic tie-breaking.

        ``normalize_query`` is required because it changes what is ranked: True
        scores cosine against this index's normalized rows; False scores the
        raw dot product against whatever magnitudes the index stores.  There is
        deliberately no default.
        """

        return exact_inner_product_rank(
            self.trial_ids,
            self.embeddings,
            query_vector,
            top_k=top_k,
            normalize_query=normalize_query,
        )


def prepare_dense_index(
    documents: Sequence[TrialDocument],
    *,
    directory: str | Path,
    corpus_hash: str,
    encoder: TextEncoder | None = None,
    document_batch_size: int = DEFAULT_DOCUMENT_BATCH_SIZE,
    checkpoint_interval_rows: int = DEFAULT_CHECKPOINT_INTERVAL_ROWS,
    expected_dimension: int | None = None,
) -> DenseIndex:
    """Build, resume, or checksum-validate and reuse one immutable dense index."""

    if not isinstance(corpus_hash, str) or not corpus_hash.strip():
        raise ValueError("corpus_hash must be a non-empty string")
    batch_size = _positive_integer(document_batch_size, "document_batch_size")
    checkpoint_interval = _positive_integer(
        checkpoint_interval_rows,
        "checkpoint_interval_rows",
    )
    resolved_encoder = cast(
        TextEncoder,
        encoder
        if encoder is not None
        else DENSE_ENCODER_REGISTRY.create_encoder("dense-bge-m3", device="cpu"),
    )
    policy_dimension = _encoder_policy(resolved_encoder).embedding_dimension
    if expected_dimension is None:
        expected_dimension = policy_dimension
    if expected_dimension is not None:
        expected_dimension = _positive_integer(expected_dimension, "expected_dimension")
    if expected_dimension != policy_dimension:
        raise DenseIndexError(
            f"expected_dimension {expected_dimension} does not match encoder policy dimension "
            f"{policy_dimension}"
        )
    rows = _validate_documents(documents)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / DENSE_MATRIX_FILENAME
    ids_path = destination / DENSE_ID_MAP_FILENAME
    manifest_path = destination / DENSE_INDEX_MANIFEST_FILENAME
    artifact_paths = (matrix_path, ids_path, manifest_path)
    build_paths = _build_artifact_paths(destination)
    staging_paths = _staging_artifact_paths(destination)
    configuration = _build_configuration(
        resolved_encoder,
        document_batch_size=batch_size,
        checkpoint_interval_rows=checkpoint_interval,
        expected_dimension=expected_dimension,
    )
    trial_ids = tuple(document.trial_id for document in rows)

    existing = tuple(path.is_file() for path in artifact_paths)
    if all(existing):
        recorded_manifest = _read_manifest(manifest_path)
        selected_configuration = _select_recorded_build_configuration(
            recorded_manifest.get("build_configuration"),
            configuration,
        )
        index = load_dense_index(
            destination,
            expected_corpus_hash=corpus_hash,
            expected_build_configuration=selected_configuration,
            expected_trial_ids=trial_ids,
        )
        _unlink_paths((*build_paths, *staging_paths))
        return index

    _unlink_paths(staging_paths)
    progress_path = destination / DENSE_BUILD_PROGRESS_FILENAME
    if progress_path.is_file():
        recorded_progress = _read_build_progress(progress_path)
        configuration = _select_recorded_build_configuration(
            recorded_progress.get("build_configuration"),
            configuration,
        )
    build_state = _recover_or_initialize_build_state(
        destination,
        trial_ids=trial_ids,
        corpus_hash=corpus_hash,
        configuration=configuration,
    )
    if any(existing):
        # A final manifest is the commit marker.  Incomplete finals can only be
        # discarded after the independently checksummed checkpoint validates.
        _unlink_paths(artifact_paths)

    partial_matrix_path = destination / DENSE_BUILD_MATRIX_FILENAME
    dimension = build_state.dimension
    pending_chunks: list[bytes] = []
    pending_rows = 0
    try:
        for start in range(build_state.completed_rows, len(rows), checkpoint_interval):
            end = min(start + checkpoint_interval, len(rows))
            values = resolved_encoder.encode_documents(
                [document.canonical_text for document in rows[start:end]],
                batch_size=batch_size,
            )
            document_matrix = _float32_matrix(
                values,
                expected_rows=end - start,
                field_name="document embeddings",
            )
            if _policy_normalizes_vectors(_encoder_policy(resolved_encoder)):
                encoded = _normalize_rows(
                    document_matrix,
                    field_name="document embeddings",
                )
            else:
                encoded = document_matrix
            encoded_dimension = int(encoded.shape[1])
            if dimension is None:
                dimension = encoded_dimension
                if expected_dimension is not None and dimension != expected_dimension:
                    raise DenseIndexError(
                        f"document embeddings have dimension {dimension}; "
                        f"expected {expected_dimension}"
                    )
            elif encoded_dimension != dimension:
                raise DenseIndexError("encoder returned inconsistent embedding dimensions")

            encoded_bytes = encoded.tobytes(order="C")
            expected_bytes = (end - start) * dimension * 4
            if len(encoded_bytes) != expected_bytes:
                raise DenseIndexError("encoder returned an invalid dense embedding byte layout")
            pending_chunks.append(encoded_bytes)
            pending_rows += end - start
            if pending_rows >= checkpoint_interval or end == len(rows):
                with partial_matrix_path.open("ab") as stream:
                    for chunk in pending_chunks:
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())

                next_digest = build_state.embeddings_digest.copy()
                for chunk in pending_chunks:
                    next_digest.update(chunk)
                next_state = _DenseBuildState(
                    completed_rows=end,
                    dimension=dimension,
                    embeddings_digest=next_digest,
                    trial_ids_byte_size=build_state.trial_ids_byte_size,
                    trial_ids_sha256=build_state.trial_ids_sha256,
                )
                _commit_build_progress(
                    destination,
                    corpus_hash=corpus_hash,
                    document_count=len(rows),
                    configuration=configuration,
                    state=next_state,
                )
                build_state = next_state
                pending_chunks.clear()
                pending_rows = 0
    except BaseException:
        if build_state.completed_rows == 0:
            _unlink_paths((*build_paths, *staging_paths))
        raise

    dimension = build_state.dimension
    if build_state.completed_rows != len(rows) or dimension is None:
        raise DenseIndexError("dense build checkpoint did not complete every document")
    expected_partial_size = len(rows) * dimension * 4
    if partial_matrix_path.stat().st_size != expected_partial_size:
        raise DenseIndexError("completed dense embedding checkpoint has an invalid byte size")
    expected_partial_hash = f"sha256:{build_state.embeddings_digest.hexdigest()}"
    if sha256_file(partial_matrix_path) != expected_partial_hash:
        raise DenseIndexError("SHA-256 mismatch for completed dense embedding checkpoint")

    numpy = _require_numpy()
    matrix_temporary, ids_temporary, manifest_temporary = staging_paths
    try:
        matrix = numpy.lib.format.open_memmap(
            matrix_temporary,
            mode="w+",
            dtype=numpy.dtype("<f4"),
            shape=(len(rows), dimension),
        )
        partial_matrix = numpy.memmap(
            partial_matrix_path,
            mode="r",
            dtype=numpy.dtype("<f4"),
            shape=(len(rows), dimension),
        )
        rows_per_copy = max(1, (16 * 1024 * 1024) // (dimension * 4))
        for start in range(0, len(rows), rows_per_copy):
            end = min(start + rows_per_copy, len(rows))
            matrix[start:end] = partial_matrix[start:end]
        matrix.flush()
        del partial_matrix
        del matrix
        _fsync_file(matrix_temporary)

        _write_id_map(ids_temporary, trial_ids)
        _fsync_file(ids_temporary)
        matrix_hash = sha256_file(matrix_temporary)
        ids_hash = sha256_file(ids_temporary)
        manifest: dict[str, object] = {
            "artifact_type": DENSE_INDEX_ARTIFACT_TYPE,
            "schema_version": DENSE_INDEX_SCHEMA_VERSION,
            "corpus_hash": corpus_hash,
            "document_count": len(rows),
            "dimension": dimension,
            "index_type": "exact_inner_product",
            "build_configuration": configuration,
            "artifacts": {
                "embeddings": {
                    "filename": DENSE_MATRIX_FILENAME,
                    "byte_size": matrix_temporary.stat().st_size,
                    "sha256": matrix_hash,
                    "shape": [len(rows), dimension],
                    "dtype": "float32 little-endian",
                },
                "trial_ids": {
                    "filename": DENSE_ID_MAP_FILENAME,
                    "byte_size": ids_temporary.stat().st_size,
                    "sha256": ids_hash,
                    "row_count": len(rows),
                },
            },
        }
        _write_json(manifest_temporary, manifest)
        _fsync_file(manifest_temporary)
        _replace_and_sync(matrix_temporary, matrix_path)
        _replace_and_sync(ids_temporary, ids_path)
        # The manifest is the final commit marker and is published last.
        _replace_and_sync(manifest_temporary, manifest_path)
    finally:
        for temporary in (matrix_temporary, ids_temporary, manifest_temporary):
            temporary.unlink(missing_ok=True)

    try:
        index = load_dense_index(
            destination,
            expected_corpus_hash=corpus_hash,
            expected_build_configuration=configuration,
            expected_trial_ids=trial_ids,
        )
    except BaseException:
        # These finals were produced by this call; retain the validated build
        # checkpoint so a subsequent call can finalize again without encoding.
        _unlink_paths(artifact_paths)
        raise
    _unlink_paths((*build_paths, *staging_paths))
    return index


def _read_manifest(path: Path) -> dict[str, JsonValue]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DenseIndexError(f"invalid dense index manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise DenseIndexError("dense index manifest must be a JSON object")
    return _json_mapping(payload, "dense index manifest")


def _read_id_map(path: Path) -> tuple[str, ...]:
    trial_ids: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise DenseIndexError(f"blank trial ID map record at line {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DenseIndexError(
                    f"invalid trial ID map record at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict) or set(payload) != {"row", "trial_id"}:
                raise DenseIndexError(f"invalid trial ID map record at line {line_number}")
            row = payload["row"]
            trial_id = payload["trial_id"]
            if isinstance(row, bool) or not isinstance(row, int) or row != line_number - 1:
                raise DenseIndexError(f"invalid trial ID map row at line {line_number}")
            if not isinstance(trial_id, str) or not trial_id.strip():
                raise DenseIndexError(f"invalid trial_id in ID map at line {line_number}")
            trial_ids.append(trial_id)
    if len(trial_ids) != len(set(trial_ids)):
        raise DenseIndexError("dense trial ID map contains duplicate IDs")
    return tuple(trial_ids)


def _artifact_entry(manifest: Mapping[str, JsonValue], name: str) -> Mapping[str, JsonValue]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DenseIndexError("dense index manifest artifacts must be an object")
    entry = artifacts.get(name)
    if not isinstance(entry, dict):
        raise DenseIndexError(f"dense index manifest is missing artifact {name!r}")
    return entry


def _validate_encoder_policy_binding(build_configuration: Mapping[str, JsonValue]) -> None:
    policy = build_configuration.get("encoder_policy")
    recorded_hash = build_configuration.get("encoder_policy_sha256")
    if policy is None and recorded_hash is None:
        return
    if not isinstance(policy, dict) or not isinstance(recorded_hash, str):
        raise DenseIndexError("dense index encoder policy binding is incomplete")
    normalized = _json_mapping(policy, "dense index encoder policy")
    canonical = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    observed_hash = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    if recorded_hash != observed_hash:
        raise DenseIndexError("dense index encoder policy SHA-256 does not match")


def _validate_artifact(
    path: Path,
    entry: Mapping[str, JsonValue],
    *,
    expected_filename: str,
) -> None:
    if entry.get("filename") != expected_filename:
        raise DenseIndexError(f"unexpected dense artifact filename for {expected_filename}")
    expected_size = entry.get("byte_size")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise DenseIndexError(f"invalid byte size for {expected_filename}")
    if path.stat().st_size != expected_size:
        raise DenseIndexError(f"byte size mismatch for {expected_filename}")
    expected_hash = entry.get("sha256")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise DenseIndexError(f"SHA-256 mismatch for {expected_filename}")


def load_dense_index(
    directory: str | Path,
    *,
    expected_corpus_hash: str | None = None,
    expected_build_configuration: Mapping[str, object] | None = None,
    expected_trial_ids: Sequence[str] | None = None,
) -> DenseIndex:
    """Load a dense index only after validating all recorded hashes and metadata."""

    source = Path(directory)
    matrix_path = source / DENSE_MATRIX_FILENAME
    ids_path = source / DENSE_ID_MAP_FILENAME
    manifest_path = source / DENSE_INDEX_MANIFEST_FILENAME
    for path in (matrix_path, ids_path, manifest_path):
        if not path.is_file():
            raise DenseIndexError(f"dense index artifact is missing: {path.name}")

    manifest = _read_manifest(manifest_path)
    if manifest.get("artifact_type") != DENSE_INDEX_ARTIFACT_TYPE:
        raise DenseIndexError("unsupported dense index artifact type")
    if manifest.get("schema_version") != DENSE_INDEX_SCHEMA_VERSION:
        raise DenseIndexError("unsupported dense index schema version")
    corpus_hash = manifest.get("corpus_hash")
    if not isinstance(corpus_hash, str) or not corpus_hash:
        raise DenseIndexError("dense index manifest has an invalid corpus_hash")
    if expected_corpus_hash is not None and corpus_hash != expected_corpus_hash:
        raise DenseIndexError(
            f"dense index corpus hash {corpus_hash!r} does not match {expected_corpus_hash!r}"
        )

    build_configuration = manifest.get("build_configuration")
    if not isinstance(build_configuration, dict):
        raise DenseIndexError("dense index build_configuration must be an object")
    _validate_encoder_policy_binding(build_configuration)
    if expected_build_configuration is not None:
        normalized_expected = _json_mapping(
            expected_build_configuration,
            "expected build configuration",
        )
        if build_configuration != normalized_expected:
            raise DenseIndexError("dense index build configuration does not match")

    matrix_entry = _artifact_entry(manifest, "embeddings")
    ids_entry = _artifact_entry(manifest, "trial_ids")
    _validate_artifact(matrix_path, matrix_entry, expected_filename=DENSE_MATRIX_FILENAME)
    _validate_artifact(ids_path, ids_entry, expected_filename=DENSE_ID_MAP_FILENAME)

    trial_ids = _read_id_map(ids_path)
    if expected_trial_ids is not None and trial_ids != tuple(expected_trial_ids):
        raise DenseIndexError("dense trial ID map does not match the requested corpus order")

    document_count = manifest.get("document_count")
    dimension = manifest.get("dimension")
    if isinstance(document_count, bool) or not isinstance(document_count, int):
        raise DenseIndexError("dense index document_count must be an integer")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise DenseIndexError("dense index dimension must be a positive integer")
    if document_count != len(trial_ids):
        raise DenseIndexError("dense index document count does not match the ID map")

    numpy = _require_numpy()
    try:
        embeddings = numpy.load(matrix_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DenseIndexError(f"invalid dense embedding matrix: {exc}") from exc
    if embeddings.ndim != 2 or embeddings.shape != (document_count, dimension):
        raise DenseIndexError("dense embedding matrix shape does not match its manifest")
    if embeddings.dtype != numpy.dtype("<f4"):
        raise DenseIndexError("dense embedding matrix must use little-endian float32")

    return DenseIndex(
        directory=source,
        trial_ids=trial_ids,
        embeddings=embeddings,
        manifest=manifest,
    )


def exact_inner_product_rank(
    trial_ids: Sequence[str],
    document_embeddings: object,
    query_embedding: object,
    *,
    top_k: int = DEFAULT_RETRIEVAL_DEPTH,
    normalize_query: bool,
) -> list[tuple[str, float]]:
    """Rank document vectors by exact inner product.

    ``normalize_query`` is required and is the metric control: True normalizes
    the query so scores are cosine against L2-normalized index rows; False
    keeps the query's trained magnitude and scores the raw dot product.  There
    is deliberately no default - scoring an unnormalized matrix with cosine, or
    a normalized one with raw dot, silently produces plausible but wrong
    rankings.
    """

    if not isinstance(normalize_query, bool):
        raise ValueError("normalize_query must be a boolean")
    limit = _non_negative_integer(top_k, "top_k")
    ids = tuple(trial_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("trial_ids must be unique")
    if any(not isinstance(trial_id, str) or not trial_id for trial_id in ids):
        raise ValueError("trial_ids must be non-empty strings")

    numpy = _require_numpy()
    matrix = numpy.asarray(document_embeddings)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise DenseIndexError("document embedding rows must match trial_ids")
    if matrix.shape[1] < 1:
        raise DenseIndexError("document embeddings must have at least one column")
    if matrix.dtype.kind != "f":
        raise DenseIndexError("document embeddings must use a floating-point dtype")
    if normalize_query:
        query = _normalized_query(query_embedding, dimension=int(matrix.shape[1]))
    else:
        query = numpy.asarray(query_embedding, dtype=numpy.float32)
        if query.ndim != 1 or query.shape[0] != int(matrix.shape[1]):
            raise DenseIndexError(f"query vector must have shape ({int(matrix.shape[1])},)")
        if not bool(numpy.isfinite(query).all()):
            raise DenseIndexError("query vector contains non-finite values")
        if not bool((query != 0.0).any()):
            raise DenseIndexError("query vector must be non-zero")
    scores = numpy.asarray(matrix @ query, dtype=numpy.float32)
    if not bool(numpy.isfinite(scores).all()):
        raise DenseIndexError("inner-product search produced non-finite scores")

    result_count = min(limit, len(ids))
    if result_count == 0:
        return []
    if result_count == len(ids):
        selected = list(range(len(ids)))
    else:
        partition = numpy.argpartition(scores, len(ids) - result_count)[len(ids) - result_count :]
        threshold = float(scores[partition].min())
        above = [int(index) for index in numpy.flatnonzero(scores > threshold)]
        tied = sorted(
            (int(index) for index in numpy.flatnonzero(scores == threshold)),
            key=lambda index: ids[index],
        )
        selected = above + tied[: result_count - len(above)]
    selected.sort(key=lambda index: (-float(scores[index]), ids[index]))
    return [(ids[index], float(scores[index])) for index in selected]


def dense_index_normalization_control(index: DenseIndex) -> str:
    """Read the index manifest's recorded normalization control."""

    build_configuration = index.manifest.get("build_configuration")
    if not isinstance(build_configuration, dict):
        raise DenseIndexError("dense index build_configuration must be an object")
    normalization = build_configuration.get("normalization")
    if normalization not in (NORMALIZATION_L2, NORMALIZATION_NONE):
        raise DenseIndexError("dense index records an unsupported normalization control")
    return normalization


def dense_index_normalizes_documents(index: DenseIndex) -> bool:
    return dense_index_normalization_control(index) != NORMALIZATION_NONE


class DenseRetriever:
    """Encode benchmark topics and search one verified exact dense index."""

    def __init__(
        self,
        index: DenseIndex,
        encoder: TextEncoder,
        *,
        query_batch_size: int = DEFAULT_QUERY_BATCH_SIZE,
    ) -> None:
        if not isinstance(index, DenseIndex):
            raise TypeError("index must be a DenseIndex")
        self.index = index
        self.encoder = encoder
        self.query_batch_size = _positive_integer(query_batch_size, "query_batch_size")
        build_configuration = index.manifest.get("build_configuration")
        indexed_encoder = (
            build_configuration.get("encoder") if isinstance(build_configuration, dict) else None
        )
        if indexed_encoder != _encoder_configuration(encoder):
            raise DenseIndexError("query encoder configuration does not match the dense index")
        if not isinstance(build_configuration, dict):
            raise DenseIndexError("dense index build_configuration must be an object")
        document_batch_size = build_configuration.get("document_batch_size")
        checkpoint_interval_rows = build_configuration.get("checkpoint_interval_rows")
        expected_dimension = build_configuration.get("expected_dimension")
        expected_configuration = _build_configuration(
            encoder,
            document_batch_size=_positive_integer(
                document_batch_size,
                "indexed document_batch_size",
            ),
            checkpoint_interval_rows=_positive_integer(
                checkpoint_interval_rows,
                "indexed checkpoint_interval_rows",
            ),
            expected_dimension=_positive_integer(
                expected_dimension,
                "indexed expected_dimension",
            ),
        )
        selected_configuration = _select_recorded_build_configuration(
            build_configuration,
            expected_configuration,
        )
        indexed_policy = selected_configuration.get("encoder_policy")
        indexed_policy_hash = selected_configuration.get("encoder_policy_sha256")
        policy = _encoder_policy(encoder)
        if indexed_policy != policy.to_dict() or indexed_policy_hash != policy.policy_hash():
            raise DenseIndexError("query encoder policy does not match the dense index")
        # Fail closed on a metric mismatch: an unnormalized index scored with
        # cosine - or a normalized index scored with raw dot product - is a
        # configuration error, never a silent choice.
        self.query_normalization = _policy_normalizes_vectors(policy)
        if self.query_normalization is not dense_index_normalizes_documents(index):
            raise DenseIndexError(
                "query normalization control does not match the dense index "
                f"normalization control ({policy.normalization} query vs "
                f"{dense_index_normalization_control(index)} index)"
            )
        self.encoder_policy_binding = "registry-policy-sha256"

    def retrieve_many(
        self,
        topics: Sequence[BenchmarkTopic],
        *,
        run_id: str,
        system_id: str = "dense-bge-m3",
        top_k: int = DEFAULT_RETRIEVAL_DEPTH,
    ) -> list[Candidate]:
        """Retrieve exact dense candidates for each topic in topic input order."""

        limit = _non_negative_integer(top_k, "top_k")
        rows = tuple(topics)
        if any(
            not isinstance(getattr(topic, "topic_id", None), str)
            or not isinstance(getattr(topic, "canonical_text", None), str)
            for topic in rows
        ):
            raise TypeError("topics must contain BenchmarkTopic records")
        topic_ids = [topic.topic_id for topic in rows]
        if len(topic_ids) != len(set(topic_ids)):
            raise ValueError("topic_id values must be unique")
        if not rows:
            return []

        encoded = _float32_matrix(
            self.encoder.encode_queries(
                [topic.canonical_text for topic in rows],
                batch_size=self.query_batch_size,
            ),
            expected_rows=len(rows),
            field_name="query embeddings",
        )
        if encoded.shape[1] != self.index.dimension:
            raise DenseIndexError("query embedding dimension does not match the dense index")

        candidates: list[Candidate] = []
        for topic, query_vector in zip(rows, encoded, strict=True):
            ranked = self.index.rank(
                query_vector,
                top_k=limit,
                normalize_query=self.query_normalization,
            )
            candidates.extend(
                Candidate(
                    run_id=run_id,
                    system_id=system_id,
                    topic_id=topic.topic_id,
                    trial_id=trial_id,
                    rank=rank,
                    score=score,
                )
                for rank, (trial_id, score) in enumerate(ranked, start=1)
            )
        return candidates

    def configuration(self, *, top_k: int = DEFAULT_RETRIEVAL_DEPTH) -> dict[str, JsonValue]:
        """Return complete JSON-compatible retrieval/model/index configuration."""

        retrieval_depth = _non_negative_integer(top_k, "top_k")
        artifacts = self.index.manifest.get("artifacts")
        policy = _encoder_policy(self.encoder)
        return {
            "implementation": "taim.baselines.dense.DenseRetriever",
            "indexed_fields": ["TrialDocument.canonical_text"],
            "requested_top_k": retrieval_depth,
            "tie_breaking": "score descending, trial_id ascending",
            "model": _encoder_configuration(self.encoder),
            "inference": {
                "query_batch_size": self.query_batch_size,
                "query_format": policy.query_format.description,
                "output_dtype": "float32",
                "normalization": _recorded_normalization(policy),
            },
            "index": {
                "type": "exact_inner_product",
                **(
                    {"normalized_vectors_equivalent_to": "cosine"}
                    if self.query_normalization
                    else {
                        "normalized_vectors_equivalent_to": None,
                        "score_metric": ("unnormalized dot product over raw-magnitude vectors"),
                    }
                ),
                "directory": str(self.index.directory),
                "manifest_filename": DENSE_INDEX_MANIFEST_FILENAME,
                "manifest_sha256": sha256_file(
                    self.index.directory / DENSE_INDEX_MANIFEST_FILENAME
                ),
                "corpus_hash": self.index.corpus_hash,
                "document_count": len(self.index.trial_ids),
                "dimension": self.index.dimension,
                "build_configuration": self.index.manifest["build_configuration"],
                "encoder_policy_binding": self.encoder_policy_binding,
                "artifacts": artifacts,
            },
        }


__all__ = [
    "BGE_M3_ATTENTION_HEAD_COUNT",
    "BGE_M3_ATTENTION_SCORE_BUDGET_BYTES",
    "BGE_M3_EMBEDDING_DIMENSION",
    "BGE_M3_MAX_LENGTH",
    "BGE_M3_MODEL_ID",
    "BGE_M3_MODEL_REVISION",
    "BGE_M3_TOKENIZER_REVISION",
    "DEFAULT_CHECKPOINT_INTERVAL_ROWS",
    "DEFAULT_DOCUMENT_BATCH_SIZE",
    "DEFAULT_QUERY_BATCH_SIZE",
    "DEFAULT_RETRIEVAL_DEPTH",
    "DENSE_BUILD_ID_MAP_FILENAME",
    "DENSE_BUILD_MATRIX_FILENAME",
    "DENSE_BUILD_PROGRESS_FILENAME",
    "DENSE_ID_MAP_FILENAME",
    "DENSE_INDEX_MANIFEST_FILENAME",
    "DENSE_MATRIX_FILENAME",
    "QWEN3_EMBEDDING_06B_ATTENTION_HEAD_COUNT",
    "QWEN3_EMBEDDING_06B_ATTENTION_SCORE_BUDGET_BYTES",
    "QWEN3_EMBEDDING_06B_INFERENCE_VALUE_BYTES",
    "QWEN3_EMBEDDING_06B_MODEL_FILES",
    "QWEN3_EMBEDDING_06B_TOKENIZER_FILES",
    "BgeM3Encoder",
    "DenseDependencyError",
    "DenseIndex",
    "DenseIndexError",
    "DenseRetriever",
    "Qwen3Embedding06BEncoder",
    "TextEncoder",
    "TransformerBiEncoder",
    "cls_pool",
    "exact_inner_product_rank",
    "load_dense_index",
    "masked_mean_pool",
    "prepare_dense_index",
    "qwen_last_token_pool",
    "reset_dense_index",
    "sha256_file",
]
