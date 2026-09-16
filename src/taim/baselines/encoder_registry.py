"""Immutable dense-encoder policies and their System bindings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar, TypeAlias

from taim.schemas import JsonValue

ENCODER_POLICY_SCHEMA_VERSION = "1.0"
TEXT_PLACEHOLDER = "{text}"

_ALLOWED_POOLING = frozenset({"CLS", "last_token", "mean"})
_ALLOWED_PADDING = frozenset({"dynamic", "maximum_length"})
_ALLOWED_SIDES = frozenset({"left", "right"})
_ALLOWED_TRUNCATION = frozenset({"disabled", "maximum_length"})
_ALLOWED_DTYPES = frozenset({"bfloat16", "float16", "float32"})
_ALLOWED_NORMALIZATION = frozenset({"L2", "encoder-L2", "none"})
_ALLOWED_DEVICE_POLICIES = frozenset({"cpu-only", "runtime-selected-exact-match"})


class EncoderRegistryError(ValueError):
    """Raised when a dense encoder policy, binding, or factory is invalid."""


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EncoderRegistryError(f"{field_name} must be a non-empty string")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EncoderRegistryError(f"{field_name} must be a positive integer")
    return value


def _validated_choice(value: object, allowed: frozenset[str], field_name: str) -> str:
    text = _non_empty_string(value, field_name)
    if text not in allowed:
        choices = ", ".join(sorted(allowed))
        raise EncoderRegistryError(f"{field_name} must be one of: {choices}; got {text!r}")
    return text


@dataclass(frozen=True, slots=True)
class TextFormatPolicy:
    """One role-specific, executable text-formatting policy."""

    template: str
    description: str

    def __post_init__(self) -> None:
        template = _non_empty_string(self.template, "template")
        _non_empty_string(self.description, "description")
        occurrences = template.count(TEXT_PLACEHOLDER)
        if occurrences != 1:
            raise EncoderRegistryError(
                f"template must contain exactly one {TEXT_PLACEHOLDER} placeholder; "
                f"found {occurrences}"
            )

    def render(self, text: str) -> str:
        """Format exactly one role-specific input string."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self.template.replace(TEXT_PLACEHOLDER, text)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"description": self.description, "template": self.template}


@dataclass(frozen=True, slots=True)
class EncoderPolicy:
    """Complete, content-addressed policy for one text-to-vector encoder."""

    schema_version: ClassVar[str] = ENCODER_POLICY_SCHEMA_VERSION

    encoder_id: str
    factory_id: str
    implementation: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    embedding_dimension: int
    maximum_length: int
    query_format: TextFormatPolicy
    document_format: TextFormatPolicy
    pooling: str
    padding: str
    padding_side: str
    truncation: str
    truncation_side: str
    model_dtype: str
    output_dtype: str
    normalization: str
    device_policy: str
    backend: str
    attention_implementation: str
    trust_remote_code: bool
    document_model_id: str | None = None
    document_model_revision: str | None = None
    document_tokenizer_id: str | None = None
    document_tokenizer_revision: str | None = None
    query_maximum_length: int | None = None
    document_maximum_length: int | None = None

    @property
    def is_asymmetric(self) -> bool:
        """Report whether queries and documents use two different encoders."""

        return self.document_model_id is not None

    @property
    def effective_document_model_id(self) -> str:
        return self.document_model_id or self.model_id

    @property
    def effective_document_model_revision(self) -> str:
        return self.document_model_revision or self.model_revision

    @property
    def effective_document_tokenizer_id(self) -> str:
        return self.document_tokenizer_id or self.tokenizer_id

    @property
    def effective_document_tokenizer_revision(self) -> str:
        return self.document_tokenizer_revision or self.tokenizer_revision

    @property
    def effective_query_maximum_length(self) -> int:
        return self.query_maximum_length or self.maximum_length

    @property
    def effective_document_maximum_length(self) -> int:
        return self.document_maximum_length or self.maximum_length

    def __post_init__(self) -> None:
        for field_name in (
            "encoder_id",
            "factory_id",
            "implementation",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "backend",
            "attention_implementation",
        ):
            _non_empty_string(getattr(self, field_name), field_name)
        _positive_integer(self.embedding_dimension, "embedding_dimension")
        _positive_integer(self.maximum_length, "maximum_length")
        if not isinstance(self.query_format, TextFormatPolicy):
            raise EncoderRegistryError("query_format must be a TextFormatPolicy")
        if not isinstance(self.document_format, TextFormatPolicy):
            raise EncoderRegistryError("document_format must be a TextFormatPolicy")
        _validated_choice(self.pooling, _ALLOWED_POOLING, "pooling")
        _validated_choice(self.padding, _ALLOWED_PADDING, "padding")
        _validated_choice(self.padding_side, _ALLOWED_SIDES, "padding_side")
        _validated_choice(self.truncation, _ALLOWED_TRUNCATION, "truncation")
        _validated_choice(self.truncation_side, _ALLOWED_SIDES, "truncation_side")
        _validated_choice(self.model_dtype, _ALLOWED_DTYPES, "model_dtype")
        _validated_choice(self.output_dtype, _ALLOWED_DTYPES, "output_dtype")
        _validated_choice(self.normalization, _ALLOWED_NORMALIZATION, "normalization")
        _validated_choice(self.device_policy, _ALLOWED_DEVICE_POLICIES, "device_policy")
        if not isinstance(self.trust_remote_code, bool):
            raise EncoderRegistryError("trust_remote_code must be a boolean")
        document_side = (
            self.document_model_id,
            self.document_model_revision,
            self.document_tokenizer_id,
            self.document_tokenizer_revision,
        )
        if any(value is not None for value in document_side):
            if any(value is None for value in document_side):
                raise EncoderRegistryError(
                    "an asymmetric encoder must declare document_model_id, "
                    "document_model_revision, document_tokenizer_id, and "
                    "document_tokenizer_revision together"
                )
            for field_name in (
                "document_model_id",
                "document_model_revision",
                "document_tokenizer_id",
                "document_tokenizer_revision",
            ):
                _non_empty_string(getattr(self, field_name), field_name)
        for field_name in ("query_maximum_length", "document_maximum_length"):
            value = getattr(self, field_name)
            if value is None:
                continue
            _positive_integer(value, field_name)
            if value > self.maximum_length:
                raise EncoderRegistryError(
                    f"{field_name} must not exceed maximum_length {self.maximum_length}"
                )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return deterministic JSON-compatible policy data."""

        payload: dict[str, JsonValue] = {
            "attention_implementation": self.attention_implementation,
            "backend": self.backend,
            "device_policy": self.device_policy,
            "document_format": self.document_format.to_dict(),
            "embedding_dimension": self.embedding_dimension,
            "encoder_id": self.encoder_id,
            "factory_id": self.factory_id,
            "implementation": self.implementation,
            "maximum_length": self.maximum_length,
            "model_dtype": self.model_dtype,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "normalization": self.normalization,
            "output_dtype": self.output_dtype,
            "padding": self.padding,
            "padding_side": self.padding_side,
            "pooling": self.pooling,
            "query_format": self.query_format.to_dict(),
            "schema_version": self.schema_version,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "truncation": self.truncation,
            "truncation_side": self.truncation_side,
            "trust_remote_code": self.trust_remote_code,
        }
        # Optional keys are emitted only when set.  A symmetric policy therefore
        # serializes - and hashes - exactly as it did before these fields existed.
        optional: dict[str, JsonValue] = {
            "document_model_id": self.document_model_id,
            "document_model_revision": self.document_model_revision,
            "document_tokenizer_id": self.document_tokenizer_id,
            "document_tokenizer_revision": self.document_tokenizer_revision,
            "query_maximum_length": self.query_maximum_length,
            "document_maximum_length": self.document_maximum_length,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload

    def to_json(self) -> str:
        """Return the canonical JSON representation used for hashing."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def policy_hash(self) -> str:
        """Return the stable SHA-256 identity of the complete policy."""

        digest = hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


@dataclass(frozen=True, slots=True)
class SystemEncoderBinding:
    """Bind one runnable System ID to one encoder policy."""

    system_id: str
    encoder_id: str

    def __post_init__(self) -> None:
        _non_empty_string(self.system_id, "system_id")
        _non_empty_string(self.encoder_id, "encoder_id")


EncoderBuilder: TypeAlias = Callable[[EncoderPolicy, str], object]


@dataclass(frozen=True, slots=True)
class EncoderRegistry:
    """Immutable validated registry plus its runtime encoder factories."""

    entries: tuple[EncoderPolicy, ...]
    bindings: tuple[SystemEncoderBinding, ...]
    factories: Mapping[str, EncoderBuilder] = field(repr=False, compare=False, hash=False)
    _entries_by_id: Mapping[str, EncoderPolicy] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _bindings_by_id: Mapping[str, SystemEncoderBinding] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        bindings = tuple(self.bindings)
        factories = MappingProxyType(dict(self.factories))
        if not entries:
            raise EncoderRegistryError("encoder registry requires at least one entry")
        if any(not isinstance(entry, EncoderPolicy) for entry in entries):
            raise EncoderRegistryError("encoder registry entries must be EncoderPolicy values")
        if not bindings:
            raise EncoderRegistryError("encoder registry requires at least one System binding")
        if any(not isinstance(binding, SystemEncoderBinding) for binding in bindings):
            raise EncoderRegistryError(
                "encoder registry bindings must be SystemEncoderBinding values"
            )
        if any(
            not isinstance(factory_id, str) or not factory_id or not callable(factory)
            for factory_id, factory in factories.items()
        ):
            raise EncoderRegistryError(
                "encoder registry factories must map non-empty IDs to callables"
            )
        entry_ids = [entry.encoder_id for entry in entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise EncoderRegistryError("encoder registry entry IDs must be unique")
        binding_ids = [binding.system_id for binding in bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise EncoderRegistryError("dense System IDs must be unique")
        entries_by_id = {entry.encoder_id: entry for entry in entries}
        bindings_by_id = {binding.system_id: binding for binding in bindings}
        for entry in entries:
            if entry.factory_id not in factories:
                raise EncoderRegistryError(
                    f"encoder {entry.encoder_id!r} references unknown factory {entry.factory_id!r}"
                )
        for binding in bindings:
            if binding.encoder_id not in entries_by_id:
                raise EncoderRegistryError(
                    f"System {binding.system_id!r} references unknown encoder "
                    f"{binding.encoder_id!r}"
                )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "factories", factories)
        object.__setattr__(self, "_entries_by_id", MappingProxyType(entries_by_id))
        object.__setattr__(self, "_bindings_by_id", MappingProxyType(bindings_by_id))

    def available_encoder_ids(self) -> tuple[str, ...]:
        return tuple(sorted(entry.encoder_id for entry in self.entries))

    def available_system_ids(self) -> tuple[str, ...]:
        return tuple(sorted(binding.system_id for binding in self.bindings))

    def policy_for_encoder(self, encoder_id: str) -> EncoderPolicy:
        key = _non_empty_string(encoder_id, "encoder_id")
        entry = self._entries_by_id.get(key)
        if entry is not None:
            return entry
        known = ", ".join(self.available_encoder_ids())
        raise EncoderRegistryError(f"unknown encoder ID {key!r}; registered encoder IDs: {known}")

    def binding_for_system(self, system_id: str) -> SystemEncoderBinding:
        key = _non_empty_string(system_id, "system_id")
        binding = self._bindings_by_id.get(key)
        if binding is not None:
            return binding
        known = ", ".join(self.available_system_ids())
        raise EncoderRegistryError(
            f"unknown dense System ID {key!r}; registered dense System IDs: {known}"
        )

    def policy_for_system(self, system_id: str) -> EncoderPolicy:
        return self.policy_for_encoder(self.binding_for_system(system_id).encoder_id)

    def create_encoder(self, system_id: str, *, device: str) -> object:
        """Construct and validate the encoder bound to ``system_id``."""

        selected_device = _non_empty_string(device, "device")
        policy = self.policy_for_system(system_id)
        if policy.device_policy == "cpu-only" and selected_device != "cpu":
            raise EncoderRegistryError(
                f"encoder {policy.encoder_id!r} requires device 'cpu'; got {selected_device!r}"
            )
        encoder = self.factories[policy.factory_id](policy, selected_device)
        if getattr(encoder, "policy", None) != policy:
            raise EncoderRegistryError(
                f"factory {policy.factory_id!r} returned an encoder with the wrong policy"
            )
        for method_name in ("configuration", "encode_documents", "encode_queries"):
            if not callable(getattr(encoder, method_name, None)):
                raise EncoderRegistryError(
                    f"factory {policy.factory_id!r} returned an encoder without {method_name}()"
                )
        return encoder


BGE_M3_POLICY = EncoderPolicy(
    encoder_id="bge-m3",
    factory_id="sentence-transformers-bge-m3-v1",
    implementation="taim.baselines.dense.BgeM3Encoder",
    model_id="BAAI/bge-m3",
    model_revision="5617a9f61b028005a4858fdac845db406aefb181",
    tokenizer_id="BAAI/bge-m3",
    tokenizer_revision="5617a9f61b028005a4858fdac845db406aefb181",
    embedding_dimension=1_024,
    maximum_length=8_192,
    query_format=TextFormatPolicy(
        template=TEXT_PLACEHOLDER,
        description="BenchmarkTopic.canonical_text unchanged; no instruction",
    ),
    document_format=TextFormatPolicy(
        template=TEXT_PLACEHOLDER,
        description="TrialDocument.canonical_text unchanged",
    ),
    pooling="CLS",
    padding="dynamic",
    padding_side="right",
    truncation="maximum_length",
    truncation_side="right",
    model_dtype="float32",
    output_dtype="float32",
    normalization="L2",
    device_policy="runtime-selected-exact-match",
    backend="torch",
    attention_implementation="eager",
    trust_remote_code=False,
)

QWEN3_EMBEDDING_06B_QUERY_TEMPLATE = (
    "Instruct: Given a patient description, retrieve clinical trials for which the patient "
    "may be eligible.\nQuery: {text}"
)

QWEN3_EMBEDDING_06B_POLICY = EncoderPolicy(
    encoder_id="qwen3-embedding-0.6b",
    factory_id="transformers-qwen3-embedding-0.6b-v1",
    implementation="taim.baselines.dense.Qwen3Embedding06BEncoder",
    model_id="Qwen/Qwen3-Embedding-0.6B",
    model_revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    tokenizer_id="Qwen/Qwen3-Embedding-0.6B",
    tokenizer_revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    embedding_dimension=1_024,
    maximum_length=8_192,
    query_format=TextFormatPolicy(
        template=QWEN3_EMBEDDING_06B_QUERY_TEMPLATE,
        description=("BenchmarkTopic.canonical_text with the frozen patient-to-trial instruction"),
    ),
    document_format=TextFormatPolicy(
        template=TEXT_PLACEHOLDER,
        description="TrialDocument.canonical_text unchanged; no instruction",
    ),
    pooling="last_token",
    padding="dynamic",
    padding_side="left",
    truncation="maximum_length",
    truncation_side="right",
    model_dtype="bfloat16",
    output_dtype="float32",
    normalization="encoder-L2",
    device_policy="runtime-selected-exact-match",
    backend="torch-transformers",
    attention_implementation="eager",
    trust_remote_code=False,
)

QWEN3_EMBEDDING_06B_NO_TEMPLATE_POLICY = replace(
    QWEN3_EMBEDDING_06B_POLICY,
    encoder_id="qwen3-embedding-0.6b-no-template",
    query_format=TextFormatPolicy(
        template=TEXT_PLACEHOLDER,
        description="BenchmarkTopic.canonical_text unchanged; no instruction",
    ),
)


def _create_bge_m3_encoder(policy: EncoderPolicy, device: str) -> object:
    from taim.baselines.dense import BgeM3Encoder

    return BgeM3Encoder(policy=policy, device=device)


def _create_qwen3_embedding_06b_encoder(policy: EncoderPolicy, device: str) -> object:
    from taim.baselines.dense import Qwen3Embedding06BEncoder

    return Qwen3Embedding06BEncoder(policy=policy, device=device)


DENSE_ENCODER_REGISTRY = EncoderRegistry(
    entries=(
        BGE_M3_POLICY,
        QWEN3_EMBEDDING_06B_POLICY,
        QWEN3_EMBEDDING_06B_NO_TEMPLATE_POLICY,
    ),
    bindings=(
        SystemEncoderBinding(
            system_id="dense-bge-m3",
            encoder_id=BGE_M3_POLICY.encoder_id,
        ),
        SystemEncoderBinding(
            system_id="dense-qwen3-embedding-0.6b",
            encoder_id=QWEN3_EMBEDDING_06B_POLICY.encoder_id,
        ),
        SystemEncoderBinding(
            system_id="dense-qwen3-embedding-0.6b-no-template",
            encoder_id=QWEN3_EMBEDDING_06B_NO_TEMPLATE_POLICY.encoder_id,
        ),
    ),
    factories={
        BGE_M3_POLICY.factory_id: _create_bge_m3_encoder,
        QWEN3_EMBEDDING_06B_POLICY.factory_id: _create_qwen3_embedding_06b_encoder,
    },
)


__all__ = [
    "BGE_M3_POLICY",
    "DENSE_ENCODER_REGISTRY",
    "ENCODER_POLICY_SCHEMA_VERSION",
    "QWEN3_EMBEDDING_06B_NO_TEMPLATE_POLICY",
    "QWEN3_EMBEDDING_06B_POLICY",
    "QWEN3_EMBEDDING_06B_QUERY_TEMPLATE",
    "TEXT_PLACEHOLDER",
    "EncoderBuilder",
    "EncoderPolicy",
    "EncoderRegistry",
    "EncoderRegistryError",
    "SystemEncoderBinding",
    "TextFormatPolicy",
]
