"""Checksum-locked Source Bundle metadata for Snapshot preparation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from taim.contracts import (
    SOURCE_BUNDLE_VERSION,
    content_sha256,
    require_non_empty,
    require_sha256,
)
from taim.schemas import JsonValue, SchemaValidationError


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """One checksum-locked raw artifact descriptor, not its raw bytes."""

    artifact_id: str
    role: str
    filename: str
    byte_size: int
    sha256: str
    acquisition_date: str
    access_terms: str
    redistribution_terms: str
    url: str

    def __post_init__(self) -> None:
        require_non_empty(self.artifact_id, "artifact_id")
        require_non_empty(self.role, "source artifact role")
        filename = require_non_empty(self.filename, "source artifact filename")
        if Path(filename).name != filename:
            raise SchemaValidationError("source artifact filename must be a plain filename")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size < 0
        ):
            raise SchemaValidationError("source artifact byte_size must be non-negative")
        require_sha256(self.sha256, "source artifact sha256")
        for name in ("acquisition_date", "access_terms", "redistribution_terms", "url"):
            require_non_empty(getattr(self, name), f"source artifact {name}")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "artifact_id": self.artifact_id,
            "role": self.role,
            "filename": self.filename,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "acquisition_date": self.acquisition_date,
            "access_terms": self.access_terms,
            "redistribution_terms": self.redistribution_terms,
            "url": self.url,
        }

    def identity_dict(self) -> dict[str, JsonValue]:
        return {
            "artifact_id": self.artifact_id,
            "role": self.role,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "acquisition_date": self.acquisition_date,
            "access_terms": self.access_terms,
            "redistribution_terms": self.redistribution_terms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SourceArtifact:
        return cls(
            artifact_id=cast(str, payload.get("artifact_id")),
            role=cast(str, payload.get("role")),
            filename=cast(str, payload.get("filename")),
            byte_size=cast(int, payload.get("byte_size")),
            sha256=cast(str, payload.get("sha256")),
            acquisition_date=cast(str, payload.get("acquisition_date")),
            access_terms=cast(str, payload.get("access_terms")),
            redistribution_terms=cast(str, payload.get("redistribution_terms")),
            url=cast(str, payload.get("url")),
        )


@dataclass(frozen=True, slots=True)
class SourceBundle:
    """Resolved Source Bundle metadata used for preparation and audit."""

    artifacts: tuple[SourceArtifact, ...]
    lock_filename: str
    lock_sha256: str
    source_bundle_id: str = field(init=False)

    schema_version = SOURCE_BUNDLE_VERSION

    def __post_init__(self) -> None:
        artifacts = tuple(self.artifacts)
        if not artifacts or any(not isinstance(item, SourceArtifact) for item in artifacts):
            raise SchemaValidationError("Source Bundle must contain source artifacts")
        ids = [item.artifact_id for item in artifacts]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError("Source Bundle artifact_id values must be unique")
        filename = require_non_empty(self.lock_filename, "Source Bundle lock filename")
        if Path(filename).name != filename:
            raise SchemaValidationError("Source Bundle lock filename must be plain")
        require_sha256(self.lock_sha256, "Source Bundle lock sha256")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "source_bundle_id", self.identity_for_artifacts(ids))

    def identity_for_artifacts(self, artifact_ids: Iterable[str]) -> str:
        """Identify a declared artifact subset without exposing excluded material."""

        requested = frozenset(artifact_ids)
        known = {item.artifact_id for item in self.artifacts}
        if not requested or not requested <= known:
            unknown = sorted(requested - known)
            raise SchemaValidationError(
                "Source Bundle identity subset is empty or unknown: " + ", ".join(unknown)
            )
        selected = sorted(
            (item for item in self.artifacts if item.artifact_id in requested),
            key=lambda item: item.artifact_id,
        )
        return content_sha256(
            {
                "source_bundle_schema_version": self.schema_version,
                "artifacts": [item.identity_dict() for item in selected],
            }
        )

    def artifact_by_id(self, artifact_id: str) -> SourceArtifact:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise SchemaValidationError(f"unknown Source Bundle artifact_id {artifact_id!r}")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "source_bundle_id": self.source_bundle_id,
            "lock_filename": self.lock_filename,
            "lock_sha256": self.lock_sha256,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SourceBundle:
        if payload.get("schema_version") != SOURCE_BUNDLE_VERSION:
            raise SchemaValidationError("unsupported Source Bundle schema_version")
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise SchemaValidationError("Source Bundle artifacts must be an array")
        if any(not isinstance(item, Mapping) for item in raw_artifacts):
            raise SchemaValidationError("Source Bundle artifact entries must be objects")
        bundle = cls(
            artifacts=tuple(
                SourceArtifact.from_dict(cast(Mapping[str, object], item)) for item in raw_artifacts
            ),
            lock_filename=cast(str, payload.get("lock_filename")),
            lock_sha256=cast(str, payload.get("lock_sha256")),
        )
        if payload.get("source_bundle_id") != bundle.source_bundle_id:
            raise SchemaValidationError("Source Bundle identity does not match its artifacts")
        return bundle


__all__ = ["SourceArtifact", "SourceBundle"]
