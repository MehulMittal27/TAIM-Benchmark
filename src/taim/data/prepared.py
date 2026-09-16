"""Physical prepared-package seam for the Benchmark Snapshot contract."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar, cast, overload

from taim.contracts import (
    EVALUATION_PACKAGE_VERSION,
    SNAPSHOT_CANONICALIZATION_VERSION,
    SNAPSHOT_CONTRACT_VERSION,
    content_sha256,
)
from taim.evaluation_package import ConnectorDiagnostic, EvaluationPackage
from taim.file_hash import sha256_file
from taim.judgments import TREC_CT_JUDGMENT_SCHEME, JudgmentScheme
from taim.schemas import (
    JsonValue,
    RelevanceJudgment,
    SchemaValidationError,
    freeze_json_value_serializable,
    json_value_to_builtins,
)
from taim.snapshot import (
    BenchmarkSnapshot,
    BenchmarkTopic,
    DerivedView,
    FieldProvenance,
    FrozenTrialDocuments,
    TrialDocument,
)
from taim.source import SourceArtifact, SourceBundle

PREPARED_MANIFEST_FILENAME = "prepared-manifest.json"
PREPARED_OUTPUT_FILENAMES = {
    "topics": "topics.jsonl",
    "trials": "trials.jsonl",
    "diagnostics": "diagnostics.jsonl",
    "judgments": "judgments.jsonl",
}
_SNAPSHOT_MANIFEST_FIELDS = frozenset(
    {
        "contract_version",
        "canonicalization_version",
        "benchmark_lineage",
        "snapshot_name",
        "snapshot_id",
        "base_snapshot_id",
        "available_capabilities",
        "record_counts",
        "logical_content_hashes",
        "derived_views",
    }
)
_DERIVED_VIEW_REFERENCE_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "version",
        "input_snapshot_id",
        "configuration",
        "view_id",
        "content_reference",
    }
)
_DERIVED_VIEW_CONTENT_REFERENCE_FIELDS = frozenset(
    {"format", "output_name", "content_schema_version", "mapping_key"}
)

_RecordT = TypeVar("_RecordT")
_ResolvedOutput = tuple[Path, Mapping[str, object], str, int]
_TrialShard = tuple[Path, int, str, int]


def _system_input_artifact_ids(recipe: Mapping[str, object]) -> tuple[str, ...]:
    raw_ids = recipe.get("system_input_artifact_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or not all(isinstance(item, str) and item for item in raw_ids)
    ):
        raise SchemaValidationError(
            "Snapshot Source Recipe system_input_artifact_ids must be a non-empty string array"
        )
    if len(raw_ids) != len(set(raw_ids)):
        raise SchemaValidationError("system_input_artifact_ids must be unique")
    return tuple(raw_ids)


def _allowed_artifact_ids(recipe: Mapping[str, object]) -> tuple[str, ...]:
    raw_ids = recipe.get("allowed_artifact_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or not all(isinstance(item, str) and item for item in raw_ids)
    ):
        raise SchemaValidationError(
            "Snapshot Source Recipe allowed_artifact_ids must be a non-empty string array"
        )
    if len(raw_ids) != len(set(raw_ids)):
        raise SchemaValidationError("Snapshot Source Recipe allowed artifacts must be unique")
    return tuple(raw_ids)


def _required_recipe_mapping(recipe: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = recipe.get(name)
    if not isinstance(value, Mapping) or not value:
        raise SchemaValidationError(f"Snapshot Source Recipe {name} must be a non-empty object")
    return value


def _required_recipe_string(recipe: Mapping[str, object], name: str) -> str:
    value = recipe.get(name)
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"Snapshot Source Recipe {name} must be non-empty")
    return value


def _require_non_empty_string_fields(
    value: Mapping[str, object], *, name: str, fields: Iterable[str]
) -> None:
    missing = [
        field
        for field in fields
        if not isinstance(value.get(field), str) or not cast(str, value.get(field)).strip()
    ]
    if missing:
        raise SchemaValidationError(f"Snapshot Source Recipe {name} requires: {', '.join(missing)}")


def _validate_source_recipe(
    recipe: Mapping[str, object],
    *,
    recipe_id: str,
    benchmark_lineage: str,
    source_bundle: SourceBundle,
) -> None:
    if recipe.get("recipe_id") != recipe_id:
        raise SchemaValidationError("Snapshot Source Recipe id does not match its definition")
    if recipe.get("snapshot_contract_version") != SNAPSHOT_CONTRACT_VERSION:
        raise SchemaValidationError("Snapshot Source Recipe contract version is unsupported")
    if recipe.get("benchmark_lineage") != benchmark_lineage:
        raise SchemaValidationError(
            "Snapshot Source Recipe benchmark lineage does not match the Snapshot"
        )
    connector = _required_recipe_mapping(recipe, "connector")
    _require_non_empty_string_fields(connector, name="connector", fields=("name", "version"))
    parsers = _required_recipe_mapping(recipe, "parsers")
    if any(
        not isinstance(key, str) or not isinstance(value, str) or not value.strip()
        for key, value in parsers.items()
    ):
        raise SchemaValidationError("Snapshot Source Recipe parsers must identify frozen parsers")
    dispositions = recipe.get("field_disposition_table")
    if not isinstance(dispositions, list) or not dispositions:
        raise SchemaValidationError(
            "Snapshot Source Recipe field_disposition_table must be a non-empty array"
        )
    if any(not isinstance(item, Mapping) for item in dispositions):
        raise SchemaValidationError(
            "Snapshot Source Recipe field_disposition_table entries must be objects"
        )
    for item in cast(list[Mapping[str, object]], dispositions):
        if not all(
            isinstance(item.get(key), str) and cast(str, item.get(key)).strip()
            for key in ("source", "disposition")
        ):
            raise SchemaValidationError(
                "Snapshot Source Recipe field dispositions require source and disposition"
            )
        required_fields: tuple[str, ...]
        if item.get("disposition") == "intentionally_ignored":
            required_detail = "rationale"
            required_fields = (required_detail,)
        else:
            required_detail = "normalization_rule"
            required_fields = ("target", required_detail)
        _require_non_empty_string_fields(item, name="field disposition", fields=required_fields)
    common_rendering_fields = (
        "id",
        "whitespace",
        "unicode_normalization",
        "separator",
        "omission",
    )
    patient_recipe = _required_recipe_mapping(recipe, "canonical_patient_text_recipe")
    _require_non_empty_string_fields(
        patient_recipe,
        name="canonical_patient_text_recipe",
        fields=common_rendering_fields,
    )
    trial_recipe = _required_recipe_mapping(recipe, "canonical_trial_text_recipe")
    _require_non_empty_string_fields(
        trial_recipe,
        name="canonical_trial_text_recipe",
        fields=(*common_rendering_fields, "repetition"),
    )
    for name in ("section_order", "typed_field_order"):
        order = trial_recipe.get(name)
        if (
            not isinstance(order, list)
            or not order
            or not all(isinstance(item, str) and item for item in order)
        ):
            raise SchemaValidationError(
                f"Snapshot Source Recipe canonical_trial_text_recipe {name} is invalid"
            )
    for name in ("section_labels", "typed_labels"):
        labels = trial_recipe.get(name)
        if (
            not isinstance(labels, Mapping)
            or not labels
            or not all(
                isinstance(key, str) and isinstance(value, str) and value
                for key, value in labels.items()
            )
        ):
            raise SchemaValidationError(
                f"Snapshot Source Recipe canonical_trial_text_recipe {name} is invalid"
            )
    diagnostic_policy = _required_recipe_mapping(recipe, "diagnostic_policy")
    _require_non_empty_string_fields(
        diagnostic_policy,
        name="diagnostic_policy",
        fields=("version", "field_invalidity", "record_invalidity", "conflict", "order"),
    )
    if _required_recipe_string(recipe, "external_enrichment") != "forbidden":
        raise SchemaValidationError(
            "Snapshot Source Recipe external_enrichment policy must be 'forbidden'"
        )

    raw_roles = recipe.get("allowed_source_roles")
    if (
        not isinstance(raw_roles, list)
        or not raw_roles
        or not all(isinstance(item, str) and item for item in raw_roles)
        or len(raw_roles) != len(set(raw_roles))
    ):
        raise SchemaValidationError(
            "Snapshot Source Recipe allowed_source_roles must be a unique string array"
        )
    allowed_ids = frozenset(_allowed_artifact_ids(recipe))
    bundle_by_id = {artifact.artifact_id: artifact for artifact in source_bundle.artifacts}
    bundle_ids = set(bundle_by_id)
    if not allowed_ids <= bundle_ids:
        raise SchemaValidationError(
            "Snapshot Source Recipe allowed artifacts are not in the Source Bundle"
        )
    allowed_roles = {bundle_by_id[artifact_id].role for artifact_id in allowed_ids}
    if set(raw_roles) != allowed_roles:
        raise SchemaValidationError(
            "Snapshot Source Recipe allowed roles do not match its allowed artifacts"
        )
    input_ids = frozenset(_system_input_artifact_ids(recipe))
    if not input_ids <= allowed_ids:
        raise SchemaValidationError(
            "Snapshot Source Recipe System input artifacts must be declared allowed artifacts"
        )


@dataclass(frozen=True, slots=True)
class SnapshotPreparationManifest:
    """Preparation-only lineage for one logical Snapshot package."""

    source_recipe_id: str
    source_recipe_hash: str
    source_bundle_id: str

    def __post_init__(self) -> None:
        _required_recipe_string({"recipe_id": self.source_recipe_id}, "recipe_id")
        if not isinstance(self.source_recipe_hash, str) or not self.source_recipe_hash.startswith(
            "sha256:"
        ):
            raise SchemaValidationError("Snapshot preparation recipe hash is invalid")
        if not isinstance(self.source_bundle_id, str) or not self.source_bundle_id.startswith(
            "sha256:"
        ):
            raise SchemaValidationError("Snapshot preparation Source Bundle identity is invalid")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "source_recipe_id": self.source_recipe_id,
            "source_recipe_hash": self.source_recipe_hash,
            "source_bundle_id": self.source_bundle_id,
        }


@dataclass(frozen=True, slots=True)
class PreparedBenchmark:
    """Validated Snapshot plus preparation lineage and evaluator material."""

    dataset_id: str
    directory: Path
    snapshot: BenchmarkSnapshot
    evaluation_package: EvaluationPackage
    source_bundle: SourceBundle
    preparation: SnapshotPreparationManifest
    diagnostics: tuple[ConnectorDiagnostic, ...]
    package_checksums: Mapping[str, str]
    prepared_manifest: Mapping[str, Any]
    prepared_manifest_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.package_checksums, Mapping) or any(
            not isinstance(filename, str) or not isinstance(checksum, str)
            for filename, checksum in self.package_checksums.items()
        ):
            raise SchemaValidationError("prepared package checksums must map strings to strings")
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(
            self,
            "package_checksums",
            _freeze_json_mapping(self.package_checksums, path="prepared package checksums"),
        )
        object.__setattr__(
            self,
            "prepared_manifest",
            _freeze_json_mapping(self.prepared_manifest, path="prepared manifest"),
        )

    @property
    def name(self) -> str:
        return self.snapshot.snapshot_name

    @property
    def topics(self) -> tuple[BenchmarkTopic, ...]:
        return self.snapshot.topics

    @property
    def trials(self) -> Sequence[TrialDocument]:
        return self.snapshot.trials

    def resolve_provenance(self, provenance: FieldProvenance) -> SourceArtifact:
        """Resolve and validate one field link against Source Bundle metadata."""

        artifact = self.source_bundle.artifact_by_id(provenance.artifact_id)
        if provenance.artifact_sha256 != artifact.sha256:
            raise SchemaValidationError(
                f"provenance digest for {provenance.artifact_id!r} does not match Source Bundle"
            )
        return artifact


@dataclass(frozen=True, slots=True)
class PreparationResult:
    output_directory: Path
    manifest_path: Path
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest",
            _freeze_json_mapping(self.manifest, path="prepared manifest"),
        )


def _require_string_json_keys(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"{path} keys must be strings")
            _require_string_json_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _require_string_json_keys(item, path=f"{path}[{index}]")


def _freeze_json_mapping(value: Mapping[str, object], *, path: str) -> Mapping[str, JsonValue]:
    """Validate, detach, and recursively freeze a JSON object, including finite floats."""

    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{path} must be a JSON object")
    _require_string_json_keys(value, path=path)
    try:
        detached = json.loads(
            json.dumps(
                json_value_to_builtins(value),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{path} contains a non-JSON value") from exc
    return cast(Mapping[str, JsonValue], freeze_json_value_serializable(detached))


def _write_jsonl(path: Path, rows: Iterable[object]) -> dict[str, JsonValue]:
    count = 0
    byte_size = 0
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for row in rows:
            if isinstance(row, RelevanceJudgment):
                serialized = row.to_json()
            else:
                to_json = getattr(row, "to_json", None)
                if not callable(to_json):
                    raise TypeError("prepared package records must provide to_json()")
                serialized = to_json()
            encoded = f"{serialized}\n".encode()
            stream.write(encoded)
            digest.update(encoded)
            byte_size += len(encoded)
            count += 1
    return {
        "filename": path.name,
        "record_count": count,
        "byte_size": byte_size,
        "sha256": "sha256:" + digest.hexdigest(),
    }


def _write_json(path: Path, payload: object) -> None:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(f"{serialized}\n", encoding="utf-8")


def write_prepared_benchmark(
    output_directory: str | Path,
    *,
    dataset_id: str,
    snapshot: BenchmarkSnapshot,
    evaluation_package: EvaluationPackage,
    source_bundle: SourceBundle,
    source_recipe: Mapping[str, JsonValue],
    diagnostics: Iterable[ConnectorDiagnostic],
    created_at: datetime,
    build_provenance: Mapping[str, JsonValue],
) -> PreparationResult:
    """Atomically materialize one manifest-plus-JSONL prepared package."""

    destination = Path(output_directory)
    if destination.exists():
        raise SchemaValidationError(f"output directory already exists: {destination}")
    if not dataset_id.strip():
        raise SchemaValidationError("dataset_id must be non-empty")
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise SchemaValidationError("created_at must include a UTC offset")
    recipe = cast(dict[str, JsonValue], json_value_to_builtins(source_recipe))
    recipe_id = _required_recipe_string(recipe, "recipe_id")
    input_bundle_id = source_bundle.identity_for_artifacts(_system_input_artifact_ids(recipe))
    recipe_hash = content_sha256(recipe)
    preparation = SnapshotPreparationManifest(
        source_recipe_id=recipe_id,
        source_recipe_hash=recipe_hash,
        source_bundle_id=input_bundle_id,
    )
    diagnostics_tuple = tuple(diagnostics)
    _validate_prepared_components(
        snapshot=snapshot,
        evaluation_package=evaluation_package,
        source_bundle=source_bundle,
        source_recipe=recipe,
        preparation=preparation,
        diagnostics=diagnostics_tuple,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name or 'snapshot'}.", dir=destination.parent)
    )
    try:
        paths = {role: staging / filename for role, filename in PREPARED_OUTPUT_FILENAMES.items()}
        outputs = {
            "topics": _write_jsonl(paths["topics"], snapshot.topics),
            "trials": _write_jsonl(paths["trials"], snapshot.trials),
            "diagnostics": _write_jsonl(paths["diagnostics"], diagnostics_tuple),
            "judgments": _write_jsonl(paths["judgments"], evaluation_package.judgments),
        }
        timestamp = created_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        manifest: dict[str, Any] = {
            "schema_version": SNAPSHOT_CONTRACT_VERSION,
            "manifest_type": "taim-prepared-benchmark",
            "dataset_id": dataset_id,
            "created_at": timestamp,
            "source_bundle": source_bundle.to_dict(),
            "snapshot_source_recipe": {
                "recipe_id": preparation.source_recipe_id,
                "sha256": recipe_hash,
                "definition": recipe,
            },
            "snapshot_preparation": preparation.to_dict(),
            "snapshot": snapshot.manifest_dict(),
            "evaluation_package": evaluation_package.manifest_dict(),
            "outputs": outputs,
            "build_provenance": json_value_to_builtins(build_provenance),
        }
        manifest_path = staging / PREPARED_MANIFEST_FILENAME
        _write_json(manifest_path, manifest)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    resolved = destination.resolve()
    return PreparationResult(
        output_directory=resolved,
        manifest_path=resolved / PREPARED_MANIFEST_FILENAME,
        manifest=manifest,
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{name} must be a JSON object")
    return cast(Mapping[str, object], value)


def _safe_output_path(directory: Path, filename: object, *, field: str) -> Path:
    if not isinstance(filename, str) or not filename or "\\" in filename:
        raise SchemaValidationError(f"{field} is invalid")
    relative = PurePosixPath(filename)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SchemaValidationError(f"{field} must be a safe relative POSIX path")
    package_root = directory.resolve()
    path = (directory / Path(*relative.parts)).resolve()
    if not path.is_relative_to(package_root):
        raise SchemaValidationError(f"{field} escapes the prepared package")
    return path


def _validate_output_entry(
    directory: Path,
    entry: Mapping[str, object],
    *,
    role: str,
    sharded: bool,
) -> tuple[Path, Mapping[str, object], str, int]:
    single_file_keys = {"filename", "record_count", "byte_size", "sha256"}
    sharded_keys = single_file_keys | {"compression", "shard_index"}
    expected_keys = sharded_keys if sharded else single_file_keys
    if set(entry) != expected_keys:
        raise SchemaValidationError(f"outputs.{role} has unexpected or missing fields")

    path = _safe_output_path(
        directory,
        entry.get("filename"),
        field=f"outputs.{role}.filename",
    )
    record_count = entry.get("record_count")
    byte_size = entry.get("byte_size")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 0:
        raise SchemaValidationError(f"outputs.{role}.record_count is invalid")
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
        raise SchemaValidationError(f"outputs.{role}.byte_size is invalid")
    expected_hash = entry.get("sha256")
    if not isinstance(expected_hash, str) or not expected_hash.startswith("sha256:"):
        raise SchemaValidationError(f"outputs.{role}.sha256 is invalid")

    compression = entry.get("compression", "none")
    if compression not in {"none", "gzip"}:
        raise SchemaValidationError(f"outputs.{role}.compression is invalid")
    shard_index = entry.get("shard_index", 0)
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0:
        raise SchemaValidationError(f"outputs.{role}.shard_index is invalid")

    if not path.is_file() or path.stat().st_size != byte_size:
        raise SchemaValidationError(f"prepared {role} byte size does not match manifest")
    if sha256_file(path) != expected_hash:
        raise SchemaValidationError(f"prepared {role} hash does not match manifest")
    return path, entry, cast(str, compression), shard_index


def _output_paths(
    directory: Path,
    outputs: Mapping[str, object],
    role: str,
) -> tuple[tuple[Path, Mapping[str, object], str, int], ...]:
    raw_entries = outputs.get(role)
    entries: tuple[_ResolvedOutput, ...]
    if isinstance(raw_entries, Mapping):
        entries = (
            _validate_output_entry(
                directory,
                cast(Mapping[str, object], raw_entries),
                role=role,
                sharded=False,
            ),
        )
    elif isinstance(raw_entries, list) and raw_entries:
        entries = tuple(
            _validate_output_entry(
                directory,
                _mapping(raw_entry, f"outputs.{role}[{index}]"),
                role=role,
                sharded=True,
            )
            for index, raw_entry in enumerate(raw_entries)
        )
    else:
        raise SchemaValidationError(
            f"outputs.{role} must be one file entry or a non-empty shard array"
        )

    filenames = [cast(str, entry[1]["filename"]) for entry in entries]
    if len(filenames) != len(set(filenames)):
        raise SchemaValidationError(f"outputs.{role} filenames must be unique")
    ordered = tuple(sorted(entries, key=lambda entry: entry[3]))
    if [entry[3] for entry in ordered] != list(range(len(ordered))):
        raise SchemaValidationError(f"outputs.{role} shard indexes must be contiguous from zero")
    return ordered


def _iter_jsonl(
    path: Path,
    *,
    name: str,
    parse: Callable[[str], _RecordT],
    compression: str = "none",
) -> Iterator[_RecordT]:
    open_stream = gzip.open if compression == "gzip" else Path.open
    try:
        with open_stream(path, mode="rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise SchemaValidationError(f"blank {name} record at line {line_number}")
                try:
                    yield parse(line)
                except (SchemaValidationError, json.JSONDecodeError) as exc:
                    raise SchemaValidationError(
                        f"invalid {name} record at line {line_number}: {exc}"
                    ) from exc
    except (OSError, UnicodeError) as exc:
        raise SchemaValidationError(f"invalid {name} stream: {exc}") from exc


def _read_jsonl(
    path: Path,
    *,
    name: str,
    parse: Callable[[str], _RecordT],
    compression: str = "none",
) -> tuple[_RecordT, ...]:
    return tuple(_iter_jsonl(path, name=name, parse=parse, compression=compression))


@dataclass(frozen=True, slots=True)
class _PreparedTrialDocuments(FrozenTrialDocuments):
    """Immutable trial sequence streamed from verified prepared-package shards."""

    _shards: tuple[_TrialShard, ...]

    def __len__(self) -> int:
        return sum(record_count for _, record_count, _, _ in self._shards)

    def __iter__(self) -> Iterator[TrialDocument]:
        for path, record_count, compression, shard_index in self._shards:
            count = 0
            for trial in _iter_jsonl(
                path,
                name=f"trial shard {shard_index}",
                parse=TrialDocument.from_json,
                compression=compression,
            ):
                count += 1
                yield trial
            if count != record_count:
                raise SchemaValidationError(
                    f"prepared trial shard {shard_index} count does not match manifest"
                )

    @overload
    def __getitem__(self, index: int) -> TrialDocument: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TrialDocument, ...]: ...

    def __getitem__(self, index: int | slice) -> TrialDocument | tuple[TrialDocument, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step > 0:
                return tuple(islice(self, start, stop, step))
            return tuple(self)[index]
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("trial index out of range")
        for position, trial in enumerate(self):
            if position == normalized:
                return trial
        raise IndexError("trial index out of range")


def _read_output_rows(
    entries: tuple[tuple[Path, Mapping[str, object], str, int], ...],
    *,
    name: str,
    parse: Callable[[str], _RecordT],
) -> tuple[_RecordT, ...]:
    rows: list[_RecordT] = []
    for path, entry, compression, shard_index in entries:
        shard_rows = _read_jsonl(
            path,
            name=f"{name} shard {shard_index}",
            parse=parse,
            compression=compression,
        )
        if len(shard_rows) != entry["record_count"]:
            raise SchemaValidationError(
                f"prepared {name} shard {shard_index} count does not match manifest"
            )
        rows.extend(shard_rows)
    return tuple(rows)


def _read_derived_view_records(
    entries: tuple[tuple[Path, Mapping[str, object], str, int], ...],
    *,
    mapping_key: str,
    view_name: str,
) -> dict[str, JsonValue]:
    records: dict[str, JsonValue] = {}
    previous_key: str | None = None
    for path, entry, compression, shard_index in entries:
        shard_count = 0
        open_stream = gzip.open if compression == "gzip" else Path.open
        try:
            with open_stream(path, mode="rt", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        raise SchemaValidationError(
                            f"blank Derived View {view_name} record at "
                            f"shard {shard_index} line {line_number}"
                        )
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise SchemaValidationError(
                            f"Derived View {view_name} records must be objects"
                        )
                    key = payload.get(mapping_key)
                    if not isinstance(key, str) or not key:
                        raise SchemaValidationError(
                            f"Derived View {view_name} record mapping key is invalid"
                        )
                    if previous_key is not None and key <= previous_key:
                        raise SchemaValidationError(
                            f"Derived View {view_name} records are not in identity order"
                        )
                    previous_key = key
                    records[key] = cast(JsonValue, payload)
                    shard_count += 1
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SchemaValidationError(f"invalid Derived View {view_name} stream: {exc}") from exc
        if shard_count != entry["record_count"]:
            raise SchemaValidationError(
                f"Derived View {view_name} shard {shard_index} count does not match manifest"
            )
    return records


def _load_manifest_derived_views(
    *,
    raw_views: list[object],
    resolved_outputs: Mapping[str, tuple[tuple[Path, Mapping[str, object], str, int], ...]],
) -> tuple[DerivedView, ...]:
    views: list[DerivedView] = []
    referenced_outputs: set[str] = set()
    for index, raw_view in enumerate(raw_views):
        if not isinstance(raw_view, Mapping):
            raise SchemaValidationError("Snapshot Derived Views must contain objects")
        descriptor = cast(Mapping[str, object], raw_view)
        reference = descriptor.get("content_reference")
        if reference is None:
            views.append(DerivedView.from_dict(descriptor))
            continue
        if set(descriptor) != _DERIVED_VIEW_REFERENCE_FIELDS:
            raise SchemaValidationError(
                f"Snapshot Derived View reference {index} has unexpected or missing fields"
            )
        if not isinstance(reference, Mapping) or set(reference) != (
            _DERIVED_VIEW_CONTENT_REFERENCE_FIELDS
        ):
            raise SchemaValidationError(
                f"Snapshot Derived View reference {index} content reference is invalid"
            )
        if reference.get("format") != "jsonl-object-map-v1":
            raise SchemaValidationError("Snapshot Derived View storage format is unsupported")
        output_name = reference.get("output_name")
        content_schema_version = reference.get("content_schema_version")
        mapping_key = reference.get("mapping_key")
        if (
            not isinstance(output_name, str)
            or output_name not in resolved_outputs
            or not isinstance(content_schema_version, str)
            or not content_schema_version
            or not isinstance(mapping_key, str)
            or not mapping_key
        ):
            raise SchemaValidationError("Snapshot Derived View content reference is invalid")
        if output_name in referenced_outputs:
            raise SchemaValidationError("Snapshot Derived View output is referenced more than once")
        referenced_outputs.add(output_name)
        records = _read_derived_view_records(
            resolved_outputs[output_name],
            mapping_key=mapping_key,
            view_name=cast(str, descriptor.get("name")),
        )
        configuration = descriptor.get("configuration")
        if not isinstance(configuration, Mapping):
            raise SchemaValidationError("Snapshot Derived View configuration must be an object")
        view = DerivedView(
            name=cast(str, descriptor.get("name")),
            version=cast(str, descriptor.get("version")),
            input_snapshot_id=cast(str, descriptor.get("input_snapshot_id")),
            configuration=cast(Mapping[str, JsonValue], configuration),
            content={
                "schema_version": content_schema_version,
                "trials": records,
            },
        )
        if descriptor.get("schema_version") != view.schema_version:
            raise SchemaValidationError("Snapshot Derived View schema version is unsupported")
        if descriptor.get("view_id") != view.view_id:
            raise SchemaValidationError("Snapshot Derived View identity does not match its content")
        views.append(view)
    if referenced_outputs != set(resolved_outputs):
        raise SchemaValidationError("prepared Derived View outputs are not referenced exactly once")
    return tuple(views)


def _parse_judgment(serialized: str) -> RelevanceJudgment:
    return RelevanceJudgment.from_json(serialized)


def _validate_relations(
    snapshot: BenchmarkSnapshot,
    evaluation_package: EvaluationPackage,
    source_bundle: SourceBundle,
    source_recipe: Mapping[str, object],
    diagnostics: tuple[ConnectorDiagnostic, ...],
) -> None:
    system_input_artifact_ids = frozenset(_system_input_artifact_ids(source_recipe))
    allowed_artifact_ids = frozenset(_allowed_artifact_ids(source_recipe))
    topic_ids = {item.topic_id for item in snapshot.topics}
    trial_ids = {item.trial_id for item in snapshot.trials}
    for judgment in evaluation_package.judgments:
        if judgment.topic_id not in topic_ids:
            raise SchemaValidationError(
                f"Evaluation Package references unknown topic {judgment.topic_id}"
            )
        if judgment.trial_id not in trial_ids:
            raise SchemaValidationError(
                f"Evaluation Package references unknown trial {judgment.trial_id}"
            )
    for provenance in snapshot.iter_provenance():
        if provenance.artifact_id not in system_input_artifact_ids:
            raise SchemaValidationError(
                "Snapshot provenance must reference only recipe System-input artifacts"
            )
        artifact = source_bundle.artifact_by_id(provenance.artifact_id)
        if provenance.artifact_sha256 != artifact.sha256:
            raise SchemaValidationError("Snapshot provenance digest does not resolve")
    for diagnostic in diagnostics:
        for provenance in diagnostic.provenance:
            if provenance.artifact_id not in allowed_artifact_ids:
                raise SchemaValidationError(
                    "connector diagnostic provenance must reference a recipe allowed artifact"
                )
            artifact = source_bundle.artifact_by_id(provenance.artifact_id)
            if provenance.artifact_sha256 != artifact.sha256:
                raise SchemaValidationError("diagnostic provenance digest does not resolve")
    evaluation_artifact_id = evaluation_package.provenance.get("source_artifact_id")
    evaluation_artifact_sha256 = evaluation_package.provenance.get("source_artifact_sha256")
    if evaluation_artifact_id is not None or evaluation_artifact_sha256 is not None:
        if not isinstance(evaluation_artifact_id, str) or not isinstance(
            evaluation_artifact_sha256, str
        ):
            raise SchemaValidationError(
                "Evaluation Package provenance artifact identity is malformed"
            )
        if evaluation_artifact_id not in allowed_artifact_ids:
            raise SchemaValidationError(
                "Evaluation Package provenance must reference a recipe allowed artifact"
            )
        evaluation_artifact = source_bundle.artifact_by_id(evaluation_artifact_id)
        if evaluation_artifact.sha256 != evaluation_artifact_sha256:
            raise SchemaValidationError("Evaluation Package provenance digest does not resolve")


def _validate_prepared_components(
    *,
    snapshot: BenchmarkSnapshot,
    evaluation_package: EvaluationPackage,
    source_bundle: SourceBundle,
    source_recipe: Mapping[str, object],
    preparation: SnapshotPreparationManifest,
    diagnostics: tuple[ConnectorDiagnostic, ...],
) -> None:
    recipe_id = _required_recipe_string(source_recipe, "recipe_id")
    recipe_hash = content_sha256(source_recipe)
    _validate_source_recipe(
        source_recipe,
        recipe_id=recipe_id,
        benchmark_lineage=snapshot.benchmark_lineage,
        source_bundle=source_bundle,
    )
    expected_preparation = SnapshotPreparationManifest(
        source_recipe_id=recipe_id,
        source_recipe_hash=recipe_hash,
        source_bundle_id=source_bundle.identity_for_artifacts(
            _system_input_artifact_ids(source_recipe)
        ),
    )
    if preparation != expected_preparation:
        raise SchemaValidationError("Snapshot preparation metadata does not match frozen inputs")
    if evaluation_package.snapshot_id != snapshot.snapshot_id:
        raise SchemaValidationError("Evaluation Package is related to another Snapshot")
    if evaluation_package.benchmark_lineage != snapshot.benchmark_lineage:
        raise SchemaValidationError(
            "Evaluation Package benchmark lineage does not match the Snapshot"
        )
    _validate_relations(
        snapshot,
        evaluation_package,
        source_bundle,
        source_recipe,
        diagnostics,
    )


def validate_prepared_benchmark(prepared: PreparedBenchmark) -> PreparedBenchmark:
    """Validate in-memory Prepared Benchmark relations through the package contract seam."""

    if not isinstance(prepared, PreparedBenchmark):
        raise TypeError("prepared must be a PreparedBenchmark")
    prepared_manifest = cast(
        Mapping[str, object], json_value_to_builtins(prepared.prepared_manifest)
    )
    recipe_manifest = _mapping(
        prepared_manifest.get("snapshot_source_recipe"),
        "snapshot_source_recipe",
    )
    recipe = _mapping(recipe_manifest.get("definition"), "recipe definition")
    recipe_id = _required_recipe_string(recipe, "recipe_id")
    recipe_hash = content_sha256(recipe)
    if recipe_manifest.get("recipe_id") != recipe_id:
        raise SchemaValidationError("Snapshot Source Recipe id does not match its manifest")
    if recipe_manifest.get("sha256") != recipe_hash:
        raise SchemaValidationError("Snapshot Source Recipe hash does not match its definition")
    _validate_prepared_components(
        snapshot=prepared.snapshot,
        evaluation_package=prepared.evaluation_package,
        source_bundle=prepared.source_bundle,
        source_recipe=recipe,
        preparation=prepared.preparation,
        diagnostics=prepared.diagnostics,
    )
    return prepared


def load_prepared_benchmark(directory: str | Path) -> PreparedBenchmark:
    """Load and fully validate one Benchmark Snapshot prepared package."""

    prepared_directory = Path(directory)
    manifest_path = prepared_directory / PREPARED_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid prepared manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SchemaValidationError("prepared manifest must be a JSON object")
    if manifest.get("schema_version") != SNAPSHOT_CONTRACT_VERSION:
        raise SchemaValidationError("unsupported prepared manifest schema_version")
    if manifest.get("manifest_type") != "taim-prepared-benchmark":
        raise SchemaValidationError("prepared manifest_type is invalid")
    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise SchemaValidationError("prepared dataset_id must be non-empty")
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str):
        raise SchemaValidationError("prepared created_at must be an ISO 8601 string")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError("prepared created_at must be valid ISO 8601") from exc
    if parsed_created_at.tzinfo is None or parsed_created_at.utcoffset() is None:
        raise SchemaValidationError("prepared created_at must include a UTC offset")
    _mapping(manifest.get("build_provenance"), "build_provenance")

    outputs = _mapping(manifest.get("outputs"), "outputs")
    if set(outputs) != set(PREPARED_OUTPUT_FILENAMES):
        raise SchemaValidationError("prepared outputs must contain all declared roles")
    resolved_outputs = {
        role: _output_paths(prepared_directory, outputs, role) for role in PREPARED_OUTPUT_FILENAMES
    }
    raw_derived_view_outputs = manifest.get("derived_view_outputs", {})
    if not isinstance(raw_derived_view_outputs, Mapping):
        raise SchemaValidationError("derived_view_outputs must be an object")
    resolved_derived_view_outputs = {
        cast(str, name): _output_paths(
            prepared_directory,
            cast(Mapping[str, object], raw_derived_view_outputs),
            cast(str, name),
        )
        for name in raw_derived_view_outputs
        if isinstance(name, str) and name
    }
    if len(resolved_derived_view_outputs) != len(raw_derived_view_outputs):
        raise SchemaValidationError("derived_view_outputs names must be non-empty strings")
    output_filenames = [
        cast(str, entry["filename"])
        for entries in (*resolved_outputs.values(), *resolved_derived_view_outputs.values())
        for _, entry, _, _ in entries
    ]
    if len(output_filenames) != len(set(output_filenames)):
        raise SchemaValidationError("prepared output filenames must be globally unique")
    topics = _read_output_rows(
        resolved_outputs["topics"],
        name="topic",
        parse=BenchmarkTopic.from_json,
    )
    trials = _PreparedTrialDocuments(
        tuple(
            (path, cast(int, entry["record_count"]), compression, shard_index)
            for path, entry, compression, shard_index in resolved_outputs["trials"]
        )
    )
    diagnostics = _read_output_rows(
        resolved_outputs["diagnostics"],
        name="diagnostic",
        parse=ConnectorDiagnostic.from_json,
    )
    judgments = _read_output_rows(
        resolved_outputs["judgments"],
        name="judgment",
        parse=_parse_judgment,
    )

    source_bundle = SourceBundle.from_dict(_mapping(manifest.get("source_bundle"), "source_bundle"))
    recipe_manifest = _mapping(manifest.get("snapshot_source_recipe"), "snapshot_source_recipe")
    recipe_definition = _mapping(recipe_manifest.get("definition"), "recipe definition")
    recipe_hash = content_sha256(recipe_definition)
    if recipe_manifest.get("sha256") != recipe_hash:
        raise SchemaValidationError("Snapshot Source Recipe hash does not match its definition")
    recipe_id = recipe_manifest.get("recipe_id")
    if not isinstance(recipe_id, str) or not recipe_id.strip():
        raise SchemaValidationError("Snapshot Source Recipe id is invalid")
    if recipe_id != _required_recipe_string(recipe_definition, "recipe_id"):
        raise SchemaValidationError("Snapshot Source Recipe id does not match its definition")
    snapshot_manifest = _mapping(manifest.get("snapshot"), "snapshot")
    if set(snapshot_manifest) != _SNAPSHOT_MANIFEST_FIELDS:
        raise SchemaValidationError("Snapshot manifest has unexpected or missing fields")
    input_bundle_id = source_bundle.identity_for_artifacts(
        _system_input_artifact_ids(recipe_definition)
    )
    preparation = SnapshotPreparationManifest(
        source_recipe_id=recipe_id,
        source_recipe_hash=recipe_hash,
        source_bundle_id=input_bundle_id,
    )
    serialized_preparation = _mapping(manifest.get("snapshot_preparation"), "snapshot_preparation")
    if serialized_preparation != preparation.to_dict():
        raise SchemaValidationError("Snapshot preparation metadata does not match frozen inputs")

    if snapshot_manifest.get("contract_version") != SNAPSHOT_CONTRACT_VERSION:
        raise SchemaValidationError("unsupported Snapshot contract_version")
    if snapshot_manifest.get("canonicalization_version") != SNAPSHOT_CANONICALIZATION_VERSION:
        raise SchemaValidationError("unsupported Snapshot canonicalization_version")
    raw_capabilities = snapshot_manifest.get("available_capabilities")
    raw_views = snapshot_manifest.get("derived_views")
    if not isinstance(raw_capabilities, list) or not all(
        isinstance(item, str) for item in raw_capabilities
    ):
        raise SchemaValidationError("Snapshot capabilities must be an array of strings")
    if len(raw_capabilities) != len(set(raw_capabilities)):
        raise SchemaValidationError("Snapshot capabilities must be unique")
    if not isinstance(raw_views, list):
        raise SchemaValidationError("Snapshot Derived Views must be an array")
    derived_views = _load_manifest_derived_views(
        raw_views=raw_views,
        resolved_outputs=resolved_derived_view_outputs,
    )
    snapshot = BenchmarkSnapshot(
        benchmark_lineage=cast(str, snapshot_manifest.get("benchmark_lineage")),
        snapshot_name=cast(str, snapshot_manifest.get("snapshot_name")),
        topics=topics,
        trials=trials,
        available_capabilities=frozenset(raw_capabilities),
        derived_views=derived_views,
    )
    if snapshot_manifest.get("snapshot_id") != snapshot.snapshot_id:
        raise SchemaValidationError("Snapshot identity does not match logical content")
    if snapshot_manifest.get("base_snapshot_id") != snapshot.base_snapshot_id:
        raise SchemaValidationError("base Snapshot identity does not match logical content")
    if snapshot_manifest.get("logical_content_hashes") != snapshot.logical_content_hashes():
        raise SchemaValidationError("Snapshot layer hashes do not match logical content")
    if snapshot_manifest.get("record_counts") != {
        "topics": len(topics),
        "trials": len(trials),
    }:
        raise SchemaValidationError("Snapshot record counts do not match logical content")

    evaluation_manifest = _mapping(manifest.get("evaluation_package"), "evaluation_package")
    if evaluation_manifest.get("schema_version") != EVALUATION_PACKAGE_VERSION:
        raise SchemaValidationError("unsupported Evaluation Package schema_version")
    evaluation_provenance = _mapping(evaluation_manifest.get("provenance"), "evaluation provenance")
    judgment_scheme_payload = evaluation_manifest.get("judgment_scheme")
    if judgment_scheme_payload is not None and not isinstance(judgment_scheme_payload, Mapping):
        raise SchemaValidationError("Evaluation Package judgment_scheme must be an object")
    evaluation_package = EvaluationPackage(
        benchmark_lineage=cast(str, evaluation_manifest.get("benchmark_lineage")),
        task_id=cast(str, evaluation_manifest.get("task_id")),
        snapshot_id=cast(str, evaluation_manifest.get("snapshot_id")),
        judgments=judgments,
        provenance=cast(Mapping[str, JsonValue], evaluation_provenance),
        judgment_scheme=(
            JudgmentScheme.from_dict(cast(Mapping[str, object], judgment_scheme_payload))
            if judgment_scheme_payload is not None
            else TREC_CT_JUDGMENT_SCHEME
        ),
    )
    if evaluation_manifest.get("evaluation_package_id") != evaluation_package.evaluation_package_id:
        raise SchemaValidationError("Evaluation Package identity does not match content")
    if evaluation_manifest.get("judgment_count") != len(judgments):
        raise SchemaValidationError("Evaluation Package count does not match content")
    package_checksums = {
        cast(str, entry["filename"]): cast(str, entry["sha256"])
        for entries in (*resolved_outputs.values(), *resolved_derived_view_outputs.values())
        for _, entry, _, _ in entries
    }
    return validate_prepared_benchmark(
        PreparedBenchmark(
            dataset_id=dataset_id,
            directory=prepared_directory.resolve(),
            snapshot=snapshot,
            evaluation_package=evaluation_package,
            source_bundle=source_bundle,
            preparation=preparation,
            diagnostics=diagnostics,
            package_checksums=package_checksums,
            prepared_manifest=manifest,
            prepared_manifest_hash=sha256_file(manifest_path),
        )
    )


__all__ = [
    "PREPARED_MANIFEST_FILENAME",
    "PREPARED_OUTPUT_FILENAMES",
    "PreparationResult",
    "PreparedBenchmark",
    "SnapshotPreparationManifest",
    "load_prepared_benchmark",
    "validate_prepared_benchmark",
    "write_prepared_benchmark",
]
