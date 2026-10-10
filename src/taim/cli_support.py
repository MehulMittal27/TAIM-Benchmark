"""Argument parsing and JSON input shared by benchmark commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from taim.contracts import require_sha256
from taim.entity_versions import parse_clinical_as_of
from taim.pipeline_extensions import add_argument_once

DEFAULT_RETRIEVAL_DEPTH = 1_000
DEFAULT_METRIC_CUTOFF = 5


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def sha256_identity(value: str) -> str:
    try:
        require_sha256(value, "SHA-256 identity")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def print_payload(payload: object) -> None:
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


def read_json_object(path: Path, role: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {role}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must contain a JSON object")
    return payload


def read_pool_ids(path: Path | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    try:
        if path.suffix.casefold() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("JSON pool file must contain an array")
            values = tuple(payload)
        else:
            values = tuple(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read external-fidelity pool file: {exc}") from exc
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("external-fidelity pool file must contain non-empty trial IDs")
    if len(values) != len(set(values)):
        raise ValueError("external-fidelity pool file contains duplicate trial IDs")
    return cast(tuple[str, ...], values)


def add_release_identity_options(parser: argparse.ArgumentParser, *, rule: str = "") -> None:
    """The three producer inputs that bind a command's output to one frozen release."""

    parser.add_argument(
        "--release-manifest",
        type=Path,
        help="frozen manifest of the running release, given with --package-artifact and "
        "--dependency-lock or not at all" + (f"; {rule}" if rule else ""),
    )
    parser.add_argument(
        "--package-artifact",
        type=Path,
        help="the release's wheel; the installed package must match it file by file",
    )
    parser.add_argument(
        "--dependency-lock",
        type=Path,
        help="public uv.lock whose bytes are bound by the release manifest",
    )


def add_run_output_options(parser: argparse.ArgumentParser, *, top_k: int) -> None:
    parser.add_argument(
        "--run-id",
        help="name of the run and of its directory under --output-dir; omitted, public-<System>-"
        "<UTC time>",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="directory that receives the run directory",
    )
    parser.add_argument(
        "--top-k",
        type=positive_integer,
        default=top_k,
        help="ranking depth per query, the run's budget K; bound into the System Input ID",
    )
    parser.add_argument(
        "--metric-cutoff",
        type=positive_integer,
        default=DEFAULT_METRIC_CUTOFF,
        help="primary metric cutoff, bound into the System Input ID; a Result Bundle admits only "
        "the Profile's cutoffs",
    )


def add_real_input_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory written by data prepare for this Track",
    )
    parser.add_argument(
        "--clinical-as-of",
        type=parse_clinical_as_of,
        required=True,
        help="clinical as-of time, bound into the Task Input ID: another value is another Task "
        "Input",
    )


def add_pool_options(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    """The pool pins of a compute-bounded Profile, which ``data pool inspect`` emits."""

    parser.add_argument(
        "--pool-file",
        type=Path,
        help="trial IDs of the external-fidelity pool, as a JSON array or one ID per line, written "
        "by data external-pool prepare; required on that Profile",
    )
    parser.add_argument(
        "--pool-count",
        type=positive_integer,
        required=required,
        help="pool_count from the Profile's data pool inspect receipt. Judgment-union and "
        "external-fidelity Profiles require it, --pool-ids-sha256 and --pool-receipt-id; "
        "full-corpus Profiles refuse all three",
    )
    parser.add_argument(
        "--pool-ids-sha256",
        type=sha256_identity,
        required=required,
        help="pool_ids_sha256 from the same receipt",
    )
    parser.add_argument(
        "--pool-receipt-id",
        type=sha256_identity,
        required=required,
        help="pool_receipt_id from the same receipt",
    )
    parser.add_argument(
        "--protocol-approval-id",
        type=sha256_identity,
        required=required,
        help="SHA-256 of the frozen protocol governing the run: the external-baseline protocol on "
        "the external-fidelity Profile and for external baseline Systems, the paper protocol "
        "otherwise. Required on judgment-union, external-fidelity and TREC 2022/2023 "
        "confirmation Profiles, refused on any other",
    )


def add_bm25_options(parser: argparse.ArgumentParser) -> None:
    """BM25's two parameters, shared by every command that runs a BM25 System."""

    add_argument_once(
        parser, "--k1", type=float, default=1.2, help="BM25 term-frequency saturation parameter"
    )
    add_argument_once(
        parser, "--b", type=float, default=0.75, help="BM25 document-length normalisation"
    )
