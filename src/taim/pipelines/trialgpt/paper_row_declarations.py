"""The declarations of a target's two TrialGPT paper rows, derived rather than written by hand.

Protocol v12 Section 7 requires each new paper row to commit a declaration before its run starts.
v14 Section 5 requires both TrialGPT Systems' declarations on a target to be committed before the
target's generation starts, each carrying ``amendment``. v15 Section 4 replaces the stratum's
Prepared Snapshot, Task Input, Evaluation Package and pool receipt IDs with those of the target's
complete-eligibility package.

Every field is derived: the release from its verified manifest, wheel and lock, and the stratum
from the package itself. ``scripts/write_trialgpt_row_declarations.py`` derives the stratum, as the
package's builder script does, because pipeline code does not read the release's Profile
inspection. A package that does not reproduce v15 Section 4 is refused, which is v15 Section 9's
step 0 check, so a declaration can only name the identities the rows must record. The TrialGPT 2.0
row registers no comparison (v14 Section 7).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

from taim.artifacts import sha256_file
from taim.benchmark_release import validate_release_package_artifact
from taim.cli_support import DEFAULT_METRIC_CUTOFF
from taim.pipelines.trialgpt.trialgpt2_declaration import TRIALGPT2_TAIM_LUNA_V1
from taim.schemas import JsonValue

PAPER_ROW_DECLARATION_ARTIFACT_TYPE = "taim-paper-row-declaration"
PAPER_ROW_PROTOCOL_ID = "taim-paper-analysis-protocol-2026-09-28-v12"
PAPER_ROW_PROTOCOL_SHA256 = (
    "sha256:3e7160f1484033d3a43cf9892d70a3f7f3fa7daa1168a1d5927631b3ee5dea4c"
)
PAPER_ROW_AMENDMENT = MappingProxyType(
    {
        "protocol_id": "taim-paper-analysis-protocol-2026-10-07-v14",
        "protocol_sha256": (
            "sha256:1fd0cce1d08be4ebb02b6a01de3eb6b2222262b98270c33a93d325d1b1c42848"
        ),
        "statement": "This row consumes its target's one retrieval under this amendment.",
    }
)
#: The budget protocol v9 Section 4 freezes, which ``run_trialgpt_target.py`` gives both rows.
PAPER_ROW_BUDGET_K = 20
#: Both rows make provider calls and no GPU computation.
PAPER_ROW_ENVIRONMENT = "CPU"
#: What a registered comparison is computed over: protocol v1 Section 7.2's effectiveness metrics.
PAPER_ROW_COMPARISON_METRICS = (
    "macro graded nDCG@10",
    "eligible P@10",
    "macro and micro judged eligible recall at K=100, 500, and 1,000",
)
TRIALGPT_PAPER_SYSTEM_ID = "TrialGPT-TAIM-Luna-v1"


@dataclass(frozen=True, slots=True)
class PaperRowStratum:
    """One target's identities: v15 Section 4's, with v12 Section 4's other columns."""

    track: str
    profile: str
    profile_definition_sha256: str
    prepared_snapshot_id: str
    task_input_id: str
    evaluation_package_id: str
    pool_count: int
    pool_ids_sha256: str
    pool_receipt_id: str
    clinical_as_of: str


PAPER_ROW_STRATA = (
    PaperRowStratum(
        track="trec-ct-2021",
        profile="trec-ct-2021-external-fidelity-26149",
        profile_definition_sha256=(
            "sha256:358292a857f071ba3fb39653eeb1368cebb75e265c2e7ee4a79e553a21db0ef9"
        ),
        prepared_snapshot_id=(
            "sha256:db95af606db708b463dc8bf14515e3662b655de6105dd0f31f28fc10e46b6905"
        ),
        task_input_id="sha256:f426d027a92227ec3eea40e5b9ad8ed48422a0c82fbf75147615b31262b340a8",
        evaluation_package_id=(
            "sha256:35fa37ea430b7255d904152dc967326fdcf30b27cd5ef93c02109de0b82cb1db"
        ),
        pool_count=26_149,
        pool_ids_sha256="sha256:fed85fadb5a0e0a42a39ed9cf65984926a13f66d4a2a0a1deb100996ac56ffa7",
        pool_receipt_id="sha256:1a3432e205252e689ff481914be53f0ed0d78e134b9cf4d06876744dd8fe7cb1",
        clinical_as_of="2021-04-27T00:00:00Z",
    ),
    PaperRowStratum(
        track="trec-ct-2022",
        profile="trec-ct-2022-judgment-union",
        profile_definition_sha256=(
            "sha256:3a83d2ddb9cb33c83dd983cf0466858fac80325146d3de471cb468eda6f93a3b"
        ),
        prepared_snapshot_id=(
            "sha256:d95c9e284cb60f9e9ce2d3eb95d377e35a61217656c95c784213699155cd961f"
        ),
        task_input_id="sha256:73e9378c72da535dfa4cf984f7a8f876d5f59a90ca4d965370e39b2723ded536",
        evaluation_package_id=(
            "sha256:a6884d8343b34a7cdb76f90da611319abc032f2e5001c0757bc8c3f25a7f2f31"
        ),
        pool_count=26_585,
        pool_ids_sha256="sha256:c56129596aac032d62644fc851d61af95da77f470a164650d0a4b953f261c553",
        pool_receipt_id="sha256:e65da82ad2178a74e1a4998b63876db95f306bc78c63f783aa8332df8f7d5c6a",
        clinical_as_of="2022-01-01T00:00:00Z",
    ),
    PaperRowStratum(
        track="sigir-ct-2016",
        profile="sigir-ct-2016-summary-judgment-union",
        profile_definition_sha256=(
            "sha256:0ef0aedf49dc34b5e85ce47ff0b1a8d5a94c1d781ca3186f868341336946e743"
        ),
        prepared_snapshot_id=(
            "sha256:de939758bf653505ef83b4efbe9c0a437e9f83a9706238dd4bcb798b8799db70"
        ),
        task_input_id="sha256:546348a7b4700cdf26dba1eff17dc6b5f398d0f086665b9617252601f1244a79",
        evaluation_package_id=(
            "sha256:6a6e7084b9b16de4d45c6bf8b9d2982698084056d07932e9568ecc49869da8ca"
        ),
        pool_count=3_626,
        pool_ids_sha256="sha256:9a106146c9e1a5da49e15cda3dcc44add3c028e6b0d26a55a40bf60813f5ec5a",
        pool_receipt_id="sha256:340cc0636ac0212bfc1ae3e3a7829070ca8ab59e84ba3e37352f98cbc86cfe64",
        clinical_as_of="2015-12-16T00:00:00Z",
    ),
    PaperRowStratum(
        track="sigir-ct-2016",
        profile="sigir-ct-2016-description-judgment-union",
        profile_definition_sha256=(
            "sha256:ccaf5ad32ee527ef2c1831276111e67678819befcd2a73f5f345d242281d4391"
        ),
        prepared_snapshot_id=(
            "sha256:345813701966b573f0c30b89d8110684199ac92b0af90553f2af75c53b48d20c"
        ),
        task_input_id="sha256:30f9353ef7d1155fe24f3f1fedb876d968392607f7e1d5273b0de43c2e3b6c4d",
        evaluation_package_id=(
            "sha256:36899499f7218e005db9f1f6aa808d6ee34049bea00dc33e0e1404421156e095"
        ),
        pool_count=3_626,
        pool_ids_sha256="sha256:9a106146c9e1a5da49e15cda3dcc44add3c028e6b0d26a55a40bf60813f5ec5a",
        pool_receipt_id="sha256:9cba221f7271ffff938d70ffd3b44313f2485e86c5703fa4458807c53bb02a69",
        clinical_as_of="2015-12-16T00:00:00Z",
    ),
)


class PaperRowDeclarationError(ValueError):
    """A declaration cannot be derived, or would name identities the rows must not record."""


def paper_row_stratum(track: str, profile: str) -> PaperRowStratum:
    """The admitted target's identities."""

    for stratum in PAPER_ROW_STRATA:
        if (stratum.track, stratum.profile) == (track, profile):
            return stratum
    raise PaperRowDeclarationError(f"no TrialGPT paper row is declared on {track}/{profile}")


def require_paper_row_stratum(derived: PaperRowStratum) -> None:
    """Refuse a package whose identities are not protocol v15 Section 4's (v15 Section 9)."""

    expected = asdict(paper_row_stratum(derived.track, derived.profile))
    differing = [
        f"{name}: the package has {value}, protocol v15 Section 4 has {expected[name]}"
        for name, value in asdict(derived).items()
        if value != expected[name]
    ]
    if differing:
        raise PaperRowDeclarationError(
            "this package does not reproduce protocol v15 Section 4, so it is not used and no "
            "row is declared on it: " + "; ".join(differing)
        )


def release_declaration_identity(
    release_manifest: Path, package_artifact: Path, dependency_lock: Path
) -> dict[str, JsonValue]:
    """The release fields of v12 Section 7: version, manifest, tree, wheel and lock digest."""

    verified = validate_release_package_artifact(release_manifest, package_artifact)
    manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
    locks = [entry for entry in manifest["files"] if entry["destination"] == "uv.lock"]
    lock_sha256 = sha256_file(dependency_lock)
    if [entry["sha256"] for entry in locks] != [lock_sha256]:
        raise PaperRowDeclarationError(
            f"dependency lock {dependency_lock} ({lock_sha256}) is not the release's uv.lock"
        )
    return {
        "release_version": cast(str, manifest["release_version"]),
        **verified.release_identity(),
        "dependency_lock_sha256": lock_sha256,
    }


def paper_row_declaration(
    *,
    system_id: str,
    run_id: str,
    release: dict[str, JsonValue],
    stratum: PaperRowStratum,
    comparisons: Sequence[str],
    declared_by: str,
    declared_at: datetime,
) -> dict[str, JsonValue]:
    """One row's declaration: v12 Section 7's fields and v14 Section 5's ``amendment``."""

    if not run_id or not declared_by.strip():
        raise PaperRowDeclarationError("a declaration names its run ID and who declares it")
    return {
        "artifact_type": PAPER_ROW_DECLARATION_ARTIFACT_TYPE,
        "schema_version": "1.0",
        "protocol_id": PAPER_ROW_PROTOCOL_ID,
        "protocol_sha256": PAPER_ROW_PROTOCOL_SHA256,
        "run_id": run_id,
        "system_id": system_id,
        "release": release,
        "stratum": {"task": "patient-to-trial", **asdict(stratum)},
        "budget_k": PAPER_ROW_BUDGET_K,
        "metric_cutoff": DEFAULT_METRIC_CUTOFF,
        "environment": PAPER_ROW_ENVIRONMENT,
        "comparisons": [
            {"run_id": other, "metrics": list(PAPER_ROW_COMPARISON_METRICS)}
            for other in comparisons
        ],
        "ordinal": 1,
        "declared_by": declared_by.strip(),
        "declared_at": declared_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "amendment": dict(PAPER_ROW_AMENDMENT),
    }


def write_paper_row_declarations(
    *,
    stratum: PaperRowStratum,
    release: dict[str, JsonValue],
    trialgpt_run_id: str,
    trialgpt2_run_id: str,
    trialgpt_comparisons: Sequence[str],
    declared_by: str,
    output_dir: Path,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    """Write both rows' declarations once, after checking the package against v15 Section 4."""

    require_paper_row_stratum(stratum)
    if trialgpt_run_id == trialgpt2_run_id:
        raise PaperRowDeclarationError("the two rows are two runs, with two run IDs")
    if trialgpt2_run_id in trialgpt_comparisons or trialgpt_run_id in trialgpt_comparisons:
        raise PaperRowDeclarationError(
            f"{TRIALGPT_PAPER_SYSTEM_ID} cannot register a comparison with itself or with the "
            f"{TRIALGPT2_TAIM_LUNA_V1.system_id} row (protocol v14 Section 7)"
        )
    if len(set(trialgpt_comparisons)) != len(trialgpt_comparisons):
        raise PaperRowDeclarationError("each comparison is registered once")
    # Comparable only to itself, TrialGPT 2.0 registers no comparison (v14 Section 7).
    trialgpt2_comparisons = sorted(
        TRIALGPT2_TAIM_LUNA_V1.comparability.comparable_to - {TRIALGPT2_TAIM_LUNA_V1.system_id}
    )
    declared_at = now or datetime.now(UTC)
    rows = (
        (TRIALGPT_PAPER_SYSTEM_ID, trialgpt_run_id, list(trialgpt_comparisons)),
        (TRIALGPT2_TAIM_LUNA_V1.system_id, trialgpt2_run_id, trialgpt2_comparisons),
    )
    declarations = [
        (
            output_dir / f"{run_id}.json",
            paper_row_declaration(
                system_id=system_id,
                run_id=run_id,
                release=release,
                stratum=stratum,
                comparisons=comparisons,
                declared_by=declared_by,
                declared_at=declared_at,
            ),
        )
        for system_id, run_id, comparisons in rows
    ]
    existing = [str(path) for path, _ in declarations if path.exists()]
    if existing:
        raise PaperRowDeclarationError(f"a declaration is written once: {existing} exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, declaration in declarations:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(declaration, indent=2, sort_keys=True) + "\n")
    return declarations[0][0], declarations[1][0]


__all__ = [
    "PAPER_ROW_AMENDMENT",
    "PAPER_ROW_BUDGET_K",
    "PAPER_ROW_COMPARISON_METRICS",
    "PAPER_ROW_DECLARATION_ARTIFACT_TYPE",
    "PAPER_ROW_ENVIRONMENT",
    "PAPER_ROW_PROTOCOL_ID",
    "PAPER_ROW_PROTOCOL_SHA256",
    "PAPER_ROW_STRATA",
    "PaperRowDeclarationError",
    "PaperRowStratum",
    "paper_row_declaration",
    "paper_row_stratum",
    "release_declaration_identity",
    "require_paper_row_stratum",
    "write_paper_row_declarations",
]
