#!/usr/bin/env python3
"""Run both TrialGPT Systems' rows on one target: TrialGPT 1.0, then TrialGPT 2.0.

One command per target, in protocol v14 Section 9 step 7's order: TrialGPT-TAIM-Luna-v1's
preflight and row, then TrialGPT2-TAIM-Luna-v1's row, both on the target's one frozen retrieval.
Each step resumes in place. A row whose run directory exists is complete and is skipped; an
interrupted row continues from its workspace. The preflight is kept once written, and a later
preflight that would write a different contract stops the command, because the interrupted
TrialGPT 1.0 row could not resume under it. The first failing step stops the command; running
the same command again continues from that step.

What must hold before it runs is not done here: both rows' declarations, the receipt, the freeze
and the recomputation, each committed (v14 Section 9 steps 1 to 5), and on TREC 2021 both
Systems' qualification gates. Every step runs this release's own ``trial-benchmark`` commands,
which make the provider calls; nothing else is called.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

TRIALGPT = "TrialGPT-TAIM-Luna-v1"
TRIALGPT2 = "TrialGPT2-TAIM-Luna-v1"
#: Protocol v9 Section 4 freezes both, and protocol v14 keeps them for both Systems.
TOP_K = "20"
GENERATION_WORKERS = "48"

Runner = Callable[[Sequence[str]], int]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--clinical-as-of", required=True)
    parser.add_argument("--pool-file", type=Path)
    parser.add_argument("--pool-count", required=True)
    parser.add_argument("--pool-ids-sha256", required=True)
    parser.add_argument("--pool-receipt-id", required=True)
    parser.add_argument("--protocol-approval-id", required=True)
    parser.add_argument("--frozen-retrieval", type=Path, required=True)
    parser.add_argument("--frozen-retrieval-lock", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--package-artifact", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument(
        "--trialgpt2-checkout",
        type=Path,
        required=True,
        help="a clean checkout of TrialGPT 2.0 at the pinned commit",
    )
    parser.add_argument("--trialgpt-run-id", required=True, help="the declared TrialGPT 1.0 row")
    parser.add_argument("--trialgpt2-run-id", required=True, help="the declared TrialGPT 2.0 row")
    parser.add_argument("--output-dir", type=Path, required=True, help="where run directories go")
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="this target's contract and both rows' workspaces; reuse it to resume",
    )
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands without running any"
    )
    return parser


def _inputs(args: argparse.Namespace) -> list[str]:
    """The target and release options every step takes."""

    inputs = [
        "--track",
        args.track,
        "--profile",
        args.profile,
        "--data-dir",
        str(args.data_dir),
        "--clinical-as-of",
        args.clinical_as_of,
        "--pool-count",
        str(args.pool_count),
        "--pool-ids-sha256",
        args.pool_ids_sha256,
        "--pool-receipt-id",
        args.pool_receipt_id,
        "--protocol-approval-id",
        args.protocol_approval_id,
        "--release-manifest",
        str(args.release_manifest),
        "--package-artifact",
        str(args.package_artifact),
        "--dependency-lock",
        str(args.dependency_lock),
    ]
    if args.pool_file is not None:
        inputs += ["--pool-file", str(args.pool_file)]
    return inputs


def _trial_benchmark(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "taim.release_cli", *arguments]


def preflight_command(args: argparse.Namespace, output: Path) -> list[str]:
    return _trial_benchmark(
        "patient-to-trial",
        "trialgpt",
        "preflight",
        *_inputs(args),
        "--frozen-retrieval",
        str(args.frozen_retrieval),
        "--frozen-retrieval-lock",
        str(args.frozen_retrieval_lock),
        "--generation-workers",
        GENERATION_WORKERS,
        "--codex-executable",
        args.codex_executable,
        "--output",
        str(output),
    )


def _row(args: argparse.Namespace, system: str, run_id: str, *options: str) -> list[str]:
    return _trial_benchmark(
        "patient-to-trial",
        "benchmark",
        "run",
        *_inputs(args),
        "--system",
        system,
        "--trialgpt-frozen-retrieval",
        str(args.frozen_retrieval),
        "--trialgpt-frozen-retrieval-lock",
        str(args.frozen_retrieval_lock),
        "--codex-executable",
        args.codex_executable,
        *options,
        "--top-k",
        TOP_K,
        "--run-id",
        run_id,
        "--output-dir",
        str(args.output_dir),
    )


def trialgpt_row_command(args: argparse.Namespace) -> list[str]:
    return _row(
        args,
        TRIALGPT,
        args.trialgpt_run_id,
        "--trialgpt-publication-contract",
        str(args.work_dir / "trialgpt-publication-contract.json"),
        "--trialgpt-workspace",
        str(args.work_dir / "trialgpt-workspace"),
        "--trialgpt-generation-workers",
        GENERATION_WORKERS,
    )


def trialgpt2_row_command(args: argparse.Namespace) -> list[str]:
    return _row(
        args,
        TRIALGPT2,
        args.trialgpt2_run_id,
        "--trialgpt2-checkout",
        str(args.trialgpt2_checkout),
        "--trialgpt2-workspace",
        str(args.work_dir / "trialgpt2-workspace"),
        "--trialgpt2-generation-workers",
        GENERATION_WORKERS,
    )


def _subprocess(command: Sequence[str]) -> int:
    return subprocess.run(command, check=False).returncode  # noqa: S603 - this release's CLI


def run_target(args: argparse.Namespace, run: Runner = _subprocess) -> int:
    """Run the steps that are not complete, in order; the first failure's status, or 0."""

    def step(name: str, command: Sequence[str]) -> int:
        print(f"== {name}\n{shlex.join(command)}", flush=True)
        return 0 if args.dry_run else run(command)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    contract = args.work_dir / "trialgpt-publication-contract.json"
    if (args.output_dir / args.trialgpt_run_id).exists():
        print(f"== {TRIALGPT} row {args.trialgpt_run_id} is complete; skipped", flush=True)
    else:
        candidate = contract.with_name(contract.name + ".new")
        candidate.unlink(missing_ok=True)
        status = step(f"{TRIALGPT} preflight", preflight_command(args, candidate))
        if status:
            return status
        if not args.dry_run:
            if contract.exists() and json.loads(contract.read_text(encoding="utf-8")) != (
                json.loads(candidate.read_text(encoding="utf-8"))
            ):
                print(
                    f"the preflight contract would change from {contract}; the interrupted "
                    f"{TRIALGPT} row cannot resume under another contract. The new one is "
                    f"{candidate}; resolve the difference before running again",
                    file=sys.stderr,
                )
                return 2
            candidate.replace(contract)
        status = step(f"{TRIALGPT} row {args.trialgpt_run_id}", trialgpt_row_command(args))
        if status:
            return status
    if (args.output_dir / args.trialgpt2_run_id).exists():
        print(f"== {TRIALGPT2} row {args.trialgpt2_run_id} is complete; skipped", flush=True)
        return 0
    return step(f"{TRIALGPT2} row {args.trialgpt2_run_id}", trialgpt2_row_command(args))


def main() -> int:
    return run_target(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
