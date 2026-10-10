"""The one command per target: TrialGPT 1.0's preflight and row, then TrialGPT 2.0's row."""

# ruff: noqa: S101
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_trialgpt_target.py"
_spec = importlib.util.spec_from_file_location("_taim_run_trialgpt_target", _SCRIPT)
assert _spec and _spec.loader
_runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _runner
_spec.loader.exec_module(_runner)


def _args(tmp_path: Path):
    return _runner._parser().parse_args(
        [
            "--track",
            "sigir-ct-2016",
            "--profile",
            "sigir-ct-2016-description-judgment-union",
            "--data-dir",
            str(tmp_path / "prepared"),
            "--clinical-as-of",
            "2016-01-01T00:00:00Z",
            "--pool-count",
            "3626",
            "--pool-ids-sha256",
            "sha256:" + "1" * 64,
            "--pool-receipt-id",
            "sha256:" + "2" * 64,
            "--protocol-approval-id",
            "sha256:" + "3" * 64,
            "--frozen-retrieval",
            str(tmp_path / "retrieval.jsonl"),
            "--frozen-retrieval-lock",
            str(tmp_path / "lock.json"),
            "--release-manifest",
            str(tmp_path / "manifest.json"),
            "--package-artifact",
            str(tmp_path / "taim.whl"),
            "--dependency-lock",
            str(tmp_path / "uv.lock"),
            "--trialgpt2-checkout",
            str(tmp_path / "trialgpt2"),
            "--trialgpt-run-id",
            "paper-trialgpt1-sigir-desc-row-1",
            "--trialgpt2-run-id",
            "paper-trialgpt2-sigir-desc-row-1",
            "--output-dir",
            str(tmp_path / "runs"),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )


class _Commands:
    """The release CLI, standing in: it records each command and does what it would write."""

    def __init__(self, *, failing: str | None = None, contract: object = "first") -> None:
        self.commands: list[list[str]] = []
        self.failing = failing
        self.contract = contract

    def __call__(self, command) -> int:
        command = list(command)
        self.commands.append(command)
        if command[3:6] == ["patient-to-trial", "trialgpt", "preflight"]:
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps({"contract": self.contract}))
            return 0
        system = command[command.index("--system") + 1]
        if system == self.failing:
            return 1
        run_directory = (
            Path(command[command.index("--output-dir") + 1])
            / (command[command.index("--run-id") + 1])
        )
        run_directory.mkdir(parents=True)
        return 0

    def steps(self) -> list[str]:
        return [
            "preflight" if "preflight" in command else command[command.index("--system") + 1]
            for command in self.commands
        ]


def test_one_command_runs_the_preflight_then_both_rows(tmp_path: Path) -> None:
    args = _args(tmp_path)
    commands = _Commands()
    assert _runner.run_target(args, commands) == 0
    assert commands.steps() == ["preflight", _runner.TRIALGPT, _runner.TRIALGPT2]
    preflight, first, second = commands.commands
    assert preflight[:3] == [sys.executable, "-m", "taim.release_cli"]
    assert first[first.index("--trialgpt-generation-workers") + 1] == "48"
    assert second[second.index("--trialgpt2-generation-workers") + 1] == "48"
    for row in (first, second):
        assert row[row.index("--top-k") + 1] == "20"
        assert row[row.index("--trialgpt-frozen-retrieval-lock") + 1] == str(tmp_path / "lock.json")
    contract = tmp_path / "work" / "trialgpt-publication-contract.json"
    assert first[first.index("--trialgpt-publication-contract") + 1] == str(contract)
    assert json.loads(contract.read_text()) == {"contract": "first"}
    assert second[second.index("--trialgpt2-checkout") + 1] == str(tmp_path / "trialgpt2")
    assert "--trialgpt-publication-contract" not in second


def test_a_failed_step_stops_the_command_and_the_same_command_resumes(tmp_path: Path) -> None:
    args = _args(tmp_path)
    failing = _Commands(failing=_runner.TRIALGPT)
    assert _runner.run_target(args, failing) == 1
    assert failing.steps() == ["preflight", _runner.TRIALGPT]

    resumed = _Commands()
    assert _runner.run_target(args, resumed) == 0
    assert resumed.steps() == ["preflight", _runner.TRIALGPT, _runner.TRIALGPT2]

    again = _Commands(failing=_runner.TRIALGPT2)
    (tmp_path / "runs" / "paper-trialgpt2-sigir-desc-row-1").rmdir()
    assert _runner.run_target(args, again) == 1
    # The complete TrialGPT 1.0 row is skipped, and its preflight with it.
    assert again.steps() == [_runner.TRIALGPT2]
    assert _runner.run_target(args, _Commands()) == 0


def test_a_changed_preflight_contract_stops_before_the_row(tmp_path: Path, capsys) -> None:
    args = _args(tmp_path)
    assert _runner.run_target(args, _Commands(failing=_runner.TRIALGPT)) == 1
    changed = _Commands(contract="second")
    assert _runner.run_target(args, changed) == 2
    assert changed.steps() == ["preflight"]
    assert "cannot resume under another contract" in capsys.readouterr().err
    contract = tmp_path / "work" / "trialgpt-publication-contract.json"
    assert json.loads(contract.read_text()) == {"contract": "first"}


def test_a_dry_run_prints_every_command_and_runs_none(tmp_path: Path, capsys) -> None:
    args = _args(tmp_path)
    args.dry_run = True
    commands = _Commands()
    assert _runner.run_target(args, commands) == 0
    assert commands.commands == []
    printed = capsys.readouterr().out
    assert printed.count("taim.release_cli") == 3


@pytest.mark.parametrize("pool_file", [None, "pool.txt"])
def test_the_pool_file_is_passed_only_when_given(tmp_path: Path, pool_file) -> None:
    args = _args(tmp_path)
    args.pool_file = None if pool_file is None else tmp_path / pool_file
    command = _runner.preflight_command(args, tmp_path / "contract.json")
    assert ("--pool-file" in command) is (pool_file is not None)


def test_the_trialgpt_commands_are_the_release_clis_own(tmp_path: Path) -> None:
    from taim.release_cli import build_parser

    args = _args(tmp_path)
    parser = build_parser()
    preflight = parser.parse_args(_runner.preflight_command(args, tmp_path / "c.json")[3:])
    assert preflight.generation_workers == 48 and preflight.output == tmp_path / "c.json"
    row = parser.parse_args(_runner.trialgpt_row_command(args)[3:])
    assert row.system == _runner.TRIALGPT and row.top_k == 20
    assert row.trialgpt_generation_workers == 48
    second = parser.parse_args(_runner.trialgpt2_row_command(args)[3:])
    assert second.system == _runner.TRIALGPT2 and second.top_k == 20
    assert second.trialgpt2_generation_workers == 48
    assert second.trialgpt2_checkout == tmp_path / "trialgpt2"
