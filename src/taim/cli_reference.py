"""The ``trial-benchmark`` command reference, rendered from the command's own argument parser.

``docs/cli-reference.md`` is this module's output and nothing else, so the reference cannot say
anything the parser does not: a test regenerates the document and fails on any difference. Every
command and option must carry help text, and rendering refuses one without it, because a silent row
sends the reader back to prose that nothing checks.

A pipeline that a release may hold back adds options and subcommands. Each of those is wrapped in
the release's pipeline markers naming the Systems that add it, so a release's copy describes exactly
the commands its package accepts. System identifiers are never written as values or defaults: which
Systems a release runs is its selection, which ``trial-benchmark support list`` reports.

Regenerate it from the package root with ``python -m taim.cli_reference > docs/cli-reference.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Collection, Iterator, Mapping, Sequence
from pathlib import Path

from taim.cli_support import positive_integer, sha256_identity
from taim.entity_versions import parse_clinical_as_of
from taim.pipeline_extensions import PipelineExtension, pipeline_extensions
from taim.release_cli import build_parser
from taim.release_support import SYSTEM_TASKS

_PROGRAM = "trial-benchmark"
_INTRODUCTION = f"""\
# Command reference

Every `{_PROGRAM}` command and option, generated from the command's own argument parser:
`{_PROGRAM} <command> --help` prints the same descriptions. Do not edit this page; change the
help text in the code and regenerate it with `python -m taim.cli_reference > docs/cli-reference.md`.
`{_PROGRAM} support list` reports the Systems this release runs.
"""
_VALUE_TYPES: Mapping[object, str] = {
    None: "text",
    Path: "path",
    float: "number",
    positive_integer: "positive integer",
    sha256_identity: "`sha256:` digest",
    parse_clinical_as_of: "ISO 8601 time with offset",
}
_DIRECTION_TASKS = {"patient-to-trial": "patient_to_trial", "trial-to-patient": "trial_to_patient"}
_TABLE_HEADER = "| Option | Value | Default | Description |\n| --- | --- | --- | --- |\n"


def _subcommands(parser: argparse.ArgumentParser) -> list[tuple[str, str, argparse.ArgumentParser]]:
    """A parser's subcommands in declaration order, each with its help text."""

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            helps = {choice.dest: choice.help for choice in action._choices_actions}
            commands = []
            for name, child in action.choices.items():
                summary = helps.get(name)
                if not summary:
                    raise ValueError(f"{child.prog} has no help text")
                commands.append((name, summary, child))
            return commands
    return []


def _options(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [
        action
        for action in parser._actions
        if action.option_strings and not isinstance(action, argparse._HelpAction)
    ]


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _row(action: argparse.Action, *, command: str, systems: Collection[str]) -> str:
    """One option as a table row: flags, value, default and help text."""

    flags = ", ".join(f"`{flag}`" for flag in action.option_strings)
    if not action.help:
        raise ValueError(f"{command} {action.option_strings[0]} has no help text")
    if action.nargs == 0:
        value = "none"
    elif action.metavar is not None:
        value = f"`{action.metavar}`"
    elif action.choices is not None:
        choices = [str(choice) for choice in action.choices]
        if set(choices) & set(systems):
            raise ValueError(
                f"{command} {action.option_strings[0]} lists System choices; give it a metavar"
            )
        value = "one of " + ", ".join(f"`{choice}`" for choice in choices)
    elif action.type in _VALUE_TYPES:
        value = _VALUE_TYPES[action.type]
    else:
        raise ValueError(f"{command} {action.option_strings[0]} has an undescribed value type")
    if action.nargs == "+":
        value += ", one or more"
    if action.required:
        default = "required"
    elif action.nargs == 0 or action.default is None or action.default in systems:
        default = ""
    elif isinstance(action.default, Path):
        default = f"`{action.default.as_posix()}`"
    else:
        default = f"`{action.default}`"
    return f"| {flags} | {value} | {default} | {_cell(action.help)} |"


def _option_owners(
    extensions: Sequence[PipelineExtension], systems: Collection[str]
) -> dict[str, tuple[frozenset[str], str]]:
    """Each run option an extension adds: the Systems that add it and its one rendered row.

    An option several extensions add must render identically from each, or the row a release
    shows would depend on which of them it ships.
    """

    owners: dict[str, tuple[frozenset[str], str]] = {}
    for extension in extensions:
        scratch = argparse.ArgumentParser(add_help=False)
        extension.add_run_options(scratch)
        for action in _options(scratch):
            row = _row(action, command="run option", systems=systems)
            flag = action.option_strings[0]
            known, known_row = owners.get(flag, (frozenset(), row))
            if known_row != row:
                raise ValueError(f"extensions add {flag} with different definitions")
            owners[flag] = (known | frozenset(extension.system_tasks), row)
    return owners


def _command_owners(
    extensions: Sequence[PipelineExtension],
) -> dict[tuple[str, ...], frozenset[str]]:
    """Each patient-to-trial subcommand an extension adds, at any depth, with the Systems that add
    it. Extensions may share a command group and each add their own commands to it."""

    owners: dict[tuple[str, ...], frozenset[str]] = {}
    for extension in extensions:
        if extension.add_patient_to_trial_commands is None:
            continue
        scratch = argparse.ArgumentParser(add_help=False)
        extension.add_patient_to_trial_commands(scratch.add_subparsers())
        pending: list[tuple[tuple[str, ...], argparse.ArgumentParser]] = [((), scratch)]
        while pending:
            path, parser = pending.pop()
            for name, _summary, child in _subcommands(parser):
                command = (*path, name)
                owners[command] = owners.get(command, frozenset()) | frozenset(
                    extension.system_tasks
                )
                pending.append((command, child))
    return owners


def _markers(owners: Collection[str]) -> str:
    return ", ".join(sorted(owners))


def _sections(
    parser: argparse.ArgumentParser,
    core: argparse.ArgumentParser | None,
    path: tuple[str, ...],
    owners: frozenset[str] | None,
    *,
    option_owners: Mapping[str, tuple[frozenset[str], str]],
    command_owners: Mapping[tuple[str, ...], frozenset[str]],
    systems: Collection[str],
    markers: bool,
    extension_path: tuple[str, ...] | None = None,
) -> Iterator[str]:
    """One section per command below ``parser``; ``core`` is the same command without extensions."""

    core_commands = {name: child for name, _summary, child in _subcommands(core)} if core else {}
    for name, summary, child in _subcommands(parser):
        command = " ".join((_PROGRAM, *path, name))
        child_core = core_commands.get(name)
        child_owners = owners
        child_extension_path: tuple[str, ...] | None = None
        if not path and name in _DIRECTION_TASKS:
            task = _DIRECTION_TASKS[name]
            child_owners = frozenset(
                system for system, tasks in SYSTEM_TASKS.items() if task in tasks
            )
        elif core is not None and child_core is None:
            child_extension_path = (name,)
        elif extension_path is not None:
            child_extension_path = (*extension_path, name)
        if child_extension_path is not None:
            if child_extension_path not in command_owners:
                raise ValueError(f"{command} is added by no pipeline extension")
            child_owners = command_owners[child_extension_path]
        lines = []
        if markers and child_owners is not None:
            lines.append(f"<!-- pipeline-section: {_markers(child_owners)} -->")
        lines += [f"{'##' if not path else '###'} `{command}`", "", _cell(summary)]
        options = _options(child)
        if options:
            lines += ["", _TABLE_HEADER.rstrip("\n")]
        elif not _subcommands(child):
            lines += ["", "No options."]
        for action in options:
            row = _row(action, command=command, systems=systems)
            flag = action.option_strings[0]
            if not markers or child_core is None or flag in child_core._option_string_actions:
                lines.append(row)
                continue
            if flag not in option_owners:
                raise ValueError(f"{command} {flag} is added by no pipeline extension")
            row_owners, _extension_row = option_owners[flag]
            lines.append(f"{row} <!-- pipeline-row: {_markers(row_owners)} -->")
        if markers and child_owners is not None:
            lines.append("<!-- /pipeline-section -->")
        yield "\n".join(lines) + "\n"
        yield from _sections(
            child,
            child_core,
            (*path, name),
            child_owners,
            option_owners=option_owners,
            command_owners=command_owners,
            systems=systems,
            markers=markers,
            extension_path=child_extension_path,
        )


def render_cli_reference(
    extensions: Sequence[PipelineExtension] | None = None, *, markers: bool = True
) -> str:
    """The Markdown command reference of ``trial-benchmark`` with ``extensions`` installed.

    Without ``markers`` it is the copy a release publishing exactly ``extensions`` ships.
    """

    extensions = pipeline_extensions() if extensions is None else tuple(extensions)
    systems = frozenset(SYSTEM_TASKS)
    sections = _sections(
        build_parser(extensions),
        build_parser(()),
        (),
        None,
        option_owners=_option_owners(extensions, systems),
        command_owners=_command_owners(extensions),
        systems=systems,
        markers=markers,
    )
    return "\n".join((_INTRODUCTION, *sections))


def main() -> int:
    print(render_cli_reference(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
