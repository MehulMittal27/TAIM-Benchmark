"""Pipelines whose public release code lives in a module of their own.

A pipeline that not every release publishes keeps its command options, its subcommands, its run
dispatch and the release rules for its files in a module of this package, and only that pipeline
owns the module. The release commands find the modules this package ships and never name one, so a
release that holds a pipeline back leaves its module out and nothing else refers to it.

Each module assigns one public name, ``EXTENSION``: a single :class:`PipelineExtension` call that
writes its :data:`LITERAL_FIELDS` as literals. Its other names are private. The release validator
reads those literals from a candidate tree without importing the module.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from taim.artifacts import RunEvaluationExtension
    from taim.release_profiles import ResolvedBenchmark
    from taim.schemas import JsonValue
    from taim.system_contracts import SystemRunRequest, SystemRunResult

SubcommandAdder = Callable[["argparse._SubParsersAction[argparse.ArgumentParser]"], None]
# The fields an extension module writes as literals, so that they can be read without importing it.
LITERAL_FIELDS = frozenset(
    {
        "system_tasks",
        "pipeline_depth",
        "external_baseline",
        "subcommands",
        "protocol_bindings",
        "module_exports",
        "forbidden_classes",
        "forbidden_text",
        "protected_reference_paths",
        "dependency_declarations",
        "ci_jobs",
    }
)


@dataclass(frozen=True, slots=True)
class ComponentRun:
    """What a pipeline assembled from validated component runs receives from the release CLI.

    ``request`` is the System Request before the pipeline binds its own options. The producer
    identities are None unless the command was given a release identity.
    """

    args: argparse.Namespace
    benchmark: ResolvedBenchmark
    request: SystemRunRequest
    profile_id: str
    run_id: str
    producer_release_identity: Mapping[str, str] | None
    producer_dependency_environment: Mapping[str, JsonValue] | None


@dataclass(frozen=True, slots=True)
class PipelineExtension:
    """What one pipeline adds to the public release commands, and the release rules for its files.

    ``protocol_bindings`` are (module, digest constant, document) triples: the constant in the
    module must be the document's SHA-256. ``module_exports`` are the names a module's ``__all__``
    receives from this pipeline, ``forbidden_classes`` the classes none of those modules may define,
    and ``forbidden_text`` text a module must not contain. ``protected_reference_paths`` are the
    files that may name the protected TREC Tracks; each ships whenever this module does, because the
    release refuses a listed file it does not ship. ``dependency_declarations`` are the files that
    declare the pipeline's external models and runtime inputs. ``ci_jobs`` are the public CI jobs
    the pipeline needs: each names the extras one ``uv sync`` must install and the taim modules the
    job must import. ``evaluation_extension`` returns
    the re-evaluation extension for the Profiles in ``evaluation_profiles``.

    ``required_distributions`` are the locked distributions a producer environment must have
    installed to run the pipeline's Systems. ``run_from_components`` runs a System assembled from
    validated component runs in place of ``run_options`` and ``run``; ``stage_depths`` (each stage's
    pipeline depth and ranking depth, in order) and ``validate_component_producers`` are the rules
    its Result Bundles meet.
    """

    system_tasks: Mapping[str, str]
    pipeline_depth: str
    external_baseline: bool
    subcommands: Mapping[tuple[str, str], str]
    run: Callable[[str, SystemRunRequest], SystemRunResult]
    add_run_options: Callable[[argparse.ArgumentParser], None]
    run_options: Callable[[argparse.Namespace, ResolvedBenchmark, str], dict[str, object]]
    add_patient_to_trial_commands: SubcommandAdder | None = None
    protocol_bindings: tuple[tuple[str, str, str], ...] = ()
    module_exports: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    forbidden_classes: tuple[str, ...] = ()
    forbidden_text: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    protected_reference_paths: tuple[str, ...] = ()
    dependency_declarations: tuple[str, ...] = ()
    ci_jobs: tuple[Mapping[str, tuple[str, ...]], ...] = ()
    evaluation_profiles: tuple[str, ...] = ()
    evaluation_extension: Callable[[], RunEvaluationExtension] | None = None
    required_distributions: tuple[str, ...] = ()
    run_from_components: Callable[[ComponentRun], SystemRunResult] | None = None
    stage_depths: Mapping[str, tuple[str, int]] = field(default_factory=dict)
    validate_component_producers: (
        Callable[[Mapping[str, object], Mapping[str, JsonValue]], None] | None
    ) = None

    def __post_init__(self) -> None:
        for name in (
            "system_tasks",
            "subcommands",
            "module_exports",
            "forbidden_text",
            "stage_depths",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


# Each extension module assigns this name once. It is declared here, unassigned, because the name is
# this package's contract and belongs to no one pipeline.
EXTENSION: PipelineExtension


@cache
def pipeline_extensions() -> tuple[PipelineExtension, ...]:
    """Every pipeline extension this installation ships, in module-name order."""

    found: list[PipelineExtension] = []
    for module_info in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        extension = getattr(module, "EXTENSION", None)
        if not isinstance(extension, PipelineExtension):
            raise TypeError(f"pipeline extension module {module.__name__} defines no EXTENSION")
        found.append(extension)
    systems = [system for extension in found for system in extension.system_tasks]
    if len(systems) != len(set(systems)):
        raise ValueError("two pipeline extensions declare the same System")
    return tuple(found)


def extension_for(system_id: str) -> PipelineExtension | None:
    """The shipped extension that runs ``system_id``, or None when core code runs it."""

    for extension in pipeline_extensions():
        if system_id in extension.system_tasks:
            return extension
    return None


__all__ = [
    "LITERAL_FIELDS",
    "ComponentRun",
    "PipelineExtension",
    "SubcommandAdder",
    "extension_for",
    "pipeline_extensions",
]
