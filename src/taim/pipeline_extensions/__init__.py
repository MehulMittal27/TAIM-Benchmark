"""The contract for pipelines whose public release code lives in a module of their own.

A pipeline that not every release publishes keeps its command options, its subcommands, its run
dispatch and the release rules for its files in an extension module inside its family folder, and
only that pipeline owns the module: ``taim.pipelines.<family>.extension`` when the family is one
release unit, ``taim.pipelines.<family>.<unit>_extension`` for each unit otherwise. The release
commands find the extension modules this installation ships and never name one, so a release that
holds a pipeline back leaves its module out and nothing else refers to it.

Each module assigns ``EXTENSION`` once: a single :class:`PipelineExtension` call that writes its
:data:`LITERAL_FIELDS` as literals. The release validator reads those literals from a candidate
tree without importing the module. This package holds only the contract the modules share.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypeVar

import taim.pipelines

if TYPE_CHECKING:
    from taim.artifacts import RunEvaluationExtension
    from taim.patient_to_trial import PatientToTrialRunRequest, PatientToTrialRunResult
    from taim.release_profiles import ResolvedBenchmark
    from taim.schemas import JsonValue
    from taim.trial_to_patient import TrialToPatientRunRequest, TrialToPatientRunResult

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class RankingExecution(Generic[RequestT, ResultT]):
    """A typed direction's execution and optional effective-request projection.

    Preparation is pure: persistence can reproduce the exact request without executing a System.
    Runners that resolve physical inputs themselves retain that resolution inside ``run``.
    """

    run: Callable[[str, RequestT], ResultT]
    prepare_request: Callable[[str, RequestT], RequestT] | None = None

    def prepare(self, system_id: str, request: RequestT) -> RequestT:
        return request if self.prepare_request is None else self.prepare_request(system_id, request)


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
        "default_system",
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
    request: PatientToTrialRunRequest
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
    installed to run the pipeline's Systems. ``default_system`` names one of this unit's Systems
    that can run the synthetic quick start, and offers a CLI default in that System's direction.
    ``run_from_components`` runs a System assembled from
    validated component runs in place of ``run_options`` and ``patient_to_trial.run``;
    ``stage_depths`` records each stage's pipeline depth and ranking depth, in order.
    ``validate_component_producers`` checks the producer identities in its Result Bundles.
    """

    system_tasks: Mapping[str, str]
    pipeline_depth: str
    external_baseline: bool
    subcommands: Mapping[tuple[str, str], str]
    add_run_options: Callable[[argparse.ArgumentParser], None]
    patient_to_trial: RankingExecution[PatientToTrialRunRequest, PatientToTrialRunResult] | None = (
        None
    )
    trial_to_patient: RankingExecution[TrialToPatientRunRequest, TrialToPatientRunResult] | None = (
        None
    )
    run_options: (
        Callable[[argparse.Namespace, ResolvedBenchmark, str], dict[str, object]] | None
    ) = None
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
    default_system: str | None = None
    run_from_components: Callable[[ComponentRun], PatientToTrialRunResult] | None = None
    stage_depths: Mapping[str, tuple[str, int]] = field(default_factory=dict)
    validate_component_producers: (
        Callable[[Mapping[str, object], Mapping[str, JsonValue]], None] | None
    ) = None

    def __post_init__(self) -> None:
        tasks = set(self.system_tasks.values())
        if not tasks or tasks - {"patient_to_trial", "trial_to_patient"}:
            raise ValueError("pipeline must declare recognized ranking tasks")
        if ("patient_to_trial" in tasks) != (self.patient_to_trial is not None):
            raise ValueError("patient-to-trial declaration and execution must agree")
        if ("trial_to_patient" in tasks) != (self.trial_to_patient is not None):
            raise ValueError("trial-to-patient declaration and execution must agree")
        if self.patient_to_trial is None and self.run_from_components is not None:
            raise ValueError("component runs require patient-to-trial execution")
        if self.default_system is not None and self.default_system not in self.system_tasks:
            raise ValueError("pipeline default_system must name one of its declared Systems")
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


def _is_extension_module(name: str) -> bool:
    return name == "extension" or name.endswith("_extension")


@cache
def pipeline_extensions() -> tuple[PipelineExtension, ...]:
    """Every pipeline extension this installation ships, in family and then module-name order."""

    names: list[str] = []
    for family in sorted(pkgutil.iter_modules(taim.pipelines.__path__), key=lambda item: item.name):
        if not family.ispkg:
            continue
        package = importlib.import_module(f"taim.pipelines.{family.name}")
        names.extend(
            f"{package.__name__}.{module_info.name}"
            for module_info in sorted(
                pkgutil.iter_modules(package.__path__), key=lambda item: item.name
            )
            if _is_extension_module(module_info.name)
        )
    found: list[PipelineExtension] = []
    for name in names:
        module = importlib.import_module(name)
        extension = getattr(module, "EXTENSION", None)
        if not isinstance(extension, PipelineExtension):
            raise TypeError(f"pipeline extension module {module.__name__} defines no EXTENSION")
        found.append(extension)
    systems = [system for extension in found for system in extension.system_tasks]
    if len(systems) != len(set(systems)):
        raise ValueError("two pipeline extensions declare the same System")
    return tuple(found)


def add_argument_once(parser: argparse.ArgumentParser, *flags: str, **options: object) -> None:
    """Shared options are installed once when several shipped units use them."""

    if not any(flag in parser._option_string_actions for flag in flags):
        parser.add_argument(*flags, **options)  # type: ignore[arg-type]


def extension_for(system_id: str) -> PipelineExtension | None:
    """The shipped extension that runs ``system_id``, or None for an unselected System."""

    for extension in pipeline_extensions():
        if system_id in extension.system_tasks:
            return extension
    return None


__all__ = [
    "LITERAL_FIELDS",
    "ComponentRun",
    "PipelineExtension",
    "RankingExecution",
    "SubcommandAdder",
    "add_argument_once",
    "extension_for",
    "pipeline_extensions",
]
