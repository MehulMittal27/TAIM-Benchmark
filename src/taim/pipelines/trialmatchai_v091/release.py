"""The TrialMatchAI v0.9.1 commands in the public release: stage the pinned models, and verify a
built concept store and write the manifest a run binds.

Both run a module of this package by path in the pinned checkout's own environment
(``uv run --frozen --project <checkout>``, as a run launches the checkout):

- ``trial-benchmark patient-to-trial trialmatchai stage-models`` runs
  :mod:`taim.pipelines.trialmatchai_v091.stage_models` with the System's six model pins;
- ``trial-benchmark patient-to-trial trialmatchai verify-concept-store`` runs
  :mod:`taim.pipelines.trialmatchai_v091.store_manifest` over the store and the source snapshot it
  was built from. It checks them against the snapshot's receipt, which ships with this package byte
  for byte, so the manifest it writes for the bound store is the pinned one.

The commands join the ``trialmatchai`` group beside the corpus commands both families share.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from collections.abc import Callable
from importlib.resources import as_file, files
from pathlib import Path

from taim.pipelines.trialmatchai.corpus import trialmatchai_checkout_configuration
from taim.pipelines.trialmatchai.execution import trialmatchai_command_group
from taim.pipelines.trialmatchai_v091 import stage_models, store_manifest
from taim.pipelines.trialmatchai_v091.system import (
    TRIALMATCHAI_UPSTREAM_COMMIT,
    TRIALMATCHAI_UPSTREAM_MODEL_PINS,
    TRIALMATCHAI_UPSTREAM_REPOSITORY,
)

#: The receipt of the 2026-09-26 source snapshot the bound store was built from. The bound
#: manifest records this file's digest, so it ships unchanged.
TRIALMATCHAI_UPSTREAM_CONCEPT_RECEIPT = (
    "trialmatchai-upstream-v091-open-concept-sources-2026-09-26.json"
)
TRIALMATCHAI_UPSTREAM_CONCEPT_RECEIPT_SHA256 = (
    "sha256:1652b68a06f66e8374e7bfe47d0959e21f2d68b0abf345c2bb79dd7596daecd4"
)


def trialmatchai_upstream_add_patient_to_trial_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add ``stage-models`` and ``verify-concept-store`` to the ``trialmatchai`` group."""

    group = trialmatchai_command_group(commands)
    stage = group.add_parser(
        "stage-models",
        help="download the six models of TrialMatchAI v0.9.1 at their pinned revisions into one "
        "Hugging Face hub cache, in the pinned checkout's environment, and point each refs/main "
        "at its pin",
    )
    stage.add_argument(
        "--checkout",
        type=Path,
        required=True,
        help="the clean pinned TrialMatchAI v0.9.1 checkout, with its environment built",
    )
    stage.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="the Hugging Face hub cache a run names in HF_HUB_CACHE, outside the checkout",
    )
    stage.set_defaults(handler=_stage_models)

    verify = group.add_parser(
        "verify-concept-store",
        help="verify a TrialMatchAI v0.9.1 concept store against its source snapshot's receipt, "
        "in the pinned checkout's environment, and write the manifest a run binds",
    )
    verify.add_argument(
        "--checkout",
        type=Path,
        required=True,
        help="the clean pinned TrialMatchAI v0.9.1 checkout, with its environment built",
    )
    verify.add_argument(
        "--concept-cache",
        type=Path,
        required=True,
        help="the concept cache the store was built from: raw/ with the six source files, and "
        "the converted dict_<domain>.txt files",
    )
    verify.add_argument(
        "--store",
        type=Path,
        required=True,
        help="the concept store, the directory a run names in TRIALMATCHAI_CONCEPT_DB_PATH",
    )
    verify.add_argument(
        "--output",
        type=Path,
        required=True,
        help="file that receives the manifest; a run binds it as concept-store-manifest.json at "
        "the store's root",
    )
    verify.set_defaults(handler=_verify_concept_store)


def _stage_models(args: argparse.Namespace) -> int:
    return stage_trialmatchai_upstream_models(args.checkout, cache=args.cache)


def stage_trialmatchai_upstream_models(
    checkout: Path, *, cache: Path, call: Callable[..., int] = subprocess.call
) -> int:
    """Run the model stager in the verified checkout's environment; return its exit status.

    A gated model (``google/gemma-2-2b-it``) downloads only with an ``HF_TOKEN`` whose account has
    accepted its terms; the variable passes through to the stager.
    """

    cache = cache.resolve()
    pins = [
        f"{repository}@{revision}" for _, repository, revision in TRIALMATCHAI_UPSTREAM_MODEL_PINS
    ]
    return _run_in_checkout(
        checkout,
        stage_models.__file__,
        ["--cache", str(cache), *(token for pin in pins for token in ("--pin", pin))],
        outside=cache,
        what="model cache",
        call=call,
    )


def _verify_concept_store(args: argparse.Namespace) -> int:
    return verify_trialmatchai_upstream_concept_store(
        args.checkout, concept_cache=args.concept_cache, store=args.store, output=args.output
    )


def verify_trialmatchai_upstream_concept_store(
    checkout: Path,
    *,
    concept_cache: Path,
    store: Path,
    output: Path,
    call: Callable[..., int] = subprocess.call,
) -> int:
    """Run the manifest writer in the verified checkout's environment; return its exit status."""

    output = output.resolve()
    resource = files("taim").joinpath(
        "pipelines", "trialmatchai_v091", "data", TRIALMATCHAI_UPSTREAM_CONCEPT_RECEIPT
    )
    actual = f"sha256:{hashlib.sha256(resource.read_bytes()).hexdigest()}"
    if actual != TRIALMATCHAI_UPSTREAM_CONCEPT_RECEIPT_SHA256:
        raise ValueError(
            f"TrialMatchAI v0.9.1 concept receipt expected "
            f"{TRIALMATCHAI_UPSTREAM_CONCEPT_RECEIPT_SHA256}, found {actual}"
        )
    with as_file(resource) as receipt:
        return _run_in_checkout(
            checkout,
            store_manifest.__file__,
            [
                *("--receipt", str(receipt)),
                *("--concept-cache", str(concept_cache.resolve())),
                *("--store", str(store.resolve())),
                *("--output", str(output)),
            ],
            outside=output,
            what="concept-store manifest",
            call=call,
        )


def _run_in_checkout(
    checkout: Path,
    module: str,
    arguments: list[str],
    *,
    outside: Path,
    what: str,
    call: Callable[..., int],
) -> int:
    """Run ``module`` by path in the verified checkout's environment; return its exit status.

    The module is executed isolated, so it imports nothing from this installation. Its output
    streams through. ``outside`` is the path it writes, which must leave the checkout clean.
    """

    from taim.executables import resolve_executable

    checkout = checkout.resolve()
    if outside.is_relative_to(checkout):
        raise ValueError(f"TrialMatchAI {what} must be outside the checkout, which must stay clean")
    trialmatchai_checkout_configuration(
        checkout,
        (),
        expected_commit=TRIALMATCHAI_UPSTREAM_COMMIT,
        expected_repository=TRIALMATCHAI_UPSTREAM_REPOSITORY,
    )
    environment = dict(os.environ)
    # TAIM runs from its own uv environment; the checkout has a separate one.
    environment.pop("VIRTUAL_ENV", None)
    argv = [
        resolve_executable("uv"),
        *("run", "--frozen", "--project", str(checkout), "python", "-I"),
        str(Path(module).resolve()),
        *arguments,
    ]
    return call(argv, cwd=checkout, env=environment)
