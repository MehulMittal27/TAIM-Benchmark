"""Stage the pinned models of the TrialMatchAI v0.9.1 System into one Hugging Face hub cache.

``trial-benchmark patient-to-trial trialmatchai stage-models`` runs this module by path in the
pinned checkout's own environment, which has ``huggingface_hub``, and passes it the System's six
pins. It imports nothing from TAIM.

For each pin it downloads the revision with ``huggingface_hub.snapshot_download`` and then points
the cache's ``refs/main`` at it, because two loaders on the tag's path ignore the revision they are
given (gliner2 1.3.1, and ``vllm_loader._infer_max_ctx_len``) and resolve ``main`` instead. Nothing
is verified here: the System verifies the cache at every run start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, required=True, help="Hugging Face hub cache")
    parser.add_argument(
        "--pin",
        action="append",
        required=True,
        metavar="REPOSITORY@REVISION",
        help="a model repository and the commit it is pinned to; repeat for each model",
    )
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    args.cache.mkdir(parents=True, exist_ok=True)
    for pin in args.pin:
        repository, _, revision = pin.partition("@")
        snapshot = Path(snapshot_download(repository, revision=revision, cache_dir=args.cache))
        if snapshot.name != revision:
            print(f"{repository}: snapshot {snapshot} is not revision {revision}", file=sys.stderr)
            return 1
        refs = args.cache / ("models--" + repository.replace("/", "--")) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision, encoding="utf-8")
        size = sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())
        print(f"{repository}@{revision}: {size} bytes at {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
