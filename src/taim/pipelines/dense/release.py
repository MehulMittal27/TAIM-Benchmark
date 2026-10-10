"""Shared CLI integration for selectable dense encoder bindings."""

from pathlib import Path


def add_dense_options(parser) -> None:
    from taim.cli_support import positive_integer
    from taim.pipeline_extensions import add_argument_once

    add_argument_once(
        parser,
        "--index-dir",
        type=Path,
        help="directory where a dense System builds its corpus index, resumes an interrupted build "
        "or reuses a finished one; required by the dense Systems",
    )
    add_argument_once(
        parser, "--device", default="cpu", help="PyTorch device that encodes, such as cpu or cuda"
    )
    add_argument_once(
        parser,
        "--document-batch-size",
        type=positive_integer,
        default=8,
        help="trial documents encoded per batch",
    )
    add_argument_once(
        parser,
        "--query-batch-size",
        type=positive_integer,
        default=8,
        help="queries encoded per batch",
    )
    add_argument_once(
        parser,
        "--rebuild-index",
        action="store_true",
        help="delete the index and build checkpoints in --index-dir and encode the corpus again",
    )


def dense_options(args, benchmark, task_input_id):
    if args.index_dir is None:
        raise ValueError("dense Systems require --index-dir")
    return {
        "corpus_hash": benchmark.corpus_hash,
        "device": args.device,
        "document_batch_size": args.document_batch_size,
        "index_dir": args.index_dir,
        "query_batch_size": args.query_batch_size,
        "rebuild_index": args.rebuild_index,
    }


def run_dense_system(system_id, request):
    from taim.pipelines.dense.system import ReleaseDenseSystem
    from taim.release_reference_systems import run_normalized_system

    return run_normalized_system(ReleaseDenseSystem(system_id), request)
