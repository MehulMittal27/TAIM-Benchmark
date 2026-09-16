"""Retrieval baselines provided by TAIM."""

from taim.baselines.bm25 import (
    BM25Retriever,
    StreamingBM25Index,
    query_vocabulary,
    tokenize,
)
from taim.baselines.dense import BgeM3Encoder, DenseRetriever
from taim.baselines.encoder_registry import DENSE_ENCODER_REGISTRY, EncoderPolicy, EncoderRegistry
from taim.baselines.rrf import fuse_rrf, fuse_rrf_n

__all__ = [
    "DENSE_ENCODER_REGISTRY",
    "BM25Retriever",
    "BgeM3Encoder",
    "DenseRetriever",
    "EncoderPolicy",
    "EncoderRegistry",
    "StreamingBM25Index",
    "fuse_rrf",
    "fuse_rrf_n",
    "query_vocabulary",
    "tokenize",
]
