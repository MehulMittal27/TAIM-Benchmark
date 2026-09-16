"""Pinned, provider-free retrieval backends for the TrialGPT paper producer."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from taim.trialgpt_paper import TrialGPTPaperError, TrialGPTTrialView, word_tokenize
from taim.trialgpt_retrieval_producer import TRIALGPT_RETRIEVAL_BM25_ID

TAIM_CONTROLLED_BM25_VARIANT_ID = TRIALGPT_RETRIEVAL_BM25_ID
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_REVISION = "d05a736da4bb84ee4057b7f7999485be6ed85465"
MEDCPT_QUERY_REVISION = "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc"
MEDCPT_ARTICLE_MAX_LENGTH = 512
MEDCPT_QUERY_MAX_LENGTH = 256


class TaimControlledBM25:
    """TAIM-controlled BM25 variant over the shared source-grounded Snapshot view."""

    def __init__(self, trials: Sequence[TrialGPTTrialView]) -> None:
        self.trials = tuple(trials)
        self.documents = tuple(
            word_tokenize(trial.brief_title) * 3
            + tuple(token for disease in trial.diseases for token in word_tokenize(disease)) * 2
            + word_tokenize(trial.retrieval_text)
            for trial in self.trials
        )
        self.document_term_frequencies = tuple(Counter(document) for document in self.documents)
        self.average_length = (
            sum(len(document) for document in self.documents) / len(self.documents)
            if self.documents
            else 0.0
        )
        self.document_frequency: Counter[str] = Counter()
        for frequencies in self.document_term_frequencies:
            self.document_frequency.update(frequencies.keys())

    def rank(self, condition: str, depth: int) -> list[str]:
        if depth < 1:
            raise TrialGPTPaperError("BM25 depth must be positive")
        query = Counter(word_tokenize(condition))
        scores: list[float] = []
        for document, frequencies in zip(
            self.documents, self.document_term_frequencies, strict=True
        ):
            score = 0.0
            for term, query_frequency in query.items():
                document_frequency = self.document_frequency.get(term, 0)
                if not document_frequency:
                    continue
                idf = math.log(
                    1
                    + (len(self.documents) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_norm = 1.2 * (1 - 0.75 + 0.75 * len(document) / self.average_length)
                frequency = frequencies.get(term, 0)
                score += query_frequency * idf * frequency * 2.2 / (frequency + length_norm)
            scores.append(score)
        order = sorted(range(len(scores)), key=lambda index: -scores[index])
        return [self.trials[index].trial_id for index in order[:depth]]


def controlled_bm25_provenance() -> dict[str, str]:
    """Return the canonical provenance for TAIM's explicitly non-parity BM25 variant."""

    return {
        "implementation": TAIM_CONTROLLED_BM25_VARIANT_ID,
        "upstream_parity": "not_proven",
        "classification": "controlled_taim_variant",
        "supersedes": "taim-controlled-bm25-v1-document-frequency-defect",
        "upstream_reference": "rank_bm25.BM25Okapi over TrialGPT corpus.jsonl",
        "input_view": "shared_taim_complete_eligibility",
    }


class TorchMedCPT:
    """Pinned MedCPT Article/Query encoders with exact inner-product retrieval."""

    model_configuration: Mapping[str, object]

    def __init__(
        self,
        *,
        workspace: Path,
        device: str,
        precision: str,
        model_cache_dir: Path | None = None,
        offline: bool = False,
        batch_size: int = 16,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise TrialGPTPaperError(f"MedCPT device must be cpu or cuda, got {device!r}")
        if precision != "float32":
            raise TrialGPTPaperError("paper-faithful MedCPT currently requires float32")
        if isinstance(batch_size, bool) or batch_size < 1:
            raise TrialGPTPaperError("paper-faithful MedCPT batch_size must be positive")
        self.workspace = workspace
        self.device = device
        self.precision = precision
        self.model_cache_dir = model_cache_dir
        self.batch_size = batch_size
        self.model_configuration = {
            "article_model": MEDCPT_ARTICLE_MODEL,
            "article_revision": MEDCPT_ARTICLE_REVISION,
            "query_model": MEDCPT_QUERY_MODEL,
            "query_revision": MEDCPT_QUERY_REVISION,
            "article_max_length": MEDCPT_ARTICLE_MAX_LENGTH,
            "query_max_length": MEDCPT_QUERY_MAX_LENGTH,
            "pooling": "last_hidden_state[:, 0, :]",
            "normalization": "none",
            "index": "exact_numpy_inner_product",
            "device": device,
            "precision": precision,
            "offline": offline,
            "batch_size": batch_size,
        }
        try:
            import numpy as np
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise TrialGPTPaperError(
                "TrialGPT retrieval production requires the dense optional dependencies"
            ) from exc
        self._np = np
        self._torch = torch
        load_options = {
            "cache_dir": str(model_cache_dir) if model_cache_dir is not None else None,
            "local_files_only": offline,
        }
        self._article_tokenizer = AutoTokenizer.from_pretrained(
            MEDCPT_ARTICLE_MODEL,
            revision=MEDCPT_ARTICLE_REVISION,
            **load_options,
        )
        self._article_model = AutoModel.from_pretrained(
            MEDCPT_ARTICLE_MODEL,
            revision=MEDCPT_ARTICLE_REVISION,
            **load_options,
        ).to(device)
        self._query_tokenizer = AutoTokenizer.from_pretrained(
            MEDCPT_QUERY_MODEL,
            revision=MEDCPT_QUERY_REVISION,
            **load_options,
        )
        self._query_model = AutoModel.from_pretrained(
            MEDCPT_QUERY_MODEL,
            revision=MEDCPT_QUERY_REVISION,
            **load_options,
        ).to(device)
        self._article_model.eval()
        self._query_model.eval()
        article_resolved = getattr(self._article_model.config, "_commit_hash", None)
        query_resolved = getattr(self._query_model.config, "_commit_hash", None)
        if article_resolved != MEDCPT_ARTICLE_REVISION or query_resolved != MEDCPT_QUERY_REVISION:
            raise TrialGPTPaperError(
                "resolved MedCPT model revisions do not match the release pins"
            )
        self.model_configuration = {
            **self.model_configuration,
            "article_resolved_revision": article_resolved,
            "query_resolved_revision": query_resolved,
        }

    def _encode_articles(self, trials: Sequence[TrialGPTTrialView]) -> Any:
        embeddings = []
        for start in range(0, len(trials), self.batch_size):
            batch = trials[start : start + self.batch_size]
            encoded = self._article_tokenizer(
                [[trial.brief_title, trial.retrieval_text] for trial in batch],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=MEDCPT_ARTICLE_MAX_LENGTH,
            ).to(self.device)
            with self._torch.no_grad():
                embeddings.append(
                    self._article_model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
                )
        return self._np.concatenate(embeddings, axis=0)

    @staticmethod
    def _trial_signature(trials: Sequence[TrialGPTTrialView]) -> str:
        payload = [
            {
                "trial_id": trial.trial_id,
                "brief_title": trial.brief_title,
                "retrieval_text": trial.retrieval_text,
            }
            for trial in trials
        ]
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def build_trial_embeddings(self, trials: Sequence[TrialGPTTrialView]) -> Any:
        """Build the article matrix once and reuse it across all topics and query arms."""

        signature = self._trial_signature(trials)
        verified = getattr(self, "_verified_trial_embeddings", None)
        if isinstance(verified, tuple) and len(verified) == 2 and verified[0] == signature:
            return verified[1]
        embeddings = self._encode_articles(trials)
        sha256 = "sha256:" + hashlib.sha256(embeddings.tobytes()).hexdigest()
        self._verified_trial_embeddings = (signature, embeddings)
        self.model_configuration = {
            **self.model_configuration,
            "trial_embeddings": {
                "input_signature": signature,
                "sha256": sha256,
                "trial_count": int(embeddings.shape[0]),
                "embedding_dimension": int(embeddings.shape[1]),
                "dtype": str(embeddings.dtype),
            },
        }
        return embeddings

    def _encode_queries(self, conditions: Sequence[str]) -> Any:
        encoded = self._query_tokenizer(
            list(conditions),
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=MEDCPT_QUERY_MAX_LENGTH,
        ).to(self.device)
        with self._torch.no_grad():
            return self._query_model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()

    def rank(
        self,
        *,
        conditions: Sequence[str],
        trials: Sequence[TrialGPTTrialView],
        depth: int,
    ) -> list[list[str]]:
        if depth < 1:
            raise TrialGPTPaperError("MedCPT depth must be positive")
        articles = self.build_trial_embeddings(trials)
        queries = self._encode_queries(conditions)
        scores = self._np.matmul(queries, articles.T)
        # Stable sort makes equal-score ordering inherit the canonical trial order.
        indexes = self._np.argsort(-scores, axis=1, kind="stable")[:, : min(depth, len(trials))]
        return [[trials[int(index)].trial_id for index in row] for row in indexes]


__all__ = [
    "MEDCPT_ARTICLE_MODEL",
    "MEDCPT_ARTICLE_REVISION",
    "MEDCPT_QUERY_MODEL",
    "MEDCPT_QUERY_REVISION",
    "TAIM_CONTROLLED_BM25_VARIANT_ID",
    "TaimControlledBM25",
    "TorchMedCPT",
    "controlled_bm25_provenance",
]
