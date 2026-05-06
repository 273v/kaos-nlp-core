"""HybridRetriever -- Reciprocal Rank Fusion over two retrievers.

Combines sparse (BM25) and dense (embedding) retrieval via Reciprocal
Rank Fusion (RRF).  RRF is the consensus 2024-2026 score fusion method:
it is parameter-free (except the constant *k*) and outperforms linear
score interpolation when the two retrievers produce scores on
incompatible scales (BM25 vs. cosine similarity).

Formula::

    RRF_score(d) = sum over retrievers r of weight_r / (k + rank_r(d))

Reference: Cormack, Clarke & Butt (2009) "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods".
"""

from __future__ import annotations

import asyncio
from typing import Any

from kaos_nlp_core.retrieval.protocol import Retriever
from kaos_nlp_core.search import SearchHit


class HybridRetriever:
    """Fuses two ``Retriever`` instances via Reciprocal Rank Fusion.

    Example::

        hybrid = HybridRetriever(sparse=bm25_retriever, dense=embedding_retriever)
        results = await hybrid.retrieve("statute of limitations defense", top_k=10)
    """

    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        *,
        k: int = 60,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        """
        Args:
            sparse: First retriever (typically BM25).
            dense: Second retriever (typically embedding-based).
            k: RRF smoothing constant. Default 60 is the standard value
                from the original RRF paper.
            sparse_weight: Multiplicative weight for the sparse retriever's
                RRF contribution. Default 1.0 (equal weight).
            dense_weight: Multiplicative weight for the dense retriever's
                RRF contribution. Default 1.0 (equal weight).
        """
        self._sparse = sparse
        self._dense = dense
        self._k = k
        self._sparse_weight = sparse_weight
        self._dense_weight = dense_weight

    async def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[SearchHit]:
        """Retrieve documents by fusing sparse and dense results with RRF.

        Over-retrieves by 3x from each retriever, then fuses and returns
        the top *top_k* results sorted by fused RRF score.

        Sparse and dense retrieval run concurrently via ``asyncio.gather``
        so wall-clock latency is ``max(sparse_time, dense_time)`` instead
        of the sum. With a typical sparse query at ~5 ms and a dense
        query at ~30 ms, this halves end-to-end hybrid latency on
        realistic embedding workloads (audit perf finding #3c).
        """
        fetch_k = top_k * 3
        sparse_hits, dense_hits = await asyncio.gather(
            self._sparse.retrieve(query, top_k=fetch_k, **kwargs),
            self._dense.retrieve(query, top_k=fetch_k, **kwargs),
        )

        scores: dict[int, float] = {}
        hit_map: dict[int, SearchHit] = {}

        for rank, hit in enumerate(sparse_hits):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + self._sparse_weight / (
                self._k + rank + 1
            )
            hit_map[hit.doc_id] = hit

        for rank, hit in enumerate(dense_hits):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + self._dense_weight / (
                self._k + rank + 1
            )
            if hit.doc_id not in hit_map:
                hit_map[hit.doc_id] = hit

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [
            SearchHit(
                doc_id=doc_id,
                score=rrf_score,
                text=hit_map[doc_id].text,
                external_id=hit_map[doc_id].external_id,
                fields=dict(hit_map[doc_id].fields),
                metadata=dict(hit_map[doc_id].metadata),
            )
            for doc_id, rrf_score in ranked
        ]

    @classmethod
    def from_corpus(
        cls,
        corpus: Any,
        *,
        group_by: str | None = None,
        model_id: str | None = None,
        batch_size: int = 32,
        k: int = 60,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        **bm25_kwargs: Any,
    ) -> HybridRetriever:
        """Build a hybrid BM25+embedding retriever from a kaos-ml-core ``Corpus``.

        Constructs both a ``BM25Retriever`` and an ``EmbeddingRetriever``
        from the same Corpus, then fuses them with RRF.

        Args:
            corpus: A ``kaos_ml_core.Corpus`` instance.
            group_by: Optional attribute name on ``CorpusUnit`` to group
                by before indexing (forwarded to both retrievers).
            model_id: Embedding model id (forwarded to EmbeddingRetriever).
            batch_size: Embedding batch size.
            k: RRF smoothing constant.
            sparse_weight: Weight for BM25 in the RRF fusion.
            dense_weight: Weight for embeddings in the RRF fusion.
            **bm25_kwargs: Forwarded to ``BM25Retriever.from_corpus``
                (e.g. ``lexicon``, ``scoring``).

        Example::

            from kaos_ml_core import Corpus
            corpus = Corpus.from_documents([doc1, doc2])
            retriever = HybridRetriever.from_corpus(corpus)
        """
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        sparse = BM25Retriever.from_corpus(corpus, group_by=group_by, **bm25_kwargs)

        try:
            # kaos-nlp-transformers ships in 0.1.0a2 (Wave 3); the lazy
            # import resolves at runtime when the optional sibling is
            # installed. ty can't see it statically.
            from kaos_nlp_transformers.retrieval import EmbeddingRetriever  # ty: ignore[unresolved-import]

            dense = EmbeddingRetriever.from_corpus(
                corpus, group_by=group_by, model_id=model_id, batch_size=batch_size
            )
        except ImportError as exc:
            msg = (
                "HybridRetriever.from_corpus requires kaos-nlp-transformers. "
                "Fix: install kaos-nlp-transformers (or use BM25Retriever.from_corpus)."
            )
            raise ImportError(msg) from exc

        return cls(
            sparse=sparse,
            dense=dense,
            k=k,
            sparse_weight=sparse_weight,
            dense_weight=dense_weight,
        )


__all__ = ["HybridRetriever"]
