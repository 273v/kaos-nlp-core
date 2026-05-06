"""Retrieval subpackage -- protocol, BM25, hybrid retrievers, and reranking.

This package defines the ``Retriever`` protocol, the ``Reranker`` protocol,
and concrete implementations that live in kaos-nlp-core (no ML dependencies):

- ``BM25Retriever``: wraps the existing ``Searcher`` class.
- ``HybridRetriever``: fuses two ``Retriever`` instances via RRF.

The ``EmbeddingRetriever`` (dense retrieval) lives in
``kaos-nlp-transformers`` because it depends on ``EmbeddingModel``.

Concrete ``Reranker`` implementations live in downstream modules:

- ``JudgeReranker`` in ``kaos-llm-core`` (LLM-as-reranker, zero new deps).
- ``CrossEncoderReranker`` in ``kaos-nlp-transformers`` (planned v1.1).
"""

from kaos_nlp_core.retrieval.bm25 import BM25Retriever
from kaos_nlp_core.retrieval.hybrid import HybridRetriever
from kaos_nlp_core.retrieval.protocol import (
    RetrievalResult,
    Retriever,
    corpus_unit_passage_uri,
    search_hit_to_retrieval_result,
)
from kaos_nlp_core.retrieval.reranker import RankedResult, Reranker

__all__ = [
    "BM25Retriever",
    "HybridRetriever",
    "RankedResult",
    "Reranker",
    "RetrievalResult",
    "Retriever",
    "corpus_unit_passage_uri",
    "search_hit_to_retrieval_result",
]
