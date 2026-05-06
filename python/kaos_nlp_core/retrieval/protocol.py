"""Retriever protocol -- unified interface for sparse and dense retrieval.

Defines the ``Retriever`` protocol and the ``RetrievalResult`` return type.
Both the BM25Retriever (wrapping the existing Searcher) and the
EmbeddingRetriever (kaos-nlp-transformers) implement this protocol.
HybridRetriever composes two Retrievers via Reciprocal Rank Fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from kaos_nlp_core.search import SearchHit


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A single retrieval result with provenance metadata.

    This is a richer type than ``SearchHit`` — it carries optional
    character offsets and page numbers for provenance tracking.
    Consumers that only need ``(doc_id, score, text)`` can use the
    fields directly; consumers that need provenance can check
    ``char_start`` / ``char_end`` / ``page``.
    """

    text: str
    """Retrieved text chunk."""

    score: float
    """Relevance score (higher is better). Scale depends on the retriever."""

    doc_id: str
    """Document identifier (string for compatibility with external systems)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Caller-supplied metadata (page number, section ref, etc.)."""

    char_start: int | None = None
    """Character offset start in the source document (optional)."""

    char_end: int | None = None
    """Character offset end in the source document (optional)."""

    page: int | None = None
    """Page number in the source document (optional)."""


@runtime_checkable
class Retriever(Protocol):
    """Any object that can retrieve ranked documents from a query.

    Both the existing BM25 Searcher and the new EmbeddingRetriever
    implement this protocol.  HybridRetriever composes two Retrievers.

    The ``retrieve`` method is async to be consistent with KAOS async
    conventions and to support provider-backed embedding APIs.
    """

    async def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[SearchHit]:
        """Return the *top_k* most relevant hits for *query*."""
        ...


def search_hit_to_retrieval_result(hit: SearchHit) -> RetrievalResult:
    """Convert a ``SearchHit`` to a ``RetrievalResult``.

    Utility for consumers that want the richer ``RetrievalResult`` type
    from a retriever that returns ``SearchHit`` objects.
    """
    return RetrievalResult(
        text=hit.text,
        score=hit.score,
        doc_id=str(hit.doc_id),
        metadata=dict(hit.metadata) if hit.metadata else {},
        char_start=hit.metadata.get("char_start") if hit.metadata else None,
        char_end=hit.metadata.get("char_end") if hit.metadata else None,
        page=hit.metadata.get("page") if hit.metadata else None,
    )


def corpus_unit_passage_uri(unit: Any) -> str:
    """Build a passage-level URI from a CorpusUnit's doc_uri + block_ref.

    Canonical function for passage URI construction. Used by
    BM25Retriever, EmbeddingRetriever, HybridRetriever, and the RAG
    pipeline to produce consistent URIs for verification.

    The URI format is ``{doc_uri}{block_ref}`` when block_ref is present
    and doc_uri does not already contain a ``#`` fragment. Otherwise,
    returns doc_uri unchanged.

    Args:
        unit: Any object with ``doc_uri: str`` and ``block_ref: str``
            attributes (e.g. ``CorpusUnit``).

    Returns:
        A passage-level URI string.
    """
    doc_uri: str = unit.doc_uri
    block_ref: str = unit.block_ref
    if block_ref and "#" not in doc_uri:
        return f"{doc_uri}{block_ref}"
    return doc_uri


__all__ = [
    "RetrievalResult",
    "Retriever",
    "corpus_unit_passage_uri",
    "search_hit_to_retrieval_result",
]
