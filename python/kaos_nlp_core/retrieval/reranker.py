"""Reranker protocol -- unified interface for reranking retrieval results.

Defines the ``Reranker`` protocol and the ``RankedResult`` return type.
Any reranking strategy (LLM judge, cross-encoder, API-based) implements
this protocol.  The protocol lives in kaos-nlp-core (no LLM dependency)
so that both kaos-llm-core and kaos-nlp-transformers can depend on it
without circular imports.

Follows the same pattern as ``kaos_llm_core.router.Router``:
one protocol, multiple implementations, composable via the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kaos_nlp_core.retrieval.protocol import RetrievalResult


@dataclass(frozen=True, slots=True)
class RankedResult:
    """A retrieval result with an additional reranking score.

    Wraps the original ``RetrievalResult`` and adds a ``rerank_score``
    that represents how relevant the passage is to the query, as judged
    by the reranker.  Scores are normalized to [0.0, 1.0] for
    composability across reranking strategies.
    """

    result: RetrievalResult
    """The original retrieval result, preserved with all metadata."""

    rerank_score: float
    """Reranker relevance score in [0.0, 1.0]. Higher is more relevant."""


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking retrieval results.

    A Reranker takes a query and a list of ``RetrievalResult`` objects
    and returns them rescored and reordered by relevance.  All
    implementations normalize scores to [0.0, 1.0].

    Follows the same pattern as ``kaos_llm_core.router.Router``:
    one protocol with one core method, multiple implementations
    (JudgeReranker, CrossEncoderReranker, APIReranker), composable
    via the same interface.
    """

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """Rerank results by relevance to the query.

        Args:
            query: The search query.
            results: Retrieval results to rerank.
            top_k: Return only the top K results.  ``None`` returns all,
                reordered by rerank score.

        Returns:
            ``list[RankedResult]`` sorted by ``rerank_score`` descending,
            truncated to ``top_k`` if specified.
        """
        ...


__all__ = [
    "RankedResult",
    "Reranker",
]
