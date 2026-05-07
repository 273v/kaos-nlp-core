"""Tests for the Reranker protocol and RankedResult dataclass.

These tests use only kaos-nlp-core components (no LLM dependencies).
"""

from __future__ import annotations

import pytest

from kaos_nlp_core.retrieval import (
    RankedResult,
    Reranker,
    RetrievalResult,
)
from kaos_nlp_core.retrieval.reranker import RankedResult as DirectRankedResult
from kaos_nlp_core.retrieval.reranker import Reranker as DirectReranker

# ── Fake reranker for protocol testing ────────────────────────────────────────


class FakeReranker:
    """A minimal reranker that assigns scores based on keyword overlap."""

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        query_words = set(query.lower().split())
        ranked = []
        for r in results:
            passage_words = set(r.text.lower().split())
            overlap = len(query_words & passage_words)
            score = min(1.0, overlap / max(len(query_words), 1))
            ranked.append(RankedResult(result=r, rerank_score=score))
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


# ── Protocol tests ────────────────────────────────────────────────────────────


class TestRerankerProtocol:
    def test_fake_reranker_is_reranker(self) -> None:
        fake = FakeReranker()
        assert isinstance(fake, Reranker)

    def test_protocol_is_runtime_checkable(self) -> None:
        """Objects that don't have a rerank method are not Rerankers."""

        class NotAReranker:
            pass

        assert not isinstance(NotAReranker(), Reranker)

    def test_exports_from_package(self) -> None:
        """RankedResult and Reranker should be importable from the package."""
        assert RankedResult is DirectRankedResult
        assert Reranker is DirectReranker


# ── RankedResult tests ────────────────────────────────────────────────────────


class TestRankedResult:
    def test_basic_construction(self) -> None:
        r = RetrievalResult(text="hello", score=0.5, doc_id="1")
        ranked = RankedResult(result=r, rerank_score=0.9)
        assert ranked.result is r
        assert ranked.rerank_score == 0.9

    def test_is_frozen(self) -> None:
        r = RetrievalResult(text="hello", score=0.5, doc_id="1")
        ranked = RankedResult(result=r, rerank_score=0.9)
        with pytest.raises(AttributeError):
            ranked.rerank_score = 0.5  # type: ignore[misc]  # ty: ignore[invalid-assignment]

    def test_preserves_retrieval_metadata(self) -> None:
        r = RetrievalResult(
            text="hello",
            score=0.5,
            doc_id="1",
            metadata={"page": 3, "section_ref": "s1"},
            char_start=10,
            char_end=15,
            page=3,
        )
        ranked = RankedResult(result=r, rerank_score=0.8)
        assert ranked.result.metadata["page"] == 3
        assert ranked.result.char_start == 10
        assert ranked.result.char_end == 15
        assert ranked.result.page == 3


# ── FakeReranker functional tests ────────────────────────────────────────────


class TestFakeReranker:
    async def test_rerank_basic(self) -> None:
        results = [
            RetrievalResult(text="unrelated stuff", score=1.0, doc_id="0"),
            RetrievalResult(text="contract breach damages", score=0.5, doc_id="1"),
            RetrievalResult(text="breach of contract law", score=0.3, doc_id="2"),
        ]
        reranker = FakeReranker()
        ranked = await reranker.rerank("contract breach", results)

        assert len(ranked) == 3
        # Results with more query-word overlap should rank higher.
        assert ranked[0].result.doc_id in ("1", "2")
        assert ranked[-1].result.doc_id == "0"

    async def test_rerank_top_k(self) -> None:
        results = [
            RetrievalResult(text=f"doc {i}", score=float(i), doc_id=str(i)) for i in range(10)
        ]
        reranker = FakeReranker()
        ranked = await reranker.rerank("doc", results, top_k=3)
        assert len(ranked) == 3

    async def test_rerank_empty(self) -> None:
        reranker = FakeReranker()
        ranked = await reranker.rerank("query", [])
        assert ranked == []

    async def test_scores_in_range(self) -> None:
        results = [
            RetrievalResult(text="exact match query", score=1.0, doc_id="0"),
            RetrievalResult(text="nothing here", score=0.5, doc_id="1"),
        ]
        reranker = FakeReranker()
        ranked = await reranker.rerank("exact match query", results)
        for r in ranked:
            assert 0.0 <= r.rerank_score <= 1.0

    async def test_sorted_descending(self) -> None:
        results = [
            RetrievalResult(text="a", score=1.0, doc_id="0"),
            RetrievalResult(text="query word match", score=0.5, doc_id="1"),
            RetrievalResult(text="b", score=0.3, doc_id="2"),
        ]
        reranker = FakeReranker()
        ranked = await reranker.rerank("query word", results)
        scores = [r.rerank_score for r in ranked]
        assert scores == sorted(scores, reverse=True)
