"""Tests for the Retriever protocol, BM25Retriever, and HybridRetriever.

These tests use only kaos-nlp-core components (no ML dependencies).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from kaos_nlp_core.documents import DocumentCollection
from kaos_nlp_core.retrieval import (
    BM25Retriever,
    HybridRetriever,
    RetrievalResult,
    Retriever,
    search_hit_to_retrieval_result,
)
from kaos_nlp_core.retrieval.protocol import RetrievalResult as ProtocolRetrievalResult
from kaos_nlp_core.search import SearchHit

# ---- Fixtures ---------------------------------------------------------------

SAMPLE_RECORDS = [
    {"id": 0, "text": "The contract was breached by the defendant."},
    {"id": 1, "text": "Privacy policy governs the use of cookies and trackers."},
    {"id": 2, "text": "The statute of limitations expired before filing."},
    {"id": 3, "text": "Employment termination requires proper notice and severance."},
    {"id": 4, "text": "Patent claims must be novel and non-obvious inventions."},
]


def build_collection() -> DocumentCollection:
    return DocumentCollection.from_records(SAMPLE_RECORDS)


# ---- Fake retriever for protocol testing ------------------------------------


class FakeRetriever:
    """A minimal retriever for testing the Retriever protocol."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits

    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> list[SearchHit]:
        return self._hits[:top_k]


# ---- Protocol tests ---------------------------------------------------------


class TestRetrieverProtocol:
    def test_fake_retriever_is_retriever(self) -> None:
        fake = FakeRetriever([])
        assert isinstance(fake, Retriever)

    def test_bm25_retriever_is_retriever(self) -> None:
        collection = build_collection()
        bm25 = BM25Retriever.from_collection(collection)
        assert isinstance(bm25, Retriever)

    def test_retrieval_result_defaults(self) -> None:
        r = RetrievalResult(text="hello", score=0.5, doc_id="1")
        assert r.text == "hello"
        assert r.score == 0.5
        assert r.doc_id == "1"
        assert r.metadata == {}
        assert r.char_start is None
        assert r.char_end is None
        assert r.page is None

    def test_retrieval_result_with_provenance(self) -> None:
        r = RetrievalResult(
            text="hello",
            score=0.5,
            doc_id="1",
            metadata={"source": "test"},
            char_start=10,
            char_end=15,
            page=3,
        )
        assert r.char_start == 10
        assert r.char_end == 15
        assert r.page == 3
        assert r.metadata["source"] == "test"

    def test_protocol_retrieval_result_is_same(self) -> None:
        assert RetrievalResult is ProtocolRetrievalResult


# ---- search_hit_to_retrieval_result tests -----------------------------------


class TestSearchHitConversion:
    def test_basic_conversion(self) -> None:
        hit = SearchHit(doc_id=42, score=1.5, text="test text")
        result = search_hit_to_retrieval_result(hit)
        assert result.text == "test text"
        assert result.score == 1.5
        assert result.doc_id == "42"
        assert result.metadata == {}

    def test_conversion_with_metadata(self) -> None:
        hit = SearchHit(
            doc_id=1,
            score=0.8,
            text="test",
            metadata={"char_start": 10, "char_end": 14, "page": 2},
        )
        result = search_hit_to_retrieval_result(hit)
        assert result.char_start == 10
        assert result.char_end == 14
        assert result.page == 2


# ---- BM25Retriever tests ---------------------------------------------------


class TestBM25Retriever:
    @pytest.fixture()
    def retriever(self) -> BM25Retriever:
        return BM25Retriever.from_collection(build_collection())

    @pytest.fixture()
    def retriever_from_docs(self) -> BM25Retriever:
        return BM25Retriever.from_documents(SAMPLE_RECORDS)

    async def test_basic_retrieval(self, retriever: BM25Retriever) -> None:
        results = await retriever.retrieve("contract breach")
        assert len(results) >= 1
        assert isinstance(results[0], SearchHit)
        assert results[0].doc_id == 0
        assert "contract" in results[0].text.lower()

    async def test_top_k_limit(self, retriever: BM25Retriever) -> None:
        results = await retriever.retrieve("the", top_k=2)
        assert len(results) <= 2

    async def test_no_results(self, retriever: BM25Retriever) -> None:
        results = await retriever.retrieve("xyzzy_nonexistent_42")
        assert results == []

    async def test_from_documents_works(self, retriever_from_docs: BM25Retriever) -> None:
        results = await retriever_from_docs.retrieve("patent")
        assert len(results) >= 1
        assert results[0].doc_id == 4

    async def test_searcher_property(self, retriever: BM25Retriever) -> None:
        assert retriever.searcher is not None
        # Verify the underlying searcher still works directly
        direct = retriever.searcher.search("contract", top_k=1)
        assert len(direct) >= 1

    async def test_sorted_by_score(self, retriever: BM25Retriever) -> None:
        results = await retriever.retrieve("the", top_k=5)
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


# ---- HybridRetriever tests ------------------------------------------------


class TestHybridRetriever:
    def _make_hit(self, doc_id: int, score: float, text: str = "") -> SearchHit:
        return SearchHit(doc_id=doc_id, score=score, text=text)

    async def test_basic_fusion(self) -> None:
        sparse_hits = [
            self._make_hit(0, 3.0, "doc zero"),
            self._make_hit(1, 2.0, "doc one"),
            self._make_hit(2, 1.0, "doc two"),
        ]
        dense_hits = [
            self._make_hit(2, 0.9, "doc two"),
            self._make_hit(0, 0.8, "doc zero"),
            self._make_hit(3, 0.7, "doc three"),
        ]
        sparse = FakeRetriever(sparse_hits)
        dense = FakeRetriever(dense_hits)
        hybrid = HybridRetriever(sparse=sparse, dense=dense)

        results = await hybrid.retrieve("test query", top_k=4)
        assert len(results) == 4
        # Doc 0 and Doc 2 appear in both lists, should have highest fused scores
        result_ids = [r.doc_id for r in results]
        assert 0 in result_ids
        assert 2 in result_ids

    async def test_rrf_scores_are_positive(self) -> None:
        sparse = FakeRetriever([self._make_hit(0, 5.0, "a")])
        dense = FakeRetriever([self._make_hit(1, 0.9, "b")])
        hybrid = HybridRetriever(sparse=sparse, dense=dense)

        results = await hybrid.retrieve("test", top_k=5)
        for r in results:
            assert r.score > 0.0

    async def test_top_k_respected(self) -> None:
        sparse_hits = [self._make_hit(i, 10 - i, f"doc {i}") for i in range(10)]
        dense_hits = [self._make_hit(i + 5, 0.9 - i * 0.1, f"doc {i + 5}") for i in range(10)]
        sparse = FakeRetriever(sparse_hits)
        dense = FakeRetriever(dense_hits)
        hybrid = HybridRetriever(sparse=sparse, dense=dense)

        results = await hybrid.retrieve("test", top_k=3)
        assert len(results) == 3

    async def test_sorted_by_fused_score(self) -> None:
        sparse_hits = [
            self._make_hit(0, 3.0, "doc zero"),
            self._make_hit(1, 2.0, "doc one"),
        ]
        dense_hits = [
            self._make_hit(1, 0.9, "doc one"),
            self._make_hit(0, 0.8, "doc zero"),
        ]
        sparse = FakeRetriever(sparse_hits)
        dense = FakeRetriever(dense_hits)
        hybrid = HybridRetriever(sparse=sparse, dense=dense)

        results = await hybrid.retrieve("test", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_custom_weights(self) -> None:
        sparse_hits = [self._make_hit(0, 3.0, "doc zero")]
        dense_hits = [self._make_hit(1, 0.9, "doc one")]
        sparse = FakeRetriever(sparse_hits)
        dense = FakeRetriever(dense_hits)

        # Heavy dense weight should boost the dense-only document
        hybrid_dense = HybridRetriever(
            sparse=sparse, dense=dense, sparse_weight=0.1, dense_weight=10.0
        )
        results_dense = await hybrid_dense.retrieve("test", top_k=2)
        assert results_dense[0].doc_id == 1  # dense doc should be first

        # Heavy sparse weight should boost the sparse-only document
        hybrid_sparse = HybridRetriever(
            sparse=sparse, dense=dense, sparse_weight=10.0, dense_weight=0.1
        )
        results_sparse = await hybrid_sparse.retrieve("test", top_k=2)
        assert results_sparse[0].doc_id == 0  # sparse doc should be first

    async def test_custom_k(self) -> None:
        """Different k values should produce different scores."""
        sparse_hits = [self._make_hit(0, 3.0, "a"), self._make_hit(1, 2.0, "b")]
        dense_hits = [self._make_hit(0, 0.9, "a"), self._make_hit(1, 0.8, "b")]

        hybrid_k10 = HybridRetriever(
            sparse=FakeRetriever(sparse_hits), dense=FakeRetriever(dense_hits), k=10
        )
        hybrid_k100 = HybridRetriever(
            sparse=FakeRetriever(sparse_hits), dense=FakeRetriever(dense_hits), k=100
        )

        results_k10 = await hybrid_k10.retrieve("test", top_k=2)
        results_k100 = await hybrid_k100.retrieve("test", top_k=2)

        # Same ranking but different absolute scores
        assert results_k10[0].doc_id == results_k100[0].doc_id
        assert results_k10[0].score != results_k100[0].score

    async def test_empty_retrievers(self) -> None:
        sparse = FakeRetriever([])
        dense = FakeRetriever([])
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        results = await hybrid.retrieve("test")
        assert results == []

    async def test_metadata_preserved(self) -> None:
        hit = SearchHit(
            doc_id=0,
            score=1.0,
            text="test",
            external_id="ext-0",
            metadata={"page": 5},
        )
        sparse = FakeRetriever([hit])
        dense = FakeRetriever([])
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        results = await hybrid.retrieve("test", top_k=1)
        assert results[0].external_id == "ext-0"
        assert results[0].metadata["page"] == 5

    async def test_hybrid_is_retriever(self) -> None:
        hybrid = HybridRetriever(
            sparse=FakeRetriever([]),
            dense=FakeRetriever([]),
        )
        assert isinstance(hybrid, Retriever)

    async def test_sparse_empty_dense_has_results(self) -> None:
        """When sparse returns nothing, dense results still surface."""
        sparse = FakeRetriever([])
        dense_hits = [
            SearchHit(doc_id=0, score=0.9, text="doc zero"),
            SearchHit(doc_id=1, score=0.8, text="doc one"),
        ]
        dense = FakeRetriever(dense_hits)
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        results = await hybrid.retrieve("test", top_k=5)
        assert len(results) == 2
        assert results[0].doc_id in (0, 1)

    async def test_dense_empty_sparse_has_results(self) -> None:
        """When dense returns nothing, sparse results still surface."""
        sparse_hits = [
            SearchHit(doc_id=0, score=5.0, text="doc zero"),
        ]
        sparse = FakeRetriever(sparse_hits)
        dense = FakeRetriever([])
        hybrid = HybridRetriever(sparse=sparse, dense=dense)
        results = await hybrid.retrieve("test", top_k=5)
        assert len(results) == 1
        assert results[0].doc_id == 0


# ---- HybridRetriever concurrency tests (audit perf finding #3c) -----------


class _SlowRetriever:
    """Retriever that sleeps for a fixed duration before returning hits.

    Used to assert that ``HybridRetriever.retrieve`` runs sparse + dense
    concurrently via ``asyncio.gather`` — wall-clock should be ~max(d_s,
    d_d), not d_s + d_d.
    """

    def __init__(self, hits: list[SearchHit], delay_s: float) -> None:
        self._hits = hits
        self._delay_s = delay_s

    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> list[SearchHit]:
        await asyncio.sleep(self._delay_s)
        return self._hits[:top_k]


@pytest.mark.asyncio
class TestHybridRetrieverConcurrency:
    """P1 / audit perf #3c — sparse and dense must run concurrently."""

    async def test_retrieve_runs_sparse_and_dense_concurrently(self) -> None:
        # Each retriever sleeps 100 ms. Sequential await would take ≥200 ms;
        # gather() should finish in ~100 ms plus event-loop overhead.
        delay = 0.1
        sparse = _SlowRetriever([SearchHit(doc_id=0, score=1.0, text="s")], delay)
        dense = _SlowRetriever([SearchHit(doc_id=1, score=0.9, text="d")], delay)
        hybrid = HybridRetriever(sparse=sparse, dense=dense)

        t0 = time.perf_counter()
        results = await hybrid.retrieve("query", top_k=2)
        elapsed = time.perf_counter() - t0

        # Concurrent: ~delay (100 ms). Allow generous slack for slow CI.
        # If we ever regress to sequential await, elapsed would be ≥2*delay.
        assert elapsed < 1.5 * delay, (
            f"HybridRetriever.retrieve took {elapsed:.3f}s for two "
            f"{delay:.3f}s retrievers — sparse and dense are running "
            f"sequentially again. Use asyncio.gather()."
        )
        assert len(results) == 2
