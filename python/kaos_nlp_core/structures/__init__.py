"""Data structures for text processing: vocabularies, inverted index, matrices.

Inverted-index query results (``ScoredDoc``, ``PostingEntry``) are
native PyO3 pyclasses emitted directly from Rust — no Python
dataclass conversion (audit perf finding #1 / P3).
"""

from __future__ import annotations

from kaos_nlp_core._rust.structures import (
    BloomVocabulary,
    FrequencyVocabulary,
    IndexedVocabulary,
    LabeledSpan,
    SetVocabulary,
    SimilarityMatrix,
    SpanIndex,
    SparseTermMatrix,
    _similarity_matrix_unpickle,
)
from kaos_nlp_core._rust.structures import InvertedIndex as _RustInvertedIndex
from kaos_nlp_core.types import PostingEntry, ScoredDoc


def _unpickle_similarity_matrix(state: bytes) -> SimilarityMatrix:
    """Helper for SimilarityMatrix pickle deserialization."""
    return _similarity_matrix_unpickle(state)


class InvertedIndex:
    """Inverted index with typed query results.

    Thin delegate over the Rust ``InvertedIndex``. ``query_bm25``,
    ``query_tf_idf``, and ``get_postings`` already emit typed
    ``ScoredDoc`` / ``PostingEntry`` pyclasses, so no per-result
    conversion happens here.
    """

    def __init__(self) -> None:
        self._inner = _RustInvertedIndex()

    @staticmethod
    def load(path: str) -> InvertedIndex:
        idx = InvertedIndex.__new__(InvertedIndex)
        idx._inner = _RustInvertedIndex.load(path)
        return idx

    @staticmethod
    def build_batch(documents: list[tuple[int, list[str]]]) -> InvertedIndex:
        idx = InvertedIndex.__new__(InvertedIndex)
        idx._inner = _RustInvertedIndex.build_batch(documents)
        return idx

    def save(self, path: str) -> None:
        self._inner.save(path)

    def add(self, term: str, doc_id: int) -> None:
        self._inner.add(term, doc_id)

    def add_document(self, doc_id: int, terms: list[str]) -> None:
        self._inner.add_document(doc_id, terms)

    def doc_freq(self, term: str) -> int:
        return self._inner.doc_freq(term)

    def idf(self, term: str) -> float:
        return self._inner.idf(term)

    def tf_idf(self, term: str, doc_id: int) -> float:
        return self._inner.tf_idf(term, doc_id)

    def tf_idf_weighted(
        self, term: str, doc_id: int, tf_weight: str = "raw", idf_weight: str = "standard"
    ) -> float:
        return self._inner.tf_idf_weighted(term, doc_id, tf_weight, idf_weight)

    def score_tf_idf(
        self,
        terms: list[str],
        doc_id: int,
        tf_weight: str = "raw",
        idf_weight: str = "standard",
    ) -> float:
        return self._inner.score_tf_idf(terms, doc_id, tf_weight, idf_weight)

    def score_bm25(self, terms: list[str], doc_id: int, k1: float = 1.5, b: float = 0.75) -> float:
        return self._inner.score_bm25(terms, doc_id, k1, b)

    def query_bm25(
        self,
        terms: list[str],
        top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[ScoredDoc]:
        return self._inner.query_bm25(terms, top_k, k1, b)

    def query_tf_idf(
        self,
        terms: list[str],
        top_k: int = 10,
        tf_weight: str = "sublinear",
        idf_weight: str = "smooth",
    ) -> list[ScoredDoc]:
        return self._inner.query_tf_idf(terms, top_k, tf_weight, idf_weight)

    def query_bm25_batch(
        self,
        queries: list[list[str]],
        top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[list[ScoredDoc]]:
        """BM25 retrieval over many pre-tokenized queries.

        Runs queries in parallel via Rayon under a single ``py.detach``.
        Each query is a list of expanded query terms. Returns one
        ``list[ScoredDoc]`` per input query, in the same order. Audit
        perf finding #3a / P4.
        """
        return self._inner.query_bm25_batch(queries, top_k, k1, b)

    def query_tf_idf_batch(
        self,
        queries: list[list[str]],
        top_k: int = 10,
        tf_weight: str = "sublinear",
        idf_weight: str = "smooth",
    ) -> list[list[ScoredDoc]]:
        """TF-IDF retrieval over many pre-tokenized queries (parallel)."""
        return self._inner.query_tf_idf_batch(queries, top_k, tf_weight, idf_weight)

    def query_and(self, terms: list[str]) -> list[int]:
        return self._inner.query_and(terms)

    def query_or(self, terms: list[str]) -> list[int]:
        return self._inner.query_or(terms)

    def term_count(self) -> int:
        return self._inner.term_count()

    def doc_count(self) -> int:
        return self._inner.doc_count()

    def doc_length(self, doc_id: int) -> int:
        return self._inner.doc_length(doc_id)

    def avg_doc_length(self) -> float:
        return self._inner.avg_doc_length()

    def get_postings(self, term: str) -> list[PostingEntry] | None:
        return self._inner.get_postings(term)


__all__ = [
    "BloomVocabulary",
    "FrequencyVocabulary",
    "IndexedVocabulary",
    "InvertedIndex",
    "LabeledSpan",
    "PostingEntry",
    "ScoredDoc",
    "SetVocabulary",
    "SimilarityMatrix",
    "SpanIndex",
    "SparseTermMatrix",
]
