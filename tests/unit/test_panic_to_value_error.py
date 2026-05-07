"""Regression tests for F5: Rust core APIs that previously panicked on
invalid input must now raise `ValueError` at the Python boundary.

Each test exercises the path where the underlying Rust function used to
`assert!`/`assert_eq!`/index out of bounds. After F5 these paths return
`Result` from the core and translate to `PyValueError` in the bindings.
"""

from __future__ import annotations

import pytest

from kaos_nlp_core._rust.hashing import MinHasher, MinHashIndex
from kaos_nlp_core._rust.structures import SimilarityMatrix, SparseTermMatrix


class TestMinHashSignatureJaccard:
    """`MinHashSignature.jaccard` previously panicked on size mismatch."""

    def test_jaccard_size_mismatch_raises(self) -> None:
        sig_64 = MinHasher(num_perm=64).hash_set(["a", "b", "c"])
        sig_128 = MinHasher(num_perm=128).hash_set(["a", "b", "c"])
        with pytest.raises(ValueError, match="signature size mismatch"):
            sig_64.jaccard(sig_128)

    def test_jaccard_same_size_ok(self) -> None:
        h = MinHasher(num_perm=64)
        sig1 = h.hash_set(["a", "b", "c"])
        sig2 = h.hash_set(["a", "b", "c"])
        assert sig1.jaccard(sig2) == 1.0


class TestMinHashIndex:
    """`MinHashIndex.insert` and `query_candidates` previously panicked when
    a signature size did not match `bands * rows`.
    """

    def test_insert_wrong_size_raises(self) -> None:
        index = MinHashIndex.with_threshold(num_perm=128, threshold=0.5)
        wrong = MinHasher(num_perm=64).hash_set(["a", "b"])
        with pytest.raises(ValueError, match="signature size mismatch"):
            index.insert(0, wrong)

    def test_query_candidates_wrong_size_raises(self) -> None:
        index = MinHashIndex.with_threshold(num_perm=128, threshold=0.5)
        # Index left empty is fine; the size check fires first.
        wrong = MinHasher(num_perm=32).hash_set(["x"])
        with pytest.raises(ValueError, match="signature size mismatch"):
            index.query_candidates(wrong)

    def test_query_above_threshold_wrong_size_raises(self) -> None:
        index = MinHashIndex.with_threshold(num_perm=128, threshold=0.5)
        wrong = MinHasher(num_perm=32).hash_set(["x"])
        with pytest.raises(ValueError, match="signature size mismatch"):
            index.query_above_threshold(wrong, 0.5)

    def test_insert_correct_size_ok(self) -> None:
        h = MinHasher(num_perm=128)
        index = MinHashIndex.with_threshold(num_perm=128, threshold=0.5)
        index.insert(0, h.hash_set(["a", "b", "c", "d"]))
        cands = index.query_candidates(h.hash_set(["a", "b", "c", "d"]))
        assert 0 in cands


class TestSimilarityMatrixIndexBounds:
    """`SimilarityMatrix.{get_distance, get_similarity, k_nearest_neighbors}`
    previously panicked via unchecked vector indexing on out-of-range docs.
    """

    def _two_doc_matrix(self) -> SimilarityMatrix:
        # Two simple sparse vectors → matrix with n_docs == 2.
        return SimilarityMatrix.from_sparse_vectors(
            [[(0, 1.0), (1, 0.0)], [(0, 0.0), (1, 1.0)]],
            metric="euclidean",
        )

    def test_get_distance_out_of_range_raises(self) -> None:
        mat = self._two_doc_matrix()
        with pytest.raises(ValueError, match="out of range"):
            mat.get_distance(0, 99)

    def test_get_distance_both_oor_raises(self) -> None:
        mat = self._two_doc_matrix()
        with pytest.raises(ValueError, match="out of range"):
            mat.get_distance(50, 99)

    def test_get_similarity_out_of_range_raises(self) -> None:
        mat = self._two_doc_matrix()
        with pytest.raises(ValueError, match="out of range"):
            mat.get_similarity(99, 0)

    def test_knn_doc_idx_out_of_range_raises(self) -> None:
        mat = self._two_doc_matrix()
        with pytest.raises(ValueError, match="out of range"):
            mat.k_nearest_neighbors(99, 1)

    def test_in_range_still_works(self) -> None:
        mat = self._two_doc_matrix()
        assert mat.get_distance(0, 0) == 0.0
        assert mat.get_distance(0, 1) > 0.0


class TestSparseTermMatrixDocBounds:
    """`SparseTermMatrix.iter_document_terms` previously panicked when the
    document id exceeded `num_docs`.
    """

    def _two_doc_matrix(self) -> SparseTermMatrix:
        m = SparseTermMatrix()
        m.add_document([(0, 1), (1, 2)])
        m.add_document([(0, 3), (2, 1)])
        return m

    def test_get_document_terms_out_of_range_raises(self) -> None:
        m = self._two_doc_matrix()
        with pytest.raises(ValueError, match="out of range"):
            m.get_document_terms(99)

    def test_get_document_terms_in_range_ok(self) -> None:
        m = self._two_doc_matrix()
        terms = m.get_document_terms(0)
        assert sorted(terms) == [(0, 1), (1, 2)]
