"""Tests for :mod:`kaos_nlp_core.similarity` — Rust-backed dense-vector primitives.

These tests assert the Python-side contract: shape / dtype validation,
typed result wrappers (TopKResult / MMRResult), and bit-level
agreement with numpy reference implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from kaos_nlp_core.similarity import (
    MMRResult,
    TopKResult,
    cosine,
    cosine_adjacent,
    cosine_one_to_many,
    l2_normalize_in_place,
    mmr_select,
    top_k_cosine,
)

# ---------------------------------------------------------------------------
# cosine — single pair
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors_is_one(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_anti_parallel_is_minus_one(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = -a
        assert cosine(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_is_zero(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_clipped_to_unit_interval(self) -> None:
        # 256-d vector of repeated values — naive cosine can over/under-shoot
        # the unit interval. Our clip absorbs that.
        v = np.full(256, 0.5773, dtype=np.float32)
        c = cosine(v, v)
        assert -1.0 <= c <= 1.0
        assert c == pytest.approx(1.0, abs=1e-6)

    def test_dim_mismatch_raises(self) -> None:
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="dimension"):
            cosine(a, b)

    def test_empty_raises(self) -> None:
        empty = np.array([], dtype=np.float32)
        v = np.array([1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            cosine(empty, v)


# ---------------------------------------------------------------------------
# cosine_one_to_many — against numpy reference
# ---------------------------------------------------------------------------


def _numpy_one_to_many(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    qn = np.linalg.norm(query)
    mn = np.linalg.norm(matrix, axis=1)
    denom = mn * qn
    denom = np.where(denom == 0.0, 1.0, denom)
    return (matrix @ query) / denom


class TestCosineOneToMany:
    def test_matches_numpy_reference(self) -> None:
        rng = np.random.default_rng(7)
        M = rng.standard_normal((100, 64)).astype(np.float32)
        q = rng.standard_normal(64).astype(np.float32)
        rust = cosine_one_to_many(q, M)
        ref = _numpy_one_to_many(q, M)
        assert rust.shape == (100,)
        assert rust.dtype == np.float32
        assert np.allclose(rust, ref, atol=1e-5)

    def test_zero_norm_row_is_zero(self) -> None:
        M = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        q = np.array([1.0, 0.0], dtype=np.float32)
        out = cosine_one_to_many(q, M)
        assert out[0] == pytest.approx(1.0)
        # Zero-norm row → 0 by convention (not NaN).
        assert out[1] == 0.0

    def test_dim_mismatch_raises(self) -> None:
        M = np.zeros((3, 4), dtype=np.float32)
        q = np.zeros(3, dtype=np.float32)
        with pytest.raises(ValueError):
            cosine_one_to_many(q, M)

    def test_dtype_enforced(self) -> None:
        # f64 inputs must raise (not silently downcast).
        M = np.zeros((3, 4), dtype=np.float64)
        q = np.zeros(4, dtype=np.float32)
        with pytest.raises(TypeError):
            cosine_one_to_many(q, M)


# ---------------------------------------------------------------------------
# cosine_adjacent
# ---------------------------------------------------------------------------


class TestCosineAdjacent:
    def test_basic(self) -> None:
        M = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        out = cosine_adjacent(M)
        assert out.shape == (3,)
        assert out[0] == pytest.approx(1.0, abs=1e-6)
        assert out[1] == pytest.approx(0.0, abs=1e-6)
        assert out[2] == pytest.approx(-1.0, abs=1e-6)

    def test_single_row_returns_empty(self) -> None:
        M = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = cosine_adjacent(M)
        assert out.shape == (0,)

    def test_against_numpy_reference(self) -> None:
        rng = np.random.default_rng(42)
        M = rng.standard_normal((50, 256)).astype(np.float32)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        Mn = M / np.where(norms == 0, 1.0, norms)
        # numpy reference for adjacent cosine
        ref = np.einsum("ij,ij->i", Mn[:-1], Mn[1:])
        rust = cosine_adjacent(M)
        assert np.allclose(rust, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# top_k_cosine — typed result + numpy-reference parity
# ---------------------------------------------------------------------------


class TestTopKCosine:
    def test_returns_typed_result(self) -> None:
        M = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.5]], dtype=np.float32)
        q = np.array([1.0, 0.0], dtype=np.float32)
        r = top_k_cosine(q, M, k=2)
        assert isinstance(r, TopKResult)
        assert len(r) == 2
        assert r.indices.tolist() == [0, 2]

    def test_matches_numpy_argsort_reference(self) -> None:
        rng = np.random.default_rng(13)
        M = rng.standard_normal((200, 128)).astype(np.float32)
        q = rng.standard_normal(128).astype(np.float32)
        sims = _numpy_one_to_many(q, M)
        # numpy reference: argsort descending, then take top 10
        ref_order = np.argsort(-sims, kind="stable")[:10]
        rust = top_k_cosine(q, M, k=10)
        # Score order must agree exactly (no ties expected on random
        # data at this size); scores must agree within f32 tolerance.
        assert rust.indices.tolist() == ref_order.tolist()
        for i, idx in enumerate(rust.indices):
            assert rust.scores[i] == pytest.approx(sims[idx], abs=1e-5)

    def test_k_zero(self) -> None:
        M = np.array([[1.0, 0.0]], dtype=np.float32)
        q = np.array([1.0, 0.0], dtype=np.float32)
        r = top_k_cosine(q, M, k=0)
        assert len(r) == 0
        assert r.indices.dtype == np.uint32

    def test_k_larger_than_rows_caps(self) -> None:
        M = np.eye(3, dtype=np.float32)
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        r = top_k_cosine(q, M, k=100)
        assert len(r) == 3

    def test_tiebreak_by_lower_index(self) -> None:
        # Two identical rows tied for the best score; k=1 picks the
        # lower index.
        M = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        q = np.array([1.0, 0.0], dtype=np.float32)
        r = top_k_cosine(q, M, k=1)
        assert r.indices.tolist() == [0]


# ---------------------------------------------------------------------------
# mmr_select
# ---------------------------------------------------------------------------


class TestMMRSelect:
    def test_returns_typed_result(self) -> None:
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        rel = np.array([0.9, 0.5], dtype=np.float32)
        r = mmr_select(M, rel, k=2)
        assert isinstance(r, MMRResult)
        assert len(r) == 2
        assert list(r) == [(0, pytest.approx(0.45, abs=1e-5)), (1, pytest.approx(0.25, abs=1e-5))]

    def test_lambda_one_collapses_to_relevance_rank(self) -> None:
        M = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        rel = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        r = mmr_select(M, rel, k=3, lambda_=1.0)
        assert r.indices.tolist() == [1, 2, 0]

    def test_diversity_swaps_picks(self) -> None:
        # rows 0 and 1 are identical → near-perfect similarity →
        # with low lambda MMR prefers the orthogonal row 2 over row 1.
        M = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        rel = np.array([0.9, 0.8, 0.5], dtype=np.float32)
        r = mmr_select(M, rel, k=2, lambda_=0.3)
        assert r.indices.tolist() == [0, 2]

    def test_lambda_clamped(self) -> None:
        # Values outside [0, 1] are silently clamped — should not raise.
        M = np.eye(3, dtype=np.float32)
        rel = np.array([0.9, 0.5, 0.1], dtype=np.float32)
        r1 = mmr_select(M, rel, k=3, lambda_=1.5)
        r2 = mmr_select(M, rel, k=3, lambda_=1.0)
        assert r1.indices.tolist() == r2.indices.tolist()


# ---------------------------------------------------------------------------
# l2_normalize_in_place
# ---------------------------------------------------------------------------


class TestL2Normalize:
    def test_unit_vector_unchanged(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        was_normed = l2_normalize_in_place(v)
        assert was_normed is True
        assert np.allclose(v, [1.0, 0.0, 0.0])

    def test_arbitrary_vector_normalized_to_unit(self) -> None:
        v = np.array([3.0, 4.0], dtype=np.float32)
        l2_normalize_in_place(v)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_left_alone(self) -> None:
        v = np.zeros(4, dtype=np.float32)
        was_normed = l2_normalize_in_place(v)
        assert was_normed is False
        assert np.allclose(v, np.zeros(4))


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_top_level_exports_similarity() -> None:
    import kaos_nlp_core

    assert "similarity" in kaos_nlp_core.__all__
    assert hasattr(kaos_nlp_core, "similarity")
