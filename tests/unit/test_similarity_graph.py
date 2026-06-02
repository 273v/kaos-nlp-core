"""Unit tests for the dense similarity-graph primitives.

Exercises :func:`kaos_nlp_core.similarity.knn_graph`,
:func:`~kaos_nlp_core.similarity.near_duplicates`, and
:func:`~kaos_nlp_core.similarity.as_contiguous_f32` through the public
wrapper module (not the private ``_rust`` extension).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from kaos_nlp_core.similarity import (
    NO_NEIGHBOR,
    KnnGraph,
    NearDuplicates,
    as_contiguous_f32,
    knn_graph,
    near_duplicates,
)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize a float32 matrix (for the fast-path tests)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


# ---------------------------------------------------------------------------
# knn_graph
# ---------------------------------------------------------------------------


def test_knn_graph_shapes_and_self_exclusion() -> None:
    matrix = np.array(
        [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    g = knn_graph(matrix, k=2)
    assert isinstance(g, KnnGraph)
    assert g.indices.shape == (4, 2)
    assert g.scores.shape == (4, 2)
    assert g.indices.dtype == np.uint32
    assert g.scores.dtype == np.float32
    assert g.n_rows == 4
    assert g.k == 2
    # No row is its own neighbour.
    for i in range(4):
        assert i not in g.indices[i].tolist()
    # Row 0's nearest is row 1 (closest direction).
    assert g.indices[0, 0] == 1


def test_knn_graph_scores_descending_per_row() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((30, 16)).astype(np.float32)
    g = knn_graph(matrix, k=5)
    for row in g.scores:
        assert np.all(np.diff(row) <= 1e-6), f"not descending: {row}"


def test_knn_graph_include_self_top_hit_is_self() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    g = knn_graph(matrix, k=1, include_self=True)
    for i in range(3):
        assert g.indices[i, 0] == i
        assert g.scores[i, 0] == pytest.approx(1.0, abs=1e-5)


def test_knn_graph_k_capped_at_available() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    g = knn_graph(matrix, k=99)  # only 2 neighbours available per row
    assert g.k == 2
    assert g.indices.shape == (3, 2)


def test_knn_graph_single_row_is_empty() -> None:
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    g = knn_graph(matrix, k=5)
    assert g.k == 0
    assert g.indices.shape == (1, 0)
    assert g.edges().shape == (0, 2)


def test_knn_graph_normalized_matches_generic() -> None:
    rng = np.random.default_rng(7)
    matrix = _unit_rows(rng.standard_normal((40, 24)).astype(np.float32))
    generic = knn_graph(matrix, k=4, assume_normalized=False)
    fast = knn_graph(matrix, k=4, assume_normalized=True)
    np.testing.assert_array_equal(generic.indices, fast.indices)
    np.testing.assert_allclose(generic.scores, fast.scores, atol=1e-5)


def test_knn_graph_edges_bridge() -> None:
    # Two tight clusters: rows {0,1} and {2,3}.
    matrix = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    g = knn_graph(matrix, k=1)
    edges = g.edges()
    assert edges.dtype == np.uint32
    assert edges.shape[1] == 2
    # Every edge points within a cluster, never across.
    for a, b in edges.tolist():
        assert (a < 2) == (b < 2)


def test_knn_graph_pads_short_rows_with_sentinel() -> None:
    # A NaN row produces unrankable cosines, so other rows can't fill all
    # k slots; the tail is padded with NO_NEIGHBOR / NaN, keeping the table
    # rectangular. (Finite embeddings never hit this — it's the documented
    # degenerate path.)
    matrix = np.array(
        [[1.0, 0.0], [np.nan, np.nan], [0.5, 0.5]],
        dtype=np.float32,
    )
    g = knn_graph(matrix, k=2)
    assert g.indices.shape == (3, 2)
    # Row 0 sees one finite neighbour (row 2); the NaN row is unrankable,
    # so the second slot is the sentinel with a NaN score.
    assert g.indices[0, 0] == 2
    assert g.indices[0, 1] == NO_NEIGHBOR
    assert np.isnan(g.scores[0, 1])
    # edges() drops sentinel padding.
    assert (g.edges() == NO_NEIGHBOR).sum() == 0


def test_knn_graph_rejects_non_contiguous() -> None:
    # A column slice is not C-contiguous; the numpy buffer layer rejects
    # it with TypeError (same as the existing top_k_cosine sibling).
    base = np.arange(40, dtype=np.float32).reshape(4, 10)
    sliced = base[:, ::2]
    assert not sliced.flags["C_CONTIGUOUS"]
    with pytest.raises(TypeError):
        knn_graph(sliced, k=1)
    # ...and as_contiguous_f32 is the documented fix.
    fixed = as_contiguous_f32(sliced)
    assert knn_graph(fixed, k=1).n_rows == 4


# ---------------------------------------------------------------------------
# near_duplicates
# ---------------------------------------------------------------------------


def test_near_duplicates_finds_pairs() -> None:
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    nd = near_duplicates(matrix, threshold=0.9)
    assert isinstance(nd, NearDuplicates)
    assert nd.pairs.tolist() == [[0, 1]]
    assert nd.scores[0] == pytest.approx(1.0, abs=1e-5)
    assert nd.truncated is False
    assert len(nd) == 1


def test_near_duplicates_lexicographic_order() -> None:
    matrix = np.array([[1.0, 0.0]] * 3, dtype=np.float32)
    nd = near_duplicates(matrix, threshold=0.5)
    assert nd.pairs.tolist() == [[0, 1], [0, 2], [1, 2]]


def test_near_duplicates_max_pairs_warns_and_truncates() -> None:
    matrix = np.array([[1.0, 0.0]] * 3, dtype=np.float32)
    with pytest.warns(UserWarning, match="truncated"):
        nd = near_duplicates(matrix, threshold=0.5, max_pairs=2)
    assert nd.truncated is True
    assert len(nd) == 2
    assert nd.pairs.tolist() == [[0, 1], [0, 2]]


def test_near_duplicates_none_below_threshold() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no spurious truncation warning
        nd = near_duplicates(matrix, threshold=0.5)
    assert nd.pairs.shape == (0, 2)
    assert not nd.truncated


def test_near_duplicates_normalized_matches_generic() -> None:
    rng = np.random.default_rng(3)
    matrix = _unit_rows(rng.standard_normal((25, 12)).astype(np.float32))
    generic = near_duplicates(matrix, threshold=0.2, assume_normalized=False)
    fast = near_duplicates(matrix, threshold=0.2, assume_normalized=True)
    np.testing.assert_array_equal(generic.pairs, fast.pairs)


# ---------------------------------------------------------------------------
# as_contiguous_f32
# ---------------------------------------------------------------------------


def test_as_contiguous_f32_casts_and_contiguates() -> None:
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)[:, ::-1]  # non-contiguous f64
    out = as_contiguous_f32(arr)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(out, arr.astype(np.float32))


def test_as_contiguous_f32_noop_on_conforming_input() -> None:
    arr = np.ones((3, 4), dtype=np.float32)
    out = as_contiguous_f32(arr)
    # Already C-contiguous float32 → returned without a copy.
    assert out is arr
