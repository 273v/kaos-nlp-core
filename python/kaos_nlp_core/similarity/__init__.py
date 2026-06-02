"""Dense-vector similarity primitives — hardware-accelerated Rust core.

Companion to :mod:`kaos_nlp_core.algorithms` (string similarity),
:mod:`kaos_nlp_core.aggregation` (label aggregation), and the sparse
matrix types in :mod:`kaos_nlp_core.structures`. This module covers
the **dense f32 vector** case: cosine similarity, top-k retrieval,
MMR reranking, and L2 normalization, all routed through hand-written
``std::arch`` SIMD kernels in Rust with runtime ISA dispatch (AVX-512F
/ AVX2+FMA / NEON / scalar fallback). The kernel pattern is ported
from NumKong (Apache-2.0); see ``docs/design-similarity-simd.md``.

Surface
-------

Generic path (computes both norms — safe default):

- :func:`cosine` — single pair cosine similarity.
- :func:`cosine_one_to_many` — cos(query, every row of a matrix).
- :func:`cosine_adjacent` — cos(M[i], M[i+1]) for every adjacent
  pair; used by the semantic chunker.

Pre-normalised fast path — for callers that guarantee unit-norm
inputs (skips both norm computations + final sqrt; ~2x faster at
production shapes):

- :func:`cosine_normalized` — single pair, both inputs unit-norm.
- :func:`cosine_one_to_many_normalized` — unit-norm query against a
  matrix of unit-norm rows.
- :func:`cosine_adjacent_normalized` — adjacent-row cosine on a
  matrix of unit-norm rows.

Selection / reranking:

- :func:`top_k_cosine` — argpartition + sort, returns
  :class:`TopKResult`.
- :func:`mmr_select` — Maximal Marginal Relevance reranking, returns
  :class:`MMRResult`.
- :func:`l2_normalize_in_place` — unit-norm a vector in place.

Similarity-graph primitives (all-pairs) — one fused Rust call each, so
callers don't hand-roll an ``N x top_k_cosine`` loop:

- :func:`knn_graph` — for every row, its ``k`` nearest other rows;
  returns :class:`KnnGraph`.
- :func:`near_duplicates` — every row pair with cosine ``>= threshold``;
  returns :class:`NearDuplicates`.

These build the *similarity edge set* from dense vectors. Turning that
edge set into clusters/groups (connected components, community
detection) is the **graph layer's** job, not this module's: feed the
``(m, 2)`` edges from :attr:`NearDuplicates.pairs` or
:meth:`KnnGraph.edges` into ``kaos_graph`` (e.g.
``kaos_graph.algorithms.connected_components_from_edges``). Higher-level
*semantic-dedup orchestration* (policy, canonical-record selection,
clustering levels) lives in ``kaos-content``. This module deliberately
owns neither — it stays the exact, deterministic dense-vector layer that
both build on.

Input contract
--------------

All functions accept ``numpy`` float32 arrays. 2-D inputs must be
**C-contiguous** and shaped ``(n_rows, dim)``. The Rust layer
validates this and raises ``ValueError`` with a precise reason on
shape / dtype / contiguity violations.

:func:`as_contiguous_f32` is the one-call escape hatch when an array is
not yet in that layout: ``EmbeddingModel.embed()`` already returns
C-contiguous float32, but slicing columns, ``np.stack``-ing
heterogeneous sources, or loading from a foreign dtype can break it.

Numerical contract
------------------

- Cosine results are clipped to ``[-1.0, 1.0]`` to absorb floating-point
  round-off near ``±1``.
- Zero-norm vectors yield cosine ``0.0`` (rather than ``NaN``).
- NaN inputs in any batch are dropped (top-k) or skipped (MMR) — they
  cannot be ranked.
- Determinism: same inputs, same SIMD lane width → bit-for-bit
  identical results. Tie-breaks (top-k, MMR) use ascending row index.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# Re-export the Rust-side primitives under stable Python names. The
# Rust functions all accept and return numpy arrays so there is no
# runtime conversion layer here — this module exists mostly for
# documentation, typed helpers (TopKResult, MMRResult), and a single
# canonical import path callers can use.
from kaos_nlp_core._rust.similarity import (
    cosine,
    cosine_adjacent,
    cosine_adjacent_normalized,
    cosine_normalized,
    cosine_one_to_many,
    cosine_one_to_many_normalized,
    l2_normalize_in_place,
)
from kaos_nlp_core._rust.similarity import (
    knn_graph as _rust_knn_graph,
)
from kaos_nlp_core._rust.similarity import (
    mmr_select as _rust_mmr_select,
)
from kaos_nlp_core._rust.similarity import (
    near_duplicates as _rust_near_duplicates,
)
from kaos_nlp_core._rust.similarity import (
    top_k_cosine as _rust_top_k_cosine,
)


@dataclass(frozen=True, slots=True)
class TopKResult:
    """Result of a top-k cosine retrieval.

    Attributes:
        indices: ``uint32`` 1-D numpy array of row indices into the
            original matrix, in result order (score descending).
        scores: ``float32`` 1-D numpy array of cosine similarity
            scores, aligned element-wise with ``indices``.
    """

    indices: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __iter__(self):
        """Iterate as ``(index, score)`` pairs."""
        return zip(self.indices.tolist(), self.scores.tolist(), strict=True)


@dataclass(frozen=True, slots=True)
class MMRResult:
    """Result of an MMR (Maximal Marginal Relevance) selection.

    Attributes:
        indices: ``uint32`` 1-D numpy array of row indices in pick
            order. The first index is ``argmax(relevance)``; each
            subsequent pick balances relevance against pairwise
            cosine to the already-picked set, weighted by
            ``lambda_``.
        scores: ``float32`` 1-D numpy array of MMR scores at pick
            time. Element ``i`` is
            ``lambda_ * relevance[indices[i]] - (1 - lambda_) * max_sim_at_pick_i``.
    """

    indices: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __iter__(self):
        """Iterate as ``(index, score)`` pairs in pick order."""
        return zip(self.indices.tolist(), self.scores.tolist(), strict=True)


def top_k_cosine(
    query: np.ndarray,
    matrix: np.ndarray,
    k: int,
) -> TopKResult:
    """Top-k rows of ``matrix`` by cosine similarity to ``query``.

    The Rust core performs the cosine sweep via NumKong-dispatched
    SIMD kernels (AVX-512 / AVX2 / NEON / scalar fallback at
    runtime), then runs an ``O(n log k)`` heap-based selection with
    ascending-index tie-break.

    Args:
        query: 1-D float32 numpy array.
        matrix: 2-D float32 numpy array of shape ``(n_rows, dim)``.
            Must be C-contiguous.
        k: number of rows to return. Capped silently at ``n_rows``.
            ``k=0`` returns an empty result.

    Returns:
        :class:`TopKResult` ordered by score descending; ties broken
        by ascending row index.

    Raises:
        ValueError: dimension mismatch between query and matrix,
            non-contiguous matrix, or wrong dtype.
    """
    indices, scores = _rust_top_k_cosine(query, matrix, k)
    return TopKResult(indices=indices, scores=scores)


def mmr_select(
    matrix: np.ndarray,
    relevance: np.ndarray,
    k: int,
    lambda_: float = 0.5,
) -> MMRResult:
    """Greedy Maximal Marginal Relevance selection over a dense matrix.

    Picks ``k`` rows in order, balancing per-row relevance against
    similarity to the already-picked set. ``lambda_=1.0`` collapses
    to "rank by relevance"; ``lambda_=0.0`` anchors on
    ``argmax(relevance)`` then maximally diversifies.

    Args:
        matrix: 2-D float32 numpy array of shape ``(n_rows, dim)``.
        relevance: 1-D float32 numpy array of length ``n_rows``.
        k: number of picks. Capped at ``n_rows``.
        lambda_: diversity weight in ``[0.0, 1.0]``. Default
            ``0.5`` (equal weight on relevance and diversity).
            Values outside the range are clamped.

    Returns:
        :class:`MMRResult` in pick order.

    Raises:
        ValueError: shape mismatch (matrix vs relevance), wrong
            dtype, or non-contiguous matrix.
    """
    indices, scores = _rust_mmr_select(matrix, relevance, k, lambda_)
    return MMRResult(indices=indices, scores=scores)


# Sentinel row index for an unfilled :class:`KnnGraph` neighbour slot.
# A row reports fewer than ``k`` neighbours only when the matrix holds
# NaN/inf rows (their cosines are unrankable); the empty slots carry this
# index with a ``NaN`` score. With finite inputs it never appears.
NO_NEIGHBOR: int = np.iinfo(np.uint32).max


@dataclass(frozen=True, slots=True)
class KnnGraph:
    """Result of a :func:`knn_graph` build — a dense neighbour table.

    Attributes:
        indices: ``uint32`` 2-D numpy array of shape ``(n_rows, k)``.
            ``indices[i, r]`` is the ``r``-th nearest neighbour of row
            ``i`` (score descending, ascending-index tie-break), or
            :data:`NO_NEIGHBOR` for an unfilled slot.
        scores: ``float32`` 2-D numpy array of shape ``(n_rows, k)``,
            aligned element-wise with ``indices``; ``NaN`` in unfilled
            slots.
    """

    indices: np.ndarray
    scores: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.indices.shape[0])

    @property
    def k(self) -> int:
        """Neighbours per row (the trailing dimension)."""
        return int(self.indices.shape[1]) if self.indices.ndim == 2 else 0

    def edges(self) -> np.ndarray:
        """Return the graph as a ``(m, 2)`` ``uint32`` ``(src, dst)`` edge
        array, dropping :data:`NO_NEIGHBOR` padding.

        Convenience bridge to the graph layer — feed the result into
        ``kaos_graph`` for component labelling / clustering::

            from kaos_graph.algorithms import connected_components_from_edges

            g = knn_graph(matrix, k=5)
            labels = connected_components_from_edges(g.n_rows, g.edges())
        """
        if self.k == 0 or self.n_rows == 0:
            return np.empty((0, 2), dtype=np.uint32)
        src = np.repeat(np.arange(self.n_rows, dtype=np.uint32), self.k)
        dst = self.indices.reshape(-1)
        mask = dst != NO_NEIGHBOR
        return np.stack([src[mask], dst[mask]], axis=1).astype(np.uint32, copy=False)


@dataclass(frozen=True, slots=True)
class NearDuplicates:
    """Result of a :func:`near_duplicates` sweep — the over-threshold edges.

    Attributes:
        pairs: ``uint32`` 2-D numpy array of shape ``(m, 2)``. Each row is
            an ``(i, j)`` pair with ``i < j``, in lexicographic order.
        scores: ``float32`` 1-D numpy array of length ``m``, the cosine
            for each pair.
        truncated: ``True`` if a ``max_pairs`` cap clamped the output —
            more pairs met the threshold than were returned.
    """

    pairs: np.ndarray
    scores: np.ndarray
    truncated: bool

    def __len__(self) -> int:
        return int(self.pairs.shape[0])


def knn_graph(
    matrix: np.ndarray,
    k: int,
    *,
    include_self: bool = False,
    assume_normalized: bool = False,
) -> KnnGraph:
    """Build a k-nearest-neighbour graph over the rows of ``matrix``.

    For every row, computes its ``k`` most cosine-similar rows in a single
    fused Rust call — the per-row cosine sweeps run the SIMD kernels with
    the GIL released and fan out across Rayon threads, so this replaces an
    ``N x top_k_cosine`` Python loop.

    Args:
        matrix: 2-D float32 numpy array of shape ``(n_rows, dim)``. Must be
            C-contiguous (see :func:`as_contiguous_f32`).
        k: neighbours per row. Silently capped at the number available
            (``n_rows - 1`` when ``include_self`` is false).
        include_self: when ``False`` (default) a row is never its own
            neighbour. When ``True`` each row's top hit is itself at
            cosine ``1.0``.
        assume_normalized: when ``True``, take the unit-norm fast path —
            the caller guarantees every row is L2-unit-norm (which
            ``kaos-nlp-transformers`` ``EmbeddingModel`` output already is).

    Returns:
        :class:`KnnGraph` with rectangular ``(n_rows, effective_k)``
        ``indices`` and ``scores``.

    Raises:
        TypeError: ``matrix`` is not C-contiguous or not float32 (the
            numpy buffer contract — see :func:`as_contiguous_f32`).
        ValueError: ``matrix`` is not a rectangular 2-D array.
    """
    indices, scores = _rust_knn_graph(matrix, k, include_self, assume_normalized)
    return KnnGraph(indices=indices, scores=scores)


def near_duplicates(
    matrix: np.ndarray,
    threshold: float,
    *,
    assume_normalized: bool = False,
    max_pairs: int | None = None,
) -> NearDuplicates:
    """Find every row pair ``(i, j)``, ``i < j``, with cosine ``>= threshold``.

    The edge set a dedup pre-pass needs. Each row's sweep against the rows
    after it runs in one SIMD pass; rows fan out across Rayon threads and
    results concatenate in row order, so the pair list is deterministic.

    Args:
        matrix: 2-D float32 numpy array of shape ``(n_rows, dim)``. Must be
            C-contiguous (see :func:`as_contiguous_f32`).
        threshold: minimum cosine for a pair to be reported.
        assume_normalized: unit-norm fast path (same contract as
            :func:`knn_graph`).
        max_pairs: optional cap on the number of returned pairs. When the
            qualifying count exceeds it, the first ``max_pairs`` in
            ``(i, j)`` order are kept and a :class:`UserWarning` is emitted
            (the result's ``truncated`` flag is also set) — the cap is
            never silent.

    Returns:
        :class:`NearDuplicates`.

    Raises:
        TypeError: ``matrix`` is not C-contiguous or not float32 (the
            numpy buffer contract — see :func:`as_contiguous_f32`).
        ValueError: ``matrix`` is not a rectangular 2-D array.
    """
    pairs, scores, truncated = _rust_near_duplicates(
        matrix, threshold, assume_normalized, max_pairs
    )
    if truncated:
        warnings.warn(
            f"near_duplicates: more pairs met threshold={threshold} than the "
            f"max_pairs={max_pairs} cap; output truncated to {len(scores)} pairs "
            "in (i, j) order. Raise max_pairs to capture the full edge set.",
            UserWarning,
            stacklevel=2,
        )
    return NearDuplicates(pairs=pairs, scores=scores, truncated=bool(truncated))


def as_contiguous_f32(array: np.ndarray) -> np.ndarray:
    """Coerce ``array`` to the C-contiguous float32 layout the similarity
    primitives require.

    The similarity functions accept only C-contiguous float32 arrays and
    raise ``ValueError`` otherwise (the coercion is explicit, never hidden
    in the hot path). ``EmbeddingModel.embed()`` output already satisfies
    the contract; reach for this only when an array might not — after
    slicing columns, ``np.stack``-ing heterogeneous sources, or loading
    from a different dtype. Returns the input unchanged (no copy) when it
    is already C-contiguous float32.

    Args:
        array: any numpy array-like.

    Returns:
        A C-contiguous ``float32`` numpy array.
    """
    return np.ascontiguousarray(array, dtype=np.float32)


__all__ = [
    "NO_NEIGHBOR",
    "KnnGraph",
    "MMRResult",
    "NearDuplicates",
    "TopKResult",
    "as_contiguous_f32",
    "cosine",
    "cosine_adjacent",
    "cosine_adjacent_normalized",
    "cosine_normalized",
    "cosine_one_to_many",
    "cosine_one_to_many_normalized",
    "knn_graph",
    "l2_normalize_in_place",
    "mmr_select",
    "near_duplicates",
    "top_k_cosine",
]
