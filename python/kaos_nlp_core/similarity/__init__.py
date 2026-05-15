"""Dense-vector similarity primitives — hardware-accelerated Rust core.

Companion to :mod:`kaos_nlp_core.algorithms` (string similarity),
:mod:`kaos_nlp_core.aggregation` (label aggregation), and the sparse
matrix types in :mod:`kaos_nlp_core.structures`. This module covers
the **dense f32 vector** case: cosine similarity, top-k retrieval,
MMR reranking, and L2 normalization, all routed through the
NumKong-backed Rust core for SIMD-accelerated execution (AVX-512 /
AVX2 / NEON / SVE / scalar fallback chosen at runtime).

Surface
-------

- :func:`cosine` — single pair cosine similarity.
- :func:`cosine_one_to_many` — cos(query, every row of a matrix).
- :func:`cosine_adjacent` — cos(M[i], M[i+1]) for every adjacent
  pair; used by the semantic chunker.
- :func:`top_k_cosine` — argpartition + sort, returns
  :class:`TopKResult`.
- :func:`mmr_select` — Maximal Marginal Relevance reranking, returns
  :class:`MMRResult`.
- :func:`l2_normalize_in_place` — unit-norm a vector in place.

Input contract
--------------

All functions accept ``numpy`` float32 arrays. 2-D inputs must be
**C-contiguous** and shaped ``(n_rows, dim)``. The Rust layer
validates this and raises ``ValueError`` with a precise reason on
shape / dtype / contiguity violations.

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
    cosine_one_to_many,
    l2_normalize_in_place,
)
from kaos_nlp_core._rust.similarity import (
    mmr_select as _rust_mmr_select,
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


__all__ = [
    "MMRResult",
    "TopKResult",
    "cosine",
    "cosine_adjacent",
    "cosine_one_to_many",
    "l2_normalize_in_place",
    "mmr_select",
    "top_k_cosine",
]
