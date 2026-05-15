"""Type stubs for ``kaos_nlp_core._rust.aggregation``.

Runtime implementation: ``rust/bindings/aggregation.rs``.

The kernels operate on CSR-style ragged-array inputs: ``flat_ids`` is
the concatenation of every chunk's interned ``uint32`` label ids;
``chunk_offsets`` is a ``uint32`` array of length ``n_chunks + 1``
delimiting the chunks. ``n_labels`` is the exclusive upper bound on
the ids (assigned in first-appearance order by the Python wrapper so
that "lowest id wins" matches the original "lowest first-seen wins"
tiebreak).
"""

import numpy as np

def vote(
    flat_ids: np.ndarray,
    chunk_offsets: np.ndarray,
    n_labels: int,
) -> int | None:
    """Plurality vote across chunks.

    Returns the id of the label appearing in the most chunks, with
    lowest-id tiebreak; ``None`` if every chunk was empty.
    """

def majority(
    flat_ids: np.ndarray,
    chunk_offsets: np.ndarray,
    n_labels: int,
    threshold: float,
) -> int | None:
    """Threshold-gated majority vote.

    Returns the id of the first label whose distinct-chunk count is at
    least ``threshold * n_chunks``; ``None`` when nobody qualifies.
    """

def union(
    flat_ids: np.ndarray,
    chunk_offsets: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """Ids of every label appearing in any chunk, ascending."""

def intersection(
    flat_ids: np.ndarray,
    chunk_offsets: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """Ids of labels appearing in *every* chunk, ascending."""

def weighted_single(
    flat_ids: np.ndarray,
    chunk_offsets: np.ndarray,
    weights: np.ndarray,
    n_labels: int,
    threshold: float,
) -> int | None:
    """Highest-weight label crossing ``threshold * sum(weights)``."""

def weighted_multi(
    flat_ids: np.ndarray,
    chunk_offsets: np.ndarray,
    weights: np.ndarray,
    n_labels: int,
    threshold: float,
) -> np.ndarray:
    """Ids whose accumulated weight crosses the threshold, ascending."""

def max_score_single(
    flat_ids: np.ndarray,
    flat_scores: np.ndarray,
    n_labels: int,
    threshold: float | None = None,
) -> int | None:
    """Label with the highest pooled max-score (None below threshold)."""

def max_score_multi(
    flat_ids: np.ndarray,
    flat_scores: np.ndarray,
    n_labels: int,
    threshold: float | None = None,
) -> np.ndarray:
    """Ids whose pooled max-score strictly exceeds the cutoff, ascending."""
