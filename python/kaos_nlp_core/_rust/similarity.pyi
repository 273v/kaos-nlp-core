"""Type stubs for ``kaos_nlp_core._rust.similarity``.

Runtime implementation: ``rust/bindings/similarity.rs``.
"""

import numpy as np

def cosine(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Cosine similarity between two equal-length float32 vectors.

    Returns a value in ``[-1.0, 1.0]``. Raises ``ValueError`` on
    empty input or length mismatch.
    """

def cosine_normalized(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Cosine of two **unit-norm** float32 vectors (fast path).

    Skips both squared-norm computations; cosine of two unit-norm
    vectors is their dot product, clamped to ``[-1, 1]``.
    **Caller contract**: both ``a`` and ``b`` must already be
    unit-norm — the function does not verify this.
    """

def cosine_one_to_many(
    query: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Cosine of a 1-D query against every row of a 2-D matrix.

    ``query`` is a 1-D float32 array of length ``dim``; ``matrix`` is
    a C-contiguous 2-D float32 array of shape ``(n_rows, dim)``.
    Returns a 1-D float32 array of length ``n_rows``.
    """

def cosine_one_to_many_normalized(
    query: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Cosine of a **unit-norm** query against every row of a 2-D matrix
    of **unit-norm** rows (fast path).

    Each row reduces to a single SIMD dot product, clamped to
    ``[-1, 1]``. **Caller contract**: ``query`` and every row of
    ``matrix`` must already be unit-norm — the function does not
    verify this.
    """

def cosine_adjacent(
    matrix: np.ndarray,
) -> np.ndarray:
    """Cosine between each pair of adjacent rows in a 2-D float32 matrix.

    Returns a 1-D float32 array of length ``n_rows - 1``; empty when
    ``n_rows < 2``.
    """

def cosine_adjacent_normalized(
    matrix: np.ndarray,
) -> np.ndarray:
    """Adjacent-row cosine on a matrix of **unit-norm** rows (fast path).

    Each adjacent pair reduces to a single SIMD dot product, clamped
    to ``[-1, 1]``. **Caller contract**: every row of ``matrix`` must
    already be unit-norm — the function does not verify this.
    """

def top_k_cosine(
    query: np.ndarray,
    matrix: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine retrieval.

    Returns ``(indices, scores)``: ``indices`` is a 1-D ``uint32``
    array of row indices, ``scores`` is a 1-D ``float32`` array of
    cosine similarities, both ordered by score descending with
    ascending-index tie-break.
    """

def mmr_select(
    matrix: np.ndarray,
    relevance: np.ndarray,
    k: int,
    lambda_: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximal Marginal Relevance selection.

    Returns ``(indices, scores)``: row indices picked greedily by
    ``lambda_ * relevance - (1 - lambda_) * max_sim_to_picked`` and
    the MMR scores at pick time.
    """

def l2_normalize_in_place(
    vector: np.ndarray,
) -> bool:
    """L2-normalize a 1-D float32 array in place.

    Returns ``True`` if the vector was normalized, ``False`` if its
    L2 norm was zero (in which case the vector is left untouched).
    """
