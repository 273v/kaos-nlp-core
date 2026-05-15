"""Type stubs for ``kaos_nlp_core._rust.chunking``.

Runtime implementation: ``rust/bindings/chunking.rs``.
"""

import numpy as np

def pack_units(
    starts: np.ndarray,
    ends: np.ndarray,
    token_counts: np.ndarray,
    max_tokens: int,
    overlap_units: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedy unit packer under a token budget.

    All three input arrays are 1-D ``uint32``, same length, in unit
    order. Returns five parallel ``uint32`` arrays describing the
    resulting groups: ``(group_starts, group_ends, group_unit_starts,
    group_unit_ends, group_token_sums)``. Raises ``ValueError`` for
    invalid configuration (``max_tokens == 0``, length mismatch).
    """

def semantic_pack(
    starts: np.ndarray,
    ends: np.ndarray,
    token_counts: np.ndarray,
    adj_sim: np.ndarray,
    max_tokens: int,
    drop_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedy packer that also cuts on adjacent-cosine drops.

    Inputs as in :func:`pack_units`, plus a 1-D ``float32`` array
    ``adj_sim`` of length ``n_units - 1`` (or ``0`` if ``n_units <=
    1``) carrying the adjacent-pair cosine similarities, and a
    ``drop_threshold`` cosine value below which a chunk boundary is
    forced. Returns the same five-array shape as
    :func:`pack_units`. Raises ``ValueError`` on invalid
    configuration.
    """
