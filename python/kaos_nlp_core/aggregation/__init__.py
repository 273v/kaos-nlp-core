"""Deterministic label aggregation primitives.

Pure functions that combine multiple per-chunk classification results
into a single decision. These primitives are LLM-free and have no
external dependencies; classifier strategy classes in
``kaos-llm-core`` wrap them with the
:class:`~kaos_llm_core.composition.aggregate.Aggregator` protocol.

All functions operate on **sequences of label name sets** rather than
on classification result objects. The caller is responsible for
extracting label names from whichever result type they hold. This
keeps the aggregation layer independent of any specific result class
(``Classification[L]``, ``ToolResult``, raw dicts, …).

Implementation note
-------------------

The heavy lifting runs in the Rust extension
:mod:`kaos_nlp_core._rust.aggregation`. The Python layer here only
*interns* string label names to ``u32`` ids (preserving first-
appearance order for the tiebreak contract) and marshals them into
the CSR-style ragged-array format the Rust kernels accept. Rust does
the counts, threshold checks, and ordering with the GIL released, then
the Python layer maps winning ids back to names.

The public function signatures match the original pure-Python
implementation exactly; downstream callers do not need to change.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from kaos_nlp_core._rust.aggregation import intersection as _rust_intersection
from kaos_nlp_core._rust.aggregation import majority as _rust_majority
from kaos_nlp_core._rust.aggregation import max_score_multi as _rust_max_score_multi
from kaos_nlp_core._rust.aggregation import max_score_single as _rust_max_score_single
from kaos_nlp_core._rust.aggregation import union as _rust_union
from kaos_nlp_core._rust.aggregation import vote as _rust_vote
from kaos_nlp_core._rust.aggregation import weighted_multi as _rust_weighted_multi
from kaos_nlp_core._rust.aggregation import weighted_single as _rust_weighted_single


def _intern_chunks(
    per_chunk: Sequence[Iterable[str]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Intern per-chunk label names into the CSR-style ragged arrays.

    Returns
    -------
    flat_ids
        1-D ``uint32`` array — concatenation of every chunk's
        deduplicated label ids in chunk order. Within a chunk,
        duplicates are dropped using :func:`dict.fromkeys` so the
        Rust kernels see the same dedupe contract the Python
        implementation used (preserving insertion order and avoiding
        the hash-randomization leak that plain :class:`set` would
        introduce).
    chunk_offsets
        1-D ``uint32`` array of length ``n_chunks + 1``; chunk ``c``
        occupies ``flat_ids[chunk_offsets[c]:chunk_offsets[c + 1]]``.
    name_table
        List of label names, ids ordered by first appearance. Ids are
        therefore assigned in input order, which makes the
        Rust-side "lowest id wins" tiebreak equivalent to the
        Python "lowest first_seen wins" tiebreak.
    """
    name_to_id: dict[str, int] = {}
    name_table: list[str] = []
    flat: list[int] = []
    offsets: list[int] = [0]
    for chunk_labels in per_chunk:
        for name in dict.fromkeys(chunk_labels):
            nid = name_to_id.get(name)
            if nid is None:
                nid = len(name_table)
                name_to_id[name] = nid
                name_table.append(name)
            flat.append(nid)
        offsets.append(len(flat))
    flat_arr = np.asarray(flat, dtype=np.uint32)
    off_arr = np.asarray(offsets, dtype=np.uint32)
    return flat_arr, off_arr, name_table


def _intern_score_chunks(
    per_chunk_scores: Sequence[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Intern per-chunk score maps into parallel flat arrays.

    Returns ``(flat_ids, flat_scores, name_table)``. Empty maps are
    permitted (they contribute nothing). Insertion order across
    chunks defines the id assignment, matching the Python
    implementation's first-appearance tiebreak.
    """
    name_to_id: dict[str, int] = {}
    name_table: list[str] = []
    flat_ids: list[int] = []
    flat_scores: list[float] = []
    for chunk_map in per_chunk_scores:
        for name, value in chunk_map.items():
            nid = name_to_id.get(name)
            if nid is None:
                nid = len(name_table)
                name_to_id[name] = nid
                name_table.append(name)
            flat_ids.append(nid)
            flat_scores.append(value)
    ids_arr = np.asarray(flat_ids, dtype=np.uint32)
    scores_arr = np.asarray(flat_scores, dtype=np.float64)
    return ids_arr, scores_arr, name_table


def vote(per_chunk: Sequence[Iterable[str]]) -> str | None:
    """Return the label that appears in the most chunks.

    Single-label voting. Each input is a chunk's set of picked labels;
    each chunk contributes one vote per *distinct* label name it
    contains (duplicates within the same chunk do not double-count).

    Args:
        per_chunk: Sequence of per-chunk label iterables.

    Returns:
        The single most-voted label name, or ``None`` when
        ``per_chunk`` is empty or contains only empty iterables. Ties
        are broken by order of first appearance, where "first
        appearance" is defined by **input order**, not Python hash
        order.
    """
    flat, offsets, name_table = _intern_chunks(per_chunk)
    if len(name_table) == 0:
        return None
    winner_id = _rust_vote(flat, offsets, len(name_table))
    return None if winner_id is None else name_table[winner_id]


def majority(
    per_chunk: Sequence[Iterable[str]],
    *,
    threshold: float = 0.5,
) -> str | None:
    """Return a label only when it appears in at least ``threshold`` of chunks.

    Args:
        per_chunk: Sequence of per-chunk label iterables.
        threshold: Fraction of chunks (``0.0`` to ``1.0``) the winning
            label must appear in. ``0.5`` (default) requires a strict
            majority; ``1.0`` requires unanimity.

    Returns:
        The label that meets the threshold (with ties broken by order
        of first appearance), or ``None`` when no label crosses it.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    flat, offsets, name_table = _intern_chunks(per_chunk)
    if len(name_table) == 0:
        return None
    winner_id = _rust_majority(flat, offsets, len(name_table), threshold)
    return None if winner_id is None else name_table[winner_id]


def union(per_chunk: Sequence[Iterable[str]]) -> frozenset[str]:
    """Return the set of labels appearing in *any* chunk.

    Multi-label aggregator. High recall, low precision: a label
    appearing in a single chunk is included.
    """
    flat, offsets, name_table = _intern_chunks(per_chunk)
    if len(name_table) == 0:
        return frozenset()
    ids = _rust_union(flat, offsets, len(name_table))
    return frozenset(name_table[i] for i in ids.tolist())


def intersection(per_chunk: Sequence[Iterable[str]]) -> frozenset[str]:
    """Return the set of labels appearing in *every* chunk.

    Multi-label aggregator. High precision, low recall: only labels
    that show up in every chunk are included.

    For an empty input, returns ``frozenset()`` (consistent with set
    intersection on the empty collection being undefined; we choose
    the conservative empty-output convention).
    """
    flat, offsets, name_table = _intern_chunks(per_chunk)
    if len(name_table) == 0:
        return frozenset()
    ids = _rust_intersection(flat, offsets, len(name_table))
    return frozenset(name_table[i] for i in ids.tolist())


def weighted(
    per_chunk: Sequence[Iterable[str]],
    *,
    weights: Sequence[float] | None = None,
    threshold: float = 0.5,
    multi: bool = False,
) -> frozenset[str] | str | None:
    """Weighted aggregation.

    Each chunk contributes ``weights[i]`` (default ``1.0`` per chunk)
    to every label name it contains. A label "wins" when its total
    weight is at least ``threshold * sum(weights)``.

    Args:
        per_chunk: Sequence of per-chunk label iterables.
        weights: Per-chunk weights. ``None`` (default) uses ``1.0`` for
            every chunk. Length must match ``per_chunk`` when provided.
        threshold: Fraction of total weight required to "win".
        multi: When ``True``, return the frozenset of all winners.
            When ``False`` (default), return the single highest-weight
            winner; ties broken by order of first appearance.

    Returns:
        - ``multi=False``: single winning label name, or ``None``
          when no label crosses the threshold.
        - ``multi=True``: frozenset of winning label names (possibly
          empty).
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    flat, offsets, name_table = _intern_chunks(per_chunk)
    n_chunks = len(offsets) - 1
    if weights is None:
        weights_arr = np.ones(n_chunks, dtype=np.float64)
    else:
        if len(weights) != n_chunks:
            raise ValueError(
                f"weights length ({len(weights)}) must match per_chunk length ({n_chunks})"
            )
        weights_arr = np.asarray(weights, dtype=np.float64)
    if len(name_table) == 0:
        return frozenset() if multi else None
    if multi:
        ids = _rust_weighted_multi(flat, offsets, weights_arr, len(name_table), threshold)
        return frozenset(name_table[i] for i in ids.tolist())
    winner_id = _rust_weighted_single(flat, offsets, weights_arr, len(name_table), threshold)
    return None if winner_id is None else name_table[winner_id]


def max_score(
    per_chunk_scores: Sequence[dict[str, float]],
    *,
    multi: bool = False,
    threshold: float | None = None,
) -> frozenset[str] | str | None:
    """Aggregate per-chunk score maps by taking the max score per label.

    Useful when each per-chunk classification produces a probability
    distribution and the caller wants the document-level answer to be
    the label that achieved the highest single-chunk confidence.

    Args:
        per_chunk_scores: Sequence of ``{label: score}`` maps. Any
            label missing from a map is treated as score ``0.0``.
        multi: When ``True`` return every label whose max score is
            at or above ``threshold`` (if provided) or ``> 0``
            (otherwise). When ``False`` return only the single
            top-scoring label name.
        threshold: Minimum max-score required to be returned. ``None``
            means "any positive score qualifies" in ``multi`` mode and
            "always return the top" in single mode.

    Returns:
        - ``multi=False``: top-scoring label name, or ``None`` when
          all maps are empty.
        - ``multi=True``: frozenset of qualifying label names.
    """
    flat_ids, flat_scores, name_table = _intern_score_chunks(per_chunk_scores)
    if len(name_table) == 0:
        return frozenset() if multi else None
    if multi:
        ids = _rust_max_score_multi(flat_ids, flat_scores, len(name_table), threshold)
        return frozenset(name_table[i] for i in ids.tolist())
    winner_id = _rust_max_score_single(flat_ids, flat_scores, len(name_table), threshold)
    return None if winner_id is None else name_table[winner_id]


__all__ = [
    "intersection",
    "majority",
    "max_score",
    "union",
    "vote",
    "weighted",
]
