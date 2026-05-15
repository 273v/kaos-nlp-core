"""Shared packing helper for sentence/paragraph/section chunkers.

A "packer" walks an ordered sequence of indivisible *units* (sentences,
paragraphs, lines) and groups consecutive units into chunks that stay
under a token budget. The packer respects unit boundaries; it never
splits a single unit even if that unit is itself larger than the
budget — in that case the oversize unit emits as its own chunk, and
the caller is expected to subdivide it via a finer-grained chunker.

The hot loop runs in the Rust extension
(``kaos_nlp_core._rust.chunking.pack_units``); this Python wrapper
marshals offset/token arrays into Rust, then materialises
:class:`Chunk` objects from the returned group records (slicing source
text, merging unit metadata, and calling ``token_counter`` for the
final per-chunk count).

This module is private. Public chunkers in
``kaos_nlp_core.chunking.*`` invoke :func:`pack_units` with their
unit list and token counter.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from kaos_nlp_core._rust.chunking import pack_units as _rust_pack_units
from kaos_nlp_core.chunking import Chunk


def default_token_counter(text: str) -> int:
    """Approximate token count for ``text``.

    Uses the ``ceil(len(text) / 4)`` heuristic, which is within roughly
    25% of the OpenAI ``cl100k_base`` count for English prose. Good
    enough for chunk-budget planning, which is itself approximate;
    callers that need provider-accurate counts can pass a
    tokenizer-backed counter into the chunker constructor.

    The empty string returns ``0``, not ``1`` — important for empty
    paragraph handling.
    """
    if not text:
        return 0
    # Ceiling division
    return -(-len(text) // 4)


_EMPTY_METADATA_DICT: dict[str, object] = {}


class _Unit:
    """Internal sortable unit used by :func:`pack_units`.

    Carries the verbatim text, its absolute offsets in the source, an
    optional pre-computed token count, and arbitrary metadata. Each
    unit is the smallest piece the packer is willing to keep whole.

    Memory note: when ``metadata`` is ``None`` (the common case) the
    unit shares the module-level empty dict via
    :data:`_EMPTY_METADATA_DICT` rather than allocating a fresh
    ``{}``. This shaves per-unit allocations on hot chunker paths
    (SentenceChunker reported a 253 MB RSS delta on 1000 USC docs
    before the metadata bypass landed).
    """

    __slots__ = ("end", "metadata", "start", "text", "token_count")

    def __init__(
        self,
        *,
        text: str,
        start: int,
        end: int,
        token_count: int,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.token_count = token_count
        self.metadata = metadata if metadata is not None else _EMPTY_METADATA_DICT


def pack_units(
    units: Sequence[_Unit],
    *,
    source: str,
    parent_id: str | None,
    max_tokens: int,
    overlap_units: int = 0,
    token_counter: Callable[[str], int] = default_token_counter,
    depth: int = 0,
    chunk_metadata: dict[str, object] | None = None,
) -> list[Chunk]:
    """Group ``units`` into :class:`Chunk` instances under ``max_tokens``.

    The packer is greedy: it keeps appending units to the current
    chunk until adding the next one would exceed the budget, at which
    point it emits the current chunk and starts a new one.

    Args:
        units: Ordered, non-overlapping units in source-order. Empty
            units (zero-length) are filtered before packing.
        source: The full source text; used to compute chunk text via
            offsets so adjacent units that include whitespace are
            preserved.
        parent_id: Parent identifier to attach to each emitted chunk.
        max_tokens: Soft ceiling on chunk token count. A chunk may
            exceed the ceiling when it consists of exactly one
            oversize unit (the packer never splits a unit).
        overlap_units: Number of trailing units to repeat at the start
            of the next chunk. Used by retrieval chunkers to maintain
            local context across boundaries.
        token_counter: Callable used to score the chunk text. Default
            is :func:`default_token_counter`.
        depth: Recursive depth passed through to each emitted chunk.
        chunk_metadata: Additional metadata copied into each chunk's
            ``metadata`` field. Unit-specific metadata is merged on
            top of this.

    Returns:
        A list of :class:`Chunk` ordered by ``start`` ascending. The
        chunks' offsets are derived from the slice
        ``source[first_unit.start:last_unit.end]``, so any whitespace
        captured between units in the original source is preserved in
        the chunk text.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
    if overlap_units < 0:
        raise ValueError(f"overlap_units must be >= 0, got {overlap_units}")

    filtered = [u for u in units if u.end > u.start]
    if not filtered:
        return []

    # Marshal unit offsets and per-unit token counts into uint32
    # arrays for the Rust kernel. The Rust packer runs the greedy
    # budget loop and returns parallel arrays describing the resulting
    # groups; we materialise Chunks from those records in Python so
    # the Rust side never sees a Python string, dict, or callable.
    n = len(filtered)
    starts = np.empty(n, dtype=np.uint32)
    ends = np.empty(n, dtype=np.uint32)
    token_counts = np.empty(n, dtype=np.uint32)
    for i, u in enumerate(filtered):
        starts[i] = u.start
        ends[i] = u.end
        token_counts[i] = u.token_count

    (
        group_starts,
        group_ends,
        group_unit_starts,
        group_unit_ends,
        _group_token_sums,
    ) = _rust_pack_units(starts, ends, token_counts, max_tokens, overlap_units)

    base_metadata = chunk_metadata or {}
    chunks: list[Chunk] = []
    for gs, ge, us, ue in zip(
        group_starts.tolist(),
        group_ends.tolist(),
        group_unit_starts.tolist(),
        group_unit_ends.tolist(),
        strict=True,
    ):
        chunk_text = source[gs:ge]
        merged_metadata: dict[str, object] = dict(base_metadata)
        merged_metadata["units"] = ue - us
        for unit in filtered[us:ue]:
            for key, value in unit.metadata.items():
                merged_metadata.setdefault(key, value)
        chunks.append(
            Chunk(
                text=chunk_text,
                start=gs,
                end=ge,
                parent_id=parent_id,
                token_count=token_counter(chunk_text),
                depth=depth,
                metadata=merged_metadata,
            )
        )

    return chunks
