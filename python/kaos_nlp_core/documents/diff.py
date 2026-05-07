"""Sentence- and paragraph-level document diffing.

Identifies modifications, additions, removals, and (optionally) moves
between two text documents using similarity matching at the chosen
granularity. The Rust core is at ``rust/core/diff.rs``; this wrapper
layers typed result objects and the bundled Punkt default on top of
the PyO3 binding.

Quick start::

    from kaos_nlp_core.documents import diff_documents

    a = "The Lessor agrees to lease. Rent is monthly. Term is one year."
    b = "The Lessor agrees to lease. Rent is quarterly. Term is one year."
    for change in diff_documents(a, b):
        print(change.kind, change.score, change.left_text, "->", change.right_text)

The default tokenizer is the trained legal Punkt model embedded in the
native extension (`DEFAULT_PUNKT_BYTES`). Pass a custom ``PunktTokenizer``
to override it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from kaos_nlp_core._rust.diff import py_diff_documents as _raw_diff_documents
from kaos_nlp_core._rust.segmentation import PunktTokenizer

Granularity = Literal["sentence", "paragraph", "line", "paragraph_simple"]
ChangeKind = Literal["unchanged", "modified", "moved", "added", "removed"]


@dataclass(frozen=True, slots=True)
class SegmentRef:
    """Reference to a segment within one of the diffed documents.

    Offsets are **character offsets** suitable for Python ``str`` slicing.
    """

    index: int
    """0-based segment index within its source document."""

    start: int
    """Character offset start (inclusive)."""

    end: int
    """Character offset end (exclusive)."""

    text: str
    """The segment text, sliced from the source."""


@dataclass(frozen=True, slots=True)
class SegmentChange:
    """One segment-level change between two documents."""

    kind: ChangeKind
    """One of ``"unchanged"``, ``"modified"``, ``"moved"``, ``"added"``,
    ``"removed"``."""

    left: SegmentRef | None
    """Segment in the source document; ``None`` for ``"added"``."""

    right: SegmentRef | None
    """Segment in the target document; ``None`` for ``"removed"``."""

    score: float
    """Similarity in ``[0, 1]``. ``0.0`` for ``"added"`` / ``"removed"``."""

    @property
    def left_text(self) -> str:
        """Convenience accessor; empty string when ``left`` is ``None``."""
        return self.left.text if self.left is not None else ""

    @property
    def right_text(self) -> str:
        """Convenience accessor; empty string when ``right`` is ``None``."""
        return self.right.text if self.right is not None else ""


def _to_segref(raw: dict[str, Any] | None) -> SegmentRef | None:
    if raw is None:
        return None
    return SegmentRef(
        index=int(raw["index"]),
        start=int(raw["start"]),
        end=int(raw["end"]),
        text=str(raw["text"]),
    )


def _to_change(raw: dict[str, Any]) -> SegmentChange:
    return SegmentChange(
        kind=raw["kind"],
        left=_to_segref(raw.get("left")),
        right=_to_segref(raw.get("right")),
        score=float(raw["score"]),
    )


def diff_documents(
    a: str,
    b: str,
    *,
    granularity: Granularity = "sentence",
    algorithm: str = "token-jaccard",
    n: int = 2,
    lowercase: bool = True,
    prefix_weight: float = 0.1,
    match_threshold: float = 0.85,
    modify_threshold: float = 0.4,
    detect_moves: bool = False,
    move_distance_ratio: float = 0.1,
    tokenizer: PunktTokenizer | None = None,
) -> list[SegmentChange]:
    """Compute a segment-level diff between two documents.

    Args:
        a: Source text.
        b: Target text.
        granularity: One of ``"sentence"`` (default), ``"paragraph"``,
            ``"line"``, ``"paragraph_simple"``. ``"sentence"`` and
            ``"paragraph"`` use Punkt; the other two are model-free.
        algorithm: Similarity metric name. Same keys as
            :func:`kaos_nlp_core.algorithms.compare_batch`. Sensible
            defaults: ``"token-jaccard"`` for sentences/paragraphs,
            ``"jaro-winkler"`` for short lines.
        n: N-gram size for n-gram metrics.
        lowercase: Lowercase tokens for token-level metrics. Defaults to
            ``True`` here because diffing is usually case-insensitive.
        prefix_weight: Jaro-Winkler prefix factor.
        match_threshold: Score at or above which a pair is ``"unchanged"``
            (or ``"moved"`` when ``detect_moves=True``). Default 0.85.
        modify_threshold: Score floor for considering a pair a match at
            all. Pairs in ``[modify_threshold, match_threshold)`` are
            ``"modified"``. Default 0.4.
        detect_moves: Post-classify high-scoring matches whose normalized
            index distance exceeds ``move_distance_ratio`` as ``"moved"``.
            Default ``False``.
        move_distance_ratio: Index-shift threshold for ``"moved"``,
            expressed as a fraction of the longer document's segment count.
            Default 0.1 (10% of segments).
        tokenizer: Optional :class:`PunktTokenizer`. When ``None`` the
            Rust core falls back to the bundled legal model embedded at
            build time.

    Returns:
        List of :class:`SegmentChange`, sorted by left-segment index (with
        ``"added"`` rows interleaved by their right-segment index).
    """
    raw = _raw_diff_documents(
        a,
        b,
        granularity=granularity,
        algorithm=algorithm,
        n=n,
        lowercase=lowercase,
        prefix_weight=prefix_weight,
        match_threshold=match_threshold,
        modify_threshold=modify_threshold,
        detect_moves=detect_moves,
        move_distance_ratio=move_distance_ratio,
        tokenizer=tokenizer,
    )
    return [_to_change(c) for c in raw]


def summarize_changes(changes: Sequence[SegmentChange]) -> dict[str, int]:
    """Return a count per change kind. Useful for high-level reporting."""
    counts: dict[str, int] = {
        "unchanged": 0,
        "modified": 0,
        "moved": 0,
        "added": 0,
        "removed": 0,
    }
    for c in changes:
        counts[c.kind] = counts.get(c.kind, 0) + 1
    return counts


__all__ = [
    "ChangeKind",
    "Granularity",
    "SegmentChange",
    "SegmentRef",
    "diff_documents",
    "summarize_changes",
]
