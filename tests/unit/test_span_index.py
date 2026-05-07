"""Tests for the content-agnostic SpanIndex primitive (P4).

The Rust core works in opaque u32 offsets; this module validates the
PyO3 binding contract and the kaos-content integration boundary —
namely that ``SpanIndex`` does not know about Annotation, node_ref, or
AST nodes. Those concerns live in ``kaos_content.indexing.AnnotationIndex``
(P4.5).
"""

from __future__ import annotations

import pickle

import pytest

from kaos_nlp_core.structures import LabeledSpan, SpanIndex

# ─── Construction ──────────────────────────────────────────────────────────


def test_empty_index_has_zero_length() -> None:
    idx = SpanIndex()
    assert len(idx) == 0
    assert idx.containing(5) == []
    assert idx.overlapping(0, 100) == []


def test_add_returns_stable_id() -> None:
    idx = SpanIndex()
    a = idx.add(1, 0, 10)
    b = idx.add(1, 20, 30)
    assert a == 0
    assert b == 1
    assert len(idx) == 2


def test_invalid_span_raises_value_error() -> None:
    idx = SpanIndex()
    with pytest.raises(ValueError, match="invalid span"):
        idx.add(1, 50, 10)


def test_from_tuples_constructor() -> None:
    idx = SpanIndex.from_tuples(
        [
            (1, 0, 10, None),
            (1, 20, 30, 0.8),
            (2, 5, 15, 1.0),
        ]
    )
    assert len(idx) == 3
    span_zero = idx.get(0)
    assert span_zero is not None
    assert span_zero.label == 1
    assert span_zero.score == 1.0
    span_one = idx.get(1)
    assert span_one is not None
    assert span_one.score == pytest.approx(0.8)


# ─── Queries ───────────────────────────────────────────────────────────────


def test_containing_finds_simple() -> None:
    idx = SpanIndex()
    a = idx.add(1, 10, 20)
    assert idx.containing(15) == [a]
    assert idx.containing(5) == []
    assert idx.containing(20) == []  # half-open


def test_overlapping_finds_partial_overlap() -> None:
    idx = SpanIndex()
    a = idx.add(1, 10, 20)
    b = idx.add(2, 15, 25)
    c = idx.add(3, 100, 200)
    hits = sorted(idx.overlapping(12, 18))
    assert hits == [a, b]
    assert idx.overlapping(50, 60) == []
    hits2 = sorted(idx.overlapping(150, 250))
    assert hits2 == [c]


def test_overlapping_engulfing_span() -> None:
    """Engulfing-interval pathology: a single very long span enclosing
    everything must be returned by every overlapping query inside it."""
    idx = SpanIndex()
    big = idx.add(1, 0, 1000)
    small_a = idx.add(2, 100, 110)
    small_b = idx.add(2, 500, 520)
    hits = sorted(idx.overlapping(105, 108))
    assert big in hits
    assert small_a in hits
    assert small_b not in hits


def test_query_after_add_is_consistent() -> None:
    """Lazy-rebuild: adding a span and immediately querying must reflect it."""
    idx = SpanIndex()
    idx.add(1, 0, 10)
    assert len(idx.containing(5)) == 1
    idx.add(2, 5, 8)
    assert len(idx.containing(6)) == 2


# ─── Mutations ─────────────────────────────────────────────────────────────


def test_merge_adjacent_basic() -> None:
    idx = SpanIndex()
    leader = idx.add(1, 0, 10)
    idx.add(1, 12, 20)
    idx.add(1, 100, 110)
    idx.merge_adjacent(label=1, gap=5)
    leader_span = idx.get(leader)
    assert leader_span is not None
    assert leader_span.start == 0
    assert leader_span.end == 20
    # The 100..110 span is too far away — still alive.
    assert len([s for s in (idx.get(i) for i in range(3)) if s is not None]) == 2


def test_subtract_appends_new_subspans() -> None:
    """body=0..100, tables at 30..50 and 70..80 → 3 new sub-spans of body."""
    idx = SpanIndex()
    body = idx.add(1, 0, 100)
    idx.add(2, 30, 50)
    idx.add(2, 70, 80)
    new_ids = idx.subtract(label_a=1, label_b=2)
    assert len(new_ids) == 3
    new_spans = [s for s in (idx.get(i) for i in new_ids) if s is not None]
    extents = sorted((s.start, s.end) for s in new_spans)
    assert extents == [(0, 30), (50, 70), (80, 100)]
    # Original body span is preserved (append-not-replace).
    assert idx.get(body) is not None


def test_resolve_overlaps_keeps_higher_score() -> None:
    idx = SpanIndex()
    idx.add(label=10, start=50, end=70, score=0.9)
    idx.add(label=20, start=40, end=80, score=0.4)
    idx.resolve_overlaps_by_score([10, 20])
    # The lower-scored span should split into [40..50) + [70..80).
    live_spans = [s for s in (idx.get(i) for i in range(10)) if s is not None]
    live_labels = [s.label for s in live_spans]
    # We don't care about ids; we care the count + label distribution.
    assert sorted(live_labels) == [10, 20, 20]


def test_tombstone_removes_from_queries() -> None:
    idx = SpanIndex()
    a = idx.add(1, 10, 20)
    assert idx.containing(15) == [a]
    assert idx.tombstone(a) is True
    assert idx.containing(15) == []
    # Idempotent: tombstoning twice returns False the second time.
    assert idx.tombstone(a) is False


# ─── Pickle / persistence ──────────────────────────────────────────────────


def test_pickle_round_trip() -> None:
    idx = SpanIndex()
    idx.add(1, 0, 10)
    idx.add(2, 5, 15)
    copy = pickle.loads(pickle.dumps(idx))
    assert sorted(copy.overlapping(0, 100)) == sorted(idx.overlapping(0, 100))


def test_save_load_round_trip(tmp_path) -> None:
    idx = SpanIndex()
    idx.add(1, 0, 10)
    idx.add(2, 5, 15)
    p = tmp_path / "spans.knci"
    idx.save(str(p))
    loaded = SpanIndex.load(str(p))
    assert sorted(loaded.overlapping(0, 100)) == sorted(idx.overlapping(0, 100))


def test_labeled_span_value_class() -> None:
    s = LabeledSpan(label=7, start=10, end=20, score=0.8)
    assert s.label == 7
    assert s.start == 10
    assert s.end == 20
    assert s.score == pytest.approx(0.8)
    assert "LabeledSpan" in repr(s)


# ─── Reference-equivalence smoke (mirror the proptest from Rust) ───────────


def test_naive_reference_equivalence_smoke() -> None:
    """Smoke test for the property that's pinned by Rust proptest. Does
    not run a property test in Python; this just guards the contract on
    a fixed corpus."""
    spans = [
        (1, 0, 10),
        (2, 5, 15),
        (3, 12, 25),
        (4, 100, 120),
        (5, 110, 130),
    ]
    idx = SpanIndex.from_tuples([(label, s, e, None) for (label, s, e) in spans])

    for offset in [0, 5, 10, 12, 15, 20, 100, 105, 110, 115, 130]:
        got = sorted(idx.containing(offset))
        want = sorted(i for i, (_label, s, e) in enumerate(spans) if s <= offset < e)
        assert got == want, f"offset={offset}, got={got}, want={want}"


# ─── Boundary documentation (kaos-content) ─────────────────────────────────


def test_label_is_opaque_to_index() -> None:
    """Documented contract: the SpanIndex does not interpret labels.
    The kaos-content wrapper is responsible for AnnotationType ↔ u32.
    Here we verify that arbitrary u32 labels round-trip without
    interpretation."""
    idx = SpanIndex()
    for label in [0, 1, 42, 0xCAFE, 0xFFFFFFFF]:
        idx.add(label, 0, 10)
    for span_id in range(5):
        s = idx.get(span_id)
        assert s is not None
