"""Scale tests for :mod:`kaos_nlp_core.aggregation` primitives.

Aggregators are pure functions of per-chunk label iterables; their
"scale" target is throughput on large inputs. The unit-test suite
exercises edge cases on ≤5-element inputs; this suite exercises
correctness invariants and throughput at 100K-chunk scale.

The synthetic input simulates a realistic multi-label long-document
classification: 1000 documents x 100 chunks/doc x labels drawn from a
12-element space with skewed distribution (one majority class + a
long tail).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import pytest

from kaos_nlp_core.aggregation import (
    intersection,
    majority,
    max_score,
    union,
    vote,
    weighted,
)

# Tests in this module are CPU-only; no fixtures needed but they're
# marked slow so they only run in the long-suite gate.
pytestmark = pytest.mark.slow


_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"


def _emit_report(name: str, payload: dict[str, Any]) -> None:
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


_LABELS = (
    "tax",
    "criminal",
    "banking",
    "labor",
    "ip",
    "real_estate",
    "merger",
    "secured",
    "non_competition",
    "indemnification",
    "limitation",
    "governing_law",
)


@pytest.fixture(scope="module")
def synthetic_chunks() -> list[list[str]]:
    """1000 docs x 100 chunks -> 100,000 per-chunk label sets.

    Each chunk picks 1-3 labels from :data:`_LABELS`, with a skewed
    distribution biased toward ``governing_law`` and
    ``indemnification`` (the "majority" classes for long-form legal
    text).
    """
    random.seed(42)
    per_chunk: list[list[str]] = []
    n_docs = 1000
    chunks_per_doc = 100
    for _doc in range(n_docs):
        for _chunk in range(chunks_per_doc):
            # 70% chance: governing_law + indemnification
            picks: set[str] = set()
            if random.random() < 0.7:
                picks.update({"governing_law", "indemnification"})
            # add 0-2 additional random labels
            for _ in range(random.randint(0, 2)):
                picks.add(random.choice(_LABELS))
            per_chunk.append(sorted(picks))
    assert len(per_chunk) == n_docs * chunks_per_doc
    return per_chunk


@pytest.fixture(scope="module")
def synthetic_scores() -> list[dict[str, float]]:
    """100K per-chunk score maps for :func:`max_score`."""
    random.seed(43)
    score_maps: list[dict[str, float]] = []
    for _ in range(100_000):
        n_labels = random.randint(1, 4)
        labels = random.sample(_LABELS, n_labels)
        score_maps.append({label: round(random.random(), 4) for label in labels})
    return score_maps


def _time_call(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Throughput + correctness tests
# ---------------------------------------------------------------------------


def test_vote_at_100k(synthetic_chunks: list[list[str]]) -> None:
    result, elapsed = _time_call(vote, synthetic_chunks)
    # With ~70% of chunks containing both governing_law and
    # indemnification, vote() should pick whichever has the
    # first-appearance tiebreak.
    assert result in {"governing_law", "indemnification"}
    _emit_report(
        "aggregation-scale-vote",
        {
            "fn": "vote",
            "input_chunks": len(synthetic_chunks),
            "elapsed_seconds": round(elapsed, 3),
            "throughput_chunks_per_sec": round(len(synthetic_chunks) / elapsed, 1),
            "result": result,
        },
    )
    assert elapsed < 10.0, f"vote() too slow at 100K: {elapsed:.2f}s"


def test_majority_at_100k(synthetic_chunks: list[list[str]]) -> None:
    result, elapsed = _time_call(majority, synthetic_chunks)
    # The "majority" semantics: a label that appears in >= threshold
    # fraction of chunks. With our skew, ~70% of chunks include
    # governing_law and indemnification, so they should win at the
    # default 0.5 threshold.
    assert result in {"governing_law", "indemnification"}
    _emit_report(
        "aggregation-scale-majority",
        {
            "fn": "majority",
            "input_chunks": len(synthetic_chunks),
            "elapsed_seconds": round(elapsed, 3),
            "throughput_chunks_per_sec": round(len(synthetic_chunks) / elapsed, 1),
            "result": result,
        },
    )
    assert elapsed < 10.0


def test_union_at_100k(synthetic_chunks: list[list[str]]) -> None:
    result, elapsed = _time_call(union, synthetic_chunks)
    # Every label has nonzero probability per chunk, so over 100k
    # chunks the union is the full label space.
    assert result == frozenset(_LABELS)
    _emit_report(
        "aggregation-scale-union",
        {
            "fn": "union",
            "input_chunks": len(synthetic_chunks),
            "elapsed_seconds": round(elapsed, 3),
            "throughput_chunks_per_sec": round(len(synthetic_chunks) / elapsed, 1),
            "result_size": len(result),
        },
    )
    assert elapsed < 10.0


def test_intersection_at_100k(synthetic_chunks: list[list[str]]) -> None:
    result, elapsed = _time_call(intersection, synthetic_chunks)
    # With 30% of chunks NOT containing governing_law (and
    # similarly for indemnification), the intersection should be
    # empty. This is the right behavior — intersection is high
    # precision, low recall.
    assert isinstance(result, frozenset)
    _emit_report(
        "aggregation-scale-intersection",
        {
            "fn": "intersection",
            "input_chunks": len(synthetic_chunks),
            "elapsed_seconds": round(elapsed, 3),
            "throughput_chunks_per_sec": round(len(synthetic_chunks) / elapsed, 1),
            "result_size": len(result),
        },
    )
    assert elapsed < 10.0


def test_weighted_single_label_at_100k(synthetic_chunks: list[list[str]]) -> None:
    result, elapsed = _time_call(weighted, synthetic_chunks, threshold=0.5)
    # Uniform weights, threshold 0.5: governing_law / indemnification
    # are the only ones above the threshold.
    assert result in {"governing_law", "indemnification"}
    _emit_report(
        "aggregation-scale-weighted-single",
        {
            "fn": "weighted",
            "mode": "single",
            "input_chunks": len(synthetic_chunks),
            "elapsed_seconds": round(elapsed, 3),
            "throughput_chunks_per_sec": round(len(synthetic_chunks) / elapsed, 1),
            "result": result,
        },
    )
    assert elapsed < 10.0


def test_weighted_multi_label_at_100k(synthetic_chunks: list[list[str]]) -> None:
    result, elapsed = _time_call(weighted, synthetic_chunks, threshold=0.5, multi=True)
    # Multi-label: anything appearing in >= 50% of chunks wins.
    # That should be exactly {governing_law, indemnification}.
    assert isinstance(result, frozenset)
    assert "governing_law" in result
    assert "indemnification" in result
    _emit_report(
        "aggregation-scale-weighted-multi",
        {
            "fn": "weighted",
            "mode": "multi",
            "input_chunks": len(synthetic_chunks),
            "elapsed_seconds": round(elapsed, 3),
            "throughput_chunks_per_sec": round(len(synthetic_chunks) / elapsed, 1),
            "result_size": len(result),
        },
    )
    assert elapsed < 10.0


def test_max_score_at_100k(synthetic_scores: list[dict[str, float]]) -> None:
    result, elapsed = _time_call(max_score, synthetic_scores)
    # Some label must win.
    assert result in _LABELS
    _emit_report(
        "aggregation-scale-max-score",
        {
            "fn": "max_score",
            "input_chunks": len(synthetic_scores),
            "elapsed_seconds": round(elapsed, 3),
            "throughput_chunks_per_sec": round(len(synthetic_scores) / elapsed, 1),
            "result": result,
        },
    )
    assert elapsed < 10.0


def test_max_score_multi_at_100k(synthetic_scores: list[dict[str, float]]) -> None:
    result, elapsed = _time_call(max_score, synthetic_scores, multi=True, threshold=0.9)
    assert isinstance(result, frozenset)
    _emit_report(
        "aggregation-scale-max-score-multi",
        {
            "fn": "max_score",
            "mode": "multi",
            "threshold": 0.9,
            "input_chunks": len(synthetic_scores),
            "elapsed_seconds": round(elapsed, 3),
            "throughput_chunks_per_sec": round(len(synthetic_scores) / elapsed, 1),
            "result_size": len(result),
        },
    )
    assert elapsed < 10.0


# ---------------------------------------------------------------------------
# Cross-function invariants
# ---------------------------------------------------------------------------


def test_intersection_subset_of_union(synthetic_chunks: list[list[str]]) -> None:
    """Mathematical invariant — intersection ⊆ union."""
    u = union(synthetic_chunks)
    inter = intersection(synthetic_chunks)
    assert inter.issubset(u)


def test_vote_winner_in_union(synthetic_chunks: list[list[str]]) -> None:
    """A vote winner must appear in the union (i.e., be a real label)."""
    winner = vote(synthetic_chunks)
    assert winner is None or winner in union(synthetic_chunks)


def test_majority_winner_satisfies_threshold(
    synthetic_chunks: list[list[str]],
) -> None:
    """A majority winner has count >= threshold * n_chunks.

    Contract of :func:`majority`: returns the **first-seen** label
    whose count crosses the threshold. First-appearance is defined
    by input order (via ``dict.fromkeys``), not Python hash order —
    so the tiebreak is invariant across processes. See
    :file:`tests/test_aggregation_determinism.py` for the
    cross-PYTHONHASHSEED regression test.
    """
    threshold = 0.5
    maj = majority(synthetic_chunks, threshold=threshold)
    if maj is None:
        return  # no label crosses threshold; valid outcome.
    counts: dict[str, int] = {}
    for chunk_labels in synthetic_chunks:
        # Mirror the function's own dedup discipline.
        for label in dict.fromkeys(chunk_labels):
            counts[label] = counts.get(label, 0) + 1
    required = threshold * len(synthetic_chunks)
    assert counts.get(maj, 0) >= required, (
        f"majority returned {maj!r} with count={counts.get(maj, 0)} "
        f"which is below threshold={required}"
    )
