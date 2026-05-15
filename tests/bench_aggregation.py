"""Throughput benchmarks for :mod:`kaos_nlp_core.aggregation`.

Compares the new Rust-backed primitives (via the production Python
wrappers in ``kaos_nlp_core.aggregation``) against the pure-Python
baseline that lived in the module before the Rust kernels landed.
The baseline is reproduced inline below so we can run both on
identical inputs after the production path has migrated to Rust.

Run with::

    KNC_BENCH_PRINT=1 uv run --no-sync pytest tests/bench_aggregation.py \
        --no-cov -s

Workload shapes target the chunk counts a ``ChunkedClassify`` or
``EnsembleClassify`` call actually produces:

* ``n_chunks=10``    — small contract, few-shot classification.
* ``n_chunks=100``   — a full 10-K or large filing.
* ``n_chunks=1_000`` — long-form / multi-document aggregation.

Each chunk picks ``picks_per_chunk`` labels from a vocabulary of
``vocab_size`` distinct names. Tie-rate is mild — every chunk's picks
are shifted modulo the vocabulary so no single label dominates.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pytest

from kaos_nlp_core import aggregation as agg

# ---------------------------------------------------------------------------
# Pure-Python baseline — frozen copy of the pre-Rust implementation.
# Lives here (and ONLY here) for bench A/B comparison.
# ---------------------------------------------------------------------------


def py_vote(per_chunk: Sequence[Iterable[str]]) -> str | None:
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for chunk_labels in per_chunk:
        for name in dict.fromkeys(chunk_labels):
            counts[name] = counts.get(name, 0) + 1
            if name not in first_seen:
                first_seen[name] = order
                order += 1
    if not counts:
        return None
    max_count = max(counts.values())
    candidates = [name for name, count in counts.items() if count == max_count]
    return min(candidates, key=lambda n: first_seen[n])


def py_majority(per_chunk: Sequence[Iterable[str]], *, threshold: float = 0.5) -> str | None:
    n = sum(1 for _ in per_chunk)
    if n == 0:
        return None
    required = threshold * n
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for chunk_labels in per_chunk:
        for name in dict.fromkeys(chunk_labels):
            counts[name] = counts.get(name, 0) + 1
            if name not in first_seen:
                first_seen[name] = order
                order += 1
    candidates = [name for name, count in counts.items() if count >= required]
    if not candidates:
        return None
    return min(candidates, key=lambda n: first_seen[n])


def py_union(per_chunk: Sequence[Iterable[str]]) -> frozenset[str]:
    out: set[str] = set()
    for chunk_labels in per_chunk:
        out.update(chunk_labels)
    return frozenset(out)


def py_intersection(per_chunk: Sequence[Iterable[str]]) -> frozenset[str]:
    materialized = [set(chunk_labels) for chunk_labels in per_chunk]
    if not materialized:
        return frozenset()
    result = materialized[0].copy()
    for chunk_labels in materialized[1:]:
        result &= chunk_labels
    return frozenset(result)


def py_weighted(
    per_chunk: Sequence[Iterable[str]],
    *,
    weights: Sequence[float] | None = None,
    threshold: float = 0.5,
    multi: bool = False,
):
    chunks = list(per_chunk)
    n = len(chunks)
    if n == 0:
        return frozenset() if multi else None
    applied_weights = [1.0] * n if weights is None else list(weights)
    total = sum(applied_weights)
    if total <= 0:
        return frozenset() if multi else None
    required = threshold * total
    score: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for chunk_index, chunk_labels in enumerate(chunks):
        chunk_weight = applied_weights[chunk_index]
        for name in dict.fromkeys(chunk_labels):
            score[name] = score.get(name, 0.0) + chunk_weight
            if name not in first_seen:
                first_seen[name] = order
                order += 1
    winners = [name for name, total_weight in score.items() if total_weight >= required]
    if multi:
        return frozenset(winners)
    if not winners:
        return None
    max_weight = max(score[name] for name in winners)
    top = [name for name in winners if score[name] == max_weight]
    return min(top, key=lambda n: first_seen[n])


def py_max_score(per_chunk_scores, *, multi: bool = False, threshold=None):
    pooled: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for chunk_map in per_chunk_scores:
        for name, value in chunk_map.items():
            if name not in pooled or value > pooled[name]:
                pooled[name] = value
            if name not in first_seen:
                first_seen[name] = order
                order += 1
    if not pooled:
        return frozenset() if multi else None
    if multi:
        cutoff = threshold if threshold is not None else 0.0
        winners = [name for name, value in pooled.items() if value > cutoff]
        return frozenset(winners)
    top_score = max(pooled.values())
    if threshold is not None and top_score < threshold:
        return None
    top = [name for name, value in pooled.items() if value == top_score]
    return min(top, key=lambda n: first_seen[n])


# ---------------------------------------------------------------------------
# Workload generators.
# ---------------------------------------------------------------------------


def _corpus(n_chunks: int, picks_per_chunk: int, vocab_size: int):
    labels = [f"L{i:03d}" for i in range(vocab_size)]
    per_chunk = [
        [labels[(c * picks_per_chunk + i) % vocab_size] for i in range(picks_per_chunk)]
        for c in range(n_chunks)
    ]
    per_chunk_scores = [
        {
            labels[(c * picks_per_chunk + i) % vocab_size]: 0.5 + 0.1 * i
            for i in range(picks_per_chunk)
        }
        for c in range(n_chunks)
    ]
    return per_chunk, per_chunk_scores


SHAPES = [
    (10, 5, 30),
    (100, 5, 30),
    (1_000, 5, 30),
]


def _bench(fn, *args, iters: int = 200) -> float:
    """Median wall-clock seconds across ``iters`` calls."""
    for _ in range(3):
        fn(*args)
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


PRIMITIVES: list[tuple[str, Callable[..., object], Callable[..., object]]] = [
    ("vote", agg.vote, py_vote),
    ("majority", agg.majority, py_majority),
    ("union", agg.union, py_union),
    ("intersection", agg.intersection, py_intersection),
    ("weighted", agg.weighted, py_weighted),
]


@pytest.mark.parametrize(("n_chunks", "picks_per_chunk", "vocab_size"), SHAPES)
def test_aggregation_speedup(n_chunks: int, picks_per_chunk: int, vocab_size: int) -> None:
    per_chunk, per_chunk_scores = _corpus(n_chunks, picks_per_chunk, vocab_size)

    results: list[dict[str, object]] = []
    for name, rust_fn, py_fn in PRIMITIVES:
        rust_out = rust_fn(per_chunk)
        py_out = py_fn(per_chunk)
        # frozenset equality and str equality both work directly.
        assert rust_out == py_out, f"{name} divergence: rust={rust_out!r} py={py_out!r}"
        rust_t = _bench(rust_fn, per_chunk)
        py_t = _bench(py_fn, per_chunk)
        speedup = py_t / rust_t if rust_t > 0 else float("inf")
        results.append(
            {
                "primitive": name,
                "n_chunks": n_chunks,
                "rust_seconds": rust_t,
                "python_seconds": py_t,
                "speedup": speedup,
            }
        )
        if os.environ.get("KNC_BENCH_PRINT"):
            print(
                f"  {name:12s} n={n_chunks:>5} "
                f"rust={rust_t * 1e6:8.2f}us "
                f"python={py_t * 1e6:8.2f}us speedup={speedup:5.2f}x"
            )

    # max_score has a different input shape (score maps). Bench separately.
    rust_out = agg.max_score(per_chunk_scores)
    py_out = py_max_score(per_chunk_scores)
    assert rust_out == py_out
    rust_t = _bench(agg.max_score, per_chunk_scores)
    py_t = _bench(py_max_score, per_chunk_scores)
    speedup = py_t / rust_t if rust_t > 0 else float("inf")
    results.append(
        {
            "primitive": "max_score",
            "n_chunks": n_chunks,
            "rust_seconds": rust_t,
            "python_seconds": py_t,
            "speedup": speedup,
        }
    )
    if os.environ.get("KNC_BENCH_PRINT"):
        print(
            f"  {'max_score':12s} n={n_chunks:>5} "
            f"rust={rust_t * 1e6:8.2f}us "
            f"python={py_t * 1e6:8.2f}us speedup={speedup:5.2f}x"
        )

    report_path = (
        Path(__file__).parents[1] / "docs" / "benchmarks" / "aggregation-rust-vs-python.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []
    if not isinstance(existing, list):
        existing = []
    existing = [r for r in existing if r.get("n_chunks") != n_chunks]
    existing.extend(results)
    report_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
