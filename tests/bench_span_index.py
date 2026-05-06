"""Python-side SpanIndex benchmarks (pytest-benchmark).

Two purposes:

1. FFI overhead — the same workloads as the criterion bench so we can
   compare bytes/sec at both layers.
2. Compare to a naive pure-Python implementation that does what
   ``SpanIndex.containing`` does without an interval tree (filter the
   list). The naive version is the same one used by the
   reference-equivalence proptest in the Rust test module — it's
   provably correct, just O(n) per query.

Run with:
    uv run pytest tests/bench_span_index.py --benchmark-only
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from kaos_nlp_core.structures import SpanIndex

# ─── Synthetic span generator (deterministic) ──────────────────────────────


class _Span(NamedTuple):
    label: int
    start: int
    end: int


def _synthetic_spans(n: int) -> list[_Span]:
    spans: list[_Span] = []
    rng = 0x12345678
    for _ in range(n):
        rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        start = (rng >> 1) % 1_000_000
        kind = (rng >> 33) & 0b11
        if kind == 0:
            length = 5 + (rng >> 7) % 15
        elif kind == 1:
            length = 50 + (rng >> 7) % 200
        elif kind == 2:
            length = 500 + (rng >> 7) % 1500
        else:
            length = 5_000 + (rng >> 7) % 20_000
        label = (rng >> 50) % 16
        spans.append(_Span(int(label), int(start), int(start + length)))
    return spans


def _python_containing(spans: list[_Span], offset: int) -> list[int]:
    """Naive O(n) reference. Same as the Rust proptest's `naive_containing`."""
    return [i for i, s in enumerate(spans) if s.start <= offset < s.end]


# ─── Benchmarks ────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="span_index/rust")
@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_bench_rust_bulk_build(benchmark, n: int) -> None:
    spans = _synthetic_spans(n)
    payload = [(s.label, s.start, s.end, None) for s in spans]
    benchmark.extra_info["n"] = n
    benchmark(SpanIndex.from_tuples, payload)


@pytest.mark.benchmark(group="span_index/rust")
@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_bench_rust_containing(benchmark, n: int) -> None:
    spans = _synthetic_spans(n)
    idx = SpanIndex.from_tuples([(s.label, s.start, s.end, None) for s in spans])
    idx.freeze()
    benchmark.extra_info["n"] = n

    counter = {"i": 0}

    def query() -> list[int]:
        counter["i"] = (counter["i"] + 7919) % 1_000_000
        return idx.containing(counter["i"])

    benchmark(query)


@pytest.mark.benchmark(group="span_index/rust")
@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_bench_rust_overlapping(benchmark, n: int) -> None:
    spans = _synthetic_spans(n)
    idx = SpanIndex.from_tuples([(s.label, s.start, s.end, None) for s in spans])
    idx.freeze()
    benchmark.extra_info["n"] = n

    counter = {"i": 0}

    def query() -> list[int]:
        counter["i"] = (counter["i"] + 31337) % 1_000_000
        return idx.overlapping(counter["i"], counter["i"] + 1_000)

    benchmark(query)


@pytest.mark.benchmark(group="span_index/python")
@pytest.mark.parametrize("n", [1_000, 10_000])
def test_bench_python_naive_containing(benchmark, n: int) -> None:
    spans = _synthetic_spans(n)
    benchmark.extra_info["n"] = n

    counter = {"i": 0}

    def query() -> list[int]:
        counter["i"] = (counter["i"] + 7919) % 1_000_000
        return _python_containing(spans, counter["i"])

    benchmark(query)
