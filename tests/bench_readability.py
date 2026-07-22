"""Benchmarks for kaos_nlp_core.readability (pytest-benchmark).

Tracks per-MB throughput of the Rust-backed counting pass (with and
without the CMUdict syllable map) and the syllable kernel itself, so
regressions are visible per release.

Run with::

    uv run pytest tests/bench_readability.py --benchmark-only
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaos_nlp_core.readability import (
    compute_counts,
    readability_report,
    syllable_count,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ─── Corpora ──────────────────────────────────────────────────────────────


def _load_text(target_bytes: int) -> str:
    """Load a fixture corpus and tile it up to target_bytes."""
    pool = []
    for name in ("shakespeare.txt", "war_and_peace.txt"):
        path = FIXTURES / name
        if path.exists():
            pool.append(path.read_text(encoding="utf-8", errors="replace"))
    if not pool:
        pytest.skip("no Gutenberg fixtures available")
    base = "\n\n".join(pool)
    if len(base) >= target_bytes:
        return base[:target_bytes]
    multiplier = (target_bytes // len(base)) + 1
    return (base * multiplier)[:target_bytes]


@pytest.fixture(scope="module")
def text_100k() -> str:
    return _load_text(100_000)


@pytest.fixture(scope="module")
def text_1m() -> str:
    return _load_text(1_000_000)


# ─── Benchmarks ───────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="readability-counts")
def test_compute_counts_100k(benchmark, text_100k: str) -> None:
    benchmark(compute_counts, text_100k)


@pytest.mark.benchmark(group="readability-counts")
def test_compute_counts_1m(benchmark, text_1m: str) -> None:
    benchmark(compute_counts, text_1m)


@pytest.mark.benchmark(group="readability-counts")
def test_compute_counts_1m_heuristic_only(benchmark, text_1m: str) -> None:
    benchmark(lambda t: compute_counts(t, use_syllable_map=False), text_1m)


@pytest.mark.benchmark(group="readability-report")
def test_readability_report_1m(benchmark, text_1m: str) -> None:
    benchmark(readability_report, text_1m)


@pytest.mark.benchmark(group="syllable-kernel")
def test_syllable_count_heuristic(benchmark) -> None:
    words = [
        "the",
        "extraordinary",
        "state-of-the-art",
        "bureaucracy",
        "internationalization",
        "don't",
        "cat",
    ]

    def run() -> int:
        return sum(syllable_count(w, use_syllable_map=False) for w in words)

    benchmark(run)
