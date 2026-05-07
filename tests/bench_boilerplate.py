"""Python-side boilerplate-detector benchmarks (pytest-benchmark).

Two purposes:

1. FFI overhead — same workloads as the criterion bench so we can compare
   bytes/sec at both layers.
2. Python baseline using ``collections.Counter`` over normalized lines so
   the speedup is visible per release. The baseline is intentionally
   simplified (no MinHash, no zone classification) — it answers "how
   much does the Python version cost just for the exact-dup tier?" so
   the comparison is honest.

Run with:
    uv run pytest tests/bench_boilerplate.py --benchmark-only
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from kaos_nlp_core.segmentation import detect_boilerplate

_WS_RUN = re.compile(r"\s+")


# ─── Synthetic corpora ──────────────────────────────────────────────────────


def _synthetic_clean(n_pages: int) -> str:
    parts: list[str] = []
    for i in range(n_pages):
        parts.append("FILED 5/5/2026 SMITH V JONES — PAGE BANNER\n")
        parts.append("Body content paragraph one.\n")
        parts.append(f"Body content paragraph two on page {i}.\n")
        parts.append("Body content paragraph three.\n")
        parts.append("CONFIDENTIAL — INTERNAL ONLY\n")
        if i + 1 < n_pages:
            parts.append("\f")
    return "".join(parts)


def _synthetic_ocr_drift(n_pages: int) -> str:
    template = "FILED IN COURT 5_5_2026 SMITH V JONES — PAGE BANNER"
    parts: list[str] = []
    for i in range(n_pages):
        # Rotate a single-character corruption position each page.
        pos = 1 + i % (len(template) - 1)
        chars = list(template)
        chars[pos] = "1"
        parts.append("".join(chars))
        parts.append("\n")
        parts.append("Body content paragraph one.\n")
        parts.append(f"Body content paragraph two on page {i}.\n")
        parts.append("Body content paragraph three.\n")
        parts.append("CONFIDENTIAL — INTERNAL ONLY\n")
        if i + 1 < n_pages:
            parts.append("\f")
    return "".join(parts)


# ─── Pure-Python baseline (exact-dup tier only) ─────────────────────────────


def _python_baseline_exact(text: str) -> list[tuple[str, int]]:
    """Bucket lines by normalized form and emit clusters with ≥ 3 occurrences."""
    counter: Counter[str] = Counter()
    for line in text.splitlines():
        # Crude normalize: collapse whitespace, casefold, normalize ASCII quotes.
        canon = _WS_RUN.sub(" ", line.strip()).casefold()
        if not canon:
            continue
        counter[canon] += 1
    return [(c, n) for c, n in counter.items() if n >= 3]


# ─── Benchmarks ─────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="boilerplate/rust")
@pytest.mark.parametrize("pages", [10, 100, 500])
def test_bench_rust_clean(benchmark, pages: int) -> None:
    text = _synthetic_clean(pages)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    benchmark.extra_info["pages"] = pages
    result = benchmark(detect_boilerplate, text)
    assert result


@pytest.mark.benchmark(group="boilerplate/rust")
@pytest.mark.parametrize("pages", [10, 100])
def test_bench_rust_ocr_drift(benchmark, pages: int) -> None:
    text = _synthetic_ocr_drift(pages)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    benchmark.extra_info["pages"] = pages
    result = benchmark(detect_boilerplate, text)
    assert result


@pytest.mark.benchmark(group="boilerplate/rust")
@pytest.mark.parametrize("pages", [10, 100])
def test_bench_rust_ocr_drift_skip_near_dup(benchmark, pages: int) -> None:
    text = _synthetic_ocr_drift(pages)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    benchmark.extra_info["pages"] = pages
    benchmark(detect_boilerplate, text, skip_near_dup=True)


@pytest.mark.benchmark(group="boilerplate/python")
@pytest.mark.parametrize("pages", [10, 100, 500])
def test_bench_python_baseline_clean(benchmark, pages: int) -> None:
    text = _synthetic_clean(pages)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    result = benchmark(_python_baseline_exact, text)
    assert result
