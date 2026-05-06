"""Python-side enumerator-parser benchmarks (pytest-benchmark).

Two purposes:

1. FFI overhead — measure each kind through the PyO3 boundary so the
   per-release cost is visible.
2. Compare to a pure-Python regex baseline that does the same job. The
   baseline is intentionally simple (one regex per kind, first match wins)
   so the speedup is honest, not contrived.

Run with:
    uv run pytest tests/bench_enumerators.py --benchmark-only
"""

from __future__ import annotations

import re

import pytest

from kaos_nlp_core.segmentation import parse_enumerator

# ─── Pure-Python regex baseline ─────────────────────────────────────────────
#
# Approximates what a careful Python-only implementation would do. We compile
# the patterns once at module load and try them in priority order — same as
# the Rust parser's first-match-wins ordering.

_DECIMAL = re.compile(r"^(\d+)(\.\d+){0,3}\.?")
_ROMAN_LOWER = re.compile(r"^(?:i{1,3}|iv|v?i{0,3}|ix|x{1,3})\.")
_ROMAN_UPPER = re.compile(r"^(?:I{1,3}|IV|V?I{0,3}|IX|X{1,3})\.")
_ALPHA = re.compile(r"^[A-Za-z]\.")
_PAREN = re.compile(r"^\(([A-Za-z]|\d+|[ivxIVX]+)\)")
_SECTION = re.compile(
    r"^(?:Section|Sec\.|Chapter|Subpart|Subchapter|Title|Subtitle|Part|Appendix|Schedule|Article|Paragraph|§)\s+\S+",
    re.IGNORECASE,
)


def _python_baseline(line: str) -> str | None:
    """Return a kind label or None — same shape as parse_enumerator(line).kind."""
    if not line:
        return None
    if _SECTION.match(line):
        return "section_word"
    if _PAREN.match(line):
        return "paren"
    m = _DECIMAL.match(line)
    if m and (m.group(0).endswith(".") or "." in m.group(0)[:-1]):
        return "decimal"
    if _ROMAN_UPPER.match(line):
        return "roman_upper"
    if _ROMAN_LOWER.match(line):
        return "roman_lower"
    if _ALPHA.match(line):
        return "alpha"
    return None


# ─── Fixture corpus ─────────────────────────────────────────────────────────


_MIXED_LINES = [
    "1. Introduction",
    "1.2 Definitions",
    "1.2.3 Subitem",
    "(a) item",
    "(1) item",
    "(iv) note",
    "I. Background",
    "II. Discussion",
    "Section 5 Title",
    "Sec. 5.2 Heading",
    "§ 12.2 Notices",
    "Chapter 7 Title",
    "Subpart B Filings",
    "Article III Authority",
    "The Borrower hereby agrees to repay all sums advanced.",
    "All payments shall be made in lawful currency of the United States.",
    "Whereas, the parties enter into this agreement on the date written above.",
]


def _scan_lines(n: int) -> list[str]:
    return [_MIXED_LINES[i % len(_MIXED_LINES)] for i in range(n)]


# ─── Benchmarks ─────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="enumerator/rust")
@pytest.mark.parametrize(
    "src",
    [
        "1. Introduction",
        "(a) item",
        "I. Background",
        "Section 5 Title",
        "The Borrower hereby agrees.",
    ],
    ids=["decimal", "paren_alpha", "roman", "section_word", "no_match"],
)
def test_bench_rust_single_call(benchmark, src: str) -> None:
    benchmark(parse_enumerator, src)


@pytest.mark.benchmark(group="enumerator/python")
@pytest.mark.parametrize(
    "src",
    [
        "1. Introduction",
        "(a) item",
        "I. Background",
        "Section 5 Title",
        "The Borrower hereby agrees.",
    ],
    ids=["decimal", "paren_alpha", "roman", "section_word", "no_match"],
)
def test_bench_python_single_call(benchmark, src: str) -> None:
    benchmark(_python_baseline, src)


@pytest.mark.benchmark(group="enumerator/rust")
@pytest.mark.parametrize("n", [100, 1_000, 10_000])
def test_bench_rust_scan(benchmark, n: int) -> None:
    lines = _scan_lines(n)
    benchmark.extra_info["lines"] = n
    benchmark(lambda: [parse_enumerator(line) for line in lines])


@pytest.mark.benchmark(group="enumerator/python")
@pytest.mark.parametrize("n", [100, 1_000, 10_000])
def test_bench_python_scan(benchmark, n: int) -> None:
    lines = _scan_lines(n)
    benchmark.extra_info["lines"] = n
    benchmark(lambda: [_python_baseline(line) for line in lines])
