"""Benchmarks for kaos_nlp_core.quality (pytest-benchmark).

Two purposes:

1. Track per-MB throughput of the Rust-backed analyzer so regressions
   are visible per release.
2. Compare against a pure-Python baseline that mirrors the pre-rewrite
   implementation, so the speed-up is auditable.

Run with::

    uv run pytest tests/bench_quality.py --benchmark-only
"""

from __future__ import annotations

import math
import string
from pathlib import Path

import pytest

from kaos_nlp_core.quality import (
    compute_metrics,
    default_english_wordset,
    quality_report,
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


@pytest.fixture(scope="module")
def english_wordset():
    return default_english_wordset()


# ─── Pure-Python baseline (mirrors the pre-rewrite implementation) ───────


def _python_compute_metrics(text: str) -> dict:
    total_chars = len(text)
    if total_chars == 0:
        return {"score": 0.0}

    whitespace_count = 0
    alpha_count = 0
    digit_count = 0
    capital_count = 0
    punctuation_count = 0
    non_ascii_count = 0
    line_count = 1
    paragraph_count = 0
    char_freq: dict[str, int] = {}

    punct_set = set(string.punctuation)

    i = 0
    while i < total_chars:
        c = text[i]
        char_freq[c] = char_freq.get(c, 0) + 1
        if c.isspace():
            whitespace_count += 1
            if c == "\n":
                line_count += 1
        if c.isalpha():
            alpha_count += 1
            if c.isupper():
                capital_count += 1
        elif c.isdigit():
            digit_count += 1
        if c in punct_set:
            punctuation_count += 1
        if ord(c) > 127:
            non_ascii_count += 1
        if c == "." and i + 2 < total_chars:
            pair = text[i + 1 : i + 3]
            if pair == "\r\n" or pair == "\n\n":
                paragraph_count += 1
        i += 1
    paragraph_count += 1

    words = text.split()
    num_words = len(words)
    if num_words:
        word_freq: dict[str, int] = {}
        total_word_len = 0
        for w in words:
            total_word_len += len(w)
            word_freq[w] = word_freq.get(w, 0) + 1
        unique_words = len(word_freq)
        token_entropy = 0.0
        for c in word_freq.values():
            p = c / num_words
            token_entropy -= p * math.log2(p)
    else:
        unique_words = 0
        token_entropy = 0.0
        total_word_len = 0

    char_entropy = 0.0
    for c in char_freq.values():
        p = c / total_chars
        char_entropy -= p * math.log2(p)

    return {
        "total_characters": total_chars,
        "whitespace": whitespace_count,
        "alpha": alpha_count,
        "digit": digit_count,
        "capital": capital_count,
        "punct": punctuation_count,
        "non_ascii": non_ascii_count,
        "line_count": line_count,
        "paragraph_count": paragraph_count,
        "char_entropy": char_entropy,
        "num_words": num_words,
        "unique_words": unique_words,
        "token_entropy": token_entropy,
        "total_word_chars": total_word_len,
    }


# ─── Benchmarks ───────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="quality_compute_metrics_no_lex")
def test_rust_compute_metrics_100k(benchmark, text_100k: str) -> None:
    benchmark(compute_metrics, text_100k)


@pytest.mark.benchmark(group="quality_compute_metrics_no_lex")
def test_python_baseline_100k(benchmark, text_100k: str) -> None:
    benchmark(_python_compute_metrics, text_100k)


@pytest.mark.benchmark(group="quality_compute_metrics_with_lex")
def test_rust_compute_metrics_with_lex_100k(benchmark, text_100k: str, english_wordset) -> None:
    benchmark(compute_metrics, text_100k, lexicon=english_wordset)


@pytest.mark.benchmark(group="quality_full_report")
def test_quality_report_100k(benchmark, text_100k: str) -> None:
    benchmark(quality_report, text_100k)


@pytest.mark.benchmark(group="quality_full_report_1m")
def test_quality_report_1m(benchmark, text_1m: str) -> None:
    benchmark(quality_report, text_1m)
