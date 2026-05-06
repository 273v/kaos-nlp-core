"""Benchmarks for kaos_nlp_core.vocabulary (pytest-benchmark).

Measures the FrequencyVocabulary-backed `token_frequency()` against a
pure-Python ``Counter`` baseline so the speed-up is auditable per
release.

Run with::

    uv run pytest tests/bench_vocabulary.py --benchmark-only --benchmark-disable-gc
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from kaos_nlp_core.quality import default_english_wordset
from kaos_nlp_core.vocabulary import token_frequency

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_text(target_bytes: int) -> str:
    pool: list[str] = []
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


def _python_counter(text: str) -> dict[str, int]:
    """Naïve baseline: split + Counter, no lexicon filter."""
    return dict(Counter(text.lower().split()))


def _python_counter_filtered(text: str, vocab: set[str]) -> dict[str, int]:
    return dict(Counter(t for t in text.lower().split() if t in vocab))


@pytest.mark.benchmark(group="vocab_no_lex")
def test_rust_no_lex_100k(benchmark, text_100k: str) -> None:
    benchmark(token_frequency, text_100k, lowercase=True)


@pytest.mark.benchmark(group="vocab_no_lex")
def test_python_no_lex_100k(benchmark, text_100k: str) -> None:
    benchmark(_python_counter, text_100k)


@pytest.mark.benchmark(group="vocab_with_lex")
def test_rust_with_lex_100k(benchmark, text_100k: str, english_wordset) -> None:
    benchmark(token_frequency, text_100k, lexicon=english_wordset, lowercase=True)


@pytest.mark.benchmark(group="vocab_1m")
def test_rust_1m(benchmark, text_1m: str, english_wordset) -> None:
    benchmark(token_frequency, text_1m, lexicon=english_wordset)
