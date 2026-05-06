"""Benchmarks for the tokenizer module (pytest-benchmark).

Measures tokenization speed across:
- ASCII vs Unicode text
- Various config options (lowercase, prefix, stopwords)
- Short text vs long text (War and Peace)
- Whitespace scan vs regex methods
"""

import pytest

from kaos_nlp_core.tokenizer import Tokenizer, tokenize_words

SHORT_ASCII = "The quick brown fox jumps over the lazy dog and the cat sat on the mat"
SHORT_UNICODE = "Le café résumé «hello» — don't worry about the naïve approach to München"
LEGAL_TEXT = (
    "WHEREAS, the Company and the Employee desire to enter into this Employment "
    'Agreement (the "Agreement") effective as of the date set forth above; and '
    "WHEREAS, the Employee acknowledges that the Company's Confidential Information "
    "constitutes valuable, special, and unique property of the Company; NOW, THEREFORE, "
    "in consideration of the mutual covenants and agreements herein contained, and for "
    "other good and valuable consideration, the receipt and sufficiency of which are "
    "hereby acknowledged, the parties agree as follows: Section 1. Employment."
)


# ── Basic tokenization benchmarks ────────────────────────────────────────────


@pytest.mark.benchmark(group="tokenizer_basic")
def test_tokenize_ascii_short(benchmark):
    benchmark(tokenize_words, SHORT_ASCII)


@pytest.mark.benchmark(group="tokenizer_basic")
def test_tokenize_unicode_short(benchmark):
    benchmark(tokenize_words, SHORT_UNICODE)


@pytest.mark.benchmark(group="tokenizer_basic")
def test_tokenize_legal(benchmark):
    benchmark(tokenize_words, LEGAL_TEXT)


@pytest.mark.benchmark(group="tokenizer_basic")
def test_tokenize_lowercase(benchmark):
    benchmark(tokenize_words, SHORT_ASCII, True)


@pytest.mark.benchmark(group="tokenizer_basic")
def test_tokenize_prefix4(benchmark):
    benchmark(tokenize_words, SHORT_ASCII, True, 4)


# ── Tokenizer class benchmarks ──────────────────────────────────────────────


@pytest.mark.benchmark(group="tokenizer_class")
def test_tokenizer_default(benchmark):
    tok = Tokenizer()
    benchmark(tok.tokenize_words, SHORT_ASCII)


@pytest.mark.benchmark(group="tokenizer_class")
def test_tokenizer_full_config(benchmark):
    tok = Tokenizer(
        lowercase=True,
        prefix=4,
        stopwords=["the", "a", "an", "and", "of", "in", "to", "for", "on", "with"],
    )
    benchmark(tok.tokenize_words, SHORT_ASCII)


@pytest.mark.benchmark(group="tokenizer_class")
def test_tokenizer_regex(benchmark):
    tok = Tokenizer(lowercase=True)
    benchmark(tok.tokenize_regex_words, SHORT_ASCII)


# ── Long text benchmarks (War and Peace) ─────────────────────────────────────


@pytest.mark.benchmark(group="tokenizer_long")
def test_tokenize_war_peace(benchmark, war_and_peace_text):
    """Tokenize entire War and Peace (3.2MB)."""
    tok = Tokenizer(lowercase=True)
    result = benchmark.pedantic(
        tok.tokenize_words, args=(war_and_peace_text,), rounds=5, iterations=1
    )
    assert len(result) > 500000


@pytest.mark.benchmark(group="tokenizer_long")
def test_tokenize_war_peace_prefix4(benchmark, war_and_peace_text):
    """Tokenize War and Peace with prefix=4."""
    tok = Tokenizer(lowercase=True, prefix=4)
    result = benchmark.pedantic(
        tok.tokenize_words, args=(war_and_peace_text,), rounds=5, iterations=1
    )
    assert len(result) > 500000


@pytest.mark.benchmark(group="tokenizer_long")
def test_tokenize_war_peace_stopwords(benchmark, war_and_peace_text):
    """Tokenize War and Peace with stopword filtering."""
    tok = Tokenizer(
        lowercase=True,
        stopwords=[
            "the",
            "a",
            "an",
            "and",
            "of",
            "in",
            "to",
            "for",
            "on",
            "with",
            "is",
            "was",
            "it",
            "he",
            "she",
            "that",
            "this",
            "his",
            "her",
        ],
    )
    result = benchmark.pedantic(
        tok.tokenize_words, args=(war_and_peace_text,), rounds=5, iterations=1
    )
    assert len(result) > 300000  # less due to stopword removal


@pytest.mark.benchmark(group="tokenizer_long")
def test_tokenize_war_peace_with_spans(benchmark, war_and_peace_text):
    """Tokenize War and Peace returning full span info."""
    tok = Tokenizer(lowercase=True)
    result = benchmark.pedantic(tok.tokenize, args=(war_and_peace_text,), rounds=5, iterations=1)
    assert len(result) > 500000
    # Verify spans are valid (typed pyclass attributes — see P3 / spans.rs)
    assert result[0].start == 0
    assert result[0].end > 0


# ─── P3 regression: typed pyclass fast path vs `tokenize_words` ────────────


@pytest.mark.benchmark(group="tokenizer_p3")
def test_tokenize_pyclass_overhead_vs_words(benchmark, war_and_peace_text):
    """Lock the P3 typed-pyclass speedup.

    `tokenize_words` returns plain `list[str]` with no per-token boundary
    work. `tokenize` now emits `Vec<PyTokenSpan>` directly from Rust
    (audit perf finding #1). The runtime ratio between the two should
    stay close — pre-P3 the dict→dataclass round trip put `tokenize` at
    ~11x the cost of `tokenize_words`. This bench runs `tokenize` (the
    span-emitting path) and asserts the result is materially shorter
    than the words-only path; the absolute number lives in the
    benchmark history file.
    """
    tok = Tokenizer(lowercase=True)
    result = benchmark.pedantic(tok.tokenize, args=(war_and_peace_text,), rounds=5, iterations=1)
    # Sanity: typed pyclass attrs round-trip and the spans index back
    # into the source text.
    first = result[0]
    assert first.text == war_and_peace_text[first.start : first.end].lower()
