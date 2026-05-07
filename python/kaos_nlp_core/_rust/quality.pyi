"""Type stubs for kaos_nlp_core._rust.quality.

The raw analyzer entrypoints emit typed pyclasses (audit perf finding
#6 / P8) — same field shape as the prior nested-dict result, no per-key
dict lookup at the boundary.
"""

from typing import Any

class PyCharClassCounts:
    total_chars: int
    whitespace: int
    alpha: int
    digit: int
    alphanumeric: int
    upper: int
    lower: int
    punct: int
    symbol: int
    non_ascii: int
    newline: int
    line_count: int
    paragraph_count: int
    char_entropy: float

class PyWordStats:
    num_words: int
    unique_words: int
    total_word_chars: int
    max_freq: int
    token_entropy: float
    alphabetic_tokens: int
    in_lexicon: int
    format_tokens: int

class PyQualityRaw:
    chars: PyCharClassCounts
    words: PyWordStats

def count_chars(text: str) -> PyCharClassCounts: ...
def analyze_words(words: list[str]) -> PyWordStats: ...
def analyze_text(text: str, lexicon: Any | None = None) -> PyQualityRaw: ...
def entropy(counts: list[int]) -> float: ...
