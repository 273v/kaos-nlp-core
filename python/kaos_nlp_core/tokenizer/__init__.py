"""Word tokenization with Unicode whitespace, punctuation stripping,
prefix truncation, stopword filtering, and span tracking.

Two usage patterns:

1. Stateless convenience functions:
    from kaos_nlp_core.tokenizer import tokenize, tokenize_words
    tokens = tokenize("Hello, world!", lowercase=True)
    words = tokenize_words("Hello, world!", lowercase=True, prefix=4)

2. Reusable configured tokenizer:
    from kaos_nlp_core.tokenizer import Tokenizer
    tok = Tokenizer(lowercase=True, prefix=4, stopwords=["the", "a"])
    words = tok.tokenize_words("The Automobile Industry")

Result shape: every ``tokenize`` / ``tokenize_batch`` /
``tokenize_regex*`` call returns native ``TokenSpan`` pyclasses (audit
perf finding #1 / P3) rather than the prior dict→dataclass round trip.
The Python-side wrapper is now a passthrough — Rust hands the typed
pyclass directly to the consumer.
"""

from __future__ import annotations

from kaos_nlp_core._rust.tokenizer import (
    Tokenizer as _RustTokenizer,
)
from kaos_nlp_core._rust.tokenizer import (
    tokenize as _raw_tokenize,
)
from kaos_nlp_core._rust.tokenizer import (
    tokenize_batch as _raw_tokenize_batch,
)
from kaos_nlp_core._rust.tokenizer import tokenize_words, tokenize_words_batch
from kaos_nlp_core.types import TokenSpan


def tokenize(text: str, lowercase: bool = False, prefix: int = 0) -> list[TokenSpan]:
    """Tokenize text and return ``TokenSpan`` pyclasses with character offsets."""
    return _raw_tokenize(text, lowercase, prefix)


def tokenize_batch(
    texts: list[str], lowercase: bool = False, prefix: int = 0
) -> list[list[TokenSpan]]:
    """Tokenize multiple texts and return ``TokenSpan`` pyclasses."""
    return _raw_tokenize_batch(texts, lowercase, prefix)


class Tokenizer:
    """Configurable word tokenizer with typed span output.

    Wraps the Rust ``Tokenizer`` to return ``list[TokenSpan]`` from
    ``tokenize()``, ``tokenize_batch()``, and ``tokenize_regex*()``.
    The ``tokenize_words*`` methods return plain ``list[str]``.

    The wrapper is a thin passthrough — Rust emits the typed
    ``TokenSpan`` pyclasses directly, so no per-token dict→dataclass
    conversion happens in Python.
    """

    def __init__(
        self,
        lowercase: bool = False,
        keep_punctuation: bool = False,
        prefix: int = 0,
        stopwords: list[str] | None = None,
        index: list[str] | None = None,
    ) -> None:
        self._inner = _RustTokenizer(
            lowercase=lowercase,
            keep_punctuation=keep_punctuation,
            prefix=prefix,
            stopwords=stopwords,
            index=index,
        )

    def tokenize(self, text: str) -> list[TokenSpan]:
        return self._inner.tokenize(text)

    def tokenize_words(self, text: str) -> list[str]:
        return self._inner.tokenize_words(text)

    def tokenize_batch(self, texts: list[str]) -> list[list[TokenSpan]]:
        return self._inner.tokenize_batch(texts)

    def tokenize_words_batch(self, texts: list[str]) -> list[list[str]]:
        return self._inner.tokenize_words_batch(texts)

    def tokenize_regex(self, text: str, pattern: str | None = None) -> list[TokenSpan]:
        return self._inner.tokenize_regex(text, pattern)

    def tokenize_regex_words(self, text: str, pattern: str | None = None) -> list[str]:
        return self._inner.tokenize_regex_words(text, pattern)

    def tokenize_regex_batch(
        self, texts: list[str], pattern: str | None = None
    ) -> list[list[TokenSpan]]:
        return self._inner.tokenize_regex_batch(texts, pattern)

    def tokenize_regex_words_batch(
        self, texts: list[str], pattern: str | None = None
    ) -> list[list[str]]:
        return self._inner.tokenize_regex_words_batch(texts, pattern)


__all__ = [
    "TokenSpan",
    "Tokenizer",
    "tokenize",
    "tokenize_batch",
    "tokenize_words",
    "tokenize_words_batch",
]
