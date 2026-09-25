"""Pattern matching: substring search, multi-pattern, regex, FST.

Substring and regex functions return native ``MatchSpan`` /
``RegexMatchSpan`` / ``PatternMatchSpan`` / ``FstSearchResult``
pyclasses with character offsets — no Python dataclass conversion
in the hot path (audit perf finding #1 / P3).
"""

from __future__ import annotations

from kaos_nlp_core._rust.matching import (
    FstMap,
    RegexSetMatcher,
    substring_count,
    substring_count_batch,
)
from kaos_nlp_core._rust.matching import FstSet as _RustFstSet
from kaos_nlp_core._rust.matching import MultiPatternMatcher as _RustMultiPatternMatcher
from kaos_nlp_core._rust.matching import RegexMatcher as _RustRegexMatcher
from kaos_nlp_core._rust.matching import substring_find_all as _raw_find_all
from kaos_nlp_core._rust.matching import substring_find_all_batch as _raw_find_all_batch
from kaos_nlp_core._rust.matching import (
    substring_find_all_case_insensitive as _raw_find_all_ci,
)
from kaos_nlp_core._rust.matching import substring_find_first as _raw_find_first
from kaos_nlp_core.types import FstSearchResult, MatchSpan, PatternMatchSpan, RegexMatchSpan

# ─── Substring search ──────────────────────────────────────────────────────


def substring_find_all(haystack: str, needle: str) -> list[MatchSpan]:
    """Find all occurrences of needle in haystack (SIMD-accelerated)."""
    return _raw_find_all(haystack, needle)


def substring_find_all_batch(haystacks: list[str], needle: str) -> list[list[MatchSpan]]:
    """Find all occurrences across many haystacks."""
    return _raw_find_all_batch(haystacks, needle)


def substring_find_first(haystack: str, needle: str) -> MatchSpan | None:
    """Find the first occurrence of needle, or None."""
    return _raw_find_first(haystack, needle)


def substring_find_all_case_insensitive(haystack: str, needle: str) -> list[MatchSpan]:
    """Find all case-insensitive occurrences."""
    return _raw_find_all_ci(haystack, needle)


# ─── Multi-pattern (Aho-Corasick) ──────────────────────────────────────────


class MultiPatternMatcher:
    """Multi-pattern matcher (composition over the Rust pyclass).

    The Rust ``find_all`` / ``find_all_batch`` already emit typed
    ``PatternMatchSpan`` pyclasses, so this wrapper is a thin
    delegating shim — no per-match conversion happens here.

    ``start`` / ``end`` on every match are **character offsets** into the
    original haystack (Python ``str`` indexing) and ``text`` is the
    original haystack slice, whatever the options.

    :param patterns: non-empty list of literal patterns.
    :param case_insensitive: compare Unicode full case folds (the mapping
        of :meth:`str.casefold`), so ``"são paulo"`` matches
        ``"SÃO PAULO"`` and ``"strasse"`` matches ``"STRAßE"``. A match
        must cover whole folds: ``"s"`` never matches half of a ``"ß"``.
        No normalization form is applied (NFC ``"é"`` and NFD
        ``"e\u0301"`` differ), and ``"İ"`` folds to ``"i\u0307"``, so it
        does not match a bare ``"i"``.
    :param longest_match: leftmost-longest instead of leftmost-first
        selection.
    :param word_boundary: keep a match only if it does not extend a word
        at either edge. At each edge the char inside the match and the
        char outside it must not both be word chars. Word chars are
        Unicode letters/digits (``str.isalnum``-like: ``Alphabetic`` or
        ``Numeric``) plus combining marks; underscore, apostrophes,
        hyphens, other punctuation, symbols, emoji and whitespace are
        separators. So ``"georgia"`` does not match in ``"Georgian"`` or
        ``"georgia2"`` but does in ``"Georgia's"`` and ``"georgia_x"``.
        Scripts written without spaces (Unicode Line_Break ``ID``, ``CJ``
        and ``SA``: Han, kana, Thai, ...) never require a boundary, so
        ``"東京"`` still matches in ``"東京都"``. Haystack edges are
        boundaries. Rejected candidates are dropped before leftmost
        selection, so another pattern can still match at that position.
        Consistent across ``find_all``, ``find_all_batch``, ``is_match``,
        ``count`` and ``replace_all``.
    """

    def __init__(
        self,
        patterns: list[str],
        case_insensitive: bool = False,
        longest_match: bool = False,
        *,
        word_boundary: bool = False,
    ) -> None:
        self._inner = _RustMultiPatternMatcher(
            patterns, case_insensitive, longest_match, word_boundary
        )

    def find_all(self, haystack: str) -> list[PatternMatchSpan]:
        return self._inner.find_all(haystack)

    def find_all_batch(self, haystacks: list[str]) -> list[list[PatternMatchSpan]]:
        return self._inner.find_all_batch(haystacks)

    def is_match(self, haystack: str) -> bool:
        return self._inner.is_match(haystack)

    def count(self, haystack: str) -> int:
        return self._inner.count(haystack)

    def replace_all(self, haystack: str, replacements: list[str]) -> str:
        return self._inner.replace_all(haystack, replacements)

    def pattern_count(self) -> int:
        return self._inner.pattern_count()


# ─── Regex ──────────────────────────────────────────────────────────────────


class RegexMatcher:
    """Compiled regex matcher returning native ``RegexMatchSpan`` pyclasses."""

    def __init__(self, pattern: str) -> None:
        self._inner = _RustRegexMatcher(pattern)

    def find_all(self, haystack: str) -> list[RegexMatchSpan]:
        return self._inner.find_all(haystack)

    def find_all_batch(self, haystacks: list[str]) -> list[list[RegexMatchSpan]]:
        return self._inner.find_all_batch(haystacks)

    def find_first(self, haystack: str) -> RegexMatchSpan | None:
        return self._inner.find_first(haystack)

    def is_match(self, haystack: str) -> bool:
        return self._inner.is_match(haystack)

    def count(self, haystack: str) -> int:
        return self._inner.count(haystack)

    def replace_all(self, haystack: str, replacement: str) -> str:
        return self._inner.replace_all(haystack, replacement)

    def split(self, haystack: str) -> list[str]:
        return self._inner.split(haystack)

    def pattern(self) -> str:
        return self._inner.pattern()


# ─── FST ────────────────────────────────────────────────────────────────────


class FstSet:
    """FST set returning native ``FstSearchResult`` pyclasses for fuzzy search."""

    def __init__(self, keys: list[str]) -> None:
        self._inner = _RustFstSet(keys)

    @classmethod
    def _from_inner(cls, inner: _RustFstSet) -> FstSet:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    @classmethod
    def load(cls, path: str) -> FstSet:
        """Load an FST set from disk (raw FST bytes written by ``save``)."""
        return cls._from_inner(_RustFstSet.load(path))

    def save(self, path: str) -> None:
        """Write the raw FST bytes to disk."""
        self._inner.save(path)

    def contains(self, key: str) -> bool:
        return self._inner.contains(key)

    def fuzzy_search(self, query: str, max_distance: int) -> list[FstSearchResult]:
        return self._inner.fuzzy_search(query, max_distance)

    def prefix_search(self, prefix: str) -> list[str]:
        return self._inner.prefix_search(prefix)

    def __len__(self) -> int:
        return len(self._inner)

    def __contains__(self, key: str) -> bool:
        return self._inner.contains(key)


__all__ = [
    "FstMap",
    "FstSearchResult",
    "FstSet",
    "MatchSpan",
    "MultiPatternMatcher",
    "PatternMatchSpan",
    "RegexMatchSpan",
    "RegexMatcher",
    "RegexSetMatcher",
    "substring_count",
    "substring_count_batch",
    "substring_find_all",
    "substring_find_all_batch",
    "substring_find_all_case_insensitive",
    "substring_find_first",
]
