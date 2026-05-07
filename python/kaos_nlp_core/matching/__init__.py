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
    """

    def __init__(
        self,
        patterns: list[str],
        case_insensitive: bool = False,
        longest_match: bool = False,
    ) -> None:
        self._inner = _RustMultiPatternMatcher(patterns, case_insensitive, longest_match)

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
