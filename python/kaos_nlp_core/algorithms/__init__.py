"""String distance and similarity algorithms.

All distance functions return a ``DistanceResult`` with typed fields:
``distance``, ``normalized``, ``similarity``.
"""

from __future__ import annotations

from typing import Any

from kaos_nlp_core._rust import algorithms as _raw
from kaos_nlp_core.types import DistanceResult


def _to_result(raw: Any) -> DistanceResult:
    return DistanceResult(
        distance=raw["distance"], normalized=raw["normalized"], similarity=raw["similarity"]
    )


# ─── Edit distance ──────────────────────────────────────────────────────────


def levenshtein(a: str, b: str) -> DistanceResult:
    """Levenshtein edit distance (insert, delete, substitute)."""
    return _to_result(_raw.levenshtein(a, b))


def damerau_levenshtein(a: str, b: str) -> DistanceResult:
    """Damerau-Levenshtein distance (insert, delete, substitute, transpose)."""
    return _to_result(_raw.damerau_levenshtein(a, b))


def osa(a: str, b: str) -> DistanceResult:
    """Optimal String Alignment distance."""
    return _to_result(_raw.osa(a, b))


def hamming(a: str, b: str) -> DistanceResult:
    """Hamming distance (strings must be equal length)."""
    return _to_result(_raw.hamming(a, b))


def jaro(a: str, b: str) -> DistanceResult:
    """Jaro similarity for short strings and names."""
    return _to_result(_raw.jaro(a, b))


def jaro_winkler(a: str, b: str, prefix_weight: float = 0.1) -> DistanceResult:
    """Jaro-Winkler similarity (Jaro with prefix bonus)."""
    return _to_result(_raw.jaro_winkler(a, b, prefix_weight))


def sorensen_dice(a: str, b: str) -> DistanceResult:
    """Sorensen-Dice coefficient over character bigrams."""
    return _to_result(_raw.sorensen_dice(a, b))


# ─── Phonetic ───────────────────────────────────────────────────────────────


def soundex_distance(a: str, b: str) -> DistanceResult:
    """Soundex phonetic distance."""
    return _to_result(_raw.soundex_distance(a, b))


def soundex_encode(a: str) -> str:
    """Encode a string to its Soundex code."""
    return _raw.soundex_encode(a)


def metaphone_distance(a: str, b: str) -> DistanceResult:
    """Metaphone phonetic distance."""
    return _to_result(_raw.metaphone_distance(a, b))


def metaphone_encode(a: str) -> str:
    """Encode a string to its Metaphone code."""
    return _raw.metaphone_encode(a)


# ─── Sequence ───────────────────────────────────────────────────────────────


def lcs_distance(a: str, b: str) -> DistanceResult:
    """Longest Common Subsequence distance."""
    return _to_result(_raw.lcs_distance(a, b))


def lcs_length(a: str, b: str) -> int:
    """Length of the Longest Common Subsequence."""
    return _raw.lcs_length(a, b)


def longest_common_substring(a: str, b: str) -> DistanceResult:
    """Longest Common Substring (contiguous) similarity."""
    return _to_result(_raw.longest_common_substring(a, b))


def longest_common_substring_length(a: str, b: str) -> int:
    """Length of the Longest Common Substring."""
    return _raw.longest_common_substring_length(a, b)


# ─── N-gram ─────────────────────────────────────────────────────────────────


def ngram_jaccard(a: str, b: str, n: int = 2) -> DistanceResult:
    """Jaccard similarity over character n-grams."""
    return _to_result(_raw.ngram_jaccard(a, b, n))


def ngram_cosine(a: str, b: str, n: int = 2) -> DistanceResult:
    """Cosine similarity over character n-gram frequency vectors."""
    return _to_result(_raw.ngram_cosine(a, b, n))


def ngram_overlap(a: str, b: str, n: int = 2) -> DistanceResult:
    """Overlap coefficient over character n-gram sets."""
    return _to_result(_raw.ngram_overlap(a, b, n))


# ─── Token n-gram ──────────────────────────────────────────────────────────


def token_jaccard(a: str, b: str, lowercase: bool = False) -> DistanceResult:
    """Jaccard similarity over word tokens."""
    return _to_result(_raw.token_jaccard(a, b, lowercase))


def token_ngram_jaccard(a: str, b: str, n: int = 2, lowercase: bool = False) -> DistanceResult:
    """Jaccard similarity over token n-grams."""
    return _to_result(_raw.token_ngram_jaccard(a, b, n, lowercase))


def token_ngram_cosine(a: str, b: str, n: int = 2, lowercase: bool = False) -> DistanceResult:
    """Cosine similarity over token n-gram frequency vectors."""
    return _to_result(_raw.token_ngram_cosine(a, b, n, lowercase))


def token_ngram_overlap(a: str, b: str, n: int = 2, lowercase: bool = False) -> DistanceResult:
    """Overlap coefficient over token n-gram sets."""
    return _to_result(_raw.token_ngram_overlap(a, b, n, lowercase))


# ─── Batch ──────────────────────────────────────────────────────────────────


def compare_batch(
    pairs: list[tuple[str, str]],
    algorithm: str = "jaro-winkler",
    n: int = 2,
    lowercase: bool = False,
    prefix_weight: float = 0.1,
) -> list[DistanceResult]:
    """Compare many string pairs with one configured algorithm."""
    return [
        _to_result(r) for r in _raw.compare_batch(pairs, algorithm, n, lowercase, prefix_weight)
    ]


__all__ = [
    "DistanceResult",
    "compare_batch",
    "damerau_levenshtein",
    "hamming",
    "jaro",
    "jaro_winkler",
    "lcs_distance",
    "lcs_length",
    "levenshtein",
    "longest_common_substring",
    "longest_common_substring_length",
    "metaphone_distance",
    "metaphone_encode",
    "ngram_cosine",
    "ngram_jaccard",
    "ngram_overlap",
    "osa",
    "sorensen_dice",
    "soundex_distance",
    "soundex_encode",
    "token_jaccard",
    "token_ngram_cosine",
    "token_ngram_jaccard",
    "token_ngram_overlap",
]
