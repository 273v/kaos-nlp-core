"""Lexicon-aware token frequency.

Computes per-document term frequencies in a single pass, optionally
filtered by membership in a controlled lexicon (``FstSet`` or full
``Lexicon``). Reuses the Rust-backed ``FrequencyVocabulary`` for
counting and the existing ``Membership`` lookup convention exposed by
``FstSet.contains`` and ``Lexicon.contains`` — no new Rust path.

Returns a typed :class:`VocabularyCounts` whose ``.counts`` attribute is
sorted by frequency descending. The container is iterable, sliceable
via ``.top(k)``, and convertible via ``.to_dict()`` so callers can pick
their preferred shape without paying for a re-sort.

Usage::

    from kaos_nlp_core.vocabulary import token_frequency
    from kaos_nlp_core.quality import default_english_wordset

    result = token_frequency(text, lexicon=default_english_wordset())
    print(result.coverage, result.top(10))

    # Or without a lexicon — count everything.
    result = token_frequency(text)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_nlp_core.lexicon import Lexicon
from kaos_nlp_core.matching import FstSet
from kaos_nlp_core.structures import FrequencyVocabulary
from kaos_nlp_core.tokenizer import tokenize_words

# ── Result types ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TokenCount:
    """A term and its document-level frequency."""

    text: str
    count: int


@dataclass(frozen=True, slots=True)
class VocabularyCounts:
    """Document term-frequency table with coverage statistics.

    ``counts`` is sorted by ``count`` descending. ``total_tokens`` is the
    pre-filter token count; ``kept_tokens`` is the post-filter sum (so
    ``kept_tokens / total_tokens`` is the in-vocabulary coverage when a
    lexicon was supplied).
    """

    counts: tuple[TokenCount, ...]
    total_tokens: int
    kept_tokens: int

    @property
    def coverage(self) -> float:
        """Fraction of tokens that survived the lexicon filter (0.0 to 1.0).

        Returns 1.0 when no lexicon was supplied (all tokens kept).
        """
        if self.total_tokens == 0:
            return 0.0
        return self.kept_tokens / self.total_tokens

    @property
    def unique_terms(self) -> int:
        """Number of distinct terms in :attr:`counts`."""
        return len(self.counts)

    def top(self, k: int) -> list[TokenCount]:
        """Top-k terms by frequency (sorted; already pre-sorted internally)."""
        if k <= 0:
            return []
        return list(self.counts[:k])

    def to_dict(self) -> dict[str, int]:
        """Flat ``{term: count}`` dict for ergonomic consumers."""
        return {tc.text: tc.count for tc in self.counts}

    def __iter__(self):
        """Iterate :class:`TokenCount` records in descending-count order."""
        return iter(self.counts)

    def __len__(self) -> int:
        return len(self.counts)


# ── Public API ────────────────────────────────────────────────────────────


def token_frequency(
    text: str,
    *,
    lexicon: FstSet | Lexicon | None = None,
    lowercase: bool = True,
    min_count: int = 1,
    top_k: int | None = None,
    prefix: int = 0,
) -> VocabularyCounts:
    """Compute a document's term-frequency table.

    Args:
        text: Document text to analyze.
        lexicon: Optional ``FstSet`` or ``Lexicon``. When supplied, only
            tokens that match ``lexicon.contains(token)`` are counted.
            Lower-case the lexicon's keys to match the default
            ``lowercase=True`` token form (the bundled English wordset
            already is).
        lowercase: Lowercase tokens before counting (default True).
        min_count: Drop terms with fewer than ``min_count`` occurrences
            from the result. Total/kept tokens still reflect the
            pre-trim counts.
        top_k: Return only the top-k most frequent terms. ``None``
            returns every distinct term.
        prefix: Optional prefix-truncation length (approximate stem).
            ``0`` disables (default). See ``kaos_nlp_core.tokenizer``.

    Returns:
        A :class:`VocabularyCounts` with the sorted counts and coverage
        statistics.
    """
    tokens = tokenize_words(text, lowercase=lowercase, prefix=prefix)
    accept = _membership_predicate(lexicon)

    vocab = FrequencyVocabulary()
    total_tokens = 0
    kept_tokens = 0
    for tok in tokens:
        total_tokens += 1
        if not accept(tok):
            continue
        vocab.insert(tok)
        kept_tokens += 1

    n_unique = len(vocab)
    if n_unique == 0:
        return VocabularyCounts(
            counts=(),
            total_tokens=total_tokens,
            kept_tokens=kept_tokens,
        )

    n = top_k if top_k is not None else n_unique
    pairs: list[tuple[str, int]] = vocab.top_n(n)

    if min_count > 1:
        pairs = [(t, c) for t, c in pairs if c >= min_count]

    return VocabularyCounts(
        counts=tuple(TokenCount(text=t, count=c) for t, c in pairs),
        total_tokens=total_tokens,
        kept_tokens=kept_tokens,
    )


# ── Internals ─────────────────────────────────────────────────────────────


def _membership_predicate(lexicon: Any) -> Any:
    """Return a callable that decides whether a token is in-vocabulary."""
    if lexicon is None:
        return lambda _t: True
    contains = getattr(lexicon, "contains", None)
    if not callable(contains):
        raise TypeError(
            "lexicon must expose a callable 'contains(token) -> bool'. "
            "Pass kaos_nlp_core.matching.FstSet or kaos_nlp_core.lexicon.Lexicon, "
            "or set lexicon=None to disable filtering."
        )
    return contains


__all__ = [
    "TokenCount",
    "VocabularyCounts",
    "token_frequency",
]
