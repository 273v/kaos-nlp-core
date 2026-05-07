"""Typed result objects for kaos-nlp-core APIs.

All public Python APIs return typed objects rather than raw dicts.
The hot per-result types — ``TokenSpan``, ``MatchSpan``,
``PatternMatchSpan``, ``RegexMatchSpan``, ``FstSearchResult``,
``ScoredDoc``, ``PostingEntry`` — are native PyO3 ``#[pyclass]``
instances, re-exported here under their unprefixed names so callers
keep using ``from kaos_nlp_core.types import TokenSpan`` unchanged.

Audit perf finding #1 / P3: emitting pyclasses directly from Rust
skips the dict→Python-dataclass round trip that measured ~11x slower
than ``tokenize_words`` on a 10K-token text. The remaining types in
this module (``Segment``, ``DistanceResult``, ``TokenProperties``,
``SenseEntry``, ``RelatedTerm``) are still Python dataclasses because
their construction is not on a tight per-token loop.
"""

from __future__ import annotations

from dataclasses import dataclass

# Native pyclasses -- emitted directly from PyO3 bindings, no Python
# dataclass rebuild. Re-exported under the unprefixed name so the
# import path ``from kaos_nlp_core.types import TokenSpan`` remains
# stable. ``ruff`` would otherwise flag these as unused; ``__all__``
# below makes the re-export intent explicit.
from kaos_nlp_core._rust.spans import PyFstSearchResult as FstSearchResult
from kaos_nlp_core._rust.spans import PyMatchSpan as MatchSpan
from kaos_nlp_core._rust.spans import PyPatternMatchSpan as PatternMatchSpan
from kaos_nlp_core._rust.spans import PyPostingEntry as PostingEntry
from kaos_nlp_core._rust.spans import PyRegexMatchSpan as RegexMatchSpan
from kaos_nlp_core._rust.spans import PyScoredDoc as ScoredDoc
from kaos_nlp_core._rust.spans import PyTokenSpan as TokenSpan


@dataclass(frozen=True, slots=True)
class Segment:
    """A text segment (sentence, paragraph, or line) with span and confidence."""

    text: str
    """The segment text."""

    start: int
    """Character offset start in the source text."""

    end: int
    """Character offset end in the source text."""

    confidence: float = 1.0
    """Segmentation confidence from the tokenizer model."""


@dataclass(frozen=True, slots=True)
class DistanceResult:
    """Result of a string distance/similarity computation."""

    distance: float
    """Raw distance value (0.0 = identical for edit distances)."""

    normalized: float
    """Distance normalized to [0.0, 1.0] range."""

    similarity: float
    """Similarity = 1.0 - normalized (1.0 = identical)."""


@dataclass(frozen=True, slots=True)
class TokenProperties:
    """Classification flags for a single token."""

    is_letter_word: bool = False
    is_uppercase_word: bool = False
    is_lowercase_word: bool = False
    is_mixed_case_word: bool = False
    is_title_case_word: bool = False
    is_numeric_word: bool = False
    is_alphanumeric_word: bool = False
    starts_with_digit: bool = False
    has_punctuation: bool = False
    has_dash: bool = False
    is_hyphenated: bool = False
    is_abbreviation: bool = False
    has_emoji: bool = False
    is_symbolic_word: bool = False
    ends_with_terminal: bool = False


@dataclass(frozen=True, slots=True)
class SenseEntry:
    """A word sense from a Lexicon entry."""

    part_of_speech: str = ""
    sense_index: int = 0
    definition: str = ""
    synonyms: list[str] | None = None
    antonyms: list[str] | None = None
    hypernyms: list[str] | None = None
    hyponyms: list[str] | None = None


@dataclass(frozen=True, slots=True)
class RelatedTerm:
    """A related term returned from a Lexicon relation lookup.

    The relation is attached so callers can flatten multiple lookups
    (synonyms + hypernyms + inflections) into a single typed list
    without losing track of how each term was reached.
    """

    text: str
    """The related word or phrase."""

    relation: str
    """Relation type: synonym, antonym, hypernym, hyponym, inflection,
    collocation, derivation_noun/verb/adjective/adverb, cognate,
    etymology_parent, or other."""

    pos: str | None = None
    """Part of speech of the source word, when the lookup was sense-specific."""

    sense_index: int | None = None
    """Sense index of the source word, when the lookup was sense-specific."""


__all__ = [
    "DistanceResult",
    "FstSearchResult",
    "MatchSpan",
    "PatternMatchSpan",
    "PostingEntry",
    "RegexMatchSpan",
    "RelatedTerm",
    "ScoredDoc",
    "Segment",
    "SenseEntry",
    "TokenProperties",
    "TokenSpan",
]
