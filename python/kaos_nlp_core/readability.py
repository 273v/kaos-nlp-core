"""Readability scoring: Flesch, Flesch-Kincaid, ARI, Coleman-Liau, SMOG,
Gunning Fog, Dale-Chall, LIX, and RIX with verified formula provenance.

Counting (words, letters, syllables, polysyllables, Fog complex words,
unfamiliar words) runs in the Rust core in one GIL-released pass through
the shared tokenizer, with syllables resolved by exact CMUdict lookup
(bundled ~660 kB FST, 2-clause BSD) and a tuned heuristic fallback.
Sentence counting reuses the bundled Punkt model. The formula constants
live here as module-level tables so they can be tuned without rebuilding
the wheel.

Quick start — most users want one of these::

    from kaos_nlp_core.readability import (
        flesch_kincaid_grade, flesch_reading_ease, gunning_fog,
    )

    grade = flesch_kincaid_grade("The quick brown fox jumps over the lazy dog.")

Or from a shell with no install (``uv`` fetches the wheel on the fly)::

    uv run --with kaos-nlp-core python -c \
        "from kaos_nlp_core.readability import flesch_kincaid_grade; \
         print(flesch_kincaid_grade('Hello, world.'))"

Full report::

    from kaos_nlp_core.readability import readability_report

    report = readability_report(text)
    print(report.scores.flesch_kincaid_grade, report.counts.words)
    print(report.to_dict())

Dale-Chall needs a familiar-word list, which is copyrighted (Chall &
Dale 1995) and therefore not bundled; build one with
``scripts/build_familiar_wordset.py`` and pass it as ``familiar_words``.

Formula sources (constants verified against the originals):

- Flesch Reading Ease (Flesch 1948).
- Flesch-Kincaid Grade Level (Kincaid et al. 1975, DTIC AD-A006655).
- ARI (Smith & Senter 1967, AMRL-TR-66-220): characters are letters +
  digits; the published convention rounds the result *up* to a grade —
  this module returns the raw value and leaves rounding to callers.
- Coleman-Liau (Coleman & Liau 1975): per-100-word form, letters only.
- SMOG (McLaughlin 1969): calibrated on 30-sentence samples; scores on
  shorter texts are reported with ``smog_valid=False`` instead of being
  silently returned or raised.
- Gunning Fog (Gunning 1952): complex-word counting implements the
  mechanizable exclusions (suffix -es/-ed/-ing third syllables, proper
  nouns, hyphenated compounds), each flag-configurable. textstat skips
  all exclusions and reports higher Fog scores; pass
  ``fog_exclude_*=False`` for comparable numbers.
- Dale-Chall (Dale & Chall 1948 regression — the formula every major
  library implements — usable with the 1995 word list you supply).
- LIX / RIX (Björnsson 1968; Anderson 1983): long words have more than
  six letters/digits.

All scores are English-calibrated; counting is deterministic and
panic-free for any input (CJK, emoji, mixed-script), but scores on
non-English text are not meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from kaos_nlp_core._defaults import get_default_punkt_tokenizer
from kaos_nlp_core._rust.readability import SyllableMap
from kaos_nlp_core._rust.readability import analyze as _rust_analyze
from kaos_nlp_core._rust.readability import syllable_count as _rust_syllable_count
from kaos_nlp_core.lexicon import Lexicon
from kaos_nlp_core.matching import FstSet

# ── Formula constants (tunable without rebuilding the wheel) ───────────────

FLESCH_READING_EASE: dict[str, float] = {
    "base": 206.835,
    "per_words_per_sentence": 1.015,
    "per_syllables_per_word": 84.6,
}

FLESCH_KINCAID_GRADE: dict[str, float] = {
    "per_words_per_sentence": 0.39,
    "per_syllables_per_word": 11.8,
    "base": -15.59,
}

AUTOMATED_READABILITY_INDEX: dict[str, float] = {
    "per_chars_per_word": 4.71,
    "per_words_per_sentence": 0.5,
    "base": -21.43,
}

COLEMAN_LIAU: dict[str, float] = {
    "per_letters_per_100_words": 0.0588,
    "per_sentences_per_100_words": 0.296,
    "base": -15.8,
}

SMOG: dict[str, float] = {
    "scale": 1.0430,
    "sentences_norm": 30.0,
    "base": 3.1291,
}

#: Minimum sentence count for which the SMOG calibration is valid.
SMOG_MIN_SENTENCES = 30

GUNNING_FOG: dict[str, float] = {
    "scale": 0.4,
    "complex_percent_scale": 100.0,
}

DALE_CHALL: dict[str, float] = {
    "per_pdw": 0.1579,
    "per_asl": 0.0496,
    "adjustment": 3.6365,
    "adjustment_threshold_pdw": 5.0,
}

LIX_LONG_WORD_SCALE = 100.0

# ── Default syllable map (lazy, cached) ────────────────────────────────────

_DEFAULT_SYLLABLE_MAP: SyllableMap | None = None
_DEFAULT_SYLLABLE_MAP_FILE = "cmudict_syllables.fst"


def default_syllable_map() -> SyllableMap:
    """Lazy-load the bundled CMUdict-derived syllable map.

    Returns a process-cached :class:`SyllableMap` of ~125k lowercase
    words → syllable counts (first pronunciations). The underlying
    ~660 kB FST ships inside the wheel under ``kaos_nlp_core/data/``.
    """
    global _DEFAULT_SYLLABLE_MAP
    if _DEFAULT_SYLLABLE_MAP is None:
        resource = files("kaos_nlp_core.data").joinpath(_DEFAULT_SYLLABLE_MAP_FILE)
        _DEFAULT_SYLLABLE_MAP = SyllableMap.load(str(resource))
    return _DEFAULT_SYLLABLE_MAP


# ── Typed result objects ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TextCounts:
    """Raw readability counts (Unicode-codepoint character counts).

    A *word* is a whitespace-delimited token containing at least one
    letter or digit ("don't" and "mother-in-law" are one word each).
    ``unfamiliar_words`` is ``None`` when no familiar-word lexicon was
    supplied. ``long_words`` counts words of more than six
    letters/digits (LIX/RIX).
    """

    sentences: int
    words: int
    letters: int
    letters_and_digits: int
    syllables: int
    polysyllable_words: int
    fog_complex_words: int
    long_words: int
    unfamiliar_words: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return the counts as a flat dict (``None`` fields omitted)."""
        out: dict[str, Any] = {
            "sentences": self.sentences,
            "words": self.words,
            "letters": self.letters,
            "letters_and_digits": self.letters_and_digits,
            "syllables": self.syllables,
            "polysyllable_words": self.polysyllable_words,
            "fog_complex_words": self.fog_complex_words,
            "long_words": self.long_words,
        }
        if self.unfamiliar_words is not None:
            out["unfamiliar_words"] = self.unfamiliar_words
        return out


@dataclass(frozen=True, slots=True)
class ReadabilityScores:
    """Readability scores; ``None`` where undefined for the input.

    Scores are ``None`` when the text has no words or no sentences,
    and ``dale_chall`` additionally requires a familiar-word lexicon.
    ``smog_index`` is always computed when defined, but the SMOG
    calibration assumes ≥30 sentences — check ``smog_valid`` before
    trusting it on short texts. ``automated_readability_index`` is the
    raw value; the published convention rounds it up to a grade.
    """

    flesch_reading_ease: float | None
    flesch_kincaid_grade: float | None
    automated_readability_index: float | None
    coleman_liau_index: float | None
    smog_index: float | None
    smog_valid: bool
    gunning_fog: float | None
    dale_chall: float | None
    lix: float | None
    rix: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return the scores as a flat dict (``None`` fields omitted)."""
        out: dict[str, Any] = {}
        for name in (
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "automated_readability_index",
            "coleman_liau_index",
            "smog_index",
            "gunning_fog",
            "dale_chall",
            "lix",
            "rix",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out["smog_valid"] = self.smog_valid
        return out


@dataclass(frozen=True, slots=True)
class ReadabilityReport:
    """Counts plus scores for a single text."""

    counts: TextCounts
    scores: ReadabilityScores

    def to_dict(self) -> dict[str, Any]:
        return {"counts": self.counts.to_dict(), "scores": self.scores.to_dict()}


# ── Public API ─────────────────────────────────────────────────────────────


def compute_counts(
    text: str,
    *,
    familiar_words: FstSet | Lexicon | None = None,
    use_syllable_map: bool = True,
    fog_exclude_suffixes: bool = True,
    fog_exclude_proper_nouns: bool = True,
    fog_exclude_compounds: bool = True,
) -> TextCounts:
    """Count readability primitives for ``text``.

    Args:
        text: Input text.
        familiar_words: Optional ``FstSet`` or ``Lexicon`` of familiar
            words (lowercase keys) enabling the Dale-Chall
            unfamiliar-word count.
        use_syllable_map: Resolve syllables by exact CMUdict lookup with
            heuristic fallback (default). ``False`` forces the pure
            heuristic.
        fog_exclude_suffixes: Gunning's exclusion of 3-syllable words
            whose third syllable is -es/-ed/-ing ("created").
        fog_exclude_proper_nouns: Gunning's exclusion of proper nouns,
            approximated as title-case words not at a detected sentence
            start.
        fog_exclude_compounds: Gunning's exclusion of hyphenated
            compounds ("state-of-the-art").

    Returns:
        A :class:`TextCounts` value object (sentence count via the
        bundled Punkt model).
    """
    counts = _rust_analyze(
        text,
        _unwrap(familiar_words),
        default_syllable_map() if use_syllable_map else None,
        fog_exclude_suffixes,
        fog_exclude_proper_nouns,
        fog_exclude_compounds,
    )
    sentences = get_default_punkt_tokenizer().count_sentences(text) if counts.words else 0
    return TextCounts(
        sentences=sentences,
        words=counts.words,
        letters=counts.letters,
        letters_and_digits=counts.letters_and_digits,
        syllables=counts.syllables,
        polysyllable_words=counts.polysyllable_words,
        fog_complex_words=counts.fog_complex_words,
        long_words=counts.long_words,
        unfamiliar_words=counts.unfamiliar_words,
    )


def score_counts(counts: TextCounts) -> ReadabilityScores:
    """Compute every score from precomputed :class:`TextCounts`.

    Pure arithmetic over the module-level constant tables; no text
    processing happens here.
    """
    w = counts.words
    s = counts.sentences
    if w == 0 or s == 0:
        return ReadabilityScores(
            flesch_reading_ease=None,
            flesch_kincaid_grade=None,
            automated_readability_index=None,
            coleman_liau_index=None,
            smog_index=None,
            smog_valid=False,
            gunning_fog=None,
            dale_chall=None,
            lix=None,
            rix=None,
        )

    wps = w / s  # words per sentence
    spw = counts.syllables / w  # syllables per word

    fre = (
        FLESCH_READING_EASE["base"]
        - FLESCH_READING_EASE["per_words_per_sentence"] * wps
        - FLESCH_READING_EASE["per_syllables_per_word"] * spw
    )
    fkg = (
        FLESCH_KINCAID_GRADE["per_words_per_sentence"] * wps
        + FLESCH_KINCAID_GRADE["per_syllables_per_word"] * spw
        + FLESCH_KINCAID_GRADE["base"]
    )
    ari = (
        AUTOMATED_READABILITY_INDEX["per_chars_per_word"] * (counts.letters_and_digits / w)
        + AUTOMATED_READABILITY_INDEX["per_words_per_sentence"] * wps
        + AUTOMATED_READABILITY_INDEX["base"]
    )
    cli = (
        COLEMAN_LIAU["per_letters_per_100_words"] * (counts.letters / w * 100.0)
        - COLEMAN_LIAU["per_sentences_per_100_words"] * (s / w * 100.0)
        + COLEMAN_LIAU["base"]
    )
    smog = (
        SMOG["scale"] * math.sqrt(counts.polysyllable_words * SMOG["sentences_norm"] / s)
        + SMOG["base"]
    )
    fog = GUNNING_FOG["scale"] * (
        wps + GUNNING_FOG["complex_percent_scale"] * (counts.fog_complex_words / w)
    )

    dale_chall: float | None = None
    if counts.unfamiliar_words is not None:
        pdw = counts.unfamiliar_words / w * 100.0  # percent difficult words
        dale_chall = DALE_CHALL["per_pdw"] * pdw + DALE_CHALL["per_asl"] * wps
        if pdw > DALE_CHALL["adjustment_threshold_pdw"]:
            dale_chall += DALE_CHALL["adjustment"]

    lix = wps + LIX_LONG_WORD_SCALE * (counts.long_words / w)
    rix = counts.long_words / s

    return ReadabilityScores(
        flesch_reading_ease=fre,
        flesch_kincaid_grade=fkg,
        automated_readability_index=ari,
        coleman_liau_index=cli,
        smog_index=smog,
        smog_valid=s >= SMOG_MIN_SENTENCES,
        gunning_fog=fog,
        dale_chall=dale_chall,
        lix=lix,
        rix=rix,
    )


def readability_report(
    text: str,
    *,
    familiar_words: FstSet | Lexicon | None = None,
    use_syllable_map: bool = True,
    fog_exclude_suffixes: bool = True,
    fog_exclude_proper_nouns: bool = True,
    fog_exclude_compounds: bool = True,
) -> ReadabilityReport:
    """Convenience: count + score in a single call.

    See :func:`compute_counts` for parameter semantics.
    """
    counts = compute_counts(
        text,
        familiar_words=familiar_words,
        use_syllable_map=use_syllable_map,
        fog_exclude_suffixes=fog_exclude_suffixes,
        fog_exclude_proper_nouns=fog_exclude_proper_nouns,
        fog_exclude_compounds=fog_exclude_compounds,
    )
    return ReadabilityReport(counts=counts, scores=score_counts(counts))


# ── One-shot conveniences (the scores most users want) ─────────────────────


def flesch_kincaid_grade(text: str) -> float | None:
    """Flesch-Kincaid Grade Level, or ``None`` for empty input."""
    return readability_report(text).scores.flesch_kincaid_grade


def flesch_reading_ease(text: str) -> float | None:
    """Flesch Reading Ease (higher = easier), or ``None`` for empty input."""
    return readability_report(text).scores.flesch_reading_ease


def gunning_fog(text: str) -> float | None:
    """Gunning Fog index (literature-faithful exclusions), or ``None``."""
    return readability_report(text).scores.gunning_fog


def syllable_count(word: str, *, use_syllable_map: bool = True) -> int:
    """Syllables in a single token (≥1 for non-empty input).

    Exact CMUdict lookup when available, tuned heuristic otherwise.
    """
    return _rust_syllable_count(word, default_syllable_map() if use_syllable_map else None)


# ── Internals ──────────────────────────────────────────────────────────────


def _unwrap(lex: FstSet | Lexicon | None) -> Any:
    """Pass the underlying PyO3 pyclass to the Rust binding."""
    if lex is None:
        return None
    return lex._inner


__all__ = [
    "AUTOMATED_READABILITY_INDEX",
    "COLEMAN_LIAU",
    "DALE_CHALL",
    "FLESCH_KINCAID_GRADE",
    "FLESCH_READING_EASE",
    "GUNNING_FOG",
    "LIX_LONG_WORD_SCALE",
    "SMOG",
    "SMOG_MIN_SENTENCES",
    "ReadabilityReport",
    "ReadabilityScores",
    "SyllableMap",
    "TextCounts",
    "compute_counts",
    "default_syllable_map",
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "gunning_fog",
    "readability_report",
    "score_counts",
    "syllable_count",
]
