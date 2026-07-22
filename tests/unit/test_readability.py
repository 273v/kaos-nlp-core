"""Tests for kaos_nlp_core.readability — readability counts and scores.

The simple-text expectations are hand-computed from the published
formulas (see module docstring in kaos_nlp_core/readability.py for
sources); the syllable-accuracy test pins the tuned heuristic against a
committed CMUdict-derived sample.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaos_nlp_core.matching import FstSet
from kaos_nlp_core.readability import (
    ReadabilityReport,
    ReadabilityScores,
    TextCounts,
    compute_counts,
    default_syllable_map,
    flesch_kincaid_grade,
    flesch_reading_ease,
    gunning_fog,
    readability_report,
    score_counts,
    syllable_count,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

SIMPLE = "The cat sat on the mat. The dog ate a bone."


# ── Counts ─────────────────────────────────────────────────────────────────


class TestComputeCounts:
    def test_simple_text_hand_counted(self) -> None:
        c = compute_counts(SIMPLE)
        assert c.sentences == 2
        assert c.words == 11
        assert c.letters == 31
        assert c.letters_and_digits == 31
        assert c.syllables == 11  # all monosyllables
        assert c.polysyllable_words == 0
        assert c.fog_complex_words == 0
        assert c.long_words == 0
        assert c.unfamiliar_words is None

    def test_empty_text(self) -> None:
        c = compute_counts("")
        assert c.words == 0
        assert c.sentences == 0
        assert c.syllables == 0

    def test_whitespace_only(self) -> None:
        c = compute_counts("   \n\t  \n")
        assert c.words == 0
        assert c.sentences == 0

    def test_contractions_and_compounds_are_single_words(self) -> None:
        c = compute_counts("Don't touch the state-of-the-art machine.")
        assert c.words == 5

    def test_digits_count_toward_ari_chars_only(self) -> None:
        c = compute_counts("Pay 100 dollars.")
        assert c.words == 3
        assert c.letters == 10  # Pay + dollars
        assert c.letters_and_digits == 13
        # numeric token counts one syllable
        assert c.syllables == 1 + 1 + 2

    def test_polysyllables(self) -> None:
        c = compute_counts("An extraordinary bureaucracy emerged.")
        # extraordinary=6, bureaucracy=4, emerged=2 (CMUdict lookups)
        assert c.syllables == 1 + 6 + 4 + 2
        assert c.polysyllable_words == 2

    def test_unfamiliar_words_with_lexicon(self) -> None:
        lex = FstSet(["the", "cat", "sat", "on", "mat", "dog", "ate", "a"])
        c = compute_counts(SIMPLE, familiar_words=lex)
        assert c.unfamiliar_words == 1  # "bone"

    def test_proper_nouns_are_familiar(self) -> None:
        lex = FstSet(["we", "visited", "yesterday"])
        c = compute_counts("We visited Chattanooga yesterday.", familiar_words=lex)
        assert c.unfamiliar_words == 0

    def test_fog_suffix_exclusion_flag(self) -> None:
        text = "He trespasses. She trespasses. They trespasses."
        strict = compute_counts(text)
        naive = compute_counts(text, fog_exclude_suffixes=False)
        assert strict.fog_complex_words == 0
        assert naive.fog_complex_words == 3

    def test_fog_proper_noun_exclusion_flag(self) -> None:
        text = "We toured Wisconsin gladly."
        strict = compute_counts(text)
        naive = compute_counts(text, fog_exclude_proper_nouns=False)
        assert strict.fog_complex_words == 0
        assert naive.fog_complex_words == 1

    def test_fog_compound_exclusion_flag(self) -> None:
        text = "A state-of-the-art design."
        strict = compute_counts(text)
        naive = compute_counts(text, fog_exclude_compounds=False)
        assert strict.fog_complex_words == 0
        assert naive.fog_complex_words == 1

    def test_long_words(self) -> None:
        c = compute_counts("The dazzling firmament glittered.")
        # dazzling(8), firmament(9), glittered(9) > 6 letters
        assert c.long_words == 3

    def test_counts_deterministic(self) -> None:
        text = "Mixed 東京 text with emoji 🎉 and café accents."
        assert compute_counts(text) == compute_counts(text)

    def test_to_dict_omits_none(self) -> None:
        d = compute_counts(SIMPLE).to_dict()
        assert "unfamiliar_words" not in d
        lex = FstSet(["the"])
        d2 = compute_counts(SIMPLE, familiar_words=lex).to_dict()
        assert "unfamiliar_words" in d2


# ── Scores ─────────────────────────────────────────────────────────────────


class TestScores:
    def test_simple_text_scores_hand_computed(self) -> None:
        scores = readability_report(SIMPLE).scores
        # W=11, S=2, syll=11, letters=31 → published formulas by hand.
        assert scores.flesch_reading_ease == pytest.approx(116.6525, abs=1e-4)
        assert scores.flesch_kincaid_grade == pytest.approx(-1.645, abs=1e-4)
        assert scores.automated_readability_index == pytest.approx(-5.4064, abs=1e-4)
        assert scores.coleman_liau_index == pytest.approx(-4.6109, abs=1e-4)
        assert scores.smog_index == pytest.approx(3.1291, abs=1e-4)
        assert scores.smog_valid is False
        assert scores.gunning_fog == pytest.approx(2.2, abs=1e-4)
        assert scores.lix == pytest.approx(5.5, abs=1e-4)
        assert scores.rix == pytest.approx(0.0, abs=1e-4)
        assert scores.dale_chall is None

    def test_dale_chall_hand_computed(self) -> None:
        lex = FstSet(["the", "cat", "sat", "on", "mat", "dog", "ate", "a"])
        scores = readability_report(SIMPLE, familiar_words=lex).scores
        # PDW = 1/11*100 = 9.0909 > 5 → 0.1579*PDW + 0.0496*5.5 + 3.6365
        assert scores.dale_chall == pytest.approx(5.3447, abs=1e-3)

    def test_dale_chall_no_adjustment_below_threshold(self) -> None:
        lex = FstSet(["the", "cat", "sat", "on", "mat", "dog", "ate", "a", "bone"])
        scores = readability_report(SIMPLE, familiar_words=lex).scores
        # PDW = 0 → 0.0496 * 5.5 only
        assert scores.dale_chall == pytest.approx(0.2728, abs=1e-3)

    def test_empty_text_all_none(self) -> None:
        scores = readability_report("").scores
        assert scores.flesch_reading_ease is None
        assert scores.flesch_kincaid_grade is None
        assert scores.automated_readability_index is None
        assert scores.coleman_liau_index is None
        assert scores.smog_index is None
        assert scores.gunning_fog is None
        assert scores.dale_chall is None
        assert scores.lix is None
        assert scores.rix is None
        assert scores.smog_valid is False

    def test_smog_valid_on_long_text(self) -> None:
        text = "The regulatory intermediary consolidated operations. " * 35
        scores = readability_report(text).scores
        assert scores.smog_valid is True
        assert scores.smog_index is not None and scores.smog_index > 3.2

    def test_naive_fog_at_least_default(self) -> None:
        text = (
            "We visited Wisconsin. The state-of-the-art trespasses "
            "created extraordinary responses yesterday."
        )
        strict = readability_report(text).scores
        naive = readability_report(
            text,
            fog_exclude_suffixes=False,
            fog_exclude_proper_nouns=False,
            fog_exclude_compounds=False,
        ).scores
        assert naive.gunning_fog is not None and strict.gunning_fog is not None
        assert naive.gunning_fog >= strict.gunning_fog

    def test_score_counts_pure(self) -> None:
        counts = TextCounts(
            sentences=30,
            words=600,
            letters=2700,
            letters_and_digits=2760,
            syllables=900,
            polysyllable_words=60,
            fog_complex_words=48,
            long_words=120,
            unfamiliar_words=None,
        )
        scores = score_counts(counts)
        assert scores.flesch_kincaid_grade == pytest.approx(
            0.39 * 20 + 11.8 * 1.5 - 15.59, abs=1e-9
        )
        assert scores.smog_valid is True
        assert scores.rix == pytest.approx(4.0)

    def test_to_dict_shape(self) -> None:
        d = readability_report(SIMPLE).to_dict()
        assert set(d) == {"counts", "scores"}
        assert "dale_chall" not in d["scores"]
        assert d["scores"]["smog_valid"] is False

    def test_report_types(self) -> None:
        report = readability_report(SIMPLE)
        assert isinstance(report, ReadabilityReport)
        assert isinstance(report.counts, TextCounts)
        assert isinstance(report.scores, ReadabilityScores)


# ── Convenience functions ──────────────────────────────────────────────────


class TestConvenience:
    def test_one_shot_helpers_match_report(self) -> None:
        scores = readability_report(SIMPLE).scores
        assert flesch_kincaid_grade(SIMPLE) == scores.flesch_kincaid_grade
        assert flesch_reading_ease(SIMPLE) == scores.flesch_reading_ease
        assert gunning_fog(SIMPLE) == scores.gunning_fog

    def test_one_shot_helpers_none_on_empty(self) -> None:
        assert flesch_kincaid_grade("") is None
        assert flesch_reading_ease("") is None
        assert gunning_fog("") is None


# ── Syllables ──────────────────────────────────────────────────────────────


class TestSyllables:
    def test_known_words_via_map(self) -> None:
        # CMUdict counts, exact lookup
        assert syllable_count("fire") == 2
        assert syllable_count("extraordinary") == 6
        assert syllable_count("bureaucracy") == 4
        assert syllable_count("cat") == 1

    def test_heuristic_fallback(self) -> None:
        # not in CMUdict → heuristic
        assert syllable_count("frobnicator") == 4
        assert syllable_count("zzz") == 1

    def test_non_empty_minimum_one(self) -> None:
        for token in ["x", "東京", "🎉", "123", "-"]:
            assert syllable_count(token) >= 1
        assert syllable_count("") == 0

    def test_case_insensitive(self) -> None:
        assert syllable_count("Extraordinary") == syllable_count("extraordinary")

    def test_default_map_loads_and_caches(self) -> None:
        m1 = default_syllable_map()
        m2 = default_syllable_map()
        assert m1 is m2
        assert len(m1) > 100_000
        assert "fire" in m1
        assert m1.get("fire") == 2

    def test_heuristic_accuracy_on_cmudict_sample(self) -> None:
        """Pin the tuned heuristic ≥90% exact on the committed sample."""
        rows = [
            line.split("\t")
            for line in (FIXTURES / "syllables_cmudict_sample.tsv")
            .read_text(encoding="utf-8")
            .splitlines()
            if line and not line.startswith("#")
        ]
        assert len(rows) == 500
        exact = sum(1 for word, n in rows if syllable_count(word, use_syllable_map=False) == int(n))
        assert exact / len(rows) >= 0.90

    def test_map_accuracy_on_cmudict_sample(self) -> None:
        rows = [
            line.split("\t")
            for line in (FIXTURES / "syllables_cmudict_sample.tsv")
            .read_text(encoding="utf-8")
            .splitlines()
            if line and not line.startswith("#")
        ]
        assert all(syllable_count(word) == int(n) for word, n in rows)


# ── Unicode boundaries ─────────────────────────────────────────────────────


class TestUnicode:
    @pytest.mark.parametrize(
        "text",
        [
            "東京は日本の首都です。大阪も大きい。",
            "Emoji 🎉 party 🎊 text!",
            "Ce texte français contient des accents: café, déjà, naïve.",
            "Mixed 中文 and English text with ½ symbols.",
            "\u2028line sep\u2029paragraph sep",
        ],
    )
    def test_deterministic_and_panic_free(self, text: str) -> None:
        a = readability_report(text)
        b = readability_report(text)
        assert a == b
        assert a.counts.syllables >= a.counts.words

    def test_cjk_counts_do_not_crash_scoring(self) -> None:
        report = readability_report("東京は日本の首都です。")
        # English-calibrated scores on CJK are not meaningful but must
        # be finite and deterministic.
        if report.scores.flesch_kincaid_grade is not None:
            assert report.scores.flesch_kincaid_grade == pytest.approx(
                readability_report("東京は日本の首都です。").scores.flesch_kincaid_grade
            )
