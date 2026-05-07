"""Tests for string distance and similarity algorithms."""

import pytest

from kaos_nlp_core.algorithms import (
    compare_batch,
    damerau_levenshtein,
    hamming,
    jaro,
    jaro_winkler,
    lcs_distance,
    lcs_length,
    levenshtein,
    longest_common_substring,
    longest_common_substring_length,
    metaphone_distance,
    metaphone_encode,
    ngram_cosine,
    ngram_jaccard,
    ngram_overlap,
    osa,
    sorensen_dice,
    soundex_distance,
    soundex_encode,
    token_jaccard,
    token_ngram_cosine,
    token_ngram_jaccard,
    token_ngram_overlap,
)

# --- Edit distance ---


class TestLevenshtein:
    def test_basic(self):
        r = levenshtein("kitten", "sitting")
        assert r.distance == 3.0
        assert 0.0 < r.normalized < 1.0
        assert 0.0 < r.similarity < 1.0

    def test_identical(self):
        r = levenshtein("hello", "hello")
        assert r.distance == 0.0
        assert r.similarity == 1.0

    def test_empty(self):
        r = levenshtein("", "")
        assert r.distance == 0.0

    def test_one_empty(self):
        r = levenshtein("abc", "")
        assert r.distance == 3.0

    def test_unicode(self):
        r = levenshtein("café", "cafe")
        assert r.distance == 1.0

    def test_compare_batch(self):
        results = compare_batch(
            [("kitten", "sitting"), ("hello", "hello")],
            algorithm="levenshtein",
        )
        assert results[0].distance == 3.0
        assert results[1].similarity == 1.0


class TestDamerauLevenshtein:
    def test_transposition(self):
        r = damerau_levenshtein("ab", "ba")
        assert r.distance == 1.0

    def test_same_as_levenshtein_no_transposition(self):
        r = damerau_levenshtein("abc", "def")
        assert r.distance == 3.0


class TestOsa:
    def test_basic(self):
        r = osa("CA", "ABC")
        assert r.distance >= 1.0


class TestHamming:
    def test_basic(self):
        r = hamming("karolin", "kathrin")
        assert r.distance == 3.0

    def test_unequal_length(self):
        with pytest.raises(ValueError):
            hamming("abc", "ab")


class TestJaro:
    def test_basic(self):
        r = jaro("martha", "marhta")
        assert r.similarity > 0.94


class TestJaroWinkler:
    def test_basic(self):
        r = jaro_winkler("martha", "marhta")
        assert r.similarity > 0.96

    def test_prefix_boost(self):
        j = jaro("dwayne", "duane")
        jw = jaro_winkler("dwayne", "duane")
        # Jaro-Winkler should be >= Jaro for strings with common prefix
        assert jw.similarity >= j.similarity


class TestSorensenDice:
    def test_identical(self):
        r = sorensen_dice("night", "night")
        assert r.similarity == 1.0

    def test_different(self):
        r = sorensen_dice("night", "nacht")
        assert 0.0 < r.similarity < 1.0


# --- Phonetic ---


class TestSoundex:
    def test_encode(self):
        assert soundex_encode("Robert") == "R163"
        assert soundex_encode("Rupert") == "R163"

    def test_distance_same_code(self):
        r = soundex_distance("Robert", "Rupert")
        assert r.similarity >= 0.5

    def test_identical(self):
        r = soundex_distance("Smith", "Smith")
        assert r.similarity == 1.0


class TestMetaphone:
    def test_encode_same(self):
        assert metaphone_encode("Smith") == metaphone_encode("Smyth")

    def test_distance(self):
        r = metaphone_distance("Smith", "Smyth")
        assert r.similarity == 1.0


# --- Sequence ---


class TestLcs:
    def test_basic(self):
        r = lcs_distance("kitten", "sitting")
        assert r.distance == 3.0

    def test_length(self):
        assert lcs_length("abcde", "ace") == 3  # "ace"


class TestLongestCommonSubstring:
    def test_basic(self):
        r = longest_common_substring("abcdef", "xbcdey")
        assert abs(r.similarity - 4.0 / 6.0) < 1e-10

    def test_length(self):
        assert longest_common_substring_length("abcdef", "xbcdey") == 4


# --- N-gram ---


class TestNgramJaccard:
    def test_identical(self):
        r = ngram_jaccard("hello", "hello", n=2)
        assert r.similarity == 1.0

    def test_different(self):
        r = ngram_jaccard("abc", "xyz", n=2)
        assert r.similarity == 0.0

    def test_trigram(self):
        r = ngram_jaccard("hello world", "hello world", n=3)
        assert r.similarity == 1.0


class TestNgramCosine:
    def test_identical(self):
        r = ngram_cosine("hello", "hello", n=2)
        assert abs(r.similarity - 1.0) < 1e-10

    def test_partial(self):
        r = ngram_cosine("night", "nacht", n=2)
        assert 0.0 < r.similarity < 1.0


class TestNgramOverlap:
    def test_subset(self):
        r = ngram_overlap("ab", "abc", n=2)
        assert r.similarity == 1.0

    def test_no_overlap(self):
        r = ngram_overlap("abc", "xyz", n=2)
        assert r.similarity == 0.0


# --- Token n-gram ---


class TestTokenJaccard:
    def test_identical(self):
        r = token_jaccard("the quick brown fox", "the quick brown fox")
        assert r.similarity == 1.0

    def test_partial(self):
        r = token_jaccard("the quick brown fox", "a quick brown dog", lowercase=True)
        # shared: {quick, brown} = 2, union: {the, quick, brown, fox, a, dog} = 6
        assert abs(r.similarity - 2.0 / 6.0) < 1e-10

    def test_no_overlap(self):
        r = token_jaccard("hello world", "foo bar")
        assert r.similarity == 0.0

    def test_case_sensitive(self):
        r = token_jaccard("Hello", "hello")
        assert r.similarity == 0.0

    def test_case_insensitive(self):
        r = token_jaccard("Hello", "hello", lowercase=True)
        assert r.similarity == 1.0

    def test_empty(self):
        r = token_jaccard("", "")
        assert r.similarity == 1.0

    def test_strips_punctuation(self):
        r = token_jaccard("hello, world!", "hello world")
        assert r.similarity == 1.0

    def test_repeated_words(self):
        r = token_jaccard("the the the", "the the")
        # multiset: a={the:3}, b={the:2}, intersection=min(3,2)=2, union=max(3,2)=3
        assert abs(r.similarity - 2.0 / 3.0) < 1e-10


class TestTokenNgramJaccard:
    def test_identical(self):
        r = token_ngram_jaccard("the quick brown fox jumps", "the quick brown fox jumps", n=2)
        assert r.similarity == 1.0

    def test_partial(self):
        r = token_ngram_jaccard("the quick brown fox", "a quick brown dog", n=2, lowercase=True)
        # bigrams A: {the quick, quick brown, brown fox}
        # bigrams B: {a quick, quick brown, brown dog}
        # intersection: {quick brown} = 1, union = 5
        assert abs(r.similarity - 1.0 / 5.0) < 1e-10

    def test_no_overlap(self):
        r = token_ngram_jaccard("hello world", "foo bar", n=2)
        assert r.similarity == 0.0

    def test_trigram(self):
        r = token_ngram_jaccard("a b c d e", "a b c d e", n=3, lowercase=True)
        assert r.similarity == 1.0

    def test_single_token(self):
        r = token_ngram_jaccard("hello", "hello", n=2)
        assert r.similarity == 1.0


class TestTokenNgramCosine:
    def test_identical(self):
        r = token_ngram_cosine("the quick brown fox", "the quick brown fox", n=2)
        assert abs(r.similarity - 1.0) < 1e-10

    def test_partial(self):
        r = token_ngram_cosine("the quick brown fox", "a quick brown dog", n=2, lowercase=True)
        assert 0.0 < r.similarity < 1.0

    def test_no_overlap(self):
        r = token_ngram_cosine("hello world", "foo bar", n=2)
        assert r.similarity == 0.0


class TestTokenNgramOverlap:
    def test_subset(self):
        r = token_ngram_overlap("quick brown", "the quick brown fox", n=2, lowercase=True)
        assert r.similarity == 1.0

    def test_no_overlap(self):
        r = token_ngram_overlap("hello world", "foo bar", n=2)
        assert r.similarity == 0.0


# ── Edge cases across all algorithms ────────────────────────────────────────


class TestAlgorithmEdgeCases:
    """Edge cases that should work without error across algorithms."""

    def test_hamming_both_empty(self):
        r = hamming("", "")
        assert r.distance == 0.0

    def test_sorensen_dice_empty_strings(self):
        r = sorensen_dice("", "")
        # Either 1.0 (identical empty sets) or 0.0 (no bigrams) — both valid
        assert r.similarity >= 0.0

    def test_sorensen_dice_one_empty(self):
        r = sorensen_dice("hello", "")
        assert r.similarity == 0.0

    def test_levenshtein_unicode(self):
        r = levenshtein("café", "cafe")
        assert r.distance == 1.0  # é → e

    def test_jaro_winkler_unicode(self):
        r = jaro_winkler("café", "café")
        assert r.similarity == 1.0

    def test_token_jaccard_unicode(self):
        r = token_jaccard("café résumé", "café résumé")
        assert r.similarity == 1.0

    def test_ngram_jaccard_unicode(self):
        r = ngram_jaccard("café", "café")
        assert r.similarity == 1.0

    def test_levenshtein_both_empty(self):
        r = levenshtein("", "")
        assert r.distance == 0.0

    def test_damerau_levenshtein_both_empty(self):
        r = damerau_levenshtein("", "")
        assert r.distance == 0.0

    def test_osa_basic(self):
        """OSA should count transpositions as 1 edit."""
        r = osa("ab", "ba")
        assert r.distance == 1.0

    def test_osa_vs_levenshtein(self):
        """OSA allows transposition, Levenshtein doesn't."""
        r_osa = osa("ab", "ba")
        r_lev = levenshtein("ab", "ba")
        assert r_osa.distance <= r_lev.distance

    def test_lcs_empty(self):
        assert lcs_length("", "hello") == 0
        assert lcs_length("", "") == 0

    def test_longest_common_substring_empty(self):
        assert longest_common_substring_length("", "hello") == 0

    def test_soundex_unicode(self):
        result = soundex_encode("Müller")
        assert len(result) > 0

    def test_metaphone_unicode(self):
        result = metaphone_encode("Müller")
        assert len(result) > 0
