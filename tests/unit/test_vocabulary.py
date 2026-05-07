"""Tests for kaos_nlp_core.vocabulary — lexicon-aware token frequency."""

from __future__ import annotations

import pytest

from kaos_nlp_core.lexicon import Lexicon
from kaos_nlp_core.matching import FstSet
from kaos_nlp_core.vocabulary import (
    TokenCount,
    VocabularyCounts,
    token_frequency,
)


@pytest.fixture(scope="module")
def small_fst() -> FstSet:
    return FstSet(["the", "quick", "brown", "fox", "agreement", "contract", "party", "shall"])


@pytest.fixture(scope="module")
def small_lex() -> Lexicon:
    lex = Lexicon()
    for w in ["the", "fox", "contract", "agreement"]:
        lex.add_entry({"word": w})
    return lex


class TestBasicCounting:
    def test_empty_text(self) -> None:
        r = token_frequency("")
        assert isinstance(r, VocabularyCounts)
        assert r.total_tokens == 0
        assert r.kept_tokens == 0
        assert r.unique_terms == 0
        assert r.counts == ()
        assert r.coverage == 0.0

    def test_only_whitespace(self) -> None:
        r = token_frequency("   \t  \n  ")
        assert r.total_tokens == 0
        assert r.unique_terms == 0

    def test_single_token(self) -> None:
        r = token_frequency("hello")
        assert r.total_tokens == 1
        assert r.kept_tokens == 1
        assert r.unique_terms == 1
        assert r.counts[0] == TokenCount(text="hello", count=1)

    def test_repeated_tokens(self) -> None:
        r = token_frequency("the the the cat sat")
        assert r.total_tokens == 5
        d = r.to_dict()
        assert d["the"] == 3
        assert d["cat"] == 1
        assert d["sat"] == 1

    def test_sorted_descending(self) -> None:
        r = token_frequency("the the the cat cat dog")
        counts = [tc.count for tc in r]
        assert counts == sorted(counts, reverse=True)

    def test_lowercase_default(self) -> None:
        r = token_frequency("Hello HELLO hello")
        assert r.unique_terms == 1
        assert r.to_dict() == {"hello": 3}

    def test_lowercase_off(self) -> None:
        r = token_frequency("Hello HELLO hello", lowercase=False)
        # Three distinct casings.
        assert r.unique_terms == 3


class TestLexiconFilter:
    def test_fst_set_filter(self, small_fst: FstSet) -> None:
        text = "The quick brown fox xyzzy contract"
        r = token_frequency(text, lexicon=small_fst)
        assert r.total_tokens == 6
        assert r.kept_tokens == 5  # xyzzy filtered
        d = r.to_dict()
        assert "xyzzy" not in d
        assert "the" in d
        assert "fox" in d

    def test_lexicon_filter(self, small_lex: Lexicon) -> None:
        text = "the cat ate the contract"
        r = token_frequency(text, lexicon=small_lex)
        # cat, ate not in lexicon
        d = r.to_dict()
        assert "cat" not in d
        assert "ate" not in d
        assert d["the"] == 2
        assert d["contract"] == 1

    def test_no_lexicon_keeps_all(self) -> None:
        r = token_frequency("xyzzy plover frobnicate")
        assert r.kept_tokens == 3
        assert r.coverage == 1.0

    def test_coverage_computation(self, small_fst: FstSet) -> None:
        # 4 of 6 alphabetic tokens are in-vocab; 2 are not.
        r = token_frequency("the the contract xyzzy unknown party", lexicon=small_fst)
        assert r.total_tokens == 6
        assert r.kept_tokens == 4
        assert r.coverage == pytest.approx(4 / 6, abs=1e-6)

    def test_invalid_lexicon_type(self) -> None:
        with pytest.raises(TypeError, match="contains"):
            token_frequency("hello", lexicon=42)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_invalid_lexicon_string(self) -> None:
        # str also has no 'contains' callable that returns bool by token,
        # but it does have __contains__ — so we explicitly reject non-FstSet/Lexicon.
        # str.contains doesn't exist as an attribute.
        with pytest.raises(TypeError):
            token_frequency("hello", lexicon="not a lexicon")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


class TestMinCountAndTopK:
    def test_min_count_drops_singletons(self) -> None:
        r = token_frequency("the the cat dog", min_count=2)
        # Only "the" survives; total/kept reflect pre-trim counts.
        assert r.unique_terms == 1
        assert r.counts[0].text == "the"
        assert r.total_tokens == 4
        assert r.kept_tokens == 4

    def test_min_count_one_keeps_all(self) -> None:
        r = token_frequency("the cat dog", min_count=1)
        assert r.unique_terms == 3

    def test_top_k_truncation(self) -> None:
        text = " ".join(["a", "a", "a", "b", "b", "c", "d", "e", "f"])
        r = token_frequency(text, top_k=3)
        assert r.unique_terms == 3
        # Most frequent first.
        assert r.counts[0].text == "a"

    def test_top_k_zero_via_top_method(self) -> None:
        r = token_frequency("the cat sat")
        assert r.top(0) == []

    def test_top_method_returns_list(self) -> None:
        r = token_frequency("the the the cat dog")
        out = r.top(2)
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0].text == "the"


class TestUnicode:
    def test_cjk_round_trip(self) -> None:
        r = token_frequency("東京 大阪 東京 京都")
        d = r.to_dict()
        assert d["東京"] == 2
        assert d["大阪"] == 1
        assert d["京都"] == 1

    def test_emoji_dropped_or_kept_consistently(self) -> None:
        # Emoji-only "tokens" may or may not survive tokenize_words punctuation
        # stripping; we just want the call not to crash and to return a sensible
        # object.
        r = token_frequency("hello 😀 world 🎉 hello")
        assert r.total_tokens >= 2
        d = r.to_dict()
        assert d.get("hello") == 2

    def test_diacritics_preserved(self) -> None:
        r = token_frequency("café résumé café")
        d = r.to_dict()
        assert d["café"] == 2
        assert d["résumé"] == 1


class TestPrefixTruncation:
    def test_prefix_stems(self) -> None:
        r = token_frequency("automobile automotive automated", prefix=4)
        # All three start with "auto" → collapse to one bucket.
        assert r.unique_terms == 1
        assert r.counts[0].text == "auto"
        assert r.counts[0].count == 3


class TestVocabularyCountsContainer:
    def test_iteration(self) -> None:
        r = token_frequency("the the cat")
        seen = list(r)
        assert all(isinstance(tc, TokenCount) for tc in seen)
        assert len(seen) == 2

    def test_len(self) -> None:
        r = token_frequency("the the cat dog")
        assert len(r) == 3

    def test_to_dict_round_trip(self) -> None:
        r = token_frequency("the the cat")
        d = r.to_dict()
        assert d == {"the": 2, "cat": 1}

    def test_immutable(self) -> None:
        r = token_frequency("hello")
        with pytest.raises((AttributeError, TypeError)):
            r.total_tokens = 999  # type: ignore[misc]  # ty: ignore[invalid-assignment]

    def test_coverage_no_lexicon_is_one(self) -> None:
        r = token_frequency("hello world")
        assert r.coverage == 1.0
