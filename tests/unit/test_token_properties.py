"""Tests for Python-exposed token property classification."""

from __future__ import annotations

from kaos_nlp_core.token_properties import (
    classify_token,
    classify_tokens,
    has_emoji,
    is_abbreviation,
    is_alphanumeric_word,
    is_numeric_word,
    is_uppercase_word,
)


class TestTokenProperties:
    def test_classify_token(self) -> None:
        flags = classify_token("U.S.A.")
        assert flags.is_abbreviation
        assert flags.has_punctuation

    def test_classify_tokens_batch(self) -> None:
        batch = classify_tokens(["FBI", "abc123", "😀"])
        assert batch[0].is_uppercase_word
        assert batch[1].is_alphanumeric_word
        assert batch[2].has_emoji

    def test_predicate_helpers(self) -> None:
        assert is_uppercase_word("FBI")
        assert is_numeric_word("12345")
        assert is_alphanumeric_word("abc123")
        assert is_abbreviation("Dr.")
        assert has_emoji("hello 😀")
