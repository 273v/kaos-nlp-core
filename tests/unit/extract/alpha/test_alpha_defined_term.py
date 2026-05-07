"""Tests for AlphaDefinedTermExtractor — parenthesized contract definitions."""

from __future__ import annotations

import pytest

from kaos_nlp_core.extract.alpha.defined_term import (
    AlphaDefinedTermExtractor,
    DefinedTermMatch,
)


@pytest.fixture
def extractor() -> AlphaDefinedTermExtractor:
    return AlphaDefinedTermExtractor()


class TestBasicPatterns:
    def test_the_quoted(self, extractor: AlphaDefinedTermExtractor) -> None:
        out = list(extractor.extract_values('Acme Inc. (the "Borrower") borrowed.'))
        assert any(m.term == "Borrower" and m.intro_phrase == "the" for m in out)

    def test_bare_quoted(self, extractor: AlphaDefinedTermExtractor) -> None:
        out = list(extractor.extract_values('The party referred to as ("Tenant") shall pay.'))
        assert any(m.term == "Tenant" and m.intro_phrase is None for m in out)

    def test_hereinafter_referred_to_as(self, extractor: AlphaDefinedTermExtractor) -> None:
        text = 'Foo Inc. (hereinafter referred to as "Foo") and Bar LLC.'
        out = list(extractor.extract_values(text))
        assert any(m.term == "Foo" and m.intro_phrase == "hereinafter referred to as" for m in out)

    def test_collectively(self, extractor: AlphaDefinedTermExtractor) -> None:
        text = 'The parties (collectively, the "Parties") agree.'
        out = list(extractor.extract_values(text))
        assert any(m.term == "Parties" and "collectively" in (m.intro_phrase or "") for m in out)

    def test_individually(self, extractor: AlphaDefinedTermExtractor) -> None:
        text = '(individually, a "Party")'
        out = list(extractor.extract_values(text))
        assert any(m.term == "Party" and "individually" in (m.intro_phrase or "") for m in out)


class TestMultiTermInOnePeren:
    def test_two_terms(self, extractor: AlphaDefinedTermExtractor) -> None:
        text = 'The premises (the "Premises" or the "Real Property") are leased.'
        terms = [m.term for m in extractor.extract_values(text)]
        assert "Premises" in terms
        assert "Real Property" in terms

    def test_intro_resets_per_term(self, extractor: AlphaDefinedTermExtractor) -> None:
        text = 'The parties (collectively, the "Parties" and individually, a "Party") agree.'
        out = list(extractor.extract_values(text))
        parties = next(m for m in out if m.term == "Parties")
        party = next(m for m in out if m.term == "Party")
        assert "collectively" in (parties.intro_phrase or "")
        assert "individually" in (party.intro_phrase or "")


class TestQuoteStyles:
    def test_double_quotes(self, extractor: AlphaDefinedTermExtractor) -> None:
        out = list(extractor.extract_values('(the "X")'))
        assert out and out[0].quote_style == "double"

    def test_curly_quotes(self, extractor: AlphaDefinedTermExtractor) -> None:
        out = list(extractor.extract_values("(the “X”)"))
        assert out and out[0].quote_style == "curly"


class TestApostrophes:
    def test_dont_not_a_term(self, extractor: AlphaDefinedTermExtractor) -> None:
        # Apostrophes in regular text shouldn't fire as defined terms.
        out = list(extractor.extract_values("don't and won't are not terms"))
        assert out == []

    def test_short_single_quoted_skipped(self, extractor: AlphaDefinedTermExtractor) -> None:
        # Single-quote content of length ≤ 2 is treated as apostrophe.
        out = list(extractor.extract_values("(the 'a')"))
        assert out == []


class TestDefinitionClause:
    def test_captures_preceding_sentence(self, extractor: AlphaDefinedTermExtractor) -> None:
        text = 'Acme Corporation, a Delaware corporation (the "Borrower"), borrowed.'
        out = list(extractor.extract_values(text))
        borrower = next(m for m in out if m.term == "Borrower")
        assert borrower.definition_clause is not None
        assert "Acme" in borrower.definition_clause

    def test_no_clause_when_term_at_start(self, extractor: AlphaDefinedTermExtractor) -> None:
        out = list(extractor.extract_values('(the "Borrower") borrowed money.'))
        borrower = next(m for m in out if m.term == "Borrower")
        assert borrower.definition_clause is None


class TestSpans:
    def test_span_covers_full_paren(self, extractor: AlphaDefinedTermExtractor) -> None:
        text = 'Acme Inc. (the "Borrower") owes a debt.'
        spans = list(extractor.extract_spans(text))
        assert spans
        s = spans[0]
        captured = text[s.start : s.end]
        assert captured.startswith("(")
        assert captured.endswith(")")
        assert "Borrower" in captured


class TestEdgeCases:
    def test_empty(self, extractor: AlphaDefinedTermExtractor) -> None:
        assert list(extractor.extract_values("")) == []

    def test_no_parens(self, extractor: AlphaDefinedTermExtractor) -> None:
        out = list(extractor.extract_values('The "Borrower" without parens.'))
        # Without the paren wrapper, this doesn't match — the paren is
        # the high-precision anchor.
        assert out == []

    def test_unclosed_quote_skipped(self, extractor: AlphaDefinedTermExtractor) -> None:
        out = list(extractor.extract_values('(the "unclosed) text'))
        # The mismatched quote shouldn't crash.
        assert all(isinstance(m, DefinedTermMatch) for m in out)

    def test_nested_parens_only_outer(self, extractor: AlphaDefinedTermExtractor) -> None:
        text = "Acme (a Delaware corp. (Inc.)) operates here."
        out = list(extractor.extract_values(text))
        # No quoted terms inside, so no matches.
        assert out == []
