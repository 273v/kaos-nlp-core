"""Tests for kaos_nlp_core.concepts — graph-driven concept extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from kaos_nlp_core.concepts import (
    DEFAULT_STOP_HYPERNYMS,
    DEFAULT_STOP_HYPONYMS,
    Concept,
    extract_concepts,
)
from kaos_nlp_core.lexicon import OPENGLOSS_FILENAME, Lexicon


def _opengloss_available() -> bool:
    return (Path(__file__).resolve().parent.parent / "data" / OPENGLOSS_FILENAME).is_file()


# ── Synthetic lexicon — guaranteed to work without OpenGloss ─────────────


@pytest.fixture
def litigation_lex() -> Lexicon:
    """Tiny lexicon mapping legal terms → 'litigation' / 'legal proceeding'."""
    lex = Lexicon()
    for word, hypernyms in [
        ("plaintiff", ["litigant", "party"]),
        ("defendant", ["litigant", "party"]),
        ("complaint", ["legal document", "filing"]),
        ("summons", ["legal document"]),
        ("judgment", ["court ruling", "decision"]),
        ("appeal", ["legal action"]),
        ("deposition", ["legal procedure", "discovery"]),
        ("litigation", ["legal proceeding"]),
    ]:
        lex.add_entry(
            {
                "word": word,
                "all_hypernyms": hypernyms,
                "all_hyponyms": [],
            }
        )
    # Add a few hyponyms for testing the hyponym direction.
    lex.add_entry(
        {
            "word": "court",
            "all_hypernyms": ["body"],
            "all_hyponyms": [
                "appellate court",
                "supreme court",
                "trial court",
                "small claims court",
            ],
        }
    )
    return lex


class TestSyntheticLexicon:
    def test_hypernyms_surface_concept(self, litigation_lex: Lexicon) -> None:
        text = "The plaintiff filed a complaint and the defendant received a summons."
        concepts = extract_concepts(
            text,
            lexicon=litigation_lex,
            top_k=10,
            extra_stop_terms=set(),
        )
        terms = {c.term for c in concepts}
        # The aggregating concept should appear even though "litigant" /
        # "legal document" are never in the source text.
        assert "litigant" in terms or "legal document" in terms

    def test_concept_carries_source_terms(self, litigation_lex: Lexicon) -> None:
        text = "plaintiff defendant plaintiff defendant"
        concepts = extract_concepts(
            text,
            lexicon=litigation_lex,
            extra_stop_terms=set(),
        )
        litigant = next(c for c in concepts if c.term == "litigant")
        assert "plaintiff" in litigant.source_terms
        assert "defendant" in litigant.source_terms
        assert litigant.frequency == 4  # 2 + 2

    def test_hyponym_direction(self, litigation_lex: Lexicon) -> None:
        text = "The court ruled. The court issued an order. The court adjourned."
        concepts = extract_concepts(
            text,
            lexicon=litigation_lex,
            direction="hyponym",
            top_k=10,
            extra_stop_terms=set(),
        )
        # "court" hyponyms include appellate/supreme/trial court.
        terms = {c.term for c in concepts}
        assert "appellate court" in terms
        assert "supreme court" in terms

    def test_both_direction_returns_typed_records(self, litigation_lex: Lexicon) -> None:
        text = "The court issued a summons to the defendant."
        concepts = extract_concepts(
            text,
            lexicon=litigation_lex,
            direction="both",
            top_k=5,
            extra_stop_terms=set(),
        )
        assert all(isinstance(c, Concept) for c in concepts)
        # Both directions present.
        directions = {c.direction for c in concepts}
        assert "hypernym" in directions
        assert "hyponym" in directions

    def test_record_immutable(self, litigation_lex: Lexicon) -> None:
        concepts = extract_concepts(
            "plaintiff defendant", lexicon=litigation_lex, extra_stop_terms=set()
        )
        if not concepts:
            pytest.skip("no concepts produced; trivial text")
        with pytest.raises((AttributeError, TypeError)):
            concepts[0].score = 999.0  # type: ignore[misc]  # ty: ignore[invalid-assignment]


class TestStopTerms:
    def test_extra_stop_terms_filtered(self, litigation_lex: Lexicon) -> None:
        text = "plaintiff defendant"
        all_concepts = extract_concepts(text, lexicon=litigation_lex, extra_stop_terms=set())
        terms = {c.term for c in all_concepts}
        assert "litigant" in terms

        filtered = extract_concepts(
            text,
            lexicon=litigation_lex,
            extra_stop_terms={"litigant"},
        )
        assert "litigant" not in {c.term for c in filtered}

    def test_default_stop_lists_are_frozensets(self) -> None:
        assert isinstance(DEFAULT_STOP_HYPERNYMS, frozenset)
        assert isinstance(DEFAULT_STOP_HYPONYMS, frozenset)

    def test_empty_extra_keeps_defaults(self, litigation_lex: Lexicon) -> None:
        # The litigation_lex's hypernyms aren't in DEFAULT_STOP_HYPERNYMS,
        # so an empty extra-set should leave them visible.
        text = "plaintiff defendant"
        result = extract_concepts(text, lexicon=litigation_lex, extra_stop_terms=set())
        assert any(c.term == "litigant" for c in result)


class TestArguments:
    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            extract_concepts("text", direction="sideways")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_invalid_weight_raises(self, litigation_lex: Lexicon) -> None:
        with pytest.raises(ValueError, match="weight"):
            extract_concepts("text", lexicon=litigation_lex, weight="quadratic")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_invalid_max_depth_raises(self, litigation_lex: Lexicon) -> None:
        with pytest.raises(ValueError, match="max_depth"):
            extract_concepts("text", lexicon=litigation_lex, max_depth=99)
        with pytest.raises(ValueError, match="max_depth"):
            extract_concepts("text", lexicon=litigation_lex, max_depth=0)

    def test_top_k_truncates_per_direction(self, litigation_lex: Lexicon) -> None:
        text = "plaintiff defendant complaint summons judgment appeal deposition"
        concepts = extract_concepts(
            text,
            lexicon=litigation_lex,
            top_k=2,
            extra_stop_terms=set(),
        )
        assert len(concepts) <= 2

    def test_min_term_count_drops_singletons(self, litigation_lex: Lexicon) -> None:
        text = "plaintiff plaintiff defendant"
        # min_term_count=2 drops "defendant" (count=1).
        result = extract_concepts(
            text,
            lexicon=litigation_lex,
            min_term_count=2,
            extra_stop_terms=set(),
        )
        # Only plaintiff contributed → litigant gets only plaintiff sources.
        litigant = next((c for c in result if c.term == "litigant"), None)
        if litigant is not None:
            assert "defendant" not in litigant.source_terms

    def test_log_vs_linear_weight(self, litigation_lex: Lexicon) -> None:
        # Single high-frequency term: log saturates, linear amplifies.
        text = "plaintiff " * 100 + " defendant"
        log_res = extract_concepts(
            text, lexicon=litigation_lex, weight="log", extra_stop_terms=set()
        )
        lin_res = extract_concepts(
            text, lexicon=litigation_lex, weight="linear", extra_stop_terms=set()
        )
        log_scores = {c.term: c.score for c in log_res}
        lin_scores = {c.term: c.score for c in lin_res}
        # Linear should produce a much larger absolute number for the
        # litigant concept since it's count-weighted.
        if "litigant" in log_scores and "litigant" in lin_scores:
            assert lin_scores["litigant"] > log_scores["litigant"]


class TestEmptyAndEdgeCases:
    def test_empty_text(self, litigation_lex: Lexicon) -> None:
        result = extract_concepts("", lexicon=litigation_lex)
        assert result == []

    def test_no_in_vocab_terms(self, litigation_lex: Lexicon) -> None:
        result = extract_concepts("xyzzy plover frobnicate", lexicon=litigation_lex)
        assert result == []

    def test_word_with_no_hypernyms(self) -> None:
        lex = Lexicon()
        lex.add_entry({"word": "isolated"})  # no hypernyms
        result = extract_concepts("isolated isolated", lexicon=lex)
        assert result == []


@pytest.mark.skipif(not _opengloss_available(), reason="OpenGloss data file not present")
class TestOpenGlossDefault:
    def test_default_lexicon_loads_when_none(self) -> None:
        text = "The plaintiff filed a complaint and the defendant received a summons."
        result = extract_concepts(text, top_k=5)
        assert isinstance(result, list)
        assert all(isinstance(c, Concept) for c in result)

    def test_legal_text_surfaces_legal_concepts(self) -> None:
        text = (
            "WHEREAS, the Plaintiff filed a complaint and summons against the "
            "Defendant alleging breach of contract; and the Court issued a "
            "judgment after deposition. The Defendant filed an appeal."
        )
        result = extract_concepts(text, top_k=10)
        terms = {c.term.lower() for c in result}
        # Calibrated stop-list filters out 'communication', 'event', etc., so
        # legal-specific concepts should dominate.
        legal_indicators = {
            "legal action",
            "legal document",
            "legal procedure",
            "legal proceeding",
            "litigant",
        }
        assert legal_indicators & terms, f"Expected at least one legal concept in: {sorted(terms)}"

    def test_caching_is_idempotent(self) -> None:
        text = "The plaintiff filed a complaint."
        a = extract_concepts(text, top_k=5)
        b = extract_concepts(text, top_k=5)
        assert [(c.term, c.score) for c in a] == [(c.term, c.score) for c in b]
