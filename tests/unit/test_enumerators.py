"""Tests for :func:`kaos_nlp_core.segmentation.parse_enumerator` at the
PyO3 boundary.

The parser is intentionally non-discriminating per Q6 of the design
reference: it reports the leading enumerator-shaped token and leaves
context-based filtering (e.g., distinguishing `e.g.` from a real `e.`
list marker) to the heading scorer downstream. These tests pin that
contract.
"""

from __future__ import annotations

import pickle

import pytest

from kaos_nlp_core.segmentation import Enumerator, parse_enumerator


def _kind(s: str) -> str | None:
    e = parse_enumerator(s)
    return e.kind if e is not None else None


# ─── Decimal ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "src,segments,depth",
    [
        ("1. Introduction", [1], 1),
        ("12. Heading", [12], 1),
        ("1.2 Definitions", [1, 2], 2),
        ("1.2.3 Subitem", [1, 2, 3], 3),
        ("1.2.3.4 Deepest", [1, 2, 3, 4], 4),
    ],
)
def test_decimal_segments(src: str, segments: list[int], depth: int) -> None:
    e = parse_enumerator(src)
    assert e is not None
    assert e.kind == "decimal"
    assert e.depth == depth
    assert e.segments() == segments


def test_decimal_five_segment_rejected() -> None:
    assert parse_enumerator("1.2.3.4.5 Too deep") is None


def test_decimal_segment_overflow_rejected() -> None:
    # Segments must fit in u8.
    assert parse_enumerator("258.3 X") is None


def test_decimal_single_without_period_rejected() -> None:
    # `1` without trailing dot is not a list marker.
    assert parse_enumerator("1 Item") is None


# ─── Alpha vs Roman (Pandoc rule) ──────────────────────────────────────────


@pytest.mark.parametrize(
    "src,kind",
    [
        # Pandoc rule: single-letter I/V are Roman, others are Alpha.
        ("I. text", "roman_upper"),
        ("V. text", "roman_upper"),
        ("i. text", "roman_lower"),
        ("v. text", "roman_lower"),
        ("A. First", "alpha_upper"),
        ("c. third", "alpha_lower"),
        ("M. text", "alpha_upper"),
        ("L. text", "alpha_upper"),
        # Multi-letter is always Roman if it parses.
        ("II. text", "roman_upper"),
        ("XIII. text", "roman_upper"),
        ("iv. text", "roman_lower"),
        ("xliii. text", "roman_lower"),
    ],
)
def test_alpha_roman_pandoc(src: str, kind: str) -> None:
    assert _kind(src) == kind


def test_alpha_roman_value_extraction() -> None:
    e = parse_enumerator("A. First")
    assert e is not None and e.value == 1
    e = parse_enumerator("XIII. text")
    assert e is not None and e.value == 13


def test_multiletter_non_roman_rejected() -> None:
    assert parse_enumerator("AB. text") is None
    assert parse_enumerator("XYZ. text") is None


# ─── Parenthetical ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "src,kind,value",
    [
        ("(a) item", "paren_alpha", 1),
        ("(B) item", "paren_alpha", 2),
        # F-R7: single-segment decimals store the value directly.
        ("(1) item", "paren_decimal", 1),
        ("(42) item", "paren_decimal", 42),
        ("(iv) item", "paren_roman", 4),
        ("(III) item", "paren_roman", 3),
    ],
)
def test_parenthetical(src: str, kind: str, value: int) -> None:
    e = parse_enumerator(src)
    assert e is not None
    assert e.kind == kind
    assert e.value == value


@pytest.mark.parametrize(
    "src",
    [
        "a) item",  # half-open
        "(a item",  # missing close
        "( a ) item",  # inner whitespace
        "() item",  # empty
        "(a.1) item",  # inner dot
    ],
)
def test_parenthetical_invalid_forms_rejected(src: str) -> None:
    assert parse_enumerator(src) is None


# ─── Word-prefixed ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "src,kind",
    [
        ("Section 5 Title", "section_word"),
        ("Sec. 5 Title", "section_word"),
        ("Section 5.2 Definitions", "section_word"),
        ("§ 5 Title", "section"),
        ("§ 5.2.3 Sub", "section"),
        ("Chapter 7 Title", "chapter_word"),
        ("Title 11 Bankruptcy", "chapter_word"),
        ("Subtitle B Heading", "chapter_word"),
        ("Part X Filings", "chapter_word"),
        ("Appendix A Tables", "chapter_word"),
        ("Schedule A Items", "chapter_word"),
        ("Subpart B Title", "subpart_word"),
        ("Subchapter II Description", "subpart_word"),
        ("Article III Authority", "subpart_word"),
        ("Paragraph 3 follows", "subpart_word"),
    ],
)
def test_word_prefixed(src: str, kind: str) -> None:
    assert _kind(src) == kind


def test_word_prefix_case_insensitive() -> None:
    for src in ["SECTION 5", "section 5", "Section 5"]:
        assert _kind(src) == "section_word"


def test_word_prefix_no_space_rejected() -> None:
    # OCR artefact, not a drafting-style enumerator.
    assert parse_enumerator("Sec.5 Title") is None


def test_subchapter_wins_over_sub() -> None:
    # LeftmostLongest must pick `Subchapter` over a hypothetical `Sub` prefix.
    e = parse_enumerator("Subchapter II Topic")
    assert e is not None
    assert e.kind == "subpart_word"


# ─── Char offset round-trip (FFI boundary regression net) ──────────────────


def test_offsets_are_char_offsets_on_unicode() -> None:
    # Section sigil U+00A7 is 2 bytes UTF-8, 1 char. The binding must report
    # CHAR offsets, so prefix_end must be 5 ('§ 5 ' = 4 chars then 'T').
    src = "§ 5 Title"
    e = parse_enumerator(src)
    assert e is not None
    assert e.kind == "section"
    # § (1 char) + space (1) + 5 (1) + space (1) = 4 → prefix_end at index 4.
    assert e.prefix_end == 4
    # Slice using char offsets must work in Python str indexing.
    assert src[e.prefix_end :] == "Title"


def test_offsets_in_bounds_for_each_kind() -> None:
    cases = [
        "1. Introduction",
        "(a) Definitions",
        "Section 5 Title",
        "I. Background",
    ]
    for src in cases:
        e = parse_enumerator(src)
        assert e is not None
        assert 0 <= e.raw_start <= e.raw_end <= e.prefix_end <= len(src)


# ─── Negatives — empty + non-enumerator ────────────────────────────────────


def test_empty_input_returns_none() -> None:
    assert parse_enumerator("") is None


def test_pub_l_does_not_match() -> None:
    # `Pub.` is not in the prefix lexicon and doesn't form a valid bare-letter
    # enumerator (multi-letter, non-Roman).
    assert parse_enumerator("Pub. L. 123") is None


def test_recall_first_e_g_parses_as_alpha() -> None:
    """Documented recall-first design: `e.g.` starts with `e.` which is a
    valid AlphaLower marker. The heading scorer is responsible for
    contextual filtering, NOT the parser."""
    e = parse_enumerator("e.g. text")
    assert e is not None
    assert e.kind == "alpha_lower"


# ─── segments() helper ─────────────────────────────────────────────────────


def test_segments_for_decimal() -> None:
    e = parse_enumerator("1.2.3 Sub")
    assert e is not None
    assert e.segments() == [1, 2, 3]


def test_segments_for_paren_decimal_is_single_value() -> None:
    e = parse_enumerator("(42) item")
    assert e is not None
    # ParenDecimal segments() yields the depth-1 segment list, NOT the packed value.
    assert e.segments() == [42]


def test_segments_for_alpha_returns_value() -> None:
    e = parse_enumerator("c. third")
    assert e is not None
    assert e.segments() == [3]


# ─── Pickle round-trip ─────────────────────────────────────────────────────


def test_pickle_round_trip() -> None:
    cases = ["1.2.3 Sub", "(iv) item", "Section 5 Title", "§ 5 Header"]
    for src in cases:
        e = parse_enumerator(src)
        assert e is not None
        copy = pickle.loads(pickle.dumps(e))
        assert copy.kind == e.kind
        assert copy.value == e.value
        assert copy.depth == e.depth


def test_repr_is_informative() -> None:
    e = parse_enumerator("1.2 Definitions")
    assert e is not None
    assert "Enumerator" in repr(e)
    assert "depth=2" in repr(e)


def test_re_export_class() -> None:
    e = parse_enumerator("1. x")
    assert isinstance(e, Enumerator)


# ─── Pluggable word-prefix lexicons (P7.0e) ────────────────────────────────


@pytest.mark.parametrize(
    "src,lexicon,expected_kind",
    [
        # English legal (default).
        ("Section 5 Title", None, "section_word"),
        ("Section 5 Title", "english_legal_us", "section_word"),
        # French legal.
        ("Article 5 Texte", "french_legal", "subpart_word"),
        ("Chapitre 2 Titre", "french_legal", "chapter_word"),
        ("Section 4 Détails", "french_legal", "section_word"),
        ("Annexe A Tableaux", "french_legal", "chapter_word"),
        # German legal.
        ("Artikel 5 Text", "german_legal", "subpart_word"),
        ("Kapitel 2 Titel", "german_legal", "chapter_word"),
        ("Abschnitt 4 Details", "german_legal", "section_word"),
        # Spanish legal — diacritic-bearing words.
        ("Artículo 5 Texto", "spanish_legal", "subpart_word"),
        ("Capítulo 2 Título", "spanish_legal", "chapter_word"),
        ("Sección 3 Detalles", "spanish_legal", "section_word"),
        # Italian legal.
        ("Articolo 5 Testo", "italian_legal", "subpart_word"),
        ("Capitolo 2 Titolo", "italian_legal", "chapter_word"),
        ("Sezione 4 Dettagli", "italian_legal", "section_word"),
        # Portuguese legal — Iberian + Brazilian spellings.
        ("Artigo 5 Texto", "portuguese_legal", "subpart_word"),
        ("Secção 3 Detalhes", "portuguese_legal", "section_word"),
        ("Seção 3 Detalhes", "portuguese_legal", "section_word"),
    ],
)
def test_lexicon_registry(src: str, lexicon: str | None, expected_kind: str) -> None:
    e = parse_enumerator(src, lexicon=lexicon)
    assert e is not None, f"expected match for {src!r} (lexicon={lexicon})"
    assert e.kind == expected_kind


def test_lexicons_dont_cross_match() -> None:
    """The English-legal-US lexicon must not fire on Italian `Capitolo`."""
    src = "Capitolo 2 Titolo"
    assert parse_enumerator(src, lexicon="english_legal_us") is None
    assert parse_enumerator(src, lexicon="italian_legal") is not None


def test_unknown_lexicon_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown lexicon"):
        parse_enumerator("Section 5", lexicon="klingon_legal")


# ─── Markdown ATX (P7.0e) ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "src,depth",
    [
        ("# Title", 1),
        ("## Subtitle", 2),
        ("### Sub-sub", 3),
        ("#### Level 4", 4),
        ("##### Level 5", 5),
        ("###### Level 6", 6),
    ],
)
def test_markdown_atx_h1_through_h6(src: str, depth: int) -> None:
    e = parse_enumerator(src, lexicon="markdown_atx")
    assert e is not None
    assert e.kind == "section_word"
    assert e.depth == depth


def test_markdown_atx_no_space_rejected() -> None:
    assert parse_enumerator("#NoSpace", lexicon="markdown_atx") is None


def test_markdown_atx_falls_through_to_other_kinds() -> None:
    """A markdown_atx lexicon still parses decimal/parenthetical/etc. when
    the line doesn't start with `#`."""
    e = parse_enumerator("1. Heading", lexicon="markdown_atx")
    assert e is not None
    assert e.kind == "decimal"


# ─── Custom lexicon (P7.0e) ─────────────────────────────────────────────────


def test_custom_lexicon_software_docs_smoke() -> None:
    """A custom lexicon for software-documentation conventions
    (Step/Phase/Stage)."""
    custom = [("Step", "section_word"), ("Phase", "chapter_word")]
    e = parse_enumerator("Step 3 Boil water", custom_lexicon=custom)
    assert e is not None
    assert e.kind == "section_word"
    e = parse_enumerator("Phase 2 Cleanup", custom_lexicon=custom)
    assert e is not None
    assert e.kind == "chapter_word"


def test_custom_lexicon_overrides_named_lexicon() -> None:
    """If both `lexicon` and `custom_lexicon` are supplied the custom one
    wins (so callers can extend a built-in by re-listing it)."""
    custom = [("Step", "section_word")]
    e = parse_enumerator("Step 3 Boil", lexicon="german_legal", custom_lexicon=custom)
    assert e is not None
    assert e.kind == "section_word"


def test_custom_lexicon_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown enum kind"):
        parse_enumerator("Foo 5 Bar", custom_lexicon=[("Foo", "made_up_kind_42")])


def test_custom_lexicon_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        parse_enumerator("Foo 5 Bar", custom_lexicon=[])
