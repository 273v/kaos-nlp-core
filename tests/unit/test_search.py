"""Tests for search_sentences and search_paragraphs PyO3 bindings.

Verifies BM25 search, character offset conversion (byte→char),
and round-trip correctness for ASCII, multi-byte Latin, CJK, and emoji text.
"""

from __future__ import annotations

from kaos_nlp_core.search import SegmentHit, search_paragraphs, search_sentences
from kaos_nlp_core.segmentation import PunktTokenizer

# ─── Fixtures ────────────────────────────────────────────────────────────────


MULTI_SENTENCE = "The cat sat on the mat. The dog chased a ball. A bird flew away."

MULTI_PARAGRAPH = (
    "The cat sat on the mat. It was a lazy day.\n\n"
    "The dog ran in the park. It was fun.\n\n"
    "A bird flew over the field. It sang a song."
)

LATIN_MULTIBYTE = "Le café est bon. Le thé est chaud. Le jus est frais."

EMOJI_TEXT = "I love 😀 cats. I love 🌍 dogs. I love 🎉 birds."


# ─── search_sentences tests ──────────────────────────────────────────────────


class TestSearchSentences:
    def test_basic_match(self):
        results = search_sentences(MULTI_SENTENCE, "cat mat")
        assert len(results) >= 1
        assert isinstance(results[0], SegmentHit)
        assert results[0].text == "The cat sat on the mat."
        assert results[0].score > 0

    def test_has_typed_fields(self):
        results = search_sentences(MULTI_SENTENCE, "cat")
        assert len(results) >= 1
        r = results[0]
        assert hasattr(r, "text")
        assert hasattr(r, "start")
        assert hasattr(r, "end")
        assert hasattr(r, "score")

    def test_round_trip_offsets_ascii(self):
        results = search_sentences(MULTI_SENTENCE, "dog ball")
        for r in results:
            assert MULTI_SENTENCE[r.start : r.end] == r.text

    def test_round_trip_offsets_latin(self):
        results = search_sentences(LATIN_MULTIBYTE, "café")
        assert len(results) >= 1
        for r in results:
            assert LATIN_MULTIBYTE[r.start : r.end] == r.text

    def test_round_trip_offsets_emoji(self):
        results = search_sentences(EMOJI_TEXT, "cats")
        assert len(results) >= 1
        for r in results:
            assert EMOJI_TEXT[r.start : r.end] == r.text

    def test_top_k_limit(self):
        results = search_sentences(MULTI_SENTENCE, "the", top_k=2)
        assert len(results) <= 2

    def test_top_k_one(self):
        results = search_sentences(MULTI_SENTENCE, "cat", top_k=1)
        assert len(results) == 1

    def test_no_matches(self):
        results = search_sentences(MULTI_SENTENCE, "xyznonexistent")
        assert len(results) == 0

    def test_empty_document(self):
        results = search_sentences("", "cat")
        assert len(results) == 0

    def test_empty_query(self):
        results = search_sentences(MULTI_SENTENCE, "")
        assert len(results) == 0

    def test_sorted_by_score(self):
        results = search_sentences(MULTI_SENTENCE, "cat dog", top_k=10)
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_with_tokenizer(self):
        tok = PunktTokenizer()
        results = search_sentences(MULTI_SENTENCE, "cat mat", tokenizer=tok)
        assert len(results) >= 1
        assert results[0].text == "The cat sat on the mat."

    def test_lowercase_true(self):
        results = search_sentences(MULTI_SENTENCE, "CAT MAT", lowercase=True)
        assert len(results) >= 1

    def test_lowercase_false(self):
        # With lowercase=False, "CAT" won't match "cat"
        results = search_sentences(MULTI_SENTENCE, "CAT", lowercase=False)
        assert len(results) == 0


# ─── search_paragraphs tests ────────────────────────────────────────────────


class TestSearchParagraphs:
    def test_basic_match(self):
        results = search_paragraphs(MULTI_PARAGRAPH, "cat mat")
        assert len(results) >= 1
        assert "cat" in results[0].text

    def test_returns_full_paragraph(self):
        results = search_paragraphs(MULTI_PARAGRAPH, "cat")
        assert len(results) >= 1
        # First paragraph should contain both sentences
        assert "lazy day" in results[0].text

    def test_round_trip_offsets(self):
        results = search_paragraphs(MULTI_PARAGRAPH, "dog park")
        for r in results:
            assert MULTI_PARAGRAPH[r.start : r.end] == r.text

    def test_round_trip_offsets_latin(self):
        latin_para = "Le café est bon. C'est délicieux.\n\nLe thé est chaud. C'est agréable."
        results = search_paragraphs(latin_para, "café")
        for r in results:
            assert latin_para[r.start : r.end] == r.text

    def test_top_k_limit(self):
        results = search_paragraphs(MULTI_PARAGRAPH, "the", top_k=2)
        assert len(results) <= 2

    def test_no_matches(self):
        results = search_paragraphs(MULTI_PARAGRAPH, "xyznonexistent")
        assert len(results) == 0

    def test_empty_document(self):
        results = search_paragraphs("", "cat")
        assert len(results) == 0

    def test_empty_query(self):
        results = search_paragraphs(MULTI_PARAGRAPH, "")
        assert len(results) == 0

    def test_sorted_by_score(self):
        results = search_paragraphs(MULTI_PARAGRAPH, "cat dog bird", top_k=10)
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_with_tokenizer(self):
        tok = PunktTokenizer()
        results = search_paragraphs(MULTI_PARAGRAPH, "bird song", tokenizer=tok)
        assert len(results) >= 1

    def test_single_paragraph(self):
        text = "This is one paragraph with no line breaks at all."
        results = search_paragraphs(text, "paragraph")
        assert len(results) >= 1
        assert results[0].text == text


# ─── Edge cases ──────────────────────────────────────────────────────────────


class TestSearchEdgeCases:
    def test_section_symbol_offsets(self):
        """§ is 2 bytes in UTF-8."""
        text = "See § 101 for details. Check § 202 for updates. Read § 303 for notes."
        results = search_sentences(text, "details")
        for r in results:
            assert text[r.start : r.end] == r.text

    def test_mixed_multibyte(self):
        """Mix of 1, 2, 3, and 4-byte characters."""
        text = "The café in 東京 serves 😀 food. The pub in London serves ale."
        results = search_sentences(text, "café 東京")
        for r in results:
            assert text[r.start : r.end] == r.text

    def test_very_short_document(self):
        results = search_sentences("Hi.", "hi")
        assert len(results) <= 1

    def test_whitespace_only_query(self):
        results = search_sentences(MULTI_SENTENCE, "   ")
        assert len(results) == 0
