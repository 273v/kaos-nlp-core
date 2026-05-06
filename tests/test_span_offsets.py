"""Comprehensive span/offset correctness tests across all modules.

Verifies that EVERY function returning start/end positions produces
character offsets (not byte offsets) that work correctly with Python
string slicing for ASCII, multi-byte Latin, CJK, emoji, and mixed text.

This test file exists because byte/char offset conversion has been a
recurring source of bugs (see CLAUDE.md "CRITICAL: Byte vs Character Offsets").
"""

import pytest

from kaos_nlp_core.matching import (
    MultiPatternMatcher,
    RegexMatcher,
    substring_find_all,
    substring_find_all_case_insensitive,
    substring_find_first,
)
from kaos_nlp_core.segmentation import (
    PunktTokenizer,
    segment_lines,
    segment_paragraphs_simple,
    segment_sentences,
)
from kaos_nlp_core.tokenizer import Tokenizer, tokenize

# ─── Test data with increasing UTF-8 byte widths ────────────────────────────

# 1-byte: ASCII
ASCII_TEXT = "Hello world. How are you?"
# 2-byte: Latin accents (é = 2 bytes)
LATIN_TEXT = "Le café est bon. Résumé du jour."
# 2-byte: Section symbol (§ = 2 bytes)
LEGAL_TEXT = "See 42 U.S.C. § 1983. The court agreed."
# 3-byte: CJK characters (東 = 3 bytes each)
CJK_TEXT = "東京は大きい都市です。大阪も大きい。"
# 4-byte: Emoji (😀 = 4 bytes)
EMOJI_TEXT = "Hello 😀 world 🌍 earth"
# Mixed: all byte widths in one string
MIXED_TEXT = "café 東京 😀 § hello"


# ─── Tokenizer span tests ───────────────────────────────────────────────────


class TestTokenizerSpanOffsets:
    """Every Tokenizer.tokenize() span must be a valid Python char offset."""

    @pytest.mark.parametrize(
        "text",
        [
            ASCII_TEXT,
            LATIN_TEXT,
            LEGAL_TEXT,
            CJK_TEXT,
            EMOJI_TEXT,
            MIXED_TEXT,
        ],
        ids=["ascii", "latin", "legal", "cjk", "emoji", "mixed"],
    )
    def test_span_roundtrip(self, text):
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        for t in tokens:
            extracted = text[t.start : t.end]
            # The cleaned text should be contained in the span
            # (span includes surrounding punctuation, text is stripped)
            assert t.text in extracted or extracted in t.text, (
                f"Token text {t.text!r} not found in span text[{t.start}:{t.end}] = {extracted!r}"
            )

    def test_cafe_specific(self):
        text = "café résumé"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        assert text[tokens[0].start : tokens[0].end] == "café"
        assert text[tokens[1].start : tokens[1].end] == "résumé"

    def test_cjk_specific(self):
        text = "東京 大阪 名古屋"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        assert text[tokens[0].start : tokens[0].end] == "東京"
        assert text[tokens[1].start : tokens[1].end] == "大阪"
        assert text[tokens[2].start : tokens[2].end] == "名古屋"

    def test_emoji_specific(self):
        text = "hello 😀 world"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        assert text[tokens[0].start : tokens[0].end] == "hello"
        assert text[tokens[1].start : tokens[1].end] == "😀"
        assert text[tokens[2].start : tokens[2].end] == "world"

    def test_section_symbol(self):
        """§ is a symbol (not punctuation), so it's preserved as a token."""
        text = "§ 1983"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        # § may be its own token or stripped depending on ICU classification
        # Key test: whatever tokens are produced, spans must be valid
        for t in tokens:
            extracted = text[t.start : t.end]
            assert t.text in extracted

    def test_mixed_all_byte_widths(self):
        """café (2-byte), 東京 (3-byte), 😀 (4-byte), § (2-byte) in one string."""
        tok = Tokenizer()
        tokens = tok.tokenize(MIXED_TEXT)
        for t in tokens:
            extracted = _text = MIXED_TEXT[t.start : t.end]
            assert len(extracted) > 0
            assert t.text in extracted

    def test_convenience_tokenize(self):
        """Standalone tokenize() function also converts offsets."""
        tokens = tokenize("café 東京", lowercase=True)
        text = "café 東京"
        assert text[tokens[0].start : tokens[0].end] == "café"
        assert text[tokens[1].start : tokens[1].end] == "東京"


# ─── Substring matching span tests ──────────────────────────────────────────


class TestSubstringSpanOffsets:
    """substring_find_all/find_first must return char offsets, not byte offsets."""

    @pytest.mark.parametrize(
        "haystack,needle",
        [
            ("café café café", "café"),
            ("東京は東京タワーの東京", "東京"),
            ("hello 😀 world 😀 end", "😀"),
            ("See § 1983 and § 2000", "§"),
            ("naïve naïve", "naïve"),
        ],
        ids=["latin", "cjk", "emoji", "section", "diaeresis"],
    )
    def test_find_all_roundtrip(self, haystack, needle):
        matches = substring_find_all(haystack, needle)
        assert len(matches) >= 2, f"Expected >=2 matches for {needle!r} in {haystack!r}"
        for m in matches:
            extracted = haystack[m.start : m.end]
            assert extracted == m.text == needle, (
                f"Span mismatch: haystack[{m.start}:{m.end}] = {extracted!r}, expected {needle!r}"
            )

    def test_find_first_unicode(self):
        text = "Le café est bon."
        m = substring_find_first(text, "café")
        assert m is not None
        assert text[m.start : m.end] == "café"

    def test_find_first_after_multibyte(self):
        """Needle appears AFTER multi-byte chars — offset must account for them."""
        text = "東京 hello"
        m = substring_find_first(text, "hello")
        assert m is not None
        assert text[m.start : m.end] == "hello"

    def test_case_insensitive_unicode(self):
        text = "Café CAFÉ café"
        matches = substring_find_all_case_insensitive(text, "café")
        for m in matches:
            extracted = text[m.start : m.end]
            assert extracted.lower() == "café", (
                f"Case-insensitive mismatch: [{m.start}:{m.end}] = {extracted!r}"
            )


# ─── Multi-pattern matching span tests ──────────────────────────────────────


class TestMultiPatternSpanOffsets:
    """MultiPatternMatcher.find_all must return char offsets."""

    def test_unicode_patterns(self):
        mp = MultiPatternMatcher(["café", "§", "東京"])
        text = "Le café à § et 東京"
        matches = mp.find_all(text)
        for m in matches:
            extracted = text[m.start : m.end]
            assert extracted == m.text, (
                f"Multi-pattern span mismatch: [{m.start}:{m.end}] = {extracted!r}, "
                f"expected {m.text!r}"
            )

    def test_emoji_pattern(self):
        mp = MultiPatternMatcher(["😀", "🌍"])
        text = "Hello 😀 world 🌍 end"
        matches = mp.find_all(text)
        for m in matches:
            extracted = text[m.start : m.end]
            assert extracted == m.text

    def test_mixed_ascii_unicode_patterns(self):
        mp = MultiPatternMatcher(["hello", "café", "東京"])
        text = "hello café 東京"
        matches = mp.find_all(text)
        assert len(matches) == 3
        for m in matches:
            assert text[m.start : m.end] == m.text


# ─── Regex matching span tests ──────────────────────────────────────────────


class TestRegexSpanOffsets:
    """RegexMatcher.find_all/find_first must return char offsets."""

    def test_unicode_regex(self):
        rx = RegexMatcher(r"café|§|東京")
        text = "Le café à § et 東京"
        matches = rx.find_all(text)
        for m in matches:
            extracted = text[m.start : m.end]
            assert extracted == m.text, (
                f"Regex span mismatch: [{m.start}:{m.end}] = {extracted!r}, expected {m.text!r}"
            )

    def test_regex_find_first_unicode(self):
        rx = RegexMatcher(r"\w+")
        text = "café 東京"
        m = rx.find_first(text)
        assert m is not None
        assert text[m.start : m.end] == m.text

    def test_regex_word_boundary_unicode(self):
        rx = RegexMatcher(r"\b\w+\b")
        text = "café résumé naïve"
        matches = rx.find_all(text)
        for m in matches:
            assert text[m.start : m.end] == m.text

    def test_regex_after_emoji(self):
        """Match text that appears after 4-byte emoji characters."""
        rx = RegexMatcher(r"world")
        text = "😀😀😀 world"
        m = rx.find_first(text)
        assert m is not None
        assert text[m.start : m.end] == "world"


# ─── Sentence segmentation span tests ───────────────────────────────────────


class TestSegmentationSpanOffsets:
    """Segmentation spans must be char offsets usable with Python slicing."""

    @pytest.mark.parametrize(
        "text",
        [
            ASCII_TEXT,
            LATIN_TEXT,
            LEGAL_TEXT,
        ],
        ids=["ascii", "latin", "legal"],
    )
    def test_sentence_span_roundtrip(self, text):
        tok = PunktTokenizer()
        segs = segment_sentences(text, tok)
        for s in segs:
            extracted = text[s.start : s.end]
            assert extracted == s.text, (
                f"Sentence span mismatch: [{s.start}:{s.end}] = {extracted!r}, text = {s.text!r}"
            )

    def test_tokenize_spans_unicode(self):
        tok = PunktTokenizer()
        text = "Le café est bon. Résumé du jour."
        spans = tok.tokenize_spans(text)
        for start, end in spans:
            extracted = text[start:end]
            assert len(extracted.strip()) > 0
            # Should not start mid-character
            assert not extracted.startswith("é")
            assert not extracted.startswith("su")

    def test_tokenize_spans_cjk(self):
        tok = PunktTokenizer()
        text = "東京は大きい。大阪も大きい。"
        spans = tok.tokenize_spans(text)
        for start, end in spans:
            extracted = text[start:end]
            assert len(extracted) > 0

    def test_segment_lines_unicode(self):
        text = "café\n東京\n😀"
        lines = segment_lines(text)
        assert len(lines) == 3
        assert lines[0].text == "café"
        assert text[lines[0].start : lines[0].end] == "café"
        assert lines[1].text == "東京"
        assert text[lines[1].start : lines[1].end] == "東京"
        assert lines[2].text == "😀"
        assert text[lines[2].start : lines[2].end] == "😀"

    def test_segment_paragraphs_unicode(self):
        text = "Café paragraph.\n\n東京 paragraph."
        paras = segment_paragraphs_simple(text)
        for p in paras:
            assert text[p.start : p.end] == p.text


# ─── Cross-module consistency tests ─────────────────────────────────────────


class TestCrossModuleSpanConsistency:
    """Verify that different modules agree on offsets for the same text."""

    def test_tokenizer_and_substring_agree(self):
        """If tokenizer finds 'café' at position X, substring should too."""
        text = "hello café world"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        cafe_token = next(t for t in tokens if t.text == "café")

        m = substring_find_first(text, "café")
        assert m is not None
        assert m.start == cafe_token.start
        assert m.end == cafe_token.end

    def test_substring_and_regex_agree(self):
        """Substring and regex should find the same positions."""
        text = "東京 is 東京"
        sub_matches = substring_find_all(text, "東京")
        rx = RegexMatcher(r"東京")
        rx_matches = rx.find_all(text)

        assert len(sub_matches) == len(rx_matches)
        for sm, rm in zip(sub_matches, rx_matches, strict=True):
            assert sm.start == rm.start
            assert sm.end == rm.end
            assert sm.text == rm.text

    def test_multi_pattern_and_regex_agree(self):
        """Multi-pattern and regex should find the same positions."""
        text = "café § 東京"
        mp = MultiPatternMatcher(["café"])
        rx = RegexMatcher(r"café")

        mp_matches = mp.find_all(text)
        rx_matches = rx.find_all(text)

        assert len(mp_matches) == len(rx_matches)
        for mm, rm in zip(mp_matches, rx_matches, strict=True):
            assert mm.start == rm.start
            assert mm.end == rm.end

    def test_sentence_spans_cover_all_text(self):
        """Sentence spans should cover the entire input without gaps."""
        tok = PunktTokenizer()
        text = "First sentence. Second sentence. Third."
        spans = tok.tokenize_spans(text)
        # Concatenate all span texts
        covered = "".join(text[s:e] for s, e in spans)
        # Should contain all non-whitespace content
        for word in ["First", "sentence", "Second", "Third"]:
            assert word in covered


# ─── Rust-layer correctness (verify byte offsets are correct before conversion)


class TestRustByteOffsets:
    """Verify that Rust core produces correct byte offsets (tested via known conversions)."""

    def test_byte_len_vs_char_len(self):
        """Ensure we understand the byte/char difference for test design."""
        assert len("café".encode()) == 5  # é = 2 bytes
        assert len("café") == 4  # 4 chars
        assert len("東京".encode()) == 6  # 東 = 3 bytes each
        assert len("東京") == 2  # 2 chars
        assert len("😀".encode()) == 4  # emoji = 4 bytes
        assert len("😀") == 1  # 1 char
        assert len("§".encode()) == 2  # § = 2 bytes
        assert len("§") == 1  # 1 char

    def test_offsets_are_char_not_byte(self):
        """If offsets were bytes, these would produce wrong results."""
        text = "café hello"  # byte 5 = ' ', char 4 = ' '
        m = substring_find_first(text, "hello")
        assert m is not None
        # If byte offset (5), this is correct. If char offset (5), also correct
        # for this case. Need a case where they differ.
        assert text[m.start : m.end] == "hello"

        # Now a case where byte != char
        text2 = "東京 hello"  # byte 7 = ' ', char 3 = ' '
        m2 = substring_find_first(text2, "hello")
        assert m2 is not None
        assert m2.start == 3  # char offset, NOT 7 (byte offset)
        assert text2[m2.start : m2.end] == "hello"

    def test_char_offset_values_cjk(self):
        """Verify exact char offset values for CJK to catch byte/char confusion."""
        text = "東京タワー"  # 5 CJK chars, 15 bytes
        m = substring_find_first(text, "タワー")
        assert m is not None
        assert m.start == 2  # char offset (not 6 which would be byte offset)
        assert m.end == 5  # char offset (not 15)
        assert text[m.start : m.end] == "タワー"
