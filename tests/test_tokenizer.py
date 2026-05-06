"""Tests for the tokenizer module.

Validates all features from the kelvin-nlp tokenizer:
- Whitespace scan and regex-based tokenization
- Span positions (start, end)
- Prefix truncation for approximate stemming
- Stopword filtering
- Keep punctuation option
- Unicode whitespace (22 characters including NBSP, CJK space, ZWSP)
- Unicode-aware punctuation stripping with ASCII fast path
- Custom regex patterns
- Pickle support
"""

import pickle

from kaos_nlp_core.tokenizer import (
    Tokenizer,
    tokenize,
    tokenize_batch,
    tokenize_words,
    tokenize_words_batch,
)

# ── Convenience functions ────────────────────────────────────────────────────


class TestTokenizeFunction:
    """Tests for the stateless tokenize() convenience function."""

    def test_basic(self):
        tokens = tokenize("Hello, world!")
        assert len(tokens) == 2
        assert tokens[0].text == "Hello"
        assert tokens[1].text == "world"

    def test_spans(self):
        tokens = tokenize("Hello, world!")
        assert tokens[0].start == 0
        assert tokens[0].end == 6  # includes comma in span
        assert tokens[1].start == 7

    def test_lowercase(self):
        tokens = tokenize("Hello World", lowercase=True)
        assert tokens[0].text == "hello"

    def test_prefix(self):
        tokens = tokenize("automobile transportation", prefix=4)
        assert tokens[0].text == "auto"
        assert tokens[1].text == "tran"


class TestTokenizeWords:
    """Tests for the stateless tokenize_words() convenience function."""

    def test_basic(self):
        words = tokenize_words("Hello, world!", lowercase=True)
        assert words == ["hello", "world"]

    def test_prefix(self):
        words = tokenize_words("automobile transportation", prefix=4, lowercase=True)
        assert words == ["auto", "tran"]

    def test_batch(self):
        words = tokenize_words_batch(["Hello, world!", "Second line"], lowercase=True)
        assert words == [["hello", "world"], ["second", "line"]]


# ── Tokenizer class ─────────────────────────────────────────────────────────


class TestTokenizerBasic:
    def test_default(self):
        tok = Tokenizer()
        tokens = tok.tokenize("Hello, world!")
        assert len(tokens) == 2
        assert tokens[0].text == "Hello"
        assert tokens[1].text == "world"

    def test_words_only(self):
        tok = Tokenizer(lowercase=True)
        words = tok.tokenize_words("Hello, World!")
        assert words == ["hello", "world"]

    def test_batch_methods(self):
        tok = Tokenizer(lowercase=True)
        token_batches = tok.tokenize_batch(["Hello, World!", "Second line"])
        word_batches = tok.tokenize_words_batch(["Hello, World!", "Second line"])
        assert len(token_batches) == 2
        assert token_batches[0][0].text == "hello"
        assert word_batches[1] == ["second", "line"]

    def test_regex_batch_methods(self):
        tok = Tokenizer(lowercase=True)
        token_batches = tok.tokenize_regex_batch(["A-1", "B-2"], pattern=r"[A-Z]-\d")
        word_batches = tok.tokenize_regex_words_batch(["A-1", "B-2"], pattern=r"[A-Z]-\d")
        assert len(token_batches) == 2
        assert word_batches == [["a-1"], ["b-2"]]

    def test_top_level_batch(self):
        token_batches = tokenize_batch(["Hello, world!", "Second line"], lowercase=True)
        assert token_batches[0][0].text == "hello"


class TestTokenizerSpans:
    """Span positions must be byte offsets into the original text."""

    def test_ascii_spans(self):
        text = "Hello, world!"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        # First token span covers "Hello,"
        assert text[tokens[0].start : tokens[0].end] == "Hello,"
        # Second token span covers "world!"
        assert text[tokens[1].start : tokens[1].end] == "world!"

    def test_unicode_spans(self):
        text = "café résumé"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        assert text[tokens[0].start : tokens[0].end] == "café"
        assert text[tokens[1].start : tokens[1].end] == "résumé"

    def test_span_with_punctuation_stripped(self):
        """Span covers original text including punctuation, text is cleaned."""
        text = "(hello) [world]"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        assert tokens[0].text == "hello"
        assert text[tokens[0].start : tokens[0].end] == "(hello)"
        assert tokens[1].text == "world"
        assert text[tokens[1].start : tokens[1].end] == "[world]"

    def test_cjk_spans(self):
        """CJK chars (3 bytes each) — char offsets must not be byte offsets."""
        text = "東京 大阪 名古屋"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        assert text[tokens[0].start : tokens[0].end] == "東京"
        assert text[tokens[1].start : tokens[1].end] == "大阪"
        assert text[tokens[2].start : tokens[2].end] == "名古屋"

    def test_emoji_spans(self):
        """Emoji (4 bytes each) — char offsets must not be byte offsets."""
        text = "hello 😀 world"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        assert text[tokens[0].start : tokens[0].end] == "hello"
        assert text[tokens[1].start : tokens[1].end] == "😀"
        assert text[tokens[2].start : tokens[2].end] == "world"

    def test_section_symbol_span(self):
        """§ is 2 bytes — verify char offset conversion."""
        text = "See § 1983"
        tok = Tokenizer()
        tokens = tok.tokenize(text)
        for t in tokens:
            assert text[t.start : t.end] == t.text or t.text in text[t.start : t.end]


class TestPrefixTruncation:
    """Prefix truncation for approximate stemming."""

    def test_prefix_4(self):
        tok = Tokenizer(prefix=4, lowercase=True)
        words = tok.tokenize_words("automobile automobiles automotive")
        assert words == ["auto", "auto", "auto"]

    def test_prefix_3(self):
        tok = Tokenizer(prefix=3, lowercase=True)
        words = tok.tokenize_words("contract contracts contracted contracting")
        assert words == ["con", "con", "con", "con"]

    def test_short_words_preserved(self):
        tok = Tokenizer(prefix=4)
        words = tok.tokenize_words("the cat sat")
        assert words == ["the", "cat", "sat"]

    def test_prefix_unicode_char_boundary(self):
        """Prefix must truncate at character boundaries, not byte boundaries."""
        tok = Tokenizer(prefix=4)
        words = tok.tokenize_words("café résumé")
        assert words[0] == "café"  # 4 chars (5 bytes) — correct
        assert words[1] == "résu"  # 4 chars


class TestStopwordFiltering:
    def test_stopwords(self):
        tok = Tokenizer(lowercase=True, stopwords=["the", "a", "an", "and", "of"])
        words = tok.tokenize_words("The cat and the dog of an owner")
        assert words == ["cat", "dog", "owner"]

    def test_stopwords_with_prefix(self):
        tok = Tokenizer(lowercase=True, prefix=4, stopwords=["the"])
        words = tok.tokenize_words("the automobile industry")
        assert "the" not in words
        assert "auto" in words

    def test_empty_stopwords(self):
        tok = Tokenizer(stopwords=[])
        words = tok.tokenize_words("the cat")
        assert words == ["the", "cat"]


class TestKeepPunctuation:
    def test_keep(self):
        tok = Tokenizer(keep_punctuation=True)
        words = tok.tokenize_words("Hello, world!")
        assert words == ["Hello,", "world!"]

    def test_strip_default(self):
        tok = Tokenizer()
        words = tok.tokenize_words("Hello, world!")
        assert words == ["Hello", "world"]

    def test_internal_punctuation_preserved(self):
        """Internal punctuation like apostrophes should always be preserved."""
        tok = Tokenizer()
        words = tok.tokenize_words("don't mother-in-law")
        assert words == ["don't", "mother-in-law"]


class TestUnicodeWhitespace:
    """All 22 Unicode whitespace characters from kelvin-nlp."""

    def test_nbsp(self):
        tok = Tokenizer()
        words = tok.tokenize_words("hello\u00a0world")
        assert words == ["hello", "world"]

    def test_cjk_space(self):
        tok = Tokenizer()
        words = tok.tokenize_words("東京\u3000大阪")
        assert words == ["東京", "大阪"]

    def test_zero_width_space(self):
        tok = Tokenizer()
        words = tok.tokenize_words("hello\u200bworld")
        assert words == ["hello", "world"]

    def test_en_space(self):
        tok = Tokenizer()
        words = tok.tokenize_words("hello\u2002world")
        assert words == ["hello", "world"]

    def test_em_space(self):
        tok = Tokenizer()
        words = tok.tokenize_words("hello\u2003world")
        assert words == ["hello", "world"]

    def test_thin_space(self):
        tok = Tokenizer()
        words = tok.tokenize_words("hello\u2009world")
        assert words == ["hello", "world"]

    def test_mixed_whitespace(self):
        tok = Tokenizer()
        words = tok.tokenize_words("a\tb\nc\u00a0d\u2003e\u3000f")
        assert words == ["a", "b", "c", "d", "e", "f"]


class TestPunctuationStripping:
    def test_outer_stripped(self):
        tok = Tokenizer()
        words = tok.tokenize_words("'hello' (world) --test-- «café»")
        assert words == ["hello", "world", "test", "café"]

    def test_pure_punctuation_removed(self):
        tok = Tokenizer()
        words = tok.tokenize_words("--- ... !!! ???")
        assert words == []

    def test_unicode_quotes(self):
        tok = Tokenizer()
        words = tok.tokenize_words("\u201chello\u201d")
        assert words == ["hello"]


class TestRegexTokenization:
    def test_regex_basic(self):
        tok = Tokenizer(lowercase=True)
        tokens = tok.tokenize_regex("Hello, world!")
        assert len(tokens) == 2
        assert tokens[0].text == "hello"

    def test_regex_custom_pattern(self):
        tok = Tokenizer()
        tokens = tok.tokenize_regex("abc 123 def 456", pattern=r"[a-zA-Z]+")
        assert len(tokens) == 2
        assert tokens[0].text == "abc"
        assert tokens[1].text == "def"

    def test_regex_words(self):
        tok = Tokenizer(lowercase=True, prefix=4)
        words = tok.tokenize_regex_words("Automobile Transportation")
        assert words == ["auto", "tran"]


class TestControlledLexicon:
    """Controlled lexicon filter (index=) — only emit tokens in the allowed set."""

    def test_index_basic(self):
        tok = Tokenizer(lowercase=True, index=["hello", "world"])
        words = tok.tokenize_words("Hello foo World bar baz")
        assert words == ["hello", "world"]

    def test_index_with_prefix(self):
        tok = Tokenizer(lowercase=True, prefix=4, index=["auto", "tran"])
        words = tok.tokenize_words("automobile transportation other stuff")
        assert words == ["auto", "tran"]

    def test_index_empty_passes_all(self):
        tok = Tokenizer(lowercase=True, index=[])
        words = tok.tokenize_words("hello world")
        assert words == ["hello", "world"]

    def test_index_with_stopwords(self):
        """Stopwords are applied before index."""
        tok = Tokenizer(lowercase=True, stopwords=["the"], index=["cat", "dog"])
        words = tok.tokenize_words("the cat and the dog")
        assert words == ["cat", "dog"]


class TestEdgeCases:
    def test_empty_string(self):
        tok = Tokenizer()
        assert tok.tokenize_words("") == []

    def test_whitespace_only(self):
        tok = Tokenizer()
        assert tok.tokenize_words("   \t\n  ") == []

    def test_single_word(self):
        tok = Tokenizer()
        tokens = tok.tokenize("hello")
        assert len(tokens) == 1
        assert tokens[0].text == "hello"
        assert tokens[0].start == 0
        assert tokens[0].end == 5

    def test_multiple_spaces(self):
        tok = Tokenizer()
        assert tok.tokenize_words("hello    world") == ["hello", "world"]


class TestPickle:
    def test_roundtrip(self):
        tok = Tokenizer(lowercase=True, prefix=4, stopwords=["the", "a"])
        tok2 = pickle.loads(pickle.dumps(tok))
        words = tok2.tokenize_words("The automobile industry")
        assert words == ["auto", "indu"]

    def test_preserves_config(self):
        tok = Tokenizer(lowercase=True, keep_punctuation=True, prefix=3)
        tok2 = pickle.loads(pickle.dumps(tok))
        words = tok2.tokenize_words("Hello, World!")
        assert words == ["hel", "wor"]
