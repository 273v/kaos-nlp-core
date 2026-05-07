"""Edge case tests for boundary conditions, Unicode, and API completeness.

Covers gaps identified in coverage audit:
- InvertedIndex.add() and .idf() direct tests
- Unicode beyond BMP (emoji, CJK)
- Zero-width characters
- Single-character strings
- Whitespace-only strings
- Empty/degenerate inputs across all APIs
"""

import math

from kaos_nlp_core.algorithms import (
    jaro_winkler,
    lcs_length,
    levenshtein,
    ngram_jaccard,
    soundex_encode,
    token_jaccard,
)
from kaos_nlp_core.matching import (
    FstSet,
    MultiPatternMatcher,
    substring_find_all,
)
from kaos_nlp_core.structures import InvertedIndex

# ── InvertedIndex.add() and .idf() direct tests ─────────────────────────────


class TestInvertedIndexDirect:
    """Direct tests for InvertedIndex.add() and .idf()."""

    def test_add_single_term(self):
        idx = InvertedIndex()
        idx.add("hello", 0)
        idx.add("hello", 0)  # same term, same doc — should increment tf
        idx.add("hello", 1)  # same term, different doc
        assert idx.doc_freq("hello") == 2
        postings = idx.get_postings("hello")
        assert postings is not None
        doc0 = next(p for p in postings if p.doc_id == 0)
        assert doc0.term_freq == 2

    def test_idf_direct(self):
        idx = InvertedIndex()
        idx.add_document(0, ["the", "cat", "sat"])
        idx.add_document(1, ["the", "dog", "sat"])
        idx.add_document(2, ["a", "bird", "flew"])

        # "the" appears in 2/3 docs: idf = ln(3/2)
        idf_the = idx.idf("the")
        assert abs(idf_the - math.log(3.0 / 2.0)) < 1e-10

        # "bird" appears in 1/3 docs: idf = ln(3/1)
        idf_bird = idx.idf("bird")
        assert abs(idf_bird - math.log(3.0)) < 1e-10

        # nonexistent term: idf = 0
        assert idx.idf("xyz") == 0.0

    def test_add_empty_doc(self):
        idx = InvertedIndex()
        idx.add_document(0, [])
        assert idx.doc_count() == 0  # no terms indexed, no docs tracked
        assert idx.term_count() == 0


# ── Unicode beyond BMP ───────────────────────────────────────────────────────


class TestUnicodeBeyondBMP:
    """Test with emoji, CJK characters, and other non-Latin scripts."""

    def test_levenshtein_emoji(self):
        r = levenshtein("hello 👋", "hello 🌍")
        assert r.distance == 1.0

    def test_levenshtein_cjk(self):
        r = levenshtein("你好世界", "你好世间")
        assert r.distance == 1.0

    def test_jaro_winkler_cjk(self):
        r = jaro_winkler("东京都", "东京市")
        assert r.similarity > 0.5  # shared prefix 东京

    def test_ngram_jaccard_emoji(self):
        r = ngram_jaccard("I ❤️ cats", "I ❤️ dogs", n=2)
        assert r.similarity > 0.0  # shared "I " bigram

    def test_ngram_jaccard_cjk(self):
        r = ngram_jaccard("机器学习", "深度学习", n=2)
        assert r.similarity > 0.0  # shared "学习" bigram

    def test_token_jaccard_mixed_scripts(self):
        r = token_jaccard("hello 世界", "hello world", lowercase=True)
        assert r.similarity > 0.0  # shared "hello"

    def test_substring_emoji(self):
        matches = substring_find_all("I ❤️ NLP and I ❤️ Rust", "❤️")
        assert len(matches) == 2

    def test_substring_cjk(self):
        matches = substring_find_all("机器学习和深度学习", "学习")
        assert len(matches) == 2

    def test_fst_cjk_vocabulary(self):
        fst = FstSet(["你好", "世界", "东京", "机器学习"])
        assert fst.contains("东京")
        assert not fst.contains("北京")
        assert len(fst) == 4

    def test_multi_pattern_emoji(self):
        m = MultiPatternMatcher(["❤️", "🌍", "🎉"])
        matches = m.find_all("I ❤️ the 🌍 and celebrate with 🎉")
        assert len(matches) == 3

    def test_inverted_index_cjk(self):
        idx = InvertedIndex()
        idx.add_document(0, ["机器", "学习", "算法"])
        idx.add_document(1, ["深度", "学习", "模型"])
        assert idx.doc_freq("学习") == 2
        results = idx.query_bm25(["学习"], top_k=2)
        assert len(results) == 2


# ── Zero-width characters ───────────────────────────────────────────────────


class TestZeroWidthCharacters:
    """Test handling of zero-width joiners, ZWNBSP, etc."""

    def test_levenshtein_zwsp(self):
        """Zero-width space should count as a character."""
        r = levenshtein("hello", "hel\u200blo")  # zero-width space
        assert r.distance == 1.0

    def test_substring_with_bom(self):
        """BOM (U+FEFF) shouldn't break substring search."""
        text = "\ufeffhello world"
        matches = substring_find_all(text, "hello")
        assert len(matches) == 1

    def test_ngram_with_combining_chars(self):
        """Combining diacriticals: é (e + combining acute) vs é (precomposed)."""
        # These are different Unicode representations of the same visual character
        r = ngram_jaccard("caf\u0065\u0301", "café", n=2)
        # They may or may not match depending on normalization — just ensure no crash
        assert 0.0 <= r.similarity <= 1.0


# ── Single-character strings ────────────────────────────────────────────────


class TestSingleCharStrings:
    """Algorithms should handle single-character inputs gracefully."""

    def test_levenshtein_single_chars(self):
        r = levenshtein("a", "b")
        assert r.distance == 1.0
        r = levenshtein("a", "a")
        assert r.distance == 0.0

    def test_jaro_single_char(self):
        r = jaro_winkler("a", "a")
        assert r.similarity == 1.0

    def test_ngram_single_char(self):
        """N-gram with n > string length should return empty n-gram set."""
        r = ngram_jaccard("a", "b", n=2)
        # Both produce empty n-gram sets → similarity = 1.0 (both empty)
        assert r.similarity == 1.0

    def test_soundex_single_char(self):
        code = soundex_encode("A")
        assert isinstance(code, str) and len(code) > 0

    def test_substring_single_char(self):
        matches = substring_find_all("abcabc", "a")
        assert len(matches) == 2


# ── Whitespace-only strings ─────────────────────────────────────────────────


class TestWhitespaceStrings:
    """Algorithms should handle whitespace-only inputs."""

    def test_levenshtein_whitespace(self):
        r = levenshtein("   ", "  ")
        assert r.distance == 1.0

    def test_token_jaccard_whitespace_only(self):
        """Whitespace-only strings produce no tokens."""
        r = token_jaccard("   ", "   ")
        assert r.similarity == 1.0  # both empty → identical

    def test_token_jaccard_whitespace_vs_text(self):
        r = token_jaccard("   ", "hello")
        # One empty, one non-empty
        assert 0.0 <= r.similarity <= 1.0

    def test_ngram_whitespace(self):
        r = ngram_jaccard("  ", "  ", n=2)
        assert 0.0 <= r.similarity <= 1.0

    def test_substring_whitespace(self):
        matches = substring_find_all("hello world", " ")
        assert len(matches) == 1


# ── Degenerate / stress inputs ──────────────────────────────────────────────


class TestDegenerateInputs:
    """Stress tests with repetitive, very long, or pathological inputs."""

    def test_levenshtein_repeated_chars(self):
        """Repeated character strings."""
        a = "a" * 100
        b = "b" * 100
        r = levenshtein(a, b)
        assert r.distance == 100.0

    def test_lcs_length_long_strings(self):
        a = "abcde" * 200  # 1000 chars
        b = "abcxe" * 200
        result = lcs_length(a, b)
        assert result > 0

    def test_substring_many_matches(self):
        """String with thousands of overlapping matches."""
        text = "aa" * 5000  # 10000 chars
        matches = substring_find_all(text, "aa")
        assert len(matches) >= 5000  # overlapping matches

    def test_ngram_identical_long(self):
        text = "abcdef " * 100
        r = ngram_jaccard(text, text, n=3)
        assert r.similarity == 1.0

    def test_inverted_index_many_docs(self):
        """Index 10K single-term documents."""
        idx = InvertedIndex()
        for i in range(10000):
            idx.add_document(i, [f"term_{i % 100}", "common"])
        assert idx.doc_count() == 10000
        assert idx.doc_freq("common") == 10000
        # term_42 appears in 100 docs (42, 142, 242, ...)
        assert idx.doc_freq("term_42") == 100
        results = idx.query_bm25(["term_42"], top_k=5)
        assert len(results) == 5
        assert all(r.score > 0 for r in results)

    def test_fst_large_vocabulary(self):
        """FST with 50K terms."""
        terms = sorted(f"term_{i:06d}" for i in range(50000))
        fst = FstSet(terms)
        assert len(fst) == 50000
        assert fst.contains("term_025000")
        assert not fst.contains("term_050001")
