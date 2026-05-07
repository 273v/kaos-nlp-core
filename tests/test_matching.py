"""Tests for pattern matching: substring, multi-pattern, regex, FST."""

import pytest

from kaos_nlp_core.matching import (
    FstMap,
    FstSet,
    MultiPatternMatcher,
    RegexMatcher,
    RegexSetMatcher,
    substring_count,
    substring_count_batch,
    substring_find_all,
    substring_find_all_batch,
    substring_find_all_case_insensitive,
    substring_find_first,
)

# --- Substring ---


class TestSubstring:
    def test_find_all(self):
        matches = substring_find_all("abcabcabc", "abc")
        assert len(matches) == 3
        assert matches[0].start == 0
        assert matches[1].start == 3
        assert matches[2].start == 6

    def test_find_first(self):
        m = substring_find_first("hello world hello", "hello")
        assert m is not None
        assert m.start == 0

    def test_find_first_not_found(self):
        m = substring_find_first("hello", "xyz")
        assert m is None

    def test_count(self):
        assert substring_count("aaaa", "aa") == 2  # non-overlapping

    def test_case_insensitive(self):
        matches = substring_find_all_case_insensitive("Hello HELLO hello", "hello")
        assert len(matches) == 3

    def test_empty_needle(self):
        assert substring_find_all("abc", "") == []

    def test_no_match(self):
        assert substring_find_all("abc", "xyz") == []

    def test_find_all_batch(self):
        matches = substring_find_all_batch(["abcabc", "xyz"], "abc")
        assert len(matches) == 2
        assert len(matches[0]) == 2
        assert matches[1] == []

    def test_count_batch(self):
        counts = substring_count_batch(["aaaa", "abc"], "aa")
        assert counts == [2, 0]


# --- Multi-pattern (Aho-Corasick) ---


class TestMultiPattern:
    def test_basic(self):
        m = MultiPatternMatcher(["cat", "dog"])
        matches = m.find_all("I have a cat and a dog")
        assert len(matches) == 2
        texts = [match.text for match in matches]
        assert "cat" in texts
        assert "dog" in texts

    def test_is_match(self):
        m = MultiPatternMatcher(["hello"])
        assert m.is_match("say hello")
        assert not m.is_match("say goodbye")

    def test_count(self):
        m = MultiPatternMatcher(["the"])
        assert m.count("the cat and the dog and the bird") == 3

    def test_longest_match(self):
        m = MultiPatternMatcher(["he", "hello", "hell"], longest_match=True)
        matches = m.find_all("hello world")
        assert len(matches) == 1
        assert matches[0].text == "hello"

    def test_case_insensitive(self):
        m = MultiPatternMatcher(["hello"], case_insensitive=True)
        assert m.is_match("HELLO WORLD")

    def test_replace(self):
        m = MultiPatternMatcher(["cat", "dog"])
        result = m.replace_all("I have a cat and a dog", ["CAT", "DOG"])
        assert result == "I have a CAT and a DOG"

    def test_find_all_batch(self):
        m = MultiPatternMatcher(["cat", "dog"])
        matches = m.find_all_batch(["cat dog", "dog"])
        assert len(matches) == 2
        assert len(matches[0]) == 2
        assert len(matches[1]) == 1

    def test_empty_patterns_error(self):
        with pytest.raises(ValueError):
            MultiPatternMatcher([])


# --- Regex ---


class TestRegex:
    def test_find_all(self):
        r = RegexMatcher(r"\b\d+\b")
        matches = r.find_all("I have 3 cats and 42 dogs")
        assert len(matches) == 2
        assert matches[0].text == "3"
        assert matches[1].text == "42"

    def test_capture_groups(self):
        r = RegexMatcher(r"(\d{4})-(\d{2})-(\d{2})")
        m = r.find_first("Date: 2026-03-24")
        assert m is not None
        assert m.groups[1] == "2026"
        assert m.groups[2] == "03"
        assert m.groups[3] == "24"

    def test_is_match(self):
        r = RegexMatcher(r"\d+")
        assert r.is_match("abc123")
        assert not r.is_match("abcdef")

    def test_replace(self):
        r = RegexMatcher(r"\d+")
        assert r.replace_all("a1b2c3", "X") == "aXbXcX"

    def test_split(self):
        r = RegexMatcher(r"[,;]\s*")
        parts = r.split("a, b; c, d")
        assert parts == ["a", "b", "c", "d"]

    def test_count(self):
        r = RegexMatcher(r"\d+")
        assert r.count("a1b22c333") == 3

    def test_invalid_pattern(self):
        with pytest.raises(ValueError):
            RegexMatcher(r"[invalid")

    def test_pattern(self):
        r = RegexMatcher(r"\d+")
        assert r.pattern() == r"\d+"

    def test_find_all_batch(self):
        r = RegexMatcher(r"\d+")
        matches = r.find_all_batch(["abc123", "x9y8"])
        assert len(matches) == 2
        assert matches[0][0].text == "123"
        assert len(matches[1]) == 2


class TestRegexSet:
    def test_matching_patterns(self):
        rs = RegexSetMatcher([r"\d+", r"[a-z]+", r"[A-Z]+"])
        indices = rs.matching_patterns("Hello 42 world")
        assert 0 in indices  # digits
        assert 1 in indices  # lowercase
        assert 2 in indices  # uppercase

    def test_is_match(self):
        rs = RegexSetMatcher([r"hello", r"world"])
        assert rs.is_match("hello there")
        assert not rs.is_match("goodbye")

    def test_pattern_count(self):
        rs = RegexSetMatcher([r"a", r"b", r"c"])
        assert rs.pattern_count() == 3


# --- FST ---


class TestFstSet:
    def test_contains(self):
        s = FstSet(["apple", "banana", "cherry", "date"])
        assert s.contains("banana")
        assert not s.contains("grape")
        assert len(s) == 4

    def test_in_operator(self):
        s = FstSet(["hello", "world"])
        assert "hello" in s
        assert "xyz" not in s

    def test_fuzzy_search(self):
        s = FstSet(["apple", "apply", "ample", "maple", "orange"])
        results = s.fuzzy_search("aple", 2)
        keys = [r.key for r in results]
        assert "apple" in keys
        assert "ample" in keys

    def test_prefix_search(self):
        s = FstSet(["app", "apple", "application", "banana"])
        results = s.prefix_search("app")
        assert len(results) == 3
        assert "apple" in results

    def test_dedup(self):
        s = FstSet(["a", "b", "a", "c", "b"])
        assert len(s) == 3

    def test_empty(self):
        s = FstSet([])
        assert len(s) == 0
        assert not s.contains("anything")


class TestFstMap:
    def test_get(self):
        m = FstMap([("cat", 10), ("dog", 20), ("bird", 5)])
        assert m.get("cat") == 10
        assert m.get("dog") == 20
        assert m.get("fish") is None

    def test_contains(self):
        m = FstMap([("hello", 1)])
        assert m.contains_key("hello")
        assert "hello" in m
        assert "xyz" not in m

    def test_len(self):
        m = FstMap([("a", 1), ("b", 2), ("c", 3)])
        assert len(m) == 3

    def test_empty(self):
        m = FstMap([])
        assert len(m) == 0
        assert m.get("anything") is None


# ── Unicode span correctness (see CLAUDE.md byte/char offset rules) ─────────


class TestSubstringUnicodeSpans:
    """Verify substring matching returns char offsets, not byte offsets."""

    def test_cafe_find_all(self):
        text = "café café"
        matches = substring_find_all(text, "café")
        assert len(matches) == 2
        for m in matches:
            assert text[m.start : m.end] == "café"

    def test_cjk_find_all(self):
        text = "東京は東京"
        matches = substring_find_all(text, "東京")
        assert len(matches) == 2
        for m in matches:
            assert text[m.start : m.end] == "東京"

    def test_emoji_find_first(self):
        text = "hello 😀 world"
        m = substring_find_first(text, "world")
        assert m is not None
        assert text[m.start : m.end] == "world"
        assert m.start == 8  # char offset, not byte offset 10

    def test_section_symbol(self):
        text = "§ 1983 and § 2000"
        matches = substring_find_all(text, "§")
        assert len(matches) == 2
        for m in matches:
            assert text[m.start : m.end] == "§"

    def test_case_insensitive_unicode(self):
        text = "Café CAFÉ café"
        matches = substring_find_all_case_insensitive(text, "café")
        assert len(matches) == 3
        for m in matches:
            assert text[m.start : m.end].lower() == "café"


class TestMultiPatternUnicodeSpans:
    """Verify multi-pattern matching returns char offsets."""

    def test_mixed_patterns(self):
        mp = MultiPatternMatcher(["café", "§", "hello"])
        text = "hello at café near §"
        matches = mp.find_all(text)
        for m in matches:
            assert text[m.start : m.end] == m.text

    def test_cjk_patterns(self):
        mp = MultiPatternMatcher(["東京", "大阪"])
        text = "東京と大阪"
        matches = mp.find_all(text)
        assert len(matches) == 2
        for m in matches:
            assert text[m.start : m.end] == m.text

    def test_emoji_patterns(self):
        mp = MultiPatternMatcher(["😀", "🌍"])
        text = "Earth 🌍 is great 😀"
        matches = mp.find_all(text)
        assert len(matches) == 2
        for m in matches:
            assert text[m.start : m.end] == m.text

    def test_replace_all_wrong_count(self):
        mp = MultiPatternMatcher(["cat", "dog"])
        with pytest.raises(ValueError):
            mp.replace_all("cat and dog", ["CAT"])  # Too few replacements


class TestRegexUnicodeSpans:
    """Verify regex matching returns char offsets."""

    def test_unicode_word_match(self):
        r = RegexMatcher(r"café")
        text = "Le café est bon"
        matches = r.find_all(text)
        assert len(matches) == 1
        assert text[matches[0].start : matches[0].end] == "café"

    def test_match_after_cjk(self):
        r = RegexMatcher(r"hello")
        text = "東京 hello"
        m = r.find_first(text)
        assert m is not None
        assert text[m.start : m.end] == "hello"
        assert m.start == 3  # char offset, not byte offset 7

    def test_match_after_emoji(self):
        r = RegexMatcher(r"\w+")
        text = "😀 world"
        matches = r.find_all(text)
        # Should find "world" at correct char offset
        world_match = next(m for m in matches if m.text == "world")
        assert text[world_match.start : world_match.end] == "world"

    def test_cjk_regex(self):
        r = RegexMatcher(r"[\u4e00-\u9fff]+")
        text = "hello 東京タワー world"
        matches = r.find_all(text)
        assert len(matches) >= 1
        for m in matches:
            assert text[m.start : m.end] == m.text


class TestFstSetEdgeCases:
    def test_fuzzy_search_distance_0(self):
        s = FstSet(["apple", "banana"])
        results = s.fuzzy_search("apple", 0)
        keys = [r.key for r in results]
        assert "apple" in keys
        assert "banana" not in keys
