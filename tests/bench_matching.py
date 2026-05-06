"""Benchmarks for pattern matching: substring, multi-pattern, regex, FST (pytest-benchmark)."""

import pytest

from kaos_nlp_core.matching import (
    FstSet,
    MultiPatternMatcher,
    RegexMatcher,
    substring_count,
    substring_find_all,
    substring_find_all_case_insensitive,
    substring_find_first,
)

# --- Substring search on War and Peace ---


@pytest.mark.benchmark(group="substring")
@pytest.mark.parametrize(
    "needle",
    ["Prince", "the", "Natasha Rostova", "NONEXISTENT_STRING_XYZ"],
    ids=lambda n: n[:20],
)
def test_substring_find_all(benchmark, war_and_peace_text, needle):
    benchmark(substring_find_all, war_and_peace_text, needle)


@pytest.mark.benchmark(group="substring")
def test_substring_count_the(benchmark, war_and_peace_text):
    benchmark(substring_count, war_and_peace_text, "the")


@pytest.mark.benchmark(group="substring")
def test_substring_find_first_prince(benchmark, war_and_peace_text):
    benchmark(substring_find_first, war_and_peace_text, "Prince")


@pytest.mark.benchmark(group="substring")
def test_substring_case_insensitive_prince(benchmark, war_and_peace_text):
    benchmark(substring_find_all_case_insensitive, war_and_peace_text, "prince")


# --- Multi-pattern (Aho-Corasick) on War and Peace ---


@pytest.mark.benchmark(group="multi_pattern")
def test_multi_pattern_4(benchmark, war_and_peace_text):
    m = MultiPatternMatcher(["Prince", "Princess", "Count", "Countess"])
    benchmark(m.find_all, war_and_peace_text)


@pytest.mark.benchmark(group="multi_pattern")
def test_multi_pattern_20(benchmark, war_and_peace_text):
    m = MultiPatternMatcher(
        [
            "the",
            "and",
            "was",
            "for",
            "that",
            "with",
            "his",
            "her",
            "from",
            "they",
            "have",
            "this",
            "been",
            "would",
            "could",
            "their",
            "which",
            "about",
            "other",
            "into",
        ]
    )
    benchmark(m.find_all, war_and_peace_text)


@pytest.mark.benchmark(group="multi_pattern")
def test_multi_pattern_20_count(benchmark, war_and_peace_text):
    m = MultiPatternMatcher(
        [
            "the",
            "and",
            "was",
            "for",
            "that",
            "with",
            "his",
            "her",
            "from",
            "they",
            "have",
            "this",
            "been",
            "would",
            "could",
            "their",
            "which",
            "about",
            "other",
            "into",
        ]
    )
    benchmark(m.count, war_and_peace_text)


# --- Regex on Shakespeare ---


@pytest.mark.benchmark(group="regex")
def test_regex_word_boundary_love(benchmark, shakespeare_text):
    r = RegexMatcher(r"\b[Ll]ove\b")
    benchmark(r.find_all, shakespeare_text)


@pytest.mark.benchmark(group="regex")
def test_regex_four_digit_numbers(benchmark, shakespeare_text):
    r = RegexMatcher(r"\b\d{4}\b")
    benchmark(r.find_all, shakespeare_text)


@pytest.mark.benchmark(group="regex")
def test_regex_capitalized_words(benchmark, shakespeare_text):
    r = RegexMatcher(r"\b[A-Z][a-z]{3,}\b")
    benchmark(r.find_all, shakespeare_text)


@pytest.mark.benchmark(group="regex")
def test_regex_character_names(benchmark, shakespeare_text):
    r = RegexMatcher(r"\b(Romeo|Juliet|Hamlet|Othello|Macbeth|Lear)\b")
    benchmark(r.find_all, shakespeare_text)


# --- FST on War and Peace vocabulary ---


@pytest.fixture(scope="module")
def war_peace_vocab(war_and_peace_text):
    """Sorted unique words from War and Peace."""
    from conftest import DEFAULT_TOKENIZER

    return sorted(set(DEFAULT_TOKENIZER.tokenize_words(war_and_peace_text)))


@pytest.mark.benchmark(group="fst")
def test_fst_build(benchmark, war_peace_vocab):
    benchmark(FstSet, war_peace_vocab)


@pytest.mark.benchmark(group="fst")
def test_fst_contains(benchmark, war_peace_vocab):
    fst = FstSet(war_peace_vocab)
    queries = ["prince", "war", "peace", "love", "death", "nonexistent"]
    benchmark(lambda: [fst.contains(q) for q in queries])


@pytest.mark.benchmark(group="fst")
def test_fst_prefix_search(benchmark, war_peace_vocab):
    fst = FstSet(war_peace_vocab)
    benchmark(fst.prefix_search, "pr")


@pytest.mark.benchmark(group="fst")
def test_fst_fuzzy_d1(benchmark, war_peace_vocab):
    fst = FstSet(war_peace_vocab)
    benchmark(fst.fuzzy_search, "princ", 1)


@pytest.mark.benchmark(group="fst")
def test_fst_fuzzy_d2(benchmark, war_peace_vocab):
    fst = FstSet(war_peace_vocab)
    benchmark(fst.fuzzy_search, "princ", 2)
