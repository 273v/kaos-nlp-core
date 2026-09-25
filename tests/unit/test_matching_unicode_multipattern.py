"""Unicode case-insensitive and word-boundary multi-pattern matching.

Offsets on ``PatternMatchSpan`` are character offsets into the original
haystack; every test checks ``haystack[start:end] == text``.
"""

from __future__ import annotations

import pickle

import pytest

from kaos_nlp_core.matching import MultiPatternMatcher, PatternMatchSpan


def _spans(haystack: str, matches: list[PatternMatchSpan]) -> list[tuple[str, int]]:
    for m in matches:
        assert haystack[m.start : m.end] == m.text
    return [(m.text, m.pattern_index) for m in matches]


# ─── Case-insensitive: Unicode full case folding ───────────────────────────


def test_case_insensitive_matches_non_ascii_uppercase() -> None:
    m = MultiPatternMatcher(["são paulo"], case_insensitive=True)
    text = "SÃO PAULO, São Paulo e são paulo"
    assert _spans(text, m.find_all(text)) == [
        ("SÃO PAULO", 0),
        ("São Paulo", 0),
        ("são paulo", 0),
    ]
    assert m.count(text) == 3
    assert m.is_match("SÃO PAULO")


def test_case_insensitive_uppercase_pattern_lowercase_text() -> None:
    m = MultiPatternMatcher(["ÉCOLE"], case_insensitive=True)
    assert m.is_match("une école")


def test_case_insensitive_sharp_s_whole_fold_only() -> None:
    text = "Hauptstraße / HAUPTSTRASSE"
    assert MultiPatternMatcher(["hauptstrasse"], case_insensitive=True).count(text) == 2
    assert MultiPatternMatcher(["STRAßE"], case_insensitive=True).count(text) == 2
    # "stras" would need half of the "ss" that ß folds to: only the ASCII
    # spelling matches.
    m = MultiPatternMatcher(["stras"], case_insensitive=True)
    assert _spans(text, m.find_all(text)) == [("STRAS", 0)]


def test_case_insensitive_dotted_capital_i() -> None:
    # İ folds to "i" + U+0307 (default, non-Turkic folding).
    assert MultiPatternMatcher(["İstanbul"], case_insensitive=True).is_match("İSTANBUL")
    assert MultiPatternMatcher(["i̇stanbul"], case_insensitive=True).is_match("İstanbul")
    assert not MultiPatternMatcher(["istanbul"], case_insensitive=True).is_match("İstanbul")


def test_case_insensitive_greek_sigma_forms() -> None:
    m = MultiPatternMatcher(["οδυσσευς"], case_insensitive=True)
    assert m.is_match("ΟΔΥΣΣΕΥΣ")


def test_case_insensitive_offsets_are_char_offsets_with_astral_chars() -> None:
    text = "😀😀 STRAßE 東京 straße"
    m = MultiPatternMatcher(["strasse", "東京"], case_insensitive=True)
    matches = m.find_all(text)
    assert _spans(text, matches) == [("STRAßE", 0), ("東京", 1), ("straße", 0)]
    assert (matches[0].start, matches[0].end) == (3, 9)


def test_case_insensitive_no_normalization_form() -> None:
    m = MultiPatternMatcher(["café"], case_insensitive=True)
    assert m.is_match("CAFÉ")
    assert not m.is_match("CAFÉ")


def test_case_insensitive_replace_all_and_batch() -> None:
    m = MultiPatternMatcher(["são paulo", "straße"], case_insensitive=True)
    assert m.replace_all("SÃO PAULO / STRASSE", ["SP", "ST"]) == "SP / ST"
    batch = m.find_all_batch(["SÃO PAULO", "", "Straße"])
    assert [[x.text for x in row] for row in batch] == [["SÃO PAULO"], [], ["Straße"]]


def test_case_insensitive_ascii_regression() -> None:
    m = MultiPatternMatcher(["hello", "world"], case_insensitive=True)
    text = "Hello WORLD hello"
    assert _spans(text, m.find_all(text)) == [("Hello", 0), ("WORLD", 1), ("hello", 0)]


def test_case_sensitive_default_unchanged() -> None:
    m = MultiPatternMatcher(["são paulo"])
    assert not m.is_match("SÃO PAULO")
    assert m.is_match("são paulo")


def test_empty_haystack() -> None:
    for kwargs in ({}, {"case_insensitive": True}, {"word_boundary": True}):
        m = MultiPatternMatcher(["a"], **kwargs)
        assert m.find_all("") == []
        assert m.count("") == 0
        assert not m.is_match("")


# ─── Word boundaries ────────────────────────────────────────────────────────


def test_word_boundary_rejects_partial_words() -> None:
    m = MultiPatternMatcher(["georgia", "Méx"], case_insensitive=True, word_boundary=True)
    assert m.find_all("Georgian food in Méxicali") == []
    text = "Georgia, not Georgian; Méx."
    assert _spans(text, m.find_all(text)) == [("Georgia", 0), ("Méx", 1)]


def test_word_boundary_default_off_keeps_substring_behaviour() -> None:
    m = MultiPatternMatcher(["georgia"], case_insensitive=True)
    assert m.count("Georgian") == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dog", 1),  # haystack edges are boundaries
        ("dog's", 1),  # apostrophe separates
        ("dog\N{RIGHT SINGLE QUOTATION MARK}s", 1),  # typographic apostrophe separates
        ("dog-house", 1),  # hyphen separates
        ("dog_bar", 1),  # underscore separates (not a letter/digit)
        ("(dog)", 1),
        ("dog😀dog", 2),  # emoji is not a word char
        ("dogs", 0),  # letters extend the word
        ("hotdog", 0),
        ("dog2", 0),  # digits are word chars
        ("2dog", 0),
        ("doǵ", 0),  # a combining mark continues the word
    ],
)
def test_word_boundary_separator_rules(text: str, expected: int) -> None:
    m = MultiPatternMatcher(["dog"], word_boundary=True)
    assert m.count(text) == expected
    assert len(m.find_all(text)) == expected
    assert m.is_match(text) is (expected > 0)


def test_word_boundary_numbers() -> None:
    m = MultiPatternMatcher(["2024"], word_boundary=True)
    text = "FY2024 2024 20245 2024-25"
    assert [x.start for x in m.find_all(text)] == [7, 18]


def test_word_boundary_unspaced_scripts_still_match() -> None:
    m = MultiPatternMatcher(["東京", "タワー", "กรุงเทพ"], word_boundary=True)
    text = "東京都の東京タワーとกรุงเทพมหานคร"
    assert _spans(text, m.find_all(text)) == [
        ("東京", 0),
        ("東京", 0),
        ("タワー", 1),
        ("กรุงเทพ", 2),
    ]


def test_word_boundary_latin_next_to_han() -> None:
    m = MultiPatternMatcher(["iphone"], case_insensitive=True, word_boundary=True)
    assert m.count("新型iPhone発売") == 1


def test_word_boundary_hangul_uses_spaces() -> None:
    m = MultiPatternMatcher(["서울"], word_boundary=True)
    assert m.count("서울시 서울") == 1


def test_word_boundary_rejection_lets_other_patterns_match() -> None:
    m = MultiPatternMatcher(["georgia", "georgian"], case_insensitive=True, word_boundary=True)
    assert _spans("Georgian", m.find_all("Georgian")) == [("Georgian", 1)]
    m = MultiPatternMatcher(
        ["new york", "york"], case_insensitive=True, longest_match=True, word_boundary=True
    )
    text = "New Yorker in York"
    assert _spans(text, m.find_all(text)) == [("York", 1)]


def test_word_boundary_consistent_across_methods() -> None:
    m = MultiPatternMatcher(["georgia"], case_insensitive=True, word_boundary=True)
    text = "Georgian GEORGIA georgia"
    assert len(m.find_all(text)) == 2
    assert m.count(text) == 2
    assert m.is_match(text)
    assert not m.is_match("Georgian")
    assert [len(r) for r in m.find_all_batch([text, "Georgian", ""])] == [2, 0, 0]
    assert m.replace_all(text, ["X"]) == "Georgian X X"


def test_word_boundary_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        MultiPatternMatcher(["a"], False, False, True)  # ty: ignore[too-many-positional-arguments]


def test_pickle_round_trip_keeps_options() -> None:
    m = MultiPatternMatcher(["georgia"], case_insensitive=True, word_boundary=True)
    m2 = pickle.loads(pickle.dumps(m))
    assert m2.count("Georgian GEORGIA") == 1
