"""Tests for `extract_line_records` at the PyO3 boundary.

These tests intentionally exercise Python ``str`` slicing on the offsets
returned by the binding — that is the test that catches byte/char-offset
regressions. The Rust core works in byte offsets; the binding converts at
the FFI boundary; these tests verify the conversion stayed honest.
"""

from __future__ import annotations

import pickle

import pytest

from kaos_nlp_core.segmentation import (
    LineRecord,
    PunctProfile,
    extract_line_records,
)

# ─── Round-trip slicing on Python str (the byte/char regression net) ─────────


@pytest.mark.parametrize(
    "text",
    [
        "ASCII only line\nsecond line\n",
        "café latte\nédition spéciale\n",
        "東京\n京都タワー\n",
        "😀\n😎🎉\n",
        # Mixed: multi-byte char on one line, ASCII on the next, emoji on the last.
        "ASCII\ncafé\n東京\n😀\n",
        # Trailing line with no terminator.
        "first\nsecond",
        # CRLF and bare CR.
        "windows\r\nold-mac\r\n",
        # All blanks.
        "\n\n\n",
        # Empty.
        "",
    ],
)
def test_text_slicing_round_trip(text: str) -> None:
    """``text[r.start:r.end]`` must reproduce each line content verbatim."""
    records = extract_line_records(text)
    if not text:
        assert records == []
        return
    cursor = 0
    for r in records:
        # Char-level slice — Python ``str`` indexing is char-based.
        line = text[r.start : r.end]
        # The line content must be free of newline characters.
        assert "\n" not in line, f"line {line!r} contained \\n"
        assert "\r" not in line, f"line {line!r} contained \\r"
        # Cursor must move forward.
        assert r.start >= cursor
        cursor = r.end + r.term_len


def test_char_len_matches_python_slicing() -> None:
    text = "café\n東京\n😀\n"
    records = extract_line_records(text)
    assert len(records) == 3
    assert records[0].char_len == len("café")
    assert records[1].char_len == len("東京")
    assert records[2].char_len == len("😀")


def test_byte_len_matches_utf8_size() -> None:
    text = "café\n東京\n😀\n"
    records = extract_line_records(text)
    assert records[0].byte_len == len("café".encode())
    assert records[1].byte_len == len("東京".encode())
    assert records[2].byte_len == len("😀".encode())


# ─── Terminator classification ────────────────────────────────────────────────


def test_terminator_lf() -> None:
    [r] = extract_line_records("hello\n")
    assert r.terminator == "lf"
    assert r.term_len == 1


def test_terminator_crlf() -> None:
    [r] = extract_line_records("hello\r\n")
    assert r.terminator == "crlf"
    assert r.term_len == 2


def test_terminator_cr_only() -> None:
    [r] = extract_line_records("hello\r")
    assert r.terminator == "cr"
    assert r.term_len == 1


def test_terminator_none_for_trailing_line() -> None:
    records = extract_line_records("a\nb")
    assert records[1].terminator == "none"
    assert records[1].term_len == 0


# ─── Indent + stripped offsets ────────────────────────────────────────────────


def test_indent_chars() -> None:
    [r] = extract_line_records("    indented")
    assert r.indent_chars == 4
    assert r.stripped_start == 4
    text = "    indented"
    assert text[r.stripped_start : r.stripped_end] == "indented"


def test_indent_chars_unicode_indent() -> None:
    # U+3000 IDEOGRAPHIC SPACE counts as whitespace.
    text = "　　hi"
    [r] = extract_line_records(text)
    assert r.indent_chars == 2
    assert text[r.stripped_start : r.stripped_end] == "hi"


def test_stripped_offsets_inside_line() -> None:
    text = "  padded body line.   \n"
    [r] = extract_line_records(text)
    assert r.start <= r.stripped_start <= r.stripped_end <= r.end
    assert text[r.stripped_start : r.stripped_end] == "padded body line."


# ─── Case + punct profiles ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("HELLO WORLD", "all_caps"),
        ("Hello World", "title_case"),
        ("hello world", "all_lower"),
        ("Hello world", "initial_cap"),
        ("HelLO worLD", "mixed_case"),
        ("12345", "no_alpha"),
    ],
)
def test_case_profile(text: str, expected: str) -> None:
    [r] = extract_line_records(text)
    assert r.case_profile == expected


def test_punct_profile_endings() -> None:
    text = "Section 1.\nWhereas:\nList,\nQuestion?"
    records = extract_line_records(text)
    flags0 = PunctProfile(records[0].punct_profile)
    flags1 = PunctProfile(records[1].punct_profile)
    flags2 = PunctProfile(records[2].punct_profile)
    flags3 = PunctProfile(records[3].punct_profile)
    assert PunctProfile.ENDS_PERIOD in flags0
    assert PunctProfile.ENDS_COLON in flags1
    assert PunctProfile.ENDS_COMMA in flags2
    assert PunctProfile.ENDS_QUESTION in flags3


def test_punct_profile_signals() -> None:
    text = "a | b | c\n(1) parens\n[2] brackets\n§ 5 statute"
    records = extract_line_records(text)
    assert PunctProfile.HAS_PIPE in PunctProfile(records[0].punct_profile)
    assert PunctProfile.HAS_PARENS in PunctProfile(records[1].punct_profile)
    assert PunctProfile.HAS_BRACKETS in PunctProfile(records[2].punct_profile)
    assert PunctProfile.HAS_SECTION_SIG in PunctProfile(records[3].punct_profile)


def test_punct_profile_column_gaps_signal() -> None:
    text = "col1     col2     col3"
    [r] = extract_line_records(text)
    assert PunctProfile.HAS_COLUMN_GAPS in PunctProfile(r.punct_profile)


# ─── Blank lines + neighbour flags ───────────────────────────────────────────


def test_blank_line_recorded() -> None:
    records = extract_line_records("a\n\nb\n")
    assert len(records) == 3
    assert records[0].blank is False
    assert records[1].blank is True
    assert records[2].blank is False
    # Neighbour flags
    assert records[0].blank_after is True
    assert records[2].blank_before is True


def test_blank_line_with_only_whitespace() -> None:
    records = extract_line_records("a\n   \nb\n")
    assert records[1].blank is True
    assert records[1].stripped_start == records[1].stripped_end


# ─── Pickle round-trip ────────────────────────────────────────────────────────


def test_pickle_round_trip() -> None:
    records = extract_line_records("hello world\nSection 5: rights\n")
    for r in records:
        # Some IDE settings might warn; pickle uses __getstate__/__setstate__.
        copy = pickle.loads(pickle.dumps(r))
        assert copy.start == r.start
        assert copy.end == r.end
        assert copy.case_profile == r.case_profile
        assert copy.punct_profile == r.punct_profile
        assert copy.blank == r.blank


# ─── Edge cases ───────────────────────────────────────────────────────────────


def test_empty_input_returns_empty() -> None:
    assert extract_line_records("") == []


def test_single_blank_line() -> None:
    [r] = extract_line_records("\n")
    assert r.blank is True
    assert r.terminator == "lf"


def test_record_repr_is_informative() -> None:
    [r] = extract_line_records("Hello world.")
    s = repr(r)
    assert "LineRecord" in s
    assert "case=" in s


def test_constructor_creates_placeholder() -> None:
    """`LineRecord()` must succeed for pickle's protocol; it creates a blank record."""
    blank = LineRecord()
    assert blank.start == 0
    assert blank.end == 0
    assert blank.blank is True
