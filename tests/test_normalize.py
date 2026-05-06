"""Tests for :func:`kaos_nlp_core.segmentation.normalize` at the PyO3 boundary.

Two responsibilities here:

1. Verify that source-side **character** offsets the Python wrapper exposes
   actually round-trip through Python ``str`` indexing. The Rust core works
   in byte offsets; this test class is the contract that the FFI conversion
   stayed honest.
2. Cover the shape of the public Python API (kwargs, fast path, error mode
   for the deferred enumerator option).
"""

from __future__ import annotations

import pytest

from kaos_nlp_core.segmentation import NormalizedText, normalize

# ─── Fast path / no-op contract ─────────────────────────────────────────────


def test_no_flags_returns_input_unchanged() -> None:
    src = "Hello, World!"
    r = normalize(src)
    assert isinstance(r, NormalizedText)
    assert r.text == src
    assert r.orig_char_offsets is None


def test_ascii_input_with_unicode_flag_only_is_no_op() -> None:
    src = "Plain ASCII text."
    r = normalize(src, normalize_unicode_punct=True)
    assert r.text == src
    assert r.orig_char_offsets is None


def test_no_op_original_char_uses_identity() -> None:
    r = normalize("abc")
    assert r.original_char(0) == 0
    assert r.original_char(3) == 3
    assert r.original_char(4) is None


# ─── Smart-quote / dash / ellipsis collapse ─────────────────────────────────


def test_smart_quotes_collapse() -> None:
    r = normalize("‘hello’ “world”", normalize_unicode_punct=True)
    assert r.text == "'hello' \"world\""


def test_dash_family_collapses_to_hyphen() -> None:
    r = normalize("en–dash em—dash minus−sign", normalize_unicode_punct=True)
    assert r.text == "en-dash em-dash minus-sign"


def test_ellipsis_expands() -> None:
    r = normalize("wait…what?", normalize_unicode_punct=True)
    assert r.text == "wait...what?"


def test_unicode_spaces_collapse_to_ascii_space() -> None:
    r = normalize("a b c　d", normalize_unicode_punct=True)
    assert r.text == "a b c d"


def test_zero_width_and_soft_hyphen_deleted() -> None:
    src = "co­ope​rate"  # SOFT HYPHEN + ZWSP
    r = normalize(src, normalize_unicode_punct=True)
    assert r.text == "cooperate"


# ─── Whitespace collapse ────────────────────────────────────────────────────


def test_collapse_runs() -> None:
    r = normalize("a  \t\n b", collapse_whitespace=True)
    assert r.text == "a b"


def test_collapse_preserves_edge_whitespace_as_single_space() -> None:
    r = normalize("   a   b   ", collapse_whitespace=True)
    assert r.text == " a b "


# ─── Fold case ──────────────────────────────────────────────────────────────


def test_fold_case_ascii() -> None:
    r = normalize("HELLO World", fold_case=True)
    assert r.text == "hello world"


def test_fold_case_only_affects_ascii_letters() -> None:
    r = normalize("ÉCOLE", fold_case=True)
    assert r.text == "École"


# ─── Strip ASCII punctuation ────────────────────────────────────────────────


def test_strip_punctuation() -> None:
    r = normalize("Hello, World! (here)", strip_punctuation=True)
    assert r.text == "Hello World here"


# ─── Deferred enumerator-prefix option (P3) ────────────────────────────────


def test_strip_enumerator_prefix_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Enumerator parser"):
        normalize("I. Introduction", strip_enumerator_prefix=True)


# ─── Source-side **character** offsets through the FFI ─────────────────────


def test_orig_char_offsets_lengths_match_text_chars() -> None:
    r = normalize("a…b", normalize_unicode_punct=True)
    assert r.text == "a...b"
    assert r.orig_char_offsets is not None
    assert len(r.orig_char_offsets) == len(r.text)


def test_orig_char_offsets_for_ellipsis_expansion_point_to_same_source_char() -> None:
    src = "a…b"
    r = normalize(src, normalize_unicode_punct=True)
    map_ = r.orig_char_offsets
    assert map_ is not None
    # Source: 'a' at char 0, '…' at char 1, 'b' at char 2.
    # Output: 'a' at 0 ← src 0; '.', '.', '.' at 1,2,3 all ← src 1; 'b' at 4 ← src 2.
    assert map_[0] == 0
    assert map_[1] == 1
    assert map_[2] == 1
    assert map_[3] == 1
    assert map_[4] == 2


def test_orig_char_offsets_round_trip_on_python_str() -> None:
    """Each output char must map to a valid source char via Python str indexing."""
    src = "  “HELLO,—World…”  "
    r = normalize(
        src,
        collapse_whitespace=True,
        fold_case=True,
        normalize_unicode_punct=True,
    )
    assert r.orig_char_offsets is not None
    for i, _ in enumerate(r.text):
        src_idx = r.orig_char_offsets[i]
        assert 0 <= src_idx <= len(src), (i, src_idx)
        # Index must land at a real char position: src[src_idx] should not raise.
        if src_idx < len(src):
            _ = src[src_idx]


def test_orig_char_offsets_are_monotonic() -> None:
    src = "  “HELLO,—World…”  "
    r = normalize(
        src,
        collapse_whitespace=True,
        fold_case=True,
        normalize_unicode_punct=True,
    )
    assert r.orig_char_offsets is not None
    for prev, nxt in zip(r.orig_char_offsets, r.orig_char_offsets[1:], strict=False):
        assert prev <= nxt


# ─── Composition of all flags ──────────────────────────────────────────────


def test_aggressive_legal_quote_normalization() -> None:
    src = "  “HELLO,—World…”  "
    r = normalize(
        src,
        collapse_whitespace=True,
        fold_case=True,
        normalize_unicode_punct=True,
    )
    assert r.text == ' "hello,-world..." '


# ─── Repr ──────────────────────────────────────────────────────────────────


def test_repr_is_informative() -> None:
    r = normalize("a…b", normalize_unicode_punct=True)
    s = repr(r)
    assert "NormalizedText" in s
    assert "chars=" in s
