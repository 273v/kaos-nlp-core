"""Integration tests for sentence/paragraph-level document diffing."""

from __future__ import annotations

from collections import Counter

import pytest

from kaos_nlp_core.documents import (
    SegmentChange,
    SegmentRef,
    diff_documents,
    summarize_changes,
)


def kinds(changes: list[SegmentChange]) -> list[str]:
    return [c.kind for c in changes]


# ── Identity ────────────────────────────────────────────────────────────────


def test_identical_documents_all_unchanged():
    text = "The Lessor leases the property. Rent is monthly. Term is one year."
    out = diff_documents(text, text)
    assert all(c.kind == "unchanged" for c in out)
    assert all(c.left is not None and c.right is not None for c in out)
    assert all(c.score == pytest.approx(1.0) for c in out)


def test_summarize_changes_counts_each_kind():
    text = "Sentence one. Sentence two."
    out = diff_documents(text, text)
    counts = summarize_changes(out)
    assert counts["unchanged"] == len(out)
    assert counts["modified"] == 0
    assert counts["added"] == 0
    assert counts["removed"] == 0


# ── Single edits ────────────────────────────────────────────────────────────


def test_single_edited_sentence_classified_as_modified():
    a = "The Lessor leases the property. Rent is monthly. Term is one year."
    b = "The Lessor leases the property. Rent is quarterly. Term is one year."
    out = diff_documents(a, b)
    counts = Counter(kinds(out))
    assert counts["unchanged"] == 2
    assert counts["modified"] == 1
    # Modified row should reference both sides with positive sub-1.0 score.
    modified = next(c for c in out if c.kind == "modified")
    assert modified.left is not None and modified.right is not None
    assert 0.0 < modified.score < 1.0
    assert "monthly" in modified.left_text
    assert "quarterly" in modified.right_text


def test_appended_sentence_classified_as_added():
    a = "Sentence one. Sentence two."
    b = "Sentence one. Sentence two. Sentence three."
    out = diff_documents(a, b)
    counts = Counter(kinds(out))
    assert counts["added"] == 1
    added = next(c for c in out if c.kind == "added")
    assert added.left is None
    assert added.right is not None
    assert "three" in added.right_text


def test_deleted_sentence_classified_as_removed():
    a = "Sentence one. Sentence two. Sentence three."
    b = "Sentence one. Sentence three."
    out = diff_documents(a, b)
    counts = Counter(kinds(out))
    assert counts["removed"] == 1
    removed = next(c for c in out if c.kind == "removed")
    assert removed.left is not None
    assert removed.right is None
    assert "two" in removed.left_text


def test_completely_disjoint_documents():
    a = "Alpha bravo charlie."
    b = "Lorem ipsum dolor."
    out = diff_documents(a, b)
    counts = Counter(kinds(out))
    assert counts["unchanged"] == 0
    assert counts["modified"] == 0
    assert counts["removed"] >= 1
    assert counts["added"] >= 1


# ── Empty input behaviour ───────────────────────────────────────────────────


def test_empty_left_yields_only_added():
    out = diff_documents("", "Brand new sentence.")
    assert all(c.kind == "added" for c in out)
    assert all(c.left is None for c in out)


def test_empty_right_yields_only_removed():
    out = diff_documents("Old removed sentence.", "")
    assert all(c.kind == "removed" for c in out)
    assert all(c.right is None for c in out)


def test_both_empty_returns_empty_list():
    assert diff_documents("", "") == []


# ── Granularity ─────────────────────────────────────────────────────────────


def test_paragraph_granularity_picks_up_paragraph_break():
    a = "First paragraph here.\n\nSecond paragraph here."
    b = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
    out = diff_documents(a, b, granularity="paragraph")
    counts = Counter(kinds(out))
    assert counts["added"] >= 1
    assert counts["unchanged"] >= 1


def test_paragraph_simple_granularity_works_without_punkt():
    a = "Paragraph A.\n\nParagraph B."
    b = "Paragraph A.\n\nParagraph B.\n\nParagraph C."
    out = diff_documents(a, b, granularity="paragraph_simple")
    counts = Counter(kinds(out))
    assert counts["added"] == 1


def test_line_granularity_compares_each_line():
    a = "alpha\nbravo\ncharlie"
    b = "alpha\ndelta\ncharlie"
    out = diff_documents(a, b, granularity="line")
    counts = Counter(kinds(out))
    assert counts["unchanged"] >= 2  # alpha, charlie
    # bravo and delta likely come back as removed/added (low overlap on tiny tokens)
    assert counts["added"] + counts["removed"] >= 2


def test_unknown_granularity_raises_value_error():
    with pytest.raises(ValueError, match="bogus"):
        diff_documents("a.", "b.", granularity="bogus")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


# ── Move detection ──────────────────────────────────────────────────────────


def test_detect_moves_off_by_default():
    a = "Line one is here.\nLine two is here.\nLine three is here.\nLine four is here."
    b = "Line two is here.\nLine three is here.\nLine four is here.\nLine one is here."
    out = diff_documents(a, b, granularity="line")
    assert all(c.kind != "moved" for c in out)


def test_detect_moves_relabels_position_swaps():
    a = "Line one is here.\nLine two is here.\nLine three is here.\nLine four is here."
    b = "Line two is here.\nLine three is here.\nLine four is here.\nLine one is here."
    out = diff_documents(
        a,
        b,
        granularity="line",
        detect_moves=True,
        move_distance_ratio=0.1,
    )
    counts = Counter(kinds(out))
    assert counts["moved"] >= 1


# ── Threshold tuning ────────────────────────────────────────────────────────


def test_low_modify_threshold_pairs_more_aggressively():
    """Lowering the modify floor should turn formerly removed/added pairs into modified."""
    a = "Some short clause here."
    b = "Different short clause text."
    strict = diff_documents(a, b, modify_threshold=0.9, match_threshold=0.95)
    loose = diff_documents(a, b, modify_threshold=0.1, match_threshold=0.95)
    strict_counts = Counter(kinds(strict))
    loose_counts = Counter(kinds(loose))
    # Strict version should have unmatched (added/removed); loose should pair them.
    assert strict_counts["modified"] == 0
    assert loose_counts["modified"] >= 1


def test_match_threshold_separates_unchanged_from_modified():
    """A near-identical pair counts as Modified when match_threshold is raised above the score."""
    a = "Rent shall be paid monthly."
    b = "Rent shall be paid weekly."
    high = diff_documents(a, b, match_threshold=0.99, modify_threshold=0.4)
    high_counts = Counter(kinds(high))
    assert high_counts["modified"] >= 1
    assert high_counts["unchanged"] == 0


# ── Algorithm selection ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "algorithm",
    [
        "token-jaccard",
        "token-ngram-jaccard",
        "ngram-cosine",
        "jaro-winkler",
        "levenshtein",
    ],
)
def test_diff_with_alternate_metric(algorithm):
    a = "Sentence one. Sentence two."
    b = "Sentence one. Sentence two."
    out = diff_documents(a, b, algorithm=algorithm)
    assert all(c.kind == "unchanged" for c in out)


def test_unknown_algorithm_raises_value_error():
    with pytest.raises(ValueError, match="bogus"):
        diff_documents("a.", "b.", algorithm="bogus-metric")


# ── Result shape ───────────────────────────────────────────────────────────


def test_segment_refs_populate_offsets_and_text():
    a = "First sentence. Second sentence."
    out = diff_documents(a, a)
    for c in out:
        assert c.left is not None
        assert isinstance(c.left, SegmentRef)
        assert c.left.start <= c.left.end
        assert a[c.left.start : c.left.end] == c.left.text


def test_unicode_offsets_are_character_offsets():
    """Char-offset rule: offsets should index Python str directly."""
    a = "Café au lait. Crème brûlée."
    b = "Café au lait. Crème brûlée."
    out = diff_documents(a, b)
    for c in out:
        assert c.left is not None
        # Char-offset slice into the source must equal the segment text.
        assert a[c.left.start : c.left.end] == c.left.text


# ── Bundled Punkt model is the default ─────────────────────────────────────


def test_default_uses_bundled_punkt_for_real_legal_text():
    """The default tokenizer should split realistic legal sentences correctly."""
    a = (
        "The parties hereby agree to the following terms. "
        "Section 1. Definitions. As used herein, the term Lessor shall mean. "
        "Section 2. Term. The term of this agreement is one year."
    )
    b = (
        "The parties hereby agree to the following terms. "
        "Section 1. Definitions. As used herein, the term Lessor shall mean. "
        "Section 2. Term. The term of this agreement is two years."
    )
    out = diff_documents(a, b)
    counts = Counter(kinds(out))
    # Expect at least one unchanged and one modified in this realistic split.
    assert counts["unchanged"] >= 1
    assert counts["modified"] >= 1
