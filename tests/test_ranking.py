"""Integration tests for `most_similar` / `least_similar` ranking helpers."""

from __future__ import annotations

import pytest

from kaos_nlp_core.algorithms import least_similar, most_similar

# ── Basic behaviour ─────────────────────────────────────────────────────────


def test_most_similar_picks_self_match():
    """A query that's exactly in the candidate list should rank first at 1.0."""
    choices = ["apple", "apply", "ample", "orange", "banana"]
    out = most_similar("apple", choices, k=3)
    assert out[0] == ("apple", pytest.approx(1.0))
    assert len(out) == 3


def test_least_similar_picks_most_different():
    """Ascending order pulls the most-different candidate first."""
    choices = ["apple", "apply", "ample", "completely-different-zzz"]
    out = least_similar("apple", choices, k=1)
    assert out[0][0] == "completely-different-zzz"


def test_most_similar_default_metric_is_jaro_winkler():
    """Per-pair scores should match jaro_winkler() output exactly."""
    from kaos_nlp_core.algorithms import jaro_winkler

    choices = ["apply", "ample"]
    out = most_similar("apple", choices, k=2)
    out_map = dict(out)
    for c in choices:
        assert out_map[c] == pytest.approx(jaro_winkler("apple", c).similarity)


# ── Edge cases ──────────────────────────────────────────────────────────────


def test_empty_choices_returns_empty():
    assert most_similar("apple", [], k=5) == []
    assert least_similar("apple", [], k=5) == []


def test_k_zero_returns_empty():
    assert most_similar("apple", ["a", "b"], k=0) == []


def test_k_larger_than_choices_returns_all():
    out = most_similar("apple", ["apply", "ample"], k=99)
    assert len(out) == 2


def test_threshold_filters_below_floor():
    """Descending threshold drops scores below the floor."""
    choices = ["apple", "apply", "zzzz", "qqqq"]
    out = most_similar("apple", choices, k=10, threshold=0.6)
    assert all(s >= 0.6 for _, s in out)
    candidates = [c for c, _ in out]
    assert "zzzz" not in candidates
    assert "qqqq" not in candidates


def test_least_similar_threshold_acts_as_ceiling():
    """Ascending threshold drops scores ABOVE the ceiling."""
    choices = ["apple", "apply", "zzzz"]
    out = least_similar("apple", choices, k=10, threshold=0.5)
    assert all(s <= 0.5 for _, s in out)


def test_ties_broken_by_original_order():
    """Three identical candidates → output preserves input order."""
    out = most_similar("apple", ["apple", "apple", "apple"], k=3)
    assert [s for _, s in out] == [pytest.approx(1.0)] * 3


# ── Metric coverage ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "algorithm",
    [
        "levenshtein",
        "damerau",
        "osa",
        "jaro",
        "jaro-winkler",
        "sorensen-dice",
        "ngram-jaccard",
        "ngram-cosine",
        "ngram-overlap",
        "token-jaccard",
        "token-ngram-jaccard",
        "lcs",
        "longest-common-substring",
    ],
)
def test_metric_keys_accepted(algorithm):
    """Every dispatch-supported metric should rank without error."""
    out = most_similar("apple", ["apply", "orange"], algorithm=algorithm, k=2)
    assert len(out) == 2
    assert all(0.0 <= s <= 1.0 for _, s in out)


def test_unknown_algorithm_raises_value_error():
    with pytest.raises(ValueError, match="bogus"):
        most_similar("apple", ["apply"], algorithm="bogus", k=1)


def test_token_jaccard_lowercase_picks_case_insensitive_match():
    """token-jaccard with lowercase=True should match across case."""
    choices = ["The Quick Brown Fox", "Slow green turtle"]
    out = most_similar(
        "the quick brown fox",
        choices,
        algorithm="token-jaccard",
        k=1,
        lowercase=True,
    )
    assert out[0][0] == "The Quick Brown Fox"
    assert out[0][1] == pytest.approx(1.0)


def test_jaro_winkler_prefix_weight_passthrough():
    """High prefix_weight should boost candidates with shared prefixes."""
    choices = ["application", "pineapple"]
    high = most_similar("apple", choices, algorithm="jaro-winkler", k=2, prefix_weight=0.25)
    low = most_similar("apple", choices, algorithm="jaro-winkler", k=2, prefix_weight=0.0)
    high_map = dict(high)
    low_map = dict(low)
    # 'application' shares the 'app' prefix; the prefix bonus should lift it.
    assert high_map["application"] >= low_map["application"]


def test_hamming_unequal_length_drops_candidates():
    """Hamming requires equal-length strings; mismatched candidates drop."""
    out = most_similar("ax", ["abc", "ab", "ax"], algorithm="hamming", k=10)
    candidates = [c for c, _ in out]
    assert "abc" not in candidates  # length 3 vs query length 2
    assert "ab" in candidates
    assert "ax" in candidates


# ── Sequence input acceptance ───────────────────────────────────────────────


def test_accepts_tuple_of_choices():
    """Wrapper coerces any Sequence to list before crossing FFI."""
    out = most_similar("apple", ("apply", "ample"), k=2)
    assert len(out) == 2
