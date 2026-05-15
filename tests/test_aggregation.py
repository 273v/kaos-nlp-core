"""Tests for :mod:`kaos_nlp_core.aggregation` pure aggregation functions."""

from __future__ import annotations

import pytest

from kaos_nlp_core.aggregation import (
    intersection,
    majority,
    max_score,
    union,
    vote,
    weighted,
)

# ---------------------------------------------------------------------------
# vote
# ---------------------------------------------------------------------------


class TestVote:
    def test_empty_input_returns_none(self) -> None:
        assert vote([]) is None

    def test_all_empty_chunks_returns_none(self) -> None:
        assert vote([[], [], []]) is None

    def test_unanimous(self) -> None:
        assert vote([["a"], ["a"], ["a"]]) == "a"

    def test_plurality(self) -> None:
        assert vote([["a"], ["b"], ["a"], ["c"]]) == "a"

    def test_tie_breaks_by_first_appearance(self) -> None:
        # ``b`` appears first, then ``a``. Both appear twice.
        assert vote([["b"], ["a"], ["b"], ["a"]]) == "b"

    def test_duplicates_within_chunk_dont_double_count(self) -> None:
        # ``a`` is duplicated inside chunk 0 but only counts once.
        assert vote([["a", "a"], ["b"], ["b"]]) == "b"

    def test_multi_label_chunks(self) -> None:
        assert vote([["a", "b"], ["b", "c"], ["b", "d"]]) == "b"


# ---------------------------------------------------------------------------
# majority
# ---------------------------------------------------------------------------


class TestMajority:
    def test_default_threshold_is_50_percent(self) -> None:
        # ``a`` has 2/3 ~= 0.67 > 0.5 -> wins.
        assert majority([["a"], ["a"], ["b"]]) == "a"

    def test_below_threshold_returns_none(self) -> None:
        # No label has > 50% of 4 chunks; each has 25%.
        assert majority([["a"], ["b"], ["c"], ["d"]]) is None

    def test_strict_majority_requires_more_than_half(self) -> None:
        # Two labels each appear in two of four chunks. The default
        # threshold is "at least half", so both qualify; the
        # tiebreak picks one deterministically by first appearance.
        result = majority([["a"], ["a"], ["b"], ["b"]])
        assert result in {"a", "b"}

    def test_custom_threshold(self) -> None:
        # ``a`` has 3/4 = 0.75; ``b`` has 1/4 = 0.25.
        assert majority([["a"], ["a"], ["a"], ["b"]], threshold=0.75) == "a"
        assert majority([["a"], ["a"], ["a"], ["b"]], threshold=0.8) is None

    def test_threshold_1_requires_unanimity(self) -> None:
        assert majority([["a"], ["a"], ["a"]], threshold=1.0) == "a"
        assert majority([["a"], ["a"], ["b"]], threshold=1.0) is None

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            majority([["a"]], threshold=0)
        with pytest.raises(ValueError):
            majority([["a"]], threshold=1.5)

    def test_empty_input(self) -> None:
        assert majority([]) is None


# ---------------------------------------------------------------------------
# union / intersection
# ---------------------------------------------------------------------------


class TestUnion:
    def test_empty_input(self) -> None:
        assert union([]) == frozenset()

    def test_basic(self) -> None:
        assert union([["a", "b"], ["b", "c"]]) == frozenset({"a", "b", "c"})

    def test_collapses_duplicates(self) -> None:
        assert union([["a"], ["a"], ["a"]]) == frozenset({"a"})

    def test_empty_chunk_skipped(self) -> None:
        assert union([[], ["a"], []]) == frozenset({"a"})


class TestIntersection:
    def test_empty_input(self) -> None:
        assert intersection([]) == frozenset()

    def test_basic(self) -> None:
        assert intersection([["a", "b"], ["b", "c"]]) == frozenset({"b"})

    def test_all_match(self) -> None:
        assert intersection([["a", "b"], ["a", "b"]]) == frozenset({"a", "b"})

    def test_no_intersection(self) -> None:
        assert intersection([["a"], ["b"]]) == frozenset()

    def test_single_chunk_yields_that_chunk(self) -> None:
        assert intersection([["a", "b"]]) == frozenset({"a", "b"})


# ---------------------------------------------------------------------------
# weighted
# ---------------------------------------------------------------------------


class TestWeighted:
    def test_uniform_weights_match_majority(self) -> None:
        assert weighted([["a"], ["a"], ["b"]]) == "a"

    def test_custom_weights_swing_decision(self) -> None:
        # Without weights: each appears in 1 chunk -> below threshold.
        # With weights, chunk 1 dominates -> ``b`` wins.
        result = weighted([["a"], ["b"]], weights=[0.1, 10.0])
        assert result == "b"

    def test_multi_returns_frozenset(self) -> None:
        result = weighted([["a"], ["a"], ["b"]], multi=True, threshold=0.5)
        assert result == frozenset({"a"})

    def test_invalid_weights_length(self) -> None:
        with pytest.raises(ValueError):
            weighted([["a"], ["b"]], weights=[1.0])

    def test_negative_total_weight_returns_none(self) -> None:
        assert weighted([["a"]], weights=[0.0]) is None

    def test_empty_input(self) -> None:
        assert weighted([]) is None
        assert weighted([], multi=True) == frozenset()


# ---------------------------------------------------------------------------
# max_score
# ---------------------------------------------------------------------------


class TestMaxScore:
    def test_empty(self) -> None:
        assert max_score([]) is None
        assert max_score([], multi=True) == frozenset()

    def test_single_chunk(self) -> None:
        assert max_score([{"a": 0.5, "b": 0.9}]) == "b"

    def test_takes_max_per_label(self) -> None:
        # ``a`` peaks at 0.4 in chunk 1; ``b`` peaks at 0.6 in chunk 2.
        assert max_score([{"a": 0.4, "b": 0.5}, {"a": 0.3, "b": 0.6}]) == "b"

    def test_multi_includes_all_positive(self) -> None:
        result = max_score([{"a": 0.5, "b": 0.4}], multi=True)
        assert result == frozenset({"a", "b"})

    def test_multi_threshold_filters(self) -> None:
        result = max_score(
            [{"a": 0.5, "b": 0.4, "c": 0.1}],
            multi=True,
            threshold=0.45,
        )
        assert result == frozenset({"a"})

    def test_threshold_below_top_returns_none(self) -> None:
        assert max_score([{"a": 0.2}], threshold=0.5) is None


# ---------------------------------------------------------------------------
# Module exposure
# ---------------------------------------------------------------------------


def test_aggregation_exposed_from_top_level() -> None:
    import kaos_nlp_core

    assert hasattr(kaos_nlp_core, "aggregation")
    assert "aggregation" in kaos_nlp_core.__all__


def test_all_functions_exported() -> None:
    from kaos_nlp_core import aggregation

    for name in ("vote", "majority", "union", "intersection", "weighted", "max_score"):
        assert name in aggregation.__all__
