"""Unit tests for the class-based TF-IDF (c-TF-IDF) wrapper."""

from __future__ import annotations

import pytest

from kaos_nlp_core.ctfidf import class_tfidf

_STOP = {"the", "a", "for", "of", "and", "to", "was", "by", "into", "with"}

_LITIGATION = [
    "the court granted the motion for summary judgment",
    "plaintiff filed a motion for summary judgment",
]
_BAKING = [
    "mix the flour sugar and eggs into a batter",
    "bake the batter with flour for thirty minutes",
]


def test_separates_class_vocabularies() -> None:
    texts = [*_LITIGATION, *_BAKING]
    ids = [0, 0, 1, 1]
    labels = class_tfidf(texts, ids, top_k=5, ngram_range=(1, 1), stopwords=_STOP)
    assert set(labels) == {0, 1}
    kw0 = {t for t, _ in labels[0]}
    kw1 = {t for t, _ in labels[1]}
    assert kw0 & {"motion", "summary", "judgment", "court"}
    assert kw1 & {"flour", "batter", "eggs", "bake"}
    assert "flour" not in kw0 and "judgment" not in kw1


def test_returns_descending_scores() -> None:
    labels = class_tfidf(["alpha alpha beta gamma", "delta epsilon"], [0, 1], stopwords=_STOP)
    for terms in labels.values():
        scores = [s for _, s in terms]
        assert scores == sorted(scores, reverse=True)


def test_arbitrary_class_ids_preserved_in_first_seen_order() -> None:
    texts = ["cat dog", "fish bird", "cat dog"]
    labels = class_tfidf(texts, ["animals", "water", "animals"], stopwords=set())
    assert list(labels.keys()) == ["animals", "water"]  # first-seen order


def test_top_k_respected() -> None:
    labels = class_tfidf(["one two three four five six", "x y z"], [0, 1], top_k=2, stopwords=set())
    assert all(len(terms) <= 2 for terms in labels.values())


def test_ngram_range_produces_bigrams() -> None:
    labels = class_tfidf(
        ["summary judgment motion", "kitchen baking recipe"],
        [0, 1],
        ngram_range=(1, 2),
        top_k=20,
        stopwords=_STOP,
    )
    assert any(" " in t for t, _ in labels[0])


def test_token_prefix_conflates_variants() -> None:
    texts = ["automobile automotive autos", "kitchen cooking recipe"]
    plain = class_tfidf(texts, [0, 1], ngram_range=(1, 1), stopwords=set())
    prefixed = class_tfidf(texts, [0, 1], ngram_range=(1, 1), stopwords=set(), token_prefix=4)
    kw_prefixed = {t for t, _ in prefixed[0]}
    assert "auto" in kw_prefixed
    assert "automobile" not in kw_prefixed
    assert len(prefixed[0]) < len(plain[0])  # variants collapsed


def test_min_df_filters() -> None:
    labels = class_tfidf(
        ["common common rare", "common common common"], [0, 1], min_df=2, stopwords=set()
    )
    assert "rare" not in {t for t, _ in labels[0]}


def test_bm25_and_reduce_frequent_words_finite() -> None:
    labels = class_tfidf(
        ["alpha alpha beta", "alpha gamma gamma"],
        [0, 1],
        bm25_weighting=True,
        reduce_frequent_words=True,
        stopwords=set(),
    )
    for terms in labels.values():
        for _, s in terms:
            assert s >= 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="match in length"):
        class_tfidf(["a", "b"], [0])


def test_deterministic() -> None:
    texts = ["the quick brown fox", "lazy dog sleeps", "quick brown dog"]
    ids = [0, 1, 0]
    a = class_tfidf(texts, ids, ngram_range=(1, 2), stopwords=_STOP)
    b = class_tfidf(texts, ids, ngram_range=(1, 2), stopwords=_STOP)
    assert a == b


def test_empty() -> None:
    assert class_tfidf([], []) == {}
