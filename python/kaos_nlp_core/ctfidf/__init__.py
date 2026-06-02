"""Class-based TF-IDF (c-TF-IDF) — distinctive terms per class/cluster.

The BERTopic cluster-labelling kernel, in Rust: treat each class (cluster)
as one concatenated document and run TF-IDF treating the *set of classes*
as the corpus, so a term scores high when it's frequent **within** a class
and distinctive **across** classes — what a cluster label wants.

The compute (tokenize → n-gram → per-class counts → weighting) lives in the
Rust core ``crate::core::ctfidf`` and reuses the crate's word tokenizer, so
labels are consistent with the MinHash / retrieval paths. This module is a
thin typed wrapper that maps arbitrary (hashable) class ids onto the Rust
kernel's ``0..n_classes`` indices and back.

Example::

    from kaos_nlp_core.ctfidf import class_tfidf

    texts = ["motion for summary judgment", "plaintiff's summary judgment motion",
             "mix flour and eggs", "bake the flour batter"]
    labels = class_tfidf(texts, [0, 0, 1, 1], top_k=3)
    # {0: [("summary judgment", ...), ("motion", ...), ...],
    #  1: [("flour", ...), ("batter", ...), ...]}
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from kaos_nlp_core._rust.ctfidf import class_tfidf as _rust_class_tfidf


def class_tfidf(
    texts: Sequence[str],
    class_ids: Sequence[Any],
    *,
    top_k: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
    stopwords: Iterable[str] | None = None,
    min_df: int = 1,
    reduce_frequent_words: bool = False,
    bm25_weighting: bool = False,
    lowercase: bool = True,
    token_prefix: int = 0,
) -> dict[Any, list[tuple[str, float]]]:
    """Top distinctive terms per class via class-based TF-IDF.

    Args:
        texts: one string per item.
        class_ids: the class/cluster id for each text (any hashable; e.g.
            component labels). Same length as ``texts``.
        top_k: terms kept per class (most distinctive first).
        ngram_range: inclusive ``(min_n, max_n)`` for n-gram terms.
        stopwords: tokens to drop before counting. ``None`` (default)
            removes none — this primitive is language-neutral; callers pass
            their own stopword set (e.g. an English list).
        min_df: drop terms whose total count across all classes is below
            this (noise floor).
        reduce_frequent_words: take ``sqrt`` of the L1-normalised term
            frequency (suppresses residual high-frequency words).
        bm25_weighting: use the smoothed BM25-style IDF (steadier on small
            corpora / few classes).
        lowercase: lowercase tokens before counting.
        token_prefix: when ``> 0``, truncate each token to this many
            characters before counting — a dependency-free conflation of
            morphological/derivational variants (``4`` merges
            ``automobile``/``automotive``/``autos`` → ``auto``). Precision-
            light and yields truncated surface forms, so prefer it for
            grouping over human-facing labels.

    Returns:
        ``{class_id: [(term, weight), ...]}`` for every distinct class id,
        in first-seen order, each list ranked by descending weight.

    Raises:
        ValueError: ``texts`` and ``class_ids`` differ in length, or an
            invalid n-gram range.
    """
    texts_list = list(texts)
    ids_list = list(class_ids)
    if len(texts_list) != len(ids_list):
        msg = f"texts and class_ids must match in length ({len(texts_list)} != {len(ids_list)})."
        raise ValueError(msg)

    # Map arbitrary class ids onto contiguous 0..n indices (first-seen order).
    index_of: dict[Any, int] = {}
    order: list[Any] = []
    indices: list[int] = []
    for cid in ids_list:
        if cid not in index_of:
            index_of[cid] = len(order)
            order.append(cid)
        indices.append(index_of[cid])

    lo, hi = ngram_range
    raw = _rust_class_tfidf(
        texts_list,
        indices,
        len(order),
        lo,
        hi,
        top_k,
        min_df,
        reduce_frequent_words,
        bm25_weighting,
        lowercase,
        token_prefix,
        list(stopwords) if stopwords is not None else None,
    )
    return {order[i]: [(term, float(score)) for term, score in raw[i]] for i in range(len(order))}


__all__ = ["class_tfidf"]
