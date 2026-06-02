"""Tests for the curated English stopword resource."""

from __future__ import annotations

import pytest

from kaos_nlp_core.stopwords import stopwords, stopwords_provenance


def test_loads_nonempty_frozenset() -> None:
    sw = stopwords()
    assert isinstance(sw, frozenset)
    assert len(sw) > 100  # ~200 curated function words
    assert all(isinstance(w, str) and w == w.lower() for w in sw)


def test_contains_core_function_words() -> None:
    sw = stopwords()
    for w in (
        "the",
        "of",
        "and",
        "to",
        "is",
        "are",
        "was",
        "be",
        "in",
        "on",
        "by",
        "for",
        "with",
        "that",
        "not",
        "it",
        "he",
        "she",
        "they",
        "their",
        "this",
        "which",
        "but",
        "if",
        "or",
        "as",
        "from",
    ):
        assert w in sw, f"expected function word {w!r} missing"


def test_excludes_legal_content_words() -> None:
    # Cross-domain contrast + curation must keep distinctive legal/content
    # NOUNS out of the stopword list (they should remain available as labels).
    sw = stopwords()
    for w in (
        "act",
        "section",
        "agreement",
        "court",
        "united",
        "states",
        "president",
        "contract",
        "agency",
    ):
        assert w not in sw, f"content word {w!r} leaked into stopwords"


def test_cached_identity() -> None:
    assert stopwords() is stopwords()  # lru_cache


def test_provenance_records_method_and_sources() -> None:
    prov = stopwords_provenance()
    assert "cross-domain" in prov["method"].lower()
    assert prov["statistical"]["metric"] == "document_frequency"
    assert prov["statistical"]["sources"]  # per-source doc counts
    assert prov["total"] == len(stopwords())


def test_unknown_language_raises() -> None:
    with pytest.raises(ValueError, match="no stopword resource"):
        stopwords("xx")
