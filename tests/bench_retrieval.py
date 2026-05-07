"""Benchmarks using real legal/regulatory corpora (pytest-benchmark).

Measures retrieval performance at realistic scale:
  - BM25 over 69K US Code sections
  - TF-IDF variants at scale
  - Index build time
  - Query latency for common, rare, and multi-term queries
"""

import pytest

from kaos_nlp_core.algorithms import token_jaccard, token_ngram_jaccard
from kaos_nlp_core.structures import InvertedIndex


def tokenize_doc(text: str) -> list[str]:
    """Tokenize using the shared ICU-based tokenizer."""
    from conftest import DEFAULT_TOKENIZER

    return DEFAULT_TOKENIZER.tokenize_words(text)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def usc_index(usc_docs):
    """Pre-built inverted index over full US Code."""
    idx = InvertedIndex()
    for doc in usc_docs:
        terms = tokenize_doc(doc["text"])
        idx.add_document(doc["id"], terms)
    return idx


@pytest.fixture(scope="module")
def edgar_index(edgar_docs):
    """Pre-built inverted index over 200 EDGAR agreements."""
    idx = InvertedIndex()
    for doc in edgar_docs:
        terms = tokenize_doc(doc["text"])
        idx.add_document(doc["id"], terms)
    return idx


# ── Index build benchmarks ───────────────────────────────────────────────────


@pytest.mark.benchmark(group="index_build")
def test_build_usc_index(benchmark, usc_docs):
    """Build inverted index over ~69K US Code sections."""

    def run():
        idx = InvertedIndex()
        for doc in usc_docs:
            terms = tokenize_doc(doc["text"])
            idx.add_document(doc["id"], terms)
        return idx.doc_count()

    result = benchmark.pedantic(run, rounds=3, iterations=1)
    assert result >= 60000


@pytest.mark.benchmark(group="index_build")
def test_build_edgar_index(benchmark, edgar_docs):
    """Build inverted index over 200 EDGAR agreements."""

    def run():
        idx = InvertedIndex()
        for doc in edgar_docs:
            terms = tokenize_doc(doc["text"])
            idx.add_document(doc["id"], terms)
        return idx.doc_count()

    result = benchmark.pedantic(run, rounds=5, iterations=1)
    assert result == 200


# ── BM25 query benchmarks at scale ──────────────────────────────────────────


@pytest.mark.benchmark(group="bm25_usc")
def test_bm25_common_term(benchmark, usc_index):
    """BM25 query for a very common term (high df, low IDF)."""
    benchmark(usc_index.query_bm25, ["the"], 10)


@pytest.mark.benchmark(group="bm25_usc")
def test_bm25_2_terms(benchmark, usc_index):
    """BM25 query: 2 moderately common legal terms."""
    benchmark(usc_index.query_bm25, ["tax", "income"], 10)


@pytest.mark.benchmark(group="bm25_usc")
def test_bm25_5_terms(benchmark, usc_index):
    """BM25 query: 5 specific legal terms."""
    benchmark(
        usc_index.query_bm25,
        ["copyright", "infringement", "damages", "author", "work"],
        10,
    )


@pytest.mark.benchmark(group="bm25_usc")
def test_bm25_rare_term(benchmark, usc_index):
    """BM25 query for a rare/specialized term."""
    benchmark(usc_index.query_bm25, ["cryptocurrency"], 10)


@pytest.mark.benchmark(group="bm25_usc")
def test_bm25_top100(benchmark, usc_index):
    """BM25 top-100 retrieval (larger result set)."""
    benchmark(
        usc_index.query_bm25,
        ["military", "armed", "forces"],
        100,
    )


@pytest.mark.benchmark(group="bm25_usc")
def test_bm25_no_results(benchmark, usc_index):
    """BM25 query with zero results (nonexistent term)."""
    benchmark(usc_index.query_bm25, ["xyzzy_nonexistent_42"], 10)


# ── TF-IDF query benchmarks ─────────────────────────────────────────────────


@pytest.mark.benchmark(group="tfidf_usc")
def test_tfidf_sublinear_smooth(benchmark, usc_index):
    """TF-IDF (sublinear/smooth) top-10 retrieval."""
    benchmark(
        usc_index.query_tf_idf,
        ["bankruptcy", "debtor", "creditor"],
        10,
        "sublinear",
        "smooth",
    )


@pytest.mark.benchmark(group="tfidf_usc")
def test_tfidf_raw_standard(benchmark, usc_index):
    """TF-IDF (raw/standard) top-10 retrieval."""
    benchmark(
        usc_index.query_tf_idf,
        ["bankruptcy", "debtor", "creditor"],
        10,
        "raw",
        "standard",
    )


# ── Document similarity benchmarks ──────────────────────────────────────────


@pytest.mark.benchmark(group="doc_similarity")
def test_token_jaccard_edgar_pair(benchmark, edgar_docs):
    """Token Jaccard between two real contracts (2000 char prefix)."""
    text1 = edgar_docs[0]["text"][:2000]
    text2 = edgar_docs[1]["text"][:2000]
    benchmark(token_jaccard, text1, text2, True)


@pytest.mark.benchmark(group="doc_similarity")
def test_token_ngram_jaccard_edgar_pair(benchmark, edgar_docs):
    """Token bigram Jaccard between two real contracts."""
    text1 = edgar_docs[0]["text"][:2000]
    text2 = edgar_docs[1]["text"][:2000]
    benchmark(token_ngram_jaccard, text1, text2, 2, True)


@pytest.mark.benchmark(group="doc_similarity")
def test_token_jaccard_edgar_10_pairs(benchmark, edgar_docs):
    """Token Jaccard across 10 contract pairs."""
    texts = [d["text"][:2000] for d in edgar_docs[:10]]
    pairs = [(texts[i], texts[i + 1]) for i in range(0, len(texts) - 1, 2)]
    benchmark(lambda: [token_jaccard(a, b, lowercase=True) for a, b in pairs])


# ── EDGAR BM25 benchmarks ───────────────────────────────────────────────────


@pytest.mark.benchmark(group="bm25_edgar")
def test_bm25_edgar_employment(benchmark, edgar_index):
    """BM25 search for employment terms across 200 contracts."""
    benchmark(
        edgar_index.query_bm25,
        ["employment", "termination", "severance"],
        10,
    )


@pytest.mark.benchmark(group="bm25_edgar")
def test_bm25_edgar_indemnification(benchmark, edgar_index):
    """BM25 search for indemnification clauses."""
    benchmark(
        edgar_index.query_bm25,
        ["indemnification", "liability", "damages", "breach"],
        10,
    )
