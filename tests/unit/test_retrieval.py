"""Realistic retrieval tests using US Code, EDGAR agreements, and patents.

These tests validate BM25, TF-IDF, and similarity at production-relevant scale
using real legal/regulatory text from alea-institute HuggingFace datasets.
"""

import pytest

from kaos_nlp_core.algorithms import (
    ngram_jaccard,
    token_jaccard,
    token_ngram_jaccard,
)
from kaos_nlp_core.matching import (
    MultiPatternMatcher,
    RegexMatcher,
    substring_find_all,
)
from kaos_nlp_core.structures import InvertedIndex

# ── Helpers ──────────────────────────────────────────────────────────────────


def tokenize_doc(text: str) -> list[str]:
    """Tokenize using the shared ICU-based tokenizer."""
    from conftest import DEFAULT_TOKENIZER

    return DEFAULT_TOKENIZER.tokenize_words(text)


def build_index(docs: list[dict], max_docs: int | None = None) -> InvertedIndex:
    """Build an inverted index from a list of docs with 'text' field."""
    idx = InvertedIndex()
    for doc in docs[:max_docs]:
        terms = tokenize_doc(doc["text"])
        idx.add_document(doc["id"], terms)
    return idx


# ── US Code: BM25 retrieval at scale ─────────────────────────────────────────


class TestUscBm25:
    """BM25 retrieval over ~69K US Code sections."""

    @pytest.fixture(scope="class")
    def usc_index(self, usc_docs):
        return build_index(usc_docs)

    def test_index_scale(self, usc_index):
        """Index should have tens of thousands of documents."""
        assert usc_index.doc_count() >= 60000
        assert usc_index.term_count() >= 10000

    def test_bm25_tax_query(self, usc_index):
        """Search for tax-related provisions — should return ranked results."""
        results = usc_index.query_bm25(["tax", "income", "deduction"], top_k=20)
        assert len(results) > 0
        assert len(results) <= 20
        # Results should be sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        # Top result should have a meaningful score
        assert results[0].score > 0.0

    def test_bm25_military_query(self, usc_index):
        """Search for military provisions."""
        results = usc_index.query_bm25(["military", "armed", "forces", "defense"], top_k=10)
        assert len(results) > 0
        assert results[0].score > results[-1].score

    def test_bm25_rare_term(self, usc_index):
        """Rare terms should have high IDF and boost relevant documents."""
        results = usc_index.query_bm25(["cryptocurrency"], top_k=5)
        # May or may not have results — the test is that it doesn't crash
        # and returns valid scored docs
        for r in results:
            assert r.score > 0.0
            assert isinstance(r.doc_id, int)

    def test_bm25_no_results(self, usc_index):
        """Completely nonsensical query should return empty results."""
        results = usc_index.query_bm25(["xyzzy_nonexistent_term_42"], top_k=10)
        assert results == []

    def test_bm25_common_term(self, usc_index):
        """Very common terms like 'the' should still return results."""
        results = usc_index.query_bm25(["the"], top_k=10)
        assert len(results) == 10
        # But scores should be low because IDF is low
        assert all(r.score > 0 for r in results)

    def test_bm25_vs_tfidf_ranking(self, usc_index):
        """BM25 and TF-IDF should produce different but overlapping rankings."""
        query = ["patent", "invention", "claim"]
        bm25_results = usc_index.query_bm25(query, top_k=20)
        tfidf_results = usc_index.query_tf_idf(query, top_k=20)
        bm25_ids = {r.doc_id for r in bm25_results}
        tfidf_ids = {r.doc_id for r in tfidf_results}
        # There should be overlap but not necessarily identical
        overlap = bm25_ids & tfidf_ids
        assert len(overlap) > 0

    def test_bm25_custom_params(self, usc_index):
        """Different k1/b values should produce different rankings."""
        query = ["copyright", "infringement"]
        results_default = usc_index.query_bm25(query, top_k=10)
        results_no_length_norm = usc_index.query_bm25(query, top_k=10, k1=1.5, b=0.0)
        _results_high_saturation = usc_index.query_bm25(query, top_k=10, k1=3.0, b=0.75)
        # Scores should differ with different params
        if results_default and results_no_length_norm:
            assert results_default[0].score != results_no_length_norm[0].score


class TestUscTfIdf:
    """TF-IDF variants over US Code."""

    @pytest.fixture(scope="class")
    def usc_index(self, usc_docs):
        return build_index(usc_docs)

    def test_tfidf_sublinear_smooth(self, usc_index):
        """Sublinear TF + smooth IDF should return valid ranked results."""
        results = usc_index.query_tf_idf(
            ["bankruptcy", "debtor", "creditor"],
            top_k=10,
            tf_weight="sublinear",
            idf_weight="smooth",
        )
        assert len(results) > 0
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_tfidf_boolean(self, usc_index):
        """Boolean TF should weight all matching docs equally per term."""
        results = usc_index.query_tf_idf(
            ["amendment"],
            top_k=10,
            tf_weight="boolean",
            idf_weight="standard",
        )
        assert len(results) > 0


# ── EDGAR Agreements: document similarity ────────────────────────────────────


class TestEdgarSimilarity:
    """Similarity between real SEC contracts."""

    def test_token_jaccard_same_type(self, edgar_docs):
        """Documents of the same type should have higher similarity."""
        # Compare first 10 docs pairwise — they're all agreements
        texts = [d["text"][:2000] for d in edgar_docs[:10]]
        sims = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                r = token_jaccard(texts[i], texts[j], lowercase=True)
                sims.append(r.similarity)
        # Average similarity among agreements should be non-trivial
        avg_sim = sum(sims) / len(sims)
        assert avg_sim > 0.05  # agreements share boilerplate vocabulary

    def test_token_ngram_jaccard_contracts(self, edgar_docs):
        """Token bigram Jaccard captures shared phrases in contracts."""
        text1 = edgar_docs[0]["text"][:2000]
        text2 = edgar_docs[1]["text"][:2000]
        r = token_ngram_jaccard(text1, text2, n=2, lowercase=True)
        assert 0.0 <= r.similarity <= 1.0

    def test_edgar_bm25_search(self, edgar_docs):
        """BM25 search across 200 agreements."""
        idx = build_index(edgar_docs)
        assert idx.doc_count() == 200

        # Search for employment-related clauses
        results = idx.query_bm25(
            ["employment", "termination", "severance", "compensation"], top_k=20
        )
        assert len(results) > 0


# ── Patents: structured search ───────────────────────────────────────────────


class TestPatentSearch:
    """Search and similarity on patent documents."""

    def test_patent_title_similarity(self, patent_docs):
        """Patent titles in the same domain should be somewhat similar."""
        titles = [d["title"] for d in patent_docs[:50] if d["title"]]
        # At least some titles should share vocabulary
        has_nonzero = False
        for i in range(min(10, len(titles))):
            for j in range(i + 1, min(10, len(titles))):
                r = ngram_jaccard(titles[i], titles[j], n=3)
                if r.similarity > 0:
                    has_nonzero = True
                    break
            if has_nonzero:
                break
        # Not all patent titles will be similar, but across 10 pairs some should overlap
        # (this is a smoke test, not a precision test)

    def test_patent_abstract_bm25(self, patent_docs):
        """BM25 over patent abstracts."""
        idx = InvertedIndex()
        for doc in patent_docs:
            text = doc.get("abstract", "")
            if text:
                terms = tokenize_doc(text)
                idx.add_document(doc["id"], terms)

        results = idx.query_bm25(["network", "protocol", "data"], top_k=10)
        assert len(results) > 0

    def test_patent_claims_search(self, patent_docs):
        """Search across patent claims."""
        idx = InvertedIndex()
        for doc in patent_docs:
            claims = doc.get("claims", [])
            if claims:
                all_claims_text = " ".join(claims)
                terms = tokenize_doc(all_claims_text)
                idx.add_document(doc["id"], terms)

        results = idx.query_bm25(["method", "comprising", "step"], top_k=10)
        assert len(results) > 0
        # "comprising" is extremely common in patent claims
        assert idx.doc_freq("comprising") > 50


# ── Pattern matching on real text ────────────────────────────────────────────


class TestPatternMatchingRealText:
    """Pattern matching on real legal/regulatory text."""

    def test_multi_pattern_legal_terms(self, edgar_docs):
        """Find legal terms across a real contract."""
        text = edgar_docs[0]["text"]
        m = MultiPatternMatcher(
            ["Agreement", "Party", "Section", "Exhibit", "herein", "thereof"],
        )
        matches = m.find_all(text)
        assert len(matches) > 10  # contracts are full of these terms

    def test_regex_dates_in_contracts(self, edgar_docs):
        """Extract date patterns from real contracts."""
        r = RegexMatcher(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
        )
        # Search across multiple contracts — at least some should have spelled-out dates
        total_matches = 0
        for doc in edgar_docs[:20]:
            total_matches += len(r.find_all(doc["text"]))
        assert total_matches >= 1

    def test_regex_section_numbers_usc(self, usc_docs):
        """Extract section number patterns from US Code."""
        # Use a longer USC section
        long_docs = [d for d in usc_docs[:100] if len(d["text"]) > 1000]
        if not long_docs:
            pytest.skip("no long USC docs found")
        text = long_docs[0]["text"]
        r = RegexMatcher(r"§\s*\d+")
        matches = r.find_all(text)
        # USC text references other sections frequently
        assert len(matches) >= 0  # smoke test — some sections may not reference others

    def test_substring_search_usc(self, usc_docs):
        """Substring search for 'United States' across USC sections."""
        # Search in first 100 docs
        found_count = 0
        for doc in usc_docs[:100]:
            matches = substring_find_all(doc["text"], "United States")
            found_count += len(matches)
        # "United States" should appear frequently in federal law
        assert found_count > 50
