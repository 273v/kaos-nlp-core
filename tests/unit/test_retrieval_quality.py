"""Retrieval quality validation using known US Code provisions.

These tests verify that BM25, TF-IDF, substring search, regex, and FST fuzzy
search produce correct results against ground truth from the US Code corpus.

Each test uses known legal phrases/sections where we can independently verify
that the correct documents are retrieved.
"""

import json
from pathlib import Path

import pytest

from kaos_nlp_core.algorithms import ngram_jaccard
from kaos_nlp_core.matching import (
    FstSet,
    MultiPatternMatcher,
    RegexMatcher,
    substring_find_all,
)
from kaos_nlp_core.structures import InvertedIndex

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def tokenize_doc(text: str) -> list[str]:
    """Tokenize using the shared ICU-based tokenizer."""
    from conftest import DEFAULT_TOKENIZER

    return DEFAULT_TOKENIZER.tokenize_words(text)


@pytest.fixture(scope="module")
def usc_corpus():
    """Load USC corpus and build ground-truth lookups."""
    path = FIXTURE_DIR / "usc.jsonl"
    if not path.exists():
        pytest.skip("USC fixtures not downloaded")
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


@pytest.fixture(scope="module")
def usc_index(usc_corpus):
    """Pre-built inverted index over USC."""
    idx = InvertedIndex()
    for d in usc_corpus:
        idx.add_document(d["id"], tokenize_doc(d["text"]))
    return idx


@pytest.fixture(scope="module")
def usc_vocab(usc_corpus):
    """FST vocabulary from first 5000 USC docs."""
    vocab = set()
    for d in usc_corpus[:5000]:
        for w in tokenize_doc(d["text"]):
            vocab.add(w)
    return FstSet(sorted(vocab))


# ── BM25: Distinctive legal phrases (should have high precision) ─────────────


class TestBm25DistinctivePhrases:
    """BM25 retrieval of two-word phrases where both words are distinctive.

    When both query terms are relatively uncommon and strongly co-occur,
    BM25 should achieve near-perfect precision in the top results.
    """

    @pytest.mark.parametrize(
        "terms,phrase",
        [
            (["gross", "income"], "gross income"),
            (["habeas", "corpus"], "habeas corpus"),
            (["eminent", "domain"], "eminent domain"),
        ],
        ids=["gross_income", "habeas_corpus", "eminent_domain"],
    )
    def test_bm25_precision_at_20(self, usc_index, usc_corpus, terms, phrase):
        """Top-20 BM25 results should overwhelmingly contain the exact phrase."""
        results = usc_index.query_bm25(terms, top_k=20)
        truth_ids = {d["id"] for d in usc_corpus if phrase in d["text"].lower()}

        assert len(truth_ids) > 0, f"No docs contain '{phrase}'"
        assert len(results) == 20

        retrieved_ids = {r.doc_id for r in results}
        precision = len(retrieved_ids & truth_ids) / len(results)
        assert precision >= 0.9, f"Precision@20 for '{phrase}' = {precision:.0%}, expected >= 90%"

    def test_bm25_patent_infringement_both_words(self, usc_index, usc_corpus):
        """'patent infringement' — BM25 retrieves docs with both words, even in
        different order ('infringement of patent'). All top results should contain
        both query terms, even if not as an exact phrase."""
        results = usc_index.query_bm25(["patent", "infringement"], top_k=20)
        truth_ids = {
            d["id"]
            for d in usc_corpus
            if "patent" in d["text"].lower() and "infringement" in d["text"].lower()
        }
        retrieved_ids = {r.doc_id for r in results}
        precision = len(retrieved_ids & truth_ids) / len(results)
        assert precision >= 0.9, (
            f"Precision@20 (both words) for 'patent infringement' = {precision:.0%}"
        )

    @pytest.mark.parametrize(
        "terms,phrase",
        [
            (["gross", "income"], "gross income"),
            (["habeas", "corpus"], "habeas corpus"),
        ],
        ids=["gross_income", "habeas_corpus"],
    )
    def test_bm25_results_are_ranked(self, usc_index, terms, phrase):
        """Results should be strictly ordered by decreasing score."""
        results = usc_index.query_bm25(terms, top_k=20)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ── BM25: Common-word phrases (expected lower precision) ─────────────────────


class TestBm25CommonWordPhrases:
    """BM25 on phrases where constituent words are individually common.

    Bag-of-words BM25 doesn't enforce word adjacency, so queries like
    "due process" may retrieve docs that use "due" and "process" separately.
    Precision is expected to be lower but still above random.
    """

    @pytest.mark.parametrize(
        "terms,phrase,min_precision",
        [
            (["due", "process"], "due process", 0.3),
            (["equal", "protection"], "equal protection", 0.3),
            (["interstate", "commerce"], "interstate commerce", 0.5),
            (["freedom", "speech"], "freedom speech", 0.5),
        ],
        ids=["due_process", "equal_protection", "interstate_commerce", "freedom_speech"],
    )
    def test_bm25_above_random(self, usc_index, usc_corpus, terms, phrase, min_precision):
        """BM25 should still do better than random even for common-word queries."""
        results = usc_index.query_bm25(terms, top_k=20)
        # Check if both words appear (not necessarily as a phrase)
        truth_ids = {d["id"] for d in usc_corpus if all(t in d["text"].lower() for t in terms)}
        retrieved_ids = {r.doc_id for r in results}
        precision = len(retrieved_ids & truth_ids) / len(results) if results else 0
        assert precision >= min_precision, (
            f"Precision@20 for {terms} = {precision:.0%}, expected >= {min_precision:.0%}"
        )


# ── BM25 vs TF-IDF ranking comparison ───────────────────────────────────────


class TestRankingComparison:
    """Compare BM25 and TF-IDF rankings on the same queries."""

    def test_both_retrieve_relevant_docs(self, usc_index, usc_corpus):
        """Both BM25 and TF-IDF should retrieve docs containing query terms."""
        terms = ["bankruptcy", "creditor"]
        bm25 = usc_index.query_bm25(terms, top_k=10)
        tfidf = usc_index.query_tf_idf(terms, top_k=10, tf_weight="sublinear", idf_weight="smooth")

        truth = {d["id"] for d in usc_corpus if all(t in d["text"].lower() for t in terms)}

        bm25_precision = len({r.doc_id for r in bm25} & truth) / 10
        tfidf_precision = len({r.doc_id for r in tfidf} & truth) / 10

        assert bm25_precision >= 0.5
        assert tfidf_precision >= 0.5

    def test_rankings_overlap(self, usc_index):
        """BM25 and TF-IDF top-20 should have meaningful overlap."""
        terms = ["copyright", "license"]
        bm25_ids = {r.doc_id for r in usc_index.query_bm25(terms, top_k=20)}
        tfidf_ids = {
            r.doc_id
            for r in usc_index.query_tf_idf(
                terms, top_k=20, tf_weight="sublinear", idf_weight="smooth"
            )
        }
        overlap = len(bm25_ids & tfidf_ids)
        assert overlap >= 5, f"Only {overlap} docs overlap between BM25 and TF-IDF top-20"


# ── Substring search: exact phrase retrieval ─────────────────────────────────


class TestSubstringSearch:
    """Exact substring search should find all occurrences."""

    def test_501c3_count(self, usc_corpus):
        """Find all USC sections mentioning 501(c)(3)."""
        count = sum(1 for d in usc_corpus if substring_find_all(d["text"], "501(c)(3)"))
        assert count >= 250  # known: 284

    def test_united_states_frequency(self, usc_corpus):
        """'United States' should appear very frequently in federal law."""
        total = sum(len(substring_find_all(d["text"], "United States")) for d in usc_corpus[:500])
        assert total >= 500


# ── Regex: structural patterns in USC ────────────────────────────────────────


class TestRegexPatterns:
    """Regex extraction of structural patterns from USC text."""

    def test_section_references(self, usc_corpus):
        """USC text is full of cross-references like § 1234."""
        r = RegexMatcher(r"§\s*\d+[a-z]?(?:\(\w+\))*")
        total = sum(len(r.find_all(d["text"])) for d in usc_corpus[:1000])
        assert total >= 5000  # known: ~12K in first 1000 docs

    def test_title_headers(self, usc_corpus):
        """USC sections have markdown headers like ### §123."""
        r = RegexMatcher(r"###\s+\*?\*?§\s*\d+")
        found = sum(1 for d in usc_corpus[:1000] if r.is_match(d["text"]))
        assert found >= 500


# ── FST: fuzzy spelling correction ──────────────────────────────────────────


class TestFstSpellingCorrection:
    """FST fuzzy search should correct common legal misspellings."""

    @pytest.mark.parametrize(
        "misspelling,expected_correction,max_dist",
        [
            ("bankrupcy", "bankruptcy", 1),
            ("infringment", "infringement", 1),
            ("ammendment", "amendment", 2),
            ("juristiction", "jurisdiction", 2),
            ("negligance", "negligence", 1),
        ],
        ids=["bankruptcy", "infringement", "amendment", "jurisdiction", "negligence"],
    )
    def test_fuzzy_correction(self, usc_vocab, misspelling, expected_correction, max_dist):
        """Fuzzy search should return the correct spelling."""
        results = usc_vocab.fuzzy_search(misspelling, max_dist)
        corrections = [r.key for r in results]
        assert expected_correction in corrections, (
            f"'{expected_correction}' not in fuzzy results for '{misspelling}': {corrections[:10]}"
        )


# ── Multi-pattern: legal boilerplate detection ──────────────────────────────


class TestMultiPatternLegal:
    """Multi-pattern matching for common legal language."""

    def test_legal_boilerplate_terms(self, usc_corpus):
        """Common legal terms should appear frequently in USC."""
        m = MultiPatternMatcher(
            ["herein", "thereof", "pursuant to", "notwithstanding", "subsection"],
            case_insensitive=True,
        )
        total = sum(m.count(d["text"]) for d in usc_corpus[:500])
        assert total >= 1000


# ── N-gram similarity: same-domain vs cross-domain ─────────────────────────


class TestDomainSimilarity:
    """Documents from the same legal domain should be more similar."""

    def test_same_domain_more_similar(self, usc_corpus):
        """Two tax sections should be more similar than a tax + military section."""
        tax1 = next(
            d for d in usc_corpus if "gross income" in d["text"].lower() and len(d["text"]) > 500
        )
        tax2 = next(
            d
            for d in usc_corpus
            if "taxable income" in d["text"].lower()
            and d["id"] != tax1["id"]
            and len(d["text"]) > 500
        )
        mil = next(
            d for d in usc_corpus if "armed forces" in d["text"].lower() and len(d["text"]) > 500
        )

        sim_same = ngram_jaccard(tax1["text"][:1000], tax2["text"][:1000], n=3)
        sim_cross = ngram_jaccard(tax1["text"][:1000], mil["text"][:1000], n=3)

        assert sim_same.similarity > sim_cross.similarity, (
            f"Same-domain similarity ({sim_same.similarity:.4f}) should exceed "
            f"cross-domain ({sim_cross.similarity:.4f})"
        )
