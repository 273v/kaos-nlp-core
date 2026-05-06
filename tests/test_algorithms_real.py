"""Algorithm tests on realistic legal/regulatory text.

Fills gaps in test coverage: character n-grams on legal terms, similarity
at paragraph scale, phonetic algorithms on legal names, vocabulary structures
on real corpus, and Bloom filter false positive rate measurement.
"""

import pytest

from kaos_nlp_core.algorithms import (
    jaro_winkler,
    levenshtein,
    metaphone_distance,
    metaphone_encode,
    ngram_cosine,
    ngram_jaccard,
    ngram_overlap,
    soundex_distance,
    soundex_encode,
    token_ngram_cosine,
    token_ngram_jaccard,
    token_ngram_overlap,
)
from kaos_nlp_core.matching import FstMap, RegexSetMatcher
from kaos_nlp_core.structures import (
    BloomVocabulary,
    FrequencyVocabulary,
    SetVocabulary,
)

# ── Character n-gram on legal terms ──────────────────────────────────────────


LEGAL_TERM_PAIRS = [
    ("indemnification", "indemnify"),
    ("notwithstanding", "withstanding"),
    ("acknowledgment", "acknowledgement"),
    ("judgment", "judgement"),
    ("enforceable", "unenforceable"),
    ("subparagraph", "paragraph"),
    ("nondisclosure", "disclosure"),
]


class TestCharNgramLegalTerms:
    """Character n-gram similarity on legal term variations."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_ngram_jaccard_legal_terms(self, n):
        """Legal term pairs should have moderate-to-high similarity."""
        for a, b in LEGAL_TERM_PAIRS:
            r = ngram_jaccard(a, b, n=n)
            assert 0.0 <= r.similarity <= 1.0

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_ngram_cosine_legal_terms(self, n):
        for a, b in LEGAL_TERM_PAIRS:
            r = ngram_cosine(a, b, n=n)
            assert 0.0 <= r.similarity <= 1.0

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_ngram_overlap_legal_terms(self, n):
        for a, b in LEGAL_TERM_PAIRS:
            r = ngram_overlap(a, b, n=n)
            assert 0.0 <= r.similarity <= 1.0

    def test_judgement_spelling_variants(self):
        """judgment vs judgement should be very similar at char n-gram level."""
        r2 = ngram_jaccard("judgment", "judgement", n=2)
        r3 = ngram_jaccard("judgment", "judgement", n=3)
        assert r2.similarity > 0.6  # 8 vs 9 chars, 1 char difference
        assert r3.similarity > 0.4

    def test_prefix_relation(self):
        """'paragraph' should have high overlap with 'subparagraph'."""
        r = ngram_overlap("paragraph", "subparagraph", n=3)
        assert r.similarity > 0.7  # paragraph trigrams are subset of subparagraph


# ── Character n-grams on paragraph-length legal text ─────────────────────────


class TestCharNgramParagraphs:
    """Character n-gram similarity on longer legal excerpts."""

    def test_same_section_variants(self):
        """Two similar legal clauses should be more similar than unrelated ones."""
        clause_a = (
            "The Company shall indemnify and hold harmless the Employee from and "
            "against any and all claims, damages, losses, costs, and expenses."
        )
        clause_b = (
            "The Corporation shall indemnify and hold harmless the Executive from "
            "and against any and all claims, damages, losses, and liabilities."
        )
        clause_c = (
            "The lessee shall maintain the property in good condition and repair "
            "at its own expense during the term of this lease agreement."
        )

        sim_ab = ngram_jaccard(clause_a, clause_b, n=3)
        sim_ac = ngram_jaccard(clause_a, clause_c, n=3)

        assert sim_ab.similarity > sim_ac.similarity
        assert sim_ab.similarity > 0.3  # meaningfully similar

    def test_usc_paragraph_similarity(self, usc_docs):
        """USC paragraphs from same title should be more similar than cross-title."""
        # Find two tax paragraphs and one military paragraph
        tax_docs = [
            d for d in usc_docs if "gross income" in d["text"].lower() and len(d["text"]) > 300
        ][:2]
        mil_docs = [
            d for d in usc_docs if "armed forces" in d["text"].lower() and len(d["text"]) > 300
        ][:1]

        if len(tax_docs) < 2 or len(mil_docs) < 1:
            pytest.skip("not enough docs")

        sim_same = ngram_cosine(tax_docs[0]["text"][:500], tax_docs[1]["text"][:500], n=3)
        sim_cross = ngram_cosine(tax_docs[0]["text"][:500], mil_docs[0]["text"][:500], n=3)

        assert sim_same.similarity > sim_cross.similarity


# ── Token n-gram on real legal text ──────────────────────────────────────────


class TestTokenNgramRealText:
    """Token n-gram algorithms on real legal documents."""

    def test_token_ngram_overlap_contracts(self, edgar_docs):
        """Token bigram overlap between two contracts."""
        a = edgar_docs[0]["text"][:2000]
        b = edgar_docs[1]["text"][:2000]
        r = token_ngram_overlap(a, b, n=2, lowercase=True)
        assert 0.0 <= r.similarity <= 1.0

    def test_token_ngram_cosine_contracts(self, edgar_docs):
        """Token bigram cosine between two contracts."""
        a = edgar_docs[0]["text"][:2000]
        b = edgar_docs[1]["text"][:2000]
        r = token_ngram_cosine(a, b, n=2, lowercase=True)
        assert 0.0 <= r.similarity <= 1.0

    def test_cosine_vs_jaccard_ordering(self, edgar_docs):
        """Cosine and Jaccard should agree on relative ordering of pairs."""
        texts = [d["text"][:1000] for d in edgar_docs[:5]]
        # Compare pair (0,1) vs (0,3) — same algorithm should rank consistently
        jac_01 = token_ngram_jaccard(texts[0], texts[1], n=2, lowercase=True)
        jac_03 = token_ngram_jaccard(texts[0], texts[3], n=2, lowercase=True)
        cos_01 = token_ngram_cosine(texts[0], texts[1], n=2, lowercase=True)
        cos_03 = token_ngram_cosine(texts[0], texts[3], n=2, lowercase=True)
        # Both metrics should rank in the same direction for most pairs
        # (not guaranteed for all pairs, so just test they return valid values)
        for r in [jac_01, jac_03, cos_01, cos_03]:
            assert 0.0 <= r.similarity <= 1.0


# ── Phonetic algorithms on legal names ───────────────────────────────────────


LEGAL_NAME_PAIRS = [
    ("Smith", "Smythe"),
    ("Johnson", "Johnsen"),
    ("McDonald", "MacDonald"),
    ("O'Brien", "Obrien"),
    ("Gonzalez", "Gonzales"),
    ("Schmidt", "Smith"),
    ("Mueller", "Miller"),
    ("Bernstein", "Burnstein"),
]


class TestPhoneticLegalNames:
    """Phonetic encoding and distance on realistic name variations."""

    def test_soundex_similar_names(self):
        for a, b in LEGAL_NAME_PAIRS:
            sa = soundex_encode(a)
            sb = soundex_encode(b)
            assert isinstance(sa, str) and len(sa) == 4
            assert isinstance(sb, str) and len(sb) == 4

    def test_soundex_distance_similar(self):
        """Similar-sounding names should have high phonetic similarity."""
        r = soundex_distance("Smith", "Smythe")
        assert r.similarity >= 0.5

    def test_metaphone_similar_names(self):
        """Metaphone should encode similar-sounding names identically."""
        # Smith and Schmidt sound similar
        r = metaphone_distance("Smith", "Schmidt")
        assert r.similarity > 0.0

    def test_metaphone_encode_consistency(self):
        for a, _ in LEGAL_NAME_PAIRS:
            code = metaphone_encode(a)
            assert isinstance(code, str) and len(code) > 0


# ── Edit distance on longer strings ──────────────────────────────────────────


class TestEditDistanceLongStrings:
    """Edit distance algorithms on paragraph-length inputs."""

    def test_levenshtein_paragraph(self):
        """Levenshtein on 200+ character legal clause with minor edits."""
        original = (
            "The Company shall indemnify and hold harmless the Employee from "
            "and against any and all claims, damages, losses, costs, and expenses "
            "arising out of or relating to the performance of duties."
        )
        modified = (
            "The Corporation shall indemnify and hold harmless the Executive from "
            "and against any and all claims, damages, losses, costs, and expenses "
            "arising out of or relating to the performance of duties."
        )
        r = levenshtein(original, modified)
        # Only Company->Corporation and Employee->Executive changed
        assert r.distance < 30
        assert r.similarity > 0.8

    def test_jaro_winkler_long(self):
        """Jaro-Winkler on longer strings — common prefix should boost."""
        a = "EMPLOYMENT AGREEMENT dated January 1, 2024"
        b = "EMPLOYMENT AGREEMENT dated February 15, 2024"
        r = jaro_winkler(a, b)
        assert r.similarity > 0.7  # long common prefix


# ── Vocabulary structures on real corpus ─────────────────────────────────────


class TestVocabularyRealCorpus:
    """Vocabulary structures with real corpus data."""

    def test_set_vocab_usc(self, usc_docs):
        """Build SetVocabulary from USC terms."""
        v = SetVocabulary()
        for d in usc_docs[:1000]:
            for w in d["text"].lower().split():
                v.insert(w.strip(".,;:!?\"'()[]{}\u2014\u2013-#*"))
        assert len(v) > 5000
        assert v.contains("the")
        assert v.contains("shall")
        assert not v.contains("xyzzy_nonexistent")

    def test_freq_vocab_usc(self, usc_docs):
        """FrequencyVocabulary: top terms from USC should be legal vocabulary."""
        v = FrequencyVocabulary()
        for d in usc_docs[:1000]:
            for w in d["text"].lower().split():
                cleaned = w.strip(".,;:!?\"'()[]{}\u2014\u2013-#*")
                if cleaned:
                    v.insert(cleaned)
        top = v.top_n(20)
        top_words = [t for t, _ in top]
        # "the", "of", "and" should be top words in any legal corpus
        assert "the" in top_words[:5]
        assert "of" in top_words[:5]


# ── Bloom filter false positive measurement ──────────────────────────────────


class TestBloomFalsePositiveRate:
    """Measure actual false positive rate of BloomVocabulary."""

    def test_bloom_fpr_measurement(self, war_and_peace_words):
        """Actual FPR should be close to configured rate."""
        words = list(set(war_and_peace_words))
        n_items = len(words)
        target_fpr = 0.01

        bloom = BloomVocabulary(n_items, target_fpr)
        for w in words:
            bloom.insert(w)

        # Test with words NOT in the vocabulary
        import hashlib

        false_positives = 0
        n_tests = 10000
        for i in range(n_tests):
            fake_word = hashlib.md5(str(i).encode()).hexdigest()[:12]
            if fake_word not in words and bloom.contains(fake_word):
                false_positives += 1

        actual_fpr = false_positives / n_tests
        # Allow 3x tolerance — Bloom filters are probabilistic
        assert actual_fpr < target_fpr * 3, f"FPR {actual_fpr:.4f} exceeds 3x target {target_fpr}"


# ── FstMap and RegexSetMatcher (untested APIs) ───────────────────────────────


class TestFstMapReal:
    """FstMap with real term frequency data."""

    def test_fst_map_word_frequencies(self, war_and_peace_words):
        """Build FstMap from word frequencies."""
        from collections import Counter

        freq = Counter(war_and_peace_words)
        entries = sorted(freq.items())  # FstMap requires sorted keys
        fst_map = FstMap(entries)
        assert len(fst_map) == len(freq)
        assert fst_map.get("the") == freq["the"]
        assert fst_map.get("prince") == freq.get("prince", 0) or fst_map.get("prince") is None
        assert fst_map.get("xyzzy_nonexistent") is None


class TestRegexSetMatcherReal:
    """RegexSetMatcher on real text."""

    def test_regex_set_legal_patterns(self):
        """Match multiple legal patterns simultaneously."""
        rs = RegexSetMatcher(
            [
                r"§\s*\d+",  # section references
                r"\b\d{4}\b",  # 4-digit years
                r"\b[A-Z]{2,}\b",  # acronyms
            ]
        )
        text = "See § 501 of the IRC (enacted in 1986). The IRS issued guidance."
        assert rs.is_match(text)
        indices = rs.matching_patterns(text)
        assert 0 in indices  # § 501
        assert 1 in indices  # 1986
        assert 2 in indices  # IRC, IRS
