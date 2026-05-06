"""Benchmarks for data structures: vocabularies, inverted index (pytest-benchmark)."""

import pytest

from kaos_nlp_core.structures import (
    BloomVocabulary,
    FrequencyVocabulary,
    IndexedVocabulary,
    InvertedIndex,
    SetVocabulary,
)

# --- Vocabulary insert ---


@pytest.mark.benchmark(group="vocabulary_insert")
def test_set_vocab_insert(benchmark, war_and_peace_words):
    def run():
        v = SetVocabulary()
        for w in war_and_peace_words:
            v.insert(w)
        return v.contains("prince")

    benchmark(run)


@pytest.mark.benchmark(group="vocabulary_insert")
def test_freq_vocab_insert(benchmark, war_and_peace_words):
    def run():
        v = FrequencyVocabulary()
        for w in war_and_peace_words:
            v.insert(w)
        return v.contains("prince")

    benchmark(run)


@pytest.mark.benchmark(group="vocabulary_insert")
def test_indexed_vocab_insert(benchmark, war_and_peace_words):
    def run():
        v = IndexedVocabulary()
        for w in war_and_peace_words:
            v.insert(w)
        return v.contains("prince")

    benchmark(run)


@pytest.mark.benchmark(group="vocabulary_insert")
def test_bloom_vocab_insert(benchmark, war_and_peace_words):
    def run():
        v = BloomVocabulary(50000, 0.01)
        for w in war_and_peace_words:
            v.insert(w)
        return v.contains("prince")

    benchmark(run)


# --- Vocabulary lookup ---


LOOKUP_QUERIES = [
    "prince",
    "war",
    "peace",
    "love",
    "death",
    "soldier",
    "napoleon",
    "nonexistent_word_xyz",
    "another_missing_word",
    "zzzzzzz",
]


@pytest.fixture(scope="module")
def built_vocabs(war_and_peace_words):
    """Pre-built vocabularies for lookup benchmarks."""
    set_v = SetVocabulary()
    freq_v = FrequencyVocabulary()
    idx_v = IndexedVocabulary()
    bloom_v = BloomVocabulary(50000, 0.01)
    for w in war_and_peace_words:
        set_v.insert(w)
        freq_v.insert(w)
        idx_v.insert(w)
        bloom_v.insert(w)
    return {"set": set_v, "freq": freq_v, "indexed": idx_v, "bloom": bloom_v}


@pytest.mark.benchmark(group="vocabulary_lookup")
def test_set_vocab_lookup(benchmark, built_vocabs):
    v = built_vocabs["set"]
    benchmark(lambda: [v.contains(q) for q in LOOKUP_QUERIES])


@pytest.mark.benchmark(group="vocabulary_lookup")
def test_freq_vocab_lookup(benchmark, built_vocabs):
    v = built_vocabs["freq"]
    benchmark(lambda: [v.contains(q) for q in LOOKUP_QUERIES])


@pytest.mark.benchmark(group="vocabulary_lookup")
def test_indexed_vocab_lookup(benchmark, built_vocabs):
    v = built_vocabs["indexed"]
    benchmark(lambda: [v.contains(q) for q in LOOKUP_QUERIES])


@pytest.mark.benchmark(group="vocabulary_lookup")
def test_bloom_vocab_lookup(benchmark, built_vocabs):
    v = built_vocabs["bloom"]
    benchmark(lambda: [v.contains(q) for q in LOOKUP_QUERIES])


# --- FrequencyVocabulary top_n ---


@pytest.mark.benchmark(group="vocabulary_top_n")
@pytest.mark.parametrize("n", [10, 100, 1000], ids=lambda n: f"top_{n}")
def test_freq_vocab_top_n(benchmark, built_vocabs, n):
    v = built_vocabs["freq"]
    benchmark(v.top_n, n)


# --- Inverted Index ---


@pytest.fixture(scope="module")
def inverted_index_docs(war_and_peace_text):
    """Split War and Peace into paragraph-sized documents."""
    return [p.split() for p in war_and_peace_text.split("\n\n") if len(p) > 100]


@pytest.fixture(scope="module")
def built_index(inverted_index_docs):
    """Pre-built inverted index for query benchmarks."""
    idx = InvertedIndex()
    for i, doc in enumerate(inverted_index_docs):
        idx.add_document(i, doc)
    return idx


@pytest.mark.benchmark(group="inverted_index")
def test_inverted_index_build(benchmark, inverted_index_docs):
    def run():
        idx = InvertedIndex()
        for i, doc in enumerate(inverted_index_docs):
            idx.add_document(i, doc)
        return idx.term_count()

    benchmark(run)


@pytest.mark.benchmark(group="inverted_index")
def test_inverted_index_query_and_2(benchmark, built_index):
    benchmark(built_index.query_and, ["Prince", "war"])


@pytest.mark.benchmark(group="inverted_index")
def test_inverted_index_query_or_2(benchmark, built_index):
    benchmark(built_index.query_or, ["Prince", "war"])


@pytest.mark.benchmark(group="inverted_index")
def test_inverted_index_query_and_5(benchmark, built_index):
    benchmark(built_index.query_and, ["the", "and", "was", "Prince", "war"])


@pytest.mark.benchmark(group="inverted_index")
def test_inverted_index_tf_idf(benchmark, built_index):
    terms = ["Prince", "war", "peace", "the", "and"]
    benchmark(lambda: [built_index.tf_idf(t, 0) for t in terms])


# --- BM25 ---


@pytest.mark.benchmark(group="bm25")
def test_bm25_score_2_terms(benchmark, built_index):
    benchmark(built_index.score_bm25, ["Prince", "war"], 0)


@pytest.mark.benchmark(group="bm25")
def test_bm25_score_5_terms(benchmark, built_index):
    benchmark(
        built_index.score_bm25,
        ["Prince", "war", "peace", "love", "death"],
        0,
    )


@pytest.mark.benchmark(group="bm25")
def test_bm25_query_top10_2_terms(benchmark, built_index):
    benchmark(built_index.query_bm25, ["Prince", "war"], 10)


@pytest.mark.benchmark(group="bm25")
def test_bm25_query_top10_5_terms(benchmark, built_index):
    benchmark(
        built_index.query_bm25,
        ["Prince", "war", "peace", "love", "death"],
        10,
    )


@pytest.mark.benchmark(group="bm25")
def test_bm25_query_top100_5_terms(benchmark, built_index):
    benchmark(
        built_index.query_bm25,
        ["Prince", "war", "peace", "love", "death"],
        100,
    )


# --- TF-IDF variants ---


@pytest.mark.benchmark(group="tf_idf_variants")
def test_tf_idf_sublinear_smooth(benchmark, built_index):
    benchmark(
        built_index.score_tf_idf,
        ["Prince", "war", "peace"],
        0,
        "sublinear",
        "smooth",
    )


@pytest.mark.benchmark(group="tf_idf_variants")
def test_tf_idf_query_top10(benchmark, built_index):
    benchmark(
        built_index.query_tf_idf,
        ["Prince", "war", "peace"],
        10,
        "sublinear",
        "smooth",
    )
