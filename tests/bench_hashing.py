"""Benchmarks for fuzzy hashing: MinHash/LSH, CTPH (pytest-benchmark)."""

from pathlib import Path

import pytest

from kaos_nlp_core.hashing import (
    CTPH,
    MinHasher,
    MinHashIndex,
    TokenCTPH,
    find_duplicates,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def war_and_peace_text():
    fixture = Path(__file__).parent / "fixtures" / "war_and_peace.txt"
    try:
        return fixture.read_text()
    except FileNotFoundError:
        return "The quick brown fox jumps over the lazy dog. " * 10000


@pytest.fixture(scope="module")
def war_and_peace_words(war_and_peace_text):
    return war_and_peace_text.split()


@pytest.fixture(scope="module")
def hasher():
    return MinHasher(128, 42)


# =============================================================================
# MinHash benchmarks
# =============================================================================


@pytest.mark.benchmark(group="minhash")
def test_minhash_hash_set_100(benchmark, hasher):
    items = [f"item_{i}" for i in range(100)]
    benchmark(hasher.hash_set, items)


@pytest.mark.benchmark(group="minhash")
def test_minhash_hash_set_10000(benchmark, hasher):
    items = [f"item_{i}" for i in range(10_000)]
    benchmark(hasher.hash_set, items)


@pytest.mark.benchmark(group="minhash")
def test_minhash_char_shingles_war_peace(benchmark, hasher, war_and_peace_text):
    benchmark(hasher.hash_char_shingles, war_and_peace_text, 5)


@pytest.mark.benchmark(group="minhash")
def test_minhash_token_shingles_100(benchmark, hasher):
    tokens = [f"token_{i}" for i in range(100)]
    benchmark(hasher.hash_token_shingles, tokens, 2)


@pytest.mark.benchmark(group="minhash")
def test_minhash_jaccard(benchmark, hasher):
    sig1 = hasher.hash_set(["a", "b", "c", "d", "e"])
    sig2 = hasher.hash_set(["a", "b", "c", "f", "g"])
    benchmark(sig1.jaccard, sig2)


# =============================================================================
# LSH benchmarks
# =============================================================================


@pytest.mark.benchmark(group="lsh")
def test_lsh_build_1000(benchmark, hasher):
    sigs = []
    for i in range(1000):
        items = [f"term_{i}_{j}" for j in range(20)]
        sigs.append((i, hasher.hash_set(items)))

    def build():
        index = MinHashIndex.with_threshold(128, 0.5)
        for doc_id, sig in sigs:
            index.insert(doc_id, sig)
        return index

    benchmark(build)


@pytest.mark.benchmark(group="lsh")
def test_lsh_query_1000(benchmark, hasher):
    index = MinHashIndex.with_threshold(128, 0.5)
    for i in range(1000):
        sig = hasher.hash_set([f"term_{i}_{j}" for j in range(20)])
        index.insert(i, sig)

    query_sig = hasher.hash_set(["term_0_0", "term_0_1", "term_0_2"])
    benchmark(index.query_candidates, query_sig)


@pytest.mark.benchmark(group="lsh")
def test_lsh_query_above_threshold_1000(benchmark, hasher):
    index = MinHashIndex.with_threshold(128, 0.5)
    for i in range(1000):
        sig = hasher.hash_set([f"term_{i}_{j}" for j in range(20)])
        index.insert(i, sig)

    query_sig = hasher.hash_set(["term_0_0", "term_0_1", "term_0_2"])
    benchmark(index.query_above_threshold, query_sig, 0.3)


# =============================================================================
# find_duplicates benchmark
# =============================================================================


@pytest.mark.benchmark(group="dedup")
def test_find_duplicates_100(benchmark, hasher):
    docs = []
    for i in range(100):
        tokens = [f"word_{i % 10}_{j}" for j in range(30)]
        docs.append((i, tokens))
    benchmark(find_duplicates, hasher, docs, 2, 0.5)


# =============================================================================
# CTPH benchmarks
# =============================================================================


@pytest.mark.benchmark(group="ctph")
def test_ctph_short(benchmark):
    c = CTPH(64, 8, 4)
    data = b"The quick brown fox jumps over the lazy dog."
    benchmark(c.compute, data)


@pytest.mark.benchmark(group="ctph")
def test_ctph_war_peace(benchmark, war_and_peace_text):
    c = CTPH(64, 8, 4)
    benchmark(c.hash_str, war_and_peace_text)


@pytest.mark.benchmark(group="ctph")
def test_ctph_block_similarity(benchmark, war_and_peace_text):
    c = CTPH(64, 8, 4)
    base = war_and_peace_text[:10000]
    mid = len(base) // 2
    modified = base[: mid - 250] + "X" * 500 + base[mid + 250 :]
    d1 = c.hash_str(base)
    d2 = c.hash_str(modified)
    benchmark(d1.similarity, d2)


@pytest.mark.benchmark(group="ctph")
def test_ctph_piece_similarity(benchmark, war_and_peace_text):
    c = CTPH(64, 8, 4)
    base = war_and_peace_text[:10000]
    mid = len(base) // 2
    modified = base[: mid - 250] + "X" * 500 + base[mid + 250 :]
    d1 = c.hash_str(base)
    d2 = c.hash_str(modified)
    benchmark(d1.piece_similarity, d2)


# =============================================================================
# Token CTPH benchmarks
# =============================================================================


@pytest.mark.benchmark(group="token_ctph")
def test_token_ctph_100(benchmark):
    c = TokenCTPH(4, 8)
    tokens = list(range(100))
    benchmark(c.compute, tokens)


@pytest.mark.benchmark(group="token_ctph")
def test_token_ctph_10000(benchmark):
    c = TokenCTPH(4, 8)
    tokens = list(range(10_000))
    benchmark(c.compute, tokens)
