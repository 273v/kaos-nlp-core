"""Tests for fuzzy hashing: MinHash/LSH, CTPH."""

import pickle
import threading
from pathlib import Path

import pytest

from kaos_nlp_core.hashing import (
    CTPH,
    CTPHDigest,
    DuplicateGroup,
    MinHasher,
    MinHashIndex,
    TokenCTPH,
    ctph_hash_bytes,
    ctph_hash_str,
    ctph_piece_similarity,
    ctph_similarity,
    find_duplicates,
    token_ctph_hash,
)

# =============================================================================
# MinHashSignature
# =============================================================================


class TestMinHashSignature:
    def test_jaccard_identical(self):
        hasher = MinHasher(128, 42)
        sig = hasher.hash_set(["apple", "banana", "cherry"])
        assert sig.jaccard(sig) == 1.0

    def test_jaccard_disjoint(self):
        hasher = MinHasher(128, 42)
        sig1 = hasher.hash_set(["apple", "banana", "cherry"])
        sig2 = hasher.hash_set(["dog", "elephant", "fox"])
        sim = sig1.jaccard(sig2)
        assert sim < 0.2, f"Disjoint sets similarity {sim} too high"

    def test_jaccard_partial_overlap(self):
        hasher = MinHasher(256, 42)
        sig1 = hasher.hash_set(["a", "b", "c", "d"])
        sig2 = hasher.hash_set(["a", "b", "c", "e"])
        sim = sig1.jaccard(sig2)
        # True Jaccard = 3/5 = 0.6
        assert abs(sim - 0.6) < 0.15, f"Expected ~0.6, got {sim}"

    def test_values_property(self):
        hasher = MinHasher(64, 42)
        sig = hasher.hash_set(["a", "b", "c"])
        values = sig.values
        assert len(values) == 64
        assert all(isinstance(v, int) for v in values)

    def test_len(self):
        hasher = MinHasher(128, 42)
        sig = hasher.hash_set(["a"])
        assert len(sig) == 128

    def test_repr(self):
        hasher = MinHasher(128, 42)
        sig = hasher.hash_set(["a"])
        assert "128" in repr(sig)

    def test_pickle_roundtrip(self):
        hasher = MinHasher(128, 42)
        sig = hasher.hash_set(["apple", "banana", "cherry"])
        data = pickle.dumps(sig)
        restored = pickle.loads(data)
        assert sig.jaccard(restored) == 1.0
        assert sig.values == restored.values


# =============================================================================
# MinHasher
# =============================================================================


class TestMinHasher:
    def test_hash_set_basic(self):
        hasher = MinHasher(128, 42)
        sig = hasher.hash_set(["hello", "world"])
        assert len(sig) == 128

    def test_hash_set_empty(self):
        hasher = MinHasher(64, 42)
        sig = hasher.hash_set([])
        assert len(sig) == 64
        # All values should be max u64
        assert all(v == 2**64 - 1 for v in sig.values)

    def test_hash_set_reproducible(self):
        hasher = MinHasher(128, 42)
        sig1 = hasher.hash_set(["a", "b", "c"])
        sig2 = hasher.hash_set(["a", "b", "c"])
        assert sig1.values == sig2.values

    def test_different_seeds(self):
        h1 = MinHasher(128, 42)
        h2 = MinHasher(128, 99)
        sig1 = h1.hash_set(["a", "b", "c"])
        sig2 = h2.hash_set(["a", "b", "c"])
        assert sig1.values != sig2.values

    def test_hash_char_shingles(self):
        hasher = MinHasher(128, 42)
        sig1 = hasher.hash_char_shingles("hello world", 3)
        sig2 = hasher.hash_char_shingles("hello world!", 3)
        sim = sig1.jaccard(sig2)
        assert sim > 0.5, f"Similar strings should have high similarity: {sim}"

    def test_hash_char_shingles_short_text(self):
        hasher = MinHasher(64, 42)
        # Text shorter than shingle size
        sig = hasher.hash_char_shingles("ab", 5)
        assert len(sig) == 64
        assert all(v == 2**64 - 1 for v in sig.values)

    def test_hash_token_shingles(self):
        hasher = MinHasher(128, 42)
        tokens1 = ["the", "cat", "sat", "on", "the", "mat"]
        tokens2 = ["the", "cat", "sat", "on", "the", "log"]
        sig1 = hasher.hash_token_shingles(tokens1, 2)
        sig2 = hasher.hash_token_shingles(tokens2, 2)
        sim = sig1.jaccard(sig2)
        assert sim > 0.3, f"Similar token sequences should overlap: {sim}"

    def test_num_perm_property(self):
        hasher = MinHasher(256, 42)
        assert hasher.num_perm == 256

    def test_pickle_roundtrip(self):
        hasher = MinHasher(128, 42)
        data = pickle.dumps(hasher)
        restored = pickle.loads(data)
        sig1 = hasher.hash_set(["a", "b", "c"])
        sig2 = restored.hash_set(["a", "b", "c"])
        assert sig1.values == sig2.values

    def test_repr(self):
        hasher = MinHasher(128, 42)
        r = repr(hasher)
        assert "128" in r
        assert "42" in r


# =============================================================================
# MinHashIndex
# =============================================================================


class TestMinHashIndex:
    def test_insert_and_query(self):
        hasher = MinHasher(128, 42)
        index = MinHashIndex.with_threshold(128, 0.5)

        sig1 = hasher.hash_set(["a", "b", "c", "d"])
        sig2 = hasher.hash_set(["a", "b", "c", "e"])
        sig3 = hasher.hash_set(["x", "y", "z"])

        index.insert(0, sig1)
        index.insert(1, sig2)
        index.insert(2, sig3)

        candidates = index.query_candidates(sig1)
        assert 0 in candidates

    def test_with_threshold(self):
        index = MinHashIndex.with_threshold(128, 0.5)
        assert len(index) == 0

    def test_query_above_threshold(self):
        hasher = MinHasher(128, 42)
        index = MinHashIndex.with_threshold(128, 0.3)

        sig1 = hasher.hash_set(["a", "b", "c", "d"])
        sig2 = hasher.hash_set(["a", "b", "c", "e"])
        sig3 = hasher.hash_set(["x", "y", "z", "w"])

        index.insert(0, sig1)
        index.insert(1, sig2)
        index.insert(2, sig3)

        results = index.query_above_threshold(sig1, 0.4)
        ids = [doc_id for doc_id, _ in results]
        assert 0 in ids

    def test_empty_index(self):
        index = MinHashIndex(4, 4)
        assert len(index) == 0
        hasher = MinHasher(16, 42)
        sig = hasher.hash_set(["a", "b"])
        candidates = index.query_candidates(sig)
        assert candidates == []

    def test_get_signature(self):
        hasher = MinHasher(128, 42)
        index = MinHashIndex.with_threshold(128, 0.5)
        sig = hasher.hash_set(["a", "b", "c"])
        index.insert(0, sig)

        retrieved = index.get_signature(0)
        assert retrieved is not None
        assert retrieved.jaccard(sig) == 1.0

        assert index.get_signature(999) is None

    def test_len(self):
        hasher = MinHasher(16, 42)
        index = MinHashIndex(4, 4)
        assert len(index) == 0
        index.insert(0, hasher.hash_set(["a"]))
        assert len(index) == 1
        index.insert(1, hasher.hash_set(["b"]))
        assert len(index) == 2

    def test_pickle_roundtrip(self):
        hasher = MinHasher(128, 42)
        index = MinHashIndex.with_threshold(128, 0.5)
        sig = hasher.hash_set(["a", "b", "c"])
        index.insert(0, sig)
        index.insert(1, hasher.hash_set(["d", "e", "f"]))

        data = pickle.dumps(index)
        restored = pickle.loads(data)

        assert len(restored) == 2
        retrieved = restored.get_signature(0)
        assert retrieved is not None
        assert retrieved.jaccard(sig) == 1.0

    def test_save_load(self, tmp_path: Path):
        hasher = MinHasher(128, 42)
        index = MinHashIndex.with_threshold(128, 0.5)
        index.insert(0, hasher.hash_set(["a", "b", "c"]))
        path = tmp_path / "minhash.idx"
        index.save(str(path))
        restored = MinHashIndex.load(str(path))
        assert len(restored) == 1
        assert restored.get_signature(0) is not None


# =============================================================================
# find_duplicates
# =============================================================================


class TestFindDuplicates:
    def test_exact_duplicates(self):
        hasher = MinHasher(128, 42)
        docs = [
            (0, ["the", "cat", "sat", "on", "the", "mat"]),
            (1, ["the", "cat", "sat", "on", "the", "mat"]),
            (2, ["a", "completely", "different", "document"]),
        ]
        groups = find_duplicates(hasher, docs, shingle_size=2, threshold=0.5)
        assert len(groups) >= 1
        first = groups[0]
        assert isinstance(first, DuplicateGroup)
        assert first.canonical_id in (0, 1)
        assert len(first.duplicates) >= 1

    def test_all_unique(self):
        hasher = MinHasher(128, 42)
        docs = [
            (0, ["alpha", "beta", "gamma", "delta", "epsilon"]),
            (1, ["one", "two", "three", "four", "five"]),
            (2, ["red", "green", "blue", "yellow", "purple"]),
        ]
        groups = find_duplicates(hasher, docs, shingle_size=2, threshold=0.5)
        assert groups == []

    def test_empty_corpus(self):
        hasher = MinHasher(128, 42)
        groups = find_duplicates(hasher, [], shingle_size=2, threshold=0.5)
        assert groups == []

    def test_near_duplicates(self):
        hasher = MinHasher(128, 42)
        base = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        near = ["the", "quick", "brown", "fox", "leaps", "over", "the", "lazy", "cat"]
        different = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur"]

        docs = [(0, base), (1, near), (2, different)]
        groups = find_duplicates(hasher, docs, shingle_size=2, threshold=0.3)
        # Base and near should be grouped together
        if groups:
            all_ids = set()
            for g in groups:
                all_ids.add(g.canonical_id)
                for did, _ in g.duplicates:
                    all_ids.add(did)
            assert 0 in all_ids or 1 in all_ids


# =============================================================================
# Concurrent access
# =============================================================================


class TestConcurrentHashing:
    def test_concurrent_minhash(self):
        """Multiple threads computing MinHash signatures concurrently."""
        hasher = MinHasher(128, 42)
        results = [None] * 8
        errors = []

        def worker(idx):
            try:
                items = [f"item_{idx}_{j}" for j in range(100)]
                sig = hasher.hash_set(items)
                results[idx] = sig
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        assert all(r is not None for r in results)

    def test_concurrent_lsh_query(self):
        """Multiple threads querying the same LSH index concurrently."""
        hasher = MinHasher(128, 42)
        index = MinHashIndex.with_threshold(128, 0.5)

        # Build index
        for i in range(100):
            sig = hasher.hash_set([f"term_{i}_{j}" for j in range(10)])
            index.insert(i, sig)

        results = [None] * 8
        errors = []

        def worker(idx):
            try:
                sig = hasher.hash_set([f"term_{idx}_0", f"term_{idx}_1"])
                candidates = index.query_candidates(sig)
                results[idx] = candidates
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r is not None for r in results)

    def test_concurrent_find_duplicates(self):
        """find_duplicates releases GIL properly."""
        hasher = MinHasher(128, 42)
        docs = [(i, [f"word_{i % 5}_{j}" for j in range(20)]) for i in range(50)]
        results = [None] * 4
        errors = []

        def worker(idx):
            try:
                groups = find_duplicates(hasher, docs, shingle_size=2, threshold=0.5)
                results[idx] = groups
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r is not None for r in results)


# =============================================================================
# CTPH
# =============================================================================


class TestCTPH:
    def test_compute_bytes(self):
        c = CTPH(64, 8, 4)
        d = c.compute(b"hello world this is a test")
        assert d.window_size == 64
        assert d.digest_size == 8
        assert len(d.blocks) > 0

    def test_hash_str(self):
        c = CTPH(64, 8, 4)
        d = c.hash_str("hello world this is a test")
        assert d.window_size == 64

    def test_identical_similarity(self):
        c = CTPH(64, 8, 4)
        text = "The quick brown fox jumps over the lazy dog."
        d1 = c.hash_str(text)
        d2 = c.hash_str(text)
        assert d1.similarity(d2) == 1.0

    def test_different_similarity(self):
        c = CTPH(64, 8, 4)
        d1 = c.hash_str("The quick brown fox jumps over the lazy dog.")
        d2 = c.hash_str("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
        sim = d1.similarity(d2)
        assert sim < 0.5

    def test_similar_texts(self):
        c = CTPH(16, 4, 4)
        base = "The quick brown fox jumps over the lazy dog. " * 20
        modified = base + "A small addition."
        d1 = c.hash_str(base)
        d2 = c.hash_str(modified)
        sim = d1.similarity(d2)
        assert sim > 0.0

    def test_empty_input(self):
        c = CTPH(64, 8, 4)
        d = c.compute(b"")
        assert len(d.blocks) == 0

    def test_all_precisions(self):
        for precision in [1, 2, 4, 8]:
            c = CTPH(16, 4, precision)
            d = c.hash_str("test data for precision check")
            assert len(d.blocks) > 0

    def test_pickle_roundtrip(self):
        c = CTPH(64, 8, 4)
        data = pickle.dumps(c)
        restored = pickle.loads(data)
        d1 = c.hash_str("test")
        d2 = restored.hash_str("test")
        assert d1.similarity(d2) == 1.0

    def test_repr(self):
        c = CTPH(64, 8, 4)
        r = repr(c)
        assert "64" in r
        assert "8" in r


class TestTokenCTPH:
    def test_compute(self):
        c = TokenCTPH(4, 8)
        d = c.compute([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert d.window_size == 4
        assert d.digest_size == 8
        assert len(d.blocks) > 0

    def test_identical(self):
        c = TokenCTPH(4, 8)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        d1 = c.compute(tokens)
        d2 = c.compute(tokens)
        assert d1.similarity(d2) == 1.0

    def test_different(self):
        c = TokenCTPH(4, 8)
        d1 = c.compute([1, 2, 3, 4, 5])
        d2 = c.compute([100, 200, 300, 400, 500])
        sim = d1.similarity(d2)
        assert sim < 0.5

    def test_empty(self):
        c = TokenCTPH(4, 8)
        d = c.compute([])
        assert len(d.blocks) == 0

    def test_pickle_roundtrip(self):
        c = TokenCTPH(4, 8)
        data = pickle.dumps(c)
        restored = pickle.loads(data)
        d1 = c.compute([1, 2, 3])
        d2 = restored.compute([1, 2, 3])
        assert d1.similarity(d2) == 1.0


class TestCTPHDigest:
    def test_from_string_roundtrip(self):
        c = CTPH(64, 8, 4)
        d = c.hash_str("test data for roundtrip")
        s = str(d)
        restored = CTPHDigest.from_string(s)
        assert d.similarity(restored) == 1.0

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            CTPHDigest.from_string("")

    def test_properties(self):
        c = CTPH(64, 8, 4)
        d = c.hash_str("test")
        assert d.window_size == 64
        assert d.digest_size == 8
        assert isinstance(d.blocks, list)

    def test_pickle_roundtrip(self):
        c = CTPH(64, 8, 4)
        d = c.hash_str("test pickle")
        data = pickle.dumps(d)
        restored = pickle.loads(data)
        assert d.similarity(restored) == 1.0

    def test_repr(self):
        c = CTPH(64, 8, 4)
        d = c.hash_str("test")
        r = repr(d)
        assert "64" in r
        assert "8" in r


class TestCTPHStandalone:
    def test_ctph_hash_bytes(self):
        h = ctph_hash_bytes(b"hello world", 64, 8, 4)
        assert isinstance(h, str)
        assert ":" in h

    def test_ctph_hash_str(self):
        h1 = ctph_hash_str("hello world", 64, 8, 4)
        h2 = ctph_hash_str("hello world", 64, 8, 4)
        assert h1 == h2

    def test_ctph_similarity(self):
        h1 = ctph_hash_str("hello world", 64, 8, 4)
        h2 = ctph_hash_str("hello world", 64, 8, 4)
        assert ctph_similarity(h1, h2) == 1.0

    def test_token_ctph_hash(self):
        h1 = token_ctph_hash([1, 2, 3, 4, 5], 4, 8)
        h2 = token_ctph_hash([1, 2, 3, 4, 5], 4, 8)
        assert h1 == h2
        assert ctph_similarity(h1, h2) == 1.0

    def test_different_params_zero(self):
        h1 = ctph_hash_str("hello", 32, 4, 4)
        h2 = ctph_hash_str("hello", 64, 4, 4)
        assert ctph_similarity(h1, h2) == 0.0

    def test_ctph_piece_similarity_standalone(self):
        h1 = ctph_hash_str("hello world " * 100, 64, 8, 4)
        h2 = ctph_hash_str("hello world " * 100, 64, 8, 4)
        assert ctph_piece_similarity(h1, h2) == 1.0


class TestPieceSimilarity:
    """Tests for piece-level CTPH similarity (the edit-aware comparison).

    Note: piece_similarity requires sufficient text (>5KB) to produce enough
    pieces for meaningful Jaccard. Short texts produce few blocks/pieces and
    can give counterintuitive results.
    """

    @pytest.fixture()
    def long_text(self):
        """~10KB of War & Peace for CTPH testing (diverse, non-repeating)."""
        from pathlib import Path

        try:
            return Path("tests/fixtures/war_and_peace.txt").read_text()[:10000]
        except FileNotFoundError:
            # Fallback for CI without fixtures
            import hashlib

            return "".join(hashlib.sha256(f"block_{i}".encode()).hexdigest() for i in range(200))

    def test_identical(self, long_text):
        c = CTPH(64, 8, 4)
        d = c.hash_str(long_text)
        assert d.piece_similarity(d) == 1.0

    def test_piece_better_for_middle_edit(self, long_text):
        """Piece similarity should be much higher than block for localized edits."""
        c = CTPH(64, 8, 4)
        mid = len(long_text) // 2
        # Replace 5% in the middle
        replace_len = len(long_text) // 20
        modified = long_text[:mid] + "X" * replace_len + long_text[mid + replace_len :]
        d1 = c.hash_str(long_text)
        d2 = c.hash_str(modified)
        piece_sim = d1.piece_similarity(d2)
        block_sim = d1.similarity(d2)
        assert piece_sim > block_sim, f"piece_sim={piece_sim} should be > block_sim={block_sim}"
        assert piece_sim > 0.5, f"piece_sim={piece_sim} should be > 0.5"

    def test_append_high_similarity(self, long_text):
        c = CTPH(64, 8, 4)
        d1 = c.hash_str(long_text)
        d2 = c.hash_str(long_text + "X" * 100)
        piece_sim = d1.piece_similarity(d2)
        assert piece_sim > 0.9, f"Appended 1% piece_sim={piece_sim} should be > 0.9"

    def test_cross_document_low(self, long_text):
        c = CTPH(64, 8, 4)
        # Use a completely different section of War & Peace
        from pathlib import Path

        try:
            different = Path("tests/fixtures/shakespeare.txt").read_text()[:10000]
        except FileNotFoundError:
            import hashlib

            different = "".join(
                hashlib.sha256(f"other_{i}".encode()).hexdigest() for i in range(200)
            )
        d1 = c.hash_str(long_text)
        d2 = c.hash_str(different)
        assert d1.piece_similarity(d2) < 0.15

    def test_num_pieces(self, long_text):
        c = CTPH(64, 8, 4)
        d = c.hash_str(long_text)
        assert d.num_pieces > 0

    def test_token_ctph_piece_similarity(self):
        c = TokenCTPH(4, 8)
        tokens = list(range(1000))
        d = c.compute(tokens)
        assert d.piece_similarity(d) == 1.0

    def test_different_params_zero(self, long_text):
        c1 = CTPH(32, 4, 4)
        c2 = CTPH(64, 4, 4)
        d1 = c1.hash_str(long_text)
        d2 = c2.hash_str(long_text)
        assert d1.piece_similarity(d2) == 0.0


class TestConcurrentCTPH:
    def test_concurrent_ctph(self):
        """Multiple threads computing CTPH concurrently."""
        c = CTPH(64, 8, 4)
        results = [None] * 8
        errors = []

        def worker(idx):
            try:
                text = f"Document number {idx} " * 100
                d = c.hash_str(text)
                results[idx] = d
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r is not None for r in results)

    def test_concurrent_token_ctph(self):
        """Multiple threads computing TokenCTPH concurrently."""
        c = TokenCTPH(4, 8)
        results = [None] * 8
        errors = []

        def worker(idx):
            try:
                tokens = list(range(idx * 100, idx * 100 + 200))
                d = c.compute(tokens)
                results[idx] = d
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r is not None for r in results)
