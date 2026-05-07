"""Tests verifying that long-running functions can be called concurrently.

These tests verify that py.allow_threads() works correctly:
- Two threads can call long-running Rust functions simultaneously
- No deadlocks or data corruption
- Results are correct regardless of concurrency
"""

import threading
from typing import Any

from kaos_nlp_core.matching import FstSet, MultiPatternMatcher, RegexMatcher
from kaos_nlp_core.segmentation import PunktTokenizer
from kaos_nlp_core.structures import InvertedIndex
from kaos_nlp_core.tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_index() -> InvertedIndex:
    """Build a small test index."""
    idx = InvertedIndex()
    docs = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "a cat and a dog",
        "the quick brown fox jumps over the lazy dog",
        "the lazy dog sleeps all day",
    ]
    for i, doc in enumerate(docs):
        idx.add_document(i, doc.split())
    return idx


LONG_TEXT = "Hello world. " * 500 + "This is a test. " * 500


# ---------------------------------------------------------------------------
# InvertedIndex concurrent queries
# ---------------------------------------------------------------------------


class TestConcurrentInvertedIndex:
    """Verify two threads can query an InvertedIndex simultaneously."""

    def test_concurrent_bm25_queries(self):
        idx = _build_test_index()
        results: list[Any] = [None, None]
        errors: list[Any] = [None, None]

        def query_thread(thread_id, terms, top_k):
            try:
                r = idx.query_bm25(terms, top_k=top_k)
                results[thread_id] = r
            except Exception as e:
                errors[thread_id] = e

        t1 = threading.Thread(target=query_thread, args=(0, ["cat", "dog"], 10))
        t2 = threading.Thread(target=query_thread, args=(1, ["quick", "fox"], 10))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors[0] is None, f"Thread 0 error: {errors[0]}"
        assert errors[1] is None, f"Thread 1 error: {errors[1]}"
        assert results[0] is not None
        assert results[1] is not None
        # Verify correctness
        assert len(results[0]) > 0  # cat/dog appear in multiple docs
        assert len(results[1]) > 0  # quick/fox appear in doc 3

    def test_concurrent_query_and_add(self):
        """One thread queries while another builds a separate index."""
        idx = _build_test_index()
        query_results: list[Any] = [None]
        build_results: list[Any] = [None]
        errors = []

        def query_thread():
            try:
                for _ in range(50):
                    r = idx.query_bm25(["cat"], top_k=5)
                    query_results[0] = r
            except Exception as e:
                errors.append(e)

        def build_thread():
            try:
                new_idx = InvertedIndex()
                for i in range(100):
                    new_idx.add_document(i, [f"term_{j}" for j in range(20)])
                build_results[0] = new_idx
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=query_thread)
        t2 = threading.Thread(target=build_thread)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"
        assert query_results[0] is not None
        assert build_results[0] is not None
        assert build_results[0].doc_count() == 100

    def test_build_batch_parallel(self):
        """InvertedIndex.build_batch uses rayon internally."""
        documents = [
            (i, f"doc {i} with some words term_{i} common shared".split()) for i in range(200)
        ]
        idx = InvertedIndex.build_batch(documents)
        assert idx.doc_count() == 200
        assert idx.doc_freq("common") == 200
        assert idx.doc_freq("shared") == 200
        # Each doc has a unique term
        assert idx.doc_freq("term_0") == 1
        assert idx.doc_freq("term_199") == 1


# ---------------------------------------------------------------------------
# Tokenizer concurrent access
# ---------------------------------------------------------------------------


class TestConcurrentTokenizer:
    """Verify concurrent tokenization works."""

    def test_concurrent_tokenize(self):
        tok = Tokenizer(lowercase=True)
        results: list[Any] = [None, None]
        errors: list[Any] = [None, None]

        def tokenize_thread(thread_id, text):
            try:
                r = tok.tokenize_words(text)
                results[thread_id] = r
            except Exception as e:
                errors[thread_id] = e

        t1 = threading.Thread(target=tokenize_thread, args=(0, "Hello World Test"))
        t2 = threading.Thread(target=tokenize_thread, args=(1, "Another Text Here"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors[0] is None
        assert errors[1] is None
        assert results[0] == ["hello", "world", "test"]
        assert results[1] == ["another", "text", "here"]


# ---------------------------------------------------------------------------
# Punkt tokenizer concurrent access
# ---------------------------------------------------------------------------


class TestConcurrentPunkt:
    """Verify concurrent Punkt tokenization works."""

    def test_concurrent_sentence_tokenize(self):
        tok = PunktTokenizer()
        results: list[Any] = [None, None]
        errors: list[Any] = [None, None]

        def sentence_thread(thread_id, text):
            try:
                r = tok.tokenize(text)
                results[thread_id] = r
            except Exception as e:
                errors[thread_id] = e

        t1 = threading.Thread(target=sentence_thread, args=(0, "Hello world. How are you? Fine."))
        t2 = threading.Thread(
            target=sentence_thread, args=(1, "First sentence. Second one. Third.")
        )
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors[0] is None
        assert errors[1] is None
        assert len(results[0]) == 3
        assert len(results[1]) == 3

    def test_concurrent_tokenize_batch(self):
        """tokenize_batch uses rayon for 4+ texts."""
        tok = PunktTokenizer()
        texts: list[str] = [LONG_TEXT for _ in range(8)]
        results: list[Any] = [None, None]
        errors: list[Any] = [None, None]

        def batch_thread(thread_id):
            try:
                r = tok.tokenize_batch(texts)
                results[thread_id] = r
            except Exception as e:
                errors[thread_id] = e

        t1 = threading.Thread(target=batch_thread, args=(0,))
        t2 = threading.Thread(target=batch_thread, args=(1,))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert errors[0] is None
        assert errors[1] is None
        assert len(results[0]) == 8
        assert len(results[1]) == 8
        # Both should produce identical results
        for a, b in zip(results[0], results[1], strict=True):
            assert a == b


# ---------------------------------------------------------------------------
# Matching concurrent access
# ---------------------------------------------------------------------------


class TestConcurrentMatching:
    """Verify concurrent pattern matching works."""

    def test_concurrent_regex(self):
        matcher = RegexMatcher(r"\b\w+ing\b")
        texts = [
            "The running fox was jumping over the sleeping dog",
            "Walking and talking while drinking coffee",
        ]
        results: list[Any] = [None, None]
        errors: list[Any] = [None, None]

        def match_thread(thread_id, text):
            try:
                r = matcher.find_all(text)
                results[thread_id] = r
            except Exception as e:
                errors[thread_id] = e

        t1 = threading.Thread(target=match_thread, args=(0, texts[0]))
        t2 = threading.Thread(target=match_thread, args=(1, texts[1]))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors[0] is None
        assert errors[1] is None
        assert len(results[0]) > 0
        assert len(results[1]) > 0

    def test_concurrent_multi_pattern(self):
        matcher = MultiPatternMatcher(["cat", "dog", "fox"])
        results: list[Any] = [None, None]
        errors: list[Any] = [None, None]

        def match_thread(thread_id, text):
            try:
                r = matcher.find_all(text)
                results[thread_id] = r
            except Exception as e:
                errors[thread_id] = e

        t1 = threading.Thread(target=match_thread, args=(0, "the cat and the dog"))
        t2 = threading.Thread(target=match_thread, args=(1, "a quick fox"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors[0] is None
        assert errors[1] is None
        assert len(results[0]) == 2  # cat, dog
        assert len(results[1]) == 1  # fox

    def test_concurrent_fst(self):
        terms = sorted([f"term_{i:04d}" for i in range(1000)])
        fst = FstSet(terms)
        results: list[Any] = [None, None]
        errors: list[Any] = [None, None]

        def search_thread(thread_id, query):
            try:
                r = fst.fuzzy_search(query, max_distance=1)
                results[thread_id] = r
            except Exception as e:
                errors[thread_id] = e

        t1 = threading.Thread(target=search_thread, args=(0, "term_0001"))
        t2 = threading.Thread(target=search_thread, args=(1, "term_0500"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors[0] is None
        assert errors[1] is None
        assert len(results[0]) > 0
        assert len(results[1]) > 0


# ---------------------------------------------------------------------------
# Stress test: many concurrent operations
# ---------------------------------------------------------------------------


class TestConcurrentStress:
    """Stress test with many threads."""

    def test_many_concurrent_queries(self):
        """8 threads querying simultaneously."""
        idx = _build_test_index()
        num_threads = 8
        iterations = 20
        results: list[Any] = [None] * num_threads
        errors = []

        def query_thread(thread_id):
            try:
                for _ in range(iterations):
                    r = idx.query_bm25(["cat", "dog", "the"], top_k=5)
                    results[thread_id] = r
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=query_thread, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"
        # All threads should get the same result
        first = results[0]
        assert first is not None
        for i in range(num_threads):
            r = results[i]
            assert r is not None
            assert len(r) == len(first)

    def test_mixed_operations_concurrent(self):
        """Different operations running concurrently."""
        idx = _build_test_index()
        tok = Tokenizer(lowercase=True)
        punkt = PunktTokenizer()
        matcher = RegexMatcher(r"\w+")
        errors = []

        def idx_thread():
            try:
                for _ in range(20):
                    idx.query_bm25(["cat"], top_k=3)
            except Exception as e:
                errors.append(("idx", e))

        def tok_thread():
            try:
                for _ in range(20):
                    tok.tokenize_words("hello world test case")
            except Exception as e:
                errors.append(("tok", e))

        def punkt_thread():
            try:
                for _ in range(20):
                    punkt.tokenize("Hello world. Test sentence.")
            except Exception as e:
                errors.append(("punkt", e))

        def regex_thread():
            try:
                for _ in range(20):
                    matcher.find_all("hello world test case")
            except Exception as e:
                errors.append(("regex", e))

        threads = [
            threading.Thread(target=idx_thread),
            threading.Thread(target=tok_thread),
            threading.Thread(target=punkt_thread),
            threading.Thread(target=regex_thread),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"
