"""Threading and concurrency tests for PyO3 extension module.

Tests for common PyO3/Maturin issues:
1. Pickle support (required for multiprocessing)
2. Thread safety of pure functions (concurrent calls from ThreadPoolExecutor)
3. Thread safety of shared read-only objects
4. GIL contention under load
5. Multiprocessing with spawn (workers creating fresh objects)

See: https://pyo3.rs/main/parallelism.html
"""

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from kaos_nlp_core.algorithms import (
    levenshtein,
    ngram_jaccard,
    soundex_encode,
    token_jaccard,
)
from kaos_nlp_core.matching import (
    FstMap,
    FstSet,
    MultiPatternMatcher,
    RegexMatcher,
    RegexSetMatcher,
    substring_find_all,
)
from kaos_nlp_core.segmentation import (
    PunktParameters,
    PunktTokenizer,
)
from kaos_nlp_core.structures import (
    BloomVocabulary,
    FrequencyVocabulary,
    IndexedVocabulary,
    InvertedIndex,
    SetVocabulary,
)

# ── 1. Pickle support ───────────────────────────────────────────────────────


ALL_PYCLASS_FACTORIES = {
    "SetVocabulary": lambda: SetVocabulary(["hello", "world"]),
    "FrequencyVocabulary": lambda: _freq_vocab(),
    "IndexedVocabulary": lambda: _indexed_vocab(),
    "BloomVocabulary": lambda: _bloom_vocab(),
    "InvertedIndex": lambda: _inverted_index(),
    "FstSet": lambda: FstSet(["hello", "world"]),
    "FstMap": lambda: FstMap([("hello", 1), ("world", 2)]),
    "MultiPatternMatcher": lambda: MultiPatternMatcher(["hello", "world"]),
    "RegexMatcher": lambda: RegexMatcher(r"\d+"),
    "RegexSetMatcher": lambda: RegexSetMatcher([r"\d+", r"[a-z]+"]),
}


def _freq_vocab():
    v = FrequencyVocabulary()
    v.insert("hello")
    return v


def _indexed_vocab():
    v = IndexedVocabulary()
    v.insert("hello")
    return v


def _bloom_vocab():
    v = BloomVocabulary()
    v.insert("hello")
    return v


def _inverted_index():
    idx = InvertedIndex()
    idx.add_document(0, ["hello", "world"])
    return idx


class TestPickleSupport:
    """Verify pickle round-trip for all pyclass types.

    All types implement __getstate__/__setstate__ (serde types) or
    __getnewargs__ (reconstructable types) for full pickle support.
    """

    @pytest.mark.parametrize("name", sorted(ALL_PYCLASS_FACTORIES.keys()))
    def test_pyclass_pickle_roundtrip(self, name):
        """All pyclass objects should survive pickle.dumps/loads."""
        obj = ALL_PYCLASS_FACTORIES[name]()
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
        assert type(obj2) is type(obj)

    def test_inverted_index_pickle_preserves_state(self):
        """InvertedIndex should preserve all indexed data through pickle."""
        idx = InvertedIndex()
        idx.add_document(0, ["the", "cat", "sat"])
        idx.add_document(1, ["the", "dog", "ran"])

        idx2 = pickle.loads(pickle.dumps(idx))
        assert idx2.doc_count() == 2
        assert idx2.doc_freq("the") == 2
        assert idx2.doc_freq("cat") == 1
        results = idx2.query_bm25(["cat"], top_k=5)
        assert len(results) == 1
        assert results[0].doc_id == 0

    def test_fst_set_pickle_preserves_data(self):
        """FstSet should preserve all keys through pickle."""
        fst = FstSet(["alpha", "beta", "gamma"])
        fst2 = pickle.loads(pickle.dumps(fst))
        assert len(fst2) == 3
        assert fst2.contains("beta")
        assert not fst2.contains("delta")

    def test_regex_matcher_pickle_works(self):
        """RegexMatcher should recompile pattern after unpickle."""
        r = RegexMatcher(r"\b\d+\b")
        r2 = pickle.loads(pickle.dumps(r))
        assert r2.count("abc 123 def 456") == 2

    def test_multi_pattern_pickle_preserves_config(self):
        """MultiPatternMatcher preserves case_insensitive and longest_match."""
        m = MultiPatternMatcher(["hello", "world"], case_insensitive=True)
        m2 = pickle.loads(pickle.dumps(m))
        assert m2.is_match("HELLO WORLD")
        assert m2.count("HELLO WORLD") == 2


# ── 2. Thread safety: pure functions ─────────────────────────────────────────


class TestThreadSafePureFunctions:
    """Pure function calls from multiple threads should be safe.

    These functions don't share state — each call creates its own Rust objects.
    The GIL ensures only one thread runs Python at a time, but the Rust code
    should not have data races.
    """

    def test_levenshtein_concurrent(self):
        """8 threads calling levenshtein concurrently."""
        errors = []

        def worker():
            try:
                for _ in range(200):
                    r = levenshtein("kitten", "sitting")
                    assert r.distance == 3.0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread errors: {errors}"

    def test_mixed_algorithms_concurrent(self):
        """Different algorithms running concurrently on different threads."""
        errors = []

        def worker_levenshtein():
            try:
                for _ in range(100):
                    levenshtein("hello", "world")
            except Exception as e:
                errors.append(("levenshtein", e))

        def worker_ngram():
            try:
                for _ in range(100):
                    ngram_jaccard("hello world", "foo bar", n=2)
            except Exception as e:
                errors.append(("ngram", e))

        def worker_token():
            try:
                for _ in range(100):
                    token_jaccard("the quick brown fox", "a quick dog", lowercase=True)
            except Exception as e:
                errors.append(("token", e))

        def worker_soundex():
            try:
                for _ in range(100):
                    soundex_encode("Philadelphia")
            except Exception as e:
                errors.append(("soundex", e))

        thread_fns = [worker_levenshtein, worker_ngram, worker_token, worker_soundex] * 2
        threads = [threading.Thread(target=fn) for fn in thread_fns]

        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread errors: {errors}"

    def test_substring_search_concurrent(self):
        """Substring search from multiple threads."""
        text = "hello world " * 100
        errors = []

        def worker():
            try:
                for _ in range(100):
                    matches = substring_find_all(text, "hello")
                    assert len(matches) == 100
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ── 3. Thread safety: shared read-only objects ──────────────────────────────


class TestThreadSafeSharedObjects:
    """Concurrent reads on shared pyclass objects should be safe.

    PyO3 classes are behind the GIL, so concurrent reads are serialized.
    These tests verify no corruption occurs.
    """

    def test_shared_inverted_index_reads(self):
        """Multiple threads reading from the same InvertedIndex."""
        idx = InvertedIndex()
        idx.add_document(0, ["the", "cat", "sat", "on", "the", "mat"])
        idx.add_document(1, ["the", "dog", "sat", "on", "the", "log"])
        idx.add_document(2, ["a", "cat", "and", "a", "dog"])

        errors = []

        def reader():
            try:
                for _ in range(100):
                    assert idx.doc_count() == 3
                    assert idx.doc_freq("the") == 2
                    results = idx.query_bm25(["cat"], top_k=5)
                    assert len(results) >= 1
                    score = idx.tf_idf("the", 0)
                    assert score > 0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread errors: {errors}"

    def test_shared_fst_reads(self):
        """Multiple threads searching the same FstSet."""
        fst = FstSet(sorted(["apple", "banana", "cherry", "date", "elderberry"]))
        errors = []

        def reader():
            try:
                for _ in range(200):
                    assert fst.contains("banana")
                    assert not fst.contains("fig")
                    results = fst.fuzzy_search("banan", 1)
                    assert any(r.key == "banana" for r in results)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_shared_multi_pattern_reads(self):
        """Multiple threads matching against the same MultiPatternMatcher."""
        m = MultiPatternMatcher(["cat", "dog", "bird"])
        text = "I have a cat and a dog and a bird"
        errors = []

        def reader():
            try:
                for _ in range(200):
                    matches = m.find_all(text)
                    assert len(matches) == 3
                    assert m.count(text) == 3
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ── 4. ThreadPoolExecutor (real-world pattern) ──────────────────────────────


class TestThreadPoolExecutor:
    """ThreadPoolExecutor is the most common real-world threading pattern."""

    def test_executor_pure_functions(self):
        """Map pure functions across a thread pool."""
        pairs = [
            ("kitten", "sitting"),
            ("martha", "marhta"),
            ("Robert", "Rupert"),
        ] * 100  # 300 pairs

        def do_levenshtein(pair):
            return levenshtein(pair[0], pair[1])

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(do_levenshtein, pairs))

        assert len(results) == 300
        # First 100 pairs are all ("kitten", "sitting") -> distance 3
        for i in range(0, 300, 3):
            assert results[i].distance == 3.0

    def test_executor_shared_index(self):
        """Multiple queries against a shared index via thread pool."""
        idx = InvertedIndex()
        idx.add_document(0, ["tax", "income", "deduction"])
        idx.add_document(1, ["military", "defense", "armed"])
        idx.add_document(2, ["tax", "code", "section"])

        queries = [["tax"], ["military"], ["income"], ["defense"]] * 50

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda q: idx.query_bm25(q, top_k=5), queries))

        assert len(results) == 200
        # All results should be valid lists of dicts
        for r in results:
            assert isinstance(r, list)

    def test_executor_token_similarity_batch(self):
        """Batch token similarity via thread pool."""
        pairs = [
            ("the quick brown fox", "a quick brown dog"),
            ("machine learning", "deep learning"),
            ("natural language processing", "natural language understanding"),
        ] * 100

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda p: token_jaccard(p[0], p[1], lowercase=True),
                    pairs,
                )
            )

        assert len(results) == 300
        assert all(0 <= r.similarity <= 1 for r in results)


# ── 5. Multiprocessing (spawn mode) ─────────────────────────────────────────
# Note: multiprocessing spawn tests need __name__ == "__main__" guard
# and must be in a real .py file (not stdin). We test via subprocess.


class TestMultiprocessingSpawn:
    """Test multiprocessing with spawn start method via subprocess.

    Spawn-mode multiprocessing requires a real .py file (not -c or stdin)
    because spawned workers re-import __main__. We use tempfile for this.
    """

    def _run_mp_script(self, script: str, timeout: int = 30) -> tuple[str, str, int]:
        """Write script to tempfile and run it."""
        import subprocess
        import sys
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        return result.stdout, result.stderr, result.returncode

    def test_spawn_workers_create_fresh_objects(self):
        """Workers that create fresh objects should work with spawn."""
        stdout, stderr, rc = self._run_mp_script("""
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def worker(worker_id):
    from kaos_nlp_core.algorithms import levenshtein
    from kaos_nlp_core.structures import InvertedIndex
    for _ in range(10):
        levenshtein("hello", "world")
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "world"])
        idx.query_bm25(["hello"], top_k=1)
    return worker_id

if __name__ == "__main__":
    with mp.Pool(2) as pool:
        results = pool.map(worker, range(2))
    assert sorted(results) == [0, 1], f"Got {results}"
    print("OK")
""")
        assert rc == 0, f"rc={rc}, stderr={stderr[:500]}"
        assert "OK" in stdout

    def test_spawn_passes_pickled_index_to_worker(self):
        """Workers should receive pickled InvertedIndex and query it."""
        stdout, stderr, rc = self._run_mp_script("""
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def worker(idx):
    return idx.doc_count()

if __name__ == "__main__":
    from kaos_nlp_core.structures import InvertedIndex
    idx = InvertedIndex()
    idx.add_document(0, ["hello", "world"])
    idx.add_document(1, ["foo", "bar"])
    with mp.Pool(2) as pool:
        results = pool.map(worker, [idx, idx])
    assert results == [2, 2], f"Got {results}"
    print("OK")
""")
        assert rc == 0, f"rc={rc}, stderr={stderr[:500]}"
        assert "OK" in stdout

    def test_spawn_bm25_query_in_worker(self):
        """Workers should be able to run BM25 queries on pickled index."""
        stdout, stderr, rc = self._run_mp_script("""
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def worker(args):
    idx, query = args
    results = idx.query_bm25(query, top_k=5)
    return len(results)

if __name__ == "__main__":
    from kaos_nlp_core.structures import InvertedIndex
    idx = InvertedIndex()
    idx.add_document(0, ["tax", "income", "deduction"])
    idx.add_document(1, ["military", "defense"])
    idx.add_document(2, ["tax", "code"])
    queries = [["tax"], ["military"], ["income"], ["defense"]]
    with mp.Pool(2) as pool:
        results = pool.map(worker, [(idx, q) for q in queries])
    assert all(r >= 0 for r in results), f"Got {results}"
    print("OK")
""")
        assert rc == 0, f"rc={rc}, stderr={stderr[:500]}"
        assert "OK" in stdout


# ── 6. Segmentation threading ────────────────────────────────────────────────


class TestSegmentationThreading:
    """Thread safety and pickle tests for segmentation types."""

    def test_punkt_tokenizer_pickle(self):
        """PunktTokenizer should survive pickle and produce same results."""
        tok = PunktTokenizer()
        text = "Hello world. How are you? I am fine."
        before = tok.tokenize(text)

        tok2 = pickle.loads(pickle.dumps(tok))
        after = tok2.tokenize(text)
        assert before == after

    def test_punkt_parameters_pickle(self):
        """PunktParameters loaded from model file should survive pickle."""
        from pathlib import Path

        path = Path(__file__).parent / ".." / ".." / "models" / "default.npkt.gz"
        if not path.exists():
            pytest.skip("default.npkt.gz model not available")

        params = PunktParameters.load(str(path))
        params2 = pickle.loads(pickle.dumps(params))
        assert params2.num_abbreviations == params.num_abbreviations
        assert params2.num_collocations == params.num_collocations
        assert params2.num_sent_starters == params.num_sent_starters

    def test_punkt_tokenizer_threading(self):
        """4 threads calling tokenize() on the same PunktTokenizer."""
        tok = PunktTokenizer()
        texts = [
            "Hello world. How are you?",
            "Dr. Smith went to Washington. He was happy.",
            "First sentence. Second sentence. Third sentence.",
            "Just one sentence",
        ]
        errors = []

        def worker(text):
            try:
                for _ in range(100):
                    sents = tok.tokenize(text)
                    assert len(sents) >= 1
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, t) for t in texts]
            for f in futures:
                f.result()

        assert not errors, f"Thread errors: {errors}"

    def test_punkt_tokenizer_multiprocessing(self):
        """Spawn 2 workers that each tokenize a text."""
        import subprocess
        import sys
        import tempfile

        script = """
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def worker(text):
    from kaos_nlp_core.segmentation import PunktTokenizer
    tok = PunktTokenizer()
    sents = tok.tokenize(text)
    return len(sents)

if __name__ == "__main__":
    texts = [
        "Hello world. How are you? I am fine.",
        "First sentence. Second sentence.",
    ]
    with mp.Pool(2) as pool:
        results = pool.map(worker, texts)
    assert results == [3, 2], f"Got {results}"
    print("OK")
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=30,
            )
        from pathlib import Path

        Path(f.name).unlink()
        assert result.returncode == 0, f"rc={result.returncode}, stderr={result.stderr[:500]}"
        assert "OK" in result.stdout
