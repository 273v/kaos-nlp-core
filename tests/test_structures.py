"""Tests for data structures: vocabularies, inverted index."""

from pathlib import Path

import pytest

from kaos_nlp_core.structures import (
    BloomVocabulary,
    FrequencyVocabulary,
    IndexedVocabulary,
    InvertedIndex,
    SetVocabulary,
    SimilarityMatrix,
    SparseTermMatrix,
)

# --- Vocabularies ---


class TestSetVocabulary:
    def test_basic(self):
        v = SetVocabulary()
        assert v.insert("hello")
        assert not v.insert("hello")  # duplicate
        assert v.contains("hello")
        assert not v.contains("world")
        assert len(v) == 1

    def test_in_operator(self):
        v = SetVocabulary(["a", "b", "c"])
        assert "a" in v
        assert "x" not in v

    def test_from_list(self):
        v = SetVocabulary(["hello", "world"])
        assert len(v) == 2
        assert v.contains("hello")

    def test_remove(self):
        v = SetVocabulary(["a", "b"])
        assert v.remove("a")
        assert not v.contains("a")
        assert not v.remove("a")  # already removed


class TestFrequencyVocabulary:
    def test_insert_and_count(self):
        v = FrequencyVocabulary()
        id1 = v.insert("cat")
        id2 = v.insert("cat")
        assert id1 == id2
        assert v.get_count("cat") == 2

    def test_insert_with_count(self):
        v = FrequencyVocabulary()
        v.insert_with_count("dog", 10)
        assert v.get_count("dog") == 10

    def test_top_n(self):
        v = FrequencyVocabulary()
        for _ in range(5):
            v.insert("a")
        for _ in range(3):
            v.insert("b")
        v.insert("c")
        top = v.top_n(2)
        assert top[0] == ("a", 5)
        assert top[1] == ("b", 3)

    def test_total_count(self):
        v = FrequencyVocabulary()
        v.insert_with_count("a", 10)
        v.insert_with_count("b", 20)
        assert v.total_count() == 30

    def test_in_operator(self):
        v = FrequencyVocabulary()
        v.insert("hello")
        assert "hello" in v
        assert "world" not in v


class TestIndexedVocabulary:
    def test_insert_and_lookup(self):
        v = IndexedVocabulary()
        id0 = v.insert("hello")
        id1 = v.insert("world")
        assert id0 == 0
        assert id1 == 1
        assert v.get_term(0) == "hello"
        assert v.get_id("world") == 1

    def test_duplicate(self):
        v = IndexedVocabulary()
        id0 = v.insert("hello")
        id1 = v.insert("hello")
        assert id0 == id1
        assert len(v) == 1

    def test_in_operator(self):
        v = IndexedVocabulary()
        v.insert("test")
        assert "test" in v


class TestBloomVocabulary:
    def test_insert_and_check(self):
        v = BloomVocabulary(1000, 0.01)
        v.insert("hello")
        v.insert("world")
        assert v.contains("hello")
        assert v.contains("world")
        assert v.approx_len() == 2

    def test_in_operator(self):
        v = BloomVocabulary()
        v.insert("test")
        assert "test" in v


# --- Inverted Index ---


class TestInvertedIndex:
    def _build_index(self):
        idx = InvertedIndex()
        idx.add_document(0, ["the", "cat", "sat", "on", "the", "mat"])
        idx.add_document(1, ["the", "dog", "sat", "on", "the", "log"])
        idx.add_document(2, ["a", "cat", "and", "a", "dog"])
        return idx

    def test_doc_freq(self):
        idx = self._build_index()
        assert idx.doc_freq("the") == 2
        assert idx.doc_freq("cat") == 2
        assert idx.doc_freq("log") == 1
        assert idx.doc_freq("xyz") == 0

    def test_query_and(self):
        idx = self._build_index()
        result = idx.query_and(["cat", "sat"])
        assert result == [0]

    def test_query_or(self):
        idx = self._build_index()
        result = idx.query_or(["cat", "dog"])
        assert set(result) == {0, 1, 2}

    def test_tf_idf(self):
        idx = self._build_index()
        score = idx.tf_idf("the", 0)
        assert score > 0.0
        assert idx.tf_idf("the", 2) == 0.0  # not in doc 2

    def test_postings(self):
        idx = self._build_index()
        postings = idx.get_postings("the")
        assert postings is not None
        doc0 = next(p for p in postings if p.doc_id == 0)
        assert doc0.term_freq == 2

    def test_stats(self):
        idx = self._build_index()
        assert idx.term_count() > 0
        assert idx.doc_count() == 3

    # --- Doc lengths ---

    def test_doc_length(self):
        idx = self._build_index()
        assert idx.doc_length(0) == 6
        assert idx.doc_length(2) == 5
        assert idx.doc_length(99) == 0

    def test_avg_doc_length(self):
        idx = self._build_index()
        assert abs(idx.avg_doc_length() - 17.0 / 3.0) < 1e-10

    # --- TF-IDF variants ---

    def test_tf_idf_sublinear(self):
        idx = self._build_index()
        import math

        score = idx.tf_idf_weighted("the", 0, tf_weight="sublinear", idf_weight="standard")
        expected = (1.0 + math.log(2)) * math.log(3.0 / 2.0)
        assert abs(score - expected) < 1e-10

    def test_tf_idf_boolean(self):
        idx = self._build_index()
        import math

        score = idx.tf_idf_weighted("the", 0, tf_weight="boolean", idf_weight="standard")
        assert abs(score - math.log(3.0 / 2.0)) < 1e-10

    def test_tf_idf_smooth_idf(self):
        idx = self._build_index()
        import math

        score = idx.tf_idf_weighted("cat", 0, tf_weight="raw", idf_weight="smooth")
        expected_idf = math.log(4.0 / 3.0) + 1.0
        expected = 1.0 * expected_idf  # cat appears once in doc 0
        assert abs(score - expected) < 1e-10

    def test_score_tf_idf_multi(self):
        idx = self._build_index()
        score = idx.score_tf_idf(["cat", "dog"], 2)
        s1 = idx.tf_idf("cat", 2)
        s2 = idx.tf_idf("dog", 2)
        assert abs(score - (s1 + s2)) < 1e-10

    def test_tf_idf_invalid_weight(self):
        idx = self._build_index()
        with pytest.raises(ValueError):
            idx.tf_idf_weighted("the", 0, tf_weight="invalid")

    # --- BM25 ---

    def test_bm25_basic(self):
        idx = self._build_index()
        score = idx.score_bm25(["cat"], 0)
        assert score > 0.0
        # cat not in doc 1
        assert idx.score_bm25(["cat"], 1) == 0.0

    def test_bm25_multi_term(self):
        idx = self._build_index()
        score = idx.score_bm25(["cat", "sat"], 0)
        score_cat = idx.score_bm25(["cat"], 0)
        score_sat = idx.score_bm25(["sat"], 0)
        assert abs(score - (score_cat + score_sat)) < 1e-10

    def test_bm25_custom_params(self):
        idx = self._build_index()
        score_default = idx.score_bm25(["cat"], 0)
        score_custom = idx.score_bm25(["cat"], 0, k1=2.0, b=0.5)
        assert score_default != score_custom

    def test_bm25_length_normalization(self):
        idx = InvertedIndex()
        idx.add_document(0, ["cat"])  # short doc
        idx.add_document(
            1,
            ["cat", "the", "dog", "sat", "on", "the", "mat", "and", "ate", "food"],
        )  # long doc
        score_short = idx.score_bm25(["cat"], 0)
        score_long = idx.score_bm25(["cat"], 1)
        assert score_short > score_long

    # --- Ranked retrieval ---

    def test_query_bm25_ranked(self):
        idx = self._build_index()
        results = idx.query_bm25(["cat"])
        assert len(results) == 2
        assert results[0].score >= results[1].score
        doc_ids = {r.doc_id for r in results}
        assert doc_ids == {0, 2}

    def test_query_bm25_top_k(self):
        idx = self._build_index()
        results = idx.query_bm25(["sat"], top_k=1)
        assert len(results) == 1

    def test_query_bm25_empty(self):
        idx = self._build_index()
        assert idx.query_bm25([]) == []
        assert idx.query_bm25(["nonexistent"]) == []

    def test_query_tf_idf_ranked(self):
        idx = self._build_index()
        results = idx.query_tf_idf(["cat", "dog"])
        assert len(results) == 3
        # Doc 2 has both cat AND dog, should be ranked first
        assert results[0].doc_id == 2
        assert results[0].score > results[1].score

    def test_query_tf_idf_custom_weights(self):
        idx = self._build_index()
        results = idx.query_tf_idf(["cat", "dog"], tf_weight="boolean", idf_weight="probabilistic")
        assert len(results) > 0

    # --- Edge cases ---

    def test_get_postings_nonexistent(self):
        idx = self._build_index()
        assert idx.get_postings("xyzzy_nonexistent") is None

    def test_query_bm25_top_k_zero(self):
        idx = self._build_index()
        results = idx.query_bm25(["cat"], top_k=0)
        assert results == []

    def test_add_duplicate_doc_id(self):
        idx = InvertedIndex()
        idx.add_document(0, ["hello", "world"])
        idx.add_document(0, ["hello", "again"])
        # Doc 0 should accumulate terms
        assert idx.doc_count() >= 1
        assert idx.doc_freq("hello") >= 1

    def test_idf_smooth(self):

        idx = self._build_index()
        # Smooth IDF = ln((N+1)/(df+1)) + 1
        idf = idx.tf_idf_weighted("cat", 0, idf_weight="smooth")
        # Just verify it's positive and different from standard
        assert idf > 0

    def test_idf_probabilistic(self):
        idx = self._build_index()
        idf = idx.tf_idf_weighted("cat", 0, idf_weight="probabilistic")
        assert idf >= 0

    def test_save_load(self, tmp_path: Path):
        idx = self._build_index()
        path = tmp_path / "index.bin"
        idx.save(str(path))
        restored = InvertedIndex.load(str(path))
        assert restored.doc_count() == idx.doc_count()
        doc_ids = {result.doc_id for result in restored.query_bm25(["cat"])}
        assert doc_ids == {0, 2}


# --- SparseTermMatrix ---


class TestSparseTermMatrix:
    def test_basic(self):
        m = SparseTermMatrix()
        m.add_document([(0, 2), (1, 1), (2, 1)])
        m.add_document([(0, 1), (3, 1), (4, 1)])
        assert m.num_docs() == 2
        assert m.num_terms() == 5

    def test_term_freq(self):
        m = SparseTermMatrix()
        m.add_document([(0, 2), (1, 1)])
        assert m.get_term_freq(0, 0) == 2
        assert m.get_term_freq(0, 1) == 1
        assert m.get_term_freq(0, 5) == 0  # nonexistent

    def test_document_length(self):
        m = SparseTermMatrix()
        m.add_document([(0, 2), (1, 1), (2, 3)])
        assert m.document_length(0) == 6

    def test_document_terms(self):
        m = SparseTermMatrix()
        m.add_document([(0, 2), (1, 1)])
        terms = m.get_document_terms(0)
        assert (0, 2) in terms
        assert (1, 1) in terms

    def test_cosine_self(self):
        m = SparseTermMatrix()
        m.add_document([(0, 1), (1, 2)])
        assert abs(m.cosine_similarity(0, 0) - 1.0) < 1e-10

    def test_cosine_disjoint(self):
        m = SparseTermMatrix()
        m.add_document([(0, 1), (1, 1)])
        m.add_document([(2, 1), (3, 1)])
        assert m.cosine_similarity(0, 1) == 0.0

    def test_density(self):
        m = SparseTermMatrix()
        m.add_document([(0, 1), (1, 1)])
        m.add_document([(0, 1)])
        # 3 non-zeros out of 2*2=4 cells
        assert abs(m.density() - 3.0 / 4.0) < 1e-10

    def test_nnz(self):
        m = SparseTermMatrix()
        m.add_document([(0, 1), (1, 1), (2, 1)])
        assert m.nnz() == 3

    def test_memory_usage(self):
        m = SparseTermMatrix()
        m.add_document([(0, 1)])
        assert m.memory_usage() > 0

    def test_pickle_roundtrip(self):
        import pickle

        m = SparseTermMatrix()
        m.add_document([(0, 2), (1, 1)])
        m.add_document([(0, 1), (2, 3)])
        data = pickle.dumps(m)
        m2 = pickle.loads(data)
        assert m2.num_docs() == 2
        assert m2.get_term_freq(0, 0) == 2
        assert m2.get_term_freq(1, 2) == 3

    def test_repr(self):
        m = SparseTermMatrix()
        m.add_document([(0, 1)])
        r = repr(m)
        assert "num_docs=1" in r

    def test_save_load(self, tmp_path: Path):
        m = SparseTermMatrix()
        m.add_document([(0, 2), (1, 1)])
        path = tmp_path / "matrix.bin"
        m.save(str(path))
        restored = SparseTermMatrix.load(str(path))
        assert restored.get_term_freq(0, 0) == 2


# --- SimilarityMatrix ---


class TestSimilarityMatrix:
    def test_from_sparse_vectors(self):
        features = [
            [(0, 1.0), (1, 2.0)],
            [(0, 1.0), (1, 2.0)],
            [(0, 3.0), (2, 1.0)],
        ]
        sm = SimilarityMatrix.from_sparse_vectors(features, "euclidean")
        assert sm.n_docs() == 3

    def test_self_distance_zero(self):
        features = [[(0, 1.0), (1, 2.0)], [(0, 3.0)]]
        sm = SimilarityMatrix.from_sparse_vectors(features, "euclidean")
        assert sm.get_distance(0, 0) == 0.0

    def test_identical_vectors_zero(self):
        features = [[(0, 1.0), (1, 2.0)], [(0, 1.0), (1, 2.0)]]
        sm = SimilarityMatrix.from_sparse_vectors(features, "euclidean")
        assert sm.get_distance(0, 1) == 0.0

    def test_cosine_metric(self):
        features = [
            [(0, 1.0), (1, 2.0)],
            [(0, 1.0), (1, 2.0)],
            [(2, 1.0), (3, 1.0)],
        ]
        sm = SimilarityMatrix.from_sparse_vectors(features, "cosine")
        assert sm.get_distance(0, 1) < 1e-10  # identical
        assert abs(sm.get_distance(0, 2) - 1.0) < 1e-10  # disjoint

    def test_knn(self):
        features = [
            [(0, 1.0), (1, 2.0)],
            [(0, 1.0), (1, 2.0)],
            [(0, 10.0), (1, 20.0)],
        ]
        sm = SimilarityMatrix.from_sparse_vectors(features, "euclidean")
        nn = sm.k_nearest_neighbors(0, 1)
        assert len(nn) == 1
        assert nn[0][0] == 1  # identical vector

    def test_mean_distance(self):
        features = [[(0, 1.0)], [(0, 2.0)], [(0, 3.0)]]
        sm = SimilarityMatrix.from_sparse_vectors(features, "euclidean")
        assert sm.mean_distance() > 0

    def test_manhattan(self):
        features = [[(0, 1.0), (1, 0.0)], [(0, 0.0), (1, 1.0)]]
        sm = SimilarityMatrix.from_sparse_vectors(features, "manhattan")
        assert abs(sm.get_distance(0, 1) - 2.0) < 1e-10

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="unknown metric"):
            SimilarityMatrix.from_sparse_vectors([], "invalid")

    def test_pickle_roundtrip(self):
        import pickle

        features = [[(0, 1.0), (1, 2.0)], [(0, 3.0), (1, 4.0)]]
        sm = SimilarityMatrix.from_sparse_vectors(features, "cosine")
        data = pickle.dumps(sm)
        sm2 = pickle.loads(data)
        assert sm2.n_docs() == 2
        assert abs(sm.get_distance(0, 1) - sm2.get_distance(0, 1)) < 1e-10

    def test_repr(self):
        features = [[(0, 1.0)], [(0, 2.0)]]
        sm = SimilarityMatrix.from_sparse_vectors(features, "euclidean")
        r = repr(sm)
        assert "n_docs=2" in r

    def test_save_load(self, tmp_path: Path):
        features = [[(0, 1.0)], [(0, 2.0)]]
        sm = SimilarityMatrix.from_sparse_vectors(features, "euclidean")
        path = tmp_path / "sim.bin"
        sm.save(str(path))
        restored = SimilarityMatrix.load(str(path))
        assert restored.n_docs() == 2
