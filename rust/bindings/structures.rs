//! PyO3 bindings for data structures: vocabularies, inverted index.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use crate::core::structures::{
    inverted_index::{Bm25Params, IdfWeight, InvertedIndex, TfWeight},
    similarity_matrix::{DistanceMetric, SimilarityMatrix},
    span_index::{LabeledSpan, SpanIndex, SpanIndexError},
    sparse_term_matrix::SparseTermMatrix,
    vocabulary::{BloomVocabulary, FrequencyVocabulary, IndexedVocabulary, SetVocabulary},
};

use super::spans::{PyPostingEntry, PyScoredDoc};
use super::util::{
    bincode_getstate, bincode_setstate, load_bincode_from_path, save_bincode_to_path,
};

// =============================================================================
// Vocabularies
// =============================================================================

/// Fast set-based vocabulary for O(1) membership testing.
///
/// Args:
///     terms: Optional list of initial terms.
#[pyclass(name = "SetVocabulary", module = "kaos_nlp_core._rust.structures")]
struct PySetVocabulary {
    inner: SetVocabulary,
}

#[pymethods]
impl PySetVocabulary {
    #[new]
    #[pyo3(signature = (terms=None))]
    fn new(terms: Option<Vec<String>>) -> Self {
        let inner = match terms {
            Some(t) => SetVocabulary::build(t),
            None => SetVocabulary::new(),
        };
        Self { inner }
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }

    /// Insert a term. Returns True if the term was new, False if already present.
    fn insert(&mut self, term: &str) -> bool {
        self.inner.insert(term)
    }

    /// Check if a term exists in the vocabulary.
    fn contains(&self, term: &str) -> bool {
        self.inner.contains(term)
    }

    /// Remove a term. Returns True if it was present.
    fn remove(&mut self, term: &str) -> bool {
        self.inner.remove(term)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, term: &str) -> bool {
        self.inner.contains(term)
    }
}

/// Vocabulary with frequency counts and auto-assigned IDs.
#[pyclass(
    name = "FrequencyVocabulary",
    module = "kaos_nlp_core._rust.structures"
)]
struct PyFrequencyVocabulary {
    inner: FrequencyVocabulary,
}

#[pymethods]
impl PyFrequencyVocabulary {
    #[new]
    fn new() -> Self {
        Self {
            inner: FrequencyVocabulary::new(),
        }
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }

    /// Insert a term (increments count by 1). Returns the term's ID.
    fn insert(&mut self, term: &str) -> u32 {
        self.inner.insert(term)
    }

    /// Insert a term with a specific count. Returns the term's ID.
    fn insert_with_count(&mut self, term: &str, count: u64) -> u32 {
        self.inner.insert_with_count(term, count)
    }

    /// Check if a term exists.
    fn contains(&self, term: &str) -> bool {
        self.inner.contains(term)
    }

    /// Get the frequency count for a term, or None if absent.
    fn get_count(&self, term: &str) -> Option<u64> {
        self.inner.get_count(term)
    }

    /// Get the assigned ID for a term, or None if absent.
    fn get_id(&self, term: &str) -> Option<u32> {
        self.inner.get_id(term)
    }

    /// Return the top N terms by frequency as a list of (term, count) tuples.
    fn top_n(&self, py: Python<'_>, n: usize) -> PyResult<Py<PyList>> {
        let inner = &self.inner;
        let entries = py.detach(|| inner.top_n(n));
        let list = PyList::empty(py);
        for (term, count) in &entries {
            let t = pyo3::types::PyTuple::new(
                py,
                &[
                    term.into_pyobject(py)?.into_any(),
                    count.into_pyobject(py)?.into_any(),
                ],
            )?;
            list.append(t)?;
        }
        Ok(list.into())
    }

    /// Total count across all terms.
    fn total_count(&self) -> u64 {
        self.inner.total_count()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, term: &str) -> bool {
        self.inner.contains(term)
    }
}

/// Vocabulary with bidirectional term <-> ID lookup.
#[pyclass(name = "IndexedVocabulary", module = "kaos_nlp_core._rust.structures")]
struct PyIndexedVocabulary {
    inner: IndexedVocabulary,
}

#[pymethods]
impl PyIndexedVocabulary {
    #[new]
    fn new() -> Self {
        Self {
            inner: IndexedVocabulary::new(),
        }
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }

    /// Insert a term. Returns its ID (existing terms return their current ID).
    fn insert(&mut self, term: &str) -> u32 {
        self.inner.insert(term)
    }

    /// Get the ID for a term, or None if absent.
    fn get_id(&self, term: &str) -> Option<u32> {
        self.inner.get_id(term)
    }

    /// Get the term for an ID, or None if out of range.
    fn get_term(&self, id: u32) -> Option<String> {
        self.inner.get_term(id).map(|s| s.to_string())
    }

    /// Check if a term exists.
    fn contains(&self, term: &str) -> bool {
        self.inner.contains(term)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, term: &str) -> bool {
        self.inner.contains(term)
    }
}

/// Bloom filter vocabulary for approximate membership testing.
///
/// Uses much less memory than a set but has a small false positive rate.
/// No false negatives: if contains() returns False, the term is definitely absent.
///
/// Note: Pickle/serialization stores all inserted terms in a Vec and rebuilds
/// the bloom filter on deserialization (BloomFilter does not implement serde).
/// For very large vocabularies (millions of terms), this negates the bloom
/// filter's space advantage. Prefer SetVocabulary when serialization size matters.
///
/// Args:
///     expected_items: Expected number of items (default 10000).
///     false_positive_rate: Desired false positive rate (default 0.01).
#[pyclass(name = "BloomVocabulary", module = "kaos_nlp_core._rust.structures")]
struct PyBloomVocabulary {
    inner: BloomVocabulary,
    /// Keep inserted terms for pickle reconstruction.
    terms: Vec<String>,
    expected_items: usize,
    false_positive_rate: f64,
}

#[pymethods]
impl PyBloomVocabulary {
    #[new]
    #[pyo3(signature = (expected_items=10000, false_positive_rate=0.01))]
    fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        Self {
            inner: BloomVocabulary::new(expected_items, false_positive_rate),
            terms: Vec::new(),
            expected_items,
            false_positive_rate,
        }
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Store (expected_items, fpr, terms) as bincode
        let state = (self.expected_items, self.false_positive_rate, &self.terms);
        bincode_getstate(py, &state)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let py = state.py();
        let (expected_items, fpr, terms): (usize, f64, Vec<String>) = bincode_setstate(state)?;
        self.expected_items = expected_items;
        self.false_positive_rate = fpr;
        // Rebuild bloom filter without GIL (bulk insert can be slow)
        let mut bloom = BloomVocabulary::new(expected_items, fpr);
        let terms_ref = &terms;
        py.detach(|| {
            for t in terms_ref {
                bloom.insert(t);
            }
        });
        self.inner = bloom;
        self.terms = terms;
        Ok(())
    }

    /// Insert a term into the bloom filter.
    fn insert(&mut self, term: &str) {
        self.inner.insert(term);
        self.terms.push(term.to_string());
    }

    /// Check if a term is probably in the set. May return false positives.
    fn contains(&self, term: &str) -> bool {
        self.inner.contains(term)
    }

    /// Approximate number of items inserted.
    fn approx_len(&self) -> usize {
        self.inner.approx_len()
    }

    fn __contains__(&self, term: &str) -> bool {
        self.inner.contains(term)
    }
}

// =============================================================================
// Inverted Index
// =============================================================================

/// Parse TF weighting scheme from string.
fn parse_tf_weight(s: &str) -> PyResult<TfWeight> {
    match s {
        "raw" => Ok(TfWeight::Raw),
        "sublinear" | "log" => Ok(TfWeight::Sublinear),
        "boolean" | "bool" => Ok(TfWeight::Boolean),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown tf_weight: {s}. Expected: raw, sublinear, boolean"
        ))),
    }
}

/// Parse IDF weighting scheme from string.
fn parse_idf_weight(s: &str) -> PyResult<IdfWeight> {
    match s {
        "standard" => Ok(IdfWeight::Standard),
        "smooth" => Ok(IdfWeight::Smooth),
        "probabilistic" | "bm25" => Ok(IdfWeight::Probabilistic),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown idf_weight: {s}. Expected: standard, smooth, probabilistic"
        ))),
    }
}

/// Convert scored results to a `Vec<PyScoredDoc>` (typed pyclass).
///
/// Audit perf finding #1 / P3 — emitting `PyScoredDoc` directly skips
/// the dict→Python-dataclass round trip on the BM25 hot path.
fn scored_docs_to_pyclasses(
    results: &[crate::core::structures::inverted_index::ScoredDoc],
) -> Vec<PyScoredDoc> {
    results
        .iter()
        .map(|r| PyScoredDoc {
            doc_id: r.doc_id,
            score: r.score,
        })
        .collect()
}

/// Inverted index mapping terms to document posting lists.
///
/// Supports boolean AND/OR queries, TF-IDF scoring variants, BM25 ranking,
/// and top-K ranked retrieval. Supports pickle for multiprocessing.
#[pyclass(name = "InvertedIndex", module = "kaos_nlp_core._rust.structures")]
struct PyInvertedIndex {
    inner: InvertedIndex,
}

#[pymethods]
impl PyInvertedIndex {
    #[new]
    fn new() -> Self {
        Self {
            inner: InvertedIndex::new(),
        }
    }

    /// Load an `InvertedIndex` from disk.
    ///
    /// **Trusted-source only.** Files carry a `KNC1` magic header + format
    /// version and are bounded by `KAOS_NLP_MAX_LOAD_BYTES` (default 256 MiB).
    /// The header guards against truncated / mistyped files and the cap
    /// guards against unbounded allocation, but neither defends against an
    /// adversarial author of the file. Treat `.load()` like Python's
    /// `pickle.load()` — only call it on files you produced or trust.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_owned = path.to_string();
        let inner = py.detach(|| load_bincode_from_path::<InvertedIndex>(&path_owned));
        inner
            .map(|inner| Self { inner })
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }

    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let inner = &self.inner;
        let path_owned = path.to_string();
        py.detach(|| save_bincode_to_path(inner, &path_owned))
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Add a single term occurrence for a document.
    fn add(&mut self, term: &str, doc_id: u32) {
        self.inner.add(term, doc_id);
    }

    /// Index a full document (list of terms).
    fn add_document(&mut self, py: Python<'_>, doc_id: u32, terms: Vec<String>) {
        let inner = &mut self.inner;
        py.detach(|| {
            let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
            inner.add_document(doc_id, &refs);
        });
    }

    /// Number of documents containing the term.
    fn doc_freq(&self, term: &str) -> u32 {
        self.inner.doc_freq(term)
    }

    /// Inverse document frequency: ln(N / df).
    fn idf(&self, term: &str) -> f64 {
        self.inner.idf(term)
    }

    /// TF-IDF score for a term in a document (raw TF, standard IDF).
    fn tf_idf(&self, term: &str, doc_id: u32) -> f64 {
        self.inner.tf_idf(term, doc_id)
    }

    /// TF-IDF score with configurable weighting.
    #[pyo3(signature = (term, doc_id, tf_weight="raw", idf_weight="standard"))]
    fn tf_idf_weighted(
        &self,
        term: &str,
        doc_id: u32,
        tf_weight: &str,
        idf_weight: &str,
    ) -> PyResult<f64> {
        Ok(self.inner.tf_idf_weighted(
            term,
            doc_id,
            parse_tf_weight(tf_weight)?,
            parse_idf_weight(idf_weight)?,
        ))
    }

    /// Multi-term TF-IDF score for a document.
    #[pyo3(signature = (terms, doc_id, tf_weight="raw", idf_weight="standard"))]
    fn score_tf_idf(
        &self,
        py: Python<'_>,
        terms: Vec<String>,
        doc_id: u32,
        tf_weight: &str,
        idf_weight: &str,
    ) -> PyResult<f64> {
        let tw = parse_tf_weight(tf_weight)?;
        let iw = parse_idf_weight(idf_weight)?;
        let inner = &self.inner;
        Ok(py.detach(|| {
            let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
            inner.score_tf_idf(&refs, doc_id, tw, iw)
        }))
    }

    /// BM25 score for a multi-term query against a document.
    #[pyo3(signature = (terms, doc_id, k1=1.5, b=0.75))]
    fn score_bm25(&self, py: Python<'_>, terms: Vec<String>, doc_id: u32, k1: f64, b: f64) -> f64 {
        let inner = &self.inner;
        let params = Bm25Params { k1, b };
        py.detach(|| {
            let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
            inner.score_bm25(&refs, doc_id, &params)
        })
    }

    /// Ranked retrieval: top-K documents by BM25 score.
    #[pyo3(signature = (terms, top_k=10, k1=1.5, b=0.75))]
    fn query_bm25(
        &self,
        py: Python<'_>,
        terms: Vec<String>,
        top_k: usize,
        k1: f64,
        b: f64,
    ) -> Vec<PyScoredDoc> {
        let inner = &self.inner;
        let params = Bm25Params { k1, b };
        py.detach(|| {
            let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
            let results = inner.query_bm25(&refs, &params, top_k);
            scored_docs_to_pyclasses(&results)
        })
    }

    /// Ranked retrieval: top-K documents by TF-IDF score.
    #[pyo3(signature = (terms, top_k=10, tf_weight="sublinear", idf_weight="smooth"))]
    fn query_tf_idf(
        &self,
        py: Python<'_>,
        terms: Vec<String>,
        top_k: usize,
        tf_weight: &str,
        idf_weight: &str,
    ) -> PyResult<Vec<PyScoredDoc>> {
        let tw = parse_tf_weight(tf_weight)?;
        let iw = parse_idf_weight(idf_weight)?;
        let inner = &self.inner;
        Ok(py.detach(|| {
            let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
            let results = inner.query_tf_idf(&refs, tw, iw, top_k);
            scored_docs_to_pyclasses(&results)
        }))
    }

    /// Ranked retrieval over many queries in a single GIL release.
    ///
    /// Each query is a list of pre-tokenized terms. Queries run in
    /// parallel via Rayon — a single ``py.detach`` covers the whole
    /// batch, so multi-query workloads (RAG eval, BEIR runs) avoid
    /// the per-query GIL re-acquisition that ``search_batch`` paid in
    /// the Python loop. Audit perf finding #3a / P4 — measured 2-4×
    /// on >=8-core boxes.
    #[pyo3(signature = (queries, top_k=10, k1=1.5, b=0.75))]
    fn query_bm25_batch(
        &self,
        py: Python<'_>,
        queries: Vec<Vec<String>>,
        top_k: usize,
        k1: f64,
        b: f64,
    ) -> Vec<Vec<PyScoredDoc>> {
        use rayon::prelude::*;
        let inner = &self.inner;
        let params = Bm25Params { k1, b };
        py.detach(|| {
            queries
                .par_iter()
                .map(|terms| {
                    let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
                    let results = inner.query_bm25(&refs, &params, top_k);
                    scored_docs_to_pyclasses(&results)
                })
                .collect()
        })
    }

    /// Ranked retrieval over many queries by TF-IDF in a single GIL
    /// release. Same parallel shape as ``query_bm25_batch``.
    #[pyo3(signature = (queries, top_k=10, tf_weight="sublinear", idf_weight="smooth"))]
    fn query_tf_idf_batch(
        &self,
        py: Python<'_>,
        queries: Vec<Vec<String>>,
        top_k: usize,
        tf_weight: &str,
        idf_weight: &str,
    ) -> PyResult<Vec<Vec<PyScoredDoc>>> {
        use rayon::prelude::*;
        let tw = parse_tf_weight(tf_weight)?;
        let iw = parse_idf_weight(idf_weight)?;
        let inner = &self.inner;
        Ok(py.detach(|| {
            queries
                .par_iter()
                .map(|terms| {
                    let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
                    let results = inner.query_tf_idf(&refs, tw, iw, top_k);
                    scored_docs_to_pyclasses(&results)
                })
                .collect()
        }))
    }

    /// Boolean AND query: documents containing ALL terms.
    fn query_and(&self, py: Python<'_>, terms: Vec<String>) -> Vec<u32> {
        let inner = &self.inner;
        py.detach(|| {
            let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
            inner.query_and(&refs)
        })
    }

    /// Boolean OR query: documents containing ANY term.
    fn query_or(&self, py: Python<'_>, terms: Vec<String>) -> Vec<u32> {
        let inner = &self.inner;
        py.detach(|| {
            let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
            inner.query_or(&refs)
        })
    }

    /// Build an index from a batch of documents in parallel using rayon.
    ///
    /// Args:
    ///     documents: List of (doc_id, terms) tuples.
    ///
    /// Returns a new InvertedIndex built in parallel.
    #[staticmethod]
    fn build_batch(py: Python<'_>, documents: Vec<(u32, Vec<String>)>) -> Self {
        let inner = py.detach(|| InvertedIndex::build_batch(&documents));
        Self { inner }
    }

    /// Build an index by tokenizing each rendered document text in
    /// Rust under a single GIL release (parallel via Rayon).
    ///
    /// Audit perf finding #2 / P5 — fuses the prior Python-side tokenize
    /// loop, the ``Vec<(u32, Vec<String>)>`` materialization (which
    /// could reach GBs on large corpora), and the parallel
    /// ``build_batch`` into one ``py.detach`` scope. The tokenizer's
    /// configuration is inherited from the supplied ``Tokenizer``
    /// instance so callers don't have to re-pass each option.
    ///
    /// Args:
    ///     doc_ids: Document ids in the same order as ``texts``.
    ///     texts: Rendered document text (one per doc, post field-weight
    ///         expansion if any).
    ///     tokenizer: ``Tokenizer`` whose config (lowercase, prefix,
    ///         stopwords, index, keep_punctuation) is applied to each text.
    #[staticmethod]
    fn build_from_texts(
        py: Python<'_>,
        doc_ids: Vec<u32>,
        texts: Vec<String>,
        tokenizer: PyRef<'_, super::tokenizer::PyTokenizer>,
    ) -> PyResult<Self> {
        if doc_ids.len() != texts.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "doc_ids ({}) and texts ({}) must have the same length",
                doc_ids.len(),
                texts.len(),
            )));
        }
        let config = tokenizer.config.clone();
        let inner = py.detach(|| {
            use rayon::prelude::*;
            let documents: Vec<(u32, Vec<String>)> = doc_ids
                .par_iter()
                .zip(texts.par_iter())
                .map(|(doc_id, text)| {
                    (
                        *doc_id,
                        crate::core::tokenizer::tokenize_words(text, &config),
                    )
                })
                .collect();
            InvertedIndex::build_batch(&documents)
        });
        Ok(Self { inner })
    }

    /// Number of unique terms indexed.
    fn term_count(&self) -> usize {
        self.inner.term_count()
    }

    /// Number of documents indexed.
    fn doc_count(&self) -> u32 {
        self.inner.doc_count()
    }

    /// Length (total token count) of a document.
    fn doc_length(&self, doc_id: u32) -> u32 {
        self.inner.doc_length(doc_id)
    }

    /// Average document length across all indexed documents.
    fn avg_doc_length(&self) -> f64 {
        self.inner.avg_doc_length()
    }

    /// Get the posting list for a term as `PostingEntry` pyclasses
    /// (`doc_id`, `term_freq`).
    fn get_postings(&self, term: &str) -> Option<Vec<PyPostingEntry>> {
        self.inner.get(term).map(|postings| {
            postings
                .iter()
                .map(|p| PyPostingEntry {
                    doc_id: p.doc_id,
                    term_freq: p.term_freq,
                })
                .collect()
        })
    }
}

// =============================================================================
// SimilarityMatrix
// =============================================================================

/// Parse a distance metric string into the enum.
fn parse_metric(s: &str) -> PyResult<DistanceMetric> {
    match s {
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "cosine" => Ok(DistanceMetric::Cosine),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown metric: {s}. Expected: euclidean, cosine, manhattan"
        ))),
    }
}

/// Pairwise similarity/distance matrix (upper triangular storage).
///
/// Computed from sparse feature vectors using Euclidean, Cosine, or Manhattan distance.
#[pyclass(name = "SimilarityMatrix", module = "kaos_nlp_core._rust.structures")]
struct PySimilarityMatrix {
    inner: SimilarityMatrix,
}

#[pymethods]
impl PySimilarityMatrix {
    /// Build a similarity matrix from sparse feature vectors.
    ///
    /// Args:
    ///     features: List of sparse vectors, each as list of (term_id, value) tuples.
    ///     metric: Distance metric — "euclidean", "cosine", or "manhattan".
    #[staticmethod]
    #[pyo3(signature = (features, metric="cosine"))]
    fn from_sparse_vectors(
        py: Python<'_>,
        features: Vec<Vec<(u32, f64)>>,
        metric: &str,
    ) -> PyResult<Self> {
        let m = parse_metric(metric)?;
        let inner = py.detach(|| SimilarityMatrix::from_sparse_vectors(&features, m));
        Ok(Self { inner })
    }

    /// Load a `SimilarityMatrix` from disk.
    ///
    /// **Trusted-source only** — see `InvertedIndex.load` for the format
    /// + size-cap details.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_owned = path.to_string();
        let inner = py.detach(|| load_bincode_from_path::<SimilarityMatrix>(&path_owned));
        inner
            .map(|inner| Self { inner })
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Get distance between two documents.
    ///
    /// Raises `ValueError` when either index is out of range.
    fn get_distance(&self, i: usize, j: usize) -> PyResult<f64> {
        self.inner
            .get_distance(i, j)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get similarity between two documents.
    ///
    /// Raises `ValueError` when either index is out of range.
    fn get_similarity(&self, i: usize, j: usize) -> PyResult<f64> {
        self.inner
            .get_similarity(i, j)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Find K nearest neighbors for a document.
    ///
    /// Returns list of (doc_index, distance) tuples sorted by distance ascending.
    /// Raises `ValueError` when `doc_idx` is out of range.
    fn k_nearest_neighbors(
        &self,
        py: Python<'_>,
        doc_idx: usize,
        k: usize,
    ) -> PyResult<Vec<(usize, f64)>> {
        let inner = &self.inner;
        py.detach(|| inner.k_nearest_neighbors(doc_idx, k))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Number of documents.
    fn n_docs(&self) -> usize {
        self.inner.n_docs()
    }

    /// Average distance across all pairs.
    fn mean_distance(&self) -> f64 {
        self.inner.mean_distance()
    }

    fn __repr__(&self) -> String {
        format!("SimilarityMatrix(n_docs={})", self.inner.n_docs())
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }

    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let inner = &self.inner;
        let path_owned = path.to_string();
        py.detach(|| save_bincode_to_path(inner, &path_owned))
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let unpickle_fn = py
            .import("kaos_nlp_core.structures")?
            .getattr("_unpickle_similarity_matrix")?;
        let state = self.__getstate__(py)?;
        let args = (state,).into_pyobject(py)?.into_any().unbind();
        Ok((unpickle_fn.unbind(), args))
    }
}

/// Helper for SimilarityMatrix pickle deserialization.
#[pyfunction]
fn _similarity_matrix_unpickle(state: &Bound<'_, PyBytes>) -> PyResult<PySimilarityMatrix> {
    let inner: SimilarityMatrix = bincode_setstate(state)?;
    Ok(PySimilarityMatrix { inner })
}

// =============================================================================
// SparseTermMatrix
// =============================================================================

/// Compressed Sparse Row (CSR) document-term matrix.
///
/// Memory-efficient representation of document-term frequency data.
/// Supports TF-IDF weighting and cosine similarity between documents.
#[pyclass(name = "SparseTermMatrix", module = "kaos_nlp_core._rust.structures")]
struct PySparseTermMatrix {
    inner: SparseTermMatrix,
}

#[pymethods]
impl PySparseTermMatrix {
    #[new]
    fn new() -> Self {
        Self {
            inner: SparseTermMatrix::new(),
        }
    }

    /// Load a `SparseTermMatrix` from disk.
    ///
    /// **Trusted-source only** — see `InvertedIndex.load` for the format
    /// + size-cap details.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_owned = path.to_string();
        let inner = py.detach(|| load_bincode_from_path::<SparseTermMatrix>(&path_owned));
        inner
            .map(|inner| Self { inner })
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Add a document as a list of (term_id, frequency) pairs.
    fn add_document(&mut self, term_freqs: Vec<(u32, u32)>) {
        let mut pairs = term_freqs;
        self.inner.add_document(&mut pairs);
    }

    /// Get term frequency for a document and term.
    fn get_term_freq(&self, doc_id: usize, term_id: u32) -> u32 {
        self.inner.get_term_freq(doc_id, term_id)
    }

    /// Get all non-zero terms for a document as list of (term_id, frequency) tuples.
    ///
    /// Raises `ValueError` when `doc_id` is out of range.
    fn get_document_terms(&self, doc_id: usize) -> PyResult<Vec<(u32, u32)>> {
        self.inner
            .iter_document_terms(doc_id)
            .map(|it| it.collect())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Number of unique terms in a document.
    fn document_term_count(&self, doc_id: usize) -> usize {
        self.inner.document_term_count(doc_id)
    }

    /// Total tokens in a document (sum of frequencies).
    fn document_length(&self, doc_id: usize) -> u64 {
        self.inner.document_length(doc_id)
    }

    /// Cosine similarity between two documents using raw TF vectors.
    fn cosine_similarity(&self, py: Python<'_>, doc_a: usize, doc_b: usize) -> f64 {
        let inner = &self.inner;
        py.detach(|| inner.cosine_similarity(doc_a, doc_b))
    }

    /// Cosine similarity between two documents using TF-IDF weighted vectors.
    fn cosine_similarity_tfidf(
        &self,
        py: Python<'_>,
        doc_a: usize,
        doc_b: usize,
        doc_freqs: Vec<u32>,
    ) -> f64 {
        let inner = &self.inner;
        py.detach(|| inner.cosine_similarity_tfidf(doc_a, doc_b, &doc_freqs))
    }

    /// Number of documents.
    fn num_docs(&self) -> usize {
        self.inner.num_docs()
    }

    /// Number of terms (columns).
    fn num_terms(&self) -> usize {
        self.inner.num_terms()
    }

    /// Number of non-zero entries.
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Density: fraction of non-zero cells.
    fn density(&self) -> f64 {
        self.inner.density()
    }

    /// Estimated memory usage in bytes.
    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    fn __repr__(&self) -> String {
        format!(
            "SparseTermMatrix(num_docs={}, num_terms={}, nnz={})",
            self.inner.num_docs(),
            self.inner.num_terms(),
            self.inner.nnz()
        )
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }

    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let inner = &self.inner;
        let path_owned = path.to_string();
        py.detach(|| save_bincode_to_path(inner, &path_owned))
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }
}

// =============================================================================
// SpanIndex (P4)
// =============================================================================
//
// Wraps `core::structures::span_index::SpanIndex`. The Rust core stores u32
// offsets in the caller-supplied coordinate space (typically character
// offsets in a kaos-content document); the binding does NOT convert offsets
// because `SpanIndex` is content-agnostic — the caller decides whether the
// offsets are bytes, chars, paragraphs, or any other scalar.
//
// `label: u32` is opaque, same as `InvertedIndex` term IDs. The intended
// caller is `kaos_content.indexing.AnnotationIndex` (P4.5), which handles
// the AnnotationType ↔ u32 mapping Python-side.

/// One labeled half-open span `[start, end)`. Plain value object exposed
/// to Python as a frozen dataclass-like struct.
#[pyclass(
    name = "LabeledSpan",
    module = "kaos_nlp_core._rust.structures",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyLabeledSpan {
    inner: LabeledSpan,
}

#[pymethods]
impl PyLabeledSpan {
    #[new]
    #[pyo3(signature = (label, start, end, score = 1.0))]
    fn new(label: u32, start: u32, end: u32, score: f32) -> Self {
        Self {
            inner: LabeledSpan {
                label,
                start,
                end,
                score,
            },
        }
    }
    #[getter]
    fn label(&self) -> u32 {
        self.inner.label
    }
    #[getter]
    fn start(&self) -> u32 {
        self.inner.start
    }
    #[getter]
    fn end(&self) -> u32 {
        self.inner.end
    }
    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }
    fn __repr__(&self) -> String {
        format!(
            "LabeledSpan(label={}, start={}, end={}, score={})",
            self.inner.label, self.inner.start, self.inner.end, self.inner.score
        )
    }
    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }
    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }
}

#[pyclass(name = "SpanIndex", module = "kaos_nlp_core._rust.structures")]
struct PySpanIndex {
    inner: SpanIndex,
}

#[pymethods]
impl PySpanIndex {
    #[new]
    fn new() -> Self {
        Self {
            inner: SpanIndex::new(),
        }
    }

    /// Build an index from a list of (label, start, end, score) tuples.
    /// score defaults to 1.0 if the 4th element is omitted.
    #[staticmethod]
    fn from_tuples(spans: Vec<(u32, u32, u32, Option<f32>)>) -> PyResult<Self> {
        let labelled: Vec<LabeledSpan> = spans
            .into_iter()
            .map(|(label, start, end, score)| LabeledSpan {
                label,
                start,
                end,
                score: score.unwrap_or(1.0),
            })
            .collect();
        let inner = SpanIndex::bulk_build(labelled).map_err(span_err_to_py)?;
        Ok(Self { inner })
    }

    /// Append one span. Returns its stable id.
    #[pyo3(signature = (label, start, end, score=None))]
    fn add(&mut self, label: u32, start: u32, end: u32, score: Option<f32>) -> PyResult<u32> {
        self.inner
            .add(LabeledSpan {
                label,
                start,
                end,
                score: score.unwrap_or(1.0),
            })
            .map_err(span_err_to_py)
    }

    /// Number of live spans (excludes tombstoned slots).
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get a span by id.
    fn get(&self, id: u32) -> Option<PyLabeledSpan> {
        self.inner.get(id).map(|s| PyLabeledSpan { inner: *s })
    }

    /// Span ids whose [start, end) contains `offset`.
    fn containing(&mut self, offset: u32) -> Vec<u32> {
        self.inner.containing(offset).into_iter().collect()
    }

    /// Span ids overlapping [start, end).
    fn overlapping(&mut self, start: u32, end: u32) -> Vec<u32> {
        self.inner.overlapping(start, end).into_iter().collect()
    }

    /// Merge same-label spans within `gap` of each other.
    fn merge_adjacent(&mut self, label: u32, gap: u32) {
        self.inner.merge_adjacent(label, gap);
    }

    /// label_a minus label_b — returns the new sub-span ids that were
    /// appended (originals are NOT removed; caller may tombstone them).
    fn subtract(&mut self, label_a: u32, label_b: u32) -> Vec<u32> {
        self.inner.subtract(label_a, label_b)
    }

    /// Where labels in `label_set` overlap, keep the higher-scored span.
    /// Returns ids of new sub-spans emitted by splits.
    fn resolve_overlaps_by_score(&mut self, label_set: Vec<u32>) -> Vec<u32> {
        self.inner.resolve_overlaps_by_score(&label_set)
    }

    fn tombstone(&mut self, id: u32) -> bool {
        self.inner.tombstone(id)
    }

    fn freeze(&mut self) {
        self.inner.freeze();
    }

    fn compact(&mut self) -> Vec<u32> {
        self.inner.compact()
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }
    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }

    /// Save to disk via the standing trusted-source helper.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let inner = &self.inner;
        let path_owned = path.to_string();
        py.detach(|| save_bincode_to_path(inner, &path_owned))
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Load from disk. **Trusted-source only** — see `Lexicon.load` for
    /// the format + size-cap details.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_owned = path.to_string();
        let inner = py
            .detach(|| load_bincode_from_path::<SpanIndex>(&path_owned))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }
}

fn span_err_to_py(e: SpanIndexError) -> PyErr {
    match e {
        SpanIndexError::InvalidSpan { .. } => {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        }
        SpanIndexError::InvalidSpanId { .. } => {
            pyo3::exceptions::PyKeyError::new_err(e.to_string())
        }
    }
}

/// Register structures submodule.
pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "structures")?;

    m.add_class::<PySetVocabulary>()?;
    m.add_class::<PyFrequencyVocabulary>()?;
    m.add_class::<PyIndexedVocabulary>()?;
    m.add_class::<PyBloomVocabulary>()?;
    m.add_class::<PyInvertedIndex>()?;
    m.add_class::<PySimilarityMatrix>()?;
    m.add_class::<PySparseTermMatrix>()?;
    m.add_class::<PyLabeledSpan>()?;
    m.add_class::<PySpanIndex>()?;
    m.add_function(wrap_pyfunction!(_similarity_matrix_unpickle, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.structures", &m)?;

    Ok(())
}
