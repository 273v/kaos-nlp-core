//! PyO3 bindings for fuzzy hashing: MinHash/LSH, CTPH.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::core::ctph;
use crate::core::minhash::{self, DuplicateGroup, MinHashIndex, MinHashSignature, MinHasher};

use super::util::{
    bincode_getstate, bincode_setstate, load_bincode_from_path, save_bincode_to_path,
};

// =============================================================================
// MinHashSignature
// =============================================================================

/// A MinHash signature: fixed-size sketch of a set for Jaccard similarity estimation.
///
/// Created by MinHasher methods — not constructed directly.
#[pyclass(
    name = "MinHashSignature",
    module = "kaos_nlp_core._rust.hashing",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyMinHashSignature {
    inner: MinHashSignature,
}

#[pymethods]
impl PyMinHashSignature {
    #[new]
    fn new() -> Self {
        Self {
            inner: MinHashSignature { values: Vec::new() },
        }
    }

    /// Estimate Jaccard similarity with another signature.
    ///
    /// Raises `ValueError` if the two signatures disagree on permutation count.
    fn jaccard(&self, other: &PyMinHashSignature) -> PyResult<f64> {
        self.inner
            .jaccard(&other.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// The raw hash values (one per permutation).
    #[getter]
    fn values(&self) -> Vec<u64> {
        self.inner.values.clone()
    }

    fn __len__(&self) -> usize {
        self.inner.values.len()
    }

    fn __repr__(&self) -> String {
        format!("MinHashSignature(num_perm={})", self.inner.values.len())
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }
}

// =============================================================================
// MinHasher
// =============================================================================

/// MinHash hasher: generates MinHash signatures from sets of items.
///
/// Args:
///     num_perm: Number of hash permutations (default 128). More = more accurate.
///     seed: Base seed for reproducibility (default 42).
#[pyclass(name = "MinHasher", module = "kaos_nlp_core._rust.hashing")]
struct PyMinHasher {
    inner: MinHasher,
    seed: u64,
}

#[pymethods]
impl PyMinHasher {
    #[new]
    #[pyo3(signature = (num_perm=128, seed=42))]
    fn new(num_perm: usize, seed: u64) -> Self {
        Self {
            inner: MinHasher::with_seed(num_perm, seed),
            seed,
        }
    }

    /// Compute MinHash signature from a set of string items.
    fn hash_set(&self, py: Python<'_>, items: Vec<String>) -> PyMinHashSignature {
        let inner = &self.inner;
        let sig = py.detach(|| {
            let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
            inner.hash_set(refs.iter().copied())
        });
        PyMinHashSignature { inner: sig }
    }

    /// Compute MinHash signature from character shingles of a text.
    ///
    /// Args:
    ///     text: Input text.
    ///     shingle_size: Number of characters per shingle (e.g., 3 for trigrams).
    fn hash_char_shingles(
        &self,
        py: Python<'_>,
        text: &str,
        shingle_size: usize,
    ) -> PyMinHashSignature {
        let inner = &self.inner;
        let owned = text.to_owned();
        let sig = py.detach(|| inner.hash_char_shingles(&owned, shingle_size));
        PyMinHashSignature { inner: sig }
    }

    /// Compute MinHash signature from token (word) shingles.
    ///
    /// Args:
    ///     tokens: List of token strings.
    ///     shingle_size: Number of consecutive tokens per shingle (e.g., 2 for bigrams).
    fn hash_token_shingles(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        shingle_size: usize,
    ) -> PyMinHashSignature {
        let inner = &self.inner;
        let sig = py.detach(|| {
            let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
            inner.hash_token_shingles(&refs, shingle_size)
        });
        PyMinHashSignature { inner: sig }
    }

    /// Number of permutations (signature size).
    #[getter]
    fn num_perm(&self) -> usize {
        self.inner.num_perm()
    }

    fn __repr__(&self) -> String {
        format!(
            "MinHasher(num_perm={}, seed={})",
            self.inner.num_perm(),
            self.seed
        )
    }

    fn __getnewargs__(&self) -> (usize, u64) {
        (self.inner.num_perm(), self.seed)
    }
}

// =============================================================================
// MinHashIndex (LSH)
// =============================================================================

/// LSH index for approximate nearest-neighbor queries on MinHash signatures.
///
/// Uses band/row decomposition. Two items are candidates if they share
/// all hash values in any band.
///
/// Args:
///     bands: Number of bands.
///     rows: Number of rows per band. bands * rows must equal num_perm.
#[pyclass(name = "MinHashIndex", module = "kaos_nlp_core._rust.hashing")]
struct PyMinHashIndex {
    inner: MinHashIndex,
}

#[pymethods]
impl PyMinHashIndex {
    #[new]
    #[pyo3(signature = (bands=1, rows=1))]
    fn new(bands: usize, rows: usize) -> Self {
        Self {
            inner: MinHashIndex::new(bands, rows),
        }
    }

    /// Load a MinHashIndex from disk.
    ///
    /// **Trusted-source only.** See `kaos_nlp_core._rust.lexicon.Lexicon.load`
    /// for the format + size-cap details.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_owned = path.to_string();
        let inner: MinHashIndex = py
            .detach(|| load_bincode_from_path(&path_owned))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Create an index tuned for a target similarity threshold.
    ///
    /// Args:
    ///     num_perm: Signature size (e.g., 128).
    ///     threshold: Target Jaccard similarity threshold (e.g., 0.5).
    #[staticmethod]
    fn with_threshold(num_perm: usize, threshold: f64) -> Self {
        Self {
            inner: MinHashIndex::with_threshold(num_perm, threshold),
        }
    }

    /// Insert a document's MinHash signature.
    ///
    /// Raises `ValueError` if the signature size does not equal `bands * rows`.
    fn insert(&mut self, doc_id: u32, signature: &PyMinHashSignature) -> PyResult<()> {
        self.inner
            .insert(doc_id, &signature.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Query for candidate near-duplicates (approximate).
    ///
    /// Raises `ValueError` if the signature size does not match the index.
    fn query_candidates(
        &self,
        py: Python<'_>,
        signature: &PyMinHashSignature,
    ) -> PyResult<Vec<u32>> {
        let inner = &self.inner;
        let sig = &signature.inner;
        py.detach(|| inner.query_candidates(sig))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Query and filter by actual estimated Jaccard similarity.
    ///
    /// Returns list of (doc_id, estimated_jaccard) tuples above the threshold,
    /// sorted by similarity descending. Raises `ValueError` if the signature
    /// size does not match the index.
    fn query_above_threshold(
        &self,
        py: Python<'_>,
        signature: &PyMinHashSignature,
        threshold: f64,
    ) -> PyResult<Vec<(u32, f64)>> {
        let inner = &self.inner;
        let sig = &signature.inner;
        py.detach(|| inner.query_above_threshold(sig, threshold))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get a stored signature by doc_id, or None.
    fn get_signature(&self, doc_id: u32) -> Option<PyMinHashSignature> {
        self.inner
            .get_signature(doc_id)
            .map(|s| PyMinHashSignature { inner: s.clone() })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "MinHashIndex(num_perm={}, len={})",
            self.inner.num_perm(),
            self.inner.len()
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
        let path_owned = path.to_string();
        let inner = &self.inner;
        py.detach(|| save_bincode_to_path(inner, &path_owned))
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }
}

// =============================================================================
// DuplicateGroup
// =============================================================================

/// A group of near-duplicate documents found by find_duplicates.
#[pyclass(name = "DuplicateGroup", module = "kaos_nlp_core._rust.hashing")]
struct PyDuplicateGroup {
    inner: DuplicateGroup,
}

#[pymethods]
impl PyDuplicateGroup {
    /// The canonical (first-seen) document ID.
    #[getter]
    fn canonical_id(&self) -> u32 {
        self.inner.canonical_id
    }

    /// Duplicate document IDs with their similarity to the canonical.
    #[getter]
    fn duplicates(&self) -> Vec<(u32, f64)> {
        self.inner.duplicates.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DuplicateGroup(canonical_id={}, num_duplicates={})",
            self.inner.canonical_id,
            self.inner.duplicates.len()
        )
    }
}

// =============================================================================
// Standalone functions
// =============================================================================

/// Find duplicate groups in a set of documents.
///
/// Args:
///     hasher: MinHasher instance.
///     documents: List of (doc_id, tokens) tuples.
///     shingle_size: Shingle size for token shingles.
///     threshold: Jaccard similarity threshold for duplicates.
///
/// Returns:
///     List of DuplicateGroup objects.
#[pyfunction]
#[pyo3(signature = (hasher, documents, shingle_size=2, threshold=0.5))]
fn find_duplicates(
    py: Python<'_>,
    hasher: &PyMinHasher,
    documents: Vec<(u32, Vec<String>)>,
    shingle_size: usize,
    threshold: f64,
) -> Vec<PyDuplicateGroup> {
    let inner_hasher = &hasher.inner;

    let groups = py.detach(|| {
        let docs: Vec<(u32, Vec<&str>)> = documents
            .iter()
            .map(|(id, tokens)| (*id, tokens.iter().map(|s| s.as_str()).collect()))
            .collect();
        minhash::find_duplicates(inner_hasher, &docs, shingle_size, threshold)
    });

    groups
        .into_iter()
        .map(|g| PyDuplicateGroup { inner: g })
        .collect()
}

// =============================================================================
// CTPHDigest
// =============================================================================

/// A CTPH digest: typed representation of a context-triggered piecewise hash.
///
/// Two digests are comparable only if they share the same window_size and digest_size.
#[pyclass(name = "CTPHDigest", module = "kaos_nlp_core._rust.hashing")]
struct PyCTPHDigest {
    inner: ctph::CTPHDigest,
}

#[pymethods]
impl PyCTPHDigest {
    #[new]
    fn new() -> Self {
        Self {
            inner: ctph::CTPHDigest {
                window_size: 0,
                digest_size: 0,
                blocks: Vec::new(),
                piece_hex_len: 8,
            },
        }
    }

    /// Block-level Jaccard similarity (strict, ssdeep-style).
    ///
    /// Each block is a compound string of concatenated piece hashes.
    /// A single changed piece invalidates the entire block.
    /// Best for exact copy detection.
    fn similarity(&self, other: &PyCTPHDigest) -> f64 {
        self.inner.similarity(&other.inner)
    }

    /// Piece-level Jaccard similarity (tolerant, edit-aware).
    ///
    /// Splits compound blocks into individual piece hashes and compares those.
    /// A single changed piece only affects that piece, not the whole block.
    /// **Use this for document versioning and edit detection** (Word/PDF with
    /// minor changes, contract revisions, etc.).
    fn piece_similarity(&self, other: &PyCTPHDigest) -> f64 {
        self.inner.piece_similarity(&other.inner)
    }

    /// Number of individual pieces across all blocks.
    #[getter]
    fn num_pieces(&self) -> usize {
        self.inner.num_pieces()
    }

    #[getter]
    fn window_size(&self) -> usize {
        self.inner.window_size
    }

    #[getter]
    fn digest_size(&self) -> usize {
        self.inner.digest_size
    }

    #[getter]
    fn blocks(&self) -> Vec<String> {
        self.inner.blocks.clone()
    }

    fn __str__(&self) -> String {
        self.inner.to_string_repr()
    }

    fn __repr__(&self) -> String {
        format!(
            "CTPHDigest(window_size={}, digest_size={}, num_blocks={})",
            self.inner.window_size,
            self.inner.digest_size,
            self.inner.blocks.len()
        )
    }

    /// Parse a digest from its string representation.
    #[staticmethod]
    fn from_string(s: &str) -> PyResult<Self> {
        ctph::CTPHDigest::from_string(s)
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
}

// =============================================================================
// CTPH (byte-level)
// =============================================================================

/// Context-Triggered Piecewise Hashing for byte/text data.
///
/// Uses a rolling hash to identify block boundaries, then hashes each block
/// with blake3. The result is compared via Jaccard similarity over blocks.
///
/// Args:
///     window_size: Rolling hash window size (default 64).
///     digest_size: Block boundary trigger modulus (default 8).
///     precision: Blake3 output bytes per piece: 1, 2, 4, or 8 (default 4).
#[pyclass(name = "CTPH", module = "kaos_nlp_core._rust.hashing")]
struct PyCTPH {
    inner: ctph::CTPH,
}

#[pymethods]
impl PyCTPH {
    #[new]
    #[pyo3(signature = (window_size=64, digest_size=8, precision=4))]
    fn new(window_size: usize, digest_size: usize, precision: u8) -> Self {
        Self {
            inner: ctph::CTPH::new(window_size, digest_size, precision),
        }
    }

    /// Compute CTPH digest of byte data.
    fn compute(&self, py: Python<'_>, data: &[u8]) -> PyCTPHDigest {
        let inner = &self.inner;
        let owned = data.to_vec();
        let digest = py.detach(|| inner.compute(&owned));
        PyCTPHDigest { inner: digest }
    }

    /// Compute CTPH digest of a string (hashes as UTF-8 bytes).
    fn hash_str(&self, py: Python<'_>, text: &str) -> PyCTPHDigest {
        let inner = &self.inner;
        let owned = text.to_owned();
        let digest = py.detach(|| inner.hash_str(&owned));
        PyCTPHDigest { inner: digest }
    }

    fn __repr__(&self) -> String {
        format!(
            "CTPH(window_size={}, digest_size={}, precision={})",
            self.inner.window_size, self.inner.digest_size, self.inner.precision
        )
    }

    fn __getnewargs__(&self) -> (usize, usize, u8) {
        (
            self.inner.window_size,
            self.inner.digest_size,
            self.inner.precision,
        )
    }
}

// =============================================================================
// TokenCTPH
// =============================================================================

/// Context-Triggered Piecewise Hashing for integer token arrays.
///
/// Designed for LLM tokenizer outputs. Uses blake3 for piece hashing.
///
/// Args:
///     window_size: Rolling hash window size (default 4).
///     digest_size: Block boundary trigger modulus (default 8).
#[pyclass(name = "TokenCTPH", module = "kaos_nlp_core._rust.hashing")]
struct PyTokenCTPH {
    inner: ctph::TokenCTPH,
}

#[pymethods]
impl PyTokenCTPH {
    #[new]
    #[pyo3(signature = (window_size=4, digest_size=8))]
    fn new(window_size: usize, digest_size: usize) -> Self {
        Self {
            inner: ctph::TokenCTPH::new(window_size, digest_size),
        }
    }

    /// Compute CTPH digest of a token array.
    fn compute(&self, py: Python<'_>, tokens: Vec<i64>) -> PyCTPHDigest {
        let inner = &self.inner;
        let digest = py.detach(|| inner.compute(&tokens));
        PyCTPHDigest { inner: digest }
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenCTPH(window_size={}, digest_size={})",
            self.inner.window_size, self.inner.digest_size
        )
    }

    fn __getnewargs__(&self) -> (usize, usize) {
        (self.inner.window_size, self.inner.digest_size)
    }
}

// =============================================================================
// CTPH standalone functions
// =============================================================================

/// Compute CTPH hash of byte data, returning the string representation.
#[pyfunction]
#[pyo3(signature = (data, window_size=64, digest_size=8, precision=4))]
fn ctph_hash_bytes(
    py: Python<'_>,
    data: &[u8],
    window_size: usize,
    digest_size: usize,
    precision: u8,
) -> String {
    let owned = data.to_vec();
    py.detach(|| ctph::hash_bytes(&owned, window_size, digest_size, precision))
}

/// Compute CTPH hash of a string, returning the string representation.
#[pyfunction]
#[pyo3(signature = (text, window_size=64, digest_size=8, precision=4))]
fn ctph_hash_str(
    py: Python<'_>,
    text: &str,
    window_size: usize,
    digest_size: usize,
    precision: u8,
) -> String {
    let owned = text.to_owned();
    py.detach(|| ctph::hash_str(&owned, window_size, digest_size, precision))
}

/// Compare two CTPH hash strings and return Jaccard similarity.
#[pyfunction]
fn ctph_similarity(hash1: &str, hash2: &str) -> f64 {
    ctph::similarity(hash1, hash2)
}

/// Piece-level similarity between two CTPH digest strings.
///
/// More tolerant than ctph_similarity — compares individual piece hashes
/// rather than compound blocks. Best for document versioning/edit detection.
#[pyfunction]
fn ctph_piece_similarity(hash1: &str, hash2: &str) -> f64 {
    match (
        ctph::CTPHDigest::from_string(hash1),
        ctph::CTPHDigest::from_string(hash2),
    ) {
        (Ok(d1), Ok(d2)) => d1.piece_similarity(&d2),
        _ => 0.0,
    }
}

/// Compute token CTPH hash, returning the string representation.
#[pyfunction]
#[pyo3(signature = (tokens, window_size=4, digest_size=8))]
fn token_ctph_hash(
    py: Python<'_>,
    tokens: Vec<i64>,
    window_size: usize,
    digest_size: usize,
) -> String {
    py.detach(|| ctph::hash_tokens(&tokens, window_size, digest_size))
}

// =============================================================================
// Module registration
// =============================================================================

pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "hashing")?;

    // MinHash / LSH
    m.add_class::<PyMinHashSignature>()?;
    m.add_class::<PyMinHasher>()?;
    m.add_class::<PyMinHashIndex>()?;
    m.add_class::<PyDuplicateGroup>()?;
    m.add_function(wrap_pyfunction!(find_duplicates, &m)?)?;

    // CTPH
    m.add_class::<PyCTPHDigest>()?;
    m.add_class::<PyCTPH>()?;
    m.add_class::<PyTokenCTPH>()?;
    m.add_function(wrap_pyfunction!(ctph_hash_bytes, &m)?)?;
    m.add_function(wrap_pyfunction!(ctph_hash_str, &m)?)?;
    m.add_function(wrap_pyfunction!(ctph_similarity, &m)?)?;
    m.add_function(wrap_pyfunction!(ctph_piece_similarity, &m)?)?;
    m.add_function(wrap_pyfunction!(token_ctph_hash, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.hashing", &m)?;

    Ok(())
}
