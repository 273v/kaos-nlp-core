//! PyO3 bindings for `crate::core::similarity` — dense-vector distance
//! / similarity primitives.
//!
//! Exposes a small, focused surface:
//!
//! ```text
//! kaos_nlp_core._rust.similarity
//!   ├── cosine(a, b) -> float
//!   ├── cosine_one_to_many(query, matrix) -> np.ndarray
//!   ├── cosine_adjacent(matrix) -> np.ndarray
//!   ├── top_k_cosine(query, matrix, k) -> (np.ndarray, np.ndarray)
//!   ├── mmr_select(matrix, relevance, k, lambda_) -> (np.ndarray, np.ndarray)
//!   ├── knn_graph(matrix, k, include_self, normalized) -> (np.ndarray, np.ndarray)
//!   ├── near_duplicates(matrix, threshold, normalized, max_pairs) -> (np.ndarray, np.ndarray, bool)
//!   └── l2_normalize_in_place(vector) -> bool
//! ```
//!
//! All numpy inputs must be contiguous `float32` arrays. `cosine_*`
//! and `top_k_cosine` accept `PyReadonlyArray1<f32>` (query) and
//! `PyReadonlyArray2<f32>` (matrix), do their compute through
//! NumKong-dispatched SIMD kernels with the GIL released, and hand
//! back fresh numpy arrays via `IntoPyArray`.
//!
//! The GIL is released around every non-trivial computation via
//! `py.detach`, matching the repo convention in `bindings/structures.rs`
//! (the `detach`/`allow_threads` patterns there). Single-pair cosine
//! is short enough that GIL release isn't worth the overhead.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::core::similarity::{
    cosine as core_cosine, cosine_adjacent as core_cosine_adjacent,
    cosine_adjacent_normalized as core_cosine_adjacent_normalized,
    cosine_normalized as core_cosine_normalized, cosine_one_to_many as core_cosine_one_to_many,
    cosine_one_to_many_normalized as core_cosine_one_to_many_normalized,
    knn_graph as core_knn_graph, l2_normalize_in_place as core_l2_normalize,
    mmr_select as core_mmr_select, near_duplicates as core_near_duplicates,
    top_k_cosine as core_top_k_cosine, SimilarityError,
};

// ----------------------------------------------------------------------------
// Error mapping
// ----------------------------------------------------------------------------

fn map_err(e: SimilarityError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ----------------------------------------------------------------------------
// cosine(a, b)
// ----------------------------------------------------------------------------

/// Cosine similarity between two equal-length float32 vectors.
///
/// Args:
///     a: 1-D contiguous numpy float32 array.
///     b: 1-D contiguous numpy float32 array of the same length.
///
/// Returns:
///     float in [-1.0, 1.0].
///
/// Raises:
///     ValueError: empty input or length mismatch.
#[pyfunction]
fn cosine(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    core_cosine(a_slice, b_slice).map_err(map_err)
}

// ----------------------------------------------------------------------------
// cosine_normalized(a, b)
// ----------------------------------------------------------------------------

/// Cosine similarity assuming both inputs are unit-norm.
///
/// Pre-normalised fast path: skips both squared-norm computations and
/// the final ``sqrt`` — cosine of two unit-norm vectors is their dot
/// product, clamped to ``[-1, 1]``.
///
/// **Caller contract**: ``a`` and ``b`` must already be unit-norm.
/// The function does not verify this; violating the contract silently
/// returns a value that does not correspond to the true cosine.
#[pyfunction]
fn cosine_normalized(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    core_cosine_normalized(a_slice, b_slice).map_err(map_err)
}

// ----------------------------------------------------------------------------
// cosine_one_to_many(query, matrix)
// ----------------------------------------------------------------------------

/// Cosine similarity of one query vector against every row of a 2-D
/// matrix.
///
/// Args:
///     query: 1-D contiguous float32 array of length ``dim``.
///     matrix: 2-D contiguous float32 array of shape ``(n_rows, dim)``.
///
/// Returns:
///     1-D float32 numpy array of length ``n_rows`` with the cosine
///     similarity between ``query`` and each row.
#[pyfunction]
fn cosine_one_to_many<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'_, f32>,
    matrix: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query_slice = query.as_slice()?;
    let m_view = matrix.as_array();
    let dim = m_view.shape()[1];
    // Materialise a contiguous slice; PyReadonlyArray2 is already
    // row-major because numpy default order is C, and we required
    // contiguity above via `as_slice` on related arrays. Here we
    // take ownership of a copy only if the array is not contiguous;
    // otherwise reuse the underlying buffer.
    let matrix_slice = matrix.as_slice()?;
    let result = py
        .detach(|| core_cosine_one_to_many(query_slice, matrix_slice, dim))
        .map_err(map_err)?;
    Ok(result.into_pyarray(py))
}

// ----------------------------------------------------------------------------
// cosine_one_to_many_normalized(query, matrix)
// ----------------------------------------------------------------------------

/// Cosine of one **unit-norm** query against every row of a 2-D matrix
/// of **unit-norm** rows.
///
/// Pre-normalised fast path: each row reduces to a single SIMD dot
/// product clamped to ``[-1, 1]``. Saves both the per-call query norm
/// and every per-row row norm.
///
/// **Caller contract**: both ``query`` and every row of ``matrix``
/// must be unit-norm. Use this when the upstream embedder already
/// L2-normalises (which the kaos-nlp-transformers ``EmbeddingModel``
/// does — SemanticChunker and ExtractiveRanker should route through
/// here).
#[pyfunction]
fn cosine_one_to_many_normalized<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'_, f32>,
    matrix: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query_slice = query.as_slice()?;
    let m_view = matrix.as_array();
    let dim = m_view.shape()[1];
    let matrix_slice = matrix.as_slice()?;
    let result = py
        .detach(|| core_cosine_one_to_many_normalized(query_slice, matrix_slice, dim))
        .map_err(map_err)?;
    Ok(result.into_pyarray(py))
}

// ----------------------------------------------------------------------------
// cosine_adjacent(matrix)
// ----------------------------------------------------------------------------

/// Cosine similarity between each pair of adjacent rows in a 2-D
/// matrix. Used by the semantic chunker.
///
/// Args:
///     matrix: 2-D contiguous float32 array of shape ``(n_rows, dim)``.
///         When ``n_rows < 2`` the result is an empty array.
///
/// Returns:
///     1-D float32 numpy array of length ``n_rows - 1`` with
///     ``cosine(matrix[i], matrix[i+1])`` for each adjacent pair.
#[pyfunction]
fn cosine_adjacent<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let dim = matrix.as_array().shape()[1];
    let matrix_slice = matrix.as_slice()?;
    let result = py
        .detach(|| core_cosine_adjacent(matrix_slice, dim))
        .map_err(map_err)?;
    Ok(result.into_pyarray(py))
}

// ----------------------------------------------------------------------------
// cosine_adjacent_normalized(matrix)
// ----------------------------------------------------------------------------

/// Adjacent-row cosine on a matrix of **unit-norm** rows.
///
/// Pre-normalised fast path: each adjacent pair reduces to a single
/// SIMD dot product clamped to ``[-1, 1]``.
///
/// **Caller contract**: every row of ``matrix`` must be unit-norm.
/// SemanticChunker (kaos-nlp-transformers) feeds unit-norm embeddings;
/// this is its fast path.
#[pyfunction]
fn cosine_adjacent_normalized<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let dim = matrix.as_array().shape()[1];
    let matrix_slice = matrix.as_slice()?;
    let result = py
        .detach(|| core_cosine_adjacent_normalized(matrix_slice, dim))
        .map_err(map_err)?;
    Ok(result.into_pyarray(py))
}

// ----------------------------------------------------------------------------
// top_k_cosine(query, matrix, k)
// ----------------------------------------------------------------------------

/// Top-k cosine retrieval: return the indices and scores of the
/// ``k`` rows of ``matrix`` most similar to ``query``, ordered by
/// score descending with ascending-index tie-break.
///
/// Args:
///     query: 1-D contiguous float32 array.
///     matrix: 2-D contiguous float32 array of shape ``(n_rows, dim)``.
///     k: number of rows to return.
///
/// Returns:
///     Tuple of ``(indices, scores)``:
///       - ``indices``: 1-D uint32 numpy array.
///       - ``scores``: 1-D float32 numpy array.
#[pyfunction]
fn top_k_cosine<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'_, f32>,
    matrix: PyReadonlyArray2<'_, f32>,
    k: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let query_slice = query.as_slice()?;
    let dim = matrix.as_array().shape()[1];
    let matrix_slice = matrix.as_slice()?;
    let result = py
        .detach(|| core_top_k_cosine(query_slice, matrix_slice, dim, k))
        .map_err(map_err)?;
    let indices = result.indices.into_pyarray(py);
    let scores = result.scores.into_pyarray(py);
    PyTuple::new(py, [indices.into_any(), scores.into_any()])
}

// ----------------------------------------------------------------------------
// mmr_select(matrix, relevance, k, lambda_)
// ----------------------------------------------------------------------------

/// Maximal Marginal Relevance selection over a dense matrix.
///
/// Greedy reranker: balance per-row ``relevance`` against pairwise
/// cosine within the chosen set. Returns the indices of the selected
/// rows in pick order along with the MMR scores at pick time.
///
/// Args:
///     matrix: 2-D contiguous float32 array of shape ``(n_rows, dim)``.
///     relevance: 1-D contiguous float32 array of length ``n_rows``.
///     k: number of picks. Capped at ``n_rows``.
///     lambda_: diversity weight in ``[0.0, 1.0]``. ``1.0`` =
///         rank by relevance only; ``0.0`` = first pick is
///         ``argmax(relevance)`` then maximally diversify.
///
/// Returns:
///     Tuple of ``(indices, scores)``:
///       - ``indices``: 1-D uint32 numpy array.
///       - ``scores``: 1-D float32 numpy array, MMR scores at pick
///         time.
#[pyfunction]
#[pyo3(signature = (matrix, relevance, k, lambda_=0.5_f32))]
fn mmr_select<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'_, f32>,
    relevance: PyReadonlyArray1<'_, f32>,
    k: usize,
    lambda_: f32,
) -> PyResult<Bound<'py, PyTuple>> {
    let dim = matrix.as_array().shape()[1];
    let matrix_slice = matrix.as_slice()?;
    let relevance_slice = relevance.as_slice()?;
    let result = py
        .detach(|| core_mmr_select(matrix_slice, dim, relevance_slice, k, lambda_))
        .map_err(map_err)?;
    let indices = result.indices.into_pyarray(py);
    let scores = result.scores.into_pyarray(py);
    PyTuple::new(py, [indices.into_any(), scores.into_any()])
}

// ----------------------------------------------------------------------------
// l2_normalize_in_place(vec)
// ----------------------------------------------------------------------------

/// L2-normalise a 1-D float32 array **in place**.
///
/// Args:
///     vector: 1-D contiguous float32 array. Mutated to its unit
///         vector unless its norm is zero, in which case it is left
///         untouched.
///
/// Returns:
///     bool: ``True`` if the vector was normalised, ``False`` if it
///     had zero norm.
#[pyfunction]
fn l2_normalize_in_place(vector: &Bound<'_, PyArray1<f32>>) -> PyResult<bool> {
    // Safety: we own a unique writeable view for the duration of the
    // call (the PyArray1 was passed exclusively to us; Python doesn't
    // hand out aliasing mutable references to numpy arrays).
    let py = vector.py();
    let mut writeable = vector.readwrite();
    let slice = writeable.as_slice_mut()?;
    py.detach(|| core_l2_normalize(slice)).map_err(map_err)
}

// ----------------------------------------------------------------------------
// knn_graph(matrix, k, include_self, normalized)
// ----------------------------------------------------------------------------

/// k-nearest-neighbour graph: for every row of ``matrix``, the indices and
/// cosine scores of its ``k`` most-similar rows.
///
/// Args:
///     matrix: 2-D C-contiguous float32 array of shape ``(n_rows, dim)``.
///     k: neighbours per row. Capped at the number available
///         (``n_rows - 1`` when ``include_self`` is false).
///     include_self: when false (default), a row is never its own
///         neighbour.
///     normalized: when true, take the unit-norm fast path (caller
///         guarantees every row is L2-unit-norm).
///
/// Returns:
///     Tuple of ``(indices, scores)``, both 2-D arrays of shape
///     ``(n_rows, effective_k)``:
///       - ``indices``: ``uint32``; padded slots carry ``u32::MAX``.
///       - ``scores``: ``float32``; padded slots carry ``NaN``.
#[pyfunction]
#[pyo3(signature = (matrix, k, include_self=false, normalized=false))]
fn knn_graph<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'_, f32>,
    k: usize,
    include_self: bool,
    normalized: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let dim = matrix.as_array().shape()[1];
    let matrix_slice = matrix.as_slice()?;
    let graph = py
        .detach(|| core_knn_graph(matrix_slice, dim, k, include_self, normalized))
        .map_err(map_err)?;
    let shape = (graph.n_rows, graph.k);
    let indices = Array2::from_shape_vec(shape, graph.indices)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    let scores = Array2::from_shape_vec(shape, graph.scores)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    PyTuple::new(py, [indices.into_any(), scores.into_any()])
}

// ----------------------------------------------------------------------------
// near_duplicates(matrix, threshold, normalized, max_pairs)
// ----------------------------------------------------------------------------

/// All upper-triangle row pairs ``(i, j)``, ``i < j``, with cosine
/// similarity ``>= threshold``.
///
/// Args:
///     matrix: 2-D C-contiguous float32 array of shape ``(n_rows, dim)``.
///     threshold: minimum cosine for a pair to be reported.
///     normalized: unit-norm fast path (same contract as ``knn_graph``).
///     max_pairs: optional cap on the number of returned pairs.
///
/// Returns:
///     Tuple ``(pairs, scores, truncated)``:
///       - ``pairs``: ``uint32`` 2-D array of shape ``(m, 2)`` in
///         ``(i, j)`` lexicographic order.
///       - ``scores``: ``float32`` 1-D array of length ``m``.
///       - ``truncated``: ``bool``, true when ``max_pairs`` clamped the
///         output.
#[pyfunction]
#[pyo3(signature = (matrix, threshold, normalized=false, max_pairs=None))]
fn near_duplicates<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'_, f32>,
    threshold: f32,
    normalized: bool,
    max_pairs: Option<usize>,
) -> PyResult<Bound<'py, PyTuple>> {
    let dim = matrix.as_array().shape()[1];
    let matrix_slice = matrix.as_slice()?;
    let nd = py
        .detach(|| core_near_duplicates(matrix_slice, dim, threshold, normalized, max_pairs))
        .map_err(map_err)?;
    let m = nd.scores.len();
    let pairs = Array2::from_shape_vec((m, 2), nd.pairs)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    let scores = nd.scores.into_pyarray(py);
    let truncated = nd.truncated.into_pyobject(py)?.to_owned();
    PyTuple::new(
        py,
        [pairs.into_any(), scores.into_any(), truncated.into_any()],
    )
}

// ----------------------------------------------------------------------------
// Module registration
// ----------------------------------------------------------------------------

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "similarity")?;

    m.add_function(wrap_pyfunction!(cosine, &m)?)?;
    m.add_function(wrap_pyfunction!(cosine_normalized, &m)?)?;
    m.add_function(wrap_pyfunction!(cosine_one_to_many, &m)?)?;
    m.add_function(wrap_pyfunction!(cosine_one_to_many_normalized, &m)?)?;
    m.add_function(wrap_pyfunction!(cosine_adjacent, &m)?)?;
    m.add_function(wrap_pyfunction!(cosine_adjacent_normalized, &m)?)?;
    m.add_function(wrap_pyfunction!(top_k_cosine, &m)?)?;
    m.add_function(wrap_pyfunction!(mmr_select, &m)?)?;
    m.add_function(wrap_pyfunction!(l2_normalize_in_place, &m)?)?;
    m.add_function(wrap_pyfunction!(knn_graph, &m)?)?;
    m.add_function(wrap_pyfunction!(near_duplicates, &m)?)?;

    parent.add_submodule(&m)?;
    // Required so `from kaos_nlp_core._rust.similarity import ...` resolves.
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.similarity", &m)?;
    Ok(())
}
