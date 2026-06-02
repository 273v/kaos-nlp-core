//! PyO3 bindings for `crate::core::ctfidf` — class-based TF-IDF.
//!
//! ```text
//! kaos_nlp_core._rust.ctfidf
//!   └── class_tfidf(texts, class_ids, n_classes, ngram_min, ngram_max,
//!                   top_k, min_df, reduce_frequent_words, bm25_weighting,
//!                   lowercase, token_prefix, stopwords)
//!       -> list[list[tuple[str, float]]]
//! ```
//!
//! The heavy kernel (tokenise + count + score) runs with the GIL released
//! via `py.detach`, matching the repo convention.

use ahash::AHashSet;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::ctfidf::{class_tfidf as core_class_tfidf, CtfidfError};

fn map_err(e: CtfidfError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Class-based TF-IDF (c-TF-IDF): top terms per class.
///
/// Args:
///     texts: one string per item.
///     class_ids: ``uint32`` class index per item (``0..n_classes``),
///         parallel to ``texts``.
///     n_classes: number of classes (the output length).
///     ngram_min / ngram_max: inclusive n-gram range (``1 <= min <= max``).
///     top_k: terms kept per class.
///     min_df: drop terms whose global count is below this.
///     reduce_frequent_words: take ``sqrt`` of the L1-normalised TF.
///     bm25_weighting: use the smoothed BM25-style IDF.
///     lowercase: lowercase tokens before counting.
///     token_prefix: truncate tokens to this many chars before counting
///         (``0`` disables); conflates morphological/derivational variants.
///     stopwords: lowercase tokens to drop (``None`` = none).
///
/// Returns:
///     A list of length ``n_classes``; entry ``c`` is the ranked
///     ``(term, weight)`` list for class ``c`` (descending weight,
///     alphabetical tie-break).
///
/// Raises:
///     ValueError: length mismatch, a class id ``>= n_classes``, or an
///         invalid n-gram range.
#[pyfunction]
#[pyo3(signature = (
    texts, class_ids, n_classes, ngram_min, ngram_max, top_k,
    min_df=1, reduce_frequent_words=false, bm25_weighting=false,
    lowercase=true, token_prefix=0, stopwords=None,
))]
#[allow(clippy::too_many_arguments)]
fn class_tfidf(
    py: Python<'_>,
    texts: Vec<String>,
    class_ids: Vec<u32>,
    n_classes: usize,
    ngram_min: usize,
    ngram_max: usize,
    top_k: usize,
    min_df: u32,
    reduce_frequent_words: bool,
    bm25_weighting: bool,
    lowercase: bool,
    token_prefix: usize,
    stopwords: Option<Vec<String>>,
) -> PyResult<Vec<Vec<(String, f64)>>> {
    let stops: AHashSet<String> = stopwords.unwrap_or_default().into_iter().collect();
    py.detach(move || {
        core_class_tfidf(
            &texts,
            &class_ids,
            n_classes,
            (ngram_min, ngram_max),
            top_k,
            min_df,
            reduce_frequent_words,
            bm25_weighting,
            lowercase,
            token_prefix,
            &stops,
        )
    })
    .map_err(map_err)
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "ctfidf")?;
    m.add_function(wrap_pyfunction!(class_tfidf, &m)?)?;
    parent.add_submodule(&m)?;
    // Required so `from kaos_nlp_core._rust.ctfidf import ...` resolves.
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.ctfidf", &m)?;
    Ok(())
}
