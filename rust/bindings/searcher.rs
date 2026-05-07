//! PyO3 bindings for sentence and paragraph search.
//!
//! Exposes `search_sentences` and `search_paragraphs` from the Rust core,
//! converting byte offsets to character offsets before returning to Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use crate::core::searcher::{search_paragraphs, search_sentences, ScoredSegment};
use crate::core::segmentation::PunktSentenceTokenizer;
use crate::core::tokenizer::TokenizerConfig;

use super::segmentation::PyPunktTokenizer;
use super::util::build_byte_to_char_table_always;

// ─── Helper: convert ScoredSegments to Python list of dicts ─────────────────

fn scored_segments_to_pylist(
    py: Python<'_>,
    text: &str,
    segments: &[ScoredSegment],
) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);

    if text.is_ascii() {
        // ASCII fast path: byte offsets == char offsets
        for seg in segments {
            let d = PyDict::new(py);
            d.set_item("text", &seg.text)?;
            d.set_item("start", seg.start)?;
            d.set_item("end", seg.end)?;
            d.set_item("score", seg.score)?;
            list.append(d)?;
        }
    } else {
        // Unicode: single-pass offset table
        let char_offsets = build_byte_to_char_table_always(text);
        for seg in segments {
            let d = PyDict::new(py);
            d.set_item("text", &seg.text)?;
            let cs = if seg.start < char_offsets.len() {
                char_offsets[seg.start]
            } else {
                char_offsets.last().copied().unwrap_or(0)
            };
            let ce = if seg.end < char_offsets.len() {
                char_offsets[seg.end]
            } else {
                char_offsets.last().copied().unwrap_or(0)
            };
            d.set_item("start", cs)?;
            d.set_item("end", ce)?;
            d.set_item("score", seg.score)?;
            list.append(d)?;
        }
    }

    Ok(list.into())
}

// ─── Search functions ───────────────────────────────────────────────────────

/// Search within a document's sentences using BM25 scoring.
///
/// Segments the document into sentences, builds a temporary inverted index,
/// queries it, and returns matching sentences with character offsets and scores.
///
/// Args:
///     text: Document text to search within.
///     query: Search query (one or more terms).
///     tokenizer: Optional PunktTokenizer for sentence segmentation.
///         If None, uses an empty model.
///     top_k: Maximum number of results to return (default 10).
///     lowercase: Whether to lowercase terms for matching (default true).
///
/// Returns:
///     List of dicts with keys: text, start, end, score.
///     Offsets are character-based (safe for Python string indexing).
#[pyfunction]
#[pyo3(signature = (text, query, tokenizer=None, top_k=10, lowercase=true))]
fn py_search_sentences(
    py: Python<'_>,
    text: &str,
    query: &str,
    tokenizer: Option<&PyPunktTokenizer>,
    top_k: usize,
    lowercase: bool,
) -> PyResult<Py<PyList>> {
    let default_tok;
    let tok = match tokenizer {
        Some(t) => &t.inner,
        None => {
            default_tok = PunktSentenceTokenizer::new();
            &default_tok
        }
    };

    let mut config = TokenizerConfig::new();
    if lowercase {
        config = config.lowercase();
    }

    let results = py.detach(|| search_sentences(text, query, tok, &config, top_k));
    scored_segments_to_pylist(py, text, &results)
}

/// Search within a document's paragraphs using BM25 scoring.
///
/// Segments the document into paragraphs (sentence-aware boundaries),
/// builds a temporary inverted index, queries it, and returns matching
/// paragraphs with character offsets and scores.
///
/// Args:
///     text: Document text to search within.
///     query: Search query (one or more terms).
///     tokenizer: Optional PunktTokenizer for sentence/paragraph segmentation.
///         If None, uses an empty model.
///     top_k: Maximum number of results to return (default 10).
///     lowercase: Whether to lowercase terms for matching (default true).
///
/// Returns:
///     List of dicts with keys: text, start, end, score.
///     Offsets are character-based (safe for Python string indexing).
#[pyfunction]
#[pyo3(signature = (text, query, tokenizer=None, top_k=10, lowercase=true))]
fn py_search_paragraphs(
    py: Python<'_>,
    text: &str,
    query: &str,
    tokenizer: Option<&PyPunktTokenizer>,
    top_k: usize,
    lowercase: bool,
) -> PyResult<Py<PyList>> {
    let default_tok;
    let tok = match tokenizer {
        Some(t) => &t.inner,
        None => {
            default_tok = PunktSentenceTokenizer::new();
            &default_tok
        }
    };

    let mut config = TokenizerConfig::new();
    if lowercase {
        config = config.lowercase();
    }

    let results = py.detach(|| search_paragraphs(text, query, tok, &config, top_k));
    scored_segments_to_pylist(py, text, &results)
}

// ─── Module registration ────────────────────────────────────────────────────

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "searcher")?;

    m.add_function(wrap_pyfunction!(py_search_sentences, &m)?)?;
    m.add_function(wrap_pyfunction!(py_search_paragraphs, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.searcher", &m)?;

    Ok(())
}
