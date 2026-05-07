//! PyO3 bindings for sentence/paragraph-level document diffing.
//!
//! All offsets surfaced to Python are **character offsets** (per the
//! crate-wide rule); the Rust core retains byte offsets internally and
//! we translate at the FFI boundary using the same byte-to-char table
//! pattern as `bindings::segmentation`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::bindings::segmentation::PyPunktTokenizer;
use crate::core::algorithms::dispatch::MetricConfig;
use crate::core::diff::{diff_documents, DiffConfig, Granularity, SegmentChange};

fn parse_granularity(s: &str) -> PyResult<Granularity> {
    match s {
        "sentence" => Ok(Granularity::Sentence),
        "paragraph" => Ok(Granularity::Paragraph),
        "line" => Ok(Granularity::Line),
        "paragraph_simple" => Ok(Granularity::ParagraphSimple),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown granularity {other:?}; valid options: sentence, paragraph, line, paragraph_simple"
        ))),
    }
}

/// Build a byte-index → char-index lookup table for non-ASCII text.
/// (Local copy; the `bindings::segmentation` version is private.)
fn build_byte_to_char_table(text: &str) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(text.len() + 1);
    let mut char_count = 0usize;
    for (byte_idx, _) in text.char_indices() {
        while offsets.len() <= byte_idx {
            offsets.push(char_count);
        }
        char_count += 1;
    }
    while offsets.len() <= text.len() {
        offsets.push(char_count);
    }
    offsets
}

fn segref_to_dict(
    py: Python<'_>,
    text: &str,
    seg: Option<&crate::core::diff::SegmentRef>,
    table: Option<&Vec<usize>>,
) -> PyResult<Option<Py<PyDict>>> {
    let Some(s) = seg else {
        return Ok(None);
    };
    let (start_char, end_char) = match table {
        None => (s.start as usize, s.end as usize),
        Some(t) => {
            let i = (s.start as usize).min(t.len() - 1);
            let j = (s.end as usize).min(t.len() - 1);
            (t[i], t[j])
        }
    };
    let d = PyDict::new(py);
    d.set_item("index", s.index)?;
    d.set_item("start", start_char)?;
    d.set_item("end", end_char)?;
    // Slice the source text by *byte* offsets — those are the natural,
    // safe slice keys on a `&str` in Rust. The char offsets we expose
    // are derived from the same span, so the substring matches.
    d.set_item("text", &text[s.start as usize..s.end as usize])?;
    Ok(Some(d.into()))
}

fn changes_to_pylist(
    py: Python<'_>,
    a: &str,
    b: &str,
    changes: &[SegmentChange],
) -> PyResult<Py<PyList>> {
    let table_a = if a.is_ascii() {
        None
    } else {
        Some(build_byte_to_char_table(a))
    };
    let table_b = if b.is_ascii() {
        None
    } else {
        Some(build_byte_to_char_table(b))
    };

    let list = PyList::empty(py);
    for c in changes {
        let d = PyDict::new(py);
        d.set_item("kind", c.kind.as_str())?;
        d.set_item("score", c.score)?;
        match segref_to_dict(py, a, c.left.as_ref(), table_a.as_ref())? {
            Some(left) => d.set_item("left", left)?,
            None => d.set_item("left", py.None())?,
        }
        match segref_to_dict(py, b, c.right.as_ref(), table_b.as_ref())? {
            Some(right) => d.set_item("right", right)?,
            None => d.set_item("right", py.None())?,
        }
        list.append(d)?;
    }
    Ok(list.into())
}

/// Compute a segment-level diff between two documents.
///
/// Args:
///     a, b: Source and target text.
///     granularity: One of `"sentence"` (default), `"paragraph"`, `"line"`,
///         `"paragraph_simple"`. `"sentence"` and `"paragraph"` use Punkt;
///         the other two are model-free.
///     algorithm: Similarity metric name (default `"token-jaccard"`). Same
///         keys as `algorithms.compare_batch`.
///     n: N-gram size for n-gram metrics.
///     lowercase: Lowercase tokens for token-level metrics (default True
///         here — diffing is usually case-insensitive).
///     prefix_weight: Jaro-Winkler prefix factor.
///     match_threshold: Score at or above which a pair is `unchanged`
///         (or `moved` when `detect_moves=True`).
///     modify_threshold: Score floor for considering a pair a match at all.
///         Pairs in `[modify_threshold, match_threshold)` are `modified`.
///     detect_moves: When True, post-classify `unchanged` pairs whose
///         normalized index distance exceeds `move_distance_ratio` as
///         `moved`.
///     move_distance_ratio: Position-shift ratio threshold for `moved`.
///     tokenizer: Optional `PunktTokenizer`. When None, the binding uses
///         an empty model — pass `get_default_punkt_tokenizer()` from the
///         Python wrapper for the bundled legal model.
///
/// Returns a list of dicts. Each dict has:
///     - `"kind"`: one of `"unchanged"`, `"modified"`, `"moved"`,
///       `"added"`, `"removed"`.
///     - `"score"`: similarity in `[0, 1]`. `0.0` for `added` / `removed`.
///     - `"left"`: `None` for `added`, otherwise a dict with
///       `index`, `start`, `end` (char offsets) and `text`.
///     - `"right"`: `None` for `removed`, otherwise the same shape.
#[pyfunction]
#[pyo3(signature = (
    a, b,
    *,
    granularity = "sentence",
    algorithm = "token-jaccard",
    n = 2,
    lowercase = true,
    prefix_weight = 0.1,
    match_threshold = 0.85,
    modify_threshold = 0.4,
    detect_moves = false,
    move_distance_ratio = 0.1,
    tokenizer = None,
))]
#[allow(clippy::too_many_arguments)]
fn py_diff_documents(
    py: Python<'_>,
    a: &str,
    b: &str,
    granularity: &str,
    algorithm: &str,
    n: usize,
    lowercase: bool,
    prefix_weight: f64,
    match_threshold: f32,
    modify_threshold: f32,
    detect_moves: bool,
    move_distance_ratio: f32,
    tokenizer: Option<&PyPunktTokenizer>,
) -> PyResult<Py<PyList>> {
    let g = parse_granularity(granularity)?;
    let cfg = DiffConfig {
        granularity: g,
        algorithm: algorithm.to_string(),
        metric: MetricConfig {
            n,
            lowercase,
            prefix_weight,
        },
        match_threshold,
        modify_threshold,
        detect_moves,
        move_distance_ratio,
    };
    let a_owned = a.to_string();
    let b_owned = b.to_string();
    let tokenizer_clone = tokenizer.map(|t| t.inner.clone());

    let changes = py
        .detach(|| {
            let tok_ref = tokenizer_clone.as_ref();
            diff_documents(&a_owned, &b_owned, &cfg, tok_ref)
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    changes_to_pylist(py, &a_owned, &b_owned, &changes)
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "diff")?;
    m.add_function(wrap_pyfunction!(py_diff_documents, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.diff", &m)?;
    Ok(())
}
