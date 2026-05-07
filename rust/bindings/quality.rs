//! PyO3 bindings for document quality metrics.
//!
//! Exposes `analyze_text(text, lexicon=None)` which returns a typed
//! `PyQualityRaw` pyclass containing nested `PyCharClassCounts` and
//! `PyWordStats`. Anomaly weighting and expected ranges live in the
//! Python `quality` module so they can be tuned without rebuilding the
//! wheel.
//!
//! Result shape: typed pyclasses replace the prior nested-dict FFI
//! boundary (audit perf finding #6 / P8). Field names match the prior
//! dict keys 1:1, so callers that previously did `raw["chars"]["total_chars"]`
//! now do `raw.chars.total_chars` — attribute access, no per-key dict
//! lookup.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::bindings::lexicon::PyLexicon;
use crate::bindings::matching::PyFstSet;
use crate::core::quality::{
    analyze, analyze_no_lexicon, count_char_classes, shannon_entropy_from_iter, word_stats,
    CharClassCounts, NoMembership, QualityRaw, WordStats,
};

// ── Typed result pyclasses ─────────────────────────────────────────────────

/// Single-pass character-class counts.
#[pyclass(
    frozen,
    get_all,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.quality"
)]
#[derive(Clone)]
pub(crate) struct PyCharClassCounts {
    pub(crate) total_chars: u64,
    pub(crate) whitespace: u64,
    pub(crate) alpha: u64,
    pub(crate) digit: u64,
    pub(crate) alphanumeric: u64,
    pub(crate) upper: u64,
    pub(crate) lower: u64,
    pub(crate) punct: u64,
    pub(crate) symbol: u64,
    pub(crate) non_ascii: u64,
    pub(crate) newline: u64,
    pub(crate) line_count: u64,
    pub(crate) paragraph_count: u64,
    pub(crate) char_entropy: f64,
}

/// Token-level statistics.
#[pyclass(
    frozen,
    get_all,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.quality"
)]
#[derive(Clone)]
pub(crate) struct PyWordStats {
    pub(crate) num_words: u64,
    pub(crate) unique_words: u64,
    pub(crate) total_word_chars: u64,
    pub(crate) max_freq: u64,
    pub(crate) token_entropy: f64,
    pub(crate) alphabetic_tokens: u64,
    pub(crate) in_lexicon: u64,
    pub(crate) format_tokens: u64,
}

/// Combined raw-metrics result returned by `analyze_text`.
#[pyclass(
    frozen,
    get_all,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.quality"
)]
#[derive(Clone)]
pub(crate) struct PyQualityRaw {
    pub(crate) chars: PyCharClassCounts,
    pub(crate) words: PyWordStats,
}

fn chars_to_pyclass(c: &CharClassCounts) -> PyCharClassCounts {
    PyCharClassCounts {
        total_chars: c.total_chars,
        whitespace: c.whitespace,
        alpha: c.alpha,
        digit: c.digit,
        alphanumeric: c.alphanumeric,
        upper: c.upper,
        lower: c.lower,
        punct: c.punct,
        symbol: c.symbol,
        non_ascii: c.non_ascii,
        newline: c.newline,
        line_count: c.line_count,
        paragraph_count: c.paragraph_count,
        char_entropy: c.char_entropy,
    }
}

fn words_to_pyclass(w: &WordStats) -> PyWordStats {
    PyWordStats {
        num_words: w.num_words,
        unique_words: w.unique_words,
        total_word_chars: w.total_word_chars,
        max_freq: w.max_freq,
        token_entropy: w.token_entropy,
        alphabetic_tokens: w.alphabetic_tokens,
        in_lexicon: w.in_lexicon,
        format_tokens: w.format_tokens,
    }
}

fn raw_to_pyclass(r: &QualityRaw) -> PyQualityRaw {
    PyQualityRaw {
        chars: chars_to_pyclass(&r.chars),
        words: words_to_pyclass(&r.words),
    }
}

// ── Public bindings ────────────────────────────────────────────────────────

/// Compute single-pass character-class counts.
///
/// Returns a `CharClassCounts` pyclass with fields: total_chars,
/// whitespace, alpha, digit, alphanumeric, upper, lower, punct, symbol,
/// non_ascii, newline, line_count, paragraph_count, char_entropy.
#[pyfunction]
fn count_chars(py: Python<'_>, text: &str) -> PyCharClassCounts {
    let counts = py.detach(|| count_char_classes(text));
    chars_to_pyclass(&counts)
}

/// Compute token-level statistics from a list of tokens (no lexicon).
#[pyfunction]
fn analyze_words(py: Python<'_>, words: Vec<String>) -> PyWordStats {
    let stats = py.detach(|| word_stats::<NoMembership>(&words, None));
    words_to_pyclass(&stats)
}

/// Run the full raw-metrics pipeline on `text`.
///
/// `lexicon` may be `None`, a `kaos_nlp_core.matching.FstSet`, or a
/// `kaos_nlp_core.lexicon.Lexicon`. Lexicon membership is checked
/// case-insensitively (the analyzer also tries the lower-cased form).
///
/// Returns a `QualityRaw` pyclass with two fields:
///   `chars` → `CharClassCounts` pyclass
///   `words` → `WordStats` pyclass
#[pyfunction]
#[pyo3(signature = (text, lexicon=None))]
fn analyze_text(
    py: Python<'_>,
    text: &str,
    lexicon: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyQualityRaw> {
    // Inline downcasting keeps the PyRef borrow guard alive for the duration
    // of the analysis call — `inner_ref` returns a reference whose lifetime
    // is tied to the PyRef, so the GIL-released analyze closure is safe.
    let result: QualityRaw = match lexicon {
        None => py.detach(|| analyze_no_lexicon(text)),
        Some(obj) if obj.is_none() => py.detach(|| analyze_no_lexicon(text)),
        Some(obj) => {
            if let Ok(fst_bound) = obj.cast::<PyFstSet>() {
                let borrow = fst_bound.borrow();
                let inner = borrow.inner_ref();
                py.detach(|| analyze(text, Some(inner)))
            } else if let Ok(lex_bound) = obj.cast::<PyLexicon>() {
                let borrow = lex_bound.borrow();
                let inner = borrow.inner_ref();
                py.detach(|| analyze(text, Some(inner)))
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "lexicon must be None, a kaos_nlp_core.matching.FstSet, \
                     or a kaos_nlp_core.lexicon.Lexicon",
                ));
            }
        }
    };
    Ok(raw_to_pyclass(&result))
}

/// Compute Shannon entropy (log base 2) over a list of frequency counts.
#[pyfunction]
fn entropy(counts: Vec<u64>) -> f64 {
    let total: u64 = counts.iter().sum();
    shannon_entropy_from_iter(counts.into_iter(), total)
}

// ── Module registration ────────────────────────────────────────────────────

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "quality")?;
    m.add_class::<PyCharClassCounts>()?;
    m.add_class::<PyWordStats>()?;
    m.add_class::<PyQualityRaw>()?;
    m.add_function(wrap_pyfunction!(count_chars, &m)?)?;
    m.add_function(wrap_pyfunction!(analyze_words, &m)?)?;
    m.add_function(wrap_pyfunction!(analyze_text, &m)?)?;
    m.add_function(wrap_pyfunction!(entropy, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.quality", &m)?;
    Ok(())
}
