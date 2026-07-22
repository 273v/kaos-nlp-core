//! PyO3 bindings for readability text primitives.
//!
//! Exposes `analyze(text, ...)` returning a typed `PyTextCounts`
//! pyclass, plus `syllable_count(word)` and a `SyllableMap` pyclass
//! wrapping a word→syllable-count FST (CMUdict-derived). Formula
//! arithmetic and tunable constants live in the Python `readability`
//! module so they can change without rebuilding the wheel.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::bindings::lexicon::PyLexicon;
use crate::bindings::matching::PyFstSet;
use crate::core::matching::fst_match::FstMap;
use crate::core::readability::{
    count_text, count_text_no_lexicon, syllable::estimate_syllables, ReadabilityConfig, TextCounts,
};

// ── SyllableMap ────────────────────────────────────────────────────────────

/// Immutable word → syllable-count map backed by an FST.
///
/// Built from CMUdict-style data by `scripts/build_syllable_map.py`;
/// keys are lowercase words (apostrophes allowed).
#[pyclass(
    frozen,
    name = "SyllableMap",
    module = "kaos_nlp_core._rust.readability"
)]
pub(crate) struct PySyllableMap {
    pub(crate) inner: FstMap,
}

#[pymethods]
impl PySyllableMap {
    /// Build a map from (word, count) pairs.
    #[new]
    fn new(py: Python<'_>, entries: Vec<(String, u64)>) -> PyResult<Self> {
        let inner = py.detach(|| FstMap::build(entries.iter().map(|(k, v)| (k.as_str(), *v))));
        let inner = inner.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Load a map from disk (raw FST bytes written by ``save``).
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py.detach(|| FstMap::load_from_path(path));
        let inner = inner.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Write the raw FST bytes to disk.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.inner.save_to_path(path))
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Look up a word (callers should pass lowercase).
    fn get(&self, word: &str) -> Option<u64> {
        self.inner.get(word)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, word: &str) -> bool {
        self.inner.contains_key(word)
    }
}

// ── TextCounts pyclass ─────────────────────────────────────────────────────

/// Raw readability counts (see the Rust core docs for definitions).
#[pyclass(
    frozen,
    get_all,
    skip_from_py_object,
    name = "TextCounts",
    module = "kaos_nlp_core._rust.readability"
)]
#[derive(Clone)]
pub(crate) struct PyTextCounts {
    pub(crate) words: u64,
    pub(crate) letters: u64,
    pub(crate) letters_and_digits: u64,
    pub(crate) syllables: u64,
    pub(crate) polysyllable_words: u64,
    pub(crate) fog_complex_words: u64,
    pub(crate) long_words: u64,
    pub(crate) unfamiliar_words: Option<u64>,
}

fn counts_to_pyclass(c: &TextCounts) -> PyTextCounts {
    PyTextCounts {
        words: c.words,
        letters: c.letters,
        letters_and_digits: c.letters_and_digits,
        syllables: c.syllables,
        polysyllable_words: c.polysyllable_words,
        fog_complex_words: c.fog_complex_words,
        long_words: c.long_words,
        unfamiliar_words: c.unfamiliar_words,
    }
}

// ── Functions ──────────────────────────────────────────────────────────────

/// Count readability primitives for `text` in a single GIL-released pass.
///
/// `lexicon` may be `None`, a `kaos_nlp_core.matching.FstSet`, or a
/// `kaos_nlp_core.lexicon.Lexicon` (enables the Dale-Chall
/// unfamiliar-word count). `syllable_map` may be a `SyllableMap` for
/// exact syllable lookup with heuristic fallback. The three `fog_*`
/// flags control Gunning's complex-word exclusions (default: all on).
#[pyfunction]
#[pyo3(signature = (
    text,
    lexicon=None,
    syllable_map=None,
    fog_exclude_suffixes=true,
    fog_exclude_proper_nouns=true,
    fog_exclude_compounds=true,
))]
#[allow(clippy::too_many_arguments)]
fn analyze(
    py: Python<'_>,
    text: &str,
    lexicon: Option<&Bound<'_, PyAny>>,
    syllable_map: Option<&Bound<'_, PySyllableMap>>,
    fog_exclude_suffixes: bool,
    fog_exclude_proper_nouns: bool,
    fog_exclude_compounds: bool,
) -> PyResult<PyTextCounts> {
    let config = ReadabilityConfig {
        fog_exclude_suffixes,
        fog_exclude_proper_nouns,
        fog_exclude_compounds,
    };
    let map_ref = syllable_map.map(|m| &m.get().inner);

    // Inline downcasting keeps the PyRef borrow guards alive while the
    // GIL-released closure runs (same shape as quality::analyze_text).
    let result: TextCounts = match lexicon {
        None => py.detach(|| count_text_no_lexicon(text, map_ref, &config)),
        Some(obj) if obj.is_none() => py.detach(|| count_text_no_lexicon(text, map_ref, &config)),
        Some(obj) => {
            if let Ok(fst_bound) = obj.cast::<PyFstSet>() {
                let borrow = fst_bound.borrow();
                let inner = borrow.inner_ref();
                py.detach(|| count_text(text, Some(inner), map_ref, &config))
            } else if let Ok(lex_bound) = obj.cast::<PyLexicon>() {
                let borrow = lex_bound.borrow();
                let inner = borrow.inner_ref();
                py.detach(|| count_text(text, Some(inner), map_ref, &config))
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "lexicon must be None, a kaos_nlp_core.matching.FstSet, \
                     or a kaos_nlp_core.lexicon.Lexicon",
                ));
            }
        }
    };
    Ok(counts_to_pyclass(&result))
}

/// Estimate syllables for a single token.
///
/// Uses the exact map when supplied (lowercased lookup, heuristic
/// fallback). Deterministic; at least 1 for any non-empty token.
#[pyfunction]
#[pyo3(signature = (word, syllable_map=None))]
fn syllable_count(word: &str, syllable_map: Option<&Bound<'_, PySyllableMap>>) -> u32 {
    if word.is_empty() {
        return 0;
    }
    if let Some(m) = syllable_map {
        let key = word.to_lowercase();
        if let Some(v) = m.get().inner.get(&key) {
            return (v as u32).max(1);
        }
    }
    estimate_syllables(word)
}

// ── Module registration ────────────────────────────────────────────────────

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "readability")?;
    m.add_class::<PyTextCounts>()?;
    m.add_class::<PySyllableMap>()?;
    m.add_function(wrap_pyfunction!(analyze, &m)?)?;
    m.add_function(wrap_pyfunction!(syllable_count, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.readability", &m)?;
    Ok(())
}
