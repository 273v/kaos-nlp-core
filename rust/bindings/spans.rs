//! Typed PyO3 pyclasses for the hot per-span / per-result FFI boundary.
//!
//! Audit perf finding #1 / P3 — every `tokenize`, `match`, and BM25 call
//! used to materialize a `PyDict` per result in Rust, then immediately
//! rebuild a Python dataclass wrapper on the Python side. That double
//! materialization measured ~11x slower than `tokenize_words` (which
//! returns `Vec<String>` directly) on a 10K-token text.
//!
//! These pyclasses replace the `PyDict` step. They are:
//!   - `frozen`: immutable after construction; cheaper attribute access
//!     and trivially `Send + Sync`.
//!   - `get_all`: every field is exposed read-only via `#[pyo3(get)]`
//!     property descriptors.
//!   - `eq`, `hash`: structural equality + hashing for use in sets/dicts
//!     (matches the prior `@dataclass(frozen=True, slots=True)` shape).
//!
//! Python `__init__.py` re-exports each one under the original
//! `kaos_nlp_core.types` name so callers see the same import path
//! (`from kaos_nlp_core.types import TokenSpan`) and the same attribute
//! shape — the change is invisible to existing code.

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// A token with its character offset span in the source text.
///
/// Replaces the prior `@dataclass(frozen=True, slots=True) TokenSpan` —
/// same field shape (`text`, `start`, `end`), now native PyO3 so the
/// tokenizer hot path skips a Rust→PyDict→Python-dataclass round trip.
#[pyclass(
    frozen,
    get_all,
    eq,
    hash,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.spans"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyTokenSpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[pymethods]
impl PyTokenSpan {
    #[new]
    fn new(text: String, start: usize, end: usize) -> Self {
        Self { text, start, end }
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenSpan(text={:?}, start={}, end={})",
            self.text, self.start, self.end
        )
    }

    /// Pickle support — `__getnewargs__` returns the constructor args
    /// so unpickling round-trips through `__new__`. `frozen` pyclasses
    /// can't carry `&mut self` methods, so `__setstate__` isn't an option.
    fn __getnewargs__(&self) -> (String, usize, usize) {
        (self.text.clone(), self.start, self.end)
    }
}

/// A substring or pattern match with character offsets.
#[pyclass(
    frozen,
    get_all,
    eq,
    hash,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.spans"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyMatchSpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[pymethods]
impl PyMatchSpan {
    #[new]
    fn new(text: String, start: usize, end: usize) -> Self {
        Self { text, start, end }
    }

    fn __repr__(&self) -> String {
        format!(
            "MatchSpan(text={:?}, start={}, end={})",
            self.text, self.start, self.end
        )
    }

    fn __getnewargs__(&self) -> (String, usize, usize) {
        (self.text.clone(), self.start, self.end)
    }
}

/// A multi-pattern (Aho-Corasick) match with pattern index.
#[pyclass(
    frozen,
    get_all,
    eq,
    hash,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.spans"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyPatternMatchSpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub pattern_index: usize,
}

#[pymethods]
impl PyPatternMatchSpan {
    #[new]
    fn new(text: String, start: usize, end: usize, pattern_index: usize) -> Self {
        Self {
            text,
            start,
            end,
            pattern_index,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PatternMatchSpan(text={:?}, start={}, end={}, pattern_index={})",
            self.text, self.start, self.end, self.pattern_index
        )
    }

    fn __getnewargs__(&self) -> (String, usize, usize, usize) {
        (self.text.clone(), self.start, self.end, self.pattern_index)
    }
}

/// A regex match with capture groups. Hash/eq excluded because
/// `Vec<Option<String>>` doesn't trivially hash; equality alone is enough
/// for tests.
#[pyclass(
    frozen,
    get_all,
    eq,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.spans"
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyRegexMatchSpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub groups: Vec<Option<String>>,
}

#[pymethods]
impl PyRegexMatchSpan {
    #[new]
    fn new(text: String, start: usize, end: usize, groups: Vec<Option<String>>) -> Self {
        Self {
            text,
            start,
            end,
            groups,
        }
    }

    fn __repr__(&self) -> String {
        // Render groups in Python style (`['x', None]`) rather than Rust
        // debug (`[Some("x"), None]`) so the repr matches what callers see
        // when they print the list directly.
        let groups: Vec<String> = self
            .groups
            .iter()
            .map(|g| match g {
                Some(s) => format!("{:?}", s),
                None => "None".to_string(),
            })
            .collect();
        format!(
            "RegexMatchSpan(text={:?}, start={}, end={}, groups=[{}])",
            self.text,
            self.start,
            self.end,
            groups.join(", ")
        )
    }

    fn __getnewargs__(&self) -> (String, usize, usize, Vec<Option<String>>) {
        (self.text.clone(), self.start, self.end, self.groups.clone())
    }
}

/// A fuzzy FST search result.
#[pyclass(
    frozen,
    get_all,
    eq,
    hash,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.spans"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyFstSearchResult {
    pub key: String,
    pub distance: u32,
}

#[pymethods]
impl PyFstSearchResult {
    #[new]
    fn new(key: String, distance: u32) -> Self {
        Self { key, distance }
    }

    fn __repr__(&self) -> String {
        format!(
            "FstSearchResult(key={:?}, distance={})",
            self.key, self.distance
        )
    }

    fn __getnewargs__(&self) -> (String, u32) {
        (self.key.clone(), self.distance)
    }
}

/// A scored document from `InvertedIndex` retrieval. Equality/hash use
/// only `doc_id` because `f64` doesn't `Hash`/`Eq`; downstream callers
/// that need score-aware equality should compare both fields explicitly.
#[pyclass(
    frozen,
    get_all,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.spans"
)]
#[derive(Clone, Debug)]
pub struct PyScoredDoc {
    pub doc_id: u32,
    pub score: f64,
}

#[pymethods]
impl PyScoredDoc {
    #[new]
    fn new(doc_id: u32, score: f64) -> Self {
        Self { doc_id, score }
    }

    fn __repr__(&self) -> String {
        format!("ScoredDoc(doc_id={}, score={})", self.doc_id, self.score)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id && self.score == other.score
    }

    fn __hash__(&self) -> u64 {
        // Match the prior dataclass(frozen) hash semantics — both fields
        // contribute. f64 hashed via its bit pattern.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.doc_id.hash(&mut h);
        self.score.to_bits().hash(&mut h);
        h.finish()
    }

    fn __getnewargs__(&self) -> (u32, f64) {
        (self.doc_id, self.score)
    }
}

/// A term posting in an `InvertedIndex`.
#[pyclass(
    frozen,
    get_all,
    eq,
    hash,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.spans"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyPostingEntry {
    pub doc_id: u32,
    pub term_freq: u32,
}

#[pymethods]
impl PyPostingEntry {
    #[new]
    fn new(doc_id: u32, term_freq: u32) -> Self {
        Self { doc_id, term_freq }
    }

    fn __repr__(&self) -> String {
        format!(
            "PostingEntry(doc_id={}, term_freq={})",
            self.doc_id, self.term_freq
        )
    }

    fn __getnewargs__(&self) -> (u32, u32) {
        (self.doc_id, self.term_freq)
    }
}

/// A sentence/paragraph/line search hit (used by `search.search_paragraphs`,
/// `search.search_sentences`, and the `Searcher.search()` corpus-level path).
/// `score` excluded from hash/eq for the same f64 reason as `PyScoredDoc`.
#[pyclass(
    frozen,
    get_all,
    skip_from_py_object,
    module = "kaos_nlp_core._rust.spans"
)]
#[derive(Clone, Debug)]
pub struct PySegmentHit {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub score: f64,
}

#[pymethods]
impl PySegmentHit {
    #[new]
    fn new(text: String, start: usize, end: usize, score: f64) -> Self {
        Self {
            text,
            start,
            end,
            score,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SegmentHit(text={:?}, start={}, end={}, score={})",
            self.text, self.start, self.end, self.score
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.text == other.text
            && self.start == other.start
            && self.end == other.end
            && self.score == other.score
    }

    fn __getnewargs__(&self) -> (String, usize, usize, f64) {
        (self.text.clone(), self.start, self.end, self.score)
    }
}

/// Register the `kaos_nlp_core._rust.spans` submodule with all
/// typed result pyclasses.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "spans")?;
    m.add_class::<PyTokenSpan>()?;
    m.add_class::<PyMatchSpan>()?;
    m.add_class::<PyPatternMatchSpan>()?;
    m.add_class::<PyRegexMatchSpan>()?;
    m.add_class::<PyFstSearchResult>()?;
    m.add_class::<PyScoredDoc>()?;
    m.add_class::<PyPostingEntry>()?;
    m.add_class::<PySegmentHit>()?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.spans", &m)?;
    Ok(())
}
