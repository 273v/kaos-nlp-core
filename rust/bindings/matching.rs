//! PyO3 bindings for pattern matching: substring, multi-pattern, regex, FST.
//!
//! Result shape: every match-returning entrypoint emits typed pyclasses
//! from `super::spans` (`PyMatchSpan`, `PyPatternMatchSpan`,
//! `PyRegexMatchSpan`, `PyFstSearchResult`) instead of `PyDict`. Audit
//! perf finding #1 / P3 — the prior dict→Python-dataclass round trip was
//! ~11× slower than emitting the pyclass directly. Field shapes match
//! the prior `@dataclass` definitions, so callers keep working unchanged.

use pyo3::prelude::*;

use crate::core::matching::{
    fst_match::{FstMap, FstMatch, FstSet},
    multi_pattern::{MultiPatternMatchKind, MultiPatternMatcher, PatternMatch},
    regex_match::{RegexMatcher, RegexSetMatcher},
    substring,
};

use super::spans::{PyFstSearchResult, PyMatchSpan, PyPatternMatchSpan, PyRegexMatchSpan};
use super::util::{build_byte_to_char_table, byte_to_char};

// =============================================================================
// Substring search
// =============================================================================

fn submatch_to_pyclass(
    m: &substring::SubstringMatch,
    char_table: &Option<Vec<usize>>,
) -> PyMatchSpan {
    PyMatchSpan {
        text: m.text.clone(),
        start: byte_to_char(char_table, m.start),
        end: byte_to_char(char_table, m.end),
    }
}

/// Find all occurrences of needle in haystack (SIMD-accelerated, overlapping).
///
/// Returns a list of `MatchSpan` pyclasses (`text`, `start`, `end`).
/// Offsets are character positions suitable for Python string slicing.
#[pyfunction]
fn substring_find_all(py: Python<'_>, haystack: &str, needle: &str) -> Vec<PyMatchSpan> {
    py.detach(|| {
        let matches = substring::find_all(haystack, needle);
        let char_table = build_byte_to_char_table(haystack);
        matches
            .iter()
            .map(|m| submatch_to_pyclass(m, &char_table))
            .collect()
    })
}

/// Find all occurrences of needle across many haystacks.
#[pyfunction]
fn substring_find_all_batch(
    py: Python<'_>,
    haystacks: Vec<String>,
    needle: &str,
) -> Vec<Vec<PyMatchSpan>> {
    let needle_owned = needle.to_string();
    py.detach(|| {
        haystacks
            .iter()
            .map(|haystack| {
                let matches = substring::find_all(haystack, &needle_owned);
                let char_table = build_byte_to_char_table(haystack);
                matches
                    .iter()
                    .map(|m| submatch_to_pyclass(m, &char_table))
                    .collect()
            })
            .collect()
    })
}

/// Find the first occurrence of needle in haystack (SIMD-accelerated).
///
/// Returns a `MatchSpan` pyclass — or None if not found.
#[pyfunction]
fn substring_find_first(haystack: &str, needle: &str) -> Option<PyMatchSpan> {
    let char_table = build_byte_to_char_table(haystack);
    substring::find_first(haystack, needle).map(|m| submatch_to_pyclass(&m, &char_table))
}

/// Count non-overlapping occurrences of needle in haystack.
#[pyfunction]
fn substring_count(haystack: &str, needle: &str) -> usize {
    substring::count(haystack, needle)
}

/// Count non-overlapping occurrences across many haystacks.
#[pyfunction]
fn substring_count_batch(py: Python<'_>, haystacks: Vec<String>, needle: &str) -> Vec<usize> {
    let needle_owned = needle.to_string();
    py.detach(|| {
        haystacks
            .iter()
            .map(|haystack| substring::count(haystack, &needle_owned))
            .collect()
    })
}

/// Find all case-insensitive occurrences of needle in haystack.
///
/// Returns a list of `MatchSpan` pyclasses (`text`, `start`, `end`).
/// Offsets are character positions suitable for Python string slicing.
#[pyfunction]
fn substring_find_all_case_insensitive(
    py: Python<'_>,
    haystack: &str,
    needle: &str,
) -> Vec<PyMatchSpan> {
    py.detach(|| {
        let matches = substring::find_all_case_insensitive(haystack, needle);
        let char_table = build_byte_to_char_table(haystack);
        matches
            .iter()
            .map(|m| submatch_to_pyclass(m, &char_table))
            .collect()
    })
}

// =============================================================================
// Multi-pattern (Aho-Corasick)
// =============================================================================

fn pmatch_to_pyclass(m: &PatternMatch, char_table: &Option<Vec<usize>>) -> PyPatternMatchSpan {
    PyPatternMatchSpan {
        text: m.text.clone(),
        start: byte_to_char(char_table, m.start),
        end: byte_to_char(char_table, m.end),
        pattern_index: m.pattern_index,
    }
}

/// Multi-pattern matcher using Aho-Corasick automaton.
///
/// Searches for all patterns simultaneously in a single linear-time pass.
/// Supports pickle for multiprocessing.
///
/// Args:
///     patterns: List of pattern strings to search for.
///     case_insensitive: If True, match patterns case-insensitively.
///     longest_match: If True, prefer longest match at each position.
#[pyclass(name = "MultiPatternMatcher", module = "kaos_nlp_core._rust.matching")]
struct PyMultiPatternMatcher {
    inner: MultiPatternMatcher,
    // Keep construction params for pickle
    patterns: Vec<String>,
    case_insensitive: bool,
    longest_match: bool,
}

#[pymethods]
impl PyMultiPatternMatcher {
    #[new]
    #[pyo3(signature = (patterns, case_insensitive=false, longest_match=false))]
    fn new(
        py: Python<'_>,
        patterns: Vec<String>,
        case_insensitive: bool,
        longest_match: bool,
    ) -> PyResult<Self> {
        let kind = if longest_match {
            MultiPatternMatchKind::LeftmostLongest
        } else {
            MultiPatternMatchKind::LeftmostFirst
        };
        let inner = py.detach(|| {
            let refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
            if case_insensitive {
                MultiPatternMatcher::new_case_insensitive(&refs, kind)
            } else {
                MultiPatternMatcher::new(&refs, kind)
            }
        });
        let inner = inner.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner,
            patterns,
            case_insensitive,
            longest_match,
        })
    }

    fn __getnewargs__(&self) -> (Vec<String>, bool, bool) {
        (
            self.patterns.clone(),
            self.case_insensitive,
            self.longest_match,
        )
    }

    /// Find all non-overlapping matches. Returns list of `PatternMatchSpan`
    /// pyclasses (`text`, `start`, `end`, `pattern_index`).
    fn find_all(&self, py: Python<'_>, haystack: &str) -> Vec<PyPatternMatchSpan> {
        let inner = &self.inner;
        py.detach(|| {
            let matches = inner.find_all(haystack);
            let char_table = build_byte_to_char_table(haystack);
            matches
                .iter()
                .map(|m| pmatch_to_pyclass(m, &char_table))
                .collect()
        })
    }

    /// Find matches across many haystacks in one call.
    fn find_all_batch(
        &self,
        py: Python<'_>,
        haystacks: Vec<String>,
    ) -> Vec<Vec<PyPatternMatchSpan>> {
        let inner = &self.inner;
        py.detach(|| {
            haystacks
                .iter()
                .map(|haystack| {
                    let matches = inner.find_all(haystack);
                    let char_table = build_byte_to_char_table(haystack);
                    matches
                        .iter()
                        .map(|m| pmatch_to_pyclass(m, &char_table))
                        .collect()
                })
                .collect()
        })
    }

    /// Check if any pattern matches in the haystack.
    fn is_match(&self, haystack: &str) -> bool {
        self.inner.is_match(haystack)
    }

    /// Count total non-overlapping matches across all patterns.
    fn count(&self, haystack: &str) -> usize {
        self.inner.count(haystack)
    }

    /// Replace all matches with corresponding replacement strings.
    fn replace_all(
        &self,
        py: Python<'_>,
        haystack: &str,
        replacements: Vec<String>,
    ) -> PyResult<String> {
        let inner = &self.inner;
        let result = py.detach(|| {
            let refs: Vec<&str> = replacements.iter().map(|s| s.as_str()).collect();
            inner.replace_all(haystack, &refs)
        });
        result.map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Return the number of patterns in this matcher.
    fn pattern_count(&self) -> usize {
        self.inner.pattern_count()
    }
}

// =============================================================================
// Regex
// =============================================================================

fn rmatch_to_pyclass(
    m: &crate::core::matching::regex_match::RegexMatch,
    char_table: &Option<Vec<usize>>,
) -> PyRegexMatchSpan {
    PyRegexMatchSpan {
        text: m.text.clone(),
        start: byte_to_char(char_table, m.start),
        end: byte_to_char(char_table, m.end),
        groups: m.groups.clone(),
    }
}

/// Compiled regular expression matcher. Supports pickle for multiprocessing.
///
/// Args:
///     pattern: A regular expression pattern string.
///
/// Raises:
///     ValueError: If the pattern is invalid.
#[pyclass(name = "RegexMatcher", module = "kaos_nlp_core._rust.matching")]
struct PyRegexMatcher {
    inner: RegexMatcher,
    pattern_str: String,
}

#[pymethods]
impl PyRegexMatcher {
    #[new]
    fn new(py: Python<'_>, pattern: &str) -> PyResult<Self> {
        let pattern_owned = pattern.to_string();
        let inner = py.detach(|| RegexMatcher::new(&pattern_owned));
        let inner = inner.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner,
            pattern_str: pattern_owned,
        })
    }

    fn __getnewargs__(&self) -> (String,) {
        (self.pattern_str.clone(),)
    }

    /// Find all non-overlapping matches. Returns list of `RegexMatchSpan`
    /// pyclasses (`text`, `start`, `end`, `groups`).
    fn find_all(&self, py: Python<'_>, haystack: &str) -> Vec<PyRegexMatchSpan> {
        let inner = &self.inner;
        py.detach(|| {
            let matches = inner.find_all(haystack);
            let char_table = build_byte_to_char_table(haystack);
            matches
                .iter()
                .map(|m| rmatch_to_pyclass(m, &char_table))
                .collect()
        })
    }

    /// Find all non-overlapping matches across many haystacks.
    fn find_all_batch(&self, py: Python<'_>, haystacks: Vec<String>) -> Vec<Vec<PyRegexMatchSpan>> {
        let inner = &self.inner;
        py.detach(|| {
            haystacks
                .iter()
                .map(|haystack| {
                    let matches = inner.find_all(haystack);
                    let char_table = build_byte_to_char_table(haystack);
                    matches
                        .iter()
                        .map(|m| rmatch_to_pyclass(m, &char_table))
                        .collect()
                })
                .collect()
        })
    }

    /// Find the first match. Returns a `RegexMatchSpan` pyclass — or None.
    fn find_first(&self, py: Python<'_>, haystack: &str) -> Option<PyRegexMatchSpan> {
        let inner = &self.inner;
        py.detach(|| {
            let first = inner.find_first(haystack);
            let char_table = build_byte_to_char_table(haystack);
            first.map(|m| rmatch_to_pyclass(&m, &char_table))
        })
    }

    /// Check if the pattern matches anywhere in the haystack.
    fn is_match(&self, haystack: &str) -> bool {
        self.inner.is_match(haystack)
    }

    /// Count non-overlapping matches.
    fn count(&self, py: Python<'_>, haystack: &str) -> usize {
        let inner = &self.inner;
        py.detach(|| inner.count(haystack))
    }

    /// Replace all matches with the replacement string.
    fn replace_all(&self, py: Python<'_>, haystack: &str, replacement: &str) -> String {
        let inner = &self.inner;
        py.detach(|| inner.replace_all(haystack, replacement))
    }

    /// Split the haystack by the pattern.
    fn split(&self, py: Python<'_>, haystack: &str) -> Vec<String> {
        let inner = &self.inner;
        py.detach(|| {
            inner
                .split(haystack)
                .into_iter()
                .map(|s| s.to_string())
                .collect()
        })
    }

    /// Return the pattern string.
    fn pattern(&self) -> &str {
        &self.pattern_str
    }
}

/// Match multiple regex patterns in a single pass. Supports pickle.
///
/// Does not return match positions — only which pattern indices matched.
///
/// Args:
///     patterns: List of regex pattern strings.
#[pyclass(name = "RegexSetMatcher", module = "kaos_nlp_core._rust.matching")]
struct PyRegexSetMatcher {
    inner: RegexSetMatcher,
    patterns: Vec<String>,
}

#[pymethods]
impl PyRegexSetMatcher {
    #[new]
    fn new(py: Python<'_>, patterns: Vec<String>) -> PyResult<Self> {
        let inner = py.detach(|| {
            let refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
            RegexSetMatcher::new(&refs)
        });
        let inner = inner.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner, patterns })
    }

    fn __getnewargs__(&self) -> (Vec<String>,) {
        (self.patterns.clone(),)
    }

    /// Return indices of all patterns that match the haystack.
    fn matching_patterns(&self, py: Python<'_>, haystack: &str) -> Vec<usize> {
        let inner = &self.inner;
        py.detach(|| inner.matching_patterns(haystack))
    }

    /// Check if any pattern matches.
    fn is_match(&self, haystack: &str) -> bool {
        self.inner.is_match(haystack)
    }

    /// Return the number of patterns.
    fn pattern_count(&self) -> usize {
        self.inner.pattern_count()
    }
}

// =============================================================================
// FST
// =============================================================================

fn fst_match_to_pyclass(m: &FstMatch) -> PyFstSearchResult {
    PyFstSearchResult {
        key: m.key.clone(),
        distance: m.distance,
    }
}

/// Compact, immutable string set backed by a Finite State Transducer.
///
/// Supports exact lookup, prefix search, and Levenshtein fuzzy search.
/// Supports pickle for multiprocessing.
///
/// Pickle uses the raw FST byte buffer (not a key list) — audit perf
/// finding #5 / P7 — so the in-memory footprint is just the FST itself,
/// not a duplicated `Vec<String>`. Pickles produced by 0.1.0a1+ won't
/// be readable by older versions and vice versa; the
/// `kaos_nlp_core._rust.matching.FstSet` module path is stable so the
/// import succeeds, but unpickling an old (key-list) state under the
/// new `__setstate__` (bytes-state) raises immediately.
///
/// Args:
///     keys: List of strings to include in the set.
#[pyclass(name = "FstSet", module = "kaos_nlp_core._rust.matching")]
pub struct PyFstSet {
    pub(crate) inner: FstSet,
}

impl PyFstSet {
    /// Borrow the underlying FST set. Used by sibling binding modules.
    pub fn inner_ref(&self) -> &FstSet {
        &self.inner
    }
}

#[pymethods]
impl PyFstSet {
    #[new]
    fn new(py: Python<'_>, keys: Vec<String>) -> PyResult<Self> {
        let inner = py.detach(|| FstSet::build(keys.iter().map(|s| s.as_str())));
        let inner = inner.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Pickle: emit the raw FST byte buffer. `__setstate__` (below)
    /// reconstructs the set in O(1) from these bytes — no key streaming
    /// or rebuild required.
    ///
    /// Using `Vec<u8>` rather than a borrowed `Bound<PyBytes>` so that
    /// PyO3 0.28 routes the value through the standard pickle bytes
    /// protocol (a borrowed `PyBytes` here was returned via the
    /// auto-derived `__getstate__` slot as a dict by mistake).
    fn __getstate__(&self) -> Vec<u8> {
        self.inner.as_bytes().to_vec()
    }

    /// Pickle: restore the set from a raw FST byte buffer produced by
    /// `__getstate__`. The bytes must be a valid FST set (the format
    /// embeds its own header/length, so corrupt bytes fail loudly).
    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        let inner = FstSet::from_bytes(state).map_err(pyo3::exceptions::PyValueError::new_err)?;
        self.inner = inner;
        Ok(())
    }

    /// Pickle: hand `__new__` an empty key list. The real state is
    /// restored by `__setstate__` immediately afterwards, so the
    /// placeholder set is never observed by callers.
    fn __getnewargs__(&self) -> (Vec<String>,) {
        (Vec::new(),)
    }

    /// Check if a key exists in the set (exact match).
    fn contains(&self, key: &str) -> bool {
        self.inner.contains(key)
    }

    /// Find all keys within max_distance edits of query.
    ///
    /// Returns a list of `FstSearchResult` pyclasses (`key`, `distance`),
    /// sorted by distance.
    fn fuzzy_search(
        &self,
        py: Python<'_>,
        query: &str,
        max_distance: u32,
    ) -> PyResult<Vec<PyFstSearchResult>> {
        let inner = &self.inner;
        let results = py.detach(|| inner.fuzzy_search(query, max_distance));
        let results = results.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(results.iter().map(fst_match_to_pyclass).collect())
    }

    /// Find all keys starting with the given prefix.
    fn prefix_search(&self, prefix: &str) -> Vec<String> {
        self.inner.prefix_search(prefix)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.inner.contains(key)
    }

    /// Write the raw FST bytes to disk.
    ///
    /// The file contains only the FST byte buffer (no KNC framing); the
    /// FST format embeds its own header and length, so it can also be
    /// loaded by external `fst` crate consumers.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let inner = &self.inner;
        let path_owned = path.to_string();
        py.detach(|| inner.save_to_path(&path_owned))
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Load an FstSet from a file written by `save`.
    ///
    /// O(1) materialization — pickle/load both use the raw FST bytes,
    /// so we no longer stream every key off the FST to populate a side
    /// `keys` field (audit perf finding #5 / P7).
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_owned = path.to_string();
        let inner = py
            .detach(|| FstSet::load_from_path(&path_owned))
            .map_err(pyo3::exceptions::PyIOError::new_err)?;
        Ok(Self { inner })
    }
}

/// Compact, immutable string-to-integer map backed by a Finite State Transducer.
///
/// Supports pickle for multiprocessing.
///
/// Args:
///     entries: List of (key, value) tuples where value is a non-negative integer.
#[pyclass(name = "FstMap", module = "kaos_nlp_core._rust.matching")]
struct PyFstMap {
    inner: FstMap,
    entries: Vec<(String, u64)>,
}

#[pymethods]
impl PyFstMap {
    #[new]
    fn new(py: Python<'_>, entries: Vec<(String, u64)>) -> PyResult<Self> {
        let inner = py.detach(|| FstMap::build(entries.iter().map(|(k, v)| (k.as_str(), *v))));
        let inner = inner.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner, entries })
    }

    fn __getnewargs__(&self) -> (Vec<(String, u64)>,) {
        (self.entries.clone(),)
    }

    /// Look up a key and return its value, or None if not found.
    fn get(&self, key: &str) -> Option<u64> {
        self.inner.get(key)
    }

    /// Check if a key exists in the map.
    fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }
}

/// Register matching submodule.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "matching")?;

    // Substring
    m.add_function(wrap_pyfunction!(substring_find_all, &m)?)?;
    m.add_function(wrap_pyfunction!(substring_find_all_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(substring_find_first, &m)?)?;
    m.add_function(wrap_pyfunction!(substring_count, &m)?)?;
    m.add_function(wrap_pyfunction!(substring_count_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(substring_find_all_case_insensitive, &m)?)?;

    // Multi-pattern (Aho-Corasick)
    m.add_class::<PyMultiPatternMatcher>()?;

    // Regex
    m.add_class::<PyRegexMatcher>()?;
    m.add_class::<PyRegexSetMatcher>()?;

    // FST
    m.add_class::<PyFstSet>()?;
    m.add_class::<PyFstMap>()?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.matching", &m)?;

    Ok(())
}
