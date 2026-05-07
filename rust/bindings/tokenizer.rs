//! PyO3 bindings for the tokenizer module.
//!
//! Result shape: every `tokenize*` method returns
//! `Vec<PyTokenSpan>` (defined in `super::spans`) instead of dicts —
//! audit perf finding #1 / P3 measured ~11× slowdown for the
//! Rust→PyDict→Python-dataclass round trip vs. typed pyclasses on a
//! 10K-token text. Field shape (`text`, `start`, `end`) matches the
//! prior `@dataclass(frozen=True, slots=True) TokenSpan`, so callers
//! (`from kaos_nlp_core.types import TokenSpan`) keep working.

use pyo3::prelude::*;

use crate::core::tokenizer::{self, Token, TokenizerConfig};

use super::spans::PyTokenSpan;
use super::util::build_byte_to_char_table;

fn token_to_pyclass(t: &Token, char_offsets: Option<&[usize]>) -> PyTokenSpan {
    let (char_start, char_end) = match char_offsets {
        Some(table) => (table[t.start], table[t.end]),
        None => (t.start, t.end), // ASCII fast path: byte == char
    };
    PyTokenSpan {
        text: t.text.clone(),
        start: char_start,
        end: char_end,
    }
}

fn build_token_batch(text: &str, tokens: &[Token]) -> Vec<PyTokenSpan> {
    let char_offsets = build_byte_to_char_table(text);
    tokens
        .iter()
        .map(|t| token_to_pyclass(t, char_offsets.as_deref()))
        .collect()
}

/// Configurable word tokenizer with Unicode whitespace, punctuation stripping,
/// prefix truncation, stopword filtering, and span tracking.
///
/// Two tokenization methods:
/// - `tokenize(text)` — whitespace-scan based (configurable Unicode whitespace)
/// - `tokenize_regex(text, pattern=None)` — regex-based (custom patterns)
///
/// Both return a list of dicts with keys: text, start, end.
///
/// Example:
///     tok = Tokenizer(lowercase=True, prefix=4, stopwords=["the", "a"])
///     tokens = tok.tokenize("The Automobile Industry")
///     # → [{"text": "auto", "start": 4, "end": 14}, {"text": "indu", "start": 15, "end": 23}]
///     words = tok.tokenize_words("The Automobile Industry")
///     # → ["auto", "indu"]
#[pyclass(name = "Tokenizer", module = "kaos_nlp_core._rust.tokenizer")]
pub(crate) struct PyTokenizer {
    pub(crate) config: TokenizerConfig,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    #[pyo3(signature = (
        lowercase=false,
        keep_punctuation=false,
        prefix=0,
        stopwords=None,
        index=None,
    ))]
    fn new(
        lowercase: bool,
        keep_punctuation: bool,
        prefix: usize,
        stopwords: Option<Vec<String>>,
        index: Option<Vec<String>>,
    ) -> Self {
        let mut config = TokenizerConfig::new();
        if lowercase {
            config = config.lowercase();
        }
        if keep_punctuation {
            config = config.keep_punctuation();
        }
        if prefix > 0 {
            config = config.with_prefix(prefix);
        }
        if let Some(sw) = stopwords {
            config = config.with_stopwords(sw);
        }
        if let Some(idx) = index {
            config = config.with_index(idx);
        }
        Self { config }
    }

    fn __getnewargs__(&self) -> (bool, bool, usize, Option<Vec<String>>, Option<Vec<String>>) {
        let sw = if self.config.stopwords.is_empty() {
            None
        } else {
            Some(self.config.stopwords.iter().cloned().collect())
        };
        let idx = if self.config.index.is_empty() {
            None
        } else {
            Some(self.config.index.iter().cloned().collect())
        };
        (
            self.config.lowercase,
            self.config.keep_punctuation,
            self.config.prefix,
            sw,
            idx,
        )
    }

    /// Tokenize text using whitespace scanning.
    ///
    /// Returns a list of `TokenSpan` pyclasses (`text`, `start`, `end`).
    fn tokenize(&self, py: Python<'_>, text: &str) -> Vec<PyTokenSpan> {
        let config = &self.config;
        py.detach(|| {
            let tokens = tokenizer::tokenize(text, config);
            let char_offsets = build_byte_to_char_table(text);
            tokens
                .iter()
                .map(|t| token_to_pyclass(t, char_offsets.as_deref()))
                .collect()
        })
    }

    /// Tokenize and return only the token strings (no spans).
    fn tokenize_words(&self, py: Python<'_>, text: &str) -> Vec<String> {
        let config = &self.config;
        py.detach(|| tokenizer::tokenize_words(text, config))
    }

    /// Tokenize multiple texts in one call.
    fn tokenize_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<PyTokenSpan>> {
        let config = &self.config;
        py.detach(|| {
            texts
                .iter()
                .map(|text| {
                    let tokens = tokenizer::tokenize(text, config);
                    build_token_batch(text, &tokens)
                })
                .collect()
        })
    }

    /// Tokenize multiple texts and return token strings only.
    fn tokenize_words_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<String>> {
        let config = &self.config;
        py.detach(|| {
            texts
                .iter()
                .map(|text| tokenizer::tokenize_words(text, config))
                .collect()
        })
    }

    /// Tokenize using regex pattern matching.
    ///
    /// Args:
    ///     text: The text to tokenize.
    ///     pattern: Optional regex pattern string. If None, uses the default
    ///         non-whitespace pattern.
    ///
    /// Returns a list of `TokenSpan` pyclasses (`text`, `start`, `end`).
    #[pyo3(signature = (text, pattern=None))]
    fn tokenize_regex(
        &mut self,
        py: Python<'_>,
        text: &str,
        pattern: Option<&str>,
    ) -> PyResult<Vec<PyTokenSpan>> {
        // Compile regex with GIL (fast, needs error conversion)
        let re = match pattern {
            Some(p) => {
                let re = regex::Regex::new(p)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Some(re)
            }
            None => None,
        };

        let config = &self.config;
        Ok(py.detach(|| {
            let tokens = tokenizer::tokenize_regex(text, config, re.as_ref());
            let char_offsets = build_byte_to_char_table(text);
            tokens
                .iter()
                .map(|t| token_to_pyclass(t, char_offsets.as_deref()))
                .collect()
        }))
    }

    /// Tokenize using regex and return only the token strings.
    #[pyo3(signature = (text, pattern=None))]
    fn tokenize_regex_words(
        &mut self,
        py: Python<'_>,
        text: &str,
        pattern: Option<&str>,
    ) -> PyResult<Vec<String>> {
        let re = match pattern {
            Some(p) => {
                let re = regex::Regex::new(p)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Some(re)
            }
            None => None,
        };
        let config = &self.config;
        let tokens = py.detach(|| tokenizer::tokenize_regex(text, config, re.as_ref()));
        Ok(tokens.into_iter().map(|t| t.text).collect())
    }

    /// Tokenize multiple texts with an optional regex pattern.
    #[pyo3(signature = (texts, pattern=None))]
    fn tokenize_regex_batch(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        pattern: Option<&str>,
    ) -> PyResult<Vec<Vec<PyTokenSpan>>> {
        let re = match pattern {
            Some(p) => Some(
                regex::Regex::new(p)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };
        let config = &self.config;
        Ok(py.detach(|| {
            texts
                .iter()
                .map(|text| {
                    let tokens = tokenizer::tokenize_regex(text, config, re.as_ref());
                    build_token_batch(text, &tokens)
                })
                .collect()
        }))
    }

    /// Tokenize multiple texts with an optional regex pattern and return words only.
    #[pyo3(signature = (texts, pattern=None))]
    fn tokenize_regex_words_batch(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        pattern: Option<&str>,
    ) -> PyResult<Vec<Vec<String>>> {
        let re = match pattern {
            Some(p) => Some(
                regex::Regex::new(p)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };
        let config = &self.config;
        Ok(py.detach(|| {
            texts
                .iter()
                .map(|text| {
                    tokenizer::tokenize_regex(text, config, re.as_ref())
                        .into_iter()
                        .map(|token| token.text)
                        .collect::<Vec<_>>()
                })
                .collect()
        }))
    }
}

/// Convenience function: tokenize text with default settings.
///
/// Returns a list of `TokenSpan` pyclasses (`text`, `start`, `end`).
#[pyfunction]
#[pyo3(signature = (text, lowercase=false, prefix=0))]
fn tokenize(py: Python<'_>, text: &str, lowercase: bool, prefix: usize) -> Vec<PyTokenSpan> {
    let mut config = TokenizerConfig::new();
    if lowercase {
        config = config.lowercase();
    }
    if prefix > 0 {
        config = config.with_prefix(prefix);
    }
    py.detach(|| {
        let tokens = tokenizer::tokenize(text, &config);
        let char_offsets = build_byte_to_char_table(text);
        tokens
            .iter()
            .map(|t| token_to_pyclass(t, char_offsets.as_deref()))
            .collect()
    })
}

/// Convenience function: tokenize and return just the word strings.
#[pyfunction]
#[pyo3(signature = (text, lowercase=false, prefix=0))]
fn tokenize_words(py: Python<'_>, text: &str, lowercase: bool, prefix: usize) -> Vec<String> {
    let mut config = TokenizerConfig::new();
    if lowercase {
        config = config.lowercase();
    }
    if prefix > 0 {
        config = config.with_prefix(prefix);
    }
    py.detach(|| tokenizer::tokenize_words(text, &config))
}

/// Convenience function: tokenize many texts with default settings.
#[pyfunction]
#[pyo3(signature = (texts, lowercase=false, prefix=0))]
fn tokenize_batch(
    py: Python<'_>,
    texts: Vec<String>,
    lowercase: bool,
    prefix: usize,
) -> Vec<Vec<PyTokenSpan>> {
    let tokenizer = PyTokenizer::new(lowercase, false, prefix, None, None);
    tokenizer.tokenize_batch(py, texts)
}

/// Convenience function: tokenize many texts and return only words.
#[pyfunction]
#[pyo3(signature = (texts, lowercase=false, prefix=0))]
fn tokenize_words_batch(
    py: Python<'_>,
    texts: Vec<String>,
    lowercase: bool,
    prefix: usize,
) -> Vec<Vec<String>> {
    let tokenizer = PyTokenizer::new(lowercase, false, prefix, None, None);
    tokenizer.tokenize_words_batch(py, texts)
}

/// Register tokenizer submodule.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "tokenizer")?;

    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(tokenize, &m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_words, &m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_words_batch, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.tokenizer", &m)?;

    Ok(())
}
