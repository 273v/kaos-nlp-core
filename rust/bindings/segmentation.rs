//! PyO3 bindings for segmentation: PunktTokenizer, PunktTrainer, PunktParameters.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::sync::Arc;

use crate::bindings::util::{bincode_getstate, bincode_setstate};
use crate::core::segmentation::{
    detect_boilerplate, extract_line_records, normalize, parse_enumerator, parse_enumerator_with,
    segment_lines, segment_paragraphs, segment_paragraphs_simple, segment_sentences,
    BoilerplateKind, BoilerplateOptions, BoilerplateRun, CaseProfile, CustomLexicon, EnumKind,
    Enumerator, InferenceConfig, LineRecord, LineTerminator, NormalizeError, NormalizeOptions,
    PunctProfile, PunktParameters, PunktSentenceTokenizer, PunktTrainer, Segment, WordLexicon,
};

// Default Punkt model (~12 MB gzipped JSON) baked into _rust.abi3.so at
// compile time. Mirrors the embedded-OpenGloss-lexicon pattern in
// `bindings::lexicon` so end users get default sentence segmentation with
// no filesystem state after `pip install kaos-nlp-core`. The file ships in
// the sdist; the wheel ships only the .so that contains the bytes.
const EMBEDDED_PUNKT_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/python/kaos_nlp_core/models/default.npkt.gz"
));

// ─── PyPunktParameters ──────────────────────────────────────────────────────

/// Punkt model parameters (abbreviations, collocations, sentence starters, decision weights).
///
/// Supports JSON and compressed (.npkt.gz) serialization.
#[pyclass(
    name = "PunktParameters",
    module = "kaos_nlp_core._rust.segmentation",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyPunktParameters {
    inner: Arc<PunktParameters>,
}

#[pymethods]
impl PyPunktParameters {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(PunktParameters::new()),
        }
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let json = self
            .inner
            .to_json()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, json.as_bytes()).into())
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let json = std::str::from_utf8(state.as_bytes())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let params = PunktParameters::from_json(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.inner = Arc::new(params);
        Ok(())
    }

    /// Load from compressed file (.npkt.gz).
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_owned = path.to_string();
        let result =
            py.detach(|| PunktParameters::load_compressed(&path_owned).map_err(|e| e.to_string()));
        result
            .map(|p| Self { inner: Arc::new(p) })
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Load the default Punkt model embedded in this `_rust` shared object.
    /// ~12 MB gzipped JSON baked in at build time via `include_bytes!`. End
    /// users get a working sentence/paragraph segmenter after
    /// `pip install kaos-nlp-core` with no filesystem state, mirroring the
    /// embedded-OpenGloss-lexicon pattern. The override path
    /// (`PunktParameters.load(path)`) still works for users with custom
    /// trained models.
    #[staticmethod]
    fn default_embedded(py: Python<'_>) -> PyResult<Self> {
        let result = py.detach(|| {
            PunktParameters::from_compressed_bytes(EMBEDDED_PUNKT_BYTES)
                .map_err(|e| e.to_string())
        });
        result
            .map(|p| Self { inner: Arc::new(p) })
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Save to compressed file (.npkt.gz).
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let path_owned = path.to_string();
        let inner = &self.inner;
        py.detach(|| {
            inner
                .save_compressed(&path_owned)
                .map_err(|e| e.to_string())
        })
        .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Load from JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        PunktParameters::from_json(json)
            .map(|p| Self { inner: Arc::new(p) })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn num_abbreviations(&self) -> usize {
        self.inner.abbrev_types.len()
    }

    #[getter]
    fn num_collocations(&self) -> usize {
        self.inner.collocations.len()
    }

    #[getter]
    fn num_sent_starters(&self) -> usize {
        self.inner.sent_starters.len()
    }

    #[getter]
    fn abbreviations(&self) -> Vec<String> {
        self.inner.abbrev_types.keys().cloned().collect()
    }

    #[getter]
    fn collocations(&self) -> Vec<(String, String)> {
        self.inner.collocations.iter().cloned().collect()
    }

    #[getter]
    fn sent_starters(&self) -> Vec<String> {
        self.inner.sent_starters.iter().cloned().collect()
    }

    fn __len__(&self) -> usize {
        self.inner.abbrev_types.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PunktParameters(abbreviations={}, collocations={}, starters={})",
            self.inner.abbrev_types.len(),
            self.inner.collocations.len(),
            self.inner.sent_starters.len()
        )
    }
}

// ─── PyPunktTokenizer ───────────────────────────────────────────────────────

/// Punkt sentence tokenizer.
///
/// Splits text into sentences using the Punkt algorithm.
/// Supports tunable precision/recall balance and custom model parameters.
///
/// Args:
///     params: Optional PunktParameters (default: empty model).
///
/// Example:
///     tokenizer = PunktTokenizer()
///     sentences = tokenizer.tokenize("Hello world. How are you?")
#[pyclass(name = "PunktTokenizer", module = "kaos_nlp_core._rust.segmentation")]
pub(crate) struct PyPunktTokenizer {
    pub(crate) inner: PunktSentenceTokenizer,
}

#[pymethods]
impl PyPunktTokenizer {
    #[new]
    #[pyo3(signature = (params=None))]
    fn new(params: Option<&PyPunktParameters>) -> Self {
        let tokenizer = match params {
            Some(p) => PunktSentenceTokenizer::from_parameters(p.inner.clone()),
            None => PunktSentenceTokenizer::new(),
        };
        Self { inner: tokenizer }
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let json = self
            .inner
            .parameters()
            .to_json()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let pr = self.inner.inference_config().precision_recall_balance;
        let state = format!("{}|{}", pr, json);
        Ok(PyBytes::new(py, state.as_bytes()).into())
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let s = std::str::from_utf8(state.as_bytes())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let (pr_str, json) = s
            .split_once('|')
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid state format"))?;
        let pr: f64 = pr_str.parse().map_err(|e: std::num::ParseFloatError| {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        })?;
        let params = PunktParameters::from_json(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.inner = PunktSentenceTokenizer::from_parameters(Arc::new(params));
        self.inner.set_precision_recall_balance(pr);
        Ok(())
    }

    /// Tokenize text into sentences.
    ///
    /// Args:
    ///     text: Text to tokenize.
    ///     precision_recall: Optional PR balance override (0.0–1.0).
    #[pyo3(signature = (text, precision_recall=None))]
    fn tokenize(&self, py: Python<'_>, text: &str, precision_recall: Option<f64>) -> Vec<String> {
        let inner = &self.inner;
        py.detach(|| match precision_recall {
            Some(pr) => {
                let config = InferenceConfig {
                    precision_recall_balance: pr.clamp(0.0, 1.0),
                };
                inner.tokenize_with_config(text, &config)
            }
            None => inner.tokenize(text),
        })
    }

    /// Get sentence boundaries as (start, end) character spans.
    ///
    /// Spans are character offsets suitable for Python string slicing.
    /// Use text[start:end] to extract each sentence.
    #[pyo3(signature = (text, precision_recall=None))]
    fn tokenize_spans(
        &self,
        py: Python<'_>,
        text: &str,
        precision_recall: Option<f64>,
    ) -> Vec<(usize, usize)> {
        let inner = &self.inner;
        py.detach(|| {
            let byte_spans = match precision_recall {
                Some(pr) => {
                    let config = InferenceConfig {
                        precision_recall_balance: pr.clamp(0.0, 1.0),
                    };
                    inner.tokenize_spans_with_config(text, &config)
                }
                None => inner.tokenize_spans(text),
            };

            // Convert byte offsets to char offsets for Python
            if text.is_ascii() {
                byte_spans // byte == char for ASCII
            } else {
                let char_offsets = build_byte_to_char_table(text);
                byte_spans
                    .into_iter()
                    .map(|(start, end)| {
                        let cs = if start < char_offsets.len() {
                            char_offsets[start]
                        } else {
                            char_offsets.last().copied().unwrap_or(0)
                        };
                        let ce = if end < char_offsets.len() {
                            char_offsets[end]
                        } else {
                            char_offsets.last().copied().unwrap_or(0)
                        };
                        (cs, ce)
                    })
                    .collect()
            }
        })
    }

    /// Tokenize into paragraphs (list of lists of sentences).
    #[pyo3(signature = (text, precision_recall=None))]
    fn tokenize_paragraphs(
        &self,
        py: Python<'_>,
        text: &str,
        precision_recall: Option<f64>,
    ) -> Vec<Vec<String>> {
        let inner = &self.inner;
        py.detach(|| match precision_recall {
            Some(pr) => {
                let mut temp = inner.clone();
                temp.set_precision_recall_balance(pr);
                temp.tokenize_paragraphs(text)
            }
            None => inner.tokenize_paragraphs(text),
        })
    }

    /// Tokenize into paragraphs as flat strings.
    #[pyo3(signature = (text, precision_recall=None))]
    fn tokenize_paragraphs_flat(
        &self,
        py: Python<'_>,
        text: &str,
        precision_recall: Option<f64>,
    ) -> Vec<String> {
        let inner = &self.inner;
        py.detach(|| match precision_recall {
            Some(pr) => {
                let mut temp = inner.clone();
                temp.set_precision_recall_balance(pr);
                temp.tokenize_paragraphs_flat(text)
            }
            None => inner.tokenize_paragraphs_flat(text),
        })
    }

    /// Set precision/recall balance (0.0 = max recall, 1.0 = max precision).
    fn set_precision_recall(&mut self, balance: f64) {
        self.inner.set_precision_recall_balance(balance);
    }

    /// Count sentences without returning the strings.
    fn count_sentences(&self, py: Python<'_>, text: &str) -> usize {
        let inner = &self.inner;
        py.detach(|| inner.tokenize(text).len())
    }

    /// Tokenize multiple texts in a single call.
    ///
    /// Uses rayon parallelism for batches of 4+ texts.
    #[pyo3(signature = (texts, precision_recall=None))]
    fn tokenize_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        precision_recall: Option<f64>,
    ) -> Vec<Vec<String>> {
        let inner = &self.inner;
        py.detach(|| match precision_recall {
            Some(pr) => {
                let config = InferenceConfig {
                    precision_recall_balance: pr.clamp(0.0, 1.0),
                };
                inner.tokenize_batch_parallel_with_config(&texts, &config)
            }
            None => inner.tokenize_batch_parallel(&texts),
        })
    }
}

// ─── PyPunktTrainer ─────────────────────────────────────────────────────────

/// Punkt trainer: learn sentence boundary parameters from text.
///
/// Example:
///     trainer = PunktTrainer()
///     trainer.add_abbreviations(["Dr.", "Mr.", "U.S."])
///     params = trainer.train("Training text here...")
///     tokenizer = PunktTokenizer(params)
#[pyclass(name = "PunktTrainer", module = "kaos_nlp_core._rust.segmentation")]
struct PyPunktTrainer {
    inner: PunktTrainer,
}

#[pymethods]
impl PyPunktTrainer {
    #[new]
    fn new() -> Self {
        Self {
            inner: PunktTrainer::new(),
        }
    }

    /// Train on text and return learned parameters.
    #[pyo3(signature = (text, verbose=false))]
    fn train(&mut self, py: Python<'_>, text: &str, verbose: bool) -> PyResult<PyPunktParameters> {
        let inner = &mut self.inner;
        let result = py.detach(|| inner.train(text, verbose).map_err(|e| e.to_string()));
        result
            .map(|p| PyPunktParameters { inner: Arc::new(p) })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Load abbreviations from a JSON file (list of strings).
    fn load_abbreviations_from_json(&mut self, path: &str) -> PyResult<usize> {
        self.inner
            .load_abbreviations_from_json(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Add abbreviations directly (marked as provided/ground truth).
    fn add_abbreviations(&mut self, abbreviations: Vec<String>) {
        self.inner.add_abbreviations(abbreviations);
    }

    /// Train incrementally on a chunk of text.
    fn train_incremental(&mut self, py: Python<'_>, text: &str) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| inner.train_incremental(text).map_err(|e| e.to_string()))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Finalize training after all chunks and return parameters.
    #[pyo3(signature = (verbose=false))]
    fn finalize_training(&mut self, py: Python<'_>, verbose: bool) -> PyResult<PyPunktParameters> {
        let inner = &mut self.inner;
        let result = py.detach(|| inner.finalize_training(verbose).map_err(|e| e.to_string()));
        result
            .map(|p| PyPunktParameters { inner: Arc::new(p) })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
}

// ─── Byte-to-char offset conversion ─────────────────────────────────────────

/// Build a byte-index → char-index lookup table for non-ASCII text.
fn build_byte_to_char_table(text: &str) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(text.len() + 1);
    let mut char_count = 0;
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

// ─── Standalone functions ───────────────────────────────────────────────────

/// Split text into lines. Returns list of dicts with start, end, text.
#[pyfunction]
fn py_segment_lines(py: Python<'_>, text: &str) -> PyResult<Py<PyList>> {
    let segs = segment_lines(text);
    segments_to_pylist(py, text, &segs)
}

/// Segment text into sentences using the Punkt algorithm.
///
/// Args:
///     text: Text to segment.
///     tokenizer: Optional PunktTokenizer. If None, uses an empty model.
///
/// Returns list of dicts with start, end, text, confidence.
#[pyfunction]
#[pyo3(signature = (text, tokenizer=None))]
fn py_segment_sentences(
    py: Python<'_>,
    text: &str,
    tokenizer: Option<&PyPunktTokenizer>,
) -> PyResult<Py<PyList>> {
    let default_tok;
    let tok = match tokenizer {
        Some(t) => &t.inner,
        None => {
            default_tok = PunktSentenceTokenizer::new();
            &default_tok
        }
    };
    let segs = py.detach(|| segment_sentences(text, tok));
    segments_to_pylist(py, text, &segs)
}

/// Segment text into paragraphs using sentence-aware boundaries.
///
/// Paragraph breaks only occur at sentence boundaries followed by blank lines.
///
/// Args:
///     text: Text to segment.
///     tokenizer: Optional PunktTokenizer. If None, uses an empty model.
///
/// Returns list of dicts with start, end, text, confidence.
#[pyfunction]
#[pyo3(signature = (text, tokenizer=None))]
fn py_segment_paragraphs(
    py: Python<'_>,
    text: &str,
    tokenizer: Option<&PyPunktTokenizer>,
) -> PyResult<Py<PyList>> {
    let default_tok;
    let tok = match tokenizer {
        Some(t) => &t.inner,
        None => {
            default_tok = PunktSentenceTokenizer::new();
            &default_tok
        }
    };
    let segs = py.detach(|| segment_paragraphs(text, tok));
    segments_to_pylist(py, text, &segs)
}

/// Split text into paragraphs by blank lines only (no sentence awareness).
///
/// Use `segment_paragraphs()` for sentence-aware splitting.
#[pyfunction]
fn py_segment_paragraphs_simple(py: Python<'_>, text: &str) -> PyResult<Py<PyList>> {
    let segs = segment_paragraphs_simple(text);
    segments_to_pylist(py, text, &segs)
}

fn segments_to_pylist(py: Python<'_>, text: &str, segs: &[Segment]) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);

    if text.is_ascii() {
        // ASCII fast path: byte offsets == char offsets
        for seg in segs {
            let d = PyDict::new(py);
            d.set_item("start", seg.start)?;
            d.set_item("end", seg.end)?;
            d.set_item("text", seg.text(text))?;
            d.set_item("confidence", seg.confidence)?;
            list.append(d)?;
        }
    } else {
        // Unicode: single-pass offset table
        let char_offsets = build_byte_to_char_table(text);
        for seg in segs {
            let d = PyDict::new(py);
            d.set_item("start", char_offsets[seg.start])?;
            d.set_item("end", char_offsets[seg.end])?;
            d.set_item("text", seg.text(text))?;
            d.set_item("confidence", seg.confidence)?;
            list.append(d)?;
        }
    }

    Ok(list.into())
}

// ─── PyLineRecord ──────────────────────────────────────────────────────────
//
// Wraps `core::segmentation::LineRecord`. Offsets stored on the wrapper are
// **char offsets** (computed once at construction via `build_byte_to_char_table`)
// because Python's `str` indexing is char-based; the underlying Rust core
// preserves byte offsets. This translation is the standing rule across this
// crate — never expose byte offsets through PyO3.

/// One physical-line record with offset-preserving layout features.
///
/// Fields use **character offsets** (Python `str` indexing). The kind of
/// `terminator` is exposed as a string: `"none"`, `"lf"`, `"cr"`, `"crlf"`,
/// or `"other_unicode"`.
#[pyclass(
    name = "LineRecord",
    module = "kaos_nlp_core._rust.segmentation",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyLineRecord {
    inner: LineRecord,
    char_start: u32,
    char_end: u32,
    char_stripped_start: u32,
    char_stripped_end: u32,
    char_term_len: u32,
}

#[pymethods]
impl PyLineRecord {
    #[getter]
    fn start(&self) -> u32 {
        self.char_start
    }
    #[getter]
    fn end(&self) -> u32 {
        self.char_end
    }
    #[getter]
    fn stripped_start(&self) -> u32 {
        self.char_stripped_start
    }
    #[getter]
    fn stripped_end(&self) -> u32 {
        self.char_stripped_end
    }
    #[getter]
    fn term_len(&self) -> u32 {
        self.char_term_len
    }
    #[getter]
    fn terminator(&self) -> &'static str {
        match self.inner.terminator {
            LineTerminator::None => "none",
            LineTerminator::Lf => "lf",
            LineTerminator::Cr => "cr",
            LineTerminator::CrLf => "crlf",
            LineTerminator::OtherUnicode => "other_unicode",
        }
    }
    #[getter]
    fn indent_chars(&self) -> u16 {
        self.inner.indent_chars
    }
    #[getter]
    fn byte_len(&self) -> u32 {
        self.inner.byte_len
    }
    #[getter]
    fn char_len(&self) -> u32 {
        self.inner.char_len
    }
    #[getter]
    fn token_count(&self) -> u16 {
        self.inner.token_count
    }
    #[getter]
    fn case_profile(&self) -> &'static str {
        match self.inner.case_profile {
            CaseProfile::NoAlpha => "no_alpha",
            CaseProfile::AllCaps => "all_caps",
            CaseProfile::TitleCase => "title_case",
            CaseProfile::InitialCap => "initial_cap",
            CaseProfile::AllLower => "all_lower",
            CaseProfile::MixedCase => "mixed_case",
        }
    }
    /// Bitfield of `PunctProfile` flags as a u16; see the Python wrapper for
    /// flag constants.
    #[getter]
    fn punct_profile(&self) -> u16 {
        self.inner.punct_profile.bits()
    }
    #[getter]
    fn blank(&self) -> bool {
        self.inner.blank
    }
    #[getter]
    fn blank_before(&self) -> bool {
        self.inner.blank_before
    }
    #[getter]
    fn blank_after(&self) -> bool {
        self.inner.blank_after
    }

    fn __repr__(&self) -> String {
        format!(
            "LineRecord(start={}, end={}, indent={}, blank={}, case={:?})",
            self.char_start,
            self.char_end,
            self.inner.indent_chars,
            self.inner.blank,
            self.inner.case_profile
        )
    }

    /// Pickle support via postcard (per the standing crate-wide pattern).
    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Pack the wrapper-side char offsets alongside the inner core record.
        let state = (
            &self.inner,
            self.char_start,
            self.char_end,
            self.char_stripped_start,
            self.char_stripped_end,
            self.char_term_len,
        );
        bincode_getstate(py, &state)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let (inner, char_start, char_end, char_stripped_start, char_stripped_end, char_term_len): (
            LineRecord,
            u32,
            u32,
            u32,
            u32,
            u32,
        ) = bincode_setstate(state)?;
        self.inner = inner;
        self.char_start = char_start;
        self.char_end = char_end;
        self.char_stripped_start = char_stripped_start;
        self.char_stripped_end = char_stripped_end;
        self.char_term_len = char_term_len;
        Ok(())
    }

    #[new]
    fn __new__() -> Self {
        // Required for pickle: a placeholder we'll overwrite in __setstate__.
        Self {
            inner: LineRecord {
                start: 0,
                end: 0,
                term_len: 0,
                terminator: LineTerminator::None,
                stripped_start: 0,
                stripped_end: 0,
                indent_chars: 0,
                byte_len: 0,
                char_len: 0,
                token_count: 0,
                case_profile: CaseProfile::NoAlpha,
                punct_profile: PunctProfile::empty(),
                blank: true,
                blank_before: false,
                blank_after: false,
            },
            char_start: 0,
            char_end: 0,
            char_stripped_start: 0,
            char_stripped_end: 0,
            char_term_len: 0,
        }
    }
}

// ─── Normalizer binding ────────────────────────────────────────────────────
//
// Wraps `core::segmentation::normalize`. The Rust core stores byte offsets
// (since `Normalized.orig_offsets` is indexed by char position in the
// **normalized** output and holds **byte** offsets in the source); the
// PyO3 binding converts the *source-side byte offsets* in the table to
// *source-side char offsets* before exposing them, so Python consumers can
// index the original `str` with the values directly.

/// Result of [`py_normalize`]. Wraps the normalized text and a list of
/// source-side character offsets (one per char in `.text`).
#[pyclass(
    name = "NormalizedText",
    module = "kaos_nlp_core._rust.segmentation",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyNormalizedText {
    /// The normalized text. Ownership is on the Python side once returned.
    #[pyo3(get)]
    text: String,
    /// `None` when the fast path returned the input unchanged. Otherwise
    /// `len(orig_char_offsets) == len(text)` and each entry is the source
    /// **character** offset of the source codepoint that contributed it.
    #[pyo3(get)]
    orig_char_offsets: Option<Vec<u32>>,
}

#[pymethods]
impl PyNormalizedText {
    /// Resolve the source-side char offset of the source codepoint that
    /// produced ``text[char_index]``. Returns ``None`` when out of range.
    fn original_char(&self, char_index: usize) -> Option<u32> {
        match &self.orig_char_offsets {
            None => {
                let n = self.text.chars().count();
                if char_index <= n {
                    Some(char_index as u32)
                } else {
                    None
                }
            }
            Some(map) => map.get(char_index).copied(),
        }
    }

    fn __repr__(&self) -> String {
        let len = self.text.chars().count();
        let mapped = self
            .orig_char_offsets
            .as_ref()
            .map(|m| m.len())
            .unwrap_or(0);
        format!("NormalizedText(chars={}, mapped={})", len, mapped)
    }
}

/// Normalize `text` per the boolean flags. See `kaos_nlp_core.segmentation`
/// Python wrapper for canonical kwarg names; this PyO3 entrypoint exposes
/// every flag as a positional kwarg so the wrapper can stay forward-compatible.
#[pyfunction]
#[pyo3(signature = (
    text,
    *,
    collapse_whitespace = false,
    fold_case = false,
    normalize_unicode_punct = false,
    strip_enumerator_prefix = false,
    strip_punctuation = false,
))]
#[allow(clippy::too_many_arguments)]
fn py_normalize(
    py: Python<'_>,
    text: &str,
    collapse_whitespace: bool,
    fold_case: bool,
    normalize_unicode_punct: bool,
    strip_enumerator_prefix: bool,
    strip_punctuation: bool,
) -> PyResult<PyNormalizedText> {
    let opts = NormalizeOptions {
        collapse_whitespace,
        fold_case,
        normalize_unicode_punct,
        strip_enumerator_prefix,
        strip_punctuation,
    };
    let owned = text.to_string();
    // Run the (potentially expensive) normalization off the GIL.
    let result = py.detach(|| {
        let normalized = normalize(&owned, opts)?;
        // Convert the borrowed/owned outcome to Python-friendly state, plus
        // translate source-side byte offsets to source-side char offsets if
        // an offsets vector is present. We do this in one place so the
        // Python wrapper never sees byte offsets.
        let needs_table = !owned.is_ascii() && normalized.orig_offsets.is_some();
        let table = if needs_table {
            Some(build_byte_to_char_table(&owned))
        } else {
            None
        };
        let out_text = normalized.text.into_owned();
        let out_offsets = normalized.orig_offsets.map(|byte_offsets| {
            byte_offsets
                .into_iter()
                .map(|b| match &table {
                    None => b, // ASCII source: byte == char
                    Some(t) => {
                        let i = (b as usize).min(t.len() - 1);
                        t[i] as u32
                    }
                })
                .collect::<Vec<u32>>()
        });
        Ok::<_, NormalizeError>((out_text, out_offsets))
    });
    match result {
        Ok((text, offsets)) => Ok(PyNormalizedText {
            text,
            orig_char_offsets: offsets,
        }),
        Err(NormalizeError::UnsupportedOption(msg)) => Err(
            pyo3::exceptions::PyNotImplementedError::new_err(msg.to_string()),
        ),
    }
}

/// Extract `LineRecord`s for `text`, with all offsets converted to chars.
#[pyfunction]
fn py_extract_line_records(py: Python<'_>, text: &str) -> PyResult<Py<PyList>> {
    let owned = text.to_string();
    let (records, table) = py.detach(|| {
        let recs = extract_line_records(&owned);
        let table = if owned.is_ascii() {
            None
        } else {
            Some(build_byte_to_char_table(&owned))
        };
        (recs, table)
    });

    let to_char = |byte_off: u32| -> u32 {
        match &table {
            None => byte_off,
            Some(t) => {
                // Defensive clamp; the table is sized to text.len() + 1.
                let idx = (byte_off as usize).min(t.len() - 1);
                t[idx] as u32
            }
        }
    };

    let list = PyList::empty(py);
    for r in records {
        let char_start = to_char(r.start);
        let char_end = to_char(r.end);
        let char_stripped_start = to_char(r.stripped_start);
        let char_stripped_end = to_char(r.stripped_end);
        let char_term_len = to_char(r.start + r.byte_len + r.term_len) - char_end;
        list.append(PyLineRecord {
            inner: r,
            char_start,
            char_end,
            char_stripped_start,
            char_stripped_end,
            char_term_len,
        })?;
    }
    Ok(list.into())
}

// ─── Boilerplate detector binding ──────────────────────────────────────────
//
// Wraps `core::segmentation::detect_boilerplate`. The Rust core takes a slice
// of `LineRecord`; the binding rebuilds that slice from the source string
// (so the Python caller does not have to construct a list of opaque records
// at the FFI boundary). All offsets exposed to Python are char offsets
// inherited from the surrounding `extract_line_records` convention.

/// One detected run of repeated boilerplate.
#[pyclass(
    name = "BoilerplateRun",
    module = "kaos_nlp_core._rust.segmentation",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyBoilerplateRun {
    inner: BoilerplateRun,
}

#[pymethods]
impl PyBoilerplateRun {
    /// Indices into the line-record list that produced this run.
    #[getter]
    fn line_indices(&self) -> Vec<u32> {
        self.inner.line_indices.clone()
    }
    /// Normalized canonical form of the recurring line.
    #[getter]
    fn canonical_text(&self) -> &str {
        &self.inner.canonical_text
    }
    #[getter]
    fn occurrences(&self) -> u32 {
        self.inner.occurrences
    }
    #[getter]
    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint
    }
    /// One of: ``"page_number"``, ``"caption"``, ``"header"``, ``"footer"``,
    /// ``"unknown"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            BoilerplateKind::PageNumber => "page_number",
            BoilerplateKind::Caption => "caption",
            BoilerplateKind::Header => "header",
            BoilerplateKind::Footer => "footer",
            BoilerplateKind::Unknown => "unknown",
        }
    }
    /// For caption runs only: the Western-language lexicon that matched the
    /// caption prefix (one of ``"english"``, ``"german"``, ``"french"``,
    /// ``"spanish"``, ``"italian"``, ``"portuguese"``). ``None`` for
    /// non-caption runs. P7.0f generality contract.
    #[getter]
    fn language_hint(&self) -> Option<String> {
        self.inner.language_hint.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "BoilerplateRun(kind={:?}, occurrences={}, canonical={:?})",
            self.kind(),
            self.inner.occurrences,
            self.inner.canonical_text,
        )
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bincode_getstate(py, &self.inner)
    }
    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = bincode_setstate(state)?;
        Ok(())
    }
    #[new]
    fn __new__() -> Self {
        Self {
            inner: BoilerplateRun {
                line_indices: Vec::new(),
                canonical_text: String::new(),
                occurrences: 0,
                fingerprint: 0,
                kind: BoilerplateKind::Unknown,
                language_hint: None,
            },
        }
    }
}

/// Detect boilerplate runs in `text`. Returns a list of `BoilerplateRun`.
///
/// All keyword arguments default to the values documented on the Python
/// wrapper. The function rebuilds line records internally, so the caller
/// does not have to construct a list of `LineRecord` objects.
#[pyfunction]
#[pyo3(signature = (
    text,
    *,
    lines_per_page = 50,
    header_zone_lines = 3,
    footer_zone_lines = 3,
    min_occurrences = 3,
    min_rate = 0.5,
    skip_near_dup = false,
    near_dup_threshold = 0.75,
    num_perm = 64,
    shingle_size = 4,
    zone_dominance = 0.7,
    drop_empty = true,
))]
#[allow(clippy::too_many_arguments)]
fn py_detect_boilerplate(
    py: Python<'_>,
    text: &str,
    lines_per_page: u32,
    header_zone_lines: u32,
    footer_zone_lines: u32,
    min_occurrences: u32,
    min_rate: f64,
    skip_near_dup: bool,
    near_dup_threshold: f64,
    num_perm: usize,
    shingle_size: usize,
    zone_dominance: f64,
    drop_empty: bool,
) -> PyResult<Vec<PyBoilerplateRun>> {
    let owned = text.to_string();
    let opts = BoilerplateOptions {
        lines_per_page,
        header_zone_lines,
        footer_zone_lines,
        min_occurrences,
        min_rate,
        skip_near_dup,
        near_dup_threshold,
        num_perm,
        shingle_size,
        zone_dominance,
        drop_empty,
    };
    let runs = py.detach(|| {
        let recs = extract_line_records(&owned);
        detect_boilerplate(&recs, &owned, opts)
    });
    Ok(runs
        .into_iter()
        .map(|inner| PyBoilerplateRun { inner })
        .collect())
}

// ─── Enumerator parser binding ─────────────────────────────────────────────
//
// Wraps `core::segmentation::parse_enumerator`. The Rust core works in byte
// offsets; the binding converts to char offsets at the FFI boundary per the
// standing rule.

/// One detected enumerator at the start of a line.
#[pyclass(
    name = "Enumerator",
    module = "kaos_nlp_core._rust.segmentation",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyEnumerator {
    inner: Enumerator,
    char_raw_start: u32,
    char_raw_end: u32,
    char_prefix_end: u32,
}

#[pymethods]
impl PyEnumerator {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            EnumKind::RomanUpper => "roman_upper",
            EnumKind::RomanLower => "roman_lower",
            EnumKind::AlphaUpper => "alpha_upper",
            EnumKind::AlphaLower => "alpha_lower",
            EnumKind::Decimal => "decimal",
            EnumKind::ParenAlpha => "paren_alpha",
            EnumKind::ParenDecimal => "paren_decimal",
            EnumKind::ParenRoman => "paren_roman",
            EnumKind::Section => "section",
            EnumKind::SectionWord => "section_word",
            EnumKind::ChapterWord => "chapter_word",
            EnumKind::SubpartWord => "subpart_word",
            EnumKind::Bullet => "bullet",
        }
    }

    /// Packed value. For Decimal / ParenDecimal kinds this is the dotted-
    /// decimal segment encoding `(s1<<24)|(s2<<16)|(s3<<8)|s4`. For Roman
    /// kinds it is the Roman value 1..=3999. For Alpha kinds it is the
    /// letter index 1..=26. See `Enumerator.value` Rust docs for the full
    /// encoding table.
    #[getter]
    fn value(&self) -> u32 {
        self.inner.value
    }

    /// Number of dotted-decimal segments (1..=4 for Decimal). 1 for all
    /// other kinds.
    #[getter]
    fn depth(&self) -> u8 {
        self.inner.depth
    }

    #[getter]
    fn raw_start(&self) -> u32 {
        self.char_raw_start
    }
    #[getter]
    fn raw_end(&self) -> u32 {
        self.char_raw_end
    }
    /// Char offset where the heading text begins (points past the
    /// enumerator + any trailing space).
    #[getter]
    fn prefix_end(&self) -> u32 {
        self.char_prefix_end
    }

    /// Decode Decimal `value` into a list of segment integers.
    ///
    /// * Single-segment (`depth = 1`): returns `[value]` directly.
    ///   Per F-R7, single-segment values can exceed 255 (e.g. USC
    ///   section 271).
    /// * Multi-segment (`depth >= 2`): each byte of `value` is one
    ///   segment, e.g. `0x01020300` (`1.2.3`) -> `[1, 2, 3]`.
    /// * Non-Decimal kinds: returns `[value]` directly.
    fn segments(&self) -> Vec<u32> {
        if !matches!(self.inner.kind, EnumKind::Decimal | EnumKind::ParenDecimal) {
            return vec![self.inner.value];
        }
        if self.inner.depth <= 1 {
            return vec![self.inner.value];
        }
        let mut segs = Vec::with_capacity(self.inner.depth as usize);
        for i in 0..self.inner.depth {
            let shift = 8 * (3 - i);
            let v = (self.inner.value >> shift) & 0xFF;
            segs.push(v);
        }
        segs
    }

    fn __repr__(&self) -> String {
        format!(
            "Enumerator(kind={:?}, value={}, depth={}, prefix_end={})",
            self.kind(),
            self.inner.value,
            self.inner.depth,
            self.char_prefix_end,
        )
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = (
            &self.inner,
            self.char_raw_start,
            self.char_raw_end,
            self.char_prefix_end,
        );
        bincode_getstate(py, &state)
    }
    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let (inner, crs, cre, cpe): (Enumerator, u32, u32, u32) = bincode_setstate(state)?;
        self.inner = inner;
        self.char_raw_start = crs;
        self.char_raw_end = cre;
        self.char_prefix_end = cpe;
        Ok(())
    }
    #[new]
    fn __new__() -> Self {
        Self {
            inner: Enumerator {
                kind: EnumKind::Decimal,
                value: 0,
                depth: 0,
                raw_start: 0,
                raw_end: 0,
                prefix_end: 0,
            },
            char_raw_start: 0,
            char_raw_end: 0,
            char_prefix_end: 0,
        }
    }
}

/// Parse the leading enumerator of `line`. Caller is expected to pre-strip
/// leading whitespace; the parser anchors at byte 0. Returns `None` when
/// no enumerator-shaped prefix is found.
///
/// Optional `lexicon` selects the word-prefix dictionary used for
/// `Section / Chapter / Article / …` matching. Accepted values:
///
///     "english_legal_us" (default — anchor for US statutes / EDGAR)
///     "french_legal"     (Article / Chapitre / Titre / Section / Annexe / Préambule / Livre)
///     "german_legal"     (Artikel / Kapitel / Titel / Abschnitt / Anhang / Buch / Teil)
///     "spanish_legal"    (Artículo / Capítulo / Título / Sección / Anexo / Libro / Parte)
///     "italian_legal"    (Articolo / Capo / Capitolo / Titolo / Sezione / Allegato / Libro)
///     "portuguese_legal" (Artigo / Capítulo / Título / Secção/Seção / Anexo / Livro / Parte)
///     "markdown_atx"     (depth from `#` count; H1..H6)
///
/// Pass a list of `(pattern, kind)` tuples via `custom_lexicon` to override
/// with your own keyword set; if both `lexicon` and `custom_lexicon` are
/// supplied the custom one wins.
#[pyfunction]
#[pyo3(signature = (line, *, lexicon = None, custom_lexicon = None))]
fn py_parse_enumerator(
    line: &str,
    lexicon: Option<&str>,
    custom_lexicon: Option<Vec<(String, String)>>,
) -> PyResult<Option<PyEnumerator>> {
    let lex = if let Some(entries) = custom_lexicon {
        let mut tagged: Vec<(String, EnumKind)> = Vec::with_capacity(entries.len());
        for (pat, kind_str) in entries {
            let kind = enum_kind_from_str(&kind_str).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown enum kind {kind_str:?}; valid kinds: {}",
                    ENUM_KIND_NAMES.join(", ")
                ))
            })?;
            tagged.push((pat, kind));
        }
        let custom = CustomLexicon::new(tagged).map_err(pyo3::exceptions::PyValueError::new_err)?;
        WordLexicon::Custom(Arc::new(custom))
    } else {
        match lexicon {
            None | Some("english_legal_us") => WordLexicon::EnglishLegalUs,
            Some("french_legal") => WordLexicon::FrenchLegal,
            Some("german_legal") => WordLexicon::GermanLegal,
            Some("spanish_legal") => WordLexicon::SpanishLegal,
            Some("italian_legal") => WordLexicon::ItalianLegal,
            Some("portuguese_legal") => WordLexicon::PortugueseLegal,
            Some("markdown_atx") => WordLexicon::MarkdownAtx,
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown lexicon {other:?}; valid options: english_legal_us, french_legal, \
                     german_legal, spanish_legal, italian_legal, portuguese_legal, markdown_atx"
                )));
            }
        }
    };
    let inner = match parse_enumerator_with(line, &lex) {
        Some(e) => e,
        None => return Ok(None),
    };
    let needs_table = !line.is_ascii();
    let table = if needs_table {
        Some(build_byte_to_char_table(line))
    } else {
        None
    };
    let to_char = |byte_off: u32| -> u32 {
        match &table {
            None => byte_off,
            Some(t) => t[(byte_off as usize).min(t.len() - 1)] as u32,
        }
    };
    Ok(Some(PyEnumerator {
        char_raw_start: to_char(inner.raw_start),
        char_raw_end: to_char(inner.raw_end),
        char_prefix_end: to_char(inner.prefix_end),
        inner,
    }))
}

const ENUM_KIND_NAMES: &[&str] = &[
    "roman_upper",
    "roman_lower",
    "alpha_upper",
    "alpha_lower",
    "decimal",
    "paren_alpha",
    "paren_decimal",
    "paren_roman",
    "section",
    "section_word",
    "chapter_word",
    "subpart_word",
    "bullet",
];

fn enum_kind_from_str(s: &str) -> Option<EnumKind> {
    match s {
        "roman_upper" => Some(EnumKind::RomanUpper),
        "roman_lower" => Some(EnumKind::RomanLower),
        "alpha_upper" => Some(EnumKind::AlphaUpper),
        "alpha_lower" => Some(EnumKind::AlphaLower),
        "decimal" => Some(EnumKind::Decimal),
        "paren_alpha" => Some(EnumKind::ParenAlpha),
        "paren_decimal" => Some(EnumKind::ParenDecimal),
        "paren_roman" => Some(EnumKind::ParenRoman),
        "section" => Some(EnumKind::Section),
        "section_word" => Some(EnumKind::SectionWord),
        "chapter_word" => Some(EnumKind::ChapterWord),
        "subpart_word" => Some(EnumKind::SubpartWord),
        "bullet" => Some(EnumKind::Bullet),
        _ => None,
    }
}

// Suppress an unused-import lint when `parse_enumerator` isn't called
// directly by binding code (we call `parse_enumerator_with` everywhere).
#[allow(dead_code)]
fn _force_parse_enumerator_keep_alive(s: &str) -> Option<Enumerator> {
    parse_enumerator(s)
}

// ─── Module registration ────────────────────────────────────────────────────

pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "segmentation")?;

    m.add_class::<PyPunktParameters>()?;
    m.add_class::<PyPunktTokenizer>()?;
    m.add_class::<PyPunktTrainer>()?;
    m.add_class::<PyLineRecord>()?;
    m.add_class::<PyNormalizedText>()?;
    m.add_class::<PyBoilerplateRun>()?;
    m.add_class::<PyEnumerator>()?;
    m.add_function(wrap_pyfunction!(py_segment_lines, &m)?)?;
    m.add_function(wrap_pyfunction!(py_segment_sentences, &m)?)?;
    m.add_function(wrap_pyfunction!(py_segment_paragraphs, &m)?)?;
    m.add_function(wrap_pyfunction!(py_segment_paragraphs_simple, &m)?)?;
    m.add_function(wrap_pyfunction!(py_extract_line_records, &m)?)?;
    m.add_function(wrap_pyfunction!(py_normalize, &m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_boilerplate, &m)?)?;
    m.add_function(wrap_pyfunction!(py_parse_enumerator, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.segmentation", &m)?;

    Ok(())
}
