//! PyO3 bindings for the document-structure layer (P7).
//!
//! Three primitives:
//!
//! * [`PyHeadingFeatureVector`] + `py_score_heading_features` — per-line
//!   heading feature extraction (P7.1).
//! * [`PyLineLabel`] + `py_decode_line_labels` — Viterbi sequence decoder
//!   (P7.4).
//! * [`PyHeadingCandidate`] + `py_infer_hierarchy` — heading-stack
//!   inferencer (P7.6).
//!
//! All three are also wrapped by a one-shot convenience entry point
//! `py_label_lines`, which runs the full pipeline (extract → score →
//! decode → infer hierarchy) and returns the labels + candidates.
//!
//! The Rust core works in byte offsets internally; this binding is
//! offset-free (it consumes records / labels by index, not source
//! positions), so the byte→char conversion that segmentation.rs needs
//! does not apply here.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::bindings::util::{bincode_getstate, bincode_setstate};
use crate::core::segmentation::{
    detect_boilerplate_with, extract_line_records, parse_enumerator_with, BoilerplateOptions,
    EnumKind, Enumerator, WordLexicon,
};
use crate::core::structure::{
    citation_density, decode_line_labels, infer_hierarchy, score_heading_features,
    CustomHeadingLexicon, CustomHierarchyLexicon, DecoderOptions, EmissionCosts, HeadingCandidate,
    HeadingFeatureVector, HeadingLexicon, HierarchyLexicon, HierarchyOptions, LineLabel,
    ScoringOptions, ScoringWeights, TransitionMatrix,
};

// ─── HeadingFeatureVector ─────────────────────────────────────────────────

/// Per-line heading-feature vector. Field semantics match the Rust
/// [`HeadingFeatureVector`] one-to-one.
#[pyclass(
    name = "HeadingFeatureVector",
    module = "kaos_nlp_core._rust.structure",
    skip_from_py_object
)]
#[derive(Clone, Copy)]
struct PyHeadingFeatureVector {
    inner: HeadingFeatureVector,
}

#[pymethods]
impl PyHeadingFeatureVector {
    #[getter]
    fn short_line(&self) -> u8 {
        self.inner.short_line
    }
    #[getter]
    fn blank_before(&self) -> u8 {
        self.inner.blank_before
    }
    #[getter]
    fn blank_after(&self) -> u8 {
        self.inner.blank_after
    }
    #[getter]
    fn indent_le_4(&self) -> u8 {
        self.inner.indent_le_4
    }
    #[getter]
    fn case_allcaps(&self) -> u8 {
        self.inner.case_allcaps
    }
    #[getter]
    fn case_titlecase(&self) -> u8 {
        self.inner.case_titlecase
    }
    #[getter]
    fn case_initcap(&self) -> u8 {
        self.inner.case_initcap
    }
    #[getter]
    fn no_terminal_period(&self) -> u8 {
        self.inner.no_terminal_period
    }
    #[getter]
    fn colon_suffix(&self) -> u8 {
        self.inner.colon_suffix
    }
    #[getter]
    fn inline_colon(&self) -> u8 {
        self.inner.inline_colon
    }
    #[getter]
    fn has_enumerator(&self) -> u8 {
        self.inner.has_enumerator
    }
    #[getter]
    fn hierarchy_keyword(&self) -> u8 {
        self.inner.hierarchy_keyword
    }
    #[getter]
    fn lexical_heading(&self) -> u8 {
        self.inner.lexical_heading
    }
    #[getter]
    fn table_row_shape(&self) -> u8 {
        self.inner.table_row_shape
    }
    #[getter]
    fn column_gap_only(&self) -> u8 {
        self.inner.column_gap_only
    }
    #[getter]
    fn definition_shape(&self) -> u8 {
        self.inner.definition_shape
    }
    #[getter]
    fn form_field_shape(&self) -> u8 {
        self.inner.form_field_shape
    }
    #[getter]
    fn citation_density(&self) -> f32 {
        self.inner.citation_density
    }
    #[getter]
    fn boilerplate(&self) -> u8 {
        self.inner.boilerplate
    }
    #[getter]
    fn long_prose(&self) -> u8 {
        self.inner.long_prose
    }
    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }
    #[getter]
    fn is_blank(&self) -> u8 {
        self.inner.is_blank
    }
    #[getter]
    fn hierarchy_depth(&self) -> u8 {
        self.inner.hierarchy_depth
    }
    #[getter]
    fn atx_depth(&self) -> u8 {
        self.inner.atx_depth
    }

    fn heading_emit_cost(&self) -> f32 {
        self.inner.heading_emit_cost()
    }

    fn __repr__(&self) -> String {
        format!(
            "HeadingFeatureVector(score={:+.3}, short_line={}, allcaps={}, has_enumerator={}, hierarchy_keyword={}, table_row={}, boilerplate={})",
            self.inner.score,
            self.inner.short_line,
            self.inner.case_allcaps,
            self.inner.has_enumerator,
            self.inner.hierarchy_keyword,
            self.inner.table_row_shape,
            self.inner.boilerplate,
        )
    }
}

// ─── HeadingCandidate ─────────────────────────────────────────────────────

#[pyclass(
    name = "HeadingCandidate",
    module = "kaos_nlp_core._rust.structure",
    skip_from_py_object
)]
#[derive(Clone, Copy)]
struct PyHeadingCandidate {
    inner: HeadingCandidate,
}

#[pymethods]
impl PyHeadingCandidate {
    #[getter]
    fn line_index(&self) -> u32 {
        self.inner.line_index
    }
    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }
    /// Depth from the configured hierarchy lexicon. `0` means no
    /// keyword fired. Range 1..=255 with 1 = shallowest.
    #[getter]
    fn hierarchy_level(&self) -> u8 {
        self.inner.hierarchy_level
    }
    /// Depth derived from the line's enumerator shape. `0` means no
    /// enumerator.
    #[getter]
    fn numeric_depth(&self) -> u8 {
        self.inner.numeric_depth
    }
    /// The enumerator kind that fired, or `None` when no enumerator
    /// matched. Value: stable name string `"roman_upper"`, `"decimal"`, …
    #[getter]
    fn enumerator_kind(&self) -> Option<&'static str> {
        const NO_ENUM: u8 = 255;
        if self.inner.enumerator_kind == NO_ENUM {
            return None;
        }
        // Map back through the EnumKind enum.
        let kind = match self.inner.enumerator_kind {
            0 => EnumKind::RomanUpper,
            1 => EnumKind::RomanLower,
            2 => EnumKind::AlphaUpper,
            3 => EnumKind::AlphaLower,
            4 => EnumKind::Decimal,
            5 => EnumKind::ParenAlpha,
            6 => EnumKind::ParenDecimal,
            7 => EnumKind::ParenRoman,
            8 => EnumKind::Section,
            9 => EnumKind::SectionWord,
            10 => EnumKind::ChapterWord,
            11 => EnumKind::SubpartWord,
            12 => EnumKind::Bullet,
            _ => return None,
        };
        Some(match kind {
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
        })
    }
    /// Markdown ATX heading depth (1..=6) iff the line begins with that
    /// many `#` characters; 0 otherwise.
    #[getter]
    fn atx_depth(&self) -> u8 {
        self.inner.atx_depth
    }

    /// Default depth-pick rule per Q6 of the design reference: prefer
    /// ATX depth if present, then hierarchy keyword, then numeric depth
    /// shifted by 6 (so keyword headings sit shallower than numeric).
    /// Returns 0 if no signal fired.
    fn picked_depth(&self) -> u8 {
        self.inner.picked_depth()
    }

    fn __repr__(&self) -> String {
        format!(
            "HeadingCandidate(line={}, score={:+.3}, hier={}, numeric={}, atx={}, kind={:?})",
            self.inner.line_index,
            self.inner.score,
            self.inner.hierarchy_level,
            self.inner.numeric_depth,
            self.inner.atx_depth,
            self.enumerator_kind(),
        )
    }
}

// ─── Lexicon resolution helpers ───────────────────────────────────────────

fn resolve_heading_lexicon(
    name: Option<&str>,
    custom: Option<Vec<String>>,
) -> PyResult<HeadingLexicon> {
    if let Some(entries) = custom {
        let lex =
            CustomHeadingLexicon::new(entries).map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(HeadingLexicon::Custom(Arc::new(lex)));
    }
    Ok(match name {
        None | Some("english_legal_us") => HeadingLexicon::EnglishLegalUs,
        Some("english_academic") => HeadingLexicon::EnglishAcademic,
        Some("english_software") => HeadingLexicon::EnglishSoftware,
        Some("french_legal") => HeadingLexicon::FrenchLegal,
        Some("german_legal") => HeadingLexicon::GermanLegal,
        Some("spanish_legal") => HeadingLexicon::SpanishLegal,
        Some("italian_legal") => HeadingLexicon::ItalianLegal,
        Some("portuguese_legal") => HeadingLexicon::PortugueseLegal,
        Some("none") => HeadingLexicon::None,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown heading_lexicon {other:?}; valid: english_legal_us, \
                 english_academic, english_software, french_legal, german_legal, \
                 spanish_legal, italian_legal, portuguese_legal, none"
            )));
        }
    })
}

fn resolve_hierarchy_lexicon(
    name: Option<&str>,
    custom: Option<Vec<(String, u8)>>,
) -> PyResult<HierarchyLexicon> {
    if let Some(entries) = custom {
        let lex = CustomHierarchyLexicon::new(entries)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(HierarchyLexicon::Custom(Arc::new(lex)));
    }
    Ok(match name {
        None | Some("english_legal_us") => HierarchyLexicon::EnglishLegalUs,
        Some("french_legal") => HierarchyLexicon::FrenchLegal,
        Some("german_legal") => HierarchyLexicon::GermanLegal,
        Some("spanish_legal") => HierarchyLexicon::SpanishLegal,
        Some("italian_legal") => HierarchyLexicon::ItalianLegal,
        Some("portuguese_legal") => HierarchyLexicon::PortugueseLegal,
        Some("markdown_atx") => HierarchyLexicon::MarkdownAtx,
        Some("none") => HierarchyLexicon::None,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown hierarchy_lexicon {other:?}; valid: english_legal_us, \
                 french_legal, german_legal, spanish_legal, italian_legal, \
                 portuguese_legal, markdown_atx, none"
            )));
        }
    })
}

/// Build [`ScoringOptions`] from a user-supplied dict. Unknown keys are
/// rejected (so misspellings don't silently drop weight overrides).
fn build_scoring_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<ScoringOptions> {
    let mut opts = ScoringOptions::default();
    let kw = match kwargs {
        Some(k) => k,
        None => return Ok(opts),
    };

    // Lexicons.
    let heading_name: Option<String> = kw
        .get_item("heading_lexicon")?
        .and_then(|v| v.extract().ok());
    let custom_heading: Option<Vec<String>> = kw
        .get_item("custom_heading_lexicon")?
        .and_then(|v| v.extract().ok());
    opts.heading_lexicon = resolve_heading_lexicon(heading_name.as_deref(), custom_heading)?;

    let hier_name: Option<String> = kw
        .get_item("hierarchy_lexicon")?
        .and_then(|v| v.extract().ok());
    let custom_hier: Option<Vec<(String, u8)>> = kw
        .get_item("custom_hierarchy_lexicon")?
        .and_then(|v| v.extract().ok());
    opts.hierarchy_lexicon = resolve_hierarchy_lexicon(hier_name.as_deref(), custom_hier)?;

    // Scalar opts.
    if let Some(v) = kw.get_item("threshold")? {
        opts.threshold = v.extract::<f32>()?;
    }
    if let Some(v) = kw.get_item("short_line_chars")? {
        opts.short_line_chars = v.extract::<u32>()?;
    }
    if let Some(v) = kw.get_item("long_prose_chars")? {
        opts.long_prose_chars = v.extract::<u32>()?;
    }
    if let Some(v) = kw.get_item("max_heading_indent")? {
        opts.max_heading_indent = v.extract::<u16>()?;
    }
    if let Some(v) = kw.get_item("inline_colon_max_left_chars")? {
        opts.inline_colon_max_left_chars = v.extract::<usize>()?;
    }
    if let Some(v) = kw.get_item("definition_head_chars")? {
        opts.definition_head_chars = v.extract::<usize>()?;
    }
    if let Some(v) = kw.get_item("definition_verbs")? {
        // We need 'static-lifetime str slices in ScoringOptions, so leak
        // the strings the caller supplies. This is a deliberate trade-off
        // — verb lists are tiny (≤ ~20 entries) and configured once per
        // pipeline invocation, so the leak cost is negligible. If callers
        // start churning many distinct verb lists this can be revisited.
        let verbs: Vec<String> = v.extract()?;
        let static_verbs: Vec<&'static str> = verbs
            .into_iter()
            .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
            .collect();
        opts.definition_verbs = static_verbs;
    }
    if let Some(v) = kw.get_item("form_field_brackets")? {
        // Each tuple is (open, close) chars represented as 1-char strings.
        let pairs: Vec<(String, String)> = v.extract()?;
        let mut out = Vec::with_capacity(pairs.len());
        for (open, close) in pairs {
            let o = open.chars().next().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "form_field_brackets entries must be (open_char, close_char) — empty string given",
                )
            })?;
            let c = close.chars().next().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "form_field_brackets entries must be (open_char, close_char) — empty close",
                )
            })?;
            out.push((o, c));
        }
        opts.form_field_brackets = out;
    }

    // Per-feature weights — accept a nested dict.
    if let Some(w_obj) = kw.get_item("weights")? {
        let w_dict = w_obj.cast::<PyDict>()?;
        let mut w = ScoringWeights::default();
        macro_rules! grab_f32 {
            ($field:ident) => {
                if let Some(v) = w_dict.get_item(stringify!($field))? {
                    w.$field = v.extract::<f32>()?;
                }
            };
        }
        grab_f32!(short_line);
        grab_f32!(blank_before);
        grab_f32!(blank_after);
        grab_f32!(indent_le_4);
        grab_f32!(case_allcaps);
        grab_f32!(case_titlecase);
        grab_f32!(case_initcap);
        grab_f32!(no_terminal_period);
        grab_f32!(colon_suffix);
        grab_f32!(inline_colon);
        grab_f32!(has_enumerator);
        grab_f32!(hierarchy_keyword);
        grab_f32!(lexical_heading);
        grab_f32!(table_row_shape);
        grab_f32!(column_gap_only);
        grab_f32!(definition_shape);
        grab_f32!(form_field_shape);
        grab_f32!(citation_density);
        grab_f32!(boilerplate);
        grab_f32!(long_prose);
        opts.weights = w;
    }

    Ok(opts)
}

fn build_decoder_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<DecoderOptions> {
    let mut opts = DecoderOptions::default();
    let kw = match kwargs {
        Some(k) => k,
        None => return Ok(opts),
    };

    if let Some(v) = kw.get_item("force_blank_label")? {
        opts.force_blank_label = v.extract::<bool>()?;
    }
    if let Some(v) = kw.get_item("force_boilerplate_label")? {
        opts.force_boilerplate_label = v.extract::<bool>()?;
    }

    if let Some(em_obj) = kw.get_item("emissions")? {
        let em_dict = em_obj.cast::<PyDict>()?;
        let mut em = EmissionCosts::default();
        macro_rules! grab_f32 {
            ($field:ident) => {
                if let Some(v) = em_dict.get_item(stringify!($field))? {
                    em.$field = v.extract::<f32>()?;
                }
            };
        }
        grab_f32!(heading_emit_scale);
        grab_f32!(body_baseline);
        grab_f32!(table_row_strong);
        grab_f32!(table_row_weak);
        grab_f32!(list_item_strong);
        grab_f32!(list_item_with_enumerator);
        grab_f32!(list_item_weak);
        grab_f32!(metadata_strong);
        grab_f32!(metadata_weak);
        opts.emissions = em;
    }

    if let Some(t_obj) = kw.get_item("transitions")? {
        // 7×7 list of lists.
        let rows: Vec<Vec<f32>> = t_obj.extract()?;
        if rows.len() != LineLabel::COUNT || rows.iter().any(|r| r.len() != LineLabel::COUNT) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "transitions must be {n}x{n} (got {r}x?)",
                n = LineLabel::COUNT,
                r = rows.len(),
            )));
        }
        let mut tm = TransitionMatrix::v1_default();
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                tm.costs[i][j] = v;
            }
        }
        opts.transitions = tm;
    }

    if let Some(pd_obj) = kw.get_item("post_decode")? {
        let pd_dict = pd_obj.cast::<PyDict>()?;
        let mut pd = opts.post_decode;
        if let Some(v) = pd_dict.get_item("enable")? {
            pd.enable = v.extract::<bool>()?;
        }
        if let Some(v) = pd_dict.get_item("max_continuation_lines")? {
            pd.max_continuation_lines = v.extract::<u8>()?;
        }
        if let Some(v) = pd_dict.get_item("max_run_lines")? {
            pd.max_run_lines = v.extract::<u8>()?;
        }
        if let Some(v) = pd_dict.get_item("require_caps_in_run")? {
            pd.require_caps_in_run = v.extract::<bool>()?;
        }
        if let Some(v) = pd_dict.get_item("toc_pair_recognition")? {
            pd.toc_pair_recognition = v.extract::<bool>()?;
        }
        opts.post_decode = pd;
    }

    Ok(opts)
}

// ─── Public PyO3 entry points ─────────────────────────────────────────────

/// Score every line in `text` and return one `HeadingFeatureVector`
/// per line.
///
/// Internal: extract line records (P1) → parse leading enumerators
/// (P3, with `enum_lexicon`) → detect boilerplate (P5) → score (P7.1).
/// The full pipeline is hidden from Python.
#[pyfunction]
#[pyo3(signature = (text, *, enum_lexicon = None, custom_enum_lexicon = None, **scoring_kwargs))]
fn py_score_heading_features(
    py: Python<'_>,
    text: &str,
    enum_lexicon: Option<&str>,
    custom_enum_lexicon: Option<Vec<(String, String)>>,
    scoring_kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<PyHeadingFeatureVector>> {
    let opts = build_scoring_options(scoring_kwargs)?;
    let lex = resolve_word_lexicon(enum_lexicon, custom_enum_lexicon)?;
    let owned = text.to_string();
    py.detach(|| {
        let recs = extract_line_records(&owned);
        let enums: Vec<Option<Enumerator>> = recs
            .iter()
            .map(|r| {
                if r.blank {
                    None
                } else {
                    parse_enumerator_with(r.stripped_text(&owned), &lex)
                }
            })
            .collect();
        let runs = detect_boilerplate_with(&recs, &owned, BoilerplateOptions::default(), &lex);
        let features = score_heading_features(&owned, &recs, &enums, &runs, &opts);
        Ok(features
            .into_iter()
            .map(|inner| PyHeadingFeatureVector { inner })
            .collect())
    })
}

/// Decode the most-likely line-label sequence over `text`. Returns one
/// label string per line: `"blank"`, `"heading"`, `"body"`,
/// `"list_item"`, `"table_row"`, `"metadata"`, `"boilerplate"`.
#[pyfunction]
#[pyo3(signature = (
    text,
    *,
    enum_lexicon = None,
    custom_enum_lexicon = None,
    scoring = None,
    decoder = None,
))]
fn py_decode_line_labels(
    py: Python<'_>,
    text: &str,
    enum_lexicon: Option<&str>,
    custom_enum_lexicon: Option<Vec<(String, String)>>,
    scoring: Option<&Bound<'_, PyDict>>,
    decoder: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<&'static str>> {
    let s_opts = build_scoring_options(scoring)?;
    let d_opts = build_decoder_options(decoder)?;
    let lex = resolve_word_lexicon(enum_lexicon, custom_enum_lexicon)?;
    let owned = text.to_string();
    py.detach(|| {
        let recs = extract_line_records(&owned);
        let enums: Vec<Option<Enumerator>> = recs
            .iter()
            .map(|r| {
                if r.blank {
                    None
                } else {
                    parse_enumerator_with(r.stripped_text(&owned), &lex)
                }
            })
            .collect();
        let runs = detect_boilerplate_with(&recs, &owned, BoilerplateOptions::default(), &lex);
        let features = score_heading_features(&owned, &recs, &enums, &runs, &s_opts);
        let labels = decode_line_labels(&features, &d_opts);
        Ok(labels.into_iter().map(|l| l.name()).collect())
    })
}

/// Run the full pipeline: features + labels + heading candidates.
///
/// Returns a dict `{"labels": [...], "features": [...], "candidates":
/// [...]}` so the caller can correlate them by line index.
#[pyfunction]
#[pyo3(signature = (
    text,
    *,
    enum_lexicon = None,
    custom_enum_lexicon = None,
    scoring = None,
    decoder = None,
))]
fn py_label_lines(
    py: Python<'_>,
    text: &str,
    enum_lexicon: Option<&str>,
    custom_enum_lexicon: Option<Vec<(String, String)>>,
    scoring: Option<&Bound<'_, PyDict>>,
    decoder: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyDict>> {
    let s_opts = build_scoring_options(scoring)?;
    let d_opts = build_decoder_options(decoder)?;
    let lex = resolve_word_lexicon(enum_lexicon, custom_enum_lexicon)?;
    let owned = text.to_string();

    let (labels_str, features, candidates) = py.detach(|| {
        let recs = extract_line_records(&owned);
        let enums: Vec<Option<Enumerator>> = recs
            .iter()
            .map(|r| {
                if r.blank {
                    None
                } else {
                    parse_enumerator_with(r.stripped_text(&owned), &lex)
                }
            })
            .collect();
        let runs = detect_boilerplate_with(&recs, &owned, BoilerplateOptions::default(), &lex);
        let features = score_heading_features(&owned, &recs, &enums, &runs, &s_opts);
        let labels = decode_line_labels(&features, &d_opts);
        let h_opts = HierarchyOptions {
            lexicon: s_opts.hierarchy_lexicon.clone(),
        };
        let candidates = infer_hierarchy(&labels, &features, &enums, &h_opts);
        let labels_str: Vec<&'static str> = labels.into_iter().map(|l| l.name()).collect();
        (labels_str, features, candidates)
    });

    let out = PyDict::new(py);
    out.set_item("labels", labels_str)?;
    let f_list = PyList::empty(py);
    for f in features {
        f_list.append(PyHeadingFeatureVector { inner: f })?;
    }
    out.set_item("features", f_list)?;
    let c_list = PyList::empty(py);
    for c in candidates {
        c_list.append(PyHeadingCandidate { inner: c })?;
    }
    out.set_item("candidates", c_list)?;
    Ok(out.into())
}

/// Citation-density signal exposed as a stand-alone helper. Returns a
/// fraction in `[0, 1]` for an arbitrary `text`. Useful for callers that
/// want the metric without running the full pipeline.
#[pyfunction]
fn py_citation_density(text: &str) -> f32 {
    citation_density(text)
}

// ── Resolve a P3 word lexicon for enumerator parsing ──────────────────────

fn resolve_word_lexicon(
    name: Option<&str>,
    custom: Option<Vec<(String, String)>>,
) -> PyResult<WordLexicon> {
    use crate::core::segmentation::{CustomLexicon, EnumKind};

    if let Some(entries) = custom {
        let mut tagged: Vec<(String, EnumKind)> = Vec::with_capacity(entries.len());
        for (pat, kind_str) in entries {
            let kind = match kind_str.as_str() {
                "roman_upper" => EnumKind::RomanUpper,
                "roman_lower" => EnumKind::RomanLower,
                "alpha_upper" => EnumKind::AlphaUpper,
                "alpha_lower" => EnumKind::AlphaLower,
                "decimal" => EnumKind::Decimal,
                "paren_alpha" => EnumKind::ParenAlpha,
                "paren_decimal" => EnumKind::ParenDecimal,
                "paren_roman" => EnumKind::ParenRoman,
                "section" => EnumKind::Section,
                "section_word" => EnumKind::SectionWord,
                "chapter_word" => EnumKind::ChapterWord,
                "subpart_word" => EnumKind::SubpartWord,
                "bullet" => EnumKind::Bullet,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown enum kind {other:?}"
                    )))
                }
            };
            tagged.push((pat, kind));
        }
        let custom = CustomLexicon::new(tagged).map_err(pyo3::exceptions::PyValueError::new_err)?;
        return Ok(WordLexicon::Custom(Arc::new(custom)));
    }
    Ok(match name {
        None | Some("english_legal_us") => WordLexicon::EnglishLegalUs,
        Some("french_legal") => WordLexicon::FrenchLegal,
        Some("german_legal") => WordLexicon::GermanLegal,
        Some("spanish_legal") => WordLexicon::SpanishLegal,
        Some("italian_legal") => WordLexicon::ItalianLegal,
        Some("portuguese_legal") => WordLexicon::PortugueseLegal,
        Some("markdown_atx") => WordLexicon::MarkdownAtx,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown enum_lexicon {other:?}; valid: english_legal_us, \
                 french_legal, german_legal, spanish_legal, italian_legal, \
                 portuguese_legal, markdown_atx"
            )))
        }
    })
}

// ── Suppress unused-import warnings for symbols only used in tests ──
#[allow(dead_code)]
fn _keep_alive(_: &PyBytes) {
    // keep PyBytes import live for future pickle support
}
#[allow(dead_code)]
fn _keep_pickle_helpers_alive(py: Python<'_>) -> PyResult<Py<PyAny>> {
    bincode_getstate(py, &0u32)?;
    bincode_setstate::<u32>(&PyBytes::new(py, &[]))?;
    Ok(py.None())
}

// ─── Module registration ──────────────────────────────────────────────────

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "structure")?;

    m.add_class::<PyHeadingFeatureVector>()?;
    m.add_class::<PyHeadingCandidate>()?;
    m.add_function(wrap_pyfunction!(py_score_heading_features, &m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_line_labels, &m)?)?;
    m.add_function(wrap_pyfunction!(py_label_lines, &m)?)?;
    m.add_function(wrap_pyfunction!(py_citation_density, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.structure", &m)?;

    Ok(())
}
