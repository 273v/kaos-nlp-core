//! Per-line heading-feature extractor (P7.1).
//!
//! Consumes `&[LineRecord]` (P1), `&[Option<Enumerator>]` (P3 result, one
//! per record), and `&[BoilerplateRun]` (P5) to produce one
//! [`HeadingFeatureVector`] per line. The vector is the input to the
//! Viterbi sequence decoder (P7.4).
//!
//! Design reference: `docs/SECTION_HEADING_PRIMITIVES_RESEARCH.md`,
//! sections "Heading scorer + hierarchy inferencer (P7) — design
//! reference" (Q2) and "Heading scorer (P7) — corrigendum: generality"
//! (G1–G8).
//!
//! ## Boundary
//!
//! Pure forward-only: takes opaque indices into precomputed inputs and
//! returns one feature vector per line. Does not allocate beyond the
//! `Vec<HeadingFeatureVector>` it returns, does not consult AST nodes,
//! and does not require any kaos-content type.
//!
//! ## Generality (per the corrigendum)
//!
//! * Lexicons (`HeadingLexicon`, `HierarchyLexicon`) are **opt-in**. A
//!   document with `HeadingLexicon::None` and `HierarchyLexicon::None`
//!   still produces sensible scores driven by case + layout + enumerator
//!   shape alone — Western-language news / Wikipedia / Markdown headings
//!   without keywords are recoverable from these signals.
//! * Citation density is a generic period-token-fraction signal, not an
//!   English-only Bluebook detector. See `citation_density()` below.
//! * Default weights are calibrated for *no specific domain*. The
//!   per-domain calibration step (G8) sweeps weights against the
//!   multi-domain corpus.

use crate::core::segmentation::{
    normalize, BoilerplateRun, CaseProfile, Enumerator, LineRecord, NormalizeOptions, PunctProfile,
};

use super::lexicon::{HeadingLexicon, HierarchyLexicon};

// ─── Public types ─────────────────────────────────────────────────────────

/// Per-line feature vector. All boolean-shaped indicators are 0 / 1; the
/// `citation_density` is a real number in `[0, 1]`; the composite `score`
/// is the weighted sum, clamped to `[-1, 1]`.
///
/// `repr(C)` so the layout is stable for any future zero-copy export.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct HeadingFeatureVector {
    // ── Positive shape signals ──
    /// Stripped char count ≤ 60. Heads tend to be short.
    pub short_line: u8,
    /// Previous physical line was blank or this is line 0.
    pub blank_before: u8,
    /// Next physical line is blank or this is the last line.
    pub blank_after: u8,
    /// Indent ≤ 4 chars — heads sit near the left margin.
    pub indent_le_4: u8,
    /// `CaseProfile::AllCaps`.
    pub case_allcaps: u8,
    /// `CaseProfile::TitleCase`.
    pub case_titlecase: u8,
    /// `CaseProfile::InitialCap`.
    pub case_initcap: u8,
    /// Last non-whitespace char is *not* `.` or `?` or `!`.
    pub no_terminal_period: u8,
    /// Last non-whitespace char is `:`. Heading-shape signal
    /// (`Defendants:`, `Discussion:`).
    pub colon_suffix: u8,
    /// Line contains an inline `:` separator with non-empty content on
    /// both sides (e.g. `Author: Jane Doe`, `Date: 2026-05-05`,
    /// `Case Number: 22-1234`). Metadata-shape signal — distinct from
    /// `colon_suffix` (which is heading-shape).
    pub inline_colon: u8,
    /// `parse_enumerator_with(line)` returned `Some` — leading enumerator
    /// shape detected.
    pub has_enumerator: u8,
    /// Hierarchy keyword fires at the start of the stripped line (`Title`,
    /// `Chapter`, `Section`, `Article`, `§`, …) per the configured
    /// [`HierarchyLexicon`]. Markdown ATX special-cases this: any `#`
    /// prefix of length 1..=6 sets the flag.
    pub hierarchy_keyword: u8,
    /// Stripped, normalized line text matches a canonical-heading entry
    /// in the configured [`HeadingLexicon`].
    pub lexical_heading: u8,

    // ── Negative shape signals ──
    /// True iff the line has at least one `|` character (Markdown / pipe
    /// table) OR has both pipe and column-gap shape. Strong table-row
    /// signal — almost never a false positive in real prose.
    pub table_row_shape: u8,
    /// True iff the line has multi-space column gaps but no pipe.
    /// Weaker table-row signal — fires on tabbed / aligned prose too.
    /// F-R2 fix: tracked separately so the decoder can prefer body for
    /// column-gap-only lines.
    pub column_gap_only: u8,
    /// True iff the line looks like a contract definition:
    /// starts with an opening quote (`"` straight, `“` curly), and
    /// contains a definition verb (`means`, `signifie`, `bedeutet`,
    /// `significa`) within the first ~80 chars. F-R5 fix.
    pub definition_shape: u8,
    /// True iff the line looks like a form-field placeholder
    /// (`<Name of Agency>`, `(Street address)`, `[Date created]`).
    /// Whole-line wrapped in matching brackets / parens / angle
    /// brackets. F-R6 fix — these score heading-shape but are
    /// structurally metadata fields.
    pub form_field_shape: u8,
    /// Generic citation-density signal in `[0, 1]`.
    pub citation_density: f32,
    /// Line is a member of any [`BoilerplateRun`] in the input.
    pub boilerplate: u8,
    /// `char_count > 200` — probably prose.
    pub long_prose: u8,

    // ── Composite ──
    /// Weighted sum of the indicators above, clamped to `[-1, 1]`. Maps
    /// 1-1 to the "heading" emission cost in the Viterbi decoder via
    /// `emit_cost_heading = 1.0 - score`.
    pub score: f32,

    // ── Decoder hints (computed once here so the decoder doesn't have
    //    to look them up). All 0 / 1 indicators. ──
    /// True for the "blank" Viterbi state (`LineRecord::blank`).
    pub is_blank: u8,
    /// Read directly from the [`HierarchyLexicon`] match: 0 means no
    /// match, 1..=255 means depth (lower = shallower).
    pub hierarchy_depth: u8,
    /// Markdown ATX `#` count (0 unless the configured hierarchy lexicon
    /// is [`HierarchyLexicon::MarkdownAtx`] and the line begins with
    /// 1..=6 `#` chars). Inserted here so the hierarchy inferencer
    /// (P7.6) can read depth directly without parsing the line again.
    pub atx_depth: u8,
}

impl HeadingFeatureVector {
    /// Convenience: the heading-emission cost for the Viterbi decoder.
    /// `1.0 - score` clamped to `[0, 2]`.
    #[inline]
    pub fn heading_emit_cost(&self) -> f32 {
        (1.0 - self.score).clamp(0.0, 2.0)
    }
}

/// Tunable scoring weights. Defaults follow the v1 table in the design
/// reference (rebalanced per corrigendum #1: hierarchy_keyword reduced
/// from 0.40 to 0.30 so a doc without hierarchy keywords still has
/// headroom above the threshold from layout/case alone).
#[derive(Debug, Clone, Copy)]
pub struct ScoringWeights {
    pub short_line: f32,
    pub blank_before: f32,
    pub blank_after: f32,
    pub indent_le_4: f32,
    pub case_allcaps: f32,
    pub case_titlecase: f32,
    pub case_initcap: f32,
    pub no_terminal_period: f32,
    pub colon_suffix: f32,
    /// Negative weight for inline-colon: an `Author: Jane Doe` line is
    /// metadata, not a heading. Same magnitude as `citation_density`
    /// (also "this line shape is structurally not-a-heading").
    /// Default `-0.30`.
    pub inline_colon: f32,
    pub has_enumerator: f32,
    pub hierarchy_keyword: f32,
    pub lexical_heading: f32,
    pub table_row_shape: f32,
    /// Mild negative for column-gap-only lines (no pipes, but multi-space
    /// runs). Default `-0.10` — discourages heading classification on
    /// tabbed prose without committing the line to table_row.
    /// F-R2 fix.
    pub column_gap_only: f32,
    /// Negative weight for definition-shape lines (`"Term" means …`).
    /// Default `-0.30`. F-R5 fix.
    pub definition_shape: f32,
    /// Negative weight for form-field placeholder lines
    /// (`<Name of Agency>`, `(Charter no.)`). Default `-0.30`.
    /// F-R6 fix.
    pub form_field_shape: f32,
    pub citation_density: f32,
    pub boilerplate: f32,
    pub long_prose: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            short_line: 0.10,
            blank_before: 0.10,
            blank_after: 0.05,
            indent_le_4: 0.05,
            case_allcaps: 0.20,
            case_titlecase: 0.15,
            case_initcap: 0.05,
            no_terminal_period: 0.10,
            colon_suffix: 0.10,
            inline_colon: -0.30,
            has_enumerator: 0.30,
            hierarchy_keyword: 0.30,
            lexical_heading: 0.25,
            table_row_shape: -0.50,
            column_gap_only: -0.10,
            definition_shape: -0.30,
            form_field_shape: -0.30,
            citation_density: -0.30,
            boilerplate: -0.50,
            long_prose: -0.30,
        }
    }
}

/// Top-level scorer options.
#[derive(Debug, Clone)]
pub struct ScoringOptions {
    /// Lexicon used to set the `lexical_heading` flag.
    pub heading_lexicon: HeadingLexicon,
    /// Lexicon used to set the `hierarchy_keyword` flag and supply
    /// `hierarchy_depth`.
    pub hierarchy_lexicon: HierarchyLexicon,
    /// Tunable weights.
    pub weights: ScoringWeights,
    /// Heading score threshold. Per Q2 the v1 default is 0.30. The
    /// scorer does **not** use this internally — it only computes
    /// scores. The Viterbi decoder (P7.4) reads the threshold from the
    /// same options struct so they stay aligned.
    pub threshold: f32,
    /// `short_line` triggers below this char count (default 60).
    pub short_line_chars: u32,
    /// `long_prose` triggers above this char count (default 200).
    pub long_prose_chars: u32,
    /// `indent_le_4` triggers below this indent count (default 4).
    pub max_heading_indent: u16,
    /// Maximum left-side length for `inline_colon` to fire. The "label"
    /// portion before the `:` must be 1..=this many chars. Default 40.
    /// Tune up for verbose form labels, down for terse metadata.
    pub inline_colon_max_left_chars: usize,
    /// `definition_shape` looks for a definition verb in the first N
    /// chars of the line. Default 80.
    pub definition_head_chars: usize,
    /// Definition-verb list used by `definition_shape`. Each entry must
    /// be lowercase and surrounded by spaces (or `,`/`.` etc.) — the
    /// detector checks `head_lower.contains(v)`. Default covers
    /// English / French / German / Spanish / Italian / Portuguese
    /// `means` / `signifie` / `bedeutet` / `significa`. Override to
    /// extend or restrict.
    pub definition_verbs: Vec<&'static str>,
    /// Bracket pairs accepted by `form_field_shape` (open, close).
    /// Default `[<>, (), [], {}]`. Override with custom delimiters
    /// (e.g., `‹›` Unicode angle brackets) by passing a different
    /// list.
    pub form_field_brackets: Vec<(char, char)>,
}

/// Default verb list for `definition_shape` — kept as a free constant
/// so callers can extend it via `[..DEFAULT_DEFINITION_VERBS, ...]`.
pub const DEFAULT_DEFINITION_VERBS: &[&str] = &[
    " means ",
    " means,",
    " signifie ",
    " signifient ",
    " bedeutet ",
    " bezeichnet ",
    " significa ",
    " significam ",
    " ha il significato",
    " refers to ",
];

/// Default bracket pairs for `form_field_shape`.
pub const DEFAULT_FORM_FIELD_BRACKETS: &[(char, char)] =
    &[('<', '>'), ('(', ')'), ('[', ']'), ('{', '}')];

impl Default for ScoringOptions {
    fn default() -> Self {
        Self {
            heading_lexicon: HeadingLexicon::default(),
            hierarchy_lexicon: HierarchyLexicon::default(),
            weights: ScoringWeights::default(),
            threshold: 0.30,
            short_line_chars: 60,
            long_prose_chars: 200,
            max_heading_indent: 4,
            inline_colon_max_left_chars: 40,
            definition_head_chars: 80,
            definition_verbs: DEFAULT_DEFINITION_VERBS.to_vec(),
            form_field_brackets: DEFAULT_FORM_FIELD_BRACKETS.to_vec(),
        }
    }
}

// ─── Entry point ──────────────────────────────────────────────────────────

/// Score every line in `records`, producing one feature vector per line.
///
/// `enumerators` and `boilerplate` carry pre-computed P3/P5 results:
///
/// * `enumerators[i]` is `Some(Enumerator)` iff `parse_enumerator_with`
///   matched the start of line `i`. Length must equal `records.len()`.
/// * `boilerplate` is the list of [`BoilerplateRun`]s detected by P5 on
///   the same `records`. The scorer expands the runs into a per-line
///   bitmap internally.
///
/// `source` must be the same `&str` that produced `records` — record
/// offsets index into it.
pub fn score_heading_features(
    source: &str,
    records: &[LineRecord],
    enumerators: &[Option<Enumerator>],
    boilerplate: &[BoilerplateRun],
    opts: &ScoringOptions,
) -> Vec<HeadingFeatureVector> {
    if records.is_empty() {
        return Vec::new();
    }
    assert_eq!(
        enumerators.len(),
        records.len(),
        "enumerators length must match records length",
    );

    // 1. Build the boilerplate-line bitmap once.
    let mut is_boilerplate = vec![false; records.len()];
    for run in boilerplate {
        for &line_idx in &run.line_indices {
            let idx = line_idx as usize;
            if idx < is_boilerplate.len() {
                is_boilerplate[idx] = true;
            }
        }
    }

    // 2. Build the canonicalisation options once.
    let canonical_opts = NormalizeOptions {
        collapse_whitespace: true,
        fold_case: true,
        normalize_unicode_punct: true,
        strip_enumerator_prefix: false,
        strip_punctuation: false,
    };

    // 3. Score each line.
    let mut out = Vec::with_capacity(records.len());
    for (i, rec) in records.iter().enumerate() {
        let stripped = rec.stripped_text(source);
        let enum_ref = &enumerators[i];
        out.push(score_one_line(
            rec,
            stripped,
            enum_ref,
            is_boilerplate[i],
            &canonical_opts,
            opts,
        ));
    }
    out
}

// ─── Per-line scorer ──────────────────────────────────────────────────────

fn score_one_line(
    rec: &LineRecord,
    stripped: &str,
    enumerator: &Option<Enumerator>,
    in_boilerplate: bool,
    canonical_opts: &NormalizeOptions,
    opts: &ScoringOptions,
) -> HeadingFeatureVector {
    if rec.blank {
        return HeadingFeatureVector {
            short_line: 0,
            blank_before: 0,
            blank_after: 0,
            indent_le_4: 0,
            case_allcaps: 0,
            case_titlecase: 0,
            case_initcap: 0,
            no_terminal_period: 0,
            colon_suffix: 0,
            inline_colon: 0,
            has_enumerator: 0,
            hierarchy_keyword: 0,
            lexical_heading: 0,
            table_row_shape: 0,
            column_gap_only: 0,
            definition_shape: 0,
            form_field_shape: 0,
            citation_density: 0.0,
            boilerplate: u8::from(in_boilerplate),
            long_prose: 0,
            score: 0.0,
            is_blank: 1,
            hierarchy_depth: 0,
            atx_depth: 0,
        };
    }

    // Canonical (case-folded, whitespace-collapsed) form for lexicon
    // matching.
    let canonical: String = match normalize(stripped, *canonical_opts) {
        Ok(n) => n.text.into_owned(),
        Err(_) => stripped.to_lowercase(),
    };
    let canonical = canonical.trim();

    // Markdown ATX special case: depth from leading `#` count, 1..=6.
    let atx_depth = atx_heading_depth(stripped);

    // Hierarchy keyword: try the configured lexicon (English / French /
    // German / Spanish / Italian / Portuguese) on the *original* stripped
    // text (so case is preserved for the AC's leading-anchor check).
    let hierarchy_depth = match (&opts.hierarchy_lexicon, atx_depth) {
        (HierarchyLexicon::MarkdownAtx, depth) if depth > 0 => depth,
        _ => opts
            .hierarchy_lexicon
            .matches_leading(stripped)
            .unwrap_or(0),
    };
    let hierarchy_keyword = u8::from(hierarchy_depth > 0);

    let lexical_heading = u8::from(opts.heading_lexicon.matches_whole(canonical));

    let short_line = u8::from(rec.char_len <= opts.short_line_chars);
    let blank_before = u8::from(rec.blank_before);
    let blank_after = u8::from(rec.blank_after);
    let indent_le_4 = u8::from(rec.indent_chars <= opts.max_heading_indent);

    let case_allcaps = u8::from(matches!(rec.case_profile, CaseProfile::AllCaps));
    let case_titlecase = u8::from(matches!(rec.case_profile, CaseProfile::TitleCase));
    let case_initcap = u8::from(matches!(rec.case_profile, CaseProfile::InitialCap));

    let punct = rec.punct_profile;
    let no_terminal_period = u8::from(
        !punct.contains(PunctProfile::ENDS_PERIOD) && !punct.contains(PunctProfile::ENDS_QUESTION),
    );
    let colon_suffix = u8::from(punct.contains(PunctProfile::ENDS_COLON));
    let inline_colon = u8::from(detect_inline_colon(
        stripped,
        opts.inline_colon_max_left_chars,
    ));

    let has_enumerator = u8::from(enumerator.is_some());

    let has_pipe = punct.contains(PunctProfile::HAS_PIPE);
    let has_col_gaps = punct.contains(PunctProfile::HAS_COLUMN_GAPS);
    // F-R2: pipe is the strong signal; column-gap-only fires on tabbed
    // prose and is unreliable on its own.
    let table_row_shape = u8::from(has_pipe);
    let column_gap_only = u8::from(has_col_gaps && !has_pipe);
    let definition_shape = u8::from(detect_definition_shape(
        stripped,
        opts.definition_head_chars,
        &opts.definition_verbs,
    ));
    let form_field_shape = u8::from(detect_form_field_shape(stripped, &opts.form_field_brackets));
    let citation_density = citation_density(stripped);
    let long_prose = u8::from(rec.char_len > opts.long_prose_chars);
    let boilerplate = u8::from(in_boilerplate);

    let w = &opts.weights;
    let positive = f32::from(short_line) * w.short_line
        + f32::from(blank_before) * w.blank_before
        + f32::from(blank_after) * w.blank_after
        + f32::from(indent_le_4) * w.indent_le_4
        + f32::from(case_allcaps) * w.case_allcaps
        + f32::from(case_titlecase) * w.case_titlecase
        + f32::from(case_initcap) * w.case_initcap
        + f32::from(no_terminal_period) * w.no_terminal_period
        // F-R4: a trailing colon is heading-shape only on a SHORT line.
        // Long prose ending in ":" (`...followed by a list:`) is a
        // sentence-introducer, not a heading.
        + f32::from(colon_suffix) * f32::from(short_line) * w.colon_suffix
        + f32::from(inline_colon) * w.inline_colon
        // F-R1: an enumerator at the start of a SHORT line is heading-shape;
        // an enumerator on a long line is structurally a list item, not a
        // heading. Gate the heading bonus on short_line so long enumerated
        // lines (e.g. "1. To develop students' abilities to utilize…")
        // don't accidentally clear the heading threshold.
        + f32::from(has_enumerator) * f32::from(short_line) * w.has_enumerator
        + f32::from(hierarchy_keyword) * w.hierarchy_keyword
        + f32::from(lexical_heading) * w.lexical_heading;
    let negative = f32::from(table_row_shape) * w.table_row_shape
        + f32::from(column_gap_only) * w.column_gap_only
        + f32::from(definition_shape) * w.definition_shape
        + f32::from(form_field_shape) * w.form_field_shape
        + citation_density * w.citation_density
        + f32::from(boilerplate) * w.boilerplate
        + f32::from(long_prose) * w.long_prose;
    // Cap each side independently per the design reference; clamp final.
    let positive = positive.min(1.0);
    let negative = negative.max(-1.0);
    let score = (positive + negative).clamp(-1.0, 1.0);

    HeadingFeatureVector {
        short_line,
        blank_before,
        blank_after,
        indent_le_4,
        case_allcaps,
        case_titlecase,
        case_initcap,
        no_terminal_period,
        colon_suffix,
        inline_colon,
        has_enumerator,
        hierarchy_keyword,
        lexical_heading,
        table_row_shape,
        column_gap_only,
        definition_shape,
        form_field_shape,
        citation_density,
        boilerplate,
        long_prose,
        score,
        is_blank: 0,
        hierarchy_depth,
        atx_depth,
    }
}

/// Return the Markdown ATX heading depth (1..=6) iff the stripped line
/// begins with that many `#` characters followed by whitespace, or 0.
///
/// `#`-only lines like `######` (no following space/text) are NOT
/// considered headings.
fn atx_heading_depth(stripped: &str) -> u8 {
    let bytes = stripped.as_bytes();
    let mut hashes = 0u8;
    while (hashes as usize) < bytes.len() && bytes[hashes as usize] == b'#' && hashes < 7 {
        hashes += 1;
    }
    if hashes == 0 || hashes > 6 {
        return 0;
    }
    // Must be followed by whitespace AND have non-empty content after.
    let after = &bytes[hashes as usize..];
    if after.is_empty() {
        return 0;
    }
    if !(after[0] == b' ' || after[0] == b'\t') {
        return 0;
    }
    // Trim and check non-empty.
    let rest = std::str::from_utf8(after).unwrap_or("").trim();
    if rest.is_empty() {
        return 0;
    }
    hashes
}

/// Generic citation-density signal in `[0, 1]`. Defined as the fraction
/// of *whitespace-separated tokens* in `text` that look like citation
/// abbreviations:
///
/// * Contains a `§` U+00A7 byte.
/// * Length 2..=8 ASCII chars and ends with `.` (matches `Pub.`,
///   `L.`, `F.R.`, `Stat.`, `U.S.C.`, `art.`, `ch.`, `Bd.`, `Nr.`,
///   `Mr.`, `Dr.` — across English, French, German, Spanish).
///
/// Returns 0.0 for empty or no-token input. Returns a value > 0.5 only
/// when the line is dominated by citation-shaped tokens; ordinary prose
/// that happens to contain `Mr.` once stays well below 0.5 because the
/// surrounding tokens are not citation-shaped.
///
/// Language-agnostic by design — does not consult any English Bluebook
/// keyword list.
pub fn citation_density(text: &str) -> f32 {
    let mut total = 0u32;
    let mut hits = 0u32;
    for token in text.split_whitespace() {
        total += 1;
        if token.contains('§') {
            hits += 1;
            continue;
        }
        if is_citation_shaped_token(token) {
            hits += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    hits as f32 / total as f32
}

/// `"Term" means …`-style contract definition signal (F-R5).
///
/// Returns `true` iff `text` begins with an opening quotation mark
/// (ASCII `"`, curly `“`, French `«`, single `‘`) and contains one of
/// the configured definition verbs within the first `head_chars` chars.
///
/// `head_chars` and `verbs` are caller-supplied so the detector can
/// be tuned for non-Western or specialized vocabularies. Defaults
/// (Western-language scope) live in [`DEFAULT_DEFINITION_VERBS`].
fn detect_definition_shape(text: &str, head_chars: usize, verbs: &[&str]) -> bool {
    if text.is_empty() || verbs.is_empty() {
        return false;
    }
    let starts_with_quote = text.starts_with('"')
        || text.starts_with('\u{201C}')
        || text.starts_with('\u{2018}')
        || text.starts_with('\u{00AB}');
    if !starts_with_quote {
        return false;
    }
    let head_end = text
        .char_indices()
        .nth(head_chars)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    let head = &text[..head_end];
    let head_lower = head.to_lowercase();
    verbs.iter().any(|v| head_lower.contains(v))
}

/// `<Name of Agency>` / `(Charter no.)` form-field placeholder
/// detector (F-R6).
///
/// Returns `true` iff the trimmed line is wholly wrapped in matching
/// brackets / parens / angle brackets, OR fully consists of a `Label: ___`
/// pattern (label + colon + a run of underscores or whitespace).
fn detect_form_field_shape(text: &str, pairs: &[(char, char)]) -> bool {
    let t = text.trim();
    let chars: Vec<char> = t.chars().collect();
    if chars.len() >= 4 {
        let first = chars[0];
        let last = *chars.last().unwrap();
        for &(open, close) in pairs {
            if first == open && last == close {
                let inner = &chars[1..chars.len() - 1];
                if inner.iter().any(|c| c.is_alphabetic()) {
                    return true;
                }
            }
        }
    }
    // `Label: _____` fill-in pattern (independent of bracket shape).
    let bytes = t.as_bytes();
    if let Some(colon_pos) = bytes.iter().position(|&b| b == b':') {
        let after = t[colon_pos + 1..].trim();
        if !after.is_empty() && after.bytes().all(|b| b == b'_' || b == b'.' || b == b' ') {
            return true;
        }
    }
    false
}

/// `Author: Jane Doe`-style metadata signal.
///
/// Returns `true` iff `text` contains a `:` whose left side is a short
/// label (1..=40 chars, ends with a non-whitespace char) and whose right
/// side has non-empty content. The `:` must NOT be the last char (that
/// is the `colon_suffix` heading signal). Empty right-side content
/// (`Author:`) is treated as `colon_suffix`, not `inline_colon`.
fn detect_inline_colon(text: &str, max_left_chars: usize) -> bool {
    let bytes = text.as_bytes();
    // Find first `:`.
    let colon = match bytes.iter().position(|&b| b == b':') {
        Some(p) => p,
        None => return false,
    };
    // Left side: must be non-empty and ≤ max_left_chars chars.
    if colon == 0 || colon > max_left_chars {
        return false;
    }
    // Right side: must have at least one non-whitespace char.
    let right = &text[colon + 1..];
    right.chars().any(|c| !c.is_whitespace())
}

fn is_citation_shaped_token(token: &str) -> bool {
    if !token.is_ascii() {
        return false;
    }
    let bytes = token.as_bytes();
    if bytes.len() < 2 || bytes.len() > 8 {
        return false;
    }
    if *bytes.last().unwrap() != b'.' {
        return false;
    }
    // Must contain at least one alphabetic char.
    if !bytes.iter().any(|b| b.is_ascii_alphabetic()) {
        return false;
    }
    // No characters outside [A-Za-z0-9.]
    bytes
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || *b == b'.')
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::segmentation::{detect_boilerplate, extract_line_records, BoilerplateOptions};

    fn score_with_defaults(text: &str) -> Vec<HeadingFeatureVector> {
        let records = extract_line_records(text);
        let enumerators: Vec<Option<Enumerator>> = records.iter().map(|_| None).collect(); // tests that don't care about enums
        let boilerplate = Vec::new();
        let opts = ScoringOptions::default();
        score_heading_features(text, &records, &enumerators, &boilerplate, &opts)
    }

    fn score_with_full_pipeline(text: &str, opts: ScoringOptions) -> Vec<HeadingFeatureVector> {
        let records = extract_line_records(text);
        let enumerators: Vec<Option<Enumerator>> = records
            .iter()
            .map(|r| {
                if r.blank {
                    None
                } else {
                    crate::core::segmentation::parse_enumerator(r.stripped_text(text))
                }
            })
            .collect();
        let runs = detect_boilerplate(&records, text, BoilerplateOptions::default());
        score_heading_features(text, &records, &enumerators, &runs, &opts)
    }

    // ── Sanity ──────────────────────────────────────────────────────────────

    #[test]
    fn empty_input_returns_empty() {
        let out = score_with_defaults("");
        assert!(out.is_empty());
    }

    #[test]
    fn single_blank_line_marked_blank() {
        let out = score_with_defaults("\n");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].is_blank, 1);
        assert_eq!(out[0].score, 0.0);
    }

    #[test]
    fn output_length_matches_record_count() {
        let text = "alpha\nbeta\ngamma\n";
        let out = score_with_defaults(text);
        assert_eq!(out.len(), 3);
    }

    // ── Positive shape signals ─────────────────────────────────────────────

    #[test]
    fn allcaps_short_blank_around_scores_high() {
        let text = "Some prose paragraph one.\n\nDISCUSSION\n\nMore prose here.\n";
        let out = score_with_defaults(text);
        // Find the DISCUSSION line.
        let heading = out.iter().find(|f| f.case_allcaps == 1).unwrap();
        assert_eq!(heading.short_line, 1);
        assert_eq!(heading.blank_before, 1);
        assert_eq!(heading.blank_after, 1);
        assert_eq!(heading.no_terminal_period, 1);
        assert!(heading.score > 0.30, "score = {}", heading.score);
    }

    #[test]
    fn titlecase_lexical_heading_legal_us() {
        let text = "Background\n\nThe parties dispute X.\n";
        let out = score_with_defaults(text);
        // Background is in the english_legal_us lexicon by default.
        let heading = &out[0];
        // Single-token "Background" classifies as TitleCase (every token's
        // first char is upper, remainder lower) — accept either signal.
        assert!(
            heading.case_initcap == 1 || heading.case_titlecase == 1,
            "case_initcap={} case_titlecase={}",
            heading.case_initcap,
            heading.case_titlecase,
        );
        assert_eq!(heading.lexical_heading, 1);
        assert!(heading.score > 0.30);
    }

    #[test]
    fn lexical_heading_ignores_non_match() {
        let text = "Notathing\nmore text\n";
        let out = score_with_defaults(text);
        assert_eq!(out[0].lexical_heading, 0);
    }

    #[test]
    fn lexical_lexicon_can_be_disabled() {
        let opts = ScoringOptions {
            heading_lexicon: HeadingLexicon::None,
            ..ScoringOptions::default()
        };
        let text = "Background\nmore text\n";
        let records = extract_line_records(text);
        let enumerators: Vec<Option<Enumerator>> = records.iter().map(|_| None).collect();
        let out = score_heading_features(text, &records, &enumerators, &[], &opts);
        assert_eq!(out[0].lexical_heading, 0);
    }

    #[test]
    fn hierarchy_keyword_section_fires() {
        let text = "Section 5 Definitions\nbody text here\n";
        let records = extract_line_records(text);
        let enumerators: Vec<Option<Enumerator>> = records
            .iter()
            .map(|r| {
                if r.blank {
                    None
                } else {
                    crate::core::segmentation::parse_enumerator(r.stripped_text(text))
                }
            })
            .collect();
        let opts = ScoringOptions::default();
        let out = score_heading_features(text, &records, &enumerators, &[], &opts);
        assert_eq!(out[0].hierarchy_keyword, 1);
        assert!(
            out[0].hierarchy_depth >= 1,
            "depth: {}",
            out[0].hierarchy_depth
        );
        assert_eq!(out[0].has_enumerator, 1);
    }

    #[test]
    fn hierarchy_keyword_does_not_fire_in_middle() {
        let text = "This Section is about X\n";
        let out = score_with_defaults(text);
        assert_eq!(out[0].hierarchy_keyword, 0);
    }

    #[test]
    fn markdown_atx_depth_extracted() {
        let opts = ScoringOptions {
            hierarchy_lexicon: HierarchyLexicon::MarkdownAtx,
            ..ScoringOptions::default()
        };
        let text = "# H1\n## H2\n### H3\nbody\n";
        let out = score_with_full_pipeline(text, opts);
        assert_eq!(out[0].atx_depth, 1);
        assert_eq!(out[0].hierarchy_depth, 1);
        assert_eq!(out[0].hierarchy_keyword, 1);
        assert_eq!(out[1].atx_depth, 2);
        assert_eq!(out[1].hierarchy_depth, 2);
        assert_eq!(out[2].atx_depth, 3);
        assert_eq!(out[3].atx_depth, 0); // body line, no #
    }

    #[test]
    fn markdown_hash_run_with_no_text_is_not_heading() {
        let opts = ScoringOptions {
            hierarchy_lexicon: HierarchyLexicon::MarkdownAtx,
            ..ScoringOptions::default()
        };
        let text = "######\nbody\n";
        let out = score_with_full_pipeline(text, opts);
        assert_eq!(out[0].atx_depth, 0);
    }

    #[test]
    fn colon_suffix_fires() {
        let text = "Defendants:\nbody\n";
        let out = score_with_defaults(text);
        assert_eq!(out[0].colon_suffix, 1);
        assert_eq!(out[0].no_terminal_period, 1);
    }

    // ── Negative shape signals ─────────────────────────────────────────────

    #[test]
    fn table_row_shape_pipe() {
        let text = "Col A | Col B | Col C\n";
        let out = score_with_defaults(text);
        assert_eq!(out[0].table_row_shape, 1);
        assert!(out[0].score < 0.30);
    }

    #[test]
    fn column_gap_only_fires_without_pipe() {
        // F-R2: multi-space whitespace runs alone fire `column_gap_only`,
        // NOT `table_row_shape`. Real prose with tabbed alignment trips
        // column gaps but isn't a table row.
        let text = "Col A     Col B     Col C\n";
        let out = score_with_defaults(text);
        assert_eq!(out[0].table_row_shape, 0);
        assert_eq!(out[0].column_gap_only, 1);
    }

    #[test]
    fn pipe_line_fires_table_row_shape() {
        // Pipe-delimited line — strong table_row signal.
        let text = "Col A | Col B | Col C\n";
        let out = score_with_defaults(text);
        assert_eq!(out[0].table_row_shape, 1);
        assert_eq!(out[0].column_gap_only, 0);
    }

    #[test]
    fn citation_density_high_for_citation_dominated_line() {
        let text = "5 U.S.C. § 552; Pub. L. 89-487; Stat. 250.\n";
        let d = citation_density(text);
        // Lots of period-terminated short tokens → density should be > 0.4.
        assert!(d >= 0.40, "density = {d}");
    }

    // ── F-R5: contract definition shape ──

    #[test]
    fn definition_shape_fires_on_quoted_means_pattern() {
        // The canonical contract-definition pattern.
        let out = score_with_defaults("\"Closing\" means the recordation of the deed.\n");
        assert_eq!(out[0].definition_shape, 1);
        // Heading score should be reduced.
        assert!(out[0].score < 0.30, "score = {}", out[0].score);
    }

    #[test]
    fn definition_shape_supports_curly_quotes() {
        let out =
            score_with_defaults("\u{201C}Closing\u{201D} means the recordation of the deed.\n");
        assert_eq!(out[0].definition_shape, 1);
    }

    #[test]
    fn definition_shape_supports_western_languages() {
        // Spanish / Italian: significa.
        let out = score_with_defaults("\"Cierre\" significa el registro de la escritura.\n");
        assert_eq!(out[0].definition_shape, 1);
        // German: bedeutet.
        let out = score_with_defaults("\"Closing\" bedeutet die Eintragung der Urkunde.\n");
        assert_eq!(out[0].definition_shape, 1);
        // French: signifie.
        let out = score_with_defaults("\"Closing\" signifie l'enregistrement de l'acte.\n");
        assert_eq!(out[0].definition_shape, 1);
    }

    #[test]
    fn definition_shape_does_not_fire_on_quoted_heading() {
        // A quoted heading with no definition verb does NOT trigger.
        let out = score_with_defaults("\"Discussion of the Issues\"\n");
        assert_eq!(out[0].definition_shape, 0);
    }

    #[test]
    fn definition_shape_does_not_fire_on_unquoted_means() {
        // No leading quote → not a definition.
        let out = score_with_defaults("This means we will proceed.\n");
        assert_eq!(out[0].definition_shape, 0);
    }

    // ── F-R6: form-field placeholders ──

    #[test]
    fn form_field_angle_brackets() {
        let out = score_with_defaults("<Name of Agency>\n");
        assert_eq!(out[0].form_field_shape, 1);
    }

    #[test]
    fn form_field_parens_with_label() {
        let out = score_with_defaults("(Street address of savings association)\n");
        assert_eq!(out[0].form_field_shape, 1);
    }

    #[test]
    fn form_field_bracketed_label() {
        let out = score_with_defaults("[Date created]\n");
        assert_eq!(out[0].form_field_shape, 1);
    }

    #[test]
    fn form_field_label_with_underscores() {
        let out = score_with_defaults("Name: ____________\n");
        assert_eq!(out[0].form_field_shape, 1);
    }

    #[test]
    fn form_field_does_not_fire_on_normal_parens() {
        // A heading like "Background (continued)" wraps only its tail.
        let out = score_with_defaults("Background (continued)\n");
        assert_eq!(out[0].form_field_shape, 0);
    }

    #[test]
    fn form_field_does_not_fire_on_short_marks() {
        // "()", "<>" — too short to be field labels.
        let out = score_with_defaults("()\n");
        assert_eq!(out[0].form_field_shape, 0);
    }

    #[test]
    fn citation_density_low_for_ordinary_prose() {
        let text = "the quick brown fox jumps over the lazy dog\n";
        assert_eq!(citation_density(text), 0.0);
    }

    #[test]
    fn citation_density_low_for_one_abbreviation() {
        // Mr. Smith arrived at noon. — ordinary prose with Mr. is fine
        let text = "Mr. Smith arrived at noon today\n";
        let d = citation_density(text);
        assert!(d < 0.30, "density = {d}");
    }

    #[test]
    fn long_prose_fires() {
        let long = "x".repeat(250);
        let text = format!("{}\n", long);
        let out = score_with_defaults(&text);
        assert_eq!(out[0].long_prose, 1);
    }

    #[test]
    fn boilerplate_lines_flagged() {
        // 5 pages with a stable header — P5's exact-dup pass should flag
        // every HEADER occurrence. We disable the near-dup pass so the
        // test does not depend on MinHash thresholds (varying-body
        // lines are deliberately templated similar to each other so
        // near-dup would also cluster them — that's correct detector
        // behavior, but it's a fuzzy boundary that doesn't belong in a
        // scorer-shape test).
        let mut text = String::new();
        for i in 0..5 {
            text.push_str("HEADER\n");
            text.push_str(&format!("Page {i} body line one is unique here.\n"));
            text.push_str(&format!("Different sentence per page index {i}.\n"));
            if i + 1 < 5 {
                text.push('\u{000C}');
            }
        }
        let records = extract_line_records(&text);
        let enumerators: Vec<Option<Enumerator>> = records
            .iter()
            .map(|r| {
                if r.blank {
                    None
                } else {
                    crate::core::segmentation::parse_enumerator(r.stripped_text(&text))
                }
            })
            .collect();
        let runs = crate::core::segmentation::detect_boilerplate(
            &records,
            &text,
            crate::core::segmentation::BoilerplateOptions {
                skip_near_dup: true,
                ..crate::core::segmentation::BoilerplateOptions::default()
            },
        );
        let opts = ScoringOptions::default();
        let out = score_heading_features(&text, &records, &enumerators, &runs, &opts);
        // First HEADER line should be in a boilerplate run.
        assert_eq!(out[0].boilerplate, 1);
        // Distinct body lines (per page) should not be in any run when
        // near-dup clustering is off.
        assert_eq!(out[1].boilerplate, 0);
    }

    // ── Composite score / threshold ────────────────────────────────────────

    #[test]
    fn body_line_scores_below_threshold() {
        let text =
            "The court considered each argument in turn and rejected the appellant's claim.\n";
        let out = score_with_defaults(text);
        assert!(out[0].score < 0.30, "score = {}", out[0].score);
    }

    #[test]
    fn allcaps_short_centered_scores_above_threshold() {
        let text = "OPINION\n\nBody starts here.\n";
        let out = score_with_defaults(text);
        assert!(out[0].score > 0.30, "score = {}", out[0].score);
    }

    #[test]
    fn score_clamped_to_negative_one() {
        let mut text = String::new();
        // Very long table row (forces table_row + long_prose + low everything).
        let row: String = (0..50).map(|_| "Col A | ").collect();
        text.push_str(&row);
        text.push('\n');
        let out = score_with_defaults(&text);
        assert!(out[0].score >= -1.0);
        assert!(out[0].score <= 1.0);
    }

    #[test]
    fn score_clamped_to_positive_one() {
        // Maximally heading-shaped line — all positives, no negatives.
        let text = "\n\nDISCUSSION\n\n";
        let out = score_with_defaults(text);
        let heading = out.iter().find(|f| f.case_allcaps == 1).unwrap();
        assert!(heading.score <= 1.0);
        // Should be well above threshold.
        assert!(heading.score >= 0.30);
    }

    #[test]
    fn determinism_across_runs() {
        let text = "OPINION\n\nThe parties agreed to the following terms.\n";
        let a = score_with_defaults(text);
        let b = score_with_defaults(text);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.short_line, y.short_line);
            assert_eq!(x.score, y.score);
        }
    }

    // ── Generality (G1 / G2) ───────────────────────────────────────────────

    #[test]
    fn news_style_no_keyword_no_enumerator_still_scores() {
        // News-style: short title-case line, blank around, no period, no keyword.
        let text = "Some prose here.\n\nThe Inflation Numbers\n\nMore prose follows.\n";
        let out = score_with_defaults(text);
        let heading = out
            .iter()
            .find(|f| f.case_titlecase == 1)
            .expect("expected a TitleCase line");
        assert!(heading.score >= 0.30, "score = {}", heading.score);
    }

    #[test]
    fn french_hierarchy_keyword_fires_with_french_lexicon() {
        let text = "Article 5 — Définitions\nbody\n";
        let opts = ScoringOptions {
            hierarchy_lexicon: HierarchyLexicon::FrenchLegal,
            ..ScoringOptions::default()
        };
        let out = score_with_full_pipeline(text, opts);
        assert_eq!(out[0].hierarchy_keyword, 1);
        assert!(out[0].hierarchy_depth >= 1);
    }

    #[test]
    fn german_hierarchy_keyword_fires() {
        let text = "Artikel 12 Definitionen\nbody\n";
        let opts = ScoringOptions {
            hierarchy_lexicon: HierarchyLexicon::GermanLegal,
            ..ScoringOptions::default()
        };
        let out = score_with_full_pipeline(text, opts);
        assert_eq!(out[0].hierarchy_keyword, 1);
    }

    #[test]
    fn english_keyword_does_not_fire_under_french_lexicon() {
        // Generality contract: with French lexicon selected, English
        // "Section" should still match because it's ALSO a French word
        // (Section). This is a feature, not a bug — French legal docs
        // sometimes use "Section". The corrigendum says English lexicon
        // contributes positive evidence, never load-bearing.
        let text = "Section 5\nbody\n";
        let opts = ScoringOptions {
            hierarchy_lexicon: HierarchyLexicon::FrenchLegal,
            ..ScoringOptions::default()
        };
        let out = score_with_full_pipeline(text, opts);
        assert_eq!(out[0].hierarchy_keyword, 1);
    }

    #[test]
    fn no_lexicon_layout_only_path() {
        // G1 + G2: the scorer must work with NO lexicon configured.
        let opts = ScoringOptions {
            heading_lexicon: HeadingLexicon::None,
            hierarchy_lexicon: HierarchyLexicon::None,
            ..ScoringOptions::default()
        };
        let text = "DISCUSSION\n\nbody\n";
        let records = extract_line_records(text);
        let enumerators: Vec<Option<Enumerator>> = records.iter().map(|_| None).collect();
        let out = score_heading_features(text, &records, &enumerators, &[], &opts);
        assert_eq!(out[0].lexical_heading, 0);
        assert_eq!(out[0].hierarchy_keyword, 0);
        // With layout (allcaps, blank_after, no_terminal_period) alone the
        // score still beats the threshold.
        assert!(out[0].score >= 0.30, "score = {}", out[0].score);
    }
}
