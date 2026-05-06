//! Viterbi sequence decoder over per-line heading-feature vectors (P7.4).
//!
//! Consumes `&[HeadingFeatureVector]` and emits one [`LineLabel`] per line
//! by minimising emission + transition costs in negative-log-prob units.
//!
//! ## Why Viterbi
//!
//! Per-line scoring is independent — it cannot tell that a single
//! `heading`-shaped line in a run of 12 `table_row`s is much more likely
//! a stray table cell than a real heading. The Viterbi pass smooths
//! these out by finding the globally lowest-cost label sequence under
//! the 7×7 transition matrix from Q4 of the design reference.
//!
//! Algorithm: standard log-space Viterbi, O(N × K²) for N lines and
//! K = 7 states. 100k lines = 4.9M ops; sub-100ms in Rust.
//!
//! Reference:
//! * <https://en.wikipedia.org/wiki/Viterbi_algorithm>
//! * GROBID's CRF segmentation pipeline shape, see
//!   <https://grobid.readthedocs.io/en/latest/training/segmentation/>
//!
//! ## Cost convention
//!
//! Costs are non-negative `f32`. The decoder minimises sum of costs;
//! lower is better. `INF` (here `f32::INFINITY`) encodes "forbidden".
//! Transitions are integers in `{0, 1, 5}` per the v1 design — INF is
//! reserved for v2.

use super::scoring::HeadingFeatureVector;

// ─── Public types ─────────────────────────────────────────────────────────

/// Seven-state line label set (Q3 of the design reference).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LineLabel {
    Blank = 0,
    Heading = 1,
    Body = 2,
    ListItem = 3,
    TableRow = 4,
    Metadata = 5,
    Boilerplate = 6,
}

impl LineLabel {
    pub const ALL: [Self; 7] = [
        Self::Blank,
        Self::Heading,
        Self::Body,
        Self::ListItem,
        Self::TableRow,
        Self::Metadata,
        Self::Boilerplate,
    ];

    pub const COUNT: usize = 7;

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Blank => "blank",
            Self::Heading => "heading",
            Self::Body => "body",
            Self::ListItem => "list_item",
            Self::TableRow => "table_row",
            Self::Metadata => "metadata",
            Self::Boilerplate => "boilerplate",
        }
    }
}

/// 7×7 transition cost matrix. Rows = from, cols = to. Default values
/// are pinned in Q4 of the design reference.
#[derive(Debug, Clone, Copy)]
pub struct TransitionMatrix {
    /// `costs[from][to]` — non-negative `f32`.
    pub costs: [[f32; LineLabel::COUNT]; LineLabel::COUNT],
}

impl TransitionMatrix {
    /// Returns the v1 default transition matrix from Q4 of the design
    /// reference (corrigendum after F-R7 real-corpus inspection).
    /// Costs are integer `{0, 1, 5}` packed into `f32`.
    ///
    /// F-R7 corrigendum: `heading → list_item = 0` (was 1) because
    /// "## Methods\n- step one" / "Sec.\n271. …" / "## Installation\n
    /// pip install …" patterns are canonical in Markdown, RFCs, and
    /// statutory TOCs. Penalizing them was a v1 guess that did not
    /// match real corpora.
    pub const fn v1_default() -> Self {
        // Index order matches `LineLabel::ALL`:
        // 0 blank · 1 heading · 2 body · 3 list_item · 4 table_row · 5 metadata · 6 boilerplate
        let costs = [
            //  blank  head  body  list  table  meta  boil
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // from blank
            [0.0, 1.0, 0.0, 0.0, 5.0, 1.0, 5.0], // from heading
            [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 5.0], // from body
            [0.0, 1.0, 1.0, 0.0, 1.0, 5.0, 5.0], // from list_item
            [0.0, 5.0, 1.0, 1.0, 0.0, 5.0, 5.0], // from table_row
            [0.0, 1.0, 1.0, 5.0, 5.0, 0.0, 1.0], // from metadata
            [0.0, 5.0, 0.0, 5.0, 5.0, 1.0, 0.0], // from boilerplate
        ];
        Self { costs }
    }

    #[inline]
    pub fn cost(&self, from: LineLabel, to: LineLabel) -> f32 {
        self.costs[from.index()][to.index()]
    }
}

impl Default for TransitionMatrix {
    fn default() -> Self {
        Self::v1_default()
    }
}

/// Per-label emission cost baselines.
///
/// Per Q5 of the design reference, these are the v1 placeholder costs.
/// They are intentionally exposed (not buried as function-local
/// constants) so callers can override them and so weight-calibration
/// against the multi-domain corpus (G8 contract) can sweep them
/// alongside [`ScoringWeights`](super::scoring::ScoringWeights). All
/// fields are non-negative `f32`; `f32::INFINITY` encodes "forbidden".
#[derive(Debug, Clone, Copy)]
pub struct EmissionCosts {
    /// Multiplier on the per-line heading score: `(1.0 - score) *
    /// heading_emit_scale`. Default `1.0`.
    pub heading_emit_scale: f32,
    /// Body-label baseline emission cost (catch-all). Default `0.6`.
    pub body_baseline: f32,
    /// `TableRow` cost when `table_row_shape == 1`. Default `0.2`.
    pub table_row_strong: f32,
    /// `TableRow` cost otherwise. Default `1.5`.
    pub table_row_weak: f32,
    /// `ListItem` cost when `has_enumerator == 1 && short_line == 1`.
    /// Default `0.3`.
    pub list_item_strong: f32,
    /// `ListItem` cost when `has_enumerator == 1` but the line is not
    /// short (long enumerated lines — typical regulation / contract
    /// list items). Default `0.5` so list_item is preferred over body
    /// (0.6 baseline) but heading still wins on truly heading-shaped
    /// short enumerated lines. F-R1 fix.
    pub list_item_with_enumerator: f32,
    /// `ListItem` cost otherwise. Default `1.2`.
    pub list_item_weak: f32,
    /// `Metadata` cost when `inline_colon && !has_enumerator &&
    /// (case_initcap || case_titlecase)`. Default `0.4`.
    pub metadata_strong: f32,
    /// `Metadata` cost otherwise. Default `1.5`.
    pub metadata_weak: f32,
}

impl Default for EmissionCosts {
    fn default() -> Self {
        // v1 placeholders. Calibration against the multi-domain corpus
        // (G8 contract) may move these.
        Self {
            heading_emit_scale: 1.0,
            body_baseline: 0.6,
            table_row_strong: 0.2,
            table_row_weak: 1.5,
            list_item_strong: 0.3,
            list_item_with_enumerator: 0.5,
            list_item_weak: 1.2,
            metadata_strong: 0.4,
            metadata_weak: 1.5,
        }
    }
}

/// Post-decode title-promotion pass parameters (F-R9).
///
/// The Viterbi decoder optimizes a per-line label sequence under the
/// transition matrix, but multi-line document titles (`REAL PROPERTY's
/// PURCHASE OF DEVELOPMENT RIGHTS / AND SALE AGREEMENT`) sometimes
/// score below threshold per-line and need a post-pass to recover.
///
/// All knobs configurable so callers can disable the pass entirely or
/// adjust its conservatism.
#[derive(Debug, Clone, Copy)]
pub struct PostDecodeOptions {
    /// Master toggle. When `false` the post-pass is skipped entirely
    /// and the raw Viterbi output is returned. Default `true`.
    pub enable: bool,
    /// Maximum number of body lines to promote *after* a heading
    /// (heading-continuation case). Default 3.
    pub max_continuation_lines: u8,
    /// Maximum length of a "heading-shaped run" — a sequence of body
    /// lines all bounded by blanks that look heading-shaped on their
    /// own. Default 3.
    pub max_run_lines: u8,
    /// Require at least one line in the heading-shaped run to have
    /// `case_allcaps=1`. Default `true`. Set `false` to allow
    /// title-case-only titles to be promoted.
    pub require_caps_in_run: bool,
    /// F-R7: Recognize naked-enum + title two-line pairs (USC TOC,
    /// statutory section indexes) and promote both to `list_item`.
    /// Default `true`. Set `false` to disable the pass — useful when
    /// callers want strict per-line Viterbi output.
    pub toc_pair_recognition: bool,
}

impl Default for PostDecodeOptions {
    fn default() -> Self {
        Self {
            enable: true,
            max_continuation_lines: 3,
            max_run_lines: 3,
            require_caps_in_run: true,
            toc_pair_recognition: true,
        }
    }
}

/// Decoder options.
#[derive(Debug, Clone, Copy)]
pub struct DecoderOptions {
    pub transitions: TransitionMatrix,
    pub emissions: EmissionCosts,
    /// Post-decode promotion pass (F-R9 multi-line titles).
    pub post_decode: PostDecodeOptions,
    /// `Boilerplate` label is forced for any line with `boilerplate=1`
    /// in the feature vector. Default `true`.
    pub force_boilerplate_label: bool,
    /// `Blank` label is forced for any line with `is_blank=1` in the
    /// feature vector. Default `true`.
    pub force_blank_label: bool,
}

impl Default for DecoderOptions {
    fn default() -> Self {
        Self {
            transitions: TransitionMatrix::v1_default(),
            emissions: EmissionCosts::default(),
            post_decode: PostDecodeOptions::default(),
            force_boilerplate_label: true,
            force_blank_label: true,
        }
    }
}

// ─── Emission costs ───────────────────────────────────────────────────────

/// Compute the emission cost vector for a single line.
///
/// Per Q5 of the design reference:
///
/// * heading: `(1.0 - score) * heading_emit_scale`
/// * body: catch-all (`1.0 - max(other heading-positive evidence)`)
/// * list_item: low cost iff `has_enumerator` AND `short_line`
/// * table_row: low cost iff `table_row_shape`
/// * metadata: low cost iff `colon_suffix` AND NOT `has_enumerator` AND
///   `case_initcap`
/// * boilerplate: 0 if `boilerplate=1`, else INF (P5 pre-determines —
///   the decoder cannot invent boilerplate)
/// * blank: 0 if `is_blank=1`, else INF
fn emission_costs(fv: &HeadingFeatureVector, opts: &DecoderOptions) -> [f32; LineLabel::COUNT] {
    let mut e = [1.0f32; LineLabel::COUNT];

    // blank
    e[LineLabel::Blank.index()] = if fv.is_blank == 1 { 0.0 } else { f32::INFINITY };

    // boilerplate
    e[LineLabel::Boilerplate.index()] = if fv.boilerplate == 1 {
        0.0
    } else {
        f32::INFINITY
    };

    if fv.is_blank == 1 {
        // For blank lines, all non-blank labels are forbidden when the
        // option is set; otherwise just stay at defaults.
        if opts.force_blank_label {
            for l in LineLabel::ALL {
                if l != LineLabel::Blank {
                    e[l.index()] = f32::INFINITY;
                }
            }
        }
        return e;
    }

    if fv.boilerplate == 1 && opts.force_boilerplate_label {
        // Pin boilerplate label.
        for l in LineLabel::ALL {
            if l != LineLabel::Boilerplate {
                e[l.index()] = f32::INFINITY;
            }
        }
        return e;
    }

    let em = &opts.emissions;

    // heading: lower cost = stronger evidence.
    let heading_cost = (1.0 - fv.score).clamp(0.0, 2.0) * em.heading_emit_scale;
    e[LineLabel::Heading.index()] = heading_cost;

    // table_row: low iff table_row_shape; otherwise discouraged.
    e[LineLabel::TableRow.index()] = if fv.table_row_shape == 1 {
        em.table_row_strong
    } else {
        em.table_row_weak
    };

    // list_item: three tiers (F-R1).
    //   * strong: has_enumerator AND short_line — canonical list item.
    //   * with_enumerator: has_enumerator only — long enumerated line,
    //     still structurally a list item (lower than body baseline).
    //   * weak: no enumerator — discouraged.
    e[LineLabel::ListItem.index()] = if fv.has_enumerator == 1 && fv.short_line == 1 {
        em.list_item_strong
    } else if fv.has_enumerator == 1 {
        em.list_item_with_enumerator
    } else {
        em.list_item_weak
    };

    // metadata: low iff inline_colon AND NOT has_enumerator AND
    // short_line. The inline-colon shape (`Author: Jane Doe`,
    // `Date: 2026-05-05`, `Case Number: 22-1234`) is the canonical
    // metadata signature; case profile is unreliable when values are
    // digit- or punctuation-heavy. `short_line` rules out long prose
    // sentences that happen to contain a mid-sentence colon.
    // `Libro Primero: De las personas` / `Chapter 5: Definitions` look
    // metadata-shaped from inline_colon+short, but the hierarchy_keyword
    // signal disambiguates them as headings.
    let metadata_strong = fv.inline_colon == 1
        && fv.has_enumerator == 0
        && fv.short_line == 1
        && fv.hierarchy_keyword == 0;
    e[LineLabel::Metadata.index()] = if metadata_strong {
        em.metadata_strong
    } else {
        em.metadata_weak
    };

    // body: catch-all. Low when other labels are weak. Define as a
    // baseline so the decoder doesn't pay to switch off body.
    e[LineLabel::Body.index()] = em.body_baseline;

    e
}

// ─── Public entry point ───────────────────────────────────────────────────

/// Decode the most-likely label sequence over `features`. Returns one
/// [`LineLabel`] per feature vector.
///
/// O(N × K²) time, O(N × K) auxiliary space (for the back-pointers).
pub fn decode_line_labels(
    features: &[HeadingFeatureVector],
    opts: &DecoderOptions,
) -> Vec<LineLabel> {
    let n = features.len();
    if n == 0 {
        return Vec::new();
    }
    const K: usize = LineLabel::COUNT;

    // Costs at line t for ending in state s.
    let mut prev_cost = [f32::INFINITY; K];
    let mut curr_cost = [f32::INFINITY; K];
    // Back-pointers: backptr[t][s] = best previous state.
    let mut backptr: Vec<[u8; K]> = vec![[0u8; K]; n];

    // ── Init t = 0 ──
    let e0 = emission_costs(&features[0], opts);
    for s in 0..K {
        prev_cost[s] = e0[s];
        backptr[0][s] = 0;
    }

    // ── Recurse t = 1..n ──
    for t in 1..n {
        let e = emission_costs(&features[t], opts);
        for to in 0..K {
            if !e[to].is_finite() {
                curr_cost[to] = f32::INFINITY;
                continue;
            }
            let mut best = f32::INFINITY;
            let mut best_from = 0u8;
            for (from, &prev) in prev_cost.iter().enumerate() {
                if !prev.is_finite() {
                    continue;
                }
                let cand = prev + opts.transitions.costs[from][to] + e[to];
                if cand < best {
                    best = cand;
                    best_from = from as u8;
                }
            }
            curr_cost[to] = best;
            backptr[t][to] = best_from;
        }
        prev_cost.copy_from_slice(&curr_cost);
    }

    // ── Find the best final state ──
    let mut best_final = 0usize;
    let mut best_final_cost = f32::INFINITY;
    for (s, &cost) in prev_cost.iter().enumerate() {
        if cost < best_final_cost {
            best_final_cost = cost;
            best_final = s;
        }
    }

    // ── Trace back ──
    let mut path: Vec<LineLabel> = Vec::with_capacity(n);
    let mut s = best_final as u8;
    for t in (0..n).rev() {
        path.push(LineLabel::ALL[s as usize]);
        if t > 0 {
            s = backptr[t][s as usize];
        }
    }
    path.reverse();

    // F-R9: multi-line title continuation pass. Tunable via
    // opts.post_decode; skip when disabled.
    if opts.post_decode.enable {
        promote_multi_line_title_continuations(&mut path, features, &opts.post_decode);
        if opts.post_decode.toc_pair_recognition {
            promote_toc_pairs(&mut path, features);
        }
    }

    path
}

/// F-R7: Detect TOC two-line entries — a "naked enumerator" line
/// (`271.`, `1.`, `(a)`) followed immediately by a title line — and
/// promote both to `list_item`. Pattern is canonical in USC tables of
/// contents, statutory section indexes, RFC TOCs.
///
/// The pattern signature is intentionally tight to avoid false
/// promotions on prose where a number happens to land alone on a line,
/// and on real ATX-numbered headings (`## 1.`):
///
/// * Line A: non-blank, `has_enumerator=1`, **no letter case at all**
///   (`case_allcaps=0 && case_titlecase=0 && case_initcap=0`),
///   `atx_depth==0` (so a real markdown ATX heading is never demoted),
///   currently labeled `Body`, `ListItem`, or `Heading`. The Heading
///   case covers the F-R7 scenario where the scorer's per-line signal
///   pushes a naked TOC enumerator above threshold.
/// * Line B: index `A+1` (immediately consecutive — no intervening
///   blank), non-blank, `has_enumerator=0`, currently labeled `Body`,
///   and **not** structurally a different element: no `table_row_shape`,
///   no `boilerplate`, no `column_gap_only`. We deliberately do NOT
///   require letter-case bits — title lines with stop-words ("of",
///   "to", "for") fall into `MixedCase` which is not exposed as a bit
///   but is the dominant shape for real-world TOC titles.
///
/// Both lines are then re-labeled `ListItem`. Operates left-to-right
/// without backtracking; once a pair is promoted, scanning resumes
/// at `B+1` so the same line is never reused as the A of a later pair.
fn promote_toc_pairs(path: &mut [LineLabel], features: &[HeadingFeatureVector]) {
    if path.len() < 2 {
        return;
    }
    let mut i = 0usize;
    while i + 1 < path.len() {
        let a = &features[i];
        let a_label_eligible = matches!(
            path[i],
            LineLabel::Body | LineLabel::ListItem | LineLabel::Heading
        );
        let a_is_naked_enum = a.is_blank == 0
            && a.has_enumerator == 1
            && a.case_allcaps == 0
            && a.case_titlecase == 0
            && a.case_initcap == 0
            && a.atx_depth == 0
            && a_label_eligible;
        if !a_is_naked_enum {
            i += 1;
            continue;
        }
        let b_idx = i + 1;
        let b = &features[b_idx];
        let b_is_title_partner = b.is_blank == 0
            && b.has_enumerator == 0
            && b.table_row_shape == 0
            && b.boilerplate == 0
            && b.column_gap_only == 0
            && path[b_idx] == LineLabel::Body;
        if b_is_title_partner {
            path[i] = LineLabel::ListItem;
            path[b_idx] = LineLabel::ListItem;
            i = b_idx + 1;
        } else {
            i += 1;
        }
    }
}

/// Promote body lines that are clearly part of a multi-line title to
/// heading. Two distinct cases:
///
/// 1. **Heading continuation:** body line follows a heading without an
///    intervening blank, AND the body line is itself heading-shaped
///    (short, no terminal period, no inline-colon, no enumerator, no
///    table-row shape). Up-to-3 promotions per heading.
/// 2. **Heading-shaped run:** a consecutive sequence of body lines
///    bounded by blanks where every line is heading-shaped, the run
///    is short (≤3 lines), and at least one line has `case_allcaps=1`.
///    Promotes the entire run to heading.
fn promote_multi_line_title_continuations(
    path: &mut [LineLabel],
    features: &[HeadingFeatureVector],
    opts: &PostDecodeOptions,
) {
    if path.len() < 2 {
        return;
    }

    // Case 1: heading continuation.
    let max_cont = opts.max_continuation_lines;
    let mut i = 1;
    while i < path.len() {
        if path[i - 1] != LineLabel::Heading {
            i += 1;
            continue;
        }
        let mut promoted = 0u8;
        let mut j = i;
        while j < path.len() && promoted < max_cont {
            if path[j] != LineLabel::Body || !is_heading_shaped(&features[j]) {
                break;
            }
            path[j] = LineLabel::Heading;
            promoted += 1;
            j += 1;
        }
        i = (i + 1).max(j);
    }

    // Case 2: heading-shaped run bounded by blanks.
    let max_run = opts.max_run_lines as usize;
    let require_caps = opts.require_caps_in_run;
    let mut i = 0;
    while i < path.len() {
        if path[i] != LineLabel::Body || features[i].is_blank == 1 {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < path.len() && path[j] == LineLabel::Body && features[j].is_blank == 0 {
            j += 1;
        }
        let run_len = j - i;
        if run_len > 0 && run_len <= max_run {
            let all_shaped = (i..j).all(|k| is_heading_shaped(&features[k]));
            let caps_ok = !require_caps || (i..j).any(|k| features[k].case_allcaps == 1);
            let blank_bounded = features[i].blank_before == 1 && features[j - 1].blank_after == 1;
            if all_shaped && caps_ok && blank_bounded {
                for slot in path.iter_mut().take(j).skip(i) {
                    *slot = LineLabel::Heading;
                }
            }
        }
        i = j.max(i + 1);
    }
}

/// Is a line heading-shaped on its own? Used by the post-decoder
/// promotion heuristics. Conservative — requires multiple positive
/// signals AND a non-AllLower case profile so plain prose lines
/// don't trip the promotion path.
fn is_heading_shaped(fv: &HeadingFeatureVector) -> bool {
    let case_ok = fv.case_allcaps == 1 || fv.case_titlecase == 1 || fv.case_initcap == 1;
    fv.is_blank == 0
        && fv.short_line == 1
        && fv.no_terminal_period == 1
        && fv.inline_colon == 0
        && fv.has_enumerator == 0
        && fv.table_row_shape == 0
        && fv.boilerplate == 0
        && fv.long_prose == 0
        && case_ok
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::segmentation::{
        detect_boilerplate, extract_line_records, parse_enumerator, BoilerplateOptions, Enumerator,
    };
    use crate::core::structure::scoring::{score_heading_features, ScoringOptions};

    fn end_to_end(text: &str, opts: ScoringOptions) -> Vec<LineLabel> {
        let records = extract_line_records(text);
        let enumerators: Vec<Option<Enumerator>> = records
            .iter()
            .map(|r| {
                if r.blank {
                    None
                } else {
                    parse_enumerator(r.stripped_text(text))
                }
            })
            .collect();
        let runs = detect_boilerplate(&records, text, BoilerplateOptions::default());
        let features = score_heading_features(text, &records, &enumerators, &runs, &opts);
        decode_line_labels(&features, &DecoderOptions::default())
    }

    #[test]
    fn empty_returns_empty() {
        let labels = decode_line_labels(&[], &DecoderOptions::default());
        assert!(labels.is_empty());
    }

    #[test]
    fn blank_lines_get_blank_label() {
        let text = "OPINION\n\nBody content here.\n";
        let labels = end_to_end(text, ScoringOptions::default());
        // Exactly one blank line at position 1.
        assert_eq!(labels[1], LineLabel::Blank);
    }

    #[test]
    fn allcaps_short_blank_around_decoded_as_heading() {
        let text = "Body text.\n\nDISCUSSION\n\nMore body.\n";
        let labels = end_to_end(text, ScoringOptions::default());
        // The DISCUSSION line should be tagged heading.
        let discussion = labels
            .iter()
            .position(|l| *l == LineLabel::Heading)
            .expect("expected at least one heading label");
        // Verify it lines up with the third record (DISCUSSION).
        let recs = extract_line_records(text);
        assert!(recs[discussion].stripped_text(text).contains("DISCUSSION"));
    }

    #[test]
    fn boilerplate_lines_decoded_as_boilerplate() {
        let mut text = String::new();
        for i in 0..5 {
            text.push_str("HEADER LINE\n");
            text.push_str("body content\n");
            text.push_str(&format!("body unique on page {i}\n"));
            if i + 1 < 5 {
                text.push('\u{000C}');
            }
        }
        let labels = end_to_end(&text, ScoringOptions::default());
        // First line is the recurring HEADER → boilerplate.
        assert_eq!(labels[0], LineLabel::Boilerplate);
    }

    #[test]
    fn table_rows_decoded_as_table() {
        let text = "Col A | Col B | Col C\nVal 1 | Val 2 | Val 3\nVal 4 | Val 5 | Val 6\n";
        let labels = end_to_end(text, ScoringOptions::default());
        // All three lines should be table_row.
        for l in &labels {
            assert_eq!(*l, LineLabel::TableRow, "got {:?}", l);
        }
    }

    #[test]
    fn enumerated_short_lines_decoded_as_list_items() {
        let text = "(a) Apples\n(b) Bananas\n(c) Cherries\n";
        let labels = end_to_end(text, ScoringOptions::default());
        for l in &labels {
            assert!(
                matches!(l, LineLabel::ListItem | LineLabel::Heading),
                "got {:?}",
                l
            );
        }
    }

    #[test]
    fn body_prose_decoded_as_body() {
        let text =
            "The court considered each argument in turn and rejected the appellant's claim.\n\
                    There were several issues raised at trial that bear repetition here.\n\
                    The defendant filed a timely notice of appeal in due course.\n";
        let labels = end_to_end(text, ScoringOptions::default());
        for l in &labels {
            assert_eq!(*l, LineLabel::Body, "got {:?}", l);
        }
    }

    #[test]
    fn metadata_inline_colon_decoded_as_metadata() {
        let text =
            "Author: Jane Doe\nDate: 2026-05-05\nCase Number: 22-1234\n\nbody text follows.\n";
        let labels = end_to_end(text, ScoringOptions::default());
        // The first three inline-colon lines should land on metadata.
        for label in labels.iter().take(3) {
            assert_eq!(*label, LineLabel::Metadata);
        }
    }

    // ── Viterbi smoothing: stray heading-shaped line in a table ──

    #[test]
    fn stray_heading_inside_table_run_gets_smoothed() {
        // 12 table rows surround a single heading-shaped short line.
        // Per the design reference's `heading → table_row` cost = 5 and
        // `table_row → heading` cost = 5, the decoder should NOT label
        // the middle line `heading` because that would pay both
        // transition penalties.
        let mut text = String::new();
        for _ in 0..6 {
            text.push_str("Col A | Col B | Col C\n");
        }
        // Stray short uppercase line that would score above threshold
        // independently:
        text.push_str("ROW\n");
        for _ in 0..6 {
            text.push_str("Col A | Col B | Col C\n");
        }
        let labels = end_to_end(&text, ScoringOptions::default());
        let stray_idx = 6;
        // Note: depending on weights, "ROW" may be classified as table_row
        // (preferred by smoothing) or list_item. It must NOT be heading.
        assert_ne!(labels[stray_idx], LineLabel::Heading);
    }

    // ── Transition matrix shape ──

    #[test]
    fn transition_matrix_heading_to_list_item_is_zero() {
        // F-R7: heading -> list_item must be 0 in the v1_default matrix
        // so TOC patterns ("Sec.\n271. Title") flow naturally.
        let m = TransitionMatrix::v1_default();
        assert_eq!(m.cost(LineLabel::Heading, LineLabel::ListItem), 0.0);
    }

    // ── F-R7: TOC two-line-pair recognition ──

    #[test]
    fn toc_pair_recognition_promotes_naked_enum_plus_title() {
        // USC-style TOC: "271.\nUse of information ...\n272.\nUse of military ..."
        // Without the post-pass these all fall to Body (the title lines have
        // no enumerator and are too long for short_line). With the post-pass
        // both halves of each pair become ListItem.
        let text = "Section index follows.\n\n\
                    271.\n\
                    Use of information collected during military operations.\n\
                    272.\n\
                    Use of military equipment and facilities.\n\
                    273.\n\
                    Training and advising civilian law enforcement officials.\n\n\
                    More body text after the TOC block.\n";
        let labels = end_to_end(text, ScoringOptions::default());
        // First non-blank, non-TOC line: the prose intro → Body.
        // Then the 6 TOC lines (3 pairs) → ListItem.
        let recs = extract_line_records(text);
        let pair_indices: Vec<usize> = (0..recs.len())
            .filter(|&i| {
                let s = recs[i].stripped_text(text);
                s.starts_with('2') || s.starts_with("Use ") || s.starts_with("Training")
            })
            .collect();
        assert!(pair_indices.len() >= 6, "expected ≥6 TOC pair lines");
        for idx in pair_indices {
            assert_eq!(
                labels[idx],
                LineLabel::ListItem,
                "expected list_item at line {idx}: {:?}",
                recs[idx].stripped_text(text)
            );
        }
    }

    #[test]
    fn toc_pair_recognition_can_be_disabled() {
        // Same fixture, but disable the post-pass — labels should NOT be
        // ListItem on the TOC pairs (they fall back to whatever the raw
        // Viterbi sequence emits).
        let text = "271.\nUse of information collected during military operations.\n";
        let mut decoder_opts = DecoderOptions::default();
        decoder_opts.post_decode.toc_pair_recognition = false;
        let records = extract_line_records(text);
        let enumerators: Vec<Option<Enumerator>> = records
            .iter()
            .map(|r| {
                if r.blank {
                    None
                } else {
                    parse_enumerator(r.stripped_text(text))
                }
            })
            .collect();
        let runs = detect_boilerplate(&records, text, BoilerplateOptions::default());
        let scoring_opts = ScoringOptions::default();
        let features = score_heading_features(text, &records, &enumerators, &runs, &scoring_opts);
        let labels_off = decode_line_labels(&features, &decoder_opts);

        // With the pass off, the title line (no enumerator, no other heading
        // signal) lands on Body — at least one of the two lines must NOT be
        // ListItem to confirm the pass is doing the work.
        let on_labels = decode_line_labels(&features, &DecoderOptions::default());
        let pass_changed_something = on_labels.iter().zip(labels_off.iter()).any(|(a, b)| a != b);
        assert!(
            pass_changed_something,
            "post-pass should change at least one label vs. the disabled run"
        );
    }

    #[test]
    fn toc_pair_recognition_does_not_promote_unrelated_short_numbers() {
        // A naked "5." in prose should NOT pull the next line into ListItem
        // unless that next line is partner-shaped (no enumerator, has letters,
        // immediately consecutive).
        let text = "Conclusion.\n\nThe total was 5.\nThis sentence stands alone.\n";
        let labels = end_to_end(text, ScoringOptions::default());
        // None of the lines should be promoted to ListItem because none of them
        // is a naked-enum (line "5." is not isolated — it ends a sentence).
        for label in &labels {
            assert_ne!(*label, LineLabel::ListItem, "got unexpected list_item");
        }
    }

    #[test]
    fn transition_matrix_default_zero_diagonal() {
        let m = TransitionMatrix::v1_default();
        // Same-state transitions have cost 0 (or 1 for heading→heading,
        // 0 for body→body).
        assert_eq!(m.cost(LineLabel::Body, LineLabel::Body), 0.0);
        assert_eq!(m.cost(LineLabel::TableRow, LineLabel::TableRow), 0.0);
        // Crossing into table_row from heading is heavily penalized.
        assert!(m.cost(LineLabel::Heading, LineLabel::TableRow) >= 5.0);
        // boilerplate → body is canonical (cost 0).
        assert_eq!(m.cost(LineLabel::Boilerplate, LineLabel::Body), 0.0);
    }
}
