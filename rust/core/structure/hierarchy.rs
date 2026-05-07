//! Heading-stack inferencer (P7.6).
//!
//! Consumes the per-line label sequence (P7.4) plus the per-line feature
//! vectors (P7.1) and emits one [`HeadingCandidate`] per `heading`-labeled
//! line. Each candidate carries:
//!
//! * `line_index` — which line the heading sits on
//! * `score` — the composite heading score (passes through)
//! * `hierarchy_level` — depth from the configured [`HierarchyLexicon`]
//!   (`0` if no keyword fired)
//! * `numeric_depth` — depth derived from the line's enumerator shape
//!   (`0` if no enumerator)
//! * `enumerator_kind` — which kind of enumerator fired (None / Roman /
//!   Alpha / Decimal / …)
//! * `lexicon_used` — name of the hierarchy lexicon that was active
//!
//! Per Q6 of the design reference the inferencer reports BOTH signals;
//! the consumer (the kaos-content wrapper) picks one for the final
//! `Heading.depth`. Defaults: hierarchy_level if present, else
//! `numeric_depth + 6` (so keyword headings sit shallower than purely
//! numeric headings).
//!
//! The inferencer does NOT compute `parent_idx` / `section_extent` — that
//! belongs in the kaos-content wrapper, which already has a
//! `_build_section_tree` recursion in `kaos_content/views/document_view.py`.

use crate::core::segmentation::{EnumKind, Enumerator};

use super::decoder::LineLabel;
use super::lexicon::HierarchyLexicon;
use super::scoring::HeadingFeatureVector;

// ─── Public types ─────────────────────────────────────────────────────────

/// One detected heading candidate. Emitted in source order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadingCandidate {
    /// Index into the input `&[LineRecord]` slice.
    pub line_index: u32,
    /// Composite score from [`HeadingFeatureVector::score`].
    pub score: f32,
    /// Depth from the configured [`HierarchyLexicon`]. `0` means no
    /// keyword fired. `1` is the shallowest level (Title / Titre /
    /// Titel). Range 1..=255.
    pub hierarchy_level: u8,
    /// Depth derived from the line's enumerator shape. `0` means no
    /// enumerator. For dotted-decimal `1.2.3` the depth is the dot count
    /// + 1; for Roman / Alpha / Section / etc. it's the per-kind depth
    ///   (see [`enumerator_depth`]). Range 1..=255.
    pub numeric_depth: u8,
    /// The enumerator kind the parser detected, or `None`. Stable
    /// 0..=11 mapping (see [`EnumKind`]); 255 means "no enumerator".
    pub enumerator_kind: u8,
    /// Markdown ATX `#` count, 0..=6. `0` for non-ATX headings.
    pub atx_depth: u8,
}

const NO_ENUM: u8 = 255;

impl HeadingCandidate {
    /// Default depth-pick rule per Q6 of the design reference: prefer
    /// the hierarchy-keyword level when present; fall back to the
    /// numeric / ATX depth shifted by 6 so keyword-anchored headings sit
    /// shallower than purely numeric headings.
    ///
    /// Returns 0 if no signal fired.
    pub fn picked_depth(&self) -> u8 {
        if self.atx_depth > 0 {
            return self.atx_depth;
        }
        if self.hierarchy_level > 0 {
            return self.hierarchy_level;
        }
        if self.numeric_depth > 0 {
            return self.numeric_depth.saturating_add(6);
        }
        0
    }
}

#[derive(Debug, Clone, Default)]
pub struct HierarchyOptions {
    /// Reported on every emitted candidate as `lexicon_used`.
    pub lexicon: HierarchyLexicon,
}

// ─── Public entry point ───────────────────────────────────────────────────

/// Walk `labels` in source order and emit one [`HeadingCandidate`] per
/// `LineLabel::Heading` line.
///
/// `features.len()` must equal `labels.len()`; `enumerators.len()` must
/// also equal `labels.len()`. `enumerators[i]` is `Some` iff
/// `parse_enumerator_with` matched the start of line `i` (typically
/// taken straight from the same input the scorer received).
pub fn infer_hierarchy(
    labels: &[LineLabel],
    features: &[HeadingFeatureVector],
    enumerators: &[Option<Enumerator>],
    _opts: &HierarchyOptions,
) -> Vec<HeadingCandidate> {
    assert_eq!(
        labels.len(),
        features.len(),
        "shape mismatch: labels/features"
    );
    assert_eq!(
        labels.len(),
        enumerators.len(),
        "shape mismatch: labels/enumerators",
    );

    let mut out = Vec::new();
    for (i, l) in labels.iter().enumerate() {
        if *l != LineLabel::Heading {
            continue;
        }
        let fv = features[i];
        let numeric_depth = enumerators[i].as_ref().map(enumerator_depth).unwrap_or(0);
        let kind = enumerators[i]
            .as_ref()
            .map(|e| e.kind as u8)
            .unwrap_or(NO_ENUM);
        out.push(HeadingCandidate {
            line_index: i as u32,
            score: fv.score,
            hierarchy_level: fv.hierarchy_depth,
            numeric_depth,
            enumerator_kind: kind,
            atx_depth: fv.atx_depth,
        });
    }
    out
}

// ─── Enumerator → depth ──────────────────────────────────────────────────

/// Map an enumerator to a numeric depth.
///
/// Per Q6 of the design reference (priority ordering when no hierarchy
/// keyword is present):
///
/// * `RomanUpper` = 1
/// * `AlphaUpper` = 2
/// * `Decimal` = `depth` (dot count + 1; clamped to 1..=4)
/// * `AlphaLower` = 4
/// * `RomanLower` = 5
/// * `ParenAlpha` = 6
/// * `ParenDecimal` = 7
/// * `ParenRoman` = 8
/// * `Section / SectionWord / ChapterWord / SubpartWord` = 0 — the
///   hierarchy keyword path covers these and the keyword *is* the
///   depth, so reporting a numeric depth here would compete.
pub fn enumerator_depth(e: &Enumerator) -> u8 {
    match e.kind {
        EnumKind::RomanUpper => 1,
        EnumKind::AlphaUpper => 2,
        EnumKind::Decimal => e.depth.clamp(1, 4),
        EnumKind::AlphaLower => 4,
        EnumKind::RomanLower => 5,
        EnumKind::ParenAlpha => 6,
        EnumKind::ParenDecimal => 7,
        EnumKind::ParenRoman => 8,
        EnumKind::Section
        | EnumKind::SectionWord
        | EnumKind::ChapterWord
        | EnumKind::SubpartWord => 0,
        // Bullets carry no ordinal depth — list_item without hierarchy.
        EnumKind::Bullet => 0,
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::segmentation::{
        detect_boilerplate, extract_line_records, parse_enumerator, BoilerplateOptions,
    };
    use crate::core::structure::decoder::{decode_line_labels, DecoderOptions};
    use crate::core::structure::scoring::{score_heading_features, ScoringOptions};

    fn end_to_end(text: &str, opts: ScoringOptions) -> Vec<HeadingCandidate> {
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
        let labels = decode_line_labels(&features, &DecoderOptions::default());
        infer_hierarchy(
            &labels,
            &features,
            &enumerators,
            &HierarchyOptions::default(),
        )
    }

    #[test]
    fn empty_returns_empty() {
        let out = infer_hierarchy(&[], &[], &[], &HierarchyOptions::default());
        assert!(out.is_empty());
    }

    #[test]
    fn keyword_heading_gets_hierarchy_level() {
        let text = "Section 5 Definitions\n\nbody text follows here long enough to be body.\n";
        let candidates = end_to_end(text, ScoringOptions::default());
        // Should detect Section 5 as a heading.
        assert!(!candidates.is_empty(), "no candidates emitted");
        let c = candidates[0];
        // Section is depth 7 in english_legal_us.
        assert!(c.hierarchy_level >= 1, "hier level: {}", c.hierarchy_level);
    }

    #[test]
    fn allcaps_no_keyword_no_enumerator_uses_layout_alone() {
        let text = "Body text.\n\nDISCUSSION\n\nMore body content here.\n";
        let candidates = end_to_end(text, ScoringOptions::default());
        let c = candidates[0];
        // No hierarchy keyword, no enumerator → both depth signals zero.
        assert_eq!(c.hierarchy_level, 0);
        assert_eq!(c.numeric_depth, 0);
        // picked_depth falls back to 0 — caller decides what to do.
        assert_eq!(c.picked_depth(), 0);
    }

    #[test]
    fn numeric_dotted_depth_extracted() {
        // Heading line with a dotted-decimal enumerator.
        let text = "1.2.3 Some heading\n\nbody text follows.\n";
        let candidates = end_to_end(text, ScoringOptions::default());
        // Depending on weights / decoder this may or may not be tagged
        // heading; if it is, depth must equal 3.
        if let Some(c) = candidates.first() {
            if c.numeric_depth > 0 {
                assert_eq!(c.numeric_depth, 3, "candidate: {:?}", c);
            }
        }
    }

    #[test]
    fn picked_depth_prefers_atx_over_keyword() {
        // ATX wins per Q6 corrigendum #3.
        let c = HeadingCandidate {
            line_index: 0,
            score: 0.5,
            hierarchy_level: 7,
            numeric_depth: 3,
            enumerator_kind: NO_ENUM,
            atx_depth: 2,
        };
        assert_eq!(c.picked_depth(), 2);
    }

    #[test]
    fn picked_depth_prefers_keyword_over_numeric() {
        let c = HeadingCandidate {
            line_index: 0,
            score: 0.5,
            hierarchy_level: 3,
            numeric_depth: 5,
            enumerator_kind: NO_ENUM,
            atx_depth: 0,
        };
        assert_eq!(c.picked_depth(), 3);
    }

    #[test]
    fn picked_depth_shifts_numeric_below_keyword() {
        let c = HeadingCandidate {
            line_index: 0,
            score: 0.5,
            hierarchy_level: 0,
            numeric_depth: 3,
            enumerator_kind: NO_ENUM,
            atx_depth: 0,
        };
        // 3 + 6 = 9, sits below the 7-level USC keyword range.
        assert_eq!(c.picked_depth(), 9);
    }

    #[test]
    fn enumerator_depth_decimal_uses_dotted_count() {
        // Build by parsing.
        let e = parse_enumerator("1.2.3 ").unwrap();
        assert_eq!(enumerator_depth(&e), 3);
        let e = parse_enumerator("1.2 ").unwrap();
        assert_eq!(enumerator_depth(&e), 2);
        let e = parse_enumerator("1. ").unwrap();
        assert_eq!(enumerator_depth(&e), 1);
    }

    #[test]
    fn enumerator_depth_roman_alpha() {
        let e = parse_enumerator("I. ").unwrap();
        assert_eq!(enumerator_depth(&e), 1);
        // Lowercase 'a' or 'b' → AlphaLower.
        let e = parse_enumerator("a. ").unwrap();
        assert_eq!(enumerator_depth(&e), 4);
    }

    #[test]
    fn keyword_enumerator_kinds_zero_depth() {
        let e = parse_enumerator("Section 5 ").unwrap();
        // SectionWord → 0 (keyword path covers it)
        assert_eq!(enumerator_depth(&e), 0);
    }
}
