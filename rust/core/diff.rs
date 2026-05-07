//! Sentence- and paragraph-level document diffing.
//!
//! Two documents are segmented at a chosen granularity (sentence,
//! paragraph, line), every left-segment is scored against every
//! right-segment with a configurable similarity metric, and a greedy
//! highest-score-first assignment produces a list of `SegmentChange`s.
//! Pairs above `match_threshold` are `Unchanged`; pairs in
//! `[modify_threshold, match_threshold)` are `Modified`; unmatched
//! left-segments are `Removed`; unmatched right-segments are `Added`.
//!
//! When `detect_moves` is enabled, an `Unchanged` pair whose normalized
//! index distance exceeds `move_distance_ratio` is reclassified as
//! `Moved`. Hungarian-optimal assignment is intentionally out of scope
//! for v1 — the greedy variant is good enough for most documents and
//! avoids a much larger implementation surface.
//!
//! Byte offsets are preserved through the core; the binding layer
//! converts them to char offsets at the FFI boundary per the project's
//! standing rule.

use rayon::prelude::*;

use crate::core::algorithms::dispatch::{compute_similarity, MetricConfig};
use crate::core::segmentation::{
    default_tokenizer, segment_lines, segment_paragraphs, segment_paragraphs_simple,
    segment_sentences, PunktSentenceTokenizer, Segment,
};

/// Granularity of segmentation for diffing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// Sentence-level using Punkt.
    Sentence,
    /// Paragraph-level (sentence-aware via Punkt; only breaks at sentence boundaries).
    Paragraph,
    /// Line-level (no Punkt needed).
    Line,
    /// Simple paragraph splitting (blank-line, no Punkt).
    ParagraphSimple,
}

/// Kind of change for a `SegmentChange`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    /// High-similarity match, segments are at comparable positions.
    Unchanged,
    /// Match in `[modify_threshold, match_threshold)` — recognisable, but altered.
    Modified,
    /// High-similarity match but the position shifted by more than
    /// `move_distance_ratio` (only emitted when `detect_moves` is true).
    Moved,
    /// Segment in `b` had no qualifying match in `a`.
    Added,
    /// Segment in `a` had no qualifying match in `b`.
    Removed,
}

impl ChangeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unchanged => "unchanged",
            Self::Modified => "modified",
            Self::Moved => "moved",
            Self::Added => "added",
            Self::Removed => "removed",
        }
    }
}

/// Reference to a segment within one of the two documents being diffed.
#[derive(Debug, Clone, Copy)]
pub struct SegmentRef {
    /// 0-based segment index within its source document.
    pub index: u32,
    /// Byte offset start (inclusive) into the source document.
    pub start: u32,
    /// Byte offset end (exclusive) into the source document.
    pub end: u32,
}

/// One segment-level change between two documents.
#[derive(Debug, Clone)]
pub struct SegmentChange {
    pub kind: ChangeKind,
    /// Segment in the left (a) document, when present.
    pub left: Option<SegmentRef>,
    /// Segment in the right (b) document, when present.
    pub right: Option<SegmentRef>,
    /// Similarity score on `[0, 1]`. `0.0` for `Added` / `Removed`.
    pub score: f32,
}

/// Configuration for `diff_documents`.
#[derive(Debug, Clone)]
pub struct DiffConfig {
    pub granularity: Granularity,
    /// Algorithm key — same names as the dispatch layer (e.g. `"token-jaccard"`,
    /// `"jaro-winkler"`, `"ngram-cosine"`).
    pub algorithm: String,
    /// Knobs for the metric (n, lowercase, prefix_weight).
    pub metric: MetricConfig,
    /// Pairs scoring at or above this similarity are `Unchanged` (or `Moved`).
    pub match_threshold: f32,
    /// Pairs in `[modify_threshold, match_threshold)` are `Modified`.
    /// Pairs below this floor are not paired at all.
    pub modify_threshold: f32,
    /// When true, post-classify high-scoring matches whose normalized index
    /// distance exceeds `move_distance_ratio` as `Moved` instead of `Unchanged`.
    pub detect_moves: bool,
    /// Threshold on `|i - j| / max(len_a, len_b)` for the move post-pass.
    pub move_distance_ratio: f32,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            granularity: Granularity::Sentence,
            algorithm: "token-jaccard".to_string(),
            metric: MetricConfig {
                lowercase: true,
                ..MetricConfig::default()
            },
            match_threshold: 0.85,
            modify_threshold: 0.4,
            detect_moves: false,
            move_distance_ratio: 0.1,
        }
    }
}

/// Compute a segment-level diff of `a` vs `b`. See module docs for the
/// algorithm overview and `DiffConfig` for the knobs.
pub fn diff_documents(
    a: &str,
    b: &str,
    cfg: &DiffConfig,
    tokenizer: Option<&PunktSentenceTokenizer>,
) -> Result<Vec<SegmentChange>, String> {
    // Validate the algorithm/parameters once before doing any work.
    compute_similarity(&cfg.algorithm, "", "", &cfg.metric)?;

    let segs_a = segment(a, cfg.granularity, tokenizer);
    let segs_b = segment(b, cfg.granularity, tokenizer);

    if segs_a.is_empty() && segs_b.is_empty() {
        return Ok(Vec::new());
    }
    if segs_a.is_empty() {
        return Ok(segs_b
            .iter()
            .enumerate()
            .map(|(j, s)| SegmentChange {
                kind: ChangeKind::Added,
                left: None,
                right: Some(seg_ref(s, j)),
                score: 0.0,
            })
            .collect());
    }
    if segs_b.is_empty() {
        return Ok(segs_a
            .iter()
            .enumerate()
            .map(|(i, s)| SegmentChange {
                kind: ChangeKind::Removed,
                left: Some(seg_ref(s, i)),
                right: None,
                score: 0.0,
            })
            .collect());
    }

    let texts_a: Vec<&str> = segs_a.iter().map(|s| s.text(a)).collect();
    let texts_b: Vec<&str> = segs_b.iter().map(|s| s.text(b)).collect();

    // Pairwise scoring matrix. Parallelize over rows (left segments) so
    // workers don't fight for `texts_b` in cache-unfriendly ways. Errors
    // from the dispatch layer become 0.0 — they survive only as below-
    // threshold "no match" scores.
    let scores: Vec<Vec<f32>> = (0..texts_a.len())
        .into_par_iter()
        .map(|i| {
            (0..texts_b.len())
                .map(|j| {
                    compute_similarity(&cfg.algorithm, texts_a[i], texts_b[j], &cfg.metric)
                        .map(|s| s as f32)
                        .unwrap_or(0.0)
                })
                .collect()
        })
        .collect();

    // Collect all (i, j, score) triples that clear the modify floor, then
    // greedily claim the highest-scoring still-unclaimed pairs.
    let mut pairs: Vec<(usize, usize, f32)> =
        Vec::with_capacity(texts_a.len().saturating_mul(texts_b.len()));
    for (i, row) in scores.iter().enumerate() {
        for (j, &s) in row.iter().enumerate() {
            if s >= cfg.modify_threshold {
                pairs.push((i, j, s));
            }
        }
    }
    pairs.sort_by(|x, y| {
        // Highest score first; tiebreak on (i, j) for determinism.
        y.2.partial_cmp(&x.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.0.cmp(&y.0))
            .then(x.1.cmp(&y.1))
    });

    let mut left_taken = vec![false; texts_a.len()];
    let mut right_taken = vec![false; texts_b.len()];
    let mut matched: Vec<(usize, usize, f32)> =
        Vec::with_capacity(texts_a.len().min(texts_b.len()));
    for (i, j, s) in pairs {
        if !left_taken[i] && !right_taken[j] {
            left_taken[i] = true;
            right_taken[j] = true;
            matched.push((i, j, s));
        }
    }

    let max_len = texts_a.len().max(texts_b.len()).max(1) as f32;
    let mut changes: Vec<SegmentChange> = Vec::with_capacity(
        matched.len() + (texts_a.len() - matched.len()) + (texts_b.len() - matched.len()),
    );

    for &(i, j, s) in &matched {
        let kind = if s >= cfg.match_threshold {
            if cfg.detect_moves {
                let dist = ((i as f32) - (j as f32)).abs() / max_len;
                if dist > cfg.move_distance_ratio {
                    ChangeKind::Moved
                } else {
                    ChangeKind::Unchanged
                }
            } else {
                ChangeKind::Unchanged
            }
        } else {
            ChangeKind::Modified
        };
        changes.push(SegmentChange {
            kind,
            left: Some(seg_ref(&segs_a[i], i)),
            right: Some(seg_ref(&segs_b[j], j)),
            score: s,
        });
    }
    for (i, seg) in segs_a.iter().enumerate() {
        if !left_taken[i] {
            changes.push(SegmentChange {
                kind: ChangeKind::Removed,
                left: Some(seg_ref(seg, i)),
                right: None,
                score: 0.0,
            });
        }
    }
    for (j, seg) in segs_b.iter().enumerate() {
        if !right_taken[j] {
            changes.push(SegmentChange {
                kind: ChangeKind::Added,
                left: None,
                right: Some(seg_ref(seg, j)),
                score: 0.0,
            });
        }
    }

    // Sort for stable, reader-friendly output: by left index when
    // present (so the diff reads in order of doc A), then by right
    // index for `Added` rows.
    changes.sort_by_key(|c| {
        let l = c.left.map(|r| r.index).unwrap_or(u32::MAX);
        let r = c.right.map(|r| r.index).unwrap_or(u32::MAX);
        (l, r)
    });

    Ok(changes)
}

fn seg_ref(s: &Segment, index: usize) -> SegmentRef {
    SegmentRef {
        index: index as u32,
        start: s.start as u32,
        end: s.end as u32,
    }
}

fn segment(text: &str, gran: Granularity, tok: Option<&PunktSentenceTokenizer>) -> Vec<Segment> {
    match gran {
        Granularity::Sentence => {
            // When no tokenizer is supplied, fall back to the bundled
            // default model (`DEFAULT_PUNKT_BYTES`) rather than the empty
            // model — the empty parameters effectively never split, so
            // sentence-level diffs would degenerate into one big chunk.
            let owned;
            let t = match tok {
                Some(t) => t,
                None => {
                    owned = default_tokenizer();
                    &owned
                }
            };
            segment_sentences(text, t)
        }
        Granularity::Paragraph => {
            let owned;
            let t = match tok {
                Some(t) => t,
                None => {
                    owned = default_tokenizer();
                    &owned
                }
            };
            segment_paragraphs(text, t)
        }
        Granularity::Line => segment_lines(text),
        Granularity::ParagraphSimple => segment_paragraphs_simple(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(granularity: Granularity, detect_moves: bool) -> DiffConfig {
        DiffConfig {
            granularity,
            detect_moves,
            ..DiffConfig::default()
        }
    }

    fn count_kind(changes: &[SegmentChange], kind: ChangeKind) -> usize {
        changes.iter().filter(|c| c.kind == kind).count()
    }

    #[test]
    fn identical_documents_all_unchanged() {
        let text = "The cat sat on the mat. It was very fluffy. Then it fell asleep.";
        let out = diff_documents(text, text, &cfg(Granularity::Sentence, false), None).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().all(|c| c.kind == ChangeKind::Unchanged));
        // Every change has both sides populated.
        assert!(out.iter().all(|c| c.left.is_some() && c.right.is_some()));
    }

    #[test]
    fn entirely_disjoint_documents() {
        let a = "Alpha bravo charlie. Delta echo foxtrot.";
        let b = "Lorem ipsum dolor. Sit amet consectetur.";
        let out = diff_documents(a, b, &cfg(Granularity::Sentence, false), None).unwrap();
        // Every segment in a is Removed; every segment in b is Added.
        assert!(count_kind(&out, ChangeKind::Removed) >= 2);
        assert!(count_kind(&out, ChangeKind::Added) >= 2);
        assert_eq!(count_kind(&out, ChangeKind::Unchanged), 0);
        assert_eq!(count_kind(&out, ChangeKind::Modified), 0);
    }

    #[test]
    fn empty_left_means_all_added() {
        let out = diff_documents(
            "",
            "New content here.",
            &cfg(Granularity::Sentence, false),
            None,
        )
        .unwrap();
        assert!(out.iter().all(|c| c.kind == ChangeKind::Added));
        assert!(out.iter().all(|c| c.left.is_none() && c.right.is_some()));
    }

    #[test]
    fn empty_right_means_all_removed() {
        let out = diff_documents(
            "Old content here.",
            "",
            &cfg(Granularity::Sentence, false),
            None,
        )
        .unwrap();
        assert!(out.iter().all(|c| c.kind == ChangeKind::Removed));
        assert!(out.iter().all(|c| c.left.is_some() && c.right.is_none()));
    }

    #[test]
    fn both_empty_returns_empty() {
        let out = diff_documents("", "", &cfg(Granularity::Sentence, false), None).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn modified_sentence_classified_as_modified() {
        let a = "The cat sat on the mat. It was very fluffy. Then it fell asleep.";
        let b = "The dog sat on the rug. It was very fluffy. Then it fell asleep.";
        let out = diff_documents(a, b, &cfg(Granularity::Sentence, false), None).unwrap();
        let modified = count_kind(&out, ChangeKind::Modified);
        let unchanged = count_kind(&out, ChangeKind::Unchanged);
        assert!(unchanged >= 2, "expected 2 unchanged, got {unchanged}");
        assert_eq!(modified, 1, "expected 1 modified, got {modified}");
    }

    #[test]
    fn appended_sentence_classified_as_added() {
        let a = "Sentence one. Sentence two.";
        let b = "Sentence one. Sentence two. Sentence three.";
        let out = diff_documents(a, b, &cfg(Granularity::Sentence, false), None).unwrap();
        assert_eq!(count_kind(&out, ChangeKind::Added), 1);
        assert!(count_kind(&out, ChangeKind::Unchanged) >= 2);
    }

    #[test]
    fn deleted_sentence_classified_as_removed() {
        let a = "Sentence one. Sentence two. Sentence three.";
        let b = "Sentence one. Sentence three.";
        let out = diff_documents(a, b, &cfg(Granularity::Sentence, false), None).unwrap();
        assert_eq!(count_kind(&out, ChangeKind::Removed), 1);
    }

    #[test]
    fn detect_moves_relabels_swapped_lines() {
        // Use Line granularity so this test doesn't depend on a trained
        // Punkt model — `PunktSentenceTokenizer::new()` without parameters
        // emits the whole text as one chunk.
        let a = "alpha bravo charlie delta echo foxtrot golf hotel\ngolf hotel india juliet kilo lima mike november oscar\nmike november oscar papa quebec romeo sierra tango\nsierra tango uniform victor whiskey xray yankee zulu";
        // Move the first line to the end; every line is preserved verbatim.
        let b = "golf hotel india juliet kilo lima mike november oscar\nmike november oscar papa quebec romeo sierra tango\nsierra tango uniform victor whiskey xray yankee zulu\nalpha bravo charlie delta echo foxtrot golf hotel";
        let cfg_no_moves = cfg(Granularity::Line, false);
        let no_moves = diff_documents(a, b, &cfg_no_moves, None).unwrap();
        assert_eq!(count_kind(&no_moves, ChangeKind::Moved), 0);
        let cfg_with_moves = DiffConfig {
            detect_moves: true,
            move_distance_ratio: 0.1,
            ..cfg_no_moves
        };
        let with_moves = diff_documents(a, b, &cfg_with_moves, None).unwrap();
        assert!(
            count_kind(&with_moves, ChangeKind::Moved) >= 1,
            "expected at least one Moved line, got {:?}",
            with_moves
                .iter()
                .map(|c| c.kind.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn line_granularity_works_without_punkt() {
        let a = "alpha\nbravo\ncharlie";
        let b = "alpha\ndelta\ncharlie";
        let out = diff_documents(a, b, &cfg(Granularity::Line, false), None).unwrap();
        assert!(count_kind(&out, ChangeKind::Unchanged) >= 2); // alpha, charlie
                                                               // bravo -> delta is a removed/added pair (low similarity for tiny tokens)
        assert!(count_kind(&out, ChangeKind::Removed) + count_kind(&out, ChangeKind::Added) >= 2);
    }

    #[test]
    fn paragraph_simple_granularity() {
        let a = "First paragraph here.\n\nSecond paragraph here.";
        let b = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let out = diff_documents(a, b, &cfg(Granularity::ParagraphSimple, false), None).unwrap();
        assert_eq!(count_kind(&out, ChangeKind::Added), 1);
    }

    #[test]
    fn unknown_algorithm_errors() {
        let bad_cfg = DiffConfig {
            algorithm: "bogus-metric".to_string(),
            ..DiffConfig::default()
        };
        let err = diff_documents("a.", "b.", &bad_cfg, None).unwrap_err();
        assert!(err.contains("bogus-metric"));
    }

    #[test]
    fn change_kind_string_names_match_python_contract() {
        // The Python wrapper exposes `kind` as a string literal; if the names
        // ever drift, both sides need updates. Lock them down in a test.
        assert_eq!(ChangeKind::Unchanged.as_str(), "unchanged");
        assert_eq!(ChangeKind::Modified.as_str(), "modified");
        assert_eq!(ChangeKind::Moved.as_str(), "moved");
        assert_eq!(ChangeKind::Added.as_str(), "added");
        assert_eq!(ChangeKind::Removed.as_str(), "removed");
    }

    /// Sentence-granularity diff using the trained Punkt model embedded
    /// at build time (`DEFAULT_PUNKT_BYTES`). When the caller passes
    /// `tokenizer = None`, the diff core implicitly loads this same model.
    #[test]
    fn sentence_diff_with_default_embedded_punkt() {
        // Three sentences: identical, edited, identical.
        let a = "The Lessor hereby agrees to lease the property. Rent shall be paid monthly. The lease term is one year.";
        let b = "The Lessor hereby agrees to lease the property. Rent shall be paid quarterly. The lease term is one year.";

        let out = diff_documents(a, b, &cfg(Granularity::Sentence, false), None).unwrap();

        let unchanged = count_kind(&out, ChangeKind::Unchanged);
        let modified = count_kind(&out, ChangeKind::Modified);
        assert_eq!(
            unchanged,
            2,
            "expected 2 unchanged sentences, kinds={:?}",
            out.iter().map(|c| c.kind.as_str()).collect::<Vec<_>>()
        );
        assert_eq!(
            modified,
            1,
            "expected 1 modified sentence, kinds={:?}",
            out.iter().map(|c| c.kind.as_str()).collect::<Vec<_>>()
        );
    }
}
