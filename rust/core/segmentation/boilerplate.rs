//! Boilerplate detector — finds repeated headers, footers, page numbers, and
//! captions in a `LineRecord` stream. Pure orchestrator; the heavy lifting is
//! delegated to the existing in-tree primitives:
//!
//! - [`crate::core::segmentation::extract_line_records`] supplies the input.
//! - [`crate::core::segmentation::normalize`] supplies canonical text per line.
//! - [`crate::core::minhash`] supplies the near-duplicate detection on OCR-drift.
//! - `ahash` (already a dep) supplies the exact-fingerprint hashing.
//!
//! Three-stage algorithm:
//!
//! 1. **Bucket by position.** Use `\f` (U+000C) when present as the page
//!    boundary; fall back to a `lines_per_page`-line window. Inside each page
//!    bucket, the top-`header_zone_lines` and bottom-`footer_zone_lines` are
//!    *zone* candidates; everything else is *body*.
//! 2. **Page-number sequence detection (pre-clustering).** Real page numbers
//!    are *monotonically increasing* values appearing in a fixed zone across
//!    consecutive pages. We detect them as a sequence rather than as a
//!    repeated string: this distinguishes pages numbered `1, 2, 3, ...` from
//!    section references, footnote markers, or sentinel cells in numeric
//!    tables — all of which are pure-digit-shaped but **not** monotonic.
//!    Lines participating in a detected sequence are marked consumed and
//!    skipped by stage 3.
//! 3. **Cluster the remaining lines by exact ahash u64 fingerprint** of the
//!    normalized line. Any cluster passing `(min_occurrences, min_rate)`
//!    becomes a `BoilerplateRun`. Then a residual **MinHash near-duplicate
//!    pass** for OCR drift over character 4-grams clusters survivors using
//!    `MinHashIndex::query_above_threshold` at 0.75 (one OCR typo on a
//!    30-char line drops Jaccard to ~0.85, so we keep headroom).
//! 4. **Classify** each surviving cluster into a `BoilerplateKind`
//!    (`Caption` / `Header` / `Footer` / `Unknown`) by zone dominance and
//!    canonical-shape rules. `PageNumber` is emitted only by stage 2 — the
//!    classifier never produces it.
//!
//! Design reference: `docs/SECTION_HEADING_PRIMITIVES_RESEARCH.md`
//! (`## Boilerplate detector (P5) — design reference`). Empirically validated
//! against real legal PDFs in `scripts/validate_boilerplate_pdf.py`; the
//! page-number-as-sequence rule (P5.6) replaced an earlier shape-based
//! classifier that misclassified docket numbers, footnote markers, and
//! sentinel cells in numeric tables.

use ahash::{AHashMap, AHashSet};
use serde::{Deserialize, Serialize};

use crate::core::minhash::{MinHashIndex, MinHashSignature, MinHasher};
use crate::core::segmentation::{
    enumerators::{parse_enumerator, parse_enumerator_with, WordLexicon},
    line_record::{LineRecord, LineTerminator},
    normalize::{normalize, NormalizeOptions},
};

// ─── Public types ─────────────────────────────────────────────────────────

/// Classification of a detected boilerplate run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum BoilerplateKind {
    /// Page number — Roman, Arabic, or `Page N (of M)?` shape.
    PageNumber = 0,
    /// Figure / Table / Exhibit caption (recurring caption template).
    Caption = 1,
    /// Top-of-page header (≥ 70 % of occurrences land in the top zone).
    Header = 2,
    /// Bottom-of-page footer (≥ 70 % in the bottom zone).
    Footer = 3,
    /// Recurrence threshold passed but no specific shape rule applies.
    Unknown = 4,
}

/// One detected run of repeated lines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoilerplateRun {
    /// Indices into the input `&[LineRecord]` slice (one per occurrence).
    pub line_indices: Vec<u32>,
    /// Normalized canonical form of the recurring line.
    pub canonical_text: String,
    /// Number of occurrences (`= line_indices.len()`).
    pub occurrences: u32,
    /// Cluster fingerprint. For exact-dup clusters this is the ahash of the
    /// canonical text; for MinHash clusters it is the ahash of the seed
    /// occurrence (so two runs always have distinct fingerprints).
    pub fingerprint: u64,
    /// Classification per Q4 of the design reference.
    pub kind: BoilerplateKind,
    /// For captions only: which Western-language lexicon matched the
    /// caption prefix. One of `"english"`, `"german"`, `"french"`,
    /// `"spanish"`, `"italian"`, `"portuguese"`, or `None` for non-caption
    /// runs. Per the P7.0f generality contract.
    #[serde(default)]
    pub language_hint: Option<String>,
}

/// Detector configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoilerplateOptions {
    /// Lines per fall-back page when no `\f` (U+000C) is present in input.
    /// 50 is the documented default; configurable for unusual layouts.
    pub lines_per_page: u32,
    /// Top-of-page zone size in lines. 3 is the default; tighter than
    /// "top 10 %" and matches empirical legal-PDF layouts.
    pub header_zone_lines: u32,
    /// Bottom-of-page zone size in lines.
    pub footer_zone_lines: u32,
    /// Minimum absolute occurrences to accept a cluster.
    pub min_occurrences: u32,
    /// Minimum fraction of *eligible* page buckets that must contain the
    /// cluster line. 0.5 means "appears in at least half of the pages where
    /// such a position exists at all".
    pub min_rate: f64,
    /// Skip the MinHash near-duplicate residual pass entirely. Default `false`.
    pub skip_near_dup: bool,
    /// MinHash threshold for the residual pass. 0.75 is one-typo-tolerant
    /// for ~30-character lines.
    pub near_dup_threshold: f64,
    /// MinHash permutation count. 64 fits the 20–80-char header workload
    /// (cardinality ≤ ~80 for 4-gram shingles).
    pub num_perm: usize,
    /// Character shingle size for MinHash. 4 is language-independent and
    /// resilient to OCR character drift.
    pub shingle_size: usize,
    /// Zone-dominance threshold for the Header/Footer classification step.
    /// 0.7 means "≥ 70 % of cluster occurrences are in the top (or bottom)
    /// zone" — handles cover-page logo offsets without insisting on 100 %.
    pub zone_dominance: f64,
    /// Drop boilerplate runs whose canonical line is empty / whitespace-only.
    /// Default `true` — recurring blank lines are rarely interesting.
    pub drop_empty: bool,
}

impl Default for BoilerplateOptions {
    fn default() -> Self {
        Self {
            lines_per_page: 50,
            header_zone_lines: 3,
            footer_zone_lines: 3,
            min_occurrences: 3,
            min_rate: 0.5,
            skip_near_dup: false,
            near_dup_threshold: 0.75,
            num_perm: 64,
            shingle_size: 4,
            zone_dominance: 0.7,
            drop_empty: true,
        }
    }
}

// ─── Internal types ───────────────────────────────────────────────────────

/// Page number a line falls in, plus position-within-page in lines from top.
#[derive(Debug, Clone, Copy)]
struct PagePos {
    page: u32,
    line_in_page: u32,
    /// Number of lines on this page (so we can compute "is this in the
    /// bottom-N zone?" without re-scanning).
    page_total: u32,
}

/// One candidate page-number occurrence: a line whose normalized canonical
/// parses as a digit or Roman value AND falls in a top/bottom zone.
#[derive(Debug, Clone, Copy)]
struct PageNumberCandidate {
    page: u32,
    line_idx: u32,
    value: u32,
    is_roman: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageZone {
    Top,
    Bottom,
}

/// Module-level seed for the MinHasher. Constant so two runs of the detector
/// over the same input produce identical signatures (Q6 determinism).
const BOILERPLATE_SEED: u64 = 0x006B_616F_736E_6C70_u64; // "kaosnlp"

// ─── Entry point ──────────────────────────────────────────────────────────

/// Detect repeated boilerplate runs in `records`.
///
/// `source` must be the same `&str` that produced `records` via
/// `extract_line_records` — record offsets index into it.
pub fn detect_boilerplate(
    records: &[LineRecord],
    source: &str,
    opts: BoilerplateOptions,
) -> Vec<BoilerplateRun> {
    detect_boilerplate_with(records, source, opts, &WordLexicon::EnglishLegalUs)
}

/// Same as [`detect_boilerplate`] but takes a caller-supplied
/// [`WordLexicon`] used by F-R8's `is_pure_enumerator` filter so
/// language-specific enumerator shapes (`Article 5`, `Artikel 12`,
/// `Capítulo 3`) are correctly recognized as headings instead of
/// boilerplate when the source is non-English.
pub fn detect_boilerplate_with(
    records: &[LineRecord],
    source: &str,
    opts: BoilerplateOptions,
    enumerator_lexicon: &WordLexicon,
) -> Vec<BoilerplateRun> {
    if records.is_empty() {
        return Vec::new();
    }

    // 1. Bucket each line into (page, line_in_page) coordinates.
    let positions = bucket_positions(records, &opts);
    debug_assert_eq!(positions.len(), records.len());

    // 2. Normalize each non-blank line and ahash-fingerprint it. Skip blank
    //    records when `drop_empty` is on.
    let canonical_opts = NormalizeOptions {
        collapse_whitespace: true,
        fold_case: true,
        normalize_unicode_punct: true,
        strip_enumerator_prefix: false,
        strip_punctuation: false,
    };

    let mut canonicals: Vec<Option<(u64, String)>> = Vec::with_capacity(records.len());
    for r in records {
        if opts.drop_empty && r.blank {
            canonicals.push(None);
            continue;
        }
        let stripped = r.stripped_text(source);
        if opts.drop_empty && stripped.is_empty() {
            canonicals.push(None);
            continue;
        }
        // Normalize the *stripped* text, not the raw line — we want
        // "Section 5." and "  Section 5.  " to share a fingerprint.
        match normalize(stripped, canonical_opts) {
            Ok(n) => {
                let canonical = n.text.into_owned();
                if canonical.trim().is_empty() && opts.drop_empty {
                    canonicals.push(None);
                } else {
                    let h = ahash_u64(&canonical);
                    canonicals.push(Some((h, canonical)));
                }
            }
            // The only error is the deferred enumerator option; we never set
            // it here, so this arm is unreachable in practice.
            Err(_) => canonicals.push(None),
        }
    }

    // 2.5. Page-number sequence detection (pre-clustering, P5.6).
    //
    // Real page numbers are monotonically increasing across pages — so we
    // detect them as a *sequence*, not as repeated strings. This separates
    // them from recurring digit-shaped content (section references, footnote
    // markers, sentinel cells in numeric tables) that share canonical-text
    // properties but never form an across-page monotonic run.
    //
    // Each line participating in a detected sequence is recorded in
    // `page_number_consumed` so step 3 below skips it.
    let (page_number_runs, page_number_consumed) =
        detect_page_number_sequences(&canonicals, &positions, &opts);
    let mut runs: Vec<BoilerplateRun> = page_number_runs;

    // 3. Group lines by exact ahash fingerprint.
    let mut by_fp: AHashMap<u64, Vec<u32>> = AHashMap::new();
    for (i, slot) in canonicals.iter().enumerate() {
        if page_number_consumed.contains(&(i as u32)) {
            continue;
        }
        if let Some((fp, _)) = slot {
            by_fp.entry(*fp).or_default().push(i as u32);
        }
    }

    // 4. Per cluster, compute eligibility / occurrence rate. Eligible buckets
    //    = number of distinct (page, line_in_page) coords that exist *at all*
    //    in the document with the same line_in_page as the cluster's lines.
    //    This is the "denominator must be eligible buckets" rule.
    //
    // `already_grouped` is seeded with every line consumed by the page-number
    // sequence pass so neither MinHash nor classification can re-cluster them.
    let mut already_grouped: AHashSet<u32> = page_number_consumed.clone();

    for (fp, indices) in by_fp.iter() {
        if (indices.len() as u32) < opts.min_occurrences {
            continue;
        }
        // Accept the cluster if (a) it passes min_occurrences and (b) the
        // rate over eligible pages is ≥ min_rate. The rate is approximated by
        // (occurrences / number of distinct pages the cluster appears on);
        // we don't insist on identical line_in_page across occurrences (real
        // headers shift down by one line on the cover page, etc.).
        let distinct_pages: AHashSet<u32> = indices
            .iter()
            .map(|&i| positions[i as usize].page)
            .collect();
        let total_pages = positions.iter().map(|p| p.page).max().unwrap_or(0) + 1;
        let rate = distinct_pages.len() as f64 / total_pages as f64;
        if rate < opts.min_rate {
            continue;
        }
        // Accept: emit run.
        let canonical = canonicals[indices[0] as usize]
            .as_ref()
            .map(|(_, c)| c.clone())
            .unwrap_or_default();
        // F-R8: a canonical that parses as a single enumerator (e.g.
        // "1.1", "271.", "(a)") is structurally a heading anchor or
        // list marker, not boilerplate. Skip the cluster so the
        // structure layer can label it correctly.
        if is_pure_enumerator(&canonical, enumerator_lexicon) {
            continue;
        }
        let cls = classify(&canonical, indices, &positions, &opts);
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_unstable();
        for &i in &sorted_indices {
            already_grouped.insert(i);
        }
        runs.push(BoilerplateRun {
            occurrences: sorted_indices.len() as u32,
            line_indices: sorted_indices,
            canonical_text: canonical,
            fingerprint: *fp,
            kind: cls.kind,
            language_hint: cls.language_hint,
        });
    }

    // 5. Residual MinHash pass for OCR drift. Cheap to skip, valuable when on.
    if !opts.skip_near_dup {
        let unmatched: Vec<u32> = canonicals
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                if slot.is_some() && !already_grouped.contains(&(i as u32)) {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .collect();

        if (unmatched.len() as u32) >= opts.min_occurrences {
            let near = minhash_cluster(&unmatched, &canonicals, &opts);
            for cluster in near {
                if (cluster.len() as u32) < opts.min_occurrences {
                    continue;
                }
                let distinct_pages: AHashSet<u32> = cluster
                    .iter()
                    .map(|&i| positions[i as usize].page)
                    .collect();
                let total_pages = positions.iter().map(|p| p.page).max().unwrap_or(0) + 1;
                let rate = distinct_pages.len() as f64 / total_pages as f64;
                if rate < opts.min_rate {
                    continue;
                }
                // Use the seed line's canonical as the run's representative;
                // its ahash becomes the fingerprint so distinct runs cannot
                // share it (Q6 invariant 3).
                let seed = cluster[0];
                let (seed_fp, seed_canonical) = canonicals[seed as usize]
                    .clone()
                    .unwrap_or((0, String::new()));
                // F-R8: skip pure-enumerator canonicals from the
                // near-dup pass too.
                if is_pure_enumerator(&seed_canonical, enumerator_lexicon) {
                    continue;
                }
                let cls = classify(&seed_canonical, &cluster, &positions, &opts);
                let mut sorted_cluster = cluster;
                sorted_cluster.sort_unstable();
                runs.push(BoilerplateRun {
                    occurrences: sorted_cluster.len() as u32,
                    line_indices: sorted_cluster,
                    canonical_text: seed_canonical,
                    fingerprint: seed_fp,
                    kind: cls.kind,
                    language_hint: cls.language_hint,
                });
            }
        }
    }

    // 6. Determinism (Q6 invariant 1): sort the output by fingerprint. ahash
    //    iteration order is non-deterministic; we want byte-identical output
    //    on identical input.
    runs.sort_by(|a, b| {
        a.fingerprint
            .cmp(&b.fingerprint)
            .then_with(|| a.line_indices.cmp(&b.line_indices))
    });
    runs
}

// ─── Position bucketing ───────────────────────────────────────────────────

/// F-R8: returns `true` iff `text` (canonical form) is structurally a
/// single enumerator (`1.`, `1.1.2`, `(a)`, `271.`, `Section 5`,
/// `Article 5`, `Artikel 12`, `§ 552`, `# Heading`, `- bullet text`).
/// Such lines are list / heading anchors, not boilerplate, even when
/// they recur across pages.
///
/// `lexicon` controls which word-prefix lexicon is consulted. For
/// English-default callers this stays as `WordLexicon::EnglishLegalUs`;
/// non-English callers should pass the matching variant so language-
/// specific section keywords are recognized.
fn is_pure_enumerator(text: &str, lexicon: &WordLexicon) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let Some(e) = parse_enumerator_with(trimmed, lexicon) else {
        return false;
    };
    let consumed = e.prefix_end as usize;
    consumed >= trimmed.len()
}

/// Suppress "unused import" warning for `parse_enumerator` — this name
/// remains available to other callers that don't need a lexicon.
#[allow(dead_code)]
fn _keep_parse_enumerator_alive(s: &str) -> Option<crate::core::segmentation::Enumerator> {
    parse_enumerator(s)
}

fn bucket_positions(records: &[LineRecord], opts: &BoilerplateOptions) -> Vec<PagePos> {
    // Detect whether `\f` appears in the input. LineRecord classifies the
    // U+000C form-feed under `LineTerminator::OtherUnicode` with `term_len == 1`.
    let has_form_feed = records
        .iter()
        .any(|r| r.terminator == LineTerminator::OtherUnicode && r.term_len == 1);

    if has_form_feed {
        bucket_by_form_feed(records)
    } else {
        bucket_by_window(records, opts.lines_per_page)
    }
}

fn bucket_by_form_feed(records: &[LineRecord]) -> Vec<PagePos> {
    let n = records.len();
    let mut positions = Vec::with_capacity(n);
    let mut page = 0u32;
    let mut line_in_page = 0u32;
    let mut page_starts = vec![0u32]; // start-line of each page
    for r in records {
        positions.push(PagePos {
            page,
            line_in_page,
            page_total: 0, // back-fill below
        });
        if r.terminator == LineTerminator::OtherUnicode && r.term_len == 1 {
            page += 1;
            page_starts.push(positions.len() as u32);
            line_in_page = 0;
        } else {
            line_in_page += 1;
        }
    }
    page_starts.push(n as u32);
    // Back-fill page_total: page_total[p] = page_starts[p+1] - page_starts[p].
    for (i, p) in positions.iter_mut().enumerate() {
        let pg = p.page as usize;
        let total = page_starts.get(pg + 1).copied().unwrap_or(n as u32)
            - page_starts.get(pg).copied().unwrap_or(0);
        p.page_total = total;
        debug_assert!(i < n);
    }
    positions
}

fn bucket_by_window(records: &[LineRecord], lines_per_page: u32) -> Vec<PagePos> {
    let lpp = lines_per_page.max(1);
    let n = records.len() as u32;
    let mut positions = Vec::with_capacity(records.len());
    for i in 0..records.len() {
        let global = i as u32;
        let page = global / lpp;
        let line_in_page = global % lpp;
        // Last (possibly partial) page total may be shorter.
        let start = page * lpp;
        let end = (start + lpp).min(n);
        let page_total = end - start;
        positions.push(PagePos {
            page,
            line_in_page,
            page_total,
        });
    }
    positions
}

// ─── MinHash near-dup pass ────────────────────────────────────────────────

fn minhash_cluster(
    unmatched: &[u32],
    canonicals: &[Option<(u64, String)>],
    opts: &BoilerplateOptions,
) -> Vec<Vec<u32>> {
    if unmatched.is_empty() {
        return Vec::new();
    }
    let hasher = MinHasher::with_seed(opts.num_perm, BOILERPLATE_SEED);
    let mut index = MinHashIndex::with_threshold(opts.num_perm, opts.near_dup_threshold);

    // Compute every signature first so we can iterate in deterministic order.
    let mut sigs: Vec<(u32, MinHashSignature)> = Vec::with_capacity(unmatched.len());
    for &line_idx in unmatched {
        let canon = match &canonicals[line_idx as usize] {
            Some((_, s)) if !s.is_empty() => s.as_str(),
            _ => continue,
        };
        let sig = hasher.hash_char_shingles(canon, opts.shingle_size);
        // `insert` returns Result because of size validation; we constructed
        // hasher and index with the same num_perm, so this never errors.
        let _ = index.insert(line_idx, &sig);
        sigs.push((line_idx, sig));
    }

    // Greedy clustering: iterate signatures in input order; for each, ask the
    // index for above-threshold neighbours; group them; mark all as grouped
    // so we don't process them again.
    let mut clusters: Vec<Vec<u32>> = Vec::new();
    let mut grouped: AHashSet<u32> = AHashSet::new();
    for (line_idx, sig) in &sigs {
        if grouped.contains(line_idx) {
            continue;
        }
        let neighbours = index
            .query_above_threshold(sig, opts.near_dup_threshold)
            .unwrap_or_default();
        let mut cluster: Vec<u32> = neighbours
            .into_iter()
            .map(|(id, _sim)| id)
            .filter(|id| !grouped.contains(id))
            .collect();
        // The seed is always in its own neighbour list (jaccard = 1.0). If
        // the index excluded it for any reason, force-include.
        if !cluster.contains(line_idx) {
            cluster.push(*line_idx);
        }
        for &id in &cluster {
            grouped.insert(id);
        }
        if cluster.len() > 1 {
            clusters.push(cluster);
        }
    }
    clusters
}

// ─── Classifier ───────────────────────────────────────────────────────────

/// Result of classification: kind + (for captions) the matched-language tag.
struct Classification {
    kind: BoilerplateKind,
    language_hint: Option<String>,
}

fn classify(
    canonical: &str,
    indices: &[u32],
    positions: &[PagePos],
    opts: &BoilerplateOptions,
) -> Classification {
    // PageNumber is emitted only by `detect_page_number_sequences` (P5.6) — this
    // classifier never produces it. The sequence pass operates on monotonicity,
    // which is the property that distinguishes real page numbering from
    // recurring digit-shaped content (footnote markers, section refs, table
    // cells).

    // 1. Caption — try every Western-language lexicon; first match wins.
    if !in_zone(indices, positions, opts, Zone::Top)
        && !in_zone(indices, positions, opts, Zone::Bottom)
    {
        if let Some(lang) = match_caption_language(canonical) {
            return Classification {
                kind: BoilerplateKind::Caption,
                language_hint: Some(lang.to_string()),
            };
        }
    }

    // 2. Header — top-zone dominant + char_len ≤ 80.
    if canonical.chars().count() <= 80 && in_zone(indices, positions, opts, Zone::Top) {
        return Classification {
            kind: BoilerplateKind::Header,
            language_hint: None,
        };
    }

    // 3. Footer — bottom-zone dominant.
    if canonical.chars().count() <= 80 && in_zone(indices, positions, opts, Zone::Bottom) {
        return Classification {
            kind: BoilerplateKind::Footer,
            language_hint: None,
        };
    }

    Classification {
        kind: BoilerplateKind::Unknown,
        language_hint: None,
    }
}

#[derive(Debug, Clone, Copy)]
enum Zone {
    Top,
    Bottom,
}

fn in_zone(indices: &[u32], positions: &[PagePos], opts: &BoilerplateOptions, zone: Zone) -> bool {
    if indices.is_empty() {
        return false;
    }
    let in_zone_count = indices
        .iter()
        .filter(|&&i| {
            let p = &positions[i as usize];
            match zone {
                Zone::Top => p.line_in_page < opts.header_zone_lines,
                Zone::Bottom => {
                    let bottom_threshold = p.page_total.saturating_sub(opts.footer_zone_lines);
                    p.line_in_page >= bottom_threshold
                }
            }
        })
        .count();
    let dominance = in_zone_count as f64 / indices.len() as f64;
    dominance >= opts.zone_dominance
}

/// Caption-prefix lexicon entry: language tag + the prefix list (post-
/// normalize, lowercase + trailing space). Per P7.0f.
struct CaptionLexicon {
    language: &'static str,
    prefixes: &'static [&'static str],
}

const CAPTION_LEXICONS: &[CaptionLexicon] = &[
    CaptionLexicon {
        language: "english",
        prefixes: &[
            "figure ",
            "fig. ",
            "fig ",
            "table ",
            "exhibit ",
            "schedule ",
            "appendix ",
        ],
    },
    CaptionLexicon {
        language: "german",
        // Abbildung (figure), Tabelle (table), Anhang (appendix), Anlage
        // (exhibit), Schaubild (chart/figure).
        prefixes: &[
            "abbildung ",
            "abb. ",
            "abb ",
            "tabelle ",
            "tab. ",
            "tab ",
            "anhang ",
            "anlage ",
            "schaubild ",
        ],
    },
    CaptionLexicon {
        language: "french",
        // Figure, Tableau (table), Annexe (appendix/exhibit).
        prefixes: &["figure ", "fig. ", "fig ", "tableau ", "tab. ", "annexe "],
    },
    CaptionLexicon {
        language: "spanish",
        // Figura, Tabla, Cuadro (chart/table), Anexo (appendix), Apéndice.
        prefixes: &[
            "figura ",
            "fig. ",
            "fig ",
            "tabla ",
            "cuadro ",
            "anexo ",
            "apéndice ",
        ],
    },
    CaptionLexicon {
        language: "italian",
        // Figura, Tabella, Tavola, Allegato (appendix/exhibit), Appendice.
        prefixes: &[
            "figura ",
            "fig. ",
            "fig ",
            "tabella ",
            "tavola ",
            "allegato ",
            "appendice ",
        ],
    },
    CaptionLexicon {
        language: "portuguese",
        // Figura, Tabela, Quadro (chart), Anexo (appendix), Apêndice.
        prefixes: &[
            "figura ",
            "fig. ",
            "fig ",
            "tabela ",
            "quadro ",
            "anexo ",
            "apêndice ",
            "apendice ",
        ],
    },
];

/// Try each Western caption lexicon in order. Returns the first matching
/// language tag, or `None` if no caption prefix matches. Input is post-
/// normalize (lowercased + ws-collapsed).
fn match_caption_language(s: &str) -> Option<&'static str> {
    let lowered = s.trim_start();
    for lex in CAPTION_LEXICONS {
        if lex.prefixes.iter().any(|p| lowered.starts_with(p)) {
            return Some(lex.language);
        }
    }
    None
}

// ─── Page-number sequence detector (P5.6) ─────────────────────────────────
//
// Real page numbers are a *monotonically increasing sequence* across pages
// — that property is what distinguishes them from any other recurring
// digit-shaped content. This pass walks the line stream once, gathers
// every (page, value) candidate that could be a page number, then finds
// the longest monotonically increasing sub-sequence per (zone, kind) by
// page index. If the sub-sequence is at least `min_occurrences` long, all
// participating lines are emitted as one `BoilerplateRun` of kind
// `PageNumber` and marked consumed so the downstream clustering passes
// skip them.

/// Parse a normalized canonical line into a page-number value, if it could
/// plausibly be one. Returns `(value, is_roman)` on success.
///
/// Accepts:
/// * pure ASCII digits (`"5"`, `"42"`)
/// * pure lowercase Roman, length ≤ 8 (`"iv"`, `"xliii"`)
/// * `"page N"` / `"page N of M"` — extracts `N`
fn parse_page_number_value(s: &str) -> Option<(u32, bool)> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        return trimmed.parse::<u32>().ok().map(|n| (n, false));
    }
    if trimmed.len() <= 8
        && trimmed
            .chars()
            .all(|c| matches!(c, 'i' | 'v' | 'x' | 'l' | 'c' | 'd' | 'm'))
    {
        return roman_to_u32(trimmed).map(|n| (n, true));
    }
    if let Some(rest) = trimmed.strip_prefix("page ") {
        let n_str = match rest.find(" of ") {
            Some(idx) => &rest[..idx],
            None => rest,
        };
        return n_str.parse::<u32>().ok().map(|n| (n, false));
    }
    None
}

/// Convert lowercase Roman numerals to a u32. Strict: rejects malformed
/// patterns like `"iiii"` or `"vv"` by re-encoding the parsed value and
/// requiring round-trip equality (i.e., `s` must be the *canonical* Roman
/// form of its value).
pub(crate) fn roman_to_u32(s: &str) -> Option<u32> {
    fn val(c: char) -> Option<u32> {
        Some(match c {
            'i' => 1,
            'v' => 5,
            'x' => 10,
            'l' => 50,
            'c' => 100,
            'd' => 500,
            'm' => 1000,
            _ => return None,
        })
    }
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return None;
    }
    let mut total: u32 = 0;
    let mut i = 0;
    while i < chars.len() {
        let cur = val(chars[i])?;
        // Subtractive notation: if the next symbol is larger, this pair
        // contributes `nxt - cur` and we consume two symbols.
        if i + 1 < chars.len() {
            if let Some(nxt) = val(chars[i + 1]) {
                if cur < nxt {
                    total = total.checked_add(nxt - cur)?;
                    i += 2;
                    continue;
                }
            }
        }
        total = total.checked_add(cur)?;
        i += 1;
    }
    if total == 0 {
        return None;
    }
    // Reject malformed patterns (e.g. "iiii", "vv") by re-encoding.
    if u32_to_roman(total)? != s {
        return None;
    }
    Some(total)
}

fn u32_to_roman(mut n: u32) -> Option<String> {
    if n == 0 || n > 3999 {
        return None;
    }
    const PAIRS: &[(u32, &str)] = &[
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ];
    let mut out = String::new();
    for &(v, sym) in PAIRS {
        while n >= v {
            out.push_str(sym);
            n -= v;
        }
    }
    Some(out)
}

/// Detect page-number sequences. Returns the runs plus the set of
/// `line_indices` consumed (so downstream clustering passes can skip them).
fn detect_page_number_sequences(
    canonicals: &[Option<(u64, String)>],
    positions: &[PagePos],
    opts: &BoilerplateOptions,
) -> (Vec<BoilerplateRun>, AHashSet<u32>) {
    // Gather candidates per zone + kind (Roman vs Arabic).
    let mut top_arabic: Vec<PageNumberCandidate> = Vec::new();
    let mut top_roman: Vec<PageNumberCandidate> = Vec::new();
    let mut bot_arabic: Vec<PageNumberCandidate> = Vec::new();
    let mut bot_roman: Vec<PageNumberCandidate> = Vec::new();

    for (i, slot) in canonicals.iter().enumerate() {
        let Some((_, canon)) = slot else { continue };
        let pos = positions[i];
        let Some((value, is_roman)) = parse_page_number_value(canon) else {
            continue;
        };
        let cand = PageNumberCandidate {
            page: pos.page,
            line_idx: i as u32,
            value,
            is_roman,
        };
        let in_top = pos.line_in_page < opts.header_zone_lines;
        let in_bot = pos.line_in_page >= pos.page_total.saturating_sub(opts.footer_zone_lines);
        match (in_top, in_bot, is_roman) {
            (true, _, false) => top_arabic.push(cand),
            (true, _, true) => top_roman.push(cand),
            (_, true, false) => bot_arabic.push(cand),
            (_, true, true) => bot_roman.push(cand),
            _ => {} // Outside both zones — not a page-number candidate.
        }
    }

    let mut runs = Vec::new();
    let mut consumed: AHashSet<u32> = AHashSet::new();

    for (zone, kind, mut cands) in [
        (PageZone::Top, false, top_arabic),
        (PageZone::Top, true, top_roman),
        (PageZone::Bottom, false, bot_arabic),
        (PageZone::Bottom, true, bot_roman),
    ] {
        if (cands.len() as u32) < opts.min_occurrences {
            continue;
        }
        // Sort by page so we can scan left-to-right.
        cands.sort_by_key(|c| (c.page, c.value));
        let sequence = longest_monotonic_run(&cands, opts);
        if (sequence.len() as u32) < opts.min_occurrences {
            continue;
        }

        let canonical = if kind {
            "(roman page numbers)".to_string()
        } else {
            "(arabic page numbers)".to_string()
        };
        let mut indices: Vec<u32> = sequence.iter().map(|c| c.line_idx).collect();
        indices.sort_unstable();
        for &idx in &indices {
            consumed.insert(idx);
        }
        // Fingerprint a deterministic string so distinct page-number runs
        // (top/bottom × Arabic/Roman) collide-free.
        let fp_input = format!(
            "page-numbers/{}/{}/{}",
            match zone {
                PageZone::Top => "top",
                PageZone::Bottom => "bottom",
            },
            if kind { "roman" } else { "arabic" },
            indices.len()
        );
        let fingerprint = ahash_u64(&fp_input);
        runs.push(BoilerplateRun {
            occurrences: indices.len() as u32,
            line_indices: indices,
            canonical_text: canonical,
            fingerprint,
            kind: BoilerplateKind::PageNumber,
            language_hint: None,
        });
    }

    (runs, consumed)
}

/// Find the longest run of candidates whose `value` strictly increases as
/// `page` strictly increases. Each page contributes at most one candidate
/// (if multiple candidates per page exist, the one that extends the
/// current run wins; the others are discarded for this pass).
///
/// Greedy: walk pages in order; at each step pick the smallest candidate
/// whose value is strictly greater than the run's last value. Reset to a
/// fresh single-element run if no extension is found, and remember the
/// best run seen so far.
fn longest_monotonic_run<'a>(
    candidates: &'a [PageNumberCandidate],
    opts: &BoilerplateOptions,
) -> Vec<&'a PageNumberCandidate> {
    if candidates.is_empty() {
        return Vec::new();
    }
    // Group by page.
    let mut by_page: Vec<(u32, Vec<&PageNumberCandidate>)> = Vec::new();
    for c in candidates {
        if let Some(last) = by_page.last_mut() {
            if last.0 == c.page {
                last.1.push(c);
                continue;
            }
        }
        by_page.push((c.page, vec![c]));
    }

    let mut current: Vec<&PageNumberCandidate> = Vec::new();
    let mut best: Vec<&PageNumberCandidate> = Vec::new();
    let mut last_page: Option<u32> = None;
    let mut last_value: Option<u32> = None;

    for (page, page_cands) in &by_page {
        // Same-page extension would violate "strictly increasing page" — skip.
        if Some(*page) == last_page {
            continue;
        }
        // Pick the smallest candidate on this page whose value > last_value.
        let extender = page_cands
            .iter()
            .filter(|c| match last_value {
                Some(v) => c.value > v,
                None => true,
            })
            .min_by_key(|c| c.value);
        match extender {
            Some(c) => {
                current.push(*c);
                last_page = Some(*page);
                last_value = Some(c.value);
            }
            None => {
                if current.len() > best.len() {
                    best = current.clone();
                }
                // Restart with the smallest candidate on this page.
                if let Some(c) = page_cands.iter().min_by_key(|c| c.value) {
                    current = vec![*c];
                    last_page = Some(*page);
                    last_value = Some(c.value);
                }
            }
        }
    }
    if current.len() > best.len() {
        best = current;
    }
    if (best.len() as u32) < opts.min_occurrences {
        return Vec::new();
    }
    best
}

// ─── Tiny ahash helper ────────────────────────────────────────────────────

#[inline]
fn ahash_u64(s: &str) -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let state = ahash::RandomState::with_seeds(
        BOILERPLATE_SEED,
        BOILERPLATE_SEED.wrapping_add(1),
        BOILERPLATE_SEED.wrapping_add(2),
        BOILERPLATE_SEED.wrapping_add(3),
    );
    let mut h = state.build_hasher();
    h.write(s.as_bytes());
    h.finish()
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::segmentation::extract_line_records;
    use proptest::prelude::*;

    fn build_synthetic_pages(header: &str, footer: &str, body: &str, n_pages: u32) -> String {
        // 5-line page: header / body / body / body / footer, with form-feed
        // between pages so the detector picks up the boundary explicitly.
        let mut s = String::new();
        for i in 0..n_pages {
            s.push_str(header);
            s.push('\n');
            s.push_str(body);
            s.push('\n');
            s.push_str(body);
            s.push('\n');
            // Vary the body slightly so it is *not* boilerplate.
            s.push_str(&format!("{} (page {})", body, i + 1));
            s.push('\n');
            s.push_str(footer);
            s.push('\n');
            // Form-feed between pages, except the last.
            if i + 1 < n_pages {
                s.push('\u{000C}');
            }
        }
        s
    }

    #[test]
    fn empty_input_returns_empty() {
        let runs = detect_boilerplate(&[], "", BoilerplateOptions::default());
        assert!(runs.is_empty());
    }

    #[test]
    fn detects_repeated_header_with_form_feed() {
        let src = build_synthetic_pages("FILED 5/5/2026", "Page 1", "Body content", 5);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());

        assert!(!runs.is_empty(), "expected at least one boilerplate run");
        let header = runs
            .iter()
            .find(|r| r.canonical_text.contains("filed"))
            .expect("filed header missing");
        assert_eq!(header.kind, BoilerplateKind::Header);
        assert_eq!(header.occurrences, 5);
        // Header runs carry no language hint (only Caption runs do).
        assert!(header.language_hint.is_none());
    }

    /// Build pages where a recurring caption sits OUTSIDE the top/bottom
    /// zone (i.e., in the body) so the classifier emits a Caption.
    fn build_pages_with_caption(caption: &str, n_pages: u32) -> String {
        let mut s = String::new();
        for i in 0..n_pages {
            s.push_str("HEADER\n");
            s.push_str("Body line one\n");
            s.push_str("Body line two\n");
            s.push_str(caption);
            s.push('\n');
            s.push_str("Body line four\n");
            s.push_str(&format!("Body unique line {}\n", i + 1));
            s.push_str("FOOTER\n");
            if i + 1 < n_pages {
                s.push('\u{000C}');
            }
        }
        s
    }

    // ── Multi-language caption-prefix lexicons (P7.0f) ───────────────────

    #[test]
    fn detects_english_caption() {
        let src = build_pages_with_caption("Figure 5: Architecture diagram", 4);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let cap = runs
            .iter()
            .find(|r| r.kind == BoilerplateKind::Caption)
            .expect("expected English caption");
        assert_eq!(cap.language_hint.as_deref(), Some("english"));
    }

    #[test]
    fn detects_german_caption() {
        let src = build_pages_with_caption("Abbildung 3: Diagramm der Komponenten", 4);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let cap = runs
            .iter()
            .find(|r| r.kind == BoilerplateKind::Caption)
            .expect("expected German caption");
        assert_eq!(cap.language_hint.as_deref(), Some("german"));
    }

    #[test]
    fn detects_french_caption() {
        let src = build_pages_with_caption("Tableau 2 — Résumé des résultats", 4);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let cap = runs
            .iter()
            .find(|r| r.kind == BoilerplateKind::Caption)
            .expect("expected French caption");
        assert_eq!(cap.language_hint.as_deref(), Some("french"));
    }

    #[test]
    fn detects_spanish_caption() {
        let src = build_pages_with_caption("Tabla 4: Comparación de modelos", 4);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let cap = runs
            .iter()
            .find(|r| r.kind == BoilerplateKind::Caption)
            .expect("expected Spanish caption");
        assert_eq!(cap.language_hint.as_deref(), Some("spanish"));
    }

    #[test]
    fn detects_italian_caption() {
        let src = build_pages_with_caption("Tabella 1: Riepilogo dei dati", 4);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let cap = runs
            .iter()
            .find(|r| r.kind == BoilerplateKind::Caption)
            .expect("expected Italian caption");
        assert_eq!(cap.language_hint.as_deref(), Some("italian"));
    }

    #[test]
    fn detects_portuguese_caption() {
        let src = build_pages_with_caption("Quadro 2: Resumo das medidas", 4);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let cap = runs
            .iter()
            .find(|r| r.kind == BoilerplateKind::Caption)
            .expect("expected Portuguese caption");
        assert_eq!(cap.language_hint.as_deref(), Some("portuguese"));
    }

    #[test]
    fn detects_monotonic_page_number_sequence() {
        // Build pages with a real monotonic page-number footer (1..=5) — what
        // a normal printed document looks like.
        let mut src = String::new();
        for i in 1..=5u32 {
            src.push_str("HEADER\n");
            src.push_str("Body line one\n");
            src.push_str(&format!("Body line two on page {}\n", i));
            src.push_str("Body line three\n");
            src.push_str(&i.to_string());
            src.push('\n');
            if i < 5 {
                src.push('\u{000C}');
            }
        }
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let pn = runs
            .iter()
            .find(|r| r.kind == BoilerplateKind::PageNumber)
            .expect("expected a page-number sequence run");
        assert_eq!(pn.occurrences, 5);
        assert_eq!(pn.canonical_text, "(arabic page numbers)");
    }

    #[test]
    fn constant_digit_footer_is_not_page_number() {
        // P5.6: a footer like "1" repeated identically on every page is NOT
        // a page-number sequence (no monotonicity). It should be classified
        // as a Footer instead.
        let src = build_synthetic_pages("HEADER", "1", "Body", 5);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        // No PageNumber run.
        assert!(
            !runs.iter().any(|r| r.kind == BoilerplateKind::PageNumber),
            "constant '1' must NOT be flagged as PageNumber: got {:?}",
            runs.iter()
                .map(|r| (&r.canonical_text, r.kind))
                .collect::<Vec<_>>()
        );
        // The constant "1" cluster does still pass min_occurrences and shows
        // up as a Footer (it sits in the bottom zone).
        let footer = runs
            .iter()
            .find(|r| r.canonical_text == "1")
            .expect("expected constant '1' to surface as a Footer cluster");
        assert_eq!(footer.kind, BoilerplateKind::Footer);
    }

    #[test]
    fn footnote_markers_per_page_are_not_page_numbers() {
        // Each page has its own footnote marker sequence 1, 2, 3 (resets per
        // page). Across pages this is NOT monotonic — running 1, 2, 3, 1, 2,
        // 3, ... — so it must not be classified as a page-number sequence.
        // We put the markers at the bottom zone so they qualify as candidates.
        let mut src = String::new();
        for _page in 1..=4u32 {
            src.push_str("HEADER\n");
            src.push_str("Body...\n");
            // Three footnote markers in the bottom zone.
            src.push_str("1\n");
            src.push_str("2\n");
            src.push_str("3\n");
            src.push('\u{000C}');
        }
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        // No page-number sequence emerges.
        assert!(
            !runs.iter().any(|r| r.kind == BoilerplateKind::PageNumber),
            "per-page footnote markers must not form a page-number sequence: got {:?}",
            runs.iter()
                .map(|r| (&r.canonical_text, r.kind))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn page_number_sequence_with_gaps_still_detected() {
        // Pages 1..=5 with a real page number, but skip page 3 (e.g., a
        // full-page figure with no number). The monotonic run is still
        // 1, 2, 4, 5 — strictly increasing, length 4, ≥ min_occurrences.
        let mut src = String::new();
        for i in 1..=5u32 {
            src.push_str("HEADER\n");
            src.push_str("Body...\n");
            if i != 3 {
                src.push_str(&i.to_string());
                src.push('\n');
            } else {
                src.push_str("(no page number on this page)\n");
            }
            if i < 5 {
                src.push('\u{000C}');
            }
        }
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let pn = runs
            .iter()
            .find(|r| r.kind == BoilerplateKind::PageNumber)
            .expect("page-number sequence with one missing page should still detect");
        assert_eq!(pn.occurrences, 4);
    }

    #[test]
    fn roman_page_number_sequence_detected_separately() {
        // i, ii, iii, iv across the bottom of 4 pages.
        let mut src = String::new();
        for r in &["i", "ii", "iii", "iv"] {
            src.push_str("FRONT MATTER\n");
            src.push_str("Body...\n");
            src.push_str(r);
            src.push('\n');
            src.push('\u{000C}');
        }
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let pn = runs
            .iter()
            .find(|r| r.kind == BoilerplateKind::PageNumber && r.canonical_text.contains("roman"))
            .expect("Roman page-number sequence should be detected");
        assert_eq!(pn.occurrences, 4);
    }

    #[test]
    fn recurring_section_reference_is_not_page_number() {
        // P5.6 / P5.5 regression: a digit-shaped recurring header like
        // "8.8" or "15" appearing multiple times per document is a SECTION
        // REFERENCE, not a page number. (Empirical case from
        // tests/fixtures/edgar_agreements.jsonl record #0 — '8.8' × 63.)
        // Build 5 pages with the same digit-string at bottom every page.
        // It would have triggered the old `looks_like_page_number`; the new
        // sequence rule rejects because the values do not increase.
        let mut src = String::new();
        for i in 1..=5u32 {
            src.push_str("HEADER\n");
            src.push_str(&format!("Body of page {}\n", i));
            src.push_str("8.8\n"); // <- same recurring "section ref" each page
            src.push('\u{000C}');
        }
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        assert!(
            !runs.iter().any(|r| r.kind == BoilerplateKind::PageNumber),
            "recurring '8.8' must NOT be a PageNumber: got {:?}",
            runs.iter()
                .map(|r| (&r.canonical_text, r.kind))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn body_text_below_threshold_is_not_boilerplate() {
        let src = build_synthetic_pages("H", "F", "Body", 3);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        // The "Body content" line repeats 3× but body lines also vary; the
        // exact-dup cluster size is 2 (lines 2 and 3 of each page), 3 pages = 6.
        // Our `min_occurrences = 3` accepts it; `min_rate` accepts it. So it
        // *will* show up — assert that's exactly what happens, but it should
        // be classified as Unknown (no zone-dominance).
        for run in &runs {
            if run.canonical_text == "body" {
                assert!(matches!(
                    run.kind,
                    BoilerplateKind::Unknown | BoilerplateKind::Header
                ));
            }
        }
    }

    #[test]
    fn min_occurrences_filters_small_clusters() {
        // Two pages: the header repeats only twice, below default min_occurrences.
        let src = build_synthetic_pages("HEADER", "FOOTER", "Body", 2);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let headers: Vec<_> = runs
            .iter()
            .filter(|r| r.canonical_text == "header")
            .collect();
        assert!(
            headers.is_empty(),
            "header below min_occurrences should not be reported"
        );
    }

    #[test]
    fn near_dup_groups_ocr_drift() {
        // Realistic header length (~35 chars). Each occurrence has one OCR-
        // corrupted character — the kind of drift that defeats exact-dup but
        // should cluster via MinHash + 4-gram shingles (Q3 in the design
        // reference: one typo on a 30-char string drops Jaccard to ~0.85,
        // comfortably above the 0.75 threshold).
        let src = "FILED IN COURT 5/5/2026 SMITH V JONES\nbody one\nbody two\nfooter\n\u{000C}\
                   FlLED IN COURT 5/5/2026 SMITH V JONES\nbody one\nbody two\nfooter\n\u{000C}\
                   F1LED IN COURT 5/5/2026 SMITH V JONES\nbody one\nbody two\nfooter\n";
        let recs = extract_line_records(src);
        let runs = detect_boilerplate(&recs, src, BoilerplateOptions::default());
        let near_dup = runs.iter().find(|r| r.canonical_text.contains("court"));
        assert!(
            near_dup.is_some(),
            "MinHash residual pass should cluster OCR-drift headers"
        );
        if let Some(r) = near_dup {
            assert_eq!(r.occurrences, 3);
        }
    }

    #[test]
    fn skip_near_dup_disables_ocr_drift_clustering() {
        let src = "FILED IN COURT 5/5/2026 SMITH V JONES\nbody\n\u{000C}\
                   FlLED IN COURT 5/5/2026 SMITH V JONES\nbody\n\u{000C}\
                   F1LED IN COURT 5/5/2026 SMITH V JONES\nbody\n";
        let recs = extract_line_records(src);
        let opts = BoilerplateOptions {
            skip_near_dup: true,
            ..BoilerplateOptions::default()
        };
        let runs = detect_boilerplate(&recs, src, opts);
        let near_dup = runs.iter().find(|r| r.canonical_text.contains("court"));
        assert!(
            near_dup.is_none(),
            "skip_near_dup=true must not produce near-dup runs"
        );
    }

    #[test]
    fn fingerprints_unique_per_run() {
        // Build a large enough doc for several distinct boilerplate clusters.
        let src = build_synthetic_pages("Header A", "Page 1", "Body", 5);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let fps: AHashSet<u64> = runs.iter().map(|r| r.fingerprint).collect();
        assert_eq!(fps.len(), runs.len(), "fingerprints must be unique");
    }

    #[test]
    fn line_indices_are_valid_and_sorted() {
        let src = build_synthetic_pages("HEADER", "FOOTER", "Body", 4);
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        for run in &runs {
            assert!(run.line_indices.windows(2).all(|w| w[0] < w[1]));
            for &idx in &run.line_indices {
                assert!((idx as usize) < recs.len());
            }
        }
    }

    #[test]
    fn determinism_byte_identical_output() {
        let src = build_synthetic_pages("HEADER", "FOOTER", "Body", 5);
        let recs = extract_line_records(&src);
        let r1 = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let r2 = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        assert_eq!(r1, r2);
    }

    #[test]
    fn threshold_monotonicity() {
        let src = build_synthetic_pages("HEADER", "FOOTER", "Body", 4);
        let recs = extract_line_records(&src);
        let opts3 = BoilerplateOptions {
            min_occurrences: 3,
            ..BoilerplateOptions::default()
        };
        let r3 = detect_boilerplate(&recs, &src, opts3);
        let opts5 = BoilerplateOptions {
            min_occurrences: 5,
            ..BoilerplateOptions::default()
        };
        let r5 = detect_boilerplate(&recs, &src, opts5);
        // r5 must be a subset (by canonical text) of r3.
        let canon3: AHashSet<_> = r3.iter().map(|r| r.canonical_text.clone()).collect();
        for r in &r5 {
            assert!(
                canon3.contains(&r.canonical_text),
                "min_occurrences=5 produced a run absent at =3: {:?}",
                r.canonical_text
            );
        }
    }

    #[test]
    fn windowed_bucketing_when_no_form_feed() {
        // 50 lines per page → header at line 0 of every window. 4 pages.
        let mut src = String::new();
        for i in 0..4 {
            for line in 0..50 {
                if line == 0 {
                    src.push_str("WINDOWED HEADER");
                } else {
                    src.push_str(&format!("body line {} of page {}", line, i));
                }
                src.push('\n');
            }
        }
        let recs = extract_line_records(&src);
        let runs = detect_boilerplate(&recs, &src, BoilerplateOptions::default());
        let header = runs.iter().find(|r| r.canonical_text == "windowed header");
        assert!(header.is_some(), "windowed header should be detected");
        assert_eq!(header.unwrap().kind, BoilerplateKind::Header);
        assert_eq!(header.unwrap().occurrences, 4);
    }

    /// P5.6 — `parse_page_number_value` accepts only what unambiguously is
    /// a page-number value. This is a narrow shape test; the higher-level
    /// "is this a page-number RUN?" question is answered by the sequence
    /// detector via `detects_monotonic_page_number_sequence` etc.
    #[test]
    fn parse_page_number_value_accepts_canonical_shapes() {
        for (s, expected) in [
            ("1", Some((1, false))),
            ("27", Some((27, false))),
            ("iv", Some((4, true))),
            ("xliii", Some((43, true))),
            ("page 12", Some((12, false))),
            ("page 12 of 99", Some((12, false))),
        ] {
            assert_eq!(parse_page_number_value(s), expected, "{s:?}");
        }
    }

    #[test]
    fn parse_page_number_value_rejects_docket_numbers() {
        // Real-corpus false-positive shapes that we MUST NOT parse as a
        // page-number value (they would feed the sequence detector with
        // garbage candidates).
        for s in [
            "j-a21035-22",
            "08cr0085-l",
            "case number: 08cr0085-l",
            "8.8",
            "1.2.3",
        ] {
            assert_eq!(parse_page_number_value(s), None, "{s:?}");
        }
    }

    /// Roman conversion is strict: malformed Roman numerals must return None.
    #[test]
    fn roman_to_u32_rejects_malformed() {
        assert_eq!(roman_to_u32("iiii"), None);
        assert_eq!(roman_to_u32("vv"), None);
        assert_eq!(roman_to_u32("ic"), None); // not legal Roman for 99
                                              // Valid forms still work.
        assert_eq!(roman_to_u32("iv"), Some(4));
        assert_eq!(roman_to_u32("xc"), Some(90));
        assert_eq!(roman_to_u32("mcmxcix"), Some(1999));
    }

    proptest! {
        #[test]
        fn detect_never_panics(text in "\\PC{0,512}") {
            let recs = extract_line_records(&text);
            let _ = detect_boilerplate(&recs, &text, BoilerplateOptions::default());
        }

        #[test]
        fn line_indices_always_in_bounds(text in "\\PC{0,512}") {
            let recs = extract_line_records(&text);
            let runs = detect_boilerplate(&recs, &text, BoilerplateOptions::default());
            for run in &runs {
                for &idx in &run.line_indices {
                    prop_assert!((idx as usize) < recs.len());
                }
                prop_assert_eq!(run.occurrences as usize, run.line_indices.len());
            }
        }

        #[test]
        fn fingerprints_unique_property(text in "\\PC{0,512}") {
            let recs = extract_line_records(&text);
            let runs = detect_boilerplate(&recs, &text, BoilerplateOptions::default());
            let mut fps: Vec<u64> = runs.iter().map(|r| r.fingerprint).collect();
            fps.sort_unstable();
            let original_len = fps.len();
            fps.dedup();
            prop_assert_eq!(fps.len(), original_len);
        }
    }
}
