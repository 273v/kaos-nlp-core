//! Benchmarks for the document-structure layer (P7.3).
//!
//! Three benchmark groups:
//!
//! 1. **scoring** — per-line feature extraction throughput on synthetic
//!    legal-shaped text.
//! 2. **decoder** — Viterbi sequence decode over precomputed feature
//!    vectors. Independent of the scorer so we can quantify decode cost.
//! 3. **end_to_end** — the full pipeline (extract → score → decode →
//!    hierarchy) so the user sees what `label_lines` costs.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use _rust::core::segmentation::{
    detect_boilerplate, extract_line_records, parse_enumerator, BoilerplateOptions, Enumerator,
};
use _rust::core::structure::{
    decode_line_labels, infer_hierarchy, score_heading_features, DecoderOptions, HierarchyOptions,
    ScoringOptions,
};

fn synthetic_legal_doc(n_pages: u32) -> String {
    // Mix of headings, body, list items, table-shaped lines, and metadata
    // — broadly the shape the structure layer is meant to handle.
    let mut s = String::with_capacity((n_pages as usize) * 1024);
    s.push_str("UNITED STATES DISTRICT COURT\n");
    s.push_str("DISTRICT OF COLUMBIA\n\n");
    s.push_str("Author: Jane Doe\nDate: 2026-05-05\nCase Number: 22-1234\n\n");
    for page in 0..n_pages {
        s.push_str(&format!("Page {} of {}\n\n", page + 1, n_pages));
        s.push_str("BACKGROUND\n\n");
        s.push_str("The plaintiffs filed suit alleging a violation of 5 U.S.C. § 552. ");
        s.push_str(
            "They argued that the agency had failed to produce records in a timely manner. ",
        );
        s.push_str("The defendants moved to dismiss on grounds of sovereign immunity.\n\n");
        s.push_str("Section 5 Definitions\n\n");
        s.push_str("(a) Apples\n(b) Bananas\n(c) Cherries\n\n");
        s.push_str("DISCUSSION\n\n");
        s.push_str("The court considered each argument in turn.\n");
        s.push_str("Col A | Col B | Col C\nRow 1 | Val | Val\nRow 2 | Val | Val\n\n");
        s.push_str(&format!("ORDER {}\n\n", page));
        s.push_str("It is hereby ordered that the motion is denied.\n");
        if page + 1 < n_pages {
            s.push('\u{000C}');
        }
    }
    s
}

fn precompute_inputs(
    text: &str,
) -> (
    Vec<_rust::core::segmentation::LineRecord>,
    Vec<Option<Enumerator>>,
    Vec<_rust::core::segmentation::BoilerplateRun>,
) {
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
    (records, enumerators, runs)
}

fn bench_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("structure/scoring");
    let opts = ScoringOptions::default();

    for &pages in &[10u32, 100, 500] {
        let text = synthetic_legal_doc(pages);
        let (records, enums, runs) = precompute_inputs(&text);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("score", format!("{}p", pages)),
            &(records, enums, runs, text.clone()),
            |b, (records, enums, runs, text)| {
                b.iter(|| {
                    let features = score_heading_features(
                        black_box(text),
                        black_box(records),
                        black_box(enums),
                        black_box(runs),
                        black_box(&opts),
                    );
                    black_box(features);
                });
            },
        );
    }
    group.finish();
}

fn bench_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("structure/decoder");
    let s_opts = ScoringOptions::default();
    let d_opts = DecoderOptions::default();

    for &pages in &[10u32, 100, 500] {
        let text = synthetic_legal_doc(pages);
        let (records, enums, runs) = precompute_inputs(&text);
        let features = score_heading_features(&text, &records, &enums, &runs, &s_opts);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("decode", format!("{}p", pages)),
            &features,
            |b, features| {
                b.iter(|| {
                    let labels = decode_line_labels(black_box(features), black_box(&d_opts));
                    black_box(labels);
                });
            },
        );
    }
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("structure/end_to_end");
    let s_opts = ScoringOptions::default();
    let d_opts = DecoderOptions::default();
    let h_opts = HierarchyOptions::default();

    for &pages in &[10u32, 100, 500] {
        let text = synthetic_legal_doc(pages);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("full_pipeline", format!("{}p", pages)),
            &text,
            |b, text| {
                b.iter(|| {
                    let recs = extract_line_records(black_box(text));
                    let enums: Vec<Option<Enumerator>> = recs
                        .iter()
                        .map(|r| {
                            if r.blank {
                                None
                            } else {
                                parse_enumerator(r.stripped_text(text))
                            }
                        })
                        .collect();
                    let runs = detect_boilerplate(&recs, text, BoilerplateOptions::default());
                    let features = score_heading_features(text, &recs, &enums, &runs, &s_opts);
                    let labels = decode_line_labels(&features, &d_opts);
                    let candidates = infer_hierarchy(&labels, &features, &enums, &h_opts);
                    black_box(candidates);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_scoring, bench_decoder, bench_end_to_end);
criterion_main!(benches);
