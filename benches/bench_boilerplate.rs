//! Benchmarks for the boilerplate detector.
//!
//! Three workloads:
//!
//! 1. **Pure-exact-dup** — synthetic clean PDF text with form-feed page
//!    breaks; every page has the same header / footer; tests the ahash
//!    bucket path.
//! 2. **MinHash residual** — same shape but every header occurrence has one
//!    OCR-corrupted character; tests the slow path that hits the existing
//!    `MinHasher` + `MinHashIndex` machinery.
//! 3. **Skip-near-dup** — same workload as (2) but with `skip_near_dup =
//!    true`, so we can quantify the cost of the residual pass.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use _rust::core::segmentation::{detect_boilerplate, extract_line_records, BoilerplateOptions};

fn synthetic_clean_pages(n_pages: u32) -> String {
    let mut s = String::with_capacity((n_pages as usize) * 5 * 64);
    for i in 0..n_pages {
        s.push_str("FILED 5/5/2026 SMITH V JONES — PAGE BANNER\n");
        s.push_str("Body content paragraph one.\n");
        s.push_str(&format!("Body content paragraph two on page {}.\n", i));
        s.push_str("Body content paragraph three.\n");
        s.push_str("CONFIDENTIAL — INTERNAL ONLY\n");
        if i + 1 < n_pages {
            s.push('\u{000C}');
        }
    }
    s
}

fn synthetic_ocr_drift_pages(n_pages: u32) -> String {
    // Header has a one-char OCR error per page that varies by index, so
    // exact-dup misses but MinHash 4-gram catches.
    let header_template = "FILED IN COURT 5_5_2026 SMITH V JONES — PAGE BANNER";
    let mut s = String::with_capacity((n_pages as usize) * 5 * 64);
    for i in 0..n_pages {
        // Replace one character with a digit at a position that rotates over
        // the line, so each page has a unique drift.
        let pos = 1 + (i as usize) % (header_template.len() - 1);
        let mut header_bytes = header_template.as_bytes().to_vec();
        header_bytes[pos] = b'1';
        let header = std::str::from_utf8(&header_bytes).unwrap();
        s.push_str(header);
        s.push('\n');
        s.push_str("Body content paragraph one.\n");
        s.push_str(&format!("Body content paragraph two on page {}.\n", i));
        s.push_str("Body content paragraph three.\n");
        s.push_str("CONFIDENTIAL — INTERNAL ONLY\n");
        if i + 1 < n_pages {
            s.push('\u{000C}');
        }
    }
    s
}

fn bench_detect(c: &mut Criterion) {
    let mut group = c.benchmark_group("boilerplate/detect");
    let opts_default = BoilerplateOptions::default();
    let opts_skip_minhash = BoilerplateOptions {
        skip_near_dup: true,
        ..BoilerplateOptions::default()
    };

    for &pages in &[10u32, 100, 1000] {
        // 1. Pure exact-dup — the fast path.
        let clean = synthetic_clean_pages(pages);
        let recs = extract_line_records(&clean);
        group.throughput(Throughput::Bytes(clean.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("clean_exact_dup", format!("{}p", pages)),
            &(recs.clone(), clean.clone()),
            |b, (recs, clean)| {
                b.iter(|| {
                    let runs = detect_boilerplate(black_box(recs), black_box(clean), opts_default);
                    black_box(runs);
                });
            },
        );

        // 2. OCR drift — exact-dup is empty; MinHash residual carries the load.
        let drift = synthetic_ocr_drift_pages(pages);
        let recs_drift = extract_line_records(&drift);
        group.throughput(Throughput::Bytes(drift.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("ocr_drift_minhash", format!("{}p", pages)),
            &(recs_drift.clone(), drift.clone()),
            |b, (recs, drift)| {
                b.iter(|| {
                    let runs = detect_boilerplate(black_box(recs), black_box(drift), opts_default);
                    black_box(runs);
                });
            },
        );

        // 3. OCR drift with `skip_near_dup` — quantifies the MinHash cost.
        group.bench_with_input(
            BenchmarkId::new("ocr_drift_skip_minhash", format!("{}p", pages)),
            &(recs_drift, drift),
            |b, (recs, drift)| {
                b.iter(|| {
                    let runs =
                        detect_boilerplate(black_box(recs), black_box(drift), opts_skip_minhash);
                    black_box(runs);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_detect);
criterion_main!(benches);
