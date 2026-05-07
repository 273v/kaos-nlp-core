//! Benchmarks for the LineRecord extractor.
//!
//! Three workloads:
//!   - synthetic 100 KiB ASCII prose (line-record-heavy)
//!   - real legal text from `tests/fixtures/shakespeare.txt`
//!   - synthetic Unicode-heavy text (CJK + emoji) so the non-ASCII path is
//!     exercised
//!
//! Measure throughput in bytes/sec via Criterion's `bytes_per_second`. The
//! ASCII fast path should approach memcpy speeds; the Unicode path is
//! intentionally slower because it walks `char_indices`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use _rust::core::segmentation::extract_line_records;

fn synthetic_ascii(target_bytes: usize) -> String {
    // Mix of short / medium / long lines so the bench reflects real prose.
    let line_pool = [
        "ARTICLE I. PURPOSES",
        "  1. The Borrower hereby agrees to repay all sums advanced.",
        "  (a) Notwithstanding the foregoing, no waiver shall be effective unless made in writing.",
        "    (i) The interest rate shall accrue daily from the date of disbursement.",
        "",
        "Section 5. Severability.",
        "If any provision is held invalid, the remainder shall remain in full force and effect.",
    ];
    let mut s = String::with_capacity(target_bytes + 256);
    while s.len() < target_bytes {
        for line in &line_pool {
            s.push_str(line);
            s.push('\n');
            if s.len() >= target_bytes {
                break;
            }
        }
    }
    s
}

fn synthetic_unicode(target_bytes: usize) -> String {
    let line_pool = [
        "第一条 目的（こうもく）。",
        "  一、本契約の目的は債権の返済である。",
        "    （イ）貸付日より日々利息が発生する。",
        "",
        "Article II — Définitions et applications 😀",
        "Article III — Стороны соглашаются на следующих условиях.",
    ];
    let mut s = String::with_capacity(target_bytes + 256);
    while s.len() < target_bytes {
        for line in &line_pool {
            s.push_str(line);
            s.push('\n');
            if s.len() >= target_bytes {
                break;
            }
        }
    }
    s
}

fn load_fixture(name: &str) -> Option<String> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    std::fs::read_to_string(path).ok()
}

fn bench_extract(c: &mut Criterion) {
    let mut group = c.benchmark_group("line_record/extract");

    // ── Synthetic ASCII at multiple sizes ─────────────────────────────────
    for &kb in &[16usize, 100, 1024] {
        let text = synthetic_ascii(kb * 1024);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("ascii_synthetic", format!("{}KiB", kb)),
            &text,
            |b, t| {
                b.iter(|| {
                    let recs = extract_line_records(black_box(t));
                    black_box(recs);
                });
            },
        );
    }

    // ── Synthetic Unicode (slow path) ─────────────────────────────────────
    for &kb in &[16usize, 100] {
        let text = synthetic_unicode(kb * 1024);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("unicode_synthetic", format!("{}KiB", kb)),
            &text,
            |b, t| {
                b.iter(|| {
                    let recs = extract_line_records(black_box(t));
                    black_box(recs);
                });
            },
        );
    }

    // ── Real-text fixture (Shakespeare ~5.4 MiB ASCII English) ─────────────
    if let Some(text) = load_fixture("shakespeare.txt") {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("real_text", "shakespeare"),
            &text,
            |b, t| {
                b.iter(|| {
                    let recs = extract_line_records(black_box(t));
                    black_box(recs);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_extract);
criterion_main!(benches);
