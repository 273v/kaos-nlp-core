//! Benchmarks for the enumerator parser.
//!
//! Per the design reference (Q7) the budget is < 200 ns per call for the
//! bare forms (Roman / Decimal / Alpha / Parenthetical) and < 500 ns per
//! call for the word-prefixed form. These benches measure each kind in
//! isolation and amortised over a 1k-line scan.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use _rust::core::segmentation::parse_enumerator;

fn bench_kinds(c: &mut Criterion) {
    let mut group = c.benchmark_group("enumerator/single_call");

    let cases: &[(&str, &str)] = &[
        ("decimal_single", "1. Introduction"),
        ("decimal_dotted_2", "1.2 Definitions"),
        ("decimal_dotted_3", "1.2.3 Subitem"),
        ("alpha_upper", "A. First section"),
        ("alpha_lower", "c. third item"),
        ("roman_single", "I. Background"),
        ("roman_multi", "XIII. Conclusion of analysis"),
        ("paren_alpha", "(a) Definitions"),
        ("paren_decimal", "(1) The first item"),
        ("paren_roman", "(iv) Note about prior"),
        ("section_word", "Section 5 Title"),
        ("section_abbrev", "Sec. 5.2 Heading"),
        ("section_sigil", "§ 5 Title"),
        ("chapter_word", "Chapter 7 — Title 11"),
        ("subpart_word", "Subpart B Filings"),
        // Negative case: body text that should return None quickly.
        (
            "body_no_match",
            "The Borrower hereby agrees to repay sums advanced.",
        ),
    ];

    for (name, src) in cases {
        group.bench_with_input(BenchmarkId::new("kind", name), src, |b, s| {
            b.iter(|| black_box(parse_enumerator(black_box(s))));
        });
    }
    group.finish();
}

fn bench_scan(c: &mut Criterion) {
    // Realistic mixed scan: 1000 lines drawn from the same case set.
    let cases: &[&str] = &[
        "1. Introduction",
        "1.2 Definitions",
        "(a) item",
        "(1) item",
        "(iv) note",
        "I. Background",
        "II. Discussion",
        "Section 5 Title",
        "Sec. 5.2 Heading",
        "Chapter 7 Title",
        "Subpart B Filings",
        "Article III Authority",
        "The Borrower hereby agrees.",
        "All payments shall be made in lawful currency.",
        "Whereas, the parties enter into this agreement.",
    ];

    let mut lines: Vec<&str> = Vec::with_capacity(1000);
    for i in 0..1000 {
        lines.push(cases[i % cases.len()]);
    }

    let mut group = c.benchmark_group("enumerator/scan");
    let total_bytes: usize = lines.iter().map(|s| s.len()).sum();
    group.throughput(Throughput::Elements(lines.len() as u64));
    group.bench_function(BenchmarkId::new("mixed_1k_lines", total_bytes), |b| {
        b.iter(|| {
            for line in &lines {
                black_box(parse_enumerator(black_box(line)));
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_kinds, bench_scan);
criterion_main!(benches);
