//! Benchmarks for the offset-preserving normalizer.
//!
//! Goals:
//!   - Confirm the ASCII fast path approaches memcpy speed (Cow::Borrowed
//!     return; no allocation).
//!   - Quantify the cost of each transform individually (whitespace
//!     collapse, fold_case, unicode-punct, strip_punct).
//!   - Confirm aggressive (all-on) is dominated by the per-char loop, not
//!     by any single transform.
//!
//! Each input is fed to Criterion with `Throughput::Bytes` so the report
//! reads in MiB/s.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use _rust::core::segmentation::{normalize, NormalizeOptions};

fn synthetic_ascii(target_bytes: usize) -> String {
    let pool = [
        "ARTICLE I. PURPOSES",
        "  1. The Borrower hereby agrees to repay all sums advanced.",
        "  (a) Notwithstanding the foregoing, no waiver shall be effective.",
        "    (i) The interest rate shall accrue daily from disbursement.",
        "Section 5. Severability.",
        "If any provision is held invalid, the remainder shall remain.",
    ];
    let mut s = String::with_capacity(target_bytes + 256);
    while s.len() < target_bytes {
        for line in &pool {
            s.push_str(line);
            s.push('\n');
            if s.len() >= target_bytes {
                break;
            }
        }
    }
    s
}

fn synthetic_unicode_legal(target_bytes: usize) -> String {
    // Mix of smart quotes, dashes, NBSP, ellipsis — common in PDF-extracted
    // legal text where the smart-punct flag is the whole point.
    let pool = [
        "“The Parties”\u{00A0}agree as follows:",
        "  • Item 1—the obligor shall…",
        "  • Item 2—the obligee shall…",
        "Section\u{00A0}5. Severability\u{2014}see Annex A.",
        "Whereas,\u{2009}the foregoing recital is incorporated\u{2026}",
    ];
    let mut s = String::with_capacity(target_bytes + 256);
    while s.len() < target_bytes {
        for line in &pool {
            s.push_str(line);
            s.push('\n');
            if s.len() >= target_bytes {
                break;
            }
        }
    }
    s
}

fn synthetic_unicode_mixed_case(target_bytes: usize) -> String {
    // Upper/mixed-case non-ASCII text that exercises the Unicode case-fold
    // path, including expanding folds (ß → ss, İ → i + U+0307).
    let pool = [
        "SÃO PAULO É A MAIOR CIDADE DO BRASIL.",
        "İSTANBUL ŞEHRİ VE BOĞAZİÇİ KÖPRÜSÜ.",
        "DIE HAUPTSTRAßE FÜHRT ZUM ÖFFENTLICHEN PLATZ.",
        "ΟΙ ΑΘΗΝΑΙΟΙ ΣΥΝΕΛΕΓΗΣΑΝ ΣΤΗΝ ΑΓΟΡΑ.",
        "Mixed ASCII Heading With Ümlaut And Ñandú",
    ];
    let mut s = String::with_capacity(target_bytes + 256);
    while s.len() < target_bytes {
        for line in &pool {
            s.push_str(line);
            s.push('\n');
            if s.len() >= target_bytes {
                break;
            }
        }
    }
    s
}

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");

    let opts_noop = NormalizeOptions::default();
    let opts_punct = NormalizeOptions {
        normalize_unicode_punct: true,
        ..NormalizeOptions::default()
    };
    let opts_collapse = NormalizeOptions {
        collapse_whitespace: true,
        ..NormalizeOptions::default()
    };
    let opts_fold = NormalizeOptions {
        fold_case: true,
        ..NormalizeOptions::default()
    };
    let opts_aggressive = NormalizeOptions::aggressive();

    // ── Fast path — ASCII + only the unicode flag → Cow::Borrowed ──────────
    for &kb in &[16usize, 100, 1024] {
        let text = synthetic_ascii(kb * 1024);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("ascii_fast_path", format!("{}KiB", kb)),
            &text,
            |b, t| {
                b.iter(|| {
                    let r = normalize(black_box(t), opts_punct).unwrap();
                    black_box(r);
                });
            },
        );
    }

    // ── Single transforms on ASCII ─────────────────────────────────────────
    for &kb in &[16usize, 100] {
        let text = synthetic_ascii(kb * 1024);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("ascii_collapse", format!("{}KiB", kb)),
            &text,
            |b, t| {
                b.iter(|| black_box(normalize(black_box(t), opts_collapse).unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ascii_fold", format!("{}KiB", kb)),
            &text,
            |b, t| {
                b.iter(|| black_box(normalize(black_box(t), opts_fold).unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ascii_aggressive", format!("{}KiB", kb)),
            &text,
            |b, t| {
                b.iter(|| black_box(normalize(black_box(t), opts_aggressive).unwrap()));
            },
        );
    }

    // ── Unicode legal text — slow path with real punct work ────────────────
    for &kb in &[16usize, 100] {
        let text = synthetic_unicode_legal(kb * 1024);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("unicode_punct", format!("{}KiB", kb)),
            &text,
            |b, t| {
                b.iter(|| black_box(normalize(black_box(t), opts_punct).unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("unicode_aggressive", format!("{}KiB", kb)),
            &text,
            |b, t| {
                b.iter(|| black_box(normalize(black_box(t), opts_aggressive).unwrap()));
            },
        );
    }

    // ── Unicode case fold — non-ASCII mixed-case text ─────────────────────
    //
    // `fold_case` applies full Unicode case folding (icu_casemap). ASCII
    // chars keep a table-free fast path, so `ascii_fold` above is the
    // pure-ASCII number; this group measures the per-char ICU lookup on
    // text where most letters are non-ASCII or uppercase.
    for &kb in &[16usize, 100] {
        let text = synthetic_unicode_mixed_case(kb * 1024);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("unicode_fold", format!("{}KiB", kb)),
            &text,
            |b, t| {
                b.iter(|| black_box(normalize(black_box(t), opts_fold).unwrap()));
            },
        );
    }

    // ── No-op control (default opts on ASCII = literal Cow::Borrowed) ─────
    let text = synthetic_ascii(100 * 1024);
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_with_input(BenchmarkId::new("control_noop", "100KiB"), &text, |b, t| {
        b.iter(|| black_box(normalize(black_box(t), opts_noop).unwrap()));
    });

    group.finish();
}

criterion_group!(benches, bench_normalize);
criterion_main!(benches);
