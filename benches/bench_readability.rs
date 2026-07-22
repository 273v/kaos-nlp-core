//! Benchmarks for readability counting and the syllable kernel.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use _rust::core::readability::syllable::estimate_syllables;
use _rust::core::readability::{count_text_no_lexicon, ReadabilityConfig};

fn load_text() -> String {
    std::fs::read_to_string("tests/fixtures/war_and_peace.txt")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. ".repeat(10000))
}

fn bench_syllable_kernel(c: &mut Criterion) {
    let words = [
        "the",
        "cat",
        "extraordinary",
        "internationalization",
        "state-of-the-art",
        "don't",
        "bureaucracy",
        "table",
        "asked",
        "appreciate",
    ];
    c.bench_function("syllable_estimate_10_words", |b| {
        b.iter(|| {
            let mut total = 0u32;
            for w in &words {
                total += estimate_syllables(black_box(w));
            }
            total
        })
    });
}

fn bench_count_text(c: &mut Criterion) {
    let text = load_text();
    let sample: String = text.chars().take(1_000_000).collect();
    let config = ReadabilityConfig::default();
    c.bench_function("count_text_1mb_heuristic", |b| {
        b.iter(|| count_text_no_lexicon(black_box(&sample), None, &config))
    });
}

criterion_group!(benches, bench_syllable_kernel, bench_count_text);
criterion_main!(benches);
