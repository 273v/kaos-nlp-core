//! Benchmarks for pattern matching: substring, multi-pattern, regex, FST.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use _rust::core::matching::{
    fst_match::FstSet,
    multi_pattern::{MultiPatternMatchKind, MultiPatternMatcher},
    regex_match::RegexMatcher,
    substring,
};

fn load_text(filename: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{filename}"))
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. ".repeat(10000))
}

fn bench_substring_search(c: &mut Criterion) {
    let text = load_text("war_and_peace.txt");
    let mut group = c.benchmark_group("substring");

    for needle in ["Prince", "the", "Natasha Rostova", "NONEXISTENT_STRING_XYZ"] {
        group.bench_with_input(BenchmarkId::new("find_all", needle), needle, |b, needle| {
            b.iter(|| black_box(substring::find_all(&text, needle)))
        });
    }

    group.bench_function("count/the", |b| {
        b.iter(|| black_box(substring::count(&text, "the")))
    });

    group.bench_function("find_first/Prince", |b| {
        b.iter(|| black_box(substring::find_first(&text, "Prince")))
    });

    group.bench_function("case_insensitive/prince", |b| {
        b.iter(|| black_box(substring::find_all_case_insensitive(&text, "prince")))
    });

    group.finish();
}

fn bench_multi_pattern(c: &mut Criterion) {
    let text = load_text("war_and_peace.txt");
    let mut group = c.benchmark_group("multi_pattern");

    // Small pattern set
    let small = MultiPatternMatcher::new(
        &["Prince", "Princess", "Count", "Countess"],
        MultiPatternMatchKind::LeftmostFirst,
    )
    .unwrap();

    group.bench_function("4_patterns", |b| {
        b.iter(|| black_box(small.find_all(&text)))
    });

    // Medium pattern set — common English words
    let medium_patterns: Vec<&str> = vec![
        "the", "and", "was", "for", "that", "with", "his", "her", "from", "they", "have", "this",
        "been", "would", "could", "their", "which", "about", "other", "into",
    ];
    let medium =
        MultiPatternMatcher::new(&medium_patterns, MultiPatternMatchKind::LeftmostFirst).unwrap();

    group.bench_function("20_patterns", |b| {
        b.iter(|| black_box(medium.find_all(&text)))
    });

    group.bench_function("20_patterns/count", |b| {
        b.iter(|| black_box(medium.count(&text)))
    });

    group.finish();
}

fn bench_regex(c: &mut Criterion) {
    let text = load_text("shakespeare.txt");
    let mut group = c.benchmark_group("regex");

    // Simple word boundary match
    let word_re = RegexMatcher::new(r"\b[Ll]ove\b").unwrap();
    group.bench_function("word_boundary/love", |b| {
        b.iter(|| black_box(word_re.find_all(&text)))
    });

    // Date-like patterns
    let date_re = RegexMatcher::new(r"\b\d{4}\b").unwrap();
    group.bench_function("four_digit_numbers", |b| {
        b.iter(|| black_box(date_re.find_all(&text)))
    });

    // Capitalized words
    let cap_re = RegexMatcher::new(r"\b[A-Z][a-z]{3,}\b").unwrap();
    group.bench_function("capitalized_words", |b| {
        b.iter(|| black_box(cap_re.find_all(&text)))
    });

    // Complex alternation
    let alt_re = RegexMatcher::new(r"\b(Romeo|Juliet|Hamlet|Othello|Macbeth|Lear)\b").unwrap();
    group.bench_function("character_names", |b| {
        b.iter(|| black_box(alt_re.find_all(&text)))
    });

    group.finish();
}

fn bench_fst(c: &mut Criterion) {
    let text = load_text("war_and_peace.txt");
    let mut group = c.benchmark_group("fst");

    // Build vocabulary from the text
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();
    let unique: std::collections::BTreeSet<&str> = words.iter().map(|s| s.as_str()).collect();
    let vocab: Vec<&str> = unique.into_iter().collect();

    // Build FST from War and Peace vocabulary
    group.bench_function("build/war_and_peace_vocab", |b| {
        b.iter(|| black_box(FstSet::build(vocab.iter().copied()).unwrap()))
    });

    let fst = FstSet::build(vocab.iter().copied()).unwrap();
    let vocab_size = vocab.len();

    group.bench_function(format!("contains/{vocab_size}_terms"), |b| {
        b.iter(|| {
            for word in ["prince", "war", "peace", "love", "death", "nonexistent"] {
                black_box(fst.contains(word));
            }
        })
    });

    group.bench_function("prefix_search/pr", |b| {
        b.iter(|| black_box(fst.prefix_search("pr")))
    });

    group.bench_function("fuzzy_search/princ/d1", |b| {
        b.iter(|| black_box(fst.fuzzy_search("princ", 1).unwrap()))
    });

    group.bench_function("fuzzy_search/princ/d2", |b| {
        b.iter(|| black_box(fst.fuzzy_search("princ", 2).unwrap()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_substring_search,
    bench_multi_pattern,
    bench_regex,
    bench_fst,
);
criterion_main!(benches);
