//! Benchmarks for data structures: vocabularies, inverted index.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use _rust::core::structures::{
    inverted_index::{Bm25Params, IdfWeight, InvertedIndex, TfWeight},
    vocabulary::{BloomVocabulary, FrequencyVocabulary, IndexedVocabulary, SetVocabulary},
};

fn load_words() -> Vec<String> {
    let text = std::fs::read_to_string("tests/fixtures/war_and_peace.txt")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. ".repeat(10000));
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn bench_vocabulary_insert(c: &mut Criterion) {
    let words = load_words();
    let mut group = c.benchmark_group("vocabulary/insert");

    group.bench_function("set", |b| {
        b.iter(|| {
            let mut v = SetVocabulary::new();
            for w in &words {
                v.insert(w);
            }
            black_box(v.len())
        })
    });

    group.bench_function("frequency", |b| {
        b.iter(|| {
            let mut v = FrequencyVocabulary::new();
            for w in &words {
                v.insert(w);
            }
            black_box(v.len())
        })
    });

    group.bench_function("indexed", |b| {
        b.iter(|| {
            let mut v = IndexedVocabulary::new();
            for w in &words {
                v.insert(w);
            }
            black_box(v.len())
        })
    });

    group.bench_function("bloom", |b| {
        b.iter(|| {
            let mut v = BloomVocabulary::new(50000, 0.01);
            for w in &words {
                v.insert(w);
            }
            black_box(v.approx_len())
        })
    });

    group.finish();
}

fn bench_vocabulary_lookup(c: &mut Criterion) {
    let words = load_words();

    // Build vocabs
    let mut set_v = SetVocabulary::new();
    let mut freq_v = FrequencyVocabulary::new();
    let mut idx_v = IndexedVocabulary::new();
    let mut bloom_v = BloomVocabulary::new(50000, 0.01);
    for w in &words {
        set_v.insert(w);
        freq_v.insert(w);
        idx_v.insert(w);
        bloom_v.insert(w);
    }

    let queries = [
        "prince",
        "war",
        "peace",
        "love",
        "death",
        "soldier",
        "napoleon",
        "nonexistent_word_xyz",
        "another_missing_word",
        "zzzzzzz",
    ];

    let mut group = c.benchmark_group("vocabulary/lookup");

    group.bench_function("set", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(set_v.contains(q));
            }
        })
    });

    group.bench_function("frequency", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(freq_v.contains(q));
            }
        })
    });

    group.bench_function("indexed", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(idx_v.contains(q));
            }
        })
    });

    group.bench_function("bloom", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(bloom_v.contains(q));
            }
        })
    });

    group.finish();
}

fn bench_frequency_top_n(c: &mut Criterion) {
    let words = load_words();
    let mut freq_v = FrequencyVocabulary::new();
    for w in &words {
        freq_v.insert(w);
    }

    let mut group = c.benchmark_group("vocabulary/top_n");
    for n in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| black_box(freq_v.top_n(n)))
        });
    }
    group.finish();
}

fn bench_inverted_index(c: &mut Criterion) {
    let text = std::fs::read_to_string("tests/fixtures/war_and_peace.txt")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. ".repeat(10000));

    // Split into ~500-word "documents" (paragraphs)
    let docs: Vec<Vec<&str>> = text
        .split("\n\n")
        .filter(|p| p.len() > 100)
        .map(|p| p.split_whitespace().collect())
        .collect();

    let mut group = c.benchmark_group("inverted_index");

    // Build
    group.bench_function("build", |b| {
        b.iter(|| {
            let mut idx = InvertedIndex::new();
            for (i, doc) in docs.iter().enumerate() {
                idx.add_document(i as u32, doc);
            }
            black_box(idx.term_count())
        })
    });

    // Build once for query benchmarks
    let mut idx = InvertedIndex::new();
    for (i, doc) in docs.iter().enumerate() {
        idx.add_document(i as u32, doc);
    }

    group.bench_function("query_and/2_terms", |b| {
        b.iter(|| black_box(idx.query_and(&["Prince", "war"])))
    });

    group.bench_function("query_or/2_terms", |b| {
        b.iter(|| black_box(idx.query_or(&["Prince", "war"])))
    });

    group.bench_function("query_and/5_terms", |b| {
        b.iter(|| black_box(idx.query_and(&["the", "and", "was", "Prince", "war"])))
    });

    group.bench_function("tf_idf", |b| {
        b.iter(|| {
            for term in ["Prince", "war", "peace", "the", "and"] {
                black_box(idx.tf_idf(term, 0));
            }
        })
    });

    group.finish();
}

fn bench_bm25(c: &mut Criterion) {
    let text = std::fs::read_to_string("tests/fixtures/war_and_peace.txt")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. ".repeat(10000));

    let docs: Vec<Vec<&str>> = text
        .split("\n\n")
        .filter(|p| p.len() > 100)
        .map(|p| p.split_whitespace().collect())
        .collect();

    let mut idx = InvertedIndex::new();
    for (i, doc) in docs.iter().enumerate() {
        idx.add_document(i as u32, doc);
    }

    let params = Bm25Params::default();
    let mut group = c.benchmark_group("bm25");

    group.bench_function("score/2_terms", |b| {
        b.iter(|| black_box(idx.score_bm25(&["Prince", "war"], 0, &params)))
    });

    group.bench_function("score/5_terms", |b| {
        b.iter(|| {
            black_box(idx.score_bm25(&["Prince", "war", "peace", "love", "death"], 0, &params))
        })
    });

    group.bench_function("query_top10/2_terms", |b| {
        b.iter(|| black_box(idx.query_bm25(&["Prince", "war"], &params, 10)))
    });

    group.bench_function("query_top10/5_terms", |b| {
        b.iter(|| {
            black_box(idx.query_bm25(&["Prince", "war", "peace", "love", "death"], &params, 10))
        })
    });

    group.bench_function("query_top100/5_terms", |b| {
        b.iter(|| {
            black_box(idx.query_bm25(&["Prince", "war", "peace", "love", "death"], &params, 100))
        })
    });

    group.finish();
}

fn bench_tf_idf_variants(c: &mut Criterion) {
    let text = std::fs::read_to_string("tests/fixtures/war_and_peace.txt")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. ".repeat(10000));

    let docs: Vec<Vec<&str>> = text
        .split("\n\n")
        .filter(|p| p.len() > 100)
        .map(|p| p.split_whitespace().collect())
        .collect();

    let mut idx = InvertedIndex::new();
    for (i, doc) in docs.iter().enumerate() {
        idx.add_document(i as u32, doc);
    }

    let mut group = c.benchmark_group("tf_idf_variants");

    group.bench_function("raw_standard", |b| {
        b.iter(|| {
            black_box(idx.score_tf_idf(
                &["Prince", "war", "peace"],
                0,
                TfWeight::Raw,
                IdfWeight::Standard,
            ))
        })
    });

    group.bench_function("sublinear_smooth", |b| {
        b.iter(|| {
            black_box(idx.score_tf_idf(
                &["Prince", "war", "peace"],
                0,
                TfWeight::Sublinear,
                IdfWeight::Smooth,
            ))
        })
    });

    group.bench_function("query_top10/sublinear_smooth", |b| {
        b.iter(|| {
            black_box(idx.query_tf_idf(
                &["Prince", "war", "peace"],
                TfWeight::Sublinear,
                IdfWeight::Smooth,
                10,
            ))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_vocabulary_insert,
    bench_vocabulary_lookup,
    bench_frequency_top_n,
    bench_inverted_index,
    bench_bm25,
    bench_tf_idf_variants,
);
criterion_main!(benches);
