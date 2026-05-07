//! Benchmarks for string distance and similarity algorithms.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use _rust::core::algorithms::{
    edit::{DamerauLevenshtein, Hamming, Jaro, JaroWinkler, Levenshtein, Osa, SorensenDice},
    ngram::{
        NgramCosine, NgramJaccard, TokenJaccard, TokenNgramCosine, TokenNgramJaccard,
        TokenNgramOverlap,
    },
    phonetic::{Metaphone, Soundex},
    sequence::Lcs,
    traits::StringDistance,
};

/// Short string pairs for name-matching benchmarks.
const SHORT_PAIRS: &[(&str, &str)] = &[
    ("kitten", "sitting"),
    ("martha", "marhta"),
    ("Robert", "Rupert"),
    ("William", "Williams"),
    ("Philadelphia", "Philadlephia"),
    ("algorithm", "altruistic"),
];

/// Longer strings extracted from War and Peace (first ~500 chars of paragraphs).
fn load_war_and_peace_paragraphs() -> Vec<String> {
    let text = std::fs::read_to_string("tests/fixtures/war_and_peace.txt").unwrap_or_else(|_| {
        // Fallback if file not present
        "The quick brown fox jumps over the lazy dog. ".repeat(100)
    });
    text.split("\n\n")
        .filter(|p| p.len() > 200)
        .take(20)
        .map(|p| p.chars().take(500).collect())
        .collect()
}

fn bench_edit_distance_short(c: &mut Criterion) {
    let mut group = c.benchmark_group("edit_distance/short");

    let algos: Vec<(&str, Box<dyn StringDistance>)> = vec![
        (
            "levenshtein",
            Box::new(Levenshtein) as Box<dyn StringDistance>,
        ),
        ("damerau_levenshtein", Box::new(DamerauLevenshtein)),
        ("osa", Box::new(Osa)),
        ("jaro", Box::new(Jaro)),
        ("jaro_winkler", Box::new(JaroWinkler::default())),
        ("sorensen_dice", Box::new(SorensenDice)),
    ];
    for (name, algo) in &algos {
        group.bench_function(*name, |b| {
            b.iter(|| {
                for (a, s) in SHORT_PAIRS {
                    black_box(algo.distance(a, s).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_hamming(c: &mut Criterion) {
    let h = Hamming;
    let pairs: Vec<(String, String)> = (0..6)
        .map(|i| {
            let a: String = (0..1000)
                .map(|j| ((65 + (i + j) % 26) as u8) as char)
                .collect();
            let b: String = (0..1000)
                .map(|j| ((65 + (i + j + 1) % 26) as u8) as char)
                .collect();
            (a, b)
        })
        .collect();

    c.bench_function("hamming/1000_chars", |b| {
        b.iter(|| {
            for (a, s) in &pairs {
                black_box(h.distance(a, s).unwrap());
            }
        })
    });
}

fn bench_edit_distance_paragraphs(c: &mut Criterion) {
    let paras = load_war_and_peace_paragraphs();
    if paras.len() < 2 {
        return;
    }

    let mut group = c.benchmark_group("edit_distance/paragraphs");
    group.sample_size(10);

    // Compare consecutive paragraph pairs
    let pairs: Vec<(&str, &str)> = paras
        .windows(2)
        .map(|w| (w[0].as_str(), w[1].as_str()))
        .collect();

    let algos: Vec<(&str, Box<dyn StringDistance>)> = vec![
        (
            "levenshtein",
            Box::new(Levenshtein) as Box<dyn StringDistance>,
        ),
        ("jaro_winkler", Box::new(JaroWinkler::default())),
        ("sorensen_dice", Box::new(SorensenDice)),
    ];
    for (name, algo) in &algos {
        group.bench_function(*name, |b| {
            b.iter(|| {
                for (a, s) in &pairs {
                    black_box(algo.distance(a, s).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_ngram_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("ngram_similarity");

    for n in [2, 3, 4] {
        let jac = NgramJaccard { n };
        let cos = NgramCosine { n };

        group.bench_with_input(BenchmarkId::new("jaccard", n), &n, |b, _| {
            b.iter(|| {
                for (a, s) in SHORT_PAIRS {
                    black_box(jac.distance(a, s).unwrap());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("cosine", n), &n, |b, _| {
            b.iter(|| {
                for (a, s) in SHORT_PAIRS {
                    black_box(cos.distance(a, s).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_phonetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonetic");
    let soundex = Soundex;
    let metaphone = Metaphone;

    let names = [
        "Robert",
        "Rupert",
        "William",
        "Williams",
        "Philadelphia",
        "Mississippi",
    ];

    group.bench_function("soundex_encode", |b| {
        b.iter(|| {
            for name in &names {
                black_box(soundex.encode(name));
            }
        })
    });

    group.bench_function("metaphone_encode", |b| {
        b.iter(|| {
            for name in &names {
                black_box(metaphone.encode(name));
            }
        })
    });

    group.finish();
}

fn bench_lcs(c: &mut Criterion) {
    let lcs = Lcs;

    c.bench_function("lcs/short", |b| {
        b.iter(|| {
            for (a, s) in SHORT_PAIRS {
                black_box(lcs.lcs_length(a, s));
            }
        })
    });

    // LCS on longer strings (100 chars)
    let a: String = "abcdefghij".repeat(10);
    let s: String = "abcxefghij".repeat(10);
    c.bench_function("lcs/100_chars", |b| {
        b.iter(|| {
            black_box(lcs.lcs_length(&a, &s));
        })
    });
}

/// Sentence pairs for token n-gram benchmarks.
const SENTENCE_PAIRS: &[(&str, &str)] = &[
    ("the quick brown fox", "a quick brown dog"),
    ("New York City is great", "New York City is wonderful"),
    ("machine learning algorithms", "deep learning algorithms"),
    (
        "United States of America",
        "United Kingdom of Great Britain",
    ),
    (
        "natural language processing",
        "natural language understanding",
    ),
    ("the cat sat on the mat", "the dog sat on the log"),
];

fn bench_token_ngram_short(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_ngram/short");

    let tj = TokenJaccard { lowercase: true };
    group.bench_function("token_jaccard", |b| {
        b.iter(|| {
            for (a, s) in SENTENCE_PAIRS {
                black_box(tj.distance(a, s).unwrap());
            }
        })
    });

    for n in [2, 3] {
        let tnj = TokenNgramJaccard { n, lowercase: true };
        let tnc = TokenNgramCosine { n, lowercase: true };
        let tno = TokenNgramOverlap { n, lowercase: true };

        group.bench_with_input(BenchmarkId::new("jaccard", n), &n, |b, _| {
            b.iter(|| {
                for (a, s) in SENTENCE_PAIRS {
                    black_box(tnj.distance(a, s).unwrap());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("cosine", n), &n, |b, _| {
            b.iter(|| {
                for (a, s) in SENTENCE_PAIRS {
                    black_box(tnc.distance(a, s).unwrap());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("overlap", n), &n, |b, _| {
            b.iter(|| {
                for (a, s) in SENTENCE_PAIRS {
                    black_box(tno.distance(a, s).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_token_ngram_paragraphs(c: &mut Criterion) {
    let paras = load_war_and_peace_paragraphs();
    if paras.len() < 2 {
        return;
    }

    let mut group = c.benchmark_group("token_ngram/paragraphs");
    group.sample_size(10);

    let pairs: Vec<(&str, &str)> = paras
        .windows(2)
        .map(|w| (w[0].as_str(), w[1].as_str()))
        .collect();

    let tj = TokenJaccard { lowercase: true };
    group.bench_function("token_jaccard", |b| {
        b.iter(|| {
            for (a, s) in &pairs {
                black_box(tj.distance(a, s).unwrap());
            }
        })
    });

    let tnj = TokenNgramJaccard {
        n: 2,
        lowercase: true,
    };
    group.bench_function("bigram_jaccard", |b| {
        b.iter(|| {
            for (a, s) in &pairs {
                black_box(tnj.distance(a, s).unwrap());
            }
        })
    });

    let tnc = TokenNgramCosine {
        n: 2,
        lowercase: true,
    };
    group.bench_function("bigram_cosine", |b| {
        b.iter(|| {
            for (a, s) in &pairs {
                black_box(tnc.distance(a, s).unwrap());
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_edit_distance_short,
    bench_hamming,
    bench_edit_distance_paragraphs,
    bench_ngram_similarity,
    bench_phonetic,
    bench_lcs,
    bench_token_ngram_short,
    bench_token_ngram_paragraphs,
);
criterion_main!(benches);
