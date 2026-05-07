//! Benchmarks for fuzzy hashing: MinHash/LSH, CTPH.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use _rust::core::ctph::{TokenCTPH, CTPH};
use _rust::core::minhash::{self, MinHashIndex, MinHasher};

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

fn load_text() -> String {
    std::fs::read_to_string("tests/fixtures/war_and_peace.txt")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. ".repeat(10000))
}

fn random_strings(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("item_{i}")).collect()
}

// =============================================================================
// MinHash benchmarks
// =============================================================================

fn bench_minhash_hash_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash/hash_set");
    let hasher = MinHasher::new(128);

    let items_100 = random_strings(100);
    group.bench_function("100_items", |b| {
        b.iter(|| {
            let refs: Vec<&str> = items_100.iter().map(|s| s.as_str()).collect();
            black_box(hasher.hash_set(refs.iter().copied()))
        })
    });

    let items_10k = random_strings(10_000);
    group.bench_function("10000_items", |b| {
        b.iter(|| {
            let refs: Vec<&str> = items_10k.iter().map(|s| s.as_str()).collect();
            black_box(hasher.hash_set(refs.iter().copied()))
        })
    });

    group.finish();
}

fn bench_minhash_char_shingles(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash/char_shingles");
    let hasher = MinHasher::new(128);

    let short_text = "The quick brown fox jumps over the lazy dog.".repeat(3);
    group.bench_function("short", |b| {
        b.iter(|| black_box(hasher.hash_char_shingles(&short_text, 3)))
    });

    let war_peace = load_text();
    group.bench_function("war_peace", |b| {
        b.iter(|| black_box(hasher.hash_char_shingles(&war_peace, 5)))
    });

    group.finish();
}

fn bench_minhash_token_shingles(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash/token_shingles");
    let hasher = MinHasher::new(128);

    let tokens_100: Vec<String> = (0..100).map(|i| format!("token_{i}")).collect();
    group.bench_function("100_tokens", |b| {
        b.iter(|| {
            let refs: Vec<&str> = tokens_100.iter().map(|s| s.as_str()).collect();
            black_box(hasher.hash_token_shingles(&refs, 2))
        })
    });

    group.finish();
}

fn bench_minhash_jaccard(c: &mut Criterion) {
    let hasher = MinHasher::new(128);
    let sig1 = hasher.hash_set(["a", "b", "c", "d", "e"].iter().copied());
    let sig2 = hasher.hash_set(["a", "b", "c", "f", "g"].iter().copied());

    c.bench_function("minhash/jaccard", |b| {
        b.iter(|| black_box(sig1.jaccard(&sig2)))
    });
}

// =============================================================================
// LSH benchmarks
// =============================================================================

fn bench_lsh_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh/insert");
    let hasher = MinHasher::new(128);

    // Pre-generate signatures
    let sigs_1k: Vec<_> = (0..1_000)
        .map(|i| {
            let items: Vec<String> = (0..20).map(|j| format!("term_{i}_{j}")).collect();
            let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
            hasher.hash_set(refs.iter().copied())
        })
        .collect();

    group.bench_function("1000_docs", |b| {
        b.iter(|| {
            let mut index = MinHashIndex::with_threshold(128, 0.5);
            for (i, sig) in sigs_1k.iter().enumerate() {
                let _ = index.insert(i as u32, sig);
            }
            black_box(index.len())
        })
    });

    let sigs_10k: Vec<_> = (0..10_000)
        .map(|i| {
            let items: Vec<String> = (0..20).map(|j| format!("term_{i}_{j}")).collect();
            let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
            hasher.hash_set(refs.iter().copied())
        })
        .collect();

    group.bench_function("10000_docs", |b| {
        b.iter(|| {
            let mut index = MinHashIndex::with_threshold(128, 0.5);
            for (i, sig) in sigs_10k.iter().enumerate() {
                let _ = index.insert(i as u32, sig);
            }
            black_box(index.len())
        })
    });

    group.finish();
}

fn bench_lsh_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh/query");
    let hasher = MinHasher::new(128);

    // Build a 10K-doc index
    let mut index = MinHashIndex::with_threshold(128, 0.5);
    for i in 0..10_000u32 {
        let items: Vec<String> = (0..20).map(|j| format!("term_{i}_{j}")).collect();
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        let sig = hasher.hash_set(refs.iter().copied());
        let _ = index.insert(i, &sig);
    }

    let query_sig = hasher.hash_set(["term_0_0", "term_0_1", "term_0_2"].iter().copied());

    group.bench_function("candidates_10k", |b| {
        b.iter(|| black_box(index.query_candidates(&query_sig).unwrap()))
    });

    group.bench_function("above_threshold_10k", |b| {
        b.iter(|| black_box(index.query_above_threshold(&query_sig, 0.3)))
    });

    group.finish();
}

fn bench_find_duplicates(c: &mut Criterion) {
    let hasher = MinHasher::new(128);
    let words = load_words();

    // Create 1000 docs from War & Peace sections
    let chunk_size = words.len() / 1000;
    let docs: Vec<(u32, Vec<&str>)> = (0..1000)
        .map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(words.len());
            let tokens: Vec<&str> = words[start..end].iter().map(|s| s.as_str()).collect();
            (i as u32, tokens)
        })
        .collect();

    c.bench_function("find_duplicates/1000_docs", |b| {
        b.iter(|| black_box(minhash::find_duplicates(&hasher, &docs, 2, 0.5)))
    });
}

// =============================================================================
// CTPH benchmarks
// =============================================================================

fn bench_ctph_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("ctph/compute");

    let short = b"The quick brown fox jumps over the lazy dog.".to_vec();
    let ctph = CTPH::new(64, 8, 4);
    group.bench_function("short_44B", |b| b.iter(|| black_box(ctph.compute(&short))));

    let medium = vec![0x42u8; 64 * 1024]; // 64KB
    group.bench_function("medium_64KB", |b| {
        b.iter(|| black_box(ctph.compute(&medium)))
    });

    let war_peace_text = load_text();
    let large = war_peace_text.as_bytes();
    group.bench_function("war_peace", |b| b.iter(|| black_box(ctph.compute(large))));

    group.finish();
}

fn bench_ctph_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("ctph/similarity");
    let ctph = CTPH::new(64, 8, 4);
    let war_peace_text = load_text();
    let base = &war_peace_text[..10000.min(war_peace_text.len())];
    let d1 = ctph.hash_str(base);
    let mid = base.len() / 2;
    let modified = format!(
        "{}{}{}",
        &base[..mid - 250],
        "X".repeat(500),
        &base[mid + 250..]
    );
    let d2 = ctph.hash_str(&modified);

    group.bench_function("block", |b| b.iter(|| black_box(d1.similarity(&d2))));

    group.bench_function("piece", |b| b.iter(|| black_box(d1.piece_similarity(&d2))));

    group.finish();
}

fn bench_token_ctph(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_ctph/compute");
    let ctph = TokenCTPH::new(4, 8);

    let tokens_100: Vec<i64> = (0..100).collect();
    group.bench_function("100_tokens", |b| {
        b.iter(|| black_box(ctph.compute(&tokens_100)))
    });

    let tokens_10k: Vec<i64> = (0..10_000).collect();
    group.bench_function("10000_tokens", |b| {
        b.iter(|| black_box(ctph.compute(&tokens_10k)))
    });

    group.finish();
}

fn bench_rolling_hash(c: &mut Criterion) {
    use _rust::core::ctph::RollingHash;
    c.bench_function("rolling_hash/update_1m", |b| {
        b.iter(|| {
            let mut rh: RollingHash<u32> = RollingHash::new(64);
            for i in 0..1_000_000u32 {
                rh.update(i);
            }
            black_box(rh.hash())
        })
    });
}

criterion_group!(
    benches,
    bench_minhash_hash_set,
    bench_minhash_char_shingles,
    bench_minhash_token_shingles,
    bench_minhash_jaccard,
    bench_lsh_insert,
    bench_lsh_query,
    bench_find_duplicates,
    bench_ctph_compute,
    bench_ctph_similarity,
    bench_token_ctph,
    bench_rolling_hash,
);
criterion_main!(benches);
