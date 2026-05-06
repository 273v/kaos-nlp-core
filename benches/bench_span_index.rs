//! Benchmarks for SpanIndex.
//!
//! Three workloads:
//!
//! - `bulk_build` — sort + end_max compute over N spans.
//! - `containing(offset)` — point-in-interval query at increasing N.
//! - `overlapping(start, end)` — range query at increasing N.
//!
//! All tests use a deterministic synthetic span generator. Spans are
//! laid out as a mix of short (length 5–20) and longer (length 100–500)
//! intervals over a 0..=1_000_000 coordinate range so the engulfing-
//! interval pathology is exercised on the wider workloads.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use _rust::core::structures::{LabeledSpan, SpanIndex};

fn synthetic_spans(n: usize) -> Vec<LabeledSpan> {
    let mut spans = Vec::with_capacity(n);
    let mut rng = 0x1234_5678u64;
    for _ in 0..n {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let start = ((rng >> 1) as u32) % 1_000_000;
        let len_kind = (rng >> 33) & 0b11;
        let len = match len_kind {
            0 => 5 + ((rng >> 7) as u32) % 15,
            1 => 50 + ((rng >> 7) as u32) % 200,
            2 => 500 + ((rng >> 7) as u32) % 1500,
            _ => 5_000 + ((rng >> 7) as u32) % 20_000,
        };
        let label = (rng >> 50) as u32 % 16;
        let score = ((rng >> 40) as u32 % 1000) as f32 / 1000.0;
        spans.push(LabeledSpan {
            label,
            start,
            end: start.saturating_add(len),
            score,
        });
    }
    spans
}

fn bench_bulk_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_index/bulk_build");
    for &n in &[1_000usize, 10_000, 100_000] {
        let spans = synthetic_spans(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("n", n), &spans, |b, spans| {
            b.iter(|| {
                let idx = SpanIndex::bulk_build(black_box(spans.clone())).unwrap();
                black_box(idx);
            });
        });
    }
    group.finish();
}

fn bench_containing(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_index/containing");
    for &n in &[1_000usize, 10_000, 100_000] {
        let spans = synthetic_spans(n);
        let mut idx = SpanIndex::bulk_build(spans).unwrap();
        idx.freeze();
        group.bench_with_input(BenchmarkId::new("n", n), &idx, |b, _| {
            // Cycle through 64 query offsets so we measure cold-cache + warm-cache.
            let mut i: u32 = 0;
            b.iter(|| {
                i = i.wrapping_add(7919);
                let offset = i % 1_000_000;
                let mut local = idx.clone();
                let hits = local.containing(black_box(offset));
                black_box(hits);
            });
        });
    }
    group.finish();
}

fn bench_overlapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_index/overlapping");
    for &n in &[1_000usize, 10_000, 100_000] {
        let spans = synthetic_spans(n);
        let mut idx = SpanIndex::bulk_build(spans).unwrap();
        idx.freeze();
        group.bench_with_input(BenchmarkId::new("n", n), &idx, |b, _| {
            let mut i: u32 = 0;
            b.iter(|| {
                i = i.wrapping_add(31337);
                let qs = i % 1_000_000;
                let mut local = idx.clone();
                let hits = local.overlapping(black_box(qs), black_box(qs + 1000));
                black_box(hits);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bulk_build,
    bench_containing,
    bench_overlapping
);
criterion_main!(benches);
