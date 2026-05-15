//! Dense-vector similarity primitives.
//!
//! The Rust-backed companion to the existing
//! [`crate::core::structures::similarity_matrix`] (sparse, CSR) and
//! [`crate::core::structures::sparse_term_matrix`] (sparse TF-IDF).
//!
//! What this module owns
//! ---------------------
//!
//! Dense float-vector kernels for the patterns exercised by downstream
//! retrieval / chunking / reranking code in the KAOS stack:
//!
//! * **Single pair** — `cosine(a, b)` for two `&[f32]` vectors of equal
//!   length. Used by per-cache-lookup paths
//!   ([`kaos_llm_core.cache.semantic`]).
//! * **One-vs-many** — `cosine_one_to_many(query, matrix)`. Used by
//!   embedding retrievers and document/corpus indexing.
//! * **Pairwise adjacent** — `cosine_adjacent(matrix)`. Used by the
//!   semantic chunker, which compares each row to its immediate
//!   neighbour.
//! * **Top-k** — `top_k_cosine(query, matrix, k)`. Used by every
//!   retriever (`numpy.argpartition` equivalent, but argpartition is
//!   not the bottleneck — the cosine is).
//! * **MMR selection** — `mmr_select(matrix, scores, k, lambda)`.
//!   Used by extractive-summarisation reranking when diversity matters.
//!
//! What this module does **not** own
//! ---------------------------------
//!
//! * Sparse cosine / Euclidean / Manhattan — see
//!   [`crate::core::structures::similarity_matrix::SimilarityMatrix`]
//!   and [`crate::core::structures::sparse_term_matrix::SparseTermMatrix`].
//!   Those types are the right home for sparse TF / TF-IDF /
//!   term-vector workloads; they pre-compute norms and store an
//!   upper-triangular distance matrix.
//! * Approximate nearest neighbour (HNSW, IVF, etc.). This module is
//!   exact / brute-force. Up to ~100k rows on commodity hardware the
//!   exact path is well under a millisecond per query thanks to
//!   NumKong's SIMD dispatch; if a consumer ever needs ANN it lives in
//!   a separate (likely sibling) module so this one stays small and
//!   deterministic.
//! * Token-frequency cosine over `HashMap<String, u64>` — that path
//!   already exists at [`crate::core::algorithms::ngram::cosine_from_freqs`]
//!   for character / word n-grams.
//!
//! Backend
//! -------
//!
//! All compute paths route through the portable f32-with-f64-accumulator
//! kernels in [`crate::core::similarity::kernels`]. The kernels use 8
//! parallel `f64` accumulators per loop chunk so LLVM
//! auto-vectorises them to AVX2 (Haswell/Skylake), NEON (aarch64),
//! and the scalar fallback on the rest -- no external SIMD crate is
//! pulled in, and Windows wheels (both `x86_64-pc-windows-msvc` and
//! `aarch64-pc-windows-msvc`) build cleanly. The 8-lane reduction
//! also gives a pairwise-style `O(log N)` error bound on the
//! accumulator, which keeps cosine within machine f32 epsilon for
//! the dimensionalities we serve in production (256-1536).
//!
//! Earlier alpha drafts used the `numkong` Apache-2.0 crate (vendored
//! C kernels with runtime AVX-512 / SVE dispatch). That dependency
//! was dropped at v0.1.0a5 because the vendored C
//! (1) `#include`s `immintrin.h` unconditionally, which MSVC rejects
//! on `aarch64-pc-windows-msvc`, and
//! (2) ships a `DllMain` symbol that collides with `stringzilla`
//! (which we already link for SIMD substring matching).
//! Both break the Windows wheel matrix. The portable-SIMD path here
//! follows the *design* of NumKong's compensated kernel without
//! taking the vendored-C dependency.
//!
//! Numerical guarantees
//! --------------------
//!
//! * **f32 cosine accumulates in f64** via 8 parallel accumulators
//!   that LLVM auto-vectorises (see [`crate::core::similarity::kernels`]).
//!   The 8-lane reduction gives a pairwise-style `O(log N)` error
//!   bound -- comparable to Neumaier compensation for the
//!   dimensionalities we serve, without the branch that compensated
//!   accumulators introduce inside the hot loop. End-to-end relative
//!   error stays inside machine f32 epsilon (~`1e-7`).
//! * **NaN inputs** are not silently propagated — every function
//!   returns `Err` (or skips, for batch) so callers can decide policy.
//! * **Determinism** — same inputs, same SIMD lane width → same bits.
//!   No reduction-tree reordering across runs, so cosine of a known
//!   pair round-trips bit-exactly across the wheel matrix.
//!
//! See also the design rationale in
//! `docs/standards/rust-pyo3-design-and-architecture.md` and the
//! consumer-pattern survey at
//! `kaos-llm-core/docs/summarization-classification-plan.md` (the
//! "Where dense similarity is needed" section).

#![allow(missing_docs)]

pub mod dense;
pub mod kernels;
pub mod mmr;
pub mod topk;

pub use dense::{
    cosine, cosine_adjacent, cosine_one_to_many, l2_normalize_in_place, SimilarityError,
};
pub use mmr::mmr_select;
pub use topk::top_k_cosine;
