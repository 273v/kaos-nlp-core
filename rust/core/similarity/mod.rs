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
//! All compute routes through hand-written `std::arch`-intrinsic
//! kernels at [`crate::core::similarity::kernels`] with runtime ISA
//! dispatch:
//!
//! * **x86_64 AVX-512F** — 16-wide f32 FMA when `avx512f` is present
//!   (Skylake-X+, Ice Lake / Sapphire Rapids servers, Zen 4+).
//! * **x86_64 AVX2 + FMA** — 8-wide f32 FMA when `avx2 + fma` are
//!   present (Haswell+, Zen+; this is the universal desktop/server
//!   baseline today).
//! * **aarch64 NEON** — 4-wide f32 FMA via `vfmaq_f32` (mandatory on
//!   every aarch64 wheel target).
//! * **Scalar fallback** — pure-Rust 4-lane FMA loop for any other
//!   target. LLVM still auto-vectorises this on its own; the explicit
//!   intrinsic kernels are just the authoritative path.
//!
//! The kernel pattern (fused single-pass `(dot, ‖a‖², ‖b‖²)`,
//! `rsqrt`+Newton-Raphson finalisation) is ported from NumKong
//! (Apache-2.0); see `kernels.rs` and `docs/design-similarity-simd.md`
//! for the file-and-function-level provenance.
//!
//! Earlier alpha drafts depended on the `numkong` Apache-2.0 crate
//! (vendored C kernels). That dependency was dropped at v0.1.0a5
//! because the vendored C (a) `#include`s `immintrin.h` unconditionally,
//! which MSVC rejects on `aarch64-pc-windows-msvc`, and (b) ships a
//! `DllMain` symbol that collides with `stringzilla`. Both break the
//! Windows wheel matrix. The current hand-rolled `std::arch` kernels
//! port NumKong's algorithm without the vendored-C baggage.
//!
//! Pre-normalised fast path
//! ------------------------
//!
//! Callers that supply unit-norm rows should use the `_normalized`
//! variants ([`cosine_one_to_many_normalized`],
//! [`cosine_adjacent_normalized`], [`cosine_normalized`]): cosine of
//! two unit-norm vectors is their dot product, so we skip both
//! `‖query‖²` and `‖row‖²` and the final `sqrt`. SemanticChunker and
//! ExtractiveRanker in `kaos-nlp-transformers` always feed unit-norm
//! embeddings — they should route through the `_normalized` path.
//!
//! Numerical guarantees
//! --------------------
//!
//! * f32 inputs, f32 SIMD accumulators inside the loop, `f64`
//!   finalisation (sqrt + divide) for bit-stability of the cosine
//!   value across the ISA-dispatch buckets.
//! * Cosine clipped to `[-1.0, 1.0]`.
//! * Zero-norm vectors yield cosine `0.0`.
//! * NaN inputs are not silently propagated — every function returns
//!   `Err` (or skips, for batch) so callers can decide policy.
//! * **Determinism**: same inputs, same dispatched ISA → bit-identical
//!   output. Across ISA buckets the cosine is identical up to f32
//!   rounding (the SIMD reduction tree differs in lane count, so the
//!   sum order differs by ≤ log(lane_width) ULP).
//!
//! See also the design rationale in
//! `docs/design-similarity-simd.md` and the consumer-pattern survey at
//! `kaos-llm-core/docs/summarization-classification-plan.md` (the
//! "Where dense similarity is needed" section).

#![allow(missing_docs)]

pub mod dense;
pub mod kernels;
pub mod mmr;
pub mod topk;

pub use dense::{
    cosine, cosine_adjacent, cosine_adjacent_normalized, cosine_normalized, cosine_one_to_many,
    cosine_one_to_many_normalized, l2_normalize_in_place, SimilarityError,
};
pub use mmr::mmr_select;
pub use topk::top_k_cosine;
