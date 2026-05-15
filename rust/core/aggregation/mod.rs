//! Deterministic label-aggregation kernels.
//!
//! Pure-Rust ports of the six primitives in
//! ``kaos_nlp_core/aggregation/__init__.py`` (vote, majority, union,
//! intersection, weighted, max_score). Together they cover the
//! aggregation modes used by ``kaos_llm_core.composition.aggregate``
//! to combine per-chunk classifications into a document-level
//! decision.
//!
//! Input contract
//! --------------
//!
//! Kernels take per-chunk label sets as a *ragged* (CSR-style)
//! array:
//!
//! * `flat_ids: &[u32]` — concatenation of every chunk's label
//!   ids, in chunk order.
//! * `chunk_offsets: &[u32]` — length ``n_chunks + 1``; chunk ``c``
//!   occupies ``flat_ids[chunk_offsets[c]..chunk_offsets[c + 1]]``.
//! * `n_labels: u32` — exclusive upper bound on the label ids.
//!
//! The Python wrapper layer interns string label names to u32 ids
//! once per call. **Ids are assigned in order of first appearance**,
//! which means the tie-break rule from the Python implementation
//! (lowest first-seen index wins) maps trivially to "lowest id
//! wins" in the Rust kernels. This is the only invariant the
//! kernels rely on for determinism.
//!
//! Determinism
//! -----------
//!
//! All kernels are pure functions of their inputs and free of
//! hash-randomization side-channels (no `HashSet` / `HashMap` on
//! string keys). Per-chunk duplicates inside a single chunk are
//! collapsed via a tiny `seen` bitmap built in O(unique-ids-per-call)
//! space.

#![allow(missing_docs)]

pub mod kernels;

pub use kernels::{
    intersection, majority, max_score_multi, max_score_single, union, vote, weighted_multi,
    weighted_single, AggregationError,
};
