//! Document chunking — pure-Rust kernels for the deterministic packers.
//!
//! The deterministic chunkers in
//! ``kaos_nlp_core.chunking.{fixed,sentence,paragraph,section,hierarchical}``
//! all share one operation: walk an ordered sequence of *units*
//! (sentences, paragraphs, lines, fixed-window slices) and group
//! consecutive units into chunks that stay under a token budget,
//! optionally retaining a tail of overlapping units in the next
//! chunk for context preservation.
//!
//! Prior to this module that loop lived in Python (`_pack.py`'s
//! `pack_units` helper). The segmentation + tokenization the loop
//! consumes are already Rust (Punkt segmenter, `kaos_nlp_core.tokenizer`),
//! so the Python loop was the only remaining bottleneck — and the
//! consumer survey identified it as the highest-ROI port target.
//!
//! The Rust kernel returns raw *grouping records* (offset pairs +
//! unit-index ranges + token-count sums) rather than `Chunk` Python
//! objects. The Python wrapper does the final slice + metadata-dict
//! merge + `Chunk` construction — that step is dominated by Python
//! dict allocations and string slicing, neither of which benefits
//! from being in Rust. This factoring matches the NumKong /
//! similarity pattern (Rust returns arrays, Python composes).
//!
//! Public surface: `pack_units` (generic packer) and `PackedGroup`
//! (result record). The five concrete chunker variants in
//! `kaos_nlp_core.chunking.*` differ only in the unit-segmentation
//! step that runs *before* this kernel; they all funnel into one
//! Rust loop.

#![allow(missing_docs)]

pub mod pack;
pub mod semantic;

pub use pack::{pack_units, PackError, PackedGroup};
pub use semantic::{semantic_pack, SemanticPackError};
