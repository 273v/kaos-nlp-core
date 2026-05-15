//! Pure-Rust algorithm/data-structure core (no PyO3 dependency).
//!
//! Items here are public so the `bindings/` layer (and downstream Rust
//! consumers, if any) can call them directly. Per-module documentation
//! lives inside each submodule.

#![allow(missing_docs)]

pub mod aggregation;
pub mod algorithms;
pub mod characters;
pub mod chunking;
pub mod content_type;
pub mod ctph;
pub mod diff;
pub mod lexicon;
pub mod matching;
pub mod minhash;
pub mod quality;
pub mod searcher;
pub mod segmentation;
pub mod similarity;
pub mod structure;
pub mod structures;
pub mod token_properties;
pub mod tokenizer;
