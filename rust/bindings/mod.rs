//! PyO3 binding modules — exposed to Python via `kaos_nlp_core._rust`.
//!
//! Each submodule registers a child `PyModule` on the root extension module.

#![allow(missing_docs)]

pub(crate) mod algorithms;
pub(crate) mod characters;
pub(crate) mod diff;
pub(crate) mod hashing;
pub(crate) mod lexicon;
pub(crate) mod matching;
pub(crate) mod quality;
pub(crate) mod searcher;
pub(crate) mod segmentation;
pub(crate) mod spans;
pub(crate) mod structure;
pub(crate) mod structures;
pub(crate) mod token_properties;
pub(crate) mod tokenizer;
pub(crate) mod util;
