//! kaos-nlp-core: High-performance NLP primitives for the Kelvin Agentic OS.
//!
//! This crate provides:
//! - String distance and similarity algorithms (edit, phonetic, sequence, n-gram)
//! - Pattern matching (SIMD substring, Aho-Corasick, regex, FST)
//! - Data structures (vocabularies, inverted index)
//!
//! The Rust core is exposed to Python via PyO3.

// Many public core items are only consumed by the bindings layer, not within
// the crate itself. Allow dead_code at the crate root to avoid false positives.
#![allow(dead_code)]

#[cfg(feature = "python")]
mod bindings;
pub mod core;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// The root Python module `kaos_nlp_core._rust`.
#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "_rust")]
fn kaos_nlp_core_rust(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    bindings::algorithms::register_module(m)?;
    bindings::characters::register_module(m)?;
    bindings::hashing::register_module(m)?;
    bindings::lexicon::register_module(m)?;
    bindings::matching::register_module(m)?;
    bindings::quality::register_module(m)?;
    bindings::searcher::register_module(m)?;
    bindings::segmentation::register_module(m)?;
    bindings::spans::register_module(m)?;
    bindings::structure::register_module(m)?;
    bindings::structures::register_module(m)?;
    bindings::token_properties::register_module(m)?;
    bindings::tokenizer::register_module(m)?;

    // Set __path__ so Python treats this as a package.
    m.setattr("__path__", pyo3::types::PyList::empty(py))?;
    m.setattr("__package__", "kaos_nlp_core._rust")?;

    Ok(())
}
