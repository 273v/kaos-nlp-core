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
// Core five — non-negotiable lints from docs/oss/30-rust-packaging/clippy-and-quality.md.
// `missing_docs` is opted out at the submodule level (see `core/mod.rs` and
// `bindings/mod.rs`) for now; backfilling per-item docs is tracked under
// KNC-008-followup.
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![warn(rust_2021_compatibility)]
#![warn(unreachable_pub)]
#![warn(unused_qualifications)]
// Selected `clippy::pedantic` lints. Audit-01 follow-up: these fire 360+ times
// across the existing core (numeric casts on lengths/indices, doc-only `# Errors`
// / `# Panics` sections, shared-ref args). They are downgraded to `allow` so the
// CI gate still passes; flip back to `warn` and clean up incrementally.
// TODO(KNC-008-followup): re-enable each of these lints and fix the hits.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::redundant_clone)]

#[cfg(feature = "python")]
mod bindings;
pub mod core;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// The root Python module `kaos_nlp_core._rust`.
///
/// Declared `gil_used = false` per audit KNC-008 to declare free-threaded
/// Python compatibility. Every `#[pyclass]` registered below has been
/// audited to be `Send + Sync` (no `RefCell`/`Cell`/`Rc` in pyclass fields;
/// stateful classes hold owned types or `Arc<T>` over plain data).
#[cfg(feature = "python")]
#[pymodule(gil_used = false)]
#[pyo3(name = "_rust")]
fn kaos_nlp_core_rust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    bindings::aggregation::register_module(m)?;
    bindings::algorithms::register_module(m)?;
    bindings::characters::register_module(m)?;
    bindings::ctfidf::register_module(m)?;
    bindings::chunking::register_module(m)?;
    bindings::content_type::register_module(m)?;
    bindings::diff::register_module(m)?;
    bindings::hashing::register_module(m)?;
    bindings::lexicon::register_module(m)?;
    bindings::matching::register_module(m)?;
    bindings::quality::register_module(m)?;
    bindings::searcher::register_module(m)?;
    bindings::segmentation::register_module(m)?;
    bindings::similarity::register_module(m)?;
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
