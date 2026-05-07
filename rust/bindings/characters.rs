//! PyO3 bindings for character classification.
//!
//! Re-exports the ICU4X-backed character predicates from
//! `kaos_nlp_core::core::characters` as Python free functions. These
//! were previously only reachable through the tokenizer's internal
//! call sites; exposing them lets quality / analytics consumers reuse
//! the same classification rules.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::core::characters::CharacterProperties;

#[pyfunction]
fn is_letter(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_letter)
        .unwrap_or(false)
}

#[pyfunction]
fn is_number(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_number)
        .unwrap_or(false)
}

#[pyfunction]
fn is_punctuation(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_punctuation)
        .unwrap_or(false)
}

#[pyfunction]
fn is_symbol(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_symbol)
        .unwrap_or(false)
}

#[pyfunction]
fn is_whitespace(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_whitespace)
        .unwrap_or(false)
}

#[pyfunction]
fn is_uppercase(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_uppercase)
        .unwrap_or(false)
}

#[pyfunction]
fn is_lowercase(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_lowercase)
        .unwrap_or(false)
}

#[pyfunction]
fn is_terminal_punctuation(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_terminal_punctuation)
        .unwrap_or(false)
}

#[pyfunction]
fn is_internal_punctuation(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_internal_punctuation)
        .unwrap_or(false)
}

#[pyfunction]
fn is_newline(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_newline)
        .unwrap_or(false)
}

#[pyfunction]
fn is_dash(ch: &str) -> bool {
    first_char(ch)
        .map(CharacterProperties::is_dash)
        .unwrap_or(false)
}

#[inline]
fn first_char(s: &str) -> Option<char> {
    s.chars().next()
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "characters")?;
    m.add_function(wrap_pyfunction!(is_letter, &m)?)?;
    m.add_function(wrap_pyfunction!(is_number, &m)?)?;
    m.add_function(wrap_pyfunction!(is_punctuation, &m)?)?;
    m.add_function(wrap_pyfunction!(is_symbol, &m)?)?;
    m.add_function(wrap_pyfunction!(is_whitespace, &m)?)?;
    m.add_function(wrap_pyfunction!(is_uppercase, &m)?)?;
    m.add_function(wrap_pyfunction!(is_lowercase, &m)?)?;
    m.add_function(wrap_pyfunction!(is_terminal_punctuation, &m)?)?;
    m.add_function(wrap_pyfunction!(is_internal_punctuation, &m)?)?;
    m.add_function(wrap_pyfunction!(is_newline, &m)?)?;
    m.add_function(wrap_pyfunction!(is_dash, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.characters", &m)?;
    Ok(())
}
