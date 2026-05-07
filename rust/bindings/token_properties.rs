//! PyO3 bindings for token property classification.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use crate::core::token_properties;

fn flags_to_dict(
    py: Python<'_>,
    flags: &token_properties::TokenPropertyFlags,
) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("is_letter_word", flags.is_letter_word)?;
    d.set_item("is_uppercase_word", flags.is_uppercase_word)?;
    d.set_item("is_lowercase_word", flags.is_lowercase_word)?;
    d.set_item("is_mixed_case_word", flags.is_mixed_case_word)?;
    d.set_item("is_title_case_word", flags.is_title_case_word)?;
    d.set_item("is_numeric_word", flags.is_numeric_word)?;
    d.set_item("is_alphanumeric_word", flags.is_alphanumeric_word)?;
    d.set_item("starts_with_digit", flags.starts_with_digit)?;
    d.set_item("has_punctuation", flags.has_punctuation)?;
    d.set_item("has_dash", flags.has_dash)?;
    d.set_item("is_hyphenated", flags.is_hyphenated)?;
    d.set_item("is_abbreviation", flags.is_abbreviation)?;
    d.set_item("has_emoji", flags.has_emoji)?;
    d.set_item("is_symbolic_word", flags.is_symbolic_word)?;
    d.set_item("ends_with_terminal", flags.ends_with_terminal)?;
    Ok(d.into())
}

#[pyfunction]
fn classify_token(py: Python<'_>, token: &str) -> PyResult<Py<PyDict>> {
    let flags = token_properties::classify_token(token);
    flags_to_dict(py, &flags)
}

#[pyfunction]
fn classify_tokens(py: Python<'_>, tokens: Vec<String>) -> PyResult<Py<PyList>> {
    let results = py.detach(|| {
        tokens
            .iter()
            .map(|token| token_properties::classify_token(token))
            .collect::<Vec<_>>()
    });
    let list = PyList::empty(py);
    for flags in &results {
        list.append(flags_to_dict(py, flags)?)?;
    }
    Ok(list.into())
}

#[pyfunction]
fn is_letter_word(token: &str) -> bool {
    token_properties::is_letter_word(token)
}

#[pyfunction]
fn is_uppercase_word(token: &str) -> bool {
    token_properties::is_uppercase_word(token)
}

#[pyfunction]
fn is_lowercase_word(token: &str) -> bool {
    token_properties::is_lowercase_word(token)
}

#[pyfunction]
fn is_title_case_word(token: &str) -> bool {
    token_properties::is_title_case_word(token)
}

#[pyfunction]
fn is_numeric_word(token: &str) -> bool {
    token_properties::is_numeric_word(token)
}

#[pyfunction]
fn is_alphanumeric_word(token: &str) -> bool {
    token_properties::is_alphanumeric_word(token)
}

#[pyfunction]
fn is_abbreviation(token: &str) -> bool {
    token_properties::is_abbreviation(token)
}

#[pyfunction]
fn has_emoji(token: &str) -> bool {
    token_properties::has_emoji(token)
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "token_properties")?;

    m.add_function(wrap_pyfunction!(classify_token, &m)?)?;
    m.add_function(wrap_pyfunction!(classify_tokens, &m)?)?;
    m.add_function(wrap_pyfunction!(is_letter_word, &m)?)?;
    m.add_function(wrap_pyfunction!(is_uppercase_word, &m)?)?;
    m.add_function(wrap_pyfunction!(is_lowercase_word, &m)?)?;
    m.add_function(wrap_pyfunction!(is_title_case_word, &m)?)?;
    m.add_function(wrap_pyfunction!(is_numeric_word, &m)?)?;
    m.add_function(wrap_pyfunction!(is_alphanumeric_word, &m)?)?;
    m.add_function(wrap_pyfunction!(is_abbreviation, &m)?)?;
    m.add_function(wrap_pyfunction!(has_emoji, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.token_properties", &m)?;

    Ok(())
}
