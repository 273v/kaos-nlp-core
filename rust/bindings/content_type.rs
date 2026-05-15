//! PyO3 bindings for `core::content_type`.
//!
//! Exposes a single `py_detect` function that returns a typed dict with
//! `mime_type` / `extension` / `group` keys.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};

use crate::core::content_type::detect;

/// Detect the content type of a bytes-like object via magic-bytes signature.
#[pyfunction]
#[pyo3(signature = (data,))]
fn py_detect<'py>(py: Python<'py>, data: &Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyDict>> {
    let bytes = data.as_bytes();
    // GIL release isn't worth the syscall here: `infer::get` is a
    // bounded read of the header (typically <128 bytes) — single-digit
    // microseconds, far below the GIL-release overhead.
    let result = detect(bytes);
    let dict = PyDict::new(py);
    dict.set_item("mime_type", result.mime_type)?;
    dict.set_item("extension", result.extension)?;
    dict.set_item("group", result.group)?;
    Ok(dict)
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "content_type")?;
    m.add_function(wrap_pyfunction!(py_detect, &m)?)?;
    parent.add_submodule(&m)?;
    // Register the submodule in `sys.modules` so
    // ``from kaos_nlp_core._rust.content_type import ...`` resolves —
    // matches the convention used by every other binding in this crate.
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.content_type", &m)?;
    Ok(())
}
