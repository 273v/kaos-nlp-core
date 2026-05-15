//! PyO3 bindings for the aggregation kernels.
//!
//! Module: ``kaos_nlp_core._rust.aggregation``. Exposes the six
//! primitives in :mod:`crate::core::aggregation` (vote, majority,
//! union, intersection, weighted_single, weighted_multi,
//! max_score_single, max_score_multi). Input contract is the CSR-style
//! ragged-array layout documented in the core module — the Python
//! wrapper interns label names to u32 ids once per call and passes
//! the resulting arrays here.
//!
//! Return shapes:
//!
//! * Single-label kernels (vote, majority, weighted_single,
//!   max_score_single) return ``Option<u32>`` → Python ``int | None``.
//! * Multi-label kernels (union, intersection, weighted_multi,
//!   max_score_multi) return a 1-D ``uint32`` numpy array of
//!   qualifying ids in ascending order.
//!
//! All work runs with the GIL released via ``py.detach``.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::aggregation::{
    intersection as core_intersection, majority as core_majority,
    max_score_multi as core_max_score_multi, max_score_single as core_max_score_single,
    union as core_union, vote as core_vote, weighted_multi as core_weighted_multi,
    weighted_single as core_weighted_single, AggregationError,
};

fn map_err(e: AggregationError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pyfunction]
#[pyo3(signature = (flat_ids, chunk_offsets, n_labels))]
fn vote(
    py: Python<'_>,
    flat_ids: PyReadonlyArray1<'_, u32>,
    chunk_offsets: PyReadonlyArray1<'_, u32>,
    n_labels: u32,
) -> PyResult<Option<u32>> {
    let ids = flat_ids.as_slice()?;
    let off = chunk_offsets.as_slice()?;
    py.detach(|| core_vote(ids, off, n_labels)).map_err(map_err)
}

#[pyfunction]
#[pyo3(signature = (flat_ids, chunk_offsets, n_labels, threshold))]
fn majority(
    py: Python<'_>,
    flat_ids: PyReadonlyArray1<'_, u32>,
    chunk_offsets: PyReadonlyArray1<'_, u32>,
    n_labels: u32,
    threshold: f64,
) -> PyResult<Option<u32>> {
    let ids = flat_ids.as_slice()?;
    let off = chunk_offsets.as_slice()?;
    py.detach(|| core_majority(ids, off, n_labels, threshold))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(signature = (flat_ids, chunk_offsets, n_labels))]
fn union<'py>(
    py: Python<'py>,
    flat_ids: PyReadonlyArray1<'_, u32>,
    chunk_offsets: PyReadonlyArray1<'_, u32>,
    n_labels: u32,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let ids = flat_ids.as_slice()?;
    let off = chunk_offsets.as_slice()?;
    let out = py
        .detach(|| core_union(ids, off, n_labels))
        .map_err(map_err)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (flat_ids, chunk_offsets, n_labels))]
fn intersection<'py>(
    py: Python<'py>,
    flat_ids: PyReadonlyArray1<'_, u32>,
    chunk_offsets: PyReadonlyArray1<'_, u32>,
    n_labels: u32,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let ids = flat_ids.as_slice()?;
    let off = chunk_offsets.as_slice()?;
    let out = py
        .detach(|| core_intersection(ids, off, n_labels))
        .map_err(map_err)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (flat_ids, chunk_offsets, weights, n_labels, threshold))]
fn weighted_single(
    py: Python<'_>,
    flat_ids: PyReadonlyArray1<'_, u32>,
    chunk_offsets: PyReadonlyArray1<'_, u32>,
    weights: PyReadonlyArray1<'_, f64>,
    n_labels: u32,
    threshold: f64,
) -> PyResult<Option<u32>> {
    let ids = flat_ids.as_slice()?;
    let off = chunk_offsets.as_slice()?;
    let w = weights.as_slice()?;
    py.detach(|| core_weighted_single(ids, off, w, n_labels, threshold))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(signature = (flat_ids, chunk_offsets, weights, n_labels, threshold))]
fn weighted_multi<'py>(
    py: Python<'py>,
    flat_ids: PyReadonlyArray1<'_, u32>,
    chunk_offsets: PyReadonlyArray1<'_, u32>,
    weights: PyReadonlyArray1<'_, f64>,
    n_labels: u32,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let ids = flat_ids.as_slice()?;
    let off = chunk_offsets.as_slice()?;
    let w = weights.as_slice()?;
    let out = py
        .detach(|| core_weighted_multi(ids, off, w, n_labels, threshold))
        .map_err(map_err)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (flat_ids, flat_scores, n_labels, threshold=None))]
fn max_score_single(
    py: Python<'_>,
    flat_ids: PyReadonlyArray1<'_, u32>,
    flat_scores: PyReadonlyArray1<'_, f64>,
    n_labels: u32,
    threshold: Option<f64>,
) -> PyResult<Option<u32>> {
    let ids = flat_ids.as_slice()?;
    let scores = flat_scores.as_slice()?;
    py.detach(|| core_max_score_single(ids, scores, n_labels, threshold))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(signature = (flat_ids, flat_scores, n_labels, threshold=None))]
fn max_score_multi<'py>(
    py: Python<'py>,
    flat_ids: PyReadonlyArray1<'_, u32>,
    flat_scores: PyReadonlyArray1<'_, f64>,
    n_labels: u32,
    threshold: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let ids = flat_ids.as_slice()?;
    let scores = flat_scores.as_slice()?;
    let out = py
        .detach(|| core_max_score_multi(ids, scores, n_labels, threshold))
        .map_err(map_err)?;
    Ok(out.into_pyarray(py))
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "aggregation")?;
    m.add_function(wrap_pyfunction!(vote, &m)?)?;
    m.add_function(wrap_pyfunction!(majority, &m)?)?;
    m.add_function(wrap_pyfunction!(union, &m)?)?;
    m.add_function(wrap_pyfunction!(intersection, &m)?)?;
    m.add_function(wrap_pyfunction!(weighted_single, &m)?)?;
    m.add_function(wrap_pyfunction!(weighted_multi, &m)?)?;
    m.add_function(wrap_pyfunction!(max_score_single, &m)?)?;
    m.add_function(wrap_pyfunction!(max_score_multi, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.aggregation", &m)?;
    Ok(())
}
