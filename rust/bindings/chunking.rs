//! PyO3 bindings for the chunking kernel
//! (`crate::core::chunking::pack_units`).
//!
//! Exposes a single function:
//!
//! ```text
//! kaos_nlp_core._rust.chunking
//!   └── pack_units(starts, ends, token_counts, max_tokens, overlap_units)
//!         -> (group_starts, group_ends, group_unit_starts, group_unit_ends, group_token_sums)
//! ```
//!
//! Five parallel ``uint32`` arrays come back, one element per group.
//! The Python wrapper materialises ``Chunk`` instances from these
//! (slicing the source text, computing the final per-chunk
//! token_count, merging metadata). The Rust kernel never touches a
//! Python string or dict; the loop runs GIL-released.
//!
//! Input contract:
//!
//! * `starts`, `ends`, `token_counts` — 1-D contiguous ``uint32``
//!   numpy arrays, same length.
//! * `max_tokens` — `u32`, must be > 0.
//! * `overlap_units` — `u32`, may be 0.
//!
//! The bindings here match the existing convention in
//! ``bindings::similarity`` and ``bindings::structures``: build a
//! small `PyModule`, register pyfunctions, attach to ``sys.modules``
//! with the full dotted path so ``from
//! kaos_nlp_core._rust.chunking import pack_units`` resolves.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::core::chunking::{
    pack_units as core_pack_units, semantic_pack as core_semantic_pack, PackError, PackedGroup,
    SemanticPackError,
};

fn map_pack_err(e: PackError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn map_semantic_pack_err(e: SemanticPackError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn groups_to_tuple<'py>(py: Python<'py>, groups: &[PackedGroup]) -> PyResult<Bound<'py, PyTuple>> {
    let n = groups.len();
    let mut group_starts = Vec::with_capacity(n);
    let mut group_ends = Vec::with_capacity(n);
    let mut group_unit_starts = Vec::with_capacity(n);
    let mut group_unit_ends = Vec::with_capacity(n);
    let mut group_token_sums = Vec::with_capacity(n);
    for g in groups {
        group_starts.push(g.start);
        group_ends.push(g.end);
        group_unit_starts.push(g.unit_start);
        group_unit_ends.push(g.unit_end);
        group_token_sums.push(g.unit_token_sum);
    }
    PyTuple::new(
        py,
        [
            group_starts.into_pyarray(py).into_any(),
            group_ends.into_pyarray(py).into_any(),
            group_unit_starts.into_pyarray(py).into_any(),
            group_unit_ends.into_pyarray(py).into_any(),
            group_token_sums.into_pyarray(py).into_any(),
        ],
    )
}

/// Pack ordered units into chunks under a token budget.
///
/// See :mod:`crate::core::chunking::pack` for the algorithm. The
/// PyO3 layer simply marshals numpy arrays to Rust slices, runs the
/// pure-Rust kernel with the GIL released, and returns five parallel
/// ``uint32`` arrays describing the resulting groups.
///
/// Args:
///     starts: 1-D ``uint32`` array of per-unit start char offsets.
///     ends: 1-D ``uint32`` array of per-unit end char offsets.
///     token_counts: 1-D ``uint32`` array of per-unit token counts.
///     max_tokens: Soft ceiling on per-chunk token sum. Must be > 0.
///     overlap_units: Number of trailing units to repeat in the next
///         group. ``0`` for no overlap.
///
/// Returns:
///     Tuple of five ``uint32`` numpy arrays:
///     ``(group_starts, group_ends, group_unit_starts,
///     group_unit_ends, group_token_sums)``. All have the same length
///     (one element per group).
///
/// Raises:
///     ValueError: invalid configuration (zero ``max_tokens``,
///         parallel-array length mismatch, etc.).
#[pyfunction]
#[pyo3(signature = (starts, ends, token_counts, max_tokens, overlap_units=0))]
fn pack_units<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'_, u32>,
    ends: PyReadonlyArray1<'_, u32>,
    token_counts: PyReadonlyArray1<'_, u32>,
    max_tokens: u32,
    overlap_units: u32,
) -> PyResult<Bound<'py, PyTuple>> {
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;
    let token_counts_slice = token_counts.as_slice()?;

    let groups: Vec<PackedGroup> = py
        .detach(|| {
            core_pack_units(
                starts_slice,
                ends_slice,
                token_counts_slice,
                max_tokens,
                overlap_units,
            )
        })
        .map_err(map_pack_err)?;

    groups_to_tuple(py, &groups)
}

/// Pack ordered units under a token budget *and* semantic-shift cuts.
///
/// Behaves like :func:`pack_units` for the budget rule, but also
/// inserts a chunk boundary whenever the adjacent cosine similarity
/// at position ``i`` (`adj_sim[i - 1]`) falls below
/// ``drop_threshold``. Used by
/// :class:`kaos_nlp_transformers.SemanticChunker`. The Python caller
/// computes ``adj_sim`` via
/// :func:`kaos_nlp_core.similarity.cosine_adjacent` (SIMD-dispatched
/// via NumKong) before invoking this kernel; the kernel itself only
/// runs the greedy boundary scan with the GIL released.
///
/// Args:
///     starts: 1-D ``uint32`` array of per-unit start char offsets.
///     ends: 1-D ``uint32`` array of per-unit end char offsets.
///     token_counts: 1-D ``uint32`` array of per-unit token counts.
///     adj_sim: 1-D ``float32`` array of length ``n_units - 1``
///         (``0`` if ``n_units <= 1``) carrying adjacent-pair cosine
///         similarities.
///     max_tokens: Soft ceiling on per-chunk token sum. Must be > 0.
///     drop_threshold: Cosine similarity below which a topic shift is
///         declared and a cut is forced. Must be in ``[-1, 1]``.
///
/// Returns:
///     Tuple of five ``uint32`` numpy arrays as in :func:`pack_units`.
///
/// Raises:
///     ValueError: invalid configuration (``max_tokens == 0``,
///         ``drop_threshold`` out of range, parallel-array length
///         mismatch, or ``adj_sim`` length not ``n_units - 1``).
#[pyfunction]
#[pyo3(signature = (starts, ends, token_counts, adj_sim, max_tokens, drop_threshold))]
fn semantic_pack<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'_, u32>,
    ends: PyReadonlyArray1<'_, u32>,
    token_counts: PyReadonlyArray1<'_, u32>,
    adj_sim: PyReadonlyArray1<'_, f32>,
    max_tokens: u32,
    drop_threshold: f32,
) -> PyResult<Bound<'py, PyTuple>> {
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;
    let token_counts_slice = token_counts.as_slice()?;
    let adj_sim_slice = adj_sim.as_slice()?;

    let groups: Vec<PackedGroup> = py
        .detach(|| {
            core_semantic_pack(
                starts_slice,
                ends_slice,
                token_counts_slice,
                adj_sim_slice,
                max_tokens,
                drop_threshold,
            )
        })
        .map_err(map_semantic_pack_err)?;

    groups_to_tuple(py, &groups)
}

pub(crate) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "chunking")?;
    m.add_function(wrap_pyfunction!(pack_units, &m)?)?;
    m.add_function(wrap_pyfunction!(semantic_pack, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.chunking", &m)?;
    Ok(())
}
