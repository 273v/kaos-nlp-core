//! Dense-vector cosine / dot / Euclidean primitives.
//!
//! All compute routes through the SIMD-dispatched f32 kernels in
//! [`crate::core::similarity::kernels`]. The functions here:
//!
//! 1. Validate input shapes (lengths, non-zero rows, dim agreement).
//! 2. Call the kernel layer (`cosine_components_f32`, `dot_f32`,
//!    `norm_sq_f32`, `finalize_cosine_f32`) with the validated slices.
//! 3. Surface a typed [`SimilarityError`] for the small set of
//!    invariant violations callers care about.
//!
//! Numerical contract
//! ------------------
//!
//! * Input vectors are `&[f32]` (we do not currently expose `f64` or
//!   half-precision — that's a follow-on once a consumer needs it).
//! * Empty inputs return `Err(SimilarityError::EmptyInput)`.
//! * Length mismatch returns `Err(SimilarityError::DimensionMismatch)`.
//! * Vectors of all-zero (zero L2 norm) yield cosine = 0.0 by
//!   convention (the only sensible answer when the angle is
//!   undefined).
//! * Cosine results are clipped to `[-1.0, 1.0]` inside the kernel
//!   to absorb f64 round-off near `±1`.
//!
//! Pre-normalised fast path
//! ------------------------
//!
//! `cosine_one_to_many_normalized` / `cosine_adjacent_normalized` skip
//! both per-row `‖row‖²` computations and the final `sqrt`/clamp, on
//! the contract that callers feed unit-norm rows. This is the right
//! path for SemanticChunker / ExtractiveRanker in
//! `kaos-nlp-transformers` — their `EmbeddingModel` guarantees unit-norm
//! rows. See `docs/design-similarity-simd.md`.

use super::kernels::{
    cosine_adjacent_into, cosine_adjacent_normalized_into, cosine_f32, cosine_f32_normalized,
    cosine_one_to_many_into, cosine_one_to_many_normalized_into, norm_sq_f32,
};
use thiserror::Error;

/// Errors surfaced by the dense-similarity primitives.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SimilarityError {
    /// One of the input vectors was empty.
    #[error("input vector is empty")]
    EmptyInput,
    /// The two vectors disagree on dimension.
    #[error("dimension mismatch: {a} vs {b}")]
    DimensionMismatch { a: usize, b: usize },
    /// A batch input was not a rectangular matrix.
    #[error("batch shape mismatch: row {row} has {actual} elements, expected {expected}")]
    RaggedBatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
}

/// L2-normalise a vector in place. Zero-norm vectors are left
/// untouched (no division by zero) and the function returns
/// `Ok(false)`; non-zero norms return `Ok(true)`.
///
/// The normalisation uses the SIMD-dispatched `norm_sq_f32` kernel
/// so the result inherits the same numerical accuracy as the cosine
/// path.
pub fn l2_normalize_in_place(vec: &mut [f32]) -> Result<bool, SimilarityError> {
    if vec.is_empty() {
        return Err(SimilarityError::EmptyInput);
    }
    let sq: f32 = norm_sq_f32(vec);
    if sq <= 0.0 {
        return Ok(false);
    }
    let inv = (sq as f64).sqrt().recip() as f32;
    for x in vec.iter_mut() {
        *x *= inv;
    }
    Ok(true)
}

/// Cosine similarity between two equal-length `f32` vectors.
///
/// Returns a value in `[-1.0, 1.0]`. Calls the fused single-pass
/// kernel which computes `(dot, ‖a‖², ‖b‖²)` in one SIMD pass and
/// then normalises via a scalar `f64` finalisation.
pub fn cosine(a: &[f32], b: &[f32]) -> Result<f32, SimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(SimilarityError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(SimilarityError::DimensionMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    Ok(cosine_f32(a, b))
}

/// Cosine similarity assuming both inputs are unit-norm.
///
/// Pre-normalised fast path: cosine of two unit-norm vectors is their
/// dot product (clamped to `[-1, 1]`). **The caller must guarantee
/// unit-norm inputs** — this function does not verify them. If the
/// contract is violated the result is silently wrong (because there's
/// no efficient way to detect this without the very norm computation
/// we're trying to skip).
pub fn cosine_normalized(a: &[f32], b: &[f32]) -> Result<f32, SimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(SimilarityError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(SimilarityError::DimensionMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    Ok(cosine_f32_normalized(a, b))
}

/// Cosine similarity of one query vector against many rows.
///
/// `matrix` is a row-major 2-D buffer of shape `(n_rows, dim)` packed
/// as a flat `&[f32]`. The query must have length `dim`. Output:
/// `Vec<f32>` of length `n_rows` with element `i` =
/// `cosine(query, matrix[i])`.
///
/// Hot path for retrieval. We pre-compute the query's squared norm
/// once (no per-row redundant work) and call the fused single-pass
/// kernel per row to get `(dot, ‖row‖²)` in one SIMD pass over the
/// row. The query norm² stays in a scalar across all rows; per-row
/// finalisation is one `f64` sqrt + one `f64` divide.
pub fn cosine_one_to_many(
    query: &[f32],
    matrix: &[f32],
    dim: usize,
) -> Result<Vec<f32>, SimilarityError> {
    if query.is_empty() {
        return Err(SimilarityError::EmptyInput);
    }
    if query.len() != dim {
        return Err(SimilarityError::DimensionMismatch {
            a: query.len(),
            b: dim,
        });
    }
    if !matrix.len().is_multiple_of(dim) {
        return Err(SimilarityError::RaggedBatch {
            row: matrix.len() / dim,
            expected: dim,
            actual: matrix.len() % dim,
        });
    }
    let n_rows = matrix.len() / dim;
    let mut out = vec![0.0_f32; n_rows];
    // Hot path: dispatch ONCE and run the whole row loop inside the
    // ISA-specific kernel. The kernel computes the query norm once at
    // the top of the call and then runs two FMAs per element per row
    // (`(dot, ‖row‖²)`), rather than three. The redundant f32→f64 cast
    // also moves out of the row loop because finalisation is per-row
    // scalar but the SIMD norm² stays in f32 across rows.
    cosine_one_to_many_into(query, matrix, dim, &mut out);
    Ok(out)
}

/// Cosine similarity of one **unit-norm** query against many
/// **unit-norm** rows.
///
/// Pre-normalised fast path: skips both `‖q‖²` and `‖row‖²` and the
/// `rsqrt` finalisation; each row is a single SIMD dot product
/// clamped to `[-1, 1]`. **Caller responsibility**: both `query` and
/// every row of `matrix` must already be unit-norm. The function does
/// not verify this contract.
///
/// SemanticChunker and ExtractiveRanker (kaos-nlp-transformers) feed
/// unit-norm embeddings; this is their fast path.
pub fn cosine_one_to_many_normalized(
    query: &[f32],
    matrix: &[f32],
    dim: usize,
) -> Result<Vec<f32>, SimilarityError> {
    if query.is_empty() {
        return Err(SimilarityError::EmptyInput);
    }
    if query.len() != dim {
        return Err(SimilarityError::DimensionMismatch {
            a: query.len(),
            b: dim,
        });
    }
    if !matrix.len().is_multiple_of(dim) {
        return Err(SimilarityError::RaggedBatch {
            row: matrix.len() / dim,
            expected: dim,
            actual: matrix.len() % dim,
        });
    }
    let n_rows = matrix.len() / dim;
    let mut out = vec![0.0_f32; n_rows];
    // Same single-dispatch-then-loop pattern as the generic path; the
    // ISA kernel runs `dot` per row (one FMA per element) and clamps.
    cosine_one_to_many_normalized_into(query, matrix, dim, &mut out);
    Ok(out)
}

/// Cosine similarity between adjacent rows of a `(n_rows, dim)`
/// matrix. Used by the semantic chunker.
///
/// Output has length `n_rows - 1`. When `n_rows < 2`, returns an
/// empty `Vec`.
pub fn cosine_adjacent(matrix: &[f32], dim: usize) -> Result<Vec<f32>, SimilarityError> {
    if dim == 0 {
        return Err(SimilarityError::DimensionMismatch { a: dim, b: 1 });
    }
    if !matrix.len().is_multiple_of(dim) {
        return Err(SimilarityError::RaggedBatch {
            row: matrix.len() / dim,
            expected: dim,
            actual: matrix.len() % dim,
        });
    }
    let n_rows = matrix.len() / dim;
    if n_rows < 2 {
        return Ok(Vec::new());
    }
    // Route through the kernel layer's `cosine_adjacent_into` so we
    // pick the ISA-specific path (AVX2/AVX-512/NEON/scalar) instead of
    // dispatching once per pair via `cosine_f32`.
    let mut out = vec![0.0_f32; n_rows - 1];
    cosine_adjacent_into(matrix, dim, &mut out);
    Ok(out)
}

/// Adjacent-row cosine on a matrix of **unit-norm** rows.
///
/// Pre-normalised fast path: every pair becomes a single SIMD dot
/// product clamped to `[-1, 1]`. **Caller responsibility**: every row
/// of `matrix` must be unit-norm.
///
/// SemanticChunker (kaos-nlp-transformers) feeds unit-norm embeddings;
/// this is its fast path.
pub fn cosine_adjacent_normalized(matrix: &[f32], dim: usize) -> Result<Vec<f32>, SimilarityError> {
    if dim == 0 {
        return Err(SimilarityError::DimensionMismatch { a: dim, b: 1 });
    }
    if !matrix.len().is_multiple_of(dim) {
        return Err(SimilarityError::RaggedBatch {
            row: matrix.len() / dim,
            expected: dim,
            actual: matrix.len() % dim,
        });
    }
    let n_rows = matrix.len() / dim;
    if n_rows < 2 {
        return Ok(Vec::new());
    }
    // Same dispatch routing as the unnormalised path -- the kernel's
    // `cosine_adjacent_normalized_into` fuses dot+clamp in the SIMD
    // body, so we let it do the per-pair work in one ISA-specific
    // sweep rather than calling `dot_f32` per pair from this loop.
    let mut out = vec![0.0_f32; n_rows - 1];
    cosine_adjacent_normalized_into(matrix, dim, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let c = cosine(&v, &v).unwrap();
        assert!((c - 1.0).abs() < TOL, "got {c}");
    }

    #[test]
    fn cosine_opposite_is_minus_one() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![-1.0_f32, -2.0, -3.0];
        let c = cosine(&a, &b).unwrap();
        assert!((c + 1.0).abs() < TOL, "got {c}");
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let c = cosine(&a, &b).unwrap();
        assert!(c.abs() < TOL, "got {c}");
    }

    #[test]
    fn cosine_empty_errors() {
        let empty: Vec<f32> = vec![];
        let v = vec![1.0_f32];
        assert!(matches!(
            cosine(&empty, &v),
            Err(SimilarityError::EmptyInput)
        ));
    }

    #[test]
    fn cosine_dim_mismatch_errors() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        assert!(matches!(
            cosine(&a, &b),
            Err(SimilarityError::DimensionMismatch { a: 2, b: 3 })
        ));
    }

    #[test]
    fn cosine_clipped_to_unit_interval() {
        let v = vec![0.5773_f32; 256];
        let c = cosine(&v, &v).unwrap();
        assert!(
            (-1.0..=1.0).contains(&c),
            "cosine must stay in [-1, 1], got {c}"
        );
        assert!((c - 1.0).abs() < TOL);
    }

    #[test]
    fn cosine_one_to_many_works() {
        let matrix: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, 0.0,
        ];
        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let sims = cosine_one_to_many(&query, &matrix, 4).unwrap();
        assert_eq!(sims.len(), 3);
        assert!((sims[0] - 1.0).abs() < TOL);
        assert!(sims[1].abs() < TOL);
        assert!((sims[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < TOL);
    }

    #[test]
    fn cosine_one_to_many_dim_mismatch() {
        let matrix = vec![1.0_f32; 12];
        let query = vec![1.0_f32; 3];
        assert!(matches!(
            cosine_one_to_many(&query, &matrix, 4),
            Err(SimilarityError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn cosine_one_to_many_zero_query_returns_zeros() {
        let matrix = vec![1.0_f32, 2.0, 3.0, 4.0];
        let query = vec![0.0_f32, 0.0];
        let sims = cosine_one_to_many(&query, &matrix, 2).unwrap();
        assert_eq!(sims, vec![0.0_f32, 0.0]);
    }

    #[test]
    fn cosine_adjacent_works() {
        let matrix: Vec<f32> = vec![
            1.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, -1.0, 0.0,
        ];
        let sims = cosine_adjacent(&matrix, 3).unwrap();
        assert_eq!(sims.len(), 3);
        assert!((sims[0] - 1.0).abs() < TOL);
        assert!(sims[1].abs() < TOL);
        assert!((sims[2] + 1.0).abs() < TOL);
    }

    #[test]
    fn cosine_adjacent_single_row_returns_empty() {
        let matrix = vec![1.0_f32; 4];
        let sims = cosine_adjacent(&matrix, 4).unwrap();
        assert!(sims.is_empty());
    }

    #[test]
    fn l2_normalize_unit_vector_unchanged() {
        let mut v = vec![1.0_f32, 0.0, 0.0];
        let was_norm = l2_normalize_in_place(&mut v).unwrap();
        assert!(was_norm);
        assert!((v[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn l2_normalize_zero_vector_left_alone() {
        let mut v = vec![0.0_f32; 4];
        let was_norm = l2_normalize_in_place(&mut v).unwrap();
        assert!(!was_norm);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn l2_normalize_produces_unit_norm() {
        let mut v = vec![3.0_f32, 4.0];
        l2_normalize_in_place(&mut v).unwrap();
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < TOL);
    }

    // ------------------------------------------------------------------
    // Normalised fast path — matches the generic path bit-for-bit on
    // valid unit-norm inputs.
    // ------------------------------------------------------------------

    fn l2_normalize_clone(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn cosine_normalized_matches_cosine_on_unit_norm_inputs() {
        // 384-d production-shape pair
        let a: Vec<f32> = (0..384).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..384)
            .map(|i| ((i as f32 * 0.02).cos() + 0.1).abs())
            .collect();
        let a_u = l2_normalize_clone(&a);
        let b_u = l2_normalize_clone(&b);
        let generic = cosine(&a_u, &b_u).unwrap();
        let fast = cosine_normalized(&a_u, &b_u).unwrap();
        assert!(
            (generic - fast).abs() < 1e-5,
            "generic={generic}, fast={fast}"
        );
    }

    #[test]
    fn cosine_one_to_many_normalized_matches_generic() {
        // 100 × 256 production-shape matrix, all rows unit-norm
        let mut matrix: Vec<f32> = Vec::with_capacity(100 * 256);
        for i in 0..100 {
            let row: Vec<f32> = (0..256)
                .map(|j| ((i as f32) * 0.1 + (j as f32) * 0.01).sin().abs() + 0.05)
                .collect();
            matrix.extend(l2_normalize_clone(&row));
        }
        let query_raw: Vec<f32> = (0..256).map(|j| (j as f32 + 0.5).cos()).collect();
        let query = l2_normalize_clone(&query_raw);
        let generic = cosine_one_to_many(&query, &matrix, 256).unwrap();
        let fast = cosine_one_to_many_normalized(&query, &matrix, 256).unwrap();
        assert_eq!(generic.len(), fast.len());
        for (g, f) in generic.iter().zip(fast.iter()) {
            assert!((g - f).abs() < 1e-5, "g={g}, f={f}");
        }
    }

    #[test]
    fn cosine_adjacent_normalized_matches_generic() {
        // 50 × 384 unit-norm rows
        let mut matrix: Vec<f32> = Vec::with_capacity(50 * 384);
        for i in 0..50 {
            let row: Vec<f32> = (0..384)
                .map(|j| ((i as f32) * 0.03 + (j as f32) * 0.002).cos() + 0.5)
                .collect();
            matrix.extend(l2_normalize_clone(&row));
        }
        let generic = cosine_adjacent(&matrix, 384).unwrap();
        let fast = cosine_adjacent_normalized(&matrix, 384).unwrap();
        assert_eq!(generic.len(), fast.len());
        for (g, f) in generic.iter().zip(fast.iter()) {
            assert!((g - f).abs() < 1e-5, "g={g}, f={f}");
        }
    }

    #[test]
    fn cosine_one_to_many_normalized_empty_query_errors() {
        let empty: Vec<f32> = vec![];
        let matrix = vec![1.0_f32, 0.0];
        assert!(matches!(
            cosine_one_to_many_normalized(&empty, &matrix, 2),
            Err(SimilarityError::EmptyInput)
        ));
    }

    #[test]
    fn cosine_adjacent_normalized_single_row_returns_empty() {
        let matrix = vec![1.0_f32, 0.0, 0.0];
        let sims = cosine_adjacent_normalized(&matrix, 3).unwrap();
        assert!(sims.is_empty());
    }
}
