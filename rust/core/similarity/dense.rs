//! Dense-vector cosine / dot / Euclidean primitives.
//!
//! All compute routes through the `numkong` crate's `SpatialSimilarity`
//! / `Dot` traits, which dispatch to the best available SIMD kernel at
//! runtime. The functions here are thin Rust wrappers that:
//!
//! 1. Validate input shapes (lengths, non-zero rows, dim agreement).
//! 2. Map NumKong's "angular distance" (`1 - cos`) into "cosine
//!    similarity" (`cos`) and clip the well-known floating-point
//!    overshoot (cosine of nearly-parallel vectors can drift to
//!    `1.0 + 1e-7` after f64 rounding — we clamp to `[-1, 1]`).
//! 3. Surface a typed `SimilarityError` for the small set of
//!    invariant violations callers care about.
//!
//! Numerical contract
//! ------------------
//!
//! * Input vectors are `&[f32]` (we do not currently expose f64 or
//!   half-precision — that's a follow-on follow the precedent in
//!   the existing `SimilarityMatrix` once a consumer needs it).
//! * Empty inputs return `Err(SimilarityError::EmptyInput)`.
//! * Length mismatch returns `Err(SimilarityError::DimensionMismatch)`.
//! * Vectors of all-zero (zero L2 norm) yield cosine = 0.0 by
//!   convention (the only sensible answer when the angle is
//!   undefined).
//! * Cosine results are clipped to `[-1.0, 1.0]` to absorb f64
//!   round-off near `±1`.

use numkong::{Angular, Dot};
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
/// The normalisation uses the NumKong-accumulated `f32::dot` to compute
/// the squared norm, so the result inherits the same compensated-sum
/// numerical accuracy as the cosine kernel itself.
pub fn l2_normalize_in_place(vec: &mut [f32]) -> Result<bool, SimilarityError> {
    if vec.is_empty() {
        return Err(SimilarityError::EmptyInput);
    }
    // NumKong's dot accumulates in f64 with Neumaier compensation.
    let sq: f64 = f32::dot(vec, vec).unwrap_or(0.0);
    if sq <= 0.0 {
        return Ok(false);
    }
    let inv = (sq.sqrt() as f32).recip();
    for x in vec.iter_mut() {
        *x *= inv;
    }
    Ok(true)
}

/// Cosine similarity between two equal-length `f32` vectors.
///
/// Returns a value in `[-1.0, 1.0]`. The naive numpy implementation
/// would compute `(a @ b) / (||a|| * ||b||)`; NumKong fuses these into
/// a single pass with f64 compensated accumulators and an
/// `rsqrt`-based normalization for higher numerical stability on long
/// vectors.
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
    // NumKong returns angular distance = 1 - cos.
    let ang: f64 = f32::angular(a, b).unwrap_or(1.0);
    let cos: f64 = 1.0_f64 - ang;
    Ok(cos.clamp(-1.0_f64, 1.0_f64) as f32)
}

/// Cosine similarity of one query vector against many rows.
///
/// `matrix` is a row-major 2-D buffer of shape `(n_rows, dim)` packed
/// as a flat `&[f32]` (the standard Rust+NumPy contract). The query
/// must have length `dim`. Output: `Vec<f32>` of length `n_rows` with
/// element `i` = `cosine(query, matrix[i])`.
///
/// This is the hot path for retrieval (`EmbeddingRetriever.retrieve`,
/// `SearchableDocument.search`, `SearchableCorpus.search`,
/// `kmedoid_seeds`). NumKong dispatches to AVX-512 / AVX2 / NEON
/// internally; we don't release the GIL here because that's the
/// caller's (PyO3 binding's) responsibility.
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
    if matrix.len() % dim != 0 {
        return Err(SimilarityError::RaggedBatch {
            row: matrix.len() / dim,
            expected: dim,
            actual: matrix.len() % dim,
        });
    }
    let n_rows = matrix.len() / dim;
    let mut out = Vec::with_capacity(n_rows);
    for row_index in 0..n_rows {
        let row = &matrix[row_index * dim..(row_index + 1) * dim];
        let ang: f64 = f32::angular(query, row).unwrap_or(1.0);
        let cos: f64 = (1.0_f64 - ang).clamp(-1.0_f64, 1.0_f64);
        out.push(cos as f32);
    }
    Ok(out)
}

/// Cosine similarity between adjacent rows of a `(n_rows, dim)`
/// matrix. Used by the semantic chunker: a topic shift between
/// paragraphs i and i+1 is signalled by a drop in this similarity.
///
/// Output has length `n_rows - 1`. When `n_rows < 2`, returns an
/// empty `Vec`.
pub fn cosine_adjacent(matrix: &[f32], dim: usize) -> Result<Vec<f32>, SimilarityError> {
    if dim == 0 {
        return Err(SimilarityError::DimensionMismatch { a: dim, b: 1 });
    }
    if matrix.len() % dim != 0 {
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
    let mut out = Vec::with_capacity(n_rows - 1);
    for i in 0..n_rows - 1 {
        let a = &matrix[i * dim..(i + 1) * dim];
        let b = &matrix[(i + 1) * dim..(i + 2) * dim];
        let ang: f64 = f32::angular(a, b).unwrap_or(1.0);
        let cos: f64 = (1.0_f64 - ang).clamp(-1.0_f64, 1.0_f64);
        out.push(cos as f32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Allow small tolerance vs textbook math; NumKong widens to f64
    // internally but we return f32, so the comparison floor is f32
    // epsilon.
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
        // Repeated identical floats can produce a numerical cosine
        // very slightly above 1.0 in naive impls. NumKong's
        // compensated path + our explicit clamp keeps us in range.
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
        // 3 rows of dim 4, query = first row → first sim is 1.0.
        let matrix: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, 0.0,
        ];
        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let sims = cosine_one_to_many(&query, &matrix, 4).unwrap();
        assert_eq!(sims.len(), 3);
        assert!((sims[0] - 1.0).abs() < TOL);
        assert!(sims[1].abs() < TOL); // orthogonal
                                      // (1,0,0,0) · (1,1,0,0) / (1 * sqrt(2)) ≈ 0.7071
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
    fn cosine_adjacent_works() {
        // 4 rows of dim 3: row 0 == row 1 (sim=1), row 2 orth to row 1
        // (sim=0), row 3 anti-parallel to row 2 (sim=-1).
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
        let mut v = vec![3.0_f32, 4.0]; // 5-norm
        l2_normalize_in_place(&mut v).unwrap();
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < TOL);
    }
}
