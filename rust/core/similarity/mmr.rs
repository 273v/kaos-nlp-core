//! Maximal Marginal Relevance (MMR) selection over a dense matrix.
//!
//! The classic Carbonell-Goldstein 1998 reranker. Given a candidate
//! pool of `n` rows in a `(n, dim)` matrix, an externally-supplied
//! `relevance` score per row (e.g. cosine to a query, or a
//! cross-encoder score), and a diversity weight `lambda ∈ [0, 1]`,
//! select `k` rows greedily by:
//!
//! ```text
//! score(i) = lambda * relevance[i] - (1 - lambda) * max_j∈picked(cos(i, j))
//! ```
//!
//! Implementation notes
//! --------------------
//!
//! * The first pick is always `argmax(relevance)`.
//! * `max_sim` is maintained incrementally — each new pick contributes
//!   one `cosine_one_to_many(picked_row, all_rows)` SIMD pass through
//!   NumKong, then `np.maximum` (here, a single in-place pass) folds
//!   the new similarities into the running maximum.
//! * Overall cost is `O(k * n * dim)` BLAS ops, dominated by the
//!   `k` NumKong-dispatched cosine sweeps. This is the equivalent of
//!   numpy's `normed @ normed[picked].T` per pick, but routed
//!   through SIMD kernels rather than naive matmul.
//! * Ties (within a single pick step) are broken by ascending index,
//!   matching the [`crate::core::algorithms::ranking`] convention.

use super::dense::{cosine_one_to_many, SimilarityError};

/// Result of an MMR selection: indices of the picked rows in pick order
/// and the MMR scores at the moment each was picked.
#[derive(Debug, Clone, PartialEq)]
pub struct MmrResult {
    /// Row indices into the original matrix, in pick order
    /// (most-relevant first, with diversity penalty applied).
    pub indices: Vec<u32>,
    /// MMR scores at pick time — `lambda * relevance - (1 - lambda) * max_sim`.
    pub scores: Vec<f32>,
}

/// Greedy MMR selection.
///
/// Arguments
/// ---------
///
/// * `matrix` — flat row-major `(n_rows, dim)` candidate embeddings.
/// * `dim` — vector dimensionality.
/// * `relevance` — per-row relevance score (length must equal
///   `n_rows`). Typically cosine to a query, or a cross-encoder
///   score; we don't care about absolute scale.
/// * `k` — number of picks to return. Capped at `n_rows`.
/// * `lambda` — diversity weight in `[0, 1]`. `1.0` collapses to
///   "rank by relevance" (no diversity); `0.0` collapses to
///   "MMR drives entirely by diversity" (still anchored by the first
///   relevance-argmax pick).
pub fn mmr_select(
    matrix: &[f32],
    dim: usize,
    relevance: &[f32],
    k: usize,
    lambda: f32,
) -> Result<MmrResult, SimilarityError> {
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
    if n_rows == 0 || k == 0 {
        return Ok(MmrResult {
            indices: Vec::new(),
            scores: Vec::new(),
        });
    }
    if relevance.len() != n_rows {
        return Err(SimilarityError::DimensionMismatch {
            a: relevance.len(),
            b: n_rows,
        });
    }
    let cap = k.min(n_rows);
    // Clamp lambda to [0, 1] silently rather than erroring — caller
    // intent is usually "diversity penalty proportional to one minus
    // this knob", and out-of-range is a UX bug, not a correctness one.
    let lambda = lambda.clamp(0.0, 1.0);

    // First pick: argmax(relevance) with ascending-index tie-break.
    let first = pick_argmax(relevance, &vec![false; n_rows]);
    let Some(first_idx) = first else {
        return Ok(MmrResult {
            indices: Vec::new(),
            scores: Vec::new(),
        });
    };

    let mut indices = Vec::with_capacity(cap);
    let mut scores = Vec::with_capacity(cap);
    let mut picked = vec![false; n_rows];

    indices.push(first_idx as u32);
    scores.push(lambda * relevance[first_idx]);
    picked[first_idx] = true;

    // Running max similarity of every row against the picked set.
    // Initialize with cosine to the first picked row.
    let first_row = &matrix[first_idx * dim..(first_idx + 1) * dim];
    let mut max_sim = cosine_one_to_many(first_row, matrix, dim)?;

    while indices.len() < cap {
        // Compute MMR for every candidate.
        let mut best: Option<(usize, f32)> = None;
        for (i, rel) in relevance.iter().enumerate() {
            if picked[i] {
                continue;
            }
            let candidate = lambda * rel - (1.0 - lambda) * max_sim[i];
            if candidate.is_nan() {
                continue;
            }
            match best {
                None => best = Some((i, candidate)),
                Some((_, best_score)) if candidate > best_score => best = Some((i, candidate)),
                // Ascending-index tie-break: only replace if strictly
                // greater; equal-but-later loses to equal-but-earlier
                // because of natural iteration order.
                _ => {}
            }
        }
        let Some((next_idx, next_score)) = best else {
            break;
        };
        indices.push(next_idx as u32);
        scores.push(next_score);
        picked[next_idx] = true;

        // Update running max-sim with cosines to the new pick.
        let next_row = &matrix[next_idx * dim..(next_idx + 1) * dim];
        let sims_to_new = cosine_one_to_many(next_row, matrix, dim)?;
        for (running, new) in max_sim.iter_mut().zip(sims_to_new.iter()) {
            if *new > *running {
                *running = *new;
            }
        }
    }

    Ok(MmrResult { indices, scores })
}

/// Argmax with ascending-index tie-break, optionally skipping rows
/// flagged in `skip`. Returns `None` only when every row is skipped
/// or every score is NaN.
fn pick_argmax(values: &[f32], skip: &[bool]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, v) in values.iter().enumerate() {
        if skip[i] || v.is_nan() {
            continue;
        }
        match best {
            None => best = Some((i, *v)),
            Some((_, bv)) if *v > bv => best = Some((i, *v)),
            _ => {}
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    #[test]
    fn mmr_lambda_one_collapses_to_relevance_rank() {
        // 3 rows; relevance = [0.1, 0.9, 0.5]. lambda=1.0 means
        // ignore diversity. Expected picks (k=3): [1, 2, 0].
        let matrix = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let relevance = vec![0.1_f32, 0.9, 0.5];
        let r = mmr_select(&matrix, 2, &relevance, 3, 1.0).unwrap();
        assert_eq!(r.indices, vec![1, 2, 0]);
    }

    #[test]
    fn mmr_first_pick_is_argmax_relevance() {
        let matrix = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let relevance = vec![0.2_f32, 0.9, 0.5];
        let r = mmr_select(&matrix, 2, &relevance, 1, 0.5).unwrap();
        assert_eq!(r.indices, vec![1]);
    }

    #[test]
    fn mmr_diversity_prefers_orthogonal() {
        // Row 0 and row 1 are identical → highly similar. Row 2 is
        // orthogonal. With high relevance on row 0 + moderate on row 1
        // + moderate on row 2 + low lambda, MMR should prefer row 2
        // over row 1 on the second pick.
        let matrix: Vec<f32> = vec![
            1.0, 0.0, // row 0
            1.0, 0.0, // row 1 (identical to row 0)
            0.0, 1.0, // row 2 (orthogonal)
        ];
        let relevance = vec![0.9_f32, 0.8, 0.5];
        let r = mmr_select(&matrix, 2, &relevance, 2, 0.3).unwrap();
        assert_eq!(r.indices[0], 0); // argmax relevance
                                     // Second pick: with lambda=0.3, the diversity term dominates
                                     // for row 1 (max_sim=1 vs row 0) → score = 0.3*0.8 - 0.7*1 = -0.46
                                     // For row 2: score = 0.3*0.5 - 0.7*0 = 0.15
        assert_eq!(r.indices[1], 2);
    }

    #[test]
    fn mmr_k_zero_returns_empty() {
        let matrix = vec![1.0_f32; 4];
        let relevance = vec![1.0_f32, 0.0];
        let r = mmr_select(&matrix, 2, &relevance, 0, 0.5).unwrap();
        assert!(r.indices.is_empty());
    }

    #[test]
    fn mmr_k_larger_than_n_caps() {
        let matrix = vec![1.0_f32, 0.0, 0.0, 1.0];
        let relevance = vec![1.0_f32, 0.5];
        let r = mmr_select(&matrix, 2, &relevance, 100, 0.5).unwrap();
        assert_eq!(r.indices.len(), 2);
    }

    #[test]
    fn mmr_dim_mismatch_error() {
        let matrix = vec![1.0_f32; 4];
        let relevance = vec![1.0_f32; 4]; // wrong count
        assert!(mmr_select(&matrix, 2, &relevance, 1, 0.5).is_err());
    }

    #[test]
    fn mmr_lambda_zero_anchored_by_argmax() {
        // lambda=0 → pure diversity penalty. First pick is still
        // relevance-argmax (the anchor); subsequent picks ignore
        // relevance entirely.
        let matrix: Vec<f32> = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let relevance = vec![0.9_f32, 0.5, 0.1];
        let r = mmr_select(&matrix, 3, &relevance, 3, 0.0).unwrap();
        assert_eq!(r.indices[0], 0);
        // After picking row 0 (which is orthogonal to rows 1 and 2),
        // diversity scores are identical → tie-break by ascending
        // index picks row 1, then row 2.
        assert_eq!(r.indices, vec![0, 1, 2]);
    }

    #[test]
    fn mmr_scores_monotone_decreasing_with_diversity() {
        // With diversity penalty, MMR scores should generally
        // decrease (relevance-only would be monotone; with diversity
        // it's near-monotone). At minimum the first score >= second.
        let matrix: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let relevance = vec![0.9_f32, 0.7, 0.5];
        let r = mmr_select(&matrix, 2, &relevance, 3, 0.5).unwrap();
        for w in r.scores.windows(2) {
            assert!(
                w[0] >= w[1] - TOL,
                "non-monotone MMR scores: {:?}",
                r.scores
            );
        }
    }
}
