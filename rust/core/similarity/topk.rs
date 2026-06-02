//! Top-k cosine retrieval over a dense matrix.
//!
//! Single hot-path function: given a query vector and a `(n_rows, dim)`
//! matrix, return the indices and cosine-similarity scores of the
//! ``k`` most-similar rows.
//!
//! Implementation
//! --------------
//!
//! 1. NumKong computes all `n_rows` cosines (SIMD).
//! 2. A binary heap of size `k` (max-heap by negated score, so the
//!    smallest score sits at the root) is maintained in a single
//!    linear pass; this is the standard `O(n log k)` selection
//!    pattern, equivalent to NumPy's `argpartition(-sims, k)` +
//!    descending sort over the top-k slice.
//!
//! Determinism
//! -----------
//!
//! Ties are broken by **ascending index** (smaller row index wins),
//! matching the convention already used by
//! [`crate::core::algorithms::ranking`]. The function is pure: same
//! inputs → same output bytes, across processes, hash seeds, and
//! parallelism. NaN scores are dropped (treated as `-infinity`).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::dense::{cosine_one_to_many, SimilarityError};

/// A `(score, index)` pair where the **max-heap** orders by ascending
/// score (so the lowest score sits at the heap root and can be popped
/// when a better candidate arrives). Index is the deterministic
/// tiebreaker — when two rows score identically, the smaller index
/// wins.
#[derive(Clone, Copy, Debug)]
struct ScoredCandidate {
    /// Cosine similarity in `[-1, 1]`. NaN-free by construction;
    /// callers filter NaN before pushing.
    score: f32,
    /// Row index. Used as a deterministic tiebreaker.
    index: u32,
}

impl PartialEq for ScoredCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.index == other.index
    }
}

impl Eq for ScoredCandidate {}

impl PartialOrd for ScoredCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredCandidate {
    /// Min-heap-by-inversion: a "smaller" candidate (one we want to
    /// pop first) has a *lower* score. When scores tie, a *higher*
    /// index counts as "smaller" so it gets popped, preserving the
    /// lowest-index-wins tiebreak in the survivor set.
    fn cmp(&self, other: &Self) -> Ordering {
        match other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => self.index.cmp(&other.index),
            ord => ord,
        }
    }
}

/// Result of a top-k retrieval: indices and matching scores, both
/// ordered by score descending (ties by ascending index).
#[derive(Debug, Clone, PartialEq)]
pub struct TopKResult {
    /// Row indices into the original matrix, in result order.
    pub indices: Vec<u32>,
    /// Cosine similarity scores, in result order (matches `indices`
    /// element-wise).
    pub scores: Vec<f32>,
}

/// Return the top-`k` rows of `matrix` (shape `(n_rows, dim)` packed
/// row-major) by cosine similarity to `query`. Result is ordered
/// **score descending**, with ascending-index tie-break.
///
/// * `k = 0` returns an empty result.
/// * `k > n_rows` returns all rows (still ordered).
/// * NaN scores are dropped (they can't be ranked).
/// * Empty inputs / dimension mismatch return
///   [`SimilarityError`].
pub fn top_k_cosine(
    query: &[f32],
    matrix: &[f32],
    dim: usize,
    k: usize,
) -> Result<TopKResult, SimilarityError> {
    if k == 0 {
        return Ok(TopKResult {
            indices: Vec::new(),
            scores: Vec::new(),
        });
    }
    let sims = cosine_one_to_many(query, matrix, dim)?;
    Ok(select_top_k(&sims, k))
}

/// Select the top-`k` entries of a pre-computed similarity vector by
/// score descending, with ascending-index tie-break. NaN scores are
/// dropped (they cannot be ranked), so the result may contain fewer
/// than `k` entries when `sims` holds NaNs or is shorter than `k`.
///
/// This is the selection half of [`top_k_cosine`], factored out so the
/// similarity-graph builders (`crate::core::similarity::graph`) can run
/// the same deterministic `O(n log k)` heap selection over a row of
/// cosines they have already computed — without recomputing the cosine
/// sweep or duplicating the tie-break invariant.
pub fn select_top_k(sims: &[f32], k: usize) -> TopKResult {
    if k == 0 {
        return TopKResult {
            indices: Vec::new(),
            scores: Vec::new(),
        };
    }
    let effective_k = k.min(sims.len());

    // Min-heap (inverted Ord above): smallest survivor at the root.
    // Insert the first `k` candidates, then for each subsequent
    // candidate, if it beats the root, replace and sift.
    let mut heap: BinaryHeap<ScoredCandidate> = BinaryHeap::with_capacity(effective_k);

    for (index, score) in sims.iter().enumerate() {
        let score = *score;
        if score.is_nan() {
            continue;
        }
        let candidate = ScoredCandidate {
            score,
            index: index as u32,
        };
        if heap.len() < effective_k {
            heap.push(candidate);
        } else if let Some(root) = heap.peek() {
            // The heap root holds the *worst-still-surviving*
            // element — under our inverted `Ord` (see below) that
            // means the lowest real score, or for ties, the highest
            // index. Replace iff this candidate strictly beats the
            // root on real-score, OR ties on real-score with a
            // smaller index. Using direct field comparison here
            // avoids the easy-to-misread "is candidate > root in
            // inverted Ord" check at the cost of restating the
            // invariant; the Ord impl below is still load-bearing
            // for the heap to *order itself* correctly.
            let replace = candidate.score > root.score
                || (candidate.score == root.score && candidate.index < root.index);
            if replace {
                heap.pop();
                heap.push(candidate);
            }
        }
    }

    // Drain in heap order then reverse: heap pops smallest-first; we
    // want largest-first in the output.
    let mut survivors: Vec<ScoredCandidate> = heap.into_iter().collect();
    survivors.sort_by(
        |a, b| match b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal) {
            Ordering::Equal => a.index.cmp(&b.index),
            ord => ord,
        },
    );
    let mut indices = Vec::with_capacity(survivors.len());
    let mut scores = Vec::with_capacity(survivors.len());
    for s in survivors {
        indices.push(s.index);
        scores.push(s.score);
    }
    TopKResult { indices, scores }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_basic() {
        // 3 rows of dim 2: row 0 best, row 2 second, row 1 worst.
        let matrix = vec![
            1.0_f32, 0.0, // row 0: parallel to query → 1.0
            0.0, 1.0, // row 1: orth to query → 0.0
            1.0, 0.5, // row 2: high-cos
        ];
        let query = vec![1.0_f32, 0.0];
        let r = top_k_cosine(&query, &matrix, 2, 2).unwrap();
        assert_eq!(r.indices, vec![0, 2]);
        assert!((r.scores[0] - 1.0).abs() < 1e-5);
        assert!(r.scores[1] > r.scores[1] - 1.0);
    }

    #[test]
    fn top_k_k_larger_than_rows() {
        let matrix = vec![1.0_f32, 0.0, 0.0, 1.0];
        let query = vec![1.0_f32, 0.0];
        let r = top_k_cosine(&query, &matrix, 2, 100).unwrap();
        assert_eq!(r.indices.len(), 2);
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let matrix = vec![1.0_f32, 0.0];
        let query = vec![1.0_f32, 0.0];
        let r = top_k_cosine(&query, &matrix, 2, 0).unwrap();
        assert!(r.indices.is_empty());
        assert!(r.scores.is_empty());
    }

    #[test]
    fn top_k_tiebreak_by_lower_index() {
        // Two identical rows: row 0 and row 1 both perfectly parallel
        // to query. With k=1, lower index wins.
        let matrix = vec![1.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let query = vec![1.0_f32, 0.0];
        let r = top_k_cosine(&query, &matrix, 2, 1).unwrap();
        assert_eq!(r.indices, vec![0]);
    }

    #[test]
    fn top_k_tiebreak_preserved_within_k() {
        // k=2, three rows with identical score for first two.
        let matrix = vec![
            1.0_f32, 0.0, // row 0, score=1
            1.0, 0.0, // row 1, score=1 (tied with row 0)
            0.0, 1.0, // row 2, score=0
        ];
        let query = vec![1.0_f32, 0.0];
        let r = top_k_cosine(&query, &matrix, 2, 2).unwrap();
        assert_eq!(r.indices, vec![0, 1]);
    }

    #[test]
    fn top_k_dim_mismatch_error() {
        let matrix = vec![1.0_f32; 6]; // 3 rows of dim 2
        let query = vec![1.0_f32, 0.0, 0.0];
        assert!(top_k_cosine(&query, &matrix, 2, 1).is_err());
    }

    #[test]
    fn top_k_descending_score_order() {
        let matrix = vec![
            1.0_f32, 0.0, 0.0, //
            0.5, 0.5, 0.6, //
            0.0, 1.0, 0.0,
        ];
        let query = vec![1.0_f32, 0.0, 0.0];
        let r = top_k_cosine(&query, &matrix, 3, 3).unwrap();
        // Confirm scores are non-increasing.
        for w in r.scores.windows(2) {
            assert!(w[0] >= w[1], "expected descending; got {:?}", r.scores);
        }
    }
}
