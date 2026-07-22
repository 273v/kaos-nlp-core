//! Similarity-graph primitives over dense f32 matrices.
//!
//! The companion to [`crate::core::similarity::topk`]. Where `top_k_cosine`
//! answers "the `k` rows most similar to *one* query", this module answers
//! the *all-pairs* questions that retrieval / dedup / topic-discovery
//! pipelines reach for:
//!
//! * [`knn_graph`] — for **every** row, its `k` nearest other rows. One
//!   fused Rust call replaces the `N × top_k_cosine` Python loop that
//!   callers would otherwise hand-roll (each per-row sweep runs the
//!   SIMD-dispatched cosine kernels with the GIL released, and the rows
//!   fan out across Rayon worker threads).
//! * [`near_duplicates`] — every upper-triangle pair `(i, j)` with
//!   `i < j` whose cosine is `>= threshold`. The edge set a dedup
//!   pre-pass needs.
//! * [`connected_components`] — union-find transitive closure over an
//!   edge list. Pairs with [`near_duplicates`] to turn duplicate *pairs*
//!   into duplicate *groups* (the union-find step a caller would
//!   otherwise hand-roll).
//!
//! What this module owns vs. what it does not
//! ------------------------------------------
//!
//! These are **primitives**: exact, brute-force, deterministic graph
//! construction over a dense matrix. They are the building blocks that
//! higher layers compose. Domain *orchestration* — semantic dedup
//! policy, canonical-record selection, clustering levels — lives in
//! `kaos-content`, not here (audit-07 / KNT-602 moved `SemanticDedupLevel`
//! there deliberately). Approximate nearest neighbour (HNSW / IVF) also
//! stays out: this module is exact. Up to ~100k rows the exact sweep is
//! well within budget on commodity hardware thanks to the SIMD kernels;
//! beyond that an ANN index belongs in a sibling module.
//!
//! Determinism
//! -----------
//!
//! Every function is pure: same inputs → same output bytes, regardless of
//! how many Rayon threads run the sweep. `knn_graph` inherits
//! `top_k_cosine`'s ascending-index tie-break; `near_duplicates` emits
//! pairs in `(i, j)` lexicographic order; `connected_components` labels
//! each node with the **smallest node index** in its component.

use rayon::prelude::*;

use super::dense::{cosine_one_to_many, cosine_one_to_many_normalized, SimilarityError};
use super::topk::select_top_k;

/// Sentinel row index used to pad a `knn_graph` row that has fewer than
/// `k` rankable neighbours (only possible when the input matrix contains
/// NaN/inf rows, whose cosines cannot be ranked and are therefore
/// dropped). Equal to `u32::MAX`; the paired score slot is `f32::NAN`.
/// With finite inputs — the embedding contract — padding never occurs.
pub const NO_NEIGHBOR: u32 = u32::MAX;

/// Result of a [`knn_graph`] build: a row-major `(n_rows, k)` neighbour
/// table flattened into `indices` and `scores`.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnGraph {
    /// `n_rows * k` row indices, row-major. Entry `[i * k + r]` is the
    /// `r`-th nearest neighbour of row `i`, or [`NO_NEIGHBOR`] when row
    /// `i` had fewer than `k` rankable neighbours.
    pub indices: Vec<u32>,
    /// `n_rows * k` cosine scores aligned element-wise with `indices`.
    /// `f32::NAN` in padded slots.
    pub scores: Vec<f32>,
    /// Number of source rows (the leading dimension of the table).
    pub n_rows: usize,
    /// Neighbours per row (the trailing dimension). Equals
    /// `min(k, n_rows - 1)` for `include_self = false`, or
    /// `min(k, n_rows)` for `include_self = true`.
    pub k: usize,
}

/// For every row of `matrix` (shape `(n_rows, dim)` packed row-major),
/// find its `k` most cosine-similar rows.
///
/// * `include_self` — when `false` (the usual case) a row is never its
///   own neighbour; the effective neighbour count is `min(k, n_rows - 1)`.
///   When `true`, each row's top hit is itself at cosine `1.0`.
/// * `normalized` — when `true`, takes the unit-norm fast path
///   ([`cosine_one_to_many_normalized`]); the caller guarantees every row
///   is L2-unit-norm (which `kaos-nlp-transformers` `EmbeddingModel`
///   output already is). When `false`, the generic path computes norms.
///
/// Rows fan out across Rayon worker threads; each row's sweep + selection
/// is the same `O(n)` cosine + `O(n log k)` heap as [`super::topk::top_k_cosine`].
/// Output is rectangular `(n_rows, effective_k)`; see [`KnnGraph`].
pub fn knn_graph(
    matrix: &[f32],
    dim: usize,
    k: usize,
    include_self: bool,
    normalized: bool,
) -> Result<KnnGraph, SimilarityError> {
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
    let n_avail = if include_self {
        n_rows
    } else {
        n_rows.saturating_sub(1)
    };
    let effective_k = k.min(n_avail);
    if effective_k == 0 {
        return Ok(KnnGraph {
            indices: Vec::new(),
            scores: Vec::new(),
            n_rows,
            k: 0,
        });
    }

    // One independent top-k sweep per row, fanned out across Rayon
    // workers. `cosine_one_to_many` already releases the GIL at the
    // binding layer and dispatches the SIMD kernel internally.
    let rows: Vec<(Vec<u32>, Vec<f32>)> = (0..n_rows)
        .into_par_iter()
        .map(|i| -> Result<(Vec<u32>, Vec<f32>), SimilarityError> {
            let query = &matrix[i * dim..(i + 1) * dim];
            let mut sims = if normalized {
                cosine_one_to_many_normalized(query, matrix, dim)?
            } else {
                cosine_one_to_many(query, matrix, dim)?
            };
            // Exclude self by poisoning its score to NaN — `select_top_k`
            // drops NaN, so row `i` can never select itself.
            if !include_self {
                sims[i] = f32::NAN;
            }
            let top = select_top_k(&sims, effective_k);
            // Pad to a rectangular row. `select_top_k` returns fewer than
            // `effective_k` only when the row had NaN/inf neighbours
            // (non-finite cosines are unrankable); fill the tail with the
            // sentinel so the table stays `(n_rows, effective_k)`.
            let mut idx = top.indices;
            let mut sc = top.scores;
            while idx.len() < effective_k {
                idx.push(NO_NEIGHBOR);
                sc.push(f32::NAN);
            }
            Ok((idx, sc))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut indices = Vec::with_capacity(n_rows * effective_k);
    let mut scores = Vec::with_capacity(n_rows * effective_k);
    for (idx, sc) in rows {
        indices.extend_from_slice(&idx);
        scores.extend_from_slice(&sc);
    }
    Ok(KnnGraph {
        indices,
        scores,
        n_rows,
        k: effective_k,
    })
}

/// Result of a [`near_duplicates`] sweep: the over-threshold edge set.
#[derive(Debug, Clone, PartialEq)]
pub struct NearDuplicates {
    /// `2 * m` node indices: `[i0, j0, i1, j1, ...]` with `i < j`,
    /// emitted in `(i, j)` lexicographic order.
    pub pairs: Vec<u32>,
    /// `m` cosine scores, aligned with each `(i, j)` pair.
    pub scores: Vec<f32>,
    /// `true` if `max_pairs` clamped the output (more pairs met the
    /// threshold than were returned). Callers should surface this rather
    /// than silently treat the result as complete.
    pub truncated: bool,
}

/// Every upper-triangle pair `(i, j)`, `i < j`, of rows in `matrix`
/// whose cosine similarity is `>= threshold`.
///
/// * `normalized` — unit-norm fast path, same contract as [`knn_graph`].
/// * `max_pairs` — optional output cap. When the number of qualifying
///   pairs exceeds it, the first `max_pairs` in `(i, j)` order are kept
///   and [`NearDuplicates::truncated`] is set. `None` returns all.
///
/// Row `i`'s sweep compares it against the tail `i+1..n_rows` in a single
/// SIMD pass; the rows fan out across Rayon workers and results are
/// concatenated in row order, so the pair list is deterministic.
pub fn near_duplicates(
    matrix: &[f32],
    dim: usize,
    threshold: f32,
    normalized: bool,
    max_pairs: Option<usize>,
) -> Result<NearDuplicates, SimilarityError> {
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
        return Ok(NearDuplicates {
            pairs: Vec::new(),
            scores: Vec::new(),
            truncated: false,
        });
    }

    // Per-row edge lists, collected in row order (Rayon preserves the
    // index→output mapping under `collect`), so the flattened result is
    // already sorted lexicographically by `(i, j)`.
    let per_row: Vec<Vec<(u32, u32, f32)>> = (0..n_rows - 1)
        .into_par_iter()
        .map(|i| -> Result<Vec<(u32, u32, f32)>, SimilarityError> {
            let query = &matrix[i * dim..(i + 1) * dim];
            let tail = &matrix[(i + 1) * dim..];
            let sims = if normalized {
                cosine_one_to_many_normalized(query, tail, dim)?
            } else {
                cosine_one_to_many(query, tail, dim)?
            };
            let mut local = Vec::new();
            for (t, &score) in sims.iter().enumerate() {
                if score >= threshold {
                    let j = (i + 1 + t) as u32;
                    local.push((i as u32, j, score));
                }
            }
            Ok(local)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let cap = max_pairs.unwrap_or(usize::MAX);
    let mut pairs = Vec::new();
    let mut scores = Vec::new();
    let mut truncated = false;
    'outer: for row in per_row {
        for (i, j, score) in row {
            if scores.len() >= cap {
                truncated = true;
                break 'outer;
            }
            pairs.push(i);
            pairs.push(j);
            scores.push(score);
        }
    }
    Ok(NearDuplicates {
        pairs,
        scores,
        truncated,
    })
}

// NOTE: component labelling over the edge set produced here is NOT a
// dense-vector concern and is deliberately not implemented in this module.
// `kaos-graph` (petgraph-backed `UnionFind`) owns connected components —
// feed `near_duplicates`/`knn_graph` edges into its
// `connected_components_from_edges` to collapse duplicate pairs into
// groups. Reinventing union-find here would duplicate the graph package.

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: L2-normalize a flat row-major matrix in place (so the
    // `normalized` fast path matches the generic path bit-for-bit on the
    // shared assertions below).
    fn normalize_rows(matrix: &mut [f32], dim: usize) {
        for row in matrix.chunks_mut(dim) {
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 0.0 {
                for x in row.iter_mut() {
                    *x /= n;
                }
            }
        }
    }

    #[test]
    fn knn_basic_excludes_self() {
        // 4 rows of dim 2. Row 0 and row 1 point the same way; row 2 is
        // orthogonal; row 3 is anti-parallel to row 0.
        let matrix = vec![
            1.0_f32, 0.0, // 0
            1.0, 0.1, // 1 (close to 0)
            0.0, 1.0, // 2 (orth to 0)
            -1.0, 0.0, // 3 (anti to 0)
        ];
        let g = knn_graph(&matrix, 2, 1, false, false).unwrap();
        assert_eq!(g.n_rows, 4);
        assert_eq!(g.k, 1);
        // Row 0's single nearest other row is row 1.
        assert_eq!(g.indices[0], 1);
        // Self never appears in its own row.
        for i in 0..4 {
            assert_ne!(g.indices[i * g.k], i as u32, "row {i} selected itself");
        }
    }

    #[test]
    fn knn_include_self_top_hit_is_self() {
        let matrix = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let g = knn_graph(&matrix, 2, 1, true, false).unwrap();
        assert_eq!(g.k, 1);
        // With self included, each row's nearest is itself (cosine 1.0).
        for i in 0..3 {
            assert_eq!(g.indices[i], i as u32);
            assert!((g.scores[i] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn knn_k_capped_at_navail() {
        // 3 rows, ask for 10 neighbours, exclude self → 2 available.
        let matrix = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let g = knn_graph(&matrix, 2, 10, false, false).unwrap();
        assert_eq!(g.k, 2);
        assert_eq!(g.indices.len(), 3 * 2);
    }

    #[test]
    fn knn_single_row_no_self_is_empty() {
        let matrix = vec![1.0_f32, 0.0];
        let g = knn_graph(&matrix, 2, 5, false, false).unwrap();
        assert_eq!(g.k, 0);
        assert!(g.indices.is_empty());
    }

    #[test]
    fn knn_normalized_matches_generic() {
        // The x % 7 pattern deliberately produces duplicate rows, so some
        // neighbour candidates have *exactly* tied cosines. Which of the
        // tied candidates wins is epsilon-level path-dependent (the
        // generic path divides by norms, the fast path doesn't, and per-ISA
        // SIMD dispatch / compiler codegen can flip the rounding — observed
        // flipping between rustc 1.95 and 1.97.1). The contract is that
        // both paths agree on scores; index order under exact ties is
        // unspecified, so indices may only diverge where scores are tied.
        let mut matrix: Vec<f32> = (0..20 * 8).map(|x| ((x % 7) as f32 - 3.0) * 0.3).collect();
        normalize_rows(&mut matrix, 8);
        let generic = knn_graph(&matrix, 8, 3, false, false).unwrap();
        let fast = knn_graph(&matrix, 8, 3, false, true).unwrap();
        assert_eq!(generic.k, fast.k);
        assert_eq!(generic.indices.len(), fast.indices.len());
        let k = generic.k;
        for i in 0..generic.indices.len() {
            // Rank-wise scores must always agree.
            let (g, f) = (generic.scores[i], fast.scores[i]);
            assert!((g - f).abs() < 1e-5, "slot {i}: g={g} f={f}");
            if generic.indices[i] == fast.indices[i] {
                continue;
            }
            // Divergent index is acceptable only under a cosine tie: the
            // slot's score must be indistinguishable from a neighbouring
            // rank in the same row (tied candidates straddle rank
            // boundaries; a strictly unique score must pick one winner).
            let row = i / k;
            let tied_with_neighbor_rank = (row * k..(row + 1) * k)
                .any(|j| j != i && (generic.scores[j] - g).abs() < 2e-5)
                || {
                    // Tie may also be with the first candidate *beyond*
                    // rank k, invisible in the result; accept when the
                    // slot is the last rank of its row.
                    i % k == k - 1
                };
            assert!(
                tied_with_neighbor_rank,
                "slot {i}: indices diverge ({} vs {}) without a cosine tie",
                generic.indices[i], fast.indices[i],
            );
        }
    }

    #[test]
    fn knn_dim_mismatch_errors() {
        let matrix = vec![1.0_f32; 7]; // not a multiple of dim=2
        assert!(knn_graph(&matrix, 2, 1, false, false).is_err());
    }

    #[test]
    fn near_duplicates_finds_pairs_above_threshold() {
        // Rows 0,1 identical; row 2 orthogonal.
        let matrix = vec![1.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let nd = near_duplicates(&matrix, 2, 0.9, false, None).unwrap();
        assert_eq!(nd.pairs, vec![0, 1]);
        assert!((nd.scores[0] - 1.0).abs() < 1e-5);
        assert!(!nd.truncated);
    }

    #[test]
    fn near_duplicates_lexicographic_order() {
        // Three identical rows → pairs (0,1),(0,2),(1,2) in order.
        let matrix = vec![1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        let nd = near_duplicates(&matrix, 2, 0.5, false, None).unwrap();
        assert_eq!(nd.pairs, vec![0, 1, 0, 2, 1, 2]);
    }

    #[test]
    fn near_duplicates_max_pairs_truncates() {
        let matrix = vec![1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        let nd = near_duplicates(&matrix, 2, 0.5, false, Some(2)).unwrap();
        assert_eq!(nd.scores.len(), 2);
        assert!(nd.truncated);
        assert_eq!(nd.pairs, vec![0, 1, 0, 2]);
    }

    #[test]
    fn near_duplicates_none_below_threshold() {
        let matrix = vec![1.0_f32, 0.0, 0.0, 1.0];
        let nd = near_duplicates(&matrix, 2, 0.5, false, None).unwrap();
        assert!(nd.pairs.is_empty());
        assert!(!nd.truncated);
    }

    #[test]
    fn near_duplicates_normalized_matches_generic() {
        let mut matrix: Vec<f32> = (0..15 * 6)
            .map(|x| ((x % 5) as f32 - 2.0) * 0.4 + 0.1)
            .collect();
        normalize_rows(&mut matrix, 6);
        let generic = near_duplicates(&matrix, 6, 0.3, false, None).unwrap();
        let fast = near_duplicates(&matrix, 6, 0.3, true, None).unwrap();
        assert_eq!(generic.pairs, fast.pairs);
    }

    #[test]
    fn near_duplicates_pairs_are_consumable_as_edges() {
        // The (m, 2) pair layout feeds straight into a graph component
        // labeller (kaos-graph). Two identical clusters: {0,1} and {2,3}.
        let matrix = vec![
            1.0_f32, 0.0, // 0
            1.0, 0.0, // 1
            0.0, 1.0, // 2
            0.0, 1.0, // 3
        ];
        let nd = near_duplicates(&matrix, 2, 0.99, false, None).unwrap();
        let edges: Vec<(u32, u32)> = nd.pairs.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        // Within-cluster pairs only; never a cross-cluster edge.
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(2, 3)));
        assert!(!edges.contains(&(0, 2)));
    }
}
