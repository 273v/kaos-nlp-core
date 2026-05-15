//! Semantic packer — pure-Rust port of the inner loop in
//! ``kaos_nlp_transformers/chunking.py::SemanticChunker._pack``.
//!
//! Where :func:`pack::pack_units` splits purely on a token budget,
//! :func:`semantic_pack` additionally inserts a cut whenever the
//! cosine similarity between two *adjacent* unit embeddings drops
//! below `drop_threshold` — the empirical signal that the document
//! has shifted topic. The Python caller has already computed the
//! `n_units - 1` adjacency similarities via
//! :func:`kaos_nlp_core.similarity.cosine_adjacent` (SIMD-dispatched
//! via NumKong), so this kernel only needs:
//!
//! * `starts` / `ends` — per-unit char offsets, length `n`.
//! * `token_counts` — per-unit token-count estimates, length `n`.
//! * `adj_sim` — adjacent cosine similarities, length `n - 1` (or
//!   `0` if `n <= 1`).
//! * `max_tokens` — token budget per chunk.
//! * `drop_threshold` — cosine threshold below which a cut is forced.
//!
//! Determinism: pure function of its inputs. The Rust kernel emits
//! the same group boundaries the Python version produced.
//!
//! The current implementation does not support `overlap_units` —
//! SemanticChunker never used overlap in production (topical cuts
//! make overlap less meaningful). Adding overlap support is
//! mechanical (mirror the tail logic in `pack::pack_units`) when a
//! downstream consumer needs it.

use thiserror::Error;

use super::PackedGroup;

/// Errors raised by `semantic_pack` on invalid configuration.
#[derive(Debug, Error, PartialEq)]
pub enum SemanticPackError {
    /// `max_tokens` was zero.
    #[error("max_tokens must be > 0, got {0}")]
    InvalidMaxTokens(i64),
    /// `drop_threshold` outside `[-1, 1]`. We allow negative
    /// thresholds (downstream uses `[0, 1]` but the cosine kernel
    /// clips to `[-1, 1]`).
    #[error("drop_threshold must be in [-1, 1], got {0}")]
    InvalidDropThreshold(f32),
    /// Parallel-array length mismatch among `starts` / `ends` /
    /// `token_counts`.
    #[error(
        "parallel-array length mismatch: starts={starts}, ends={ends}, token_counts={token_counts}"
    )]
    LengthMismatch {
        starts: usize,
        ends: usize,
        token_counts: usize,
    },
    /// `adj_sim` length must equal `starts.len().saturating_sub(1)`.
    #[error("adj_sim length must equal n_units - 1; got adj_sim={adj_sim}, n_units={n_units}")]
    AdjSimLength { adj_sim: usize, n_units: usize },
}

/// Greedy-pack units into groups respecting both a token budget and
/// topic-shift cuts driven by adjacent cosine similarity.
///
/// See module docs for the high-level contract. The kernel does
/// **not** filter zero-length units — the Python caller is expected
/// to pre-filter, because the semantic chunker takes its units from
/// `kaos_nlp_core.segmentation`, which already strips empties.
pub fn semantic_pack(
    starts: &[u32],
    ends: &[u32],
    token_counts: &[u32],
    adj_sim: &[f32],
    max_tokens: u32,
    drop_threshold: f32,
) -> Result<Vec<PackedGroup>, SemanticPackError> {
    if max_tokens == 0 {
        return Err(SemanticPackError::InvalidMaxTokens(0));
    }
    if !(-1.0..=1.0).contains(&drop_threshold) {
        return Err(SemanticPackError::InvalidDropThreshold(drop_threshold));
    }
    if starts.len() != ends.len() || starts.len() != token_counts.len() {
        return Err(SemanticPackError::LengthMismatch {
            starts: starts.len(),
            ends: ends.len(),
            token_counts: token_counts.len(),
        });
    }
    let n_units = starts.len();
    let expected_adj = n_units.saturating_sub(1);
    if adj_sim.len() != expected_adj {
        return Err(SemanticPackError::AdjSimLength {
            adj_sim: adj_sim.len(),
            n_units,
        });
    }
    if n_units == 0 {
        return Ok(Vec::new());
    }

    let mut groups: Vec<PackedGroup> = Vec::new();
    // Index into `starts` of the first unit in the active group, or
    // `None` if no group is open (i.e., we just flushed).
    let mut group_start_idx: Option<usize> = None;
    let mut current_tokens: u32 = 0;

    for index in 0..n_units {
        let unit_tokens = token_counts[index];
        match group_start_idx {
            None => {
                // First unit of a fresh group — always accepted, even
                // if it's individually oversize (matches Python).
                group_start_idx = Some(index);
                current_tokens = unit_tokens;
            }
            Some(gs) => {
                // index >= 1, so adj_sim[index - 1] is in range.
                let similarity = adj_sim[index - 1];
                let would_exceed = current_tokens.saturating_add(unit_tokens) > max_tokens;
                let topic_shift = similarity < drop_threshold;
                if would_exceed || topic_shift {
                    // Flush the current group, then start a new one
                    // containing `index`.
                    let last = index - 1;
                    groups.push(PackedGroup {
                        start: starts[gs],
                        end: ends[last],
                        unit_start: gs as u32,
                        unit_end: index as u32,
                        unit_token_sum: current_tokens,
                    });
                    group_start_idx = Some(index);
                    current_tokens = unit_tokens;
                } else {
                    current_tokens = current_tokens.saturating_add(unit_tokens);
                }
            }
        }
    }

    if let Some(gs) = group_start_idx {
        let last = n_units - 1;
        groups.push(PackedGroup {
            start: starts[gs],
            end: ends[last],
            unit_start: gs as u32,
            unit_end: n_units as u32,
            unit_token_sum: current_tokens,
        });
    }

    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        starts: &[u32],
        ends: &[u32],
        tokens: &[u32],
        adj: &[f32],
        max_tokens: u32,
        threshold: f32,
    ) -> Vec<PackedGroup> {
        semantic_pack(starts, ends, tokens, adj, max_tokens, threshold).unwrap()
    }

    #[test]
    fn empty_returns_empty() {
        let g = run(&[], &[], &[], &[], 100, 0.5);
        assert!(g.is_empty());
    }

    #[test]
    fn single_unit_returns_one_group() {
        let g = run(&[0], &[10], &[5], &[], 100, 0.5);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].unit_start, 0);
        assert_eq!(g[0].unit_end, 1);
        assert_eq!(g[0].unit_token_sum, 5);
    }

    #[test]
    fn high_similarity_keeps_units_together() {
        // 3 units, all highly similar, plenty of budget → 1 group.
        let g = run(
            &[0, 10, 20],
            &[10, 20, 30],
            &[2, 2, 2],
            &[0.9, 0.95],
            100,
            0.5,
        );
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].unit_token_sum, 6);
    }

    #[test]
    fn low_similarity_forces_split() {
        // 3 units; similarity (0->1) is high but (1->2) is below
        // threshold → cut between units 1 and 2.
        let g = run(
            &[0, 10, 20],
            &[10, 20, 30],
            &[2, 2, 2],
            &[0.9, 0.1],
            100,
            0.5,
        );
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].unit_start, 0);
        assert_eq!(g[0].unit_end, 2);
        assert_eq!(g[1].unit_start, 2);
        assert_eq!(g[1].unit_end, 3);
    }

    #[test]
    fn budget_forces_split_when_similarity_high() {
        // 3 units of 5 tokens each, all similar → budget 8 splits
        // after unit 0 (since 5+5>8). Threshold doesn't engage.
        let g = run(
            &[0, 10, 20],
            &[10, 20, 30],
            &[5, 5, 5],
            &[0.99, 0.99],
            8,
            0.5,
        );
        assert_eq!(g.len(), 3);
        assert_eq!(g[0].unit_token_sum, 5);
    }

    #[test]
    fn single_oversize_unit_emits_alone_even_with_low_threshold() {
        // 2 units, first one is oversize. Second unit gets cut
        // because it would exceed budget (single oversize unit
        // emits, then unit 2 starts fresh).
        let g = run(&[0, 100], &[100, 110], &[500, 1], &[0.9], 50, 0.5);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].unit_token_sum, 500);
        assert_eq!(g[1].unit_token_sum, 1);
    }

    #[test]
    fn threshold_at_boundary_uses_strict_lt() {
        // similarity == drop_threshold → no cut (strict <).
        let g = run(&[0, 10], &[10, 20], &[1, 1], &[0.5], 100, 0.5);
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn rejects_zero_max_tokens() {
        let r = semantic_pack(&[0], &[10], &[5], &[], 0, 0.5);
        assert!(matches!(r, Err(SemanticPackError::InvalidMaxTokens(_))));
    }

    #[test]
    fn rejects_threshold_out_of_range() {
        let r = semantic_pack(&[0], &[10], &[5], &[], 100, 2.0);
        assert!(matches!(r, Err(SemanticPackError::InvalidDropThreshold(_))));
    }

    #[test]
    fn rejects_length_mismatch() {
        let r = semantic_pack(&[0, 1], &[10], &[5, 5], &[0.5], 100, 0.5);
        assert!(matches!(r, Err(SemanticPackError::LengthMismatch { .. })));
    }

    #[test]
    fn rejects_adj_sim_length() {
        let r = semantic_pack(&[0, 1], &[10, 20], &[5, 5], &[0.5, 0.5], 100, 0.5);
        assert!(matches!(r, Err(SemanticPackError::AdjSimLength { .. })));
    }
}
