//! Generic unit packer — the core loop ported from
//! ``kaos_nlp_core/chunking/_pack.py::pack_units``.
//!
//! Walks an ordered sequence of units described by three parallel
//! arrays (`start`, `end`, `token_count`), greedy-packs them into
//! groups that stay under `max_tokens`, and returns the groupings
//! as a `Vec<PackedGroup>`. The packer never splits an individual
//! unit; an oversize unit emits as its own group, and the Python
//! caller is responsible for subdividing it via a finer-grained
//! chunker (see `ParagraphChunker` falling back to `SentenceChunker`
//! on oversize paragraphs).
//!
//! Determinism: pure function of its inputs. No allocations beyond
//! the result vector. Identical inputs → identical bit output across
//! processes, hash seeds, parallelism.

use thiserror::Error;

/// A single grouping decision — the result of packing one or more
/// adjacent units into a chunk.
///
/// Offsets are in the same coordinate space as the input `start`
/// / `end` arrays (typically character offsets in the source
/// document). `unit_start`/`unit_end` are half-open indices into the
/// caller's unit list (`units[unit_start..unit_end]` = the units in
/// this group). `unit_token_sum` is the *sum of constituent
/// `token_count` values* — the eventual chunk's full token count
/// may differ (because the chunk text includes inter-unit
/// whitespace), and the Python caller recomputes it from the
/// assembled slice. The sum lets the Python caller verify the
/// budget invariant without re-summing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedGroup {
    /// Start char offset of the first unit in the group.
    pub start: u32,
    /// End char offset (exclusive) of the last unit in the group.
    pub end: u32,
    /// Index of the first unit (inclusive).
    pub unit_start: u32,
    /// Index of the last unit (exclusive).
    pub unit_end: u32,
    /// Sum of constituent units' `token_count` values.
    pub unit_token_sum: u32,
}

/// Errors raised by `pack_units` on invalid configuration.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PackError {
    /// `max_tokens` was zero or negative.
    #[error("max_tokens must be > 0, got {0}")]
    InvalidMaxTokens(i64),
    /// Parallel-array length mismatch.
    #[error(
        "parallel-array length mismatch: starts={starts}, ends={ends}, token_counts={token_counts}"
    )]
    LengthMismatch {
        starts: usize,
        ends: usize,
        token_counts: usize,
    },
    /// A single unit had `end <= start` after filtering.
    #[error("unit {index} has end ({end}) <= start ({start})")]
    InvalidUnitBounds { index: usize, start: u32, end: u32 },
}

/// Greedy-pack units into groups that stay under `max_tokens`.
///
/// Arguments
/// ---------
///
/// * `starts` — per-unit start char offsets.
/// * `ends` — per-unit end char offsets (exclusive). Must be the
///   same length as `starts`.
/// * `token_counts` — per-unit token-count estimates. Must be the
///   same length as `starts`. Used to decide which units fit; the
///   Python caller separately recomputes the chunk's final token
///   count from the assembled slice.
/// * `max_tokens` — soft ceiling on the sum of constituent unit
///   token counts in one group. A group may exceed the ceiling when
///   it consists of a single oversize unit.
/// * `overlap_units` — number of trailing units from the just-flushed
///   group to repeat at the start of the next group. `0` = no
///   overlap (contiguous groups).
///
/// Behavior
/// --------
///
/// * Units with `end <= start` (zero-length) are filtered before
///   packing (matches the Python implementation).
/// * Empty input returns an empty `Vec` (not an error).
/// * The packer is greedy: it keeps appending units until the next
///   one would exceed the budget, then flushes.
/// * Single oversize units flush immediately (one group containing
///   only that unit, even if `unit_token_sum > max_tokens`).
pub fn pack_units(
    starts: &[u32],
    ends: &[u32],
    token_counts: &[u32],
    max_tokens: u32,
    overlap_units: u32,
) -> Result<Vec<PackedGroup>, PackError> {
    if max_tokens == 0 {
        return Err(PackError::InvalidMaxTokens(0));
    }
    if starts.len() != ends.len() || starts.len() != token_counts.len() {
        return Err(PackError::LengthMismatch {
            starts: starts.len(),
            ends: ends.len(),
            token_counts: token_counts.len(),
        });
    }
    if starts.is_empty() {
        return Ok(Vec::new());
    }

    // Validate bounds + filter zero-length units in one pass into a
    // small scratch Vec of (orig_idx, start, end, token_count). We
    // need to remember the *original* index so the returned
    // `unit_start` / `unit_end` map back to the caller's unit list,
    // not to our filtered view.
    let mut filtered: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(starts.len());
    for i in 0..starts.len() {
        let s = starts[i];
        let e = ends[i];
        if e <= s {
            // Match the Python contract: silently drop zero-length
            // units (rather than erroring) — this is the documented
            // behavior in `_pack.py::pack_units` and exercised by the
            // existing test suite.
            continue;
        }
        filtered.push((i as u32, s, e, token_counts[i]));
    }
    if filtered.is_empty() {
        return Ok(Vec::new());
    }

    let mut groups: Vec<PackedGroup> = Vec::new();
    // Current group's running members, as indices into `filtered`.
    // We materialise indices rather than copies to keep the inner
    // loop allocation-free after the initial Vec growth.
    let mut current: Vec<usize> = Vec::with_capacity(16);
    let mut current_tokens: u32 = 0;

    // Closure-equivalent flush: emit the current group, then prepare
    // the overlap tail (if any) for the next group.
    let flush = |current: &mut Vec<usize>,
                 current_tokens: &mut u32,
                 groups: &mut Vec<PackedGroup>,
                 filtered: &[(u32, u32, u32, u32)],
                 overlap_units: u32| {
        if current.is_empty() {
            return;
        }
        let first_idx = current[0];
        let last_idx = *current.last().unwrap();
        let (first_orig, first_start, _, _) = filtered[first_idx];
        let (last_orig, _, last_end, _) = filtered[last_idx];
        groups.push(PackedGroup {
            start: first_start,
            end: last_end,
            unit_start: first_orig,
            unit_end: last_orig + 1,
            unit_token_sum: *current_tokens,
        });
        // Set up the next group's overlap tail.
        if overlap_units > 0 {
            let keep = (overlap_units as usize).min(current.len());
            let tail_start = current.len() - keep;
            // Recompute token-sum from the tail. Cheap — overlap is
            // bounded and small in practice (1-5 units typically).
            let new_tokens: u32 = current[tail_start..]
                .iter()
                .map(|&idx| filtered[idx].3)
                .sum();
            current.drain(..tail_start);
            *current_tokens = new_tokens;
        } else {
            current.clear();
            *current_tokens = 0;
        }
    };

    for (filtered_idx, &(_orig_idx, _start, _end, token_count)) in filtered.iter().enumerate() {
        // Saturating add — pathological inputs (corrupted token_counts
        // summing past u32::MAX) wouldn't be useful, but we also
        // shouldn't panic.
        let would_total = current_tokens.saturating_add(token_count);

        // If the current group has at least one unit and adding this
        // one would exceed the budget, flush first.
        if !current.is_empty() && would_total > max_tokens {
            flush(
                &mut current,
                &mut current_tokens,
                &mut groups,
                &filtered,
                overlap_units,
            );
        }
        current.push(filtered_idx);
        current_tokens = current_tokens.saturating_add(token_count);

        // If a single unit on its own already exceeds the budget,
        // emit it immediately so the caller can subdivide. Matches
        // the Python contract.
        if current.len() == 1 && current_tokens > max_tokens {
            flush(
                &mut current,
                &mut current_tokens,
                &mut groups,
                &filtered,
                overlap_units,
            );
        }
    }

    flush(
        &mut current,
        &mut current_tokens,
        &mut groups,
        &filtered,
        overlap_units,
    );
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _pack(
        starts: &[u32],
        ends: &[u32],
        tokens: &[u32],
        max_tokens: u32,
        overlap_units: u32,
    ) -> Vec<PackedGroup> {
        pack_units(starts, ends, tokens, max_tokens, overlap_units).unwrap()
    }

    #[test]
    fn empty_input_returns_empty() {
        let g = _pack(&[], &[], &[], 100, 0);
        assert!(g.is_empty());
    }

    #[test]
    fn single_unit_under_budget() {
        let g = _pack(&[0], &[10], &[5], 100, 0);
        assert_eq!(
            g,
            vec![PackedGroup {
                start: 0,
                end: 10,
                unit_start: 0,
                unit_end: 1,
                unit_token_sum: 5,
            }]
        );
    }

    #[test]
    fn single_oversize_unit_emits_alone() {
        let g = _pack(&[0], &[100], &[500], 100, 0);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].unit_token_sum, 500);
    }

    #[test]
    fn budget_fits_all_in_one() {
        let g = _pack(&[0, 10, 20, 30], &[10, 20, 30, 40], &[2, 3, 4, 5], 100, 0);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].start, 0);
        assert_eq!(g[0].end, 40);
        assert_eq!(g[0].unit_token_sum, 14);
        assert_eq!(g[0].unit_start, 0);
        assert_eq!(g[0].unit_end, 4);
    }

    #[test]
    fn budget_forces_split() {
        // Four units of tokens [5, 5, 4, 2], total 16, budget 8.
        // Trace:
        //   u0 (5):   current=[0], tokens=5.
        //   u1 (5):   5+5=10>8 → flush [0]. current=[1], tokens=5.
        //   u2 (4):   5+4=9>8 → flush [1]. current=[2], tokens=4.
        //   u3 (2):   4+2=6 ≤8 → current=[2,3], tokens=6.
        //   end:      flush [2,3].
        // → 3 groups: [0], [1], [2,3].
        let g = _pack(&[0, 10, 20, 30], &[10, 20, 30, 40], &[5, 5, 4, 2], 8, 0);
        assert_eq!(g.len(), 3);
        assert_eq!(g[0].unit_start, 0);
        assert_eq!(g[0].unit_end, 1);
        assert_eq!(g[0].unit_token_sum, 5);
        assert_eq!(g[1].unit_start, 1);
        assert_eq!(g[1].unit_end, 2);
        assert_eq!(g[2].unit_start, 2);
        assert_eq!(g[2].unit_end, 4);
        assert_eq!(g[2].unit_token_sum, 6);
    }

    #[test]
    fn zero_length_units_filtered() {
        // Unit 1 has end == start — should be dropped.
        let g = _pack(&[0, 10, 10, 20], &[10, 10, 20, 30], &[2, 2, 3, 4], 100, 0);
        // Expect 3 surviving units in one group.
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].unit_token_sum, 2 + 3 + 4);
        // The unit indices map back to the *original* indices,
        // skipping the dropped one. So unit_start=0, unit_end=4
        // (last orig idx 3 + 1).
        assert_eq!(g[0].unit_start, 0);
        assert_eq!(g[0].unit_end, 4);
    }

    #[test]
    fn overlap_tail_propagates() {
        // 4 units of 3 tokens each = 12; budget 6 → 2 groups w/o
        // overlap (3+3 each), or with overlap_units=1 we expect the
        // last unit of group 0 to start group 1.
        let g = _pack(&[0, 10, 20, 30], &[10, 20, 30, 40], &[3, 3, 3, 3], 6, 1);
        // Without overlap: groups (0,1), (2,3) — 2 groups.
        // With overlap=1: group 0 keeps unit 1 as overlap, so
        // group 1 = (1, 2), group 2 = (3,) (or (2, 3) etc).
        // Iteration with overlap=1:
        //   unit0 (3): current=[0], tokens=3
        //   unit1 (3): 3+3=6 ≤6 → current=[0,1], tokens=6
        //   unit2 (3): 6+3=9>6 → flush [0,1]. Group 0: start=0 end=20 unit_start=0 unit_end=2 sum=6.
        //     Overlap=1: keep last unit (idx 1), current=[1], tokens=3.
        //   unit2 (3): 3+3=6 ≤6 → current=[1,2], tokens=6
        //   unit3 (3): 6+3=9>6 → flush [1,2]. Group 1: start=10 end=30 unit_start=1 unit_end=3 sum=6.
        //     Overlap=1: keep last unit (idx 2), current=[2], tokens=3.
        //   unit3 (3): 3+3=6 ≤6 → current=[2,3], tokens=6
        //   end → flush [2,3]. Group 2: start=20 end=40 unit_start=2 unit_end=4 sum=6.
        assert_eq!(g.len(), 3);
        assert_eq!(g[0].unit_start, 0);
        assert_eq!(g[0].unit_end, 2);
        assert_eq!(g[1].unit_start, 1);
        assert_eq!(g[1].unit_end, 3);
        assert_eq!(g[2].unit_start, 2);
        assert_eq!(g[2].unit_end, 4);
    }

    #[test]
    fn rejects_zero_max_tokens() {
        let r = pack_units(&[0], &[10], &[5], 0, 0);
        assert!(matches!(r, Err(PackError::InvalidMaxTokens(_))));
    }

    #[test]
    fn rejects_length_mismatch() {
        let r = pack_units(&[0, 1], &[10], &[5], 100, 0);
        assert!(matches!(r, Err(PackError::LengthMismatch { .. })));
    }
}
