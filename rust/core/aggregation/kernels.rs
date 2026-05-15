//! Aggregation kernels — see :mod:`super` for the input contract.
//!
//! Each function returns either a single winner id (`Option<u32>`)
//! or a list of qualifying ids (`Vec<u32>`). For multi-label
//! kernels we always return ids in **ascending order** so the
//! Python caller, which assigned ids in first-appearance order,
//! gets a frozenset whose iteration order matches the original
//! pure-Python implementation.

use thiserror::Error;

/// Errors raised by the aggregation kernels on malformed inputs.
///
/// Validation is cheap and explicit. The Python wrappers translate
/// these into `ValueError` so the Python API contract matches the
/// pure-Python version. Ranges noted in the variants are the
/// closed/open ranges actually checked by the kernels.
#[derive(Debug, Error, PartialEq)]
pub enum AggregationError {
    /// `chunk_offsets` length must be exactly `n_chunks + 1`. Empty
    /// inputs are represented as `chunk_offsets = [0]`.
    #[error("chunk_offsets must have length >= 1, got {0}")]
    EmptyOffsets(usize),
    /// Cumulative offsets must be non-decreasing.
    #[error("chunk_offsets not monotonic at index {index}: {prev} > {curr}")]
    OffsetsNotMonotonic { index: usize, prev: u32, curr: u32 },
    /// The last offset must equal the length of `flat_ids`.
    #[error("last offset ({last_offset}) must equal flat_ids length ({flat_len})")]
    LastOffsetMismatch { last_offset: u32, flat_len: u32 },
    /// A label id was >= `n_labels`.
    #[error("label id {id} out of range (n_labels = {n_labels})")]
    IdOutOfRange { id: u32, n_labels: u32 },
    /// `threshold` outside `(0, 1]` for the kernels that require it.
    #[error("threshold must be in (0, 1], got {0}")]
    InvalidThreshold(f64),
    /// `weights` length must equal `n_chunks` when provided.
    #[error("weights length ({weights}) must equal n_chunks ({n_chunks})")]
    WeightsLengthMismatch { weights: usize, n_chunks: usize },
    /// `flat_scores` length must equal `flat_ids` length.
    #[error("flat_scores length ({scores}) must equal flat_ids length ({ids})")]
    ScoresLengthMismatch { scores: usize, ids: usize },
}

/// Validate the shared inputs and return ``n_chunks``.
///
/// All the kernels share the same CSR-style ragged-array contract, so
/// the validation lives in one place. The kernels do not re-validate.
fn validate(
    flat_ids: &[u32],
    chunk_offsets: &[u32],
    n_labels: u32,
) -> Result<usize, AggregationError> {
    if chunk_offsets.is_empty() {
        return Err(AggregationError::EmptyOffsets(0));
    }
    let n_chunks = chunk_offsets.len() - 1;
    let mut prev = chunk_offsets[0];
    for (i, &curr) in chunk_offsets.iter().enumerate().skip(1) {
        if curr < prev {
            return Err(AggregationError::OffsetsNotMonotonic {
                index: i,
                prev,
                curr,
            });
        }
        prev = curr;
    }
    let last_offset = *chunk_offsets.last().unwrap();
    if last_offset as usize != flat_ids.len() {
        return Err(AggregationError::LastOffsetMismatch {
            last_offset,
            flat_len: flat_ids.len() as u32,
        });
    }
    for &id in flat_ids {
        if id >= n_labels {
            return Err(AggregationError::IdOutOfRange { id, n_labels });
        }
    }
    Ok(n_chunks)
}

/// Plurality vote: return the id appearing in the most chunks.
///
/// Per-chunk duplicates are deduplicated. Ties resolve to the id with
/// the lowest value (i.e., first appearance, by the Python contract).
///
/// Returns ``None`` if every chunk is empty.
pub fn vote(
    flat_ids: &[u32],
    chunk_offsets: &[u32],
    n_labels: u32,
) -> Result<Option<u32>, AggregationError> {
    let n_chunks = validate(flat_ids, chunk_offsets, n_labels)?;
    if n_chunks == 0 || n_labels == 0 {
        return Ok(None);
    }
    let mut counts: Vec<u32> = vec![0; n_labels as usize];
    let mut seen: Vec<u32> = vec![0; n_labels as usize];
    let mut total_picks: u32 = 0;
    for c in 0..n_chunks {
        let stamp = (c as u32) + 1;
        let lo = chunk_offsets[c] as usize;
        let hi = chunk_offsets[c + 1] as usize;
        for &id in &flat_ids[lo..hi] {
            let idx = id as usize;
            if seen[idx] == stamp {
                continue;
            }
            seen[idx] = stamp;
            counts[idx] += 1;
            total_picks += 1;
        }
    }
    if total_picks == 0 {
        return Ok(None);
    }
    let mut best_id: Option<u32> = None;
    let mut best_count: u32 = 0;
    for (id, &count) in counts.iter().enumerate() {
        if count > best_count {
            best_count = count;
            best_id = Some(id as u32);
        }
    }
    Ok(best_id)
}

/// Threshold-gated majority vote. Returns the id appearing in at
/// least `threshold * n_chunks` distinct chunks, with the
/// lowest-id-wins tiebreak. Returns ``None`` when no id qualifies.
pub fn majority(
    flat_ids: &[u32],
    chunk_offsets: &[u32],
    n_labels: u32,
    threshold: f64,
) -> Result<Option<u32>, AggregationError> {
    if !(0.0 < threshold && threshold <= 1.0) {
        return Err(AggregationError::InvalidThreshold(threshold));
    }
    let n_chunks = validate(flat_ids, chunk_offsets, n_labels)?;
    if n_chunks == 0 || n_labels == 0 {
        return Ok(None);
    }
    let required = threshold * (n_chunks as f64);
    let mut counts: Vec<u32> = vec![0; n_labels as usize];
    let mut seen: Vec<u32> = vec![0; n_labels as usize];
    for c in 0..n_chunks {
        let stamp = (c as u32) + 1;
        let lo = chunk_offsets[c] as usize;
        let hi = chunk_offsets[c + 1] as usize;
        for &id in &flat_ids[lo..hi] {
            let idx = id as usize;
            if seen[idx] == stamp {
                continue;
            }
            seen[idx] = stamp;
            counts[idx] += 1;
        }
    }
    for (id, &count) in counts.iter().enumerate() {
        if (count as f64) >= required {
            return Ok(Some(id as u32));
        }
    }
    Ok(None)
}

/// Multi-label union: ids appearing in any chunk, in ascending order.
pub fn union(
    flat_ids: &[u32],
    chunk_offsets: &[u32],
    n_labels: u32,
) -> Result<Vec<u32>, AggregationError> {
    validate(flat_ids, chunk_offsets, n_labels)?;
    if n_labels == 0 {
        return Ok(Vec::new());
    }
    let mut seen: Vec<bool> = vec![false; n_labels as usize];
    for &id in flat_ids {
        seen[id as usize] = true;
    }
    let mut out: Vec<u32> = Vec::new();
    for (id, &flag) in seen.iter().enumerate() {
        if flag {
            out.push(id as u32);
        }
    }
    Ok(out)
}

/// Multi-label intersection: ids appearing in *every* chunk, in
/// ascending order. Empty input → empty output.
pub fn intersection(
    flat_ids: &[u32],
    chunk_offsets: &[u32],
    n_labels: u32,
) -> Result<Vec<u32>, AggregationError> {
    let n_chunks = validate(flat_ids, chunk_offsets, n_labels)?;
    if n_chunks == 0 || n_labels == 0 {
        return Ok(Vec::new());
    }
    // Count distinct-per-chunk occurrences. An id is in the
    // intersection iff its count == n_chunks.
    let mut counts: Vec<u32> = vec![0; n_labels as usize];
    let mut seen: Vec<u32> = vec![0; n_labels as usize];
    for c in 0..n_chunks {
        let stamp = (c as u32) + 1;
        let lo = chunk_offsets[c] as usize;
        let hi = chunk_offsets[c + 1] as usize;
        for &id in &flat_ids[lo..hi] {
            let idx = id as usize;
            if seen[idx] == stamp {
                continue;
            }
            seen[idx] = stamp;
            counts[idx] += 1;
        }
    }
    let need = n_chunks as u32;
    let mut out: Vec<u32> = Vec::new();
    for (id, &count) in counts.iter().enumerate() {
        if count == need {
            out.push(id as u32);
        }
    }
    Ok(out)
}

/// Internal: accumulate per-id weight totals.
///
/// Each chunk contributes `weights[c]` to every distinct id it
/// contains. Returns `(score, total_weight)`. The `total_weight` is
/// the sum of `weights`, returned so the caller can compute the
/// `threshold * total` cutoff without re-summing.
fn weighted_score(
    flat_ids: &[u32],
    chunk_offsets: &[u32],
    weights: &[f64],
    n_labels: u32,
) -> Vec<f64> {
    let n_chunks = chunk_offsets.len() - 1;
    let mut score: Vec<f64> = vec![0.0; n_labels as usize];
    let mut seen: Vec<u32> = vec![0; n_labels as usize];
    for c in 0..n_chunks {
        let w = weights[c];
        let stamp = (c as u32) + 1;
        let lo = chunk_offsets[c] as usize;
        let hi = chunk_offsets[c + 1] as usize;
        for &id in &flat_ids[lo..hi] {
            let idx = id as usize;
            if seen[idx] == stamp {
                continue;
            }
            seen[idx] = stamp;
            score[idx] += w;
        }
    }
    score
}

/// Weighted vote, single-label mode. Returns the highest-scoring id
/// among those crossing the `threshold * sum(weights)` cutoff;
/// ``None`` when no id qualifies. Ties → lowest id.
pub fn weighted_single(
    flat_ids: &[u32],
    chunk_offsets: &[u32],
    weights: &[f64],
    n_labels: u32,
    threshold: f64,
) -> Result<Option<u32>, AggregationError> {
    if !(0.0 < threshold && threshold <= 1.0) {
        return Err(AggregationError::InvalidThreshold(threshold));
    }
    let n_chunks = validate(flat_ids, chunk_offsets, n_labels)?;
    if weights.len() != n_chunks {
        return Err(AggregationError::WeightsLengthMismatch {
            weights: weights.len(),
            n_chunks,
        });
    }
    if n_chunks == 0 || n_labels == 0 {
        return Ok(None);
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Ok(None);
    }
    let required = threshold * total;
    let score = weighted_score(flat_ids, chunk_offsets, weights, n_labels);
    let mut best_id: Option<u32> = None;
    let mut best_score: f64 = f64::NEG_INFINITY;
    for (id, &val) in score.iter().enumerate() {
        if val >= required && val > best_score {
            best_score = val;
            best_id = Some(id as u32);
        }
    }
    Ok(best_id)
}

/// Weighted vote, multi-label mode. Returns ids whose accumulated
/// weight is at least `threshold * sum(weights)`, in ascending order.
pub fn weighted_multi(
    flat_ids: &[u32],
    chunk_offsets: &[u32],
    weights: &[f64],
    n_labels: u32,
    threshold: f64,
) -> Result<Vec<u32>, AggregationError> {
    if !(0.0 < threshold && threshold <= 1.0) {
        return Err(AggregationError::InvalidThreshold(threshold));
    }
    let n_chunks = validate(flat_ids, chunk_offsets, n_labels)?;
    if weights.len() != n_chunks {
        return Err(AggregationError::WeightsLengthMismatch {
            weights: weights.len(),
            n_chunks,
        });
    }
    if n_chunks == 0 || n_labels == 0 {
        return Ok(Vec::new());
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Ok(Vec::new());
    }
    let required = threshold * total;
    let score = weighted_score(flat_ids, chunk_offsets, weights, n_labels);
    let mut out: Vec<u32> = Vec::new();
    for (id, &val) in score.iter().enumerate() {
        if val >= required {
            out.push(id as u32);
        }
    }
    Ok(out)
}

/// Internal: pool per-id max scores.
fn pool_max_scores(flat_ids: &[u32], flat_scores: &[f64], n_labels: u32) -> Vec<Option<f64>> {
    let mut pooled: Vec<Option<f64>> = vec![None; n_labels as usize];
    for (i, &id) in flat_ids.iter().enumerate() {
        let idx = id as usize;
        let v = flat_scores[i];
        pooled[idx] = match pooled[idx] {
            None => Some(v),
            Some(prev) if v > prev => Some(v),
            other => other,
        };
    }
    pooled
}

/// Max-score, single-label mode. Returns the id with the highest
/// max-score; respects `threshold` when provided (top-id must beat
/// `threshold`). Ties → lowest id.
pub fn max_score_single(
    flat_ids: &[u32],
    flat_scores: &[f64],
    n_labels: u32,
    threshold: Option<f64>,
) -> Result<Option<u32>, AggregationError> {
    if flat_scores.len() != flat_ids.len() {
        return Err(AggregationError::ScoresLengthMismatch {
            scores: flat_scores.len(),
            ids: flat_ids.len(),
        });
    }
    for &id in flat_ids {
        if id >= n_labels {
            return Err(AggregationError::IdOutOfRange { id, n_labels });
        }
    }
    if n_labels == 0 {
        return Ok(None);
    }
    let pooled = pool_max_scores(flat_ids, flat_scores, n_labels);
    let mut best_id: Option<u32> = None;
    let mut best_score: f64 = f64::NEG_INFINITY;
    for (id, slot) in pooled.iter().enumerate() {
        if let Some(v) = slot {
            if *v > best_score {
                best_score = *v;
                best_id = Some(id as u32);
            }
        }
    }
    match (best_id, threshold) {
        (Some(_), Some(t)) if best_score < t => Ok(None),
        (id, _) => Ok(id),
    }
}

/// Max-score, multi-label mode. Returns ids whose max-score is
/// strictly greater than the cutoff (`threshold` if provided, else
/// ``0.0``) — matches the Python ``> cutoff`` semantics.
pub fn max_score_multi(
    flat_ids: &[u32],
    flat_scores: &[f64],
    n_labels: u32,
    threshold: Option<f64>,
) -> Result<Vec<u32>, AggregationError> {
    if flat_scores.len() != flat_ids.len() {
        return Err(AggregationError::ScoresLengthMismatch {
            scores: flat_scores.len(),
            ids: flat_ids.len(),
        });
    }
    for &id in flat_ids {
        if id >= n_labels {
            return Err(AggregationError::IdOutOfRange { id, n_labels });
        }
    }
    if n_labels == 0 {
        return Ok(Vec::new());
    }
    let cutoff = threshold.unwrap_or(0.0);
    let pooled = pool_max_scores(flat_ids, flat_scores, n_labels);
    let mut out: Vec<u32> = Vec::new();
    for (id, slot) in pooled.iter().enumerate() {
        if let Some(v) = slot {
            if *v > cutoff {
                out.push(id as u32);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vote_picks_plurality() {
        // 3 chunks: [0,1], [0,2], [1] → counts: 0→2, 1→2, 2→1. Tie 0/1; lowest id wins → 0.
        let flat = vec![0, 1, 0, 2, 1];
        let off = vec![0, 2, 4, 5];
        let r = vote(&flat, &off, 3).unwrap();
        assert_eq!(r, Some(0));
    }

    #[test]
    fn vote_empty_returns_none() {
        let r = vote(&[], &[0], 3).unwrap();
        assert_eq!(r, None);
    }

    #[test]
    fn vote_dedupes_within_chunk() {
        // single chunk repeating id 0 three times → 0 wins with count 1, but id 1 has count 0.
        let flat = vec![0, 0, 0];
        let off = vec![0, 3];
        let r = vote(&flat, &off, 2).unwrap();
        assert_eq!(r, Some(0));
    }

    #[test]
    fn majority_requires_threshold() {
        // 3 chunks: [0], [0], [1] → 0 in 2/3 = 0.667 ≥ 0.5 → 0.
        let flat = vec![0, 0, 1];
        let off = vec![0, 1, 2, 3];
        let r = majority(&flat, &off, 2, 0.5).unwrap();
        assert_eq!(r, Some(0));
        // threshold 0.8 → nobody qualifies.
        let r = majority(&flat, &off, 2, 0.8).unwrap();
        assert_eq!(r, None);
    }

    #[test]
    fn union_ascending_order() {
        let flat = vec![2, 0, 1, 0];
        let off = vec![0, 2, 4];
        let r = union(&flat, &off, 3).unwrap();
        assert_eq!(r, vec![0, 1, 2]);
    }

    #[test]
    fn intersection_requires_every_chunk() {
        // 2 chunks: [0,1,2] and [0,2] → 0 and 2 in both, 1 only first.
        let flat = vec![0, 1, 2, 0, 2];
        let off = vec![0, 3, 5];
        let r = intersection(&flat, &off, 3).unwrap();
        assert_eq!(r, vec![0, 2]);
    }

    #[test]
    fn intersection_empty_returns_empty() {
        let r = intersection(&[], &[0], 3).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn weighted_single_uses_weights() {
        // 3 chunks all with [0,1]; weights [3.0, 1.0, 1.0]; total=5; threshold 0.5 → cutoff 2.5.
        // Both 0 and 1 score 5 → tie; lowest id wins → 0.
        let flat = vec![0, 1, 0, 1, 0, 1];
        let off = vec![0, 2, 4, 6];
        let weights = vec![3.0, 1.0, 1.0];
        let r = weighted_single(&flat, &off, &weights, 2, 0.5).unwrap();
        assert_eq!(r, Some(0));
    }

    #[test]
    fn weighted_single_threshold_filters() {
        // chunks [0], [1]; weights [3.0, 0.5]; total=3.5; threshold 0.5 → required 1.75.
        // id 0 score 3.0 (qualifies); id 1 score 0.5 (no). → winner id 0.
        let flat = vec![0, 1];
        let off = vec![0, 1, 2];
        let weights = vec![3.0, 0.5];
        let r = weighted_single(&flat, &off, &weights, 2, 0.5).unwrap();
        assert_eq!(r, Some(0));
        // Tightening to 0.9 → required 3.15 > 3.0 → nobody qualifies.
        let r = weighted_single(&flat, &off, &weights, 2, 0.9).unwrap();
        assert_eq!(r, None);
    }

    #[test]
    fn weighted_multi_returns_qualifiers() {
        // 3 chunks: [0,1], [0], [0]; weights [1,1,1]; total=3; threshold 0.5 → 1.5.
        // 0 score = 3 (qualifies), 1 score = 1 (no).
        let flat = vec![0, 1, 0, 0];
        let off = vec![0, 2, 3, 4];
        let weights = vec![1.0, 1.0, 1.0];
        let r = weighted_multi(&flat, &off, &weights, 2, 0.5).unwrap();
        assert_eq!(r, vec![0]);
    }

    #[test]
    fn max_score_single_top_label() {
        // 2 chunks. flat_ids=[0,1,0,2], flat_scores=[0.1,0.9,0.5,0.3]; pooled: 0→0.5, 1→0.9, 2→0.3.
        let flat = vec![0, 1, 0, 2];
        let scores = vec![0.1, 0.9, 0.5, 0.3];
        let r = max_score_single(&flat, &scores, 3, None).unwrap();
        assert_eq!(r, Some(1));
    }

    #[test]
    fn max_score_single_below_threshold() {
        let flat = vec![0, 1];
        let scores = vec![0.1, 0.2];
        let r = max_score_single(&flat, &scores, 2, Some(0.5)).unwrap();
        assert_eq!(r, None);
    }

    #[test]
    fn max_score_multi_uses_strict_gt() {
        // pooled: 0→0.5, 1→0.9, 2→0.3. threshold=0.3 → only 0 and 1 (strict >).
        let flat = vec![0, 1, 0, 2];
        let scores = vec![0.1, 0.9, 0.5, 0.3];
        let r = max_score_multi(&flat, &scores, 3, Some(0.3)).unwrap();
        assert_eq!(r, vec![0, 1]);
    }

    #[test]
    fn max_score_multi_default_cutoff_zero() {
        // None threshold → cutoff 0.0; everyone with positive score qualifies.
        let flat = vec![0, 1, 0, 2];
        let scores = vec![0.1, 0.9, 0.5, 0.3];
        let r = max_score_multi(&flat, &scores, 3, None).unwrap();
        assert_eq!(r, vec![0, 1, 2]);
    }

    #[test]
    fn rejects_offsets_mismatch() {
        let r = vote(&[0, 1, 2], &[0, 2], 3);
        assert!(matches!(
            r,
            Err(AggregationError::LastOffsetMismatch { .. })
        ));
    }

    #[test]
    fn rejects_non_monotonic() {
        let r = vote(&[0, 1, 2], &[0, 2, 1, 3], 3);
        assert!(matches!(
            r,
            Err(AggregationError::OffsetsNotMonotonic { .. })
        ));
    }

    #[test]
    fn rejects_oor_id() {
        let r = vote(&[0, 5], &[0, 2], 3);
        assert!(matches!(r, Err(AggregationError::IdOutOfRange { .. })));
    }

    #[test]
    fn rejects_bad_threshold() {
        let r = majority(&[0], &[0, 1], 1, 0.0);
        assert!(matches!(r, Err(AggregationError::InvalidThreshold(_))));
    }

    #[test]
    fn weighted_zero_total_returns_none() {
        let flat = vec![0, 1];
        let off = vec![0, 1, 2];
        let weights = vec![0.0, 0.0];
        let r = weighted_single(&flat, &off, &weights, 2, 0.5).unwrap();
        assert_eq!(r, None);
        let r = weighted_multi(&flat, &off, &weights, 2, 0.5).unwrap();
        assert!(r.is_empty());
    }
}
