//! Interval-index structure for content-agnostic labeled spans.
//!
//! Stores `(label: u32, start: u32, end: u32, score: f32)` tuples (the
//! `LabeledSpan` value type, 16 bytes, `Copy`, `repr(C)`) and supports four
//! query algorithms over them:
//!
//! - `containing(offset)` — span ids whose `[start, end)` covers `offset`
//! - `overlapping(start, end)` — span ids whose `[start, end)` overlaps the
//!   query range
//! - `merge_adjacent(label, gap)` — sweep-and-tombstone merge of same-label
//!   spans separated by ≤ `gap` characters
//! - `subtract(label_a, label_b)` — append-not-replace difference: emits
//!   sub-spans of label_a that fall outside any span of label_b
//! - `resolve_overlaps_by_score(label_set)` — keep highest-scored span at
//!   each contested offset, truncate / split / drop the lower-scored ones
//!
//! Internal representation: spans live in a `Vec<Slot>` with stable ids
//! (slot index never changes after assignment). A companion sorted-by-start
//! `by_start: Vec<u32>` permutation plus `end_max: Vec<u32>` enables
//! `O(log n + k)` queries via binary search + bounded backward walk.
//! Mutations invalidate `by_start`/`end_max`; the next query rebuilds them
//! lazily. Tombstoned spans (cleared by `merge_adjacent` or
//! `resolve_overlaps_by_score`) stay in the slot vector with a sentinel
//! tombstone marker until `compact()` is called.
//!
//! **Trust model:** `label: u32` is opaque to this module. Callers own the
//! string ↔ id mapping (same convention as `InvertedIndex` term IDs). The
//! kaos-content `AnnotationIndex` wrapper layer does the string interning
//! and AST-node mapping; this module is content-agnostic.
//!
//! Design reference: `docs/SECTION_HEADING_PRIMITIVES_RESEARCH.md`
//! (`## SpanIndex (P4) — design reference` + the corrigendum that pins
//! the kaos-content boundary).

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use thiserror::Error;

// ─── Public types ─────────────────────────────────────────────────────────

/// One labeled half-open span `[start, end)` over an opaque coordinate
/// space (typically character offsets in a document). 16 bytes; `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct LabeledSpan {
    pub label: u32,
    pub start: u32,
    pub end: u32,
    pub score: f32,
}

impl LabeledSpan {
    /// Convenience constructor with score = 1.0.
    pub fn new(label: u32, start: u32, end: u32) -> Self {
        Self {
            label,
            start,
            end,
            score: 1.0,
        }
    }

    /// True iff `start <= offset < end`.
    #[inline]
    pub fn contains(&self, offset: u32) -> bool {
        self.start <= offset && offset < self.end
    }

    /// True iff `[start, end)` overlaps `[query_start, query_end)`.
    #[inline]
    pub fn overlaps(&self, query_start: u32, query_end: u32) -> bool {
        self.start < query_end && query_start < self.end
    }
}

/// Errors raised by SpanIndex on invalid input.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SpanIndexError {
    #[error("invalid span: start ({start}) > end ({end})")]
    InvalidSpan { start: u32, end: u32 },
    #[error("span id {id} out of range or tombstoned")]
    InvalidSpanId { id: u32 },
}

/// One stored slot. `Some` when live, `None` after tombstoning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Slot {
    span: LabeledSpan,
    alive: bool,
    /// Insertion-order tie-breaker for deterministic conflict resolution.
    insertion_id: u32,
}

/// Span-index data structure.
///
/// Construct with `SpanIndex::new()` + `add()`, or with `SpanIndex::bulk_build()`
/// for an already-collected `Vec<LabeledSpan>`. Span ids are slot indices and
/// remain stable across mutations (so external maps from id → side-table
/// data are safe).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpanIndex {
    slots: Vec<Slot>,
    /// Indices into `slots`, sorted by (`span.start`, then `slot index`).
    /// Stale after any mutation; rebuilt by `ensure_sorted()`.
    by_start: Vec<u32>,
    /// `end_max[i]` = max `span.end` over `by_start[..=i]` (live slots only).
    end_max: Vec<u32>,
    /// True when `by_start` / `end_max` reflect current `slots` state.
    sorted: bool,
}

impl SpanIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a known set of spans. O(n log n).
    pub fn bulk_build(spans: Vec<LabeledSpan>) -> Result<Self, SpanIndexError> {
        let mut idx = Self::new();
        for s in spans {
            idx.add(s)?;
        }
        idx.ensure_sorted();
        Ok(idx)
    }

    /// Append a span. Returns its stable id. O(1) amortised; lazy resort on
    /// next query.
    pub fn add(&mut self, span: LabeledSpan) -> Result<u32, SpanIndexError> {
        if span.start > span.end {
            return Err(SpanIndexError::InvalidSpan {
                start: span.start,
                end: span.end,
            });
        }
        let id = self.slots.len() as u32;
        self.slots.push(Slot {
            span,
            alive: true,
            insertion_id: id,
        });
        self.sorted = false;
        Ok(id)
    }

    /// Number of live spans (excludes tombstoned slots).
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.alive).count()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| !s.alive)
    }

    /// Borrow a live span by id, or `None` if id is out of range or tombstoned.
    pub fn get(&self, id: u32) -> Option<&LabeledSpan> {
        let slot = self.slots.get(id as usize)?;
        if slot.alive {
            Some(&slot.span)
        } else {
            None
        }
    }

    /// Force the lazy sort. Useful for benchmarks; queries call this
    /// automatically.
    pub fn freeze(&mut self) {
        self.ensure_sorted();
    }

    /// Drop tombstoned slots and renumber. Invalidates external id refs —
    /// callers that depend on stable ids must NOT call this. Provided for
    /// callers building a fresh index from a heavily-mutated one.
    pub fn compact(&mut self) -> Vec<u32> {
        // Returns a remap table: old_id -> new_id for surviving slots; tomb
        // entries map to u32::MAX.
        let mut remap = vec![u32::MAX; self.slots.len()];
        let mut new_slots: Vec<Slot> = Vec::with_capacity(self.slots.len());
        for (old_id, slot) in self.slots.iter().enumerate() {
            if slot.alive {
                let new_id = new_slots.len() as u32;
                remap[old_id] = new_id;
                let mut new_slot = *slot;
                new_slot.insertion_id = new_id;
                new_slots.push(new_slot);
            }
        }
        self.slots = new_slots;
        self.sorted = false;
        remap
    }

    /// Build/refresh `by_start` and `end_max` over live slots.
    fn ensure_sorted(&mut self) {
        if self.sorted {
            return;
        }
        let mut live: Vec<u32> = (0..self.slots.len() as u32)
            .filter(|&i| self.slots[i as usize].alive)
            .collect();
        // Sort by (start, slot index). Stable secondary key keeps property
        // tests deterministic vs the naive reference.
        live.sort_by(|&a, &b| {
            let sa = self.slots[a as usize].span.start;
            let sb = self.slots[b as usize].span.start;
            sa.cmp(&sb).then_with(|| a.cmp(&b))
        });
        let mut end_max = Vec::with_capacity(live.len());
        let mut max_so_far = 0u32;
        for &i in &live {
            let e = self.slots[i as usize].span.end;
            if e > max_so_far {
                max_so_far = e;
            }
            end_max.push(max_so_far);
        }
        self.by_start = live;
        self.end_max = end_max;
        self.sorted = true;
    }

    /// All live span ids whose `[start, end)` contains `offset`.
    /// `O(log n + k)` where `k` is the result size.
    pub fn containing(&mut self, offset: u32) -> SmallVec<[u32; 8]> {
        self.ensure_sorted();
        let mut out: SmallVec<[u32; 8]> = SmallVec::new();
        if self.by_start.is_empty() {
            return out;
        }
        // partition_point returns the first index whose start > offset; all
        // candidates with start <= offset are at indices < `j`.
        let j = self
            .by_start
            .partition_point(|&i| self.slots[i as usize].span.start <= offset);
        // Walk backward from j-1 while end_max[i] > offset (the bounded walk
        // gives O(log n + k) on real workloads).
        if j == 0 {
            return out;
        }
        let mut i = j;
        while i > 0 {
            i -= 1;
            if self.end_max[i] <= offset {
                break;
            }
            let id = self.by_start[i];
            let span = &self.slots[id as usize].span;
            if span.contains(offset) {
                out.push(id);
            }
        }
        out
    }

    /// All live span ids that overlap `[start, end)`.
    /// Half-open. `O(log n + k)`. Matches the mathematical definition
    /// `s.start < query_end && query_start < s.end` — so an empty query
    /// `[k, k)` returns spans whose `[start, end)` strictly brackets `k`,
    /// consistent with the naive reference.
    pub fn overlapping(&mut self, start: u32, end: u32) -> SmallVec<[u32; 8]> {
        self.ensure_sorted();
        let mut out: SmallVec<[u32; 8]> = SmallVec::new();
        if self.by_start.is_empty() {
            return out;
        }
        // First index whose start >= end — no further candidate can overlap.
        // Use partition_point with `start < end_q` so we include spans whose
        // start equals the query end only when end_q > start_q (mathematical
        // overlap allows touching at the boundary if the query is non-empty;
        // for empty queries, see the dedicated branch below).
        let end_pp = if start < end {
            end
        } else {
            start.saturating_add(1)
        };
        let j = self
            .by_start
            .partition_point(|&i| self.slots[i as usize].span.start < end_pp);
        if j == 0 {
            return out;
        }
        let mut i = j;
        while i > 0 {
            i -= 1;
            if self.end_max[i] <= start {
                break;
            }
            let id = self.by_start[i];
            let span = &self.slots[id as usize].span;
            if span.overlaps(start, end) {
                out.push(id);
            }
        }
        out
    }

    /// Merge spans of label `label` that are within `gap` of each other.
    /// Sweep-and-tombstone — span ids of merged-away spans are tombstoned;
    /// the surviving leader-id stays valid.
    pub fn merge_adjacent(&mut self, label: u32, gap: u32) {
        self.ensure_sorted();
        // Collect live ids of this label, sorted by start (already sorted in by_start).
        let mut group: Vec<u32> = self
            .by_start
            .iter()
            .copied()
            .filter(|&i| self.slots[i as usize].span.label == label)
            .collect();
        if group.len() < 2 {
            return;
        }
        // Sweep.
        let mut i = 0;
        while i < group.len() {
            let leader = group[i];
            let mut j = i + 1;
            while j < group.len() {
                let next = group[j];
                let leader_end = self.slots[leader as usize].span.end;
                let next_start = self.slots[next as usize].span.start;
                if next_start.saturating_sub(leader_end) > gap {
                    break;
                }
                let next_end = self.slots[next as usize].span.end;
                if next_end > leader_end {
                    self.slots[leader as usize].span.end = next_end;
                }
                self.slots[next as usize].alive = false;
                j += 1;
            }
            i = j;
        }
        self.sorted = false;
        group.clear();
    }

    /// Append-not-replace difference: for each live span of `label_a`,
    /// produce sub-spans of `label_a` (preserving score) that fall outside
    /// every live span of `label_b`. Original `label_a` spans remain;
    /// callers that want them removed should call `tombstone` themselves.
    pub fn subtract(&mut self, label_a: u32, label_b: u32) -> Vec<u32> {
        self.ensure_sorted();
        let a_ids: Vec<u32> = self
            .by_start
            .iter()
            .copied()
            .filter(|&i| self.slots[i as usize].span.label == label_a)
            .collect();
        let b_ids: Vec<u32> = self
            .by_start
            .iter()
            .copied()
            .filter(|&i| self.slots[i as usize].span.label == label_b)
            .collect();
        let mut new_ids = Vec::new();
        for a_id in a_ids {
            let a = self.slots[a_id as usize].span;
            // Collect b-spans that overlap this `a`, sorted by start.
            let mut holes: Vec<(u32, u32)> = b_ids
                .iter()
                .filter_map(|&b_id| {
                    let b = self.slots[b_id as usize].span;
                    if b.overlaps(a.start, a.end) {
                        Some((b.start.max(a.start), b.end.min(a.end)))
                    } else {
                        None
                    }
                })
                .collect();
            holes.sort_by_key(|&(s, _)| s);
            // Walk and emit gaps.
            let mut cursor = a.start;
            for (hs, he) in holes {
                if hs > cursor {
                    let span = LabeledSpan {
                        label: label_a,
                        start: cursor,
                        end: hs,
                        score: a.score,
                    };
                    if let Ok(id) = self.add(span) {
                        new_ids.push(id);
                    }
                }
                if he > cursor {
                    cursor = he;
                }
            }
            if cursor < a.end {
                let span = LabeledSpan {
                    label: label_a,
                    start: cursor,
                    end: a.end,
                    score: a.score,
                };
                if let Ok(id) = self.add(span) {
                    new_ids.push(id);
                }
            }
        }
        self.sorted = false;
        new_ids
    }

    /// Where two live spans whose `label` is in `label_set` overlap, keep
    /// the higher-scored span; for the lower-scored span, truncate or drop
    /// (or split into a head + tail) to remove the contested region.
    /// Tombstones the dropped spans; emits new sub-spans for splits.
    pub fn resolve_overlaps_by_score(&mut self, label_set: &[u32]) -> Vec<u32> {
        self.ensure_sorted();
        let allow: AHashMap<u32, ()> = label_set.iter().copied().map(|l| (l, ())).collect();
        let candidate_ids: Vec<u32> = self
            .by_start
            .iter()
            .copied()
            .filter(|&i| {
                let sp = &self.slots[i as usize].span;
                allow.contains_key(&sp.label)
            })
            .collect();
        let mut emitted = Vec::new();
        // Compare every pair (O(k^2) on candidates; fine for the heading-
        // scorer scale where k is in the hundreds).
        for &a_id in &candidate_ids {
            if !self.slots[a_id as usize].alive {
                continue;
            }
            for &b_id in &candidate_ids {
                if a_id == b_id {
                    continue;
                }
                if !self.slots[a_id as usize].alive || !self.slots[b_id as usize].alive {
                    continue;
                }
                let a = self.slots[a_id as usize].span;
                let b = self.slots[b_id as usize].span;
                if !a.overlaps(b.start, b.end) {
                    continue;
                }
                // Decide a winner: higher score wins; ties break on
                // (smaller start, then smaller insertion_id).
                let a_better =
                    compare_winner(&self.slots[a_id as usize], &self.slots[b_id as usize]);
                let (winner_id, loser_id) = if a_better { (a_id, b_id) } else { (b_id, a_id) };
                let winner = self.slots[winner_id as usize].span;
                let loser = self.slots[loser_id as usize].span;
                // Resolve loser against winner.
                let head_present = loser.start < winner.start;
                let tail_present = loser.end > winner.end;
                self.slots[loser_id as usize].alive = false;
                if head_present {
                    let span = LabeledSpan {
                        start: loser.start,
                        end: winner.start,
                        ..loser
                    };
                    if let Ok(id) = self.add(span) {
                        emitted.push(id);
                    }
                }
                if tail_present {
                    let span = LabeledSpan {
                        start: winner.end,
                        end: loser.end,
                        ..loser
                    };
                    if let Ok(id) = self.add(span) {
                        emitted.push(id);
                    }
                }
                // If neither head nor tail, loser is fully contained — drop.
            }
        }
        self.sorted = false;
        emitted
    }

    /// Tombstone a live span by id. Returns true if a live span was
    /// tombstoned, false if id is out of range or already dead.
    pub fn tombstone(&mut self, id: u32) -> bool {
        match self.slots.get_mut(id as usize) {
            Some(slot) if slot.alive => {
                slot.alive = false;
                self.sorted = false;
                true
            }
            _ => false,
        }
    }
}

fn compare_winner(a: &Slot, b: &Slot) -> bool {
    // True if `a` beats `b`. Higher score wins; ties → smaller start →
    // smaller insertion_id.
    if a.span.score > b.span.score {
        return true;
    }
    if a.span.score < b.span.score {
        return false;
    }
    if a.span.start < b.span.start {
        return true;
    }
    if a.span.start > b.span.start {
        return false;
    }
    a.insertion_id < b.insertion_id
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn span(label: u32, start: u32, end: u32) -> LabeledSpan {
        LabeledSpan::new(label, start, end)
    }

    #[test]
    fn empty_queries_return_empty() {
        let mut idx = SpanIndex::new();
        assert!(idx.containing(5).is_empty());
        assert!(idx.overlapping(0, 100).is_empty());
    }

    #[test]
    fn containing_finds_simple_span() {
        let mut idx = SpanIndex::new();
        let id = idx.add(span(1, 10, 20)).unwrap();
        assert_eq!(idx.containing(15).as_slice(), &[id]);
        assert!(idx.containing(5).is_empty());
        assert!(idx.containing(20).is_empty()); // half-open
    }

    #[test]
    fn overlapping_finds_overlap() {
        let mut idx = SpanIndex::new();
        let a = idx.add(span(1, 10, 20)).unwrap();
        let b = idx.add(span(2, 15, 25)).unwrap();
        let c = idx.add(span(3, 100, 200)).unwrap();
        let mut hits = idx.overlapping(12, 18);
        hits.sort();
        assert_eq!(hits.as_slice(), &[a, b]);
        assert!(idx.overlapping(50, 60).is_empty());
        let mut hits2 = idx.overlapping(150, 250);
        hits2.sort();
        assert_eq!(hits2.as_slice(), &[c]);
    }

    #[test]
    fn invalid_span_rejected() {
        let mut idx = SpanIndex::new();
        let err = idx.add(LabeledSpan::new(1, 50, 10)).unwrap_err();
        assert!(matches!(err, SpanIndexError::InvalidSpan { .. }));
    }

    #[test]
    fn merge_adjacent_basic() {
        let mut idx = SpanIndex::new();
        let a = idx.add(span(1, 0, 10)).unwrap();
        let _b = idx.add(span(1, 12, 20)).unwrap();
        let _c = idx.add(span(1, 100, 110)).unwrap();
        idx.merge_adjacent(1, 5);
        // 0..10 and 12..20 should merge (gap 2 ≤ 5); 100..110 remains separate.
        let leader = idx.get(a).unwrap();
        assert_eq!(leader.start, 0);
        assert_eq!(leader.end, 20);
    }

    #[test]
    fn subtract_basic() {
        let mut idx = SpanIndex::new();
        let _a = idx.add(span(1, 0, 100)).unwrap(); // body
        let _b = idx.add(span(2, 30, 50)).unwrap(); // table
        let _c = idx.add(span(2, 70, 80)).unwrap(); // table
        let new_ids = idx.subtract(1, 2);
        // Should produce 3 sub-spans of label 1: [0..30), [50..70), [80..100).
        assert_eq!(new_ids.len(), 3);
        let extents: Vec<(u32, u32)> = new_ids
            .iter()
            .map(|&id| {
                let s = idx.get(id).unwrap();
                (s.start, s.end)
            })
            .collect();
        let mut extents = extents;
        extents.sort();
        assert_eq!(extents, vec![(0, 30), (50, 70), (80, 100)]);
    }

    #[test]
    fn resolve_overlaps_higher_score_wins() {
        let mut idx = SpanIndex::new();
        let _high = idx
            .add(LabeledSpan {
                label: 10,
                start: 50,
                end: 70,
                score: 0.9,
            })
            .unwrap();
        let _low = idx
            .add(LabeledSpan {
                label: 20,
                start: 40,
                end: 80,
                score: 0.4,
            })
            .unwrap();
        idx.resolve_overlaps_by_score(&[10, 20]);
        // The lower-scored span should split into [40..50) + [70..80).
        let mut hits = idx.overlapping(0, 200);
        hits.sort();
        let labels: Vec<u32> = hits.iter().map(|&i| idx.get(i).unwrap().label).collect();
        // Expect three live spans: the high winner + two pieces of the low.
        let mut sorted_labels = labels.clone();
        sorted_labels.sort();
        assert_eq!(sorted_labels, vec![10, 20, 20]);
    }

    #[test]
    fn tombstone_removes_from_queries() {
        let mut idx = SpanIndex::new();
        let id = idx.add(span(1, 10, 20)).unwrap();
        assert_eq!(idx.containing(15).as_slice(), &[id]);
        assert!(idx.tombstone(id));
        assert!(idx.containing(15).is_empty());
    }

    #[test]
    fn bulk_build_matches_sequential() {
        let spans = vec![span(1, 0, 10), span(1, 5, 15), span(2, 20, 30)];
        let mut bulk = SpanIndex::bulk_build(spans.clone()).unwrap();
        let mut seq = SpanIndex::new();
        for s in &spans {
            seq.add(*s).unwrap();
        }
        assert_eq!(bulk.containing(7).len(), seq.containing(7).len());
    }

    /// Naive reference implementation for `containing`. Used by the
    /// reference-equivalence proptest below.
    fn naive_containing(spans: &[LabeledSpan], offset: u32) -> Vec<u32> {
        spans
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if s.contains(offset) {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .collect()
    }

    fn naive_overlapping(spans: &[LabeledSpan], start: u32, end: u32) -> Vec<u32> {
        spans
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if s.overlaps(start, end) {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .collect()
    }

    fn span_strategy() -> impl Strategy<Value = LabeledSpan> {
        (0u32..=10, 0u32..200, 0u32..50, -1.0f32..2.0).prop_map(|(label, start, len, score)| {
            LabeledSpan {
                label,
                start,
                end: start.saturating_add(len),
                score,
            }
        })
    }

    proptest! {
        #[test]
        fn containing_matches_naive(
            spans in prop::collection::vec(span_strategy(), 0..32),
            offset in 0u32..250,
        ) {
            let mut idx = SpanIndex::bulk_build(spans.clone()).unwrap();
            let mut got: Vec<u32> = idx.containing(offset).into_iter().collect();
            got.sort();
            let mut want = naive_containing(&spans, offset);
            want.sort();
            prop_assert_eq!(got, want);
        }

        #[test]
        fn overlapping_matches_naive(
            spans in prop::collection::vec(span_strategy(), 0..32),
            qs in 0u32..200,
            qlen in 0u32..100,
        ) {
            let qe = qs.saturating_add(qlen);
            let mut idx = SpanIndex::bulk_build(spans.clone()).unwrap();
            let mut got: Vec<u32> = idx.overlapping(qs, qe).into_iter().collect();
            got.sort();
            let mut want = naive_overlapping(&spans, qs, qe);
            want.sort();
            prop_assert_eq!(got, want);
        }

        #[test]
        fn add_then_query_does_not_panic(
            spans in prop::collection::vec(span_strategy(), 0..32),
            offset in 0u32..250,
        ) {
            let mut idx = SpanIndex::new();
            for s in spans { let _ = idx.add(s); }
            let _ = idx.containing(offset);
        }

        #[test]
        fn merge_idempotent_for_zero_gap(
            spans in prop::collection::vec(span_strategy(), 0..16),
        ) {
            let mut idx = SpanIndex::bulk_build(spans).unwrap();
            idx.merge_adjacent(1, 0);
            let len_after_first = idx.len();
            idx.merge_adjacent(1, 0);
            prop_assert_eq!(len_after_first, idx.len());
        }
    }
}
