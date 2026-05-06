//! Inverted index for term-to-document mapping with BM25 and TF-IDF scoring.

use ahash::AHashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

/// A posting: a document ID and the term frequency in that document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Posting {
    pub doc_id: u32,
    pub term_freq: u32,
}

/// A scored document result.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredDoc {
    pub doc_id: u32,
    pub score: f64,
}

/// For min-heap (we pop lowest scores to keep top-K highest).
impl Eq for ScoredDoc {}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap: lower scores at the top.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// BM25 parameter set.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bm25Params {
    /// Term frequency saturation. Higher = more weight for repeated terms.
    /// Typical range: [1.2, 2.0]. Default: 1.5.
    pub k1: f64,
    /// Length normalization. 0 = no normalization, 1 = full normalization.
    /// Typical: 0.75. Default: 0.75.
    pub b: f64,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

/// TF weighting scheme for TF-IDF variants.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TfWeight {
    /// Raw term frequency: tf
    Raw,
    /// Sublinear / log-normalized: 1 + ln(tf), or 0 if tf == 0
    Sublinear,
    /// Boolean: 1 if tf > 0, else 0
    Boolean,
}

/// IDF weighting scheme for TF-IDF variants.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum IdfWeight {
    /// Standard: ln(N / df)
    Standard,
    /// Smoothed: ln((N + 1) / (df + 1)) + 1
    Smooth,
    /// Probabilistic (BM25-style): ln((N - df + 0.5) / (df + 0.5) + 1)
    Probabilistic,
}

/// Inverted index mapping terms to sorted posting lists.
///
/// Supports boolean queries, TF-IDF variants, BM25, and ranked retrieval.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InvertedIndex {
    index: AHashMap<String, Vec<Posting>>,
    doc_count: u32,
    /// Number of tokens per document (for length normalization).
    doc_lengths: AHashMap<u32, u32>,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self::default()
    }

    // =========================================================================
    // Indexing
    // =========================================================================

    /// Add a term occurrence for a document.
    pub fn add(&mut self, term: &str, doc_id: u32) {
        let postings = self.index.entry(term.to_string()).or_default();

        match postings.binary_search_by_key(&doc_id, |p| p.doc_id) {
            Ok(idx) => postings[idx].term_freq += 1,
            Err(idx) => postings.insert(
                idx,
                Posting {
                    doc_id,
                    term_freq: 1,
                },
            ),
        }

        *self.doc_lengths.entry(doc_id).or_insert(0) += 1;
        if self.doc_lengths.len() as u32 > self.doc_count {
            self.doc_count = self.doc_lengths.len() as u32;
        }
    }

    /// Index a full document (list of terms).
    pub fn add_document(&mut self, doc_id: u32, terms: &[&str]) {
        for term in terms {
            self.add(term, doc_id);
        }
    }

    /// Build an index from a batch of documents in parallel using rayon.
    ///
    /// Each document is a `(doc_id, terms)` pair. Documents are partitioned across
    /// rayon threads, each thread builds a local index, and results are merged.
    ///
    /// Produces identical results to sequential `add_document()` calls.
    pub fn build_batch(documents: &[(u32, Vec<String>)]) -> Self {
        if documents.is_empty() {
            return Self::new();
        }

        // Parallel: each rayon task builds a thread-local InvertedIndex
        let merged = documents
            .par_iter()
            .fold(Self::new, |mut local_idx, (doc_id, terms)| {
                let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
                local_idx.add_document(*doc_id, &refs);
                local_idx
            })
            .reduce(Self::new, |mut a, b| {
                a.merge(b);
                a
            });

        merged
    }

    /// Merge another index into this one.
    ///
    /// Posting lists are merged and re-sorted by doc_id. Term frequencies for
    /// the same (term, doc_id) pair are summed (though this shouldn't happen
    /// when documents are unique across the merge).
    pub fn merge(&mut self, other: InvertedIndex) {
        for (term, other_postings) in other.index {
            let postings = self.index.entry(term).or_default();
            for op in other_postings {
                match postings.binary_search_by_key(&op.doc_id, |p| p.doc_id) {
                    Ok(idx) => postings[idx].term_freq += op.term_freq,
                    Err(idx) => postings.insert(idx, op),
                }
            }
        }
        for (doc_id, length) in other.doc_lengths {
            *self.doc_lengths.entry(doc_id).or_insert(0) += length;
        }
        self.doc_count = self.doc_lengths.len() as u32;
    }

    // =========================================================================
    // Lookups
    // =========================================================================

    /// Look up the posting list for a term.
    pub fn get(&self, term: &str) -> Option<&[Posting]> {
        self.index.get(term).map(|v| v.as_slice())
    }

    /// Document frequency: number of documents containing the term.
    pub fn doc_freq(&self, term: &str) -> u32 {
        self.index.get(term).map_or(0, |p| p.len() as u32)
    }

    /// Number of unique terms.
    pub fn term_count(&self) -> usize {
        self.index.len()
    }

    /// Number of indexed documents.
    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    /// Get the length (total token count) of a document.
    pub fn doc_length(&self, doc_id: u32) -> u32 {
        self.doc_lengths.get(&doc_id).copied().unwrap_or(0)
    }

    /// Average document length across all indexed documents.
    pub fn avg_doc_length(&self) -> f64 {
        if self.doc_count == 0 {
            return 0.0;
        }
        let total: u64 = self.doc_lengths.values().map(|&v| v as u64).sum();
        total as f64 / self.doc_count as f64
    }

    // =========================================================================
    // TF-IDF scoring
    // =========================================================================

    /// Inverse document frequency: ln(N / df). Returns 0 if term not in index.
    pub fn idf(&self, term: &str) -> f64 {
        self.idf_weighted(term, IdfWeight::Standard)
    }

    /// IDF with configurable weighting scheme.
    pub fn idf_weighted(&self, term: &str, weight: IdfWeight) -> f64 {
        let df = self.doc_freq(term) as f64;
        let n = self.doc_count as f64;
        if df == 0.0 {
            return 0.0;
        }
        match weight {
            IdfWeight::Standard => (n / df).ln(),
            IdfWeight::Smooth => ((n + 1.0) / (df + 1.0)).ln() + 1.0,
            IdfWeight::Probabilistic => ((n - df + 0.5) / (df + 0.5) + 1.0).ln(),
        }
    }

    /// TF-IDF score for a term in a document (raw TF, standard IDF).
    pub fn tf_idf(&self, term: &str, doc_id: u32) -> f64 {
        self.tf_idf_weighted(term, doc_id, TfWeight::Raw, IdfWeight::Standard)
    }

    /// TF-IDF score with configurable TF and IDF weighting.
    pub fn tf_idf_weighted(
        &self,
        term: &str,
        doc_id: u32,
        tf_weight: TfWeight,
        idf_weight: IdfWeight,
    ) -> f64 {
        let raw_tf = self
            .get(term)
            .and_then(|postings| {
                postings
                    .binary_search_by_key(&doc_id, |p| p.doc_id)
                    .ok()
                    .map(|idx| postings[idx].term_freq as f64)
            })
            .unwrap_or(0.0);

        let tf = match tf_weight {
            TfWeight::Raw => raw_tf,
            TfWeight::Sublinear => {
                if raw_tf > 0.0 {
                    1.0 + raw_tf.ln()
                } else {
                    0.0
                }
            }
            TfWeight::Boolean => {
                if raw_tf > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        };

        tf * self.idf_weighted(term, idf_weight)
    }

    /// Multi-term TF-IDF score: sum of per-term TF-IDF scores for a document.
    pub fn score_tf_idf(
        &self,
        terms: &[&str],
        doc_id: u32,
        tf_weight: TfWeight,
        idf_weight: IdfWeight,
    ) -> f64 {
        terms
            .iter()
            .map(|t| self.tf_idf_weighted(t, doc_id, tf_weight, idf_weight))
            .sum()
    }

    // =========================================================================
    // BM25 scoring
    // =========================================================================

    /// BM25 score for a single term in a document.
    ///
    /// BM25(t, d) = IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avgdl))
    pub fn bm25_term(&self, term: &str, doc_id: u32, params: &Bm25Params, avgdl: f64) -> f64 {
        let tf = self
            .get(term)
            .and_then(|postings| {
                postings
                    .binary_search_by_key(&doc_id, |p| p.doc_id)
                    .ok()
                    .map(|idx| postings[idx].term_freq as f64)
            })
            .unwrap_or(0.0);

        if tf == 0.0 {
            return 0.0;
        }

        let idf = self.idf_weighted(term, IdfWeight::Probabilistic);
        let dl = self.doc_length(doc_id) as f64;
        let k1 = params.k1;
        let b = params.b;

        let numerator = tf * (k1 + 1.0);
        let denominator = tf + k1 * (1.0 - b + b * dl / avgdl);

        idf * (numerator / denominator)
    }

    /// BM25 score for a multi-term query against a document.
    pub fn score_bm25(&self, terms: &[&str], doc_id: u32, params: &Bm25Params) -> f64 {
        let avgdl = self.avg_doc_length();
        if avgdl == 0.0 {
            return 0.0;
        }
        terms
            .iter()
            .map(|t| self.bm25_term(t, doc_id, params, avgdl))
            .sum()
    }

    // =========================================================================
    // Ranked retrieval
    // =========================================================================

    /// Retrieve top-K documents ranked by BM25 score for a multi-term query.
    ///
    /// Uses a min-heap for O(n log k) efficiency instead of sorting all results.
    pub fn query_bm25(&self, terms: &[&str], params: &Bm25Params, top_k: usize) -> Vec<ScoredDoc> {
        if terms.is_empty() || top_k == 0 {
            return vec![];
        }

        let avgdl = self.avg_doc_length();
        if avgdl == 0.0 {
            return vec![];
        }

        // Collect candidate doc_ids from the union of all query term posting lists.
        let mut candidates = ahash::AHashSet::new();
        for term in terms {
            if let Some(postings) = self.get(term) {
                for p in postings {
                    candidates.insert(p.doc_id);
                }
            }
        }

        // Score each candidate and keep top-K via min-heap.
        let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::new();

        for doc_id in candidates {
            let score: f64 = terms
                .iter()
                .map(|t| self.bm25_term(t, doc_id, params, avgdl))
                .sum();

            if score <= 0.0 {
                continue;
            }

            if heap.len() < top_k {
                heap.push(ScoredDoc { doc_id, score });
            } else if let Some(min) = heap.peek() {
                if score > min.score {
                    heap.pop();
                    heap.push(ScoredDoc { doc_id, score });
                }
            }
        }

        // Drain heap into sorted-by-score-descending vec.
        let mut results: Vec<ScoredDoc> = heap.into_vec();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Retrieve top-K documents ranked by TF-IDF score for a multi-term query.
    pub fn query_tf_idf(
        &self,
        terms: &[&str],
        tf_weight: TfWeight,
        idf_weight: IdfWeight,
        top_k: usize,
    ) -> Vec<ScoredDoc> {
        if terms.is_empty() || top_k == 0 {
            return vec![];
        }

        // Collect candidate doc_ids.
        let mut candidates = ahash::AHashSet::new();
        for term in terms {
            if let Some(postings) = self.get(term) {
                for p in postings {
                    candidates.insert(p.doc_id);
                }
            }
        }

        let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::new();

        for doc_id in candidates {
            let score = self.score_tf_idf(terms, doc_id, tf_weight, idf_weight);
            if score <= 0.0 {
                continue;
            }

            if heap.len() < top_k {
                heap.push(ScoredDoc { doc_id, score });
            } else if let Some(min) = heap.peek() {
                if score > min.score {
                    heap.pop();
                    heap.push(ScoredDoc { doc_id, score });
                }
            }
        }

        let mut results: Vec<ScoredDoc> = heap.into_vec();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    // =========================================================================
    // Boolean queries
    // =========================================================================

    /// Boolean AND query: documents containing ALL terms.
    pub fn query_and(&self, terms: &[&str]) -> Vec<u32> {
        if terms.is_empty() {
            return vec![];
        }

        // Start with the shortest posting list for efficiency.
        let mut sorted_terms: Vec<&&str> = terms.iter().collect();
        sorted_terms.sort_by_key(|t| self.doc_freq(t));

        let first = sorted_terms[0];
        let mut result: Vec<u32> = self
            .get(first)
            .map(|p| p.iter().map(|posting| posting.doc_id).collect())
            .unwrap_or_default();

        for term in &sorted_terms[1..] {
            if result.is_empty() {
                break;
            }
            if let Some(postings) = self.get(term) {
                result.retain(|doc_id| postings.binary_search_by_key(doc_id, |p| p.doc_id).is_ok());
            } else {
                result.clear();
            }
        }
        result
    }

    /// Boolean OR query: documents containing ANY term.
    pub fn query_or(&self, terms: &[&str]) -> Vec<u32> {
        let mut doc_ids = ahash::AHashSet::new();
        for term in terms {
            if let Some(postings) = self.get(term) {
                for p in postings {
                    doc_ids.insert(p.doc_id);
                }
            }
        }
        let mut result: Vec<u32> = doc_ids.into_iter().collect();
        result.sort();
        result
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_index() -> InvertedIndex {
        let mut idx = InvertedIndex::new();
        idx.add_document(0, &["the", "cat", "sat", "on", "the", "mat"]);
        idx.add_document(1, &["the", "dog", "sat", "on", "the", "log"]);
        idx.add_document(2, &["a", "cat", "and", "a", "dog"]);
        idx
    }

    #[test]
    fn test_doc_freq() {
        let idx = build_test_index();
        assert_eq!(idx.doc_freq("the"), 2);
        assert_eq!(idx.doc_freq("cat"), 2);
        assert_eq!(idx.doc_freq("log"), 1);
        assert_eq!(idx.doc_freq("xyz"), 0);
    }

    #[test]
    fn test_query_and() {
        let idx = build_test_index();
        let result = idx.query_and(&["cat", "sat"]);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_query_or() {
        let idx = build_test_index();
        let result = idx.query_or(&["cat", "dog"]);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_tf_idf() {
        let idx = build_test_index();
        let score = idx.tf_idf("the", 0);
        assert!(score > 0.0);
        let score_absent = idx.tf_idf("the", 2);
        assert_eq!(score_absent, 0.0);
    }

    #[test]
    fn test_term_freq() {
        let idx = build_test_index();
        let postings = idx.get("the").unwrap();
        let doc0 = postings.iter().find(|p| p.doc_id == 0).unwrap();
        assert_eq!(doc0.term_freq, 2);
    }

    // --- Doc lengths & avg ---

    #[test]
    fn test_doc_lengths() {
        let idx = build_test_index();
        assert_eq!(idx.doc_length(0), 6); // "the cat sat on the mat"
        assert_eq!(idx.doc_length(1), 6);
        assert_eq!(idx.doc_length(2), 5); // "a cat and a dog"
        assert_eq!(idx.doc_length(99), 0); // non-existent
    }

    #[test]
    fn test_avg_doc_length() {
        let idx = build_test_index();
        let avgdl = idx.avg_doc_length();
        // (6 + 6 + 5) / 3 = 5.666...
        assert!((avgdl - 17.0 / 3.0).abs() < 1e-10);
    }

    // --- TF-IDF variants ---

    #[test]
    fn test_idf_smooth() {
        let idx = build_test_index();
        let idf = idx.idf_weighted("cat", IdfWeight::Smooth);
        // ln((3+1)/(2+1)) + 1 = ln(4/3) + 1
        let expected = (4.0f64 / 3.0).ln() + 1.0;
        assert!((idf - expected).abs() < 1e-10);
    }

    #[test]
    fn test_idf_probabilistic() {
        let idx = build_test_index();
        let idf = idx.idf_weighted("cat", IdfWeight::Probabilistic);
        // ln((3 - 2 + 0.5)/(2 + 0.5) + 1) = ln(1.5/2.5 + 1) = ln(1.6)
        let expected = (1.5f64 / 2.5 + 1.0).ln();
        assert!((idf - expected).abs() < 1e-10);
    }

    #[test]
    fn test_tf_idf_sublinear() {
        let idx = build_test_index();
        // "the" appears twice in doc 0, so sublinear TF = 1 + ln(2)
        let score = idx.tf_idf_weighted("the", 0, TfWeight::Sublinear, IdfWeight::Standard);
        let expected_tf = 1.0 + 2.0f64.ln();
        let expected_idf = (3.0f64 / 2.0).ln();
        assert!((score - expected_tf * expected_idf).abs() < 1e-10);
    }

    #[test]
    fn test_tf_idf_boolean() {
        let idx = build_test_index();
        let score = idx.tf_idf_weighted("the", 0, TfWeight::Boolean, IdfWeight::Standard);
        let expected_idf = (3.0f64 / 2.0).ln();
        assert!((score - expected_idf).abs() < 1e-10);
    }

    // --- BM25 ---

    #[test]
    fn test_bm25_single_term() {
        let idx = build_test_index();
        let params = Bm25Params::default();
        let score = idx.score_bm25(&["cat"], 0, &params);
        assert!(score > 0.0);
        // cat is not in doc 1
        let score1 = idx.score_bm25(&["cat"], 1, &params);
        assert_eq!(score1, 0.0);
    }

    #[test]
    fn test_bm25_multi_term() {
        let idx = build_test_index();
        let params = Bm25Params::default();
        let score = idx.score_bm25(&["cat", "sat"], 0, &params);
        let score_cat = idx.score_bm25(&["cat"], 0, &params);
        let score_sat = idx.score_bm25(&["sat"], 0, &params);
        // Multi-term score should be sum of individual term scores
        assert!((score - (score_cat + score_sat)).abs() < 1e-10);
    }

    #[test]
    fn test_bm25_absent_term() {
        let idx = build_test_index();
        let params = Bm25Params::default();
        let score = idx.score_bm25(&["nonexistent"], 0, &params);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_bm25_length_normalization() {
        // Create index where the same term appears once in a short doc and once in a long doc.
        let mut idx = InvertedIndex::new();
        idx.add_document(0, &["cat"]); // length 1
        idx.add_document(
            1,
            &[
                "cat", "the", "dog", "sat", "on", "the", "mat", "and", "ate", "food",
            ],
        ); // length 10

        let params = Bm25Params::default();
        let score_short = idx.score_bm25(&["cat"], 0, &params);
        let score_long = idx.score_bm25(&["cat"], 1, &params);
        // BM25 should score the shorter doc higher (same TF, shorter length).
        assert!(score_short > score_long);
    }

    // --- Ranked retrieval ---

    #[test]
    fn test_query_bm25_ranked() {
        let idx = build_test_index();
        let params = Bm25Params::default();
        let results = idx.query_bm25(&["cat"], &params, 10);
        // "cat" appears in docs 0 and 2
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
        let doc_ids: Vec<u32> = results.iter().map(|r| r.doc_id).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&2));
    }

    #[test]
    fn test_query_bm25_top_k() {
        let idx = build_test_index();
        let params = Bm25Params::default();
        // Ask for top 1
        let results = idx.query_bm25(&["sat"], &params, 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_tf_idf_ranked() {
        let idx = build_test_index();
        let results = idx.query_tf_idf(&["cat", "dog"], TfWeight::Sublinear, IdfWeight::Smooth, 10);
        // All 3 docs have cat or dog
        assert_eq!(results.len(), 3);
        // Doc 2 has both cat AND dog, should be ranked first
        assert_eq!(results[0].doc_id, 2);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_query_bm25_empty() {
        let idx = build_test_index();
        let params = Bm25Params::default();
        assert!(idx.query_bm25(&[], &params, 10).is_empty());
        assert!(idx.query_bm25(&["nonexistent"], &params, 10).is_empty());
    }

    // --- Larger index ---

    #[test]
    fn test_bm25_larger_index() {
        let mut idx = InvertedIndex::new();
        let docs = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox",
            "the lazy dog sleeps all day",
            "a fox and a dog are friends",
            "brown bears eat fish in the river",
        ];
        for (i, doc) in docs.iter().enumerate() {
            let terms: Vec<&str> = doc.split_whitespace().collect();
            idx.add_document(i as u32, &terms);
        }

        let params = Bm25Params::default();
        let results = idx.query_bm25(&["quick", "fox"], &params, 3);
        assert!(!results.is_empty());
        // Docs 0 and 1 have both "quick" and "fox", doc 3 has only "fox"
        let top_ids: Vec<u32> = results.iter().map(|r| r.doc_id).collect();
        assert!(top_ids.contains(&0));
        assert!(top_ids.contains(&1));
        // Doc 1 is shorter (4 terms vs 9), should rank higher for same query terms
        assert_eq!(results[0].doc_id, 1);
    }

    #[test]
    fn test_score_tf_idf_multi() {
        let idx = build_test_index();
        let score = idx.score_tf_idf(&["cat", "dog"], 2, TfWeight::Raw, IdfWeight::Standard);
        let s1 = idx.tf_idf("cat", 2);
        let s2 = idx.tf_idf("dog", 2);
        assert!((score - (s1 + s2)).abs() < 1e-10);
    }

    // --- Parallel build (rayon) ---

    #[test]
    fn test_build_batch_determinism() {
        // Use 100 docs to ensure rayon actually parallelizes (3 docs is too few
        // and causes flaky failures from rayon thread pool contention).
        let base_terms = [
            &["the", "cat", "sat", "on", "the", "mat"][..],
            &["the", "dog", "sat", "on", "the", "log"],
            &["a", "cat", "and", "a", "dog"],
            &["the", "quick", "brown", "fox"],
            &["lazy", "dog", "sleeps", "all", "day"],
        ];

        // Build sequentially
        let mut seq = InvertedIndex::new();
        for i in 0..100u32 {
            let terms = base_terms[(i as usize) % base_terms.len()];
            seq.add_document(i, terms);
        }

        // Build in parallel
        let documents: Vec<(u32, Vec<String>)> = (0..100u32)
            .map(|i| {
                let terms = base_terms[(i as usize) % base_terms.len()];
                (i, terms.iter().map(|s| s.to_string()).collect())
            })
            .collect();
        let par = InvertedIndex::build_batch(&documents);

        // Same doc counts and term counts
        assert_eq!(seq.doc_count(), par.doc_count());
        assert_eq!(seq.term_count(), par.term_count());

        // Same doc lengths
        for doc_id in 0..100 {
            assert_eq!(
                seq.doc_length(doc_id),
                par.doc_length(doc_id),
                "doc_length mismatch for doc {}",
                doc_id
            );
        }

        // Same doc frequencies
        for term in &[
            "the", "cat", "dog", "sat", "on", "mat", "log", "a", "and", "quick", "brown", "fox",
            "lazy", "sleeps", "all", "day",
        ] {
            assert_eq!(
                seq.doc_freq(term),
                par.doc_freq(term),
                "doc_freq mismatch for '{}'",
                term
            );
        }

        // Same avg doc length (critical for BM25 scoring)
        assert!(
            (seq.avg_doc_length() - par.avg_doc_length()).abs() < 1e-10,
            "avg_doc_length mismatch: seq={} par={}",
            seq.avg_doc_length(),
            par.avg_doc_length()
        );

        // Same BM25 results
        let params = Bm25Params::default();
        let seq_results = seq.query_bm25(&["cat", "dog"], &params, 10);
        let par_results = par.query_bm25(&["cat", "dog"], &params, 10);
        assert_eq!(
            seq_results.len(),
            par_results.len(),
            "result count mismatch: seq={} par={}",
            seq_results.len(),
            par_results.len()
        );
        // Scores should match (doc_ids may differ due to tied scores and
        // non-deterministic AHashSet iteration order in candidate collection)
        let mut seq_scores: Vec<i64> = seq_results
            .iter()
            .map(|r| (r.score * 1e10) as i64)
            .collect();
        let mut par_scores: Vec<i64> = par_results
            .iter()
            .map(|r| (r.score * 1e10) as i64)
            .collect();
        seq_scores.sort();
        par_scores.sort();
        assert_eq!(seq_scores, par_scores, "BM25 score distributions differ");
    }

    #[test]
    fn test_build_batch_empty() {
        let idx = InvertedIndex::build_batch(&[]);
        assert_eq!(idx.doc_count(), 0);
        assert_eq!(idx.term_count(), 0);
    }

    #[test]
    fn test_build_batch_single_doc() {
        let documents = vec![(
            42u32,
            vec!["hello", "world"]
                .into_iter()
                .map(String::from)
                .collect(),
        )];
        let idx = InvertedIndex::build_batch(&documents);
        assert_eq!(idx.doc_count(), 1);
        assert_eq!(idx.doc_freq("hello"), 1);
        assert_eq!(idx.doc_length(42), 2);
    }

    #[test]
    fn test_merge() {
        let mut a = InvertedIndex::new();
        a.add_document(0, &["cat", "dog"]);

        let mut b = InvertedIndex::new();
        b.add_document(1, &["cat", "fish"]);

        a.merge(b);
        assert_eq!(a.doc_count(), 2);
        assert_eq!(a.doc_freq("cat"), 2);
        assert_eq!(a.doc_freq("dog"), 1);
        assert_eq!(a.doc_freq("fish"), 1);
    }
}
