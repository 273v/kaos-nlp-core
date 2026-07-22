//! Compressed Sparse Row (CSR) document-term matrix.
//!
//! Memory-efficient representation of document-term frequency data.
//! Supports TF-IDF weighting and cosine similarity between documents.
//!
//! Architecture follows kelvin_nlp_v1's CSR format:
//! - `values`: Flattened term frequencies
//! - `col_indices`: Term IDs (sorted per document for binary search)
//! - `row_ptrs`: Row pointers delimiting each document's data

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors raised by `SparseTermMatrix` for out-of-range document indices.
///
/// At the PyO3 boundary these are translated to `ValueError` rather than
/// surfacing as panics from unchecked vector indexing.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SparseTermMatrixError {
    #[error("document index {index} out of range (matrix has {num_docs} documents)")]
    DocIndexOutOfRange { index: usize, num_docs: usize },
}

/// Compressed Sparse Row (CSR) document-term matrix.
///
/// Each row represents a document, each column a term.
/// Values are term frequencies (raw counts).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SparseTermMatrix {
    /// Flattened term frequencies.
    values: Vec<u32>,
    /// Term IDs (sorted per row for O(log k) lookup).
    col_indices: Vec<u32>,
    /// Row pointers: row_ptrs[i] is the start of document i's data in values/col_indices.
    /// Length = num_docs + 1.
    row_ptrs: Vec<usize>,
    /// Number of documents (rows).
    num_docs: usize,
    /// Number of terms (columns).
    num_terms: usize,
}

impl SparseTermMatrix {
    /// Create an empty matrix.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptrs: vec![0],
            num_docs: 0,
            num_terms: 0,
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(doc_capacity: usize, estimated_nonzeros: usize) -> Self {
        Self {
            values: Vec::with_capacity(estimated_nonzeros),
            col_indices: Vec::with_capacity(estimated_nonzeros),
            row_ptrs: Vec::with_capacity(doc_capacity + 1),
            num_docs: 0,
            num_terms: 0,
        }
    }

    /// Add a document as a set of (term_id, frequency) pairs.
    ///
    /// Pairs will be sorted by term_id for efficient lookup.
    pub fn add_document(&mut self, term_freqs: &mut [(u32, u32)]) {
        // Sort by term_id for binary search
        term_freqs.sort_by_key(|&(tid, _)| tid);

        if self.row_ptrs.is_empty() {
            self.row_ptrs.push(0);
        }

        for &(tid, freq) in term_freqs.iter() {
            self.col_indices.push(tid);
            self.values.push(freq);
            if (tid as usize) >= self.num_terms {
                self.num_terms = tid as usize + 1;
            }
        }

        self.num_docs += 1;
        self.row_ptrs.push(self.values.len());
    }

    /// Add a document from a HashMap of term_id → frequency.
    pub fn add_document_from_map(&mut self, freqs: &AHashMap<u32, u32>) {
        let mut pairs: Vec<(u32, u32)> = freqs.iter().map(|(&k, &v)| (k, v)).collect();
        self.add_document(&mut pairs);
    }

    /// Get the term frequency for a document and term.
    ///
    /// O(log k) where k is the number of unique terms in the document.
    pub fn get_term_freq(&self, doc_id: usize, term_id: u32) -> u32 {
        if doc_id >= self.num_docs {
            return 0;
        }
        let start = self.row_ptrs[doc_id];
        let end = self.row_ptrs[doc_id + 1];
        let cols = &self.col_indices[start..end];

        match cols.binary_search(&term_id) {
            Ok(idx) => self.values[start + idx],
            Err(_) => 0,
        }
    }

    /// Iterate over non-zero terms for a document.
    ///
    /// Returns (term_id, frequency) pairs. Errors with `DocIndexOutOfRange`
    /// when `doc_id >= num_docs` — pre-0.1.0a1 builds panicked via unchecked
    /// indexing into `row_ptrs`.
    pub fn iter_document_terms(
        &self,
        doc_id: usize,
    ) -> Result<impl Iterator<Item = (u32, u32)> + '_, SparseTermMatrixError> {
        if doc_id >= self.num_docs {
            return Err(SparseTermMatrixError::DocIndexOutOfRange {
                index: doc_id,
                num_docs: self.num_docs,
            });
        }
        let start = self.row_ptrs[doc_id];
        let end = self.row_ptrs[doc_id + 1];
        Ok(self.col_indices[start..end]
            .iter()
            .zip(self.values[start..end].iter())
            .map(|(&tid, &freq)| (tid, freq)))
    }

    /// Number of unique terms in a document.
    pub fn document_term_count(&self, doc_id: usize) -> usize {
        if doc_id >= self.num_docs {
            return 0;
        }
        self.row_ptrs[doc_id + 1] - self.row_ptrs[doc_id]
    }

    /// Total tokens in a document (sum of frequencies).
    pub fn document_length(&self, doc_id: usize) -> u64 {
        if doc_id >= self.num_docs {
            return 0;
        }
        let start = self.row_ptrs[doc_id];
        let end = self.row_ptrs[doc_id + 1];
        self.values[start..end].iter().map(|&v| v as u64).sum()
    }

    /// Number of documents.
    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    /// Number of terms (columns).
    pub fn num_terms(&self) -> usize {
        self.num_terms
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Density: fraction of cells that are non-zero.
    pub fn density(&self) -> f64 {
        if self.num_docs == 0 || self.num_terms == 0 {
            return 0.0;
        }
        self.values.len() as f64 / (self.num_docs * self.num_terms) as f64
    }

    /// Estimated memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.values.len() * 4 + self.col_indices.len() * 4 + self.row_ptrs.len() * 8
    }

    /// Shrink internal buffers to fit.
    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
        self.col_indices.shrink_to_fit();
        self.row_ptrs.shrink_to_fit();
    }

    /// Compute TF-IDF weighted vector for a document.
    ///
    /// Uses sublinear TF (1 + ln(tf)) and smooth IDF (ln((N+1)/(df+1)) + 1).
    pub fn tfidf_vector(
        &self,
        doc_id: usize,
        doc_freqs: &[u32], // Term document frequencies, indexed by term_id
    ) -> Vec<(u32, f64)> {
        if doc_id >= self.num_docs {
            return vec![];
        }

        let n = self.num_docs as f64;
        let start = self.row_ptrs[doc_id];
        let end = self.row_ptrs[doc_id + 1];

        self.col_indices[start..end]
            .iter()
            .zip(self.values[start..end].iter())
            .map(|(&tid, &raw_tf)| {
                let tf = if raw_tf > 0 {
                    1.0 + (raw_tf as f64).ln()
                } else {
                    0.0
                };
                let df = doc_freqs.get(tid as usize).copied().unwrap_or(0) as f64;
                let idf = if df > 0.0 {
                    ((n + 1.0) / (df + 1.0)).ln() + 1.0
                } else {
                    0.0
                };
                (tid, tf * idf)
            })
            .collect()
    }

    /// Cosine similarity between two documents using raw TF vectors.
    pub fn cosine_similarity(&self, doc_a: usize, doc_b: usize) -> f64 {
        if doc_a >= self.num_docs || doc_b >= self.num_docs {
            return 0.0;
        }

        let start_a = self.row_ptrs[doc_a];
        let end_a = self.row_ptrs[doc_a + 1];
        let start_b = self.row_ptrs[doc_b];
        let end_b = self.row_ptrs[doc_b + 1];

        let cols_a = &self.col_indices[start_a..end_a];
        let vals_a = &self.values[start_a..end_a];
        let cols_b = &self.col_indices[start_b..end_b];
        let vals_b = &self.values[start_b..end_b];

        // Two-pointer merge for sorted term IDs
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        let mut ia = 0;
        let mut ib = 0;

        while ia < cols_a.len() && ib < cols_b.len() {
            let ta = cols_a[ia];
            let tb = cols_b[ib];
            let va = vals_a[ia] as f64;
            let vb = vals_b[ib] as f64;

            if ta == tb {
                dot += va * vb;
                norm_a += va * va;
                norm_b += vb * vb;
                ia += 1;
                ib += 1;
            } else if ta < tb {
                norm_a += va * va;
                ia += 1;
            } else {
                norm_b += vb * vb;
                ib += 1;
            }
        }

        // Remaining terms
        while ia < cols_a.len() {
            let va = vals_a[ia] as f64;
            norm_a += va * va;
            ia += 1;
        }
        while ib < cols_b.len() {
            let vb = vals_b[ib] as f64;
            norm_b += vb * vb;
            ib += 1;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            dot / denom
        }
    }

    /// Cosine similarity between two documents using TF-IDF weighted vectors.
    pub fn cosine_similarity_tfidf(&self, doc_a: usize, doc_b: usize, doc_freqs: &[u32]) -> f64 {
        let vec_a = self.tfidf_vector(doc_a, doc_freqs);
        let vec_b = self.tfidf_vector(doc_b, doc_freqs);

        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        let map_b: AHashMap<u32, f64> = vec_b.into_iter().collect();

        for &(tid, val_a) in &vec_a {
            norm_a += val_a * val_a;
            if let Some(&val_b) = map_b.get(&tid) {
                dot += val_a * val_b;
            }
        }
        for &val_b in map_b.values() {
            norm_b += val_b * val_b;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            dot / denom
        }
    }
}

// ─── Build from InvertedIndex ───────────────────────────────────────────────

/// Build a SparseTermMatrix from a vocabulary mapping and per-document term frequencies.
///
/// `vocab`: term → term_id mapping
/// `doc_term_freqs`: doc_id → {term → freq}
pub fn build_matrix(
    vocab: &AHashMap<String, u32>,
    doc_term_freqs: &[(u32, AHashMap<String, u32>)],
) -> SparseTermMatrix {
    let mut matrix =
        SparseTermMatrix::with_capacity(doc_term_freqs.len(), doc_term_freqs.len() * 50);

    for (_doc_id, freqs) in doc_term_freqs {
        let mut pairs: Vec<(u32, u32)> = freqs
            .iter()
            .filter_map(|(term, &freq)| vocab.get(term).map(|&tid| (tid, freq)))
            .collect();
        matrix.add_document(&mut pairs);
    }

    matrix
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_matrix() -> SparseTermMatrix {
        let mut m = SparseTermMatrix::new();
        // Doc 0: "the cat sat" → terms {0:2, 1:1, 2:1}
        m.add_document(&mut [(0, 2), (1, 1), (2, 1)]);
        // Doc 1: "the dog ran" → terms {0:1, 3:1, 4:1}
        m.add_document(&mut [(0, 1), (3, 1), (4, 1)]);
        // Doc 2: "a cat and a dog" → terms {1:1, 3:1, 5:2, 6:1}
        m.add_document(&mut [(1, 1), (3, 1), (5, 2), (6, 1)]);
        m
    }

    #[test]
    fn test_basic_properties() {
        let m = build_test_matrix();
        assert_eq!(m.num_docs(), 3);
        assert_eq!(m.num_terms(), 7); // terms 0-6
    }

    #[test]
    fn test_term_freq_lookup() {
        let m = build_test_matrix();
        assert_eq!(m.get_term_freq(0, 0), 2); // "the" appears 2x in doc 0
        assert_eq!(m.get_term_freq(0, 1), 1); // "cat" appears 1x in doc 0
        assert_eq!(m.get_term_freq(0, 3), 0); // "dog" not in doc 0
        assert_eq!(m.get_term_freq(1, 0), 1); // "the" 1x in doc 1
        assert_eq!(m.get_term_freq(99, 0), 0); // out of range
    }

    #[test]
    fn test_document_length() {
        let m = build_test_matrix();
        assert_eq!(m.document_length(0), 4); // 2+1+1
        assert_eq!(m.document_length(1), 3); // 1+1+1
        assert_eq!(m.document_length(2), 5); // 1+1+2+1
    }

    #[test]
    fn test_document_term_count() {
        let m = build_test_matrix();
        assert_eq!(m.document_term_count(0), 3);
        assert_eq!(m.document_term_count(1), 3);
        assert_eq!(m.document_term_count(2), 4);
    }

    #[test]
    fn test_iter_document_terms() {
        let m = build_test_matrix();
        let terms: Vec<(u32, u32)> = m.iter_document_terms(0).unwrap().collect();
        assert_eq!(terms, vec![(0, 2), (1, 1), (2, 1)]);
    }

    #[test]
    fn test_cosine_self_similarity() {
        let m = build_test_matrix();
        let sim = m.cosine_similarity(0, 0);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_disjoint() {
        let mut m = SparseTermMatrix::new();
        m.add_document(&mut [(0, 1), (1, 1)]);
        m.add_document(&mut [(2, 1), (3, 1)]);
        let sim = m.cosine_similarity(0, 1);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_partial_overlap() {
        let m = build_test_matrix();
        // Doc 0 shares term 0 ("the") with doc 1
        let sim = m.cosine_similarity(0, 1);
        assert!(sim > 0.0);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_tfidf_vector() {
        let m = build_test_matrix();
        let doc_freqs = vec![2, 2, 1, 2, 1, 1, 1]; // DF for each term
        let vec0 = m.tfidf_vector(0, &doc_freqs);
        assert_eq!(vec0.len(), 3); // 3 unique terms in doc 0
                                   // All TF-IDF values should be positive
        for &(_, v) in &vec0 {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_density() {
        let m = build_test_matrix();
        let d = m.density();
        // 10 non-zeros out of 3*7=21 cells
        assert!((d - 10.0 / 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_memory_usage() {
        let m = build_test_matrix();
        let mem = m.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_empty_matrix() {
        let m = SparseTermMatrix::new();
        assert_eq!(m.num_docs(), 0);
        assert_eq!(m.num_terms(), 0);
        assert_eq!(m.nnz(), 0);
        assert_eq!(m.density(), 0.0);
    }

    #[test]
    fn test_from_hashmap() {
        let mut m = SparseTermMatrix::new();
        let mut freqs = AHashMap::new();
        freqs.insert(0u32, 3u32);
        freqs.insert(5, 1);
        freqs.insert(2, 2);
        m.add_document_from_map(&freqs);
        assert_eq!(m.num_docs(), 1);
        assert_eq!(m.get_term_freq(0, 0), 3);
        assert_eq!(m.get_term_freq(0, 2), 2);
        assert_eq!(m.get_term_freq(0, 5), 1);
    }

    #[test]
    fn test_cosine_similarity_tfidf() {
        let m = build_test_matrix();
        // doc_freqs: term 0 in 2 docs, term 1 in 2 docs, term 2 in 1 doc,
        //            term 3 in 2 docs, term 4 in 1 doc, term 5 in 1 doc, term 6 in 1 doc
        let doc_freqs = vec![2, 2, 1, 2, 1, 1, 1];

        // Self-similarity with TF-IDF should be 1.0
        let self_sim = m.cosine_similarity_tfidf(0, 0, &doc_freqs);
        assert!(
            (self_sim - 1.0).abs() < 1e-10,
            "Self TF-IDF similarity should be 1.0, got {}",
            self_sim
        );

        // Doc 0 and Doc 1 share term 0 ("the"), should have partial similarity
        let sim_01 = m.cosine_similarity_tfidf(0, 1, &doc_freqs);
        assert!(
            sim_01 > 0.0,
            "Docs sharing a term should have positive TF-IDF similarity"
        );
        assert!(
            sim_01 < 1.0,
            "Non-identical docs should have similarity < 1.0"
        );
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut m = build_test_matrix();
        let num_docs_before = m.num_docs();
        let num_terms_before = m.num_terms();
        let nnz_before = m.nnz();

        m.shrink_to_fit();

        assert_eq!(m.num_docs(), num_docs_before);
        assert_eq!(m.num_terms(), num_terms_before);
        assert_eq!(m.nnz(), nnz_before);
        // Verify data is still intact
        assert_eq!(m.get_term_freq(0, 0), 2);
        assert_eq!(m.document_length(1), 3);
    }

    #[test]
    fn test_out_of_range_doc() {
        let m = build_test_matrix();
        // doc_id 99 is out of range — all should return 0
        assert_eq!(m.get_term_freq(99, 0), 0);
        assert_eq!(m.document_length(99), 0);
        assert_eq!(m.document_term_count(99), 0);
    }
}
