//! MinHash and LSH (Locality-Sensitive Hashing) for near-duplicate detection.
//!
//! Provides:
//! - `MinHash`: Compute MinHash signatures from token sets using `ahash`
//! - `MinHashIndex`: LSH index for approximate nearest-neighbor lookups
//! - Jaccard similarity estimation from MinHash signatures
//!
//! Architecture:
//! - Uses ahash for fast hashing (multiple hash functions via seed variation)
//! - Band/row decomposition for LSH
//! - Configurable shingle size for token-level or character-level hashing

use serde::{Deserialize, Serialize};
use thiserror::Error;

use ahash::{AHashMap, AHashSet};

/// Errors raised by MinHash operations on invalid input.
///
/// At the PyO3 boundary these are translated to `ValueError` rather than
/// surfacing as panics.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MinHashError {
    /// Two signatures or a signature and an index disagree on permutation count.
    #[error("MinHash signature size mismatch: expected {expected} permutations, got {actual}")]
    SignatureSizeMismatch { expected: usize, actual: usize },
}

// ─── MinHash Signature ──────────────────────────────────────────────────────

/// A MinHash signature: a fixed-size sketch of a set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinHashSignature {
    /// The hash values (one per permutation).
    pub values: Vec<u64>,
}

impl MinHashSignature {
    /// Estimate Jaccard similarity between two MinHash signatures.
    ///
    /// Returns the fraction of hash positions where the values agree.
    /// Errors with `SignatureSizeMismatch` if the two signatures disagree on
    /// permutation count; pre-0.1.0a1 builds panicked on this case instead.
    pub fn jaccard(&self, other: &MinHashSignature) -> Result<f64, MinHashError> {
        if self.values.len() != other.values.len() {
            return Err(MinHashError::SignatureSizeMismatch {
                expected: self.values.len(),
                actual: other.values.len(),
            });
        }
        if self.values.is_empty() {
            return Ok(0.0);
        }
        let matches = self
            .values
            .iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a == b)
            .count();
        Ok(matches as f64 / self.values.len() as f64)
    }
}

/// MinHash hasher: generates MinHash signatures from sets of items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinHasher {
    /// Number of hash functions (permutations).
    num_perm: usize,
    /// Seeds for each hash function.
    seeds: Vec<u64>,
}

impl MinHasher {
    /// Create a new MinHasher with the specified number of permutations.
    ///
    /// More permutations = more accurate Jaccard estimation, but larger signatures.
    /// Typical values: 64, 128, 256.
    pub fn new(num_perm: usize) -> Self {
        Self::with_seed(num_perm, 42)
    }

    /// Create a MinHasher with a specific base seed for reproducibility.
    pub fn with_seed(num_perm: usize, base_seed: u64) -> Self {
        let seeds: Vec<u64> = (0..num_perm)
            .map(|i| base_seed.wrapping_add(i as u64))
            .collect();
        Self { num_perm, seeds }
    }

    /// Number of permutations (signature size).
    pub fn num_perm(&self) -> usize {
        self.num_perm
    }

    /// Compute MinHash signature from a set of string items.
    pub fn hash_set<'a, I>(&self, items: I) -> MinHashSignature
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut min_hashes = vec![u64::MAX; self.num_perm];
        let items: Vec<&str> = items.into_iter().collect();

        if items.is_empty() {
            return MinHashSignature { values: min_hashes };
        }

        for item in &items {
            for (j, seed) in self.seeds.iter().enumerate() {
                let h = hash_with_seed(item, *seed);
                if h < min_hashes[j] {
                    min_hashes[j] = h;
                }
            }
        }

        MinHashSignature { values: min_hashes }
    }

    /// Compute MinHash signature from character shingles of a text.
    ///
    /// `shingle_size`: number of characters per shingle (e.g., 3 for trigrams).
    pub fn hash_char_shingles(&self, text: &str, shingle_size: usize) -> MinHashSignature {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < shingle_size {
            return MinHashSignature {
                values: vec![u64::MAX; self.num_perm],
            };
        }

        let mut min_hashes = vec![u64::MAX; self.num_perm];

        for window in chars.windows(shingle_size) {
            let shingle: String = window.iter().collect();
            for (j, seed) in self.seeds.iter().enumerate() {
                let h = hash_with_seed(&shingle, *seed);
                if h < min_hashes[j] {
                    min_hashes[j] = h;
                }
            }
        }

        MinHashSignature { values: min_hashes }
    }

    /// Compute MinHash signature from token (word) shingles.
    ///
    /// `shingle_size`: number of consecutive tokens per shingle (e.g., 2 for bigrams).
    pub fn hash_token_shingles(&self, tokens: &[&str], shingle_size: usize) -> MinHashSignature {
        if tokens.len() < shingle_size {
            return MinHashSignature {
                values: vec![u64::MAX; self.num_perm],
            };
        }

        let mut min_hashes = vec![u64::MAX; self.num_perm];

        for window in tokens.windows(shingle_size) {
            let shingle = window.join(" ");
            for (j, seed) in self.seeds.iter().enumerate() {
                let h = hash_with_seed(&shingle, *seed);
                if h < min_hashes[j] {
                    min_hashes[j] = h;
                }
            }
        }

        MinHashSignature { values: min_hashes }
    }
}

const MINHASH_KEY0: u64 = 0x243f_6a88_85a3_08d3;
const MINHASH_KEY1: u64 = 0x1319_8a2e_0370_7344;
const MINHASH_KEY2: u64 = 0xa409_3822_299f_31d0;
const MINHASH_KEY3: u64 = 0x082e_fa98_ec4e_6c89;
const MINHASH_MIX: u64 = 0x9e37_79b9_7f4a_7c15;

/// Hash a string with a specific seed using explicitly keyed ahash.
///
/// `AHasher::default()` uses process-global keys, so it is reproducible only
/// within one process. MinHash permutations need cross-process determinism.
#[inline]
fn hash_with_seed(s: &str, seed: u64) -> u64 {
    let state = ahash::RandomState::with_seeds(
        MINHASH_KEY0 ^ seed,
        MINHASH_KEY1 ^ seed.rotate_left(17),
        MINHASH_KEY2 ^ seed.wrapping_mul(MINHASH_MIX),
        MINHASH_KEY3 ^ seed.rotate_right(11),
    );
    state.hash_one(s)
}

// ─── LSH Index ──────────────────────────────────────────────────────────────

/// LSH (Locality-Sensitive Hashing) index for approximate nearest-neighbor queries.
///
/// Uses band/row decomposition: the signature is split into `bands` bands
/// of `rows` rows each. Two items are candidates if they agree in ALL rows
/// of ANY band.
///
/// With `b` bands and `r` rows, the probability of two items with Jaccard
/// similarity `s` being candidates is: 1 - (1 - s^r)^b
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinHashIndex {
    /// Number of bands.
    bands: usize,
    /// Number of rows per band.
    rows: usize,
    /// Band hash tables: band_index → {band_hash → [doc_ids]}.
    tables: Vec<AHashMap<u64, Vec<u32>>>,
    /// Stored signatures for similarity computation.
    signatures: AHashMap<u32, MinHashSignature>,
}

impl MinHashIndex {
    /// Create a new LSH index.
    ///
    /// `num_perm` must equal `bands * rows`.
    ///
    /// Higher bands/lower rows = higher recall, lower precision.
    /// Lower bands/higher rows = lower recall, higher precision.
    ///
    /// For a target threshold `t`, optimal parameters are:
    /// - rows ≈ 1/t
    /// - bands ≈ num_perm / rows
    pub fn new(bands: usize, rows: usize) -> Self {
        Self {
            bands,
            rows,
            tables: (0..bands).map(|_| AHashMap::new()).collect(),
            signatures: AHashMap::new(),
        }
    }

    /// Create an index tuned for a target similarity threshold.
    ///
    /// `num_perm`: signature size (e.g., 128).
    /// `threshold`: target Jaccard similarity threshold (e.g., 0.5).
    pub fn with_threshold(num_perm: usize, threshold: f64) -> Self {
        let (bands, rows) = optimal_params(num_perm, threshold);
        Self::new(bands, rows)
    }

    /// Number of permutations this index expects.
    pub fn num_perm(&self) -> usize {
        self.bands * self.rows
    }

    /// Insert a document's MinHash signature into the index.
    ///
    /// Errors with `SignatureSizeMismatch` if the signature does not contain
    /// exactly `bands * rows` permutations. Pre-0.1.0a1 builds panicked.
    pub fn insert(
        &mut self,
        doc_id: u32,
        signature: &MinHashSignature,
    ) -> Result<(), MinHashError> {
        if signature.values.len() != self.num_perm() {
            return Err(MinHashError::SignatureSizeMismatch {
                expected: self.num_perm(),
                actual: signature.values.len(),
            });
        }

        // Hash each band
        for (band_idx, band_table) in self.tables.iter_mut().enumerate() {
            let start = band_idx * self.rows;
            let end = start + self.rows;
            let band_hash = hash_band(&signature.values[start..end]);
            band_table
                .entry(band_hash)
                .or_insert_with(Vec::new)
                .push(doc_id);
        }

        self.signatures.insert(doc_id, signature.clone());
        Ok(())
    }

    /// Query for candidate near-duplicates.
    ///
    /// Returns doc_ids that are candidates (share a band hash with the query).
    /// These are approximate — use `query_with_similarity` for filtered results.
    /// Errors with `SignatureSizeMismatch` if the query signature size does
    /// not match the index's permutation count.
    pub fn query_candidates(&self, signature: &MinHashSignature) -> Result<Vec<u32>, MinHashError> {
        if signature.values.len() != self.num_perm() {
            return Err(MinHashError::SignatureSizeMismatch {
                expected: self.num_perm(),
                actual: signature.values.len(),
            });
        }

        let mut candidates = AHashSet::new();

        for (band_idx, band_table) in self.tables.iter().enumerate() {
            let start = band_idx * self.rows;
            let end = start + self.rows;
            let band_hash = hash_band(&signature.values[start..end]);

            if let Some(doc_ids) = band_table.get(&band_hash) {
                for &id in doc_ids {
                    candidates.insert(id);
                }
            }
        }

        Ok(candidates.into_iter().collect())
    }

    /// Query and filter by actual estimated Jaccard similarity.
    ///
    /// Returns (doc_id, estimated_jaccard) pairs above the threshold.
    /// Errors with `SignatureSizeMismatch` if the query signature size does
    /// not match the index's permutation count.
    pub fn query_above_threshold(
        &self,
        signature: &MinHashSignature,
        threshold: f64,
    ) -> Result<Vec<(u32, f64)>, MinHashError> {
        let candidates = self.query_candidates(signature)?;
        let mut results: Vec<(u32, f64)> = candidates
            .into_iter()
            .filter_map(|doc_id| {
                self.signatures.get(&doc_id).and_then(|stored| {
                    // jaccard returns Err only on size mismatch, which we already
                    // validated for `signature` above; stored signatures are
                    // always inserted at the index's num_perm.
                    signature.jaccard(stored).ok().map(|sim| (doc_id, sim))
                })
            })
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }

    /// Get a stored signature by doc_id.
    pub fn get_signature(&self, doc_id: u32) -> Option<&MinHashSignature> {
        self.signatures.get(&doc_id)
    }
}

/// Hash a band (slice of signature values) into a single u64.
fn hash_band(values: &[u64]) -> u64 {
    let state = ahash::RandomState::with_seeds(
        MINHASH_KEY0,
        MINHASH_KEY1,
        MINHASH_KEY2,
        MINHASH_KEY3 ^ MINHASH_MIX,
    );
    state.hash_one(values)
}

/// Compute optimal band/row parameters for a target threshold.
fn optimal_params(num_perm: usize, threshold: f64) -> (usize, usize) {
    // Find the (b, r) pair where b*r <= num_perm that minimizes
    // the difference between the S-curve threshold and the target.
    let mut best_bands = 1;
    let mut best_rows = num_perm;
    let mut best_error = f64::MAX;

    for rows in 1..=num_perm {
        let bands = num_perm / rows;
        if bands == 0 || bands * rows != num_perm {
            continue;
        }
        // The threshold of the S-curve: (1/b)^(1/r)
        let curve_threshold = (1.0 / bands as f64).powf(1.0 / rows as f64);
        let error = (curve_threshold - threshold).abs();
        if error < best_error {
            best_error = error;
            best_bands = bands;
            best_rows = rows;
        }
    }

    (best_bands, best_rows)
}

// ─── Document Deduplication ─────────────────────────────────────────────────

/// Result of duplicate detection for a set of documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// The canonical (first-seen) document ID.
    pub canonical_id: u32,
    /// Duplicate document IDs with their similarity to the canonical.
    pub duplicates: Vec<(u32, f64)>,
}

/// Find duplicate groups in a set of documents.
///
/// Returns groups where each group has a canonical document and its duplicates.
pub fn find_duplicates(
    hasher: &MinHasher,
    documents: &[(u32, Vec<&str>)], // (doc_id, tokens)
    shingle_size: usize,
    threshold: f64,
) -> Vec<DuplicateGroup> {
    let num_perm = hasher.num_perm();
    let mut index = MinHashIndex::with_threshold(num_perm, threshold);

    // Build signatures and index. Sizes are structurally consistent here:
    // every signature comes from the same `hasher` whose `num_perm()` matches
    // the `MinHashIndex::with_threshold` we just constructed, so insert /
    // query never observe a mismatch from user input.
    let mut signatures: Vec<(u32, MinHashSignature)> = Vec::with_capacity(documents.len());
    for (doc_id, tokens) in documents {
        let sig = hasher.hash_token_shingles(tokens, shingle_size);
        index
            .insert(*doc_id, &sig)
            .expect("internal invariant: hasher and index share num_perm");
        signatures.push((*doc_id, sig));
    }

    // Find duplicate groups
    let mut seen = AHashSet::new();
    let mut groups = Vec::new();

    for (doc_id, sig) in &signatures {
        if seen.contains(doc_id) {
            continue;
        }

        let matches = index
            .query_above_threshold(sig, threshold)
            .expect("internal invariant: hasher and index share num_perm");
        let duplicates: Vec<(u32, f64)> = matches
            .into_iter()
            .filter(|(id, _)| *id != *doc_id && !seen.contains(id))
            .collect();

        if !duplicates.is_empty() {
            for (dup_id, _) in &duplicates {
                seen.insert(*dup_id);
            }
            groups.push(DuplicateGroup {
                canonical_id: *doc_id,
                duplicates,
            });
        }

        seen.insert(*doc_id);
    }

    groups
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minhash_identical_sets() {
        let hasher = MinHasher::new(128);
        let items = ["apple", "banana", "cherry"];
        let sig1 = hasher.hash_set(items.iter().copied());
        let sig2 = hasher.hash_set(items.iter().copied());
        assert_eq!(sig1.jaccard(&sig2).unwrap(), 1.0);
    }

    #[test]
    fn test_minhash_disjoint_sets() {
        let hasher = MinHasher::new(128);
        let sig1 = hasher.hash_set(["apple", "banana", "cherry"].iter().copied());
        let sig2 = hasher.hash_set(["dog", "elephant", "fox"].iter().copied());
        let sim = sig1.jaccard(&sig2).unwrap();
        // Disjoint sets should have near-zero similarity
        assert!(sim < 0.2, "Disjoint sets similarity {} too high", sim);
    }

    #[test]
    fn test_minhash_partial_overlap() {
        let hasher = MinHasher::new(256);
        let sig1 = hasher.hash_set(["a", "b", "c", "d"].iter().copied());
        let sig2 = hasher.hash_set(["a", "b", "c", "e"].iter().copied());
        let sim = sig1.jaccard(&sig2).unwrap();
        // True Jaccard = 3/5 = 0.6
        assert!((sim - 0.6).abs() < 0.15, "Expected ~0.6, got {}", sim);
    }

    #[test]
    fn test_minhash_char_shingles() {
        let hasher = MinHasher::new(128);
        let sig1 = hasher.hash_char_shingles("hello world", 3);
        let sig2 = hasher.hash_char_shingles("hello world!", 3);
        let sim = sig1.jaccard(&sig2).unwrap();
        assert!(
            sim > 0.5,
            "Similar strings should have high similarity: {}",
            sim
        );
    }

    #[test]
    fn test_minhash_token_shingles() {
        let hasher = MinHasher::new(128);
        let tokens1 = vec!["the", "cat", "sat", "on", "the", "mat"];
        let tokens2 = vec!["the", "cat", "sat", "on", "the", "log"];
        let sig1 = hasher.hash_token_shingles(&tokens1, 2);
        let sig2 = hasher.hash_token_shingles(&tokens2, 2);
        let sim = sig1.jaccard(&sig2).unwrap();
        assert!(sim > 0.3, "Similar token sequences should overlap: {}", sim);
    }

    #[test]
    fn test_lsh_index() {
        let hasher = MinHasher::new(128);
        let mut index = MinHashIndex::with_threshold(128, 0.5);

        let sig1 = hasher.hash_set(["a", "b", "c", "d"].iter().copied());
        let sig2 = hasher.hash_set(["a", "b", "c", "e"].iter().copied());
        let sig3 = hasher.hash_set(["x", "y", "z"].iter().copied());

        index.insert(0, &sig1).unwrap();
        index.insert(1, &sig2).unwrap();
        index.insert(2, &sig3).unwrap();

        // Query with sig1 — should find sig2 as candidate
        let candidates = index.query_candidates(&sig1).unwrap();
        assert!(candidates.contains(&0));
        // sig3 should generally NOT be a candidate
    }

    #[test]
    fn test_lsh_query_with_threshold() {
        let hasher = MinHasher::new(128);
        let mut index = MinHashIndex::with_threshold(128, 0.3);

        let sig1 = hasher.hash_set(["a", "b", "c", "d"].iter().copied());
        let sig2 = hasher.hash_set(["a", "b", "c", "e"].iter().copied());
        let sig3 = hasher.hash_set(["x", "y", "z", "w"].iter().copied());

        index.insert(0, &sig1).unwrap();
        index.insert(1, &sig2).unwrap();
        index.insert(2, &sig3).unwrap();

        let results = index.query_above_threshold(&sig1, 0.4).unwrap();
        // Should include self (0) and similar (1), but probably not dissimilar (2)
        let ids: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&0));
    }

    #[test]
    fn test_find_duplicates() {
        let hasher = MinHasher::new(128);
        let docs = vec![
            (0, vec!["the", "cat", "sat", "on", "the", "mat"]),
            (1, vec!["the", "cat", "sat", "on", "the", "mat"]), // exact duplicate
            (2, vec!["a", "completely", "different", "document"]),
        ];
        let groups = find_duplicates(&hasher, &docs, 2, 0.5);
        // Doc 0 and 1 should be in the same duplicate group
        assert!(!groups.is_empty());
        let first = &groups[0];
        assert!(first.canonical_id == 0 || first.canonical_id == 1);
        assert!(!first.duplicates.is_empty());
    }

    #[test]
    fn test_optimal_params() {
        let (bands, rows) = optimal_params(128, 0.5);
        assert_eq!(bands * rows, 128);
        // For threshold 0.5, the S-curve threshold should be close to 0.5
        let curve_t = (1.0 / bands as f64).powf(1.0 / rows as f64);
        assert!(
            (curve_t - 0.5).abs() < 0.15,
            "S-curve threshold {} not close to 0.5",
            curve_t
        );
    }

    #[test]
    fn test_minhash_empty_set() {
        let hasher = MinHasher::new(64);
        let sig = hasher.hash_set(std::iter::empty());
        assert_eq!(sig.values.len(), 64);
        assert!(sig.values.iter().all(|&v| v == u64::MAX));
    }

    #[test]
    fn test_different_seeds_different_signatures() {
        let hasher_a = MinHasher::with_seed(128, 42);
        let hasher_b = MinHasher::with_seed(128, 99);
        let items = ["apple", "banana", "cherry"];
        let sig_a = hasher_a.hash_set(items.iter().copied());
        let sig_b = hasher_b.hash_set(items.iter().copied());
        // Different seeds should produce different signatures
        assert_ne!(sig_a.values, sig_b.values);
    }

    #[test]
    fn test_seeded_minhash_stable_values() {
        let hasher = MinHasher::with_seed(4, 123);
        let sig = hasher.hash_set(["alpha", "beta", "gamma"].iter().copied());
        assert_eq!(
            sig.values,
            vec![
                3521687781905391119,
                3954226141651166389,
                130310390849401082,
                4096582154791826708,
            ]
        );
    }

    #[test]
    fn test_find_duplicates_empty() {
        let hasher = MinHasher::new(128);
        let docs: Vec<(u32, Vec<&str>)> = vec![];
        let groups = find_duplicates(&hasher, &docs, 2, 0.5);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_find_duplicates_all_unique() {
        let hasher = MinHasher::new(128);
        let docs = vec![
            (0, vec!["alpha", "beta", "gamma", "delta", "epsilon"]),
            (1, vec!["one", "two", "three", "four", "five"]),
            (2, vec!["red", "green", "blue", "yellow", "purple"]),
        ];
        let groups = find_duplicates(&hasher, &docs, 2, 0.5);
        // Completely different documents should not form any duplicate groups
        assert!(
            groups.is_empty(),
            "Expected no duplicate groups, got {:?}",
            groups
        );
    }

    #[test]
    fn test_lsh_index_len_and_empty() {
        let mut index = MinHashIndex::new(4, 4); // 16 permutations
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        let hasher = MinHasher::new(16);
        let sig = hasher.hash_set(["a", "b", "c"].iter().copied());
        index.insert(0, &sig).unwrap();
        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);

        let sig2 = hasher.hash_set(["d", "e", "f"].iter().copied());
        index.insert(1, &sig2).unwrap();
        assert_eq!(index.len(), 2);
    }
}
