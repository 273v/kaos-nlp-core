//! Vocabulary data structures for fast term lookup.

use ahash::{AHashMap, AHashSet};
use fastbloom::BloomFilter;
use serde::{Deserialize, Serialize};

/// Simple set-based vocabulary for O(1) membership checking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SetVocabulary {
    terms: AHashSet<String>,
}

impl SetVocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self {
            terms: iter.into_iter().map(|s| s.as_ref().to_string()).collect(),
        }
    }

    pub fn insert(&mut self, term: &str) -> bool {
        self.terms.insert(term.to_string())
    }

    pub fn contains(&self, term: &str) -> bool {
        self.terms.contains(term)
    }

    pub fn remove(&mut self, term: &str) -> bool {
        self.terms.remove(term)
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.terms.iter()
    }
}

/// Vocabulary with frequency counts per term.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrequencyVocabulary {
    /// term -> (id, count)
    terms: AHashMap<String, (u32, u64)>,
    next_id: u32,
}

impl FrequencyVocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or increment a term. Returns the term's ID.
    pub fn insert(&mut self, term: &str) -> u32 {
        if let Some((id, count)) = self.terms.get_mut(term) {
            *count += 1;
            *id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.terms.insert(term.to_string(), (id, 1));
            id
        }
    }

    /// Insert with a specific count.
    pub fn insert_with_count(&mut self, term: &str, count: u64) -> u32 {
        if let Some((id, existing)) = self.terms.get_mut(term) {
            *existing += count;
            *id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.terms.insert(term.to_string(), (id, count));
            id
        }
    }

    pub fn contains(&self, term: &str) -> bool {
        self.terms.contains_key(term)
    }

    pub fn get_count(&self, term: &str) -> Option<u64> {
        self.terms.get(term).map(|(_, c)| *c)
    }

    pub fn get_id(&self, term: &str) -> Option<u32> {
        self.terms.get(term).map(|(id, _)| *id)
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Return terms sorted by frequency (descending).
    pub fn top_n(&self, n: usize) -> Vec<(String, u64)> {
        let mut entries: Vec<_> = self
            .terms
            .iter()
            .map(|(k, (_, c))| (k.clone(), *c))
            .collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.1));
        entries.truncate(n);
        entries
    }

    /// Total count across all terms.
    pub fn total_count(&self) -> u64 {
        self.terms.values().map(|(_, c)| c).sum()
    }
}

/// ID-indexed vocabulary with bidirectional lookup.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexedVocabulary {
    term_to_id: AHashMap<String, u32>,
    id_to_term: Vec<String>,
}

impl IndexedVocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a term, returning its ID. Existing terms return their current ID.
    pub fn insert(&mut self, term: &str) -> u32 {
        if let Some(&id) = self.term_to_id.get(term) {
            id
        } else {
            let id = self.id_to_term.len() as u32;
            self.term_to_id.insert(term.to_string(), id);
            self.id_to_term.push(term.to_string());
            id
        }
    }

    pub fn get_id(&self, term: &str) -> Option<u32> {
        self.term_to_id.get(term).copied()
    }

    pub fn get_term(&self, id: u32) -> Option<&str> {
        self.id_to_term.get(id as usize).map(|s| s.as_str())
    }

    pub fn contains(&self, term: &str) -> bool {
        self.term_to_id.contains_key(term)
    }

    pub fn len(&self) -> usize {
        self.id_to_term.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_term.is_empty()
    }
}

/// Bloom filter vocabulary for approximate membership testing.
///
/// Uses much less memory than a HashSet, but has a small false positive rate.
/// No false negatives: if `contains` returns false, the term is definitely absent.
pub struct BloomVocabulary {
    filter: BloomFilter,
    approx_count: usize,
}

impl BloomVocabulary {
    /// Create a bloom vocabulary sized for `expected_items` with given false positive rate.
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let filter =
            BloomFilter::with_false_pos(false_positive_rate).expected_items(expected_items);
        Self {
            filter,
            approx_count: 0,
        }
    }

    pub fn insert(&mut self, term: &str) {
        self.filter.insert(term);
        self.approx_count += 1;
    }

    /// Check membership. May return false positives but never false negatives.
    pub fn contains(&self, term: &str) -> bool {
        self.filter.contains(term)
    }

    pub fn approx_len(&self) -> usize {
        self.approx_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_vocabulary() {
        let mut v = SetVocabulary::new();
        assert!(v.insert("hello"));
        assert!(!v.insert("hello")); // duplicate
        assert!(v.contains("hello"));
        assert!(!v.contains("world"));
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_frequency_vocabulary() {
        let mut v = FrequencyVocabulary::new();
        let id1 = v.insert("cat");
        let id2 = v.insert("cat");
        assert_eq!(id1, id2);
        assert_eq!(v.get_count("cat"), Some(2));
        let top = v.top_n(1);
        assert_eq!(top[0].0, "cat");
    }

    #[test]
    fn test_indexed_vocabulary() {
        let mut v = IndexedVocabulary::new();
        let id0 = v.insert("hello");
        let id1 = v.insert("world");
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(v.get_term(0), Some("hello"));
        assert_eq!(v.get_id("world"), Some(1));
    }

    #[test]
    fn test_bloom_vocabulary() {
        let mut v = BloomVocabulary::new(1000, 0.01);
        v.insert("hello");
        v.insert("world");
        assert!(v.contains("hello"));
        assert!(v.contains("world"));
        // "xyz" should almost certainly be absent (1% FP rate)
        // We can't assert !contains for bloom, but we can test the pattern
    }

    #[test]
    fn test_set_from_iter() {
        let v = SetVocabulary::build(["a", "b", "c"]);
        assert_eq!(v.len(), 3);
        assert!(v.contains("b"));
    }
}
