//! Finite State Transducer (FST) based matching.
//!
//! Uses the `fst` crate for compact, memory-mappable ordered string sets and maps.
//! Supports exact lookup, prefix search, and Levenshtein fuzzy search.

use fst::automaton::Levenshtein;
use fst::{Automaton, IntoStreamer, Set, SetBuilder, Streamer};
use serde::{Deserialize, Serialize};

/// A match result from FST lookup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FstMatch {
    /// The matched key from the FST.
    pub key: String,
    /// Edit distance from the query (0 for exact matches).
    pub distance: u32,
}

/// An immutable FST-backed string set for fast exact and fuzzy lookup.
///
/// The set is compact (typically 2-10x smaller than a HashSet) and supports
/// Levenshtein-automaton fuzzy search natively.
pub struct FstSet {
    set: Set<Vec<u8>>,
    len: usize,
}

impl FstSet {
    /// Build an FST set from an iterator of strings.
    ///
    /// **Important:** Strings must be provided in sorted (lexicographic byte) order,
    /// or they will be sorted internally.
    pub fn build<I, S>(iter: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut keys: Vec<String> = iter.into_iter().map(|s| s.as_ref().to_string()).collect();
        keys.sort();
        keys.dedup();
        let len = keys.len();

        let mut builder = SetBuilder::memory();
        for key in &keys {
            builder
                .insert(key.as_bytes())
                .map_err(|e| format!("FST build error: {e}"))?;
        }
        let bytes = builder
            .into_inner()
            .map_err(|e| format!("FST finalize error: {e}"))?;
        let set = Set::new(bytes).map_err(|e| format!("FST load error: {e}"))?;
        Ok(Self { set, len })
    }

    /// Check if a key exists in the set (exact match).
    pub fn contains(&self, key: &str) -> bool {
        self.set.contains(key.as_bytes())
    }

    /// Find all keys that are within `max_distance` edits of `query`.
    pub fn fuzzy_search(&self, query: &str, max_distance: u32) -> Result<Vec<FstMatch>, String> {
        let lev = Levenshtein::new(query, max_distance)
            .map_err(|e| format!("Levenshtein automaton error: {e}"))?;
        let mut stream = self.set.search(&lev).into_stream();
        let mut results = Vec::new();
        while let Some(key_bytes) = stream.next() {
            if let Ok(key) = std::str::from_utf8(key_bytes) {
                // Compute actual edit distance for the result.
                let dist = strsim::levenshtein(query, key) as u32;
                results.push(FstMatch {
                    key: key.to_string(),
                    distance: dist,
                });
            }
        }
        results.sort_by_key(|m| (m.distance, m.key.clone()));
        Ok(results)
    }

    /// Return the raw FST byte buffer.
    ///
    /// The FST format is self-describing — the same bytes can be loaded with
    /// `FstSet::from_bytes` or memory-mapped by callers that want to avoid a
    /// heap copy.
    pub fn as_bytes(&self) -> &[u8] {
        self.set.as_fst().as_bytes()
    }

    /// Load an FstSet from a raw FST byte buffer.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        let set = Set::new(bytes).map_err(|e| format!("FST load error: {e}"))?;
        let len = set.len();
        Ok(Self { set, len })
    }

    /// Write the raw FST bytes to disk.
    ///
    /// The on-disk file is the FST byte buffer with no additional framing —
    /// the FST format embeds its own header and length.
    pub fn save_to_path(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.as_bytes()).map_err(|e| format!("write {path}: {e}"))
    }

    /// Load an FstSet from a file written by `save_to_path`.
    pub fn load_from_path(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        Self::from_bytes(bytes)
    }

    /// Find all keys with the given prefix.
    pub fn prefix_search(&self, prefix: &str) -> Vec<String> {
        let automaton = fst::automaton::Str::new(prefix).starts_with();
        let mut stream = self.set.search(&automaton).into_stream();
        let mut results = Vec::new();
        while let Some(key_bytes) = stream.next() {
            if let Ok(key) = std::str::from_utf8(key_bytes) {
                results.push(key.to_string());
            }
        }
        results
    }

    /// Return the number of keys in the set.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// An immutable FST-backed string-to-u64 map.
///
/// Useful for weighted dictionaries (e.g., term frequencies).
pub struct FstMap {
    map: fst::Map<Vec<u8>>,
    len: usize,
}

impl FstMap {
    /// Build an FST map from key-value pairs.
    /// Keys will be sorted internally.
    pub fn build<I, S>(iter: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = (S, u64)>,
        S: AsRef<str>,
    {
        let mut pairs: Vec<(String, u64)> = iter
            .into_iter()
            .map(|(k, v)| (k.as_ref().to_string(), v))
            .collect();
        pairs.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        pairs.dedup_by(|a, b| a.0 == b.0);
        let len = pairs.len();

        let mut builder = fst::MapBuilder::memory();
        for (key, val) in &pairs {
            builder
                .insert(key.as_bytes(), *val)
                .map_err(|e| format!("FST map build error: {e}"))?;
        }
        let bytes = builder
            .into_inner()
            .map_err(|e| format!("FST map finalize error: {e}"))?;
        let map = fst::Map::new(bytes).map_err(|e| format!("FST map load error: {e}"))?;
        Ok(Self { map, len })
    }

    /// Look up a key and return its value.
    pub fn get(&self, key: &str) -> Option<u64> {
        self.map.get(key.as_bytes())
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key.as_bytes())
    }

    /// Return the number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fst_set_contains() {
        let set = FstSet::build(["apple", "banana", "cherry", "date"]).unwrap();
        assert!(set.contains("banana"));
        assert!(!set.contains("grape"));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_fst_set_fuzzy() {
        let set = FstSet::build(["apple", "apply", "ample", "maple", "orange"]).unwrap();
        let results = set.fuzzy_search("aple", 2).unwrap();
        let keys: Vec<&str> = results.iter().map(|m| m.key.as_str()).collect();
        assert!(keys.contains(&"apple"));
        assert!(keys.contains(&"ample"));
    }

    #[test]
    fn test_fst_set_prefix() {
        let set = FstSet::build(["app", "apple", "application", "banana"]).unwrap();
        let results = set.prefix_search("app");
        assert_eq!(results.len(), 3);
        assert!(results.contains(&"app".to_string()));
        assert!(results.contains(&"apple".to_string()));
        assert!(results.contains(&"application".to_string()));
    }

    #[test]
    fn test_fst_map() {
        let map = FstMap::build([("cat", 10u64), ("dog", 20), ("bird", 5)]).unwrap();
        assert_eq!(map.get("cat"), Some(10));
        assert_eq!(map.get("dog"), Some(20));
        assert_eq!(map.get("fish"), None);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_fst_set_dedup() {
        let set = FstSet::build(["a", "b", "a", "c", "b"]).unwrap();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_fst_set_empty() {
        let empty: Vec<&str> = vec![];
        let set = FstSet::build(empty).unwrap();
        assert!(set.is_empty());
        assert!(!set.contains("anything"));
    }
}
