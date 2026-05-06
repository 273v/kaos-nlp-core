//! N-gram based similarity metrics: character-level and token/word-level.

use ahash::AHashMap;

use crate::core::algorithms::traits::{DistanceOutput, DistanceResult, StringDistance};

// ---------------------------------------------------------------------------
// Shared helpers for computing similarity from frequency maps
// ---------------------------------------------------------------------------

/// Multiset Jaccard: sum(min) / sum(max) over all keys.
fn jaccard_from_freqs<K: std::hash::Hash + Eq>(
    a: &AHashMap<K, usize>,
    b: &AHashMap<K, usize>,
) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }

    let mut intersection = 0usize;
    let mut union = 0usize;

    let mut all_keys: ahash::AHashSet<&K> = ahash::AHashSet::new();
    all_keys.extend(a.keys());
    all_keys.extend(b.keys());

    for key in &all_keys {
        let ca = a.get(*key).copied().unwrap_or(0);
        let cb = b.get(*key).copied().unwrap_or(0);
        intersection += ca.min(cb);
        union += ca.max(cb);
    }

    if union > 0 {
        intersection as f64 / union as f64
    } else {
        1.0
    }
}

/// Cosine similarity from two frequency maps.
fn cosine_from_freqs<K: std::hash::Hash + Eq>(
    a: &AHashMap<K, usize>,
    b: &AHashMap<K, usize>,
) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for &v in a.values() {
        norm_a += (v as f64).powi(2);
    }
    for &v in b.values() {
        norm_b += (v as f64).powi(2);
    }
    for (key, &va) in a {
        if let Some(&vb) = b.get(key) {
            dot += va as f64 * vb as f64;
        }
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Overlap coefficient from two frequency maps: sum(min) / min(sum_a, sum_b).
fn overlap_from_freqs<K: std::hash::Hash + Eq>(
    a: &AHashMap<K, usize>,
    b: &AHashMap<K, usize>,
) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }

    let mut intersection = 0usize;
    for (key, &va) in a {
        if let Some(&vb) = b.get(key) {
            intersection += va.min(vb);
        }
    }

    let total_a: usize = a.values().sum();
    let total_b: usize = b.values().sum();
    let min_total = total_a.min(total_b);

    if min_total > 0 {
        intersection as f64 / min_total as f64
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Character n-grams
// ---------------------------------------------------------------------------

/// Extract character n-grams from a string.
fn char_ngrams(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < n {
        return vec![];
    }
    chars.windows(n).map(|w| w.iter().collect()).collect()
}

/// Build a frequency map of character n-grams.
fn ngram_freq(s: &str, n: usize) -> AHashMap<String, usize> {
    let mut freq = AHashMap::new();
    for ng in char_ngrams(s, n) {
        *freq.entry(ng).or_insert(0) += 1;
    }
    freq
}

/// Jaccard similarity over character n-grams.
///
/// Jaccard(A, B) = |A ∩ B| / |A ∪ B| where A, B are n-gram multisets.
#[derive(Debug, Clone)]
pub struct NgramJaccard {
    pub n: usize,
}

impl Default for NgramJaccard {
    fn default() -> Self {
        Self { n: 2 }
    }
}

impl StringDistance for NgramJaccard {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let a_freq = ngram_freq(a, self.n);
        let b_freq = ngram_freq(b, self.n);
        Ok(DistanceOutput::from_similarity(jaccard_from_freqs(
            &a_freq, &b_freq,
        )))
    }
}

/// Cosine similarity over character n-gram frequency vectors.
#[derive(Debug, Clone)]
pub struct NgramCosine {
    pub n: usize,
}

impl Default for NgramCosine {
    fn default() -> Self {
        Self { n: 2 }
    }
}

impl StringDistance for NgramCosine {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let a_freq = ngram_freq(a, self.n);
        let b_freq = ngram_freq(b, self.n);
        Ok(DistanceOutput::from_similarity(cosine_from_freqs(
            &a_freq, &b_freq,
        )))
    }
}

/// Overlap coefficient over character n-gram sets.
///
/// Overlap(A, B) = |A ∩ B| / min(|A|, |B|).
#[derive(Debug, Clone)]
pub struct NgramOverlap {
    pub n: usize,
}

impl Default for NgramOverlap {
    fn default() -> Self {
        Self { n: 2 }
    }
}

impl StringDistance for NgramOverlap {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let a_freq = ngram_freq(a, self.n);
        let b_freq = ngram_freq(b, self.n);
        Ok(DistanceOutput::from_similarity(overlap_from_freqs(
            &a_freq, &b_freq,
        )))
    }
}

// ---------------------------------------------------------------------------
// Token (word) extraction — delegates to crate::core::tokenizer
// ---------------------------------------------------------------------------

use crate::core::tokenizer::{tokenize_words, TokenizerConfig};

/// Build a TokenizerConfig for token similarity operations.
fn make_token_config(lowercase: bool) -> TokenizerConfig {
    let config = TokenizerConfig::new();
    if lowercase {
        config.lowercase()
    } else {
        config
    }
}

/// Build a frequency map of word tokens.
fn token_freq(s: &str, lowercase: bool) -> AHashMap<String, usize> {
    let config = make_token_config(lowercase);
    let mut freq = AHashMap::new();
    for tok in tokenize_words(s, &config) {
        *freq.entry(tok).or_insert(0) += 1;
    }
    freq
}

/// Extract token n-grams (sliding windows of n consecutive words).
///
/// Each n-gram is represented as a tuple of words joined by "\t" to form a
/// hashable key. This matches the kelvin-nlp `get_ngrams` behavior of
/// returning tuples of tokens.
fn token_ngram_freq(s: &str, n: usize, lowercase: bool) -> AHashMap<String, usize> {
    let config = make_token_config(lowercase);
    let tokens = tokenize_words(s, &config);
    let mut freq = AHashMap::new();
    if tokens.len() < n {
        if tokens.len() == 1 {
            let key = tokens[0].clone();
            *freq.entry(key).or_insert(0) += 1;
        }
        return freq;
    }
    for window in tokens.windows(n) {
        let key: String = window.join("\t");
        *freq.entry(key).or_insert(0) += 1;
    }
    freq
}

// ---------------------------------------------------------------------------
// Token-level similarity structs
// ---------------------------------------------------------------------------

/// Jaccard similarity over word tokens (unigrams).
///
/// Tokenizes both strings, builds multiset frequency maps, and computes
/// multiset Jaccard. This matches `get_token_jaccard` from kelvin-nlp.
#[derive(Debug, Clone, Default)]
pub struct TokenJaccard {
    pub lowercase: bool,
}

impl StringDistance for TokenJaccard {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let a_freq = token_freq(a, self.lowercase);
        let b_freq = token_freq(b, self.lowercase);
        Ok(DistanceOutput::from_similarity(jaccard_from_freqs(
            &a_freq, &b_freq,
        )))
    }
}

/// Jaccard similarity over token n-grams (word bigrams, trigrams, etc.).
///
/// This matches `get_ngram_jaccard` from kelvin-nlp.
#[derive(Debug, Clone)]
pub struct TokenNgramJaccard {
    pub n: usize,
    pub lowercase: bool,
}

impl Default for TokenNgramJaccard {
    fn default() -> Self {
        Self {
            n: 2,
            lowercase: false,
        }
    }
}

impl StringDistance for TokenNgramJaccard {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let a_freq = token_ngram_freq(a, self.n, self.lowercase);
        let b_freq = token_ngram_freq(b, self.n, self.lowercase);
        Ok(DistanceOutput::from_similarity(jaccard_from_freqs(
            &a_freq, &b_freq,
        )))
    }
}

/// Cosine similarity over token n-gram frequency vectors.
#[derive(Debug, Clone)]
pub struct TokenNgramCosine {
    pub n: usize,
    pub lowercase: bool,
}

impl Default for TokenNgramCosine {
    fn default() -> Self {
        Self {
            n: 2,
            lowercase: false,
        }
    }
}

impl StringDistance for TokenNgramCosine {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let a_freq = token_ngram_freq(a, self.n, self.lowercase);
        let b_freq = token_ngram_freq(b, self.n, self.lowercase);
        Ok(DistanceOutput::from_similarity(cosine_from_freqs(
            &a_freq, &b_freq,
        )))
    }
}

/// Overlap coefficient over token n-gram sets.
#[derive(Debug, Clone)]
pub struct TokenNgramOverlap {
    pub n: usize,
    pub lowercase: bool,
}

impl Default for TokenNgramOverlap {
    fn default() -> Self {
        Self {
            n: 2,
            lowercase: false,
        }
    }
}

impl StringDistance for TokenNgramOverlap {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let a_freq = token_ngram_freq(a, self.n, self.lowercase);
        let b_freq = token_ngram_freq(b, self.n, self.lowercase);
        Ok(DistanceOutput::from_similarity(overlap_from_freqs(
            &a_freq, &b_freq,
        )))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Character n-gram tests ---

    #[test]
    fn test_ngram_jaccard_identical() {
        let nj = NgramJaccard { n: 2 };
        let r = nj.distance("hello", "hello").unwrap();
        assert_eq!(r.similarity, 1.0);
    }

    #[test]
    fn test_ngram_jaccard_different() {
        let nj = NgramJaccard { n: 2 };
        let r = nj.distance("abc", "xyz").unwrap();
        assert_eq!(r.similarity, 0.0);
    }

    #[test]
    fn test_ngram_cosine() {
        let nc = NgramCosine { n: 2 };
        let r = nc.distance("night", "nacht").unwrap();
        assert!(r.similarity > 0.0 && r.similarity < 1.0);
    }

    #[test]
    fn test_ngram_overlap_subset() {
        let no = NgramOverlap { n: 2 };
        // "ab" bigrams are a subset of "abc" bigrams
        let r = no.distance("ab", "abc").unwrap();
        assert_eq!(r.similarity, 1.0);
    }

    #[test]
    fn test_trigrams() {
        let nj = NgramJaccard { n: 3 };
        let r = nj.distance("hello world", "hello world").unwrap();
        assert_eq!(r.similarity, 1.0);
    }

    // --- Token Jaccard tests ---

    #[test]
    fn test_token_jaccard_identical() {
        let tj = TokenJaccard { lowercase: false };
        let r = tj
            .distance("the quick brown fox", "the quick brown fox")
            .unwrap();
        assert_eq!(r.similarity, 1.0);
    }

    #[test]
    fn test_token_jaccard_partial() {
        let tj = TokenJaccard { lowercase: true };
        let r = tj
            .distance("the quick brown fox", "a quick brown dog")
            .unwrap();
        // shared: {quick, brown} = 2, union: {the, quick, brown, fox, a, dog} = 6
        assert!((r.similarity - 2.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_token_jaccard_no_overlap() {
        let tj = TokenJaccard { lowercase: false };
        let r = tj.distance("hello world", "foo bar").unwrap();
        assert_eq!(r.similarity, 0.0);
    }

    #[test]
    fn test_token_jaccard_case_sensitive() {
        let tj = TokenJaccard { lowercase: false };
        let r = tj.distance("Hello", "hello").unwrap();
        assert_eq!(r.similarity, 0.0);
    }

    #[test]
    fn test_token_jaccard_case_insensitive() {
        let tj = TokenJaccard { lowercase: true };
        let r = tj.distance("Hello", "hello").unwrap();
        assert_eq!(r.similarity, 1.0);
    }

    // --- Token n-gram Jaccard tests ---

    #[test]
    fn test_token_ngram_jaccard_identical() {
        let tnj = TokenNgramJaccard {
            n: 2,
            lowercase: false,
        };
        let r = tnj
            .distance("the quick brown fox jumps", "the quick brown fox jumps")
            .unwrap();
        assert_eq!(r.similarity, 1.0);
    }

    #[test]
    fn test_token_ngram_jaccard_partial() {
        let tnj = TokenNgramJaccard {
            n: 2,
            lowercase: true,
        };
        let r = tnj
            .distance("the quick brown fox", "a quick brown dog")
            .unwrap();
        // bigrams A: {the quick, quick brown, brown fox}
        // bigrams B: {a quick, quick brown, brown dog}
        // intersection: {quick brown} = 1, union = 5
        assert!((r.similarity - 1.0 / 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_token_ngram_jaccard_no_overlap() {
        let tnj = TokenNgramJaccard {
            n: 2,
            lowercase: false,
        };
        let r = tnj.distance("hello world", "foo bar").unwrap();
        assert_eq!(r.similarity, 0.0);
    }

    // --- Token n-gram cosine tests ---

    #[test]
    fn test_token_ngram_cosine_identical() {
        let tnc = TokenNgramCosine {
            n: 2,
            lowercase: false,
        };
        let r = tnc
            .distance("the quick brown fox", "the quick brown fox")
            .unwrap();
        assert!((r.similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_token_ngram_cosine_partial() {
        let tnc = TokenNgramCosine {
            n: 2,
            lowercase: true,
        };
        let r = tnc
            .distance("the quick brown fox", "a quick brown dog")
            .unwrap();
        assert!(r.similarity > 0.0 && r.similarity < 1.0);
    }

    // --- Token n-gram overlap tests ---

    #[test]
    fn test_token_ngram_overlap_subset() {
        let tno = TokenNgramOverlap {
            n: 2,
            lowercase: true,
        };
        // "quick brown" is a subset bigram of the longer string
        let r = tno.distance("quick brown", "the quick brown fox").unwrap();
        assert_eq!(r.similarity, 1.0);
    }

    // --- Empty / edge cases ---

    #[test]
    fn test_token_jaccard_both_empty() {
        let tj = TokenJaccard { lowercase: false };
        let r = tj.distance("", "").unwrap();
        assert_eq!(r.similarity, 1.0);
    }

    #[test]
    fn test_token_ngram_jaccard_single_token() {
        let tnj = TokenNgramJaccard {
            n: 2,
            lowercase: false,
        };
        // Single token with bigram n=2 — should still produce a key
        let r = tnj.distance("hello", "hello").unwrap();
        assert_eq!(r.similarity, 1.0);
    }
}
