//! Utility types for the Punkt algorithm: text preprocessing, frequency distributions.
//!
//! Ported from nupunkt-rs/src/utils.rs.

use ahash::AHashMap;
use regex::Regex;

// ─── PairIter ────────────────────────────────────────────────────────────────

/// Iterator over consecutive pairs of items.
pub struct PairIter<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    iter: I,
    prev: Option<T>,
}

impl<I, T> PairIter<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    pub fn new(mut iter: I) -> Self {
        let prev = iter.next();
        Self { iter, prev }
    }
}

impl<I, T> Iterator for PairIter<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    type Item = (T, Option<T>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(prev) = self.prev.take() {
            let next = self.iter.next();
            self.prev = next.clone();
            Some((prev, next))
        } else {
            None
        }
    }
}

/// Create a pair iterator from an iterator.
pub fn pair_iter<I, T>(iter: I) -> PairIter<I::IntoIter, T>
where
    I: IntoIterator<Item = T>,
    T: Clone,
{
    PairIter::new(iter.into_iter())
}

// ─── TextPreprocessor ────────────────────────────────────────────────────────

/// Text preprocessing: word tokenization with spacing information.
#[derive(Clone)]
pub struct TextPreprocessor {
    word_tokenize_pattern: Regex,
}

impl TextPreprocessor {
    pub fn new(pattern: &str) -> Result<Self, regex::Error> {
        Ok(Self {
            word_tokenize_pattern: Regex::new(pattern)?,
        })
    }

    /// Tokenize text into words.
    pub fn word_tokenize(&self, text: &str) -> Vec<String> {
        self.word_tokenize_pattern
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    /// Tokenize with spacing information: (word, spaces_after, has_newline, byte_position).
    pub fn word_tokenize_with_spacing(&self, text: &str) -> Vec<(String, u8, bool, usize)> {
        let mut result = Vec::new();
        let matches: Vec<_> = self.word_tokenize_pattern.find_iter(text).collect();

        for (i, m) in matches.iter().enumerate() {
            let word = m.as_str().to_string();
            let byte_pos = m.start();
            let mut spaces_after = 0u8;
            let mut has_newline = false;

            let start_pos = m.end();
            let end_pos = if i + 1 < matches.len() {
                matches[i + 1].start()
            } else {
                text.len()
            };

            if start_pos < end_pos {
                let between = &text[start_pos..end_pos];
                has_newline = between.contains('\n');
                if !has_newline {
                    spaces_after = between.chars().filter(|&c| c == ' ').count().min(255) as u8;
                }
            }

            result.push((word, spaces_after, has_newline, byte_pos));
        }

        result
    }
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new(r"\S+").unwrap()
    }
}

// ─── FreqDist ────────────────────────────────────────────────────────────────

/// Frequency distribution for counting items.
#[derive(Debug, Clone)]
pub struct FreqDist<T: Eq + std::hash::Hash> {
    counts: AHashMap<T, usize>,
    total: usize,
}

impl<T: Eq + std::hash::Hash> FreqDist<T> {
    pub fn new() -> Self {
        Self {
            counts: AHashMap::new(),
            total: 0,
        }
    }

    pub fn add(&mut self, item: T) {
        *self.counts.entry(item).or_insert(0) += 1;
        self.total += 1;
    }

    pub fn add_count(&mut self, item: T, count: usize) {
        *self.counts.entry(item).or_insert(0) += count;
        self.total += count;
    }

    pub fn get(&self, item: &T) -> usize {
        self.counts.get(item).copied().unwrap_or(0)
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn frequency(&self, item: &T) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.get(item) as f64 / self.total as f64
        }
    }

    pub fn most_common(&self) -> Vec<(&T, usize)> {
        let mut items: Vec<_> = self.counts.iter().map(|(k, &v)| (k, v)).collect();
        items.sort_by_key(|item| std::cmp::Reverse(item.1));
        items
    }

    pub fn prune(&mut self, min_count: usize) {
        self.counts.retain(|_, &mut count| count >= min_count);
        self.total = self.counts.values().sum();
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

impl<T: Eq + std::hash::Hash> Default for FreqDist<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_iter() {
        let items = vec![1, 2, 3, 4];
        let pairs: Vec<_> = pair_iter(items).collect();
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0], (1, Some(2)));
        assert_eq!(pairs[1], (2, Some(3)));
        assert_eq!(pairs[2], (3, Some(4)));
        assert_eq!(pairs[3], (4, None));
    }

    #[test]
    fn test_freq_dist() {
        let mut dist = FreqDist::new();
        dist.add("hello");
        dist.add("world");
        dist.add("hello");
        assert_eq!(dist.get(&"hello"), 2);
        assert_eq!(dist.get(&"world"), 1);
        assert_eq!(dist.total(), 3);
        assert!((dist.frequency(&"hello") - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_text_preprocessor() {
        let preprocessor = TextPreprocessor::default();
        let tokens = preprocessor.word_tokenize("Hello, world! How are you?");
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], "Hello,");
        assert_eq!(tokens[1], "world!");
    }

    #[test]
    fn test_spacing() {
        let preprocessor = TextPreprocessor::default();
        let result = preprocessor.word_tokenize_with_spacing("hello  world");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1, 2); // 2 spaces after "hello"
    }
}
