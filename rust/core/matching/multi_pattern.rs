//! Multi-pattern matching using Aho-Corasick.
//!
//! Searches a haystack for all occurrences of any pattern from a set,
//! in a single linear-time pass.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use serde::{Deserialize, Serialize};

/// A match from multi-pattern search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatternMatch {
    /// Index of the matched pattern in the original pattern list.
    pub pattern_index: usize,
    /// Byte offset of the match start.
    pub start: usize,
    /// Byte offset of the match end (exclusive).
    pub end: usize,
    /// The matched text from the haystack.
    pub text: String,
}

/// Match semantics for overlapping / priority.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum MultiPatternMatchKind {
    /// Report the first pattern that matches at each position (default).
    #[default]
    LeftmostFirst,
    /// Report the longest match at each position.
    LeftmostLongest,
}

/// A compiled multi-pattern matcher backed by Aho-Corasick.
pub struct MultiPatternMatcher {
    automaton: AhoCorasick,
    patterns: Vec<String>,
}

impl MultiPatternMatcher {
    /// Build a matcher from a list of patterns.
    pub fn new(patterns: &[&str], match_kind: MultiPatternMatchKind) -> Result<Self, String> {
        if patterns.is_empty() {
            return Err("pattern list must not be empty".into());
        }
        let ak_match_kind = match match_kind {
            MultiPatternMatchKind::LeftmostFirst => MatchKind::LeftmostFirst,
            MultiPatternMatchKind::LeftmostLongest => MatchKind::LeftmostLongest,
        };
        let automaton = AhoCorasickBuilder::new()
            .match_kind(ak_match_kind)
            .build(patterns)
            .map_err(|e| format!("failed to build Aho-Corasick automaton: {e}"))?;

        Ok(Self {
            automaton,
            patterns: patterns.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Build a case-insensitive matcher.
    pub fn new_case_insensitive(
        patterns: &[&str],
        match_kind: MultiPatternMatchKind,
    ) -> Result<Self, String> {
        if patterns.is_empty() {
            return Err("pattern list must not be empty".into());
        }
        let ak_match_kind = match match_kind {
            MultiPatternMatchKind::LeftmostFirst => MatchKind::LeftmostFirst,
            MultiPatternMatchKind::LeftmostLongest => MatchKind::LeftmostLongest,
        };
        let automaton = AhoCorasickBuilder::new()
            .match_kind(ak_match_kind)
            .ascii_case_insensitive(true)
            .build(patterns)
            .map_err(|e| format!("failed to build Aho-Corasick automaton: {e}"))?;

        Ok(Self {
            automaton,
            patterns: patterns.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Find all non-overlapping matches in the haystack.
    pub fn find_all(&self, haystack: &str) -> Vec<PatternMatch> {
        self.automaton
            .find_iter(haystack)
            .map(|m| PatternMatch {
                pattern_index: m.pattern().as_usize(),
                start: m.start(),
                end: m.end(),
                text: haystack[m.start()..m.end()].to_string(),
            })
            .collect()
    }

    /// Check if any pattern matches.
    pub fn is_match(&self, haystack: &str) -> bool {
        self.automaton.is_match(haystack)
    }

    /// Count total non-overlapping matches.
    pub fn count(&self, haystack: &str) -> usize {
        self.automaton.find_iter(haystack).count()
    }

    /// Replace all matches with corresponding replacements.
    /// `replacements` must have the same length as the original pattern list.
    pub fn replace_all(&self, haystack: &str, replacements: &[&str]) -> Result<String, String> {
        if replacements.len() != self.patterns.len() {
            return Err(format!(
                "replacements length ({}) must match patterns length ({})",
                replacements.len(),
                self.patterns.len()
            ));
        }
        Ok(self.automaton.replace_all(haystack, replacements))
    }

    /// Return the number of patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Return the pattern at the given index.
    pub fn pattern(&self, index: usize) -> Option<&str> {
        self.patterns.get(index).map(|s| s.as_str())
    }
}

/// Convenience: search a haystack for all patterns in one call.
pub fn multi_pattern_search(
    haystack: &str,
    patterns: &[&str],
) -> Result<Vec<PatternMatch>, String> {
    let matcher = MultiPatternMatcher::new(patterns, MultiPatternMatchKind::LeftmostFirst)?;
    Ok(matcher.find_all(haystack))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let matcher = MultiPatternMatcher::new(
            &["he", "she", "his", "hers"],
            MultiPatternMatchKind::LeftmostFirst,
        )
        .unwrap();
        let matches = matcher.find_all("ushers");
        assert!(!matches.is_empty());
        // "she" or "he" should be found
        let texts: Vec<&str> = matches.iter().map(|m| m.text.as_str()).collect();
        assert!(texts.contains(&"she") || texts.contains(&"he"));
    }

    #[test]
    fn test_longest_match() {
        let matcher = MultiPatternMatcher::new(
            &["he", "hello", "hell"],
            MultiPatternMatchKind::LeftmostLongest,
        )
        .unwrap();
        let matches = matcher.find_all("hello world");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "hello");
    }

    #[test]
    fn test_case_insensitive() {
        let matcher = MultiPatternMatcher::new_case_insensitive(
            &["hello"],
            MultiPatternMatchKind::LeftmostFirst,
        )
        .unwrap();
        assert!(matcher.is_match("HELLO WORLD"));
    }

    #[test]
    fn test_replace() {
        let matcher =
            MultiPatternMatcher::new(&["cat", "dog"], MultiPatternMatchKind::LeftmostFirst)
                .unwrap();
        let result = matcher
            .replace_all("I have a cat and a dog", &["CAT", "DOG"])
            .unwrap();
        assert_eq!(result, "I have a CAT and a DOG");
    }

    #[test]
    fn test_count() {
        let matcher =
            MultiPatternMatcher::new(&["the"], MultiPatternMatchKind::LeftmostFirst).unwrap();
        assert_eq!(matcher.count("the cat and the dog and the bird"), 3);
    }

    #[test]
    fn test_convenience() {
        let matches = multi_pattern_search("hello world", &["hello", "world"]).unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_empty_patterns_error() {
        let empty: &[&str] = &[];
        assert!(MultiPatternMatcher::new(empty, MultiPatternMatchKind::LeftmostFirst).is_err());
    }

    // --- Unicode byte offset correctness ---

    #[test]
    fn test_unicode_patterns_byte_offsets() {
        let matcher =
            MultiPatternMatcher::new(&["café", "東京"], MultiPatternMatchKind::LeftmostFirst)
                .unwrap();
        let text = "café 東京";
        let matches = matcher.find_all(text);
        assert_eq!(matches.len(), 2);
        for m in &matches {
            assert_eq!(&text[m.start..m.end], m.text);
        }
    }

    #[test]
    fn test_pattern_after_multibyte() {
        let matcher =
            MultiPatternMatcher::new(&["hello"], MultiPatternMatchKind::LeftmostFirst).unwrap();
        let text = "東京 hello";
        let matches = matcher.find_all(text);
        assert_eq!(matches.len(), 1);
        assert_eq!(&text[matches[0].start..matches[0].end], "hello");
    }

    #[test]
    fn test_emoji_pattern() {
        let matcher =
            MultiPatternMatcher::new(&["😀", "🌍"], MultiPatternMatchKind::LeftmostFirst).unwrap();
        let text = "Hello 😀 world 🌍";
        let matches = matcher.find_all(text);
        assert_eq!(matches.len(), 2);
        for m in &matches {
            assert_eq!(&text[m.start..m.end], m.text);
        }
    }
}
