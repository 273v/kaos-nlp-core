//! Regex-based matching with compiled pattern caching.

use regex::Regex;
use serde::{Deserialize, Serialize};

/// A regex match with capture group information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegexMatch {
    /// Byte offset of the full match start.
    pub start: usize,
    /// Byte offset of the full match end (exclusive).
    pub end: usize,
    /// The full matched text.
    pub text: String,
    /// Capture groups (index 0 = full match, 1.. = groups).
    pub groups: Vec<Option<String>>,
}

/// A compiled regex matcher.
pub struct RegexMatcher {
    pattern: Regex,
    pattern_str: String,
}

impl RegexMatcher {
    /// Compile a regex pattern.
    pub fn new(pattern: &str) -> Result<Self, String> {
        let re = Regex::new(pattern).map_err(|e| format!("invalid regex: {e}"))?;
        Ok(Self {
            pattern: re,
            pattern_str: pattern.to_string(),
        })
    }

    /// Find all non-overlapping matches.
    pub fn find_all(&self, haystack: &str) -> Vec<RegexMatch> {
        self.pattern
            .captures_iter(haystack)
            .map(|cap| {
                let full = cap.get(0).unwrap();
                let groups: Vec<Option<String>> = cap
                    .iter()
                    .map(|g| g.map(|m| m.as_str().to_string()))
                    .collect();
                RegexMatch {
                    start: full.start(),
                    end: full.end(),
                    text: full.as_str().to_string(),
                    groups,
                }
            })
            .collect()
    }

    /// Find the first match.
    pub fn find_first(&self, haystack: &str) -> Option<RegexMatch> {
        self.pattern.captures(haystack).map(|cap| {
            let full = cap.get(0).unwrap();
            let groups: Vec<Option<String>> = cap
                .iter()
                .map(|g| g.map(|m| m.as_str().to_string()))
                .collect();
            RegexMatch {
                start: full.start(),
                end: full.end(),
                text: full.as_str().to_string(),
                groups,
            }
        })
    }

    /// Check if the pattern matches anywhere.
    pub fn is_match(&self, haystack: &str) -> bool {
        self.pattern.is_match(haystack)
    }

    /// Count non-overlapping matches.
    pub fn count(&self, haystack: &str) -> usize {
        self.pattern.find_iter(haystack).count()
    }

    /// Replace all matches.
    pub fn replace_all(&self, haystack: &str, replacement: &str) -> String {
        self.pattern.replace_all(haystack, replacement).into_owned()
    }

    /// Split text by the pattern.
    pub fn split<'a>(&self, haystack: &'a str) -> Vec<&'a str> {
        self.pattern.split(haystack).collect()
    }

    /// Return the pattern string.
    pub fn pattern_str(&self) -> &str {
        &self.pattern_str
    }
}

/// A compiled set of regex patterns for matching any-of in a single pass.
pub struct RegexSetMatcher {
    set: regex::RegexSet,
    patterns: Vec<String>,
}

impl RegexSetMatcher {
    /// Build a regex set from multiple patterns.
    pub fn new(patterns: &[&str]) -> Result<Self, String> {
        let set = regex::RegexSet::new(patterns).map_err(|e| format!("invalid regex set: {e}"))?;
        Ok(Self {
            set,
            patterns: patterns.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Return indices of all patterns that match.
    pub fn matching_patterns(&self, haystack: &str) -> Vec<usize> {
        self.set.matches(haystack).iter().collect()
    }

    /// Check if any pattern matches.
    pub fn is_match(&self, haystack: &str) -> bool {
        self.set.is_match(haystack)
    }

    /// Return the number of patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_all() {
        let re = RegexMatcher::new(r"\b\d+\b").unwrap();
        let matches = re.find_all("I have 3 cats and 42 dogs");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].text, "3");
        assert_eq!(matches[1].text, "42");
    }

    #[test]
    fn test_capture_groups() {
        let re = RegexMatcher::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();
        let m = re.find_first("Date: 2026-03-24").unwrap();
        assert_eq!(m.groups[1], Some("2026".into()));
        assert_eq!(m.groups[2], Some("03".into()));
        assert_eq!(m.groups[3], Some("24".into()));
    }

    #[test]
    fn test_replace() {
        let re = RegexMatcher::new(r"\d+").unwrap();
        assert_eq!(re.replace_all("a1b2c3", "X"), "aXbXcX");
    }

    #[test]
    fn test_split() {
        let re = RegexMatcher::new(r"[,;]\s*").unwrap();
        let parts = re.split("a, b; c, d");
        assert_eq!(parts, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_regex_set() {
        let set = RegexSetMatcher::new(&[r"\d+", r"[a-z]+", r"[A-Z]+"]).unwrap();
        let indices = set.matching_patterns("Hello 42 world");
        assert!(indices.contains(&0)); // digits
        assert!(indices.contains(&1)); // lowercase
        assert!(indices.contains(&2)); // uppercase
    }

    #[test]
    fn test_invalid_regex() {
        assert!(RegexMatcher::new(r"[invalid").is_err());
    }

    // --- Unicode byte offset correctness ---

    #[test]
    fn test_unicode_cafe_byte_offsets() {
        let re = RegexMatcher::new(r"café").unwrap();
        let text = "Le café est bon, le café est chaud.";
        let matches = re.find_all(text);
        assert_eq!(matches.len(), 2);
        for m in &matches {
            assert_eq!(&text[m.start..m.end], "café");
        }
    }

    #[test]
    fn test_unicode_cjk_byte_offsets() {
        let re = RegexMatcher::new(r"東京").unwrap();
        let text = "東京は大きい。東京タワー。";
        let matches = re.find_all(text);
        assert_eq!(matches.len(), 2);
        for m in &matches {
            assert_eq!(&text[m.start..m.end], "東京");
        }
    }

    #[test]
    fn test_unicode_after_emoji() {
        let re = RegexMatcher::new(r"world").unwrap();
        let text = "😀😀 world";
        let m = re.find_first(text).unwrap();
        assert_eq!(&text[m.start..m.end], "world");
    }
}
