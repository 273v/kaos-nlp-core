//! Exact and approximate substring search.
//!
//! Uses `stringzilla` for SIMD-accelerated exact substring matching,
//! falling back to stdlib for features stringzilla doesn't cover.

use serde::{Deserialize, Serialize};

/// A match found in a haystack.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubstringMatch {
    /// Byte offset of the match start in the haystack.
    pub start: usize,
    /// Byte offset of the match end (exclusive).
    pub end: usize,
    /// The matched text.
    pub text: String,
}

/// Find all occurrences of `needle` in `haystack` (exact, case-sensitive).
///
/// Uses stringzilla's SIMD-accelerated search when available.
pub fn find_all(haystack: &str, needle: &str) -> Vec<SubstringMatch> {
    if needle.is_empty() || haystack.is_empty() || needle.len() > haystack.len() {
        return vec![];
    }

    let mut matches = Vec::new();
    let haystack_bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    let mut offset = 0;

    loop {
        let remaining = &haystack_bytes[offset..];
        // Use stringzilla's sz::find for SIMD-accelerated search.
        match stringzilla::sz::find(remaining, needle_bytes) {
            Some(pos) => {
                let abs_start = offset + pos;
                let abs_end = abs_start + needle.len();
                matches.push(SubstringMatch {
                    start: abs_start,
                    end: abs_end,
                    text: needle.to_string(),
                });
                offset = abs_start + 1; // advance past start for overlapping matches
            }
            None => break,
        }
    }
    matches
}

/// Find the first occurrence of `needle` in `haystack`.
pub fn find_first(haystack: &str, needle: &str) -> Option<SubstringMatch> {
    if needle.is_empty() || haystack.is_empty() {
        return None;
    }
    stringzilla::sz::find(haystack.as_bytes(), needle.as_bytes()).map(|pos| SubstringMatch {
        start: pos,
        end: pos + needle.len(),
        text: needle.to_string(),
    })
}

/// Find the last occurrence of `needle` in `haystack`.
pub fn find_last(haystack: &str, needle: &str) -> Option<SubstringMatch> {
    if needle.is_empty() || haystack.is_empty() {
        return None;
    }
    stringzilla::sz::rfind(haystack.as_bytes(), needle.as_bytes()).map(|pos| SubstringMatch {
        start: pos,
        end: pos + needle.len(),
        text: needle.to_string(),
    })
}

/// Count occurrences of `needle` in `haystack` (non-overlapping).
pub fn count(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() || haystack.is_empty() {
        return 0;
    }
    let haystack_bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    let mut n = 0;
    let mut offset = 0;
    while let Some(pos) = stringzilla::sz::find(&haystack_bytes[offset..], needle_bytes) {
        n += 1;
        offset += pos + needle.len();
    }
    n
}

/// Case-insensitive find all (ASCII fast path, Unicode fallback).
pub fn find_all_case_insensitive(haystack: &str, needle: &str) -> Vec<SubstringMatch> {
    if needle.is_empty() || haystack.is_empty() {
        return vec![];
    }

    // ASCII fast path
    if haystack.is_ascii() && needle.is_ascii() {
        let h_lower = haystack.to_ascii_lowercase();
        let n_lower = needle.to_ascii_lowercase();
        let byte_matches = find_all(&h_lower, &n_lower);
        return byte_matches
            .into_iter()
            .map(|m| SubstringMatch {
                start: m.start,
                end: m.end,
                text: haystack[m.start..m.end].to_string(),
            })
            .collect();
    }

    // Unicode fallback
    let h_lower: String = haystack.chars().flat_map(|c| c.to_lowercase()).collect();
    let n_lower: String = needle.chars().flat_map(|c| c.to_lowercase()).collect();
    let byte_matches = find_all(&h_lower, &n_lower);

    // Map byte offsets from lowered string back to original.
    // This works when lowering preserves byte length (most common case).
    // For edge cases (e.g., ß -> ss), we fall back to approximate mapping.
    byte_matches
        .into_iter()
        .filter_map(|m| {
            if m.end <= haystack.len() {
                Some(SubstringMatch {
                    start: m.start,
                    end: m.end,
                    text: haystack.get(m.start..m.end)?.to_string(),
                })
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_all() {
        let matches = find_all("abcabcabc", "abc");
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[1].start, 3);
        assert_eq!(matches[2].start, 6);
    }

    #[test]
    fn test_find_first() {
        let m = find_first("hello world hello", "hello");
        assert!(m.is_some());
        assert_eq!(m.unwrap().start, 0);
    }

    #[test]
    fn test_find_last() {
        let m = find_last("hello world hello", "hello");
        assert!(m.is_some());
        assert_eq!(m.unwrap().start, 12);
    }

    #[test]
    fn test_count() {
        assert_eq!(count("aaaa", "aa"), 2); // non-overlapping
    }

    #[test]
    fn test_find_all_overlapping() {
        let matches = find_all("aaaa", "aa");
        assert_eq!(matches.len(), 3); // overlapping: positions 0, 1, 2
    }

    #[test]
    fn test_case_insensitive() {
        let matches = find_all_case_insensitive("Hello HELLO hello", "hello");
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_empty_needle() {
        assert!(find_all("abc", "").is_empty());
        assert!(find_first("abc", "").is_none());
    }

    #[test]
    fn test_no_match() {
        assert!(find_all("abc", "xyz").is_empty());
    }

    // --- Unicode byte offset correctness ---
    // These verify that Rust byte offsets are correct for multi-byte chars.
    // The PyO3 bindings convert these to char offsets for Python.

    #[test]
    fn test_unicode_cafe_byte_offsets() {
        // "café" = 5 bytes (c=1, a=1, f=1, é=2)
        let text = "café café";
        let matches = find_all(text, "café");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[0].end, 5); // byte offset
        assert_eq!(&text[matches[0].start..matches[0].end], "café");
        assert_eq!(&text[matches[1].start..matches[1].end], "café");
    }

    #[test]
    fn test_unicode_cjk_byte_offsets() {
        // "東" = 3 bytes, "京" = 3 bytes
        let text = "東京は東京";
        let matches = find_all(text, "東京");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[0].end, 6); // 2 CJK chars = 6 bytes
        assert_eq!(&text[matches[0].start..matches[0].end], "東京");
        assert_eq!(&text[matches[1].start..matches[1].end], "東京");
    }

    #[test]
    fn test_unicode_after_multibyte() {
        // Find ASCII text after multi-byte prefix
        let text = "東京 hello";
        let m = find_first(text, "hello").unwrap();
        assert_eq!(m.start, 7); // byte offset: 3+3+1 = 7
        assert_eq!(&text[m.start..m.end], "hello");
    }

    #[test]
    fn test_unicode_emoji_byte_offsets() {
        let text = "😀 hello";
        let m = find_first(text, "hello").unwrap();
        assert_eq!(m.start, 5); // emoji=4 bytes + space=1
        assert_eq!(&text[m.start..m.end], "hello");
    }
}
