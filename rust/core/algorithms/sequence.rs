//! Sequence-based distance algorithms: LCS, Longest Common Substring.

use crate::core::algorithms::traits::{DistanceOutput, DistanceResult, StringDistance};

/// Longest Common Subsequence distance.
///
/// Distance = max(|a|, |b|) - LCS_length.
#[derive(Debug, Clone, Default)]
pub struct Lcs;

impl Lcs {
    /// Compute the length of the longest common subsequence.
    pub fn lcs_length(&self, a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        if m == 0 || n == 0 {
            return 0;
        }

        // Space-optimized: two rows instead of full matrix.
        let mut prev = vec![0usize; n + 1];
        let mut curr = vec![0usize; n + 1];

        for i in 1..=m {
            for j in 1..=n {
                if a_chars[i - 1] == b_chars[j - 1] {
                    curr[j] = prev[j - 1] + 1;
                } else {
                    curr[j] = prev[j].max(curr[j - 1]);
                }
            }
            std::mem::swap(&mut prev, &mut curr);
            curr.fill(0);
        }
        prev[n]
    }
}

impl StringDistance for Lcs {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let lcs_len = self.lcs_length(a, b);
        let max_len = a.chars().count().max(b.chars().count());
        let d = (max_len - lcs_len) as f64;
        Ok(DistanceOutput::new(d, max_len as f64))
    }
}

/// Longest Common Substring (contiguous) similarity.
///
/// Similarity = longest_common_substring_length / max(|a|, |b|).
#[derive(Debug, Clone, Default)]
pub struct LongestCommonSubstring;

impl LongestCommonSubstring {
    /// Find the length of the longest common contiguous substring.
    pub fn lcss_length(&self, a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        if m == 0 || n == 0 {
            return 0;
        }

        let mut max_len = 0usize;
        let mut prev = vec![0usize; n + 1];
        let mut curr = vec![0usize; n + 1];

        for i in 1..=m {
            for j in 1..=n {
                if a_chars[i - 1] == b_chars[j - 1] {
                    curr[j] = prev[j - 1] + 1;
                    max_len = max_len.max(curr[j]);
                } else {
                    curr[j] = 0;
                }
            }
            std::mem::swap(&mut prev, &mut curr);
            curr.fill(0);
        }
        max_len
    }
}

impl StringDistance for LongestCommonSubstring {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let lcss = self.lcss_length(a, b);
        let max_len = a.chars().count().max(b.chars().count());
        let sim = if max_len > 0 {
            lcss as f64 / max_len as f64
        } else {
            1.0
        };
        Ok(DistanceOutput::from_similarity(sim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs_distance() {
        let lcs = Lcs;
        // LCS("kitten", "sitting") = "ittn" (length 4)
        // distance = max(6,7) - 4 = 3
        let r = lcs.distance("kitten", "sitting").unwrap();
        assert_eq!(r.distance, 3.0);
    }

    #[test]
    fn test_lcs_identical() {
        let lcs = Lcs;
        let r = lcs.distance("hello", "hello").unwrap();
        assert_eq!(r.distance, 0.0);
        assert_eq!(r.similarity, 1.0);
    }

    #[test]
    fn test_lcss() {
        let lcss = LongestCommonSubstring;
        let r = lcss.distance("abcdef", "xbcdey").unwrap();
        // Longest common substring is "bcde" (4), max len = 6
        assert!((r.similarity - 4.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_lcss_no_common() {
        let lcss = LongestCommonSubstring;
        let r = lcss.distance("abc", "xyz").unwrap();
        assert_eq!(r.similarity, 0.0);
    }
}
