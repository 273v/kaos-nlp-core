//! Edit distance algorithms: Levenshtein, Damerau-Levenshtein, OSA, Hamming, Jaro, Jaro-Winkler.
//!
//! Uses `strsim` for proven implementations, wrapped in our trait interface.

use crate::core::algorithms::traits::{
    DistanceError, DistanceOutput, DistanceResult, StringDistance,
};

/// Classic Levenshtein edit distance (insert, delete, substitute).
#[derive(Debug, Clone, Default)]
pub struct Levenshtein;

impl StringDistance for Levenshtein {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let d = strsim::levenshtein(a, b) as f64;
        let max = a.chars().count().max(b.chars().count()) as f64;
        Ok(DistanceOutput::new(d, max))
    }
}

/// Damerau-Levenshtein distance (insert, delete, substitute, transpose).
#[derive(Debug, Clone, Default)]
pub struct DamerauLevenshtein;

impl StringDistance for DamerauLevenshtein {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let d = strsim::damerau_levenshtein(a, b) as f64;
        let max = a.chars().count().max(b.chars().count()) as f64;
        Ok(DistanceOutput::new(d, max))
    }
}

/// Optimal String Alignment distance (restricted Damerau-Levenshtein —
/// no substring may be edited more than once).
#[derive(Debug, Clone, Default)]
pub struct Osa;

impl StringDistance for Osa {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let d = strsim::osa_distance(a, b) as f64;
        let max = a.chars().count().max(b.chars().count()) as f64;
        Ok(DistanceOutput::new(d, max))
    }
}

/// Hamming distance — number of positions where characters differ.
/// Requires strings of equal length.
#[derive(Debug, Clone, Default)]
pub struct Hamming;

impl StringDistance for Hamming {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        match strsim::hamming(a, b) {
            Ok(d) => {
                let max = a.chars().count() as f64;
                Ok(DistanceOutput::new(d as f64, max))
            }
            Err(_) => Err(DistanceError::UnequalLength(
                a.chars().count(),
                b.chars().count(),
            )),
        }
    }
}

/// Jaro similarity (primarily for short strings and names).
#[derive(Debug, Clone, Default)]
pub struct Jaro;

impl StringDistance for Jaro {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let sim = strsim::jaro(a, b);
        Ok(DistanceOutput::from_similarity(sim))
    }
}

/// Jaro-Winkler similarity — Jaro with a prefix bonus.
#[derive(Debug, Clone)]
pub struct JaroWinkler {
    /// Prefix scaling factor, typically 0.1. Must be in [0.0, 0.25].
    pub prefix_weight: f64,
}

impl Default for JaroWinkler {
    fn default() -> Self {
        Self { prefix_weight: 0.1 }
    }
}

impl StringDistance for JaroWinkler {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        if !(0.0..=0.25).contains(&self.prefix_weight) {
            return Err(DistanceError::ParameterOutOfRange(format!(
                "prefix_weight must be in [0.0, 0.25], got {}",
                self.prefix_weight
            )));
        }
        let sim = strsim::jaro_winkler(a, b);
        Ok(DistanceOutput::from_similarity(sim))
    }
}

/// Sorensen-Dice coefficient over bigrams.
#[derive(Debug, Clone, Default)]
pub struct SorensenDice;

impl StringDistance for SorensenDice {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let sim = strsim::sorensen_dice(a, b);
        Ok(DistanceOutput::from_similarity(sim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein() {
        let lev = Levenshtein;
        let r = lev.distance("kitten", "sitting").unwrap();
        assert_eq!(r.distance, 3.0);
        assert!((r.normalized - 3.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_damerau_levenshtein() {
        let dl = DamerauLevenshtein;
        // "ab" -> "ba" is one transposition
        let r = dl.distance("ab", "ba").unwrap();
        assert_eq!(r.distance, 1.0);
    }

    #[test]
    fn test_hamming_equal_length() {
        let h = Hamming;
        let r = h.distance("karolin", "kathrin").unwrap();
        assert_eq!(r.distance, 3.0);
    }

    #[test]
    fn test_hamming_unequal_length() {
        let h = Hamming;
        assert!(h.distance("abc", "ab").is_err());
    }

    #[test]
    fn test_jaro() {
        let j = Jaro;
        let r = j.distance("martha", "marhta").unwrap();
        assert!(r.similarity > 0.94);
    }

    #[test]
    fn test_jaro_winkler() {
        let jw = JaroWinkler::default();
        let r = jw.distance("martha", "marhta").unwrap();
        assert!(r.similarity > 0.96);
    }

    #[test]
    fn test_sorensen_dice() {
        let sd = SorensenDice;
        let r = sd.distance("night", "nacht").unwrap();
        assert!(r.similarity > 0.0 && r.similarity < 1.0);
    }

    #[test]
    fn test_identical_strings() {
        let lev = Levenshtein;
        let r = lev.distance("hello", "hello").unwrap();
        assert_eq!(r.distance, 0.0);
        assert_eq!(r.similarity, 1.0);
    }

    #[test]
    fn test_empty_strings() {
        let lev = Levenshtein;
        let r = lev.distance("", "").unwrap();
        assert_eq!(r.distance, 0.0);
        assert_eq!(r.similarity, 1.0); // 0/0 case => 0 normalized => 1.0 similarity
    }
}
