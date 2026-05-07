//! Algorithm-by-name dispatch shared between PyO3 bindings and the
//! ranking module.
//!
//! The string keys mirror the public Python API (`compare_batch`'s
//! `algorithm` argument) so callers can pick a metric with one name and
//! reuse it across one-shot scoring, batch scoring, and top-k ranking.

use crate::core::algorithms::edit::{
    DamerauLevenshtein, Hamming, Jaro, JaroWinkler, Levenshtein, Osa, SorensenDice,
};
use crate::core::algorithms::ngram::{
    NgramCosine, NgramJaccard, NgramOverlap, TokenJaccard, TokenNgramCosine, TokenNgramJaccard,
    TokenNgramOverlap,
};
use crate::core::algorithms::phonetic::{Metaphone, Soundex};
use crate::core::algorithms::sequence::{Lcs, LongestCommonSubstring};
use crate::core::algorithms::traits::{DistanceOutput, StringDistance};

/// Knobs that some metrics consume. `n` is the n-gram size, `lowercase`
/// applies to token-level metrics, `prefix_weight` is the Jaro-Winkler
/// prefix scaling factor. Metrics that don't consume a knob ignore it.
#[derive(Debug, Clone, Copy)]
pub struct MetricConfig {
    pub n: usize,
    pub lowercase: bool,
    pub prefix_weight: f64,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            n: 2,
            lowercase: false,
            prefix_weight: 0.1,
        }
    }
}

/// Compute a `DistanceOutput` for the named algorithm. Returns the
/// algorithm name back as the error string when the key is unknown so
/// callers can surface that to users without re-deriving it.
pub fn compute_distance_output(
    algorithm: &str,
    a: &str,
    b: &str,
    cfg: &MetricConfig,
) -> Result<DistanceOutput, String> {
    let MetricConfig {
        n,
        lowercase,
        prefix_weight,
    } = *cfg;
    let result = match algorithm {
        "levenshtein" => Levenshtein.distance(a, b),
        "damerau" | "damerau-levenshtein" => DamerauLevenshtein.distance(a, b),
        "osa" => Osa.distance(a, b),
        "hamming" => Hamming.distance(a, b),
        "jaro" => Jaro.distance(a, b),
        "jaro-winkler" => JaroWinkler { prefix_weight }.distance(a, b),
        "dice" | "sorensen-dice" => SorensenDice.distance(a, b),
        "soundex" => Soundex.distance(a, b),
        "metaphone" => Metaphone.distance(a, b),
        "lcs" => Lcs.distance(a, b),
        "longest-common-substring" => LongestCommonSubstring.distance(a, b),
        "ngram-jaccard" => NgramJaccard { n }.distance(a, b),
        "ngram-cosine" => NgramCosine { n }.distance(a, b),
        "ngram-overlap" => NgramOverlap { n }.distance(a, b),
        "token-jaccard" => TokenJaccard { lowercase }.distance(a, b),
        "token-ngram-jaccard" => TokenNgramJaccard { n, lowercase }.distance(a, b),
        "token-ngram-cosine" => TokenNgramCosine { n, lowercase }.distance(a, b),
        "token-ngram-overlap" => TokenNgramOverlap { n, lowercase }.distance(a, b),
        _ => return Err(format!("unknown algorithm: {algorithm}")),
    };
    result.map_err(|e| e.to_string())
}

/// Convenience: just the similarity scalar for the named algorithm.
#[inline]
pub fn compute_similarity(
    algorithm: &str,
    a: &str,
    b: &str,
    cfg: &MetricConfig,
) -> Result<f64, String> {
    compute_distance_output(algorithm, a, b, cfg).map(|o| o.similarity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_levenshtein() {
        let cfg = MetricConfig::default();
        let r = compute_distance_output("levenshtein", "kitten", "sitting", &cfg).unwrap();
        assert_eq!(r.distance, 3.0);
    }

    #[test]
    fn dispatch_jaro_winkler_prefix_weight() {
        let cfg = MetricConfig {
            prefix_weight: 0.2,
            ..Default::default()
        };
        let high = compute_similarity("jaro-winkler", "martha", "marhta", &cfg).unwrap();
        let cfg0 = MetricConfig {
            prefix_weight: 0.0,
            ..Default::default()
        };
        let low = compute_similarity("jaro-winkler", "martha", "marhta", &cfg0).unwrap();
        assert!(high >= low);
    }

    #[test]
    fn dispatch_token_jaccard_lowercase() {
        let cfg_cs = MetricConfig {
            lowercase: false,
            ..Default::default()
        };
        let cfg_ci = MetricConfig {
            lowercase: true,
            ..Default::default()
        };
        let cs = compute_similarity("token-jaccard", "Hello", "hello", &cfg_cs).unwrap();
        let ci = compute_similarity("token-jaccard", "Hello", "hello", &cfg_ci).unwrap();
        assert_eq!(cs, 0.0);
        assert_eq!(ci, 1.0);
    }

    #[test]
    fn dispatch_unknown_algorithm() {
        let cfg = MetricConfig::default();
        let err = compute_distance_output("not-a-real-metric", "a", "b", &cfg).unwrap_err();
        assert!(err.contains("not-a-real-metric"));
    }

    #[test]
    fn dispatch_damerau_alias() {
        let cfg = MetricConfig::default();
        let a = compute_similarity("damerau", "ab", "ba", &cfg).unwrap();
        let b = compute_similarity("damerau-levenshtein", "ab", "ba", &cfg).unwrap();
        assert_eq!(a, b);
    }
}
