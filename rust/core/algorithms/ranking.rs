//! Top-k similarity ranking against a candidate list.
//!
//! Given a query and a list of candidate strings, return the most-similar
//! (or least-similar) candidates by a chosen metric. Scoring is parallelised
//! with rayon for large candidate sets and falls back to sequential for
//! small ones to avoid per-call overhead.

use rayon::prelude::*;

use crate::core::algorithms::dispatch::{compute_similarity, MetricConfig};

/// Threshold below which we don't bother spawning rayon tasks. The cutover
/// is conservative — for cheap metrics (e.g. ASCII Levenshtein on short
/// strings) the parallel overhead can match the work itself, and the test
/// suite calls `rank` on tiny lists where determinism matters more than
/// speed.
const PARALLEL_THRESHOLD: usize = 64;

/// Direction of the ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Highest similarity first (most-similar).
    Descending,
    /// Lowest similarity first (least-similar / most-different).
    Ascending,
}

/// Score every choice, optionally filter by threshold, then return up to
/// `k` `(index, score)` pairs ordered per `direction`.
///
/// * `direction == Descending` filters out scores **below** `threshold`.
/// * `direction == Ascending`  filters out scores **above** `threshold`.
/// * Ties are broken by the original `choices` index, so the output is
///   stable across runs and platforms.
/// * NaN scores are dropped (they would otherwise sort unpredictably).
/// * `k == 0` or empty `choices` returns an empty vector.
///
/// Errors from individual metric calls (e.g. Hamming on unequal-length
/// strings) are *not* fatal — the offending candidate is treated as a NaN
/// score and dropped. Use the dispatch layer directly if you need to
/// surface per-pair errors.
pub fn rank(
    query: &str,
    choices: &[String],
    algorithm: &str,
    k: usize,
    direction: Direction,
    threshold: Option<f64>,
    cfg: &MetricConfig,
) -> Result<Vec<(usize, f64)>, String> {
    if k == 0 || choices.is_empty() {
        return Ok(Vec::new());
    }

    // Validate the algorithm key (and configuration knobs like Jaro-Winkler's
    // prefix_weight) once against an empty-string probe. Two empty strings
    // satisfy every length precondition (e.g. Hamming) so this lets us
    // surface unknown-algorithm and bad-parameter errors at the top level
    // without rejecting metrics that error per-pair on length mismatch.
    compute_similarity(algorithm, "", "", cfg)?;

    let scored: Vec<(usize, f64)> = if choices.len() >= PARALLEL_THRESHOLD {
        choices
            .par_iter()
            .enumerate()
            .map(|(i, c)| {
                (
                    i,
                    compute_similarity(algorithm, query, c, cfg).unwrap_or(f64::NAN),
                )
            })
            .collect()
    } else {
        choices
            .iter()
            .enumerate()
            .map(|(i, c)| {
                (
                    i,
                    compute_similarity(algorithm, query, c, cfg).unwrap_or(f64::NAN),
                )
            })
            .collect()
    };

    let descending = direction == Direction::Descending;
    let mut filtered: Vec<(usize, f64)> = scored
        .into_iter()
        .filter(|(_, s)| !s.is_nan())
        .filter(|(_, s)| match threshold {
            Some(t) if descending => *s >= t,
            Some(t) => *s <= t,
            None => true,
        })
        .collect();

    filtered.sort_by(|(ia, sa), (ib, sb)| {
        let primary = if descending {
            sb.partial_cmp(sa).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            sa.partial_cmp(sb).unwrap_or(std::cmp::Ordering::Equal)
        };
        primary.then(ia.cmp(ib))
    });

    filtered.truncate(k);
    Ok(filtered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn picks_most_similar_jaro_winkler() {
        let cfg = MetricConfig::default();
        let choices = s(&["apple", "ample", "orange", "applesauce"]);
        let out = rank(
            "apple",
            &choices,
            "jaro-winkler",
            2,
            Direction::Descending,
            None,
            &cfg,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        // "apple" itself should win.
        assert_eq!(out[0].0, 0);
        assert!((out[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn picks_least_similar() {
        let cfg = MetricConfig::default();
        let choices = s(&["apple", "ample", "completely-different-string"]);
        let out = rank(
            "apple",
            &choices,
            "jaro-winkler",
            1,
            Direction::Ascending,
            None,
            &cfg,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 2);
    }

    #[test]
    fn k_zero_returns_empty() {
        let cfg = MetricConfig::default();
        let choices = s(&["apple", "banana"]);
        let out = rank(
            "apple",
            &choices,
            "jaro-winkler",
            0,
            Direction::Descending,
            None,
            &cfg,
        )
        .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn empty_choices_returns_empty() {
        let cfg = MetricConfig::default();
        let choices: Vec<String> = vec![];
        let out = rank(
            "apple",
            &choices,
            "jaro-winkler",
            5,
            Direction::Descending,
            None,
            &cfg,
        )
        .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn k_larger_than_choices() {
        let cfg = MetricConfig::default();
        let choices = s(&["a", "b"]);
        let out = rank(
            "a",
            &choices,
            "jaro-winkler",
            10,
            Direction::Descending,
            None,
            &cfg,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn threshold_filters_below() {
        let cfg = MetricConfig::default();
        let choices = s(&["apple", "ample", "zzzz"]);
        let out = rank(
            "apple",
            &choices,
            "jaro-winkler",
            10,
            Direction::Descending,
            Some(0.5),
            &cfg,
        )
        .unwrap();
        // "zzzz" should be excluded by the floor.
        assert!(out.iter().all(|(i, _)| *i != 2));
    }

    #[test]
    fn ascending_threshold_filters_above() {
        let cfg = MetricConfig::default();
        let choices = s(&["apple", "ample", "zzzz"]);
        let out = rank(
            "apple",
            &choices,
            "jaro-winkler",
            10,
            Direction::Ascending,
            Some(0.5),
            &cfg,
        )
        .unwrap();
        // "apple" / "ample" sit above the ceiling, only "zzzz" survives.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 2);
    }

    #[test]
    fn ties_broken_by_index() {
        let cfg = MetricConfig::default();
        // Three identical candidates — every score is 1.0; the original
        // order should win the tiebreak.
        let choices = s(&["apple", "apple", "apple"]);
        let out = rank(
            "apple",
            &choices,
            "jaro-winkler",
            3,
            Direction::Descending,
            None,
            &cfg,
        )
        .unwrap();
        assert_eq!(
            out.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn unknown_algorithm_errors() {
        let cfg = MetricConfig::default();
        let choices = s(&["a", "b"]);
        let err = rank("a", &choices, "bogus", 1, Direction::Descending, None, &cfg).unwrap_err();
        assert!(err.contains("bogus"));
    }

    #[test]
    fn hamming_unequal_lengths_treated_as_nan() {
        // Hamming requires equal-length strings; on mismatched candidates
        // the underlying error becomes a NaN score and is dropped.
        let cfg = MetricConfig::default();
        let choices = s(&["abc", "ab", "ax"]);
        let out = rank(
            "ax",
            &choices,
            "hamming",
            10,
            Direction::Descending,
            None,
            &cfg,
        )
        .unwrap();
        // Only the equal-length "abc"? No: query is "ax" len 2; only "ab" and "ax" qualify.
        assert!(out.iter().all(|(i, _)| *i == 1 || *i == 2));
    }

    #[test]
    fn larger_list_uses_parallel_path() {
        // Just exercise the code path — correctness is the point, not perf.
        let cfg = MetricConfig::default();
        let choices: Vec<String> = (0..200).map(|i| format!("item-{i:04}")).collect();
        let out = rank(
            "item-0042",
            &choices,
            "jaro-winkler",
            3,
            Direction::Descending,
            None,
            &cfg,
        )
        .unwrap();
        assert_eq!(out.len(), 3);
        // Self-match should be rank 0.
        assert_eq!(choices[out[0].0], "item-0042");
    }

    #[test]
    fn token_jaccard_lowercase_via_metric_config() {
        let cfg = MetricConfig {
            lowercase: true,
            ..Default::default()
        };
        let choices = s(&[
            "The Quick Brown Fox",
            "Slow green turtle",
            "the quick brown fox",
        ]);
        let out = rank(
            "the quick brown fox",
            &choices,
            "token-jaccard",
            1,
            Direction::Descending,
            None,
            &cfg,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        // First candidate (case-folded) ties with the third — index tiebreak picks index 0.
        assert!((out[0].1 - 1.0).abs() < 1e-9);
        assert_eq!(out[0].0, 0);
    }
}
