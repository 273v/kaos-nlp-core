//! Statistical functions for the Punkt algorithm: log-likelihood scoring, abbreviation/collocation/starter detection.
//!
//! Ported from nupunkt-rs/src/statistics.rs.

use std::f64::consts::E;

/// Calculate Dunning log-likelihood for abbreviation detection.
/// Higher scores indicate more likely abbreviations.
pub fn dunning_log_likelihood(count_a: usize, count_b: usize, count_ab: usize, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }

    let n = n as f64;
    let count_a = count_a as f64;
    let count_b = count_b as f64;
    let count_ab = count_ab as f64;

    let p1 = count_b / n;
    let p2: f64 = 0.99;

    if count_a == 0.0 || count_ab > count_a {
        return 0.0;
    }

    let null_hypo = if count_ab > 0.0 {
        count_ab * (p1 + 1e-8).ln()
    } else {
        0.0
    } + if count_a > count_ab {
        (count_a - count_ab) * (1.0 - p1 + 1e-8).ln()
    } else {
        0.0
    };

    let alt_hypo = if count_ab > 0.0 {
        count_ab * p2.ln()
    } else {
        0.0
    } + if count_a > count_ab {
        (count_a - count_ab) * (1.0 - p2).ln()
    } else {
        0.0
    };

    let ll = -2.0 * (null_hypo - alt_hypo);

    if ll.is_nan() || ll.is_infinite() {
        return 0.0;
    }

    ll * 1.5
}

/// Calculate log-likelihood ratio for collocations.
pub fn collocation_log_likelihood(
    count_a: usize,
    count_b: usize,
    count_ab: usize,
    n: usize,
) -> f64 {
    if n == 0 || count_a == 0 {
        return 0.0;
    }

    let n = n as f64;
    let count_a = count_a as f64;
    let count_b = count_b as f64;
    let count_ab = count_ab as f64;

    let p = count_b / n;
    let p1 = count_ab / count_a;
    let p2 = if n > count_a {
        (count_b - count_ab) / (n - count_a)
    } else {
        0.0
    };

    let mut summand1 = 0.0;
    let mut summand2 = 0.0;
    let mut summand3 = 0.0;
    let mut summand4 = 0.0;

    if p > 0.0 && p < 1.0 {
        summand1 = count_ab * p.ln() + (count_a - count_ab) * (1.0 - p).ln();
        summand2 =
            (count_b - count_ab) * p.ln() + (n - count_a - count_b + count_ab) * (1.0 - p).ln();
    }

    if p1 > 0.0 && p1 < 1.0 && count_a != count_ab {
        summand3 = count_ab * p1.ln() + (count_a - count_ab) * (1.0 - p1).ln();
    }

    if p2 > 0.0 && p2 < 1.0 && count_b != count_ab {
        summand4 =
            (count_b - count_ab) * p2.ln() + (n - count_a - count_b + count_ab) * (1.0 - p2).ln();
    }

    -2.0 * (summand1 + summand2 - summand3 - summand4)
}

/// Calculate abbreviation score with additional factors.
pub fn calculate_abbreviation_score(
    candidate: &str,
    count_with_period: usize,
    count_without_period: usize,
    total_period_tokens: usize,
    total_tokens: usize,
) -> f64 {
    let log_likelihood = dunning_log_likelihood(
        count_with_period + count_without_period,
        total_period_tokens,
        count_with_period,
        total_tokens,
    );

    let num_periods = candidate.chars().filter(|&c| c == '.').count();
    let num_nonperiods = candidate.chars().filter(|&c| c != '.').count();

    let f_length = E.powf(-(num_nonperiods as f64));
    let f_periods = 1.0 + num_periods as f64;

    let f_penalty = if candidate.len() <= 3 {
        1.0
    } else if count_without_period > 0 {
        (num_nonperiods as f64).powf(-(count_without_period as f64 * 0.5))
    } else {
        1.0
    };

    let total_count = count_with_period + count_without_period;
    let consistency_boost = if total_count > 0 {
        count_with_period as f64 / total_count as f64
    } else {
        0.0
    };

    log_likelihood * f_length * f_periods * f_penalty * (1.0 + consistency_boost)
}

// ─── Scorers ─────────────────────────────────────────────────────────────────

/// Score-based abbreviation detection.
pub struct AbbreviationScorer {
    threshold: f64,
    consistency_threshold: f64,
}

impl AbbreviationScorer {
    pub fn new(threshold: f64, _boost_factor: f64, consistency_threshold: f64) -> Self {
        Self {
            threshold,
            consistency_threshold,
        }
    }

    pub fn is_abbreviation(
        &self,
        candidate: &str,
        count_with_period: usize,
        count_without_period: usize,
        total_period_tokens: usize,
        total_tokens: usize,
    ) -> bool {
        let score = calculate_abbreviation_score(
            candidate,
            count_with_period,
            count_without_period,
            total_period_tokens,
            total_tokens,
        );

        let total_count = count_with_period + count_without_period;
        let consistency = if total_count > 0 {
            count_with_period as f64 / total_count as f64
        } else {
            0.0
        };

        if consistency >= self.consistency_threshold {
            score >= self.threshold * 0.5
        } else {
            score >= self.threshold
        }
    }
}

impl Default for AbbreviationScorer {
    fn default() -> Self {
        Self::new(0.1, 1.5, 0.25)
    }
}

/// Score-based collocation detection.
pub struct CollocationScorer {
    threshold: f64,
    min_freq: usize,
}

impl CollocationScorer {
    pub fn new(threshold: f64, min_freq: usize) -> Self {
        Self {
            threshold,
            min_freq,
        }
    }

    pub fn is_collocation(
        &self,
        count_first: usize,
        count_second: usize,
        count_together: usize,
        total_tokens: usize,
    ) -> bool {
        if count_together < self.min_freq {
            return false;
        }

        let score =
            collocation_log_likelihood(count_first, count_second, count_together, total_tokens);

        if total_tokens > 0 && count_first > 0 && count_together > 0 {
            let ratio1 = total_tokens as f64 / count_first as f64;
            let ratio2 = count_second as f64 / count_together as f64;
            score >= self.threshold && ratio1 > ratio2
        } else {
            false
        }
    }
}

impl Default for CollocationScorer {
    fn default() -> Self {
        Self::new(5.0, 5)
    }
}

/// Score-based sentence starter detection.
pub struct SentenceStarterScorer {
    threshold: f64,
    min_freq: usize,
}

impl SentenceStarterScorer {
    pub fn new(threshold: f64, min_freq: usize) -> Self {
        Self {
            threshold,
            min_freq,
        }
    }

    pub fn is_sentence_starter(
        &self,
        sentbreak_count: usize,
        token_count: usize,
        starter_count: usize,
        total_tokens: usize,
    ) -> bool {
        if starter_count < self.min_freq || token_count < starter_count {
            return false;
        }

        let score =
            collocation_log_likelihood(sentbreak_count, token_count, starter_count, total_tokens);

        if total_tokens > 0 && sentbreak_count > 0 && starter_count > 0 {
            let ratio1 = total_tokens as f64 / sentbreak_count as f64;
            let ratio2 = token_count as f64 / starter_count as f64;
            score >= self.threshold && ratio1 > ratio2
        } else {
            false
        }
    }
}

impl Default for SentenceStarterScorer {
    fn default() -> Self {
        Self::new(25.0, 5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dunning_log_likelihood_basic() {
        let score = dunning_log_likelihood(50, 48, 48, 10000);
        assert!(score > 0.0);
    }

    #[test]
    fn test_dunning_no_nan_edge_cases() {
        let cases = vec![
            (0, 0, 0, 100),
            (10, 10, 10, 100),
            (10, 0, 0, 100),
            (10, 100, 10, 100),
            (1, 1, 1, 1),
            (100, 50, 100, 100),
            (0, 100, 0, 100),
            (100, 0, 0, 100),
        ];
        for (a, b, ab, n) in cases {
            let r = dunning_log_likelihood(a, b, ab, n);
            assert!(!r.is_nan(), "NaN for ({}, {}, {}, {})", a, b, ab, n);
            assert!(r.is_finite(), "Inf for ({}, {}, {}, {})", a, b, ab, n);
        }
    }

    #[test]
    fn test_collocation_log_likelihood_basic() {
        let score = collocation_log_likelihood(100, 80, 20, 10000);
        assert!(score >= 0.0);
    }

    #[test]
    fn test_collocation_no_nan() {
        let cases = vec![
            (0, 0, 0, 100),
            (10, 10, 10, 100),
            (10, 10, 0, 100),
            (100, 100, 50, 1000),
            (0, 100, 0, 100),
        ];
        for (a, b, ab, n) in cases {
            let r = collocation_log_likelihood(a, b, ab, n);
            assert!(!r.is_nan(), "NaN for colloc({}, {}, {}, {})", a, b, ab, n);
            assert!(r.is_finite(), "Inf for colloc({}, {}, {}, {})", a, b, ab, n);
        }
    }

    #[test]
    fn test_abbreviation_scorer() {
        let scorer = AbbreviationScorer::default();
        let is_abbrev = scorer.is_abbreviation("the", 5, 495, 1000, 10000);
        assert!(!is_abbrev);
    }

    #[test]
    fn test_collocation_scorer() {
        let scorer = CollocationScorer::new(5.0, 3);

        // Strong collocation: "New York" appears together frequently
        let is_colloc = scorer.is_collocation(200, 150, 100, 10000);
        assert!(
            is_colloc,
            "Strong co-occurrence should be detected as collocation"
        );

        // Below min_freq threshold
        let is_colloc = scorer.is_collocation(200, 150, 2, 10000);
        assert!(!is_colloc, "Below min_freq should not be a collocation");

        // Rare words, no real association
        let is_colloc = scorer.is_collocation(5000, 5000, 5, 10000);
        assert!(
            !is_colloc,
            "Common words with rare co-occurrence should not be a collocation"
        );
    }

    #[test]
    fn test_sentence_starter_scorer() {
        let scorer = SentenceStarterScorer::new(25.0, 5);

        // "The" appears often as a sentence starter
        let is_starter = scorer.is_sentence_starter(500, 1000, 200, 50000);
        assert!(is_starter, "Frequent sentence starter should be detected");

        // Below min_freq
        let is_starter = scorer.is_sentence_starter(500, 1000, 2, 50000);
        assert!(!is_starter, "Below min_freq should not be a starter");

        // token_count < starter_count should return false
        let is_starter = scorer.is_sentence_starter(500, 10, 20, 50000);
        assert!(
            !is_starter,
            "Invalid counts (token_count < starter_count) should return false"
        );
    }
}
