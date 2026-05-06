//! Unified sentence boundary decision engine.
//!
//! Ported from nupunkt-rs/src/decision.rs.

use smallvec::smallvec;

use super::parameters::PunktParameters;
use super::statistics::dunning_log_likelihood;
use super::tokens::PunktToken;
use super::types::{
    DecisionFactor, FactorType, FactorVec, ORTHO_BEG_LC, ORTHO_LC, ORTHO_MID_UC, ORTHO_UC,
};

/// Configuration for inference-time adjustments.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Precision/recall balance (0.0 = max recall, 0.5 = balanced, 1.0 = max precision).
    pub precision_recall_balance: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            precision_recall_balance: 0.5,
        }
    }
}

/// Result of a sentence boundary decision.
#[derive(Debug, Clone)]
pub struct BoundaryDecision {
    pub should_break: bool,
    pub confidence: f64,
    pub factors: FactorVec,
    pub primary_reason: String,
}

/// Unified sentence boundary decider.
pub struct SentenceBoundaryDecider<'a> {
    params: &'a PunktParameters,
    config: &'a InferenceConfig,
}

impl<'a> SentenceBoundaryDecider<'a> {
    pub fn new(params: &'a PunktParameters, config: &'a InferenceConfig) -> Self {
        Self { params, config }
    }

    /// Decide if there should be a sentence break after this token.
    pub fn decide(&self, token: &PunktToken, next_token: Option<&PunktToken>) -> BoundaryDecision {
        // Semicolon with newline — strong sentence break signal
        if token.semicolon_final && token.has_newline_after {
            return BoundaryDecision {
                should_break: true,
                confidence: 0.9,
                factors: smallvec![DecisionFactor {
                    factor_type: FactorType::Whitespace,
                    weight: 0.9,
                    description: "Semicolon followed by newline".to_string(),
                }],
                primary_reason: "Semicolon with newline".to_string(),
            };
        }

        // No sentence-ending punctuation → no break
        if !token.sentence_end_punct {
            return BoundaryDecision {
                should_break: false,
                confidence: 1.0,
                factors: smallvec![],
                primary_reason: "No sentence-ending punctuation".to_string(),
            };
        }

        // Ellipsis → no break
        if token.ellipsis {
            return BoundaryDecision {
                should_break: false,
                confidence: 0.9,
                factors: smallvec![DecisionFactor {
                    factor_type: FactorType::Abbreviation,
                    weight: -0.9,
                    description: "Ellipsis".to_string(),
                }],
                primary_reason: "Ellipsis".to_string(),
            };
        }

        let mut factors = smallvec![];
        let mut break_evidence = 0.0;

        let type_for_abbrev = if token.period_final {
            token.type_no_period()
        } else {
            token.type_no_sentence_punct()
        };

        // Factor 1: Abbreviation scoring (only for periods)
        if token.period_final {
            break_evidence += self.evaluate_abbreviation(&type_for_abbrev, &mut factors);
        }

        // Factor 2: Collocation evidence
        if let Some(next) = next_token {
            break_evidence += self.evaluate_collocation(token, next, &mut factors);
            // Factor 3: Capitalization
            break_evidence += self.evaluate_capitalization(next, &mut factors);
            // Factor 4: Sentence starter
            break_evidence += self.evaluate_sentence_starter(next, &mut factors);
        } else {
            // End of text — always break
            factors.push(DecisionFactor {
                factor_type: FactorType::EndOfText,
                weight: 1.0,
                description: "End of text".to_string(),
            });
            return BoundaryDecision {
                should_break: true,
                confidence: 1.0,
                factors,
                primary_reason: "End of text".to_string(),
            };
        }

        // Factor 5: Orthographic heuristics
        if let Some(next) = next_token {
            break_evidence += self.evaluate_orthographic(token, next, &mut factors);
        }

        // Factor 6: Whitespace signals
        let is_provided_abbrev = if token.period_final {
            self.params
                .is_provided_abbreviation(&token.type_no_period())
        } else {
            false
        };

        let whitespace_evidence = self.evaluate_whitespace(token, &mut factors);

        let is_true_paragraph_break = next_token.is_some_and(|n| n.parastart);

        if is_provided_abbrev && whitespace_evidence > 0.0 {
            if is_true_paragraph_break {
                break_evidence += whitespace_evidence;
            } else {
                break_evidence += whitespace_evidence * 0.1;
            }
        } else {
            break_evidence += whitespace_evidence;
        }

        // Final decision
        let evidence_strength = break_evidence.abs();
        let confidence = (0.5 + evidence_strength * 0.3).min(1.0);
        let pr = self.config.precision_recall_balance;
        let threshold = self.params.decision_weights.break_threshold(pr);
        let should_break = break_evidence >= threshold;

        let primary_reason = factors
            .iter()
            .max_by(|a, b| {
                a.weight
                    .abs()
                    .partial_cmp(&b.weight.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|f| f.description.clone())
            .unwrap_or_else(|| "No strong evidence".to_string());

        BoundaryDecision {
            should_break,
            confidence,
            factors,
            primary_reason,
        }
    }

    fn evaluate_abbreviation(&self, type_no_period: &str, factors: &mut FactorVec) -> f64 {
        let pr = self.config.precision_recall_balance;

        if let Some(abbrev_type) = self.params.get_abbreviation_type(type_no_period) {
            let weight = abbrev_type.get_weight(pr, &self.params.decision_weights);
            let description = match abbrev_type {
                super::parameters::AbbreviationType::Provided { source, .. } => {
                    format!("Provided abbreviation (from {})", source)
                }
                super::parameters::AbbreviationType::Learned {
                    confidence, score, ..
                } => {
                    format!(
                        "Learned abbreviation (conf={:.2}, score={:.1})",
                        confidence, score
                    )
                }
            };
            factors.push(DecisionFactor {
                factor_type: FactorType::Abbreviation,
                weight,
                description,
            });
            return weight;
        }

        if let Some(stats) = self.params.get_token_stats(type_no_period) {
            if stats.count_with_period > 0 || stats.count_without_period > 0 {
                let score = dunning_log_likelihood(
                    (stats.count_with_period + stats.count_without_period) as usize,
                    self.params.total_period_tokens as usize,
                    stats.count_with_period as usize,
                    self.params.total_tokens as usize,
                );
                let normalized = (score / 100.0).clamp(-1.0, 1.0);
                let weight = -normalized * (0.1 + 0.5 * pr);
                factors.push(DecisionFactor {
                    factor_type: FactorType::Score,
                    weight,
                    description: format!("Abbrev score: {:.1}", score),
                });
                return weight;
            }
        }

        0.0
    }

    fn evaluate_collocation(
        &self,
        token: &PunktToken,
        next_token: &PunktToken,
        factors: &mut FactorVec,
    ) -> f64 {
        let pr = self.config.precision_recall_balance;
        let type1 = token.type_no_period();
        let type2 = next_token.type_no_sentperiod();

        if self.params.is_collocation(&type1, &type2) {
            let weight = self.params.decision_weights.colloc_weight(pr);
            factors.push(DecisionFactor {
                factor_type: FactorType::Collocation,
                weight,
                description: format!("Known collocation: {} + {}", type1, type2),
            });
            return weight;
        }

        if let Some(stats) = self.params.get_token_stats(&type1) {
            if let Some(&count) = stats.collocation_counts.get(&type2) {
                if count > 0 {
                    let normalized_freq = (count as f64).ln() / 10.0;
                    let weight =
                        -normalized_freq * self.params.decision_weights.colloc_weight(pr).abs();
                    factors.push(DecisionFactor {
                        factor_type: FactorType::Collocation,
                        weight,
                        description: format!("Statistical collocation ({} occurrences)", count),
                    });
                    return weight;
                }
            }
        }

        0.0
    }

    fn evaluate_capitalization(&self, next_token: &PunktToken, factors: &mut FactorVec) -> f64 {
        let pr = self.config.precision_recall_balance;

        if next_token.first_upper() {
            let weight = self.params.decision_weights.capital_weight(pr);
            factors.push(DecisionFactor {
                factor_type: FactorType::Capitalization,
                weight,
                description: "Next word capitalized".to_string(),
            });
            weight
        } else if next_token.first_lower() {
            let weight = self.params.decision_weights.lowercase_next_weight(pr);
            factors.push(DecisionFactor {
                factor_type: FactorType::Capitalization,
                weight,
                description: "Next word lowercase".to_string(),
            });
            weight
        } else {
            0.0
        }
    }

    fn evaluate_sentence_starter(&self, next_token: &PunktToken, factors: &mut FactorVec) -> f64 {
        let pr = self.config.precision_recall_balance;
        let next_type = next_token.type_no_sentperiod();

        if self.params.is_sent_starter(&next_type) {
            let weight = self.params.decision_weights.starter_weight(pr);
            factors.push(DecisionFactor {
                factor_type: FactorType::SentenceStarter,
                weight,
                description: "Known sentence starter".to_string(),
            });
            return weight;
        }

        if let Some(stats) = self.params.get_token_stats(&next_type) {
            if stats.count_as_starter > 0 && stats.count_without_period > 0 {
                let starter_ratio =
                    stats.count_as_starter as f64 / stats.count_without_period as f64;
                if starter_ratio > 0.5 {
                    let multiplier = self.params.decision_weights.starter_ratio_multiplier(pr);
                    let weight = starter_ratio * multiplier;
                    factors.push(DecisionFactor {
                        factor_type: FactorType::SentenceStarter,
                        weight,
                        description: format!(
                            "Often starts sentences ({:.0}%)",
                            starter_ratio * 100.0
                        ),
                    });
                    return weight;
                }
            }
        }

        0.0
    }

    fn evaluate_orthographic(
        &self,
        token: &PunktToken,
        next_token: &PunktToken,
        factors: &mut FactorVec,
    ) -> f64 {
        let pr = self.config.precision_recall_balance;
        let type1 = token.type_no_period();
        let ortho_context = self.params.get_ortho_context(&type1);

        if next_token.first_upper()
            && (ortho_context & ORTHO_LC) != 0
            && (ortho_context & ORTHO_MID_UC) == 0
        {
            let weight = self.params.decision_weights.ortho_positive_weight(pr);
            factors.push(DecisionFactor {
                factor_type: FactorType::Consistency,
                weight,
                description: "Usually followed by lowercase, now uppercase".to_string(),
            });
            return weight;
        }

        if next_token.first_lower()
            && ((ortho_context & ORTHO_UC) != 0 || (ortho_context & ORTHO_BEG_LC) == 0)
        {
            let weight = self.params.decision_weights.ortho_negative_weight(pr);
            factors.push(DecisionFactor {
                factor_type: FactorType::Consistency,
                weight,
                description: "Usually followed by uppercase, now lowercase".to_string(),
            });
            return weight;
        }

        0.0
    }

    fn evaluate_whitespace(&self, token: &PunktToken, factors: &mut FactorVec) -> f64 {
        let pr = self.config.precision_recall_balance;

        if token.spaces_after >= 2 && token.has_newline_after {
            let weight = 1.5;
            factors.push(DecisionFactor {
                factor_type: FactorType::Whitespace,
                weight,
                description: "Paragraph break (double newline)".to_string(),
            });
            return weight;
        }

        if token.spaces_after >= 2 {
            let weight = if token.sentence_end_punct {
                0.4 + 0.1 * pr
            } else {
                0.2 + 0.1 * pr
            };
            factors.push(DecisionFactor {
                factor_type: FactorType::Whitespace,
                weight,
                description: format!("Double space ({} spaces)", token.spaces_after),
            });
            return weight;
        }

        if token.has_newline_after {
            let weight = if token.sentence_end_punct || token.semicolon_final {
                0.3 + 0.1 * pr
            } else {
                0.1 + 0.05 * pr
            };
            factors.push(DecisionFactor {
                factor_type: FactorType::Whitespace,
                weight,
                description: "Newline after token".to_string(),
            });
            return weight;
        }

        if token.parastart && token.spaces_after >= 2 {
            let weight = 1.2;
            factors.push(DecisionFactor {
                factor_type: FactorType::Whitespace,
                weight,
                description: "True paragraph break".to_string(),
            });
            return weight;
        }

        if token.parastart {
            factors.push(DecisionFactor {
                factor_type: FactorType::Whitespace,
                weight: 0.6,
                description: "Paragraph start".to_string(),
            });
            return 0.6;
        }

        if token.linestart && !token.parastart {
            let weight = 0.3 + 0.2 * pr;
            factors.push(DecisionFactor {
                factor_type: FactorType::Whitespace,
                weight,
                description: "Line start".to_string(),
            });
            return weight;
        }

        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::super::parameters::PunktParameters;
    use super::*;

    #[test]
    fn test_provided_abbreviation_no_break() {
        let mut params = PunktParameters::new();
        params.add_provided_abbreviation("v", "test");

        for pr in [0.1, 0.3, 0.5] {
            let config = InferenceConfig {
                precision_recall_balance: pr,
            };
            let decider = SentenceBoundaryDecider::new(&params, &config);

            let token = PunktToken::new("v.", false, false);
            let next = PunktToken::new("Jones", false, false);
            let decision = decider.decide(&token, Some(&next));
            assert!(!decision.should_break, "v. broke at PR={}", pr);
        }
    }

    #[test]
    fn test_end_of_text_break() {
        let params = PunktParameters::new();
        let config = InferenceConfig::default();
        let decider = SentenceBoundaryDecider::new(&params, &config);

        let token = PunktToken::new("end.", false, false);
        let decision = decider.decide(&token, None);
        assert!(decision.should_break);
    }

    #[test]
    fn test_no_punct_no_break() {
        let params = PunktParameters::new();
        let config = InferenceConfig::default();
        let decider = SentenceBoundaryDecider::new(&params, &config);

        let token = PunktToken::new("hello", false, false);
        let next = PunktToken::new("world", false, false);
        let decision = decider.decide(&token, Some(&next));
        assert!(!decision.should_break);
    }

    #[test]
    fn test_ellipsis_no_break() {
        let params = PunktParameters::new();
        let config = InferenceConfig::default();
        let decider = SentenceBoundaryDecider::new(&params, &config);

        let mut token = PunktToken::new("...", false, false);
        token.ellipsis = true;
        let next = PunktToken::new("continued", false, false);
        let decision = decider.decide(&token, Some(&next));
        assert!(!decision.should_break);
    }
}
