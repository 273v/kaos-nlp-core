//! Punkt trainer: learns sentence boundary parameters from text.
//!
//! Ported from nupunkt-rs/src/trainers/trainer.rs (PyO3 stripped).

use super::decision::{InferenceConfig, SentenceBoundaryDecider};
use super::parameters::PunktParameters;
use super::statistics::{
    calculate_abbreviation_score, AbbreviationScorer, CollocationScorer, SentenceStarterScorer,
};
use super::tokens::PunktToken;
use super::types::{
    ScoringConfig, ORTHO_BEG_LC, ORTHO_BEG_UC, ORTHO_MID_LC, ORTHO_MID_UC, ORTHO_UNK_LC,
    ORTHO_UNK_UC,
};
use super::utils::{pair_iter, FreqDist, TextPreprocessor};

/// Punkt trainer for learning from text.
pub struct PunktTrainer {
    config: ScoringConfig,
    params: PunktParameters,
    type_fdist: FreqDist<String>,
    collocation_fdist: FreqDist<(String, String)>,
    sent_starter_fdist: FreqDist<String>,
    num_period_toks: usize,
    sentbreak_count: usize,
    preprocessor: TextPreprocessor,
    common_abbrevs: Vec<String>,
}

impl PunktTrainer {
    pub fn new() -> Self {
        Self::with_config(ScoringConfig::default())
    }

    pub fn with_config(config: ScoringConfig) -> Self {
        Self {
            config,
            params: PunktParameters::new(),
            type_fdist: FreqDist::new(),
            collocation_fdist: FreqDist::new(),
            sent_starter_fdist: FreqDist::new(),
            num_period_toks: 0,
            sentbreak_count: 0,
            preprocessor: TextPreprocessor::default(),
            common_abbrevs: vec!["...".to_string()],
        }
    }

    /// Load abbreviations from a JSON file (list of strings).
    pub fn load_abbreviations_from_json(
        &mut self,
        path: &str,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let abbreviations: Vec<String> = serde_json::from_str(&contents)?;
        let count = abbreviations.len();
        let source = path.to_string();
        for abbrev in abbreviations {
            self.params
                .add_provided_abbreviation(abbrev, source.clone());
        }
        Ok(count)
    }

    /// Add abbreviations directly (marked as provided).
    pub fn add_abbreviations(&mut self, abbreviations: Vec<String>) {
        for abbrev in abbreviations {
            self.params.add_provided_abbreviation(abbrev, "direct");
        }
    }

    /// Train on text and return the learned parameters.
    pub fn train(
        &mut self,
        text: &str,
        verbose: bool,
    ) -> Result<PunktParameters, Box<dyn std::error::Error>> {
        if verbose {
            eprintln!("Starting training on {} characters...", text.len());
        }

        let tokens = self.word_tokenize_with_context(text);
        if verbose {
            eprintln!("Found {} tokens", tokens.len());
        }

        self.collect_frequencies(&tokens, verbose);
        self.find_abbreviations(verbose);

        let annotated = self.annotate_tokens(tokens);
        self.collect_ortho_data(&annotated);
        self.find_collocations_and_starters(&annotated, verbose);

        self.params.freeze();

        if verbose {
            eprintln!("Training complete!");
            eprintln!("  Abbreviations: {}", self.params.abbrev_types.len());
            eprintln!("  Collocations: {}", self.params.collocations.len());
            eprintln!("  Sentence starters: {}", self.params.sent_starters.len());
        }

        Ok(self.params.clone())
    }

    /// Train incrementally on chunks of text (for streaming).
    pub fn train_incremental(&mut self, text: &str) -> Result<(), Box<dyn std::error::Error>> {
        let tokens = self.word_tokenize_with_context(text);
        self.collect_frequencies(&tokens, false);

        let annotated = self.annotate_tokens(tokens);
        self.collect_ortho_data(&annotated);

        for i in 0..annotated.len() {
            let token = &annotated[i];
            if token.sentbreak {
                self.sentbreak_count += 1;
            }

            if i > 0 {
                let prev = &annotated[i - 1];
                if prev.period_final && !prev.sentbreak {
                    let has_other_evidence =
                        token.first_lower() || (prev.parastart && token.first_upper());
                    if has_other_evidence || !prev.abbr {
                        let prev_type = prev.type_no_period();
                        let curr_type = token.type_no_sentperiod();
                        self.collocation_fdist.add_count((prev_type, curr_type), 1);
                    }
                }
            }

            if token.sentbreak || token.parastart {
                self.sent_starter_fdist.add(token.type_no_sentperiod());
            }
        }

        Ok(())
    }

    /// Finalize training after all chunks have been processed.
    pub fn finalize_training(
        &mut self,
        verbose: bool,
    ) -> Result<PunktParameters, Box<dyn std::error::Error>> {
        self.find_abbreviations(verbose);

        let total = self.type_fdist.total();
        let min_freq = (self.config.min_colloc_rate * total as f64) as usize;
        let colloc_scorer =
            CollocationScorer::new(self.config.collocation_threshold, min_freq.max(1));

        for ((first, second), count) in self.collocation_fdist.most_common() {
            let first_count =
                self.type_fdist.get(first) + self.type_fdist.get(&format!("{}.", first));
            let second_count =
                self.type_fdist.get(second) + self.type_fdist.get(&format!("{}.", second));
            if colloc_scorer.is_collocation(first_count, second_count, count, total) {
                self.params.add_collocation(first, second);
            }
        }

        let _min_freq = (self.config.min_starter_rate * self.type_fdist.total() as f64) as usize;
        for (token_type, count) in self.sent_starter_fdist.most_common() {
            let rate = count as f64 / total as f64;
            if rate < self.config.min_starter_rate {
                continue;
            }
            if self.config.require_alpha_starters {
                if !token_type.chars().any(|c| c.is_alphabetic()) {
                    continue;
                }
                if token_type.chars().all(|c| !c.is_alphanumeric()) {
                    continue;
                }
                if token_type.starts_with("##number##") {
                    continue;
                }
            }

            let type_count = self.type_fdist.get(token_type);
            if type_count > 0 && self.sentbreak_count > 0 {
                let score = super::statistics::dunning_log_likelihood(
                    count + type_count,
                    count,
                    type_count,
                    total,
                );
                if score > self.config.sent_starter_threshold {
                    self.params.add_sent_starter(token_type);
                }
            }
        }

        self.params.freeze();

        if verbose {
            eprintln!("Training complete!");
            eprintln!("  Abbreviations: {}", self.params.abbrev_types.len());
            eprintln!("  Collocations: {}", self.params.collocations.len());
            eprintln!("  Sentence starters: {}", self.params.sent_starters.len());
        }

        Ok(self.params.clone())
    }

    /// Get the current parameters (for inspection during training).
    pub fn parameters(&self) -> &PunktParameters {
        &self.params
    }

    fn word_tokenize_with_context(&self, text: &str) -> Vec<PunktToken> {
        let mut tokens = Vec::new();
        let mut parastart = true;

        let lines: Vec<&str> = text.lines().collect();

        for (line_idx, line) in lines.iter().enumerate() {
            let mut linestart = true;
            let words_with_spacing = self.preprocessor.word_tokenize_with_spacing(line);

            for (word_idx, (word, spaces_after, _, byte_pos)) in
                words_with_spacing.iter().enumerate()
            {
                let mut token = PunktToken::new(word.clone(), parastart, linestart);
                token.byte_position = Some(*byte_pos);

                if token.is_ellipsis() {
                    token.ellipsis = true;
                }

                if word_idx == words_with_spacing.len() - 1 && line_idx < lines.len() - 1 {
                    token.has_newline_after = true;
                    token.spaces_after = 0;
                    if line_idx + 1 < lines.len() && lines[line_idx + 1].trim().is_empty() {
                        token.spaces_after = 2;
                    }
                } else {
                    token.spaces_after = *spaces_after;
                    token.has_newline_after = false;
                }

                tokens.push(token);
                parastart = false;
                linestart = false;
            }

            if line.trim().is_empty() {
                parastart = true;
            }
        }

        tokens
    }

    fn collect_frequencies(&mut self, tokens: &[PunktToken], _verbose: bool) {
        self.params.total_tokens = tokens.len() as u32;

        for i in 0..tokens.len() {
            let token = &tokens[i];
            self.type_fdist.add(token.token_type.clone());
            let type_no_period = token.type_no_period();

            if token.period_final {
                self.num_period_toks += 1;
                self.params.total_period_tokens += 1;
                self.params.update_token_stats(&type_no_period, |stats| {
                    stats.count_with_period += 1;
                });
            } else {
                self.params.update_token_stats(&type_no_period, |stats| {
                    stats.count_without_period += 1;
                });
            }

            if token.period_final && i + 1 < tokens.len() {
                let next_type = tokens[i + 1].type_no_sentperiod();
                self.params.update_token_stats(&type_no_period, |stats| {
                    *stats.collocation_counts.entry(next_type).or_insert(0) += 1;
                });
            }
        }

        for abbrev in &self.common_abbrevs {
            self.params.add_abbreviation(abbrev.clone());
        }
    }

    fn find_abbreviations(&mut self, _verbose: bool) {
        let scorer = AbbreviationScorer::new(
            self.config.abbrev_threshold,
            self.config.abbrev_boost,
            self.config.abbrev_consistency,
        );

        let total = self.type_fdist.total();

        for (token_type, _) in self.type_fdist.most_common() {
            if token_type.len() > self.config.max_abbrev_length
                || token_type.starts_with("##number##")
            {
                continue;
            }

            if token_type.ends_with('.') {
                let candidate = &token_type[..token_type.len() - 1];

                if candidate.len() > 10
                    || (candidate.len() > 3 && candidate.chars().all(|c| c.is_lowercase()))
                {
                    continue;
                }

                if !candidate.chars().any(|c| c.is_alphabetic()) {
                    continue;
                }

                let count_with = self.type_fdist.get(token_type);
                let count_without = self.type_fdist.get(&candidate.to_string());

                if count_without > count_with * 2 {
                    continue;
                }

                if scorer.is_abbreviation(
                    candidate,
                    count_with,
                    count_without,
                    self.num_period_toks,
                    total,
                ) {
                    let score = calculate_abbreviation_score(
                        candidate,
                        count_with,
                        count_without,
                        self.num_period_toks,
                        total,
                    );
                    self.params.add_learned_abbreviation(
                        candidate.to_string(),
                        score,
                        count_with,
                        count_without,
                    );
                }
            }
        }
    }

    fn annotate_tokens(&mut self, mut tokens: Vec<PunktToken>) -> Vec<PunktToken> {
        let training_config = InferenceConfig {
            precision_recall_balance: 0.5,
        };
        let decider = SentenceBoundaryDecider::new(&self.params, &training_config);
        let len = tokens.len();

        for i in 0..len {
            if tokens[i].period_final {
                let type_no_period = tokens[i].type_no_period();
                if self.params.is_abbreviation(&type_no_period) {
                    tokens[i].abbr = true;
                }
            }

            if tokens[i].ellipsis {
                tokens[i].sentbreak = false;
                continue;
            }

            let next_token = if i + 1 < len {
                Some(&tokens[i + 1])
            } else {
                None
            };
            let decision = decider.decide(&tokens[i], next_token);
            tokens[i].sentbreak = decision.should_break;

            if decision.should_break {
                self.sentbreak_count += 1;
            }
        }

        tokens
    }

    fn collect_ortho_data(&mut self, tokens: &[PunktToken]) {
        let mut context = "initial";

        for token in tokens {
            if token.parastart && context != "unknown" {
                context = "initial";
            }

            let ortho_key = token.type_no_sentperiod();
            let flag = match (context, token.first_upper(), token.first_lower()) {
                ("initial", true, _) => ORTHO_BEG_UC,
                ("initial", _, true) => ORTHO_BEG_LC,
                ("internal", true, _) => ORTHO_MID_UC,
                ("internal", _, true) => ORTHO_MID_LC,
                ("unknown", true, _) => ORTHO_UNK_UC,
                ("unknown", _, true) => ORTHO_UNK_LC,
                _ => 0,
            };

            if flag > 0 {
                self.params.add_ortho_context(ortho_key, flag);
            }

            if context == "initial" && token.is_non_punct() {
                let type_no_period = token.type_no_period();
                self.params.update_token_stats(&type_no_period, |stats| {
                    stats.count_as_starter += 1;
                });
            }

            if token.sentbreak {
                context = "initial";
            } else if !token.parastart && token.is_non_punct() {
                context = "internal";
            }
        }
    }

    fn find_collocations_and_starters(&mut self, tokens: &[PunktToken], _verbose: bool) {
        for (token1, next) in pair_iter(tokens.iter()) {
            if let Some(token2) = next {
                if !token1.period_final {
                    continue;
                }

                if token1.period_final && !token1.sentbreak {
                    let has_other_evidence =
                        token2.first_lower() || (token1.parastart && token2.first_upper());
                    if has_other_evidence || !token1.abbr {
                        let type1 = token1.type_no_period();
                        let type2 = token2.type_no_sentperiod();
                        self.collocation_fdist.add((type1, type2));
                    }
                }

                if token1.sentbreak {
                    self.sent_starter_fdist.add(token2.token_type.clone());
                }
            }
        }

        let min_freq = (self.config.min_colloc_rate * self.type_fdist.total() as f64) as usize;
        let colloc_scorer =
            CollocationScorer::new(self.config.collocation_threshold, min_freq.max(1));
        let total = self.type_fdist.total();

        for ((type1, type2), count) in self.collocation_fdist.most_common() {
            let count1 = self.type_fdist.get(type1) + self.type_fdist.get(&format!("{}.", type1));
            let count2 = self.type_fdist.get(type2) + self.type_fdist.get(&format!("{}.", type2));
            if colloc_scorer.is_collocation(count1, count2, count, total) {
                self.params.add_collocation(type1, type2);
            }
        }

        let starter_scorer = SentenceStarterScorer::new(
            self.config.sent_starter_threshold,
            (self.config.min_starter_rate * total as f64) as usize,
        );

        for (starter, count) in self.sent_starter_fdist.most_common() {
            let type_count = self.type_fdist.get(starter);
            if starter_scorer.is_sentence_starter(self.sentbreak_count, type_count, count, total) {
                self.params.add_sent_starter(starter);
            }
        }
    }
}

impl Default for PunktTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_training() {
        let mut trainer = PunktTrainer::new();
        let text = "Hello world. How are you? I am fine. Thank you.";
        let params = trainer.train(text, false).unwrap();
        assert!(params.total_tokens > 0);
    }

    #[test]
    fn test_provided_abbreviations_preserved() {
        let mut trainer = PunktTrainer::new();
        trainer.params.add_provided_abbreviation("v", "test");

        let text = "Smith v. Jones established precedent. This is important.";
        let params = trainer.train(text, false).unwrap();

        assert!(params.is_abbreviation("v"));
        assert!(params.is_provided_abbreviation("v"));
    }

    #[test]
    fn test_incremental_training() {
        let mut trainer = PunktTrainer::new();

        trainer
            .train_incremental("First chunk. With sentences.")
            .unwrap();
        trainer
            .train_incremental("Second chunk. More sentences.")
            .unwrap();

        let params = trainer.finalize_training(false).unwrap();
        assert!(params.total_tokens > 0);
    }
}
