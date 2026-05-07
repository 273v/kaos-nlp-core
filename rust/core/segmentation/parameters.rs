//! Parameters for the Punkt algorithm: abbreviation types, decision weights, token stats.
//!
//! Ported from nupunkt-rs/src/parameters.rs (PyO3 stripped).

use ahash::AHashSet;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;

thread_local! {
    static ABBREV_CACHE: RefCell<LruCache<String, bool>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(512).unwrap()));
}

/// Distinguishes between provided (ground truth) and learned abbreviations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AbbreviationType {
    Provided {
        source: String,
        added_at: usize,
    },
    Learned {
        score: f64,
        with_period: usize,
        without_period: usize,
        confidence: f64,
    },
}

impl AbbreviationType {
    pub fn is_provided(&self) -> bool {
        matches!(self, AbbreviationType::Provided { .. })
    }

    pub fn confidence(&self) -> f64 {
        match self {
            AbbreviationType::Provided { .. } => 1.0,
            AbbreviationType::Learned { confidence, .. } => *confidence,
        }
    }

    pub fn get_weight(&self, pr: f64, weights: &DecisionWeights) -> f64 {
        match self {
            AbbreviationType::Provided { .. } => weights.provided_abbrev_weight(pr),
            AbbreviationType::Learned { confidence, .. } => {
                weights.learned_abbrev_weight(pr) * confidence
            }
        }
    }
}

/// Decision weights for sentence boundary detection (hand-tuned, 11 values for PR 0.0–1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionWeights {
    pub provided_abbrev_weights: Vec<f64>,
    pub learned_abbrev_weights: Vec<f64>,
    pub capital_weights: Vec<f64>,
    pub colloc_weights: Vec<f64>,
    pub starter_weights: Vec<f64>,
    pub lowercase_next_weights: Vec<f64>,
    pub starter_ratio_multipliers: Vec<f64>,
    pub ortho_positive_weights: Vec<f64>,
    pub ortho_negative_weights: Vec<f64>,
    pub break_thresholds: Vec<f64>,
}

impl Default for DecisionWeights {
    fn default() -> Self {
        Self {
            provided_abbrev_weights: vec![
                -0.70, -0.72, -0.74, -0.76, -0.78, -0.80, -0.82, -0.84, -0.86, -0.88, -0.90,
            ],
            learned_abbrev_weights: vec![
                -0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4, -0.45, -0.5, -0.55, -0.6,
            ],
            capital_weights: vec![
                0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40,
            ],
            colloc_weights: vec![
                -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.75, -0.8, -0.85, -0.9, -0.95,
            ],
            starter_weights: vec![0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1],
            lowercase_next_weights: vec![
                -0.05, -0.07, -0.09, -0.11, -0.13, -0.15, -0.17, -0.19, -0.21, -0.23, -0.25,
            ],
            starter_ratio_multipliers: vec![
                0.400, 0.388, 0.376, 0.364, 0.352, 0.340, 0.328, 0.316, 0.304, 0.292, 0.280,
            ],
            ortho_positive_weights: vec![
                0.300, 0.291, 0.282, 0.273, 0.264, 0.255, 0.246, 0.237, 0.228, 0.219, 0.210,
            ],
            ortho_negative_weights: vec![
                -0.20, -0.215, -0.23, -0.245, -0.26, -0.275, -0.29, -0.305, -0.32, -0.335, -0.35,
            ],
            break_thresholds: vec![
                -0.30, -0.26, -0.22, -0.18, -0.14, -0.10, -0.06, -0.02, 0.02, 0.06, 0.10,
            ],
        }
    }
}

impl DecisionWeights {
    #[inline(always)]
    pub fn get_weight(&self, weights: &[f64], pr: f64) -> f64 {
        let pr = pr.clamp(0.0, 1.0);
        let index = (pr * 10.0).round() as usize;
        let index = index.min(weights.len() - 1);
        weights[index]
    }

    #[inline]
    pub fn provided_abbrev_weight(&self, pr: f64) -> f64 {
        self.get_weight(&self.provided_abbrev_weights, pr)
    }
    #[inline]
    pub fn learned_abbrev_weight(&self, pr: f64) -> f64 {
        self.get_weight(&self.learned_abbrev_weights, pr)
    }
    #[inline]
    pub fn capital_weight(&self, pr: f64) -> f64 {
        self.get_weight(&self.capital_weights, pr)
    }
    #[inline]
    pub fn colloc_weight(&self, pr: f64) -> f64 {
        self.get_weight(&self.colloc_weights, pr)
    }
    #[inline]
    pub fn starter_weight(&self, pr: f64) -> f64 {
        self.get_weight(&self.starter_weights, pr)
    }
    #[inline]
    pub fn lowercase_next_weight(&self, pr: f64) -> f64 {
        self.get_weight(&self.lowercase_next_weights, pr)
    }
    #[inline]
    pub fn starter_ratio_multiplier(&self, pr: f64) -> f64 {
        self.get_weight(&self.starter_ratio_multipliers, pr)
    }
    #[inline]
    pub fn ortho_positive_weight(&self, pr: f64) -> f64 {
        self.get_weight(&self.ortho_positive_weights, pr)
    }
    #[inline]
    pub fn ortho_negative_weight(&self, pr: f64) -> f64 {
        self.get_weight(&self.ortho_negative_weights, pr)
    }
    #[inline]
    pub fn break_threshold(&self, pr: f64) -> f64 {
        self.get_weight(&self.break_thresholds, pr)
    }
}

/// Statistics for individual tokens.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenStats {
    pub count_with_period: u32,
    pub count_without_period: u32,
    pub count_as_starter: u32,
    pub collocation_counts: HashMap<String, u32>,
}

/// Punkt parameters: the learned model for sentence boundary detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunktParameters {
    pub abbrev_types: HashMap<String, AbbreviationType>,
    pub collocations: HashSet<(String, String)>,
    pub sent_starters: HashSet<String>,
    pub ortho_context: HashMap<String, u32>,
    pub token_stats: HashMap<String, TokenStats>,
    pub total_period_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub decision_weights: DecisionWeights,
    #[serde(skip)]
    frozen_abbrev_types: Option<Arc<AHashSet<String>>>,
    #[serde(skip)]
    frozen_collocations: Option<Arc<AHashSet<(String, String)>>>,
    #[serde(skip)]
    frozen_sent_starters: Option<Arc<AHashSet<String>>>,
}

impl PunktParameters {
    pub fn new() -> Self {
        Self {
            abbrev_types: HashMap::new(),
            collocations: HashSet::new(),
            sent_starters: HashSet::new(),
            ortho_context: HashMap::new(),
            token_stats: HashMap::new(),
            total_period_tokens: 0,
            total_tokens: 0,
            decision_weights: DecisionWeights::default(),
            frozen_abbrev_types: None,
            frozen_collocations: None,
            frozen_sent_starters: None,
        }
    }

    pub fn add_provided_abbreviation(
        &mut self,
        abbrev: impl Into<String>,
        source: impl Into<String>,
    ) {
        let abbrev_str = abbrev.into();
        let clean = if abbrev_str.ends_with('.') {
            abbrev_str[..abbrev_str.len() - 1].to_string()
        } else {
            abbrev_str
        };
        let clean = clean.to_lowercase();

        self.abbrev_types.insert(
            clean,
            AbbreviationType::Provided {
                source: source.into(),
                added_at: self.total_tokens as usize,
            },
        );
        self.frozen_abbrev_types = None;
    }

    pub fn add_learned_abbreviation(
        &mut self,
        abbrev: impl Into<String>,
        score: f64,
        with_period: usize,
        without_period: usize,
    ) {
        let abbrev_str = abbrev.into().to_lowercase();

        if let Some(existing) = self.abbrev_types.get(&abbrev_str) {
            if existing.is_provided() {
                return;
            }
        }

        let total = with_period + without_period;
        let ratio = if total > 0 {
            with_period as f64 / total as f64
        } else {
            0.0
        };
        let confidence = if total < 10 {
            ratio * 0.5
        } else if ratio > 0.9 {
            0.9 + (score / 100.0).min(0.1)
        } else {
            ratio * 0.7
        };

        self.abbrev_types.insert(
            abbrev_str,
            AbbreviationType::Learned {
                score,
                with_period,
                without_period,
                confidence,
            },
        );
        self.frozen_abbrev_types = None;
    }

    pub fn add_abbreviation(&mut self, abbrev: impl Into<String>) {
        self.add_learned_abbreviation(abbrev, 10.0, 10, 1);
    }

    pub fn add_collocation(&mut self, first: impl Into<String>, second: impl Into<String>) {
        self.collocations
            .insert((first.into().to_lowercase(), second.into().to_lowercase()));
        self.frozen_collocations = None;
    }

    pub fn add_sent_starter(&mut self, starter: impl Into<String>) {
        self.sent_starters.insert(starter.into());
        self.frozen_sent_starters = None;
    }

    pub fn add_ortho_context(&mut self, token_type: impl Into<String>, flag: u32) {
        *self.ortho_context.entry(token_type.into()).or_insert(0) |= flag;
    }

    #[inline]
    pub fn is_abbreviation(&self, token: &str) -> bool {
        let normalized = token.to_lowercase();
        ABBREV_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(&cached) = cache.get(&normalized) {
                return cached;
            }
            let result = if let Some(frozen) = &self.frozen_abbrev_types {
                frozen.contains(&normalized)
            } else {
                self.abbrev_types.contains_key(&normalized)
            };
            cache.put(normalized, result);
            result
        })
    }

    #[inline]
    pub fn is_provided_abbreviation(&self, token: &str) -> bool {
        let normalized = token.to_lowercase();
        self.abbrev_types
            .get(&normalized)
            .map(|t| t.is_provided())
            .unwrap_or(false)
    }

    #[inline]
    pub fn get_abbreviation_type(&self, token: &str) -> Option<&AbbreviationType> {
        self.abbrev_types.get(&token.to_lowercase())
    }

    pub fn clear_caches() {
        ABBREV_CACHE.with(|cache| cache.borrow_mut().clear());
    }

    #[inline]
    pub fn is_collocation(&self, first: &str, second: &str) -> bool {
        let pair = (first.to_lowercase(), second.to_lowercase());
        if let Some(frozen) = &self.frozen_collocations {
            frozen.contains(&pair)
        } else {
            self.collocations.contains(&pair)
        }
    }

    pub fn is_sent_starter(&self, token: &str) -> bool {
        if let Some(frozen) = &self.frozen_sent_starters {
            frozen.contains(token)
        } else {
            self.sent_starters.contains(token)
        }
    }

    pub fn get_ortho_context(&self, token_type: &str) -> u32 {
        self.ortho_context.get(token_type).copied().unwrap_or(0)
    }

    #[inline]
    pub fn get_token_stats(&self, token_type: &str) -> Option<&TokenStats> {
        self.token_stats.get(token_type)
    }

    pub fn update_token_stats(
        &mut self,
        token_type: impl Into<String>,
        update_fn: impl FnOnce(&mut TokenStats),
    ) {
        let stats = self.token_stats.entry(token_type.into()).or_default();
        update_fn(stats);
    }

    pub fn freeze(&mut self) {
        self.frozen_abbrev_types = Some(Arc::new(self.abbrev_types.keys().cloned().collect()));
        self.frozen_collocations = Some(Arc::new(self.collocations.iter().cloned().collect()));
        self.frozen_sent_starters = Some(Arc::new(self.sent_starters.iter().cloned().collect()));
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let mut params: Self = serde_json::from_str(json)?;
        params.freeze();
        Ok(params)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn save_compressed(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::fs::File;
        use std::io::Write;

        let json = self.to_json()?;
        let file = File::create(path)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(json.as_bytes())?;
        encoder.finish()?;
        Ok(())
    }

    pub fn load_compressed(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use flate2::read::GzDecoder;
        use std::fs::File;
        use std::io::Read;

        let file = File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        let mut json = String::new();
        decoder.read_to_string(&mut json)?;

        let mut params = Self::from_json(&json)?;
        params.freeze();
        Ok(params)
    }

    pub fn from_compressed_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(bytes);
        let mut json = String::new();
        decoder.read_to_string(&mut json)?;

        let mut params = Self::from_json(&json)?;
        params.freeze();
        Ok(params)
    }

    pub fn filter_tokens(
        &mut self,
        min_frequency: usize,
        max_tokens: Option<usize>,
    ) -> (usize, usize) {
        let original_count = self.token_stats.len();

        let mut token_frequencies: Vec<(String, usize)> = self
            .token_stats
            .iter()
            .map(|(token, stats)| {
                let freq = stats.count_with_period as usize
                    + stats.count_without_period as usize
                    + stats.count_as_starter as usize;
                (token.clone(), freq)
            })
            .filter(|(_, freq)| *freq >= min_frequency)
            .collect();

        token_frequencies.sort_by_key(|item| std::cmp::Reverse(item.1));
        if let Some(max) = max_tokens {
            token_frequencies.truncate(max);
        }

        let tokens_to_keep: HashSet<String> =
            token_frequencies.into_iter().map(|(t, _)| t).collect();

        let mut filtered_stats = HashMap::new();
        for (token, mut stats) in self.token_stats.drain() {
            if tokens_to_keep.contains(&token) {
                stats
                    .collocation_counts
                    .retain(|k, _| tokens_to_keep.contains(k));
                filtered_stats.insert(token, stats);
            }
        }
        self.token_stats = filtered_stats;

        self.ortho_context.retain(|k, _| tokens_to_keep.contains(k));

        let filtered_count = self.token_stats.len();
        self.frozen_abbrev_types = None;
        self.frozen_collocations = None;
        self.frozen_sent_starters = None;

        (original_count, filtered_count)
    }
}

impl Default for PunktParameters {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provided_abbreviation() {
        let mut params = PunktParameters::new();
        params.add_provided_abbreviation("Mr", "test.json");
        assert!(params.is_abbreviation("Mr"));
        assert!(params.is_abbreviation("mr")); // case insensitive
        assert!(params.is_provided_abbreviation("Mr"));
    }

    #[test]
    fn test_learned_abbreviation() {
        let mut params = PunktParameters::new();
        params.add_learned_abbreviation("Dr", 15.5, 50, 10);
        assert!(params.is_abbreviation("Dr"));
        assert!(!params.is_provided_abbreviation("Dr"));
    }

    #[test]
    fn test_provided_not_overridden() {
        let mut params = PunktParameters::new();
        params.add_provided_abbreviation("Inc", "external.json");
        params.add_learned_abbreviation("Inc", 10.0, 20, 5);
        assert!(params.is_provided_abbreviation("Inc"));
    }

    #[test]
    fn test_period_removal() {
        let mut params = PunktParameters::new();
        params.add_provided_abbreviation("U.S.", "test.json");
        assert!(params.is_abbreviation("U.S"));
        assert!(!params.is_abbreviation("U.S."));
    }

    #[test]
    fn test_collocation() {
        let mut params = PunktParameters::new();
        params.add_collocation("St", "Louis");
        assert!(params.is_collocation("st", "louis"));
        assert!(params.is_collocation("St", "Louis"));
    }

    #[test]
    fn test_json_roundtrip() {
        let mut params = PunktParameters::new();
        params.add_provided_abbreviation("Dr", "test");
        params.add_collocation("New", "York");
        params.add_sent_starter("The");

        let json = params.to_json().unwrap();
        let restored = PunktParameters::from_json(&json).unwrap();

        assert!(restored.is_abbreviation("dr"));
        assert!(restored.is_collocation("new", "york"));
        assert!(restored.is_sent_starter("The"));
    }

    #[test]
    fn test_filter_tokens() {
        let mut params = PunktParameters::new();

        // Add token stats with varying frequencies
        params.update_token_stats("the", |s| {
            s.count_with_period = 0;
            s.count_without_period = 100;
            s.count_as_starter = 10;
        });
        params.update_token_stats("cat", |s| {
            s.count_with_period = 0;
            s.count_without_period = 50;
            s.count_as_starter = 5;
        });
        params.update_token_stats("xyzzy", |s| {
            s.count_with_period = 0;
            s.count_without_period = 1;
            s.count_as_starter = 0;
        });

        assert_eq!(params.token_stats.len(), 3);

        // Filter with min_frequency=10 — "xyzzy" (freq=1) should be removed
        let (original, filtered) = params.filter_tokens(10, None);
        assert_eq!(original, 3);
        assert_eq!(filtered, 2);
        assert!(params.get_token_stats("the").is_some());
        assert!(params.get_token_stats("cat").is_some());
        assert!(params.get_token_stats("xyzzy").is_none());
    }
}
