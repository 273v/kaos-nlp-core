//! Core types for segmentation: segments, decisions, scoring config, orthographic constants.
//!
//! Ported from nupunkt-rs analysis.rs and core.rs.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

// ─── Segment ─────────────────────────────────────────────────────────────────

/// A text segment with byte-offset span and confidence score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    /// Byte offset start (inclusive).
    pub start: usize,
    /// Byte offset end (exclusive).
    pub end: usize,
    /// Confidence score [0.0, 1.0].
    pub confidence: f64,
}

impl Segment {
    #[inline]
    pub fn new(start: usize, end: usize, confidence: f64) -> Self {
        Self {
            start,
            end,
            confidence,
        }
    }

    #[inline]
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start..self.end]
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.end - self.start
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

// ─── Decision types ──────────────────────────────────────────────────────────

/// Type alias for small factor collections (typically 1-6 elements).
pub type FactorVec = SmallVec<[DecisionFactor; 4]>;

/// Decision made at a token boundary.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BreakDecision {
    Break,
    NoBreak,
    Continue,
    Uncertain,
}

/// A factor contributing to a boundary decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionFactor {
    pub factor_type: FactorType,
    pub weight: f64,
    pub description: String,
}

/// Types of factors that influence boundary decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FactorType {
    Abbreviation,
    Collocation,
    Capitalization,
    SentenceStarter,
    Consistency,
    Score,
    EndOfText,
    Whitespace,
}

// ─── Scoring configuration ──────────────────────────────────────────────────

/// Configuration for scoring thresholds during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Threshold for abbreviation detection (default: 0.1).
    pub abbrev_threshold: f64,
    /// Boosting factor for abbreviation scores (default: 1.5).
    pub abbrev_boost: f64,
    /// Threshold for collocation detection (default: 5.0).
    pub collocation_threshold: f64,
    /// Threshold for sentence starter detection (default: 25.0).
    pub sent_starter_threshold: f64,
    /// Minimum frequency for collocations as a rate (default: 0.00001).
    pub min_colloc_rate: f64,
    /// Maximum length for abbreviation candidates (default: 9).
    pub max_abbrev_length: usize,
    /// Consistency threshold for abbreviations (default: 0.25).
    pub abbrev_consistency: f64,
    /// Minimum rate for sentence starters (default: 0.00005).
    pub min_starter_rate: f64,
    /// Require sentence starters to be alphabetic (default: true).
    pub require_alpha_starters: bool,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            abbrev_threshold: 0.1,
            abbrev_boost: 1.5,
            collocation_threshold: 5.0,
            sent_starter_threshold: 25.0,
            min_colloc_rate: 0.00001,
            max_abbrev_length: 9,
            abbrev_consistency: 0.25,
            min_starter_rate: 0.00005,
            require_alpha_starters: true,
        }
    }
}

// ─── Orthographic context flags ─────────────────────────────────────────────

pub const ORTHO_BEG_UC: u32 = 1 << 1; // Uppercase at sentence beginning
pub const ORTHO_MID_UC: u32 = 1 << 2; // Uppercase mid-sentence
pub const ORTHO_UNK_UC: u32 = 1 << 3; // Unknown position uppercase
pub const ORTHO_BEG_LC: u32 = 1 << 4; // Lowercase at sentence beginning
pub const ORTHO_MID_LC: u32 = 1 << 5; // Lowercase mid-sentence
pub const ORTHO_UNK_LC: u32 = 1 << 6; // Unknown position lowercase
pub const ORTHO_UC: u32 = ORTHO_BEG_UC | ORTHO_MID_UC | ORTHO_UNK_UC;
pub const ORTHO_LC: u32 = ORTHO_BEG_LC | ORTHO_MID_LC | ORTHO_UNK_LC;
