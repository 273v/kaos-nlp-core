//! Core traits for string distance and similarity algorithms.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during distance/similarity computation.
#[derive(Debug, Error)]
pub enum DistanceError {
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("strings must be equal length (got {0} and {1})")]
    UnequalLength(usize, usize),

    #[error("parameter out of range: {0}")]
    ParameterOutOfRange(String),
}

/// Result type for distance/similarity computations.
pub type DistanceResult<T> = Result<T, DistanceError>;

/// Result of a distance computation with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceOutput {
    /// Raw distance value.
    pub distance: f64,
    /// Normalized distance in [0.0, 1.0] where 0.0 = identical.
    pub normalized: f64,
    /// Similarity in [0.0, 1.0] where 1.0 = identical.
    pub similarity: f64,
}

impl DistanceOutput {
    pub fn new(distance: f64, max_possible: f64) -> Self {
        let normalized = if max_possible > 0.0 {
            distance / max_possible
        } else {
            0.0
        };
        Self {
            distance,
            normalized,
            similarity: 1.0 - normalized,
        }
    }

    /// Create from a similarity value in [0.0, 1.0].
    pub fn from_similarity(similarity: f64) -> Self {
        Self {
            distance: 1.0 - similarity,
            normalized: 1.0 - similarity,
            similarity,
        }
    }
}

/// Trait for algorithms that compute a distance between two strings.
///
/// Distance is a non-negative value where 0 means identical strings.
pub trait StringDistance: Send + Sync {
    /// Compute the distance between two strings.
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput>;

    /// Compute distance with early termination if it exceeds threshold.
    /// Default implementation ignores the threshold.
    fn distance_with_threshold(
        &self,
        a: &str,
        b: &str,
        _threshold: f64,
    ) -> DistanceResult<Option<DistanceOutput>> {
        self.distance(a, b).map(Some)
    }
}

/// Trait for algorithms that compute similarity between two strings.
///
/// Similarity is a value in [0.0, 1.0] where 1.0 means identical.
pub trait StringSimilarity: Send + Sync {
    /// Compute similarity between two strings.
    fn similarity(&self, a: &str, b: &str) -> DistanceResult<f64>;
}

// Every StringDistance is also a StringSimilarity.
impl<T: StringDistance> StringSimilarity for T {
    fn similarity(&self, a: &str, b: &str) -> DistanceResult<f64> {
        Ok(self.distance(a, b)?.similarity)
    }
}
