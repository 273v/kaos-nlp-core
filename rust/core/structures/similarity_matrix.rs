//! Pairwise similarity matrix for document collections.
//!
//! Stores upper-triangular distances/similarities between N documents.
//! Supports Euclidean, Cosine, and Manhattan distance metrics.
//! Optimized for small-to-medium collections (N < 10000).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors raised by `SimilarityMatrix` on out-of-range document indices.
///
/// At the PyO3 boundary these are translated to `ValueError` rather than
/// surfacing as panics from unchecked vector indexing.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SimilarityMatrixError {
    #[error("document index {index} out of range (matrix has {n_docs} documents)")]
    IndexOutOfRange { index: usize, n_docs: usize },
}

/// Distance/similarity metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    Manhattan,
}

/// Pairwise similarity/distance matrix (upper triangular storage).
///
/// Stores N*(N-1)/2 values for N documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMatrix {
    /// Upper triangular values: distances[tri_index(i, j)] for i < j.
    distances: Vec<f64>,
    /// Number of documents.
    n_docs: usize,
    /// The metric used.
    metric: DistanceMetric,
}

impl SimilarityMatrix {
    /// Compute pairwise distances from sparse feature vectors.
    ///
    /// `features`: Vec of sparse vectors, each as `Vec<(term_id, value)>` sorted by term_id.
    pub fn from_sparse_vectors(features: &[Vec<(u32, f64)>], metric: DistanceMetric) -> Self {
        let n = features.len();
        let size = n * (n - 1) / 2;
        let mut distances = vec![0.0; size];

        // Pre-compute norms for cosine
        let norms: Vec<f64> = if metric == DistanceMetric::Cosine {
            features
                .iter()
                .map(|v| {
                    let sum_sq: f64 = v.iter().map(|(_, val)| val * val).sum();
                    sum_sq.sqrt()
                })
                .collect()
        } else {
            vec![]
        };

        for i in 0..n {
            for j in (i + 1)..n {
                let idx = tri_index(i, j, n);
                distances[idx] = match metric {
                    DistanceMetric::Euclidean => sparse_euclidean(&features[i], &features[j]),
                    DistanceMetric::Cosine => {
                        sparse_cosine_distance(&features[i], &features[j], norms[i], norms[j])
                    }
                    DistanceMetric::Manhattan => sparse_manhattan(&features[i], &features[j]),
                };
            }
        }

        Self {
            distances,
            n_docs: n,
            metric,
        }
    }

    /// Get distance between two documents.
    ///
    /// Returns 0.0 if i == j. Errors with `IndexOutOfRange` when either index
    /// is `>= n_docs` — pre-0.1.0a1 builds panicked via unchecked indexing.
    pub fn get_distance(&self, i: usize, j: usize) -> Result<f64, SimilarityMatrixError> {
        if i >= self.n_docs {
            return Err(SimilarityMatrixError::IndexOutOfRange {
                index: i,
                n_docs: self.n_docs,
            });
        }
        if j >= self.n_docs {
            return Err(SimilarityMatrixError::IndexOutOfRange {
                index: j,
                n_docs: self.n_docs,
            });
        }
        if i == j {
            return Ok(0.0);
        }
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        Ok(self.distances[tri_index(a, b, self.n_docs)])
    }

    /// Get similarity (1 - distance for Euclidean/Manhattan, or cosine similarity).
    ///
    /// For cosine metric, returns similarity directly.
    /// For others, returns 1 / (1 + distance).
    pub fn get_similarity(&self, i: usize, j: usize) -> Result<f64, SimilarityMatrixError> {
        if i == j && i < self.n_docs {
            return Ok(1.0);
        }
        let dist = self.get_distance(i, j)?;
        Ok(match self.metric {
            DistanceMetric::Cosine => 1.0 - dist, // Distance was (1 - cosine_sim)
            _ => 1.0 / (1.0 + dist),
        })
    }

    /// Find K nearest neighbors for a document.
    ///
    /// Returns (doc_index, distance) pairs sorted by distance ascending.
    /// Errors with `IndexOutOfRange` if `doc_idx >= n_docs`.
    pub fn k_nearest_neighbors(
        &self,
        doc_idx: usize,
        k: usize,
    ) -> Result<Vec<(usize, f64)>, SimilarityMatrixError> {
        if doc_idx >= self.n_docs {
            return Err(SimilarityMatrixError::IndexOutOfRange {
                index: doc_idx,
                n_docs: self.n_docs,
            });
        }
        // Distances inside this loop are guaranteed in-range by the check above
        // and the `i < n_docs` filter, so .expect is an internal invariant.
        let mut neighbors: Vec<(usize, f64)> = (0..self.n_docs)
            .filter(|&i| i != doc_idx)
            .map(|i| {
                (
                    i,
                    self.get_distance(doc_idx, i)
                        .expect("internal invariant: i and doc_idx both < n_docs"),
                )
            })
            .collect();

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(k);
        Ok(neighbors)
    }

    /// Number of documents.
    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    /// The metric used.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get all pairwise distances as a flat vector.
    pub fn distances(&self) -> &[f64] {
        &self.distances
    }

    /// Average distance across all pairs.
    pub fn mean_distance(&self) -> f64 {
        if self.distances.is_empty() {
            return 0.0;
        }
        self.distances.iter().sum::<f64>() / self.distances.len() as f64
    }
}

/// Convert (i, j) pair (i < j) to upper-triangular index.
#[inline]
fn tri_index(i: usize, j: usize, _n: usize) -> usize {
    debug_assert!(i < j);
    // Row i starts at position: i*(2n-i-1)/2 - i
    // Within row i, column j is at offset j - i - 1
    // Total: i*n - i*(i+1)/2 + j - i - 1
    // Simplified: (2*n*i - i*i - i) / 2 + j - i - 1
    let n = _n;
    (2 * n * i - i * i - i) / 2 + j - i - 1
}

/// Euclidean distance between two sparse vectors.
fn sparse_euclidean(a: &[(u32, f64)], b: &[(u32, f64)]) -> f64 {
    let mut sum_sq = 0.0;
    let mut ia = 0;
    let mut ib = 0;

    while ia < a.len() && ib < b.len() {
        let (ta, va) = a[ia];
        let (tb, vb) = b[ib];
        if ta == tb {
            let diff = va - vb;
            sum_sq += diff * diff;
            ia += 1;
            ib += 1;
        } else if ta < tb {
            sum_sq += va * va;
            ia += 1;
        } else {
            sum_sq += vb * vb;
            ib += 1;
        }
    }
    while ia < a.len() {
        sum_sq += a[ia].1 * a[ia].1;
        ia += 1;
    }
    while ib < b.len() {
        sum_sq += b[ib].1 * b[ib].1;
        ib += 1;
    }

    sum_sq.sqrt()
}

/// Cosine distance (1 - cosine similarity) between two sparse vectors.
fn sparse_cosine_distance(a: &[(u32, f64)], b: &[(u32, f64)], norm_a: f64, norm_b: f64) -> f64 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance
    }

    let mut dot = 0.0;
    let mut ia = 0;
    let mut ib = 0;

    while ia < a.len() && ib < b.len() {
        let (ta, va) = a[ia];
        let (tb, vb) = b[ib];
        if ta == tb {
            dot += va * vb;
            ia += 1;
            ib += 1;
        } else if ta < tb {
            ia += 1;
        } else {
            ib += 1;
        }
    }

    let sim = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    1.0 - sim
}

/// Manhattan distance between two sparse vectors.
fn sparse_manhattan(a: &[(u32, f64)], b: &[(u32, f64)]) -> f64 {
    let mut sum = 0.0;
    let mut ia = 0;
    let mut ib = 0;

    while ia < a.len() && ib < b.len() {
        let (ta, va) = a[ia];
        let (tb, vb) = b[ib];
        if ta == tb {
            sum += (va - vb).abs();
            ia += 1;
            ib += 1;
        } else if ta < tb {
            sum += va.abs();
            ia += 1;
        } else {
            sum += vb.abs();
            ib += 1;
        }
    }
    while ia < a.len() {
        sum += a[ia].1.abs();
        ia += 1;
    }
    while ib < b.len() {
        sum += b[ib].1.abs();
        ib += 1;
    }

    sum
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vectors() -> Vec<Vec<(u32, f64)>> {
        vec![
            vec![(0, 1.0), (1, 2.0), (2, 3.0)],
            vec![(0, 1.0), (1, 2.0), (2, 3.0)], // identical to [0]
            vec![(0, 4.0), (1, 5.0), (2, 6.0)],
            vec![(3, 1.0), (4, 2.0)], // disjoint from [0]
        ]
    }

    #[test]
    fn test_self_distance_zero() {
        let vecs = sample_vectors();
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Euclidean);
        for i in 0..vecs.len() {
            assert_eq!(mat.get_distance(i, i).unwrap(), 0.0);
        }
    }

    #[test]
    fn test_identical_vectors_zero_distance() {
        let vecs = sample_vectors();
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Euclidean);
        assert_eq!(mat.get_distance(0, 1).unwrap(), 0.0);
    }

    #[test]
    fn test_euclidean_symmetric() {
        let vecs = sample_vectors();
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Euclidean);
        assert_eq!(
            mat.get_distance(0, 2).unwrap(),
            mat.get_distance(2, 0).unwrap()
        );
    }

    #[test]
    fn test_cosine_distance() {
        let vecs = sample_vectors();
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Cosine);
        // Identical vectors should have cosine distance ≈ 0
        assert!(mat.get_distance(0, 1).unwrap() < 1e-10);
        // Parallel vectors (same direction, different magnitude) should be close
        let d = mat.get_distance(0, 2).unwrap();
        assert!(d < 0.1, "Parallel vectors cosine distance {} too high", d);
    }

    #[test]
    fn test_cosine_disjoint() {
        let vecs = sample_vectors();
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Cosine);
        // Disjoint vectors should have cosine distance = 1.0
        assert!((mat.get_distance(0, 3).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan() {
        let vecs = vec![vec![(0, 1.0), (1, 0.0)], vec![(0, 0.0), (1, 1.0)]];
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Manhattan);
        assert!((mat.get_distance(0, 1).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn() {
        let vecs = sample_vectors();
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Euclidean);
        let nn = mat.k_nearest_neighbors(0, 2).unwrap();
        assert_eq!(nn.len(), 2);
        // Nearest should be doc 1 (identical)
        assert_eq!(nn[0].0, 1);
        assert_eq!(nn[0].1, 0.0);
    }

    #[test]
    fn test_similarity() {
        let vecs = sample_vectors();
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Cosine);
        assert!((mat.get_similarity(0, 1).unwrap() - 1.0).abs() < 1e-10); // identical
        assert!(mat.get_similarity(0, 3).unwrap() < 0.1); // disjoint
    }

    #[test]
    fn test_mean_distance() {
        let vecs = sample_vectors();
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Euclidean);
        let mean = mat.mean_distance();
        assert!(mean > 0.0);
    }

    #[test]
    fn test_n_docs() {
        let vecs = sample_vectors();
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Euclidean);
        assert_eq!(mat.n_docs(), 4);
    }

    #[test]
    fn test_single_vector() {
        let vecs = vec![vec![(0, 1.0), (1, 2.0)]];
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Euclidean);
        assert_eq!(mat.n_docs(), 1);
        assert!(mat.distances().is_empty());
        assert_eq!(mat.get_distance(0, 0).unwrap(), 0.0);
        assert_eq!(mat.mean_distance(), 0.0);
    }

    #[test]
    fn test_knn_k_larger_than_n() {
        let vecs = sample_vectors(); // 4 vectors
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Euclidean);
        // k=10 but only 3 other docs exist (excluding self)
        let nn = mat.k_nearest_neighbors(0, 10).unwrap();
        assert_eq!(
            nn.len(),
            3,
            "Should return all 3 neighbors when k > n_docs-1"
        );
        // Should still be sorted by distance
        for i in 1..nn.len() {
            assert!(
                nn[i].1 >= nn[i - 1].1,
                "Neighbors should be sorted by distance"
            );
        }
    }

    #[test]
    fn test_manhattan_identical() {
        let vecs = vec![
            vec![(0, 3.0), (1, 4.0), (2, 5.0)],
            vec![(0, 3.0), (1, 4.0), (2, 5.0)],
        ];
        let mat = SimilarityMatrix::from_sparse_vectors(&vecs, DistanceMetric::Manhattan);
        let dist = mat.get_distance(0, 1).unwrap();
        assert!(
            (dist - 0.0).abs() < 1e-10,
            "Manhattan distance between identical vectors should be 0, got {}",
            dist
        );
    }
}
