//! Context-Triggered Piecewise Hashing (CTPH) for near-duplicate detection.
//!
//! Provides:
//! - `RollingHash`: Generic rolling hash over a sliding window (O(1) updates)
//! - `CTPH`: Byte-level CTPH using blake3 piece hashing
//! - `TokenCTPH`: Token-level CTPH for integer token arrays (LLM outputs)
//! - `CTPHDigest`: Typed digest with Jaccard similarity comparison
//!
//! Ported from alea-preprocess reference implementation with improvements:
//! - Generic `RollingHash` instead of 4 copy-pasted structs
//! - `VecDeque` for O(1) window sliding (reference used `Vec::remove(0)` = O(n))
//! - `CTPHDigest` as typed struct (reference used raw strings)
//! - blake3 for piece hashing across all precisions

use ahash::AHashSet;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;
use std::num::Wrapping;

// ─── Rolling Hash ────────────────────────────────────────────────────────────

/// Generic rolling hash over a sliding window.
///
/// Uses wrapping addition/subtraction and rotate-left for distribution.
/// `T` must be an unsigned integer type implementing the required arithmetic.
#[derive(Debug, Clone)]
pub struct RollingHash<T> {
    window: VecDeque<T>,
    window_size: usize,
    hash: Wrapping<T>,
}

/// Trait bounds for rolling hash value types.
pub trait RollingHashValue:
    Copy
    + Default
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + PartialEq
    + std::hash::Hash
{
    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn rotate_left(self, n: u32) -> Self;
    fn rem_usize(self, divisor: usize) -> usize;
    fn from_u8(v: u8) -> Self;
}

macro_rules! impl_rolling_hash_value {
    ($($t:ty),+) => {
        $(
            impl RollingHashValue for $t {
                #[inline]
                fn wrapping_add(self, rhs: Self) -> Self { self.wrapping_add(rhs) }
                #[inline]
                fn wrapping_sub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }
                #[inline]
                fn rotate_left(self, n: u32) -> Self { self.rotate_left(n) }
                #[inline]
                fn rem_usize(self, divisor: usize) -> usize { (self as usize) % divisor }
                #[inline]
                fn from_u8(v: u8) -> Self { v as Self }
            }
        )+
    };
}

impl_rolling_hash_value!(u8, u16, u32, u64);

impl<T: RollingHashValue> RollingHash<T> {
    /// Create a new rolling hash with the given window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            hash: Wrapping(T::default()),
        }
    }

    /// Update the rolling hash with a new value.
    #[inline]
    pub fn update(&mut self, value: T) {
        if self.window.len() == self.window_size {
            if let Some(old) = self.window.pop_front() {
                self.hash = Wrapping(self.hash.0.wrapping_sub(old));
            }
        }
        self.window.push_back(value);
        self.hash = Wrapping(self.hash.0.wrapping_add(value).rotate_left(1));
    }

    /// Current hash value.
    #[inline]
    pub fn hash(&self) -> T {
        self.hash.0
    }

    /// Current window contents.
    pub fn window(&self) -> &VecDeque<T> {
        &self.window
    }

    /// Window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

// ─── CTPH Digest ─────────────────────────────────────────────────────────────

/// A CTPH digest: serializable representation of a context-triggered piecewise hash.
///
/// Contains the parameters used to generate the hash and the resulting blocks.
/// Two digests are comparable only if they share the same window_size and digest_size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CTPHDigest {
    /// Window size used for the rolling hash.
    pub window_size: usize,
    /// Digest size (trigger modulus).
    pub digest_size: usize,
    /// Hash blocks (hex-encoded blake3 piece hashes, concatenated per block).
    pub blocks: Vec<String>,
    /// Length of each piece hash in hex characters (precision * 2 for byte CTPH, 16 for token CTPH).
    pub piece_hex_len: usize,
}

impl CTPHDigest {
    /// Compute block-level Jaccard similarity between two digests.
    ///
    /// Compares compound blocks (each containing multiple concatenated piece hashes).
    /// This is the original ssdeep-style comparison — strict, best for exact copy detection.
    ///
    /// Returns 0.0 if parameters don't match or both are empty.
    /// Returns 1.0 for identical digests.
    pub fn similarity(&self, other: &CTPHDigest) -> f64 {
        if self.window_size != other.window_size || self.digest_size != other.digest_size {
            return 0.0;
        }

        let set1: AHashSet<&str> = self.blocks.iter().map(|s| s.as_str()).collect();
        let set2: AHashSet<&str> = other.blocks.iter().map(|s| s.as_str()).collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.len() + set2.len() - intersection;

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Compute piece-level Jaccard similarity between two digests.
    ///
    /// Splits compound blocks into individual piece hashes and compares those.
    /// More tolerant of edits than `similarity()` because a single changed piece
    /// only invalidates that piece, not the entire block.
    ///
    /// **Use this for document versioning and edit detection** (Word/PDF with minor
    /// changes, contract revisions, etc.). Use `similarity()` for exact copy detection.
    ///
    /// Returns 0.0 if parameters don't match or both are empty.
    /// Returns 1.0 for identical digests.
    pub fn piece_similarity(&self, other: &CTPHDigest) -> f64 {
        if self.window_size != other.window_size || self.digest_size != other.digest_size {
            return 0.0;
        }

        let plen = self.piece_hex_len;
        if plen == 0 {
            return self.similarity(other);
        }

        let pieces1 = Self::extract_pieces(&self.blocks, plen);
        let pieces2 = Self::extract_pieces(&other.blocks, plen);

        let intersection = pieces1.intersection(&pieces2).count();
        let union = pieces1.len() + pieces2.len() - intersection;

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Extract individual piece hashes from compound blocks.
    fn extract_pieces(blocks: &[String], piece_hex_len: usize) -> AHashSet<&str> {
        let mut pieces = AHashSet::new();
        for block in blocks {
            let mut i = 0;
            while i + piece_hex_len <= block.len() {
                pieces.insert(&block[i..i + piece_hex_len]);
                i += piece_hex_len;
            }
        }
        pieces
    }

    /// Number of individual pieces across all blocks.
    pub fn num_pieces(&self) -> usize {
        if self.piece_hex_len == 0 {
            return 0;
        }
        self.blocks
            .iter()
            .map(|b| b.len() / self.piece_hex_len)
            .sum()
    }

    /// Serialize to string format: "window_size:digest_size:block1:block2:..."
    pub fn to_string_repr(&self) -> String {
        if self.blocks.is_empty() {
            format!("{}:{}:", self.window_size, self.digest_size)
        } else {
            let blocks_str = self.blocks.join(":");
            format!("{}:{}:{}", self.window_size, self.digest_size, blocks_str)
        }
    }

    /// Parse from string format: "window_size:digest_size:block1:block2:..."
    pub fn from_string(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() < 2 {
            return Err("CTPH digest must have at least window_size:digest_size".to_string());
        }

        let window_size = parts[0]
            .parse::<usize>()
            .map_err(|e| format!("Invalid window_size: {e}"))?;
        let digest_size = parts[1]
            .parse::<usize>()
            .map_err(|e| format!("Invalid digest_size: {e}"))?;

        let blocks: Vec<String> = parts[2..]
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        // Infer piece_hex_len from first non-empty block
        // For byte CTPH with precision p: piece_hex_len = p * 2
        // For token CTPH: piece_hex_len = 16 (8 bytes)
        // Heuristic: look for common divisors of block length
        let piece_hex_len = blocks
            .first()
            .map(|b| Self::infer_piece_hex_len(b.len()))
            .unwrap_or(8);

        Ok(Self {
            window_size,
            digest_size,
            blocks,
            piece_hex_len,
        })
    }

    /// Infer piece hex length from a block length.
    /// Common piece sizes: 2 (1-byte), 4 (2-byte), 8 (4-byte), 16 (8-byte).
    fn infer_piece_hex_len(block_len: usize) -> usize {
        for &candidate in &[8, 16, 4, 2] {
            if block_len.is_multiple_of(candidate) {
                return candidate;
            }
        }
        8 // default to 4-byte precision
    }
}

impl fmt::Display for CTPHDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_repr())
    }
}

// ─── Byte CTPH ───────────────────────────────────────────────────────────────

/// Context-Triggered Piecewise Hashing for byte data.
///
/// Uses a rolling hash to identify block boundaries, then hashes each block
/// with blake3. The result is a sequence of hex-encoded block hashes that can
/// be compared via Jaccard similarity.
///
/// `precision`: Number of bytes in the blake3 output per piece (1, 2, 4, or 8).
/// Higher precision = fewer collisions but larger digests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CTPH {
    /// Rolling hash window size.
    pub window_size: usize,
    /// Trigger modulus for block boundaries.
    pub digest_size: usize,
    /// Bytes of blake3 output per piece (1, 2, 4, or 8).
    pub precision: u8,
}

impl CTPH {
    /// Create a new CTPH hasher.
    ///
    /// - `window_size`: Rolling hash window size (e.g., 64, 128).
    /// - `digest_size`: Block boundary trigger modulus (e.g., 8, 16).
    /// - `precision`: Blake3 output bytes per piece: 1 (8-bit), 2 (16-bit), 4 (32-bit), 8 (64-bit).
    pub fn new(window_size: usize, digest_size: usize, precision: u8) -> Self {
        Self {
            window_size,
            digest_size,
            precision: match precision {
                1 | 2 | 4 | 8 => precision,
                _ => 4, // default to 32-bit
            },
        }
    }

    /// Hash a piece of data with blake3, returning hex-encoded output.
    fn hash_piece(&self, data: &[u8]) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(data);
        let n = self.precision as usize;
        let mut result = vec![0u8; n];
        hasher.finalize_xof().fill(&mut result);
        hex_encode(&result)
    }

    /// Compute CTPH digest of byte data.
    pub fn compute(&self, data: &[u8]) -> CTPHDigest {
        let mut rolling: RollingHash<u32> = RollingHash::new(self.window_size);
        let mut blocks = vec![String::new()];
        let mut current_piece = Vec::new();
        let mut trigger_count = 0;

        for &byte in data {
            rolling.update(byte as u32);
            current_piece.push(byte);

            // Trigger on rolling hash value or max piece size
            if rolling.hash().rem_usize(self.digest_size) == self.digest_size - 1
                || current_piece.len() >= 64 * self.window_size
            {
                let piece_hash = self.hash_piece(&current_piece);
                blocks.last_mut().unwrap().push_str(&piece_hash);
                current_piece.clear();
                trigger_count += 1;

                if trigger_count % self.digest_size == 0 {
                    blocks.push(String::new());
                }
            }
        }

        // Hash remaining data
        if !current_piece.is_empty() {
            let piece_hash = self.hash_piece(&current_piece);
            blocks.last_mut().unwrap().push_str(&piece_hash);
        }

        // Remove empty blocks
        blocks.retain(|block| !block.is_empty());

        CTPHDigest {
            window_size: self.window_size,
            digest_size: self.digest_size,
            blocks,
            piece_hex_len: self.precision as usize * 2,
        }
    }

    /// Compute CTPH digest of a string (hashes as UTF-8 bytes).
    pub fn hash_str(&self, text: &str) -> CTPHDigest {
        self.compute(text.as_bytes())
    }
}

// ─── Token CTPH ──────────────────────────────────────────────────────────────

/// Context-Triggered Piecewise Hashing for integer token arrays.
///
/// Designed for LLM tokenizer outputs. Uses blake3 to hash each piece
/// of the token sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenCTPH {
    /// Rolling hash window size.
    pub window_size: usize,
    /// Trigger modulus for block boundaries.
    pub digest_size: usize,
}

impl TokenCTPH {
    pub fn new(window_size: usize, digest_size: usize) -> Self {
        Self {
            window_size,
            digest_size,
        }
    }

    /// Hash a piece of the token sequence with blake3 (8 bytes / 64-bit).
    fn hash_piece(&self, tokens: &[i64]) -> String {
        let mut hasher = blake3::Hasher::new();
        for &token in tokens {
            hasher.update(&token.to_le_bytes());
        }
        let mut result = [0u8; 8];
        hasher.finalize_xof().fill(&mut result);
        hex_encode(&result)
    }

    /// Compute CTPH digest of a token array.
    pub fn compute(&self, tokens: &[i64]) -> CTPHDigest {
        let mut rolling: RollingHash<u64> = RollingHash::new(self.window_size);
        let mut blocks = vec![String::new()];
        let mut current_piece = Vec::new();
        let mut trigger_count = 0;

        for &token in tokens {
            rolling.update(token as u64);
            current_piece.push(token);

            // Trigger on rolling hash value or max piece size
            if rolling.hash().rem_usize(self.digest_size) == self.digest_size - 1
                || current_piece.len() >= self.window_size
            {
                let piece_hash = self.hash_piece(&current_piece);
                blocks.last_mut().unwrap().push_str(&piece_hash);
                current_piece.clear();
                trigger_count += 1;

                if trigger_count % self.digest_size == 0 {
                    blocks.push(String::new());
                }
            }
        }

        // Hash remaining tokens
        if !current_piece.is_empty() {
            let piece_hash = self.hash_piece(&current_piece);
            blocks.last_mut().unwrap().push_str(&piece_hash);
        }

        blocks.retain(|block| !block.is_empty());

        CTPHDigest {
            window_size: self.window_size,
            digest_size: self.digest_size,
            blocks,
            piece_hex_len: 16, // 8 bytes = 16 hex chars for token CTPH
        }
    }
}

// ─── Convenience functions ───────────────────────────────────────────────────

/// Compute CTPH hash of byte data, returning the string representation.
pub fn hash_bytes(data: &[u8], window_size: usize, digest_size: usize, precision: u8) -> String {
    CTPH::new(window_size, digest_size, precision)
        .compute(data)
        .to_string_repr()
}

/// Compute CTPH hash of a string, returning the string representation.
pub fn hash_str(text: &str, window_size: usize, digest_size: usize, precision: u8) -> String {
    hash_bytes(text.as_bytes(), window_size, digest_size, precision)
}

/// Compare two CTPH hash strings and return Jaccard similarity.
pub fn similarity(hash1: &str, hash2: &str) -> f64 {
    match (
        CTPHDigest::from_string(hash1),
        CTPHDigest::from_string(hash2),
    ) {
        (Ok(d1), Ok(d2)) => d1.similarity(&d2),
        _ => 0.0,
    }
}

/// Compute token CTPH hash, returning the string representation.
pub fn hash_tokens(tokens: &[i64], window_size: usize, digest_size: usize) -> String {
    TokenCTPH::new(window_size, digest_size)
        .compute(tokens)
        .to_string_repr()
}

// ─── Hex encoding (no crate dependency) ──────────────────────────────────────

/// Hex-encode a byte slice (lowercase).
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Rolling Hash tests
    // ──────────────────

    #[test]
    fn test_rolling_hash_basic() {
        let mut rh: RollingHash<u32> = RollingHash::new(4);
        rh.update(1);
        rh.update(2);
        rh.update(3);
        assert_eq!(rh.window().len(), 3);
        assert_ne!(rh.hash(), 0);
    }

    #[test]
    fn test_rolling_hash_window_overflow() {
        let mut rh: RollingHash<u32> = RollingHash::new(2);
        rh.update(10);
        rh.update(20);
        assert_eq!(rh.window().len(), 2);
        let hash_before = rh.hash();
        rh.update(30);
        assert_eq!(rh.window().len(), 2);
        assert_ne!(rh.hash(), hash_before);
        // Window should contain [20, 30], not [10, 20]
        assert_eq!(rh.window()[0], 20);
        assert_eq!(rh.window()[1], 30);
    }

    #[test]
    fn test_rolling_hash_stability() {
        let mut rh1: RollingHash<u32> = RollingHash::new(3);
        let mut rh2: RollingHash<u32> = RollingHash::new(3);
        for v in [1, 2, 3] {
            rh1.update(v);
            rh2.update(v);
        }
        assert_eq!(rh1.hash(), rh2.hash());
    }

    #[test]
    fn test_rolling_hash_different_precisions() {
        let mut rh8: RollingHash<u8> = RollingHash::new(3);
        let mut rh64: RollingHash<u64> = RollingHash::new(3);
        for v in [1u8, 2, 3] {
            rh8.update(v);
            rh64.update(v as u64);
        }
        // Different types should produce different hash values (type widths differ)
        // Just verify both run without panic
        assert_ne!(rh8.hash(), 0);
        assert_ne!(rh64.hash(), 0);
    }

    // CTPH tests
    // ──────────

    #[test]
    fn test_ctph_compute_empty() {
        let ctph = CTPH::new(8, 4, 4);
        let digest = ctph.compute(b"");
        assert!(digest.blocks.is_empty());
    }

    #[test]
    fn test_ctph_compute_basic() {
        let ctph = CTPH::new(8, 4, 4);
        let digest = ctph.compute(b"hello world this is a test");
        assert_eq!(digest.window_size, 8);
        assert_eq!(digest.digest_size, 4);
        assert!(!digest.blocks.is_empty());
    }

    #[test]
    fn test_ctph_identical_inputs() {
        let ctph = CTPH::new(64, 8, 4);
        let text = "The quick brown fox jumps over the lazy dog.";
        let d1 = ctph.hash_str(text);
        let d2 = ctph.hash_str(text);
        assert_eq!(d1.similarity(&d2), 1.0);
    }

    #[test]
    fn test_ctph_different_inputs() {
        let ctph = CTPH::new(64, 8, 4);
        let d1 = ctph.hash_str("The quick brown fox jumps over the lazy dog.");
        let d2 = ctph.hash_str("Lorem ipsum dolor sit amet, consectetur adipiscing elit.");
        let sim = d1.similarity(&d2);
        assert!(sim < 0.5, "Different texts similarity {} too high", sim);
    }

    #[test]
    fn test_ctph_similar_inputs() {
        let ctph = CTPH::new(16, 4, 4);
        let base = "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let modified = format!("{}{}", base, "A small addition at the end.");
        let d1 = ctph.hash_str(&base);
        let d2 = ctph.hash_str(&modified);
        let sim = d1.similarity(&d2);
        assert!(
            sim > 0.0,
            "Similar texts should have positive similarity, got {sim}"
        );
    }

    #[test]
    fn test_ctph_digest_roundtrip() {
        let ctph = CTPH::new(64, 8, 4);
        let digest = ctph.hash_str("test data for roundtrip");
        let s = digest.to_string_repr();
        let restored = CTPHDigest::from_string(&s).unwrap();
        assert_eq!(digest, restored);
        assert_eq!(digest.similarity(&restored), 1.0);
    }

    #[test]
    fn test_ctph_different_params_zero_similarity() {
        let d1 = CTPHDigest {
            window_size: 64,
            digest_size: 8,
            blocks: vec!["abc".to_string()],
            piece_hex_len: 8,
        };
        let d2 = CTPHDigest {
            window_size: 128,
            digest_size: 8,
            blocks: vec!["abc".to_string()],
            piece_hex_len: 8,
        };
        assert_eq!(d1.similarity(&d2), 0.0);
    }

    #[test]
    fn test_ctph_all_precisions() {
        for precision in [1, 2, 4, 8] {
            let ctph = CTPH::new(16, 4, precision);
            let d = ctph.compute(b"test data for all precisions check");
            assert!(
                !d.blocks.is_empty(),
                "precision={precision} produced no blocks"
            );
        }
    }

    #[test]
    fn test_ctph_convenience_functions() {
        let h1 = hash_str("hello world", 8, 4, 4);
        let h2 = hash_str("hello world", 8, 4, 4);
        assert_eq!(similarity(&h1, &h2), 1.0);

        let h3 = hash_bytes(b"hello world", 8, 4, 4);
        assert_eq!(h1, h3);
    }

    // Token CTPH tests
    // ────────────────

    #[test]
    fn test_token_ctph_basic() {
        let ctph = TokenCTPH::new(4, 8);
        let tokens = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let digest = ctph.compute(&tokens);
        assert!(!digest.blocks.is_empty());
        assert_eq!(digest.window_size, 4);
        assert_eq!(digest.digest_size, 8);
    }

    #[test]
    fn test_token_ctph_identical() {
        let ctph = TokenCTPH::new(4, 8);
        let tokens = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let d1 = ctph.compute(&tokens);
        let d2 = ctph.compute(&tokens);
        assert_eq!(d1.similarity(&d2), 1.0);
    }

    #[test]
    fn test_token_ctph_different() {
        let ctph = TokenCTPH::new(4, 8);
        let d1 = ctph.compute(&[1, 2, 3, 4, 5]);
        let d2 = ctph.compute(&[100, 200, 300, 400, 500]);
        let sim = d1.similarity(&d2);
        assert!(
            sim < 0.5,
            "Different token sequences similarity {sim} too high"
        );
    }

    #[test]
    fn test_token_ctph_similar() {
        // Real-ish tokenizer output: two similar sentences
        let tokens1: Vec<i64> = vec![
            6153, 424, 24, 300, 281, 17938, 295, 281, 1032, 922, 377, 300, 281, 45261, 24, 2413,
            24, 377, 38148, 295, 4639, 3184, 54456, 310, 2899, 295, 281, 1032, 922, 1171, 9018,
            377, 777, 4845, 281, 1974, 1412, 15118, 295, 2507, 16228, 1228, 3156,
        ];
        let tokens2: Vec<i64> = vec![
            6153, 424, 24, 300, 281, 17938, 295, 281, 1032, 922, 377, 300, 281, 45261, 24, 2413,
            24, 377, 38148, 295, 4639, 3184, 54456, 310, 2899, 295, 281, 1032, 922, 1171, 9018,
            377, 777, 4845, 281, 1974, 1412, 15118, 295, 2507, 16228, 1228, 23805,
        ];
        let ctph = TokenCTPH::new(4, 8);
        let d1 = ctph.compute(&tokens1);
        let d2 = ctph.compute(&tokens2);
        let sim = d1.similarity(&d2);
        // Should have non-zero similarity (sequences differ by 1 token)
        assert!(
            sim > 0.0,
            "Similar token sequences should have positive similarity, got {sim}"
        );
    }

    #[test]
    fn test_token_ctph_empty() {
        let ctph = TokenCTPH::new(4, 8);
        let digest = ctph.compute(&[]);
        assert!(digest.blocks.is_empty());
    }

    #[test]
    fn test_token_ctph_large_values() {
        let ctph = TokenCTPH::new(4, 8);
        let tokens = vec![i64::MAX, i64::MIN, 0, 42];
        let digest = ctph.compute(&tokens);
        // Just verify no panic
        assert!(!digest.blocks.is_empty());
    }

    #[test]
    fn test_token_ctph_convenience() {
        let h1 = hash_tokens(&[1, 2, 3, 4, 5], 4, 8);
        let h2 = hash_tokens(&[1, 2, 3, 4, 5], 4, 8);
        assert_eq!(similarity(&h1, &h2), 1.0);
    }

    #[test]
    fn test_token_ctph_different_params() {
        let h1 = hash_tokens(&[1, 2, 3, 4, 5], 2, 4);
        let h2 = hash_tokens(&[1, 2, 3, 4, 5], 3, 4);
        assert_eq!(similarity(&h1, &h2), 0.0); // Different window sizes
    }

    // CTPHDigest tests
    // ────────────────

    #[test]
    fn test_digest_from_string_invalid() {
        assert!(CTPHDigest::from_string("").is_err());
        assert!(CTPHDigest::from_string("abc").is_err());
    }

    #[test]
    fn test_digest_display() {
        let d = CTPHDigest {
            window_size: 64,
            digest_size: 8,
            blocks: vec!["abc123".to_string(), "def456".to_string()],
            piece_hex_len: 8,
        };
        let s = format!("{d}");
        assert_eq!(s, "64:8:abc123:def456");
    }

    #[test]
    fn test_digest_empty_blocks() {
        let d = CTPHDigest {
            window_size: 64,
            digest_size: 8,
            blocks: vec![],
            piece_hex_len: 8,
        };
        assert_eq!(d.to_string_repr(), "64:8:");
        assert_eq!(d.similarity(&d), 0.0); // empty union = 0
    }

    // Hex encoding test
    // ─────────────────

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[0xff, 0x00, 0xab, 0xcd]), "ff00abcd");
        assert_eq!(hex_encode(&[]), "");
    }

    // piece_similarity tests
    // ──────────────────────

    #[test]
    fn test_piece_similarity_identical() {
        let ctph = CTPH::new(64, 8, 4);
        let d = ctph.hash_str("The quick brown fox jumps over the lazy dog.");
        assert_eq!(d.piece_similarity(&d), 1.0);
    }

    #[test]
    fn test_piece_similarity_better_than_block() {
        let ctph = CTPH::new(64, 8, 4);
        let base = "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let modified = {
            let mid = base.len() / 2;
            format!(
                "{}{}{}",
                &base[..mid - 100],
                "X".repeat(200),
                &base[mid + 100..]
            )
        };
        let d1 = ctph.hash_str(&base);
        let d2 = ctph.hash_str(&modified);

        let block_sim = d1.similarity(&d2);
        let piece_sim = d1.piece_similarity(&d2);

        // Piece similarity should be >= block similarity
        assert!(
            piece_sim >= block_sim,
            "piece_sim={piece_sim} should be >= block_sim={block_sim}"
        );
        // For moderate edits, piece should be significantly higher
        assert!(
            piece_sim > 0.3,
            "piece_sim={piece_sim} should be > 0.3 for moderate edit"
        );
    }

    #[test]
    fn test_piece_similarity_different_params_zero() {
        let d1 = CTPHDigest {
            window_size: 64,
            digest_size: 8,
            blocks: vec!["aabbccdd".to_string()],
            piece_hex_len: 8,
        };
        let d2 = CTPHDigest {
            window_size: 128,
            digest_size: 8,
            blocks: vec!["aabbccdd".to_string()],
            piece_hex_len: 8,
        };
        assert_eq!(d1.piece_similarity(&d2), 0.0);
    }

    #[test]
    fn test_piece_similarity_token_ctph() {
        let ctph = TokenCTPH::new(4, 8);
        let orig: Vec<i64> = (0..1000).collect();
        let mut modified = orig.clone();
        // Replace 5% of tokens
        for i in (0..modified.len()).step_by(20) {
            modified[i] = 99999;
        }
        let d1 = ctph.compute(&orig);
        let d2 = ctph.compute(&modified);

        let block_sim = d1.similarity(&d2);
        let piece_sim = d1.piece_similarity(&d2);

        // Piece should detect more shared content than block
        assert!(
            piece_sim >= block_sim,
            "piece_sim={piece_sim} should be >= block_sim={block_sim}"
        );
    }

    #[test]
    fn test_num_pieces() {
        let ctph = CTPH::new(64, 8, 4);
        let d = ctph.hash_str("test data for piece counting");
        assert!(d.num_pieces() > 0);
        // Each piece is 8 hex chars (precision=4)
        let expected = d.blocks.iter().map(|b| b.len() / 8).sum::<usize>();
        assert_eq!(d.num_pieces(), expected);
    }
}
