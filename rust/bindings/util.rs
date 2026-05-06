//! Shared utilities for PyO3 bindings.
//!
//! Common patterns used across multiple binding modules:
//! - Byte-to-character offset conversion for returning text positions to Python
//! - Bincode 2.x pickle helpers for serde-compatible types
//! - File-load helpers with magic header, format version, and size cap

use pyo3::prelude::*;
use pyo3::types::PyBytes;

// =============================================================================
// Byte → char offset conversion
// =============================================================================

/// Build a byte-index → char-index lookup table for non-ASCII text.
///
/// Returns `None` for ASCII text where byte offsets == char offsets.
/// For non-ASCII text, returns a table where `table[byte_offset]` gives the
/// corresponding character offset suitable for Python string indexing.
///
/// Uses a single O(n) pass. NEVER use `text[..byte_pos].chars().count()`
/// in a loop — that's O(n×m).
pub fn build_byte_to_char_table(text: &str) -> Option<Vec<usize>> {
    if text.is_ascii() {
        return None; // ASCII fast path: byte offsets == char offsets
    }
    let mut offsets = Vec::with_capacity(text.len() + 1);
    let mut char_count = 0;
    for (byte_idx, _) in text.char_indices() {
        while offsets.len() <= byte_idx {
            offsets.push(char_count);
        }
        char_count += 1;
    }
    while offsets.len() <= text.len() {
        offsets.push(char_count);
    }
    Some(offsets)
}

/// Convert a byte offset to a char offset using an optional lookup table.
///
/// When the table is `None` (ASCII text), returns the byte offset unchanged.
/// Bounds-checked: clamps to the last valid entry.
#[inline]
pub fn byte_to_char(table: &Option<Vec<usize>>, byte_offset: usize) -> usize {
    match table {
        Some(t) => t[byte_offset.min(t.len() - 1)],
        None => byte_offset,
    }
}

/// Build a byte-to-char table unconditionally (always returns a Vec).
///
/// Used by the searcher binding where the caller handles the ASCII
/// fast path at a higher level.
pub fn build_byte_to_char_table_always(text: &str) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(text.len() + 1);
    let mut char_count = 0;
    for (byte_idx, _) in text.char_indices() {
        while offsets.len() <= byte_idx {
            offsets.push(char_count);
        }
        char_count += 1;
    }
    while offsets.len() <= text.len() {
        offsets.push(char_count);
    }
    offsets
}

// =============================================================================
// postcard helpers (serde-compatible, replaces unmaintained bincode)
// =============================================================================

/// Serialize a serde-compatible value to a `Vec<u8>` via postcard.
fn encode_to_vec<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, postcard::Error> {
    postcard::to_allocvec(value)
}

/// Deserialize a serde-compatible value from a slice via postcard.
fn decode_from_slice<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, postcard::Error> {
    postcard::from_bytes(bytes)
}

// -----------------------------------------------------------------------------
// Pickle helpers (trusted: bytes come from Python pickle stream)
// -----------------------------------------------------------------------------

/// Serialize a serde-compatible struct to PyBytes via postcard.
///
/// Naming kept as `bincode_*` for compatibility with existing call sites;
/// the underlying codec is postcard since the bincode crate became
/// unmaintained (RUSTSEC-2025-0141).
pub fn bincode_getstate<T: serde::Serialize>(py: Python<'_>, obj: &T) -> PyResult<Py<PyAny>> {
    let bytes =
        encode_to_vec(obj).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &bytes).into())
}

/// Deserialize a serde-compatible struct from PyBytes via postcard.
pub fn bincode_setstate<T: serde::de::DeserializeOwned>(state: &Bound<'_, PyBytes>) -> PyResult<T> {
    decode_from_slice(state.as_bytes())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

// -----------------------------------------------------------------------------
// File-load hardening: magic header + format version + size cap
// -----------------------------------------------------------------------------
//
// File layout for KNC binary artifacts written by save_bincode_to_path:
//
//   [0..4]   magic      = b"KNC1"
//   [4..6]   version    = u16 LE (currently 1)
//   [6..]    payload    = postcard encoding of T
//
// Files written by older 0.x prereleases lacked this header. They are not
// readable by post-0.1.0a1 builds — the format change is a one-time break
// tied to migrating off the unmaintained bincode crate.

const KNC_MAGIC: &[u8; 4] = b"KNC1";
const KNC_VERSION: u16 = 1;
const KNC_HEADER_LEN: usize = KNC_MAGIC.len() + 2;

/// Default cap on bincode-loaded file size (256 MiB).
///
/// Override via the `KAOS_NLP_MAX_LOAD_BYTES` env var. The cap exists so a
/// crafted header cannot trigger an unbounded allocation during deserialization.
const DEFAULT_MAX_LOAD_BYTES: u64 = 256 * 1024 * 1024;

fn max_load_bytes() -> u64 {
    std::env::var("KAOS_NLP_MAX_LOAD_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_LOAD_BYTES)
}

/// Serialize a serde-compatible struct to a file via postcard, prefixed with
/// the KNC magic + format version.
pub fn save_bincode_to_path<T: serde::Serialize>(obj: &T, path: &str) -> Result<(), String> {
    let payload = encode_to_vec(obj).map_err(|e| e.to_string())?;
    let mut buf = Vec::with_capacity(KNC_HEADER_LEN + payload.len());
    buf.extend_from_slice(KNC_MAGIC);
    buf.extend_from_slice(&KNC_VERSION.to_le_bytes());
    buf.extend_from_slice(&payload);
    std::fs::write(path, buf).map_err(|e| e.to_string())
}

/// Deserialize a serde-compatible struct from a file via postcard.
///
/// Refuses files larger than `KAOS_NLP_MAX_LOAD_BYTES` (default 256 MiB) and
/// requires the KNC magic + supported version. **Trust model:** these files
/// are still expected to come from a trusted source — the size cap and header
/// are integrity / DoS guards, not a defence against an adversarial author of
/// the underlying types.
pub fn load_bincode_from_path<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, String> {
    let metadata = std::fs::metadata(path).map_err(|e| format!("could not stat {path}: {e}"))?;
    let limit = max_load_bytes();
    if metadata.len() > limit {
        return Err(format!(
            "{path}: file size {} bytes exceeds KAOS_NLP_MAX_LOAD_BYTES limit of {limit} bytes",
            metadata.len()
        ));
    }

    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < KNC_HEADER_LEN {
        return Err(format!(
            "{path}: file too short to contain KNC header ({} bytes)",
            bytes.len()
        ));
    }
    if &bytes[0..4] != KNC_MAGIC {
        return Err(format!(
            "{path}: missing KNC magic header — file was not written by kaos-nlp-core ≥ 0.1.0a1"
        ));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != KNC_VERSION {
        return Err(format!(
            "{path}: unsupported KNC format version {version} (expected {KNC_VERSION})"
        ));
    }

    decode_from_slice(&bytes[KNC_HEADER_LEN..]).map_err(|e| e.to_string())
}
