//! Shared utilities for PyO3 bindings.
//!
//! Common patterns used across multiple binding modules:
//! - Byte-to-character offset conversion for returning text positions to Python
//! - Bincode 2.x pickle helpers for serde-compatible types
//! - File-load helpers with magic header, format version, and size cap

use std::io::Read;

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
pub(crate) fn encode_to_vec<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, postcard::Error> {
    postcard::to_allocvec(value)
}

/// Deserialize a serde-compatible value from a slice via postcard.
pub(crate) fn decode_from_slice<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
) -> Result<T, postcard::Error> {
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
//   [4..6]   version    = u16 LE
//   [6..]    payload    = postcard encoding of T (v1: raw, v2: zstd-compressed)
//
// Files written by older 0.x prereleases lacked this header. They are not
// readable by post-0.1.0a1 builds — the format change is a one-time break
// tied to migrating off the unmaintained bincode crate.
//
// v1 -> v2 rationale: the OpenGloss lexicon ships at 238 MB raw postcard.
// zstd-compressing the payload trims that to ~85 MB at level 3, ~70 MB at
// level 19. Decompression is in the noise (~150-300 ms for the lexicon on
// modern CPUs). v1 files remain readable so users who already have a cached
// uncompressed lexicon at ~/.cache/kaos/ keep working.

pub(crate) const KNC_MAGIC: &[u8; 4] = b"KNC1";
pub(crate) const KNC_VERSION_RAW: u16 = 1;
pub(crate) const KNC_VERSION_ZSTD: u16 = 2;
/// KNC v3 — Lexicon-specific format with global string interning + dropped
/// `Sense.definition`. Emitted/accepted only by the Lexicon save/load path
/// (`save_lexicon_to_path` / `load_lexicon_from_path`). Other consumers
/// (`InvertedIndex`, `SimilarityMatrix`, hash structures) do not benefit
/// from string interning and continue to use v2 (zstd-compressed postcard).
pub(crate) const KNC_VERSION_INTERN: u16 = 3;
/// Version emitted by `save_bincode_to_path` (generic writer — v2). The
/// Lexicon path emits v3 separately.
const KNC_VERSION_CURRENT: u16 = KNC_VERSION_ZSTD;
pub(crate) const KNC_HEADER_LEN: usize = KNC_MAGIC.len() + 2;

/// Default cap on bincode-loaded file size (256 MiB).
///
/// Override via the `KAOS_NLP_MAX_LOAD_BYTES` env var. The cap exists so a
/// crafted header cannot trigger an unbounded allocation during deserialization.
const DEFAULT_MAX_LOAD_BYTES: u64 = 256 * 1024 * 1024;

/// Default cap on the **decompressed** payload size for v2 (zstd) files.
/// 1 GiB. Tighter than the gzip-bomb-style 1 TB defaults zstd allows by
/// default; the canonical OpenGloss lexicon is ~250 MB decompressed and
/// no save site in this repo produces anything larger today.
const DEFAULT_MAX_DECOMPRESSED_BYTES: u64 = 1024 * 1024 * 1024;

/// zstd compression level used by `save_bincode_to_path`. Level 3 is the
/// libzstd default — fast (~200 MB/s compress) with reasonable ratio.
/// The OpenGloss build script overrides via `KAOS_NLP_ZSTD_LEVEL=19` for
/// the canonical asset (smaller wire size; build-time-only cost).
const DEFAULT_ZSTD_LEVEL: i32 = 3;

pub(crate) fn max_load_bytes() -> u64 {
    std::env::var("KAOS_NLP_MAX_LOAD_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_LOAD_BYTES)
}

pub(crate) fn max_decompressed_bytes() -> u64 {
    std::env::var("KAOS_NLP_MAX_DECOMPRESSED_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_DECOMPRESSED_BYTES)
}

pub(crate) fn zstd_level() -> i32 {
    std::env::var("KAOS_NLP_ZSTD_LEVEL")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(DEFAULT_ZSTD_LEVEL)
}

/// Serialize a serde-compatible struct to a file via postcard, prefixed with
/// the KNC magic + format version. Writes KNC v2 (zstd-compressed payload).
pub fn save_bincode_to_path<T: serde::Serialize>(obj: &T, path: &str) -> Result<(), String> {
    let raw = encode_to_vec(obj).map_err(|e| e.to_string())?;
    let compressed =
        zstd::encode_all(raw.as_slice(), zstd_level()).map_err(|e| format!("zstd encode: {e}"))?;
    let mut buf = Vec::with_capacity(KNC_HEADER_LEN + compressed.len());
    buf.extend_from_slice(KNC_MAGIC);
    buf.extend_from_slice(&KNC_VERSION_CURRENT.to_le_bytes());
    buf.extend_from_slice(&compressed);
    std::fs::write(path, buf).map_err(|e| e.to_string())
}

/// Deserialize a serde-compatible struct from a file via postcard.
///
/// Accepts both KNC v1 (raw postcard) and KNC v2 (zstd-compressed postcard).
/// Refuses files larger than `KAOS_NLP_MAX_LOAD_BYTES` (default 256 MiB on
/// disk) and decompressed payloads larger than `KAOS_NLP_MAX_DECOMPRESSED_BYTES`
/// (default 1 GiB) for v2. **Trust model:** these files are still expected to
/// come from a trusted source — the size caps and header are integrity / DoS
/// guards, not a defence against an adversarial author of the underlying types.
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
    let payload = &bytes[KNC_HEADER_LEN..];
    match version {
        KNC_VERSION_RAW => decode_from_slice(payload).map_err(|e| e.to_string()),
        KNC_VERSION_ZSTD => {
            let decompressed = zstd_decompress_capped(payload, path)?;
            decode_from_slice(&decompressed).map_err(|e| e.to_string())
        }
        KNC_VERSION_INTERN => Err(format!(
            "{path}: KNC v{version} is a Lexicon-specific format and cannot be loaded via the generic path — load it via Lexicon.load() instead"
        )),
        _ => Err(format!(
            "{path}: unsupported KNC format version {version} (this build supports {KNC_VERSION_RAW}, {KNC_VERSION_ZSTD}, and Lexicon-only {KNC_VERSION_INTERN})"
        )),
    }
}

/// Decompress a zstd payload with the configured size cap.
///
/// Caps the decompressed size before decoding to defend against a crafted
/// compressed payload that explodes to many GB. Reads at most `cap + 1` bytes
/// from the streaming decoder; if we got more than `cap`, the source overflowed
/// and we reject. Shared by the generic v2 path and the Lexicon-specific v3 path.
pub(crate) fn zstd_decompress_capped(payload: &[u8], path: &str) -> Result<Vec<u8>, String> {
    let cap = max_decompressed_bytes();
    let decoder = zstd::Decoder::new(payload).map_err(|e| format!("zstd decode init: {e}"))?;
    let mut limited = decoder.take(cap.saturating_add(1));
    let mut decompressed = Vec::with_capacity(payload.len() * 4);
    std::io::copy(&mut limited, &mut decompressed).map_err(|e| format!("zstd decode: {e}"))?;
    if (decompressed.len() as u64) > cap {
        return Err(format!(
            "{path}: decompressed payload exceeds KAOS_NLP_MAX_DECOMPRESSED_BYTES limit of {cap} bytes"
        ));
    }
    Ok(decompressed)
}

/// Read + validate KNC header from a file. Returns (version, payload bytes).
///
/// Used by callers that need to dispatch on KNC version before decoding (e.g.
/// the Lexicon save/load path which handles v3 separately from v1/v2).
pub(crate) fn read_knc_header(path: &str) -> Result<(u16, Vec<u8>), String> {
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
    let payload = bytes[KNC_HEADER_LEN..].to_vec();
    Ok((version, payload))
}
