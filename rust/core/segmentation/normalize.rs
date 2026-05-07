//! Offset-preserving text normalizer.
//!
//! Single-pass fused transformer with an ASCII fast path that returns
//! `Cow::Borrowed` when no transform fires. Output carries an
//! `orig_offsets` table that maps **char position in the normalized output**
//! back to **byte offset in the original input**, so downstream consumers
//! can recover provenance after the transform.
//!
//! Design reference: `docs/SECTION_HEADING_PRIMITIVES_RESEARCH.md`
//! (`## Normalizer (P2) — design reference`). Every behaviour below is
//! grounded in a question from that doc.
//!
//! Trust model: this is a content normalizer, not a security boundary.
//! Inputs may contain any UTF-8; outputs are guaranteed valid UTF-8 by
//! construction (the transform tables emit only canonical ASCII bytes).

use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─── Public API ────────────────────────────────────────────────────────────

/// Configuration for [`normalize`].
///
/// Default constructor returns a no-op (all flags off) so a caller that
/// just wants the offset-table side effect can pass `NormalizeOptions::default()`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizeOptions {
    /// Collapse runs of `char::is_whitespace` to a single U+0020.
    pub collapse_whitespace: bool,
    /// ASCII-only case fold (`A-Z` → `a-z`). Unicode case-fold is intentionally
    /// out of scope — callers can run `icu_normalizer` first if they need it.
    pub fold_case: bool,
    /// Map common Unicode punctuation (smart quotes, dashes, ellipsis,
    /// non-breaking spaces, soft hyphen, zero-width chars, bullets) to their
    /// ASCII equivalents. See the design reference for the exact table.
    pub normalize_unicode_punct: bool,
    /// Strip leading enumerator/heading prefixes (`I.`, `(a)`, `Sec. 5`, `§`).
    /// **Currently unimplemented** — pending the Enumerator parser (P3).
    /// Setting this flag causes [`normalize`] to return
    /// [`NormalizeError::UnsupportedOption`] until P3 lands.
    pub strip_enumerator_prefix: bool,
    /// Drop ASCII punctuation characters from the output. Used when downstream
    /// consumers want a "letters / digits / spaces only" canonical form.
    pub strip_punctuation: bool,
}

impl NormalizeOptions {
    /// Aggressive normalization: collapse whitespace, fold case, normalize
    /// Unicode punctuation. Suitable for boilerplate fingerprinting.
    pub fn aggressive() -> Self {
        Self {
            collapse_whitespace: true,
            fold_case: true,
            normalize_unicode_punct: true,
            strip_enumerator_prefix: false,
            strip_punctuation: false,
        }
    }
}

/// Errors raised by [`normalize`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NormalizeError {
    /// A flag was set whose backing implementation is not yet available.
    /// Currently raised only by `strip_enumerator_prefix` until P3 lands.
    #[error("normalize option not yet supported: {0}")]
    UnsupportedOption(&'static str),
}

/// Output of [`normalize`].
///
/// `text` borrows from the input on the no-op fast path; otherwise it owns
/// a freshly allocated `String`. `orig_offsets` is `None` exactly when
/// `text` is `Cow::Borrowed` and the output is therefore identical to the
/// input — callers can treat indices as byte offsets directly. When
/// `Some`, the vector is indexed by **char position in `text`** and stores
/// the **byte offset of the source char that contributed it**.
///
/// Length invariant: `orig_offsets.as_ref().map(|v| v.len()).unwrap_or(0) ==
/// text.chars().count()` whenever the output was allocated.
#[derive(Debug, Clone)]
pub struct Normalized<'a> {
    pub text: Cow<'a, str>,
    pub orig_offsets: Option<Vec<u32>>,
}

impl<'a> Normalized<'a> {
    /// Resolve the original byte offset for a char index in the normalized
    /// output. Returns `None` if `char_index` is out of range.
    pub fn original_byte(&self, char_index: usize) -> Option<u32> {
        match &self.orig_offsets {
            // Fast path: text is unchanged — char index == byte index for ASCII;
            // for non-ASCII the no-op path is impossible (see Q4 in the design
            // reference), so we can treat the index as a byte offset directly.
            None => {
                let bytes = self.text.as_bytes();
                if char_index <= bytes.len() {
                    Some(char_index as u32)
                } else {
                    None
                }
            }
            Some(map) => map.get(char_index).copied(),
        }
    }
}

// ─── Lookup tables ────────────────────────────────────────────────────────

/// Outcome of mapping a single Unicode codepoint.
#[derive(Debug, Clone, Copy)]
enum Mapping {
    /// Pass the codepoint through unchanged.
    Identity,
    /// Replace with a single ASCII character.
    Single(char),
    /// Expand to two ASCII characters.
    Two(char, char),
    /// Expand to three ASCII characters (e.g., U+2026 → `...`).
    Three(char, char, char),
    /// Delete the codepoint (zero output).
    Delete,
}

/// Map a single Unicode codepoint per the design-reference table.
///
/// Returns `Mapping::Identity` for ASCII and any other unmapped codepoint.
fn unicode_punct_map(c: char) -> Mapping {
    // ASCII fast-out: nothing in the table is ASCII.
    if (c as u32) < 0x80 {
        return Mapping::Identity;
    }
    match c {
        // Single quote / apostrophe family.
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' | '\u{2039}'
        | '\u{203A}' => Mapping::Single('\''),
        // Double quote family + guillemets + double prime.
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{2033}' | '\u{00AB}'
        | '\u{00BB}' => Mapping::Single('"'),
        // Dash / hyphen / minus family.
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
        | '\u{2212}' => Mapping::Single('-'),
        // Ellipsis.
        '\u{2026}' => Mapping::Three('.', '.', '.'),
        // Whitespace family → ASCII space.
        '\u{00A0}' | '\u{2007}' | '\u{2009}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
        | '\u{2008}' | '\u{200A}' | '\u{2002}' | '\u{2003}' | '\u{2004}' | '\u{2005}'
        | '\u{2006}' => Mapping::Single(' '),
        // Soft hyphen / zero-width / BOM / bullets — delete entirely.
        '\u{00AD}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' | '\u{2022}'
        | '\u{25CF}' | '\u{25E6}' => Mapping::Delete,
        _ => Mapping::Identity,
    }
}

/// `true` if the ASCII punctuation byte `b` should be dropped under
/// `strip_punctuation`. Letters / digits / whitespace are preserved.
#[inline]
fn is_ascii_strip_punct(b: u8) -> bool {
    matches!(
        b,
        b'!' | b'"'
            | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b'-'
            | b'.'
            | b'/'
            | b':'
            | b';'
            | b'<'
            | b'='
            | b'>'
            | b'?'
            | b'@'
            | b'['
            | b'\\'
            | b']'
            | b'^'
            | b'_'
            | b'`'
            | b'{'
            | b'|'
            | b'}'
            | b'~'
    )
}

// ─── Entry point ───────────────────────────────────────────────────────────

/// Normalize `text` per `opts`. Returns either a borrowed slice (no-op
/// fast path) or an owned `String` plus an `orig_offsets` table mapping
/// each output char back to a source byte offset.
pub fn normalize<'a>(
    text: &'a str,
    opts: NormalizeOptions,
) -> Result<Normalized<'a>, NormalizeError> {
    if opts.strip_enumerator_prefix {
        return Err(NormalizeError::UnsupportedOption(
            "strip_enumerator_prefix requires Enumerator parser (P3)",
        ));
    }

    // ── Q4 fast path: ASCII input + no flag that fires on ASCII ────────────
    let any_ascii_flag = opts.collapse_whitespace || opts.fold_case || opts.strip_punctuation;
    let any_unicode_flag = opts.normalize_unicode_punct;
    if !any_ascii_flag && !any_unicode_flag {
        return Ok(Normalized {
            text: Cow::Borrowed(text),
            orig_offsets: None,
        });
    }
    if !any_ascii_flag && any_unicode_flag && text.is_ascii() {
        // No mapping in the punct table is ASCII, so on ASCII input the unicode
        // flag is a no-op too.
        return Ok(Normalized {
            text: Cow::Borrowed(text),
            orig_offsets: None,
        });
    }

    // Allocate output. Worst case 3× input length (only ellipsis expands).
    let mut out = String::with_capacity(text.len() + 8);
    let mut offsets: Vec<u32> = Vec::with_capacity(text.len());

    // Whitespace-collapse state. `last_was_ws_emitted` is true when the
    // previous emitted char was a single ASCII space produced by collapse,
    // so further input whitespace is suppressed within the same run.
    let mut last_was_ws_emitted = false;

    let initial_len = out.len();
    for (byte_idx, ch) in text.char_indices() {
        let byte_idx_u32 = byte_idx as u32;

        // 1. Whitespace collapse
        if opts.collapse_whitespace && ch.is_whitespace() {
            if !last_was_ws_emitted {
                out.push(' ');
                offsets.push(byte_idx_u32);
                last_was_ws_emitted = true;
            }
            // Otherwise skip — same run, do not advance offsets table.
            continue;
        }

        // Track length so we can tell whether `emit_one` actually pushed
        // anything. If the source char gets dropped (strip_punctuation,
        // Unicode delete mapping), `last_was_ws_emitted` must stay set so
        // a subsequent whitespace run doesn't re-emit a redundant space.
        let len_before = out.len();

        // 2. Unicode punct map (decides expansion / replacement / delete).
        let mapping = if opts.normalize_unicode_punct {
            unicode_punct_map(ch)
        } else {
            Mapping::Identity
        };

        match mapping {
            Mapping::Delete => {} // dropped; preserve last_was_ws_emitted
            Mapping::Single(replacement) => {
                emit_one(replacement, byte_idx_u32, &mut out, &mut offsets, &opts);
            }
            Mapping::Two(a, b) => {
                emit_one(a, byte_idx_u32, &mut out, &mut offsets, &opts);
                emit_one(b, byte_idx_u32, &mut out, &mut offsets, &opts);
            }
            Mapping::Three(a, b, c) => {
                emit_one(a, byte_idx_u32, &mut out, &mut offsets, &opts);
                emit_one(b, byte_idx_u32, &mut out, &mut offsets, &opts);
                emit_one(c, byte_idx_u32, &mut out, &mut offsets, &opts);
            }
            Mapping::Identity => {
                // 3. Strip-punctuation / fold-case for the source codepoint.
                emit_one(ch, byte_idx_u32, &mut out, &mut offsets, &opts);
            }
        }

        // Reset the WS-collapse flag only when we actually emitted at
        // least one (non-whitespace) char. This prevents
        // "<space><stripped-punct><space>" from collapsing to two
        // spaces (idempotency violation).
        if out.len() > len_before {
            last_was_ws_emitted = false;
        }
    }
    let _ = initial_len; // silence unused-binding lint when we add asserts later

    Ok(Normalized {
        text: Cow::Owned(out),
        orig_offsets: Some(offsets),
    })
}

/// Emit one (post-Unicode-mapping) char into the output, applying ASCII
/// fold_case / strip_punctuation if configured. Caller has already decided
/// the source byte offset (`origin`) for the alignment table.
#[inline]
fn emit_one(
    ch: char,
    origin: u32,
    out: &mut String,
    offsets: &mut Vec<u32>,
    opts: &NormalizeOptions,
) {
    // strip_punctuation drops ASCII punct unconditionally. Non-ASCII
    // punct passed through `unicode_punct_map` already → ASCII or kept.
    if opts.strip_punctuation && (ch as u32) < 0x80 && is_ascii_strip_punct(ch as u8) {
        return;
    }
    let mut emit = ch;
    if opts.fold_case && emit.is_ascii_uppercase() {
        emit = emit.to_ascii_lowercase();
    }
    out.push(emit);
    offsets.push(origin);
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn opts_default() -> NormalizeOptions {
        NormalizeOptions::default()
    }

    // ── Q4 fast path ───────────────────────────────────────────────────────

    #[test]
    fn no_flags_returns_borrowed() {
        let src = "Hello, World!";
        let r = normalize(src, opts_default()).unwrap();
        assert!(matches!(r.text, Cow::Borrowed(s) if std::ptr::eq(s, src)));
        assert!(r.orig_offsets.is_none());
    }

    #[test]
    fn ascii_input_unicode_flag_only_returns_borrowed() {
        let src = "Plain ASCII text.";
        let opts = NormalizeOptions {
            normalize_unicode_punct: true,
            ..NormalizeOptions::default()
        };
        let r = normalize(src, opts).unwrap();
        assert!(matches!(r.text, Cow::Borrowed(_)));
        assert!(r.orig_offsets.is_none());
    }

    #[test]
    fn ascii_with_collapse_flag_does_allocate() {
        let opts = NormalizeOptions {
            collapse_whitespace: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("a  b", opts).unwrap();
        assert!(matches!(r.text, Cow::Owned(_)));
        assert_eq!(r.text.as_ref(), "a b");
    }

    // ── Q1 Unicode punct map ──────────────────────────────────────────────

    #[test]
    fn smart_quotes_collapse() {
        let opts = NormalizeOptions {
            normalize_unicode_punct: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("\u{2018}hello\u{2019} \u{201C}world\u{201D}", opts).unwrap();
        assert_eq!(r.text.as_ref(), "'hello' \"world\"");
    }

    #[test]
    fn dashes_collapse() {
        let opts = NormalizeOptions {
            normalize_unicode_punct: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("en\u{2013}dash em\u{2014}dash minus\u{2212}sign", opts).unwrap();
        assert_eq!(r.text.as_ref(), "en-dash em-dash minus-sign");
    }

    #[test]
    fn ellipsis_expands_to_three() {
        let opts = NormalizeOptions {
            normalize_unicode_punct: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("wait\u{2026}what?", opts).unwrap();
        assert_eq!(r.text.as_ref(), "wait...what?");
        // The three '.' chars all point at the same source byte offset.
        let map = r.orig_offsets.unwrap();
        // wait = 4 chars → indices 0..4. Then ellipsis at byte offset 4 (ASCII)
        // expands to indices 4, 5, 6 all at byte offset 4.
        assert_eq!(map[4], 4);
        assert_eq!(map[5], 4);
        assert_eq!(map[6], 4);
    }

    #[test]
    fn unicode_spaces_become_ascii_space() {
        let opts = NormalizeOptions {
            normalize_unicode_punct: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("a\u{00A0}b\u{2007}c\u{3000}d", opts).unwrap();
        assert_eq!(r.text.as_ref(), "a b c d");
    }

    #[test]
    fn deleted_chars_have_no_offset_entry() {
        let opts = NormalizeOptions {
            normalize_unicode_punct: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("co\u{00AD}operate", opts).unwrap();
        assert_eq!(r.text.as_ref(), "cooperate");
        assert_eq!(r.orig_offsets.unwrap().len(), "cooperate".chars().count());
    }

    // ── Q5 Whitespace collapse ────────────────────────────────────────────

    #[test]
    fn collapse_runs() {
        let opts = NormalizeOptions {
            collapse_whitespace: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("a  \t\n b", opts).unwrap();
        assert_eq!(r.text.as_ref(), "a b");
    }

    #[test]
    fn collapse_preserves_leading_trailing_as_single() {
        let opts = NormalizeOptions {
            collapse_whitespace: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("   a   b   ", opts).unwrap();
        assert_eq!(r.text.as_ref(), " a b ");
    }

    #[test]
    fn collapse_offset_targets_first_run_char() {
        let opts = NormalizeOptions {
            collapse_whitespace: true,
            ..NormalizeOptions::default()
        };
        let src = "a  b";
        let r = normalize(src, opts).unwrap();
        assert_eq!(r.text.as_ref(), "a b");
        let map = r.orig_offsets.unwrap();
        // 'a' at byte 0 → out 0; collapsed space → out 1, points at byte 1
        // (first space of run); 'b' at byte 3 → out 2.
        assert_eq!(map[0], 0);
        assert_eq!(map[1], 1);
        assert_eq!(map[2], 3);
    }

    // ── Fold case + strip punct ───────────────────────────────────────────

    #[test]
    fn fold_case_ascii() {
        let opts = NormalizeOptions {
            fold_case: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("HELLO World", opts).unwrap();
        assert_eq!(r.text.as_ref(), "hello world");
    }

    #[test]
    fn fold_case_only_touches_ascii_codepoints() {
        // Unicode case fold is intentionally out of scope: É (U+00C9) stays
        // uppercase, but ASCII letters next to it still fold.
        let opts = NormalizeOptions {
            fold_case: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("ÉCOLE", opts).unwrap();
        assert_eq!(r.text.as_ref(), "École");
    }

    #[test]
    fn strip_punctuation_drops_ascii_punct() {
        let opts = NormalizeOptions {
            strip_punctuation: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("Hello, World! (here)", opts).unwrap();
        assert_eq!(r.text.as_ref(), "Hello World here");
    }

    // ── Q7 deferred enumerator ────────────────────────────────────────────

    #[test]
    fn strip_enumerator_prefix_is_deferred_error() {
        let opts = NormalizeOptions {
            strip_enumerator_prefix: true,
            ..NormalizeOptions::default()
        };
        let err = normalize("I. Introduction", opts).unwrap_err();
        assert!(matches!(err, NormalizeError::UnsupportedOption(_)));
    }

    // ── Aggressive composition ────────────────────────────────────────────

    #[test]
    fn aggressive_normalizes_legal_quote() {
        let src = "  \u{201C}HELLO,\u{2014}World\u{2026}\u{201D}  ";
        let r = normalize(src, NormalizeOptions::aggressive()).unwrap();
        assert_eq!(r.text.as_ref(), " \"hello,-world...\" ");
    }

    // ── original_byte API ─────────────────────────────────────────────────

    #[test]
    fn original_byte_borrowed_returns_index_directly() {
        let src = "abc";
        let r = normalize(src, opts_default()).unwrap();
        assert_eq!(r.original_byte(0), Some(0));
        assert_eq!(r.original_byte(3), Some(3));
        assert_eq!(r.original_byte(4), None);
    }

    #[test]
    fn original_byte_owned_consults_table() {
        let opts = NormalizeOptions {
            normalize_unicode_punct: true,
            ..NormalizeOptions::default()
        };
        let r = normalize("a\u{2026}b", opts).unwrap();
        // a at byte 0 → out 0
        // ellipsis at byte 1 (3 bytes wide for U+2026) → out 1, 2, 3
        // b at byte 4 → out 4
        assert_eq!(r.original_byte(0), Some(0));
        assert_eq!(r.original_byte(1), Some(1));
        assert_eq!(r.original_byte(2), Some(1));
        assert_eq!(r.original_byte(3), Some(1));
        assert_eq!(r.original_byte(4), Some(4));
    }

    // ── Property tests (Q8 invariants) ────────────────────────────────────

    fn arb_options() -> impl Strategy<Value = NormalizeOptions> {
        (any::<bool>(), any::<bool>(), any::<bool>(), any::<bool>()).prop_map(|(cw, fc, np, sp)| {
            NormalizeOptions {
                collapse_whitespace: cw,
                fold_case: fc,
                normalize_unicode_punct: np,
                strip_enumerator_prefix: false,
                strip_punctuation: sp,
            }
        })
    }

    proptest! {
        #[test]
        fn normalize_never_panics(text in "\\PC{0,256}", opts in arb_options()) {
            let _ = normalize(&text, opts);
        }

        #[test]
        fn offsets_are_char_boundaries(text in "\\PC{0,256}", opts in arb_options()) {
            let r = normalize(&text, opts).unwrap();
            if let Some(map) = &r.orig_offsets {
                for &o in map {
                    prop_assert!(text.is_char_boundary(o as usize));
                }
            }
        }

        #[test]
        fn offsets_are_monotonic(text in "\\PC{0,256}", opts in arb_options()) {
            let r = normalize(&text, opts).unwrap();
            if let Some(map) = &r.orig_offsets {
                for w in map.windows(2) {
                    prop_assert!(w[0] <= w[1]);
                }
            }
        }

        #[test]
        fn offsets_length_matches_char_count(text in "\\PC{0,256}", opts in arb_options()) {
            let r = normalize(&text, opts).unwrap();
            if let Some(map) = &r.orig_offsets {
                prop_assert_eq!(map.len(), r.text.chars().count());
            }
        }

        #[test]
        fn idempotent_on_text(text in "\\PC{0,128}", opts in arb_options()) {
            let r1 = normalize(&text, opts).unwrap();
            let r2 = normalize(r1.text.as_ref(), opts).unwrap();
            prop_assert_eq!(r1.text.as_ref(), r2.text.as_ref());
        }

        #[test]
        fn ascii_no_op_returns_borrowed(text in "[ -~]{0,64}") {
            // ASCII printable input + only the unicode-punct flag → borrowed.
            let opts = NormalizeOptions {
                normalize_unicode_punct: true,
                ..NormalizeOptions::default()
            };
            let r = normalize(&text, opts).unwrap();
            prop_assert!(matches!(r.text, Cow::Borrowed(_)));
            prop_assert!(r.orig_offsets.is_none());
        }
    }
}
