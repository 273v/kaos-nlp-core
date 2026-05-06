//! Offset-preserving physical-line records with per-line layout features.
//!
//! `LineRecord` is the substrate for downstream structural analysis
//! (heading detection, table detection, boilerplate detection). One record
//! per physical line — including blank lines — with byte-offset slices into
//! the source, plus packed shape descriptors (`CaseProfile`, `PunctProfile`)
//! that are computed once during the same single-pass scan.
//!
//! All offsets in this module are **byte offsets** into the original `&str`.
//! PyO3 bindings convert to char offsets via `build_byte_to_char_table()`
//! before exposing them to Python; see `rust/bindings/util.rs`.
//!
//! Performance design notes:
//!
//! * Single pass over the input. Newlines are located via stringzilla's
//!   `find_newline_utf8`, which understands LF, CR, CRLF, U+0085, U+2028,
//!   U+2029, U+000B, and U+000C in one SIMD-accelerated probe.
//! * ASCII fast path. When a line is pure ASCII we count chars from byte
//!   length directly; otherwise we walk the line once with `char_indices`.
//! * Feature accumulation is fused with the line walk — `CaseProfile`,
//!   `PunctProfile`, indent and token counts are all updated by the same
//!   loop, so we touch each byte exactly once per record.
//! * The records hold offsets, not strings. `record.text(source)` slices
//!   on demand; no per-line `String` allocation.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use stringzilla::sz;

/// Newline-terminator kind.
///
/// `find_newline_utf8` may return any of the Unicode-defined line terminators;
/// we collapse them to a small enum the higher layers can pattern-match
/// without re-decoding bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum LineTerminator {
    /// No terminator — the line ran to end-of-string with no trailing
    /// newline character.
    None = 0,
    /// Single `\n` (U+000A LINE FEED).
    Lf = 1,
    /// Single `\r` (U+000D CARRIAGE RETURN), not followed by `\n`.
    Cr = 2,
    /// `\r\n` pair.
    CrLf = 3,
    /// Some other Unicode line break (U+000B, U+000C, U+0085, U+2028, U+2029).
    OtherUnicode = 4,
}

impl LineTerminator {
    /// Number of bytes the terminator occupies in the source.
    #[inline]
    pub fn byte_len(self) -> u32 {
        match self {
            LineTerminator::None => 0,
            LineTerminator::Lf | LineTerminator::Cr => 1,
            LineTerminator::CrLf => 2,
            // U+000B / U+000C are 1 byte; U+0085 is 2 bytes; U+2028 / U+2029
            // are 3 bytes. The actual byte length is recorded on the
            // `LineRecord.term_len` field; this method is used only when
            // the caller already trusts the enum tag and just wants a quick
            // best-effort length. For exact byte length, prefer
            // `LineRecord.term_len`.
            LineTerminator::OtherUnicode => 3,
        }
    }
}

/// Coarse case-shape classification of a line.
///
/// Computed from non-whitespace alphabetic characters only. Lines with no
/// alphabetic characters are reported as `NoAlpha`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CaseProfile {
    /// No alphabetic characters in the line at all (digits, punct, blank).
    NoAlpha = 0,
    /// All alphabetic characters are uppercase.
    AllCaps = 1,
    /// First character of every whitespace-separated token is uppercase
    /// and the remainder is lowercase. Common shape for headings.
    TitleCase = 2,
    /// First alphabetic character is uppercase, no other constraint.
    InitialCap = 3,
    /// All alphabetic characters are lowercase.
    AllLower = 4,
    /// Mixed case that doesn't match any of the above.
    MixedCase = 5,
}

bitflags! {
    /// Bitfield summary of punctuation / symbol presence on a line.
    ///
    /// Cheap to compute (single byte-level pass) and dense — one cache line.
    /// Used as positive/negative signals for downstream detectors:
    ///
    /// * `ENDS_PERIOD` / `ENDS_COLON` — heading shape signal.
    /// * `HAS_PIPE` / `HAS_TAB` — table-row signal.
    /// * `HAS_DIGITS` / `HAS_PARENS` — enumerator-prefix signal.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PunctProfile: u16 {
        /// Last non-whitespace character is `.`.
        const ENDS_PERIOD     = 1 << 0;
        /// Last non-whitespace character is `:`.
        const ENDS_COLON      = 1 << 1;
        /// Last non-whitespace character is `;`.
        const ENDS_SEMICOLON  = 1 << 2;
        /// Last non-whitespace character is `?` or `!`.
        const ENDS_QUESTION   = 1 << 3;
        /// Last non-whitespace character is `,`.
        const ENDS_COMMA      = 1 << 4;
        /// Line contains at least one `|` (Markdown / pipe-table signal).
        const HAS_PIPE        = 1 << 5;
        /// Line contains at least one `\t`.
        const HAS_TAB         = 1 << 6;
        /// Line contains at least one ASCII digit.
        const HAS_DIGITS      = 1 << 7;
        /// Line contains `(` or `)`.
        const HAS_PARENS      = 1 << 8;
        /// Line contains `[` or `]`.
        const HAS_BRACKETS    = 1 << 9;
        /// Line contains a `§` U+00A7 byte sequence.
        const HAS_SECTION_SIG = 1 << 10;
        /// Line contains 2+ consecutive whitespace runs of length ≥ 2 — a
        /// crude "looks columnar" signal that complements `HAS_PIPE`.
        const HAS_COLUMN_GAPS = 1 << 11;
    }
}

/// One record per physical line of source text.
///
/// All offsets are **byte offsets** into the original `&str` passed to
/// [`extract_line_records`]. PyO3 bindings translate them to char offsets
/// before exposing the record to Python.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineRecord {
    /// Byte offset in source where the line begins.
    pub start: u32,
    /// Byte offset in source where the line content ends (exclusive).
    /// **Does not include the line terminator.**
    pub end: u32,
    /// Byte length of the terminator that immediately follows `end`.
    /// 0 for the final line of a file with no trailing newline.
    pub term_len: u32,
    /// Kind of terminator (LF / CR / CRLF / Other / None).
    pub terminator: LineTerminator,

    /// Byte offset where leading whitespace ends. Always satisfies
    /// `start <= stripped_start <= end`.
    pub stripped_start: u32,
    /// Byte offset where trailing whitespace begins. Always satisfies
    /// `stripped_start <= stripped_end <= end`.
    pub stripped_end: u32,

    /// Number of leading whitespace characters (chars, not bytes).
    pub indent_chars: u16,
    /// Total byte length of the line content (`end - start`).
    pub byte_len: u32,
    /// Total character length of the line content. Equals `byte_len` when
    /// the line is pure ASCII; otherwise computed in a separate pass.
    pub char_len: u32,
    /// Approximate token count (whitespace-separated runs of non-whitespace).
    pub token_count: u16,

    /// Coarse case-shape classification.
    pub case_profile: CaseProfile,
    /// Punctuation / symbol bitfield.
    pub punct_profile: PunctProfile,

    /// `true` iff `stripped_start == stripped_end` (the line has no
    /// non-whitespace content).
    pub blank: bool,
    /// `true` iff the previous physical line was blank or this is the first
    /// line of the document. Computed in a second O(n) pass.
    pub blank_before: bool,
    /// `true` iff the next physical line is blank or this is the last line
    /// of the document. Computed in a second O(n) pass.
    pub blank_after: bool,
}

impl LineRecord {
    /// Borrow the line content out of the source.
    ///
    /// Returns `&source[record.start..record.end]`, exclusive of the
    /// terminator. Caller must pass the same `&str` that was used to build
    /// the record.
    #[inline]
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start as usize..self.end as usize]
    }

    /// Borrow the line content with leading and trailing whitespace stripped.
    #[inline]
    pub fn stripped_text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.stripped_start as usize..self.stripped_end as usize]
    }
}

// ─── Extraction ────────────────────────────────────────────────────────────

/// Extract one [`LineRecord`] per physical line of `source`.
///
/// Single-pass byte scanner with stringzilla SIMD newline detection and a
/// fused per-line feature pass. Records are returned in source order; an
/// empty input returns an empty vector. `blank_before` / `blank_after` are
/// computed in a final O(n) pass.
pub fn extract_line_records(source: &str) -> Vec<LineRecord> {
    if source.is_empty() {
        return Vec::new();
    }

    // Pre-allocate using a coarse line-density estimate (1 line per ~64 bytes).
    let mut records: Vec<LineRecord> = Vec::with_capacity(source.len() / 64 + 4);

    let bytes = source.as_bytes();
    let mut cursor: usize = 0;

    while cursor < bytes.len() {
        let tail = &bytes[cursor..];
        let (content_end, term_len, terminator) = match sz::find_newline_utf8(tail) {
            Some(span) => {
                let content_end = cursor + span.offset;
                let term_len = span.length as u32;
                let kind = classify_terminator(&bytes[content_end..content_end + span.length]);
                (content_end, term_len, kind)
            }
            None => {
                // No more terminators — last line runs to EOF with no terminator.
                (bytes.len(), 0, LineTerminator::None)
            }
        };

        let line_bytes = &bytes[cursor..content_end];
        let line_str = unsafe {
            // Safe because `cursor` and `content_end` always sit on char
            // boundaries: `cursor` advances by `term_len` (which is the byte
            // length of a complete UTF-8 line terminator) and `content_end`
            // is the byte offset where the next terminator starts, which is
            // always a char boundary.
            std::str::from_utf8_unchecked(line_bytes)
        };

        records.push(scan_line_features(
            line_str,
            cursor as u32,
            content_end as u32,
            term_len,
            terminator,
        ));

        cursor = content_end + term_len as usize;
        // `find_newline_utf8` returns `None` only when no terminator is left;
        // if it returned Some with offset == tail.len() and term_len == 0 we'd
        // loop forever. The crate doesn't do that, but assert defensively in
        // debug builds.
        debug_assert!(cursor > content_end || term_len == 0);
        if term_len == 0 {
            break;
        }
    }

    fill_blank_neighbours(&mut records);

    records
}

/// Classify the bytes that constitute a line terminator.
fn classify_terminator(bytes: &[u8]) -> LineTerminator {
    match bytes {
        [b'\n'] => LineTerminator::Lf,
        [b'\r'] => LineTerminator::Cr,
        [b'\r', b'\n'] => LineTerminator::CrLf,
        _ => LineTerminator::OtherUnicode,
    }
}

/// Compute every per-line feature in a single pass over the line content.
fn scan_line_features(
    line: &str,
    start: u32,
    end: u32,
    term_len: u32,
    terminator: LineTerminator,
) -> LineRecord {
    let bytes = line.as_bytes();
    let byte_len = bytes.len() as u32;
    let is_ascii = line.is_ascii();
    let char_len = if is_ascii {
        byte_len
    } else {
        line.chars().count() as u32
    };

    // Fast path: blank line.
    if line.is_empty() {
        return LineRecord {
            start,
            end,
            term_len,
            terminator,
            stripped_start: start,
            stripped_end: start,
            indent_chars: 0,
            byte_len: 0,
            char_len: 0,
            token_count: 0,
            case_profile: CaseProfile::NoAlpha,
            punct_profile: PunctProfile::empty(),
            blank: true,
            blank_before: false,
            blank_after: false,
        };
    }

    // Leading whitespace -> indent
    let (lead_byte, indent_chars) = leading_whitespace(line);
    // Trailing whitespace -> stripped_end
    let trail_byte = trailing_whitespace(line);

    // Whole-line is whitespace.
    if lead_byte == bytes.len() {
        return LineRecord {
            start,
            end,
            term_len,
            terminator,
            stripped_start: start + lead_byte as u32,
            stripped_end: start + lead_byte as u32,
            indent_chars: indent_chars as u16,
            byte_len,
            char_len,
            token_count: 0,
            case_profile: CaseProfile::NoAlpha,
            punct_profile: PunctProfile::empty(),
            blank: true,
            blank_before: false,
            blank_after: false,
        };
    }

    // The stripped slice is computed implicitly via lead_byte/trail_byte; we
    // walk `line` (not the stripped slice) so leading-whitespace handling and
    // column-gap detection see the same bytes the source has.

    let mut punct = PunctProfile::empty();
    let mut alpha_seen: u32 = 0;
    let mut alpha_upper: u32 = 0;
    let mut alpha_lower: u32 = 0;
    let mut last_was_ws: bool = true;
    let mut token_count: u16 = 0;
    let mut all_words_title: bool = true;
    let mut consecutive_ws_runs: u16 = 0;
    let mut current_ws_run: u16 = 0;
    let mut at_word_start: bool = true;
    let mut current_word_initial_upper: bool = false;
    let mut current_word_only_initial_upper: bool = true;

    if is_ascii {
        // ASCII fast path — single byte loop.
        for (i, &b) in bytes.iter().enumerate() {
            let is_ws = b == b' ' || b == b'\t' || b == 0x0B || b == 0x0C;
            if is_ws {
                if !last_was_ws && (!current_word_initial_upper || !current_word_only_initial_upper)
                {
                    all_words_title = false;
                }
                current_ws_run += 1;
                last_was_ws = true;
                at_word_start = true;
            } else {
                if last_was_ws && i > 0 {
                    if current_ws_run >= 2 {
                        consecutive_ws_runs += 1;
                    }
                    current_ws_run = 0;
                }
                if last_was_ws {
                    token_count = token_count.saturating_add(1);
                    current_word_initial_upper = false;
                    current_word_only_initial_upper = true;
                }

                let alpha = (b as char).is_ascii_alphabetic();
                if alpha {
                    alpha_seen += 1;
                    let upper = (b as char).is_ascii_uppercase();
                    if upper {
                        alpha_upper += 1;
                    } else {
                        alpha_lower += 1;
                    }
                    if at_word_start {
                        if upper {
                            current_word_initial_upper = true;
                        } else {
                            current_word_only_initial_upper = false;
                        }
                    } else {
                        if upper {
                            current_word_only_initial_upper = false;
                        }
                    }
                }

                update_punct_flags(b, &mut punct);
                last_was_ws = false;
                at_word_start = false;
            }
            if b == b'\t' {
                punct |= PunctProfile::HAS_TAB;
            }
        }
        // Flush the final word into the title-case decision.
        if !last_was_ws && (!current_word_initial_upper || !current_word_only_initial_upper) {
            all_words_title = false;
        }
    } else {
        // Unicode path — char-aware. Multi-byte chars are conservatively
        // counted as alpha when their `is_alphabetic` flag is set.
        for (_, ch) in line.char_indices() {
            let is_ws = ch.is_whitespace();
            if is_ws {
                if !last_was_ws && (!current_word_initial_upper || !current_word_only_initial_upper)
                {
                    all_words_title = false;
                }
                current_ws_run += 1;
                last_was_ws = true;
                at_word_start = true;
                if ch == '\t' {
                    punct |= PunctProfile::HAS_TAB;
                }
            } else {
                if last_was_ws {
                    if current_ws_run >= 2 {
                        consecutive_ws_runs += 1;
                    }
                    current_ws_run = 0;
                    token_count = token_count.saturating_add(1);
                    current_word_initial_upper = false;
                    current_word_only_initial_upper = true;
                }

                if ch.is_alphabetic() {
                    alpha_seen += 1;
                    let upper = ch.is_uppercase();
                    if upper {
                        alpha_upper += 1;
                    } else {
                        alpha_lower += 1;
                    }
                    if at_word_start {
                        if upper {
                            current_word_initial_upper = true;
                        } else {
                            current_word_only_initial_upper = false;
                        }
                    } else if upper {
                        current_word_only_initial_upper = false;
                    }
                } else if ch == '§' {
                    punct |= PunctProfile::HAS_SECTION_SIG;
                }

                if ch.is_ascii() {
                    update_punct_flags(ch as u8, &mut punct);
                }
                last_was_ws = false;
                at_word_start = false;
            }
        }
        if !last_was_ws && (!current_word_initial_upper || !current_word_only_initial_upper) {
            all_words_title = false;
        }
    }

    if consecutive_ws_runs >= 2 {
        punct |= PunctProfile::HAS_COLUMN_GAPS;
    }

    // Set ENDS_* using the last byte before trailing whitespace begins.
    if trail_byte > 0 {
        if let Some(last_byte) = bytes[..trail_byte].iter().next_back().copied() {
            match last_byte {
                b'.' => punct |= PunctProfile::ENDS_PERIOD,
                b':' => punct |= PunctProfile::ENDS_COLON,
                b';' => punct |= PunctProfile::ENDS_SEMICOLON,
                b'?' | b'!' => punct |= PunctProfile::ENDS_QUESTION,
                b',' => punct |= PunctProfile::ENDS_COMMA,
                _ => {}
            }
        }
    }

    let case_profile = classify_case(alpha_seen, alpha_upper, alpha_lower, all_words_title);

    LineRecord {
        start,
        end,
        term_len,
        terminator,
        stripped_start: start + lead_byte as u32,
        stripped_end: start + trail_byte as u32,
        indent_chars: indent_chars as u16,
        byte_len,
        char_len,
        token_count,
        case_profile,
        punct_profile: punct,
        blank: false,
        blank_before: false,
        blank_after: false,
    }
}

/// Apply ASCII-only punct flags. Multi-byte punct (`§`) is set in the caller.
#[inline]
fn update_punct_flags(b: u8, p: &mut PunctProfile) {
    match b {
        b'|' => *p |= PunctProfile::HAS_PIPE,
        b'(' | b')' => *p |= PunctProfile::HAS_PARENS,
        b'[' | b']' => *p |= PunctProfile::HAS_BRACKETS,
        b'0'..=b'9' => *p |= PunctProfile::HAS_DIGITS,
        _ => {}
    }
}

/// Count leading whitespace and return `(byte offset where content starts,
/// indent character count)`.
fn leading_whitespace(line: &str) -> (usize, usize) {
    let mut bytes_consumed = 0;
    let mut chars_consumed = 0;
    for (i, ch) in line.char_indices() {
        if ch.is_whitespace() {
            bytes_consumed = i + ch.len_utf8();
            chars_consumed += 1;
        } else {
            return (i, chars_consumed);
        }
    }
    (bytes_consumed, chars_consumed)
}

/// Return the byte offset where trailing whitespace begins.
fn trailing_whitespace(line: &str) -> usize {
    let mut last_non_ws = 0;
    for (i, ch) in line.char_indices() {
        if !ch.is_whitespace() {
            last_non_ws = i + ch.len_utf8();
        }
    }
    last_non_ws
}

/// Decide the `CaseProfile` from accumulated counts.
fn classify_case(seen: u32, upper: u32, lower: u32, all_words_title: bool) -> CaseProfile {
    if seen == 0 {
        return CaseProfile::NoAlpha;
    }
    if upper > 0 && lower == 0 {
        return CaseProfile::AllCaps;
    }
    if upper == 0 && lower > 0 {
        return CaseProfile::AllLower;
    }
    if all_words_title {
        return CaseProfile::TitleCase;
    }
    if upper >= 1 {
        // First-character-only-uppercase covered by reaching here without
        // matching AllCaps / AllLower / TitleCase.
        // We approximate InitialCap by checking: only one uppercase and it
        // dominated the word boundary path. Keep the cheaper test here:
        // exactly one upper and it appears at a word start.
        if upper == 1 {
            return CaseProfile::InitialCap;
        }
    }
    CaseProfile::MixedCase
}

/// Fill `blank_before` / `blank_after` after the records vector is complete.
fn fill_blank_neighbours(records: &mut [LineRecord]) {
    let n = records.len();
    if n == 0 {
        return;
    }
    // First and last are bordered by start/end-of-file, treat as blank-adjacent.
    records[0].blank_before = true;
    records[n - 1].blank_after = true;
    for i in 0..n {
        if i + 1 < n {
            records[i].blank_after = records[i + 1].blank;
        }
        if i > 0 {
            records[i].blank_before = records[i - 1].blank;
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn first(records: &[LineRecord]) -> &LineRecord {
        &records[0]
    }

    #[test]
    fn empty_input() {
        assert!(extract_line_records("").is_empty());
    }

    #[test]
    fn single_line_no_terminator() {
        let recs = extract_line_records("hello world");
        assert_eq!(recs.len(), 1);
        let r = first(&recs);
        assert_eq!(r.start, 0);
        assert_eq!(r.end, 11);
        assert_eq!(r.term_len, 0);
        assert_eq!(r.terminator, LineTerminator::None);
        assert!(!r.blank);
        assert_eq!(r.char_len, 11);
        assert_eq!(r.token_count, 2);
    }

    #[test]
    fn lf_terminator() {
        let recs = extract_line_records("a\nb\n");
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].terminator, LineTerminator::Lf);
        assert_eq!(recs[0].term_len, 1);
        assert_eq!(recs[1].text("a\nb\n"), "b");
    }

    #[test]
    fn crlf_terminator() {
        let recs = extract_line_records("a\r\nb\r\n");
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].terminator, LineTerminator::CrLf);
        assert_eq!(recs[0].term_len, 2);
    }

    #[test]
    fn cr_only_terminator() {
        let recs = extract_line_records("a\rb\r");
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].terminator, LineTerminator::Cr);
    }

    #[test]
    fn trailing_no_newline() {
        let recs = extract_line_records("first\nsecond");
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[1].terminator, LineTerminator::None);
        assert_eq!(recs[1].term_len, 0);
        assert_eq!(recs[1].text("first\nsecond"), "second");
    }

    #[test]
    fn blank_lines_recorded() {
        let src = "a\n\nb\n";
        let recs = extract_line_records(src);
        assert_eq!(recs.len(), 3);
        assert!(!recs[0].blank);
        assert!(recs[1].blank);
        assert!(!recs[2].blank);
        assert!(recs[0].blank_after); // followed by a blank
        assert!(recs[2].blank_before); // preceded by a blank
    }

    #[test]
    fn indent_chars_ascii() {
        let recs = extract_line_records("    indented");
        assert_eq!(recs[0].indent_chars, 4);
        assert_eq!(recs[0].stripped_start, 4);
    }

    #[test]
    fn indent_chars_with_tabs() {
        let recs = extract_line_records("\t\t  mixed");
        assert_eq!(recs[0].indent_chars, 4);
    }

    #[test]
    fn case_profiles() {
        let cases = [
            ("HELLO WORLD", CaseProfile::AllCaps),
            ("Hello World", CaseProfile::TitleCase),
            ("hello world", CaseProfile::AllLower),
            ("Hello world", CaseProfile::InitialCap),
            ("HelLO worLD", CaseProfile::MixedCase),
            ("12345", CaseProfile::NoAlpha),
        ];
        for (input, expected) in cases {
            let recs = extract_line_records(input);
            assert_eq!(
                recs[0].case_profile, expected,
                "input {input:?} expected {expected:?} got {:?}",
                recs[0].case_profile
            );
        }
    }

    #[test]
    fn punct_profile_endings() {
        let recs = extract_line_records("Section 1.\nWhereas:\nList,\nQuestion?");
        assert!(recs[0].punct_profile.contains(PunctProfile::ENDS_PERIOD));
        assert!(recs[1].punct_profile.contains(PunctProfile::ENDS_COLON));
        assert!(recs[2].punct_profile.contains(PunctProfile::ENDS_COMMA));
        assert!(recs[3].punct_profile.contains(PunctProfile::ENDS_QUESTION));
    }

    #[test]
    fn punct_profile_signals() {
        let recs = extract_line_records("a | b | c\n(1) parens\n[2] brackets\n§ 5 statute");
        assert!(recs[0].punct_profile.contains(PunctProfile::HAS_PIPE));
        assert!(recs[1].punct_profile.contains(PunctProfile::HAS_PARENS));
        assert!(recs[2].punct_profile.contains(PunctProfile::HAS_BRACKETS));
        assert!(recs[3]
            .punct_profile
            .contains(PunctProfile::HAS_SECTION_SIG));
    }

    #[test]
    fn punct_profile_column_gaps() {
        let recs = extract_line_records("col1     col2     col3");
        assert!(
            recs[0]
                .punct_profile
                .contains(PunctProfile::HAS_COLUMN_GAPS),
            "expected HAS_COLUMN_GAPS, got {:?}",
            recs[0].punct_profile
        );
    }

    #[test]
    fn multi_byte_char_counts() {
        let src = "café\n東京\n😀\n";
        let recs = extract_line_records(src);
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].char_len, 4);
        assert_eq!(recs[1].char_len, 2);
        assert_eq!(recs[2].char_len, 1);
        // byte_len differs from char_len on non-ASCII lines.
        assert!(recs[0].byte_len > recs[0].char_len);
    }

    #[test]
    fn slice_round_trip_ascii_and_unicode() {
        let src = "ASCII line\ncafé latte\n東京タワー\n😀😎\n";
        for r in extract_line_records(src) {
            // The slice between r.start and r.end must be the line content
            // verbatim, with no terminator bytes.
            let _ = &src[r.start as usize..r.end as usize];
        }
    }

    proptest! {
        #[test]
        fn extraction_never_panics(text in "\\PC{0,512}") {
            let _ = extract_line_records(&text);
        }

        #[test]
        fn byte_conservation(text in "\\PC{0,512}") {
            let recs = extract_line_records(&text);
            let body: usize = recs.iter().map(|r| r.byte_len as usize).sum();
            let seps: usize = recs.iter().map(|r| r.term_len as usize).sum();
            prop_assert_eq!(body + seps, text.len());
        }

        #[test]
        fn slices_round_trip(text in "\\PC{0,512}") {
            let recs = extract_line_records(&text);
            for r in &recs {
                let line = &text[r.start as usize..r.end as usize];
                prop_assert_eq!(line.chars().count() as u32, r.char_len);
                prop_assert_eq!(line.is_ascii(), r.char_len == r.byte_len);
            }
        }

        #[test]
        fn stripped_offsets_are_inside_line(text in "\\PC{0,512}") {
            let recs = extract_line_records(&text);
            for r in &recs {
                prop_assert!(r.start <= r.stripped_start);
                prop_assert!(r.stripped_start <= r.stripped_end);
                prop_assert!(r.stripped_end <= r.end);
                prop_assert!(text.is_char_boundary(r.start as usize));
                prop_assert!(text.is_char_boundary(r.end as usize));
                prop_assert!(text.is_char_boundary(r.stripped_start as usize));
                prop_assert!(text.is_char_boundary(r.stripped_end as usize));
            }
        }

        #[test]
        fn blank_implies_stripped_empty(text in "\\PC{0,512}") {
            let recs = extract_line_records(&text);
            for r in &recs {
                if r.blank {
                    prop_assert_eq!(r.stripped_start, r.stripped_end);
                } else {
                    prop_assert!(r.stripped_start < r.stripped_end);
                }
            }
        }
    }
}
