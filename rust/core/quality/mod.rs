//! Document quality metrics.
//!
//! Single-pass character-class counts, token-level statistics, and
//! optional lexicon-based out-of-vocabulary detection. All char
//! classification routes through the shared `characters` module
//! (ICU4X-backed with ASCII fast paths). Token statistics reuse
//! `tokenize_words` and `classify_token` so segmentation rules stay
//! consistent with the rest of the crate.
//!
//! The pure-Rust API is deliberately split from the anomaly scoring
//! step: this crate produces raw counts; the Python `quality` module
//! turns those into ratios and applies the (tunable) weights and
//! expected ranges.

use ahash::AHashMap;

use crate::core::characters::CharacterProperties;
use crate::core::token_properties::classify_token;
use crate::core::tokenizer::{tokenize_words, TokenizerConfig};

// ─── Membership trait ──────────────────────────────────────────────────────

/// Membership test against a precomputed word set or full lexicon.
///
/// Implemented for `FstSet` (compact, immutable) and `Lexicon` (full
/// semantic graph). The trait keeps the quality analyzer agnostic to
/// the underlying data structure so callers can plug in either one.
pub trait Membership {
    /// True if `word` is a member of the set/lexicon.
    fn contains_word(&self, word: &str) -> bool;
}

impl Membership for crate::core::matching::fst_match::FstSet {
    #[inline]
    fn contains_word(&self, word: &str) -> bool {
        self.contains(word)
    }
}

impl Membership for crate::core::lexicon::Lexicon {
    #[inline]
    fn contains_word(&self, word: &str) -> bool {
        self.contains(word)
    }
}

/// A no-op membership impl that always returns false.
///
/// Used as the default type parameter when no lexicon is supplied,
/// avoiding the need for `Option<Box<dyn Membership>>` in the public API.
pub struct NoMembership;

impl Membership for NoMembership {
    #[inline]
    fn contains_word(&self, _word: &str) -> bool {
        false
    }
}

// ─── CharClassCounts ───────────────────────────────────────────────────────

/// Single-pass character-class counts for a string.
///
/// All counts are codepoint counts (not bytes). `paragraph_count` and
/// `line_count` use simple newline-based heuristics: a paragraph break
/// is any run of two or more newlines; a line break is any single
/// newline character (CR, LF, LS, PS).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CharClassCounts {
    pub total_chars: u64,
    pub whitespace: u64,
    pub alpha: u64,
    pub digit: u64,
    pub alphanumeric: u64,
    pub upper: u64,
    pub lower: u64,
    pub punct: u64,
    pub symbol: u64,
    pub non_ascii: u64,
    pub newline: u64,
    pub line_count: u64,
    pub paragraph_count: u64,
    pub char_entropy: f64,
}

/// Compute per-character classification counts in a single pass.
///
/// On ASCII input, uses a fixed 128-bucket frequency table for
/// `char_entropy`. On Unicode input, falls back to an `AHashMap`
/// keyed by codepoint.
pub fn count_char_classes(text: &str) -> CharClassCounts {
    if text.is_empty() {
        return CharClassCounts::default();
    }

    if text.is_ascii() {
        count_char_classes_ascii(text)
    } else {
        count_char_classes_unicode(text)
    }
}

fn count_char_classes_ascii(text: &str) -> CharClassCounts {
    let bytes = text.as_bytes();
    let mut counts = CharClassCounts::default();
    let mut freq = [0u64; 128];
    let mut consecutive_newlines = 0u32;
    counts.line_count = 1;
    counts.paragraph_count = 1;

    for &b in bytes {
        counts.total_chars += 1;
        freq[b as usize] += 1;

        let ch = b as char;
        if ch.is_ascii_whitespace() {
            counts.whitespace += 1;
            if matches!(ch, '\n' | '\r') {
                counts.newline += 1;
                if ch == '\n' {
                    counts.line_count += 1;
                    consecutive_newlines += 1;
                    if consecutive_newlines == 2 {
                        counts.paragraph_count += 1;
                    }
                }
                continue;
            }
            consecutive_newlines = 0;
            continue;
        }

        consecutive_newlines = 0;

        if ch.is_ascii_alphabetic() {
            counts.alpha += 1;
            if ch.is_ascii_uppercase() {
                counts.upper += 1;
            } else {
                counts.lower += 1;
            }
        } else if ch.is_ascii_digit() {
            counts.digit += 1;
        } else if ch.is_ascii_punctuation() {
            // Match the ICU-based path: $, +, <, =, >, ^, `, |, ~ are symbols.
            if matches!(
                b,
                b'$' | b'+' | b'<' | b'=' | b'>' | b'^' | b'`' | b'|' | b'~'
            ) {
                counts.symbol += 1;
            } else {
                counts.punct += 1;
            }
        } else if matches!(
            b,
            b'$' | b'+' | b'<' | b'=' | b'>' | b'^' | b'`' | b'|' | b'~'
        ) {
            counts.symbol += 1;
        }
    }

    counts.alphanumeric = counts.alpha + counts.digit;
    counts.char_entropy = shannon_entropy_from_iter(freq.iter().copied(), counts.total_chars);
    counts
}

fn count_char_classes_unicode(text: &str) -> CharClassCounts {
    let mut counts = CharClassCounts::default();
    let mut freq: AHashMap<char, u64> = AHashMap::new();
    let mut consecutive_newlines = 0u32;
    counts.line_count = 1;
    counts.paragraph_count = 1;

    for ch in text.chars() {
        counts.total_chars += 1;
        *freq.entry(ch).or_insert(0) += 1;

        if !ch.is_ascii() {
            counts.non_ascii += 1;
        }

        if CharacterProperties::is_whitespace(ch) {
            counts.whitespace += 1;
            if CharacterProperties::is_newline(ch) {
                counts.newline += 1;
                // Only LF (and Unicode line/paragraph separators) bump line_count
                // — CR alone is rare in modern text and CRLF would otherwise
                // double-count.
                if ch == '\n' || ch == '\u{2028}' || ch == '\u{2029}' {
                    counts.line_count += 1;
                    consecutive_newlines += 1;
                    if consecutive_newlines == 2 {
                        counts.paragraph_count += 1;
                    }
                    continue;
                }
                continue;
            }
            consecutive_newlines = 0;
            continue;
        }

        consecutive_newlines = 0;

        if CharacterProperties::is_letter(ch) {
            counts.alpha += 1;
            if CharacterProperties::is_uppercase(ch) {
                counts.upper += 1;
            } else if CharacterProperties::is_lowercase(ch) {
                counts.lower += 1;
            }
        } else if CharacterProperties::is_number(ch) {
            counts.digit += 1;
        } else if CharacterProperties::is_punctuation(ch) {
            counts.punct += 1;
        } else if CharacterProperties::is_symbol(ch) {
            counts.symbol += 1;
        }
    }

    counts.alphanumeric = counts.alpha + counts.digit;
    counts.char_entropy = shannon_entropy_from_iter(freq.values().copied(), counts.total_chars);
    counts
}

// ─── Word-level stats ──────────────────────────────────────────────────────

/// Token-level statistics derived from a list of cleaned tokens.
///
/// Tokens are expected to come from `tokenize_words` (whitespace scan,
/// outer punctuation stripped). All counts are token counts unless
/// noted otherwise.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WordStats {
    pub num_words: u64,
    pub unique_words: u64,
    pub total_word_chars: u64,
    pub max_freq: u64,
    pub token_entropy: f64,
    /// Tokens whose entire content is letters (per `classify_token`).
    /// This is the denominator for `ratio_in_lexicon`.
    pub alphabetic_tokens: u64,
    /// Tokens that match the supplied lexicon (lower-cased lookup).
    /// Always 0 when no lexicon is provided.
    pub in_lexicon: u64,
    /// "Format" tokens: not letter-words, not numeric, not alphanumeric,
    /// not hyphenated, not abbreviations. These are tokens that
    /// shouldn't appear in clean prose ("$$$", "..." remnants, garbled
    /// extraction artifacts).
    pub format_tokens: u64,
}

/// Compute token-level statistics in a single pass.
///
/// When `lexicon` is `Some`, also computes `alphabetic_tokens` and
/// `in_lexicon` for OCR / extraction-quality detection. The lexicon
/// lookup uses the lower-cased token; callers should ensure the
/// lexicon was built with lower-cased keys.
pub fn word_stats<L: Membership>(words: &[String], lexicon: Option<&L>) -> WordStats {
    if words.is_empty() {
        return WordStats::default();
    }

    let n = words.len() as u64;
    let mut total_word_chars = 0u64;
    let mut alphabetic_tokens = 0u64;
    let mut format_tokens = 0u64;
    let mut in_lex = 0u64;
    let mut freq: AHashMap<&str, u64> = AHashMap::with_capacity(words.len());

    for w in words {
        // Codepoint length, not byte length — matches Python `len(str)`.
        total_word_chars += w.chars().count() as u64;
        *freq.entry(w.as_str()).or_insert(0) += 1;

        let flags = classify_token(w);
        if flags.is_letter_word {
            alphabetic_tokens += 1;
            if let Some(lex) = lexicon {
                if lex.contains_word(w) {
                    in_lex += 1;
                } else {
                    // Try lower-cased once. Avoids the alloc when the token
                    // is already lower-case (common after tokenize_words with
                    // lowercase=true, but tokenize_words default is false so
                    // we have to handle both).
                    let needs_lower = w.chars().any(|c| c.is_uppercase());
                    if needs_lower {
                        let lowered = w.to_lowercase();
                        if lex.contains_word(&lowered) {
                            in_lex += 1;
                        }
                    }
                }
            }
        } else if !flags.is_numeric_word
            && !flags.is_alphanumeric_word
            && !flags.is_hyphenated
            && !flags.is_abbreviation
        {
            // Anything that doesn't fit a legitimate token shape.
            format_tokens += 1;
        }
    }

    let unique = freq.len() as u64;
    let max_freq = freq.values().copied().max().unwrap_or(0);
    let token_entropy = shannon_entropy_from_iter(freq.values().copied(), n);

    WordStats {
        num_words: n,
        unique_words: unique,
        total_word_chars,
        max_freq,
        token_entropy,
        alphabetic_tokens,
        in_lexicon: in_lex,
        format_tokens,
    }
}

// ─── Combined raw analysis ─────────────────────────────────────────────────

/// Combined raw quality metrics: char-class counts plus word stats.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QualityRaw {
    pub chars: CharClassCounts,
    pub words: WordStats,
}

/// Run the full raw-metrics pipeline.
///
/// Performs:
/// 1. Single-pass char classification (`count_char_classes`).
/// 2. Default whitespace tokenization with outer-punctuation stripping
///    (`tokenize_words`).
/// 3. Token frequency / entropy / OOV stats (`word_stats`).
pub fn analyze<L: Membership>(text: &str, lexicon: Option<&L>) -> QualityRaw {
    let chars = count_char_classes(text);
    let cfg = TokenizerConfig::new();
    let words = tokenize_words(text, &cfg);
    let words = word_stats(&words, lexicon);
    QualityRaw { chars, words }
}

/// Convenience: run analysis without a lexicon.
///
/// Equivalent to `analyze::<NoMembership>(text, None)`.
pub fn analyze_no_lexicon(text: &str) -> QualityRaw {
    analyze::<NoMembership>(text, None)
}

// ─── Entropy helper ────────────────────────────────────────────────────────

/// Shannon entropy in bits (log base 2) over a frequency iterator.
///
/// `total` is the sum of the frequencies; passing it explicitly avoids a
/// second iteration of the source.
pub fn shannon_entropy_from_iter<I: Iterator<Item = u64>>(counts: I, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut entropy = 0.0f64;
    for c in counts {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total_f;
        entropy -= p * p.log2();
    }
    entropy
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::matching::fst_match::FstSet;

    fn legal_lex() -> FstSet {
        FstSet::build([
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "and",
            "agreement",
            "contract",
            "party",
            "shall",
        ])
        .unwrap()
    }

    // ── count_char_classes ───────────────────────────────────────────────

    #[test]
    fn empty_text_zero_counts() {
        let c = count_char_classes("");
        assert_eq!(c.total_chars, 0);
        assert_eq!(c.line_count, 0);
        assert_eq!(c.paragraph_count, 0);
        assert_eq!(c.char_entropy, 0.0);
    }

    #[test]
    fn ascii_basic() {
        let text = "Hello, World!\n";
        let c = count_char_classes(text);
        assert_eq!(c.total_chars, 14);
        assert_eq!(c.alpha, 10);
        assert_eq!(c.upper, 2);
        assert_eq!(c.lower, 8);
        assert_eq!(c.punct, 2); // ',' and '!'
        assert_eq!(c.whitespace, 2); // ' ' and '\n'
        assert_eq!(c.newline, 1);
        assert_eq!(c.non_ascii, 0);
    }

    #[test]
    fn ascii_dollar_is_symbol_not_punct() {
        let c = count_char_classes("$100");
        assert_eq!(c.symbol, 1);
        assert_eq!(c.punct, 0);
        assert_eq!(c.digit, 3);
    }

    #[test]
    fn unicode_chars_classified() {
        // c,a,f,é + space + 東,京 + space + €,1,0,0
        let text = "café 東京 €100";
        let c = count_char_classes(text);
        assert_eq!(c.total_chars, 12);
        assert!(c.non_ascii > 0);
        assert_eq!(c.alpha, 6);
    }

    #[test]
    fn unicode_alpha_count_correct() {
        let c = count_char_classes("café 東京 €100");
        assert_eq!(c.alpha, 6);
        assert_eq!(c.digit, 3);
        assert_eq!(c.symbol, 1);
        assert_eq!(c.whitespace, 2);
    }

    #[test]
    fn paragraph_count_double_newline() {
        let text = "Para one.\n\nPara two.\n\nPara three.";
        let c = count_char_classes(text);
        assert_eq!(c.paragraph_count, 3);
    }

    #[test]
    fn line_count_single_newlines() {
        let text = "line1\nline2\nline3";
        let c = count_char_classes(text);
        assert_eq!(c.line_count, 3);
        assert_eq!(c.paragraph_count, 1);
    }

    #[test]
    fn entropy_uniform_higher_than_skewed() {
        let uniform = count_char_classes("abcdefgh");
        let skewed = count_char_classes("aaaaaaab");
        assert!(uniform.char_entropy > skewed.char_entropy);
    }

    // ── word_stats ───────────────────────────────────────────────────────

    #[test]
    fn empty_words_default() {
        let words: Vec<String> = vec![];
        let s = word_stats::<NoMembership>(&words, None);
        assert_eq!(s, WordStats::default());
    }

    #[test]
    fn type_token_ratio_via_unique() {
        let words = vec![
            "the".to_string(),
            "cat".to_string(),
            "the".to_string(),
            "dog".to_string(),
        ];
        let s = word_stats::<NoMembership>(&words, None);
        assert_eq!(s.num_words, 4);
        assert_eq!(s.unique_words, 3);
        assert_eq!(s.max_freq, 2);
    }

    #[test]
    fn lexicon_hits_count() {
        let words = vec![
            "the".to_string(),
            "Quick".to_string(),
            "FOX".to_string(),
            "xyzzy".to_string(),
        ];
        let lex = legal_lex();
        let s = word_stats(&words, Some(&lex));
        // alphabetic_tokens = 4 (all are letter words)
        assert_eq!(s.alphabetic_tokens, 4);
        // "the" hits, "Quick" → "quick" hits, "FOX" → "fox" hits, "xyzzy" misses.
        assert_eq!(s.in_lexicon, 3);
    }

    #[test]
    fn format_tokens_detected() {
        let words = vec![
            "hello".to_string(),
            "$$$".to_string(), // symbolic
            "...".to_string(), // punctuation — but tokenize_words filters these out
            "@@@".to_string(), // punctuation-only
            "12".to_string(),  // numeric
        ];
        let s = word_stats::<NoMembership>(&words, None);
        // "hello" and "12" are legitimate; "$$$" "..." "@@@" are format/garbage.
        assert!(s.format_tokens >= 2);
    }

    // ── analyze (combined) ───────────────────────────────────────────────

    #[test]
    fn analyze_clean_text() {
        let text = "The quick brown fox jumps over the lazy dog. The party shall agree.";
        let lex = legal_lex();
        let r = analyze(text, Some(&lex));
        assert!(r.words.num_words >= 12);
        assert!(r.words.in_lexicon >= 8);
        assert_eq!(r.chars.line_count, 1);
    }

    #[test]
    fn analyze_garbled_text_low_lexicon_hit() {
        let text = "Tlie qiiick browii rox jLnnps oxer tlie iazy dog.";
        let lex = legal_lex();
        let r = analyze(text, Some(&lex));
        // Of 9 alphabetic tokens only "dog" matches. Sanity check the OCR signal.
        assert!(r.words.in_lexicon < r.words.alphabetic_tokens / 2);
    }

    #[test]
    fn analyze_no_lexicon_zero_in_lexicon() {
        let r = analyze_no_lexicon("hello world");
        assert_eq!(r.words.in_lexicon, 0);
        assert_eq!(r.words.alphabetic_tokens, 2);
    }

    // ── Round-trip / ASCII path parity ───────────────────────────────────

    #[test]
    fn ascii_and_unicode_paths_agree_on_ascii() {
        let text = "Hello, World! 123\nNew line.";
        let ascii = count_char_classes_ascii(text);
        let unicode = count_char_classes_unicode(text);
        assert_eq!(ascii.total_chars, unicode.total_chars);
        assert_eq!(ascii.alpha, unicode.alpha);
        assert_eq!(ascii.digit, unicode.digit);
        assert_eq!(ascii.upper, unicode.upper);
        assert_eq!(ascii.lower, unicode.lower);
        assert_eq!(ascii.punct, unicode.punct);
        assert_eq!(ascii.symbol, unicode.symbol);
        assert_eq!(ascii.whitespace, unicode.whitespace);
        assert_eq!(ascii.newline, unicode.newline);
        assert_eq!(ascii.line_count, unicode.line_count);
        assert_eq!(ascii.paragraph_count, unicode.paragraph_count);
        assert!((ascii.char_entropy - unicode.char_entropy).abs() < 1e-9);
    }

    // ── shannon_entropy_from_iter ────────────────────────────────────────

    #[test]
    fn entropy_zero_total_returns_zero() {
        let counts: Vec<u64> = vec![0, 0, 0];
        assert_eq!(shannon_entropy_from_iter(counts.into_iter(), 0), 0.0);
    }

    #[test]
    fn entropy_uniform_two_outcomes_is_one() {
        let counts: Vec<u64> = vec![5, 5];
        let e = shannon_entropy_from_iter(counts.into_iter(), 10);
        assert!((e - 1.0).abs() < 1e-9);
    }
}
