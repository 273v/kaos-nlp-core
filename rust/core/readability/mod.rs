//! Readability text primitives.
//!
//! Single-pass token counting for readability formulas: words, letters,
//! syllables, polysyllabic words, Gunning-Fog complex words, long words,
//! and (optionally) Dale-Chall unfamiliar words. The formula arithmetic
//! itself lives in the Python `readability` module so constants stay
//! tunable without rebuilding the wheel; this module only produces
//! deterministic counts.
//!
//! Definitions (documented behavior, covered by tests):
//! - A **word** is a whitespace-delimited token (outer punctuation
//!   stripped by the shared tokenizer) containing at least one letter or
//!   digit. "don't" and "mother-in-law" are one word each; "—" is none.
//! - **letters** and **letters+digits** are Unicode-codepoint counts over
//!   word tokens only (ARI uses letters+digits, Coleman-Liau letters).
//! - **syllables** come from an optional word→count map (CMUdict-derived)
//!   with the tuned heuristic in [`syllable`] as fallback; purely
//!   numeric or non-Latin tokens count as one syllable.
//! - A **polysyllable** is a word of ≥3 syllables (SMOG).
//! - A **Fog complex word** is a polysyllable surviving Gunning's
//!   exclusions, each individually configurable: hyphenated compounds;
//!   proper nouns (title-case words not at a detected sentence start);
//!   and 3-syllable words whose third syllable is an -es/-ed/-ing suffix.
//! - A **long word** has more than 6 letters+digits (LIX/RIX).
//! - An **unfamiliar word** (only when a lexicon is supplied) is a
//!   letter-bearing word whose lowercased form is not in the lexicon and
//!   which is not a detected proper noun (Dale-Chall counts names as
//!   familiar).
//!
//! Sentence starts are approximated deterministically: the first word,
//! any word whose preceding raw token ends with `.`, `!`, `?` or `…`
//! (ignoring trailing closing quotes/brackets), and any word preceded by
//! a blank line.

pub mod syllable;

use crate::core::matching::fst_match::FstMap;
use crate::core::quality::Membership;
use crate::core::tokenizer::{tokenize, TokenizerConfig};

// ─── Config ────────────────────────────────────────────────────────────────

/// Gunning-Fog complex-word exclusion switches.
///
/// Defaults are literature-faithful (all exclusions on). Disable all
/// three for textstat-style naive counting.
#[derive(Debug, Clone, Copy)]
pub struct ReadabilityConfig {
    /// Exclude 3-syllable words whose third syllable is -es/-ed/-ing
    /// ("created", "trespasses").
    pub fog_exclude_suffixes: bool,
    /// Exclude title-case words not at a detected sentence start.
    pub fog_exclude_proper_nouns: bool,
    /// Exclude hyphenated compounds ("state-of-the-art").
    pub fog_exclude_compounds: bool,
}

impl Default for ReadabilityConfig {
    fn default() -> Self {
        Self {
            fog_exclude_suffixes: true,
            fog_exclude_proper_nouns: true,
            fog_exclude_compounds: true,
        }
    }
}

// ─── TextCounts ────────────────────────────────────────────────────────────

/// Raw readability counts for a text (see module docs for definitions).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextCounts {
    pub words: u64,
    pub letters: u64,
    pub letters_and_digits: u64,
    pub syllables: u64,
    pub polysyllable_words: u64,
    pub fog_complex_words: u64,
    pub long_words: u64,
    /// `None` when no lexicon was supplied.
    pub unfamiliar_words: Option<u64>,
}

// ─── Analyzer ──────────────────────────────────────────────────────────────

/// Per-token structural facts computed in one scan of the cleaned token.
struct TokenFacts {
    letters: u64,
    digits: u64,
    title_case: bool,
    hyphenated: bool,
}

fn scan_token(token: &str) -> TokenFacts {
    let mut letters = 0u64;
    let mut digits = 0u64;
    let mut first_letter_upper = false;
    let mut upper_after_first = false;
    let mut lower_seen = false;
    let mut hyphenated = false;
    let mut prev_was_letter = false;
    let mut pending_hyphen = false;

    for ch in token.chars() {
        if ch.is_alphabetic() {
            if letters == 0 {
                first_letter_upper = ch.is_uppercase();
            } else if ch.is_uppercase() {
                upper_after_first = true;
            }
            if ch.is_lowercase() {
                lower_seen = true;
            }
            letters += 1;
            if pending_hyphen {
                hyphenated = true;
            }
            prev_was_letter = true;
            pending_hyphen = false;
        } else if ch.is_numeric() {
            digits += 1;
            prev_was_letter = false;
            pending_hyphen = false;
        } else {
            pending_hyphen = prev_was_letter
                && matches!(
                    ch,
                    '-' | '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}'
                );
            prev_was_letter = false;
        }
    }

    TokenFacts {
        letters,
        digits,
        title_case: first_letter_upper && lower_seen && !upper_after_first && letters >= 2,
        hyphenated,
    }
}

/// True when the raw whitespace-delimited chunk ends a sentence:
/// terminal punctuation, ignoring trailing closing quotes/brackets.
fn chunk_ends_sentence(raw: &str) -> bool {
    for ch in raw.chars().rev() {
        match ch {
            '"' | '\'' | ')' | ']' | '}' | '\u{2019}' | '\u{201D}' | '»' | '›' => continue,
            '.' | '!' | '?' | '…' => return true,
            _ => return false,
        }
    }
    false
}

/// True when the gap between two tokens contains a blank line
/// (two newlines separated only by other whitespace).
fn gap_has_blank_line(gap: &str) -> bool {
    let mut newlines = 0u32;
    for ch in gap.chars() {
        if ch == '\n' || ch == '\u{2028}' || ch == '\u{2029}' {
            newlines += 1;
            if newlines >= 2 {
                return true;
            }
        }
    }
    false
}

/// Syllables for a normalized token buffer, preferring the lookup map.
fn syllables_for(buf: &[u8], map: Option<&FstMap>) -> u32 {
    if let Some(m) = map {
        // Map keys are pure lowercase ASCII (CMUdict-style, apostrophes
        // included); skip the lookup when the buffer left the alphabet.
        if !buf.is_empty() && buf.iter().all(|&b| b != b' ' && b != b'#') {
            if let Ok(key) = std::str::from_utf8(buf) {
                if let Some(v) = m.get(key) {
                    return (v as u32).max(1);
                }
            }
        }
    }
    syllable::normalized_syllables(buf)
}

/// Strip one of -es/-ed/-ing from a normalized buffer, if present.
fn strip_fog_suffix(buf: &[u8]) -> Option<&[u8]> {
    if buf.len() > 3 && (buf.ends_with(b"es") || buf.ends_with(b"ed")) {
        Some(&buf[..buf.len() - 2])
    } else if buf.len() > 4 && buf.ends_with(b"ing") {
        Some(&buf[..buf.len() - 3])
    } else {
        None
    }
}

/// Count readability primitives in a single pass over `text`.
///
/// `lexicon` enables the Dale-Chall unfamiliar-word count; `syllable_map`
/// upgrades syllable accuracy via exact lookup (heuristic fallback).
/// Deterministic and panic-free for any input.
pub fn count_text<L: Membership>(
    text: &str,
    lexicon: Option<&L>,
    syllable_map: Option<&FstMap>,
    config: &ReadabilityConfig,
) -> TextCounts {
    let cfg = TokenizerConfig::new();
    let tokens = tokenize(text, &cfg);

    let mut counts = TextCounts {
        unfamiliar_words: lexicon.map(|_| 0u64),
        ..TextCounts::default()
    };

    let mut buf: Vec<u8> = Vec::with_capacity(32);
    let mut prev_span_end: Option<usize> = None;
    let mut prev_chunk_terminal = true; // first word starts a "sentence"

    for tok in &tokens {
        let facts = scan_token(&tok.text);
        let alnum = facts.letters + facts.digits;

        // Sentence-start state uses the raw chunk (with punctuation).
        let raw = &text[tok.start..tok.end];
        let sentence_initial = prev_chunk_terminal
            || prev_span_end
                .map(|e| gap_has_blank_line(&text[e..tok.start]))
                .unwrap_or(false);
        prev_chunk_terminal = chunk_ends_sentence(raw);
        prev_span_end = Some(tok.end);

        if alnum == 0 {
            continue; // not a word (symbols the tokenizer kept, emoji, …)
        }

        counts.words += 1;
        counts.letters += facts.letters;
        counts.letters_and_digits += alnum;
        if alnum > 6 {
            counts.long_words += 1;
        }

        syllable::normalize_token(&tok.text, &mut buf);
        let syl = syllables_for(&buf, syllable_map) as u64;
        counts.syllables += syl;

        let proper_noun = facts.title_case && !sentence_initial;

        if syl >= 3 {
            counts.polysyllable_words += 1;

            let mut complex = true;
            if config.fog_exclude_compounds && facts.hyphenated {
                complex = false;
            }
            if complex && config.fog_exclude_proper_nouns && proper_noun {
                complex = false;
            }
            if complex && config.fog_exclude_suffixes && syl == 3 {
                if let Some(stem) = strip_fog_suffix(&buf) {
                    if syllables_for(stem, syllable_map) < 3 {
                        complex = false;
                    }
                }
            }
            if complex {
                counts.fog_complex_words += 1;
            }
        }

        if let (Some(lex), Some(unfam)) = (lexicon, counts.unfamiliar_words.as_mut()) {
            if facts.letters > 0 && !proper_noun {
                let familiar = if let Ok(key) = std::str::from_utf8(&buf) {
                    lex.contains_word(key.trim_matches(' '))
                } else {
                    false
                };
                if !familiar {
                    *unfam += 1;
                }
            }
        }
    }

    counts
}

/// Convenience: count without a lexicon.
pub fn count_text_no_lexicon(
    text: &str,
    syllable_map: Option<&FstMap>,
    config: &ReadabilityConfig,
) -> TextCounts {
    count_text::<crate::core::quality::NoMembership>(text, None, syllable_map, config)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::matching::fst_match::FstSet;

    fn defaults() -> ReadabilityConfig {
        ReadabilityConfig::default()
    }

    #[test]
    fn empty_text() {
        let c = count_text_no_lexicon("", None, &defaults());
        assert_eq!(c, TextCounts::default());
    }

    #[test]
    fn simple_sentence_counts() {
        // "The cat sat on the mat." → 6 words, all monosyllables.
        let c = count_text_no_lexicon("The cat sat on the mat.", None, &defaults());
        assert_eq!(c.words, 6);
        assert_eq!(c.letters, 17);
        assert_eq!(c.letters_and_digits, 17);
        assert_eq!(c.syllables, 6);
        assert_eq!(c.polysyllable_words, 0);
        assert_eq!(c.fog_complex_words, 0);
        assert_eq!(c.long_words, 0);
        assert_eq!(c.unfamiliar_words, None);
    }

    #[test]
    fn digits_and_symbols() {
        // "$100" keeps the symbol (not punctuation) but has 3 digits;
        // "—" alone is stripped to nothing by the tokenizer.
        let c = count_text_no_lexicon("Pay $100 now — cash only.", None, &defaults());
        // words: Pay, $100, now, cash, only
        assert_eq!(c.words, 5);
        assert_eq!(c.letters_and_digits - c.letters, 3);
        assert_eq!(c.syllables, 1 + 1 + 1 + 1 + 2);
    }

    #[test]
    fn polysyllables_and_long_words() {
        let c = count_text_no_lexicon("The barbarian civilization crumbled.", None, &defaults());
        // barbarian (4), civilization (5) are polysyllabic; "crumbled" is 2.
        assert_eq!(c.words, 4);
        assert_eq!(c.polysyllable_words, 2);
        assert_eq!(c.long_words, 3); // barbarian, civilization, crumbled
    }

    #[test]
    fn fog_suffix_exclusion() {
        // "trespasses" = 3 syllables, stem "trespass" = 2 → excluded.
        let on = count_text_no_lexicon("He trespasses often.", None, &defaults());
        assert_eq!(on.polysyllable_words, 1);
        assert_eq!(on.fog_complex_words, 0);

        let naive = ReadabilityConfig {
            fog_exclude_suffixes: false,
            ..defaults()
        };
        let off = count_text_no_lexicon("He trespasses often.", None, &naive);
        assert_eq!(off.fog_complex_words, 1);
    }

    #[test]
    fn fog_proper_noun_exclusion() {
        // "Wisconsin" mid-sentence is a proper noun → excluded; the same
        // word at sentence start is not detectable → counted.
        let mid = count_text_no_lexicon("We toured Wisconsin today.", None, &defaults());
        assert_eq!(mid.polysyllable_words, 1);
        assert_eq!(mid.fog_complex_words, 0);

        let start = count_text_no_lexicon("Wisconsin is cold.", None, &defaults());
        assert_eq!(start.fog_complex_words, 1);
    }

    #[test]
    fn fog_proper_noun_after_terminal_counts_as_sentence_start() {
        let c = count_text_no_lexicon("It rained. Wisconsin froze.", None, &defaults());
        // "Wisconsin" follows "rained." → sentence-initial → not excluded.
        assert_eq!(c.fog_complex_words, 1);
    }

    #[test]
    fn fog_proper_noun_after_blank_line() {
        let c = count_text_no_lexicon("a heading\n\nWisconsin froze", None, &defaults());
        assert_eq!(c.fog_complex_words, 1);
    }

    #[test]
    fn fog_compound_exclusion() {
        let on = count_text_no_lexicon("A state-of-the-art design.", None, &defaults());
        assert_eq!(on.polysyllable_words, 1);
        assert_eq!(on.fog_complex_words, 0);

        let naive = ReadabilityConfig {
            fog_exclude_compounds: false,
            ..defaults()
        };
        let off = count_text_no_lexicon("A state-of-the-art design.", None, &naive);
        assert_eq!(off.fog_complex_words, 1);
    }

    #[test]
    fn unfamiliar_words_with_lexicon() {
        let lex = FstSet::build(["the", "cat", "sat", "on", "mat"]).unwrap();
        let c = count_text(
            "The cat sat on the frobnicator.",
            Some(&lex),
            None,
            &defaults(),
        );
        assert_eq!(c.words, 6);
        assert_eq!(c.unfamiliar_words, Some(1));
    }

    #[test]
    fn proper_nouns_are_familiar_for_dale_chall() {
        let lex = FstSet::build(["the", "went", "to"]).unwrap();
        let c = count_text(
            "The dog went to Chattanooga.",
            Some(&lex),
            None,
            &defaults(),
        );
        // "dog" unfamiliar; "Chattanooga" is a mid-sentence proper noun →
        // familiar per the Dale-Chall convention.
        assert_eq!(c.unfamiliar_words, Some(1));
    }

    #[test]
    fn syllable_map_overrides_heuristic() {
        use crate::core::matching::fst_match::FstMap;
        let map = FstMap::build([("cat", 9u64)]).unwrap();
        let c = count_text_no_lexicon("cat", Some(&map), &defaults());
        assert_eq!(c.syllables, 9);
        let c2 = count_text_no_lexicon("dog", Some(&map), &defaults());
        assert_eq!(c2.syllables, 1);
    }

    #[test]
    fn unicode_never_counts_are_deterministic() {
        for text in [
            "東京 大阪 café",
            "🎉🎉 emoji only",
            "à la carte",
            "ｆｕｌｌｗｉｄｔｈ",
        ] {
            let a = count_text_no_lexicon(text, None, &defaults());
            let b = count_text_no_lexicon(text, None, &defaults());
            assert_eq!(a, b);
            assert!(a.syllables >= a.words, "text {text:?}");
        }
    }

    #[test]
    fn numeric_tokens_count_one_syllable() {
        let c = count_text_no_lexicon("1234 5678", None, &defaults());
        assert_eq!(c.words, 2);
        assert_eq!(c.syllables, 2);
        assert_eq!(c.letters, 0);
        assert_eq!(c.letters_and_digits, 8);
    }

    #[cfg(test)]
    mod props {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn count_text_never_panics(s in "\\PC*") {
                let c = count_text_no_lexicon(&s, None, &ReadabilityConfig::default());
                prop_assert!(c.syllables >= c.words);
                prop_assert!(c.polysyllable_words <= c.words);
                prop_assert!(c.fog_complex_words <= c.polysyllable_words);
                prop_assert!(c.long_words <= c.words);
                prop_assert!(c.letters <= c.letters_and_digits);
            }
        }
    }
}
