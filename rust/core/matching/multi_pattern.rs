//! Multi-pattern matching using Aho-Corasick.
//!
//! Searches a haystack for all occurrences of any pattern from a set,
//! in a single linear-time pass.
//!
//! # Case-insensitive matching
//!
//! Case-insensitive matchers compare Unicode full case folds (see
//! [`crate::core::characters::casefold`]; the same mapping as
//! `normalize(fold_case=True)` and Python's `str.casefold()`). Patterns are
//! folded once at build time; the haystack is folded per search with a
//! byte map back to the source text, so reported offsets and `text` always
//! refer to the original haystack. A match must cover whole fold units:
//! pattern `strasse` matches `STRAßE` (`ß` folds to `ss`), but pattern `s`
//! does not match half of a `ß`. No Unicode normalization form is applied;
//! precomposed `é` and `e` + U+0301 are different strings. `İ` folds to
//! `i` + U+0307 (default, non-Turkic folding), so it matches `İ` or `i̇`
//! but not a bare `i`. Pure-ASCII haystacks skip the fold.
//!
//! # Word boundaries
//!
//! With `word_boundary` on, a match is kept only if it does not extend a
//! word at either edge. At each edge, the char just inside the match and
//! the char just outside it are compared; the match is rejected when both
//! are *spaced word chars*:
//!
//! - word chars are Unicode alphanumerics (`Alphabetic` or `Numeric`) and
//!   combining marks (general category `M*`, so `e` + U+0301 stays one
//!   word). Digits are word chars (`georgia` does not match in
//!   `georgia2`). Underscore, apostrophes, hyphens, other punctuation,
//!   symbols, emoji and whitespace are not, so `dog` matches in `dog's`
//!   and `foo_bar`.
//! - chars of scripts written without spaces between words (Unicode
//!   Line_Break classes `ID` ideographic, `CJ` small kana and `SA`
//!   South-East Asian, e.g. Han, Hiragana, Katakana, Thai) are word chars
//!   that never *require* a boundary: `東京` matches in `東京都`, and a
//!   Latin pattern next to Han text (`iPhone発売`) still matches.
//! - the start and end of the haystack are boundaries.
//!
//! Candidates that fail the check are discarded **before** leftmost
//! selection, so a shorter or lower-priority pattern can still match at a
//! position where a preferred pattern was rejected.

use std::borrow::Cow;
use std::sync::OnceLock;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use icu_properties::props::{GeneralCategoryGroup, LineBreak};
use icu_properties::{CodePointMapData, CodePointMapDataBorrowed};
use serde::{Deserialize, Serialize};

use crate::core::characters::casefold::{fold_str, fold_with_source_map};
use crate::core::characters::GeneralCategoryCache;

/// A match from multi-pattern search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatternMatch {
    /// Index of the matched pattern in the original pattern list.
    pub pattern_index: usize,
    /// Byte offset of the match start.
    pub start: usize,
    /// Byte offset of the match end (exclusive).
    pub end: usize,
    /// The matched text from the haystack.
    pub text: String,
}

/// Match semantics for overlapping / priority.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum MultiPatternMatchKind {
    /// Report the first pattern that matches at each position (default).
    #[default]
    LeftmostFirst,
    /// Report the longest match at each position.
    LeftmostLongest,
}

/// Build options for [`MultiPatternMatcher::with_options`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiPatternOptions {
    /// Leftmost-first or leftmost-longest selection.
    pub match_kind: MultiPatternMatchKind,
    /// Compare Unicode full case folds of patterns and haystack.
    pub case_insensitive: bool,
    /// Reject matches that extend a word (see the module docs).
    pub word_boundary: bool,
}

/// A compiled multi-pattern matcher backed by Aho-Corasick.
pub struct MultiPatternMatcher {
    /// Leftmost automaton over the (folded, when case-insensitive) patterns.
    automaton: AhoCorasick,
    /// Standard-semantics automaton for overlapping candidate enumeration;
    /// built on first use by the filtered search path.
    overlapping: OnceLock<AhoCorasick>,
    /// Patterns as compiled (folded when case-insensitive).
    compiled: Vec<String>,
    /// Patterns as given by the caller.
    patterns: Vec<String>,
    options: MultiPatternOptions,
}

impl MultiPatternMatcher {
    /// Build a case-sensitive matcher from a list of patterns.
    pub fn new(patterns: &[&str], match_kind: MultiPatternMatchKind) -> Result<Self, String> {
        Self::with_options(
            patterns,
            MultiPatternOptions {
                match_kind,
                ..MultiPatternOptions::default()
            },
        )
    }

    /// Build a case-insensitive matcher (Unicode full case folding).
    pub fn new_case_insensitive(
        patterns: &[&str],
        match_kind: MultiPatternMatchKind,
    ) -> Result<Self, String> {
        Self::with_options(
            patterns,
            MultiPatternOptions {
                match_kind,
                case_insensitive: true,
                word_boundary: false,
            },
        )
    }

    /// Build a matcher with explicit options.
    pub fn with_options(patterns: &[&str], options: MultiPatternOptions) -> Result<Self, String> {
        if patterns.is_empty() {
            return Err("pattern list must not be empty".into());
        }
        let compiled: Vec<String> = if options.case_insensitive {
            patterns.iter().map(|p| fold_str(p)).collect()
        } else {
            patterns.iter().map(|p| p.to_string()).collect()
        };
        let ak_match_kind = match options.match_kind {
            MultiPatternMatchKind::LeftmostFirst => MatchKind::LeftmostFirst,
            MultiPatternMatchKind::LeftmostLongest => MatchKind::LeftmostLongest,
        };
        // Folded patterns contain no ASCII uppercase; ASCII case-insensitivity
        // lets pure-ASCII haystacks be searched without folding them.
        let automaton = AhoCorasickBuilder::new()
            .match_kind(ak_match_kind)
            .ascii_case_insensitive(options.case_insensitive)
            .build(&compiled)
            .map_err(|e| format!("failed to build Aho-Corasick automaton: {e}"))?;

        Ok(Self {
            automaton,
            overlapping: OnceLock::new(),
            compiled,
            patterns: patterns.iter().map(|s| s.to_string()).collect(),
            options,
        })
    }

    /// The options this matcher was built with.
    pub fn options(&self) -> MultiPatternOptions {
        self.options
    }

    fn overlapping_automaton(&self) -> &AhoCorasick {
        self.overlapping.get_or_init(|| {
            AhoCorasickBuilder::new()
                .match_kind(MatchKind::Standard)
                .ascii_case_insensitive(self.options.case_insensitive)
                .build(&self.compiled)
                // Same patterns already compiled successfully above.
                .expect("overlapping automaton over validated patterns")
        })
    }

    /// `true` when results can come straight from the leftmost automaton on
    /// the raw haystack: no word-boundary filter, and either case-sensitive
    /// or an ASCII haystack (whose full fold is its ASCII lowercase).
    #[inline]
    fn is_plain(&self, haystack: &str) -> bool {
        !self.options.word_boundary && (!self.options.case_insensitive || haystack.is_ascii())
    }

    /// Find all non-overlapping matches in the haystack. Offsets are byte
    /// offsets into `haystack`.
    pub fn find_all(&self, haystack: &str) -> Vec<PatternMatch> {
        if self.is_plain(haystack) {
            return self
                .automaton
                .find_iter(haystack)
                .map(|m| make_match(haystack, m.pattern().as_usize(), m.start(), m.end()))
                .collect();
        }
        let folded = if self.options.case_insensitive && !haystack.is_ascii() {
            Some(fold_with_source_map(haystack))
        } else {
            None
        };

        // Case-insensitive, no word boundary, and no multi-char fold: every
        // folded match starts and ends on a fold-unit boundary, so the
        // leftmost automaton on the folded text is exact.
        if let Some(f) = folded.as_ref().filter(|f| !f.has_expansion) {
            if !self.options.word_boundary {
                return self
                    .automaton
                    .find_iter(&f.text)
                    .map(|m| {
                        let s = f.source_offset(m.start(), haystack.len());
                        let e = f.source_offset(m.end(), haystack.len());
                        make_match(haystack, m.pattern().as_usize(), s, e)
                    })
                    .collect();
            }
        }

        // General path: enumerate all overlapping candidates, drop those
        // that split a fold unit or fail the word-boundary check, then
        // apply leftmost-first / leftmost-longest selection.
        let search_text: Cow<'_, str> = match &folded {
            Some(f) => Cow::Borrowed(f.text.as_str()),
            None => Cow::Borrowed(haystack),
        };
        let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
        for m in self
            .overlapping_automaton()
            .find_overlapping_iter(&*search_text)
        {
            let (s, e) = match &folded {
                Some(f) => {
                    if !f.is_unit_boundary(m.start()) || !f.is_unit_boundary(m.end()) {
                        continue;
                    }
                    (
                        f.source_offset(m.start(), haystack.len()),
                        f.source_offset(m.end(), haystack.len()),
                    )
                }
                None => (m.start(), m.end()),
            };
            if self.options.word_boundary && !is_whole_word(haystack, s, e) {
                continue;
            }
            candidates.push((s, e, m.pattern().as_usize()));
        }
        select_leftmost(&mut candidates, self.options.match_kind)
            .into_iter()
            .map(|(s, e, p)| make_match(haystack, p, s, e))
            .collect()
    }

    /// Check if any pattern matches (same semantics as [`Self::find_all`]).
    pub fn is_match(&self, haystack: &str) -> bool {
        if self.is_plain(haystack) {
            return self.automaton.is_match(haystack);
        }
        !self.find_all(haystack).is_empty()
    }

    /// Count total non-overlapping matches (same semantics as
    /// [`Self::find_all`]).
    pub fn count(&self, haystack: &str) -> usize {
        if self.is_plain(haystack) {
            return self.automaton.find_iter(haystack).count();
        }
        self.find_all(haystack).len()
    }

    /// Replace all matches with corresponding replacements.
    /// `replacements` must have the same length as the original pattern list.
    pub fn replace_all(&self, haystack: &str, replacements: &[&str]) -> Result<String, String> {
        if replacements.len() != self.patterns.len() {
            return Err(format!(
                "replacements length ({}) must match patterns length ({})",
                replacements.len(),
                self.patterns.len()
            ));
        }
        if self.is_plain(haystack) {
            return Ok(self.automaton.replace_all(haystack, replacements));
        }
        let mut out = String::with_capacity(haystack.len());
        let mut last = 0;
        for m in self.find_all(haystack) {
            out.push_str(&haystack[last..m.start]);
            out.push_str(replacements[m.pattern_index]);
            last = m.end;
        }
        out.push_str(&haystack[last..]);
        Ok(out)
    }

    /// Return the number of patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Return the pattern at the given index (as originally given).
    pub fn pattern(&self, index: usize) -> Option<&str> {
        self.patterns.get(index).map(|s| s.as_str())
    }
}

#[inline]
fn make_match(haystack: &str, pattern_index: usize, start: usize, end: usize) -> PatternMatch {
    PatternMatch {
        pattern_index,
        start,
        end,
        text: haystack[start..end].to_string(),
    }
}

/// Leftmost non-overlapping selection over `(start, end, pattern)` candidates,
/// mirroring Aho-Corasick leftmost-first / leftmost-longest semantics.
fn select_leftmost(
    candidates: &mut [(usize, usize, usize)],
    kind: MultiPatternMatchKind,
) -> Vec<(usize, usize, usize)> {
    match kind {
        // Earliest start, then earliest pattern in the list.
        MultiPatternMatchKind::LeftmostFirst => {
            candidates.sort_unstable_by_key(|&(s, _, p)| (s, p))
        }
        // Earliest start, then longest, then earliest pattern.
        MultiPatternMatchKind::LeftmostLongest => {
            candidates.sort_unstable_by_key(|&(s, e, p)| (s, std::cmp::Reverse(e), p))
        }
    }
    let mut out = Vec::new();
    // Next allowed start; `strict` forbids a second match at the position of
    // a previous empty match.
    let mut pos = 0usize;
    let mut strict = false;
    for &(s, e, p) in candidates.iter() {
        if s < pos || (strict && s == pos) {
            continue;
        }
        out.push((s, e, p));
        strict = s == e;
        pos = e;
    }
    out
}

fn line_break_map() -> &'static CodePointMapDataBorrowed<'static, LineBreak> {
    static LB: OnceLock<CodePointMapDataBorrowed<'static, LineBreak>> = OnceLock::new();
    LB.get_or_init(CodePointMapData::<LineBreak>::new)
}

/// Word char that requires a boundary against another such char: Unicode
/// alphanumeric or combining mark, excluding scripts written without
/// spaces between words (Line_Break ID / CJ / SA).
#[inline]
fn is_spaced_word_char(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric();
    }
    let is_word = c.is_alphanumeric()
        || GeneralCategoryGroup::Mark
            .contains(GeneralCategoryCache::instance().general_category(c));
    if !is_word {
        return false;
    }
    !matches!(
        line_break_map().get(c),
        LineBreak::Ideographic | LineBreak::ConditionalJapaneseStarter | LineBreak::ComplexContext
    )
}

/// `true` when the byte span `[start, end)` of `text` does not extend a
/// word at either edge. Empty spans are never whole words.
fn is_whole_word(text: &str, start: usize, end: usize) -> bool {
    if start >= end {
        return false;
    }
    let inner = &text[start..end];
    let (Some(first), Some(last)) = (inner.chars().next(), inner.chars().next_back()) else {
        return false;
    };
    if let Some(prev) = text[..start].chars().next_back() {
        if is_spaced_word_char(prev) && is_spaced_word_char(first) {
            return false;
        }
    }
    if let Some(next) = text[end..].chars().next() {
        if is_spaced_word_char(last) && is_spaced_word_char(next) {
            return false;
        }
    }
    true
}

/// Convenience: search a haystack for all patterns in one call.
pub fn multi_pattern_search(
    haystack: &str,
    patterns: &[&str],
) -> Result<Vec<PatternMatch>, String> {
    let matcher = MultiPatternMatcher::new(patterns, MultiPatternMatchKind::LeftmostFirst)?;
    Ok(matcher.find_all(haystack))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let matcher = MultiPatternMatcher::new(
            &["he", "she", "his", "hers"],
            MultiPatternMatchKind::LeftmostFirst,
        )
        .unwrap();
        let matches = matcher.find_all("ushers");
        assert!(!matches.is_empty());
        // "she" or "he" should be found
        let texts: Vec<&str> = matches.iter().map(|m| m.text.as_str()).collect();
        assert!(texts.contains(&"she") || texts.contains(&"he"));
    }

    #[test]
    fn test_longest_match() {
        let matcher = MultiPatternMatcher::new(
            &["he", "hello", "hell"],
            MultiPatternMatchKind::LeftmostLongest,
        )
        .unwrap();
        let matches = matcher.find_all("hello world");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "hello");
    }

    #[test]
    fn test_case_insensitive() {
        let matcher = MultiPatternMatcher::new_case_insensitive(
            &["hello"],
            MultiPatternMatchKind::LeftmostFirst,
        )
        .unwrap();
        assert!(matcher.is_match("HELLO WORLD"));
    }

    #[test]
    fn test_replace() {
        let matcher =
            MultiPatternMatcher::new(&["cat", "dog"], MultiPatternMatchKind::LeftmostFirst)
                .unwrap();
        let result = matcher
            .replace_all("I have a cat and a dog", &["CAT", "DOG"])
            .unwrap();
        assert_eq!(result, "I have a CAT and a DOG");
    }

    #[test]
    fn test_count() {
        let matcher =
            MultiPatternMatcher::new(&["the"], MultiPatternMatchKind::LeftmostFirst).unwrap();
        assert_eq!(matcher.count("the cat and the dog and the bird"), 3);
    }

    #[test]
    fn test_convenience() {
        let matches = multi_pattern_search("hello world", &["hello", "world"]).unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_empty_patterns_error() {
        let empty: &[&str] = &[];
        assert!(MultiPatternMatcher::new(empty, MultiPatternMatchKind::LeftmostFirst).is_err());
    }

    // --- Unicode byte offset correctness ---

    #[test]
    fn test_unicode_patterns_byte_offsets() {
        let matcher =
            MultiPatternMatcher::new(&["café", "東京"], MultiPatternMatchKind::LeftmostFirst)
                .unwrap();
        let text = "café 東京";
        let matches = matcher.find_all(text);
        assert_eq!(matches.len(), 2);
        for m in &matches {
            assert_eq!(&text[m.start..m.end], m.text);
        }
    }

    #[test]
    fn test_pattern_after_multibyte() {
        let matcher =
            MultiPatternMatcher::new(&["hello"], MultiPatternMatchKind::LeftmostFirst).unwrap();
        let text = "東京 hello";
        let matches = matcher.find_all(text);
        assert_eq!(matches.len(), 1);
        assert_eq!(&text[matches[0].start..matches[0].end], "hello");
    }

    #[test]
    fn test_emoji_pattern() {
        let matcher =
            MultiPatternMatcher::new(&["😀", "🌍"], MultiPatternMatchKind::LeftmostFirst).unwrap();
        let text = "Hello 😀 world 🌍";
        let matches = matcher.find_all(text);
        assert_eq!(matches.len(), 2);
        for m in &matches {
            assert_eq!(&text[m.start..m.end], m.text);
        }
    }

    // --- Unicode case-insensitive matching ---

    fn ci(patterns: &[&str]) -> MultiPatternMatcher {
        MultiPatternMatcher::new_case_insensitive(patterns, MultiPatternMatchKind::LeftmostFirst)
            .unwrap()
    }

    fn opts(ci: bool, wb: bool, longest: bool) -> MultiPatternOptions {
        MultiPatternOptions {
            match_kind: if longest {
                MultiPatternMatchKind::LeftmostLongest
            } else {
                MultiPatternMatchKind::LeftmostFirst
            },
            case_insensitive: ci,
            word_boundary: wb,
        }
    }

    fn spans(ms: &[PatternMatch]) -> Vec<(&str, usize)> {
        ms.iter()
            .map(|m| (m.text.as_str(), m.pattern_index))
            .collect()
    }

    #[test]
    fn test_ci_non_ascii_letters() {
        let m = ci(&["são paulo"]);
        let text = "Em SÃO PAULO e são paulo";
        let found = m.find_all(text);
        assert_eq!(spans(&found), vec![("SÃO PAULO", 0), ("são paulo", 0)]);
        for f in &found {
            assert_eq!(&text[f.start..f.end], f.text);
        }
        // Pattern written in uppercase matches lowercase text too.
        assert!(ci(&["SÃO PAULO"]).is_match("são paulo"));
    }

    #[test]
    fn test_ci_multi_char_folds() {
        // ß folds to "ss": whole-unit matches in both directions.
        let text = "Hauptstraße und HAUPTSTRASSE";
        assert_eq!(ci(&["hauptstrasse"]).count(text), 2);
        assert_eq!(ci(&["STRAßE"]).count(text), 2);
        // A match may not split the "ss" produced by one ß.
        let m = ci(&["stras"]);
        assert_eq!(spans(&m.find_all(text)), vec![("STRAS", 0)]);
        // İ folds to i + U+0307; it matches İ or i̇, not a bare i.
        assert!(ci(&["i\u{0307}stanbul"]).is_match("İSTANBUL"));
        assert!(ci(&["İstanbul"]).is_match("i\u{0307}stanbul"));
        assert!(!ci(&["istanbul"]).is_match("İstanbul"));
        // Final sigma and capital sigma fold together.
        assert!(ci(&["οδυσσευς"]).is_match("ΟΔΥΣΣΕΥΣ"));
    }

    #[test]
    fn test_ci_offsets_refer_to_original_text() {
        let text = "😀 STRAßE 東京 Straße";
        let found = ci(&["strasse", "東京"]).find_all(text);
        assert_eq!(
            spans(&found),
            vec![("STRAßE", 0), ("東京", 1), ("Straße", 0)]
        );
        for f in &found {
            assert!(text.is_char_boundary(f.start) && text.is_char_boundary(f.end));
            assert_eq!(&text[f.start..f.end], f.text);
        }
    }

    #[test]
    fn test_ci_replace_all_uses_original_spans() {
        let m = ci(&["straße"]);
        let out = m.replace_all("die STRASSE, die Straße", &["X"]).unwrap();
        assert_eq!(out, "die X, die X");
    }

    #[test]
    fn test_ci_ascii_behaviour_unchanged() {
        // Leftmost-first: "hello" is listed first, so it wins at position 0.
        let m = ci(&["hello", "HELLO world"]);
        let found = m.find_all("Hello WORLD, hello");
        assert_eq!(spans(&found), vec![("Hello", 0), ("hello", 0)]);
    }

    #[test]
    fn test_ci_empty_haystack() {
        assert!(ci(&["a"]).find_all("").is_empty());
        assert!(!ci(&["a"]).is_match(""));
        assert_eq!(ci(&["a"]).count(""), 0);
    }

    // --- Word boundaries ---

    fn wb(patterns: &[&str], ci: bool) -> MultiPatternMatcher {
        MultiPatternMatcher::with_options(patterns, opts(ci, true, false)).unwrap()
    }

    #[test]
    fn test_word_boundary_rejects_partial_words() {
        let m = wb(&["georgia", "Méx"], true);
        assert!(m.find_all("Georgian food in Méxicali").is_empty());
        let found = m.find_all("Georgia, not Georgian; Méx.");
        assert_eq!(spans(&found), vec![("Georgia", 0), ("Méx", 1)]);
        // Without the flag the partial matches are still reported.
        let loose =
            MultiPatternMatcher::with_options(&["georgia"], opts(true, false, false)).unwrap();
        assert_eq!(loose.count("Georgian"), 1);
    }

    #[test]
    fn test_word_boundary_edges_and_separators() {
        let m = wb(&["dog"], false);
        assert_eq!(m.count("dog"), 1); // haystack edges are boundaries
        assert_eq!(m.count("dog's dog-house (dog) dog_bar"), 4);
        assert_eq!(m.count("dogs hotdog dog2 2dog"), 0); // letters/digits extend
        assert_eq!(m.count("dog😀dog"), 2); // emoji is not a word char
    }

    #[test]
    fn test_word_boundary_digits() {
        let m = wb(&["2024"], false);
        assert_eq!(m.count("FY2024 2024 20245 2024-25"), 2);
    }

    #[test]
    fn test_word_boundary_combining_marks_extend_words() {
        // "cafe" + U+0301 is "café" in decomposed form: no whole-word "cafe".
        let m = wb(&["cafe"], false);
        assert_eq!(m.count("cafe\u{0301} cafe"), 1);
    }

    #[test]
    fn test_word_boundary_unspaced_scripts_still_match() {
        let m = wb(&["東京", "タワー", "กรุงเทพ"], false);
        let found = m.find_all("東京都の東京タワーとกรุงเทพมหานคร");
        assert_eq!(
            spans(&found),
            vec![("東京", 0), ("東京", 0), ("タワー", 1), ("กรุงเทพ", 2)]
        );
        // Latin next to Han: the Han side needs no boundary.
        assert_eq!(wb(&["iphone"], true).count("新型iPhone発売"), 1);
        // Hangul is written with spaces, so it keeps boundaries.
        assert_eq!(wb(&["서울"], false).count("서울시 서울"), 1);
    }

    #[test]
    fn test_word_boundary_rejection_does_not_block_other_patterns() {
        // Leftmost-first would prefer "georgia" at 0; it is rejected, so
        // "georgian" (later in the list) is reported instead.
        let m = wb(&["georgia", "georgian"], true);
        assert_eq!(spans(&m.find_all("Georgian")), vec![("Georgian", 1)]);
        // Longest-match semantics also filter before selection.
        let m = MultiPatternMatcher::with_options(&["new york", "york"], opts(true, true, true))
            .unwrap();
        assert_eq!(spans(&m.find_all("New Yorker in York")), vec![("York", 1)]);
    }

    #[test]
    fn test_word_boundary_patterns_with_punctuation_edges() {
        // Edges that are not word chars never need a boundary.
        // "U.S." ends in '.', so the following 'A' does not matter; "#tag"
        // ends in a letter, so "#tags" is rejected.
        let m = wb(&["U.S.", "#tag"], false);
        assert_eq!(
            spans(&m.find_all("the U.S.A. and #tag, #tags")),
            vec![("U.S.", 0), ("#tag", 1)]
        );
    }

    #[test]
    fn test_word_boundary_consistent_api() {
        let m = wb(&["georgia"], true);
        let text = "Georgian GEORGIA georgia";
        assert_eq!(m.find_all(text).len(), 2);
        assert_eq!(m.count(text), 2);
        assert!(m.is_match(text));
        assert!(!m.is_match("Georgian"));
        assert_eq!(m.replace_all(text, &["X"]).unwrap(), "Georgian X X");
    }

    #[test]
    fn test_empty_pattern_filtered_path_terminates() {
        let m = MultiPatternMatcher::with_options(&["", "ß"], opts(true, false, false)).unwrap();
        let found = m.find_all("aSS");
        assert!(found.iter().all(|f| f.start <= f.end));
        let m = wb(&[""], false);
        assert!(m.find_all("a b").is_empty());
    }

    proptest::proptest! {
        /// The filtered selection path reproduces Aho-Corasick leftmost
        /// semantics when no candidate is filtered out.
        #[test]
        fn select_leftmost_matches_aho_corasick(
            text in "[abc]{0,40}",
            pats in proptest::collection::vec("[abc]{1,4}", 1..6),
            longest in proptest::bool::ANY,
        ) {
            let refs: Vec<&str> = pats.iter().map(|s| s.as_str()).collect();
            let m = MultiPatternMatcher::with_options(&refs, opts(false, false, longest)).unwrap();
            let expected: Vec<(usize, usize, usize)> = m
                .find_all(&text)
                .into_iter()
                .map(|x| (x.start, x.end, x.pattern_index))
                .collect();
            let mut cands: Vec<(usize, usize, usize)> = m
                .overlapping_automaton()
                .find_overlapping_iter(&text)
                .map(|x| (x.start(), x.end(), x.pattern().as_usize()))
                .collect();
            let got = select_leftmost(&mut cands, m.options().match_kind);
            proptest::prop_assert_eq!(got, expected);
        }

        /// Case-insensitive offsets always slice the original haystack on
        /// char boundaries, and the slice folds to the matched pattern.
        #[test]
        fn ci_matches_fold_to_pattern(
            text in "[aAßẞsSİiΣσς東 ]{0,30}",
            pats in proptest::collection::vec("[aAßsSİiΣς東]{1,3}", 1..4),
            wbf in proptest::bool::ANY,
        ) {
            let refs: Vec<&str> = pats.iter().map(|s| s.as_str()).collect();
            let m = MultiPatternMatcher::with_options(&refs, opts(true, wbf, false)).unwrap();
            let mut last_end = 0;
            for f in m.find_all(&text) {
                proptest::prop_assert!(text.is_char_boundary(f.start));
                proptest::prop_assert!(text.is_char_boundary(f.end));
                proptest::prop_assert!(f.start >= last_end);
                last_end = f.end;
                proptest::prop_assert_eq!(fold_str(&f.text), fold_str(&pats[f.pattern_index]));
            }
        }
    }
}
