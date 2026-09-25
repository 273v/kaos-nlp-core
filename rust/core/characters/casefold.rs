//! Unicode full case folding.
//!
//! Wraps ICU4X [`icu_casemap`] full case folding: the `C` (common) and `F`
//! (full) mappings of `CaseFolding.txt`, using the default (non-Turkic)
//! mappings. Full case folding is defined per code point and is
//! context-free, so folding a string one char at a time yields exactly
//! the same output as folding the whole string. That property is what lets
//! offset-preserving callers attribute every folded output char to the
//! single source char that produced it.
//!
//! Notable mappings (all from the Unicode data, none special-cased here):
//!
//! - `ß` (U+00DF) and `ẞ` (U+1E9E) fold to the two chars `ss`.
//! - `İ` (U+0130) folds to the two chars `i` + U+0307 COMBINING DOT ABOVE.
//! - `ς` (final sigma) and `Σ` fold to `σ`.
//! - Ligatures such as `ﬁ` (U+FB01) fold to `fi`.
//! - ASCII `I` folds to `i` (default mapping, not the Turkic dotless `ı`).
//!
//! ASCII input takes a table-free fast path.

use icu_casemap::CaseMapper;

/// Append the full case fold of `ch` to `out` and return the number of
/// chars appended (1 for most code points, 2 or 3 for expanding folds).
#[inline]
pub fn fold_char_into(ch: char, out: &mut String) -> usize {
    if ch.is_ascii() {
        out.push(ch.to_ascii_lowercase());
        return 1;
    }
    let mut buf = [0u8; 4];
    let src: &str = ch.encode_utf8(&mut buf);
    let folded = CaseMapper::new().fold_string(src);
    out.push_str(&folded);
    folded.chars().count()
}

/// Full case fold of `text` (equivalent to folding each char in turn).
pub fn fold_str(text: &str) -> String {
    if text.is_ascii() {
        return text.to_ascii_lowercase();
    }
    CaseMapper::new().fold_string(text).into_owned()
}

/// Full case fold of `text` together with a byte-level source map.
///
/// Returned `FoldedText::source_byte[i]` is the byte offset, in `text`, of
/// the source char that produced folded byte `i`. Consecutive folded bytes
/// that come from the same source char form one *fold unit*; a unit starts
/// wherever `source_byte` changes. A span of the folded text that starts
/// and ends on unit starts (or at the end) maps back to the source span
/// `[source_byte[start], source_byte[end] or text.len())`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldedText {
    /// The folded text.
    pub text: String,
    /// Source byte offset for every byte of `text`.
    pub source_byte: Vec<u32>,
    /// `true` when at least one source char folded to more than one char,
    /// i.e. some fold unit can be split by a match boundary.
    pub has_expansion: bool,
}

impl FoldedText {
    /// Whether folded byte offset `i` is a fold-unit boundary (the start
    /// of a unit, or the end of the text).
    #[inline]
    pub fn is_unit_boundary(&self, i: usize) -> bool {
        i == 0 || i >= self.source_byte.len() || self.source_byte[i] != self.source_byte[i - 1]
    }

    /// Map a folded byte offset that is a unit boundary back to a source
    /// byte offset. `source_len` is the byte length of the source text.
    #[inline]
    pub fn source_offset(&self, i: usize, source_len: usize) -> usize {
        if i >= self.source_byte.len() {
            source_len
        } else {
            self.source_byte[i] as usize
        }
    }
}

/// Fold `text` and record, per folded byte, the producing source char.
pub fn fold_with_source_map(text: &str) -> FoldedText {
    let mut out = String::with_capacity(text.len() + 8);
    let mut source_byte: Vec<u32> = Vec::with_capacity(text.len() + 8);
    let mut has_expansion = false;
    for (b, ch) in text.char_indices() {
        let before = out.len();
        if fold_char_into(ch, &mut out) > 1 {
            has_expansion = true;
        }
        source_byte.extend(std::iter::repeat_n(b as u32, out.len() - before));
    }
    FoldedText {
        text: out,
        source_byte,
        has_expansion,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn fold_one(ch: char) -> (String, usize) {
        let mut s = String::new();
        let n = fold_char_into(ch, &mut s);
        (s, n)
    }

    #[test]
    fn ascii_folds_to_lowercase() {
        assert_eq!(fold_one('A'), ("a".to_string(), 1));
        assert_eq!(fold_one('z'), ("z".to_string(), 1));
        assert_eq!(fold_one('1'), ("1".to_string(), 1));
        assert_eq!(fold_one('I'), ("i".to_string(), 1));
    }

    #[test]
    fn latin_with_diacritics() {
        assert_eq!(fold_one('Ã').0, "ã");
        assert_eq!(fold_one('É').0, "é");
    }

    #[test]
    fn multi_char_folds() {
        assert_eq!(fold_one('ß'), ("ss".to_string(), 2));
        assert_eq!(fold_one('\u{1E9E}'), ("ss".to_string(), 2));
        assert_eq!(fold_one('İ'), ("i\u{0307}".to_string(), 2));
        assert_eq!(fold_one('\u{FB01}'), ("fi".to_string(), 2));
    }

    #[test]
    fn sigma_variants_fold_together() {
        assert_eq!(fold_one('Σ').0, "σ");
        assert_eq!(fold_one('ς').0, "σ");
    }

    #[test]
    fn caseless_chars_are_identity() {
        for ch in ['東', '😀', '\u{0301}', 'ı'] {
            assert_eq!(fold_one(ch), (ch.to_string(), 1));
        }
    }

    #[test]
    fn fold_str_matches_icu() {
        assert_eq!(
            fold_str("SÃO PAULO İstanbul STRAßE"),
            "são paulo i\u{0307}stanbul strasse"
        );
        assert_eq!(fold_str(""), "");
        assert_eq!(fold_str("ABC"), "abc");
    }

    #[test]
    fn source_map_tracks_expansions() {
        let f = fold_with_source_map("Aß!");
        assert_eq!(f.text, "ass!");
        assert_eq!(f.source_byte, vec![0, 1, 1, 3]);
        assert!(f.has_expansion);
        assert!(f.is_unit_boundary(1));
        assert!(!f.is_unit_boundary(2));
        assert!(f.is_unit_boundary(3));
        assert!(f.is_unit_boundary(4));
        assert_eq!(f.source_offset(3, 4), 3);
        assert_eq!(f.source_offset(4, 4), 4);

        let f = fold_with_source_map("ÉA");
        assert_eq!(f.text, "éa");
        assert_eq!(f.source_byte, vec![0, 0, 2]);
        assert!(!f.has_expansion);

        let f = fold_with_source_map("");
        assert!(f.text.is_empty() && f.source_byte.is_empty());
    }

    proptest! {
        #[test]
        fn source_map_is_consistent(text in "\\PC{0,128}") {
            let f = fold_with_source_map(&text);
            prop_assert_eq!(&f.text, &fold_str(&text));
            prop_assert_eq!(f.source_byte.len(), f.text.len());
            for &b in &f.source_byte {
                prop_assert!(text.is_char_boundary(b as usize));
            }
            // Every folded char boundary inside one unit maps to the same
            // source char; unit boundaries are char boundaries.
            for i in 0..=f.text.len() {
                if f.is_unit_boundary(i) {
                    prop_assert!(f.text.is_char_boundary(i));
                }
            }
        }

        #[test]
        fn per_char_fold_equals_string_fold(text in "\\PC{0,128}") {
            let mut per_char = String::new();
            for ch in text.chars() {
                fold_char_into(ch, &mut per_char);
            }
            prop_assert_eq!(per_char, fold_str(&text));
        }

        #[test]
        fn fold_is_idempotent(text in "\\PC{0,128}") {
            let once = fold_str(&text);
            prop_assert_eq!(fold_str(&once), once.clone());
        }
    }
}
