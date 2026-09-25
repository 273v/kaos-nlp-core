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

    proptest! {
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
