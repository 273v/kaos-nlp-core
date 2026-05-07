//! Character property detection with LRU caching and ASCII optimization.
//!
//! Provides fast character classification with thread-local caching
//! and optimized ASCII fast paths for common operations.
//!
//! Ported from kelvin_nlp_v1/rust/core/characters/properties.rs.

use super::unicode::UnicodeCategories;
use lru::LruCache;
use std::cell::RefCell;
use std::num::NonZeroUsize;

/// Thread-local LRU cache size for character properties.
const CACHE_SIZE: usize = 1000;

// Function pointer wrappers for methods that aren't free functions.
fn char_is_uppercase(ch: char) -> bool {
    ch.is_uppercase()
}
fn char_is_lowercase(ch: char) -> bool {
    ch.is_lowercase()
}

/// Character property detection with caching.
pub struct CharacterProperties;

impl CharacterProperties {
    // Discriminator IDs for cache key differentiation.
    const DISC_LETTER: u8 = 0;
    const DISC_NUMBER: u8 = 1;
    const DISC_PUNCTUATION: u8 = 2;
    const DISC_WHITESPACE: u8 = 3;
    const DISC_SYMBOL: u8 = 4;
    const DISC_UPPERCASE: u8 = 5;
    const DISC_LOWERCASE: u8 = 6;

    /// Check if character is a letter (with caching).
    #[inline]
    pub fn is_letter(ch: char) -> bool {
        if ch.is_ascii() {
            ch.is_ascii_alphabetic()
        } else {
            Self::cached_unicode_check_with_id(ch, Self::DISC_LETTER, UnicodeCategories::is_letter)
        }
    }

    /// Check if character is a number (with caching).
    #[inline]
    pub fn is_number(ch: char) -> bool {
        if ch.is_ascii() {
            ch.is_ascii_digit()
        } else {
            Self::cached_unicode_check_with_id(ch, Self::DISC_NUMBER, UnicodeCategories::is_number)
        }
    }

    /// Check if character is punctuation (with caching).
    ///
    /// Uses ICU GeneralCategoryGroup::Punctuation — does NOT include symbols
    /// like €, ©, ™. This is the key difference from `!is_alphanumeric()`.
    #[inline]
    pub fn is_punctuation(ch: char) -> bool {
        if ch.is_ascii() {
            ch.is_ascii_punctuation()
        } else {
            Self::cached_unicode_check_with_id(
                ch,
                Self::DISC_PUNCTUATION,
                UnicodeCategories::is_punctuation,
            )
        }
    }

    /// Check if character is whitespace/separator (with caching).
    ///
    /// Uses ICU GeneralCategoryGroup::Separator plus additional format characters
    /// that act as word boundaries (ZWSP, ZWNBSP, etc.), matching kelvin-nlp's
    /// VALID_WHITESPACE set.
    #[inline]
    pub fn is_whitespace(ch: char) -> bool {
        if ch.is_ascii() {
            ch.is_ascii_whitespace()
        } else {
            // Check for format characters that act as word boundaries
            // (ICU classifies these as Format, not Separator, but they break words)
            matches!(ch, '\u{200B}' | '\u{FEFF}' | '\u{180E}')
                || Self::cached_unicode_check_with_id(
                    ch,
                    Self::DISC_WHITESPACE,
                    UnicodeCategories::is_separator,
                )
        }
    }

    /// Check if character is a symbol (with caching).
    #[inline]
    pub fn is_symbol(ch: char) -> bool {
        if ch.is_ascii() {
            matches!(
                ch as u8,
                b'$' | b'+' | b'<' | b'=' | b'>' | b'^' | b'`' | b'|' | b'~'
            )
        } else {
            Self::cached_unicode_check_with_id(ch, Self::DISC_SYMBOL, UnicodeCategories::is_symbol)
        }
    }

    /// Check if character is uppercase.
    #[inline]
    pub fn is_uppercase(ch: char) -> bool {
        if ch.is_ascii() {
            ch.is_ascii_uppercase()
        } else {
            // is_uppercase is a method on char, wrap as fn pointer
            Self::cached_unicode_check_with_id(ch, Self::DISC_UPPERCASE, char_is_uppercase)
        }
    }

    /// Check if character is lowercase.
    #[inline]
    pub fn is_lowercase(ch: char) -> bool {
        if ch.is_ascii() {
            ch.is_ascii_lowercase()
        } else {
            Self::cached_unicode_check_with_id(ch, Self::DISC_LOWERCASE, char_is_lowercase)
        }
    }

    /// Check if character is a letter or number.
    #[inline]
    pub fn is_letter_or_number(ch: char) -> bool {
        Self::is_letter(ch) || Self::is_number(ch)
    }

    /// Check if character is terminal punctuation (.!?…‼⁇⁈⁉‽).
    #[inline]
    pub fn is_terminal_punctuation(ch: char) -> bool {
        matches!(ch, '.' | '!' | '?' | '⁇' | '⁈' | '⁉' | '‼' | '‽' | '…')
    }

    /// Check if character is internal punctuation (,;:·).
    #[inline]
    pub fn is_internal_punctuation(ch: char) -> bool {
        matches!(ch, ',' | ';' | ':' | '·' | '、' | '；' | '：')
    }

    /// Check if character is a newline.
    #[inline]
    pub fn is_newline(ch: char) -> bool {
        matches!(ch, '\n' | '\r' | '\u{2028}' | '\u{2029}')
    }

    /// Check if character is a dash/hyphen.
    #[inline]
    pub fn is_dash(ch: char) -> bool {
        matches!(ch, '-' | '–' | '—' | '―' | '‐' | '‑' | '‒' | '⸗')
    }

    /// Thread-local cached Unicode property check.
    ///
    /// The `discriminator` distinguishes which property is being checked so
    /// different property checks on the same character don't collide in cache.
    fn cached_unicode_check_with_id(
        ch: char,
        discriminator: u8,
        check_fn: fn(char) -> bool,
    ) -> bool {
        thread_local! {
            static CACHE: RefCell<LruCache<(char, u8), bool>> = RefCell::new(
                LruCache::new(NonZeroUsize::new(CACHE_SIZE).unwrap())
            );
        }

        let key = (ch, discriminator);

        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(&result) = cache.get(&key) {
                result
            } else {
                let result = check_fn(ch);
                cache.put(key, result);
                result
            }
        })
    }
}

// Convenience free functions
#[inline]
pub fn is_letter(ch: char) -> bool {
    CharacterProperties::is_letter(ch)
}

#[inline]
pub fn is_number(ch: char) -> bool {
    CharacterProperties::is_number(ch)
}

#[inline]
pub fn is_punctuation(ch: char) -> bool {
    CharacterProperties::is_punctuation(ch)
}

#[inline]
pub fn is_whitespace(ch: char) -> bool {
    CharacterProperties::is_whitespace(ch)
}

#[inline]
pub fn is_symbol(ch: char) -> bool {
    CharacterProperties::is_symbol(ch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_properties() {
        assert!(is_letter('a'));
        assert!(is_letter('Z'));
        assert!(!is_letter('1'));
        assert!(!is_letter(' '));

        assert!(is_number('0'));
        assert!(is_number('9'));
        assert!(!is_number('a'));

        assert!(is_punctuation('.'));
        assert!(is_punctuation('!'));
        assert!(is_punctuation('?'));
        assert!(!is_punctuation('a'));
        assert!(!is_punctuation(' '));

        assert!(is_whitespace(' '));
        assert!(is_whitespace('\t'));
        assert!(is_whitespace('\n'));
        assert!(!is_whitespace('a'));
    }

    #[test]
    fn test_unicode_properties() {
        assert!(is_letter('ñ'));
        assert!(is_letter('中'));
        assert!(is_letter('α'));

        assert!(is_punctuation('¿'));
        assert!(is_punctuation('…'));

        assert!(is_symbol('€'));
        assert!(is_symbol('™'));
    }

    #[test]
    fn test_punctuation_vs_symbol() {
        // This is the critical distinction that is_alphanumeric() gets wrong
        assert!(is_punctuation('.'));
        assert!(!is_symbol('.'));

        assert!(is_symbol('€'));
        assert!(!is_punctuation('€'));

        assert!(is_symbol('©'));
        assert!(!is_punctuation('©'));
    }

    #[test]
    fn test_unicode_whitespace() {
        assert!(is_whitespace('\u{00A0}')); // NBSP
        assert!(is_whitespace('\u{2003}')); // em space
        assert!(is_whitespace('\u{3000}')); // CJK ideographic space
        assert!(!is_whitespace('a'));
    }

    #[test]
    fn test_specialized() {
        assert!(CharacterProperties::is_terminal_punctuation('.'));
        assert!(CharacterProperties::is_terminal_punctuation('!'));
        assert!(CharacterProperties::is_terminal_punctuation('…'));
        assert!(!CharacterProperties::is_terminal_punctuation(','));

        assert!(CharacterProperties::is_internal_punctuation(','));
        assert!(CharacterProperties::is_internal_punctuation(';'));
        assert!(!CharacterProperties::is_internal_punctuation('.'));

        assert!(CharacterProperties::is_dash('-'));
        assert!(CharacterProperties::is_dash('—'));
        assert!(!CharacterProperties::is_dash('.'));

        assert!(CharacterProperties::is_newline('\n'));
        assert!(CharacterProperties::is_newline('\u{2028}'));
        assert!(!CharacterProperties::is_newline(' '));
    }

    #[test]
    fn test_letter_or_number() {
        assert!(CharacterProperties::is_letter_or_number('a'));
        assert!(CharacterProperties::is_letter_or_number('5'));
        assert!(CharacterProperties::is_letter_or_number('中'));
        assert!(!CharacterProperties::is_letter_or_number('.'));
        assert!(!CharacterProperties::is_letter_or_number(' '));
        assert!(!CharacterProperties::is_letter_or_number('€'));
    }
}
