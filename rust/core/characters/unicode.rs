//! Unicode support using ICU4X.
//!
//! Provides thread-safe access to Unicode properties and categories
//! with compiled data for zero runtime dependencies.
//!
//! Ported from kelvin_nlp_v1/rust/core/characters/unicode.rs.

use icu_properties::props::{GeneralCategory, GeneralCategoryGroup};
use icu_properties::{CodePointMapData, CodePointMapDataBorrowed};
use std::sync::OnceLock;

/// Thread-safe cache for Unicode general category lookups.
pub struct GeneralCategoryCache {
    gc_map: &'static CodePointMapDataBorrowed<'static, GeneralCategory>,
}

impl GeneralCategoryCache {
    /// Get the singleton instance of the cache.
    #[inline]
    pub fn instance() -> &'static Self {
        static CACHE: OnceLock<GeneralCategoryCache> = OnceLock::new();
        CACHE.get_or_init(Self::new)
    }

    fn new() -> Self {
        static GC_DATA: OnceLock<CodePointMapDataBorrowed<'static, GeneralCategory>> =
            OnceLock::new();
        let gc_map = GC_DATA.get_or_init(CodePointMapData::<GeneralCategory>::new);
        Self { gc_map }
    }

    /// Get the general category for a character.
    #[inline]
    pub fn general_category(&self, ch: char) -> GeneralCategory {
        self.gc_map.get(ch)
    }

    /// Get the general category group for a character.
    #[inline]
    pub fn general_category_group(&self, ch: char) -> GeneralCategoryGroup {
        let category = self.general_category(ch);

        if GeneralCategoryGroup::Letter.contains(category) {
            GeneralCategoryGroup::Letter
        } else if GeneralCategoryGroup::Number.contains(category) {
            GeneralCategoryGroup::Number
        } else if GeneralCategoryGroup::Punctuation.contains(category) {
            GeneralCategoryGroup::Punctuation
        } else if GeneralCategoryGroup::Symbol.contains(category) {
            GeneralCategoryGroup::Symbol
        } else if GeneralCategoryGroup::Separator.contains(category) {
            GeneralCategoryGroup::Separator
        } else if GeneralCategoryGroup::Mark.contains(category) {
            GeneralCategoryGroup::Mark
        } else if GeneralCategoryGroup::Other.contains(category) {
            GeneralCategoryGroup::Other
        } else {
            GeneralCategoryGroup::from(category)
        }
    }
}

/// High-level Unicode category operations with ASCII fast paths.
pub struct UnicodeCategories;

impl UnicodeCategories {
    /// Check if character belongs to Letter category group.
    #[inline]
    pub fn is_letter(ch: char) -> bool {
        if ch.is_ascii() {
            ch.is_ascii_alphabetic()
        } else {
            matches!(
                GeneralCategoryCache::instance().general_category_group(ch),
                GeneralCategoryGroup::Letter
            )
        }
    }

    /// Check if character belongs to Number category group.
    #[inline]
    pub fn is_number(ch: char) -> bool {
        if ch.is_ascii() {
            ch.is_ascii_digit()
        } else {
            matches!(
                GeneralCategoryCache::instance().general_category_group(ch),
                GeneralCategoryGroup::Number
            )
        }
    }

    /// Check if character belongs to Punctuation category group.
    #[inline]
    pub fn is_punctuation(ch: char) -> bool {
        if ch.is_ascii() {
            ch.is_ascii_punctuation()
        } else {
            matches!(
                GeneralCategoryCache::instance().general_category_group(ch),
                GeneralCategoryGroup::Punctuation
            )
        }
    }

    /// Check if character belongs to Symbol category group.
    #[inline]
    pub fn is_symbol(ch: char) -> bool {
        if ch.is_ascii() {
            // ASCII has overlap between punctuation and symbols in std.
            // ICU classifies $+<=>^`|~ as symbols, rest as punctuation.
            matches!(
                ch as u8,
                b'$' | b'+' | b'<' | b'=' | b'>' | b'^' | b'`' | b'|' | b'~'
            )
        } else {
            matches!(
                GeneralCategoryCache::instance().general_category_group(ch),
                GeneralCategoryGroup::Symbol
            )
        }
    }

    /// Check if character belongs to Separator category group.
    #[inline]
    pub fn is_separator(ch: char) -> bool {
        if ch.is_ascii() {
            matches!(ch, ' ' | '\t' | '\n' | '\r')
        } else {
            matches!(
                GeneralCategoryCache::instance().general_category_group(ch),
                GeneralCategoryGroup::Separator
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_initialization() {
        let cache = GeneralCategoryCache::instance();
        assert_eq!(
            cache.general_category('a'),
            GeneralCategory::LowercaseLetter
        );
        assert_eq!(
            cache.general_category('A'),
            GeneralCategory::UppercaseLetter
        );
        assert_eq!(cache.general_category('1'), GeneralCategory::DecimalNumber);
        assert_eq!(cache.general_category(' '), GeneralCategory::SpaceSeparator);
    }

    #[test]
    fn test_unicode_categories_ascii() {
        assert!(UnicodeCategories::is_letter('a'));
        assert!(UnicodeCategories::is_letter('Z'));
        assert!(!UnicodeCategories::is_letter('1'));

        assert!(UnicodeCategories::is_number('5'));
        assert!(!UnicodeCategories::is_number('a'));

        assert!(UnicodeCategories::is_punctuation('.'));
        assert!(UnicodeCategories::is_punctuation('!'));
        assert!(!UnicodeCategories::is_punctuation('a'));

        assert!(UnicodeCategories::is_separator(' '));
        assert!(UnicodeCategories::is_separator('\t'));
        assert!(!UnicodeCategories::is_separator('a'));
    }

    #[test]
    fn test_unicode_categories_unicode() {
        assert!(UnicodeCategories::is_letter('ñ'));
        assert!(UnicodeCategories::is_letter('中'));
        assert!(UnicodeCategories::is_letter('é'));

        assert!(UnicodeCategories::is_punctuation('¿'));
        assert!(UnicodeCategories::is_punctuation('…'));

        assert!(UnicodeCategories::is_symbol('€'));
        assert!(UnicodeCategories::is_symbol('™'));

        // NBSP is a separator, not a letter
        assert!(UnicodeCategories::is_separator('\u{00A0}'));
        // Ideographic space
        assert!(UnicodeCategories::is_separator('\u{3000}'));
    }

    #[test]
    fn test_symbol_vs_punctuation() {
        // ICU distinguishes these — this is why we need ICU, not is_alphanumeric
        assert!(UnicodeCategories::is_symbol('€'));
        assert!(!UnicodeCategories::is_punctuation('€'));

        assert!(UnicodeCategories::is_symbol('©'));
        assert!(!UnicodeCategories::is_punctuation('©'));

        assert!(UnicodeCategories::is_punctuation('.'));
        assert!(!UnicodeCategories::is_symbol('.'));
    }
}
