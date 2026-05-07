//! Character-level properties and classification.
//!
//! Provides Unicode-correct character classification using ICU4X with:
//! - ASCII fast paths for common characters
//! - Thread-local LRU caching for Unicode property lookups
//! - Proper distinction between Punctuation, Symbol, Separator, Letter, Number
//!
//! Architecture mirrors kelvin_nlp_v1/rust/core/characters/.

pub mod properties;
pub mod unicode;

pub use properties::CharacterProperties;
pub use unicode::{GeneralCategoryCache, UnicodeCategories};

// Convenience free functions
pub use properties::{is_letter, is_number, is_punctuation, is_symbol, is_whitespace};
