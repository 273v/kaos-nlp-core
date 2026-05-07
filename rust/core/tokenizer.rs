//! Word tokenizer with ICU4X-based Unicode support.
//!
//! Uses the `characters` module for proper Unicode classification:
//! - Whitespace detection via ICU GeneralCategoryGroup::Separator
//! - Punctuation stripping via ICU GeneralCategoryGroup::Punctuation
//! - ASCII fast paths for performance
//!
//! Features:
//! - Whitespace scan tokenization (configurable via ICU separator detection)
//! - Regex-based tokenization (custom patterns)
//! - Span tracking (byte offsets into source text)
//! - Prefix truncation for approximate stemming
//! - Stopword filtering
//! - Keep/strip punctuation option
//!
//! Architecture mirrors kelvin-nlp tokenize() / tokenize_regex().

use ahash::AHashSet;
use serde::{Deserialize, Serialize};

use crate::core::characters::CharacterProperties;

// ─── Token ─────────────────────────────────────────────────────────────────

/// A token with its text and byte span in the source string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    /// The token text (cleaned: punctuation stripped, optionally lowercased/prefix-truncated).
    pub text: String,
    /// Byte offset of the original span start in the source text.
    pub start: usize,
    /// Byte offset of the original span end (exclusive) in the source text.
    pub end: usize,
}

// ─── Punctuation trimming (using ICU-based CharacterProperties) ────────────

/// Strip leading punctuation from a token. Returns a byte-offset slice.
///
/// Uses `CharacterProperties::is_punctuation()` which properly distinguishes
/// punctuation (. , ! ¿ …) from symbols (€ © ™) via ICU4X.
#[inline]
fn trim_start_punctuation(token: &str) -> &str {
    if token.is_ascii() {
        // ASCII fast path: byte-level scan
        let bytes = token.as_bytes();
        let start = bytes
            .iter()
            .position(|&b| !(b as char).is_ascii_punctuation())
            .unwrap_or(bytes.len());
        &token[start..]
    } else {
        // Unicode: use ICU-based check
        let start = token
            .char_indices()
            .find(|(_, ch)| !CharacterProperties::is_punctuation(*ch))
            .map(|(idx, _)| idx)
            .unwrap_or(token.len());
        &token[start..]
    }
}

/// Strip trailing punctuation from a token. Returns a byte-offset slice.
#[inline]
fn trim_end_punctuation(token: &str) -> &str {
    if token.is_ascii() {
        let bytes = token.as_bytes();
        let end = bytes
            .iter()
            .rposition(|&b| !(b as char).is_ascii_punctuation())
            .map(|i| i + 1)
            .unwrap_or(0);
        &token[..end]
    } else {
        let end = token
            .char_indices()
            .rfind(|(_, ch)| !CharacterProperties::is_punctuation(*ch))
            .map(|(idx, ch)| idx + ch.len_utf8())
            .unwrap_or(0);
        &token[..end]
    }
}

/// Strip leading and trailing punctuation. Preserves internal punctuation
/// (e.g., "don't" → "don't", "(hello)" → "hello").
///
/// Returns a zero-copy `&str` slice into the original.
#[inline]
pub fn trim_punctuation(token: &str) -> &str {
    trim_end_punctuation(trim_start_punctuation(token))
}

// ─── TokenizerConfig ───────────────────────────────────────────────────────

/// Configuration for tokenization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Whether to lowercase tokens.
    pub lowercase: bool,
    /// Whether to keep punctuation on tokens (if false, strip leading/trailing).
    pub keep_punctuation: bool,
    /// Prefix length for approximate stemming (0 = no truncation).
    /// e.g., prefix=4 turns "automobile" → "auto".
    pub prefix: usize,
    /// Stopwords to filter out. Empty = no filtering.
    pub stopwords: AHashSet<String>,
    /// Controlled lexicon: if non-empty, only emit tokens that appear in this set.
    /// Applied after lowercase/prefix/stopwords.
    pub index: AHashSet<String>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            lowercase: false,
            keep_punctuation: false,
            prefix: 0,
            stopwords: AHashSet::new(),
            index: AHashSet::new(),
        }
    }
}

impl TokenizerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lowercase(mut self) -> Self {
        self.lowercase = true;
        self
    }

    pub fn keep_punctuation(mut self) -> Self {
        self.keep_punctuation = true;
        self
    }

    pub fn with_prefix(mut self, prefix: usize) -> Self {
        self.prefix = prefix;
        self
    }

    pub fn with_stopwords<I, S>(mut self, stopwords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.stopwords = stopwords
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        self
    }

    /// Set a controlled lexicon: only emit tokens that appear in this set.
    pub fn with_index<I, S>(mut self, index: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.index = index.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }

    /// Clean a raw token span according to this config.
    /// Returns None if the token should be filtered out.
    #[inline]
    fn clean_token(&self, raw: &str) -> Option<String> {
        let cleaned = if self.keep_punctuation {
            raw
        } else {
            trim_punctuation(raw)
        };

        if cleaned.is_empty() {
            return None;
        }

        let mut result = if self.lowercase {
            cleaned.to_lowercase()
        } else {
            cleaned.to_string()
        };

        // Prefix truncation at char boundary
        if self.prefix > 0 && result.chars().count() > self.prefix {
            let end = result
                .char_indices()
                .nth(self.prefix)
                .map(|(idx, _)| idx)
                .unwrap_or(result.len());
            result.truncate(end);
        }

        // Stopword filtering (after lowercase + prefix)
        if !self.stopwords.is_empty() && self.stopwords.contains(&result) {
            return None;
        }

        // Controlled lexicon filter (after all transformations)
        if !self.index.is_empty() && !self.index.contains(&result) {
            return None;
        }

        Some(result)
    }
}

// ─── Tokenization functions ────────────────────────────────────────────────

/// Tokenize by scanning for whitespace boundaries using ICU separator detection.
///
/// Uses `CharacterProperties::is_whitespace()` which covers all Unicode
/// separators (NBSP, CJK space, ZWSP, en/em space, etc.) via ICU4X,
/// with an ASCII fast path.
///
/// Returns a Vec of Tokens with byte-offset spans.
pub fn tokenize(text: &str, config: &TokenizerConfig) -> Vec<Token> {
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let len = text.len();
    let mut pos = 0;

    // ASCII fast path: if entire text is ASCII, scan bytes directly
    if text.is_ascii() {
        while pos < len {
            // Skip whitespace
            while pos < len && (bytes[pos] as char).is_ascii_whitespace() {
                pos += 1;
            }
            if pos >= len {
                break;
            }

            // Collect non-whitespace span
            let span_start = pos;
            while pos < len && !(bytes[pos] as char).is_ascii_whitespace() {
                pos += 1;
            }
            let raw = &text[span_start..pos];

            if let Some(cleaned) = config.clean_token(raw) {
                tokens.push(Token {
                    text: cleaned,
                    start: span_start,
                    end: pos,
                });
            }
        }
        return tokens;
    }

    // Unicode path: char-by-char scan using ICU separator detection
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let num_chars = chars.len();
    let mut i = 0;

    while i < num_chars {
        // Skip whitespace (ICU separator)
        while i < num_chars && CharacterProperties::is_whitespace(chars[i].1) {
            i += 1;
        }
        if i >= num_chars {
            break;
        }

        // Collect non-whitespace span
        let span_start = chars[i].0;
        while i < num_chars && !CharacterProperties::is_whitespace(chars[i].1) {
            i += 1;
        }
        let span_end = if i < num_chars { chars[i].0 } else { len };
        let raw = &text[span_start..span_end];

        if let Some(cleaned) = config.clean_token(raw) {
            tokens.push(Token {
                text: cleaned,
                start: span_start,
                end: span_end,
            });
        }
    }

    tokens
}

/// Tokenize using a regex pattern (non-whitespace runs).
///
/// If no pattern is provided, uses a default pattern matching consecutive
/// non-separator characters.
pub fn tokenize_regex(
    text: &str,
    config: &TokenizerConfig,
    pattern: Option<&regex::Regex>,
) -> Vec<Token> {
    let default_re = default_regex();
    let re = pattern.unwrap_or(&default_re);
    let mut tokens = Vec::new();

    for m in re.find_iter(text) {
        let raw = m.as_str();
        if let Some(cleaned) = config.clean_token(raw) {
            tokens.push(Token {
                text: cleaned,
                start: m.start(),
                end: m.end(),
            });
        }
    }

    tokens
}

/// Default regex: match consecutive non-whitespace (including Unicode separators).
fn default_regex() -> regex::Regex {
    // Match runs of characters that are NOT Unicode separators.
    // \p{Z} is the Unicode Separator category (space, line, paragraph separators).
    // Also include ASCII whitespace chars \s.
    regex::Regex::new(r"[^\s\p{Z}]+").unwrap()
}

/// Convenience: tokenize and return only the token strings (no spans).
pub fn tokenize_words(text: &str, config: &TokenizerConfig) -> Vec<String> {
    tokenize(text, config).into_iter().map(|t| t.text).collect()
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TokenizerConfig {
        TokenizerConfig::default()
    }

    // --- Basic tokenization ---

    #[test]
    fn test_basic() {
        let tokens = tokenize("Hello, world!", &default_config());
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_spans() {
        let text = "Hello, world!";
        let tokens = tokenize(text, &default_config());
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 6); // "Hello," span
        assert_eq!(tokens[1].start, 7);
        assert_eq!(tokens[1].end, 13);
        // Span covers original text including punctuation
        assert_eq!(&text[tokens[0].start..tokens[0].end], "Hello,");
    }

    #[test]
    fn test_lowercase() {
        let config = TokenizerConfig::new().lowercase();
        let tokens = tokenize("Hello World", &config);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_keep_punctuation() {
        let config = TokenizerConfig::new().keep_punctuation();
        let tokens = tokenize("Hello, world!", &config);
        assert_eq!(tokens[0].text, "Hello,");
        assert_eq!(tokens[1].text, "world!");
    }

    // --- Prefix truncation ---

    #[test]
    fn test_prefix() {
        let config = TokenizerConfig::new().with_prefix(4);
        let tokens = tokenize("automobile transportation", &config);
        assert_eq!(tokens[0].text, "auto");
        assert_eq!(tokens[1].text, "tran");
    }

    #[test]
    fn test_prefix_short_word() {
        let config = TokenizerConfig::new().with_prefix(4);
        let tokens = tokenize("the cat", &config);
        assert_eq!(tokens[0].text, "the");
        assert_eq!(tokens[1].text, "cat");
    }

    #[test]
    fn test_prefix_unicode() {
        let config = TokenizerConfig::new().with_prefix(4);
        let tokens = tokenize("café résumé", &config);
        assert_eq!(tokens[0].text, "café");
        assert_eq!(tokens[1].text, "résu");
    }

    // --- Stopword filtering ---

    #[test]
    fn test_stopwords() {
        let config = TokenizerConfig::new()
            .lowercase()
            .with_stopwords(["the", "a", "an", "and", "of"]);
        let tokens = tokenize("The cat and the dog", &config);
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(words, vec!["cat", "dog"]);
    }

    // --- Unicode whitespace (via ICU Separator) ---

    #[test]
    fn test_nbsp() {
        let tokens = tokenize("hello\u{00A0}world", &default_config());
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_cjk_space() {
        let tokens = tokenize("東京\u{3000}大阪", &default_config());
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "東京");
        assert_eq!(tokens[1].text, "大阪");
    }

    #[test]
    fn test_en_space() {
        let tokens = tokenize("hello\u{2002}world", &default_config());
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_em_space() {
        let tokens = tokenize("hello\u{2003}world", &default_config());
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_thin_space() {
        let tokens = tokenize("hello\u{2009}world", &default_config());
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_mixed_whitespace() {
        let tokens = tokenize("a\tb\nc\u{00A0}d\u{2003}e", &default_config());
        assert_eq!(tokens.len(), 5);
    }

    // --- Punctuation stripping (via ICU Punctuation, not is_alphanumeric) ---

    #[test]
    fn test_strips_outer_punctuation() {
        let tokens = tokenize("'hello' (world) --test--", &default_config());
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "test");
    }

    #[test]
    fn test_preserves_internal_punctuation() {
        let tokens = tokenize("don't mother-in-law", &default_config());
        assert_eq!(tokens[0].text, "don't");
        assert_eq!(tokens[1].text, "mother-in-law");
    }

    #[test]
    fn test_pure_punctuation_filtered() {
        let tokens = tokenize("--- ... !!!", &default_config());
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_symbol_not_stripped() {
        // € and © are SYMBOLS, not PUNCTUATION — should NOT be stripped
        let _config = default_config();
        let trimmed = trim_punctuation("€100");
        assert_eq!(trimmed, "€100");

        let trimmed = trim_punctuation("©2024");
        assert_eq!(trimmed, "©2024");
    }

    #[test]
    fn test_unicode_quotes_stripped() {
        // « » are PUNCTUATION — should be stripped
        let trimmed = trim_punctuation("«café»");
        assert_eq!(trimmed, "café");

        // Smart quotes
        let trimmed = trim_punctuation("\u{201C}hello\u{201D}");
        assert_eq!(trimmed, "hello");
    }

    // --- Regex tokenization ---

    #[test]
    fn test_regex_basic() {
        let tokens = tokenize_regex("Hello, world!", &default_config(), None);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_regex_custom_pattern() {
        let re = regex::Regex::new(r"[a-zA-Z]+").unwrap();
        let tokens = tokenize_regex("abc 123 def 456", &default_config(), Some(&re));
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "abc");
        assert_eq!(tokens[1].text, "def");
    }

    // --- Edge cases ---

    #[test]
    fn test_empty_string() {
        assert!(tokenize("", &default_config()).is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        assert!(tokenize("   \t\n  ", &default_config()).is_empty());
    }

    #[test]
    fn test_single_word() {
        let tokens = tokenize("hello", &default_config());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 5);
    }

    // --- Controlled lexicon ---

    #[test]
    fn test_index_filter() {
        let config = TokenizerConfig::new()
            .lowercase()
            .with_index(["hello", "world"]);
        let tokens = tokenize("Hello foo World bar", &config);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_index_with_prefix() {
        let config = TokenizerConfig::new()
            .lowercase()
            .with_prefix(4)
            .with_index(["auto", "tran"]);
        let tokens = tokenize("automobile transportation other", &config);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "auto");
        assert_eq!(tokens[1].text, "tran");
    }

    #[test]
    fn test_emoji_with_keep_punct() {
        let config = TokenizerConfig::new().keep_punctuation();
        let tokens = tokenize("I ❤️ NLP", &config);
        assert_eq!(tokens.len(), 3);
    }

    // --- trim_punctuation unit tests ---

    #[test]
    fn test_trim_basic() {
        assert_eq!(trim_punctuation("hello"), "hello");
        assert_eq!(trim_punctuation("(hello)"), "hello");
        assert_eq!(trim_punctuation("--test--"), "test");
        assert_eq!(trim_punctuation("..."), "");
    }

    #[test]
    fn test_trim_preserves_internal() {
        assert_eq!(trim_punctuation("don't"), "don't");
        assert_eq!(trim_punctuation("mother-in-law"), "mother-in-law");
        assert_eq!(trim_punctuation("test@email"), "test@email");
    }
}
