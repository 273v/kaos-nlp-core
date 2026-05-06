//! Token-level property classification.
//!
//! Classifies tokens by structural properties: case, numeric, symbolic,
//! punctuation, hyphenation, abbreviation, emoji, etc.
//!
//! Uses the existing `characters` module for ICU4X-based character classification
//! with ASCII fast paths.

use crate::core::characters::CharacterProperties;

/// Classification results for a token.
#[derive(Debug, Clone, Default)]
pub struct TokenPropertyFlags {
    /// All characters are letters.
    pub is_letter_word: bool,
    /// All letters are uppercase.
    pub is_uppercase_word: bool,
    /// All letters are lowercase.
    pub is_lowercase_word: bool,
    /// Has both upper and lower case letters (not title case).
    pub is_mixed_case_word: bool,
    /// First letter uppercase, rest lowercase.
    pub is_title_case_word: bool,
    /// All characters are digits.
    pub is_numeric_word: bool,
    /// Contains both letters and digits.
    pub is_alphanumeric_word: bool,
    /// Starts with a digit.
    pub starts_with_digit: bool,
    /// Contains punctuation characters.
    pub has_punctuation: bool,
    /// Contains a dash/hyphen.
    pub has_dash: bool,
    /// Contains at least one letter-dash-letter pattern.
    pub is_hyphenated: bool,
    /// Looks like an abbreviation (e.g., "Dr.", "U.S.A.", single uppercase letter).
    pub is_abbreviation: bool,
    /// Contains emoji characters.
    pub has_emoji: bool,
    /// Contains only symbol characters (€, ©, etc.).
    pub is_symbolic_word: bool,
    /// Ends with terminal punctuation (. ! ?).
    pub ends_with_terminal: bool,
}

/// Classify a token and return all property flags.
pub fn classify_token(token: &str) -> TokenPropertyFlags {
    if token.is_empty() {
        return TokenPropertyFlags::default();
    }

    let chars: Vec<char> = token.chars().collect();
    let n = chars.len();

    let mut has_upper = false;
    let mut has_lower = false;
    let mut has_letter = false;
    let mut has_digit = false;
    let mut all_letter = true;
    let mut all_digit = true;
    let mut all_symbol = true;
    let mut has_punct = false;
    let mut has_dash = false;
    let mut has_emoji = false;

    for &ch in &chars {
        if CharacterProperties::is_letter(ch) {
            has_letter = true;
            all_digit = false;
            all_symbol = false;
            if ch.is_uppercase() {
                has_upper = true;
            }
            if ch.is_lowercase() {
                has_lower = true;
            }
        } else if CharacterProperties::is_number(ch) {
            has_digit = true;
            all_letter = false;
            all_symbol = false;
        } else if CharacterProperties::is_punctuation(ch) {
            has_punct = true;
            all_letter = false;
            all_digit = false;
            all_symbol = false;
            if ch == '-'
                || ch == '\u{2010}'
                || ch == '\u{2011}'
                || ch == '\u{2012}'
                || ch == '\u{2013}'
                || ch == '\u{2014}'
                || ch == '\u{2015}'
            {
                has_dash = true;
            }
        } else if is_emoji(ch) {
            has_emoji = true;
            all_letter = false;
            all_digit = false;
        } else {
            all_letter = false;
            all_digit = false;
            // Symbol check is handled by CharacterProperties::is_symbol
            if !CharacterProperties::is_symbol(ch) {
                all_symbol = false;
            }
        }
    }

    // Letter-based classification
    let is_letter_word = has_letter && all_letter;
    let is_uppercase_word = is_letter_word && has_upper && !has_lower;
    let is_lowercase_word = is_letter_word && has_lower && !has_upper;

    let is_title_case_word = is_letter_word
        && n >= 2
        && chars[0].is_uppercase()
        && chars[1..].iter().all(|c| c.is_lowercase());

    let is_mixed_case_word = is_letter_word && has_upper && has_lower && !is_title_case_word;

    // Numeric
    let is_numeric_word = has_digit && all_digit;
    let is_alphanumeric_word = has_letter && has_digit && !has_punct;
    let starts_with_digit = chars[0].is_ascii_digit() || CharacterProperties::is_number(chars[0]);

    // Hyphenation: letter-dash-letter pattern (any dash type)
    let is_hyphenated = if has_dash && n >= 3 {
        chars.windows(3).any(|w| {
            CharacterProperties::is_letter(w[0])
                && is_dash_char(w[1])
                && CharacterProperties::is_letter(w[2])
        })
    } else {
        false
    };

    // Abbreviation detection
    let is_abbreviation = detect_abbreviation(token, &chars);

    // Symbol
    let is_symbolic_word = all_symbol && !has_letter && !has_digit && !has_punct && n > 0;

    // Terminal punctuation
    let last = chars[n - 1];
    let ends_with_terminal = last == '.' || last == '!' || last == '?';

    TokenPropertyFlags {
        is_letter_word,
        is_uppercase_word,
        is_lowercase_word,
        is_mixed_case_word,
        is_title_case_word,
        is_numeric_word,
        is_alphanumeric_word,
        starts_with_digit,
        has_punctuation: has_punct,
        has_dash,
        is_hyphenated,
        is_abbreviation,
        has_emoji,
        is_symbolic_word,
        ends_with_terminal,
    }
}

/// Detect if a token is an abbreviation.
///
/// Patterns:
/// - Single uppercase letter: "A"
/// - Letters separated by periods: "U.S.A.", "Dr.", "i.e."
/// - Short token ending with period: "etc.", "Corp."
fn detect_abbreviation(_token: &str, chars: &[char]) -> bool {
    let n = chars.len();
    if n == 0 {
        return false;
    }

    // Single uppercase letter
    if n == 1 && chars[0].is_uppercase() {
        return true;
    }

    // Ends with period and is short (< 8 chars)
    if chars[n - 1] == '.' && n <= 8 {
        // Check for dotted pattern: "U.S.A." or "U.S."
        if n >= 3 {
            let non_period: Vec<char> = chars.iter().filter(|&&c| c != '.').copied().collect();
            let period_count = n - non_period.len();
            // If periods are roughly as many as letters, it's a dotted abbreviation
            if period_count >= non_period.len() && non_period.iter().all(|c| c.is_alphabetic()) {
                return true;
            }
        }
        // Short word ending in period with all letters before
        if n <= 5 && chars[..n - 1].iter().all(|c| c.is_alphabetic()) {
            return true;
        }
    }

    false
}

/// Check if a character is a dash/hyphen (any type).
#[inline]
fn is_dash_char(ch: char) -> bool {
    ch == '-'
        || ch == '\u{2010}' // Hyphen
        || ch == '\u{2011}' // Non-breaking hyphen
        || ch == '\u{2012}' // Figure dash
        || ch == '\u{2013}' // En dash
        || ch == '\u{2014}' // Em dash
        || ch == '\u{2015}' // Horizontal bar
}

/// Check if a character is an emoji.
///
/// Simplified check covering common emoji ranges.
fn is_emoji(ch: char) -> bool {
    let cp = ch as u32;
    // Common emoji ranges
    (0x1F600..=0x1F64F).contains(&cp) // Emoticons
        || (0x1F300..=0x1F5FF).contains(&cp) // Misc Symbols and Pictographs
        || (0x1F680..=0x1F6FF).contains(&cp) // Transport and Map
        || (0x1F1E0..=0x1F1FF).contains(&cp) // Flags
        || (0x2600..=0x26FF).contains(&cp) // Misc symbols
        || (0x2700..=0x27BF).contains(&cp) // Dingbats
        || (0xFE00..=0xFE0F).contains(&cp) // Variation Selectors
        || (0x1F900..=0x1F9FF).contains(&cp) // Supplemental Symbols
        || (0x1FA00..=0x1FA6F).contains(&cp) // Chess Symbols
        || (0x1FA70..=0x1FAFF).contains(&cp) // Symbols Extended-A
        || ch == '❤' || ch == '⭐' || ch == '✅' || ch == '❌'
}

// ─── Individual property functions ──────────────────────────────────────────

/// Check if token consists entirely of letters.
pub fn is_letter_word(token: &str) -> bool {
    !token.is_empty() && token.chars().all(CharacterProperties::is_letter)
}

/// Check if token is all uppercase letters.
pub fn is_uppercase_word(token: &str) -> bool {
    is_letter_word(token) && token.chars().all(|c| c.is_uppercase())
}

/// Check if token is all lowercase letters.
pub fn is_lowercase_word(token: &str) -> bool {
    is_letter_word(token) && token.chars().all(|c| c.is_lowercase())
}

/// Check if token is title case (first letter upper, rest lower).
pub fn is_title_case_word(token: &str) -> bool {
    let mut chars = token.chars();
    match chars.next() {
        Some(first) if first.is_uppercase() => chars.all(|c| c.is_lowercase()),
        _ => false,
    }
}

/// Check if token contains a hyphen with letters on both sides.
pub fn is_hyphenated(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    chars.len() >= 3
        && chars.windows(3).any(|w| {
            CharacterProperties::is_letter(w[0])
                && is_dash_char(w[1])
                && CharacterProperties::is_letter(w[2])
        })
}

/// Check if token is all digits.
pub fn is_numeric_word(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|c| c.is_ascii_digit() || CharacterProperties::is_number(c))
}

/// Check if token contains both letters and digits but no punctuation.
pub fn is_alphanumeric_word(token: &str) -> bool {
    classify_token(token).is_alphanumeric_word
}

/// Check if token looks like an abbreviation.
pub fn is_abbreviation(token: &str) -> bool {
    classify_token(token).is_abbreviation
}

/// Check if token contains emoji characters.
pub fn has_emoji(token: &str) -> bool {
    token.chars().any(is_emoji)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letter_word() {
        let flags = classify_token("hello");
        assert!(flags.is_letter_word);
        assert!(flags.is_lowercase_word);
        assert!(!flags.is_uppercase_word);
        assert!(!flags.is_title_case_word);
    }

    #[test]
    fn test_uppercase_word() {
        let flags = classify_token("FBI");
        assert!(flags.is_uppercase_word);
        assert!(!flags.is_lowercase_word);
    }

    #[test]
    fn test_title_case() {
        let flags = classify_token("Hello");
        assert!(flags.is_title_case_word);
        assert!(!flags.is_uppercase_word);
        assert!(!flags.is_lowercase_word);
    }

    #[test]
    fn test_mixed_case() {
        let flags = classify_token("iPhone");
        assert!(flags.is_mixed_case_word);
        assert!(!flags.is_title_case_word);
    }

    #[test]
    fn test_numeric() {
        let flags = classify_token("12345");
        assert!(flags.is_numeric_word);
        assert!(!flags.is_letter_word);
    }

    #[test]
    fn test_alphanumeric() {
        let flags = classify_token("abc123");
        assert!(flags.is_alphanumeric_word);
        assert!(!flags.is_numeric_word);
        assert!(!flags.is_letter_word);
    }

    #[test]
    fn test_hyphenated() {
        assert!(is_hyphenated("mother-in-law"));
        assert!(is_hyphenated("well-known"));
        assert!(!is_hyphenated("hello"));
        assert!(!is_hyphenated("-test"));
    }

    #[test]
    fn test_abbreviation_dotted() {
        let flags = classify_token("U.S.A.");
        assert!(flags.is_abbreviation);

        let flags = classify_token("Dr.");
        assert!(flags.is_abbreviation);
    }

    #[test]
    fn test_abbreviation_single_letter() {
        let flags = classify_token("A");
        assert!(flags.is_abbreviation);
    }

    #[test]
    fn test_not_abbreviation() {
        let flags = classify_token("hello");
        assert!(!flags.is_abbreviation);

        let flags = classify_token("Washington");
        assert!(!flags.is_abbreviation);
    }

    #[test]
    fn test_punctuation() {
        let flags = classify_token("hello!");
        assert!(flags.has_punctuation);
        assert!(flags.ends_with_terminal);
    }

    #[test]
    fn test_emoji() {
        assert!(has_emoji("hello😀"));
        assert!(!has_emoji("hello"));
    }

    #[test]
    fn test_symbolic() {
        let flags = classify_token("€");
        assert!(flags.is_symbolic_word);

        let flags = classify_token("©™");
        assert!(flags.is_symbolic_word);
    }

    #[test]
    fn test_starts_with_digit() {
        let flags = classify_token("3rd");
        assert!(flags.starts_with_digit);

        let flags = classify_token("hello");
        assert!(!flags.starts_with_digit);
    }

    #[test]
    fn test_empty_token() {
        let flags = classify_token("");
        assert!(!flags.is_letter_word);
        assert!(!flags.is_numeric_word);
    }

    #[test]
    fn test_dash_types() {
        let flags = classify_token("hello\u{2013}world"); // en-dash
        assert!(flags.has_dash);
        assert!(flags.is_hyphenated);
    }

    #[test]
    fn test_individual_functions() {
        assert!(is_letter_word("hello"));
        assert!(!is_letter_word("hello1"));
        assert!(is_uppercase_word("FBI"));
        assert!(!is_uppercase_word("Fbi"));
        assert!(is_lowercase_word("hello"));
        assert!(is_title_case_word("Hello"));
        assert!(is_numeric_word("123"));
        assert!(!is_numeric_word("12a"));
    }

    #[test]
    fn test_cjk_letter_word() {
        let flags = classify_token("\u{6771}\u{4EAC}"); // 東京
        assert!(
            flags.is_letter_word,
            "CJK characters should be classified as letter word"
        );
    }

    #[test]
    fn test_empty_string_individual() {
        assert!(!is_letter_word(""));
        assert!(!is_numeric_word(""));
        assert!(!has_emoji(""));
    }
}
