//! Token representation for the Punkt algorithm.
//!
//! Ported from nupunkt-rs/src/tokens.rs.

use once_cell::sync::Lazy;
use regex::Regex;

static RE_NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new(r"^-?[\.,]?\d[\d,\.-]*\.?$").unwrap());
static RE_ELLIPSIS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.\.+$").unwrap());
static RE_INITIAL: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[^\W\d]\.$").unwrap());
static RE_ALPHA: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[^\W\d]+$").unwrap());

/// A token in the Punkt algorithm.
#[derive(Debug, Clone)]
pub struct PunktToken {
    pub tok: String,
    pub parastart: bool,
    pub linestart: bool,
    pub sentbreak: bool,
    pub abbr: bool,
    pub ellipsis: bool,
    pub period_final: bool,
    pub exclamation_final: bool,
    pub question_final: bool,
    pub semicolon_final: bool,
    pub sentence_end_punct: bool,
    pub token_type: String,
    pub valid_abbrev_candidate: bool,
    pub spaces_after: u8,
    pub has_newline_after: bool,
    pub char_position: Option<usize>,
    pub byte_position: Option<usize>,
    first_upper: bool,
    first_lower: bool,
}

impl PunktToken {
    pub fn new(tok: impl Into<String>, parastart: bool, linestart: bool) -> Self {
        let tok = tok.into();
        let period_final = tok.ends_with('.');
        let exclamation_final = tok.ends_with('!');
        let question_final = tok.ends_with('?');
        let semicolon_final = tok.ends_with(';');
        let sentence_end_punct = period_final || exclamation_final || question_final;
        let token_type = Self::get_token_type(&tok);

        let first_char = tok.chars().next();
        let first_upper = first_char.is_some_and(|c| c.is_uppercase());
        let first_lower = first_char.is_some_and(|c| c.is_lowercase());

        let valid_abbrev_candidate = if period_final {
            let has_alpha = tok.chars().any(|c| c.is_alphabetic());
            let alpha_count = tok.chars().filter(|c| c.is_alphabetic()).count();
            let digit_count = tok.chars().filter(|c| c.is_numeric()).count();
            has_alpha && token_type != "##number##" && alpha_count >= digit_count && tok.len() <= 10
        } else {
            false
        };

        Self {
            tok,
            parastart,
            linestart,
            sentbreak: false,
            abbr: false,
            ellipsis: false,
            period_final,
            exclamation_final,
            question_final,
            semicolon_final,
            sentence_end_punct,
            token_type,
            valid_abbrev_candidate,
            spaces_after: 1,
            has_newline_after: false,
            char_position: None,
            byte_position: None,
            first_upper,
            first_lower,
        }
    }

    fn get_token_type(tok: &str) -> String {
        if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
            return "##number##".to_string();
        }
        if tok
            .chars()
            .any(|c| c.is_alphabetic() && !c.is_ascii_digit())
        {
            return tok.to_lowercase();
        }
        if RE_NUMBER.is_match(tok) {
            "##number##".to_string()
        } else {
            tok.to_lowercase()
        }
    }

    #[inline]
    pub fn type_no_period(&self) -> String {
        if self.token_type.ends_with('.') && self.token_type.len() > 1 {
            self.token_type[..self.token_type.len() - 1].to_string()
        } else {
            self.token_type.clone()
        }
    }

    #[inline]
    pub fn type_no_sentence_punct(&self) -> String {
        if self.token_type.len() > 1
            && (self.token_type.ends_with('.')
                || self.token_type.ends_with('!')
                || self.token_type.ends_with('?'))
        {
            self.token_type[..self.token_type.len() - 1].to_string()
        } else {
            self.token_type.clone()
        }
    }

    #[inline]
    pub fn type_no_sentperiod(&self) -> String {
        if self.sentbreak {
            self.type_no_period()
        } else {
            self.token_type.clone()
        }
    }

    #[inline]
    pub fn first_upper(&self) -> bool {
        self.first_upper
    }

    #[inline]
    pub fn first_lower(&self) -> bool {
        self.first_lower
    }

    #[inline]
    pub fn is_ellipsis(&self) -> bool {
        match self.tok.as_str() {
            "..." | ".." | "\u{2026}" => true,
            _ if self.tok.ends_with("\u{2026}") => true,
            _ if self.tok.len() >= 2 && self.tok.ends_with("..") => RE_ELLIPSIS.is_match(&self.tok),
            _ => false,
        }
    }

    #[inline]
    pub fn is_number(&self) -> bool {
        self.token_type == "##number##"
    }

    #[inline]
    pub fn is_initial(&self) -> bool {
        if self.tok.len() == 2 && self.tok.ends_with('.') {
            if let Some(first) = self.tok.chars().next() {
                if first.is_alphabetic() && !first.is_numeric() {
                    return true;
                }
            }
        }
        RE_INITIAL.is_match(&self.tok)
    }

    #[inline]
    pub fn is_alpha(&self) -> bool {
        if !self.tok.is_empty() && self.tok.chars().all(|c| c.is_alphabetic()) {
            return true;
        }
        RE_ALPHA.is_match(&self.tok)
    }

    #[inline]
    pub fn is_non_punct(&self) -> bool {
        self.tok.chars().any(|c| c.is_alphanumeric())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_type_normalization() {
        let t = PunktToken::new("Hello", false, false);
        assert_eq!(t.token_type, "hello");

        let t = PunktToken::new("123", false, false);
        assert_eq!(t.token_type, "##number##");

        let t = PunktToken::new("3.14", false, false);
        assert_eq!(t.token_type, "##number##");
    }

    #[test]
    fn test_period_detection() {
        let t = PunktToken::new("end.", false, false);
        assert!(t.period_final);
        assert!(t.sentence_end_punct);

        let t = PunktToken::new("wow!", false, false);
        assert!(t.exclamation_final);
        assert!(t.sentence_end_punct);

        let t = PunktToken::new("hello", false, false);
        assert!(!t.sentence_end_punct);
    }

    #[test]
    fn test_type_no_period() {
        let t = PunktToken::new("Dr.", false, false);
        assert_eq!(t.type_no_period(), "dr");
    }

    #[test]
    fn test_ellipsis() {
        assert!(PunktToken::new("...", false, false).is_ellipsis());
        assert!(PunktToken::new("..", false, false).is_ellipsis());
        assert!(PunktToken::new("\u{2026}", false, false).is_ellipsis());
        assert!(!PunktToken::new("hello", false, false).is_ellipsis());
    }

    #[test]
    fn test_initial() {
        assert!(PunktToken::new("J.", false, false).is_initial());
        assert!(PunktToken::new("A.", false, false).is_initial());
        assert!(!PunktToken::new("Dr.", false, false).is_initial());
    }

    #[test]
    fn test_first_upper_lower() {
        let t = PunktToken::new("Hello", false, false);
        assert!(t.first_upper());
        assert!(!t.first_lower());

        let t = PunktToken::new("hello", false, false);
        assert!(!t.first_upper());
        assert!(t.first_lower());
    }

    #[test]
    fn test_valid_abbrev_candidate() {
        let t = PunktToken::new("Dr.", false, false);
        assert!(t.valid_abbrev_candidate);

        let t = PunktToken::new("hello", false, false);
        assert!(!t.valid_abbrev_candidate);

        let t = PunktToken::new("123.", false, false);
        assert!(!t.valid_abbrev_candidate);
    }
}
