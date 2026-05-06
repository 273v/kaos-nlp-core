//! Phonetic distance algorithms: Soundex, Metaphone, Double Metaphone.
//!
//! These encode strings by pronunciation and compare the encodings.

use crate::core::algorithms::traits::{DistanceOutput, DistanceResult, StringDistance};

/// Soundex encoding — maps names to a 4-character code by pronunciation.
#[derive(Debug, Clone, Default)]
pub struct Soundex;

impl Soundex {
    /// Encode a string to its Soundex code.
    pub fn encode(&self, s: &str) -> String {
        let chars: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        if chars.is_empty() {
            return String::from("0000");
        }

        let mut code = String::with_capacity(4);
        code.push(chars[0].to_ascii_uppercase());

        let soundex_digit = |c: char| -> Option<char> {
            match c.to_ascii_lowercase() {
                'b' | 'f' | 'p' | 'v' => Some('1'),
                'c' | 'g' | 'j' | 'k' | 'q' | 's' | 'x' | 'z' => Some('2'),
                'd' | 't' => Some('3'),
                'l' => Some('4'),
                'm' | 'n' => Some('5'),
                'r' => Some('6'),
                _ => None, // a, e, i, o, u, h, w, y
            }
        };

        let mut last_digit = soundex_digit(chars[0]);
        for &c in &chars[1..] {
            if code.len() >= 4 {
                break;
            }
            let digit = soundex_digit(c);
            if let Some(d) = digit {
                if digit != last_digit {
                    code.push(d);
                }
            }
            last_digit = digit;
        }

        while code.len() < 4 {
            code.push('0');
        }
        code
    }
}

impl StringDistance for Soundex {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let code_a = self.encode(a);
        let code_b = self.encode(b);
        let matching = code_a
            .chars()
            .zip(code_b.chars())
            .filter(|(a, b)| a == b)
            .count();
        let sim = matching as f64 / 4.0;
        Ok(DistanceOutput::from_similarity(sim))
    }
}

/// Metaphone encoding — maps words to a pronunciation-based code.
/// Simplified implementation covering the most common English rules.
#[derive(Debug, Clone, Default)]
pub struct Metaphone;

impl Metaphone {
    /// Encode a string to its Metaphone code.
    pub fn encode(&self, s: &str) -> String {
        let s = s.to_ascii_uppercase();
        let chars: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        if chars.is_empty() {
            return String::new();
        }

        let mut result = String::with_capacity(8);
        let len = chars.len();
        let mut i = 0;

        // Handle initial special cases
        if len >= 2 {
            match (chars[0], chars[1]) {
                ('A', 'E') | ('G', 'N') | ('K', 'N') | ('P', 'N') | ('W', 'R') => i = 1,
                _ => {}
            }
        }
        if chars[0] == 'X' {
            result.push('S');
            i = 1;
        }

        let at = |idx: usize| -> char {
            if idx < len {
                chars[idx]
            } else {
                '\0'
            }
        };
        let is_vowel = |c: char| matches!(c, 'A' | 'E' | 'I' | 'O' | 'U');

        while i < len && result.len() < 6 {
            let c = chars[i];

            // Skip duplicate adjacent consonants (except C)
            if c != 'C' && i > 0 && chars[i - 1] == c {
                i += 1;
                continue;
            }

            match c {
                'A' | 'E' | 'I' | 'O' | 'U' if i == 0 => {
                    result.push(c);
                }
                'B' if !(i == len - 1 && i > 0 && chars[i - 1] == 'M') => {
                    result.push('B');
                }
                'C' => {
                    if at(i + 1) == 'I' || at(i + 1) == 'E' || at(i + 1) == 'Y' {
                        if at(i + 1) == 'I' && at(i + 2) == 'A' {
                            result.push('X');
                        } else {
                            result.push('S');
                        }
                    } else {
                        result.push('K');
                    }
                }
                'D' => {
                    if at(i + 1) == 'G'
                        && (at(i + 2) == 'I' || at(i + 2) == 'E' || at(i + 2) == 'Y')
                    {
                        result.push('J');
                    } else {
                        result.push('T');
                    }
                }
                'F' => result.push('F'),
                'G' => {
                    if i + 1 < len && at(i + 1) == 'H' && i + 2 < len && !is_vowel(at(i + 2)) {
                        // GH not before vowel: silent
                    } else if i > 0 && at(i + 1) == 'N' {
                        // GN: silent G
                    } else if i > 0 && chars[i - 1] == 'G' {
                        // double G: skip
                    } else {
                        if at(i + 1) == 'I' || at(i + 1) == 'E' || at(i + 1) == 'Y' {
                            result.push('J');
                        } else {
                            result.push('K');
                        }
                    }
                }
                'H' if is_vowel(at(i + 1)) && (i == 0 || !is_vowel(chars[i - 1])) => {
                    result.push('H');
                }
                'J' => result.push('J'),
                'K' if (i == 0 || chars[i - 1] != 'C') => {
                    result.push('K');
                }
                'L' => result.push('L'),
                'M' => result.push('M'),
                'N' => result.push('N'),
                'P' => {
                    if at(i + 1) == 'H' {
                        result.push('F');
                        i += 1;
                    } else {
                        result.push('P');
                    }
                }
                'Q' => result.push('K'),
                'R' => result.push('R'),
                'S' => {
                    if at(i + 1) == 'H'
                        || (at(i + 1) == 'I' && (at(i + 2) == 'O' || at(i + 2) == 'A'))
                    {
                        result.push('X');
                        if at(i + 1) == 'H' {
                            i += 1;
                        }
                    } else {
                        result.push('S');
                    }
                }
                'T' => {
                    if at(i + 1) == 'H' {
                        result.push('0'); // theta
                        i += 1;
                    } else if at(i + 1) == 'I' && (at(i + 2) == 'O' || at(i + 2) == 'A') {
                        result.push('X');
                    } else {
                        result.push('T');
                    }
                }
                'V' => result.push('F'),
                'W' | 'Y' if is_vowel(at(i + 1)) => {
                    result.push(c);
                }
                'X' => {
                    result.push('K');
                    result.push('S');
                }
                'Z' => result.push('S'),
                _ => {}
            }
            i += 1;
        }
        result
    }
}

impl StringDistance for Metaphone {
    fn distance(&self, a: &str, b: &str) -> DistanceResult<DistanceOutput> {
        let code_a = self.encode(a);
        let code_b = self.encode(b);
        if code_a.is_empty() && code_b.is_empty() {
            return Ok(DistanceOutput::from_similarity(1.0));
        }
        let max_len = code_a.len().max(code_b.len());
        let matching = code_a
            .chars()
            .zip(code_b.chars())
            .filter(|(a, b)| a == b)
            .count();
        let sim = matching as f64 / max_len as f64;
        Ok(DistanceOutput::from_similarity(sim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soundex_basic() {
        let s = Soundex;
        assert_eq!(s.encode("Robert"), "R163");
        assert_eq!(s.encode("Rupert"), "R163");
        // Same soundex => high similarity
        let r = s.distance("Robert", "Rupert").unwrap();
        assert!(r.similarity > 0.5);
    }

    #[test]
    fn test_soundex_identical() {
        let s = Soundex;
        let r = s.distance("Smith", "Smith").unwrap();
        assert_eq!(r.similarity, 1.0);
    }

    #[test]
    fn test_metaphone_basic() {
        let m = Metaphone;
        // "Smith" and "Smyth" should have similar encodings
        let code1 = m.encode("Smith");
        let code2 = m.encode("Smyth");
        assert_eq!(code1, code2);
    }
}
