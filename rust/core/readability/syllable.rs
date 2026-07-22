//! English syllable estimation.
//!
//! A tuned vowel-group heuristic: count vowel groups, split known hiatus
//! pairs ("ia", "io", "eo", "ua", …) with consonant-context guards, then
//! apply word-final and internal silent-e adjustments and a small
//! exception table. Tuned against CMU Pronouncing Dictionary first
//! pronunciations (92.2% exact match over all ~125k entries, which are
//! roughly half proper names; common-prose accuracy is higher).
//!
//! The heuristic is English-calibrated. Non-Latin letters are treated as
//! consonants, so any letter run without recognized vowels contributes
//! zero and the per-token floor of one syllable applies. Accented Latin
//! vowels are folded to their base vowel before counting.

/// Exception table for high-frequency words the rules cannot reach.
/// Sorted by key for binary search; keys are normalized (lowercase,
/// accent-folded) whole letter-runs.
static EXCEPTIONS: &[(&str, u32)] = &[
    ("aisle", 1),
    ("aisles", 1),
    ("amphitheater", 5),
    ("amphitheaters", 5),
    ("anyone", 3),
    ("area", 3),
    ("areas", 3),
    ("average", 3),
    ("awfully", 2),
    ("beautiful", 3),
    ("beautifully", 3),
    ("business", 2),
    ("businesses", 3),
    ("camera", 3),
    ("carriage", 2),
    ("carriages", 3),
    ("chocolate", 2),
    ("colonel", 2),
    ("conscience", 2),
    ("create", 2),
    ("created", 3),
    ("creates", 2),
    ("creating", 3),
    ("creation", 3),
    ("diet", 2),
    ("different", 3),
    ("evening", 2),
    ("every", 2),
    ("everyone", 3),
    ("everything", 3),
    ("family", 3),
    ("friend", 1),
    ("friendly", 2),
    ("friends", 1),
    ("friendship", 2),
    ("idea", 3),
    ("ideas", 3),
    ("interest", 3),
    ("interested", 4),
    ("interesting", 4),
    ("island", 2),
    ("islands", 2),
    ("isle", 1),
    ("isles", 1),
    ("lion", 2),
    ("lions", 2),
    ("marriage", 2),
    ("marriages", 3),
    ("people", 2),
    ("poem", 2),
    ("poet", 2),
    ("quiet", 2),
    ("quietly", 3),
    ("restaurant", 3),
    ("science", 2),
    ("sciences", 3),
    ("several", 3),
    ("someone", 2),
    ("temperature", 4),
    ("theater", 3),
    ("theaters", 3),
    ("theatre", 3),
    ("theatres", 3),
    ("vegetable", 3),
    ("wednesday", 2),
];

#[inline]
fn is_vowel(b: u8) -> bool {
    matches!(b, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

/// Fold a lowercase char into the normalized byte alphabet:
/// ASCII letters map to themselves, accented Latin vowels fold to their
/// base vowel, other letters become `b'#'` (an opaque consonant), and
/// `None` means "not a letter".
#[inline]
fn fold_letter(ch: char) -> Option<u8> {
    if ch.is_ascii() {
        return if ch.is_ascii_alphabetic() {
            Some(ch.to_ascii_lowercase() as u8)
        } else {
            None
        };
    }
    if !ch.is_alphabetic() {
        return None;
    }
    Some(match ch {
        'à'..='å' | 'ā' | 'ă' | 'ą' => b'a',
        'è'..='ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => b'e',
        'ì'..='ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => b'i',
        'ò'..='ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => b'o',
        'ù'..='ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => b'u',
        'ý' | 'ÿ' => b'y',
        _ => b'#',
    })
}

/// Normalize a token into `buf`: lowercase, accent-folded letters;
/// apostrophes kept as `b'\''`; every other char becomes `b' '`
/// (a letter-run separator).
pub(crate) fn normalize_token(token: &str, buf: &mut Vec<u8>) {
    buf.clear();
    for ch in token.chars() {
        if ch == '\'' || ch == '\u{2019}' {
            buf.push(b'\'');
            continue;
        }
        // Lowercase may expand to multiple chars (e.g. İ); fold each.
        if ch.is_uppercase() {
            for lc in ch.to_lowercase() {
                push_folded(lc, buf);
            }
        } else {
            push_folded(ch, buf);
        }
    }
}

#[inline]
fn push_folded(ch: char, buf: &mut Vec<u8>) {
    match ch {
        // Acute-accented e is pronounced ("café", "déjà") — fold to 'a'
        // so the silent-e rules never swallow it.
        'é' => buf.push(b'a'),
        // Diaeresis marks a separately pronounced vowel ("naïve",
        // "Noël") — insert an opaque consonant to force a group split.
        'ï' => {
            buf.push(b'#');
            buf.push(b'i');
        }
        'ë' => {
            buf.push(b'#');
            buf.push(b'e');
        }
        _ => buf.push(fold_letter(ch).unwrap_or(b' ')),
    }
}

/// Count vowel groups in a normalized letter run, splitting hiatus
/// pairs by consonant context.
fn count_vowel_groups(w: &[u8]) -> i32 {
    let mut n = 0i32;
    let mut prev_v = false;
    for i in 0..w.len() {
        let c = w[i];
        let v = is_vowel(c);
        if v {
            if !prev_v {
                n += 1;
            } else {
                let a = w[i - 1];
                let before = if i >= 2 { w[i - 2] } else { 0 };
                let after = if i + 1 < w.len() { w[i + 1] } else { 0 };
                if a == b'i' && matches!(c, b'a' | b'o' | b'u') {
                    // ia/io/iu hiatus: "median", "violin", "medium" —
                    // unless a palatalizing consonant precedes ("nation",
                    // "special", "vision", "militia") or -ion ("union").
                    if c == b'a' && after == b't' {
                        n += 1; // "-iate", "-iation": split even after c/t
                    } else if matches!(before, b'c' | b's' | b'x' | b'g' | b't') {
                        // no split
                    } else if c == b'o' && after == b'n' && matches!(before, b'n' | b'l') {
                        // "union", "onion", "million", "billion"
                    } else {
                        n += 1;
                    }
                } else if a == b'i'
                    && c == b'e'
                    && after == b'n'
                    && !matches!(before, b'c' | b's' | b't' | b'v' | b'n')
                {
                    // "alien", "client", "orient" — not "patient",
                    // "ancient", "convenient"; "friend" is an exception
                    n += 1;
                } else if a == b'e' && c == b'o' && before != b'g' {
                    n += 1; // "video", "meteor" — not "pigeon"
                } else if a == b'u'
                    && matches!(c, b'a' | b'o' | b'i')
                    && !matches!(before, b'g' | b'q')
                {
                    n += 1; // "usual", "duo", "fluid" — not "guard", "quote"
                } else if a == b'a' && c == b'o' {
                    n += 1; // "chaos"
                }
            }
        }
        prev_v = v;
    }
    n
}

#[inline]
fn ends_with(w: &[u8], suf: &[u8]) -> bool {
    w.len() >= suf.len() && &w[w.len() - suf.len()..] == suf
}

/// Byte at offset `back` from the end (1-based, like Python's `w[-back]`),
/// or 0 when the word is too short.
#[inline]
fn from_end(w: &[u8], back: usize) -> u8 {
    if w.len() >= back {
        w[w.len() - back]
    } else {
        0
    }
}

/// Estimate syllables for one normalized lowercase letter run.
///
/// Returns 0 for vowel-less runs; callers clamp the per-token total to 1.
fn run_syllables(w: &[u8]) -> i32 {
    if w.is_empty() {
        return 0;
    }
    if let Ok(idx) = EXCEPTIONS.binary_search_by(|(k, _)| k.as_bytes().cmp(w)) {
        return EXCEPTIONS[idx].1 as i32;
    }
    let n = w.len();
    let mut count = count_vowel_groups(w);
    if n <= 3 {
        return count;
    }

    // ── word-final adjustments (first match wins) ──
    if ends_with(w, b"gue") || ends_with(w, b"que") {
        count -= 1; // "vague", "unique"
    } else if ends_with(w, b"lle") {
        count -= 1; // "belle", "-ville"
    } else if ends_with(w, b"re") && !is_vowel(from_end(w, 3)) && from_end(w, 3) != b'r' {
        // consonant+re is syllabic: "acre", "centre", "massacre"
    } else if ends_with(w, b"res") && !is_vowel(from_end(w, 4)) && from_end(w, 4) != b'r' {
        // "acres", "centres"
    } else if ends_with(w, b"e")
        && !(ends_with(w, b"ee")
            || ends_with(w, b"ie")
            || ends_with(w, b"oe")
            || ends_with(w, b"ye")
            || ends_with(w, b"ue"))
    {
        if ends_with(w, b"le") && !is_vowel(from_end(w, 3)) {
            // consonant+le is syllabic: "table"
        } else {
            count -= 1; // silent e: "make", "sale"
        }
    } else if ends_with(w, b"les") && !is_vowel(from_end(w, 4)) {
        // "tables"
    } else if ends_with(w, b"ies")
        || ends_with(w, b"ied")
        || ends_with(w, b"ees")
        || ends_with(w, b"eed")
        || ends_with(w, b"oes")
    {
        // "abilities", "accompanied", "employees", "agreed", "heroes" —
        // the vowel pair is syllabic; monosyllables like "ties"/"seed"/
        // "goes" have a single group and are unaffected.
    } else if ends_with(w, b"es") {
        if matches!(from_end(w, 3), b's' | b'x' | b'z')
            || ends_with(w, b"ches")
            || ends_with(w, b"shes")
            || ends_with(w, b"ges")
            || ends_with(w, b"ces")
        {
            // "boxes", "riches"
        } else {
            count -= 1; // "makes"
        }
    } else if ends_with(w, b"ed") {
        if matches!(from_end(w, 3), b't' | b'd')
            || (ends_with(w, b"led")
                && matches!(
                    from_end(w, 4),
                    b'b' | b'c' | b'd' | b'f' | b'g' | b'k' | b'p' | b's' | b't' | b'z'
                ))
        {
            // "wanted", "added", "tabled" — but "balled", "curled"
        } else {
            count -= 1; // "asked"
        }
    } else if ends_with(w, b"ing")
        && (matches!(from_end(w, 4), b'e' | b'o' | b'y')
            || (from_end(w, 4) == b'i' && from_end(w, 5) == b'i'))
    {
        count += 1; // "agreeing", "doing", "accompanying", "skiing"
    } else if ends_with(w, b"ism") || ends_with(w, b"isms") {
        count += 1; // "activism": -ism is two syllables
    } else if n >= 5
        && ((ends_with(w, b"ier") && !matches!(from_end(w, 4), b'd' | b't'))
            || (ends_with(w, b"iers") && !matches!(from_end(w, 5), b'd' | b't')))
    {
        count += 1; // "carrier", "easier" — not "soldier", "frontier"
    } else if ends_with(w, b"iest") || ends_with(w, b"ifier") || ends_with(w, b"ifiers") {
        count += 1; // "happiest", "amplifier"
    } else if ends_with(w, b"uing") && from_end(w, 5) == b'g' && !is_vowel(from_end(w, 6)) {
        count += 1; // "arguing" — but not "intriguing" (silent u)
    }

    // internal silent e in suffixed/compound forms: "lovely", "carefully",
    // "abatement", "baseman", "lifetime"
    const E_SUFFIXES: &[&[u8]] = &[
        b"ely", b"efully", b"eful", b"eless", b"eness", b"eman", b"etime", b"etimes", b"ewhere",
        b"ehouse",
    ];
    let mut matched = false;
    for suf in E_SUFFIXES {
        if ends_with(w, suf) && n > suf.len() + 1 && !is_vowel(from_end(w, suf.len() + 1)) {
            let is_leman = *suf == b"eman" && from_end(w, 5) == b'l' && !is_vowel(from_end(w, 6));
            if !is_leman {
                // "addleman"/"appleman" keep the syllabic -le-
                count -= 1;
            }
            matched = true;
            break;
        }
    }
    if !matched
        && ends_with(w, b"ement")
        && n > 6
        && !matches!(from_end(w, 6), b'l' | b'r')
        && !is_vowel(from_end(w, 6))
    {
        count -= 1; // "abatement" — but "element", "requirement"
    }
    if ends_with(w, b"ically") {
        count -= 1; // "academically": pronounced -icly
    }

    count.max(1)
}

/// Estimate syllables for a normalized token buffer (as produced by
/// [`normalize_token`]): sum letter runs, apply the possessive-sibilant
/// rule, clamp to at least 1.
pub(crate) fn normalized_syllables(buf: &[u8]) -> u32 {
    let mut total = 0i32;
    let mut start: Option<usize> = None;
    for (i, &b) in buf.iter().enumerate() {
        let is_letter = b != b' ' && b != b'\'';
        if is_letter {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            total += run_syllables(&buf[s..i]);
        }
    }
    if let Some(s) = start {
        total += run_syllables(&buf[s..]);
    }
    // possessive after a sibilant adds a syllable: "boss's", "advance's"
    if ends_with(buf, b"'s") {
        let stem = &buf[..buf.len() - 2];
        if matches!(stem.last(), Some(b's' | b'z' | b'x'))
            || ends_with(stem, b"ce")
            || ends_with(stem, b"ge")
            || ends_with(stem, b"ch")
            || ends_with(stem, b"sh")
        {
            total += 1;
        }
    }
    total.max(1) as u32
}

/// Estimate the number of syllables in `token`.
///
/// Deterministic, panic-free on any input, and always at least 1 for a
/// non-empty token. Non-letter characters split the token into letter
/// runs whose estimates are summed.
pub fn estimate_syllables(token: &str) -> u32 {
    if token.is_empty() {
        return 0;
    }
    let mut buf = Vec::with_capacity(token.len());
    normalize_token(token, &mut buf);
    normalized_syllables(&buf)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_syllables(cases: &[(&str, u32)]) {
        for &(w, expected) in cases {
            assert_eq!(
                estimate_syllables(w),
                expected,
                "word {w:?} expected {expected}"
            );
        }
    }

    #[test]
    fn exceptions_table_is_sorted() {
        for pair in EXCEPTIONS.windows(2) {
            assert!(pair[0].0 < pair[1].0, "{} !< {}", pair[0].0, pair[1].0);
        }
    }

    #[test]
    fn basic_words() {
        assert_syllables(&[
            ("the", 1),
            ("hello", 2),
            ("world", 1),
            ("ago", 2),
            ("oboe", 2),
            ("cat", 1),
            ("syllable", 3),
            ("estimation", 4),
        ]);
    }

    #[test]
    fn silent_e_family() {
        assert_syllables(&[
            ("make", 1),
            ("sale", 1),
            ("table", 2),
            ("tables", 2),
            ("little", 2),
            ("vague", 1),
            ("unique", 2),
            ("acre", 2),
            ("acres", 2),
            ("massacre", 3),
        ]);
    }

    #[test]
    fn suffix_family() {
        assert_syllables(&[
            ("asked", 1),
            ("wanted", 2),
            ("added", 2),
            ("tabled", 2),
            ("makes", 1),
            ("boxes", 2),
            ("abilities", 4),
            ("agreed", 2),
            ("trustees", 2),
            ("agreeing", 3),
            ("doing", 2),
            ("activism", 4),
            ("easier", 3),
            ("happiest", 3),
            ("lovely", 2),
            ("carefully", 3),
            ("abatement", 3),
        ]);
    }

    #[test]
    fn hiatus_family() {
        assert_syllables(&[
            ("median", 3),
            ("violin", 3),
            ("medium", 3),
            ("nation", 2),
            ("special", 2),
            ("vision", 2),
            ("union", 2),
            ("million", 2),
            ("video", 3),
            ("usual", 3),
            ("client", 2),
            ("patient", 2),
            ("appreciate", 4),
        ]);
    }

    #[test]
    fn exception_words() {
        assert_syllables(&[
            ("people", 2),
            ("business", 2),
            ("friend", 1),
            ("science", 2),
            ("Wednesday", 2),
            ("every", 2),
        ]);
    }

    #[test]
    fn tokens_with_punctuation_and_possessives() {
        assert_syllables(&[
            ("don't", 1),
            ("boss's", 2),
            ("advance's", 3),
            ("mother-in-law", 4),
            ("state-of-the-art", 4),
        ]);
    }

    #[test]
    fn unicode_and_degenerate_inputs() {
        assert_eq!(estimate_syllables(""), 0);
        assert_eq!(estimate_syllables("東京"), 1); // no vowels → floor
        assert_eq!(estimate_syllables("café"), 2);
        assert_eq!(estimate_syllables("naïve"), 2);
        assert_eq!(estimate_syllables("123"), 1); // no letters → floor
        assert_eq!(estimate_syllables("❤️"), 1);
        assert!(estimate_syllables("Ⅷ") >= 1);
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(estimate_syllables("HELLO"), estimate_syllables("hello"));
        assert_eq!(estimate_syllables("Table"), estimate_syllables("table"));
    }

    #[cfg(test)]
    mod props {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn never_panics_and_nonempty_is_positive(s in "\\PC*") {
                let n = estimate_syllables(&s);
                if !s.is_empty() {
                    prop_assert!(n >= 1);
                }
            }

            #[test]
            fn deterministic(s in "\\PC{0,40}") {
                prop_assert_eq!(estimate_syllables(&s), estimate_syllables(&s));
            }
        }
    }
}
