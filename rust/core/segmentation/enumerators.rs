//! Enumerator parser — recognise leading list / heading prefixes such as
//! `1.`, `(a)`, `I.`, `Section 5`, `§ 5.2`.
//!
//! `parse_enumerator(line)` returns `Some(Enumerator)` when `line` begins
//! with a recognisable enumerator and `None` otherwise. The output struct
//! is `Copy`, holds no allocations, and reports byte offsets into the
//! input slice — same convention as `LineRecord` and `Normalized`.
//!
//! Design reference: `docs/SECTION_HEADING_PRIMITIVES_RESEARCH.md`
//! (`## Enumerator parser (P3) — design reference`). Every grammar rule and
//! threshold below traces back to a Q-section of that doc.
//!
//! Hot-path budget: < 200 ns for the bare forms (Roman / Decimal / Alpha /
//! Parenthetical), < 500 ns for the word-prefixed form. Hand-rolled byte
//! loops + one shared Aho-Corasick automaton; no `regex` dependency.

use std::sync::{Arc, LazyLock};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use serde::{Deserialize, Serialize};

// ─── Public types ─────────────────────────────────────────────────────────

/// Twelve kinds of enumerator the parser recognises. Case-pair separation
/// (Upper / Lower for Roman + Alpha) matters for hierarchy inference
/// downstream — outer levels conventionally use UPPER, inner LOWER.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum EnumKind {
    /// `I.`, `II.`, `IV.`, `XIII.` — uppercase Roman.
    RomanUpper = 0,
    /// `i.`, `ii.`, `iv.`, `xiii.` — lowercase Roman.
    RomanLower = 1,
    /// `A.`, `B.`, … `Z.` — uppercase single ASCII letter (excluding `I`/`V`).
    AlphaUpper = 2,
    /// `a.`, `b.`, … `z.` — lowercase single ASCII letter (excluding `i`/`v`).
    AlphaLower = 3,
    /// `1.`, `1.1`, `1.2.3` — Arabic decimal, possibly dotted to depth 4.
    Decimal = 4,
    /// `(a)`, `(B)`. Inner content is single-segment Alpha.
    ParenAlpha = 5,
    /// `(1)`, `(42)`. Inner content is single-segment Decimal (no dots).
    ParenDecimal = 6,
    /// `(i)`, `(IV)`. Inner content is Roman.
    ParenRoman = 7,
    /// `§ 5`, `§ 5.2(a)` — the section sigil. Mapped to `Section` kind too.
    Section = 8,
    /// `Section 5`, `Sec. 5` — the word "Section" or its abbreviation.
    SectionWord = 9,
    /// `Chapter 7`, `Title 11`, etc. — top-of-CFR-hierarchy keywords.
    ChapterWord = 10,
    /// `Subpart B`, `Subchapter II`, `Article III` — mid/inner hierarchy.
    SubpartWord = 11,
    /// `- item`, `* item`, `+ item`, `• item` — Markdown / typographic
    /// bullet. Carries no ordinal; `value = 0`. F-R3 fix.
    Bullet = 12,
}

/// One detected enumerator at the start of a line.
///
/// All offsets are **byte offsets** into the input `&str`. The PyO3 binding
/// converts to char offsets at the FFI boundary per the standing rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Enumerator {
    pub kind: EnumKind,
    /// Packed value:
    ///
    /// * `Decimal` / `ParenDecimal`: bytes packed `(s1<<24)|(s2<<16)|(s3<<8)|s4`,
    ///   where `s1..s4` are dotted-decimal segments and unfilled segments are
    ///   zero. `1.2.3` → `0x01020300`. Each segment is 0..=255; values
    ///   exceeding that range cause the parser to return `None`.
    /// * `RomanUpper` / `RomanLower` / `ParenRoman`: the Roman value, 1..=3999.
    /// * `AlphaUpper` / `AlphaLower` / `ParenAlpha`: 1 (`A`/`a`) .. 26 (`Z`/`z`).
    /// * `Section` / `SectionWord` / `ChapterWord` / `SubpartWord`: the
    ///   trailing value's encoding under whatever sub-kind matched
    ///   (Decimal-packed if numeric, alpha if letter, Roman if Roman).
    pub value: u32,
    /// Number of dotted-decimal segments (Decimal / ParenDecimal). 1 for all
    /// other kinds.
    pub depth: u8,
    /// Byte offset within the input slice where the enumerator's source text
    /// begins (always 0 in current grammar — leading whitespace must be
    /// stripped by the caller).
    pub raw_start: u32,
    /// Byte offset (exclusive) where the enumerator's source text ends.
    pub raw_end: u32,
    /// Byte offset where the heading text begins (typically the position
    /// after a trailing space). For `Section 5.2 Title`, `prefix_end` points
    /// at `T`.
    pub prefix_end: u32,
}

// ─── Word-prefix lexicon registry (P7.0e) ─────────────────────────────────
//
// Each lexicon supplies a parallel-array of `(pattern_str, EnumKind)`. The
// AC automaton is built once per built-in variant via `LazyLock`. Custom
// lexicons own their automaton.
//
// **Generality contract** (per `INTEGRATION_BOUNDARIES.md` "Out-of-domain
// rule"): every Western-language legal lexicon must include at least one
// hierarchy-level (`Article` or equivalent) and one chapter-level
// (`Chapter` / `Titre` / `Capítulo` / `Capitolo` / `Capítulo`) keyword.
// Markdown ATX is a special case — depth is read from the leading `#` count,
// not from a keyword.

/// One entry in a word-prefix lexicon: literal pattern + the `EnumKind` to
/// emit on match.
#[derive(Debug, Clone)]
struct LexEntry {
    pattern: &'static str,
    kind: EnumKind,
}

/// Built-in lexicons + custom + Markdown.
#[derive(Debug, Clone, Default)]
pub enum WordLexicon {
    /// Anglo-American legal: `Section / Chapter / Title / Subpart / …`.
    /// Backward-compatible default.
    #[default]
    EnglishLegalUs,
    /// French legal: `Article / Chapitre / Titre / Section / Annexe / …`.
    /// Title-case + lowercase + uppercase variants are all included so the
    /// case-insensitive AC matches any source casing.
    FrenchLegal,
    /// German legal: `Artikel / Kapitel / Titel / Abschnitt / Anhang / …`.
    GermanLegal,
    /// Spanish legal: `Artículo / Capítulo / Título / Sección / Anexo / …`.
    SpanishLegal,
    /// Italian legal: `Articolo / Capo / Titolo / Sezione / Allegato / …`.
    ItalianLegal,
    /// Portuguese legal: `Artigo / Capítulo / Título / Secção / Anexo / …`.
    PortugueseLegal,
    /// Markdown ATX: `# / ## / ### …` — depth read from leading `#` count.
    MarkdownAtx,
    /// Caller-supplied lexicon. Patterns must be ASCII for full case-fold
    /// coverage; non-ASCII patterns work but only Title-case / lower / UPPER
    /// at byte level (no full Unicode case fold).
    Custom(Arc<CustomLexicon>),
}

/// Caller-supplied word-prefix lexicon.
#[derive(Debug)]
pub struct CustomLexicon {
    automaton: AhoCorasick,
    kinds: Vec<EnumKind>,
}

impl CustomLexicon {
    /// Build a custom lexicon from `(pattern, kind)` pairs. Patterns are
    /// matched ASCII-case-insensitively under `LeftmostLongest` — same as
    /// the built-in lexicons.
    pub fn new(entries: Vec<(String, EnumKind)>) -> Result<Self, String> {
        if entries.is_empty() {
            return Err("CustomLexicon requires at least one entry".into());
        }
        let patterns: Vec<&str> = entries.iter().map(|(p, _)| p.as_str()).collect();
        let kinds: Vec<EnumKind> = entries.iter().map(|(_, k)| *k).collect();
        let automaton = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .map_err(|e| e.to_string())?;
        Ok(Self { automaton, kinds })
    }
}

// ── Built-in lexicons. Each entry list ends with the `EnumKind` mapping ──

const ENGLISH_LEGAL_US: &[LexEntry] = &[
    LexEntry {
        pattern: "Section",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "Sec.",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "Chapter",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Subchapter",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "Subpart",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "Subtitle",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Title",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Part",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Appendix",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Schedule",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Article",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "Paragraph",
        kind: EnumKind::SubpartWord,
    },
];

// For each non-ASCII Western lexicon we list TitleCase + lowercase + UPPER
// variants so ASCII-case-insensitive AC covers all real casings. Non-ASCII
// chars (`é`, `ó`, `ç`, `ã`) do not participate in ASCII case folding, so
// the explicit triple is necessary.

const FRENCH_LEGAL: &[LexEntry] = &[
    LexEntry {
        pattern: "Article",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "Chapitre",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Titre",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Section",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "Annexe",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Paragraphe",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "Livre",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Préambule",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "préambule",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "PRÉAMBULE",
        kind: EnumKind::ChapterWord,
    },
];

const GERMAN_LEGAL: &[LexEntry] = &[
    LexEntry {
        pattern: "Artikel",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "Kapitel",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Titel",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Abschnitt",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "Anhang",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Paragraph",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "Buch",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Teil",
        kind: EnumKind::ChapterWord,
    },
];

const SPANISH_LEGAL: &[LexEntry] = &[
    LexEntry {
        pattern: "Artículo",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "artículo",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "ARTÍCULO",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "Capítulo",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "capítulo",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "CAPÍTULO",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Título",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "título",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "TÍTULO",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Sección",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "sección",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "SECCIÓN",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "Anexo",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Libro",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Parte",
        kind: EnumKind::ChapterWord,
    },
];

const ITALIAN_LEGAL: &[LexEntry] = &[
    LexEntry {
        pattern: "Articolo",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "Capo",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Capitolo",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Titolo",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Sezione",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "Allegato",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Libro",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Parte",
        kind: EnumKind::ChapterWord,
    },
];

const PORTUGUESE_LEGAL: &[LexEntry] = &[
    LexEntry {
        pattern: "Artigo",
        kind: EnumKind::SubpartWord,
    },
    LexEntry {
        pattern: "Capítulo",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "capítulo",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "CAPÍTULO",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Título",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "título",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "TÍTULO",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Secção",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "secção",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "SECÇÃO",
        kind: EnumKind::SectionWord,
    },
    // Brazilian Portuguese spelling (no ç in Seção).
    LexEntry {
        pattern: "Seção",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "seção",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "SEÇÃO",
        kind: EnumKind::SectionWord,
    },
    LexEntry {
        pattern: "Anexo",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Livro",
        kind: EnumKind::ChapterWord,
    },
    LexEntry {
        pattern: "Parte",
        kind: EnumKind::ChapterWord,
    },
];

fn build_automaton(entries: &[LexEntry]) -> (AhoCorasick, Vec<EnumKind>) {
    let patterns: Vec<&str> = entries.iter().map(|e| e.pattern).collect();
    let kinds: Vec<EnumKind> = entries.iter().map(|e| e.kind).collect();
    let automaton = AhoCorasickBuilder::new()
        .ascii_case_insensitive(true)
        .match_kind(MatchKind::LeftmostLongest)
        .build(&patterns)
        .expect("built-in lexicon patterns are valid");
    (automaton, kinds)
}

static EN_LEGAL_AUT: LazyLock<(AhoCorasick, Vec<EnumKind>)> =
    LazyLock::new(|| build_automaton(ENGLISH_LEGAL_US));
static FR_LEGAL_AUT: LazyLock<(AhoCorasick, Vec<EnumKind>)> =
    LazyLock::new(|| build_automaton(FRENCH_LEGAL));
static DE_LEGAL_AUT: LazyLock<(AhoCorasick, Vec<EnumKind>)> =
    LazyLock::new(|| build_automaton(GERMAN_LEGAL));
static ES_LEGAL_AUT: LazyLock<(AhoCorasick, Vec<EnumKind>)> =
    LazyLock::new(|| build_automaton(SPANISH_LEGAL));
static IT_LEGAL_AUT: LazyLock<(AhoCorasick, Vec<EnumKind>)> =
    LazyLock::new(|| build_automaton(ITALIAN_LEGAL));
static PT_LEGAL_AUT: LazyLock<(AhoCorasick, Vec<EnumKind>)> =
    LazyLock::new(|| build_automaton(PORTUGUESE_LEGAL));

impl WordLexicon {
    /// Return the AC automaton + parallel `EnumKind` slice for word-shaped
    /// lexicons. `MarkdownAtx` returns `None` (it has no AC; depth comes
    /// from `#` count instead).
    fn automaton(&self) -> Option<(&AhoCorasick, &[EnumKind])> {
        match self {
            WordLexicon::EnglishLegalUs => Some((&EN_LEGAL_AUT.0, &EN_LEGAL_AUT.1)),
            WordLexicon::FrenchLegal => Some((&FR_LEGAL_AUT.0, &FR_LEGAL_AUT.1)),
            WordLexicon::GermanLegal => Some((&DE_LEGAL_AUT.0, &DE_LEGAL_AUT.1)),
            WordLexicon::SpanishLegal => Some((&ES_LEGAL_AUT.0, &ES_LEGAL_AUT.1)),
            WordLexicon::ItalianLegal => Some((&IT_LEGAL_AUT.0, &IT_LEGAL_AUT.1)),
            WordLexicon::PortugueseLegal => Some((&PT_LEGAL_AUT.0, &PT_LEGAL_AUT.1)),
            WordLexicon::MarkdownAtx => None,
            WordLexicon::Custom(c) => Some((&c.automaton, &c.kinds)),
        }
    }

    fn is_markdown(&self) -> bool {
        matches!(self, WordLexicon::MarkdownAtx)
    }
}

// ─── Entry point ──────────────────────────────────────────────────────────

/// Parse the leading enumerator of `line` using the default English-legal-US
/// word-prefix lexicon. Caller is expected to pre-strip leading whitespace;
/// the parser anchors at byte 0.
pub fn parse_enumerator(line: &str) -> Option<Enumerator> {
    parse_enumerator_with(line, &WordLexicon::EnglishLegalUs)
}

/// Parse the leading enumerator of `line` against an explicit word-prefix
/// lexicon. Use a domain-specific built-in (`WordLexicon::FrenchLegal`,
/// `MarkdownAtx`, …) or pass a `Custom` lexicon you built yourself.
pub fn parse_enumerator_with(line: &str, lexicon: &WordLexicon) -> Option<Enumerator> {
    let bytes = line.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    // Priority order from §Q6 — first match wins. The Markdown ATX path is
    // tried first for `MarkdownAtx` lexicons because `# Header` would
    // otherwise have no word-prefix match at all.
    if lexicon.is_markdown() {
        if let Some(e) = parse_markdown_atx(line) {
            return Some(e);
        }
    }
    if let Some(e) = parse_section_sigil(line) {
        return Some(e);
    }
    if let Some(e) = parse_word_prefix(line, lexicon) {
        return Some(e);
    }
    if let Some(e) = parse_parenthetical(line) {
        return Some(e);
    }
    if let Some(e) = parse_decimal(line) {
        return Some(e);
    }
    if let Some(e) = parse_bullet(line) {
        return Some(e);
    }
    parse_bare_letter(line)
}

// ─── Markdown / typographic bullet (`- item`, `* item`, `+ item`, `• item`) ─
//
// F-R3 fix. Recognized markers: ASCII `-`, `*`, `+`, plus `•` U+2022
// BULLET. Must be followed by whitespace (single space or tab) AND have
// non-empty content after — bare `-` / `*` lines are not bullets.

fn parse_bullet(line: &str) -> Option<Enumerator> {
    let bytes = line.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    // Detect the marker byte length: ASCII single byte, U+2022 = 3 bytes.
    let (marker_len, is_bullet_char) = if bytes[0] == b'-' || bytes[0] == b'*' || bytes[0] == b'+' {
        (1usize, true)
    } else if bytes.len() >= 3 && bytes[0..3] == [0xE2, 0x80, 0xA2] {
        // U+2022 in UTF-8.
        (3usize, true)
    } else {
        (0, false)
    };
    if !is_bullet_char {
        return None;
    }
    // Require whitespace after the marker.
    if marker_len >= bytes.len() {
        return None;
    }
    let next = bytes[marker_len];
    if next != b' ' && next != b'\t' {
        return None;
    }
    // Skip the run of whitespace.
    let mut prefix_end = marker_len + 1;
    while prefix_end < bytes.len() && (bytes[prefix_end] == b' ' || bytes[prefix_end] == b'\t') {
        prefix_end += 1;
    }
    // Require non-empty content after.
    if prefix_end >= bytes.len() {
        return None;
    }
    Some(Enumerator {
        kind: EnumKind::Bullet,
        value: 0,
        depth: 1,
        raw_start: 0,
        raw_end: marker_len as u32,
        prefix_end: prefix_end as u32,
    })
}

// ─── 0. Markdown ATX (`# Header`, `## Sub`, `### Sub-sub`, …) ─────────────
//
// Markdown ATX headings are detected by counting leading `#` characters
// followed by a single space and at least one printable character. The
// `EnumKind::SectionWord` is emitted with `value = depth_packed_decimal`
// so consumers can read the heading depth from `Enumerator.value`.

fn parse_markdown_atx(line: &str) -> Option<Enumerator> {
    let bytes = line.as_bytes();
    let mut hash_count = 0;
    while hash_count < bytes.len() && hash_count < 6 && bytes[hash_count] == b'#' {
        hash_count += 1;
    }
    if hash_count == 0 {
        return None;
    }
    // Require a space after the `#` run (rejects `#FOO` non-headings).
    if bytes.get(hash_count).copied() != Some(b' ') {
        return None;
    }
    // Require at least one non-space char after the leading `# ` group.
    let space_consumed = consume_space(&line[hash_count..])?;
    let after_space_idx = hash_count + space_consumed;
    if after_space_idx >= bytes.len() {
        return None;
    }
    let depth = hash_count as u32;
    Some(Enumerator {
        kind: EnumKind::SectionWord,
        value: depth << 24,
        depth: hash_count as u8,
        raw_start: 0,
        raw_end: hash_count as u32,
        prefix_end: after_space_idx as u32,
    })
}

// ─── 1. Section sigil `§` ─────────────────────────────────────────────────

fn parse_section_sigil(line: &str) -> Option<Enumerator> {
    // U+00A7 SECTION SIGN is two bytes in UTF-8: 0xC2 0xA7.
    let bytes = line.as_bytes();
    let sigil_len = if bytes.starts_with(b"\xC2\xA7") {
        2
    } else {
        return None;
    };
    // Require ≥ 1 ASCII space (or U+00A0, but the latter is normalized to
    // ' ' by P2 if the caller ran the unicode-punct flag).
    let after_sigil = &line[sigil_len..];
    let space_consumed = consume_space(after_sigil)?;
    let after_space_idx = sigil_len + space_consumed;

    // Trailing value: prefer Decimal (with possible dot-segments), else Alpha,
    // else Roman.
    let tail = &line[after_space_idx..];
    let (value, depth, value_len) = parse_trailing_value(tail)?;
    Some(Enumerator {
        kind: EnumKind::Section,
        value,
        depth,
        raw_start: 0,
        raw_end: (after_space_idx + value_len) as u32,
        prefix_end: skip_after_value(line, after_space_idx + value_len) as u32,
    })
}

// ─── 2. Word-prefixed (`Section 5`, `Chapter 7`, `Subpart B`, …) ──────────

fn parse_word_prefix(line: &str, lexicon: &WordLexicon) -> Option<Enumerator> {
    let (automaton, kinds) = lexicon.automaton()?;
    let mat = automaton.find(line)?;
    if mat.start() != 0 {
        return None; // anchor at start of stripped line
    }
    let pattern_idx = mat.pattern().as_usize();
    let kind = kinds[pattern_idx];
    let after_word = mat.end();
    // Require at least one ASCII space after the word (Sec.5 → reject;
    // Sec. 5 → accept).
    let after_word_str = &line[after_word..];
    let space_consumed = consume_space(after_word_str)?;
    let after_space_idx = after_word + space_consumed;

    let tail = &line[after_space_idx..];
    let (value, depth, value_len) = parse_trailing_value(tail)?;
    Some(Enumerator {
        kind,
        value,
        depth,
        raw_start: 0,
        raw_end: (after_space_idx + value_len) as u32,
        prefix_end: skip_after_value(line, after_space_idx + value_len) as u32,
    })
}

// ─── 3. Parenthetical `(a)` / `(1)` / `(iv)` ──────────────────────────────

fn parse_parenthetical(line: &str) -> Option<Enumerator> {
    let bytes = line.as_bytes();
    if bytes.first() != Some(&b'(') {
        return None;
    }
    let close = bytes[1..].iter().position(|&b| b == b')')?;
    if close == 0 {
        return None; // empty `()`
    }
    let inner = &line[1..1 + close];
    if inner.starts_with(' ') || inner.ends_with(' ') {
        return None; // reject `( a )`
    }
    if inner.contains('.') {
        return None; // reject `(a.1)` — collapses 8-level nesting
    }
    let kind;
    let value;
    let depth;
    // Single-segment alpha?
    if let Some((kn, v)) = recognise_alpha(inner) {
        kind = match kn {
            EnumKind::AlphaUpper => EnumKind::ParenAlpha,
            EnumKind::AlphaLower => EnumKind::ParenAlpha,
            _ => unreachable!(),
        };
        value = v;
        depth = 1;
    } else if let Some((kn, v)) = recognise_roman(inner) {
        kind = match kn {
            EnumKind::RomanUpper | EnumKind::RomanLower => EnumKind::ParenRoman,
            _ => unreachable!(),
        };
        value = v;
        depth = 1;
    } else if let Some((v, _, len)) = parse_decimal_value(inner) {
        if len != inner.len() {
            return None;
        }
        kind = EnumKind::ParenDecimal;
        value = v;
        depth = 1;
    } else {
        return None;
    }
    let total_len = 1 + close + 1;
    Some(Enumerator {
        kind,
        value,
        depth,
        raw_start: 0,
        raw_end: total_len as u32,
        prefix_end: skip_after_value(line, total_len) as u32,
    })
}

// ─── 4. Bare decimal `1.`, `1.1`, `1.2.3` ─────────────────────────────────

fn parse_decimal(line: &str) -> Option<Enumerator> {
    let (value, depth, len) = parse_decimal_value(line)?;
    // Single-segment decimals MUST end with a trailing dot (`1.` is a list
    // marker; `1` alone is not). Multi-segment is allowed without dot.
    if depth == 1
        && !line
            .as_bytes()
            .get(len)
            .map(|&b| b == b'.')
            .unwrap_or(false)
    {
        // Already consumed the trailing dot inside parse_decimal_value? No —
        // parse_decimal_value handles dotted form but does NOT require the
        // trailing dot for single-segment. Add it here.
        return None;
    }
    let consumed = if depth == 1 { len + 1 } else { len };
    Some(Enumerator {
        kind: EnumKind::Decimal,
        value,
        depth,
        raw_start: 0,
        raw_end: consumed as u32,
        prefix_end: skip_after_value(line, consumed) as u32,
    })
}

/// Parse a possibly-dotted decimal at the start of `s`. Returns
/// `(packed_value, depth, byte_length)`. Single-segment results have
/// `depth=1`. Caller decides whether to require a trailing dot.
///
/// For single-segment numerals the full value is stored directly in
/// `value` (range 0..=u32::MAX). For multi-segment dotted decimals
/// (`1.2.3`), each segment is packed into one byte and so capped at
/// 255.
fn parse_decimal_value(s: &str) -> Option<(u32, u8, usize)> {
    let bytes = s.as_bytes();
    let mut segments: [u32; 4] = [0; 4];
    let mut depth: u8 = 0;
    let mut cursor: usize = 0;
    loop {
        // Read one segment of digits.
        let seg_start = cursor;
        let mut seg: u64 = 0;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            seg = seg * 10 + (bytes[cursor] - b'0') as u64;
            if seg > u32::MAX as u64 {
                // Truly massive numerics don't make sense for an enumerator.
                return None;
            }
            cursor += 1;
        }
        if cursor == seg_start {
            // No digits read.
            if depth == 0 {
                return None;
            }
            break;
        }
        depth = depth.saturating_add(1);
        if depth > 4 {
            return None;
        }
        segments[(depth - 1) as usize] = seg as u32;
        // Continue if next byte is `.` AND another digit follows.
        if cursor + 1 < bytes.len() && bytes[cursor] == b'.' && bytes[cursor + 1].is_ascii_digit() {
            cursor += 1; // consume the dot
            continue;
        }
        break;
    }
    if depth == 0 {
        return None;
    }
    let packed = if depth == 1 {
        // Single-segment: store the full value directly. USC sections
        // like 271, 552, 1404 routinely exceed 255 (F-R7).
        segments[0]
    } else {
        // Multi-segment: each segment must fit in one byte (per-byte
        // packing convention).
        let n = depth as usize;
        if segments.iter().take(n).any(|&s| s > 255) {
            return None;
        }
        let mut packed = 0u32;
        for (i, &seg) in segments.iter().take(n).enumerate() {
            let shift = 8 * (3 - i);
            packed |= seg << shift;
        }
        packed
    };
    Some((packed, depth, cursor))
}

// ─── 5. Bare letter `A.` / `a.` / `I.` / `i.` ─────────────────────────────

fn parse_bare_letter(line: &str) -> Option<Enumerator> {
    // Read the letter run + trailing dot.
    let bytes = line.as_bytes();
    let mut letter_len = 0;
    while letter_len < bytes.len() && bytes[letter_len].is_ascii_alphabetic() {
        letter_len += 1;
        if letter_len > 12 {
            return None; // Roman cap (mmmcmxcix is 9 chars)
        }
    }
    if letter_len == 0 {
        return None;
    }
    if bytes.get(letter_len).copied() != Some(b'.') {
        return None;
    }
    let candidate = &line[..letter_len];
    let total_len = letter_len + 1; // include the trailing dot

    // Pandoc disambiguation rule (Q6): single-letter `I` / `V` / `i` / `v`
    // are Roman; everything else is Alpha. Multi-letter is always Roman.
    let kind_value = if letter_len == 1 {
        match candidate {
            "I" | "V" => Some((EnumKind::RomanUpper, recognise_roman(candidate)?.1)),
            "i" | "v" => Some((EnumKind::RomanLower, recognise_roman(candidate)?.1)),
            _ => recognise_alpha(candidate),
        }
    } else {
        // Multi-letter: must be valid Roman (case-uniform).
        recognise_roman(candidate)
    };
    let (kind, value) = kind_value?;
    Some(Enumerator {
        kind,
        value,
        depth: 1,
        raw_start: 0,
        raw_end: total_len as u32,
        prefix_end: skip_after_value(line, total_len) as u32,
    })
}

// ─── Roman recogniser (reuses `boilerplate::roman_to_u32`) ────────────────

fn recognise_roman(s: &str) -> Option<(EnumKind, u32)> {
    if s.is_empty() {
        return None;
    }
    // Determine case uniformity. Lower-case into a stack buffer (Q1 caps at 12).
    let mut buf = [0u8; 12];
    if s.len() > buf.len() {
        return None;
    }
    let mut all_upper = true;
    let mut all_lower = true;
    for (i, &b) in s.as_bytes().iter().enumerate() {
        if b.is_ascii_uppercase() {
            all_lower = false;
            buf[i] = b.to_ascii_lowercase();
        } else if b.is_ascii_lowercase() {
            all_upper = false;
            buf[i] = b;
        } else {
            return None;
        }
    }
    if !(all_upper || all_lower) {
        return None;
    }
    let lower = std::str::from_utf8(&buf[..s.len()]).ok()?;
    let value = crate::core::segmentation::boilerplate_roman_to_u32(lower)?;
    let kind = if all_upper {
        EnumKind::RomanUpper
    } else {
        EnumKind::RomanLower
    };
    Some((kind, value))
}

// ─── Alpha recogniser ─────────────────────────────────────────────────────

fn recognise_alpha(s: &str) -> Option<(EnumKind, u32)> {
    let bytes = s.as_bytes();
    if bytes.len() != 1 {
        return None;
    }
    let b = bytes[0];
    if b.is_ascii_uppercase() {
        Some((EnumKind::AlphaUpper, (b - b'A' + 1) as u32))
    } else if b.is_ascii_lowercase() {
        Some((EnumKind::AlphaLower, (b - b'a' + 1) as u32))
    } else {
        None
    }
}

// ─── Trailing value helpers ───────────────────────────────────────────────

/// Parse the value that follows a word-prefix or `§`. Tries Decimal (with
/// optional dotted form), then Alpha, then Roman. Returns
/// `(packed_value, depth, byte_length)` of the consumed value.
fn parse_trailing_value(s: &str) -> Option<(u32, u8, usize)> {
    if let Some((v, d, len)) = parse_decimal_value(s) {
        return Some((v, d, len));
    }
    let bytes = s.as_bytes();
    let mut alpha_len = 0;
    while alpha_len < bytes.len() && bytes[alpha_len].is_ascii_alphabetic() {
        alpha_len += 1;
        if alpha_len > 12 {
            break;
        }
    }
    if alpha_len == 0 {
        return None;
    }
    let candidate = &s[..alpha_len];
    if let Some((_, v)) = recognise_roman(candidate) {
        return Some((v, 1, alpha_len));
    }
    if let Some((_, v)) = recognise_alpha(candidate) {
        return Some((v, 1, alpha_len));
    }
    None
}

/// Consume one or more ASCII spaces. Returns the byte length consumed; `None`
/// if no space is found.
fn consume_space(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }
    if i == 0 {
        None
    } else {
        Some(i)
    }
}

/// After consuming the enumerator value, skip a single trailing space (so
/// `prefix_end` points at the start of the heading text).
fn skip_after_value(line: &str, after_value: usize) -> usize {
    let bytes = line.as_bytes();
    let mut i = after_value;
    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }
    i
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn parse_kind(s: &str) -> Option<EnumKind> {
        parse_enumerator(s).map(|e| e.kind)
    }

    // ── Decimal ────────────────────────────────────────────────────────────

    #[test]
    fn decimal_single_segment() {
        // F-R7: single-segment decimals now store the value directly
        // (not packed into the high byte) so USC sections like 271 fit.
        let e = parse_enumerator("1. Introduction").unwrap();
        assert_eq!(e.kind, EnumKind::Decimal);
        assert_eq!(e.value, 1);
        assert_eq!(e.depth, 1);
        assert_eq!(e.raw_end, 2);
    }

    #[test]
    fn decimal_single_segment_large_value() {
        // F-R7: USC section 271 is real, must parse.
        let e = parse_enumerator("271. Use of information").unwrap();
        assert_eq!(e.kind, EnumKind::Decimal);
        assert_eq!(e.value, 271);
        assert_eq!(e.depth, 1);
        // Naked-enumerator-only line (no following text) also works:
        let e = parse_enumerator("552.").unwrap();
        assert_eq!(e.value, 552);
    }

    #[test]
    fn decimal_two_segment() {
        let e = parse_enumerator("1.2 Definitions").unwrap();
        assert_eq!(e.kind, EnumKind::Decimal);
        assert_eq!(e.value, (1u32 << 24) | (2u32 << 16));
        assert_eq!(e.depth, 2);
    }

    #[test]
    fn decimal_three_segment() {
        let e = parse_enumerator("1.2.3 Sub").unwrap();
        assert_eq!(e.kind, EnumKind::Decimal);
        assert_eq!(e.value, (1u32 << 24) | (2u32 << 16) | (3u32 << 8));
        assert_eq!(e.depth, 3);
    }

    #[test]
    fn decimal_four_segment() {
        let e = parse_enumerator("1.2.3.4 Deepest").unwrap();
        assert_eq!(e.kind, EnumKind::Decimal);
        assert_eq!(e.value, (1u32 << 24) | (2u32 << 16) | (3u32 << 8) | 4);
        assert_eq!(e.depth, 4);
    }

    #[test]
    fn decimal_five_segment_rejected() {
        assert!(parse_enumerator("1.2.3.4.5 Too deep").is_none());
    }

    #[test]
    fn decimal_segment_overflow_rejected() {
        assert!(parse_enumerator("258.3 Title").is_none());
    }

    #[test]
    fn decimal_single_without_trailing_dot_rejected() {
        // `1` alone is not a list marker.
        assert!(parse_enumerator("1 Item").is_none());
    }

    // ── Roman / Alpha ──────────────────────────────────────────────────────

    #[test]
    fn roman_upper_single_letter() {
        let e = parse_enumerator("I. Background").unwrap();
        assert_eq!(e.kind, EnumKind::RomanUpper);
        assert_eq!(e.value, 1);
    }

    #[test]
    fn roman_lower_single_letter() {
        let e = parse_enumerator("v. note").unwrap();
        assert_eq!(e.kind, EnumKind::RomanLower);
        assert_eq!(e.value, 5);
    }

    #[test]
    fn roman_multi_letter() {
        for (s, v, k) in [
            ("II. Discussion", 2, EnumKind::RomanUpper),
            ("XIII. Conclusion", 13, EnumKind::RomanUpper),
            ("iv. footnote", 4, EnumKind::RomanLower),
            ("xliii. note", 43, EnumKind::RomanLower),
        ] {
            let e = parse_enumerator(s).unwrap_or_else(|| panic!("failed on {s:?}"));
            assert_eq!(e.kind, k, "kind for {s:?}");
            assert_eq!(e.value, v, "value for {s:?}");
        }
    }

    #[test]
    fn alpha_single_letter() {
        let e = parse_enumerator("A. First").unwrap();
        assert_eq!(e.kind, EnumKind::AlphaUpper);
        assert_eq!(e.value, 1);

        let e = parse_enumerator("c. third").unwrap();
        assert_eq!(e.kind, EnumKind::AlphaLower);
        assert_eq!(e.value, 3);
    }

    #[test]
    fn alpha_pandoc_rule_for_i_and_v() {
        // Single-letter I / V / i / v are Roman per Pandoc's published rule;
        // single-letter C / D / L / M are Alpha (not Roman).
        assert_eq!(parse_kind("I. text"), Some(EnumKind::RomanUpper));
        assert_eq!(parse_kind("V. text"), Some(EnumKind::RomanUpper));
        assert_eq!(parse_kind("i. text"), Some(EnumKind::RomanLower));
        assert_eq!(parse_kind("v. text"), Some(EnumKind::RomanLower));
        assert_eq!(parse_kind("C. text"), Some(EnumKind::AlphaUpper));
        assert_eq!(parse_kind("M. text"), Some(EnumKind::AlphaUpper));
    }

    #[test]
    fn alpha_multiletter_rejected_unless_roman() {
        // Multi-letter Roman is OK; multi-letter non-Roman is not.
        assert_eq!(parse_kind("AB. text"), None);
        assert_eq!(parse_kind("XYZ. text"), None); // not canonical Roman
        assert_eq!(parse_kind("II. text"), Some(EnumKind::RomanUpper));
        // Note: `xx.` IS canonical Roman (= 20), so it parses as RomanLower.
        assert_eq!(parse_kind("xx. text"), Some(EnumKind::RomanLower));
    }

    #[test]
    fn bare_letter_no_period_rejected() {
        assert!(parse_enumerator("A Item").is_none());
        assert!(parse_enumerator("I Discussion").is_none());
    }

    // ── Parenthetical ──────────────────────────────────────────────────────

    #[test]
    fn parenthetical_alpha() {
        let e = parse_enumerator("(a) Definitions").unwrap();
        assert_eq!(e.kind, EnumKind::ParenAlpha);
        assert_eq!(e.value, 1);
        assert_eq!(e.raw_end, 3);
    }

    #[test]
    fn parenthetical_decimal() {
        let e = parse_enumerator("(1) item").unwrap();
        assert_eq!(e.kind, EnumKind::ParenDecimal);
        // F-R7: single-segment decimals store the value directly.
        assert_eq!(e.value, 1);
        assert_eq!(e.depth, 1);
    }

    #[test]
    fn parenthetical_roman() {
        let e = parse_enumerator("(iv) note").unwrap();
        assert_eq!(e.kind, EnumKind::ParenRoman);
        assert_eq!(e.value, 4);
    }

    #[test]
    fn parenthetical_half_open_rejected() {
        // 1 CFR § 21.11(h) uses balanced parens only.
        assert!(parse_enumerator("a) item").is_none());
        assert!(parse_enumerator("1) item").is_none());
    }

    #[test]
    fn parenthetical_inner_whitespace_rejected() {
        assert!(parse_enumerator("( a ) item").is_none());
    }

    #[test]
    fn parenthetical_inner_dot_rejected() {
        // `(a.1)` collides with USC's 8-level nesting that uses `(a)(1)`.
        assert!(parse_enumerator("(a.1) item").is_none());
    }

    #[test]
    fn parenthetical_empty_rejected() {
        assert!(parse_enumerator("() item").is_none());
    }

    // ── Word-prefixed ──────────────────────────────────────────────────────

    #[test]
    fn section_word_with_decimal_value() {
        let e = parse_enumerator("Section 5 Title").unwrap();
        assert_eq!(e.kind, EnumKind::SectionWord);
        // F-R7: single-segment decimal stored directly.
        assert_eq!(e.value, 5);
        assert_eq!(e.depth, 1);
    }

    #[test]
    fn section_abbreviation() {
        let e = parse_enumerator("Sec. 5 Title").unwrap();
        assert_eq!(e.kind, EnumKind::SectionWord);
        assert_eq!(e.value, 5);
    }

    #[test]
    fn section_dotted_decimal_value() {
        let e = parse_enumerator("Section 5.2 Definitions").unwrap();
        assert_eq!(e.kind, EnumKind::SectionWord);
        assert_eq!(e.value, (5u32 << 24) | (2u32 << 16));
        assert_eq!(e.depth, 2);
    }

    #[test]
    fn section_sigil_with_decimal() {
        let e = parse_enumerator("§ 5 Title").unwrap();
        assert_eq!(e.kind, EnumKind::Section);
        assert_eq!(e.value, 5);
    }

    #[test]
    fn section_word_case_insensitive() {
        for s in ["SECTION 5", "section 5", "Section 5"] {
            assert_eq!(parse_kind(s), Some(EnumKind::SectionWord), "{s:?}");
        }
    }

    #[test]
    fn section_word_no_space_rejected() {
        // "Sec.5" is an OCR artefact, not a drafting-style enumerator.
        assert!(parse_enumerator("Sec.5 Title").is_none());
    }

    #[test]
    fn chapter_subpart_article_word() {
        for (s, k) in [
            ("Chapter 7 Title", EnumKind::ChapterWord),
            ("Title 11 Bankruptcy", EnumKind::ChapterWord),
            ("Subpart B Title", EnumKind::SubpartWord),
            ("Subchapter II Description", EnumKind::SubpartWord),
            ("Article III Authority", EnumKind::SubpartWord),
            ("Appendix A Tables", EnumKind::ChapterWord),
            ("Schedule A Items", EnumKind::ChapterWord),
        ] {
            assert_eq!(parse_kind(s), Some(k), "{s:?}");
        }
    }

    #[test]
    fn longest_word_wins() {
        // `Subchapter` must win over `Sub` (LeftmostLongest).
        let e = parse_enumerator("Subchapter II Topic").unwrap();
        assert_eq!(e.kind, EnumKind::SubpartWord);
    }

    // ── Negatives — common single-letter abbreviations ────────────────────

    #[test]
    fn parser_returns_first_token_not_full_context() {
        // The parser is intentionally non-discriminating per Q6 of the design
        // reference: it reports the leading enumerator-shaped token and lets
        // the heading scorer filter context-based false positives.
        //
        // `e.g.` starts with `e.` which IS a valid AlphaLower marker — this
        // is the recall-first design intent, not a bug. The same holds for
        // `U.S.C.` (starts with `U.` → AlphaUpper) and `i.e.` (starts with
        // `i.` → RomanLower per Pandoc's I/V rule).
        assert_eq!(parse_kind("e.g. text"), Some(EnumKind::AlphaLower));
        assert_eq!(parse_kind("U.S.C. text"), Some(EnumKind::AlphaUpper));
        assert_eq!(parse_kind("i.e. text"), Some(EnumKind::RomanLower));
    }

    #[test]
    fn pub_l_does_not_word_prefix_match() {
        // `Pub. L. 123` starts with `Pub.` which is not in the prefix lexicon;
        // however `Pub.` after the L does not anchor at start. The parser
        // falls through to `parse_bare_letter` on `Pub.` which fails (multi-
        // letter non-Roman). Then nothing else fires.
        assert_eq!(parse_kind("Pub. L. 123"), None);
    }

    // ── prefix_end correctness ────────────────────────────────────────────

    #[test]
    fn prefix_end_skips_trailing_space() {
        let e = parse_enumerator("1. Introduction").unwrap();
        assert_eq!(e.prefix_end, 3); // points at 'I'
    }

    #[test]
    fn prefix_end_for_section_word() {
        let e = parse_enumerator("Section 5 Title").unwrap();
        assert_eq!(e.prefix_end, 10); // points at 'T'
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(parse_enumerator("").is_none());
    }

    // ── Lexicon-registry coverage (P7.0e) ─────────────────────────────────

    #[test]
    fn french_legal_lexicon_basic() {
        let lex = WordLexicon::FrenchLegal;
        for (input, expected) in [
            ("Article 5 Texte", EnumKind::SubpartWord),
            ("Chapitre 2 Titre", EnumKind::ChapterWord),
            ("Titre III Constitution", EnumKind::ChapterWord),
            ("Section 4 Détails", EnumKind::SectionWord),
            ("Annexe A Tableaux", EnumKind::ChapterWord),
        ] {
            let e = parse_enumerator_with(input, &lex)
                .unwrap_or_else(|| panic!("expected match for {input:?}"));
            assert_eq!(e.kind, expected, "{input:?}");
        }
    }

    #[test]
    fn german_legal_lexicon_basic() {
        let lex = WordLexicon::GermanLegal;
        for (input, expected) in [
            ("Artikel 5 Text", EnumKind::SubpartWord),
            ("Kapitel 2 Titel", EnumKind::ChapterWord),
            ("Abschnitt 4 Details", EnumKind::SectionWord),
            ("Anhang A Tabellen", EnumKind::ChapterWord),
            ("Buch I Allgemein", EnumKind::ChapterWord),
        ] {
            let e = parse_enumerator_with(input, &lex)
                .unwrap_or_else(|| panic!("expected match for {input:?}"));
            assert_eq!(e.kind, expected, "{input:?}");
        }
    }

    #[test]
    fn spanish_legal_lexicon_handles_diacritics() {
        let lex = WordLexicon::SpanishLegal;
        for (input, expected) in [
            ("Artículo 5 Texto", EnumKind::SubpartWord),
            ("artículo 5 texto", EnumKind::SubpartWord),
            ("ARTÍCULO 5 TEXTO", EnumKind::SubpartWord),
            ("Capítulo 2 Título", EnumKind::ChapterWord),
            ("Sección 3 Detalles", EnumKind::SectionWord),
            ("Anexo A Cuadros", EnumKind::ChapterWord),
        ] {
            let e = parse_enumerator_with(input, &lex)
                .unwrap_or_else(|| panic!("expected match for {input:?}"));
            assert_eq!(e.kind, expected, "{input:?}");
        }
    }

    #[test]
    fn italian_legal_lexicon_basic() {
        let lex = WordLexicon::ItalianLegal;
        for (input, expected) in [
            ("Articolo 5 Testo", EnumKind::SubpartWord),
            ("Capitolo 2 Titolo", EnumKind::ChapterWord),
            ("Capo II Generalità", EnumKind::ChapterWord),
            ("Sezione 4 Dettagli", EnumKind::SectionWord),
            ("Allegato A Tabelle", EnumKind::ChapterWord),
        ] {
            let e = parse_enumerator_with(input, &lex)
                .unwrap_or_else(|| panic!("expected match for {input:?}"));
            assert_eq!(e.kind, expected, "{input:?}");
        }
    }

    #[test]
    fn portuguese_legal_handles_iberian_and_brazilian() {
        let lex = WordLexicon::PortugueseLegal;
        // Iberian (Secção) + Brazilian (Seção) spellings both covered.
        for (input, expected) in [
            ("Artigo 5 Texto", EnumKind::SubpartWord),
            ("Capítulo 2 Título", EnumKind::ChapterWord),
            ("Secção 3 Detalhes", EnumKind::SectionWord),
            ("Seção 3 Detalhes", EnumKind::SectionWord),
            ("Anexo A Tabelas", EnumKind::ChapterWord),
        ] {
            let e = parse_enumerator_with(input, &lex)
                .unwrap_or_else(|| panic!("expected match for {input:?}"));
            assert_eq!(e.kind, expected, "{input:?}");
        }
    }

    #[test]
    fn lexicons_dont_cross_match() {
        // English-legal-US lexicon should NOT fire on `Capitolo 2 Titolo`
        // (Italian) — the English-only word list doesn't include `Capitolo`.
        let it_input = "Capitolo 2 Titolo";
        assert!(parse_enumerator_with(it_input, &WordLexicon::EnglishLegalUs).is_none());
        // Italian lexicon SHOULD fire.
        assert!(parse_enumerator_with(it_input, &WordLexicon::ItalianLegal).is_some());
    }

    // ── Markdown ATX (P7.0e) ──────────────────────────────────────────────

    #[test]
    fn markdown_atx_h1_through_h6() {
        let lex = WordLexicon::MarkdownAtx;
        for (input, depth) in [
            ("# Title", 1u8),
            ("## Subtitle", 2),
            ("### Sub-sub", 3),
            ("#### Level 4", 4),
            ("##### Level 5", 5),
            ("###### Level 6", 6),
        ] {
            let e = parse_enumerator_with(input, &lex)
                .unwrap_or_else(|| panic!("expected match for {input:?}"));
            assert_eq!(e.kind, EnumKind::SectionWord, "kind for {input:?}");
            assert_eq!(e.depth, depth, "depth for {input:?}");
            assert_eq!(e.value, (depth as u32) << 24, "value for {input:?}");
            // prefix_end must skip the leading "# " group.
            let expected_skip = depth as u32 + 1;
            assert_eq!(e.prefix_end, expected_skip, "prefix_end for {input:?}");
        }
    }

    #[test]
    fn markdown_atx_seven_hashes_is_not_atx() {
        // Markdown spec: only #..###### are valid; 7+ # is not a heading.
        let lex = WordLexicon::MarkdownAtx;
        let e = parse_enumerator_with("####### too many", &lex);
        // Either None or the parser should return at most depth 6.
        if let Some(e) = e {
            assert!(
                e.depth <= 6,
                "got depth {} on too-many-hashes input",
                e.depth
            );
        }
    }

    #[test]
    fn markdown_atx_no_space_after_hash_rejected() {
        let lex = WordLexicon::MarkdownAtx;
        assert!(parse_enumerator_with("#NoSpace", &lex).is_none());
    }

    #[test]
    fn markdown_atx_falls_back_to_other_kinds() {
        let lex = WordLexicon::MarkdownAtx;
        // A Markdown lexicon should still parse decimal/parenthetical/etc.
        // when there's no `#` prefix — the lexicon governs only the word-
        // prefix path.
        let e = parse_enumerator_with("1. Heading", &lex).unwrap();
        assert_eq!(e.kind, EnumKind::Decimal);
    }

    // ── Custom lexicon (P7.0e) ────────────────────────────────────────────

    #[test]
    fn custom_lexicon_round_trip() {
        let custom = CustomLexicon::new(vec![
            ("Step".to_string(), EnumKind::SectionWord),
            ("Phase".to_string(), EnumKind::ChapterWord),
        ])
        .unwrap();
        let lex = WordLexicon::Custom(Arc::new(custom));
        let e = parse_enumerator_with("Step 3 Boil water", &lex).unwrap();
        assert_eq!(e.kind, EnumKind::SectionWord);
        let e = parse_enumerator_with("Phase 2 Cleanup", &lex).unwrap();
        assert_eq!(e.kind, EnumKind::ChapterWord);
    }

    #[test]
    fn custom_lexicon_empty_rejected() {
        assert!(CustomLexicon::new(vec![]).is_err());
    }

    // ── Property tests ─────────────────────────────────────────────────────

    proptest! {
        #[test]
        fn parse_never_panics(text in "\\PC{0,128}") {
            let _ = parse_enumerator(&text);
        }

        #[test]
        fn parsed_offsets_are_in_bounds(text in "\\PC{0,128}") {
            if let Some(e) = parse_enumerator(&text) {
                prop_assert!(e.raw_end as usize <= text.len());
                prop_assert!(e.prefix_end as usize <= text.len());
                prop_assert!(e.raw_start <= e.raw_end);
                prop_assert!(e.raw_end <= e.prefix_end);
            }
        }

        /// The packed-byte encoding for Decimal must round-trip — depth and
        /// segment count agree.
        ///
        /// F-R7: single-segment values are stored directly (`value =
        /// seg1`), multi-segment values are byte-packed
        /// (`value = (seg1<<24) | (seg2<<16) | …`).
        #[test]
        fn decimal_packed_round_trip(seg1 in 1u32..=255, seg2 in 0u32..=255) {
            let s = if seg2 == 0 {
                format!("{}.", seg1)
            } else {
                format!("{}.{}", seg1, seg2)
            };
            let e = parse_enumerator(&s).unwrap();
            prop_assert_eq!(e.kind, EnumKind::Decimal);
            let expected_depth = if seg2 == 0 { 1 } else { 2 };
            prop_assert_eq!(e.depth, expected_depth);
            if seg2 == 0 {
                // Single-segment: value is the raw integer.
                prop_assert_eq!(e.value, seg1);
            } else {
                // Multi-segment: byte-packed.
                let s1 = (e.value >> 24) & 0xFF;
                let s2 = (e.value >> 16) & 0xFF;
                prop_assert_eq!(s1, seg1);
                prop_assert_eq!(s2, seg2);
            }
        }
    }

    // ─── F-R3: Markdown / typographic bullets ─────────────────────────────

    #[test]
    fn bullet_dash_with_content() {
        let e = parse_enumerator("- a list item").unwrap();
        assert_eq!(e.kind, EnumKind::Bullet);
        assert_eq!(e.value, 0);
        assert_eq!(e.depth, 1);
        assert_eq!(e.raw_start, 0);
        assert_eq!(e.raw_end, 1);
        assert_eq!(e.prefix_end, 2); // past the "- "
    }

    #[test]
    fn bullet_asterisk_plus_unicode() {
        assert_eq!(parse_enumerator("* item").unwrap().kind, EnumKind::Bullet);
        assert_eq!(parse_enumerator("+ item").unwrap().kind, EnumKind::Bullet);
        assert_eq!(parse_enumerator("• item").unwrap().kind, EnumKind::Bullet);
    }

    #[test]
    fn bullet_requires_whitespace_after_marker() {
        // "*emphasis*" must NOT match — no whitespace after the marker.
        assert!(parse_enumerator("*emphasis*").is_none());
        // "-1.2.3" must NOT match — no whitespace.
        assert!(parse_enumerator("-1.2.3").is_none());
        // "+5" must NOT match — sign for a number.
        assert!(parse_enumerator("+5").is_none());
    }

    #[test]
    fn bullet_requires_non_empty_content() {
        // Bare "- " (or with trailing whitespace only) must NOT match.
        assert!(parse_enumerator("- ").is_none());
        assert!(parse_enumerator("- \t  ").is_none());
        assert!(parse_enumerator("*").is_none());
        assert!(parse_enumerator("•").is_none());
    }

    #[test]
    fn bullet_horizontal_rule_does_not_match() {
        // Markdown horizontal rules use 3+ dashes/asterisks. Don't match.
        assert!(parse_enumerator("---").is_none());
        assert!(parse_enumerator("***").is_none());
    }

    #[test]
    fn bullet_tab_after_marker_works() {
        let e = parse_enumerator("-\titem").unwrap();
        assert_eq!(e.kind, EnumKind::Bullet);
    }

    #[test]
    fn bullet_does_not_displace_other_enumerators() {
        // Bullet parsing comes after parens/decimal in the priority order
        // so "(a)" still parses as ParenAlpha, "1.2" still as Decimal.
        assert_eq!(
            parse_enumerator("(a) text").unwrap().kind,
            EnumKind::ParenAlpha
        );
        assert_eq!(
            parse_enumerator("1.2 text").unwrap().kind,
            EnumKind::Decimal
        );
        assert_eq!(
            parse_enumerator("I. heading").unwrap().kind,
            EnumKind::RomanUpper
        );
    }
}
