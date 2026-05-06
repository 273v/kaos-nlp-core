//! Heading lexicon registry and hierarchy-keyword registry for the
//! structural-analysis layer (P7).
//!
//! Two parallel registries:
//!
//! 1. **`HeadingLexicon`** — canonical-heading short list (e.g. `Background`,
//!    `Discussion`, `Order`, `Conclusion`, `Abstract`, `Introduction`,
//!    `Methods`, `Installation`, `Usage`, `License`). Used by the scorer to
//!    set the `lexical_heading` flag.
//!
//! 2. **`HierarchyLexicon`** — hierarchy-level keyword list (`Title`,
//!    `Chapter`, `Section`, `Article`, `Annexe`, …). Used by the scorer to
//!    set the `hierarchy_keyword` flag and by the hierarchy inferencer to
//!    map keywords → hierarchy depth.
//!
//! Both registries share the *generality contract* from
//! `INTEGRATION_BOUNDARIES.md`: every shipped lexicon is opt-in, every
//! lexicon must accept a caller-supplied override, and graceful degradation
//! (no keyword match, no language hint) must produce sensible output.
//!
//! Scope (per `SECTION_HEADING_PRIMITIVES_RESEARCH.md` §"corrigendum"):
//! Western languages only — English, German, French, Spanish, Italian,
//! Portuguese. Latin script. ASCII digits.

use std::sync::{Arc, LazyLock};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

// ─── Heading lexicon ───────────────────────────────────────────────────────

/// Built-in heading-lexicon variants. Each shipped variant is curated for a
/// document family; users can supply their own via [`HeadingLexicon::Custom`].
#[derive(Debug, Clone, Default)]
pub enum HeadingLexicon {
    /// Anglo-American court-opinion headings: `Background`, `Discussion`,
    /// `Conclusion`, `Order`, `Procedural History`, etc.
    #[default]
    EnglishLegalUs,
    /// IMRAD academic-paper section heads: `Abstract`, `Introduction`,
    /// `Methods`, `Results`, `Discussion`, `Conclusion`, `References`,
    /// `Acknowledgements`, plus common variants (`Materials and Methods`).
    EnglishAcademic,
    /// Software-doc / README headings: `Installation`, `Usage`,
    /// `Configuration`, `API`, `Examples`, `License`, `Contributing`,
    /// `Changelog`, `Roadmap`.
    EnglishSoftware,
    /// French legal/admin: `Préambule`, `Considérants`, `Dispositif`,
    /// `Objet`, `Définitions`.
    FrenchLegal,
    /// German legal/admin: `Präambel`, `Erwägungsgründe`, `Anhang`,
    /// `Begründung`, `Geltungsbereich`.
    GermanLegal,
    /// Spanish legal/admin: `Preámbulo`, `Considerandos`, `Disposiciones`,
    /// `Objeto`, `Definiciones`.
    SpanishLegal,
    /// Italian legal/admin: `Preambolo`, `Considerando`, `Disposizioni`,
    /// `Oggetto`, `Definizioni`.
    ItalianLegal,
    /// Portuguese legal/admin: `Preâmbulo`, `Considerandos`, `Disposições`,
    /// `Objeto`, `Definições`.
    PortugueseLegal,
    /// Caller-supplied lexicon. Patterns are matched
    /// ASCII-case-insensitively against the *normalized* (whitespace-
    /// collapsed) line text.
    Custom(Arc<CustomHeadingLexicon>),
    /// No lexicon — `lexical_heading` is always `false`. Use when the
    /// caller wants to rely solely on layout/case features.
    None,
}

/// Caller-supplied heading-lexicon. ASCII-case-insensitive whole-line match.
#[derive(Debug)]
pub struct CustomHeadingLexicon {
    automaton: AhoCorasick,
}

impl CustomHeadingLexicon {
    /// Build a custom heading lexicon from canonical labels (e.g.
    /// `["Background", "Discussion", "Conclusion"]`). Patterns must be
    /// non-empty; whitespace is significant for multi-word labels.
    pub fn new(entries: Vec<String>) -> Result<Self, String> {
        if entries.is_empty() {
            return Err("CustomHeadingLexicon requires at least one entry".into());
        }
        let automaton = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(&entries)
            .map_err(|e| e.to_string())?;
        Ok(Self { automaton })
    }

    /// Whole-line match: returns `true` iff the entire `text` (already
    /// trimmed and case-folded by the caller) is one of the lexicon
    /// entries.
    pub fn matches_whole(&self, text: &str) -> bool {
        let mut iter = self.automaton.find_iter(text);
        if let Some(m) = iter.next() {
            m.start() == 0 && m.end() == text.len()
        } else {
            false
        }
    }
}

// ── Built-in heading lists ────────────────────────────────────────────────

const ENGLISH_LEGAL_US_HEADINGS: &[&str] = &[
    "background",
    "procedural history",
    "facts",
    "findings of fact",
    "conclusions of law",
    "standard of review",
    "discussion",
    "analysis",
    "order",
    "conclusion",
    "appendix",
    "introduction",
    "summary",
    "argument",
    "statement",
    "issue",
    "issues",
    "questions presented",
    "decision",
    "opinion",
    "syllabus",
    "holdings",
    "disposition",
    "judgment",
    "memorandum",
];

const ENGLISH_ACADEMIC_HEADINGS: &[&str] = &[
    "abstract",
    "introduction",
    "background",
    "related work",
    "methods",
    "method",
    "methodology",
    "materials and methods",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "limitations",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "supplementary material",
    "supplement",
];

const ENGLISH_SOFTWARE_HEADINGS: &[&str] = &[
    "installation",
    "install",
    "getting started",
    "quickstart",
    "quick start",
    "usage",
    "configuration",
    "config",
    "api",
    "api reference",
    "examples",
    "example",
    "license",
    "licence",
    "contributing",
    "changelog",
    "release notes",
    "roadmap",
    "faq",
    "troubleshooting",
    "requirements",
    "dependencies",
    "build",
    "testing",
    "tests",
    "documentation",
    "overview",
];

const FRENCH_LEGAL_HEADINGS: &[&str] = &[
    "préambule",
    "considérants",
    "dispositif",
    "objet",
    "définitions",
    "champ d'application",
    "introduction",
    "annexe",
    "annexes",
    "résumé",
    "conclusion",
    "conclusions",
];

const GERMAN_LEGAL_HEADINGS: &[&str] = &[
    "präambel",
    "erwägungsgründe",
    "anhang",
    "begründung",
    "geltungsbereich",
    "definitionen",
    "einleitung",
    "zusammenfassung",
    "schluss",
    "fazit",
    "ergebnis",
];

const SPANISH_LEGAL_HEADINGS: &[&str] = &[
    "preámbulo",
    "considerandos",
    "disposiciones",
    "objeto",
    "definiciones",
    "ámbito de aplicación",
    "introducción",
    "anexo",
    "anexos",
    "resumen",
    "conclusión",
    "conclusiones",
];

const ITALIAN_LEGAL_HEADINGS: &[&str] = &[
    "preambolo",
    "considerando",
    "disposizioni",
    "oggetto",
    "definizioni",
    "campo di applicazione",
    "introduzione",
    "allegato",
    "allegati",
    "riassunto",
    "conclusione",
    "conclusioni",
];

const PORTUGUESE_LEGAL_HEADINGS: &[&str] = &[
    "preâmbulo",
    "considerandos",
    "disposições",
    "objeto",
    "definições",
    "âmbito de aplicação",
    "introdução",
    "anexo",
    "anexos",
    "resumo",
    "conclusão",
    "conclusões",
];

// ── Per-lexicon AC automatons (built once, lock-free re-read) ────────────

fn build_heading_ac(patterns: &[&str]) -> AhoCorasick {
    AhoCorasickBuilder::new()
        .ascii_case_insensitive(true)
        .match_kind(MatchKind::LeftmostLongest)
        .build(patterns)
        .expect("built-in heading lexicon should always build")
}

static AC_ENGLISH_LEGAL_US: LazyLock<AhoCorasick> =
    LazyLock::new(|| build_heading_ac(ENGLISH_LEGAL_US_HEADINGS));
static AC_ENGLISH_ACADEMIC: LazyLock<AhoCorasick> =
    LazyLock::new(|| build_heading_ac(ENGLISH_ACADEMIC_HEADINGS));
static AC_ENGLISH_SOFTWARE: LazyLock<AhoCorasick> =
    LazyLock::new(|| build_heading_ac(ENGLISH_SOFTWARE_HEADINGS));
static AC_FRENCH_LEGAL: LazyLock<AhoCorasick> =
    LazyLock::new(|| build_heading_ac(FRENCH_LEGAL_HEADINGS));
static AC_GERMAN_LEGAL: LazyLock<AhoCorasick> =
    LazyLock::new(|| build_heading_ac(GERMAN_LEGAL_HEADINGS));
static AC_SPANISH_LEGAL: LazyLock<AhoCorasick> =
    LazyLock::new(|| build_heading_ac(SPANISH_LEGAL_HEADINGS));
static AC_ITALIAN_LEGAL: LazyLock<AhoCorasick> =
    LazyLock::new(|| build_heading_ac(ITALIAN_LEGAL_HEADINGS));
static AC_PORTUGUESE_LEGAL: LazyLock<AhoCorasick> =
    LazyLock::new(|| build_heading_ac(PORTUGUESE_LEGAL_HEADINGS));

impl HeadingLexicon {
    /// Whole-line match against the lexicon. `text` must already be
    /// case-folded to lowercase and whitespace-collapsed by the caller —
    /// see [`crate::core::segmentation::normalize`] for the canonical path.
    ///
    /// Returns `true` iff `text` exactly matches a lexicon entry. Returns
    /// `false` for the [`HeadingLexicon::None`] variant unconditionally.
    pub fn matches_whole(&self, text: &str) -> bool {
        if text.is_empty() {
            return false;
        }
        match self {
            Self::None => false,
            Self::EnglishLegalUs => ac_matches_whole(&AC_ENGLISH_LEGAL_US, text),
            Self::EnglishAcademic => ac_matches_whole(&AC_ENGLISH_ACADEMIC, text),
            Self::EnglishSoftware => ac_matches_whole(&AC_ENGLISH_SOFTWARE, text),
            Self::FrenchLegal => ac_matches_whole(&AC_FRENCH_LEGAL, text),
            Self::GermanLegal => ac_matches_whole(&AC_GERMAN_LEGAL, text),
            Self::SpanishLegal => ac_matches_whole(&AC_SPANISH_LEGAL, text),
            Self::ItalianLegal => ac_matches_whole(&AC_ITALIAN_LEGAL, text),
            Self::PortugueseLegal => ac_matches_whole(&AC_PORTUGUESE_LEGAL, text),
            Self::Custom(custom) => custom.matches_whole(text),
        }
    }

    /// Stable string identifier — useful for body-attribute reporting.
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::EnglishLegalUs => "english_legal_us",
            Self::EnglishAcademic => "english_academic",
            Self::EnglishSoftware => "english_software",
            Self::FrenchLegal => "french_legal",
            Self::GermanLegal => "german_legal",
            Self::SpanishLegal => "spanish_legal",
            Self::ItalianLegal => "italian_legal",
            Self::PortugueseLegal => "portuguese_legal",
            Self::Custom(_) => "custom",
        }
    }
}

fn ac_matches_whole(ac: &AhoCorasick, text: &str) -> bool {
    let mut iter = ac.find_iter(text);
    if let Some(m) = iter.next() {
        m.start() == 0 && m.end() == text.len()
    } else {
        false
    }
}

// ─── Hierarchy lexicon ─────────────────────────────────────────────────────

/// Built-in hierarchy-keyword lexicons. Each entry is a `(keyword, depth)`
/// pair — depth 1 is the shallowest level (Title / Titre / Titel).
///
/// Per Q6 of the design reference: a heading line carrying a hierarchy
/// keyword (`Section 5`, `Chapter 7`, `Article 3`) takes its depth from
/// the keyword, not from the trailing enumerator.
#[derive(Debug, Clone, Default)]
pub enum HierarchyLexicon {
    /// USC-style: `Title > Subtitle > Chapter > Subchapter > Part > Subpart >
    /// Section > Subsection > Paragraph > Subparagraph > Clause > Subclause >
    /// Item`. 13 levels.
    #[default]
    EnglishLegalUs,
    /// French legal: `Livre > Titre > Chapitre > Section > Sous-section >
    /// Article > Paragraphe`. 7 levels.
    FrenchLegal,
    /// German legal: `Buch > Teil > Titel > Abschnitt > Kapitel > Artikel >
    /// Paragraph`. 7 levels.
    GermanLegal,
    /// Spanish legal: `Libro > Título > Capítulo > Sección > Artículo >
    /// Párrafo`. 6 levels.
    SpanishLegal,
    /// Italian legal: `Libro > Titolo > Capo > Sezione > Articolo >
    /// Paragrafo`. 6 levels.
    ItalianLegal,
    /// Portuguese legal: `Livro > Título > Capítulo > Secção > Artigo >
    /// Parágrafo`. 6 levels.
    PortugueseLegal,
    /// Markdown ATX: depth comes from the `#` count, not from a keyword. The
    /// scorer must handle this specially — `matches` returns `None` for
    /// every keyword, and the caller is expected to read depth from the
    /// `LineRecord`'s leading `#` run.
    MarkdownAtx,
    /// Caller-supplied lexicon.
    Custom(Arc<CustomHierarchyLexicon>),
    /// No lexicon — every keyword lookup returns `None`.
    None,
}

#[derive(Debug)]
pub struct CustomHierarchyLexicon {
    automaton: AhoCorasick,
    depths: Vec<u8>,
}

impl CustomHierarchyLexicon {
    /// Build a custom hierarchy lexicon from `(keyword, depth)` pairs.
    /// Depths must be 1..=255; depth `0` is reserved for "no keyword".
    pub fn new(entries: Vec<(String, u8)>) -> Result<Self, String> {
        if entries.is_empty() {
            return Err("CustomHierarchyLexicon requires at least one entry".into());
        }
        if entries.iter().any(|(_, d)| *d == 0) {
            return Err("CustomHierarchyLexicon depth must be >= 1".into());
        }
        let patterns: Vec<&str> = entries.iter().map(|(p, _)| p.as_str()).collect();
        let depths: Vec<u8> = entries.iter().map(|(_, d)| *d).collect();
        let automaton = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .map_err(|e| e.to_string())?;
        Ok(Self { automaton, depths })
    }

    /// Match the *leading word* of `text`; returns the depth of the matched
    /// keyword or `None`. The match is anchored at offset 0 — only a leading
    /// keyword can fire, so `Body Section 5` does NOT count as Section.
    pub fn matches_leading(&self, text: &str) -> Option<u8> {
        let m = self.automaton.find(text)?;
        if m.start() != 0 {
            return None;
        }
        // Require a word boundary after the keyword (whitespace or end).
        let after = &text[m.end()..];
        let next_is_boundary = after
            .chars()
            .next()
            .map(|c| c.is_whitespace() || c == '.' || c == ',' || c == ':' || c == ';')
            .unwrap_or(true);
        if !next_is_boundary {
            return None;
        }
        Some(self.depths[m.pattern().as_usize()])
    }
}

// ── Built-in hierarchy entries ────────────────────────────────────────────

/// (keyword, depth). Depth 1 is the shallowest level.
type HierEntry = (&'static str, u8);

const ENGLISH_LEGAL_US_HIER: &[HierEntry] = &[
    ("Title", 1),
    ("Subtitle", 2),
    ("Chapter", 3),
    ("Subchapter", 4),
    ("Part", 5),
    ("Subpart", 6),
    ("Section", 7),
    ("Sec.", 7),
    ("§", 7),
    ("Subsection", 8),
    ("Article", 7),
    ("Paragraph", 9),
    ("Subparagraph", 10),
    ("Clause", 11),
    ("Subclause", 12),
    ("Item", 13),
    ("Appendix", 1),
    ("Schedule", 1),
];

const FRENCH_LEGAL_HIER: &[HierEntry] = &[
    ("Livre", 1),
    ("livre", 1),
    ("LIVRE", 1),
    ("Titre", 2),
    ("titre", 2),
    ("TITRE", 2),
    ("Chapitre", 3),
    ("chapitre", 3),
    ("CHAPITRE", 3),
    ("Section", 4),
    ("section", 4),
    ("SECTION", 4),
    ("Sous-section", 5),
    ("sous-section", 5),
    ("Article", 6),
    ("article", 6),
    ("ARTICLE", 6),
    ("Paragraphe", 7),
    ("paragraphe", 7),
    ("Annexe", 1),
    ("annexe", 1),
    ("ANNEXE", 1),
];

const GERMAN_LEGAL_HIER: &[HierEntry] = &[
    ("Buch", 1),
    ("Teil", 2),
    ("Titel", 3),
    ("Abschnitt", 4),
    ("Kapitel", 5),
    ("Artikel", 6),
    ("Paragraph", 7),
    ("§", 7),
    ("Anhang", 1),
];

const SPANISH_LEGAL_HIER: &[HierEntry] = &[
    ("Libro", 1),
    ("libro", 1),
    ("LIBRO", 1),
    ("Título", 2),
    ("título", 2),
    ("TÍTULO", 2),
    ("Capítulo", 3),
    ("capítulo", 3),
    ("CAPÍTULO", 3),
    ("Sección", 4),
    ("sección", 4),
    ("SECCIÓN", 4),
    ("Artículo", 5),
    ("artículo", 5),
    ("ARTÍCULO", 5),
    ("Párrafo", 6),
    ("párrafo", 6),
    ("Anexo", 1),
    ("anexo", 1),
    ("ANEXO", 1),
];

const ITALIAN_LEGAL_HIER: &[HierEntry] = &[
    ("Libro", 1),
    ("libro", 1),
    ("LIBRO", 1),
    ("Titolo", 2),
    ("titolo", 2),
    ("TITOLO", 2),
    ("Capo", 3),
    ("capo", 3),
    ("CAPO", 3),
    ("Sezione", 4),
    ("sezione", 4),
    ("SEZIONE", 4),
    ("Articolo", 5),
    ("articolo", 5),
    ("ARTICOLO", 5),
    ("Paragrafo", 6),
    ("paragrafo", 6),
    ("Allegato", 1),
    ("allegato", 1),
    ("ALLEGATO", 1),
];

const PORTUGUESE_LEGAL_HIER: &[HierEntry] = &[
    ("Livro", 1),
    ("livro", 1),
    ("LIVRO", 1),
    ("Título", 2),
    ("título", 2),
    ("TÍTULO", 2),
    ("Capítulo", 3),
    ("capítulo", 3),
    ("CAPÍTULO", 3),
    ("Secção", 4),
    ("secção", 4),
    ("SECÇÃO", 4),
    ("Seção", 4),
    ("seção", 4),
    ("SEÇÃO", 4),
    ("Artigo", 5),
    ("artigo", 5),
    ("ARTIGO", 5),
    ("Parágrafo", 6),
    ("parágrafo", 6),
    ("Anexo", 1),
    ("anexo", 1),
    ("ANEXO", 1),
];

fn build_hierarchy_ac(entries: &[HierEntry]) -> (AhoCorasick, Vec<u8>) {
    let patterns: Vec<&str> = entries.iter().map(|(p, _)| *p).collect();
    let depths: Vec<u8> = entries.iter().map(|(_, d)| *d).collect();
    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(true)
        .match_kind(MatchKind::LeftmostLongest)
        .build(&patterns)
        .expect("built-in hierarchy lexicon should always build");
    (ac, depths)
}

static HIER_ENGLISH_LEGAL_US: LazyLock<(AhoCorasick, Vec<u8>)> =
    LazyLock::new(|| build_hierarchy_ac(ENGLISH_LEGAL_US_HIER));
static HIER_FRENCH_LEGAL: LazyLock<(AhoCorasick, Vec<u8>)> =
    LazyLock::new(|| build_hierarchy_ac(FRENCH_LEGAL_HIER));
static HIER_GERMAN_LEGAL: LazyLock<(AhoCorasick, Vec<u8>)> =
    LazyLock::new(|| build_hierarchy_ac(GERMAN_LEGAL_HIER));
static HIER_SPANISH_LEGAL: LazyLock<(AhoCorasick, Vec<u8>)> =
    LazyLock::new(|| build_hierarchy_ac(SPANISH_LEGAL_HIER));
static HIER_ITALIAN_LEGAL: LazyLock<(AhoCorasick, Vec<u8>)> =
    LazyLock::new(|| build_hierarchy_ac(ITALIAN_LEGAL_HIER));
static HIER_PORTUGUESE_LEGAL: LazyLock<(AhoCorasick, Vec<u8>)> =
    LazyLock::new(|| build_hierarchy_ac(PORTUGUESE_LEGAL_HIER));

impl HierarchyLexicon {
    /// Match the *leading word* of `text`; returns the depth of the matched
    /// keyword or `None`. Anchored at offset 0 — only a leading keyword
    /// fires.
    ///
    /// Note: Markdown ATX is keyword-less (depth comes from `#` count). The
    /// scorer must read depth directly from the line record for that
    /// variant.
    pub fn matches_leading(&self, text: &str) -> Option<u8> {
        if text.is_empty() {
            return None;
        }
        match self {
            Self::None | Self::MarkdownAtx => None,
            Self::EnglishLegalUs => match_leading_static(&HIER_ENGLISH_LEGAL_US, text),
            Self::FrenchLegal => match_leading_static(&HIER_FRENCH_LEGAL, text),
            Self::GermanLegal => match_leading_static(&HIER_GERMAN_LEGAL, text),
            Self::SpanishLegal => match_leading_static(&HIER_SPANISH_LEGAL, text),
            Self::ItalianLegal => match_leading_static(&HIER_ITALIAN_LEGAL, text),
            Self::PortugueseLegal => match_leading_static(&HIER_PORTUGUESE_LEGAL, text),
            Self::Custom(custom) => custom.matches_leading(text),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::MarkdownAtx => "markdown_atx",
            Self::EnglishLegalUs => "english_legal_us",
            Self::FrenchLegal => "french_legal",
            Self::GermanLegal => "german_legal",
            Self::SpanishLegal => "spanish_legal",
            Self::ItalianLegal => "italian_legal",
            Self::PortugueseLegal => "portuguese_legal",
            Self::Custom(_) => "custom",
        }
    }
}

fn match_leading_static(slot: &(AhoCorasick, Vec<u8>), text: &str) -> Option<u8> {
    let (ac, depths) = slot;
    let m = ac.find(text)?;
    if m.start() != 0 {
        return None;
    }
    let after = &text[m.end()..];
    let next_is_boundary = after
        .chars()
        .next()
        .map(|c| c.is_whitespace() || c == '.' || c == ',' || c == ':' || c == ';')
        .unwrap_or(true);
    if !next_is_boundary {
        return None;
    }
    Some(depths[m.pattern().as_usize()])
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HeadingLexicon ──────────────────────────────────────────────────────

    #[test]
    fn english_legal_us_matches_canonical_headings() {
        let lex = HeadingLexicon::EnglishLegalUs;
        assert!(lex.matches_whole("background"));
        assert!(lex.matches_whole("BACKGROUND")); // case-insensitive
        assert!(lex.matches_whole("procedural history"));
        assert!(lex.matches_whole("conclusions of law"));
    }

    #[test]
    fn english_academic_imrad() {
        let lex = HeadingLexicon::EnglishAcademic;
        assert!(lex.matches_whole("abstract"));
        assert!(lex.matches_whole("introduction"));
        assert!(lex.matches_whole("methods"));
        assert!(lex.matches_whole("materials and methods"));
        assert!(lex.matches_whole("results"));
        assert!(lex.matches_whole("references"));
    }

    #[test]
    fn english_software_readme() {
        let lex = HeadingLexicon::EnglishSoftware;
        assert!(lex.matches_whole("installation"));
        assert!(lex.matches_whole("usage"));
        assert!(lex.matches_whole("license"));
        assert!(lex.matches_whole("api reference"));
    }

    #[test]
    fn western_legal_lexicons_match_diacritics() {
        assert!(HeadingLexicon::FrenchLegal.matches_whole("préambule"));
        assert!(HeadingLexicon::SpanishLegal.matches_whole("preámbulo"));
        assert!(HeadingLexicon::PortugueseLegal.matches_whole("preâmbulo"));
        assert!(HeadingLexicon::ItalianLegal.matches_whole("preambolo"));
        assert!(HeadingLexicon::GermanLegal.matches_whole("präambel"));
    }

    #[test]
    fn no_lexicon_never_matches() {
        let lex = HeadingLexicon::None;
        assert!(!lex.matches_whole("background"));
        assert!(!lex.matches_whole(""));
    }

    #[test]
    fn whole_match_required() {
        // "background of the case" must NOT match — we do whole-line match
        let lex = HeadingLexicon::EnglishLegalUs;
        assert!(!lex.matches_whole("background of the case"));
        assert!(!lex.matches_whole("the discussion"));
    }

    #[test]
    fn custom_heading_lexicon() {
        let lex = HeadingLexicon::Custom(Arc::new(
            CustomHeadingLexicon::new(vec!["my heading".into(), "another".into()]).unwrap(),
        ));
        assert!(lex.matches_whole("my heading"));
        assert!(lex.matches_whole("MY HEADING"));
        assert!(lex.matches_whole("another"));
        assert!(!lex.matches_whole("not in lexicon"));
    }

    #[test]
    fn custom_heading_lexicon_rejects_empty() {
        assert!(CustomHeadingLexicon::new(Vec::new()).is_err());
    }

    // ── HierarchyLexicon ───────────────────────────────────────────────────

    #[test]
    fn english_legal_hierarchy_depth_order() {
        let h = HierarchyLexicon::EnglishLegalUs;
        // Title shallower than Section, Section shallower than Subsection
        let title_depth = h.matches_leading("Title 26 Internal Revenue").unwrap();
        let section_depth = h.matches_leading("Section 5 Definitions").unwrap();
        let subsection_depth = h.matches_leading("Subsection 5.1 Limits").unwrap();
        assert!(title_depth < section_depth);
        assert!(section_depth < subsection_depth);
    }

    #[test]
    fn hierarchy_section_sigil() {
        let h = HierarchyLexicon::EnglishLegalUs;
        assert_eq!(h.matches_leading("§ 552"), Some(7));
        assert_eq!(h.matches_leading("Sec. 5"), Some(7));
    }

    #[test]
    fn hierarchy_must_be_leading() {
        let h = HierarchyLexicon::EnglishLegalUs;
        // Keyword in middle of line does not match
        assert_eq!(h.matches_leading("This Section is about X"), None);
    }

    #[test]
    fn hierarchy_word_boundary() {
        let h = HierarchyLexicon::EnglishLegalUs;
        // "Sectional" should not match Section
        assert_eq!(h.matches_leading("Sectional view"), None);
        // But "Section." (with period) should
        assert_eq!(h.matches_leading("Section. 5"), Some(7));
    }

    #[test]
    fn hierarchy_french_legal() {
        let h = HierarchyLexicon::FrenchLegal;
        let livre = h.matches_leading("Livre Premier — Des personnes").unwrap();
        let chapitre = h.matches_leading("Chapitre III — Du mariage").unwrap();
        let article = h.matches_leading("Article 5").unwrap();
        assert!(livre < chapitre);
        assert!(chapitre < article);
    }

    #[test]
    fn hierarchy_german_legal() {
        let h = HierarchyLexicon::GermanLegal;
        let buch = h.matches_leading("Buch 1").unwrap();
        let kapitel = h.matches_leading("Kapitel 5").unwrap();
        let artikel = h.matches_leading("Artikel 12").unwrap();
        assert!(buch < kapitel);
        assert!(kapitel < artikel);
    }

    #[test]
    fn hierarchy_markdown_atx_returns_none() {
        // Markdown depth comes from `#` count — keyword path returns None.
        let h = HierarchyLexicon::MarkdownAtx;
        assert_eq!(h.matches_leading("# Heading"), None);
        assert_eq!(h.matches_leading("Section 5"), None);
    }

    #[test]
    fn hierarchy_no_lexicon() {
        let h = HierarchyLexicon::None;
        assert_eq!(h.matches_leading("Section 5"), None);
        assert_eq!(h.matches_leading("Article 1"), None);
    }

    #[test]
    fn hierarchy_custom() {
        let custom =
            CustomHierarchyLexicon::new(vec![("Step".into(), 1), ("Substep".into(), 2)]).unwrap();
        let h = HierarchyLexicon::Custom(Arc::new(custom));
        assert_eq!(h.matches_leading("Step 1: Configure"), Some(1));
        assert_eq!(h.matches_leading("Substep 2.1"), Some(2));
        assert_eq!(h.matches_leading("Random text"), None);
    }

    #[test]
    fn hierarchy_custom_rejects_zero_depth() {
        assert!(CustomHierarchyLexicon::new(vec![("Foo".into(), 0)]).is_err());
    }
}
