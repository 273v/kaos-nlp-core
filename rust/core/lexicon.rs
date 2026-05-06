//! Lexicon: a semantic knowledge graph for query expansion.
//!
//! Generic over the data source — works with OpenGloss or any dataset that
//! provides words, senses, and typed semantic edges.

use ahash::{AHashMap, AHashSet};
use serde::{Deserialize, Serialize};

// ─── Relationship types ─────────────────────────────────────────────────────

/// Semantic relationship types between lexeme senses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    Synonym,
    Antonym,
    Hypernym,
    Hyponym,
    Collocation,
    Inflection,
    DerivationNoun,
    DerivationVerb,
    DerivationAdjective,
    DerivationAdverb,
    Cognate,
    EtymologyParent,
    Other,
}

impl RelationType {
    /// Parse from a string label (case-insensitive).
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "synonym" => Self::Synonym,
            "antonym" => Self::Antonym,
            "hypernym" => Self::Hypernym,
            "hyponym" => Self::Hyponym,
            "collocation" => Self::Collocation,
            "inflection" => Self::Inflection,
            "derivation_noun" => Self::DerivationNoun,
            "derivation_verb" => Self::DerivationVerb,
            "derivation_adjective" => Self::DerivationAdjective,
            "derivation_adverb" => Self::DerivationAdverb,
            "cognate" => Self::Cognate,
            "etymology_parent" => Self::EtymologyParent,
            _ => Self::Other,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Synonym => "synonym",
            Self::Antonym => "antonym",
            Self::Hypernym => "hypernym",
            Self::Hyponym => "hyponym",
            Self::Collocation => "collocation",
            Self::Inflection => "inflection",
            Self::DerivationNoun => "derivation_noun",
            Self::DerivationVerb => "derivation_verb",
            Self::DerivationAdjective => "derivation_adjective",
            Self::DerivationAdverb => "derivation_adverb",
            Self::Cognate => "cognate",
            Self::EtymologyParent => "etymology_parent",
            Self::Other => "other",
        }
    }
}

// ─── Edge ───────────────────────────────────────────────────────────────────

/// A directed semantic edge between two terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub relationship_type: RelationType,
    pub target: String,
    /// Part of speech of the source (if sense-specific).
    pub source_pos: Option<String>,
    /// Sense index of the source (if sense-specific).
    pub sense_index: Option<u32>,
}

// ─── Sense ──────────────────────────────────────────────────────────────────

/// A word sense with its part of speech, definition, and per-sense relations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sense {
    pub part_of_speech: String,
    pub sense_index: u32,
    pub definition: String,
    pub synonyms: Vec<String>,
    pub antonyms: Vec<String>,
    pub hypernyms: Vec<String>,
    pub hyponyms: Vec<String>,
}

// ─── LexemeEntry ────────────────────────────────────────────────────────────

/// A lexeme entry in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexemeEntry {
    pub word: String,
    pub senses: Vec<Sense>,
    pub edges: Vec<Edge>,
    // Aggregated lists (all senses combined)
    pub all_synonyms: Vec<String>,
    pub all_antonyms: Vec<String>,
    pub all_hypernyms: Vec<String>,
    pub all_hyponyms: Vec<String>,
    pub all_inflections: Vec<String>,
    pub all_derivations: Vec<String>,
    pub all_collocations: Vec<String>,
}

// ─── Lexicon ────────────────────────────────────────────────────────────────

/// A semantic knowledge graph / lexicon for query expansion.
///
/// Generic — works with OpenGloss or any dataset providing words, senses,
/// and typed edges. Supports dynamic loading and subset loading.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Lexicon {
    entries: AHashMap<String, LexemeEntry>,
}

impl Lexicon {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries in the lexicon.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add or replace an entry. Supports dynamic/incremental loading.
    pub fn add_entry(&mut self, entry: LexemeEntry) {
        self.entries.insert(entry.word.clone(), entry);
    }

    /// Check if a word exists in the lexicon.
    pub fn contains(&self, word: &str) -> bool {
        self.entries.contains_key(word)
    }

    /// Get an entry by word.
    pub fn get(&self, word: &str) -> Option<&LexemeEntry> {
        self.entries.get(word)
    }

    /// Get all words in the lexicon.
    pub fn words(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Build a compact FST-backed word set from this lexicon.
    ///
    /// Includes every headword and (if `include_inflections` is true)
    /// every aggregated inflection, lower-cased and de-duplicated.
    /// This is the data structure to use for fast in-vocabulary
    /// membership checks (e.g. for OCR-quality scoring).
    pub fn to_fst_set(
        &self,
        include_inflections: bool,
    ) -> Result<crate::core::matching::fst_match::FstSet, String> {
        let mut keys: Vec<String> = Vec::with_capacity(self.entries.len() * 2);
        for (word, entry) in &self.entries {
            keys.push(word.to_lowercase());
            if include_inflections {
                for infl in &entry.all_inflections {
                    keys.push(infl.to_lowercase());
                }
            }
        }
        crate::core::matching::fst_match::FstSet::build(keys)
    }

    // ─── Relation lookups ───────────────────────────────────────────────

    /// Get related terms by relationship type.
    /// If `pos` and `sense_index` are provided, filter to that sense.
    pub fn related(
        &self,
        word: &str,
        relation: RelationType,
        pos: Option<&str>,
        sense_index: Option<u32>,
    ) -> Vec<String> {
        let entry = match self.entries.get(word) {
            Some(e) => e,
            None => return vec![],
        };

        // If sense-specific and the relation supports per-sense lists, use those
        if let (Some(pos_str), Some(si)) = (pos, sense_index) {
            if let Some(sense) = entry
                .senses
                .iter()
                .find(|s| s.part_of_speech == pos_str && s.sense_index == si)
            {
                return match relation {
                    RelationType::Synonym => sense.synonyms.clone(),
                    RelationType::Antonym => sense.antonyms.clone(),
                    RelationType::Hypernym => sense.hypernyms.clone(),
                    RelationType::Hyponym => sense.hyponyms.clone(),
                    _ => {
                        // Fall through to edge-based lookup for other types
                        entry
                            .edges
                            .iter()
                            .filter(|e| {
                                e.relationship_type == relation
                                    && e.source_pos.as_deref() == Some(pos_str)
                                    && e.sense_index == Some(si)
                            })
                            .map(|e| e.target.clone())
                            .collect()
                    }
                };
            }
        }

        // Aggregated (all senses)
        match relation {
            RelationType::Synonym => entry.all_synonyms.clone(),
            RelationType::Antonym => entry.all_antonyms.clone(),
            RelationType::Hypernym => entry.all_hypernyms.clone(),
            RelationType::Hyponym => entry.all_hyponyms.clone(),
            RelationType::Inflection => entry.all_inflections.clone(),
            RelationType::Collocation => entry.all_collocations.clone(),
            _ => {
                // Derivations or other — filter from edges
                if matches!(
                    relation,
                    RelationType::DerivationNoun
                        | RelationType::DerivationVerb
                        | RelationType::DerivationAdjective
                        | RelationType::DerivationAdverb
                ) {
                    return entry.all_derivations.clone();
                }
                entry
                    .edges
                    .iter()
                    .filter(|e| e.relationship_type == relation)
                    .map(|e| e.target.clone())
                    .collect()
            }
        }
    }

    /// Convenience: get synonyms for a word.
    pub fn synonyms(&self, word: &str) -> Vec<String> {
        self.related(word, RelationType::Synonym, None, None)
    }

    /// Convenience: get antonyms.
    pub fn antonyms(&self, word: &str) -> Vec<String> {
        self.related(word, RelationType::Antonym, None, None)
    }

    /// Convenience: get hypernyms.
    pub fn hypernyms(&self, word: &str) -> Vec<String> {
        self.related(word, RelationType::Hypernym, None, None)
    }

    /// Convenience: get hyponyms.
    pub fn hyponyms(&self, word: &str) -> Vec<String> {
        self.related(word, RelationType::Hyponym, None, None)
    }

    /// Convenience: get inflections.
    pub fn inflections(&self, word: &str) -> Vec<String> {
        self.related(word, RelationType::Inflection, None, None)
    }

    /// Convenience: get collocations.
    pub fn collocations(&self, word: &str) -> Vec<String> {
        self.related(word, RelationType::Collocation, None, None)
    }

    // ─── Query expansion ────────────────────────────────────────────────

    /// Expand a set of query terms using specified relation types.
    ///
    /// Returns the union of all original terms plus expanded terms.
    /// `max_depth` controls how many hops to follow (1 = direct relations only).
    pub fn expand_query(
        &self,
        terms: &[&str],
        relations: &[RelationType],
        max_depth: usize,
    ) -> AHashSet<String> {
        let mut expanded = AHashSet::new();
        let mut frontier: Vec<String> = terms.iter().map(|t| t.to_string()).collect();

        // Add original terms
        for t in &frontier {
            expanded.insert(t.clone());
        }

        for _depth in 0..max_depth {
            let mut next_frontier = Vec::new();
            for term in &frontier {
                for rel in relations {
                    for related in self.related(term, *rel, None, None) {
                        if expanded.insert(related.clone()) {
                            next_frontier.push(related);
                        }
                    }
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        expanded
    }

    /// Expand with sense-aware filtering.
    ///
    /// Only expands the specified sense of each term, avoiding cross-sense pollution.
    pub fn expand_query_sense_aware(
        &self,
        terms: &[(&str, Option<&str>, Option<u32>)],
        relations: &[RelationType],
    ) -> AHashSet<String> {
        let mut expanded = AHashSet::new();
        for &(word, pos, sense_idx) in terms {
            expanded.insert(word.to_string());
            for rel in relations {
                for related in self.related(word, *rel, pos, sense_idx) {
                    expanded.insert(related);
                }
            }
        }
        expanded
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_lexicon() -> Lexicon {
        let mut lex = Lexicon::new();

        lex.add_entry(LexemeEntry {
            word: "contract".to_string(),
            senses: vec![
                Sense {
                    part_of_speech: "noun".to_string(),
                    sense_index: 0,
                    definition: "A legally binding agreement".to_string(),
                    synonyms: vec!["agreement".to_string(), "pact".to_string()],
                    antonyms: vec!["breach".to_string()],
                    hypernyms: vec!["legal document".to_string()],
                    hyponyms: vec![
                        "employment contract".to_string(),
                        "service contract".to_string(),
                    ],
                },
                Sense {
                    part_of_speech: "verb".to_string(),
                    sense_index: 0,
                    definition: "To shrink or become smaller".to_string(),
                    synonyms: vec!["shrink".to_string(), "compress".to_string()],
                    antonyms: vec!["expand".to_string()],
                    hypernyms: vec!["size change".to_string()],
                    hyponyms: vec![],
                },
            ],
            edges: vec![],
            all_synonyms: vec![
                "agreement".to_string(),
                "pact".to_string(),
                "shrink".to_string(),
                "compress".to_string(),
            ],
            all_antonyms: vec!["breach".to_string(), "expand".to_string()],
            all_hypernyms: vec!["legal document".to_string(), "size change".to_string()],
            all_hyponyms: vec![
                "employment contract".to_string(),
                "service contract".to_string(),
            ],
            all_inflections: vec![
                "contracts".to_string(),
                "contracted".to_string(),
                "contracting".to_string(),
            ],
            all_derivations: vec!["contractual".to_string(), "contractor".to_string()],
            all_collocations: vec![
                "sign a contract".to_string(),
                "breach a contract".to_string(),
            ],
        });

        lex.add_entry(LexemeEntry {
            word: "agreement".to_string(),
            senses: vec![Sense {
                part_of_speech: "noun".to_string(),
                sense_index: 0,
                definition: "A mutual understanding between parties".to_string(),
                synonyms: vec!["contract".to_string(), "accord".to_string()],
                antonyms: vec!["disagreement".to_string()],
                hypernyms: vec!["arrangement".to_string()],
                hyponyms: vec!["treaty".to_string()],
            }],
            edges: vec![],
            all_synonyms: vec!["contract".to_string(), "accord".to_string()],
            all_antonyms: vec!["disagreement".to_string()],
            all_hypernyms: vec!["arrangement".to_string()],
            all_hyponyms: vec!["treaty".to_string()],
            all_inflections: vec!["agreements".to_string()],
            all_derivations: vec![],
            all_collocations: vec![],
        });

        lex
    }

    #[test]
    fn test_basic_lookup() {
        let lex = build_test_lexicon();
        assert!(lex.contains("contract"));
        assert!(!lex.contains("nonexistent"));
        assert_eq!(lex.len(), 2);
    }

    #[test]
    fn test_synonyms() {
        let lex = build_test_lexicon();
        let syns = lex.synonyms("contract");
        assert!(syns.contains(&"agreement".to_string()));
        assert!(syns.contains(&"pact".to_string()));
    }

    #[test]
    fn test_sense_aware_synonyms() {
        let lex = build_test_lexicon();
        // noun sense 0: legal agreement
        let syns = lex.related("contract", RelationType::Synonym, Some("noun"), Some(0));
        assert!(syns.contains(&"agreement".to_string()));
        assert!(!syns.contains(&"shrink".to_string()));

        // verb sense 0: to shrink
        let syns = lex.related("contract", RelationType::Synonym, Some("verb"), Some(0));
        assert!(syns.contains(&"shrink".to_string()));
        assert!(!syns.contains(&"agreement".to_string()));
    }

    #[test]
    fn test_expand_query_synonyms() {
        let lex = build_test_lexicon();
        let expanded = lex.expand_query(&["contract"], &[RelationType::Synonym], 1);
        assert!(expanded.contains("contract"));
        assert!(expanded.contains("agreement"));
        assert!(expanded.contains("pact"));
        assert!(expanded.contains("shrink"));
    }

    #[test]
    fn test_expand_query_depth2() {
        let lex = build_test_lexicon();
        // depth 2: contract -> agreement (syn) -> accord (syn of agreement)
        let expanded = lex.expand_query(&["contract"], &[RelationType::Synonym], 2);
        assert!(expanded.contains("accord")); // 2-hop synonym
    }

    #[test]
    fn test_expand_query_inflections() {
        let lex = build_test_lexicon();
        let expanded = lex.expand_query(&["contract"], &[RelationType::Inflection], 1);
        assert!(expanded.contains("contracts"));
        assert!(expanded.contains("contracted"));
        assert!(expanded.contains("contracting"));
    }

    #[test]
    fn test_expand_query_multiple_relations() {
        let lex = build_test_lexicon();
        let expanded = lex.expand_query(
            &["contract"],
            &[RelationType::Synonym, RelationType::Inflection],
            1,
        );
        assert!(expanded.contains("agreement")); // synonym
        assert!(expanded.contains("contracts")); // inflection
    }

    #[test]
    fn test_expand_query_sense_aware() {
        let lex = build_test_lexicon();
        // Only expand the legal noun sense, not the verb "shrink" sense
        let expanded = lex.expand_query_sense_aware(
            &[("contract", Some("noun"), Some(0))],
            &[RelationType::Synonym, RelationType::Hyponym],
        );
        assert!(expanded.contains("agreement"));
        assert!(expanded.contains("employment contract"));
        assert!(!expanded.contains("shrink"));
    }

    #[test]
    fn test_missing_word() {
        let lex = build_test_lexicon();
        assert!(lex.synonyms("nonexistent").is_empty());
        let expanded = lex.expand_query(&["nonexistent"], &[RelationType::Synonym], 1);
        assert_eq!(expanded.len(), 1); // just the original term
    }

    #[test]
    fn test_dynamic_loading() {
        let mut lex = Lexicon::new();
        assert_eq!(lex.len(), 0);

        lex.add_entry(LexemeEntry {
            word: "test".to_string(),
            senses: vec![],
            edges: vec![],
            all_synonyms: vec!["exam".to_string()],
            all_antonyms: vec![],
            all_hypernyms: vec![],
            all_hyponyms: vec![],
            all_inflections: vec![],
            all_derivations: vec![],
            all_collocations: vec![],
        });
        assert_eq!(lex.len(), 1);
        assert_eq!(lex.synonyms("test"), vec!["exam".to_string()]);
    }
}
