//! High-level search pipeline composing Tokenizer + Lexicon + InvertedIndex.
//!
//! Provides a `Searcher` that:
//! 1. Tokenizes the query text using the configured `TokenizerConfig`
//! 2. Optionally expands query terms via a `Lexicon`
//! 3. Retrieves ranked results from an `InvertedIndex` (BM25 or TF-IDF)
//!
//! Also supports sentence-level and paragraph-level search via the Punkt
//! segmenter: segment a document → index each segment → search → return
//! matching segments with their spans.

use crate::core::lexicon::{Lexicon, RelationType};
use crate::core::segmentation::PunktSentenceTokenizer;
use crate::core::structures::inverted_index::{
    Bm25Params, IdfWeight, InvertedIndex, ScoredDoc, TfWeight,
};
use crate::core::tokenizer::{tokenize_words, TokenizerConfig};

/// Scoring method for retrieval.
#[derive(Debug, Clone)]
pub enum ScoringMethod {
    Bm25(Bm25Params),
    TfIdf { tf: TfWeight, idf: IdfWeight },
}

impl Default for ScoringMethod {
    fn default() -> Self {
        ScoringMethod::Bm25(Bm25Params::default())
    }
}

/// Expansion configuration for query terms.
#[derive(Debug, Clone)]
pub struct ExpansionConfig {
    /// Relation types to expand on.
    pub relations: Vec<RelationType>,
    /// Maximum expansion depth (hops).
    pub max_depth: usize,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            relations: vec![RelationType::Synonym, RelationType::Inflection],
            max_depth: 1,
        }
    }
}

/// A composable search pipeline: Tokenizer → optional Lexicon expansion → InvertedIndex retrieval.
pub struct Searcher<'a> {
    tokenizer_config: &'a TokenizerConfig,
    index: &'a InvertedIndex,
    lexicon: Option<&'a Lexicon>,
    expansion: Option<ExpansionConfig>,
    scoring: ScoringMethod,
}

impl<'a> Searcher<'a> {
    /// Create a new Searcher with required components.
    pub fn new(tokenizer_config: &'a TokenizerConfig, index: &'a InvertedIndex) -> Self {
        Self {
            tokenizer_config,
            index,
            lexicon: None,
            expansion: None,
            scoring: ScoringMethod::default(),
        }
    }

    /// Set the lexicon for query expansion.
    pub fn with_lexicon(mut self, lexicon: &'a Lexicon, expansion: ExpansionConfig) -> Self {
        self.lexicon = Some(lexicon);
        self.expansion = Some(expansion);
        self
    }

    /// Set the scoring method.
    pub fn with_scoring(mut self, scoring: ScoringMethod) -> Self {
        self.scoring = scoring;
        self
    }

    /// Search for a query string. Returns top-K ranked results.
    ///
    /// Pipeline:
    /// 1. Tokenize query text → terms
    /// 2. If lexicon configured, expand terms via configured relations
    /// 3. Score and rank documents, return top_k
    pub fn search(&self, query: &str, top_k: usize) -> Vec<ScoredDoc> {
        // Step 1: Tokenize
        let mut terms = tokenize_words(query, self.tokenizer_config);

        if terms.is_empty() {
            return vec![];
        }

        // Step 2: Expand (optional)
        if let (Some(lexicon), Some(expansion)) = (self.lexicon, &self.expansion) {
            let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
            let expanded = lexicon.expand_query(&refs, &expansion.relations, expansion.max_depth);
            terms = expanded.into_iter().collect();
        }

        // Step 3: Retrieve
        let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
        match &self.scoring {
            ScoringMethod::Bm25(params) => self.index.query_bm25(&refs, params, top_k),
            ScoringMethod::TfIdf { tf, idf } => self.index.query_tf_idf(&refs, *tf, *idf, top_k),
        }
    }

    /// Search and return expanded terms along with results (useful for debugging).
    pub fn search_with_expansion(
        &self,
        query: &str,
        top_k: usize,
    ) -> (Vec<String>, Vec<ScoredDoc>) {
        let mut terms = tokenize_words(query, self.tokenizer_config);

        if terms.is_empty() {
            return (vec![], vec![]);
        }

        // Expand
        if let (Some(lexicon), Some(expansion)) = (self.lexicon, &self.expansion) {
            let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
            let expanded = lexicon.expand_query(&refs, &expansion.relations, expansion.max_depth);
            terms = expanded.into_iter().collect();
        }

        let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
        let results = match &self.scoring {
            ScoringMethod::Bm25(params) => self.index.query_bm25(&refs, params, top_k),
            ScoringMethod::TfIdf { tf, idf } => self.index.query_tf_idf(&refs, *tf, *idf, top_k),
        };

        (terms, results)
    }
}

/// Convenience: build an index from document texts using a tokenizer config.
pub fn build_index<'a, I>(docs: I, config: &TokenizerConfig) -> InvertedIndex
where
    I: IntoIterator<Item = (u32, &'a str)>,
{
    let mut index = InvertedIndex::new();
    for (doc_id, text) in docs {
        let words = tokenize_words(text, config);
        let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        index.add_document(doc_id, &refs);
    }
    index
}

// ─── Sentence / paragraph search ────────────────────────────────────────────

/// A scored text segment (sentence or paragraph) with its span in the source.
#[derive(Debug, Clone)]
pub struct ScoredSegment {
    /// The segment text.
    pub text: String,
    /// Byte offset start in the source document.
    pub start: usize,
    /// Byte offset end in the source document.
    pub end: usize,
    /// BM25/TF-IDF score.
    pub score: f64,
}

/// Search within a document's sentences.
///
/// Segments the document into sentences, builds a temporary index,
/// queries it, and returns the matching sentences with their spans and scores.
pub fn search_sentences(
    document: &str,
    query: &str,
    segmenter: &PunktSentenceTokenizer,
    word_config: &TokenizerConfig,
    top_k: usize,
) -> Vec<ScoredSegment> {
    if document.is_empty() || query.is_empty() {
        return vec![];
    }

    // 1. Segment into sentences with byte spans
    let spans = segmenter.tokenize_spans(document);
    if spans.is_empty() {
        return vec![];
    }

    // 2. Build a temporary index over sentences
    let mut index = InvertedIndex::new();
    for (i, &(start, end)) in spans.iter().enumerate() {
        let sent_text = &document[start..end];
        let words = tokenize_words(sent_text, word_config);
        let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        index.add_document(i as u32, &refs);
    }

    // 3. Tokenize the query
    let query_terms = tokenize_words(query, word_config);
    if query_terms.is_empty() {
        return vec![];
    }

    // 4. Search
    let refs: Vec<&str> = query_terms.iter().map(|s| s.as_str()).collect();
    let results = index.query_bm25(&refs, &Bm25Params::default(), top_k);

    // 5. Map back to segments
    results
        .into_iter()
        .filter_map(|scored| {
            let idx = scored.doc_id as usize;
            if idx < spans.len() {
                let (start, end) = spans[idx];
                Some(ScoredSegment {
                    text: document[start..end].to_string(),
                    start,
                    end,
                    score: scored.score,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Search within a document's paragraphs.
///
/// Segments the document into paragraphs (sentence-aware), builds a temporary
/// index, queries it, and returns the matching paragraphs with spans and scores.
pub fn search_paragraphs(
    document: &str,
    query: &str,
    segmenter: &PunktSentenceTokenizer,
    word_config: &TokenizerConfig,
    top_k: usize,
) -> Vec<ScoredSegment> {
    if document.is_empty() || query.is_empty() {
        return vec![];
    }

    // 1. Segment into paragraphs
    let para_segments = crate::core::segmentation::segment_paragraphs(document, segmenter);
    if para_segments.is_empty() {
        return vec![];
    }

    // 2. Build a temporary index over paragraphs
    let mut index = InvertedIndex::new();
    for (i, seg) in para_segments.iter().enumerate() {
        let para_text = seg.text(document);
        let words = tokenize_words(para_text, word_config);
        let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        index.add_document(i as u32, &refs);
    }

    // 3. Tokenize the query
    let query_terms = tokenize_words(query, word_config);
    if query_terms.is_empty() {
        return vec![];
    }

    // 4. Search
    let refs: Vec<&str> = query_terms.iter().map(|s| s.as_str()).collect();
    let results = index.query_bm25(&refs, &Bm25Params::default(), top_k);

    // 5. Map back to segments
    results
        .into_iter()
        .filter_map(|scored| {
            let idx = scored.doc_id as usize;
            if idx < para_segments.len() {
                let seg = &para_segments[idx];
                Some(ScoredSegment {
                    text: seg.text(document).to_string(),
                    start: seg.start,
                    end: seg.end,
                    score: scored.score,
                })
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> TokenizerConfig {
        TokenizerConfig::new().lowercase()
    }

    fn sample_index(config: &TokenizerConfig) -> InvertedIndex {
        let docs = [
            (0, "The cat sat on the mat"),
            (1, "The dog sat on the log"),
            (2, "A cat and a dog played together"),
            (3, "The quick brown fox jumps over the lazy dog"),
            (4, "Automobiles and transportation infrastructure"),
        ];
        build_index(docs, config)
    }

    fn sample_lexicon() -> Lexicon {
        use crate::core::lexicon::LexemeEntry;
        let mut lex = Lexicon::new();
        lex.add_entry(LexemeEntry {
            word: "cat".to_string(),
            senses: vec![],
            edges: vec![],
            all_synonyms: vec!["feline".to_string(), "kitty".to_string()],
            all_antonyms: vec![],
            all_hypernyms: vec!["animal".to_string()],
            all_hyponyms: vec![],
            all_inflections: vec!["cats".to_string()],
            all_derivations: vec![],
            all_collocations: vec![],
        });
        lex.add_entry(LexemeEntry {
            word: "dog".to_string(),
            senses: vec![],
            edges: vec![],
            all_synonyms: vec!["canine".to_string(), "hound".to_string()],
            all_antonyms: vec![],
            all_hypernyms: vec!["animal".to_string()],
            all_hyponyms: vec![],
            all_inflections: vec!["dogs".to_string()],
            all_derivations: vec![],
            all_collocations: vec![],
        });
        lex
    }

    #[test]
    fn test_basic_search() {
        let config = sample_config();
        let index = sample_index(&config);
        let searcher = Searcher::new(&config, &index);
        let results = searcher.search("cat", 10);
        assert!(!results.is_empty());
        // Should find docs 0 and 2
        let ids: Vec<u32> = results.iter().map(|r| r.doc_id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_search_with_expansion() {
        let config = sample_config();
        let index = sample_index(&config);
        let lexicon = sample_lexicon();
        let expansion = ExpansionConfig::default();
        let searcher = Searcher::new(&config, &index).with_lexicon(&lexicon, expansion);
        let (terms, results) = searcher.search_with_expansion("cat", 10);
        // Should include expanded terms
        assert!(terms.contains(&"cat".to_string()));
        // Original results should still be found
        let ids: Vec<u32> = results.iter().map(|r| r.doc_id).collect();
        assert!(ids.contains(&0));
    }

    #[test]
    fn test_search_empty_query() {
        let config = sample_config();
        let index = sample_index(&config);
        let searcher = Searcher::new(&config, &index);
        assert!(searcher.search("", 10).is_empty());
    }

    #[test]
    fn test_search_no_results() {
        let config = sample_config();
        let index = sample_index(&config);
        let searcher = Searcher::new(&config, &index);
        assert!(searcher.search("xyzzy_nonexistent", 10).is_empty());
    }

    #[test]
    fn test_search_tfidf_scoring() {
        let config = sample_config();
        let index = sample_index(&config);
        let searcher = Searcher::new(&config, &index).with_scoring(ScoringMethod::TfIdf {
            tf: TfWeight::Sublinear,
            idf: IdfWeight::Smooth,
        });
        let results = searcher.search("cat dog", 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_build_index() {
        let config = sample_config();
        let docs = [(0, "hello world"), (1, "hello there")];
        let index = build_index(docs, &config);
        assert_eq!(index.doc_count(), 2);
        assert!(index.doc_freq("hello") == 2);
    }

    // --- Sentence / paragraph search ---

    #[test]
    fn test_search_sentences() {
        let segmenter = PunktSentenceTokenizer::new();
        let config = sample_config();
        let doc = "The cat sat on the mat. The dog chased a ball. A bird flew away.";
        let results = search_sentences(doc, "cat mat", &segmenter, &config, 3);
        assert!(!results.is_empty());
        // First result should be the sentence about cats
        assert!(results[0].text.contains("cat"));
    }

    #[test]
    fn test_search_sentences_empty() {
        let segmenter = PunktSentenceTokenizer::new();
        let config = sample_config();
        assert!(search_sentences("", "cat", &segmenter, &config, 3).is_empty());
        assert!(search_sentences("hello.", "", &segmenter, &config, 3).is_empty());
    }

    #[test]
    fn test_search_paragraphs() {
        let segmenter = PunktSentenceTokenizer::new();
        let config = sample_config();
        let doc =
            "The cat sat on the mat. It was a lazy day.\n\nThe dog ran in the park. It was fun.";
        let results = search_paragraphs(doc, "cat mat", &segmenter, &config, 3);
        assert!(!results.is_empty());
        assert!(results[0].text.contains("cat"));
    }

    #[test]
    fn test_search_sentences_spans_valid() {
        let segmenter = PunktSentenceTokenizer::new();
        let config = sample_config();
        let doc = "First sentence here. Second one here. Third sentence.";
        let results = search_sentences(doc, "second", &segmenter, &config, 3);
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.start <= r.end);
            assert!(r.end <= doc.len());
            assert_eq!(&doc[r.start..r.end], r.text);
        }
    }
}
