//! Text segmentation: sentences (Punkt algorithm), lines, paragraphs.
//!
//! The sentence segmentation is a port of nupunkt-rs — a high-performance
//! implementation of the Punkt algorithm (Kiss & Strunk, 2006) optimized
//! for legal text. It supports training custom models and tunable
//! precision/recall balance.
//!
//! Line and paragraph splitting use simple, reliable newline-based detection.

pub mod boilerplate;
pub mod decision;
pub mod enumerators;
pub mod line_record;
pub mod normalize;
pub mod parameters;
pub mod statistics;
pub mod tokens;
pub mod trainer;
pub mod types;
pub mod utils;

pub use boilerplate::{
    detect_boilerplate, detect_boilerplate_with, BoilerplateKind, BoilerplateOptions,
    BoilerplateRun,
};
pub use enumerators::{
    parse_enumerator, parse_enumerator_with, CustomLexicon, EnumKind, Enumerator, WordLexicon,
};
pub use line_record::{
    extract_line_records, CaseProfile, LineRecord, LineTerminator, PunctProfile,
};
pub use normalize::{normalize, NormalizeError, NormalizeOptions, Normalized};

/// Re-export for the enumerator parser to reuse the strict Roman parser
/// that already lives in the boilerplate module — see `enumerators::recognise_roman`.
pub(crate) use boilerplate::roman_to_u32 as boilerplate_roman_to_u32;

// Re-exports for convenient access
pub use decision::{BoundaryDecision, InferenceConfig, SentenceBoundaryDecider};
pub use parameters::{AbbreviationType, DecisionWeights, PunktParameters, TokenStats};
pub use tokens::PunktToken;
pub use trainer::PunktTrainer;
pub use types::{FactorVec, ScoringConfig, Segment};
pub use utils::{FreqDist, TextPreprocessor};

use rayon::prelude::*;
use std::sync::Arc;

// ─── PunktSentenceTokenizer ─────────────────────────────────────────────────

/// Punkt sentence tokenizer: the main entry point for sentence segmentation.
#[derive(Clone)]
pub struct PunktSentenceTokenizer {
    params: Arc<PunktParameters>,
    preprocessor: TextPreprocessor,
    inference_config: InferenceConfig,
}

impl PunktSentenceTokenizer {
    /// Create a tokenizer with given parameters.
    pub fn from_parameters(params: Arc<PunktParameters>) -> Self {
        Self {
            params,
            preprocessor: TextPreprocessor::default(),
            inference_config: InferenceConfig::default(),
        }
    }

    /// Create a tokenizer with empty parameters.
    pub fn new() -> Self {
        Self::from_parameters(Arc::new(PunktParameters::new()))
    }

    /// Set precision/recall balance (0.0 = max recall, 1.0 = max precision).
    pub fn set_precision_recall_balance(&mut self, balance: f64) {
        self.inference_config.precision_recall_balance = balance.clamp(0.0, 1.0);
    }

    /// Get the current inference config.
    pub fn inference_config(&self) -> &InferenceConfig {
        &self.inference_config
    }

    /// Get a reference to the parameters.
    pub fn parameters(&self) -> &PunktParameters {
        &self.params
    }

    /// Tokenize text into sentences.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let tokens = self.tokenize_words(text);
        let annotated = self.annotate_sentence_boundaries(tokens);
        self.extract_sentences(annotated)
    }

    /// Tokenize with a specific inference config.
    pub fn tokenize_with_config(&self, text: &str, config: &InferenceConfig) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let tokens = self.tokenize_words(text);
        let annotated = self.annotate_sentence_boundaries_with_config(tokens, config);
        self.extract_sentences(annotated)
    }

    /// Get sentence boundaries as byte spans: Vec<(start, end)>.
    pub fn tokenize_spans(&self, text: &str) -> Vec<(usize, usize)> {
        self.tokenize_spans_with_config(text, &self.inference_config)
    }

    /// Get sentence spans with a specific config.
    pub fn tokenize_spans_with_config(
        &self,
        text: &str,
        config: &InferenceConfig,
    ) -> Vec<(usize, usize)> {
        if text.is_empty() {
            return Vec::new();
        }

        let tokens = self.tokenize_words(text);
        let annotated = self.annotate_sentence_boundaries_with_config(tokens, config);

        let mut spans = Vec::new();
        let mut start = 0;

        for (i, token) in annotated.iter().enumerate() {
            if token.sentbreak {
                if let Some(token_start) = token.byte_position {
                    let token_end = token_start + token.tok.len();
                    spans.push((start, token_end));

                    if i + 1 < annotated.len() {
                        if let Some(next_pos) = annotated[i + 1].byte_position {
                            start = next_pos;
                        } else {
                            start = token_end;
                            while start < text.len() && text.as_bytes()[start].is_ascii_whitespace()
                            {
                                start += 1;
                            }
                        }
                    } else {
                        start = text.len();
                    }
                }
            }
        }

        if start < text.len() && !annotated.is_empty() && !annotated.last().unwrap().sentbreak {
            spans.push((start, text.len()));
        }

        spans
    }

    /// Tokenize into paragraphs, where each paragraph is a list of sentences.
    pub fn tokenize_paragraphs(&self, text: &str) -> Vec<Vec<String>> {
        if text.is_empty() {
            return Vec::new();
        }
        let tokens = self.tokenize_words(text);
        let annotated = self.annotate_sentence_boundaries(tokens);
        self.extract_paragraphs(annotated)
    }

    /// Tokenize into paragraphs as flat strings.
    pub fn tokenize_paragraphs_flat(&self, text: &str) -> Vec<String> {
        self.tokenize_paragraphs(text)
            .into_iter()
            .map(|sents| sents.join(" "))
            .collect()
    }

    /// Tokenize multiple texts in parallel using rayon.
    ///
    /// Each text is tokenized independently, so this scales linearly with cores.
    /// Falls back to sequential processing for small batches (< 4 texts).
    pub fn tokenize_batch_parallel(&self, texts: &[String]) -> Vec<Vec<String>> {
        if texts.len() < 4 {
            // Sequential for small batches — rayon overhead not worth it
            return texts.iter().map(|t| self.tokenize(t)).collect();
        }
        texts.par_iter().map(|t| self.tokenize(t)).collect()
    }

    /// Tokenize multiple texts in parallel with a specific inference config.
    pub fn tokenize_batch_parallel_with_config(
        &self,
        texts: &[String],
        config: &InferenceConfig,
    ) -> Vec<Vec<String>> {
        if texts.len() < 4 {
            return texts
                .iter()
                .map(|t| self.tokenize_with_config(t, config))
                .collect();
        }
        texts
            .par_iter()
            .map(|t| self.tokenize_with_config(t, config))
            .collect()
    }

    // ── Internal methods ────────────────────────────────────────────────────

    fn tokenize_words(&self, text: &str) -> Vec<PunktToken> {
        let mut tokens = Vec::new();
        let mut parastart = true;

        let lines: Vec<&str> = text.lines().collect();

        // Pre-calculate line byte offsets
        let mut line_byte_offsets = Vec::with_capacity(lines.len());
        let mut cumulative_offset = 0;
        for (i, line) in lines.iter().enumerate() {
            line_byte_offsets.push(cumulative_offset);
            cumulative_offset += line.len();
            if i < lines.len() - 1 {
                cumulative_offset += 1; // newline
            }
        }

        for (line_idx, line) in lines.iter().enumerate() {
            let mut linestart = true;
            let words_with_spacing = self.preprocessor.word_tokenize_with_spacing(line);

            for (word_idx, (word, spaces_after, _, byte_pos)) in
                words_with_spacing.iter().enumerate()
            {
                let mut token = PunktToken::new(word.clone(), parastart, linestart);

                let line_byte_offset = line_byte_offsets[line_idx];
                token.byte_position = Some(line_byte_offset + byte_pos);

                // Mark abbreviations
                if token.period_final {
                    let type_no_period = token.type_no_period();
                    if self.params.is_abbreviation(&type_no_period) {
                        token.abbr = true;
                    }
                }

                // Mark ellipsis
                if token.is_ellipsis() {
                    token.ellipsis = true;
                }

                // Set spacing
                if word_idx == words_with_spacing.len() - 1 && line_idx < lines.len() - 1 {
                    token.has_newline_after = true;
                    token.spaces_after = 0;
                    if line_idx + 1 < lines.len() && lines[line_idx + 1].trim().is_empty() {
                        token.spaces_after = 2;
                    }
                } else {
                    token.spaces_after = *spaces_after;
                    token.has_newline_after = false;
                }

                tokens.push(token);
                parastart = false;
                linestart = false;
            }

            if line.trim().is_empty() {
                parastart = true;
            }
        }

        tokens
    }

    fn annotate_sentence_boundaries(&self, tokens: Vec<PunktToken>) -> Vec<PunktToken> {
        self.annotate_sentence_boundaries_with_config(tokens, &self.inference_config)
    }

    fn annotate_sentence_boundaries_with_config(
        &self,
        mut tokens: Vec<PunktToken>,
        config: &InferenceConfig,
    ) -> Vec<PunktToken> {
        let decider = SentenceBoundaryDecider::new(&self.params, config);
        let len = tokens.len();

        for i in 0..len {
            if tokens[i].ellipsis {
                tokens[i].sentbreak = false;
                continue;
            }

            let next_token = if i + 1 < len {
                Some(&tokens[i + 1])
            } else {
                None
            };
            let decision = decider.decide(&tokens[i], next_token);
            tokens[i].sentbreak = decision.should_break;
        }

        tokens
    }

    fn extract_sentences(&self, tokens: Vec<PunktToken>) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = Vec::new();

        for token in tokens {
            current.push(token.tok.clone());
            if token.sentbreak && !current.is_empty() {
                sentences.push(current.join(" "));
                current.clear();
            }
        }

        if !current.is_empty() {
            sentences.push(current.join(" "));
        }

        sentences
    }

    fn extract_paragraphs(&self, tokens: Vec<PunktToken>) -> Vec<Vec<String>> {
        let mut paragraphs = Vec::new();
        let mut current_paragraph = Vec::new();
        let mut current_sentence = Vec::new();

        for (i, token) in tokens.iter().enumerate() {
            current_sentence.push(token.tok.clone());

            if token.sentbreak && !current_sentence.is_empty() {
                current_paragraph.push(current_sentence.join(" "));
                current_sentence.clear();

                let is_para_break = (token.spaces_after >= 2 && token.has_newline_after)
                    || (i + 1 < tokens.len() && tokens[i + 1].parastart);

                if is_para_break && !current_paragraph.is_empty() {
                    paragraphs.push(current_paragraph.clone());
                    current_paragraph.clear();
                }
            }
        }

        if !current_sentence.is_empty() {
            current_paragraph.push(current_sentence.join(" "));
        }
        if !current_paragraph.is_empty() {
            paragraphs.push(current_paragraph);
        }

        paragraphs
    }
}

impl Default for PunktSentenceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Line splitting ─────────────────────────────────────────────────────────

/// Split text into lines by newline characters.
pub fn segment_lines(text: &str) -> Vec<Segment> {
    if text.is_empty() {
        return vec![];
    }

    let mut segments = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut start = 0;
    let mut i = 0;

    while i < len {
        if bytes[i] == b'\r' {
            segments.push(Segment::new(start, i, 1.0));
            if i + 1 < len && bytes[i + 1] == b'\n' {
                i += 2;
            } else {
                i += 1;
            }
            start = i;
        } else if bytes[i] == b'\n' {
            segments.push(Segment::new(start, i, 1.0));
            i += 1;
            start = i;
        } else {
            i += 1;
        }
    }

    if start < len {
        segments.push(Segment::new(start, len, 1.0));
    }

    segments
}

// ─── Sentence-aware segmentation ────────────────────────────────────────────

/// Segment text into sentences using the Punkt algorithm.
///
/// Returns byte-offset `Segment`s. Uses the given tokenizer's parameters and
/// precision/recall balance.
pub fn segment_sentences(text: &str, tokenizer: &PunktSentenceTokenizer) -> Vec<Segment> {
    tokenizer
        .tokenize_spans(text)
        .into_iter()
        .map(|(start, end)| Segment::new(start, end, 0.9))
        .collect()
}

/// Segment text into paragraphs using sentence-aware boundaries.
///
/// A paragraph break only occurs at a sentence boundary that is also
/// followed by a blank line (or end of text). This ensures no paragraph
/// is split mid-sentence.
pub fn segment_paragraphs(text: &str, tokenizer: &PunktSentenceTokenizer) -> Vec<Segment> {
    if text.is_empty() {
        return vec![];
    }

    // Get sentence spans first
    let sent_spans = tokenizer.tokenize_spans(text);
    if sent_spans.is_empty() {
        // No sentences found — treat the whole text as one paragraph
        let trimmed = text.trim_end();
        if trimmed.is_empty() {
            return vec![];
        }
        return vec![Segment::new(0, trimmed.len(), 0.9)];
    }

    // Group sentences into paragraphs.
    // A paragraph break occurs when there's a blank line between the end of
    // one sentence and the start of the next.
    let mut paragraphs = Vec::new();
    let mut para_start = sent_spans[0].0;

    for i in 0..sent_spans.len() {
        let (_sent_start, sent_end) = sent_spans[i];

        // Check if there's a paragraph break after this sentence
        let is_last = i + 1 >= sent_spans.len();
        let has_para_break = if !is_last {
            let next_start = sent_spans[i + 1].0;
            let between = &text[sent_end..next_start];
            // A paragraph break = blank line (two or more newlines with optional whitespace)
            let newline_count = between.chars().filter(|&c| c == '\n').count();
            newline_count >= 2
        } else {
            false
        };

        if has_para_break || is_last {
            // End this paragraph at the end of the current sentence
            if sent_end > para_start {
                paragraphs.push(Segment::new(para_start, sent_end, 0.9));
            }
            // Next paragraph starts at the next sentence
            if !is_last {
                para_start = sent_spans[i + 1].0;
            }
        }
    }

    paragraphs
}

/// Segment text into paragraphs using simple blank-line splitting (no sentence awareness).
///
/// Use `segment_paragraphs()` for sentence-aware paragraph splitting.
/// This function is only useful when you don't have or need a Punkt model.
pub fn segment_paragraphs_simple(text: &str) -> Vec<Segment> {
    if text.is_empty() {
        return vec![];
    }

    let lines = segment_lines(text);
    let mut paragraphs = Vec::new();
    let mut para_start: Option<usize> = None;
    let mut blank_count = 0;

    for line in &lines {
        let line_text = line.text(text);
        let is_blank = line_text.trim().is_empty();

        if is_blank {
            blank_count += 1;
        } else {
            if blank_count >= 1 && para_start.is_some() {
                let start = para_start.unwrap();
                let trimmed_end = text[start..line.start].trim_end().len() + start;
                if trimmed_end > start {
                    paragraphs.push(Segment::new(start, trimmed_end, 0.9));
                }
                para_start = Some(line.start);
            } else if para_start.is_none() {
                para_start = Some(line.start);
            }
            blank_count = 0;
        }
    }

    if let Some(start) = para_start {
        let trimmed_end = text[start..].trim_end().len() + start;
        if trimmed_end > start {
            paragraphs.push(Segment::new(start, trimmed_end, 0.9));
        }
    }

    paragraphs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sentences() {
        let tokenizer = PunktSentenceTokenizer::new();
        let sents = tokenizer.tokenize("Hello world. How are you? I am fine!");
        assert_eq!(sents.len(), 3);
    }

    #[test]
    fn test_empty_text() {
        let tokenizer = PunktSentenceTokenizer::new();
        assert!(tokenizer.tokenize("").is_empty());
    }

    #[test]
    fn test_no_terminal_punct() {
        let tokenizer = PunktSentenceTokenizer::new();
        let sents = tokenizer.tokenize("No period here");
        assert_eq!(sents.len(), 1);
    }

    #[test]
    fn test_provided_abbreviation() {
        let mut params = PunktParameters::new();
        params.add_provided_abbreviation("Dr", "test");
        params.add_provided_abbreviation("Mr", "test");
        params.freeze();

        let tokenizer = PunktSentenceTokenizer::from_parameters(Arc::new(params));
        let sents = tokenizer.tokenize("Dr. Smith met Mr. Jones. They talked.");
        // "Dr." and "Mr." should not cause breaks
        assert!(sents.len() <= 3);
        assert!(sents[0].contains("Dr."));
    }

    #[test]
    fn test_precision_recall_balance() {
        let tokenizer = PunktSentenceTokenizer::new();
        let text = "Hello world. How are you.";

        let high_recall = {
            let config = InferenceConfig {
                precision_recall_balance: 0.0,
            };
            tokenizer.tokenize_with_config(text, &config)
        };

        let high_precision = {
            let config = InferenceConfig {
                precision_recall_balance: 1.0,
            };
            tokenizer.tokenize_with_config(text, &config)
        };

        // High recall should produce >= as many sentences as high precision
        assert!(high_recall.len() >= high_precision.len());
    }

    #[test]
    fn test_spans() {
        let tokenizer = PunktSentenceTokenizer::new();
        let text = "Hello world. How are you?";
        let spans = tokenizer.tokenize_spans(text);
        assert!(!spans.is_empty());
        // Verify spans cover valid byte ranges
        for (start, end) in &spans {
            assert!(*start <= *end);
            assert!(*end <= text.len());
            let _ = &text[*start..*end]; // Should not panic
        }
    }

    #[test]
    fn test_paragraphs() {
        let tokenizer = PunktSentenceTokenizer::new();
        let text = "First para sentence.\n\nSecond para sentence.";
        let paras = tokenizer.tokenize_paragraphs(text);
        assert!(!paras.is_empty());
    }

    // --- Lines ---

    #[test]
    fn test_segment_lines() {
        let segs = segment_lines("hello\nworld");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text("hello\nworld"), "hello");
        assert_eq!(segs[1].text("hello\nworld"), "world");
    }

    #[test]
    fn test_segment_lines_empty() {
        assert!(segment_lines("").is_empty());
    }

    #[test]
    fn test_segment_lines_crlf() {
        let text = "line1\r\nline2";
        let segs = segment_lines(text);
        assert_eq!(segs.len(), 2);
    }

    // --- Sentence segmentation ---

    #[test]
    fn test_segment_sentences() {
        let tok = PunktSentenceTokenizer::new();
        let text = "Hello world. How are you? I am fine!";
        let segs = segment_sentences(text, &tok);
        assert_eq!(segs.len(), 3);
        for seg in &segs {
            assert!(!seg.text(text).is_empty());
        }
    }

    // --- Paragraphs (sentence-aware) ---

    #[test]
    fn test_segment_paragraphs() {
        let tok = PunktSentenceTokenizer::new();
        let text = "First para.\n\nSecond para.";
        let segs = segment_paragraphs(text, &tok);
        assert_eq!(segs.len(), 2);
    }

    // --- Paragraphs (simple / no model) ---

    #[test]
    fn test_segment_paragraphs_simple() {
        let text = "First para.\n\nSecond para.";
        let segs = segment_paragraphs_simple(text);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text(text), "First para.");
        assert_eq!(segs[1].text(text), "Second para.");
    }

    #[test]
    fn test_segment_paragraphs_simple_no_break() {
        let text = "Single paragraph\nwith line wrap.";
        let segs = segment_paragraphs_simple(text);
        assert_eq!(segs.len(), 1);
    }
}
