//! Class-based TF-IDF (c-TF-IDF) — the BERTopic cluster-labelling kernel.
//!
//! Where ordinary TF-IDF weights a term within one document against a
//! corpus of documents, **c-TF-IDF** treats each *class* (cluster) as a
//! single concatenated document and the *set of classes* as the corpus.
//! A term scores high for a class when it is frequent **within** that
//! class and distinctive **across** classes — exactly the signal a
//! cluster label wants. (Ordinary per-document TF-IDF mislabels clusters:
//! its IDF is computed over individual documents, so a term common to
//! every document inside a cluster is wrongly penalised.)
//!
//! Formula (per term `t`, class `c`):
//!
//! ```text
//! tf_{t,c} = X_{t,c} / Σ_t' X_{t',c}        (L1-normalised class counts)
//! idf_t    = ln(1 + A / f_t)                 (A = avg tokens/class, f_t = global count)
//! W_{t,c}  = tf_{t,c} · idf_t
//! ```
//!
//! with two optional toggles matching BERTopic's `ClassTfidfTransformer`:
//! `reduce_frequent_words` (`tf → sqrt(tf)`, suppresses residual
//! stopwords) and `bm25_weighting` (the smoothed
//! `ln(1 + (A - f_t + 0.5)/(f_t + 0.5))` IDF, steadier on small corpora;
//! clamped at 0 so a more-frequent-than-average term floors at weight 0
//! rather than going negative/NaN).
//!
//! Tokenisation reuses the crate's [`crate::core::tokenizer`] (the same
//! word tokeniser the rest of the stack uses), so labels are consistent
//! with the MinHash / retrieval paths. Per-document tokenisation and
//! per-class scoring fan out across Rayon threads; output is deterministic
//! (terms ranked by descending weight, ties broken alphabetically).
//!
//! Natural log matches both BERTopic (`np.log`) and the crate's existing
//! TF-IDF (`crate::core::structures::inverted_index`), and `f64` matches
//! that module's precision.

use std::cmp::Ordering;

use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use thiserror::Error;

use crate::core::tokenizer::{self, TokenizerConfig};

/// Errors surfaced by [`class_tfidf`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CtfidfError {
    /// `texts` and `class_ids` were different lengths.
    #[error("texts and class_ids length mismatch: {texts} vs {class_ids}")]
    LengthMismatch { texts: usize, class_ids: usize },
    /// A `class_ids` entry was `>= n_classes`.
    #[error("class id {id} is out of range for n_classes {n_classes}")]
    ClassOutOfRange { id: u32, n_classes: usize },
    /// `ngram_range` was not `1 <= min <= max`.
    #[error("invalid ngram range: ({min}, {max}); require 1 <= min <= max")]
    InvalidNgramRange { min: usize, max: usize },
}

/// Append the `[lo, hi]`-gram terms of `tokens` to `out` (space-joined),
/// mirroring the shingle join in `crate::core::minhash`.
fn push_ngrams(tokens: &[String], lo: usize, hi: usize, out: &mut Vec<String>) {
    let n_tokens = tokens.len();
    for n in lo..=hi {
        if n == 0 || n > n_tokens {
            continue;
        }
        for window in tokens.windows(n) {
            out.push(window.join(" "));
        }
    }
}

/// Compute top-`top_k` c-TF-IDF terms for each of `n_classes` classes.
///
/// `texts[i]` belongs to class `class_ids[i]` (`0..n_classes`). Returns a
/// `Vec` of length `n_classes`; entry `c` is the ranked `(term, weight)`
/// list for class `c` (descending weight, alphabetical tie-break),
/// truncated to `top_k`. Empty classes (no surviving terms) yield an
/// empty inner `Vec`.
///
/// * `stopwords` — lowercase tokens to drop before counting (pass an empty
///   set to disable). Single-character tokens are always dropped (matching
///   the common `\w\w+` vectorizer default).
/// * `min_df` — drop terms whose **global** count across all classes is
///   below this (noise floor).
/// * `token_prefix` — when `> 0`, truncate each token to this many
///   characters before counting (the tokenizer's
///   [`TokenizerConfig::with_prefix`]). A cheap, dependency-free,
///   language-agnostic conflation of morphological *and* derivational
///   variants — `token_prefix = 4` merges `automobile` / `automotive` /
///   `autos` into `auto`. Precision-light (it also merges unrelated
///   shared-prefix words) and yields truncated surface forms, so it suits
///   ranking/grouping more than human-facing display. `0` disables it.
#[allow(clippy::too_many_arguments)]
pub fn class_tfidf(
    texts: &[String],
    class_ids: &[u32],
    n_classes: usize,
    ngram_range: (usize, usize),
    top_k: usize,
    min_df: u32,
    reduce_frequent_words: bool,
    bm25: bool,
    lowercase: bool,
    token_prefix: usize,
    stopwords: &AHashSet<String>,
) -> Result<Vec<Vec<(String, f64)>>, CtfidfError> {
    if texts.len() != class_ids.len() {
        return Err(CtfidfError::LengthMismatch {
            texts: texts.len(),
            class_ids: class_ids.len(),
        });
    }
    let (lo, hi) = ngram_range;
    if lo < 1 || hi < lo {
        return Err(CtfidfError::InvalidNgramRange { min: lo, max: hi });
    }
    for &c in class_ids {
        if c as usize >= n_classes {
            return Err(CtfidfError::ClassOutOfRange { id: c, n_classes });
        }
    }
    if n_classes == 0 {
        return Ok(Vec::new());
    }

    let mut cfg = TokenizerConfig::new();
    if lowercase {
        cfg = cfg.lowercase();
    }
    if token_prefix > 0 {
        cfg = cfg.with_prefix(token_prefix);
    }

    // Per-document term counts (parallel over documents).
    let per_doc: Vec<(usize, AHashMap<String, u32>)> = (0..texts.len())
        .into_par_iter()
        .map(|i| {
            let words: Vec<String> = tokenizer::tokenize_words(&texts[i], &cfg)
                .into_iter()
                .filter(|w| w.chars().count() >= 2 && !stopwords.contains(w))
                .collect();
            let mut terms: Vec<String> = Vec::new();
            push_ngrams(&words, lo, hi, &mut terms);
            let mut counts: AHashMap<String, u32> = AHashMap::new();
            for t in terms {
                *counts.entry(t).or_insert(0) += 1;
            }
            (class_ids[i] as usize, counts)
        })
        .collect();

    // Reduce into per-class counts and the global term frequency f_t.
    let mut class_counts: Vec<AHashMap<String, u32>> = vec![AHashMap::new(); n_classes];
    for (cls, counts) in per_doc {
        let target = &mut class_counts[cls];
        for (term, n) in counts {
            *target.entry(term).or_insert(0) += n;
        }
    }

    let mut global: AHashMap<String, u32> = AHashMap::new();
    let mut total_tokens: u64 = 0;
    for counts in &class_counts {
        for (term, n) in counts {
            *global.entry(term.clone()).or_insert(0) += *n;
            total_tokens += u64::from(*n);
        }
    }
    let avg = total_tokens as f64 / n_classes as f64;

    // Per-class scoring (parallel; output order = class index order).
    let result: Vec<Vec<(String, f64)>> = class_counts
        .par_iter()
        .map(|counts| {
            let class_total: u64 = counts.values().map(|v| u64::from(*v)).sum();
            if class_total == 0 {
                return Vec::new();
            }
            let denom = class_total as f64;
            let mut scored: Vec<(String, f64)> = Vec::with_capacity(counts.len());
            for (term, &x) in counts {
                let f_t = global[term];
                if f_t < min_df {
                    continue;
                }
                let mut tf = x as f64 / denom;
                if reduce_frequent_words {
                    tf = tf.sqrt();
                }
                let f = f_t as f64;
                let idf = if bm25 {
                    // Clamp at 0: a term more frequent than the class average
                    // floors at weight 0 rather than going negative/NaN.
                    (1.0 + (avg - f + 0.5) / (f + 0.5)).max(1.0).ln()
                } else {
                    (1.0 + avg / f).ln()
                };
                let weight = tf * idf;
                if weight > 0.0 {
                    scored.push((term.clone(), weight));
                }
            }
            scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            scored.truncate(top_k);
            scored
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops() -> AHashSet<String> {
        ["the", "a", "for", "of", "and", "to", "was", "by"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn separates_class_vocabularies() {
        let t = texts(&[
            "the court granted the motion for summary judgment",
            "plaintiff filed a motion for summary judgment",
            "mix the flour sugar and eggs into a batter",
            "bake the batter with flour for thirty minutes",
        ]);
        let cls = vec![0u32, 0, 1, 1];
        let out = class_tfidf(&t, &cls, 2, (1, 1), 5, 1, false, false, true, 0, &stops()).unwrap();
        let top0: Vec<&str> = out[0].iter().map(|(t, _)| t.as_str()).collect();
        let top1: Vec<&str> = out[1].iter().map(|(t, _)| t.as_str()).collect();
        // Litigation terms in class 0, baking terms in class 1, no leakage.
        assert!(top0
            .iter()
            .any(|w| ["motion", "summary", "judgment", "court"].contains(w)));
        assert!(top1
            .iter()
            .any(|w| ["flour", "batter", "eggs", "bake"].contains(w)));
        assert!(!top0.contains(&"flour"));
        assert!(!top1.contains(&"judgment"));
    }

    #[test]
    fn ranks_descending_with_alpha_tiebreak() {
        let t = texts(&["alpha alpha beta gamma", "delta delta epsilon"]);
        let out = class_tfidf(
            &t,
            &[0, 1],
            2,
            (1, 1),
            10,
            1,
            false,
            false,
            true,
            0,
            &stops(),
        )
        .unwrap();
        for class in &out {
            for w in class.windows(2) {
                assert!(w[0].1 > w[1].1 || (w[0].1 == w[1].1 && w[0].0 <= w[1].0));
            }
        }
    }

    #[test]
    fn bigrams_present_with_ngram_range() {
        let t = texts(&["summary judgment motion", "unrelated baking content here"]);
        let out = class_tfidf(
            &t,
            &[0, 1],
            2,
            (1, 2),
            20,
            1,
            false,
            false,
            true,
            0,
            &stops(),
        )
        .unwrap();
        let top0: Vec<&str> = out[0].iter().map(|(t, _)| t.as_str()).collect();
        assert!(
            top0.iter().any(|w| w.contains(' ')),
            "expected a bigram in {top0:?}"
        );
    }

    #[test]
    fn top_k_truncates() {
        let t = texts(&["one two three four five six seven", "x y z"]);
        let out = class_tfidf(
            &t,
            &[0, 1],
            2,
            (1, 1),
            3,
            1,
            false,
            false,
            true,
            0,
            &AHashSet::new(),
        )
        .unwrap();
        assert!(out[0].len() <= 3);
    }

    #[test]
    fn min_df_drops_rare_terms() {
        // 'unique' appears once globally; min_df=2 should drop it.
        let t = texts(&["common common unique", "common common common"]);
        let out = class_tfidf(
            &t,
            &[0, 1],
            2,
            (1, 1),
            10,
            2,
            false,
            false,
            true,
            0,
            &AHashSet::new(),
        )
        .unwrap();
        let top0: Vec<&str> = out[0].iter().map(|(t, _)| t.as_str()).collect();
        assert!(!top0.contains(&"unique"));
    }

    #[test]
    fn stopwords_removed() {
        let t = texts(&["the the the motion", "the the the baking"]);
        let out = class_tfidf(
            &t,
            &[0, 1],
            2,
            (1, 1),
            10,
            1,
            false,
            false,
            true,
            0,
            &stops(),
        )
        .unwrap();
        for class in &out {
            assert!(!class.iter().any(|(w, _)| w == "the"));
        }
    }

    #[test]
    fn reduce_frequent_words_and_bm25_run_and_stay_finite() {
        let t = texts(&["alpha alpha beta", "alpha gamma gamma"]);
        let out = class_tfidf(
            &t,
            &[0, 1],
            2,
            (1, 1),
            10,
            1,
            true,
            true,
            true,
            0,
            &AHashSet::new(),
        )
        .unwrap();
        for class in &out {
            for (_, w) in class {
                assert!(
                    w.is_finite() && *w >= 0.0,
                    "weight {w} not finite/non-negative"
                );
            }
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let t = texts(&[
            "the quick brown fox",
            "lazy dog sleeps soundly",
            "quick brown dog runs",
        ]);
        let cls = vec![0u32, 1, 0];
        let a = class_tfidf(&t, &cls, 2, (1, 2), 10, 1, false, false, true, 0, &stops()).unwrap();
        let b = class_tfidf(&t, &cls, 2, (1, 2), 10, 1, false, false, true, 0, &stops()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn length_mismatch_errors() {
        let t = texts(&["a b", "c d"]);
        assert!(matches!(
            class_tfidf(
                &t,
                &[0],
                1,
                (1, 1),
                5,
                1,
                false,
                false,
                true,
                0,
                &AHashSet::new()
            ),
            Err(CtfidfError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn class_out_of_range_errors() {
        let t = texts(&["a b"]);
        assert!(matches!(
            class_tfidf(
                &t,
                &[5],
                2,
                (1, 1),
                5,
                1,
                false,
                false,
                true,
                0,
                &AHashSet::new()
            ),
            Err(CtfidfError::ClassOutOfRange {
                id: 5,
                n_classes: 2
            })
        ));
    }

    #[test]
    fn invalid_ngram_range_errors() {
        let t = texts(&["a b"]);
        assert!(matches!(
            class_tfidf(
                &t,
                &[0],
                1,
                (2, 1),
                5,
                1,
                false,
                false,
                true,
                0,
                &AHashSet::new()
            ),
            Err(CtfidfError::InvalidNgramRange { .. })
        ));
    }

    #[test]
    fn token_prefix_conflates_variants() {
        // Without prefix, automobile/automotive/autos are distinct terms;
        // with prefix=4 they all collapse to "auto" and count together.
        let t = texts(&["automobile automotive autos", "kitchen cooking recipe"]);
        let no_prefix = class_tfidf(
            &t,
            &[0, 1],
            2,
            (1, 1),
            10,
            1,
            false,
            false,
            true,
            0,
            &AHashSet::new(),
        )
        .unwrap();
        let prefixed = class_tfidf(
            &t,
            &[0, 1],
            2,
            (1, 1),
            10,
            1,
            false,
            false,
            true,
            4,
            &AHashSet::new(),
        )
        .unwrap();
        let class0_prefixed: Vec<&str> = prefixed[0].iter().map(|(t, _)| t.as_str()).collect();
        // Three distinct auto* terms become one "auto".
        assert!(class0_prefixed.contains(&"auto"));
        assert!(!class0_prefixed.contains(&"automobile"));
        // And there are fewer distinct terms in class 0 after conflation.
        assert!(prefixed[0].len() < no_prefix[0].len());
    }

    #[test]
    fn empty_input_ok() {
        let out = class_tfidf(
            &[],
            &[],
            0,
            (1, 1),
            5,
            1,
            false,
            false,
            true,
            0,
            &AHashSet::new(),
        )
        .unwrap();
        assert!(out.is_empty());
    }
}
