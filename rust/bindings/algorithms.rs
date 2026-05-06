//! PyO3 bindings for string distance and similarity algorithms.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::core::algorithms::{
    edit::{DamerauLevenshtein, Hamming, Jaro, JaroWinkler, Levenshtein, Osa, SorensenDice},
    ngram::{
        NgramCosine, NgramJaccard, NgramOverlap, TokenJaccard, TokenNgramCosine, TokenNgramJaccard,
        TokenNgramOverlap,
    },
    phonetic::{Metaphone, Soundex},
    sequence::{Lcs, LongestCommonSubstring},
    traits::{DistanceOutput, StringDistance},
};

/// Convert a DistanceOutput to a Python dict.
fn output_to_dict(py: Python<'_>, out: &DistanceOutput) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("distance", out.distance)?;
    dict.set_item("normalized", out.normalized)?;
    dict.set_item("similarity", out.similarity)?;
    Ok(dict.into())
}

/// Helper: run a distance algorithm and return the result dict.
fn run_distance(
    py: Python<'_>,
    algo: &dyn StringDistance,
    a: &str,
    b: &str,
) -> PyResult<Py<PyDict>> {
    let out = algo
        .distance(a, b)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    output_to_dict(py, &out)
}

fn compute_distance_output(
    algorithm: &str,
    a: &str,
    b: &str,
    n: usize,
    lowercase: bool,
    prefix_weight: f64,
) -> Result<DistanceOutput, String> {
    match algorithm {
        "levenshtein" => Levenshtein.distance(a, b),
        "damerau" | "damerau-levenshtein" => DamerauLevenshtein.distance(a, b),
        "osa" => Osa.distance(a, b),
        "hamming" => Hamming.distance(a, b),
        "jaro" => Jaro.distance(a, b),
        "jaro-winkler" => JaroWinkler { prefix_weight }.distance(a, b),
        "dice" | "sorensen-dice" => SorensenDice.distance(a, b),
        "soundex" => Soundex.distance(a, b),
        "metaphone" => Metaphone.distance(a, b),
        "lcs" => Lcs.distance(a, b),
        "longest-common-substring" => LongestCommonSubstring.distance(a, b),
        "ngram-jaccard" => NgramJaccard { n }.distance(a, b),
        "ngram-cosine" => NgramCosine { n }.distance(a, b),
        "ngram-overlap" => NgramOverlap { n }.distance(a, b),
        "token-jaccard" => TokenJaccard { lowercase }.distance(a, b),
        "token-ngram-jaccard" => TokenNgramJaccard { n, lowercase }.distance(a, b),
        "token-ngram-cosine" => TokenNgramCosine { n, lowercase }.distance(a, b),
        "token-ngram-overlap" => TokenNgramOverlap { n, lowercase }.distance(a, b),
        _ => return Err(format!("unknown algorithm: {algorithm}")),
    }
    .map_err(|e| e.to_string())
}

// --- Edit distance functions ---

/// Levenshtein edit distance (insert, delete, substitute).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
fn levenshtein(py: Python<'_>, a: &str, b: &str) -> PyResult<Py<PyDict>> {
    run_distance(py, &Levenshtein, a, b)
}

/// Damerau-Levenshtein distance (insert, delete, substitute, transpose).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
fn damerau_levenshtein(py: Python<'_>, a: &str, b: &str) -> PyResult<Py<PyDict>> {
    run_distance(py, &DamerauLevenshtein, a, b)
}

/// Optimal String Alignment distance (restricted Damerau-Levenshtein).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
fn osa(py: Python<'_>, a: &str, b: &str) -> PyResult<Py<PyDict>> {
    run_distance(py, &Osa, a, b)
}

/// Hamming distance (positional character differences).
///
/// Strings must be equal length. Raises ValueError otherwise.
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
fn hamming(py: Python<'_>, a: &str, b: &str) -> PyResult<Py<PyDict>> {
    run_distance(py, &Hamming, a, b)
}

/// Jaro similarity for short strings and names.
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
fn jaro(py: Python<'_>, a: &str, b: &str) -> PyResult<Py<PyDict>> {
    run_distance(py, &Jaro, a, b)
}

/// Jaro-Winkler similarity (Jaro with prefix bonus).
///
/// Args:
///     a: First string.
///     b: Second string.
///     prefix_weight: Prefix scaling factor (default 0.1, must be in [0.0, 0.25]).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
#[pyo3(signature = (a, b, prefix_weight=0.1))]
fn jaro_winkler(py: Python<'_>, a: &str, b: &str, prefix_weight: f64) -> PyResult<Py<PyDict>> {
    run_distance(py, &JaroWinkler { prefix_weight }, a, b)
}

/// Sorensen-Dice coefficient over character bigrams.
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
fn sorensen_dice(py: Python<'_>, a: &str, b: &str) -> PyResult<Py<PyDict>> {
    run_distance(py, &SorensenDice, a, b)
}

// --- Phonetic functions ---

/// Soundex phonetic distance between two strings.
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
fn soundex_distance(py: Python<'_>, a: &str, b: &str) -> PyResult<Py<PyDict>> {
    run_distance(py, &Soundex, a, b)
}

/// Encode a string to its Soundex code (e.g., "Robert" -> "R163").
#[pyfunction]
fn soundex_encode(a: &str) -> String {
    Soundex.encode(a)
}

/// Metaphone phonetic distance between two strings.
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
fn metaphone_distance(py: Python<'_>, a: &str, b: &str) -> PyResult<Py<PyDict>> {
    run_distance(py, &Metaphone, a, b)
}

/// Encode a string to its Metaphone code.
#[pyfunction]
fn metaphone_encode(a: &str) -> String {
    Metaphone.encode(a)
}

// --- Sequence functions ---

/// Longest Common Subsequence distance.
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
fn lcs_distance(py: Python<'_>, a: &str, b: &str) -> PyResult<Py<PyDict>> {
    run_distance(py, &Lcs, a, b)
}

/// Length of the Longest Common Subsequence.
#[pyfunction]
fn lcs_length(a: &str, b: &str) -> usize {
    Lcs.lcs_length(a, b)
}

/// Longest Common Substring (contiguous) similarity.
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
fn longest_common_substring(py: Python<'_>, a: &str, b: &str) -> PyResult<Py<PyDict>> {
    run_distance(py, &LongestCommonSubstring, a, b)
}

/// Length of the Longest Common Substring (contiguous).
#[pyfunction]
fn longest_common_substring_length(a: &str, b: &str) -> usize {
    LongestCommonSubstring.lcss_length(a, b)
}

// --- N-gram functions ---

/// Jaccard similarity over character n-grams.
///
/// Args:
///     a: First string.
///     b: Second string.
///     n: N-gram size (default 2 for bigrams).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
#[pyo3(signature = (a, b, n=2))]
fn ngram_jaccard(py: Python<'_>, a: &str, b: &str, n: usize) -> PyResult<Py<PyDict>> {
    run_distance(py, &NgramJaccard { n }, a, b)
}

/// Cosine similarity over character n-gram frequency vectors.
///
/// Args:
///     a: First string.
///     b: Second string.
///     n: N-gram size (default 2 for bigrams).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
#[pyo3(signature = (a, b, n=2))]
fn ngram_cosine(py: Python<'_>, a: &str, b: &str, n: usize) -> PyResult<Py<PyDict>> {
    run_distance(py, &NgramCosine { n }, a, b)
}

/// Overlap coefficient over character n-gram sets.
///
/// Args:
///     a: First string.
///     b: Second string.
///     n: N-gram size (default 2 for bigrams).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
#[pyo3(signature = (a, b, n=2))]
fn ngram_overlap(py: Python<'_>, a: &str, b: &str, n: usize) -> PyResult<Py<PyDict>> {
    run_distance(py, &NgramOverlap { n }, a, b)
}

// --- Token n-gram functions ---

/// Jaccard similarity over word tokens (unigrams).
///
/// Tokenizes by splitting on whitespace and stripping punctuation.
///
/// Args:
///     a: First string.
///     b: Second string.
///     lowercase: Lowercase tokens before comparison (default False).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
#[pyo3(signature = (a, b, lowercase=false))]
fn token_jaccard(py: Python<'_>, a: &str, b: &str, lowercase: bool) -> PyResult<Py<PyDict>> {
    run_distance(py, &TokenJaccard { lowercase }, a, b)
}

/// Jaccard similarity over token n-grams (word bigrams, trigrams, etc.).
///
/// Args:
///     a: First string.
///     b: Second string.
///     n: Token n-gram size (default 2 for bigrams).
///     lowercase: Lowercase tokens before comparison (default False).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
#[pyo3(signature = (a, b, n=2, lowercase=false))]
fn token_ngram_jaccard(
    py: Python<'_>,
    a: &str,
    b: &str,
    n: usize,
    lowercase: bool,
) -> PyResult<Py<PyDict>> {
    run_distance(py, &TokenNgramJaccard { n, lowercase }, a, b)
}

/// Cosine similarity over token n-gram frequency vectors.
///
/// Args:
///     a: First string.
///     b: Second string.
///     n: Token n-gram size (default 2 for bigrams).
///     lowercase: Lowercase tokens before comparison (default False).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
#[pyo3(signature = (a, b, n=2, lowercase=false))]
fn token_ngram_cosine(
    py: Python<'_>,
    a: &str,
    b: &str,
    n: usize,
    lowercase: bool,
) -> PyResult<Py<PyDict>> {
    run_distance(py, &TokenNgramCosine { n, lowercase }, a, b)
}

/// Overlap coefficient over token n-gram sets.
///
/// Args:
///     a: First string.
///     b: Second string.
///     n: Token n-gram size (default 2 for bigrams).
///     lowercase: Lowercase tokens before comparison (default False).
///
/// Returns a dict with keys: distance, normalized, similarity.
#[pyfunction]
#[pyo3(signature = (a, b, n=2, lowercase=false))]
fn token_ngram_overlap(
    py: Python<'_>,
    a: &str,
    b: &str,
    n: usize,
    lowercase: bool,
) -> PyResult<Py<PyDict>> {
    run_distance(py, &TokenNgramOverlap { n, lowercase }, a, b)
}

/// Compare many string pairs with one configured algorithm.
#[pyfunction]
#[pyo3(signature = (pairs, algorithm="jaro-winkler", n=2, lowercase=false, prefix_weight=0.1))]
fn compare_batch(
    py: Python<'_>,
    pairs: Vec<(String, String)>,
    algorithm: &str,
    n: usize,
    lowercase: bool,
    prefix_weight: f64,
) -> PyResult<Py<PyList>> {
    let algorithm_owned = algorithm.to_string();
    let outputs = py
        .detach(|| {
            pairs
                .iter()
                .map(|(a, b)| {
                    compute_distance_output(&algorithm_owned, a, b, n, lowercase, prefix_weight)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let list = PyList::empty(py);
    for output in &outputs {
        list.append(output_to_dict(py, output)?)?;
    }
    Ok(list.into())
}

/// Register algorithms submodule.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "algorithms")?;

    // Edit distance
    m.add_function(wrap_pyfunction!(levenshtein, &m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein, &m)?)?;
    m.add_function(wrap_pyfunction!(osa, &m)?)?;
    m.add_function(wrap_pyfunction!(hamming, &m)?)?;
    m.add_function(wrap_pyfunction!(jaro, &m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler, &m)?)?;
    m.add_function(wrap_pyfunction!(sorensen_dice, &m)?)?;

    // Phonetic
    m.add_function(wrap_pyfunction!(soundex_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(soundex_encode, &m)?)?;
    m.add_function(wrap_pyfunction!(metaphone_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(metaphone_encode, &m)?)?;

    // Sequence
    m.add_function(wrap_pyfunction!(lcs_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(lcs_length, &m)?)?;
    m.add_function(wrap_pyfunction!(longest_common_substring, &m)?)?;
    m.add_function(wrap_pyfunction!(longest_common_substring_length, &m)?)?;

    // Character n-gram
    m.add_function(wrap_pyfunction!(ngram_jaccard, &m)?)?;
    m.add_function(wrap_pyfunction!(ngram_cosine, &m)?)?;
    m.add_function(wrap_pyfunction!(ngram_overlap, &m)?)?;

    // Token n-gram
    m.add_function(wrap_pyfunction!(token_jaccard, &m)?)?;
    m.add_function(wrap_pyfunction!(token_ngram_jaccard, &m)?)?;
    m.add_function(wrap_pyfunction!(token_ngram_cosine, &m)?)?;
    m.add_function(wrap_pyfunction!(token_ngram_overlap, &m)?)?;
    m.add_function(wrap_pyfunction!(compare_batch, &m)?)?;

    // Register as submodule
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("kaos_nlp_core._rust.algorithms", &m)?;

    Ok(())
}
