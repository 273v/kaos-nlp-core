"""Benchmarks for string distance/similarity algorithms (pytest-benchmark)."""

import itertools

import pytest

from kaos_nlp_core.algorithms import (
    damerau_levenshtein,
    hamming,
    jaro,
    jaro_winkler,
    lcs_distance,
    lcs_length,
    levenshtein,
    longest_common_substring_length,
    metaphone_encode,
    ngram_cosine,
    ngram_jaccard,
    osa,
    sorensen_dice,
    soundex_encode,
    token_jaccard,
    token_ngram_cosine,
    token_ngram_jaccard,
    token_ngram_overlap,
)

SHORT_PAIRS = [
    ("kitten", "sitting"),
    ("martha", "marhta"),
    ("Robert", "Rupert"),
    ("William", "Williams"),
    ("Philadelphia", "Philadlephia"),
    ("algorithm", "altruistic"),
]


# --- Edit distance: short strings ---


@pytest.mark.benchmark(group="edit_distance_short")
def test_levenshtein_short(benchmark):
    benchmark(lambda: [levenshtein(a, b) for a, b in SHORT_PAIRS])


@pytest.mark.benchmark(group="edit_distance_short")
def test_damerau_levenshtein_short(benchmark):
    benchmark(lambda: [damerau_levenshtein(a, b) for a, b in SHORT_PAIRS])


@pytest.mark.benchmark(group="edit_distance_short")
def test_osa_short(benchmark):
    benchmark(lambda: [osa(a, b) for a, b in SHORT_PAIRS])


@pytest.mark.benchmark(group="edit_distance_short")
def test_jaro_short(benchmark):
    benchmark(lambda: [jaro(a, b) for a, b in SHORT_PAIRS])


@pytest.mark.benchmark(group="edit_distance_short")
def test_jaro_winkler_short(benchmark):
    benchmark(lambda: [jaro_winkler(a, b) for a, b in SHORT_PAIRS])


@pytest.mark.benchmark(group="edit_distance_short")
def test_sorensen_dice_short(benchmark):
    benchmark(lambda: [sorensen_dice(a, b) for a, b in SHORT_PAIRS])


# --- Hamming on 1000-char strings ---


@pytest.mark.benchmark(group="hamming")
def test_hamming_1000(benchmark):
    pairs = [
        (
            "".join(chr(65 + (i + j) % 26) for j in range(1000)),
            "".join(chr(65 + (i + j + 1) % 26) for j in range(1000)),
        )
        for i in range(6)
    ]
    benchmark(lambda: [hamming(a, b) for a, b in pairs])


# --- Edit distance: paragraph-length strings from War and Peace ---


@pytest.mark.benchmark(group="edit_distance_paragraphs")
def test_levenshtein_paragraphs(benchmark, war_and_peace_paragraphs):
    paras = war_and_peace_paragraphs[:20]
    if len(paras) < 2:
        pytest.skip("not enough paragraphs")
    pairs = list(itertools.pairwise(paras))
    benchmark(lambda: [levenshtein(a, b) for a, b in pairs])


@pytest.mark.benchmark(group="edit_distance_paragraphs")
def test_jaro_winkler_paragraphs(benchmark, war_and_peace_paragraphs):
    paras = war_and_peace_paragraphs[:20]
    if len(paras) < 2:
        pytest.skip("not enough paragraphs")
    pairs = list(itertools.pairwise(paras))
    benchmark(lambda: [jaro_winkler(a, b) for a, b in pairs])


@pytest.mark.benchmark(group="edit_distance_paragraphs")
def test_sorensen_dice_paragraphs(benchmark, war_and_peace_paragraphs):
    paras = war_and_peace_paragraphs[:20]
    if len(paras) < 2:
        pytest.skip("not enough paragraphs")
    pairs = list(itertools.pairwise(paras))
    benchmark(lambda: [sorensen_dice(a, b) for a, b in pairs])


# --- N-gram similarity ---


@pytest.mark.benchmark(group="ngram")
@pytest.mark.parametrize("n", [2, 3, 4], ids=lambda n: f"n={n}")
def test_ngram_jaccard(benchmark, n):
    benchmark(lambda: [ngram_jaccard(a, b, n=n) for a, b in SHORT_PAIRS])


@pytest.mark.benchmark(group="ngram")
@pytest.mark.parametrize("n", [2, 3, 4], ids=lambda n: f"n={n}")
def test_ngram_cosine(benchmark, n):
    benchmark(lambda: [ngram_cosine(a, b, n=n) for a, b in SHORT_PAIRS])


# --- Phonetic ---


@pytest.mark.benchmark(group="phonetic")
def test_soundex_encode(benchmark):
    names = ["Robert", "Rupert", "William", "Williams", "Philadelphia", "Mississippi"]
    benchmark(lambda: [soundex_encode(n) for n in names])


@pytest.mark.benchmark(group="phonetic")
def test_metaphone_encode(benchmark):
    names = ["Robert", "Rupert", "William", "Williams", "Philadelphia", "Mississippi"]
    benchmark(lambda: [metaphone_encode(n) for n in names])


# --- LCS ---


@pytest.mark.benchmark(group="lcs")
def test_lcs_short(benchmark):
    benchmark(lambda: [lcs_distance(a, b) for a, b in SHORT_PAIRS])


@pytest.mark.benchmark(group="lcs")
def test_lcs_100_chars(benchmark):
    a = "abcdefghij" * 10
    b = "abcxefghij" * 10
    benchmark(lambda: lcs_length(a, b))


@pytest.mark.benchmark(group="lcs")
def test_longest_common_substring_100(benchmark):
    a = "abcdefghij" * 10
    b = "abcxefghij" * 10
    benchmark(lambda: longest_common_substring_length(a, b))


# --- Token n-gram similarity: short sentences ---

SENTENCE_PAIRS = [
    ("the quick brown fox", "a quick brown dog"),
    ("New York City is great", "New York City is wonderful"),
    ("machine learning algorithms", "deep learning algorithms"),
    ("United States of America", "United Kingdom of Great Britain"),
    ("natural language processing", "natural language understanding"),
    ("the cat sat on the mat", "the dog sat on the log"),
]


@pytest.mark.benchmark(group="token_ngram_short")
def test_token_jaccard_short(benchmark):
    benchmark(lambda: [token_jaccard(a, b, lowercase=True) for a, b in SENTENCE_PAIRS])


@pytest.mark.benchmark(group="token_ngram_short")
@pytest.mark.parametrize("n", [2, 3], ids=lambda n: f"n={n}")
def test_token_ngram_jaccard_short(benchmark, n):
    benchmark(lambda: [token_ngram_jaccard(a, b, n=n, lowercase=True) for a, b in SENTENCE_PAIRS])


@pytest.mark.benchmark(group="token_ngram_short")
@pytest.mark.parametrize("n", [2, 3], ids=lambda n: f"n={n}")
def test_token_ngram_cosine_short(benchmark, n):
    benchmark(lambda: [token_ngram_cosine(a, b, n=n, lowercase=True) for a, b in SENTENCE_PAIRS])


@pytest.mark.benchmark(group="token_ngram_short")
@pytest.mark.parametrize("n", [2, 3], ids=lambda n: f"n={n}")
def test_token_ngram_overlap_short(benchmark, n):
    benchmark(lambda: [token_ngram_overlap(a, b, n=n, lowercase=True) for a, b in SENTENCE_PAIRS])


# --- Token n-gram similarity: paragraphs from War and Peace ---


@pytest.mark.benchmark(group="token_ngram_paragraphs")
def test_token_jaccard_paragraphs(benchmark, war_and_peace_paragraphs):
    paras = war_and_peace_paragraphs[:20]
    if len(paras) < 2:
        pytest.skip("not enough paragraphs")
    pairs = list(itertools.pairwise(paras))
    benchmark(lambda: [token_jaccard(a, b, lowercase=True) for a, b in pairs])


@pytest.mark.benchmark(group="token_ngram_paragraphs")
def test_token_ngram_jaccard_paragraphs(benchmark, war_and_peace_paragraphs):
    paras = war_and_peace_paragraphs[:20]
    if len(paras) < 2:
        pytest.skip("not enough paragraphs")
    pairs = list(itertools.pairwise(paras))
    benchmark(lambda: [token_ngram_jaccard(a, b, n=2, lowercase=True) for a, b in pairs])


@pytest.mark.benchmark(group="token_ngram_paragraphs")
def test_token_ngram_cosine_paragraphs(benchmark, war_and_peace_paragraphs):
    paras = war_and_peace_paragraphs[:20]
    if len(paras) < 2:
        pytest.skip("not enough paragraphs")
    pairs = list(itertools.pairwise(paras))
    benchmark(lambda: [token_ngram_cosine(a, b, n=2, lowercase=True) for a, b in pairs])
