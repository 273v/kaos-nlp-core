"""Type stubs for ``kaos_nlp_core._rust.ctfidf``.

Runtime implementation: ``rust/bindings/ctfidf.rs``.
"""

def class_tfidf(
    texts: list[str],
    class_ids: list[int],
    n_classes: int,
    ngram_min: int,
    ngram_max: int,
    top_k: int,
    min_df: int = 1,
    reduce_frequent_words: bool = False,
    bm25_weighting: bool = False,
    lowercase: bool = True,
    token_prefix: int = 0,
    stopwords: list[str] | None = None,
) -> list[list[tuple[str, float]]]:
    """Class-based TF-IDF: ranked ``(term, weight)`` per class index.

    Returns a list of length ``n_classes``; entry ``c`` is the top-``top_k``
    ranked terms for class ``c`` (descending weight, alphabetical
    tie-break).
    """
