"""Type stubs for kaos_nlp_core._rust.searcher."""

from typing import TypedDict

from kaos_nlp_core._rust.segmentation import PunktTokenizer

class SearchResult(TypedDict):
    text: str
    start: int
    end: int
    score: float

def py_search_sentences(
    text: str,
    query: str,
    tokenizer: PunktTokenizer | None = None,
    top_k: int = 10,
    lowercase: bool = True,
) -> list[SearchResult]: ...
def py_search_paragraphs(
    text: str,
    query: str,
    tokenizer: PunktTokenizer | None = None,
    top_k: int = 10,
    lowercase: bool = True,
) -> list[SearchResult]: ...
