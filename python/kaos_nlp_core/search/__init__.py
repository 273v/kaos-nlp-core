"""Search pipeline APIs for documents, sentences, and paragraphs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from kaos_nlp_core._defaults import get_default_punkt_tokenizer
from kaos_nlp_core._rust.searcher import py_search_paragraphs as _search_paragraphs
from kaos_nlp_core._rust.searcher import py_search_sentences as _search_sentences
from kaos_nlp_core.documents import DocumentCollection
from kaos_nlp_core.lexicon import Lexicon
from kaos_nlp_core.segmentation import PunktTokenizer
from kaos_nlp_core.structures import InvertedIndex, ScoredDoc
from kaos_nlp_core.tokenizer import Tokenizer

# ─── Typed result ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A single search result with typed fields.

    All search APIs return ``list[SearchHit]`` so consumers get typed
    attribute access (``hit.score``) instead of dict-key guessing.

    ``external_id`` and ``metadata`` carry caller-supplied context through
    the search pipeline unchanged — e.g. AST block_refs, page numbers,
    section refs.
    """

    doc_id: int
    """Internal document ID within the index."""

    score: float
    """Relevance score (BM25 or TF-IDF). Higher is better."""

    text: str = ""
    """Document text (populated when a DocumentCollection is attached)."""

    external_id: str | None = None
    """Caller-supplied external identifier (e.g. AST block_ref)."""

    fields: dict[str, str] = field(default_factory=dict)
    """Named text fields from the Document."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Caller-supplied metadata (e.g. page number, section ref)."""


@dataclass(frozen=True, slots=True)
class SearchDebugResult:
    """Debug output from ``Searcher.search_debug()``."""

    query: str
    original_terms: list[str]
    expanded_terms: list[str]
    scoring: str
    results: list[SearchHit]


@dataclass(frozen=True, slots=True)
class SegmentHit:
    """A scored text segment (sentence or paragraph) with character offsets."""

    text: str
    """The matched segment text."""

    start: int
    """Character offset start in the source document."""

    end: int
    """Character offset end in the source document."""

    score: float
    """BM25 relevance score."""


# ─── Segment search (sentence / paragraph) ─────────────────────────────────


def search_sentences(
    text: str,
    query: str,
    tokenizer: PunktTokenizer | None = None,
    top_k: int = 10,
    lowercase: bool = True,
) -> list[SegmentHit]:
    """Search within document sentences using the bundled legal model by default."""
    active_tokenizer = tokenizer or get_default_punkt_tokenizer()
    raw = _search_sentences(text, query, active_tokenizer, top_k, lowercase)
    return [
        SegmentHit(text=r["text"], start=r["start"], end=r["end"], score=r["score"]) for r in raw
    ]


def search_paragraphs(
    text: str,
    query: str,
    tokenizer: PunktTokenizer | None = None,
    top_k: int = 10,
    lowercase: bool = True,
) -> list[SegmentHit]:
    """Search within document paragraphs using the bundled legal model by default."""
    active_tokenizer = tokenizer or get_default_punkt_tokenizer()
    raw = _search_paragraphs(text, query, active_tokenizer, top_k, lowercase)
    return [
        SegmentHit(text=r["text"], start=r["start"], end=r["end"], score=r["score"]) for r in raw
    ]


# ─── Searcher ───────────────────────────────────────────────────────────────


class Searcher:
    """Composable Python search pipeline for indexed document collections."""

    def __init__(
        self,
        *,
        index: InvertedIndex,
        tokenizer: Tokenizer | None = None,
        lexicon: Lexicon | None = None,
        collection: DocumentCollection | None = None,
        expand_relations: list[str] | None = None,
        expansion_depth: int = 1,
        scoring: str = "bm25",
        tf_weight: str = "sublinear",
        idf_weight: str = "smooth",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.index = index
        self.tokenizer = tokenizer or Tokenizer(lowercase=True)
        self.lexicon = lexicon
        self.collection = collection
        self.expand_relations = (
            expand_relations if expand_relations is not None else ["synonym", "inflection"]
        )
        self.expansion_depth = expansion_depth
        self.scoring = scoring
        self.tf_weight = tf_weight
        self.idf_weight = idf_weight
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    @classmethod
    def from_collection(
        cls,
        collection: DocumentCollection,
        *,
        tokenizer: Tokenizer | None = None,
        lexicon: Lexicon | None = None,
        field_weights: dict[str, int | float] | None = None,
        **kwargs: Any,
    ) -> Searcher:
        """Build a searcher directly from a document collection."""
        active_tokenizer = tokenizer or Tokenizer(lowercase=True)
        index = collection.build_index(tokenizer=active_tokenizer, field_weights=field_weights)
        return cls(
            index=index,
            tokenizer=active_tokenizer,
            lexicon=lexicon,
            collection=collection,
            **kwargs,
        )

    @classmethod
    def from_documents(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        tokenizer: Tokenizer | None = None,
        lexicon: Lexicon | None = None,
        field_weights: dict[str, int | float] | None = None,
        id_field: str = "id",
        text_field: str = "text",
        external_id_field: str | None = None,
        metadata_fields: Iterable[str] | None = None,
        field_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Searcher:
        """Build a searcher from plain records."""
        collection = DocumentCollection.from_records(
            records,
            id_field=id_field,
            text_field=text_field,
            external_id_field=external_id_field,
            metadata_fields=metadata_fields,
            field_map=field_map,
        )
        return cls.from_collection(
            collection,
            tokenizer=tokenizer,
            lexicon=lexicon,
            field_weights=field_weights,
            **kwargs,
        )

    def _prepare_query(self, query: str) -> tuple[list[str], list[str]]:
        query_terms = self.tokenizer.tokenize_words(query)
        expanded_terms = list(query_terms)
        if self.lexicon and query_terms:
            expanded_terms = self.lexicon.expand_query(
                query_terms,
                self.expand_relations,
                max_depth=self.expansion_depth,
            )
        return query_terms, expanded_terms

    def _enrich(self, raw_results: list[ScoredDoc]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for r in raw_results:
            doc_id = r.doc_id
            score = r.score
            if self.collection is not None:
                doc = self.collection.get(doc_id)
                if doc is not None:
                    hits.append(
                        SearchHit(
                            doc_id=doc_id,
                            score=score,
                            text=doc.text,
                            external_id=doc.external_id,
                            fields=dict(doc.fields),
                            metadata=dict(doc.metadata),
                        )
                    )
                    continue
            hits.append(SearchHit(doc_id=doc_id, score=score))
        return hits

    def _retrieve(self, expanded_terms: list[str], top_k: int) -> list[SearchHit]:
        if self.scoring == "bm25":
            results = self.index.query_bm25(
                expanded_terms,
                top_k=top_k,
                k1=self.bm25_k1,
                b=self.bm25_b,
            )
        elif self.scoring == "tfidf":
            results = self.index.query_tf_idf(
                expanded_terms,
                top_k=top_k,
                tf_weight=self.tf_weight,
                idf_weight=self.idf_weight,
            )
        else:
            msg = (
                f"Unsupported scoring method: '{self.scoring}'. "
                f"Use 'bm25' or 'tfidf'. "
                f"Pass scoring='bm25' (default) or scoring='tfidf' when constructing the Searcher."
            )
            raise ValueError(msg)
        return self._enrich(results)

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        """Search indexed documents using BM25 or TF-IDF."""
        _original_terms, expanded_terms = self._prepare_query(query)
        if not expanded_terms:
            return []
        return self._retrieve(expanded_terms, top_k)

    def search_debug(self, query: str, top_k: int = 10) -> SearchDebugResult:
        """Search with debug info: query terms, expansion, scoring method."""
        original_terms, expanded_terms = self._prepare_query(query)
        if not expanded_terms:
            return SearchDebugResult(
                query=query,
                original_terms=original_terms,
                expanded_terms=expanded_terms,
                scoring=self.scoring,
                results=[],
            )
        results = self._retrieve(expanded_terms, top_k)
        return SearchDebugResult(
            query=query,
            original_terms=original_terms,
            expanded_terms=expanded_terms,
            scoring=self.scoring,
            results=results,
        )

    def search_batch(self, queries: list[str], top_k: int = 10) -> list[list[SearchHit]]:
        """Search multiple queries with one configured pipeline.

        Tokenization + lexicon expansion happen serially in Python (cheap,
        per-query work). The retrieval step uses ``InvertedIndex.query_bm25_batch``
        / ``query_tf_idf_batch`` which runs every query in parallel via Rayon
        under a single GIL release — audit perf finding #3a / P4. Empty
        queries (after expansion) yield an empty result list and skip the
        Rust round trip for that slot.
        """
        prepared: list[list[str]] = []
        empty_slots: list[int] = []
        for idx, query in enumerate(queries):
            _original, expanded = self._prepare_query(query)
            if not expanded:
                empty_slots.append(idx)
                prepared.append([])
            else:
                prepared.append(expanded)

        non_empty_queries = [terms for terms in prepared if terms]
        if not non_empty_queries:
            return [[] for _ in queries]

        if self.scoring == "bm25":
            raw_batches = self.index.query_bm25_batch(
                non_empty_queries,
                top_k=top_k,
                k1=self.bm25_k1,
                b=self.bm25_b,
            )
        elif self.scoring == "tfidf":
            raw_batches = self.index.query_tf_idf_batch(
                non_empty_queries,
                top_k=top_k,
                tf_weight=self.tf_weight,
                idf_weight=self.idf_weight,
            )
        else:
            msg = (
                f"Unsupported scoring method: '{self.scoring}'. "
                f"Use 'bm25' or 'tfidf'. "
                f"Pass scoring='bm25' (default) or scoring='tfidf' when constructing the Searcher."
            )
            raise ValueError(msg)

        results: list[list[SearchHit]] = []
        raw_iter = iter(raw_batches)
        empty = set(empty_slots)
        for idx in range(len(queries)):
            if idx in empty:
                results.append([])
            else:
                results.append(self._enrich(next(raw_iter)))
        return results


__all__ = [
    "SearchDebugResult",
    "SearchHit",
    "Searcher",
    "SegmentHit",
    "search_paragraphs",
    "search_sentences",
]
