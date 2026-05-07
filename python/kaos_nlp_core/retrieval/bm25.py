"""BM25Retriever -- async adapter around the existing ``Searcher``.

Wraps the synchronous ``Searcher.search()`` call in an async method
so it conforms to the ``Retriever`` protocol.  The underlying Searcher
is built from a ``DocumentCollection`` or provided directly.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from kaos_nlp_core.documents import DocumentCollection
from kaos_nlp_core.retrieval.protocol import corpus_unit_passage_uri
from kaos_nlp_core.search import Searcher, SearchHit


def _group_corpus_units(corpus: Any, group_by: str) -> list[dict[str, Any]]:
    """Group CorpusUnits by an attribute and return one record per group.

    Units are ordered by their row index (iteration order of the corpus).
    The group key becomes ``{first_unit.doc_uri}#{group_value}``; units
    whose group attribute is None fall under ``{doc_uri}#ungrouped-{row}``.
    """
    groups: OrderedDict[str, list[Any]] = OrderedDict()
    ungrouped_counter = 0
    for unit in corpus:
        value = getattr(unit, group_by)
        if value is None:
            key = f"{unit.doc_uri}#ungrouped-{ungrouped_counter}"
            ungrouped_counter += 1
        else:
            key = f"{unit.doc_uri}#{value}"
        groups.setdefault(key, []).append(unit)

    records: list[dict[str, Any]] = []
    for idx, (key, units) in enumerate(groups.items()):
        first = units[0]
        records.append(
            {
                "id": idx,
                "text": "\n".join(u.text for u in units),
                "external_id": key,
                "doc_uri": first.doc_uri,
                "page": first.page,
                "section_ref": first.section_ref,
                "section_title": first.section_title,
            }
        )
    return records


class BM25Retriever:
    """Wraps the existing ``Searcher`` to implement the ``Retriever`` protocol.

    Construction mirrors ``Searcher`` — you can build one from a
    ``DocumentCollection`` via the class method, or wrap an existing
    ``Searcher`` instance directly.

    Example::

        collection = DocumentCollection.from_records(records)
        retriever = BM25Retriever.from_collection(collection)
        results = await retriever.retrieve("statute of limitations", top_k=5)

    From a kaos-ml-core ``Corpus`` (preferred for multi-document RAG)::

        from kaos_ml_core import Corpus
        corpus = Corpus.from_documents([doc1, doc2], level="paragraph")
        retriever = BM25Retriever.from_corpus(corpus)
    """

    def __init__(self, searcher: Searcher) -> None:
        self._searcher = searcher

    @property
    def searcher(self) -> Searcher:
        """Access the underlying ``Searcher`` for advanced configuration."""
        return self._searcher

    async def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[SearchHit]:
        """Retrieve documents using BM25 scoring.

        ``Searcher.search()`` is implemented in Rust and already
        ``py.detach()``-es from the GIL inside the relevant inner loops,
        so the event loop is not held by the C-level work even though
        this method itself is synchronous from Python's perspective.
        Wrapping the call in ``asyncio.to_thread`` only adds an
        executor-scheduling round-trip with no parallelism gain (audit
        perf finding #3b). Awaiting the synchronous call directly under
        ``async def`` is the right shape for the ``Retriever`` protocol.
        """
        return self._searcher.search(query, top_k)

    @classmethod
    def from_collection(
        cls,
        collection: DocumentCollection,
        **kwargs: Any,
    ) -> BM25Retriever:
        """Build a retriever from a document collection.

        All keyword arguments are forwarded to ``Searcher.from_collection``
        (e.g. ``lexicon``, ``scoring``, ``field_weights``).
        """
        searcher = Searcher.from_collection(collection, **kwargs)
        return cls(searcher)

    @classmethod
    def from_documents(
        cls,
        records: Iterable[dict[str, Any]],
        **kwargs: Any,
    ) -> BM25Retriever:
        """Build a retriever from plain records.

        All keyword arguments are forwarded to ``Searcher.from_documents``
        (e.g. ``id_field``, ``text_field``, ``lexicon``).
        """
        searcher = Searcher.from_documents(records, **kwargs)
        return cls(searcher)

    @classmethod
    def from_corpus(
        cls,
        corpus: Any,
        *,
        group_by: str | None = None,
        **kwargs: Any,
    ) -> BM25Retriever:
        """Build a BM25 retriever from a kaos-ml-core ``Corpus``.

        Converts every ``CorpusUnit`` into a record for
        ``DocumentCollection``, builds a **single corpus-wide
        index** (correct IDF statistics across all documents),
        and wraps the resulting ``Searcher``.

        Provenance threading: each record's ``external_id`` is set
        to ``{doc_uri}{block_ref}`` (the passage URI), and
        metadata carries ``page``, ``section_ref``,
        ``section_title``, ``doc_uri`` for downstream verification.

        Args:
            corpus: A ``kaos_ml_core.Corpus`` instance.
            group_by: Optional attribute name on ``CorpusUnit`` to group
                by before indexing (e.g. ``"section_ref"``).  When set,
                units sharing the same attribute value are concatenated
                into one document whose ``external_id`` is
                ``{doc_uri}#{group_value}``.  This gives coarser-grained
                retrieval (e.g. section-level) from a finer-grained
                Corpus (e.g. paragraph-level).
            **kwargs: Forwarded to ``Searcher.from_collection``
                (e.g. ``lexicon``, ``scoring``, ``field_weights``).
        """
        if group_by is not None:
            records = _group_corpus_units(corpus, group_by)
        else:
            records = [
                {
                    "id": unit.row,
                    "text": unit.text,
                    "external_id": corpus_unit_passage_uri(unit),
                    "doc_uri": unit.doc_uri,
                    "page": unit.page,
                    "section_ref": unit.section_ref,
                    "section_title": unit.section_title,
                }
                for unit in corpus
            ]
        collection = DocumentCollection.from_records(
            records,
            id_field="id",
            text_field="text",
            external_id_field="external_id",
            metadata_fields=["doc_uri", "page", "section_ref", "section_title"],
        )
        searcher = Searcher.from_collection(collection, **kwargs)
        return cls(searcher)


__all__ = ["BM25Retriever"]
