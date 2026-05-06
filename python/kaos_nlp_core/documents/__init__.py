"""Document collections and corpus loaders for search and indexing workflows."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaos_nlp_core._rust.structures import InvertedIndex as _RustInvertedIndex
from kaos_nlp_core.structures import InvertedIndex
from kaos_nlp_core.tokenizer import Tokenizer


def _normalize_weight(weight: int | float) -> int:
    value = max(1, round(weight))
    return value


@dataclass(slots=True)  # Mutable: __post_init__ normalizes fields/text after construction
class Document:
    """A single text document with optional fields and metadata."""

    doc_id: int
    text: str
    external_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fields:
            self.fields = {"text": self.text}
        elif "text" not in self.fields:
            self.fields["text"] = self.text
        if not self.text:
            self.text = self.fields.get("text", "")

    def render_text(
        self,
        field_weights: dict[str, int | float] | None = None,
        separator: str = "\n\n",
    ) -> str:
        """Render a document into a single indexable text string."""
        if not field_weights:
            return self.text

        parts: list[str] = []
        for field_name, weight in field_weights.items():
            value = self.fields.get(field_name, "").strip()
            if not value:
                continue
            parts.extend([value] * _normalize_weight(weight))
        if not parts:
            return self.text
        return separator.join(parts)


class DocumentCollection:
    """Managed collection of documents with simple corpus loaders."""

    def __init__(self, documents: Iterable[Document] | None = None) -> None:
        self._documents: list[Document] = []
        self._by_id: dict[int, Document] = {}
        if documents is not None:
            for document in documents:
                self.add(document)

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents)

    def __contains__(self, doc_id: int) -> bool:
        return doc_id in self._by_id

    def add(self, document: Document) -> None:
        """Add a document to the collection."""
        if document.doc_id in self._by_id:
            msg = f"duplicate document id: {document.doc_id}"
            raise ValueError(msg)
        self._documents.append(document)
        self._by_id[document.doc_id] = document

    def get(self, doc_id: int) -> Document | None:
        """Look up a document by numeric id."""
        return self._by_id.get(doc_id)

    def ids(self) -> list[int]:
        """Return all document ids in insertion order."""
        return [document.doc_id for document in self._documents]

    def to_records(self) -> list[dict[str, Any]]:
        """Serialize the collection to plain Python records."""
        records: list[dict[str, Any]] = []
        for document in self._documents:
            records.append(
                {
                    "id": document.doc_id,
                    "external_id": document.external_id,
                    "text": document.text,
                    "fields": document.fields,
                    "metadata": document.metadata,
                }
            )
        return records

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        id_field: str = "id",
        text_field: str = "text",
        external_id_field: str | None = None,
        metadata_fields: Iterable[str] | None = None,
        field_map: dict[str, str] | None = None,
    ) -> DocumentCollection:
        """Build a collection from plain records."""
        documents: list[Document] = []
        for auto_id, record in enumerate(records):
            doc_id = int(record.get(id_field, auto_id))
            metadata: dict[str, Any]
            if metadata_fields is None:
                metadata = {
                    key: value
                    for key, value in record.items()
                    if key not in {id_field, text_field, external_id_field}
                }
            else:
                metadata = {key: record[key] for key in metadata_fields if key in record}

            fields = {"text": str(record.get(text_field, ""))}
            if field_map:
                for field_name, source_key in field_map.items():
                    if source_key in record and record[source_key] is not None:
                        fields[field_name] = str(record[source_key])

            external_id = None
            if external_id_field and record.get(external_id_field) is not None:
                external_id = str(record[external_id_field])

            documents.append(
                Document(
                    doc_id=doc_id,
                    text=fields.get("text", ""),
                    external_id=external_id,
                    metadata=metadata,
                    fields=fields,
                )
            )
        return cls(documents)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        id_field: str = "id",
        text_field: str = "text",
        external_id_field: str | None = None,
        metadata_fields: Iterable[str] | None = None,
        field_map: dict[str, str] | None = None,
    ) -> DocumentCollection:
        """Load documents from a JSONL file."""
        file_path = Path(path)
        records: list[dict[str, Any]] = []
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return cls.from_records(
            records,
            id_field=id_field,
            text_field=text_field,
            external_id_field=external_id_field,
            metadata_fields=metadata_fields,
            field_map=field_map,
        )

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        *,
        split: str = "train",
        subset: str | None = None,
        revision: str | None = None,
        id_field: str = "id",
        text_field: str = "text",
        external_id_field: str | None = None,
        metadata_fields: Iterable[str] | None = None,
        field_map: dict[str, str] | None = None,
    ) -> DocumentCollection:
        """Load a collection from a HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset identifier (``"owner/dataset"``).
            split: dataset split (default ``"train"``).
            subset: dataset configuration / subset name.
            revision: **Strongly recommended.** Pin to a specific git revision
                (commit SHA, branch, or tag) on the HuggingFace dataset
                repository so reruns are reproducible and a malicious or
                accidental upstream change cannot silently alter what your
                code consumes. ``None`` (the default) follows the dataset's
                default branch — fine for ad-hoc exploration, *not* fine for
                production pipelines or anything reproducibility-sensitive.
                Bandit B615 flags unpinned downloads.
            id_field, text_field, external_id_field, metadata_fields, field_map:
                forwarded to :meth:`from_records`.
        """
        try:
            from datasets import load_dataset  # ty: ignore[unresolved-import]
        except ImportError as exc:
            msg = "datasets is required for from_huggingface()"
            raise ImportError(msg) from exc

        dataset = load_dataset(dataset_name, subset, split=split, revision=revision)
        return cls.from_records(
            dataset,
            id_field=id_field,
            text_field=text_field,
            external_id_field=external_id_field,
            metadata_fields=metadata_fields,
            field_map=field_map,
        )

    def build_index(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        field_weights: dict[str, int | float] | None = None,
    ) -> InvertedIndex:
        """Build an inverted index over the collection.

        Tokenization runs in Rust under a single ``py.detach`` (parallel
        via Rayon), with the supplied ``Tokenizer``'s config inherited
        — audit perf finding #2 / P5. The intermediate
        ``list[tuple[int, list[str]]]`` Python materialization that the
        old path produced (potentially GBs on large corpora) is gone.
        """
        active_tokenizer = tokenizer or Tokenizer(lowercase=True)
        doc_ids: list[int] = []
        texts: list[str] = []
        for document in self._documents:
            doc_ids.append(document.doc_id)
            texts.append(document.render_text(field_weights=field_weights))
        idx = InvertedIndex.__new__(InvertedIndex)
        idx._inner = _RustInvertedIndex.build_from_texts(
            doc_ids,
            texts,
            active_tokenizer._inner,
        )
        return idx

    def build_searcher(self, **kwargs: Any):
        """Create a Searcher over this collection."""
        from kaos_nlp_core.search import Searcher

        return Searcher.from_collection(self, **kwargs)


__all__ = [
    "Document",
    "DocumentCollection",
]
