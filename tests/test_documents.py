"""Tests for document collections and corpus loaders."""

from __future__ import annotations

import json
from pathlib import Path

from kaos_nlp_core.documents import Document, DocumentCollection
from kaos_nlp_core.search import Searcher
from kaos_nlp_core.tokenizer import Tokenizer


class TestDocument:
    def test_render_text_with_field_weights(self) -> None:
        document = Document(
            doc_id=1,
            text="body text",
            fields={"title": "Important Title", "text": "body text"},
        )
        rendered = document.render_text(field_weights={"title": 2, "text": 1})
        assert rendered.count("Important Title") == 2
        assert "body text" in rendered


class TestDocumentCollection:
    def test_from_records(self) -> None:
        collection = DocumentCollection.from_records(
            [
                {"id": 10, "identifier": "doc-10", "title": "Alpha", "text": "cat dog"},
                {"id": 11, "identifier": "doc-11", "title": "Beta", "text": "bird fish"},
            ],
            external_id_field="identifier",
            field_map={"title": "title"},
        )
        assert len(collection) == 2
        doc = collection.get(10)
        assert doc is not None
        assert doc.external_id == "doc-10"
        assert doc.fields["title"] == "Alpha"

    def test_from_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "docs.jsonl"
        path.write_text(
            json.dumps({"id": 1, "text": "first"})
            + "\n"
            + json.dumps({"id": 2, "text": "second"})
            + "\n",
            encoding="utf-8",
        )
        collection = DocumentCollection.from_jsonl(path)
        assert len(collection) == 2
        doc = collection.get(2)
        assert doc is not None
        assert doc.text == "second"

    def test_build_index(self) -> None:
        collection = DocumentCollection.from_records(
            [{"id": 1, "text": "cat dog"}, {"id": 2, "text": "bird cat"}]
        )
        index = collection.build_index(tokenizer=Tokenizer(lowercase=True))
        results = index.query_bm25(["cat"], top_k=5)
        assert len(results) == 2

    def test_build_searcher(self) -> None:
        collection = DocumentCollection.from_records(
            [{"id": 1, "title": "Contract Guide", "text": "breach and remedies"}],
            field_map={"title": "title"},
        )
        searcher = collection.build_searcher(field_weights={"title": 2, "text": 1})
        results = searcher.search("contract")
        assert len(results) == 1
        assert results[0].doc_id == 1


class TestSearcherIntegration:
    def test_from_documents_with_field_weights(self) -> None:
        searcher = Searcher.from_documents(
            [
                {"id": 1, "title": "Employment Contract", "text": "termination and severance"},
                {"id": 2, "title": "Privacy Notice", "text": "cookies and tracking"},
            ],
            field_map={"title": "title"},
            field_weights={"title": 2, "text": 1},
        )
        results = searcher.search("employment")
        assert results[0].doc_id == 1
