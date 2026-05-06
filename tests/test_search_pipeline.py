"""Tests for the high-level Python search pipeline."""

from __future__ import annotations

from kaos_nlp_core.lexicon import Lexicon
from kaos_nlp_core.search import (
    Searcher,
    SearchHit,
    SegmentHit,
    search_paragraphs,
    search_sentences,
)


def build_test_lexicon() -> Lexicon:
    lexicon = Lexicon()
    lexicon.add_entry(
        {
            "word": "contract",
            "all_synonyms": ["agreement"],
            "all_inflections": ["contracts"],
        }
    )
    return lexicon


class TestSearcher:
    def test_basic_bm25_search(self) -> None:
        searcher = Searcher.from_documents(
            [
                {"id": 1, "text": "contract breach remedies"},
                {"id": 2, "text": "privacy cookies tracker"},
            ]
        )
        results = searcher.search("breach")
        assert isinstance(results[0], SearchHit)
        assert results[0].doc_id == 1
        assert "contract breach remedies" in results[0].text

    def test_query_expansion(self) -> None:
        searcher = Searcher.from_documents(
            [{"id": 1, "text": "agreement reached by both parties"}],
            lexicon=build_test_lexicon(),
        )
        debug = searcher.search_debug("contract")
        assert "agreement" in debug.expanded_terms
        assert debug.results[0].doc_id == 1

    def test_search_batch(self) -> None:
        searcher = Searcher.from_documents(
            [
                {"id": 1, "text": "contract breach remedies"},
                {"id": 2, "text": "privacy cookies tracker"},
            ]
        )
        batch = searcher.search_batch(["contract", "privacy"])
        assert len(batch) == 2
        assert batch[0][0].doc_id == 1
        assert batch[1][0].doc_id == 2

    def test_tfidf_search(self) -> None:
        searcher = Searcher.from_documents(
            [
                {"id": 1, "text": "contract breach remedies"},
                {"id": 2, "text": "privacy cookies tracker"},
            ],
            scoring="tfidf",
        )
        results = searcher.search("privacy")
        assert results[0].doc_id == 2

    def test_external_id_roundtrip(self) -> None:
        searcher = Searcher.from_documents(
            [
                {"id": 1, "text": "contract breach remedies", "ref": "#/body/0"},
                {"id": 2, "text": "privacy cookies tracker", "ref": "#/body/1"},
            ],
            external_id_field="ref",
        )
        results = searcher.search("breach")
        assert results[0].external_id == "#/body/0"

    def test_metadata_roundtrip(self) -> None:
        searcher = Searcher.from_documents(
            [
                {"id": 1, "text": "contract breach", "page": 3, "section": "intro"},
            ],
            metadata_fields=["page", "section"],
        )
        results = searcher.search("breach")
        assert results[0].metadata["page"] == 3
        assert results[0].metadata["section"] == "intro"


class TestSegmentSearchDefaults:
    def test_sentence_search_uses_default_model(self) -> None:
        text = "See 42 U.S.C. § 1983. This is important."
        results = search_sentences(text, "important")
        assert len(results) >= 1
        assert isinstance(results[0], SegmentHit)
        assert results[0].text.endswith("important.")

    def test_paragraph_search_uses_default_model(self) -> None:
        text = "Dr. Smith reviewed the contract.\n\nThe agreement was signed."
        results = search_paragraphs(text, "agreement")
        assert len(results) == 1
        assert isinstance(results[0], SegmentHit)
        assert "agreement" in results[0].text.lower()

    def test_segment_hit_offsets(self) -> None:
        text = "First sentence here. Second one here."
        results = search_sentences(text, "second")
        assert results[0].start >= 0
        assert results[0].end <= len(text)
        assert results[0].end > results[0].start
