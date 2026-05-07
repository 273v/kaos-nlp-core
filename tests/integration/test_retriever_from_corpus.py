"""Unit tests for Retriever.from_corpus() class methods.

Tests that BM25Retriever and HybridRetriever correctly build from
a kaos-ml-core Corpus with full provenance threading.

Requires kaos-ml-core and kaos-content as dev dependencies. At
kaos-nlp-core v0.1.0a1 those siblings are not yet on PyPI, so this
file skips cleanly via importorskip — re-included for execution at
0.1.0a2 once the siblings ship.
"""

from __future__ import annotations

import asyncio

import pytest

# Skip the entire module at collection time when the optional siblings
# are missing. importorskip raises pytest.skip — collection succeeds,
# all tests in the file are reported as skipped instead of erroring.
pytest.importorskip("kaos_content")
pytest.importorskip("kaos_ml_core")


def _make_corpus():
    """Build a small multi-document Corpus for testing."""
    from kaos_content.builders import DocumentBuilder
    from kaos_content.model.attr import SourceRef
    from kaos_content.model.metadata import DocumentMetadata
    from kaos_ml_core import Corpus

    def make_doc(paragraphs: list[str], uri: str):
        builder = DocumentBuilder()
        for p in paragraphs:
            builder.paragraph(p)
        doc = builder.build()
        return doc.model_copy(update={"metadata": DocumentMetadata(source=SourceRef(uri=uri))})

    doc1 = make_doc(
        [
            "The filing fee is $89 for a Delaware certificate of incorporation.",
            "The certificate must include the corporation's name, "
            "registered office, and registered agent.",
        ],
        "doc:delaware-gcl",
    )
    doc2 = make_doc(
        [
            "Rule 10b-5 prohibits fraud in connection with the purchase or sale of any security.",
            "It is unlawful to employ any device, scheme, or artifice to defraud.",
            "Making an untrue statement of a material fact is also prohibited.",
        ],
        "doc:sec-10b5",
    )

    return Corpus.from_documents([doc1, doc2], level="paragraph")


@pytest.fixture(scope="module")
def corpus():
    return _make_corpus()


class TestBM25RetrieverFromCorpus:
    def test_builds_successfully(self, corpus) -> None:
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever.from_corpus(corpus)
        assert retriever is not None
        assert retriever.searcher is not None

    def test_retrieves_relevant_result(self, corpus) -> None:
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever.from_corpus(corpus)
        hits = asyncio.run(retriever.retrieve("filing fee certificate", top_k=3))
        assert len(hits) >= 1
        # Top hit should be from the Delaware document
        assert hits[0].external_id is not None
        assert "delaware" in hits[0].external_id

    def test_passage_uri_includes_block_ref(self, corpus) -> None:
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever.from_corpus(corpus)
        hits = asyncio.run(retriever.retrieve("filing fee", top_k=1))
        # external_id should be doc_uri + block_ref
        assert hits[0].external_id is not None
        assert "#/body/" in hits[0].external_id

    def test_metadata_carries_provenance(self, corpus) -> None:
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever.from_corpus(corpus)
        hits = asyncio.run(retriever.retrieve("filing fee", top_k=1))
        meta = hits[0].metadata
        assert "doc_uri" in meta
        assert meta["doc_uri"] == "doc:delaware-gcl"

    def test_corpus_wide_idf(self, corpus) -> None:
        """Results from both documents should be ranked together."""
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever.from_corpus(corpus)
        hits = asyncio.run(retriever.retrieve("prohibited unlawful", top_k=5))
        # Should find results from the SEC document
        sec_hits = [h for h in hits if "sec-10b5" in (h.external_id or "")]
        assert len(sec_hits) >= 1

    def test_all_corpus_units_indexed(self, corpus) -> None:
        """Every unit in the corpus should be findable."""
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever.from_corpus(corpus)
        # The corpus has 5 units (2 + 3 paragraphs)
        assert len(corpus) == 5
        # Searching for a unique term from each document should find it
        for query, expected_uri in [
            ("filing fee", "delaware"),
            ("artifice defraud", "10b5"),
        ]:
            hits = asyncio.run(retriever.retrieve(query, top_k=1))
            assert any(expected_uri in (h.external_id or "") for h in hits), (
                f"Query {query!r} didn't find {expected_uri}"
            )


class TestEmbeddingRetrieverFromCorpus:
    """Test EmbeddingRetriever.from_corpus if kaos-nlp-transformers is available."""

    def test_builds_successfully(self, corpus) -> None:
        try:
            from kaos_nlp_transformers.retrieval import EmbeddingRetriever
        except ImportError:
            pytest.skip("kaos-nlp-transformers not installed")

        retriever = EmbeddingRetriever.from_corpus(corpus)
        assert retriever is not None

    def test_semantic_retrieval(self, corpus) -> None:
        try:
            from kaos_nlp_transformers.retrieval import EmbeddingRetriever
        except ImportError:
            pytest.skip("kaos-nlp-transformers not installed")

        retriever = EmbeddingRetriever.from_corpus(corpus)
        hits = asyncio.run(retriever.retrieve("securities fraud prohibition", top_k=3))
        # Top hit should be from the SEC document (semantic match)
        top_ext = hits[0].metadata.get("doc_id", "")
        assert "sec-10b5" in top_ext, f"Expected sec-10b5, got {top_ext}"

    def test_passage_uri_in_metadata(self, corpus) -> None:
        try:
            from kaos_nlp_transformers.retrieval import EmbeddingRetriever
        except ImportError:
            pytest.skip("kaos-nlp-transformers not installed")

        retriever = EmbeddingRetriever.from_corpus(corpus)
        hits = asyncio.run(retriever.retrieve("filing fee", top_k=1))
        doc_id = hits[0].metadata.get("doc_id", "")
        assert "#/body/" in doc_id


def _make_sectioned_corpus():
    """Build a multi-document Corpus where paragraphs have section_refs.

    Structure:
      doc1 ("doc:contract-law"):
        Heading "Formation"        -> 2 paragraphs (share section_ref)
        Heading "Breach"           -> 1 paragraph
      doc2 ("doc:torts"):
        Heading "Negligence"       -> 2 paragraphs (share section_ref)
        Heading "Strict Liability" -> 1 paragraph
    Total: 6 paragraphs, 4 sections.
    """
    from kaos_content.builders import DocumentBuilder
    from kaos_content.model.attr import SourceRef
    from kaos_content.model.metadata import DocumentMetadata
    from kaos_ml_core import Corpus

    def make_doc(sections: list[tuple[str, list[str]]], uri: str):
        builder = DocumentBuilder()
        for heading_text, paragraphs in sections:
            builder.heading(1, heading_text)
            for p in paragraphs:
                builder.paragraph(p)
        doc = builder.build()
        return doc.model_copy(update={"metadata": DocumentMetadata(source=SourceRef(uri=uri))})

    doc1 = make_doc(
        [
            (
                "Formation",
                [
                    "A contract requires an offer, acceptance, and consideration.",
                    "The offeror must manifest willingness to enter into a bargain.",
                ],
            ),
            (
                "Breach",
                [
                    "A material breach excuses the non-breaching party from performance.",
                ],
            ),
        ],
        "doc:contract-law",
    )
    doc2 = make_doc(
        [
            (
                "Negligence",
                [
                    "Negligence requires duty, breach, causation, and damages.",
                    "The reasonable person standard determines the duty of care.",
                ],
            ),
            (
                "Strict Liability",
                [
                    "Strict liability applies to abnormally dangerous activities.",
                ],
            ),
        ],
        "doc:torts",
    )
    return Corpus.from_documents([doc1, doc2], level="paragraph")


@pytest.fixture(scope="module")
def sectioned_corpus():
    return _make_sectioned_corpus()


class TestBM25RetrieverGroupBy:
    """Tests for BM25Retriever.from_corpus(group_by=...)."""

    def test_grouped_builds_successfully(self, sectioned_corpus) -> None:
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever.from_corpus(sectioned_corpus, group_by="section_ref")
        assert retriever is not None

    def test_grouped_fewer_documents_than_ungrouped(self, sectioned_corpus) -> None:
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        ungrouped = BM25Retriever.from_corpus(sectioned_corpus)
        grouped = BM25Retriever.from_corpus(sectioned_corpus, group_by="section_ref")
        # 6 paragraphs ungrouped, 4 sections grouped
        assert ungrouped.searcher.collection is not None
        assert grouped.searcher.collection is not None
        ungrouped_count = len(ungrouped.searcher.collection)
        grouped_count = len(grouped.searcher.collection)
        assert ungrouped_count == 6
        assert grouped_count == 4
        assert grouped_count < ungrouped_count

    def test_grouped_external_id_uses_section_ref(self, sectioned_corpus) -> None:
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever.from_corpus(sectioned_corpus, group_by="section_ref")
        hits = asyncio.run(retriever.retrieve("offer acceptance consideration", top_k=1))
        assert len(hits) >= 1
        ext_id = hits[0].external_id
        # Should be doc_uri#section_ref
        assert ext_id is not None
        assert "doc:contract-law#" in ext_id
        assert "#/body/" in ext_id  # section_ref is a block_ref like #/body/0

    def test_grouped_text_is_concatenated(self, sectioned_corpus) -> None:
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever.from_corpus(sectioned_corpus, group_by="section_ref")
        hits = asyncio.run(retriever.retrieve("offer acceptance bargain willingness", top_k=1))
        assert len(hits) >= 1
        # The grouped text should contain both paragraphs from Formation
        assert "offer" in hits[0].text
        assert "willingness" in hits[0].text

    def test_ungrouped_behavior_unchanged(self, sectioned_corpus) -> None:
        """group_by=None should not change existing behavior."""
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever.from_corpus(sectioned_corpus)
        hits = asyncio.run(retriever.retrieve("offer acceptance", top_k=1))
        assert len(hits) >= 1
        # Should be a single paragraph, not concatenated
        assert "willingness" not in hits[0].text or "offer" not in hits[0].text


class TestEmbeddingRetrieverGroupBy:
    """Test EmbeddingRetriever.from_corpus(group_by=...)."""

    def test_grouped_builds_successfully(self, sectioned_corpus) -> None:
        try:
            from kaos_nlp_transformers.retrieval import EmbeddingRetriever
        except ImportError:
            pytest.skip("kaos-nlp-transformers not installed")

        retriever = EmbeddingRetriever.from_corpus(sectioned_corpus, group_by="section_ref")
        assert retriever is not None
        # Should have 4 sections, not 6 paragraphs
        assert retriever.num_documents == 4

    def test_grouped_semantic_retrieval(self, sectioned_corpus) -> None:
        try:
            from kaos_nlp_transformers.retrieval import EmbeddingRetriever
        except ImportError:
            pytest.skip("kaos-nlp-transformers not installed")

        retriever = EmbeddingRetriever.from_corpus(sectioned_corpus, group_by="section_ref")
        hits = asyncio.run(retriever.retrieve("forming a contract", top_k=1))
        assert len(hits) >= 1
        assert "contract-law" in (hits[0].external_id or "")


class TestHybridRetrieverFromCorpus:
    """Test HybridRetriever.from_corpus."""

    def test_builds_successfully(self, corpus) -> None:
        try:
            from kaos_nlp_core.retrieval.hybrid import HybridRetriever
        except ImportError:
            pytest.skip("Missing dependencies")

        try:
            retriever = HybridRetriever.from_corpus(corpus)
        except ImportError:
            pytest.skip("kaos-nlp-transformers not installed")
        assert retriever is not None

    def test_hybrid_retrieval(self, corpus) -> None:
        try:
            from kaos_nlp_core.retrieval.hybrid import HybridRetriever

            retriever = HybridRetriever.from_corpus(corpus)
        except ImportError:
            pytest.skip("kaos-nlp-transformers not installed")

        hits = asyncio.run(retriever.retrieve("filing fee certificate", top_k=3))
        assert len(hits) >= 1
        # Should find the Delaware doc
        assert any("delaware" in (h.external_id or "") for h in hits)
