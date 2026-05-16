"""Tests for the concrete chunkers in :mod:`kaos_nlp_core.chunking`.

Round-trip offset invariants and determinism are property-checked
across all chunkers in :class:`TestChunkerInvariants` so adding a new
chunker only requires registering it there.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest

from kaos_nlp_core.chunking import (
    Chunk,
    Chunker,
    FixedTokenChunker,
    HierarchicalChunker,
    ParagraphChunker,
    SectionChunker,
    SentenceChunker,
    default_token_counter,
    validate_chunk_offsets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHORT_TEXT = "Hello world. This is a small test."

LONG_PROSE = (
    "This is the first paragraph. It contains two sentences.\n"
    "\n"
    "Second paragraph here. With another two sentences inside.\n"
    "\n"
    "Third paragraph follows. It is the last paragraph in this text.\n"
)

LEGAL_TEXT = (
    "Introductory matter before the first section.\n"
    "\n"
    "1. Definitions.\n"
    'For purposes of this Agreement, "Service" means any of the products.\n'
    "Additional definitional content goes here.\n"
    "\n"
    "2. Term.\n"
    "This Agreement commences on the Effective Date.\n"
    "It continues until terminated.\n"
    "\n"
    "3. Confidentiality.\n"
    "Each party shall maintain in confidence all Confidential Information.\n"
)

UNICODE_TEXT = (
    "Café résumé naïve. 测试中文分句。 これは日本語のテストです。\n"
    "\n"
    "Emoji paragraph 🎉🎊🌍. More 🚀 content here.\n"
)


# ---------------------------------------------------------------------------
# Cross-chunker invariants
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        ("FixedTokenChunker", lambda: FixedTokenChunker(max_tokens=20, overlap_tokens=0)),
        (
            "FixedTokenChunkerOverlap",
            lambda: FixedTokenChunker(max_tokens=20, overlap_tokens=4),
        ),
        ("SentenceChunker", lambda: SentenceChunker(max_tokens=20)),
        (
            "SentenceChunkerOverlap",
            lambda: SentenceChunker(max_tokens=30, overlap_sentences=1),
        ),
        ("ParagraphChunker", lambda: ParagraphChunker(max_tokens=50)),
        (
            "ParagraphChunkerLarge",
            lambda: ParagraphChunker(max_tokens=10_000),
        ),
        ("SectionChunker", lambda: SectionChunker(max_tokens=200)),
        ("HierarchicalChunker", lambda: HierarchicalChunker(max_tokens=200)),
    ],
    ids=lambda p: p[0],
)
def any_chunker(request: pytest.FixtureRequest) -> Chunker:
    _name, factory = request.param
    return factory()


@pytest.fixture(
    params=[
        ("short", SHORT_TEXT),
        ("long_prose", LONG_PROSE),
        ("legal", LEGAL_TEXT),
        ("unicode", UNICODE_TEXT),
    ],
    ids=lambda p: p[0],
)
def any_text(request: pytest.FixtureRequest) -> str:
    _name, text = request.param
    return text


class TestChunkerInvariants:
    def test_empty_text_returns_empty(self, any_chunker: Chunker) -> None:
        assert any_chunker.chunk("") == []

    def test_round_trip_offsets(self, any_chunker: Chunker, any_text: str) -> None:
        chunks = any_chunker.chunk(any_text, parent_id="doc-1")
        for chunk in chunks:
            assert validate_chunk_offsets(any_text, chunk), (
                f"offset mismatch in {type(any_chunker).__name__}: "
                f"start={chunk.start} end={chunk.end} "
                f"slice={any_text[chunk.start : chunk.end]!r} "
                f"text={chunk.text!r}"
            )

    def test_parent_id_propagates(self, any_chunker: Chunker, any_text: str) -> None:
        chunks = any_chunker.chunk(any_text, parent_id="doc-1")
        assert all(chunk.parent_id == "doc-1" for chunk in chunks)

    def test_chunks_are_ordered_by_start(self, any_chunker: Chunker, any_text: str) -> None:
        chunks = any_chunker.chunk(any_text, parent_id="doc-1")
        starts = [chunk.start for chunk in chunks]
        assert starts == sorted(starts)

    def test_deterministic(self, any_chunker: Chunker, any_text: str) -> None:
        first = any_chunker.chunk(any_text, parent_id="doc-1")
        second = any_chunker.chunk(any_text, parent_id="doc-1")
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_chunks_implement_protocol(self, any_chunker: Chunker, any_text: str) -> None:
        chunks = any_chunker.chunk(any_text, parent_id="doc-1")
        for chunk in chunks:
            assert isinstance(chunk, Chunk)


# ---------------------------------------------------------------------------
# default_token_counter
# ---------------------------------------------------------------------------


class TestDefaultTokenCounter:
    def test_empty_string_is_zero(self) -> None:
        assert default_token_counter("") == 0

    def test_one_char(self) -> None:
        assert default_token_counter("a") == 1

    def test_four_chars(self) -> None:
        assert default_token_counter("abcd") == 1

    def test_five_chars_ceils(self) -> None:
        assert default_token_counter("abcde") == 2

    def test_long_string(self) -> None:
        # 100 chars → 25 tokens
        assert default_token_counter("x" * 100) == 25


# ---------------------------------------------------------------------------
# FixedTokenChunker
# ---------------------------------------------------------------------------


class TestFixedTokenChunker:
    def test_basic_split(self) -> None:
        text = "a" * 100  # ~25 tokens with the default counter
        chunker = FixedTokenChunker(max_tokens=10)
        chunks = chunker.chunk(text, parent_id="doc-1")
        # Window ≈ 40 chars; expect 3 chunks: 0-40, 40-80, 80-100.
        assert len(chunks) == 3
        assert chunks[0].start == 0
        assert chunks[-1].end == 100

    def test_overlap_produces_overlapping_windows(self) -> None:
        text = "a" * 200
        chunker = FixedTokenChunker(max_tokens=10, overlap_tokens=4)
        chunks = chunker.chunk(text, parent_id="doc-1")
        # Adjacent chunks must overlap.
        for prev, curr in itertools.pairwise(chunks):
            assert curr.start < prev.end, "adjacent windows should overlap"

    def test_zero_overlap_is_contiguous(self) -> None:
        text = "a" * 100
        chunker = FixedTokenChunker(max_tokens=5, overlap_tokens=0)
        chunks = chunker.chunk(text, parent_id="doc-1")
        for prev, curr in itertools.pairwise(chunks):
            assert curr.start == prev.end

    def test_short_text_fits_in_single_chunk(self) -> None:
        chunker = FixedTokenChunker(max_tokens=1000)
        chunks = chunker.chunk("hello world", parent_id="doc-1")
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"

    def test_rejects_invalid_max_tokens(self) -> None:
        with pytest.raises(ValueError, match=r"max_tokens must be > 0"):
            FixedTokenChunker(max_tokens=0)

    def test_rejects_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match=r"overlap_tokens must be >= 0"):
            FixedTokenChunker(max_tokens=10, overlap_tokens=-1)

    def test_rejects_overlap_geq_max(self) -> None:
        with pytest.raises(ValueError, match=r"overlap_tokens .* must be <"):
            FixedTokenChunker(max_tokens=10, overlap_tokens=10)

    def test_metadata_includes_chunker_name(self) -> None:
        chunker = FixedTokenChunker(max_tokens=20)
        chunks = chunker.chunk("hello world", parent_id="doc-1")
        assert chunks[0].metadata["chunker"] == "FixedTokenChunker"


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------


class TestSentenceChunker:
    def test_single_sentence(self) -> None:
        chunker = SentenceChunker(max_tokens=100)
        chunks = chunker.chunk("Just one sentence here.", parent_id="doc-1")
        assert len(chunks) == 1
        assert chunks[0].text == "Just one sentence here."

    def test_packs_multiple_sentences(self) -> None:
        chunker = SentenceChunker(max_tokens=1000)
        chunks = chunker.chunk(SHORT_TEXT, parent_id="doc-1")
        # Two sentences should fit in one chunk.
        assert len(chunks) == 1

    def test_respects_sentence_boundaries(self) -> None:
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunker = SentenceChunker(max_tokens=5)  # tiny budget
        chunks = chunker.chunk(text, parent_id="doc-1")
        for chunk in chunks:
            # Each chunk ends with sentence-final punctuation (".", "!", "?")
            assert chunk.text.rstrip().endswith((".", "!", "?"))

    def test_oversize_sentence_emits_alone(self) -> None:
        # A single very long sentence has to be its own chunk because
        # the chunker never splits a sentence.
        long_sentence = "word " * 200 + "end."
        chunker = SentenceChunker(max_tokens=10)
        chunks = chunker.chunk(long_sentence, parent_id="doc-1")
        assert len(chunks) == 1
        assert chunks[0].text == long_sentence.strip()

    def test_overlap_repeats_sentences(self) -> None:
        text = "S1. S2. S3. S4. S5. S6. S7. S8. S9. S10."
        chunker = SentenceChunker(max_tokens=4, overlap_sentences=1)
        chunks = chunker.chunk(text, parent_id="doc-1")
        # Adjacent chunks should share at least one sentence's worth.
        for prev, curr in itertools.pairwise(chunks):
            assert curr.start <= prev.end

    def test_metadata(self) -> None:
        chunker = SentenceChunker(max_tokens=100)
        chunks = chunker.chunk("Hello there.", parent_id="doc-1")
        assert chunks[0].metadata["chunker"] == "SentenceChunker"


# ---------------------------------------------------------------------------
# ParagraphChunker
# ---------------------------------------------------------------------------


class TestParagraphChunker:
    def test_packs_paragraphs(self) -> None:
        chunker = ParagraphChunker(max_tokens=10_000)
        chunks = chunker.chunk(LONG_PROSE, parent_id="doc-1")
        # With a huge budget, all paragraphs fit in one chunk.
        assert len(chunks) == 1

    def test_splits_when_oversize(self) -> None:
        chunker = ParagraphChunker(max_tokens=20)
        chunks = chunker.chunk(LONG_PROSE, parent_id="doc-1")
        # Many small chunks expected.
        assert len(chunks) >= 3

    def test_oversize_paragraph_falls_back_to_sentences(self) -> None:
        # One huge paragraph, plus one normal one.
        huge_paragraph = ". ".join(["This is sentence " + str(i) for i in range(50)]) + "."
        text = huge_paragraph + "\n\nNormal paragraph here.\n"
        chunker = ParagraphChunker(max_tokens=20)
        chunks = chunker.chunk(text, parent_id="doc-1")
        # The huge paragraph should produce multiple chunks (one per
        # sentence batch).
        markers = [c for c in chunks if c.metadata.get("paragraph_subdivided")]
        assert markers, "expected subdivided sub-chunks for oversize paragraph"

    def test_chunks_are_ordered(self) -> None:
        chunker = ParagraphChunker(max_tokens=20)
        chunks = chunker.chunk(LONG_PROSE, parent_id="doc-1")
        starts = [c.start for c in chunks]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# SectionChunker
# ---------------------------------------------------------------------------


class TestSectionChunker:
    def test_detects_numbered_sections(self) -> None:
        chunker = SectionChunker(max_tokens=1000)
        chunks = chunker.chunk(LEGAL_TEXT, parent_id="doc-1")
        section_indices = sorted({c.metadata["section_index"] for c in chunks})
        # We expect at least one implicit prologue + three numbered sections.
        assert len(section_indices) >= 3

    def test_metadata_carries_section_kind(self) -> None:
        chunker = SectionChunker(max_tokens=1000)
        chunks = chunker.chunk(LEGAL_TEXT, parent_id="doc-1")
        kinds = {c.metadata["section_kind"] for c in chunks}
        # At least one chunk should have ``decimal`` kind because the
        # legal text uses ``1.``/``2.`` numbering.
        assert "decimal" in kinds

    def test_no_sections_emits_single_implicit_section(self) -> None:
        chunker = SectionChunker(max_tokens=1000)
        chunks = chunker.chunk(SHORT_TEXT, parent_id="doc-1")
        assert len(chunks) >= 1
        assert all(c.metadata.get("section_kind") == "" for c in chunks)

    def test_section_chunks_respect_max_tokens(self) -> None:
        chunker = SectionChunker(max_tokens=8)
        chunks = chunker.chunk(LEGAL_TEXT, parent_id="doc-1")
        # The chunker may exceed the budget when a single sentence is
        # oversize, but the median chunk should be near the budget.
        small_chunks = [c for c in chunks if c.token_count <= 8 * 2]
        assert len(small_chunks) >= len(chunks) // 2


# ---------------------------------------------------------------------------
# HierarchicalChunker
# ---------------------------------------------------------------------------


class TestHierarchicalChunker:
    def test_emits_depths(self) -> None:
        chunker = HierarchicalChunker(max_tokens=200)
        chunks = chunker.chunk(LEGAL_TEXT, parent_id="doc-1")
        depths = {c.depth for c in chunks}
        # Section + paragraph levels are mandatory.
        assert {0, 1} <= depths

    def test_section_chunks_at_depth_zero(self) -> None:
        chunker = HierarchicalChunker(max_tokens=200)
        chunks = chunker.chunk(LEGAL_TEXT, parent_id="doc-1")
        section_chunks = [c for c in chunks if c.depth == 0]
        # Each section chunk should carry a section_kind.
        for chunk in section_chunks:
            assert chunk.metadata["level"] == "section"

    def test_paragraph_chunks_at_depth_one(self) -> None:
        chunker = HierarchicalChunker(max_tokens=200)
        chunks = chunker.chunk(LEGAL_TEXT, parent_id="doc-1")
        paragraph_chunks = [c for c in chunks if c.depth == 1]
        for chunk in paragraph_chunks:
            assert chunk.metadata["level"] == "paragraph"

    def test_sentence_subdivision_when_paragraph_oversize(self) -> None:
        # Depth-2 sentence subdivision should fire when the injected
        # SentenceChunker has a tighter budget than the surrounding
        # ParagraphChunker — that is the case where re-splitting actually
        # adds granularity beyond depth-1.
        big_paragraph = ". ".join("S" + str(i) for i in range(50)) + "."
        text = big_paragraph + "\n\nShort.\n"
        chunker = HierarchicalChunker(
            max_tokens=10,
            sentence_chunker=SentenceChunker(max_tokens=4),
        )
        chunks = chunker.chunk(text, parent_id="doc-1")
        sentence_chunks = [c for c in chunks if c.depth == 2]
        assert sentence_chunks, "expected sentence-level chunks when paragraph oversize"
        for chunk in sentence_chunks:
            assert chunk.metadata["level"] == "sentence"

    def test_depth_two_suppressed_when_sentence_split_is_noop(self) -> None:
        # When the injected SentenceChunker has the same budget as the
        # ParagraphChunker (the default), re-splitting a sub-chunk just
        # returns the same single chunk back. The HierarchicalChunker
        # should suppress depth-2 emission in that case so callers do
        # not see duplicate depth=1 / depth=2 pairs.
        big_paragraph = ". ".join("S" + str(i) for i in range(50)) + "."
        text = big_paragraph + "\n\nShort.\n"
        chunker = HierarchicalChunker(max_tokens=10)
        chunks = chunker.chunk(text, parent_id="doc-1")
        depth_two = [c for c in chunks if c.depth == 2]
        assert depth_two == [], "depth-2 must be suppressed when sentence-split returns one chunk"

    def test_depth_zero_over_budget_flag(self) -> None:
        # A section that exceeds max_tokens must be flagged as
        # over_budget on its depth-0 chunk so callers can filter coarse
        # views down to TOC-sized entries.
        big_paragraph = ". ".join("S" + str(i) for i in range(50)) + "."
        chunker = HierarchicalChunker(max_tokens=10)
        chunks = chunker.chunk(big_paragraph, parent_id="doc-1")
        depth_zero = [c for c in chunks if c.depth == 0]
        assert depth_zero, "expected at least one depth-0 section chunk"
        assert all(c.metadata.get("over_budget") is True for c in depth_zero), (
            "all depth-0 chunks for an oversize section must be flagged"
        )
        assert all(c.metadata.get("max_tokens") == 10 for c in depth_zero)

    def test_depth_zero_within_budget_flag(self) -> None:
        # A section that fits inside max_tokens must report
        # over_budget=False on its depth-0 chunk.
        chunker = HierarchicalChunker(max_tokens=1024)
        chunks = chunker.chunk("Short section.\n", parent_id="doc-1")
        depth_zero = [c for c in chunks if c.depth == 0]
        assert depth_zero, "expected at least one depth-0 chunk"
        assert all(c.metadata.get("over_budget") is False for c in depth_zero)

    def test_round_trip_at_every_depth(self) -> None:
        chunker = HierarchicalChunker(max_tokens=200)
        chunks = chunker.chunk(LEGAL_TEXT, parent_id="doc-1")
        for chunk in chunks:
            assert validate_chunk_offsets(LEGAL_TEXT, chunk)


# ---------------------------------------------------------------------------
# Custom token counter
# ---------------------------------------------------------------------------


def _word_counter(text: str) -> int:
    return len(text.split())


class TestCustomTokenCounter:
    @pytest.mark.parametrize(
        "chunker_factory",
        [
            lambda c: FixedTokenChunker(max_tokens=5, token_counter=c),
            lambda c: SentenceChunker(max_tokens=5, token_counter=c),
            lambda c: ParagraphChunker(max_tokens=5, token_counter=c),
        ],
    )
    def test_word_counter_accepted(
        self,
        chunker_factory: Callable[[Callable[[str], int]], Chunker],
    ) -> None:
        chunker = chunker_factory(_word_counter)
        chunks = chunker.chunk(LONG_PROSE, parent_id="doc-1")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert validate_chunk_offsets(LONG_PROSE, chunk)
