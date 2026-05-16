"""Hierarchical chunker.

Emits chunks at multiple :attr:`Chunk.depth` levels. The default
configuration produces a three-level hierarchy:

- ``depth=0``: section-level chunks (one per detected section).
- ``depth=1``: paragraph-level chunks within each section, bounded by
  ``max_tokens``.
- ``depth=2``: sentence-level chunks for any paragraph that exceeds
  ``max_tokens``.

Callers can inspect ``chunk.depth`` to choose which level fits their
downstream task: depth-0 for a coarse table of contents, depth-1 for
embedding/retrieval, depth-2 for citation-anchored extraction.

The hierarchy is encoded *in the same flat list* via the ``depth``
field. Tree reconstruction is the caller's responsibility — keeping
the output flat avoids forcing a tree type on consumers that prefer
a sequence.
"""

from __future__ import annotations

from collections.abc import Callable

from kaos_nlp_core.chunking import Chunk
from kaos_nlp_core.chunking._pack import default_token_counter
from kaos_nlp_core.chunking.paragraph import ParagraphChunker
from kaos_nlp_core.chunking.section import SectionChunker, _detect_sections
from kaos_nlp_core.chunking.sentence import SentenceChunker


class HierarchicalChunker:
    """Produce chunks at multiple depths from one source text.

    Args:
        max_tokens: Soft token budget for the paragraph/sentence
            levels. Default ``1024``.
        token_counter: Callable used to score chunk text.
        section_chunker: Optional pre-configured section detector.
        paragraph_chunker: Optional pre-configured paragraph chunker
            for the ``depth=1`` level.
        sentence_chunker: Optional pre-configured sentence chunker
            for the ``depth=2`` fallback.

    Returned chunks:
        Listed in source order. Each chunk's ``depth`` field is
        ``0``, ``1``, or ``2``. Each chunk's ``metadata`` includes
        ``parent_section_index`` so consumers can rebuild the
        section → paragraph → sentence tree.

        ``depth=0`` chunks carry ``metadata["over_budget"]``
        indicating whether ``token_count > max_tokens`` for that
        section. Callers that want a coarse-only view (e.g., a
        table of contents) can filter for
        ``depth == 0 and not over_budget``.

        ``depth=2`` chunks are emitted only when sentence
        subdivision of the paragraph actually adds granularity
        (i.e., yields more than one chunk). When the underlying
        text cannot be split further — for example, a single
        oversize sentence — depth=2 is suppressed to avoid emitting
        a chunk identical to its depth=1 parent.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 1024,
        token_counter: Callable[[str], int] = default_token_counter,
        section_chunker: SectionChunker | None = None,
        paragraph_chunker: ParagraphChunker | None = None,
        sentence_chunker: SentenceChunker | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        self.max_tokens = max_tokens
        self.token_counter = token_counter
        self._section_chunker = section_chunker or SectionChunker(
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
        self._paragraph_chunker = paragraph_chunker or ParagraphChunker(
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
        self._sentence_chunker = sentence_chunker or SentenceChunker(
            max_tokens=max_tokens,
            token_counter=token_counter,
        )

    def chunk(self, text: str, *, parent_id: str | None = None) -> list[Chunk]:
        if not text:
            return []
        sections = _detect_sections(text)
        chunks: list[Chunk] = []
        for section_index, section in enumerate(sections):
            section_text = text[section.start : section.end]
            if not section_text.strip():
                continue
            # Depth-0: the section itself. ``over_budget`` flags sections
            # whose own token count exceeds ``max_tokens`` — useful for
            # callers that want to filter depth-0 down to coarse-toc-only
            # views without having to recompute the budget comparison.
            section_token_count = self.token_counter(section_text)
            chunks.append(
                Chunk(
                    text=section_text,
                    start=section.start,
                    end=section.end,
                    parent_id=parent_id,
                    token_count=section_token_count,
                    depth=0,
                    metadata={
                        "chunker": "HierarchicalChunker",
                        "section_index": section_index,
                        "section_kind": section.kind,
                        "section_depth": section.depth,
                        "section_heading": section.heading,
                        "level": "section",
                        "max_tokens": self.max_tokens,
                        "over_budget": section_token_count > self.max_tokens,
                    },
                )
            )
            # Depth-1: paragraphs within the section.
            paragraph_chunks = self._paragraph_chunker.chunk(section_text, parent_id=parent_id)
            for paragraph_chunk in paragraph_chunks:
                token_count = paragraph_chunk.token_count
                paragraph_meta = {
                    **dict(paragraph_chunk.metadata),
                    "chunker": "HierarchicalChunker",
                    "section_index": section_index,
                    "section_kind": section.kind,
                    "section_depth": section.depth,
                    "section_heading": section.heading,
                    "level": "paragraph",
                }
                paragraph_global = Chunk(
                    text=paragraph_chunk.text,
                    start=section.start + paragraph_chunk.start,
                    end=section.start + paragraph_chunk.end,
                    parent_id=parent_id,
                    token_count=token_count,
                    depth=1,
                    metadata=paragraph_meta,
                )
                chunks.append(paragraph_global)
                # Depth-2: sentence subdivision for any paragraph whose
                # token count is still above budget *and* where the
                # subdivision actually adds granularity. ParagraphChunker
                # already routes oversize paragraphs through
                # SentenceChunker, so a naïve re-run on the resulting
                # sub-chunk often returns one chunk identical to the
                # input — emitting that as depth=2 would just duplicate
                # depth=1. Require ``len(sentence_chunks) > 1`` to skip
                # the no-op case.
                if token_count > self.max_tokens:
                    sentence_chunks = self._sentence_chunker.chunk(
                        paragraph_chunk.text, parent_id=parent_id
                    )
                    if len(sentence_chunks) <= 1:
                        continue
                    for sentence_chunk in sentence_chunks:
                        chunks.append(
                            Chunk(
                                text=sentence_chunk.text,
                                start=section.start + paragraph_chunk.start + sentence_chunk.start,
                                end=section.start + paragraph_chunk.start + sentence_chunk.end,
                                parent_id=parent_id,
                                token_count=sentence_chunk.token_count,
                                depth=2,
                                metadata={
                                    **dict(sentence_chunk.metadata),
                                    "chunker": "HierarchicalChunker",
                                    "section_index": section_index,
                                    "section_kind": section.kind,
                                    "section_depth": section.depth,
                                    "section_heading": section.heading,
                                    "level": "sentence",
                                },
                            )
                        )
        return chunks


__all__ = ["HierarchicalChunker"]
