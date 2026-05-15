"""Paragraph-aware chunker with sentence fallback.

Packs whole paragraphs up to a token budget. When a single paragraph
exceeds ``max_tokens``, it is subdivided via :class:`SentenceChunker`
so individual chunks stay near the budget while still preserving
paragraph boundaries everywhere else.
"""

from __future__ import annotations

from collections.abc import Callable

from kaos_nlp_core.chunking import Chunk
from kaos_nlp_core.chunking._pack import _Unit, default_token_counter, pack_units
from kaos_nlp_core.chunking.sentence import SentenceChunker
from kaos_nlp_core.segmentation import segment_paragraphs


class ParagraphChunker:
    """Pack whole paragraphs up to a token budget, falling back to sentences.

    Args:
        max_tokens: Soft token-budget per chunk. Default ``1024``.
        overlap_paragraphs: Number of trailing paragraphs to repeat at
            the start of each subsequent chunk. Default ``0``.
        token_counter: Callable used to score chunk text.
        sentence_chunker: Optional override for the fallback chunker
            used when a paragraph exceeds ``max_tokens``. If omitted,
            a :class:`SentenceChunker` configured to the same
            ``max_tokens`` and ``token_counter`` is constructed.

    Behavior on oversize paragraphs:
        Each oversize paragraph is fed to ``sentence_chunker.chunk``
        in isolation. The resulting sub-chunks are emitted in source
        order, preserving the original character offsets.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 1024,
        overlap_paragraphs: int = 0,
        token_counter: Callable[[str], int] = default_token_counter,
        sentence_chunker: SentenceChunker | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        if overlap_paragraphs < 0:
            raise ValueError(f"overlap_paragraphs must be >= 0, got {overlap_paragraphs}")
        self.max_tokens = max_tokens
        self.overlap_paragraphs = overlap_paragraphs
        self.token_counter = token_counter
        self._sentence_chunker = sentence_chunker or SentenceChunker(
            max_tokens=max_tokens,
            token_counter=token_counter,
        )

    def chunk(self, text: str, *, parent_id: str | None = None) -> list[Chunk]:
        if not text:
            return []
        paragraphs = segment_paragraphs(text)
        if not paragraphs:
            return []

        # Identify oversize paragraphs up front; they bypass the
        # packer and emit sub-chunks via the sentence chunker.
        normal_units: list[_Unit] = []
        oversize_indices: set[int] = set()
        for index, paragraph in enumerate(paragraphs):
            token_count = self.token_counter(paragraph.text)
            if token_count > self.max_tokens:
                oversize_indices.add(index)
            else:
                normal_units.append(
                    _Unit(
                        text=paragraph.text,
                        start=paragraph.start,
                        end=paragraph.end,
                        token_count=token_count,
                        metadata={"paragraph_index": index},
                    )
                )

        chunks: list[Chunk] = []
        normal_index = 0

        for index, paragraph in enumerate(paragraphs):
            if index in oversize_indices:
                # Flush any pending normal units that come before this
                # oversize paragraph.
                if normal_index < len(normal_units):
                    pending = []
                    while normal_index < len(normal_units):
                        nu = normal_units[normal_index]
                        if nu.start < paragraph.start:
                            pending.append(nu)
                            normal_index += 1
                        else:
                            break
                    if pending:
                        chunks.extend(
                            pack_units(
                                pending,
                                source=text,
                                parent_id=parent_id,
                                max_tokens=self.max_tokens,
                                overlap_units=self.overlap_paragraphs,
                                token_counter=self.token_counter,
                                chunk_metadata=self._chunk_metadata(),
                            )
                        )
                # Subdivide the oversize paragraph.
                sub = self._sentence_chunker.chunk(paragraph.text, parent_id=parent_id)
                for sub_chunk in sub:
                    chunks.append(
                        Chunk(
                            text=sub_chunk.text,
                            start=paragraph.start + sub_chunk.start,
                            end=paragraph.start + sub_chunk.end,
                            parent_id=parent_id,
                            token_count=sub_chunk.token_count,
                            depth=sub_chunk.depth,
                            metadata={
                                **dict(sub_chunk.metadata),
                                "chunker": "ParagraphChunker",
                                "paragraph_index": index,
                                "paragraph_subdivided": True,
                                "max_tokens": self.max_tokens,
                            },
                        )
                    )

        # Pack any remaining normal units after the last oversize
        # paragraph (or all of them if no paragraphs were oversize).
        remaining = normal_units[normal_index:]
        if remaining:
            chunks.extend(
                pack_units(
                    remaining,
                    source=text,
                    parent_id=parent_id,
                    max_tokens=self.max_tokens,
                    overlap_units=self.overlap_paragraphs,
                    token_counter=self.token_counter,
                    chunk_metadata=self._chunk_metadata(),
                )
            )

        return sorted(chunks, key=lambda c: c.start)

    def _chunk_metadata(self) -> dict[str, object]:
        return {
            "chunker": "ParagraphChunker",
            "max_tokens": self.max_tokens,
            "overlap_paragraphs": self.overlap_paragraphs,
        }


__all__ = ["ParagraphChunker"]
