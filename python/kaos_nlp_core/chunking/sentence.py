"""Sentence-aware chunker.

Splits text on sentence boundaries via Punkt segmentation, then packs
whole sentences up to a token budget. A sentence is never broken
across chunks; an oversize sentence is emitted as its own chunk so
the caller can subdivide if desired.
"""

from __future__ import annotations

from collections.abc import Callable

from kaos_nlp_core.chunking import Chunk
from kaos_nlp_core.chunking._pack import _Unit, default_token_counter, pack_units
from kaos_nlp_core.segmentation import segment_sentences


class SentenceChunker:
    """Pack whole sentences up to a token budget.

    Args:
        max_tokens: Soft token-budget per chunk. Default ``512``.
        overlap_sentences: Number of trailing sentences to repeat at
            the start of each subsequent chunk for context preservation.
            Default ``0`` (no overlap).
        token_counter: Callable used to score chunk text. Default is
            an approximate ``chars/4`` heuristic; pass a precise
            tokenizer for provider-accurate sizing.

    The chunker accepts an empty string and returns ``[]``.
    Single-sentence inputs return one chunk. Tail whitespace between
    sentences (newlines, runs of spaces) is preserved in the chunk
    text because chunk offsets are derived from
    ``source[first_sentence.start : last_sentence.end]``.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 512,
        overlap_sentences: int = 0,
        token_counter: Callable[[str], int] = default_token_counter,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        if overlap_sentences < 0:
            raise ValueError(f"overlap_sentences must be >= 0, got {overlap_sentences}")
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.token_counter = token_counter

    def chunk(self, text: str, *, parent_id: str | None = None) -> list[Chunk]:
        if not text:
            return []
        segments = segment_sentences(text)
        if not segments:
            return []
        # Avoid per-unit metadata dict allocation. The earlier
        # ``metadata={"sentence_confidence": segment.confidence}``
        # caused 3x dict allocations per chunk (unit dict -> packer's
        # merged_metadata dict → ``Chunk.__post_init__``'s
        # ``MappingProxyType(dict(...))`` wrap) and showed up as a
        # 253 MB RSS delta on a 1000-USC-doc scale run. The
        # ``sentence_confidence`` value is never consumed downstream
        # (grep across all three repos confirms zero readers), and the
        # ``Segment.confidence`` field is still available on the
        # underlying Punkt output for any caller that needs it.
        units: list[_Unit] = [
            _Unit(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                token_count=self.token_counter(segment.text),
            )
            for segment in segments
        ]
        return pack_units(
            units,
            source=text,
            parent_id=parent_id,
            max_tokens=self.max_tokens,
            overlap_units=self.overlap_sentences,
            token_counter=self.token_counter,
            chunk_metadata={
                "chunker": "SentenceChunker",
                "max_tokens": self.max_tokens,
                "overlap_sentences": self.overlap_sentences,
            },
        )


__all__ = ["SentenceChunker"]
