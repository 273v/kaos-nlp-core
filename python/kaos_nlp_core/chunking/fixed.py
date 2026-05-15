"""Fixed-size token-window chunker.

Splits a text into windows of approximately ``max_tokens`` tokens, with
optional overlap. The chunker is deterministic and does **not** respect
sentence or paragraph boundaries — use :class:`SentenceChunker` or
:class:`ParagraphChunker` when you need clean breaks.

Token counting uses an approximate character-based heuristic by default
(``ceil(chars / 4)``); supply a precise tokenizer via the
``token_counter`` constructor argument when accuracy matters.
"""

from __future__ import annotations

from collections.abc import Callable

from kaos_nlp_core.chunking import Chunk
from kaos_nlp_core.chunking._pack import default_token_counter


class FixedTokenChunker:
    """Sliding character-window chunker scaled by approximate tokens.

    The chunker walks the source string with a window of
    ``window_chars`` characters and a stride of ``stride_chars``. The
    window is sized so that ``token_counter(window_text)`` is close to
    ``max_tokens``; the stride is sized so adjacent windows overlap by
    ``overlap_tokens`` worth of characters.

    Implementation note:
        The chunker is deterministic. Window/stride sizes are computed
        once from ``max_tokens``, ``overlap_tokens``, and the
        ``token_counter``'s observed characters-per-token ratio (or
        the ``chars_per_token`` override). This avoids re-counting
        tokens for every candidate cut, which can be expensive when
        ``token_counter`` is a real tokenizer.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        chars_per_token: float = 4.0,
        token_counter: Callable[[str], int] = default_token_counter,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        if overlap_tokens < 0:
            raise ValueError(f"overlap_tokens must be >= 0, got {overlap_tokens}")
        if overlap_tokens >= max_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})"
            )
        if chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be > 0, got {chars_per_token}")

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.chars_per_token = chars_per_token
        self.token_counter = token_counter

        self._window_chars = max(1, round(max_tokens * chars_per_token))
        overlap_chars = max(0, round(overlap_tokens * chars_per_token))
        self._stride_chars = max(1, self._window_chars - overlap_chars)

    def chunk(self, text: str, *, parent_id: str | None = None) -> list[Chunk]:
        if not text:
            return []
        chunks: list[Chunk] = []
        n = len(text)
        start = 0
        while start < n:
            end = min(start + self._window_chars, n)
            slice_text = text[start:end]
            chunks.append(
                Chunk(
                    text=slice_text,
                    start=start,
                    end=end,
                    parent_id=parent_id,
                    token_count=self.token_counter(slice_text),
                    metadata={
                        "chunker": "FixedTokenChunker",
                        "max_tokens": self.max_tokens,
                        "overlap_tokens": self.overlap_tokens,
                    },
                )
            )
            if end == n:
                break
            start += self._stride_chars
        return chunks


__all__ = ["FixedTokenChunker"]
