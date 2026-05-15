"""Document chunking primitives.

A *chunk* is a contiguous slice of source text packaged for downstream
processing (search, summarization, classification, embedding). Chunks
preserve character offsets back to the source so every result that
flows through the stack can be attributed back to a verifiable region
of the original document.

This module defines the foundation types only:

- :class:`Chunk` — frozen value object with text, offsets, and a
  deterministic identifier.
- :class:`Chunker` — protocol for callables that split text into
  :class:`Chunk` instances.

Concrete :class:`Chunker` implementations (token, sentence, paragraph,
section, hierarchical) land in subsequent releases. The protocol is
declared now so consumers in ``kaos-llm-core`` and
``kaos-nlp-transformers`` can program against a stable signature while
implementations are filled in.

Determinism
-----------
A chunker is deterministic when the same ``(text, options)`` always
produces the same sequence of :class:`Chunk` instances (including
identical ``chunk_id`` values). The default
:func:`compute_chunk_id` derives the identifier from the parent id and
the chunk's verbatim text + offsets, which is stable for the lifetime
of the source document.

Round-trip guarantee
--------------------
For every :class:`Chunk` produced by a chunker, the slice
``source[chunk.start:chunk.end]`` is expected to equal ``chunk.text``.
Implementations enforce this in their construction logic; the
:func:`validate_chunk_offsets` helper verifies the invariant for
callers and tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

_EMPTY_METADATA: Mapping[str, Any] = MappingProxyType({})


def compute_chunk_id(
    *,
    parent_id: str | None,
    start: int,
    end: int,
    text: str,
) -> str:
    """Derive a deterministic identifier for a chunk.

    The identifier is the first 32 hex characters of the SHA-256 of
    ``parent_id|start|end|text``. The encoding is stable across
    platforms and Python versions.

    Args:
        parent_id: Source document or parent chunk id. ``None`` is
            encoded as the empty string.
        start: Half-open start offset in the parent's character space.
        end: Half-open end offset (exclusive).
        text: Verbatim chunk text.

    Returns:
        128-bit hex digest (32 lowercase characters).
    """
    payload = f"{parent_id or ''}|{start}|{end}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous, offset-bearing slice of a source document.

    A :class:`Chunk` is the unit of work for downstream processing.
    The combination of ``parent_id`` plus ``(start, end)`` uniquely
    locates the chunk within its source.

    Attributes:
        text: Verbatim text of the chunk.
        start: Half-open start offset in ``parent_id``'s character
            space. ``0`` when the chunk covers from the beginning.
        end: Half-open end offset (exclusive). ``len(parent.text)``
            when the chunk covers to the end.
        parent_id: Identifier of the source document or parent chunk.
            ``None`` for unattached chunks; concrete chunkers should
            always populate it.
        chunk_id: Deterministic identifier derived from
            ``(parent_id, start, end, text)``. Populated automatically
            by :meth:`__post_init__` when left empty.
        token_count: Approximate token count under whichever tokenizer
            the chunker uses. ``-1`` means "not computed".
        depth: Recursive depth. ``0`` indicates a chunk produced
            directly from source text; a value of ``n > 0`` indicates
            the chunk was produced at level ``n`` of a reducer.
        metadata: Read-only mapping for chunker-specific annotations
            (section headings, heading depth, token offsets, etc.).
            Implementations should pass a ``MappingProxyType`` to
            preserve immutability; passing a plain dict is also
            accepted and wrapped on construction.

    The dataclass is frozen and slotted; instances are hashable by
    identity-affecting fields (``parent_id``, ``start``, ``end``,
    ``text``). ``metadata`` is intentionally excluded from the hash
    so two chunks of the same source region with different annotations
    compare equal.
    """

    text: str
    start: int
    end: int
    parent_id: str | None = None
    chunk_id: str = ""
    token_count: int = -1
    depth: int = 0
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_METADATA)

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        if not self.chunk_id:
            cid = compute_chunk_id(
                parent_id=self.parent_id,
                start=self.start,
                end=self.end,
                text=self.text,
            )
            object.__setattr__(self, "chunk_id", cid)
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __hash__(self) -> int:
        return hash((self.parent_id, self.start, self.end, self.text))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return (
            self.parent_id == other.parent_id
            and self.start == other.start
            and self.end == other.end
            and self.text == other.text
        )

    @property
    def char_span(self) -> tuple[int, int]:
        """Return ``(start, end)`` as a half-open offset tuple."""
        return (self.start, self.end)

    @property
    def length(self) -> int:
        """Number of characters covered by the chunk."""
        return self.end - self.start

    def slice_from(self, source: str) -> str:
        """Return ``source[start:end]``.

        Provided as a convenience for callers that want to confirm the
        chunk's offsets resolve to its text against a reconstructed
        source.
        """
        return source[self.start : self.end]


def validate_chunk_offsets(source: str, chunk: Chunk) -> bool:
    """Return ``True`` iff ``source[start:end] == chunk.text``.

    Tests and chunker implementations use this helper to assert the
    round-trip invariant. Returns ``False`` rather than raising so
    callers can collect failures across a batch.
    """
    if chunk.end > len(source):
        return False
    return source[chunk.start : chunk.end] == chunk.text


@runtime_checkable
class Chunker(Protocol):
    """Callable that splits a source text into :class:`Chunk` instances.

    Implementations are expected to be deterministic for fixed inputs
    and configuration. Implementations should populate ``parent_id``
    on every produced chunk and should produce chunks whose
    ``(start, end)`` offsets round-trip against the input string.

    The protocol is :func:`typing.runtime_checkable` so callers can use
    ``isinstance(obj, Chunker)`` to gate runtime hook-in.
    """

    def chunk(
        self,
        text: str,
        *,
        parent_id: str | None = None,
    ) -> list[Chunk]:
        """Split ``text`` into chunks.

        Args:
            text: Source text. Implementations should accept the empty
                string and return ``[]``.
            parent_id: Identifier to attach to each produced chunk's
                ``parent_id``. When ``None``, implementations may leave
                ``parent_id`` unset; the resulting chunks are still
                valid but cannot be re-anchored to a source after
                construction.

        Returns:
            A list of :class:`Chunk` ordered by ``start`` ascending.
        """
        ...


__all__ = [
    "Chunk",
    "Chunker",
    "FixedTokenChunker",
    "HierarchicalChunker",
    "ParagraphChunker",
    "SectionChunker",
    "SentenceChunker",
    "compute_chunk_id",
    "default_token_counter",
    "validate_chunk_offsets",
]


# ---------------------------------------------------------------------------
# Concrete chunker re-exports.
#
# Imported at the bottom of the module so the protocol and value type
# above are visible to the chunker implementations (which import
# ``Chunk`` and ``Chunker`` from this module).
# ---------------------------------------------------------------------------


from kaos_nlp_core.chunking._pack import default_token_counter  # noqa: E402
from kaos_nlp_core.chunking.fixed import FixedTokenChunker  # noqa: E402
from kaos_nlp_core.chunking.hierarchical import HierarchicalChunker  # noqa: E402
from kaos_nlp_core.chunking.paragraph import ParagraphChunker  # noqa: E402
from kaos_nlp_core.chunking.section import SectionChunker  # noqa: E402
from kaos_nlp_core.chunking.sentence import SentenceChunker  # noqa: E402
