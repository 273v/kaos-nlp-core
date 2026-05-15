"""Section-aware chunker for structured documents.

Detects section boundaries via :func:`parse_enumerator` over each line.
A line that parses as an enumerator (``1.``, ``§ 2.1``, ``Article III.``,
``# Heading``, etc.) opens a new section; everything from that line up
to (but not including) the next enumerator-bearing line forms a single
section.

Each section is then packed via the supplied paragraph chunker, with
``max_tokens`` enforced on the resulting chunks. Sections smaller than
the budget emit as one chunk; oversize sections are subdivided.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kaos_nlp_core.chunking import Chunk
from kaos_nlp_core.chunking._pack import default_token_counter
from kaos_nlp_core.chunking.paragraph import ParagraphChunker
from kaos_nlp_core.segmentation import parse_enumerator, segment_lines


@dataclass(frozen=True, slots=True)
class _Section:
    """A detected section in the source text.

    Attributes:
        start: Half-open start offset.
        end: Half-open end offset.
        heading: First line of the section (verbatim).
        kind: Enumerator kind on the heading (``decimal``, ``roman``,
            ``alpha``, ``markdown``, etc.) or empty when the section
            is the implicit prologue before any enumerator.
        depth: Enumerator depth (``1`` for top-level, ``2`` for
            sub-sections, …). ``0`` for the implicit prologue.
    """

    start: int
    end: int
    heading: str
    kind: str
    depth: int


def _detect_sections(text: str) -> list[_Section]:
    """Return the section boundaries inferred from enumerator-bearing lines."""
    lines = segment_lines(text)
    headings: list[tuple[int, int, str, str, int]] = []
    for line in lines:
        if not line.text.strip():
            continue
        enumerator = parse_enumerator(line.text)
        if enumerator is None:
            continue
        # ``Enumerator`` exposes ``kind`` and ``depth`` attributes via
        # PyO3 — see ``segmentation/__init__.py`` for the types.
        headings.append(
            (
                line.start,
                line.end,
                line.text,
                getattr(enumerator, "kind", ""),
                getattr(enumerator, "depth", 1) or 1,
            )
        )

    if not headings:
        # No structure detected — return a single implicit section
        # covering the whole document so callers always get at least
        # one section to chunk.
        return [
            _Section(start=0, end=len(text), heading="", kind="", depth=0),
        ]

    sections: list[_Section] = []
    # Implicit prologue before the first heading, if any.
    first_start = headings[0][0]
    if first_start > 0:
        prologue_text = text[0:first_start].rstrip()
        if prologue_text:
            sections.append(
                _Section(
                    start=0,
                    end=first_start,
                    heading="",
                    kind="",
                    depth=0,
                )
            )

    for index, (start, _end, heading_text, kind, depth) in enumerate(headings):
        next_start = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        sections.append(
            _Section(
                start=start,
                end=next_start,
                heading=heading_text,
                kind=kind,
                depth=depth,
            )
        )

    return sections


class SectionChunker:
    """Detect enumerator-bearing sections, then chunk within each section.

    Args:
        max_tokens: Soft token-budget per chunk. Default ``1024``.
        token_counter: Callable used to score chunk text.
        paragraph_chunker: Optional override for the within-section
            packer. Defaults to a :class:`ParagraphChunker` configured
            with the same ``max_tokens`` and ``token_counter``.

    Output:
        Each chunk's :attr:`Chunk.metadata` includes ``section_kind``,
        ``section_depth``, and ``section_heading`` so consumers can
        group chunks back by section if needed.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 1024,
        token_counter: Callable[[str], int] = default_token_counter,
        paragraph_chunker: ParagraphChunker | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        self.max_tokens = max_tokens
        self.token_counter = token_counter
        self._paragraph_chunker = paragraph_chunker or ParagraphChunker(
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
            sub_chunks = self._paragraph_chunker.chunk(section_text, parent_id=parent_id)
            for sub in sub_chunks:
                chunks.append(
                    Chunk(
                        text=sub.text,
                        start=section.start + sub.start,
                        end=section.start + sub.end,
                        parent_id=parent_id,
                        token_count=sub.token_count,
                        depth=sub.depth,
                        metadata={
                            **dict(sub.metadata),
                            "chunker": "SectionChunker",
                            "section_index": section_index,
                            "section_kind": section.kind,
                            "section_depth": section.depth,
                            "section_heading": section.heading,
                            "max_tokens": self.max_tokens,
                        },
                    )
                )
        return chunks


__all__ = ["SectionChunker"]
