"""Benchmarks for :mod:`kaos_nlp_core.chunking`.

Run with ``uv run pytest tests/bench_chunking.py -q --no-cov``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")

from kaos_nlp_core.chunking import (
    FixedTokenChunker,
    HierarchicalChunker,
    ParagraphChunker,
    SectionChunker,
    SentenceChunker,
)

_SHORT = "Hello world. This is a sentence. " * 10
_MEDIUM = (
    "Section header.\n\nThis is a paragraph with several sentences. "
    "Another sentence here. And one more. "
    "Now starting a new thought to fill out content.\n\n"
) * 50
_LONG = _MEDIUM * 20


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("short", _SHORT),
        ("medium", _MEDIUM),
        ("long", _LONG),
    ],
    ids=["short", "medium", "long"],
)
class TestBenchmarks:
    def test_fixed_token(self, benchmark, name, text) -> None:
        chunker = FixedTokenChunker(max_tokens=512, overlap_tokens=64)
        result = benchmark(chunker.chunk, text)
        assert len(result) >= 1

    def test_sentence(self, benchmark, name, text) -> None:
        chunker = SentenceChunker(max_tokens=512)
        result = benchmark(chunker.chunk, text)
        assert len(result) >= 1

    def test_paragraph(self, benchmark, name, text) -> None:
        chunker = ParagraphChunker(max_tokens=1024)
        result = benchmark(chunker.chunk, text)
        assert len(result) >= 1

    def test_section(self, benchmark, name, text) -> None:
        chunker = SectionChunker(max_tokens=1024)
        result = benchmark(chunker.chunk, text)
        assert len(result) >= 1

    def test_hierarchical(self, benchmark, name, text) -> None:
        chunker = HierarchicalChunker(max_tokens=1024)
        result = benchmark(chunker.chunk, text)
        assert len(result) >= 1
