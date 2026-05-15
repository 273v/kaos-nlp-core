"""Foundation tests for :mod:`kaos_nlp_core.chunking` types.

Concrete chunker implementations land in Phase 1; these tests cover
only the value types and protocol stub introduced in Phase 0:

- ``Chunk`` construction, immutability, hashing, identifier derivation,
  round-trip offset validation.
- ``compute_chunk_id`` determinism across processes.
- ``Chunker`` runtime-checkable protocol.
- ``validate_chunk_offsets`` helper.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any

import pytest

from kaos_nlp_core.chunking import (
    Chunk,
    Chunker,
    compute_chunk_id,
    validate_chunk_offsets,
)

# ---------------------------------------------------------------------------
# compute_chunk_id
# ---------------------------------------------------------------------------


class TestComputeChunkId:
    def test_returns_32_hex_characters(self) -> None:
        cid = compute_chunk_id(parent_id="doc-1", start=0, end=5, text="hello")
        assert len(cid) == 32
        assert all(c in "0123456789abcdef" for c in cid)

    def test_deterministic_for_same_inputs(self) -> None:
        a = compute_chunk_id(parent_id="doc-1", start=0, end=5, text="hello")
        b = compute_chunk_id(parent_id="doc-1", start=0, end=5, text="hello")
        assert a == b

    def test_changes_with_parent(self) -> None:
        a = compute_chunk_id(parent_id="doc-1", start=0, end=5, text="hello")
        b = compute_chunk_id(parent_id="doc-2", start=0, end=5, text="hello")
        assert a != b

    def test_changes_with_offsets(self) -> None:
        a = compute_chunk_id(parent_id="doc-1", start=0, end=5, text="hello")
        b = compute_chunk_id(parent_id="doc-1", start=1, end=6, text="hello")
        assert a != b

    def test_changes_with_text(self) -> None:
        a = compute_chunk_id(parent_id="doc-1", start=0, end=5, text="hello")
        b = compute_chunk_id(parent_id="doc-1", start=0, end=5, text="world")
        assert a != b

    def test_none_parent_id_encoded_as_empty(self) -> None:
        none_cid = compute_chunk_id(parent_id=None, start=0, end=5, text="hello")
        empty_cid = compute_chunk_id(parent_id="", start=0, end=5, text="hello")
        assert none_cid == empty_cid


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


class TestChunkConstruction:
    def test_basic_chunk(self) -> None:
        c = Chunk(text="hello", start=0, end=5, parent_id="doc-1")
        assert c.text == "hello"
        assert c.start == 0
        assert c.end == 5
        assert c.parent_id == "doc-1"
        assert c.token_count == -1
        assert c.depth == 0

    def test_chunk_id_auto_derived_when_empty(self) -> None:
        c = Chunk(text="hello", start=0, end=5, parent_id="doc-1")
        expected = compute_chunk_id(parent_id="doc-1", start=0, end=5, text="hello")
        assert c.chunk_id == expected

    def test_explicit_chunk_id_preserved(self) -> None:
        c = Chunk(text="hello", start=0, end=5, parent_id="doc-1", chunk_id="custom-id")
        assert c.chunk_id == "custom-id"

    def test_rejects_negative_start(self) -> None:
        with pytest.raises(ValueError, match=r"start must be >= 0"):
            Chunk(text="hello", start=-1, end=5)

    def test_rejects_end_before_start(self) -> None:
        with pytest.raises(ValueError, match=r"end .* must be >= start"):
            Chunk(text="hello", start=5, end=3)

    def test_empty_chunk_allowed(self) -> None:
        c = Chunk(text="", start=10, end=10)
        assert c.length == 0
        assert c.text == ""

    def test_metadata_wrapped_as_mappingproxy(self) -> None:
        c = Chunk(text="x", start=0, end=1, metadata={"section": "intro"})
        assert isinstance(c.metadata, MappingProxyType)
        assert c.metadata["section"] == "intro"

    def test_metadata_default_is_empty(self) -> None:
        c = Chunk(text="x", start=0, end=1)
        assert dict(c.metadata) == {}


class TestChunkImmutability:
    def test_chunk_is_frozen(self) -> None:
        c = Chunk(text="hello", start=0, end=5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.text = "world"  # ty: ignore[invalid-assignment]

    def test_metadata_cannot_be_mutated(self) -> None:
        c = Chunk(text="hello", start=0, end=5, metadata={"k": "v"})
        with pytest.raises(TypeError):
            c.metadata["k"] = "other"  # ty: ignore[invalid-assignment]


class TestChunkHashAndEquality:
    def test_equal_chunks_hash_equal(self) -> None:
        c1 = Chunk(text="hello", start=0, end=5, parent_id="doc-1")
        c2 = Chunk(text="hello", start=0, end=5, parent_id="doc-1")
        assert c1 == c2
        assert hash(c1) == hash(c2)

    def test_metadata_does_not_affect_equality(self) -> None:
        c1 = Chunk(text="hello", start=0, end=5, parent_id="doc-1", metadata={"a": 1})
        c2 = Chunk(text="hello", start=0, end=5, parent_id="doc-1", metadata={"b": 2})
        assert c1 == c2

    def test_chunks_usable_in_set(self) -> None:
        c1 = Chunk(text="hello", start=0, end=5, parent_id="doc-1")
        c2 = Chunk(text="hello", start=0, end=5, parent_id="doc-1")
        c3 = Chunk(text="world", start=6, end=11, parent_id="doc-1")
        assert len({c1, c2, c3}) == 2

    def test_different_parent_produces_different_hash(self) -> None:
        c1 = Chunk(text="hello", start=0, end=5, parent_id="doc-1")
        c2 = Chunk(text="hello", start=0, end=5, parent_id="doc-2")
        assert c1 != c2

    def test_chunk_not_equal_to_other_types(self) -> None:
        c = Chunk(text="hello", start=0, end=5)
        assert c != "hello"
        assert c != (0, 5, "hello")


class TestChunkRoundTrip:
    def test_slice_from_returns_text(self) -> None:
        source = "hello world"
        c = Chunk(text="hello", start=0, end=5)
        assert c.slice_from(source) == "hello"

    def test_validate_chunk_offsets_accepts_round_trip(self) -> None:
        source = "hello world"
        c = Chunk(text="hello", start=0, end=5)
        assert validate_chunk_offsets(source, c) is True

    def test_validate_chunk_offsets_rejects_mismatch(self) -> None:
        source = "hello world"
        c = Chunk(text="goodbye", start=0, end=5)
        assert validate_chunk_offsets(source, c) is False

    def test_validate_chunk_offsets_rejects_out_of_bounds(self) -> None:
        source = "hello"
        c = Chunk(text="hello world", start=0, end=11)
        assert validate_chunk_offsets(source, c) is False

    def test_unicode_round_trip(self) -> None:
        source = "café — naïve résumé"
        c = Chunk(text="café", start=0, end=4)
        assert validate_chunk_offsets(source, c) is True

    def test_cjk_round_trip(self) -> None:
        source = "测试中文chunking"
        c = Chunk(text="测试中文", start=0, end=4)
        assert validate_chunk_offsets(source, c) is True

    def test_emoji_round_trip(self) -> None:
        source = "hello 🌍 world"
        c = Chunk(text="🌍", start=6, end=7)
        assert validate_chunk_offsets(source, c) is True


class TestChunkProperties:
    def test_char_span(self) -> None:
        c = Chunk(text="hello", start=3, end=8)
        assert c.char_span == (3, 8)

    def test_length(self) -> None:
        c = Chunk(text="hello", start=3, end=8)
        assert c.length == 5

    def test_length_zero(self) -> None:
        c = Chunk(text="", start=3, end=3)
        assert c.length == 0


# ---------------------------------------------------------------------------
# Chunker protocol
# ---------------------------------------------------------------------------


class _OneChunkChunker:
    """Minimal Chunker implementation for protocol conformance tests."""

    def chunk(self, text: str, *, parent_id: str | None = None) -> list[Chunk]:
        if not text:
            return []
        return [Chunk(text=text, start=0, end=len(text), parent_id=parent_id)]


class TestChunkerProtocol:
    def test_minimal_implementation_is_chunker(self) -> None:
        impl = _OneChunkChunker()
        assert isinstance(impl, Chunker)

    def test_random_object_is_not_chunker(self) -> None:
        class _Unrelated:
            def fizz(self, text: str) -> int:
                return len(text)

        assert not isinstance(_Unrelated(), Chunker)

    def test_minimal_impl_returns_list_of_chunk(self) -> None:
        impl = _OneChunkChunker()
        result = impl.chunk("hello", parent_id="doc-1")
        assert len(result) == 1
        assert isinstance(result[0], Chunk)
        assert result[0].parent_id == "doc-1"

    def test_empty_text_returns_empty_list(self) -> None:
        impl = _OneChunkChunker()
        assert impl.chunk("") == []

    @pytest.mark.parametrize(
        "text,expected_len",
        [
            ("hello", 5),
            ("a", 1),
            ("café", 4),
            ("测试", 2),
        ],
    )
    def test_chunker_preserves_text(self, text: str, expected_len: int) -> None:
        impl = _OneChunkChunker()
        chunks = impl.chunk(text, parent_id="p")
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].length == expected_len


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


def test_chunking_module_exports() -> None:
    import kaos_nlp_core
    from kaos_nlp_core import chunking

    assert hasattr(kaos_nlp_core, "chunking")
    assert "chunking" in kaos_nlp_core.__all__
    expected: set[str] = {"Chunk", "Chunker", "compute_chunk_id", "validate_chunk_offsets"}
    assert expected <= set(chunking.__all__)


def test_chunk_metadata_accepts_arbitrary_payload() -> None:
    payload: dict[str, Any] = {"section": "intro", "depth": 1, "tokens": [1, 2, 3]}
    c = Chunk(text="x", start=0, end=1, metadata=payload)
    assert c.metadata["tokens"] == [1, 2, 3]
