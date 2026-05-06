"""Regression tests for F4: file-load hardening on the `.load()` APIs.

`load_bincode_from_path` enforces a `KNC1` magic header + format version
and a configurable size cap (`KAOS_NLP_MAX_LOAD_BYTES`, default 256 MiB).
Each public `.load()` static method routes through that helper and so
inherits these guarantees.

These tests verify that:
- A round-trip save → load works (sanity).
- Files with the wrong magic are rejected.
- Files with the wrong version are rejected.
- Files larger than `KAOS_NLP_MAX_LOAD_BYTES` are rejected.
- Truncated files are rejected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kaos_nlp_core._rust.structures import InvertedIndex


def _build_small_index() -> InvertedIndex:
    return InvertedIndex.build_batch(
        [
            (0, ["alpha", "beta", "gamma"]),
            (1, ["alpha", "delta"]),
            (2, ["beta", "epsilon"]),
        ]
    )


def test_round_trip(tmp_path: Path) -> None:
    """Sanity: save then load produces an equivalent index."""
    idx = _build_small_index()
    p = tmp_path / "idx.kncidx"
    idx.save(str(p))
    loaded = InvertedIndex.load(str(p))
    # Round-trip preserves the same set of documents per term.
    assert loaded.doc_freq("alpha") == 2
    assert loaded.doc_freq("beta") == 2
    assert loaded.doc_freq("epsilon") == 1


def test_load_rejects_wrong_magic(tmp_path: Path) -> None:
    """A file without the `KNC1` magic must be rejected with a clear error."""
    p = tmp_path / "bad-magic.kncidx"
    # Write something that's plausibly long enough but starts with the
    # wrong magic bytes (looks like a JSON `{"a":1}…` start).
    p.write_bytes(b'{"a"' + b"\x01\x00" + b"\x00" * 256)
    with pytest.raises(ValueError, match="missing KNC magic header"):
        InvertedIndex.load(str(p))


def test_load_rejects_wrong_version(tmp_path: Path) -> None:
    """`KNC1` magic + an unknown version must be rejected with a clear error."""
    p = tmp_path / "bad-version.kncidx"
    # Correct magic, version 99 (LE u16), then enough payload bytes.
    payload = b"\x00" * 64
    p.write_bytes(b"KNC1" + (99).to_bytes(2, "little") + payload)
    with pytest.raises(ValueError, match="unsupported KNC format version"):
        InvertedIndex.load(str(p))


def test_load_rejects_too_short(tmp_path: Path) -> None:
    """Files shorter than the header are rejected (cannot contain magic+version)."""
    p = tmp_path / "tiny.kncidx"
    p.write_bytes(b"KN")
    with pytest.raises(ValueError, match="too short"):
        InvertedIndex.load(str(p))


def test_load_honors_size_cap(tmp_path: Path) -> None:
    """`KAOS_NLP_MAX_LOAD_BYTES` rejects files larger than the configured cap.

    We write a real round-trip artifact (several hundred bytes), then set
    the cap to 16 bytes and confirm the load is refused before any bincode
    deserialization runs.
    """
    idx = _build_small_index()
    p = tmp_path / "idx.kncidx"
    idx.save(str(p))

    # Cap below the actual file size.
    actual_size = p.stat().st_size
    assert actual_size > 16
    os.environ["KAOS_NLP_MAX_LOAD_BYTES"] = "16"
    try:
        with pytest.raises(ValueError, match="exceeds KAOS_NLP_MAX_LOAD_BYTES"):
            InvertedIndex.load(str(p))
    finally:
        os.environ.pop("KAOS_NLP_MAX_LOAD_BYTES", None)


def test_load_size_cap_above_file_size_ok(tmp_path: Path) -> None:
    """Cap larger than the file leaves the load path working unchanged."""
    idx = _build_small_index()
    p = tmp_path / "idx.kncidx"
    idx.save(str(p))

    os.environ["KAOS_NLP_MAX_LOAD_BYTES"] = str(10 * 1024 * 1024)
    try:
        loaded = InvertedIndex.load(str(p))
        assert loaded.doc_freq("alpha") == 2
    finally:
        os.environ.pop("KAOS_NLP_MAX_LOAD_BYTES", None)
