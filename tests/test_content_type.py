"""Tests for `kaos_nlp_core.content_type`.

Cross-boundary tests through the public Python entry point — the
underlying Rust core is also tested in
`rust/core/content_type/mod.rs` (5 unit tests). These tests pin the
public facade shape (typed `ContentTypeResult`, the `group`
enumeration the kaos-agents planner depends on) and exercise the
PyO3 boundary on a representative slice of the kelvin-legal upload
set.
"""

from __future__ import annotations

import pytest

from kaos_nlp_core.content_type import detect


class TestDetectKnownFormats:
    """The planner needs stable `group` labels — pin them here."""

    def test_pdf_magic_bytes(self) -> None:
        result = detect(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        assert result.mime_type == "application/pdf"
        assert result.extension == "pdf"
        assert result.group == "pdf"
        assert result.is_known is True

    def test_png_magic_bytes(self) -> None:
        result = detect(b"\x89PNG\r\n\x1a\n")
        assert result.extension == "png"
        assert result.group == "image"
        assert result.is_known is True

    def test_jpeg_magic_bytes(self) -> None:
        # JPEG SOI marker: FF D8 FF
        result = detect(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        assert result.extension == "jpg"
        assert result.group == "image"

    def test_zip_archive(self) -> None:
        # ZIP magic: PK\x03\x04
        result = detect(b"PK\x03\x04")
        assert result.extension == "zip"
        assert result.group == "archive"

    def test_gzip_archive(self) -> None:
        # gzip magic: 1f 8b
        result = detect(b"\x1f\x8b\x08\x00")
        assert result.group == "archive"


class TestDetectUnknown:
    """Unknown bytes route to ``group == "unknown"`` — the planner
    treats this as "fall back to file-extension or generic-text"."""

    def test_plain_ascii_is_unknown(self) -> None:
        # `infer` does not classify plaintext — only magic-byte formats.
        result = detect(b"hello world, plain text without magic bytes")
        assert result.mime_type == ""
        assert result.extension == ""
        assert result.group == "unknown"
        assert result.is_known is False

    def test_empty_bytes(self) -> None:
        result = detect(b"")
        assert result.group == "unknown"


class TestPublicShape:
    """Pin the public dataclass shape — kaos-agents and kaos-content
    consumers depend on this."""

    def test_result_is_frozen(self) -> None:
        result = detect(b"%PDF-1.4")
        # Frozen + slotted — attempting mutation raises.
        with pytest.raises((AttributeError, TypeError)):
            result.group = "image"  # ty: ignore[invalid-assignment]

    def test_result_fields_are_strings(self) -> None:
        result = detect(b"%PDF-1.4")
        assert isinstance(result.mime_type, str)
        assert isinstance(result.extension, str)
        assert isinstance(result.group, str)

    def test_str_input_raises_type_error(self) -> None:
        # PyBytes-only at the PyO3 boundary; callers must encode first.
        with pytest.raises(TypeError):
            detect("not bytes")  # ty: ignore[invalid-argument-type]


class TestGroupTaxonomy:
    """The `group` enumeration is a stable contract with the kaos-agents
    planner's few-shot examples. Pin the full set so a future change
    surfaces here before it breaks the planner."""

    _ALLOWED_GROUPS = frozenset(
        {
            "pdf",
            "office-docx",
            "office-xlsx",
            "office-pptx",
            "office-doc",
            "office-xls",
            "office-ppt",
            "image",
            "audio",
            "video",
            "archive",
            "email",
            "html",
            "text",
            "font",
            "binary",
            "unknown",
        }
    )

    def test_known_inputs_return_documented_groups(self) -> None:
        samples = [
            b"%PDF-1.4",
            b"\x89PNG\r\n\x1a\n",
            b"\xff\xd8\xff\xe0",
            b"PK\x03\x04",
            b"\x1f\x8b\x08\x00",
            b"",
            b"unknown garbage bytes",
        ]
        for sample in samples:
            result = detect(sample)
            assert result.group in self._ALLOWED_GROUPS, (
                f"detect({sample!r}) returned undocumented group {result.group!r}"
            )
