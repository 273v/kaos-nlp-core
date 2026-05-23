"""Tests for `kaos_nlp_core.content_type`.

Cross-boundary tests through the public Python entry point. The
underlying Rust core is also tested in `rust/core/content_type/mod.rs`
(5 unit tests). These tests pin the public facade shape (typed
``ContentTypeResult``, the ``group`` enumeration the kaos-agents planner
depends on) and exercise the PyO3 boundary on a representative slice of
the kelvin-legal upload set.

Plus the OPC + OLE disambiguation fallbacks added in 0.1.1: real
DOCX / PPTX / XLSX bytes (synthesized in-memory) and the legacy OLE
CLSID sniffer.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pytest

from kaos_nlp_core.content_type import detect

# ─── Helpers: build minimal valid OPC + OLE byte streams in-memory ───────


def _build_opc_zip(
    main_part_content_type: str, main_part_name: str, payload: bytes = b"<doc/>"
) -> bytes:
    """Build a minimal valid Office Open XML (OPC) zip in memory.

    The OPC spec requires a ``[Content_Types].xml`` part that declares
    the Override ContentType for the main payload. The OPC fallback in
    ``kaos_nlp_core.content_type._disambiguate_zip`` greps for that
    Override string — anything matching one of the documented payload
    types should be recognised.
    """
    buf = io.BytesIO()
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/{main_part_name}" ContentType="{main_part_content_type}"/>'
        "</Types>"
    ).encode()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr(main_part_name, payload)
    return buf.getvalue()


_DOCX_PAYLOAD_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
)
_XLSX_PAYLOAD_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
_PPTX_PAYLOAD_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
)


def _build_ole_bytes(
    sig_512_513: tuple[int, int], sig_516_517: tuple[int, int] | None = None
) -> bytes:
    """Build a 1 KB byte buffer that mimics an OLE compound file.

    Only the bytes the disambiguator inspects need real values: the
    8-byte magic at offset 0 and the CLSID signature bytes at offset
    512 (and optionally 516, which disambiguates legacy .doc vs .ppt
    when both use the 0xFD, 0xFF prefix).
    """
    buf = bytearray(1024)
    # OLE magic: D0 CF 11 E0 A1 B1 1A E1
    buf[0:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    buf[512] = sig_512_513[0]
    buf[513] = sig_512_513[1]
    if sig_516_517 is not None:
        buf[516] = sig_516_517[0]
        buf[517] = sig_516_517[1]
    return bytes(buf)


# ─── Legacy plain-bytes coverage (preserved from earlier releases) ───────


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
        # ZIP magic: PK\x03\x04 — but bare zip without OPC markers
        # stays an archive (OPC fallback returns None).
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


# ─── OPC fallback coverage (0.1.1) ───────────────────────────────────────


class TestOPCFallbackSynthetic:
    """Real DOCX / PPTX / XLSX often arrive with central-directory
    layouts that the Rust `infer` crate cannot peek into; it returns
    ``mime=application/zip group=archive``. The Python facade adds an
    OPC fallback that opens the zip and matches the
    ``[Content_Types].xml`` Override string for the main payload.

    These tests synthesize minimal valid OPC zips in memory (no
    licensing concerns, deterministic on every platform) so the
    fallback's contract is pinned without depending on real Office
    binaries.
    """

    def test_synthetic_docx(self) -> None:
        data = _build_opc_zip(_DOCX_PAYLOAD_TYPE, "word/document.xml")
        result = detect(data)
        assert result.group == "office-docx", (
            f"OPC fallback failed; got group={result.group!r} mime={result.mime_type!r}"
        )
        assert (
            result.mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert result.extension == "docx"

    def test_synthetic_xlsx(self) -> None:
        data = _build_opc_zip(_XLSX_PAYLOAD_TYPE, "xl/workbook.xml")
        result = detect(data)
        assert result.group == "office-xlsx"
        assert (
            result.mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert result.extension == "xlsx"

    def test_synthetic_pptx(self) -> None:
        data = _build_opc_zip(_PPTX_PAYLOAD_TYPE, "ppt/presentation.xml")
        result = detect(data)
        assert result.group == "office-pptx"
        assert (
            result.mime_type
            == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
        assert result.extension == "pptx"

    def test_zip_without_opc_markers_stays_archive(self) -> None:
        # A bare zip with no [Content_Types].xml is still an archive.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("hello.txt", b"world")
        result = detect(buf.getvalue())
        assert result.group == "archive"
        assert result.mime_type == "application/zip"

    def test_zip_with_unknown_opc_payload_stays_archive(self) -> None:
        # OPC container, but Override payload is not one of the three
        # supported office formats — keep the archive classification
        # rather than mis-attributing.
        data = _build_opc_zip(
            "application/vnd.adobe.indesign-idml-package+zip",
            "design.idml",
        )
        result = detect(data)
        assert result.group == "archive"

    def test_corrupt_zip_falls_back_to_archive(self) -> None:
        # Bytes that look like a zip header but truncate before the
        # central directory must NOT raise — fall back to the original
        # ``archive`` classification.
        data = b"PK\x03\x04" + b"\x00" * 50
        result = detect(data)
        assert result.group == "archive"


# ─── OLE fallback coverage (0.1.1) ───────────────────────────────────────


class TestOLEFallbackSynthetic:
    """Legacy .doc / .xls / .ppt all share the OLE Compound File
    container magic ``D0 CF 11 E0``. The specific office format is
    recorded in a CLSID at byte offset 512. The facade sniffs that
    CLSID when the Rust core has not already produced a specific
    legacy-office classification.
    """

    def test_synthetic_legacy_doc(self) -> None:
        data = _build_ole_bytes(sig_512_513=(0xEC, 0xA5))
        result = detect(data)
        assert result.group == "office-doc", (
            f"OLE fallback failed; got group={result.group!r} mime={result.mime_type!r}"
        )
        assert result.mime_type == "application/msword"
        assert result.extension == "doc"

    def test_synthetic_legacy_xls(self) -> None:
        data = _build_ole_bytes(sig_512_513=(0x09, 0x08))
        result = detect(data)
        assert result.group == "office-xls"
        assert result.mime_type == "application/vnd.ms-excel"
        assert result.extension == "xls"

    def test_synthetic_legacy_ppt(self) -> None:
        # 0xFD 0xFF at 512:514, NO 0xFE 0xFF at 516:518 → PPT.
        data = _build_ole_bytes(sig_512_513=(0xFD, 0xFF), sig_516_517=(0x00, 0x00))
        result = detect(data)
        assert result.group == "office-ppt"
        assert result.mime_type == "application/vnd.ms-powerpoint"
        assert result.extension == "ppt"

    def test_synthetic_legacy_doc_via_alternate_clsid(self) -> None:
        # 0xFD 0xFF at 512:514 + 0xFE 0xFF at 516:518 → DOC.
        data = _build_ole_bytes(sig_512_513=(0xFD, 0xFF), sig_516_517=(0xFE, 0xFF))
        result = detect(data)
        assert result.group == "office-doc"

    def test_ole_magic_without_known_clsid_returns_rust_result(self) -> None:
        # OLE magic but with an unrecognized CLSID — the fallback
        # returns None, so we trust whatever the Rust core produced
        # (often ``unknown`` for this synthetic shape, or a specific
        # OLE-ish classification if the crate matched something).
        data = _build_ole_bytes(sig_512_513=(0xAB, 0xCD))
        result = detect(data)
        # We do NOT assert a specific group — just that nothing crashed
        # and the result remains a typed ContentTypeResult.
        assert hasattr(result, "group")


# ─── Real-fixture coverage (skipped when fixtures absent) ────────────────


def _sibling_fixtures(rel_glob: str) -> Iterable[Path]:
    """Yield real Office fixtures from the kaos-office sibling repo, if present.

    Many developer machines have ``../kaos-office/tests/fixtures/...``
    available because of the workspace layout. CI also clones the
    sibling for cross-package integration runs. When the directory is
    absent we yield nothing and the dependent tests skip cleanly.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "kaos-office" / "tests" / "fixtures",
        here.parents[3] / "kaos-office" / "tests" / "fixtures",
    ]
    for base in candidates:
        if base.is_dir():
            yield from sorted(base.glob(rel_glob))
            return


class TestOPCFallbackRealFixtures:
    """Regression coverage against real Office files that previously
    misclassified as ``archive``. Skipped when the fixtures aren't
    available (e.g. sdist install without sibling repos)."""

    @pytest.mark.parametrize(
        "pattern,expected_group",
        [
            ("docx/*.docx", "office-docx"),
            ("pptx/*.pptx", "office-pptx"),
            ("xlsx/*.xlsx", "office-xlsx"),
        ],
    )
    def test_real_office_files_classify_correctly(self, pattern: str, expected_group: str) -> None:
        fixtures = list(_sibling_fixtures(pattern))
        if not fixtures:
            pytest.skip(f"no sibling kaos-office fixtures matching {pattern!r}")
        misclassified: list[tuple[str, str]] = []
        for path in fixtures:
            data = path.read_bytes()
            result = detect(data)
            if result.group != expected_group:
                misclassified.append((path.name, result.group))
        assert not misclassified, (
            f"{len(misclassified)}/{len(fixtures)} {pattern} files misclassified: "
            f"{misclassified[:5]}{'...' if len(misclassified) > 5 else ''}"
        )


# ─── Taxonomy contract (preserved from earlier releases) ─────────────────


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
            _build_opc_zip(_DOCX_PAYLOAD_TYPE, "word/document.xml"),
            _build_opc_zip(_XLSX_PAYLOAD_TYPE, "xl/workbook.xml"),
            _build_opc_zip(_PPTX_PAYLOAD_TYPE, "ppt/presentation.xml"),
            _build_ole_bytes(sig_512_513=(0xEC, 0xA5)),
            _build_ole_bytes(sig_512_513=(0x09, 0x08)),
            _build_ole_bytes(sig_512_513=(0xFD, 0xFF), sig_516_517=(0x00, 0x00)),
        ]
        for sample in samples:
            result = detect(sample)
            assert result.group in self._ALLOWED_GROUPS, (
                f"detect({sample[:32]!r}…) returned undocumented group {result.group!r}"
            )
