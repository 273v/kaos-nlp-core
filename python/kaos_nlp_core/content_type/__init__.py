"""Content-type detection by magic-byte signature.

Typed facade over the Rust ``core::content_type`` module. Feeds the
kaos-agents per-turn planner's ``corpus_kinds`` Signature input — the
planner uses the coarse ``group`` field (one of ``"pdf"``,
``"office-docx"``, ``"image"``, ``"archive"``, ``"email"``,
``"html"``, ``"text"``, ``"binary"``, ``"unknown"``, ...) to decide
whether a session's uploaded corpus calls for PDF tools, office tools,
image tools, or a generic text path.

See ``kaos-modules/docs/internal/dynamic-tool-planning-prd.md`` §4
(PR 4) for the architectural context. The Rust core wraps the `infer`
crate (MIT, zero runtime deps) — pure magic-byte sniffing, no ML model.

This facade adds two stdlib-based disambiguation passes on top of the
Rust result to fix the kind of false-negative the planner cannot afford:

- **OPC fallback** for Office Open XML (DOCX / PPTX / XLSX). ``infer``
  recognizes the zip container but cannot peek inside; real-world Word
  and PowerPoint files with non-standard central-directory layouts
  (fillable forms, compact PowerPoint exports) often fall through as
  ``mime=application/zip group=archive``. We open the zip, locate
  ``[Content_Types].xml``, and match the documented Override
  ``ContentType`` string for the main payload to recover the correct
  office-docx / office-xlsx / office-pptx classification.
- **OLE CLSID fallback** for legacy ``.doc`` / ``.xls`` / ``.ppt``
  (Microsoft Compound File Binary). When the file carries the OLE magic
  ``D0 CF 11 E0`` at offset 0, we sniff the CLSID at offset 512 to
  distinguish the three legacy office formats.

Both passes are ported from the battle-tested
``kelvin/source/headers.py`` implementation that shipped in the prior
``kelvin-source`` generation; see that module for the historical
context. They are pure stdlib (``zipfile``) so adding them does not
affect the Rust core, the maturin wheel matrix, or the PyO3 boundary.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

from kaos_nlp_core._rust.content_type import py_detect as _detect

__all__ = ["ContentTypeResult", "detect"]


@dataclass(frozen=True, slots=True)
class ContentTypeResult:
    """One detection result: precise MIME + extension + coarse group.

    ``group`` is the kaos-agents ``corpus_kinds`` bucket — a fixed
    enumeration the planner's few-shot examples are written against:

    - ``"pdf"`` — application/pdf
    - ``"office-docx"`` / ``"office-xlsx"`` / ``"office-pptx"`` —
      Open XML Office documents
    - ``"office-doc"`` / ``"office-xls"`` / ``"office-ppt"`` — legacy
      binary Office formats
    - ``"image"`` — any image/*
    - ``"audio"`` — any audio/*
    - ``"video"`` — any video/*
    - ``"archive"`` — zip, tar, gz, bz2, 7z, rar, etc.
    - ``"email"`` — message/rfc822, application/mbox, .eml
    - ``"html"`` — text/html
    - ``"text"`` — text/plain
    - ``"font"`` — any font/*
    - ``"binary"`` — application/octet-stream or recognized binary
      that doesn't fall into the above
    - ``"unknown"`` — no signature matched
    """

    mime_type: str
    extension: str
    group: str

    @property
    def is_known(self) -> bool:
        """True when a magic-byte signature recognized the input."""
        return self.group != "unknown"


# ─── Office Open XML (.docx / .pptx / .xlsx) disambiguation ──────────────
#
# When the Rust ``infer`` path reports ``application/zip`` we may still be
# looking at a DOCX / PPTX / XLSX whose central-directory layout did not
# expose the OPC markers within ``infer``'s seek window. The OPC
# ``[Content_Types].xml`` part carries a deterministic Override
# ``ContentType`` for the main payload — match on that. Ported from
# ``kelvin-source/kelvin/source/headers.py::check_office_xml_zip``.

_OPC_PAYLOAD_MARKERS: tuple[tuple[bytes, str, str, str], ...] = (
    (
        b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx",
        "office-docx",
    ),
    (
        b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xlsx",
        "office-xlsx",
    ),
    (
        b"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "pptx",
        "office-pptx",
    ),
)


def _disambiguate_zip(data: bytes) -> ContentTypeResult | None:
    """Peek inside a zip for OPC markers; return None when not Office Open XML.

    Returns ``None`` when the bytes are not a valid zip, do not contain
    ``[Content_Types].xml``, or carry payload types we do not recognize.
    The caller keeps the original ``archive`` classification in that case.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            try:
                content_types = zf.read("[Content_Types].xml")
            except KeyError:
                return None
            for marker, mime, ext, group in _OPC_PAYLOAD_MARKERS:
                if marker in content_types:
                    return ContentTypeResult(mime_type=mime, extension=ext, group=group)
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError):
        # RuntimeError covers password-protected entries on read().
        return None
    return None


# ─── Legacy OLE compound document (.doc / .xls / .ppt) disambiguation ────
#
# Microsoft Compound File Binary (the container for legacy .doc / .xls /
# .ppt) starts with the magic bytes ``D0 CF 11 E0 A1 B1 1A E1``. The
# specific office format is recorded in the CLSID at byte 512. Ported
# from ``kelvin-source/kelvin/source/headers.py::get_ole_content_type``.

_OLE_MAGIC: tuple[int, int, int, int] = (0xD0, 0xCF, 0x11, 0xE0)

_LEGACY_OFFICE_GROUPS: frozenset[str] = frozenset({"office-doc", "office-xls", "office-ppt"})


def _disambiguate_ole(data: bytes) -> ContentTypeResult | None:
    """Sniff the OLE CLSID at offset 512 → office-doc/office-xls/office-ppt.

    Returns ``None`` when the input is too short to inspect, does not
    carry the OLE magic, or carries a CLSID we do not recognize.
    """
    if len(data) < 518:
        return None
    if (data[0], data[1], data[2], data[3]) != _OLE_MAGIC:
        return None

    sig = (data[512], data[513])
    if sig == (0xEC, 0xA5):
        return ContentTypeResult(
            mime_type="application/msword", extension="doc", group="office-doc"
        )
    if sig == (0x09, 0x08):
        return ContentTypeResult(
            mime_type="application/vnd.ms-excel", extension="xls", group="office-xls"
        )
    if sig == (0xFD, 0xFF):
        # PPT and DOC share this prefix. Disambiguate at 516:518.
        if (data[516], data[517]) == (0xFE, 0xFF):
            return ContentTypeResult(
                mime_type="application/msword", extension="doc", group="office-doc"
            )
        return ContentTypeResult(
            mime_type="application/vnd.ms-powerpoint",
            extension="ppt",
            group="office-ppt",
        )
    return None


def detect(data: bytes) -> ContentTypeResult:
    """Detect the content type of ``data`` by magic-byte signature.

    Returns ``ContentTypeResult(mime_type="", extension="",
    group="unknown")`` when no signature matched — callers can fall back
    to file-extension heuristics, NLP-driven content sniffing, or just
    route to the generic-text path.

    The Rust core handles the bulk of recognized formats. This facade
    layers two stdlib-based disambiguation passes on top to fix the
    classes of false-negative that the planner cannot tolerate:

    - When the Rust core returns ``application/zip``, peek inside for
      OPC markers and upgrade to ``office-docx`` / ``office-xlsx`` /
      ``office-pptx`` when present.
    - When the bytes start with the OLE magic but the Rust core has not
      already classified them as one of the legacy office formats,
      inspect the CLSID at offset 512 to recover ``office-doc`` /
      ``office-xls`` / ``office-ppt``.

    ``data`` must be ``bytes``. Passing ``str`` raises ``TypeError`` at
    the PyO3 boundary; callers must encode first.
    """
    raw = _detect(data)
    result = ContentTypeResult(
        mime_type=raw["mime_type"],
        extension=raw["extension"],
        group=raw["group"],
    )

    # OPC fallback: real DOCX/PPTX/XLSX frequently arrive with
    # mime=application/zip when their central directory does not place
    # the OPC markers in infer's seek window. See module docstring.
    if result.mime_type == "application/zip":
        upgraded = _disambiguate_zip(data)
        if upgraded is not None:
            return upgraded

    # OLE fallback: only act when the bytes carry the OLE magic AND the
    # Rust core has not already produced a specific legacy office group.
    # This keeps the helper strictly additive — never downgrades a more
    # specific Rust answer.
    if (
        result.group not in _LEGACY_OFFICE_GROUPS
        and len(data) >= 4
        and (data[0], data[1], data[2], data[3]) == _OLE_MAGIC
    ):
        upgraded = _disambiguate_ole(data)
        if upgraded is not None:
            return upgraded

    return result
