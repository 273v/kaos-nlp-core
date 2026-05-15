"""Content-type detection by magic-byte signature.

Thin typed facade over the Rust ``core::content_type`` module. Feeds the
kaos-agents per-turn planner's ``corpus_kinds`` Signature input — the
planner uses the coarse ``group`` field (one of ``"pdf"``,
``"office-docx"``, ``"image"``, ``"archive"``, ``"email"``,
``"html"``, ``"text"``, ``"binary"``, ``"unknown"``, ...) to decide
whether a session's uploaded corpus calls for PDF tools, office tools,
image tools, or a generic text path.

See ``kaos-modules/docs/internal/dynamic-tool-planning-prd.md`` §4
(PR 4) for the architectural context. The detector wraps the `infer`
Rust crate (MIT, zero runtime deps) — pure magic-byte sniffing, no ML
model. Sufficient for the kelvin-legal upload set (PDF / DOCX / XLSX /
PPTX / JPEG / PNG / ZIP / EML / ...); upgrade to Google Magika 1.0 is
tracked separately if accuracy demands it.
"""

from __future__ import annotations

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
        """True when `infer` recognized a magic-byte signature."""
        return self.group != "unknown"


def detect(data: bytes) -> ContentTypeResult:
    """Detect the content type of ``data`` by magic-byte signature.

    Returns ``ContentTypeResult(mime_type="", extension="",
    group="unknown")`` when no signature matched — callers can fall
    back to file-extension heuristics, NLP-driven content sniffing,
    or just route to the generic-text path.

    For a sample of at least 256 bytes, detection accuracy on the
    kelvin-legal upload set (PDF / DOCX / XLSX / PPTX / JPEG / PNG /
    ZIP / EML / ...) is effectively 100% — these formats all carry
    stable magic-byte signatures in their header.

    ``data`` must be a ``bytes`` instance. Passing ``str`` raises
    ``TypeError`` at the PyO3 boundary; callers must encode first.
    """
    raw = _detect(data)
    return ContentTypeResult(
        mime_type=raw["mime_type"],
        extension=raw["extension"],
        group=raw["group"],
    )
