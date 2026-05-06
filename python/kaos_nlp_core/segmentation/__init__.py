"""Segmentation: sentence (Punkt), line, and paragraph splitting.

Public helpers default to the bundled legal Punkt model when available.
All paragraph segmentation is sentence-aware and only breaks at sentence
boundaries. All functions return ``list[Segment]`` with typed fields.

`extract_line_records` is the structural-analysis entry point that returns
typed `LineRecord` instances with case/punct profile, indent, blank-flags
and char-offset slices into the source text.
"""

from __future__ import annotations

from enum import IntFlag
from typing import Any

from kaos_nlp_core._defaults import (
    get_default_punkt_tokenizer,
    load_default_punkt_parameters,
)
from kaos_nlp_core._rust.segmentation import (
    BoilerplateRun as _RustBoilerplateRun,
)
from kaos_nlp_core._rust.segmentation import (
    Enumerator as _RustEnumerator,
)
from kaos_nlp_core._rust.segmentation import (
    LineRecord as _RustLineRecord,
)
from kaos_nlp_core._rust.segmentation import (
    NormalizedText as _RustNormalizedText,
)
from kaos_nlp_core._rust.segmentation import (
    PunktParameters,
    PunktTokenizer,
    PunktTrainer,
)
from kaos_nlp_core._rust.segmentation import (
    py_detect_boilerplate as _raw_detect_boilerplate,
)
from kaos_nlp_core._rust.segmentation import (
    py_extract_line_records as _raw_extract_line_records,
)
from kaos_nlp_core._rust.segmentation import py_normalize as _raw_normalize
from kaos_nlp_core._rust.segmentation import (
    py_parse_enumerator as _raw_parse_enumerator,
)
from kaos_nlp_core._rust.segmentation import py_segment_lines as _raw_segment_lines
from kaos_nlp_core._rust.segmentation import py_segment_paragraphs as _raw_segment_paragraphs
from kaos_nlp_core._rust.segmentation import (
    py_segment_paragraphs_simple as _raw_segment_paragraphs_simple,
)
from kaos_nlp_core._rust.segmentation import py_segment_sentences as _raw_segment_sentences
from kaos_nlp_core.types import Segment


class PunctProfile(IntFlag):
    """Bitfield summary of punctuation / symbol presence on a line.

    Values mirror ``rust/core/segmentation/line_record.rs::PunctProfile``.
    Use ``PunctProfile.HAS_PIPE in profile_value`` for membership checks.
    """

    ENDS_PERIOD = 1 << 0
    ENDS_COLON = 1 << 1
    ENDS_SEMICOLON = 1 << 2
    ENDS_QUESTION = 1 << 3
    ENDS_COMMA = 1 << 4
    HAS_PIPE = 1 << 5
    HAS_TAB = 1 << 6
    HAS_DIGITS = 1 << 7
    HAS_PARENS = 1 << 8
    HAS_BRACKETS = 1 << 9
    HAS_SECTION_SIG = 1 << 10
    HAS_COLUMN_GAPS = 1 << 11


def extract_line_records(text: str) -> list[_RustLineRecord]:
    """Extract typed ``LineRecord`` instances from ``text``.

    All offsets on the returned records are **character offsets** (Python
    ``str`` indexing). The Rust core works in byte offsets and the binding
    converts at the FFI boundary.
    """
    return list(_raw_extract_line_records(text))


def normalize(
    text: str,
    *,
    collapse_whitespace: bool = False,
    fold_case: bool = False,
    normalize_unicode_punct: bool = False,
    strip_enumerator_prefix: bool = False,
    strip_punctuation: bool = False,
) -> _RustNormalizedText:
    """Normalize ``text`` with offset preservation.

    The returned :class:`NormalizedText` carries the transformed string
    in ``.text`` and an optional ``.orig_char_offsets`` vector. When
    ``orig_char_offsets`` is ``None`` the input was returned unchanged
    (fast path); otherwise ``len(orig_char_offsets) == len(text)`` and
    each entry is the source-side **character offset** of the codepoint
    that contributed it. ``orig_char_offsets`` is monotonically
    non-decreasing.

    :raises NotImplementedError: when ``strip_enumerator_prefix=True``;
        the underlying Enumerator parser (P3) is not yet built.
    """
    return _raw_normalize(
        text,
        collapse_whitespace=collapse_whitespace,
        fold_case=fold_case,
        normalize_unicode_punct=normalize_unicode_punct,
        strip_enumerator_prefix=strip_enumerator_prefix,
        strip_punctuation=strip_punctuation,
    )


def detect_boilerplate(
    text: str,
    *,
    lines_per_page: int = 50,
    header_zone_lines: int = 3,
    footer_zone_lines: int = 3,
    min_occurrences: int = 3,
    min_rate: float = 0.5,
    skip_near_dup: bool = False,
    near_dup_threshold: float = 0.75,
    num_perm: int = 64,
    shingle_size: int = 4,
    zone_dominance: float = 0.7,
    drop_empty: bool = True,
) -> list[_RustBoilerplateRun]:
    """Detect repeated boilerplate (headers, footers, page numbers, captions).

    The detector runs a two-stage algorithm:

    1. Bucket each line by page — using ``\\f`` (form feed, U+000C) when
       present, otherwise a ``lines_per_page``-line window.
    2. Cluster by exact normalized fingerprint, then by MinHash near-duplicate
       (for OCR drift) on the residual lines.

    A cluster is reported as boilerplate when both ``min_occurrences`` and
    ``min_rate`` (fraction of distinct pages it appears on) are satisfied.

    Returns a list of :class:`BoilerplateRun` instances ordered deterministically
    by fingerprint.
    """
    return list(
        _raw_detect_boilerplate(
            text,
            lines_per_page=lines_per_page,
            header_zone_lines=header_zone_lines,
            footer_zone_lines=footer_zone_lines,
            min_occurrences=min_occurrences,
            min_rate=min_rate,
            skip_near_dup=skip_near_dup,
            near_dup_threshold=near_dup_threshold,
            num_perm=num_perm,
            shingle_size=shingle_size,
            zone_dominance=zone_dominance,
            drop_empty=drop_empty,
        )
    )


def parse_enumerator(
    line: str,
    *,
    lexicon: str | None = None,
    custom_lexicon: list[tuple[str, str]] | None = None,
) -> _RustEnumerator | None:
    """Parse the leading enumerator of ``line``.

    Caller is expected to pre-strip leading whitespace; the parser anchors
    at byte 0. Returns ``None`` when no enumerator-shaped prefix is found.

    Recognised kinds:

    * ``"decimal"`` — ``1.``, ``1.2``, ``1.2.3``, depth ≤ 4.
    * ``"roman_upper"`` / ``"roman_lower"`` — ``I.``, ``IV.``, ``i.``, ``xliii.``.
    * ``"alpha_upper"`` / ``"alpha_lower"`` — ``A.``..``Z.`` / ``a.``..``z.``.
    * ``"paren_alpha"`` / ``"paren_decimal"`` / ``"paren_roman"`` — balanced
      parentheses around a single segment, e.g. ``(a)``, ``(1)``, ``(iv)``.
    * ``"section"`` — the ``§`` sigil followed by a value.
    * ``"section_word"`` — ``Section 5``, ``Sec. 5``, plus the localized
      equivalents from the active ``lexicon`` (Article, Chapitre, Artículo,
      Artikel, Articolo, Artigo, etc.) AND Markdown ATX (``# Title`` →
      ``section_word`` with depth read from leading ``#`` count).
    * ``"chapter_word"`` — ``Chapter``, ``Title``, ``Subtitle``, ``Part``,
      ``Appendix``, ``Schedule`` followed by a value (per active lexicon).
    * ``"subpart_word"`` — ``Subpart``, ``Subchapter``, ``Article``,
      ``Paragraph`` followed by a value (per active lexicon).

    The ``lexicon`` keyword selects which word-prefix dictionary fires for
    Section / Chapter / Article / etc. matching. Western-language built-ins:

    * ``"english_legal_us"`` — default; Section / Chapter / Title / …
    * ``"french_legal"`` — Article / Chapitre / Titre / Section / Annexe / …
    * ``"german_legal"`` — Artikel / Kapitel / Titel / Abschnitt / Anhang / …
    * ``"spanish_legal"`` — Artículo / Capítulo / Título / Sección / Anexo / …
    * ``"italian_legal"`` — Articolo / Capo / Capitolo / Titolo / Sezione / …
    * ``"portuguese_legal"`` — Artigo / Capítulo / Título / Secção / Seção / …
    * ``"markdown_atx"`` — H1..H6 from leading ``#`` count.

    For domain-specific keyword sets (e.g., software docs `Step` / `Phase`),
    pass ``custom_lexicon`` as a list of ``(pattern, kind)`` tuples; if both
    are supplied the custom one wins.

    Single-letter ``I`` / ``V`` (and lowercase) parse as Roman per Pandoc's
    documented rule; other single letters parse as Alpha.

    Offsets on the returned record are **character offsets** (Python ``str``
    indexing).
    """
    return _raw_parse_enumerator(
        line,
        lexicon=lexicon,
        custom_lexicon=custom_lexicon,
    )


# Re-export so callers can `from kaos_nlp_core.segmentation import LineRecord`.
BoilerplateRun = _RustBoilerplateRun
Enumerator = _RustEnumerator
LineRecord = _RustLineRecord
NormalizedText = _RustNormalizedText


def _to_segments(raw: Any) -> list[Segment]:
    return [
        Segment(
            text=s["text"],
            start=s["start"],
            end=s["end"],
            confidence=s.get("confidence", 1.0),
        )
        for s in raw
    ]


def segment_sentences(text: str, tokenizer: PunktTokenizer | None = None) -> list[Segment]:
    """Segment text into sentences using the bundled legal model by default."""
    active_tokenizer = tokenizer or get_default_punkt_tokenizer()
    return _to_segments(_raw_segment_sentences(text, active_tokenizer))


def segment_paragraphs(text: str, tokenizer: PunktTokenizer | None = None) -> list[Segment]:
    """Segment text into paragraphs using the bundled legal model by default."""
    active_tokenizer = tokenizer or get_default_punkt_tokenizer()
    return _to_segments(_raw_segment_paragraphs(text, active_tokenizer))


def segment_lines(text: str) -> list[Segment]:
    """Segment text into lines."""
    return _to_segments(_raw_segment_lines(text))


def segment_paragraphs_simple(text: str) -> list[Segment]:
    """Segment text into paragraphs by blank lines only (no sentence awareness)."""
    return _to_segments(_raw_segment_paragraphs_simple(text))


__all__ = [
    "BoilerplateRun",
    "Enumerator",
    "LineRecord",
    "NormalizedText",
    "PunctProfile",
    "PunktParameters",
    "PunktTokenizer",
    "PunktTrainer",
    "Segment",
    "detect_boilerplate",
    "extract_line_records",
    "get_default_punkt_tokenizer",
    "load_default_punkt_parameters",
    "normalize",
    "parse_enumerator",
    "segment_lines",
    "segment_paragraphs",
    "segment_paragraphs_simple",
    "segment_sentences",
]
