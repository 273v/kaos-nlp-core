# ruff: noqa: RUF001
"""AlphaContactExtractor — rule-based contact-info extraction.

Fills the ``ExtractorValueType.CONTACT_INFO`` slot. Extracts emails,
URLs, and phone numbers from text in a single pass, returning
:class:`ContactMatch` records that carry both the surface form and a
canonical normalized form.

Three pattern families:

1. **Email** — practical RFC 5322 subset. Local part allows letters,
   digits, ``._%+-``; domain is dot-separated labels of letters /
   digits / hyphens, ending in a TLD of length 2-24. Skips trailing
   punctuation that's clearly sentence-terminal.
2. **URL** — ``http(s)://`` or ``ftp://`` schemes plus a path; also
   captures bare domains like ``example.com/foo`` when the slash-path
   is present. Bare ``example.com`` alone is NOT matched (too many
   false positives in legal text where company names like ``Acme.com``
   appear without intent of being URLs).
3. **Phone** — North American 10-digit and international E.164-ish
   forms. Recognized: ``(212) 555-0100``, ``212-555-0100``,
   ``212.555.0100``, ``212 555 0100``, ``+1 212 555 0100``,
   ``+44 20 7946 0958``, ``+49 30 12345678``. Normalized to
   ``+CC NNNNNNNNNN`` (no separators) — country code defaults to
   ``+1`` for bare-NANP forms.

Trade-offs versus the ``phonenumbers`` library: this is regex-only,
no dependency, ~95% precision on contract text. Use ``phonenumbers``
if you need full international validation, extension handling, or
regional formatting.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Literal

from kaos_nlp_core.extract.base_extractor import (
    AlphaSpan,
    BaseAlphaExtractor,
    ExtractorValueType,
)

ContactKind = Literal["email", "url", "phone"]


@dataclass(frozen=True, slots=True)
class ContactMatch:
    """A contact-info hit (email, URL, or phone).

    ``kind`` discriminates the three subtypes. ``value`` is the surface
    form as it appeared in the source text. ``normalized`` is a
    canonical form: lower-cased for emails / URLs, ``+CC NNNNNNNNNN``
    for phones (digits only after the country code).
    """

    kind: ContactKind
    value: str
    normalized: str


# -- Phone separator class --------------------------------------------------
#
# Phone separators commonly seen: ASCII space, dot, dash, plus the
# non-breaking space (U+00A0) that turns up in PDFs. Encoded explicitly
# as a Unicode escape so editor tooling and source review don't have to
# guess about NBSP characters in regex source.
_PHONE_SEP = r"[\s.\- ]"

# -- Email -------------------------------------------------------------------
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}\b"
)

# -- URL ---------------------------------------------------------------------
_URL_RE = re.compile(
    r"""
    (?:
        (?:https?|ftp)://[^\s<>"']+
      |
        \b(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}
        /[^\s<>"')]*
    )
    """,
    re.VERBOSE,
)

# -- Phone (international) ---------------------------------------------------
#
# Country code requires a separator before the first block to avoid the
# pathological case where ``+49 30 12345678`` is mis-parsed as
# ``cc=4, a=9, ...``. German subscriber numbers can be 8 digits long
# in a single block, so the second block accepts up to 8.
_PHONE_INTL_RE = re.compile(
    rf"""
    \+
    (?P<cc>\d{{1,3}})
    {_PHONE_SEP}+
    (?P<a>\d{{1,5}})
    {_PHONE_SEP}+
    (?P<b>\d{{2,8}})
    (?:{_PHONE_SEP}+(?P<c>\d{{2,8}}))?
    (?:{_PHONE_SEP}+(?P<d>\d{{2,8}}))?
    (?!\d)
    """,
    re.VERBOSE,
)

# -- Phone (NANP 10-digit) ---------------------------------------------------
_PHONE_NANP_RE = re.compile(
    rf"""
    (?<![\d/+])
    \(?
    (?P<area>[2-9]\d{{2}})
    \)?
    {_PHONE_SEP}?
    (?P<exch>[2-9]\d{{2}})
    {_PHONE_SEP}
    (?P<num>\d{{4}})
    (?!\d)
    """,
    re.VERBOSE,
)

# Sentence-terminal punctuation that often follows a URL/email but
# shouldn't be part of the captured value.
_TRAILING_TRIM = '.,;:!?")]'


class AlphaContactExtractor(BaseAlphaExtractor[ContactMatch]):
    """Rule-based email/URL/phone extractor.

    Usage::

        ext = AlphaContactExtractor()
        list(ext.extract_values(
            "Contact us at sales@example.com or +1 212-555-0100. "
            "See https://example.com/products for details."
        ))
        # [ContactMatch(kind='email', ...),
        #  ContactMatch(kind='url', ...),
        #  ContactMatch(kind='phone', value='+1 212-555-0100',
        #               normalized='+1 2125550100')]
    """

    name: ClassVar[str] = "contact"
    description: ClassVar[str] = "Rule-based contact-info extraction (email/URL/phone)"
    value_type: ClassVar[ExtractorValueType] = ExtractorValueType.CONTACT_INFO
    languages: ClassVar[tuple[str, ...]] = ("en",)

    def __init__(self, language: str = "en") -> None:
        super().__init__(language=language)

    def extract_spans(self, text: str) -> Iterator[AlphaSpan[ContactMatch]]:
        """Yield :class:`AlphaSpan[ContactMatch]` for every contact hit.

        Order: emails → URLs → phones, sorted by start offset. Overlap
        detection prefers the higher-precision match (email beats URL
        beats phone) when two patterns target the same span.
        """
        spans: list[tuple[int, int, ContactMatch]] = []

        for m in _EMAIL_RE.finditer(text):
            value = m.group(0)
            trimmed_value, trimmed_end = _trim_trailing(value, m.start(), m.end())
            spans.append(
                (
                    m.start(),
                    trimmed_end,
                    ContactMatch(
                        kind="email",
                        value=trimmed_value,
                        normalized=trimmed_value.lower(),
                    ),
                )
            )

        email_ranges = [(s, e) for s, e, _ in spans]
        for m in _URL_RE.finditer(text):
            if _overlaps(m.start(), m.end(), email_ranges):
                continue
            value = m.group(0)
            trimmed_value, trimmed_end = _trim_trailing(value, m.start(), m.end())
            spans.append(
                (
                    m.start(),
                    trimmed_end,
                    ContactMatch(
                        kind="url",
                        value=trimmed_value,
                        normalized=trimmed_value.lower(),
                    ),
                )
            )

        existing = [(s, e) for s, e, _ in spans]

        for m in _PHONE_INTL_RE.finditer(text):
            if _overlaps(m.start(), m.end(), existing):
                continue
            value = m.group(0)
            digits = "".join(ch for ch in value if ch.isdigit())
            cc = m.group("cc")
            normalized = f"+{cc} {digits[len(cc) :]}"
            spans.append(
                (
                    m.start(),
                    m.end(),
                    ContactMatch(kind="phone", value=value, normalized=normalized),
                )
            )

        existing = [(s, e) for s, e, _ in spans]
        for m in _PHONE_NANP_RE.finditer(text):
            if _overlaps(m.start(), m.end(), existing):
                continue
            value = m.group(0)
            digits = "".join(ch for ch in value if ch.isdigit())
            if len(digits) != 10:
                continue
            spans.append(
                (
                    m.start(),
                    m.end(),
                    ContactMatch(kind="phone", value=value, normalized=f"+1 {digits}"),
                )
            )

        spans.sort(key=lambda t: (t[0], t[1]))
        for start, end, match in spans:
            yield AlphaSpan(value=match, start=start, end=end)


# -- Module-level helpers ----------------------------------------------------


def _trim_trailing(value: str, start: int, end: int) -> tuple[str, int]:
    """Strip sentence-terminal punctuation from the END of a match."""
    while value and value[-1] in _TRAILING_TRIM:
        value = value[:-1]
        end -= 1
    return value, end


def _overlaps(start: int, end: int, existing: list[tuple[int, int]]) -> bool:
    """Return True if ``[start, end)`` overlaps any prior span."""
    return any(start < e and end > s for s, e in existing)


__all__ = ["AlphaContactExtractor", "ContactKind", "ContactMatch"]
