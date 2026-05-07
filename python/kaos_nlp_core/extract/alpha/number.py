"""AlphaNumberExtractor — rule-based numeric extraction.

Ports ``kelvin.nlp.extract.alpha.number.NumberExtractor`` onto the
WS-TR foundation. Three detection branches (in order of precedence):

1. **Arabic numbers** — tokens like ``"1234"``, ``"1,234"``,
   ``"1,234.56"``. Radix validation enforces 3-digit comma blocks for
   English locale (``"12,3"`` rejected; ``"1,234"`` accepted).
2. **Roman numerals** — ``"IV"``, ``"XIV"``, etc. Skips the bare letter
   ``"I"`` (too common as a pronoun / list marker).
3. **Written numbers** — ``"ten"``, ``"million"``, hyphenated forms like
   ``"twenty-three"``. Uses ``kaos_nlp_core.locale_data.WRITTEN_NUMBER_MAP``.

Returns :class:`decimal.Decimal` values (not ``float``) — precision-
preserving, serializes cleanly to SQL ``money`` / ``numeric`` columns,
avoids the IEEE-754 rounding trap.

Known algorithmic limitations (inherited from kelvin, documented so
downstream consumers know when to fall back to the LLM):

- **Hyphenated written numbers sum, they don't multiply**:
  ``"two-hundred"`` → ``Decimal('102')`` (2+100), not ``Decimal('200')``.
  Kelvin's normalizer is additive only. For multiplicative forms, the
  LLM or a downstream chunker must provide context.
- **No support for "and"**: ``"one hundred and five"`` → detects
  ``"one"`` (1), ``"hundred"`` (100), ``"five"`` (5) as three separate
  spans; no composition.
- **Bare ``"I"`` is always suppressed**: the Roman path skips it to
  avoid false positives on English first-person pronouns / list markers.
- **Non-English locales not implemented**: kelvin's German radix path is
  deferred to a future PR.

Divergences from kelvin's algorithm:

1. **Instance-based API.** Kelvin used ``@classmethod`` on all methods,
   making subclassing awkward. Our :class:`AlphaNumberExtractor` is
   instance-based with an instance-owned tokenizer so callers can swap
   in a different :class:`Tokenizer` configuration.
2. **``keep_punctuation=True`` tokenizer.** Kelvin's ``tokenize_regex``
   already stripped punctuation; kaos-nlp-core's tokenizer keeps it so
   we match the span-offset promise of the rest of the alpha stack
   (``AlphaSpan.start:end`` must round-trip with ``source_text[start:end]``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from decimal import ROUND_HALF_EVEN, Context, Decimal
from typing import ClassVar

from kaos_nlp_core.extract.base_extractor import (
    AlphaSpan,
    BaseAlphaExtractor,
    ExtractorValueType,
)
from kaos_nlp_core.locale_data import WRITTEN_NUMBER_MAP
from kaos_nlp_core.tokenizer import Tokenizer

# 10-digit precision matches kelvin's `Context(prec=10)`. Enough for any
# monetary amount up to USD 10 billion with 2-decimal precision.
_DECIMAL_CONTEXT = Context(prec=10, rounding=ROUND_HALF_EVEN)

# Punctuation we strip from tokens before numeric classification. Commas
# and periods are KEPT because they're meaningful in numbers (thousands
# separator, decimal point). Parentheses / brackets / quotes / terminal
# punctuation are stripped.
_STRIP_PUNCT = ",;:!?\"'()[]"

# Roman numeral regex — uppercase only, matches canonical form
# (I, IV, V, IX, X, XL, L, XC, C, CD, D, CM, M). We accept mixed-case
# callers by uppercasing before testing.
_ROMAN_RE = re.compile(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _parse_arabic_number(value: str) -> Decimal | None:
    """Parse an English-locale Arabic number token into Decimal.

    Enforces English radix rules: single period is the decimal mark;
    commas separate thousands and must appear in 3-digit blocks after
    the first block. Rejects ``"12,3"`` (2-digit block), accepts
    ``"1,234.56"`` and ``"1234"``.

    Returns ``None`` if the value doesn't parse cleanly — caller decides
    whether to try another branch.
    """
    if not value:
        return None

    # Must contain at least one digit; pure punctuation is not a number.
    if not any(c.isdigit() for c in value):
        return None

    # Split on the decimal mark first.
    period_count = value.count(".")
    if period_count > 1:
        return None
    pre_decimal = value.split(".")[0] if period_count == 1 else value

    # Validate comma positioning in the pre-decimal portion.
    radix_parts = pre_decimal.split(",")
    if len(radix_parts) > 1:
        # Every part after the first must be exactly 3 digits.
        for part in radix_parts[1:]:
            if len(part) != 3 or not part.isdigit():
                return None
        # The first part must be 1-3 digits.
        if len(radix_parts[0]) < 1 or len(radix_parts[0]) > 3:
            return None
        if not radix_parts[0].isdigit():
            return None

    # Strip commas, attempt Decimal construction.
    cleaned = value.replace(",", "")
    try:
        return Decimal(cleaned, context=_DECIMAL_CONTEXT)
    except (ValueError, TypeError, ArithmeticError):
        return None


def _is_roman_numeral(value: str) -> bool:
    """Return True if ``value`` matches the canonical Roman numeral form.

    Empty strings and non-Roman-letter tokens return False. The regex is
    anchored so ``"IVX"`` and ``"XIIII"`` (non-canonical) don't match.
    """
    if not value:
        return False
    upper = value.upper()
    if upper == "":
        return False
    return bool(_ROMAN_RE.match(upper))


def _roman_to_decimal(value: str) -> Decimal | None:
    """Convert a Roman numeral string into Decimal. Expects ``value`` to
    have been validated via :func:`_is_roman_numeral` first."""
    if not value:
        return None
    weights = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }
    total = 0
    prev = 0
    for ch in reversed(value.upper()):
        w = weights.get(ch)
        if w is None:
            return None
        if w < prev:
            total -= w
        else:
            total += w
            prev = w
    if total <= 0:
        return None
    return Decimal(total, context=_DECIMAL_CONTEXT)


def _parse_written_number(value: str, language_map: dict[str, int | str]) -> Decimal | None:
    """Parse a written-number token (``"ten"``, ``"twenty-three"``, etc.)
    into Decimal by summing the integer values of each hyphen-separated
    component.

    Non-integer entries in the map (e.g., ``"point": "."``) are ignored.
    Returns ``None`` if nothing in the token resolved to an integer.
    """
    if not value:
        return None

    if "-" in value:
        total = 0
        matched_any = False
        for part in value.split("-"):
            low = part.lower()
            if low == "and":
                continue
            inc = language_map.get(low)
            if isinstance(inc, int):
                total += inc
                matched_any = True
            else:
                # A component that doesn't resolve disqualifies the whole
                # token (e.g., "twenty-foo" is not a number).
                return None
        if not matched_any:
            return None
        return Decimal(total, context=_DECIMAL_CONTEXT)

    inc = language_map.get(value.lower())
    if isinstance(inc, int):
        return Decimal(inc, context=_DECIMAL_CONTEXT)
    return None


class AlphaNumberExtractor(BaseAlphaExtractor[Decimal]):
    """Rule-based numeric extractor.

    Emits :class:`AlphaSpan[Decimal]` for every Arabic / Roman / written
    number found in the input. Used standalone for ``INTEGER`` and
    ``FLOAT`` columns, and internally by :class:`AlphaMoneyExtractor` to
    parse the quantity portion of a money expression.

    The Decimal precision is 10 significant digits (matching kelvin);
    rounding mode is ``ROUND_HALF_EVEN`` (banker's rounding).
    """

    name: ClassVar[str] = "number"
    description: ClassVar[str] = "Rule-based numeric extraction (Arabic, Roman, written)"
    value_type: ClassVar[ExtractorValueType] = ExtractorValueType.NUMBER
    languages: ClassVar[tuple[str, ...]] = ("en",)

    def __init__(self, language: str = "en") -> None:
        super().__init__(language=language)
        self._tokenizer = Tokenizer(keep_punctuation=True)

    def parse_token(self, token: str) -> Decimal | None:
        """Parse a single already-tokenized string into a ``Decimal``.

        Returns ``None`` if the token isn't a recognized arabic / roman /
        written number. Audit perf finding #4 / P6 — exposes the
        per-token branch logic so callers (notably ``AlphaMoneyExtractor``)
        can avoid re-running the tokenizer on a single-token substring
        they already extracted.
        """
        if not token:
            return None
        stripped = token.strip(_STRIP_PUNCT)
        if not stripped:
            return None
        language_map = WRITTEN_NUMBER_MAP.get(self.language, {})

        # Branch 1: Arabic number.
        arabic = _parse_arabic_number(stripped)
        if arabic is not None:
            return arabic

        # Branch 2: Roman numeral (skip bare "I").
        if stripped.upper() != "I" and _is_roman_numeral(stripped):
            roman = _roman_to_decimal(stripped)
            if roman is not None:
                return roman

        # Branch 3: Written number.
        low = stripped.lower()
        if low in language_map or ("-" in stripped):
            return _parse_written_number(stripped, language_map)
        return None

    def extract_spans(self, text: str) -> Iterator[AlphaSpan[Decimal]]:
        """Yield :class:`AlphaSpan` objects for every numeric token.

        Spans point at the numeric token within the source text; the
        span's ``start:end`` round-trips with ``text[start:end]`` modulo
        stripped trailing punctuation.
        """
        for ts in self._tokenizer.tokenize(text):
            raw = ts.text
            if not raw:
                continue
            token = raw.strip(_STRIP_PUNCT)
            if not token:
                continue

            value = self.parse_token(raw)
            if value is None:
                continue

            # Compute the span for the stripped token — the tokenizer gave
            # us raw offsets; we shift the start forward by the number of
            # leading chars we stripped and shrink the end by trailing.
            lead_strip = len(raw) - len(raw.lstrip(_STRIP_PUNCT))
            trail_strip = len(raw) - len(raw.rstrip(_STRIP_PUNCT))
            yield AlphaSpan(
                value=value,
                start=ts.start + lead_strip,
                end=ts.end - trail_strip,
            )


__all__ = ["AlphaNumberExtractor"]
