"""AlphaQuantityExtractor — rule-based number-with-units extraction.

Fills the ``ExtractorValueType.NUMBER_WITH_UNITS`` slot. Composes
:class:`AlphaNumberExtractor` for the numeric side and the
:data:`UNIT_MAP` gazetteer for unit recognition.

Two detection branches:

1. **Word-unit form** — a unit token (``"kg"``, ``"meters"``,
   ``"acres"``) preceded by a quantity. ``5 kg``, ``2.5 meters``,
   ``twenty acres``.
2. **Symbol-suffix form** — degree-Celsius / degree-Fahrenheit fused
   onto a number: ``20°C``, ``68°F``, ``98.6°F``. Also ``20C`` /
   ``68F`` (no degree sign), with case-insensitive suffix.

Returns :class:`QuantityMatch(amount, unit, dimension)` records:

- ``amount`` — :class:`decimal.Decimal` (preserves precision).
- ``unit`` — canonical form from the gazetteer (``"kg"``, ``"m^2"``,
  ``"L"``, ``"°C"``).
- ``dimension`` — one of the strings in
  :data:`kaos_nlp_core.locale_data.UNIT_DIMENSIONS` (``"mass"``,
  ``"length"``, ``"area"``, etc.).

Out of scope: composite units (``kg/m^3``, ``N·m``), unit conversion,
range expressions (``5-10 kg``). Use a dedicated units library (Pint,
unyt) for those.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from kaos_nlp_core.extract.alpha.number import AlphaNumberExtractor
from kaos_nlp_core.extract.base_extractor import (
    AlphaSpan,
    BaseAlphaExtractor,
    ExtractorValueType,
)
from kaos_nlp_core.locale_data import UNIT_MAP
from kaos_nlp_core.tokenizer import Tokenizer

_STRIP_PUNCT = ".,;:!?\"'()[]"

# Degree-symbol-fused temperature: 20°C, 98.6°F. We require the actual
# degree sign — bare-letter forms ``20C`` / ``68F`` were dropped after
# false positives like ``Section 21F`` (SEC rule reference) and
# ``Section 5C`` got parsed as Fahrenheit / Celsius. The degree sign is
# universal in formal documents that mean temperature.
_TEMP_FUSED_RE = re.compile(
    r"^(?P<num>-?\d+(?:\.\d+)?)°(?P<scale>[CF])$",
    re.IGNORECASE,
)

# A token that looks like a section / enumerator label, not a quantity.
# Conservative rules — false-rejecting a real decimal like "2.5" hurts
# more than missing a section header like "10.03 Notes". So we only
# match clearly-structural forms:
#   (1), (a), (iii), (c)         — parenthesized enumerator
#   i., ii., iv., XII.           — Roman with trailing dot
#   10.03.    or  10.3.4.        — decimal section WITH trailing dot
#                                   (a decimal alone is ambiguous —
#                                   "2.5 meters" is a real quantity)
_SECTION_NUMBER_RE = re.compile(
    r"""
    ^(?:
        \(\w+\)                       # parenthesized: (1), (a), (iii)
      |
        [ivxlcdmIVXLCDM]+\.           # Roman with trailing dot: ii. iii.
      |
        \d+(?:\.\d+){1,3}\.           # decimal section WITH trailing dot
    )$
    """,
    re.VERBOSE,
)

# Token shape acceptable for COUNT-dimension prior values: integer or
# integer-with-thousands-separator (``5``, ``10,000``). Decimals are
# rejected because counts of shares / warrants / pallets are integer
# in practice, and decimal-shaped tokens like "10.03" are usually
# section labels.
_COUNT_INTEGER_RE = re.compile(r"^[\d,]+$")

# Pure-Roman-numeral tokens (M, MM, MMI, IV, etc.). When the source
# value of a count-dimensioned quantity comes from a Roman-numeral
# parse, it's almost always a section enumerator rather than a real
# count.
_PURE_ROMAN_RE = re.compile(r"^[ivxlcdmIVXLCDM]+$")

# "Count" units have weak semantic signal — they accept almost any
# noun-like word. We apply the Roman-numeral / section-number rejection
# only to count units; physical units (kg, ft, MMcf) are unambiguous
# enough that a Roman-numeral source is fine ("MMI kg" is genuinely 2001
# kg if it appears in the wild).
_COUNT_DIMENSION = "count"


@dataclass(frozen=True, slots=True)
class QuantityMatch:
    """A number-with-units expression.

    - ``amount`` — :class:`Decimal` quantity. Preserves precision.
    - ``unit`` — canonical form (e.g., ``"kg"`` for "kilograms",
      ``"m^2"`` for "square meters", ``"°C"`` for "20°C").
    - ``dimension`` — one of :data:`UNIT_DIMENSIONS`.
    """

    amount: Decimal
    unit: str
    dimension: str


class AlphaQuantityExtractor(BaseAlphaExtractor[QuantityMatch]):
    """Rule-based quantity extractor (number + unit).

    Usage::

        ext = AlphaQuantityExtractor()
        list(ext.extract_values("Cargo weighs 5 kg and spans 2.5 meters."))
        # [QuantityMatch(amount=Decimal('5'), unit='kg', dimension='mass'),
        #  QuantityMatch(amount=Decimal('2.5'), unit='m', dimension='length')]
    """

    name: ClassVar[str] = "quantity"
    description: ClassVar[str] = "Rule-based number-with-units extraction (SI + imperial)"
    value_type: ClassVar[ExtractorValueType] = ExtractorValueType.NUMBER_WITH_UNITS
    languages: ClassVar[tuple[str, ...]] = ("en",)

    def __init__(self, language: str = "en") -> None:
        super().__init__(language=language)
        if language not in UNIT_MAP:
            valid = sorted(UNIT_MAP.keys())
            msg = (
                f"AlphaQuantityExtractor: language {language!r} has no "
                f"UNIT_MAP entry. Supported: {valid}."
            )
            raise ValueError(msg)
        self._tokenizer = Tokenizer(keep_punctuation=True)
        self._number_extractor = AlphaNumberExtractor(language=language)

    def extract_spans(self, text: str) -> Iterator[AlphaSpan[QuantityMatch]]:
        """Yield :class:`AlphaSpan[QuantityMatch]` for every quantity.

        Detection runs in this order at each token:

        1. Fused suffix (``"20°C"``, ``"68F"``).
        2. **Multi-token unit** (``"square feet"``, ``"cubic yards"``,
           ``"linear meters"``). Tries the lower-cased
           ``prior + " " + current`` key against the gazetteer; if it
           hits, the quantity comes from ``tokens[i-2]``.
        3. Single-token unit (``"5 kg"``, ``"100 mph"``).

        Multi-token forms beat single-token forms when both could match
        (otherwise ``"5 square feet"`` would emit ``"square"`` as the
        anchor and ``"feet"`` as a length unit, which is wrong).
        """
        unit_lookup = UNIT_MAP[self.language]
        tokens = list(self._tokenizer.tokenize(text))

        # Track which token indices already produced a multi-token match
        # so the single-token branch doesn't double-emit on the trailing
        # word (e.g., the "feet" in "5 square feet").
        consumed_in_multi: set[int] = set()

        for i, ts in enumerate(tokens):
            if i in consumed_in_multi:
                continue
            raw = ts.text
            if not raw:
                continue
            token = raw.rstrip(_STRIP_PUNCT)
            if not token:
                continue

            # 1. Fused suffix (degree-Celsius/Fahrenheit).
            fused = self._try_fused_temperature(token, ts)
            if fused is not None:
                yield fused
                continue

            low = token.lower()

            # 2. Multi-token unit: try (prior_token + " " + current).
            if i >= 2:
                prior = tokens[i - 1]
                prior_raw = (prior.text or "").rstrip(_STRIP_PUNCT)
                if prior_raw:
                    compound_key = f"{prior_raw.lower()} {low}"
                    multi = unit_lookup.get(compound_key)
                    if multi is not None:
                        quantity_token = tokens[i - 2]
                        qty_raw = (quantity_token.text or "").rstrip(_STRIP_PUNCT)
                        canonical, dimension = multi
                        if _is_section_marker(qty_raw):
                            continue
                        quantity = (
                            self._number_extractor.extract_first_value(qty_raw) if qty_raw else None
                        )
                        if (
                            quantity is not None
                            and not _is_count_from_roman(qty_raw, dimension)
                            and _count_prior_is_acceptable(qty_raw, dimension)
                        ):
                            yield AlphaSpan(
                                value=QuantityMatch(
                                    amount=quantity,
                                    unit=canonical,
                                    dimension=dimension,
                                ),
                                start=quantity_token.start,
                                end=ts.end,
                            )
                            consumed_in_multi.add(i - 1)
                            consumed_in_multi.add(i)
                            continue

            # 3. Single-token unit. Lookup is case-insensitive.
            unit_entry = unit_lookup.get(low)
            if unit_entry is None:
                continue
            if i == 0:
                continue

            canonical, dimension = unit_entry
            prior = tokens[i - 1]
            prior_raw = (prior.text or "").rstrip(_STRIP_PUNCT)
            if not prior_raw:
                continue

            # Reject section-marker-shaped prior tokens — they're document
            # scaffolding ("10.03 Notes" → 10.03 notes is a false positive).
            if _is_section_marker(prior_raw):
                continue

            quantity = self._number_extractor.extract_first_value(prior_raw)
            if quantity is None:
                continue

            # Reject Roman-numeral source values for count-dimension units
            # (these are nearly always section enumerators).
            if _is_count_from_roman(prior_raw, dimension):
                continue

            # Reject decimal-shaped priors for count-dimension units
            # (shares / warrants are integer-valued; "10.03 Notes" is a
            # section, not a count).
            if not _count_prior_is_acceptable(prior_raw, dimension):
                continue

            yield AlphaSpan(
                value=QuantityMatch(
                    amount=quantity,
                    unit=canonical,
                    dimension=dimension,
                ),
                start=prior.start,
                end=ts.end,
            )

    # -- Internals -----------------------------------------------------

    def _try_fused_temperature(self, token: str, ts):
        return _try_fused_temperature_impl(token, ts)


# -- Module-level helpers ---------------------------------------------------


def _is_section_marker(raw: str) -> bool:
    """Heuristic: does this token look like a section / enumerator label?

    Catches ``10.03``, ``2.4``, ``(1)``, ``(iii)``, ``(c)``, ``i.``,
    ``ii.``. Used to reject prior-token candidates that are document
    scaffolding rather than real quantities.
    """
    return bool(_SECTION_NUMBER_RE.match(raw))


def _is_count_from_roman(raw: str, dimension: str) -> bool:
    """Reject Roman-numeral source values for count units.

    ``MMI warrants`` parses to ``2001 warrants`` because MMI is a valid
    Roman numeral, but in practice MMI is always a section heading
    (year reference, list enumerator). The same is true for other
    pure-Roman tokens. We allow Roman sources for unambiguous physical
    units where the meaning is clear from context.
    """
    return dimension == _COUNT_DIMENSION and bool(_PURE_ROMAN_RE.match(raw))


def _count_prior_is_acceptable(raw: str, dimension: str) -> bool:
    """For count-dimension matches, require an integer-shaped prior.

    Drops decimal-shaped tokens like ``"10.03"`` and ``"0.5"`` from
    matching count units (shares / warrants / pallets are integer in
    practice). Physical units accept any well-formed Decimal source.
    """
    if dimension != _COUNT_DIMENSION:
        return True
    return bool(_COUNT_INTEGER_RE.match(raw))


def _try_fused_temperature_impl(token: str, ts) -> AlphaSpan[QuantityMatch] | None:
    """Parse temperature literals fused with °C / °F (or bare C/F).

    Bare ``"20C"`` is intentionally accepted because contracts sometimes
    drop the degree sign. The downside: a one-token ``"5C"`` could be
    mistaken for "5 Celsius" when the author meant "5C" as a section
    identifier. The risk is small — typical legal section refs are
    ``"5(c)"`` or ``"§5C"`` — and the value of catching ``"68°F"`` and
    ``"68F"`` is high.
    """
    m = _TEMP_FUSED_RE.match(token)
    if not m:
        return None
    try:
        amount = Decimal(m.group("num"))
    except Exception:
        return None
    scale = m.group("scale").upper()
    unit = "°C" if scale == "C" else "°F"
    return AlphaSpan(
        value=QuantityMatch(amount=amount, unit=unit, dimension="temperature"),
        start=ts.start,
        end=ts.end,
    )


__all__ = ["AlphaQuantityExtractor", "QuantityMatch"]
