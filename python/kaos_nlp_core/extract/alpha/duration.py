"""AlphaDurationExtractor — rule-based duration extraction.

Ports ``kelvin.nlp.extract.alpha.duration.DurationExtractor`` onto the
WS-TR foundation. Emits :class:`DurationMatch` value objects carrying
both the human-readable ``(quantity, unit)`` pair and the canonical
total seconds (Decimal).

``"3 days"`` → ``DurationMatch(quantity=Decimal('3'), unit='days',
total_seconds=Decimal('259200'))``.
``"thirteen months"`` → ``DurationMatch(quantity=Decimal('13'),
unit='months', total_seconds=Decimal('33696000'))``.
``"a year"`` → ``DurationMatch(quantity=Decimal('1'), unit='year',
total_seconds=Decimal('31536000'))``.

The total_seconds value uses the calendar approximations from
:data:`DURATION_MAP`:
- month ≈ 30 days (2,592,000 s)
- year ≈ 365 days (31,536,000 s)
- anniversary = year

For contract-date arithmetic that needs calendar-accurate months/years,
consumers should use :mod:`dateutil.relativedelta` on the ``(quantity,
unit)`` pair. The Decimal seconds form is for comparison / sorting only.

Detection: token is in :data:`DURATION_MAP` (case-insensitive). Prior
token is examined for a numeric quantity (via :class:`AlphaNumberExtractor`)
or an indefinite article (``"a"``, ``"an"``, ``"the"``, ``"this"`` →
quantity 1).

Skipped modifier tokens: ``"business"``, ``"calendar"``, ``"working"``
don't trigger emission on their own (they are adjectives that precede a
real unit word like ``"business days"``).

Divergences from kelvin:

1. **Instance-based API.**
2. **Return a :class:`DurationMatch` value object** instead of
   :class:`dateutil.relativedelta`. Avoids adding a dateutil dependency
   and lets downstream consumers pick their own representation. Callers
   who want a relativedelta can construct it from ``(quantity, unit)``.
3. **Fractional quantities**: kelvin converts ``"1.5 hours"`` to
   ``relativedelta(minutes=90)``. We keep the fractional Decimal
   unchanged and let downstream consumers decide on rounding.
"""

from __future__ import annotations

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
from kaos_nlp_core.locale_data import DURATION_MAP
from kaos_nlp_core.tokenizer import Tokenizer

# Articles that imply a quantity of 1 when preceding a unit word.
_UNIT_ARTICLES: frozenset[str] = frozenset({"a", "an", "the", "this"})

# Adjectives that precede (not contain) a duration unit. They land on
# the unit-word token but shouldn't trigger emission — the real unit
# token follows.
_SKIP_MODIFIERS: frozenset[str] = frozenset({"business", "calendar", "working"})

_STRIP_PUNCT = ".,;:!?\"'()[]"


@dataclass(frozen=True, slots=True)
class DurationMatch:
    """A duration expression extracted by :class:`AlphaDurationExtractor`.

    - ``quantity`` — the scalar quantity as :class:`Decimal`.
    - ``unit`` — the unit word verbatim from the input, lowercased
      (e.g., ``"days"``, ``"month"``, ``"anniversary"``).
    - ``total_seconds`` — canonical seconds form using the calendar
      approximations in :data:`DURATION_MAP`. Use this for sorting or
      coarse comparison only.
    """

    quantity: Decimal
    unit: str
    total_seconds: Decimal


class AlphaDurationExtractor(BaseAlphaExtractor[DurationMatch]):
    """Rule-based duration extractor. Composes :class:`AlphaNumberExtractor`.

    Usage::

        ext = AlphaDurationExtractor()
        list(ext.extract_values("Termination notice period is 90 days."))
        # [DurationMatch(quantity=Decimal('90'), unit='days',
        #                total_seconds=Decimal('7776000'))]
    """

    name: ClassVar[str] = "duration"
    description: ClassVar[str] = "Rule-based duration extraction"
    value_type: ClassVar[ExtractorValueType] = ExtractorValueType.DURATION
    languages: ClassVar[tuple[str, ...]] = ("en",)

    def __init__(self, language: str = "en") -> None:
        super().__init__(language=language)
        if language not in DURATION_MAP:
            msg = (
                f"AlphaDurationExtractor: language {language!r} has no "
                f"DURATION_MAP entry. Supported: {sorted(DURATION_MAP.keys())}."
            )
            raise ValueError(msg)
        self._tokenizer = Tokenizer(keep_punctuation=True)
        self._number_extractor = AlphaNumberExtractor(language=language)

    def extract_spans(self, text: str) -> Iterator[AlphaSpan[DurationMatch]]:
        """Yield :class:`AlphaSpan[DurationMatch]` for every duration."""
        language_map = DURATION_MAP[self.language]
        tokens = list(self._tokenizer.tokenize(text))

        for i, ts in enumerate(tokens):
            raw = ts.text
            if not raw:
                continue
            token = raw.rstrip(_STRIP_PUNCT)
            if not token:
                continue

            low = token.lower()

            # Skip adjectival modifiers — they precede the real unit.
            if low in _SKIP_MODIFIERS:
                continue

            # Must be a unit word.
            if low not in language_map:
                continue
            if i == 0:
                continue

            prior = tokens[i - 1]
            prior_raw = prior.text or ""
            prior_token = prior_raw.rstrip(_STRIP_PUNCT)
            if not prior_token:
                continue

            prior_low = prior_token.lower()

            # Indefinite-article path: "a day" / "this month" → qty=1.
            if prior_low in _UNIT_ARTICLES:
                quantity = Decimal(1)
                start = prior.start
            else:
                # Numeric path.
                quantity = self._number_extractor.parse_token(prior_token)
                if quantity is None:
                    continue
                start = prior.start

            seconds_per = Decimal(language_map[low])
            total_seconds = quantity * seconds_per
            yield AlphaSpan(
                value=DurationMatch(
                    quantity=quantity,
                    unit=low,
                    total_seconds=total_seconds,
                ),
                start=start,
                end=ts.end,
            )


__all__ = ["AlphaDurationExtractor", "DurationMatch"]
