"""AlphaTimeExtractor — rule-based time-of-day extraction.

Fills the ``ExtractorValueType.TIME`` slot. Emits :class:`datetime.time`
values with character offsets into the source text.

Detection branches:

1. **Colon-delimited HH:MM[:SS]** — ``09:00``, ``23:45``, ``3:30:15``,
   optionally followed by an AM/PM marker (``"PM"``, ``"P.M."``,
   ``"a.m."``, ``"am"``).
2. **Bare hour + AM/PM** — ``9 AM``, ``5 P.M.``, ``11 a.m.``,
   ``"12 noon"`` (the noon/midnight pair count as marker words).
3. **Word-only** — bare ``"noon"`` (12:00) or ``"midnight"`` (00:00),
   not preceded by a number.

Validation:

- Hour must be in ``[0, 23]``. AM-marked ``12`` rolls to ``00`` (12:30
  AM = 00:30); PM-marked ``12`` stays at ``12`` (12:30 PM = 12:30).
  Hours ``13..23`` with an AM/PM marker are rejected.
- Minutes must be in ``[0, 59]``; seconds (when present) in ``[0, 59]``.

The emitted ``AlphaSpan.value`` is ``datetime.time(hour, minute, second)``
— naive, no timezone. Timezone tokens like ``EST`` / ``UTC`` immediately
following the time are not captured by this extractor; a downstream
``DateTimeExtractor`` (planned) is the right home for fusing date + time
+ tz into a single :class:`datetime.datetime`.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from decimal import Decimal
from typing import ClassVar

from kaos_nlp_core.extract.alpha.number import AlphaNumberExtractor
from kaos_nlp_core.extract.base_extractor import (
    AlphaSpan,
    BaseAlphaExtractor,
    ExtractorValueType,
)
from kaos_nlp_core.tokenizer import Tokenizer

_STRIP_PUNCT = ".,;:!?\"'()[]"

# AM/PM markers (lower-cased, with internal periods stripped). The
# tokenizer treats "a.m." as one token; we lowercase it and strip
# trailing punctuation, so the comparison key here is "a.m" or "am".
_AM_MARKERS: frozenset[str] = frozenset({"am", "a.m"})
_PM_MARKERS: frozenset[str] = frozenset({"pm", "p.m"})

# Word-only times. "noon" → 12:00, "midnight" → 00:00. Also accepted
# after a quantity ("12 noon" → 12:00 PM-equivalent; "12 midnight" → 00:00).
_NOON_WORDS: frozenset[str] = frozenset({"noon"})
_MIDNIGHT_WORDS: frozenset[str] = frozenset({"midnight"})

# Decimal-12, kept as a constant for the hour-rolling rule.
_TWELVE = Decimal(12)


class AlphaTimeExtractor(BaseAlphaExtractor[datetime.time]):
    """Rule-based time-of-day extractor.

    Usage::

        ext = AlphaTimeExtractor()
        list(ext.extract_values("Delivery by 5 PM EST."))
        # [datetime.time(17, 0)]
        list(ext.extract_values("The meeting starts at 09:30."))
        # [datetime.time(9, 30)]
        list(ext.extract_values("Effective at noon."))
        # [datetime.time(12, 0)]

    Composes with :class:`AlphaDateExtractor` for combined date+time
    detection: a downstream ``DateTimeExtractor`` (not yet shipped) will
    fuse adjacent date/time spans and surface a :class:`datetime.datetime`.
    """

    name: ClassVar[str] = "time"
    description: ClassVar[str] = "Rule-based time-of-day extraction"
    value_type: ClassVar[ExtractorValueType] = ExtractorValueType.TIME
    languages: ClassVar[tuple[str, ...]] = ("en",)

    def __init__(self, language: str = "en") -> None:
        super().__init__(language=language)
        self._tokenizer = Tokenizer(keep_punctuation=True)
        self._number_extractor = AlphaNumberExtractor(language=language)

    def extract_spans(self, text: str) -> Iterator[AlphaSpan[datetime.time]]:
        """Yield :class:`AlphaSpan[datetime.time]` for every time expression."""
        tokens = list(self._tokenizer.tokenize(text))

        for i, ts in enumerate(tokens):
            raw = ts.text
            if not raw:
                continue
            token = raw.rstrip(_STRIP_PUNCT)
            if not token:
                continue

            # Branch 1: colon-delimited HH:MM[:SS], optionally with marker.
            if ":" in token:
                emitted = self._try_colon_form(token, ts, tokens, i)
                if emitted is not None:
                    yield emitted
                    continue

            # Branch 2: bare quantity + AM/PM marker (handled at the
            # marker token, looking back).
            low = token.lower().rstrip(".")
            if low in _AM_MARKERS or low in _PM_MARKERS:
                emitted = self._try_bare_marker_form(token, ts, tokens, i)
                if emitted is not None:
                    yield emitted
                    continue

            # Branch 3: word-only "noon" / "midnight" with no preceding
            # quantity (the "12 noon" form is handled by branch 2).
            if low in _NOON_WORDS:
                # Skip if preceded by a quantity — branch 2 will pick it up.
                if not _has_preceding_number(tokens, i, self._number_extractor):
                    yield AlphaSpan(
                        value=datetime.time(12, 0),
                        start=ts.start,
                        end=ts.end,
                    )
                continue
            if low in _MIDNIGHT_WORDS:
                if not _has_preceding_number(tokens, i, self._number_extractor):
                    yield AlphaSpan(
                        value=datetime.time(0, 0),
                        start=ts.start,
                        end=ts.end,
                    )
                continue

    # -- Internals -----------------------------------------------------

    def _try_colon_form(
        self,
        token: str,
        ts,
        tokens: list,
        i: int,
    ) -> AlphaSpan[datetime.time] | None:
        """Parse ``HH:MM[:SS]`` with an optional trailing AM/PM marker."""
        parts = token.split(":")
        if len(parts) not in (2, 3):
            return None
        try:
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) == 3 else 0
        except ValueError:
            return None
        if not (0 <= minute <= 59 and 0 <= second <= 59):
            return None

        # Look ahead for an AM/PM marker token. The tokenizer keeps "PM"
        # as a separate token from the colon-form; tokens with internal
        # periods like "p.m." are also separate.
        end = ts.end
        marker = _peek_marker(tokens, i + 1)
        if marker is not None:
            kind, marker_end = marker
            end = marker_end
            hour = _apply_meridian(hour, kind)
            if hour is None:
                return None
        elif not (0 <= hour <= 23):
            return None

        try:
            t = datetime.time(hour, minute, second)
        except ValueError:
            return None
        return AlphaSpan(value=t, start=ts.start, end=end)

    def _try_bare_marker_form(
        self,
        token: str,
        ts,
        tokens: list,
        i: int,
    ) -> AlphaSpan[datetime.time] | None:
        """Parse ``HOUR <am|pm|noon|midnight>``: looking back for a number."""
        if i == 0:
            return None
        low = token.lower().rstrip(".")
        kind = "am" if low in _AM_MARKERS else "pm"

        prior = tokens[i - 1]
        prior_raw = (prior.text or "").rstrip(_STRIP_PUNCT)
        if not prior_raw:
            return None

        # The prior token may itself be a colon-form ("9:30 PM") — the
        # colon branch handles those. Here we only fire on bare-hour
        # forms ("9 PM", "12 PM").
        if ":" in prior_raw:
            return None

        quantity = self._number_extractor.extract_first_value(prior_raw)
        if quantity is None:
            return None
        try:
            hour = int(quantity)
        except (ValueError, OverflowError):
            return None
        if hour != quantity:
            # Reject fractional hours like "5.5 PM".
            return None
        if not (0 <= hour <= 23):
            return None

        hour_resolved = _apply_meridian(hour, kind)
        if hour_resolved is None:
            return None

        try:
            t = datetime.time(hour_resolved, 0)
        except ValueError:
            return None
        return AlphaSpan(value=t, start=prior.start, end=ts.end)


# -- Module-level helpers ---------------------------------------------------


def _peek_marker(tokens: list, idx: int) -> tuple[str, int] | None:
    """Look at ``tokens[idx]`` for an AM/PM/noon/midnight marker.

    Returns ``(kind, end_offset)`` where ``kind`` is ``"am"`` or
    ``"pm"``, or ``None`` if the next token isn't a marker.
    """
    if idx >= len(tokens):
        return None
    next_ts = tokens[idx]
    raw = (next_ts.text or "").rstrip(_STRIP_PUNCT)
    if not raw:
        return None
    low = raw.lower().rstrip(".")
    if low in _AM_MARKERS:
        return ("am", next_ts.end)
    if low in _PM_MARKERS:
        return ("pm", next_ts.end)
    if low in _NOON_WORDS:
        # "12:00 noon" is functionally a PM-marker for the 12 case.
        return ("noon", next_ts.end)
    if low in _MIDNIGHT_WORDS:
        return ("midnight", next_ts.end)
    return None


def _apply_meridian(hour: int, kind: str) -> int | None:
    """Apply AM/PM/noon/midnight to a 12-hour-format hour.

    Returns the 24-hour-format hour, or ``None`` if the combination is
    invalid (e.g., hour 13 with an AM marker).
    """
    if kind == "noon":
        # Only 12 noon is meaningful.
        return 12 if hour == 12 else None
    if kind == "midnight":
        return 0 if hour == 12 else None
    if kind == "am":
        if not (1 <= hour <= 12):
            return None
        return 0 if hour == 12 else hour
    if kind == "pm":
        if not (1 <= hour <= 12):
            return None
        return 12 if hour == 12 else hour + 12
    return None


def _has_preceding_number(tokens: list, idx: int, extractor: AlphaNumberExtractor) -> bool:
    """Check whether ``tokens[idx-1]`` is a quantity. Used to avoid
    double-emitting ``"12 noon"`` from branch 2 *and* branch 3."""
    if idx == 0:
        return False
    prior = tokens[idx - 1]
    prior_raw = (prior.text or "").rstrip(_STRIP_PUNCT)
    if not prior_raw:
        return False
    return extractor.extract_first_value(prior_raw) is not None


__all__ = ["AlphaTimeExtractor"]
