"""AlphaDateExtractor — rule-based date extraction.

Ports ``kelvin.nlp.extract.alpha.date.DateExtractor`` onto the WS-TR
foundation: consumes :class:`kaos_nlp_core.tokenizer.Tokenizer` output
for tokenization + char-span tracking, reads
:data:`kaos_nlp_core.locale_data.MONTH_MAP` and
:data:`kaos_nlp_core.locale_data.ORDINAL_MAP` for gazetteer lookups,
emits :class:`kaos_nlp_core.extract.base_extractor.AlphaSpan` tuples
keyed by :class:`datetime.datetime`.

The algorithm mirrors kelvin's implementation faithfully — same
five-branch detection (inline separator, DD Month YYYY, Ordinal Month
YYYY, Month DD YYYY, Month Ordinal YYYY) plus the English-only "Nth day
of Month YYYY" form. The 2-digit-year century resolver and
``min_year``/``max_year`` bounds check are preserved verbatim.

The span returned ALWAYS points to the full matched token range in the
source text — ``source_text[span.start:span.end]`` reconstructs the
verbatim date string. This matters for the PR-6f.3 merger: the span we
emit is the same span the LLM would cite.

Design notes:

- **No regex.** Follows kelvin: sequential token-by-token branches on
  shape (``isdigit()``, ``"/" in token``). Regex-free because the
  branches are small and the token stream is short; a regex would just
  add cognitive overhead.

- **Uses the existing kaos-nlp-core ``Tokenizer``** with
  ``keep_punctuation=True``. Char offsets, not byte offsets. The
  ``strip_punctuation`` helper in this file handles ASCII punctuation
  via ``str.strip()``.

- **Refuses to emit ``None`` values.** Kelvin yields ``(None, start,
  end)`` when normalization fails inside a branch. We drop those. The
  merger downstream only cares about extractions that produced a real
  date; a bare span with no value is meaningless.

- **Validates calendar dates.** Python's ``datetime.datetime(year,
  month, day)`` raises ``ValueError`` for invalid combinations (Feb 30,
  etc.). We catch and skip.

Usage::

    extractor = AlphaDateExtractor()
    for span in extractor.extract_spans(
        "Agreement made as of February 17, 1999, between Acme and Beta."
    ):
        print(span.value.date(), span.start, span.end)
    # datetime.date(1999, 2, 17) 21 38
"""

from __future__ import annotations

import datetime
import string
from collections.abc import Iterator
from typing import ClassVar

from kaos_nlp_core.extract.base_extractor import (
    AlphaSpan,
    BaseAlphaExtractor,
    ExtractorValueType,
)
from kaos_nlp_core.locale_data import MONTH_MAP, ORDINAL_MAP
from kaos_nlp_core.tokenizer import Tokenizer

# Punctuation strip set. kelvin uses Unicode-category-aware stripping;
# the date extractor only ever hits ASCII punctuation in practice
# (dates don't contain curly quotes, em-dashes, etc.), so the simpler
# ASCII set gives us zero false strips and zero missed strips.
_STRIP_PUNCT = string.punctuation


def _strip(token: str) -> str:
    """Strip leading/trailing ASCII punctuation from a token.

    This is deliberately simpler than kelvin's Unicode-mapper-driven
    ``strip_punctuation`` — the date extractor only branches on
    ``isdigit()`` / gazetteer lookup, which never interacts with
    non-ASCII punctuation. If future extractors need richer punctuation
    handling, hoist this into ``kaos_nlp_core`` and make it Unicode-aware.
    """
    return token.strip(_STRIP_PUNCT)


class AlphaDateExtractor(BaseAlphaExtractor[datetime.datetime]):
    """Rule-based date extractor — LLM-free, regex-free, deterministic.

    Handles these patterns in English contract text:

    - ``MM/DD/YYYY`` / ``MM-DD-YYYY`` / ``MM.DD.YYYY`` (inline separator)
    - ``Jan/1/21`` / ``1/Jan/21`` (month word inside a slashed date)
    - ``DD Month YYYY`` (``19 May 2010``)
    - ``Ordinal Month YYYY`` (``first May 2010``, ``1st May 2010``)
    - ``Month DD YYYY`` (``May 19, 2010``)
    - ``Month Ordinal YYYY`` (``May first, 2010``)
    - ``Ordinal day of Month YYYY`` (``first day of May 2010``) — English only

    Years outside ``[min_year, max_year]`` = ``[1900, 2050]`` are
    rejected in ``strict=True`` mode; in non-strict mode (default), a
    2-digit year picks the closer of the current or prior century.

    February 30 and other invalid-by-calendar dates are rejected.

    The emitted ``AlphaSpan`` carries ``datetime.datetime`` (midnight
    UTC, naive). Callers that want ``datetime.date`` should use
    ``span.value.date()``.
    """

    name: ClassVar[str] = "date"
    description: ClassVar[str] = "Rule-based date extraction"
    value_type: ClassVar[ExtractorValueType] = ExtractorValueType.DATE_FULL
    languages: ClassVar[tuple[str, ...]] = ("en",)

    min_year: ClassVar[int] = 1900
    max_year: ClassVar[int] = 2050

    def __init__(self, language: str = "en") -> None:
        super().__init__(language=language)
        # Cache gazetteers + tokenizer to avoid lookup on every call.
        self._month_map = MONTH_MAP.get(language, MONTH_MAP["en"])
        self._ordinal_map = ORDINAL_MAP.get(language, ORDINAL_MAP["en"])
        self._tokenizer = Tokenizer(keep_punctuation=True)

    # -- Normalization (ported verbatim from kelvin) -----------------------

    def normalize_date_tokens(
        self,
        date_tokens: list[str],
        strict: bool = False,
        default_date: datetime.datetime | None = None,
    ) -> datetime.datetime | None:
        """Resolve an ambiguous 3-token date into a datetime.

        Algorithm ported from kelvin-nlp date.py:92-277. See that file
        for the full case-analysis walkthrough; the key moves are:

        1. Pick the year token (4-digit preferred; else 2-digit > 31).
        2. If year at position 0 → ``YYYY-MM-DD``; else search for day
           token (2-digit in 12..31) and assign month to whatever's left.
        3. Fall back to locale default (English: ``MM/DD/YYYY`` unless
           year is first).
        4. Expand 2-digit years by picking the closer of the current or
           prior century.
        5. Validate month (1..12) and day (1..31).
        6. Build ``datetime.datetime``; catch ``ValueError`` on invalid
           combinations (e.g., Feb 30).
        """
        if len(date_tokens) != 3:
            return None

        year_token: int | None = None
        month_token: int | None = None
        day_token: int | None = None

        # -- Step 1: find the year token -----------------------------
        for i, token in enumerate(date_tokens):
            if len(token) == 4 and token.isdigit():
                # Year cannot be in the middle of the 3-token tuple.
                if i == 1:
                    return None
                year_token = i
                break
            # 2-digit > 31 is unambiguously a year (no day can be > 31).
            if len(token) == 2 and token.isdigit() and not strict and int(token) > 31:
                year_token = i
                break

        # -- Step 2: year at position 0 → assume YYYY-MM-DD -----------
        if year_token == 0:
            month_token = 1
            day_token = 2
        else:
            # -- Step 3: find day token by being > 12 ----------------
            for i, token in enumerate(date_tokens):
                if i == year_token:
                    continue
                if len(token) == 2 and token.isdigit() and 12 < int(token) < 31:
                    day_token = i
                    break
            # If we have year + day, month is whichever slot is left.
            if year_token is not None and day_token is not None:
                month_token = (
                    0
                    if 0 not in (year_token, day_token)
                    else 1
                    if 1 not in (year_token, day_token)
                    else None
                )
                if month_token is None:
                    return None

        # -- Step 4: locale-default fallback -------------------------
        if month_token is None or day_token is None:
            if self.language in ("en",):
                # English: MM/DD unless year is first.
                if year_token == 0:
                    month_token = 1
                    day_token = 2
                else:
                    month_token = 0
                    day_token = 1
            else:
                return None

        # -- Step 5: 2-digit year inference --------------------------
        if not strict and month_token is not None and day_token is not None:
            for i, token in enumerate(date_tokens):
                if i not in (month_token, day_token) and len(token) == 2 and token.isdigit():
                    year_token = i
                    break

        # -- Step 6: parse the year value ---------------------------
        year_value: int
        if year_token is None:
            if default_date is None:
                return None
            year_value = default_date.year
        else:
            year_value = int(date_tokens[year_token])
            if year_value < self.min_year or year_value > self.max_year:
                if not strict:
                    # Pick the closer of current or prior century.
                    current_year = datetime.datetime.now().year
                    current_century = current_year // 100
                    current_century_value = year_value + (current_century * 100)
                    prior_century_value = year_value + ((current_century - 1) * 100)
                    if abs(prior_century_value - current_year) < abs(
                        current_century_value - current_year
                    ):
                        year_value = prior_century_value
                    else:
                        year_value = current_century_value
                else:
                    return None

        # -- Step 7: parse the month value --------------------------
        month_value: int | None
        if month_token is None:
            if default_date is None:
                return None
            month_value = default_date.month
        else:
            month_tok = date_tokens[month_token]
            if month_tok.isdigit():
                month_value = int(month_tok)
            elif month_tok.lower() in self._month_map:
                month_value = self._month_map[month_tok.lower()]
            else:
                month_value = None
            if not month_value or month_value < 1 or month_value > 12:
                return None

        # -- Step 8: parse the day value ----------------------------
        day_value: int
        if day_token is None:
            if default_date is None:
                return None
            day_value = default_date.day
        else:
            try:
                day_value = int(date_tokens[day_token])
            except ValueError:
                return None
            if day_value < 1 or day_value > 31:
                return None

        # -- Step 9: final year-bounds check + calendar validation --
        if year_value < self.min_year or year_value > self.max_year:
            return None
        try:
            return datetime.datetime(year=year_value, month=month_value, day=day_value)
        except ValueError:
            # Feb 30, etc.
            return None

    # -- Extraction (ported from kelvin date.py:280-494) ------------------

    def extract_spans(
        self,
        text: str,
        default_date: datetime.datetime | None = None,
    ) -> Iterator[AlphaSpan[datetime.datetime]]:
        """Yield every date found in ``text`` as an ``AlphaSpan``.

        Runs five detection branches per token-in-context:

        1. Inline separator (``MM/DD/YYYY``).
        2. ``DD Month YYYY`` (digit, month, digit).
        3. ``Ordinal Month YYYY`` (``first`` / ``1st``, month, digit).
        4. ``Month DD YYYY`` (month, digit, digit).
        5. ``Month Ordinal YYYY`` (month, ordinal, digit).
        6. ``Ordinal day of Month YYYY`` (English only, 5-token form).

        Overlapping matches are all yielded; deduplication happens in
        the merger. Invalid dates (normalization returns ``None``)
        are dropped silently.
        """
        tokens = list(self._tokenizer.tokenize(text))
        n = len(tokens)

        for i, ts in enumerate(tokens):
            raw_text = ts.text
            if not raw_text:
                continue
            token = _strip(raw_text)
            if not token:
                continue

            # -- Branch 1: inline separators (MM/DD/YYYY etc.) ---------
            if "/" in token or "-" in token or "." in token:
                date_sep = next(
                    (ch for ch in token if ch in ("/", "-", ".")),
                    None,
                )
                if date_sep is not None:
                    date_tokens = token.split(date_sep)
                    if len(date_tokens) == 3:
                        digits = [t.isdigit() for t in date_tokens]
                        if all(digits) and all(len(t) in (1, 2, 4) for t in date_tokens):
                            date_value = self.normalize_date_tokens(
                                date_tokens, default_date=default_date
                            )
                            if date_value is not None:
                                yield AlphaSpan(date_value, ts.start, ts.end)
                        elif sum(digits) == 2:
                            # Month-word inside a slashed date:
                            # "Jan/1/21" or "1/Jan/21".
                            if date_tokens[0].lower() in self._month_map:
                                date_value = self.normalize_date_tokens(
                                    [
                                        date_tokens[2],
                                        str(self._month_map[date_tokens[0].lower()]),
                                        date_tokens[1],
                                    ],
                                    default_date=default_date,
                                )
                                if date_value is not None:
                                    yield AlphaSpan(date_value, ts.start, ts.end)
                            elif date_tokens[1].lower() in self._month_map:
                                date_value = self.normalize_date_tokens(
                                    [
                                        date_tokens[2],
                                        str(self._month_map[date_tokens[1].lower()]),
                                        date_tokens[0],
                                    ],
                                    default_date=default_date,
                                )
                                if date_value is not None:
                                    yield AlphaSpan(date_value, ts.start, ts.end)

            # -- Branch 2+: token is a month word ----------------------
            if token.lower() in self._month_map:
                pre_token = _strip(tokens[i - 1].text) if i > 0 else ""
                post1_token = _strip(tokens[i + 1].text) if i < n - 1 else ""
                post2_token = _strip(tokens[i + 2].text) if i < n - 2 else ""

                # -- DD Month YYYY ------------------------------------
                if (
                    pre_token
                    and post1_token
                    and pre_token.isdigit()
                    and post1_token.isdigit()
                    and len(post1_token) in (2, 4)
                ):
                    date_tokens = [
                        post1_token,
                        str(self._month_map[token.lower()]),
                        pre_token,
                    ]
                    date_value = self.normalize_date_tokens(date_tokens, default_date=default_date)
                    if date_value is not None:
                        yield AlphaSpan(date_value, tokens[i - 1].start, tokens[i + 1].end)

                # -- Ordinal Month YYYY -------------------------------
                if (
                    pre_token
                    and post1_token
                    and pre_token.lower() in self._ordinal_map
                    and post1_token.isdigit()
                    and len(post1_token) in (2, 4)
                ):
                    date_tokens = [
                        post1_token,
                        str(self._month_map[token.lower()]),
                        str(self._ordinal_map[pre_token.lower()]),
                    ]
                    date_value = self.normalize_date_tokens(date_tokens, default_date=default_date)
                    if date_value is not None:
                        yield AlphaSpan(date_value, tokens[i - 1].start, tokens[i + 1].end)

                # -- Two-post-token branches --------------------------
                if post1_token and post2_token:
                    # -- Month DD YYYY --------------------------------
                    if (
                        post1_token.isdigit()
                        and post2_token.isdigit()
                        and len(post2_token) in (2, 4)
                    ):
                        date_tokens = [
                            post2_token,
                            str(self._month_map[token.lower()]),
                            post1_token,
                        ]
                        date_value = self.normalize_date_tokens(
                            date_tokens, default_date=default_date
                        )
                        if date_value is not None:
                            yield AlphaSpan(date_value, tokens[i].start, tokens[i + 2].end)

                    # -- Month Ordinal YYYY ---------------------------
                    if (
                        post1_token.lower() in self._ordinal_map
                        and post2_token.isdigit()
                        and len(post2_token) in (2, 4)
                    ):
                        date_tokens = [
                            post2_token,
                            str(self._month_map[token.lower()]),
                            str(self._ordinal_map[post1_token.lower()]),
                        ]
                        date_value = self.normalize_date_tokens(
                            date_tokens, default_date=default_date
                        )
                        if date_value is not None:
                            yield AlphaSpan(date_value, tokens[i].start, tokens[i + 2].end)

                # -- English-specific: "Ord day of Month YYYY" -------
                if self.language == "en" and pre_token.lower() == "of" and i >= 3:
                    pre2_token = _strip(tokens[i - 2].text)
                    pre3_token = _strip(tokens[i - 3].text)
                    if (
                        post1_token
                        and pre2_token
                        and pre3_token
                        and pre2_token.lower() == "day"
                        and pre3_token.lower() in self._ordinal_map
                    ):
                        date_tokens = [
                            post1_token,
                            str(self._month_map[token.lower()]),
                            str(self._ordinal_map[pre3_token.lower()]),
                        ]
                        date_value = self.normalize_date_tokens(
                            date_tokens, default_date=default_date
                        )
                        if date_value is not None:
                            yield AlphaSpan(date_value, tokens[i - 3].start, tokens[i + 1].end)


__all__ = ["AlphaDateExtractor"]
