# ruff: noqa: RUF001, RUF002
"""AlphaPercentExtractor — rule-based percentage extraction.

Ports ``kelvin.nlp.extract.alpha.percent.PercentExtractor`` onto the
WS-TR foundation. Emits :class:`decimal.Decimal` fractional values.

``30%`` → ``Decimal("0.3")``. ``50 bps`` → ``Decimal("0.005")``. ``5
percent`` → ``Decimal("0.05")``. ``10‰`` (per-mille) → ``Decimal("0.01")``.
``1‱`` (per-myriad) → ``Decimal("0.0001")``.

Detection branches:

1. **Symbol suffix** — tokens ending in ``%``, ``﹪`` (small form),
   ``％`` (fullwidth), ``‰``, ``‱``, or ``bps``. Split off the suffix,
   parse the numeric prefix, scale by the appropriate factor.
2. **Separate word token** — tokens like ``"percent"``, ``"percentage"``,
   ``"basis"`` / ``"points"``, ``"ppm"``, ``"ppb"``. Parse the prior
   token as a number, multiply by the scale from :data:`PERCENT_MAP`.

The ``"basis"`` token requires the NEXT token to be ``"point"`` /
``"points"`` / ``"pts"`` (otherwise it's an unrelated word; skipped).

Divergences from kelvin:

1. **Instance-based** API for consistency with sibling extractors.
2. **``keep_punctuation=True`` tokenizer** — the ``%`` / ``‰`` / ``bps``
   suffix has to stay fused to the number token. Our tokenizer keeps
   these attached by default.
3. **Decimal scale factors from PERCENT_MAP are converted to
   ``Decimal(str(...))``** to avoid float→Decimal precision loss (our
   gazetteer stores ``0.01`` / ``1e-09`` as Python floats).
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from typing import ClassVar

from kaos_nlp_core.extract.alpha.number import AlphaNumberExtractor
from kaos_nlp_core.extract.base_extractor import (
    AlphaSpan,
    BaseAlphaExtractor,
    ExtractorValueType,
)
from kaos_nlp_core.locale_data import PERCENT_MAP
from kaos_nlp_core.tokenizer import Tokenizer
from kaos_nlp_core.types import TokenSpan

_STRIP_PUNCT = ".,;:!?\"'()[]"

# The explicit symbol-suffix table — scale factors for tokens like
# ``"30%"`` where the suffix is glued onto the number. Keys are the
# suffix strings (sorted by length, longest first, for greedy match).
_SUFFIX_SCALES: tuple[tuple[str, Decimal], ...] = (
    ("bps", Decimal("0.0001")),
    ("‱", Decimal("0.0001")),
    ("‰", Decimal("0.001")),
    ("%", Decimal("0.01")),
    ("﹪", Decimal("0.01")),
    ("％", Decimal("0.01")),
)

# Basis-point context guard: "basis" only counts as a percent anchor if
# the NEXT token is one of these.
_BASIS_NEXT_TOKENS: frozenset[str] = frozenset({"point", "points", "pts"})


class AlphaPercentExtractor(BaseAlphaExtractor[Decimal]):
    """Rule-based percentage extractor. Composes :class:`AlphaNumberExtractor`
    for quantity parsing.

    Usage::

        ext = AlphaPercentExtractor()
        list(ext.extract_values("Interest rate is 5.25% per annum."))
        # [Decimal('0.0525')]
    """

    name: ClassVar[str] = "percent"
    description: ClassVar[str] = "Rule-based percentage extraction"
    value_type: ClassVar[ExtractorValueType] = ExtractorValueType.PERCENTAGE
    languages: ClassVar[tuple[str, ...]] = ("en",)

    def __init__(self, language: str = "en") -> None:
        super().__init__(language=language)
        if language not in PERCENT_MAP:
            msg = (
                f"AlphaPercentExtractor: language {language!r} has no "
                f"PERCENT_MAP entry. Supported: {sorted(PERCENT_MAP.keys())}."
            )
            raise ValueError(msg)
        self._tokenizer = Tokenizer(keep_punctuation=True)
        self._number_extractor = AlphaNumberExtractor(language=language)

    def extract_spans(self, text: str) -> Iterator[AlphaSpan[Decimal]]:
        """Yield :class:`AlphaSpan[Decimal]` for every percent expression."""
        language_map = PERCENT_MAP[self.language]
        # Pre-convert scale factors to Decimal for exact arithmetic.
        word_scales = {k: Decimal(str(v)) for k, v in language_map.items()}

        tokens = list(self._tokenizer.tokenize(text))

        for i, ts in enumerate(tokens):
            raw = ts.text
            if not raw:
                continue
            token = raw.rstrip(_STRIP_PUNCT)
            if not token:
                continue

            # Branch 1: symbol-suffix form (longest-match).
            emitted = self._try_suffix(token, ts)
            if emitted is not None:
                yield emitted
                continue

            # Branch 2: separate percent-word token.
            low = token.lower()
            if low in word_scales and i > 0:
                # Special case: "basis" only counts if followed by
                # point/points/pts.
                if low == "basis":
                    if i + 1 >= len(tokens):
                        continue
                    next_raw = tokens[i + 1].text or ""
                    if next_raw.rstrip(_STRIP_PUNCT).lower() not in _BASIS_NEXT_TOKENS:
                        continue

                prior = tokens[i - 1]
                prior_raw = prior.text or ""
                prior_token = prior_raw.rstrip(_STRIP_PUNCT)
                if not prior_token:
                    continue
                quantity = self._number_extractor.parse_token(prior_token)
                if quantity is None:
                    continue
                scale = word_scales[low]
                yield AlphaSpan(
                    value=quantity * scale,
                    start=prior.start,
                    end=ts.end,
                )

    # -- Internals -----------------------------------------------------

    def _try_suffix(self, token: str, ts: TokenSpan) -> AlphaSpan[Decimal] | None:
        """Try to parse a symbol-suffix percent token like ``"30%"`` or
        ``"50bps"``. Returns the span or ``None`` if no match."""
        for suffix, scale in _SUFFIX_SCALES:
            # Case-insensitive match for 'bps' only (the Unicode symbols
            # are unambiguous).
            matches = token.endswith(suffix) if suffix != "bps" else token.lower().endswith(suffix)
            if not matches:
                continue
            if len(token) == len(suffix):
                # Bare suffix with no number — skip.
                return None
            numeric_part = token[: -len(suffix)]
            quantity = self._number_extractor.parse_token(numeric_part)
            if quantity is None:
                return None
            return AlphaSpan(
                value=quantity * scale,
                start=ts.start,
                end=ts.end,
            )
        return None


__all__ = ["AlphaPercentExtractor"]
