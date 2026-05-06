"""AlphaMoneyExtractor — rule-based money extraction.

Ports ``kelvin.nlp.extract.alpha.money.MoneyExtractor`` onto the WS-TR
foundation. Emits ``(Decimal amount, str ISO-code)`` tuples for every
money expression found in the input.

Detection branches:

1. **Currency word with preceding quantity** — tokens like
   ``"dollars"``, ``"euros"``, ``"CHF"``, etc. (anything in
   :data:`MONEY_MAP`). The prior token is examined for a quantity —
   arabic number, written number, or an indefinite article (``"a"``,
   ``"an"``, ``"the"``, ``"one"`` → ``Decimal(1)``). Fails silently
   if the prior token isn't a number.

2. **Currency symbol fused with number** — tokens like ``"$13.50"``,
   ``"€10"``, ``"100$"``. Detected by first/last char being a symbol
   in the gazetteer with the rest being parsable as an Arabic number.

Always returns ISO 4217 codes via :data:`MONEY_MAP` lookup (kelvin's
branch 1 had a bug where it returned the raw currency word — we fix
that here; every code path goes through the gazetteer).

Known algorithmic limitations (inherited from kelvin, documented for
downstream consumers):

- **Two-word currency phrases are NOT recognized** as a unit:
  ``"US dollars"`` → our tokenizer yields ``["US", "dollars"]``. The
  prior-token branch sees ``"US"`` which is not a number, so nothing is
  emitted. Similarly ``"British pounds"``.
- **Redacted values**: ``"$[***]"`` — the symbol is present but the
  trailing chars aren't a number, so the token is skipped. This is
  intentional; never fabricate a value from a redaction marker.
- **Decorated numbers**: ``"$1 million"`` — tokenized as
  ``["$1", "million"]``. The first token extracts ``(Decimal(1), USD)``
  and the second extracts ``Decimal(1_000_000)``, but there's no
  composition between them. The LLM or a downstream cross-token
  aggregator must combine them.
- **Parenthesized quantities**: ``"one hundred dollars ($100)"`` —
  emitted as two separate extractions because the alpha layer has no
  cross-reference resolver.

Divergences from kelvin's algorithm:

1. **Always canonicalize to ISO code.** Kelvin's branch 1
   (``money.py:151``) yielded ``(quantity, token)`` — ``token`` being the
   raw currency word like ``"dollars"``. Our port always looks up the
   ISO code in the gazetteer: ``MONEY_MAP[language][token]``. Matches
   kelvin's branch 2 behavior, the docstring, and the intent.
2. **Instance-based API** — mirrors :class:`AlphaNumberExtractor`.
3. **``keep_punctuation=True`` tokenizer** — so ``"$13.50"`` stays a
   single token rather than being split by the ``$``.
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
from kaos_nlp_core.locale_data import MONEY_MAP
from kaos_nlp_core.tokenizer import Tokenizer
from kaos_nlp_core.types import TokenSpan

# Articles that imply a quantity of 1 when preceding a currency word.
_UNIT_ARTICLES: frozenset[str] = frozenset({"a", "an", "the", "one"})

# Punctuation stripped from token edges before gazetteer / currency-
# symbol testing. Includes trailing periods because currency words
# often appear at sentence boundaries (``"... ten dollars."``). The
# period is safe to strip here because our detection branches don't
# look inside the gazetteer for period-bearing forms — ``"U.S."`` is
# not a currency word, ``"U.S.A."`` is not in MONEY_MAP.
_STRIP_PUNCT = ".,;:!?\"'()[]"


@dataclass(frozen=True, slots=True)
class MoneyMatch:
    """A money expression extracted by :class:`AlphaMoneyExtractor`.

    - ``amount`` — the monetary quantity as a :class:`Decimal`
      (preserves precision; safe for SQL ``money`` / ``numeric`` columns).
    - ``currency`` — ISO 4217 code (``"USD"``, ``"GBP"``, ``"EUR"``,
      ``"JPY"``, ``"CNY"``, ``"INR"``, ``"KRW"``, ``"CHF"``).
    """

    amount: Decimal
    currency: str


class AlphaMoneyExtractor(BaseAlphaExtractor[MoneyMatch]):
    """Rule-based money extractor. Composes :class:`AlphaNumberExtractor`
    for quantity parsing.

    Usage::

        ext = AlphaMoneyExtractor()
        list(ext.extract_values("The cap is $1,000,000 per claim."))
        # [MoneyMatch(amount=Decimal('1000000'), currency='USD')]
    """

    name: ClassVar[str] = "money"
    description: ClassVar[str] = "Rule-based money extraction (currency + amount)"
    value_type: ClassVar[ExtractorValueType] = ExtractorValueType.MONEY
    languages: ClassVar[tuple[str, ...]] = ("en",)

    def __init__(self, language: str = "en") -> None:
        super().__init__(language=language)
        if language not in MONEY_MAP:
            msg = (
                f"AlphaMoneyExtractor: language {language!r} has no MONEY_MAP entry. "
                f"Supported languages: {sorted(MONEY_MAP.keys())}"
            )
            raise ValueError(msg)
        self._tokenizer = Tokenizer(keep_punctuation=True)
        self._number_extractor = AlphaNumberExtractor(language=language)

    def extract_spans(self, text: str) -> Iterator[AlphaSpan[MoneyMatch]]:
        """Yield :class:`AlphaSpan[MoneyMatch]` for every money expression."""
        language_map = MONEY_MAP[self.language]
        tokens = list(self._tokenizer.tokenize(text))

        for i, ts in enumerate(tokens):
            raw = ts.text
            if not raw:
                continue
            # Strip only TRAILING punctuation — we need to preserve
            # leading `.` in case a future gazetteer entry starts with
            # one, and to keep "$0.50" parseable in the fused branch.
            token = raw.rstrip(_STRIP_PUNCT)
            if not token:
                continue

            # Branch 1: token IS a currency word / symbol in the gazetteer
            # (e.g., "dollars", "$", "USD"). Look backward for a quantity.
            if token in language_map:
                iso = language_map[token]
                yield from self._emit_from_prior_token(tokens, i, token_end=ts.end, iso=iso)
                continue

            # Branch 2: token has a currency SYMBOL fused with a number
            # (e.g., "$13.50", "€10", "100$").
            yield from self._emit_from_fused_token(token, ts, language_map)

    # ---- Internal: branch 1 (currency word preceded by quantity) ----

    def _emit_from_prior_token(
        self,
        tokens: list,
        i: int,
        *,
        token_end: int,
        iso: str,
    ) -> Iterator[AlphaSpan[MoneyMatch]]:
        if i == 0:
            return
        prior = tokens[i - 1]
        prior_raw = prior.text or ""
        prior_token = prior_raw.rstrip(_STRIP_PUNCT)
        if not prior_token:
            return

        low = prior_token.lower()
        if low in _UNIT_ARTICLES:
            # "a dollar" / "an euro" / "one dollar" — quantity = 1.
            yield AlphaSpan(
                value=MoneyMatch(amount=Decimal(1), currency=iso),
                start=prior.start,
                end=token_end,
            )
            return

        # Parse the prior token as a number (handles arabic, roman,
        # written). The prior token is already a single token from our
        # own tokenizer pass — use ``parse_token`` to skip a redundant
        # tokenization (audit perf finding #4 / P6). The prior code path
        # called ``extract_values`` which re-ran the tokenizer; for a
        # single-token input that yielded a single result anyway, so the
        # behavior is preserved while saving a per-money-candidate
        # tokenizer round trip.
        quantity = self._number_extractor.parse_token(prior_token)
        if quantity is None:
            return
        yield AlphaSpan(
            value=MoneyMatch(amount=quantity, currency=iso),
            start=prior.start,
            end=token_end,
        )

    # ---- Internal: branch 2 (symbol fused with number) ----

    def _emit_from_fused_token(
        self,
        token: str,
        ts: TokenSpan,
        language_map: dict[str, str],
    ) -> Iterator[AlphaSpan[MoneyMatch]]:
        start = ts.start
        end = ts.end

        # Prefix symbol: first char is a currency symbol, rest is a number.
        first = token[:1]
        if first in language_map and len(token) > 1:
            iso = language_map[first]
            numeric_part = token[1:]
            values = list(self._number_extractor.extract_values(numeric_part))
            if values:
                yield AlphaSpan(
                    value=MoneyMatch(amount=values[-1], currency=iso),
                    start=start,
                    end=end,
                )
                return

        # Suffix symbol: last char is a currency symbol, preceding is a number.
        last = token[-1:]
        if last in language_map and len(token) > 1:
            iso = language_map[last]
            numeric_part = token[:-1]
            values = list(self._number_extractor.extract_values(numeric_part))
            if values:
                yield AlphaSpan(
                    value=MoneyMatch(amount=values[-1], currency=iso),
                    start=start,
                    end=end,
                )


__all__ = ["AlphaMoneyExtractor", "MoneyMatch"]
