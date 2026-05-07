"""Rule-based alpha extractors — the deterministic half of WS-TR extraction.

Ports `kelvin.nlp.extract.alpha` one extractor at a time. Each extractor
is a concrete :class:`kaos_nlp_core.extract.base_extractor.BaseAlphaExtractor`
subclass. See ``docs/design/ws-tr-alpha-extractor-sprint.md`` for the
full sprint plan.
"""

from __future__ import annotations

from kaos_nlp_core.extract.alpha.contact import (
    AlphaContactExtractor,
    ContactKind,
    ContactMatch,
)
from kaos_nlp_core.extract.alpha.date import AlphaDateExtractor
from kaos_nlp_core.extract.alpha.defined_term import (
    AlphaDefinedTermExtractor,
    DefinedTermMatch,
    QuoteStyle,
)
from kaos_nlp_core.extract.alpha.duration import AlphaDurationExtractor, DurationMatch
from kaos_nlp_core.extract.alpha.entity import AlphaEntityExtractor, EntityMatch
from kaos_nlp_core.extract.alpha.money import AlphaMoneyExtractor, MoneyMatch
from kaos_nlp_core.extract.alpha.number import AlphaNumberExtractor
from kaos_nlp_core.extract.alpha.percent import AlphaPercentExtractor
from kaos_nlp_core.extract.alpha.quantity import AlphaQuantityExtractor, QuantityMatch
from kaos_nlp_core.extract.alpha.time import AlphaTimeExtractor

__all__ = [
    "AlphaContactExtractor",
    "AlphaDateExtractor",
    "AlphaDefinedTermExtractor",
    "AlphaDurationExtractor",
    "AlphaEntityExtractor",
    "AlphaMoneyExtractor",
    "AlphaNumberExtractor",
    "AlphaPercentExtractor",
    "AlphaQuantityExtractor",
    "AlphaTimeExtractor",
    "ContactKind",
    "ContactMatch",
    "DefinedTermMatch",
    "DurationMatch",
    "EntityMatch",
    "MoneyMatch",
    "QuantityMatch",
    "QuoteStyle",
]
