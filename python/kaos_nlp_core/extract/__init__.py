"""Rule-based extraction primitives — the deterministic, LLM-free
half of structured extraction.

This module contains rule-based extractors that work directly on text
using kaos-nlp-core's tokenizer and locale gazetteers. They emit typed
:class:`AlphaSpan` objects with character offsets so consumers can
reconstruct the matched substring verbatim.

The extractors here moved from ``kaos_llm_core.extract`` in 2026-05.
Rationale: the alpha extractors have no LLM dependency — they only
use kaos-nlp-core primitives — so they belong here. The LLM-coupled
``AlphaLLMMerger`` and the typed extraction Programs that compose
alpha + LLM output remain in ``kaos_llm_core``.

Submodules:

* :mod:`kaos_nlp_core.extract.alpha` — concrete extractors
  (date, duration, entity, money, number, percent).
* :mod:`kaos_nlp_core.extract.base_extractor` — abstract base class
  + :class:`AlphaSpan` value type + :class:`ExtractorValueType`
  enum.
"""

from __future__ import annotations

from kaos_nlp_core.extract.base_extractor import (
    AlphaSpan,
    BaseAlphaExtractor,
    ExtractorValueType,
)

__all__ = [
    "AlphaSpan",
    "BaseAlphaExtractor",
    "ExtractorValueType",
]
