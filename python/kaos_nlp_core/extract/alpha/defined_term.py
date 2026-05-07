# ruff: noqa: RUF001
"""AlphaDefinedTermExtractor — defined-term detection in contract text.

Fills the ``ExtractorValueType.DEFINED_TERM`` slot. Surfaces the
parenthetical-definition pattern that pervades commercial contracts:

- ``(the "Borrower")``
- ``(the "Effective Date")``
- ``(hereinafter referred to as "Tenant")``
- ``(collectively, the "Parties")``
- ``(individually, a "Party")``
- ``("Premises")``
- ``(the "Borrower" or "Lender")`` — multiple terms in one paren

Returns :class:`DefinedTermMatch(term, intro_phrase, definition_clause,
quote_style)`. The character span of each :class:`AlphaSpan` covers the
entire parenthesized region so consumers can highlight the definition
in the original document.

Definition clause heuristic: the text from the START of the enclosing
sentence up to the open-paren. This is "the sentence that introduces
the term" for ~80% of typical contracts. When the introducing sentence
straddles multiple lines or uses semicolons / colons, the heuristic
captures more than necessary; consumers should treat it as a search
anchor, not a verbatim definition.

Out of scope: defined terms not enclosed in parens (e.g., "the term
'Borrower' shall mean ..."), schedule references, and signature-block
party names. The kelvin-nlp ``EntityExtractor`` covers party names by
suffix; this extractor covers the parenthetical-definition pattern.
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

QuoteStyle = Literal["double", "single", "curly", "guillemets"]


@dataclass(frozen=True, slots=True)
class DefinedTermMatch:
    """A parenthesized-definition hit.

    - ``term`` — the quoted defined term, with quotes stripped
      (``"Borrower"``).
    - ``intro_phrase`` — the in-paren preamble before the quoted term
      (``"the"``, ``"hereinafter referred to as"``, ``"collectively,
      the"``). ``None`` if the paren is just the bare quoted term.
    - ``definition_clause`` — the text from the start of the enclosing
      sentence up to the open-paren. ``None`` if the start of sentence
      can't be located within a reasonable look-back window.
    - ``quote_style`` — which kind of quote the term used. Useful for
      consumers normalizing terms across documents.
    """

    term: str
    intro_phrase: str | None
    definition_clause: str | None
    quote_style: QuoteStyle


# Quote-pair table. Each entry is (open, close, style).
_QUOTE_PAIRS: tuple[tuple[str, str, QuoteStyle], ...] = (
    ('"', '"', "double"),
    ("“", "”", "curly"),  # left/right double curly quotes
    ("'", "'", "single"),
    ("‘", "’", "curly"),  # left/right single curly quotes
    ("«", "»", "guillemets"),
)

# Open-quote characters indexed for fast lookup.
_OPEN_QUOTES: dict[str, tuple[str, QuoteStyle]] = {o: (c, s) for o, c, s in _QUOTE_PAIRS}

# Common intro-phrase tokens that precede the quoted term inside the
# paren. Comparison is case-insensitive on the joined word sequence.
_INTRO_TOKENS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "hereinafter",
        "referred",
        "to",
        "as",
        "collectively",
        "individually",
        "or",
        "and",
        "each",
        "any",
        "such",
        "this",
    }
)

# Maximum look-back distance for the definition_clause heuristic.
# Sentences in legal text can be very long; cap at 1000 chars to avoid
# crossing paragraph boundaries.
_DEFINITION_LOOKBACK = 1000

# Sentence-boundary markers. Crude but consistent with the rest of the
# segmentation rules used elsewhere in the module.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])|(?:\n\s*\n)|^")


class AlphaDefinedTermExtractor(BaseAlphaExtractor[DefinedTermMatch]):
    """Rule-based extractor for parenthesized contract definitions.

    Usage::

        ext = AlphaDefinedTermExtractor()
        text = (
            "Acme Corporation, a Delaware corporation (the \\"Borrower\\"), "
            "and First National Bank (the \\"Lender\\"), agree as follows."
        )
        for span in ext.extract_spans(text):
            print(span.value.term, "←", span.value.intro_phrase)
        # Borrower ← the
        # Lender ← the
    """

    name: ClassVar[str] = "defined_term"
    description: ClassVar[str] = "Rule-based parenthesized-definition extraction (contract terms)"
    value_type: ClassVar[ExtractorValueType] = ExtractorValueType.DEFINED_TERM
    languages: ClassVar[tuple[str, ...]] = ("en",)

    def __init__(self, language: str = "en") -> None:
        super().__init__(language=language)

    def extract_spans(self, text: str) -> Iterator[AlphaSpan[DefinedTermMatch]]:
        """Yield :class:`AlphaSpan[DefinedTermMatch]` for each
        parenthesized definition. One span per quoted term — a paren
        with multiple quotes (``(the "Borrower" or "Lender")``)
        produces multiple spans."""
        for paren_start, paren_end in _find_paren_regions(text):
            inner = text[paren_start + 1 : paren_end - 1]
            quoted_spans = list(_find_quoted_spans(inner))
            if not quoted_spans:
                continue
            prev_close = 0
            for q_start, q_end, term, style in quoted_spans:
                # For the 2nd+ quoted term in a paren, look only at
                # text since the previous closing quote — otherwise
                # the prior term's quote characters poison the intro
                # tokenization.
                intro = _extract_intro_phrase(inner, q_start, prev_close)
                definition = _extract_definition_clause(text, paren_start)
                yield AlphaSpan(
                    value=DefinedTermMatch(
                        term=term,
                        intro_phrase=intro,
                        definition_clause=definition,
                        quote_style=style,
                    ),
                    start=paren_start,
                    end=paren_end,
                )
                prev_close = q_end


# -- Module-level helpers ---------------------------------------------------


def _find_paren_regions(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(open_offset, close_offset)`` for each top-level
    parenthesized region. ``close_offset`` is exclusive (Python slice
    convention). Nested parens are tracked but only the outermost
    region is yielded.
    """
    depth = 0
    open_at = -1
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                open_at = i
            depth += 1
        elif ch == ")":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and open_at >= 0:
                yield (open_at, i + 1)
                open_at = -1


def _find_quoted_spans(
    inner: str,
) -> Iterator[tuple[int, int, str, QuoteStyle]]:
    """Yield ``(start, end, content, style)`` for each quoted region in
    ``inner``. Offsets are relative to ``inner``."""
    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]
        if ch in _OPEN_QUOTES:
            close_ch, style = _OPEN_QUOTES[ch]
            j = inner.find(close_ch, i + 1)
            if j == -1:
                # Unclosed quote — skip to end.
                return
            term = inner[i + 1 : j].strip()
            if term and not _looks_like_apostrophe(inner, i, j, ch, style):
                yield (i, j + 1, term, style)
            i = j + 1
        else:
            i += 1


def _looks_like_apostrophe(
    inner: str, open_at: int, close_at: int, open_ch: str, style: QuoteStyle
) -> bool:
    """Heuristic: a single-quote pair around a single-letter or
    contraction-suffix string is probably an apostrophe pair, not a
    defined-term quote. Only applies to single-quote styles."""
    if style != "single":
        return False
    inside = inner[open_at + 1 : close_at]
    if len(inside) <= 2:
        return True
    # If the open-quote is immediately preceded by a letter, it's
    # probably an apostrophe ("don't" -> "dont").
    return bool(open_at > 0 and inner[open_at - 1].isalpha())


def _extract_intro_phrase(inner: str, q_start: int, since: int = 0) -> str | None:
    """Capture the ``(the | hereinafter referred to as | collectively, the)``
    preamble before the quoted term. Returns the joined surface form,
    or ``None`` if no recognized intro tokens precede the term.

    ``since`` is the offset (within ``inner``) to start scanning from —
    used by callers to skip over the preceding quoted term in a
    multi-term paren like ``(the "A" or the "B")``.
    """
    pre = inner[since:q_start].strip().rstrip(",;:")
    if not pre:
        return None
    # Split on whitespace; only return when every token is in the
    # intro vocabulary. This rejects pre-text like "this Agreement"
    # (where "this" + "Agreement" isn't a clean intro pattern but
    # "this" alone is in the vocab).
    tokens = pre.replace(",", " ").split()
    if not tokens:
        return None
    if not all(t.lower().rstrip(",.;:") in _INTRO_TOKENS for t in tokens):
        return None
    return pre


def _extract_definition_clause(text: str, paren_start: int) -> str | None:
    """Capture the sentence preceding the open-paren, capped at
    ``_DEFINITION_LOOKBACK`` chars. Returns ``None`` if no clear
    sentence start is found within the window."""
    if paren_start == 0:
        return None
    look_start = max(0, paren_start - _DEFINITION_LOOKBACK)
    snippet = text[look_start:paren_start]
    # Find the LAST sentence boundary in the snippet; the clause is
    # everything after it. Boundaries: paragraph break or "[.!?]\s+[A-Z]".
    boundaries = list(_SENTENCE_BOUNDARY_RE.finditer(snippet))
    if not boundaries:
        return None
    last = boundaries[-1]
    clause_start = look_start + last.end()
    clause = text[clause_start:paren_start].strip()
    return clause or None


__all__ = [
    "AlphaDefinedTermExtractor",
    "DefinedTermMatch",
    "QuoteStyle",
]
