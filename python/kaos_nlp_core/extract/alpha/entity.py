"""AlphaEntityExtractor — rule-based corporate-entity extraction.

Ports ``kelvin.nlp.extract.alpha.entity.EntityExtractor`` onto the
WS-TR foundation. The algorithm is gazetteer-driven: when a token
matches one of ~108 corporate-entity-type suffixes (``Inc``, ``LLC``,
``GmbH``, ``SARL``, ``SpA``, ``K.K.``, etc.), the extractor scans
backward to collect the capitalized tokens that form the entity's
name.

Zero LLM calls. No regex. Deterministic. Consumed by the
``AlphaLLMMerger`` (PR-6f.3) to give the LLM an exhaustive list of
every candidate party name in a contract; the LLM's job becomes
selecting which candidate fills the ``parties`` column, not finding
them.

Known algorithmic limitations (inherited from kelvin, documented here
so downstream consumers know when to fall back to the LLM):

- **Hyphenated names are truncated**: ``"Coca-Cola Company"`` →
  ``"Coca"``. The hyphen fails the inclusion checks and breaks the
  backward scan.
- **Numeric-start names are rejected**: ``"3M Corp."`` →
  ``"3M"`` alone with no suffix (because ``"3M"``'s first char is a
  digit but the second char is uppercase; this path can extract
  ``"3M"`` if the digit branch fires). ``"7-Eleven"`` similarly
  truncated at the hyphen.
- **Lowercase-first names are rejected** in mixed-case text:
  ``"eBay Inc."`` — ``"eBay"[0].isupper()`` is ``False``, so the
  backward scan breaks before appending. If the entire text is
  all-upper, different path applies.
- **Dotted abbreviations inside names break the scan**: ``"J.P.
  Morgan Chase & Co."`` — the period after ``P`` is its own token
  and fails inclusion, so the scan truncates.
- **Lowercase suffix variants not in the gazetteer**: ``"acme inc."``
  won't trigger; only ``Inc`` / ``INC`` / ``Inc.`` / ``INC.`` match.
- **Suffix-less entity mentions not detected**: ``"Acme"`` alone,
  without a corporate suffix, produces no output.

The PR-6f.3 merger resolves these limitations by unioning with the
LLM's parties output — the LLM catches the cases the gazetteer
misses, and the gazetteer catches the cases the LLM misses. Neither
alone is sufficient; together they approach 100% recall.

Divergences from kelvin's algorithm:

1. **Span offset bug fixed.** Kelvin's ``name_start`` calculation
   (``prev_token_start + 1 + len(prev_token)``) is off by the
   whitespace delta between tokens — it points past the first
   appended name token instead of to its start. We track
   ``first_name_token_start`` while appending, so the span points
   to the actual start of the name.
2. **Backward scan extracted** as a private method (``_scan_backward``)
   — 45 lines of nested logic in a single method is hard to read and
   debug. No behavior change.
3. **Typed value.** Kelvin yields a bare ``((name, suffix), start,
   end)`` tuple. We yield an ``AlphaSpan[EntityMatch]`` where
   ``EntityMatch`` is a frozen dataclass with named ``name`` and
   ``entity_type`` fields.

Gazetteer is ported **verbatim** from kelvin — 108 entries across
English / German / French / Italian / Spanish-Portuguese / Dutch /
Japanese / Nordic / Czech / Polish / Malay / Hungarian / etc.
"""

from __future__ import annotations

import string
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

from kaos_nlp_core.extract.base_extractor import (
    AlphaSpan,
    BaseAlphaExtractor,
    ExtractorValueType,
)
from kaos_nlp_core.tokenizer import Tokenizer

# Punctuation set used to strip trailing comma/semicolon/colon — kelvin
# strips only ``,;:`` before the entity_set lookup (preserving periods,
# which are meaningful in ``Inc.`` / ``Ltd.`` / ``K.K.``). Matches
# kelvin-nlp entity.py:287.
_STRIP_TAIL_PUNCT = ",;:"

# Straight + curly quote characters. Kelvin-nlp entity.py:322-326 treats
# ANY of these appearing inside a token as a hard-stop for the backward
# scan (quotes typically delimit a quoted party name).
_QUOTE_CHARS = ('"', "\u201c", "\u201d")


@dataclass(frozen=True, slots=True)
class EntityMatch:
    """A corporate entity extracted by :class:`AlphaEntityExtractor`.

    ``name`` is the capitalized portion preceding the suffix, e.g.,
    ``"Acme Corporation"``. ``entity_type`` is the suffix that triggered
    the match, verbatim, e.g., ``"Inc."`` or ``"GmbH"``. Consumers that
    want a canonical form should lowercase + strip whitespace
    themselves (normalization is deferred to the merger layer).
    """

    name: str
    entity_type: str


class AlphaEntityExtractor(BaseAlphaExtractor[EntityMatch]):
    """Rule-based corporate-entity extractor.

    Gazetteer: 108 corporate-entity-type suffixes covering English,
    German, French, Italian, Spanish/Portuguese, Dutch, Japanese,
    Nordic, Czech, Polish, Malay, Hungarian, and more. See the
    ``entity_set`` class attribute for the full list.

    Algorithm: for each token in the input, if the token matches one
    of the gazetteer entries, scan backward collecting capitalized
    tokens (or special tokens ``&`` / ``limited`` / ``liability``)
    until a stop condition fires. The accumulated tokens (reversed)
    form the entity's name.

    Stop conditions (in order of precedence):

    1. Straight or curly quote inside the token → hard stop.
    2. Token is lowercase and in the stopword set (``"by"``, ``"the"``,
       ``"of"``, ``"between"``, ``"and"``, ``"whereas"``, etc.) →
       stop, UNLESS the accumulated tokens so far are all additional
       entity suffixes (in which case walk past to find the real name).
    3. Next token is ALSO in the gazetteer → skip this iteration and
       continue to the outer one (handles ``"Pvt. Ltd."`` by waiting
       for ``"Ltd."``).

    Entity-type gazetteer is exposed as ``AlphaEntityExtractor.entity_set``
    for consumers who want to extend it. Modify via ``entity_set.add(...)``;
    changes affect future extractions immediately.
    """

    name: ClassVar[str] = "entity"
    description: ClassVar[str] = "Rule-based corporate-entity extraction"
    value_type: ClassVar[ExtractorValueType] = ExtractorValueType.ACTOR_ENTITY
    languages: ClassVar[tuple[str, ...]] = ("en",)

    # -- 108-entry gazetteer (ported verbatim from kelvin-nlp entity.py:101-210)
    entity_set: ClassVar[set[str]] = {
        "A.B.",
        "A.G.",
        "A.S.",
        "A/S",
        "AB",
        "AG",
        "AS",
        "ApS",
        "Association",
        "B.V.",
        "BV",
        "BVBA",
        "Bank",
        "Bhd",
        "Bhd.",
        "C.A.",
        "C.V",
        "C.V.",
        "CORP.",
        "CV",
        "Co.",
        "CO.",
        "Company",
        "COMPANY",
        "Corp",
        "Corp.",
        "Corporation",
        "Holding",
        "Holdings",
        "HOLDING",
        "HOLDINGS",
        "CORPORATION",
        "G.K.",
        "GK",
        "GmbH",
        "Gmbh",
        "INC",
        "INC.",
        "Inc",
        "Inc.",
        "Incorporated",
        "INCORPORATED",
        "K.K.",
        "KG",
        "KK",
        "Kft",
        "L.L.C.",
        "L.L.P.",
        "L.P.",
        "LLC",
        "LLC.",
        "LLP",
        "LLP.",
        "LP",
        "LP.",
        "LTD",
        "LTD.",
        "LTDA",
        "Lda",
        "Lda.",
        "Limitada",
        "Limited",
        "LIMITED",
        "Ltd",
        "Ltd.",
        "Ltda",
        "Ltda.",
        "N.V.",
        "NV",
        "OY",
        "Oy",
        "Private",
        "PRIVATE",
        "R.L.",
        "S.A.",
        "S.A.R.L.",
        "S.A.S.",
        "S.L.",
        "S.L.U.",
        "S.R.L",
        "S.R.L.",
        "S.a.r.l",
        "S.a.r.l.",
        "S.p.A.",
        "S.r.l.",
        "S.\u00e1.r.l.",
        "SA",
        "SARL",
        "SAS",
        "SCSp",
        "SL",
        "SLU",
        "SPC",
        "SRL",
        "Sarl",
        "SpA",
        "Srl",
        "S\u00e0rl",
        "Trust",
        "U.A.",
        "ULC",
        "Unlimited",
        "d.o.o.",
        "o.o",
        "o.o.",
        "r.l.",
        "s.r.o.",
        "z.o.o.",
    }

    # -- Names starting with these tokens are rejected outright (kelvin
    # -- entity.py:212-220). Prevents "the Corporation" from being emitted
    # -- when "Corporation" is a standalone entity suffix.
    exclude_name_set: ClassVar[frozenset[str]] = frozenset(
        {
            "acquired",
            "acquiring",
            "the",
            "of",
            "such",
            "any",
            "by",
        }
    )

    # -- Backward scan stops when it sees a lowercase token from this set
    # -- (kelvin entity.py:259-273).
    _BACKWARD_STOP_TOKENS: ClassVar[frozenset[str]] = frozenset(
        {
            "by",
            "between",
            "and",
            "the",
            "of",
            "for",
            "in",
            "on",
            "to",
            "at",
            "from",
            "with",
            "whereas",
        }
    )

    # -- Tokens always included in the name even if not title-cased
    # -- (kelvin entity.py:334-338).
    _NAME_INCLUSION_OVERRIDES: ClassVar[frozenset[str]] = frozenset({"&", "limited", "liability"})

    def __init__(self, language: str = "en") -> None:
        super().__init__(language=language)
        self._tokenizer = Tokenizer(keep_punctuation=True)

    # -- Extraction -----------------------------------------------------

    def extract_spans(self, text: str) -> Iterator[AlphaSpan[EntityMatch]]:
        """Yield every corporate entity hit as an :class:`AlphaSpan`.

        The algorithm is the same as kelvin's, with the kelvin span-offset
        bug fixed. ``AlphaSpan.start`` points to the first character of the
        extracted name; ``AlphaSpan.end`` points to the end of the entity
        suffix.
        """
        tokens = list(self._tokenizer.tokenize(text))
        n = len(tokens)
        is_upper_text = text.isupper()

        for i, ts in enumerate(tokens):
            raw = ts.text
            if not raw:
                continue
            # Kelvin strips ``,;:`` only (not periods — ``Inc.`` must
            # match verbatim against the gazetteer).
            token = raw.rstrip(_STRIP_TAIL_PUNCT)
            if not token:
                continue

            # Gate: current token must be a known entity suffix.
            if token not in self.entity_set:
                continue

            # Skip if the NEXT token is also a suffix — handles chains
            # like "Pvt. Ltd." by waiting for "Ltd." to be current.
            if i < n - 1:
                next_raw = tokens[i + 1].text
                if next_raw and _strip_ascii_punct(next_raw) in self.entity_set:
                    continue

            # Need at least one prior token for a name.
            if i == 0:
                continue

            name_tokens, first_name_token_start = self._scan_backward(
                tokens=tokens,
                i=i,
                is_upper_text=is_upper_text,
            )

            if not name_tokens:
                continue

            # Reject names starting with a stopword (after reverse).
            if name_tokens[0].lower() in self.exclude_name_set:
                continue

            name_tokens_rev = list(reversed(name_tokens))

            # Handle the "single internal comma" case from kelvin — a
            # stray comma inside the accumulated tokens (e.g., "Acme
            # Corp., a Delaware corporation, Inc.") means the first
            # comma is the boundary; drop tokens before it.
            comma_positions = [j for j, t in enumerate(name_tokens_rev[:-1]) if t.endswith(",")]
            if len(comma_positions) == 1:
                comma_idx = comma_positions[0]
                # Skip everything up to and including the comma. The new
                # first name token determines the span start.
                name_tokens_rev = name_tokens_rev[comma_idx + 1 :]
                # Recompute the start: the first surviving token's
                # position is (i - len(name_tokens_rev))'s token start.
                kept = len(name_tokens_rev)
                first_name_token_start = tokens[i - kept].start

            if not name_tokens_rev:
                continue

            name = " ".join(name_tokens_rev).rstrip(_STRIP_TAIL_PUNCT)
            # Reject if the "name" is itself just a suffix.
            if name in self.entity_set or not name:
                continue

            yield AlphaSpan(
                value=EntityMatch(name=name, entity_type=token),
                start=first_name_token_start,
                end=ts.end,
            )

    # -- Internal: backward scan ---------------------------------------

    def _scan_backward(
        self,
        *,
        tokens: list,
        i: int,
        is_upper_text: bool,
    ) -> tuple[list[str], int]:
        """Walk backward from ``tokens[i]`` collecting the entity name.

        Returns ``(name_tokens_in_reverse_order, first_name_token_start)``.
        If no name is found, returns ``([], 0)``.

        Fixes the kelvin span-offset bug by tracking the start position
        of every token we append, so the final span points to the
        actual start of the first name token.
        """
        name_tokens: list[str] = []
        first_name_token_start = tokens[i].start  # worst-case fallback

        j = i
        while j > 0:
            j -= 1
            prev = tokens[j]
            prev_token = prev.text
            if prev_token is None:
                continue

            # Hard stop: quote characters.
            if any(q in prev_token for q in _QUOTE_CHARS):
                break

            lower = prev_token.lower()

            # Inclusion: first-capital / all-upper / digit, AND not a
            # stopword.
            if lower not in self._BACKWARD_STOP_TOKENS and (
                prev_token.isdigit()
                or (prev_token[:1].isupper())
                or (prev_token.isupper() and not is_upper_text)
            ):
                name_tokens.append(prev_token)
                first_name_token_start = prev.start
                continue

            # Inclusion: special always-included tokens.
            if lower in self._NAME_INCLUSION_OVERRIDES:
                name_tokens.append(prev_token)
                first_name_token_start = prev.start
                continue

            # All-upper-text fallback: any non-stopword token is included.
            if is_upper_text and lower not in self._BACKWARD_STOP_TOKENS:
                name_tokens.append(prev_token)
                first_name_token_start = prev.start
                continue

            # Continuation through a suffix chain: if everything we've
            # accumulated so far is already a suffix, walk past this
            # break-looking token in search of the actual name.
            if name_tokens and all(t.rstrip(",") in self.entity_set for t in name_tokens):
                name_tokens.append(prev_token)
                first_name_token_start = prev.start
                continue

            break

        return name_tokens, first_name_token_start


def _strip_ascii_punct(s: str) -> str:
    """Strip leading/trailing ASCII punctuation — mirrors kelvin's
    ``strip_punctuation`` for the next-token lookup."""
    return s.strip(string.punctuation)


__all__ = ["AlphaEntityExtractor", "EntityMatch"]
