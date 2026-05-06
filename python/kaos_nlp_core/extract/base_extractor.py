"""BaseAlphaExtractor — abstract base for rule-based (LLM-free) extractors.

Ports :class:`kelvin.nlp.extract.base_extractor.BaseExtractor` and
:class:`kelvin.nlp.extract.base_extractor.ExtractorValueType` with these
deltas from kelvin's original:

- **Return typed :class:`AlphaSpan` instead of bare tuple.** Kelvin emits
  ``(value, start, end)`` tuples; we emit an ``AlphaSpan(value, start,
  end)`` frozen dataclass so consumers get attribute access and the type
  doesn't drift across extractor implementations. This is otherwise
  structurally identical — ``span.value``, ``span.start``, ``span.end``
  map 1:1.

- **Instance-configurable language.** Kelvin's extractors use
  ``@classmethod`` with a class-level ``language`` attribute. We use
  instance methods with an optional ``language`` constructor arg so
  callers can instantiate multiple extractors with different gazetteers
  without subclassing.

- **Synchronous by design.** Alpha extractors are pure CPU. Every
  extractor runs in microseconds on paragraph-sized input. Making them
  ``async`` would impose event-loop overhead for zero benefit. The
  ``AlphaLLMMerger`` (PR-6f.3) that composes alpha with the LLM
  pipeline IS async; the merger awaits the LLM call and then calls the
  alpha extractor synchronously.

- **Trimmed value-type enum.** Dropped kelvin's ``EVENT``, ``TITLE``,
  ``SECTION``, ``STRUCTURE``, ``DEFINITION``, ``RIGHTS_OBLIGATIONS``
  (those are LLM-only tasks in the WS-TR world). Added nothing. Enum
  values are stable integers for serialization.

See ``docs/design/ws-tr-alpha-extractor-sprint.md`` §3.1 for the
design rationale.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar


class ExtractorValueType(IntEnum):
    """The logical kind of value an extractor emits.

    Integer values are stable for serialization. The enum is a subset of
    kelvin's ``ExtractorValueType`` — LLM-only types (events, titles,
    sections) are omitted because they don't have a deterministic
    rule-based extraction path.
    """

    NUMBER = 0
    NUMBER_WITH_UNITS = 1
    RATIO = 2
    FRACTION = 3
    PERCENTAGE = 4
    DURATION = 5
    MONEY = 6
    DATE_FULL = 7
    DATE_YEAR = 8
    DATE_MONTH = 9
    TIME = 10
    GEO_LOCATION = 12
    GEO_ENTITY = 13
    ACTOR_PERSON = 14
    ACTOR_ENTITY = 15
    ACTOR_GOVERNMENT = 17
    ACTOR_PARTY = 19
    # Added 2026-05-06 for the four new alpha extractors. Integers are
    # stable for serialization; gaps at 11/16/18 stay reserved.
    CONTACT_INFO = 20
    DEFINED_TERM = 21


@dataclass(frozen=True, slots=True)
class AlphaSpan[T]:
    """One rule-based extraction hit.

    ``value`` is typed per extractor (e.g., ``datetime.date`` for
    :class:`AlphaDateExtractor`, ``(Decimal, str)`` for
    ``AlphaMoneyExtractor``). ``start`` / ``end`` are **character**
    offsets into the original source text (half-open; Python slice
    convention). Consumers reconstruct the matched substring via
    ``source_text[span.start:span.end]``.

    Frozen + slotted so the merger (PR-6f.3) can hash these for
    cross-chunk dedup without cloning.
    """

    value: T
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class BaseAlphaExtractor[T](abc.ABC):
    """Abstract base class for rule-based extractors.

    Subclasses override :meth:`extract_spans`. Everything else
    (:meth:`extract_values`, :meth:`extract_first_span`, etc.) has a
    default implementation that iterates :meth:`extract_spans`.

    Subclasses declare class-level metadata (``name``, ``description``,
    ``value_type``, ``languages``) as :class:`ClassVar` attributes.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    value_type: ClassVar[ExtractorValueType]
    languages: ClassVar[tuple[str, ...]] = ("en",)

    def __init__(self, language: str = "en") -> None:
        if language not in self.languages:
            msg = (
                f"{type(self).__name__} does not support language {language!r}. "
                f"Supported: {list(self.languages)}. "
                f"Fix: pass a supported language to the constructor, or "
                f"subclass and extend ``languages``."
            )
            raise ValueError(msg)
        self.language = language

    # -- Metadata accessors (mirror kelvin, kept for API parity) -----------

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_value_type(self) -> ExtractorValueType:
        return self.value_type

    def get_languages(self) -> list[str]:
        return list(self.languages)

    # -- Core extraction method (subclasses MUST override) -----------------

    @abc.abstractmethod
    def extract_spans(self, text: str) -> Iterator[AlphaSpan[T]]:
        """Extract all rule-matched values from ``text`` with char spans.

        Implementations MUST:

        - Yield :class:`AlphaSpan` instances, not bare tuples.
        - Use character offsets (Python slice convention), not byte
          offsets. When building on top of Rust tokenizer output, use
          the tokenizer's char-offset mode.
        - Never raise on bad input — yield nothing instead. Exceptions
          leak into the extract_corpus batch log and are hard to
          correlate back to the regex pattern that failed.
        - Be idempotent: calling twice on the same text yields the same
          spans.
        """

    # -- Convenience helpers (default implementations) ---------------------

    def extract_values(self, text: str) -> Iterator[T]:
        """Iterate values only, dropping spans. Useful when span
        positions don't matter (e.g., canonicalization of names)."""
        for span in self.extract_spans(text):
            yield span.value

    def extract_first_span(self, text: str) -> AlphaSpan[T] | None:
        """Return the earliest-starting span, or ``None``. Breaks on
        first hit — O(1) after tokenization."""
        for span in self.extract_spans(text):
            return span
        return None

    def extract_first_value(self, text: str) -> T | None:
        """Return the value of the earliest-starting span, or ``None``."""
        span = self.extract_first_span(text)
        return span.value if span is not None else None

    def extract_last_span(self, text: str) -> AlphaSpan[T] | None:
        """Return the latest-finishing span, or ``None``. Drains the
        iterator — O(N) over all matches."""
        last: AlphaSpan[T] | None = None
        for span in self.extract_spans(text):
            last = span
        return last

    def extract_last_value(self, text: str) -> T | None:
        """Return the value of the latest-finishing span, or ``None``."""
        span = self.extract_last_span(text)
        return span.value if span is not None else None


__all__ = [
    "AlphaSpan",
    "BaseAlphaExtractor",
    "ExtractorValueType",
]


# Keep ``Any`` importable from here so subclasses don't need their own
# typing imports for generic ``AlphaSpan[Any]`` escape hatches.
_AnyAlias = Any
