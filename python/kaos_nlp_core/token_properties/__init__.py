"""Token-level property classification helpers.

``classify_token()`` and ``classify_tokens()`` return typed
``TokenProperties`` dataclasses instead of raw dicts.
"""

from __future__ import annotations

from typing import Any

from kaos_nlp_core._rust.token_properties import classify_token as _raw_classify
from kaos_nlp_core._rust.token_properties import classify_tokens as _raw_classify_batch
from kaos_nlp_core._rust.token_properties import (
    has_emoji,
    is_abbreviation,
    is_alphanumeric_word,
    is_letter_word,
    is_lowercase_word,
    is_numeric_word,
    is_title_case_word,
    is_uppercase_word,
)
from kaos_nlp_core.types import TokenProperties


def _to_props(raw: Any) -> TokenProperties:
    return TokenProperties(
        is_letter_word=raw["is_letter_word"],
        is_uppercase_word=raw["is_uppercase_word"],
        is_lowercase_word=raw["is_lowercase_word"],
        is_mixed_case_word=raw["is_mixed_case_word"],
        is_title_case_word=raw["is_title_case_word"],
        is_numeric_word=raw["is_numeric_word"],
        is_alphanumeric_word=raw["is_alphanumeric_word"],
        starts_with_digit=raw["starts_with_digit"],
        has_punctuation=raw["has_punctuation"],
        has_dash=raw["has_dash"],
        is_hyphenated=raw["is_hyphenated"],
        is_abbreviation=raw["is_abbreviation"],
        has_emoji=raw["has_emoji"],
        is_symbolic_word=raw["is_symbolic_word"],
        ends_with_terminal=raw["ends_with_terminal"],
    )


def classify_token(token: str) -> TokenProperties:
    """Classify a token and return all property flags."""
    return _to_props(_raw_classify(token))


def classify_tokens(tokens: list[str]) -> list[TokenProperties]:
    """Classify multiple tokens in batch."""
    return [_to_props(r) for r in _raw_classify_batch(tokens)]


__all__ = [
    "TokenProperties",
    "classify_token",
    "classify_tokens",
    "has_emoji",
    "is_abbreviation",
    "is_alphanumeric_word",
    "is_letter_word",
    "is_lowercase_word",
    "is_numeric_word",
    "is_title_case_word",
    "is_uppercase_word",
]
