"""Regression test for audit-04 F-002: public-module re-export drift.

audit-04/kaos-nlp-core.md flagged that the top-level
``kaos_nlp_core.__all__`` omitted five subpackages whose own
``__all__`` and CHANGELOG entries treat them as public API:

  - ``content_type`` — called out in 0.1.0a7 and 0.1.1 CHANGELOG as a
    shipped public feature.
  - ``concepts`` — knowledge-graph concept extraction.
  - ``extract`` — base + GLiNER extractor surface.
  - ``structure`` — line-label scorer + decoder.
  - ``vocabulary`` — frequency vocabulary helpers.

The README and project standards both say documented modules are
public API. Pre-fix, ``import kaos_nlp_core; kaos_nlp_core.content_type``
raised ``AttributeError`` even though the changelog promised it.

This test pins the corrected surface so the regression cannot reopen
silently.
"""

from __future__ import annotations

import importlib

import kaos_nlp_core

# Subpackages that audit-04 confirmed should be on the top-level facade.
_PUBLIC_SUBPACKAGES_FROM_AUDIT: tuple[str, ...] = (
    "content_type",
    "concepts",
    "extract",
    "structure",
    "vocabulary",
)


def test_audit04_subpackages_are_accessible_via_top_level() -> None:
    """``kaos_nlp_core.<name>`` must resolve without an extra import.

    Before audit-04 F-002, attribute access on these names raised
    ``AttributeError`` because the top-level ``__init__`` never bound
    them. The fix re-exports each as a module attribute.
    """
    for name in _PUBLIC_SUBPACKAGES_FROM_AUDIT:
        assert hasattr(kaos_nlp_core, name), (
            f"audit-04 regression: kaos_nlp_core.{name} missing (was re-exported by F-002 fix)"
        )
        # And the bound attribute must be the actual subpackage object,
        # not a fresh re-import or a stub.
        direct = importlib.import_module(f"kaos_nlp_core.{name}")
        assert getattr(kaos_nlp_core, name) is direct


def test_audit04_subpackages_in_dunder_all() -> None:
    """``__all__`` is the wildcard-import contract — pin the membership."""
    for name in _PUBLIC_SUBPACKAGES_FROM_AUDIT:
        assert name in kaos_nlp_core.__all__, (
            f"audit-04 F-002 regression: {name!r} missing from kaos_nlp_core.__all__"
        )
