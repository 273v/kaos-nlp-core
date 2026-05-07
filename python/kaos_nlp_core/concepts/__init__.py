"""Knowledge-graph-driven concept extraction via the OpenGloss hypergraph.

Aggregates document term frequencies up the hypernym graph (or down the
hyponym graph) to surface the categories or specific instances a
document discusses, *without an LLM*. Symmetric over both directions
so the same code path produces:

- **Hypernym aggregation** (default, ``direction="hypernym"``): bottom-up.
  Source terms ``plaintiff``, ``summons``, ``complaint`` aggregate to
  the concept ``litigation`` even when the word "litigation" never
  appears in the document.

- **Hyponym aggregation** (``direction="hyponym"``): top-down. Source
  terms ``vehicle``, ``automobile`` surface specific instances like
  ``sedan``, ``hatchback``, ``minivan`` mentioned by their parent term.

- **Both** (``direction="both"``): runs both passes and concatenates
  the typed result list. Each :class:`Concept` carries its
  ``direction`` so callers can split the list per-direction.

## Anti-pattern warning

This is a **tagging** tool. Feeding the extracted concepts back into a
BM25 retrieval query reproduces a known anti-pattern: lexicon
hypernym expansion in retrieval has been benchmarked to cost
**-18%** to **-22%** NDCG@10 on cross-domain BEIR tasks (see
``kaos-agents/CLAUDE.md`` and ``docs/design/adaptive-retrieval-roadmap.md``
in the kaos-modules monorepo). The default ``max_depth=1`` keeps that
expansion conservative; larger depths are allowed for experimentation
but documented as harmful for retrieval.

## Quality expectations (manual review, 2026-05-06)

A 10-document EDGAR review rated the top-5 concepts per document per
direction:

- **Hypernyms**: 58% relevant, 16% miss. Usable for coarse contract
  classification ("organization", "document", "property", "obligation",
  "control" cleanly distinguish lease vs loan vs employment vs
  securitization).
- **Hyponyms**: 38% relevant, 34% miss. Recurring boilerplate vocabulary
  (``fee``, ``grant``, ``salary``, ``license``, ``consent``) appears on
  every contract regardless of topic, with verb-sense bleed from common
  contract verbs (``trigger``, ``induce``, ``stipulate``). Treat the
  hyponym output as exploratory, not as final tags.

The hyponym noise mode is structural — IDF-style discriminativity
weighting against a corpus would likely fix it. Out of scope for v1;
documented here so callers don't over-trust the hyponym layer.

Usage::

    from kaos_nlp_core.concepts import extract_concepts

    concepts = extract_concepts(text)              # hypernyms, default lexicon
    for c in concepts[:5]:
        print(c.term, c.score, c.source_terms)

    # Both directions in one call.
    both = extract_concepts(text, direction="both", top_k=30)
    abstractions = [c for c in both if c.direction == "hypernym"]
    instances = [c for c in both if c.direction == "hyponym"]
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from kaos_nlp_core.lexicon import Lexicon, default_opengloss_lexicon
from kaos_nlp_core.vocabulary import VocabularyCounts, token_frequency

Direction = Literal["hypernym", "hyponym", "both"]
Weight = Literal["log", "linear"]

# ── Bundled stop-term defaults (calibrated by scripts/calibrate_stop_terms.py) ──
#
# These are the highest-document-frequency terms in each direction across the
# bundled USC + EDGAR + Project Gutenberg fixtures. They are the "every
# document" abstract roots ("entity", "thing", "process") for hypernyms, and
# extremely common subordinate forms for hyponyms. Recalibrate via
# ``scripts/calibrate_stop_terms.py`` if you change the corpus mix.
#
# Callers who want to extend should pass ``extra_stop_terms=`` — do not mutate
# these constants in place (per CLAUDE.md "Python-only modules" guidance).

# Calibrated 2026-05-06 by ``scripts/calibrate_stop_terms.py`` over USC,
# EDGAR agreements, and chunked Project Gutenberg fixtures (n=799 docs).
# Threshold: terms appearing in ≥ 50% of documents. Mostly part-of-speech
# roots (adjective, verb, preposition) and broad abstract categories
# (entity, event, communication) — the OpenGloss "anything is a thing"
# layer that drowns the signal otherwise.
#
# A second batch was added by hand after a manual EDGAR review surfaced
# adjacent POS-tag and WordNet-supersense artifacts that the 50%
# threshold missed but that are uniformly noise (``transitive verb``,
# ``manner adverb``, ``action verb``, ``attitude``, ``cognition``,
# ``concept``, ``behavior``, ``part``).
DEFAULT_STOP_HYPERNYMS: frozenset[str] = frozenset(
    {
        # df ≥ 0.50 calibrated.
        "action",
        "activity",
        "adjective",
        "adverb",
        "auxiliary verb",
        "change",
        "communication",
        "condition",
        "conjunction",
        "degree adverb",
        "descriptive adjective",
        "descriptor",
        "determiner",
        "directional adverb",
        "discourse marker",
        "entity",
        "event",
        "exclamation",
        "expression",
        "interjection",
        "modifier",
        "move",
        "person",
        "preposition",
        "process",
        "pronoun",
        "quality",
        "quantifier",
        "quantity",
        "spatial relation",
        "state",
        "subordinating conjunction",
        "temporal preposition",
        "temporal relation",
        "verb",
        # Hand-added 2026-05-06 after manual EDGAR review: WordNet
        # supersense / POS-variant artifacts that aren't useful tags.
        "action verb",
        "attitude",
        "behavior",
        "cognition",
        "concept",
        "manner adverb",
        "part",
        "transitive verb",
    }
)
"""Common abstract hypernym roots filtered by default. Calibrated."""

# Calibrated on the same fixture set. Dominated by OpenGloss artifacts
# where very common words ("the", "is", "and") have spurious illustrative
# hyponyms ("the sun", "the data", "additive conjunction") that bubble up
# unhelpfully.
DEFAULT_STOP_HYPONYMS: frozenset[str] = frozenset(
    {
        "additive conjunction",
        "aim",
        "coordinating conjunction",
        "extremely",
        "live",
        "teacher",
        "the data",
        "the study",
        "the sun",
        "toward",
    }
)
"""Common subordinate hyponym terms filtered by default. Calibrated."""

DEFAULT_STOP_TERMS: dict[str, frozenset[str]] = {
    "hypernym": DEFAULT_STOP_HYPERNYMS,
    "hyponym": DEFAULT_STOP_HYPONYMS,
}


# ── Result type ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Concept:
    """A concept aggregated from document terms via the lexicon graph.

    The ``direction`` distinguishes hypernym (broader) from hyponym
    (narrower) aggregation so a mixed result list stays interpretable.
    ``source_terms`` carries the document terms whose hypernyms /
    hyponyms contributed to this concept's score — essential for
    explainability and downstream UI affordances.
    """

    term: str
    """The aggregated concept (hypernym or hyponym of source terms)."""

    direction: Literal["hypernym", "hyponym"]
    """Which graph direction was walked. Never "both" at the record level."""

    score: float
    """Weighted aggregated score (log1p-saturated by default)."""

    frequency: int
    """Sum of source-term raw frequencies (linear, unweighted)."""

    source_terms: tuple[str, ...] = field(default_factory=tuple)
    """Document terms that contributed to this concept (sorted)."""


# ── Public API ───────────────────────────────────────────────────────────


def extract_concepts(
    text: str,
    *,
    lexicon: Lexicon | None = None,
    direction: Direction = "hypernym",
    top_k: int = 20,
    max_depth: int = 1,
    weight: Weight = "log",
    min_term_count: int = 1,
    extra_stop_terms: Iterable[str] | None = None,
) -> list[Concept]:
    """Aggregate document terms up (or down) the lexicon graph.

    Args:
        text: Document text.
        lexicon: Full :class:`~kaos_nlp_core.lexicon.Lexicon` (the
            OpenGloss graph). When ``None``, lazy-loads
            :func:`~kaos_nlp_core.lexicon.default_opengloss_lexicon`.
        direction: ``"hypernym"`` (default), ``"hyponym"``, or
            ``"both"``. ``"both"`` runs each direction independently
            and concatenates results.
        top_k: Maximum number of concepts per direction.
        max_depth: Graph-traversal depth. ``1`` (default) walks direct
            relations only. ``> 1`` is allowed but warned: deeper walks
            pull in increasingly abstract roots (for hypernyms) or noisy
            instances (for hyponyms), and have been benchmarked to
            **harm** downstream retrieval. Hard cap: 3.
        weight: ``"log"`` (default, ``log1p(frequency)``) or
            ``"linear"`` (raw frequency). Log saturation prevents one
            high-frequency term from dominating the result.
        min_term_count: Skip source terms with fewer than this many
            occurrences. Useful for very long documents to suppress
            noise.
        extra_stop_terms: Optional additional terms to filter from the
            aggregated result, **augmenting** the bundled defaults
            (per direction). Pass an empty iterable to keep the
            defaults untouched.

    Returns:
        List of :class:`Concept` records sorted by score descending.
        When ``direction="both"``, the list contains records of both
        directions (still sorted by score within each direction);
        callers can split via ``[c for c in result if c.direction == ...]``.

    Raises:
        ValueError: on unrecognized ``direction`` / ``weight`` / out-of-range ``max_depth``.
        FileNotFoundError: if ``lexicon=None`` and the bundled OpenGloss
            file cannot be located. See
            :func:`~kaos_nlp_core.lexicon.default_opengloss_lexicon`.
    """
    if direction not in ("hypernym", "hyponym", "both"):
        raise ValueError(f"direction must be 'hypernym', 'hyponym', or 'both' — got {direction!r}.")
    if weight not in ("log", "linear"):
        raise ValueError(f"weight must be 'log' or 'linear' — got {weight!r}.")
    if not 1 <= max_depth <= 3:
        raise ValueError(
            f"max_depth must be between 1 and 3 (got {max_depth}). "
            "Default 1 (one hop). Larger depths pull in noise; the BEIR "
            "regression on lexicon expansion in retrieval is documented in "
            "kaos-agents/CLAUDE.md."
        )

    active_lex = lexicon if lexicon is not None else default_opengloss_lexicon()

    # Lexicon-filtered token frequency in one pass — only count document
    # terms that actually exist in the graph, since out-of-vocabulary
    # terms have no hypernyms/hyponyms to aggregate to. Building the FST
    # view from a 200k-entry Lexicon takes ~100ms, so we cache it on the
    # Lexicon instance via a private attribute. The cache is per-Lexicon
    # so swapping in a custom lexicon still works.
    fst_view = _get_fst_view(active_lex)
    freq = token_frequency(text, lexicon=fst_view, lowercase=True, min_count=min_term_count)

    if direction == "both":
        hypernym_concepts = _aggregate(
            freq, active_lex, "hypernym", max_depth, weight, extra_stop_terms, top_k
        )
        hyponym_concepts = _aggregate(
            freq, active_lex, "hyponym", max_depth, weight, extra_stop_terms, top_k
        )
        return hypernym_concepts + hyponym_concepts
    return _aggregate(freq, active_lex, direction, max_depth, weight, extra_stop_terms, top_k)


# ── Internals ────────────────────────────────────────────────────────────


def _aggregate(
    freq: VocabularyCounts,
    lexicon: Lexicon,
    direction: Literal["hypernym", "hyponym"],
    max_depth: int,
    weight: Weight,
    extra_stop_terms: Iterable[str] | None,
    top_k: int,
) -> list[Concept]:
    """Walk one graph direction and aggregate scores."""
    stop_terms = _resolve_stop_terms(direction, extra_stop_terms)

    score_acc: dict[str, float] = {}
    freq_acc: dict[str, int] = {}
    sources_acc: dict[str, set[str]] = {}

    for tc in freq:
        related = _walk(lexicon, tc.text, direction, max_depth)
        if not related:
            continue
        contribution = math.log1p(tc.count) if weight == "log" else float(tc.count)
        for concept_term, depth_factor in related:
            if concept_term in stop_terms:
                continue
            score_acc[concept_term] = score_acc.get(concept_term, 0.0) + contribution * depth_factor
            freq_acc[concept_term] = freq_acc.get(concept_term, 0) + tc.count
            sources_acc.setdefault(concept_term, set()).add(tc.text)

    if not score_acc:
        return []

    # Sort by score desc, then term asc so ties resolve deterministically
    # — important for reproducible output across calls.
    ranked = sorted(score_acc.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [
        Concept(
            term=term,
            direction=direction,
            score=round(score_acc[term], 6),
            frequency=freq_acc[term],
            source_terms=tuple(sorted(sources_acc[term])),
        )
        for term, _ in ranked
    ]


def _walk(
    lexicon: Lexicon,
    word: str,
    direction: Literal["hypernym", "hyponym"],
    max_depth: int,
) -> list[tuple[str, float]]:
    """BFS up to ``max_depth`` hops; return ``(term, depth_factor)`` pairs.

    ``depth_factor`` is ``1 / depth`` so deeper walks contribute less to
    the score. The original word itself is excluded.
    """
    direct = lexicon.hypernyms(word) if direction == "hypernym" else lexicon.hyponyms(word)
    pairs: list[tuple[str, float]] = [(t, 1.0) for t in direct]

    if max_depth == 1:
        return pairs

    seen: set[str] = {word, *direct}
    frontier: list[str] = list(direct)
    for hop in range(2, max_depth + 1):
        next_frontier: list[str] = []
        depth_factor = 1.0 / hop
        for term in frontier:
            ancestors = (
                lexicon.hypernyms(term) if direction == "hypernym" else lexicon.hyponyms(term)
            )
            for a in ancestors:
                if a in seen:
                    continue
                seen.add(a)
                pairs.append((a, depth_factor))
                next_frontier.append(a)
        if not next_frontier:
            break
        frontier = next_frontier

    return pairs


_FST_VIEW_ATTR = "_concepts_fst_view"


def _get_fst_view(lexicon: Lexicon):
    """Lazy + per-Lexicon-instance cached FST view of headwords.

    Avoids rebuilding the ~200k-entry FST on every ``extract_concepts``
    call. The cache is set as a private attribute on the Lexicon
    instance, so each lexicon (default or user-supplied) keeps its own.
    """
    cached = getattr(lexicon, _FST_VIEW_ATTR, None)
    if cached is None:
        cached = lexicon.to_fst_set(include_inflections=False)
        try:
            object.__setattr__(lexicon, _FST_VIEW_ATTR, cached)
        except (AttributeError, TypeError):
            # Frozen class — fall back to per-call build.
            return cached
    return cached


def _resolve_stop_terms(
    direction: Literal["hypernym", "hyponym"],
    extra_stop_terms: Iterable[str] | None,
) -> frozenset[str]:
    """Combine the per-direction default stop-list with caller additions."""
    base = DEFAULT_STOP_TERMS[direction]
    if extra_stop_terms is None:
        return base
    extra = frozenset(extra_stop_terms)
    if not extra:
        return base
    return base | extra


__all__ = [
    "DEFAULT_STOP_HYPERNYMS",
    "DEFAULT_STOP_HYPONYMS",
    "DEFAULT_STOP_TERMS",
    "Concept",
    "Direction",
    "Weight",
    "extract_concepts",
]
