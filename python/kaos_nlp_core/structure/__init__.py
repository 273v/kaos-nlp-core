"""Document-structure analysis: heading scorer, sequence decoder, hierarchy
inferencer (P7).

The three primitives compose into one pipeline:

* :func:`score_heading_features` — per-line feature vectors (P7.1).
* :func:`decode_line_labels` — Viterbi sequence decoder over the 7-state
  label set (P7.4).
* :func:`label_lines` — full pipeline (extract → score → decode →
  hierarchy) returning ``(labels, features, candidates)``.

All three accept caller-supplied lexicons via :class:`HeadingLexicon`,
:class:`HierarchyLexicon`, and the P3 enumerator's ``WordLexicon``. The
generality contract (G1-G8 in the design reference) is honored by
default: the scorer + decoder produce sensible output for documents
without keywords or enumerators, and the per-domain weight calibration
(G8) is a future task.

See ``docs/SECTION_HEADING_PRIMITIVES_RESEARCH.md`` for the full design.

Example
-------
>>> from kaos_nlp_core.structure import label_lines
>>> text = "DISCUSSION\\n\\nThe court ruled today.\\n"
>>> result = label_lines(text)
>>> result.labels
['heading', 'blank', 'body']
>>> [c.score for c in result.candidates]  # doctest: +ELLIPSIS
[...]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kaos_nlp_core._rust.structure import (
    HeadingCandidate as _RustHeadingCandidate,
)
from kaos_nlp_core._rust.structure import (
    HeadingFeatureVector as _RustHeadingFeatureVector,
)
from kaos_nlp_core._rust.structure import py_citation_density as _raw_citation_density
from kaos_nlp_core._rust.structure import py_decode_line_labels as _raw_decode_line_labels
from kaos_nlp_core._rust.structure import py_label_lines as _raw_label_lines
from kaos_nlp_core._rust.structure import (
    py_score_heading_features as _raw_score_heading_features,
)


class LineLabel(StrEnum):
    """Seven-state line-label set."""

    BLANK = "blank"
    HEADING = "heading"
    BODY = "body"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"
    METADATA = "metadata"
    BOILERPLATE = "boilerplate"


@dataclass(frozen=True, slots=True)
class StructureResult:
    """Full pipeline output for one document."""

    labels: list[str]
    features: list[_RustHeadingFeatureVector]
    candidates: list[_RustHeadingCandidate]


def score_heading_features(
    text: str,
    *,
    enum_lexicon: str | None = None,
    custom_enum_lexicon: list[tuple[str, str]] | None = None,
    **scoring: Any,
) -> list[_RustHeadingFeatureVector]:
    """Score every line in ``text`` and return one feature vector per line.

    Parameters
    ----------
    text:
        Source text. Line records, enumerators, and boilerplate runs are
        computed internally.
    enum_lexicon:
        P3 word-prefix lexicon name. One of ``english_legal_us`` (default),
        ``french_legal``, ``german_legal``, ``spanish_legal``,
        ``italian_legal``, ``portuguese_legal``, ``markdown_atx``.
    custom_enum_lexicon:
        Caller-supplied ``[(pattern, kind), ...]`` overrides ``enum_lexicon``.
    **scoring:
        Forwarded to :class:`ScoringOptions` (heading_lexicon,
        hierarchy_lexicon, weights, threshold, short_line_chars,
        long_prose_chars, max_heading_indent).

    Returns
    -------
    list[HeadingFeatureVector]
        One vector per physical line in ``text``.
    """
    return _raw_score_heading_features(
        text,
        enum_lexicon=enum_lexicon,
        custom_enum_lexicon=custom_enum_lexicon,
        **scoring,
    )


def decode_line_labels(
    text: str,
    *,
    enum_lexicon: str | None = None,
    custom_enum_lexicon: list[tuple[str, str]] | None = None,
    scoring: dict[str, Any] | None = None,
    decoder: dict[str, Any] | None = None,
) -> list[str]:
    """Decode the most-likely line-label sequence over ``text``.

    Returns one label string per physical line, drawn from the
    :class:`LineLabel` set.
    """
    return _raw_decode_line_labels(
        text,
        enum_lexicon=enum_lexicon,
        custom_enum_lexicon=custom_enum_lexicon,
        scoring=scoring,
        decoder=decoder,
    )


def label_lines(
    text: str,
    *,
    enum_lexicon: str | None = None,
    custom_enum_lexicon: list[tuple[str, str]] | None = None,
    scoring: dict[str, Any] | None = None,
    decoder: dict[str, Any] | None = None,
) -> StructureResult:
    """Run the full pipeline and return labels + features + heading
    candidates.
    """
    raw = _raw_label_lines(
        text,
        enum_lexicon=enum_lexicon,
        custom_enum_lexicon=custom_enum_lexicon,
        scoring=scoring,
        decoder=decoder,
    )
    return StructureResult(
        labels=list(raw["labels"]),
        features=list(raw["features"]),
        candidates=list(raw["candidates"]),
    )


def citation_density(text: str) -> float:
    """Return the citation-density signal in ``[0, 1]``.

    Defined as the fraction of whitespace-separated tokens that look
    like citation abbreviations (contain ``§`` or are 2-8 ASCII chars
    ending in ``.``). Language-agnostic — covers English Bluebook,
    French ``art.``, German ``Bd.``, Spanish ``art.``, etc.
    """
    return _raw_citation_density(text)


@dataclass(frozen=True, slots=True)
class OutlineNode:
    """One node in the document outline tree.

    Each node corresponds to a heading line. ``section_start`` /
    ``section_end`` are the line indices that bracket this heading's
    content (inclusive heading line, exclusive end). ``children``
    contains nested outline nodes (subheadings of this heading).
    """

    line_index: int
    """Index of the heading line in the source text."""
    text: str
    """The heading line text (stripped)."""
    depth: int
    """Picked depth from `HeadingCandidate.picked_depth()`. 0 means
    no depth signal (rare — implies layout-only heading)."""
    score: float
    """Composite heading score for transparency."""
    section_start: int
    """First line index of the section this heading owns
    (== line_index)."""
    section_end: int
    """Last line index of the section (inclusive). Bounded by the
    next heading at the same or shallower depth, or end of document."""
    enumerator_kind: str | None
    """The enumerator kind if any (`decimal`, `roman_upper`, …)."""
    children: list[OutlineNode]
    """Nested headings of greater depth that fall within this section."""


@dataclass(frozen=True, slots=True)
class StructureSummary:
    """Document-shape summary for an agent or quick triage.

    Intentionally flat — every field is a primitive or a small dict —
    so a model can reason about it directly.
    """

    n_lines: int
    """Total physical-line count."""
    label_counts: dict[str, int]
    """Count per label class. Keys: ``blank``, ``heading``, ``body``,
    ``list_item``, ``table_row``, ``metadata``, ``boilerplate``."""
    n_headings: int
    """``label_counts['heading']`` (convenience)."""
    max_depth: int
    """Maximum heading depth observed (from ``picked_depth``).
    Zero if no heading carried a depth signal."""
    has_metadata_block: bool
    """True iff at least 2 metadata lines appeared in the first 30
    lines — typical of contracts / forms with a front-matter block."""
    has_boilerplate: bool
    """True iff at least one boilerplate line was detected."""
    has_table_rows: bool
    """True iff at least one table-row was detected."""
    looks_like_form: bool
    """True iff form-field shape was detected on multiple lines
    (placeholders like ``<Name>``, ``Field: ____``)."""
    dominant_label: str
    """The non-blank label with the highest count (typical: ``body``).
    Useful for shape-of-document classification."""
    body_ratio: float
    """Fraction of non-blank lines labeled body — `0.0` (highly
    structured) to `1.0` (pure prose)."""


def build_outline(
    text: str,
    *,
    enum_lexicon: str | None = None,
    custom_enum_lexicon: list[tuple[str, str]] | None = None,
    scoring: dict[str, Any] | None = None,
    decoder: dict[str, Any] | None = None,
) -> list[OutlineNode]:
    """Run the full pipeline and return a heading hierarchy as a tree.

    Each top-level entry is a heading at the shallowest detected
    depth. ``section_start`` / ``section_end`` bracket the lines
    each heading owns (inclusive of the heading line, exclusive of
    the next same-or-shallower heading or end of doc).

    Layout-only headings (no enumerator, no hierarchy keyword,
    `picked_depth() == 0`) are placed at depth 1 by convention so
    they appear at the top level of the tree.
    """
    result = label_lines(
        text,
        enum_lexicon=enum_lexicon,
        custom_enum_lexicon=custom_enum_lexicon,
        scoring=scoring,
        decoder=decoder,
    )
    return _build_outline_tree(text, result)


def summarize_structure(
    text: str,
    *,
    enum_lexicon: str | None = None,
    custom_enum_lexicon: list[tuple[str, str]] | None = None,
    scoring: dict[str, Any] | None = None,
    decoder: dict[str, Any] | None = None,
) -> StructureSummary:
    """Compact document-shape summary for agent triage.

    Single call returns counts + booleans an agent can use to decide
    "is this a form, a contract, a regulation, prose?" without
    walking the per-line labels itself.
    """
    result = label_lines(
        text,
        enum_lexicon=enum_lexicon,
        custom_enum_lexicon=custom_enum_lexicon,
        scoring=scoring,
        decoder=decoder,
    )
    return _summarize(text, result)


def _build_outline_tree(text: str, result: StructureResult) -> list[OutlineNode]:
    """Convert a flat label sequence + heading candidates into a tree."""
    if not result.candidates:
        return []
    lines = text.split("\n")
    candidates_sorted = sorted(result.candidates, key=lambda c: c.line_index)

    # Pre-compute depth and section_end per candidate.
    flat: list[OutlineNode] = []
    for k, c in enumerate(candidates_sorted):
        line_idx = int(c.line_index)
        depth = c.picked_depth()
        if depth == 0:
            depth = 1  # layout-only heading — place at top level
        # section_end: the line BEFORE the next same-or-shallower heading
        # (or end of doc).
        end = len(result.labels) - 1
        for next_c in candidates_sorted[k + 1 :]:
            next_depth = next_c.picked_depth() or 1
            if next_depth <= depth:
                end = int(next_c.line_index) - 1
                break
        flat.append(
            OutlineNode(
                line_index=line_idx,
                text=lines[line_idx].strip() if line_idx < len(lines) else "",
                depth=int(depth),
                score=float(c.score),
                section_start=line_idx,
                section_end=end,
                enumerator_kind=c.enumerator_kind,
                children=[],
            )
        )

    # Build the tree by depth: walk in source order, pop the stack to a
    # parent shallower than current depth, append.
    stack: list[OutlineNode] = []
    roots: list[OutlineNode] = []
    for node in flat:
        # Pop until top of stack is shallower than `node.depth`.
        while stack and stack[-1].depth >= node.depth:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def _summarize(text: str, result: StructureResult) -> StructureSummary:
    labels = result.labels
    label_counts = dict.fromkeys(
        (
            "blank",
            "heading",
            "body",
            "list_item",
            "table_row",
            "metadata",
            "boilerplate",
        ),
        0,
    )
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    non_blank = sum(c for label, c in label_counts.items() if label != "blank")
    body_count = label_counts.get("body", 0)
    body_ratio = body_count / non_blank if non_blank else 0.0

    max_depth = 0
    for c in result.candidates:
        d = c.picked_depth()
        if d > max_depth:
            max_depth = d

    # Metadata-block heuristic: ≥2 metadata lines in first 30 records.
    head_window = labels[:30]
    has_metadata_block = sum(1 for label in head_window if label == "metadata") >= 2

    # Form-field heuristic via the per-line feature vector.
    n_form_field = sum(1 for f in result.features if f.form_field_shape == 1)
    looks_like_form = n_form_field >= 3

    # Dominant label among non-blank.
    if non_blank:
        non_blank_counts = {label: c for label, c in label_counts.items() if label != "blank"}
        dominant = max(non_blank_counts.items(), key=lambda kv: kv[1])[0]
    else:
        dominant = "blank"

    return StructureSummary(
        n_lines=len(labels),
        label_counts=label_counts,
        n_headings=label_counts.get("heading", 0),
        max_depth=int(max_depth),
        has_metadata_block=has_metadata_block,
        has_boilerplate=label_counts.get("boilerplate", 0) > 0,
        has_table_rows=label_counts.get("table_row", 0) > 0,
        looks_like_form=looks_like_form,
        dominant_label=dominant,
        body_ratio=body_ratio,
    )


__all__ = [
    "HeadingCandidate",
    "HeadingFeatureVector",
    "LineLabel",
    "OutlineNode",
    "StructureResult",
    "StructureSummary",
    "build_outline",
    "citation_density",
    "decode_line_labels",
    "label_lines",
    "score_heading_features",
    "summarize_structure",
]


# Re-export raw classes so consumers can use them as types.
HeadingFeatureVector = _RustHeadingFeatureVector
HeadingCandidate = _RustHeadingCandidate
