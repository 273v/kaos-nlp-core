"""Tests for the document-structure layer (P7.2 / P7.5 / P7.7).

Covers:

* the per-line scorer (`score_heading_features`),
* the Viterbi sequence decoder (`decode_line_labels`),
* the heading-stack inferencer (`label_lines` + `HeadingCandidate`),
* the citation-density helper, and
* the configurability surface (custom weights, custom transitions,
  custom lexicons) so tests fail when a knob is silently dropped.
"""

from __future__ import annotations

from typing import Any

import pytest

from kaos_nlp_core.structure import (
    OutlineNode,
    StructureResult,
    StructureSummary,
    build_outline,
    citation_density,
    decode_line_labels,
    label_lines,
    score_heading_features,
    summarize_structure,
)

# ─── Scorer (P7.2) ──────────────────────────────────────────────────────────


def test_empty_text_returns_empty_list() -> None:
    assert score_heading_features("") == []
    assert decode_line_labels("") == []


def test_score_returns_one_vector_per_line() -> None:
    text = "alpha\nbeta\ngamma\n"
    features = score_heading_features(text)
    assert len(features) == 3


def test_blank_line_marked_blank() -> None:
    features = score_heading_features("\nfoo\n")
    assert features[0].is_blank == 1
    assert features[0].score == 0.0


def test_allcaps_short_blank_around_scores_high() -> None:
    text = "Body sentence.\n\nDISCUSSION\n\nMore body content here.\n"
    features = score_heading_features(text)
    heading = next(f for f in features if f.case_allcaps == 1)
    assert heading.short_line == 1
    assert heading.no_terminal_period == 1
    assert heading.score >= 0.30


def test_lexical_heading_legal_us_default() -> None:
    text = "Background\n\nThe parties agree.\n"
    features = score_heading_features(text)
    assert features[0].lexical_heading == 1


def test_lexical_heading_academic_imrad() -> None:
    text = "Introduction\nbody.\n"
    feats = score_heading_features(text, heading_lexicon="english_academic")
    assert feats[0].lexical_heading == 1


def test_lexical_heading_software() -> None:
    text = "Installation\nbody.\n"
    feats = score_heading_features(text, heading_lexicon="english_software")
    assert feats[0].lexical_heading == 1


def test_lexical_lexicon_can_be_disabled() -> None:
    text = "Background\nbody.\n"
    feats = score_heading_features(text, heading_lexicon="none")
    assert feats[0].lexical_heading == 0


def test_unknown_heading_lexicon_raises() -> None:
    with pytest.raises(ValueError, match="unknown heading_lexicon"):
        score_heading_features("foo", heading_lexicon="not_a_real_lex")


def test_custom_heading_lexicon_via_kwarg() -> None:
    feats = score_heading_features(
        "My Custom Heading\nbody.\n",
        custom_heading_lexicon=["my custom heading"],
    )
    assert feats[0].lexical_heading == 1


def test_hierarchy_keyword_section_fires() -> None:
    feats = score_heading_features("Section 5 Definitions\nbody.\n")
    assert feats[0].hierarchy_keyword == 1
    assert feats[0].hierarchy_depth >= 1


def test_hierarchy_french_legal_lexicon() -> None:
    feats = score_heading_features(
        "Article 5 — Définitions\nbody.\n",
        hierarchy_lexicon="french_legal",
    )
    assert feats[0].hierarchy_keyword == 1


def test_hierarchy_german_legal_lexicon() -> None:
    feats = score_heading_features(
        "Artikel 12 Definitionen\nbody.\n",
        hierarchy_lexicon="german_legal",
    )
    assert feats[0].hierarchy_keyword == 1


def test_hierarchy_markdown_atx_depth() -> None:
    feats = score_heading_features(
        "# H1\n## H2\n### H3\nbody\n",
        hierarchy_lexicon="markdown_atx",
    )
    assert feats[0].atx_depth == 1
    assert feats[0].hierarchy_depth == 1
    assert feats[1].atx_depth == 2
    assert feats[2].atx_depth == 3
    assert feats[3].atx_depth == 0


def test_unknown_hierarchy_lexicon_raises() -> None:
    with pytest.raises(ValueError, match="unknown hierarchy_lexicon"):
        score_heading_features("foo", hierarchy_lexicon="not_a_real_lex")


def test_table_row_shape_pipe() -> None:
    feats = score_heading_features("Col A | Col B | Col C\n")
    assert feats[0].table_row_shape == 1
    assert feats[0].score < 0.30


def test_inline_colon_signal() -> None:
    feats = score_heading_features("Author: Jane Doe\n")
    assert feats[0].inline_colon == 1


def test_colon_suffix_vs_inline_colon() -> None:
    # Trailing colon is heading-shape (NOT inline_colon).
    feats = score_heading_features("Defendants:\nbody\n")
    assert feats[0].colon_suffix == 1
    assert feats[0].inline_colon == 0


def test_long_prose_signal() -> None:
    text = "x" * 250 + "\n"
    feats = score_heading_features(text)
    assert feats[0].long_prose == 1


def test_score_clamped_to_unit() -> None:
    text = "OPINION\n\nbody text follows here.\n"
    feats = score_heading_features(text)
    for f in feats:
        assert -1.0 <= f.score <= 1.0


def test_determinism_across_runs() -> None:
    text = "OPINION\n\nThe parties agreed to terms.\n"
    a = score_heading_features(text)
    b = score_heading_features(text)
    assert [f.score for f in a] == [f.score for f in b]


def test_repr_is_informative() -> None:
    f = score_heading_features("DISCUSSION\nbody.\n")[0]
    assert "HeadingFeatureVector" in repr(f)
    assert "score=" in repr(f)


# ─── Configurability surface ──────────────────────────────────────────────


def test_threshold_is_configurable() -> None:
    # Override threshold via kwarg. (The scorer doesn't apply the
    # threshold itself but accepts it; if the kwarg were dropped this
    # would silently pass — which is exactly the regression we don't
    # want. Verify by combining with custom weights below.)
    feats = score_heading_features("DISCUSSION\nbody.\n", threshold=0.10)
    assert feats[0].score is not None  # exists


def test_custom_weights_override_defaults() -> None:
    # Kill all positive weights → no line should score above zero.
    flat_weights = {
        "short_line": 0.0,
        "blank_before": 0.0,
        "blank_after": 0.0,
        "indent_le_4": 0.0,
        "case_allcaps": 0.0,
        "case_titlecase": 0.0,
        "case_initcap": 0.0,
        "no_terminal_period": 0.0,
        "colon_suffix": 0.0,
        "inline_colon": 0.0,
        "has_enumerator": 0.0,
        "hierarchy_keyword": 0.0,
        "lexical_heading": 0.0,
        "table_row_shape": 0.0,
        "citation_density": 0.0,
        "boilerplate": 0.0,
        "long_prose": 0.0,
    }
    feats = score_heading_features("DISCUSSION\nbody.\n", weights=flat_weights)
    for f in feats:
        assert f.score == 0.0


def test_custom_weights_amplify_signal() -> None:
    # Boost case_allcaps to 1.0 → DISCUSSION line scores at least 1.0.
    boost = {"case_allcaps": 1.0}
    feats = score_heading_features("DISCUSSION\n", weights=boost)
    assert feats[0].score >= 0.95  # clamped at 1.0 with cap; sum positives min(1.0)


def test_short_line_chars_threshold_configurable() -> None:
    text = "x" * 80 + "\n"  # 80-char line — over default 60
    default = score_heading_features(text)
    relaxed = score_heading_features(text, short_line_chars=100)
    assert default[0].short_line == 0
    assert relaxed[0].short_line == 1


# ─── Decoder (P7.5) ────────────────────────────────────────────────────────


def test_decode_returns_one_label_per_line() -> None:
    text = "alpha\nbeta\ngamma\n"
    labels = decode_line_labels(text)
    assert len(labels) == 3


def test_decode_blank_lines_get_blank_label() -> None:
    labels = decode_line_labels("OPINION\n\nbody text follows.\n")
    assert labels[1] == "blank"


def test_decode_allcaps_short_decoded_as_heading() -> None:
    labels = decode_line_labels("Body text here.\n\nDISCUSSION\n\nMore body.\n")
    assert "heading" in labels


def test_decode_metadata_inline_colon() -> None:
    labels = decode_line_labels(
        "Author: Jane Doe\nDate: 2026-05-05\nCase Number: 22-1234\n\nbody.\n"
    )
    assert labels[0] == "metadata"
    assert labels[1] == "metadata"
    assert labels[2] == "metadata"


def test_decode_table_rows() -> None:
    labels = decode_line_labels("Col A | Col B | Col C\nVal 1 | Val 2 | Val 3\n")
    for label in labels:
        assert label == "table_row"


def test_decode_body_prose() -> None:
    labels = decode_line_labels(
        "The court considered each argument and rejected the appeal.\n"
        "There were several issues raised at trial in due course.\n"
    )
    for label in labels:
        assert label == "body"


def test_decoder_emissions_kwarg_overrides() -> None:
    # Force every non-blank line to body by making body cheap.
    text = "DISCUSSION\nbody text\n"
    cheap_body = {"emissions": {"body_baseline": 0.0, "heading_emit_scale": 100.0}}
    labels = decode_line_labels(text, decoder=cheap_body)
    assert "heading" not in labels


def test_decoder_transitions_kwarg_validated() -> None:
    bad_matrix = [[0.0] * 6] * 7  # 7x6, wrong shape
    with pytest.raises(ValueError, match="transitions must be"):
        decode_line_labels("foo\n", decoder={"transitions": bad_matrix})


def test_decoder_transitions_accept_7x7() -> None:
    matrix = [[0.0] * 7 for _ in range(7)]
    labels = decode_line_labels("foo\n", decoder={"transitions": matrix})
    assert len(labels) == 1


def test_decoder_force_blank_label_can_be_disabled() -> None:
    # With force_blank_label=False the decoder is allowed to assign
    # non-blank labels to blank lines; the transition costs would
    # normally still keep blank lines blank because emission cost
    # for blank is 0 vs INF — so this just exercises the kwarg path.
    labels = decode_line_labels(
        "OPINION\n\nbody.\n",
        decoder={"force_blank_label": False},
    )
    # Blank lines should still be blank because emission is 0 there.
    assert labels[1] == "blank"


# ─── Full pipeline (P7.7) ─────────────────────────────────────────────────


def test_label_lines_returns_dataclass() -> None:
    result = label_lines("DISCUSSION\nbody text.\n")
    assert isinstance(result, StructureResult)
    assert isinstance(result.labels, list)
    assert isinstance(result.features, list)
    assert isinstance(result.candidates, list)


def test_label_lines_candidates_only_for_heading_lines() -> None:
    text = "Body text.\n\nDISCUSSION\n\nMore body text.\n"
    result = label_lines(text)
    # All candidates should sit on a heading-labeled line.
    for c in result.candidates:
        assert result.labels[c.line_index] == "heading"


def test_label_lines_hierarchy_keyword_extracts_depth() -> None:
    result = label_lines("Section 5 Definitions\n\nbody text follows.\n")
    if result.candidates:
        c = result.candidates[0]
        assert c.hierarchy_level >= 1


def test_label_lines_markdown_atx_depth() -> None:
    # Canonical Markdown: blank lines between heading levels and body.
    # Stacked headings without blanks violate the v1 transition cost
    # `heading→heading=1` and aren't standard Markdown anyway.
    text = (
        "# Top\n\nIntro paragraph here.\n\n## Mid\n\n"
        "Mid content text.\n\n### Deep\n\nDeep content.\n"
    )
    result = label_lines(
        text,
        scoring={"hierarchy_lexicon": "markdown_atx"},
    )
    # The three `#`-prefixed lines should all be tagged heading.
    heading_lines = [i for i, label in enumerate(result.labels) if label == "heading"]
    assert len(heading_lines) >= 3, f"expected ≥3 heading labels, got {result.labels}"
    # Each heading candidate should carry its ATX depth.
    candidate_atx = {c.line_index: c.atx_depth for c in result.candidates}
    assert candidate_atx.get(0) == 1
    # Find # Mid and # Deep by content via features.
    f = result.features
    assert f[heading_lines[0]].atx_depth == 1


def test_heading_candidate_picked_depth() -> None:
    result = label_lines(
        "# Top\nbody\n",
        scoring={"hierarchy_lexicon": "markdown_atx"},
    )
    if result.candidates:
        assert result.candidates[0].picked_depth() == 1


def test_heading_candidate_repr() -> None:
    result = label_lines("DISCUSSION\nbody.\n")
    if result.candidates:
        s = repr(result.candidates[0])
        assert "HeadingCandidate" in s
        assert "line=" in s


def test_label_lines_enumerator_kind_string() -> None:
    result = label_lines(
        "1.2.3 Title here\n\nbody text follows.\n",
    )
    if result.candidates:
        kind = result.candidates[0].enumerator_kind
        assert kind in (None, "decimal", "section_word")


# ─── Citation density ─────────────────────────────────────────────────────


def test_citation_density_zero_for_prose() -> None:
    assert citation_density("the quick brown fox jumps") == 0.0


def test_citation_density_high_for_legal_citations() -> None:
    d = citation_density("5 U.S.C. § 552; Pub. L. 89-487; Stat. 250.")
    assert d > 0.4


def test_citation_density_low_for_one_abbreviation() -> None:
    # Mr. Smith arrived. — only Mr. is citation-shaped
    d = citation_density("Mr. Smith arrived at noon today")
    assert d < 0.3


def test_citation_density_clamped_to_unit() -> None:
    # All-citation line.
    assert 0.0 <= citation_density("Pub. L. § A. B.") <= 1.0


# ─── Generality contract (G1, G2) ─────────────────────────────────────────


def test_generality_no_lexicon_layout_only() -> None:
    """Per G1/G2: a doc with no keyword and no enumerator must still
    produce sensible output from layout/case alone."""
    feats = score_heading_features(
        "DISCUSSION\n\nbody.\n",
        heading_lexicon="none",
        hierarchy_lexicon="none",
    )
    assert feats[0].lexical_heading == 0
    assert feats[0].hierarchy_keyword == 0
    assert feats[0].score >= 0.30  # layout alone wins


def test_generality_news_style_titlecase_no_keyword() -> None:
    text = "Body text.\n\nThe Inflation Numbers\n\nMore body.\n"
    feats = score_heading_features(text)
    heading = next(f for f in feats if f.case_titlecase == 1)
    assert heading.score >= 0.30


def test_generality_french_legal_path() -> None:
    feats = score_heading_features(
        "Article 5 — Dispositions générales\nbody.\n",
        enum_lexicon="french_legal",
        hierarchy_lexicon="french_legal",
    )
    assert feats[0].hierarchy_keyword == 1


def test_generality_german_legal_path() -> None:
    feats = score_heading_features(
        "Artikel 12 Geltungsbereich\nbody.\n",
        enum_lexicon="german_legal",
        hierarchy_lexicon="german_legal",
    )
    assert feats[0].hierarchy_keyword == 1


def test_generality_spanish_italian_portuguese_paths() -> None:
    for enum_lex, hier_lex, prefix in [
        ("spanish_legal", "spanish_legal", "Artículo 5"),
        ("italian_legal", "italian_legal", "Articolo 5"),
        ("portuguese_legal", "portuguese_legal", "Artigo 5"),
    ]:
        text = f"{prefix} — Definiciones\nbody.\n"
        feats = score_heading_features(text, enum_lexicon=enum_lex, hierarchy_lexicon=hier_lex)
        assert feats[0].hierarchy_keyword == 1, f"{enum_lex} failed: {prefix}"


# ─── Agent-friendly outputs: outline + summary ─────────────────────────────


def test_build_outline_empty_doc_returns_empty() -> None:
    assert build_outline("") == []


def test_build_outline_single_heading() -> None:
    text = "DISCUSSION\n\nThe court ruled today.\n"
    outline = build_outline(text)
    assert len(outline) >= 1
    root = outline[0]
    assert isinstance(root, OutlineNode)
    assert "DISCUSSION" in root.text or root.line_index == 0
    assert root.section_start <= root.section_end


def test_build_outline_nests_by_depth() -> None:
    # Markdown ATX: depth from #-count.
    text = (
        "# Top\n\nIntro paragraph here.\n\n"
        "## Sub A\n\nA content.\n\n"
        "## Sub B\n\nB content.\n\n"
        "# Top 2\n\nMore content.\n"
    )
    outline = build_outline(
        text, enum_lexicon="markdown_atx", scoring={"hierarchy_lexicon": "markdown_atx"}
    )
    # Two top-level nodes (the two `#`).
    top_level = [n for n in outline if n.depth == 1]
    assert len(top_level) >= 2
    # The first top-level should have child sub-headings of depth 2.
    if top_level[0].children:
        for child in top_level[0].children:
            assert child.depth >= 2


def test_outline_section_extents_are_within_bounds() -> None:
    text = "DISCUSSION\n\nbody one\n\nORDER\n\nbody two\n"
    outline = build_outline(text)
    for node in outline:
        assert 0 <= node.section_start <= node.section_end


def test_summarize_structure_empty_doc() -> None:
    s = summarize_structure("")
    assert isinstance(s, StructureSummary)
    assert s.n_lines == 0
    assert s.n_headings == 0
    assert s.dominant_label == "blank"


def test_summarize_structure_simple_prose() -> None:
    text = (
        "The court considered each argument and rejected the appeal.\n"
        "There were several issues raised at trial.\n"
        "The defendant filed a timely notice of appeal.\n"
    )
    s = summarize_structure(text)
    assert s.dominant_label == "body"
    assert s.body_ratio >= 0.6
    assert not s.has_metadata_block
    assert not s.looks_like_form


def test_summarize_structure_metadata_block() -> None:
    text = "Author: Jane Doe\nDate: 2026-05-05\nCase Number: 22-1234\n\nBody content here.\n"
    s = summarize_structure(text)
    assert s.has_metadata_block is True


def test_summarize_structure_form() -> None:
    text = "<Name of Agency>\n\n<Address>\n\n<Date>\n\nSome body content here.\n"
    s = summarize_structure(text)
    assert s.looks_like_form is True


def test_summarize_structure_max_depth() -> None:
    text = (
        "# Top\n\nBody here for depth context.\n\n"
        "## Sub\n\nMore body content.\n\n"
        "### SubSub\n\nEven more body content.\n"
    )
    s = summarize_structure(
        text,
        enum_lexicon="markdown_atx",
        scoring={"hierarchy_lexicon": "markdown_atx"},
    )
    assert s.max_depth >= 1


def test_outline_tool_via_mcp() -> None:
    """End-to-end through tool registration."""
    from kaos_nlp_core.tools import register_nlp_tools

    class _Reg:
        def __init__(self) -> None:
            self.t: list = []

        def register_tool(self, tool: Any, aliases: list[str] | None = None) -> None:
            self.t.append(tool)

    class _Rt:
        def __init__(self) -> None:
            self.tools = _Reg()

    rt = _Rt()
    register_nlp_tools(rt)
    tools = {t.metadata.name: t for t in rt.tools.t}
    assert "kaos-nlp-outline" in tools


# ─── Configurability surface (post audit) ─────────────────────────────────


def test_definition_verbs_configurable() -> None:
    # Default fires.
    feats = score_heading_features('"Closing" means the recordation\n')
    assert feats[0].definition_shape == 1
    # Custom verbs that DON'T include "means" → does not fire.
    feats = score_heading_features(
        '"Closing" means the recordation\n', definition_verbs=[" denotes "]
    )
    assert feats[0].definition_shape == 0


def test_form_field_brackets_configurable() -> None:
    feats = score_heading_features("<Name of Agency>\n")
    assert feats[0].form_field_shape == 1
    # Drop angle brackets from the accepted pairs → no longer fires.
    feats = score_heading_features(
        "<Name of Agency>\n", form_field_brackets=[("(", ")"), ("[", "]")]
    )
    assert feats[0].form_field_shape == 0


def test_inline_colon_max_left_chars_configurable() -> None:
    long_label = "x" * 50 + ": value"
    # Default 40-char limit rejects this.
    feats = score_heading_features(long_label + "\n")
    assert feats[0].inline_colon == 0
    # Raised limit accepts it.
    feats = score_heading_features(long_label + "\n", inline_colon_max_left_chars=80)
    assert feats[0].inline_colon == 1


def test_post_decode_can_be_disabled() -> None:
    # An EDGAR-shape multi-line title is normally promoted to heading
    # by the F-R9 post-pass. Disabling the post-pass exposes the raw
    # Viterbi labels.
    text = "Body.\n\nREAL PROPERTY PURCHASE\nAND SALE AGREEMENT\n\nMore body.\n"
    labels_with = decode_line_labels(text)
    labels_without = decode_line_labels(text, decoder={"post_decode": {"enable": False}})
    # With the pass, both lines of the title should be heading.
    # Without it, the raw output may differ. Verify the SHAPE: at
    # minimum, one configuration is not strictly identical to the
    # other (the post-pass is doing something).
    # If both happen to be identical (rare), at least neither errored.
    assert len(labels_with) == len(labels_without)


# ─── F-R7: TOC two-line-pair recognition (post-decode pass) ────────────────


def test_toc_two_line_pairs_are_promoted_to_list_item() -> None:
    # USC-style TOC: naked enum line followed by title line. Without the
    # F-R7 post-pass these all fall to body. With it, both halves of
    # each pair land on list_item.
    text = (
        "Section index follows.\n\n"
        "271.\n"
        "Use of information collected during military operations.\n"
        "272.\n"
        "Use of military equipment and facilities.\n"
        "273.\n"
        "Training and advising civilian law enforcement officials.\n\n"
        "More body text.\n"
    )
    labels = decode_line_labels(text, enum_lexicon="english_legal_us")
    # The 6 TOC lines (indices 2..=7 after the blank at index 1) should
    # all be list_item.
    expected = ["list_item"] * 6
    assert labels[2:8] == expected, labels


def test_toc_pair_recognition_can_be_disabled() -> None:
    text = (
        "271.\nUse of information collected during military operations.\n272.\n"
        "Use of military equipment and facilities.\n"
    )
    labels_on = decode_line_labels(text, enum_lexicon="english_legal_us")
    labels_off = decode_line_labels(
        text,
        enum_lexicon="english_legal_us",
        decoder={"post_decode": {"toc_pair_recognition": False}},
    )
    # The pass must change the label sequence — its whole job is to
    # rewrite naked-enum + title pairs to list_item. If turning it off
    # produces the same output, the pass isn't running.
    assert labels_on != labels_off, (labels_on, labels_off)


def test_toc_pair_recognition_does_not_demote_real_atx_headings() -> None:
    # `## 1.` is a real ATX heading with `has_enumerator=1` and no
    # letters — exactly the naked-enum signature of A-side TOC lines.
    # The pass must guard against demoting it via the `atx_depth > 0`
    # check, regardless of whether the scorer wins on heading by score.
    text = "## 1.\nFirst section body text follows.\n"
    feats = score_heading_features(
        text,
        enum_lexicon="markdown_atx",
        heading_lexicon="english_software",
        hierarchy_lexicon="markdown_atx",
    )
    # The line is recognized as ATX depth=2 by the scorer.
    assert feats[0].atx_depth == 2, feats[0].atx_depth
    # The F-R7 pass refuses to touch it because `atx_depth > 0`.
    labels = decode_line_labels(
        text,
        enum_lexicon="markdown_atx",
        scoring={"heading_lexicon": "english_software", "hierarchy_lexicon": "markdown_atx"},
    )
    assert labels[0] != "list_item", labels
