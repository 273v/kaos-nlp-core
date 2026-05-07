"""End-to-end integration: every kaos-nlp-core segmentation primitive
chained on a real USC fixture.

This is the P6 acceptance test. It verifies the five primitives compose
without crashing on real legal text and produce sensible outputs.
Specifically:

1. **LineRecord** — extract physical-line records.
2. **Normalize** — canonicalize each non-blank line.
3. **Enumerator** — try to parse a leading enumerator on each line.
4. **Boilerplate** — detect repeated runs across multiple sections.
5. **SpanIndex** — index the (start, end, label) tuples for each
   detected enumerator and run a containing-offset query.

The test does not pin exact output (USC content is large and fluid);
instead it asserts shape invariants and counts that prove every
primitive ran end-to-end without breaking.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaos_nlp_core.segmentation import (
    detect_boilerplate,
    extract_line_records,
    normalize,
    parse_enumerator,
)
from kaos_nlp_core.structures import SpanIndex

_FIXTURE = Path(__file__).parent / "fixtures" / "usc.jsonl"


@pytest.fixture
def usc_text() -> str:
    """Concatenate the first 100 USC sections, form-feed separated, into a
    single long document. 100 sections is enough to exercise boilerplate +
    enumerator parsing on real legal hierarchy."""
    if not _FIXTURE.exists():
        pytest.skip("usc.jsonl fixture missing")
    pieces: list[str] = []
    with _FIXTURE.open() as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            d = json.loads(line)
            pieces.append(d.get("text", ""))
    if not pieces:
        pytest.skip("usc.jsonl had no records")
    return "\f".join(pieces)


def test_full_pipeline_runs_clean(usc_text: str) -> None:
    """Every primitive runs without exception and returns sensible counts."""
    # 1. LineRecord
    records = extract_line_records(usc_text)
    assert len(records) > 100, "expected at least 100 line records"
    blank_count = sum(1 for r in records if r.blank)
    assert blank_count >= 1
    # All offsets bounded in source.
    for r in records:
        assert 0 <= r.start <= r.end <= len(usc_text)

    # 2. Normalize a sample of non-blank lines (whole doc is too big).
    sampled_normalizations = 0
    for r in records[:200]:
        if r.blank:
            continue
        line_text = usc_text[r.start : r.end]
        n = normalize(
            line_text,
            collapse_whitespace=True,
            fold_case=True,
            normalize_unicode_punct=True,
        )
        assert n.text  # non-empty result
        sampled_normalizations += 1
    assert sampled_normalizations > 0

    # 3. Enumerator parser on every non-blank line; collect successes.
    enumerator_hits: list[tuple[int, int, str]] = []  # (line_idx, value, kind)
    for line_idx, r in enumerate(records):
        if r.blank:
            continue
        stripped = usc_text[r.stripped_start : r.stripped_end]
        e = parse_enumerator(stripped)
        if e is None:
            continue
        enumerator_hits.append((line_idx, e.value, e.kind))
    # USC has a lot of `Sec.` / `§` / numeric headings — expect a non-trivial
    # number of successful parses.
    assert len(enumerator_hits) > 10, (
        f"expected >10 enumerators in 100 USC sections; got {len(enumerator_hits)}"
    )

    # 4. Boilerplate detector. Concatenated USC has \f between sections, so
    # the form-feed bucketing path fires.
    runs = detect_boilerplate(usc_text)
    # We don't assert exact counts — USC sections vary — but the call must
    # complete. Each detected run must report valid line indices.
    for run in runs:
        assert run.occurrences == len(run.line_indices)
        assert all(idx < len(records) for idx in run.line_indices)

    # 5. SpanIndex over the enumerator hits + a containing-offset query.
    idx = SpanIndex()
    label_for_kind: dict[str, int] = {}
    for line_idx, _, kind in enumerator_hits:
        if kind not in label_for_kind:
            label_for_kind[kind] = len(label_for_kind)
        # Use the line's char-offset range as the span.
        r = records[line_idx]
        idx.add(label_for_kind[kind], r.start, r.end)
    assert len(idx) == len(enumerator_hits)
    # Pick a non-blank line we know was indexed.
    if enumerator_hits:
        line_idx = enumerator_hits[len(enumerator_hits) // 2][0]
        r = records[line_idx]
        # Query an offset inside the span — we should find at least one hit
        # (the span itself).
        query_offset = (r.start + r.end) // 2
        hits = idx.containing(query_offset)
        assert len(hits) >= 1


def test_pipeline_round_trip_offsets(usc_text: str) -> None:
    """The byte/char-offset contract holds across the full pipeline."""
    records = extract_line_records(usc_text)
    for r in records[:200]:
        # Python str indexing on the offsets must work without raising.
        line = usc_text[r.start : r.end]
        assert "\n" not in line and "\r" not in line  # terminator excluded
        assert len(line) == r.char_len


def test_pipeline_idempotent(usc_text: str) -> None:
    """Running the pipeline twice on identical input yields identical
    boilerplate runs and identical line-record counts. Determinism is the
    cache-correctness guarantee."""
    r1 = extract_line_records(usc_text)
    r2 = extract_line_records(usc_text)
    assert len(r1) == len(r2)

    b1 = detect_boilerplate(usc_text)
    b2 = detect_boilerplate(usc_text)
    assert [(r.canonical_text, r.occurrences, r.kind) for r in b1] == [
        (r.canonical_text, r.occurrences, r.kind) for r in b2
    ]
