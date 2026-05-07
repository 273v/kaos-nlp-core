"""Python-side structure-pipeline benchmarks (pytest-benchmark).

Run with:
    uv run pytest tests/bench_structure.py --benchmark-only
"""

from __future__ import annotations

import pytest

from kaos_nlp_core.structure import (
    decode_line_labels,
    label_lines,
    score_heading_features,
)


def _synthetic_legal_doc(n_pages: int) -> str:
    parts: list[str] = []
    parts.append("UNITED STATES DISTRICT COURT\n")
    parts.append("DISTRICT OF COLUMBIA\n\n")
    parts.append("Author: Jane Doe\nDate: 2026-05-05\nCase Number: 22-1234\n\n")
    for page in range(n_pages):
        parts.append(f"Page {page + 1} of {n_pages}\n\n")
        parts.append("BACKGROUND\n\n")
        parts.append(
            "The plaintiffs filed suit alleging a violation of 5 U.S.C. § 552. "
            "They argued that the agency had failed to produce records timely. "
            "The defendants moved to dismiss on grounds of sovereign immunity.\n\n"
        )
        parts.append("Section 5 Definitions\n\n")
        parts.append("(a) Apples\n(b) Bananas\n(c) Cherries\n\n")
        parts.append("DISCUSSION\n\n")
        parts.append("The court considered each argument in turn.\n")
        parts.append("Col A | Col B | Col C\nRow 1 | Val | Val\nRow 2 | Val | Val\n\n")
        parts.append(f"ORDER {page}\n\n")
        parts.append("It is hereby ordered that the motion is denied.\n")
        if page + 1 < n_pages:
            parts.append("\f")
    return "".join(parts)


@pytest.mark.benchmark(group="structure/score")
@pytest.mark.parametrize("pages", [10, 100, 500])
def test_bench_score(benchmark, pages: int) -> None:
    text = _synthetic_legal_doc(pages)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    benchmark.extra_info["pages"] = pages
    result = benchmark(score_heading_features, text)
    assert result


@pytest.mark.benchmark(group="structure/decode")
@pytest.mark.parametrize("pages", [10, 100, 500])
def test_bench_decode(benchmark, pages: int) -> None:
    text = _synthetic_legal_doc(pages)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    benchmark.extra_info["pages"] = pages
    result = benchmark(decode_line_labels, text)
    assert result


@pytest.mark.benchmark(group="structure/full_pipeline")
@pytest.mark.parametrize("pages", [10, 100, 500])
def test_bench_label_lines(benchmark, pages: int) -> None:
    text = _synthetic_legal_doc(pages)
    benchmark.extra_info["bytes"] = len(text.encode("utf-8"))
    benchmark.extra_info["pages"] = pages
    result = benchmark(label_lines, text)
    assert result.labels
