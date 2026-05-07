#!/usr/bin/env python3
"""Validate the structure layer (P7) against the multi-domain corpus.

Loads each `tests/fixtures/multi_domain/*.txt` + `*.gold.jsonl` pair,
runs the full pipeline, and reports per-domain metrics:

* heading precision / recall / F1
* table-row exclusion accuracy
* per-line overall accuracy

These metrics map to the G7 acceptance numbers in
`docs/SECTION_HEADING_PRIMITIVES_RESEARCH.md` (corrigendum). Sub-bound
results trigger a follow-up: either weight calibration (G8) or a
larger fixture corpus.

Run from the kaos-nlp-core/ directory::

    uv run python scripts/validate_heading.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaos_nlp_core.structure import label_lines

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests" / "fixtures" / "multi_domain"

# Per-fixture pipeline configuration. Keys must match `<name>.txt` /
# `<name>.gold.jsonl` pairs in `tests/fixtures/multi_domain/`.
FIXTURES: list[dict[str, Any]] = [
    # ── Hand-written synthetic Tier-A fixtures (original 5) ──
    {
        "name": "academic_imrad",
        "domain": "academic",
        "enum_lexicon": "english_legal_us",
        "heading_lexicon": "english_academic",
        "hierarchy_lexicon": "english_legal_us",
    },
    {
        "name": "software_readme",
        "domain": "software",
        "enum_lexicon": "markdown_atx",
        "heading_lexicon": "english_software",
        "hierarchy_lexicon": "markdown_atx",
    },
    {
        "name": "de_bgb_section",
        "domain": "de_legal",
        "enum_lexicon": "german_legal",
        "heading_lexicon": "german_legal",
        "hierarchy_lexicon": "german_legal",
    },
    {
        "name": "fr_civil_section",
        "domain": "fr_legal",
        "enum_lexicon": "french_legal",
        "heading_lexicon": "french_legal",
        "hierarchy_lexicon": "french_legal",
    },
    {
        "name": "wikipedia_short",
        "domain": "news",
        "enum_lexicon": "english_legal_us",
        "heading_lexicon": "none",
        "hierarchy_lexicon": "none",
    },
    # ── Real-text Tier-A fixtures (sourced + hand-labeled) ──
    {
        "name": "usc_ch15_military_support",
        "domain": "us_statute",
        "enum_lexicon": "english_legal_us",
        "heading_lexicon": "english_legal_us",
        "hierarchy_lexicon": "markdown_atx",
    },
    {
        "name": "edgar_agreement_002",
        "domain": "us_contract",
        "enum_lexicon": "english_legal_us",
        "heading_lexicon": "english_legal_us",
        "hierarchy_lexicon": "english_legal_us",
    },
    {
        "name": "patent_001",
        "domain": "patent",
        "enum_lexicon": "markdown_atx",
        "heading_lexicon": "english_academic",
        "hierarchy_lexicon": "markdown_atx",
    },
    {
        "name": "gutenberg_war_peace_prose",
        "domain": "literature",
        "enum_lexicon": "english_legal_us",
        "heading_lexicon": "none",
        "hierarchy_lexicon": "none",
    },
    {
        "name": "pdf_staten_v_us_court_order",
        "domain": "us_court_pdf",
        "enum_lexicon": "english_legal_us",
        "heading_lexicon": "english_legal_us",
        "hierarchy_lexicon": "english_legal_us",
    },
    {
        "name": "pdf_fda_guidance_federal_register",
        "domain": "federal_register",
        "enum_lexicon": "english_legal_us",
        "heading_lexicon": "english_legal_us",
        "hierarchy_lexicon": "english_legal_us",
    },
    {
        "name": "docx_form_intervention_planning",
        "domain": "form",
        "enum_lexicon": "english_legal_us",
        "heading_lexicon": "english_legal_us",
        "hierarchy_lexicon": "english_legal_us",
    },
    {
        "name": "docx_cheese_curriculum",
        "domain": "curriculum",
        "enum_lexicon": "english_legal_us",
        "heading_lexicon": "english_legal_us",
        "hierarchy_lexicon": "english_legal_us",
    },
    {
        "name": "docx_multiparagraph_sample",
        "domain": "prose_with_lists",
        "enum_lexicon": "english_legal_us",
        "heading_lexicon": "none",
        "hierarchy_lexicon": "none",
    },
]


@dataclass
class DomainMetrics:
    domain: str
    n_lines: int
    correct: int
    heading_tp: int
    heading_fp: int
    heading_fn: int
    table_row_correct: int
    table_row_total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n_lines if self.n_lines else 0.0

    @property
    def heading_precision(self) -> float:
        d = self.heading_tp + self.heading_fp
        return self.heading_tp / d if d else 0.0

    @property
    def heading_recall(self) -> float:
        d = self.heading_tp + self.heading_fn
        return self.heading_tp / d if d else 0.0

    @property
    def heading_f1(self) -> float:
        p = self.heading_precision
        r = self.heading_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def table_row_exclusion(self) -> float:
        return self.table_row_correct / self.table_row_total if self.table_row_total else 1.0


def evaluate_fixture(fixture: dict[str, Any]) -> DomainMetrics:
    name = fixture["name"]
    text_path = CORPUS / f"{name}.txt"
    gold_path = CORPUS / f"{name}.gold.jsonl"
    text = text_path.read_text(encoding="utf-8")
    gold: list[dict[str, Any]] = [
        json.loads(line) for line in gold_path.read_text().splitlines() if line
    ]

    scoring: dict[str, Any] = {}
    if fixture.get("heading_lexicon"):
        scoring["heading_lexicon"] = fixture["heading_lexicon"]
    if fixture.get("hierarchy_lexicon"):
        scoring["hierarchy_lexicon"] = fixture["hierarchy_lexicon"]

    result = label_lines(
        text,
        enum_lexicon=fixture.get("enum_lexicon"),
        scoring=scoring or None,
    )
    pred = result.labels

    n = min(len(pred), len(gold))
    correct = sum(1 for i in range(n) if pred[i] == gold[i]["label"])
    h_tp = sum(1 for i in range(n) if pred[i] == "heading" and gold[i]["label"] == "heading")
    h_fp = sum(1 for i in range(n) if pred[i] == "heading" and gold[i]["label"] != "heading")
    h_fn = sum(1 for i in range(n) if pred[i] != "heading" and gold[i]["label"] == "heading")
    table_total = sum(1 for i in range(n) if gold[i]["label"] in {"table_row", "list_item"})
    table_correct = sum(
        1
        for i in range(n)
        if gold[i]["label"] in {"table_row", "list_item"} and pred[i] != "heading"
    )

    return DomainMetrics(
        domain=fixture["domain"],
        n_lines=n,
        correct=correct,
        heading_tp=h_tp,
        heading_fp=h_fp,
        heading_fn=h_fn,
        table_row_correct=table_correct,
        table_row_total=table_total,
    )


def main() -> int:
    print(f"Multi-domain validation corpus at {CORPUS}\n")
    rows: list[DomainMetrics] = []
    for fixture in FIXTURES:
        if not (CORPUS / f"{fixture['name']}.txt").exists():
            print(f"  [skip] {fixture['name']}: fixture missing")
            continue
        m = evaluate_fixture(fixture)
        rows.append(m)
        print(
            f"  {m.domain:>10}  acc={m.accuracy:.3f}  "
            f"heading P={m.heading_precision:.3f} R={m.heading_recall:.3f} "
            f"F1={m.heading_f1:.3f}  "
            f"table-row exclusion={m.table_row_exclusion:.3f}  "
            f"({m.n_lines} lines)"
        )

    if rows:
        mean_acc = sum(r.accuracy for r in rows) / len(rows)
        mean_f1 = sum(r.heading_f1 for r in rows) / len(rows)
        print()
        print(f"  mean accuracy={mean_acc:.3f}  mean heading F1={mean_f1:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
