#!/usr/bin/env python3
"""Sample REAL text from existing fixture sources into the multi-domain
validation corpus.

Goal: replace the 5 hand-written synthetic fixtures with hand-labeled
slices of real public-domain/permissive text so the G8 calibration
sweep tunes against documents the classifier might actually see.

Sources (all already on disk under kaos-nlp-core/tests/fixtures/):
- usc.jsonl              — US Code chapters / sections (kl3m-data)
- edgar_agreements.jsonl — SEC EDGAR contracts (public filings)
- patents.jsonl          — USPTO patents (public)
- war_and_peace.txt      — Project Gutenberg
- shakespeare.txt        — Project Gutenberg

This script is a one-shot extractor — it only WRITES the .txt fixtures.
Per-line gold labels go in sibling .gold.jsonl files which must be
hand-labeled (NOT synthesized) — see corpus README for the labeling
rules and the labeling pass that follows this build.

Run::

    uv run python scripts/build_multi_domain_corpus.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
CORPUS = FIXTURES / "multi_domain"


# ─── Sourcing rules ───────────────────────────────────────────────────────
#
# Each entry yields one fixture by sampling a JSONL row OR slicing a
# Gutenberg text. We deliberately pick documents of MIXED shape — some
# heavy in TOCs, some pure prose, some heavy with metadata — to stress
# different scorer/decoder branches.
#
# `index` selects a stable row in the JSONL file; if the source ever
# changes the corpus content changes too (re-label required).

JSONL_SAMPLES: list[dict] = [
    # USC — US statutes
    {
        "name": "usc_ch15_military_support",
        "source": "usc.jsonl",
        "index": 0,
        "lines": (0, 66),
        "domain": "us_statute",
    },
    {
        "name": "usc_ch20_humanitarian",
        "source": "usc.jsonl",
        "index": 20,
        "lines": (0, 52),
        "domain": "us_statute",
    },
    {
        "name": "usc_ch23_misc",
        "source": "usc.jsonl",
        "index": 28,
        "lines": (0, 60),
        "domain": "us_statute",
    },
    # EDGAR — US contracts
    {
        "name": "edgar_real_estate_purchase",
        "source": "edgar_agreements.jsonl",
        "index": 0,
        "lines": (0, 80),
        "domain": "us_contract",
    },
    {
        "name": "edgar_agreement_002",
        "source": "edgar_agreements.jsonl",
        "index": 1,
        "lines": (0, 80),
        "domain": "us_contract",
    },
    {
        "name": "edgar_agreement_003",
        "source": "edgar_agreements.jsonl",
        "index": 5,
        "lines": (0, 80),
        "domain": "us_contract",
    },
    # Patents
    {
        "name": "patent_001",
        "source": "patents.jsonl",
        "index": 0,
        "lines": (0, 60),
        "domain": "patent",
    },
    {
        "name": "patent_002",
        "source": "patents.jsonl",
        "index": 5,
        "lines": (0, 60),
        "domain": "patent",
    },
]

# Plain-text slices.
TEXT_SLICES: list[dict] = [
    {
        "name": "gutenberg_war_peace_toc",
        "source": "war_and_peace.txt",
        "lines": (200, 280),  # the chapter-listing TOC
        "domain": "literature",
    },
    {
        "name": "gutenberg_war_peace_prose",
        "source": "war_and_peace.txt",
        "lines": (1500, 1580),  # actual prose, deeper into the book
        "domain": "literature",
    },
    {
        "name": "gutenberg_shakespeare_play_list",
        "source": "shakespeare.txt",
        "lines": (50, 120),
        "domain": "literature",
    },
]


# PDF slices — extracted via kaos-pdf (now that the bbox fix unlocks
# the previously-failing fixtures). Each emits one .txt fixture from
# the first ~80 serialized lines of the PDF. We deliberately span:
# - federal-court orders (caption + opinion shape)
# - FDA Federal Register page (multi-column wrap stress case)
# - GPO report (OCR-heavy stamped historical doc)

PDF_SLICES: list[dict] = [
    {
        "name": "pdf_staten_v_us_court_order",
        "source": "kaos-pdf/tests/fixtures/staten_v_united_states.pdf",
        "lines": (0, 80),
        "domain": "us_court_pdf",
    },
    {
        "name": "pdf_fda_guidance_federal_register",
        "source": "kaos-pdf/tests/fixtures/kl3m_fda_guidance.pdf",
        "lines": (0, 80),
        "domain": "federal_register",
        # Multi-column GPO publication — extract with column-paragraph
        # merging so column-wrap fragments coalesce into flowing text
        # before the structure scorer sees them.
        "merge_column_paragraphs": True,
    },
    {
        "name": "pdf_casd_court_order",
        "source": "kaos-pdf/tests/fixtures/casd_court_order.pdf",
        "lines": (0, 80),
        "domain": "us_court_pdf",
    },
]


# DOCX slices — extracted via kaos-office. Span: forms (with field
# placeholders), policy templates, and a footnote-bearing doc to
# stress the footnote line-mapping path (T3b).

DOCX_SLICES: list[dict] = [
    {
        "name": "docx_form_intervention_planning",
        "source": "kaos-office/tests/fixtures/docx/Burnout_Intervention_Planning_Guide_Fillable_Form_1.docx",
        "lines": (0, 80),
        "domain": "form",
    },
    {
        "name": "docx_policy_template",
        "source": "kaos-office/tests/fixtures/docx/PolicyProcedureTemplate_PhysicalFacility_Final.docx",
        "lines": (0, 80),
        "domain": "policy_template",
    },
    {
        "name": "docx_cheese_curriculum",
        "source": "kaos-office/tests/fixtures/docx/CheeseSample.docx",
        "lines": (0, 80),
        "domain": "curriculum",
    },
    {
        "name": "docx_consumer_rights",
        "source": "kaos-office/tests/fixtures/docx/bcfp_consumer-rights-summary_2018-09.docx",
        "lines": (0, 80),
        "domain": "consumer_legal",
    },
    {
        "name": "docx_multiparagraph_sample",
        "source": "kaos-office/tests/fixtures/docx/MultiParagraphSample.docx",
        "lines": (0, 30),
        "domain": "prose_with_lists",
    },
]


KAOS_MODULES = ROOT.parent  # …/kaos-modules


def _read_jsonl_row(path: Path, index: int) -> dict:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"index {index} past end of {path}")


def _slice_lines(text: str, start: int, end: int) -> str:
    lines = text.split("\n")
    return "\n".join(lines[start:end])


def _extract_pdf_text(path: Path, *, merge_column_paragraphs: bool = False) -> str:
    from kaos_content.serializers.text import serialize_text
    from kaos_pdf import extract_pdf

    doc = extract_pdf(path, merge_column_paragraphs=merge_column_paragraphs)
    return serialize_text(doc)


def _extract_docx_text(path: Path) -> str:
    from kaos_content.serializers.text import serialize_text
    from kaos_office import parse_docx

    doc = parse_docx(path)
    return serialize_text(doc)


def build(dry_run: bool = False) -> None:
    if not dry_run:
        CORPUS.mkdir(parents=True, exist_ok=True)

    written = 0
    for spec in JSONL_SAMPLES:
        out = CORPUS / f"{spec['name']}.txt"
        rec = _read_jsonl_row(FIXTURES / spec["source"], spec["index"])
        text = rec.get("text") or rec.get("content") or ""
        sliced = _slice_lines(text, *spec["lines"])
        n_lines = len(sliced.split("\n"))
        action = "would write" if dry_run else "wrote"
        print(f"  [{spec['domain']:>14}] {action} {out.name} ({n_lines} lines)")
        if not dry_run:
            out.write_text(sliced + "\n", encoding="utf-8")
        written += 1

    for spec in TEXT_SLICES:
        out = CORPUS / f"{spec['name']}.txt"
        text = (FIXTURES / spec["source"]).read_text(encoding="utf-8")
        sliced = _slice_lines(text, *spec["lines"])
        n_lines = len(sliced.split("\n"))
        action = "would write" if dry_run else "wrote"
        print(f"  [{spec['domain']:>16}] {action} {out.name} ({n_lines} lines)")
        if not dry_run:
            out.write_text(sliced + "\n", encoding="utf-8")
        written += 1

    for spec in PDF_SLICES:
        out = CORPUS / f"{spec['name']}.txt"
        src = KAOS_MODULES / spec["source"]
        if not src.exists():
            print(f"  [skip] {out.name}: source PDF missing ({src})")
            continue
        try:
            text = _extract_pdf_text(
                src, merge_column_paragraphs=spec.get("merge_column_paragraphs", False)
            )
        except Exception as exc:
            print(f"  [skip] {out.name}: PDF extract failed ({exc})")
            continue
        sliced = _slice_lines(text, *spec["lines"])
        n_lines = len(sliced.split("\n"))
        action = "would write" if dry_run else "wrote"
        print(f"  [{spec['domain']:>16}] {action} {out.name} ({n_lines} lines)")
        if not dry_run:
            out.write_text(sliced + "\n", encoding="utf-8")
        written += 1

    for spec in DOCX_SLICES:
        out = CORPUS / f"{spec['name']}.txt"
        src = KAOS_MODULES / spec["source"]
        if not src.exists():
            print(f"  [skip] {out.name}: source DOCX missing ({src})")
            continue
        try:
            text = _extract_docx_text(src)
        except Exception as exc:
            print(f"  [skip] {out.name}: DOCX parse failed ({exc})")
            continue
        sliced = _slice_lines(text, *spec["lines"])
        n_lines = len(sliced.split("\n"))
        action = "would write" if dry_run else "wrote"
        print(f"  [{spec['domain']:>16}] {action} {out.name} ({n_lines} lines)")
        if not dry_run:
            out.write_text(sliced + "\n", encoding="utf-8")
        written += 1

    print(f"\n{written} fixture(s) {'planned' if dry_run else 'written'} to {CORPUS}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
