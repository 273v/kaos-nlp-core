#!/usr/bin/env python3
"""Sample the structure-layer output across real local corpora.

For each fixture, prints:
  * the predicted label distribution
  * up to N example lines for each label, with the source text

This script is for HUMAN INSPECTION — it does not assert anything. The
goal is to look at the labels with our own eyes across diverse real
inputs (USC, EDGAR contracts, patents, court PDFs, DOCX) and surface
failure modes that synthetic fixtures cannot.

Run from kaos-nlp-core/::

    uv run python scripts/sample_real_corpora.py [--max-lines 200]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from kaos_nlp_core.structure import label_lines

NLP_CORE_ROOT = Path(__file__).resolve().parent.parent
KAOS_MODULES_ROOT = NLP_CORE_ROOT.parent
PDF_FIXTURES = KAOS_MODULES_ROOT / "kaos-pdf" / "tests" / "fixtures"
OFFICE_FIXTURES = KAOS_MODULES_ROOT / "kaos-office" / "tests" / "fixtures" / "docx"
USC_PATH = NLP_CORE_ROOT / "tests" / "fixtures" / "usc.jsonl"
EDGAR_PATH = NLP_CORE_ROOT / "tests" / "fixtures" / "edgar_agreements.jsonl"
PATENTS_PATH = NLP_CORE_ROOT / "tests" / "fixtures" / "patents.jsonl"


def sample_jsonl(path: Path, n: int) -> list[dict]:
    """Read first ``n`` records from a JSONL file."""
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def render_pdf(path: Path) -> str:
    """Extract a PDF to plain text via kaos-pdf + serialize_text."""
    from kaos_content.serializers.text import serialize_text
    from kaos_pdf import extract_pdf

    doc = extract_pdf(path)
    return serialize_text(doc)


def render_docx(path: Path) -> str:
    from kaos_content.serializers.text import serialize_text
    from kaos_office import parse_docx

    doc = parse_docx(path)
    return serialize_text(doc)


def label_text(text: str, *, max_lines: int) -> tuple[list[str], list[str]]:
    """Run the pipeline and return (lines, labels) trimmed to max_lines."""
    if max_lines is not None and max_lines > 0:
        truncated = "\n".join(text.split("\n")[:max_lines])
        text = truncated
    result = label_lines(text)
    return text.split("\n"), list(result.labels)


def report(fixture: str, lines: list[str], labels: list[str], *, k: int = 4) -> None:
    """Print per-label distribution + up to k examples per label."""
    counts = Counter(labels)
    n = len(labels)
    examples: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for i, (line, lab) in enumerate(zip(lines, labels, strict=False)):
        if len(examples[lab]) < k and line.strip():
            examples[lab].append((i, line))

    print(f"\n{'=' * 72}\n{fixture}\n{'=' * 72}")
    print(f"  {n} lines, label distribution:")
    for label in (
        "blank",
        "heading",
        "body",
        "list_item",
        "table_row",
        "metadata",
        "boilerplate",
    ):
        c = counts.get(label, 0)
        pct = 100.0 * c / n if n else 0.0
        print(f"    {label:>12}  {c:>5}  ({pct:5.1f}%)")

    print()
    for label in (
        "heading",
        "list_item",
        "table_row",
        "metadata",
        "boilerplate",
        "body",
    ):
        ex = examples.get(label, [])
        if not ex:
            continue
        print(f"  -- examples ({label}) --")
        for idx, line in ex:
            disp = line if len(line) <= 110 else line[:107] + "..."
            print(f"    [{idx:>4}] {disp!r}")
        print()


def run_jsonl(path: Path, label: str, n: int, max_lines: int) -> None:
    if not path.exists():
        print(f"  [skip] {path.name}: missing")
        return
    records = sample_jsonl(path, n)
    for i, rec in enumerate(records):
        text = rec.get("text") or rec.get("content") or ""
        if not text:
            continue
        lines, labels = label_text(text, max_lines=max_lines)
        ident = rec.get("identifier") or rec.get("id") or f"#{i}"
        report(f"{label} :: {ident}", lines, labels)


def run_pdfs(names: list[str], max_lines: int) -> None:
    for name in names:
        path = PDF_FIXTURES / name
        if not path.exists():
            print(f"  [skip] {path.name}: missing")
            continue
        try:
            text = render_pdf(path)
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}")
            continue
        lines, labels = label_text(text, max_lines=max_lines)
        report(f"PDF :: {path.name}", lines, labels)


def run_docx(names: list[str], max_lines: int) -> None:
    for name in names:
        path = OFFICE_FIXTURES / name
        if not path.exists():
            print(f"  [skip] {path.name}: missing")
            continue
        try:
            text = render_docx(path)
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}")
            continue
        lines, labels = label_text(text, max_lines=max_lines)
        report(f"DOCX :: {path.name}", lines, labels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lines", type=int, default=200)
    parser.add_argument("--n-jsonl", type=int, default=2)
    args = parser.parse_args()

    # USC sections — statutory text, hierarchy keywords, mixed shape
    run_jsonl(USC_PATH, "USC", args.n_jsonl, args.max_lines)

    # EDGAR agreements — contract shape, exhibit headers, signature blocks
    run_jsonl(EDGAR_PATH, "EDGAR", args.n_jsonl, args.max_lines)

    # Patents — claim language, numbered claims
    run_jsonl(PATENTS_PATH, "PATENT", args.n_jsonl, args.max_lines)

    # PDFs — pick three diverse fixtures
    run_pdfs(
        ["staten_v_united_states.pdf", "kl3m_fda_guidance.pdf", "gpo_report.pdf"],
        args.max_lines,
    )

    # DOCX — pick three diverse fixtures
    run_docx(
        [
            "Toro 2022 Term Loan.docx",
            "PolicyProcedureTemplate_PhysicalFacility_Final.docx",
            "MultiParagraphSample.docx",
        ],
        args.max_lines,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
