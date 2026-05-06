"""Empirical validation of detect_boilerplate against real PDFs / DOCX
extracted via kaos-pdf and kaos-office.

Two reasons this script exists:

1. The fixture-jsonl files were already pre-cleaned of form-feeds, so we
   never exercised the `\f`-bucketing path on real production input.
2. The kaos-pdf high-level API extracts text **per-page**; the caller is
   responsible for joining. This script tests both joining strategies
   (`\f` between pages — the right way; `\n\n` — a likely user mistake)
   so we can document which the detector tolerates.

Run:
    uv run python scripts/validate_boilerplate_pdf.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from kaos_nlp_core.segmentation import (
    BoilerplateRun,
    detect_boilerplate,
    extract_line_records,
)


def _summary(runs: list[BoilerplateRun]) -> dict:
    by_kind: Counter[str] = Counter()
    for r in runs:
        by_kind[r.kind] += 1
    return dict(by_kind)


def _show(runs: list[BoilerplateRun], k: int = 8) -> None:
    sorted_runs = sorted(runs, key=lambda r: -r.occurrences)
    for i, r in enumerate(sorted_runs[:k]):
        canon = r.canonical_text[:80]
        if len(r.canonical_text) > 80:
            canon += "…"
        print(f"    [{i:2d}] kind={r.kind:11s} occ={r.occurrences:4d}  {canon!r}")


def _try_pdf(path: Path) -> tuple[list[str], int] | None:
    """Return (per-page text list, page count). Return None on extraction error."""
    try:
        from kaos_pdf import extract_page_text, get_page_count  # ty: ignore[unresolved-import]
    except ImportError as exc:
        print(f"  kaos-pdf not available: {exc}")
        return None
    try:
        n = get_page_count(str(path))
    except Exception as exc:
        print(f"  get_page_count error: {exc}")
        return None
    pages: list[str] = []
    for i in range(n):
        try:
            pages.append(extract_page_text(str(path), i))
        except Exception as exc:
            print(f"  page {i} extract error: {exc}")
            pages.append("")
    return pages, n


def validate_pdf(path: Path) -> None:
    print(f"\n=== {path.name} ===")
    res = _try_pdf(path)
    if res is None:
        print("  SKIP — extraction failed")
        return
    pages, n_pages = res
    sizes = [len(p) for p in pages]
    print(f"  pages: {n_pages}")
    print(f"  per-page bytes: min={min(sizes)} max={max(sizes)} mean={sum(sizes) // len(sizes)}")

    # Strategy A: form-feed join (what we documented as the canonical way).
    text_ff = "\f".join(pages)
    n_lines_ff = sum(1 for _ in extract_line_records(text_ff))
    runs_ff = detect_boilerplate(text_ff)
    print(
        f"\n  -- form-feed join ({len(text_ff)} bytes, {n_lines_ff} lines, "
        f"\\f count={text_ff.count(chr(12))}) --"
    )
    print(f"     runs: {len(runs_ff)}, summary: {_summary(runs_ff)}")
    _show(runs_ff, k=8)

    # Strategy B: blank-line join (common naive concat).
    text_nn = "\n\n".join(pages)
    n_lines_nn = sum(1 for _ in extract_line_records(text_nn))
    runs_nn = detect_boilerplate(text_nn)
    print(f"\n  -- \\n\\n join ({len(text_nn)} bytes, {n_lines_nn} lines, no \\f) --")
    print(f"     runs: {len(runs_nn)}, summary: {_summary(runs_nn)}")
    _show(runs_nn, k=8)


def _try_docx(path: Path) -> str | None:
    try:
        from kaos_office import extract_to_markdown  # ty: ignore[unresolved-import]
    except ImportError as exc:
        print(f"  kaos-office not available: {exc}")
        return None
    try:
        return extract_to_markdown(str(path))
    except Exception as exc:
        print(f"  extract error: {exc}")
        return None


def validate_docx(path: Path) -> None:
    print(f"\n=== {path.name} ===")
    text = _try_docx(path)
    if text is None or not text:
        print("  SKIP")
        return
    n_lines = sum(1 for _ in extract_line_records(text))
    runs = detect_boilerplate(text)
    print(f"  bytes: {len(text)}  lines: {n_lines}  runs: {len(runs)}")
    print(f"  summary: {_summary(runs)}")
    _show(runs, k=8)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults resolve sibling-package fixtures relative to this script's
    # location in the monorepo: <repo-root>/kaos-nlp-core/scripts/<file>.
    # In a per-module repo (kaos-nlp-core split), these siblings don't
    # exist on disk; pass --pdf-fixtures / --docx-fixtures explicitly to
    # point at a local checkout. See docs/oss/checklists/per-package-release.md
    # Phase A inventory step.
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument(
        "--pdf-fixtures",
        type=Path,
        default=repo_root / "kaos-pdf" / "tests" / "fixtures",
    )
    parser.add_argument(
        "--docx-fixtures",
        type=Path,
        default=repo_root / "kaos-office" / "tests" / "fixtures" / "docx",
    )
    parser.add_argument(
        "--pdfs",
        nargs="*",
        default=[
            "kl3m_court_burns.pdf",
            "kl3m_court_woods.pdf",
            "casd_court_order.pdf",
            "staten_v_united_states.pdf",
            "kl3m_fda_guidance.pdf",
            "gpo_report.pdf",
        ],
    )
    parser.add_argument(
        "--docx",
        nargs="*",
        default=[
            "Toro 2022 Term Loan.docx",
            "MultiParagraphSample.docx",
            "Letter of Commitment for Packaged Furniture Program.docx",
        ],
    )
    args = parser.parse_args(argv)

    for name in args.pdfs:
        p = args.pdf_fixtures / name
        if p.exists():
            validate_pdf(p)
        else:
            print(f"\n=== {name} (skipped — not found) ===")

    for name in args.docx:
        p = args.docx_fixtures / name
        if p.exists():
            validate_docx(p)
        else:
            print(f"\n=== {name} (skipped — not found) ===")


if __name__ == "__main__":
    main()
