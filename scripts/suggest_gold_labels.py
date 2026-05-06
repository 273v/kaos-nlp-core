#!/usr/bin/env python3
"""Pre-fill gold-label suggestions for a fixture, line by line.

OUTPUT IS A STARTING POINT, NOT GROUND TRUTH. Every label must be
reviewed and corrected by hand before the file is checked in. The
rules below are intentionally narrow and explainable so the reviewer
sees what shape signal triggered each guess.

The suggester DOES NOT call the kaos-nlp-core classifier — using the
classifier's own output as gold would be circular and would bake in
exactly the failure modes G8 calibration is meant to detect.

Usage::

    uv run python scripts/suggest_gold_labels.py FIXTURE.txt
    # → writes FIXTURE.gold.jsonl with suggestions + a review_needed flag

The reviewer hand-edits the JSONL, removes the flags, and the corpus
is ready to feed the validator.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Principled, transparent rules. Each returns (label, confidence_marker).
# `confidence_marker == "review"` means the rule is a guess and the
# reviewer MUST look at it. `"high"` means the rule is structurally
# unambiguous (blank lines, pipe-table rows).


def _shape(line: str, prev: str | None, next_: str | None) -> tuple[str, str]:
    stripped = line.strip()

    if not stripped:
        return "blank", "high"

    # Pipe-delimited table row — unambiguous shape.
    if stripped.count("|") >= 2:
        return "table_row", "high"

    # Markdown ATX heading.
    if re.match(r"^#{1,6}\s+\S", stripped):
        return "heading", "high"

    # All-caps short — usually a heading. Reviewer must verify.
    if len(stripped) <= 60 and any(c.isalpha() for c in stripped) and stripped == stripped.upper():
        return "heading", "review"

    # Naked numeric or roman enumerator: "1.", "271.", "I.", "iv.",
    # "(a)", "(1)". Often a list-item anchor in a TOC, sometimes a
    # heading number. Reviewer disambiguates.
    if re.match(r"^[\(\[]?(?:\d+|[IVXLCMivxlcm]+|[A-Za-z])[\)\]\.]?\s*$", stripped):
        return "list_item", "review"

    # `Author: Jane Doe`-style metadata: short label + colon mid-line +
    # short value, no terminal period.
    if (
        len(stripped) <= 80
        and ":" in stripped
        and not stripped.endswith(":")
        and not stripped.endswith(".")
    ):
        colon = stripped.find(":")
        if 1 <= colon <= 40:
            return "metadata", "review"

    # Markdown bullet (`-`, `*`, `+`) followed by content.
    if re.match(r"^[\-\*\+]\s+\S", stripped):
        return "list_item", "high"

    # Ordered list `1. text`, `(1) text`, `(a) text`.
    if re.match(r"^[\(\[]?(?:\d+|[ivxIVX]+|[a-zA-Z])[\)\]\.]\s+\S", stripped):
        return "list_item", "review"

    # Long prose / sentence-ending → body.
    if len(stripped) > 80 or stripped.endswith((".", "!", "?", ";")):
        return "body", "review"

    # Short non-period line with no other signal — could be a heading,
    # could be a wrapped body line. Reviewer must decide.
    return "body", "review"


def suggest(text: str) -> list[dict]:
    lines = text.split("\n")
    out: list[dict] = []
    for i, line in enumerate(lines):
        prev = lines[i - 1] if i > 0 else None
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        label, marker = _shape(line, prev, nxt)
        rec: dict = {
            "line_index": i,
            "label": label,
            "hierarchy_level": None,
            "enumerator_kind": None,
        }
        if marker == "review":
            rec["__review_needed__"] = True
            rec["__line_text__"] = line[:120]
        out.append(rec)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fixture", type=Path)
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .gold.jsonl",
    )
    args = p.parse_args()
    if not args.fixture.exists():
        print(f"missing {args.fixture}", file=sys.stderr)
        return 1
    gold = args.fixture.with_suffix(".gold.jsonl")
    if gold.exists() and not args.force:
        print(
            f"refusing to overwrite {gold} (pass --force to replace)",
            file=sys.stderr,
        )
        return 2
    text = args.fixture.read_text(encoding="utf-8")
    records = suggest(text)
    review = sum(1 for r in records if r.get("__review_needed__"))
    print(f"  {len(records)} lines, {review} need review")
    with gold.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → {gold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
