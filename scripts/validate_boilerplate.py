"""Empirical validation of detect_boilerplate against real fixtures.

Runs the detector on EDGAR agreements, USC sections, and patents fixtures
that already live in tests/fixtures/. Prints a candid report of what it
finds: detected runs, occurrences, classification, plus a sample of the
underlying line text so a human can eyeball the false-positive / false-
negative rates.

Not a unit test — this is a calibration tool. The numbers below feed back
into the BoilerplateOptions defaults and the design reference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from kaos_nlp_core.segmentation import (
    BoilerplateRun,
    detect_boilerplate,
    extract_line_records,
)


def _summary(runs: list[BoilerplateRun]) -> dict:
    by_kind: Counter[str] = Counter()
    occ_buckets: Counter[str] = Counter()
    for r in runs:
        by_kind[r.kind] += 1
        if r.occurrences < 3:
            occ_buckets["<3"] += 1
        elif r.occurrences < 10:
            occ_buckets["3-9"] += 1
        elif r.occurrences < 100:
            occ_buckets["10-99"] += 1
        else:
            occ_buckets["100+"] += 1
    return {"by_kind": dict(by_kind), "occurrences": dict(occ_buckets)}


def _show_sample(runs: list[BoilerplateRun], k: int = 12) -> None:
    """Print the top-K runs by occurrence with canonical text + classification.

    The reader scans these manually for false positives / negatives.
    """
    sorted_runs = sorted(runs, key=lambda r: -r.occurrences)
    for i, r in enumerate(sorted_runs[:k]):
        # Truncate canonical to keep the report scannable.
        canon = r.canonical_text[:80]
        if len(r.canonical_text) > 80:
            canon += "…"
        print(f"  [{i:2d}] kind={r.kind:11s} occ={r.occurrences:4d}  {canon!r}")


def _run_one(text: str, *, label: str, **kwargs) -> tuple[int, list[BoilerplateRun]]:
    n_lines = sum(1 for _ in extract_line_records(text))
    runs = detect_boilerplate(text, **kwargs)
    return n_lines, runs


def validate_edgar(fixture_path: Path, n_records: int) -> None:
    """Per-document validation. Each EDGAR record is one agreement."""
    print(f"\n=== EDGAR agreements (per-document) — first {n_records} records ===\n")
    with fixture_path.open() as f:
        for idx, line in enumerate(f):
            if idx >= n_records:
                break
            d = json.loads(line)
            text = d.get("text", "")
            n_lines, runs = _run_one(text, label=f"edgar-{idx}")
            summary = _summary(runs)
            print(
                f"[edgar #{idx} id={str(d.get('id', '?'))[:30]}]: "
                f"{len(text)} bytes, {n_lines} lines, {len(runs)} runs"
            )
            print(f"  summary: {summary}")
            _show_sample(runs, k=5)
            print()


def validate_usc_concatenated(fixture_path: Path, n_records: int) -> None:
    """USC sections are tiny on their own; concatenate them to simulate
    a multi-section document or a "title" of the code."""
    print(
        f"\n=== USC concatenated — first {n_records} sections "
        f"(simulates a multi-section document) ===\n"
    )
    pieces: list[str] = []
    with fixture_path.open() as f:
        for idx, line in enumerate(f):
            if idx >= n_records:
                break
            d = json.loads(line)
            pieces.append(d.get("text", ""))
    text = "\n\n".join(pieces)
    n_lines, runs = _run_one(text, label="usc-concat")
    summary = _summary(runs)
    print(f"USC-concat: {len(text)} bytes, {n_lines} lines, {len(runs)} runs")
    print(f"  summary: {summary}")
    _show_sample(runs, k=15)


def validate_patents(fixture_path: Path, n_records: int) -> None:
    print(f"\n=== Patents — first {n_records} records ===\n")
    with fixture_path.open() as f:
        for idx, line in enumerate(f):
            if idx >= n_records:
                break
            d = json.loads(line)
            text = d.get("text", "")
            n_lines, runs = _run_one(text, label=f"patent-{idx}")
            summary = _summary(runs)
            print(
                f"[patent #{idx} id={str(d.get('id', '?'))[:30]}]: "
                f"{len(text)} bytes, {n_lines} lines, {len(runs)} runs"
            )
            print(f"  summary: {summary}")
            _show_sample(runs, k=5)
            print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "fixtures",
    )
    parser.add_argument("--n-edgar", type=int, default=5)
    parser.add_argument("--n-usc", type=int, default=200)
    parser.add_argument("--n-patents", type=int, default=3)
    args = parser.parse_args(argv)

    if args.n_edgar > 0:
        validate_edgar(args.fixtures / "edgar_agreements.jsonl", args.n_edgar)
    if args.n_usc > 0:
        validate_usc_concatenated(args.fixtures / "usc.jsonl", args.n_usc)
    if args.n_patents > 0:
        validate_patents(args.fixtures / "patents.jsonl", args.n_patents)


if __name__ == "__main__":
    main()
