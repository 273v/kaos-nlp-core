#!/usr/bin/env python3
"""Compare targeted weight/emission changes against defaults.

Random search shows that two signals dominate the per-fixture wins:

* ``column_gap_only`` (default -0.10) more negative — kills column-wrap
  PDF over-fire (federal_register).
* ``list_item_strong`` (default 0.30) lower — lets enumerated short lines
  win against body baseline (us_statute).

This script tests each change in isolation and combined, against
defaults, on the multi-domain corpus. The goal is a Pareto improvement:
some fixtures gain, none regress.

Run from ``kaos-nlp-core/`` ::

    uv run python scripts/test_targeted_changes.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from calibrate_weights import (
    DEFAULT_EMISSIONS,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    FIXTURES,
    _evaluate_config,
)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests" / "fixtures" / "multi_domain"


def _load_fixtures() -> list[tuple[dict[str, Any], str, list[dict[str, Any]]]]:
    out = []
    for fixture in FIXTURES:
        text_path = CORPUS / f"{fixture['name']}.txt"
        gold_path = CORPUS / f"{fixture['name']}.gold.jsonl"
        if not text_path.exists() or not gold_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8")
        gold = [json.loads(line) for line in gold_path.read_text().splitlines() if line.strip()]
        out.append((fixture, text, gold))
    return out


CONFIGS = [
    ("default", DEFAULT_WEIGHTS, DEFAULT_EMISSIONS, DEFAULT_THRESHOLD),
    (
        "column_gap=-0.20",
        {**DEFAULT_WEIGHTS, "column_gap_only": -0.20},
        DEFAULT_EMISSIONS,
        DEFAULT_THRESHOLD,
    ),
    (
        "column_gap=-0.30",
        {**DEFAULT_WEIGHTS, "column_gap_only": -0.30},
        DEFAULT_EMISSIONS,
        DEFAULT_THRESHOLD,
    ),
    (
        "list_item_strong=0.20",
        DEFAULT_WEIGHTS,
        {**DEFAULT_EMISSIONS, "list_item_strong": 0.20},
        DEFAULT_THRESHOLD,
    ),
    (
        "list_item_strong=0.15",
        DEFAULT_WEIGHTS,
        {**DEFAULT_EMISSIONS, "list_item_strong": 0.15},
        DEFAULT_THRESHOLD,
    ),
    (
        "combined: gap=-0.20, list_strong=0.20",
        {**DEFAULT_WEIGHTS, "column_gap_only": -0.20},
        {**DEFAULT_EMISSIONS, "list_item_strong": 0.20},
        DEFAULT_THRESHOLD,
    ),
    (
        "combined: gap=-0.30, list_strong=0.15",
        {**DEFAULT_WEIGHTS, "column_gap_only": -0.30},
        {**DEFAULT_EMISSIONS, "list_item_strong": 0.15},
        DEFAULT_THRESHOLD,
    ),
]


def main() -> int:
    fixtures = _load_fixtures()
    print(f"Loaded {len(fixtures)} fixtures.\n")

    # Run defaults first to use as comparison baseline.
    _default_scores, default_metrics = _evaluate_config(
        fixtures, DEFAULT_WEIGHTS, DEFAULT_EMISSIONS, DEFAULT_THRESHOLD
    )
    default_per_fixture = {m.name: m for m in default_metrics}

    print(f"{'config':>40} {'geo':>7} {'arith':>7} {'min':>7} {'regressed':>10} {'improved':>10}")
    for label, weights, emissions, threshold in CONFIGS:
        scores, metrics = _evaluate_config(fixtures, weights, emissions, threshold)
        regressed = sum(
            1 for m in metrics if m.composite + 1e-6 < default_per_fixture[m.name].composite
        )
        improved = sum(
            1 for m in metrics if m.composite > default_per_fixture[m.name].composite + 1e-6
        )
        print(
            f"{label:>40} {scores.geo_mean:.4f}  {scores.arith_mean:.4f}  "
            f"{scores.min_composite:.4f}  {regressed:>10}/{len(metrics)}  "
            f"{improved:>10}/{len(metrics)}"
        )

    # Detail row for each non-default config.
    print()
    print("Per-fixture deltas from default (composite):")
    print(f"  {'fixture':>22}  {'default':>8}", end="")
    for label, _, _, _ in CONFIGS[1:]:
        print(f"  {label[:18]:>18}", end="")
    print()
    for default_m in default_metrics:
        print(f"  {default_m.name:>22}  {default_m.composite:>7.3f}", end="")
        for _label, weights, emissions, threshold in CONFIGS[1:]:
            _, metrics = _evaluate_config(fixtures, weights, emissions, threshold)
            m = next(x for x in metrics if x.name == default_m.name)
            delta = m.composite - default_m.composite
            sign = "+" if delta >= 0 else ""
            print(f"  {sign}{delta:>17.3f}", end="")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
