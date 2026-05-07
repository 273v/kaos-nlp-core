#!/usr/bin/env python3
"""Quick inspection of the saved calibration results."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "tests" / "fixtures" / "multi_domain" / "calibration_results.json"


def main() -> int:
    data = json.loads(RESULTS.read_text())
    default = data["default"]
    print(f"Default geo-mean = {default['scores']['geo_mean']:.4f}")
    print(f"Default min-fixture = {default['scores']['min_composite']:.4f}\n")

    print(f"Top-{len(data['top_k'])} configs from N={data['trials']} sweep:")
    print(f"  {'rank':>4} {'geo':>7} {'arith':>7} {'min':>7} {'regressed':>10}")
    for k, t in enumerate(data["top_k"]):
        print(
            f"  {k + 1:>4} {t['geo_mean']:.4f}  {t['arith_mean']:.4f}  "
            f"{t['min_composite']:.4f}  {t['regressed_count']:>10}"
        )

    rb = data.get("robust_best")
    if rb is None:
        print("\nNo robust candidate found — every trial regressed >1 fixture.")
    else:
        print(
            f"\nROBUST-BEST  geo={rb['geo_mean']:.4f}  "
            f"min={rb['min_composite']:.4f}  regressed={rb['regressed_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
