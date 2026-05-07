"""Calibrate expected-range tables for ``kaos_nlp_core.quality``.

Runs ``compute_metrics`` over every fixture relevant to a domain and
emits the 2/98 percentile bounds for each metric. The output is meant
to be pasted into ``LEGAL_RANGES`` / ``GENERAL_RANGES`` in
``python/kaos_nlp_core/quality.py``.

Domains and their fixture sources:

  legal:
    tests/fixtures/usc.jsonl              (US Code sections)
    tests/fixtures/edgar_agreements.jsonl (SEC EDGAR contract filings)

  general:
    tests/fixtures/shakespeare.txt        (Project Gutenberg)
    tests/fixtures/war_and_peace.txt      (Project Gutenberg)
    tests/fixtures/edgar_agreements.jsonl (general prose adjacent)

Usage::

    uv run python scripts/calibrate_quality_ranges.py
    uv run python scripts/calibrate_quality_ranges.py --sample 5000
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from statistics import quantiles
from typing import Any

from kaos_nlp_core.quality import (
    METRIC_WEIGHTS,
    QualityMetrics,
    compute_metrics,
    default_english_wordset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"

DOMAIN_SOURCES: dict[str, list[Path]] = {
    "legal": [
        FIXTURES / "usc.jsonl",
        FIXTURES / "edgar_agreements.jsonl",
    ],
    "general": [
        FIXTURES / "shakespeare.txt",
        FIXTURES / "war_and_peace.txt",
        FIXTURES / "edgar_agreements.jsonl",
    ],
}

# Skip metrics whose ranges are intentionally hand-set rather than
# data-derived (e.g. ratio_format_tokens floors at 0).
_HAND_SET_METRICS = {"ratio_format_tokens"}


def iter_documents(path: Path) -> Iterable[str]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
                if text:
                    yield text
    elif path.suffix == ".txt":
        # Treat the whole file as one document; for Project Gutenberg files
        # this is fine — they have stable internal stats.
        text = path.read_text(encoding="utf-8", errors="replace")
        if text:
            yield text
    else:
        raise ValueError(f"Unsupported fixture format: {path}")


def collect_metrics(
    paths: Iterable[Path],
    *,
    max_docs: int | None,
    lexicon: Any,
) -> list[QualityMetrics]:
    out: list[QualityMetrics] = []
    for path in paths:
        if not path.exists():
            print(f"  warn: missing fixture {path}, skipping")
            continue
        n_local = 0
        for text in iter_documents(path):
            if not text or len(text) < 200:
                # Skip stubs and tiny entries that distort percentiles.
                continue
            out.append(compute_metrics(text, lexicon=lexicon))
            n_local += 1
            if max_docs is not None and len(out) >= max_docs:
                break
        print(f"  {path.name}: {n_local} documents")
        if max_docs is not None and len(out) >= max_docs:
            break
    return out


def metric_values(metrics: list[QualityMetrics], name: str) -> list[float]:
    out: list[float] = []
    for m in metrics:
        v = getattr(m, name, None)
        if v is None or not isinstance(v, (int, float)):
            continue
        if math.isinf(v) or math.isnan(v):
            continue
        out.append(float(v))
    return out


def percentiles(values: list[float], p_low: float, p_high: float) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) < 50:
        return (min(values), max(values))
    qs = quantiles(values, n=100, method="inclusive")
    lo = qs[int(p_low) - 1]
    hi = qs[int(p_high) - 1]
    return (lo, hi)


def calibrate(
    domain: str,
    metrics: list[QualityMetrics],
    *,
    p_low: float,
    p_high: float,
) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for metric in METRIC_WEIGHTS:
        if metric in _HAND_SET_METRICS:
            continue
        values = metric_values(metrics, metric)
        ranges[metric] = percentiles(values, p_low, p_high)
    return ranges


def render_dict(name: str, ranges: dict[str, tuple[float, float]]) -> str:
    lines = [f"{name}: dict[str, tuple[float, float]] = {{"]
    for metric in METRIC_WEIGHTS:
        if metric in _HAND_SET_METRICS:
            continue
        lo, hi = ranges[metric]
        lines.append(f'    "{metric}": ({lo:.6f}, {hi:.6f}),')
    lines.append("}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=2000,
        help="Maximum documents per domain (default 2000).",
    )
    parser.add_argument("--p-low", type=float, default=2.0)
    parser.add_argument("--p-high", type=float, default=98.0)
    parser.add_argument(
        "--domain",
        choices=("legal", "general", "all"),
        default="all",
    )
    args = parser.parse_args(argv)

    print("Loading default English wordset for ratio_in_lexicon ...")
    lex = default_english_wordset()
    print(f"  loaded {len(lex)} keys")

    domains = ["legal", "general"] if args.domain == "all" else [args.domain]

    for domain in domains:
        print(f"\nCalibrating domain {domain!r} ...")
        metrics = collect_metrics(
            DOMAIN_SOURCES[domain],
            max_docs=args.sample,
            lexicon=lex,
        )
        print(f"  total documents: {len(metrics)}")
        ranges = calibrate(domain, metrics, p_low=args.p_low, p_high=args.p_high)
        name = "LEGAL_RANGES" if domain == "legal" else "GENERAL_RANGES"
        print()
        print(render_dict(name, ranges))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
