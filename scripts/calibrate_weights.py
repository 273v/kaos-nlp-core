#!/usr/bin/env python3
"""G8 weight calibration sweep over the multi-domain validation corpus.

Random search over ``ScoringWeights`` (20 fields) and ``EmissionCosts``
(9 fields) plus the heading threshold, evaluated on every Tier-A
fixture in ``tests/fixtures/multi_domain``. Reports the default config
score, the best-found config score, and the per-fixture deltas.

The objective is the **geometric mean of per-fixture composite scores**,
where each composite is ``0.5 * accuracy + 0.5 * heading_f1`` for
fixtures with at least one gold heading, and ``accuracy`` only for
fixtures whose gold has zero headings (so 0-gold-heading fixtures can
neither help nor hurt the heading metric, but accuracy still matters).

Geometric mean (instead of arithmetic) is intentional: it punishes a
configuration that improves the average at the cost of any single
domain crashing to zero. We do not want to overfit to the most-common
domain shapes.

Run from ``kaos-nlp-core/`` ::

    uv run python scripts/calibrate_weights.py --trials 300 --seed 42

Outputs ``tests/fixtures/multi_domain/calibration_results.json`` with
the default + best config + top-K candidates and per-fixture metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kaos_nlp_core.structure import label_lines

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests" / "fixtures" / "multi_domain"
OUTPUT = CORPUS / "calibration_results.json"

# Per-fixture pipeline configuration. Mirrors `validate_heading.py`.
FIXTURES: list[dict[str, Any]] = [
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


# Defaults mirrored from `rust/core/structure/scoring.rs::ScoringWeights::default`.
DEFAULT_WEIGHTS: dict[str, float] = {
    "short_line": 0.10,
    "blank_before": 0.10,
    "blank_after": 0.05,
    "indent_le_4": 0.05,
    "case_allcaps": 0.20,
    "case_titlecase": 0.15,
    "case_initcap": 0.05,
    "no_terminal_period": 0.10,
    "colon_suffix": 0.10,
    "inline_colon": -0.30,
    "has_enumerator": 0.30,
    "hierarchy_keyword": 0.30,
    "lexical_heading": 0.25,
    "table_row_shape": -0.50,
    "column_gap_only": -0.10,
    "definition_shape": -0.30,
    "form_field_shape": -0.30,
    "citation_density": -0.30,
    "boilerplate": -0.50,
    "long_prose": -0.30,
}

# Defaults mirrored from `rust/core/structure/decoder.rs::EmissionCosts::default`.
DEFAULT_EMISSIONS: dict[str, float] = {
    "heading_emit_scale": 1.0,
    "body_baseline": 0.6,
    "table_row_strong": 0.2,
    "table_row_weak": 1.5,
    "list_item_strong": 0.3,
    "list_item_with_enumerator": 0.5,
    "list_item_weak": 1.2,
    "metadata_strong": 0.4,
    "metadata_weak": 1.5,
}

DEFAULT_THRESHOLD: float = 0.30


@dataclass(frozen=True, slots=True)
class FixtureMetrics:
    name: str
    domain: str
    n_lines: int
    n_gold_headings: int
    accuracy: float
    heading_precision: float
    heading_recall: float
    heading_f1: float
    composite: float


def _composite(metrics_kwargs: dict[str, Any]) -> float:
    """Per-fixture composite. F1 only counts when ≥1 gold heading."""
    acc = metrics_kwargs["accuracy"]
    n_gold = metrics_kwargs["n_gold_headings"]
    if n_gold == 0:
        return acc
    f1 = metrics_kwargs["heading_f1"]
    return 0.5 * acc + 0.5 * f1


def _evaluate_fixture(
    fixture: dict[str, Any],
    text: str,
    gold: list[dict[str, Any]],
    weights: dict[str, float],
    emissions: dict[str, float],
    threshold: float,
) -> FixtureMetrics:
    scoring: dict[str, Any] = {
        "weights": weights,
        "threshold": threshold,
    }
    if fixture.get("heading_lexicon"):
        scoring["heading_lexicon"] = fixture["heading_lexicon"]
    if fixture.get("hierarchy_lexicon"):
        scoring["hierarchy_lexicon"] = fixture["hierarchy_lexicon"]
    decoder = {"emissions": emissions}

    result = label_lines(
        text,
        enum_lexicon=fixture.get("enum_lexicon"),
        scoring=scoring,
        decoder=decoder,
    )
    pred = result.labels

    n = min(len(pred), len(gold))
    correct = sum(1 for i in range(n) if pred[i] == gold[i]["label"])
    h_tp = sum(1 for i in range(n) if pred[i] == "heading" and gold[i]["label"] == "heading")
    h_fp = sum(1 for i in range(n) if pred[i] == "heading" and gold[i]["label"] != "heading")
    h_fn = sum(1 for i in range(n) if pred[i] != "heading" and gold[i]["label"] == "heading")
    n_gold = sum(1 for g in gold if g["label"] == "heading")

    precision = h_tp / (h_tp + h_fp) if (h_tp + h_fp) else 0.0
    recall = h_tp / (h_tp + h_fn) if (h_tp + h_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = correct / n if n else 0.0

    composite = _composite(
        {
            "accuracy": accuracy,
            "heading_f1": f1,
            "n_gold_headings": n_gold,
        }
    )
    return FixtureMetrics(
        name=fixture["name"],
        domain=fixture["domain"],
        n_lines=n,
        n_gold_headings=n_gold,
        accuracy=accuracy,
        heading_precision=precision,
        heading_recall=recall,
        heading_f1=f1,
        composite=composite,
    )


@dataclass(frozen=True, slots=True)
class ConfigScores:
    geo_mean: float
    """Geometric mean of per-fixture composites — primary objective."""
    min_composite: float
    """Worst per-fixture composite. Robustness signal."""
    arith_mean: float
    """Arithmetic mean — for comparison with geo-mean."""


def _evaluate_config(
    fixtures_data: list[tuple[dict[str, Any], str, list[dict[str, Any]]]],
    weights: dict[str, float],
    emissions: dict[str, float],
    threshold: float,
) -> tuple[ConfigScores, list[FixtureMetrics]]:
    """Evaluate one config; return (aggregate scores, per-fixture)."""
    metrics = [
        _evaluate_fixture(fx, text, gold, weights, emissions, threshold)
        for fx, text, gold in fixtures_data
    ]
    floor = 0.01
    log_sum = sum(math.log(max(m.composite, floor)) for m in metrics)
    geo_mean = math.exp(log_sum / len(metrics))
    arith_mean = sum(m.composite for m in metrics) / len(metrics)
    min_composite = min(m.composite for m in metrics)
    return ConfigScores(
        geo_mean=geo_mean,
        min_composite=min_composite,
        arith_mean=arith_mean,
    ), metrics


def _sample_weights(
    rng: random.Random, base: dict[str, float], *, tight: bool = False
) -> dict[str, float]:
    """Perturb each default; preserve sign of negatives.

    Bounded sampling: positives stay >= 0, negatives stay <= 0. This
    keeps the semantic interpretation of each weight intact (a
    `boilerplate` weight should never become positive — that would say
    "looks like boilerplate → looks like heading").

    ``tight=True`` halves the perturbation radius. Use this to search
    for Pareto-clean refinements close to defaults; the wide setting
    can find higher-scoring configs but with higher overfit risk.
    """
    radius = 0.10 if tight else 0.20
    out = {}
    for k, v in base.items():
        delta = rng.uniform(-radius, radius)
        new = v + delta
        new = max(0.0, new) if v >= 0 else min(0.0, new)
        out[k] = round(new, 4)
    return out


def _sample_emissions(
    rng: random.Random, base: dict[str, float], *, tight: bool = False
) -> dict[str, float]:
    """Multiplicative perturbation of emission costs.

    ``tight=False``: x ← x * U(0.5, 1.8).
    ``tight=True``:  x ← x * U(0.8, 1.25).
    """
    lo, hi = (0.8, 1.25) if tight else (0.5, 1.8)
    out = {}
    for k, v in base.items():
        scale = rng.uniform(lo, hi)
        new = round(v * scale, 4)
        out[k] = max(0.01, new)
    return out


def _sample_threshold(rng: random.Random, base: float, *, tight: bool = False) -> float:
    radius = 0.08 if tight else 0.20
    return round(rng.uniform(max(0.05, base - radius), min(0.95, base + radius)), 4)


def _format_fixture_row(m: FixtureMetrics) -> str:
    return (
        f"  {m.domain:>17}  acc={m.accuracy:.3f}  "
        f"P={m.heading_precision:.3f} R={m.heading_recall:.3f} "
        f"F1={m.heading_f1:.3f}  composite={m.composite:.3f}  "
        f"({m.n_lines} lines, {m.n_gold_headings} gold-h)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials",
        type=int,
        default=300,
        help="number of random configs to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="number of top configs to report",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="skip writing calibration_results.json",
    )
    parser.add_argument(
        "--tight",
        action="store_true",
        help="halve perturbation radius — searches near defaults for Pareto-clean tweaks",
    )
    args = parser.parse_args()

    print(
        f"G8 calibration sweep — {args.trials} trials, seed={args.seed}, "
        f"tight={'yes' if args.tight else 'no'}"
    )
    print(f"Corpus: {CORPUS}\n")

    # Load all fixtures once.
    fixtures_data: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []
    for fixture in FIXTURES:
        text_path = CORPUS / f"{fixture['name']}.txt"
        gold_path = CORPUS / f"{fixture['name']}.gold.jsonl"
        if not text_path.exists() or not gold_path.exists():
            print(f"  [skip] {fixture['name']}: fixture missing")
            continue
        text = text_path.read_text(encoding="utf-8")
        gold = [json.loads(line) for line in gold_path.read_text().splitlines() if line.strip()]
        fixtures_data.append((fixture, text, gold))

    print(f"Loaded {len(fixtures_data)} fixtures.\n")

    # Default config baseline.
    default_scores, default_metrics = _evaluate_config(
        fixtures_data, DEFAULT_WEIGHTS, DEFAULT_EMISSIONS, DEFAULT_THRESHOLD
    )
    print(
        f"DEFAULT  geo-mean={default_scores.geo_mean:.4f}  "
        f"arith-mean={default_scores.arith_mean:.4f}  "
        f"min-fixture={default_scores.min_composite:.4f}"
    )
    for m in default_metrics:
        print(_format_fixture_row(m))
    print()

    default_per_fixture = {m.name: m for m in default_metrics}

    rng = random.Random(args.seed)
    trials: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for trial_idx in range(args.trials):
        weights = _sample_weights(rng, DEFAULT_WEIGHTS, tight=args.tight)
        emissions = _sample_emissions(rng, DEFAULT_EMISSIONS, tight=args.tight)
        threshold = _sample_threshold(rng, DEFAULT_THRESHOLD, tight=args.tight)
        scores, metrics = _evaluate_config(fixtures_data, weights, emissions, threshold)
        regressed = sum(
            1 for m in metrics if m.composite + 1e-6 < default_per_fixture[m.name].composite
        )
        trials.append(
            {
                "trial": trial_idx,
                "geo_mean": scores.geo_mean,
                "arith_mean": scores.arith_mean,
                "min_composite": scores.min_composite,
                "regressed_count": regressed,
                "weights": weights,
                "emissions": emissions,
                "threshold": threshold,
                "per_fixture": [asdict(m) for m in metrics],
            }
        )
        if (trial_idx + 1) % 250 == 0 or (trial_idx + 1) == args.trials:
            elapsed = time.perf_counter() - t0
            best_so_far = max(t["geo_mean"] for t in trials)
            print(
                f"  trial {trial_idx + 1:>4}/{args.trials}  "
                f"best-geo={best_so_far:.4f}  elapsed={elapsed:.1f}s"
            )
    elapsed_total = time.perf_counter() - t0
    print(f"\nSweep complete in {elapsed_total:.1f}s.\n")

    # Sort by geo-mean.
    trials.sort(key=lambda t: t["geo_mean"], reverse=True)
    best_geo = trials[0]
    # Robust-best: highest geo-mean among trials with regressed_count <= 1
    # AND min_composite >= default min. This is the config we'd actually
    # promote — improves the average without breaking any fixture.
    robust_candidates = [
        t
        for t in trials
        if t["regressed_count"] <= 1 and t["min_composite"] >= default_scores.min_composite - 0.02
    ]
    robust_best = robust_candidates[0] if robust_candidates else None

    # Pareto-clean: zero regressions AND geo-mean strictly above default.
    pareto_candidates = [
        t
        for t in trials
        if t["regressed_count"] == 0 and t["geo_mean"] > default_scores.geo_mean + 1e-6
    ]
    pareto_best = pareto_candidates[0] if pareto_candidates else None
    print(
        f"Pareto-clean trials (zero regressions, geo > default): "
        f"{len(pareto_candidates)} / {args.trials}"
    )
    if pareto_best is not None:
        print(
            f"  best geo={pareto_best['geo_mean']:.4f} "
            f"(default {default_scores.geo_mean:.4f}, "
            f"+{pareto_best['geo_mean'] - default_scores.geo_mean:.4f})"
        )
    print()

    def _print_config(label: str, t: dict[str, Any]) -> None:
        print(
            f"{label} geo-mean={t['geo_mean']:.4f}  "
            f"arith-mean={t['arith_mean']:.4f}  "
            f"min-fixture={t['min_composite']:.4f}  "
            f"regressed={t['regressed_count']}/{len(default_metrics)}"
        )
        for row in t["per_fixture"]:
            m = FixtureMetrics(**row)
            d = default_per_fixture[m.name]
            print(
                _format_fixture_row(m) + f"   Δacc={m.accuracy - d.accuracy:+.3f} "
                f"ΔF1={m.heading_f1 - d.heading_f1:+.3f}"
            )
        print()

    _print_config("HIGHEST-GEO ", best_geo)
    if robust_best is not None and robust_best is not best_geo:
        _print_config("ROBUST-BEST  ", robust_best)
    elif robust_best is None:
        print("No robust candidate (every trial regressed >1 fixture or hurt the min).\n")
    if pareto_best is not None and pareto_best is not best_geo and pareto_best is not robust_best:
        _print_config("PARETO-CLEAN ", pareto_best)

    print(f"Top-{args.top_k} configs (geo-mean):")
    for k, t in enumerate(trials[: args.top_k]):
        print(
            f"  #{k + 1}: geo={t['geo_mean']:.4f}  "
            f"min={t['min_composite']:.4f}  "
            f"regressed={t['regressed_count']}  "
            f"threshold={t['threshold']}"
        )
    print()

    if pareto_best is not None:
        promoted = pareto_best
        promoted_label = "pareto-clean (no fixture regressed)"
    elif robust_best is not None:
        promoted = robust_best
        promoted_label = "robust-best (≤1 fixture regressed)"
    else:
        promoted = best_geo
        promoted_label = "highest-geo (no Pareto / robust candidate)"
    print(f"PROMOTED config ({promoted_label})")
    print("  weight diffs from default:")
    for k, v in promoted["weights"].items():
        d = DEFAULT_WEIGHTS[k]
        if abs(v - d) > 0.01:
            print(f"    {k:>22}: {d:+.3f} → {v:+.3f}  (Δ {v - d:+.3f})")
    print("  emission diffs from default:")
    for k, v in promoted["emissions"].items():
        d = DEFAULT_EMISSIONS[k]
        if abs(v - d) > 0.01:
            print(f"    {k:>22}: {d:+.3f} → {v:+.3f}  (Δ {v - d:+.3f})")
    print(f"    {'threshold':>22}: {DEFAULT_THRESHOLD:+.3f} → {promoted['threshold']:+.3f}")

    if not args.no_write:
        payload = {
            "seed": args.seed,
            "trials": args.trials,
            "default": {
                "weights": DEFAULT_WEIGHTS,
                "emissions": DEFAULT_EMISSIONS,
                "threshold": DEFAULT_THRESHOLD,
                "scores": asdict(default_scores),
                "per_fixture": [asdict(m) for m in default_metrics],
            },
            "highest_geo": best_geo,
            "robust_best": robust_best,
            "pareto_best": pareto_best,
            "pareto_count": len(pareto_candidates),
            "promoted": promoted,
            "promoted_label": promoted_label,
            "top_k": trials[: args.top_k],
        }
        OUTPUT.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {OUTPUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
