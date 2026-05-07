"""Document quality scoring: 18 metrics with weighted anomaly detection.

Single-pass character + word analysis is delegated to the Rust core
(`kaos_nlp_core._rust.quality.analyze_text`) which routes through the
shared ICU-backed character classifier and the project tokenizer, so
quality scoring stays consistent with the rest of the module's
segmentation rules. The expected-range tables and weights live here in
Python so they can be tuned without rebuilding the wheel.

The ``ratio_in_lexicon`` metric is the strongest OCR / extraction
quality signal: scrambled text ("rn"→"m", missing spaces, garbled
ligatures) tanks lexicon hit-rate even when entropy and char ratios
look fine. By default ``quality_report()`` loads a bundled ~2 MB
English FST (~382k headwords + inflections, derived from OpenGloss)
on first use and caches it module-level.

Usage::

    from kaos_nlp_core.quality import compute_metrics, score_quality

    metrics = compute_metrics("The quick brown fox ...")
    result = score_quality(metrics)              # uses "general" preset
    result = score_quality(metrics, "legal")

    # Convenience: combined metrics + anomaly score, default lexicon loaded.
    from kaos_nlp_core.quality import quality_report
    report = quality_report("The quick brown fox ...")
    print(report.score.score, report.metrics.ratio_in_lexicon)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from kaos_nlp_core._rust.quality import analyze_text as _rust_analyze
from kaos_nlp_core.lexicon import Lexicon
from kaos_nlp_core.matching import FstSet

# ── Default English wordset (lazy, cached) ─────────────────────────────────

_DEFAULT_WORDSET: FstSet | None = None
_DEFAULT_WORDSET_FILE = "english_wordset.fst"


def default_english_wordset() -> FstSet:
    """Lazy-load the bundled English wordset.

    Returns a process-cached :class:`FstSet` of ~382 k lower-cased
    headwords + inflections. The underlying ~2 MB FST ships inside the
    wheel under ``kaos_nlp_core/data/``.
    """
    global _DEFAULT_WORDSET
    if _DEFAULT_WORDSET is None:
        # importlib.resources gives a Path-like Traversable across
        # editable installs and built wheels.
        resource = files("kaos_nlp_core.data").joinpath(_DEFAULT_WORDSET_FILE)
        _DEFAULT_WORDSET = FstSet.load(str(resource))
    return _DEFAULT_WORDSET


# ── Metric weights ─────────────────────────────────────────────────────────

METRIC_WEIGHTS: dict[str, float] = {
    "ratio_whitespace": 1.0,
    "average_line_length": 1.0,
    "average_paragraph_length": 1.0,
    "ratio_alphanumeric": 1.0,
    "ratio_alpha_to_numeric": 0.1,
    "ratio_non_ascii": 2.0,
    "ratio_capital": 1.0,
    "ratio_punctuation": 1.0,
    "ratio_symbol": 1.5,
    "average_word_length": 1.5,
    "type_token_ratio": 1.5,
    "token_entropy": 0.5,
    "char_entropy": 0.5,
    "max_token_frequency_ratio": 1.0,
    "repetition_rate": 1.5,
    "ratio_format_tokens": 5.0,
    "ratio_in_lexicon": 3.0,
}

# ── Expected ranges by domain ──────────────────────────────────────────────
#
# The numeric bounds below are seeded from the kl3m calibration plus a
# first-pass calibration of the new metrics on USC, EDGAR, and Project
# Gutenberg fixtures. Re-run scripts/calibrate_quality_ranges.py to
# refresh from current fixtures; the script writes back these constants.

# Calibrated on 2/98 percentiles of USC text fixtures (n≈2000) via
# scripts/calibrate_quality_ranges.py. `ratio_in_lexicon` upper is
# capped at 1.0 so legitimately-clean documents aren't penalized for
# being above the 98th percentile — high values are good for that
# metric. `ratio_format_tokens` is hand-set tight; the calibration
# fixtures don't contain enough garbage tokens to derive a meaningful
# upper percentile.
LEGAL_RANGES: dict[str, tuple[float, float]] = {
    "ratio_whitespace": (0.137870, 0.180792),
    "average_line_length": (24.305538, 136.422968),
    "average_paragraph_length": (76.323030, 281.938743),
    "ratio_alphanumeric": (0.680670, 0.809528),
    "ratio_alpha_to_numeric": (3.028098, 45.335247),
    "ratio_non_ascii": (0.002260, 0.015846),
    "ratio_capital": (0.023680, 0.271061),
    "ratio_punctuation": (0.037710, 0.155426),
    "ratio_symbol": (0.000000, 0.001000),
    "average_word_length": (4.449907, 6.109622),
    "type_token_ratio": (0.177298, 0.853858),
    "token_entropy": (5.169418, 8.329634),
    "char_entropy": (4.500733, 5.300008),
    "max_token_frequency_ratio": (0.032838, 0.102571),
    "repetition_rate": (0.146142, 0.822702),
    "ratio_format_tokens": (0.000000, 0.005000),
    "ratio_in_lexicon": (0.813552, 1.000000),
}

# Wider ranges for generic prose (default). Calibrated on Project
# Gutenberg (Shakespeare + War and Peace) plus EDGAR agreements.
GENERAL_RANGES: dict[str, tuple[float, float]] = {
    "ratio_whitespace": (0.160216, 0.277068),
    "average_line_length": (19.496698, 228.832237),
    "average_paragraph_length": (68.707317, 2812.406977),
    "ratio_alphanumeric": (0.650823, 0.811845),
    "ratio_alpha_to_numeric": (31.309771, 308.670455),
    "ratio_non_ascii": (0.000114, 0.007580),
    "ratio_capital": (0.032333, 0.246233),
    "ratio_punctuation": (0.021195, 0.067181),
    "ratio_symbol": (0.000000, 0.001000),
    "average_word_length": (4.861909, 5.492872),
    "type_token_ratio": (0.052073, 0.500931),
    "token_entropy": (6.996280, 8.885831),
    "char_entropy": (4.371583, 5.139810),
    "max_token_frequency_ratio": (0.042945, 0.100028),
    "repetition_rate": (0.499069, 0.947927),
    "ratio_format_tokens": (0.000000, 0.010000),
    "ratio_in_lexicon": (0.895759, 1.000000),
}

DOMAIN_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "legal": LEGAL_RANGES,
    "general": GENERAL_RANGES,
}

DEFAULT_DOMAIN = "general"

# ── Typed result objects ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Per-document quality metrics.

    Counts use Unicode codepoints (not bytes). Ratios with no defined
    denominator (e.g. ``ratio_alpha_to_numeric`` when there are no
    digits) return ``math.inf``. ``ratio_in_lexicon`` is ``None`` when
    no lexicon was supplied.
    """

    total_characters: int
    ratio_whitespace: float
    average_line_length: float
    average_paragraph_length: float
    ratio_alphanumeric: float
    ratio_alpha_to_numeric: float
    ratio_non_ascii: float
    ratio_capital: float
    ratio_punctuation: float
    ratio_symbol: float
    average_word_length: float
    type_token_ratio: float
    token_entropy: float
    char_entropy: float
    max_token_frequency_ratio: float
    repetition_rate: float
    ratio_format_tokens: float
    ratio_in_lexicon: float | None
    num_words: int
    num_lines: int
    num_paragraphs: int

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a flat dict (numeric values, no None)."""
        out: dict[str, Any] = {
            "total_characters": self.total_characters,
            "ratio_whitespace": self.ratio_whitespace,
            "average_line_length": self.average_line_length,
            "average_paragraph_length": self.average_paragraph_length,
            "ratio_alphanumeric": self.ratio_alphanumeric,
            "ratio_alpha_to_numeric": self.ratio_alpha_to_numeric,
            "ratio_non_ascii": self.ratio_non_ascii,
            "ratio_capital": self.ratio_capital,
            "ratio_punctuation": self.ratio_punctuation,
            "ratio_symbol": self.ratio_symbol,
            "average_word_length": self.average_word_length,
            "type_token_ratio": self.type_token_ratio,
            "token_entropy": self.token_entropy,
            "char_entropy": self.char_entropy,
            "max_token_frequency_ratio": self.max_token_frequency_ratio,
            "repetition_rate": self.repetition_rate,
            "ratio_format_tokens": self.ratio_format_tokens,
            "num_words": self.num_words,
            "num_lines": self.num_lines,
            "num_paragraphs": self.num_paragraphs,
        }
        if self.ratio_in_lexicon is not None:
            out["ratio_in_lexicon"] = self.ratio_in_lexicon
        return out


@dataclass(frozen=True, slots=True)
class ComponentDeviation:
    """A single metric's contribution to the anomaly score."""

    value: float
    expected_range: tuple[float, float]
    weight: float
    deviation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 6),
            "expected_range": [round(self.expected_range[0], 6), round(self.expected_range[1], 6)],
            "weight": self.weight,
            "deviation": round(self.deviation, 6),
        }


@dataclass(frozen=True, slots=True)
class QualityScore:
    """Anomaly score: weighted sum of out-of-range deviations."""

    score: float
    domain: str
    components: dict[str, ComponentDeviation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "domain": self.domain,
            "components": {k: v.to_dict() for k, v in self.components.items()},
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Combined metrics + anomaly score for a single document."""

    metrics: QualityMetrics
    score: QualityScore

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics.to_dict(), "score": self.score.to_dict()}


# ── Public API ─────────────────────────────────────────────────────────────


def compute_metrics(
    text: str,
    *,
    lexicon: FstSet | Lexicon | None = None,
) -> QualityMetrics:
    """Compute the 18-metric quality suite from raw text.

    Args:
        text: Input text to analyze.
        lexicon: Optional ``FstSet`` or ``Lexicon`` used to compute
            ``ratio_in_lexicon``. When ``None``, ``ratio_in_lexicon`` is
            ``None`` and contributes no penalty during scoring. Pass
            :func:`default_english_wordset` to use the bundled wordset.

    Returns:
        A :class:`QualityMetrics` value object.
    """
    raw = _rust_analyze(text, _unwrap(lexicon))
    return _build_metrics(raw, lexicon_present=lexicon is not None)


def score_quality(
    metrics: QualityMetrics,
    domain: str = DEFAULT_DOMAIN,
) -> QualityScore:
    """Score a metrics dict as weighted anomaly from expected ranges.

    Lower scores are better — zero means every metric falls inside the
    domain's expected range. ``domain`` selects between ``"general"``
    (default, wider ranges) and ``"legal"`` (USC/CFR-calibrated).
    """
    ranges = DOMAIN_RANGES.get(domain)
    if ranges is None:
        valid = ", ".join(sorted(DOMAIN_RANGES))
        raise ValueError(f"Unknown domain {domain!r}. Choose from: {valid}")

    eps = 1e-8
    total_score = 0.0
    components: dict[str, ComponentDeviation] = {}
    md = metrics.to_dict()

    for metric, weight in METRIC_WEIGHTS.items():
        value = md.get(metric)
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            continue
        if math.isinf(value) or math.isnan(value):
            continue

        lower, upper = ranges.get(metric, (0.0, 0.0))
        component = 0.0

        if lower == upper:
            if abs(value - lower) > eps:
                component = weight * abs(value - lower)
        elif value < lower:
            component = weight * (lower - value) / (abs(lower) + eps)
        elif value > upper:
            component = weight * (value - upper) / (abs(upper) + eps)

        if math.isnan(component) or math.isinf(component):
            continue

        if component > 0:
            components[metric] = ComponentDeviation(
                value=float(value),
                expected_range=(lower, upper),
                weight=weight,
                deviation=component,
            )

        total_score += component

    return QualityScore(
        score=round(total_score, 6),
        domain=domain,
        components=components,
    )


def quality_report(
    text: str,
    *,
    domain: str = DEFAULT_DOMAIN,
    lexicon: FstSet | Lexicon | None = None,
    use_default_lexicon: bool = True,
) -> QualityReport:
    """Convenience: compute + score in a single call.

    When ``lexicon`` is ``None`` and ``use_default_lexicon`` is true
    (the default), the bundled English wordset is loaded lazily and
    cached for the rest of the process.
    """
    effective_lex = lexicon
    if effective_lex is None and use_default_lexicon:
        effective_lex = default_english_wordset()

    metrics = compute_metrics(text, lexicon=effective_lex)
    score = score_quality(metrics, domain=domain)
    return QualityReport(metrics=metrics, score=score)


# ── Internals ──────────────────────────────────────────────────────────────


def _unwrap(lex: FstSet | Lexicon | None) -> Any:
    """Pass the underlying PyO3 pyclass to the Rust binding."""
    if lex is None:
        return None
    return lex._inner


def _build_metrics(raw: Any, *, lexicon_present: bool) -> QualityMetrics:
    """Convert the raw analyzer result into the public ``QualityMetrics``.

    ``raw`` is a ``kaos_nlp_core._rust.quality.QualityRaw`` pyclass with
    nested ``chars`` and ``words`` typed views — attribute access only,
    no per-key dict lookup (audit perf finding #6 / P8).
    """
    chars = raw.chars
    words = raw.words
    total = chars.total_chars

    if total == 0:
        return _zero_metrics(lexicon_present)

    num_words = words.num_words
    line_count = max(1, chars.line_count)
    paragraph_count = max(1, chars.paragraph_count)

    if num_words == 0:
        avg_word_length = 0.0
        type_token_ratio = 0.0
        max_token_freq_ratio = 0.0
        repetition_rate = 0.0
        ratio_format_tokens = 0.0
    else:
        avg_word_length = words.total_word_chars / num_words
        type_token_ratio = words.unique_words / num_words
        max_token_freq_ratio = words.max_freq / num_words
        repetition_rate = 1.0 - type_token_ratio
        ratio_format_tokens = words.format_tokens / num_words

    digit = chars.digit
    alpha = chars.alpha
    # No digits → infinite ratio; the score_quality math.isinf guard skips it.
    ratio_alpha_to_numeric = math.inf if digit == 0 else alpha / digit

    ratio_in_lex: float | None
    if lexicon_present:
        denom = words.alphabetic_tokens
        ratio_in_lex = words.in_lexicon / denom if denom > 0 else 1.0
    else:
        ratio_in_lex = None

    return QualityMetrics(
        total_characters=int(total),
        ratio_whitespace=chars.whitespace / total,
        average_line_length=total / line_count,
        average_paragraph_length=total / paragraph_count,
        ratio_alphanumeric=chars.alphanumeric / total,
        ratio_alpha_to_numeric=ratio_alpha_to_numeric,
        ratio_non_ascii=chars.non_ascii / total,
        ratio_capital=(chars.upper / alpha) if alpha > 0 else 0.0,
        ratio_punctuation=chars.punct / total,
        ratio_symbol=chars.symbol / total,
        average_word_length=avg_word_length,
        type_token_ratio=type_token_ratio,
        token_entropy=words.token_entropy,
        char_entropy=chars.char_entropy,
        max_token_frequency_ratio=max_token_freq_ratio,
        repetition_rate=repetition_rate,
        ratio_format_tokens=ratio_format_tokens,
        ratio_in_lexicon=ratio_in_lex,
        num_words=int(num_words),
        num_lines=int(line_count),
        num_paragraphs=int(paragraph_count),
    )


def _zero_metrics(lexicon_present: bool) -> QualityMetrics:
    return QualityMetrics(
        total_characters=0,
        ratio_whitespace=0.0,
        average_line_length=0.0,
        average_paragraph_length=0.0,
        ratio_alphanumeric=0.0,
        ratio_alpha_to_numeric=0.0,
        ratio_non_ascii=0.0,
        ratio_capital=0.0,
        ratio_punctuation=0.0,
        ratio_symbol=0.0,
        average_word_length=0.0,
        type_token_ratio=0.0,
        token_entropy=0.0,
        char_entropy=0.0,
        max_token_frequency_ratio=0.0,
        repetition_rate=0.0,
        ratio_format_tokens=0.0,
        ratio_in_lexicon=0.0 if lexicon_present else None,
        num_words=0,
        num_lines=0,
        num_paragraphs=0,
    )


__all__ = [
    "DEFAULT_DOMAIN",
    "DOMAIN_RANGES",
    "GENERAL_RANGES",
    "LEGAL_RANGES",
    "METRIC_WEIGHTS",
    "ComponentDeviation",
    "QualityMetrics",
    "QualityReport",
    "QualityScore",
    "compute_metrics",
    "default_english_wordset",
    "quality_report",
    "score_quality",
]
