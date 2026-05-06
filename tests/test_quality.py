"""Tests for kaos_nlp_core.quality — text quality metrics and scoring.

Tests compute_metrics() and score_quality() directly, independent of
the MCP tool wrapper. The 0.1.0a2 rewrite moves char + token analysis
to Rust and returns typed dataclasses instead of raw dicts.
"""

from __future__ import annotations

import math

import pytest

from kaos_nlp_core.matching import FstSet
from kaos_nlp_core.quality import (
    DOMAIN_RANGES,
    METRIC_WEIGHTS,
    ComponentDeviation,
    QualityMetrics,
    QualityReport,
    QualityScore,
    compute_metrics,
    default_english_wordset,
    quality_report,
    score_quality,
)


@pytest.fixture(scope="module")
def small_lexicon() -> FstSet:
    return FstSet(
        [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "hello",
            "world",
            "agreement",
            "contract",
            "party",
            "shall",
        ]
    )


class TestComputeMetrics:
    def test_empty_string(self) -> None:
        m = compute_metrics("")
        assert m.total_characters == 0
        assert m.num_words == 0
        assert m.ratio_whitespace == 0.0

    def test_basic_text(self) -> None:
        m = compute_metrics("Hello world")
        assert m.total_characters == 11
        assert m.num_words == 2
        assert m.ratio_whitespace == pytest.approx(1 / 11, abs=1e-6)
        assert m.ratio_alphanumeric == pytest.approx(10 / 11, abs=1e-6)

    def test_multiline(self) -> None:
        text = "Line one.\nLine two.\nLine three."
        m = compute_metrics(text)
        assert m.num_lines == 3
        assert m.average_line_length == pytest.approx(len(text) / 3, abs=0.1)

    def test_paragraph_detection(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        m = compute_metrics(text)
        assert m.num_paragraphs == 3

    def test_all_caps(self) -> None:
        m = compute_metrics("HELLO WORLD")
        assert m.ratio_capital == pytest.approx(1.0, abs=1e-6)

    def test_all_lowercase(self) -> None:
        m = compute_metrics("hello world")
        assert m.ratio_capital == pytest.approx(0.0, abs=1e-6)

    def test_digits(self) -> None:
        m = compute_metrics("abc 123")
        assert m.ratio_alpha_to_numeric == pytest.approx(3 / 3, abs=1e-6)

    def test_no_digits_gives_inf(self) -> None:
        m = compute_metrics("hello world")
        assert math.isinf(m.ratio_alpha_to_numeric)

    def test_non_ascii(self) -> None:
        m = compute_metrics("café résumé")
        assert m.ratio_non_ascii > 0

    def test_punctuation(self) -> None:
        m = compute_metrics("Hello, world!")
        assert m.ratio_punctuation > 0

    def test_symbol_separated_from_punctuation(self) -> None:
        # ICU-correct: € is a symbol, not punctuation.
        m = compute_metrics("€100 plus tax.")
        assert m.ratio_symbol > 0
        assert m.ratio_punctuation > 0

    def test_entropy_positive(self) -> None:
        m = compute_metrics("The quick brown fox jumps over the lazy dog")
        assert m.char_entropy > 0
        assert m.token_entropy > 0

    def test_repetition_rate(self) -> None:
        m_varied = compute_metrics("one two three four five six seven eight")
        m_repeat = compute_metrics("the the the the the the the the")
        assert m_repeat.repetition_rate > m_varied.repetition_rate

    def test_type_token_ratio(self) -> None:
        m_varied = compute_metrics("one two three four five")
        m_repeat = compute_metrics("one one one one one")
        assert m_varied.type_token_ratio > m_repeat.type_token_ratio

    def test_format_tokens_zero_for_clean(self) -> None:
        m = compute_metrics("Any text at all and twelve 12 tokens.")
        assert m.ratio_format_tokens == 0.0

    def test_format_tokens_detects_garbage(self) -> None:
        # Tokens with internal punctuation that aren't valid abbreviations
        # are flagged. `tokenize_words` strips outer punct/symbols, so
        # symbol-only tokens get filtered before they reach the analyzer.
        m = compute_metrics("hello ab.cd world qz.rt valid words here")
        assert m.ratio_format_tokens > 0.0

    def test_lexicon_none_returns_none_metric(self) -> None:
        m = compute_metrics("hello world")
        assert m.ratio_in_lexicon is None

    def test_lexicon_hit_ratio(self, small_lexicon: FstSet) -> None:
        m = compute_metrics(
            "The quick brown fox xyzzyy jumps over.",
            lexicon=small_lexicon,
        )
        # 6/7 alphabetic tokens hit (xyzzyy is the miss).
        assert m.ratio_in_lexicon is not None
        assert m.ratio_in_lexicon > 0.8

    def test_lexicon_uppercase_lowered(self, small_lexicon: FstSet) -> None:
        m = compute_metrics("THE QUICK BROWN FOX.", lexicon=small_lexicon)
        assert m.ratio_in_lexicon == pytest.approx(1.0, abs=1e-6)

    def test_metric_weights_coverage(self) -> None:
        """Every weighted metric should be present in compute_metrics output."""
        m = compute_metrics(
            "Some reasonable text with several words.",
            lexicon=FstSet(["some", "text"]),
        )
        d = m.to_dict()
        for key in METRIC_WEIGHTS:
            assert key in d, f"Metric {key!r} missing from compute_metrics"


class TestScoreQuality:
    def test_legal_domain(self) -> None:
        m = compute_metrics("The quick brown fox jumps over the lazy dog.")
        result = score_quality(m, domain="legal")
        assert isinstance(result, QualityScore)
        assert result.domain == "legal"
        assert result.score >= 0.0

    def test_general_domain(self) -> None:
        m = compute_metrics("Hello world.")
        result = score_quality(m, domain="general")
        assert result.domain == "general"

    def test_default_domain_is_general(self) -> None:
        m = compute_metrics("Hello world.")
        result = score_quality(m)
        assert result.domain == "general"

    def test_invalid_domain_raises(self) -> None:
        m = compute_metrics("test")
        with pytest.raises(ValueError, match="Unknown domain"):
            score_quality(m, domain="invalid")

    def test_garbage_scores_high(self) -> None:
        garbage = "XXXX " * 500
        m = compute_metrics(garbage)
        result = score_quality(m, domain="legal")
        assert result.score > 5.0

    def test_components_track_deviations(self) -> None:
        m = compute_metrics("A")  # Extremely short, deviates on many metrics.
        result = score_quality(m, domain="legal")
        assert len(result.components) > 0
        for component in result.components.values():
            assert isinstance(component, ComponentDeviation)
            assert component.weight > 0

    def test_all_domains_have_all_metrics(self) -> None:
        for domain_name, ranges in DOMAIN_RANGES.items():
            for metric in METRIC_WEIGHTS:
                assert metric in ranges, f"Domain {domain_name!r} missing range for {metric!r}"

    def test_empty_metrics_score_finite(self) -> None:
        m = compute_metrics("")
        result = score_quality(m, domain="legal")
        assert math.isfinite(result.score)


class TestLexiconBasedScoring:
    def test_clean_text_high_lex_ratio(self, small_lexicon: FstSet) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 4
        m = compute_metrics(text, lexicon=small_lexicon)
        assert m.ratio_in_lexicon is not None
        assert m.ratio_in_lexicon == pytest.approx(1.0, abs=1e-6)

    def test_garbled_text_low_lex_ratio(self, small_lexicon: FstSet) -> None:
        text = "Tlie qiiick browii rox jLnnps oxer tlie iazy dog. " * 4
        m = compute_metrics(text, lexicon=small_lexicon)
        assert m.ratio_in_lexicon is not None
        assert m.ratio_in_lexicon < 0.3

    def test_clean_vs_garbled_score_gap(self, small_lexicon: FstSet) -> None:
        clean = "The quick brown fox jumps over the lazy dog. " * 4
        garbled = "Tlie qiiick browii rox jLnnps oxer tlie iazy dog. " * 4
        m_clean = compute_metrics(clean, lexicon=small_lexicon)
        m_garbled = compute_metrics(garbled, lexicon=small_lexicon)
        s_clean = score_quality(m_clean, domain="general")
        s_garbled = score_quality(m_garbled, domain="general")
        assert s_garbled.score > s_clean.score


class TestQualityReport:
    def test_returns_combined_object(self) -> None:
        report = quality_report("Hello world.", use_default_lexicon=False)
        assert isinstance(report, QualityReport)
        assert isinstance(report.metrics, QualityMetrics)
        assert isinstance(report.score, QualityScore)
        assert report.metrics.ratio_in_lexicon is None

    def test_default_lexicon_loads_lazily(self) -> None:
        report = quality_report("The quick brown fox.")
        assert report.metrics.ratio_in_lexicon is not None
        assert report.metrics.ratio_in_lexicon > 0.5

    def test_custom_lexicon_overrides_default(self, small_lexicon: FstSet) -> None:
        report = quality_report(
            "The quick brown fox.",
            lexicon=small_lexicon,
            use_default_lexicon=True,
        )
        # Even with use_default_lexicon=True, an explicit lexicon wins.
        assert report.metrics.ratio_in_lexicon == pytest.approx(1.0, abs=1e-6)

    def test_to_dict_round_trip(self) -> None:
        report = quality_report("Hello world.", use_default_lexicon=False)
        d = report.to_dict()
        assert "metrics" in d
        assert "score" in d
        assert "score" in d["score"]
        assert "domain" in d["score"]


class TestDefaultWordset:
    def test_default_wordset_loads(self) -> None:
        ws = default_english_wordset()
        assert len(ws) > 100_000
        assert ws.contains("the")
        assert ws.contains("contract")
        assert not ws.contains("xyzzyzy")

    def test_default_wordset_is_cached(self) -> None:
        ws1 = default_english_wordset()
        ws2 = default_english_wordset()
        assert ws1 is ws2


class TestUnicodeAndEdgeCases:
    def test_cjk_round_trip(self) -> None:
        # Every char in this string (including the fullwidth period) is
        # non-ASCII; CJK characters classify as alpha but not upper/lower.
        m = compute_metrics("東京は日本の首都です。")
        assert m.total_characters > 0
        assert m.ratio_alphanumeric > 0
        assert m.ratio_non_ascii == pytest.approx(1.0, abs=1e-6)
        assert m.ratio_capital == 0.0

    def test_emoji_doesnt_crash(self) -> None:
        m = compute_metrics("Great work 😀 keep it up 🎉")
        assert m.total_characters > 0

    def test_only_whitespace(self) -> None:
        m = compute_metrics("   \t\n   ")
        assert m.num_words == 0
        assert m.ratio_whitespace > 0
