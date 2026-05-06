"""Tests for AlphaTimeExtractor — rule-based time-of-day extraction."""

from __future__ import annotations

import datetime

import pytest

from kaos_nlp_core.extract.alpha.time import AlphaTimeExtractor


@pytest.fixture
def extractor() -> AlphaTimeExtractor:
    return AlphaTimeExtractor()


class TestColonForm:
    def test_basic_24h(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("Meeting at 09:30."))
        assert datetime.time(9, 30) in out

    def test_with_seconds(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("23:59:59 is the boundary."))
        assert datetime.time(23, 59, 59) in out

    def test_with_pm_marker(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("3:30 PM is the standup."))
        assert datetime.time(15, 30) in out

    def test_with_am_marker(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("9:00 a.m. start."))
        assert datetime.time(9, 0) in out

    def test_invalid_minute_rejected(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("Bogus: 09:99."))
        assert out == []

    def test_invalid_hour_rejected(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("Bogus: 25:00."))
        assert out == []

    def test_multiple_times(self, extractor: AlphaTimeExtractor) -> None:
        text = "Both 09:00 and 17:30 are valid."
        out = list(extractor.extract_values(text))
        assert datetime.time(9, 0) in out
        assert datetime.time(17, 30) in out


class TestBareHourForm:
    def test_pm(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("Delivery by 5 PM."))
        assert datetime.time(17, 0) in out

    def test_am(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("Open at 9 AM."))
        assert datetime.time(9, 0) in out

    def test_12_pm_stays_at_12(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("12 PM is noon."))
        assert datetime.time(12, 0) in out

    def test_12_am_rolls_to_zero(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("12 AM is midnight."))
        assert datetime.time(0, 0) in out

    def test_13_with_am_rejected(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("13 AM is invalid."))
        assert datetime.time(13, 0) not in out

    def test_fractional_hour_rejected(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("5.5 PM is not valid."))
        assert out == []

    def test_dotted_marker(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("11:45 a.m. notice."))
        assert datetime.time(11, 45) in out


class TestWordOnly:
    def test_noon(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("Effective at noon."))
        assert datetime.time(12, 0) in out

    def test_midnight(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("By midnight on the 30th."))
        assert datetime.time(0, 0) in out

    def test_lunchtime_not_match(self, extractor: AlphaTimeExtractor) -> None:
        out = list(extractor.extract_values("Just lunchtime."))
        assert out == []


class TestSpans:
    def test_span_covers_marker(self, extractor: AlphaTimeExtractor) -> None:
        text = "Delivery by 5 PM EST."
        spans = list(extractor.extract_spans(text))
        assert len(spans) == 1
        s = spans[0]
        # The span should at least cover "5 PM" — the EST tz marker
        # is intentionally not captured.
        assert "5 PM" in text[s.start : s.end]

    def test_multiple_spans_in_order(self, extractor: AlphaTimeExtractor) -> None:
        text = "3:30 PM and 4:45 PM are both scheduled."
        spans = list(extractor.extract_spans(text))
        starts = [s.start for s in spans]
        assert starts == sorted(starts)


class TestEdgeCases:
    def test_empty(self, extractor: AlphaTimeExtractor) -> None:
        assert list(extractor.extract_values("")) == []

    def test_no_times(self, extractor: AlphaTimeExtractor) -> None:
        assert list(extractor.extract_values("Just regular prose.")) == []
