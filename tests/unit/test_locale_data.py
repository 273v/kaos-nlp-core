# ruff: noqa: RUF001, RUF003
"""Tests for kaos_nlp_core.locale_data gazetteers (WS-TR.PR-6f.0)."""

from __future__ import annotations

import pytest

from kaos_nlp_core.locale_data import (
    DURATION_MAP,
    MONEY_MAP,
    MONTH_MAP,
    ORDINAL_MAP,
    PERCENT_MAP,
    WEEKDAY_MAP,
    WRITTEN_NUMBER_MAP,
)


class TestMonthMap:
    def test_english_full_coverage(self) -> None:
        # 12 months × 2 forms (full + abbr, May has no separate abbr) = 23 entries.
        assert MONTH_MAP["en"]["january"] == 1
        assert MONTH_MAP["en"]["jan"] == 1
        assert MONTH_MAP["en"]["may"] == 5
        assert MONTH_MAP["en"]["december"] == 12
        assert MONTH_MAP["en"]["dec"] == 12

    def test_all_english_months_map_to_1_through_12(self) -> None:
        values = set(MONTH_MAP["en"].values())
        assert values == set(range(1, 13))

    def test_four_languages_supported(self) -> None:
        assert set(MONTH_MAP.keys()) == {"en", "es", "fr", "de"}

    def test_french_accented_forms(self) -> None:
        assert MONTH_MAP["fr"]["février"] == 2
        assert MONTH_MAP["fr"]["août"] == 8
        assert MONTH_MAP["fr"]["décembre"] == 12

    def test_german_umlaut_forms(self) -> None:
        assert MONTH_MAP["de"]["märz"] == 3
        assert MONTH_MAP["de"]["jän"] == 1

    def test_spanish_ambiguous_mar_is_march(self) -> None:
        # Deliberate ambiguity — "mar" is marzo (March) in MONTH_MAP,
        # martes (Tuesday) in WEEKDAY_MAP. Extractors must disambiguate.
        assert MONTH_MAP["es"]["mar"] == 3
        assert WEEKDAY_MAP["es"]["mar"] == 2


class TestOrdinalMap:
    def test_english_days_1_through_31(self) -> None:
        # Every day-of-month ordinal must parse.
        for day in range(1, 32):
            word_forms = {v: k for k, v in ORDINAL_MAP["en"].items() if v == day}
            assert word_forms, f"no ordinal form for day {day}"

    def test_numeric_suffix_forms(self) -> None:
        assert ORDINAL_MAP["en"]["1st"] == 1
        assert ORDINAL_MAP["en"]["2nd"] == 2
        assert ORDINAL_MAP["en"]["3rd"] == 3
        assert ORDINAL_MAP["en"]["4th"] == 4
        assert ORDINAL_MAP["en"]["21st"] == 21
        assert ORDINAL_MAP["en"]["31st"] == 31

    def test_word_forms(self) -> None:
        assert ORDINAL_MAP["en"]["first"] == 1
        assert ORDINAL_MAP["en"]["twenty-third"] == 23
        assert ORDINAL_MAP["en"]["thirty-first"] == 31


class TestDurationMap:
    def test_english_base_units(self) -> None:
        assert DURATION_MAP["en"]["second"] == 1
        assert DURATION_MAP["en"]["minute"] == 60
        assert DURATION_MAP["en"]["hour"] == 3600
        assert DURATION_MAP["en"]["day"] == 86400
        assert DURATION_MAP["en"]["week"] == 604800
        assert DURATION_MAP["en"]["year"] == 31536000  # 365 days

    def test_month_is_approximated_as_30_days(self) -> None:
        assert DURATION_MAP["en"]["month"] == 30 * 86400

    def test_anniversary_equals_year(self) -> None:
        assert DURATION_MAP["en"]["anniversary"] == DURATION_MAP["en"]["year"]

    def test_four_languages(self) -> None:
        assert set(DURATION_MAP.keys()) == {"en", "es", "fr", "de"}

    def test_plural_forms_match_singular(self) -> None:
        for lang in DURATION_MAP:
            pairs = [
                (s, p)
                for s, p in [
                    ("en:second", "en:seconds"),
                    ("en:minute", "en:minutes"),
                    ("en:hour", "en:hours"),
                    ("en:day", "en:days"),
                ]
                if s.startswith(f"{lang}:")
            ]
            for s_key, p_key in pairs:
                s_k = s_key.split(":", 1)[1]
                p_k = p_key.split(":", 1)[1]
                if s_k in DURATION_MAP[lang] and p_k in DURATION_MAP[lang]:
                    assert DURATION_MAP[lang][s_k] == DURATION_MAP[lang][p_k]


class TestMoneyMap:
    @pytest.mark.parametrize(
        ("token", "iso"),
        [
            ("$", "USD"),
            ("US$", "USD"),
            ("dollars", "USD"),
            ("Dollars", "USD"),
            ("USD", "USD"),
            ("£", "GBP"),
            ("pounds", "GBP"),
            ("€", "EUR"),
            ("euros", "EUR"),
            ("¥", "JPY"),
            ("yen", "JPY"),
            ("yuan", "CNY"),
            ("₹", "INR"),
            ("rupee", "INR"),
            ("₩", "KRW"),
            ("won", "KRW"),
            ("₣", "CHF"),
            ("franc", "CHF"),
        ],
    )
    def test_symbol_and_word_resolve_to_iso_code(self, token: str, iso: str) -> None:
        assert MONEY_MAP["en"][token] == iso

    def test_case_variants_collapse_to_same_iso(self) -> None:
        assert (
            MONEY_MAP["en"]["dollar"]
            == MONEY_MAP["en"]["Dollar"]
            == MONEY_MAP["en"]["DOLLAR"]
            == MONEY_MAP["en"]["dollars"]
            == MONEY_MAP["en"]["Dollars"]
            == MONEY_MAP["en"]["DOLLARS"]
            == "USD"
        )


class TestPercentMap:
    def test_basic_percent_is_one_hundredth(self) -> None:
        assert PERCENT_MAP["en"]["percent"] == 0.01
        assert PERCENT_MAP["en"]["%"] == 0.01
        assert PERCENT_MAP["en"]["pct."] == 0.01

    def test_per_mille_and_per_ten_thousand(self) -> None:
        assert PERCENT_MAP["en"]["‰"] == 0.001
        assert PERCENT_MAP["en"]["‱"] == 0.0001

    def test_parts_per_million_and_billion(self) -> None:
        assert PERCENT_MAP["en"]["ppm"] == 1e-6
        assert PERCENT_MAP["en"]["ppb"] == 1e-9

    def test_basis_points_is_one_ten_thousandth(self) -> None:
        # 50 bps = 0.005 — the scale is 0.01/100 = 1e-4.
        assert PERCENT_MAP["en"]["bps"] == 1e-4
        assert PERCENT_MAP["en"]["basis"] == 1e-4
        # Derived: 50 bps → 50 * 1e-4 = 0.005.
        assert 50 * PERCENT_MAP["en"]["bps"] == pytest.approx(0.005)

    def test_fullwidth_percent_variants(self) -> None:
        assert PERCENT_MAP["en"]["％"] == 0.01
        assert PERCENT_MAP["en"]["﹪"] == 0.01


class TestWrittenNumberMap:
    def test_english_digits_zero_through_nine(self) -> None:
        for k, v in [
            ("zero", 0),
            ("one", 1),
            ("two", 2),
            ("five", 5),
            ("nine", 9),
        ]:
            assert WRITTEN_NUMBER_MAP["en"][k] == v

    def test_powers_of_ten(self) -> None:
        assert WRITTEN_NUMBER_MAP["en"]["hundred"] == 100
        assert WRITTEN_NUMBER_MAP["en"]["thousand"] == 10**3
        assert WRITTEN_NUMBER_MAP["en"]["million"] == 10**6
        assert WRITTEN_NUMBER_MAP["en"]["billion"] == 10**9
        assert WRITTEN_NUMBER_MAP["en"]["trillion"] == 10**12

    def test_plural_forms_match(self) -> None:
        assert WRITTEN_NUMBER_MAP["en"]["hundred"] == WRITTEN_NUMBER_MAP["en"]["hundreds"]
        assert WRITTEN_NUMBER_MAP["en"]["thousand"] == WRITTEN_NUMBER_MAP["en"]["thousands"]

    def test_point_alias_for_decimal(self) -> None:
        assert WRITTEN_NUMBER_MAP["en"]["point"] == "."
        assert WRITTEN_NUMBER_MAP["en"]["spot"] == "."


class TestWeekdayMap:
    def test_english_1_through_7_monday_start(self) -> None:
        assert WEEKDAY_MAP["en"]["monday"] == 1
        assert WEEKDAY_MAP["en"]["sunday"] == 7

    def test_abbreviations_match(self) -> None:
        assert WEEKDAY_MAP["en"]["mon"] == WEEKDAY_MAP["en"]["monday"]
        assert WEEKDAY_MAP["en"]["fri"] == WEEKDAY_MAP["en"]["friday"]


class TestImmutabilityGuard:
    """The module-level dicts are the source of truth. Callers who want
    to extend the gazetteer must deep-copy — never mutate."""

    def test_modules_do_not_mutate_each_other(self) -> None:
        """Confirm MONTH_MAP and WEEKDAY_MAP reference distinct dicts
        (no aliasing that would propagate mutations)."""
        assert MONTH_MAP["en"] is not WEEKDAY_MAP["en"]
        assert MONEY_MAP["en"] is not MONTH_MAP["en"]

    def test_consumers_can_safely_deep_copy(self) -> None:
        import copy

        extended = copy.deepcopy(MONTH_MAP)
        extended["en"]["midsummer"] = 6
        # Original not affected.
        assert "midsummer" not in MONTH_MAP["en"]
