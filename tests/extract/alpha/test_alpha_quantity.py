"""Tests for AlphaQuantityExtractor — number-with-units."""

from __future__ import annotations

from decimal import Decimal

import pytest

from kaos_nlp_core.extract.alpha.quantity import (
    AlphaQuantityExtractor,
    QuantityMatch,
)


@pytest.fixture
def extractor() -> AlphaQuantityExtractor:
    return AlphaQuantityExtractor()


class TestSingleTokenUnits:
    def test_mass_kg(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Cargo: 5 kg."))
        assert QuantityMatch(amount=Decimal(5), unit="kg", dimension="mass") in out

    def test_length_meters(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Span: 2.5 meters."))
        assert QuantityMatch(amount=Decimal("2.5"), unit="m", dimension="length") in out

    def test_speed_mph(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Top speed 100 mph."))
        assert QuantityMatch(amount=Decimal(100), unit="mph", dimension="speed") in out

    def test_data_gb(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("32 GB RAM."))
        assert QuantityMatch(amount=Decimal(32), unit="GB", dimension="data") in out

    def test_pressure_psi(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Tire pressure 35 psi."))
        assert QuantityMatch(amount=Decimal(35), unit="psi", dimension="pressure") in out

    def test_acres(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Lot size 10 acres."))
        assert QuantityMatch(amount=Decimal(10), unit="acre", dimension="area") in out


class TestFusedTemperature:
    def test_celsius_fused(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Operating at 20°C ambient."))
        assert QuantityMatch(amount=Decimal(20), unit="°C", dimension="temperature") in out

    def test_fahrenheit_fused(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Body temp 98.6°F."))
        assert QuantityMatch(amount=Decimal("98.6"), unit="°F", dimension="temperature") in out

    def test_no_degree_sign_rejected(self, extractor: AlphaQuantityExtractor) -> None:
        # Bare-letter temperature forms like "32F" are deliberately
        # rejected because they collide with SEC rule references
        # (e.g., "Section 21F"). Use "32°F" or "32 Fahrenheit".
        out = list(extractor.extract_values("Set to 32F minimum."))
        assert all(m.unit not in ("°C", "°F") for m in out), (
            f"unexpected fused temperature in: {out}"
        )

    def test_negative_temperature(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Cold storage at -40°C."))
        assert QuantityMatch(amount=Decimal(-40), unit="°C", dimension="temperature") in out


class TestMultiTokenUnits:
    def test_square_feet(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Premises: 2,500 square feet."))
        assert QuantityMatch(amount=Decimal(2500), unit="ft^2", dimension="area") in out

    def test_cubic_yards(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Pour 500 cubic yards of concrete."))
        assert QuantityMatch(amount=Decimal(500), unit="yd^3", dimension="volume") in out

    def test_linear_feet(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Install 1,200 linear feet of rebar."))
        assert QuantityMatch(amount=Decimal(1200), unit="lin-ft", dimension="length") in out

    def test_board_feet(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Lumber: 800 board feet of pine."))
        assert QuantityMatch(amount=Decimal(800), unit="bf", dimension="volume") in out

    def test_metric_tons(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Cargo: 50 metric tons."))
        assert QuantityMatch(amount=Decimal(50), unit="t", dimension="mass") in out

    def test_parking_spaces(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Property includes 24 parking spaces."))
        assert QuantityMatch(amount=Decimal(24), unit="parking-space", dimension="count") in out

    def test_multi_does_not_double_emit(self, extractor: AlphaQuantityExtractor) -> None:
        # "5 square feet" should NOT also emit "feet" as a separate
        # length unit on top of the area match.
        out = list(extractor.extract_values("Premises: 100 square feet."))
        units = [m.unit for m in out]
        assert "ft^2" in units
        assert "ft" not in units


class TestContractDomains:
    def test_real_estate(self, extractor: AlphaQuantityExtractor) -> None:
        text = "Property: 2,500 square feet on a 0.5 acre lot, 3 bedrooms."
        units = [m.unit for m in extractor.extract_values(text)]
        assert "ft^2" in units
        assert "acre" in units
        assert "bedroom" in units

    def test_construction(self, extractor: AlphaQuantityExtractor) -> None:
        text = "Pour 500 cubic yards; install 1,200 linear feet of rebar at 60 ksi."
        units = [m.unit for m in extractor.extract_values(text)]
        assert "yd^3" in units
        assert "lin-ft" in units
        assert "ksi" in units

    def test_supply_chain(self, extractor: AlphaQuantityExtractor) -> None:
        text = "Shipment: 40 TEU, 200 pallets, 1500 cases of widgets."
        units = [m.unit for m in extractor.extract_values(text)]
        assert "TEU" in units
        assert "pallet" in units
        assert "case" in units

    def test_financial(self, extractor: AlphaQuantityExtractor) -> None:
        text = "Buyer purchases 10,000 shares and 500 warrants under this Agreement."
        units = [m.unit for m in extractor.extract_values(text)]
        assert "share" in units
        assert "warrant" in units

    def test_oil_and_gas(self, extractor: AlphaQuantityExtractor) -> None:
        # Note: write `MMcf` separated from any composite unit suffix —
        # the tokenizer keeps `MMcf/day` as one token, so a direct
        # `MMcf/day` won't hit the gazetteer (composite units are out
        # of scope per the module docstring).
        text = "Plant capacity 50 MMcf per day; reserves 5 BCF; fuel 200000 MMBtu."
        units = [m.unit for m in extractor.extract_values(text)]
        assert "MMcf" in units
        assert "BCF" in units
        assert "MMBtu" in units


class TestSpans:
    def test_span_covers_quantity_and_unit(self, extractor: AlphaQuantityExtractor) -> None:
        text = "Cargo weighs 5 kg total."
        spans = list(extractor.extract_spans(text))
        assert spans
        s = spans[0]
        # The span starts at "5" and ends inclusive of "kg".
        assert "5" in text[s.start : s.end]
        assert "kg" in text[s.start : s.end]

    def test_multi_token_span_covers_both_unit_words(
        self, extractor: AlphaQuantityExtractor
    ) -> None:
        text = "Premises: 1000 square meters."
        spans = list(extractor.extract_spans(text))
        s = next(sp for sp in spans if sp.value.unit == "m^2")
        captured = text[s.start : s.end]
        assert "square" in captured
        assert "meters" in captured


class TestEdgeCases:
    def test_empty(self, extractor: AlphaQuantityExtractor) -> None:
        assert list(extractor.extract_values("")) == []

    def test_no_quantities(self, extractor: AlphaQuantityExtractor) -> None:
        out = list(extractor.extract_values("Just regular prose."))
        assert out == []

    def test_unsupported_language_raises(self) -> None:
        # The base extractor's language check fires before the
        # UNIT_MAP gazetteer check (the class only lists "en").
        with pytest.raises(ValueError, match="language"):
            AlphaQuantityExtractor(language="ar")
