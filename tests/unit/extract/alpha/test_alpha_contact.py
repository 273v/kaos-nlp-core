"""Tests for AlphaContactExtractor — email/URL/phone."""

from __future__ import annotations

import pytest

from kaos_nlp_core.extract.alpha.contact import (
    AlphaContactExtractor,
    ContactMatch,
)


@pytest.fixture
def extractor() -> AlphaContactExtractor:
    return AlphaContactExtractor()


class TestEmail:
    def test_basic(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("Email: foo@bar.com."))
        assert any(m.kind == "email" and m.value == "foo@bar.com" for m in out)

    def test_plus_subaddress(self, extractor: AlphaContactExtractor) -> None:
        text = "Reply to john.doe+tag@subdomain.example.co.uk."
        out = list(extractor.extract_values(text))
        emails = [m for m in out if m.kind == "email"]
        assert emails
        assert emails[0].value == "john.doe+tag@subdomain.example.co.uk"

    def test_normalized_lowercase(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("Send to FOO@BAR.COM"))
        emails = [m for m in out if m.kind == "email"]
        assert emails[0].normalized == "foo@bar.com"


class TestURL:
    def test_https_with_path(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("See https://example.com/api?x=1."))
        urls = [m for m in out if m.kind == "url"]
        assert urls and urls[0].value == "https://example.com/api?x=1"

    def test_bare_domain_alone_rejected(self, extractor: AlphaContactExtractor) -> None:
        # No path → don't match (avoids false positives on company names).
        out = list(extractor.extract_values("Acme.com is our brand."))
        urls = [m for m in out if m.kind == "url"]
        assert not urls

    def test_bare_domain_with_path_accepted(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("Visit Acme.com/about today."))
        urls = [m for m in out if m.kind == "url"]
        assert urls and urls[0].value == "Acme.com/about"

    def test_trailing_period_stripped(self, extractor: AlphaContactExtractor) -> None:
        text = "See https://example.com/foo."
        out = list(extractor.extract_spans(text))
        urls = [s for s in out if s.value.kind == "url"]
        assert urls
        assert text[urls[0].start : urls[0].end] == "https://example.com/foo"


class TestPhoneNANP:
    def test_paren_format(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("Call (415) 555-0199."))
        phones = [m for m in out if m.kind == "phone"]
        assert phones
        assert phones[0].normalized == "+1 4155550199"

    def test_dash_format(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("212-555-0100"))
        phones = [m for m in out if m.kind == "phone"]
        assert phones[0].normalized == "+1 2125550100"

    def test_dot_format(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("415.555.0188"))
        phones = [m for m in out if m.kind == "phone"]
        assert phones[0].normalized == "+1 4155550188"

    def test_invalid_area_code_zero_one(self, extractor: AlphaContactExtractor) -> None:
        # NANP area codes start with 2-9.
        out = list(extractor.extract_values("Call 100-555-0100."))
        phones = [m for m in out if m.kind == "phone"]
        assert not phones


class TestPhoneInternational:
    def test_uk_number(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("UK: +44 20 7946 0958."))
        phones = [m for m in out if m.kind == "phone"]
        assert phones
        assert phones[0].normalized.startswith("+44")

    def test_german_8digit_subscriber(self, extractor: AlphaContactExtractor) -> None:
        # The bug fix from the smoke test — 8-digit blocks shouldn't truncate.
        out = list(extractor.extract_values("DE: +49 30 12345678."))
        phones = [m for m in out if m.kind == "phone"]
        assert phones
        digits_only = phones[0].normalized.replace("+", "").replace(" ", "")
        assert digits_only == "493012345678"

    def test_us_with_country_code(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("Call +1 212-555-0100."))
        phones = [m for m in out if m.kind == "phone"]
        assert phones[0].normalized.startswith("+1")


class TestOverlapPriority:
    def test_email_beats_url(self, extractor: AlphaContactExtractor) -> None:
        # The domain in the email shouldn't also fire as a URL.
        text = "Email me at user@example.com today."
        out = list(extractor.extract_values(text))
        kinds = {m.kind for m in out}
        # Email should win; bare-domain URL never fires here anyway.
        assert "email" in kinds

    def test_phone_does_not_capture_email_digits(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("user1234@example.com"))
        kinds = {m.kind for m in out}
        assert "phone" not in kinds


class TestMultiple:
    def test_emit_in_document_order(self, extractor: AlphaContactExtractor) -> None:
        text = "foo@bar.com, https://baz.com/x, +1 555 123 4567"
        spans = list(extractor.extract_spans(text))
        starts = [s.start for s in spans]
        assert starts == sorted(starts)


class TestRecord:
    def test_returns_contact_match(self, extractor: AlphaContactExtractor) -> None:
        out = list(extractor.extract_values("foo@bar.com"))
        assert isinstance(out[0], ContactMatch)
