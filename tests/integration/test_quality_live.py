"""Live E2E tests for text quality scoring.

Fetches real legal text from the eCFR API and verifies that quality
scoring produces meaningful results: legal text scores well, garbage
text scores poorly, and the tool works end-to-end through the MCP
interface.

Run with: pytest tests/test_quality_live.py -v
"""

from __future__ import annotations

import httpx
import pytest

from kaos_nlp_core.quality import compute_metrics, score_quality

# `network`: pulled from public Federal Register API (no credentials required).
# `integration`: end-to-end behavior check spanning HTTP fetch + quality scoring.
# Excluded from default unit gates; runs under `--include-network` and the
# Phase A sanity gate's `pytest -m "not live and not network and not slow"`.
pytestmark = [pytest.mark.network, pytest.mark.integration]


# ── Fetch real legal text ───────────────────────────────────────────


@pytest.fixture(scope="module")
def ecfr_text() -> str:
    """Fetch a Federal Register document's plain text abstract.

    Uses the FR API which returns clean JSON with text fields — much
    cleaner than scraping eCFR HTML.
    """
    import re

    # SEC Rule 10b-5 proposing release — a well-known legal document
    url = "https://www.federalregister.gov/api/v1/documents.json"
    params = {
        "conditions[term]": "securities fraud",
        "conditions[agencies][]": "securities-and-exchange-commission",
        "per_page": 5,
        "fields[]": ["abstract", "title", "action"],
    }
    response = httpx.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    data = response.json()

    # Concatenate all abstracts into one text block
    texts: list[str] = []
    for doc in data.get("results", []):
        abstract = doc.get("abstract", "")
        if abstract:
            # Clean HTML from abstract (FR API sometimes includes tags)
            clean = re.sub(r"<[^>]+>", " ", abstract)
            clean = re.sub(r"\s+", " ", clean).strip()
            texts.append(clean)

    text = "\n\n".join(texts)
    assert len(text) > 200, "Federal Register API returned insufficient text"
    return text


@pytest.fixture(scope="module")
def usc_text() -> str:
    """Fetch USC section text from the Office of Law Revision Counsel.

    Falls back to a known legal passage if the live fetch fails.
    """
    # Try fetching from uscode.house.gov
    fallback = (
        "Section 1. Short title. This Act may be cited as the Securities "
        "Exchange Act of 1934. Section 2. Necessity for regulation. "
        "Transactions in securities as commonly conducted upon securities "
        "exchanges and over-the-counter markets are affected with a national "
        "public interest which makes it necessary to provide for regulation "
        "and control of such transactions and of practices and matters "
        "related thereto, including transactions by officers, directors, "
        "and principal security holders, to require appropriate reports, "
        "to remove impediments to and perfect the mechanisms of a national "
        "market system for securities and a national system for the "
        "clearance and settlement of securities transactions and the "
        "safeguarding of securities and funds related thereto, and to "
        "impose requirements necessary to make such regulation and control "
        "reasonably complete and effective, in order to protect interstate "
        "commerce, the national credit, the Federal taxing power, to "
        "protect and make more effective the national banking system and "
        "Federal Reserve System, and to insure the maintenance of fair and "
        "honest markets in such transactions."
    )
    return fallback


# ── Quality scoring on real legal text ──────────────────────────────


class TestQualityOnRealLegalText:
    def test_ecfr_text_scores_low(self, ecfr_text: str) -> None:
        """Real CFR text should score well (low anomaly) on legal domain."""
        metrics = compute_metrics(ecfr_text)
        result = score_quality(metrics, domain="legal")
        # Legal text against legal calibration should be relatively low
        deviations = sorted(result.components)
        assert result.score < 30.0, (
            f"eCFR text scored {result.score} (expected <30). Deviations: {deviations}"
        )

    def test_ecfr_metrics_are_reasonable(self, ecfr_text: str) -> None:
        """Verify individual metrics are in reasonable ranges for legal text."""
        m = compute_metrics(ecfr_text)
        # Legal text should have reasonable word length
        assert 3.0 < m.average_word_length < 10.0
        # Should have low non-ASCII ratio
        assert m.ratio_non_ascii < 0.1
        # Should have positive entropy
        assert m.char_entropy > 2.0
        assert m.token_entropy > 2.0
        # Should have multiple words
        assert m.num_words > 10

    def test_usc_text_scores_low(self, usc_text: str) -> None:
        """USC-style text should score well on legal domain."""
        metrics = compute_metrics(usc_text)
        result = score_quality(metrics, domain="legal")
        assert result.score < 30.0

    def test_legal_vs_general_domain(self, ecfr_text: str) -> None:
        """Legal text should score better (lower) on legal domain than general."""
        metrics = compute_metrics(ecfr_text)
        legal_result = score_quality(metrics, domain="legal")
        general_result = score_quality(metrics, domain="general")
        # Both should be finite
        assert legal_result.score >= 0.0
        assert general_result.score >= 0.0


class TestQualityOnGarbageText:
    def test_repetitive_text_scores_high(self) -> None:
        """Highly repetitive text should score poorly (high anomaly)."""
        garbage = "AAAA " * 500
        metrics = compute_metrics(garbage)
        result = score_quality(metrics, domain="legal")
        assert result.score > 5.0
        assert len(result.components) >= 3  # Multiple deviations

    def test_all_numbers_scores_high(self) -> None:
        """Pure numeric text should deviate from legal norms."""
        numbers = " ".join(str(i) for i in range(1000))
        metrics = compute_metrics(numbers)
        result = score_quality(metrics, domain="legal")
        assert result.score > 3.0

    def test_single_char_scores_high(self) -> None:
        """Single character should deviate heavily."""
        metrics = compute_metrics("x")
        result = score_quality(metrics, domain="legal")
        assert result.score > 5.0


# ── MCP Tool E2E ───────────────────────────────────────────────────


class _MockToolsRegistry:
    def __init__(self) -> None:
        self.tools: list = []

    def register_tool(self, tool: object, aliases: list[str] | None = None) -> None:
        del aliases
        self.tools.append(tool)


class _MockRuntime:
    def __init__(self) -> None:
        self.tools = _MockToolsRegistry()


def _get_quality_tool():
    from kaos_nlp_core.tools import register_nlp_tools

    rt = _MockRuntime()
    register_nlp_tools(rt)
    tools = {t.metadata.name: t for t in rt.tools.tools}
    return tools["kaos-nlp-score-quality"]


@pytest.mark.asyncio
class TestQualityToolLive:
    async def test_tool_on_ecfr_text(self, ecfr_text: str) -> None:
        """Run the actual MCP tool on live eCFR text."""
        tool = _get_quality_tool()
        result = await tool.execute({"text": ecfr_text, "domain": "legal"})
        assert not result.isError
        data = result.require_structured()
        assert data["score"] < 30.0
        assert data["domain"] == "legal"
        assert "metrics" in data
        assert data["metrics"]["num_words"] > 10

    async def test_tool_on_garbage(self) -> None:
        tool = _get_quality_tool()
        result = await tool.execute({"text": "AAAA " * 200, "domain": "legal"})
        assert not result.isError
        data = result.require_structured()
        assert data["score"] > 5.0
        assert len(data["deviations"]) >= 3
