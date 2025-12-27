"""
Tests for token optimization features:
- News highlights extraction (extract_news_highlights)
- Tavily result truncation (_truncate_tavily_result)
- TAVILY_MAX_CHARS configuration
"""

import os


class TestExtractNewsHighlights:
    """Tests for the extract_news_highlights function in agents.py."""

    def test_short_report_passthrough(self):
        """Reports shorter than 300 chars should pass through unchanged."""
        from src.agents import extract_news_highlights

        short_report = "This is a short news report."
        result = extract_news_highlights(short_report)
        assert result == short_report

    def test_empty_report_passthrough(self):
        """Empty reports should pass through unchanged."""
        from src.agents import extract_news_highlights

        result = extract_news_highlights("")
        assert result == ""

    def test_none_report_passthrough(self):
        """None should pass through unchanged."""
        from src.agents import extract_news_highlights

        result = extract_news_highlights(None)
        assert result is None

    def test_extracts_us_revenue_section(self):
        """Should extract US Revenue/Geographic Revenue section."""
        from src.agents import extract_news_highlights

        report = (
            """### GEOGRAPHIC REVENUE VERIFICATION

**US Revenue**: 25%
- Source: Annual Report 2024
- Status: PASS

### OTHER SECTION
This should not be included.
"""
            + "x" * 300
        )  # Pad to exceed 300 chars

        result = extract_news_highlights(report)
        assert "**US/Geographic Revenue:**" in result
        assert "US Revenue" in result or "25%" in result

    def test_extracts_growth_catalysts(self):
        """Should extract growth catalysts section."""
        from src.agents import extract_news_highlights

        report = (
            """### SOME SECTION
Intro text.

### GROWTH CATALYSTS IDENTIFIED

1. **Acquisition**: Company acquired XYZ Corp
2. **Expansion**: New market entry in Asia
3. **R&D**: New product pipeline

### ANOTHER SECTION
More text here.
"""
            + "x" * 300
        )  # Pad to exceed 300 chars

        result = extract_news_highlights(report)
        assert "**Growth Catalysts:**" in result
        assert "Acquisition" in result

    def test_limits_to_three_catalysts(self):
        """Should limit to top 3 catalysts."""
        from src.agents import extract_news_highlights

        report = (
            """### GROWTH CATALYSTS

1. First catalyst
2. Second catalyst
3. Third catalyst
4. Fourth catalyst should be excluded
5. Fifth catalyst should be excluded
"""
            + "x" * 300
        )

        result = extract_news_highlights(report)
        # Count numbered items
        numbered_items = [
            line
            for line in result.split("\n")
            if line.strip().startswith(("1.", "2.", "3.", "4.", "5."))
        ]
        assert len(numbered_items) <= 3

    def test_respects_max_chars(self):
        """Should respect max_chars parameter."""
        from src.agents import extract_news_highlights

        report = (
            """### GEOGRAPHIC REVENUE VERIFICATION
**US Revenue**: 25%
"""
            + "x" * 2000
        )  # Make it long

        result = extract_news_highlights(report, max_chars=500)
        assert len(result) <= 550  # Allow for truncation message

    def test_single_catalyst_header(self):
        """Should not duplicate catalyst header."""
        from src.agents import extract_news_highlights

        report = (
            """### GROWTH CATALYSTS
1. First catalyst

### GROWTH CATALYSTS IDENTIFIED
2. Second catalyst
"""
            + "x" * 300
        )

        result = extract_news_highlights(report)
        # Count occurrences of the header
        header_count = result.count("**Growth Catalysts:**")
        assert header_count <= 1

    def test_fallback_for_no_sections(self):
        """Should return truncated original if no recognizable sections."""
        from src.agents import extract_news_highlights

        report = "x" * 500  # Long report with no recognizable sections
        result = extract_news_highlights(report, max_chars=100)
        assert len(result) <= 150  # Allow for some overhead


class TestTruncateTavilyResult:
    """Tests for the _truncate_tavily_result function in toolkit.py."""

    def test_short_result_passthrough(self):
        """Results shorter than max_chars should pass through unchanged."""
        from src.toolkit import _truncate_tavily_result

        short_result = "This is a short result."
        result = _truncate_tavily_result(short_result, max_chars=1000)
        assert result == short_result

    def test_long_result_truncated(self):
        """Results longer than max_chars should be truncated."""
        from src.toolkit import _truncate_tavily_result

        long_result = "x" * 10000
        result = _truncate_tavily_result(long_result, max_chars=1000)
        assert len(result) < 1100  # Allow for truncation message
        assert "[...truncated for efficiency]" in result

    def test_uses_config_default(self):
        """Should use TAVILY_MAX_CHARS from config when max_chars not specified."""
        from src.config import config
        from src.toolkit import _truncate_tavily_result

        long_result = "x" * 20000
        result = _truncate_tavily_result(long_result)

        # Result should be truncated to config value
        expected_max = config.tavily_max_chars + 50  # Allow for message
        assert len(result) < expected_max

    def test_handles_non_string_input(self):
        """Should convert non-string inputs to string."""
        from src.toolkit import _truncate_tavily_result

        dict_result = {"key": "value", "data": [1, 2, 3]}
        result = _truncate_tavily_result(dict_result, max_chars=1000)
        assert isinstance(result, str)

    def test_handles_list_input(self):
        """Should handle list inputs."""
        from src.toolkit import _truncate_tavily_result

        list_result = [{"title": "Article 1"}, {"title": "Article 2"}]
        result = _truncate_tavily_result(list_result, max_chars=1000)
        assert isinstance(result, str)


class TestTavilyMaxCharsConfig:
    """Tests for TAVILY_MAX_CHARS configuration."""

    def test_default_value(self):
        """Should default to 7000 when env var not set."""
        # Clear the env var if set
        original = os.environ.pop("TAVILY_MAX_CHARS", None)

        try:
            # Re-import to get fresh config
            import importlib

            import src.config

            importlib.reload(src.config)

            assert src.config.config.tavily_max_chars == 7000
        finally:
            # Restore original value
            if original is not None:
                os.environ["TAVILY_MAX_CHARS"] = original

    def test_custom_value_from_env(self):
        """Should use custom value from TAVILY_MAX_CHARS env var."""
        original = os.environ.get("TAVILY_MAX_CHARS")

        try:
            os.environ["TAVILY_MAX_CHARS"] = "5000"

            # Re-import to get fresh config
            import importlib

            import src.config

            importlib.reload(src.config)

            assert src.config.config.tavily_max_chars == 5000
        finally:
            # Restore original value
            if original is not None:
                os.environ["TAVILY_MAX_CHARS"] = original
            else:
                os.environ.pop("TAVILY_MAX_CHARS", None)

    def test_truncation_uses_env_value(self):
        """Truncation should respect the env var value."""
        original = os.environ.get("TAVILY_MAX_CHARS")

        try:
            os.environ["TAVILY_MAX_CHARS"] = "500"

            # Re-import to get fresh config
            import importlib

            import src.config

            importlib.reload(src.config)

            from src.toolkit import _truncate_tavily_result

            long_result = "x" * 2000
            result = _truncate_tavily_result(long_result)

            # Should be truncated to ~500 chars (plus message)
            assert len(result) < 600
        finally:
            # Restore original value
            if original is not None:
                os.environ["TAVILY_MAX_CHARS"] = original
            else:
                os.environ.pop("TAVILY_MAX_CHARS", None)

            # Reload config to restore defaults
            import importlib

            import src.config

            importlib.reload(src.config)


class TestNewsHighlightsIntegration:
    """Integration tests for news highlights in agent context."""

    def test_senior_fundamentals_gets_highlights(self):
        """Verify Senior Fundamentals receives highlights, not full report."""
        # This is tested implicitly by the existing agent tests
        # but we verify the function is used correctly
        from src.agents import extract_news_highlights

        full_report = (
            """### GEOGRAPHIC REVENUE VERIFICATION
**US Revenue**: 30%
- Source: 10-K Filing

### NEWS SOURCES REVIEW
Long section of news sources that should be removed.
"""
            + "x" * 1000
        )

        highlights = extract_news_highlights(full_report)

        # Highlights should be significantly shorter
        assert len(highlights) < len(full_report) * 0.5

        # Should preserve critical US Revenue info
        assert "US" in highlights or "Geographic" in highlights

    def test_reduction_percentage(self):
        """Verify typical reduction is significant (>50%)."""
        from src.agents import extract_news_highlights

        # Simulate a typical news report structure
        typical_report = """### GEOGRAPHIC REVENUE VERIFICATION (Priority #1)

**US Revenue**: Not disclosed in news sources
- **Source**: Q2 FY2026 Earnings Release (Nov 14, 2025)
- **Period**: H1 FY2026 (ended Sept 30, 2025)
- **Status**: NOT AVAILABLE (Neutral)

**Geographic Breakdown**:
- **Domestic (Japan)**: Primary operations through subsidiaries.
- **International**: Recent expansion via acquisition.

---

### NEWS SOURCES REVIEW

**General News Coverage**:
Western sources focus on the company as a component of global index funds.
Mentioned recently for the acquisition of a German company.

**Local/Regional Sources**:
Japanese filings provide granular H1 FY2026 data.
Key local highlight: Management maintained full-year guidance.

---

### GROWTH CATALYSTS IDENTIFIED (Priority #2)

**Verified Catalysts**:

1. **Strategic Acquisition (M&A)**: Completion of acquisition.
   - **Timeline**: Q2 FY2026.
   - **Expected Impact**: Diversification of earnings.
   - **Source**: Nikkei Disclosure.

2. **Shareholder Returns**: Aggressive buyback and dividend policy.
   - **Timeline**: Ongoing.

3. **Interest Rate Tailwind**: BOJ policy normalization.
   - **Timeline**: Gradual over FY2026-2027.
"""

        highlights = extract_news_highlights(typical_report)

        # Should achieve significant reduction
        reduction_pct = (1 - len(highlights) / len(typical_report)) * 100
        assert reduction_pct > 50, f"Expected >50% reduction, got {reduction_pct:.1f}%"
