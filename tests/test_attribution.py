"""Tests for data attribution feature.

This module tests the Tagged Ledger attribution system that tracks
per-field data source provenance through the multi-agent pipeline.
"""

import json

import pytest


class TestAttributionExtractionFromMessages:
    """Test extract_field_sources_from_messages helper (primary method)."""

    def test_extracts_from_tool_message(self):
        """Extract from ToolMessage content."""
        from langchain_core.messages import ToolMessage

        from src.agents import extract_field_sources_from_messages

        tool_content = json.dumps(
            {
                "marketCap": 1000000,
                "trailingPE": 15.2,
                "_field_sources": {"marketCap": "yfinance", "trailingPE": "eodhd"},
            }
        )
        messages = [ToolMessage(content=tool_content, tool_call_id="test-123")]

        result = extract_field_sources_from_messages(messages)
        assert result == {"marketCap": "yfinance", "trailingPE": "eodhd"}

    def test_finds_most_recent_tool_message(self):
        """Find _field_sources from most recent matching ToolMessage."""
        from langchain_core.messages import HumanMessage, ToolMessage

        from src.agents import extract_field_sources_from_messages

        old_content = json.dumps({"_field_sources": {"marketCap": "old_source"}})
        new_content = json.dumps(
            {"_field_sources": {"marketCap": "new_source", "pe": "eodhd"}}
        )
        messages = [
            HumanMessage(content="analyze AAPL"),
            ToolMessage(content=old_content, tool_call_id="old-123"),
            ToolMessage(content="no field sources here", tool_call_id="mid-123"),
            ToolMessage(content=new_content, tool_call_id="new-123"),
        ]

        result = extract_field_sources_from_messages(messages)
        assert result["marketCap"] == "new_source"
        assert result["pe"] == "eodhd"

    def test_skips_non_tool_messages(self):
        """Skip HumanMessage and AIMessage when searching."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from src.agents import extract_field_sources_from_messages

        tool_content = json.dumps({"_field_sources": {"marketCap": "yfinance"}})
        messages = [
            HumanMessage(content="analyze AAPL"),
            AIMessage(content="I'll analyze that for you"),
            ToolMessage(content=tool_content, tool_call_id="test-123"),
        ]

        result = extract_field_sources_from_messages(messages)
        assert result == {"marketCap": "yfinance"}

    def test_returns_empty_on_no_messages(self):
        """Return empty dict when no messages provided."""
        from src.agents import extract_field_sources_from_messages

        assert extract_field_sources_from_messages([]) == {}
        assert extract_field_sources_from_messages(None) == {}

    def test_returns_empty_when_no_field_sources(self):
        """Return empty dict when no ToolMessage has _field_sources."""
        from langchain_core.messages import ToolMessage

        from src.agents import extract_field_sources_from_messages

        messages = [
            ToolMessage(content='{"marketCap": 1000}', tool_call_id="test-123"),
            ToolMessage(content="plain text response", tool_call_id="test-456"),
        ]

        result = extract_field_sources_from_messages(messages)
        assert result == {}

    def test_handles_malformed_json_gracefully(self):
        """Handle malformed JSON in ToolMessage without crashing."""
        from langchain_core.messages import ToolMessage

        from src.agents import extract_field_sources_from_messages

        messages = [
            ToolMessage(content="not valid json {{{", tool_call_id="bad-123"),
            ToolMessage(
                content='{"_field_sources": {"marketCap": "yfinance"}}',
                tool_call_id="good-123",
            ),
        ]

        # Should find the valid one despite the malformed one
        result = extract_field_sources_from_messages(messages)
        assert result == {"marketCap": "yfinance"}


class TestAttributionExtraction:
    """Test extract_field_sources helper (fallback method for raw strings)."""

    def test_extracts_from_clean_json(self):
        """Extract from well-formed JSON."""
        from src.agents import extract_field_sources

        raw = json.dumps(
            {
                "marketCap": 1000000,
                "_field_sources": {"marketCap": "yfinance", "pe": "eodhd"},
            }
        )
        result = extract_field_sources(raw)
        assert result == {"marketCap": "yfinance", "pe": "eodhd"}

    def test_extracts_nested_field_sources(self):
        """Extract from JSON with other metadata fields."""
        from src.agents import extract_field_sources

        raw = json.dumps(
            {
                "marketCap": 25000000000,
                "trailingPE": 15.2,
                "_coverage_pct": 85.0,
                "_sources_used": ["yfinance", "eodhd"],
                "_field_sources": {
                    "marketCap": "eodhd",
                    "trailingPE": "yfinance",
                    "roe": "tavily",
                },
                "_gaps_filled": 3,
            }
        )
        result = extract_field_sources(raw)
        assert result["marketCap"] == "eodhd"
        assert result["trailingPE"] == "yfinance"
        assert result["roe"] == "tavily"

    def test_returns_empty_on_missing_key(self):
        """Return empty dict when _field_sources not present."""
        from src.agents import extract_field_sources

        raw = json.dumps({"marketCap": 1000000, "_sources_used": ["yfinance"]})
        result = extract_field_sources(raw)
        assert result == {}

    def test_returns_empty_on_invalid_json(self):
        """Return empty dict on parse failure."""
        from src.agents import extract_field_sources

        result = extract_field_sources("not json at all")
        assert result == {}

    def test_returns_empty_on_empty_input(self):
        """Return empty dict on empty input."""
        from src.agents import extract_field_sources

        assert extract_field_sources("") == {}
        assert extract_field_sources(None) == {}

    def test_returns_empty_on_non_string(self):
        """Return empty dict on non-string input."""
        from src.agents import extract_field_sources

        assert extract_field_sources(123) == {}
        assert extract_field_sources({"key": "value"}) == {}
        assert extract_field_sources(["list"]) == {}

    def test_handles_embedded_json_in_text(self):
        """Extract when JSON is embedded in LLM response text."""
        from src.agents import extract_field_sources

        # Sometimes raw_fundamentals_data may have preamble text
        raw = 'Here is the data: {"marketCap": 100, "_field_sources": {"marketCap": "tavily"}} end'
        result = extract_field_sources(raw)
        assert result.get("marketCap") == "tavily"

    def test_handles_json_with_no_brace(self):
        """Return empty when string has no JSON structure."""
        from src.agents import extract_field_sources

        result = extract_field_sources("plain text without braces")
        assert result == {}


class TestAttributionFormatting:
    """Test format_attribution_table helper."""

    def test_formats_priority_fields(self):
        """Format table with priority fields only."""
        from src.agents import format_attribution_table

        sources = {
            "marketCap": "eodhd",
            "trailingPE": "yfinance",
            "obscureField": "tavily",  # Should be excluded
        }
        result = format_attribution_table(sources)

        assert "marketCap" in result
        assert "eodhd" in result
        assert "trailingPE" in result
        assert "yfinance" in result
        assert "obscureField" not in result

    def test_formats_all_priority_fields_present(self):
        """Format when all priority fields have sources."""
        from src.agents import format_attribution_table

        sources = {
            "marketCap": "eodhd",
            "netIncome": "yfinance",
            "totalRevenue": "fmp",
            "trailingPE": "eodhd",
            "debtToEquity": "yfinance",
            "freeCashflow": "tavily",
            "roe": "eodhd",
            "currentPrice": "yfinance",
        }
        result = format_attribution_table(sources)

        # All 8 priority fields should be present
        for field in sources:
            assert field in result

        # Should have header
        assert "DATA SOURCE ATTRIBUTION" in result

    def test_returns_empty_on_no_sources(self):
        """Return empty string when no sources."""
        from src.agents import format_attribution_table

        assert format_attribution_table({}) == ""
        assert format_attribution_table(None) == ""

    def test_returns_empty_on_no_priority_fields(self):
        """Return empty string when only non-priority fields present."""
        from src.agents import format_attribution_table

        sources = {
            "obscureField1": "source1",
            "obscureField2": "source2",
        }
        result = format_attribution_table(sources)
        assert result == ""

    def test_output_format_is_markdown_list(self):
        """Verify output is formatted as markdown list."""
        from src.agents import format_attribution_table

        sources = {"marketCap": "eodhd", "trailingPE": "yfinance"}
        result = format_attribution_table(sources)

        # Should have header and markdown list items
        assert "### DATA SOURCE ATTRIBUTION" in result
        assert "- marketCap: eodhd" in result
        assert "- trailingPE: yfinance" in result


class TestAuditorPrompt:
    """Test auditor prompt has required forensic audit instructions."""

    def test_auditor_has_forensic_data_block(self):
        """Verify auditor prompt includes FORENSIC_DATA_BLOCK output format."""
        from pathlib import Path

        prompt_file = Path("prompts/auditor.json")
        with open(prompt_file) as f:
            config = json.load(f)

        system_message = config["system_message"]

        # Check for key forensic audit elements
        assert "FORENSIC_DATA_BLOCK" in system_message
        assert "EARNINGS_QUALITY" in system_message
        assert "ZOMBIE_RATIO" in system_message

    def test_auditor_version_current(self):
        """Verify auditor prompt version is current."""
        from pathlib import Path

        prompt_file = Path("prompts/auditor.json")
        with open(prompt_file) as f:
            config = json.load(f)

        # Version should be at least 2.3 (current stable version)
        version = float(config["version"])
        assert version >= 2.3

    def test_auditor_metadata_has_forensic_tags(self):
        """Verify auditor has forensic accounting capability tags."""
        from pathlib import Path

        prompt_file = Path("prompts/auditor.json")
        with open(prompt_file) as f:
            config = json.load(f)

        tags = config["metadata"]["capability_tags"]
        assert "forensic_accounting" in tags
        assert "multilingual_retrieval" in tags


class TestFetcherAttribution:
    """Test fetcher is configured to expose _field_sources."""

    def test_fetcher_has_field_sources_line(self):
        """Verify fetcher code includes _field_sources in output."""
        from pathlib import Path

        fetcher_path = Path("src/data/fetcher.py")
        content = fetcher_path.read_text()

        # Verify the _field_sources line is present in the fetcher
        assert '"_field_sources"' in content or "'_field_sources'" in content
        assert 'merge_metadata.get("field_sources"' in content

    def test_smart_merge_method_tracks_field_sources(self):
        """Verify _smart_merge_with_quality method initializes field_sources tracking."""
        from pathlib import Path

        fetcher_path = Path("src/data/fetcher.py")
        content = fetcher_path.read_text()

        # The method should initialize field_sources dict
        assert "field_sources = {}" in content
        # And track sources for each field
        assert "field_sources[" in content
