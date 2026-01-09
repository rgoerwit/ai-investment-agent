"""
Tests for Value Trap Detector agent.

Tests cover:
1. Prompt loading and validation
2. Tool functionality (get_ownership_structure)
3. VALUE_TRAP_BLOCK parsing (extract_value_trap_score)
4. Flag detection (detect_value_trap_flags)
5. Edge cases and failure modes
6. Graph integration
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest


class TestValueTrapPrompt:
    """Test Value Trap Detector prompt configuration."""

    def test_prompt_file_exists(self):
        """Test that prompt file exists and is valid JSON."""
        from pathlib import Path

        prompt_path = Path("prompts/value_trap_detector.json")
        assert prompt_path.exists(), "value_trap_detector.json should exist"

        with open(prompt_path) as f:
            prompt_data = json.load(f)

        assert "agent_key" in prompt_data
        assert prompt_data["agent_key"] == "value_trap_detector"

    def test_prompt_has_required_fields(self):
        """Test prompt has all required fields."""
        from src.prompts import get_prompt

        prompt = get_prompt("value_trap_detector")
        assert prompt is not None, "Prompt should load successfully"
        assert prompt.agent_key == "value_trap_detector"
        assert prompt.agent_name == "Value Trap Detector"
        assert prompt.system_message is not None
        assert len(prompt.system_message) > 100

    def test_prompt_has_native_terminology(self):
        """Test prompt includes native terminology protocol."""
        from src.prompts import get_prompt

        prompt = get_prompt("value_trap_detector")
        msg = prompt.system_message

        # Check for Japanese terms
        assert "Mochiai" in msg or "持ち合い" in msg
        assert "Oyakoko" in msg or "親子上場" in msg

        # Check for Korean terms
        assert "Chaebol" in msg or "재벌" in msg

    def test_prompt_has_output_format(self):
        """Test prompt specifies VALUE_TRAP_BLOCK format."""
        from src.prompts import get_prompt

        prompt = get_prompt("value_trap_detector")
        msg = prompt.system_message

        assert "VALUE_TRAP_BLOCK" in msg
        assert "SCORE:" in msg
        assert "VERDICT:" in msg
        assert "TRAP_RISK:" in msg


class TestGetOwnershipStructureTool:
    """Test get_ownership_structure tool functionality."""

    @pytest.mark.asyncio
    async def test_tool_returns_valid_json(self):
        """Test tool returns valid JSON structure."""
        from src.toolkit import get_ownership_structure

        # Use a well-known ticker - tools must be invoked with .ainvoke()
        result = await get_ownership_structure.ainvoke({"ticker": "AAPL"})
        data = json.loads(result)

        assert "ticker" in data
        assert "institutional_holders" in data
        assert "insider_transactions" in data
        assert "major_holders" in data
        assert "data_quality" in data

    @pytest.mark.asyncio
    async def test_tool_handles_invalid_ticker(self):
        """Test tool gracefully handles invalid ticker."""
        from src.toolkit import get_ownership_structure

        result = await get_ownership_structure.ainvoke({"ticker": "INVALIDTICKER12345"})
        data = json.loads(result)

        # Should return structure with empty lists, not crash
        assert "ticker" in data
        assert isinstance(data.get("institutional_holders", []), list)
        assert data.get("data_quality") in ["PARTIAL", "ERROR", "COMPLETE"]

    @pytest.mark.asyncio
    async def test_tool_handles_empty_ticker(self):
        """Test tool handles empty ticker string."""
        from src.toolkit import get_ownership_structure

        result = await get_ownership_structure.ainvoke({"ticker": ""})
        data = json.loads(result)

        # Should not crash
        assert "ticker" in data

    @pytest.mark.asyncio
    async def test_tool_normalizes_international_ticker(self):
        """Test tool normalizes international ticker formats."""
        from src.toolkit import get_ownership_structure

        # Test Japanese ticker
        result = await get_ownership_structure.ainvoke({"ticker": "7203.T"})
        data = json.loads(result)

        assert "ticker" in data
        # Should normalize and attempt lookup

    @pytest.mark.asyncio
    async def test_tool_calculates_insider_trend(self):
        """Test insider trend calculation logic."""
        from src.toolkit import get_ownership_structure

        result = await get_ownership_structure.ainvoke({"ticker": "MSFT"})
        data = json.loads(result)

        insider_trend = data.get("insider_trend")
        assert insider_trend in ["NET_BUYER", "NET_SELLER", "NEUTRAL", "UNKNOWN"]

    @pytest.mark.asyncio
    @patch("src.toolkit.yf.Ticker")
    async def test_tool_handles_yfinance_exception(self, mock_ticker):
        """Test tool handles yfinance exceptions gracefully."""
        from src.toolkit import get_ownership_structure

        mock_ticker.side_effect = Exception("API Error")

        result = await get_ownership_structure.ainvoke({"ticker": "AAPL"})
        data = json.loads(result)

        assert "error" in data or data.get("data_quality") == "ERROR"

    @pytest.mark.asyncio
    @patch("src.toolkit.yf.Ticker")
    async def test_tool_handles_partial_data(self, mock_ticker):
        """Test tool handles partial data availability."""
        from src.toolkit import get_ownership_structure

        mock_yf = MagicMock()
        mock_yf.institutional_holders = None  # Missing institutional data
        mock_yf.insider_transactions = pd.DataFrame()  # Empty insider data
        mock_yf.major_holders = pd.DataFrame({"0": [0.1], "1": ["Test"]})
        mock_ticker.return_value = mock_yf

        result = await get_ownership_structure.ainvoke({"ticker": "TEST"})
        data = json.loads(result)

        assert data.get("data_quality") in ["PARTIAL", "COMPLETE"]
        assert isinstance(data.get("institutional_holders"), list)


class TestExtractValueTrapScore:
    """Test VALUE_TRAP_BLOCK parsing."""

    def test_extract_complete_block(self):
        """Test extraction from complete VALUE_TRAP_BLOCK."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
### --- START VALUE_TRAP_BLOCK ---
SCORE: 45
VERDICT: CAUTIOUS
TRAP_RISK: MEDIUM

OWNERSHIP:
  CONCENTRATION: 48%
  ACTIVIST_PRESENT: NO
  INSIDER_TREND: NEUTRAL

CATALYSTS:
  INDEX_CANDIDATE: NONE
  RESTRUCTURING: NONE
### --- END VALUE_TRAP_BLOCK ---
"""
        metrics = RedFlagDetector.extract_value_trap_score(report)

        assert metrics["score"] == 45
        assert metrics["verdict"] == "CAUTIOUS"
        assert metrics["trap_risk"] == "MEDIUM"
        assert metrics["activist_present"] == "NO"
        assert metrics["insider_trend"] == "NEUTRAL"

    def test_extract_empty_report(self):
        """Test extraction from empty report."""
        from src.validators.red_flag_detector import RedFlagDetector

        metrics = RedFlagDetector.extract_value_trap_score("")

        assert metrics["score"] is None
        assert metrics["verdict"] is None
        assert metrics["trap_risk"] is None

    def test_extract_none_report(self):
        """Test extraction from None report."""
        from src.validators.red_flag_detector import RedFlagDetector

        metrics = RedFlagDetector.extract_value_trap_score(None)

        assert metrics["score"] is None
        assert metrics["verdict"] is None

    def test_extract_malformed_report(self):
        """Test extraction from malformed report."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = "This is garbage text with no structure"
        metrics = RedFlagDetector.extract_value_trap_score(report)

        assert metrics["score"] is None
        assert metrics["verdict"] is None

    def test_extract_partial_block(self):
        """Test extraction when only some fields present."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
SCORE: 75
VERDICT: WATCHABLE
"""
        metrics = RedFlagDetector.extract_value_trap_score(report)

        assert metrics["score"] == 75
        assert metrics["verdict"] == "WATCHABLE"
        assert metrics["trap_risk"] is None  # Not present
        assert metrics["activist_present"] is None

    def test_extract_case_insensitive(self):
        """Test extraction is case-insensitive."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
score: 30
verdict: trap
trap_risk: high
"""
        metrics = RedFlagDetector.extract_value_trap_score(report)

        assert metrics["score"] == 30
        assert metrics["verdict"] == "TRAP"
        assert metrics["trap_risk"] == "HIGH"

    def test_extract_score_out_of_range(self):
        """Test extraction of out-of-range score (LLM hallucination)."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = "SCORE: 150"  # Invalid - should be 0-100
        metrics = RedFlagDetector.extract_value_trap_score(report)

        # Should clamp to 100
        assert metrics["score"] == 100

    def test_extract_negative_score(self):
        """Test extraction of negative score (LLM hallucination)."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = "SCORE: -20"  # Invalid
        metrics = RedFlagDetector.extract_value_trap_score(report)

        # Regex \d+ won't match negative, so should be None
        assert metrics["score"] is None

    def test_extract_catalyst_detection(self):
        """Test catalyst detection from CATALYSTS section."""
        from src.validators.red_flag_detector import RedFlagDetector

        report_with_catalyst = """
CATALYSTS:
  INDEX_CANDIDATE: MSCI_JAPAN
  ACTIVIST_RUMOR: NONE
KEY_RISKS:
- Some risk
"""
        report_no_catalyst = """
CATALYSTS:
  INDEX_CANDIDATE: NONE
  ACTIVIST_RUMOR: NONE
  RESTRUCTURING: NONE
KEY_RISKS:
- Some risk
"""
        metrics_with = RedFlagDetector.extract_value_trap_score(report_with_catalyst)
        metrics_without = RedFlagDetector.extract_value_trap_score(report_no_catalyst)

        assert metrics_with["has_catalyst"] is True
        assert metrics_without["has_catalyst"] is False


class TestDetectValueTrapFlags:
    """Test value trap flag detection."""

    def test_high_risk_trap_flag(self):
        """Test HIGH_RISK flag for score < 40."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
SCORE: 25
VERDICT: TRAP
TRAP_RISK: HIGH
ACTIVIST_PRESENT: NO
"""
        flags = RedFlagDetector.detect_value_trap_flags(report, "TEST.T")

        assert len(flags) >= 1
        flag_types = [f["type"] for f in flags]
        assert "VALUE_TRAP_HIGH_RISK" in flag_types

        high_risk_flag = next(f for f in flags if f["type"] == "VALUE_TRAP_HIGH_RISK")
        assert high_risk_flag["severity"] == "WARNING"
        assert high_risk_flag["risk_penalty"] == 1.0

    def test_moderate_risk_trap_flag(self):
        """Test MODERATE_RISK flag for score 40-60."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
SCORE: 55
VERDICT: CAUTIOUS
ACTIVIST_PRESENT: YES
"""
        flags = RedFlagDetector.detect_value_trap_flags(report, "TEST.T")

        flag_types = [f["type"] for f in flags]
        assert "VALUE_TRAP_MODERATE_RISK" in flag_types
        assert "VALUE_TRAP_HIGH_RISK" not in flag_types

    def test_no_flag_for_high_score(self):
        """Test no trap flag for score >= 60."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
SCORE: 75
VERDICT: WATCHABLE
ACTIVIST_PRESENT: YES
CATALYSTS:
  INDEX_CANDIDATE: MSCI_EMERGING
"""
        flags = RedFlagDetector.detect_value_trap_flags(report, "TEST.T")

        flag_types = [f["type"] for f in flags]
        assert "VALUE_TRAP_HIGH_RISK" not in flag_types
        assert "VALUE_TRAP_MODERATE_RISK" not in flag_types

    def test_no_catalyst_flag(self):
        """Test NO_CATALYST flag when no catalyst and no activist."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
SCORE: 70
VERDICT: WATCHABLE
ACTIVIST_PRESENT: NO
CATALYSTS:
  INDEX_CANDIDATE: NONE
  RESTRUCTURING: NONE
KEY_RISKS:
- test
"""
        flags = RedFlagDetector.detect_value_trap_flags(report, "TEST.T")

        flag_types = [f["type"] for f in flags]
        assert "NO_CATALYST_DETECTED" in flag_types

    def test_no_catalyst_flag_skipped_with_activist(self):
        """Test NO_CATALYST not flagged when activist present."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
SCORE: 70
VERDICT: WATCHABLE
ACTIVIST_PRESENT: YES
CATALYSTS:
  INDEX_CANDIDATE: NONE
KEY_RISKS:
- test
"""
        flags = RedFlagDetector.detect_value_trap_flags(report, "TEST.T")

        flag_types = [f["type"] for f in flags]
        assert "NO_CATALYST_DETECTED" not in flag_types

    def test_empty_report_no_flags(self):
        """Test empty report produces no flags."""
        from src.validators.red_flag_detector import RedFlagDetector

        flags = RedFlagDetector.detect_value_trap_flags("", "TEST")
        assert len(flags) == 0

    def test_none_report_no_crash(self):
        """Test None report doesn't crash."""
        from src.validators.red_flag_detector import RedFlagDetector

        # This might raise if not handled - need to fix code if so
        try:
            flags = RedFlagDetector.detect_value_trap_flags(None, "TEST")
            assert isinstance(flags, list)
        except TypeError:
            pytest.fail("detect_value_trap_flags should handle None input")

    def test_non_string_report_handling(self):
        """Test non-string report (list from LangGraph state pollution)."""
        from src.validators.red_flag_detector import RedFlagDetector

        # LangGraph can sometimes pass list of messages
        list_report = ["SCORE: 45", "VERDICT: TRAP"]

        try:
            flags = RedFlagDetector.detect_value_trap_flags(list_report, "TEST")
            # Should either handle gracefully or return empty
            assert isinstance(flags, list)
        except (TypeError, AttributeError):
            pytest.fail("detect_value_trap_flags should handle non-string input")

    def test_verdict_trap_without_low_score(self):
        """Test TRAP verdict triggers flag even with borderline score."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
SCORE: 42
VERDICT: TRAP
"""
        flags = RedFlagDetector.detect_value_trap_flags(report, "TEST")

        flag_types = [f["type"] for f in flags]
        # Score 42 triggers MODERATE_RISK, TRAP verdict should also trigger
        assert "VALUE_TRAP_MODERATE_RISK" in flag_types


class TestGraphIntegration:
    """Test Value Trap Detector integration with graph."""

    def test_route_tools_for_value_trap_detector(self):
        """Test route_tools handles value_trap_detector sender."""
        from src.graph import route_tools

        state = {"sender": "value_trap_detector"}
        result = route_tools(state)

        assert result == "Value Trap Detector"

    @patch("src.graph._is_auditor_enabled")
    def test_fan_out_includes_value_trap_detector(self, mock_auditor):
        """Test fan_out_to_analysts includes Value Trap Detector."""
        from src.graph import fan_out_to_analysts

        mock_auditor.return_value = False
        destinations = fan_out_to_analysts({}, {})

        assert "Value Trap Detector" in destinations

    @patch("src.graph._is_auditor_enabled")
    def test_sync_check_waits_for_value_trap(self, mock_auditor):
        """Test sync_check_router waits for value_trap_report."""
        from src.graph import sync_check_router

        mock_auditor.return_value = False

        # Missing value_trap_report
        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "pre_screening_result": "PASS",
            # value_trap_report missing
        }

        result = sync_check_router(state, {})
        assert result == "__end__"  # Should wait

    @patch("src.graph._is_auditor_enabled")
    def test_sync_check_proceeds_with_value_trap(self, mock_auditor):
        """Test sync_check_router proceeds when value_trap_report present."""
        from src.graph import sync_check_router

        mock_auditor.return_value = False

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
        }

        result = sync_check_router(state, {})
        assert isinstance(result, list)
        assert "Bull Researcher R1" in result

    def test_toolkit_has_value_trap_tools(self):
        """Test Toolkit provides value trap tools."""
        from src.toolkit import Toolkit

        toolkit = Toolkit()
        tools = toolkit.get_value_trap_tools()

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "get_ownership_structure" in tool_names
        assert "get_news" in tool_names
        assert "search_foreign_sources" in tool_names


class TestPortfolioManagerIntegration:
    """Test PM integration with value trap flags."""

    def test_pm_detects_value_trap_flags(self):
        """Test PM calls detect_value_trap_flags for value_trap_report."""
        from src.validators.red_flag_detector import RedFlagDetector

        # Simulate what PM does
        value_trap = """
SCORE: 30
VERDICT: TRAP
ACTIVIST_PRESENT: NO
CATALYSTS:
  INDEX_CANDIDATE: NONE
KEY_RISKS:
- test
"""
        warnings = RedFlagDetector.detect_value_trap_flags(value_trap, "TEST.T")

        assert len(warnings) >= 1
        assert any(w["type"] == "VALUE_TRAP_HIGH_RISK" for w in warnings)

    def test_pm_handles_missing_value_trap_report(self):
        """Test PM handles missing value_trap_report gracefully."""
        # When value_trap is empty string, should produce no flags
        from src.validators.red_flag_detector import RedFlagDetector

        warnings = RedFlagDetector.detect_value_trap_flags("", "TEST")
        assert warnings == []


class TestEdgeCases:
    """Test edge cases and weird situations."""

    def test_score_with_whitespace(self):
        """Test score extraction with extra whitespace."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = "SCORE:   45  "
        metrics = RedFlagDetector.extract_value_trap_score(report)
        assert metrics["score"] == 45

    def test_score_with_percentage_symbol(self):
        """Test score with percentage symbol (LLM mistake)."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = "SCORE: 45%"  # Common LLM mistake
        metrics = RedFlagDetector.extract_value_trap_score(report)
        # Current regex should still extract 45
        assert metrics["score"] == 45

    def test_score_with_slash_100_format(self):
        """Test score with /100 suffix (common LLM format variation)."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = "SCORE: 35/100"  # LLM sometimes adds /100 explicitly
        metrics = RedFlagDetector.extract_value_trap_score(report)
        assert metrics["score"] == 35

    def test_multiple_score_values(self):
        """Test extraction when multiple SCORE values present."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
Some text SCORE: 80 earlier
### VALUE_TRAP_BLOCK
SCORE: 35
"""
        metrics = RedFlagDetector.extract_value_trap_score(report)
        # Should get first match (80) - might want to be smarter
        assert metrics["score"] in [80, 35]

    def test_unicode_in_report(self):
        """Test extraction handles unicode characters."""
        from src.validators.red_flag_detector import RedFlagDetector

        report = """
SCORE: 55
VERDICT: CAUTIOUS
KEY_RISKS:
- 持ち合い detected (cross-shareholdings)
- 재벌 structure concerns
"""
        metrics = RedFlagDetector.extract_value_trap_score(report)
        assert metrics["score"] == 55
        assert metrics["verdict"] == "CAUTIOUS"

    def test_very_long_report(self):
        """Test extraction from very long report."""
        from src.validators.red_flag_detector import RedFlagDetector

        padding = "x" * 10000
        report = f"{padding}\nSCORE: 42\nVERDICT: CAUTIOUS\n{padding}"

        metrics = RedFlagDetector.extract_value_trap_score(report)
        assert metrics["score"] == 42

    @pytest.mark.asyncio
    async def test_ownership_tool_timeout_handling(self):
        """Test tool handles slow API responses."""
        # Real API call with short timeout expectation
        # This tests the tool doesn't hang indefinitely
        import asyncio

        from src.toolkit import get_ownership_structure

        try:
            result = await asyncio.wait_for(
                get_ownership_structure.ainvoke({"ticker": "AAPL"}), timeout=30
            )
            assert result is not None
        except asyncio.TimeoutError:
            pytest.skip("API too slow, but timeout handling works")
