"""Tests for TRADE_BLOCK field extraction in retrospective snapshots.

Verifies that extract_snapshot() correctly extracts structured TRADE_BLOCK
fields (entry_price, stop_price, target_1_price, target_2_price, conviction)
from the trader_plan in analysis results. These fields are used by the
IBKR portfolio reconciliation tool.
"""

from unittest.mock import patch

import pytest

from src.retrospective import (
    _extract_trade_block_fields,
    _extract_trade_block_price,
    extract_snapshot,
)


class TestExtractTradeBlockPrice:
    """Test price extraction from individual TRADE_BLOCK fields."""

    def test_entry_price(self):
        text = "ENTRY: 2,145 (Scaled Limit)\nSTOP: 1,930"
        assert _extract_trade_block_price(text, "ENTRY") == 2145.0

    def test_stop_price_with_pct(self):
        text = "STOP: 1,930 (-10.0%)\nTARGET_1: 2,575"
        assert _extract_trade_block_price(text, "STOP") == 1930.0

    def test_target_price(self):
        text = "TARGET_1: 2,575 (+20.0%)\nTARGET_2: 3,000"
        assert _extract_trade_block_price(text, "TARGET_1") == 2575.0

    def test_decimal_price(self):
        text = "ENTRY: 436.00 (Limit)"
        assert _extract_trade_block_price(text, "ENTRY") == 436.0

    def test_na_price(self):
        text = "ENTRY: N/A (Liquidity Fail)"
        assert _extract_trade_block_price(text, "ENTRY") is None

    def test_missing_field(self):
        text = "SIZE: 5.0%"
        assert _extract_trade_block_price(text, "ENTRY") is None


class TestExtractTradeBlockFields:
    """Test extraction of all 5 structured TRADE_BLOCK fields."""

    FULL_TRADE_BLOCK = """
TRADE_BLOCK:
ACTION: BUY
SIZE: 5.0%
CONVICTION: High
ENTRY: 2,145 (Scaled Limit)
STOP: 1,930 (-10.0%)
TARGET_1: 2,575 (+20.0%)
TARGET_2: 3,000 (+40.0%)
R:R: 3.2:1
SPECIAL: Tokyo hours only; JPY exposure; No ADR.
"""

    REJECT_BLOCK = """
TRADE_BLOCK:
ACTION: REJECT
SIZE: 0.0%
CONVICTION: High
ENTRY: N/A (Liquidity Fail)
STOP: N/A
TARGET_1: N/A
TARGET_2: N/A
R:R: 0:1
SPECIAL: $162 avg daily turnover.
"""

    def test_full_extraction(self):
        result = _extract_trade_block_fields(self.FULL_TRADE_BLOCK)
        assert result["entry_price"] == 2145.0
        assert result["stop_price"] == 1930.0
        assert result["target_1_price"] == 2575.0
        assert result["target_2_price"] == 3000.0
        assert result["conviction"] == "High"

    def test_reject_block(self):
        result = _extract_trade_block_fields(self.REJECT_BLOCK)
        assert result["entry_price"] is None
        assert result["stop_price"] is None
        assert result["target_1_price"] is None
        assert result["target_2_price"] is None
        assert result["conviction"] == "High"

    def test_empty_string(self):
        result = _extract_trade_block_fields("")
        assert result["entry_price"] is None
        assert result["stop_price"] is None
        assert result["conviction"] is None

    def test_no_trade_block(self):
        result = _extract_trade_block_fields("Just some text without any block.")
        assert result["entry_price"] is None
        assert result["conviction"] is None


class TestExtractSnapshotTradeBlockFields:
    """Test that extract_snapshot includes TRADE_BLOCK fields in output."""

    @patch("src.retrospective.config")
    def test_snapshot_includes_trade_block_fields(self, mock_config):
        mock_config.deep_think_llm = "gemini-3-pro-preview"
        mock_config.quick_think_llm = "gemini-2.0-flash"

        result = {
            "final_trade_decision": "VERDICT: BUY\nPM_BLOCK: ...",
            "fundamentals_report": "",
            "investment_analysis": {
                "trader_plan": (
                    "TRADE_BLOCK:\n"
                    "ACTION: BUY\n"
                    "SIZE: 5.0%\n"
                    "CONVICTION: Medium\n"
                    "ENTRY: 2,100 (Limit)\n"
                    "STOP: 1,900 (-9.5%)\n"
                    "TARGET_1: 2,500 (+19.0%)\n"
                    "TARGET_2: 3,000 (+42.9%)\n"
                    "R:R: 4.2:1\n"
                    "SPECIAL: JPY exposure\n"
                ),
            },
        }

        snapshot = extract_snapshot(result, "7203.T")

        # Verify TRADE_BLOCK fields are present
        assert "entry_price" in snapshot
        assert "stop_price" in snapshot
        assert "target_1_price" in snapshot
        assert "target_2_price" in snapshot
        assert "conviction" in snapshot

        assert snapshot["entry_price"] == 2100.0
        assert snapshot["stop_price"] == 1900.0
        assert snapshot["target_1_price"] == 2500.0
        assert snapshot["target_2_price"] == 3000.0
        assert snapshot["conviction"] == "Medium"

    @patch("src.retrospective.config")
    def test_snapshot_backward_compatible_no_trader_plan(self, mock_config):
        """Older analyses without trader_plan should still work (None values)."""
        mock_config.deep_think_llm = "gemini-3-pro-preview"
        mock_config.quick_think_llm = "gemini-2.0-flash"

        result = {
            "final_trade_decision": "VERDICT: HOLD",
            "fundamentals_report": "",
            "investment_analysis": {},  # No trader_plan
        }

        snapshot = extract_snapshot(result, "AAPL")

        assert snapshot["entry_price"] is None
        assert snapshot["stop_price"] is None
        assert snapshot["target_1_price"] is None
        assert snapshot["target_2_price"] is None
        assert snapshot["conviction"] is None

    @patch("src.retrospective.config")
    def test_snapshot_reject_has_none_prices(self, mock_config):
        mock_config.deep_think_llm = "gemini-3-pro-preview"
        mock_config.quick_think_llm = "gemini-2.0-flash"

        result = {
            "final_trade_decision": "",
            "fundamentals_report": "",
            "investment_analysis": {
                "trader_plan": (
                    "TRADE_BLOCK:\n"
                    "ACTION: REJECT\n"
                    "SIZE: 0.0%\n"
                    "CONVICTION: High\n"
                    "ENTRY: N/A (Liquidity Fail)\n"
                    "STOP: N/A\n"
                    "TARGET_1: N/A\n"
                    "TARGET_2: N/A\n"
                ),
            },
        }

        snapshot = extract_snapshot(result, "5840.T")
        assert snapshot["entry_price"] is None
        assert snapshot["stop_price"] is None
        assert snapshot["conviction"] == "High"
