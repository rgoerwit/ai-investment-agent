"""
Tests for verification mitigations:
- Source conflict detection in fetcher
- Conflict table formatting in agents
- Consultant spot-check tool
- Consultant tool loop
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Source Conflict Detection (fetcher._smart_merge_with_quality)
# =============================================================================


class TestSourceConflictDetection:
    """Tests for source disagreement recording in the data fetcher."""

    def test_conflict_recorded_on_large_variance(self):
        """When sources disagree >20%, a conflict should be recorded."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        # Two sources with >20% variance on trailingPE
        source_results = {
            "yahooquery": {"trailingPE": 10.0, "currentPrice": 100.0},
            "yfinance": {"trailingPE": 15.0, "currentPrice": 100.0},  # 50% higher
        }

        merged, metadata = fetcher._smart_merge_with_quality(source_results, "TEST.T")

        assert "source_conflicts" in metadata
        conflicts = metadata["source_conflicts"]
        # trailingPE should show conflict (50% variance)
        assert "trailingPE" in conflicts
        assert conflicts["trailingPE"]["variance_pct"] == 50.0
        # currentPrice should NOT show conflict (same value)
        assert "currentPrice" not in conflicts

    def test_no_conflict_on_small_variance(self):
        """When sources agree within 20%, no conflict recorded."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        source_results = {
            "yahooquery": {"trailingPE": 10.0},
            "yfinance": {"trailingPE": 11.5},  # 15% — below threshold
        }

        merged, metadata = fetcher._smart_merge_with_quality(source_results, "TEST.T")

        conflicts = metadata.get("source_conflicts", {})
        assert "trailingPE" not in conflicts

    def test_conflict_includes_source_names(self):
        """Conflict records should include old/new source names."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        source_results = {
            "yahooquery": {"debtToEquity": 50.0},
            "yfinance": {"debtToEquity": 150.0},  # 200% variance
        }

        merged, metadata = fetcher._smart_merge_with_quality(source_results, "TEST.T")

        conflict = metadata["source_conflicts"]["debtToEquity"]
        assert conflict["old_source"] == "yahooquery"
        assert conflict["new_source"] == "yfinance"
        assert conflict["variance_pct"] == 200.0

    def test_nonnumeric_fields_skip_conflict_check(self):
        """Non-numeric fields should not trigger conflict detection."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        source_results = {
            "yahooquery": {"sector": "Technology", "trailingPE": 10.0},
            "yfinance": {"sector": "Tech", "trailingPE": 10.0},
        }

        merged, metadata = fetcher._smart_merge_with_quality(source_results, "TEST.T")

        conflicts = metadata.get("source_conflicts", {})
        assert "sector" not in conflicts

    def test_forward_pe_single_source_outlier_is_quarantined(self):
        """A single outrageous forwardPE should be dropped when a sane peer exists."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        source_results = {
            "yahooquery": {"forwardPE": 896.7947},
            "yfinance": {"forwardPE": 11.9197},
        }

        merged, metadata = fetcher._smart_merge_with_quality(source_results, "RIO.L")

        assert merged["forwardPE"] == 11.9197
        assert metadata["field_sources"]["forwardPE"] == "yfinance"
        assert "forwardPE" not in metadata.get("source_conflicts", {})

    def test_forward_pe_higher_quality_outlier_is_also_quarantined(self):
        """Quarantine should key off single-source implausibility, not source rank."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        source_results = {
            "yahooquery": {"forwardPE": 11.9197},
            "yfinance": {"forwardPE": 896.7947},
        }

        merged, metadata = fetcher._smart_merge_with_quality(source_results, "RIO.L")

        assert merged["forwardPE"] == 11.9197
        assert metadata["field_sources"]["forwardPE"] == "yahooquery"
        assert "forwardPE" not in metadata.get("source_conflicts", {})

    def test_forward_pe_not_quarantined_when_multiple_sources_are_outrageous(self):
        """If sources agree the forwardPE is extreme, keep it and record normal conflicts."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        source_results = {
            "yahooquery": {"forwardPE": 900.0},
            "yfinance": {"forwardPE": 500.0},
        }

        merged, metadata = fetcher._smart_merge_with_quality(source_results, "TEST.L")

        assert merged["forwardPE"] == 500.0
        assert metadata["field_sources"]["forwardPE"] == "yfinance"
        assert metadata["source_conflicts"]["forwardPE"]["variance_pct"] == 44.4

    def test_forward_pe_outlier_thresholds_respect_boundaries(self):
        """Do not quarantine at the exact threshold or below the ratio cutoff."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        threshold_case = {
            "yahooquery": {"forwardPE": 200.0},
            "yfinance": {"forwardPE": 20.0},
        }
        merged_threshold, metadata_threshold = fetcher._smart_merge_with_quality(
            threshold_case, "BOUNDARY.L"
        )
        assert merged_threshold["forwardPE"] == 20.0
        assert "forwardPE" in metadata_threshold.get("source_conflicts", {})

        ratio_case = {
            "yahooquery": {"forwardPE": 250.0},
            "yfinance": {"forwardPE": 60.0},
        }
        merged_ratio, metadata_ratio = fetcher._smart_merge_with_quality(
            ratio_case, "BOUNDARY.L"
        )
        assert merged_ratio["forwardPE"] == 60.0
        assert "forwardPE" in metadata_ratio.get("source_conflicts", {})

    def test_microstructure_conflicts_are_suppressed_from_metadata_and_warnings(self):
        """Bid/ask size mismatches should not pollute operator conflict logs."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        source_results = {
            "yahooquery": {
                "bidSize": 1076000.0,
                "askSize": 1000000.0,
                "bid": 123.0,
                "ask": 125.0,
            },
            "yfinance": {
                "bidSize": 10760.0,
                "askSize": 10000.0,
                "bid": 1.23,
                "ask": 1.25,
            },
        }

        with patch("src.data.fetcher.logger") as mock_logger:
            merged, metadata = fetcher._smart_merge_with_quality(
                source_results, "RY4C.DE"
            )

        assert merged["bidSize"] == 10760.0
        assert merged["askSize"] == 10000.0
        assert metadata["field_sources"]["bidSize"] == "yfinance"
        assert metadata["field_sources"]["askSize"] == "yfinance"
        assert "bidSize" not in metadata["source_conflicts"]
        assert "askSize" not in metadata["source_conflicts"]
        assert "bid" not in metadata["source_conflicts"]
        assert "ask" not in metadata["source_conflicts"]
        assert not any(
            call.args and call.args[0] == "source_data_conflicts"
            for call in mock_logger.warning.call_args_list
        )

    def test_forward_pe_conflict_kept_in_metadata_but_downgraded_when_resolved(self):
        """Lower-trust forwardPE disagreements should stay in metadata but not warn."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        source_results = {
            "yahooquery": {"forwardPE": 77.9191},
            "yfinance": {"forwardPE": 8.1212},
        }

        with patch("src.data.fetcher.logger") as mock_logger:
            merged, metadata = fetcher._smart_merge_with_quality(
                source_results, "SEA1.OL"
            )

        assert merged["forwardPE"] == 8.1212
        conflict = metadata["source_conflicts"]["forwardPE"]
        assert conflict["field_class"] == "valuation"
        assert conflict["resolved_by_quality"] is True
        assert conflict["winner_quality"] == 9
        assert conflict["loser_quality"] == 6
        assert not any(
            call.args and call.args[0] == "source_data_conflicts"
            for call in mock_logger.warning.call_args_list
        )
        assert any(
            call.args and call.args[0] == "source_data_conflicts_resolved"
            for call in mock_logger.debug.call_args_list
        )

    def test_critical_peer_conflict_still_warns(self):
        """Near-peer high-trust conflicts should remain warning-level visible."""
        from src.data.fetcher import SmartMarketDataFetcher as MarketDataFetcher

        fetcher = MarketDataFetcher.__new__(MarketDataFetcher)
        fetcher.stats = {"basics_ok": 0, "basics_failed": 0}

        source_results = {
            "alpha_vantage": {"forwardPE": 10.0},
            "yfinance": {"forwardPE": 15.0},
        }

        with patch("src.data.fetcher.logger") as mock_logger:
            merged, metadata = fetcher._smart_merge_with_quality(
                source_results, "PEER.L"
            )

        assert merged["forwardPE"] == 10.0
        conflict = metadata["source_conflicts"]["forwardPE"]
        assert conflict["resolved_by_quality"] is False
        assert conflict["winner_quality"] == 9
        assert conflict["loser_quality"] == 9
        assert any(
            call.args and call.args[0] == "source_data_conflicts"
            for call in mock_logger.warning.call_args_list
        )


# =============================================================================
# Conflict Table Formatting (agents.format_conflict_table)
# =============================================================================


class TestFormatConflictTable:
    """Tests for format_conflict_table in agents.py."""

    def test_empty_when_no_conflicts(self):
        """Should return empty string when no conflicts in messages."""
        from src.agents import format_conflict_table

        result = format_conflict_table([])
        assert result == ""

    def test_formats_conflicts_from_tool_messages(self):
        """Should extract and format conflicts from ToolMessage content."""
        from langchain_core.messages import ToolMessage

        from src.agents import format_conflict_table

        tool_content = json.dumps(
            {
                "trailingPE": 12.5,
                "_source_conflicts": {
                    "trailingPE": {
                        "old": 10.0,
                        "old_source": "yahooquery",
                        "new": 15.0,
                        "new_source": "yfinance",
                        "variance_pct": 50.0,
                    }
                },
                "_field_sources": {"trailingPE": "yfinance"},
            }
        )
        messages = [
            ToolMessage(content=tool_content, tool_call_id="call_1"),
        ]

        result = format_conflict_table(messages)
        assert "DATA SOURCE CONFLICTS" in result
        assert "trailingPE" in result
        assert "yahooquery" in result
        assert "yfinance" in result
        assert "50.0%" in result

    def test_returns_empty_when_conflicts_dict_empty(self):
        """Should return empty string when _source_conflicts exists but is empty."""
        from langchain_core.messages import ToolMessage

        from src.agents import format_conflict_table

        tool_content = json.dumps(
            {
                "trailingPE": 12.5,
                "_source_conflicts": {},
                "_field_sources": {},
            }
        )
        messages = [ToolMessage(content=tool_content, tool_call_id="call_1")]

        result = format_conflict_table(messages)
        assert result == ""

    def test_forward_pe_conflicts_with_annotations_remain_visible(self):
        """Conflict-table rendering should ignore new metadata annotations."""
        from langchain_core.messages import ToolMessage

        from src.agents import format_conflict_table

        tool_content = json.dumps(
            {
                "forwardPE": 8.1212,
                "bidSize": 10760.0,
                "_source_conflicts": {
                    "forwardPE": {
                        "old": 77.9191,
                        "old_source": "yahooquery",
                        "new": 8.1212,
                        "new_source": "yfinance",
                        "variance_pct": 89.6,
                        "field_class": "valuation",
                        "resolved_by_quality": True,
                        "winner_quality": 9,
                        "loser_quality": 6,
                    }
                },
                "_field_sources": {"forwardPE": "yfinance", "bidSize": "yfinance"},
            }
        )
        messages = [ToolMessage(content=tool_content, tool_call_id="call_1")]

        result = format_conflict_table(messages)
        assert "forwardPE" in result
        assert "yahooquery" in result
        assert "yfinance" in result
        assert "bidSize" not in result


# =============================================================================
# Consultant spot_check_metric Tool
# =============================================================================


class TestSpotCheckMetricTool:
    """Tests for the consultant's spot_check_metric tool."""

    @pytest.mark.asyncio
    async def test_disallowed_metric_rejected(self):
        """Metrics not in ALLOWED_FIELDS should be rejected."""
        from src.consultant_tools import spot_check_metric

        result = await spot_check_metric.ainvoke(
            {"ticker": "7203.T", "metric": "secretInternalScore"}
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "allowed" in parsed

    @pytest.mark.asyncio
    async def test_allowed_metric_fetched(self):
        """Valid metric should query yfinance and return JSON."""
        from src.consultant_tools import spot_check_metric

        mock_info = {"trailingPE": 14.5, "currentPrice": 2500.0}

        with patch("src.consultant_tools.yf.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_ticker_cls.return_value = mock_ticker

            result = await spot_check_metric.ainvoke(
                {"ticker": "7203.T", "metric": "trailingPE"}
            )
            parsed = json.loads(result)
            assert parsed["value"] == 14.5
            assert parsed["source"] == "yfinance_direct"
            assert parsed["ticker"] == "7203.T"

    @pytest.mark.asyncio
    async def test_handles_yfinance_error(self):
        """Should return error JSON when yfinance fails."""
        from src.consultant_tools import spot_check_metric

        with patch("src.consultant_tools.yf.Ticker", side_effect=Exception("API down")):
            result = await spot_check_metric.ainvoke(
                {"ticker": "INVALID", "metric": "trailingPE"}
            )
            parsed = json.loads(result)
            assert "error" in parsed

    def test_get_consultant_tools_returns_list(self):
        """get_consultant_tools should return independent (non-yfinance) tools only."""
        from src.consultant_tools import get_consultant_tools

        tools = get_consultant_tools()
        assert len(tools) >= 1
        tool_names = [t.name for t in tools]
        # spot_check_metric (yfinance) deliberately excluded — circular validation
        assert "spot_check_metric" not in tool_names
        assert "spot_check_metric_alt" in tool_names


# =============================================================================
# Consultant Tool Loop Integration
# =============================================================================


class TestConsultantToolLoop:
    """Tests for the consultant node's bounded tool loop."""

    @pytest.mark.asyncio
    async def test_consultant_no_tools_single_shot(self):
        """Without tools, consultant should work as single-shot invocation."""
        from src.agents import create_consultant_node

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "CONSULTANT REVIEW: APPROVED"
        mock_response.tool_calls = None
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        node = create_consultant_node(mock_llm, "consultant", tools=None)

        state = {
            "company_of_interest": "TEST.T",
            "company_name": "Test Corp",
            "market_report": "Market OK",
            "sentiment_report": "Sentiment OK",
            "news_report": "News OK",
            "fundamentals_report": "DATA_BLOCK: OK",
            "investment_plan": "BUY",
            "investment_debate_state": {"history": "Bull vs Bear"},
            "red_flags": [],
            "pre_screening_result": "PASS",
            "auditor_report": "N/A",
            "messages": [],
        }

        result = await node(state, MagicMock())
        assert "consultant_review" in result
        assert "APPROVED" in result["consultant_review"]

    @pytest.mark.asyncio
    async def test_consultant_with_tools_executes_loop(self):
        """With tools, consultant should execute tool calls and loop."""
        from src.agents import create_consultant_node

        # First response: tool call
        tool_response = MagicMock()
        tool_response.content = ""
        tool_response.tool_calls = [
            {
                "name": "spot_check_metric",
                "args": {"ticker": "TEST.T", "metric": "trailingPE"},
                "id": "call_1",
            }
        ]

        # Second response: final answer (no tool calls)
        final_response = MagicMock()
        final_response.content = (
            "CONSULTANT REVIEW: APPROVED\nSPOT_CHECK trailingPE: CONFIRMED"
        )
        final_response.tool_calls = []

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        # Mock tool
        mock_tool = AsyncMock()
        mock_tool.name = "spot_check_metric"
        mock_tool.ainvoke = AsyncMock(
            return_value='{"ticker": "TEST.T", "metric": "trailingPE", "value": 14.5, "source": "yfinance_direct"}'
        )

        node = create_consultant_node(mock_llm, "consultant", tools=[mock_tool])

        state = {
            "company_of_interest": "TEST.T",
            "company_name": "Test Corp",
            "market_report": "Market OK",
            "sentiment_report": "Sentiment OK",
            "news_report": "News OK",
            "fundamentals_report": "DATA_BLOCK: OK",
            "investment_plan": "BUY",
            "investment_debate_state": {"history": "Bull vs Bear"},
            "red_flags": [],
            "pre_screening_result": "PASS",
            "auditor_report": "N/A",
            "messages": [],
        }

        result = await node(state, MagicMock())
        assert "consultant_review" in result
        assert "CONFIRMED" in result["consultant_review"]
        # Verify tool was actually called
        mock_tool.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_consultant_respects_max_iterations(self):
        """Tool loop should terminate after MAX_TOOL_ITERATIONS."""
        from src.agents import create_consultant_node

        # Always return tool calls (never gives final answer)
        tool_response = MagicMock()
        tool_response.content = "I need to check more metrics..."
        tool_response.tool_calls = [
            {
                "name": "spot_check_metric",
                "args": {"ticker": "X", "metric": "trailingPE"},
                "id": "call_x",
            }
        ]

        # Final forced response (when safety valve triggers)
        final_response = MagicMock()
        final_response.content = "CONSULTANT REVIEW: CONDITIONAL APPROVAL"
        final_response.tool_calls = []

        mock_llm = AsyncMock()
        # MAX_TOOL_ITERATIONS=3: iterations 0,1,2 return tool calls, iteration 3 hits
        # the max and extracts content from that response
        mock_llm.ainvoke = AsyncMock(
            side_effect=[
                tool_response,
                tool_response,
                tool_response,
                tool_response,
                final_response,
            ]
        )
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        mock_tool = AsyncMock()
        mock_tool.name = "spot_check_metric"
        mock_tool.ainvoke = AsyncMock(return_value='{"value": 10}')

        node = create_consultant_node(mock_llm, "consultant", tools=[mock_tool])

        state = {
            "company_of_interest": "TEST.T",
            "company_name": "Test Corp",
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_plan": "",
            "investment_debate_state": {"history": ""},
            "red_flags": [],
            "pre_screening_result": "PASS",
            "auditor_report": "N/A",
            "messages": [],
        }

        result = await node(state, MagicMock())
        assert "consultant_review" in result
        # MAX_TOOL_ITERATIONS=3: iterations 0, 1, 2 each produce tool calls,
        # iteration 3 == MAX extracts content from that response.
        # Total LLM calls: 4
        assert mock_llm.ainvoke.call_count == 4
