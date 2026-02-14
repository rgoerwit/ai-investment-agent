"""
Tests for Foreign Language Analyst agent and Fundamentals Sync barrier.

Tests cover:
1. Prompt loading and format validation
2. Tool availability (search_foreign_sources)
3. AgentState field for foreign_language_report
4. Fundamentals sync barrier logic
5. Graph structure with new agent
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestForeignLanguageAnalystPrompt:
    """Tests for the Foreign Language Analyst prompt configuration."""

    def test_prompt_file_exists(self):
        """Verify prompt JSON file exists."""
        prompt_path = Path("prompts/foreign_language_analyst.json")
        assert prompt_path.exists(), "Foreign Language Analyst prompt file not found"

    def test_prompt_valid_json(self):
        """Verify prompt file contains valid JSON."""
        prompt_path = Path("prompts/foreign_language_analyst.json")
        with open(prompt_path) as f:
            data = json.load(f)

        # Check required fields
        assert "agent_key" in data
        assert "agent_name" in data
        assert "system_message" in data
        assert "requires_tools" in data

        assert data["agent_key"] == "foreign_language_analyst"
        assert data["requires_tools"] is True

    def test_prompt_has_workflow_instructions(self):
        """Verify prompt contains workflow instructions."""
        prompt_path = Path("prompts/foreign_language_analyst.json")
        with open(prompt_path) as f:
            data = json.load(f)

        system_message = data["system_message"]

        # Should have workflow steps
        assert "INFER CONTEXT" in system_message
        assert "SEARCH" in system_message
        assert "EXTRACT" in system_message or "REPORT" in system_message

    def test_prompt_has_ticker_mappings(self):
        """Verify prompt contains ticker suffix to country/language mappings."""
        prompt_path = Path("prompts/foreign_language_analyst.json")
        with open(prompt_path) as f:
            data = json.load(f)

        system_message = data["system_message"]

        # Should have common suffix mappings
        assert ".T" in system_message  # Japan
        assert ".HK" in system_message  # Hong Kong
        assert ".KS" in system_message or ".KQ" in system_message  # Korea

    def test_prompt_has_fallback_instructions(self):
        """Verify prompt has fallback to premium English sources."""
        prompt_path = Path("prompts/foreign_language_analyst.json")
        with open(prompt_path) as f:
            data = json.load(f)

        system_message = data["system_message"]

        # Should mention premium sources as fallback
        assert (
            "bloomberg" in system_message.lower()
            or "morningstar" in system_message.lower()
        )


class TestSearchForeignSourcesTool:
    """Tests for the search_foreign_sources tool."""

    def test_tool_exists_in_toolkit(self):
        """Verify tool is available in toolkit."""
        from src.toolkit import toolkit

        foreign_tools = toolkit.get_foreign_language_tools()
        assert len(foreign_tools) == 2

        tool_names = [t.name for t in foreign_tools]
        assert "search_foreign_sources" in tool_names
        assert "get_official_filings" in tool_names

    def test_tool_in_all_tools(self):
        """Verify tool is included in get_all_tools."""
        from src.toolkit import toolkit

        all_tools = toolkit.get_all_tools()
        tool_names = [t.name for t in all_tools]
        assert "search_foreign_sources" in tool_names

    def test_tool_has_correct_description(self):
        """Verify tool has informative description."""
        from src.toolkit import toolkit

        foreign_tools = toolkit.get_foreign_language_tools()
        tool = foreign_tools[0]

        assert "foreign" in tool.description.lower()
        assert "source" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_tool_handles_no_tavily(self):
        """Test graceful handling when Tavily is not configured and DDG also empty."""
        from src.toolkit import search_foreign_sources

        # Mock tavily_tool as None AND DDG returning empty
        with patch("src.toolkit.tavily_tool", None):
            with patch("src.toolkit._ddg_search", return_value=[]):
                result = await search_foreign_sources.ainvoke(
                    {"ticker": "7203.T", "search_query": "Toyota 決算短信"}
                )

                assert "no results" in result.lower()


class TestAgentStateField:
    """Tests for foreign_language_report field in AgentState."""

    def test_field_exists_in_agent_state(self):
        """Verify foreign_language_report field exists."""
        from src.agents import AgentState, InvestDebateState, RiskDebateState

        # Create a minimal state
        state = AgentState(
            messages=[],
            company_of_interest="TEST",
            company_name="Test Company",
            trade_date="2025-01-01",
            sender="test",
            market_report="",
            sentiment_report="",
            news_report="",
            raw_fundamentals_data="",
            foreign_language_report="test foreign data",
            fundamentals_report="",
            investment_debate_state=InvestDebateState(
                bull_history="",
                bear_history="",
                history="",
                current_response="",
                judge_decision="",
                count=0,
            ),
            investment_plan="",
            consultant_review="",
            trader_investment_plan="",
            risk_debate_state=RiskDebateState(
                risky_history="",
                safe_history="",
                neutral_history="",
                history="",
                latest_speaker="",
                current_risky_response="",
                current_safe_response="",
                current_neutral_response="",
                judge_decision="",
                count=0,
            ),
            final_trade_decision="",
            tools_called={},
            prompts_used={},
            red_flags=[],
            pre_screening_result="",
        )

        assert state["foreign_language_report"] == "test foreign data"


class TestFundamentalsSyncRouter:
    """Tests for the fundamentals_sync_router function."""

    def test_router_waits_for_all_analysts(self):
        """Test router returns __end__ if not all three analysts complete."""
        from src.graph import fundamentals_sync_router

        # Only Junior done
        state_junior_only = {
            "raw_fundamentals_data": "some data",
            "foreign_language_report": "",
            "legal_report": "",
        }
        result = fundamentals_sync_router(state_junior_only, {})
        assert result == "__end__"

        # Only Foreign done
        state_foreign_only = {
            "raw_fundamentals_data": "",
            "foreign_language_report": "some foreign data",
            "legal_report": "",
        }
        result = fundamentals_sync_router(state_foreign_only, {})
        assert result == "__end__"

        # Only Legal done
        state_legal_only = {
            "raw_fundamentals_data": "",
            "foreign_language_report": "",
            "legal_report": "some legal data",
        }
        result = fundamentals_sync_router(state_legal_only, {})
        assert result == "__end__"

        # Junior + Foreign done (missing Legal)
        state_junior_foreign = {
            "raw_fundamentals_data": "junior data",
            "foreign_language_report": "foreign data",
            "legal_report": "",
        }
        result = fundamentals_sync_router(state_junior_foreign, {})
        assert result == "__end__"

    def test_router_proceeds_when_all_three_complete(self):
        """Test router proceeds to Fundamentals Analyst when all three complete."""
        from src.graph import fundamentals_sync_router

        state_all_done = {
            "raw_fundamentals_data": "junior data",
            "foreign_language_report": "foreign data",
            "legal_report": "legal data",
        }
        result = fundamentals_sync_router(state_all_done, {})
        assert result == "Fundamentals Analyst"

    def test_router_handles_none_values(self):
        """Test router handles None values correctly."""
        from src.graph import fundamentals_sync_router

        state_with_none = {
            "raw_fundamentals_data": None,
            "foreign_language_report": None,
            "legal_report": None,
        }
        result = fundamentals_sync_router(state_with_none, {})
        assert result == "__end__"


class TestGraphStructure:
    """Tests for graph structure with Foreign Language Analyst."""

    @patch("src.graph._is_auditor_enabled")
    def test_fan_out_includes_foreign_analyst(self, mock_auditor_enabled):
        """Test that fan_out_to_analysts includes Foreign Language Analyst."""
        from src.graph import fan_out_to_analysts

        # Disable auditor for this test to check base analyst count
        mock_auditor_enabled.return_value = False

        destinations = fan_out_to_analysts({}, {})

        assert "Foreign Language Analyst" in destinations
        assert "Value Trap Detector" in destinations
        assert (
            len(destinations) == 7
        )  # Market, Sentiment, News, Junior, Foreign, Legal, Value Trap

    def test_route_tools_includes_foreign_analyst(self):
        """Test that route_tools handles foreign_language_analyst sender."""
        from src.graph import route_tools

        state = {"sender": "foreign_language_analyst"}
        result = route_tools(state)

        assert result == "Foreign Language Analyst"

    def test_graph_creates_with_foreign_analyst(self):
        """Test that graph creation includes Foreign Language Analyst node."""
        from src.graph import create_trading_graph

        # Create graph with memory disabled to simplify
        graph = create_trading_graph(
            ticker="TEST", enable_memory=False, quick_mode=True
        )

        # Graph should compile without errors
        assert graph is not None


class TestSeniorFundamentalsContextInjection:
    """Tests for Senior Fundamentals Analyst receiving Foreign Language data."""

    def test_context_injection_code_path_exists(self):
        """Test that the context injection code path for foreign data exists in agents.py."""
        import inspect

        from src import agents

        # Get the source code of create_analyst_node
        source = inspect.getsource(agents.create_analyst_node)

        # Verify the foreign_language_report handling code exists
        assert "foreign_language_report" in source
        assert "foreign_data" in source
        assert "FOREIGN/ALTERNATIVE SOURCE DATA" in source

    def test_analyst_node_created_successfully(self):
        """Test that fundamentals_analyst node can be created."""
        from src.agents import create_analyst_node

        # Create the fundamentals_analyst node (no tools)
        # This just tests the factory function works
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        node_func = create_analyst_node(
            mock_llm, "fundamentals_analyst", [], "fundamentals_report"
        )

        # The node function exists and is callable
        assert callable(node_func)


class TestComputeDataConflicts:
    """Tests for the pre-Senior conflict detection function."""

    def test_ocf_discrepancy_flagged(self):
        """OCF mismatch >30% between Junior and FLA produces a conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"operatingCashflow": 19950000000, "marketCap": 130000000000}'
        fla = (
            "**FILING CASH FLOW**\n"
            "- Operating Cash Flow (Filing): ¥10.91B\n"
            "- Period: H1 2025\n"
        )
        result = compute_data_conflicts(junior, fla)
        assert "OCF" in result
        assert "PERIOD MISMATCH" in result
        assert "yfinance" in result

    def test_ocf_no_conflict_when_close(self):
        """OCF values within 30% produce no conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"operatingCashflow": 10000000000}'
        fla = (
            "**FILING CASH FLOW**\n"
            "- Operating Cash Flow (Filing): ¥11.5B\n"
            "- Period: FY2024\n"
        )
        result = compute_data_conflicts(junior, fla)
        assert "OCF" not in result

    def test_peg_zero_flagged(self):
        """PEG 0.00 produces an UNRELIABLE flag."""
        from src.agents import compute_data_conflicts

        junior = '{"pegRatio": 0.0}'
        result = compute_data_conflicts(junior, "")
        assert "PEG" in result
        assert "UNRELIABLE" in result

    def test_peg_near_zero_flagged(self):
        """PEG 0.02 produces an UNRELIABLE flag with implied growth."""
        from src.agents import compute_data_conflicts

        junior = '{"pegRatio": 0.02}'
        result = compute_data_conflicts(junior, "")
        assert "PEG" in result
        assert "UNRELIABLE" in result
        assert "50x" in result

    def test_peg_normal_no_flag(self):
        """PEG 0.8 produces no flag."""
        from src.agents import compute_data_conflicts

        junior = '{"pegRatio": 0.8}'
        result = compute_data_conflicts(junior, "")
        assert "PEG" not in result

    def test_low_analyst_count_for_large_cap(self):
        """Analyst count < 5 for >$500M market cap flags as anomaly."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 2, "marketCap": 1300000000}'
        result = compute_data_conflicts(junior, "")
        assert "ANALYST_COUNT" in result
        assert "ANOMALY" in result

    def test_low_analyst_count_small_cap_ok(self):
        """Analyst count < 5 for small cap ($200M) is not anomalous."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 2, "marketCap": 200000000}'
        result = compute_data_conflicts(junior, "")
        assert "ANALYST_COUNT" not in result

    def test_parent_company_found(self):
        """FLA finding a parent company flags the ownership gap."""
        from src.agents import compute_data_conflicts

        junior = '{"operatingCashflow": 5000000000}'
        fla = (
            "**OWNERSHIP STRUCTURE**\n"
            "- Controlling Shareholder: Bandai Namco Holdings (49.12%)\n"
        )
        result = compute_data_conflicts(junior, fla)
        assert "PARENT" in result
        assert "Bandai Namco" in result

    def test_empty_junior_returns_nothing(self):
        """No Junior data → empty result."""
        from src.agents import compute_data_conflicts

        result = compute_data_conflicts("", "some FLA data")
        assert result == ""

    def test_no_conflicts_returns_empty(self):
        """Clean data with no issues → empty result."""
        from src.agents import compute_data_conflicts

        junior = (
            '{"pegRatio": 1.2, "numberOfAnalystOpinions": 8, "marketCap": 500000000}'
        )
        result = compute_data_conflicts(junior, "")
        assert result == ""

    def test_header_present_when_conflicts_exist(self):
        """Conflict report starts with AUTOMATED CONFLICT CHECK header."""
        from src.agents import compute_data_conflicts

        junior = '{"pegRatio": 0.0}'
        result = compute_data_conflicts(junior, "")
        assert "AUTOMATED CONFLICT CHECK" in result
        assert "system-generated" in result


class TestLocalAnalystCoverageConflict:
    """Tests for conflict #5: local analyst coverage detection."""

    def test_fla_local_analyst_numeric(self):
        """FLA finds 25 local analysts vs Junior's 3 → conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 3, "marketCap": 500000000}'
        fla = (
            "**LOCAL ANALYST COVERAGE**\n"
            "- Estimated Local Analysts: 25\n"
            "- Key Brokerages: Nomura, Daiwa, SMBC Nikko\n"
        )
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" in result
        assert "25" in result

    def test_fla_local_analyst_tier_high(self):
        """FLA reports HIGH tier → conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 5}'
        fla = "**LOCAL ANALYST COVERAGE**\n" "- Estimated Local Analysts: HIGH\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" in result
        assert "HIGH" in result

    def test_fla_local_analyst_tier_moderate(self):
        """FLA reports MODERATE tier → conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 2}'
        fla = "**LOCAL ANALYST COVERAGE**\n" "- Estimated Local Analysts: MODERATE\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" in result
        assert "MODERATE" in result

    def test_fla_local_analyst_unknown(self):
        """FLA reports UNKNOWN → no conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 3}'
        fla = "**LOCAL ANALYST COVERAGE**\n" "- Estimated Local Analysts: UNKNOWN\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" not in result

    def test_fla_local_analyst_low(self):
        """FLA reports LOW → no conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 3}'
        fla = "**LOCAL ANALYST COVERAGE**\n" "- Estimated Local Analysts: LOW\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" not in result

    def test_fla_no_local_section(self):
        """No LOCAL ANALYST section → no conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 3}'
        fla = "**FILING CASH FLOW**\n" "- Operating Cash Flow (Filing): ¥10.91B\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" not in result

    def test_fla_local_analyst_not_higher(self):
        """FLA finds 2 local analysts but Junior has 5 → no conflict."""
        from src.agents import compute_data_conflicts

        junior = '{"numberOfAnalystOpinions": 5}'
        fla = "**LOCAL ANALYST COVERAGE**\n" "- Estimated Local Analysts: 2\n"
        result = compute_data_conflicts(junior, fla)
        assert "LOCAL_ANALYST_COVERAGE" not in result


class TestPromptLoading:
    """Tests that the prompt system correctly loads Foreign Language Analyst."""

    def test_prompt_loads_via_get_prompt(self):
        """Test that get_prompt can load the foreign_language_analyst prompt."""
        from src.prompts import get_prompt

        prompt = get_prompt("foreign_language_analyst")

        assert prompt is not None
        assert prompt.agent_name == "Foreign Language Analyst"
        assert prompt.requires_tools is True

    def test_prompt_version_is_set(self):
        """Test that prompt has version metadata."""
        from src.prompts import get_prompt

        prompt = get_prompt("foreign_language_analyst")

        assert prompt.version is not None
        assert len(prompt.version) > 0


class TestValueTrapVerdictExtraction:
    """Tests for extract_value_trap_verdict() helper used in PM input assembly."""

    def test_aligned_verdict(self):
        """ALIGNED verdict with high score produces correct header."""
        from src.agents import extract_value_trap_verdict

        report = (
            "SCORE: 85\nVERDICT: ALIGNED\nTRAP_RISK: LOW\n"
            "OWNERSHIP:\n  CONCENTRATION: LOW"
        )
        result = extract_value_trap_verdict(report)
        assert "ALIGNED" in result
        assert "85/100" in result
        assert "LOW" in result

    def test_trap_verdict(self):
        """TRAP verdict surfaces correctly."""
        from src.agents import extract_value_trap_verdict

        report = "SCORE: 25\nVERDICT: TRAP\nTRAP_RISK: HIGH"
        result = extract_value_trap_verdict(report)
        assert "TRAP" in result
        assert "25/100" in result
        assert "HIGH" in result

    def test_missing_verdict_returns_empty(self):
        """Report without VALUE_TRAP_BLOCK fields returns empty string."""
        from src.agents import extract_value_trap_verdict

        result = extract_value_trap_verdict(
            "Some narrative text without structured block"
        )
        assert result == ""

    def test_empty_report_returns_empty(self):
        """Empty or None input returns empty string."""
        from src.agents import extract_value_trap_verdict

        assert extract_value_trap_verdict("") == ""
        assert extract_value_trap_verdict(None) == ""

    def test_missing_trap_risk_still_works(self):
        """SCORE + VERDICT present but TRAP_RISK missing → still produces header."""
        from src.agents import extract_value_trap_verdict

        report = "SCORE: 60\nVERDICT: WATCHABLE"
        result = extract_value_trap_verdict(report)
        assert "WATCHABLE" in result
        assert "60/100" in result
        assert "N/A" in result
