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
        assert len(foreign_tools) == 1

        tool = foreign_tools[0]
        assert tool.name == "search_foreign_sources"

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
        """Test graceful handling when Tavily is not configured."""
        from src.toolkit import search_foreign_sources

        # Mock tavily_tool as None
        with patch("src.toolkit.tavily_tool", None):
            result = await search_foreign_sources.ainvoke(
                {"ticker": "7203.T", "search_query": "Toyota 決算短信"}
            )

            assert "unavailable" in result.lower() or "not configured" in result.lower()


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

    def test_fan_out_includes_foreign_analyst(self):
        """Test that fan_out_to_analysts includes Foreign Language Analyst."""
        from src.graph import fan_out_to_analysts

        destinations = fan_out_to_analysts({}, {})

        assert "Foreign Language Analyst" in destinations
        assert len(destinations) == 6  # Market, Sentiment, News, Junior, Foreign, Legal

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
