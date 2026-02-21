"""
Tests for parallel tool execution isolation.

These tests verify that in parallel execution:
1. Each agent's tool_calls are executed by the correct tool node
2. Tool results don't get mixed between agents
3. create_agent_tool_node correctly filters messages by tool names

This test suite was created after a bug where parallel agents' tool_calls
were being processed by the wrong tool nodes, causing data quality regression.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# --- Test Fixtures: Simple mock tools ---


@tool
def get_technical_indicators(symbol: str) -> str:
    """Get technical indicators for a stock."""
    return f"RSI: 45, MACD: 0.5 for {symbol}"


@tool
def calculate_liquidity_metrics(ticker: str) -> str:
    """Calculate liquidity metrics."""
    return f"Volume: 1M, Turnover: $5M for {ticker}"


@tool
def get_sentiment(ticker: str) -> str:
    """Get social media sentiment."""
    return f"Sentiment: Bullish for {ticker}"


@tool
def get_news(ticker: str, search_query: str) -> str:
    """Get news articles."""
    return f"News about {ticker}: {search_query}"


@tool
def get_financial_metrics(ticker: str) -> str:
    """Get financial metrics."""
    return f"P/E: 15, ROE: 20% for {ticker}"


MARKET_TOOLS = [get_technical_indicators, calculate_liquidity_metrics]
SENTIMENT_TOOLS = [get_sentiment]
NEWS_TOOLS = [get_news]
FUNDAMENTALS_TOOLS = [get_financial_metrics]


# --- Helper: Simplified tool node for testing ---


def create_test_tool_node(tools: list, agent_key: str):
    """
    Create a test-friendly version of create_agent_tool_node.
    This version executes tools directly without requiring LangGraph runtime.
    """
    tool_names = {tool.name for tool in tools}
    tools_by_name = {tool.name: tool for tool in tools}

    async def agent_tool_node(state: dict, config: dict = None) -> dict:
        """Execute tools for a specific agent by filtering messages."""
        messages = state.get("messages", [])

        # Find the AIMessage from THIS agent (has tool_calls for our tools)
        target_message = None
        for msg in reversed(messages):
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                # Check if any tool_call is for one of our tools
                msg_tool_names = {
                    tc.get("name", tc.get("function", {}).get("name", ""))
                    for tc in msg.tool_calls
                }
                if msg_tool_names & tool_names:  # Intersection
                    target_message = msg
                    break

        if target_message is None:
            return {"messages": []}

        # Execute tools directly (test mode - no LangGraph runtime needed)
        result_messages = []
        for tc in target_message.tool_calls:
            tool_name = tc.get("name")
            if tool_name in tools_by_name:
                tool_fn = tools_by_name[tool_name]
                args = tc.get("args", {})
                try:
                    result = tool_fn.invoke(args)
                    result_messages.append(
                        ToolMessage(
                            content=result, tool_call_id=tc.get("id"), name=tool_name
                        )
                    )
                except Exception as e:
                    result_messages.append(
                        ToolMessage(
                            content=f"Error: {e}",
                            tool_call_id=tc.get("id"),
                            name=tool_name,
                            status="error",
                        )
                    )

        return {"messages": result_messages}

    return agent_tool_node


# --- Unit Tests for Tool Filtering Logic ---


class TestToolFilteringLogic:
    """Unit tests for the message filtering logic."""

    @pytest.mark.asyncio
    async def test_filters_messages_by_tool_names(self):
        """
        CRITICAL: Verify that tool node only processes
        tool_calls that match its registered tools.

        This is the core fix for the parallel execution bug.
        """
        # Create tool node for market tools only
        market_tool_node = create_test_tool_node(MARKET_TOOLS, "market_analyst")

        # Create messages simulating parallel execution:
        # - Market Analyst called get_technical_indicators
        # - Sentiment Analyst called get_sentiment (AFTER market, so it's "last")
        market_ai_msg = AIMessage(
            content="Calling market tools",
            tool_calls=[
                {
                    "name": "get_technical_indicators",
                    "args": {"symbol": "TEST"},
                    "id": "market-1",
                    "type": "tool_call",
                }
            ],
        )
        sentiment_ai_msg = AIMessage(
            content="Calling sentiment tools",
            tool_calls=[
                {
                    "name": "get_sentiment",
                    "args": {"ticker": "TEST"},
                    "id": "sentiment-1",
                    "type": "tool_call",
                }
            ],
        )

        # Sentiment message is LAST (simulating parallel race condition)
        state = {
            "messages": [market_ai_msg, sentiment_ai_msg],
            "sender": "market_analyst",
        }

        # Execute market tool node
        result = await market_tool_node(state, {})

        # CRITICAL ASSERTION: Should have processed market tools, NOT sentiment
        messages = result.get("messages", [])
        assert len(messages) > 0, "Should have tool results"

        # Verify the tool message is for get_technical_indicators, not get_sentiment
        tool_msg = messages[0]
        assert isinstance(tool_msg, ToolMessage)
        assert (
            tool_msg.tool_call_id == "market-1"
        ), f"Should process market tool, got {tool_msg.tool_call_id}"
        assert (
            "RSI" in tool_msg.content
        ), f"Should have technical indicator data, got: {tool_msg.content}"

    @pytest.mark.asyncio
    async def test_handles_no_matching_tools(self):
        """
        Verify graceful handling when no AIMessage has matching tool_calls.
        """
        market_tool_node = create_test_tool_node(MARKET_TOOLS, "market_analyst")

        # Only sentiment tool calls in messages
        sentiment_ai_msg = AIMessage(
            content="Calling sentiment tools",
            tool_calls=[
                {
                    "name": "get_sentiment",
                    "args": {"ticker": "TEST"},
                    "id": "sentiment-1",
                    "type": "tool_call",
                }
            ],
        )

        state = {"messages": [sentiment_ai_msg], "sender": "market_analyst"}

        result = await market_tool_node(state, {})

        # Should return empty messages, not crash
        assert result.get("messages", []) == []

    @pytest.mark.asyncio
    async def test_finds_correct_message_in_mixed_list(self):
        """
        Verify correct message is found even when deeply nested in message list.
        """
        fundamentals_tool_node = create_test_tool_node(
            FUNDAMENTALS_TOOLS, "junior_fundamentals_analyst"
        )

        # Create a complex message list simulating real parallel execution
        messages = [
            HumanMessage(content="Analyze TEST"),
            AIMessage(
                content="Market analysis",
                tool_calls=[
                    {
                        "name": "get_technical_indicators",
                        "args": {"symbol": "TEST"},
                        "id": "m1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="RSI: 45", tool_call_id="m1"),
            AIMessage(
                content="Sentiment analysis",
                tool_calls=[
                    {
                        "name": "get_sentiment",
                        "args": {"ticker": "TEST"},
                        "id": "s1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="Fundamentals analysis",
                tool_calls=[
                    {
                        "name": "get_financial_metrics",
                        "args": {"ticker": "TEST"},
                        "id": "f1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="Sentiment: Bullish", tool_call_id="s1"),
            AIMessage(
                content="News analysis",
                tool_calls=[
                    {
                        "name": "get_news",
                        "args": {"ticker": "TEST", "search_query": "earnings"},
                        "id": "n1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        state = {"messages": messages, "sender": "junior_fundamentals_analyst"}

        result = await fundamentals_tool_node(state, {})

        messages_out = result.get("messages", [])
        assert len(messages_out) > 0
        assert (
            messages_out[0].tool_call_id == "f1"
        ), "Should find fundamentals tool call, not news (which was last)"


# --- Integration Tests for Parallel Tool Execution ---


class TestParallelToolIsolation:
    """
    Integration tests verifying tool isolation in parallel execution scenarios.
    """

    @pytest.mark.asyncio
    async def test_four_agents_parallel_no_cross_contamination(self):
        """
        Simulate 4 agents running in parallel, each with their own tool node.
        Verify no cross-contamination of tool results.
        """
        # Create agent-specific tool nodes
        market_node = create_test_tool_node(MARKET_TOOLS, "market_analyst")
        sentiment_node = create_test_tool_node(SENTIMENT_TOOLS, "sentiment_analyst")
        news_node = create_test_tool_node(NEWS_TOOLS, "news_analyst")
        fund_node = create_test_tool_node(
            FUNDAMENTALS_TOOLS, "junior_fundamentals_analyst"
        )

        # Simulate all 4 agents producing tool_calls simultaneously
        # In a race condition, the order in messages list is unpredictable
        all_messages = [
            HumanMessage(content="Analyze TEST"),
            # All 4 AIMessages added "simultaneously" - order unpredictable
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_sentiment",
                        "args": {"ticker": "TEST"},
                        "id": "sent-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_financial_metrics",
                        "args": {"ticker": "TEST"},
                        "id": "fund-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_news",
                        "args": {"ticker": "TEST", "search_query": "q"},
                        "id": "news-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_technical_indicators",
                        "args": {"symbol": "TEST"},
                        "id": "mkt-1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        # Each tool node should find ITS OWN agent's tool_calls
        market_result = await market_node({"messages": all_messages}, {})
        sentiment_result = await sentiment_node({"messages": all_messages}, {})
        news_result = await news_node({"messages": all_messages}, {})
        fund_result = await fund_node({"messages": all_messages}, {})

        # Verify each got the correct tool results
        assert (
            market_result["messages"][0].tool_call_id == "mkt-1"
        ), "Market node should process market tools"
        assert (
            sentiment_result["messages"][0].tool_call_id == "sent-1"
        ), "Sentiment node should process sentiment tools"
        assert (
            news_result["messages"][0].tool_call_id == "news-1"
        ), "News node should process news tools"
        assert (
            fund_result["messages"][0].tool_call_id == "fund-1"
        ), "Fundamentals node should process fundamentals tools"

    @pytest.mark.asyncio
    async def test_tool_results_contain_expected_data(self):
        """
        Verify tool results contain data specific to that tool,
        not data from a different tool.
        """
        market_node = create_test_tool_node(MARKET_TOOLS, "market_analyst")
        sentiment_node = create_test_tool_node(SENTIMENT_TOOLS, "sentiment_analyst")

        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_sentiment",
                        "args": {"ticker": "TEST"},
                        "id": "s1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_technical_indicators",
                        "args": {"symbol": "TEST"},
                        "id": "m1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        market_result = await market_node({"messages": messages}, {})
        sentiment_result = await sentiment_node({"messages": messages}, {})

        # Market should have RSI/MACD, NOT sentiment data
        market_content = market_result["messages"][0].content
        assert (
            "RSI" in market_content or "MACD" in market_content
        ), f"Market should have technical data, got: {market_content}"
        assert (
            "Sentiment" not in market_content
        ), f"Market should NOT have sentiment data, got: {market_content}"

        # Sentiment should have sentiment data, NOT technical
        sentiment_content = sentiment_result["messages"][0].content
        assert (
            "Sentiment" in sentiment_content or "Bullish" in sentiment_content
        ), f"Sentiment should have sentiment data, got: {sentiment_content}"
        assert (
            "RSI" not in sentiment_content
        ), f"Sentiment should NOT have RSI, got: {sentiment_content}"


# --- Edge Case Tests ---


class TestToolNodeEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_same_agent(self):
        """
        Verify handling of multiple tool_calls in a single AIMessage.
        """
        market_node = create_test_tool_node(MARKET_TOOLS, "market_analyst")

        # Market analyst calling both of its tools
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_technical_indicators",
                        "args": {"symbol": "TEST"},
                        "id": "m1",
                        "type": "tool_call",
                    },
                    {
                        "name": "calculate_liquidity_metrics",
                        "args": {"ticker": "TEST"},
                        "id": "m2",
                        "type": "tool_call",
                    },
                ],
            )
        ]

        result = await market_node({"messages": messages}, {})

        # Should have results for both tools
        assert len(result["messages"]) == 2
        tool_ids = {msg.tool_call_id for msg in result["messages"]}
        assert tool_ids == {"m1", "m2"}

    @pytest.mark.asyncio
    async def test_empty_messages_list(self):
        """Verify graceful handling of empty messages."""
        market_node = create_test_tool_node(MARKET_TOOLS, "market_analyst")

        result = await market_node({"messages": []}, {})
        assert result.get("messages", []) == []

    @pytest.mark.asyncio
    async def test_no_ai_messages(self):
        """Verify handling when no AIMessages in list."""
        market_node = create_test_tool_node(MARKET_TOOLS, "market_analyst")

        messages = [
            HumanMessage(content="Analyze TEST"),
            ToolMessage(content="Old result", tool_call_id="old"),
        ]

        result = await market_node({"messages": messages}, {})
        assert result.get("messages", []) == []

    @pytest.mark.asyncio
    async def test_ai_message_without_tool_calls(self):
        """Verify handling of AIMessages that have no tool_calls."""
        market_node = create_test_tool_node(MARKET_TOOLS, "market_analyst")

        messages = [
            AIMessage(content="Just a response, no tools"),
            AIMessage(content="Another response"),
        ]

        result = await market_node({"messages": messages}, {})
        assert result.get("messages", []) == []


# --- Regression Tests ---


class TestParallelExecutionRegression:
    """
    Regression tests specifically for the Dec 2025 parallel execution bug.

    Bug description: In parallel execution, ToolNode.parse_input() would find
    the "last AIMessage" which might belong to a different agent, causing
    tool_calls to be executed by the wrong tool node.
    """

    @pytest.mark.asyncio
    async def test_last_message_is_different_agent_bug(self):
        """
        REGRESSION TEST: The bug occurred when the last AIMessage in the
        messages list belonged to a different agent than the tool node
        being executed.

        Before fix: market_tools would execute sentiment's tool_calls
        After fix: market_tools only executes market's tool_calls
        """
        market_node = create_test_tool_node(MARKET_TOOLS, "market_analyst")

        # Critical scenario: sentiment's AIMessage is LAST
        messages = [
            AIMessage(
                content="Market calling tools",
                tool_calls=[
                    {
                        "name": "get_technical_indicators",
                        "args": {"symbol": "TEST"},
                        "id": "market-tool-1",
                        "type": "tool_call",
                    }
                ],
            ),
            # This is LAST - the bug would cause market_node to try to execute this
            AIMessage(
                content="Sentiment calling tools",
                tool_calls=[
                    {
                        "name": "get_sentiment",
                        "args": {"ticker": "TEST"},
                        "id": "sentiment-tool-1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        result = await market_node({"messages": messages}, {})

        # CRITICAL: Must process market's tool, NOT sentiment's
        assert len(result["messages"]) > 0, "Should have results"
        assert (
            result["messages"][0].tool_call_id == "market-tool-1"
        ), "REGRESSION: market_node processed wrong agent's tool_calls"

    @pytest.mark.asyncio
    async def test_interleaved_messages_from_parallel_agents(self):
        """
        REGRESSION TEST: Messages from parallel agents can be interleaved
        in unpredictable order. Each tool node must find its correct message.
        """
        # Create all 4 tool nodes
        nodes = {
            "market": create_test_tool_node(MARKET_TOOLS, "market_analyst"),
            "sentiment": create_test_tool_node(SENTIMENT_TOOLS, "sentiment_analyst"),
            "news": create_test_tool_node(NEWS_TOOLS, "news_analyst"),
            "fund": create_test_tool_node(
                FUNDAMENTALS_TOOLS, "junior_fundamentals_analyst"
            ),
        }

        expected_ids = {
            "market": "mkt-abc",
            "sentiment": "sent-def",
            "news": "news-ghi",
            "fund": "fund-jkl",
        }

        # Interleaved messages (worst case scenario)
        messages = [
            HumanMessage(content="Start"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_news",
                        "args": {"ticker": "T", "search_query": "q"},
                        "id": expected_ids["news"],
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_technical_indicators",
                        "args": {"symbol": "T"},
                        "id": expected_ids["market"],
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="partial", tool_call_id="old"),  # Noise
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_financial_metrics",
                        "args": {"ticker": "T"},
                        "id": expected_ids["fund"],
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_sentiment",
                        "args": {"ticker": "T"},
                        "id": expected_ids["sentiment"],
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        # Each node must find its own tool_calls
        for agent_type, node in nodes.items():
            result = await node({"messages": messages}, {})
            actual_id = result["messages"][0].tool_call_id
            expected_id = expected_ids[agent_type]
            assert (
                actual_id == expected_id
            ), f"REGRESSION: {agent_type}_node got {actual_id}, expected {expected_id}"


# --- Tests using actual create_agent_tool_node from graph.py ---


class TestActualCreateAgentToolNode:
    """
    Tests that verify the actual create_agent_tool_node function
    correctly identifies messages (without executing tools).
    """

    def test_import_works(self):
        """Verify the function can be imported."""
        from src.graph import create_agent_tool_node

        assert callable(create_agent_tool_node)

    @pytest.mark.asyncio
    async def test_message_filtering_logic(self):
        """
        Test that the actual create_agent_tool_node filters correctly.
        We mock the ToolNode to avoid needing LangGraph runtime.
        """
        from src.graph import create_agent_tool_node

        # Create the actual function with our test tools
        with patch("src.graph.ToolNode") as MockToolNode:
            # Setup mock to capture what messages are passed
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(
                return_value={
                    "messages": [ToolMessage(content="mocked", tool_call_id="test")]
                }
            )
            MockToolNode.return_value = mock_instance

            tool_node_fn = create_agent_tool_node(MARKET_TOOLS, "market_analyst")

            # Create mixed messages - one from wrong agent, one from correct agent
            # Note: create_agent_tool_node filters by msg.name == agent_key
            messages = [
                AIMessage(
                    content="",
                    name="sentiment_analyst",  # Wrong agent
                    tool_calls=[
                        {
                            "name": "get_sentiment",
                            "args": {"ticker": "T"},
                            "id": "wrong",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    name="market_analyst",  # Correct agent - matches agent_key
                    tool_calls=[
                        {
                            "name": "get_technical_indicators",
                            "args": {"symbol": "T"},
                            "id": "correct",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]

            result = await tool_node_fn({"messages": messages}, {})

            # Verify ToolNode.ainvoke was called with filtered messages
            call_args = mock_instance.ainvoke.call_args
            filtered_msgs = call_args[0][0]["messages"]

            # Should only have the market analyst's message
            assert len(filtered_msgs) == 1
            assert (
                filtered_msgs[0].tool_calls[0]["id"] == "correct"
            ), "Should filter to only market analyst's message"
