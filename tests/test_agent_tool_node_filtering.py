"""
Tests for agent tool node filtering to prevent cross-agent tool execution.

Critical Bug Context (Jan 2026):
When multiple agents call the same tool (e.g., get_news), the create_agent_tool_node
function could pick up the WRONG agent's AIMessage if it only checked whether any
tool_call matched the tool names. This caused:
1. Agent A's tool node executing Agent B's tool_calls
2. Missing ToolMessages for the correct agent
3. Empty LLM responses when agent didn't receive its own tool results

The fix: Check `msg.name == agent_key` in addition to tool name intersection.
These tests ensure this bug doesn't regress.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph import create_agent_tool_node


class TestAgentToolNodeFiltering:
    """Tests for create_agent_tool_node's agent filtering logic."""

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools with names."""
        tool1 = MagicMock()
        tool1.name = "get_news"
        tool2 = MagicMock()
        tool2.name = "get_macroeconomic_news"
        return [tool1, tool2]

    @pytest.fixture
    def mock_tool_node(self):
        """Mock the ToolNode to return predictable results."""
        with patch("src.graph.ToolNode") as mock:
            mock_instance = AsyncMock()
            mock_instance.ainvoke = AsyncMock(
                return_value={
                    "messages": [
                        ToolMessage(
                            content="result1", tool_call_id="1", name="get_news"
                        ),
                        ToolMessage(
                            content="result2",
                            tool_call_id="2",
                            name="get_macroeconomic_news",
                        ),
                    ]
                }
            )
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_filters_by_agent_key_not_just_tool_name(
        self, mock_tools, mock_tool_node
    ):
        """
        CRITICAL TEST: Verify tool node only picks up AIMessages from its own agent.

        Scenario:
        - news_analyst makes tool_calls for get_news
        - value_trap_detector ALSO makes tool_calls for get_news
        - news_tools (agent_key=news_analyst) should ONLY execute news_analyst's calls

        Before the fix, news_tools would pick up value_trap_detector's AIMessage
        because it contained get_news in its tool_calls.
        """
        agent_tool_node = create_agent_tool_node(mock_tools, "news_analyst")

        # Two different agents both calling get_news
        news_analyst_msg = AIMessage(
            content="",
            name="news_analyst",  # Tagged with agent name
            tool_calls=[
                {
                    "name": "get_news",
                    "args": {"ticker": "ALQ.AX"},
                    "id": "1",
                    "type": "tool_call",
                },
                {
                    "name": "get_macroeconomic_news",
                    "args": {"date": "2026-01-09"},
                    "id": "2",
                    "type": "tool_call",
                },
            ],
        )
        value_trap_msg = AIMessage(
            content="",
            name="value_trap_detector",  # Different agent
            tool_calls=[
                {
                    "name": "get_news",
                    "args": {"ticker": "ALQ.AX", "query": "governance"},
                    "id": "3",
                    "type": "tool_call",
                },
                {
                    "name": "get_ownership_structure",
                    "args": {"ticker": "ALQ.AX"},
                    "id": "4",
                    "type": "tool_call",
                },
            ],
        )

        # State with both agents' messages (value_trap most recent)
        state = {
            "messages": [
                HumanMessage(content="Analyze ALQ.AX"),
                news_analyst_msg,
                value_trap_msg,  # Most recent - wrong agent
            ]
        }

        config = {"configurable": {}}
        result = await agent_tool_node(state, config)

        # Should have called ToolNode with news_analyst's message, not value_trap's
        mock_tool_node.ainvoke.assert_called_once()
        call_args = mock_tool_node.ainvoke.call_args[0][0]
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0].name == "news_analyst"
        assert len(call_args["messages"][0].tool_calls) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_matching_agent_message(
        self, mock_tools, mock_tool_node
    ):
        """
        Test that tool node returns empty when no AIMessage matches agent_key.

        This can happen if:
        1. Agent message was lost in state
        2. Agent message wasn't tagged with name
        """
        agent_tool_node = create_agent_tool_node(mock_tools, "news_analyst")

        # Only value_trap_detector's message, no news_analyst
        state = {
            "messages": [
                HumanMessage(content="Analyze"),
                AIMessage(
                    content="",
                    name="value_trap_detector",
                    tool_calls=[
                        {"name": "get_news", "args": {}, "id": "1", "type": "tool_call"}
                    ],
                ),
            ]
        }

        config = {"configurable": {}}
        result = await agent_tool_node(state, config)

        # Should return empty, not execute wrong agent's tools
        assert result == {"messages": []}
        mock_tool_node.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_finds_correct_agent_message_among_many(
        self, mock_tools, mock_tool_node
    ):
        """Test that correct AIMessage is found among multiple agents' messages."""
        agent_tool_node = create_agent_tool_node(mock_tools, "news_analyst")

        # Multiple agents' messages interleaved
        state = {
            "messages": [
                HumanMessage(content="Analyze"),
                AIMessage(name="market_analyst", content="", tool_calls=[]),
                AIMessage(
                    name="news_analyst",
                    content="",
                    tool_calls=[
                        {"name": "get_news", "args": {}, "id": "1", "type": "tool_call"}
                    ],
                ),
                AIMessage(name="sentiment_analyst", content="", tool_calls=[]),
                AIMessage(
                    name="value_trap_detector",
                    content="",
                    tool_calls=[
                        {"name": "get_news", "args": {}, "id": "2", "type": "tool_call"}
                    ],
                ),
            ]
        }

        config = {"configurable": {}}
        result = await agent_tool_node(state, config)

        # Should find and use news_analyst's message (searching backwards)
        mock_tool_node.ainvoke.assert_called_once()
        call_args = mock_tool_node.ainvoke.call_args[0][0]
        assert call_args["messages"][0].name == "news_analyst"

    @pytest.mark.asyncio
    async def test_tags_tool_messages_with_agent_key(self, mock_tools, mock_tool_node):
        """Verify ToolMessages are tagged with agent_key for filtering."""
        agent_tool_node = create_agent_tool_node(mock_tools, "news_analyst")

        state = {
            "messages": [
                HumanMessage(content="Analyze"),
                AIMessage(
                    name="news_analyst",
                    content="",
                    tool_calls=[
                        {"name": "get_news", "args": {}, "id": "1", "type": "tool_call"}
                    ],
                ),
            ]
        }

        config = {"configurable": {}}
        result = await agent_tool_node(state, config)

        # All ToolMessages should be tagged with agent_key
        for msg in result["messages"]:
            assert isinstance(msg, ToolMessage)
            assert msg.additional_kwargs.get("agent_key") == "news_analyst"


class TestCrossAgentToolUsage:
    """
    Tests ensuring tools can be safely used by multiple agents.

    These tests verify that tools like get_news, which are in multiple
    agent tool lists, work correctly without cross-contamination.
    """

    @pytest.mark.asyncio
    async def test_same_tool_different_agents_isolated(self):
        """
        Test that same tool (get_news) used by different agents stays isolated.

        This is the exact scenario that caused the Jan 2026 bug:
        - news_analyst uses get_news_tools() which includes get_news
        - value_trap_detector uses get_value_trap_tools() which includes get_news

        When agents share tools, the tool node must correctly identify which
        agent's AIMessage to process based on msg.name, not just tool names.
        """
        from src.toolkit import toolkit

        # Verify get_news is in multiple tool lists
        news_tools = toolkit.get_news_tools()
        value_trap_tools = toolkit.get_value_trap_tools()

        news_tool_names = {t.name for t in news_tools}
        value_trap_tool_names = {t.name for t in value_trap_tools}

        # get_news should be shared (this is the condition that caused the bug)
        # If this assertion fails, the test setup is invalid
        shared_tool = "get_news"
        assert shared_tool in news_tool_names, f"{shared_tool} not in news_tools"
        assert (
            shared_tool in value_trap_tool_names
        ), f"{shared_tool} not in value_trap_tools"

        # This confirms the bug scenario: two different agents use the same tool
        # The fix ensures each agent's tool node only processes its own AIMessages

    def test_each_agent_has_unique_tool_node(self):
        """Verify each agent gets its own tool node instance."""
        from src.toolkit import toolkit

        # Create tool nodes for agents that share tools
        news_node = create_agent_tool_node(toolkit.get_news_tools(), "news_analyst")
        value_trap_node = create_agent_tool_node(
            toolkit.get_value_trap_tools(), "value_trap_detector"
        )
        foreign_node = create_agent_tool_node(
            toolkit.get_foreign_language_tools(), "foreign_language_analyst"
        )

        # Each should be a distinct function
        assert news_node is not value_trap_node
        assert news_node is not foreign_node
        assert value_trap_node is not foreign_node


class TestToolMessageFiltering:
    """Tests for filter_messages_by_agent correctly filtering ToolMessages."""

    def test_filters_tool_messages_by_agent_key(self):
        """Verify ToolMessages are filtered by agent_key tag."""
        from src.agents import filter_messages_by_agent

        messages = [
            HumanMessage(content="Start"),
            AIMessage(name="news_analyst", content="", tool_calls=[]),
            ToolMessage(
                content="News result",
                tool_call_id="1",
                name="get_news",
                additional_kwargs={"agent_key": "news_analyst"},
            ),
            ToolMessage(
                content="Value trap result",
                tool_call_id="2",
                name="get_news",  # Same tool name!
                additional_kwargs={
                    "agent_key": "value_trap_detector"
                },  # Different agent
            ),
        ]

        # Filter for news_analyst
        filtered = filter_messages_by_agent(messages, "news_analyst")

        # Should only get news_analyst's ToolMessage, not value_trap's
        tool_msgs = [m for m in filtered if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].additional_kwargs["agent_key"] == "news_analyst"

    def test_untagged_tool_messages_are_excluded(self):
        """ToolMessages without agent_key tag should be excluded."""
        from src.agents import filter_messages_by_agent

        messages = [
            HumanMessage(content="Start"),
            AIMessage(name="news_analyst", content="", tool_calls=[]),
            ToolMessage(
                content="Untagged result",
                tool_call_id="1",
                name="get_news",
                # No agent_key in additional_kwargs
            ),
        ]

        filtered = filter_messages_by_agent(messages, "news_analyst")

        # Untagged ToolMessage should be excluded
        tool_msgs = [m for m in filtered if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 0


class TestAllAgentToolCombinations:
    """
    Test that ALL agent/tool combinations work correctly.

    This ensures the fix applies to every agent that uses tools.
    """

    @pytest.fixture
    def agent_tool_configs(self):
        """Get all agent/tool configurations from the actual graph."""
        from src.toolkit import toolkit

        return [
            ("market_analyst", toolkit.get_market_tools()),
            ("sentiment_analyst", toolkit.get_sentiment_tools()),
            ("news_analyst", toolkit.get_news_tools()),
            ("junior_fundamentals_analyst", toolkit.get_junior_fundamental_tools()),
            ("foreign_language_analyst", toolkit.get_foreign_language_tools()),
            ("legal_counsel", toolkit.get_legal_tools()),
            ("value_trap_detector", toolkit.get_value_trap_tools()),
        ]

    @pytest.mark.asyncio
    async def test_each_agent_tool_node_requires_correct_agent_name(
        self, agent_tool_configs
    ):
        """Each agent's tool node should only accept its own AIMessages."""
        for agent_key, tools in agent_tool_configs:
            if not tools:  # Skip agents without tools
                continue

            with patch("src.graph.ToolNode") as mock_tn:
                mock_instance = AsyncMock()
                mock_instance.ainvoke = AsyncMock(return_value={"messages": []})
                mock_tn.return_value = mock_instance

                tool_node = create_agent_tool_node(tools, agent_key)

                # Create AIMessage tagged with a DIFFERENT agent
                wrong_agent = "some_other_agent"
                state = {
                    "messages": [
                        AIMessage(
                            name=wrong_agent,  # Wrong agent!
                            content="",
                            tool_calls=[
                                {
                                    "name": tools[0].name,
                                    "args": {},
                                    "id": "1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                }

                result = await tool_node(state, {"configurable": {}})

                # Should return empty, not execute wrong agent's tools
                assert result == {
                    "messages": []
                }, f"{agent_key} tool node accepted wrong agent's message"
                mock_instance.ainvoke.assert_not_called()
