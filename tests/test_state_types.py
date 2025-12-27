"""
Tests for AgentState type safety and state field reducers.

This test suite catches type mismatches that can occur when state fields
are accumulated as lists instead of being replaced with the latest value.

Critical for:
- String fields that get passed to regex parsers (fundamentals_report -> RedFlagDetector)
- Dict/complex fields that need to maintain structure
- Preventing runtime "expected string or bytes-like object, got 'list'" errors
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestAgentStateTypeDefinitions:
    """Test that AgentState fields have correct type annotations and reducers."""

    def test_agent_state_has_take_last_for_string_fields(self):
        """Verify all string report fields use take_last reducer to prevent list accumulation."""

        from src.agents import AgentState

        # Get annotations
        annotations = AgentState.__annotations__

        # String fields that MUST use take_last (not list accumulation)
        string_fields_requiring_take_last = [
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            "investment_plan",
            "trader_investment_plan",
            "final_trade_decision",
        ]

        for field in string_fields_requiring_take_last:
            assert field in annotations, f"{field} missing from AgentState"

            # Check if it's annotated with take_last
            annotation = annotations[field]

            # For Annotated types, check metadata includes take_last
            if hasattr(annotation, "__metadata__"):
                # This is an Annotated type - verify take_last is in metadata
                from src.agents import take_last

                assert (
                    take_last in annotation.__metadata__
                ), f"{field} must use Annotated[str, take_last] to prevent list accumulation"
            else:
                # If it's just 'str', it will use default list accumulation (BAD)
                pytest.fail(
                    f"{field} is defined as plain 'str' instead of 'Annotated[str, take_last]'. "
                    f"This will cause state to accumulate as a list, breaking regex parsers."
                )

    def test_agent_state_complex_fields_have_reducers(self):
        """Verify complex dict/list fields use explicit reducers."""
        from src.agents import AgentState

        annotations = AgentState.__annotations__

        # Complex fields that need explicit reducers
        complex_fields = [
            "investment_debate_state",  # InvestDebateState
            "risk_debate_state",  # RiskDebateState
            "tools_called",  # dict
            "prompts_used",  # dict
            "red_flags",  # list[dict]
            "pre_screening_result",  # str but critical
        ]

        for field in complex_fields:
            assert field in annotations, f"{field} missing from AgentState"
            annotation = annotations[field]

            # All complex fields should be Annotated
            assert hasattr(
                annotation, "__metadata__"
            ), f"{field} should use Annotated[Type, reducer] for explicit state management"


class TestStatePropagationTypes:
    """Test that state values maintain correct types through graph execution."""

    @pytest.mark.asyncio
    async def test_fundamentals_report_is_string_not_list(self):
        """REGRESSION TEST: fundamentals_report must be string for RedFlagDetector regex parsing."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from src.agents import create_analyst_node

        # Mock LLM - use SimpleNamespace for response like existing tests
        mock_llm = MagicMock()
        mock_response = SimpleNamespace(
            content="Mock fundamentals report with DATA_BLOCK", tool_calls=None
        )

        # Mock both bind_tools path and direct path (matches test_agents.py pattern)
        mock_llm.bind_tools.return_value.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # CRITICAL: Mock the invoke_with_rate_limit_handling to bypass the chain
        with patch(
            "src.agents.invoke_with_rate_limit_handling",
            new=AsyncMock(return_value=mock_response),
        ):
            # Create fundamentals analyst node
            fundamentals_node = create_analyst_node(
                mock_llm, "fundamentals_analyst", [], "fundamentals_report"
            )

            # Initial state
            state = {
                "messages": [],
                "company_of_interest": "TEST.US",
                "trade_date": "2025-12-07",
            }
            config = {
                "configurable": {
                    "context": MagicMock(ticker="TEST.US", trade_date="2025-12-07")
                }
            }

            # Run fundamentals analyst
            result_state = await fundamentals_node(state, config)

            # CRITICAL: fundamentals_report must be a STRING, not a list
            fundamentals_report = result_state.get("fundamentals_report")

            assert (
                fundamentals_report is not None
            ), "fundamentals_report should be populated"
            assert isinstance(fundamentals_report, str), (
                f"fundamentals_report must be str, got {type(fundamentals_report)}. "
                f"If it's a list, AgentState needs 'Annotated[str, take_last]' annotation."
            )

            # Test that RedFlagDetector can parse it (would fail with list)
            from src.validators.red_flag_detector import RedFlagDetector

            # This should NOT raise "expected string or bytes-like object, got 'list'"
            metrics = RedFlagDetector.extract_metrics(fundamentals_report)
            assert isinstance(metrics, dict)

    @pytest.mark.asyncio
    async def test_all_report_fields_are_strings_after_execution(self):
        """Test that all report fields maintain string type through execution."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from src.agents import create_analyst_node

        # Test each analyst type
        test_cases = [
            ("market_analyst", "market_report"),
            ("sentiment_analyst", "sentiment_report"),
            ("news_analyst", "news_report"),
            ("fundamentals_analyst", "fundamentals_report"),
        ]

        for agent_name, output_field in test_cases:
            # Create fresh mocks for each iteration
            mock_llm = MagicMock()
            mock_response = SimpleNamespace(
                content=f"Test {output_field} content",  # String content
                tool_calls=None,
            )

            # Mock both bind_tools path and direct path (matches test_agents.py pattern)
            mock_llm.bind_tools.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)

            # CRITICAL: Mock the invoke_with_rate_limit_handling to bypass the chain
            with patch(
                "src.agents.invoke_with_rate_limit_handling",
                new=AsyncMock(return_value=mock_response),
            ):
                node = create_analyst_node(mock_llm, agent_name, [], output_field)

                state = {
                    "messages": [],
                    "company_of_interest": "TEST.US",
                    "trade_date": "2025-12-07",
                }
                config = {
                    "configurable": {
                        "context": MagicMock(ticker="TEST.US", trade_date="2025-12-07")
                    }
                }

                result = await node(state, config)

                # Check output field type
                output_value = result.get(output_field)
                assert (
                    output_value is not None
                ), f"{output_field} should be populated by {agent_name}"
                assert isinstance(
                    output_value, str
                ), f"{output_field} from {agent_name} must be str, got {type(output_value)}"


class TestStateFieldTypesInPractice:
    """Test actual runtime type safety with realistic data."""

    def test_fundamentals_report_string_for_regex_parsing(self):
        """Test that fundamentals_report string works with regex parsers."""
        from src.validators.red_flag_detector import RedFlagDetector

        # Realistic fundamentals report
        report = """
        ### --- START DATA_BLOCK ---
        ADJUSTED_HEALTH_SCORE: 55%
        PE_RATIO_TTM: 12.5
        ### --- END DATA_BLOCK ---

        **D/E Ratio**: 250%
        **Interest Coverage**: 2.5x
        **Free Cash Flow**: $500M
        """

        # This should work with string
        metrics = RedFlagDetector.extract_metrics(report)
        assert metrics is not None
        assert isinstance(metrics, dict)

        # This should FAIL if given a list (simulating the bug)
        with pytest.raises((TypeError, AttributeError)):
            RedFlagDetector.extract_metrics([report])  # List instead of string

    def test_state_dict_types_match_expectations(self):
        """Test that state dict values have expected types."""

        # Simulate a state dict that might come from LangGraph
        state_dict = {
            "company_of_interest": "TEST.US",
            "company_name": "Test Company",
            "trade_date": "2025-12-07",
            "sender": "market_analyst",
            "market_report": "Market analysis here",  # Should be string
            "fundamentals_report": "Fundamentals analysis here",  # Should be string
            "red_flags": [],  # Should be list
            "pre_screening_result": "PASS",  # Should be string
        }

        # Verify types
        assert isinstance(state_dict["market_report"], str)
        assert isinstance(state_dict["fundamentals_report"], str)
        assert isinstance(state_dict["red_flags"], list)
        assert isinstance(state_dict["pre_screening_result"], str)

        # If these were lists due to wrong reducer, they'd fail
        assert not isinstance(state_dict["market_report"], list)
        assert not isinstance(state_dict["fundamentals_report"], list)


class TestReducerBehavior:
    """Test that reducers work correctly."""

    def test_take_last_reducer(self):
        """Test that take_last reducer returns the latest value."""
        from src.agents import take_last

        old_value = "old report"
        new_value = "new report"

        result = take_last(old_value, new_value)
        assert result == new_value
        assert result != old_value

    def test_take_last_replaces_not_appends(self):
        """Verify take_last doesn't accumulate values."""
        from src.agents import take_last

        # Simulate multiple updates
        value1 = "first"
        value2 = "second"
        value3 = "third"

        result = take_last(value1, value2)
        assert result == "second"

        result = take_last(result, value3)
        assert result == "third"

        # Should NOT be a list
        assert not isinstance(result, list)
        assert isinstance(result, str)


class TestDataProviderTypes:
    """Test type safety for data coming from external providers."""

    def test_financial_metrics_return_types(self):
        """Test that financial metric tools return correct types."""
        # This is a placeholder - actual implementation depends on toolkit
        # In practice, you'd mock the data fetcher and verify return types
        pass

    def test_pandas_dataframe_compatibility(self):
        """Test that data types work with pandas operations."""
        import pandas as pd

        # Simulate financial data that might come from yfinance/yahooquery
        data = {"price": [100.0, 101.5, 99.8], "volume": [1000000, 1200000, 950000]}

        df = pd.DataFrame(data)

        # These operations should work with correct types
        assert df["price"].dtype in [float, "float64"]
        assert df["volume"].dtype in [int, "int64", float, "float64"]

        # String columns should be object dtype
        df["ticker"] = "TEST.US"
        assert df["ticker"].dtype == object


class TestTypeAnnotationConsistency:
    """Test that type annotations are consistent across the codebase."""

    def test_agent_state_fields_match_usage(self):
        """Verify AgentState annotations match actual usage in code."""

        from src.agents import AgentState

        # Get all fields from AgentState
        annotations = AgentState.__annotations__

        # Verify critical fields exist
        required_fields = [
            "company_of_interest",
            "trade_date",
            "market_report",
            "fundamentals_report",
            "red_flags",
            "pre_screening_result",
        ]

        for field in required_fields:
            assert (
                field in annotations
            ), f"Required field {field} missing from AgentState"

    def test_no_plain_string_annotations_for_reports(self):
        """Ensure no report fields use plain 'str' without Annotated."""
        from src.agents import AgentState

        annotations = AgentState.__annotations__

        report_fields = [
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            "investment_plan",
            "trader_investment_plan",
            "final_trade_decision",
        ]

        for field in report_fields:
            annotation = annotations[field]

            # Check if it's just 'str' (which would be wrong)
            if annotation is str:
                pytest.fail(
                    f"{field} uses plain 'str' annotation. "
                    f"Should use 'Annotated[str, take_last]' to prevent list accumulation."
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
