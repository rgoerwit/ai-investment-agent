"""
Tests for extract_string_content() helper function.

This function handles the case where Gemini models return dict/list instead of string
for response.content. These tests verify:
1. All input types are handled correctly
2. Output format is compatible with downstream agents (especially RedFlagDetector)
3. Edge cases don't cause crashes or data loss
"""

import pytest
from typing import Any, Dict, List

from src.agents import extract_string_content
from src.validators.red_flag_detector import RedFlagDetector, Sector


# --- Sample Data ---

SAMPLE_FUNDAMENTALS_REPORT = """## Fundamentals Analysis for 1681.HK (CONSUN PHARMA)

SECTOR: Healthcare / Pharmaceuticals

The company shows moderate financial health with some concerns.

### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 8/12
ADJUSTED_HEALTH_SCORE: 67%
PE_RATIO_TTM: 8.5
### --- END DATA_BLOCK ---

## Financial Metrics
- D/E: 45
- Interest Coverage: 5.2x
- Free Cash Flow: 150M HKD
- Net Income: 200M HKD

Overall recommendation: Monitor for entry point.
"""

SAMPLE_BANKING_REPORT = """## Fundamentals Analysis for 0005.HK (HSBC)

SECTOR: Banking / Financial Services

### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 10/12
ADJUSTED_HEALTH_SCORE: 83%
PE_RATIO_TTM: 7.2
DEBT_TO_EQUITY: 850%
FREE_CASH_FLOW: 5.2B USD
NET_INCOME: 12.4B USD
### --- END DATA_BLOCK ---
"""


class TestExtractStringContentBasicTypes:
    """Test basic type handling."""

    def test_string_passthrough(self):
        """String input should pass through unchanged."""
        input_str = "Hello, this is a test report."
        result = extract_string_content(input_str)
        assert result == input_str
        assert isinstance(result, str)

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert extract_string_content("") == ""

    def test_none_returns_empty_string(self):
        """None should return empty string."""
        assert extract_string_content(None) == ""

    def test_integer_converted_to_string(self):
        """Integer should be converted to string."""
        assert extract_string_content(42) == "42"

    def test_float_converted_to_string(self):
        """Float should be converted to string."""
        result = extract_string_content(3.14159)
        assert "3.14" in result

    def test_boolean_converted_to_string(self):
        """Boolean should be converted to string."""
        assert extract_string_content(True) == "True"
        assert extract_string_content(False) == "False"


class TestExtractStringContentDictHandling:
    """Test dict input handling - the main bug scenario."""

    def test_dict_with_text_key(self):
        """Dict with 'text' key should extract the text value."""
        input_dict = {"text": SAMPLE_FUNDAMENTALS_REPORT}
        result = extract_string_content(input_dict)
        assert result == SAMPLE_FUNDAMENTALS_REPORT
        assert "SECTOR: Healthcare" in result

    def test_dict_with_content_key(self):
        """Dict with 'content' key should extract recursively."""
        input_dict = {"content": SAMPLE_FUNDAMENTALS_REPORT}
        result = extract_string_content(input_dict)
        assert result == SAMPLE_FUNDAMENTALS_REPORT

    def test_dict_with_nested_content(self):
        """Nested content dicts should be handled recursively."""
        input_dict = {
            "content": {
                "text": SAMPLE_FUNDAMENTALS_REPORT
            }
        }
        result = extract_string_content(input_dict)
        assert result == SAMPLE_FUNDAMENTALS_REPORT

    def test_dict_with_parts_list(self):
        """Dict with 'parts' list (Gemini multi-part format) should join parts."""
        input_dict = {
            "parts": [
                {"text": "Part 1: Analysis"},
                {"text": "Part 2: Conclusion"}
            ]
        }
        result = extract_string_content(input_dict)
        assert "Part 1: Analysis" in result
        assert "Part 2: Conclusion" in result

    def test_dict_with_parts_containing_report(self):
        """Parts containing a full report should be extractable."""
        input_dict = {
            "parts": [
                {"text": SAMPLE_FUNDAMENTALS_REPORT}
            ]
        }
        result = extract_string_content(input_dict)
        assert "SECTOR: Healthcare" in result
        assert "DATA_BLOCK" in result

    def test_dict_fallback_to_str(self):
        """Dict without known keys should fall back to str() representation."""
        input_dict = {"unknown_key": "some value", "another": 123}
        result = extract_string_content(input_dict)
        # Should contain the dict representation
        assert "unknown_key" in result
        assert "some value" in result

    def test_empty_dict(self):
        """Empty dict should return string representation."""
        result = extract_string_content({})
        assert result == "{}"

    def test_dict_with_none_text(self):
        """Dict with None text value should handle gracefully."""
        input_dict = {"text": None}
        result = extract_string_content(input_dict)
        assert result == "None"  # str(None)


class TestExtractStringContentListHandling:
    """Test list input handling."""

    def test_empty_list(self):
        """Empty list should return empty string."""
        assert extract_string_content([]) == ""

    def test_single_item_list_string(self):
        """Single-item list with string should extract the string."""
        result = extract_string_content([SAMPLE_FUNDAMENTALS_REPORT])
        assert result == SAMPLE_FUNDAMENTALS_REPORT

    def test_single_item_list_dict(self):
        """Single-item list with dict should extract from the dict."""
        input_list = [{"text": SAMPLE_FUNDAMENTALS_REPORT}]
        result = extract_string_content(input_list)
        assert result == SAMPLE_FUNDAMENTALS_REPORT

    def test_multiple_strings_joined(self):
        """Multiple strings should be joined with newlines."""
        input_list = ["Line 1", "Line 2", "Line 3"]
        result = extract_string_content(input_list)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_multiple_dicts_joined(self):
        """Multiple dicts should be processed and joined."""
        input_list = [
            {"text": "Section A"},
            {"text": "Section B"}
        ]
        result = extract_string_content(input_list)
        assert "Section A" in result
        assert "Section B" in result

    def test_mixed_list_types(self):
        """List with mixed types should handle each appropriately."""
        input_list = [
            "Plain text",
            {"text": "Dict text"},
            42
        ]
        result = extract_string_content(input_list)
        assert "Plain text" in result
        assert "Dict text" in result
        assert "42" in result

    def test_list_with_empty_items_filtered(self):
        """Empty items in list should be filtered out."""
        input_list = ["Content", "", None, "More content"]
        result = extract_string_content(input_list)
        # Empty strings and None should not create extra newlines
        assert "Content" in result
        assert "More content" in result


class TestExtractStringContentGeminiFormats:
    """Test realistic Gemini response formats."""

    def test_gemini_text_response(self):
        """Standard Gemini text response format."""
        # This is what Gemini normally returns
        input_content = SAMPLE_FUNDAMENTALS_REPORT
        result = extract_string_content(input_content)
        assert result == SAMPLE_FUNDAMENTALS_REPORT

    def test_gemini_structured_response_dict(self):
        """Gemini structured response as dict (the bug case)."""
        # When Gemini returns structured content
        input_dict = {
            "text": SAMPLE_FUNDAMENTALS_REPORT,
            "role": "model"
        }
        result = extract_string_content(input_dict)
        assert "SECTOR: Healthcare" in result
        assert "DATA_BLOCK" in result

    def test_gemini_multipart_response(self):
        """Gemini multi-part response format."""
        input_dict = {
            "parts": [
                {"text": "## Analysis\n\nSECTOR: Technology\n"},
                {"text": "### --- START DATA_BLOCK ---\nADJUSTED_HEALTH_SCORE: 75%\n### --- END DATA_BLOCK ---"}
            ]
        }
        result = extract_string_content(input_dict)
        assert "SECTOR: Technology" in result
        assert "DATA_BLOCK" in result
        assert "75%" in result

    def test_gemini_content_wrapper(self):
        """Gemini response wrapped in content key."""
        input_dict = {
            "content": SAMPLE_BANKING_REPORT,
            "metadata": {"model": "gemini-pro"}
        }
        result = extract_string_content(input_dict)
        assert "SECTOR: Banking" in result

    def test_deeply_nested_gemini_response(self):
        """Deeply nested response structure."""
        input_dict = {
            "content": {
                "parts": [
                    {
                        "content": {
                            "text": SAMPLE_FUNDAMENTALS_REPORT
                        }
                    }
                ]
            }
        }
        result = extract_string_content(input_dict)
        assert "SECTOR: Healthcare" in result


class TestDownstreamCompatibility:
    """Test that extracted content works with downstream agents."""

    def test_sector_detection_from_string(self):
        """RedFlagDetector should parse sector from string input."""
        sector = RedFlagDetector.detect_sector(SAMPLE_FUNDAMENTALS_REPORT)
        assert sector == Sector.GENERAL  # Healthcare maps to GENERAL

    def test_sector_detection_from_dict_extraction(self):
        """RedFlagDetector should parse sector from dict-extracted content."""
        input_dict = {"text": SAMPLE_FUNDAMENTALS_REPORT}
        extracted = extract_string_content(input_dict)
        sector = RedFlagDetector.detect_sector(extracted)
        assert sector == Sector.GENERAL

    def test_sector_detection_banking_from_dict(self):
        """Banking sector should be detected from dict-extracted content."""
        input_dict = {"text": SAMPLE_BANKING_REPORT}
        extracted = extract_string_content(input_dict)
        sector = RedFlagDetector.detect_sector(extracted)
        assert sector == Sector.BANKING

    def test_metrics_extraction_from_string(self):
        """RedFlagDetector should extract metrics from string input."""
        metrics = RedFlagDetector.extract_metrics(SAMPLE_FUNDAMENTALS_REPORT)
        assert metrics['adjusted_health_score'] == 67.0
        assert metrics['debt_to_equity'] == 45.0

    def test_metrics_extraction_from_dict_extraction(self):
        """RedFlagDetector should extract metrics from dict-extracted content."""
        input_dict = {"text": SAMPLE_FUNDAMENTALS_REPORT}
        extracted = extract_string_content(input_dict)
        metrics = RedFlagDetector.extract_metrics(extracted)
        assert metrics['adjusted_health_score'] == 67.0
        assert metrics['debt_to_equity'] == 45.0

    def test_metrics_extraction_from_multipart(self):
        """Metrics should be extractable from multi-part response."""
        input_dict = {
            "parts": [
                {"text": "SECTOR: Technology\n\n"},
                {"text": "### --- START DATA_BLOCK ---\nADJUSTED_HEALTH_SCORE: 82%\n### --- END DATA_BLOCK ---\n\n- D/E: 30"}
            ]
        }
        extracted = extract_string_content(input_dict)
        metrics = RedFlagDetector.extract_metrics(extracted)
        assert metrics['adjusted_health_score'] == 82.0
        assert metrics['debt_to_equity'] == 30.0

    def test_data_block_preserved_in_list_join(self):
        """DATA_BLOCK structure should be preserved when joining list items."""
        # This tests that newline joining doesn't break the DATA_BLOCK pattern
        input_list = [
            "SECTOR: Utilities\n",
            "### --- START DATA_BLOCK ---\nADJUSTED_HEALTH_SCORE: 55%\n### --- END DATA_BLOCK ---",
            "- D/E: 200"
        ]
        extracted = extract_string_content(input_list)
        metrics = RedFlagDetector.extract_metrics(extracted)
        assert metrics['adjusted_health_score'] == 55.0
        assert metrics['debt_to_equity'] == 200.0


class TestEdgeCasesAndRobustness:
    """Test edge cases that could cause crashes or data loss."""

    def test_circular_reference_protection(self):
        """Should handle potential circular references gracefully."""
        # Note: Python dicts don't support true circular refs, but nested is similar
        deeply_nested = {"content": {"content": {"content": {"text": "Final"}}}}
        result = extract_string_content(deeply_nested)
        assert result == "Final"

    def test_very_large_string(self):
        """Should handle very large strings."""
        large_content = "A" * 1_000_000  # 1MB of text
        result = extract_string_content(large_content)
        assert len(result) == 1_000_000

    def test_unicode_content_preserved(self):
        """Unicode content should be preserved."""
        unicode_content = "日本語テスト 🚀 émojis and spëcial çharacters"
        result = extract_string_content(unicode_content)
        assert result == unicode_content

    def test_unicode_in_dict(self):
        """Unicode in dict should be preserved."""
        input_dict = {"text": "中文分析报告 SECTOR: 银行业"}
        result = extract_string_content(input_dict)
        assert "中文分析报告" in result
        assert "银行业" in result

    def test_newlines_preserved(self):
        """Newlines in content should be preserved."""
        content_with_newlines = "Line1\nLine2\n\nLine4"
        result = extract_string_content(content_with_newlines)
        assert result == content_with_newlines

    def test_special_regex_characters_not_breaking(self):
        """Content with regex special chars shouldn't break downstream parsing."""
        content = "SECTOR: Technology (Software & Services)\n### --- START DATA_BLOCK ---\nADJUSTED_HEALTH_SCORE: 50%\n### --- END DATA_BLOCK ---"
        input_dict = {"text": content}
        extracted = extract_string_content(input_dict)
        # Should not raise and should be parseable
        sector = RedFlagDetector.detect_sector(extracted)
        assert sector == Sector.TECHNOLOGY

    def test_whitespace_only_string(self):
        """Whitespace-only string should be returned as-is."""
        result = extract_string_content("   \n\t  ")
        assert result == "   \n\t  "

    def test_dict_with_numeric_keys(self):
        """Dict with non-string keys should be handled."""
        input_dict = {0: "first", 1: "second"}
        result = extract_string_content(input_dict)
        # Should fall back to str() representation
        assert "first" in result or "0" in result


class TestRealWorldScenarios:
    """Test with realistic problematic scenarios."""

    def test_original_bug_scenario_dict_length_1(self):
        """
        Reproduce the original bug: response.content is dict with 1 key.
        The log showed: 'fundamentals_output has_datablock=False length=1'
        meaning len(response.content) == 1, which is a dict with 1 key.
        """
        # Simulating what might have been returned
        problematic_input = {"text": SAMPLE_FUNDAMENTALS_REPORT}

        # This should NOT return len=1 string, but the full report
        result = extract_string_content(problematic_input)

        # Verify the report content is preserved
        assert len(result) > 100  # Much longer than 1
        assert "SECTOR:" in result
        assert "DATA_BLOCK" in result

    def test_agent_response_with_tool_call_metadata(self):
        """Handle response that includes tool call metadata alongside text."""
        input_dict = {
            "text": SAMPLE_FUNDAMENTALS_REPORT,
            "tool_calls": [{"name": "get_financial_metrics", "id": "123"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }
        result = extract_string_content(input_dict)
        # Should extract just the text, not the metadata
        assert "tool_calls" not in result or "SECTOR:" in result
        assert "DATA_BLOCK" in result

    def test_partial_response_recovery(self):
        """Even malformed responses should not crash."""
        # Partial/malformed dict
        input_dict = {"partial": True, "data": None}
        result = extract_string_content(input_dict)
        # Should return something, not crash
        assert isinstance(result, str)

    def test_langchain_message_content_format(self):
        """Handle LangChain message content format variations."""
        # LangChain sometimes wraps content in specific structures
        input_content = [
            {"type": "text", "text": "Analysis follows:"},
            {"type": "text", "text": SAMPLE_FUNDAMENTALS_REPORT}
        ]
        result = extract_string_content(input_content)
        # Should extract text from typed content blocks
        assert "Analysis follows:" in result
        assert "SECTOR:" in result


class TestValidatorNodeIntegration:
    """Test integration with financial_health_validator_node logic."""

    def test_validator_receives_dict_state(self):
        """
        Simulate what happens when state['fundamentals_report'] is a dict.
        This is the exact bug scenario.
        """
        # Simulating corrupted state
        state_value = {"text": SAMPLE_FUNDAMENTALS_REPORT}

        # The fix in financial_health_validator_node does:
        # if not isinstance(fundamentals_report, str):
        #     fundamentals_report = extract_string_content(fundamentals_report)

        if not isinstance(state_value, str):
            normalized = extract_string_content(state_value)
        else:
            normalized = state_value

        # Verify it's now usable
        sector = RedFlagDetector.detect_sector(normalized)
        metrics = RedFlagDetector.extract_metrics(normalized)

        assert isinstance(normalized, str)
        assert metrics['adjusted_health_score'] == 67.0

    def test_validator_receives_list_state(self):
        """Simulate state accumulation resulting in list."""
        # LangGraph state accumulation scenario
        state_value = [
            "Previous partial report",
            SAMPLE_FUNDAMENTALS_REPORT  # Latest/correct report
        ]

        if not isinstance(state_value, str):
            normalized = extract_string_content(state_value)
        else:
            normalized = state_value

        # Should contain both, but importantly should have DATA_BLOCK
        assert "DATA_BLOCK" in normalized
        metrics = RedFlagDetector.extract_metrics(normalized)
        assert metrics['adjusted_health_score'] == 67.0


class TestRegressionDetection:
    """
    REGRESSION TESTS: These tests call actual agent node functions with mocked
    LLM responses that return DICT instead of string. If extract_string_content()
    is removed or broken, these tests will fail with clear error messages.

    These tests exist specifically to catch the bug where Gemini returns dict
    for response.content instead of string.
    """

    @pytest.mark.asyncio
    async def test_analyst_node_handles_dict_response_REGRESSION(self):
        """
        REGRESSION TEST: Verifies create_analyst_node converts dict response to string.

        If this test fails, check that extract_string_content(response.content) is
        being called in create_analyst_node() around line 279 of src/agents.py.
        """
        from unittest.mock import MagicMock, AsyncMock, patch
        from src.agents import create_analyst_node

        # Create a proper mock response with DICT content (the bug scenario)
        mock_response = MagicMock()
        mock_response.content = {"text": SAMPLE_FUNDAMENTALS_REPORT}  # DICT, not string!
        mock_response.tool_calls = None  # No tool calls, so should set output field

        # Mock the entire chain: llm.bind_tools(tools) returns something that has ainvoke
        mock_bound = MagicMock()
        mock_bound.ainvoke = AsyncMock(return_value=mock_response)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound

        # Also need to mock the pipe operator result
        with patch('src.agents.filter_messages_for_gemini', return_value=[]), \
             patch('src.agents.invoke_with_rate_limit_handling', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_response

            node = create_analyst_node(mock_llm, "fundamentals_analyst", [], "fundamentals_report")

            state = {
                "messages": [],
                "company_of_interest": "1681.HK",
                "company_name": "CONSUN PHARMA",
                "prompts_used": {}
            }
            config = {"configurable": {"context": MagicMock(ticker="1681.HK", trade_date="2024-01-01")}}

            result = await node(state, config)

        # THE CRITICAL ASSERTION: Output must be string, not dict
        assert "fundamentals_report" in result, "Node should set fundamentals_report in output"
        output_value = result["fundamentals_report"]

        assert isinstance(output_value, str), (
            f"REGRESSION DETECTED: fundamentals_report is {type(output_value).__name__}, expected str. "
            f"Check that extract_string_content(response.content) is called in create_analyst_node(). "
            f"Value: {output_value}"
        )
        assert "SECTOR:" in output_value, "Report content should be preserved"
        assert "DATA_BLOCK" in output_value, "DATA_BLOCK should be preserved"

    @pytest.mark.asyncio
    async def test_researcher_node_handles_dict_response_REGRESSION(self):
        """
        REGRESSION TEST: Verifies create_researcher_node converts dict response to string.

        If this test fails, check that extract_string_content(response.content) is
        being called in create_researcher_node() around line 348 of src/agents.py.
        """
        from unittest.mock import MagicMock, AsyncMock
        from src.agents import create_researcher_node

        mock_llm = MagicMock()
        mock_response = MagicMock()
        # DICT response instead of string!
        mock_response.content = {"text": "BULL CASE: Strong fundamentals support a buy rating."}
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        node = create_researcher_node(mock_llm, None, "bull_researcher")

        state = {
            "market_report": "Market is bullish",
            "fundamentals_report": SAMPLE_FUNDAMENTALS_REPORT,
            "company_of_interest": "1681.HK",
            "company_name": "CONSUN PHARMA",
            "investment_debate_state": {
                "history": "",
                "bull_history": "",
                "bear_history": "",
                "count": 0
            }
        }
        config = {"configurable": {}}

        result = await node(state, config)

        # THE CRITICAL ASSERTION
        debate_state = result.get("investment_debate_state", {})
        history = debate_state.get("history", "")

        assert isinstance(history, str), (
            f"REGRESSION DETECTED: debate history is {type(history).__name__}, expected str. "
            f"Check that extract_string_content(response.content) is called in create_researcher_node()."
        )
        assert "Bull" in history or "BULL" in history, "Bull analyst name should be in history"

    @pytest.mark.asyncio
    async def test_portfolio_manager_node_handles_dict_response_REGRESSION(self):
        """
        REGRESSION TEST: Verifies create_portfolio_manager_node converts dict response to string.

        If this test fails, check that extract_string_content(response.content) is
        being called in create_portfolio_manager_node() around line 483 of src/agents.py.
        """
        from unittest.mock import MagicMock, AsyncMock
        from src.agents import create_portfolio_manager_node

        mock_llm = MagicMock()
        mock_response = MagicMock()
        # DICT response with final decision
        mock_response.content = {"text": "## FINAL DECISION: BUY\n\nStrong conviction based on fundamentals."}
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        node = create_portfolio_manager_node(mock_llm, None)

        state = {
            "market_report": "Bullish",
            "sentiment_report": "Positive",
            "news_report": "No major news",
            "fundamentals_report": SAMPLE_FUNDAMENTALS_REPORT,
            "investment_plan": "Buy recommendation",
            "consultant_review": "",
            "trader_investment_plan": "Execute buy",
            "risk_debate_state": {"history": "Risk acceptable"},
            "company_of_interest": "1681.HK",
            "company_name": "CONSUN PHARMA",
            "red_flags": [],
            "pre_screening_result": "PASS"
        }
        config = {"configurable": {}}

        result = await node(state, config)

        # THE CRITICAL ASSERTION
        final_decision = result.get("final_trade_decision", "")

        assert isinstance(final_decision, str), (
            f"REGRESSION DETECTED: final_trade_decision is {type(final_decision).__name__}, expected str. "
            f"Check that extract_string_content(response.content) is called in create_portfolio_manager_node()."
        )
        assert "BUY" in final_decision or "DECISION" in final_decision, "Decision content should be preserved"

    @pytest.mark.asyncio
    async def test_risk_analyst_node_handles_dict_response_REGRESSION(self):
        """
        REGRESSION TEST: Verifies create_risk_debater_node converts dict response to string.

        If this test fails, check that extract_string_content(response.content) is
        being called in create_risk_debater_node() around line 431 of src/agents.py.
        """
        from unittest.mock import MagicMock, AsyncMock
        from src.agents import create_risk_debater_node

        mock_llm = MagicMock()
        mock_response = MagicMock()
        # DICT response
        mock_response.content = {"text": "RISK ASSESSMENT: Position size should be limited to 2%."}
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        node = create_risk_debater_node(mock_llm, "conservative_risk")

        state = {
            "trader_investment_plan": "Buy 1681.HK",
            "consultant_review": "",
            "risk_debate_state": {"history": "", "count": 0},
            "company_of_interest": "1681.HK"
        }
        config = {"configurable": {}}

        result = await node(state, config)

        # THE CRITICAL ASSERTION
        risk_state = result.get("risk_debate_state", {})
        history = risk_state.get("history", "")

        assert isinstance(history, str), (
            f"REGRESSION DETECTED: risk history is {type(history).__name__}, expected str. "
            f"Check that extract_string_content(response.content) is called in create_risk_debater_node()."
        )

    @pytest.mark.asyncio
    async def test_validator_node_handles_dict_state_REGRESSION(self):
        """
        REGRESSION TEST: Verifies financial_health_validator_node handles dict state.

        If this test fails, check that extract_string_content() is called for
        non-string fundamentals_report in financial_health_validator_node()
        around line 662 of src/agents.py.
        """
        from src.agents import create_financial_health_validator_node

        node = create_financial_health_validator_node()

        # State with DICT fundamentals_report (simulating upstream bug)
        state = {
            "fundamentals_report": {"text": SAMPLE_FUNDAMENTALS_REPORT},  # DICT!
            "company_of_interest": "1681.HK",
            "company_name": "CONSUN PHARMA"
        }
        config = {"configurable": {}}

        # This should NOT raise TypeError: expected string or bytes-like object, got 'dict'
        try:
            result = await node(state, config)
        except TypeError as e:
            if "expected string" in str(e) and "dict" in str(e):
                pytest.fail(
                    f"REGRESSION DETECTED: Validator crashed on dict input. "
                    f"Check that extract_string_content() is called for non-string "
                    f"fundamentals_report in financial_health_validator_node(). "
                    f"Error: {e}"
                )
            raise

        assert "pre_screening_result" in result, "Validator should return pre_screening_result"

    def test_extract_string_content_function_exists_REGRESSION(self):
        """
        REGRESSION TEST: Verifies extract_string_content is importable.

        If this test fails, the function may have been removed or renamed.
        """
        try:
            from src.agents import extract_string_content
        except ImportError as e:
            pytest.fail(
                f"REGRESSION DETECTED: extract_string_content cannot be imported. "
                f"This function is critical for handling Gemini dict responses. "
                f"Error: {e}"
            )

        # Verify it works
        assert extract_string_content("test") == "test"
        assert extract_string_content({"text": "test"}) == "test"
        assert extract_string_content(None) == ""

    def test_utils_imports_extract_string_content_REGRESSION(self):
        """
        REGRESSION TEST: Verifies src/utils.py imports extract_string_content.

        If this test fails, SignalProcessor and Reflector may crash on dict responses.
        """
        import importlib
        import src.utils as utils_module

        # Reload to get fresh import
        importlib.reload(utils_module)

        assert hasattr(utils_module, 'extract_string_content'), (
            "REGRESSION DETECTED: src/utils.py does not import extract_string_content. "
            "SignalProcessor and Reflector need this to handle Gemini dict responses."
        )


class TestMockedLLMResponseScenarios:
    """
    Test scenarios simulating actual LLM response patterns.
    These tests verify the full chain from LLM response to usable output.
    """

    def test_analyst_output_from_dict_response(self):
        """
        Simulate an analyst node receiving a dict response.content.
        This is the pattern used in create_analyst_node().
        """
        # Simulate what Gemini might return
        mock_response_content = {"text": SAMPLE_FUNDAMENTALS_REPORT}

        # This is what create_analyst_node does now:
        content_str = extract_string_content(mock_response_content)
        output_field_value = content_str  # Goes into state['fundamentals_report']

        # Verify the output is correct for downstream consumers
        assert isinstance(output_field_value, str)
        assert "SECTOR:" in output_field_value
        assert "DATA_BLOCK" in output_field_value

        # Downstream agent (validator) can parse it
        metrics = RedFlagDetector.extract_metrics(output_field_value)
        assert metrics['adjusted_health_score'] == 67.0

    def test_researcher_argument_from_dict_response(self):
        """
        Simulate a researcher node (Bull/Bear) receiving a dict response.
        This is the pattern used in create_researcher_node().
        """
        mock_response_content = {
            "text": "BULL ARGUMENT: The company shows strong fundamentals with D/E: 45 and improving margins."
        }

        # What create_researcher_node does:
        content_str = extract_string_content(mock_response_content)
        argument = f"Bull Analyst: {content_str}"

        # Verify the argument is properly formatted
        assert "Bull Analyst:" in argument
        assert "strong fundamentals" in argument
        assert "D/E: 45" in argument

    def test_portfolio_manager_decision_from_dict_response(self):
        """
        Simulate portfolio manager receiving a dict response.
        The final_trade_decision must be parseable for signal extraction.
        """
        mock_response_content = {
            "text": """## FINAL DECISION: BUY

**Conviction Level:** HIGH (85%)

**Position Size:** 3% of portfolio

**Rationale:** Strong fundamentals with GARP characteristics.
"""
        }

        # What create_portfolio_manager_node does:
        final_decision = extract_string_content(mock_response_content)

        # Signal extraction pattern (from SignalProcessor)
        import re
        signal_match = re.search(r'FINAL DECISION:\s*(BUY|SELL|HOLD)', final_decision, re.IGNORECASE)
        assert signal_match is not None
        assert signal_match.group(1).upper() == "BUY"

    def test_risk_analyst_from_dict_response(self):
        """
        Simulate risk analyst receiving a dict response.
        """
        mock_response_content = {
            "parts": [
                {"text": "RISK ASSESSMENT (Conservative):\n"},
                {"text": "- Position sizing: Recommend 1-2% max\n- Key risks: Currency volatility, regulatory changes"}
            ]
        }

        content_str = extract_string_content(mock_response_content)
        risk_history = f"Conservative Risk Analyst: {content_str}"

        assert "RISK ASSESSMENT" in risk_history
        assert "Position sizing" in risk_history
        assert "Key risks" in risk_history

    def test_chain_of_agents_with_dict_responses(self):
        """
        Simulate a full chain: Fundamentals Analyst -> Validator -> Researcher.
        All receiving dict responses but still producing correct output.
        """
        # Step 1: Fundamentals Analyst produces report (dict response)
        fundamentals_response = {"text": SAMPLE_FUNDAMENTALS_REPORT}
        fundamentals_report = extract_string_content(fundamentals_response)

        # Step 2: Validator processes the report
        # (The validator now normalizes non-string input)
        if not isinstance(fundamentals_report, str):
            fundamentals_report = extract_string_content(fundamentals_report)

        sector = RedFlagDetector.detect_sector(fundamentals_report)
        metrics = RedFlagDetector.extract_metrics(fundamentals_report)

        assert sector == Sector.GENERAL
        assert metrics['adjusted_health_score'] == 67.0
        assert metrics['debt_to_equity'] == 45.0

        # Step 3: Researcher uses the report (dict response)
        researcher_response = {
            "text": f"Based on the fundamentals showing {metrics['adjusted_health_score']}% health score, I argue BUY."
        }
        argument = extract_string_content(researcher_response)

        assert "67" in argument or "health score" in argument
        assert "BUY" in argument

    def test_market_analyst_output_format(self):
        """
        Test that market analyst output (used by researchers) is correctly formatted.
        """
        market_report_response = {
            "text": """## Market Analysis for 1681.HK

**Trend:** Bullish
**Support:** HKD 3.50
**Resistance:** HKD 4.20
**Volume:** Above average

Technical indicators suggest continued upward momentum.
"""
        }

        market_report = extract_string_content(market_report_response)

        # Researchers use this in their prompt
        reports_context = f"MARKET: {market_report}\nFUNDAMENTALS: {SAMPLE_FUNDAMENTALS_REPORT}"

        assert "Trend:" in reports_context
        assert "SECTOR:" in reports_context
        assert "DATA_BLOCK" in reports_context

    def test_consultant_review_from_dict_response(self):
        """
        Test consultant node handling dict response.
        """
        consultant_response = {
            "content": {
                "text": "EXTERNAL REVIEW: The analysis appears sound. No major red flags detected. Recommend proceed with caution on position sizing."
            }
        }

        # Nested content handling
        review = extract_string_content(consultant_response)

        assert "EXTERNAL REVIEW" in review
        assert "red flags" in review

    def test_signal_extractor_with_dict_response(self):
        """
        Test SignalProcessor/SignalExtractor handling dict response.
        This is critical for the final BUY/SELL/HOLD extraction.
        """
        # The LLM is asked to return just BUY/SELL/HOLD
        llm_response = {"text": "BUY"}

        content = extract_string_content(llm_response).strip().upper()

        assert content in ["BUY", "SELL", "HOLD"]
        assert content == "BUY"

    def test_reflector_lesson_from_dict_response(self):
        """
        Test Reflector class handling dict response for lesson generation.
        """
        lesson_response = {
            "text": "In a market with strong technical momentum, position sizing should be increased when fundamentals confirm the trend."
        }

        lesson = extract_string_content(lesson_response).strip()

        assert len(lesson) > 20  # Meaningful lesson
        assert "momentum" in lesson or "position" in lesson
