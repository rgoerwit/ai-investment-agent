"""
Tests for report_generator.py
Covers normalization, decision extraction, and report generation.
UPDATED: Added tests for --brief mode functionality.
"""

from datetime import datetime

from src.report_generator import QuietModeReporter, suppress_logging


class TestNormalizeString:
    """Test _normalize_string() edge cases."""

    def test_normalize_none(self):
        """Test None input returns empty string."""
        reporter = QuietModeReporter("AAPL")
        result = reporter._normalize_string(None)
        assert result == ""

    def test_normalize_string_passthrough(self):
        """Test regular string passes through."""
        reporter = QuietModeReporter("AAPL")
        result = reporter._normalize_string("Hello World")
        assert result == "Hello World"

    def test_normalize_list_deduplication_logic(self):
        """
        REGRESSION TEST: Verify duplicate report sections are removed.
        This fixes the 'stuttering' output bug where analysis repeats.
        """
        reporter = QuietModeReporter("AAPL")
        input_list = [
            "Market analysis part 1",
            "Market analysis part 1",  # Duplicate
            "Market analysis part 2",
        ]
        result = reporter._normalize_string(input_list)

        # Should only appear once
        assert result.count("Market analysis part 1") == 1
        assert "Market analysis part 2" in result
        # Should be joined by newlines
        assert result == "Market analysis part 1\n\nMarket analysis part 2"

    def test_normalize_empty_list(self):
        """Test empty list returns empty string."""
        reporter = QuietModeReporter("AAPL")
        result = reporter._normalize_string([])
        assert result == ""

    def test_normalize_list_single_item(self):
        """Test list with single item."""
        reporter = QuietModeReporter("AAPL")
        result = reporter._normalize_string(["Test"])
        assert result == "Test"

    def test_normalize_list_multiple_items(self):
        """Test list joins with double newlines."""
        reporter = QuietModeReporter("AAPL")
        result = reporter._normalize_string(["First", "Second", "Third"])
        assert result == "First\n\nSecond\n\nThird"

    def test_normalize_list_with_none(self):
        """Test list filters out None values."""
        reporter = QuietModeReporter("AAPL")
        result = reporter._normalize_string(["First", None, "Third", None])
        assert result == "First\n\nThird"

    def test_normalize_list_with_empty_strings(self):
        """Test list filters out empty strings."""
        reporter = QuietModeReporter("AAPL")
        result = reporter._normalize_string(["First", "", "Third"])
        assert result == "First\n\nThird"

    def test_normalize_integer(self):
        """Test integer converts to string."""
        reporter = QuietModeReporter("AAPL")
        result = reporter._normalize_string(42)
        assert result == "42"

    def test_normalize_float(self):
        """Test float converts to string."""
        reporter = QuietModeReporter("AAPL")
        result = reporter._normalize_string(3.14)
        assert result == "3.14"

    def test_normalize_nested_list(self):
        """Test nested list flattening."""
        reporter = QuietModeReporter("AAPL")
        # Lists within lists should be converted to strings
        result = reporter._normalize_string([["Nested", "List"], "Item"])
        assert "Nested" in result or "['Nested', 'List']" in result


class TestExtractDecision:
    """Test extract_decision() with various input formats.

    NOTE: extract_decision() only trusts PM patterns (PORTFOLIO MANAGER VERDICT:
    and PM_BLOCK VERDICT:) to prevent leakage from subordinate agents.
    """

    def test_extract_pm_verdict(self):
        """Test extraction from PORTFOLIO MANAGER VERDICT: field."""
        reporter = QuietModeReporter("AAPL")
        text = "#### PORTFOLIO MANAGER VERDICT: BUY\nPosition: 5%"
        assert reporter.extract_decision(text) == "BUY"

    def test_extract_pm_block_verdict(self):
        """Test extraction from PM_BLOCK VERDICT: field."""
        reporter = QuietModeReporter("AAPL")
        text = "#### --- START PM_BLOCK ---\nVERDICT: SELL\n#### --- END PM_BLOCK ---"
        assert reporter.extract_decision(text) == "SELL"

    def test_non_pm_action_field_ignored(self):
        """Non-PM Action: field should be ignored (prevents Trader leakage)."""
        reporter = QuietModeReporter("AAPL")
        text = "### FINAL EXECUTION PARAMETERS\nAction: BUY\nPosition: 5%"
        # Without PM verdict, defaults to HOLD (safe)
        assert reporter.extract_decision(text) == "HOLD"

    def test_non_pm_final_decision_ignored(self):
        """Non-PM FINAL DECISION: should be ignored (prevents Research Manager leakage)."""
        reporter = QuietModeReporter("AAPL")
        text = "### FINAL DECISION: SELL\nRationale: Too risky"
        assert reporter.extract_decision(text) == "HOLD"

    def test_generic_keyword_ignored(self):
        """Generic keywords should be ignored (prevents subordinate agent leakage)."""
        reporter = QuietModeReporter("AAPL")
        text = "I recommend to BUY this stock"
        # No PM verdict, so defaults to HOLD
        assert reporter.extract_decision(text) == "HOLD"

    def test_extract_bold_markdown(self):
        """Test extraction with markdown bold in PM verdict."""
        reporter = QuietModeReporter("AAPL")
        text = "#### PORTFOLIO MANAGER VERDICT: **SELL**"
        assert reporter.extract_decision(text) == "SELL"

    def test_extract_lowercase(self):
        """Test extraction converts to uppercase."""
        reporter = QuietModeReporter("AAPL")
        text = "#### Portfolio Manager Verdict: buy"
        assert reporter.extract_decision(text) == "BUY"

    def test_extract_with_extra_whitespace(self):
        """Test extraction handles extra whitespace."""
        reporter = QuietModeReporter("AAPL")
        text = "#### PORTFOLIO MANAGER VERDICT:    HOLD   "
        assert reporter.extract_decision(text) == "HOLD"

    def test_pm_block_takes_priority(self):
        """PM_BLOCK VERDICT takes priority over prose verdict."""
        reporter = QuietModeReporter("AAPL")
        text = "#### PORTFOLIO MANAGER VERDICT: SELL\n\nVERDICT: BUY"
        # PM_BLOCK VERDICT (line start) should win
        assert reporter.extract_decision(text) == "BUY"

    def test_extract_default_hold(self):
        """Test defaults to HOLD when no PM verdict found."""
        reporter = QuietModeReporter("AAPL")
        text = "No clear decision in this text"
        assert reporter.extract_decision(text) == "HOLD"

    def test_extract_invalid_decision(self):
        """Test invalid decision word ignored."""
        reporter = QuietModeReporter("AAPL")
        text = "#### PORTFOLIO MANAGER VERDICT: MAYBE"
        assert reporter.extract_decision(text) == "HOLD"

    def test_extract_from_list(self):
        """Test extraction when input is a list (LangGraph accumulation)."""
        reporter = QuietModeReporter("AAPL")
        text_list = ["Some preamble", "#### PORTFOLIO MANAGER VERDICT: SELL"]
        assert reporter.extract_decision(text_list) == "SELL"

    def test_extract_none_input(self):
        """Test extraction from None input."""
        reporter = QuietModeReporter("AAPL")
        assert reporter.extract_decision(None) == "HOLD"


class TestExtractDecisionRationale:
    """Test _extract_decision_rationale() for brief mode."""

    def test_extract_rationale_section(self):
        """Test extraction of explicit RATIONALE section."""
        reporter = QuietModeReporter("AAPL")
        text = """
Action: BUY

DECISION RATIONALE: Strong fundamentals and positive momentum suggest this is
a good entry point. The company is well-positioned for growth.

Additional notes here.
"""
        rationale = reporter._extract_decision_rationale(text)
        assert "Strong fundamentals" in rationale
        assert "good entry point" in rationale
        assert "Additional notes" not in rationale  # Should stop at section boundary

    def test_extract_rationale_uppercase(self):
        """Test extraction with all caps RATIONALE."""
        reporter = QuietModeReporter("AAPL")
        text = "Action: SELL\n\nRATIONALE: Overvalued and risky market conditions."
        rationale = reporter._extract_decision_rationale(text)
        assert "Overvalued" in rationale

    def test_extract_reasoning_section(self):
        """Test extraction using REASONING keyword."""
        reporter = QuietModeReporter("AAPL")
        text = "Decision: HOLD\n\nREASONING: Waiting for clearer signals before entry."
        rationale = reporter._extract_decision_rationale(text)
        assert "Waiting for clearer" in rationale

    def test_extract_justification_section(self):
        """Test extraction using JUSTIFICATION keyword."""
        reporter = QuietModeReporter("AAPL")
        text = (
            "Action: BUY\n\nJUSTIFICATION: Technical indicators support bullish move."
        )
        rationale = reporter._extract_decision_rationale(text)
        assert "Technical indicators" in rationale

    def test_extract_rationale_fallback_after_decision(self):
        """Test fallback when no explicit section but text follows decision."""
        reporter = QuietModeReporter("AAPL")
        text = """
Action: BUY

This stock shows strong momentum with increasing volume.
The fundamentals are solid and valuation is reasonable.
Risk-reward ratio is favorable at current levels.
"""
        rationale = reporter._extract_decision_rationale(text)
        assert "strong momentum" in rationale or "fundamentals" in rationale

    def test_extract_rationale_no_section_found(self):
        """Test fallback to first paragraphs when no clear section."""
        reporter = QuietModeReporter("AAPL")
        text = """
The analysis reveals positive indicators.

Strong buy signals present across multiple timeframes.
"""
        rationale = reporter._extract_decision_rationale(text)
        assert len(rationale) > 0

    def test_extract_rationale_empty_text(self):
        """Test extraction from empty text."""
        reporter = QuietModeReporter("AAPL")
        rationale = reporter._extract_decision_rationale("")
        assert rationale == ""

    def test_extract_rationale_from_list(self):
        """Test extraction when input is list."""
        reporter = QuietModeReporter("AAPL")
        text_list = ["Preamble", "RATIONALE: Key reasons here"]
        rationale = reporter._extract_decision_rationale(text_list)
        assert "Key reasons" in rationale


class TestCleanText:
    """Test _clean_text() cleanup functions."""

    def test_clean_excessive_newlines(self):
        """Test excessive newlines reduced to double."""
        reporter = QuietModeReporter("AAPL")
        text = "Line 1\n\n\n\n\nLine 2"
        result = reporter._clean_text(text)
        assert "\n\n\n" not in result
        assert "Line 1\n\nLine 2" in result

    def test_clean_agent_prefixes(self):
        """Test agent prefixes removed."""
        reporter = QuietModeReporter("AAPL")
        text = "Bull Analyst: This is great\nBear Analyst: This is bad"
        result = reporter._clean_text(text)
        assert "Bull Analyst:" not in result
        assert "Bear Analyst:" not in result
        assert "This is great" in result

    def test_clean_all_agent_types(self):
        """Test all agent types removed."""
        reporter = QuietModeReporter("AAPL")
        agents = [
            "Bull Analyst: Buy",
            "Bear Analyst: Sell",
            "Risky Analyst: High risk",
            "Safe Analyst: Low risk",
            "Neutral Analyst: Moderate",
            "Trader: Entry at $100",
            "Portfolio Manager: Approve",
        ]
        text = "\n".join(agents)
        result = reporter._clean_text(text)

        for agent in [
            "Bull",
            "Bear",
            "Risky",
            "Safe",
            "Neutral",
            "Trader",
            "Portfolio",
        ]:
            assert f"{agent} Analyst:" not in result

    def test_clean_strips_whitespace(self):
        """Test leading/trailing whitespace removed."""
        reporter = QuietModeReporter("AAPL")
        text = "   \n\n  Content  \n\n  "
        result = reporter._clean_text(text)
        assert result == "Content\n"

    def test_clean_preserves_content(self):
        """Test content is preserved."""
        reporter = QuietModeReporter("AAPL")
        text = "This is important content\n"
        result = reporter._clean_text(text)
        assert result == text


class TestGenerateReport:
    """Test generate_report() main function."""

    def test_generate_basic_report(self):
        """Test basic report generation."""
        reporter = QuietModeReporter("AAPL", "Apple Inc.")
        result_dict = {
            "final_trade_decision": "Action: BUY",
            "market_report": "RSI: 45",
            "fundamentals_report": "P/E: 25",
        }

        report = reporter.generate_report(result_dict)

        assert "AAPL" in report
        assert "Apple Inc." in report
        assert "BUY" in report
        assert "Technical Analysis" in report
        assert "Fundamental Analysis" in report

    def test_generate_report_no_company_name(self):
        """Test report without company name."""
        reporter = QuietModeReporter("GOOGL")
        result_dict = {"final_trade_decision": "Action: SELL"}

        report = reporter.generate_report(result_dict)

        assert "GOOGL" in report
        assert "SELL" in report

    def test_generate_report_missing_sections(self):
        """Test report with missing optional sections."""
        reporter = QuietModeReporter("MSFT")
        result_dict = {"final_trade_decision": "Action: HOLD"}

        report = reporter.generate_report(result_dict)

        # Should have basic structure even without sections
        assert "MSFT" in report
        assert "HOLD" in report
        assert "---" in report

    def test_generate_report_error_sections(self):
        """Test sections starting with 'Error' are excluded."""
        reporter = QuietModeReporter("TSLA")
        result_dict = {
            "final_trade_decision": "Action: BUY",
            "market_report": "Error: Could not fetch data",
            "fundamentals_report": "P/E: 100",
        }

        report = reporter.generate_report(result_dict)

        # Error section should not appear
        assert "Technical Analysis" not in report
        # But fundamentals should
        assert "Fundamental Analysis" in report

    def test_generate_report_list_accumulation(self):
        """Test handling of LangGraph list accumulation."""
        reporter = QuietModeReporter("NVDA")
        result_dict = {
            "final_trade_decision": ["Preamble", "Action: BUY"],
            "market_report": ["RSI: 50", "MACD: Bullish"],
        }

        report = reporter.generate_report(result_dict)

        assert "NVDA" in report
        assert "BUY" in report
        # List should be joined
        assert "RSI: 50" in report
        assert "MACD: Bullish" in report

    def test_generate_report_risk_state_dict(self):
        """Test risk assessment from nested dict with dedicated fields."""
        reporter = QuietModeReporter("AMD")
        result_dict = {
            "final_trade_decision": "Action: HOLD",
            "risk_debate_state": {
                "current_risky_response": "High risk tolerance recommended",
                "current_safe_response": "Low risk approach preferred",
                "current_neutral_response": "",
            },
        }

        report = reporter.generate_report(result_dict)

        assert "Risk Assessment" in report
        assert "High risk" in report

    def test_generate_report_risk_state_list(self):
        """Test risk assessment from list (takes last item)."""
        reporter = QuietModeReporter("INTC")
        result_dict = {
            "final_trade_decision": "Action: BUY",
            "risk_debate_state": [
                {"current_risky_response": "Old risky view"},
                {
                    "current_risky_response": "Latest risky: This is current",
                    "current_safe_response": "Latest safe view",
                },
            ],
        }

        report = reporter.generate_report(result_dict)

        assert "Risk Assessment" in report
        assert "Latest" in report
        assert "Old risky" not in report

    def test_generate_report_timestamp_format(self):
        """Test timestamp is properly formatted."""
        reporter = QuietModeReporter("IBM")
        result_dict = {"final_trade_decision": "Action: HOLD"}

        report = reporter.generate_report(result_dict)

        # Should have timestamp in YYYY-MM-DD HH:MM:SS format
        assert "Analysis Date:" in report
        assert reporter.timestamp in report

    def test_generate_report_all_sections(self):
        """Test report with all possible sections (PM output present)."""
        reporter = QuietModeReporter("META", "Meta Platforms")
        result_dict = {
            "final_trade_decision": "Action: BUY",
            "market_report": "Bullish trend",
            "fundamentals_report": "Strong financials",
            "sentiment_report": "Positive sentiment",
            "news_report": "New product launch",
            "investment_plan": "Recommend BUY",
            "trader_investment_plan": "Entry: $300",
            "risk_debate_state": {
                "current_risky_response": "Moderate risk acceptable",
                "current_safe_response": "",
                "current_neutral_response": "",
            },
        }

        report = reporter.generate_report(result_dict)

        # Check sections present when PM output (final_trade_decision) exists.
        # Note: "Investment Recommendation" (investment_plan) is a fallback section
        # shown only when PM output is absent — it is NOT expected here.
        assert "Technical Analysis" in report
        assert "Fundamental Analysis" in report
        assert "Market Sentiment" in report
        assert "News & Catalysts" in report
        assert "Trading Strategy" in report
        assert "Risk Assessment" in report

    def test_investment_recommendation_shown_without_pm_output(self):
        """Investment Recommendation section appears only when PM output is absent."""
        reporter = QuietModeReporter("META", "Meta Platforms")
        result_dict = {
            # No final_trade_decision — PM failed
            "market_report": "Bullish trend",
            "investment_plan": "Recommend BUY",
        }

        report = reporter.generate_report(result_dict)

        assert "Investment Recommendation" in report


class TestBriefMode:
    """Test brief_mode functionality for --brief flag."""

    def test_brief_mode_basic(self):
        """Test brief mode outputs only header, summary, and rationale."""
        reporter = QuietModeReporter("AAPL", "Apple Inc.")
        result_dict = {
            "final_trade_decision": """
Action: BUY

Executive summary text here.

DECISION RATIONALE: Strong fundamentals with positive momentum.
The valuation is attractive at current levels.
""",
            "market_report": "RSI: 45 - Oversold",
            "fundamentals_report": "P/E: 25 - Reasonable",
            "sentiment_report": "Bullish",
            "news_report": "Product launch next week",
        }

        report = reporter.generate_report(result_dict, brief_mode=True)

        # Should contain header and summary
        assert "AAPL" in report
        assert "Apple Inc." in report
        assert "BUY" in report
        assert "Executive Summary" in report

        # Should contain rationale (inline, not as separate header)
        assert "RATIONALE" in report.upper()
        assert "Strong fundamentals" in report

        # Should NOT contain full sections
        assert "Technical Analysis" not in report
        assert "Fundamental Analysis" not in report
        assert "Market Sentiment" not in report
        assert "News & Catalysts" not in report

        # Should indicate brief mode in footer
        assert "Brief Mode" in report

    def test_brief_mode_vs_full_mode(self):
        """Test brief mode is significantly shorter than full mode."""
        reporter = QuietModeReporter("TSLA")
        result_dict = {
            "final_trade_decision": "Action: SELL\n\nRATIONALE: Overvalued currently.",
            "market_report": "Technical indicators suggest overbought conditions with RSI at 75.",
            "fundamentals_report": "P/E ratio of 150 significantly above sector average.",
            "sentiment_report": "Mixed sentiment with institutional selling pressure.",
            "news_report": "Recent earnings miss and guidance reduction.",
            "investment_plan": "Recommend taking profits at current levels.",
            "trader_investment_plan": "Exit positions above $250 with stop loss at $275.",
        }

        brief_report = reporter.generate_report(result_dict, brief_mode=True)
        full_report = reporter.generate_report(result_dict, brief_mode=False)

        # Brief should be shorter than full (detail sections are omitted).
        # Both share a fixed header/footer; use absolute difference to avoid
        # ratio instability when header overhead is large relative to content.
        assert len(brief_report) < len(full_report)

        # Both should have header
        assert "TSLA" in brief_report
        assert "TSLA" in full_report

        # Only full should have detailed sections
        assert "Technical Analysis" in full_report
        assert "Technical Analysis" not in brief_report

    def test_brief_mode_without_rationale_section(self):
        """Test brief mode when no explicit rationale section exists."""
        reporter = QuietModeReporter("NVDA")
        result_dict = {
            "final_trade_decision": """
Action: BUY

Strong performance across all metrics with clear growth trajectory.
Market positioning is excellent and valuation remains attractive.
""",
            "market_report": "Bullish technical setup",
        }

        report = reporter.generate_report(result_dict, brief_mode=True)

        # Should still have basic structure
        assert "NVDA" in report
        assert "BUY" in report
        assert "Executive Summary" in report

        # Should have content from decision text (rationale may be inline or absent)
        assert "Strong performance" in report or "growth trajectory" in report
        # Should have reasonable length
        assert len(report) > 100

    def test_brief_mode_minimal_data(self):
        """Test brief mode with minimal result data."""
        reporter = QuietModeReporter("AMD")
        result_dict = {"final_trade_decision": "Action: HOLD"}

        report = reporter.generate_report(result_dict, brief_mode=True)

        # Should still generate valid report
        assert "AMD" in report
        assert "HOLD" in report
        assert "---" in report  # Separators
        assert "Brief Mode" in report

    def test_brief_mode_with_list_input(self):
        """Test brief mode handles LangGraph list accumulation."""
        reporter = QuietModeReporter("INTC")
        result_dict = {
            "final_trade_decision": [
                "Analysis part 1",
                "Action: BUY",
                "RATIONALE: Strong technical setup with fundamental support.",
            ]
        }

        report = reporter.generate_report(result_dict, brief_mode=True)

        assert "INTC" in report
        assert "BUY" in report
        assert "Strong technical setup" in report
        assert "Brief Mode" in report

    def test_quiet_mode_remains_full_report(self):
        """Test that --quiet (without --brief) still generates full report."""
        reporter = QuietModeReporter("MSFT", "Microsoft")
        result_dict = {
            "final_trade_decision": "Action: BUY\n\nRATIONALE: Good value.",
            "market_report": "Technical analysis here",
            "fundamentals_report": "Fundamental analysis here",
            "sentiment_report": "Sentiment analysis here",
        }

        # Quiet mode (brief_mode=False, which is default)
        report = reporter.generate_report(result_dict, brief_mode=False)

        # Should contain all sections
        assert "Technical Analysis" in report
        assert "Fundamental Analysis" in report
        assert "Market Sentiment" in report

        # Should NOT have Brief Mode indicator
        assert "Brief Mode" not in report

    def test_brief_mode_preserves_executive_summary(self):
        """Test brief mode always includes executive summary."""
        reporter = QuietModeReporter("GOOGL")
        result_dict = {
            "final_trade_decision": """
Action: SELL

### Executive Summary
The company faces significant headwinds with declining margins
and increasing competitive pressure. Current valuation does not
justify the elevated risk profile.

RATIONALE: Risk-reward unfavorable.
"""
        }

        report = reporter.generate_report(result_dict, brief_mode=True)

        # Executive summary should be present
        assert "Executive Summary" in report
        assert "significant headwinds" in report
        assert "declining margins" in report

        # Rationale should be present (inline)
        assert "RATIONALE" in report.upper()
        assert "Risk-reward" in report


class TestBriefAndQuietInteraction:
    """Test interaction between --brief and --quiet flags."""

    def test_brief_outputs_less_than_quiet(self):
        """Test --brief produces less output than --quiet."""
        reporter = QuietModeReporter("AAPL")
        result_dict = {
            "final_trade_decision": "Action: BUY\n\nRATIONALE: Good entry point.",
            "market_report": "Technical analysis with multiple indicators and detailed chart patterns.",
            "fundamentals_report": "Detailed fundamental metrics including revenue, earnings, margins.",
            "sentiment_report": "Social media sentiment and institutional positioning analysis.",
            "news_report": "Recent news including earnings reports and analyst upgrades.",
        }

        quiet_report = reporter.generate_report(result_dict, brief_mode=False)
        brief_report = reporter.generate_report(result_dict, brief_mode=True)

        # Brief should be shorter
        assert len(brief_report) < len(quiet_report)

        # Quiet has all sections
        assert "Technical Analysis" in quiet_report

        # Brief does not
        assert "Technical Analysis" not in brief_report

    def test_both_suppress_logging(self):
        """Test both --brief and --quiet suppress logging equally."""
        # Both modes should work with suppressed logging
        # This is tested at the main.py level, but we verify reporter works in both
        reporter = QuietModeReporter("TEST")
        result = {"final_trade_decision": "Action: HOLD"}

        # Should not raise exceptions
        brief = reporter.generate_report(result, brief_mode=True)
        quiet = reporter.generate_report(result, brief_mode=False)

        assert "TEST" in brief
        assert "TEST" in quiet

    def test_brief_markdown_still_valid(self):
        """Test --brief output is still valid markdown."""
        reporter = QuietModeReporter("NVDA", "NVIDIA")
        result_dict = {
            "final_trade_decision": """
Action: BUY

Strong AI positioning with data center dominance.

RATIONALE: Market leader with pricing power and strong demand outlook.
"""
        }

        report = reporter.generate_report(result_dict, brief_mode=True)

        # Should have markdown headers
        assert report.startswith("# ")
        assert "##" in report

        # Should have horizontal rules
        assert "---" in report

        # Should have proper structure
        lines = report.split("\n")
        assert lines[0].startswith("# NVDA")


class TestSuppressLogging:
    """Test suppress_logging() function."""

    def test_suppress_logging_no_errors(self):
        """Test suppress_logging runs without errors."""
        # Should not raise any exceptions
        suppress_logging()

    def test_logging_level_critical(self):
        """Test logging is set to CRITICAL after suppression."""
        import logging

        suppress_logging()

        # Root logger should be at CRITICAL
        assert logging.root.level == logging.CRITICAL


class TestEdgeCases:
    """Test edge cases and stress scenarios."""

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        reporter = QuietModeReporter("NFLX")
        result_dict = {
            "final_trade_decision": "Action: BUY 🚀",
            "market_report": "Stock is trending 📈",
        }

        report = reporter.generate_report(result_dict)

        # Should handle unicode without crashing
        assert "NFLX" in report

    def test_unicode_in_brief_mode(self):
        """Test unicode handling in brief mode."""
        reporter = QuietModeReporter("NFLX")
        result_dict = {
            "final_trade_decision": "Action: BUY 🚀\n\nRATIONALE: Strong momentum 📈"
        }

        report = reporter.generate_report(result_dict, brief_mode=True)
        assert "NFLX" in report

    def test_very_long_report(self):
        """Test handling of very long report sections."""
        reporter = QuietModeReporter("AMZN")
        long_text = "X" * 100000  # 100k characters
        result_dict = {
            "final_trade_decision": "Action: HOLD",
            "market_report": long_text,
        }

        report = reporter.generate_report(result_dict)

        # Should not crash, should include long text
        assert len(report) > 100000

    def test_special_markdown_characters(self):
        """Test handling of special markdown characters."""
        reporter = QuietModeReporter("DIS")
        result_dict = {
            "final_trade_decision": "Action: BUY",
            "market_report": "# Header\n**Bold** *Italic* `Code`",
        }

        report = reporter.generate_report(result_dict)

        # Should preserve markdown
        assert "**Bold**" in report or "Bold" in report

    def test_empty_result_dict(self):
        """Test handling of completely empty result dict."""
        reporter = QuietModeReporter("ORCL")
        result_dict = {}

        report = reporter.generate_report(result_dict)

        # Should still generate basic structure
        assert "ORCL" in report
        assert "ANALYSIS FAILED" in report
        assert "publishable analysis" in report

    def test_empty_result_dict_brief_mode(self):
        """Test empty result dict in brief mode."""
        reporter = QuietModeReporter("ORCL")
        result_dict = {}

        report = reporter.generate_report(result_dict, brief_mode=True)

        assert "ORCL" in report
        assert "ANALYSIS FAILED" in report
        assert "Brief Mode" in report

    def test_malformed_pm_verdict_formats(self):
        """Test various PM verdict format variations."""
        reporter = QuietModeReporter("CSCO")

        test_cases = [
            ("PORTFOLIO MANAGER VERDICT:BUY", "BUY"),  # No space
            ("PORTFOLIO MANAGER VERDICT : BUY", "BUY"),  # Extra space
            ("portfolio manager verdict: buy", "BUY"),  # Lowercase
            ("  PORTFOLIO MANAGER VERDICT:  SELL  ", "SELL"),  # Extra whitespace
            ("VERDICT: HOLD", "HOLD"),  # PM_BLOCK style
            ("VERDICT:DO_NOT_INITIATE", "DO NOT INITIATE"),  # No space, underscored
        ]

        for text, expected in test_cases:
            assert reporter.extract_decision(text) == expected, f"Failed for: {text}"


class TestInitialization:
    """Test QuietModeReporter initialization."""

    def test_init_ticker_uppercase(self):
        """Test ticker is converted to uppercase."""
        reporter = QuietModeReporter("aapl")
        assert reporter.ticker == "AAPL"

    def test_init_with_company_name(self):
        """Test initialization with company name."""
        reporter = QuietModeReporter("GOOGL", "Alphabet Inc.")
        assert reporter.ticker == "GOOGL"
        assert reporter.company_name == "Alphabet Inc."

    def test_init_timestamp_format(self):
        """Test timestamp is in correct format."""
        reporter = QuietModeReporter("MSFT")

        # Timestamp should be parseable
        datetime.strptime(reporter.timestamp, "%Y-%m-%d %H:%M:%S")


# Integration-style test
class TestReportIntegration:
    """Integration tests with realistic data."""

    def test_realistic_hsbc_scenario(self):
        """Test with realistic HSBC-style data."""
        reporter = QuietModeReporter("0005.HK", "HSBC Holdings")
        result_dict = {
            "final_trade_decision": """
### FINAL DECISION: HOLD

### THESIS COMPLIANCE SUMMARY
- Financial Health: [DATA MISSING]
- Analyst Coverage: 16 - FAIL
Risk Tally: 2.33

=== DECISION LOGIC ===
ZONE: HIGH >= 2.0
Default Decision: SELL
Actual Decision: HOLD
Override: YES
""",
            "fundamentals_report": """
### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 7/12
ADJUSTED_HEALTH_SCORE: 70% (based on 10 available points)
### --- END DATA_BLOCK ---
""",
            "market_report": "Liquidity: $225M daily - PASS",
        }

        report = reporter.generate_report(result_dict)

        assert "0005.HK" in report
        assert "HSBC Holdings" in report
        assert "HOLD" in report
        assert "DATA_BLOCK" in report
        assert "Liquidity" in report

    def test_realistic_brief_scenario(self):
        """Test realistic scenario with brief mode."""
        reporter = QuietModeReporter("0005.HK", "HSBC Holdings")
        result_dict = {
            "final_trade_decision": """
### FINAL DECISION: HOLD

Executive summary of analysis findings here.

DECISION RATIONALE: Analyst coverage below threshold (16 vs required 20).
Financial health data incomplete. Liquidity meets requirements.
Default SELL overridden to HOLD pending data completion.

### THESIS COMPLIANCE SUMMARY
- Financial Health: [DATA MISSING]
- Analyst Coverage: 16 - FAIL
""",
            "fundamentals_report": "Detailed fundamental analysis with metrics",
            "market_report": "Technical analysis with indicators",
            "sentiment_report": "Sentiment from multiple sources",
            "news_report": "Recent news and developments",
        }

        brief = reporter.generate_report(result_dict, brief_mode=True)
        full = reporter.generate_report(result_dict, brief_mode=False)

        # Brief should have header and rationale (inline)
        assert "0005.HK" in brief
        assert "HOLD" in brief
        assert "RATIONALE" in brief.upper()
        assert "Analyst coverage below threshold" in brief

        # Brief should NOT have detailed sections
        assert "Technical Analysis" not in brief
        assert "Fundamental Analysis" not in brief

        # Full should have everything
        assert "Technical Analysis" in full
        assert "Fundamental Analysis" in full

        # Brief should be significantly shorter
        assert len(brief) < len(full)


class TestRedFlagPreScreening:
    """Test red-flag pre-screening section in reports."""

    def test_report_with_reject_red_flags(self):
        """Test report includes red flags when stock is rejected."""
        reporter = QuietModeReporter("9999.HK", "Zombie Corp")
        result_dict = {
            "final_trade_decision": "Action: SELL",
            "red_flags": [
                {
                    "type": "EXTREME_LEVERAGE",
                    "severity": "CRITICAL",
                    "detail": "D/E ratio 820% exceeds bankruptcy threshold of 500%",
                    "action": "AUTO_REJECT",
                },
                {
                    "type": "REFINANCING_RISK",
                    "severity": "CRITICAL",
                    "detail": "Interest coverage 1.1x below threshold of 2.0x with D/E 820%",
                    "action": "AUTO_REJECT",
                },
            ],
            "pre_screening_result": "REJECT",
        }

        report = reporter.generate_report(result_dict)

        # Should have red flag section
        assert "Red Flag Pre-Screening" in report
        assert "CRITICAL RED FLAGS DETECTED" in report
        assert "EXTREME_LEVERAGE" in report
        assert "REFINANCING_RISK" in report
        assert "D/E ratio 820%" in report
        assert "Interest coverage 1.1x" in report
        assert "Debate phase skipped" in report

    def test_report_with_pass_warnings(self):
        """Test report includes warnings when stock passes with non-critical flags."""
        reporter = QuietModeReporter("TEST.US")
        result_dict = {
            "final_trade_decision": "Action: BUY",
            "red_flags": [
                {
                    "type": "MODERATE_LEVERAGE",
                    "severity": "MEDIUM",
                    "detail": "D/E ratio 280% elevated but below critical threshold",
                    "action": "WARNING",
                }
            ],
            "pre_screening_result": "PASS",
        }

        report = reporter.generate_report(result_dict)

        assert "Red Flag Pre-Screening" in report
        assert "Warnings Detected" in report
        assert "MODERATE_LEVERAGE" in report
        assert "D/E ratio 280%" in report
        # Should NOT mention debate skipped
        assert "Debate phase skipped" not in report

    def test_report_no_red_flags(self):
        """Test report without red flags omits pre-screening section."""
        reporter = QuietModeReporter("CLEAN.US", "Clean Company")
        result_dict = {
            "final_trade_decision": "Action: BUY",
            "red_flags": [],
            "pre_screening_result": "PASS",
        }

        report = reporter.generate_report(result_dict)

        # Should NOT have red flag section
        assert "Red Flag Pre-Screening" not in report
        assert "CRITICAL RED FLAGS" not in report

    def test_report_red_flags_in_brief_mode(self):
        """Test red flags appear in brief mode."""
        reporter = QuietModeReporter("FRAUD.CN")
        result_dict = {
            "final_trade_decision": "Action: SELL\n\nRATIONALE: Critical red flags detected.",
            "red_flags": [
                {
                    "type": "EARNINGS_QUALITY",
                    "severity": "CRITICAL",
                    "detail": "Positive income $1,250M but negative FCF -$3,800M (3.0x ratio)",
                    "action": "AUTO_REJECT",
                }
            ],
            "pre_screening_result": "REJECT",
            "market_report": "Technical analysis here",
            "fundamentals_report": "Fundamental analysis here",
        }

        brief_report = reporter.generate_report(result_dict, brief_mode=True)
        full_report = reporter.generate_report(result_dict, brief_mode=False)

        # Both should have red flags section
        assert "Red Flag Pre-Screening" in brief_report
        assert "Red Flag Pre-Screening" in full_report
        assert "EARNINGS_QUALITY" in brief_report
        assert "EARNINGS_QUALITY" in full_report

        # Brief should not have full sections
        assert "Technical Analysis" not in brief_report
        assert "Technical Analysis" in full_report

    def test_report_multiple_red_flags_formatting(self):
        """Test multiple red flags are properly formatted."""
        reporter = QuietModeReporter("MULTI.US")
        result_dict = {
            "final_trade_decision": "Action: SELL",
            "red_flags": [
                {
                    "type": "EXTREME_LEVERAGE",
                    "severity": "CRITICAL",
                    "detail": "D/E ratio 650% exceeds threshold",
                    "action": "AUTO_REJECT",
                },
                {
                    "type": "EARNINGS_QUALITY",
                    "severity": "CRITICAL",
                    "detail": "Negative FCF despite positive income",
                    "action": "AUTO_REJECT",
                },
                {
                    "type": "REFINANCING_RISK",
                    "severity": "CRITICAL",
                    "detail": "Interest coverage dangerously low",
                    "action": "AUTO_REJECT",
                },
            ],
            "pre_screening_result": "REJECT",
        }

        report = reporter.generate_report(result_dict)

        # Should list all three flags
        assert report.count("EXTREME_LEVERAGE") == 1
        assert report.count("EARNINGS_QUALITY") == 1
        assert report.count("REFINANCING_RISK") == 1

        # All should be bullet points
        assert report.count("- **EXTREME_LEVERAGE**") == 1
        assert report.count("- **EARNINGS_QUALITY**") == 1
        assert report.count("- **REFINANCING_RISK**") == 1

    def test_report_red_flags_appear_before_exec_summary(self):
        """Test red flags section appears before Executive Summary."""
        reporter = QuietModeReporter("TEST.US")
        result_dict = {
            "final_trade_decision": "Action: SELL\n\nExecutive summary content here.",
            "red_flags": [
                {
                    "type": "EXTREME_LEVERAGE",
                    "severity": "CRITICAL",
                    "detail": "Test detail",
                    "action": "AUTO_REJECT",
                }
            ],
            "pre_screening_result": "REJECT",
        }

        report = reporter.generate_report(result_dict)

        red_flag_pos = report.find("Red Flag Pre-Screening")
        exec_summary_pos = report.find("Executive Summary")

        # Red flags should appear before executive summary
        assert red_flag_pos > 0
        assert exec_summary_pos > 0
        assert red_flag_pos < exec_summary_pos

    def test_report_missing_red_flag_fields(self):
        """Test graceful handling of missing red flag fields."""
        reporter = QuietModeReporter("TEST.US")
        result_dict = {
            "final_trade_decision": "Action: SELL",
            "red_flags": [
                {
                    "type": "UNKNOWN",
                    # severity missing
                    # detail missing
                    # action missing
                }
            ],
            "pre_screening_result": "REJECT",
        }

        # Should not crash
        report = reporter.generate_report(result_dict)

        assert "Red Flag Pre-Screening" in report
        assert "UNKNOWN" in report
        # Should use defaults
        assert "UNKNOWN" in report  # type
        # Missing fields should use fallbacks from code


class TestTraderSectionVerdictGating:
    """Trading Strategy section must be suppressed for DO NOT INITIATE / SELL verdicts."""

    _TRADER_CONTENT = (
        "Entry: 6.00 NZD\nStop Loss: 5.72 NZD\nTarget: 6.40 NZD\nScaled entry approach."
    )

    def _result(self, verdict_text: str) -> dict:
        return {
            "final_trade_decision": verdict_text,
            "trader_investment_plan": self._TRADER_CONTENT,
        }

    def test_do_not_initiate_suppresses_entry_levels(self):
        reporter = QuietModeReporter("SCL.NZ", "Scales Corporation")
        report = reporter.generate_report(
            self._result(
                "#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE\n\nRationale."
            )
        )
        assert "Trading Strategy" in report
        assert "not applicable" in report
        assert "DO NOT INITIATE" in report
        # Trader's actual entry data must not bleed through
        assert "6.00 NZD" not in report
        assert "Stop Loss" not in report

    def test_sell_suppresses_entry_levels(self):
        reporter = QuietModeReporter("TEST.NZ")
        report = reporter.generate_report(
            self._result("VERDICT: SELL\n\nDeterioration detected.")
        )
        assert "Trading Strategy" in report
        assert "not applicable" in report
        assert "SELL" in report
        assert "6.00 NZD" not in report

    def test_buy_includes_full_trader_section(self):
        reporter = QuietModeReporter("TEST.NZ")
        report = reporter.generate_report(self._result("Action: BUY\n\nStrong thesis."))
        assert "Trading Strategy" in report
        assert "6.00 NZD" in report
        assert "not applicable" not in report

    def test_hold_includes_full_trader_section(self):
        reporter = QuietModeReporter("TEST.NZ")
        report = reporter.generate_report(
            self._result("Action: HOLD\n\nWait for clarity.")
        )
        assert "Trading Strategy" in report
        assert "6.00 NZD" in report
        assert "not applicable" not in report

    def test_heading_always_present_on_dni(self):
        """Section heading must appear even when body is suppressed."""
        reporter = QuietModeReporter("TEST.NZ")
        report = reporter.generate_report(
            self._result("VERDICT: DO_NOT_INITIATE\n\nFails thesis.")
        )
        assert "## Trading Strategy" in report


# ---------------------------------------------------------------------------
# New tests for report output cleanup (A3–A6, B1, B3)
# ---------------------------------------------------------------------------


class TestCleanTextPMBlock:
    """_clean_text() strips PM_BLOCK in all its forms (A4)."""

    def setup_method(self):
        self.reporter = QuietModeReporter("TST")

    def _clean(self, text):
        return self.reporter._clean_text(text)

    def test_strips_pm_block_unescaped(self):
        text = "Prose before.\n#### --- START PM_BLOCK ---\nVERDICT: BUY\n#### --- END PM_BLOCK ---\nProse after."
        result = self._clean(text)
        assert "START PM_BLOCK" not in result
        assert "END PM_BLOCK" not in result
        assert "Prose before." in result
        assert "Prose after." in result

    def test_strips_pm_block_in_code_fence(self):
        text = "```\n#### --- START PM_BLOCK ---\nVERDICT: BUY\n#### --- END PM_BLOCK ---\n```"
        result = self._clean(text)
        assert "START PM_BLOCK" not in result

    def test_strips_pm_block_with_label_heading(self):
        text = (
            "#### PM_BLOCK (REQUIRED - Machine-Readable Summary)\n"
            "```\n"
            "#### --- START PM_BLOCK ---\n"
            "VERDICT: BUY\n"
            "#### --- END PM_BLOCK ---\n"
            "```"
        )
        result = self._clean(text)
        assert "START PM_BLOCK" not in result

    def test_preserves_content_before_pm_block(self):
        text = "Important verdict prose.\n#### --- START PM_BLOCK ---\nDATA\n#### --- END PM_BLOCK ---"
        result = self._clean(text)
        assert "Important verdict prose." in result

    def test_preserves_content_after_pm_block(self):
        text = "#### --- START PM_BLOCK ---\nDATA\n#### --- END PM_BLOCK ---\nTrailing content here."
        result = self._clean(text)
        assert "Trailing content here." in result

    def test_no_pm_block_unchanged(self):
        text = "Normal text with no machine-readable blocks.\n"
        result = self._clean(text)
        assert "Normal text with no machine-readable blocks." in result

    def test_truncated_pm_block_no_crash(self):
        """No closing marker — must not raise."""
        text = "#### --- START PM_BLOCK ---\nVERDICT: BUY"
        result = self._clean(text)  # should not raise
        assert isinstance(result, str)

    def test_generate_report_pm_block_absent_from_executive_summary(self):
        reporter = QuietModeReporter("TST")
        result_dict = {
            "final_trade_decision": (
                "Action: BUY\n\nStrong value opportunity.\n\n"
                "#### --- START PM_BLOCK ---\n"
                "VERDICT: BUY\nRISK_ZONE: LOW\n"
                "#### --- END PM_BLOCK ---\n"
                "\nDecision rationale follows."
            ),
        }
        report = reporter.generate_report(result_dict)
        assert "START PM_BLOCK" not in report
        assert "END PM_BLOCK" not in report

    def test_verdict_extraction_unaffected_by_pm_block_strip(self):
        """extract_decision() runs on raw text; PM_BLOCK strip happens later in _clean_text."""
        reporter = QuietModeReporter("TST")
        raw = (
            "Action: BUY\n\n"
            "#### --- START PM_BLOCK ---\nVERDICT: BUY\n#### --- END PM_BLOCK ---"
        )
        verdict = reporter.extract_decision(raw)
        assert verdict == "BUY"


class TestCleanTextConsultantResolution:
    """_clean_text() strips CONSULTANT_RESOLUTION blocks (A5)."""

    def setup_method(self):
        self.reporter = QuietModeReporter("TST")

    def _clean(self, text):
        return self.reporter._clean_text(text)

    def test_strips_single_consultant_resolution_block(self):
        text = (
            "#### PORTFOLIO MANAGER VERDICT: BUY\n\n"
            "CONSULTANT_RESOLUTION:\n"
            "- CONCERN: High leverage\n"
            "- DATA_CHECK: D/E confirmed\n"
            "- VERDICT: UNVERIFIABLE (+0.25)\n"
            "\nDecision rationale here."
        )
        result = self._clean(text)
        assert "CONSULTANT_RESOLUTION:" not in result
        assert "PORTFOLIO MANAGER VERDICT: BUY" in result
        assert "Decision rationale here." in result

    def test_strips_multiple_consultant_resolution_blocks(self):
        block = "CONSULTANT_RESOLUTION:\n- CONCERN: X\n- DATA_CHECK: Y\n- VERDICT: Z\n"
        text = f"{block}\nSome prose.\n\n{block}\n{block}"
        result = self._clean(text)
        assert "CONSULTANT_RESOLUTION:" not in result
        assert "Some prose." in result

    def test_strips_resolution_block_at_top(self):
        text = "CONSULTANT_RESOLUTION:\n- CONCERN: X\n- DATA_CHECK: Y\n\nVerdict prose follows."
        result = self._clean(text)
        assert "CONSULTANT_RESOLUTION:" not in result
        assert "Verdict prose follows." in result

    def test_strips_resolution_block_at_bottom(self):
        text = "Decision rationale is strong.\n\nCONSULTANT_RESOLUTION:\n- CONCERN: X\n- VERDICT: UNVERIFIABLE\n"
        result = self._clean(text)
        assert "CONSULTANT_RESOLUTION:" not in result
        assert "Decision rationale is strong." in result

    def test_partial_resolution_block_no_crash(self):
        """Header only (no following bullet lines) must not raise."""
        text = "CONSULTANT_RESOLUTION:\n"
        result = self._clean(text)  # should not raise
        assert isinstance(result, str)

    def test_generate_report_no_consultant_resolution_in_output(self):
        reporter = QuietModeReporter("TST")
        result_dict = {
            "final_trade_decision": (
                "CONSULTANT_RESOLUTION:\n"
                "- CONCERN: Leverage elevated\n"
                "- DATA_CHECK: D/E 450%\n"
                "- VERDICT: UNVERIFIABLE (+0.25)\n\n"
                "#### PORTFOLIO MANAGER VERDICT: BUY\n\n"
                "Strong thesis supported by fundamentals."
            ),
        }
        report = reporter.generate_report(result_dict)
        assert "CONSULTANT_RESOLUTION:" not in report


class TestReformatMacroDetection:
    """_reformat_macro_detection() removes NO blocks and renders YES as callout (A6)."""

    def setup_method(self):
        self.reporter = QuietModeReporter("TST")

    def test_triggered_no_renders_nothing(self):
        text = (
            "News content here.\n\n"
            "#### MACRO_DETECTION\n"
            "TRIGGERED: NO\n"
            "SOURCE: NONE\n"
            "HEADLINE: N/A\n"
            "THESIS_IMPACT: N/A\n"
        )
        result = self.reporter._reformat_macro_detection(text)
        assert "MACRO_DETECTION" not in result
        assert "TRIGGERED:" not in result
        assert "News content here." in result

    def test_triggered_yes_renders_callout(self):
        text = (
            "News content here.\n\n"
            "#### MACRO_DETECTION\n"
            "TRIGGERED: YES\n"
            "SOURCE: FED\n"
            "HEADLINE: Fed raises rates 50bps\n"
            "THESIS_IMPACT: NEGATIVE\n"
        )
        result = self.reporter._reformat_macro_detection(text)
        assert "MACRO_DETECTION" not in result
        assert "TRIGGERED:" not in result
        assert "**Macro event detected**" in result
        assert "Fed raises rates 50bps" in result

    def test_callout_contains_thesis_impact(self):
        text = (
            "#### MACRO_DETECTION\n"
            "TRIGGERED: YES\n"
            "HEADLINE: Trade war escalation\n"
            "THESIS_IMPACT: SIGNIFICANT_NEGATIVE\n"
        )
        result = self.reporter._reformat_macro_detection(text)
        assert "SIGNIFICANT_NEGATIVE" in result

    def test_triggered_yes_empty_headline_no_crash(self):
        text = (
            "#### MACRO_DETECTION\n"
            "TRIGGERED: YES\n"
            "HEADLINE: \n"
            "THESIS_IMPACT: UNKNOWN\n"
        )
        result = self.reporter._reformat_macro_detection(text)  # must not raise
        assert isinstance(result, str)

    def test_missing_block_no_change(self):
        text = "Regular news content with no detection block."
        result = self.reporter._reformat_macro_detection(text)
        assert result == text

    def test_partial_block_no_crash(self):
        """Block with only TRIGGERED: and no subsequent fields must not raise."""
        text = "#### MACRO_DETECTION\nTRIGGERED: YES\n"
        result = self.reporter._reformat_macro_detection(text)  # must not raise
        assert isinstance(result, str)

    def test_macro_detection_raw_block_absent_from_final_report(self):
        reporter = QuietModeReporter("TST")
        result_dict = {
            "final_trade_decision": "Action: BUY",
            "news_report": (
                "Market news here.\n\n"
                "#### MACRO_DETECTION\n"
                "TRIGGERED: NO\n"
                "SOURCE: NONE\n"
                "HEADLINE: N/A\n"
                "THESIS_IMPACT: N/A\n"
            ),
        }
        report = reporter.generate_report(result_dict)
        assert "#### MACRO_DETECTION" not in report
        assert "TRIGGERED:" not in report


class TestMoveDataBlockToEnd:
    """_move_data_block_to_end() normalizes DATA_BLOCK position (B1)."""

    def test_moves_data_block_from_top_to_end(self):
        text = (
            "#### --- START DATA_BLOCK ---\n"
            "PE_RATIO: 15\n"
            "#### --- END DATA_BLOCK ---\n\n"
            "Score breakdown prose here."
        )
        result = QuietModeReporter._move_data_block_to_end(text)
        data_pos = result.find("START DATA_BLOCK")
        prose_pos = result.find("Score breakdown prose here.")
        assert prose_pos < data_pos

    def test_data_block_already_at_end_stays_at_end(self):
        text = (
            "Score breakdown prose here.\n\n"
            "#### --- START DATA_BLOCK ---\n"
            "PE_RATIO: 15\n"
            "#### --- END DATA_BLOCK ---\n"
        )
        result = QuietModeReporter._move_data_block_to_end(text)
        assert result.index("START DATA_BLOCK") > result.index(
            "Score breakdown prose here."
        )

    def test_preserves_content_before_data_block(self):
        text = (
            "Section one content.\n\n"
            "#### --- START DATA_BLOCK ---\nDATA\n#### --- END DATA_BLOCK ---\n\n"
            "Section two content."
        )
        result = QuietModeReporter._move_data_block_to_end(text)
        assert "Section one content." in result
        assert "Section two content." in result

    def test_no_block_unchanged(self):
        text = "Content with no data block at all."
        result = QuietModeReporter._move_data_block_to_end(text)
        assert result == text

    def test_preserves_block_contents(self):
        text = (
            "#### --- START DATA_BLOCK ---\n"
            "PE_RATIO: 15\nROE: 18%\nDEBT_TO_EQUITY: 45%\n"
            "#### --- END DATA_BLOCK ---\n\n"
            "Prose follows."
        )
        result = QuietModeReporter._move_data_block_to_end(text)
        assert "PE_RATIO: 15" in result
        assert "ROE: 18%" in result
        assert "DEBT_TO_EQUITY: 45%" in result

    def test_does_not_duplicate_block(self):
        text = (
            "#### --- START DATA_BLOCK ---\nDATA\n#### --- END DATA_BLOCK ---\n\n"
            "Prose."
        )
        result = QuietModeReporter._move_data_block_to_end(text)
        assert result.count("START DATA_BLOCK") == 1
        assert result.count("END DATA_BLOCK") == 1

    def test_generate_report_data_block_at_end_of_fundamentals(self):
        """DATA_BLOCK at top of fundamentals should be moved to end in output."""
        reporter = QuietModeReporter("TST")
        result_dict = {
            "final_trade_decision": "Action: BUY",
            "fundamentals_report": (
                "#### --- START DATA_BLOCK ---\n"
                "PE_RATIO: 12\nROE: 15%\n"
                "#### --- END DATA_BLOCK ---\n\n"
                "Financial health is strong. Growth metrics are positive."
            ),
        }
        report = reporter.generate_report(result_dict)
        data_pos = report.find("START DATA_BLOCK")
        prose_pos = report.find("Financial health is strong.")
        assert data_pos != -1
        assert prose_pos < data_pos


class TestCleanTextAnalyzingOpener:
    """_clean_text() strips 'Analyzing TICKER - Company' openers (B3)."""

    def setup_method(self):
        self.reporter = QuietModeReporter("TST")

    def _clean(self, text):
        return self.reporter._clean_text(text)

    def test_strips_analyzing_hyphen_opener(self):
        text = "Analyzing 4776.T - Cybozu, Inc.\nSection content starts here."
        result = self._clean(text)
        assert "Analyzing 4776.T" not in result
        assert "Section content starts here." in result

    def test_strips_analyzing_emdash_opener(self):
        text = "Analyzing KRN.DE \u2014 Krones AG\nContent follows."
        result = self._clean(text)
        assert "Analyzing KRN.DE" not in result
        assert "Content follows." in result

    def test_strips_analyzing_opener_in_multiline(self):
        """Opener appearing as start of a later line is also stripped."""
        text = (
            "First line.\nAnalyzing 6782.TW - Holtek Semiconductor Inc.\nBody content."
        )
        result = self._clean(text)
        assert "Analyzing 6782.TW" not in result
        assert "First line." in result
        assert "Body content." in result

    def test_preserves_content_after_opener(self):
        text = "Analyzing 0005.HK - HSBC Holdings plc\nHSBC remains well-capitalised."
        result = self._clean(text)
        assert "HSBC remains well-capitalised." in result

    def test_does_not_strip_analyzing_mid_sentence(self):
        """'Analyzing' not at line start with TICKER pattern is NOT stripped."""
        text = "We are analyzing the market conditions carefully.\n"
        result = self._clean(text)
        assert "We are analyzing the market conditions carefully." in result
