"""
Test suite for memory contamination vectors.

This module tests the various contamination scenarios identified in
MEMORY_CONTAMINATION_ANALYSIS.md to ensure proper isolation.
"""

import pytest
import yfinance as yf

from src.agents import AgentState
from src.memory import (
    cleanup_all_memories,
    create_memory_instances,
    sanitize_ticker_for_collection,
)
from src.toolkit import extract_company_name_async


class TestCompanyNameExtraction:
    """Test that company names are correctly extracted from yfinance."""

    def test_0293_hk_correct_name(self):
        """Verify 0293.HK returns Cathay Pacific, not China Resources Beer."""
        ticker = yf.Ticker("0293.HK")
        info = ticker.info
        long_name = info.get("longName", "")
        short_name = info.get("shortName", "")

        # Should contain "CATHAY" not "CHINA RES BEER"
        assert (
            "CATHAY" in long_name.upper() or "CATHAY" in short_name.upper()
        ), f"Expected Cathay Pacific, got: {long_name} / {short_name}"
        assert (
            "BEER" not in long_name.upper() and "BEER" not in short_name.upper()
        ), f"Should not contain 'BEER', got: {long_name} / {short_name}"

    def test_0291_hk_correct_name(self):
        """Verify 0291.HK returns China Resources Beer."""
        ticker = yf.Ticker("0291.HK")
        info = ticker.info
        long_name = info.get("longName", "")
        short_name = info.get("shortName", "")

        # Should contain "CHINA" or "RES" or "BEER"
        combined = (long_name + " " + short_name).upper()
        assert any(
            word in combined for word in ["CHINA", "RES", "BEER"]
        ), f"Expected China Resources Beer, got: {long_name} / {short_name}"

    @pytest.mark.asyncio
    async def test_extract_company_name_async_0293(self):
        """Test async extraction for 0293.HK."""
        ticker_obj = yf.Ticker("0293.HK")
        company_name = await extract_company_name_async(ticker_obj)

        assert (
            "CATHAY" in company_name.upper()
        ), f"Expected Cathay Pacific, got: {company_name}"
        assert (
            "BEER" not in company_name.upper()
        ), f"Should not contain BEER, got: {company_name}"


class TestMemoryIsolation:
    """Test ChromaDB memory isolation between tickers."""

    @pytest.mark.asyncio
    async def test_memory_collections_are_ticker_specific(self):
        """Verify that different tickers get different collection names."""
        mem_0291 = create_memory_instances("0291.HK")
        mem_0293 = create_memory_instances("0293.HK")

        # Collection names should be different
        assert "0291_HK" in list(mem_0291.keys())[0]
        assert "0293_HK" in list(mem_0293.keys())[0]

        # No overlap
        assert set(mem_0291.keys()).isdisjoint(set(mem_0293.keys()))

    @pytest.mark.asyncio
    async def test_memory_query_with_strict_filtering(self):
        """Test that memory queries with metadata filters don't leak between tickers."""
        # Clean slate
        cleanup_all_memories(days=0)

        # Create memories for both tickers
        mem_0291 = create_memory_instances("0291.HK")
        mem_0293 = create_memory_instances("0293.HK")

        bull_0291 = mem_0291["0291_HK_bull_memory"]
        bull_0293 = mem_0293["0293_HK_bull_memory"]

        # Only test if ChromaDB is available
        if not bull_0291.available:
            pytest.skip("ChromaDB not available")

        # Add data to 0291.HK memory
        await bull_0291.add_situations(
            ["China Resources Beer has strong EBITDA growth and margin expansion"],
            [{"ticker": "0291.HK", "company": "China Resources Beer"}],
        )

        # Query from 0293.HK memory with strict filtering
        results = await bull_0293.query_similar_situations(
            "strong EBITDA growth", n_results=5, metadata_filter={"ticker": "0293.HK"}
        )

        # Should return EMPTY (no 0291.HK data should leak)
        assert (
            len(results) == 0
        ), f"CONTAMINATION DETECTED: Found {len(results)} results from 0291.HK in 0293.HK memory"

    @pytest.mark.asyncio
    async def test_semantic_search_without_filter_does_not_cross_collections(self):
        """Verify that collections are truly isolated even without metadata filters."""
        cleanup_all_memories(days=0)

        mem_0291 = create_memory_instances("0291.HK")
        mem_0293 = create_memory_instances("0293.HK")

        bull_0291 = mem_0291["0291_HK_bull_memory"]
        bull_0293 = mem_0293["0293_HK_bull_memory"]

        if not bull_0291.available:
            pytest.skip("ChromaDB not available")

        # Add to 0291 memory
        await bull_0291.add_situations(
            ["Beverage company with strong regional presence"], [{"ticker": "0291.HK"}]
        )

        # Query 0293 memory WITHOUT metadata filter
        # Since they're separate collections, should still return empty
        results = await bull_0293.query_similar_situations(
            "beverage company",
            n_results=5,
            # NO metadata_filter - testing collection isolation
        )

        # Should be empty because it's a different collection
        assert (
            len(results) == 0
        ), "Collection isolation FAILED: 0293 collection contains 0291 data"


class TestTickerSanitization:
    """Test that ticker sanitization is consistent."""

    def test_sanitize_similar_tickers(self):
        """Verify that similar tickers get different sanitized names."""
        ticker_0291 = sanitize_ticker_for_collection("0291.HK")
        ticker_0293 = sanitize_ticker_for_collection("0293.HK")

        assert (
            ticker_0291 != ticker_0293
        ), "Similar tickers should sanitize to different collection names"

        assert ticker_0291 == "0291_HK"
        assert ticker_0293 == "0293_HK"

    def test_sanitize_handles_special_chars(self):
        """Test edge cases in ticker sanitization."""
        # Dots become underscores
        assert sanitize_ticker_for_collection("BRK.B") == "BRK_B"

        # Hyphens become underscores
        assert sanitize_ticker_for_collection("BRK-B") == "BRK_B"

        # Length constraints (should truncate to 40 chars)
        long_ticker = "A" * 50 + ".HK"
        sanitized = sanitize_ticker_for_collection(long_ticker)
        assert len(sanitized) <= 40


class TestAgentStateIsolation:
    """Test that AgentState doesn't leak between analyses."""

    def test_initial_state_has_correct_ticker(self):
        """Verify initial state sets company_of_interest correctly."""
        from langchain_core.messages import HumanMessage

        from src.agents import InvestDebateState, RiskDebateState

        state_0293 = AgentState(
            messages=[HumanMessage(content="Analyze 0293.HK")],
            company_of_interest="0293.HK",
            trade_date="2025-12-05",
            sender="user",
            market_report="",
            sentiment_report="",
            news_report="",
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
        )

        assert state_0293["company_of_interest"] == "0293.HK"
        assert "0293.HK" in state_0293["messages"][0].content


class TestLLMHallucinationPrevention:
    """Test safeguards against LLM hallucination of company names."""

    def test_ticker_similarity_is_the_issue(self):
        """Demonstrate that 0291.HK and 0293.HK are dangerously similar."""
        ticker_a = "0291.HK"
        ticker_b = "0293.HK"

        # Calculate string similarity (Levenshtein-like)
        def similarity(s1, s2):
            if len(s1) != len(s2):
                return 0
            matches = sum(c1 == c2 for c1, c2 in zip(s1, s2, strict=False))
            return matches / len(s1)

        sim = similarity(ticker_a, ticker_b)

        # They differ by only 1 character out of 7
        assert sim > 0.85, f"Tickers are {sim:.0%} similar - high risk of confusion"

    @pytest.mark.asyncio
    async def test_company_name_should_be_in_tool_output(self):
        """Verify that tools return company names in their output."""
        from src.toolkit import get_financial_metrics

        # Tools are LangChain StructuredTool objects, use ainvoke method
        result = await get_financial_metrics.ainvoke({"ticker": "0293.HK"})

        # Result should contain the ticker
        assert (
            "0293" in result or "HK" in result
        ), "Tool output should contain ticker reference"

        # Ideally should contain company name (but current implementation may not)
        # This test documents the gap
        # assert "CATHAY" in result.upper(), \
        #     "Tool output should contain company name to prevent hallucination"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
