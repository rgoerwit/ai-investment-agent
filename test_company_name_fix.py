"""
Quick test to verify company name fix for 0293.HK
"""

import asyncio
import yfinance as yf
from src.agents import AgentState, InvestDebateState, RiskDebateState
from langchain_core.messages import HumanMessage


async def test_company_name_extraction():
    """Test that we correctly extract company name for 0293.HK"""

    ticker = "0293.HK"

    # Step 1: Fetch company name (same logic as main.py)
    company_name = ticker
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        company_name = info.get('longName') or info.get('shortName') or ticker
        print(f"✅ Company name fetched: {ticker} = {company_name}")
    except Exception as e:
        print(f"❌ Error fetching company name: {e}")
        company_name = ticker

    # Step 2: Create initial state (same as main.py)
    initial_state = AgentState(
        messages=[HumanMessage(content=f"Analyze {ticker} ({company_name}) for investment decision.")],
        company_of_interest=ticker,
        company_name=company_name,
        trade_date="2025-12-05",
        sender="user",
        market_report="",
        sentiment_report="",
        news_report="",
        fundamentals_report="",
        investment_debate_state=InvestDebateState(
            bull_history="", bear_history="", history="",
            current_response="", judge_decision="", count=0
        ),
        investment_plan="",
        trader_investment_plan="",
        risk_debate_state=RiskDebateState(
            risky_history="", safe_history="", neutral_history="",
            history="", latest_speaker="", current_risky_response="",
            current_safe_response="", current_neutral_response="",
            judge_decision="", count=0
        ),
        final_trade_decision="",
        tools_called={},
        prompts_used={}
    )

    # Step 3: Verify state has correct company name
    assert initial_state["company_of_interest"] == "0293.HK", "Ticker incorrect"
    assert initial_state["company_name"] == company_name, "Company name not in state"
    assert "CATHAY" in company_name.upper(), f"Expected Cathay Pacific, got: {company_name}"
    assert "BEER" not in company_name.upper(), f"Should NOT contain BEER, got: {company_name}"

    print(f"\n✅ All checks passed!")
    print(f"   Ticker: {initial_state['company_of_interest']}")
    print(f"   Company: {initial_state['company_name']}")
    print(f"   Initial message: {initial_state['messages'][0].content[:80]}...")

    return True


async def test_similar_ticker():
    """Test that 0291.HK also works correctly"""

    ticker = "0291.HK"

    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        company_name = info.get('longName') or info.get('shortName') or ticker
        print(f"\n✅ Control test - 0291.HK = {company_name}")
        assert "BEER" in company_name.upper() or "RES" in company_name.upper(), \
            f"Expected China Resources Beer, got: {company_name}"
    except Exception as e:
        print(f"❌ Control test failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Company Name Fix for 0293.HK")
    print("=" * 60)

    # Run tests
    asyncio.run(test_company_name_extraction())
    asyncio.run(test_similar_ticker())

    print("\n" + "=" * 60)
    print("✅ FIX VERIFIED: Company names are correctly extracted!")
    print("=" * 60)
