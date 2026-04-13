"""Market and fundamentals tool implementations."""

import json
from typing import Annotated

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from stockstats import wrap as stockstats_wrap

from src.data.fetcher import fetcher as market_data_fetcher
from src.ticker_utils import normalize_ticker
from src.tools import shared


def extract_from_dataframe(
    df: pd.DataFrame, field_name: str, row_index: int = 0
) -> float | None:
    if df is None or df.empty:
        return None
    try:
        if field_name in df.index:
            val = df.loc[field_name].iloc[row_index]
            return float(val) if not pd.isna(val) else None
        return None
    except Exception:
        return None


@tool
async def get_financial_metrics(
    ticker: Annotated[
        str,
        "Exact ticker with exchange suffix — e.g. '3217.TWO' (Taiwan OTC/TPEx),"
        " '7203.T' (Japan), '0005.HK' (HK), '2330.TW' (Taiwan TWSE)."
        " Use exactly as provided; never alter or drop the suffix.",
    ],
) -> str:
    """Get key financial ratios and metrics as a JSON string.

    Currency note: price fields (currentPrice, 52-week range, moving averages) are
    returned in the stock's LOCAL TRADING CURRENCY (JPY for .T, HKD for .HK, etc.).
    Ratio/percentage fields (PE, PB, ROA, margins) are currency-neutral.
    The 'currency' field in the response identifies the local currency code.
    """
    try:
        normalized_symbol = normalize_ticker(ticker)
        data = await market_data_fetcher.get_financial_metrics(normalized_symbol)

        if "error" in data:
            return json.dumps({"error": data.get("error")})

        sanitized_data = shared._sanitize_for_json(data)
        return json.dumps(sanitized_data, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
async def get_yfinance_data(
    symbol: Annotated[
        str,
        "Exact ticker with exchange suffix — e.g. '3217.TWO', '7203.T', '0005.HK'."
        " Use exactly as provided; never alter or drop the suffix.",
    ],
    start_date: str = None,
    end_date: str = None,
) -> str:
    """Get historical stock price data in LOCAL TRADING CURRENCY (not USD)."""
    try:
        normalized = normalize_ticker(symbol)
        hist = await market_data_fetcher.get_historical_prices(
            normalized,
            start=start_date,
            end=end_date,
        )
        if hist.empty:
            return "No data"
        return hist.reset_index().to_csv(index=False)
    except Exception as exc:
        return f"Error: {exc}"


@tool
async def get_technical_indicators(
    symbol: Annotated[
        str,
        "Exact ticker with exchange suffix — e.g. '3217.TWO', '7203.T', '0005.HK'."
        " Use exactly as provided; never alter or drop the suffix.",
    ],
) -> str:
    """Get RSI, MACD, Bollinger Bands, and Moving Averages."""
    try:
        normalized = normalize_ticker(symbol)
        hist = await market_data_fetcher.get_historical_prices(normalized, period="2y")

        if hist.empty:
            return "No data"

        stock = stockstats_wrap(hist)
        latest = hist.iloc[-1]
        data_points = len(hist)

        sma_50 = (
            shared._safe_float(stock["close_50_sma"].iloc[-1])
            if data_points >= 50
            else None
        )
        sma_200 = (
            shared._safe_float(stock["close_200_sma"].iloc[-1])
            if data_points >= 200
            else None
        )
        rsi_14 = (
            shared._safe_float(stock["rsi_14"].iloc[-1]) if data_points >= 14 else None
        )
        macd = shared._safe_float(stock["macd"].iloc[-1]) if data_points >= 26 else None
        boll_ub = (
            shared._safe_float(stock["boll_ub"].iloc[-1]) if data_points >= 20 else None
        )
        boll_lb = (
            shared._safe_float(stock["boll_lb"].iloc[-1]) if data_points >= 20 else None
        )

        def fmt(val):
            return shared._format_val(val)

        return (
            f"Technical Indicators for {symbol}:\n"
            f"Current Price: {fmt(latest['Close'])}\n"
            f"RSI (14): {fmt(rsi_14)}\n"
            f"MACD: {fmt(macd)}\n"
            f"SMA 50: {fmt(sma_50)}\n"
            f"SMA 200: {fmt(sma_200)}\n"
            f"Bollinger Upper: {fmt(boll_ub)}\n"
            f"Bollinger Lower: {fmt(boll_lb)}"
        )
    except Exception as exc:
        return f"Error: {exc}"


@tool
async def get_fundamental_analysis(
    ticker: Annotated[
        str,
        "Exact ticker with exchange suffix — e.g. '3217.TWO', '7203.T', '0005.HK'."
        " Use exactly as provided; never alter or drop the suffix.",
    ],
) -> str:
    """
    Perform web search for qualitative fundamental factors (Analyst coverage, ADRs).

    IMPLEMENTS SURGICAL FALLBACK LOGIC:
    1. Primary Search: Uses specific ticker (best for exact listing).
    2. Check Success: If ticker search fails (insufficient data), do full fallback to Company Name search.
    3. Check ADR Miss: If ticker search succeeds but finds NO ADR info, perform SURGICAL append search using Company Name.
    """
    if not shared.tavily_tool:
        return "Tool unavailable"

    try:
        normalized_symbol = normalize_ticker(ticker)
        ticker_obj = yf.Ticker(normalized_symbol)
        company_name = await shared.extract_company_name_async(ticker_obj)

        ticker_query = f"{ticker} stock analyst coverage count consensus rating American Depositary Receipt exchange listing ADR status"
        ticker_results = await shared._tavily_search_with_timeout(
            {"query": ticker_query}
        )
        ticker_results_str = (
            shared._format_and_truncate_tavily_result(ticker_results)
            if ticker_results
            else ""
        )

        ticker_search_failed = not ticker_results or len(ticker_results_str) < 200

        if ticker_search_failed:
            if company_name and company_name != ticker:
                name_query = f'"{company_name}" stock analyst coverage count consensus rating American Depositary Receipt ADR status'
                name_results = await shared._tavily_search_with_timeout(
                    {"query": name_query}
                )
                if name_results:
                    return (
                        f"Fundamental Search Results for {company_name} ({ticker}) [Source: Fallback Name Search]:\n"
                        f"{shared._format_and_truncate_tavily_result(name_results)}\n\n"
                        f"(Note: Primary ticker search yielded insufficient data, switched to company name search)"
                    )
            return f"Fundamental Search Results for {ticker} (Limited Data):\n{ticker_results_str}"

        adr_keywords = [
            "ADR",
            "American Depositary",
            "Depositary Receipt",
            "OTC",
            "Pink Sheets",
            "sponsored",
        ]
        found_adr_info = any(
            keyword.lower() in ticker_results_str.lower() for keyword in adr_keywords
        )

        if not found_adr_info and company_name and company_name != ticker:
            adr_query = (
                f'"{company_name}" American Depositary Receipt ADR ticker status'
            )
            adr_results = await shared._tavily_search_with_timeout({"query": adr_query})
            adr_results_str = (
                shared._format_and_truncate_tavily_result(adr_results)
                if adr_results
                else ""
            )

            if any(
                keyword.lower() in adr_results_str.lower() for keyword in adr_keywords
            ):
                return (
                    f"Fundamental Search Results for {ticker} [Primary Source]:\n"
                    f"{ticker_results_str}\n\n"
                    f"=== SUPPLEMENTAL ADR SEARCH ===\n"
                    f"(Primary ticker search missed ADR info, found via name search for '{company_name}')\n"
                    f"{adr_results_str}"
                )

        return f"Fundamental Search Results for {ticker}:\n{ticker_results_str}"
    except Exception as exc:
        return f"Error searching for fundamentals: {exc}"
