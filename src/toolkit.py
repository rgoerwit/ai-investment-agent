"""
PRODUCTION-READY Enhanced Toolkit for Multi-Agent Trading System
Includes fixes for News Analyst alignment and local domain searching.
Updated for LangChain/LangGraph Fall 2025 standards.
"""

import asyncio
import html
import math
from typing import Annotated, Any

import pandas as pd
import structlog
import yfinance as yf
from langchain_core.tools import tool
from stockstats import wrap as stockstats_wrap

from src.config import config
from src.data.fetcher import fetcher as market_data_fetcher
from src.enhanced_sentiment_toolkit import get_multilingual_sentiment_search
from src.liquidity_calculation_tool import calculate_liquidity_metrics
from src.stocktwits_api import StockTwitsAPI

# FIX: Use dynamic ticker utils for normalization and name cleaning
from src.ticker_utils import normalize_company_name, normalize_ticker

logger = structlog.get_logger(__name__)
stocktwits_api = StockTwitsAPI()

# --- Modernized Tavily Import Pattern ---
# Note: API key is explicitly passed from config to avoid dependency on
# os.environ being populated by load_dotenv() (Pydantic Settings handles
# .env loading for our config, but third-party libs expect explicit api_key).
TAVILY_AVAILABLE = False
tavily_tool = None
_tavily_api_key = config.get_tavily_api_key()
if _tavily_api_key:
    try:
        from langchain_tavily import TavilySearch

        tavily_tool = TavilySearch(max_results=5, tavily_api_key=_tavily_api_key)
        TAVILY_AVAILABLE = True
    except ImportError:
        try:
            from langchain_community.tools import TavilySearchResults

            tavily_tool = TavilySearchResults(
                max_results=5, tavily_api_key=_tavily_api_key
            )
            TAVILY_AVAILABLE = True
        except ImportError:
            try:
                from langchain_community.tools.tavily_search import TavilySearchResults

                tavily_tool = TavilySearchResults(
                    max_results=5, tavily_api_key=_tavily_api_key
                )
                TAVILY_AVAILABLE = True
            except ImportError:
                logger.warning(
                    "Tavily tools not available. Install langchain-tavily or langchain-community."
                )
else:
    logger.warning("TAVILY_API_KEY not set. Tavily tools disabled.")


def _truncate_tavily_result(result: Any, max_chars: int | None = None) -> str:
    """
    Truncate Tavily search result to prevent token bloat.

    Uses TAVILY_MAX_CHARS from config (default 7000 chars ~1750 tokens).
    """
    from src.config import config

    if max_chars is None:
        max_chars = config.tavily_max_chars

    result_str = str(result)
    if len(result_str) > max_chars:
        return result_str[:max_chars] + "\n[...truncated for efficiency]"
    return result_str


async def fetch_with_timeout(coroutine, timeout_seconds=10, error_msg="Timeout"):
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"YFINANCE TIMEOUT: {error_msg}")
        return None
    except Exception as e:
        logger.warning(f"YFINANCE ERROR: {error_msg} - {str(e)}")
        return None


async def extract_company_name_async(ticker_obj) -> str:
    """Robust company name extraction with dynamic cleaning."""
    ticker_str = ticker_obj.ticker

    try:
        # 1. Try yfinance fast_info (no network call if cached)
        if hasattr(ticker_obj, "fast_info"):
            # fast_info is lazy, accessing it triggers load
            pass

        # 2. Try standard info with timeout
        info = await fetch_with_timeout(
            asyncio.to_thread(lambda: ticker_obj.info),
            timeout_seconds=5,
            error_msg="Name Extraction",
        )

        if info:
            long_name = info.get("longName") or info.get("shortName")
            if long_name:
                # Use dynamic cleaner to strip legal suffixes
                return normalize_company_name(long_name)

        return ticker_str

    except Exception:
        return ticker_str


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


# --- DATA UTILS ---


def _safe_float(value: Any) -> float | None:
    """Safely convert value to float, handling None, strings, NaN, and Inf."""
    try:
        if value is None:
            return None
        # Handle percentage strings "15%"
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "")
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _format_val(value: Any, fmt: str = "{:.2f}", default: str = "N/A") -> str:
    """Format a value safely, returning default if invalid."""
    val = _safe_float(value)
    if val is None:
        return default
    return fmt.format(val)


# --- DATA TOOLS ---

import json


def _sanitize_for_json(data: dict) -> dict:
    """
    Sanitize data for JSON encoding.
    Converts infinity/NaN to None, handles negative prices, and converts string numbers.
    Recursively handles nested dicts and lists.
    """
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively sanitize nested dicts
            sanitized[key] = _sanitize_for_json(value)
        elif isinstance(value, list):
            # Sanitize list elements
            sanitized[key] = [
                _sanitize_for_json(v) if isinstance(v, dict) else v for v in value
            ]
        elif isinstance(value, float):
            # Handle infinity and NaN - convert to None for valid JSON
            if math.isinf(value) or math.isnan(value):
                sanitized[key] = None
            # Handle negative prices (data corruption)
            elif key == "currentPrice" and value < 0:
                sanitized[key] = None
            else:
                sanitized[key] = value
        elif (
            isinstance(value, str)
            and key != "_data_source"
            and key != "currency"
            and key != "symbol"
        ):
            # Try to convert string numbers to float
            try:
                sanitized[key] = float(value)
            except (ValueError, TypeError):
                sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized


@tool
async def get_financial_metrics(ticker: Annotated[str, "Stock ticker symbol"]) -> str:
    """Get key financial ratios and metrics as a JSON string."""
    try:
        normalized_symbol = normalize_ticker(ticker)
        data = await market_data_fetcher.get_financial_metrics(normalized_symbol)

        if "error" in data:
            return json.dumps({"error": data.get("error")})

        # Sanitize data for valid JSON output (handle inf/nan/negative prices)
        sanitized_data = _sanitize_for_json(data)

        # Return sanitized data as JSON string for the agent to process
        return json.dumps(sanitized_data, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
async def get_news(
    ticker: Annotated[str, "Stock ticker symbol"],
    search_query: Annotated[str, "Specific query"] = None,
) -> str:
    """
    Get recent news using Tavily with ENHANCED multi-query strategy.
    Structures output for News Analyst prompt ingestion.
    """
    if not tavily_tool:
        return "News tool unavailable."

    try:
        normalized_symbol = normalize_ticker(ticker)
        ticker_obj = yf.Ticker(normalized_symbol)
        company_name = await extract_company_name_async(ticker_obj)

        # Local Domain Mapping
        local_source_hints = {
            ".KS": "site:pulsenews.co.kr OR site:koreatimes.co.kr OR site:koreaherald.com",
            ".HK": "site:scmp.com OR site:thestandard.com.hk OR site:ejinsight.com",
            ".T": "site:japantimes.co.jp OR site:nikkei.com",
            ".L": "site:ft.com OR site:bbc.co.uk/news/business",
            ".PA": "site:france24.com OR site:lemonde.fr",
            ".DE": "site:dw.com OR site:handelsblatt.com",
        }

        suffix = ""
        if "." in normalized_symbol:
            suffix = "." + normalized_symbol.split(".")[-1]
        local_hint = local_source_hints.get(suffix, "")

        results = []

        # 1. General Search - Use Clean Name
        general_query = (
            f'"{company_name}" {search_query}'
            if search_query
            else f'"{company_name}" (earnings OR merger OR acquisition OR regulatory)'
        )
        try:
            general_result = await tavily_tool.ainvoke({"query": general_query})
            if general_result:
                # Sanitize and truncate using global limit
                sanitized = html.escape(_truncate_tavily_result(general_result))
                results.append(f"=== GENERAL NEWS ===\n{sanitized}\n")
        except Exception as e:
            logger.warning(f"General news search failed: {e}")

        # 2. Local Search - Use Clean Name
        if local_hint and not search_query:
            local_query = (
                f'"{company_name}" {local_hint} (earnings OR guidance OR strategy)'
            )
            try:
                local_result = await tavily_tool.ainvoke({"query": local_query})
                if local_result:
                    # Sanitize and truncate using global limit
                    sanitized_local = html.escape(_truncate_tavily_result(local_result))
                    results.append(
                        f"=== LOCAL/REGIONAL NEWS SOURCES ===\n{sanitized_local}\n"
                    )
            except Exception as e:
                logger.warning(f"Local news search failed: {e}")

        if not results:
            return f"No news found for {company_name}."

        return f"News Results for {company_name}:\n\n" + "\n".join(results)
    except Exception as e:
        logger.error(f"News fetch failed for {ticker}: {e}")
        # Propagate error message instead of generic "No news found"
        return f"Error fetching news: {str(e)}"


@tool
async def get_yfinance_data(
    symbol: str, start_date: str = None, end_date: str = None
) -> str:
    """Get historical stock price data."""
    try:
        normalized = normalize_ticker(symbol)
        hist = await market_data_fetcher.get_historical_prices(normalized)
        if hist.empty:
            return "No data"
        return hist.reset_index().to_csv(index=False)
    except Exception as e:
        return f"Error: {e}"


@tool
async def get_technical_indicators(symbol: str) -> str:
    """Get RSI, MACD, Bollinger Bands, and Moving Averages."""
    try:
        normalized = normalize_ticker(symbol)
        # FIX: Fetch '2y' to ensure enough data for 200-day MA
        hist = await market_data_fetcher.get_historical_prices(normalized, period="2y")

        if hist.empty:
            return "No data"

        stock = stockstats_wrap(hist)
        latest = hist.iloc[-1]

        # Explicitly calculate MAs
        sma_50 = _safe_float(stock["close_50_sma"].iloc[-1])
        sma_200 = _safe_float(stock["close_200_sma"].iloc[-1])

        # Format with safety checks
        def fmt(val):
            return _format_val(val)

        return (
            f"Technical Indicators for {symbol}:\n"
            f"Current Price: {fmt(latest['Close'])}\n"
            f"RSI (14): {fmt(stock['rsi_14'].iloc[-1])}\n"
            f"MACD: {fmt(stock['macd'].iloc[-1])}\n"
            f"SMA 50: {fmt(sma_50)}\n"
            f"SMA 200: {fmt(sma_200)}\n"
            f"Bollinger Upper: {fmt(stock['boll_ub'].iloc[-1])}\n"
            f"Bollinger Lower: {fmt(stock['boll_lb'].iloc[-1])}"
        )
    except Exception as e:
        return f"Error: {e}"


@tool
async def get_social_media_sentiment(ticker: str) -> str:
    """Get sentiment from StockTwits."""
    try:
        # Try to resolve company name for better context if needed later
        data = await stocktwits_api.get_sentiment(ticker)
        return str(data)
    except Exception as e:
        return f"Error getting sentiment: {str(e)}"


@tool
async def get_macroeconomic_news(trade_date: str) -> str:
    """Get macroeconomic news context for a specific date."""
    if not tavily_tool:
        return "Tool unavailable"
    result = await tavily_tool.ainvoke({"query": f"macroeconomic news {trade_date}"})
    return _truncate_tavily_result(result)


@tool
async def get_fundamental_analysis(
    ticker: Annotated[str, "Stock ticker symbol"],
) -> str:
    """
    Perform web search for qualitative fundamental factors (Analyst coverage, ADRs).

    IMPLEMENTS SURGICAL FALLBACK LOGIC:
    1. Primary Search: Uses specific ticker (best for exact listing).
    2. Check Success: If ticker search fails (insufficient data), do full fallback to Company Name search.
    3. Check ADR Miss: If ticker search succeeds but finds NO ADR info, perform SURGICAL append search using Company Name.
    """
    if not tavily_tool:
        return "Tool unavailable"

    try:
        # Get company name for potential fallback/surgical search
        normalized_symbol = normalize_ticker(ticker)
        ticker_obj = yf.Ticker(normalized_symbol)
        company_name = await extract_company_name_async(ticker_obj)

        # 1. Primary Search: Ticker-based (Most specific to the listing)
        # Use strict quoting for the ticker name if we have it, otherwise just ticker
        ticker_query = f"{ticker} stock analyst coverage count consensus rating American Depositary Receipt exchange listing ADR status"
        ticker_results = await tavily_tool.ainvoke({"query": ticker_query})
        ticker_results_str = str(ticker_results)

        # Check result quality
        # If results are empty or very short (< 200 chars), the ticker search essentially failed.
        ticker_search_failed = not ticker_results or len(ticker_results_str) < 200

        # CASE A: TOTAL FAILURE -> Full Fallback
        if ticker_search_failed:
            if company_name and company_name != ticker:
                # Use quoted company name for strictness
                name_query = f'"{company_name}" stock analyst coverage count consensus rating American Depositary Receipt ADR status'
                name_results = await tavily_tool.ainvoke({"query": name_query})
                return (
                    f"Fundamental Search Results for {company_name} ({ticker}) [Source: Fallback Name Search]:\n"
                    f"{_truncate_tavily_result(name_results)}\n\n"
                    f"(Note: Primary ticker search yielded insufficient data, switched to company name search)"
                )
            return f"Fundamental Search Results for {ticker} (Limited Data):\n{_truncate_tavily_result(ticker_results)}"

        # CASE B: SUCCESS BUT POTENTIAL ADR MISS -> Surgical Append
        # Check if the ticker results actually mention ADR/Depositary keywords
        adr_keywords = [
            "ADR",
            "American Depositary",
            "Depositary Receipt",
            "OTC",
            "Pink Sheets",
            "sponsored",
        ]
        found_adr_info = any(
            kw.lower() in ticker_results_str.lower() for kw in adr_keywords
        )

        # If we have a good company name, the ticker search succeeded, BUT it missed ADR info...
        if not found_adr_info and company_name and company_name != ticker:
            # Run a targeted "Surgical" search just for the ADR
            # Use quoted company name
            adr_query = (
                f'"{company_name}" American Depositary Receipt ADR ticker status'
            )
            adr_results = await tavily_tool.ainvoke({"query": adr_query})
            adr_results_str = str(adr_results)

            # Only append if the surgical search actually found something relevant to avoid noise
            if any(kw.lower() in adr_results_str.lower() for kw in adr_keywords):
                combined_results = (
                    f"Fundamental Search Results for {ticker} [Primary Source]:\n"
                    f"{_truncate_tavily_result(ticker_results)}\n\n"
                    f"=== SUPPLEMENTAL ADR SEARCH ===\n"
                    f"(Primary ticker search missed ADR info, found via name search for '{company_name}')\n"
                    f"{_truncate_tavily_result(adr_results)}"
                )
                return combined_results

        # Case C: Success and ADR info found (or no name available to double check)
        return f"Fundamental Search Results for {ticker}:\n{_truncate_tavily_result(ticker_results)}"

    except Exception as e:
        return f"Error searching for fundamentals: {e}"


@tool
async def search_foreign_sources(
    ticker: Annotated[str, "Stock ticker symbol"],
    search_query: Annotated[str, "Search query (can include native language terms)"],
) -> str:
    """
    Search for financial data from foreign-language and premium English sources.

    Use this tool to find official filings, IR pages, and premium source data
    that may not be available through standard English-language APIs.

    Tips:
    - Include native language terms for non-US tickers
    - Include site: operator for targeting specific sources
    - Include year/date for recent data
    """
    if not tavily_tool:
        return "Foreign source search unavailable (Tavily not configured)"

    try:
        normalized_symbol = normalize_ticker(ticker)
        ticker_obj = yf.Ticker(normalized_symbol)
        company_name = await extract_company_name_async(ticker_obj)

        # Build comprehensive query with company context
        full_query = f"{search_query} {company_name} {ticker}"

        logger.info(
            "foreign_source_search",
            ticker=ticker,
            query=full_query[:100],  # Truncate for logging
        )

        results = await tavily_tool.ainvoke({"query": full_query})

        if not results:
            return f"No results found for foreign source search: {search_query}"

        # Format results for agent consumption (with truncation)
        results_str = _truncate_tavily_result(results)

        # Add context header
        output = f"""### Foreign Source Search Results
Query: {search_query}
Ticker: {ticker} ({company_name})

{results_str}

Note: Verify dates and currencies in the source data."""

        return output

    except Exception as e:
        logger.error(f"Foreign source search error: {e}")
        return f"Error searching foreign sources: {e}"


# --- Legal/Tax Disclosure Tool ---

# Static withholding tax rates by country (treaty rates for US investors)
# Last updated: 2025-01-01 | Source: IRS Publication 515, bilateral tax treaties
# Note: These are standard treaty rates; actual rates may vary based on investor status
WITHHOLDING_TAX_RATES = {
    "japan": "15%",  # US-Japan Treaty Art. 10 (reduced from 20%)
    "hong kong": "0%",  # No withholding tax on dividends
    "singapore": "0%",  # US-Singapore Treaty
    "united kingdom": "0%",  # US-UK Treaty (0% for qualified dividends)
    "uk": "0%",  # Alias
    "germany": "15%",  # US-Germany Treaty (reduced from 26.375%)
    "france": "15%",  # US-France Treaty (reduced from 30%)
    "australia": "15%",  # US-Australia Treaty
    "canada": "15%",  # US-Canada Treaty
    "taiwan": "21%",  # No treaty, statutory rate
    "south korea": "15%",  # US-Korea Treaty
    "korea": "15%",  # Alias
    "china": "10%",  # US-China Treaty
    "india": "25%",  # US-India Treaty
    "brazil": "15%",  # US-Brazil Treaty (interest; no dividend treaty)
    "switzerland": "15%",  # US-Switzerland Treaty
    "netherlands": "15%",  # US-Netherlands Treaty
    "ireland": "15%",  # US-Ireland Treaty
    "sweden": "15%",  # US-Sweden Treaty
    "norway": "15%",  # US-Norway Treaty
    "denmark": "15%",  # US-Denmark Treaty
    "finland": "15%",  # US-Finland Treaty
    "israel": "25%",  # US-Israel Treaty
    "mexico": "10%",  # US-Mexico Treaty
    "cayman islands": "0%",  # No local tax
    "british virgin islands": "0%",  # No local tax
    "bermuda": "0%",  # No local tax
}


@tool
async def search_legal_tax_disclosures(
    ticker: Annotated[str, "Stock ticker symbol (e.g., 8591.T, 0005.HK)"],
    company_name: Annotated[str, "Full company name"],
    sector: Annotated[str, "Company sector from financial data"],
    country: Annotated[str, "Country of domicile"],
) -> str:
    """
    Search for US investor legal/tax disclosures: PFIC status and VIE structures.

    Runs ONE combined search query to minimize API calls and rate limit risk.
    Only performs substantive search for high-risk profiles:
    - PFIC: Financial Services, Insurance, Banks, Leasing, REITs, Asset Management
    - VIE: China-connected tickers (.HK, .SS, .SZ) or Cayman/BVI domicile

    Returns JSON with search results and withholding tax rate.
    """
    import json

    if not tavily_tool:
        return json.dumps(
            {
                "error": "Legal/tax search unavailable (Tavily not configured)",
                "searches_performed": [],
            }
        )

    # Determine risk profile
    PFIC_RISK_SECTORS = {
        "Financial Services",
        "Insurance",
        "Banks",
        "Capital Markets",
        "Diversified Financial Services",
        "Real Estate",
        "Thrifts & Mortgage Finance",
        "Asset Management",
        "Investment Banking & Brokerage",
    }
    PFIC_RISK_KEYWORDS = [
        "Leasing",
        "REIT",
        "Investment Trust",
        "Asset Management",
        "Holding",
        "Private Equity",
        "Venture Capital",
    ]

    CHINA_SUFFIXES = (".HK", ".SS", ".SZ")
    CHINA_DOMICILES = [
        "china",
        "hong kong",
        "cayman islands",
        "british virgin islands",
        "bermuda",
    ]

    is_pfic_risk = sector in PFIC_RISK_SECTORS or any(
        kw.lower() in sector.lower() for kw in PFIC_RISK_KEYWORDS
    )

    is_china_connected = (
        any(ticker.upper().endswith(suffix) for suffix in CHINA_SUFFIXES)
        or country.lower() in CHINA_DOMICILES
    )

    # Get withholding rate from lookup table
    country_lower = country.lower().strip()
    withholding_rate = WITHHOLDING_TAX_RATES.get(country_lower, "UNKNOWN")

    # If no risk factors, return early with minimal data
    if not is_pfic_risk and not is_china_connected:
        logger.info(
            "legal_search_skipped",
            ticker=ticker,
            sector=sector,
            country=country,
            reason="Low-risk profile",
        )
        return json.dumps(
            {
                "searches_performed": [],
                "pfic_relevant": False,
                "vie_relevant": False,
                "withholding_rate": withholding_rate,
                "country": country,
                "sector": sector,
                "note": "Low-risk profile - no legal/tax search required",
            }
        )

    # Build ONE combined search query (rate limit efficient)
    search_terms = []
    if is_pfic_risk:
        search_terms.append(
            'PFIC "passive foreign investment company" 20-F "US investors" tax'
        )
    if is_china_connected:
        search_terms.append(
            'VIE "variable interest entity" "contractual arrangements" structure'
        )

    query = f'"{company_name}" ({ticker}) {" ".join(search_terms)}'

    logger.info(
        "legal_tax_search",
        ticker=ticker,
        company=company_name,
        pfic_risk=is_pfic_risk,
        vie_risk=is_china_connected,
        query_length=len(query),
    )

    try:
        results = await tavily_tool.ainvoke({"query": query})
        results_str = _truncate_tavily_result(results, max_chars=2500)

        searches = []
        if is_pfic_risk:
            searches.append("PFIC")
        if is_china_connected:
            searches.append("VIE")

        return json.dumps(
            {
                "searches_performed": searches,
                "pfic_relevant": is_pfic_risk,
                "vie_relevant": is_china_connected,
                "withholding_rate": withholding_rate,
                "country": country,
                "sector": sector,
                "results": results_str,
            }
        )

    except Exception as e:
        logger.error("legal_tax_search_error", ticker=ticker, error=str(e))
        return json.dumps(
            {
                "error": str(e),
                "searches_performed": [],
                "pfic_relevant": is_pfic_risk,
                "vie_relevant": is_china_connected,
                "withholding_rate": withholding_rate,
                "country": country,
                "sector": sector,
            }
        )


@tool
async def get_ownership_structure(
    ticker: Annotated[str, "Stock ticker symbol (e.g., '7203.T', '0005.HK')"],
) -> str:
    """
    Get institutional holders, insider transactions, and major shareholders.

    Returns structured ownership data for governance and value trap analysis.
    Includes: top institutional holders, recent insider transactions,
    and major holder concentration.

    Args:
        ticker: Stock ticker symbol with exchange suffix

    Returns:
        JSON string with ownership data or error message
    """
    normalized = normalize_ticker(ticker)
    logger.info("ownership_structure_lookup", ticker=normalized)

    result = {
        "ticker": normalized,
        "institutional_holders": [],
        "institutional_holders_status": "PENDING",  # FOUND, EMPTY, DATA_UNAVAILABLE
        "insider_transactions": [],
        "insider_transactions_status": "PENDING",  # FOUND, EMPTY, DATA_UNAVAILABLE
        "major_holders": {},
        "major_holders_status": "PENDING",  # FOUND, EMPTY, DATA_UNAVAILABLE
        "ownership_concentration": None,
        "insider_trend": "UNKNOWN",
        "data_quality": "COMPLETE",
    }

    try:
        yf_ticker = yf.Ticker(normalized)

        # Top institutional holders
        try:
            inst = yf_ticker.institutional_holders
            if inst is None:
                # API returned None - data unavailable (different from zero holders)
                result["institutional_holders_status"] = "DATA_UNAVAILABLE"
                result["data_quality"] = "PARTIAL"
            elif inst.empty:
                # DataFrame exists but is empty - genuinely no institutional holders
                result["institutional_holders_status"] = "EMPTY"
            else:
                # Convert to list of dicts, handle NaN values
                inst_records = inst.head(10).to_dict("records")
                # Sanitize for JSON
                for record in inst_records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                        elif hasattr(value, "isoformat"):  # Handle Timestamp
                            record[key] = value.isoformat()
                result["institutional_holders"] = inst_records
                result["institutional_holders_status"] = "FOUND"
        except Exception as e:
            logger.debug("institutional_holders_error", ticker=normalized, error=str(e))
            result["institutional_holders_status"] = "DATA_UNAVAILABLE"
            result["data_quality"] = "PARTIAL"

        # Recent insider transactions
        try:
            insider = yf_ticker.insider_transactions
            if insider is None:
                # API returned None - data unavailable
                result["insider_transactions_status"] = "DATA_UNAVAILABLE"
                result["data_quality"] = "PARTIAL"
            elif insider.empty:
                # DataFrame exists but is empty - genuinely no insider transactions
                result["insider_transactions_status"] = "EMPTY"
            else:
                insider_records = insider.head(15).to_dict("records")
                # Sanitize for JSON
                for record in insider_records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                        elif hasattr(value, "isoformat"):
                            record[key] = value.isoformat()
                result["insider_transactions"] = insider_records
                result["insider_transactions_status"] = "FOUND"

                # Calculate insider trend (net buyer/seller)
                buy_count = 0
                sell_count = 0
                for txn in insider_records:
                    txn_type = str(txn.get("Text", "")).lower()
                    if "buy" in txn_type or "purchase" in txn_type:
                        buy_count += 1
                    elif "sell" in txn_type or "sale" in txn_type:
                        sell_count += 1

                if buy_count > sell_count * 1.5:
                    result["insider_trend"] = "NET_BUYER"
                elif sell_count > buy_count * 1.5:
                    result["insider_trend"] = "NET_SELLER"
                else:
                    result["insider_trend"] = "NEUTRAL"
        except Exception as e:
            logger.debug("insider_transactions_error", ticker=normalized, error=str(e))
            result["insider_transactions_status"] = "DATA_UNAVAILABLE"
            result["data_quality"] = "PARTIAL"

        # Major holders summary
        try:
            major = yf_ticker.major_holders
            if major is None:
                # API returned None - data unavailable
                result["major_holders_status"] = "DATA_UNAVAILABLE"
                result["data_quality"] = "PARTIAL"
            elif major.empty:
                # DataFrame exists but is empty - genuinely no major holder data
                result["major_holders_status"] = "EMPTY"
            else:
                # Convert to dict, handling various formats
                major_dict = {}
                for idx, row in major.iterrows():
                    if len(row) >= 2:
                        key = str(row.iloc[1]) if pd.notna(row.iloc[1]) else str(idx)
                        value = row.iloc[0]
                        if pd.notna(value):
                            # Convert percentage strings
                            if isinstance(value, str) and "%" in value:
                                major_dict[key] = value
                            else:
                                major_dict[key] = (
                                    float(value)
                                    if isinstance(value, int | float)
                                    else str(value)
                                )
                result["major_holders"] = major_dict
                result["major_holders_status"] = "FOUND"

                # Calculate ownership concentration (top 5 institutional)
                if result["institutional_holders"]:
                    top5_pct = sum(
                        float(h.get("pctHeld", 0) or 0) * 100
                        for h in result["institutional_holders"][:5]
                    )
                    result["ownership_concentration"] = round(top5_pct, 2)
        except Exception as e:
            logger.debug("major_holders_error", ticker=normalized, error=str(e))
            result["major_holders_status"] = "DATA_UNAVAILABLE"
            result["data_quality"] = "PARTIAL"

        logger.info(
            "ownership_structure_complete",
            ticker=normalized,
            institutional_count=len(result["institutional_holders"]),
            insider_txn_count=len(result["insider_transactions"]),
            insider_trend=result["insider_trend"],
            data_quality=result["data_quality"],
        )

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error("ownership_structure_error", ticker=normalized, error=str(e))
        return json.dumps(
            {
                "ticker": normalized,
                "error": str(e),
                "institutional_holders": [],
                "insider_transactions": [],
                "major_holders": {},
                "data_quality": "ERROR",
            },
            indent=2,
        )


class Toolkit:
    def __init__(self):
        self.market_data_fetcher = market_data_fetcher

    def get_core_tools(self):
        return [get_yfinance_data, get_technical_indicators]

    def get_technical_tools(self):
        return [
            get_yfinance_data,
            get_technical_indicators,
            calculate_liquidity_metrics,
        ]

    # Alias for market analyst (uses technical tools)
    def get_market_tools(self):
        return self.get_technical_tools()

    def get_junior_fundamental_tools(self):
        """Tools for Junior Fundamentals Analyst (data gathering)."""
        return [get_financial_metrics, get_fundamental_analysis]

    def get_senior_fundamental_tools(self):
        """Senior Fundamentals Analyst has NO tools - receives data from Junior."""
        return []

    # Legacy alias - use get_junior_fundamental_tools instead
    def get_fundamental_tools(self):
        return self.get_junior_fundamental_tools()

    def get_sentiment_tools(self):
        return [get_social_media_sentiment, get_multilingual_sentiment_search]

    def get_news_tools(self):
        return [get_news, get_macroeconomic_news]

    def get_foreign_language_tools(self):
        """Tools for Foreign Language Analyst (supplemental data from native sources)."""
        return [search_foreign_sources]

    def get_legal_tools(self):
        """Tools for Legal Counsel (PFIC/VIE detection for US investors)."""
        return [search_legal_tax_disclosures]

    def get_value_trap_tools(self):
        """Tools for Value Trap Detector (governance & capital allocation analysis)."""
        return [
            get_ownership_structure,  # yfinance: institutional holders, insider transactions
            get_news,  # Tavily: activist campaigns, buyback announcements
            search_foreign_sources,  # Native governance searches (Mochiai, Chaebol, etc.)
        ]

    def get_all_tools(self):
        return [
            get_yfinance_data,
            get_technical_indicators,
            get_financial_metrics,
            get_news,
            get_social_media_sentiment,
            get_multilingual_sentiment_search,
            calculate_liquidity_metrics,
            get_macroeconomic_news,
            get_fundamental_analysis,
            search_foreign_sources,
            search_legal_tax_disclosures,
            get_ownership_structure,
        ]


toolkit = Toolkit()
