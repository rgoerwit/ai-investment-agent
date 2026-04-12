"""Legal and tax disclosure tool implementations."""

import json
from typing import Annotated

import structlog
from langchain_core.tools import tool

from src.ticker_policy import CHINA_SUFFIXES, ticker_in_group
from src.tools import shared

logger = structlog.get_logger(__name__)

WITHHOLDING_TAX_RATES = {
    "japan": "15%",
    "hong kong": "0%",
    "singapore": "0%",
    "united kingdom": "0%",
    "uk": "0%",
    "germany": "15%",
    "france": "15%",
    "australia": "15%",
    "canada": "15%",
    "taiwan": "21%",
    "south korea": "15%",
    "korea": "15%",
    "china": "10%",
    "india": "25%",
    "brazil": "15%",
    "switzerland": "15%",
    "netherlands": "15%",
    "ireland": "15%",
    "sweden": "15%",
    "norway": "15%",
    "denmark": "15%",
    "finland": "15%",
    "israel": "25%",
    "mexico": "10%",
    "cayman islands": "0%",
    "british virgin islands": "0%",
    "bermuda": "0%",
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
    """
    if not shared.tavily_tool:
        return json.dumps(
            {
                "error": "Legal/tax search unavailable (Tavily not configured)",
                "searches_performed": [],
            }
        )

    pfic_risk_sectors = {
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
    pfic_risk_keywords = [
        "Leasing",
        "REIT",
        "Investment Trust",
        "Asset Management",
        "Holding",
        "Private Equity",
        "Venture Capital",
    ]
    china_domiciles = [
        "china",
        "hong kong",
        "cayman islands",
        "british virgin islands",
        "bermuda",
    ]

    is_pfic_risk = sector in pfic_risk_sectors or any(
        keyword.lower() in sector.lower() for keyword in pfic_risk_keywords
    )
    is_china_connected = (
        ticker_in_group(ticker, CHINA_SUFFIXES) or country.lower() in china_domiciles
    )

    country_lower = country.lower().strip()
    withholding_rate = WITHHOLDING_TAX_RATES.get(country_lower, "UNKNOWN")

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

    results = await shared._tavily_search_with_timeout({"query": query})

    searches = []
    if is_pfic_risk:
        searches.append("PFIC")
    if is_china_connected:
        searches.append("VIE")

    if not results:
        return json.dumps(
            {
                "error": "Search timed out or failed",
                "searches_performed": searches,
                "pfic_relevant": is_pfic_risk,
                "vie_relevant": is_china_connected,
                "withholding_rate": withholding_rate,
                "country": country,
                "sector": sector,
            }
        )

    results_str = shared._format_and_truncate_tavily_result(results, max_chars=2500)

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
