"""Ownership-analysis tool implementations."""

import asyncio
import json
from typing import Annotated

import pandas as pd
import structlog
import yfinance as yf
from langchain_core.tools import tool

from src.ticker_utils import normalize_ticker

logger = structlog.get_logger(__name__)

OWNERSHIP_SECTION_TIMEOUT_SECONDS = 8.0


async def _load_ownership_property(
    yf_ticker: yf.Ticker, property_name: str, ticker: str
) -> object:
    """Bound blocking yfinance ownership-property access."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(lambda: getattr(yf_ticker, property_name)),
            timeout=OWNERSHIP_SECTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.debug(
            "ownership_structure_timeout",
            ticker=ticker,
            section=property_name,
            timeout_seconds=OWNERSHIP_SECTION_TIMEOUT_SECONDS,
        )
        raise


@tool
async def get_ownership_structure(
    ticker: Annotated[str, "Stock ticker symbol (e.g., '7203.T', '0005.HK')"],
) -> str:
    """
    Get institutional holders, insider transactions, and major shareholders.

    Returns structured ownership data for governance and value trap analysis.
    """
    normalized = normalize_ticker(ticker)
    logger.info("ownership_structure_lookup", ticker=normalized)

    result = {
        "ticker": normalized,
        "institutional_holders": [],
        "institutional_holders_status": "PENDING",
        "insider_transactions": [],
        "insider_transactions_status": "PENDING",
        "major_holders": {},
        "major_holders_status": "PENDING",
        "ownership_concentration": None,
        "insider_trend": "UNKNOWN",
        "data_quality": "COMPLETE",
    }

    try:
        yf_ticker = yf.Ticker(normalized)

        try:
            inst = await _load_ownership_property(
                yf_ticker, "institutional_holders", normalized
            )
            if inst is None:
                result["institutional_holders_status"] = "DATA_UNAVAILABLE"
                result["data_quality"] = "PARTIAL"
            elif inst.empty:
                result["institutional_holders_status"] = "EMPTY"
            else:
                inst_records = inst.head(10).to_dict("records")
                for record in inst_records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                        elif hasattr(value, "isoformat"):
                            record[key] = value.isoformat()
                result["institutional_holders"] = inst_records
                result["institutional_holders_status"] = "FOUND"
        except Exception as exc:
            logger.debug(
                "institutional_holders_error", ticker=normalized, error=str(exc)
            )
            result["institutional_holders_status"] = "DATA_UNAVAILABLE"
            result["data_quality"] = "PARTIAL"

        try:
            insider = await _load_ownership_property(
                yf_ticker, "insider_transactions", normalized
            )
            if insider is None:
                result["insider_transactions_status"] = "DATA_UNAVAILABLE"
                result["data_quality"] = "PARTIAL"
            elif insider.empty:
                result["insider_transactions_status"] = "EMPTY"
            else:
                insider_records = insider.head(15).to_dict("records")
                for record in insider_records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                        elif hasattr(value, "isoformat"):
                            record[key] = value.isoformat()
                result["insider_transactions"] = insider_records
                result["insider_transactions_status"] = "FOUND"

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
        except Exception as exc:
            logger.debug(
                "insider_transactions_error", ticker=normalized, error=str(exc)
            )
            result["insider_transactions_status"] = "DATA_UNAVAILABLE"
            result["data_quality"] = "PARTIAL"

        try:
            major = await _load_ownership_property(
                yf_ticker, "major_holders", normalized
            )
            if major is None:
                result["major_holders_status"] = "DATA_UNAVAILABLE"
                result["data_quality"] = "PARTIAL"
            elif major.empty:
                result["major_holders_status"] = "EMPTY"
            else:
                major_dict = {}
                for idx, row in major.iterrows():
                    if len(row) >= 2:
                        key = str(row.iloc[1]) if pd.notna(row.iloc[1]) else str(idx)
                        value = row.iloc[0]
                        if pd.notna(value):
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

                if result["institutional_holders"]:
                    top5_pct = sum(
                        float(holder.get("pctHeld", 0) or 0) * 100
                        for holder in result["institutional_holders"][:5]
                    )
                    result["ownership_concentration"] = round(top5_pct, 2)
        except Exception as exc:
            logger.debug("major_holders_error", ticker=normalized, error=str(exc))
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
    except Exception as exc:
        logger.error("ownership_structure_error", ticker=normalized, error=str(exc))
        return json.dumps(
            {
                "ticker": normalized,
                "error": str(exc),
                "institutional_holders": [],
                "insider_transactions": [],
                "major_holders": {},
                "data_quality": "ERROR",
            },
            indent=2,
        )


def classify_insider_selling_evidence(ownership_data: dict) -> str:
    """
    Return an evidence tier for insider-selling claims.

    Tiers (from strongest to weakest):
    - NAMED_MULTI_EXEC: structured records naming ≥2 distinct executives with share counts
    - NAMED_SINGLE_EXEC: one named executive with a share-count record
    - GENERIC_TREND: directional signal (NET_SELLER) only, no named transactions
    - NONE: no insider selling detected

    Args:
        ownership_data: Parsed dict from ``get_ownership_structure()`` JSON output
            (or the deserialized result).

    Returns:
        One of the four tier strings above.
    """
    insider_records = ownership_data.get("insider_transactions", [])
    named_execs = {
        r["name"] for r in insider_records if r.get("name") and r.get("shares")
    }
    if len(named_execs) >= 2:
        return "NAMED_MULTI_EXEC"
    if len(named_execs) == 1:
        return "NAMED_SINGLE_EXEC"
    if ownership_data.get("insider_trend") in ("NET_SELLER", "SELLING"):
        return "GENERIC_TREND"
    return "NONE"
