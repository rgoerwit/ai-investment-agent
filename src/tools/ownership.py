"""Ownership-analysis tool implementations."""

import json
from typing import Annotated

import pandas as pd
import structlog
import yfinance as yf
from langchain_core.tools import tool

from src.ticker_utils import normalize_ticker

logger = structlog.get_logger(__name__)


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
            inst = yf_ticker.institutional_holders
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
            insider = yf_ticker.insider_transactions
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
            major = yf_ticker.major_holders
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
