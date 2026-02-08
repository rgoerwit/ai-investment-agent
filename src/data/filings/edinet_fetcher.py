"""EDINET filing fetcher for Japanese stocks (.T suffix).

Uses the edinet-tools library to fetch structured data from Japan's EDINET
(Electronic Disclosure for Investors NETwork) — the Japanese equivalent of
SEC EDGAR.

Provides: major shareholders, segment breakdowns, and filing-level cash flow
from 有価証券報告書 (Annual Securities Reports) and other filings.

Requires: EDINET_API_KEY environment variable (free, registration at
https://disclosure.edinet-fsa.go.jp).
"""

import asyncio
import re

import structlog

from src.config import config
from src.data.filings.base import FilingFetcher, FilingResult

logger = structlog.get_logger(__name__)

# XBRL taxonomy keys for financial data extraction (JPX standard)
# These are common XBRL element names in EDINET filings
_XBRL_OCF_KEYS = [
    "jppfs_cor:CashFlowsFromUsedInOperatingActivities",
    "jppfs_cor:NetCashProvidedByUsedInOperatingActivities",
]


def _safe_float(value) -> float | None:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            # Handle Japanese number formatting (e.g., "1,234,567")
            value = value.replace(",", "").replace("，", "")
        return float(value)
    except (ValueError, TypeError):
        return None


def _extract_ticker_number(ticker: str) -> str:
    """Extract the numeric part from a ticker like '2767.T' -> '2767'."""
    return ticker.split(".")[0]


class EdinetFetcher(FilingFetcher):
    """Fetches filing data from Japan's EDINET API for .T tickers."""

    @property
    def supported_suffixes(self) -> list[str]:
        return ["T"]

    @property
    def source_name(self) -> str:
        return "EDINET"

    def is_available(self) -> bool:
        return bool(config.get_edinet_api_key())

    async def get_filing_data(self, ticker: str) -> FilingResult | None:
        """Fetch shareholders, segments, and cash flow from EDINET filings."""
        try:
            import edinet_tools
        except ImportError:
            logger.warning("edinet_tools_not_installed")
            return None

        ticker_num = _extract_ticker_number(ticker)

        # Look up entity in EDINET
        entity = await asyncio.to_thread(edinet_tools.entity, ticker_num)
        if entity is None:
            logger.info("edinet_entity_not_found", ticker=ticker)
            return None

        logger.info(
            "edinet_entity_found",
            ticker=ticker,
            name=getattr(entity, "name", "unknown"),
            edinet_code=getattr(entity, "edinet_code", "unknown"),
        )

        result = FilingResult(source="EDINET", ticker=ticker)

        # Fetch recent documents (annual reports = doc_type_code "120")
        docs = await asyncio.to_thread(
            lambda: entity.documents(days=400, doc_type="120")
        )

        if not docs:
            # Try broader search if no annual report found
            docs = await asyncio.to_thread(lambda: entity.documents(days=400))
            if not docs:
                logger.info("edinet_no_documents", ticker=ticker)
                result.data_gaps.append("No filings found in last 400 days")
                return result

        # Use the most recent filing
        doc = docs[0]
        result.filing_date = str(getattr(doc, "filing_datetime", ""))[:10]
        result.filing_type = getattr(doc, "doc_type_name", "Annual")
        result.filing_url = (
            f"https://disclosure.edinet-fsa.go.jp/E01EW/download?"
            f"uji.verb=W1E62071EdinetCodeDownload&uji.bean=ee.bean.W1E62071.EEW1E62071Bean"
            f"&TID=0&PID=W1E62071&SESSIONKEY=&DLKEY=&cat=yuho&edinession=&"
            f"key={getattr(doc, 'doc_id', '')}"
        )

        # Parse the document
        try:
            parsed = await asyncio.to_thread(doc.parse)
        except Exception as e:
            logger.warning("edinet_parse_error", ticker=ticker, error=str(e))
            result.data_gaps.append(f"Parse error: {str(e)[:100]}")
            return result

        # Extract data from parsed filing
        await self._extract_shareholders(parsed, doc, entity, result)
        await self._extract_segments(parsed, result)
        await self._extract_cash_flow(parsed, result)

        logger.info(
            "edinet_extraction_complete",
            ticker=ticker,
            has_shareholders=bool(result.major_shareholders),
            has_parent=bool(result.parent_company),
            has_segments=bool(result.segments),
            has_ocf=result.operating_cash_flow is not None,
            gaps=result.data_gaps,
        )

        return result

    async def _extract_shareholders(
        self, parsed, doc, entity, result: FilingResult
    ) -> None:
        """Extract major shareholder data from parsed filing or large holding reports."""
        import importlib.util

        if importlib.util.find_spec("edinet_tools") is None:
            return

        # Method 1: Try to get shareholder data from parsed annual report
        shareholders = []
        try:
            if hasattr(parsed, "to_dict"):
                parsed.to_dict()  # validate parseability; data used via attribute access below
        except Exception:
            pass

        # Method 2: Search for large shareholding reports (doc_type "350")
        try:
            holding_docs = await asyncio.to_thread(
                lambda: entity.documents(days=730, doc_type="350")
            )
            if holding_docs:
                for hdoc in holding_docs[:10]:  # Check recent holding reports
                    try:
                        hparsed = await asyncio.to_thread(hdoc.parse)
                        holder_name = getattr(hparsed, "holder_name", None)
                        ownership_pct = getattr(hparsed, "ownership_pct", None)
                        if holder_name and ownership_pct is not None:
                            pct_val = _safe_float(ownership_pct)
                            if pct_val is not None:
                                shareholders.append(
                                    {
                                        "name": str(holder_name),
                                        "percent": pct_val,
                                        "type": "large_holder",
                                    }
                                )
                    except Exception:
                        continue
        except Exception as e:
            logger.debug("edinet_holding_reports_error", error=str(e))

        if shareholders:
            # Deduplicate by name, keep highest percentage
            seen = {}
            for sh in shareholders:
                name = sh["name"]
                if name not in seen or sh["percent"] > seen[name]["percent"]:
                    seen[name] = sh
            result.major_shareholders = sorted(
                seen.values(), key=lambda x: x["percent"], reverse=True
            )

            # Identify parent company (holder with >20%)
            for sh in result.major_shareholders:
                if sh["percent"] >= 20.0:
                    result.parent_company = {
                        "name": sh["name"],
                        "percent": sh["percent"],
                        "relationship": (
                            "subsidiary" if sh["percent"] >= 50 else "equity_method"
                        ),
                    }
                    break
        else:
            result.data_gaps.append("Shareholders not extracted from filing")

    async def _extract_segments(self, parsed, result: FilingResult) -> None:
        """Extract segment breakdown from parsed filing."""
        segments = []

        try:
            parsed_dict = None
            if hasattr(parsed, "to_dict"):
                parsed_dict = await asyncio.to_thread(parsed.to_dict)

            if parsed_dict:
                # Look for segment-related XBRL fields
                for key, value in parsed_dict.items():
                    key_lower = key.lower()
                    if "segment" in key_lower and "revenue" in key_lower:
                        # Try to parse segment data from XBRL structure
                        if isinstance(value, dict):
                            for seg_name, seg_val in value.items():
                                val = _safe_float(seg_val)
                                if val is not None:
                                    segments.append(
                                        {
                                            "name": seg_name,
                                            "revenue": f"¥{val:,.0f}M",
                                            "op_profit": "N/A",
                                            "pct_of_total": None,
                                        }
                                    )

            # Also check for fields() method to find segment data
            if hasattr(parsed, "fields") and not segments:
                fields = await asyncio.to_thread(parsed.fields)
                segment_fields = [f for f in fields if "segment" in str(f).lower()]
                for sf in segment_fields[:10]:
                    try:
                        val = getattr(parsed, sf, None)
                        if val is not None:
                            segments.append(
                                {
                                    "name": str(sf),
                                    "revenue": str(val),
                                    "op_profit": "N/A",
                                    "pct_of_total": None,
                                }
                            )
                    except Exception:
                        continue
        except Exception as e:
            logger.debug("edinet_segment_extraction_error", error=str(e))

        if segments:
            # Calculate percentages if we have numeric revenue values
            total_rev = 0
            for seg in segments:
                rev_str = str(seg.get("revenue", ""))
                # Extract numeric value
                nums = re.findall(r"[\d,.]+", rev_str)
                if nums:
                    val = _safe_float(nums[0])
                    if val:
                        total_rev += val

            if total_rev > 0:
                for seg in segments:
                    rev_str = str(seg.get("revenue", ""))
                    nums = re.findall(r"[\d,.]+", rev_str)
                    if nums:
                        val = _safe_float(nums[0])
                        if val:
                            seg["pct_of_total"] = round(val / total_rev * 100, 1)

            result.segments = segments[:5]  # Max 5 segments
        else:
            result.data_gaps.append("Segment data not extracted from filing")

    async def _extract_cash_flow(self, parsed, result: FilingResult) -> None:
        """Extract operating cash flow from parsed filing."""
        try:
            parsed_dict = None
            if hasattr(parsed, "to_dict"):
                parsed_dict = await asyncio.to_thread(parsed.to_dict)

            if parsed_dict:
                # Look for OCF in XBRL data
                for xbrl_key in _XBRL_OCF_KEYS:
                    if xbrl_key in parsed_dict:
                        val = _safe_float(parsed_dict[xbrl_key])
                        if val is not None:
                            result.operating_cash_flow = val
                            result.ocf_currency = "JPY"
                            result.ocf_period = (
                                f"FY ending {result.filing_date}"
                                if result.filing_date
                                else "Latest FY"
                            )
                            return

                # Broader search for cash flow fields
                for key, value in parsed_dict.items():
                    key_lower = key.lower()
                    if (
                        "cashflow" in key_lower or "cash_flow" in key_lower
                    ) and "operating" in key_lower:
                        val = _safe_float(value)
                        if val is not None:
                            result.operating_cash_flow = val
                            result.ocf_currency = "JPY"
                            result.ocf_period = (
                                f"FY ending {result.filing_date}"
                                if result.filing_date
                                else "Latest FY"
                            )
                            return

            # Try net_sales as a proxy indicator that parsing worked
            if hasattr(parsed, "net_sales"):
                net_sales = _safe_float(getattr(parsed, "net_sales", None))
                if net_sales is not None:
                    logger.debug(
                        "edinet_has_net_sales_but_no_ocf",
                        net_sales=net_sales,
                    )

            result.data_gaps.append("OCF not extracted from filing")

        except Exception as e:
            logger.debug("edinet_cf_extraction_error", error=str(e))
            result.data_gaps.append("OCF extraction error")
