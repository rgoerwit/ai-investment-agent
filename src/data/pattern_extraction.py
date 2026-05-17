"""Regex-based extraction of financial metrics from free text."""

from __future__ import annotations

import re
from typing import Any

ROE_PERCENTAGE_THRESHOLD = 1.0


class FinancialPatternExtractor:
    """Handles regex-based extraction of financial metrics from text."""

    def __init__(self):
        self.patterns = {
            "trailingPE": [
                re.compile(
                    r"(?:Trailing P/E|P/E \(TTM\)|P/E Ratio \(TTM\))(?:.*?)\s*[:=]?\s*(\d+[\.,]\d+)",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"(?:P/E|est|trading at|valuation).*?\s+(\d+[\.,]\d+)x",
                    re.IGNORECASE,
                ),
                re.compile(r"P/E\s+(?:of|is|around)\s+(\d+[\.,]\d+)", re.IGNORECASE),
                re.compile(
                    r"(?<!Forward\s)(?<!Fwd\s)(?:P/E|Price[- ]to[- ]Earnings)(?:.*?)(?:Ratio)?\s*[:=]?\s*(\d+[\.,]\d+)",
                    re.IGNORECASE,
                ),
                re.compile(r"\btrades?\s+at\s+(\d+[\.,]\d+)x", re.IGNORECASE),
                re.compile(r"\bvalued\s+at\s+(\d+[\.,]\d+)x", re.IGNORECASE),
                re.compile(
                    r"\btrading\s+at\s+(\d+(?:[\.,]\d+)?)\s+times", re.IGNORECASE
                ),
            ],
            "forwardPE": [
                re.compile(
                    r"(?:Forward P/E|Fwd P/E)(?:.*?)\s*[:=]?\s*(\d+[\.,]\d+)",
                    re.IGNORECASE,
                ),
                re.compile(r"(?:Forward P/E|Fwd P/E).*?(\d+[\.,]\d+)x", re.IGNORECASE),
                re.compile(r"est.*?P/E.*?(\d+[\.,]\d+)x", re.IGNORECASE),
            ],
            "priceToBook": [
                re.compile(
                    r"(?:P/B|Price[- ]to[- ]Book)(?:.*?)(?:Ratio)?\s*[:=]?\s*(\d+[\.,]\d+)",
                    re.IGNORECASE,
                ),
                re.compile(r"PB\s*Ratio\s*[:=]?\s*(\d+[\.,]\d+)", re.IGNORECASE),
                re.compile(r"Price\s*/\s*Book\s*[:=]?\s*(\d+[\.,]\d+)", re.IGNORECASE),
                re.compile(r"trading at\s+(\d+[\.,]\d+)x\s+book", re.IGNORECASE),
            ],
            "returnOnEquity": [
                re.compile(r"(?:ROE|Return on Equity).*?(\d+[\.,]\d+)%?", re.IGNORECASE)
            ],
            "marketCap": [
                re.compile(
                    r"(?:Market Cap|Valuation).*?(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d+)?)\s*([TBM])",
                    re.IGNORECASE,
                )
            ],
            "enterpriseToEbitda": [
                re.compile(
                    r"(?:EV/EBITDA|Enterprise Value/EBITDA)(?:.*?)\s*[:=]?\s*(\d+[\.,]\d+)",
                    re.IGNORECASE,
                ),
                re.compile(r"EV/EBITDA.*?(\d+[\.,]\d+)x", re.IGNORECASE),
            ],
            "numberOfAnalystOpinions": [
                re.compile(r"(\d+)\s+analyst(?:s)?\s+cover", re.IGNORECASE),
                re.compile(r"covered\s+by\s+(\d+)\s+analyst", re.IGNORECASE),
                re.compile(r"(\d+)\s+analyst(?:s)?\s+rating", re.IGNORECASE),
                re.compile(r"analyst\s+coverage:\s*(\d+)", re.IGNORECASE),
                re.compile(r"based\s+on\s+(\d+)\s+analyst", re.IGNORECASE),
                re.compile(r"consensus.*?(\d+)\s+analyst", re.IGNORECASE),
                re.compile(r"(\d+)\s+wall\s+street\s+analyst", re.IGNORECASE),
            ],
            "us_revenue_pct": [
                re.compile(r"US\s+revenue\s+.*?\s+(\d+(?:\.\d+)?)%", re.IGNORECASE),
                re.compile(
                    r"North\s+America\s+revenue\s+.*?\s+(\d+(?:\.\d+)?)%", re.IGNORECASE
                ),
                re.compile(
                    r"revenue\s+from\s+.*?Americas.*?\s+(\d+(?:\.\d+)?)%", re.IGNORECASE
                ),
            ],
        }

        self.multipliers = {"T": 1e12, "B": 1e9, "M": 1e6}

    def _normalize_number(self, val_str: str) -> float:
        try:
            val_str = val_str.strip()
            val_str = re.sub(r"[xX%]$", "", val_str).strip()

            if "," in val_str and "." in val_str:
                if val_str.rfind(",") < val_str.rfind("."):
                    clean_str = val_str.replace(",", "")
                else:
                    clean_str = val_str.replace(".", "").replace(",", ".")
            elif "," in val_str:
                if re.match(r"^\d{1,3},\d{3}$", val_str):
                    clean_str = val_str.replace(",", "")
                else:
                    clean_str = val_str.replace(",", ".")
            else:
                clean_str = val_str

            return float(clean_str)
        except ValueError:
            return 0.0

    def extract_from_text(
        self, content: str, skip_fields: set[str] | None = None
    ) -> dict[str, Any]:
        skip_fields = skip_fields or set()
        extracted = {}

        for field, pattern_list in self.patterns.items():
            if field != "forwardPE" and field in skip_fields:
                continue

            for pattern in pattern_list:
                match = pattern.search(content)
                if match:
                    try:
                        val_str = match.group(1)
                        val = self._normalize_number(val_str)

                        if field == "returnOnEquity" and val > ROE_PERCENTAGE_THRESHOLD:
                            val = val / 100.0
                        elif field == "marketCap":
                            suffix = match.group(2).upper()
                            multiplier = self.multipliers.get(suffix, 1)
                            val = val * multiplier
                        elif field == "numberOfAnalystOpinions":
                            val = int(val)
                            if val < 0 or val > 200:
                                continue

                        extracted[field] = val
                        extracted[f"_{field}_source"] = "web_search_extraction"
                        break
                    except (ValueError, IndexError):
                        continue

        if (
            "trailingPE" not in skip_fields
            and "trailingPE" not in extracted
            and "forwardPE" in extracted
        ):
            extracted["trailingPE"] = extracted["forwardPE"]
            extracted["_trailingPE_source"] = "proxy_from_forward_pe"

        return extracted
