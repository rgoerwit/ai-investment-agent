"""
Red-Flag Detector for Catastrophic Financial Risk Pre-Screening

This module implements deterministic threshold-based validation to catch extreme
financial risks before they enter the bull/bear debate phase. Uses regex parsing
of the Fundamentals Analyst's DATA_BLOCK output.

Two categories of flags:
1. CRITICAL (AUTO_REJECT): Financial viability issues - extreme leverage, earnings fraud, refinancing risk
2. WARNING (RISK_PENALTY): Tax/legal issues - PFIC probable, VIE structure (not company viability)

Why code-driven instead of LLM-driven:
- Exact thresholds required (D/E > 500%, not "very high")
- Fast-fail pattern (avoid LLM calls for doomed stocks)
- Reliability (no hallucination risk on number parsing)
- Cost savings (~60% token reduction for rejected stocks)
- Deterministic JSON parsing for legal_report

Pattern matches: src/data/validator.py (also code-driven for same reasons)
"""

import json
import re
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class Sector(Enum):
    """Sector classifications for sector-aware red flag detection."""

    GENERAL = "General/Diversified"
    BANKING = "Banking"
    UTILITIES = "Utilities"
    SHIPPING = "Shipping & Cyclical Commodities"
    TECHNOLOGY = "Technology & Software"


class RedFlagDetector:
    """
    Deterministic pre-screening for catastrophic financial and legal risks.

    CRITICAL RED FLAGS (AUTO_REJECT - financial viability):
    1. Extreme Leverage: D/E > 500% (bankruptcy risk)
    2. Earnings Quality: Positive income but negative FCF >2x (fraud indicator)
    3. Refinancing Risk: Interest coverage <2.0x AND D/E >100% (default risk)
    4. Unsustainable Distribution: Payout >100% + uncovered dividend + weak ROIC + not improving
       (structural value destruction - dividend funded by debt with no recovery path)

    WARNING FLAGS (RISK_PENALTY - tax/legal, not viability):
    5. PFIC Probable: Company likely classified as PFIC (US tax reporting burden)
    6. VIE Structure: China stock uses contractual VIE structure (ownership risk)
    7. Unsustainable Distribution (recoverable): Payout >100% + uncovered but ROIC okay or improving
    """

    @staticmethod
    def detect_sector(fundamentals_report: str) -> Sector:
        """
        Detect sector from Fundamentals Analyst report.

        Looks for SECTOR field in DATA_BLOCK. Falls back to GENERAL if not found.

        Args:
            fundamentals_report: Full fundamentals analyst report text

        Returns:
            Sector enum value
        """
        if not fundamentals_report:
            return Sector.GENERAL

        # Extract SECTOR from DATA_BLOCK
        sector_match = re.search(r"SECTOR:\s*(.+?)(?:\n|$)", fundamentals_report)

        if not sector_match:
            logger.debug("no_sector_found_in_report", fallback="GENERAL")
            return Sector.GENERAL

        sector_text = sector_match.group(1).strip()

        # Map to enum
        if "Banking" in sector_text or "Bank" in sector_text:
            return Sector.BANKING
        elif "Utilities" in sector_text or "Utility" in sector_text:
            return Sector.UTILITIES
        elif (
            "Shipping" in sector_text
            or "Commodities" in sector_text
            or "Cyclical" in sector_text
        ):
            return Sector.SHIPPING
        elif "Technology" in sector_text or "Software" in sector_text:
            return Sector.TECHNOLOGY
        else:
            return Sector.GENERAL

    @staticmethod
    def extract_metrics(fundamentals_report: str) -> dict[str, Any]:
        """
        Extract financial metrics from Fundamentals Analyst DATA_BLOCK.

        Parses the structured DATA_BLOCK output to extract key metrics for
        red-flag detection. Uses the LAST DATA_BLOCK if multiple exist
        (handles agent self-correction pattern).

        Args:
            fundamentals_report: Full fundamentals analyst report text

        Returns:
            Dict with extracted metrics (values are None if not found):
            - debt_to_equity: D/E ratio as decimal (e.g., 500% -> 500.0)
            - net_income: Net income (if available)
            - fcf: Free cash flow
            - interest_coverage: Interest coverage ratio
            - pe_ratio: P/E ratio (TTM)
            - adjusted_health_score: Health score percentage (0-100)

        Example DATA_BLOCK format:
            ### --- START DATA_BLOCK ---
            RAW_HEALTH_SCORE: 7/12
            ADJUSTED_HEALTH_SCORE: 58% (7/12 available)
            PE_RATIO_TTM: 12.34
            ### --- END DATA_BLOCK ---
        """
        metrics: dict[str, Any] = {
            "debt_to_equity": None,
            "net_income": None,
            "fcf": None,
            "interest_coverage": None,
            "pe_ratio": None,
            "adjusted_health_score": None,
            "payout_ratio": None,
            "dividend_coverage": None,
            "roic_quality": None,
            "profitability_trend": None,
            "_raw_report": fundamentals_report,  # For downstream data quality checks
        }

        if not fundamentals_report:
            return metrics

        # Extract the LAST DATA_BLOCK (agent self-correction pattern)
        data_block_pattern = (
            r"### --- START DATA_BLOCK ---(.+?)### --- END DATA_BLOCK ---"
        )
        blocks = list(re.finditer(data_block_pattern, fundamentals_report, re.DOTALL))

        if not blocks:
            logger.warning("no_data_block_found_in_fundamentals_report")
            return metrics

        # Use the last (most corrected) block
        data_block = blocks[-1].group(1)

        # Extract ADJUSTED_HEALTH_SCORE (percentage)
        health_match = re.search(
            r"ADJUSTED_HEALTH_SCORE:\s*(\d+(?:\.\d+)?)%", data_block
        )
        if health_match:
            metrics["adjusted_health_score"] = float(health_match.group(1))

        # Extract PE_RATIO_TTM
        pe_match = re.search(r"PE_RATIO_TTM:\s*([0-9.]+)", data_block)
        if pe_match:
            metrics["pe_ratio"] = float(pe_match.group(1))

        # Extract PAYOUT_RATIO (percentage)
        payout_match = re.search(
            r"PAYOUT_RATIO:\s*(\d+(?:\.\d+)?)%", data_block, re.IGNORECASE
        )
        if payout_match:
            metrics["payout_ratio"] = float(payout_match.group(1))

        # Extract DIVIDEND_COVERAGE (categorical)
        coverage_match = re.search(
            r"DIVIDEND_COVERAGE:\s*(COVERED|PARTIAL|UNCOVERED|N/A)",
            data_block,
            re.IGNORECASE,
        )
        if coverage_match:
            val = coverage_match.group(1).upper()
            if val != "N/A":
                metrics["dividend_coverage"] = val

        # Extract ROIC_QUALITY (categorical - for compound checks)
        roic_quality_match = re.search(
            r"ROIC_QUALITY:\s*(STRONG|ADEQUATE|WEAK|DESTRUCTIVE|N/A)",
            data_block,
            re.IGNORECASE,
        )
        if roic_quality_match:
            val = roic_quality_match.group(1).upper()
            if val != "N/A":
                metrics["roic_quality"] = val

        # Extract PROFITABILITY_TREND (categorical - for compound checks)
        trend_match = re.search(
            r"PROFITABILITY_TREND:\s*(IMPROVING|STABLE|DECLINING|UNSTABLE|N/A)",
            data_block,
            re.IGNORECASE,
        )
        if trend_match:
            val = trend_match.group(1).upper()
            if val != "N/A":
                metrics["profitability_trend"] = val

        # Now extract from detailed sections (below DATA_BLOCK)
        metrics["debt_to_equity"] = RedFlagDetector._extract_debt_to_equity(
            fundamentals_report
        )
        metrics["interest_coverage"] = RedFlagDetector._extract_interest_coverage(
            fundamentals_report
        )
        metrics["fcf"] = RedFlagDetector._extract_free_cash_flow(fundamentals_report)
        metrics["net_income"] = RedFlagDetector._extract_net_income(fundamentals_report)

        return metrics

    @staticmethod
    def _extract_debt_to_equity(report: str) -> float | None:
        """
        Extract D/E ratio, converting from ratio to percentage if needed.

        Handles multiple format variations:
        - "D/E: 250" (already percentage)
        - "Debt/Equity: 2.5" (ratio format, converts to 250%)
        - Supports both markdown bold (**) and plain text

        Args:
            report: Full fundamentals report text

        Returns:
            D/E ratio as percentage (e.g., 250.0), or None if not found
        """
        patterns = [
            r"(?:^|\n)\s*-?\s*D/E:\s*([0-9.]+)",
            r"(?:^|\n)\s*-?\s*Debt/Equity:\s*([0-9.]+)",
            r"(?:^|\n)\s*-?\s*Debt-to-Equity:\s*([0-9.]+)",
            r"D/E:\s*([0-9.]+)",
            r"Debt/Equity:\s*([0-9.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
            if match:
                value = float(match.group(1))
                # Convert to percentage if < 10 (assume ratio like 2.5 -> 250%)
                return value if value >= 10 else value * 100
        return None

    @staticmethod
    def _extract_interest_coverage(report: str) -> float | None:
        """
        Extract interest coverage ratio.

        Searches for patterns like:
        - "Interest Coverage: 3.5x"
        - "**Interest Coverage**: 3.5"

        Args:
            report: Full fundamentals report text

        Returns:
            Interest coverage ratio (e.g., 3.5), or None if not found
        """
        patterns = [
            r"\*\*Interest Coverage\*\*:\s*([0-9.]+)x?",
            r"Interest Coverage:\s*([0-9.]+)x?",
            r"Interest Coverage Ratio:\s*([0-9.]+)x?",
        ]
        for pattern in patterns:
            match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
            if match:
                return float(match.group(1))
        return None

    @staticmethod
    def _extract_free_cash_flow(report: str) -> float | None:
        """
        Extract FCF with support for negative values and B/M/K multipliers.

        Handles various formats:
        - "$1.5B" → 1,500,000,000
        - "-$850M" → -850,000,000
        - "500K" → 500,000
        - Comma-separated: "1,200.5M" → 1,200,500,000

        Args:
            report: Full fundamentals report text

        Returns:
            FCF in dollars (e.g., 1_500_000_000), or None if not found
        """
        patterns = [
            # Support multiple currencies: $, ¥, €, £ or none
            r"\*\*Free Cash Flow\*\*:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?",
            r"(?:^|\n)\s*Free Cash Flow:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?",
            r"(?:^|\n)\s*FCF:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?",
            r"(?:Free Cash Flow|FCF):\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?",
            r"Positive FCF:\s*[$¥€£]?\s*([0-9,.]+)\s*([BMK])?",  # No negative for "Positive"
        ]
        for pattern in patterns:
            match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
            if match:
                groups = match.groups()
                # Handle two different pattern structures
                if len(groups) == 2:  # "Positive FCF" pattern
                    value = float(groups[0].replace(",", ""))
                    multiplier = groups[1] if len(groups) > 1 else None
                else:  # All other patterns with sign capture
                    sign = groups[0]  # '+' or '-' or ''
                    value = float(groups[1].replace(",", ""))
                    if sign == "-":
                        value = -value
                    multiplier = groups[2] if len(groups) > 2 else None

                # Handle B/M/K multipliers
                if multiplier:
                    if multiplier.upper() == "B":
                        value *= 1_000_000_000
                    elif multiplier.upper() == "M":
                        value *= 1_000_000
                    elif multiplier.upper() == "K":
                        value *= 1_000
                return value
        return None

    @staticmethod
    def _extract_net_income(report: str) -> float | None:
        """
        Extract net income with support for negative values and B/M/K multipliers.

        Handles various formats:
        - "$500M" → 500,000,000
        - "-$200M" → -200,000,000
        - "1.2B" → 1,200,000,000

        Args:
            report: Full fundamentals report text

        Returns:
            Net income in dollars (e.g., 500_000_000), or None if not found
        """
        patterns = [
            # Support multiple currencies: $, ¥, €, £ or none
            r"\*\*Net Income\*\*:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?",
            r"(?:^|\n)\s*Net Income:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?",
            r"Net Income:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?",
        ]
        for pattern in patterns:
            match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
            if match:
                groups = match.groups()
                sign = groups[0]  # '+' or '-' or ''
                value = float(groups[1].replace(",", ""))
                if sign == "-":
                    value = -value
                multiplier = groups[2] if len(groups) > 2 else None

                # Handle B/M/K multipliers
                if multiplier:
                    if multiplier.upper() == "B":
                        value *= 1_000_000_000
                    elif multiplier.upper() == "M":
                        value *= 1_000_000
                    elif multiplier.upper() == "K":
                        value *= 1_000
                return value
        return None

    @staticmethod
    def detect_red_flags(
        metrics: dict[str, float | None],
        ticker: str = "UNKNOWN",
        sector: Sector = Sector.GENERAL,
    ) -> tuple[list[dict], str]:
        """
        Apply sector-aware threshold-based red-flag detection logic.

        Args:
            metrics: Extracted financial metrics
            ticker: Ticker symbol for logging
            sector: Sector classification (affects D/E and coverage thresholds)

        Returns:
            Tuple of (red_flags_list, "PASS" or "REJECT")

        Red-flag criteria (sector-adjusted):
        1. D/E > SECTOR_THRESHOLD: Extreme leverage (bankruptcy risk)
        2. Positive income but negative FCF >2x income: Earnings quality (fraud)
        3. Interest coverage < SECTOR_THRESHOLD AND D/E > SECTOR_THRESHOLD: Refinancing risk

        Sector-specific thresholds:
        - GENERAL: D/E > 500%, Interest Coverage < 2.0x + D/E > 100%
        - UTILITIES/SHIPPING: D/E > 800%, Interest Coverage < 1.5x + D/E > 200%
        - BANKING: D/E check DISABLED (leverage is their business model)
        - TECHNOLOGY: Standard thresholds (D/E > 500%)
        """
        red_flags = []

        # Define sector-specific thresholds
        if sector == Sector.BANKING:
            # Banks: Leverage is their business model - skip D/E checks entirely
            leverage_threshold = None
            coverage_threshold = None
            coverage_de_threshold = None
        elif sector in (Sector.UTILITIES, Sector.SHIPPING):
            # Capital-intensive sectors: Higher thresholds
            leverage_threshold = 800  # D/E > 800% is extreme (vs 500% standard)
            coverage_threshold = 1.5  # Interest coverage < 1.5x (vs 2.0x standard)
            coverage_de_threshold = (
                200  # D/E > 200% when coverage weak (vs 100% standard)
            )
        else:
            # General/Technology: Standard thresholds
            leverage_threshold = 500
            coverage_threshold = 2.0
            coverage_de_threshold = 100

        # --- RED FLAG 1: Extreme Leverage (Leverage Bomb) ---
        debt_to_equity = metrics.get("debt_to_equity")
        if (
            leverage_threshold is not None
            and debt_to_equity is not None
            and debt_to_equity > leverage_threshold
        ):
            red_flags.append(
                {
                    "type": "EXTREME_LEVERAGE",
                    "severity": "CRITICAL",
                    "detail": f"D/E ratio {debt_to_equity:.1f}% is extreme (>{leverage_threshold}% threshold for {sector.value})",
                    "action": "AUTO_REJECT",
                    "rationale": f"Leverage exceeds sector-appropriate threshold - bankruptcy risk (sector: {sector.value})",
                }
            )
            logger.warning(
                "red_flag_extreme_leverage",
                ticker=ticker,
                debt_to_equity=debt_to_equity,
                threshold=leverage_threshold,
                sector=sector.value,
            )

        # --- RED FLAG 2: Earnings Quality Disconnect ---
        net_income = metrics.get("net_income")
        fcf = metrics.get("fcf")

        # Check if FCF data quality is flagged as uncertain in the report
        fcf_data_uncertain = "FCF DATA QUALITY UNCERTAIN" in (
            metrics.get("_raw_report", "") or ""
        )

        if (
            net_income is not None
            and net_income > 0
            and fcf is not None
            and fcf < 0
            and abs(fcf) > (2 * net_income)
        ):
            disconnect_ratio = abs(fcf / net_income) if net_income != 0 else 0

            # Downgrade to WARNING if FCF data quality uncertain or ratio extreme (>4x)
            if fcf_data_uncertain or disconnect_ratio > 4.0:
                red_flags.append(
                    {
                        "type": "EARNINGS_QUALITY_UNCERTAIN",
                        "severity": "WARNING",
                        "detail": f"NI ${net_income:,.0f} but FCF ${fcf:,.0f} ({disconnect_ratio:.1f}x) - data quality uncertain",
                        "action": "RISK_PENALTY",
                        "risk_penalty": 1.0,
                        "rationale": "FCF/NI disconnect may reflect TTM data misalignment, not fraud",
                    }
                )
            else:
                # Standard earnings quality flag (2-4x is suspicious but plausible)
                red_flags.append(
                    {
                        "type": "EARNINGS_QUALITY",
                        "severity": "CRITICAL",
                        "detail": f"Positive net income (${net_income:,.0f}) but negative FCF (${fcf:,.0f}) >2x income",
                        "action": "AUTO_REJECT",
                        "rationale": "Earnings likely fabricated through accounting tricks - FCF disconnect",
                    }
                )
                logger.warning(
                    "red_flag_earnings_quality",
                    ticker=ticker,
                    net_income=net_income,
                    fcf=fcf,
                    disconnect_multiple=disconnect_ratio,
                )

        # --- RED FLAG 3: Interest Coverage Death Spiral (Sector-Aware) ---
        interest_coverage = metrics.get("interest_coverage")

        # Only apply if sector has defined thresholds (excludes banking)
        if (
            coverage_threshold is not None
            and coverage_de_threshold is not None
            and interest_coverage is not None
            and interest_coverage < coverage_threshold
            and debt_to_equity is not None
            and debt_to_equity > coverage_de_threshold
        ):
            red_flags.append(
                {
                    "type": "REFINANCING_RISK",
                    "severity": "CRITICAL",
                    "detail": f"Interest coverage {interest_coverage:.2f}x with {debt_to_equity:.1f}% D/E ratio (thresholds: <{coverage_threshold}x coverage + >{coverage_de_threshold}% D/E for {sector.value})",
                    "action": "AUTO_REJECT",
                    "rationale": f"Cannot comfortably service debt - refinancing/default risk (sector: {sector.value})",
                }
            )
            logger.warning(
                "red_flag_refinancing_risk",
                ticker=ticker,
                interest_coverage=interest_coverage,
                debt_to_equity=debt_to_equity,
                coverage_threshold=coverage_threshold,
                de_threshold=coverage_de_threshold,
                sector=sector.value,
            )

        # --- RED FLAG 4: Unsustainable Distribution ---
        # Dividend exceeds earnings AND FCF can't cover it = structural problem
        payout_ratio = metrics.get("payout_ratio")
        dividend_coverage = metrics.get("dividend_coverage")
        roic_quality = metrics.get("roic_quality")
        profitability_trend = metrics.get("profitability_trend")

        if (
            payout_ratio is not None
            and payout_ratio > 100
            and dividend_coverage == "UNCOVERED"
        ):
            # Check if also value-destroying (ROIC weak/destructive AND not improving)
            is_value_destroying = roic_quality in ("WEAK", "DESTRUCTIVE")
            is_recovering = profitability_trend == "IMPROVING"

            if is_value_destroying and not is_recovering:
                # Hard fail: can't distribute >earnings while destroying value with no recovery
                red_flags.append(
                    {
                        "type": "UNSUSTAINABLE_DISTRIBUTION",
                        "severity": "CRITICAL",
                        "detail": f"Payout {payout_ratio:.0f}% + uncovered dividend + ROIC {roic_quality} + trend {profitability_trend}",
                        "action": "AUTO_REJECT",
                        "rationale": "Dividend exceeds earnings, FCF doesn't cover it, ROIC below hurdle, "
                        "and no improving trend. Mathematically unsustainable value destruction.",
                    }
                )
                logger.warning(
                    "red_flag_unsustainable_distribution_critical",
                    ticker=ticker,
                    payout_ratio=payout_ratio,
                    dividend_coverage=dividend_coverage,
                    roic_quality=roic_quality,
                    profitability_trend=profitability_trend,
                )
            else:
                # Warning: distribution is stretched but may be fixable (company can cut dividend,
                # or is in cyclical recovery)
                red_flags.append(
                    {
                        "type": "UNSUSTAINABLE_DISTRIBUTION",
                        "severity": "WARNING",
                        "detail": f"Payout {payout_ratio:.0f}% with {dividend_coverage} dividend coverage",
                        "action": "RISK_PENALTY",
                        "risk_penalty": 1.5,
                        "rationale": "Dividend funded by debt/reserves. Watch for dividend cut or "
                        "verify cyclical recovery thesis if ROIC improving.",
                    }
                )
                logger.info(
                    "red_flag_unsustainable_distribution_warning",
                    ticker=ticker,
                    payout_ratio=payout_ratio,
                    dividend_coverage=dividend_coverage,
                    roic_quality=roic_quality,
                )

        # Determine result
        has_auto_reject = any(flag["action"] == "AUTO_REJECT" for flag in red_flags)
        result = "REJECT" if has_auto_reject else "PASS"

        return red_flags, result

    @staticmethod
    def extract_legal_risks(legal_report: str) -> dict[str, Any]:
        """
        Extract legal/tax risk data from Legal Counsel's JSON output.

        Args:
            legal_report: Legal Counsel output (JSON string or raw text)

        Returns:
            Dict with extracted legal risks:
            - pfic_status: CLEAN/UNCERTAIN/PROBABLE/N/A
            - pfic_evidence: Quote from search results
            - vie_structure: YES/NO/N/A
            - vie_evidence: Description if VIE detected
            - cmic_status: FLAGGED/UNCERTAIN/CLEAR/N/A
            - cmic_evidence: Description if CMIC risk detected
            - other_regulatory_risks: List of {risk_type, description, severity}
            - country: Country of domicile
            - sector: Sector name
        """
        risks: dict[str, Any] = {
            "pfic_status": None,
            "pfic_evidence": None,
            "vie_structure": None,
            "vie_evidence": None,
            "cmic_status": None,
            "cmic_evidence": None,
            "other_regulatory_risks": [],
            "country": None,
            "sector": None,
        }

        if not legal_report:
            return risks

        # Try to parse as JSON first (preferred)
        try:
            # Handle potential markdown code blocks
            json_str = legal_report.strip()
            if json_str.startswith("```"):
                # Extract JSON from markdown code block
                lines = json_str.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```") and not in_block:
                        in_block = True
                        continue
                    elif line.startswith("```") and in_block:
                        break
                    elif in_block:
                        json_lines.append(line)
                json_str = "\n".join(json_lines)

            data = json.loads(json_str)
            risks["pfic_status"] = data.get("pfic_status")
            risks["pfic_evidence"] = data.get("pfic_evidence")
            risks["vie_structure"] = data.get("vie_structure")
            risks["vie_evidence"] = data.get("vie_evidence")
            risks["cmic_status"] = data.get("cmic_status")
            risks["cmic_evidence"] = data.get("cmic_evidence")
            risks["other_regulatory_risks"] = data.get("other_regulatory_risks") or []
            risks["country"] = data.get("country")
            risks["sector"] = data.get("sector")

            logger.debug(
                "legal_risks_parsed_json",
                pfic_status=risks["pfic_status"],
                vie_structure=risks["vie_structure"],
                cmic_status=risks["cmic_status"],
            )
            return risks

        except json.JSONDecodeError:
            # Fall back to regex parsing
            logger.debug("legal_report_not_json_falling_back_to_regex")

        # Regex fallback for non-JSON output
        pfic_match = re.search(
            r'"?pfic_status"?\s*:\s*"?(CLEAN|UNCERTAIN|PROBABLE|N/A)"?',
            legal_report,
            re.IGNORECASE,
        )
        if pfic_match:
            risks["pfic_status"] = pfic_match.group(1).upper()

        vie_match = re.search(
            r'"?vie_structure"?\s*:\s*"?(YES|NO|N/A)"?', legal_report, re.IGNORECASE
        )
        if vie_match:
            risks["vie_structure"] = vie_match.group(1).upper()

        cmic_match = re.search(
            r'"?cmic_status"?\s*:\s*"?(FLAGGED|UNCERTAIN|CLEAR|N/A)"?',
            legal_report,
            re.IGNORECASE,
        )
        if cmic_match:
            risks["cmic_status"] = cmic_match.group(1).upper()

        return risks

    @staticmethod
    def detect_legal_flags(
        legal_risks: dict[str, Any], ticker: str = "UNKNOWN"
    ) -> list[dict]:
        """
        Detect legal/tax warning flags from Legal Counsel output.

        Unlike financial red flags (which trigger AUTO_REJECT), legal flags
        add risk penalties but do NOT reject the stock. PFIC is a tax burden,
        not a viability issue. CMIC is currently a high penalty (2.0) but not
        auto-reject since restrictions may change - modify severity/action
        in the code if stricter enforcement is needed.

        Flags detected:
        - PFIC_PROBABLE: 1.0 penalty (tax burden)
        - PFIC_UNCERTAIN: 0.5 penalty
        - VIE_STRUCTURE: 0.5 penalty (ownership risk)
        - CMIC_FLAGGED: 2.0 penalty (near-prohibition, configurable)
        - CMIC_UNCERTAIN: 1.0 penalty
        - REGULATORY_*: 0.5-1.5 penalty based on severity (open-ended)

        Args:
            legal_risks: Extracted legal risk data from extract_legal_risks()
            ticker: Ticker symbol for logging

        Returns:
            List of warning flag dicts with risk_penalty values
        """
        warnings = []

        pfic_status = legal_risks.get("pfic_status")
        vie_structure = legal_risks.get("vie_structure")
        pfic_evidence = legal_risks.get("pfic_evidence") or "No evidence provided"

        # --- WARNING 1: PFIC Probable ---
        if pfic_status == "PROBABLE":
            warnings.append(
                {
                    "type": "PFIC_PROBABLE",
                    "severity": "WARNING",
                    "detail": f"Company likely classified as PFIC. Evidence: {pfic_evidence[:100]}...",
                    "action": "RISK_PENALTY",
                    "risk_penalty": 1.0,  # Add 1.0 to risk tally
                    "rationale": "PFIC classification requires onerous US tax reporting (Form 8621). "
                    "Mark-to-market or QEF election required. Not a viability issue, "
                    "but increases compliance burden for US investors.",
                }
            )
            logger.warning(
                "legal_flag_pfic_probable", ticker=ticker, evidence=pfic_evidence[:50]
            )

        # --- WARNING 2: PFIC Uncertain (lesser penalty) ---
        elif pfic_status == "UNCERTAIN":
            warnings.append(
                {
                    "type": "PFIC_UNCERTAIN",
                    "severity": "WARNING",
                    "detail": f"PFIC status unclear. Evidence: {pfic_evidence[:100]}...",
                    "action": "RISK_PENALTY",
                    "risk_penalty": 0.5,  # Add 0.5 to risk tally
                    "rationale": "PFIC status cannot be confirmed. Company may use hedge language "
                    "or is in a high-risk sector without clear disclosure. "
                    "Recommend consulting tax advisor before investing.",
                }
            )
            logger.info(
                "legal_flag_pfic_uncertain", ticker=ticker, evidence=pfic_evidence[:50]
            )

        # --- WARNING 3: VIE Structure ---
        if vie_structure == "YES":
            vie_evidence = legal_risks.get("vie_evidence") or "VIE structure detected"
            warnings.append(
                {
                    "type": "VIE_STRUCTURE",
                    "severity": "WARNING",
                    "detail": f"Company uses VIE contractual structure for China operations. {vie_evidence[:80]}",
                    "action": "RISK_PENALTY",
                    "risk_penalty": 0.5,  # Add 0.5 to risk tally
                    "rationale": "VIE structure means investors own contracts, not equity. "
                    "China regulatory risk if VIE agreements are invalidated. "
                    "Common for China tech/education stocks but adds legal uncertainty.",
                }
            )
            logger.warning(
                "legal_flag_vie_structure", ticker=ticker, evidence=vie_evidence[:50]
            )

        # --- WARNING 4: CMIC Flagged (configurable severity) ---
        # CMIC = Chinese Military-Industrial Complex (NS-CMIC list)
        # Default: HIGH risk penalty but NOT auto-reject (restrictions may change)
        # To escalate: change severity to "CRITICAL" and action to "AUTO_REJECT"
        cmic_status = legal_risks.get("cmic_status")
        if cmic_status == "FLAGGED":
            cmic_evidence = legal_risks.get("cmic_evidence") or "NS-CMIC list match"
            warnings.append(
                {
                    "type": "CMIC_FLAGGED",
                    "severity": "HIGH",  # Change to "CRITICAL" for auto-reject
                    "detail": f"Company appears on NS-CMIC list. {cmic_evidence[:80]}",
                    "action": "RISK_PENALTY",  # Change to "AUTO_REJECT" if needed
                    "risk_penalty": 2.0,  # High penalty - near-prohibition
                    "rationale": "US Executive Orders prohibit US persons from investing in "
                    "NS-CMIC listed companies. Verify current OFAC status before investing. "
                    "Restrictions may be modified by future executive orders.",
                }
            )
            logger.warning(
                "legal_flag_cmic_flagged", ticker=ticker, evidence=cmic_evidence[:50]
            )

        elif cmic_status == "UNCERTAIN":
            cmic_evidence = (
                legal_risks.get("cmic_evidence") or "Possible CMIC connection"
            )
            warnings.append(
                {
                    "type": "CMIC_UNCERTAIN",
                    "severity": "WARNING",
                    "detail": f"Possible CMIC connection. {cmic_evidence[:80]}",
                    "action": "RISK_PENALTY",
                    "risk_penalty": 1.0,
                    "rationale": "Company may have ties to Chinese military-industrial complex. "
                    "Recommend verifying against current OFAC NS-CMIC list before investing.",
                }
            )
            logger.info(
                "legal_flag_cmic_uncertain", ticker=ticker, evidence=cmic_evidence[:50]
            )

        # --- WARNING 5+: Other Regulatory Risks (open-ended) ---
        other_risks = legal_risks.get("other_regulatory_risks") or []
        severity_penalties = {"HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5}

        for risk in other_risks:
            if not isinstance(risk, dict):
                continue
            risk_type = risk.get("risk_type", "OTHER")
            description = risk.get("description", "Regulatory risk detected")
            severity = risk.get("severity", "MEDIUM").upper()
            penalty = severity_penalties.get(severity, 1.0)

            warnings.append(
                {
                    "type": f"REGULATORY_{risk_type}",
                    "severity": "WARNING" if severity != "HIGH" else "HIGH",
                    "detail": f"{risk_type}: {description[:100]}",
                    "action": "RISK_PENALTY",
                    "risk_penalty": penalty,
                    "rationale": f"Regulatory risk identified by Legal Counsel. "
                    f"Type: {risk_type}, Severity: {severity}. Review before investing.",
                }
            )
            logger.info(
                "legal_flag_other_regulatory",
                ticker=ticker,
                risk_type=risk_type,
                severity=severity,
            )

        return warnings

    @staticmethod
    def extract_value_trap_score(value_trap_report: str) -> dict[str, Any]:
        """
        Extract key metrics from Value Trap Detector's VALUE_TRAP_BLOCK output.

        Args:
            value_trap_report: Full Value Trap Detector report text

        Returns:
            Dict with extracted metrics:
            - score: 0-100 score (None if not found)
            - verdict: TRAP|CAUTIOUS|WATCHABLE|ALIGNED (None if not found)
            - trap_risk: HIGH|MEDIUM|LOW (None if not found)
            - activist_present: YES|NO|RUMORED (None if not found)
            - insider_trend: NET_BUYER|NET_SELLER|NEUTRAL (None if not found)
        """
        metrics: dict[str, Any] = {
            "score": None,
            "verdict": None,
            "trap_risk": None,
            "activist_present": None,
            "insider_trend": None,
            "has_catalyst": False,
        }

        if not value_trap_report:
            return metrics

        # Handle non-string input (LangGraph can pass list/dict from state pollution)
        if not isinstance(value_trap_report, str):
            try:
                # Try to convert to string
                value_trap_report = str(value_trap_report)
            except Exception:
                return metrics

        # Extract SCORE (case-insensitive, handles "SCORE: 35", "SCORE: 35/100", "SCORE: 35%")
        score_match = re.search(
            r"SCORE:\s*(\d+)(?:/100|%)?", value_trap_report, re.IGNORECASE
        )
        if score_match:
            score = int(score_match.group(1))
            # Clamp to valid range (0-100) to handle LLM hallucinations
            metrics["score"] = max(0, min(100, score))

        # Extract VERDICT
        verdict_match = re.search(
            r"VERDICT:\s*(TRAP|CAUTIOUS|WATCHABLE|ALIGNED)",
            value_trap_report,
            re.IGNORECASE,
        )
        if verdict_match:
            metrics["verdict"] = verdict_match.group(1).upper()

        # Extract TRAP_RISK
        risk_match = re.search(
            r"TRAP_RISK:\s*(HIGH|MEDIUM|LOW)", value_trap_report, re.IGNORECASE
        )
        if risk_match:
            metrics["trap_risk"] = risk_match.group(1).upper()

        # Extract ACTIVIST_PRESENT
        activist_match = re.search(
            r"ACTIVIST_PRESENT:\s*(YES|NO|RUMORED)", value_trap_report, re.IGNORECASE
        )
        if activist_match:
            metrics["activist_present"] = activist_match.group(1).upper()

        # Extract INSIDER_TREND
        insider_match = re.search(
            r"INSIDER_TREND:\s*(NET_BUYER|NET_SELLER|NEUTRAL|UNKNOWN)",
            value_trap_report,
            re.IGNORECASE,
        )
        if insider_match:
            metrics["insider_trend"] = insider_match.group(1).upper()

        # Check for catalysts (any non-NONE value in CATALYSTS section)
        catalysts_section = re.search(
            r"CATALYSTS:(.+?)(?:KEY_RISKS:|$)", value_trap_report, re.DOTALL
        )
        if catalysts_section:
            catalyst_text = catalysts_section.group(1)
            # Check if any catalyst field has a value other than NONE
            if re.search(
                r"(?:INDEX_CANDIDATE|ACTIVIST_RUMOR|RESTRUCTURING|MID_TERM_PLAN):\s*(?!NONE)[A-Za-z]",
                catalyst_text,
            ):
                metrics["has_catalyst"] = True

        logger.debug(
            "value_trap_metrics_extracted",
            score=metrics["score"],
            verdict=metrics["verdict"],
            trap_risk=metrics["trap_risk"],
        )

        return metrics

    @staticmethod
    def detect_value_trap_flags(
        value_trap_report: str, ticker: str = "UNKNOWN"
    ) -> list[dict]:
        """
        Parse VALUE_TRAP_BLOCK for deterministic warning flags.

        These are WARNINGs, not AUTO_REJECT. Governance issues don't kill companies,
        they just trap value. The Portfolio Manager should weigh these in final decision.

        Flag types:
        - VALUE_TRAP_HIGH_RISK: Score < 40 (probable trap)
        - VALUE_TRAP_MODERATE_RISK: Score 40-60 (cautious)
        - VALUE_TRAP_VERDICT: Explicit TRAP verdict from agent
        - NO_CATALYST_DETECTED: No activist, no index candidacy, no restructuring

        Args:
            value_trap_report: Full Value Trap Detector report text
            ticker: Ticker symbol for logging

        Returns:
            List of warning flag dicts with risk_penalty values
        """
        flags = []

        metrics = RedFlagDetector.extract_value_trap_score(value_trap_report)
        score = metrics.get("score")
        verdict = metrics.get("verdict")
        has_catalyst = metrics.get("has_catalyst", False)
        activist_present = metrics.get("activist_present")

        # --- WARNING 1: High Risk Value Trap (Score < 40) ---
        if score is not None and score < 40:
            flags.append(
                {
                    "type": "VALUE_TRAP_HIGH_RISK",
                    "severity": "WARNING",
                    "detail": f"Value Trap Score {score}/100 (< 40 threshold indicates probable trap)",
                    "action": "RISK_PENALTY",
                    "risk_penalty": 1.0,
                    "rationale": "Low governance score suggests entrenched ownership, "
                    "poor capital allocation, or no catalyst for re-rating.",
                }
            )
            logger.warning(
                "value_trap_flag_high_risk",
                ticker=ticker,
                score=score,
                verdict=verdict,
            )

        # --- WARNING 2: Moderate Risk Value Trap (Score 40-60) ---
        elif score is not None and score < 60:
            flags.append(
                {
                    "type": "VALUE_TRAP_MODERATE_RISK",
                    "severity": "WARNING",
                    "detail": f"Value Trap Score {score}/100 (40-60 range indicates mixed signals)",
                    "action": "RISK_PENALTY",
                    "risk_penalty": 0.5,
                    "rationale": "Moderate governance concerns. Some trap characteristics "
                    "present but not conclusive. Monitor for catalyst development.",
                }
            )
            logger.info(
                "value_trap_flag_moderate_risk",
                ticker=ticker,
                score=score,
                verdict=verdict,
            )

        # --- WARNING 3: Explicit TRAP Verdict ---
        if verdict == "TRAP":
            # Only add if we haven't already flagged for low score
            if not any(f["type"] == "VALUE_TRAP_HIGH_RISK" for f in flags):
                flags.append(
                    {
                        "type": "VALUE_TRAP_VERDICT",
                        "severity": "WARNING",
                        "detail": "Value Trap Detector verdict: TRAP",
                        "action": "RISK_PENALTY",
                        "risk_penalty": 1.0,
                        "rationale": "Agent assessment indicates high probability of value trap. "
                        "Stock may remain cheap indefinitely without catalyst.",
                    }
                )
                logger.warning(
                    "value_trap_flag_verdict", ticker=ticker, verdict=verdict
                )

        # --- WARNING 4: No Catalyst Detected ---
        if not has_catalyst and activist_present == "NO":
            flags.append(
                {
                    "type": "NO_CATALYST_DETECTED",
                    "severity": "WARNING",
                    "detail": "No activist presence, no index candidacy, no restructuring signals",
                    "action": "RISK_PENALTY",
                    "risk_penalty": 0.5,
                    "rationale": "Without a catalyst, cheap stocks can remain cheap. "
                    "Value realization depends on external pressure or internal change.",
                }
            )
            logger.info(
                "value_trap_flag_no_catalyst",
                ticker=ticker,
                activist_present=activist_present,
            )

        return flags

    @staticmethod
    def extract_moat_signals(fundamentals_report: str) -> dict[str, Any]:
        """
        Extract moat signal metrics from Fundamentals Analyst DATA_BLOCK.

        Moat signals are calculated by the fetcher and passed through Junior/Senior
        Fundamentals. This method extracts the categorical signals (HIGH/STRONG)
        which are used for threshold logic, plus numeric values for logging.

        IMPORTANT: Threshold decisions use categorical signals only, not numeric
        values. This avoids ratio/percentage parsing ambiguity (e.g., CFO/NI can
        legitimately exceed 1.0).

        Args:
            fundamentals_report: Full fundamentals analyst report text

        Returns:
            Dict with extracted moat metrics:
            - margin_stability: HIGH/MEDIUM/LOW or None (used for thresholds)
            - margin_cv: Coefficient of Variation (float) or None (for logging)
            - cash_conversion: STRONG/ADEQUATE/WEAK or None (used for thresholds)
            - cfo_ni_avg: Average CFO/NI ratio (float) or None (for logging)
        """
        metrics: dict[str, Any] = {
            "margin_stability": None,
            "margin_cv": None,
            "margin_avg": None,
            "cash_conversion": None,
            "cfo_ni_avg": None,
        }

        if not fundamentals_report:
            return metrics

        # Handle non-string input gracefully
        if not isinstance(fundamentals_report, str):
            try:
                fundamentals_report = str(fundamentals_report)
            except Exception:
                return metrics

        # Extract the LAST DATA_BLOCK (agent self-correction pattern)
        data_block_pattern = (
            r"### --- START DATA_BLOCK ---(.+?)### --- END DATA_BLOCK ---"
        )
        blocks = list(re.finditer(data_block_pattern, fundamentals_report, re.DOTALL))

        if not blocks:
            return metrics

        data_block = blocks[-1].group(1)

        # --- CATEGORICAL SIGNALS (Primary - used for threshold logic) ---

        # Extract MOAT_MARGIN_STABILITY
        stability_match = re.search(
            r"MOAT_MARGIN_STABILITY:\s*(HIGH|MEDIUM|LOW)", data_block, re.IGNORECASE
        )
        if stability_match:
            metrics["margin_stability"] = stability_match.group(1).upper()

        # Extract MOAT_CASH_CONVERSION
        cash_match = re.search(
            r"MOAT_CASH_CONVERSION:\s*(STRONG|ADEQUATE|WEAK)", data_block, re.IGNORECASE
        )
        if cash_match:
            metrics["cash_conversion"] = cash_match.group(1).upper()

        # --- NUMERIC VALUES (Secondary - for logging/detail only) ---
        # These are NOT used for threshold decisions to avoid parsing ambiguity

        # Extract MOAT_MARGIN_CV (handles trailing punctuation)
        cv_match = re.search(r"MOAT_MARGIN_CV:\s*([0-9]+\.?[0-9]*)", data_block)
        if cv_match:
            try:
                metrics["margin_cv"] = float(cv_match.group(1))
            except ValueError:
                pass  # Non-fatal: we have categorical signal

        # Extract MOAT_GROSS_MARGIN_AVG (handles % suffix and trailing punctuation)
        avg_match = re.search(
            r"MOAT_GROSS_MARGIN_AVG:\s*([0-9]+\.?[0-9]*)%?", data_block
        )
        if avg_match:
            try:
                value = float(avg_match.group(1))
                # Margin is always 0-100% or 0.0-1.0
                # If > 1, assume percentage format (e.g., "35.2%")
                metrics["margin_avg"] = value / 100 if value > 1 else value
            except ValueError:
                pass

        # Extract MOAT_CFO_NI_AVG (NO percentage normalization - ratio can exceed 1.0)
        cfo_match = re.search(r"MOAT_CFO_NI_AVG:\s*([0-9]+\.?[0-9]*)", data_block)
        if cfo_match:
            try:
                # Store raw value - CFO/NI can legitimately be > 1.0
                # Do NOT apply percentage normalization here
                metrics["cfo_ni_avg"] = float(cfo_match.group(1))
            except ValueError:
                pass  # Non-fatal: we have categorical signal

        logger.debug(
            "moat_signals_extracted",
            margin_stability=metrics["margin_stability"],
            cash_conversion=metrics["cash_conversion"],
        )

        return metrics

    @staticmethod
    def detect_moat_flags(
        fundamentals_report: str, ticker: str = "UNKNOWN"
    ) -> list[dict]:
        """
        Detect economic moat indicators and create BONUS flags (negative risk penalty).

        Unlike red flags (which add risk), moat flags REDUCE the risk tally when
        strong competitive advantages are detected. Based on S&P Global quantitative
        moat framework.

        IMPORTANT: Threshold logic uses categorical signals (HIGH/STRONG) only,
        not numeric values. This avoids parsing ambiguity and makes the system
        robust to LLM formatting variations.

        Flag types (all are BONUS - negative penalty):
        - MOAT_DURABLE_ADVANTAGE: Both margin stability HIGH + cash conversion STRONG (-1.0)
        - MOAT_PRICING_POWER: Margin stability HIGH alone (-0.5)
        - MOAT_EARNINGS_QUALITY: Cash conversion STRONG alone (-0.5)

        Args:
            fundamentals_report: Full fundamentals analyst report text
            ticker: Ticker symbol for logging

        Returns:
            List of moat flag dicts with negative risk_penalty values (bonuses).
            Empty list if no moat signals detected or signals are weak/medium.
        """
        flags = []

        metrics = RedFlagDetector.extract_moat_signals(fundamentals_report)

        # Use categorical signals for all threshold decisions
        margin_stability = metrics.get("margin_stability")
        cash_conversion = metrics.get("cash_conversion")

        # Numeric values for detail/logging only (not used in threshold logic)
        margin_cv = metrics.get("margin_cv")
        cfo_ni_avg = metrics.get("cfo_ni_avg")

        # --- BONUS 1: Durable Advantage (Both signals strong) ---
        if margin_stability == "HIGH" and cash_conversion == "STRONG":
            detail_parts = []
            if margin_cv is not None:
                detail_parts.append(f"Margin CV: {margin_cv:.3f}")
            if cfo_ni_avg is not None:
                detail_parts.append(f"CFO/NI: {cfo_ni_avg:.2f}")
            detail = (
                "; ".join(detail_parts) if detail_parts else "Multiple moat signals"
            )

            flags.append(
                {
                    "type": "MOAT_DURABLE_ADVANTAGE",
                    "severity": "POSITIVE",
                    "detail": f"Pricing power + earnings quality confirmed. {detail}",
                    "action": "RISK_BONUS",
                    "risk_penalty": -1.0,
                    "rationale": (
                        "Company exhibits both stable gross margins (CV < 8%) and high "
                        "cash conversion (CFO/NI > 90%) over multiple years. This combination "
                        "suggests a durable competitive advantage with pricing power."
                    ),
                }
            )
            logger.info(
                "moat_flag_durable_advantage",
                ticker=ticker,
                margin_stability=margin_stability,
                cash_conversion=cash_conversion,
            )
            # Return early - don't stack individual bonuses on top of combined
            return flags

        # --- BONUS 2: Pricing Power (Margin stability alone) ---
        if margin_stability == "HIGH":
            detail = (
                f"Gross margin CV: {margin_cv:.3f}"
                if margin_cv is not None
                else "CV < 8%"
            )
            flags.append(
                {
                    "type": "MOAT_PRICING_POWER",
                    "severity": "POSITIVE",
                    "detail": f"Stable gross margins over 5 years. {detail}",
                    "action": "RISK_BONUS",
                    "risk_penalty": -0.5,
                    "rationale": (
                        "Low gross margin volatility (CV < 8%) over 5 years suggests "
                        "pricing power. Company can maintain margins without aggressive "
                        "discounting, indicating competitive advantage."
                    ),
                }
            )
            logger.info(
                "moat_flag_pricing_power",
                ticker=ticker,
                margin_cv=margin_cv,
            )

        # --- BONUS 3: Earnings Quality (Cash conversion alone) ---
        if cash_conversion == "STRONG":
            detail = (
                f"3Y avg CFO/NI: {cfo_ni_avg:.2f}"
                if cfo_ni_avg is not None
                else "> 0.90"
            )
            flags.append(
                {
                    "type": "MOAT_EARNINGS_QUALITY",
                    "severity": "POSITIVE",
                    "detail": f"High cash conversion ratio. {detail}",
                    "action": "RISK_BONUS",
                    "risk_penalty": -0.5,
                    "rationale": (
                        "CFO/Net Income ratio averaging > 90% over 3 years indicates "
                        "reported earnings are converting to actual cash flow. Not "
                        "relying on accounting accruals or channel stuffing."
                    ),
                }
            )
            logger.info(
                "moat_flag_earnings_quality",
                ticker=ticker,
                cfo_ni_avg=cfo_ni_avg,
            )

        return flags

    @staticmethod
    def extract_capital_efficiency_signals(fundamentals_report: str) -> dict[str, Any]:
        """
        Extract capital efficiency signals from fundamentals report DATA_BLOCK.

        Parses ROIC, leverage quality, and related metrics for flag detection.
        Uses categorical signals for threshold decisions, numeric for logging.

        Args:
            fundamentals_report: Full fundamentals analyst report text

        Returns:
            Dict with extracted signals. Empty dict if DATA_BLOCK not found.
        """
        if not fundamentals_report or not isinstance(fundamentals_report, str):
            return {}

        signals: dict[str, Any] = {}

        # Find last DATA_BLOCK (in case of multiple)
        blocks = fundamentals_report.split("--- START DATA_BLOCK ---")
        if len(blocks) < 2:
            return {}

        data_block = blocks[-1].split("--- END DATA_BLOCK ---")[0]

        # Extract ROIC_QUALITY (categorical)
        roic_quality_match = re.search(
            r"ROIC_QUALITY:\s*(STRONG|ADEQUATE|WEAK|DESTRUCTIVE|N/A)",
            data_block,
            re.IGNORECASE,
        )
        if roic_quality_match:
            val = roic_quality_match.group(1).upper()
            if val != "N/A":
                signals["roic_quality"] = val

        # Extract LEVERAGE_QUALITY (categorical)
        leverage_quality_match = re.search(
            r"LEVERAGE_QUALITY:\s*(GENUINE|CONSERVATIVE|SUSPECT|ENGINEERED|VALUE_DESTRUCTION|N/A)",
            data_block,
            re.IGNORECASE,
        )
        if leverage_quality_match:
            val = leverage_quality_match.group(1).upper()
            if val != "N/A":
                signals["leverage_quality"] = val

        # Extract ROIC_PERCENT (numeric, for logging)
        roic_match = re.search(
            r"ROIC_PERCENT:\s*(-?[\d.]+)([%]?)",
            data_block,
            re.IGNORECASE,
        )
        if roic_match:
            try:
                val = float(roic_match.group(1))
                has_percent = bool(roic_match.group(2))

                # If explicit %, always divide by 100
                if has_percent:
                    val = val / 100
                # If no %, heuristic: if >= 2.0, assume it was meant as percentage
                # e.g. 15.0 -> 0.15, but 1.5 -> 1.5 (150%)
                elif abs(val) >= 2.0:
                    val = val / 100

                signals["roic"] = val
            except ValueError:
                pass

        # Extract ROE_ROIC_RATIO (numeric, for logging)
        ratio_match = re.search(
            r"ROE_ROIC_RATIO:\s*([\d.]+)",
            data_block,
            re.IGNORECASE,
        )
        if ratio_match:
            try:
                signals["roe_roic_ratio"] = float(ratio_match.group(1))
            except ValueError:
                pass

        return signals

    @staticmethod
    def detect_capital_efficiency_flags(
        fundamentals_report: str, ticker: str = "UNKNOWN"
    ) -> list[dict]:
        """
        Detect capital efficiency red flags and bonuses.

        Flags problematic patterns:
        - Value destruction (negative ROIC with positive ROE)
        - Leverage-engineered returns (ROE >> ROIC)
        - Below-hurdle ROIC (likely destroying value)

        Bonuses for:
        - Strong genuine returns (high ROIC with low leverage boost)

        Args:
            fundamentals_report: Full fundamentals analyst report text
            ticker: Ticker symbol for logging

        Returns:
            List of flag dicts with risk_penalty values (positive = risk, negative = bonus).
        """
        flags: list[dict] = []

        metrics = RedFlagDetector.extract_capital_efficiency_signals(
            fundamentals_report
        )

        if not metrics:
            return flags

        roic_quality = metrics.get("roic_quality")
        leverage_quality = metrics.get("leverage_quality")
        roic = metrics.get("roic")
        roe_roic_ratio = metrics.get("roe_roic_ratio")

        # --- FLAG 1: Value Destruction (most severe) ---
        if leverage_quality == "VALUE_DESTRUCTION":
            detail = f"ROIC: {roic:.1%}" if roic is not None else "Negative ROIC"
            flags.append(
                {
                    "type": "CAPITAL_VALUE_DESTRUCTION",
                    "severity": "CRITICAL",
                    "detail": f"Negative operating returns masked by leverage. {detail}",
                    "action": "REJECT_REVIEW",
                    "risk_penalty": 1.5,
                    "rationale": (
                        "Company has negative ROIC but positive ROE. This means the core "
                        "business is destroying value while financial leverage creates the "
                        "illusion of shareholder returns. Classic value trap pattern."
                    ),
                }
            )
            logger.info(
                "capital_flag_value_destruction",
                ticker=ticker,
                roic=roic,
                leverage_quality=leverage_quality,
            )
            # Don't stack other flags if value destruction detected
            return flags

        # --- FLAG 2: Leverage-Engineered Returns ---
        if leverage_quality == "ENGINEERED":
            ratio_str = f"ROE/ROIC: {roe_roic_ratio:.1f}x" if roe_roic_ratio else ""
            flags.append(
                {
                    "type": "CAPITAL_ENGINEERED_RETURNS",
                    "severity": "HIGH",
                    "detail": f"Returns primarily from financial engineering. {ratio_str}",
                    "action": "RISK_ADJUST",
                    "risk_penalty": 1.0,
                    "rationale": (
                        "ROE significantly exceeds ROIC (ratio > 3x), indicating shareholder "
                        "returns come from leverage, buybacks, or capital structure rather "
                        "than underlying business quality."
                    ),
                }
            )
            logger.info(
                "capital_flag_engineered_returns",
                ticker=ticker,
                roe_roic_ratio=roe_roic_ratio,
            )

        # --- FLAG 3: Suspect Leverage ---
        elif leverage_quality == "SUSPECT":
            ratio_str = f"ROE/ROIC: {roe_roic_ratio:.1f}x" if roe_roic_ratio else ""
            flags.append(
                {
                    "type": "CAPITAL_SUSPECT_RETURNS",
                    "severity": "MEDIUM",
                    "detail": f"Moderate leverage amplification detected. {ratio_str}",
                    "action": "RISK_ADJUST",
                    "risk_penalty": 0.5,
                    "rationale": (
                        "ROE moderately exceeds ROIC (ratio 2-3x). Returns partially "
                        "driven by leverage rather than operational excellence."
                    ),
                }
            )
            logger.info(
                "capital_flag_suspect_returns",
                ticker=ticker,
                roe_roic_ratio=roe_roic_ratio,
            )

        # --- FLAG 4: Below Hurdle ROIC ---
        if roic_quality == "WEAK":
            roic_str = f"ROIC: {roic:.1%}" if roic is not None else ""
            flags.append(
                {
                    "type": "CAPITAL_BELOW_HURDLE",
                    "severity": "MEDIUM",
                    "detail": f"Returns below cost of capital proxy. {roic_str}",
                    "action": "RISK_ADJUST",
                    "risk_penalty": 0.5,
                    "rationale": (
                        "ROIC below 8% hurdle rate suggests the company may be destroying "
                        "value on a risk-adjusted basis. Acceptable only with clear "
                        "turnaround thesis and improving trajectory."
                    ),
                }
            )
            logger.info(
                "capital_flag_below_hurdle",
                ticker=ticker,
                roic=roic,
            )

        # --- BONUS: Capital Efficient (genuine high returns) ---
        if roic_quality == "STRONG" and leverage_quality in ("GENUINE", "CONSERVATIVE"):
            roic_str = f"ROIC: {roic:.1%}" if roic is not None else ""
            flags.append(
                {
                    "type": "CAPITAL_EFFICIENT",
                    "severity": "POSITIVE",
                    "detail": f"Strong genuine capital efficiency. {roic_str}",
                    "action": "RISK_BONUS",
                    "risk_penalty": -0.5,
                    "rationale": (
                        "High ROIC (>15%) with ROE/ROIC ratio below 2x indicates returns "
                        "driven by operational excellence rather than financial leverage. "
                        "Suggests sustainable competitive advantage."
                    ),
                }
            )
            logger.info(
                "capital_flag_efficient",
                ticker=ticker,
                roic=roic,
                leverage_quality=leverage_quality,
            )

        return flags
