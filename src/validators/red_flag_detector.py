"""
Red-Flag Detector for Catastrophic Financial Risk Pre-Screening

This module implements deterministic threshold-based validation to catch extreme
financial risks before they enter the bull/bear debate phase. Uses regex parsing
of the Fundamentals Analyst's DATA_BLOCK output.

Why code-driven instead of LLM-driven:
- Exact thresholds required (D/E > 500%, not "very high")
- Fast-fail pattern (avoid LLM calls for doomed stocks)
- Reliability (no hallucination risk on number parsing)
- Cost savings (~60% token reduction for rejected stocks)

Pattern matches: src/data/validator.py (also code-driven for same reasons)
"""

import re
import structlog
from typing import Dict, Optional, List, Tuple
from enum import Enum

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
    Deterministic pre-screening for catastrophic financial risks.

    Detects three critical red flags from institutional bankruptcy/distress research:
    1. Extreme Leverage: D/E > 500% (bankruptcy risk)
    2. Earnings Quality: Positive income but negative FCF >2x (fraud indicator)
    3. Refinancing Risk: Interest coverage <2.0x AND D/E >100% (default risk)
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
        sector_match = re.search(r'SECTOR:\s*(.+?)(?:\n|$)', fundamentals_report)

        if not sector_match:
            logger.debug("no_sector_found_in_report", fallback="GENERAL")
            return Sector.GENERAL

        sector_text = sector_match.group(1).strip()

        # Map to enum
        if "Banking" in sector_text or "Bank" in sector_text:
            return Sector.BANKING
        elif "Utilities" in sector_text or "Utility" in sector_text:
            return Sector.UTILITIES
        elif "Shipping" in sector_text or "Commodities" in sector_text or "Cyclical" in sector_text:
            return Sector.SHIPPING
        elif "Technology" in sector_text or "Software" in sector_text:
            return Sector.TECHNOLOGY
        else:
            return Sector.GENERAL

    @staticmethod
    def extract_metrics(fundamentals_report: str) -> Dict[str, Optional[float]]:
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
        metrics: Dict[str, Optional[float]] = {
            'debt_to_equity': None,
            'net_income': None,
            'fcf': None,
            'interest_coverage': None,
            'pe_ratio': None,
            'adjusted_health_score': None,
        }

        if not fundamentals_report:
            return metrics

        # Extract the LAST DATA_BLOCK (agent self-correction pattern)
        data_block_pattern = r'### --- START DATA_BLOCK ---(.+?)### --- END DATA_BLOCK ---'
        blocks = list(re.finditer(data_block_pattern, fundamentals_report, re.DOTALL))

        if not blocks:
            logger.warning("no_data_block_found_in_fundamentals_report")
            return metrics

        # Use the last (most corrected) block
        data_block = blocks[-1].group(1)

        # Extract ADJUSTED_HEALTH_SCORE (percentage)
        health_match = re.search(r'ADJUSTED_HEALTH_SCORE:\s*(\d+(?:\.\d+)?)%', data_block)
        if health_match:
            metrics['adjusted_health_score'] = float(health_match.group(1))

        # Extract PE_RATIO_TTM
        pe_match = re.search(r'PE_RATIO_TTM:\s*([0-9.]+)', data_block)
        if pe_match:
            metrics['pe_ratio'] = float(pe_match.group(1))

        # Now extract from detailed sections (below DATA_BLOCK)
        metrics['debt_to_equity'] = RedFlagDetector._extract_debt_to_equity(fundamentals_report)
        metrics['interest_coverage'] = RedFlagDetector._extract_interest_coverage(fundamentals_report)
        metrics['fcf'] = RedFlagDetector._extract_free_cash_flow(fundamentals_report)
        metrics['net_income'] = RedFlagDetector._extract_net_income(fundamentals_report)

        return metrics

    @staticmethod
    def _extract_debt_to_equity(report: str) -> Optional[float]:
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
            r'(?:^|\n)\s*-?\s*D/E:\s*([0-9.]+)',
            r'(?:^|\n)\s*-?\s*Debt/Equity:\s*([0-9.]+)',
            r'(?:^|\n)\s*-?\s*Debt-to-Equity:\s*([0-9.]+)',
            r'D/E:\s*([0-9.]+)',
            r'Debt/Equity:\s*([0-9.]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
            if match:
                value = float(match.group(1))
                # Convert to percentage if < 10 (assume ratio like 2.5 -> 250%)
                return value if value >= 10 else value * 100
        return None

    @staticmethod
    def _extract_interest_coverage(report: str) -> Optional[float]:
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
            r'\*\*Interest Coverage\*\*:\s*([0-9.]+)x?',
            r'Interest Coverage:\s*([0-9.]+)x?',
            r'Interest Coverage Ratio:\s*([0-9.]+)x?',
        ]
        for pattern in patterns:
            match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
            if match:
                return float(match.group(1))
        return None

    @staticmethod
    def _extract_free_cash_flow(report: str) -> Optional[float]:
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
            r'\*\*Free Cash Flow\*\*:\s*([+-]?)\$?\s*([0-9,.]+)\s*([BMK])?',
            r'(?:^|\n)\s*Free Cash Flow:\s*([+-]?)\$?\s*([0-9,.]+)\s*([BMK])?',
            r'(?:^|\n)\s*FCF:\s*([+-]?)\$?\s*([0-9,.]+)\s*([BMK])?',
            r'(?:Free Cash Flow|FCF):\s*([+-]?)\$?\s*([0-9,.]+)\s*([BMK])?',
            r'Positive FCF:\s*\$?\s*([0-9,.]+)\s*([BMK])?',  # No negative for "Positive"
        ]
        for pattern in patterns:
            match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
            if match:
                groups = match.groups()
                # Handle two different pattern structures
                if len(groups) == 2:  # "Positive FCF" pattern
                    value = float(groups[0].replace(',', ''))
                    multiplier = groups[1] if len(groups) > 1 else None
                else:  # All other patterns with sign capture
                    sign = groups[0]  # '+' or '-' or ''
                    value = float(groups[1].replace(',', ''))
                    if sign == '-':
                        value = -value
                    multiplier = groups[2] if len(groups) > 2 else None

                # Handle B/M/K multipliers
                if multiplier:
                    if multiplier.upper() == 'B':
                        value *= 1_000_000_000
                    elif multiplier.upper() == 'M':
                        value *= 1_000_000
                    elif multiplier.upper() == 'K':
                        value *= 1_000
                return value
        return None

    @staticmethod
    def _extract_net_income(report: str) -> Optional[float]:
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
            r'\*\*Net Income\*\*:\s*([+-]?)\$?\s*([0-9,.]+)\s*([BMK])?',
            r'(?:^|\n)\s*Net Income:\s*([+-]?)\$?\s*([0-9,.]+)\s*([BMK])?',
            r'Net Income:\s*([+-]?)\$?\s*([0-9,.]+)\s*([BMK])?',
        ]
        for pattern in patterns:
            match = re.search(pattern, report, re.IGNORECASE | re.MULTILINE)
            if match:
                groups = match.groups()
                sign = groups[0]  # '+' or '-' or ''
                value = float(groups[1].replace(',', ''))
                if sign == '-':
                    value = -value
                multiplier = groups[2] if len(groups) > 2 else None

                # Handle B/M/K multipliers
                if multiplier:
                    if multiplier.upper() == 'B':
                        value *= 1_000_000_000
                    elif multiplier.upper() == 'M':
                        value *= 1_000_000
                    elif multiplier.upper() == 'K':
                        value *= 1_000
                return value
        return None

    @staticmethod
    def detect_red_flags(
        metrics: Dict[str, Optional[float]],
        ticker: str = "UNKNOWN",
        sector: Sector = Sector.GENERAL
    ) -> Tuple[List[Dict], str]:
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
            coverage_de_threshold = 200  # D/E > 200% when coverage weak (vs 100% standard)
        else:
            # General/Technology: Standard thresholds
            leverage_threshold = 500
            coverage_threshold = 2.0
            coverage_de_threshold = 100

        # --- RED FLAG 1: Extreme Leverage (Leverage Bomb) ---
        debt_to_equity = metrics.get('debt_to_equity')
        if leverage_threshold is not None and debt_to_equity is not None and debt_to_equity > leverage_threshold:
            red_flags.append({
                'type': 'EXTREME_LEVERAGE',
                'severity': 'CRITICAL',
                'detail': f"D/E ratio {debt_to_equity:.1f}% is extreme (>{leverage_threshold}% threshold for {sector.value})",
                'action': 'AUTO_REJECT',
                'rationale': f'Leverage exceeds sector-appropriate threshold - bankruptcy risk (sector: {sector.value})'
            })
            logger.warning(
                "red_flag_extreme_leverage",
                ticker=ticker,
                debt_to_equity=debt_to_equity,
                threshold=leverage_threshold,
                sector=sector.value
            )

        # --- RED FLAG 2: Earnings Quality Disconnect ---
        net_income = metrics.get('net_income')
        fcf = metrics.get('fcf')

        if (net_income is not None and net_income > 0 and
            fcf is not None and fcf < 0 and
            abs(fcf) > (2 * net_income)):

            red_flags.append({
                'type': 'EARNINGS_QUALITY',
                'severity': 'CRITICAL',
                'detail': f"Positive net income (${net_income:,.0f}) but negative FCF (${fcf:,.0f}) >2x income",
                'action': 'AUTO_REJECT',
                'rationale': 'Earnings likely fabricated through accounting tricks - FCF disconnect'
            })
            logger.warning(
                "red_flag_earnings_quality",
                ticker=ticker,
                net_income=net_income,
                fcf=fcf,
                disconnect_multiple=abs(fcf / net_income) if net_income != 0 else None
            )

        # --- RED FLAG 3: Interest Coverage Death Spiral (Sector-Aware) ---
        interest_coverage = metrics.get('interest_coverage')

        # Only apply if sector has defined thresholds (excludes banking)
        if (coverage_threshold is not None and coverage_de_threshold is not None and
            interest_coverage is not None and interest_coverage < coverage_threshold and
            debt_to_equity is not None and debt_to_equity > coverage_de_threshold):

            red_flags.append({
                'type': 'REFINANCING_RISK',
                'severity': 'CRITICAL',
                'detail': f"Interest coverage {interest_coverage:.2f}x with {debt_to_equity:.1f}% D/E ratio (thresholds: <{coverage_threshold}x coverage + >{coverage_de_threshold}% D/E for {sector.value})",
                'action': 'AUTO_REJECT',
                'rationale': f'Cannot comfortably service debt - refinancing/default risk (sector: {sector.value})'
            })
            logger.warning(
                "red_flag_refinancing_risk",
                ticker=ticker,
                interest_coverage=interest_coverage,
                debt_to_equity=debt_to_equity,
                coverage_threshold=coverage_threshold,
                de_threshold=coverage_de_threshold,
                sector=sector.value
            )

        # Determine result
        has_auto_reject = any(flag['action'] == 'AUTO_REJECT' for flag in red_flags)
        result = 'REJECT' if has_auto_reject else 'PASS'

        return red_flags, result
