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

logger = structlog.get_logger(__name__)


class RedFlagDetector:
    """
    Deterministic pre-screening for catastrophic financial risks.

    Detects three critical red flags from institutional bankruptcy/distress research:
    1. Extreme Leverage: D/E > 500% (bankruptcy risk)
    2. Earnings Quality: Positive income but negative FCF >2x (fraud indicator)
    3. Refinancing Risk: Interest coverage <2.0x AND D/E >100% (default risk)
    """

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
        ticker: str = "UNKNOWN"
    ) -> Tuple[List[Dict], str]:
        """
        Apply threshold-based red-flag detection logic.

        Args:
            metrics: Extracted financial metrics
            ticker: Ticker symbol for logging

        Returns:
            Tuple of (red_flags_list, "PASS" or "REJECT")

        Red-flag criteria:
        1. D/E > 500%: Extreme leverage (bankruptcy risk)
        2. Positive income but negative FCF >2x income: Earnings quality (fraud)
        3. Interest coverage <2.0x AND D/E >100%: Refinancing risk (default)
        """
        red_flags = []

        # --- RED FLAG 1: Extreme Leverage (Leverage Bomb) ---
        debt_to_equity = metrics.get('debt_to_equity')
        if debt_to_equity is not None and debt_to_equity > 500:
            red_flags.append({
                'type': 'EXTREME_LEVERAGE',
                'severity': 'CRITICAL',
                'detail': f"D/E ratio {debt_to_equity:.1f}% is extreme (>500% threshold)",
                'action': 'AUTO_REJECT',
                'rationale': 'Company has 5x+ more debt than equity - bankruptcy risk'
            })
            logger.warning(
                "red_flag_extreme_leverage",
                ticker=ticker,
                debt_to_equity=debt_to_equity,
                threshold=500
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

        # --- RED FLAG 3: Interest Coverage Death Spiral ---
        interest_coverage = metrics.get('interest_coverage')

        if (interest_coverage is not None and interest_coverage < 2.0 and
            debt_to_equity is not None and debt_to_equity > 100):

            red_flags.append({
                'type': 'REFINANCING_RISK',
                'severity': 'CRITICAL',
                'detail': f"Interest coverage {interest_coverage:.2f}x with {debt_to_equity:.1f}% D/E ratio",
                'action': 'AUTO_REJECT',
                'rationale': 'Cannot comfortably service debt - refinancing/default risk'
            })
            logger.warning(
                "red_flag_refinancing_risk",
                ticker=ticker,
                interest_coverage=interest_coverage,
                debt_to_equity=debt_to_equity
            )

        # Determine result
        has_auto_reject = any(flag['action'] == 'AUTO_REJECT' for flag in red_flags)
        result = 'REJECT' if has_auto_reject else 'PASS'

        return red_flags, result
