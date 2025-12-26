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

import re
import json
import structlog
from typing import Any
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
    Deterministic pre-screening for catastrophic financial and legal risks.

    CRITICAL RED FLAGS (AUTO_REJECT - financial viability):
    1. Extreme Leverage: D/E > 500% (bankruptcy risk)
    2. Earnings Quality: Positive income but negative FCF >2x (fraud indicator)
    3. Refinancing Risk: Interest coverage <2.0x AND D/E >100% (default risk)

    WARNING FLAGS (RISK_PENALTY - tax/legal, not viability):
    4. PFIC Probable: Company likely classified as PFIC (US tax reporting burden)
    5. VIE Structure: China stock uses contractual VIE structure (ownership risk)
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
    def extract_metrics(fundamentals_report: str) -> dict[str, float | None]:
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
        metrics: dict[str, float | None] = {
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
            r'\*\*Free Cash Flow\*\*:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?',
            r'(?:^|\n)\s*Free Cash Flow:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?',
            r'(?:^|\n)\s*FCF:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?',
            r'(?:Free Cash Flow|FCF):\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?',
            r'Positive FCF:\s*[$¥€£]?\s*([0-9,.]+)\s*([BMK])?',  # No negative for "Positive"
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
            r'\*\*Net Income\*\*:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?',
            r'(?:^|\n)\s*Net Income:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?',
            r'Net Income:\s*([+-]?)[$¥€£]?\s*([0-9,.]+)\s*([BMK])?',
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
        metrics: dict[str, float | None],
        ticker: str = "UNKNOWN",
        sector: Sector = Sector.GENERAL
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
            - country: Country of domicile
            - sector: Sector name
        """
        risks: dict[str, Any] = {
            'pfic_status': None,
            'pfic_evidence': None,
            'vie_structure': None,
            'vie_evidence': None,
            'country': None,
            'sector': None,
        }

        if not legal_report:
            return risks

        # Try to parse as JSON first (preferred)
        try:
            # Handle potential markdown code blocks
            json_str = legal_report.strip()
            if json_str.startswith('```'):
                # Extract JSON from markdown code block
                lines = json_str.split('\n')
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith('```') and not in_block:
                        in_block = True
                        continue
                    elif line.startswith('```') and in_block:
                        break
                    elif in_block:
                        json_lines.append(line)
                json_str = '\n'.join(json_lines)

            data = json.loads(json_str)
            risks['pfic_status'] = data.get('pfic_status')
            risks['pfic_evidence'] = data.get('pfic_evidence')
            risks['vie_structure'] = data.get('vie_structure')
            risks['vie_evidence'] = data.get('vie_evidence')
            risks['country'] = data.get('country')
            risks['sector'] = data.get('sector')

            logger.debug(
                "legal_risks_parsed_json",
                pfic_status=risks['pfic_status'],
                vie_structure=risks['vie_structure']
            )
            return risks

        except json.JSONDecodeError:
            # Fall back to regex parsing
            logger.debug("legal_report_not_json_falling_back_to_regex")

        # Regex fallback for non-JSON output
        pfic_match = re.search(
            r'"?pfic_status"?\s*:\s*"?(CLEAN|UNCERTAIN|PROBABLE|N/A)"?',
            legal_report, re.IGNORECASE
        )
        if pfic_match:
            risks['pfic_status'] = pfic_match.group(1).upper()

        vie_match = re.search(
            r'"?vie_structure"?\s*:\s*"?(YES|NO|N/A)"?',
            legal_report, re.IGNORECASE
        )
        if vie_match:
            risks['vie_structure'] = vie_match.group(1).upper()

        return risks

    @staticmethod
    def detect_legal_flags(
        legal_risks: dict[str, Any],
        ticker: str = "UNKNOWN"
    ) -> list[dict]:
        """
        Detect legal/tax warning flags from Legal Counsel output.

        Unlike financial red flags (which trigger AUTO_REJECT), legal flags
        add risk penalties but do NOT reject the stock. PFIC is a tax burden,
        not a viability issue.

        Args:
            legal_risks: Extracted legal risk data from extract_legal_risks()
            ticker: Ticker symbol for logging

        Returns:
            List of warning flag dicts with risk_penalty values
        """
        warnings = []

        pfic_status = legal_risks.get('pfic_status')
        vie_structure = legal_risks.get('vie_structure')
        pfic_evidence = legal_risks.get('pfic_evidence') or 'No evidence provided'

        # --- WARNING 1: PFIC Probable ---
        if pfic_status == 'PROBABLE':
            warnings.append({
                'type': 'PFIC_PROBABLE',
                'severity': 'WARNING',
                'detail': f"Company likely classified as PFIC. Evidence: {pfic_evidence[:100]}...",
                'action': 'RISK_PENALTY',
                'risk_penalty': 1.0,  # Add 1.0 to risk tally
                'rationale': 'PFIC classification requires onerous US tax reporting (Form 8621). '
                            'Mark-to-market or QEF election required. Not a viability issue, '
                            'but increases compliance burden for US investors.'
            })
            logger.warning(
                "legal_flag_pfic_probable",
                ticker=ticker,
                evidence=pfic_evidence[:50]
            )

        # --- WARNING 2: PFIC Uncertain (lesser penalty) ---
        elif pfic_status == 'UNCERTAIN':
            warnings.append({
                'type': 'PFIC_UNCERTAIN',
                'severity': 'WARNING',
                'detail': f"PFIC status unclear. Evidence: {pfic_evidence[:100]}...",
                'action': 'RISK_PENALTY',
                'risk_penalty': 0.5,  # Add 0.5 to risk tally
                'rationale': 'PFIC status cannot be confirmed. Company may use hedge language '
                            'or is in a high-risk sector without clear disclosure. '
                            'Recommend consulting tax advisor before investing.'
            })
            logger.info(
                "legal_flag_pfic_uncertain",
                ticker=ticker,
                evidence=pfic_evidence[:50]
            )

        # --- WARNING 3: VIE Structure ---
        if vie_structure == 'YES':
            vie_evidence = legal_risks.get('vie_evidence') or 'VIE structure detected'
            warnings.append({
                'type': 'VIE_STRUCTURE',
                'severity': 'WARNING',
                'detail': f"Company uses VIE contractual structure for China operations. {vie_evidence[:80]}",
                'action': 'RISK_PENALTY',
                'risk_penalty': 0.5,  # Add 0.5 to risk tally
                'rationale': 'VIE structure means investors own contracts, not equity. '
                            'China regulatory risk if VIE agreements are invalidated. '
                            'Common for China tech/education stocks but adds legal uncertainty.'
            })
            logger.warning(
                "legal_flag_vie_structure",
                ticker=ticker,
                evidence=vie_evidence[:50]
            )

        return warnings
