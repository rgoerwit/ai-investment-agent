"""
Quiet Mode Report Generator for Multi-Agent Investment Analysis System
FIXED: Handles LangGraph list outputs to prevent 'list object has no attribute startswith' errors.
FIXED: Added deduplication to prevent stuttering output in final reports.
FIXED: Case-insensitive regex matching for decision extraction.
UPDATED: Added brief_mode flag for condensed output.
UPDATED: Added comprehensive error handling and fallback logic for missing Portfolio Manager output.
UPDATED: Added Football Field chart generation integration.
"""

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.charts.extractors.data_block import ChartRawData
from src.charts.extractors.pm_block import extract_pm_block
from src.charts.extractors.valuation import format_iv
from src.data_block_utils import (
    extract_data_block_field,
    fenced_marker_fragment,
    normalize_structured_block_boundaries,
    unfenced_label,
)
from src.error_safety import summarize_exception
from src.pm_decision_parser import (
    PM_VERDICT_ALTERNATION,
    PM_VERDICT_HEADER_RE,
    canonicalize_pm_verdict,
)
from src.reporting.state_access import get_effective_red_flags
from src.runtime_config import get_runtime_config
from src.runtime_diagnostics import (
    build_analysis_validity,
    is_publishable_analysis,
)
from src.thesis_constants import ANALYST_COVERAGE_MAX
from src.ticker_policy import CHINA_SUFFIXES, KOREA_SUFFIXES, ticker_in_group

logger = structlog.get_logger(__name__)

# Local import for utility function to avoid circular dependency at module level
# We import inside the method where it is needed

_GOVERNANCE_TERM_FIXES = [
    # "controlled subsidiary" → "controlled company" for listed equities with a
    # controlling shareholder (HKEX/SEHK terminology; "subsidiary" implies full
    # ownership and different legal standing from a listed controlled company).
    (
        re.compile(r"\bcontrolled subsidiary\b", re.IGNORECASE),
        "controlled company",
    ),
    (
        re.compile(r"\bparent subsidiary relationship\b", re.IGNORECASE),
        "controlling shareholder relationship",
    ),
]

_RETAIL_POSITION_TERM_FIXES = [
    (re.compile(r"\bstop[- ]loss\b", re.IGNORECASE), "downside review level"),
    (re.compile(r"(?m)^\s*STOP\s*:", re.IGNORECASE), "DOWNSIDE REVIEW LEVEL:"),
    (re.compile(r"\bprofit targets?\b", re.IGNORECASE), "valuation references"),
    (re.compile(r"(?m)^\s*TARGET_1\s*:", re.IGNORECASE), "BASE-CASE REFERENCE:"),
    (re.compile(r"(?m)^\s*TARGET_2\s*:", re.IGNORECASE), "STRETCH REFERENCE:"),
    (re.compile(r"(?m)^\s*ENTRY\s*:", re.IGNORECASE), "FAIR ENTRY CONTEXT:"),
]


def normalize_governance_terms(text: str) -> str:
    """Deterministic correction of common governance terminology errors in LLM output."""
    for pattern, replacement in _GOVERNANCE_TERM_FIXES:
        text = pattern.sub(replacement, text)
    return text


def normalize_retail_position_terms(text: str) -> str:
    """Translate legacy trading labels into non-executable review context."""
    for pattern, replacement in _RETAIL_POSITION_TERM_FIXES:
        text = pattern.sub(replacement, text)
    return text


_FENCED_PM_BLOCK_PATTERN = re.compile(
    r"(?:#{2,}\s+PM_BLOCK[^\n]*\n(?:[ \t]*\n)*)?```[^\n]*\n(?P<body>.*?)\n?```",
    re.DOTALL,
)


def _strip_fenced_pm_machine_block(text: str) -> str:
    """Remove a fenced PM_BLOCK code block (and any heading) only when the fence
    actually encloses both START and END machine markers.

    The PM sometimes emits CONSULTANT_RESOLUTION / APAC_RESOLUTION prose *inside*
    the same fence, before the START marker. Gating removal on the enclosed markers
    strips that prose with the block instead of orphaning it as a stray
    "PM_BLOCK" section (the 6831.HK regression). Fences without both markers
    (e.g. DECISION LOGIC) are left untouched.
    """

    def _repl(match: re.Match[str]) -> str:
        body = match.group("body")
        if "START PM_BLOCK" in body and "END PM_BLOCK" in body:
            return ""
        return match.group(0)

    return _FENCED_PM_BLOCK_PATTERN.sub(_repl, text)


# Display-form PM verdicts whose position levels are not applicable. The PM's
# position parameters are authoritative for these; subordinate reference
# levels and the position plan are suppressed. Kept as one constant so the display
# form here never drifts from the underscore form used elsewhere (e.g.
# retrospective.py uses "DO_NOT_INITIATE").
_NON_EXECUTABLE_VERDICTS = ("HOLD", "DO NOT INITIATE", "SELL")


def _calculate_regulatory_score(result: dict[str, Any], raw: ChartRawData) -> float:
    """Score regulatory risk from the canonical flag ledger, then DATA_BLOCK.

    Legal Counsel flags are the authoritative record when they identify a risk.
    The Senior Fundamentals DATA_BLOCK remains a compatibility fallback when no
    corresponding flag was emitted.
    """
    regulatory = 100.0
    effective_flag_types = {
        str(flag.get("type", "")).upper() for flag in get_effective_red_flags(result)
    }

    legal_pfic_status = None
    if "PFIC_PROBABLE" in effective_flag_types:
        legal_pfic_status = "PROBABLE"
    elif "PFIC_UNCERTAIN" in effective_flag_types:
        legal_pfic_status = "UNCERTAIN"

    from src.agents.decision_nodes import resolve_pfic_display_status

    canonical_pfic, _pfic_note = resolve_pfic_display_status(
        legal_pfic_status, raw.pfic_risk
    )
    if canonical_pfic:
        risk_upper = canonical_pfic.upper()
        if "HIGH" in risk_upper:
            regulatory -= 40
        elif "MEDIUM" in risk_upper or "UNCERTAIN" in risk_upper:
            regulatory -= 20

    if "VIE_STRUCTURE" in effective_flag_types or raw.vie_structure is True:
        regulatory -= 25
    if "CMIC_FLAGGED" in effective_flag_types or raw.cmic_flagged is True:
        regulatory -= 35

    if raw.adr_impact and "MODERATE_CONCERN" in raw.adr_impact.upper():
        regulatory -= 10

    return max(0.0, min(100.0, regulatory))


_ENTRY_EXIT_SUBSECTION_PATTERN = re.compile(
    # Matches both the legacy header and the v4.11 retail-framing rename.
    r"(#{2,6}\s*(?:ENTRY/EXIT RECOMMENDATIONS|TECHNICAL REFERENCE LEVELS)[^\n]*\n)"
    r".*?(?=\n#{2,6}\s|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _suppress_executable_levels(market_md: str) -> str:
    """Neutralize technical reference levels for non-actionable verdicts.

    For DO_NOT_INITIATE/SELL, displaying subordinate price levels adds noise and
    can resemble conflicting instructions. Replace the subsection body with a
    non-actionable note.
    """
    return _ENTRY_EXIT_SUBSECTION_PATTERN.sub(
        r"\1*Not actionable — Portfolio Manager verdict is non-executable.*\n",
        market_md,
    )


def _markdown_asset_link(asset_path: Path, report_dir: Path | None) -> str:
    """Return a portable markdown link from a report to an asset path."""
    if not report_dir:
        return str(asset_path)

    try:
        return str(asset_path.resolve().relative_to(report_dir.resolve()))
    except ValueError:
        return os.path.relpath(asset_path.resolve(), report_dir.resolve())


class QuietModeReporter:
    """Generates clean markdown reports with minimal output."""

    def __init__(
        self,
        ticker: str,
        company_name: str | None = None,
        quick_mode: bool = False,
        chart_format: str = "png",
        transparent_charts: bool = False,
        skip_charts: bool = False,
        image_dir: Path | None = None,
        report_dir: Path | None = None,
        report_stem: str | None = None,
    ):
        self.ticker = ticker.upper()
        self.company_name = company_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.trade_date = datetime.now().strftime("%Y-%m-%d")
        self.report_stem = report_stem  # Base name for image files
        self.quick_mode = quick_mode
        self.chart_format = chart_format
        self.transparent_charts = transparent_charts
        self.skip_charts = skip_charts
        self.image_dir = image_dir  # Custom image directory (relative path)
        self.report_dir = report_dir  # Directory where report is being written
        self.valuation_context: str | None = None  # Stored for article writer

    def _store_valuation_context(
        self,
        current_price: float | None,
        target_low: float | None,
        target_high: float | None,
        methodology: str | None,
        confidence: str | None,
        unanchored_caveat: str | None = None,
    ) -> None:
        """Store valuation context for article writer to use.

        This enables the article writer to address discrepancies between
        the football field chart visuals and the investment decision.

        When ``unanchored_caveat`` is set the scenario valuation was suppressed
        (peak/distorted earnings, no normalized EPS baseline), so no point fair
        value is handed downstream — the writer is told the anchor is unavailable
        rather than a confident midpoint it could launder into a price target.
        """
        if unanchored_caveat:
            price_line = (
                f"- Current Price: {format_iv(current_price)}\n"
                if current_price
                else ""
            )
            self.valuation_context = (
                "VALUATION DATA (scenario target suppressed):\n"
                f"- Fair Value: UNANCHORED — {unanchored_caveat}\n"
                f"{price_line}"
                "NOTE: Do not present a precise point fair value as normalized. "
                "Discuss valuation as a range or as explicitly unanchored, and do "
                "not describe the chart as showing a target range (it was suppressed)."
            )
            return

        if not current_price or not target_low or not target_high:
            self.valuation_context = (
                "VALUATION DATA: Insufficient data for target calculation."
            )
            return

        # Calculate fair value (midpoint) and position in range
        fair_value = (target_low + target_high) / 2
        range_size = target_high - target_low
        position_in_range = (
            ((current_price - target_low) / range_size * 100) if range_size > 0 else 50
        )

        # Determine price position description
        if current_price > target_high:
            position_desc = "ABOVE target range (overvalued by chart methodology)"
        elif current_price > fair_value:
            position_desc = (
                f"ABOVE fair value midpoint ({position_in_range:.0f}% of range)"
            )
        elif current_price < target_low:
            position_desc = "BELOW target range (undervalued by chart methodology)"
        else:
            position_desc = (
                f"BELOW fair value midpoint ({position_in_range:.0f}% of range)"
            )

        self.valuation_context = f"""VALUATION DATA (from Football Field Chart):
- Methodology: {methodology or "P/E Normalization"}
- Target Range: ${target_low:.2f} - ${target_high:.2f}
- Fair Value (midpoint): ${fair_value:.2f}
- Current Price: ${current_price:.2f}
- Price Position: {position_desc}
- Confidence: {confidence or "N/A"}

NOTE: If price is above fair value midpoint but verdict is BUY, you MUST explain why in the Valuation section."""

    def get_valuation_context(self) -> str:
        """Return stored valuation context for article writer."""
        return (
            self.valuation_context
            or "VALUATION DATA: Not available (charts skipped or insufficient data)."
        )

    def _generate_chart(self, result: dict) -> Path | None:
        """Generate football field chart from analysis results.

        Args:
            result: Dictionary containing analysis results with fundamentals_report
                   and investment_plan fields

        Returns:
            Path to generated chart image, or None if chart generation failed/skipped
        """
        # Skip if charts disabled or quick mode
        if self.skip_charts or self.quick_mode:
            logger.debug(
                "Chart generation skipped",
                skip_charts=self.skip_charts,
                quick_mode=self.quick_mode,
            )
            return None

        try:
            from src.charts.base import ChartConfig, ChartFormat, FootballFieldData
            from src.charts.extractors.data_block import (
                extract_chart_data_from_data_block,
            )
            from src.charts.extractors.valuation import (
                calculate_valuation_targets,
                extract_valuation_scenarios_for_fundamentals,
                scenario_valuation_caveat,
            )
            from src.charts.generators.football_field import generate_football_field
            from src.config import config

            # Extract raw facts from DATA_BLOCK in fundamentals report
            fundamentals_report = self._normalize_string(
                result.get("fundamentals_report", "")
            )
            chart_data = extract_chart_data_from_data_block(fundamentals_report)

            # Extract valuation targets from Valuation Calculator output
            # Uses VALUATION_PARAMS block, Python calculates actual targets
            valuation_params = self._normalize_string(
                result.get("valuation_params", "")
            )
            targets = calculate_valuation_targets(valuation_params)

            # Parse VALUATION_SCENARIOS for chart overlay. Returns None on any
            # data-sufficiency or sanity check failure → chart falls back to
            # the legacy single-range bars.
            scenarios = None
            if valuation_params and fundamentals_report:
                try:
                    scenarios = extract_valuation_scenarios_for_fundamentals(
                        valuation_params, fundamentals_report
                    )
                except Exception as exc:  # pragma: no cover — defense-in-depth
                    from src.error_safety import summarize_exception

                    logger.warning(
                        "report_chart_scenario_extraction_failed",
                        ticker=self.ticker,
                        **summarize_exception(
                            exc, operation="report_chart_scenario_extraction"
                        ),
                    )
                    scenarios = None

            # Check if we have minimum data
            if not chart_data.current_price or not chart_data.fifty_two_week_high:
                logger.debug(
                    "Insufficient data for chart generation",
                    ticker=self.ticker,
                    current_price=chart_data.current_price,
                    fifty_two_week_high=chart_data.fifty_two_week_high,
                )
                return None

            # Combine into FootballFieldData
            quality_warnings = []
            red_flags = get_effective_red_flags(result)

            for flag in red_flags:
                # Only show CRITICAL or WARNING severity on the chart to reduce noise
                severity = str(flag.get("severity", "")).upper()
                if severity in ["CRITICAL", "WARNING"]:
                    detail = str(flag.get("detail", ""))
                    # Truncate to keep chart clean
                    quality_warnings.append(
                        detail[:50] + "..." if len(detail) > 50 else detail
                    )

            # Limit to 2 warnings to prevent visual occlusion
            quality_warnings = quality_warnings[:2]
            footnote_parts = []
            if targets.methodology:
                footnote_parts.append("Targets based on P/E normalization")
            scenario_caveat = scenario_valuation_caveat(scenarios)
            if scenarios and scenarios.normalization_required:
                footnote_parts.append(f"Scenario EPS: {scenarios.earnings_basis}")
            if scenario_caveat:
                footnote_parts.append(scenario_caveat)

            football_data = FootballFieldData(
                ticker=self.ticker,
                trade_date=self.trade_date,
                current_price=chart_data.current_price,
                fifty_two_week_high=chart_data.fifty_two_week_high,
                fifty_two_week_low=chart_data.fifty_two_week_low or 0.0,
                moving_avg_50=chart_data.moving_avg_50,
                moving_avg_200=chart_data.moving_avg_200,
                external_target_high=chart_data.external_target_high,
                external_target_low=chart_data.external_target_low,
                external_target_mean=chart_data.external_target_mean,
                our_target_low=None if scenario_caveat else targets.low,
                our_target_high=None if scenario_caveat else targets.high,
                target_methodology=targets.methodology,
                target_confidence=targets.confidence,
                quality_warnings=quality_warnings if quality_warnings else None,
                footnote=" | ".join(footnote_parts) if footnote_parts else None,
                scenarios=None if scenario_caveat else scenarios,
            )

            # Store valuation context for article writer (D1 implementation).
            # When the scenario caveat fired the chart suppressed its target
            # range above; mirror that here so the writer is not handed a
            # confident midpoint the chart deliberately withheld.
            self._store_valuation_context(
                current_price=chart_data.current_price,
                target_low=targets.low,
                target_high=targets.high,
                methodology=targets.methodology,
                confidence=targets.confidence,
                unanchored_caveat=scenario_caveat,
            )

            # Configure chart generation
            # Use custom image_dir if provided, otherwise fall back to config default
            output_dir = (
                self.image_dir
                if self.image_dir
                else get_runtime_config(config).images_dir
            )
            chart_config = ChartConfig(
                output_dir=output_dir,
                format=ChartFormat.SVG
                if self.chart_format == "svg"
                else ChartFormat.PNG,
                transparent=self.transparent_charts,
                filename_stem=self.report_stem,
            )

            # Generate chart
            chart_path = generate_football_field(football_data, chart_config)

            if chart_path:
                logger.info(
                    "Chart generated successfully",
                    ticker=self.ticker,
                    path=str(chart_path),
                )

            return chart_path

        except ImportError as e:
            logger.warning(
                "chart_deps_unavailable",
                **summarize_exception(e, operation="chart_deps_unavailable"),
            )
            return None
        except Exception as e:
            logger.warning(
                "chart_generation_failed",
                **summarize_exception(e, operation="chart_generation_failed"),
            )
            return None

    def _generate_radar_chart(self, result: dict) -> Path | None:
        """Generate thesis alignment radar chart with 6 axes.

        Axes:
        - Health: Financial health composite (D/E, ROA influence)
        - Growth: Growth transition score
        - Valuation: P/E and PEG-based value assessment
        - Undiscovered: Low analyst coverage = higher score
        - Regulatory: PFIC, VIE, CMIC, ADR risk factors
        - Jurisdiction: Country/exchange stability

        Args:
            result: Dictionary containing analysis results

        Returns:
            Path to generated chart image, or None
        """
        # Skip if charts disabled or quick mode
        if self.skip_charts or self.quick_mode:
            return None

        try:
            from src.charts.base import ChartConfig, ChartFormat, RadarChartData
            from src.charts.extractors.data_block import (
                extract_chart_data_from_data_block,
            )
            from src.charts.generators.radar_chart import generate_radar_chart
            from src.config import config

            # Extract raw facts
            fundamentals_report = self._normalize_string(
                result.get("fundamentals_report", "")
            )
            raw = extract_chart_data_from_data_block(fundamentals_report)

            # Need at least scores to chart
            if raw.adjusted_health_score is None:
                logger.debug("Insufficient data for radar chart (no health score)")
                return None

            # --- Score Normalization Logic (6 Axes) ---

            # 1. Health (Composite: base score + D/E + ROA adjustments)
            # Start with the analyst's health score, then adjust based on D/E and ROA
            health = raw.adjusted_health_score

            # D/E adjustment: Low D/E is good (thesis likes <0.8)
            # If D/E available, adjust health score slightly
            if raw.de_ratio is not None:
                if raw.de_ratio < 0.5:
                    health = min(100.0, health + 5)  # Very low leverage bonus
                elif raw.de_ratio > 2.0:
                    health = max(0.0, health - 10)  # High leverage penalty
                elif raw.de_ratio > 1.0:
                    health = max(0.0, health - 5)  # Moderate leverage penalty

            # ROA adjustment: High ROA is good (thesis likes >7%)
            if raw.roa is not None:
                if raw.roa > 10.0:
                    health = min(100.0, health + 5)  # Strong profitability bonus
                elif raw.roa < 3.0:
                    health = max(0.0, health - 5)  # Weak profitability penalty

            health = max(0.0, min(100.0, health))

            # 2. Growth (Direct from analyst score)
            growth = (
                raw.adjusted_growth_score
                if raw.adjusted_growth_score is not None
                else 50.0
            )
            growth = max(0.0, min(100.0, growth))

            # 3. Valuation (Derived from P/E and PEG)
            # Weighted blend: Favor PEG (60%) for GARP thesis
            pe_score = None
            peg_score = None

            if raw.pe_ratio_ttm and raw.pe_ratio_ttm > 0:
                # P/E Score: 25->0, 15->100
                pe_score = max(0.0, min(100.0, (25.0 - raw.pe_ratio_ttm) * 10.0))

            if raw.peg_ratio and raw.peg_ratio > 0:
                # PEG Score: 2.0->0, 1.0->100
                peg_score = max(0.0, min(100.0, (2.0 - raw.peg_ratio) * 100.0))

            if pe_score is not None and peg_score is not None:
                val_score = (pe_score * 0.4) + (peg_score * 0.6)
            elif pe_score is not None:
                val_score = pe_score
            elif peg_score is not None:
                val_score = peg_score
            else:
                val_score = 50.0  # Neutral if no data

            val_score = max(0.0, min(100.0, val_score))

            # 4. Undiscovered (Derived from Analyst Count)
            # Target: <5 analysts is 100% (hidden gem), >15 is 0% (well-covered)
            coverage = raw.analyst_coverage if raw.analyst_coverage is not None else 10
            undiscovered = (ANALYST_COVERAGE_MAX - coverage) * 10.0
            undiscovered = max(0.0, min(100.0, undiscovered))

            # 5. Regulatory Score (PFIC, VIE, CMIC, ADR risks)
            regulatory = _calculate_regulatory_score(result, raw)

            # 6. Jurisdiction Score (Country/Exchange stability)
            # Start at 100, subtract for risky jurisdictions
            jurisdiction = 100.0

            # Infer jurisdiction risk from ticker suffix or explicit field
            # High-risk jurisdictions (authoritarian, sanctions risk)
            if ticker_in_group(self.ticker, CHINA_SUFFIXES):
                jurisdiction -= 25
            elif ticker_in_group(self.ticker, KOREA_SUFFIXES):
                jurisdiction -= 10  # Moderate geopolitical risk

            # Low-risk developed markets get no penalty
            # (.T Japan, .L London, .AS Amsterdam, .DE Germany, etc.)

            # US Revenue penalty (high US exposure = less diversification benefit)
            if raw.us_revenue_percent:
                if "Not disclosed" not in raw.us_revenue_percent:
                    try:
                        rev_match = re.search(r"([\d.]+)%", raw.us_revenue_percent)
                        if rev_match:
                            rev = float(rev_match.group(1))
                            if rev > 35.0:
                                jurisdiction -= 30  # Hard fail territory
                            elif rev > 25.0:
                                jurisdiction -= 15
                    except Exception:
                        pass

            jurisdiction = max(0.0, min(100.0, jurisdiction))

            # --- Data Quality Warnings ---
            axis_warnings = {}
            footnote_parts = []
            red_flags = get_effective_red_flags(result)

            # Check flags for specific warnings
            for flag in red_flags:
                flag_type = str(flag.get("type", "")).upper()
                if "EARNINGS" in flag_type or "CASH" in flag_type or "FCF" in flag_type:
                    axis_warnings["health"] = True
                if "PFIC" in flag_type or "VIE" in flag_type or "ADR" in flag_type:
                    axis_warnings["regulatory"] = True

            # Check Fundamentals Report for specific data quality strings
            raw_report = str(result.get("fundamentals_report", "")).upper()
            if "DATA QUALITY UNCERTAIN" in raw_report:
                axis_warnings["health"] = True

            # Add footnote if we have warnings
            if axis_warnings:
                footnote_parts.append("* Data quality/risk flag detected")

            footnote = " | ".join(footnote_parts) if footnote_parts else None

            # Create Data Object with 6 axes
            radar_data = RadarChartData(
                ticker=self.ticker,
                trade_date=self.trade_date,
                health_score=health,
                growth_score=growth,
                valuation_score=val_score,
                undiscovered_score=undiscovered,
                regulatory_score=regulatory,
                jurisdiction_score=jurisdiction,
                pe_ratio=raw.pe_ratio_ttm,
                peg_ratio=raw.peg_ratio,
                de_ratio=raw.de_ratio,
                roa=raw.roa,
                analyst_count=raw.analyst_coverage,
                axis_warnings=axis_warnings,
                footnote=footnote,
            )

            # Generate chart
            output_dir = (
                self.image_dir
                if self.image_dir
                else get_runtime_config(config).images_dir
            )
            chart_config = ChartConfig(
                output_dir=output_dir,
                format=ChartFormat.SVG
                if self.chart_format == "svg"
                else ChartFormat.PNG,
                transparent=self.transparent_charts,
                filename_stem=self.report_stem,
            )

            chart_path = generate_radar_chart(radar_data, chart_config)

            if chart_path:
                logger.info("Radar chart generated", path=str(chart_path))

            return chart_path

        except Exception as e:
            logger.warning(
                "radar_chart_generation_failed",
                **summarize_exception(e, operation="radar_chart_generation_failed"),
            )
            return None

    def _normalize_string(self, content: Any) -> str:
        """
        Safely convert content to string, handling lists from LangGraph state accumulation.
        FIXED: Deduplicates list items to prevent repetition loop artifacts.
        """
        if content is None:
            return ""

        if isinstance(content, list):
            # Deduplication logic
            seen = set()
            unique_items = []
            for item in content:
                if not item:
                    continue
                item_str = str(item).strip()
                # Simple hash check for duplicates
                # We check if the first 100 chars match to catch near-duplicates
                # or identical tool outputs repeated in the loop
                key = item_str[:100]
                if key not in seen:
                    seen.add(key)
                    unique_items.append(item_str)

            return "\n\n".join(unique_items)

        return str(content)

    def _display_verdict(self, verdict: str) -> str:
        return "DO NOT INITIATE" if verdict == "DO_NOT_INITIATE" else verdict

    def extract_decision(self, final_decision: str) -> str:
        """
        Extract PM verdict from final decision text.

        Priority order (highest to lowest):
        1. PM_BLOCK VERDICT field (machine-readable, most reliable)
        2. PORTFOLIO MANAGER VERDICT prose
        3. Default to HOLD

        Does NOT use generic keyword search to prevent leakage from
        subordinate agents (Risk Analysts, Researchers, etc.).
        """
        final_decision_upper = self._normalize_string(final_decision).upper()

        # 1. PM_BLOCK VERDICT (machine-readable, highest priority)
        # Must be at line start to avoid matching within "PORTFOLIO MANAGER VERDICT"
        pm_block_match = re.search(
            rf"(?:^|\n)\s*VERDICT\s*:\s*{PM_VERDICT_ALTERNATION}\b",
            final_decision_upper,
        )
        if pm_block_match:
            verdict = canonicalize_pm_verdict(pm_block_match.group(1))
            if verdict != "UNPARSEABLE":
                return self._display_verdict(verdict)

        # 2. PORTFOLIO MANAGER VERDICT prose (captures multi-word verdicts)
        # Matches: #### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE
        verdict_match = PM_VERDICT_HEADER_RE.search(final_decision_upper)
        if verdict_match:
            verdict = canonicalize_pm_verdict(verdict_match.group(1))
            if verdict != "UNPARSEABLE":
                return self._display_verdict(verdict)

        # 3. Default to HOLD (safe fallback, no greedy matching)
        return "HOLD"

    def _extract_prose_decision(self, final_decision: str) -> str | None:
        """Extract only the narrative PM verdict, ignoring PM_BLOCK fields."""
        final_decision_upper = self._normalize_string(final_decision).upper()
        verdict_match = PM_VERDICT_HEADER_RE.search(final_decision_upper)
        if not verdict_match:
            return None
        verdict = canonicalize_pm_verdict(verdict_match.group(1))
        if verdict == "UNPARSEABLE":
            return None
        return self._display_verdict(verdict)

    def _extract_decision_rationale(self, final_decision: str) -> str:
        """
        Extract only the decision rationale section from final_trade_decision.
        Looks for patterns like "DECISION RATIONALE:" or "RATIONALE:".
        """
        final_decision = self._normalize_string(final_decision)

        # Try to find decision rationale section
        rationale_patterns = [
            r"(?:DECISION\s+)?RATIONALE\s*:(.+?)(?:\n\n|\Z)",
            r"REASONING\s*:(.+?)(?:\n\n|\Z)",
            r"JUSTIFICATION\s*:(.+?)(?:\n\n|\Z)",
        ]

        for pattern in rationale_patterns:
            match = re.search(pattern, final_decision, re.IGNORECASE | re.DOTALL)
            if match:
                rationale = match.group(1).strip()
                return self._clean_text(rationale)

        # Fallback: if no specific section found, look for paragraph after decision statement
        decision_keywords = ["BUY", "SELL", "HOLD"]
        lines = final_decision.split("\n")

        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in decision_keywords):
                # Get next non-empty lines as rationale
                rationale_lines = []
                for j in range(i + 1, min(i + 6, len(lines))):
                    if lines[j].strip():
                        rationale_lines.append(lines[j])
                if rationale_lines:
                    return self._clean_text("\n".join(rationale_lines))

        # Last resort: return first 3-4 lines of cleaned text
        paragraphs = [p.strip() for p in final_decision.split("\n\n") if p.strip()]
        if paragraphs:
            return self._clean_text("\n\n".join(paragraphs[:2]))

        return ""

    def _get_final_decision_text(self, result: dict) -> str:
        """
        Extract final decision text from result dictionary with comprehensive fallback logic.

        CRITICAL FIX: Handles cases where Portfolio Manager fails to write to final_trade_decision.
        Fallback hierarchy:
        1. result['final_trade_decision'] - Primary output field (AgentState)
        2. result['investment_plan'] - Research Manager synthesis (fallback)
        3. result['trader_investment_plan'] - Position Planner proposal (last resort)
        4. Error message with debugging info

        Returns:
            str: The final decision text, or an error message with debugging context
        """
        # Try primary field
        final_decision_raw = self._normalize_string(
            result.get("final_trade_decision", "")
        )
        if final_decision_raw and final_decision_raw.strip():
            final_decision_raw = normalize_governance_terms(final_decision_raw)
            return self._reconcile_consultant_wording(final_decision_raw, result)

        # Log warning and try fallbacks
        import structlog

        logger = structlog.get_logger(__name__)
        logger.warning(
            "final_trade_decision is empty - Portfolio Manager may have failed",
            ticker=self.ticker,
            has_investment_plan=bool(result.get("investment_plan")),
            has_trader_plan=bool(result.get("trader_investment_plan")),
        )

        # Fallback 1: Research Manager's investment plan
        investment_plan = self._normalize_string(result.get("investment_plan", ""))
        if investment_plan and investment_plan.strip():
            logger.info(
                "Using investment_plan as fallback for final decision",
                ticker=self.ticker,
            )
            return f"⚠️ **Note: Portfolio Manager output missing - using Research Manager synthesis**\n\n{investment_plan}"

        # Fallback 2: Trader's proposal
        trader_plan = self._normalize_string(result.get("trader_investment_plan", ""))
        if trader_plan and trader_plan.strip():
            logger.info(
                "Using trader_investment_plan as fallback for final decision",
                ticker=self.ticker,
            )
            return (
                "⚠️ **Note: Portfolio Manager output missing - using "
                f"Position Planner proposal**\n\n{trader_plan}"
            )

        # Complete failure - generate error report with debugging context
        logger.error(
            "All decision fields are empty - analysis likely incomplete",
            ticker=self.ticker,
            available_keys=list(result.keys()),
        )

        error_msg = f"""## ⚠️ Analysis Error

**Ticker**: {self.ticker}
**Issue**: Portfolio Manager failed to produce final decision

**Debugging Information**:
- `final_trade_decision`: Empty
- `investment_plan`: {"Present" if result.get("investment_plan") else "Missing"}
- `trader_investment_plan`: {"Present" if result.get("trader_investment_plan") else "Missing"}
- `market_report`: {"Present" if result.get("market_report") else "Missing"}
- `fundamentals_report`: {"Present" if result.get("fundamentals_report") else "Missing"}

**Possible Causes**:
1. Portfolio Manager agent crashed/timeout during LLM call
2. LangGraph routing error prevented Portfolio Manager execution
3. Rate limiting caused silent failure
4. Memory/resource constraints

**Action Required**:
Re-run analysis with verbose logging: `poetry run python -m src.main --ticker {self.ticker}` (or plain `python -m src.main ...` inside an activated venv/container)
"""
        return error_msg

    def _has_primary_final_decision(self, result: dict[str, Any]) -> bool:
        final_decision = self._normalize_string(result.get("final_trade_decision", ""))
        return bool(final_decision and final_decision.strip())

    def _artifact_ok(self, result: dict[str, Any], field: str) -> bool:
        status = (result.get("artifact_statuses", {}) or {}).get(field) or {}
        return not (isinstance(status, dict) and status.get("ok") is False)

    def _build_verification_caveat(self, result: dict[str, Any]) -> str | None:
        review = self._normalize_string(result.get("consultant_review", ""))
        if not review:
            return None

        if "N/A (consultant disabled" in review:
            return None

        consultant_status = (result.get("artifact_statuses", {}) or {}).get(
            "consultant_review"
        ) or {}
        intro = (
            "Independent consultant checks raised verification caveats. Treat any "
            "unsupported or conflicting narrative claims below as suspect until re-verified."
        )
        if consultant_status.get("ok") is False:
            message = self._normalize_string(
                consultant_status.get("message")
                or "Consultant review failed validation."
            )
            return (
                intro
                + "\n\n- External consultant review excluded from PM/report "
                + f"cross-validation: {message}"
            )

        flagged_patterns = (
            r"\b(unsubstantiated|unsupported|likely wrong|likely incorrect)\b",
            r"\b(cannot verify|unable to verify|not supported)\b",
            r"\b(tool[_ -]?error|tool failure|verification failed)\b",
        )
        has_flagged_content = any(
            re.search(pattern, review, flags=re.IGNORECASE)
            for pattern in flagged_patterns
        )
        if consultant_status.get("ok", True) and not has_flagged_content:
            return None

        lines = review.splitlines()
        material_start = next(
            (
                index + 1
                for index, raw_line in enumerate(lines)
                if re.match(
                    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?Material Errors(?:\*\*)?\s*:?.*$",
                    raw_line,
                    flags=re.IGNORECASE,
                )
            ),
            None,
        )
        candidate_source = lines
        if material_start is not None:
            material_lines: list[str] = []
            for raw_line in lines[material_start:]:
                if re.match(r"^\s*#{1,6}\s+", raw_line):
                    break
                material_lines.append(raw_line)
            if any(line.strip() for line in material_lines):
                candidate_source = material_lines

        candidate_lines = []
        for raw_line in candidate_source:
            line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", raw_line).strip()
            if not line:
                continue
            if line.startswith("#") or line.upper().startswith("CONSULTANT REVIEW"):
                continue
            if (
                material_start is None
                and has_flagged_content
                and not any(
                    re.search(pattern, line, flags=re.IGNORECASE)
                    for pattern in flagged_patterns
                )
            ):
                continue
            candidate_lines.append(f"- {line}")
            if len(candidate_lines) == 4:
                break

        if not candidate_lines:
            preview = review.strip().splitlines()
            candidate_lines = [
                f"- {line.strip()}" for line in preview[:3] if line.strip()
            ]

        if not candidate_lines:
            return None

        return intro + "\n\n" + "\n".join(candidate_lines)

    def _reconcile_consultant_wording(
        self,
        final_decision: str,
        result: dict[str, Any],
    ) -> str:
        """Remove contradictions when consultant review exists but is degraded."""
        final_decision = self._normalize_string(final_decision)
        consultant_review = self._normalize_string(result.get("consultant_review", ""))
        if not final_decision or not consultant_review:
            return final_decision
        if "N/A (consultant disabled" in consultant_review:
            return final_decision

        consultant_status = (result.get("artifact_statuses", {}) or {}).get(
            "consultant_review"
        ) or {}
        if consultant_status.get("complete") is False:
            return final_decision

        contradictory_pattern = re.compile(
            r'The pre-screening flagged a "Consultant Conditional" warning, but as '
            r"the external consultant was unavailable to provide specific conditions, "
            r"the verified `DATA_BLOCK` fundamentals and moat signals take absolute "
            r"precedence\.",
            flags=re.IGNORECASE,
        )
        replacement = (
            'The pre-screening flagged a "Consultant Conditional" warning, and the '
            "external consultant identified reservations plus tool-coverage gaps during "
            "independent spot checks. The verified `DATA_BLOCK` fundamentals and moat "
            "signals still drove the final decision, but the consultant caveats above "
            "should be reviewed before acting."
        )
        return contradictory_pattern.sub(replacement, final_decision)

    def _render_failed_analysis(
        self,
        result: dict[str, Any],
        report_parts: list[str],
        *,
        brief_mode: bool,
    ) -> str:
        """Render diagnostics without leaking unvalidated investment advice."""
        from src.analysis_snapshot import render_analysis_snapshot

        validity = result.get("analysis_validity")
        if not isinstance(validity, dict):
            validity = build_analysis_validity(result)

        report_parts.append(
            "\n## Analysis Validity\n\n"
            "This run did not produce a publishable analysis. Required LLM or "
            "data artifacts were missing or invalid. No investment verdict, "
            "position sizing, or execution guidance is presented.\n\n"
        )
        failures = validity.get("required_failures", {})
        if isinstance(failures, dict) and failures:
            report_parts.append("### Required failures\n\n")
            for field, raw_status in sorted(failures.items()):
                status = raw_status if isinstance(raw_status, dict) else {}
                message = str(status.get("message") or "Invalid or unavailable")
                report_parts.append(f"- `{field}`: {message}\n")
            report_parts.append("\n")

        snapshot_context = render_analysis_snapshot(result.get("analysis_snapshot"))
        if snapshot_context:
            report_parts.append(
                "### Canonical data diagnostics\n\n"
                "```text\n"
                f"{snapshot_context.rstrip()}\n"
                "```\n\n"
            )

        red_flags = get_effective_red_flags(result)
        if red_flags:
            report_parts.append("### Deterministic flags\n\n")
            for flag in red_flags:
                if not isinstance(flag, dict):
                    continue
                flag_type = str(flag.get("type") or "UNKNOWN")
                detail = str(flag.get("detail") or "No detail available")
                report_parts.append(f"- `{flag_type}`: {detail}\n")
            report_parts.append("\n")

        mode_indicator = (
            "Brief Mode, Quick Models"
            if brief_mode and self.quick_mode
            else "Brief Mode"
            if brief_mode
            else "Quick Models"
            if self.quick_mode
            else "Full Mode"
        )
        report_parts.append(
            f"\n*Generated by Multi-Agent Investment Analysis System "
            f"({mode_indicator}, diagnostic only) - {self.timestamp}*\n"
        )
        return "".join(report_parts)

    def generate_report(self, result: dict, brief_mode: bool = False) -> str:
        """
        Generate markdown report from analysis results.

        Args:
            result: Dictionary containing analysis results
            brief_mode: If True, output only header, summary, and decision rationale
        """

        # Get final decision with comprehensive error handling
        final_decision_raw = self._get_final_decision_text(result)
        publishable = is_publishable_analysis(result)
        pm_contract = extract_pm_block(final_decision_raw)
        block_verdict = canonicalize_pm_verdict(pm_contract.verdict)
        prose_decision = self._extract_prose_decision(final_decision_raw)
        prose_verdict = canonicalize_pm_verdict(prose_decision)
        contract_caveats: list[str] = []
        if block_verdict != "UNPARSEABLE":
            extracted_decision = self._display_verdict(block_verdict)
            if prose_verdict != "UNPARSEABLE" and prose_verdict != block_verdict:
                contract_caveats.append(
                    "- PM prose verdict was "
                    f"{self._display_verdict(prose_verdict)}; PM_BLOCK verdict was "
                    f"{self._display_verdict(block_verdict)}. PM_BLOCK was used."
                )
        else:
            extracted_decision = self.extract_decision(final_decision_raw)
        decision = extracted_decision if publishable else "ANALYSIS FAILED"

        # Build title
        if self.company_name:
            title = f"# {self.ticker} ({self.company_name}): {decision}"
        else:
            title = f"# {self.ticker}: {decision}"

        # Build report sections
        report_parts = [title, f"\n**Analysis Date:** {self.timestamp}\n", "---\n"]

        explicit_validity = result.get("analysis_validity")
        if (
            not publishable
            and isinstance(explicit_validity, dict)
            and explicit_validity.get("publishable") is False
        ):
            return self._render_failed_analysis(
                result,
                report_parts,
                brief_mode=brief_mode,
            )
        if not publishable:
            report_parts.append(
                "\n## Analysis Validity\n\n"
                "This run did not produce a publishable analysis. Required LLM or "
                "data artifacts were missing or invalid, so verdict-dependent output "
                "should be treated as diagnostics only.\n"
            )

        verification_caveat = self._build_verification_caveat(result)
        if contract_caveats:
            contract_intro = (
                "Portfolio Manager output contained structured/verbal verdict "
                "inconsistency. The structured PM_BLOCK contract controls."
            )
            contract_text = contract_intro + "\n\n" + "\n".join(contract_caveats)
            verification_caveat = (
                f"{verification_caveat}\n\n{contract_text}"
                if verification_caveat
                else contract_text
            )
        if verification_caveat:
            report_parts.append("\n## Verification Caveats\n\n")
            report_parts.append(f"{verification_caveat}\n\n---\n")

        # Investment Memo — memo-first restructure (above charts and appendix).
        # The memo aggregates verdict, thesis, key numbers, top risks, kill criteria,
        # and confidence into a tight scannable header so readers don't have to read
        # the full agent transcript to see the call. Falls back to a placeholder
        # block if PM output is unavailable.
        try:
            from src.reporting.memo import render_memo_for_state

            memo_state = dict(result)
            if self.valuation_context and "valuation_context" not in memo_state:
                memo_state["valuation_context"] = self.valuation_context
            report_parts.append("\n")
            report_parts.append(render_memo_for_state(memo_state))
        except Exception:  # pragma: no cover — defense-in-depth
            # Memo rendering should never block report publication.
            pass

        # Red Flag Pre-Screening (if applicable)
        red_flags = get_effective_red_flags(result)
        pre_screening_result = result.get("pre_screening_result", "PASS")

        if red_flags or pre_screening_result == "REJECT":
            report_parts.append("\n## 🚨 Red Flag Pre-Screening\n\n")

            if pre_screening_result == "REJECT":
                report_parts.append(
                    "**Status**: CRITICAL RED FLAGS DETECTED - AUTO-REJECT\n\n"
                )
            else:
                report_parts.append(
                    "**Status**: ⚠️ Warnings Detected - Proceed with Caution\n\n"
                )

            if red_flags:
                for flag in red_flags:
                    flag_type = flag.get("type", "UNKNOWN")
                    severity = flag.get("severity", "UNKNOWN")
                    detail = flag.get("detail", "No details")

                    report_parts.append(f"- **{flag_type}** ({severity}): {detail}\n")

            if pre_screening_result == "REJECT":
                report_parts.append(
                    "\n*Debate phase skipped due to critical red flags. "
                )
                report_parts.append(
                    "Stock routed directly to Portfolio Manager for final decision.*\n"
                )

            report_parts.append("\n---\n\n")

        # Thesis Compliance Visual (quick-scan bar charts)
        try:
            from src.thesis_visualizer import generate_thesis_visual

            thesis_visual = generate_thesis_visual(final_decision_raw)
            if thesis_visual:
                report_parts.append("## Thesis Compliance at a Glance\n\n")
                report_parts.append(f"{thesis_visual}\n\n---\n")
        except ImportError:
            pass  # Visualizer not available, skip

        # Thesis Alignment Radar Chart
        # Report assembly now prioritizes charts generated by the graph state.
        # These charts are 'verdict-aware' and reflect the Portfolio Manager's adjustments.
        # Fallback logic remains for manual/legacy runs where the terminal node is skipped.
        chart_paths = result.get("chart_paths", {})
        radar_path = None
        if chart_paths.get("radar"):
            radar_path = Path(chart_paths["radar"])
        elif not chart_paths:
            # Fallback: generate chart here if graph didn't produce it
            radar_path = self._generate_radar_chart(result)

        if radar_path:
            report_parts.append("## Thesis Alignment\n\n")

            radar_link = _markdown_asset_link(radar_path, self.report_dir)

            report_parts.append(f"![Thesis Alignment Radar]({radar_link})\n\n---\n")

        # Football Field Valuation Chart
        # Use chart_paths from graph state if available (post-PM generation)
        chart_path = None
        if chart_paths.get("football_field"):
            chart_path = Path(chart_paths["football_field"])
        elif not chart_paths:
            # Fallback: generate chart here if graph didn't produce it
            chart_path = self._generate_chart(result)

        if chart_path:
            report_parts.append("## Valuation Chart\n\n")

            chart_link = _markdown_asset_link(chart_path, self.report_dir)

            report_parts.append(f"![Football Field Chart]({chart_link})\n\n---\n")

        # Executive Summary (always included)
        if final_decision_raw:
            report_parts.append("## Executive Summary\n\n")
            # Demote headers (### → ####) since we're nesting under ## Executive Summary
            cleaned = self._clean_text(final_decision_raw, demote_headers=True)
            report_parts.append(f"{cleaned}\n\n---\n")
        else:
            # This shouldn't happen with new fallback logic, but handle it anyway
            report_parts.append("## Executive Summary\n\n")
            report_parts.append(
                "**Error**: No decision output available from any agent.\n\n---\n"
            )

        # If brief mode, skip adding duplicate Decision Rationale
        # The Executive Summary already contains the full decision with rationale
        if brief_mode:
            # Footer
            mode_indicator = (
                "Brief Mode, Quick Models" if self.quick_mode else "Brief Mode"
            )
            report_parts.append(
                f"\n*Generated by Multi-Agent Investment Analysis System ({mode_indicator}) - {self.timestamp}*\n"
            )
            return "".join(report_parts)

        # Full mode: include all sections
        # Helper function to add sections safely
        def add_section(key, title):
            raw_content = result.get(key, "")
            content = self._normalize_string(raw_content)
            if key == "trader_investment_plan":
                content = normalize_retail_position_terms(content)

            if content and not content.startswith("Error"):
                report_parts.append(f"## {title}\n\n")
                # Clean content and strip redundant leading headers that match section title
                cleaned = self._clean_text(content, demote_headers=True)
                cleaned = self._strip_redundant_header(cleaned, title)
                report_parts.append(f"{cleaned}\n\n")

        market_report = result.get("market_report", "")
        if market_report and extracted_decision in _NON_EXECUTABLE_VERDICTS:
            result["market_report"] = _suppress_executable_levels(
                self._normalize_string(market_report)
            )
        add_section("market_report", "Technical Analysis")

        # Clean fundamentals: keep only final self-corrected DATA_BLOCK
        # Import inside function to prevent circular dependency with utils.py
        fund_report = result.get("fundamentals_report", "")
        if fund_report:
            try:
                from src.utils import clean_duplicate_data_blocks

                fund_report = self._normalize_string(fund_report)
                fund_report = clean_duplicate_data_blocks(fund_report)
                fund_report = self._move_data_block_to_end(fund_report)
                result["fundamentals_report"] = fund_report
            except ImportError:
                pass  # Fallback if utils not available

        add_section("fundamentals_report", "Fundamental Analysis")

        # Qualify "undiscovered" language when coverage is not independently
        # confirmed low. Prompt-level guidance (sentiment v5.4) proved
        # insufficient (KTY.WA 2026-06-27 still asserted "Strongly Undiscovered"
        # at MODERATE total coverage), so this is a deterministic backstop keyed
        # off the DATA_BLOCK coverage fields the Fundamentals Analyst owns.
        sentiment_report = result.get("sentiment_report", "")
        if sentiment_report:
            result["sentiment_report"] = self._soften_undiscovered_language(
                self._normalize_string(sentiment_report), fund_report
            )
        add_section("sentiment_report", "Market Sentiment")

        # Reformat MACRO_DETECTION block before rendering news section
        news_report = result.get("news_report", "")
        if news_report:
            result["news_report"] = self._reformat_macro_detection(
                self._normalize_string(news_report)
            )
        add_section("news_report", "News & Catalysts")
        if not self._has_primary_final_decision(result):
            add_section("investment_plan", "Investment Recommendation")

        # CRITICAL: Include consultant review if present (external cross-validation)
        consultant_review = result.get("consultant_review", "")
        if (
            consultant_review
            and consultant_review.strip()
            and self._artifact_ok(result, "consultant_review")
        ):
            # Check if it's a real review (not an error message or "N/A")
            normalized = self._normalize_string(consultant_review)
            if (
                normalized
                and "N/A (consultant disabled" not in normalized
                and not normalized.startswith("Consultant Review Error")
            ):
                report_parts.append(
                    "## 🔍 External Consultant Review (Cross-Validation)\n\n"
                )
                report_parts.append(
                    "*Independent review by OpenAI ChatGPT to validate Gemini analysis*\n\n"
                )
                report_parts.append(
                    f"{self._clean_text(normalized, demote_headers=True)}\n\n"
                )

        _verdict = extracted_decision
        if _verdict in _NON_EXECUTABLE_VERDICTS:
            report_parts.append("## Position Plan\n\n")
            report_parts.append(
                f"*Position parameters not applicable — "
                f"Portfolio Manager verdict: **{_verdict}**.*\n\n"
            )
        else:
            add_section("trader_investment_plan", "Position Plan")

        # Risk Assessment (if present)
        risk_state = result.get("risk_debate_state", {})
        if risk_state:
            # Handle both dict and list (take last if list)
            if isinstance(risk_state, list):
                risk_state = risk_state[-1] if risk_state else {}

            if isinstance(risk_state, dict):
                # Read from dedicated fields (parallel-safe architecture)
                risky = risk_state.get("current_risky_response", "")
                safe = risk_state.get("current_safe_response", "")
                neutral = risk_state.get("current_neutral_response", "")

                if risky or safe or neutral:
                    risk_title = (
                        "Risk Assessment — Archival Debate (Non-Executable)"
                        if _verdict in _NON_EXECUTABLE_VERDICTS
                        else "Risk Assessment"
                    )
                    report_parts.append(f"## {risk_title}\n\n")
                    if _verdict in _NON_EXECUTABLE_VERDICTS:
                        report_parts.append(
                            "*These subordinate views predate or challenge the PM "
                            "override. They are retained for audit context and are "
                            "not position recommendations.*\n\n"
                        )
                    if risky:
                        report_parts.append("### Risky Analyst (Aggressive)\n\n")
                        report_parts.append(
                            f"{self._clean_text(risky, demote_headers=True)}\n\n"
                        )
                    if safe:
                        report_parts.append("### Safe Analyst (Conservative)\n\n")
                        report_parts.append(
                            f"{self._clean_text(safe, demote_headers=True)}\n\n"
                        )
                    if neutral:
                        report_parts.append("### Neutral Analyst (Balanced)\n\n")
                        report_parts.append(
                            f"{self._clean_text(neutral, demote_headers=True)}\n\n"
                        )

        # Footer
        mode_suffix = " (Quick Models)" if self.quick_mode else ""
        report_parts.append(
            f"*Generated by Multi-Agent Investment Analysis System{mode_suffix} - {self.timestamp}*\n"
        )

        return "".join(report_parts)

    def _clean_text(self, text: str, demote_headers: bool = False) -> str:
        """
        Clean up text for markdown output.

        Args:
            text: The text to clean
            demote_headers: If True, demote ### headers to #### (for nested content)
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        # Repair exact glued block-boundary defects from older artifacts or
        # upstream model-format drift before any header demotion occurs.
        text = normalize_structured_block_boundaries(text) or text

        # Remove agent prefixes if present
        text = re.sub(
            r"^(Bull Analyst:|Bear Analyst:|Risky Analyst:|Safe Analyst:|"
            r"Neutral Analyst:|Trader:|Portfolio Manager:)\s*",
            "",
            text,
            flags=re.MULTILINE,
        )

        # Strip raw OpenAI reasoning-metadata lines that leaked via str(dict) fallback
        text = re.sub(
            r"^\s*\{['\"]id['\"]\s*:\s*['\"]rs_[^'\"]+['\"]\s*,.*?['\"]type['\"]\s*:\s*['\"]reasoning['\"].*?\}\s*\n?",
            "",
            text,
            flags=re.MULTILINE,
        )

        # Strip PM_BLOCK structured block (consumed by code; not reader-facing).
        # Fenced case: remove the whole fence (and heading) only when it encloses
        # the START/END markers, so resolution prose the PM may emit before the
        # markers inside that fence is removed with them (the 6831.HK regression).
        text = _strip_fenced_pm_machine_block(text)
        # Bare (unfenced) START/END pair — LLMs drift between 3 and 4 hashes.
        # Line-count cap (30 lines) prevents runaway matching if END is absent/malformed.
        _pm_content = r"(?:[^\n]*\n){0,30}"
        _pm_start = fenced_marker_fragment("PM_BLOCK", "START")
        _pm_end = fenced_marker_fragment("PM_BLOCK", "END")
        text = re.sub(
            _pm_start + r"\n" + _pm_content + _pm_end + r"[^\n]*",
            "",
            text,
            flags=re.DOTALL,
        )

        # Preserve the PM's consultant reconciliation as reader-facing prose.
        # The PM emits CONSULTANT_RESOLUTION (CONCERN/DATA_CHECK/VERDICT bullets)
        # — often the *only* place the reconciliation appears — under the
        # "CONSULTANT DISAGREEMENT RESOLUTION" heading. Previously the whole block
        # was stripped; when the PM fenced it, the fence survived and left an empty
        # ``` block under the heading (the KTY.WA 2026-06-27 regression). Now we
        # drop only the machine label line and unwrap any fence, keeping the
        # bullets so the reader sees the reconciliation.
        # Fenced form: ```\nCONSULTANT_RESOLUTION:\n- ...\n```  -> keep bullets.
        consultant_resolution = re.escape(
            unfenced_label("CONSULTANT_RESOLUTION").rstrip(":")
        )
        text = re.sub(
            rf"```[^\S\n]*\n[#*\s]*{consultant_resolution}[*:]*\s*\n"
            r"(?P<bullets>(?:[#*\s]*-[^\n]+\n)+)\s*```",
            lambda m: m.group("bullets"),
            text,
        )
        # Unfenced form: drop only the label line, keep the bullets that follow.
        text = re.sub(
            rf"^[#*\s]*{consultant_resolution}[*:]*\s*\n(?=[#*\s]*-)",
            "",
            text,
            flags=re.MULTILINE,
        )
        # Safety net: collapse any fenced block left entirely empty by prior strips
        # (e.g. a machine block whose body was removed), without touching fences
        # that still contain content such as APAC_RESOLUTION/AUDITOR_RESOLUTION.
        text = re.sub(r"```[^\S\n]*\n[ \t]*```[ \t]*\n?", "", text)

        # Turn the unresolved-auditor machine stub (NOT_PROVIDED/UNVERIFIABLE,
        # auto-injected when the auditor named a concern the PM didn't reconcile)
        # into a readable caveat. Real AUDITOR_RESOLUTION blocks are untouched.
        text = self._reformat_unresolved_auditor_block(text)

        # Strip "Analyzing TICKER - Company" openers (redundant with report title).
        # Matches ticker by requiring a dot-delimited exchange suffix (e.g. .HK, .T, .DE).
        # Also handles bold variant: Analyzing **0148.HK (Company Name)**
        text = re.sub(
            r"^Analyzing\s+\**\S*\.[A-Z]{1,3}[A-Z0-9.-]*\**.*\n?",
            "",
            text,
            flags=re.MULTILINE,
        )

        # Normalize DECISION LOGIC blocks - ensure they're properly fenced
        # Fix orphaned code block markers from truncated content
        text = self._normalize_code_blocks(text)

        # Demote headers if requested (### → ####) for nested content
        if demote_headers:
            text = re.sub(r"^###\s+", "#### ", text, flags=re.MULTILINE)

        if not text.endswith("\n"):
            return text + "\n"
        return text

    def _strip_redundant_header(self, text: str, section_title: str) -> str:
        """
        Remove leading header if it matches the section title.

        Prevents duplicate headers like:
            ## Technical Analysis
            #### Technical Analysis  ← this gets stripped
            RSI at 65...
        """
        if not text:
            return text

        lines = text.split("\n", 1)
        first_line = lines[0].strip()

        # Check if first line is a header matching the section title
        # Match various header formats: ####, ###, ##, # followed by title
        header_match = re.match(r"^#{1,6}\s*(.+)$", first_line)
        if header_match:
            header_text = header_match.group(1).strip()
            # Normalize for comparison (remove markdown formatting)
            normalized_header = re.sub(r"\*+", "", header_text).strip().lower()
            normalized_title = section_title.lower()

            if normalized_header == normalized_title:
                # Strip the redundant header
                return lines[1].lstrip("\n") if len(lines) > 1 else ""

        return text

    @staticmethod
    def _soften_undiscovered_language(sentiment_text: str, fund_report: str) -> str:
        """Qualify unconditional "undiscovered" claims when coverage isn't confirmed low.

        Fires when the DATA_BLOCK reports MODERATE/HIGH/UNKNOWN total analyst
        coverage, or an ANALYST_COVERAGE_DATA_QUALITY_NOTE is present. Prepends a
        one-time "Coverage caveat" banner disclosing that the discovery/visibility
        framing is relative, not absolute; the raw sentiment prose is left intact
        (no fragile in-place rewriting). Idempotent via the banner-presence guard.
        Genuinely-low, confidently-measured coverage is left untouched.
        """
        if not sentiment_text:
            return sentiment_text
        total_est = (
            extract_data_block_field(fund_report, "ANALYST_COVERAGE_TOTAL_EST") or ""
        ).upper()
        has_note = "ANALYST_COVERAGE_DATA_QUALITY_NOTE" in (fund_report or "")
        if not (has_note or total_est in {"MODERATE", "HIGH", "UNKNOWN"}):
            return sentiment_text
        # A caveat banner neutralizes the whole family of overclaim synonyms
        # ("undiscovered", "effectively invisible", "entirely absent",
        # "international ignorance") without fragile in-place phrase rewriting.
        # In-place token substitution was removed: it spliced a noun phrase into
        # adjective / header / label slots ("#### low English-language aggregator
        # visibility STATUS ASSESSMENT", "the stock is … by retail crowds"),
        # producing ungrammatical prose. The banner discloses the qualification
        # cleanly; the raw prose is left grammatical.
        caveat = (
            "> **Coverage caveat:** discovery/visibility framing below reflects only "
            "low *Western English-language retail* visibility — analyst coverage is not "
            "confirmed low, so treat it as relative, not an absolute claim.\n\n"
        )
        if "Coverage caveat:" in sentiment_text:
            return sentiment_text
        return caveat + sentiment_text

    # Shared between the fenced and unfenced rewrite forms below so the two
    # regexes cannot drift (mirrors the CONSULTANT_RESOLUTION two-form handling).
    _AUDITOR_STUB_BODY = (
        r"[#*\s]*AUDITOR_RESOLUTION:?[ \t]*\n"
        r"(?:[#*\s]*-\s*FINDING:[^\n]*\n)?"
        r"[#*\s]*-\s*DATA_CHECK:\s*NOT_PROVIDED[ \t]*\n"
        r"[#*\s]*-\s*VERDICT:\s*UNVERIFIABLE[ \t]*\n?"
    )
    _AUDITOR_NOTE = (
        "> **Auditor note:** The forensic auditor flagged anomalies the PM did "
        "not explicitly reconcile; treat earnings-quality / cash-flow "
        "conclusions as unverified.\n"
    )

    @classmethod
    def _reformat_unresolved_auditor_block(cls, text: str) -> str:
        """Turn the NOT_PROVIDED/UNVERIFIABLE auditor stub into a prose caveat.

        Only the machine stub injected by ``_ensure_auditor_resolution_block``
        (DATA_CHECK: NOT_PROVIDED + VERDICT: UNVERIFIABLE) is rewritten; a
        populated AUDITOR_RESOLUTION (real DATA_CHECK / verdict) and
        ``AUDITOR_RESOLUTION: NONE`` are left untouched. Handles both the
        code-injected unfenced form and a PM-authored fenced form — the fenced
        sub runs first and consumes the whole fence, so no orphan ``` lines
        can trap the blockquote inside a code block.
        """
        text = re.sub(
            rf"(?ms)^```[^\S\n]*\n{cls._AUDITOR_STUB_BODY}\s*```[ \t]*\n?",
            cls._AUDITOR_NOTE,
            text,
        )
        return re.sub(
            rf"(?ms)^{cls._AUDITOR_STUB_BODY}",
            cls._AUDITOR_NOTE,
            text,
        )

    @staticmethod
    def _reformat_macro_detection(text: str) -> str:
        """
        Replace raw MACRO_DETECTION key=value block with a prose callout (TRIGGERED=YES)
        or remove it entirely (TRIGGERED=NO).
        """
        match = re.search(
            r"#{1,6}\s+MACRO_DETECTION\s*:?\s*\n((?:[A-Z_]+:[^\n]*\n?)+)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return text
        block = match.group(1)
        fields: dict[str, str] = {}
        for field_match in re.finditer(r"^([A-Z_]+):\s*(.*)$", block, re.MULTILINE):
            key, value = field_match.groups()
            fields[str(key)] = str(value)
        text = text[: match.start()] + text[match.end() :]
        if fields.get("TRIGGERED", "NO").upper() == "YES":
            headline = fields.get("HEADLINE", "")
            impact = fields.get("THESIS_IMPACT", "")
            callout = f"\n> **Macro event detected** ({impact}): {headline}\n"
            text = text.rstrip() + callout
        return text

    @staticmethod
    def _move_data_block_to_end(text: str) -> str:
        """Extract DATA_BLOCK from wherever it appears and re-append at end.

        Matches both `####` and `###` headers, and tolerates a parenthetical
        annotation after START DATA_BLOCK (e.g. '(INTERNAL SCORING…)').
        Missing or malformed END markers cause no match — nothing is moved.
        """
        data_start = fenced_marker_fragment("DATA_BLOCK", "START")
        data_end = fenced_marker_fragment("DATA_BLOCK", "END")
        match = re.search(
            rf"({data_start}"
            r"\n"
            r"(?:[^\n]*\n){0,120}"  # bounded: at most ~120 lines of content
            rf"{data_end}\n?)",
            text,
            re.DOTALL,
        )
        if not match:
            return text
        block = match.group(1)
        text = text[: match.start()] + text[match.end() :]
        return text.rstrip() + "\n\n" + block

    def _normalize_code_blocks(self, text: str) -> str:
        """
        Normalize code blocks to ensure proper formatting.
        Fixes orphaned markers and ensures consistent DECISION LOGIC presentation.
        """
        # Fix orphaned closing markers (=== at start of line without opening fence)
        # Pattern: line starting with === or ====================== not inside a code block
        lines = text.split("\n")
        result_lines = []
        in_code_block = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Track code block state
            if stripped.startswith("```"):
                if in_code_block:
                    in_code_block = False
                else:
                    in_code_block = True
                result_lines.append(line)
                i += 1
                continue

            # If we're not in a code block and see DECISION LOGIC markers
            if not in_code_block:
                # Check for orphaned === DECISION LOGIC === or ======================
                if stripped == "=== DECISION LOGIC ===" or stripped.startswith(
                    "====================="
                ):
                    # Skip orphaned markers - they're artifacts from truncation
                    i += 1
                    continue

                # Check for DECISION LOGIC block that needs fencing
                if "=== DECISION LOGIC ===" in stripped and not stripped.startswith(
                    "```"
                ):
                    # This is an unfenced DECISION LOGIC block - wrap it
                    result_lines.append("```")
                    result_lines.append(line)
                    # Collect until we hit the closing ===
                    i += 1
                    while i < len(lines):
                        next_line = lines[i]
                        result_lines.append(next_line)
                        if "=====================" in next_line.strip():
                            result_lines.append("```")
                            break
                        i += 1
                    i += 1
                    continue

            result_lines.append(line)
            i += 1

        return "\n".join(result_lines)


def suppress_logging():
    """
    Suppress all logging output except critical errors.
    Ensures logging goes to stderr so it doesn't pollute stdout reports.
    """
    import warnings

    # Configure root logger to only show CRITICAL errors, directed to stderr
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,  # Override any existing configuration
    )

    # Explicitly set root logger level (basicConfig might not work if already configured)
    logging.root.setLevel(logging.CRITICAL)

    # Suppress all existing loggers
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False

    # Suppress common noisy libraries
    for logger_name in ["httpx", "openai", "httpcore", "langchain", "langgraph"]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Suppress structlog (used by token_tracker and agents)
    try:
        import structlog

        def null_processor(logger, method_name, event_dict):
            """Drop all log events."""
            raise structlog.DropEvent

        structlog.configure(
            processors=[null_processor],
            wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=False,
        )
    except ImportError:
        pass  # structlog not available, skip
