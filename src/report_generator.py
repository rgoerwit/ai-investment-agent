"""
Quiet Mode Report Generator for Multi-Agent Trading System
FIXED: Handles LangGraph list outputs to prevent 'list object has no attribute startswith' errors.
FIXED: Added deduplication to prevent stuttering output in final reports.
FIXED: Case-insensitive regex matching for decision extraction.
UPDATED: Added brief_mode flag for condensed output.
UPDATED: Added comprehensive error handling and fallback logic for missing Portfolio Manager output.
UPDATED: Added Football Field chart generation integration.
"""

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Local import for utility function to avoid circular dependency at module level
# We import inside the method where it is needed


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
    ) -> None:
        """Store valuation context for article writer to use.

        This enables the article writer to address discrepancies between
        the football field chart visuals and the investment decision.
        """
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
- Methodology: {methodology or 'P/E Normalization'}
- Target Range: ${target_low:.2f} - ${target_high:.2f}
- Fair Value (midpoint): ${fair_value:.2f}
- Current Price: ${current_price:.2f}
- Price Position: {position_desc}
- Confidence: {confidence or 'N/A'}

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
            from src.charts.extractors.valuation import calculate_valuation_targets
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
            red_flags = result.get("red_flags", [])

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

            football_data = FootballFieldData(
                ticker=self.ticker,
                trade_date=self.trade_date,
                current_price=chart_data.current_price,
                fifty_two_week_high=chart_data.fifty_two_week_high,
                fifty_two_week_low=chart_data.fifty_two_week_low,
                moving_avg_50=chart_data.moving_avg_50,
                moving_avg_200=chart_data.moving_avg_200,
                external_target_high=chart_data.external_target_high,
                external_target_low=chart_data.external_target_low,
                external_target_mean=chart_data.external_target_mean,
                our_target_low=targets.low,
                our_target_high=targets.high,
                target_methodology=targets.methodology,
                target_confidence=targets.confidence,
                quality_warnings=quality_warnings if quality_warnings else None,
                footnote="Targets based on P/E normalization"
                if targets.methodology
                else None,
            )

            # Store valuation context for article writer (D1 implementation)
            self._store_valuation_context(
                current_price=chart_data.current_price,
                target_low=targets.low,
                target_high=targets.high,
                methodology=targets.methodology,
                confidence=targets.confidence,
            )

            # Configure chart generation
            # Use custom image_dir if provided, otherwise fall back to config default
            output_dir = self.image_dir if self.image_dir else config.images_dir
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
            logger.warning(f"Chart generation dependencies not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")
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
            undiscovered = (15.0 - coverage) * 10.0
            undiscovered = max(0.0, min(100.0, undiscovered))

            # 5. Regulatory Score (PFIC, VIE, CMIC, ADR risks)
            # Start at 100 (no regulatory concerns), subtract for each risk
            regulatory = 100.0

            # PFIC Penalty (passive foreign investment company)
            if raw.pfic_risk:
                risk_upper = raw.pfic_risk.upper()
                if "HIGH" in risk_upper:
                    regulatory -= 40  # Major tax/reporting burden
                elif "MEDIUM" in risk_upper:
                    regulatory -= 20

            # VIE Structure Penalty (Variable Interest Entity - China risk)
            if raw.vie_structure is True:
                regulatory -= 25  # Significant structural risk

            # CMIC Penalty (Chinese Military-Industrial Complex list)
            if raw.cmic_flagged is True:
                regulatory -= 35  # Investment restrictions risk

            # ADR Thesis Impact Penalty (Sponsored ADR = more discovered)
            if raw.adr_impact:
                impact_upper = raw.adr_impact.upper()
                if "MODERATE_CONCERN" in impact_upper:
                    regulatory -= 10  # Minor thesis concern

            regulatory = max(0.0, min(100.0, regulatory))

            # 6. Jurisdiction Score (Country/Exchange stability)
            # Start at 100, subtract for risky jurisdictions
            jurisdiction = 100.0

            # Infer jurisdiction risk from ticker suffix or explicit field
            ticker_upper = self.ticker.upper()

            # High-risk jurisdictions (authoritarian, sanctions risk)
            if any(
                suffix in ticker_upper
                for suffix in [".SS", ".SZ", ".HK"]  # China mainland, HK
            ):
                jurisdiction -= 25
            elif any(
                suffix in ticker_upper
                for suffix in [".KS", ".KQ"]  # Korea
            ):
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
            red_flags = result.get("red_flags", [])

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
            output_dir = self.image_dir if self.image_dir else config.images_dir
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
            logger.warning(f"Radar chart generation failed: {e}")
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

    def extract_decision(self, final_decision: str) -> str:
        """Extract BUY/SELL/HOLD decision from final decision text."""

        # Normalize input first
        final_decision = self._normalize_string(final_decision)

        # Look for explicit decision markers in order of preference
        # Use UPPER CASE matching since we upper() the input string

        # 0. "PORTFOLIO MANAGER VERDICT:" (NEW SENTINEL - Highest Priority)
        # Unique token that prevents capturing quoted text from other agents
        verdict_match = re.search(
            r"\bPORTFOLIO\s+MANAGER\s+VERDICT\s*:\s*\*?\*?([A-Z]+)\*?\*?",
            final_decision.upper(),
        )
        if verdict_match:
            decision = verdict_match.group(1)
            if decision in ["BUY", "SELL", "HOLD"]:
                return decision

        # 1. "Action:" in FINAL EXECUTION PARAMETERS (highest priority)
        action_match = re.search(
            r"\bACTION\s*:\s*\*?\*?([A-Z]+)\*?\*?", final_decision.upper()
        )
        if action_match:
            decision = action_match.group(1)
            if decision in ["BUY", "SELL", "HOLD"]:
                return decision

        # 2. "FINAL DECISION:"
        final_decision_match = re.search(
            r"\bFINAL\s+DECISION\s*:\s*\*?\*?([A-Z]+)\*?\*?", final_decision.upper()
        )
        if final_decision_match:
            decision = final_decision_match.group(1)
            if decision in ["BUY", "SELL", "HOLD"]:
                return decision

        # 3. "Decision:" fallback
        decision_match = re.search(
            r"\bDECISION\s*:\s*\*?\*?([A-Z]+)\*?\*?", final_decision.upper()
        )
        if decision_match:
            decision = decision_match.group(1)
            if decision in ["BUY", "SELL", "HOLD"]:
                return decision

        # 4. Generic keyword search (risky, but better than nothing)
        generic_match = re.search(r"\b(BUY|SELL|HOLD)\b", final_decision.upper())
        if generic_match:
            decision = generic_match.group(1)
            return decision

        return "HOLD"  # Default to HOLD if completely unclear

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
        3. result['trader_investment_plan'] - Trader proposal (last resort)
        4. Error message with debugging info

        Returns:
            str: The final decision text, or an error message with debugging context
        """
        # Try primary field
        final_decision_raw = self._normalize_string(
            result.get("final_trade_decision", "")
        )
        if final_decision_raw and final_decision_raw.strip():
            return final_decision_raw

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
            return f"⚠️ **Note: Portfolio Manager output missing - using Trader proposal**\n\n{trader_plan}"

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
Re-run analysis with verbose logging: `poetry run python -m src.main --ticker {self.ticker}`
"""
        return error_msg

    def generate_report(self, result: dict, brief_mode: bool = False) -> str:
        """
        Generate markdown report from analysis results.

        Args:
            result: Dictionary containing analysis results
            brief_mode: If True, output only header, summary, and decision rationale
        """

        # Get final decision with comprehensive error handling
        final_decision_raw = self._get_final_decision_text(result)
        decision = self.extract_decision(final_decision_raw)

        # Build title
        if self.company_name:
            title = f"# {self.ticker} ({self.company_name}): {decision}"
        else:
            title = f"# {self.ticker}: {decision}"

        # Build report sections
        report_parts = [title, f"\n**Analysis Date:** {self.timestamp}\n", "---\n"]

        # Red Flag Pre-Screening (if applicable)
        red_flags = result.get("red_flags", [])
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

        # Thesis Alignment Radar Chart (New)
        radar_path = self._generate_radar_chart(result)
        if radar_path:
            report_parts.append("## Thesis Alignment\n\n")

            # Calculate link path
            if self.report_dir:
                try:
                    radar_link = radar_path.resolve().relative_to(
                        self.report_dir.resolve()
                    )
                except ValueError:
                    radar_link = radar_path.resolve()
            else:
                radar_link = f"{radar_path.parent.name}/{radar_path.name}"

            report_parts.append(f"![Thesis Alignment Radar]({radar_link})\n\n---\n")

        # Football Field Valuation Chart (skip in quick mode)
        chart_path = self._generate_chart(result)
        if chart_path:
            report_parts.append("## Valuation Chart\n\n")

            # Calculate link path
            if self.report_dir:
                try:
                    # Try to make path relative to report directory
                    # We resolve both to absolute paths first to be safe
                    chart_link = chart_path.resolve().relative_to(
                        self.report_dir.resolve()
                    )
                except ValueError:
                    # If not relative (not a subdir), use absolute path
                    # This happens if --imagedir is outside --output directory tree
                    chart_link = chart_path.resolve()
            else:
                # Fallback for stdout or undefined report dir (try simple relative)
                chart_link = f"{chart_path.parent.name}/{chart_path.name}"

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
                f"\n*Generated by Multi-Agent Trading System ({mode_indicator}) - {self.timestamp}*\n"
            )
            return "".join(report_parts)

        # Full mode: include all sections
        # Helper function to add sections safely
        def add_section(key, title):
            raw_content = result.get(key, "")
            content = self._normalize_string(raw_content)

            if content and not content.startswith("Error"):
                report_parts.append(f"## {title}\n\n")
                # Clean content and strip redundant leading headers that match section title
                cleaned = self._clean_text(content, demote_headers=True)
                cleaned = self._strip_redundant_header(cleaned, title)
                report_parts.append(f"{cleaned}\n\n")

        add_section("market_report", "Technical Analysis")

        # Clean fundamentals: keep only final self-corrected DATA_BLOCK
        # Import inside function to prevent circular dependency with utils.py
        fund_report = result.get("fundamentals_report", "")
        if fund_report:
            try:
                from src.utils import clean_duplicate_data_blocks

                fund_report = self._normalize_string(fund_report)
                fund_report = clean_duplicate_data_blocks(fund_report)
                result["fundamentals_report"] = fund_report
            except ImportError:
                pass  # Fallback if utils not available

        add_section("fundamentals_report", "Fundamental Analysis")
        add_section("sentiment_report", "Market Sentiment")
        add_section("news_report", "News & Catalysts")
        add_section("investment_plan", "Investment Recommendation")

        # CRITICAL: Include consultant review if present (external cross-validation)
        consultant_review = result.get("consultant_review", "")
        if consultant_review and consultant_review.strip():
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

        add_section("trader_investment_plan", "Trading Strategy")

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
                    report_parts.append("## Risk Assessment\n\n")
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
            f"*Generated by Multi-Agent Trading System{mode_suffix} - {self.timestamp}*\n"
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

        # Remove agent prefixes if present
        text = re.sub(
            r"^(Bull Analyst:|Bear Analyst:|Risky Analyst:|Safe Analyst:|"
            r"Neutral Analyst:|Trader:|Portfolio Manager:)\s*",
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
