"""
Chart Generator Node for Post-PM Chart Generation.

This module creates a LangGraph node that generates charts AFTER the Portfolio Manager
has made its verdict. This ensures charts reflect the PM's risk-adjusted view rather
than raw fundamentals data.

Key Features:
- Uses PM_BLOCK for risk-adjusted scores when available
- Falls back to DATA_BLOCK when PM_BLOCK is missing
- Conditionally suppresses charts for SELL/DO_NOT_INITIATE verdicts
- Applies valuation discount to target ranges based on risk zone

Data Flow:
1. Inputs:
   - state["valuation_params"]: Raw math inputs from Valuation Agent (The 'Ingredients')
   - state["final_trade_decision"]: Verdict and PM_BLOCK from PM (The 'Recipe')
2. Output:
   - state["chart_paths"]: Filesystem paths to the rendered images.
"""

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from langgraph.types import RunnableConfig

from src.charts.base import CurrencyFormat

logger = structlog.get_logger(__name__)


# Exchange suffix to currency format mapping
# Handles both prefix currencies ($100) and suffix currencies (100 zł)
CURRENCY_MAP: dict[str, CurrencyFormat] = {
    # Asia-Pacific - Prefix currencies
    ".HK": CurrencyFormat("HK$", "prefix"),  # Hong Kong Dollar
    ".T": CurrencyFormat("¥", "prefix"),  # Japanese Yen
    ".TW": CurrencyFormat("NT$", "prefix"),  # Taiwan Dollar
    ".TWO": CurrencyFormat("NT$", "prefix"),  # Taiwan OTC
    ".KS": CurrencyFormat("₩", "prefix"),  # Korean Won (KOSPI)
    ".KQ": CurrencyFormat("₩", "prefix"),  # Korean Won (KOSDAQ)
    ".SS": CurrencyFormat("CN¥", "prefix"),  # Chinese Yuan (Shanghai)
    ".SZ": CurrencyFormat("CN¥", "prefix"),  # Chinese Yuan (Shenzhen)
    ".AX": CurrencyFormat("A$", "prefix"),  # Australian Dollar
    ".SI": CurrencyFormat("S$", "prefix"),  # Singapore Dollar
    ".BK": CurrencyFormat("฿", "prefix"),  # Thai Baht
    ".JK": CurrencyFormat("Rp", "prefix", space=True),  # Indonesian Rupiah
    ".KL": CurrencyFormat("RM", "prefix", space=True),  # Malaysian Ringgit
    ".NS": CurrencyFormat("₹", "prefix"),  # Indian Rupee (NSE)
    ".BO": CurrencyFormat("₹", "prefix"),  # Indian Rupee (BSE)
    # Europe - Prefix currencies
    ".L": CurrencyFormat("£", "prefix"),  # British Pound
    ".AS": CurrencyFormat("€", "prefix"),  # Euro (Amsterdam)
    ".PA": CurrencyFormat("€", "prefix"),  # Euro (Paris)
    ".DE": CurrencyFormat("€", "prefix"),  # Euro (Frankfurt/Xetra)
    ".F": CurrencyFormat("€", "prefix"),  # Euro (Frankfurt)
    ".MI": CurrencyFormat("€", "prefix"),  # Euro (Milan)
    ".MC": CurrencyFormat("€", "prefix"),  # Euro (Madrid)
    ".BR": CurrencyFormat("€", "prefix"),  # Euro (Brussels)
    ".LS": CurrencyFormat("€", "prefix"),  # Euro (Lisbon)
    ".VI": CurrencyFormat("€", "prefix"),  # Euro (Vienna)
    ".HE": CurrencyFormat("€", "prefix"),  # Euro (Helsinki)
    ".IR": CurrencyFormat("€", "prefix"),  # Euro (Dublin)
    ".AT": CurrencyFormat("€", "prefix"),  # Euro (Athens)
    ".SW": CurrencyFormat("CHF", "prefix", space=True),  # Swiss Franc
    # Europe - Suffix currencies
    ".ST": CurrencyFormat("kr", "suffix", space=True),  # Swedish Krona
    ".CO": CurrencyFormat("kr", "suffix", space=True),  # Danish Krone
    ".OL": CurrencyFormat("kr", "suffix", space=True),  # Norwegian Krone
    ".IC": CurrencyFormat("kr", "suffix", space=True),  # Icelandic Króna
    ".WA": CurrencyFormat("zł", "suffix", space=True),  # Polish Złoty
    ".PR": CurrencyFormat("Kč", "suffix", space=True),  # Czech Koruna
    ".BD": CurrencyFormat("Ft", "suffix", space=True),  # Hungarian Forint
    ".RO": CurrencyFormat("lei", "suffix", space=True),  # Romanian Leu
    # Americas
    ".TO": CurrencyFormat("C$", "prefix"),  # Canadian Dollar (Toronto)
    ".V": CurrencyFormat("C$", "prefix"),  # Canadian Dollar (TSX Venture)
    ".SA": CurrencyFormat("R$", "prefix", space=True),  # Brazilian Real
    ".MX": CurrencyFormat("MX$", "prefix"),  # Mexican Peso
    # Middle East & Africa
    ".TA": CurrencyFormat("₪", "prefix"),  # Israeli Shekel
    ".JO": CurrencyFormat("R", "prefix", space=True),  # South African Rand
}


def _get_currency_format(ticker: str) -> CurrencyFormat:
    """Derive currency format from ticker exchange suffix.

    Args:
        ticker: Stock ticker symbol (e.g., "0005.HK", "7203.T")

    Returns:
        CurrencyFormat for the exchange, defaults to USD ($) for
        US exchanges and unknown suffixes.

    Examples:
        >>> _get_currency_format("0005.HK").format_price(65.50)
        'HK$65.50'
        >>> _get_currency_format("7203.T").format_price(2850)
        '¥2850.00'
        >>> _get_currency_format("PKN.WA").format_price(42.50)
        '42.50 zł'
    """
    ticker_upper = ticker.upper()

    for suffix, currency in CURRENCY_MAP.items():
        if ticker_upper.endswith(suffix):
            return currency

    # Default to USD for US exchanges and ADRs
    return CurrencyFormat("$", "prefix")


def create_chart_generator_node(
    chart_format: str = "png",
    transparent: bool = False,
    image_dir: Path | None = None,
    skip_charts: bool = False,
    quick_mode: bool = False,
) -> Callable:
    """Create a chart generator node for the LangGraph workflow.

    This node runs after the Portfolio Manager and generates charts that
    reflect the PM's risk-adjusted view.

    Args:
        chart_format: Output format ('png' or 'svg')
        transparent: Whether to use transparent backgrounds
        image_dir: Directory for chart output (None = use config default)
        skip_charts: If True, skip all chart generation
        quick_mode: If True, skip chart generation (faster execution)

    Returns:
        Async function compatible with LangGraph StateGraph.add_node()
    """

    async def chart_generator_node(
        state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        """Generate charts based on PM verdict and analysis data.

        Args:
            state: LangGraph state containing analysis results
            config: LangGraph runnable config

        Returns:
            Dict with chart_paths (football_field, radar) or empty if skipped
        """
        # Early exit if charts disabled
        if skip_charts or quick_mode:
            logger.debug(
                "Chart generation skipped",
                skip_charts=skip_charts,
                quick_mode=quick_mode,
            )
            return {"chart_paths": {}}

        try:
            from src.charts.base import (
                ChartConfig,
                ChartFormat,
            )
            from src.charts.extractors.data_block import (
                extract_chart_data_from_data_block,
            )
            from src.charts.extractors.pm_block import extract_pm_block
            from src.config import config as app_config

            ticker = state.get("company_of_interest", "UNKNOWN")
            trade_date = datetime.now().strftime("%Y-%m-%d")

            # Extract PM_BLOCK for risk-adjusted view
            pm_output = _normalize_string(state.get("final_trade_decision", ""))
            pm_block = extract_pm_block(pm_output)

            # Extract DATA_BLOCK as fallback
            fundamentals_report = _normalize_string(
                state.get("fundamentals_report", "")
            )
            data_block = extract_chart_data_from_data_block(fundamentals_report)

            # Determine verdict (PM_BLOCK preferred, fallback to text parsing)
            verdict = pm_block.verdict
            if not verdict:
                verdict = _extract_verdict_fallback(pm_output)

            logger.info(
                "Chart generator processing",
                ticker=ticker,
                verdict=verdict,
                pm_block_found=pm_block.verdict is not None,
                show_valuation_chart=pm_block.show_valuation_chart,
            )

            # Configure chart output
            output_dir = image_dir if image_dir else app_config.images_dir
            chart_config = ChartConfig(
                output_dir=output_dir,
                format=ChartFormat.SVG if chart_format == "svg" else ChartFormat.PNG,
                transparent=transparent,
                filename_stem=ticker,
            )

            chart_paths = {}

            # --- Football Field Chart ---
            # Only generate if verdict is BUY or HOLD
            if pm_block.should_show_targets() or verdict in ("BUY", "HOLD", None):
                football_path = _generate_football_field(
                    state=state,
                    ticker=ticker,
                    trade_date=trade_date,
                    data_block=data_block,
                    pm_block=pm_block,
                    chart_config=chart_config,
                )
                if football_path:
                    chart_paths["football_field"] = str(football_path)
            else:
                logger.info(
                    "Football field chart suppressed for negative verdict",
                    ticker=ticker,
                    verdict=verdict,
                )

            # --- Radar Chart ---
            # Always generate radar (shows PM-adjusted scores, useful diagnostic)
            radar_path = _generate_radar_chart(
                state=state,
                ticker=ticker,
                trade_date=trade_date,
                data_block=data_block,
                pm_block=pm_block,
                chart_config=chart_config,
            )
            if radar_path:
                chart_paths["radar"] = str(radar_path)

            return {"chart_paths": chart_paths}

        except Exception as e:
            logger.error(f"Chart generation failed: {e}", exc_info=True)
            return {"chart_paths": {}}

    return chart_generator_node


def _normalize_string(content: Any) -> str:
    """Safely convert content to string, handling LangGraph list accumulation."""
    if content is None:
        return ""
    if isinstance(content, list):
        seen = set()
        unique_items = []
        for item in content:
            if not item:
                continue
            item_str = str(item).strip()
            key = item_str[:100]
            if key not in seen:
                seen.add(key)
                unique_items.append(item_str)
        return "\n\n".join(unique_items)
    return str(content)


def _extract_verdict_fallback(pm_output: str) -> str | None:
    """Fallback verdict extraction when PM_BLOCK is missing."""
    if not pm_output:
        return None

    patterns = [
        r"PORTFOLIO MANAGER VERDICT:\s*(BUY|HOLD|DO NOT INITIATE|SELL)",
        r"\*\*Action\*\*:\s*(BUY|HOLD|DO NOT INITIATE|SELL)",
    ]
    for pattern in patterns:
        match = re.search(pattern, pm_output, re.IGNORECASE)
        if match:
            return match.group(1).upper().replace(" ", "_")
    return None


def _generate_football_field(
    state: dict[str, Any],
    ticker: str,
    trade_date: str,
    data_block: Any,
    pm_block: Any,
    chart_config: Any,
) -> Path | None:
    """Generate football field chart with PM-adjusted targets."""
    from src.charts.base import FootballFieldData
    from src.charts.extractors.valuation import calculate_valuation_targets
    from src.charts.generators.football_field import generate_football_field

    # Check minimum data requirements
    if not data_block.current_price or not data_block.fifty_two_week_high:
        logger.debug("Insufficient data for football field chart", ticker=ticker)
        return None

    # Extract valuation targets from Valuation Calculator
    valuation_params = _normalize_string(state.get("valuation_params", ""))
    targets = calculate_valuation_targets(valuation_params)

    # Apply PM's valuation discount
    our_target_low = targets.low
    our_target_high = targets.high

    if pm_block.valuation_discount and pm_block.valuation_discount > 0:
        if our_target_low:
            our_target_low = round(our_target_low * pm_block.valuation_discount, 2)
        if our_target_high:
            our_target_high = round(our_target_high * pm_block.valuation_discount, 2)
        logger.debug(
            "Applied valuation discount to targets",
            discount=pm_block.valuation_discount,
            original_low=targets.low,
            adjusted_low=our_target_low,
        )
    elif pm_block.valuation_discount == 0:
        # Suppress "Our Target" for negative verdicts
        our_target_low = None
        our_target_high = None

    # Extract quality warnings from red flags
    quality_warnings = []
    red_flags = state.get("red_flags", [])
    for flag in red_flags:
        severity = str(flag.get("severity", "")).upper()
        if severity in ["CRITICAL", "WARNING"]:
            detail = str(flag.get("detail", ""))
            quality_warnings.append(detail[:50] + "..." if len(detail) > 50 else detail)
    quality_warnings = quality_warnings[:2]  # Limit to 2

    # Build footnote with methodology and PM adjustment note
    footnote_parts = []
    if targets.methodology:
        footnote_parts.append(targets.methodology)
    if pm_block.valuation_discount and pm_block.valuation_discount < 1.0:
        footnote_parts.append(
            f"Risk-adjusted ({pm_block.zone or 'N/A'} zone, {pm_block.valuation_discount:.0%} discount)"
        )
    if data_block.analyst_coverage and data_block.analyst_coverage > 0:
        # This is the English/Refinitiv count — labels it "analysts" not "English analysts"
        # because that's what the consensus targets are actually based on.
        footnote_parts.append(f"Consensus: {data_block.analyst_coverage} analysts")

    # Get currency format from ticker exchange suffix
    currency_format = _get_currency_format(ticker)

    football_data = FootballFieldData(
        ticker=ticker,
        trade_date=trade_date,
        current_price=data_block.current_price,
        fifty_two_week_high=data_block.fifty_two_week_high,
        fifty_two_week_low=data_block.fifty_two_week_low,
        currency_format=currency_format,
        moving_avg_50=data_block.moving_avg_50,
        moving_avg_200=data_block.moving_avg_200,
        external_target_high=data_block.external_target_high,
        external_target_low=data_block.external_target_low,
        external_target_mean=data_block.external_target_mean,
        our_target_low=our_target_low,
        our_target_high=our_target_high,
        target_methodology=targets.methodology,
        target_confidence=targets.confidence,
        quality_warnings=quality_warnings if quality_warnings else None,
        footnote=" | ".join(footnote_parts) if footnote_parts else None,
    )

    return generate_football_field(football_data, chart_config)


def _generate_radar_chart(
    state: dict[str, Any],
    ticker: str,
    trade_date: str,
    data_block: Any,
    pm_block: Any,
    chart_config: Any,
) -> Path | None:
    """Generate radar chart with PM-adjusted scores when available."""
    import re

    from src.charts.base import RadarChartData
    from src.charts.generators.radar_chart import generate_radar_chart

    # Use PM-adjusted scores when available, fallback to DATA_BLOCK
    health = (
        pm_block.health_adj
        if pm_block.health_adj is not None
        else data_block.adjusted_health_score
    )
    growth = (
        pm_block.growth_adj
        if pm_block.growth_adj is not None
        else data_block.adjusted_growth_score
    )

    # Need at least health score
    if health is None:
        logger.debug("Insufficient data for radar chart (no health score)")
        return None

    # Apply D/E and ROA adjustments if using raw scores
    if pm_block.health_adj is None and data_block.adjusted_health_score is not None:
        health = data_block.adjusted_health_score
        if data_block.de_ratio is not None:
            if data_block.de_ratio < 0.5:
                health = min(100.0, health + 5)
            elif data_block.de_ratio > 2.0:
                health = max(0.0, health - 10)
            elif data_block.de_ratio > 1.0:
                health = max(0.0, health - 5)
        if data_block.roa is not None:
            if data_block.roa > 10.0:
                health = min(100.0, health + 5)
            elif data_block.roa < 3.0:
                health = max(0.0, health - 5)

    health = max(0.0, min(100.0, float(health)))
    growth = max(0.0, min(100.0, float(growth if growth is not None else 50.0)))

    # Valuation score (from P/E and PEG)
    pe_score = None
    peg_score = None
    if data_block.pe_ratio_ttm and data_block.pe_ratio_ttm > 0:
        pe_score = max(0.0, min(100.0, (25.0 - data_block.pe_ratio_ttm) * 10.0))
    if data_block.peg_ratio and data_block.peg_ratio > 0:
        peg_score = max(0.0, min(100.0, (2.0 - data_block.peg_ratio) * 100.0))

    if pe_score is not None and peg_score is not None:
        val_score = (pe_score * 0.4) + (peg_score * 0.6)
    elif pe_score is not None:
        val_score = pe_score
    elif peg_score is not None:
        val_score = peg_score
    else:
        val_score = 50.0

    val_score = max(0.0, min(100.0, val_score))

    # Undiscovered score (low analyst coverage = high score).
    # analyst_coverage is from ANALYST_COVERAGE_ENGLISH (Refinitiv/FactSet), which
    # undercounts for ex-US stocks — but that's correct here: the "undiscovered"
    # thesis specifically measures visibility to English-language research.
    # If missing (None/0), default to neutral (10) rather than assuming undiscovered,
    # since absence of data ≠ absence of coverage.
    coverage = (
        data_block.analyst_coverage
        if data_block.analyst_coverage is not None and data_block.analyst_coverage > 0
        else 10
    )
    undiscovered = max(0.0, min(100.0, (15.0 - coverage) * 10.0))

    # Regulatory score (PFIC, VIE, CMIC, ADR penalties)
    regulatory = 100.0
    if data_block.pfic_risk:
        risk_upper = data_block.pfic_risk.upper()
        if "HIGH" in risk_upper:
            regulatory -= 40
        elif "MEDIUM" in risk_upper:
            regulatory -= 20
    if data_block.vie_structure is True:
        regulatory -= 25
    if data_block.cmic_flagged is True:
        regulatory -= 35
    if data_block.adr_impact:
        if "MODERATE_CONCERN" in data_block.adr_impact.upper():
            regulatory -= 10
    regulatory = max(0.0, min(100.0, regulatory))

    # Jurisdiction score
    jurisdiction = 100.0
    ticker_upper = ticker.upper()
    if any(suffix in ticker_upper for suffix in [".SS", ".SZ", ".HK"]):
        jurisdiction -= 25
    elif any(suffix in ticker_upper for suffix in [".KS", ".KQ"]):
        jurisdiction -= 10

    if (
        data_block.us_revenue_percent
        and "Not disclosed" not in data_block.us_revenue_percent
    ):
        try:
            rev_match = re.search(r"([\d.]+)%", data_block.us_revenue_percent)
            if rev_match:
                rev = float(rev_match.group(1))
                if rev > 35.0:
                    jurisdiction -= 30
                elif rev > 25.0:
                    jurisdiction -= 15
        except Exception:
            pass
    jurisdiction = max(0.0, min(100.0, jurisdiction))

    # Data quality warnings — map red flag types to affected radar axes
    axis_warnings = {}
    red_flags = state.get("red_flags", [])
    for flag in red_flags:
        flag_type = str(flag.get("type", "")).upper()
        # Health axis: earnings quality, cash flow, leverage, consultant flags
        if any(
            x in flag_type
            for x in [
                "EARNINGS",
                "CASH",
                "FCF",
                "OCF",
                "SUSPICIOUS",
                "CONSULTANT",
                "LEVERAGE",
                "REFINANCING",
                "UNSUSTAINABLE",
            ]
        ):
            axis_warnings["health"] = True
        # Growth axis: segment deterioration, value traps, no catalyst
        if any(
            x in flag_type
            for x in [
                "SEGMENT",
                "VALUE_TRAP",
                "CATALYST",
            ]
        ):
            axis_warnings["growth"] = True
        # Valuation axis: cyclical peaks, unreliable ratios, thin consensus
        if any(
            x in flag_type
            for x in [
                "CYCLICAL",
                "PEG",
                "FRAGILE_VALUATION",
                "THIN_CONSENSUS",
            ]
        ):
            axis_warnings["valuation"] = True
        # Regulatory axis: PFIC, VIE, CMIC, ADR flags
        if any(x in flag_type for x in ["PFIC", "VIE", "ADR", "CMIC"]):
            axis_warnings["regulatory"] = True

    # Add PM verdict to footnote
    footnote_parts = []
    if pm_block.verdict:
        footnote_parts.append(f"PM Verdict: {pm_block.verdict}")
    if pm_block.risk_tally is not None:
        footnote_parts.append(f"Risk: {pm_block.risk_tally:.2f}")
    if axis_warnings:
        footnote_parts.append("* Data quality flag")
    footnote = " | ".join(footnote_parts) if footnote_parts else None

    radar_data = RadarChartData(
        ticker=ticker,
        trade_date=trade_date,
        health_score=health,
        growth_score=growth,
        valuation_score=val_score,
        undiscovered_score=undiscovered,
        regulatory_score=regulatory,
        jurisdiction_score=jurisdiction,
        pe_ratio=data_block.pe_ratio_ttm,
        peg_ratio=data_block.peg_ratio,
        de_ratio=data_block.de_ratio,
        roa=data_block.roa,
        analyst_count=data_block.analyst_coverage,
        axis_warnings=axis_warnings,
        footnote=footnote,
    )

    return generate_radar_chart(radar_data, chart_config)
