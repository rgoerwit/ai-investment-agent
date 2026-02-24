"""
Lessons Learned / Retrospective System

Compares past analysis verdicts to actual market outcomes and generates
generalizable lessons for future analyses. Prediction snapshots are auto-saved
with every analysis; retrospective comparison runs automatically for the current
ticker on re-analysis (skipped with --no-memory).

Design principles:
- Deterministic where possible ($0 cost for snapshot extraction, comparison, confidence)
- One cheap Gemini Flash LLM call per significant delta (~$0.001)
- Early dedup: already-processed snapshots are skipped via ChromaDB metadata query (~50ms)
- Global lesson storage (cross-ticker, cross-sector) with geographic boost at retrieval
- Graceful degradation: failures never block analysis
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.config import config

logger = structlog.get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

EXCHANGE_BENCHMARK: dict[str, str] = {
    ".T": "^N225",
    ".HK": "^HSI",
    ".TW": "^TWII",
    ".KS": "^KS11",
    ".AS": "^AEX",
    ".DE": "^GDAXI",
    ".L": "^FTSE",
    ".PA": "^FCHI",
    ".TO": "^GSPTSE",
    ".AX": "^AXJO",
    ".SI": "^STI",
    ".MI": "^FTSEMIB",
    ".ST": "^OMX",
}
FALLBACK_BENCHMARK = "^GSPC"

EXCHANGE_CURRENCY: dict[str, str] = {
    ".T": "JPY",
    ".HK": "HKD",
    ".TW": "TWD",
    ".KS": "KRW",
    ".AS": "EUR",
    ".DE": "EUR",
    ".L": "GBP",
    ".PA": "EUR",
    ".TO": "CAD",
    ".AX": "AUD",
    ".SI": "SGD",
    ".MI": "EUR",
    ".ST": "SEK",
}
FALLBACK_CURRENCY = "USD"

MODEL_QUALITY: dict[str, float] = {
    "gemini-3-pro-preview": 1.0,
    "gemini-3-pro": 1.0,
    "gemini-2.5-pro": 0.9,
    "gemini-2.5-flash": 0.7,
    "gemini-2.0-flash": 0.6,
}
DEFAULT_MODEL_QUALITY = 0.5

# Temporal confidence decay curve
TEMPORAL_WEIGHTS: list[tuple[int, float]] = [
    (30, 0.3),  # 0-30 days: too early
    (90, 0.7),  # 30-90 days: early signal
    (270, 1.0),  # 90-270 days: optimal
    (540, 0.7),  # 270-540 days: degrading
]
TEMPORAL_STALE = 0.3  # 540+ days

# Lesson-trigger thresholds (excess return vs benchmark, stored as positive values)
# For BUY: wrong if excess < -15%, understated if excess > +40%
# For SELL/DNI: wrong if excess > +25%, understated if excess < -30%
# For HOLD: wrong if |excess| > 25%
THRESHOLDS: dict[str, dict[str, float]] = {
    "BUY": {"wrong": 15.0, "understated": 40.0},
    "HOLD": {"wrong": 25.0, "understated": 25.0},
    "DO_NOT_INITIATE": {"wrong": 25.0, "understated": 30.0},
    "SELL": {"wrong": 25.0, "understated": 30.0},
}

MINIMUM_DAYS_ELAPSED = 30
MAX_LESSONS_PER_TICKER = 3

FAILURE_MODES = {
    "CYCLICAL_PEAK",
    "FX_DRIVEN",
    "GOVERNANCE_BLEED",
    "OPERATIONAL_MISS",
    "REGULATORY_SHIFT",
    "MACRO_REGIME",
    "DISRUPTION",
    "VALUATION_TRAP",
    "ACCOUNTING_FRAUD",
    "GEOPOLITICAL",
    "LIQUIDITY_CRISIS",
    "DEAD_MONEY",
}

LESSON_TYPES = {"missed_risk", "false_positive", "missed_opportunity", "correct_call"}

LESSONS_COLLECTION_NAME = "lessons_learned"


# ══════════════════════════════════════════════════════════════════════════════
# Component 1: Prediction Snapshot Extraction
# ══════════════════════════════════════════════════════════════════════════════


def _get_ticker_suffix(ticker: str) -> str:
    """Extract exchange suffix from ticker (e.g., '.T' from '7203.T')."""
    dot_idx = ticker.rfind(".")
    if dot_idx >= 0:
        return ticker[dot_idx:]
    return ""


def _extract_bear_risks(result: dict) -> str:
    """Extract first ~500 chars of bear thesis key risks from debate history."""
    debate = result.get("investment_debate_state", {})
    bear_history = debate.get("bear_history", "") or ""
    if not bear_history:
        # Try round-specific fields
        bear_history = debate.get("bear_round1", "") or ""

    if not bear_history:
        return ""

    # Try to find a KEY RISKS or FAILURE MODE section
    for pattern in [
        r"(?:KEY RISKS|FAILURE MODE|KILL CRITERIA|BEAR CASE).*?(?=\n\n|\Z)",
        r"(?:risk|bear|downside).*",
    ]:
        match = re.search(pattern, bear_history, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0)[:500]

    # Fallback: first 500 chars of bear history
    return bear_history[:500]


def _extract_data_block_field(fundamentals_report: str, field_name: str) -> str | None:
    """Extract a single field from the last DATA_BLOCK in fundamentals report."""
    if not fundamentals_report:
        return None

    data_block_pattern = (
        r"### --- START DATA_BLOCK[^\n]*---(.+?)### --- END DATA_BLOCK ---"
    )
    blocks = list(re.finditer(data_block_pattern, fundamentals_report, re.DOTALL))
    if not blocks:
        return None

    data_block = blocks[-1].group(1)
    match = re.search(rf"{field_name}:\s*(.+?)(?:\n|$)", data_block, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        if value.upper() in ("N/A", "NA", "NONE", "-", ""):
            return None
        return value
    return None


def _extract_data_block_float(
    fundamentals_report: str, field_name: str
) -> float | None:
    """Extract a float field from the last DATA_BLOCK."""
    raw = _extract_data_block_field(fundamentals_report, field_name)
    if raw is None:
        return None
    # Strip trailing % or other text
    cleaned = re.match(r"[-+]?[\d.]+", raw)
    if cleaned:
        try:
            return float(cleaned.group(0))
        except ValueError:
            return None
    return None


def _extract_trade_block_price(text: str, field: str) -> float | None:
    """Extract a price from a TRADE_BLOCK field like 'ENTRY: 2,145 (Scaled Limit)'."""
    pattern = rf"{field}:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.upper().startswith("N/A"):
        return None
    price_match = re.match(r"([\d,]+(?:\.\d+)?)", raw)
    if price_match:
        try:
            return float(price_match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _extract_trade_block_text(trader_plan: str, field: str) -> str | None:
    """Extract a text value from a TRADE_BLOCK field (non-numeric)."""
    pattern = rf"{field}:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, trader_plan, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip()
    return raw if raw and raw.upper() not in ("N/A", "NA", "-", "") else None


def _extract_trade_block_fields(trader_plan: str) -> dict[str, Any]:
    """
    Extract structured TRADE_BLOCK fields from trader output.

    Zero LLM cost — pure regex. Backward-compatible: older JSONs
    without these fields will have None values (handled by reconciler).

    Returns dict with keys: entry_price, stop_price, target_1_price,
    target_2_price, conviction, investment_horizon.
    """
    if not trader_plan:
        return {
            "entry_price": None,
            "stop_price": None,
            "target_1_price": None,
            "target_2_price": None,
            "conviction": None,
            "investment_horizon": None,
        }

    conviction_match = re.search(r"CONVICTION:\s*(\w+)", trader_plan, re.IGNORECASE)

    return {
        "entry_price": _extract_trade_block_price(trader_plan, "ENTRY"),
        "stop_price": _extract_trade_block_price(trader_plan, "STOP"),
        "target_1_price": _extract_trade_block_price(trader_plan, "TARGET_1"),
        "target_2_price": _extract_trade_block_price(trader_plan, "TARGET_2"),
        "conviction": conviction_match.group(1) if conviction_match else None,
        "investment_horizon": _extract_trade_block_text(trader_plan, "HORIZON"),
    }


def extract_snapshot(
    result: dict, ticker: str, is_quick_mode: bool = False
) -> dict[str, Any]:
    """
    Extract a compact prediction snapshot from an analysis result.

    This is called from save_results_to_file() on every analysis.
    Zero LLM cost — pure regex and dict construction.

    Args:
        result: The full analysis result dict (from graph.ainvoke)
        ticker: Stock ticker symbol

    Returns:
        Compact dict (~20 fields) suitable for JSON serialization
    """
    from src.charts.extractors.pm_block import (
        extract_pm_block,
        extract_verdict_from_text,
    )

    # PM_BLOCK extraction (reuse existing extractor)
    pm_output = result.get("final_trade_decision", "") or ""
    pm_data = extract_pm_block(pm_output)

    # Fallback verdict extraction
    verdict = pm_data.verdict
    if not verdict:
        verdict = extract_verdict_from_text(pm_output)

    # DATA_BLOCK extraction from fundamentals report
    fundamentals = result.get("fundamentals_report", "") or ""

    # Exchange/currency/benchmark mapping
    suffix = _get_ticker_suffix(ticker)
    currency = EXCHANGE_CURRENCY.get(suffix, FALLBACK_CURRENCY)
    benchmark = EXCHANGE_BENCHMARK.get(suffix, FALLBACK_BENCHMARK)

    # FX rate at analysis time (synchronous fallback only — no async in snapshot)
    fx_rate = None
    try:
        from src.fx_normalization import FALLBACK_RATES_TO_USD

        fx_rate = FALLBACK_RATES_TO_USD.get(currency, 1.0)
    except ImportError:
        fx_rate = 1.0

    # TRADE_BLOCK extraction from trader plan (zero LLM cost — pure regex)
    trader_plan = result.get("investment_analysis", {}).get("trader_plan", "") or ""
    trade_block_fields = _extract_trade_block_fields(trader_plan)

    snapshot = {
        # Core verdict
        "verdict": verdict,
        "health_adj": pm_data.health_adj,
        "growth_adj": pm_data.growth_adj,
        "risk_tally": pm_data.risk_tally,
        "zone": pm_data.zone,
        "position_size": pm_data.position_size,
        # DATA_BLOCK fields
        "current_price": _extract_data_block_float(fundamentals, "CURRENT_PRICE"),
        "sector": _extract_data_block_field(fundamentals, "SECTOR"),
        "pe_ratio": _extract_data_block_float(fundamentals, "PE_RATIO_TTM"),
        "peg_ratio": _extract_data_block_float(fundamentals, "PEG_RATIO"),
        "pb_ratio": _extract_data_block_float(fundamentals, "PB_RATIO"),
        # ENGLISH = Refinitiv/FactSet (global aggregator, English-bias). TOTAL_EST =
        # Senior's synthesis including FLA local-language estimates (may be int or tier).
        "analyst_coverage": _extract_data_block_float(
            fundamentals, "ANALYST_COVERAGE_ENGLISH"
        ),
        "analyst_coverage_total_est": _extract_data_block_field(
            fundamentals, "ANALYST_COVERAGE_TOTAL_EST"
        ),
        "profitability_trend": _extract_data_block_field(
            fundamentals, "PROFITABILITY_TREND"
        ),
        "52w_high": _extract_data_block_float(fundamentals, "52W_HIGH"),
        "52w_low": _extract_data_block_float(fundamentals, "52W_LOW"),
        # TRADE_BLOCK fields (structured for portfolio reconciliation)
        **trade_block_fields,
        # Bear thesis excerpt
        "bear_risks_excerpt": _extract_bear_risks(result),
        # Exchange/currency/benchmark
        "exchange": suffix.lstrip(".") if suffix else "US",
        "currency": currency,
        "benchmark_index": benchmark,
        "fx_rate_to_usd": fx_rate,
        # Metadata (from existing save_data structure)
        "ticker": ticker,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "deep_model": config.deep_think_llm,
        "quick_model": config.quick_think_llm,
        "is_quick_mode": is_quick_mode,
    }

    logger.info(
        "prediction_snapshot_extracted",
        ticker=ticker,
        verdict=snapshot["verdict"],
        price=snapshot["current_price"],
        sector=snapshot["sector"],
    )

    return snapshot


# ══════════════════════════════════════════════════════════════════════════════
# Component 2: Retrospective Comparison
# ══════════════════════════════════════════════════════════════════════════════


def load_past_snapshots(
    ticker: str | None, results_dir: Path
) -> dict[str, list[dict[str, Any]]]:
    """
    Load prediction snapshots from saved analysis JSON files.

    Args:
        ticker: If provided, only load snapshots for this ticker.
                If None, load all tickers found.
        results_dir: Directory containing analysis JSON files.

    Returns:
        Dict mapping ticker -> list of snapshots (sorted by date descending)
    """
    snapshots: dict[str, list[dict[str, Any]]] = {}

    if not results_dir.exists():
        logger.warning("results_dir_not_found", path=str(results_dir))
        return snapshots

    # Build pattern based on ticker filter
    if ticker:
        safe_ticker = ticker.replace(".", "_").replace("/", "_")
        pattern = f"{safe_ticker}_*_analysis.json"
        # Also try with dot notation for older files
        pattern2 = f"{ticker}_*_analysis.json"
    else:
        pattern = "*_analysis.json"
        pattern2 = None

    files = sorted(results_dir.glob(pattern), reverse=True)
    if pattern2:
        files.extend(sorted(results_dir.glob(pattern2), reverse=True))
    # Deduplicate
    seen = set()
    unique_files = []
    for f in files:
        if f.name not in seen:
            seen.add(f.name)
            unique_files.append(f)
    files = unique_files

    for filepath in files:
        try:
            with open(filepath) as f:
                data = json.load(f)

            snapshot = data.get("prediction_snapshot")
            if not snapshot:
                logger.debug(
                    "no_snapshot_in_file",
                    file=filepath.name,
                    reason="predates retrospective feature",
                )
                continue

            snap_ticker = snapshot.get("ticker", "UNKNOWN")
            if snap_ticker not in snapshots:
                snapshots[snap_ticker] = []
            # Attach source file for deduplication
            snapshot["_source_file"] = filepath.name
            snapshots[snap_ticker].append(snapshot)

        except json.JSONDecodeError:
            logger.warning("malformed_json", file=filepath.name)
        except Exception as e:
            logger.warning("snapshot_load_error", file=filepath.name, error=str(e))

    return snapshots


async def compare_to_reality(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """
    Compare a past prediction snapshot to current market reality.

    Fetches current price + benchmark return via yfinance. Computes excess
    return and determines if the delta exceeds lesson-trigger thresholds.

    Args:
        snapshot: Prediction snapshot dict from extract_snapshot()

    Returns:
        Comparison dict if threshold exceeded, None otherwise.
        Also returns None if data fetch fails or elapsed days < 30.
    """
    import asyncio

    ticker = snapshot.get("ticker")
    analysis_date_str = snapshot.get("analysis_date")
    verdict = snapshot.get("verdict")
    snapshot_price = snapshot.get("current_price")

    if not ticker or not analysis_date_str or not verdict:
        logger.debug("incomplete_snapshot", ticker=ticker)
        return None

    logger.debug(
        "comparison_starting",
        ticker=ticker,
        analysis_date=analysis_date_str,
        verdict=verdict,
        snapshot_price=snapshot_price,
    )

    # Parse analysis date
    try:
        analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")
    except ValueError:
        logger.debug("invalid_date", date=analysis_date_str)
        return None

    days_elapsed = (datetime.now() - analysis_date).days
    if days_elapsed < MINIMUM_DAYS_ELAPSED:
        logger.debug(
            "too_recent", ticker=ticker, days=days_elapsed, min=MINIMUM_DAYS_ELAPSED
        )
        return None

    # Fetch current price and benchmark via yfinance
    try:
        import yfinance as yf

        def _fetch_current_data():
            result = {}

            # Current stock price (adjusted close for total return)
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=analysis_date.strftime("%Y-%m-%d"),
                    end=datetime.now().strftime("%Y-%m-%d"),
                )
                if len(hist) >= 2:
                    result["start_adj_close"] = float(hist["Close"].iloc[0])
                    result["end_adj_close"] = float(hist["Close"].iloc[-1])
                else:
                    # Fallback: use info
                    info = stock.info
                    current = info.get("currentPrice") or info.get("regularMarketPrice")
                    if current:
                        result["end_adj_close"] = float(current)
                        result["start_adj_close"] = (
                            float(snapshot_price) if snapshot_price else None
                        )
            except Exception as e:
                logger.debug("stock_fetch_failed", ticker=ticker, error=str(e))

            # Benchmark return over same period
            benchmark = snapshot.get("benchmark_index", FALLBACK_BENCHMARK)
            try:
                bench = yf.Ticker(benchmark)
                bench_hist = bench.history(
                    start=analysis_date.strftime("%Y-%m-%d"),
                    end=datetime.now().strftime("%Y-%m-%d"),
                )
                if len(bench_hist) >= 2:
                    result["bench_start"] = float(bench_hist["Close"].iloc[0])
                    result["bench_end"] = float(bench_hist["Close"].iloc[-1])
            except Exception as e:
                logger.debug(
                    "benchmark_fetch_failed", benchmark=benchmark, error=str(e)
                )
                # Fallback to S&P 500 if primary benchmark fails
                if benchmark != FALLBACK_BENCHMARK:
                    try:
                        bench = yf.Ticker(FALLBACK_BENCHMARK)
                        bench_hist = bench.history(
                            start=analysis_date.strftime("%Y-%m-%d"),
                            end=datetime.now().strftime("%Y-%m-%d"),
                        )
                        if len(bench_hist) >= 2:
                            result["bench_start"] = float(bench_hist["Close"].iloc[0])
                            result["bench_end"] = float(bench_hist["Close"].iloc[-1])
                            result["benchmark_fallback"] = FALLBACK_BENCHMARK
                    except Exception:
                        pass

            return result

        data = await asyncio.wait_for(
            asyncio.to_thread(_fetch_current_data),
            timeout=15.0,
        )

    except asyncio.TimeoutError:
        logger.warning("yfinance_timeout", ticker=ticker)
        return None
    except Exception as e:
        logger.warning("yfinance_error", ticker=ticker, error=str(e))
        return None

    # Calculate returns
    start_price = data.get("start_adj_close")
    end_price = data.get("end_adj_close")
    if not start_price or not end_price or start_price <= 0:
        logger.debug("insufficient_price_data", ticker=ticker)
        return None

    price_return_pct = ((end_price - start_price) / start_price) * 100.0

    bench_start = data.get("bench_start")
    bench_end = data.get("bench_end")
    benchmark_return_pct = 0.0
    if bench_start and bench_end and bench_start > 0:
        benchmark_return_pct = ((bench_end - bench_start) / bench_start) * 100.0

    excess_return_pct = price_return_pct - benchmark_return_pct

    # FX delta
    fx_delta_pct = 0.0
    snapshot_fx = snapshot.get("fx_rate_to_usd")
    currency = snapshot.get("currency", FALLBACK_CURRENCY)
    if snapshot_fx and currency != "USD":
        try:
            from src.fx_normalization import get_fx_rate_yfinance

            current_fx = await get_fx_rate_yfinance(currency, "USD")
            if current_fx and snapshot_fx > 0:
                fx_delta_pct = ((current_fx - snapshot_fx) / snapshot_fx) * 100.0
        except Exception:
            pass  # FX delta is informational, not critical

    # Check thresholds
    thresholds = THRESHOLDS.get(verdict, THRESHOLDS.get("HOLD", {}))
    wrong_threshold = thresholds.get("wrong", 25.0)
    understated_threshold = thresholds.get("understated", 40.0)

    triggered = False
    if verdict in ("BUY",):
        # Wrong direction: big loss
        if excess_return_pct < -wrong_threshold:
            triggered = True
        # Right but understated: huge gain
        elif excess_return_pct > understated_threshold:
            triggered = True
    elif verdict in ("DO_NOT_INITIATE", "SELL"):
        # Wrong direction: stock went up a lot
        if excess_return_pct > wrong_threshold:
            triggered = True
        # Right but understated: crashed even more
        elif excess_return_pct < -understated_threshold:
            triggered = True
    elif verdict in ("HOLD",):
        if abs(excess_return_pct) > wrong_threshold:
            triggered = True

    if not triggered:
        logger.debug(
            "below_threshold",
            ticker=ticker,
            excess_return=f"{excess_return_pct:.1f}%",
            verdict=verdict,
        )
        return None

    comparison = {
        **snapshot,
        "price_return_pct": round(price_return_pct, 2),
        "benchmark_return_pct": round(benchmark_return_pct, 2),
        "excess_return_pct": round(excess_return_pct, 2),
        "fx_delta_pct": round(fx_delta_pct, 2),
        "days_elapsed": days_elapsed,
        "start_price": round(start_price, 4),
        "end_price": round(end_price, 4),
        "benchmark_used": data.get(
            "benchmark_fallback", snapshot.get("benchmark_index")
        ),
    }

    logger.info(
        "significant_delta_detected",
        ticker=ticker,
        verdict=verdict,
        excess_return=f"{excess_return_pct:.1f}%",
        days=days_elapsed,
    )

    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# Component 3: Confidence Weighting
# ══════════════════════════════════════════════════════════════════════════════


def compute_confidence(comparison: dict[str, Any]) -> float:
    """
    Compute composite confidence score for a lesson.

    confidence = temporal × model_quality × mode × signal_strength

    Args:
        comparison: Comparison dict from compare_to_reality()

    Returns:
        Float between 0.0 and 1.0
    """
    days = comparison.get("days_elapsed", 0)

    # Temporal component
    temporal = TEMPORAL_STALE
    for max_days, weight in TEMPORAL_WEIGHTS:
        if days <= max_days:
            temporal = weight
            break

    # Model quality component
    deep_model = comparison.get("deep_model", "")
    model_q = MODEL_QUALITY.get(deep_model, DEFAULT_MODEL_QUALITY)

    # Analysis mode: prefer explicit flag (modern snapshots); fall back to
    # model-name heuristic for snapshots predating is_quick_mode field.
    if "is_quick_mode" in comparison:
        mode = 0.7 if comparison["is_quick_mode"] else 1.0
    else:
        quick_model = comparison.get("quick_model", "")
        deep = comparison.get("deep_model", "")
        mode = 0.7 if quick_model == deep else 1.0

    # Signal strength component (bigger deltas = clearer lessons)
    excess = abs(comparison.get("excess_return_pct", 0.0))
    signal = min(excess / 30.0, 1.0)

    confidence = temporal * model_q * mode * signal
    final = round(min(max(confidence, 0.0), 1.0), 3)
    logger.debug(
        "confidence_computed",
        ticker=comparison.get("ticker"),
        confidence=final,
        temporal=round(temporal, 2),
        model_q=round(model_q, 2),
        mode_factor=round(mode, 2),
        signal=round(signal, 2),
    )
    return final


# ══════════════════════════════════════════════════════════════════════════════
# Component 4: Lesson Generation (single LLM call)
# ══════════════════════════════════════════════════════════════════════════════


async def generate_lesson(
    comparison: dict[str, Any],
) -> tuple[str, str, str] | None:
    """
    Generate a generalizable lesson from a significant prediction delta.

    One Gemini Flash call with a compact prompt (~300 input tokens).
    Returns (lesson_text, lesson_type, failure_mode) or None on failure.
    """
    prompt = f"""Given this past equity analysis and its actual outcome, generate ONE generalizable lesson.

ANALYSIS ({comparison.get('analysis_date', 'unknown')}):
Ticker: {comparison.get('ticker')} | Sector: {comparison.get('sector', 'Unknown')} | Exchange: {comparison.get('exchange', 'Unknown')} | Currency: {comparison.get('currency', 'USD')}
Verdict: {comparison.get('verdict')} (Position: {comparison.get('position_size', 'N/A')}%) | Zone: {comparison.get('zone', 'N/A')}
Health: {comparison.get('health_adj', 'N/A')} | Growth: {comparison.get('growth_adj', 'N/A')} | P/E: {comparison.get('pe_ratio', 'N/A')} | PEG: {comparison.get('peg_ratio', 'N/A')}
Targets: Entry {comparison.get('entry_price') or 'N/A'} | T1 {comparison.get('target_1_price') or 'N/A'} | T2 {comparison.get('target_2_price') or 'N/A'} | Stop {comparison.get('stop_price') or 'N/A'} | Horizon: {comparison.get('investment_horizon') or 'N/A'}
Key bear risks: {comparison.get('bear_risks_excerpt', 'N/A')[:300]}

OUTCOME ({comparison.get('days_elapsed', 0)} days later):
Price: {comparison.get('start_price', 'N/A')} → {comparison.get('end_price', 'N/A')} ({comparison.get('price_return_pct', 0):+.1f}%)
Benchmark ({comparison.get('benchmark_used', 'N/A')}): {comparison.get('benchmark_return_pct', 0):+.1f}%
Excess return: {comparison.get('excess_return_pct', 0):+.1f}%
FX ({comparison.get('currency', 'USD')}/USD): {comparison.get('fx_delta_pct', 0):+.1f}%

Rules:
- Lesson must be GENERAL (applicable to similar stocks), not specific to this ticker
- One sentence, max 40 words
- Focus on what the analysis missed or over/under-weighted

LESSON: [your lesson]
TYPE: missed_risk | false_positive | missed_opportunity | correct_call
FAILURE_MODE: CYCLICAL_PEAK | FX_DRIVEN | GOVERNANCE_BLEED | OPERATIONAL_MISS | REGULATORY_SHIFT | MACRO_REGIME | DISRUPTION | VALUATION_TRAP | ACCOUNTING_FRAUD | GEOPOLITICAL | LIQUIDITY_CRISIS | DEAD_MONEY"""

    try:
        from src.llms import create_quick_thinking_llm

        llm = create_quick_thinking_llm()
        from langchain_core.messages import HumanMessage

        response = await llm.ainvoke([HumanMessage(content=prompt)])

        from src.agents import extract_string_content

        content = extract_string_content(response.content).strip()

        # Parse response
        lesson_match = re.search(r"LESSON:\s*(.+?)(?:\n|$)", content)
        type_match = re.search(r"TYPE:\s*(\S+)", content)
        mode_match = re.search(r"FAILURE_MODE:\s*(\S+)", content)

        lesson_text = lesson_match.group(1).strip() if lesson_match else content[:200]
        lesson_type = (
            type_match.group(1).strip().lower() if type_match else "missed_risk"
        )
        failure_mode = (
            mode_match.group(1).strip().upper() if mode_match else "OPERATIONAL_MISS"
        )

        # Validate against known enums
        if lesson_type not in LESSON_TYPES:
            lesson_type = "missed_risk"
        if failure_mode not in FAILURE_MODES:
            failure_mode = "OPERATIONAL_MISS"

        logger.info(
            "lesson_generated",
            ticker=comparison.get("ticker"),
            lesson_type=lesson_type,
            failure_mode=failure_mode,
        )

        return lesson_text, lesson_type, failure_mode

    except Exception as e:
        logger.error("lesson_generation_failed", error=str(e))
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Component 5: Lesson Storage
# ══════════════════════════════════════════════════════════════════════════════


async def store_lesson(
    lesson: str,
    lesson_type: str,
    failure_mode: str,
    comparison: dict[str, Any],
    confidence: float,
    lessons_memory: Any,
) -> bool:
    """
    Store a lesson in the global lessons_learned ChromaDB collection.

    Deduplicates by checking for existing lesson with matching
    (ticker, analysis_date) metadata.

    Args:
        lesson: Lesson text (what gets embedded)
        lesson_type: directional type (missed_risk, etc.)
        failure_mode: structural type (CYCLICAL_PEAK, etc.)
        comparison: Full comparison dict
        confidence: Computed confidence weight
        lessons_memory: FinancialSituationMemory instance for lessons_learned

    Returns:
        True if stored, False if skipped/failed
    """
    if not lessons_memory or not lessons_memory.available:
        logger.debug("lessons_memory_unavailable")
        return False

    ticker = comparison.get("ticker", "UNKNOWN")
    analysis_date = comparison.get("analysis_date", "")

    # Deduplication: check if lesson already exists for this ticker + date
    try:
        existing = lessons_memory.situation_collection.get(
            where={
                "$and": [
                    {"ticker": {"$eq": ticker}},
                    {"analysis_date": {"$eq": analysis_date}},
                ]
            }
        )
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            logger.info(
                "lesson_already_exists",
                ticker=ticker,
                date=analysis_date,
                count=len(existing["ids"]),
            )
            return False
    except Exception as e:
        logger.debug("dedup_check_failed", error=str(e))
        # Continue with storage — better to have a duplicate than lose a lesson

    metadata = {
        "ticker": ticker,
        "sector": comparison.get("sector", "Unknown") or "Unknown",
        "exchange": comparison.get("exchange", "US") or "US",
        "currency": comparison.get("currency", "USD") or "USD",
        "verdict": comparison.get("verdict", "UNKNOWN") or "UNKNOWN",
        "actual_return_pct": float(comparison.get("price_return_pct", 0.0)),
        "benchmark_return_pct": float(comparison.get("benchmark_return_pct", 0.0)),
        "excess_return_pct": float(comparison.get("excess_return_pct", 0.0)),
        "fx_delta_pct": float(comparison.get("fx_delta_pct", 0.0)),
        "days_elapsed": int(comparison.get("days_elapsed", 0)),
        "lesson_type": lesson_type,
        "failure_mode": failure_mode,
        "analysis_model": comparison.get("deep_model", "unknown") or "unknown",
        "analysis_date": analysis_date,
        "retrospective_date": datetime.now().strftime("%Y-%m-%d"),
        "confidence_weight": float(confidence),
        "timestamp": datetime.now().isoformat(),
    }

    stored = await lessons_memory.add_situations([lesson], [metadata])
    if stored:
        logger.info(
            "lesson_stored",
            ticker=ticker,
            lesson_type=lesson_type,
            failure_mode=failure_mode,
            confidence=confidence,
            excess_return=comparison.get("excess_return_pct"),
        )
    else:
        logger.warning(
            "lesson_storage_failed",
            ticker=ticker,
            lesson_type=lesson_type,
        )
    return stored


# ══════════════════════════════════════════════════════════════════════════════
# Component 6: Lesson Retrieval & Injection
# ══════════════════════════════════════════════════════════════════════════════


async def get_relevant_lessons(
    lessons_memory: Any,
    sector: str,
    ticker: str,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Query lessons_learned collection for relevant past lessons.

    Args:
        lessons_memory: FinancialSituationMemory for lessons_learned collection
        sector: Sector of current analysis (for query relevance)
        ticker: Current ticker (for exchange/currency matching)
        n_results: Max results to fetch from ChromaDB

    Returns:
        List of lesson dicts with 'document', 'metadata', 'distance' keys
    """
    if not lessons_memory or not lessons_memory.available:
        return []

    try:
        query = f"Investment lessons for {sector} sector stocks"
        results = await lessons_memory.query_similar_situations(
            query_text=query,
            n_results=n_results,
        )
        return results
    except Exception as e:
        logger.debug("lesson_query_failed", error=str(e))
        return []


async def format_lessons_for_injection(
    lessons_memory: Any,
    ticker: str,
    sector: str,
) -> str:
    """
    Query global lessons collection, rank by confidence + geographic boost,
    return formatted text for injection into researcher prompts.

    Called from agents.py researcher_node (2-line integration).

    Args:
        lessons_memory: FinancialSituationMemory for lessons_learned collection
        ticker: Current ticker being analyzed
        sector: Sector of current ticker

    Returns:
        Formatted string for prompt injection, or "" if no lessons available
    """
    if not lessons_memory or not lessons_memory.available:
        return ""

    # Fast-path: no lessons exist yet — skip embedding API call (~1-2ms check vs ~200ms)
    try:
        if lessons_memory.situation_collection.count() == 0:
            return ""
    except Exception:
        pass  # Fall through to normal query

    try:
        results = await get_relevant_lessons(lessons_memory, sector, ticker)
    except Exception:
        return ""

    if not results:
        return ""

    # Apply geographic boost and confidence filtering
    suffix = _get_ticker_suffix(ticker)
    current_exchange = suffix.lstrip(".") if suffix else "US"
    current_currency = EXCHANGE_CURRENCY.get(suffix, FALLBACK_CURRENCY)

    scored_lessons = []
    for r in results:
        meta = r.get("metadata", {})
        base_confidence = meta.get("confidence_weight", 0.5)

        # Geographic boost
        boost = 0.0
        if meta.get("exchange") == current_exchange:
            boost += 0.15
        if meta.get("currency") == current_currency:
            boost += 0.10

        effective_score = base_confidence + boost

        # Filter low-confidence lessons
        if effective_score < 0.4:
            continue

        scored_lessons.append(
            {
                "lesson": r["document"],
                "failure_mode": meta.get("failure_mode", "UNKNOWN"),
                "sector": meta.get("sector", "Unknown"),
                "exchange": meta.get("exchange", "??"),
                "confidence": round(effective_score, 2),
            }
        )

    # Sort by effective score descending, take top 3
    scored_lessons.sort(key=lambda x: x["confidence"], reverse=True)
    top_lessons = scored_lessons[:3]

    filtered_count = len(results) - len(scored_lessons) if results else 0
    logger.debug(
        "lesson_retrieval_stats",
        ticker=ticker,
        sector=sector,
        candidates=len(results) if results else 0,
        passed_filter=len(scored_lessons),
        filtered_out=filtered_count,
        top_n=len(top_lessons),
    )

    if not top_lessons:
        return ""

    lines = ["LESSONS FROM PAST ANALYSES (cross-market):"]
    for lesson in top_lessons:
        lines.append(
            f"- {lesson['lesson']} "
            f"({lesson['failure_mode']} | {lesson['sector']}/{lesson['exchange']} "
            f"| conf: {lesson['confidence']})"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Component 7: Early Dedup Helper
# ══════════════════════════════════════════════════════════════════════════════


def _lesson_already_processed(
    lessons_memory: Any, ticker: str, analysis_date: str
) -> bool:
    """Check if a lesson already exists for this (ticker, analysis_date).

    Uses ChromaDB metadata query — no embedding needed. ~50ms.
    """
    if not lessons_memory or not lessons_memory.available:
        return False
    try:
        existing = lessons_memory.situation_collection.get(
            where={
                "$and": [
                    {"ticker": {"$eq": ticker}},
                    {"analysis_date": {"$eq": analysis_date}},
                ]
            }
        )
        return bool(existing and existing.get("ids"))
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Component 8: Orchestrator
# ══════════════════════════════════════════════════════════════════════════════


async def run_retrospective(
    ticker: str | None,
    results_dir: Path,
    lessons_memory: Any = None,
) -> list[dict[str, Any]]:
    """
    Orchestrate retrospective: load snapshots → compare → generate → store.

    Args:
        ticker: If provided, process only this ticker. If None, all tickers.
        results_dir: Directory containing analysis JSONs.
        lessons_memory: FinancialSituationMemory for lessons_learned.
                       If None, creates one.

    Returns:
        List of generated lesson dicts (for display/logging)
    """
    # Create lessons memory if not provided
    if lessons_memory is None:
        try:
            from src.memory import FinancialSituationMemory

            lessons_memory = FinancialSituationMemory(LESSONS_COLLECTION_NAME)
        except Exception as e:
            logger.error("lessons_memory_init_failed", error=str(e))
            return []

    # Load snapshots
    all_snapshots = load_past_snapshots(ticker, results_dir)

    if not all_snapshots:
        msg = f"for {ticker}" if ticker else "in results directory"
        logger.info(f"No past analyses with snapshots found {msg}")
        return []

    total_snapshots = sum(len(s) for s in all_snapshots.values())
    logger.info(
        "retrospective_starting",
        tickers=len(all_snapshots),
        total_snapshots=total_snapshots,
        filter_ticker=ticker or "all",
    )

    generated_lessons = []

    for snap_ticker, snapshots in all_snapshots.items():
        logger.info(
            "retrospective_processing_ticker",
            ticker=snap_ticker,
            snapshot_count=len(snapshots),
        )
        ticker_lessons = 0
        # Sort by significance (we'll evaluate all, but cap stored lessons)
        comparisons = []

        for snapshot in snapshots:
            # Early dedup: skip snapshots that already have a lesson in ChromaDB
            # This avoids the expensive yfinance call for already-processed data
            snap_ticker = snapshot.get("ticker", "")
            snap_date = snapshot.get("analysis_date", "")
            if _lesson_already_processed(lessons_memory, snap_ticker, snap_date):
                logger.debug(
                    "snapshot_already_processed",
                    ticker=snap_ticker,
                    date=snap_date,
                )
                continue

            comparison = await compare_to_reality(snapshot)
            if comparison:
                comparison["_confidence"] = compute_confidence(comparison)
                comparisons.append(comparison)

        # Sort by significance (largest excess return first)
        comparisons.sort(key=lambda c: abs(c.get("excess_return_pct", 0)), reverse=True)

        for comparison in comparisons:
            if ticker_lessons >= MAX_LESSONS_PER_TICKER:
                logger.info(
                    "max_lessons_reached",
                    ticker=snap_ticker,
                    max=MAX_LESSONS_PER_TICKER,
                )
                break

            confidence = comparison["_confidence"]
            result = await generate_lesson(comparison)
            if not result:
                continue

            lesson_text, lesson_type, failure_mode = result

            stored = await store_lesson(
                lesson_text,
                lesson_type,
                failure_mode,
                comparison,
                confidence,
                lessons_memory,
            )

            lesson_record = {
                "ticker": snap_ticker,
                "lesson": lesson_text,
                "lesson_type": lesson_type,
                "failure_mode": failure_mode,
                "excess_return_pct": comparison.get("excess_return_pct"),
                "confidence": confidence,
                "stored": stored,
            }
            generated_lessons.append(lesson_record)

            if stored:
                ticker_lessons += 1

    stored_count = sum(1 for lesson in generated_lessons if lesson.get("stored"))
    logger.info(
        "retrospective_complete",
        lessons_generated=len(generated_lessons),
        lessons_stored=stored_count,
        tickers_evaluated=len(all_snapshots),
    )
    return generated_lessons


def create_lessons_memory() -> Any:
    """
    Create a FinancialSituationMemory instance for the global lessons_learned
    collection. This is a factory function for use in memory.py or main.py.

    Returns:
        FinancialSituationMemory instance (may have available=False if ChromaDB
        is not configured)
    """
    from src.memory import FinancialSituationMemory

    return FinancialSituationMemory(LESSONS_COLLECTION_NAME)
