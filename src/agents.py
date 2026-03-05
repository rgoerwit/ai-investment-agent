"""
Multi-Agent Trading System - Agent Definitions.

FIXED: All data passing issues - agents now receive complete reports.
ADDED: Debug logging to track data flow.
FIXED: Memory contextualized per ticker to prevent cross-contamination.
UPDATED: Added Negative Constraint to prompts and metadata filtering.
UPDATED: Added 429/ResourceExhausted handling for Gemini free tier.
FIXED: Corrected memory query parameter name to 'metadata_filter'.
"""

import asyncio
import json
import random
import re
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import RunnableConfig
from typing_extensions import TypedDict

from src.config import config as settings_config

logger = structlog.get_logger(__name__)

# --- Rate Limit Handling ---


async def invoke_with_rate_limit_handling(
    runnable, input_data: dict[str, Any], max_attempts: int = 3, context: str = "LLM"
) -> Any:
    """
    Invoke LLM with explicit 429/ResourceExhausted handling for free tier.

    This wrapper adds extended backoff (60-180s) beyond LangChain's default retry logic,
    which maxes out at ~60s. Critical for Gemini free tier (15 RPM).

    Args:
        runnable: LangChain runnable (LLM, chain, etc.)
        input_data: Input dictionary for runnable
        max_attempts: Number of attempts at this wrapper level (default 3)
        context: Description for logging (e.g., "Market Analyst")

    Returns:
        Result from runnable.ainvoke()

    Raises:
        Exception: Re-raises if not a rate limit error or after max attempts
    """
    quiet_mode = settings_config.quiet_mode

    for attempt in range(max_attempts):
        try:
            return await runnable.ainvoke(input_data)
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__

            # Detect rate limit errors (429, ResourceExhausted, quota exceeded)
            is_rate_limit = any(
                [
                    "429" in error_str,
                    "rate limit" in error_str,
                    "quota" in error_str,
                    "resourceexhausted" in error_str,
                    "resource exhausted" in error_str,
                    "too many requests" in error_str,
                ]
            )

            # Detect transient errors (connection issues, timeouts, service blips)
            is_transient = any(
                [
                    "connection" in error_str,
                    "timeout" in error_str,
                    "timed out" in error_str,
                    "unavailable" in error_str,
                    "503" in error_str,
                    "502" in error_str,
                    "reset" in error_str,
                    error_str == "",  # Empty error = unknown transient failure
                ]
            )

            if is_rate_limit and attempt < max_attempts - 1:
                # Extended exponential backoff: 60s, 120s, 180s + random jitter
                # Jitter prevents "thundering herd" when parallel agents retry at exact same time
                jitter = random.uniform(1, 10)
                wait_time = (60 * (attempt + 1)) + jitter

                # Log unless in quiet mode
                if not quiet_mode:
                    logger.warning(
                        "rate_limit_detected",
                        context=context,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        wait_seconds=f"{wait_time:.1f}",
                        error_type=error_type,
                        error_message=str(e)[:200],  # Truncate long errors
                    )

                await asyncio.sleep(wait_time)
                continue  # Retry

            # Transient errors get shorter backoff (5s, 10s, 15s)
            if is_transient and attempt < max_attempts - 1:
                wait_time = 5 * (attempt + 1) + random.uniform(1, 3)
                if not quiet_mode:
                    logger.warning(
                        "transient_error_retry",
                        context=context,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        wait_seconds=f"{wait_time:.1f}",
                        error_type=error_type,
                    )
                await asyncio.sleep(wait_time)
                continue  # Retry

            # Not a retriable error, or final attempt - re-raise
            raise


# --- News Report Summarization ---


def extract_news_highlights(news_report: str, max_chars: int = 25000) -> str:
    """
    Extract key highlights from a news report for Senior Fundamentals.

    Keeps only the information needed for thesis compliance scoring:
    - US Revenue status
    - Geographic breakdown (summarized)
    - Top growth catalysts

    This reduces token usage while preserving decision-relevant data.
    """
    if not news_report or len(news_report) < 300:
        return news_report

    highlights = []
    lines = news_report.split("\n")

    # Extract US Revenue section (critical for thesis)
    in_geo_section = False
    geo_lines = []
    for line in lines:
        line_upper = line.upper()
        if "GEOGRAPHIC REVENUE" in line_upper or "US REVENUE" in line_upper:
            in_geo_section = True
            # Also capture this line if it contains actual data (not just a header)
            if ":" in line and not line.strip().startswith("###"):
                geo_lines.append(line)
        elif in_geo_section:
            if line.startswith("---") or line.startswith("###"):
                in_geo_section = False
            elif line.strip():
                geo_lines.append(line)
                if len(geo_lines) >= 6:  # Limit geographic section
                    break

    if geo_lines:
        highlights.append("**US/Geographic Revenue:**")
        highlights.extend(geo_lines[:6])

    # Extract growth catalysts (key bullet points only)
    in_catalyst_section = False
    catalyst_header_added = False
    catalyst_count = 0
    for line in lines:
        if "CATALYST" in line.upper() or "GROWTH CATALYST" in line.upper():
            in_catalyst_section = True
            if not catalyst_header_added:
                highlights.append("\n**Growth Catalysts:**")
                catalyst_header_added = True
        elif in_catalyst_section:
            if line.startswith("---") or (
                line.startswith("###") and "CATALYST" not in line.upper()
            ):
                in_catalyst_section = False
            elif line.strip().startswith(("1.", "2.", "3.", "-", "*", "•")):
                # Only take the first line of each catalyst
                highlights.append(line.strip()[:150])
                catalyst_count += 1
                if catalyst_count >= 3:  # Limit to top 3 catalysts
                    break

    result = "\n".join(highlights)

    # Final safety truncation
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[...truncated for efficiency]"

    return result if result.strip() else news_report[:max_chars]


def compute_data_conflicts(raw_data: str, foreign_data: str) -> str:
    """
    Deterministically compare Junior (aggregator) vs FLA (filing) data.

    Returns a structured conflict report for injection into Senior's context.
    This replaces asking the LLM to discover discrepancies — the code finds
    them and presents them as facts the LLM cannot rationalize away.
    """
    if not raw_data:
        return ""

    conflicts: list[str] = []

    # --- Parse Junior's raw JSON for key metrics ---
    junior_ocf = None
    junior_analysts = None
    junior_peg = None
    junior_mcap = None

    # Junior output is tool-call JSON; extract from the text
    def _extract_json_number(text: str, key: str) -> float | None:
        """Find a JSON key-value pair and extract the number."""
        for pattern in [
            rf'"{key}"\s*:\s*(-?[\d.eE+]+)',
            rf"'{key}'\s*:\s*(-?[\d.eE+]+)",
        ]:
            m = re.search(pattern, text)
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, OverflowError):
                    return None
        return None

    junior_ocf = _extract_json_number(raw_data, "operatingCashflow")
    junior_analysts = _extract_json_number(raw_data, "numberOfAnalystOpinions")
    junior_peg = _extract_json_number(raw_data, "pegRatio")
    junior_mcap = _extract_json_number(raw_data, "marketCap")

    # --- Parse FLA report for filing data ---
    filing_ocf = None
    filing_ocf_period = None
    parent_company = None

    if foreign_data:
        # Filing OCF: look for "Operating Cash Flow (Filing): ¥10.91B" etc.
        ocf_match = re.search(
            r"Operating Cash Flow\s*\(?Filing\)?[:\s]*([¥$€£₩]?[\d,.]+)\s*(B|M|T|billion|million|trillion)?",
            foreign_data,
            re.IGNORECASE,
        )
        if ocf_match:
            try:
                val_str = ocf_match.group(1).replace(",", "").lstrip("¥$€£₩")
                filing_ocf = float(val_str)
                suffix = (ocf_match.group(2) or "").upper()
                if suffix in ("B", "BILLION"):
                    filing_ocf *= 1e9
                elif suffix in ("M", "MILLION"):
                    filing_ocf *= 1e6
                elif suffix in ("T", "TRILLION"):
                    filing_ocf *= 1e12
            except (ValueError, OverflowError):
                filing_ocf = None

        period_match = re.search(
            r"Period[:\s]*(FY\d{4}|H[12]\s*\d{4}|Q[1-4]\s*\d{4}|\d{4})",
            foreign_data,
            re.IGNORECASE,
        )
        if period_match:
            filing_ocf_period = period_match.group(1).strip()

        # Parent company
        parent_match = re.search(
            r"(?:Parent Company|Controlling Shareholder)[:\s]*(.+?)(?:\n|$)",
            foreign_data,
            re.IGNORECASE,
        )
        if parent_match:
            parent_val = parent_match.group(1).strip()
            if parent_val.upper() not in ("NONE", "N/A", "NOT FOUND", ""):
                parent_company = parent_val

    # --- Generate conflict report ---

    # 1. OCF conflict
    if junior_ocf is not None and filing_ocf is not None and junior_ocf != 0:
        # Normalize to same scale if possible
        j_abs = abs(junior_ocf)
        f_abs = abs(filing_ocf)
        # Only compare if both are in similar magnitude (both raw or both scaled)
        if j_abs > 0 and f_abs > 0:
            ratio = max(j_abs, f_abs) / min(j_abs, f_abs)
            if ratio > 1.3:
                period_note = f" ({filing_ocf_period})" if filing_ocf_period else ""
                conflicts.append(
                    f"- OCF: Junior={junior_ocf:,.0f} [yfinance] vs "
                    f"Filing={filing_ocf:,.0f}{period_note} [FLA] — "
                    f"{ratio:.1f}x difference. "
                    f"{'PERIOD MISMATCH — cannot directly compare' if filing_ocf_period and 'H' in filing_ocf_period.upper() else 'INVESTIGATE: same metric, material divergence'}"
                )

    # 2. Analyst count anomaly
    if junior_analysts is not None:
        analysts_int = int(junior_analysts)
        if analysts_int < 5 and junior_mcap is not None and junior_mcap > 500_000_000:
            conflicts.append(
                f"- ANALYST_COUNT: {analysts_int} [yfinance] for "
                f"${junior_mcap / 1e9:.1f}B market cap — "
                f"ANOMALY: likely data gap, not genuinely uncovered. "
                f"Verify independently before relying on 'undiscovered' thesis."
            )

    # 3. PEG anomaly
    if junior_peg is not None and 0 <= junior_peg < 0.05:
        detail = (
            "growth denominator zero/missing/infinite"
            if junior_peg == 0
            else f"implies {1/junior_peg:.0f}x expected growth"
        )
        conflicts.append(
            f"- PEG: {junior_peg:.2f} [yfinance] — UNRELIABLE ({detail}). "
            f"Do not use PEG to justify valuation."
        )

    # 5. Local analyst coverage gap
    # yfinance's numberOfAnalystOpinions reflects Refinitiv/FactSet counts, which skew
    # toward English-accessible research. For ex-US equities the true total analyst count
    # may be higher due to local-language coverage invisible to global aggregators. FLA
    # web searches provide an estimate, but overlap between local and English counts is
    # unknown — we use max(English, Local) as a conservative lower bound for TOTAL_EST.
    if foreign_data:
        local_analyst_match = re.search(
            r"Estimated Local Analysts[:\s]*(\d+|HIGH|MODERATE|LOW|UNKNOWN)",
            foreign_data,
            re.IGNORECASE,
        )
        if local_analyst_match:
            local_val = local_analyst_match.group(1).strip().upper()
            if local_val.isdigit():
                local_count = int(local_val)
                junior_count = (
                    int(junior_analysts) if junior_analysts is not None else 0
                )
                if local_count > junior_count:
                    conflicts.append(
                        f"- LOCAL_ANALYST_COVERAGE: FLA found ~{local_count} local analysts "
                        f"vs {junior_count} [yfinance English-only count]. "
                        f"Total coverage likely higher than English count suggests. "
                        f"Consensus targets may be more reliable than English count implies."
                    )
            elif local_val in ("HIGH", "MODERATE"):
                conflicts.append(
                    f"- LOCAL_ANALYST_COVERAGE: FLA estimates {local_val} local analyst coverage. "
                    f"yfinance shows {int(junior_analysts) if junior_analysts is not None else 'N/A'} "
                    f"[English-only]. Total coverage is likely higher."
                )
            # UNKNOWN or LOW: no conflict

    # 4. Ownership gap
    if foreign_data and parent_company:
        conflicts.append(
            f"- PARENT/CONTROLLER: {parent_company} [FLA] — "
            f"yfinance does not provide parent-subsidiary data. "
            f"If controlling holder >40%, minority influence is limited."
        )
    elif foreign_data and not parent_company:
        # FLA searched but didn't find — note the gap
        if "OWNERSHIP STRUCTURE" in foreign_data.upper() and (
            "NONE" in foreign_data.upper() or "NOT FOUND" in foreign_data.upper()
        ):
            conflicts.append(
                "- PARENT/CONTROLLER: Not found by FLA search. "
                "yfinance only provides institutional holders, not parent companies. "
                "Ownership structure is UNVERIFIED for this ticker."
            )

    if not conflicts:
        return ""

    header = (
        "\n\n### AUTOMATED CONFLICT CHECK (system-generated, not agent output)\n"
        "The following discrepancies were detected by comparing Junior (aggregator) "
        "data against Foreign Language Analyst (filing/search) data. These are FACTS, "
        "not suggestions. Address each in your CROSS-CHECK FLAGS.\n"
    )
    return header + "\n".join(conflicts) + "\n"


# --- Value Trap Verdict Extraction ---


def extract_value_trap_verdict(value_trap_report: str) -> str:
    """Extract structured verdict from VALUE_TRAP_BLOCK and return a 1-line header.

    The Value Trap Detector is a dedicated agent with governance/catalyst analysis.
    Its structured verdict (SCORE/VERDICT/TRAP_RISK) must not be overridden by
    narrative labels from Bear/RM. This header makes the signal impossible to miss.
    """
    if not value_trap_report:
        return ""
    score_m = re.search(r"SCORE:\s*(\d+)", value_trap_report)
    verdict_m = re.search(
        r"VERDICT:\s*(TRAP|CAUTIOUS|WATCHABLE|ALIGNED)", value_trap_report
    )
    risk_m = re.search(r"TRAP_RISK:\s*(HIGH|MEDIUM|LOW)", value_trap_report)
    if not (score_m and verdict_m):
        return ""
    score = score_m.group(1)
    verdict = verdict_m.group(1)
    risk = risk_m.group(1) if risk_m else "N/A"
    return (
        f"⚠ VALUE_TRAP_DETECTOR VERDICT: {verdict} (score {score}/100, "
        f"risk {risk}) — dedicated governance/catalyst agent assessment\n"
    )


# --- PM Input Summarization ---


def summarize_for_pm(report: str, report_type: str, max_chars: int = 3000) -> str:
    """
    Extract key information from agent reports for PM consumption.

    Preserves structured blocks (DATA_BLOCK, PM_BLOCK, etc.) but summarizes
    narrative content to reduce PM's input context size.

    Args:
        report: Full agent report text
        report_type: One of "market", "sentiment", "news", "fundamentals",
                     "value_trap", "consultant", "trader", "risk"
        max_chars: Maximum output characters

    Returns:
        Summarized report with critical data preserved
    """
    if not report or not isinstance(report, str):
        return report or ""

    if len(report) <= max_chars:
        return report

    import re

    # Structured blocks to preserve in full (these contain critical data)
    blocks_to_preserve = []
    block_patterns = [
        r"(DATA_BLOCK:.*?)(?=\n\n[A-Z]|\Z)",
        r"(### --- START DATA_BLOCK[^\n]*---.*?### --- END DATA_BLOCK ---)",
        r"(PM_BLOCK:.*?)(?=\n\n[A-Z]|\Z)",
        r"(FORENSIC_DATA_BLOCK:.*?)(?=\n\n[A-Z]|\Z)",
        r"(VALUE_TRAP_BLOCK:.*?)(?=\n\n[A-Z]|\Z)",
        r"(\*\*VERDICT\*\*:.*?)(?=\n\n|\Z)",
        r"(RECOMMENDATION:.*?)(?=\n\n|\Z)",
        r"(SCORE:\s*\d+.*?)(?=\n\n|\Z)",
    ]

    for pattern in block_patterns:
        matches = re.findall(pattern, report, re.DOTALL | re.IGNORECASE)
        blocks_to_preserve.extend(matches)

    preserved = "\n\n".join(blocks_to_preserve)

    # If preserved blocks fit, add context from the beginning
    remaining_chars = max_chars - len(preserved) - 100  # Leave buffer

    if remaining_chars > 300:
        # Extract first paragraph (usually contains key thesis/summary)
        first_section = report[:remaining_chars]
        # Try to cut at a paragraph boundary
        last_para = first_section.rfind("\n\n")
        if last_para > 200:
            first_section = first_section[:last_para]

        if preserved:
            return f"{first_section}\n\n[...summarized...]\n\n{preserved}"
        else:
            return first_section + "\n\n[...summarized...]"

    # If blocks alone exceed limit, truncate blocks
    if preserved:
        return preserved[:max_chars]

    # Fallback: just truncate
    return report[:max_chars] + "\n[...summarized...]"


def _format_date_with_fy_hint(current_date: str) -> str:
    """Format date with fiscal year hint to prevent future-dated searches.

    LLMs may otherwise search for "FY2026 annual report" in Jan 2026 when
    FY2025 is the most recent available.

    Example: "2026-01-08 (latest annual reports: FY2025)"
    """
    try:
        year = int(current_date[:4])
        return f"{current_date} (latest annual reports: FY{year - 1})"
    except (ValueError, IndexError):
        return current_date


# --- Attribution Helpers ---


def extract_field_sources_from_messages(messages: list) -> dict[str, str]:
    """
    Extract _field_sources from ToolMessage content in message history.

    Searches for get_financial_metrics tool output which contains per-field
    source attribution. This is the authoritative source since tool outputs
    are not modified by LLM summarization.

    Args:
        messages: List of messages from state["messages"]

    Returns:
        Dict mapping field names to source names, or empty dict if not found.
    """
    from langchain_core.messages import ToolMessage

    if not messages:
        return {}

    # Search in reverse order (most recent first) for efficiency
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue

        try:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if '"_field_sources"' not in content:
                continue

            data = json.loads(content)
            if isinstance(data, dict) and "_field_sources" in data:
                field_sources = data["_field_sources"]
                if isinstance(field_sources, dict) and field_sources:
                    return field_sources
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue

    logger.debug(
        "field_sources_not_found",
        message="No _field_sources found in message history",
        tool_messages_checked=sum(
            1
            for m in messages
            if hasattr(m, "type") and getattr(m, "type", None) == "tool"
        ),
    )
    return {}


def format_attribution_table(field_sources: dict[str, str] | None) -> str:
    """
    Format source attribution as a compact table for prompt injection.

    Filters to priority fields to minimize token overhead (~60 tokens).
    Only injected into Consultant and PM prompts where adjudication occurs.

    Args:
        field_sources: Dict from extract_field_sources_from_messages()

    Returns:
        Formatted attribution table, or empty string if no sources.
    """
    if not field_sources:
        return ""

    # Priority fields for adjudication (financial health, valuation, growth)
    PRIORITY_FIELDS = [
        "marketCap",
        "netIncome",
        "totalRevenue",
        "trailingPE",
        "debtToEquity",
        "freeCashflow",
        "roe",
        "currentPrice",
    ]

    lines = []
    for field in PRIORITY_FIELDS:
        if field in field_sources:
            lines.append(f"- {field}: {field_sources[field]}")

    if not lines:
        return ""

    return "\n### DATA SOURCE ATTRIBUTION\n" + "\n".join(lines) + "\n"


def extract_source_conflicts_from_messages(messages: list) -> dict[str, Any]:
    """
    Extract _source_conflicts from ToolMessage content in message history.

    Searches for get_financial_metrics tool output which contains per-field
    source conflict records (when multiple data sources disagreed >20%).

    Returns:
        Dict mapping field names to conflict details, or empty dict.
    """
    from langchain_core.messages import ToolMessage

    if not messages:
        return {}

    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        try:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if '"_source_conflicts"' not in content:
                continue
            data = json.loads(content)
            if isinstance(data, dict) and "_source_conflicts" in data:
                conflicts = data["_source_conflicts"]
                if isinstance(conflicts, dict) and conflicts:
                    return conflicts
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return {}


def format_conflict_table(messages: list) -> str:
    """
    Format source conflicts for Consultant/PM adjudication. ~30-60 tokens when populated.

    Only injected when conflicts exist — zero tokens otherwise.
    """
    conflicts = extract_source_conflicts_from_messages(messages)
    if not conflicts:
        return ""

    lines = ["\n### DATA SOURCE CONFLICTS (>20% variance between providers)"]
    for field, c in conflicts.items():
        lines.append(
            f"  - {field}: {c.get('old_source', '?')}={c.get('old', '?')}"
            f", {c.get('new_source', '?')}={c.get('new', '?')}"
            f" (Δ{c.get('variance_pct', '?')}%)"
        )
    return "\n".join(lines) + "\n"


# --- State Definitions ---
class InvestDebateState(TypedDict):
    """
    State tracking bull/bear investment debate progression (parallel-safe).

    Uses dedicated fields per round to allow parallel execution of Bull/Bear
    in each round without race conditions.
    """

    # Round 1 outputs (dedicated fields for parallel safety)
    bull_round1: str
    bear_round1: str

    # Round 2 outputs (dedicated fields for parallel safety)
    bull_round2: str
    bear_round2: str

    # Current round (1 or 2)
    current_round: int

    # Assembled histories (built at sync points for downstream consumers)
    bull_history: str  # All Bull arguments concatenated
    bear_history: str  # All Bear arguments concatenated
    history: str  # Full debate history for Research Manager

    # Legacy fields (kept for backward compatibility)
    current_response: str
    judge_decision: str
    count: int  # Total argument count (for compatibility)


class RiskDebateState(TypedDict):
    """State tracking multi-perspective risk assessment debate (parallel-safe)."""

    latest_speaker: str
    current_risky_response: str
    current_safe_response: str
    current_neutral_response: str


def take_last(x, y):
    """Reducer: takes the most recent value. Used with Annotated fields."""
    return y


def merge_dicts(x: dict | None, y: dict | None) -> dict:
    """Reducer: merges dictionaries. Used for parallel agent state updates."""
    if x is None:
        return y or {}
    if y is None:
        return x
    return {**x, **y}


def merge_risk_state(
    x: RiskDebateState | None, y: RiskDebateState | None
) -> RiskDebateState:
    """
    Reducer for RiskDebateState that merges parallel updates.

    Simple merge is safe because each parallel agent writes to a DISTINCT key
    (current_risky_response vs current_safe_response vs current_neutral_response).
    The only shared key is 'latest_speaker', where last-write-wins is acceptable.
    """
    if x is None:
        return y or RiskDebateState(
            latest_speaker="",
            current_risky_response="",
            current_safe_response="",
            current_neutral_response="",
        )
    if y is None:
        return x
    return {**x, **y}


def merge_invest_debate_state(
    x: InvestDebateState | None, y: InvestDebateState | None
) -> InvestDebateState:
    """
    Reducer for InvestDebateState that merges parallel updates.

    Safe for parallel Bull/Bear execution because each writes to DISTINCT fields:
    - Bull writes to bull_round1 or bull_round2
    - Bear writes to bear_round1 or bear_round2

    For string fields, prefers non-empty values to allow parallel writes.
    For numeric fields (current_round, count), uses last-write-wins.
    """
    default_state = InvestDebateState(
        bull_round1="",
        bear_round1="",
        bull_round2="",
        bear_round2="",
        current_round=1,
        bull_history="",
        bear_history="",
        history="",
        current_response="",
        judge_decision="",
        count=0,
    )
    if x is None:
        return y or default_state
    if y is None:
        return x

    # Merge with preference for non-empty string values
    result = {}
    all_keys = set(x.keys()) | set(y.keys())
    for key in all_keys:
        x_val = x.get(key, default_state.get(key))
        y_val = y.get(key, default_state.get(key))

        # For string fields, prefer non-empty value
        if isinstance(x_val, str) and isinstance(y_val, str):
            result[key] = y_val if y_val else x_val
        else:
            # For numeric/other fields, last-write-wins (prefer y)
            result[key] = y_val if y_val is not None else x_val

    return result


class AgentState(MessagesState):
    company_of_interest: str
    company_name: str  # ADDED: Verified company name to prevent LLM hallucination
    company_name_resolved: bool  # Whether company_name was verified from a data source
    trade_date: str
    sender: Annotated[str, take_last]  # Support parallel writes

    market_report: Annotated[str, take_last]
    sentiment_report: Annotated[str, take_last]
    news_report: Annotated[str, take_last]
    raw_fundamentals_data: Annotated[str, take_last]  # Junior Analyst output
    foreign_language_report: Annotated[
        str, take_last
    ]  # Foreign Language Analyst output
    legal_report: Annotated[str, take_last]  # Legal Counsel output (PFIC/VIE JSON)
    fundamentals_report: Annotated[str, take_last]  # Senior Analyst analysis
    auditor_report: Annotated[str, take_last]  # Independent forensic auditor report
    value_trap_report: Annotated[
        str, take_last
    ]  # Value Trap Detector governance analysis
    investment_debate_state: Annotated[InvestDebateState, merge_invest_debate_state]
    investment_plan: Annotated[str, take_last]
    valuation_params: Annotated[
        str, take_last
    ]  # Valuation Calculator output for charts
    consultant_review: Annotated[str, take_last]  # External consultant validation
    trader_investment_plan: Annotated[str, take_last]
    risk_debate_state: Annotated[RiskDebateState, merge_risk_state]
    final_trade_decision: Annotated[str, take_last]
    tools_called: Annotated[dict[str, set[str]], merge_dicts]
    prompts_used: Annotated[dict[str, dict[str, str]], merge_dicts]

    # Red-flag detection fields
    red_flags: Annotated[list[dict[str, Any]], take_last]
    pre_screening_result: Annotated[str, take_last]  # "PASS" or "REJECT"

    # Chart generation (post-PM)
    chart_paths: Annotated[
        dict[str, str], take_last
    ]  # {"football_field": path, "radar": path}


# --- Helper Functions ---


def get_context_from_config(config: RunnableConfig) -> Any | None:
    """Extract TradingContext from RunnableConfig.configurable dict."""
    try:
        configurable = config.get("configurable", {})
        return configurable.get("context")
    except (AttributeError, TypeError):
        return None


_UNRESOLVED_NAME_WARNING = (
    "\nWARNING: Company name could not be verified from any data source. "
    "The ticker may be delisted or illiquid. Do NOT guess or assume which company "
    "this ticker belongs to. If you cannot confirm the identity from your tool "
    "results, state that the company identity is unverified."
)


def _company_line(company_name: str, resolved: bool) -> str:
    """Build Company: line with optional unresolved warning."""
    line = f"Company: {company_name}"
    if not resolved:
        line += _UNRESOLVED_NAME_WARNING
    return line


def get_analysis_context(ticker: str) -> str:
    """Generate contextual guidance based on asset type (ETF vs stock)."""
    etf_indicators = [
        "VTI",
        "SPY",
        "QQQ",
        "IWM",
        "VOO",
        "VEA",
        "VWO",
        "BND",
        "AGG",
        "EFA",
        "EEM",
        "TLT",
        "GLD",
        "DIA",
    ]
    is_etf = (
        any(ind in ticker.upper() for ind in etf_indicators) or "ETF" in ticker.upper()
    )
    if is_etf:
        return (
            "This is an ETF (Exchange-Traded Fund). "
            "Focus on holdings, expense ratio, and liquidity."
        )
    return (
        "This is an individual stock. "
        "Focus on fundamentals, valuation, and competitive advantage."
    )


def filter_messages_by_agent(
    messages: list[BaseMessage], agent_key: str
) -> list[BaseMessage]:
    """
    Filter messages to only include this agent's conversation history.

    CRITICAL FIX for parallel execution: In LangGraph parallel branches, all agents
    share the messages list. Without filtering, Agent A can see Agent B's tool calls,
    causing confusion and incorrect responses.

    Keeps:
    - HumanMessage: Initial instructions (shared by all)
    - AIMessage with name == agent_key: This agent's responses
    - ToolMessage with agent_key in additional_kwargs: This agent's tool results

    Args:
        messages: All messages from state
        agent_key: The agent's identifier (e.g., "market_analyst")

    Returns:
        Filtered messages for this agent only
    """
    if not messages:
        return []

    # Trace ToolMessages and their agent tags (visible with --verbose flag)
    tool_msg_agents = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tag = (
                msg.additional_kwargs.get("agent_key")
                if msg.additional_kwargs
                else None
            )
            tool_msg_agents.append(tag)
    logger.debug(
        "filter_messages_tool_tags",
        agent_key=agent_key,
        total_tool_messages=len(tool_msg_agents),
        tool_message_tags=tool_msg_agents,
    )

    filtered = []
    for msg in messages:
        # Always include HumanMessages (initial instructions)
        if isinstance(msg, HumanMessage):
            filtered.append(msg)
        # Include AIMessages tagged with this agent
        elif isinstance(msg, AIMessage):
            if getattr(msg, "name", None) == agent_key:
                filtered.append(msg)
        # Include ToolMessages tagged with this agent
        elif isinstance(msg, ToolMessage):
            msg_agent = (
                msg.additional_kwargs.get("agent_key")
                if msg.additional_kwargs
                else None
            )
            if msg_agent == agent_key:
                filtered.append(msg)
        # Skip SystemMessages (re-added fresh by analyst node)

    return filtered


def filter_messages_for_gemini(
    messages: list[BaseMessage], agent_key: str | None = None
) -> list[BaseMessage]:
    """
    Filter and format messages for Gemini API compatibility.

    If agent_key is provided, first filters to only this agent's messages (parallel-safe).
    Then applies Gemini-specific formatting (merge consecutive HumanMessages).
    """
    # Step 1: Filter by agent if provided (parallel execution safety)
    if agent_key:
        messages = filter_messages_by_agent(messages, agent_key)

    if not messages:
        return []

    # Step 2: Apply Gemini formatting (merge consecutive HumanMessages)
    filtered = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        is_consecutive_human = (
            filtered
            and isinstance(msg, HumanMessage)
            and isinstance(filtered[-1], HumanMessage)
        )
        if is_consecutive_human:
            last_msg = filtered.pop()
            new_content = f"{last_msg.content}\n\n{msg.content}"
            filtered.append(HumanMessage(content=new_content))
        else:
            filtered.append(msg)
    return filtered


def extract_string_content(content: Any) -> str:
    """
    Safely extract string content from LLM response.content.

    Gemini models (especially with langchain-google-genai) can return structured
    data (dict/list) instead of plain strings when responses contain tool calls
    or multi-part content. This function normalizes the content to a string.

    Args:
        content: The response.content value, which may be str, dict, list, or other

    Returns:
        String representation of the content
    """
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        # Try common keys for text content in structured responses
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return extract_string_content(content["content"])  # Recursive for nested
        if "parts" in content:
            # Gemini multi-part response format
            parts = content["parts"]
            if isinstance(parts, list):
                text_parts = [extract_string_content(p) for p in parts]
                return "\n".join(filter(None, text_parts))
        # Fallback: convert dict to readable string (expected for some API response formats)
        logger.debug("response_content_is_dict", keys=list(content.keys()))
        return str(content)

    if isinstance(content, list):
        # Handle list of content parts
        if len(content) == 0:
            return ""
        if len(content) == 1:
            return extract_string_content(content[0])
        # Multiple parts - join them
        text_parts = [extract_string_content(item) for item in content]
        return "\n".join(filter(None, text_parts))

    # Fallback for any other type
    return str(content) if content is not None else ""


# --- Agent Factory Functions ---


def _is_output_insufficient(content: str, agent_key: str) -> bool:
    """
    Check if agent output is empty or truncated/insufficient.

    Used to determine if a retry with higher thinking_level is needed.

    Args:
        content: The extracted string content from the agent response
        agent_key: The agent identifier (e.g., "fundamentals_analyst")

    Returns:
        True if output is insufficient and retry may help:
        - Content is empty or very short (< 50 chars)
        - For fundamentals_analyst: Missing DATA_BLOCK marker
        - For news_analyst: Content too short (< 200 chars)
    """
    if not content or len(content) < 50:
        return True

    if agent_key == "junior_fundamentals_analyst":
        # Junior analyst should return raw tool data
        # Check for markers that indicate tools were called
        has_tool_output = (
            "get_financial_metrics" in content.lower()
            or "financial" in content.lower()
            or "roe" in content.lower()
            or "=== RAW" in content
        )
        return not has_tool_output or len(content) < 200

    if agent_key == "fundamentals_analyst":
        # Senior fundamentals report MUST contain DATA_BLOCK
        return "DATA_BLOCK" not in content

    if agent_key == "news_analyst":
        # News report should have reasonable content
        return len(content) < 200

    # For other agents (market, sentiment), just check for non-trivial content
    return False


def create_analyst_node(
    llm,
    agent_key: str,
    tools: list[Any],
    output_field: str,
    retry_llm: Any | None = None,
    allow_retry: bool = False,
) -> Callable:
    """
    Factory function creating data analyst agent nodes.

    Args:
        llm: Primary LLM (quick_think_llm with thinking_level=low)
        agent_key: Agent identifier for prompt lookup
        tools: Tools to bind to the LLM for function calling
        output_field: State field to store output (e.g., 'fundamentals_report')
        retry_llm: Optional fallback LLM (deep_think_llm, thinking_level=high)
        allow_retry: If True, retry ONCE with retry_llm if output insufficient

    Note:
        Retry only happens ONCE to prevent infinite loops.
    """

    async def analyst_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error(f"Missing prompt for agent: {agent_key}")
            return {output_field: f"Error: Could not load prompt for {agent_key}."}
        messages_template = [MessagesPlaceholder(variable_name="messages")]
        prompt_template = ChatPromptTemplate.from_messages(messages_template)
        if tools:
            runnable = prompt_template | llm.bind_tools(tools)
        else:
            runnable = prompt_template | llm
        try:
            prompts_used = state.get("prompts_used", {})
            prompts_used[output_field] = {
                "agent_name": agent_prompt.agent_name,
                "version": agent_prompt.version,
            }
            # CRITICAL: Filter messages to only this agent's history (parallel execution safety)
            filtered_messages = filter_messages_for_gemini(
                state.get("messages", []), agent_key=agent_key
            )
            # Trace message types being sent to LLM (visible with --verbose flag)
            msg_types = [type(m).__name__ for m in filtered_messages]
            msg_has_tool_calls = [
                bool(getattr(m, "tool_calls", None))
                for m in filtered_messages
                if hasattr(m, "tool_calls")
            ]
            logger.debug(
                "analyst_filtered_messages",
                agent_key=agent_key,
                total_state_messages=len(state.get("messages", [])),
                filtered_count=len(filtered_messages),
                message_types=msg_types,
                has_tool_calls_list=msg_has_tool_calls,
            )
            context = get_context_from_config(config)
            current_date = (
                context.trade_date if context else datetime.now().strftime("%Y-%m-%d")
            )
            ticker = (
                context.ticker
                if context
                else state.get("company_of_interest", "UNKNOWN")
            )
            company_name = state.get(
                "company_name", ticker
            )  # Get verified company name from state
            company_resolved = state.get("company_name_resolved", True)

            # --- Context injection for specific agents ---
            extra_context = ""

            # Junior Fundamentals Analyst: Gets news context for qualitative info
            # NOTE: In parallel mode, news_report may not be available yet
            # Junior primarily uses tools - news context is supplementary
            if agent_key == "junior_fundamentals_analyst":
                news_report = state.get("news_report", "")
                if news_report:
                    extra_context = (
                        f"\n\n### NEWS CONTEXT (for ADR/analyst search queries)"
                        f"\n{news_report}\n"
                    )
                # Don't log warning - Junior's job is tool calling, news is optional

            # Senior Fundamentals Analyst: Gets raw data from Junior AND Foreign Language Analyst
            # NOTE: Both Junior and Foreign Language analysts complete before Senior via Fundamentals Sync
            if agent_key == "fundamentals_analyst":
                raw_data = state.get("raw_fundamentals_data", "")
                foreign_data = state.get("foreign_language_report", "")
                news_report = state.get("news_report", "")

                if raw_data:
                    extra_context = (
                        f"\n\n### RAW FINANCIAL DATA FROM JUNIOR ANALYST (Primary Source)"
                        f"\n{raw_data}\n"
                    )
                else:
                    logger.warning(
                        "senior_fundamentals_no_raw_data",
                        ticker=ticker,
                        message="Junior Analyst data not available - this should not happen",
                    )

                # Foreign Language Analyst data supplements Junior's data
                if foreign_data:
                    extra_context += (
                        f"\n\n### FOREIGN/ALTERNATIVE SOURCE DATA (Cross-Reference)"
                        f"\nNote: Use this data to FILL GAPS in Junior Analyst data. "
                        f"Prioritize Junior's data when both sources have the same metric.\n"
                        f"{foreign_data}\n"
                    )
                    logger.info(
                        "senior_fundamentals_has_foreign_data",
                        ticker=ticker,
                        foreign_data_length=len(foreign_data),
                    )
                else:
                    logger.info(
                        "senior_fundamentals_no_foreign_data",
                        ticker=ticker,
                        message="Foreign Language Analyst data not available - proceeding with Junior data only",
                    )

                if news_report:
                    # Summarize news to reduce token usage (~70% reduction)
                    news_highlights = extract_news_highlights(news_report)
                    extra_context += (
                        f"\n\n### NEWS HIGHLIGHTS (for Qualitative Growth Scoring)"
                        f"\n{news_highlights}\n"
                    )
                else:
                    # Expected in parallel mode - News Analyst may still be running
                    logger.info(
                        "senior_fundamentals_no_news",
                        ticker=ticker,
                        message="News report not yet available (parallel execution) - proceeding without news context",
                    )

                # Pre-computed conflict check (deterministic, not LLM)
                conflict_report = compute_data_conflicts(raw_data, foreign_data)
                if conflict_report:
                    extra_context += conflict_report
                    logger.info(
                        "senior_fundamentals_conflicts_detected",
                        ticker=ticker,
                        conflict_count=conflict_report.count("\n- "),
                    )

                # Legal Counsel data for PFIC/VIE reconciliation
                legal_report = state.get("legal_report", "")
                if legal_report:
                    extra_context += (
                        f"\n\n### LEGAL/TAX RISK ASSESSMENT (From Legal Counsel)"
                        f"\nUse this to inform your PFIC_RISK assessment in DATA_BLOCK. "
                        f"If Legal Counsel found PFIC disclosure (pfic_status: PROBABLE), set PFIC_RISK: MEDIUM or HIGH. "
                        f"If no disclosure found in high-risk sector (pfic_status: UNCERTAIN), set PFIC_RISK: MEDIUM.\n"
                        f"{legal_report}\n"
                    )
                    logger.info(
                        "senior_fundamentals_has_legal_data",
                        ticker=ticker,
                        legal_data_length=len(legal_report),
                    )
                else:
                    logger.info(
                        "senior_fundamentals_no_legal_data",
                        ticker=ticker,
                        message="Legal Counsel data not yet available - proceeding without legal context",
                    )

            # CRITICAL FIX: Include verified company name to prevent hallucination
            full_system_instruction = f"{agent_prompt.system_message}\n\nDate: {_format_date_with_fy_hint(current_date)}\nTicker: {ticker}\n{_company_line(company_name, company_resolved)}\n{get_analysis_context(ticker)}{extra_context}"
            invocation_messages = [
                SystemMessage(content=full_system_instruction)
            ] + filtered_messages

            # Use rate limit handling wrapper for free tier support
            response = await invoke_with_rate_limit_handling(
                runnable,
                {"messages": invocation_messages},
                context=agent_prompt.agent_name,
            )

            # CRITICAL: Tag outgoing message with agent_key for parallel execution filtering
            # This allows other iterations of this agent to identify its own messages
            response.name = agent_key

            new_state = {
                "sender": agent_key,
                "messages": [response],
                "prompts_used": prompts_used,
            }

            # Check for tool calls
            has_tool_calls = False
            try:
                if hasattr(response, "tool_calls") and response.tool_calls:
                    has_tool_calls = (
                        isinstance(response.tool_calls, list)
                        and len(response.tool_calls) > 0
                    )
                # DEBUG: Log response details
                logger.info(
                    "analyst_response_details",
                    agent_key=agent_key,
                    content_type=type(response.content).__name__,
                    content_len=len(response.content) if response.content else 0,
                    tool_calls_count=(
                        len(response.tool_calls) if response.tool_calls else 0
                    ),
                    has_tool_calls=has_tool_calls,
                )
            except (AttributeError, TypeError):
                pass

            if has_tool_calls:
                return new_state

            # CRITICAL FIX: Normalize response.content to string
            # Gemini can return dict/list instead of string for structured responses
            content_str = extract_string_content(response.content)

            # --- RETRY LOGIC FOR GEMINI 3+ WITH THINKING_LEVEL ---
            # If output is insufficient (empty/truncated) and retry is allowed,
            # retry ONCE with the deep thinking LLM (thinking_level=high).
            # This handles cases where Gemini 3+ with thinking_level=low returns empty.
            # NO LOOPING: Only one retry attempt is made.
            if (
                allow_retry
                and retry_llm is not None
                and _is_output_insufficient(content_str, agent_key)
            ):
                logger.warning(
                    "analyst_retry_with_deep_thinking",
                    agent_key=agent_key,
                    ticker=ticker,
                    original_length=len(content_str),
                    has_datablock="DATA_BLOCK" in content_str if content_str else False,
                    message="Insufficient output from quick LLM (thinking_level=low), retrying ONCE with deep thinking",
                )

                # Rebuild runnable with retry_llm (deep thinking)
                retry_runnable = (
                    prompt_template | retry_llm.bind_tools(tools)
                    if tools
                    else prompt_template | retry_llm
                )

                try:
                    retry_response = await invoke_with_rate_limit_handling(
                        retry_runnable,
                        {"messages": invocation_messages},
                        context=f"{agent_prompt.agent_name} (RETRY-HIGH)",
                    )

                    # CRITICAL: Tag retry response for parallel execution filtering
                    retry_response.name = agent_key

                    # Extract content from retry response
                    retry_content_str = extract_string_content(retry_response.content)

                    # Check if retry produced tool calls (continue tool loop)
                    retry_has_tool_calls = False
                    try:
                        if (
                            hasattr(retry_response, "tool_calls")
                            and retry_response.tool_calls
                        ):
                            retry_has_tool_calls = (
                                isinstance(retry_response.tool_calls, list)
                                and len(retry_response.tool_calls) > 0
                            )
                    except (AttributeError, TypeError):
                        pass

                    if retry_has_tool_calls:
                        # Retry produced tool calls - update state and continue tool loop
                        new_state["messages"] = [retry_response]
                        logger.info(
                            "analyst_retry_produced_tool_calls",
                            agent_key=agent_key,
                            ticker=ticker,
                            message="Retry with deep thinking produced tool calls, continuing tool loop",
                        )
                        return new_state

                    # Use retry content (even if still insufficient - no further retries)
                    logger.info(
                        "analyst_retry_complete",
                        agent_key=agent_key,
                        ticker=ticker,
                        original_length=len(content_str),
                        retry_length=len(retry_content_str),
                        retry_has_datablock=(
                            "DATA_BLOCK" in retry_content_str
                            if retry_content_str
                            else False
                        ),
                        retry_improved=len(retry_content_str) > len(content_str),
                    )
                    content_str = retry_content_str

                except Exception as retry_error:
                    # Retry failed - log and proceed with original (insufficient) output
                    logger.error(
                        "analyst_retry_failed",
                        agent_key=agent_key,
                        ticker=ticker,
                        error=str(retry_error),
                        message="Retry with deep thinking failed, using original output",
                    )
                    # Keep original content_str

            new_state[output_field] = content_str

            if agent_key == "fundamentals_analyst":
                logger.info(
                    "fundamentals_output",
                    has_datablock="DATA_BLOCK" in content_str,
                    length=len(content_str),
                )
            return new_state
        except Exception as e:
            logger.error(f"Analyst node error {output_field}: {str(e)}")
            return {
                "messages": [AIMessage(content=f"Error: {str(e)}")],
                output_field: f"Error: {str(e)}",
            }

    return analyst_node


def _extract_sector_from_state(state: dict) -> str:
    """Extract sector from fundamentals report DATA_BLOCK for lesson retrieval."""
    fundamentals = state.get("fundamentals_report", "") or ""
    if not fundamentals:
        return "Unknown"
    match = re.search(r"SECTOR:\s*(.+?)(?:\n|$)", fundamentals, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        if value.upper() not in ("N/A", "NA", "NONE", "-", ""):
            return value
    return "Unknown"


def create_researcher_node(
    llm, memory: Any | None, agent_key: str, round_num: int = 1
) -> Callable:
    """
    Create a researcher node for Bull/Bear debate.

    Args:
        llm: Language model instance
        memory: Memory instance for this researcher
        agent_key: "bull_researcher" or "bear_researcher"
        round_num: Debate round (1 or 2). Round 1 runs in parallel,
                   Round 2 includes opponent's R1 output for rebuttal.
    """
    is_bull = agent_key == "bull_researcher"
    researcher_type = "bull" if is_bull else "bear"
    opponent_type = "bear" if is_bull else "bull"

    async def researcher_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error(f"Missing prompt for researcher: {agent_key}")
            # Write to dedicated round field
            field_name = f"{researcher_type}_round{round_num}"
            return {
                "investment_debate_state": {
                    field_name: f"[SYSTEM]: Error - Missing prompt for {agent_key}.",
                    "count": state.get("investment_debate_state", {}).get("count", 0)
                    + 1,
                }
            }
        agent_name = agent_prompt.agent_name

        # Include all analyst reports for comprehensive debate context
        reports = f"""MARKET ANALYST REPORT:
{state.get("market_report", "N/A")}

SENTIMENT ANALYST REPORT:
{state.get("sentiment_report", "N/A")}

NEWS ANALYST REPORT:
{state.get("news_report", "N/A")}

FUNDAMENTALS ANALYST REPORT:
{state.get("fundamentals_report", "N/A")}"""

        # Build debate history context based on round
        debate_state = state.get("investment_debate_state", {})

        if round_num == 1:
            # Round 1: No opponent context (parallel execution)
            debate_history = ""
        else:
            # Round 2: Include opponent's Round 1 for rebuttal
            opponent_r1 = debate_state.get(f"{opponent_type}_round1", "")
            own_r1 = debate_state.get(f"{researcher_type}_round1", "")
            debate_history = f"""
=== ROUND 1 ARGUMENTS ===

YOUR ROUND 1 ARGUMENT:
{own_r1}

OPPONENT'S ROUND 1 ARGUMENT (REBUT THIS):
{opponent_r1}

=== END ROUND 1 ===

Now provide your Round 2 rebuttal, addressing the opponent's key points."""

        # Contextualize memory retrieval to prevent cross-contamination
        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        company_resolved = state.get("company_name_resolved", True)

        # Retrieve RELEVANT past insights for THIS ticker
        past_insights = ""
        if memory:
            try:
                relevant = await memory.query_similar_situations(
                    f"risks and upside for {ticker}",
                    n_results=3,
                    metadata_filter={"ticker": ticker},
                )
                if relevant:
                    past_insights = (
                        f"\n\nPAST MEMORY INSIGHTS (STRICTLY FOR {ticker}):\n"
                        + "\n".join([r["document"] for r in relevant])
                    )
                else:
                    logger.info("memory_no_exact_match", ticker=ticker)
                    past_insights = ""
            except Exception as e:
                logger.error("memory_retrieval_failed", ticker=ticker, error=str(e))
                past_insights = ""

        # Retrieve lessons from past retrospective evaluations (cross-ticker)
        lessons_text = ""
        try:
            from src.retrospective import (
                create_lessons_memory,
                format_lessons_for_injection,
            )

            lessons_memory = create_lessons_memory()
            sector = _extract_sector_from_state(state)
            lessons_text = await format_lessons_for_injection(
                lessons_memory, ticker, sector
            )
            if lessons_text:
                logger.info(
                    "lessons_injected",
                    agent=agent_key,
                    ticker=ticker,
                    lessons_length=len(lessons_text),
                )
            else:
                logger.debug(
                    "no_lessons_available",
                    agent=agent_key,
                    ticker=ticker,
                )
        except Exception as e:
            logger.warning("lessons_injection_failed", agent=agent_key, error=str(e))

        # Add Negative Constraint to prevent hallucination
        unresolved_warning = "" if company_resolved else f"\n{_UNRESOLVED_NAME_WARNING}"
        negative_constraint = f"""
CRITICAL INSTRUCTION:
You are analyzing **{ticker} ({company_name})**.{unresolved_warning}
If the provided context or memory contains information about a DIFFERENT company (e.g., from a previous analysis run), you MUST IGNORE IT.
Only use data explicitly related to {ticker} ({company_name}).
"""

        # Build prompt with round context
        round_instruction = (
            "Provide your initial argument."
            if round_num == 1
            else "Provide your rebuttal to the opponent's Round 1 argument."
        )

        # Combine past insights and lessons from retrospective
        context_block = past_insights
        if lessons_text:
            context_block += f"\n\n{lessons_text}"

        prompt = f"""{agent_prompt.system_message}\n{negative_constraint}\n\nREPORTS:\n{reports}\n{context_block}\n\nDEBATE CONTEXT:\n{debate_history}\n\n{round_instruction}"""

        try:
            response = await invoke_with_rate_limit_handling(
                llm,
                [HumanMessage(content=prompt)],
                context=f"{agent_name} R{round_num}",
            )
            content_str = extract_string_content(response.content)
            argument = f"{agent_name} (Round {round_num}): {content_str}"

            # Write to dedicated round field (parallel-safe)
            field_name = f"{researcher_type}_round{round_num}"

            logger.info(
                "researcher_completed",
                agent=agent_key,
                round=round_num,
                field=field_name,
                content_length=len(content_str),
            )

            return {
                "investment_debate_state": {
                    field_name: argument,
                }
            }
        except Exception as e:
            logger.error(f"Researcher error {agent_key} R{round_num}: {str(e)}")
            # Write error to the specific round field so debate sync doesn't
            # see an empty field and Research Manager gets diagnostic context.
            field_name = f"{researcher_type}_round{round_num}"
            return {
                "investment_debate_state": {
                    field_name: f"[SYSTEM ERROR]: {agent_key} R{round_num} failed - {str(e)}",
                }
            }

    return researcher_node


def create_valuation_calculator_node(llm) -> Callable:
    """
    Factory function creating Valuation Calculator node for chart generation.

    This lightweight agent extracts valuation PARAMETERS from DATA_BLOCK.
    It does NOT calculate targets - Python code does the math.

    Args:
        llm: LLM instance (should be QUICK_MODEL - simple extraction task)

    Returns:
        Async function compatible with LangGraph StateGraph.add_node()
    """

    async def valuation_calculator_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("valuation_calculator")
        if not agent_prompt:
            logger.error("Missing prompt for valuation_calculator")
            return {"valuation_params": ""}

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        fundamentals_report = state.get("fundamentals_report", "")

        # Normalize fundamentals_report to string
        if not isinstance(fundamentals_report, str):
            fundamentals_report = extract_string_content(fundamentals_report)

        if not fundamentals_report or "DATA_BLOCK" not in fundamentals_report:
            logger.warning(
                "valuation_calculator_no_datablock",
                ticker=ticker,
                message="No DATA_BLOCK found - skipping valuation params extraction",
            )
            return {"valuation_params": ""}

        # Extract just the DATA_BLOCK for the prompt (reduce token usage)
        data_block_pattern = (
            r"### --- START DATA_BLOCK[^\n]*---(.+?)### --- END DATA_BLOCK ---"
        )
        blocks = list(re.finditer(data_block_pattern, fundamentals_report, re.DOTALL))
        if not blocks:
            logger.warning(
                "valuation_calculator_datablock_regex_failed",
                ticker=ticker,
                message="DATA_BLOCK text present but regex extraction failed — skipping",
            )
            return {"valuation_params": ""}
        data_block = blocks[-1].group(0)

        prompt = f"""{agent_prompt.system_message}

TICKER: {ticker}
COMPANY: {company_name}

DATA_BLOCK:
{data_block}

Extract valuation parameters and output in the required format."""

        try:
            response = await invoke_with_rate_limit_handling(
                llm, [HumanMessage(content=prompt)], context=agent_prompt.agent_name
            )
            content_str = extract_string_content(response.content)

            logger.info(
                "valuation_calculator_complete",
                ticker=ticker,
                has_params_block="VALUATION_PARAMS" in content_str,
                content_length=len(content_str),
            )

            return {"valuation_params": content_str}

        except Exception as e:
            logger.error(f"Valuation calculator error for {ticker}: {str(e)}")
            return {"valuation_params": ""}

    return valuation_calculator_node


_STRICT_RM_ADDENDUM = """
---
## STRICT MODE — Research Manager Instruction

You are operating in STRICT mode. Apply this lens when synthesizing analyst inputs:

1. **Evidence over narrative**: Weight your synthesis toward concrete, documented facts
   (filed cash flows, signed contracts, observable margin trends) over projections or
   "could benefit from" language. If the bull case relies primarily on potential rather
   than demonstrated momentum, frame your synthesis toward DO_NOT_INITIATE.

2. **Near-term catalyst requirement**: A BUY recommendation requires at least one
   high-probability catalyst already in motion (≤12 months). "Eventual" or "possible"
   catalysts do not qualify in strict mode.

3. **Data vacuum discipline**: If the combined analyst reports show significant data gaps
   (missing OCF, unknown ownership structure, no analyst coverage data), flag them
   explicitly and weight toward caution — do not paper over gaps with qualitative reasoning.

4. **Bear argument weighting**: Give bear arguments proportionally more weight than in
   normal mode. The burden of proof is on the bull case in strict mode.
"""

_STRICT_PM_ADDENDUM = """
---
## STRICT MODE ACTIVE — Additional Screening Rules

Apply these as a final quality gate AFTER your normal Step 1A-1C analysis.
They OVERRIDE normal zone decisions where they conflict.

### AUTOMATIC DO_NOT_INITIATE (regardless of risk zone):

1. **Any PFIC flag** (PFIC_PROBABLE or PFIC_UNCERTAIN): → DO_NOT_INITIATE
2. **VIE structure detected**: → DO_NOT_INITIATE
3. **Value Trap HIGH** (score < 40 or TRAP verdict): → DO_NOT_INITIATE
4. **Risk tally ≥ 1.5** (vs normal 2.0 cutoff): → DO_NOT_INITIATE
5. **Data vacuum present** (missing sector, OCF, or analyst coverage): → DO_NOT_INITIATE
6. **Normal HOLD verdict**: → Upgrade to DO_NOT_INITIATE (no watchlist positions)

### TIGHTER BUY REQUIREMENTS (all must hold):
- Financial Health ≥ 60% (vs 50% normal)
- Growth Score ≥ 55% (vs 50% normal)
- Analyst Coverage ≤ 10 (vs 15 normal)
- P/E ≤ 15 (vs 18/25 contextual normal)
- Liquidity ≥ $750k daily USD (vs $500k normal)
- Graham Earnings Test: PASS
- Risk tally < 1.5

### POSITION SIZING:
- Maximum: 5% (vs 10% normal)
- Authoritarian regimes: MAX 1.5% (vs 2%)
- Any remaining uncertainty: MAX 2%
"""


def create_research_manager_node(
    llm, memory: Any | None, strict_mode: bool = False
) -> Callable:
    async def research_manager_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("research_manager")
        if not agent_prompt:
            return {"investment_plan": "Error: Missing prompt"}
        debate = state.get("investment_debate_state", {})
        value_trap = state.get("value_trap_report", "N/A")

        # JIT extraction for period/provenance awareness
        field_sources = extract_field_sources_from_messages(state.get("messages", []))
        attribution_note = ""
        if field_sources:
            sources_used = sorted(set(field_sources.values()))
            attribution_note = (
                f"\n\n### DATA PROVENANCE NOTE\n"
                f"Fundamentals sourced from: {', '.join(sources_used)}. "
                f"News may reflect more recent periods (e.g., Q3 headlines vs TTM API data). "
                f"When Bull/Bear cite conflicting figures, check if they reference different time periods."
            )

        all_reports = f"""MARKET ANALYST REPORT:\n{state.get("market_report", "N/A")}\n\nSENTIMENT ANALYST REPORT:\n{state.get("sentiment_report", "N/A")}\n\nNEWS ANALYST REPORT:\n{state.get("news_report", "N/A")}\n\nFUNDAMENTALS ANALYST REPORT:\n{state.get("fundamentals_report", "N/A")}{attribution_note}\n\nVALUE TRAP ANALYSIS:\n{value_trap}\n\nBULL RESEARCHER:\n{debate.get("bull_history", "N/A")}\n\nBEAR RESEARCHER:\n{debate.get("bear_history", "N/A")}"""
        system_msg = agent_prompt.system_message
        if strict_mode:
            system_msg = system_msg + _STRICT_RM_ADDENDUM
        prompt = f"""{system_msg}\n\n{all_reports}\n\nProvide Investment Plan."""
        try:
            response = await invoke_with_rate_limit_handling(
                llm, [HumanMessage(content=prompt)], context=agent_prompt.agent_name
            )
            # CRITICAL FIX: Normalize response.content to string
            content_str = extract_string_content(response.content)

            # Truncation detection and logging
            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent="research_manager")
            if trunc_info["truncated"]:
                logger.warning(
                    "agent_output_truncated",
                    agent="research_manager",
                    ticker=state.get("company_of_interest", "UNKNOWN"),
                    source=trunc_info["source"],
                    marker=trunc_info["marker"],
                    confidence=trunc_info["confidence"],
                    output_len=len(content_str),
                )

            return {"investment_plan": content_str}
        except Exception as e:
            return {"investment_plan": f"Error: {str(e)}"}

    return research_manager_node


def create_trader_node(llm, memory: Any | None) -> Callable:
    async def trader_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("trader")
        if not agent_prompt:
            return {"trader_investment_plan": "Error: Missing prompt"}

        # Include consultant review if available (external cross-validation)
        consultant = state.get("consultant_review", "")
        consultant_section = f"""\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n{consultant if consultant else "N/A (consultant disabled or unavailable)"}"""

        # Include valuation parameters if available (from Valuation Calculator)
        valuation = state.get("valuation_params", "")
        valuation_section = (
            f"""\n\nVALUATION PARAMETERS:\n{valuation}""" if valuation else ""
        )

        # Include all 4 analyst reports for comprehensive trading context
        all_input = f"""MARKET ANALYST REPORT:
{state.get("market_report", "N/A")}

SENTIMENT ANALYST REPORT:
{state.get("sentiment_report", "N/A")}

NEWS ANALYST REPORT:
{state.get("news_report", "N/A")}

FUNDAMENTALS ANALYST REPORT:
{state.get("fundamentals_report", "N/A")}

RESEARCH MANAGER PLAN:
{state.get("investment_plan", "N/A")}{consultant_section}{valuation_section}"""
        prompt = (
            f"""{agent_prompt.system_message}\n\n{all_input}\n\nCreate Trading Plan."""
        )
        try:
            response = await invoke_with_rate_limit_handling(
                llm, [HumanMessage(content=prompt)], context=agent_prompt.agent_name
            )
            # CRITICAL FIX: Normalize response.content to string
            return {"trader_investment_plan": extract_string_content(response.content)}
        except Exception as e:
            return {"trader_investment_plan": f"Error: {str(e)}"}

    return trader_node


def create_risk_debater_node(llm, agent_key: str) -> Callable:
    # Map agent_key to the dedicated field for parallel-safe writes
    AGENT_FIELD_MAP = {
        "risky_analyst": "current_risky_response",
        "safe_analyst": "current_safe_response",
        "neutral_analyst": "current_neutral_response",
    }

    async def risk_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        field_name = AGENT_FIELD_MAP.get(agent_key, "history")

        if not agent_prompt:
            # Return only the field this agent writes to (parallel-safe)
            return {
                "risk_debate_state": {
                    field_name: f"[SYSTEM]: Error - Missing prompt for {agent_key}",
                    "latest_speaker": agent_key,
                }
            }

        # Include consultant review if available (external cross-validation)
        consultant = state.get("consultant_review", "")
        consultant_section = f"""\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n{consultant if consultant else "N/A (consultant disabled or unavailable)"}"""

        prompt = f"""{agent_prompt.system_message}\n\nTRADER PLAN: {state.get("trader_investment_plan")}{consultant_section}\n\nProvide risk assessment."""
        try:
            response = await invoke_with_rate_limit_handling(
                llm, [HumanMessage(content=prompt)], context=agent_prompt.agent_name
            )
            # CRITICAL FIX: Normalize response.content to string
            content_str = extract_string_content(response.content)

            # Each analyst writes ONLY to its dedicated field (parallel-safe)
            # The merge_risk_state reducer will combine all three
            return {
                "risk_debate_state": {
                    field_name: f"{agent_prompt.agent_name}: {content_str}",
                    "latest_speaker": agent_prompt.agent_name,
                }
            }
        except Exception as e:
            return {
                "risk_debate_state": {
                    field_name: f"[ERROR]: {agent_key} failed - {str(e)}",
                    "latest_speaker": agent_key,
                }
            }

    return risk_node


def create_portfolio_manager_node(
    llm, memory: Any | None, strict_mode: bool = False
) -> Callable:
    async def pm_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("portfolio_manager")
        if not agent_prompt:
            return {"final_trade_decision": "Error: Missing prompt"}
        market = state.get("market_report", "")
        sentiment = state.get("sentiment_report", "")
        news = state.get("news_report", "")
        fundamentals = state.get("fundamentals_report", "")
        value_trap = state.get("value_trap_report", "")
        inv_plan = state.get("investment_plan", "")
        consultant = state.get("consultant_review", "")
        trader = state.get("trader_investment_plan", "")

        # Read from all three dedicated risk fields (parallel-safe)
        risk_state = state.get("risk_debate_state", {})
        risky_view = risk_state.get("current_risky_response", "")
        safe_view = risk_state.get("current_safe_response", "")
        neutral_view = risk_state.get("current_neutral_response", "")
        # Combine all three perspectives for the PM
        risk = f"""RISKY ANALYST (Aggressive):\n{risky_view if risky_view else "N/A"}

SAFE ANALYST (Conservative):\n{safe_view if safe_view else "N/A"}

NEUTRAL ANALYST (Balanced):\n{neutral_view if neutral_view else "N/A"}"""

        # Red-flag pre-screening results
        pre_screening_result = state.get("pre_screening_result", "N/A")
        red_flags = list(state.get("red_flags", []))  # Copy to avoid mutating state

        # --- Flag Detection (Risk Bonuses & Penalties) ---
        from src.validators.red_flag_detector import RedFlagDetector

        # --- Value Trap Flag Detection ---
        # Value Trap Detector runs in parallel, so flags are detected here at PM stage
        if value_trap:
            value_trap_warnings = RedFlagDetector.detect_value_trap_flags(
                value_trap, state.get("company_of_interest", "UNKNOWN")
            )
            if value_trap_warnings:
                red_flags.extend(value_trap_warnings)
                logger.info(
                    "value_trap_warnings_detected",
                    ticker=state.get("company_of_interest", "UNKNOWN"),
                    warning_types=[w["type"] for w in value_trap_warnings],
                    total_risk_penalty=sum(
                        w.get("risk_penalty", 0) for w in value_trap_warnings
                    ),
                )

        # --- Moat Signal Detection (Risk Bonuses) ---
        # Moat signals from fundamentals provide risk bonuses (negative penalties)
        if fundamentals:
            moat_bonuses = RedFlagDetector.detect_moat_flags(
                fundamentals, state.get("company_of_interest", "UNKNOWN")
            )
            if moat_bonuses:
                red_flags.extend(moat_bonuses)
                logger.info(
                    "moat_bonuses_detected",
                    ticker=state.get("company_of_interest", "UNKNOWN"),
                    bonus_types=[b["type"] for b in moat_bonuses],
                    total_risk_bonus=sum(
                        b.get("risk_penalty", 0) for b in moat_bonuses
                    ),
                )

        # --- Capital Efficiency Detection (Leverage Quality) ---
        # Separate from moat signals - detects value destruction and leverage engineering
        if fundamentals:
            capital_flags = RedFlagDetector.detect_capital_efficiency_flags(
                fundamentals, state.get("company_of_interest", "UNKNOWN")
            )
            if capital_flags:
                red_flags.extend(capital_flags)
                logger.info(
                    "capital_efficiency_flags_detected",
                    ticker=state.get("company_of_interest", "UNKNOWN"),
                    flag_types=[f["type"] for f in capital_flags],
                    total_risk_adjustment=sum(
                        f.get("risk_penalty", 0) for f in capital_flags
                    ),
                )

        # --- Consultant Condition Enforcement ---
        # Parse consultant verdict and enforce conditions as risk penalties
        consultant_review = state.get("consultant_review", "")
        if consultant_review:
            if not isinstance(consultant_review, str):
                consultant_review = extract_string_content(consultant_review)
            consultant_conditions = RedFlagDetector.parse_consultant_conditions(
                consultant_review
            )
            consultant_flags = RedFlagDetector.detect_consultant_flags(
                consultant_conditions,
                state.get("company_of_interest", "UNKNOWN"),
            )
            if consultant_flags:
                red_flags.extend(consultant_flags)
                logger.info(
                    "consultant_flags_detected",
                    ticker=state.get("company_of_interest", "UNKNOWN"),
                    flag_types=[f["type"] for f in consultant_flags],
                    total_risk_penalty=sum(
                        f.get("risk_penalty", 0) for f in consultant_flags
                    ),
                )

        logger.info(
            "pm_inputs",
            has_market=bool(market),
            has_sentiment=bool(sentiment),
            has_news=bool(news),
            has_fundamentals=bool(fundamentals),
            has_value_trap=bool(value_trap),
            has_consultant=bool(consultant),
            has_datablock="DATA_BLOCK" in fundamentals if fundamentals else False,
            fund_len=len(fundamentals) if fundamentals else 0,
            value_trap_len=len(value_trap) if value_trap else 0,
            red_flags_count=len(red_flags),
        )

        # JIT extraction of source attribution and conflicts for adjudication
        field_sources = extract_field_sources_from_messages(state.get("messages", []))
        attribution_table = format_attribution_table(field_sources)
        conflict_table = format_conflict_table(state.get("messages", []))

        # Include consultant review in context (if available)
        consultant_section = f"""\n\nEXTERNAL CONSULTANT REVIEW (Cross-Validation):\n{consultant if consultant else "N/A (consultant disabled or unavailable)"}"""

        # Include red-flag pre-screening results (critical safety gate)
        red_flag_section = f"""\n\nRED-FLAG PRE-SCREENING:\nPre-Screening Result: {pre_screening_result}"""
        if red_flags:
            red_flag_list = "\n".join(
                [
                    f"  - {flag.get('type', 'Unknown')}: {flag.get('detail', 'No detail')}"
                    for flag in red_flags
                ]
            )
            red_flag_section += f"\nRed Flags/Warnings Detected:\n{red_flag_list}"
        else:
            red_flag_section += "\nRed Flags Detected: None"

        # Summarize verbose reports to reduce PM input context
        # This improves output completeness by leaving more "attention budget"
        all_context = f"""MARKET ANALYST REPORT:
{summarize_for_pm(market, "market", 2500) if market else "N/A"}

SENTIMENT ANALYST REPORT:
{summarize_for_pm(sentiment, "sentiment", 1500) if sentiment else "N/A"}

NEWS ANALYST REPORT:
{summarize_for_pm(news, "news", 2000) if news else "N/A"}

FUNDAMENTALS ANALYST REPORT:
{summarize_for_pm(fundamentals, "fundamentals", 6000) if fundamentals else "N/A"}{attribution_table}{conflict_table}

VALUE TRAP ANALYSIS:
{extract_value_trap_verdict(value_trap)}{summarize_for_pm(value_trap, "value_trap", 2500) if value_trap else "N/A"}{red_flag_section}

RESEARCH MANAGER RECOMMENDATION:
{summarize_for_pm(inv_plan, "research", 3000) if inv_plan else "N/A"}{consultant_section}

TRADER PROPOSAL:
{summarize_for_pm(trader, "trader", 2000) if trader else "N/A"}

RISK TEAM DEBATE:
{risk if risk else "N/A"}"""
        pm_system_msg = agent_prompt.system_message
        if strict_mode:
            pm_system_msg = pm_system_msg + _STRICT_PM_ADDENDUM
        prompt = (
            f"""{pm_system_msg}\n\n{all_context}\n\nMake Portfolio Manager Verdict."""
        )
        try:
            response = await invoke_with_rate_limit_handling(
                llm, [HumanMessage(content=prompt)], context=agent_prompt.agent_name
            )
            # CRITICAL FIX: Normalize response.content to string
            content_str = extract_string_content(response.content)

            # Truncation detection and logging
            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent="portfolio_manager")
            if trunc_info["truncated"]:
                logger.warning(
                    "agent_output_truncated",
                    agent="portfolio_manager",
                    ticker=state.get("company_of_interest", "UNKNOWN"),
                    source=trunc_info["source"],
                    marker=trunc_info["marker"],
                    confidence=trunc_info["confidence"],
                    output_len=len(content_str),
                )

            return {"final_trade_decision": content_str}
        except Exception as e:
            logger.error(f"PM error: {str(e)}")
            return {"final_trade_decision": f"Error: {str(e)}"}

    return pm_node


def create_consultant_node(
    llm, agent_key: str = "consultant", tools: list | None = None
) -> Callable:
    """
    Factory function creating external consultant node for cross-validation.

    Uses a different LLM (OpenAI) to review Gemini's analysis outputs and detect
    biases, groupthink, and factual errors that internal agents may miss.

    Supports optional tools (e.g., spot_check_metric) with a bounded tool loop
    to break the circular dependency where all agents rely on the same data pipeline.

    Args:
        llm: LLM instance (typically OpenAI ChatGPT for cross-validation)
        agent_key: Agent key for prompt lookup (default: "consultant")
        tools: Optional list of verification tools for independent data checks

    Returns:
        Async function compatible with LangGraph StateGraph.add_node()
    """
    MAX_TOOL_ITERATIONS = 3
    MAX_TOOL_CALLS_PER_TURN = 4

    # Pre-bind tools if available
    tools_by_name = {t.name: t for t in tools} if tools else {}
    llm_with_tools = llm.bind_tools(tools) if tools else None

    async def consultant_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt(agent_key)
        if not agent_prompt:
            logger.error(f"Missing prompt for consultant: {agent_key}")
            return {
                "consultant_review": "Error: Missing consultant prompt configuration"
            }

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        company_resolved = state.get("company_name_resolved", True)

        context = get_context_from_config(config)
        current_date = (
            context.trade_date if context else datetime.now().strftime("%Y-%m-%d")
        )

        # Assemble complete context (everything the Research Manager saw + the synthesis)
        # Safely extract debate history (handle None/missing debate state)
        debate_state = state.get("investment_debate_state")
        debate_history = "N/A"
        if debate_state and isinstance(debate_state, dict):
            debate_history = debate_state.get("history", "N/A")
        elif debate_state is None:
            # DIAGNOSTIC: Log when debate state is unexpectedly None
            # This shouldn't happen in normal execution (consultant runs after debate)
            # If this occurs, it indicates a potential LangGraph state propagation issue
            ticker = state.get("company_of_interest", "UNKNOWN")
            logger.error(
                "consultant_received_none_debate_state",
                ticker=ticker,
                message="Consultant node received None debate state - this may indicate a graph execution bug or fast-fail path issue",
            )
            debate_history = "[SYSTEM DIAGNOSTIC: Debate state unexpectedly None. This may indicate the debate was skipped (fast-fail path) or a state propagation issue. Consultant cross-validation may be limited without debate context.]"

        # JIT extraction of source attribution and conflicts for adjudication
        field_sources = extract_field_sources_from_messages(state.get("messages", []))
        attribution_table = format_attribution_table(field_sources)
        conflict_table = format_conflict_table(state.get("messages", []))

        # Phase 2.2: Summarize inputs to reduce context size and prevent truncation
        market = state.get("market_report", "N/A")
        sentiment = state.get("sentiment_report", "N/A")
        news = state.get("news_report", "N/A")
        fundamentals = state.get("fundamentals_report", "N/A")
        investment_plan = state.get("investment_plan", "N/A")
        auditor = state.get("auditor_report", "N/A")

        all_context = f"""
=== ANALYST REPORTS (SOURCE DATA) ===

MARKET ANALYST REPORT:
{summarize_for_pm(market, "market", 2000) if market != "N/A" else "N/A"}

SENTIMENT ANALYST REPORT:
{summarize_for_pm(sentiment, "sentiment", 1500) if sentiment != "N/A" else "N/A"}

NEWS ANALYST REPORT:
{summarize_for_pm(news, "news", 2000) if news != "N/A" else "N/A"}

FUNDAMENTALS ANALYST REPORT:
{summarize_for_pm(fundamentals, "fundamentals", 5000) if fundamentals != "N/A" else "N/A"}
{attribution_table}{conflict_table}
=== BULL/BEAR DEBATE HISTORY ===

{summarize_for_pm(debate_history, "debate", 4000) if debate_history != "N/A" else "N/A"}

=== RESEARCH MANAGER SYNTHESIS ===

{summarize_for_pm(investment_plan, "research", 4000) if investment_plan != "N/A" else "N/A"}

=== RED FLAGS (Pre-Screening Results) ===

Red Flags Detected: {state.get("red_flags", [])}
Pre-Screening Result: {state.get("pre_screening_result", "UNKNOWN")}

=== INDEPENDENT FORENSIC AUDIT ===
{summarize_for_pm(auditor, "auditor", 3000) if auditor != "N/A" else "N/A"}
"""

        company_warning = "" if company_resolved else f"\n{_UNRESOLVED_NAME_WARNING}"
        prompt = f"""{agent_prompt.system_message}

ANALYSIS DATE: {_format_date_with_fy_hint(current_date)}
TICKER: {ticker}
COMPANY: {company_name}{company_warning}

{all_context}

Provide your independent consultant review."""

        try:
            # Bounded tool loop for independent verification
            from langchain_core.messages import ToolMessage as TM

            messages = [HumanMessage(content=prompt)]
            active_llm = llm_with_tools or llm
            content_str = ""

            for iteration in range(MAX_TOOL_ITERATIONS + 1):
                response = await invoke_with_rate_limit_handling(
                    active_llm, messages, context=agent_prompt.agent_name
                )

                tool_calls = getattr(response, "tool_calls", None)
                # Ensure tool_calls is a real list (not a Mock or other truthy object)
                if (
                    not isinstance(tool_calls, list)
                    or not tool_calls
                    or iteration == MAX_TOOL_ITERATIONS
                ):
                    # Final response — no more tool calls (or cap reached)
                    content_str = extract_string_content(response.content)
                    break

                # Execute tool calls (capped per turn)
                messages.append(response)
                capped = tool_calls[:MAX_TOOL_CALLS_PER_TURN]
                if len(tool_calls) > MAX_TOOL_CALLS_PER_TURN:
                    logger.warning(
                        "consultant_tool_calls_capped",
                        ticker=ticker,
                        requested=len(tool_calls),
                        cap=MAX_TOOL_CALLS_PER_TURN,
                    )

                for tc in capped:
                    tool_fn = tools_by_name.get(tc["name"])
                    if tool_fn:
                        try:
                            result = await tool_fn.ainvoke(tc["args"])
                        except Exception as tool_err:
                            result = f"TOOL_ERROR: {str(tool_err)}"
                    else:
                        result = f"Unknown tool: {tc['name']}"
                    messages.append(TM(content=str(result), tool_call_id=tc["id"]))

                # Append SKIPPED messages for overflow tool calls
                for tc in tool_calls[MAX_TOOL_CALLS_PER_TURN:]:
                    skip_id = tc.get("id", f"skip_{tc['name']}")
                    messages.append(
                        TM(
                            content="SKIPPED: Too many tool calls in one turn.",
                            tool_call_id=skip_id,
                        )
                    )

                logger.info(
                    "consultant_tool_iteration",
                    ticker=ticker,
                    iteration=iteration + 1,
                    tools_called=[tc["name"] for tc in capped],
                )

            if not content_str:
                # Safety valve: invoke without tools to force text response
                response = await invoke_with_rate_limit_handling(
                    llm, messages, context=agent_prompt.agent_name
                )
                content_str = extract_string_content(response.content)

            # Truncation detection and logging
            from src.utils import detect_truncation

            trunc_info = detect_truncation(content_str, agent="consultant")
            if trunc_info["truncated"]:
                logger.warning(
                    "agent_output_truncated",
                    agent="consultant",
                    ticker=ticker,
                    source=trunc_info["source"],
                    marker=trunc_info["marker"],
                    confidence=trunc_info["confidence"],
                    output_len=len(content_str),
                )

            logger.info(
                "consultant_review_complete",
                ticker=ticker,
                review_length=len(content_str),
                has_errors="ERROR" in content_str.upper()
                or "FAIL" in content_str.upper(),
                truncated=trunc_info["truncated"],
            )

            return {"consultant_review": content_str}

        except Exception as e:
            logger.error(f"Consultant node error for {ticker}: {str(e)}")
            # Return error but don't block the graph
            return {
                "consultant_review": f"Consultant Review Error: {str(e)}\n\nNote: Analysis will proceed without external validation."
            }

    return consultant_node


def create_legal_counsel_node(llm, tools: list) -> Callable:
    """
    Factory function creating Legal Counsel node for PFIC/VIE detection.

    Lightweight agent that runs parallel to Foreign Language Analyst.
    Outputs structured JSON for deterministic parsing by Red Flag Detector.

    Args:
        llm: LLM instance (should be QUICK_MODEL with thinking_level=low)
        tools: List containing search_legal_tax_disclosures tool

    Returns:
        Async function compatible with LangGraph StateGraph.add_node()
    """

    async def legal_counsel_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("legal_counsel")
        if not agent_prompt:
            logger.error("Missing prompt for legal_counsel")
            return {
                "legal_report": json.dumps({"error": "Missing legal_counsel prompt"})
            }

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        company_resolved = state.get("company_name_resolved", True)

        context = get_context_from_config(config)
        current_date = (
            context.trade_date if context else datetime.now().strftime("%Y-%m-%d")
        )

        # Extract sector and country from Junior's raw data (if available)
        raw_data = state.get("raw_fundamentals_data", "")
        sector, country = _extract_sector_country(raw_data)

        company_warning = "" if company_resolved else f"\n{_UNRESOLVED_NAME_WARNING}"
        human_msg = f"""Analyze legal/tax risks for:
Ticker: {ticker}
Company: {company_name}{company_warning}
Sector: {sector}
Country: {country}
Date: {_format_date_with_fy_hint(current_date)}

Call the search_legal_tax_disclosures tool with these parameters, then provide your JSON assessment."""

        try:
            # Create agent with tools
            agent = create_react_agent(llm, tools)
            result = await agent.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=agent_prompt.system_message),
                        HumanMessage(content=human_msg),
                    ]
                }
            )

            # Extract final response
            response = result["messages"][-1].content
            response_str = extract_string_content(response)

            # Validate JSON output
            try:
                # Try to parse as JSON directly
                parsed = json.loads(response_str)
                logger.info(
                    "legal_counsel_complete",
                    ticker=ticker,
                    pfic_status=parsed.get("pfic_status"),
                    vie_structure=parsed.get("vie_structure"),
                )
                return {"legal_report": response_str, "sender": "legal_counsel"}

            except json.JSONDecodeError:
                # Try to extract JSON from response (may be wrapped in text)
                json_match = re.search(
                    r'\{[^{}]*"pfic_status"[^{}]*\}', response_str, re.DOTALL
                )
                if json_match:
                    extracted = json_match.group()
                    try:
                        json.loads(extracted)  # Validate
                        logger.info("legal_counsel_extracted_json", ticker=ticker)
                        return {"legal_report": extracted, "sender": "legal_counsel"}
                    except json.JSONDecodeError:
                        pass

                # Return raw response wrapped in error JSON
                logger.warning(
                    "legal_counsel_invalid_json",
                    ticker=ticker,
                    response_preview=response_str[:200],
                )
                return {
                    "legal_report": json.dumps(
                        {
                            "error": "Invalid JSON response",
                            "raw_response": response_str[:500],
                            "pfic_status": "UNCERTAIN",
                            "vie_structure": "N/A",
                        }
                    ),
                    "sender": "legal_counsel",
                }

        except Exception as e:
            logger.error("legal_counsel_error", ticker=ticker, error=str(e))
            return {
                "legal_report": json.dumps(
                    {
                        "error": str(e),
                        "pfic_status": "UNCERTAIN",
                        "vie_structure": "N/A",
                    }
                ),
                "sender": "legal_counsel",
            }

    return legal_counsel_node


def create_auditor_node(llm, tools: list) -> Callable:
    """
    Factory function creating the Global Forensic Auditor node.

    This agent runs in parallel with other analysts but remains completely independent.
    Its output is used ONLY by the Consultant agent for cross-validation.

    Includes tool output truncation to prevent context overflow with OpenAI's 128k limit.
    The truncation preserves head + tail of large outputs to maintain JSON structure.
    """
    # Max chars per tool output (~16k tokens at 4 chars/token, safe with 3-5 calls)
    # 63.5k chars × 5 calls = 317.5k chars ≈ 80k tokens; still under OpenAI's 128k limit
    MAX_TOOL_OUTPUT_CHARS = 63500

    def truncate_tool_outputs_hook(state: dict) -> dict:
        """
        Pre-model hook that truncates large tool outputs before LLM invocation.
        Called before each LLM call within the ReAct loop.

        Returns llm_input_messages (not messages) to avoid modifying state history.
        Preserves head + tail to maintain JSON structure validity.
        """
        from langchain_core.messages import ToolMessage

        messages = state.get("messages", [])
        modified = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                if len(content) > MAX_TOOL_OUTPUT_CHARS:
                    # Keep head (structure/main data) + tail (summaries/totals)
                    # Preserve more tail as financial reports often have key summaries at end
                    head_size = 58000
                    tail_size = 5500
                    truncated_chars = len(content) - head_size - tail_size
                    truncated = (
                        content[:head_size]
                        + f"\n\n[...TRUNCATED {truncated_chars:,} chars...]\n"
                        + "[NOTE: Data truncated due to size limits. Partial analysis may still be useful. "
                        + "Key financial metrics may appear in head or tail sections above/below.]\n\n"
                        + content[-tail_size:]
                    )
                    modified.append(
                        ToolMessage(
                            content=truncated,
                            tool_call_id=msg.tool_call_id,
                            name=getattr(msg, "name", None),
                        )
                    )
                    logger.debug(
                        "auditor_tool_output_truncated",
                        original_len=len(content),
                        truncated_len=len(truncated),
                    )
                else:
                    modified.append(msg)
            else:
                modified.append(msg)
        # Return llm_input_messages to avoid updating state history
        return {"llm_input_messages": modified}

    async def auditor_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        from src.prompts import get_prompt

        agent_prompt = get_prompt("global_forensic_auditor")
        if not agent_prompt:
            logger.error("Missing prompt for global_forensic_auditor")
            return {"auditor_report": "Error: Missing prompt"}

        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)
        company_resolved = state.get("company_name_resolved", True)

        context = get_context_from_config(config)
        current_date = (
            context.trade_date if context else datetime.now().strftime("%Y-%m-%d")
        )

        # Only provide basic identity info to ensure independence
        company_warning = "" if company_resolved else f"\n{_UNRESOLVED_NAME_WARNING}"
        human_msg = f"""Analyze financial statements for:
Ticker: {ticker}
Company: {company_name}{company_warning}
Date: {_format_date_with_fy_hint(current_date)}

Perform a forensic audit using your tools."""

        logger.info("auditor_start", ticker=ticker)

        try:
            # Create agent with pre_model_hook for proactive truncation
            # This prevents context overflow by truncating large tool outputs
            # before they're sent back to the LLM on each ReAct iteration
            agent = create_react_agent(
                llm,
                tools,
                pre_model_hook=truncate_tool_outputs_hook,
            )
            result = await agent.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=agent_prompt.system_message),
                        HumanMessage(content=human_msg),
                    ]
                },
                config={
                    "recursion_limit": 12
                },  # Reduced from 25 - fail faster on retry loops
            )

            response = result["messages"][-1].content
            response_str = extract_string_content(response)

            logger.info("auditor_complete", ticker=ticker, length=len(response_str))

            return {"auditor_report": response_str, "sender": "global_forensic_auditor"}

        except Exception as e:
            error_str = str(e)
            logger.error("auditor_error", ticker=ticker, error=error_str)

            # Classify the error for appropriate recovery
            is_context_error = (
                "context_length_exceeded" in error_str
                or "maximum context length" in error_str
            )
            is_param_error = (
                "does not support" in error_str
                or "Unsupported value" in error_str
                or "invalid_request_error" in error_str
            )

            if is_context_error:
                graceful_msg = f"""## FORENSIC AUDITOR REPORT

**STATUS**: CONTEXT_LIMIT_EXCEEDED

**Reason**: Tool results exceeded capacity even after truncation.

**Recommendation**:
Downstream agents should rely on Fundamentals Analyst DATA_BLOCK (structured APIs: yfinance, FMP, EODHD) as primary source. Independent forensic audit unavailable for {ticker}.

---
FORENSIC_DATA_BLOCK:
STATUS: UNAVAILABLE
META: CONTEXT_LIMIT_EXCEEDED
REASON: Data volume exceeded 128k token limit
VERDICT: Rely on DATA_BLOCK metrics for {ticker}.
"""
                return {
                    "auditor_report": graceful_msg,
                    "sender": "global_forensic_auditor",
                }

            if is_param_error:
                # Model rejected a parameter (temperature, max_tokens, etc.)
                # Retry with a safe fallback configuration
                logger.warning(
                    "auditor_param_error_retry",
                    ticker=ticker,
                    error=error_str,
                )
                try:
                    from langchain_openai import ChatOpenAI

                    fallback_llm = ChatOpenAI(
                        model=llm.model_name,
                        timeout=120,
                        max_retries=3,
                        streaming=False,
                        # Omit temperature and max_tokens — use model defaults
                    )
                    agent = create_react_agent(
                        fallback_llm,
                        tools,
                        pre_model_hook=truncate_tool_outputs_hook,
                    )
                    result = await agent.ainvoke(
                        {
                            "messages": [
                                SystemMessage(content=agent_prompt.system_message),
                                HumanMessage(content=human_msg),
                            ]
                        },
                        config={"recursion_limit": 12},
                    )
                    response = result["messages"][-1].content
                    response_str = extract_string_content(response)
                    logger.info(
                        "auditor_complete_after_retry",
                        ticker=ticker,
                        length=len(response_str),
                    )
                    return {
                        "auditor_report": response_str,
                        "sender": "global_forensic_auditor",
                    }
                except Exception as retry_e:
                    logger.error(
                        "auditor_retry_failed",
                        ticker=ticker,
                        error=str(retry_e),
                    )

            return {
                "auditor_report": f"Auditor Error: {error_str}",
                "sender": "global_forensic_auditor",
            }

    return auditor_node


def _extract_sector_country(raw_data: str) -> tuple:
    """
    Extract sector and country from Junior Analyst's raw JSON output.

    Args:
        raw_data: Raw fundamentals data string from Junior Analyst

    Returns:
        Tuple of (sector, country) strings
    """
    sector, country = "Unknown", "Unknown"

    if not raw_data:
        return sector, country

    try:
        # Try to find JSON block in raw data
        sector_match = re.search(r'"sector"\s*:\s*"([^"]+)"', raw_data, re.IGNORECASE)
        if sector_match:
            sector = sector_match.group(1)

        country_match = re.search(r'"country"\s*:\s*"([^"]+)"', raw_data, re.IGNORECASE)
        if country_match:
            country = country_match.group(1)

        # Fallback: try to find industry if sector not found
        if sector == "Unknown":
            industry_match = re.search(
                r'"industry"\s*:\s*"([^"]+)"', raw_data, re.IGNORECASE
            )
            if industry_match:
                sector = industry_match.group(1)

    except Exception as e:
        logger.debug(f"Error extracting sector/country: {e}")

    return sector, country


def create_state_cleaner_node() -> Callable:
    async def clean_state(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        context = get_context_from_config(config)
        ticker = (
            context.ticker if context else state.get("company_of_interest", "UNKNOWN")
        )

        logger.debug(
            "State cleaner running",
            context_ticker=context.ticker if context else None,
            state_ticker=state.get("company_of_interest"),
            final_ticker=ticker,
        )

        return {
            "messages": [HumanMessage(content=f"Analyze {ticker}")],
            "tools_called": state.get("tools_called", {}),
            "company_of_interest": ticker,
        }

    return clean_state


# --- Red-Flag Detection System ---


def create_financial_health_validator_node(strict_mode: bool = False) -> Callable:
    """
    Factory function creating a pre-screening validator node to catch extreme financial risks
    before proceeding to bull/bear debate.

    This validator implements a "red-flag detection" pattern to save token costs and enforce
    financial discipline. Uses deterministic threshold-based logic from RedFlagDetector.

    Why code-driven instead of LLM-driven:
    - Exact thresholds required (D/E > 500%, not "very high")
    - Fast-fail pattern (avoid LLM calls for doomed stocks)
    - Reliability (no hallucination risk on number parsing)
    - Cost savings (~60% token reduction for rejected stocks)

    Architecture integration:
    - Runs AFTER Fundamentals Analyst (has data)
    - Runs BEFORE Bull/Bear Debate (saves cost if doomed)
    - Sets state.pre_screening_result = "REJECT" | "PASS"
    - Graph routing: REJECT → Portfolio Manager (skip debate)
                     PASS → Bull Researcher (normal flow)

    Returns:
        Async function compatible with LangGraph StateGraph.add_node()
    """

    async def financial_health_validator_node(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        """
        Pre-screening layer to catch extreme financial risks before detailed scoring.

        Delegates to RedFlagDetector for deterministic validation logic.

        Args:
            state: Current agent state with fundamentals_report populated
            config: Runtime configuration (not currently used)

        Returns:
            Updated state dict with:
            - red_flags: List of detected red flags (severity, type, detail, action)
            - pre_screening_result: "REJECT" if any AUTO_REJECT flags, else "PASS"
        """
        ticker = state.get("company_of_interest", "UNKNOWN")
        company_name = state.get("company_name", ticker)

        try:
            from src.validators.red_flag_detector import RedFlagDetector

            fundamentals_report = state.get("fundamentals_report", "")

            # --- FIX: DEFENSIVE HANDLING FOR NON-STRING STATE VALUES ---
            # LangGraph can sometimes pass accumulated list of state updates,
            # or Gemini may return dict/structured content instead of string.
            # Normalize to string using the helper function.
            if not isinstance(fundamentals_report, str):
                fundamentals_report = extract_string_content(fundamentals_report)

            quiet_mode = settings_config.quiet_mode

            # Graceful handling of missing fundamentals
            if not fundamentals_report:
                logger.warning(
                    "validator_no_fundamentals",
                    ticker=ticker,
                    message="No fundamentals report available - skipping pre-screening",
                )
                return {"red_flags": [], "pre_screening_result": "PASS"}

            # Extract sector classification from fundamentals report
            sector = RedFlagDetector.detect_sector(fundamentals_report)

            # Extract metrics from DATA_BLOCK
            metrics = RedFlagDetector.extract_metrics(fundamentals_report)

            # --- DATA QUALITY CHECK ---
            # If DATA_BLOCK is missing or completely unparseable, all metrics
            # will be None. This means every threshold check in detect_red_flags
            # is silently skipped, producing a false PASS. Flag this explicitly.
            has_data_block = "### --- START DATA_BLOCK" in fundamentals_report
            core_metrics = [
                metrics.get("debt_to_equity"),
                metrics.get("net_income"),
                metrics.get("fcf"),
                metrics.get("adjusted_health_score"),
            ]
            if not has_data_block or all(m is None for m in core_metrics):
                logger.warning(
                    "validator_no_usable_metrics",
                    ticker=ticker,
                    has_data_block=has_data_block,
                    message="DATA_BLOCK missing or unparseable — cannot validate financial health",
                )
                return {
                    "red_flags": [
                        {
                            "type": "DATA_QUALITY_WARNING",
                            "severity": "WARNING",
                            "detail": "DATA_BLOCK missing or unparseable in fundamentals report; "
                            "financial health checks could not be performed",
                            "action": "RISK_PENALTY",
                            "risk_penalty": 1.0,
                            "rationale": "Pre-screening was unable to verify financial health "
                            "due to missing structured data. Proceeding with caution.",
                        }
                    ],
                    "pre_screening_result": "PASS",
                }

            # Log extracted metrics (unless in quiet mode)
            if not quiet_mode:
                logger.info(
                    "validator_extracted_metrics",
                    ticker=ticker,
                    sector=sector.value,
                    debt_to_equity=metrics.get("debt_to_equity"),
                    fcf=metrics.get("fcf"),
                    net_income=metrics.get("net_income"),
                    interest_coverage=metrics.get("interest_coverage"),
                    adjusted_health_score=metrics.get("adjusted_health_score"),
                )

            # Apply sector-aware red-flag detection logic
            red_flags, pre_screening_result = RedFlagDetector.detect_red_flags(
                metrics, ticker, sector, strict_mode=strict_mode
            )

            # --- Legal/Tax Flag Detection ---
            # Extract and process legal_report for PFIC/VIE warnings
            # These are WARNING flags (risk penalty) not AUTO_REJECT flags
            legal_report = state.get("legal_report", "")
            if legal_report:
                if not isinstance(legal_report, str):
                    legal_report = extract_string_content(legal_report)

                legal_risks = RedFlagDetector.extract_legal_risks(legal_report)
                legal_warnings = RedFlagDetector.detect_legal_flags(legal_risks, ticker)

                if legal_warnings:
                    red_flags.extend(legal_warnings)
                    if not quiet_mode:
                        logger.info(
                            "legal_warnings_detected",
                            ticker=ticker,
                            warning_types=[w["type"] for w in legal_warnings],
                            total_risk_penalty=sum(
                                w.get("risk_penalty", 0) for w in legal_warnings
                            ),
                        )

            # --- Strict-Mode: Value Trap Flag Detection ---
            # In strict mode, also detect value trap flags here so we can escalate to REJECT.
            # value_trap_report may be empty if the Value Trap Detector hasn't run yet
            # (parallel branch), but tests can inject it. The PM addendum handles the
            # production case where VT runs in parallel with the Validator.
            if strict_mode:
                value_trap_report = state.get("value_trap_report", "")
                if value_trap_report:
                    if not isinstance(value_trap_report, str):
                        value_trap_report = extract_string_content(value_trap_report)
                    vt_warnings = RedFlagDetector.detect_value_trap_flags(
                        value_trap_report, ticker
                    )
                    if vt_warnings:
                        red_flags.extend(vt_warnings)

            # --- Strict-Mode: Escalate WARNING-level legal/VT flags to REJECT ---
            if strict_mode and pre_screening_result == "PASS":
                flag_types = {f["type"] for f in red_flags}
                if "PFIC_PROBABLE" in flag_types or "PFIC_UNCERTAIN" in flag_types:
                    pre_screening_result = "REJECT"
                    red_flags.append(
                        {
                            "type": "STRICT_PFIC_ESCALATED",
                            "severity": "CRITICAL",
                            "detail": "PFIC risk escalated to reject in strict mode",
                            "action": "AUTO_REJECT",
                            "rationale": "PFIC tax reporting burden is disqualifying in strict mode",
                        }
                    )
                    logger.info("strict_pfic_escalated_to_reject", ticker=ticker)
                elif "VIE_STRUCTURE" in flag_types:
                    pre_screening_result = "REJECT"
                    red_flags.append(
                        {
                            "type": "STRICT_VIE_ESCALATED",
                            "severity": "CRITICAL",
                            "detail": "VIE structure escalated to reject in strict mode",
                            "action": "AUTO_REJECT",
                            "rationale": "Contractual VIE ownership (not equity) is disqualifying in strict mode",
                        }
                    )
                    logger.info("strict_vie_escalated_to_reject", ticker=ticker)
                elif (
                    "VALUE_TRAP_HIGH_RISK" in flag_types
                    or "VALUE_TRAP_VERDICT" in flag_types
                ):
                    pre_screening_result = "REJECT"
                    red_flags.append(
                        {
                            "type": "STRICT_VALUE_TRAP_ESCALATED",
                            "severity": "CRITICAL",
                            "detail": "High-risk value trap escalated to reject in strict mode",
                            "action": "AUTO_REJECT",
                            "rationale": "Value trap high-risk score is disqualifying in strict mode",
                        }
                    )
                    logger.info("strict_value_trap_escalated_to_reject", ticker=ticker)

            # Log results
            if pre_screening_result == "REJECT":
                logger.info(
                    "pre_screening_rejected",
                    ticker=ticker,
                    company_name=company_name,
                    red_flags_count=len(red_flags),
                    flag_types=[f["type"] for f in red_flags],
                    message=f"REJECTED: {ticker} ({company_name}) failed pre-screening due to {len(red_flags)} critical red flag(s)",
                )
            elif red_flags:
                logger.info(
                    "pre_screening_warnings",
                    ticker=ticker,
                    warnings_count=len(red_flags),
                    message=f"{ticker} has {len(red_flags)} warning(s) but passed pre-screening",
                )
            else:
                if not quiet_mode:
                    logger.info(
                        "pre_screening_passed",
                        ticker=ticker,
                        message=f"{ticker} passed pre-screening validation",
                    )

            return {
                "red_flags": red_flags,
                "pre_screening_result": pre_screening_result,
            }

        except Exception as e:
            # CRITICAL: Always write pre_screening_result so sync_check_router
            # doesn't hang waiting for it. Default to PASS on error to allow
            # the debate phase to proceed (better than blocking the graph).
            logger.error(
                "validator_crashed",
                ticker=ticker,
                error=str(e),
                message="Validator failed - defaulting to PASS to avoid blocking graph",
            )
            return {"red_flags": [], "pre_screening_result": "PASS"}

    return financial_health_validator_node
