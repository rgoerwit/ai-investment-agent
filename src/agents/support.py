from __future__ import annotations

import json
import re
from typing import Any

import structlog
from langchain_core.messages import ToolMessage
from langgraph.types import RunnableConfig

from src.data_block_utils import extract_last_data_block, has_parseable_data_block
from src.runtime_diagnostics import get_model_name as _get_model_name
from src.runtime_diagnostics import infer_provider

logger = structlog.get_logger(__name__)

_UNRESOLVED_NAME_WARNING = (
    "\nWARNING: Company name could not be verified from any data source. "
    "The ticker may be delisted or illiquid. Do NOT guess or assume which company "
    "this ticker belongs to. If you cannot confirm the identity from your tool "
    "results, state that the company identity is unverified."
)


def infer_provider_name(runnable: Any) -> str:
    """Infer provider name from model instance for diagnostics logging."""
    return infer_provider(
        model_name=_get_model_name(runnable),
        class_name=type(runnable).__name__,
    )


def get_model_name(runnable: Any) -> str | None:
    """Return model name for diagnostics logging when available."""
    return _get_model_name(runnable)


def get_context_from_config(config: RunnableConfig) -> Any | None:
    """Extract TradingContext from RunnableConfig.configurable dict."""
    try:
        configurable = config.get("configurable", {})
        return configurable.get("context")
    except (AttributeError, TypeError):
        return None


def _company_line(company_name: str, resolved: bool) -> str:
    """Build Company line with optional unresolved warning."""
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


def extract_news_highlights(news_report: str, max_chars: int = 25000) -> str:
    """Extract the small subset of news context used downstream."""
    if not news_report or len(news_report) < 300:
        return news_report

    highlights = []
    lines = news_report.split("\n")

    in_geo_section = False
    geo_lines = []
    for line in lines:
        line_upper = line.upper()
        if "GEOGRAPHIC REVENUE" in line_upper or "US REVENUE" in line_upper:
            in_geo_section = True
            if ":" in line and not line.strip().startswith("###"):
                geo_lines.append(line)
        elif in_geo_section:
            if line.startswith("---") or line.startswith("###"):
                in_geo_section = False
            elif line.strip():
                geo_lines.append(line)
                if len(geo_lines) >= 6:
                    break

    if geo_lines:
        highlights.append("**US/Geographic Revenue:**")
        highlights.extend(geo_lines[:6])

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
                highlights.append(line.strip()[:150])
                catalyst_count += 1
                if catalyst_count >= 3:
                    break

    result = "\n".join(highlights)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[...truncated for efficiency]"

    return result if result.strip() else news_report[:max_chars]


def compute_data_conflicts(raw_data: str, foreign_data: str) -> str:
    """Compare Junior vs FLA data and return a structured conflict block."""
    if not raw_data:
        return ""

    conflicts: list[str] = []

    def _extract_json_number(text: str, key: str) -> float | None:
        for pattern in [
            rf'"{key}"\s*:\s*(-?[\d.eE+]+)',
            rf"'{key}'\s*:\s*(-?[\d.eE+]+)",
        ]:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, OverflowError):
                    return None
        return None

    def _extract_json_string(text: str, key: str) -> str | None:
        for pattern in [
            rf'"{key}"\s*:\s*"([^"]+)"',
            rf"'{key}'\s*:\s*'([^']+)'",
        ]:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_json_bool(text: str, key: str) -> bool:
        for pattern in [
            rf'"{key}"\s*:\s*(true|false)',
            rf"'{key}'\s*:\s*(true|false)",
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower() == "true"
        return False

    junior_ocf = _extract_json_number(raw_data, "operatingCashflow")
    junior_analysts = _extract_json_number(raw_data, "numberOfAnalystOpinions")
    junior_peg = _extract_json_number(raw_data, "pegRatio")
    junior_mcap = _extract_json_number(raw_data, "marketCap")
    junior_latest_quarter_date = _extract_json_string(raw_data, "latest_quarter_date")
    quarter_date_source = _extract_json_string(raw_data, "_latest_quarter_date_source")
    split_metrics_quarantined = _extract_json_bool(
        raw_data, "_split_sensitive_metrics_quarantined"
    )

    filing_ocf = None
    filing_ocf_period = None
    parent_company = None

    if foreign_data:
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

        parent_match = re.search(
            r"(?:Parent Company|Controlling Shareholder)[:\s]*(.+?)(?:\n|$)",
            foreign_data,
            re.IGNORECASE,
        )
        if parent_match:
            parent_val = parent_match.group(1).strip()
            if parent_val.upper() not in ("NONE", "N/A", "NOT FOUND", ""):
                parent_company = parent_val

    if junior_ocf is not None and filing_ocf is not None and junior_ocf != 0:
        j_abs = abs(junior_ocf)
        f_abs = abs(filing_ocf)
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

    if junior_analysts is not None:
        analysts_int = int(junior_analysts)
        if analysts_int < 5 and junior_mcap is not None and junior_mcap > 500_000_000:
            conflicts.append(
                f"- ANALYST_COUNT: {analysts_int} [yfinance] for "
                f"${junior_mcap / 1e9:.1f}B market cap — "
                f"ANOMALY: likely data gap, not genuinely uncovered. "
                f"Verify independently before relying on 'undiscovered' thesis."
            )

    if junior_peg is not None and 0 <= junior_peg < 0.05:
        detail = (
            "growth denominator zero/missing/infinite"
            if junior_peg == 0
            else f"implies {1 / junior_peg:.0f}x expected growth"
        )
        conflicts.append(
            f"- PEG: {junior_peg:.2f} [yfinance] — UNRELIABLE ({detail}). "
            f"Do not use PEG to justify valuation."
        )

    if split_metrics_quarantined:
        conflicts.append(
            "- SPLIT_SHARE_BASIS_MISMATCH: Recent split detected; forward EPS, "
            "forward P/E, and PEG are incompatible with the current share basis. "
            "Forward EPS / PE / PEG are invalid for this run and must be reported "
            "as N/A; use trailing metrics only."
        )

    if quarter_date_source == "reconciled_most_recent_quarter":
        date_note = (
            f" ({junior_latest_quarter_date})" if junior_latest_quarter_date else ""
        )
        conflicts.append(
            "- QUARTER_DATE_RECONCILED: Newer quarter metadata supersedes stale "
            f"statement-derived quarter date{date_note}. LATEST_QUARTER_DATE must "
            "use the reconciled newer value."
        )

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

    if foreign_data and parent_company:
        conflicts.append(
            f"- PARENT/CONTROLLER: {parent_company} [FLA] — "
            f"yfinance does not provide parent-subsidiary data. "
            f"If controlling holder >40%, minority influence is limited."
        )
    elif foreign_data and not parent_company:
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


def extract_value_trap_verdict(value_trap_report: str) -> str:
    """Extract a one-line value trap verdict header from VALUE_TRAP_BLOCK."""
    if not value_trap_report:
        return ""
    score_match = re.search(r"SCORE:\s*(\d+)", value_trap_report)
    verdict_match = re.search(
        r"VERDICT:\s*(TRAP|CAUTIOUS|WATCHABLE|ALIGNED)", value_trap_report
    )
    risk_match = re.search(r"TRAP_RISK:\s*(HIGH|MEDIUM|LOW)", value_trap_report)
    if not (score_match and verdict_match):
        return ""
    score = score_match.group(1)
    verdict = verdict_match.group(1)
    risk = risk_match.group(1) if risk_match else "N/A"
    return (
        f"VALUE_TRAP_DETECTOR VERDICT: {verdict} (score {score}/100, "
        f"risk {risk}) - dedicated governance/catalyst agent assessment\n"
    )


def summarize_for_pm(report: str, report_type: str, max_chars: int = 3000) -> str:
    """Summarize verbose reports while preserving structured blocks."""
    if not report or not isinstance(report, str):
        return report or ""

    if len(report) <= max_chars:
        return report

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

    blocks_to_preserve = []
    for pattern in block_patterns:
        matches = re.findall(pattern, report, re.DOTALL | re.IGNORECASE)
        blocks_to_preserve.extend(matches)

    preserved = "\n\n".join(blocks_to_preserve)
    remaining_chars = max_chars - len(preserved) - 100

    if remaining_chars > 300:
        first_section = report[:remaining_chars]
        last_para = first_section.rfind("\n\n")
        if last_para > 200:
            first_section = first_section[:last_para]
        if preserved:
            return f"{first_section}\n\n[...summarized...]\n\n{preserved}"
        return first_section + "\n\n[...summarized...]"

    if preserved:
        return preserved[:max_chars]

    return report[:max_chars] + "\n[...summarized...]"


def _format_date_with_fy_hint(current_date: str) -> str:
    """Format date with a fiscal-year hint to reduce future-dated searches."""
    try:
        year = int(current_date[:4])
        return f"{current_date} (latest annual reports: FY{year - 1})"
    except (ValueError, IndexError):
        return current_date


def extract_field_sources_from_messages(messages: list) -> dict[str, str]:
    """Extract _field_sources from ToolMessage content in message history."""
    if not messages:
        return {}

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
            for message in messages
            if hasattr(message, "type") and getattr(message, "type", None) == "tool"
        ),
    )
    return {}


def format_attribution_table(field_sources: dict[str, str] | None) -> str:
    """Format source attribution as a compact prompt block."""
    if not field_sources:
        return ""

    priority_fields = [
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
    for field in priority_fields:
        if field in field_sources:
            lines.append(f"- {field}: {field_sources[field]}")

    if not lines:
        return ""

    return "\n### DATA SOURCE ATTRIBUTION\n" + "\n".join(lines) + "\n"


def extract_source_conflicts_from_messages(messages: list) -> dict[str, Any]:
    """Extract _source_conflicts from ToolMessage content in message history."""
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
    """Format source conflicts for Consultant or PM adjudication."""
    conflicts = extract_source_conflicts_from_messages(messages)
    if not conflicts:
        return ""

    lines = ["\n### DATA SOURCE CONFLICTS (>20% variance between providers)"]
    for field, conflict in conflicts.items():
        lines.append(
            f"  - {field}: {conflict.get('old_source', '?')}={conflict.get('old', '?')}"
            f", {conflict.get('new_source', '?')}={conflict.get('new', '?')}"
            f" (delta {conflict.get('variance_pct', '?')}%)"
        )
    return "\n".join(lines) + "\n"


def _is_output_insufficient(content: str, agent_key: str) -> bool:
    """Check whether an agent output is empty or clearly insufficient."""
    if not content or len(content) < 50:
        return True

    if agent_key == "junior_fundamentals_analyst":
        has_tool_output = (
            "get_financial_metrics" in content.lower()
            or "financial" in content.lower()
            or "roe" in content.lower()
            or "=== RAW" in content
        )
        return not has_tool_output or len(content) < 200

    if agent_key == "fundamentals_analyst":
        return not has_parseable_data_block(content)

    if agent_key == "news_analyst":
        return len(content) < 200

    return False


def _extract_sector_from_state(state: dict) -> str:
    """Extract sector from the fundamentals DATA_BLOCK for lesson retrieval."""
    fundamentals = state.get("fundamentals_report", "") or ""
    if not fundamentals:
        return "Unknown"
    data_block = extract_last_data_block(fundamentals)
    if not data_block:
        return "Unknown"
    match = re.search(r"SECTOR:\s*(.+?)(?:\n|$)", data_block, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        if value.upper() not in ("N/A", "NA", "NONE", "-", ""):
            return value
    return "Unknown"


def _extract_sector_country(raw_data: str) -> tuple:
    """Extract sector and country from Junior Analyst raw JSON output."""
    sector, country = "Unknown", "Unknown"

    if not raw_data:
        return sector, country

    try:
        sector_match = re.search(r'"sector"\s*:\s*"([^"]+)"', raw_data, re.IGNORECASE)
        if sector_match:
            sector = sector_match.group(1)

        country_match = re.search(r'"country"\s*:\s*"([^"]+)"', raw_data, re.IGNORECASE)
        if country_match:
            country = country_match.group(1)

        if sector == "Unknown":
            industry_match = re.search(
                r'"industry"\s*:\s*"([^"]+)"', raw_data, re.IGNORECASE
            )
            if industry_match:
                sector = industry_match.group(1)
    except Exception as exc:
        logger.debug("sector_country_extract_failed", error=str(exc))

    return sector, country
