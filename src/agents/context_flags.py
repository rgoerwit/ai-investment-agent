"""Small deterministic context flags for PM and writer-facing reports."""

from __future__ import annotations

import re

from src.data_block_utils import extract_data_block_field

MACRO_DRAWDOWN_TERMS = (
    "rate shock",
    "rate-shock",
    "rate hike",
    "multiple compression",
    "macro",
    "sector",
    "market sell-off",
    "technical",
)

COMPANY_DRAWDOWN_TERMS = (
    "revenue",
    "margin",
    "take-rate",
    "cash flow",
    "competition",
    "underinvestment",
    "governance",
    "profitability",
)


def _parse_number(raw: str | None) -> float | None:
    if not raw:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(raw))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


DECLINE_TERMS = (
    "drawdown",
    "downtrend",
    "declin",
    "fell",
    "fall",
    "drop",
    "sell-off",
    "selloff",
    "de-rat",
    "derat",
    "plunge",
    "slump",
    "tumble",
    "collapse",
    "sank",
    "two-thirds",
)


def _decline_context(text: str, radius: int = 160) -> str:
    """Return the text windows around explicit decline mentions.

    Causes are only meaningful when they sit near where the decline is
    discussed; this isolates those windows so generic vocabulary elsewhere in
    the report does not count as an explanation.
    """
    lower = text.lower()
    windows: list[str] = []
    for term in DECLINE_TERMS:
        idx = lower.find(term)
        while idx != -1:
            windows.append(lower[max(0, idx - radius) : idx + radius])
            idx = lower.find(term, idx + len(term))
    return " ".join(windows)


def classify_large_drawdown_context(
    text: str,
    current: float | None,
    high: float | None,
) -> str | None:
    """Classify a large drawdown by whether causes sit near the decline.

    Generic cause vocabulary ("revenue", "sector") appears in every report, so a
    cause only counts when it is mentioned near an explicit decline. A drawdown
    with no decline discussion — or none carrying a cause — is uninvestigated.
    """
    if current is None or high is None or current <= 0 or high <= 0:
        return None
    if current / high > 0.60:
        return None

    window = _decline_context(text)
    has_macro = any(term in window for term in MACRO_DRAWDOWN_TERMS)
    has_company = any(term in window for term in COMPANY_DRAWDOWN_TERMS)

    if has_macro and has_company:
        return "LARGE_DRAWDOWN_MIXED"
    if has_macro:
        return "LARGE_DRAWDOWN_MACRO_ONLY"
    if has_company:
        return "LARGE_DRAWDOWN_COMPANY_SPECIFIC"
    return "UNEXPLAINED_LARGE_DRAWDOWN"


def unresolved_related_tickers(fundamentals: str | None) -> list[str]:
    """Return RELATED_LISTED_TICKERS entries with unresolved relationship facts."""
    raw = extract_data_block_field(fundamentals, "RELATED_LISTED_TICKERS") or ""
    if raw.strip().upper() in {"", "NONE", "N/A", "UNKNOWN"}:
        return []
    unresolved: list[str] = []
    for item in raw.split(";"):
        entry = item.strip()
        if not entry:
            continue
        parts = [part.strip().upper() for part in entry.split(":")]
        if parts and parts[-1] == "UNKNOWN":
            unresolved.append(entry)
    return unresolved


def format_pm_context_flags(fundamentals: str | None, *reports: str | None) -> str:
    """Build a compact PM context section for deterministic unresolved issues."""
    lines: list[str] = []
    current = _parse_number(extract_data_block_field(fundamentals, "CURRENT_PRICE"))
    high = _parse_number(extract_data_block_field(fundamentals, "FIFTY_TWO_WEEK_HIGH"))
    low = extract_data_block_field(fundamentals, "FIFTY_TWO_WEEK_LOW")
    text = "\n".join(report for report in reports if report)

    drawdown_flag = classify_large_drawdown_context(text, current, high)
    if drawdown_flag:
        drawdown_pct = (1.0 - (current or 0.0) / (high or 1.0)) * 100.0
        low_text = f"; 52-week low {low.strip()}" if low else ""
        lines.append(
            f"- {drawdown_flag}: current price is {drawdown_pct:.1f}% below "
            f"52-week high{low_text}; distinguish macro/multiple compression "
            "from company-specific deterioration."
        )

    related = unresolved_related_tickers(fundamentals)
    if related:
        lines.append(
            "- UNRESOLVED_RELATED_TICKERS: "
            f"{'; '.join(related)}. Treat strategic-partner optionality as "
            "filing-verification-required, not proven."
        )

    if not lines:
        return ""
    return "\n\nSUPPLEMENTAL PM FLAGS:\n" + "\n".join(lines)
