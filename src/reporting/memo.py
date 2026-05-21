"""Investment-memo builder rendered at the top of every report.

The memo is a tight, fixed-shape summary that sits above the chart and the
appendix transcript. Its slots are:

- decision (BUY / HOLD / DO_NOT_INITIATE / SELL / UNAVAILABLE)
- one-line thesis (≤ ~30 words pulled from PM rationale)
- variant view (placeholder until Research Manager emits VARIANT_PERCEPTION)
- key numbers (4–6 DATA_BLOCK metrics)
- valuation (single-range string until scenario valuation ships)
- top risks (3–4 named red flags / specific risks)
- kill criteria (parsed from Bear pre-mortem fenced block)
- confidence (one sentence summarizing which optional agents ran)

Every extractor degrades to a placeholder string rather than raising. A run
that produced no PM output renders an "Investment Memo — UNAVAILABLE" stub so
the rest of the report still publishes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from src.agents.support import extract_kill_criteria, get_bear_history
from src.data_block_utils import extract_data_block_field
from src.reporting.source_confidence import (
    SourceRow,
    build_source_confidence_rows,
    render_source_confidence_markdown,
)

logger = structlog.get_logger(__name__)


_VERDICT_PM_BLOCK = re.compile(
    r"### --- START PM_BLOCK[^\n]*---(.+?)### --- END PM_BLOCK ---",
    re.DOTALL,
)
_VERDICT_LINE = re.compile(r"VERDICT:\s*([A-Z_ ]+)")
_VERDICT_NARRATIVE = re.compile(
    r"PORTFOLIO MANAGER VERDICT:\s*(BUY|HOLD|DO NOT INITIATE|SELL)",
    re.IGNORECASE,
)
_RATIONALE_HEADER = re.compile(
    r"#+\s*DECISION RATIONALE\s*\n+(.+?)(?:\n#+\s|\n---|\Z)",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class InvestmentMemo:
    """Structured memo content, ready for markdown rendering."""

    decision: str = "UNAVAILABLE"
    one_line_thesis: str = "Thesis unavailable."
    variant_view: str = "Not explicitly stated."
    key_numbers: list[str] = field(default_factory=list)
    valuation: str = "Valuation summary unavailable."
    top_risks: list[str] = field(default_factory=list)
    kill_criteria: list[str] = field(default_factory=list)
    confidence: str = "Confidence signals unavailable."
    source_confidence: list[SourceRow] = field(default_factory=list)


def _normalize_verdict(raw: str | None) -> str:
    if not raw:
        return "UNAVAILABLE"
    cleaned = raw.strip().upper().replace("-", " ").replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in {"DO NOT INITIATE", "DONOTINITATE", "DONOTINITIATE"}:
        return "DO_NOT_INITIATE"
    if cleaned in {"BUY", "HOLD", "SELL"}:
        return cleaned
    return cleaned.replace(" ", "_") if cleaned else "UNAVAILABLE"


def extract_pm_verdict(pm_text: str) -> str:
    """Return the canonical verdict label from PM output, or 'UNAVAILABLE'."""
    if not pm_text:
        return "UNAVAILABLE"
    blocks = list(_VERDICT_PM_BLOCK.finditer(pm_text))
    if blocks:
        match = _VERDICT_LINE.search(blocks[-1].group(1))
        if match:
            return _normalize_verdict(match.group(1))
    narrative = _VERDICT_NARRATIVE.search(pm_text)
    if narrative:
        return _normalize_verdict(narrative.group(1))
    return "UNAVAILABLE"


def extract_pm_thesis(pm_text: str, max_words: int = 30) -> str:
    """Pull the first sentence of the DECISION RATIONALE section, capped at max_words."""
    if not pm_text:
        return "Thesis unavailable."
    match = _RATIONALE_HEADER.search(pm_text)
    if not match:
        return "Thesis unavailable."
    body = match.group(1).strip()
    # First sentence-ish (cope with markdown bullets and bold prefixes).
    body = re.sub(r"^[*\-•\s]+", "", body)
    sentence_break = re.search(r"(.+?[.!?])\s", body + " ")
    sentence = sentence_break.group(1) if sentence_break else body.split("\n", 1)[0]
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words]).rstrip(",;:") + "…"
    return sentence.strip()


def extract_variant_view(_state: dict) -> str:
    """Placeholder until Research Manager emits VARIANT_PERCEPTION (Step 7).

    Looks for a `CONSENSUS_VIEW:` / `VARIANT_VIEW:` pair in the investment plan
    in case the prompt change has already shipped; otherwise returns the
    default placeholder.
    """
    plan = _state.get("investment_plan") or ""
    if not plan:
        return "Not explicitly stated."
    variant = re.search(
        r"VARIANT_VIEW\s*:\s*(.+?)(?:\n\n|\n[A-Z_]{3,}\s*:|\Z)",
        plan,
        re.DOTALL,
    )
    if variant:
        text = re.sub(r"\s+", " ", variant.group(1)).strip()
        words = text.split()
        if len(words) > 40:
            text = " ".join(words[:40]).rstrip(",;:") + "…"
        return text
    if "NO VARIANT" in plan.upper():
        return "Synthesis aligns with consensus — no material variant."
    return "Not explicitly stated."


_KEY_FIELDS: tuple[tuple[str, str], ...] = (
    ("P/E (TTM)", "PE_RATIO_TTM"),
    ("PEG", "PEG_RATIO"),
    ("ROIC", "ROIC_PERCENT"),
    ("FCF yield", "FCF_YIELD_PERCENT"),
    ("Revenue growth (TTM)", "REVENUE_GROWTH_TTM"),
    ("Net debt / EBITDA", "NET_DEBT_TO_EBITDA"),
    ("D/E", "DEBT_TO_EQUITY"),
    ("Analyst coverage (EN)", "ANALYST_COVERAGE_ENGLISH"),
)


def extract_key_metrics(fundamentals: str, limit: int = 6) -> list[str]:
    """Pull up to `limit` non-empty key metrics from the fundamentals DATA_BLOCK."""
    if not fundamentals:
        return []
    rows: list[str] = []
    for label, key in _KEY_FIELDS:
        value = extract_data_block_field(fundamentals, key)
        if (
            value
            and value.strip()
            and value.strip().upper() not in {"N/A", "NA", "NONE"}
        ):
            rows.append(f"{label}: {value.strip()}")
        if len(rows) >= limit:
            break
    return rows


def format_scenario_summary(_state: dict) -> str | None:
    """Placeholder until scenario valuation (Step 6) ships.

    Returns None so the caller can fall back to the legacy single-range string.
    """
    return None


def extract_legacy_target_range(state: dict) -> str:
    """Compose a one-line valuation summary from whatever valuation context exists."""
    valuation_context = state.get("valuation_context") or ""
    if "Target Range" in valuation_context:
        match = re.search(
            r"Target Range:\s*([^\n]+)\nFair Value \(midpoint\):\s*([^\n]+)",
            valuation_context,
        )
        if match:
            return (
                f"Target range {match.group(1).strip()} (mid {match.group(2).strip()})."
            )
    # Try the football-field params from chart_paths context.
    fundamentals = state.get("fundamentals_report") or ""
    price = extract_data_block_field(fundamentals, "CURRENT_PRICE")
    if price:
        return f"Current price {price.strip()}; target range unavailable."
    return "Valuation summary unavailable."


def extract_pm_risks(
    pm_text: str, red_flags: list[dict] | None, limit: int = 4
) -> list[str]:
    """Combine red-flag detail lines and PM-narrative risk bullets into a short list."""
    risks: list[str] = []
    for flag in red_flags or []:
        flag_type = flag.get("type") if isinstance(flag, dict) else None
        detail = flag.get("detail") if isinstance(flag, dict) else None
        if flag_type and detail:
            risks.append(f"{flag_type}: {detail}")
        elif flag_type:
            risks.append(str(flag_type))
        if len(risks) >= limit:
            return risks[:limit]
    if pm_text:
        # Pull bullets under "Key Risks" / "Risks" / "Top Risks" headings.
        section = re.search(
            r"#+\s*(?:TOP\s+|KEY\s+)?RISKS\b.*?\n+(.+?)(?:\n#+\s|\n---|\Z)",
            pm_text,
            re.DOTALL | re.IGNORECASE,
        )
        if section:
            for line in section.group(1).splitlines():
                stripped = re.sub(r"^[*\-•\s]+", "", line).strip()
                if stripped and len(stripped) > 4:
                    risks.append(stripped)
                if len(risks) >= limit:
                    break
    return risks[:limit]


def summarize_confidence(state: dict) -> str:
    """One-sentence summary of which optional cross-checks ran."""
    run_summary = state.get("run_summary") or {}
    bits: list[str] = []
    if run_summary.get("consultant_successful"):
        bits.append("consultant cross-check passed")
    elif run_summary.get("consultant_completed"):
        bits.append("consultant ran but did not approve")
    if run_summary.get("auditor_successful"):
        bits.append("forensic auditor clean")
    elif run_summary.get("auditor_completed"):
        bits.append("forensic auditor ran with caveats")
    if run_summary.get("apac_specialist_successful"):
        bits.append("APAC specialist engaged")
    if not bits:
        return "Optional cross-checks did not run; rely on core agents only."
    return "Anchored on " + ", ".join(bits) + "."


def build_memo(state: dict) -> InvestmentMemo:
    """Assemble the InvestmentMemo from the analysis result dict.

    Accepts either the live AgentState dict (during graph execution) or the
    saved JSON shape (for retrospective rendering).
    """
    pm = (
        state.get("final_trade_decision")
        or (state.get("reports") or {}).get("portfolio_manager")
        or ""
    )
    fundamentals = (
        state.get("fundamentals_report")
        or (state.get("reports") or {}).get("fundamentals_report")
        or ""
    )
    bear_text = get_bear_history(state)
    red_flags = state.get("red_flags") or []

    return InvestmentMemo(
        decision=extract_pm_verdict(pm),
        one_line_thesis=extract_pm_thesis(pm),
        variant_view=extract_variant_view(state),
        key_numbers=extract_key_metrics(fundamentals),
        valuation=format_scenario_summary(state) or extract_legacy_target_range(state),
        top_risks=extract_pm_risks(pm, red_flags),
        kill_criteria=extract_kill_criteria(bear_text),
        confidence=summarize_confidence(state),
        source_confidence=build_source_confidence_rows(state),
    )


def render_memo_markdown(memo: InvestmentMemo) -> str:
    """Render an InvestmentMemo as a self-contained markdown section."""
    if memo.decision == "UNAVAILABLE" and not memo.key_numbers:
        return (
            "## Investment Memo — UNAVAILABLE\n\n"
            "No Portfolio Manager output was available for this run; the appendix "
            "below contains whatever the graph did produce.\n\n---\n\n"
        )

    parts: list[str] = [f"## Investment Memo — {memo.decision}\n\n"]
    parts.append(f"**Thesis.** {memo.one_line_thesis}\n\n")
    parts.append(f"**Variant view.** {memo.variant_view}\n\n")

    if memo.key_numbers:
        parts.append("**Key numbers.**\n\n")
        parts.extend(f"- {row}\n" for row in memo.key_numbers)
        parts.append("\n")

    parts.append(f"**Valuation.** {memo.valuation}\n\n")

    if memo.top_risks:
        parts.append("**Top risks.**\n\n")
        parts.extend(f"- {risk}\n" for risk in memo.top_risks)
        parts.append("\n")

    if memo.kill_criteria:
        parts.append("**Kill criteria.**\n\n")
        parts.extend(f"- {trigger}\n" for trigger in memo.kill_criteria)
        parts.append("\n")

    parts.append(f"**Confidence.** {memo.confidence}\n\n")

    if memo.source_confidence:
        parts.append("**Source confidence.**\n\n")
        parts.append(render_source_confidence_markdown(memo.source_confidence))
        parts.append("\n")

    parts.append("---\n\n")
    return "".join(parts)


def render_memo_for_state(state: dict) -> str:
    """One-shot helper: build + render the memo from a state/result dict."""
    try:
        return render_memo_markdown(build_memo(state))
    except Exception as exc:  # pragma: no cover — defense-in-depth
        logger.warning("memo_render_failed", error=str(exc), exc_info=True)
        return (
            "## Investment Memo — UNAVAILABLE\n\n"
            "Memo rendering encountered an error; see logs.\n\n---\n\n"
        )
