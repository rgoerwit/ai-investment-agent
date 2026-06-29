"""Deterministic PM verdict-policy floor (post-parse, pre-persistence).

This is NOT a parser. ``pm_verdict_metadata`` parses; this module *normalizes* the
already-produced PM output before it is persisted, rewriting the verdict consistently
across the PM_BLOCK ``VERDICT:`` line (the source of truth all downstream consumers
read) and the human-readable ``PORTFOLIO MANAGER VERDICT:`` header.

Why it exists (APR.WA, June 2026): the Growth hard-fail has a Data-Vacuum / Marginal-
Turnaround exception (HOLD/small-BUY when Health is decent, valuation cheap, and the low
growth score reflects missing data rather than genuine shrinkage). That exception lived
in Step-1A prompt prose and was silently overridden when the Step-1C Zone tally — inflated
by soft/free-form points — crossed 2.0, force-selling a healthy name. This floor relinks
the safeguard to the decision deterministically: a soft-point tally may not convert a
healthy, data-limited name into DO_NOT_INITIATE.

Conservative by construction: only floors DO_NOT_INITIATE -> HOLD (never upgrades to BUY),
gates on the *deterministic* code subtotal (not the LLM's hand-summed total), requires no
auto-reject/critical flag, and requires positive multi-year revenue so a genuinely
shrinking name (e.g. KTY.WA: P/E > 18 and 3Y CAGR < 0) is never floored.
"""

from __future__ import annotations

import re

import structlog

from src.agents.pm_verdict_metadata import pm_verdict_metadata_from_text
from src.validators.metric_extractor import extract_metrics

logger = structlog.get_logger(__name__)

# Zone-1 (HIGH RISK) threshold in the PM decision framework.
ZONE_1_THRESHOLD = 2.0
HEALTH_FLOOR_MIN = 65.0
PE_FLOOR_MAX = 18.0
GROWTH_FAIL_MAX = 50.0

_GROWTH_SCORE_RE = re.compile(r"ADJUSTED_GROWTH_SCORE:\s*([0-9]+(?:\.[0-9]+)?)\s*%")
# REJECT canonicalizes to DO_NOT_INITIATE (src.pm_decision_parser), so the gate accepts
# a "VERDICT: REJECT" block; the rewrite must therefore match it too, or the floor
# silently no-ops on a REJECT-worded healthy name.
_FLOORED_FROM = r"(?:DO[ _]NOT[ _]INITIATE|REJECT)"
_PM_HEADER_RE = re.compile(
    rf"(?im)^(#+\s*PORTFOLIO MANAGER VERDICT:\s*){_FLOORED_FROM}\b.*$"
)
_PM_BLOCK_VERDICT_RE = re.compile(rf"(?im)^(\s*VERDICT:\s*){_FLOORED_FROM}\b.*$")
# Verdict-coupled chart-control fields that must not stay in their negative-verdict
# state after the floor (extract_pm_block reads them before falling back to verdict).
_SHOW_CHART_NO_RE = re.compile(r"(?im)^(\s*SHOW_VALUATION_CHART:\s*)NO\b.*$")
_DISCOUNT_ZERO_RE = re.compile(r"(?im)^(\s*VALUATION_DISCOUNT:\s*)0(?:\.0+)?\b.*$")


def _has_hard_flag(red_flags: list[dict]) -> bool:
    return any(
        (flag.get("action") == "AUTO_REJECT") or (flag.get("severity") == "CRITICAL")
        for flag in red_flags
    )


def _parse_growth_score(fundamentals_report: str | None) -> float | None:
    if not fundamentals_report:
        return None
    match = _GROWTH_SCORE_RE.search(fundamentals_report)
    return float(match.group(1)) if match else None


def maybe_floor_verdict_to_hold(
    content_str: str,
    *,
    fundamentals_report: str | None,
    red_flags: list[dict],
    code_subtotal: float | None,
    pre_screening_result: str | None,
    ticker: str = "UNKNOWN",
) -> tuple[str, bool]:
    """Floor a soft-point DO_NOT_INITIATE to HOLD for a healthy, data-limited name.

    Returns ``(content_str, floored)``. When ``floored`` is True the returned text has the
    PM_BLOCK verdict + header rewritten to HOLD and a note appended; the caller re-derives
    metadata from it so all surfaces agree.
    """
    if pm_verdict_metadata_from_text(content_str).verdict != "DO_NOT_INITIATE":
        return content_str, False
    if (pre_screening_result or "").strip().upper() != "PASS":
        return content_str, False
    if _has_hard_flag(red_flags):
        return content_str, False
    if code_subtotal is None or code_subtotal >= ZONE_1_THRESHOLD:
        return content_str, False

    metrics = extract_metrics(fundamentals_report or "")
    health = metrics.get("adjusted_health_score")
    pe = metrics.get("pe_ratio")
    cagr = metrics.get("revenue_cagr_3y")
    growth = _parse_growth_score(fundamentals_report)

    if health is None or health < HEALTH_FLOOR_MIN:
        return content_str, False
    if pe is None or pe > PE_FLOOR_MAX:
        return content_str, False
    # The exception only applies when the growth score actually failed AND multi-year
    # revenue is intact (positive 3Y CAGR) — i.e. the low score reflects missing data /
    # a turnaround, not genuine shrinkage. A negative/absent CAGR is never floored.
    # Conservative: if the growth score can't be parsed we cannot confirm the failure is
    # data-driven, so do NOT floor (absent data must not create a mitigation).
    if growth is None or growth >= GROWTH_FAIL_MAX:
        return content_str, False
    if cagr is None or cagr < 0:
        return content_str, False

    floored, n_header = _PM_HEADER_RE.subn(r"\1HOLD", content_str)
    floored, n_block = _PM_BLOCK_VERDICT_RE.subn(r"\1HOLD", floored)
    if n_block == 0:
        # Could not rewrite the source-of-truth verdict line — do not half-apply.
        logger.warning("verdict_floor_skipped_no_pm_block_verdict", ticker=ticker)
        return content_str, False

    # Normalize verdict-coupled PM_BLOCK fields so charts/reports don't keep behaving
    # like a negative verdict (extract_pm_block reads these fields *before* the verdict).
    # A DO_NOT_INITIATE block typically carries SHOW_VALUATION_CHART: NO and
    # VALUATION_DISCOUNT: 0.0, which would still suppress targets under the floored HOLD.
    floored = _SHOW_CHART_NO_RE.sub(r"\1YES", floored)
    floored = _DISCOUNT_ZERO_RE.sub(r"\g<1>0.8", floored)

    note = (
        "\n\n> **DETERMINISTIC VERDICT FLOOR APPLIED — DO NOT INITIATE → HOLD**\n"
        f"> The growth-score failure qualifies for the Data-Vacuum / Marginal-Turnaround "
        f"exception (Adjusted Health {health:.0f}% ≥ 65, P/E {pe:.2f} ≤ 18, "
        f"3Y revenue CAGR {cagr:.1f}% ≥ 0), and the deterministic code-computed risk "
        f"subtotal ({code_subtotal:+.2f}) is below the Zone-1 threshold ({ZONE_1_THRESHOLD}) "
        f"with no auto-reject flag. A soft-point tally may not convert a healthy, "
        f"data-limited name into an avoid/exit.\n"
    )
    floored = floored.rstrip() + note

    logger.info(
        "verdict_floored_to_hold",
        ticker=ticker,
        code_subtotal=round(code_subtotal, 2),
        adjusted_health=health,
        pe_ratio=pe,
        revenue_cagr_3y=cagr,
        growth_score=growth,
        header_rewrites=n_header,
        block_rewrites=n_block,
    )
    return floored, True
