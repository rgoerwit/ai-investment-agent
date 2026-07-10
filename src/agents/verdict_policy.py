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
# PM-owned narrative/execution decision lines. Deliberately case-sensitive:
# Trader TRADE_BLOCK uses all-caps ACTION:, and report errors use
# **Action Required**:. Do not loosen this to Action[^:]* or add re.IGNORECASE.
_PM_DECISION_SURFACE_RE = re.compile(
    r"(?m)^(?P<label>\*{0,2}(?:Actual Decision|Action)\*{0,2}):\s*"
    r"(?P<verdict>BUY|HOLD|SELL|DO NOT INITIATE)\b.*$"
)
# Verdict-coupled chart-control fields that must not stay in their negative-verdict
# state after the floor (extract_pm_block reads them before falling back to verdict).
_SHOW_CHART_NO_RE = re.compile(r"(?im)^(\s*SHOW_VALUATION_CHART:\s*)NO\b.*$")
_DISCOUNT_ZERO_RE = re.compile(r"(?im)^(\s*VALUATION_DISCOUNT:\s*)0(?:\.0+)?\b.*$")


_PM_HEADER_BUY_RE = re.compile(r"(?im)^(#+\s*PORTFOLIO MANAGER VERDICT:\s*)BUY\b.*$")
_PM_BLOCK_VERDICT_BUY_RE = re.compile(r"(?im)^(\s*VERDICT:\s*)BUY\b.*$")


def _rewrite_pm_decision_surfaces(content_str: str, canonical_display: str) -> str:
    """Rewrite PM-owned narrative/execution decision lines to the canonical verdict."""
    return _PM_DECISION_SURFACE_RE.sub(
        lambda match: f"{match.group('label')}: {canonical_display}",
        content_str,
    )


def maybe_demote_buy_on_blocking_flags(
    content_str: str,
    *,
    red_flags: list[dict],
    ticker: str = "UNKNOWN",
) -> tuple[str, bool]:
    """Demote a BUY to HOLD when any red flag carries ``blocks_buy: True``.

    Flags that block BUY (``*_SCORE_UNRELIABLE``, ``MATERIAL_UNVERIFIED_
    OPERATING_SIGNAL``) are zero-penalty by design, so the tally cannot stop a
    BUY that ignores them — and ``format_red_flag_section`` renders only
    type+detail, so their rationale prose never reaches the PM. This is the
    deterministic enforcement of that contract (the AGS.BR 2026-07-02 failure:
    an unreliable 56% health passed the <50% gate and backed a BUY).
    Conservative: only BUY -> HOLD, never touches SELL/DNI/HOLD, and never
    upgrades anything.

    Returns ``(content_str, demoted)``; on demotion the caller re-derives
    metadata so all surfaces agree.
    """
    if pm_verdict_metadata_from_text(content_str).verdict != "BUY":
        return content_str, False
    blocking = sorted(
        {
            str(flag.get("type", "UNKNOWN"))
            for flag in red_flags
            if flag.get("blocks_buy") is True
        }
    )
    if not blocking:
        return content_str, False

    demoted, n_header = _PM_HEADER_BUY_RE.subn(r"\1HOLD", content_str)
    demoted, n_block = _PM_BLOCK_VERDICT_BUY_RE.subn(r"\1HOLD", demoted)
    if n_block == 0:
        logger.warning("buy_demotion_skipped_no_pm_block_verdict", ticker=ticker)
        return content_str, False
    demoted = _rewrite_pm_decision_surfaces(demoted, "HOLD")

    note = (
        "\n\n> **DETERMINISTIC VERDICT DEMOTION APPLIED — BUY → HOLD**\n"
        f"> BUY-blocking flag(s) present: {', '.join(blocking)}. These mark the "
        "supporting evidence as indeterminate or unverified, so it may not back "
        "initiating a position. Resolve the flagged issue and re-run to restore "
        "BUY eligibility.\n"
    )
    demoted = demoted.rstrip() + note

    logger.info(
        "buy_demoted_on_blocking_flags",
        ticker=ticker,
        blocking_flags=blocking,
        header_rewrites=n_header,
        block_rewrites=n_block,
    )
    return demoted, True


def maybe_qualify_buy_in_quick_mode(
    content_str: str, *, quick_mode: bool, ticker: str = "UNKNOWN"
) -> tuple[str, bool]:
    """Append a 'screening candidate, not investable' caveat to a quick-mode BUY.

    The `--quick` tier is a screener (flash-lite gathering, cheap deep models, 1 debate
    round); a quick BUY is not trustworthy enough to act on. This appends a caveat note
    but deliberately leaves the ``VERDICT: BUY`` token intact, so no downstream parser
    (charts, article, IBKR, run-summary) changes — the note flows through the persisted
    PM text the same way ``maybe_demote_buy_on_blocking_flags`` appends its note.

    Conservative and idempotent: fires only in quick mode on a BUY verdict, and never
    a second time. Returns ``(content_str, qualified)``; ``qualified`` reflects whether
    the note is present so the caller can stamp an honest run-summary flag (never a
    recomputed ``quick and BUY`` that could lie if this no-ops on parse drift).
    """
    if not quick_mode:
        return content_str, False
    if pm_verdict_metadata_from_text(content_str).verdict != "BUY":
        return content_str, False
    if "QUICK-MODE QUALIFICATION" in content_str:
        return content_str, True
    note = (
        "\n\n> **QUICK-MODE QUALIFICATION — CANDIDATE FOR FULL ANALYSIS, NOT INVESTABLE OUTPUT**\n"
        "> This BUY came from the fast screening tier (quick models, 1 debate round). Treat it "
        "as a shortlist candidate to promote to a full analysis, not an actionable buy. Re-run in "
        "full mode before acting.\n"
    )
    logger.info("buy_qualified_quick_mode", ticker=ticker)
    return content_str.rstrip() + note, True


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
    floored = _rewrite_pm_decision_surfaces(floored, "HOLD")

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
