"""Deterministic PM verdict-policy floor (post-parse, pre-persistence).

This is NOT a parser. ``pm_verdict_metadata`` parses; this module *normalizes* the
already-produced PM output before it is persisted, rewriting the verdict consistently
across the PM_BLOCK ``VERDICT:`` line (the source of truth all downstream consumers
read) and the human-readable ``PORTFOLIO MANAGER VERDICT:`` header.

Why it exists (APR.WA, June 2026): the Growth hard-fail has a Data-Vacuum / Marginal-
Turnaround exception (HOLD when Health is decent, valuation cheap, and the low
growth score reflects missing data rather than genuine shrinkage). That exception lived
in Step-1A prompt prose and was silently overridden when the Step-1C Zone tally — inflated
by soft/free-form points — crossed 2.0, force-selling a healthy name. This floor relinks
the safeguard to the decision deterministically: a soft-point tally may not convert a
healthy, data-limited name into DO_NOT_INITIATE.

Conservative by construction: only floors DO_NOT_INITIATE -> HOLD (never upgrades to BUY),
gates on the *deterministic* code subtotal (not the LLM's hand-summed total), requires no
auto-reject/critical or individually material flag, and requires positive multi-year
revenue so a genuinely shrinking name (e.g. KTY.WA: P/E > 18 and 3Y CAGR < 0)
is never floored.

Every public rewriter in this module is a projection onto a canonical form: applying
it repeatedly must leave the text unchanged after the first application.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import structlog

from src.agents.pm_verdict_metadata import pm_verdict_metadata_from_text
from src.charts.extractors.valuation import is_weak_buy_asymmetry
from src.data_block_utils import fenced_block_pattern
from src.decision_inputs import DecisionInputs
from src.pm_decision_parser import canonicalize_pm_verdict, parse_final_decision_scores
from src.thesis_constants import (
    ANALYST_COVERAGE_MAX,
    GROWTH_MIN_PCT,
    HEALTH_EXCEPTION_CURRENT_RATIO_MIN,
    HEALTH_EXCEPTION_MIN_PCT,
    HEALTH_EXCEPTION_PB_MAX,
    HEALTH_MIN_PCT,
    US_REVENUE_MAX_PCT,
)

logger = structlog.get_logger(__name__)

# Zone-1 (HIGH RISK) threshold in the PM decision framework.
ZONE_1_THRESHOLD = 2.0
HEALTH_FLOOR_MIN = 65.0
PE_FLOOR_MAX = 18.0
GROWTH_FAIL_MAX = 50.0

# REJECT canonicalizes to DO_NOT_INITIATE (src.pm_decision_parser), so the gate accepts
# a "VERDICT: REJECT" block; the rewrite must therefore match it too, or the floor
# silently no-ops on a REJECT-worded healthy name.
_VERDICT_TOKEN = r"(?:BUY|HOLD|SELL|DO[ _]NOT[ _]INITIATE|REJECT)"
_PM_HEADER_RE = re.compile(
    rf"(?im)^(#+\s*PORTFOLIO MANAGER VERDICT:\s*){_VERDICT_TOKEN}\b.*$"
)
_PM_BLOCK_VERDICT_RE = re.compile(rf"(?im)^(\s*VERDICT:\s*){_VERDICT_TOKEN}\b.*$")
# PM-owned narrative/execution decision lines. Deliberately case-sensitive:
# Trader TRADE_BLOCK uses all-caps ACTION:, and report errors use
# **Action Required**:. Do not loosen this to Action[^:]* or add re.IGNORECASE.
_PM_DECISION_SURFACE_RE = re.compile(
    r"(?m)^(?P<label>\*{0,2}(?:Actual Decision|Action)\*{0,2}):\s*"
    r"(?P<verdict>BUY|HOLD|SELL|DO NOT INITIATE)\b.*$"
)
# Verdict-coupled chart-control fields that must not stay in their negative-verdict
# state after the floor (extract_pm_block reads them before falling back to verdict).
_SHOW_CHART_RE = re.compile(r"(?im)^(\s*SHOW_VALUATION_CHART:\s*)\S+.*$")
_DISCOUNT_RE = re.compile(r"(?im)^(\s*VALUATION_DISCOUNT:\s*)\S+.*$")
_POSITION_SIZE_PROSE_RE = re.compile(r"(?im)^(\**Recommended Position Size\**:\s*).*$")


def _rewrite_pm_decision_surfaces(content_str: str, canonical_display: str) -> str:
    """Rewrite PM-owned narrative/execution decision lines to the canonical verdict."""
    return _PM_DECISION_SURFACE_RE.sub(
        lambda match: f"{match.group('label')}: {canonical_display}",
        content_str,
    )


def normalize_pm_block_contract(pm_output: str) -> str:
    """Clamp position-size surfaces for a final no-initiation verdict."""

    blocks = list(fenced_block_pattern("PM_BLOCK").finditer(pm_output))
    if not blocks:
        return pm_output
    last = blocks[-1]
    body = last.group(1)
    verdict_match = re.search(r"(?im)^VERDICT:\s*([^\n]+)", body)
    if not verdict_match:
        return pm_output
    verdict = canonicalize_pm_verdict(verdict_match.group(1))
    if verdict not in {"HOLD", "DO_NOT_INITIATE", "SELL"}:
        return pm_output

    token_rewritten = False
    size_match = re.search(r"(?im)^(POSITION_SIZE:\s*)([\d.]+)", body)
    if size_match:
        try:
            emitted_size = float(size_match.group(2))
        except ValueError:
            emitted_size = 0.0
        if emitted_size != 0.0:
            rewritten_body = re.sub(
                r"(?im)^(POSITION_SIZE:\s*)[\d.]+",
                r"\g<1>0.0",
                body,
                count=1,
            )
            pm_output = (
                pm_output[: last.start(1)] + rewritten_body + pm_output[last.end(1) :]
            )
            token_rewritten = True

    canonical_prose_value = "0.0% (monitor only — no initiation)"
    prose_lines_rewritten = sum(
        1
        for match in _POSITION_SIZE_PROSE_RE.finditer(pm_output)
        if match.group(0) != f"{match.group(1)}{canonical_prose_value}"
    )
    pm_output = _POSITION_SIZE_PROSE_RE.sub(
        lambda match: f"{match.group(1)}{canonical_prose_value}", pm_output
    )
    if token_rewritten:
        logger.warning(
            "pm_block_position_size_rewritten",
            verdict=verdict,
            token_rewritten=token_rewritten,
            prose_lines_rewritten=prose_lines_rewritten,
        )
    return pm_output


def _apply_verdict_rewrite(
    content_str: str,
    *,
    required_verdict: str,
    allowed_from: frozenset[str],
    note: str,
    ticker: str,
) -> tuple[str, bool, int, int]:
    """Rewrite all code-owned PM verdict surfaces through one guarded seam."""

    current = pm_verdict_metadata_from_text(content_str).verdict
    if current not in allowed_from:
        return content_str, False, 0, 0
    target = canonicalize_pm_verdict(required_verdict)
    display = "DO NOT INITIATE" if target == "DO_NOT_INITIATE" else target
    if target not in {"BUY", "HOLD", "SELL", "DO_NOT_INITIATE"}:
        raise ValueError(f"Unsupported canonical PM verdict: {required_verdict}")

    rewritten, n_header = _PM_HEADER_RE.subn(rf"\g<1>{display}", content_str)
    rewritten, n_block = _PM_BLOCK_VERDICT_RE.subn(rf"\g<1>{target}", rewritten)
    if n_block == 0:
        logger.warning(
            "verdict_rewrite_skipped_no_pm_block_verdict",
            ticker=ticker,
            current_verdict=current,
            required_verdict=target,
        )
        return content_str, False, n_header, 0

    rewritten = _rewrite_pm_decision_surfaces(rewritten, display)
    if target == "HOLD" and current == "DO_NOT_INITIATE":
        rewritten = _SHOW_CHART_RE.sub(r"\g<1>YES", rewritten)
        rewritten = _DISCOUNT_RE.sub(r"\g<1>0.8", rewritten)
    elif target in {"DO_NOT_INITIATE", "SELL"}:
        rewritten = _SHOW_CHART_RE.sub(r"\g<1>NO", rewritten)
        rewritten = _DISCOUNT_RE.sub(r"\g<1>0.0", rewritten)
    rewritten = normalize_pm_block_contract(rewritten)
    if note not in rewritten:
        rewritten = rewritten.rstrip() + "\n\n" + note.rstrip() + "\n"
    return rewritten, True, n_header, n_block


def apply_required_verdict(
    content_str: str,
    *,
    required_verdict: str,
    rule: str,
    reason: str,
    ticker: str = "UNKNOWN",
) -> tuple[str, bool]:
    """Apply a deterministic policy downgrade; never create an upgrade path."""

    if canonicalize_pm_verdict(required_verdict) != "DO_NOT_INITIATE":
        raise ValueError("Deterministic policy correction may only require DNI")
    note = (
        "> **DETERMINISTIC VERDICT POLICY APPLIED — PRIOR RATIONALE SUPERSEDED**\n"
        f"> Rule {rule} requires DO NOT INITIATE because {reason}. The canonical "
        "header, PM_BLOCK, execution, chart, and sizing fields above control the "
        "decision; any earlier HOLD/BUY phrasing is non-operative context."
    )
    rewritten, changed, _, _ = _apply_verdict_rewrite(
        content_str,
        required_verdict="DO_NOT_INITIATE",
        allowed_from=frozenset({"BUY", "HOLD"}),
        note=note,
        ticker=ticker,
    )
    return rewritten, changed


def _exception_growth_ceiling_applies(
    inputs: DecisionInputs, assessment: GrowthGateAssessment
) -> bool:
    """Share the exception ceiling between enforcement and PM context."""
    return (
        assessment.exception in {"marginal_turnaround", "data_vacuum"}
        and inputs.decision_metrics.get("m_and_a_status") != "ACTIVE_TENDER"
    )


def maybe_cap_exception_growth_buy_to_hold(
    content_str: str, *, decision_inputs: DecisionInputs, ticker: str = "UNKNOWN"
) -> tuple[str, bool]:
    """A growth exception permits at most HOLD."""
    assessment = assess_growth_gate(decision_inputs)
    if not _exception_growth_ceiling_applies(decision_inputs, assessment):
        return content_str, False
    exception = assessment.exception
    assert exception is not None
    note = (
        "> **DETERMINISTIC VERDICT CEILING APPLIED — BUY → HOLD**\n"
        f"> Growth passed only through the {exception.replace('_', ' ')} "
        "exception, which permits at most HOLD."
    )
    rewritten, changed, n_header, n_block = _apply_verdict_rewrite(
        content_str,
        required_verdict="HOLD",
        allowed_from=frozenset({"BUY"}),
        note=note,
        ticker=ticker,
    )
    if changed:
        logger.info(
            "buy_capped_on_growth_exception",
            ticker=ticker,
            exception=exception,
            header_rewrites=n_header,
            block_rewrites=n_block,
        )
    return rewritten, changed


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

    note = (
        "> **DETERMINISTIC VERDICT DEMOTION APPLIED — BUY → HOLD**\n"
        f"> BUY-blocking flag(s) present: {', '.join(blocking)}. These mark the "
        "supporting evidence as indeterminate or unverified, so it may not back "
        "initiating a position. Resolve the flagged issue and re-run to restore "
        "BUY eligibility."
    )
    demoted, changed, n_header, n_block = _apply_verdict_rewrite(
        content_str,
        required_verdict="HOLD",
        allowed_from=frozenset({"BUY"}),
        note=note,
        ticker=ticker,
    )
    if not changed:
        return content_str, False

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


def maybe_qualify_weak_asymmetry_buy(
    content_str: str,
    *,
    weighted_upside: float | None,
    downside_probability: float | None,
    ticker: str = "UNKNOWN",
) -> tuple[str, bool]:
    """Append a 'weak valuation asymmetry' caveat to a BUY with thin upside.

    A BUY that trips ``is_weak_buy_asymmetry`` (thin probability-weighted
    intrinsic-value upside, or high downside probability — thresholds in
    ``src/thesis_constants.py``, shared with the memo valuation line) is not a
    clean buy. Mirrors ``maybe_qualify_buy_in_quick_mode``: appends a caveat
    note but leaves the ``VERDICT: BUY`` token intact so no downstream parser
    (charts, article, IBKR, run-summary) changes — the note rides the persisted
    PM text, and the run-summary flag is derived from marker presence, not this
    return value.

    Conservative and idempotent: fires only on a BUY that trips the shared
    predicate, and never a second time. Returns ``(content_str, qualified)``.
    """
    if weighted_upside is None:
        return content_str, False
    if pm_verdict_metadata_from_text(content_str).verdict != "BUY":
        return content_str, False
    if not is_weak_buy_asymmetry(weighted_upside, downside_probability or 0.0):
        return content_str, False
    if "WEAK VALUATION ASYMMETRY" in content_str:
        return content_str, True
    note = (
        "\n\n> **WEAK VALUATION ASYMMETRY — STARTER/VERIFY, NOT A CLEAN BUY**\n"
        f"> Weighted-IV upside is only {weighted_upside * 100:.1f}% with "
        f"{(downside_probability or 0.0):.0f}% downside probability. Size as a "
        "starter and confirm the entry before adding.\n"
    )
    logger.info("buy_qualified_weak_asymmetry", ticker=ticker)
    return content_str.rstrip() + note, True


DNI_REVIEW_CANDIDATE_MARKER = "QUALITY-GATE-PASSING DNI - REVIEW CANDIDATE"

# Category prefixes whose flags disqualify at ANY penalty: legal/tax/sanctions/
# mandate categories where even an "uncertain" signal makes a DNI a genuine avoid.
# REGULATORY_* is dynamically suffixed (supplemental_flags.py) — prefix match is
# required, a finite name set cannot cover it.
_REVIEW_DISQUALIFYING_PREFIXES = (
    "PFIC_",
    "VIE_",
    "CMIC_",
    "REGULATORY_",
    "CONSULTANT_",
)
# Named disqualifiers outside those categories. VALUE_TRAP_MODERATE_RISK (0.5) is a
# deliberate carve-out — a moderate governance signal is compatible with "review".
_REVIEW_DISQUALIFYING_FLAGS = frozenset({"VALUE_TRAP_HIGH_RISK", "VALUE_TRAP_VERDICT"})
# Any single flag carrying this much penalty is material regardless of name —
# future flags with novel names auto-disqualify with no denylist maintenance.
_REVIEW_DISQUALIFYING_PENALTY = 1.0


def _has_material_individual_risk(flag: dict) -> bool:
    """A single material penalty must not disappear into a net risk subtotal."""
    ftype = str(flag.get("type", ""))
    if ftype in _REVIEW_DISQUALIFYING_FLAGS:
        return True
    penalty = flag.get("risk_penalty")
    if isinstance(penalty, bool):
        return False
    return isinstance(penalty, int | float) and penalty >= _REVIEW_DISQUALIFYING_PENALTY


def _disqualifies_review_candidate(flag: dict) -> bool:
    return str(flag.get("type", "")).startswith(
        _REVIEW_DISQUALIFYING_PREFIXES
    ) or _has_material_individual_risk(flag)


def maybe_tag_dni_review_candidate(
    content_str: str, *, red_flags: list[dict], ticker: str = "UNKNOWN"
) -> tuple[str, bool]:
    """Tag a DO_NOT_INITIATE that clears the health/growth gates as a review candidate.

    Splits gate-passing DNIs from genuine avoids without asserting *why* the PM
    declined — liquidity, coverage, valuation, US-revenue, or data gaps may bind,
    and the predicate cannot distinguish them (strict mode also converts normal
    HOLDs and data-vacuum cases to DNI). Mirrors the BUY qualifiers: appends a
    caveat note but leaves the ``VERDICT:`` token intact so no downstream parser
    (charts, article, IBKR, run-summary) changes; the run-summary flag is derived
    from marker presence, never this return value.

    Disqualification is layered, most-structural first: any hard flag
    (AUTO_REJECT/CRITICAL), any ``blocks_buy`` flag (the gate arithmetic itself is
    indeterminate), any single flag with ``risk_penalty`` ≥ 1.0 (material by
    construction, covers future flags with novel names), and the legal/mandate
    category prefixes which disqualify even at 0.5 penalty.

    Scores are read from the PM's own decision text, NOT from ``DecisionInputs``
    — deliberately, and unlike `maybe_floor_verdict_to_hold`. This tag describes
    what the PM asserted about its own gates, so the PM text is the authoritative
    source for it; swapping in the canonical score would answer a different
    question (what the gates *should* have said) and make the tag disagree with
    the document it annotates. Do not "fix" this to use DecisionInputs.

    Conservative and idempotent. Returns ``(content_str, tagged)``.
    """
    if pm_verdict_metadata_from_text(content_str).verdict != "DO_NOT_INITIATE":
        return content_str, False
    if _has_hard_flag(red_flags):
        return content_str, False
    if any(flag.get("blocks_buy") is True for flag in red_flags):
        return content_str, False
    if any(_disqualifies_review_candidate(flag) for flag in red_flags):
        return content_str, False
    scores = parse_final_decision_scores(content_str)
    health = scores.get("health_adj")
    growth = scores.get("growth_adj")
    if health is None or growth is None:
        return content_str, False
    if health < HEALTH_MIN_PCT or growth < GROWTH_MIN_PCT:
        return content_str, False
    if DNI_REVIEW_CANDIDATE_MARKER in content_str:
        return content_str, True
    # Note wording constraints (ripple-audited): no parentheses — the PM-claim and
    # article citation audits scan un-backticked (...) groups; no `KEY:` uppercase
    # tokens or PASS/FAIL words — thesis_visualizer / parse_final_decision_scores /
    # discipline-check regexes key on those. The numbers restate values parsed from
    # this same document, so any parser that reads them stays self-consistent.
    note = (
        f"\n\n> **{DNI_REVIEW_CANDIDATE_MARKER}**\n"
        f"> This DO_NOT_INITIATE clears both hard quality gates — health "
        f"{health:.0f}%, growth {growth:.0f}% — and carries no critical, "
        "legal/governance, or material-penalty flag. This note does not assert why "
        "the PM declined; the binding constraint may be liquidity, coverage, "
        "valuation, US-revenue exposure, or data gaps — see the PM rationale above. "
        "Treat as a candidate for periodic re-review rather than a discard. Not an "
        "entry-timing signal.\n"
    )
    logger.info("dni_review_candidate", ticker=ticker, health=health, growth=growth)
    return content_str.rstrip() + note, True


def _has_hard_flag(red_flags: list[dict]) -> bool:
    return any(
        (flag.get("action") == "AUTO_REJECT") or (flag.get("severity") == "CRITICAL")
        for flag in red_flags
    )


def _auto_reject_flag_types(red_flags: list[dict]) -> tuple[str, ...]:
    """Return the canonical deterministic findings that require rejection."""
    return tuple(
        sorted(
            {
                str(flag.get("type") or "UNKNOWN")
                for flag in red_flags
                if str(flag.get("action") or "").upper() == "AUTO_REJECT"
            }
        )
    )


@dataclass(frozen=True, slots=True)
class GrowthGateAssessment:
    """One authoritative interpretation of the PM growth hard-fail contract."""

    hard_fail: bool
    exception: str | None = None
    reason: str | None = None

    @property
    def status(self) -> str:
        if self.exception:
            return f"EXCEPTION_{self.exception.upper()}"
        if self.hard_fail:
            return "HARD_FAIL"
        if self.reason in {
            "growth_score_unreliable",
            "growth_score_unavailable",
            "health_score_unreliable_for_growth_exception",
            "health_score_unavailable_for_growth_exception",
        }:
            return "INDETERMINATE"
        return "PASS"


@dataclass(frozen=True, slots=True)
class VerdictPolicyViolation:
    """A parsed PM verdict that contradicts deterministic decision policy."""

    rule: str
    reason: str
    required_verdict: str


@dataclass(frozen=True, slots=True)
class InitiationGateAssessment:
    """A shared code-owned gate assessment with evidence and applicability."""

    rule: str
    status: str
    reason: str
    value: float | int | None = None
    exception: str | None = None
    applicable: bool = True
    supporting_ids: tuple[str, ...] = ()


def assess_health_gate(inputs: DecisionInputs) -> InitiationGateAssessment:
    """Bind health and its documented exception to the same resolved snapshot."""
    health = inputs.health_decision_pct
    applicable = inputs.decision_metrics.get("m_and_a_status") != "ACTIVE_TENDER"
    if (
        not inputs.health_score_reliable
        or not isinstance(health, int | float)
        or isinstance(health, bool)
        or not 0 <= health <= 100
        or not math.isfinite(health)
    ):
        return InitiationGateAssessment(
            "health",
            "INDETERMINATE",
            "reliable_health_score_unavailable",
            applicable=applicable,
        )
    status, reason, exception = "PASS", "reliable_health_at_or_above_floor", None
    if health < HEALTH_MIN_PCT:
        pb, ratio = (
            inputs.health_exception_pb_ratio,
            inputs.health_exception_current_ratio,
        )
        qualified = (
            inputs.snapshot_authoritative
            and health >= HEALTH_EXCEPTION_MIN_PCT
            and pb is not None
            and 0 <= pb < HEALTH_EXCEPTION_PB_MAX
            and math.copysign(1.0, pb) > 0
            and ratio is not None
            and HEALTH_EXCEPTION_CURRENT_RATIO_MIN < ratio < math.inf
            and len(inputs.health_exception_claim_ids) == 2
        )
        if qualified:
            reason, exception = (
                "canonical_deep_value_liquidity_exception",
                "deep_value_liquidity",
            )
        else:
            status, reason = (
                "FAIL",
                (
                    "reliable_health_below_exception_floor"
                    if health < HEALTH_EXCEPTION_MIN_PCT
                    else "health_exception_not_established"
                ),
            )
    return InitiationGateAssessment(
        "health",
        status,
        reason,
        health,
        exception,
        applicable,
        inputs.health_exception_claim_ids if exception else (),
    )


def health_gate_from_policy(policy: Mapping[str, Any]) -> InitiationGateAssessment:
    """Absent legacy or malformed canonical health remains explicitly unknown."""
    row = policy.get("health_gate")
    if (
        type(policy.get("schema_version")) is not int
        or policy["schema_version"] != 1
        or not isinstance(row, Mapping)
    ):
        row = {}
    value, status = row.get("value"), row.get("status")
    if (
        not isinstance(status, str)
        or status not in {"PASS", "FAIL", "INDETERMINATE"}
        or (value is None and status != "INDETERMINATE")
        or (
            value is not None
            and (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not 0 <= value <= 100
                or not math.isfinite(value)
            )
        )
        or not isinstance(row.get("applicable", True), bool)
    ):
        return InitiationGateAssessment(
            "health", "INDETERMINATE", "canonical_health_gate_unavailable"
        )
    ids = row.get("supporting_ids", ())
    exception = row.get("exception")
    if (
        (exception is not None and exception != "deep_value_liquidity")
        or not isinstance(ids, list | tuple)
        or any(not isinstance(i, str) or not i for i in ids)
    ):
        return InitiationGateAssessment(
            "health", "INDETERMINATE", "canonical_health_gate_unavailable"
        )
    if (
        status == "PASS"
        and value is not None
        and value < HEALTH_MIN_PCT
        and not (
            value >= HEALTH_EXCEPTION_MIN_PCT
            and exception == "deep_value_liquidity"
            and len(set(ids)) == 2
        )
    ):
        return InitiationGateAssessment(
            "health", "INDETERMINATE", "canonical_health_gate_unavailable"
        )
    return InitiationGateAssessment(
        "health",
        status,
        str(row.get("reason") or "canonical_health_gate_unavailable"),
        value,
        exception,
        row.get("applicable", True),
        tuple(ids),
    )


def assess_initiation_gates(
    inputs: DecisionInputs,
) -> tuple[InitiationGateAssessment, ...]:
    count = inputs.english_analyst_count
    us_revenue = inputs.us_revenue_pct
    liquidity = inputs.liquidity
    return (
        InitiationGateAssessment(
            "analyst_coverage",
            "INDETERMINATE"
            if count is None
            else "FAIL"
            if count >= ANALYST_COVERAGE_MAX
            else "PASS",
            "verified_english_coverage_unavailable"
            if count is None
            else f"verified_english_analyst_count={count}",
            value=count,
        ),
        InitiationGateAssessment(
            "us_revenue",
            "INDETERMINATE"
            if us_revenue is None
            else "FAIL"
            if us_revenue > US_REVENUE_MAX_PCT
            else "PASS",
            "verified_us_revenue_unavailable"
            if us_revenue is None
            else f"verified_us_revenue_pct={us_revenue:g}",
            value=us_revenue,
        ),
        InitiationGateAssessment(
            "liquidity",
            "INDETERMINATE"
            if liquidity is None or liquidity.status in {"INSUFFICIENT_DATA", "ERROR"}
            else "FAIL"
            if liquidity.hard_fail
            else "PASS",
            "measured_liquidity_unavailable"
            if liquidity is None
            else liquidity.reason or liquidity.status,
            value=liquidity.average_daily_turnover_usd if liquidity else None,
        ),
    )


_GATE_LABELS = {
    "liquidity": "Liquidity",
    "analyst_coverage": "Analyst Coverage",
    "us_revenue": "US Revenue",
}


def format_initiation_gate(gate: InitiationGateAssessment) -> str:
    """One readable projection of canonical status and independently bound value."""
    value = "Unknown"
    if gate.value is not None:
        value = (
            f"${gate.value:,.0f} daily average"
            if gate.rule == "liquidity"
            else f"{gate.value:g}%"
            if gate.rule in {"us_revenue", "health"}
            else f"{gate.value:g} English analysts"
        )
    unavailable = {
        "verified_english_coverage_unavailable": "Verified English analyst coverage unavailable",
        "verified_us_revenue_unavailable": "Verified US revenue unavailable",
        "measured_liquidity_unavailable": "Measured liquidity unavailable",
        "reliable_health_score_unavailable": "Reliable financial health unavailable",
        "canonical_health_gate_unavailable": "Canonical financial health unavailable",
        "canonical_deep_value_liquidity_exception": "Established canonical P/B and current-ratio exception",
        "health_exception_not_established": "Below health floor; qualifying exception not established",
    }
    reason = unavailable.get(
        gate.reason, gate.reason.replace("_", " ").replace("=", ": ")
    )
    label = "Financial Health" if gate.rule == "health" else _GATE_LABELS[gate.rule]
    exemption = "; Exempt under ACTIVE_TENDER" if not gate.applicable else ""
    return f"{label}: {value} - {gate.status}; {reason}{exemption}"


def initiation_gates_from_policy(
    policy: Mapping[str, Any],
) -> tuple[InitiationGateAssessment, ...]:
    """Decode display data conservatively; legacy policies may lack measurements."""
    raw = policy.get("initiation_gates")
    rows = (
        raw
        if type(policy.get("schema_version")) is int
        and policy["schema_version"] == 1
        and isinstance(raw, list)
        else []
    )
    gates = []
    for rule in _GATE_LABELS:
        owned = [
            row for row in rows if isinstance(row, Mapping) and row.get("rule") == rule
        ]
        row = owned[0] if len(owned) == 1 else {}
        status = row.get("status")
        status = (
            status
            if isinstance(status, str) and status in {"PASS", "FAIL", "INDETERMINATE"}
            else "INDETERMINATE"
        )
        value = row.get("value")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or abs(value) > 1e308
            or not math.isfinite(value)
            or value < 0
            or (rule == "analyst_coverage" and int(value) != value)
            or (rule == "us_revenue" and value > 100)
        ):
            value = None
        reason = row.get("reason")
        gates.append(
            InitiationGateAssessment(
                rule,
                status,
                reason
                if isinstance(reason, str)
                else "Canonical gate data unavailable",
                value,
            )
        )
    return tuple(gates)


def compliance_hard_fail_result(
    gates: tuple[InitiationGateAssessment, ...],
    growth_status: str,
    health: InitiationGateAssessment | None = None,
) -> str:
    """Display applicable code-owned gates; tender exempts health and growth."""
    statuses = [gate.status for gate in gates]
    if health is None or health.applicable:
        statuses.append(growth_status)
        if health is not None:
            statuses.append(health.status)
    if any(status in {"FAIL", "HARD_FAIL"} for status in statuses):
        return "FAIL"
    return "INDETERMINATE" if "INDETERMINATE" in statuses else "PASS"


def format_compliance_result(
    gates: tuple[InitiationGateAssessment, ...],
    growth_status: str,
    health: InitiationGateAssessment,
) -> str:
    result = compliance_hard_fail_result(gates, growth_status, health)
    if result != "INDETERMINATE":
        return result
    count = sum(gate.status == "INDETERMINATE" for gate in gates)
    if health.applicable:
        count += int(health.status == "INDETERMINATE") + int(
            growth_status == "INDETERMINATE"
        )
    return f"{result} — No verified hard failure among these gates; {count} gates unverified"


def render_initiation_gate_context(inputs: DecisionInputs) -> str:
    return (
        "CODE-OWNED INITIATION GATES (independent of growth exceptions):\n"
        + "- "
        + format_initiation_gate(assess_health_gate(inputs))
        + "\n"
        + "\n".join(
            f"- {format_initiation_gate(gate)}"
            for gate in assess_initiation_gates(inputs)
        )
    )


def project_compliance_gates(content: str, inputs: DecisionInputs) -> str:
    """Replace only owned compliance labels; preserve discretionary risk prose."""
    section = re.search(
        r"(?ms)^### THESIS COMPLIANCE SUMMARY\s*\n(?P<body>.*?)(?=^### |\Z)", content
    )
    if section is None:
        return content
    body = section.group("body")
    body = re.sub(
        r"(?m)^\s*(?:[-*]\s*)?(?:\*\*)?(?:Financial Health|Growth Transition|Liquidity|Analyst Coverage|US Revenue|Hard Fail Result)(?:\*\*)?:[^\n]*\n?",
        "",
        body,
    )
    explanation = "Unresolved evidence is INDETERMINATE. Independent economic risks and the PM rationale retain their policy authority."
    body = body.replace(explanation + "\n", "")
    # These assessments are exactly those persisted in decision_policy.
    growth = assess_growth_gate(inputs)
    health = assess_health_gate(inputs)
    score = (
        "N/A"
        if inputs.growth_decision_pct is None
        else f"{inputs.growth_decision_pct:g}%"
    )
    lines = [
        "- **" + format_initiation_gate(health).replace(":", "**:", 1),
        f"- **Growth Transition**: {score} (Adjusted) - {growth.status}; {growth.reason or 'NONE'}"
        + ("; Exempt under ACTIVE_TENDER" if not health.applicable else ""),
    ]
    gates = assess_initiation_gates(inputs)
    lines.extend(
        "- **" + format_initiation_gate(gate).replace(":", "**:", 1) for gate in gates
    )
    lines.append(
        f"**Hard Fail Result**: {format_compliance_result(gates, growth.status, health)}"
    )
    lines.append(explanation)
    heading = re.search(r"(?m)^\*\*Hard Fail Checks:?\*\*:?[ \t]*\n", body)
    rows = "\n".join(lines) + "\n\n"
    if heading is None:
        body = "**Hard Fail Checks:**\n" + rows + body.lstrip("\n")
    else:
        body = body[: heading.end()] + rows + body[heading.end() :].lstrip("\n")
    return content[: section.start("body")] + body + content[section.end("body") :]


def render_growth_gate_context(decision_inputs: DecisionInputs) -> str:
    """Render the code-owned growth decision for the PM without restating policy."""
    assessment = assess_growth_gate(decision_inputs)
    missing = ",".join(sorted(decision_inputs.missing_current_growth_fields)) or "NONE"
    return (
        "CODE-OWNED GROWTH GATE (binding; do not infer a different exception):\n"
        f"- STATUS: {assessment.status}\n"
        f"- REASON: {assessment.reason or 'NONE'}\n"
        f"- MISSING_CURRENT_GROWTH_FIELDS: {missing}"
        + (
            "\n- MAX_VERDICT: HOLD (growth passed by exception)"
            if _exception_growth_ceiling_applies(decision_inputs, assessment)
            else ""
        )
    )


def assess_growth_gate(decision_inputs: DecisionInputs) -> GrowthGateAssessment:
    """Evaluate the growth hard fail and its two documented exceptions.

    This predicate is shared by verdict validation and the deterministic
    DO_NOT_INITIATE→HOLD floor. Missing evidence never satisfies an exception.
    """

    if not decision_inputs.growth_score_reliable:
        return GrowthGateAssessment(
            hard_fail=False,
            reason="growth_score_unreliable",
        )

    metrics = decision_inputs.decision_metrics
    growth = decision_inputs.growth_decision_pct
    if growth is None:
        return GrowthGateAssessment(hard_fail=False, reason="growth_score_unavailable")
    if growth >= GROWTH_FAIL_MAX:
        return GrowthGateAssessment(hard_fail=False)

    if not decision_inputs.health_score_reliable:
        return GrowthGateAssessment(
            hard_fail=False, reason="health_score_unreliable_for_growth_exception"
        )

    health = decision_inputs.health_decision_pct
    pe = metrics.get("pe_ratio")
    cagr = metrics.get("revenue_cagr_3y")
    if health is None:
        return GrowthGateAssessment(
            hard_fail=False, reason="health_score_unavailable_for_growth_exception"
        )
    if health < HEALTH_FLOOR_MIN:
        return GrowthGateAssessment(
            hard_fail=True,
            reason="adjusted_health_below_exception_floor",
        )
    if cagr is None or cagr < 0:
        return GrowthGateAssessment(
            hard_fail=True,
            reason="nonnegative_multi_year_revenue_not_established",
        )
    if pe is None:
        return GrowthGateAssessment(
            hard_fail=True,
            reason="pe_ratio_required_for_growth_exception",
        )
    if pe <= 13.0:
        return GrowthGateAssessment(
            hard_fail=False,
            exception="marginal_turnaround",
        )

    if pe <= PE_FLOOR_MAX and decision_inputs.growth_data_vacuum:
        return GrowthGateAssessment(
            hard_fail=False,
            exception="data_vacuum",
        )

    return GrowthGateAssessment(
        hard_fail=True,
        reason=(
            "data_vacuum_growth_inputs_not_established"
            if pe <= PE_FLOOR_MAX
            else "pe_ratio_above_growth_exception_ceiling"
        ),
    )


def assess_verdict_policy(
    content_str: str,
    *,
    decision_inputs: DecisionInputs,
    red_flags: list[dict] | None = None,
) -> VerdictPolicyViolation | None:
    """Return a deterministic policy violation for an impermissive verdict.

    The first covered rule is the growth hard fail that caused provider-dependent
    HOLD/DNI behavior. This is intentionally housed beside the existing verdict
    normalizers so parsing, correction, and post-processing have one authority.
    """

    verdict = pm_verdict_metadata_from_text(content_str).verdict
    if verdict not in {"BUY", "HOLD"}:
        return None
    auto_reject_flags = _auto_reject_flag_types(red_flags or [])
    if auto_reject_flags:
        return VerdictPolicyViolation(
            rule="pre_screening_auto_reject",
            reason="auto_reject_flags=" + ",".join(auto_reject_flags),
            required_verdict="DO_NOT_INITIATE",
        )
    for gate in assess_initiation_gates(decision_inputs):
        if gate.status == "FAIL":
            return VerdictPolicyViolation(gate.rule, gate.reason, "DO_NOT_INITIATE")
    if decision_inputs.decision_metrics.get("m_and_a_status") == "ACTIVE_TENDER":
        return None

    health = assess_health_gate(decision_inputs)
    if health.status == "FAIL":
        return VerdictPolicyViolation(
            "financial_health_hard_fail", health.reason, "DO_NOT_INITIATE"
        )

    growth_gate = assess_growth_gate(decision_inputs)
    if not growth_gate.hard_fail:
        return None
    return VerdictPolicyViolation(
        rule="growth_transition_hard_fail",
        reason=growth_gate.reason or "growth_exception_not_established",
        required_verdict="DO_NOT_INITIATE",
    )


def maybe_floor_verdict_to_hold(
    content_str: str,
    *,
    decision_inputs: DecisionInputs,
    red_flags: list[dict],
    code_subtotal: float | None,
    pre_screening_result: str | None,
    ticker: str = "UNKNOWN",
) -> tuple[str, bool]:
    """Floor a soft-point DO_NOT_INITIATE to HOLD for a healthy, data-limited name.

    Scores come from ``DecisionInputs`` — the one authority resolver — not from a
    private reparse of the DATA_BLOCK. Under a VALID canonical contract that is
    the projected value either way; the difference shows when the contract did
    NOT validate, where the parsed score is unprojected model arithmetic and
    ``*_score_reliable`` is False. An upgrade of the verdict must never rest on a
    score the canonical layer would not stand behind.

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
    # A positive bonus can pull the net subtotal below Zone 1 without resolving
    # a material individual risk. Review tagging shares this individual check.
    if any(_has_material_individual_risk(flag) for flag in red_flags):
        return content_str, False
    if code_subtotal is None or code_subtotal >= ZONE_1_THRESHOLD:
        return content_str, False
    if not (
        decision_inputs.health_score_reliable and decision_inputs.growth_score_reliable
    ):
        logger.info(
            "verdict_floor_skipped_unreliable_scores",
            ticker=ticker,
            snapshot_status=decision_inputs.snapshot_status,
        )
        return content_str, False

    if assess_health_gate(decision_inputs).status != "PASS":
        return content_str, False

    if any(
        gate.status == "FAIL" or (gate.rule != "us_revenue" and gate.status != "PASS")
        for gate in assess_initiation_gates(decision_inputs)
    ):
        return content_str, False
    growth_gate = assess_growth_gate(decision_inputs)
    if growth_gate.hard_fail or growth_gate.exception is None:
        return content_str, False

    metrics = decision_inputs.decision_metrics
    health = decision_inputs.health_decision_pct
    growth = decision_inputs.growth_decision_pct
    pe = metrics.get("pe_ratio")
    cagr = metrics.get("revenue_cagr_3y")
    if health is None or growth is None or pe is None or cagr is None:
        return content_str, False

    note = (
        "> **DETERMINISTIC VERDICT FLOOR APPLIED — DO NOT INITIATE → HOLD**\n"
        f"> The growth-score failure qualifies for the Data-Vacuum / Marginal-Turnaround "
        f"exception (Adjusted Health {health:.0f}% ≥ 65, P/E {pe:.2f} ≤ 18, "
        f"3Y revenue CAGR {cagr:.1f}% ≥ 0), and the deterministic code-computed risk "
        f"subtotal ({code_subtotal:+.2f}) is below the Zone-1 threshold ({ZONE_1_THRESHOLD}) "
        f"without a material disqualifying flag. A soft-point tally may not convert a healthy, "
        f"data-limited name into an avoid/exit."
    )
    floored, changed, n_header, n_block = _apply_verdict_rewrite(
        content_str,
        required_verdict="HOLD",
        allowed_from=frozenset({"DO_NOT_INITIATE"}),
        note=note,
        ticker=ticker,
    )
    if not changed:
        return content_str, False

    logger.info(
        "verdict_floored_to_hold",
        ticker=ticker,
        code_subtotal=round(code_subtotal, 2),
        adjusted_health=health,
        pe_ratio=pe,
        revenue_cagr_3y=cagr,
        growth_score=growth,
        growth_exception=growth_gate.exception,
        header_rewrites=n_header,
        block_rewrites=n_block,
    )
    return floored, True
