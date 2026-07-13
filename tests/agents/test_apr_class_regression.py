"""End-to-end regression for the APR.WA false-reject class (and KTY counter-case).

CI-safe: embeds the decisive DATA_BLOCK / VALUE_TRAP field values (results/ is gitignored)
and drives the real flag-assembly + verdict-floor path. Asserts the fix flips APR
DO_NOT_INITIATE -> HOLD while KTY (a genuine ex-growth payer) still fails.
"""

from __future__ import annotations

from src.agents.support import format_red_flag_section
from src.agents.verdict_policy import maybe_floor_verdict_to_hold
from src.validators.red_flag_detector import RedFlagDetector


def _data_block(**f: str) -> str:
    body = "\n".join(f"{k}: {v}" for k, v in f.items())
    return f"### --- START DATA_BLOCK ---\n{body}\n### --- END DATA_BLOCK ---"


def _value_trap(score: int, verdict: str, rating: str = "POOR") -> str:
    # Real APR.WA value-trap: low score driven by a POOR capital-allocation rating
    # (the detector ran blind to ROIC and mislabeled reinvestment).
    return (
        "VALUE_TRAP_BLOCK\n"
        f"SCORE: {score}\nVERDICT: {verdict}\nTRAP_RISK: HIGH\n"
        "ACTIVIST_PRESENT: NO\n"
        f"CAPITAL_ALLOCATION:\n  RATING: {rating}\n"
        "CATALYSTS:\n  INDEX_CANDIDATE: NONE\n  ACTIVIST_RUMOR: NONE\n"
        "  RESTRUCTURING: NONE\n  MID_TERM_PLAN: NONE\n"
    )


def _capital_context(db: str) -> dict[str, str]:
    from src.data_block_utils import extract_data_block_field

    return {
        "capex_to_da_status": extract_data_block_field(db, "CAPEX_TO_DA_STATUS") or "",
        "roic_quality": extract_data_block_field(db, "ROIC_QUALITY") or "",
        "capital_plan_status": extract_data_block_field(db, "CAPITAL_PLAN_STATUS")
        or "",
    }


def _assemble(db: str, vt: str) -> tuple[list[dict], float]:
    flags: list[dict] = []
    flags.extend(RedFlagDetector.detect_return_quality_fragility_flags(db, "T"))
    flags.extend(
        RedFlagDetector.detect_value_trap_flags(
            vt, "T", capital_context=_capital_context(db)
        )
    )
    _, subtotal = format_red_flag_section("PASS", flags)
    return flags, subtotal


def _pm_dni() -> str:
    return (
        "### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE\n\n"
        "```\nPM_BLOCK\nVERDICT: DO_NOT_INITIATE\nPOSITION_SIZE: 0.0%\n```\n"
    )


# Real APR.WA fields (2026-06-28 artifact).
APR_DB = _data_block(
    ADJUSTED_HEALTH_SCORE="75% (based on 12 available points)",
    ADJUSTED_GROWTH_SCORE="33% (based on 6 available points)",
    PE_RATIO_TTM="14.88",
    REVENUE_CAGR_3Y="16.0%",
    PROFITABILITY_TREND="DECLINING",
    ROA_PERCENT="8.77%",
    ROA_5Y_AVG="11.12%",
    CAPEX_TO_DA_STATUS="GROWTH_INVESTING",
    ROIC_QUALITY="ADEQUATE",
    CAPITAL_PLAN_STATUS="EXPLICIT",
)
# Real KTY.WA fields: ex-growth payout machine (P/E>18, negative 3Y CAGR, MAINTENANCE).
KTY_DB = _data_block(
    ADJUSTED_HEALTH_SCORE="67% (based on 12 available points)",
    ADJUSTED_GROWTH_SCORE="33% (based on 6 available points)",
    PE_RATIO_TTM="20.54",
    REVENUE_CAGR_3Y="-2.5%",
    PROFITABILITY_TREND="DECLINING",
    ROA_PERCENT="13.5%",
    ROA_5Y_AVG="15.8%",
    CAPEX_TO_DA_STATUS="MAINTENANCE",
    ROIC_QUALITY="STRONG",
    CAPITAL_PLAN_STATUS="EXPLICIT",
)


def test_apr_value_trap_downgraded_no_rqf_subtotal_below_zone1():
    flags, subtotal = _assemble(APR_DB, _value_trap(35, "TRAP"))
    types = [f["type"] for f in flags]
    assert "VALUE_TRAP_HIGH_RISK" not in types  # downgraded
    assert "VALUE_TRAP_VERDICT" not in types
    assert "VALUE_TRAP_MODERATE_RISK" in types  # to +0.5
    assert "RETURN_QUALITY_FRAGILITY" not in types  # DECLINING + 11.12% avg -> no fire
    # VT MODERATE (0.5) + NO_CATALYST (0.5) = 1.0 < Zone-1 (2.0)
    assert subtotal < 2.0


def test_apr_verdict_floored_to_hold():
    _, subtotal = _assemble(APR_DB, _value_trap(35, "TRAP"))
    out, floored = maybe_floor_verdict_to_hold(
        _pm_dni(),
        fundamentals_report=APR_DB,
        red_flags=_assemble(APR_DB, _value_trap(35, "TRAP"))[0],
        code_subtotal=subtotal,
        pre_screening_result="PASS",
        ticker="APR.WA",
    )
    assert floored is True
    assert "VERDICT: HOLD" in out and "VERDICT: DO_NOT_INITIATE" not in out


def test_kty_value_trap_not_downgraded():
    # KTY is MAINTENANCE (not GROWTH_INVESTING) -> reinvestment downgrade must NOT apply.
    flags, _ = _assemble(KTY_DB, _value_trap(35, "TRAP"))
    assert "VALUE_TRAP_HIGH_RISK" in [f["type"] for f in flags]


def test_kty_not_floored_pe_and_negative_cagr():
    out, floored = maybe_floor_verdict_to_hold(
        _pm_dni(),
        fundamentals_report=KTY_DB,
        red_flags=[],
        code_subtotal=1.0,
        pre_screening_result="PASS",
        ticker="KTY.WA",
    )
    assert floored is False
    assert "VERDICT: DO_NOT_INITIATE" in out
