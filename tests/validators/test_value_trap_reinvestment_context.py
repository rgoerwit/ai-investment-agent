"""Step 4: value-trap penalty downgrade when DATA_BLOCK confirms growth investment.

The Value-Trap Detector runs blind to ROIC/capex and mislabeled APR.WA's reinvestment
as POOR allocation (score 35 -> VALUE_TRAP_HIGH_RISK +1.0). When the Senior Fundamentals
DATA_BLOCK independently shows GROWTH_INVESTING + ADEQUATE/STRONG ROIC + EXPLICIT plan,
the HIGH/TRAP penalty is downgraded to MODERATE (+0.5) -- never removed.
"""

from __future__ import annotations

from src.validators.supplemental_flags import detect_value_trap_flags

_GROWTH_CTX = {
    "capex_to_da_status": "GROWTH_INVESTING",
    "roic_quality": "ADEQUATE",
    "capital_plan_status": "EXPLICIT",
}
_HOARDER_CTX = {
    "capex_to_da_status": "MAINTENANCE",
    "roic_quality": "WEAK",
    "capital_plan_status": "NONE",
}


def _block(score: int, verdict: str, rating: str = "POOR") -> str:
    return (
        "VALUE_TRAP_BLOCK\n"
        f"SCORE: {score}\n"
        f"VERDICT: {verdict}\n"
        "TRAP_RISK: HIGH\n"
        "ACTIVIST_PRESENT: NO\n"
        f"CAPITAL_ALLOCATION:\n  RATING: {rating}\n"
        "CATALYSTS:\n  INDEX_CANDIDATE: NONE\n  ACTIVIST_RUMOR: NONE\n"
        "  RESTRUCTURING: NONE\n  MID_TERM_PLAN: NONE\n"
    )


def _penalty(flags, type_):
    return sum(f["risk_penalty"] for f in flags if f["type"] == type_)


def test_high_risk_downgraded_to_moderate_under_growth_context():
    flags = detect_value_trap_flags(
        _block(35, "TRAP"), "APR", capital_context=_GROWTH_CTX
    )
    types = {f["type"] for f in flags}
    assert "VALUE_TRAP_HIGH_RISK" not in types
    assert "VALUE_TRAP_VERDICT" not in types
    assert _penalty(flags, "VALUE_TRAP_MODERATE_RISK") == 0.5
    # NO_CATALYST is NOT auto-dropped by the reinvestment downgrade.
    assert "NO_CATALYST_DETECTED" in types


def test_no_downgrade_without_context():
    flags = detect_value_trap_flags(_block(35, "TRAP"), "X")
    assert _penalty(flags, "VALUE_TRAP_HIGH_RISK") == 1.0


def test_no_downgrade_when_rating_not_poor():
    # Governance/no-catalyst trap (RATING GOOD) must NOT be whitewashed by growth context.
    flags = detect_value_trap_flags(
        _block(35, "TRAP", rating="GOOD"), "X", capital_context=_GROWTH_CTX
    )
    assert _penalty(flags, "VALUE_TRAP_HIGH_RISK") == 1.0


def test_no_downgrade_when_rating_absent():
    flags = detect_value_trap_flags(
        "VALUE_TRAP_BLOCK\nSCORE: 35\nVERDICT: TRAP\nACTIVIST_PRESENT: NO\n",
        "X",
        capital_context=_GROWTH_CTX,
    )
    assert _penalty(flags, "VALUE_TRAP_HIGH_RISK") == 1.0


def test_no_downgrade_for_hoarder_context():
    flags = detect_value_trap_flags(
        _block(35, "TRAP"), "X", capital_context=_HOARDER_CTX
    )
    assert _penalty(flags, "VALUE_TRAP_HIGH_RISK") == 1.0


def test_partial_context_does_not_downgrade():
    ctx = dict(_GROWTH_CTX, roic_quality="WEAK")
    flags = detect_value_trap_flags(_block(35, "TRAP"), "X", capital_context=ctx)
    assert _penalty(flags, "VALUE_TRAP_HIGH_RISK") == 1.0


def test_verdict_trap_above_40_downgrades_to_single_moderate():
    # score 45 (no HIGH), verdict TRAP -> normally VERDICT +1.0; with growth ctx -> +0.5 once
    flags = detect_value_trap_flags(
        _block(45, "TRAP"), "X", capital_context=_GROWTH_CTX
    )
    assert _penalty(flags, "VALUE_TRAP_VERDICT") == 0.0
    assert (
        _penalty(flags, "VALUE_TRAP_MODERATE_RISK") == 0.5
    )  # the 40-60 band, not doubled


def test_downgraded_penalty_does_not_double_count_verdict():
    # score 35 + verdict TRAP + growth ctx -> exactly one +0.5, not 0.5 + 0.5
    flags = detect_value_trap_flags(
        _block(35, "TRAP"), "X", capital_context=_GROWTH_CTX
    )
    assert _penalty(flags, "VALUE_TRAP_MODERATE_RISK") == 0.5
