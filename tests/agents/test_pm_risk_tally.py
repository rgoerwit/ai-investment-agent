"""Tests for the deterministic PM risk-tally floor.

Covers:
- ``support.format_red_flag_section``: surfaces each code-computed ``risk_penalty`` and a
  running subtotal so the synthesis model can't silently drop a weighted pre-screen penalty.
- ``decision_nodes._log_risk_tally_reconciliation``: warns (no override) when the PM's
  narrated TOTAL RISK COUNT falls below that deterministic floor.

These guard the Flash-PM "dropped mandated penalty" regression (FRAGUAB.MX, June 2026).
"""

from __future__ import annotations

import logging

from src.agents.decision_nodes import (
    _log_pm_discipline_checks,
    _log_risk_tally_reconciliation,
)
from src.agents.support import format_red_flag_section


# --------------------------------------------------------------------------- #
# format_red_flag_section
# --------------------------------------------------------------------------- #
def test_subtotal_sums_signed_penalties_and_renders_tags() -> None:
    flags = [
        {"type": "NO_CATALYST_DETECTED", "detail": "no activist", "risk_penalty": 0.5},
        {"type": "VALUE_TRAP_MODERATE_RISK", "detail": "score 55", "risk_penalty": 0.5},
        {
            "type": "MOAT_DURABLE_ADVANTAGE",
            "detail": "pricing power",
            "risk_penalty": -1.0,
        },
    ]
    text, subtotal = format_red_flag_section("PASS", flags)
    assert subtotal == 0.0
    assert "[risk_penalty +0.50]" in text
    assert "[risk_penalty -1.00]" in text
    assert (
        "CODE-COMPUTED RISK SUBTOTAL (deterministic, already weighted): +0.00" in text
    )
    # The anti-double-count instruction must be present.
    assert "do NOT re-score them" in text
    assert "do NOT omit them" in text


def test_empty_flags_returns_none_marker_and_zero() -> None:
    text, subtotal = format_red_flag_section("PASS", [])
    assert subtotal == 0.0
    assert text.endswith("Red Flags Detected: None")
    assert "CODE-COMPUTED RISK SUBTOTAL" not in text  # nothing to anchor on


def test_missing_penalty_key_counts_zero_and_no_tag() -> None:
    text, subtotal = format_red_flag_section("PASS", [{"type": "X", "detail": "y"}])
    assert subtotal == 0.0
    assert "[risk_penalty" not in text
    assert "  - X: y" in text


def test_bool_penalty_is_not_treated_as_numeric() -> None:
    # bool is an int subclass; it must not be summed or tagged as a weight.
    text, subtotal = format_red_flag_section(
        "PASS", [{"type": "B", "detail": "d", "risk_penalty": True}]
    )
    assert subtotal == 0.0
    assert "[risk_penalty" not in text


def test_string_penalty_is_ignored_not_crashed() -> None:
    text, subtotal = format_red_flag_section(
        "PASS", [{"type": "S", "detail": "d", "risk_penalty": "1.0"}]
    )
    assert subtotal == 0.0
    assert "[risk_penalty" not in text


def test_missing_type_and_detail_use_defaults() -> None:
    text, _ = format_red_flag_section("PASS", [{"risk_penalty": 0.25}])
    assert "  - Unknown [risk_penalty +0.25]: No detail" in text


def test_net_positive_subtotal_is_floored() -> None:
    flags = [
        {"type": "CMIC_FLAGGED", "detail": "x", "risk_penalty": 2.0},
        {"type": "PFIC_UNCERTAIN", "detail": "x", "risk_penalty": 0.5},
    ]
    _, subtotal = format_red_flag_section("PASS", flags)
    assert subtotal == 2.5


# --------------------------------------------------------------------------- #
# _log_risk_tally_reconciliation
# --------------------------------------------------------------------------- #
_PM_BLOCK_TEMPLATE = (
    "PORTFOLIO MANAGER VERDICT: BUY\nTOTAL RISK COUNT: {tally}\nZONE: LOW"
)


def test_warns_when_narrated_below_floor(caplog) -> None:
    content = _PM_BLOCK_TEMPLATE.format(tally="0.75")
    with caplog.at_level(logging.WARNING):
        dropped = _log_risk_tally_reconciliation(content, code_subtotal=1.5, ticker="T")
    assert dropped == 0.75
    assert "pm_risk_tally_below_code_floor" in caplog.text


def test_no_warn_when_narrated_meets_floor(caplog) -> None:
    content = _PM_BLOCK_TEMPLATE.format(tally="1.5")
    with caplog.at_level(logging.WARNING):
        dropped = _log_risk_tally_reconciliation(content, code_subtotal=1.5, ticker="T")
    assert dropped is None
    assert "pm_risk_tally_below_code_floor" not in caplog.text


def test_tolerance_band_does_not_warn_on_rounding(caplog) -> None:
    # narrated 1.49 vs floor 1.5 is within the 0.01 tolerance -> no warn.
    content = _PM_BLOCK_TEMPLATE.format(tally="1.49")
    with caplog.at_level(logging.WARNING):
        dropped = _log_risk_tally_reconciliation(content, code_subtotal=1.5, ticker="T")
    assert dropped is None


def test_unparseable_tally_does_not_warn(caplog) -> None:
    content = "PORTFOLIO MANAGER VERDICT: BUY\n(no tally line at all)"
    with caplog.at_level(logging.WARNING):
        dropped = _log_risk_tally_reconciliation(content, code_subtotal=1.5, ticker="T")
    assert dropped is None
    assert "pm_risk_tally_below_code_floor" not in caplog.text


def test_negative_floor_from_net_bonus_still_reconciles(caplog) -> None:
    # A net-bonus subtotal can be negative; a narrated tally below it still warns.
    content = _PM_BLOCK_TEMPLATE.format(tally="-1.5")
    with caplog.at_level(logging.WARNING):
        dropped = _log_risk_tally_reconciliation(
            content, code_subtotal=-1.0, ticker="T"
        )
    assert dropped == 0.5
    assert "pm_risk_tally_below_code_floor" in caplog.text


# --------------------------------------------------------------------------- #
# _log_pm_discipline_checks (override + buy-on-quarantined, log-only)
# --------------------------------------------------------------------------- #
def _pm(verdict: str, zone: str, *, health=None, growth=None, risk=None) -> str:
    lines = [f"PORTFOLIO MANAGER VERDICT: {verdict}", f"ZONE: {zone}"]
    if health is not None:
        lines.append(f"HEALTH_ADJ: {health}")
    if growth is not None:
        lines.append(f"GROWTH_ADJ: {growth}")
    if risk is not None:
        lines.append(f"TOTAL RISK COUNT: {risk}")
    return "\n".join(lines)


def test_zone2_buy_within_thresholds_is_silent(caplog) -> None:
    # ALV-style legitimate override: health high, risk <= 1.5, no blocking flag.
    content = _pm("BUY", "MODERATE", health=80, risk=1.0)
    with caplog.at_level(logging.WARNING):
        _log_pm_discipline_checks(content, [], "USABLE", "T")
    assert "pm_override_threshold_unmet" not in caplog.text


def test_zone2_buy_low_health_warns(caplog) -> None:
    content = _pm("BUY", "MODERATE", health=40, risk=1.0)
    with caplog.at_level(logging.WARNING):
        _log_pm_discipline_checks(content, [], "USABLE", "T")
    assert "pm_override_threshold_unmet" in caplog.text
    assert "health_below_50" in caplog.text


def test_zone2_buy_high_risk_warns(caplog) -> None:
    content = _pm("BUY", "MODERATE", health=80, risk=1.6)
    with caplog.at_level(logging.WARNING):
        _log_pm_discipline_checks(content, [], "USABLE", "T")
    assert "risk_above_1_5" in caplog.text


def test_zone2_buy_blocking_flag_warns_even_with_clean_tally(caplog) -> None:
    # The dropped-penalty + override combo: tally looks clean (1.0) but a BUY-blocking
    # flag is still present in red_flags.
    content = _pm("BUY", "MODERATE", health=80, risk=1.0)
    flags = [{"type": "TRANSIENT_STRENGTH_DISTORTION", "detail": "x"}]
    with caplog.at_level(logging.WARNING):
        _log_pm_discipline_checks(content, flags, "USABLE", "T")
    assert "pm_override_threshold_unmet" in caplog.text
    assert "blocking_growth_quality_flag" in caplog.text


def test_zone2_buy_growth_below_65_alone_is_silent(caplog) -> None:
    # Growth < 65 alone must NOT warn (the "Projected EPS > 15%" branch isn't parsed).
    content = _pm("BUY", "MODERATE", health=80, growth=60, risk=1.0)
    with caplog.at_level(logging.WARNING):
        _log_pm_discipline_checks(content, [], "USABLE", "T")
    assert "pm_override_threshold_unmet" not in caplog.text


def test_zone1_hold_low_health_warns(caplog) -> None:
    content = _pm("HOLD", "HIGH", health=70, growth=85)
    with caplog.at_level(logging.WARNING):
        _log_pm_discipline_checks(content, [], "USABLE", "T")
    assert "pm_hold_override_threshold_unmet" in caplog.text
    assert "health_below_80" in caplog.text


def test_zone1_hold_low_growth_warns(caplog) -> None:
    content = _pm("HOLD", "HIGH", health=85, growth=70)
    with caplog.at_level(logging.WARNING):
        _log_pm_discipline_checks(content, [], "USABLE", "T")
    assert "growth_below_80" in caplog.text


def test_buy_on_quarantined_valuation_inputs_warns(caplog) -> None:
    # Zone LOW so the override branch can't fire — isolates the quarantine log.
    content = _pm("BUY", "LOW", health=90, risk=0.5)
    with caplog.at_level(logging.WARNING):
        _log_pm_discipline_checks(content, [], "QUARANTINED", "T")
    assert "pm_buy_on_quarantined_valuation_inputs" in caplog.text


def test_hold_on_quarantined_does_not_log_buy_warning(caplog) -> None:
    content = _pm("HOLD", "LOW")
    with caplog.at_level(logging.WARNING):
        _log_pm_discipline_checks(content, [], "QUARANTINED", "T")
    assert "pm_buy_on_quarantined_valuation_inputs" not in caplog.text


def test_missing_verdict_and_zone_is_silent(caplog) -> None:
    content = "(no parseable verdict, zone, or scores here)"
    with caplog.at_level(logging.WARNING):
        _log_pm_discipline_checks(content, [], "QUARANTINED", "T")
    assert "pm_override_threshold_unmet" not in caplog.text
    assert "pm_hold_override_threshold_unmet" not in caplog.text
    assert "pm_buy_on_quarantined_valuation_inputs" not in caplog.text


def test_zero_penalty_quarantine_flag_renders_visibly_and_floor_neutral() -> None:
    # The PM-visible warning: rendered with a +0.00 tag, adds nothing to the subtotal.
    flags = [
        {"type": "PFIC_UNCERTAIN", "detail": "x", "risk_penalty": 1.0},
        {
            "type": "VALUATION_INPUT_QUARANTINED",
            "detail": "distrusted multiples",
            "risk_penalty": 0.0,
        },
    ]
    text, subtotal = format_red_flag_section("PASS", flags)
    assert subtotal == 1.0  # the 0.0 flag does not move the floor
    assert "VALUATION_INPUT_QUARANTINED [risk_penalty +0.00]" in text


def test_return_quality_fragility_lands_in_code_subtotal() -> None:
    """RQF was relocated from PM free-form to the deterministic subtotal (Step 1)."""
    flags = [
        {
            "type": "RETURN_QUALITY_FRAGILITY",
            "detail": "PROFITABILITY_TREND: UNSTABLE",
            "risk_penalty": 0.5,
        },
    ]
    text, subtotal = format_red_flag_section("PASS", flags)
    assert subtotal == 0.5
    assert "RETURN_QUALITY_FRAGILITY [risk_penalty +0.50]" in text
    assert "do NOT re-score them" in text
