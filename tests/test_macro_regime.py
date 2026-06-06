from __future__ import annotations

from src.macro_regime import MacroRegime, parse_macro_regime


def _report(**overrides: str) -> str:
    body = {
        "RISK_APPETITE": "RISK_OFF",
        "SHOCK_TYPE": "ENERGY",
        "SHOCK_PHASE": "ACUTE",
        "EQUITY_TRANSMISSION": "EARNINGS_PRESSURE",
        "DIP_POSTURE": "WAIT_FOR_CONFIRMATION",
        "CONFIDENCE": "MEDIUM",
        **overrides,
    }
    block = "\n".join(f"{key}: {value}" for key, value in body.items())
    return f"### REGIME SUMMARY\n- Stress rising.\n\nMACRO_REGIME_BLOCK:\n{block}"


def test_parse_full_block() -> None:
    regime = parse_macro_regime(_report())

    assert regime.present is True
    assert regime.risk_appetite == "RISK_OFF"
    assert regime.shock_type == "ENERGY"
    assert regime.dip_posture == "WAIT_FOR_CONFIRMATION"
    assert regime.confidence == "MEDIUM"
    assert regime.raw_block.startswith("MACRO_REGIME_BLOCK:")


def test_parse_missing_block_returns_absent_defaults() -> None:
    regime = parse_macro_regime("### REGIME SUMMARY\n- No structured block.")

    assert regime == MacroRegime()
    assert regime.present is False


def test_parse_unknown_value_normalizes_that_field_only() -> None:
    regime = parse_macro_regime(_report(RISK_APPETITE="PANIC"))

    assert regime.present is True
    assert regime.risk_appetite == "UNCERTAIN"
    assert regime.shock_type == "ENERGY"


def test_parse_partial_block_keeps_conservative_dip_posture_default() -> None:
    regime = parse_macro_regime(
        "MACRO_REGIME_BLOCK:\nRISK_APPETITE: RISK_OFF\nCONFIDENCE: HIGH"
    )

    assert regime.present is True
    assert regime.risk_appetite == "RISK_OFF"
    assert regime.dip_posture == "WAIT_FOR_CONFIRMATION"
    assert regime.confidence == "HIGH"


def test_parse_empty_block_is_present_with_defaults() -> None:
    regime = parse_macro_regime("MACRO_REGIME_BLOCK:\n")

    assert regime.present is True
    assert regime.raw_block == "MACRO_REGIME_BLOCK:"
    assert regime.to_dict()["dip_posture"] == "WAIT_FOR_CONFIRMATION"


def test_parse_double_block_picks_first() -> None:
    regime = parse_macro_regime(
        _report(RISK_APPETITE="RISK_OFF") + "\n\n" + _report(RISK_APPETITE="RISK_ON")
    )

    assert regime.risk_appetite == "RISK_OFF"


def test_parse_trailing_heading_stops_block() -> None:
    regime = parse_macro_regime(_report() + "\n\n### NOTES\n- extra")

    assert regime.confidence == "MEDIUM"
    assert "### NOTES" not in regime.raw_block
