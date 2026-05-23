"""Integration coverage for the Tranche 5 data-flow fixes.

Each test verifies one of the wire-up bugs the critique surfaced:

- Memo can render a verdict from a *saved* JSON shape (not just runtime state).
- Memo's Key Numbers section finds ``DE_RATIO`` / ``NET_DEBT_EBITDA`` — the
  field names actually in production DATA_BLOCKs.
- Scenarios fire on real production data (no direct ``EPS_TTM`` field; derived
  from ``CURRENT_PRICE / PE_RATIO_TTM``).
- ``FootballFieldData.scenarios`` carries the Protocol type through both
  call sites.
"""

from __future__ import annotations

from src.charts.base import FootballFieldData, ValuationScenariosLike
from src.charts.extractors.valuation import (
    ValuationScenarios,
    extract_valuation_scenarios,
    resolve_eps_ttm,
)
from src.reporting.memo import build_memo, extract_key_metrics

_REAL_PM_NARRATIVE = (
    "#### PORTFOLIO MANAGER VERDICT: BUY\n\n"
    "### DECISION RATIONALE\n\n"
    "Quality compounder at a 12x P/E with 22% ROIC.\n\n"
    "### --- START PM_BLOCK ---\nVERDICT: BUY\n### --- END PM_BLOCK ---\n"
)

# Production-shape DATA_BLOCK: uses DE_RATIO and NET_DEBT_EBITDA, no EPS_TTM.
_REAL_FUNDAMENTALS = (
    "### --- START DATA_BLOCK ---\n"
    "SECTOR: Industrials\n"
    "PE_RATIO_TTM: 12.0\n"
    "PEG_RATIO: 0.9\n"
    "ROIC_PERCENT: 22.0\n"
    "FCF_YIELD_PERCENT: 6.5\n"
    "REVENUE_GROWTH_TTM: 8.0\n"
    "NET_DEBT_EBITDA: 1.4\n"
    "DE_RATIO: 0.32\n"
    "ANALYST_COVERAGE_ENGLISH: 6\n"
    "CURRENT_PRICE: 100.00\n"
    "### --- END DATA_BLOCK ---\n"
)

_VAL_SCENARIOS_BLOCK = (
    "### --- START VALUATION_PARAMS ---\n"
    "METHOD: P/E_NORMALIZATION\nSECTOR: Industrials\nSECTOR_MEDIAN_PE: 17\n"
    "CURRENT_PE: 12.0\nPEG_RATIO: 0.9\nGROWTH_SCORE_PCT: 55\n"
    "CURRENT_PRICE: 100.00\nCONFIDENCE: HIGH\n"
    "### --- END VALUATION_PARAMS ---\n\n"
    "### --- START VALUATION_SCENARIOS ---\n"
    "METHODOLOGY: P/E\nDATA_SUFFICIENCY: HIGH\n"
    "BEAR_MULTIPLE: 8\nBEAR_GROWTH_PCT: -5\nBEAR_MARGIN_DELTA_BPS: -200\n"
    "BEAR_DRIVERS: Cyclical trough.\nBEAR_PROBABILITY: 30\n"
    "BASE_MULTIPLE: 12\nBASE_GROWTH_PCT: 8\nBASE_MARGIN_DELTA_BPS: 0\n"
    "BASE_DRIVERS: Mid-cycle base.\nBASE_PROBABILITY: 50\n"
    "BULL_MULTIPLE: 16\nBULL_GROWTH_PCT: 15\nBULL_MARGIN_DELTA_BPS: 100\n"
    "BULL_DRIVERS: Cycle peak.\nBULL_PROBABILITY: 20\n"
    "### --- END VALUATION_SCENARIOS ---\n"
)


# ---------- Memo reads saved-JSON shape (Step 1) ----------


def test_memo_renders_decision_from_saved_json_shape() -> None:
    """Bug fixed in Step 1: persistence writes PM into final_decision.decision.

    Pre-fix the memo returned UNAVAILABLE for every retrospective rendering.
    """
    saved = {
        "final_decision": {"decision": _REAL_PM_NARRATIVE},
        "reports": {"fundamentals_report": _REAL_FUNDAMENTALS},
    }
    memo = build_memo(saved)
    assert memo.decision == "BUY"
    assert "compounder" in memo.one_line_thesis or "Quality" in memo.one_line_thesis


def test_memo_renders_variant_from_saved_json_shape() -> None:
    """Variant view must come out of investment_analysis.investment_plan in saved JSON."""
    saved = {
        "final_decision": {"decision": _REAL_PM_NARRATIVE},
        "reports": {"fundamentals_report": _REAL_FUNDAMENTALS},
        "investment_analysis": {
            "investment_plan": (
                "CONSENSUS_VIEW: The market believes growth has peaked.\n"
                "VARIANT_VIEW: We see a 24-month margin recovery.\n"
                "BASIS: Order book +30% YoY surfaced by Foreign Language Analyst.\n"
            )
        },
    }
    memo = build_memo(saved)
    assert "margin recovery" in memo.variant_view


# ---------- Memo metrics use real DATA_BLOCK field names (Step 2) ----------


def test_extract_key_metrics_picks_up_de_ratio_and_net_debt_ebitda() -> None:
    """Pre-Step-2 these rows were silently missing on production DATA_BLOCKs."""
    rows = extract_key_metrics(_REAL_FUNDAMENTALS, limit=8)
    by_label = {row.split(":", 1)[0]: row for row in rows}
    assert "D/E" in by_label, "DE_RATIO not picked up"
    assert "0.32" in by_label["D/E"]
    assert "Net debt / EBITDA" in by_label, "NET_DEBT_EBITDA not picked up"
    assert "1.4" in by_label["Net debt / EBITDA"]


def test_extract_key_metrics_legacy_field_names_still_work() -> None:
    """Synthetic / older DATA_BLOCKs using DEBT_TO_EQUITY must still populate."""
    legacy_block = (
        "### --- START DATA_BLOCK ---\n"
        "PE_RATIO_TTM: 14.0\n"
        "DEBT_TO_EQUITY: 0.42\n"  # legacy alias
        "NET_DEBT_TO_EBITDA: 2.1\n"  # legacy alias
        "### --- END DATA_BLOCK ---\n"
    )
    rows = extract_key_metrics(legacy_block, limit=8)
    assert any(r.startswith("D/E:") and "0.42" in r for r in rows)
    assert any(r.startswith("Net debt / EBITDA:") and "2.1" in r for r in rows)


# ---------- Scenarios on real production data (Step 3) ----------


def test_scenarios_fire_on_production_shape_without_eps_field() -> None:
    """No EPS_TTM in DATA_BLOCK — resolver derives it from price/PE.

    Pre-Step-3 every production report fell back to legacy single-range
    valuation despite having all the information needed to compute IVs.
    """
    eps = resolve_eps_ttm(_REAL_FUNDAMENTALS)
    assert eps is not None and eps > 0
    scenarios = extract_valuation_scenarios(_VAL_SCENARIOS_BLOCK, eps)
    assert scenarios is not None
    assert scenarios.bear_iv > 0 and scenarios.bull_iv > scenarios.bear_iv
    assert (
        min(scenarios.bear_iv, scenarios.base_iv, scenarios.bull_iv)
        <= scenarios.weighted_iv
    )
    assert scenarios.weighted_iv <= max(
        scenarios.bear_iv, scenarios.base_iv, scenarios.bull_iv
    )


def test_memo_valuation_slot_uses_scenarios_on_production_shape() -> None:
    """End-to-end: memo's Valuation slot carries bear/base/bull on real shape."""
    state = {
        "final_trade_decision": _REAL_PM_NARRATIVE,
        "fundamentals_report": _REAL_FUNDAMENTALS,
        "valuation_params": _VAL_SCENARIOS_BLOCK,
    }
    memo = build_memo(state)
    assert (
        "Bear" in memo.valuation
        and "Base" in memo.valuation
        and "Bull" in memo.valuation
    )
    assert "weighted" in memo.valuation


# ---------- Protocol boundary (Step 5) ----------


def test_valuation_scenarios_satisfies_protocol() -> None:
    """The production ValuationScenarios dataclass satisfies the chart Protocol."""
    eps = resolve_eps_ttm(_REAL_FUNDAMENTALS)
    scenarios = extract_valuation_scenarios(_VAL_SCENARIOS_BLOCK, eps)
    assert isinstance(scenarios, ValuationScenarios)
    # Structural typing: the Protocol uses runtime_checkable so isinstance works.
    assert isinstance(scenarios, ValuationScenariosLike)


def test_football_field_data_accepts_typed_scenarios() -> None:
    """FootballFieldData carries scenarios with the new Protocol type, not bare ``object``."""
    eps = resolve_eps_ttm(_REAL_FUNDAMENTALS)
    scenarios = extract_valuation_scenarios(_VAL_SCENARIOS_BLOCK, eps)
    data = FootballFieldData(
        ticker="TEST",
        trade_date="2026-05-20",
        current_price=100.0,
        fifty_two_week_high=150.0,
        fifty_two_week_low=70.0,
        scenarios=scenarios,
    )
    assert data.scenarios is scenarios


def test_football_field_data_scenarios_optional_default_none() -> None:
    """Default keeps legacy chart path unchanged."""
    data = FootballFieldData(
        ticker="TEST",
        trade_date="2026-05-20",
        current_price=100.0,
        fifty_two_week_high=150.0,
        fifty_two_week_low=70.0,
    )
    assert data.scenarios is None
