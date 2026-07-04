"""Prompt↔code threshold parity.

Thesis numbers live as prose inside prompts/*.json and as constants in
src/thesis_constants.py. Nothing else keeps them in sync — these tests fail
CI when either side drifts (the bear-researcher coverage threshold drifted
silently for three months before this guard existed).

Patterns are derived from thesis_constants where practical so changing a
canonical value breaks this test until the prompts are re-aligned too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src import thesis_constants as tc
from src.charts.extractors import data_block as _db
from src.validators import supplemental_extractors as _se

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

_PE = f"{tc.PE_MAX:.0f}"  # "18"
_PEG = f"{tc.PEG_MAX:g}"  # "1.2"
_COV = str(tc.ANALYST_COVERAGE_MAX)  # "15"
_LIQ_MIN_K = f"${tc.LIQUIDITY_MIN_USD // 1000}k"  # "$100k"
_LIQ_PASS_K = f"${tc.LIQUIDITY_PASS_USD // 1000}k"  # "$250k"

CASES = [
    # (prompt file, regex, human label)
    (
        "portfolio_manager.json",
        rf"P/E > {_PE} AND PEG > {re.escape(_PEG)}",
        "PE/PEG hard fail",
    ),
    (
        "portfolio_manager.json",
        rf"<{re.escape(_LIQ_MIN_K)} daily: HARD FAIL",
        "liquidity hard fail",
    ),
    (
        "portfolio_manager.json",
        rf"{re.escape(_LIQ_MIN_K)}-{re.escape(_LIQ_PASS_K)} daily: MARGINAL",
        "liquidity marginal band",
    ),
    (
        "portfolio_manager.json",
        rf">{re.escape(_LIQ_PASS_K)} daily: PASS",
        "liquidity pass",
    ),
    (
        "portfolio_manager.json",
        rf"Adjusted Health < {tc.HEALTH_MIN_PCT:.0f}%",
        "health floor",
    ),
    (
        "research_manager.json",
        rf"Adjusted Score ≥ {tc.HEALTH_MIN_PCT:.0f}%",
        "health floor (RM)",
    ),
    ("portfolio_manager.json", rf"HIGH >= {tc.RISK_ZONE_HIGH:.1f}", "risk zone high"),
    (
        "portfolio_manager.json",
        rf"MODERATE {tc.RISK_ZONE_MODERATE:.1f}-1\.99",
        "risk zone moderate",
    ),
    ("market_analyst.json", rf"<{re.escape(_LIQ_MIN_K)} daily", "liquidity hard fail"),
    ("market_analyst.json", rf">{re.escape(_LIQ_PASS_K)} daily", "liquidity pass"),
    ("trader.json", rf"<{re.escape(_LIQ_PASS_K)} daily", "low-liquidity sizing cap"),
    (
        "bear_researcher.json",
        rf"≥{_COV} US/English analysts",
        "analyst-coverage hard fail",
    ),
    (
        "bull_researcher.json",
        rf"<{_COV} US/English analysts",
        "analyst-coverage requirement",
    ),
    ("fundamentals_analyst.json", rf"P/E <={_PE}", "PE scoring threshold"),
    (
        # The PE_OR_PEG health point is trailing-only; quick-mode runs scored it
        # off the forward P/E until v9.32 pinned the basis (145020.KQ).
        "fundamentals_analyst.json",
        rf"Trailing P/E <={_PE} OR PEG <={re.escape(_PEG)}: 1 pt "
        r"\(P/E basis is PE_RATIO_TTM — never PE_RATIO_FORWARD\)",
        "PE_OR_PEG trailing (TTM) basis",
    ),
    (
        "fundamentals_analyst.json",
        rf"FINANCIAL HEALTH SCORE \({tc.HEALTH_RUBRIC_POINTS:.0f} Points Total\)",
        "health rubric total",
    ),
    (
        "fundamentals_analyst.json",
        rf"GROWTH TRANSITION SCORE \({tc.GROWTH_RUBRIC_POINTS:.0f} Points Total\)",
        "growth rubric total",
    ),
    (
        "fundamentals_analyst.json",
        rf"RAW_HEALTH_SCORE: \[X\]/{tc.HEALTH_RUBRIC_POINTS:.0f}",
        "raw health score template",
    ),
    (
        "fundamentals_analyst.json",
        rf"RAW_GROWTH_SCORE: \[X\]/{tc.GROWTH_RUBRIC_POINTS:.0f}",
        "raw growth score template",
    ),
    (
        "portfolio_manager.json",
        r"HEALTH_SCORE_UNRELIABLE \(or\s*GROWTH_SCORE_UNRELIABLE\)",
        "unreliable-score gate guard",
    ),
    ("fundamentals_analyst.json", rf"PEG <={re.escape(_PEG)}", "PEG scoring threshold"),
    (
        "fundamentals_analyst.json",
        rf"ASSET_TURNOVER >= {tc.ASSET_TURNOVER_DISTRIBUTION_MIN:g}",
        "distribution-model asset-turnover gate",
    ),
    (
        "fundamentals_analyst.json",
        rf"Operating Margin >{tc.SECTOR_OPERATING_MARGIN_MIN['Consumer Discretionary']:.0f}%",
        "distribution relaxed operating-margin floor",
    ),
    (
        "fundamentals_analyst.json",
        rf"Gross Margin >{tc.SECTOR_GROSS_MARGIN_MIN['Consumer Discretionary']:.0f}%",
        "distribution relaxed gross-margin floor",
    ),
]

RETIRED_PATTERNS = [
    (r"≥10 US/English", "pre-2026-06 bear coverage threshold"),
    (r"\$500k|\$500,000", "pre-fix liquidity threshold"),
]


def _system_message(fname: str) -> str:
    return json.loads((PROMPTS_DIR / fname).read_text(encoding="utf-8"))[
        "system_message"
    ]


@pytest.mark.parametrize(("fname", "pattern", "label"), CASES)
def test_prompt_matches_canonical_threshold(fname: str, pattern: str, label: str):
    msg = _system_message(fname)
    assert re.search(pattern, msg), (
        f"{fname}: {label} drifted — expected /{pattern}/ derived from "
        "src/thesis_constants.py; update the prompt or the constant together"
    )


@pytest.mark.parametrize(("sector", "median"), sorted(tc.SECTOR_MEDIAN_PE.items()))
def test_valuation_prompt_sector_pe_table_matches_constants(sector: str, median: float):
    msg = _system_message("valuation_calculator.json")
    pattern = rf"{re.escape(sector)}\s*:?\s+{median:g}"
    assert re.search(pattern, msg), (
        f"valuation_calculator.json: sector median P/E for {sector} drifted — "
        f"expected /{pattern}/ derived from src/thesis_constants.py"
    )


@pytest.mark.parametrize("fname", sorted(p.name for p in PROMPTS_DIR.glob("*.json")))
def test_no_retired_threshold_values(fname: str):
    msg = _system_message(fname)
    for pattern, label in RETIRED_PATTERNS:
        assert not re.search(pattern, msg), f"{fname}: contains retired value ({label})"


# --- L0: enum set-equality (prompt vocabulary ≡ every consuming parser) ---------
# An enum a prompt advertises must equal the token set EVERY consuming parser
# accepts. Tokens live as named tuples on the parsers (single source); this fails
# if any of {prompt, parser-1, parser-2} drifts out of agreement. (Shared numeric
# thresholds — including per-prompt health-floor coverage — are guarded by the
# CASES table above, the canonical prompt↔constant mechanism; not duplicated here.)


def _quoted_enum_from_prompt(fname: str, field: str) -> set[str]:
    """Parse a ``"field": "A|B|C"`` enum advertised in a JSON example block."""
    m = re.search(rf'"{field}"\s*:\s*"([^"]+)"', _system_message(fname))
    assert m, f"{fname}: enum field {field!r} not advertised in JSON example"
    return {token.strip() for token in m.group(1).split("|")}


def test_cmic_enum_prompt_matches_every_consumer():
    advertised = _quoted_enum_from_prompt("legal_counsel.json", "cmic_status")
    assert advertised == set(_se.CMIC_STATUS_TOKENS) == set(_db.CMIC_STATUS_TOKENS), (
        "CMIC_STATUS tokens drifted between legal_counsel.json, "
        "supplemental_extractors.CMIC_STATUS_TOKENS, and "
        "charts/extractors/data_block.CMIC_STATUS_TOKENS"
    )


def test_new_datablock_fields_parser_compatible():
    """The APR-mitigation fields must use identical names across prompt + parser.

    Guards the FLA -> Senior DATA_BLOCK -> extract_metrics chain against silent
    field-name drift (the data would then never reach the parser).
    """
    from src.validators.metric_extractor import extract_metrics

    fund = _system_message("fundamentals_analyst.json")
    fla = _system_message("foreign_language_analyst.json")
    parser_keys = extract_metrics("")  # empty report -> dict initialized with all keys

    for field, parser_key in (
        ("ASSET_TURNOVER", "asset_turnover"),
        ("INVENTORY_TURNOVER_TREND", "inventory_turnover_trend"),
        ("CAPACITY_UTILIZATION", "capacity_utilization"),
        ("FACILITY_BUILDOUT_STATUS", "facility_buildout_status"),
    ):
        assert field in fund, f"{field} missing from Senior DATA_BLOCK template"
        assert parser_key in parser_keys, f"{parser_key} not parsed by extract_metrics"

    # The capacity/facility signals originate in the Foreign Language Analyst.
    for field in ("CAPACITY_UTILIZATION", "FACILITY_BUILDOUT_STATUS"):
        assert field in fla, f"{field} missing from Foreign Language Analyst prompt"


def _breakdown_template_keys(msg: str, field: str) -> list[str]:
    m = re.search(rf"^{field}:\s*([^\n]+)$", msg, re.MULTILINE)
    assert m, f"{field} template line missing from fundamentals_analyst prompt"
    return [item.split("=")[0].strip() for item in m.group(1).split(";")]


def test_score_breakdown_keys_match_criteria_maps():
    """Breakdown template keys ≡ thesis_constants criterion maps (order too)."""
    msg = _system_message("fundamentals_analyst.json")
    assert _breakdown_template_keys(msg, "HEALTH_SCORE_BREAKDOWN") == list(
        tc.HEALTH_SCORE_CRITERIA
    ), "HEALTH_SCORE_BREAKDOWN keys drifted from HEALTH_SCORE_CRITERIA"
    assert _breakdown_template_keys(msg, "GROWTH_SCORE_BREAKDOWN") == list(
        tc.GROWTH_SCORE_CRITERIA
    ), "GROWTH_SCORE_BREAKDOWN keys drifted from GROWTH_SCORE_CRITERIA"


def test_criteria_maps_sum_to_rubric_totals():
    assert sum(tc.HEALTH_SCORE_CRITERIA.values()) == tc.HEALTH_RUBRIC_POINTS
    assert sum(tc.GROWTH_SCORE_CRITERIA.values()) == tc.GROWTH_RUBRIC_POINTS


def _rubric_bullet_points(msg: str, start: str, end: str) -> list[float]:
    section = msg[msg.index(start) : msg.index(end)]
    return [
        float(m.group(1))
        for m in re.finditer(r"(?m)^- [^\n]*?:\s*(\d+(?:\.\d+)?)\s*pt", section)
    ]


def test_health_rubric_prose_sums_to_declared_total():
    """The v9.30 rubric summed 12.5 under a '12 Points Total' header (OCF
    double-counted between Liquidity and Cash Generation) — a correct
    per-criterion breakdown could never reconcile. Guard the repair."""
    msg = _system_message("fundamentals_analyst.json")
    points = _rubric_bullet_points(
        msg, "### FINANCIAL HEALTH SCORE", "### GROWTH TRANSITION SCORE"
    )
    assert sum(points) == tc.HEALTH_RUBRIC_POINTS, (
        f"health rubric bullets sum to {sum(points)}, "
        f"declared total is {tc.HEALTH_RUBRIC_POINTS:g}"
    )
    assert len(points) == len(tc.HEALTH_SCORE_CRITERIA)


def test_growth_rubric_prose_sums_to_declared_total():
    msg = _system_message("fundamentals_analyst.json")
    points = _rubric_bullet_points(
        msg, "### GROWTH TRANSITION SCORE", "## ADAPTIVE SCORING PROTOCOL"
    )
    assert sum(points) == tc.GROWTH_RUBRIC_POINTS
    assert len(points) == len(tc.GROWTH_SCORE_CRITERIA)


def test_material_events_enum_matches_parser():
    """news_analyst SUMMARY token line ≡ supplemental_extractors token tuple."""
    msg = _system_message("news_analyst.json")
    m = re.search(r"^MATERIAL_EVENTS_90D:\s*([^\n]+)$", msg, re.MULTILINE)
    assert m, "MATERIAL_EVENTS_90D token line missing from news_analyst SUMMARY"
    advertised = {token.strip() for token in m.group(1).split("|")}
    assert advertised == set(_se.MATERIAL_EVENTS_TOKENS), (
        "MATERIAL_EVENTS_90D tokens drifted between news_analyst.json and "
        "supplemental_extractors.MATERIAL_EVENTS_TOKENS"
    )


def test_pfic_and_vie_enums_match_parser():
    assert _quoted_enum_from_prompt("legal_counsel.json", "pfic_status") == set(
        _se.PFIC_STATUS_TOKENS
    ), "PFIC_STATUS tokens drifted between prompt and supplemental_extractors"
    assert _quoted_enum_from_prompt("legal_counsel.json", "vie_structure") == set(
        _se.VIE_STRUCTURE_TOKENS
    ), "VIE_STRUCTURE tokens drifted between prompt and supplemental_extractors"
