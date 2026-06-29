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


def test_pfic_and_vie_enums_match_parser():
    assert _quoted_enum_from_prompt("legal_counsel.json", "pfic_status") == set(
        _se.PFIC_STATUS_TOKENS
    ), "PFIC_STATUS tokens drifted between prompt and supplemental_extractors"
    assert _quoted_enum_from_prompt("legal_counsel.json", "vie_structure") == set(
        _se.VIE_STRUCTURE_TOKENS
    ), "VIE_STRUCTURE tokens drifted between prompt and supplemental_extractors"
