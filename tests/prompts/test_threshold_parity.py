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


@pytest.mark.parametrize("fname", sorted(p.name for p in PROMPTS_DIR.glob("*.json")))
def test_no_retired_threshold_values(fname: str):
    msg = _system_message(fname)
    for pattern, label in RETIRED_PATTERNS:
        assert not re.search(pattern, msg), f"{fname}: contains retired value ({label})"
