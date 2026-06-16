"""Korean Value-Up plan execution → bounded governance-discount credit.

Covers the gather→promote→credit path:
  * parsing of the two DATA_BLOCK markers (Senior-promoted from the Foreign
    Language Analyst CAPITAL POLICY section), and
  * the deterministic ``KOREA_VALUE_UP_EXECUTED`` bonus, which fires only when a
    STRONG plan is PROVEN-executed AND the value-trap signal is not a hard fail.

All fixtures are synthetic — no real tickers or figures (per plan: the live system
verifies execution via DART at run time).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.validators.red_flag_detector import RedFlagDetector
from src.validators.supplemental_extractors import extract_capital_efficiency_signals
from src.validators.supplemental_flags import (
    detect_shareholder_return_execution_flags,
)

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _data_block(**fields: str) -> str:
    body = "\n".join(f"{k}: {v}" for k, v in fields.items())
    return f"### --- START DATA_BLOCK ---\n{body}\n### --- END DATA_BLOCK ---"


# Value-trap signal that is NOT a hard fail (WATCHABLE, score >= 40).
_VT_HEALTHY = "SCORE: 65\nVERDICT: WATCHABLE\nTRAP_RISK: LOW\n"
_VT_TRAP_VERDICT = "SCORE: 55\nVERDICT: TRAP\nTRAP_RISK: HIGH\n"
_VT_LOW_SCORE = "SCORE: 35\nVERDICT: CAUTIOUS\nTRAP_RISK: HIGH\n"


class TestValueUpMarkerParsing:
    def test_parses_strong_and_proven(self):
        report = _data_block(
            VALUE_UP_PLAN_STRENGTH="STRONG",
            SHAREHOLDER_RETURN_EXECUTION="PROVEN",
        )
        signals = extract_capital_efficiency_signals(report)
        assert signals["value_up_plan_strength"] == "STRONG"
        assert signals["shareholder_return_execution"] == "PROVEN"

    def test_parses_all_enum_values(self):
        report = _data_block(
            VALUE_UP_PLAN_STRENGTH="WEAK",
            SHAREHOLDER_RETURN_EXECUTION="ANNOUNCED_ONLY",
        )
        signals = extract_capital_efficiency_signals(report)
        assert signals["value_up_plan_strength"] == "WEAK"
        assert signals["shareholder_return_execution"] == "ANNOUNCED_ONLY"

    def test_na_dropped(self):
        report = _data_block(
            VALUE_UP_PLAN_STRENGTH="N/A",
            SHAREHOLDER_RETURN_EXECUTION="N/A",
        )
        signals = extract_capital_efficiency_signals(report)
        assert "value_up_plan_strength" not in signals
        assert "shareholder_return_execution" not in signals

    def test_absent_markers(self):
        report = _data_block(ROIC_PERCENT="12.0")
        signals = extract_capital_efficiency_signals(report)
        assert "value_up_plan_strength" not in signals
        assert "shareholder_return_execution" not in signals


class TestShareholderReturnExecutionFlag:
    def test_proven_strong_fires_bonus(self):
        report = _data_block(
            SECTOR="Consumer Discretionary",
            VALUE_UP_PLAN_STRENGTH="STRONG",
            SHAREHOLDER_RETURN_EXECUTION="PROVEN",
        )
        flags = detect_shareholder_return_execution_flags(
            report, _VT_HEALTHY, "TEST.KS"
        )
        assert len(flags) == 1
        flag = flags[0]
        assert flag["type"] == "KOREA_VALUE_UP_EXECUTED"
        assert flag["risk_penalty"] == -0.5
        assert flag["action"] == "RISK_BONUS"

    def test_announced_only_no_bonus(self):
        report = _data_block(
            VALUE_UP_PLAN_STRENGTH="STRONG",
            SHAREHOLDER_RETURN_EXECUTION="ANNOUNCED_ONLY",
        )
        assert detect_shareholder_return_execution_flags(report, _VT_HEALTHY) == []

    def test_partial_execution_no_bonus(self):
        report = _data_block(
            VALUE_UP_PLAN_STRENGTH="STRONG",
            SHAREHOLDER_RETURN_EXECUTION="PARTIAL",
        )
        assert detect_shareholder_return_execution_flags(report, _VT_HEALTHY) == []

    def test_weak_plan_no_bonus(self):
        report = _data_block(
            VALUE_UP_PLAN_STRENGTH="WEAK",
            SHAREHOLDER_RETURN_EXECUTION="PROVEN",
        )
        assert detect_shareholder_return_execution_flags(report, _VT_HEALTHY) == []

    def test_trap_verdict_blocks_bonus(self):
        report = _data_block(
            VALUE_UP_PLAN_STRENGTH="STRONG",
            SHAREHOLDER_RETURN_EXECUTION="PROVEN",
        )
        assert detect_shareholder_return_execution_flags(report, _VT_TRAP_VERDICT) == []

    def test_low_value_trap_score_blocks_bonus(self):
        report = _data_block(
            VALUE_UP_PLAN_STRENGTH="STRONG",
            SHAREHOLDER_RETURN_EXECUTION="PROVEN",
        )
        assert detect_shareholder_return_execution_flags(report, _VT_LOW_SCORE) == []

    def test_financials_sector_still_eligible(self):
        # Credit-bureau / holding names are often Financials; the dedicated detector
        # must NOT be suppressed by the Financials early-return in capital-efficiency.
        report = _data_block(
            SECTOR="Financials",
            VALUE_UP_PLAN_STRENGTH="STRONG",
            SHAREHOLDER_RETURN_EXECUTION="PROVEN",
        )
        flags = detect_shareholder_return_execution_flags(report, _VT_HEALTHY)
        assert len(flags) == 1
        assert flags[0]["type"] == "KOREA_VALUE_UP_EXECUTED"

    def test_no_value_trap_report_still_fires(self):
        # Absence of a value-trap report is not a hard fail; the PM/APAC gate guards.
        report = _data_block(
            VALUE_UP_PLAN_STRENGTH="STRONG",
            SHAREHOLDER_RETURN_EXECUTION="PROVEN",
        )
        flags = detect_shareholder_return_execution_flags(report, None)
        assert len(flags) == 1

    def test_empty_report_no_bonus(self):
        assert detect_shareholder_return_execution_flags("", _VT_HEALTHY) == []

    def test_exposed_on_facade(self):
        assert hasattr(RedFlagDetector, "detect_shareholder_return_execution_flags")
        report = _data_block(
            VALUE_UP_PLAN_STRENGTH="STRONG",
            SHAREHOLDER_RETURN_EXECUTION="PROVEN",
        )
        flags = RedFlagDetector.detect_shareholder_return_execution_flags(
            report, value_trap_report=_VT_HEALTHY, ticker="TEST.KS"
        )
        assert len(flags) == 1


class TestPromptContent:
    """Guards that the four prompts carry the Value-Up execution contract."""

    @staticmethod
    def _load(name: str) -> dict:
        return json.loads((PROMPTS_DIR / name).read_text(encoding="utf-8"))

    def test_versions_are_semver(self):
        for name in (
            "foreign_language_analyst.json",
            "fundamentals_analyst.json",
            "apac_regional_specialist.json",
            "portfolio_manager.json",
        ):
            version = self._load(name)["version"]
            assert re.match(r"^\d+\.\d+$", version), (name, version)

    def test_fla_gathers_execution(self):
        sm = self._load("foreign_language_analyst.json")["system_message"]
        assert "SHAREHOLDER_RETURN_EXECUTION" in sm
        assert "VALUE_UP_PLAN_STRENGTH" in sm
        assert "Execution Track Record" in sm

    def test_senior_promotes_markers(self):
        sm = self._load("fundamentals_analyst.json")["system_message"]
        assert "VALUE_UP_PLAN_STRENGTH" in sm
        assert "SHAREHOLDER_RETURN_EXECUTION" in sm

    def test_apac_adjudicates(self):
        sm = self._load("apac_regional_specialist.json")["system_message"]
        assert "VALUE_UP_EXECUTION_CREDIT" in sm

    def test_pm_applies_bounded_credit(self):
        sm = self._load("portfolio_manager.json")["system_message"]
        assert "KOREA_VALUE_UP_EXECUTED" in sm
