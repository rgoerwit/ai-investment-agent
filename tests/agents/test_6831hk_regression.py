"""Regression tests for the 6831.HK mitigation set.

Covers the per-run rendering/labeling defects and the data-contract reconciliations
surfaced by the May→June 6831.HK comparison:

A  PM_BLOCK fenced strip leaves no resolution remnant
B1 calculate_return_trends emits roa/roe 5Y averages atomically with their year counts
B2 reconcile_high_risk_fields promotes computed 5Y averages (and never erases on absence)
C  payout reconciled to N/A when provider shows a dividend but payoutRatio is 0/absent
D  PFIC evidence truncates on a word boundary, never mid-word
E  technical ENTRY/EXIT subsection is neutralized for non-actionable verdicts
+  PFIC cash-trap threshold lock; MRQ-growth policy lock
"""

from __future__ import annotations

import json

import pandas as pd

from src.agents.fundamentals_reconciler import reconcile_high_risk_fields
from src.data.metric_extraction import calculate_return_trends
from src.report_generator import (
    _strip_fenced_pm_machine_block,
    _suppress_executable_levels,
)
from src.validators.supplemental_flags import _truncate_at_boundary, detect_legal_flags


# --------------------------------------------------------------------------- A
class TestStripFencedPmBlock:
    def test_clean_fenced_block_removed(self) -> None:
        text = (
            "Rationale here.\n\n### PM_BLOCK\n\n```\n"
            "### --- START PM_BLOCK ---\nVERDICT: BUY\n### --- END PM_BLOCK ---\n```\n"
        )
        out = _strip_fenced_pm_machine_block(text)
        assert "PM_BLOCK" not in out
        assert "Rationale here." in out

    def test_resolutions_before_markers_are_removed_with_block(self) -> None:
        # The real 6831.HK shape: resolutions inside the fence, before START.
        text = (
            "Decision text.\n\n### PM_BLOCK\n\n```\n"
            "CONSULTANT_RESOLUTION:\n- CONCERN: x\n- VERDICT: UNVERIFIABLE\n"
            "APAC_RESOLUTION:\n- FINDING: CAUTION – use HOLD\n- VERDICT: UNVERIFIABLE\n\n"
            "### --- START PM_BLOCK ---\nVERDICT: DO_NOT_INITIATE\nRISK_TALLY: 4.75\n"
            "### --- END PM_BLOCK ---\n```\n"
        )
        out = _strip_fenced_pm_machine_block(text)
        assert "APAC_RESOLUTION" not in out
        assert "CONSULTANT_RESOLUTION" not in out
        assert "PM_BLOCK" not in out
        assert "Decision text." in out

    def test_fence_without_end_marker_is_left_untouched(self) -> None:
        text = "```\n### --- START PM_BLOCK ---\nVERDICT: BUY\n```\n"
        # No END marker inside the fence: not a complete machine block, leave it.
        assert _strip_fenced_pm_machine_block(text) == text

    def test_unrelated_fenced_block_preserved(self) -> None:
        text = "```\n=== DECISION LOGIC ===\nZONE: HIGH\n```\n"
        assert _strip_fenced_pm_machine_block(text) == text


# --------------------------------------------------------------------------- E
class TestSuppressExecutableLevels:
    def test_entry_exit_subsection_neutralized(self) -> None:
        md = (
            "#### TREND\nBearish\n\n"
            "#### ENTRY/EXIT RECOMMENDATIONS\n"
            "Stop Loss: 5.75 HKD\nTargets: 6.80 HKD\n\n"
            "#### SUMMARY\nLiquidity PASS\n"
        )
        out = _suppress_executable_levels(md)
        assert "5.75" not in out
        assert "Not actionable" in out
        assert "#### SUMMARY" in out  # later sections preserved

    def test_no_entry_exit_subsection_is_noop(self) -> None:
        md = "#### TREND\nBearish\n\n#### SUMMARY\nLiquidity PASS\n"
        assert _suppress_executable_levels(md) == md


# -------------------------------------------------------------------------- B1
class TestReturnTrendAtomicInvariant:
    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cols = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"])
        fin = pd.DataFrame(
            {c: [v] for c, v in zip(cols, [486e6, 400e6, 320e6, 250e6], strict=True)},
            index=["Net Income"],
        )
        bs = pd.DataFrame(
            {c: [v] for c, v in zip(cols, [3.9e9, 3.6e9, 3.2e9, 2.9e9], strict=True)},
            index=["Total Assets"],
        )
        return fin, bs

    def test_avg_present_whenever_years_present(self) -> None:
        fin, bs = self._frames()
        sig = calculate_return_trends(fin, bs, "6831.HK")
        # The atomic contract the production observability guard enforces.
        if sig.get("_roa_5y_years"):
            assert sig.get("roa_5y_avg") is not None

    def test_insufficient_years_emits_nothing(self) -> None:
        cols = pd.to_datetime(["2025-12-31", "2024-12-31"])
        fin = pd.DataFrame(
            {c: [v] for c, v in zip(cols, [486e6, 400e6], strict=True)},
            index=["Net Income"],
        )
        bs = pd.DataFrame(
            {c: [v] for c, v in zip(cols, [3.9e9, 3.6e9], strict=True)},
            index=["Total Assets"],
        )
        sig = calculate_return_trends(fin, bs, "6831.HK")
        assert "roa_5y_avg" not in sig
        assert "_roa_5y_years" not in sig

    def test_validator_flags_missing_average(self) -> None:
        # The pairing invariant lives in the data validator (profitability checks),
        # not as a one-off inline guard — folded so it sits with the other
        # deterministic data-sanity checks.
        from src.data.validator import validator as data_validator

        violated = data_validator._validate_profitability(
            {"_roa_5y_years": 4, "roa_5y_avg": None}, "6831.HK"
        )
        assert any("pairing invariant" in w for w in violated.warnings)

        intact = data_validator._validate_profitability(
            {"_roa_5y_years": 4, "roa_5y_avg": 10.04}, "6831.HK"
        )
        assert not any("pairing invariant" in w for w in intact.warnings)


# -------------------------------------------------------------------- B2 + C + locks
def _body(**lines: str) -> str:
    inner = "\n".join(f"{k}: {v}" for k, v in lines.items())
    return f"### --- START DATA_BLOCK ---\n{inner}\n### --- END DATA_BLOCK ---\n"


def _field(body: str, key: str) -> str | None:
    for line in body.splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return None


class TestReconcileFiveYearAverages:
    def test_promotes_when_payload_present(self) -> None:
        body = _body(ROA_5Y_AVG="N/A", ROE_5Y_AVG="N/A")
        out = reconcile_high_risk_fields(
            body, {"roa_5y_avg": 10.04, "roe_5y_avg": 37.01}
        )
        assert _field(out, "ROA_5Y_AVG") == "10.04%"
        assert _field(out, "ROE_5Y_AVG") == "37.01%"

    def test_no_erase_when_payload_absent(self) -> None:
        body = _body(ROA_5Y_AVG="9.80%")
        out = reconcile_high_risk_fields(body, {})
        assert _field(out, "ROA_5Y_AVG") == "9.80%"

    def test_within_tolerance_no_churn(self) -> None:
        body = _body(ROA_5Y_AVG="10.05%")
        out = reconcile_high_risk_fields(body, {"roa_5y_avg": 10.04})
        assert _field(out, "ROA_5Y_AVG") == "10.05%"


class TestReconcileDividend:
    def test_zero_payout_with_dividend_becomes_na(self) -> None:
        body = _body(PAYOUT_RATIO="0.0%", DIVIDEND_COVERAGE="N/A")
        out = reconcile_high_risk_fields(body, {"payoutRatio": 0, "dividendRate": 0.52})
        assert _field(out, "PAYOUT_RATIO") == "N/A"
        assert "DIVIDEND_DATA_QUALITY_NOTE:" in out

    def test_real_payout_unchanged(self) -> None:
        body = _body(PAYOUT_RATIO="57.0%")
        out = reconcile_high_risk_fields(
            body, {"payoutRatio": 0.57, "dividendRate": 0.52}
        )
        assert _field(out, "PAYOUT_RATIO") == "57.0%"
        assert "DIVIDEND_DATA_QUALITY_NOTE:" not in out

    def test_genuine_non_payer_keeps_zero(self) -> None:
        body = _body(PAYOUT_RATIO="0.0%")
        out = reconcile_high_risk_fields(body, {"payoutRatio": 0, "dividendRate": 0})
        assert _field(out, "PAYOUT_RATIO") == "0.0%"
        assert "DIVIDEND_DATA_QUALITY_NOTE:" not in out


class TestContractLocks:
    def test_pfic_cash_trap_threshold(self) -> None:
        # 31.6% cash/assets is below the 50% asset-test threshold -> NO (locks the
        # June behavior so it cannot silently flip back to the pre-deterministic YES).
        body = _body(PFIC_CASH_TRAP="YES", PFIC_ASSET_RATIO="31.6%")
        out = reconcile_high_risk_fields(body, {"capital_cashToAssets": 0.316})
        assert _field(out, "PFIC_CASH_TRAP") == "NO"

    def test_quarterly_growth_not_remapped_to_mrq(self) -> None:
        # Policy lock: yfinance's earningsQuarterlyGrowth is intentionally NOT
        # promoted into EARNINGS_GROWTH_MRQ (unreliable for ex-US). The reconciler
        # must not invent the horizon from it — a pre-existing N/A stays N/A.
        body = _body(EARNINGS_GROWTH_MRQ="N/A")
        out = reconcile_high_risk_fields(body, {"earningsQuarterlyGrowth": 0.439})
        assert _field(out, "EARNINGS_GROWTH_MRQ") == "N/A"


# --------------------------------------------------------------------------- D
class TestTruncateAtBoundary:
    def test_short_text_unchanged(self) -> None:
        assert _truncate_at_boundary("Short evidence.", 100) == "Short evidence."

    def test_long_text_truncates_on_boundary(self) -> None:
        text = (
            "As a non-US entity, the company does not provide explicit PFIC "
            "disclosures for US tax purposes, and therefore the status is unclear."
        )
        out = _truncate_at_boundary(text, 100)
        assert out.endswith("…")
        assert "purposes,and" not in out  # no mid-word glue
        # ends on a real word, not a fragment
        assert not out[:-1].rstrip().endswith("purpos")

    def test_none_is_empty(self) -> None:
        assert _truncate_at_boundary(None) == ""

    def test_legal_flag_detail_uses_boundary(self) -> None:
        long_evidence = "x " * 200
        flags = detect_legal_flags(
            {"pfic_status": "UNCERTAIN", "pfic_evidence": long_evidence}
        )
        detail = flags[0]["detail"]
        assert "..." not in detail  # old mid-word triple-dot gone
        assert detail.endswith("…")


def test_real_artifact_pm_block_has_no_apac_remnant() -> None:
    """End-to-end guard against the exact 6831.HK rendering regression."""
    import pathlib

    path = pathlib.Path("results/6831.HK_20260624_230423_analysis.json")
    if not path.exists():
        return  # artifact not present in this checkout
    decision = json.loads(path.read_text())["final_decision"]["decision"]
    out = _strip_fenced_pm_machine_block(decision)
    assert "APAC_RESOLUTION" not in out
    assert "PM_BLOCK" not in out
    assert "DECISION RATIONALE" in out  # body preserved
