"""Tests for the APAC and Auditor fallback resolution-block inserters (Step 4b).

The inserters mirror the existing `_ensure_consultant_resolution_block` pattern:
pure deterministic string manipulation, no LLM retry.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from src.agents.decision_nodes import (
    _ensure_apac_resolution_block,
    _ensure_auditor_resolution_block,
    _extract_apac_verdict_line,
    _normalize_pm_block_contract,
    _requires_apac_resolution,
)

_PM_START_MARKER = "#### -- START PM_BLOCK --"
_PM_WITH_BLOCK = (
    f"Some PM rationale.\n\n{_PM_START_MARKER}\nVERDICT: BUY\n#### -- END PM_BLOCK --\n"
)

_PM_WITHOUT_BLOCK = "Some PM rationale without a PM_BLOCK fence.\n"


# ---------- _requires_apac_resolution ----------


@pytest.mark.parametrize(
    "apac",
    [None, "", "   ", "NO_MATERIAL_APAC_CONNECTION", "APAC_SPECIALIST_UNAVAILABLE"],
)
def test_requires_apac_resolution_false_for_silent_or_empty(apac) -> None:
    assert _requires_apac_resolution(apac) is False


def test_requires_apac_resolution_true_for_actual_audit() -> None:
    apac = (
        "### APAC REGIONAL AUDIT: 7203.T\n"
        "**VERDICT FOR CONSULTANT AND PM**: CAUTION — promoter pledges unresolved.\n"
    )
    assert _requires_apac_resolution(apac) is True


# ---------- _extract_apac_verdict_line ----------


def test_extract_apac_verdict_line_pulls_one_sentence() -> None:
    apac = (
        "**VERDICT FOR CONSULTANT AND PM**: CAUTION — controlling shareholder pledges unresolved.\n"
        "(other content)"
    )
    line = _extract_apac_verdict_line(apac)
    assert "CAUTION" in line
    assert "controlling shareholder" in line


def test_extract_apac_verdict_line_falls_back_to_tag_only() -> None:
    apac = "free-form APAC text\nVerdict: SUPPORT noted in body\nmore content"
    line = _extract_apac_verdict_line(apac)
    assert "SUPPORT" in line


def test_extract_apac_verdict_line_empty_when_no_signal() -> None:
    assert _extract_apac_verdict_line("nothing here") == ""
    assert _extract_apac_verdict_line("") == ""


# ---------- _ensure_apac_resolution_block ----------


def test_ensure_apac_inserts_before_pm_block_when_missing() -> None:
    apac = "**VERDICT FOR CONSULTANT AND PM**: CAUTION — promoter pledges unresolved."
    out = _ensure_apac_resolution_block(_PM_WITH_BLOCK, apac)
    assert "APAC_RESOLUTION:" in out
    apac_pos = out.find("APAC_RESOLUTION:")
    pm_pos = out.find(_PM_START_MARKER)
    assert apac_pos < pm_pos, "APAC block must precede PM_BLOCK"
    assert "VERDICT: UNVERIFIABLE" in out
    assert "promoter pledges" in out


def test_ensure_apac_appends_at_tail_when_no_pm_block() -> None:
    apac = "**VERDICT FOR CONSULTANT AND PM**: CAUTION — concern X."
    out = _ensure_apac_resolution_block(_PM_WITHOUT_BLOCK, apac)
    assert out.rstrip().endswith("- VERDICT: UNVERIFIABLE")
    assert "APAC_RESOLUTION:" in out


def test_ensure_apac_idempotent_when_pm_already_has_block() -> None:
    pm_with_apac = (
        "rationale\n\n"
        "APAC_RESOLUTION:\n- FINDING: handled by PM\n- VERDICT: REJECTED\n\n"
        + _PM_WITH_BLOCK
    )
    out = _ensure_apac_resolution_block(pm_with_apac, "any non-silent text")
    assert out == pm_with_apac
    # Only one APAC_RESOLUTION block in the output.
    assert out.count("APAC_RESOLUTION:") == 1


def test_ensure_apac_skips_when_silent_or_empty() -> None:
    assert _ensure_apac_resolution_block(_PM_WITH_BLOCK, None) == _PM_WITH_BLOCK
    assert (
        _ensure_apac_resolution_block(_PM_WITH_BLOCK, "NO_MATERIAL_APAC_CONNECTION")
        == _PM_WITH_BLOCK
    )
    assert _ensure_apac_resolution_block(_PM_WITH_BLOCK, "   ") == _PM_WITH_BLOCK


def test_ensure_apac_uses_generic_summary_when_verdict_unparseable() -> None:
    apac = "free-form APAC narrative with no verdict tag at all"
    out = _ensure_apac_resolution_block(_PM_WITH_BLOCK, apac)
    assert "APAC_RESOLUTION:" in out
    assert "not reconciled" in out


# ---------- _ensure_auditor_resolution_block ----------


def test_ensure_auditor_inserts_when_anomalies_present() -> None:
    auditor = "AUDIT: paper profit ratio 0.12; zombie ratio 1.1; trash bin rising."
    out = _ensure_auditor_resolution_block(_PM_WITH_BLOCK, auditor)
    assert "AUDITOR_RESOLUTION:" in out
    audit_pos = out.find("AUDITOR_RESOLUTION:")
    pm_pos = out.find(_PM_START_MARKER)
    assert audit_pos < pm_pos


def test_ensure_auditor_skips_when_empty() -> None:
    assert _ensure_auditor_resolution_block(_PM_WITH_BLOCK, None) == _PM_WITH_BLOCK
    assert _ensure_auditor_resolution_block(_PM_WITH_BLOCK, "") == _PM_WITH_BLOCK
    assert _ensure_auditor_resolution_block(_PM_WITH_BLOCK, "   ") == _PM_WITH_BLOCK


def test_ensure_auditor_skips_on_insufficient_data_sentinel() -> None:
    sentinel = "STATUS=INSUFFICIENT_DATA — auditor could not access primary filings."
    assert _ensure_auditor_resolution_block(_PM_WITH_BLOCK, sentinel) == _PM_WITH_BLOCK


def test_ensure_auditor_idempotent_when_block_already_present() -> None:
    pm_with_auditor = (
        "rationale\n\n"
        "AUDITOR_RESOLUTION:\n- FINDING: handled\n- VERDICT: REJECTED\n\n"
        + _PM_WITH_BLOCK
    )
    out = _ensure_auditor_resolution_block(pm_with_auditor, "AUDIT: anomalies present")
    assert out == pm_with_auditor
    assert out.count("AUDITOR_RESOLUTION:") == 1


# ---------- prompt regression ----------


def test_pm_prompt_requests_new_resolution_blocks() -> None:
    path = pathlib.Path("prompts/portfolio_manager.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(int(p) for p in data["version"].split(".")) >= (
        9,
        6,
    )  # Tranche 3 bumps to 9.7; floor is the introducing release.
    msg = data["system_message"]
    assert "APAC_RESOLUTION:" in msg
    assert "AUDITOR_RESOLUTION:" in msg
    # Original consultant resolution must still be intact.
    assert "CONSULTANT_RESOLUTION:" in msg


def test_normalize_pm_block_contract_rewrites_no_initiation_size() -> None:
    pm = (
        "Rationale.\n\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: HOLD\n"
        "ZONE: MODERATE\n"
        "POSITION_SIZE: 3.0\n"
        "### --- END PM_BLOCK ---\n"
    )
    out = _normalize_pm_block_contract(pm)
    assert "POSITION_SIZE: 0.0" in out
    assert "POSITION_SIZE: 3.0" not in out


def test_normalize_pm_block_contract_preserves_buy_size() -> None:
    pm = (
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        "ZONE: LOW\n"
        "POSITION_SIZE: 3.0\n"
        "### --- END PM_BLOCK ---\n"
    )
    assert _normalize_pm_block_contract(pm) == pm


def test_normalize_pm_block_contract_rewrites_only_final_block() -> None:
    pm = (
        "### --- START PM_BLOCK ---\n"
        "VERDICT: HOLD\n"
        "POSITION_SIZE: 2.0\n"
        "### --- END PM_BLOCK ---\n\n"
        "## -- START PM_BLOCK --\n"
        "VERDICT: DO_NOT_INITIATE\n"
        "POSITION_SIZE: 5.0\n"
        "## -- END PM_BLOCK --\n"
    )
    out = _normalize_pm_block_contract(pm)
    assert "POSITION_SIZE: 2.0" in out
    assert "POSITION_SIZE: 0.0" in out
    assert "POSITION_SIZE: 5.0" not in out


def test_normalize_pm_block_contract_reconciles_prose_sizing() -> None:
    # The 3773.T shape: HOLD with both a nonzero block token and a nonzero prose line.
    pm = (
        "**Recommended Position Size**: 2.5%\n\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: HOLD\n"
        "POSITION_SIZE: 2.5\n"
        "### --- END PM_BLOCK ---\n"
    )
    out = _normalize_pm_block_contract(pm)
    assert "POSITION_SIZE: 0.0" in out
    assert "2.5" not in out
    assert "Recommended Position Size**: 0.0% (monitor only — no initiation)" in out


def test_normalize_pm_block_contract_reconciles_prose_when_token_already_zero() -> None:
    # Block token already correct; prose still contradicts — must still be reconciled
    # (the case an emitted_size==0 early return would have skipped).
    pm = (
        "**Recommended Position Size**: 2.5%\n\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: HOLD\n"
        "POSITION_SIZE: 0.0\n"
        "### --- END PM_BLOCK ---\n"
    )
    out = _normalize_pm_block_contract(pm)
    assert "Recommended Position Size**: 0.0% (monitor only — no initiation)" in out


def test_normalize_pm_block_contract_preserves_buy_prose() -> None:
    pm = (
        "**Recommended Position Size**: 3.0%\n\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        "POSITION_SIZE: 3.0\n"
        "### --- END PM_BLOCK ---\n"
    )
    assert _normalize_pm_block_contract(pm) == pm


def test_demotion_then_normalize_zeroes_both_surfaces() -> None:
    # Proves the late placement composes: a BUY demoted to HOLD by
    # maybe_demote_buy_on_blocking_flags leaves stale sizing that
    # _normalize_pm_block_contract (run after) must clean.
    from src.agents.verdict_policy import maybe_demote_buy_on_blocking_flags

    pm = (
        "#### PORTFOLIO MANAGER VERDICT: BUY\n\n"
        "**Recommended Position Size**: 3.0%\n\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: BUY\n"
        "POSITION_SIZE: 3.0\n"
        "### --- END PM_BLOCK ---\n"
    )
    red_flags = [{"type": "HEALTH_SCORE_UNRELIABLE", "blocks_buy": True}]
    demoted, was_demoted = maybe_demote_buy_on_blocking_flags(
        pm, red_flags=red_flags, ticker="TEST"
    )
    assert was_demoted is True
    out = _normalize_pm_block_contract(demoted)
    assert "POSITION_SIZE: 0.0" in out
    assert "POSITION_SIZE: 3.0" not in out
    assert "Recommended Position Size**: 0.0% (monitor only — no initiation)" in out
