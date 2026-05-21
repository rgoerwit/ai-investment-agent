"""Tests for the truthful auditor fallback (Tranche 5, Step 7).

The original implementation fired a fallback claiming the auditor flagged
anomalies on any non-empty, non-INSUFFICIENT_DATA report — including clean
audits. The refined gate runs negative phrases first (so substring tokens
like ``RED FLAG`` don't trip on ``no red flags``) and only emits when the
auditor named at least one specific forensic check.
"""

from __future__ import annotations

import pytest

from src.agents.decision_nodes import (
    _auditor_has_material_concern,
    _ensure_auditor_resolution_block,
)

_PM_WITH_BLOCK = (
    "Some PM rationale.\n\n"
    "### --- START PM_BLOCK ---\nVERDICT: BUY\n### --- END PM_BLOCK ---\n"
)


# ---------- _auditor_has_material_concern ----------


@pytest.mark.parametrize(
    "auditor",
    [
        None,
        "",
        "   ",
        "STATUS=INSUFFICIENT_DATA — auditor could not access primary filings.",
        "Auditor ran with NO ANOMALIES detected.",
        "Audit clean: no material concerns identified.",
        "no red flags detected in the financials.",  # the substring-match false positive
        "No red flag patterns observed.",
        "ANOMALY_COUNT: 0",
        "ANOMALIES: NONE",
        "AUDITOR_VERDICT: CLEAN",
    ],
)
def test_clean_or_silent_auditor_does_not_fire(auditor) -> None:
    assert _auditor_has_material_concern(auditor) is False


@pytest.mark.parametrize(
    "auditor",
    [
        "AUDIT: paper profit ratio 0.12 indicates earnings quality concern.",
        "Zombie ratio of 1.1 — coverage below threshold.",
        "Trash bin (other receivables) growing 35% YoY.",
        "Ghost cash yield: interest income suggests restricted balances.",
        "Forensic flag: ballooning DSO from 45 to 89 days.",
        "Acquisition hangover: goodwill at 42% of assets.",
        "Inventory hoarding detected: turns down 30%.",
        "Stretching payables: DPO up from 60 to 95 days.",
    ],
)
def test_named_forensic_check_fires(auditor: str) -> None:
    assert _auditor_has_material_concern(auditor) is True


def test_generic_anomaly_word_does_not_fire_without_named_check() -> None:
    """Generic 'anomaly' or 'concern' alone is too weak — we want a named check."""
    weak = "Auditor noted some anomalies and general concerns about reporting cadence."
    assert _auditor_has_material_concern(weak) is False


def test_no_red_flags_phrasing_resists_red_flag_substring_match() -> None:
    """The reviewer-flagged regression: 'no red flags' should NOT match 'RED FLAG'."""
    cases = [
        "After detailed review, no red flags identified.",
        "Conclusion: NO RED FLAGS observed.",
        "no red flag patterns in the cash flow statement.",
    ]
    for case in cases:
        assert _auditor_has_material_concern(case) is False, case


def test_named_check_inside_a_clean_summary_still_does_not_fire() -> None:
    """If a clean negation exists alongside a positive token, prefer the negation
    (conservative: false negatives over false positives)."""
    mixed = "Paper profit ratio reviewed — no material concerns identified."
    # The negative phrase wins; this returns False (conservative).
    assert _auditor_has_material_concern(mixed) is False


# ---------- _ensure_auditor_resolution_block ----------


def test_clean_audit_does_not_inject_fallback() -> None:
    clean = "After detailed forensic review, no material concerns identified."
    assert _ensure_auditor_resolution_block(_PM_WITH_BLOCK, clean) == _PM_WITH_BLOCK


def test_named_anomaly_injects_fallback() -> None:
    flagged = "Paper profit ratio 0.18 indicates earnings quality concern."
    out = _ensure_auditor_resolution_block(_PM_WITH_BLOCK, flagged)
    assert "AUDITOR_RESOLUTION:" in out
    assert "VERDICT: UNVERIFIABLE" in out


def test_idempotent_when_pm_already_has_block() -> None:
    pm_with_block = (
        "rationale\n\n"
        "AUDITOR_RESOLUTION:\n- FINDING: handled\n- VERDICT: REJECTED\n\n"
        + _PM_WITH_BLOCK
    )
    out = _ensure_auditor_resolution_block(
        pm_with_block, "Zombie ratio 0.8 — solvency concern."
    )
    assert out == pm_with_block
    assert out.count("AUDITOR_RESOLUTION:") == 1


def test_empty_auditor_is_silent() -> None:
    assert _ensure_auditor_resolution_block(_PM_WITH_BLOCK, None) == _PM_WITH_BLOCK
    assert _ensure_auditor_resolution_block(_PM_WITH_BLOCK, "") == _PM_WITH_BLOCK
    assert _ensure_auditor_resolution_block(_PM_WITH_BLOCK, "   ") == _PM_WITH_BLOCK
