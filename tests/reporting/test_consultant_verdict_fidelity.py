"""Tests that the memo/source-confidence renderers distinguish "consultant ran"
from "consultant approved".

`consultant_successful` (bare `ok`) only means the consultant returned a
parseable review. The derived `consultant_verdict` is the approval signal. Old
saved JSON (no `consultant_verdict`) must still render via the legacy fallback.
"""

from __future__ import annotations

from src.persistence import _derive_consultant_verdict, build_run_summary
from src.reporting.memo import summarize_confidence
from src.reporting.source_confidence import build_source_confidence_rows


def _status(content: str, *, ok: bool = True, complete: bool = True) -> dict:
    return {"ok": ok, "complete": complete, "content": content}


# --------------------------------------------------------------------------- #
# _derive_consultant_verdict
# --------------------------------------------------------------------------- #


def test_derive_clean():
    assert (
        _derive_consultant_verdict(
            _status("### CONSULTANT REVIEW: APPROVED\nAnalysis is sound.")
        )
        == "CLEAN"
    )


def test_derive_conditional():
    assert (
        _derive_consultant_verdict(
            _status("### CONSULTANT REVIEW: CONDITIONAL APPROVAL\nVerify OCF period.")
        )
        == "CONDITIONAL"
    )


def test_derive_major_concerns():
    assert (
        _derive_consultant_verdict(
            _status("### CONSULTANT REVIEW: MAJOR CONCERNS\nSynthesis errors found.")
        )
        == "MAJOR_CONCERNS"
    )


def test_derive_hard_stop_is_rejected():
    assert (
        _derive_consultant_verdict(
            _status("HARD STOP: security on the restricted list.")
        )
        == "REJECTED"
    )


def test_derive_mandate_breach_is_major_concerns():
    assert (
        _derive_consultant_verdict(_status("MANDATE BREACH: PFIC threshold exceeded."))
        == "MAJOR_CONCERNS"
    )


def test_derive_conditional_when_verdict_clears_mandate_breach():
    # Regression: 3393.T 2026-07-04 16:05 — "No mandate breach triggered" in the
    # FINAL CONSULTANT VERDICT was read as a breach → false MAJOR_CONCERNS.
    review = (
        "FCF margin is compressing quarterly, though not yet a mandate breach.\n\n"
        "### FINAL CONSULTANT VERDICT\n\n"
        "**Overall Assessment**: CONDITIONAL APPROVAL\n\n"
        "- No mandate breach triggered: **PFIC_RISK=MEDIUM**, **CMIC clear**, "
        "health well above Tier-3 warning level.\n"
    )
    assert _derive_consultant_verdict(_status(review)) == "CONDITIONAL"


def test_derive_unparsed_when_ran_ok_but_no_verdict():
    assert _derive_consultant_verdict(_status("Some prose without a verdict.")) == (
        "UNPARSED"
    )


def test_derive_bypass_is_skipped_by_provider():
    # Quick-mode gate bypass writes a completed, ok artifact with provider="bypass".
    # It is an intentional skip, not a garbled review — must read as SKIPPED.
    status = {
        "ok": True,
        "complete": True,
        "provider": "bypass",
        "content": (
            "SKIPPED_BY_GATE: External Consultant bypass active for quick-mode "
            "screening. Reason: rm_clear_negative."
        ),
    }
    assert _derive_consultant_verdict(status) == "SKIPPED"


def test_derive_bypass_is_skipped_by_sentinel_prefix():
    # Even without the provider tag, the SKIPPED_BY_GATE sentinel identifies a bypass.
    assert (
        _derive_consultant_verdict(
            _status("SKIPPED_BY_GATE: bypass active. Reason: x.")
        )
        == "SKIPPED"
    )


def test_derive_error_when_complete_but_not_ok():
    assert _derive_consultant_verdict(_status("", ok=False, complete=True)) == "ERROR"


def test_derive_not_run_when_absent():
    assert _derive_consultant_verdict({}) == "NOT_RUN"
    assert _derive_consultant_verdict(_status("", ok=False, complete=False)) == (
        "NOT_RUN"
    )


# --------------------------------------------------------------------------- #
# memo.summarize_confidence
# --------------------------------------------------------------------------- #


def _memo(verdict, **extra) -> str:
    summary = {"consultant_completed": True}
    if verdict is not None:
        summary["consultant_verdict"] = verdict
    summary.update(extra)
    return summarize_confidence({"run_summary": summary})


def test_memo_clean_says_passed():
    assert "consultant cross-check passed" in _memo("CLEAN")


def test_memo_conditional_does_not_say_passed():
    text = _memo("CONDITIONAL")
    assert "approved with conditions" in text
    assert "passed" not in text


def test_memo_major_concerns():
    assert "major concerns" in _memo("MAJOR_CONCERNS")


def test_memo_rejected_says_not_approved():
    text = _memo("REJECTED")
    assert "did NOT approve" in text
    assert "passed" not in text


def test_memo_skipped_says_skipped_not_unparsed():
    text = _memo("SKIPPED")
    assert "skipped" in text
    assert "unparsed" not in text


def test_memo_legacy_fallback_passes():
    # Pre-change saved JSON: no consultant_verdict but consultant_successful True.
    text = summarize_confidence(
        {"run_summary": {"consultant_successful": True, "consultant_completed": True}}
    )
    assert "consultant cross-check passed" in text


# --------------------------------------------------------------------------- #
# source_confidence.build_source_confidence_rows
# --------------------------------------------------------------------------- #


def _consultant_row(verdict, **extra) -> tuple[str, str, str]:
    summary = {"consultant_completed": True}
    if verdict is not None:
        summary["consultant_verdict"] = verdict
    summary.update(extra)
    rows = build_source_confidence_rows({"run_summary": summary})
    return next(r for r in rows if r[0] == "Cross-model review")


def test_source_clean_is_high():
    assert _consultant_row("CLEAN") == (
        "Cross-model review",
        "Consultant (gpt-5.4)",
        "HIGH",
    )


def test_source_conditional_is_medium():
    claim, source, conf = _consultant_row("CONDITIONAL")
    assert conf == "MEDIUM"
    assert "conditional" in source.lower()


def test_source_major_concerns_is_low():
    assert _consultant_row("MAJOR_CONCERNS")[2] == "LOW"


def test_source_rejected_is_low():
    claim, source, conf = _consultant_row("REJECTED")
    assert conf == "LOW"
    assert "not approved" in source.lower()


def test_source_skipped_is_neutral_not_low():
    # A gate bypass must not read as LOW confidence (which "unparsed" would give);
    # it is a deliberate skip → neutral "—", labelled as a bypass.
    claim, source, conf = _consultant_row("SKIPPED")
    assert conf == "—"
    assert "bypassed" in source.lower()


def test_source_legacy_fallback_is_high():
    rows = build_source_confidence_rows(
        {"run_summary": {"consultant_successful": True, "consultant_completed": True}}
    )
    row = next(r for r in rows if r[0] == "Cross-model review")
    assert row == ("Cross-model review", "Consultant (gpt-5.4)", "HIGH")


def test_source_not_run():
    rows = build_source_confidence_rows({"run_summary": {}})
    row = next(r for r in rows if r[0] == "Cross-model review")
    assert row == ("Cross-model review", "Not run", "—")


# --------------------------------------------------------------------------- #
# build_run_summary integration
# --------------------------------------------------------------------------- #


def test_build_run_summary_sets_conditional_verdict_despite_ok():
    result = {
        "artifact_statuses": {
            "consultant_review": _status(
                "### CONSULTANT REVIEW: CONDITIONAL APPROVAL\nVerify the OCF period."
            )
        }
    }
    summary = build_run_summary(result, quick_mode=False, article_requested=False)
    # `ok` is True, but the verdict is CONDITIONAL — the whole point of the fix.
    assert summary["consultant_successful"] is True
    assert summary["consultant_verdict"] == "CONDITIONAL"


def test_build_run_summary_clean_verdict():
    result = {
        "artifact_statuses": {
            "consultant_review": _status("### CONSULTANT REVIEW: APPROVED\nSound.")
        }
    }
    summary = build_run_summary(result, quick_mode=False, article_requested=False)
    assert summary["consultant_verdict"] == "CLEAN"


def test_build_run_summary_not_run_when_consultant_absent():
    summary = build_run_summary({}, quick_mode=False, article_requested=False)
    assert summary["consultant_verdict"] == "NOT_RUN"


def test_build_run_summary_quick_gate_bypass_reads_skipped_not_unparsed():
    # Regression: quick-mode runs (005830.KS/AUTO.L/NRO.PA, 2026-07) bypassed the
    # Consultant via the gate but run_summary reported consultant_verdict=UNPARSED,
    # which renders as LOW-confidence "review unparsed". A bypass must read SKIPPED.
    result = {
        "artifact_statuses": {
            "consultant_review": {
                "complete": True,
                "ok": True,
                "provider": "bypass",
                "content": (
                    "SKIPPED_BY_GATE: External Consultant bypass active for "
                    "quick-mode screening. Reason: rm_clear_negative."
                ),
            }
        }
    }
    summary = build_run_summary(result, quick_mode=True, article_requested=False)
    assert summary["consultant_verdict"] == "SKIPPED"
