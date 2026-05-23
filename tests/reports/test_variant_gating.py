"""Tests for memo + quality-judge variant-view gating (Tranche 5, Step 8)."""

from __future__ import annotations

from src.eval.report_quality_judge import score_report
from src.reporting.memo import (
    _VARIANT_PLACEHOLDER,
    InvestmentMemo,
    render_memo_markdown,
)


def _memo(variant: str) -> InvestmentMemo:
    return InvestmentMemo(
        decision="BUY",
        one_line_thesis="A thesis.",
        variant_view=variant,
        key_numbers=["P/E: 10"],
        valuation="Target $20.",
        top_risks=["Risk A"],
        kill_criteria=["D/E > 1.0"],
        confidence="Anchored.",
    )


# ---------- memo rendering ----------


def test_memo_omits_variant_view_line_on_placeholder() -> None:
    md = render_memo_markdown(_memo(_VARIANT_PLACEHOLDER))
    assert "**Variant view.**" not in md


def test_memo_renders_variant_view_when_substantive() -> None:
    md = render_memo_markdown(_memo("Market misprices the recurring revenue mix."))
    assert "**Variant view.** Market misprices the recurring revenue mix." in md


def test_memo_renders_explicit_no_variant_alignment() -> None:
    """An honest 'no variant' is content — must still render."""
    md = render_memo_markdown(
        _memo("Synthesis aligns with consensus — no material variant.")
    )
    assert "**Variant view.**" in md
    assert "aligns with consensus" in md


# ---------- quality judge ----------


def test_judge_does_not_count_placeholder_variant() -> None:
    md = (
        "## Investment Memo — BUY\n\n"
        "**Variant view.** Not explicitly stated.\n\n"
        "More content."
    )
    score = score_report(md)
    assert score.has_variant_view is False


def test_judge_counts_substantive_variant() -> None:
    md = (
        "## Investment Memo — BUY\n\n"
        "**Variant view.** Market underprices recurring services growth.\n\n"
    )
    score = score_report(md)
    assert score.has_variant_view is True


def test_judge_counts_no_variant_alignment_from_saved_plan() -> None:
    """Honest 'NO VARIANT' in the investment plan counts as a positive feature."""
    md = "bare report"
    saved = {
        "investment_analysis": {
            "investment_plan": "CONSENSUS_VIEW: X. NO VARIANT — synthesis aligns.",
        }
    }
    score = score_report(md, saved)
    assert score.has_variant_view is True


def test_judge_falls_back_to_saved_variant_when_memo_absent() -> None:
    md = "no memo at all"
    saved = {
        "investment_plan": (
            "CONSENSUS_VIEW: The market believes X.\n"
            "VARIANT_VIEW: We believe Y.\n"
            "BASIS: native filing.\n"
        )
    }
    score = score_report(md, saved)
    assert score.has_variant_view is True
