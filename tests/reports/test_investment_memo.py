"""Tests for the memo-first report restructure (Tranche 1, Step 1)."""

from __future__ import annotations

import pytest

from src.agents.support import format_red_flag_section
from src.reporting.memo import (
    InvestmentMemo,
    build_memo,
    extract_key_metrics,
    extract_legacy_target_range,
    extract_pm_risks,
    extract_pm_thesis,
    extract_pm_verdict,
    extract_variant_view,
    render_memo_for_state,
    render_memo_markdown,
    summarize_confidence,
)

_PM_OUTPUT_BUY = """#### PORTFOLIO MANAGER VERDICT: BUY

### DECISION RATIONALE

JNBY Design (3306.HK) trades at 9.6x P/E with 39% ROIC and a 70% dividend payout, screening as a high-quality under-followed name despite China consumer exposure.

### TOP RISKS
- China discretionary slowdown could compress margins
- PFIC tax burden for US investors
- Family ownership concentration

### --- START PM_BLOCK ---
VERDICT: BUY
HEALTH_ADJ: 92
GROWTH_ADJ: 50
RISK_TALLY: 0.0
ZONE: LOW
SHOW_VALUATION_CHART: YES
VALUATION_DISCOUNT: 1.0
POSITION_SIZE: 3.5
### --- END PM_BLOCK ---
"""


_PM_OUTPUT_DNI = """#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE

## -- START PM_BLOCK --
VERDICT: DO_NOT_INITIATE
HEALTH_ADJ: 35
GROWTH_ADJ: 20
RISK_TALLY: 2.5
ZONE: HIGH
## -- END PM_BLOCK --
"""


_FUNDAMENTALS_REPORT = """## Fundamentals

Some narrative text.

### --- START DATA_BLOCK ---
SECTOR: Consumer Discretionary
PE_RATIO_TTM: 9.6
PEG_RATIO: 0.8
ROIC_PERCENT: 39.0
FCF_YIELD_PERCENT: 7.1
REVENUE_GROWTH_TTM: 4.5
NET_DEBT_TO_EBITDA: -1.2
DEBT_TO_EQUITY: 0.15
ANALYST_COVERAGE_ENGLISH: 8
CURRENT_PRICE: 20.26
### --- END DATA_BLOCK ---
"""


# ---------- extract_pm_verdict ----------


def test_extract_pm_verdict_pm_block_wins() -> None:
    assert extract_pm_verdict(_PM_OUTPUT_BUY) == "BUY"


def test_extract_pm_verdict_normalizes_do_not_initiate() -> None:
    assert extract_pm_verdict(_PM_OUTPUT_DNI) == "DO_NOT_INITIATE"


def test_extract_pm_verdict_falls_back_to_narrative() -> None:
    text = (
        "Some preamble.\n\n#### PORTFOLIO MANAGER VERDICT: HOLD\n\nRationale follows."
    )
    assert extract_pm_verdict(text) == "HOLD"


def test_extract_pm_verdict_handles_missing() -> None:
    assert extract_pm_verdict("") == "UNAVAILABLE"
    assert extract_pm_verdict("No verdict here.") == "UNAVAILABLE"


# ---------- extract_pm_thesis ----------


def test_extract_pm_thesis_pulls_first_sentence() -> None:
    thesis = extract_pm_thesis(_PM_OUTPUT_BUY)
    assert "JNBY Design" in thesis
    assert thesis.endswith(".") or thesis.endswith("…")
    assert len(thesis.split()) <= 31


def test_extract_pm_thesis_skips_numbered_list_marker() -> None:
    """Real PMs render DECISION RATIONALE as a numbered list ("1. **Lead-in**: ...").

    Regression: the extractor must skip the bare "1." enumerator and return the real
    sentence, not render the thesis as "1." (systemic memo bug).
    """
    numbered = (
        "#### PORTFOLIO MANAGER VERDICT: DO NOT INITIATE\n\n"
        "### DECISION RATIONALE\n\n"
        "1. **Elite but Uninvestable Quality**: Hugel is a control trap under private "
        "equity ownership despite a 15.15% ROIC. More text follows.\n"
    )
    thesis = extract_pm_thesis(numbered)
    assert thesis != "1."
    assert "Hugel" in thesis and "control trap" in thesis
    # Hashed-enumerator variant ("#### 1.") must behave the same.
    hashed = "### DECISION RATIONALE\n\n#### 1. **Macro Headwind**: Margins compress here. Next.\n"
    assert extract_pm_thesis(hashed).startswith("**Macro Headwind**")


def test_extract_pm_thesis_caps_long_rationale() -> None:
    long_rationale = (
        "### DECISION RATIONALE\n\n" + " ".join(["word"] * 80) + ". Trailing sentence."
    )
    thesis = extract_pm_thesis(long_rationale, max_words=10)
    assert thesis.endswith("…")
    assert len(thesis.split()) <= 11


def test_extract_pm_thesis_returns_placeholder_when_missing() -> None:
    assert extract_pm_thesis("") == "Thesis unavailable."
    assert extract_pm_thesis("VERDICT: BUY") == "Thesis unavailable."


# ---------- extract_variant_view ----------


def test_extract_variant_view_returns_placeholder_when_absent() -> None:
    assert extract_variant_view({}) == "Not explicitly stated."
    assert (
        extract_variant_view({"investment_plan": "Some plan."})
        == "Not explicitly stated."
    )


def test_extract_variant_view_pulls_explicit_block() -> None:
    plan = (
        "CONSENSUS_VIEW: The market sees this as a melting ice cube.\n"
        "VARIANT_VIEW: We see structural growth in the recurring services segment.\n"
        "BASIS: Foreign Language Analyst surfaced 32% YoY services growth.\n"
    )
    out = extract_variant_view({"investment_plan": plan})
    assert "structural growth" in out


def test_extract_variant_view_honors_no_variant() -> None:
    plan = "CONSENSUS_VIEW: Yes. NO VARIANT — synthesis aligns with consensus."
    out = extract_variant_view({"investment_plan": plan})
    assert "aligns with consensus" in out


# ---------- extract_key_metrics ----------


def test_extract_key_metrics_pulls_in_order() -> None:
    metrics = extract_key_metrics(_FUNDAMENTALS_REPORT, limit=6)
    assert len(metrics) == 6
    assert metrics[0].startswith("P/E (TTM):")
    assert any("ROIC" in m for m in metrics)
    assert any("Analyst coverage" not in m for m in metrics) or len(metrics) >= 6


def test_extract_key_metrics_skips_na_values() -> None:
    report = (
        "### --- START DATA_BLOCK ---\n"
        "PE_RATIO_TTM: 14.0\n"
        "PEG_RATIO: N/A\n"
        "ROIC_PERCENT: 21.5\n"
        "### --- END DATA_BLOCK ---\n"
    )
    metrics = extract_key_metrics(report, limit=6)
    assert any("P/E" in m for m in metrics)
    assert not any("PEG" in m for m in metrics)
    assert any("ROIC" in m for m in metrics)


def test_extract_key_metrics_empty_when_no_data_block() -> None:
    assert extract_key_metrics("", limit=6) == []
    assert extract_key_metrics("narrative only, no data block", limit=6) == []


# ---------- extract_legacy_target_range ----------


def test_extract_legacy_target_range_from_valuation_context() -> None:
    state = {
        "valuation_context": (
            "VALUATION DATA (from Football Field Chart):\n"
            "- Methodology: P/E Normalization\n"
            "- Target Range: $18.00 - $24.00\n"
            "Fair Value (midpoint): $21.00\n"
            "- Current Price: $20.26\n"
        )
    }
    out = extract_legacy_target_range(state)
    assert "Target range" in out
    assert "$18.00 - $24.00" in out


def test_extract_legacy_target_range_falls_back_to_current_price() -> None:
    state = {"fundamentals_report": _FUNDAMENTALS_REPORT}
    out = extract_legacy_target_range(state)
    assert "20.26" in out


def test_extract_legacy_target_range_unavailable() -> None:
    out = extract_legacy_target_range({})
    assert "unavailable" in out.lower()


# ---------- extract_pm_risks ----------


def test_extract_pm_risks_prefers_red_flags() -> None:
    red_flags = [
        {
            "type": "PFIC_UNCERTAIN",
            "detail": "PFIC status unclear",
            "risk_penalty": 1.0,
        },
        {
            "type": "HIGH_JURISDICTION_RISK",
            "detail": "China consumer exposure",
            "risk_penalty": 0.5,
        },
    ]
    risks = extract_pm_risks(_PM_OUTPUT_BUY, red_flags, limit=4)
    assert len(risks) >= 2
    assert risks[0].startswith("PFIC_UNCERTAIN:")


def test_extract_pm_risks_skips_notes_and_bonus_flags() -> None:
    red_flags = [
        {
            "type": "OCF_PERIOD_MISMATCH_RESOLVED",
            "detail": "period mismatch",
            "action": "NOTE",
            "risk_penalty": 0.0,
        },
        {
            "type": "MOAT_PRICING_POWER",
            "detail": "durable pricing power",
            "risk_penalty": -0.5,
        },
        {
            "type": "LOCAL_COVERAGE_HIGH",
            "detail": "moderate local coverage",
            "risk_penalty": 0.25,
        },
    ]
    risks = extract_pm_risks("", red_flags, limit=4)
    assert risks == ["LOCAL_COVERAGE_HIGH: moderate local coverage"]


def test_extract_pm_risks_keeps_critical_zero_penalty_flags() -> None:
    red_flags = [
        {
            "type": "STRUCTURAL_STOP",
            "detail": "hard governance stop",
            "severity": "CRITICAL",
            "risk_penalty": 0.0,
        },
    ]
    assert extract_pm_risks("", red_flags) == ["STRUCTURAL_STOP: hard governance stop"]


def test_valuation_quarantine_is_visible_but_not_a_top_risk() -> None:
    red_flags = [
        {
            "type": "VALUATION_INPUT_QUARANTINED",
            "detail": "Distrusted valuation inputs — verify before using as BUY support.",
            "severity": "WARNING",
            "action": "REVIEW",
            "risk_penalty": 0.0,
        }
    ]

    assert extract_pm_risks("", red_flags, limit=4) == []
    rendered, subtotal = format_red_flag_section("PASS", red_flags)
    assert subtotal == 0.0
    assert "VALUATION_INPUT_QUARANTINED [risk_penalty +0.00]" in rendered
    assert "verify before using as BUY support" in rendered


def test_extract_pm_risks_falls_back_to_pm_narrative() -> None:
    risks = extract_pm_risks(_PM_OUTPUT_BUY, red_flags=[], limit=4)
    assert any("China discretionary" in r for r in risks)


def test_extract_pm_risks_empty() -> None:
    assert extract_pm_risks("", None, limit=4) == []


# ---------- summarize_confidence ----------


def test_summarize_confidence_lists_successful_agents() -> None:
    state = {
        "run_summary": {
            "consultant_successful": True,
            "auditor_successful": True,
            "apac_specialist_successful": True,
        }
    }
    out = summarize_confidence(state)
    assert "consultant" in out and "auditor" in out and "APAC" in out


def test_summarize_confidence_when_nothing_ran() -> None:
    out = summarize_confidence({})
    assert "did not run" in out


def test_summarize_confidence_consultant_error_is_validation_failure() -> None:
    out = summarize_confidence(
        {
            "run_summary": {
                "consultant_completed": True,
                "consultant_successful": False,
                "consultant_verdict": "ERROR",
            }
        }
    )
    assert "consultant review failed validation" in out
    assert "consultant ran but did not approve" not in out


# ---------- build_memo + render_memo_markdown (happy/edge/error) ----------


def test_build_memo_happy_path_buy() -> None:
    state = {
        "final_trade_decision": _PM_OUTPUT_BUY,
        "fundamentals_report": _FUNDAMENTALS_REPORT,
        "investment_debate_state": {
            "bear_history": (
                "### --- START KILL_CRITERIA ---\n"
                "TRIGGER_1: D/E exceeds 1.0\n"
                "TRIGGER_2: two consecutive quarters of revenue decline\n"
                "### --- END KILL_CRITERIA ---\n"
            ),
        },
        "red_flags": [
            {
                "type": "PFIC_UNCERTAIN",
                "detail": "PFIC status unclear",
                "risk_penalty": 1.0,
            },
        ],
        "run_summary": {
            "consultant_successful": True,
            "apac_specialist_successful": True,
        },
    }
    memo = build_memo(state)
    assert memo.decision == "BUY"
    assert "JNBY Design" in memo.one_line_thesis
    assert memo.key_numbers and any("P/E" in m for m in memo.key_numbers)
    assert memo.kill_criteria == [
        "D/E exceeds 1.0",
        "two consecutive quarters of revenue decline",
    ]
    assert memo.top_risks[0].startswith("PFIC_UNCERTAIN")
    assert "consultant" in memo.confidence


def test_build_memo_uses_effective_resolved_ocf_flags() -> None:
    state = {
        "final_trade_decision": (
            "### PORTFOLIO MANAGER VERDICT: HOLD\n\n"
            "### DECISION RATIONALE\n"
            "Hold while cash-flow period mismatch is reconciled.\n\n"
            "### --- START PM_BLOCK ---\nVERDICT: HOLD\n### --- END PM_BLOCK ---\n"
        ),
        "fundamentals_report": (
            "### --- START DATA_BLOCK ---\n"
            "OPERATING_CASH_FLOW: 151.97M PLN\n"
            "OPERATING_CASH_FLOW_SOURCE: FILING\n"
            "OCF_FILING_REASON: DISCREPANCY\n"
            "### --- END DATA_BLOCK ---\n"
        ),
        "consultant_review": (
            "SPOT_CHECK operatingCashflow: DATA_BLOCK 151.97m PLN FY2025; "
            "FMP 178.06m PLN TTM/Q1 — PERIOD MISMATCH, not a data conflict."
        ),
        "auditor_report": "Operating cash flow: PLN 151.967m",
        "red_flags": [
            {
                "type": "OCF_SOURCE_DISCREPANCY",
                "detail": "OCF value sourced from filing differs from API data",
                "risk_penalty": 0.5,
            }
        ],
    }
    memo = build_memo(state)
    assert memo.top_risks == []
    assert "period mismatch" in memo.source_confidence[0][1]


def test_render_memo_markdown_happy_path_renders_all_sections() -> None:
    memo = InvestmentMemo(
        decision="BUY",
        one_line_thesis="A focused thesis.",
        variant_view="Market is wrong on margin trajectory.",
        key_numbers=["P/E: 10", "ROIC: 22%"],
        valuation="Target range $18-$24.",
        top_risks=["China consumer slowdown"],
        kill_criteria=["D/E > 1.0"],
        confidence="Anchored on consultant.",
    )
    md = render_memo_markdown(memo)
    assert md.startswith("## Investment Memo — BUY")
    assert "**Thesis.**" in md
    assert "**Variant view.**" in md
    assert "**Key numbers.**" in md
    assert "**Valuation.**" in md
    assert "**Top risks.**" in md
    assert "**Kill criteria.**" in md
    assert "**Confidence.**" in md
    assert md.rstrip().endswith("---")


def test_render_memo_markdown_dni_omits_optional_sections() -> None:
    memo = InvestmentMemo(
        decision="DO_NOT_INITIATE",
        one_line_thesis="Thesis fails on coverage.",
        variant_view="Not explicitly stated.",
        key_numbers=[],
        valuation="Valuation summary unavailable.",
        top_risks=[],
        kill_criteria=[],
        confidence="Optional cross-checks did not run.",
    )
    md = render_memo_markdown(memo)
    assert "## Investment Memo — DO_NOT_INITIATE" in md
    assert "**Kill criteria.**" not in md
    assert "**Key numbers.**" not in md
    assert "**Top risks.**" not in md


def test_render_memo_for_state_empty_input_returns_unavailable_stub() -> None:
    md = render_memo_for_state({})
    assert "Investment Memo — UNAVAILABLE" in md
    assert md.endswith("\n")


def test_render_memo_for_state_saved_json_shape_finds_bear_history() -> None:
    saved = {
        "final_trade_decision": _PM_OUTPUT_BUY,
        "fundamentals_report": _FUNDAMENTALS_REPORT,
        "investment_analysis": {
            "investment_debate": {
                "bear_history": (
                    "### --- START KILL_CRITERIA ---\n"
                    "TRIGGER_1: stop breach\n"
                    "### --- END KILL_CRITERIA ---\n"
                ),
            },
        },
    }
    md = render_memo_for_state(saved)
    assert "Kill criteria" in md
    assert "stop breach" in md


# ---------- regression: report generator wiring ----------


def test_report_generator_emits_memo_above_executive_summary() -> None:
    """End-to-end smoke test: memo appears in rendered report before Executive Summary."""
    from src.report_generator import QuietModeReporter

    reporter = QuietModeReporter(
        ticker="3306.HK",
        company_name="JNBY Design",
        quick_mode=False,
        skip_charts=True,
    )
    result = {
        "final_trade_decision": _PM_OUTPUT_BUY,
        "fundamentals_report": _FUNDAMENTALS_REPORT,
        "investment_debate_state": {
            "bear_history": (
                "### --- START KILL_CRITERIA ---\n"
                "TRIGGER_1: D/E exceeds 1.0\n"
                "### --- END KILL_CRITERIA ---\n"
            )
        },
        "red_flags": [],
        "run_summary": {"consultant_successful": True},
    }
    report = reporter.generate_report(result)
    memo_pos = report.find("## Investment Memo")
    exec_pos = report.find("## Executive Summary")
    assert memo_pos >= 0, "Memo section missing"
    assert exec_pos >= 0, "Executive Summary section missing"
    assert memo_pos < exec_pos, "Memo must precede Executive Summary"
