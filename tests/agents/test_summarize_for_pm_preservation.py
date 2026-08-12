"""Tests for block preservation in summarize_for_pm (Tranche 5, Step 6)."""

from __future__ import annotations

from src.agents.support import summarize_for_pm

_PADDING = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20).strip()


def _long_report(*tail_blocks: str, head_paragraphs: int = 50) -> str:
    """Build a report with `head_paragraphs` filler chunks then the supplied tail blocks."""
    head = "\n\n".join(f"Paragraph {i}: {_PADDING}" for i in range(head_paragraphs))
    return head + "\n\n" + "\n\n".join(tail_blocks)


def test_kill_criteria_block_preserved_after_truncation() -> None:
    kill_block = (
        "### --- START KILL_CRITERIA ---\n"
        "TRIGGER_1: D/E > 1.5\n"
        "TRIGGER_2: 2 consecutive years negative FCF\n"
        "### --- END KILL_CRITERIA ---"
    )
    report = _long_report(kill_block)
    out = summarize_for_pm(report, "research", max_chars=3000)
    assert "### --- START KILL_CRITERIA ---" in out
    assert "TRIGGER_1: D/E > 1.5" in out
    assert "[...summarized...]" in out


def test_glued_data_block_start_preserved_after_truncation() -> None:
    data_block = (
        "Prior sentence ends here.### --- START DATA_BLOCK ---\n"
        "SECTOR: Energy\n"
        "ADJUSTED_HEALTH_SCORE: 75%\n"
        "### --- END DATA_BLOCK ---"
    )
    report = _long_report(data_block)

    out = summarize_for_pm(report, "fundamentals", max_chars=3000)

    assert "### --- START DATA_BLOCK ---" in out
    assert "SECTOR: Energy" in out
    assert "[...summarized...]" in out


def test_valuation_scenarios_block_preserved() -> None:
    scenarios = (
        "### --- START VALUATION_SCENARIOS ---\n"
        "METHODOLOGY: P/E\nDATA_SUFFICIENCY: HIGH\n"
        "BEAR_MULTIPLE: 8\nBASE_MULTIPLE: 12\nBULL_MULTIPLE: 16\n"
        "### --- END VALUATION_SCENARIOS ---"
    )
    report = _long_report(scenarios)
    out = summarize_for_pm(report, "valuation", max_chars=3000)
    assert "VALUATION_SCENARIOS" in out
    assert "BEAR_MULTIPLE: 8" in out


def test_variant_perception_section_preserved() -> None:
    variant = (
        "### VARIANT PERCEPTION\n\n"
        "CONSENSUS_VIEW: Market sees decline.\n"
        "VARIANT_VIEW: We see recovery.\n"
        "BASIS: Order book +30%.\n"
    )
    report = _long_report(variant)
    out = summarize_for_pm(report, "research", max_chars=3000)
    assert "VARIANT PERCEPTION" in out
    assert "VARIANT_VIEW: We see recovery" in out


def test_multiple_resolution_blocks_preserved_together() -> None:
    blocks = [
        "APAC_RESOLUTION:\n- VERDICT: CONFIRMED_RISK",
        "AUDITOR_RESOLUTION:\n- VERDICT: REJECTED",
        "CONSULTANT_RESOLUTION:\n- VERDICT: UNVERIFIABLE",
    ]
    report = _long_report(*blocks)
    out = summarize_for_pm(report, "pm", max_chars=3000)
    assert "APAC_RESOLUTION" in out
    assert "AUDITOR_RESOLUTION" in out
    assert "CONSULTANT_RESOLUTION" in out


def test_short_report_returned_as_is() -> None:
    short = (
        "Short body.\n\n"
        "### --- START KILL_CRITERIA ---\nTRIGGER_1: x\n### --- END KILL_CRITERIA ---"
    )
    out = summarize_for_pm(short, "research", max_chars=3000)
    # No truncation marker, original content intact.
    assert out == short
    assert "[...summarized...]" not in out


def test_no_double_injection_when_block_in_head_window() -> None:
    """If the kept head already contains the block, it isn't re-appended."""
    kill_block = (
        "### --- START KILL_CRITERIA ---\nTRIGGER_1: x\n### --- END KILL_CRITERIA ---"
    )
    # Put the block near the start (within retained head) so we can verify dedup.
    report = kill_block + "\n\n" + "\n\n".join(f"P{i}: {_PADDING}" for i in range(40))
    out = summarize_for_pm(report, "research", max_chars=3000)
    assert out.count("### --- START KILL_CRITERIA ---") == 1


def test_malformed_fenced_block_safely_ignored() -> None:
    """Missing END marker → not preserved, but truncation doesn't raise."""
    half_block = "### --- START KILL_CRITERIA ---\nTRIGGER_1: x (no end marker present)"
    report = _long_report(half_block)
    out = summarize_for_pm(report, "research", max_chars=3000)
    # No crash; the half-block simply isn't preserved as a tail block.
    assert "[...summarized...]" in out


def test_empty_input_returns_empty() -> None:
    assert summarize_for_pm("", "research", max_chars=3000) == ""
    assert summarize_for_pm(None, "research", max_chars=3000) == ""  # type: ignore[arg-type]
