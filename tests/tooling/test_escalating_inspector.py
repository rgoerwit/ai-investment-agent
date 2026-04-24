"""Tests for EscalatingInspector — heuristic-first, LLM-on-escalation."""

from unittest.mock import AsyncMock

import pytest

from src.tooling.escalating_inspector import EscalatingInspector
from src.tooling.inspector import InspectionDecision, InspectionEnvelope, SourceKind


def _envelope(
    text: str = "test content",
    source_kind: SourceKind = SourceKind.financial_api,
) -> InspectionEnvelope:
    return InspectionEnvelope(
        content_text=text,
        raw_content=text,
        source_kind=source_kind,
        source_name="test",
    )


def _decision(
    action: str = "allow",
    threat_level: str = "safe",
    **kwargs,
) -> InspectionDecision:
    return InspectionDecision(action=action, threat_level=threat_level, **kwargs)


@pytest.fixture
def mock_heuristic():
    return AsyncMock()


@pytest.fixture
def mock_judge():
    return AsyncMock()


@pytest.fixture
def escalating(mock_heuristic, mock_judge):
    return EscalatingInspector(
        heuristic=mock_heuristic,
        judge=mock_judge,
        escalation_threshold="medium",
    )


# ---------------------------------------------------------------------------
# Below-threshold — judge NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_below_threshold_no_judge(escalating, mock_heuristic, mock_judge):
    """Heuristic returns safe → judge should not be called."""
    mock_heuristic.inspect.return_value = _decision("allow", "safe")

    result = await escalating.inspect(_envelope(source_kind=SourceKind.financial_api))

    mock_judge.inspect.assert_not_called()
    assert result.action == "allow"


@pytest.mark.asyncio
async def test_low_threat_no_judge(escalating, mock_heuristic, mock_judge):
    """Heuristic returns low threat → below medium threshold → no judge."""
    mock_heuristic.inspect.return_value = _decision("allow", "low")

    result = await escalating.inspect(_envelope(source_kind=SourceKind.official_filing))

    mock_judge.inspect.assert_not_called()
    assert result.threat_level == "low"


# ---------------------------------------------------------------------------
# Above-threshold — judge IS called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_above_threshold_judge_called(escalating, mock_heuristic, mock_judge):
    """Heuristic returns medium threat → judge should be called."""
    mock_heuristic.inspect.return_value = _decision(
        "degrade", "medium", threat_types=["override"]
    )
    mock_judge.inspect.return_value = _decision(
        "block", "high", threat_types=["llm_judge_malicious"]
    )

    result = await escalating.inspect(_envelope(source_kind=SourceKind.financial_api))

    mock_judge.inspect.assert_called_once()
    assert result.action == "block"
    assert result.threat_level == "high"


@pytest.mark.asyncio
async def test_high_heuristic_triggers_judge(escalating, mock_heuristic, mock_judge):
    mock_heuristic.inspect.return_value = _decision("block", "high")
    mock_judge.inspect.return_value = _decision("block", "critical")

    result = await escalating.inspect(_envelope())

    mock_judge.inspect.assert_called_once()
    assert result.threat_level == "critical"


# ---------------------------------------------------------------------------
# Optional always-judge source kinds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_not_always_judged_by_default(
    escalating, mock_heuristic, mock_judge
):
    """Default rollout should stay threshold-driven for hot-path sources."""
    mock_heuristic.inspect.return_value = _decision("allow", "safe")

    await escalating.inspect(_envelope(source_kind=SourceKind.web_search))

    mock_judge.inspect.assert_not_called()


@pytest.mark.asyncio
async def test_social_feed_not_always_judged_by_default(
    escalating, mock_heuristic, mock_judge
):
    """Default rollout should stay threshold-driven for social feeds too."""
    mock_heuristic.inspect.return_value = _decision("allow", "safe")

    await escalating.inspect(_envelope(source_kind=SourceKind.social_feed))

    mock_judge.inspect.assert_not_called()


@pytest.mark.asyncio
async def test_official_filing_no_always_judge(escalating, mock_heuristic, mock_judge):
    """official_filing is NOT in always-judge set → heuristic only when safe."""
    mock_heuristic.inspect.return_value = _decision("allow", "safe")

    await escalating.inspect(_envelope(source_kind=SourceKind.official_filing))

    mock_judge.inspect.assert_not_called()


@pytest.mark.asyncio
async def test_cached_context_no_always_judge(escalating, mock_heuristic, mock_judge):
    """cached_context is NOT in always-judge set → heuristic only when safe."""
    mock_heuristic.inspect.return_value = _decision("allow", "safe")

    await escalating.inspect(_envelope(source_kind=SourceKind.cached_context))

    mock_judge.inspect.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_always_judge_source_opt_in(mock_heuristic, mock_judge):
    esc = EscalatingInspector(
        heuristic=mock_heuristic,
        judge=mock_judge,
        always_judge_sources=frozenset({SourceKind.web_search}),
    )
    mock_heuristic.inspect.return_value = _decision("allow", "safe")
    mock_judge.inspect.return_value = _decision("allow", "safe")

    await esc.inspect(_envelope(source_kind=SourceKind.web_search))

    mock_judge.inspect.assert_called_once()


# ---------------------------------------------------------------------------
# Decision merging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_takes_more_conservative(escalating, mock_heuristic, mock_judge):
    """Merge should take the more conservative of heuristic and judge."""
    mock_heuristic.inspect.return_value = _decision(
        "degrade",
        "medium",
        threat_types=["override"],
        findings=["heuristic finding"],
        reason="heuristic reason",
    )
    mock_judge.inspect.return_value = _decision(
        "allow",
        "safe",
        threat_types=[],
        findings=["judge finding"],
        reason="judge reason",
    )

    result = await escalating.inspect(_envelope(source_kind=SourceKind.financial_api))

    # Heuristic is more conservative → its action wins.
    assert result.action == "degrade"
    assert result.threat_level == "medium"
    # Findings from both should be merged.
    assert "heuristic finding" in result.findings
    assert "judge finding" in result.findings


@pytest.mark.asyncio
async def test_merge_judge_overrides_when_more_severe(
    escalating, mock_heuristic, mock_judge
):
    mock_heuristic.inspect.return_value = _decision(
        "degrade", "medium", threat_types=["override"]
    )
    mock_judge.inspect.return_value = _decision(
        "block", "high", threat_types=["llm_judge_malicious"]
    )

    result = await escalating.inspect(_envelope(source_kind=SourceKind.financial_api))

    assert result.action == "block"
    assert result.threat_level == "high"
    assert "override" in result.threat_types
    assert "llm_judge_malicious" in result.threat_types


# ---------------------------------------------------------------------------
# Judge failure — fallback to heuristic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_failure_falls_back_to_heuristic(
    escalating, mock_heuristic, mock_judge
):
    mock_heuristic.inspect.return_value = _decision("degrade", "medium")
    mock_judge.inspect.side_effect = RuntimeError("judge crashed")

    result = await escalating.inspect(_envelope(source_kind=SourceKind.financial_api))

    assert result.action == "degrade"
    assert result.threat_level == "medium"


# ---------------------------------------------------------------------------
# Custom threshold configuration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_threshold_escalates_on_low_threat(mock_heuristic, mock_judge):
    """threshold='low' should escalate even on low-threat heuristic results."""
    esc = EscalatingInspector(
        heuristic=mock_heuristic,
        judge=mock_judge,
        escalation_threshold="low",
    )
    mock_heuristic.inspect.return_value = _decision("allow", "low")
    mock_judge.inspect.return_value = _decision("allow", "safe")

    await esc.inspect(_envelope(source_kind=SourceKind.financial_api))

    mock_judge.inspect.assert_called_once()


@pytest.mark.asyncio
async def test_high_threshold_skips_medium_threat(mock_heuristic, mock_judge):
    """threshold='high' should NOT escalate on medium-threat heuristic results."""
    esc = EscalatingInspector(
        heuristic=mock_heuristic,
        judge=mock_judge,
        escalation_threshold="high",
    )
    mock_heuristic.inspect.return_value = _decision("degrade", "medium")

    await esc.inspect(_envelope(source_kind=SourceKind.financial_api))

    mock_judge.inspect.assert_not_called()


@pytest.mark.asyncio
async def test_high_threshold_escalates_on_high_threat(mock_heuristic, mock_judge):
    """threshold='high' should escalate on high-threat heuristic results."""
    esc = EscalatingInspector(
        heuristic=mock_heuristic,
        judge=mock_judge,
        escalation_threshold="high",
    )
    mock_heuristic.inspect.return_value = _decision("block", "high")
    mock_judge.inspect.return_value = _decision("block", "critical")

    result = await esc.inspect(_envelope(source_kind=SourceKind.financial_api))

    mock_judge.inspect.assert_called_once()
    assert result.threat_level == "critical"


# ---------------------------------------------------------------------------
# Custom always-judge sources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_always_judge_sources(mock_heuristic, mock_judge):
    """Custom always_judge_sources should override default set."""
    esc = EscalatingInspector(
        heuristic=mock_heuristic,
        judge=mock_judge,
        always_judge_sources=frozenset({SourceKind.official_filing}),
    )
    mock_heuristic.inspect.return_value = _decision("allow", "safe")
    mock_judge.inspect.return_value = _decision("allow", "safe")

    # official_filing IS in custom set → judge called
    await esc.inspect(_envelope(source_kind=SourceKind.official_filing))
    mock_judge.inspect.assert_called_once()


@pytest.mark.asyncio
async def test_custom_always_judge_excludes_default(mock_heuristic, mock_judge):
    """web_search should NOT be always-judged if custom set excludes it."""
    esc = EscalatingInspector(
        heuristic=mock_heuristic,
        judge=mock_judge,
        always_judge_sources=frozenset({SourceKind.official_filing}),
    )
    mock_heuristic.inspect.return_value = _decision("allow", "safe")

    await esc.inspect(_envelope(source_kind=SourceKind.web_search))

    # web_search not in custom set AND heuristic is safe → no judge
    mock_judge.inspect.assert_not_called()


@pytest.mark.asyncio
async def test_empty_always_judge_sources(mock_heuristic, mock_judge):
    """Empty always_judge_sources means no source kind auto-escalates."""
    esc = EscalatingInspector(
        heuristic=mock_heuristic,
        judge=mock_judge,
        always_judge_sources=frozenset(),
    )
    mock_heuristic.inspect.return_value = _decision("allow", "safe")

    await esc.inspect(_envelope(source_kind=SourceKind.web_search))
    await esc.inspect(_envelope(source_kind=SourceKind.social_feed))

    mock_judge.inspect.assert_not_called()


# ---------------------------------------------------------------------------
# memory_retrieval and cached_context source kinds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_retrieval_safe_no_judge(escalating, mock_heuristic, mock_judge):
    """memory_retrieval with safe heuristic → no judge (not in always-judge)."""
    mock_heuristic.inspect.return_value = _decision("allow", "safe")

    await escalating.inspect(_envelope(source_kind=SourceKind.memory_retrieval))

    mock_judge.inspect.assert_not_called()


@pytest.mark.asyncio
async def test_memory_retrieval_medium_escalates(
    escalating, mock_heuristic, mock_judge
):
    """memory_retrieval with medium heuristic → judge IS called."""
    mock_heuristic.inspect.return_value = _decision("degrade", "medium")
    mock_judge.inspect.return_value = _decision("block", "high")

    result = await escalating.inspect(
        _envelope(source_kind=SourceKind.memory_retrieval)
    )

    mock_judge.inspect.assert_called_once()
    assert result.action == "block"


@pytest.mark.asyncio
async def test_cached_context_safe_no_judge(escalating, mock_heuristic, mock_judge):
    """cached_context with safe heuristic → no judge."""
    mock_heuristic.inspect.return_value = _decision("allow", "safe")

    await escalating.inspect(_envelope(source_kind=SourceKind.cached_context))

    mock_judge.inspect.assert_not_called()


# ---------------------------------------------------------------------------
# Sanitize action propagation through merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sanitize_propagated_from_heuristic(
    escalating, mock_heuristic, mock_judge
):
    """Sanitized content from heuristic should survive merge if judge has none."""
    mock_heuristic.inspect.return_value = _decision(
        "sanitize",
        "medium",
        sanitized_content="cleaned text",
        threat_types=["delimiter_breakout"],
    )
    mock_judge.inspect.return_value = _decision("allow", "safe")

    result = await escalating.inspect(_envelope(source_kind=SourceKind.financial_api))

    # Merge takes more conservative → sanitize wins over allow
    assert result.action == "sanitize"
    assert result.sanitized_content == "cleaned text"


@pytest.mark.asyncio
async def test_judge_sanitized_content_preferred_over_heuristic(
    escalating, mock_heuristic, mock_judge
):
    """Judge's sanitized_content should be preferred when both provide it."""
    mock_heuristic.inspect.return_value = _decision(
        "sanitize",
        "medium",
        sanitized_content="heuristic cleaned",
        threat_types=["delimiter_breakout"],
    )
    mock_judge.inspect.return_value = _decision(
        "sanitize",
        "medium",
        sanitized_content="judge cleaned",
        threat_types=["llm_judge_suspicious"],
    )

    result = await escalating.inspect(_envelope(source_kind=SourceKind.financial_api))

    assert result.sanitized_content == "judge cleaned"


# ---------------------------------------------------------------------------
# Merge edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_deduplicates_threat_types(escalating, mock_heuristic, mock_judge):
    """Duplicate threat_types across heuristic and judge should be deduped."""
    mock_heuristic.inspect.return_value = _decision(
        "degrade",
        "medium",
        threat_types=["override", "role_play"],
    )
    mock_judge.inspect.return_value = _decision(
        "degrade",
        "medium",
        threat_types=["override", "llm_judge_suspicious"],
    )

    result = await escalating.inspect(_envelope(source_kind=SourceKind.financial_api))

    # "override" appears in both — should only appear once
    assert result.threat_types.count("override") == 1
    assert "role_play" in result.threat_types
    assert "llm_judge_suspicious" in result.threat_types


@pytest.mark.asyncio
async def test_merge_combines_confidence_max(escalating, mock_heuristic, mock_judge):
    """Merged confidence should be max of both inspectors."""
    mock_heuristic.inspect.return_value = _decision("degrade", "medium", confidence=0.4)
    mock_judge.inspect.return_value = _decision("allow", "safe", confidence=0.8)

    result = await escalating.inspect(_envelope(source_kind=SourceKind.financial_api))

    assert result.confidence == 0.8


@pytest.mark.asyncio
async def test_merge_combines_reasons(escalating, mock_heuristic, mock_judge):
    """Merged reason should combine both inspectors' reasons."""
    mock_heuristic.inspect.return_value = _decision(
        "degrade", "medium", reason="heuristic flagged"
    )
    mock_judge.inspect.return_value = _decision("block", "high", reason="judge flagged")

    result = await escalating.inspect(_envelope(source_kind=SourceKind.financial_api))

    assert "heuristic" in result.reason
    assert "judge" in result.reason
