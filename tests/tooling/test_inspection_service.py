"""Tests for InspectionService and inspection primitives."""

from __future__ import annotations

import pytest

from src.tooling.inspection_service import (
    InspectionService,
    configure_content_inspection,
)
from src.tooling.inspector import (
    CompositeInspector,
    InspectionDecision,
    InspectionEnvelope,
    NullInspector,
    SourceKind,
)


def _envelope(text: str = "hello") -> InspectionEnvelope:
    return InspectionEnvelope(
        content_text=text,
        source_kind=SourceKind.web_search,
        source_name="test",
    )


class _BlockingInspector:
    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        return InspectionDecision(
            action="block",
            threat_level="high",
            threat_types=["prompt_injection"],
            findings=["injection detected"],
            reason="test block",
        )


class _SanitizingInspector:
    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        return InspectionDecision(
            action="sanitize",
            threat_level="medium",
            sanitized_content="[sanitized]",
            findings=["pii found"],
        )


class _FlaggingInspector:
    """Returns allow with findings (non-trivial threat level)."""

    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        return InspectionDecision(
            action="allow",
            threat_level="low",
            findings=["suspicious pattern"],
        )


class _ErrorInspector:
    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        raise RuntimeError("backend down")


# ---------------------------------------------------------------------------
# NullInspector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_null_inspector_always_allows():
    inspector = NullInspector()
    decision = await inspector.inspect(_envelope("anything"))
    assert decision.action == "allow"
    assert decision.threat_level == "safe"


# ---------------------------------------------------------------------------
# InspectionService — warn mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warn_mode_passes_content_through():
    svc = InspectionService(_BlockingInspector(), mode="warn")
    result = await svc.check(_envelope("dangerous content"))
    assert result == "dangerous content"


@pytest.mark.asyncio
async def test_warn_mode_preserves_raw_content_shape_when_allowed():
    raw_payload = {"results": [{"title": "hello"}]}
    svc = InspectionService(NullInspector(), mode="warn")
    result = await svc.check(
        InspectionEnvelope(
            content_text=str(raw_payload),
            raw_content=raw_payload,
            source_kind=SourceKind.web_search,
            source_name="tavily",
        )
    )
    assert result is raw_payload


@pytest.mark.asyncio
async def test_warn_mode_null_inspector_passthrough():
    svc = InspectionService(NullInspector(), mode="warn")
    result = await svc.check(_envelope("safe content"))
    assert result == "safe content"


# ---------------------------------------------------------------------------
# InspectionService — sanitize mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sanitize_mode_replaces_content():
    svc = InspectionService(_SanitizingInspector(), mode="sanitize")
    result = await svc.check(_envelope("original"))
    assert result == "[sanitized]"


@pytest.mark.asyncio
async def test_sanitize_mode_block_action_without_sanitized_content():
    """block action with no sanitized_content → pass through in sanitize mode."""

    class BlockNoSanitize:
        async def inspect(self, envelope):
            return InspectionDecision(action="block", threat_level="high")

    svc = InspectionService(BlockNoSanitize(), mode="sanitize")
    result = await svc.check(_envelope("content"))
    assert result == "content"


# ---------------------------------------------------------------------------
# InspectionService — block mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_block_mode_returns_blocked_placeholder():
    svc = InspectionService(_BlockingInspector(), mode="block")
    result = await svc.check(_envelope("dangerous"))
    assert result.startswith("TOOL_BLOCKED:")


@pytest.mark.asyncio
async def test_block_mode_null_inspector_passthrough():
    svc = InspectionService(NullInspector(), mode="block")
    result = await svc.check(_envelope("safe"))
    assert result == "safe"


# ---------------------------------------------------------------------------
# Fail policies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_open_allows_on_backend_error():
    svc = InspectionService(_ErrorInspector(), mode="block", fail_policy="fail_open")
    result = await svc.check(_envelope("content"))
    assert result == "content"


@pytest.mark.asyncio
async def test_fail_open_preserves_raw_content_shape_on_backend_error():
    raw_payload = {"results": [{"title": "hello"}]}
    svc = InspectionService(_ErrorInspector(), mode="block", fail_policy="fail_open")
    result = await svc.check(
        InspectionEnvelope(
            content_text=str(raw_payload),
            raw_content=raw_payload,
            source_kind=SourceKind.web_search,
            source_name="tavily",
        )
    )
    assert result is raw_payload


@pytest.mark.asyncio
async def test_fail_closed_blocks_on_backend_error():
    svc = InspectionService(_ErrorInspector(), mode="block", fail_policy="fail_closed")
    result = await svc.check(_envelope("content"))
    assert result.startswith("TOOL_BLOCKED:")


@pytest.mark.asyncio
async def test_evaluate_returns_decision_and_original_content():
    svc = InspectionService(NullInspector(), mode="warn")
    decision, approved = await svc.evaluate(_envelope("content"))
    assert decision.action == "allow"
    assert approved == "content"


@pytest.mark.asyncio
async def test_evaluate_returns_decision_and_blocked_placeholder():
    svc = InspectionService(_BlockingInspector(), mode="block")
    decision, approved = await svc.evaluate(_envelope("content"))
    assert decision.action == "block"
    assert approved.startswith("TOOL_BLOCKED:")


# ---------------------------------------------------------------------------
# CompositeInspector strategies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_composite_any_block_blocks_when_any_blocks():
    composite = CompositeInspector(
        [NullInspector(), _BlockingInspector()], strategy="any_block"
    )
    decision = await composite.inspect(_envelope())
    assert decision.action == "block"


@pytest.mark.asyncio
async def test_composite_any_block_allows_when_none_block():
    composite = CompositeInspector(
        [NullInspector(), NullInspector()], strategy="any_block"
    )
    decision = await composite.inspect(_envelope())
    assert decision.action == "allow"


@pytest.mark.asyncio
async def test_composite_any_block_preserves_sanitize_when_no_blockers():
    composite = CompositeInspector(
        [NullInspector(), _SanitizingInspector()], strategy="any_block"
    )
    decision = await composite.inspect(_envelope())
    assert decision.action == "sanitize"


@pytest.mark.asyncio
async def test_composite_majority_requires_more_than_half():
    # 1 blocker out of 3 → not majority → allow
    composite = CompositeInspector(
        [NullInspector(), NullInspector(), _BlockingInspector()], strategy="majority"
    )
    decision = await composite.inspect(_envelope())
    assert decision.action == "allow"


@pytest.mark.asyncio
async def test_composite_majority_blocks_when_majority():
    # 2 blockers out of 3 → majority → block
    composite = CompositeInspector(
        [NullInspector(), _BlockingInspector(), _BlockingInspector()],
        strategy="majority",
    )
    decision = await composite.inspect(_envelope())
    assert decision.action == "block"


@pytest.mark.asyncio
async def test_composite_first_flag_returns_first_non_allow():
    composite = CompositeInspector(
        [_SanitizingInspector(), _BlockingInspector()], strategy="first_flag"
    )
    decision = await composite.inspect(_envelope())
    assert decision.action == "sanitize"


@pytest.mark.asyncio
async def test_composite_ignores_single_backend_failure_when_others_succeed():
    composite = CompositeInspector(
        [_ErrorInspector(), _SanitizingInspector()], strategy="any_block"
    )
    decision = await composite.inspect(_envelope())
    assert decision.action == "sanitize"


@pytest.mark.asyncio
async def test_composite_raises_when_all_backends_fail():
    composite = CompositeInspector(
        [_ErrorInspector(), _ErrorInspector()], strategy="any_block"
    )
    with pytest.raises(RuntimeError, match="backend down|content inspectors failed"):
        await composite.inspect(_envelope())


@pytest.mark.asyncio
async def test_composite_empty_returns_allow():
    composite = CompositeInspector([], strategy="any_block")
    decision = await composite.inspect(_envelope())
    assert decision.action == "allow"


# ---------------------------------------------------------------------------
# configure_content_inspection
# ---------------------------------------------------------------------------


def test_configure_content_inspection_replaces_backend():
    from src.tooling.inspection_service import INSPECTION_SERVICE

    original_inspector = INSPECTION_SERVICE._inspector
    try:
        configure_content_inspection(
            NullInspector(), mode="block", fail_policy="fail_closed"
        )
        assert INSPECTION_SERVICE.mode == "block"
        assert isinstance(INSPECTION_SERVICE._inspector, NullInspector)
    finally:
        # Restore
        INSPECTION_SERVICE.configure(
            original_inspector, mode="warn", fail_policy="fail_open"
        )
