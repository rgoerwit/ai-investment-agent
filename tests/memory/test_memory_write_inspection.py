from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory import FinancialSituationMemory
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.inspection_service import InspectionService
from src.tooling.inspector import InspectionDecision, InspectionEnvelope
from src.tooling.runtime import ToolExecutionService


def _memory_stub(name: str = "test_memory") -> FinancialSituationMemory:
    memory = FinancialSituationMemory.__new__(FinancialSituationMemory)
    memory.name = name
    memory.available = True
    memory.situation_collection = MagicMock()
    memory._get_embedding = AsyncMock(return_value=[0.1] * 4)
    return memory


@contextmanager
def _bind_inspection_service(inspector, *, mode: str):
    service = InspectionService()
    service.configure(inspector, mode=mode)
    services = RuntimeServices(
        tool_service=ToolExecutionService(),
        inspection_service=service,
    )
    with use_runtime_services(services):
        yield service


@pytest.mark.asyncio
async def test_memory_write_stores_benign_content():
    memory = _memory_stub()

    class AllowInspector:
        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            assert envelope.source_kind.value == "memory_write"
            return InspectionDecision(action="allow", threat_level="safe")

    with _bind_inspection_service(AllowInspector(), mode="warn"):
        result = await memory.add_situations(["benign content"], [{"ticker": "SAFE"}])
        assert result is True
        memory.situation_collection.add.assert_called_once()
        documents = memory.situation_collection.add.call_args.kwargs["documents"]
        assert documents == ["benign content"]


@pytest.mark.asyncio
async def test_memory_write_block_mode_skips_poisoned_document():
    memory = _memory_stub()

    class BlockInspector:
        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            if "poison" in envelope.content_text:
                return InspectionDecision(
                    action="block",
                    threat_level="high",
                    reason="bad doc",
                )
            return InspectionDecision(action="allow", threat_level="safe")

    with _bind_inspection_service(BlockInspector(), mode="block"):
        result = await memory.add_situations(
            ["clean doc", "poison doc"],
            [{"ticker": "CLEAN"}, {"ticker": "POISON"}],
        )
        assert result is True
        documents = memory.situation_collection.add.call_args.kwargs["documents"]
        metadatas = memory.situation_collection.add.call_args.kwargs["metadatas"]
        assert documents == ["clean doc"]
        assert [meta["ticker"] for meta in metadatas] == ["CLEAN"]


@pytest.mark.asyncio
async def test_memory_write_sanitize_mode_stores_sanitized_text():
    memory = _memory_stub()

    class SanitizeInspector:
        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            return InspectionDecision(
                action="sanitize",
                threat_level="medium",
                sanitized_content="cleaned text",
            )

    with _bind_inspection_service(SanitizeInspector(), mode="sanitize"):
        result = await memory.add_situations(["dirty text"], [{"ticker": "SAN"}])
        assert result is True
        documents = memory.situation_collection.add.call_args.kwargs["documents"]
        assert documents == ["cleaned text"]


@pytest.mark.asyncio
async def test_memory_write_returns_false_when_all_documents_blocked():
    memory = _memory_stub()

    class BlockInspector:
        async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
            return InspectionDecision(
                action="block",
                threat_level="critical",
                reason="blocked",
            )

    with _bind_inspection_service(BlockInspector(), mode="block"):
        result = await memory.add_situations(["poison only"], [{"ticker": "NOPE"}])
        assert result is False
        memory.situation_collection.add.assert_not_called()
