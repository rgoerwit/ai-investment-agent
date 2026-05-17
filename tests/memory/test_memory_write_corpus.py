from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory import FinancialSituationMemory
from src.runtime_services import RuntimeServices, use_runtime_services
from src.tooling.heuristic_inspector import HeuristicInspector
from src.tooling.inspection_service import InspectionService
from src.tooling.runtime import ToolExecutionService
from tests.helpers.injection_corpus import load_corpus

pytestmark = pytest.mark.security


def _memory_stub(name: str = "security_memory") -> FinancialSituationMemory:
    memory = FinancialSituationMemory.__new__(FinancialSituationMemory)
    memory.name = name
    memory.available = True
    memory.situation_collection = MagicMock()
    memory._get_embedding = AsyncMock(return_value=[0.1] * 4)
    return memory


@contextmanager
def _security_runtime():
    inspection_service = InspectionService(
        inspector=HeuristicInspector(),
        mode="block",
        fail_policy="fail_closed",
    )
    services = RuntimeServices(
        tool_service=ToolExecutionService(),
        inspection_service=inspection_service,
    )
    with use_runtime_services(services):
        yield


@pytest.mark.security
@pytest.mark.asyncio
async def test_memory_write_stores_benign_note_unchanged():
    memory = _memory_stub()
    note = "Revenue improved while margins remained stable."

    with _security_runtime():
        result = await memory.add_situations([note], [{"ticker": "SAFE"}])

    assert result is True
    memory.situation_collection.add.assert_called_once()
    assert memory.situation_collection.add.call_args.kwargs["documents"] == [note]


@pytest.mark.security
@pytest.mark.asyncio
async def test_memory_write_mixed_batch_stores_only_clean_documents():
    memory = _memory_stub()
    clean = "Management reduced leverage after refinancing."
    poison = load_corpus(source_kind="memory_write", expectation="must_block")[0][
        "payload"
    ]

    with _security_runtime():
        result = await memory.add_situations(
            [clean, poison],
            [{"ticker": "SAFE"}, {"ticker": "POISON"}],
        )

    assert result is True
    documents = memory.situation_collection.add.call_args.kwargs["documents"]
    metadatas = memory.situation_collection.add.call_args.kwargs["metadatas"]
    assert documents == [clean]
    assert [meta["ticker"] for meta in metadatas] == ["SAFE"]


@pytest.mark.security
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    load_corpus(source_kind="memory_write", expectation="must_block"),
    ids=lambda case: case["id"],
)
async def test_memory_write_filters_poisoned_corpus_payload(case):
    memory = _memory_stub()

    with _security_runtime():
        result = await memory.add_situations(
            [case["payload"]],
            [{"ticker": "0005.HK", "category": case["category"]}],
        )

    assert result is False
    memory.situation_collection.add.assert_not_called()
