"""
Integration tests for ticker-specific memory isolation - REAL ChromaDB.
"""

import os
from unittest.mock import patch

import pytest

try:
    from tests.conftest import _REAL_GOOGLE_API_KEY
except ImportError:
    _REAL_GOOGLE_API_KEY = None

pytestmark = pytest.mark.integration


@pytest.fixture
def restore_real_env():
    """Restores real API key for integration tests."""
    if not _REAL_GOOGLE_API_KEY:
        pytest.skip(
            "Skipping integration test: No GOOGLE_API_KEY in original environment"
        )
    with patch.dict(os.environ, {"GOOGLE_API_KEY": _REAL_GOOGLE_API_KEY}):
        yield


@pytest.mark.integration
class TestRealTickerIsolation:
    """Verifies ticker-specific memory isolation with real ChromaDB."""

    @pytest.mark.asyncio
    async def test_different_tickers_use_different_collections(self, restore_real_env):
        from src.memory import cleanup_all_memories, create_memory_instances

        t1, t2 = "AAPL_ISO", "MSFT_ISO"
        try:
            m1 = create_memory_instances(t1)
            b1 = m1[f"{t1}_bull_memory"]
            if not b1.available:
                pytest.skip("Memory unavailable")
            await b1.add_situations(["AAPL situation"])
            m2 = create_memory_instances(t2)
            b2 = m2[f"{t2}_bull_memory"]
            await b2.add_situations(["MSFT situation"])
            r2 = await b2.get_relevant_memory(ticker=t2, situation_summary="valuation")
            assert r2 and "AAPL" not in r2
            r1 = await b1.get_relevant_memory(ticker=t1, situation_summary="debt")
            assert r1 and "MSFT" not in r1
        finally:
            cleanup_all_memories(days=0, ticker=t1)
            cleanup_all_memories(days=0, ticker=t2)

    @pytest.mark.asyncio
    async def test_memory_persistence_across_instances(self, restore_real_env):
        from src.memory import cleanup_all_memories, create_memory_instances

        t = "AAPL_PERSIST"
        try:
            m1 = create_memory_instances(t)
            b1 = m1[f"{t}_bull_memory"]
            if not b1.available:
                pytest.skip("Memory unavailable")
            await b1.add_situations(["AAPL strong"])
            m2 = create_memory_instances(t)
            b2 = m2[f"{t}_bull_memory"]
            r2 = await b2.get_relevant_memory(ticker=t, situation_summary="AAPL")
            assert r2 and "AAPL" in r2
        finally:
            cleanup_all_memories(days=0, ticker=t)


@pytest.mark.integration
class TestRealMemoryOperations:
    """Test actual memory operations with real ChromaDB."""

    @pytest.mark.asyncio
    async def test_add_and_query_with_real_embeddings(self, restore_real_env):
        from src.memory import cleanup_all_memories, create_memory_instances

        t = "TEST_OPS"
        try:
            m = create_memory_instances(t)[f"{t}_bull_memory"]
            if not m.available:
                pytest.skip("Memory unavailable")
            await m.add_situations(["Revenue growth 25%"])
            r = await m.get_relevant_memory(ticker=t, situation_summary="growth")
            assert r and "Revenue" in r
        finally:
            cleanup_all_memories(days=0, ticker=t)

    @pytest.mark.asyncio
    async def test_cleanup_respects_time_filter(self, restore_real_env):
        from src.memory import cleanup_all_memories, create_memory_instances

        t = "TEST_CLEAN"
        cleanup_all_memories(days=0, ticker=t)
        try:
            m = create_memory_instances(t)[f"{t}_bull_memory"]
            if not m.available:
                pytest.skip("Memory unavailable")
            await m.add_situations(["S1", "S2"])
            assert m.get_stats().get("count") == 2
            d = m.clear_old_memories(days_to_keep=30)
            assert sum(d.values()) == 0
            d = m.clear_old_memories(days_to_keep=0, ticker=t)
            assert sum(d.values()) == 2
            assert m.get_stats().get("count") == 0
        finally:
            cleanup_all_memories(days=0, ticker=t)
