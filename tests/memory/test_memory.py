"""
Tests for Core FinancialSituationMemory Class

This file tests the core FinancialSituationMemory class methods and behavior.

NOT covered here (see other test files):
- Ticker-specific isolation → test_memory_isolation.py
- Ticker sanitization → test_memory_isolation.py
- create_memory_instances() → test_memory_isolation.py
- cleanup_all_memories() → test_memory_isolation.py
- get_all_memory_stats() → test_memory_isolation.py
- Graph integration → test_memory_integration.py

Focus:
- Memory initialization and availability detection
- Situation storage and retrieval
- Memory cleanup (per-instance)
- Statistics (per-instance)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory import FinancialSituationMemory, cleanup_all_memories


class TestFinancialSituationMemoryInitialization:
    """Test memory initialization under various conditions."""

    def test_init_without_api_key(self):
        """Memory should gracefully handle missing API key.

        Note: We mock config.get_google_api_key() to return empty string,
        which simulates the case where GOOGLE_API_KEY is not set.
        This is cleaner than patching os.environ and reloading modules.
        """
        with patch("src.memory.config") as mock_config:
            mock_config.get_google_api_key.return_value = ""

            memory = FinancialSituationMemory("test_memory")

            assert memory.name == "test_memory"
            assert not memory.available
            assert memory.embeddings is None
            assert memory.situation_collection is None

    def test_init_with_embedding_failure(self):
        """Memory should handle embedding initialization failures."""
        with patch("src.memory.config") as mock_config:
            mock_config.get_google_api_key.return_value = "test-key"
            with patch(
                "src.memory.GoogleGenerativeAIEmbeddings",
                side_effect=Exception("API error"),
            ):
                memory = FinancialSituationMemory("test_memory")

                assert not memory.available
                assert memory.embeddings is None

    def test_init_with_chromadb_failure(self):
        """Memory should handle ChromaDB connection failures."""
        with patch("src.memory.config") as mock_config:
            mock_config.get_google_api_key.return_value = "test-key"
            with patch(
                "src.memory.GoogleGenerativeAIEmbeddings"
            ) as mock_embeddings_class:
                mock_embeddings = MagicMock()
                mock_embeddings.embed_query.return_value = [0.1] * 768
                mock_embeddings_class.return_value = mock_embeddings

                with patch(
                    "chromadb.PersistentClient",
                    side_effect=Exception("Connection failed"),
                ):
                    memory = FinancialSituationMemory("test_memory")

                    assert not memory.available
                    assert memory.situation_collection is None

    def test_successful_init(self):
        """Memory should initialize successfully with valid configuration."""
        # Mock config getter directly (os.environ patching no longer works due to SecretStr)
        with patch("src.memory.config") as mock_config:
            mock_config.get_google_api_key.return_value = "test-key"

            with patch(
                "src.memory.GoogleGenerativeAIEmbeddings"
            ) as mock_embeddings_class:
                mock_embeddings = MagicMock()
                mock_embeddings.embed_query.return_value = [0.1] * 768
                mock_embeddings_class.return_value = mock_embeddings

                with patch("chromadb.PersistentClient") as mock_client_class:
                    mock_client = MagicMock()
                    mock_collection = MagicMock()
                    mock_collection.count.return_value = 0
                    mock_client.get_or_create_collection.return_value = mock_collection
                    mock_client_class.return_value = mock_client

                    memory = FinancialSituationMemory("test_memory")

                    assert memory.name == "test_memory"
                    assert memory.available
                    assert memory.embeddings is not None
                    assert memory.situation_collection is not None


class TestSituationStorage:
    """Test adding financial situations to memory."""

    @pytest.mark.asyncio
    async def test_add_situations_unavailable(self):
        """add_situations should return False when memory unavailable."""
        memory = FinancialSituationMemory("test_memory")
        memory.available = False

        result = await memory.add_situations(["Test situation"])

        assert not result

    @pytest.mark.asyncio
    async def test_add_situations_empty_list(self):
        """add_situations should return False for empty list."""
        memory = FinancialSituationMemory("test_memory")
        memory.available = True

        result = await memory.add_situations([])

        assert not result

    @pytest.mark.asyncio
    async def test_add_situations_success(self):
        """add_situations should successfully store situations."""
        memory = FinancialSituationMemory("test_memory")
        memory.available = True
        memory._get_embedding = AsyncMock(return_value=[0.1] * 768)
        memory.situation_collection = MagicMock()

        situations = ["AAPL strong buy signal", "Market momentum positive"]
        result = await memory.add_situations(situations)

        assert result

        # Verify collection.add was called
        assert memory.situation_collection.add.called
        call_kwargs = memory.situation_collection.add.call_args[1]

        # Check that all required fields were provided
        assert "embeddings" in call_kwargs
        assert "documents" in call_kwargs
        assert "ids" in call_kwargs
        assert "metadatas" in call_kwargs

        # Check correct number of items
        assert len(call_kwargs["documents"]) == 2
        assert len(call_kwargs["embeddings"]) == 2


class TestSituationQuerying:
    """Test querying similar situations from memory."""

    @pytest.mark.asyncio
    async def test_query_unavailable(self):
        """query_similar_situations should return empty list when unavailable."""
        memory = FinancialSituationMemory("test_memory")
        memory.available = False

        results = await memory.query_similar_situations("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_query_success(self):
        """query_similar_situations should return formatted results."""
        memory = FinancialSituationMemory("test_memory")
        memory.available = True
        memory._get_embedding = AsyncMock(return_value=[0.1] * 768)
        memory.situation_collection = MagicMock()

        # Mock ChromaDB query response
        memory.situation_collection.query.return_value = {
            "documents": [["AAPL strong fundamentals", "MSFT growing revenue"]],
            "metadatas": [[{"ticker": "AAPL"}, {"ticker": "MSFT"}]],
            "distances": [[0.1, 0.2]],
        }

        results = await memory.query_similar_situations("tech stocks", n_results=2)

        assert len(results) == 2
        assert results[0]["document"] == "AAPL strong fundamentals"
        assert results[0]["metadata"]["ticker"] == "AAPL"
        assert results[0]["distance"] == 0.1
        assert results[1]["document"] == "MSFT growing revenue"


class TestGetRelevantMemory:
    """Test high-level memory retrieval for agent context."""

    @pytest.mark.asyncio
    async def test_get_relevant_memory_no_results(self):
        """get_relevant_memory should handle no results gracefully."""
        memory = FinancialSituationMemory("test_memory")
        memory.available = True
        memory.query_similar_situations = AsyncMock(return_value=[])

        result = await memory.get_relevant_memory("AAPL", "analysis")

        assert "No relevant past memories found" in result

    @pytest.mark.asyncio
    async def test_get_relevant_memory_success(self):
        """get_relevant_memory should format results for display."""
        memory = FinancialSituationMemory("test_memory")
        memory.available = True

        mock_results = [
            {
                "document": "AAPL shows strong fundamentals with growing revenue",
                "metadata": {"ticker": "AAPL", "timestamp": "2024-01-01T00:00:00"},
                "distance": 0.1,
            }
        ]
        memory.query_similar_situations = AsyncMock(return_value=mock_results)

        result = await memory.get_relevant_memory("AAPL", "fundamental analysis")

        assert "Relevant past memories for AAPL" in result
        assert "AAPL shows strong fundamentals" in result


class TestMemoryCleanup:
    """
    Test memory cleanup functionality.
    Note: cleanup_all_memories returns a dict[str, int], NOT an int.
    """

    def test_cleanup_unavailable(self):
        """clear_old_memories (instance method) should return dict when unavailable (it re-initializes client)."""
        memory = FinancialSituationMemory("test_memory")
        memory.available = False

        # Instance method now returns dict[str, int]
        # Even if available=False, it creates a new client to perform cleanup
        results = memory.clear_old_memories(days_to_keep=30)
        assert isinstance(results, dict)

    @patch("chromadb.PersistentClient")
    def test_cleanup_no_old_memories(self, mock_client_cls):
        """cleanup_all_memories should return dictionary of zeros when no old memories."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_memory"

        # FIXED: Explicitly set count return value to avoid MagicMock comparison error
        mock_collection.count.return_value = 0

        # Mock list_collections returning the collection
        mock_client.list_collections.return_value = [mock_collection]

        # Mock get() returning empty
        mock_collection.get.return_value = {"ids": [], "metadatas": []}

        mock_client_cls.return_value = mock_client

        results = cleanup_all_memories(days=0)

        # Should return a dict mapping collection names to counts
        assert isinstance(results, dict)
        # Verify the mocked collection name is in the results
        assert results.get("test_memory") == 0

    @patch("chromadb.PersistentClient")
    def test_cleanup_with_old_memories(self, mock_client_cls):
        """cleanup_all_memories should return correct counts in dictionary."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_memory"

        # When count() is called on the mock, return 5
        mock_collection.count.return_value = 5

        # Mock list_collections
        mock_client.list_collections.return_value = [mock_collection]

        mock_client_cls.return_value = mock_client

        # Run cleanup with days=0 (should delete everything)
        results = cleanup_all_memories(days=0)

        assert isinstance(results, dict)
        # Check that we got 5 deletions for test_memory
        # The previous failure asserted a dict != 1, so we must check the dict value
        assert results.get("test_memory") == 5
        mock_client.delete_collection.assert_called_with("test_memory")


class TestMemoryStats:
    """Test memory statistics retrieval."""

    def test_stats_unavailable(self):
        """get_stats should return unavailable status."""
        # Mock config to simulate missing API key
        with patch("src.memory.config") as mock_config:
            mock_config.get_google_api_key.return_value = ""

            memory = FinancialSituationMemory("test_memory")

            # Double check initialization state
            assert not memory.available

            stats = memory.get_stats()

            assert not stats["available"]
            assert stats["name"] == "test_memory"
            assert stats["count"] == 0

    def test_stats_success(self):
        """get_stats should return collection information."""
        memory = FinancialSituationMemory("test_memory")
        memory.available = True
        memory.situation_collection = MagicMock()
        memory.situation_collection.count.return_value = 42

        stats = memory.get_stats()

        assert stats["available"]
        assert stats["name"] == "test_memory"
        assert stats["count"] == 42
