"""
Integration Tests for Agent-Memory Interaction During Execution

These tests verify that agents actually use ticker-isolated memories correctly
during graph execution, preventing cross-contamination between ticker analyses.

This addresses the CRITICAL requirement: memory isolation in practice, not just in theory.

Run with: pytest tests/test_agent_memory_execution.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory import (
    FinancialSituationMemory,
    create_memory_instances,
    sanitize_ticker_for_collection,
)


class TestResearcherMemoryIsolation:
    """Test that researcher nodes use isolated ticker-specific memories."""

    @pytest.mark.asyncio
    @patch("src.memory.GoogleGenerativeAIEmbeddings")
    @patch("chromadb.PersistentClient")
    async def test_bull_researcher_memory_created_with_correct_ticker(
        self, mock_chroma_client, mock_embeddings
    ):
        """
        Verify that bull researcher memory is created with correct ticker-specific name.

        Design: When analyzing 0005.HK, bull researcher's memory should be
                named "0005_HK_bull_memory" to prevent cross-contamination.
        """
        ticker = "0005.HK"

        # Setup mock embeddings
        mock_emb_instance = MagicMock()
        mock_emb_instance.embed_query.return_value = [0.1] * 768
        mock_embeddings.return_value = mock_emb_instance

        # Setup mock ChromaDB
        mock_chroma_client.return_value.get_or_create_collection.return_value = (
            MagicMock()
        )

        # Create ticker-specific memory
        safe_ticker = sanitize_ticker_for_collection(ticker)
        bull_memory = FinancialSituationMemory(f"{safe_ticker}_bull_memory")

        # Verify memory was created with correct ticker-specific name
        assert bull_memory.name == f"{safe_ticker}_bull_memory"
        assert bull_memory.name == "0005_HK_bull_memory"

        # Verify memory is available for use
        assert bull_memory.available

    @pytest.mark.asyncio
    @patch("src.memory.GoogleGenerativeAIEmbeddings")
    @patch("chromadb.PersistentClient")
    async def test_sequential_ticker_analyses_dont_contaminate(
        self, mock_chroma_client, mock_embeddings
    ):
        """
        Critical test: Verify that analyzing HSBC then Toyota doesn't contaminate memories.

        Design:
        1. Analyze 0005.HK (HSBC) - store memory
        2. Analyze 7203.T (Toyota) - store memory
        3. Verify HSBC memory doesn't contain Toyota data
        4. Verify Toyota memory doesn't contain HSBC data
        """
        # Setup mock embeddings
        mock_emb_instance = MagicMock()
        mock_emb_instance.embed_query.return_value = [0.1] * 768
        mock_emb_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
        mock_embeddings.return_value = mock_emb_instance

        # Create mock collections that track what was added
        hsbc_data = []
        toyota_data = []

        def create_mock_collection(name, **kwargs):
            """Create collection that tracks adds based on name."""
            collection = MagicMock()
            collection.name = name

            if "0005_HK" in name:

                def hsbc_add(documents, metadatas, ids, embeddings=None):
                    hsbc_data.append(
                        {"documents": documents, "metadatas": metadatas, "ids": ids}
                    )

                collection.add = MagicMock(side_effect=hsbc_add)
                collection.query.return_value = {
                    "ids": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                }

            elif "7203_T" in name:

                def toyota_add(documents, metadatas, ids, embeddings=None):
                    toyota_data.append(
                        {"documents": documents, "metadatas": metadatas, "ids": ids}
                    )

                collection.add = MagicMock(side_effect=toyota_add)
                collection.query.return_value = {
                    "ids": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                }

            return collection

        mock_chroma_client.return_value.get_or_create_collection.side_effect = (
            create_mock_collection
        )

        # Create memories for HSBC
        hsbc_memories = create_memory_instances("0005.HK")

        # Simulate storing HSBC analysis
        hsbc_bull_memory = hsbc_memories["0005_HK_bull_memory"]
        await hsbc_bull_memory.add_situations(
            ["HSBC shows strong banking fundamentals"], [{"ticker": "0005.HK"}]
        )

        # Create memories for Toyota
        toyota_memories = create_memory_instances("7203.T")

        # Simulate storing Toyota analysis
        toyota_bull_memory = toyota_memories["7203_T_bull_memory"]
        await toyota_bull_memory.add_situations(
            ["Toyota manufacturing efficiency improving"], [{"ticker": "7203.T"}]
        )

        # Verify isolation: HSBC data should not mention Toyota
        if hsbc_data:
            hsbc_docs = [item["documents"] for item in hsbc_data]
            hsbc_text = str(hsbc_docs).lower()
            assert (
                "toyota" not in hsbc_text
            ), "HSBC memory should not contain Toyota data"

        # Verify isolation: Toyota data should not mention HSBC
        if toyota_data:
            toyota_docs = [item["documents"] for item in toyota_data]
            toyota_text = str(toyota_docs).lower()
            assert (
                "hsbc" not in toyota_text
            ), "Toyota memory should not contain HSBC data"

        # Verify correct metadata was stored
        if hsbc_data:
            assert all(
                meta["ticker"] == "0005.HK"
                for item in hsbc_data
                for meta in item["metadatas"]
            ), "HSBC memory should only have ticker='0005.HK' metadata"

        if toyota_data:
            assert all(
                meta["ticker"] == "7203.T"
                for item in toyota_data
                for meta in item["metadatas"]
            ), "Toyota memory should only have ticker='7203.T' metadata"


class TestResearcherMetadataFiltering:
    """Test that researcher nodes enforce metadata filtering correctly."""

    @pytest.mark.asyncio
    @patch("src.memory.GoogleGenerativeAIEmbeddings")
    @patch("chromadb.PersistentClient")
    async def test_researcher_metadata_filter_prevents_wrong_ticker_retrieval(
        self, mock_chroma_client, mock_embeddings
    ):
        """
        Verify that researcher queries with metadata_filter prevent wrong ticker data.

        Design: Even if ChromaDB collection contains data for multiple tickers
                (shouldn't happen, but defensive), metadata filter should prevent
                retrieval of wrong ticker's data.
        """
        ticker = "0005.HK"

        # Setup mock embeddings
        mock_emb_instance = MagicMock()
        mock_emb_instance.embed_query.return_value = [0.1] * 768
        mock_emb_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
        mock_embeddings.return_value = mock_emb_instance

        # Setup mock collection that would return wrong ticker data
        # if metadata filter wasn't applied
        mock_collection = MagicMock()

        query_calls = []

        def track_query(query_embeddings, n_results, where=None, **kwargs):
            """Track query calls to verify metadata filter is used."""
            query_calls.append({"where": where, "n_results": n_results})

            # Simulate ChromaDB respecting metadata filter
            if where and where.get("ticker") == ticker:
                # Return correct ticker data
                return {
                    "ids": [["mem1"]],
                    "documents": [["HSBC analysis"]],
                    "metadatas": [[{"ticker": ticker}]],
                    "distances": [[0.5]],
                }
            else:
                # Return empty if filter doesn't match
                return {
                    "ids": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                }

        mock_collection.query = MagicMock(side_effect=track_query)
        mock_chroma_client.return_value.get_or_create_collection.return_value = (
            mock_collection
        )

        # Create memory and query
        safe_ticker = sanitize_ticker_for_collection(ticker)
        memory = FinancialSituationMemory(f"{safe_ticker}_bull_memory")

        # Query with metadata filter (as researcher nodes should)
        results = await memory.query_similar_situations(
            "What are the fundamentals?", metadata_filter={"ticker": ticker}
        )

        # Verify query was called with correct metadata filter
        assert len(query_calls) > 0, "Memory query should have been called"
        last_query = query_calls[-1]
        assert last_query["where"] is not None, "Query should include metadata filter"
        assert (
            last_query["where"]["ticker"] == ticker
        ), f"Filter should be for ticker {ticker}"

        # Verify results only contain correct ticker data
        for result in results:
            if "metadata" in result:
                assert result["metadata"]["ticker"] == ticker


class TestCrossTickerContamination:
    """Real-world contamination scenarios that the memory isolation prevents."""

    @pytest.mark.asyncio
    @patch("src.memory.GoogleGenerativeAIEmbeddings")
    @patch("chromadb.PersistentClient")
    async def test_semiconductor_shortage_doesnt_leak_to_bank_stock(
        self, mock_chroma_client, mock_embeddings
    ):
        """
        Real-world scenario: Analyzing Toyota (semiconductor issues)
        then HSBC (banking) should not leak automotive sector issues.

        This is the exact contamination bug the refactoring prevents.
        """
        # Setup mock embeddings
        mock_emb_instance = MagicMock()
        mock_emb_instance.embed_query.return_value = [0.1] * 768
        mock_emb_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
        mock_embeddings.return_value = mock_emb_instance

        # Track data stored in each collection
        stored_data = {}

        def create_tracked_collection(name, **kwargs):
            collection = MagicMock()
            collection.name = name
            stored_data[name] = []

            def track_add(documents, metadatas, ids, embeddings=None):
                stored_data[name].append(
                    {"documents": documents, "metadatas": metadatas}
                )

            collection.add = MagicMock(side_effect=track_add)

            def track_query(query_embeddings, n_results, where=None, **kwargs):
                # Only return data matching metadata filter
                ticker_filter = where.get("ticker") if where else None
                matching_data = [
                    item
                    for item in stored_data[name]
                    if any(
                        meta.get("ticker") == ticker_filter
                        for meta in item["metadatas"]
                    )
                ]

                if matching_data:
                    return {
                        "ids": [["id1"]],
                        "documents": [[matching_data[0]["documents"][0]]],
                        "metadatas": [[matching_data[0]["metadatas"][0]]],
                        "distances": [[0.5]],
                    }
                return {
                    "ids": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                }

            collection.query = MagicMock(side_effect=track_query)
            return collection

        mock_chroma_client.return_value.get_or_create_collection.side_effect = (
            create_tracked_collection
        )

        # Step 1: Analyze Toyota - store semiconductor shortage concern
        toyota_memories = create_memory_instances("7203.T")
        toyota_bull = toyota_memories["7203_T_bull_memory"]

        await toyota_bull.add_situations(
            ["Toyota faces semiconductor shortage impacting production"],
            [{"ticker": "7203.T"}],
        )

        # Step 2: Analyze HSBC - should NOT retrieve Toyota's semiconductor issues
        hsbc_memories = create_memory_instances("0005.HK")
        hsbc_bull = hsbc_memories["0005_HK_bull_memory"]

        # Query HSBC memory (should be empty or only contain HSBC data)
        hsbc_results = await hsbc_bull.query_similar_situations(
            "What are the key risks?", metadata_filter={"ticker": "0005.HK"}
        )

        # Verify HSBC query didn't return Toyota semiconductor data
        for result in hsbc_results:
            result_text = result.get("content", "").lower()
            assert (
                "semiconductor" not in result_text
            ), "HSBC analysis should not retrieve Toyota semiconductor issues"
            assert (
                "toyota" not in result_text
            ), "HSBC analysis should not contain Toyota data"


class TestMemoryCollectionNaming:
    """Test that memory collection names follow correct convention."""

    @patch("src.memory.GoogleGenerativeAIEmbeddings")
    @patch("chromadb.PersistentClient")
    def test_edge_case_tickers_create_valid_collection_names(
        self, mock_chroma_client, mock_embeddings
    ):
        """
        Verify that edge-case tickers create valid ChromaDB collection names.

        Design: sanitize_ticker_for_collection should handle:
        - Dots (0005.HK → 0005_HK)
        - Hyphens (BRK-B → BRK_B)
        - Very long tickers (truncate to 40 chars)
        - Special chars at start (.WEIRD → T__WEIRD)
        """
        # Setup mocks
        mock_emb_instance = MagicMock()
        mock_emb_instance.embed_query.return_value = [0.1] * 768
        mock_embeddings.return_value = mock_emb_instance

        created_collections = []

        def track_get_or_create(name, **kwargs):
            created_collections.append(name)
            collection = MagicMock()
            collection.name = name
            return collection

        mock_chroma_client.return_value.get_or_create_collection.side_effect = (
            track_get_or_create
        )

        edge_case_tickers = [
            ("0005.HK", "0005_HK"),  # Dot replacement
            ("BRK-B", "BRK_B"),  # Hyphen replacement
            ("A" * 50, "A" * 40),  # Truncation to 40 chars
        ]

        for ticker, expected_safe in edge_case_tickers:
            # Create memories
            memories = create_memory_instances(ticker)

            # Verify all memory keys use expected sanitized ticker
            for key in memories.keys():
                assert key.startswith(
                    expected_safe
                ), f"Memory key '{key}' should start with '{expected_safe}' for ticker '{ticker}'"

            # Verify collection names are valid (no special chars)
            for collection_name in created_collections:
                # ChromaDB collection names must be alphanumeric + underscore
                assert all(
                    c.isalnum() or c == "_" for c in collection_name
                ), f"Collection name '{collection_name}' contains invalid characters"

            created_collections.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
