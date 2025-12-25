"""Edge case tests for memory system - embeddings, persistence, corruption."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

from src.memory import FinancialSituationMemory, create_memory_instances, cleanup_all_memories


@pytest.mark.asyncio
class TestEmbeddingEdgeCases:
    """Test edge cases in embedding generation and vector operations."""
    
    async def test_embedding_dimension_mismatch(self):
        """Test handling of mismatched embedding dimensions."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    # Setup mock instance
                    mock_emb_instance = MagicMock()
                    # Init check needs sync return (valid)
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    # Runtime call needs async return (invalid dimension)
                    mock_emb_instance.aembed_query = AsyncMock(return_value=[0.1] * 512)
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    # CRITICAL: Simulate Chroma raising error on add
                    mock_collection.add.side_effect = ValueError("Dimensionality mismatch")
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_dim_mismatch")
                    
                    # Should catch the error and return False (graceful degradation)
                    result = await memory.add_situations(
                        ["Test situation"],
                        [{"ticker": "TEST"}]
                    )
                    assert result is False
    
    async def test_embedding_nan_values(self):
        """Test handling of NaN in embedding vectors."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    # Init check valid
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    
                    # Runtime NaN
                    embedding_with_nan = [0.1] * 768
                    embedding_with_nan[100] = float('nan')
                    mock_emb_instance.aembed_query = AsyncMock(return_value=embedding_with_nan)
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_nan")
                    
                    result = await memory.add_situations(
                        ["Test with NaN embedding"],
                        [{"ticker": "NAN"}]
                    )
                    
                    assert result is not None
    
    async def test_embedding_infinity_values(self):
        """Test handling of infinity in embeddings."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    # Init check valid
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    
                    # Runtime Inf
                    embedding_with_inf = [0.1] * 768
                    embedding_with_inf[50] = float('inf')
                    mock_emb_instance.aembed_query = AsyncMock(return_value=embedding_with_inf)
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    # Simulate ChromaDB rejecting Infinite values
                    mock_collection.add.side_effect = ValueError("Infinity not allowed")
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_inf")
                    
                    result = await memory.add_situations(
                        ["Test with inf embedding"],
                        [{"ticker": "INF"}]
                    )
                    
                    assert result is False
    
    async def test_embedding_all_zeros(self):
        """Test handling of zero-vector embeddings."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    # Init check valid (non-zero for init to pass)
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    # Runtime zero vector
                    mock_emb_instance.aembed_query = AsyncMock(return_value=[0.0] * 768)
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_zeros")
                    
                    result = await memory.add_situations(
                        [""],
                        [{"ticker": "EMPTY"}]
                    )
                    
                    assert result is True
    
    async def test_embedding_api_rate_limit(self):
        """Test handling of embedding API rate limit."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    # Init check valid
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    # Runtime Error
                    mock_emb_instance.aembed_query = AsyncMock(side_effect=Exception("429 Rate Limit"))
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_ratelimit")
                    
                    result = await memory.add_situations(
                        ["Test situation"],
                        [{"ticker": "RATE"}]
                    )
                    
                    assert result is False


@pytest.mark.asyncio
class TestMemoryPersistenceEdgeCases:
    """Test edge cases in memory persistence and corruption."""
    
    async def test_corrupted_metadata(self):
        """Test handling of corrupted metadata in stored memories."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    mock_emb_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_collection.query.return_value = {
                        'ids': [['mem1']],
                        'documents': [['Test doc']],
                        'metadatas': [[None]],  # Corrupted metadata
                        'distances': [[0.5]]
                    }
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_corrupt_meta")
                    results = await memory.query_similar_situations("test query")
                    assert isinstance(results, list)
    
    async def test_duplicate_ids(self):
        """Test handling of duplicate memory IDs."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    mock_emb_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_dup_ids")
                    
                    await memory.add_situations(["S1"], [{"ticker": "T"}])
                    await memory.add_situations(["S2"], [{"ticker": "T"}])
                    
                    assert mock_collection.add.call_count == 2
    
    async def test_very_large_document(self):
        """Test handling of extremely large document text."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    mock_emb_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_large_doc")
                    large_doc = "A" * 1000000
                    
                    result = await memory.add_situations(
                        [large_doc],
                        [{"ticker": "LARGE"}]
                    )
                    assert result is True
    
    async def test_empty_situations_list(self):
        """Test handling of empty situations list."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_empty")
                    result = await memory.add_situations([], [])
                    assert result is False


@pytest.mark.asyncio
class TestMemoryIsolationEdgeCases:
    """Test edge cases in ticker-specific memory isolation."""
    
    async def test_ticker_with_special_chars(self):
        """Test ticker sanitization with special characters."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection

                    # This creates instances, which triggers __init__ -> embeds test query
                    memories = create_memory_instances("TEST™.HK©")
                    
                    collection_names = [m.name for m in memories.values()]
                    
                    # With new strict sanitization, ™ and © should be gone
                    assert all('™' not in k for k in collection_names)
                    assert all('©' not in k for k in collection_names)
    
    async def test_very_long_ticker(self):
        """Test handling of excessively long ticker symbols."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    mock_emb.return_value = mock_emb_instance

                    mock_collection = MagicMock()
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection

                    long_ticker = "A" * 200
                    memories = create_memory_instances(long_ticker)

                    # New logic limits base ticker to 40 chars to make room for suffixes
                    # Total length should be well within Chroma's 63 char limit
                    collection_names = [m.name for m in memories.values()]
                    assert all(len(k) <= 63 for k in collection_names)
    
    async def test_ticker_with_unicode(self):
        """Test ticker with unicode characters."""
        # Mock config instead of os.environ (Pydantic Settings pattern)
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    mock_emb.return_value = mock_emb_instance

                    mock_collection = MagicMock()
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection

                    unicode_ticker = "测试.HK"
                    memories = create_memory_instances(unicode_ticker)
                    assert isinstance(memories, dict)

                    # Check names are safe (no unicode)
                    names = [m.name for m in memories.values()]
                    assert all(n.isascii() for n in names)
    
    async def test_cleanup_with_no_memories(self):
        """Test cleanup when no memories exist."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_client.return_value.list_collections.return_value = []
            result = cleanup_all_memories(days=0)
            assert result == {}
    
    async def test_cleanup_with_locked_collection(self):
        """Test cleanup when collection is locked/in-use."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_collection.name = "locked_collection"
            mock_client.return_value.list_collections.return_value = [mock_collection]
            mock_client.return_value.delete_collection.side_effect = Exception("Collection locked")
            
            result = cleanup_all_memories(days=0)
            assert "locked_collection" in result
            assert result["locked_collection"] == 0
    
    async def test_query_with_empty_string(self):
        """Test querying with empty string."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    # Init check valid
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    # Async query call
                    mock_emb_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_collection.query.return_value = {
                        'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]
                    }
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_empty_query")
                    results = await memory.query_similar_situations("")
                    assert isinstance(results, list)
    
    async def test_query_with_very_long_text(self):
        """Test querying with extremely long query text."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    mock_emb_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_collection.query.return_value = {'documents': [[]]}
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_long_query")
                    long_query = "A" * 100000
                    results = await memory.query_similar_situations(long_query)
                    assert isinstance(results, list)
    
    async def test_query_with_invalid_filter(self):
        """Test querying with malformed metadata filter."""
        with patch('src.memory.config') as mock_config:
            mock_config.get_google_api_key.return_value = 'test-api-key'
            with patch('src.memory.GoogleGenerativeAIEmbeddings') as mock_emb:
                with patch('chromadb.PersistentClient') as mock_client:
                    mock_emb_instance = MagicMock()
                    mock_emb_instance.embed_query.return_value = [0.1] * 768
                    mock_emb_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
                    mock_emb.return_value = mock_emb_instance
                    
                    mock_collection = MagicMock()
                    mock_client.return_value.get_or_create_collection.return_value = mock_collection
                    
                    memory = FinancialSituationMemory("test_bad_filter")
                    mock_collection.query.side_effect = Exception("Invalid filter")
                    
                    results = await memory.query_similar_situations(
                        "test",
                        metadata_filter={"invalid": {"nested": "bad"}}
                    )
                    assert results == []