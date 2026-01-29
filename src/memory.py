"""
Long-term Memory System for Multi-Agent Trading System
Updated for LangChain 1.x and Google Gemini Embeddings (gemini-embedding-001).

UPDATED: Added ticker-specific memory isolation to prevent cross-contamination.
FIXED: ChromaDB v0.6.0 compatibility (list_collections returns strings).
UPDATED: Cleanup is now scoped to specific tickers to avoid wiping entire DB.
FIXED: get_stats() now gracefully handles deleted collections (zombie memories).
CLEANUP: Removed legacy global memory instances.

This module provides vector-based memory storage for financial debate history,
allowing agents to learn from past analyses and decisions.
"""

import re
from datetime import datetime
from typing import Any

import structlog
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import config

logger = structlog.get_logger(__name__)


class FinancialSituationMemory:
    """
    Vector memory storage for financial agent debate history.
    Uses Google's gemini-embedding-001 model with ChromaDB backend.

    Features:
    - Async embedding generation
    - Automatic retry with exponential backoff
    - Graceful degradation when unavailable
    - Metadata tagging for filtering
    - Ticker-specific isolation to prevent cross-contamination
    """

    def __init__(self, name: str):
        """
        Initialize a memory collection.

        Args:
            name: Unique identifier for this memory collection (e.g., "0005_HK_bull_memory")
        """
        self.name = name
        self.available = False
        self.situation_collection = None
        self.embeddings = None

        # Check for API key via config
        api_key = config.get_google_api_key()
        if not api_key:
            logger.warning(
                "memory_disabled", reason="GOOGLE_API_KEY not set", collection=name
            )
            return

        # Initialize Google embeddings
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
                task_type="retrieval_document",  # Optimized for semantic search
                output_dimensionality=768,  # Match existing ChromaDB collections
            )

            # Validate embeddings work with a test query (Sync call for init)
            try:
                test_embedding = self.embeddings.embed_query("initialization test")
                if not test_embedding or len(test_embedding) == 0:
                    raise ValueError("Embedding test returned empty result")
            except Exception as e:
                logger.warning(f"Embedding initialization test failed: {e}")
                # Don't fail completely, might be transient

            logger.info(
                "embeddings_initialized", model="gemini-embedding-001", collection=name
            )

        except Exception as e:
            logger.warning("embeddings_init_failed", error=str(e), collection=name)
            return

        # Initialize ChromaDB
        try:
            # NOTE: Telemetry env vars (ANONYMIZED_TELEMETRY, CHROMA_TELEMETRY_ENABLED)
            # are now set by config.py's setup_environment validator.
            import chromadb
            from chromadb.config import Settings

            CURRENT_EMBEDDING_MODEL = "gemini-embedding-001"

            # Initialize persistent client with telemetry explicitly disabled
            self.chroma_client = chromadb.PersistentClient(
                path=str(config.chroma_persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Check if collection exists with stale embedding model
            existing_collections = self.chroma_client.list_collections()
            collection_names = [
                c.name if hasattr(c, "name") else c for c in existing_collections
            ]

            if self.name in collection_names:
                existing = self.chroma_client.get_collection(name=self.name)
                existing_model = existing.metadata.get("embedding_model", "unknown")

                if existing_model != CURRENT_EMBEDDING_MODEL:
                    # Model mismatch - delete and recreate to avoid incompatible embeddings
                    logger.warning(
                        "stale_embedding_model_detected",
                        collection=self.name,
                        old_model=existing_model,
                        new_model=CURRENT_EMBEDDING_MODEL,
                        action="recreating_collection",
                    )
                    self.chroma_client.delete_collection(name=self.name)

            # Create or get collection
            self.situation_collection = self.chroma_client.get_or_create_collection(
                name=self.name,
                metadata={
                    "description": f"Financial debate memory for {name}",
                    "embedding_model": CURRENT_EMBEDDING_MODEL,
                    "embedding_dimension": 768,
                    "created_at": datetime.now().isoformat(),
                    "version": "2.0",
                },
            )

            self.available = True

            # Log collection stats
            count = self.situation_collection.count()
            logger.info(
                "chromadb_initialized",
                collection=self.name,
                persist_dir=str(config.chroma_persist_directory),
                existing_documents=count,
            )

        except Exception as e:
            logger.warning("chromadb_init_failed", error=str(e), collection=name)
            self.available = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def _get_embedding(self, text: str) -> list[float]:
        """
        Get embedding vector for text with retry logic.

        Args:
            text: Text to embed (will be truncated to 9000 chars)

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception if all retries fail
        """
        if not self.available or not self.embeddings:
            raise ValueError(f"Memory not available for {self.name}")

        # Truncate text to avoid token limits
        truncated_text = text[:9000]

        # Import rate limiter here to avoid circular dependency
        # Use rate limiter to share RPM quota with LLM calls
        try:
            from src.llms import GLOBAL_RATE_LIMITER

            async with GLOBAL_RATE_LIMITER:
                embedding = await self.embeddings.aembed_query(truncated_text)
        except Exception:
            # Fallback if rate limiter not available or incompatible (e.g., in tests)
            # Catch all exceptions to handle import errors, attribute errors, type errors, etc.
            embedding = await self.embeddings.aembed_query(truncated_text)

        if not embedding or len(embedding) == 0:
            raise ValueError("Empty embedding returned")

        return embedding

    async def add_situations(
        self, situations: list[str], metadata: list[dict[str, Any]] | None = None
    ) -> bool:
        """
        Add financial situations/debates to memory.

        Args:
            situations: List of situation descriptions or debate summaries
            metadata: Optional list of metadata dicts (one per situation)

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            logger.debug("memory_add_skipped", collection=self.name)
            return False

        if not situations:
            logger.debug("empty_situations_list", collection=self.name)
            return False

        try:
            # Generate embeddings for all situations
            embeddings = []
            for situation in situations:
                emb = await self._get_embedding(situation)
                embeddings.append(emb)

            # Prepare IDs (use timestamp + index)
            timestamp = datetime.now().isoformat()
            ids = [f"{timestamp}_{i}" for i in range(len(situations))]

            # Prepare metadata
            if metadata is None:
                metadata = [{"timestamp": timestamp} for _ in situations]
            else:
                # Ensure timestamp is in metadata
                for meta in metadata:
                    if "timestamp" not in meta:
                        meta["timestamp"] = timestamp

            # Add to collection
            self.situation_collection.add(
                ids=ids, embeddings=embeddings, documents=situations, metadatas=metadata
            )

            logger.info(
                "situations_added",
                collection=self.name,
                count=len(situations),
                has_metadata=metadata is not None,
            )

            return True

        except Exception as e:
            logger.error("add_situations_failed", collection=self.name, error=str(e))
            return False

    async def query_similar_situations(
        self,
        query_text: str,
        n_results: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query for similar past situations.

        Args:
            query_text: Search query
            n_results: Number of results to return
            metadata_filter: Optional metadata filter (e.g., {"ticker": "AAPL"})

        Returns:
            List of dicts with keys: document, metadata, distance
        """
        if not self.available:
            logger.debug("memory_query_skipped", collection=self.name)
            return []

        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query_text)

            # Query ChromaDB
            # Use metadata filter if provided, otherwise default to nothing (Chroma handles collection automatically)
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
            }

            if metadata_filter:
                query_kwargs["where"] = metadata_filter

            results = self.situation_collection.query(**query_kwargs)

            # Format results
            formatted_results = []
            if results and "documents" in results:
                for i in range(len(results["documents"][0])):
                    formatted_results.append(
                        {
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i]
                            if "metadatas" in results
                            else {},
                            "distance": results["distances"][0][i]
                            if "distances" in results
                            else 1.0,
                        }
                    )

            logger.debug(
                "memory_query_complete",
                collection=self.name,
                results_found=len(formatted_results),
            )

            return formatted_results

        except Exception as e:
            logger.error(
                "query_similar_situations_failed", collection=self.name, error=str(e)
            )
            return []

    async def get_relevant_memory(
        self, ticker: str, situation_summary: str, n_results: int = 3
    ) -> str:
        """
        Get relevant past memories for a ticker and situation.

        Args:
            ticker: Stock ticker symbol
            situation_summary: Brief description of current situation
            n_results: Number of past memories to retrieve

        Returns:
            Formatted string of relevant memories
        """
        if not self.available:
            return ""

        # Query for similar situations
        # NOTE: This method is a high-level helper.
        # Agents should use query_similar_situations directly for fine-grained control (e.g. filtering by ticker)
        query_text = f"{ticker}: {situation_summary}"
        results = await self.query_similar_situations(
            query_text=query_text, n_results=n_results
        )

        if not results:
            return f"No relevant past memories found for {ticker}."

        # Format results
        memory_text = f"Relevant past memories for {ticker}:\n\n"
        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            doc = result["document"]
            dist = result["distance"]

            memory_text += f"### Memory {i} (similarity: {1 - dist:.2%})\n"
            memory_text += f"Date: {meta.get('timestamp', 'Unknown')}\n"
            memory_text += f"Ticker: {meta.get('ticker', 'Unknown')}\n"
            memory_text += f"{doc[:500]}...\n\n"

        return memory_text

    def clear_old_memories(
        self, days_to_keep: int = 90, ticker: str | None = None
    ) -> dict[str, int]:
        """
        Remove memories older than specified days.

        UPDATED: Now supports ticker-scoped cleanup.

        Args:
            days_to_keep: Delete memories older than this many days (0 = delete ALL)
            ticker: If provided, ONLY clean collections starting with this ticker's ID.
                    If None, clean ALL collections in the database.

        Returns:
            Dict of collection_name -> documents_deleted
        """
        results = {}

        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(config.chroma_persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            collections = client.list_collections()

            # Calculate ticker prefix if provided
            target_prefix = None
            if ticker:
                target_prefix = sanitize_ticker_for_collection(ticker)
                logger.info(f"Scoping memory cleanup to ticker prefix: {target_prefix}")

            for collection_item in collections:
                try:
                    # --- FIX FOR CHROMA 0.6.0+ COMPATIBILITY ---
                    if isinstance(collection_item, str):
                        collection = client.get_collection(collection_item)
                        collection_name = collection_item
                    else:
                        collection = collection_item
                        collection_name = collection.name
                    # -------------------------------------------

                    # Filter by ticker if requested
                    if target_prefix and not collection_name.startswith(target_prefix):
                        continue

                    if days_to_keep == 0:
                        # Delete entire collection
                        count = collection.count()
                        client.delete_collection(collection_name)
                        results[collection_name] = count
                        logger.info(
                            "collection_deleted",
                            name=collection_name,
                            documents_deleted=count,
                        )
                    else:
                        # Delete old documents
                        from datetime import timedelta

                        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                        cutoff_iso = cutoff_date.isoformat()

                        all_docs = collection.get()
                        ids_to_delete = []

                        if all_docs and "metadatas" in all_docs:
                            for doc_id, metadata in zip(
                                all_docs["ids"], all_docs["metadatas"], strict=False
                            ):
                                timestamp = metadata.get("timestamp", "")
                                if timestamp and timestamp < cutoff_iso:
                                    ids_to_delete.append(doc_id)

                        if ids_to_delete:
                            collection.delete(ids=ids_to_delete)
                            results[collection_name] = len(ids_to_delete)
                            logger.info(
                                "old_documents_deleted",
                                collection=collection_name,
                                count=len(ids_to_delete),
                                days_kept=days_to_keep,
                            )
                        else:
                            results[collection_name] = 0

                except Exception as e:
                    # Try to get name for logging
                    name = getattr(collection_item, "name", str(collection_item))
                    logger.error(
                        "collection_cleanup_failed", collection=name, error=str(e)
                    )
                    results[name] = 0

        except Exception as e:
            logger.error("cleanup_all_memories_failed", error=str(e))

        return results

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about this memory collection.

        Returns:
            Dict with stats: available, count, name
        """
        if not self.available:
            return {"available": False, "name": self.name, "count": 0}

        try:
            count = self.situation_collection.count()
            return {"available": True, "name": self.name, "count": count}
        except Exception as e:
            # FIX: Gracefully handle deleted collections (zombies)
            if "does not exist" in str(e) or "Collection not found" in str(e):
                logger.debug("collection_deleted_externally", collection=self.name)
                return {
                    "available": False,
                    "name": self.name,
                    "count": 0,
                    "status": "deleted",
                }

            logger.error("get_stats_failed", collection=self.name, error=str(e))
            return {"available": False, "name": self.name, "count": 0, "error": str(e)}


def sanitize_ticker_for_collection(ticker: str) -> str:
    """
    Sanitize ticker symbol for use in ChromaDB collection names.

    ChromaDB collection names must be:
    - 3-63 characters long
    - Start and end with alphanumeric character
    - Only contain alphanumeric, underscores, or hyphens

    Args:
        ticker: Stock ticker symbol (e.g., "0005.HK", "BRK.B")

    Returns:
        Sanitized ticker for collection name (e.g., "0005_HK", "BRK_B")
    """
    # 1. Aggressively remove any characters that aren't alphanumeric, dot, hyphen, or underscore
    # This handles Unicode (™, ©) and other special chars
    clean_base = re.sub(r"[^a-zA-Z0-9._-]", "", ticker)

    # 2. Replace separators with underscores (Chroma safe)
    sanitized = clean_base.replace(".", "_").replace("-", "_")

    # 3. Ensure it starts with alphanumeric (prepend 'T_' if needed)
    if not sanitized or not sanitized[0].isalnum():
        sanitized = f"T_{sanitized}"

    # 4. Ensure length requirements (Chroma Max 63)
    # We append suffixes like "_risk_manager_memory" (20 chars).
    # So safe base length is 63 - 20 = 43 chars.
    if len(sanitized) > 40:
        sanitized = sanitized[:40]

    if len(sanitized) < 3:
        sanitized = f"{sanitized}_mem"

    return sanitized


def create_memory_instances(ticker: str) -> dict[str, FinancialSituationMemory]:
    """
    Create ticker-specific memory instances to prevent cross-contamination.

    CRITICAL: This creates separate memory collections for each ticker.
    Example: HSBC (0005.HK) gets "0005_HK_bull_memory", "0005_HK_bear_memory", etc.
             Canon (7915.T) gets "7915_T_bull_memory", "7915_T_bear_memory", etc.

    This prevents Canon's analysis from contaminating HSBC's memory and vice versa.

    Args:
        ticker: Stock ticker symbol (e.g., "0005.HK", "AAPL", "7915.T")

    Returns:
        Dict mapping memory role names to instances
    """
    # Sanitize ticker for use in collection names
    safe_ticker = sanitize_ticker_for_collection(ticker)

    memory_configs = [
        f"{safe_ticker}_bull_memory",
        f"{safe_ticker}_bear_memory",
        f"{safe_ticker}_trader_memory",
        f"{safe_ticker}_invest_judge_memory",
        f"{safe_ticker}_risk_manager_memory",
    ]

    instances = {}
    for name in memory_configs:
        try:
            instances[name] = FinancialSituationMemory(name)
            logger.info(
                "ticker_memory_created",
                ticker=ticker,
                collection_name=name,
                available=instances[name].available,
            )
        except Exception as e:
            logger.error(
                "ticker_memory_creation_failed",
                ticker=ticker,
                collection_name=name,
                error=str(e),
            )
            # Create a disabled instance
            instances[name] = FinancialSituationMemory(name)

    return instances


def cleanup_all_memories(days: int = 0, ticker: str | None = None) -> dict[str, int]:
    """
    Clean up memories from collections.

    UPDATED: Now supports ticker-scoped cleanup.

    Args:
        days: Delete memories older than this many days (0 = delete ALL)
        ticker: If provided, ONLY clean collections starting with this ticker's ID.
                If None, clean ALL collections in the database.

    Returns:
        Dict of collection_name -> documents_deleted
    """
    results = {}

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(config.chroma_persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        collections = client.list_collections()

        # Calculate ticker prefix if provided
        target_prefix = None
        if ticker:
            target_prefix = sanitize_ticker_for_collection(ticker)
            logger.info(f"Scoping memory cleanup to ticker prefix: {target_prefix}")

        for collection_item in collections:
            try:
                # --- FIX FOR CHROMA 0.6.0+ COMPATIBILITY ---
                if isinstance(collection_item, str):
                    collection = client.get_collection(collection_item)
                    collection_name = collection_item
                else:
                    collection = collection_item
                    collection_name = collection.name
                # -------------------------------------------

                # Filter by ticker if requested
                if target_prefix and not collection_name.startswith(target_prefix):
                    continue

                if days == 0:
                    # Delete entire collection
                    count = collection.count()
                    client.delete_collection(collection_name)
                    results[collection_name] = count
                    logger.info(
                        "collection_deleted",
                        name=collection_name,
                        documents_deleted=count,
                    )
                else:
                    # Delete old documents
                    from datetime import timedelta

                    cutoff_date = datetime.now() - timedelta(days=days)
                    cutoff_iso = cutoff_date.isoformat()

                    all_docs = collection.get()
                    ids_to_delete = []

                    if all_docs and "metadatas" in all_docs:
                        for doc_id, metadata in zip(
                            all_docs["ids"], all_docs["metadatas"], strict=False
                        ):
                            timestamp = metadata.get("timestamp", "")
                            if timestamp and timestamp < cutoff_iso:
                                ids_to_delete.append(doc_id)

                    if ids_to_delete:
                        collection.delete(ids=ids_to_delete)
                        results[collection_name] = len(ids_to_delete)
                        logger.info(
                            "old_documents_deleted",
                            collection=collection_name,
                            count=len(ids_to_delete),
                            days_kept=days,
                        )
                    else:
                        results[collection_name] = 0

            except Exception as e:
                # Try to get name for logging
                name = getattr(collection_item, "name", str(collection_item))
                logger.error("collection_cleanup_failed", collection=name, error=str(e))
                results[name] = 0

    except Exception as e:
        logger.error("cleanup_all_memories_failed", error=str(e))

    return results


def get_all_memory_stats() -> dict[str, dict[str, Any]]:
    """
    Get statistics for all memory collections.

    Returns:
        Dict mapping collection names to their stats
    """
    stats = {}

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(config.chroma_persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        collections = client.list_collections()

        for collection_item in collections:
            try:
                # --- FIX FOR CHROMA 0.6.0+ COMPATIBILITY ---
                if isinstance(collection_item, str):
                    collection = client.get_collection(collection_item)
                else:
                    collection = collection_item
                # -------------------------------------------

                count = collection.count()
                metadata = collection.metadata
                stats[collection.name] = {"count": count, "metadata": metadata}
            except Exception as e:
                name = getattr(collection_item, "name", str(collection_item))
                # Gracefully handle zombies in all-stats too
                if "does not exist" in str(e):
                    continue
                logger.error(
                    "get_collection_stats_failed", collection=name, error=str(e)
                )
                stats[name] = {"count": 0, "error": str(e)}

    except Exception as e:
        logger.error("get_all_stats_failed", error=str(e))

    return stats
