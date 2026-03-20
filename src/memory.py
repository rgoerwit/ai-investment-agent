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
from src.runtime_diagnostics import classify_failure

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

    _EMBEDDING_MODEL = "gemini-embedding-001"
    _EMBEDDING_DIMENSION = 768
    _shared_embeddings: GoogleGenerativeAIEmbeddings | None = None
    _shared_embeddings_available: bool = False
    _shared_embeddings_key: tuple[str | None, str, int] | None = None
    _shared_chroma_client: Any | None = None
    _shared_chroma_key: str | None = None

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
        self.embeddings_available = False
        self.chroma_available = False

        # Check for API key via config
        api_key = config.get_google_api_key()
        if not api_key:
            logger.warning(
                "memory_disabled", reason="GOOGLE_API_KEY not set", collection=name
            )
            return

        self.embeddings = self._get_shared_embeddings(name)
        self.embeddings_available = self._shared_embeddings_available
        if self.embeddings is None:
            return

        # Initialize ChromaDB
        try:
            self.chroma_client = self._get_shared_chroma_client(name)
            if self.chroma_client is None:
                return

            # Check if collection exists with stale embedding model
            existing_collections = self.chroma_client.list_collections()
            collection_names = [
                c.name if hasattr(c, "name") else c for c in existing_collections
            ]

            if self.name in collection_names:
                existing = self.chroma_client.get_collection(name=self.name)
                existing_model = existing.metadata.get("embedding_model", "unknown")

                if existing_model != self._EMBEDDING_MODEL:
                    # Model mismatch - delete and recreate to avoid incompatible embeddings
                    logger.warning(
                        "stale_embedding_model_detected",
                        collection=self.name,
                        old_model=existing_model,
                        new_model=self._EMBEDDING_MODEL,
                        action="recreating_collection",
                    )
                    self.chroma_client.delete_collection(name=self.name)

            # Create or get collection
            self.situation_collection = self.chroma_client.get_or_create_collection(
                name=self.name,
                metadata={
                    "description": f"Financial debate memory for {name}",
                    "embedding_model": self._EMBEDDING_MODEL,
                    "embedding_dimension": self._EMBEDDING_DIMENSION,
                    "created_at": datetime.now().isoformat(),
                    "version": "2.0",
                },
            )

            self.chroma_available = True
            self.available = self.embeddings_available and self.chroma_available

            # Log collection stats
            count = self.situation_collection.count()
            logger.info(
                "chromadb_initialized",
                collection=self.name,
                persist_dir=str(config.chroma_persist_directory),
                existing_documents=count,
            )
            if not self.available:
                logger.warning(
                    "memory_degraded",
                    collection=self.name,
                    embeddings_available=self.embeddings_available,
                    chroma_available=self.chroma_available,
                )

        except Exception as e:
            logger.warning("chromadb_init_failed", error=str(e), collection=name)
            self.available = False

    @classmethod
    def _reset_shared_state_for_tests(cls) -> None:
        """Reset shared resources to keep tests isolated."""
        cls._shared_embeddings = None
        cls._shared_embeddings_available = False
        cls._shared_embeddings_key = None
        cls._shared_chroma_client = None
        cls._shared_chroma_key = None

    @classmethod
    def _get_shared_embeddings(
        cls, collection_name: str
    ) -> GoogleGenerativeAIEmbeddings | None:
        api_key = config.get_google_api_key()
        cache_key = (api_key, cls._EMBEDDING_MODEL, cls._EMBEDDING_DIMENSION)
        if cls._shared_embeddings_key != cache_key:
            cls._shared_embeddings = None
            cls._shared_embeddings_available = False
            cls._shared_embeddings_key = cache_key

        if cls._shared_embeddings is not None or cls._shared_embeddings_available:
            return cls._shared_embeddings

        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=f"models/{cls._EMBEDDING_MODEL}",
                google_api_key=api_key,
                task_type="retrieval_document",
                output_dimensionality=cls._EMBEDDING_DIMENSION,
            )
            try:
                test_embedding = embeddings.embed_query("initialization test")
                if not test_embedding or len(test_embedding) == 0:
                    raise ValueError("Embedding test returned empty result")
                cls._shared_embeddings = embeddings
                cls._shared_embeddings_available = True
                logger.info(
                    "embeddings_initialized",
                    model=cls._EMBEDDING_MODEL,
                    collection=collection_name,
                    shared=True,
                )
            except Exception as e:
                cls._shared_embeddings = embeddings
                cls._shared_embeddings_available = False
                details = classify_failure(
                    e,
                    provider="google",
                    model_name=cls._EMBEDDING_MODEL,
                    class_name=type(embeddings).__name__,
                )
                logger.warning(
                    "embeddings_healthcheck_failed",
                    collection=collection_name,
                    provider=details.provider,
                    model=cls._EMBEDDING_MODEL,
                    failure_kind=details.kind,
                    host=details.host,
                    error_type=details.error_type,
                    root_cause_type=details.root_cause_type,
                    retryable=details.retryable,
                    error_message=details.message,
                )
            return cls._shared_embeddings
        except Exception as e:
            logger.warning(
                "embeddings_init_failed", error=str(e), collection=collection_name
            )
            return None

    @classmethod
    def _get_shared_chroma_client(cls, collection_name: str) -> Any | None:
        persist_key = str(config.chroma_persist_directory)
        if cls._shared_chroma_key != persist_key:
            cls._shared_chroma_client = None
            cls._shared_chroma_key = persist_key

        if cls._shared_chroma_client is not None:
            return cls._shared_chroma_client

        try:
            import chromadb
            from chromadb.config import Settings

            cls._shared_chroma_client = chromadb.PersistentClient(
                path=persist_key,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            logger.info(
                "chromadb_client_initialized",
                collection=collection_name,
                persist_dir=persist_key,
                shared=True,
            )
            return cls._shared_chroma_client
        except Exception as e:
            logger.warning(
                "chromadb_init_failed", error=str(e), collection=collection_name
            )
            return None

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
        except Exception as exc:
            # Fallback if rate limiter not available or incompatible (e.g., in tests)
            # Catch all exceptions to handle import errors, attribute errors, type errors, etc.
            details = classify_failure(
                exc,
                provider="google",
                model_name=self._EMBEDDING_MODEL,
                class_name=(
                    type(self.embeddings).__name__
                    if self.embeddings is not None
                    else None
                ),
            )
            logger.warning(
                "embedding_rate_limiter_fallback",
                collection=self.name,
                provider=details.provider,
                model=self._EMBEDDING_MODEL,
                failure_kind=details.kind,
                host=details.host,
                error_type=details.error_type,
                root_cause_type=details.root_cause_type,
                retryable=details.retryable,
                error_message=details.message,
                fallback="direct_embed_query",
                exc_info=True,
            )
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


def create_lessons_collection() -> FinancialSituationMemory:
    """Create a global lessons_learned ChromaDB collection for cross-ticker insights."""
    return FinancialSituationMemory("lessons_learned")


def get_ticker_memory_stats(ticker: str) -> dict[str, dict[str, Any]]:
    """Return per-role memory stats without recreating ticker memory instances."""
    safe_ticker = sanitize_ticker_for_collection(ticker)
    role_map = {
        "bull_researcher": f"{safe_ticker}_bull_memory",
        "bear_researcher": f"{safe_ticker}_bear_memory",
        "research_manager": f"{safe_ticker}_invest_judge_memory",
        "trader": f"{safe_ticker}_trader_memory",
        "portfolio_manager": f"{safe_ticker}_risk_manager_memory",
    }
    unavailable = {"available": False, "count": 0}

    client = FinancialSituationMemory._get_shared_chroma_client(
        f"{safe_ticker}_stats_probe"
    )
    if client is None:
        return {role: {"name": name, **unavailable} for role, name in role_map.items()}

    try:
        existing_collections = client.list_collections()
        existing_names = {
            collection.name if hasattr(collection, "name") else collection
            for collection in existing_collections
        }
    except Exception as exc:
        logger.warning("ticker_memory_stats_list_failed", ticker=ticker, error=str(exc))
        return {role: {"name": name, **unavailable} for role, name in role_map.items()}

    stats: dict[str, dict[str, Any]] = {}
    for role, collection_name in role_map.items():
        if collection_name not in existing_names:
            stats[role] = {"name": collection_name, **unavailable}
            continue
        try:
            collection = client.get_collection(name=collection_name)
            stats[role] = {
                "available": True,
                "name": collection_name,
                "count": collection.count(),
            }
        except Exception as exc:
            logger.warning(
                "ticker_memory_stats_failed",
                ticker=ticker,
                collection=collection_name,
                error=str(exc),
            )
            stats[role] = {
                "available": False,
                "name": collection_name,
                "count": 0,
                "error": str(exc),
            }
    return stats


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


# ══════════════════════════════════════════════════════════════════════════════
# Macro Events Store
# ══════════════════════════════════════════════════════════════════════════════


from dataclasses import dataclass  # noqa: E402 — after existing imports


@dataclass
class MacroEvent:
    """Represents a portfolio-detected macro event stored in ChromaDB."""

    event_date: str  # YYYY-MM-DD (peak_anchor from CORRELATED_SELL_EVENT)
    detected_date: str  # YYYY-MM-DD
    expiry: str  # YYYY-MM-DD — event considered stale/priced-in after this date
    impact: str  # "TRANSIENT" | "STRUCTURAL" | "UNCERTAIN"
    event_type: str  # TARIFF_TRADE | LIQUIDITY_PANIC | ... | UNKNOWN
    scope: str  # "GLOBAL" | "REGIONAL" | "SECTOR"
    primary_region: str  # most-common exchange suffix, e.g. ".T", ".HK", or "GLOBAL"
    primary_sector: str  # most-common sector string, or ""
    severity: str  # "HIGH" (≥40%), "MEDIUM" (25-40%)
    correlation_pct: float
    peak_count: int
    total_held: int
    news_headline: str  # ≤120 chars
    news_detail: str  # ≤300 chars
    forced_reanalysis: bool = (
        False  # True → STRUCTURAL events that immediately invalidate
    )


MACRO_EVENTS_COLLECTION = "macro_events"


def _date_to_int(iso_date: str) -> int:
    """Convert YYYY-MM-DD to YYYYMMDD integer for ChromaDB numeric range comparisons.

    ChromaDB >= 0.6 requires $gt/$gte/$lt/$lte operands to be int or float.
    ISO strings are stored alongside these ints for human-readable reconstruction.
    """
    return int(iso_date.replace("-", ""))


def _meta_to_macro_event(meta: dict) -> "MacroEvent":
    """Deserialize a ChromaDB metadata dict back into a MacroEvent."""
    return MacroEvent(
        event_date=meta["event_date"],
        detected_date=meta.get("detected_date", ""),
        expiry=meta.get("expiry", ""),
        impact=meta.get("impact", "UNCERTAIN"),
        event_type=meta.get("event_type", "UNKNOWN"),
        scope=meta.get("scope", "GLOBAL"),
        primary_region=meta.get("primary_region", "GLOBAL"),
        primary_sector=meta.get("primary_sector", ""),
        severity=meta.get("severity", "MEDIUM"),
        correlation_pct=float(meta.get("correlation_pct", 0)),
        peak_count=int(meta.get("peak_count", 0)),
        total_held=int(meta.get("total_held", 0)),
        news_headline=meta.get("news_headline", ""),
        news_detail=meta.get("news_detail", ""),
        forced_reanalysis=bool(meta.get("forced_reanalysis", False)),
    )


class MacroEventsStore:
    """
    Global ChromaDB collection for portfolio-detected macro events.
    Uses dummy embeddings ([0.0]*768) — queries are metadata-only.
    Non-ticker-isolated (like lessons_learned in retrospective.py).
    """

    def __init__(self) -> None:
        self.available = False
        self.collection = None
        self._init()

    def _init(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(config.chroma_persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self.collection = client.get_or_create_collection(
                name=MACRO_EVENTS_COLLECTION,
                metadata={
                    "description": "Portfolio macro events",
                    "embedding_model": "dummy",
                },
            )
            self.available = True
        except Exception as e:
            logger.warning("macro_events_store_init_failed", error=str(e))

    def store_event(self, event: "MacroEvent") -> bool:
        """Store with 7-day dedup on event_date. Returns True if stored."""
        if not self.available:
            return False
        try:
            from datetime import date as _date
            from datetime import timedelta as _td

            anchor = _date.fromisoformat(event.event_date)
            window_start_ts = _date_to_int((anchor - _td(days=7)).isoformat())
            window_end_ts = _date_to_int((anchor + _td(days=7)).isoformat())
            existing = self.collection.get(
                where={
                    "$and": [
                        {"event_date_ts": {"$gte": window_start_ts}},
                        {"event_date_ts": {"$lte": window_end_ts}},
                    ]
                }
            )
            if existing and existing.get("ids"):
                logger.info("macro_event_dedup_skipped", event_date=event.event_date)
                return False

            event_id = f"macro_{event.event_date}_{event.detected_date}"
            self.collection.add(
                ids=[event_id],
                embeddings=[[0.0] * 768],
                documents=[f"{event.event_date} {event.news_headline}"],
                metadatas=[
                    {
                        "event_date": event.event_date,
                        "event_date_ts": _date_to_int(event.event_date),
                        "detected_date": event.detected_date,
                        "expiry": event.expiry,
                        "expiry_ts": _date_to_int(event.expiry),
                        "impact": event.impact,
                        "event_type": event.event_type,
                        "scope": event.scope,
                        "primary_region": event.primary_region,
                        "primary_sector": event.primary_sector,
                        "severity": event.severity,
                        "correlation_pct": float(event.correlation_pct),
                        "peak_count": int(event.peak_count),
                        "total_held": int(event.total_held),
                        "news_headline": event.news_headline[:120],
                        "news_detail": event.news_detail[:300],
                        "forced_reanalysis": event.forced_reanalysis,
                    }
                ],
            )
            logger.info(
                "macro_event_stored",
                event_date=event.event_date,
                impact=event.impact,
                scope=event.scope,
                headline=event.news_headline[:60],
            )
            return True
        except Exception as e:
            logger.warning("macro_event_store_failed", error=str(e))
            return False

    def get_active_events(
        self,
        region_filter: str | None = None,
    ) -> list["MacroEvent"]:
        """Return events where expiry > today, optionally filtered by region."""
        if not self.available:
            return []
        try:
            from datetime import date as _date

            today_ts = _date_to_int(_date.today().isoformat())
            where_clause: dict = {"expiry_ts": {"$gt": today_ts}}
            results = self.collection.get(
                where=where_clause,
                include=["metadatas"],
            )
            events: list[MacroEvent] = []
            for meta in results.get("metadatas") or []:
                if region_filter:
                    primary = meta.get("primary_region", "GLOBAL")
                    if primary != "GLOBAL" and primary != region_filter:
                        continue
                events.append(_meta_to_macro_event(meta))
            events.sort(key=lambda e: e.event_date, reverse=True)
            return events
        except Exception as e:
            logger.warning("macro_events_get_failed", error=str(e))
            return []

    def get_structural_events_since(self, since_date: str) -> list["MacroEvent"]:
        """Return STRUCTURAL events detected after since_date (for staleness check)."""
        if not self.available:
            return []
        try:
            from datetime import date as _date

            today_ts = _date_to_int(_date.today().isoformat())
            since_date_ts = _date_to_int(since_date)
            results = self.collection.get(
                where={
                    "$and": [
                        {"impact": {"$eq": "STRUCTURAL"}},
                        {"event_date_ts": {"$gte": since_date_ts}},
                        {"expiry_ts": {"$gt": today_ts}},
                    ]
                },
                include=["metadatas"],
            )
            events = [_meta_to_macro_event(m) for m in results.get("metadatas") or []]
            events.sort(key=lambda e: e.event_date, reverse=True)
            return events
        except Exception as e:
            logger.warning("macro_structural_events_get_failed", error=str(e))
            return []


def create_macro_events_store() -> MacroEventsStore:
    """Create a global MacroEventsStore instance backed by ChromaDB."""
    return MacroEventsStore()
