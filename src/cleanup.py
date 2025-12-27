"""
Async resource cleanup utilities.

This module provides centralized cleanup for async resources (aiohttp sessions,
HTTP clients, etc.) to prevent "coroutine was never awaited" warnings and
resource leaks.

Usage:
    from src.cleanup import cleanup_async_resources

    async def main():
        try:
            # ... application logic ...
        finally:
            await cleanup_async_resources()
"""

from collections.abc import Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)

# Registry of cleanup functions
_cleanup_functions: list[Callable[[], Awaitable[None]]] = []


def register_cleanup(cleanup_fn: Callable[[], Awaitable[None]]) -> None:
    """Register an async cleanup function to be called at shutdown."""
    if cleanup_fn not in _cleanup_functions:
        _cleanup_functions.append(cleanup_fn)


async def cleanup_async_resources() -> None:
    """
    Clean up all registered async resources.

    Call this in a finally block at application shutdown to properly
    close aiohttp sessions, HTTP clients, etc.
    """
    errors = []

    for cleanup_fn in _cleanup_functions:
        try:
            await cleanup_fn()
        except Exception as e:
            errors.append((cleanup_fn.__name__, str(e)))

    # Clean up data fetcher sessions
    await _cleanup_data_fetchers()

    # Clean up Google GenAI async clients
    await _cleanup_genai_clients()

    if errors:
        for name, error in errors:
            logger.debug("cleanup_error", function=name, error=error)


async def _cleanup_data_fetchers() -> None:
    """Close all singleton data fetcher sessions."""
    # Import here to avoid circular imports
    try:
        from src.data.alpha_vantage_fetcher import get_av_fetcher

        fetcher = get_av_fetcher()
        await fetcher.close()
        logger.debug("cleanup_closed", resource="alpha_vantage_session")
    except Exception as e:
        logger.debug("cleanup_error", resource="alpha_vantage", error=str(e))

    try:
        from src.data.fmp_fetcher import get_fmp_fetcher

        fetcher = get_fmp_fetcher()
        await fetcher.close()
        logger.debug("cleanup_closed", resource="fmp_session")
    except Exception as e:
        logger.debug("cleanup_error", resource="fmp", error=str(e))

    try:
        from src.data.eodhd_fetcher import get_eodhd_fetcher

        fetcher = get_eodhd_fetcher()
        await fetcher.close()
        logger.debug("cleanup_closed", resource="eodhd_session")
    except Exception as e:
        logger.debug("cleanup_error", resource="eodhd", error=str(e))


async def _cleanup_genai_clients() -> None:
    """
    Clean up Google GenAI async clients.

    The ChatGoogleGenerativeAI class has an async_client (google.genai.client.AsyncClient)
    that needs to be properly closed to avoid "coroutine was never awaited" warnings.
    """
    try:
        from src.llms import get_all_llm_instances

        llm_instances = get_all_llm_instances()

        for name, llm in llm_instances.items():
            try:
                if hasattr(llm, "async_client") and llm.async_client is not None:
                    client = llm.async_client
                    if hasattr(client, "aclose"):
                        await client.aclose()
                        logger.debug(
                            "cleanup_closed", resource=f"genai_async_client_{name}"
                        )
            except Exception as e:
                logger.debug("cleanup_error", resource=f"genai_{name}", error=str(e))

    except ImportError:
        pass  # llms module not imported yet
    except Exception as e:
        logger.debug("cleanup_error", resource="genai_clients", error=str(e))
