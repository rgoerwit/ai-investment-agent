from __future__ import annotations

import asyncio
import random
from typing import Any

import structlog

from src.config import config as settings_config

logger = structlog.get_logger(__name__)


async def invoke_with_rate_limit_handling(
    runnable, input_data: dict[str, Any], max_attempts: int = 3, context: str = "LLM"
) -> Any:
    """
    Invoke an LLM with explicit 429 and transient error handling.
    """
    quiet_mode = settings_config.quiet_mode

    for attempt in range(max_attempts):
        try:
            return await runnable.ainvoke(input_data)
        except Exception as exc:
            error_str = str(exc).lower()
            error_type = type(exc).__name__

            is_rate_limit = any(
                marker in error_str
                for marker in (
                    "429",
                    "rate limit",
                    "quota",
                    "resourceexhausted",
                    "resource exhausted",
                    "too many requests",
                )
            )
            is_transient = (
                any(
                    marker in error_str
                    for marker in (
                        "connection",
                        "timeout",
                        "timed out",
                        "unavailable",
                        "503",
                        "502",
                        "reset",
                    )
                )
                or error_str == ""
            )

            if is_rate_limit and attempt < max_attempts - 1:
                jitter = random.uniform(1, 10)
                wait_time = (60 * (attempt + 1)) + jitter
                if not quiet_mode:
                    logger.warning(
                        "rate_limit_detected",
                        context=context,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        wait_seconds=f"{wait_time:.1f}",
                        error_type=error_type,
                        error_message=str(exc)[:200],
                    )
                await asyncio.sleep(wait_time)
                continue

            if is_transient and attempt < max_attempts - 1:
                wait_time = 5 * (attempt + 1) + random.uniform(1, 3)
                if not quiet_mode:
                    logger.warning(
                        "transient_error_retry",
                        context=context,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        wait_seconds=f"{wait_time:.1f}",
                        error_type=error_type,
                    )
                await asyncio.sleep(wait_time)
                continue

            raise
