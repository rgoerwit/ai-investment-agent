from __future__ import annotations

import asyncio
import random
from typing import Any

import structlog

from src.config import config as settings_config
from src.runtime_diagnostics import (
    classify_failure,
    get_class_name,
    get_model_name,
    infer_provider,
)

logger = structlog.get_logger(__name__)


async def invoke_with_rate_limit_handling(
    runnable,
    input_data: dict[str, Any] | list[Any],
    max_attempts: int = 3,
    context: str = "LLM",
    provider: str | None = None,
    model_name: str | None = None,
) -> Any:
    """
    Invoke an LLM with explicit 429 and transient error handling.
    """
    quiet_mode = settings_config.quiet_mode
    resolved_model = model_name or get_model_name(runnable)
    class_name = get_class_name(runnable)
    resolved_provider = provider or infer_provider(
        model_name=resolved_model,
        class_name=class_name,
    )

    if not quiet_mode:
        logger.info(
            "llm_call_start",
            context=context,
            provider=resolved_provider,
            model=resolved_model,
            runnable_class=class_name,
            max_attempts=max_attempts,
        )

    for attempt in range(max_attempts):
        try:
            result = await runnable.ainvoke(input_data)
            if not quiet_mode:
                logger.info(
                    "llm_call_success",
                    context=context,
                    provider=resolved_provider,
                    model=resolved_model,
                    runnable_class=class_name,
                    attempt=attempt + 1,
                )
            return result
        except Exception as exc:
            details = classify_failure(
                exc,
                provider=resolved_provider,
                model_name=resolved_model,
                class_name=class_name,
            )
            try:
                from src.token_tracker import get_tracker

                get_tracker().record_failure(
                    agent_name=context,
                    provider=details.provider,
                    failure_kind=details.kind,
                    model_name=resolved_model or "",
                )
            except Exception:
                pass

            is_rate_limit = details.kind in {"rate_limit", "quota_error"}
            is_transient = details.kind in {
                "dns_resolution",
                "connect_error",
                "timeout",
                "server_error",
            }

            if is_rate_limit and attempt < max_attempts - 1:
                jitter = random.uniform(1, 10)
                wait_time = (60 * (attempt + 1)) + jitter
                if not quiet_mode:
                    logger.warning(
                        "llm_call_retry",
                        context=context,
                        provider=resolved_provider,
                        model=resolved_model,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        failure_kind=details.kind,
                        host=details.host,
                        retryable=details.retryable,
                        wait_seconds=f"{wait_time:.1f}",
                        error_type=details.error_type,
                        root_cause_type=details.root_cause_type,
                        error_message=details.message,
                    )
                await asyncio.sleep(wait_time)
                continue

            if is_transient and attempt < max_attempts - 1:
                wait_time = 5 * (attempt + 1) + random.uniform(1, 3)
                if not quiet_mode:
                    logger.warning(
                        "llm_call_retry",
                        context=context,
                        provider=resolved_provider,
                        model=resolved_model,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        failure_kind=details.kind,
                        host=details.host,
                        retryable=details.retryable,
                        wait_seconds=f"{wait_time:.1f}",
                        error_type=details.error_type,
                        root_cause_type=details.root_cause_type,
                        error_message=details.message,
                    )
                await asyncio.sleep(wait_time)
                continue

            if not quiet_mode:
                logger.error(
                    "llm_call_failed",
                    context=context,
                    provider=resolved_provider,
                    model=resolved_model,
                    runnable_class=class_name,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    failure_kind=details.kind,
                    host=details.host,
                    retryable=details.retryable,
                    error_type=details.error_type,
                    root_cause_type=details.root_cause_type,
                    error_message=details.message,
                )
            raise
