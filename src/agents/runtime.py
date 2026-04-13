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

try:
    from src.eval import get_active_capture_manager as _get_capture_manager
    from src.eval.llm_capture_meta import (
        extract_token_usage as _extract_token_usage,
    )
    from src.eval.llm_capture_meta import (
        extract_vendor_reasoning_config as _extract_vendor_reasoning_config,
    )
    from src.eval.llm_capture_meta import (
        normalize_reasoning_level as _normalize_reasoning_level,
    )
    from src.eval.serialization import normalize_for_json as _normalize_for_json
except ImportError:

    def _get_capture_manager():
        return None

    def _normalize_reasoning_level(runnable, model_name):
        return None

    def _extract_vendor_reasoning_config(runnable, provider):
        return None

    def _extract_token_usage(result):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "thinking_tokens": None,
            "total_tokens": None,
        }

    def _normalize_for_json(value):
        return value


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
            try:
                capture_manager = _get_capture_manager()
                if capture_manager is not None:
                    token_usage = _extract_token_usage(result)
                    response_metadata = getattr(result, "response_metadata", None)
                    response_model = None
                    if isinstance(response_metadata, dict):
                        response_model = response_metadata.get(
                            "model_name"
                        ) or response_metadata.get("model")
                    capture_manager.record_llm_call(
                        {
                            "status": "success",
                            "context": context,
                            "provider": resolved_provider,
                            "model": resolved_model,
                            "response_model": response_model,
                            "runnable_class": class_name,
                            "reasoning_level": _normalize_reasoning_level(
                                runnable, resolved_model
                            ),
                            "thinking_config_raw": _extract_vendor_reasoning_config(
                                runnable, resolved_provider
                            ),
                            "attempt": attempt + 1,
                            **token_usage,
                            "input": _normalize_for_json(input_data),
                            "response": _normalize_for_json(result),
                        }
                    )
            except Exception:
                pass
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
                capture_manager = _get_capture_manager()
                if capture_manager is not None:
                    capture_manager.record_llm_call(
                        {
                            "status": "failure",
                            "context": context,
                            "provider": resolved_provider,
                            "model": resolved_model,
                            "runnable_class": class_name,
                            "reasoning_level": _normalize_reasoning_level(
                                runnable, resolved_model
                            ),
                            "thinking_config_raw": _extract_vendor_reasoning_config(
                                runnable, resolved_provider
                            ),
                            "attempt": attempt + 1,
                            "input_tokens": None,
                            "output_tokens": None,
                            "thinking_tokens": None,
                            "total_tokens": None,
                            "input": _normalize_for_json(input_data),
                            "failure_kind": details.kind,
                            "retryable": details.retryable,
                            "error_type": details.error_type,
                            "root_cause_type": details.root_cause_type,
                            "host": details.host,
                            "error_message": details.message,
                        }
                    )
            except Exception:
                pass
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
