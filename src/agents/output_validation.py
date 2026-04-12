from __future__ import annotations

import re
from numbers import Real
from typing import Any

import structlog

from src.data_block_utils import has_parseable_data_block, has_parseable_fenced_block
from src.llm_usage import extract_token_usage_breakdown

logger = structlog.get_logger(__name__)

_FORENSIC_VERDICT_PATTERN = re.compile(
    r"(?im)^\s*(?:\*\*)?\s*verdict\s*(?:\*\*)?\s*:\s*\S+"
)


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Real):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def extract_completion_tokens(response: Any) -> int:
    usage = extract_token_usage_breakdown(response)
    return usage.total_output_tokens or 0


def get_configured_output_cap(runnable: Any) -> int | None:
    for attr in (
        "_configured_max_output_tokens",
        "_configured_max_completion_tokens",
    ):
        coerced = _coerce_optional_int(getattr(runnable, attr, None))
        if coerced is not None:
            return coerced
    return None


def get_configured_api_output_cap(runnable: Any) -> int | None:
    for attr in (
        "_configured_api_output_tokens",
        "_configured_api_completion_tokens",
    ):
        coerced = _coerce_optional_int(getattr(runnable, attr, None))
        if coerced is not None:
            return coerced
    return None


def _has_parseable_forensic_block(content: str) -> bool:
    if has_parseable_fenced_block(content, "FORENSIC_DATA_BLOCK"):
        return True

    if "FORENSIC_DATA_BLOCK:" not in content:
        return False

    return "STATUS:" in content and _has_forensic_verdict(content)


def _has_forensic_verdict(content: str) -> bool:
    return bool(_FORENSIC_VERDICT_PATTERN.search(content))


def validate_required_output(agent_key: str, content: str) -> dict[str, Any]:
    checks: list[tuple[str, bool]] = []

    if agent_key == "fundamentals_analyst":
        checks.append(("parseable_data_block", has_parseable_data_block(content)))
    elif agent_key == "portfolio_manager":
        checks.extend(
            [
                ("verdict_header", "PORTFOLIO MANAGER VERDICT" in content),
                (
                    "thesis_summary",
                    "THESIS COMPLIANCE SUMMARY" in content
                    or "Hard Fail Checks:" in content,
                ),
                (
                    "execution_section",
                    "FINAL EXECUTION PARAMETERS" in content
                    or "Recommended Position Size" in content
                    or "PM_BLOCK" in content,
                ),
            ]
        )
    elif agent_key == "research_manager":
        checks.extend(
            [
                (
                    "recommendation",
                    bool(
                        re.search(
                            r"(?:INVESTMENT|FINAL)\s+RECOMMENDATION\s*:",
                            content,
                            re.IGNORECASE,
                        )
                    ),
                ),
                (
                    "supporting_section",
                    "THESIS COMPLIANCE" in content or "RISKS TO MONITOR" in content,
                ),
            ]
        )
    elif agent_key == "consultant":
        checks.extend(
            [
                ("review_header", "CONSULTANT REVIEW" in content),
                ("final_verdict", "FINAL CONSULTANT VERDICT" in content),
            ]
        )
    elif agent_key == "global_forensic_auditor":
        checks.extend(
            [
                ("forensic_block", _has_parseable_forensic_block(content)),
                ("status", "STATUS:" in content),
                ("verdict", _has_forensic_verdict(content)),
            ]
        )

    missing = [name for name, ok in checks if not ok]
    return {"ok": not missing, "checks": checks, "missing": missing}


def should_fail_closed(
    agent_key: str,
    *,
    validation: dict[str, Any],
    truncated: bool,
    content: str,
) -> bool:
    if validation["ok"]:
        return False

    if agent_key in {"portfolio_manager", "fundamentals_analyst"}:
        return True

    if agent_key == "global_forensic_auditor":
        return truncated or not validation["ok"]

    if agent_key == "consultant":
        return truncated and len(content.strip()) >= 200

    return truncated or len(content.strip()) < 200


def log_truncation_diagnostic(
    *,
    agent_key: str,
    ticker: str,
    runnable: Any,
    response: Any,
    content: str,
    trunc_info: dict[str, Any],
) -> None:
    if not trunc_info.get("truncated"):
        return

    configured_intent_cap = get_configured_output_cap(runnable)
    configured_api_cap = (
        get_configured_api_output_cap(runnable) or configured_intent_cap
    )
    usage = extract_token_usage_breakdown(response)
    completion_tokens = usage.total_output_tokens or 0
    thinking_tokens = usage.thinking_tokens
    visible_output_tokens = usage.visible_output_tokens
    intent_utilization = (
        round(visible_output_tokens / configured_intent_cap, 4)
        if configured_intent_cap and visible_output_tokens is not None
        else None
    )
    api_utilization = (
        round(completion_tokens / configured_api_cap, 4)
        if configured_api_cap and completion_tokens
        else None
    )

    marker = trunc_info.get("marker")
    explicit_or_structural = trunc_info.get("source") == "code" or (
        isinstance(marker, str) and marker.startswith("incomplete ")
    )
    near_cap = api_utilization is not None and api_utilization >= 0.90
    likely_real = explicit_or_structural or near_cap

    suggestion = None
    if likely_real:
        if near_cap and intent_utilization is not None and intent_utilization < 0.90:
            suggestion = "consider increasing reasoning reserve / API output cap"
        elif near_cap:
            suggestion = "consider increasing max output tokens for this agent"
        elif trunc_info.get("source") == "code":
            suggestion = "inspect tool/output size limits or truncation safeguards"
        else:
            suggestion = "inspect model output cap / provider response limits"

    payload = {
        "agent": agent_key,
        "ticker": ticker,
        "source": trunc_info.get("source"),
        "marker": marker,
        "confidence": trunc_info.get("confidence"),
        "output_len": len(content),
        "configured_output_cap": configured_intent_cap,
        "configured_output_intent_cap": configured_intent_cap,
        "configured_api_output_cap": configured_api_cap,
        "completion_tokens": completion_tokens,
        "completion_tokens_total": completion_tokens,
        "thinking_tokens": thinking_tokens,
        "visible_output_tokens": visible_output_tokens,
        "utilization_ratio": (
            intent_utilization if intent_utilization is not None else api_utilization
        ),
        "intent_utilization_ratio": intent_utilization,
        "api_utilization_ratio": api_utilization,
        "suggestion": suggestion,
    }

    if likely_real:
        logger.warning("agent_output_truncated", **payload)
    else:
        logger.info("agent_output_truncation_suspected", **payload)


def log_output_diagnostics(
    *,
    agent_key: str,
    ticker: str,
    runnable: Any,
    response: Any,
    content: str,
    truncated: bool,
    validation: dict[str, Any] | None,
) -> None:
    configured_intent_cap = get_configured_output_cap(runnable)
    configured_api_cap = (
        get_configured_api_output_cap(runnable) or configured_intent_cap
    )
    usage = extract_token_usage_breakdown(response)
    completion_tokens = usage.total_output_tokens or 0
    thinking_tokens = usage.thinking_tokens
    visible_output_tokens = usage.visible_output_tokens
    intent_utilization = (
        round(visible_output_tokens / configured_intent_cap, 4)
        if configured_intent_cap and visible_output_tokens is not None
        else None
    )
    api_utilization = (
        round(completion_tokens / configured_api_cap, 4)
        if configured_api_cap and completion_tokens
        else None
    )

    logger.info(
        "agent_output_diagnostics",
        agent=agent_key,
        ticker=ticker,
        configured_output_cap=configured_intent_cap,
        configured_output_intent_cap=configured_intent_cap,
        configured_api_output_cap=configured_api_cap,
        completion_tokens=completion_tokens,
        completion_tokens_total=completion_tokens,
        thinking_tokens=thinking_tokens,
        visible_output_tokens=visible_output_tokens,
        utilization_ratio=(
            intent_utilization if intent_utilization is not None else api_utilization
        ),
        intent_utilization_ratio=intent_utilization,
        api_utilization_ratio=api_utilization,
        truncated=truncated,
        required_structure_ok=validation["ok"] if validation is not None else None,
        missing_sections=validation["missing"] if validation is not None else [],
        output_len=len(content),
    )
