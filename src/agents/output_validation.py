from __future__ import annotations

import re
from numbers import Real
from typing import Any

import structlog

from src.data_block_utils import has_parseable_data_block, has_parseable_fenced_block

logger = structlog.get_logger(__name__)


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
    if response is None:
        return 0

    usage_metadata = getattr(response, "usage_metadata", None)
    candidates: list[Any] = []

    if isinstance(usage_metadata, dict):
        candidates.extend(
            [
                usage_metadata.get("output_tokens"),
                usage_metadata.get("completion_tokens"),
                usage_metadata.get("total_output_tokens"),
            ]
        )
    elif usage_metadata is not None and not callable(usage_metadata):
        candidates.extend(
            [
                getattr(usage_metadata, "output_tokens", None),
                getattr(usage_metadata, "completion_tokens", None),
                getattr(usage_metadata, "total_output_tokens", None),
            ]
        )

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            candidates.extend(
                [
                    token_usage.get("completion_tokens"),
                    token_usage.get("output_tokens"),
                ]
            )
        usage = response_metadata.get("usage")
        if isinstance(usage, dict):
            candidates.extend(
                [
                    usage.get("output_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_output_tokens"),
                ]
            )

    for candidate in candidates:
        coerced = _coerce_optional_int(candidate)
        if coerced is not None:
            return coerced

    return 0


def get_configured_output_cap(runnable: Any) -> int | None:
    for attr in (
        "_configured_max_output_tokens",
        "_configured_max_completion_tokens",
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

    required_fields = ("STATUS:", "VERDICT:")
    return all(field in content for field in required_fields)


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
                ("verdict", "VERDICT:" in content),
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
    configured_cap = get_configured_output_cap(runnable)
    completion_tokens = extract_completion_tokens(response)
    utilization = (
        round(completion_tokens / configured_cap, 4)
        if configured_cap and completion_tokens
        else None
    )

    logger.info(
        "agent_output_diagnostics",
        agent=agent_key,
        ticker=ticker,
        configured_output_cap=configured_cap,
        completion_tokens=completion_tokens,
        utilization_ratio=utilization,
        truncated=truncated,
        required_structure_ok=validation["ok"] if validation is not None else None,
        missing_sections=validation["missing"] if validation is not None else [],
        output_len=len(content),
    )
