from __future__ import annotations

import re
from numbers import Real
from typing import Any

import structlog

from src.data_block_utils import (
    extract_block_field,
    extract_block_field_from_text_raw,
    extract_last_data_block,
    extract_last_fenced_block,
    has_parseable_data_block,
    has_parseable_fenced_block,
    unfenced_label,
)
from src.earnings_baseline import (
    GUIDANCE_BRIDGE_STATUSES,
    GUIDANCE_COVERAGE_STATUSES,
    REQUIRED_GUIDANCE_CONTRACT_ENUMS,
    REQUIRED_GUIDANCE_CONTRACT_FIELDS,
    canonical_enum,
    canonical_guidance_enum,
)
from src.llm_usage import extract_token_usage_breakdown
from src.text_patterns import URL_RE

logger = structlog.get_logger(__name__)

_FORENSIC_VERDICT_PATTERN = re.compile(
    r"(?im)^\s*(?:\*\*)?\s*verdict\s*(?:\*\*)?\s*:\s*\S+"
)


def _has_valid_latest_results_block(content: str) -> bool:
    """Require the complete latest-results contract, including explicit N/A fields."""
    from src.agents.foreign_language_evidence import LATEST_RESULTS_SOURCE_FIELDS

    block = extract_last_fenced_block(content, "LATEST_RESULTS")
    if not block:
        return False

    coverage = canonical_enum(
        extract_block_field_from_text_raw(
            block,
            "LATEST_RESULTS_COVERAGE_STATUS",
        )
    )
    if coverage not in {"FOUND", "NOT_FOUND", "SEARCH_FAILED"}:
        return False

    required_fields = (
        "LATEST_RESULTS_COVERAGE_STATUS",
        *LATEST_RESULTS_SOURCE_FIELDS,
    )
    if any(
        extract_block_field_from_text_raw(block, field) is None
        for field in required_fields
    ):
        return False

    if coverage != "FOUND":
        return True

    period = extract_block_field_from_text_raw(block, "LATEST_RESULTS_PERIOD")
    period_end = extract_block_field_from_text_raw(
        block,
        "LATEST_RESULTS_PERIOD_END",
    )
    source_url = extract_block_field_from_text_raw(
        block,
        "LATEST_RESULTS_SOURCE_URL",
    )
    if not (
        period
        and period.upper() not in {"N/A", "NA", "NONE", "UNKNOWN"}
        and period_end
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", period_end)
        and source_url
        and URL_RE.fullmatch(source_url)
    ):
        return False
    return True


def _has_valid_management_guidance_block(content: str) -> bool:
    """Validate the minimum evidence contract for forward-guidance research."""
    if not has_parseable_fenced_block(content, "MANAGEMENT_GUIDANCE"):
        return False

    coverage_status = canonical_enum(
        extract_block_field(content, "MANAGEMENT_GUIDANCE", "COVERAGE_STATUS")
    )
    if coverage_status not in GUIDANCE_COVERAGE_STATUSES:
        return False

    searches_completed = extract_block_field(
        content, "MANAGEMENT_GUIDANCE", "SEARCHES_COMPLETED"
    )
    search_provenance = extract_block_field(
        content, "MANAGEMENT_GUIDANCE", "SEARCH_PROVENANCE"
    )
    if search_provenance != "CODE_OWNED_PREFLIGHT":
        return False

    if coverage_status == "FOUND":
        source_date = extract_block_field(content, "MANAGEMENT_GUIDANCE", "SOURCE_DATE")
        source_url = extract_block_field(content, "MANAGEMENT_GUIDANCE", "SOURCE_URL")
        direction = canonical_enum(
            extract_block_field(
                content, "MANAGEMENT_GUIDANCE", "OPERATING_VS_NET_DIRECTION"
            )
        )
        material_driver = canonical_enum(
            extract_block_field(
                content, "MANAGEMENT_GUIDANCE", "MATERIAL_NONOPERATING_DRIVER"
            )
        )
        driver_type = canonical_enum(
            extract_block_field(content, "MANAGEMENT_GUIDANCE", "DRIVER_TYPE")
        )
        persistence = canonical_enum(
            extract_block_field(content, "MANAGEMENT_GUIDANCE", "DRIVER_PERSISTENCE")
        )
        baseline_status = canonical_enum(
            extract_block_field(
                content, "MANAGEMENT_GUIDANCE", "EARNINGS_BASELINE_STATUS"
            )
        )
        bridge_status = canonical_enum(
            extract_block_field(
                content, "MANAGEMENT_GUIDANCE", "GUIDANCE_BRIDGE_STATUS"
            )
        )
        if not (
            source_date
            and source_url
            and URL_RE.fullmatch(source_url)
            and direction
            and bridge_status in GUIDANCE_BRIDGE_STATUSES
        ):
            return False
        if material_driver == "YES" and driver_type in {None, "NONE", "UNKNOWN"}:
            return False
        if direction == "OP_UP_NET_DOWN":
            if bridge_status == "RECONCILED" and (
                material_driver != "YES" or driver_type in {None, "NONE", "UNKNOWN"}
            ):
                return False
            if bridge_status == "UNRESOLVED" and baseline_status == "DURABLE":
                return False
        if (
            driver_type == "TAX_CREDIT"
            and persistence in {"ONE_TIME", "EXPIRING"}
            and baseline_status == "DURABLE"
        ):
            return False
        return True

    return bool(searches_completed and searches_completed.upper() not in {"N/A", "NA"})


def _promoted_management_guidance_issue(content: str) -> str | None:
    """Describe the first invalid Senior guidance-contract field, if any."""
    coverage_status = canonical_enum(
        extract_block_field(content, "DATA_BLOCK", "GUIDANCE_COVERAGE_STATUS")
    )
    data_block = extract_last_data_block(content)
    material_driver = canonical_guidance_enum(
        "MATERIAL_NONOPERATING_DRIVER",
        extract_block_field_from_text_raw(
            data_block,
            "MATERIAL_NONOPERATING_DRIVER",
        ),
    )
    baseline_status = canonical_guidance_enum(
        "EARNINGS_BASELINE_STATUS",
        extract_block_field_from_text_raw(
            data_block,
            "EARNINGS_BASELINE_STATUS",
        ),
    )
    # Read this field raw: its allowed set includes the literal "N/A", but the
    # normalized reader (extract_block_field) strips null tokens to None, which
    # canonical_enum then renders as "" — an unobservable, self-contradictory
    # rejection of a contractually-valid N/A. Raw-read preserves the token; a
    # genuinely absent field still yields None -> "" and still fails.
    normalized_available = canonical_enum(
        extract_block_field_from_text_raw(
            data_block,
            "NORMALIZED_EARNINGS_AVAILABLE",
        )
    )
    bridge_status = canonical_enum(
        extract_block_field(content, "DATA_BLOCK", "GUIDANCE_BRIDGE_STATUS")
    )
    # Each field is read with the reader its own contract requires (raw vs
    # normalized); the required set and its enums are canonical in
    # src/earnings_baseline.py so a field with no code-owned producer cannot be
    # added here unnoticed.
    observed = {
        "GUIDANCE_COVERAGE_STATUS": coverage_status,
        "MATERIAL_NONOPERATING_DRIVER": material_driver,
        "EARNINGS_BASELINE_STATUS": baseline_status,
        "NORMALIZED_EARNINGS_AVAILABLE": normalized_available,
        "GUIDANCE_BRIDGE_STATUS": bridge_status,
    }
    for field in REQUIRED_GUIDANCE_CONTRACT_FIELDS:
        actual = observed[field]
        allowed = REQUIRED_GUIDANCE_CONTRACT_ENUMS[field]
        if actual not in allowed:
            expected = ", ".join(sorted(allowed))
            return f"{field}={actual or '<missing>'}; expected one of: {expected}"

    if coverage_status != "FOUND":
        return None

    source_url = extract_block_field(content, "DATA_BLOCK", "GUIDANCE_SOURCE_URL")
    direction = canonical_enum(
        extract_block_field(content, "DATA_BLOCK", "OPERATING_VS_NET_DIRECTION")
    )
    if not source_url or not URL_RE.fullmatch(source_url):
        return (
            f"GUIDANCE_SOURCE_URL={source_url or '<missing>'}; expected an HTTP(S) URL"
        )
    if not direction:
        return "OPERATING_VS_NET_DIRECTION=<missing>; expected an explicit token"
    if direction == "OP_UP_NET_DOWN" and bridge_status == "UNRESOLVED":
        if baseline_status == "DURABLE":
            return (
                "EARNINGS_BASELINE_STATUS=DURABLE conflicts with "
                "OPERATING_VS_NET_DIRECTION=OP_UP_NET_DOWN and "
                "GUIDANCE_BRIDGE_STATUS=UNRESOLVED"
            )
    return None


def _has_promoted_management_guidance(content: str) -> bool:
    """Require Senior Fundamentals to preserve the evidence/baseline outcome."""
    return _promoted_management_guidance_issue(content) is None


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Real):
        return int(float(value))
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

    if unfenced_label("FORENSIC_DATA_BLOCK") not in content:
        return False

    return "STATUS:" in content and _has_forensic_verdict(content)


def _has_forensic_verdict(content: str) -> bool:
    return bool(_FORENSIC_VERDICT_PATTERN.search(content))


def validate_required_output(agent_key: str, content: str) -> dict[str, Any]:
    checks: list[tuple[str, bool]] = []
    issues: dict[str, str] = {}

    if agent_key == "foreign_language_analyst":
        checks.extend(
            [
                (
                    "management_guidance_block",
                    _has_valid_management_guidance_block(content),
                ),
                ("latest_results_block", _has_valid_latest_results_block(content)),
            ]
        )
    elif agent_key == "fundamentals_analyst":
        guidance_issue = _promoted_management_guidance_issue(content)
        checks.extend(
            [
                ("parseable_data_block", has_parseable_data_block(content)),
                (
                    "promoted_management_guidance",
                    guidance_issue is None,
                ),
            ]
        )
        if guidance_issue is not None:
            issues["promoted_management_guidance"] = guidance_issue
    elif agent_key == "portfolio_manager":
        pm_block = extract_last_fenced_block(content, "PM_BLOCK")
        checks.extend(
            [
                ("verdict_header", "PORTFOLIO MANAGER VERDICT" in content),
                (
                    "thesis_summary",
                    "THESIS COMPLIANCE SUMMARY" in content
                    or "Hard Fail Checks:" in content,
                ),
                (
                    "position_section",
                    "FINAL POSITION PARAMETERS" in content
                    or "FINAL EXECUTION PARAMETERS" in content  # legacy output
                    or "Recommended Position Size" in content
                    or "PM_BLOCK" in content,
                ),
                (
                    "decision_facts",
                    extract_block_field_from_text_raw(pm_block, "DECISION_FACTS")
                    is not None,
                ),
                (
                    "decision_gates",
                    extract_block_field_from_text_raw(pm_block, "DECISION_GATES")
                    is not None,
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
    return {
        "ok": not missing,
        "checks": checks,
        "missing": missing,
        "issues": issues,
    }


def should_fail_closed(
    agent_key: str,
    *,
    validation: dict[str, Any],
    truncated: bool,
    content: str,
) -> bool:
    if validation["ok"]:
        return False

    if agent_key in {
        "portfolio_manager",
        "fundamentals_analyst",
        "foreign_language_analyst",
    }:
        return True

    if agent_key == "global_forensic_auditor":
        return truncated or not validation["ok"]

    if agent_key == "consultant":
        return not content.strip() or (truncated and len(content.strip()) >= 200)

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

    logger.debug(
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
