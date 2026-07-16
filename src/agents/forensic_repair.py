from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents import message_utils, support
from src.agents import runtime as agent_runtime
from src.data_block_utils import unfenced_label

_RECOVERABLE_AUDITOR_STATUSES = frozenset(
    {"INSUFFICIENT_DATA", "UNAVAILABLE", "CONTEXT_LIMIT_EXCEEDED"}
)
_FORENSIC_LABEL = unfenced_label("FORENSIC_DATA_BLOCK")

FORENSIC_REPAIR_INSTRUCTION = (
    "Rewrite the following forensic auditor output into a final compliant report. "
    "Preserve the factual content. "
    "If the content indicates missing, stale, incomplete, unavailable, or "
    "unverified filings, emit the canonical INSUFFICIENT_DATA fallback form. "
    "Output only the repaired final report. "
    "Do not call tools. "
    f"End with raw labels exactly as {_FORENSIC_LABEL}, STATUS:, and VERDICT:."
)


def _inject_forensic_verdict_if_missing(content: str) -> str:
    if _FORENSIC_LABEL not in content:
        return content

    status_match = re.search(r"(?im)^\s*STATUS:\s*([A-Z_]+)\s*$", content)
    if not status_match:
        return content

    status = status_match.group(1)
    if status not in _RECOVERABLE_AUDITOR_STATUSES:
        return content
    if re.search(r"(?im)^\s*VERDICT:\s*\S+", content):
        return content

    if status == "INSUFFICIENT_DATA":
        verdict = (
            "VERDICT: Unable to perform comprehensive forensic audit from "
            "verified primary source documents."
        )
    else:
        verdict = "VERDICT: Rely on DATA_BLOCK metrics for this ticker."

    content = content.rstrip()
    return f"{content}\n{verdict}\n"


def _expand_inline_forensic_stub(content: str) -> str:
    match = re.search(
        r"(?is)^\s*FORENSIC_DATA_BLOCK:\s*STATUS\s*=\s*(?P<status>[A-Z_]+)"
        r"(?P<rest>.*)$",
        content.strip(),
    )
    if not match:
        return content

    status = match.group("status")
    rest = match.group("rest").strip()
    if not rest:
        return f"{_FORENSIC_LABEL}\nSTATUS: {status}\n"

    meta_parts: list[str] = []
    detail_lines: list[str] = []
    for part in [piece.strip(" ,") for piece in rest.split(",") if piece.strip(" ,")]:
        if "=" not in part:
            detail_lines.append(part)
            continue
        key, value = [segment.strip() for segment in part.split("=", 1)]
        key_upper = key.upper()
        if key_upper == "REASON":
            detail_lines.append(f"REASON: {value}")
        else:
            meta_parts.append(f"{key_upper}={value}")

    rebuilt = [_FORENSIC_LABEL, f"STATUS: {status}"]
    if meta_parts:
        rebuilt.append(f"META: {' | '.join(meta_parts)}")
    rebuilt.extend(detail_lines)
    return "\n".join(rebuilt) + "\n"


def canonicalize_forensic_auditor_output(content: str) -> str:
    normalized = content
    replacements = (
        (
            r"(?im)^(?P<indent>\s*)(?:\*\*)?\s*FORENSIC(?:[ _]DATA)?[ _]?BLOCK\s*(?:\*\*)?\s*:?\s*$",
            r"\g<indent>" + _FORENSIC_LABEL,
        ),
        (
            r"(?im)^(?P<indent>\s*)##+\s*FORENSIC_DATA_BLOCK\s*:?\s*$",
            r"\g<indent>" + _FORENSIC_LABEL,
        ),
        (
            r"(?im)^(?P<indent>\s*)(?:\*\*)?\s*STATUS\s*(?:\*\*)?\s*:\s*(?P<value>\S.*)$",
            r"\g<indent>STATUS: \g<value>",
        ),
        (
            r"(?im)^(?P<indent>\s*)STATUS:\s*\*\*(?P<value>.+?)\*\*\s*$",
            r"\g<indent>STATUS: \g<value>",
        ),
        (
            r"(?im)^(?P<indent>\s*)\*\*verdict:\*\*\s*(?P<value>\S.*)$",
            r"\g<indent>VERDICT: \g<value>",
        ),
        (
            r"(?im)^(?P<indent>\s*)\*\*verdict\*\*:\s*(?P<value>\S.*)$",
            r"\g<indent>VERDICT: \g<value>",
        ),
        (
            r"(?im)^(?P<indent>\s*)verdict:\s*(?P<value>\S.*)$",
            r"\g<indent>VERDICT: \g<value>",
        ),
        (
            r"(?im)^(?P<indent>\s*)VERDICT:\s*\*\*(?P<value>.+?)\*\*\s*$",
            r"\g<indent>VERDICT: \g<value>",
        ),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)

    normalized = _expand_inline_forensic_stub(normalized)
    normalized = _inject_forensic_verdict_if_missing(normalized)

    if content.endswith("\n") and not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


async def repair_forensic_auditor_output(
    llm,
    *,
    invalid_output: str,
) -> str:
    response = await agent_runtime.invoke_with_rate_limit_handling(
        llm,
        [
            SystemMessage(content=FORENSIC_REPAIR_INSTRUCTION),
            HumanMessage(content=invalid_output),
        ],
        context="global_forensic_auditor_repair",
        provider=support.infer_provider_name(llm),
        model_name=support.get_model_name(llm),
    )
    return message_utils.extract_string_content(getattr(response, "content", ""))
