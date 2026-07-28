"""Typed capture for tool outputs that feed deterministic analysis contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def build_structured_ingress_record(
    value: Any,
    *,
    agent_key: str,
    tool_name: str,
    blocked: bool = False,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Return a serializable, fail-closed record before tool text is truncated."""
    payload: Any = value
    serialized = (
        value
        if isinstance(value, str)
        else json.dumps(
            value,
            ensure_ascii=False,
            default=str,
            sort_keys=True,
        )
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    reason = failure_reason
    if reason is None and blocked:
        reason = "TOOL_OUTPUT_BLOCKED"
    if reason is None and isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            reason = "MALFORMED_JSON"
    if reason is None and not isinstance(payload, Mapping):
        reason = "JSON_OBJECT_REQUIRED"
    if reason is None and not payload:
        reason = "EMPTY_PAYLOAD"
    if reason is None and payload.get("error"):
        reason = "TOOL_ERROR_PAYLOAD"

    return {
        "status": "INVALID" if reason else "VALID",
        "reason": reason,
        "agent_key": agent_key,
        "tool_name": tool_name,
        "content_sha256": digest,
        "payload": dict(payload) if reason is None else {},
    }


def merge_structured_inputs(
    current: Mapping[str, Mapping[str, Any]] | None,
    incoming: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Merge inputs without silently replacing an established valid payload."""
    merged = {key: dict(record) for key, record in (current or {}).items()}
    for key, new_record_value in (incoming or {}).items():
        new_record = dict(new_record_value)
        existing = merged.get(key)
        if existing is None:
            merged[key] = new_record
            continue
        if existing.get("reason") == "CONFLICTING_VALID_PAYLOADS":
            continue

        old_valid = existing.get("status") == "VALID"
        new_valid = new_record.get("status") == "VALID"
        if old_valid and new_valid:
            if existing.get("payload") == new_record.get("payload"):
                merged[key] = new_record
            else:
                merged[key] = {
                    "status": "INVALID",
                    "reason": "CONFLICTING_VALID_PAYLOADS",
                    "agent_key": "multiple",
                    "tool_name": str(new_record.get("tool_name") or ""),
                    "content_sha256": "",
                    "payload": {},
                }
        elif new_valid:
            merged[key] = new_record
        elif old_valid:
            continue
        elif existing.get("reason") != new_record.get("reason"):
            merged[key] = {
                "status": "INVALID",
                "reason": "MULTIPLE_INGRESS_FAILURES",
                "agent_key": "multiple",
                "tool_name": str(new_record.get("tool_name") or ""),
                "content_sha256": "",
                "payload": {},
            }
    return merged


def get_structured_ingress_payload(
    state: Mapping[str, Any],
    contract_key: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve one typed input record and explain any contract failure."""
    records = state.get("structured_inputs")
    if not isinstance(records, Mapping):
        return None, "STRUCTURED_INPUT_REGISTRY_MISSING"
    record = records.get(contract_key)
    if not isinstance(record, Mapping):
        return None, "STRUCTURED_INPUT_MISSING"
    if record.get("status") != "VALID":
        return None, str(record.get("reason") or "STRUCTURED_INPUT_INVALID")
    payload = record.get("payload")
    if not isinstance(payload, Mapping) or not payload:
        return None, "STRUCTURED_INPUT_PAYLOAD_INVALID"
    return dict(payload), None


def render_structured_ingress_payload(
    state: Mapping[str, Any],
    contract_key: str,
) -> str:
    """Render a typed payload for compatibility consumers without model relay."""
    payload, _ = get_structured_ingress_payload(state, contract_key)
    if payload is None:
        return ""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
