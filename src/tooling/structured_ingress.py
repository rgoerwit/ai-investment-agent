"""Typed capture for tool outputs that feed deterministic analysis contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from src.data.merge_policy import NON_ACTIONABLE_CONFLICT_FIELDS, QUOTE_PRICE_FIELDS

# Fields that legitimately tick between two live fetches of the same contract
# within a single graph run (quote microstructure + volume/cap scalars that move
# with price, plus fetch-time timestamps) — not analytic fundamentals. Disjoint
# from CRITICAL_ANALYSIS_FIELDS/ANALYSIS_CRITICAL_CONFLICT_FIELDS in
# merge_policy.py, so excluding them from the *comparison* below can never mask a
# genuine fundamentals-field conflict. Never stripped from the stored payload.
#
# NOTE: this set is global, not scoped per contract_key. Today that's fine —
# STRUCTURED_INGRESS_SOURCES (src/claim_policy.py) has exactly one registered
# contract (raw_financial_metrics ← Junior Fundamentals' get_financial_metrics),
# and it's quote-shaped, so the field names line up. If a second, unrelated
# contract type is ever registered (e.g. an ownership-structure or ESG ingress)
# whose payload isn't quote-shaped, this same exclusion set would still apply to
# it — harmless unless that payload happens to reuse one of these field names for
# something non-volatile, or needs its own volatile-field carve-outs that don't
# exist yet. Make _stable_payload_for_comparison contract-scoped
# (dict[contract_key, frozenset[str]]) at that point; don't build it now.
_INGRESS_VOLATILE_RECHECK_FIELDS: frozenset[str] = (
    frozenset(QUOTE_PRICE_FIELDS)
    | NON_ACTIONABLE_CONFLICT_FIELDS
    | frozenset(
        {
            "volume",
            "averageVolume",
            "regularMarketVolume",
            "marketCap",
            "enterpriseValue",
            "regularMarketTime",
            "preMarketTime",
            "postMarketTime",
        }
    )
)


def _stable_payload_for_comparison(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip volatile/live-ticking fields before comparing two VALID payloads.

    Two independent live fetches of the same entity minutes apart are never
    byte-identical (price ticks, volume, timestamps) even when the underlying
    fundamentals are unchanged. Used only to decide "same vs. conflicting" — the
    full payload (including volatile fields) is still what gets stored.
    """
    return {
        k: v for k, v in payload.items() if k not in _INGRESS_VOLATILE_RECHECK_FIELDS
    }


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
            if _stable_payload_for_comparison(
                existing.get("payload") or {}
            ) == _stable_payload_for_comparison(new_record.get("payload") or {}):
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
