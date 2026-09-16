"""Read and render the canonical deterministic decision-policy record."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.pm_decision_parser import canonicalize_pm_verdict


def get_decision_policy(source: Any) -> dict[str, Any]:
    """Return a validated-enough policy mapping from runtime or saved state."""
    if not isinstance(source, Mapping):
        return {}
    value = source.get("decision_policy")
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def policy_final_verdict(policy: Mapping[str, Any]) -> str | None:
    """Return the canonical final verdict when the policy record carries one."""
    verdict = canonicalize_pm_verdict(policy.get("final_verdict"))
    return None if verdict == "UNPARSEABLE" else verdict


def _human_reason(value: object) -> str:
    text = str(value or "policy requirement").strip().replace("_", " ")
    return text[:1].upper() + text[1:]


def decision_policy_basis(policy: Mapping[str, Any]) -> str | None:
    """Build a deterministic one-line basis for a verdict-changing intervention."""
    if not policy.get("verdict_changed"):
        return None
    adjustments = policy.get("adjustments")
    if not isinstance(adjustments, list) or not adjustments:
        return None
    first = adjustments[0] if isinstance(adjustments[0], Mapping) else {}
    original = canonicalize_pm_verdict(policy.get("original_verdict"))
    final = policy_final_verdict(policy)
    if original == "UNPARSEABLE" or final is None:
        return None
    return (
        f"Deterministic policy changed {original} to {final}: "
        f"{_human_reason(first.get('reason') or first.get('rule'))}."
    )


def render_decision_policy_notice(policy: Mapping[str, Any]) -> str:
    """Render compact canonical adjustments and qualifications for a report."""
    rows: list[str] = []
    adjustments = policy.get("adjustments")
    if isinstance(adjustments, list):
        for item in adjustments:
            if not isinstance(item, Mapping):
                continue
            before = canonicalize_pm_verdict(item.get("from"))
            after = canonicalize_pm_verdict(item.get("to"))
            if before == "UNPARSEABLE" or after == "UNPARSEABLE":
                continue
            rows.append(
                f"- **Policy adjustment:** {before} → {after} — "
                f"{_human_reason(item.get('reason') or item.get('rule'))}."
            )
    qualifications = policy.get("qualifications")
    if isinstance(qualifications, list):
        for item in qualifications:
            if not isinstance(item, Mapping) or not item.get("kind"):
                continue
            rows.append(f"- **Qualification:** {_human_reason(item['kind'])}.")
    if not rows:
        return ""
    heading = "**Canonical decision-policy result**\n\n"
    if policy.get("verdict_changed"):
        rows.append(
            "- The narrative below originated before deterministic policy "
            "enforcement. This notice and the canonical verdict govern if prose "
            "conflicts."
        )
    return heading + "\n".join(rows) + "\n"
