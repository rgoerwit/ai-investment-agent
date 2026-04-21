"""Escalating content inspector — heuristic first, LLM judge on escalation.

Implements ``ContentInspector`` for the ``"composite"`` backend config slot.

Unlike ``CompositeInspector`` (which runs all backends in parallel via
``asyncio.gather``), this inspector is sequential and selective: the LLM
judge is only invoked when the heuristic flags content at or above a
threshold, or when the source kind is in the always-judge set.
"""

from __future__ import annotations

from typing import Literal

import structlog

from src.tooling.inspector import (
    ContentInspector,
    InspectionDecision,
    InspectionEnvelope,
    SourceKind,
)

logger = structlog.get_logger(__name__)

_THREAT_RANK: dict[str, int] = {
    "safe": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

_ACTION_RANK: dict[str, int] = {
    "allow": 0,
    "sanitize": 1,
    "degrade": 2,
    "block": 3,
}

# Default to threshold-based escalation only. Hot-path "always judge" sources
# remain opt-in so the composite backend does not silently add cost/latency.
_DEFAULT_ALWAYS_JUDGE: frozenset[SourceKind] = frozenset()


def _threat_at_or_above(
    level: str,
    threshold: Literal["low", "medium", "high"],
) -> bool:
    """Return True if *level* is at or above *threshold*."""
    return _THREAT_RANK.get(level, 0) >= _THREAT_RANK.get(threshold, 2)


class EscalatingInspector:
    """Heuristic first, LLM judge only on escalation.

    Implements ``ContentInspector`` protocol.

    Parameters
    ----------
    heuristic:
        Fast pattern-based inspector (e.g. ``HeuristicInspector``).
    judge:
        Semantic LLM-based inspector (e.g. ``LLMJudgeInspector``).
    escalation_threshold:
        Minimum heuristic threat level that triggers judge escalation.
    always_judge_sources:
        Source kinds that always get judge treatment regardless of
        heuristic result.
    """

    def __init__(
        self,
        heuristic: ContentInspector,
        judge: ContentInspector,
        *,
        escalation_threshold: Literal["low", "medium", "high"] = "medium",
        always_judge_sources: frozenset[SourceKind] | None = None,
    ) -> None:
        self._heuristic = heuristic
        self._judge = judge
        self._escalation_threshold = escalation_threshold
        self._always_judge_sources = (
            always_judge_sources
            if always_judge_sources is not None
            else _DEFAULT_ALWAYS_JUDGE
        )

    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        heuristic_result = await self._heuristic.inspect(envelope)

        needs_judge = (
            _threat_at_or_above(
                heuristic_result.threat_level, self._escalation_threshold
            )
            or envelope.source_kind in self._always_judge_sources
        )

        if not needs_judge:
            return heuristic_result

        logger.debug(
            "escalating_to_llm_judge",
            source_kind=envelope.source_kind.value,
            source_name=envelope.source_name,
            heuristic_threat=heuristic_result.threat_level,
            heuristic_action=heuristic_result.action,
            reason="threshold_exceeded"
            if _threat_at_or_above(
                heuristic_result.threat_level, self._escalation_threshold
            )
            else "always_judge_source",
        )

        try:
            judge_result = await self._judge.inspect(envelope)
        except Exception as exc:
            # Judge failure — fall back to heuristic result.
            logger.warning(
                "llm_judge_escalation_failed",
                source_kind=envelope.source_kind.value,
                error=str(exc),
            )
            return heuristic_result

        return self._merge(heuristic_result, judge_result)

    @staticmethod
    def _merge(
        heuristic: InspectionDecision,
        judge: InspectionDecision,
    ) -> InspectionDecision:
        """Merge heuristic and judge decisions, taking the more conservative."""
        h_action_rank = _ACTION_RANK.get(heuristic.action, 0)
        j_action_rank = _ACTION_RANK.get(judge.action, 0)
        h_threat_rank = _THREAT_RANK.get(heuristic.threat_level, 0)
        j_threat_rank = _THREAT_RANK.get(judge.threat_level, 0)

        # Take the more conservative action and threat level.
        if j_action_rank >= h_action_rank:
            action = judge.action
        else:
            action = heuristic.action

        if j_threat_rank >= h_threat_rank:
            threat_level = judge.threat_level
        else:
            threat_level = heuristic.threat_level

        # Merge findings and threat types.
        all_types = list(dict.fromkeys(heuristic.threat_types + judge.threat_types))
        all_findings = heuristic.findings + judge.findings
        confidence = max(heuristic.confidence, judge.confidence)

        # Prefer judge's sanitized content if available.
        sanitized = judge.sanitized_content or heuristic.sanitized_content

        reasons = []
        if heuristic.reason:
            reasons.append(f"heuristic: {heuristic.reason}")
        if judge.reason:
            reasons.append(f"judge: {judge.reason}")

        return InspectionDecision(
            action=action,
            threat_level=threat_level,
            threat_types=all_types,
            confidence=confidence,
            sanitized_content=sanitized,
            findings=all_findings,
            reason="; ".join(reasons) if reasons else None,
        )
