"""Untrusted content inspection domain model and service abstractions.

This module establishes the core types for the content inspection pipeline.
Every external-content ingress path must pass content through an
InspectionService before that content reaches an LLM context.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)


class SourceKind(str, Enum):
    """Classification of the content's origin — used for policy decisions.

    Future MCP adapters should normalize remote tool/resource output into the
    same inspection plane rather than inventing a parallel trust model.
    """

    tool_output = "tool_output"
    web_search = "web_search"
    web_fetch = "web_fetch"
    official_filing = "official_filing"
    social_feed = "social_feed"
    financial_api = "financial_api"
    memory_retrieval = "memory_retrieval"
    memory_write = "memory_write"
    cached_context = "cached_context"
    mcp_tool_output = "mcp_tool_output"  # reserved for future MCP integration


@dataclass
class InspectionEnvelope:
    """Rich context envelope passed to every inspection backend."""

    content_text: str
    source_kind: SourceKind
    source_name: str
    raw_content: Any | None = None
    tool_name: str | None = None
    agent_key: str | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InspectionDecision:
    """Result returned by a ContentInspector backend."""

    action: Literal["allow", "sanitize", "block", "degrade"]
    threat_level: Literal["safe", "low", "medium", "high", "critical"]
    threat_types: list[str] = field(default_factory=list)
    confidence: float = 1.0
    sanitized_content: str | None = None
    findings: list[str] = field(default_factory=list)
    reason: str | None = None


@runtime_checkable
class ContentInspector(Protocol):
    """Protocol implemented by all inspection backends."""

    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision: ...


class NullInspector:
    """Default no-op inspector — always allows, zero overhead."""

    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        return InspectionDecision(
            action="allow",
            threat_level="safe",
        )


class CompositeInspector:
    """Chain multiple inspection backends with a configurable decision strategy.

    Strategies:
        any_block   — block if any backend says block or degrade
        majority    — block if more than half of backends flag the content
        first_flag  — stop at the first non-allow decision and return it
    """

    def __init__(
        self,
        inspectors: list[ContentInspector],
        strategy: Literal["any_block", "majority", "first_flag"] = "any_block",
    ) -> None:
        self._inspectors = list(inspectors)
        self._strategy = strategy

    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        if not self._inspectors:
            return InspectionDecision(action="allow", threat_level="safe")

        inspection_results = await asyncio.gather(
            *[insp.inspect(envelope) for insp in self._inspectors],
            return_exceptions=True,
        )
        decisions: list[InspectionDecision] = []
        errors: list[tuple[str, Exception]] = []
        for inspector, result in zip(
            self._inspectors, inspection_results, strict=False
        ):
            if isinstance(result, Exception):
                errors.append((type(inspector).__name__, result))
            else:
                decisions.append(result)

        for inspector_name, exc in errors:
            logger.warning(
                "content_inspector_backend_failed",
                inspector=inspector_name,
                strategy=self._strategy,
                source_kind=envelope.source_kind.value,
                source_name=envelope.source_name,
                error=str(exc),
            )

        if not decisions:
            if len(errors) == 1:
                raise errors[0][1]
            raise RuntimeError(
                f"All {len(errors)} content inspectors failed for {envelope.source_name}"
            )

        if self._strategy == "first_flag":
            for decision in decisions:
                if decision.action != "allow":
                    return decision
            return self._most_conservative(decisions)

        blocking = [d for d in decisions if d.action in ("block", "degrade")]

        if self._strategy == "any_block":
            if blocking:
                return self._merge(blocking)
            return self._most_conservative_non_blocking(decisions)

        # majority
        if len(blocking) > len(decisions) / 2:
            return self._merge(blocking)
        return self._most_conservative_non_blocking(decisions)

    @staticmethod
    def _merge(blocking: list[InspectionDecision]) -> InspectionDecision:
        """Merge multiple blocking decisions into one."""
        all_findings: list[str] = []
        all_types: list[str] = []
        for d in blocking:
            all_findings.extend(d.findings)
            all_types.extend(d.threat_types)
        worst = max(
            blocking,
            key=lambda d: ["safe", "low", "medium", "high", "critical"].index(
                d.threat_level
            ),
        )
        return InspectionDecision(
            action=worst.action,
            threat_level=worst.threat_level,
            threat_types=list(dict.fromkeys(all_types)),
            confidence=max(d.confidence for d in blocking),
            sanitized_content=next(
                (d.sanitized_content for d in blocking if d.sanitized_content), None
            ),
            findings=all_findings,
            reason=worst.reason,
        )

    @staticmethod
    def _most_conservative(decisions: list[InspectionDecision]) -> InspectionDecision:
        """Prefer the strictest available decision when no blocking merge applies."""

        action_rank = {
            "allow": 0,
            "sanitize": 1,
            "degrade": 2,
            "block": 3,
        }
        threat_rank = {
            "safe": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }
        return max(
            decisions,
            key=lambda d: (
                action_rank[d.action],
                threat_rank[d.threat_level],
                d.confidence,
            ),
        )

    @classmethod
    def _most_conservative_non_blocking(
        cls, decisions: list[InspectionDecision]
    ) -> InspectionDecision:
        non_blocking = [d for d in decisions if d.action not in ("block", "degrade")]
        if non_blocking:
            return cls._most_conservative(non_blocking)
        return decisions[0]
