"""InspectionService — policy layer that wraps a ContentInspector backend.

Usage
-----
The module exposes a ``INSPECTION_SERVICE`` singleton pre-configured with a
``NullInspector`` (zero overhead, always allows).  Call
``configure_content_inspection()`` at application startup to swap in a real
backend and mode.

Every external-content callsite should call::

    approved = await INSPECTION_SERVICE.check(InspectionEnvelope(...))

and use the returned approved value in place of the raw content. This is the
single trust boundary for untrusted prompt-bound content. When the inspector
allows the content, the original ``raw_content`` value is preserved when
available so structured payload shapes are not lost.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog

from src.tooling.inspector import (
    ContentInspector,
    InspectionDecision,
    InspectionEnvelope,
    NullInspector,
)

logger = structlog.get_logger(__name__)

_BLOCKED_PLACEHOLDER = "TOOL_BLOCKED: {reason}"


class InspectionService:
    """Apply an inspection decision to produce an approved content value.

    Modes:
        warn      — log findings, return original content unchanged
        sanitize  — replace with sanitized_content when provided, else warn
        block     — return a CONTENT_BLOCKED placeholder string

    Fail policies:
        fail_open   — if the backend raises, allow the content through
        fail_closed — if the backend raises, treat as blocked
    """

    def __init__(
        self,
        inspector: ContentInspector | None = None,
        mode: Literal["warn", "sanitize", "block"] = "warn",
        fail_policy: Literal["fail_open", "fail_closed"] = "fail_open",
    ) -> None:
        self._inspector: ContentInspector = inspector or NullInspector()
        self._mode = mode
        self._fail_policy = fail_policy

    def configure(
        self,
        inspector: ContentInspector,
        mode: Literal["warn", "sanitize", "block"] = "warn",
        fail_policy: Literal["fail_open", "fail_closed"] = "fail_open",
    ) -> None:
        """Reconfigure the service (called at startup by configure_content_inspection)."""
        self._inspector = inspector
        self._mode = mode
        self._fail_policy = fail_policy

    @property
    def mode(self) -> str:
        return self._mode

    async def check(self, envelope: InspectionEnvelope) -> Any:
        """Inspect *envelope* and return the approved content string.

        Returns the original value, a sanitized replacement string, or a
        blocked-placeholder string depending on the decision and configured mode.
        """
        original_value = (
            envelope.raw_content
            if envelope.raw_content is not None
            else envelope.content_text
        )
        try:
            decision: InspectionDecision = await self._inspector.inspect(envelope)
        except Exception as exc:
            if self._fail_policy == "fail_closed":
                reason = f"inspector error: {exc}"
                logger.error(
                    "content_inspection_backend_error",
                    source_kind=envelope.source_kind.value,
                    source_name=envelope.source_name,
                    error=str(exc),
                    fail_policy=self._fail_policy,
                )
                return _BLOCKED_PLACEHOLDER.format(reason=reason)
            logger.warning(
                "content_inspection_backend_error",
                source_kind=envelope.source_kind.value,
                source_name=envelope.source_name,
                error=str(exc),
                fail_policy=self._fail_policy,
            )
            return original_value

        if decision.action == "allow" and decision.threat_level == "safe":
            return original_value

        # Log findings for non-trivial decisions regardless of mode.
        if decision.findings or decision.threat_types:
            logger.warning(
                "content_inspection_finding",
                source_kind=envelope.source_kind.value,
                source_name=envelope.source_name,
                tool_name=envelope.tool_name,
                agent_key=envelope.agent_key,
                action=decision.action,
                threat_level=decision.threat_level,
                threat_types=decision.threat_types,
                confidence=decision.confidence,
                findings=decision.findings,
                reason=decision.reason,
            )

        if decision.action in ("block", "degrade"):
            if self._mode == "block":
                reason = decision.reason or "; ".join(decision.findings) or "policy"
                return _BLOCKED_PLACEHOLDER.format(reason=reason)
            if self._mode == "sanitize" and decision.sanitized_content is not None:
                return decision.sanitized_content
            # warn (or sanitize-without-replacement): log and pass through
            return original_value

        if decision.action == "sanitize":
            if self._mode in ("sanitize", "block") and decision.sanitized_content:
                return decision.sanitized_content
            return original_value

        # action == "allow" with non-safe threat level → warn and pass through
        return original_value


# ---------------------------------------------------------------------------
# Module-level singleton — pre-configured with NullInspector (zero overhead).
# Swap at startup via configure_content_inspection().
# ---------------------------------------------------------------------------

INSPECTION_SERVICE = InspectionService()


def configure_content_inspection(
    inspector: ContentInspector,
    mode: Literal["warn", "sanitize", "block"] = "warn",
    fail_policy: Literal["fail_open", "fail_closed"] = "fail_open",
) -> None:
    """Install an inspection backend on the global singleton.

    This configures the shared service only. Callers that want tool-output
    inspection must also install ``ContentInspectionHook`` on ``TOOL_SERVICE``
    or use the higher-level runtime setup path in ``src.main``.
    """
    INSPECTION_SERVICE.configure(inspector, mode=mode, fail_policy=fail_policy)
    logger.info(
        "content_inspection_configured",
        inspector=type(inspector).__name__,
        mode=mode,
        fail_policy=fail_policy,
    )
