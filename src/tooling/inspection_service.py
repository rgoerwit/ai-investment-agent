"""InspectionService — policy layer that wraps a ContentInspector backend.

Usage
-----
The module exposes a ``INSPECTION_SERVICE`` singleton pre-configured with a
``NullInspector`` (zero overhead, always allows).  Call
``configure_content_inspection()`` at application startup to swap in a real
backend and mode.

Every external-content callsite should call::

    approved = await INSPECTION_SERVICE.check(InspectionEnvelope(...))

and use the returned approved value in place of the raw content; a callsite
that records or labels the outcome calls ``evaluate`` (or ``check_within``)
and branches on the decision's ``enforced`` disposition. This is the
single trust boundary for untrusted prompt-bound content. When the inspector
allows the content, the original ``raw_content`` value is preserved when
available so structured payload shapes are not lost.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Literal

import structlog

from src.async_utils import run_with_hard_timeout
from src.error_safety import summarize_exception
from src.tooling.inspector import (
    ContentInspector,
    InspectionDecision,
    InspectionEnvelope,
    NullInspector,
)

logger = structlog.get_logger(__name__)

_BLOCKED_PLACEHOLDER = "TOOL_BLOCKED: {reason}"
# The reason code of a fail-closed backend error: never the exception text.
INSPECTION_BACKEND_ERROR = "INSPECTION_BACKEND_ERROR"
# The tool-failure code when an enforced ``inspection_failed`` ends a call.
INSPECTION_FAILED = "INSPECTION_FAILED"
_SUPPRESSED_DUPLICATE_MILESTONES = frozenset({1, 5, 10, 25, 50, 100})


class InspectionService:
    """Apply an inspection decision to produce an approved content value.

    Modes:
        warn      — log findings, return original content unchanged
        sanitize  — replace with sanitized_content when provided, else warn
        block     — return a TOOL_BLOCKED placeholder string

    Fail policies:
        fail_open   — if the backend raises, allow the content through
        fail_closed — if the backend raises, enforce ``inspection_failed``
                      with a TOOL_BLOCKED placeholder carrying a reason code
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
        self._finding_counts: dict[tuple[str, ...], int] = {}

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
        self._finding_counts.clear()

    @property
    def mode(self) -> str:
        return self._mode

    @staticmethod
    def _is_low_risk_sanitize(decision: InspectionDecision) -> bool:
        """Should this sanitize decision be logged at debug instead of warning?

        Two cases qualify:

        1. ``threat_level == "low"`` — the inspector found a minor artifact
           and successfully scrubbed it. Bidi marks (U+200E/U+202A/…) in CJK
           or Arabic web text routinely fall here; warnings would just be
           noise for the operator. The data was already replaced or passed
           through, depending on mode.

        2. ``threat_types == {"delimiter_breakout"}`` regardless of severity
           — a stray ``</search_results>`` in Tavily output is the canonical
           case. It scores medium under the heuristic weights but is benign
           when the wrapper is the only finding.

        Mixed scrubs (e.g. delimiter_breakout + formatting_chars at "high")
        and any non-low non-delimiter-only sanitize still warn, so genuine
        attempts at delimiter breakout in CJK content remain visible.
        """
        if decision.action != "sanitize":
            return False
        if decision.threat_level == "low":
            return True
        if set(decision.threat_types) == {"delimiter_breakout"}:
            return True
        return False

    @staticmethod
    def _base_log_payload(
        envelope: InspectionEnvelope,
        decision: InspectionDecision,
    ) -> dict[str, Any]:
        return {
            "source_kind": envelope.source_kind.value,
            "source_name": envelope.source_name,
            "tool_name": envelope.tool_name,
            "agent_key": envelope.agent_key,
            "action": decision.action,
            "threat_level": decision.threat_level,
            "threat_types": decision.threat_types,
            "confidence": decision.confidence,
            "findings": decision.findings,
            "reason": decision.reason,
            "expected_wrapper_cleanup": decision.expected_wrapper_cleanup,
        }

    def _low_risk_dedupe_key(
        self,
        envelope: InspectionEnvelope,
        decision: InspectionDecision,
    ) -> tuple[str, ...]:
        content_hash = hashlib.sha1(envelope.content_text.encode("utf-8")).hexdigest()[
            :12
        ]
        return (
            envelope.source_kind.value,
            envelope.source_name,
            envelope.tool_name or "",
            envelope.agent_key or "",
            decision.action,
            ",".join(sorted(decision.threat_types)),
            decision.reason or "",
            content_hash,
        )

    def _duplicate_count(
        self,
        envelope: InspectionEnvelope,
        decision: InspectionDecision,
    ) -> int:
        dedupe_key = self._low_risk_dedupe_key(envelope, decision)
        count = self._finding_counts.get(dedupe_key, 0) + 1
        self._finding_counts[dedupe_key] = count
        return count

    def _log_inspection_finding(
        self,
        envelope: InspectionEnvelope,
        decision: InspectionDecision,
    ) -> None:
        payload = self._base_log_payload(envelope, decision)
        count = self._duplicate_count(envelope, decision)
        if count > 1:
            suppressed = count - 1
            if suppressed in _SUPPRESSED_DUPLICATE_MILESTONES:
                logger.debug(
                    "content_inspection_suppressed_duplicates",
                    suppressed_duplicates=suppressed,
                    **payload,
                )
            return

        if decision.action in ("block", "degrade"):
            logger.warning("content_inspection_finding", **payload)
            return

        if self._is_low_risk_sanitize(decision):
            logger.debug("content_inspection_finding", **payload)
            return

        if decision.action == "sanitize":
            logger.warning("content_inspection_finding", **payload)
            return

        logger.debug("content_inspection_finding", **payload)

    def _enforce(
        self, decision: InspectionDecision, original_value: Any
    ) -> tuple[InspectionDecision, Any]:
        """Apply the configured mode to a raw *decision*, once.

        Returns the decision with ``enforced`` set and the approved value:
        a ``block``/``degrade`` is enforced only in block mode (sanitize mode
        uses the replacement when the backend supplied one), a ``sanitize``
        only in sanitize or block mode with a replacement. An empty
        replacement is still a replacement: scrubbing everything is never
        read as "nothing to scrub".
        """
        replacement = decision.sanitized_content
        if decision.action in ("block", "degrade"):
            if self._mode == "block":
                reason = decision.reason or "; ".join(decision.findings) or "policy"
                return (
                    replace(decision, enforced="blocked"),
                    _BLOCKED_PLACEHOLDER.format(reason=reason),
                )
            if self._mode == "sanitize" and replacement is not None:
                return replace(decision, enforced="sanitized"), replacement
        elif decision.action == "sanitize":
            if self._mode in ("sanitize", "block") and replacement is not None:
                return replace(decision, enforced="sanitized"), replacement
        return replace(decision, enforced="allowed"), original_value

    def _backend_failed(
        self, envelope: InspectionEnvelope, exc: Exception, original_value: Any
    ) -> tuple[InspectionDecision, Any]:
        """The fail policy's outcome when the backend raised.

        The exception reaches the log only through ``summarize_exception``;
        the decision and placeholder carry the secret-free reason code.
        """
        fail_closed = self._fail_policy == "fail_closed"
        (logger.error if fail_closed else logger.warning)(
            "content_inspection_backend_error",
            source_kind=envelope.source_kind.value,
            source_name=envelope.source_name,
            **summarize_exception(exc, operation="content_inspection_backend_error"),
            fail_policy=self._fail_policy,
        )
        if not fail_closed:
            return (
                InspectionDecision(
                    action="allow", threat_level="safe", enforced="allowed"
                ),
                original_value,
            )
        return (
            InspectionDecision(
                action="block",
                threat_level="high",
                findings=[INSPECTION_BACKEND_ERROR],
                reason=INSPECTION_BACKEND_ERROR,
                enforced="inspection_failed",
            ),
            _BLOCKED_PLACEHOLDER.format(reason=INSPECTION_BACKEND_ERROR),
        )

    async def evaluate(
        self,
        envelope: InspectionEnvelope,
    ) -> tuple[InspectionDecision, Any]:
        """Inspect *envelope*; return the enforced decision and approved content.

        The decision's ``enforced`` disposition (allowed, sanitized, blocked,
        inspection_failed) is the one machine-readable outcome: callers that
        must tell a block from evidence branch on it, never on the raw
        ``action`` or on the approved text.
        """
        original_value = (
            envelope.raw_content
            if envelope.raw_content is not None
            else envelope.content_text
        )
        try:
            decision: InspectionDecision = await self._inspector.inspect(envelope)
        except Exception as exc:
            return self._backend_failed(envelope, exc, original_value)

        if decision.action == "allow" and decision.threat_level == "safe":
            return replace(decision, enforced="allowed"), original_value

        # Log findings for non-trivial decisions regardless of mode.
        if decision.findings or decision.threat_types:
            self._log_inspection_finding(envelope, decision)
        return self._enforce(decision, original_value)

    async def check(self, envelope: InspectionEnvelope) -> Any:
        """Inspect *envelope* and return the approved content only.

        For callers that use the content and nothing else: the original
        value, a sanitized replacement, or a blocked placeholder string.
        A caller that records or labels the outcome uses ``evaluate``.
        """
        _, approved = await self.evaluate(envelope)
        return approved

    async def check_within(
        self, envelope: InspectionEnvelope, *, seconds: float, label: str
    ) -> tuple[InspectionDecision, Any] | None:
        """``evaluate`` bounded deadline-only by ``seconds``; ``None`` when it cannot finish.

        For inspection that must fit inside a caller's allowance (a tool body,
        a search entrypoint). ``seconds <= 0`` starts nothing; an inspection
        that overruns is orphaned, never awaited. Either way the result is
        ``None`` -- fail closed: the uninspected payload is never returned.

        It binds no deadline. An LLM judge inside sizes itself from the LLM
        enclosing deadline alone, never from the tool deadline the caller may
        run under: refused by admission, the judge fails *open*, where this
        bound fails closed.
        """
        if seconds <= 0:
            return None
        try:
            return await run_with_hard_timeout(
                self.evaluate(envelope), timeout=seconds, label=label
            )
        except TimeoutError:
            return None


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
