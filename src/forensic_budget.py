"""Single-source budgets and telemetry for bounded research agents."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Protocol, cast
from urllib.parse import urlparse

from src.config import config

_REJECTED_HOST_RE = re.compile(r"(?m)^REJECTED_HOST:\s*(\S+)")
_REASON_RE = re.compile(r"(?m)^REASON:\s*([A-Z0-9_]+)")


class ResearchBudgetPolicy(Protocol):
    """Structural contract consumed by the shared research ledger."""

    @property
    def max_tool_iterations(self) -> int: ...

    @property
    def max_llm_calls(self) -> int: ...

    @property
    def max_tool_calls_per_turn(self) -> int: ...

    @property
    def max_evidence_chars(self) -> int: ...

    @property
    def duplicate_call_limit(self) -> int: ...

    @property
    def host_failure_limit(self) -> int: ...

    @property
    def tool_failure_limit(self) -> int: ...

    @property
    def purpose_call_limit(self) -> int: ...

    def tool_limit(self, name: str) -> int | None: ...


@dataclass(frozen=True)
class GraphResearchBudgetPolicy:
    """Code-owned limits for one shared graph analyst/tool loop."""

    tool_limits: Mapping[str, int]
    max_tool_iterations: int
    max_llm_calls: int
    max_tool_calls_per_turn: int
    max_evidence_chars: int = 80_000
    duplicate_call_limit: int = 1
    host_failure_limit: int = 2
    tool_failure_limit: int = 2
    purpose_call_limit: int = 2

    def tool_limit(self, name: str) -> int | None:
        return self.tool_limits.get(name)


@dataclass(frozen=True)
class AuditorBudgetPolicy:
    search_calls: int
    document_calls: int
    filing_calls: int
    metrics_calls: int
    news_calls: int
    calculation_calls: int
    max_document_bytes: int
    max_document_pages: int
    max_selected_pages: int
    max_evidence_chars: int
    max_tool_iterations: int
    max_llm_calls: int
    max_tool_calls_per_turn: int = 8
    duplicate_call_limit: int = 1
    host_failure_limit: int = 2
    tool_failure_limit: int = 2
    purpose_call_limit: int = 2

    @classmethod
    def from_settings(cls) -> AuditorBudgetPolicy:
        return cls(
            search_calls=config.auditor_search_call_budget,
            document_calls=config.auditor_document_budget,
            filing_calls=1,
            metrics_calls=1,
            news_calls=1,
            calculation_calls=2,
            max_document_bytes=config.auditor_max_document_bytes,
            max_document_pages=config.auditor_max_document_pages,
            max_selected_pages=config.auditor_max_selected_pages,
            max_evidence_chars=config.auditor_max_evidence_chars,
            max_tool_iterations=config.auditor_max_tool_iterations,
            max_llm_calls=config.auditor_max_llm_calls,
        )

    def tool_limit(self, name: str) -> int | None:
        return {
            "search_foreign_sources": self.search_calls,
            "get_official_document": self.document_calls,
            "get_official_filings": self.filing_calls,
            "get_financial_metrics": self.metrics_calls,
            "get_news": self.news_calls,
            "calculate_forensic_ratios": self.calculation_calls,
            "validate_forensic_evidence": self.calculation_calls,
        }.get(name)


@dataclass(frozen=True)
class ForeignLanguageBudgetPolicy:
    """Budget for source-discovery agents using the shared graph tool loop."""

    search_calls: int
    document_calls: int
    filing_calls: int
    guidance_calls: int
    max_tool_iterations: int
    max_llm_calls: int
    max_tool_calls_per_turn: int
    max_evidence_chars: int = 80_000
    duplicate_call_limit: int = 1
    host_failure_limit: int = 2
    tool_failure_limit: int = 2
    purpose_call_limit: int = 2

    @classmethod
    def from_settings(cls, *, quick_mode: bool) -> ForeignLanguageBudgetPolicy:
        return cls(
            search_calls=(
                config.foreign_research_quick_search_call_budget
                if quick_mode
                else config.foreign_research_search_call_budget
            ),
            document_calls=config.foreign_research_document_call_budget,
            filing_calls=1,
            guidance_calls=config.foreign_research_guidance_call_budget,
            max_tool_iterations=(
                config.foreign_research_quick_max_tool_iterations
                if quick_mode
                else config.foreign_research_max_tool_iterations
            ),
            # The normal loop uses at most iterations+1 calls (initial request,
            # bounded tool turns, then forced synthesis); +2 preserves one final
            # structural-recovery turn without allowing another research loop.
            max_llm_calls=(
                config.foreign_research_quick_max_tool_iterations
                if quick_mode
                else config.foreign_research_max_tool_iterations
            )
            + 2,
            max_tool_calls_per_turn=config.foreign_research_max_tool_calls_per_turn,
        )

    def tool_limit(self, name: str) -> int | None:
        return {
            "search_foreign_sources": self.search_calls,
            "get_official_document": self.document_calls,
            "get_official_filings": self.filing_calls,
            "extract_guidance_sources": self.guidance_calls,
        }.get(name)


def graph_research_budget_policies(
    *, quick_mode: bool
) -> dict[str, ResearchBudgetPolicy]:
    """Return bounded policies for every shared graph tool-loop analyst."""

    return {
        "market_analyst": GraphResearchBudgetPolicy(
            tool_limits={
                "get_yfinance_data": 1,
                "get_technical_indicators": 1,
                "calculate_liquidity_metrics": 1,
            },
            max_tool_iterations=2,
            max_llm_calls=4,
            max_tool_calls_per_turn=3,
        ),
        "sentiment_analyst": GraphResearchBudgetPolicy(
            tool_limits={
                "get_social_media_sentiment": 1,
                "get_multilingual_sentiment_search": 2,
            },
            max_tool_iterations=2,
            max_llm_calls=4,
            max_tool_calls_per_turn=2,
        ),
        "news_analyst": GraphResearchBudgetPolicy(
            tool_limits={
                "get_news": 2,
                "get_macroeconomic_news": 1,
                "search_foreign_sources": 4,
            },
            max_tool_iterations=3,
            max_llm_calls=5,
            max_tool_calls_per_turn=4,
        ),
        # One round is intentional: both deterministic data tools fit in the
        # same two-call turn. The two extra model calls reserve synthesis plus
        # one structural recovery; another tool round would repeat acquisition.
        "junior_fundamentals_analyst": GraphResearchBudgetPolicy(
            tool_limits={
                "get_financial_metrics": 1,
                "get_fundamental_analysis": 1,
            },
            max_tool_iterations=1,
            max_llm_calls=3,
            max_tool_calls_per_turn=2,
        ),
        "foreign_language_analyst": ForeignLanguageBudgetPolicy.from_settings(
            quick_mode=quick_mode
        ),
        "value_trap_detector": GraphResearchBudgetPolicy(
            tool_limits={
                "get_ownership_structure": 1,
                "get_official_filings": 1,
                "get_news": 2,
                "search_foreign_sources": 6,
            },
            max_tool_iterations=4,
            max_llm_calls=6,
            max_tool_calls_per_turn=4,
        ),
    }


@dataclass
class ResearchBudgetLedger:
    """Provider-neutral accounting and circuit breaking for research tools."""

    policy: ResearchBudgetPolicy
    tool_calls: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    evidence_chars: int = 0
    evidence_truncated: bool = False
    outcomes: list[str] = field(default_factory=list)
    tool_rounds_used: int = 0
    forced_synthesis_used: bool = False
    stop_reason: str | None = None
    final_tool_names: list[str] = field(default_factory=list)
    failed_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)
    insufficient_tools: list[str] = field(default_factory=list)
    rejected_hosts: list[str] = field(default_factory=list)
    synthesis_evidence_chars: int = 0
    repair_input_chars: int = 0
    purpose_calls: dict[str, int] = field(default_factory=dict)
    call_signatures: dict[str, int] = field(default_factory=dict)
    host_failures: dict[str, int] = field(default_factory=dict)
    tool_failures_by_name: dict[str, int] = field(default_factory=dict)
    blocked_reasons: dict[str, int] = field(default_factory=dict)
    tool_outcome_events: dict[str, int] = field(default_factory=dict)
    tool_outcome_events_by_name: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def from_telemetry(
        cls,
        policy: ResearchBudgetPolicy,
        telemetry: Mapping[str, Any] | None,
    ) -> ResearchBudgetLedger:
        ledger = cls(policy)
        if not telemetry:
            return ledger
        for name in (
            "tool_calls",
            "purpose_calls",
            "call_signatures",
            "host_failures",
            "tool_failures_by_name",
            "blocked_reasons",
            "tool_outcome_events",
        ):
            value = telemetry.get(name)
            if isinstance(value, Mapping):
                setattr(
                    ledger,
                    name,
                    {
                        str(key): int(count)
                        for key, count in value.items()
                        if isinstance(count, int | float)
                    },
                )
        raw_events_by_name = telemetry.get("tool_outcome_events_by_name")
        if isinstance(raw_events_by_name, Mapping):
            ledger.tool_outcome_events_by_name = {
                str(category): {
                    str(tool_name): int(count)
                    for tool_name, count in values.items()
                    if isinstance(count, int | float)
                }
                for category, values in raw_events_by_name.items()
                if isinstance(values, Mapping)
            }
        for name in (
            "llm_calls",
            "evidence_chars",
            "tool_rounds_used",
            "synthesis_evidence_chars",
            "repair_input_chars",
        ):
            value = telemetry.get(name)
            if isinstance(value, int | float):
                setattr(ledger, name, int(value))
        for name in (
            "outcomes",
            "final_tool_names",
            "failed_tools",
            "blocked_tools",
            "insufficient_tools",
            "rejected_hosts",
        ):
            value = telemetry.get(name)
            if isinstance(value, list):
                setattr(ledger, name, [str(item) for item in value])
        ledger.evidence_truncated = bool(telemetry.get("evidence_truncated"))
        ledger.forced_synthesis_used = bool(telemetry.get("forced_synthesis_used"))
        stop_reason = telemetry.get("stop_reason")
        ledger.stop_reason = str(stop_reason) if stop_reason else None
        return ledger

    @staticmethod
    def _signature(name: str, args: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            {"name": name, "args": dict(args)},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _host(args: Mapping[str, Any]) -> str | None:
        value = args.get("url")
        if not isinstance(value, str):
            return None
        try:
            return (urlparse(value).hostname or "").rstrip(".").lower() or None
        except ValueError:
            return None

    def _block(self, reason: str, tool_name: str) -> str:
        self.record_outcome(reason)
        self.record_tool_blocked(tool_name)
        self.blocked_reasons[reason] = self.blocked_reasons.get(reason, 0) + 1
        return reason

    def block_tool(self, name: str, reason: str) -> str:
        """Record a caller-owned structural block (for example turn fan-out)."""

        return self._block(reason, name)

    def authorize_tool(self, name: str, args: Mapping[str, Any]) -> str | None:
        """Reserve one tool call or return a stable block reason."""

        host = self._host(args)
        if host and self.host_failures.get(host, 0) >= self.policy.host_failure_limit:
            return self._block("HOST_FAILURE_CIRCUIT_OPEN", name)
        if self.tool_failures_by_name.get(name, 0) >= self.policy.tool_failure_limit:
            return self._block("TOOL_FAILURE_CIRCUIT_OPEN", name)
        signature = self._signature(name, args)
        if self.call_signatures.get(signature, 0) >= self.policy.duplicate_call_limit:
            return self._block("DUPLICATE_TOOL_CALL", name)
        purpose = str(args.get("purpose") or "general").strip().lower()
        purpose_budgeted = name == "search_foreign_sources" or "purpose" in args
        if (
            purpose_budgeted
            and self.purpose_calls.get(purpose, 0) >= self.policy.purpose_call_limit
        ):
            return self._block("PURPOSE_CALL_BUDGET_EXHAUSTED", name)
        exhausted = self.consume_tool(name)
        if exhausted:
            return self._block(exhausted, name)
        self.call_signatures[signature] = self.call_signatures.get(signature, 0) + 1
        if purpose_budgeted:
            self.purpose_calls[purpose] = self.purpose_calls.get(purpose, 0) + 1
        return None

    def consume_tool(self, name: str) -> str | None:
        limit = self.policy.tool_limit(name)
        used = self.tool_calls.get(name, 0)
        if limit is not None and used >= limit:
            self.record_outcome("TOOL_CALL_BUDGET_EXHAUSTED")
            return "TOOL_CALL_BUDGET_EXHAUSTED"
        self.tool_calls[name] = used + 1
        return None

    def consume_llm(self) -> str | None:
        if self.llm_calls >= self.policy.max_llm_calls:
            self.record_outcome("LLM_CALL_BUDGET_EXHAUSTED")
            return "LLM_CALL_BUDGET_EXHAUSTED"
        self.llm_calls += 1
        return None

    def cap_evidence(self, value: str) -> str:
        max_evidence_chars = int(getattr(self.policy, "max_evidence_chars", 0))
        if max_evidence_chars <= 0:
            return value
        remaining = max(0, max_evidence_chars - self.evidence_chars)
        if len(value) <= remaining:
            self.evidence_chars += len(value)
            return value
        self.evidence_chars = max_evidence_chars
        self.evidence_truncated = True
        self.record_outcome("EVIDENCE_CHAR_LIMIT")
        return value[:remaining] + "\nREASON: EVIDENCE_CHAR_LIMIT"

    def record_outcome(self, reason: str) -> None:
        if reason not in self.outcomes:
            self.outcomes.append(reason)

    def record_tool_round(self, tool_names: list[str]) -> None:
        self.tool_rounds_used += 1
        self.final_tool_names = list(tool_names)

    def record_tool_failure(self, tool_name: str) -> None:
        if tool_name not in self.failed_tools:
            self.failed_tools.append(tool_name)

    def record_tool_blocked(self, tool_name: str) -> None:
        if tool_name not in self.blocked_tools:
            self.blocked_tools.append(tool_name)

    def record_tool_insufficient(self, tool_name: str) -> None:
        if tool_name not in self.insufficient_tools:
            self.insufficient_tools.append(tool_name)

    def _record_tool_outcome_event(self, category: str, tool_name: str) -> None:
        self.tool_outcome_events[category] = (
            self.tool_outcome_events.get(category, 0) + 1
        )
        by_name = self.tool_outcome_events_by_name.setdefault(category, {})
        by_name[tool_name] = by_name.get(tool_name, 0) + 1

    def record_rejected_host(self, value: object) -> None:
        """Extract a REJECTED_HOST: line (e.g. from get_official_document's
        UNAPPROVED_DOCUMENT_HOST reply) so allowlist gaps are deterministically
        visible in the persisted artifact instead of depending on the LLM
        accurately paraphrasing the rejected host in its final synthesis."""
        match = _REJECTED_HOST_RE.search(str(value))
        if match and match.group(1) not in self.rejected_hosts:
            self.rejected_hosts.append(match.group(1))

    def record_tool_result(
        self,
        tool_name: str,
        value: object,
        *,
        blocked: bool = False,
        args: Mapping[str, Any] | None = None,
    ) -> None:
        """Classify typed tool outcomes without conflating missing data and faults."""
        text = str(value).strip().upper()
        if blocked or text.startswith("TOOL_BLOCKED:"):
            self.record_tool_blocked(tool_name)
        elif text.startswith("STATUS: INSUFFICIENT_DATA"):
            self.record_tool_insufficient(tool_name)
            self.record_rejected_host(value)
            reason_match = _REASON_RE.search(text)
            reason = reason_match.group(1) if reason_match else None
            if reason in {
                "DOCUMENT_DNS_FAILED",
                "GUIDANCE_EXTRACTION_FAILED",
                "GUIDANCE_EXTRACTION_AUTH_ERROR",
                "LOOKUP_TIMEOUT",
            }:
                self._record_tool_outcome_event(
                    "evidence_acquisition_failure", tool_name
                )
                self.tool_failures_by_name[tool_name] = (
                    self.tool_failures_by_name.get(tool_name, 0) + 1
                )
            else:
                self._record_tool_outcome_event("ordinary_insufficient", tool_name)
            if reason in {"DOCUMENT_DNS_FAILED", "UNAPPROVED_DOCUMENT_HOST"} and args:
                host = self._host(args)
                if host:
                    self.host_failures[host] = self.host_failures.get(host, 0) + 1
        elif text.startswith("TOOL_ERROR:"):
            self.record_tool_failure(tool_name)
            self._record_tool_outcome_event("execution_error", tool_name)
            self.tool_failures_by_name[tool_name] = (
                self.tool_failures_by_name.get(tool_name, 0) + 1
            )

    def record_forced_synthesis(self, reason: str = "TOOL_ROUND_LIMIT") -> None:
        self.forced_synthesis_used = True
        self.stop_reason = reason
        self.synthesis_evidence_chars = self.evidence_chars

    def record_model_final(self) -> None:
        self.stop_reason = "MODEL_FINAL"

    def record_repair_input(self, content: str) -> None:
        self.repair_input_chars = len(content)

    def telemetry(self) -> dict[str, object]:
        return {
            "policy": asdict(cast(Any, self.policy)),
            "tool_calls": dict(self.tool_calls),
            "llm_calls": self.llm_calls,
            "evidence_chars": self.evidence_chars,
            "evidence_truncated": self.evidence_truncated,
            "outcomes": list(self.outcomes),
            "tool_rounds_used": self.tool_rounds_used,
            "forced_synthesis_used": self.forced_synthesis_used,
            "stop_reason": self.stop_reason,
            "final_tool_names": list(self.final_tool_names),
            "failed_tools": list(self.failed_tools),
            "blocked_tools": list(self.blocked_tools),
            "insufficient_tools": list(self.insufficient_tools),
            "rejected_hosts": list(self.rejected_hosts),
            "synthesis_evidence_chars": self.synthesis_evidence_chars,
            "repair_input_chars": self.repair_input_chars,
            "purpose_calls": dict(self.purpose_calls),
            "call_signatures": dict(self.call_signatures),
            "host_failures": dict(self.host_failures),
            "tool_failures_by_name": dict(self.tool_failures_by_name),
            "blocked_reasons": dict(self.blocked_reasons),
            "tool_outcome_events": dict(self.tool_outcome_events),
            "tool_outcome_events_by_name": {
                category: dict(values)
                for category, values in self.tool_outcome_events_by_name.items()
            },
        }


class AuditorBudgetLedger(ResearchBudgetLedger):
    """Backward-compatible Auditor specialization of the shared ledger."""
