"""Single-source budgets and telemetry for the forensic Auditor."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from src.config import config

_REJECTED_HOST_RE = re.compile(r"(?m)^REJECTED_HOST:\s*(\S+)")


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


@dataclass
class AuditorBudgetLedger:
    policy: AuditorBudgetPolicy
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
        remaining = max(0, self.policy.max_evidence_chars - self.evidence_chars)
        if len(value) <= remaining:
            self.evidence_chars += len(value)
            return value
        self.evidence_chars = self.policy.max_evidence_chars
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
    ) -> None:
        """Classify typed tool outcomes without conflating missing data and faults."""
        text = str(value).strip().upper()
        if blocked or text.startswith("TOOL_BLOCKED:"):
            self.record_tool_blocked(tool_name)
        elif text.startswith("STATUS: INSUFFICIENT_DATA"):
            self.record_tool_insufficient(tool_name)
            self.record_rejected_host(value)
        elif text.startswith("TOOL_ERROR:"):
            self.record_tool_failure(tool_name)

    def record_forced_synthesis(self) -> None:
        self.forced_synthesis_used = True
        self.stop_reason = "TOOL_ROUND_LIMIT"
        self.synthesis_evidence_chars = self.evidence_chars

    def record_model_final(self) -> None:
        self.stop_reason = "MODEL_FINAL"

    def record_repair_input(self, content: str) -> None:
        self.repair_input_chars = len(content)

    def telemetry(self) -> dict[str, object]:
        return {
            "policy": asdict(self.policy),
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
        }
