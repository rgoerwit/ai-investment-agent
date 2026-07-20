"""Single-source budgets and telemetry for the forensic Auditor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.config import config


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

    def telemetry(self) -> dict[str, object]:
        return {
            "policy": asdict(self.policy),
            "tool_calls": dict(self.tool_calls),
            "llm_calls": self.llm_calls,
            "evidence_chars": self.evidence_chars,
            "evidence_truncated": self.evidence_truncated,
            "outcomes": list(self.outcomes),
        }
