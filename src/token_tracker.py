"""
Token usage tracking and cost estimation module.
Provides comprehensive logging of LLM token consumption across all agents.
"""

import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.llm_usage import extract_token_usage_breakdown

logger = structlog.get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage data for a single LLM call."""

    timestamp: str
    agent_name: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float | None = None

    @property
    def estimated_cost_usd(self) -> float:
        """
        Estimate cost in USD assuming paid tier (GCP project with billing enabled).

        IMPORTANT: If your GCP project has billing enabled, ALL API calls cost money,
        regardless of model name or "free tier" marketing. These are paid tier rates.

        Updated for Dec 2025 pricing (sources: ai.google.dev/gemini-api/docs/pricing).
        """
        # LLM pricing (per 1M tokens)
        # IMPORTANT: Order matters! More specific models must come before general ones
        pricing = {
            # OpenAI GPT-4 models (used by consultant node)
            # Pricing as of Dec 2025: https://openai.com/api/pricing/
            # Note: gpt-4o-mini must come BEFORE gpt-4o due to prefix matching
            "gpt-4o-mini": {
                "prompt": 0.15,  # $0.15 per 1M input tokens
                "completion": 0.60,  # $0.60 per 1M output tokens
            },
            "gpt-4o": {
                "prompt": 2.50,  # $2.50 per 1M input tokens
                "completion": 10.00,  # $10.00 per 1M output tokens
            },
            "gpt-4-turbo": {
                "prompt": 10.00,  # $10.00 per 1M input tokens
                "completion": 30.00,  # $30.00 per 1M output tokens
            },
            "gpt-4": {
                "prompt": 30.00,  # $30.00 per 1M input tokens
                "completion": 60.00,  # $60.00 per 1M output tokens
            },
            # Gemini pricing - PAID TIER RATES
            # NOTE: These apply when billing is enabled on your GCP project
            # Gemini 2.0 Flash variants (experimental - but PAID if billing enabled)
            "gemini-2.0-flash-thinking-exp": {
                "prompt": 0.30,  # Paid tier: $0.30 per 1M input tokens
                "completion": 2.50,  # Paid tier: $2.50 per 1M output tokens
            },
            "gemini-2.0-flash-exp": {
                "prompt": 0.30,  # Paid tier: $0.30 per 1M input tokens
                "completion": 2.50,  # Paid tier: $2.50 per 1M output tokens
            },
            # Gemini 2.5 Flash variants (more specific must come first!)
            "gemini-2.5-flash-lite": {
                "prompt": 0.10,  # $0.10 per 1M input tokens
                "completion": 0.40,  # $0.40 per 1M output tokens
            },
            "gemini-2.5-flash": {
                "prompt": 0.30,  # $0.30 per 1M input tokens
                "completion": 2.50,  # $2.50 per 1M output tokens
            },
            # Gemini 3 Pro variants
            "gemini-3-pro-preview": {
                "prompt": 2.00,  # $2.00 per 1M input tokens
                "completion": 12.00,  # $12.00 per 1M output tokens
            },
            "gemini-3-pro": {
                "prompt": 2.00,  # $2.00 per 1M input tokens (< 200k context)
                "completion": 12.00,  # $12.00 per 1M output tokens (< 200k context)
            },
        }

        # Default pricing for unknown models (assume Flash-level pricing)
        default_pricing = {"prompt": 0.30, "completion": 2.50}

        # Find pricing for this model (match by prefix)
        model_pricing = default_pricing
        for model_key, prices in pricing.items():
            if self.model_name.startswith(model_key):
                model_pricing = prices
                break

        prompt_cost = (self.prompt_tokens / 1_000_000) * model_pricing["prompt"]
        completion_cost = (self.completion_tokens / 1_000_000) * model_pricing[
            "completion"
        ]

        return prompt_cost + completion_cost


@dataclass
class AgentTokenStats:
    """Aggregate token statistics for a single agent."""

    agent_name: str
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: list[TokenUsage] = field(default_factory=list)

    def add_usage(self, usage: TokenUsage):
        """Add a token usage record to this agent's stats."""
        self.calls.append(usage)
        self.total_calls += 1
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost_usd += usage.estimated_cost_usd


@dataclass
class LLMCallAttempt:
    """Diagnostic ledger entry for one provider call attempt."""

    timestamp: str
    agent_name: str
    provider: str
    model_name: str
    status: Literal["success", "failure"]
    attempt: int
    elapsed_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    failure_kind: str | None = None
    retryable: bool | None = None


class TokenTracker:
    """
    Global token tracker that aggregates usage across all agents.
    Thread-safe singleton for tracking LLM token consumption.
    """

    _instance: "TokenTracker | None" = None
    _instance_lock = threading.Lock()
    _quiet_mode: bool = False

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        with self._instance_lock:
            if self._initialized:
                return

            self._initialized = True
            self._lock = threading.RLock()
            self.agent_stats: dict[str, AgentTokenStats] = {}
            self.all_usages: list[TokenUsage] = []
            self.failed_attempts: list[dict[str, str]] = []
            self.call_attempts: list[LLMCallAttempt] = []
            self.session_start = datetime.now().isoformat()

        if not self._quiet_mode:
            logger.debug("token_tracker_initialized", session_start=self.session_start)

    @classmethod
    def set_quiet_mode(cls, quiet: bool = True):
        """Enable or disable quiet mode to suppress logging."""
        cls._quiet_mode = quiet

    def record_usage(
        self,
        agent_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed_seconds: float | None = None,
    ):
        """Record token usage for a specific agent."""
        usage = TokenUsage(
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_seconds=elapsed_seconds,
        )

        with self._lock:
            # Add to agent-specific stats
            if agent_name not in self.agent_stats:
                self.agent_stats[agent_name] = AgentTokenStats(agent_name=agent_name)

            self.agent_stats[agent_name].add_usage(usage)
            self.all_usages.append(usage)

        if not self._quiet_mode:
            logger.debug(
                "token_usage_recorded",
                agent=agent_name,
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost_usd=f"${usage.estimated_cost_usd:.6f}",
                elapsed_seconds=elapsed_seconds,
            )

    def record_call_attempt(
        self,
        *,
        agent_name: str,
        provider: str,
        model_name: str,
        status: Literal["success", "failure"],
        attempt: int,
        elapsed_seconds: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        failure_kind: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        """Record one provider call attempt, including failed attempts.

        Aggregate token totals remain driven by ``record_usage`` callbacks.
        This ledger exists for latency/retry diagnostics and may have null token
        fields when the provider did not return usage metadata.
        """
        attempt_record = LLMCallAttempt(
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            provider=provider or "unknown",
            model_name=model_name or "unknown",
            status=status,
            attempt=attempt,
            elapsed_seconds=round(float(elapsed_seconds), 4),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            failure_kind=failure_kind,
            retryable=retryable,
        )
        with self._lock:
            self.call_attempts.append(attempt_record)
            if status == "failure":
                self.failed_attempts.append(
                    {
                        "agent_name": agent_name,
                        "provider": provider or "unknown",
                        "failure_kind": failure_kind or "unknown",
                        "model_name": model_name or "",
                        "elapsed_seconds": f"{attempt_record.elapsed_seconds:.4f}",
                    }
                )

    def get_agent_stats(self, agent_name: str) -> AgentTokenStats | None:
        """Get statistics for a specific agent."""
        with self._lock:
            return self.agent_stats.get(agent_name)

    def get_total_stats(self) -> dict[str, Any]:
        """Get aggregate statistics across all agents."""
        with self._lock:
            total_prompt = sum(
                stats.total_prompt_tokens for stats in self.agent_stats.values()
            )
            total_completion = sum(
                stats.total_completion_tokens for stats in self.agent_stats.values()
            )
            total_cost = sum(
                stats.total_cost_usd for stats in self.agent_stats.values()
            )

            return {
                "failed_attempts": len(self.failed_attempts),
                "total_calls": len(self.all_usages),
                "total_agents": len(self.agent_stats),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "total_cost_usd": total_cost,
                "session_start": self.session_start,
                "agents": {
                    name: {
                        "calls": stats.total_calls,
                        "prompt_tokens": stats.total_prompt_tokens,
                        "completion_tokens": stats.total_completion_tokens,
                        "total_tokens": stats.total_tokens,
                        "cost_usd": stats.total_cost_usd,
                    }
                    for name, stats in self.agent_stats.items()
                },
                "failed_by_provider": self._count_failures("provider"),
                "failed_by_kind": self._count_failures("failure_kind"),
                "call_attempts": [asdict(attempt) for attempt in self.call_attempts],
                "call_diagnostics": self._call_diagnostics(),
            }

    def _call_diagnostics(self) -> dict[str, Any]:
        if not self.call_attempts:
            return {
                "total_attempts": 0,
                "successful_attempts": 0,
                "failed_attempts": len(self.failed_attempts),
                "slowest_call": None,
                "timeout_seconds_lost": 0.0,
                "consultant_timeout": False,
                "failed_by_agent": {},
                "failed_by_provider": self._count_failures("provider"),
                "failed_by_kind": self._count_failures("failure_kind"),
            }

        failures = [
            attempt for attempt in self.call_attempts if attempt.status == "failure"
        ]
        slowest = max(self.call_attempts, key=lambda attempt: attempt.elapsed_seconds)
        timeout_seconds_lost = sum(
            attempt.elapsed_seconds
            for attempt in failures
            if attempt.failure_kind == "timeout"
        )
        failed_by_agent: dict[str, int] = {}
        for attempt in failures:
            failed_by_agent[attempt.agent_name] = (
                failed_by_agent.get(attempt.agent_name, 0) + 1
            )
        return {
            "total_attempts": len(self.call_attempts),
            "successful_attempts": sum(
                1 for attempt in self.call_attempts if attempt.status == "success"
            ),
            "failed_attempts": len(self.failed_attempts),
            "slowest_call": asdict(slowest),
            "timeout_seconds_lost": round(timeout_seconds_lost, 4),
            "consultant_timeout": any(
                attempt.failure_kind == "timeout"
                and "consultant" in attempt.agent_name.lower()
                for attempt in failures
            ),
            "failed_by_agent": failed_by_agent,
            "failed_by_provider": self._count_failures("provider"),
            "failed_by_kind": self._count_failures("failure_kind"),
        }

    def get_top_spenders(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return top spenders sorted by cost, then total tokens, then name."""
        if limit <= 0:
            return []

        stats = self.get_total_stats()
        sorted_agents = sorted(
            stats["agents"].items(),
            key=lambda item: (
                -item[1].get("cost_usd", 0.0),
                -item[1].get("total_tokens", 0),
                item[0],
            ),
        )

        return [
            {
                "agent": agent_name,
                "calls": agent_stats.get("calls", 0),
                "prompt_tokens": agent_stats.get("prompt_tokens", 0),
                "completion_tokens": agent_stats.get("completion_tokens", 0),
                "total_tokens": agent_stats.get("total_tokens", 0),
                "cost_usd": agent_stats.get("cost_usd", 0.0),
            }
            for agent_name, agent_stats in sorted_agents[:limit]
        ]

    def _count_failures(self, key: str) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for failure in self.failed_attempts:
                value = failure.get(key, "unknown")
                counts[value] = counts.get(value, 0) + 1
            return counts

    def record_failure(
        self, *, agent_name: str, provider: str, failure_kind: str, model_name: str = ""
    ) -> None:
        with self._lock:
            self.failed_attempts.append(
                {
                    "agent_name": agent_name,
                    "provider": provider or "unknown",
                    "failure_kind": failure_kind or "unknown",
                    "model_name": model_name,
                }
            )
        if not self._quiet_mode:
            logger.info(
                "llm_failure_recorded",
                agent=agent_name,
                provider=provider or "unknown",
                failure_kind=failure_kind or "unknown",
                model=model_name,
            )

    def reset(self):
        """Reset all tracking data (useful for new analysis runs)."""
        with self._lock:
            self.agent_stats.clear()
            self.all_usages.clear()
            self.failed_attempts.clear()
            self.call_attempts.clear()
            self.session_start = datetime.now().isoformat()
        if not self._quiet_mode:
            logger.debug("token_tracker_reset", session_start=self.session_start)

    def print_summary(self, *, ticker: str | None = None):
        """Print a formatted summary of token usage to logger."""
        if self._quiet_mode:
            return

        stats = self.get_total_stats()
        top_spenders = self.get_top_spenders()

        logger.info("TOKEN USAGE SUMMARY")
        logger.info(f"Session Start: {stats['session_start']}")
        logger.info(f"Total LLM Calls: {stats['total_calls']}")
        logger.info(f"Failed LLM Attempts: {stats['failed_attempts']}")
        logger.info(f"Total Agents: {stats['total_agents']}")
        logger.info(f"Total Prompt Tokens: {stats['total_prompt_tokens']:,}")
        logger.info(f"Total Completion Tokens: {stats['total_completion_tokens']:,}")
        logger.info(f"Total Tokens: {stats['total_tokens']:,}")
        logger.info(f"Projected Cost (Paid Tier): ${stats['total_cost_usd']:.4f} USD")
        logger.info(
            "  (Note: Actual cost = $0 if using free tier without billing enabled)"
        )

        if stats["failed_by_provider"]:
            logger.info(f"Failed by Provider: {stats['failed_by_provider']}")
        if stats["failed_by_kind"]:
            logger.info(f"Failed by Kind: {stats['failed_by_kind']}")

        logger.info(
            "analysis_cost_summary",
            ticker=ticker,
            total_calls=stats["total_calls"],
            total_prompt_tokens=stats["total_prompt_tokens"],
            total_completion_tokens=stats["total_completion_tokens"],
            total_tokens=stats["total_tokens"],
            total_cost_usd=round(stats["total_cost_usd"], 6),
            top_spenders=top_spenders,
        )

        # Sort agents by cost (descending)
        sorted_agents = sorted(
            stats["agents"].items(),
            key=lambda x: (-x[1]["cost_usd"], -x[1]["total_tokens"], x[0]),
        )

        logger.debug("Per-Agent Breakdown:")
        for agent_name, agent_stats in sorted_agents:
            logger.debug(
                f"\n{agent_name}:\n"
                f"  Calls: {agent_stats['calls']}\n"
                f"  Prompt Tokens: {agent_stats['prompt_tokens']:,}\n"
                f"  Completion Tokens: {agent_stats['completion_tokens']:,}\n"
                f"  Total Tokens: {agent_stats['total_tokens']:,}\n"
                f"  Cost: ${agent_stats['cost_usd']:.4f}"
            )


class TokenTrackingCallback(BaseCallbackHandler):
    """
    LangChain callback handler that tracks token usage per agent.
    Attach this to LLM instances to automatically log token consumption.
    """

    def __init__(
        self,
        agent_name: str,
        tracker: TokenTracker | None = None,
        output_token_cap: int | None = None,
    ):
        """
        Initialize callback with agent name.

        Args:
            agent_name: Name of the agent using this LLM
            tracker: Optional TokenTracker instance (uses singleton if not provided)
        """
        super().__init__()
        self.agent_name = agent_name
        self.tracker = tracker or TokenTracker()
        self.output_token_cap = output_token_cap
        self.api_output_token_cap = output_token_cap
        self.reasoning_reserve_tokens = 0
        self._run_starts: dict[str, float] = {}

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Remember per-run start time so callback usage records include latency."""
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        self._run_starts[key] = time.monotonic()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM completes a generation."""
        model_name = "unknown"
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        started = self._run_starts.pop(key, None)
        if started is None and run_id is None and len(self._run_starts) == 1:
            _, started = self._run_starts.popitem()
        elapsed_seconds = (
            round(time.monotonic() - started, 4) if started is not None else None
        )

        if response.generations and len(response.generations) > 0:
            first_generation_list = response.generations[0]
            if first_generation_list and len(first_generation_list) > 0:
                first_generation = first_generation_list[0]

                # Get model name from generation_info or response_metadata
                # CRITICAL: Check for None before calling .get() - hasattr returns True for None values
                gen_info = getattr(first_generation, "generation_info", None)
                if gen_info is not None:
                    model_name = gen_info.get("model_name", "unknown")

                if model_name == "unknown" and hasattr(first_generation, "message"):
                    resp_meta = getattr(
                        first_generation.message, "response_metadata", None
                    )
                    if resp_meta is not None:
                        model_name = resp_meta.get("model_name", "unknown")

        if response.llm_output and model_name == "unknown":
            model_name = response.llm_output.get("model_name", "unknown")

        usage = extract_token_usage_breakdown(response)
        prompt_tokens = usage.input_tokens or 0
        completion_tokens = usage.total_output_tokens or 0

        if prompt_tokens > 0 or completion_tokens > 0:
            self.tracker.record_usage(
                agent_name=self.agent_name,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed_seconds=elapsed_seconds,
            )
            if self.output_token_cap and completion_tokens > 0:
                intent_utilization = (
                    round(usage.visible_output_tokens / self.output_token_cap, 4)
                    if usage.visible_output_tokens is not None
                    else None
                )
                api_utilization = (
                    round(completion_tokens / self.api_output_token_cap, 4)
                    if self.api_output_token_cap
                    else None
                )
                log_method = logger.info
                utilization_values = [
                    value
                    for value in (intent_utilization, api_utilization)
                    if value is not None
                ]
                if not any(value >= 0.8 for value in utilization_values):
                    log_method = logger.debug
                log_method(
                    "llm_output_budget_usage",
                    agent=self.agent_name,
                    model=model_name,
                    configured_output_cap=self.output_token_cap,
                    configured_output_intent_cap=self.output_token_cap,
                    configured_api_output_cap=self.api_output_token_cap,
                    completion_tokens=completion_tokens,
                    completion_tokens_total=completion_tokens,
                    thinking_tokens=usage.thinking_tokens,
                    visible_output_tokens=usage.visible_output_tokens,
                    utilization_ratio=(
                        intent_utilization
                        if intent_utilization is not None
                        else api_utilization
                    ),
                    intent_utilization_ratio=intent_utilization,
                    api_utilization_ratio=api_utilization,
                )


# Global singleton instance (lazy initialization to respect quiet mode)
_global_tracker: TokenTracker | None = None


def get_tracker() -> TokenTracker:
    """Get the global TokenTracker singleton (lazy initialization)."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TokenTracker()
    return _global_tracker
