"""
Token usage tracking and cost estimation module.
Provides comprehensive logging of LLM token consumption across all agents.
"""

import re
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

# LLM pricing per 1M tokens, standard/interactive tier (July 2026).
# Sources: ai.google.dev/gemini-api/docs/pricing, developers.openai.com/api/docs/pricing,
# platform.claude.com/docs/en/pricing, docs.z.ai/guides/overview/pricing,
# api-docs.deepseek.com/quick_start/pricing.
# Matched by exact key first, then longest-prefix-wins (see
# ``_lookup_model_pricing``) — so insertion order does NOT matter and a
# more-specific variant (mini/lite/preview) always beats its parent regardless
# of where it appears. Gemini >200k-context rate differences are not modeled
# (single blended rate per model). Provider-specific cached-input rates use
# ``cached_prompt`` when present; otherwise they use CACHED_PROMPT_MULTIPLIER
# below.
MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    # Current-generation models only — retired/deprecated models are
    # deliberately absent; if one is somehow used it hits the default-pricing
    # fallback and logs unknown_model_pricing.
    # --- OpenAI gpt-5.x (consultant/auditor/editor; mini before parent) ---
    "gpt-5.6-sol": {"prompt": 5.00, "completion": 30.00},
    "gpt-5.6-terra": {"prompt": 2.50, "completion": 15.00},
    "gpt-5.6-luna": {"prompt": 1.00, "completion": 6.00},
    # Official alias for Sol; keep after the more-specific family variants.
    "gpt-5.6": {"prompt": 5.00, "completion": 30.00},
    "gpt-5.5": {"prompt": 5.00, "completion": 30.00},
    "gpt-5.4-mini": {"prompt": 0.75, "completion": 4.50},
    "gpt-5.4": {"prompt": 2.50, "completion": 15.00},
    # gpt-4o family: still the in-code fallback default for auditor/editor
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    # --- Gemini 3.x (paid tier) ---
    "gemini-3.6-flash": {"prompt": 1.50, "completion": 7.50},
    "gemini-3.5-flash": {"prompt": 1.50, "completion": 9.00},
    "gemini-3.1-flash-lite": {"prompt": 0.25, "completion": 1.50},
    # "gemini-3.1-pro" prefix also covers "-preview"
    "gemini-3.1-pro": {"prompt": 2.00, "completion": 12.00},
    "gemini-3-flash-preview": {"prompt": 0.50, "completion": 3.00},
    # "gemini-3-pro" prefix also covers "-preview"
    "gemini-3-pro": {"prompt": 2.00, "completion": 12.00},
    # --- Gemini 2.5 (paid tier; lite before parent) ---
    "gemini-2.5-flash-lite": {"prompt": 0.10, "completion": 0.40},
    "gemini-2.5-flash": {"prompt": 0.30, "completion": 2.50},
    "gemini-2.5-pro": {"prompt": 1.25, "completion": 10.00},
    # --- Anthropic (article writer) ---
    "claude-opus-4": {"prompt": 5.00, "completion": 25.00},
    "claude-sonnet-4": {"prompt": 3.00, "completion": 15.00},
    "claude-haiku-4": {"prompt": 1.00, "completion": 5.00},
    # --- Z.AI (APAC regional specialist) ---
    "glm-5.2": {"prompt": 1.40, "cached_prompt": 0.26, "completion": 4.40},
    # --- DeepSeek (APAC regional specialist) ---
    "deepseek-v4": {"prompt": 0.435, "completion": 0.87},
    # --- Moonshot Kimi (APAC regional specialist) ---
    # kimi-k3 has no separately published SKU yet; anchored to the latest
    # published Kimi (K2.6, Apr 2026): $0.95 cache-miss input / $0.16 cache-hit /
    # $4.00 output per 1M. CONFIRM against Moonshot's kimi-k3 rate once published.
    "kimi-k3": {"prompt": 0.95, "cached_prompt": 0.16, "completion": 4.00},
}

# Fallback for models missing from the table (Flash-class assumption).
# Every fallback hit logs unknown_model_pricing once — a silent fallback is
# how three months of 3-4x cost underreporting happened (July 2026).
DEFAULT_PRICING_PER_1M: dict[str, float] = {"prompt": 0.30, "completion": 2.50}

# Flex/batch service tiers are billed at 50% of standard rates on both
# Gemini and OpenAI (July 2026 published pricing).
FLEX_TIER_MULTIPLIER = 0.5

# Cached prompt-prefix tokens bill at 10% of the standard input rate on Gemini
# and OpenAI (verified 2026-07-03: ai.google.dev/gemini-api/docs/pricing context
# caching = input/10 for every 2.5/3.x model; developers.openai.com/api/docs/
# pricing cached input = input/10 for every gpt-5.x model). OpenAI documents
# that flex does NOT further discount cached input, so the cached portion is
# priced at this rate without the tier multiplier.
CACHED_PROMPT_MULTIPLIER = 0.10

# GPT-5.6 prices the full request at higher rates above this input threshold.
# Cache writes are a subset of input tokens and cost more than ordinary input.
GPT56_LONG_CONTEXT_INPUT_THRESHOLD = 272_000
GPT56_LONG_CONTEXT_PROMPT_MULTIPLIER = 2.0
GPT56_LONG_CONTEXT_COMPLETION_MULTIPLIER = 1.5
CACHE_WRITE_PROMPT_MULTIPLIER = 1.25

_warned_unknown_pricing_models: set[str] = set()


def _normalize_model_name(model_name: str) -> str:
    """Strip a single leading ``vendor/`` namespace before matching.

    OpenRouter/LiteLLM-style ids (``moonshot/kimi-k3``, ``google/gemini-3.6``)
    would otherwise miss the table. Native base-URL callers already pass bare
    names, so this is a defensive no-op for them.
    """
    return model_name.split("/", 1)[1] if "/" in model_name else model_name


def _match_pricing(model_name: str) -> dict[str, float] | None:
    """Resolve pricing by exact key, then longest-prefix-wins.

    Order-independent of ``MODEL_PRICING_PER_1M`` insertion order: a more
    specific variant (``gpt-5.4-mini``) always beats its parent (``gpt-5.4``)
    because it is the longer matching prefix. Returns ``None`` when unpriced.
    """
    name = _normalize_model_name(model_name)
    exact = MODEL_PRICING_PER_1M.get(name)
    if exact is not None:
        return exact
    best_key: str | None = None
    for model_key in MODEL_PRICING_PER_1M:
        if name.startswith(model_key) and (
            best_key is None or len(model_key) > len(best_key)
        ):
            best_key = model_key
    return MODEL_PRICING_PER_1M[best_key] if best_key is not None else None


def _is_model_priced(model_name: str) -> bool:
    """True when ``model_name`` resolves to a real table entry (no warning)."""
    return _match_pricing(model_name) is not None


def _provider_for_model(model_name: str) -> str:
    """Billing vendor for a model id, for the ``by_provider`` cost rollup.

    Grouped by who bills for the tokens (so "where does spend go" is answerable
    from model names alone) — coarser than the transport ``provider`` recorded
    in ``call_attempts``. Unknown families fall to ``"unknown"`` (a visible
    bucket that pairs with the unpriced-model flag), never silently misattributed.
    """
    name = _normalize_model_name(model_name).lower()
    if "gemini" in name:
        return "google"
    if "claude" in name:
        return "anthropic"
    if "deepseek" in name:
        return "deepseek"
    if name.startswith("kimi") or "moonshot" in name:
        return "moonshot"
    if name.startswith("glm") or "zhipu" in name:
        return "zhipu"
    if name.startswith(("gpt", "o1", "o3", "o4")) or "openai" in name:
        return "openai"
    return "unknown"


# The prompt-namespace ``agent_name`` (from prompts/*.json, recorded in
# ``call_attempts``) differs from the callback display label (recorded in the
# ``agents`` rollup) for four seats. Map them so the two ledgers join.
_AGENT_DISPLAY_NAME_MAP = {
    "External Consultant": "Consultant",
    "Global Forensic Accountant": "Global Forensic Auditor",
    "Bull Analyst": "Bull Researcher",
    "Bear Analyst": "Bear Researcher",
}

# Suffixes appended at call sites to decorate the base agent name (debate round,
# consultant/auditor final synthesis, high-thinking retry, escalation). This is a
# closed, code-controlled set — stripping it is deterministic, not free-text
# guessing.
_AGENT_SUFFIX_RE = re.compile(
    r"(?:_final_synthesis|\s+R\d+|\s+\(RETRY-HIGH\)|\s+Escalation|\s+Direct Retry)+$"
)


def canonical_display_name(raw: str) -> str:
    """Map a decorated prompt-namespace agent name to its ``agents``-rollup label.

    Strips the closed set of call-site suffixes, then applies the 4-entry
    spelling map — so ``"Bull Analyst R2"`` and ``"External Consultant_final_synthesis"``
    reconcile to ``"Bull Researcher"`` / ``"Consultant"``. Names already in the
    display namespace (e.g. ``"Portfolio Manager"``) pass through unchanged.
    """
    base = _AGENT_SUFFIX_RE.sub("", raw).strip()
    return _AGENT_DISPLAY_NAME_MAP.get(base, base)


def _lookup_model_pricing(model_name: str) -> dict[str, float]:
    prices = _match_pricing(model_name)
    if prices is not None:
        return prices
    if model_name not in _warned_unknown_pricing_models:
        _warned_unknown_pricing_models.add(model_name)
        logger.warning(
            "unknown_model_pricing",
            model=model_name,
            assumed_prompt_per_1m=DEFAULT_PRICING_PER_1M["prompt"],
            assumed_completion_per_1m=DEFAULT_PRICING_PER_1M["completion"],
            note="add this model to MODEL_PRICING_PER_1M in src/token_tracker.py",
        )
    return DEFAULT_PRICING_PER_1M


def _context_price_multipliers(
    model_name: str, prompt_tokens: int
) -> tuple[float, float]:
    if (
        model_name.startswith("gpt-5.6")
        and prompt_tokens > GPT56_LONG_CONTEXT_INPUT_THRESHOLD
    ):
        return (
            GPT56_LONG_CONTEXT_PROMPT_MULTIPLIER,
            GPT56_LONG_CONTEXT_COMPLETION_MULTIPLIER,
        )
    return 1.0, 1.0


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
    # Effective service tier for this call ("flex"/"standard"/"auto"/None).
    # Populated from provider response metadata (OpenAI echoes it natively;
    # the Gemini flex subclass stamps it) so flex calls price at flex rates
    # and fallback-to-standard calls price at full rates.
    service_tier: str | None = None
    # Prompt-prefix cache hits, a subset of prompt_tokens (both vendors count
    # cached tokens inside the reported prompt total).
    cached_prompt_tokens: int = 0
    # Cache-miss tokens written into the prompt cache, also inside prompt_tokens.
    cache_write_prompt_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        """
        Estimate cost in USD assuming paid tier (billing enabled on the account).

        Uses ``MODEL_PRICING_PER_1M`` (standard-tier rates, prefix-matched)
        with a 0.5x multiplier when this call ran on a flex tier. Cached prompt
        tokens bill at a model-specific rate when available, otherwise at
        ``CACHED_PROMPT_MULTIPLIER`` of the input rate. They are exempt from the
        flex multiplier (vendors don't stack the discounts).
        """
        model_pricing = _lookup_model_pricing(self.model_name)
        cached_tokens = min(max(self.cached_prompt_tokens, 0), self.prompt_tokens)
        cache_write_tokens = min(
            max(self.cache_write_prompt_tokens, 0),
            self.prompt_tokens - cached_tokens,
        )
        ordinary_prompt_tokens = self.prompt_tokens - cached_tokens - cache_write_tokens
        prompt_multiplier, completion_multiplier = _context_price_multipliers(
            self.model_name, self.prompt_tokens
        )
        prompt_cost = (
            (ordinary_prompt_tokens / 1_000_000)
            * model_pricing["prompt"]
            * prompt_multiplier
        )
        cache_write_cost = (
            (cache_write_tokens / 1_000_000)
            * model_pricing["prompt"]
            * CACHE_WRITE_PROMPT_MULTIPLIER
            * prompt_multiplier
        )
        cached_cost = (
            (cached_tokens / 1_000_000)
            * model_pricing.get(
                "cached_prompt",
                model_pricing["prompt"] * CACHED_PROMPT_MULTIPLIER,
            )
            * prompt_multiplier
        )
        completion_cost = (
            (self.completion_tokens / 1_000_000)
            * model_pricing["completion"]
            * completion_multiplier
        )
        multiplier = FLEX_TIER_MULTIPLIER if self.service_tier == "flex" else 1.0
        return (
            prompt_cost + cache_write_cost + completion_cost
        ) * multiplier + cached_cost


@dataclass
class AgentTokenStats:
    """Aggregate token statistics for a single agent."""

    agent_name: str
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cached_prompt_tokens: int = 0
    total_cache_write_prompt_tokens: int = 0
    total_cost_usd: float = 0.0
    wall_clock_seconds: float = 0.0
    wall_clock_max_seconds: float = 0.0
    calls: list[TokenUsage] = field(default_factory=list)

    def add_usage(self, usage: TokenUsage):
        """Add a token usage record to this agent's stats."""
        self.calls.append(usage)
        self.total_calls += 1
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_cached_prompt_tokens += usage.cached_prompt_tokens
        self.total_cache_write_prompt_tokens += usage.cache_write_prompt_tokens
        self.total_cost_usd += usage.estimated_cost_usd
        if usage.elapsed_seconds is not None:
            self.wall_clock_seconds += usage.elapsed_seconds
            if usage.elapsed_seconds > self.wall_clock_max_seconds:
                self.wall_clock_max_seconds = usage.elapsed_seconds

    def by_model(self) -> dict[str, dict[str, float]]:
        """Per-model cost/token breakdown for this agent.

        Folds the authoritative per-call ``TokenUsage`` records by model name —
        no re-pricing, so the rows sum to this agent's ``total_cost_usd``. Lets
        an agent that spans models (e.g. a quick→deep retry) show both.
        """
        out: dict[str, dict[str, float]] = {}
        for usage in self.calls:
            row = out.setdefault(
                usage.model_name, {"calls": 0, "tokens": 0, "cost_usd": 0.0}
            )
            row["calls"] += 1
            row["tokens"] += usage.total_tokens
            row["cost_usd"] += usage.estimated_cost_usd
        return out


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
    # Reconciled display-namespace name (joins to the ``agents`` rollup). Derived
    # from ``agent_name`` via ``canonical_display_name`` unless supplied.
    canonical_agent: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    failure_kind: str | None = None
    failure_origin: str | None = None
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
        service_tier: str | None = None,
        cached_prompt_tokens: int = 0,
        cache_write_prompt_tokens: int = 0,
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
            service_tier=service_tier,
            cached_prompt_tokens=cached_prompt_tokens,
            cache_write_prompt_tokens=cache_write_prompt_tokens,
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
                cached_prompt_tokens=cached_prompt_tokens or None,
                cache_write_prompt_tokens=cache_write_prompt_tokens or None,
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
        canonical_agent: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        failure_kind: str | None = None,
        failure_origin: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        """Record one provider call attempt, including failed attempts.

        Aggregate token totals remain driven by ``record_usage`` callbacks.
        This ledger exists for latency/retry diagnostics and may have null token
        fields when the provider did not return usage metadata.

        ``canonical_agent`` is the display-namespace identity (joins to the
        ``agents`` rollup). It is always run through ``canonical_display_name``
        (idempotent) — pass the undecorated agent name to skip suffix-stripping;
        omit it to derive from the (possibly decorated) ``agent_name``.
        """
        attempt_record = LLMCallAttempt(
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            provider=provider or "unknown",
            model_name=model_name or "unknown",
            status=status,
            attempt=attempt,
            elapsed_seconds=round(float(elapsed_seconds), 4),
            canonical_agent=canonical_display_name(canonical_agent or agent_name),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            failure_kind=failure_kind,
            failure_origin=failure_origin,
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
                        "failure_origin": failure_origin or "unknown",
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
            total_cached = sum(
                stats.total_cached_prompt_tokens for stats in self.agent_stats.values()
            )
            total_cache_write = sum(
                stats.total_cache_write_prompt_tokens
                for stats in self.agent_stats.values()
            )
            total_cost = sum(
                stats.total_cost_usd for stats in self.agent_stats.values()
            )

            # Provider/model rollups + unpriced-model detection, derived from the
            # authoritative per-call usages (so they reconcile to total_cost and
            # need no separate mutable state). A model that misses the pricing
            # table is surfaced here rather than only in a once-per-process log.
            by_provider: dict[str, dict[str, float]] = {}
            by_model: dict[str, dict[str, float]] = {}
            # Effective service tier of each call: "flex" (billed 0.5x) vs
            # "standard"/"auto" (full rate). When a flex-configured run can't get
            # flex capacity it falls back to standard, so this rollup makes the
            # variable "flex-unavailable → paid full rate" cost visible.
            by_tier: dict[str, dict[str, float]] = {}
            unpriced: set[str] = set()
            for usage in self.all_usages:
                if not _is_model_priced(usage.model_name):
                    unpriced.add(usage.model_name)
                for bucket, key in (
                    (by_model, usage.model_name),
                    (by_provider, _provider_for_model(usage.model_name)),
                    (by_tier, usage.service_tier or "unspecified"),
                ):
                    row = bucket.setdefault(
                        key, {"calls": 0, "tokens": 0, "cost_usd": 0.0}
                    )
                    row["calls"] += 1
                    row["tokens"] += usage.total_tokens
                    row["cost_usd"] += usage.estimated_cost_usd

            return {
                "failed_attempts": len(self.failed_attempts),
                "total_calls": len(self.all_usages),
                "total_agents": len(self.agent_stats),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "total_cached_prompt_tokens": total_cached,
                "total_cache_write_prompt_tokens": total_cache_write,
                "total_cost_usd": total_cost,
                "session_start": self.session_start,
                "agents": {
                    name: {
                        "calls": stats.total_calls,
                        "prompt_tokens": stats.total_prompt_tokens,
                        "completion_tokens": stats.total_completion_tokens,
                        "total_tokens": stats.total_tokens,
                        "cached_prompt_tokens": stats.total_cached_prompt_tokens,
                        "cache_write_prompt_tokens": (
                            stats.total_cache_write_prompt_tokens
                        ),
                        "cost_usd": stats.total_cost_usd,
                        "by_model": stats.by_model(),
                        "wall_clock_seconds": round(stats.wall_clock_seconds, 4),
                        "wall_clock_max_seconds": round(
                            stats.wall_clock_max_seconds, 4
                        ),
                    }
                    for name, stats in self.agent_stats.items()
                },
                "by_provider": by_provider,
                "by_model": by_model,
                "by_tier": by_tier,
                "unpriced_models": sorted(unpriced),
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
                "slowest_agents": self._slowest_agents_top(3),
                "timeout_seconds_lost": 0.0,
                "consultant_timeout": False,
                "failed_by_agent": {},
                "failed_by_provider": self._count_failures("provider"),
                "failed_by_kind": self._count_failures("failure_kind"),
                "failed_by_origin": self._count_failures("failure_origin"),
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
            # Key on the canonical name so retry/round-suffixed variants of one
            # seat (e.g. "Bull Analyst R1"/"R2") aggregate together.
            key = attempt.canonical_agent or attempt.agent_name
            failed_by_agent[key] = failed_by_agent.get(key, 0) + 1
        return {
            "total_attempts": len(self.call_attempts),
            "successful_attempts": sum(
                1 for attempt in self.call_attempts if attempt.status == "success"
            ),
            "failed_attempts": len(self.failed_attempts),
            "slowest_call": asdict(slowest),
            "slowest_agents": self._slowest_agents_top(3),
            "timeout_seconds_lost": round(timeout_seconds_lost, 4),
            "consultant_timeout": any(
                attempt.failure_kind == "timeout"
                and attempt.canonical_agent == "Consultant"
                for attempt in failures
            ),
            "failed_by_agent": failed_by_agent,
            "failed_by_provider": self._count_failures("provider"),
            "failed_by_kind": self._count_failures("failure_kind"),
            "failed_by_origin": self._count_failures("failure_origin"),
        }

    def _slowest_agents_top(self, limit: int) -> list[dict[str, Any]]:
        """Top-N agents by total wall-clock seconds (callback-measured).

        Complements ``slowest_call`` (which is one attempt) with cumulative
        per-agent latency. Agents with no wall-clock data are skipped.
        """
        candidates = [
            (name, stats)
            for name, stats in self.agent_stats.items()
            if stats.wall_clock_seconds > 0
        ]
        candidates.sort(key=lambda item: -item[1].wall_clock_seconds)
        return [
            {
                "agent_name": name,
                "wall_clock_seconds": round(stats.wall_clock_seconds, 4),
                "wall_clock_max_seconds": round(stats.wall_clock_max_seconds, 4),
                "calls": stats.total_calls,
            }
            for name, stats in candidates[:limit]
        ]

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

        if stats["unpriced_models"]:
            logger.warning(
                "unpriced_models_detected",
                models=stats["unpriced_models"],
                note="cost fabricated at default rate — add to MODEL_PRICING_PER_1M",
            )

        logger.info(
            "token_usage_summary",
            session_start=stats["session_start"],
            total_calls=stats["total_calls"],
            failed_attempts=stats["failed_attempts"],
            total_agents=stats["total_agents"],
            total_prompt_tokens=stats["total_prompt_tokens"],
            total_completion_tokens=stats["total_completion_tokens"],
            total_tokens=stats["total_tokens"],
            total_cost_usd=round(stats["total_cost_usd"], 4),
            failed_by_provider=stats.get("failed_by_provider") or None,
            failed_by_kind=stats.get("failed_by_kind") or None,
        )

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

        logger.debug(
            "token_usage_agent_breakdown_start", agent_count=len(sorted_agents)
        )
        for agent_name, agent_stats in sorted_agents:
            logger.debug(
                "token_usage_agent_breakdown",
                agent=agent_name,
                calls=agent_stats["calls"],
                prompt_tokens=agent_stats["prompt_tokens"],
                completion_tokens=agent_stats["completion_tokens"],
                total_tokens=agent_stats["total_tokens"],
                cost_usd=round(agent_stats["cost_usd"], 4),
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

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Drop the start timestamp on error so failed runs don't leak entries."""
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        self._run_starts.pop(key, None)

    @staticmethod
    def _extract_service_tier(response: LLMResult) -> str | None:
        """Read the effective service tier off a provider response.

        OpenAI echoes the billed tier in ``llm_output["service_tier"]``
        (langchain-openai passes it through); the Gemini flex subclass stamps
        both ``llm_output`` and each message's ``response_metadata``. A
        fallback-to-standard call therefore reports "standard"/"auto" and is
        priced at full rates.
        """
        if response.llm_output:
            tier = response.llm_output.get("service_tier")
            if isinstance(tier, str):
                return tier
        if response.generations and response.generations[0]:
            message = getattr(response.generations[0][0], "message", None)
            resp_meta = getattr(message, "response_metadata", None)
            if isinstance(resp_meta, dict):
                tier = resp_meta.get("service_tier")
                if isinstance(tier, str):
                    return tier
        return None

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

        service_tier = self._extract_service_tier(response)

        usage = extract_token_usage_breakdown(response)
        prompt_tokens = usage.input_tokens or 0
        completion_tokens = usage.total_output_tokens or 0
        cached_prompt_tokens = usage.cached_input_tokens or 0
        cache_write_prompt_tokens = usage.cache_write_input_tokens or 0

        if prompt_tokens > 0 or completion_tokens > 0:
            self.tracker.record_usage(
                agent_name=self.agent_name,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed_seconds=elapsed_seconds,
                service_tier=service_tier,
                cached_prompt_tokens=cached_prompt_tokens,
                cache_write_prompt_tokens=cache_write_prompt_tokens,
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
