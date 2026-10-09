"""The ticker run deadline: the pipeline watchdog's ceiling, seen from the child.

``scripts/run_pipeline.sh`` runs each ticker under a watchdog that signals the
child once the stage's per-ticker timeout is exceeded. Before October 2026 the
child knew nothing about it: every in-process budget was sized as if time were
unlimited, so a retry ladder near the end of the allowance ran on until the
external SIGTERM discarded the whole ticker.

The contract, end to end (``docs/CODEBASE_MEMORY.md``):

1. The shell sets ``PIPELINE_TICKER_DEADLINE_EPOCH`` (wall-clock seconds) at the
   moment it starts the child, from the same timeout its watchdog counts.
2. ``run_deadline()`` converts it **once** to a monotonic deadline, minus
   ``publication_reserve_seconds``.
3. ``src/main.py`` binds that deadline (``enclosing_deadline_at``) around the
   analysis -- the ticker retrospective and the graph run -- and not around
   final publication, which runs on the reserve.
4. Every node deadline inside becomes ``min(node, run)``; LLM admission refuses
   a request that cannot fit, and operational clamps shrink to what is left.

It caps work; it does not promise completion. ``log_deadline_ladder`` says once
at startup when the configured retry ladder cannot fit the allowance.

The epoch is read with ``get_env_value`` (shell environment first, then
``.env``), the repository's sanctioned accessor for a variable looked up by
name. The shell always sets it, so its value wins; an operator has no reason to
put it in ``.env``, and a stale value there is visible as a spent deadline.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.seats import SEATS, SeatId

logger = structlog.get_logger(__name__)

RUN_DEADLINE_EPOCH_ENV = "PIPELINE_TICKER_DEADLINE_EPOCH"

_LOGGED: set[str] = set()


def _settings(cfg: Any) -> Any:
    if cfg is not None:
        return cfg
    import src.config as config_module

    return config_module.config


def _log_once(key: str, method: str, event: str, **fields: Any) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    getattr(logger, method)(event, **fields)


def run_deadline(
    cfg: Any = None,
    *,
    wall_clock: Callable[[], float] = time.time,
    clock: Callable[[], float] | None = None,
) -> float | None:
    """The run's absolute monotonic analysis deadline, or None when there is none.

    ``epoch - reserve`` translated onto the monotonic clock every deadline here
    is measured on. A missing value (a manual run) or a malformed one means no
    run deadline; each is logged once and never raises. An epoch already past
    yields a deadline already spent: admission then refuses the first LLM call
    and the run fails closed, unpublished.
    """
    from src.config import get_env_value

    raw = get_env_value(RUN_DEADLINE_EPOCH_ENV)
    if raw is None or not raw.strip():
        _log_once("absent", "debug", "run_deadline_absent", env=RUN_DEADLINE_EPOCH_ENV)
        return None
    try:
        epoch = float(raw.strip())
    except ValueError:
        epoch = math.nan
    if not math.isfinite(epoch) or epoch <= 0:
        _log_once(
            "malformed",
            "warning",
            "run_deadline_malformed",
            env=RUN_DEADLINE_EPOCH_ENV,
            value=raw.strip()[:40],
            action="running without a run deadline",
        )
        return None
    reserve = max(0.0, float(_settings(cfg).publication_reserve_seconds))
    remaining = epoch - wall_clock()
    deadline = (clock or llm_timeouts.now)() + remaining - reserve
    _log_once(
        "bound",
        "info",
        "run_deadline_bound",
        watchdog_remaining_seconds=round(remaining, 1),
        publication_reserve_seconds=reserve,
        analysis_remaining_seconds=round(remaining - reserve, 1),
    )
    return deadline


# The graph's critical path, as stages of LLM seats. A stage's branches run in
# parallel (its cost is the slowest branch); a branch's seats run in sequence.
# Topology only -- every number comes from the seat's resolved BudgetPolicy.
# Tool loops, optional review seats and non-LLM nodes are left out, so the sum
# is a lower bound on the worst case, not a promise of completion.
_ANALYST_BRANCHES: tuple[tuple[SeatId, ...], ...] = (
    (SeatId.MARKET,),
    (SeatId.SENTIMENT,),
    (SeatId.NEWS,),
    (SeatId.FOREIGN_LANGUAGE,),
    (SeatId.LEGAL_COUNSEL,),
    (SeatId.VALUE_TRAP,),
    (SeatId.JUNIOR_FUNDAMENTALS, SeatId.SENIOR_FUNDAMENTALS),
)
_DEBATE_BRANCHES: tuple[tuple[SeatId, ...], ...] = ((SeatId.BULL,), (SeatId.BEAR,))
_TAIL_STAGES: tuple[tuple[str, tuple[tuple[SeatId, ...], ...]], ...] = (
    ("research_manager", ((SeatId.RESEARCH_MANAGER,),)),
    ("trader", ((SeatId.TRADER,),)),
    ("risk_debate", ((SeatId.RISKY,), (SeatId.SAFE,), (SeatId.NEUTRAL,))),
    ("portfolio_manager", ((SeatId.PORTFOLIO_MANAGER,),)),
)


@dataclass(frozen=True)
class CriticalPath:
    """Worst-case seconds along the critical path, per stage and in total."""

    quick_mode: bool
    debate_rounds: int
    stages: tuple[tuple[str, float], ...]
    single_attempt_stages: tuple[tuple[str, float], ...]

    @property
    def with_retries_seconds(self) -> float:
        return sum(seconds for _, seconds in self.stages)

    @property
    def single_attempt_seconds(self) -> float:
        return sum(seconds for _, seconds in self.single_attempt_stages)


def _seat_costs(
    cfg: Any, runtime_config: Any, quick: bool
) -> dict[SeatId, tuple[float, float]]:
    """seat -> (one invocation cap, cap x total attempts), per seat policy."""
    plan = None
    try:
        from src.llm_runtime.bindings import resolve_binding_plan

        plan = resolve_binding_plan(cfg)
    except Exception:  # noqa: BLE001 - a diagnostic must never stop a run
        plan = None
    costs: dict[SeatId, tuple[float, float]] = {}
    for seat_id in SeatId:
        provider: str | None = None
        if plan is not None:
            try:
                if not plan.status_for(seat_id, quick_mode=quick).enabled:
                    continue
                provider = plan.for_seat(seat_id, quick_mode=quick).provider
            except Exception:  # noqa: BLE001 - unknown provider: generic policy
                provider = None
        policy = llm_timeouts.policy_for_seat(
            SEATS[seat_id], provider=provider, cfg=cfg, runtime_config=runtime_config
        )
        costs[seat_id] = (
            policy.invocation_cap,
            policy.invocation_cap * policy.attempts,
        )
    return costs


def _stage_cost(
    branches: tuple[tuple[SeatId, ...], ...],
    costs: dict[SeatId, tuple[float, float]],
    index: int,
) -> float:
    return max(
        (
            sum(costs[seat][index] for seat in branch if seat in costs)
            for branch in branches
        ),
        default=0.0,
    )


def critical_path(cfg: Any = None, runtime_config: Any = None) -> CriticalPath:
    """Analyst branches as their max, then the debate, the tail seats and the PM."""
    settings = _settings(cfg)
    runtime = llm_timeouts._runtime(settings, runtime_config)
    quick = bool(getattr(runtime, "quick_mode_active", False))
    # Mirrors run_analysis: quick mode runs one debate round.
    rounds = 1 if quick else int(settings.max_debate_rounds)
    costs = _seat_costs(settings, runtime, quick)
    stages: list[tuple[str, float]] = []
    singles: list[tuple[str, float]] = []
    layout = (
        ("analysts", _ANALYST_BRANCHES, 1),
        ("debate", _DEBATE_BRANCHES, rounds),
        *((name, branches, 1) for name, branches in _TAIL_STAGES),
    )
    for name, branches, repeat in layout:
        singles.append((name, repeat * _stage_cost(branches, costs, 0)))
        stages.append((name, repeat * _stage_cost(branches, costs, 1)))
    return CriticalPath(
        quick_mode=quick,
        debate_rounds=rounds,
        stages=tuple(stages),
        single_attempt_stages=tuple(singles),
    )


def log_deadline_ladder(
    deadline: float | None,
    cfg: Any = None,
    runtime_config: Any = None,
    *,
    clock: Callable[[], float] | None = None,
) -> CriticalPath | None:
    """Log ``deadline_ladder_truncated`` once when the retry ladder cannot fit.

    Compares the critical path at full retries with what is left before the
    run deadline. Returns the path it computed (None without a run deadline).
    Never raises: it is a diagnostic.
    """
    if deadline is None:
        return None
    try:
        settings = _settings(cfg)
        path = critical_path(settings, runtime_config)
        available = deadline - (clock or llm_timeouts.now)()
        if path.with_retries_seconds > available:
            _log_once(
                "ladder",
                "info",
                "deadline_ladder_truncated",
                quick_mode=path.quick_mode,
                available_seconds=round(available, 1),
                critical_path_seconds=round(path.with_retries_seconds, 1),
                single_attempt_path_seconds=round(path.single_attempt_seconds, 1),
                stages={name: round(seconds, 1) for name, seconds in path.stages},
                debate_rounds=path.debate_rounds,
                stage_watchdog_seconds=(
                    settings.stage1_ticker_timeout_seconds
                    if path.quick_mode
                    else settings.stage2_ticker_timeout_seconds
                ),
                publication_reserve_seconds=settings.publication_reserve_seconds,
                note=(
                    "admission truncates the retry ladder at the run deadline; "
                    "completion is not promised"
                ),
            )
        return path
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never stop a run
        from src.error_safety import summarize_exception

        logger.warning(
            "deadline_ladder_diagnostic_failed",
            **summarize_exception(exc, operation="critical path diagnostic"),
        )
        return None
