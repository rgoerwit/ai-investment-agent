from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import structlog

from src.error_safety import summarize_exception
from src.ibkr.models import AnalysisRecord, ReconciliationItem
from src.ibkr.portfolio_defaults import (
    DEFAULT_REFRESH_CYCLE_WEIGHT,
    DEFAULT_REFRESH_DATA_QUALITY_BACKOFF_HOURS,
    DEFAULT_REFRESH_FAILURE_BACKOFF_HOURS,
    DEFAULT_REFRESH_URGENT_WEIGHT,
    DEFAULT_SELL_CONFIRMATION_MIN_SPACING_DAYS,
)
from src.ibkr.ticker import classify_ibkr_symbol
from src.ibkr.types import (
    AnalysisRunner,
    AnalysisSaver,
    CommandBuilder,
    ProgressCallback,
)

logger = structlog.get_logger(__name__)

RefreshPolicy = Literal["off", "blocking", "proactive"]
RefreshStream = Literal["urgent", "cycle"]

_OPERATOR_ONLY_BASES = frozenset(
    {"ENTRY_CONSTRAINT", "SPECIAL_SITUATION_REVIEW", "CAPITAL_ALLOCATION", "OVERWEIGHT"}
)
_SERVICE_PATTERN: tuple[RefreshStream, ...] = cast(
    tuple[RefreshStream, ...],
    ("urgent",) * DEFAULT_REFRESH_URGENT_WEIGHT
    + ("cycle",) * DEFAULT_REFRESH_CYCLE_WEIGHT,
)


def _is_research_ticker(ticker: str) -> bool:
    """Reject broker-only identifiers from durable scheduler state."""
    return classify_ibkr_symbol(ticker.split(".", 1)[0]).remedy == "use"


def parse_refresh_retry_at(raw: str | None) -> datetime | None:
    """Parse the scheduler's ISO retry timestamp, treating a naive value as UTC."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def format_refresh_retry_at(retry_at: datetime) -> str:
    """Render a retry timestamp for operators. One spelling, four call sites."""
    return retry_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


@dataclass(frozen=True)
class AnalysisFreshnessRow:
    display_ticker: str
    run_ticker: str
    bucket: str
    reason_family: str
    reason_text: str
    action: str
    action_basis: str | None
    urgency: str
    age_days: int | None
    expires_date: str | None
    days_until_due: int | None
    # Whether a paid stock analysis could plausibly change this row. Deliberately
    # a boolean and not an identity string: the previous design keyed scheduler
    # state on the buy-blocking flag *composition*, which is downstream of search
    # quality and changed on 6 of 6 consecutive runs for 2173.T, so the key never
    # repeated and the cooldown never fired. Nothing about prose, dates, or price
    # enters scheduler semantics.
    refresh_repairable: bool = False


@dataclass
class AnalysisFreshnessSummary:
    # Kept as the public compatibility field; it now means urgent analysis
    # refresh, never merely a portfolio REVIEW recommendation.
    blocking_now: list[AnalysisFreshnessRow] = field(default_factory=list)
    stale_in_queue: list[AnalysisFreshnessRow] = field(default_factory=list)
    due_soon: list[AnalysisFreshnessRow] = field(default_factory=list)
    candidate_blocked: list[AnalysisFreshnessRow] = field(default_factory=list)
    operator_review: list[AnalysisFreshnessRow] = field(default_factory=list)
    fresh: list[AnalysisFreshnessRow] = field(default_factory=list)
    # Display-only, and deliberately absent from plan()'s inputs: this run has
    # already produced the analysis the queue would ask for, so re-issuing its
    # command would contradict the report's own "Refreshed:" line.
    refreshed_this_run: list[AnalysisFreshnessRow] = field(default_factory=list)


@dataclass
class RefreshActivity:
    policy: RefreshPolicy
    limit: int
    queued: list[str] = field(default_factory=list)
    refreshed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    failed_retry_after: dict[str, str] = field(default_factory=dict)
    skipped_due_to_policy: list[str] = field(default_factory=list)
    skipped_due_to_limit: list[str] = field(default_factory=list)
    skipped_read_only: list[str] = field(default_factory=list)
    skipped_due_to_cooldown: list[str] = field(default_factory=list)
    # Failure backoff is ticker-wide because retrying the invocation cannot
    # succeed before its provider/persistence cooldown. The unrepaired backoff
    # is scoped to the action basis instead: a paid run that left the same basis
    # in place is evidence that the next one will too, but a genuinely different
    # basis (STOP_LOSS after DATA_QUALITY) is new information and stays eligible.
    skipped_due_to_failure_backoff: dict[str, str] = field(default_factory=dict)
    skipped_due_to_unrepaired: dict[str, str] = field(default_factory=dict)
    scheduled_bases: dict[str, str] = field(default_factory=dict)
    unrepaired_retry_after: dict[str, str] = field(default_factory=dict)
    # (ticker, WRR cursor position after dispatch). Attempts, successful or
    # failed, consume a quantum so one bad urgent ticker cannot monopolize runs.
    scheduled_slots: list[tuple[str, int]] = field(default_factory=list)
    scheduler_state_path: Path | None = None

    def copy(self) -> RefreshActivity:
        """Return an independent copy, sharing no mutable container.

        ``execute`` mutates its result, so it must not alias the planned
        activity. This replaced a hand-written field-by-field copy that omitted
        three fields and had to be extended for every new one — a silently
        shared list is exactly the sort of bug that copy was there to prevent.
        """
        return deepcopy(self)


@dataclass(frozen=True)
class RefreshPlanOptions:
    policy: RefreshPolicy
    limit: int
    show_recommendations: bool
    read_only: bool
    max_age_days: int
    ticker_subset: frozenset[str] | None = None
    scheduler_state_path: Path | None = None


@dataclass(frozen=True)
class RefreshExecutionOptions:
    quick_mode: bool
    skip_charts: bool = True


@dataclass(frozen=True, slots=True)
class _UnrepairedRefresh:
    """A paid refresh that completed and left the same action basis in place."""

    basis: str
    retry_after: str


@dataclass
class _SchedulerState:
    """Small durable state needed for fair service across separate CLI runs."""

    next_slot: int = 0
    retry_not_before: dict[str, str] = field(default_factory=dict)
    # One entry per ticker, not per condition: the question is "did the last
    # paid analysis change this ticker's disposition", which is a fact about the
    # position, not about which particular evidence gap happened to surface.
    unrepaired_refresh: dict[str, _UnrepairedRefresh] = field(default_factory=dict)


class AnalysisRefreshService:
    """Own portfolio freshness, eligibility, and bounded fair refreshes.

    A portfolio REVIEW is an operator decision, not an instruction to rerun an
    LLM. The service derives the narrower set of analyses that would add new
    information, then schedules urgent and normal-cycle candidates with a
    persistent weighted-round-robin cursor. The default 2:1 pattern guarantees
    a normal-cycle service opportunity in every three attempts when both streams
    remain non-empty. The cursor is persisted atomically after each attempt;
    it is deliberately not a long-held or distributed lock. Normal CLI use is
    sequential, so concurrent invocations may duplicate work but cannot corrupt
    the cursor file or block one another through a running analysis.
    """

    def resolve_policy(
        self,
        *,
        explicit_policy: RefreshPolicy | None,
        refresh_stale: bool,
        recommend: bool,
        read_only: bool,
    ) -> RefreshPolicy:
        if explicit_policy:
            return explicit_policy
        if refresh_stale:
            return "blocking"
        if recommend and not read_only:
            return "proactive"
        return "off"

    def classify(
        self,
        items: list[ReconciliationItem],
        *,
        max_age_days: int,
        already_refreshed: frozenset[str] = frozenset(),
    ) -> AnalysisFreshnessSummary:
        summary = AnalysisFreshnessSummary()
        for item in items:
            analysis = item.analysis
            expires_date, days_until_due = self._analysis_expiry_details(
                analysis, max_age_days
            )
            row = AnalysisFreshnessRow(
                display_ticker=item.ticker.ibkr,
                run_ticker=run_ticker_for(item),
                bucket="fresh",
                reason_family=self._reason_family(item, max_age_days),
                reason_text=item.reason,
                action=item.action,
                action_basis=item.action_basis,
                urgency=item.urgency,
                age_days=analysis.age_days if analysis else None,
                expires_date=expires_date,
                days_until_due=days_until_due,
                refresh_repairable=self._refresh_repairable(item),
            )

            if item.ibkr_position is None:
                if item.action == "REVIEW":
                    summary.candidate_blocked.append(
                        replace(row, bucket="candidate_blocked")
                    )
                else:
                    summary.fresh.append(row)
                continue
            if analysis is None:
                if item.action_basis == "DATA_QUALITY":
                    # Broker/operator defects remain non-repairable even when
                    # no saved stock analysis exists. The missing artifact is
                    # incidental; a paid run cannot fix the live position row.
                    summary.operator_review.append(
                        replace(row, bucket="operator_review")
                    )
                elif item.sell_type == "SOFT_REJECT":
                    # A correlated macro event can demote a historical sell
                    # to REVIEW after its artifact is no longer available.
                    # Keep that visible to the operator, but do not let it
                    # flood the urgent analysis queue with synthetic retries.
                    summary.operator_review.append(
                        replace(row, bucket="operator_review")
                    )
                else:
                    summary.blocking_now.append(replace(row, bucket="blocking_now"))
                continue
            if (
                item.action in {"SELL", "TRIM"} or item.sell_type == "SOFT_REJECT"
            ) and analysis.age_days > max_age_days:
                summary.stale_in_queue.append(replace(row, bucket="stale_in_queue"))
                continue
            if item.action == "REVIEW":
                self._classify_held_review(item, row, summary)
                continue
            if (
                item.action == "HOLD"
                and days_until_due is not None
                and days_until_due <= 7
            ):
                summary.due_soon.append(replace(row, bucket="due_soon"))
                continue
            summary.fresh.append(row)
        return self._withdraw_refreshed(summary, already_refreshed)

    @staticmethod
    def _withdraw_refreshed(
        summary: AnalysisFreshnessSummary, already_refreshed: frozenset[str]
    ) -> AnalysisFreshnessSummary:
        """Move rows this run already refreshed out of every command-bearing bucket.

        Applied as a post-pass rather than threaded through each branch: the
        classification logic is about evidence, and "we just ran it" is a fact
        about the run. Only buckets that render a rerun command are drained —
        ``operator_review`` and ``fresh`` carry no command and stay put.
        """
        if not already_refreshed:
            return summary
        for name in ("blocking_now", "stale_in_queue", "due_soon", "candidate_blocked"):
            bucket: list[AnalysisFreshnessRow] = getattr(summary, name)
            kept = [row for row in bucket if row.run_ticker not in already_refreshed]
            summary.refreshed_this_run.extend(
                replace(row, bucket="refreshed_this_run")
                for row in bucket
                if row.run_ticker in already_refreshed
            )
            setattr(summary, name, kept)
        return summary

    def plan(
        self,
        summary: AnalysisFreshnessSummary,
        *,
        options: RefreshPlanOptions,
    ) -> RefreshActivity:
        activity = RefreshActivity(
            policy=options.policy,
            limit=options.limit,
            scheduler_state_path=options.scheduler_state_path,
        )
        urgent = self._dedupe_rows(
            [*summary.blocking_now, *summary.stale_in_queue], options.ticker_subset
        )
        cycle_rows = [*summary.due_soon]
        if options.show_recommendations:
            cycle_rows.extend(summary.candidate_blocked)
        cycle = self._dedupe_rows(
            cycle_rows,
            options.ticker_subset,
            excluded={row.run_ticker for row in urgent},
        )
        state = self._load_state(options.scheduler_state_path)
        urgent, cycle = self._remove_backing_off(urgent, cycle, state, activity)

        if options.policy == "off":
            activity.skipped_due_to_policy = [
                *(row.run_ticker for row in urgent),
                *(row.run_ticker for row in cycle),
            ]
            return activity
        if options.policy == "blocking":
            # Strict priority, then top up. Passing only the urgent stream left
            # the rest of the budget unused: a portfolio with one permanently
            # urgent row and 64 due-soon rows refreshed exactly one analysis per
            # run and never reached normal-cycle work at all. Urgent still goes
            # first and still takes as many slots as it needs; "blocking" bounds
            # ordering, not throughput.
            selected, deferred = self._priority_plan(
                [*urgent, *cycle],
                options.limit,
                state.next_slot,
            )
        else:
            selected, deferred = self._weighted_fair_plan(
                urgent, cycle, options.limit, state.next_slot
            )
        activity.queued = [row.run_ticker for row, _ in selected]
        activity.scheduled_slots = [(row.run_ticker, slot) for row, slot in selected]
        activity.scheduled_bases = {
            row.run_ticker: row.action_basis
            for row, _ in selected
            if row.refresh_repairable and row.action_basis
        }
        activity.skipped_due_to_limit = [row.run_ticker for row in deferred]
        if options.read_only:
            activity.skipped_read_only = list(activity.queued)
            activity.queued = []
            activity.scheduled_slots = []
            activity.scheduled_bases = {}
        return activity

    def user_action(
        self,
        summary: AnalysisFreshnessSummary,
        activity: RefreshActivity,
        *,
        show_recommendations: bool,
        command_builder: CommandBuilder | None = None,
    ) -> str:
        if not (
            summary.blocking_now
            or summary.stale_in_queue
            or summary.due_soon
            or summary.candidate_blocked
        ):
            return "none"
        base_args: list[str] = []
        if show_recommendations:
            base_args.append("--recommend")
        base_args.extend(["--refresh-policy", "proactive"])

        def render_command() -> str:
            if command_builder is None:
                return "scripts/portfolio_manager.py " + " ".join(base_args)
            return command_builder(*base_args)

        command = render_command()
        if activity.failed:
            if activity.failed_retry_after:
                return (
                    f"refresh failed for {', '.join(activity.failed)} — "
                    "backoff active; rerun on a later refresh-enabled run"
                )
            return f"refresh failed for {', '.join(activity.failed)} — rerun {command}"
        if activity.skipped_read_only:
            return f"read-only mode blocked refresh — run {command}"
        if activity.skipped_due_to_failure_backoff:
            return self._cooldown_user_action(
                "failed refresh backoff",
                activity.skipped_due_to_failure_backoff,
            )
        if activity.skipped_due_to_unrepaired:
            return self._cooldown_user_action(
                "prior refresh did not repair",
                activity.skipped_due_to_unrepaired,
            )
        if activity.policy == "off":
            return f"run {command}"
        if activity.skipped_due_to_limit:
            return (
                "refresh limit reached — rerun with a higher --refresh-limit "
                f"(remaining: {', '.join(activity.skipped_due_to_limit)})"
            )
        return "none"

    @staticmethod
    def _cooldown_user_action(label: str, retry_times: dict[str, str]) -> str:
        tickers = ", ".join(retry_times)
        parsed = [parse_refresh_retry_at(value) for value in retry_times.values()]
        retry_at = min((value for value in parsed if value is not None), default=None)
        timing = (
            retry_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            if retry_at is not None
            else "a later refresh-enabled run"
        )
        return f"{label} active for {tickers} — retry after {timing}"

    async def execute(
        self,
        activity: RefreshActivity,
        *,
        execution: RefreshExecutionOptions,
        run_analysis_fn: AnalysisRunner,
        save_results_fn: AnalysisSaver,
        progress: ProgressCallback | None = None,
    ) -> RefreshActivity:
        updated = activity.copy()
        state = self._load_state(updated.scheduler_state_path)
        slots_by_ticker = dict(updated.scheduled_slots)
        refresh_count = len(updated.queued)
        for index, ticker in enumerate(list(updated.queued), start=1):
            if progress is not None:
                progress(f"Refreshing analysis {index}/{refresh_count}: {ticker}")
            try:
                result = await run_analysis_fn(
                    ticker=ticker,
                    quick_mode=execution.quick_mode,
                    skip_charts=execution.skip_charts,
                )
                if not result:
                    raise RuntimeError("analysis runner returned no result")
                save_results_fn(result, ticker, quick_mode=execution.quick_mode)
                from types import SimpleNamespace

                from src.persistence import _maybe_save_rejection_record

                await _maybe_save_rejection_record(
                    result,
                    SimpleNamespace(
                        ticker=ticker, quick=execution.quick_mode, strict=False
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "analysis_refresh_failed",
                    ticker=ticker,
                    **summarize_exception(exc, operation="refreshing analysis"),
                )
                updated.failed.append(ticker)
                retry_after = (
                    datetime.now(UTC)
                    + timedelta(hours=DEFAULT_REFRESH_FAILURE_BACKOFF_HOURS)
                ).isoformat()
                state.retry_not_before[ticker] = retry_after
                updated.failed_retry_after[ticker] = retry_after
            else:
                updated.refreshed.append(ticker)
                # Invocation recovery and persistent evidence are separate. A
                # successful run clears only the former; the post-refresh
                # reconciliation below decides whether the same evidence gap
                # survived and merits a condition-specific cooldown.
                state.retry_not_before.pop(ticker, None)
                updated.failed_retry_after.pop(ticker, None)
            if (slot_after := slots_by_ticker.get(ticker)) is not None:
                state.next_slot = slot_after
            self._save_state(updated.scheduler_state_path, state)
        return updated

    def record_unrepaired_refreshes(
        self,
        activity: RefreshActivity,
        summary: AnalysisFreshnessSummary,
    ) -> RefreshActivity:
        """Persist a backoff for each paid refresh that repaired nothing.

        This intentionally runs after refreshed artifacts have been loaded and
        reconciled: a successful invocation alone says nothing about whether it
        changed the disposition that justified its cost.

        The recorded key is the action *basis*, a closed ten-value enum, not the
        set of buy-blocking flags. Flag composition is downstream of search
        quality and varies run to run even when nothing about the position has
        changed, so keying on it meant the stored key never matched and the
        backoff never applied. A genuinely different basis is new information
        and stays immediately eligible.
        """
        updated = activity.copy()
        state = self._load_state(updated.scheduler_state_path)
        # operator_review and fresh are absent on purpose: a ticker that landed
        # there is no longer competing for a paid slot, so it needs no backoff.
        replanned_rows = {
            row.run_ticker: row
            for row in (
                *summary.blocking_now,
                *summary.stale_in_queue,
                *summary.due_soon,
                *summary.candidate_blocked,
                *summary.refreshed_this_run,
            )
        }
        changed = False
        for ticker in updated.refreshed:
            scheduled_basis = updated.scheduled_bases.get(ticker)
            current = replanned_rows.get(ticker)
            unrepaired = (
                scheduled_basis is not None
                and current is not None
                and current.refresh_repairable
                and current.action_basis == scheduled_basis
            )
            if unrepaired:
                retry_after = (
                    datetime.now(UTC)
                    + timedelta(hours=DEFAULT_REFRESH_DATA_QUALITY_BACKOFF_HOURS)
                ).isoformat()
                state.unrepaired_refresh[ticker] = _UnrepairedRefresh(
                    basis=str(scheduled_basis), retry_after=retry_after
                )
                updated.unrepaired_retry_after[ticker] = retry_after
                changed = True
            elif state.unrepaired_refresh.pop(ticker, None) is not None:
                # The refresh moved the position off its prior basis, so the
                # stored backoff no longer describes anything.
                changed = True
        if changed:
            self._save_state(updated.scheduler_state_path, state)
        return updated

    @staticmethod
    def _classify_held_review(
        item: ReconciliationItem,
        row: AnalysisFreshnessRow,
        summary: AnalysisFreshnessSummary,
    ) -> None:
        analysis = item.analysis
        assert analysis is not None
        basis = item.action_basis
        if basis == "DATA_QUALITY" and AnalysisRefreshService._requires_broker_repair(
            item
        ):
            # The attached analysis may itself be aging or incomplete, but paid
            # research cannot repair the broker condition driving this review.
            summary.operator_review.append(replace(row, bucket="operator_review"))
            return
        if basis in _OPERATOR_ONLY_BASES:
            # These are decisions for the operator, not same-day evidence
            # failures. Keep them visible while fresh, then put them into the
            # normal fair cycle seven days before their analysis expires.
            target = (
                summary.due_soon
                if row.days_until_due is not None and row.days_until_due <= 7
                else summary.operator_review
            )
            target.append(
                replace(
                    row,
                    bucket="due_soon"
                    if target is summary.due_soon
                    else "operator_review",
                )
            )
            return
        if basis == "THESIS_REASSESSMENT":
            if item.sell_type == "SOFT_REJECT":
                # Intact fundamentals plus price weakness is not new thesis
                # evidence. A stale soft reject was routed above; when its
                # normal review date is near, it enters the fair cycle rather
                # than consuming today's urgent budget.
                target = (
                    summary.due_soon
                    if row.days_until_due is not None and row.days_until_due <= 7
                    else summary.operator_review
                )
                target.append(
                    replace(
                        row,
                        bucket="due_soon"
                        if target is summary.due_soon
                        else "operator_review",
                    )
                )
            elif AnalysisRefreshService._confirmation_reachable(analysis):
                # A hard, unconfirmed reject needs a second full analysis
                # before a held position can gain exit authority.
                summary.blocking_now.append(replace(row, bucket="blocking_now"))
            else:
                # Re-running today cannot change the disposition, because
                # confirmation also requires the two rejecting analyses to be
                # at least DEFAULT_SELL_CONFIRMATION_MIN_SPACING_DAYS apart.
                # Without this the same ticker re-entered the urgent queue on
                # every single run for a week (7047.T: 08-15 x2, 08-18, 08-19).
                summary.operator_review.append(replace(row, bucket="operator_review"))
            return
        if basis == "DATA_QUALITY" and not row.refresh_repairable:
            # Broker defects remain operator work. A settled analytical reject
            # is also not same-day repairable, but its measurement is not
            # immutable: re-admit it to the ordinary fair cycle near expiry.
            settled_reject = bool(analysis.evidence.settled_reject_flag_types)
            target = (
                summary.due_soon
                if settled_reject
                and row.days_until_due is not None
                and row.days_until_due <= 7
                else summary.operator_review
            )
            target.append(
                replace(
                    row,
                    bucket=(
                        "due_soon" if target is summary.due_soon else "operator_review"
                    ),
                )
            )
            return
        if basis in {"DATA_QUALITY", "STOP_LOSS"}:
            # Analysis evidence gaps and review-level breaches merit a refresh.
            summary.blocking_now.append(replace(row, bucket="blocking_now"))
            return
        if row.reason_family == "stale":
            target = (
                summary.blocking_now
                if "structural macro event" in row.reason_text.lower()
                else summary.due_soon
            )
            target.append(
                replace(
                    row,
                    bucket="blocking_now"
                    if target is summary.blocking_now
                    else "due_soon",
                )
            )
            return
        if row.reason_family == "price drift":
            summary.due_soon.append(replace(row, bucket="due_soon"))
            return
        summary.operator_review.append(replace(row, bucket="operator_review"))

    @staticmethod
    def _confirmation_reachable(analysis: AnalysisRecord) -> bool:
        """Whether a refresh today could confirm an unconfirmed reject.

        ``reject_confirmed`` needs provably full-mode analyses on both sides,
        at least ``DEFAULT_SELL_CONFIRMATION_MIN_SPACING_DAYS`` apart. Two
        cases, and the distinction matters:

        * The artifact on disk is quick-mode or mode-unknown. It carries no
          sell authority at all, so a full re-run supplies something the record
          lacks and could confirm against an older full prior regardless of the
          current artifact's age. Always reachable.
        * The artifact is provably full-mode. A refresh replaces it with a
          today-dated analysis, and ``reject_confirmed`` then measures spacing
          against the most recent full prior — which is the artifact being
          replaced. So the spacing a re-run can achieve is exactly that
          artifact's age, and below the threshold the re-run is *guaranteed*
          not to change the disposition.

        Convergence is unaffected: an unconfirmed reject still becomes urgent
        on the first day confirmation is achievable, and confirms then. What
        this removes is the six intervening days on which the same ticker
        re-entered the urgent queue every run and could not possibly settle
        (7047.T was fully re-analysed on 08-15 twice, 08-18 and 08-19).
        """
        if analysis.is_quick_mode is not False:
            return True
        return analysis.age_days >= DEFAULT_SELL_CONFIRMATION_MIN_SPACING_DAYS

    @staticmethod
    def _dedupe_rows(
        rows: list[AnalysisFreshnessRow],
        allowed: frozenset[str] | None,
        *,
        excluded: set[str] | None = None,
    ) -> list[AnalysisFreshnessRow]:
        deduped: list[AnalysisFreshnessRow] = []
        seen = set(excluded or ())
        for row in rows:
            if allowed is not None and row.run_ticker not in allowed:
                continue
            if row.run_ticker not in seen:
                seen.add(row.run_ticker)
                deduped.append(row)
        return sorted(deduped, key=AnalysisRefreshService._candidate_sort_key)

    @staticmethod
    def _candidate_sort_key(row: AnalysisFreshnessRow) -> tuple[int, int, int, str]:
        urgency = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(row.urgency, 9)
        deadline = row.days_until_due if row.days_until_due is not None else 9_999
        return (urgency, deadline, -(row.age_days or 0), row.run_ticker)

    @staticmethod
    def _priority_plan(
        rows: list[AnalysisFreshnessRow], limit: int, next_slot: int
    ) -> tuple[list[tuple[AnalysisFreshnessRow, int]], list[AnalysisFreshnessRow]]:
        selected = [
            (row, next_slot + index + 1) for index, row in enumerate(rows[:limit])
        ]
        return (selected, rows[limit:])

    @staticmethod
    def _weighted_fair_plan(
        urgent: list[AnalysisFreshnessRow],
        cycle: list[AnalysisFreshnessRow],
        limit: int,
        next_slot: int,
    ) -> tuple[list[tuple[AnalysisFreshnessRow, int]], list[AnalysisFreshnessRow]]:
        # This is fixed-quantum weighted round robin, not a priority sort or
        # deficit scheduler: every full analysis costs one slot. Advance the
        # durable cursor for every attempt (including a fallback to the only
        # non-empty stream) so repeated small --refresh-limit runs cannot
        # starve normal-cycle work while urgent items keep arriving.
        queues: dict[RefreshStream, list[AnalysisFreshnessRow]] = {
            "urgent": list(urgent),
            "cycle": list(cycle),
        }
        selected: list[tuple[AnalysisFreshnessRow, int]] = []
        slot = next_slot
        while len(selected) < limit and (queues["urgent"] or queues["cycle"]):
            preferred = _SERVICE_PATTERN[slot % len(_SERVICE_PATTERN)]
            other: RefreshStream = "cycle" if preferred == "urgent" else "urgent"
            source = preferred if queues[preferred] else other
            selected.append((queues[source].pop(0), slot + 1))
            slot += 1
        return (selected, [*queues["urgent"], *queues["cycle"]])

    @staticmethod
    def _remove_backing_off(
        urgent: list[AnalysisFreshnessRow],
        cycle: list[AnalysisFreshnessRow],
        state: _SchedulerState,
        activity: RefreshActivity,
    ) -> tuple[list[AnalysisFreshnessRow], list[AnalysisFreshnessRow]]:
        now = datetime.now(UTC)

        def filter_rows(rows: list[AnalysisFreshnessRow]) -> list[AnalysisFreshnessRow]:
            eligible: list[AnalysisFreshnessRow] = []
            for row in rows:
                retry_at = parse_refresh_retry_at(
                    state.retry_not_before.get(row.run_ticker)
                )
                if retry_at is not None and retry_at > now:
                    activity.skipped_due_to_cooldown.append(row.run_ticker)
                    activity.skipped_due_to_failure_backoff[row.run_ticker] = (
                        retry_at.isoformat()
                    )
                    continue
                # A stored backoff suppresses only the basis it was recorded
                # against, so a position that has since moved to a different
                # basis is new information and competes for a slot again.
                unrepaired = state.unrepaired_refresh.get(row.run_ticker)
                backoff_at = (
                    parse_refresh_retry_at(unrepaired.retry_after)
                    if unrepaired is not None and unrepaired.basis == row.action_basis
                    else None
                )
                if backoff_at is not None and backoff_at > now:
                    activity.skipped_due_to_cooldown.append(row.run_ticker)
                    activity.skipped_due_to_unrepaired[row.run_ticker] = (
                        backoff_at.isoformat()
                    )
                    continue
                eligible.append(row)
            return eligible

        return (filter_rows(urgent), filter_rows(cycle))

    @staticmethod
    def _load_state(path: Path | None) -> _SchedulerState:
        if path is None or not path.exists():
            return _SchedulerState()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("scheduler state must be an object")
            version = raw.get("version")
            if isinstance(version, bool) or version not in {1, 2, 3}:
                raise ValueError(f"unsupported scheduler state version: {version!r}")
            next_slot = raw.get("next_slot", 0)
            retry_not_before = raw.get("retry_not_before", {})
            # v2 keyed this map on buy-blocking flag composition, which proved
            # unusable (the key effectively never repeated). Those entries
            # describe nothing a v3 reader can act on, so they are dropped
            # rather than migrated; at worst one ticker is refreshed once more.
            raw_unrepaired = raw.get("unrepaired_refresh", {}) if version == 3 else {}
            if not isinstance(next_slot, int) or next_slot < 0:
                raise ValueError("invalid next_slot")
            if not isinstance(retry_not_before, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in retry_not_before.items()
            ):
                raise ValueError("invalid retry_not_before")
            if not isinstance(raw_unrepaired, dict) or not all(
                isinstance(key, str)
                and isinstance(value, dict)
                and isinstance(value.get("basis"), str)
                and isinstance(value.get("retry_after"), str)
                for key, value in raw_unrepaired.items()
            ):
                raise ValueError("invalid unrepaired_refresh")
            now = datetime.now(UTC)
            # State is persisted only when execution advances the scheduler.
            # Pruning while loading keeps plan() read-only and ensures the next
            # normal write drops expired or malformed entries.
            retry_not_before = {
                ticker: retry_after
                for ticker, retry_after in retry_not_before.items()
                if (parsed := parse_refresh_retry_at(retry_after)) is not None
                and parsed > now
                and _is_research_ticker(ticker)
            }
            unrepaired_refresh = {
                ticker: _UnrepairedRefresh(
                    basis=entry["basis"], retry_after=entry["retry_after"]
                )
                for ticker, entry in raw_unrepaired.items()
                if (parsed := parse_refresh_retry_at(entry["retry_after"])) is not None
                and parsed > now
                and _is_research_ticker(ticker)
            }
            return _SchedulerState(
                next_slot=next_slot,
                retry_not_before=retry_not_before,
                unrepaired_refresh=unrepaired_refresh,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "refresh_scheduler_state_invalid",
                path=str(path),
                **summarize_exception(exc, operation="load_refresh_scheduler_state"),
            )
            # A fresh state is safe: execute() persists it atomically after
            # the next attempted refresh, rebuilding a clobbered cursor rather
            # than letting an invalid local control file block the queue.
            return _SchedulerState()

    @staticmethod
    def _save_state(path: Path | None, state: _SchedulerState) -> None:
        if path is None:
            return
        temp_name: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "version": 3,
                        "next_slot": state.next_slot,
                        "retry_not_before": state.retry_not_before,
                        "unrepaired_refresh": {
                            ticker: {
                                "basis": entry.basis,
                                "retry_after": entry.retry_after,
                            }
                            for ticker, entry in state.unrepaired_refresh.items()
                        },
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
            temp_name = None
        except OSError as exc:
            logger.warning(
                "refresh_scheduler_state_save_failed",
                path=str(path),
                **summarize_exception(exc, operation="save_refresh_scheduler_state"),
            )
        finally:
            if temp_name:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass

    @staticmethod
    def _analysis_expiry_details(
        analysis: AnalysisRecord | None, max_age_days: int
    ) -> tuple[str | None, int | None]:
        if not analysis or not analysis.analysis_date:
            return (None, None)
        try:
            expires_dt = datetime.strptime(
                analysis.analysis_date, "%Y-%m-%d"
            ) + timedelta(days=max_age_days)
            return (expires_dt.date().isoformat(), max_age_days - analysis.age_days)
        except (TypeError, ValueError):
            return (None, None)

    @staticmethod
    def _reason_family(item: ReconciliationItem, max_age_days: int) -> str:
        reason_lower = (item.reason or "").lower()
        if item.analysis is None or "no analysis found" in reason_lower:
            return "no analysis"
        if "target hit" in reason_lower:
            return "target hit"
        if "price drift" in reason_lower or "drift" in reason_lower:
            return "price drift"
        if "stale analysis" in reason_lower or item.analysis.age_days > max_age_days:
            return "stale"
        return "review required"

    @staticmethod
    def _requires_broker_repair(item: ReconciliationItem) -> bool:
        position = item.ibkr_position
        return position is not None and (
            position.quantity < 0 or not position.valuation_valid
        )

    @staticmethod
    def _refresh_repairable(item: ReconciliationItem) -> bool:
        """Whether a paid stock analysis could plausibly change this row.

        Only *indeterminate* buy-blocking evidence qualifies. A settled gate
        failure — ``LIQUIDITY_HARD_FAIL`` measured against a fixed thesis
        threshold — is already resolved, and re-running research cannot move
        it; treating it as repairable re-analysed 2173.T on every invocation
        for days while its liquidity verdict was never in doubt.

        Price is deliberately absent. A review-level breach raises urgency
        elsewhere, but it is not evidence that more research exists to buy, and
        letting it participate here made spend a function of a price hovering
        near a threshold.
        """
        if (
            item.action_basis != "DATA_QUALITY"
            or item.analysis is None
            or AnalysisRefreshService._requires_broker_repair(item)
        ):
            return False
        if item.analysis.evidence.indeterminate_flag_types:
            return True
        data_quality = item.analysis.data_quality
        return data_quality.get("data_vacuum") is True or (
            item.analysis.current_price is None
            and item.analysis.health_adj == 0
            and item.analysis.growth_adj == 0
        )


def refresh_scheduler_state_path(results_dir: Path) -> Path:
    """Return the ignored, run-local cursor/backoff file for a results directory."""
    return results_dir / ".refresh_scheduler_state.json"


def run_ticker_for(item: ReconciliationItem) -> str:
    """Return the canonical yfinance ticker for analysis refresh and rerun commands."""
    if item.ticker.has_suffix:
        return item.ticker.yf
    if item.analysis and "." in item.analysis.ticker:
        return item.analysis.ticker
    return item.ticker.yf
