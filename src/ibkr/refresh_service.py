from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import structlog

from src.error_safety import summarize_exception
from src.ibkr.models import AnalysisRecord, ReconciliationItem
from src.ibkr.portfolio_defaults import (
    DEFAULT_REFRESH_CYCLE_WEIGHT,
    DEFAULT_REFRESH_FAILURE_BACKOFF_HOURS,
    DEFAULT_REFRESH_URGENT_WEIGHT,
    DEFAULT_SELL_CONFIRMATION_MIN_SPACING_DAYS,
)
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
    skipped_due_to_policy: list[str] = field(default_factory=list)
    skipped_due_to_limit: list[str] = field(default_factory=list)
    skipped_read_only: list[str] = field(default_factory=list)
    skipped_due_to_cooldown: list[str] = field(default_factory=list)
    # (ticker, WRR cursor position after dispatch). Attempts, successful or
    # failed, consume a quantum so one bad urgent ticker cannot monopolize runs.
    scheduled_slots: list[tuple[str, int]] = field(default_factory=list)
    scheduler_state_path: Path | None = None


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


@dataclass
class _SchedulerState:
    """Small durable state needed for fair service across separate CLI runs."""

    next_slot: int = 0
    retry_not_before: dict[str, str] = field(default_factory=dict)


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
                if item.sell_type == "SOFT_REJECT":
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
            # Strict priority remains work-conserving: a blocking run serves
            # urgent work first, but must not idle a whole session when only
            # normal-cycle work is available.
            selected, deferred = self._priority_plan(
                urgent if urgent else cycle,
                options.limit,
                state.next_slot,
            )
        else:
            selected, deferred = self._weighted_fair_plan(
                urgent, cycle, options.limit, state.next_slot
            )
        activity.queued = [row.run_ticker for row, _ in selected]
        activity.scheduled_slots = [(row.run_ticker, slot) for row, slot in selected]
        activity.skipped_due_to_limit = [row.run_ticker for row in deferred]
        if options.read_only:
            activity.skipped_read_only = list(activity.queued)
            activity.queued = []
            activity.scheduled_slots = []
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
            return f"refresh failed for {', '.join(activity.failed)} — rerun {command}"
        if activity.skipped_read_only:
            return f"read-only mode blocked refresh — run {command}"
        if activity.policy == "off":
            return f"run {command}"
        if activity.skipped_due_to_limit:
            return (
                "refresh limit reached — rerun with a higher --refresh-limit "
                f"(remaining: {', '.join(activity.skipped_due_to_limit)})"
            )
        return "none"

    async def execute(
        self,
        activity: RefreshActivity,
        *,
        execution: RefreshExecutionOptions,
        run_analysis_fn: AnalysisRunner,
        save_results_fn: AnalysisSaver,
        progress: ProgressCallback | None = None,
    ) -> RefreshActivity:
        updated = replace(
            activity,
            queued=list(activity.queued),
            refreshed=list(activity.refreshed),
            failed=list(activity.failed),
            skipped_due_to_policy=list(activity.skipped_due_to_policy),
            skipped_due_to_limit=list(activity.skipped_due_to_limit),
            skipped_read_only=list(activity.skipped_read_only),
            skipped_due_to_cooldown=list(activity.skipped_due_to_cooldown),
            scheduled_slots=list(activity.scheduled_slots),
        )
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
                state.retry_not_before[ticker] = (
                    datetime.now(UTC)
                    + timedelta(hours=DEFAULT_REFRESH_FAILURE_BACKOFF_HOURS)
                ).isoformat()
            else:
                updated.refreshed.append(ticker)
                state.retry_not_before.pop(ticker, None)
            if (slot_after := slots_by_ticker.get(ticker)) is not None:
                state.next_slot = slot_after
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
        if basis in {"DATA_QUALITY", "STOP_LOSS"}:
            # Evidence or a review-level breach is indeterminate evidence:
            # re-run before the operator acts, even if the saved artifact is
            # from today. The failed-run backoff below prevents an unavailable
            # provider from monopolizing the urgent stream.
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
                retry_at = AnalysisRefreshService._parse_retry_at(
                    state.retry_not_before.get(row.run_ticker)
                )
                if retry_at is not None and retry_at > now:
                    activity.skipped_due_to_cooldown.append(row.run_ticker)
                else:
                    if retry_at is not None:
                        state.retry_not_before.pop(row.run_ticker, None)
                    eligible.append(row)
            return eligible

        return (filter_rows(urgent), filter_rows(cycle))

    @staticmethod
    def _load_state(path: Path | None) -> _SchedulerState:
        if path is None or not path.exists():
            return _SchedulerState()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            next_slot = raw.get("next_slot", 0)
            retry_not_before = raw.get("retry_not_before", {})
            if not isinstance(next_slot, int) or next_slot < 0:
                raise ValueError("invalid next_slot")
            if not isinstance(retry_not_before, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in retry_not_before.items()
            ):
                raise ValueError("invalid retry_not_before")
            return _SchedulerState(
                next_slot=next_slot, retry_not_before=retry_not_before
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
                        "version": 1,
                        "next_slot": state.next_slot,
                        "retry_not_before": state.retry_not_before,
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
    def _parse_retry_at(raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None

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
