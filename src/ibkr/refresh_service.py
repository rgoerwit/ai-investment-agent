from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Literal

from src.ibkr.models import AnalysisRecord, ReconciliationItem
from src.ibkr.types import (
    AnalysisRunner,
    AnalysisSaver,
    CommandBuilder,
    ProgressCallback,
)

RefreshPolicy = Literal["off", "blocking", "proactive"]


@dataclass(frozen=True)
class AnalysisFreshnessRow:
    display_ticker: str
    run_ticker: str
    bucket: str
    reason_family: str
    reason_text: str
    action: str
    age_days: int | None
    expires_date: str | None
    days_until_due: int | None


@dataclass
class AnalysisFreshnessSummary:
    blocking_now: list[AnalysisFreshnessRow] = field(default_factory=list)
    stale_in_queue: list[AnalysisFreshnessRow] = field(default_factory=list)
    due_soon: list[AnalysisFreshnessRow] = field(default_factory=list)
    candidate_blocked: list[AnalysisFreshnessRow] = field(default_factory=list)
    fresh: list[AnalysisFreshnessRow] = field(default_factory=list)


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


@dataclass(frozen=True)
class RefreshPlanOptions:
    policy: RefreshPolicy
    limit: int
    show_recommendations: bool
    read_only: bool
    max_age_days: int
    ticker_subset: frozenset[str] | None = None


@dataclass(frozen=True)
class RefreshExecutionOptions:
    quick_mode: bool
    skip_charts: bool = True


class AnalysisRefreshService:
    """Owns portfolio-manager analysis freshness and refresh behavior."""

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
            return "blocking"
        return "off"

    def classify(
        self,
        items: list[ReconciliationItem],
        *,
        max_age_days: int,
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
                age_days=analysis.age_days if analysis else None,
                expires_date=expires_date,
                days_until_due=days_until_due,
            )

            if (
                item.ibkr_position is not None
                and item.action == "REVIEW"
                and item.sell_type != "SOFT_REJECT"
            ):
                summary.blocking_now.append(replace(row, bucket="blocking_now"))
                continue

            if (
                item.ibkr_position is not None
                and analysis is not None
                and analysis.age_days > max_age_days
                and (item.action in {"SELL", "TRIM"} or item.sell_type == "SOFT_REJECT")
            ):
                summary.stale_in_queue.append(replace(row, bucket="stale_in_queue"))
                continue

            if (
                item.ibkr_position is not None
                and item.action == "HOLD"
                and analysis is not None
                and days_until_due is not None
                and 0 < days_until_due <= 7
            ):
                summary.due_soon.append(replace(row, bucket="due_soon"))
                continue

            if item.ibkr_position is None and item.action == "REVIEW":
                summary.candidate_blocked.append(
                    replace(row, bucket="candidate_blocked")
                )
                continue

            summary.fresh.append(row)

        return summary

    def plan(
        self,
        summary: AnalysisFreshnessSummary,
        *,
        options: RefreshPlanOptions,
    ) -> RefreshActivity:
        activity = RefreshActivity(policy=options.policy, limit=options.limit)

        candidates: list[str] = []
        candidates.extend(row.run_ticker for row in summary.blocking_now)
        candidates.extend(
            row.run_ticker
            for row in summary.stale_in_queue
            if row.action in {"SELL", "TRIM"}
        )
        if options.show_recommendations:
            candidates.extend(row.run_ticker for row in summary.candidate_blocked)
        if options.policy == "proactive":
            candidates.extend(row.run_ticker for row in summary.due_soon)

        deduped: list[str] = []
        seen: set[str] = set()
        allowed = options.ticker_subset
        for ticker in candidates:
            if allowed is not None and ticker not in allowed:
                continue
            if ticker in seen:
                continue
            seen.add(ticker)
            deduped.append(ticker)

        if options.policy == "off":
            activity.skipped_due_to_policy = deduped
            return activity

        activity.queued = deduped[: options.limit]
        activity.skipped_due_to_limit = deduped[options.limit :]

        if options.read_only:
            activity.skipped_read_only = list(activity.queued)
            activity.queued = []

        return activity

    def user_action(
        self,
        summary: AnalysisFreshnessSummary,
        activity: RefreshActivity,
        *,
        show_recommendations: bool,
        command_builder: CommandBuilder | None = None,
    ) -> str:
        if not summary.blocking_now and not summary.candidate_blocked:
            return "none"

        base_args: list[str] = []
        if show_recommendations:
            base_args.append("--recommend")
        base_args.extend(["--refresh-policy", "blocking"])

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
        updated = RefreshActivity(
            policy=activity.policy,
            limit=activity.limit,
            queued=list(activity.queued),
            refreshed=list(activity.refreshed),
            failed=list(activity.failed),
            skipped_due_to_policy=list(activity.skipped_due_to_policy),
            skipped_due_to_limit=list(activity.skipped_due_to_limit),
            skipped_read_only=list(activity.skipped_read_only),
        )

        refresh_count = len(updated.queued)
        for index, ticker in enumerate(list(updated.queued), start=1):
            if progress is not None:
                progress(f"Refreshing analysis {index}/{refresh_count}: {ticker}")
            result = await run_analysis_fn(
                ticker=ticker,
                quick_mode=execution.quick_mode,
                skip_charts=execution.skip_charts,
            )
            if result:
                save_results_fn(result, ticker, quick_mode=execution.quick_mode)
                updated.refreshed.append(ticker)
            else:
                updated.failed.append(ticker)

        return updated

    @staticmethod
    def _analysis_expiry_details(
        analysis: AnalysisRecord | None,
        max_age_days: int,
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


def run_ticker_for(item: ReconciliationItem) -> str:
    """Return the canonical yfinance ticker for analysis refresh and rerun commands."""
    if item.ticker.has_suffix:
        return item.ticker.yf
    if item.analysis and "." in item.analysis.ticker:
        return item.analysis.ticker
    return item.ticker.yf
