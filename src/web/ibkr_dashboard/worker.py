from __future__ import annotations

import asyncio
import time
from pathlib import Path

from src.config import config
from src.error_safety import format_error_message, summarize_exception
from src.runtime_services import (
    RuntimeServices,
    build_provider_runtime,
    build_runtime_services_from_config,
)
from src.web.ibkr_dashboard.job_store import QueuedRefreshJob, RefreshJobStore
from src.web.ibkr_dashboard.settings import DashboardSettings


def _run_analysis_sync(
    ticker: str,
    quick_mode: bool,
    *,
    runtime_services: RuntimeServices,
):
    from src.main import run_analysis

    return asyncio.run(
        run_analysis(
            ticker=ticker,
            quick_mode=quick_mode,
            skip_charts=True,
            runtime_services=runtime_services,
        )
    )


def _save_result_sync(
    result: dict,
    ticker: str,
    quick_mode: bool,
    *,
    results_dir: str,
) -> Path:
    from src.main import save_results_to_file

    return save_results_to_file(
        result,
        ticker,
        quick_mode=quick_mode,
        results_dir=results_dir,
    )


def build_worker_runtime_services() -> RuntimeServices:
    return build_runtime_services_from_config(
        config,
        enable_tool_audit=False,
        provider_runtime=build_provider_runtime(explicit=True),
    )


def run_once(
    store: RefreshJobStore,
    settings: DashboardSettings,
    *,
    runtime_services: RuntimeServices | None = None,
) -> bool:
    job = store.claim_next()
    if job is None:
        return False
    _run_job(
        store,
        job,
        settings,
        runtime_services=runtime_services or build_worker_runtime_services(),
    )
    return True


def _run_job(
    store: RefreshJobStore,
    job: QueuedRefreshJob,
    settings: DashboardSettings,
    *,
    runtime_services: RuntimeServices,
) -> None:
    succeeded = 0
    failed = 0
    if not job.request.tickers:
        store.complete_job(job.job_id, status="completed")
        return

    for ticker in job.request.tickers:
        store.update_ticker_status(job.job_id, ticker, "running")
        try:
            result = _run_analysis_sync(
                ticker,
                job.request.quick_mode,
                runtime_services=runtime_services,
            )
            if result is None:
                raise RuntimeError("run_analysis returned no result")
            output_path = _save_result_sync(
                result,
                ticker,
                job.request.quick_mode,
                results_dir=job.request.results_dir,
            )
            store.update_ticker_status(
                job.job_id,
                ticker,
                "completed",
                output_path=str(output_path),
            )
            succeeded += 1
        except Exception as exc:
            failed += 1
            summary = summarize_exception(
                exc,
                operation="dashboard refresh job",
            )
            store.update_ticker_status(
                job.job_id,
                ticker,
                "failed",
                error=format_error_message(
                    operation="dashboard refresh job",
                    error_type=summary["error_type"],
                    message_preview=summary["message_preview"],
                ),
            )

    if failed == 0:
        status = "completed"
    elif succeeded == 0:
        status = "failed"
    else:
        status = "partial"
    store.complete_job(job.job_id, status=status)


def main(poll_interval_seconds: float = 2.0) -> None:
    settings = DashboardSettings()
    store = RefreshJobStore(settings.runtime_dir / "jobs.sqlite")
    runtime_services = build_worker_runtime_services()
    while True:
        ran = run_once(store, settings, runtime_services=runtime_services)
        if not ran:
            time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    main()
