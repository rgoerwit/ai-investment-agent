from __future__ import annotations

import asyncio
import time
from pathlib import Path

from src.web.ibkr_dashboard.job_store import QueuedRefreshJob, RefreshJobStore
from src.web.ibkr_dashboard.settings import DashboardSettings


def _run_analysis_sync(ticker: str, quick_mode: bool):
    from src.main import run_analysis

    return asyncio.run(
        run_analysis(
            ticker=ticker,
            quick_mode=quick_mode,
            skip_charts=True,
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


def run_once(store: RefreshJobStore, settings: DashboardSettings) -> bool:
    job = store.claim_next()
    if job is None:
        return False
    _run_job(store, job, settings)
    return True


def _run_job(
    store: RefreshJobStore,
    job: QueuedRefreshJob,
    settings: DashboardSettings,
) -> None:
    succeeded = 0
    failed = 0
    if not job.request.tickers:
        store.complete_job(job.job_id, status="completed")
        return

    for ticker in job.request.tickers:
        store.update_ticker_status(job.job_id, ticker, "running")
        try:
            result = _run_analysis_sync(ticker, job.request.quick_mode)
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
            store.update_ticker_status(job.job_id, ticker, "failed", error=str(exc))

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
    while True:
        ran = run_once(store, settings)
        if not ran:
            time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    main()
